"""
deepinv degradation models.
"""
from __future__ import annotations
import re
from typing import Optional

import torch
import torch.nn as nn
import torchvision.io as tv_io
from utils.motionblur import Kernel

import deepinv as dinv

def get_forward_model(cfg, device, generator: torch.Generator | None = None) -> tuple[dinv.physics.DecomposablePhysics, callable]:
    noise_model = dinv.physics.GaussianNoise(sigma=cfg.exp.sigma_x0, rng=generator)
    #NOTE: set to the LDM's output shape
    img_shape = (cfg.model.output_c, cfg.model.output_h, cfg.model.output_w)
    deg = cfg.exp.deg
    deg = deg.split("-")[0]
    use_1024 = img_shape[1] >= 1024
    #TODO: add composition of degradations, might be useful for 1024 models
    if deg[:3] == 'inp':
        C, H, W = img_shape
        mask = torch.ones((1, H, W), device=device)
        size = cfg.exp.mask_size
        # Define mask region
        mask[:, H//2 - size//5 - 35:H//2 + size//5 - 35,
             W//2 - 4*size//5 - 2:W//2 + 4*size//5 + 2] = 0
        forward_model = dinv.physics.Inpainting(
            tensor_size=img_shape,
            mask=mask,
            noise_model=noise_model,
        ).to(device)
        transpose_operator = forward_model.A_adjoint
    elif deg == 'deblur_gauss':
        ksize = cfg.exp.blur_sigma
        filter = dinv.physics.blur.gaussian_blur(sigma=(ksize, ksize))
        forward_model = dinv.physics.BlurFFT(
            # deepinv expects (C, H, W)
            img_size=img_shape,
            filter=filter,
            device=device,
            noise_model=noise_model,
        )
        transpose_operator = forward_model.A_adjoint

    elif deg == 'deblur_motion':
        #raise NotImplementedError("Motion blur degradation is not implemented yet.")
        ker_size = 61 if not use_1024 else 122
        kernel = Kernel(size=(ker_size, ker_size), intensity=0.5)
        kernel_torch = torch.tensor(kernel.kernelMatrix, dtype=torch.float32)
        kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(0).to(device)
        forward_model = dinv.physics.BlurFFT(
            img_size=img_shape,
            filter=kernel_torch,
            device=device,
            noise_model=noise_model,
        )
        transpose_operator = forward_model.A_adjoint
        
    elif "bicubic" in deg:
        # bicubic superres 
        factor = int(deg[7:])
        forward_model = dinv.physics.Downsampling(
            # deepinv expects high-res image size as (C, H, W)
            img_size=img_shape,
            factor=factor,
            device=device,
            noise_model=noise_model,
            filter='bicubic',
            padding='reflect',
        )
        transpose_operator = forward_model.A_adjoint
        
    elif "jpeg" in deg:
        qf = _parse_qf_from_deg(deg)
        forward_model = GaussianNoiseJPEGOp(
            sigma_pre=float(cfg.exp.get("sigma_pre", 0.0)),
            sigma_x0=float(cfg.exp.get("sigma_x0", 0.0)),
            jpeg_quality=qf,
            device=torch.device("cpu"),  # keep JPEG on CPU
            dtype=torch.float32,
            n_prox_iter=int(cfg.exp.get("dinv_noisyjpeg_prox_n_iter", 10)),
            prox_step=cfg.exp.get("dinv_noisyjpeg_prox_step", None),
            generator=generator,
        ).to(device)
        transpose_operator = forward_model.A_adjoint

    else:
        raise ValueError(f"Unexpected problem.type {cfg.exp.deg}")

    return forward_model, transpose_operator


def build_degradation_model(cfg, generator: torch.Generator | None = None):
    if not cfg.exp.use_dinv:
        raise ValueError(
            "Only deepinv forward operators are supported; set exp.use_dinv=true."
        )
    return get_forward_model(cfg, cfg.exp.device, generator=generator)[0]


def get_degradation_image(y_0, H, cfg):
    c = cfg.dataset.channels
    h = cfg.model.output_h
    w = cfg.model.output_w
    print(f"shape of measurement: {y_0.shape}")
    if y_0.dim() == 3:
        y_0 = y_0.unsqueeze(0)
    y_0 = y_0.float()
    deg = str(cfg.exp.get("deg", "")).split("-")[0]
    if deg == "deblur_gauss" and hasattr(H, "prox_l2") and hasattr(H, "A_adjoint"):
        gamma = float(cfg.exp.get("dinv_blur_recon_gamma", 10.0))
        z0 = torch.zeros_like(H.A_adjoint(y_0))
        pinv_y = H.prox_l2(z=z0, y=y_0, gamma=gamma)
        pinv_y = pinv_y.clamp(-1, 1)
    else:
        pinv_y = H.A_dagger(y_0).reshape(-1, c, h, w)
    print(f"Range of pseudoinverse image: min {pinv_y.min().item()}, max {pinv_y.max().item()}")
    return pinv_y


def _parse_qf_from_deg(deg: str) -> int:
    m = re.search(r"(\d+)$", deg)
    if m is None:
        raise ValueError(
            f"Could not parse JPEG quality factor from degradation name '{deg}'. "
            f"Use e.g. 'noisyjpeg50' or 'jpeg30'."
        )
    qf = int(m.group(1))
    if not (1 <= qf <= 100):
        raise ValueError(f"JPEG quality must be in [1,100], got {qf}.")
    return qf


class GaussianNoiseJPEGOp(nn.Module):
    """
    Image-only forward operator:
        x -> x + N(0, sigma_pre^2) -> JPEG(qf) -> + N(0, sigma_x0^2)

    Expected input shape:
        x: (B, C, H, W) in [-1, 1]

    For the inverse side / prox:
      - we use only the deterministic JPEG part A_det
      - Gaussian noise is treated as part of the observed measurement y

    JPEG encode/decode is done on CPU.
    """

    def __init__(
        self,
        sigma_x0: float = 0.0,
        sigma_pre: float = 0.0,
        jpeg_quality: int = 50,
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = torch.float32,
        n_prox_iter: int = 10,
        prox_step: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        assert sigma_x0 >= 0.0
        assert sigma_pre >= 0.0
        assert 1 <= jpeg_quality <= 100

        self.sigma_x0 = float(sigma_x0)
        self.sigma_pre = float(sigma_pre)
        self.jpeg_quality = int(jpeg_quality)

        self.device_ = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype_ = dtype

        self.n_prox_iter = int(n_prox_iter)
        self.prox_step = prox_step
        self.rng = generator

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------
    @staticmethod
    def _to_uint8(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1.0, 1.0)
        x01 = (x + 1.0) * 0.5
        x255 = torch.round(x01 * 255.0)
        return x255.to(torch.uint8)

    @staticmethod
    def _from_uint8(xu: torch.Tensor, device, dtype) -> torch.Tensor:
        x = xu.to(device=device, dtype=torch.float32) / 255.0
        x = x * 2.0 - 1.0
        return x.to(dtype)

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        if self.rng is None:
            return torch.randn_like(x)
        eps = torch.randn(
            x.shape,
            generator=self.rng,
            device="cpu",
            dtype=torch.float32,
        )
        return eps.to(device=x.device, dtype=x.dtype)

    def _jpeg_compress_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        JPEG-compress a single image (C,H,W) in [-1,1], return (C,H,W) in [-1,1].
        Encode/decode happen on CPU.
        """
        dev = img.device
        dty = img.dtype

        img_u8_cpu = self._to_uint8(img).cpu()
        jpeg_bytes = tv_io.encode_jpeg(img_u8_cpu, quality=self.jpeg_quality)
        dec_u8 = tv_io.decode_jpeg(jpeg_bytes)  # uint8 CHW on CPU

        return self._from_uint8(dec_u8, device=dev, dtype=dty)

    def _jpeg_compress_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) -> (B,C,H,W)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        imgs = [self._jpeg_compress_image(x[i]) for i in range(x.shape[0])]
        return torch.stack(imgs, dim=0)

    @staticmethod
    def _ensure_batched(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        assert x.dim() == 4, f"Expected (B,C,H,W), got {tuple(x.shape)}"
        return x

    @staticmethod
    def _ste_identity(x_ref: torch.Tensor, y_exact: torch.Tensor) -> torch.Tensor:
        if not x_ref.requires_grad:
            return y_exact
        # Keep the exact JPEG value in the forward pass while using an
        # identity Jacobian as a surrogate during backpropagation.
        return x_ref + (y_exact - x_ref).detach()

    def _A_exact(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_batched(x)

        orig_dev = x.device
        orig_dtype = x.dtype

        dev = self.device_
        dty = self.dtype_ or torch.float32

        x = x.to(dev, dty)

        if self.sigma_pre > 0:
            x = x + self.sigma_pre * self._randn_like(x)
            x.clamp_(-1.0, 1.0)

        y = self._jpeg_compress_batch(x)

        if self.sigma_x0 > 0:
            y = y + self.sigma_x0 * self._randn_like(y)
            y.clamp_(-1.0, 1.0)

        return y.to(orig_dev, orig_dtype)

    def _A_det_exact(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_batched(x)

        orig_dev = x.device
        orig_dtype = x.dtype

        dev = self.device_
        dty = self.dtype_ or torch.float32

        x = x.to(dev, dty)
        y = self._jpeg_compress_batch(x)

        return y.to(orig_dev, orig_dtype)

    # --------------------------------------------------
    # forward operators
    # --------------------------------------------------
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stochastic data simulation operator:
            x -> x + noise_x0 -> JPEG -> + noise_y
        """
        x_batched = self._ensure_batched(x)
        with torch.no_grad():
            y_exact = self._A_exact(x_batched)
        return self._ste_identity(x_batched, y_exact)

    def A_det(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic JPEG-only operator used inside prox_l2.
        """
        x_batched = self._ensure_batched(x)
        with torch.no_grad():
            y_exact = self._A_det_exact(x_batched)
        return self._ste_identity(x_batched, y_exact)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.A(x)

    # --------------------------------------------------
    # inverse-side compatibility API
    # --------------------------------------------------
    @torch.no_grad()
    def A_adjoint(self, u: torch.Tensor) -> torch.Tensor:
        """
        Approximate adjoint of A_det.
        JPEG is nonlinear/noninvertible, so use identity.
        """
        return u

    @torch.no_grad()
    def A_dagger(self, y: torch.Tensor) -> torch.Tensor:
        """
        Approximate pseudo-inverse used for initialization / visualization.
        """
        return y.clamp(-1.0, 1.0)

    @torch.no_grad()
    def prox_l2(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        gamma: float,
        max_iter: Optional[int] = None,
        step: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Approximate proximal operator with the signature expected by your code:

            prox_{gamma/2 ||A_det(x)-y||^2}(z)

        using Landweber iterations on

            f(x) = 0.5 ||x-z||^2 + gamma/2 ||A_det(x)-y||^2

        with approximate gradient

            grad f(x) ~= (x-z) + gamma * (A_det(x) - y)

        where A_det^* is approximated by identity.
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)
        if y.dim() == 3:
            y = y.unsqueeze(0)

        assert z.dim() == 4 and y.dim() == 4

        if max_iter is None:
            max_iter = self.n_prox_iter

        orig_dev = z.device
        orig_dtype = z.dtype

        dev = self.device_
        dty = self.dtype_ or torch.float32

        x = z.to(dev, dty).clone()
        z_ = z.to(dev, dty)
        y_ = y.to(dev, dty)

        gamma_t = torch.as_tensor(gamma, device=dev, dtype=dty)

        if step is None:
            step = self.prox_step if self.prox_step is not None else 1.0 / (1.0 + float(gamma_t))

        for _ in range(max_iter):
            Ax = self.A_det(x.to(orig_dev, orig_dtype)).to(dev, dty)
            r = Ax - y_
            grad = (x - z_) + gamma_t * r
            x = x - step * grad
            x.clamp_(-1.0, 1.0)

        return x.to(orig_dev, orig_dtype)