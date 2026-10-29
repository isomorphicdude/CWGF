"""Ported from https://github.com/TReg-inverse/Treg"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import kornia
from omegaconf import DictConfig

from models.sd_inverse import SDInversePipeline
from utils.hf_utils import get_noise_pred
from .ldps import LDPS

class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


class TREG(LDPS):
    def __init__(self, model: SDInversePipeline, cfg: DictConfig):
        super().__init__(model, cfg)
        self.cfg_scale = float(cfg.algo.get("cfg_scale", 4.0))
        self.cg_lamb = float(cfg.algo.get("cg_lamb", 1e-4))
        self.cg_iters = int(cfg.algo.get("cg_iters", 5))
        self.cg_eps = float(cfg.algo.get("cg_eps", 1e-4))
        self.cg_damping = float(cfg.algo.get("cg_damping", 0.0))

        self.use_an = bool(cfg.algo.get("use_an", False))
        self.an_lr = float(cfg.algo.get("an_lr", 8e-4))
        self.an_steps = int(cfg.algo.get("an_steps", 10))

        self.scheduler = self.model.scheduler
        total_timesteps = int(
            getattr(self.scheduler.config, "num_train_timesteps", len(self.scheduler.alphas_cumprod))
        )
        num_sampling = max(1, len(self.scheduler.timesteps))
        self.skip = max(1, total_timesteps // num_sampling)
        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod

        self.clip_img_enc = FrozenClipImageEmbedder(model='ViT-L/14',
                                                    device=self.model.device)

    @torch.no_grad()
    def _conj_gradient(
        self,
        A,
        b: torch.Tensor,
        x0: torch.Tensor,
        m: int,
        eps: float,
        damping: float = 0.0,
    ) -> torch.Tensor:
        x = x0
        r = b - A(x)
        p = r.clone()

        rsold = (r * r).sum(dim=1, keepdim=True)
        for _ in range(m):
            Ap = A(p) + damping * p
            denom = (p * Ap).sum(dim=1, keepdim=True).clamp_min(1e-12)
            alpha = rsold / denom

            x = x + alpha * p
            r = r - alpha * Ap

            rsnew = (r * r).sum(dim=1, keepdim=True)
            if torch.sqrt(rsnew.max()) < eps:
                break

            p = r + (rsnew / rsold.clamp_min(1e-12)) * p
            rsold = rsnew

        return x

    @torch.no_grad()
    def data_consistency(
        self, z0t: torch.Tensor, measurement: torch.Tensor, cg_lamb: float | None = None
    ):
        """
        Applies data consistency update via conjugate gradient.
        """
        cg_lamb = self.cg_lamb if cg_lamb is None else float(cg_lamb)
        x0t = self.model.decode_latents(z0t, stay_on_device=True).detach()

        x_shape = x0t.shape
        y = measurement.to(device=x0t.device, dtype=x0t.dtype)
        y_shape = y.shape

        def A_flat(x_flat: torch.Tensor) -> torch.Tensor:
            x_img = x_flat.reshape(x_shape)
            return self._forward_measure(x_img).reshape(x_flat.shape[0], -1)

        def At_flat(y_flat: torch.Tensor) -> torch.Tensor:
            y_meas = y_flat.reshape(y_shape)
            return self.H.A_adjoint(y_meas).reshape(y_flat.shape[0], -1)

        x0_flat = x0t.reshape(x_shape[0], -1).to(torch.float32)
        y_flat = y.reshape(y_shape[0], -1).to(torch.float32)
        b = At_flat(y_flat) + cg_lamb * x0_flat

        x0y_flat = self._conj_gradient(
            lambda x: At_flat(A_flat(x)) + cg_lamb * x,
            b,
            x0_flat,
            m=self.cg_iters,
            eps=self.cg_eps,
            damping=self.cg_damping,
        )

        x0y = x0y_flat.reshape(x_shape).to(device=x0t.device)
        z0y = self.model.encode_images(x0y.to(dtype=self.model_dtype)).to(
            device=z0t.device, dtype=z0t.dtype
        )
        return z0y, x0y

    def adaptive_negation(
        self,
        x0: torch.Tensor,
        neg_prompt_embeds: torch.Tensor,
        lr: float | None = None,
        num_iter: int | None = None,
    ) -> torch.Tensor:
        """
        Updates null-text embedding to minimize the similarity in CLIP space.
        args:
            x0 (torch.Tensor): input image
            neg_prompt_embeds (torch.Tensor): null-text embedding
            lr (float): learning rate
            num_iter (int): number of iterations
        """
        # TODO: check compatibility with SDXL and variants
        # they use a different CLIP model
        lr = self.an_lr if lr is None else float(lr)
        num_iter = self.an_steps if num_iter is None else int(num_iter)
        if num_iter <= 0:
            return neg_prompt_embeds.detach()

        with torch.no_grad():
            img_feats = self.clip_img_enc(x0.detach())
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            img_feats = img_feats.unsqueeze(1).to(torch.float32)

        neg_param = torch.nn.Parameter(
            neg_prompt_embeds.detach().to(device=self.device, dtype=torch.float32)
        )

        optimizer = torch.optim.Adam([neg_param], lr=lr)
        for _ in range(num_iter):
            optimizer.zero_grad()
            sim = torch.matmul(img_feats, neg_param.transpose(1, 2))
            loss = sim.mean()
            loss.backward()
            optimizer.step()

        return neg_param.detach().to(device=self.device, dtype=self.model_dtype)

    def sample(
        self, x, y, ts, generator, y_0, results256: bool = False, **unusedkwargs
    ):
        batch_size = 1 if x.dim() == 3 else x.size(0)
        input_img = x
        y_0 = y_0.to(device=self.device)

        use_an = bool(unusedkwargs.get("use_an", self.use_an))
        cg_lamb = float(unusedkwargs.get("cg_lamb", self.cg_lamb))
        null_lr = float(unusedkwargs.get("null_lr", self.an_lr))
        an_steps = int(unusedkwargs.get("an_steps", self.an_steps))
        assert not unusedkwargs.get("use_DPS", False), "not compatible with DPS for now"

        prompt_embeds = self.prompt_embeds.to(
            device=self.device, dtype=self.model_dtype
        )
        neg_prompt_embeds = self.neg_prompt_embeds.to(
            device=self.device, dtype=self.model_dtype
        )

        latents = self.init_latents(
            x,
            generator,
            y_0,
            init_method=self.cfg.algo.get("init_method", None),
            prompt_embeds=prompt_embeds,
            neg_embeds=neg_prompt_embeds,
        )

        counter = 0
        x_list = []
        timesteps = self.scheduler.timesteps
        num_steps = len(timesteps)

        for i in range(num_steps):
            t = int(timesteps[i].item()) if torch.is_tensor(timesteps[i]) else int(timesteps[i])
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            at = torch.as_tensor(at, device=latents.device, dtype=latents.dtype)
            at_prev = torch.as_tensor(at_prev, device=latents.device, dtype=latents.dtype)

            with torch.no_grad():
                noise_pred = get_noise_pred(
                    self.model,
                    latents,
                    i,
                    batch_size,
                    self.device,
                    prompt_embeds,
                    neg_prompt_embeds,
                    guidance_scale=self.cfg_scale,
                    do_classifier_free_guidance=True,
                )
                z0t = (latents - torch.sqrt(1 - at) * noise_pred) / torch.sqrt(at)
                x0t = self.model.decode_latents(z0t, stay_on_device=True).detach()

            if i % 3 == 0 and i < 170:
                z0y, x0y = self.data_consistency(z0t, y_0, cg_lamb=cg_lamb)

                if use_an:
                    neg_prompt_embeds = self.adaptive_negation(
                        x0y, neg_prompt_embeds, lr=null_lr, num_iter=an_steps
                    )

                noise = torch.randn(
                    z0y.shape, device=x.device, dtype=z0y.dtype, generator=generator
                )
                z0_ema = at_prev * z0y + (1 - at_prev) * z0t
                latents = torch.sqrt(at_prev) * z0_ema + (1 - at_prev) * noise_pred
                latents = (
                    latents + torch.sqrt(1 - at_prev) * torch.sqrt(at_prev) * noise
                )
                x_step = x0y.detach().clip(-1, 1)
            else:
                latents = (
                    torch.sqrt(at_prev) * z0t + torch.sqrt(1 - at_prev) * noise_pred
                )
                x_step = x0t.detach().clip(-1, 1)

            self.prompt_embeds = prompt_embeds
            self.neg_prompt_embeds = neg_prompt_embeds

            if self.cfg.exp.save_evolution:
                self.save_evo(counter, x_step, latents, input_img, x_list, results256)
            else:
                x_list.append(x_step.clone().detach())
            counter += 1

        with torch.no_grad():
            x_hat = (
                self.model.decode_latents(latents, stay_on_device=True)
                .detach()
                .clip(-1, 1)
            )
        x_list.append(x_hat.clone().detach())

        self.prompt_embeds = prompt_embeds.detach()
        self.neg_prompt_embeds = neg_prompt_embeds.detach()
        return x_hat, x_hat
