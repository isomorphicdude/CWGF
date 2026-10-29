import torch
from omegaconf import DictConfig

from models.sd_inverse import SDInversePipeline
from utils.hf_utils import get_noise_pred
from .ldps import LDPS


class PSLD(LDPS):
    def __init__(self, model: SDInversePipeline, cfg: DictConfig):
        super().__init__(model, cfg)
        self.psld_gamma = float(cfg.algo.get("psld_gamma", 1e-1))
        self.psld_omega = float(cfg.algo.get("psld_omega", self.obs_weight))
        if not hasattr(self.H, "A_adjoint"):
            raise TypeError("PSLD expects self.H to have A_adjoint.")

    def _noise_pred_and_grad(
        self, latents: torch.Tensor, i: int, y_0: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg_scale = float(self.cfg.algo.get("cfg_scale", 7.5))
        t = self.model.scheduler.timesteps[i]
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)

        with torch.enable_grad():
            latents_in = latents.detach().requires_grad_(True)
            noise_pred = get_noise_pred(
                self.model,
                latents_in,
                i,
                batch_size,
                self.device,
                self.prompt_embeds,
                self.neg_prompt_embeds,
                guidance_scale=cfg_scale,
                do_classifier_free_guidance=True,
            )

            alpha_t = self.model.scheduler.alphas_cumprod[t_int].to(
                device=latents.device, dtype=latents.dtype
            )
            z0_pred = torch.sqrt(1 / alpha_t) * (
                latents_in - torch.sqrt(1 - alpha_t) * noise_pred
            )
            x0_pred = self.model.decode_latents(z0_pred, stay_on_device=True)

            x_ref = x0_pred.float()
            y_ref = y_0.to(device=x_ref.device, dtype=torch.float32)
            meas_pred = self.H.A(x_ref).to(device=x_ref.device, dtype=x_ref.dtype)
            meas_error = torch.linalg.norm(meas_pred - y_ref) * self.psld_omega

            ortho_project = x_ref - self.H.A_adjoint(meas_pred).to(
                device=x_ref.device, dtype=x_ref.dtype
            )
            parallel_project = self.H.A_adjoint(y_ref).to(
                device=x_ref.device, dtype=x_ref.dtype
            )
            projected_x = parallel_project + ortho_project

            encoded_z0 = self.model.encode_images(
                projected_x.to(dtype=self.model_dtype),
                enable_grad=True,
            ).to(device=z0_pred.device, dtype=z0_pred.dtype)

            inpaint_error = torch.linalg.norm(encoded_z0 - z0_pred) * self.psld_gamma
            total_error = meas_error + inpaint_error
            grad_nll = torch.autograd.grad(total_error, latents_in, retain_graph=False)[
                0
            ]

        return noise_pred.detach(), grad_nll.detach(), x0_pred.detach().clip(-1, 1)
