"""LATINO variant with a DMD/LCM prior projection step."""

from __future__ import annotations
import os

import torch
from omegaconf import DictConfig

from models.sd_inverse import SDInversePipeline, SDInversePipelineDiff
from .latino import LATINO


class LATINO_D(LATINO):
    def __init__(
        self, model: SDInversePipeline | SDInversePipelineDiff, cfg: DictConfig
    ):
        super().__init__(model, cfg)
        self.reuse_prior_noise = bool(cfg.algo.get("reuse_prior_noise", False))

    @torch.no_grad()
    def lcm_step(
        self,
        latents: torch.Tensor,
        idx: int,
        *,
        noise_t: torch.Tensor | None = None,
        return_noise: bool = False,
    ):
        self._reset_scheduler_state()
        t = self.model.scheduler.timesteps[idx]
        if noise_t is None:
            noise_t = torch.randn_like(latents)

        batch_size = latents.size(0)
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        t_idx = torch.full(
            (batch_size,), t_int, device=latents.device, dtype=torch.long
        )
        z_t = self.model.scheduler.add_noise(latents, noise_t, t_idx)
        noise_pred = self._predict_noise(z_t, idx, batch_size)

        step_out = self.model.scheduler.step(noise_pred, t, z_t, return_dict=False)
        if isinstance(step_out, tuple):
            prev_sample = step_out[0]
            denoised = step_out[1] if len(step_out) > 1 else step_out[0]
        else:
            if not hasattr(step_out, "prev_sample"):
                raise TypeError(
                    "Expected scheduler.step to return a tuple or an object with "
                    "`prev_sample`."
                )
            prev_sample = step_out.prev_sample
            denoised = getattr(step_out, "denoised", prev_sample)

        if return_noise:
            return prev_sample, denoised, noise_pred, noise_t
        return prev_sample, denoised

    @torch.no_grad()
    def prox_latents_from_denoised(
        self,
        z0_pred: torch.Tensor,
        y_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        alpha_t = self.model.scheduler.alphas_cumprod[t_int].to(
            device=z0_pred.device, dtype=z0_pred.dtype
        )
        x = self.model.decode_latents(z0_pred, stay_on_device=True).clip(-1, 1)
        data_term = torch.norm(self._forward_measure(x.float()) - y_0.float()).item()
        if self.cfg.algo.get("ablate", False):
            # delta = 0.1
            var_x_zt = 1.0
            # delta = self._compute_delta(t_int, data_term, self.sigma_y)
            # var_x_zt = 1 - alpha_t
            # delta = 1.5
            delta = self.cfg.algo.ablate_delta
        else:
            var_x_zt = 1 - alpha_t
            # var_x_zt = 1.0
            delta = self._compute_delta(t_int, data_term, self.sigma_y)
        # print(f"Delta at step {t_int}: {delta:.4f}")
        gamma = torch.as_tensor(
            delta * var_x_zt / (self.sigma_y**2),
            device=z0_pred.device,
            dtype=torch.float32,
        )
        debugging = False
        if debugging:
            # debugging: save delta, var_x_zt, and gamma to a file
            debug_name = f"{self.cfg.exp.deg}_debug_latino.pt"
            if os.path.exists(debug_name):
                debug_data = torch.load(debug_name)
                debug_data["delta"].append((t_int, delta))
                debug_data["var_x_zt"].append((t_int, var_x_zt.item()))
                debug_data["gamma"].append((t_int, gamma.item()))
                torch.save(debug_data, debug_name)
            else:
                debug_data = {
                    "delta": [(t_int, delta)],
                    "var_x_zt": [(t_int, var_x_zt.item())],
                    "gamma": [(t_int, gamma.item())],
                }
                torch.save(debug_data, debug_name)

        prox_x = self.H.prox_l2(x.float(), y=y_0.float(), gamma=gamma)
        prox_x = prox_x.to(device=self.device).clip(-1, 1)
        prox_latents = self.model.encode_images(
            prox_x.to(dtype=self.model_dtype), use_mean=True
        ).to(device=z0_pred.device, dtype=z0_pred.dtype)
        return prox_latents.detach(), x.detach()

    def sample(
        self, x, y, ts, generator, y_0, results256: bool = False, **unusedkwargs
    ):
        # print(f"Sampling with LATINO_D: {self.cfg}")
        self._reset_scheduler_state()
        input_img = x
        y_0 = y_0.to(device=self.device)

        init_method = self.cfg.algo.get("init_method", None)
        latents = self.init_latents(
            x,
            generator,
            y_0,
            init_method=init_method,
            prompt_embeds=self.prompt_embeds,
            neg_embeds=self.neg_prompt_embeds,
        )
        if init_method == "pinv":
            latents = self.model.scheduler.add_noise(
                latents,
                torch.randn_like(latents),
                torch.tensor(999, dtype=torch.long, device=latents.device),
            )

        counter = 0
        x_list = []
        noise_t = None
        # print(f"Starting sampling with {len(self.model.scheduler.timesteps)} steps.")
        for i, t in enumerate(self.model.scheduler.timesteps):
            # print(f"Step {counter}: timestep {t.item()}")
            prior_noise = noise_t if self.reuse_prior_noise else None
            _, z0_pred, _, noise_t = self.lcm_step(
                latents,
                i,
                noise_t=prior_noise,
                return_noise=True,
            )
            prox_latents, x0_pred = self.prox_latents_from_denoised(z0_pred, y_0, t)
            if self.cfg.algo.get("ablate", False):
                t_int = int(t.item()) if torch.is_tensor(t) else int(t)
                alpha_t = self.model.scheduler.alphas_cumprod[t_int].to(
                    device=z0_pred.device, dtype=z0_pred.dtype
                )
                cutoff = self.cfg.algo.ablate_cutoff
                N = len(self.model.scheduler.timesteps)
                u = i / max(N - 1, 1)
                floor = 0.15
                base= self.cfg.algo.ablate_kl_coeff
                threshold = self.cfg.algo.get("ablate_kl_threshold", 0.8)
                slope = self.cfg.algo.get("ablate_kl_slope", 0.25)
                if u < threshold:
                    kl_coeff = base
                else:
                    kl_coeff = base * (1 - u) / slope
                
                # kl_coeff = int(i < cutoff)
                latents = kl_coeff * prox_latents + (1 - kl_coeff) * z0_pred
                # latents = kl_coeff * prox_latents + (1 - kl_coeff) * latents
            else:
                latents = prox_latents

            if self.cfg.exp.save_evolution:
                self.save_evo(counter, x0_pred, latents, input_img, x_list, results256)
            else:
                x_list.append(x0_pred.clone().detach())
            counter += 1

        with torch.no_grad():
            x_hat = (
                self.model.decode_latents(latents, stay_on_device=True)
                .detach()
                .clip(-1, 1)
            )

        x_list.append(x_hat.clone().detach())
        return x_hat, x_hat
