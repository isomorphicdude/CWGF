import os
import re

import lpips
import torch
from omegaconf import DictConfig

from models.sd_inverse import SDInversePipeline, SDInversePipelineDiff
from utils.degradations_dinv import build_degradation_model
from utils.hf_utils import get_noise_pred, is_sdxl_model_name, repeat_to_batch
from .base_algo import AlgoBase


class LATINO(AlgoBase):
    def __init__(
        self, model: SDInversePipeline | SDInversePipelineDiff, cfg: DictConfig
    ):
        self.model = model
        self.H = build_degradation_model(cfg)

        if not hasattr(self.H, "prox_l2"):
            raise TypeError(
                "LATINO expects a degradation model with prox_l2; use exp.use_dinv=true "
                "or a compatible operator."
            )

        self.cfg = cfg
        self.device = self.model._execution_device
        self.model_dtype = getattr(self.model, "dtype", torch.float32)
        self.cfg_scale = float(cfg.algo.get("cfg_scale", 1.0))
        self.sigma_y = float(cfg.algo.get("sigma_y", cfg.exp.get("sigma_x0", 1e-2)))
        self.prox_eps = float(cfg.algo.get("prox_eps", 1e-6))
        self.prompt = str(cfg.exp.get("prompt", ""))
        self.use_dmd2 = is_sdxl_model_name(cfg.model.name)
        self.use_dmd15 = str(cfg.model.name) == "sd15"

        if not (self.use_dmd2 or self.use_dmd15):
            raise ValueError(
                "LATINO currently supports `model=dmd2`/`model=sdxllora` "
                "(SDXL LCM variants) or `model=sd15` (LCM SD1.5)."
            )

        if self.use_dmd2:
            (
                self.prompt_embeds,
                self.neg_prompt_embeds,
                self.pooled_text_embeds,
                self.neg_pooled_text_embeds,
            ) = self._encode_prompt_sdxl()
            self.time_ids = self.model._get_add_time_ids(
                original_size=(int(cfg.model.output_h), int(cfg.model.output_w)),
                crops_coords_top_left=(0, 0),
                target_size=(int(cfg.model.output_h), int(cfg.model.output_w)),
                dtype=self.model_dtype,
                text_encoder_projection_dim=1280,
            ).to(self.device)
        else:
            self.prompt_embeds, self.neg_prompt_embeds = self._encode_prompt_sd15()
            self.pooled_text_embeds = None
            self.neg_pooled_text_embeds = None
            self.time_ids = None

        self.lpips = lpips.LPIPS(net="vgg").cuda()
        algo_name_folder = cfg.algo.get("name_folder", cfg.algo.name)
        self.evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path, algo_name_folder)
        os.makedirs(self.evol_path, exist_ok=True)

    def _parse_prompt_outputs(self, out) -> tuple[torch.Tensor, ...]:
        if not isinstance(out, tuple):
            return out, None, None, None
        items = list(out)
        if len(items) < 4:
            items.extend([None] * (4 - len(items)))
        return tuple(items[:4])

    def _encode_prompt_sd15(self) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model.encode_prompt(
            [self.prompt],
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[""],
        )
        prompt_embeds, neg_embeds, _, _ = self._parse_prompt_outputs(out)

        if neg_embeds is None and prompt_embeds.shape[0] == 2:
            neg_embeds = prompt_embeds[:1]
            prompt_embeds = prompt_embeds[1:]

        if neg_embeds is None:
            neg_out = self.model.encode_prompt(
                [""],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            neg_embeds, _, _, _ = self._parse_prompt_outputs(neg_out)

        return (
            prompt_embeds.to(device=self.device, dtype=self.model_dtype),
            neg_embeds.to(device=self.device, dtype=self.model_dtype),
        )

    def _encode_prompt_sdxl(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        out = self.model.encode_prompt(
            [self.prompt],
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds, neg_embeds, pooled, neg_pooled = self._parse_prompt_outputs(out)
        if pooled is None:
            raise ValueError("SDXL LATINO requires pooled text embeddings.")

        return (
            prompt_embeds.to(device=self.device, dtype=self.model_dtype),
            None if neg_embeds is None else neg_embeds.to(device=self.device, dtype=self.model_dtype),
            pooled.to(device=self.device, dtype=self.model_dtype),
            None
            if neg_pooled is None
            else neg_pooled.to(device=self.device, dtype=self.model_dtype),
        )

    def _forward_measure(self, x: torch.Tensor) -> torch.Tensor:
        return self.H.A(x)

    def _sr_factor(self) -> int | None:
        match = re.search(r"(\d+)$", str(self.cfg.exp.deg))
        if match is None:
            return None
        return int(match.group(1))

    def _compute_delta_sd15(self, t: int, df: float, sigma_y: float) -> float:
        deg = str(self.cfg.exp.deg)
        factor = self._sr_factor()

        if deg.startswith(("bicubic", "sr")):
            if factor == 16:
                if t > 300:
                    return 1 * 0.02 * df / (1e0 * sigma_y)
                return 1 * 0.02 * df / (1e1 * sigma_y)
            if factor == 32:
                if t > 300:
                    return 2 * df / 1e0
                return 9 * df / 1e1
        elif deg.startswith("inp"):
            if t > 500:
                return 1.0
            return 0.5
        elif deg == "deblur_gauss":
            if t > 300:
                return 1 * df / 1e3
            return 4 * df / 1e4
        elif deg == "deblur_motion":
            if t > 400:
                return 8 * df / 1e4
            return 7 * df / 1e4

        if t > 200:
            return 0.01
        return 1.0

    def _compute_delta_sdxl(self, t: int, df: float, sigma_y: float) -> float:
        deg = str(self.cfg.exp.deg)
        factor = self._sr_factor()

        if deg.startswith(("bicubic", "sr")):
            if factor == 16:
                if t > 300:
                    return 3 * df / 1e1
                return 2 * df / 1e1
            if factor == 32:
                if t > 300:
                    return 1.5 * df / 1e1
                return 3 * df / 1e1
        elif deg == "deblur_gauss":
            blur_sigma = float(self.cfg.exp.get("blur_sigma", 0.0))
            if blur_sigma < 10:
                if t > 400:
                    return 5 * df / 1e4
                return 2 * df / 1e4
            if t > 400:
                return 7 * df / 1e4
            return 3 * df / 1e4
        elif deg == "deblur_motion":
            if sigma_y == 0.01:
                if t > 400:
                    return 4 * df / 1e4
                return 2 * df / 1e4
            if t > 400:
                return 5 * df / 1e5
            if t > 0:
                return 9 * df / 1e5
            return 2 * df / 1e7
        elif deg.startswith("inp"):
            if t > 500:
                return 1.0
            return 0.5

        if t > 200:
            return 0.01
        return 1.0

    def _compute_delta(self, t: int, df: float, sigma_y: float) -> float:
        if self.use_dmd2:
            return self._compute_delta_sdxl(t, df, sigma_y)
        else:
            return self._compute_delta_sd15(t, df, sigma_y)

    @torch.no_grad()
    def _predict_noise(
        self, latents: torch.Tensor, i: int, batch_size: int
    ) -> torch.Tensor:
        if self.use_dmd2:
            added_cond_kwargs = {
                "text_embeds": repeat_to_batch(self.pooled_text_embeds, batch_size),
                "time_ids": repeat_to_batch(self.time_ids, batch_size),
            }
            return get_noise_pred(
                self.model,
                latents,
                i,
                batch_size,
                self.device,
                prompt_embeds=self.prompt_embeds,
                neg_embeds=self.neg_prompt_embeds,
                guidance_scale=self.cfg_scale,
                do_classifier_free_guidance=False,
                added_cond_kwargs=added_cond_kwargs,
            )
        # sdv1.5 lora
        return get_noise_pred(
            self.model,
            latents,
            i,
            batch_size,
            self.device,
            prompt_embeds=self.prompt_embeds,
            neg_embeds=self.neg_prompt_embeds,
            guidance_scale=self.cfg_scale,
            do_classifier_free_guidance=True,
        )

    @torch.no_grad()
    def noise_pred_cond_y(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        y_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        alpha_t = self.model.scheduler.alphas_cumprod[t_int].to(
            device=latents.device, dtype=latents.dtype
        )
        z0_pred = torch.sqrt(1 / alpha_t) * (
            latents - torch.sqrt(1 - alpha_t) * noise_pred
        )
        x = self.model.decode_latents(z0_pred, stay_on_device=True).clip(-1, 1)
        data_term = torch.norm(
            self._forward_measure(x.float()) - y_0.float()
        ).item()
        var_x_zt = 1-alpha_t
        delta = self._compute_delta(t_int, data_term, self.sigma_y)
        sigma_y = self.sigma_y
        gamma = (delta * var_x_zt / (sigma_y**2)).to(
            device=latents.device, dtype=torch.float32
        )
        # gamma = (torch.tensor(0.5) / (sigma_y**2)).to(
        #     device=latents.device, dtype=torch.float32
        # )
        # print(f"gamma at step {t_int}: {gamma.item():.4e}, var_x_zt: {var_x_zt.item():.4e}, data_term: {data_term:.4e}, delta: {delta:.4e}")
        prox_x = self.H.prox_l2(x.float(), y=y_0.float(), gamma=gamma)
        prox_x = prox_x.to(device=self.device).clip(-1, 1)
        z0_pred_cond_y = self.model.encode_images(
            prox_x.to(dtype=self.model_dtype), use_mean=True
        ).to(device=latents.device, dtype=latents.dtype)
        noise_pred_cond_y = (
            torch.sqrt(1 / (1 - alpha_t)) * latents
            - torch.sqrt(alpha_t / (1 - alpha_t)) * z0_pred_cond_y
        )

        return noise_pred_cond_y.detach(), x.detach()
    
    def sample(self, x, y, ts, generator, y_0, results256: bool = False, **unusedkwargs):
        self._reset_scheduler_state()
        batch_size = 1 if x.dim() == 3 else x.size(0)
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
            latents = self.model.scheduler.add_noise(latents, 
                                                     torch.randn_like(latents), 
                                                     torch.tensor(999, dtype=torch.long, device=latents.device))

        counter = 0
        x_list = []
        # print(f"Starting sampling with {len(self.model.scheduler.timesteps)} steps.")
        for i, t in enumerate(self.model.scheduler.timesteps):
            # print(f"Step {counter}: timestep {t.item()}")
            noise_pred = self._predict_noise(latents, i, batch_size)
            noise_pred_cond_y, x0_pred = self.noise_pred_cond_y(latents, noise_pred, y_0, t)

            with torch.no_grad():
                step_out = self.model.scheduler.step(noise_pred_cond_y, t, latents)
                latents = (
                    step_out.prev_sample
                    if hasattr(step_out, "prev_sample")
                    else step_out[0]
                )

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
