import os

import lpips
import torch
from omegaconf import DictConfig

from models.sd_inverse import SDInversePipeline
from utils.degradations_dinv import build_degradation_model
from utils.hf_utils import get_noise_pred
from .base_algo import AlgoBase


class LDPS(AlgoBase):
    def __init__(self, model: SDInversePipeline, cfg: DictConfig):
        self.model = model
        self.H = build_degradation_model(cfg)

        self.cfg = cfg
        self.lr_z = float(cfg.algo.get("lr_z", 1.0))
        self.obs_weight = float(cfg.algo.get("obs_weight", 1.0))
        self.device = self.model._execution_device
        self.model_dtype = getattr(self.model, "dtype", torch.float32)

        # Set diffusion schedule (LATINO iterates over scheduler.timesteps).
        # already set in model init
        # self.model.scheduler.set_timesteps(cfg.algo.num_diffusion_steps, device=self.device)
        
        self.prompt = str(cfg.exp.get("prompt", ""))
        base_batch = int(cfg.algo.get("n_particles", 1))
        self.prompt_embeds, self.neg_prompt_embeds = self._encode_prompt_pair(base_batch)

        self.lpips = lpips.LPIPS(net="vgg").cuda()
        algo_name_folder = cfg.algo.get("name_folder", cfg.algo.name)
        self.evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path, algo_name_folder)
        os.makedirs(self.evol_path, exist_ok=True)

    def _encode_prompt_pair(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [self.prompt] * batch_size
        neg_prompts = [""] * batch_size
        out = self.model.encode_prompt(
            prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=neg_prompts,
        )

        if isinstance(out, tuple):
            prompt_embeds = out[0]
            neg_embeds = out[1] if len(out) > 1 else None
        else:
            prompt_embeds = out
            neg_embeds = None

        if neg_embeds is None:
            neg_out = self.model.encode_prompt(
                [""] * batch_size,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            neg_embeds = neg_out[0] if isinstance(neg_out, tuple) else neg_out

        if prompt_embeds.shape[0] == 2 * batch_size:
            neg_embeds = prompt_embeds[:batch_size]
            prompt_embeds = prompt_embeds[batch_size:]

        return prompt_embeds, neg_embeds

    def _forward_measure(self, x: torch.Tensor) -> torch.Tensor:
        return self.H.A(x)

    def _measurement_loss(
        self, meas_pred: torch.Tensor, y_0: torch.Tensor
    ) -> torch.Tensor:
        if meas_pred.dim() == 3:
            meas_pred = meas_pred.unsqueeze(0)
        if y_0.dim() == 3:
            y_0 = y_0.unsqueeze(0)
        batch_size = meas_pred.shape[0] if meas_pred.dim() > 0 else 1
        residual = (meas_pred - y_0.float()).reshape(batch_size, -1)
        return (
            torch.linalg.vector_norm(residual, dim=1).sum() * self.obs_weight
        )

    def _noise_pred_and_grad(
        self, latents: torch.Tensor, i: int, y_0: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg_scale = float(self.cfg.algo.get("cfg_scale", 1.2))
        scaling_factor = float(getattr(self.model.vae.config, "scaling_factor", 1.0))
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
            meas_pred = self._forward_measure(x0_pred.float())
            nlogpyx = self._measurement_loss(meas_pred, y_0)
            grad_nll = torch.autograd.grad(nlogpyx, latents_in, retain_graph=False)[0]

        return noise_pred.detach(), grad_nll.detach(), x0_pred.detach().clip(-1, 1)

    def sample(self, x, y, ts, generator, y_0, results256: bool = False, **unusedkwargs):
        self._reset_scheduler_state()
        batch_size = 1 if x.dim() == 3 else x.size(0)
        input_img = x

        y_0 = y_0.to(device=self.device)
        latents = self.init_latents(
            x,
            generator,
            y_0,
            init_method=self.cfg.algo.get("init_method", None),
            prompt_embeds=self.prompt_embeds,
            neg_embeds=self.neg_prompt_embeds,
        )

        counter = 0
        x_list = []
        timesteps = self.model.scheduler.timesteps
        for i in range(len(timesteps)):
            t = timesteps[i]
            noise_pred, grad_nll, x0_pred = self._noise_pred_and_grad(
                latents, i, y_0, batch_size
            )

            with torch.no_grad():
                step_out = self.model.scheduler.step(noise_pred, t, latents)
                latents = (
                    step_out.prev_sample
                    if hasattr(step_out, "prev_sample")
                    else step_out[0]
                )
                latents = latents - self.lr_z * grad_nll

            if self.cfg.exp.save_evolution:
                self.save_evo(counter, x0_pred, latents, input_img, x_list, results256)
            else:
                x_list.append(x0_pred.clone().detach())
            counter += 1

        with torch.no_grad():
            scaling_factor = float(getattr(self.model.vae.config, "scaling_factor", 1.0))
            x_hat = (
                self.model.vae.decode(latents / scaling_factor)
                .sample.detach()
                .clip(-1, 1)
            )
        x_list.append(x_hat.clone().detach())
        return x_hat, x_hat
