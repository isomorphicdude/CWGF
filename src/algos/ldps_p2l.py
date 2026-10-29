import torch
from omegaconf import DictConfig

from models.sd_inverse import SDInversePipeline
from utils.hf_utils import get_noise_pred
from .ldps import LDPS


class LDPSP2L(LDPS):
    def __init__(self, model: SDInversePipeline, cfg: DictConfig):
        super().__init__(model, cfg)
        self.prompt_opt_steps = int(cfg.algo.get("prompt_opt_steps", 1))
        self.prompt_lr = float(cfg.algo.get("prompt_lr", 1e-4))
        self.latent_lr = float(cfg.algo.get("latent_lr", 0.05))
        self.lr_z_mult = float(cfg.algo.get("lr_z_mult", 0.0))
        self.latent_lr_mult = float(
            cfg.algo.get("latent_lr_mult", 0.0)
        )

        self.adam_beta1 = float(cfg.algo.get("adam_beta1", 0.9))
        self.adam_beta2 = float(cfg.algo.get("adam_beta2", 0.999))
        self.prox_gamma = float(cfg.algo.get("prox_gamma", 1.0))
        self.prox_every = int(cfg.algo.get("prox_every", 8))
        self.prox_phase = int(cfg.algo.get("prox_phase", 1))
        self.z_update = str(cfg.algo.get("z_update", "adam")).lower()

        if not hasattr(self.H, "prox_l2"):
            raise TypeError("LDPS-P2L expects self.H to have prox_l2.")

    def _latent_step_size(self, alpha_t: torch.Tensor) -> float:
        alpha_bar_t = float(alpha_t.detach().float().item())
        if self.z_update == "adam":
            return self.latent_lr + self.latent_lr_mult * alpha_bar_t
        if self.z_update == "gd":
            return self.lr_z + self.lr_z_mult * alpha_bar_t
        raise ValueError(
            f"Unsupported z_update mode '{self.z_update}'. Use 'adam' or 'gd'."
        )

    def _noise_pred_and_grad(
        self, latents: torch.Tensor, i: int, y_0: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_pred, grad_nll, x0_pred = super()._noise_pred_and_grad(
            latents, i, y_0, batch_size
        )

        t = self.model.scheduler.timesteps[i]
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        if self.prox_every <= 0:
            return noise_pred, grad_nll, x0_pred
        if (t_int % self.prox_every) != self.prox_phase:
            return noise_pred, grad_nll, x0_pred

        # else run proximal update
        alpha_t = self.model.scheduler.alphas_cumprod[t_int].to(
            device=latents.device, dtype=latents.dtype
        )

        with torch.no_grad():
            prox_x = self.H.prox_l2(
                x0_pred.float(), y=y_0.float(), gamma=self.prox_gamma
            )
            prox_x = prox_x.to(device=latents.device).clip(-1, 1)
            z0_prox = self.model.encode_images(prox_x.to(dtype=latents.dtype))
            noise_pred = (
                torch.sqrt(1 / (1 - alpha_t)) * latents
                - torch.sqrt(alpha_t / (1 - alpha_t)) * z0_prox.detach()
            )

        return noise_pred.detach(), grad_nll.detach(), x0_pred.detach().clip(-1, 1)

    def sample(
        self, x, y, ts, generator, y_0, results256: bool = False, **unusedkwargs
    ):
        self._reset_scheduler_state()
        batch_size = 1 if x.dim() == 3 else x.size(0)
        input_img = x
        y_0 = y_0.to(device=self.device)

        prompt_embeds, neg_prompt_embeds = self._encode_prompt_pair(batch_size)
        prompt_param = torch.nn.Parameter(
            prompt_embeds.detach().to(device=self.device, dtype=torch.float32)
        )
        # prompt update with adam
        prompt_optimizer = torch.optim.Adam(
            [prompt_param],
            lr=self.prompt_lr,
            betas=(self.adam_beta1, self.adam_beta2),
        )
        neg_prompt_embeds = neg_prompt_embeds.to(
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

        latent_param = None
        latent_optimizer = None
        if self.z_update == "adam":
            latent_param = torch.nn.Parameter(latents.detach().to(torch.float32))
            latent_optimizer = torch.optim.Adam(
                [latent_param],
                lr=self.latent_lr,
                betas=(self.adam_beta1, self.adam_beta2),
            )

        counter = 0
        x_list = []
        timesteps = self.model.scheduler.timesteps
        cfg_scale = float(self.cfg.algo.get("cfg_scale", 1.2))

        for i in range(len(timesteps)):
            t = timesteps[i]
            t_int = int(t.item()) if torch.is_tensor(t) else int(t)
            alpha_t = self.model.scheduler.alphas_cumprod[t_int].to(
                device=latents.device, dtype=latents.dtype
            )

            if self.prompt_opt_steps > 0:
                for _ in range(self.prompt_opt_steps):
                    # inner loop 1 for 512 and 5 for 1024
                    prompt_optimizer.zero_grad()
                    noise_prompt = get_noise_pred(
                        self.model,
                        latents.detach(),
                        i,
                        batch_size,
                        self.device,
                        prompt_param,
                        neg_prompt_embeds,
                        guidance_scale=cfg_scale,
                        do_classifier_free_guidance=True,
                    )
                    z0_pred_c = torch.sqrt(1 / alpha_t) * (
                        latents - torch.sqrt(1 - alpha_t) * noise_prompt
                    )
                    x_prompt = self.model.decode_latents(
                        z0_pred_c, stay_on_device=True
                    )
                    prompt_loss = self._measurement_loss(
                        self._forward_measure(x_prompt.float()), y_0
                    )
                    grad_prompt = torch.autograd.grad(
                        prompt_loss, prompt_param, retain_graph=False
                    )[0]
                    prompt_param.grad = grad_prompt.detach().to(torch.float32)
                    prompt_optimizer.step()

            self.prompt_embeds = prompt_param.detach().to(
                device=self.device, dtype=self.model_dtype
            )
            self.neg_prompt_embeds = neg_prompt_embeds

            noise_pred, grad_nll, x0_pred = self._noise_pred_and_grad(
                latents, i, y_0, batch_size
            )

            # latent update
            with torch.no_grad():
                step_out = self.model.scheduler.step(noise_pred, t, latents)
                latents = (
                    step_out.prev_sample
                    if hasattr(step_out, "prev_sample")
                    else step_out[0]
                )
                if self.z_update == "adam":
                    latent_optimizer.zero_grad()
                    latent_optimizer.param_groups[0]["lr"] = self._latent_step_size(
                        alpha_t
                    )
                    latent_param.copy_(latents.detach().to(torch.float32))
                    latent_param.grad = grad_nll.detach().to(torch.float32)
                    latent_optimizer.step()
                    latents = latent_param.detach().to(
                        device=self.device, dtype=self.model_dtype
                    )
                elif self.z_update == "gd":
                    latents = latents - self._latent_step_size(alpha_t) * grad_nll
                else:
                    raise ValueError(
                        f"Unsupported z_update mode '{self.z_update}'. Use 'adam' or 'gd'."
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

        self.prompt_embeds = prompt_param.detach().to(
            device=self.device, dtype=self.model_dtype
        )
        self.neg_prompt_embeds = neg_prompt_embeds.detach()
        return x_hat, x_hat
