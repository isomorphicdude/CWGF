"""CWGF with distilled diffusion models (CWGF-DMD)."""
import os
import math
import shutil
from ast import literal_eval

import lpips
import numpy as np
import omegaconf
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image

from models.sd_inverse import SDInversePipeline, SDInversePipelineDiff
from utils.degradations_dinv import build_degradation_model
from utils.distributions import (
    gaussian_iso_logprob,
    curves_from_alpha_beta,
)
from utils.hf_utils import (
    get_noise_pred as hf_get_noise_pred,
    is_sdxl_model_name,
    repeat_to_batch,
)
from utils.torch_utils import get_logger, ensure_dir
from .base_algo import AlgoBase


class CWGFDMDSimple(AlgoBase):
    def __init__(
        self, model: SDInversePipeline | SDInversePipelineDiff, cfg: DictConfig
    ):
        if cfg.algo.get("use_teacher_prompt", False):
            raise ValueError("algo.use_teacher_prompt is not supported.")
        if str(cfg.algo.get("init_method", None)).lower() == "ddim":
            raise ValueError(
                "algo.init_method='ddim' is not supported; use 'none' or 'pinv'."
            )
        if cfg.algo.get("post_method", "none") not in ("none", None):
            raise ValueError("algo.post_method is not supported; use 'none'.")
        # the multi-timestep prompt update differentiates w.r.t. the prompt
        if cfg.algo.get("num_t_samples", 0) >= 1 and not cfg.algo.get(
            "prompt_opt", False
        ):
            raise ValueError(
                "algo.num_t_samples>=1 requires algo.prompt_opt=true; "
                "set algo.num_t_samples=0 to run without prompt optimisation."
            )

        self.model = model
        self.H = build_degradation_model(cfg)
        self.cfg = cfg
        self.lr_z = cfg.algo.lr_z
        self.device = self.model._execution_device
        self.lpips = lpips.LPIPS(net="vgg").cuda()
        algo_name_folder = cfg.algo.get("name_folder", cfg.algo.name)
        evol_root = os.path.join(cfg.exp.root, cfg.exp.evol_path)
        if cfg.exp.load_img_id:
            gamma = cfg.algo.get("gamma", 0.0)
            evol_leaf = (
                f"{algo_name_folder}_single_"
                f"{cfg.exp.deg}_{cfg.exp.img_id}_{gamma}_{self.lr_z}"
            )
        else:
            evol_leaf = str(algo_name_folder)
        self.evol_path = os.path.join(evol_root, evol_leaf)
        if os.path.exists(self.evol_path):
            shutil.rmtree(self.evol_path)
        os.makedirs(self.evol_path, exist_ok=True)

        self.logger = get_logger(__name__, cfg=cfg)
        self.num_steps = cfg.algo.num_steps
        self.num_diffusion_steps = cfg.algo.num_diffusion_steps
        self.cfg_scale = cfg.algo.get("cfg_scale", 1.2)
        self.prompt_opt = cfg.algo.get("prompt_opt", False)
        self.lr_prompt = cfg.algo.get("lr_prompt", 1.0)
        self.use_dmd2 = is_sdxl_model_name(cfg.model.name)
        self.model_dtype = (
            torch.float16 if cfg.model.get("use_fp16", False) else torch.float32
        )
        print(f"Using {cfg.algo.n_particles} particles for CWGF-DMD.")
        self.sdxl_guidance_mode = None
        if self.use_dmd2:
            self.sdxl_guidance_mode = cfg.algo.get("sdxl_guidance_mode", "cfg")
            self.do_classifier_free_guidance = self.sdxl_guidance_mode == "cfg"
            if self.sdxl_guidance_mode == "latino":
                self.logger.info(
                    "Using LATINO-style SDXL guidance (single conditional pass, no CFG)."
                )
        else:
            self.do_classifier_free_guidance = cfg.algo.get(
                "do_classifier_free_guidance", True
            )

        num_timesteps = self.model.scheduler.timesteps.numel()
        if num_timesteps != self.num_diffusion_steps:
            raise ValueError(
                "Scheduler timesteps do not match num_diffusion_steps: "
                f"got {num_timesteps} for requested {self.num_diffusion_steps}."
            )

        self.decoder_std = float(cfg.algo.get("decoder_std", 0.3))
        # stepwise weights over the scheduler timesteps
        w_t = self._expand_stepwise_weights(
            self._parse_config_list(
                "w_t", default_value=1.0, num_steps=self.num_diffusion_steps
            ),
            "w_t",
        )
        if cfg.algo.get("use_kl_coeff", False):
            # NOTE: overrides algo.w_t
            half_value = float(cfg.algo.get("half_value", 0.5))
            half_count = int(cfg.algo.get("half_count", 1))
            total_count = cfg.algo.num_diffusion_steps
            base_value = 1.0
            if half_count < 1 or half_count > total_count:
                raise ValueError(
                    f"half_count must be in [1, {total_count}], got {half_count}."
                )
            w_t = [
                half_value if i < half_count else base_value
                for i in range(total_count)
            ]
            self.logger.info(
                "Constructed w_t from half_value=%s half_count=%s: %s",
                half_value,
                half_count,
                w_t,
            )
        self.w_t = self._expand_stepwise_weights(w_t, "w_t")
        self.prompt_w = self._expand_stepwise_weights(
            self._parse_config_list(
                "prompt_w", default_value=1.0, num_steps=self.num_diffusion_steps
            ),
            "prompt_w",
        )

        # precompute weights and timestep sampling distribution
        alpha_bar_sd = (
            self.model.scheduler.alphas_cumprod.detach().cpu().numpy().astype(np.float64)
        )
        betas_sd = self.model.scheduler.betas.detach().cpu().numpy().astype(np.float64)
        t_sd = np.linspace(0.0, 1.0, len(alpha_bar_sd), dtype=np.float64)
        _, snr_inv, _, _, _ = curves_from_alpha_beta(
            alpha_bar_sd,
            betas_sd,
            pmean=cfg.algo.get("edm_mu", 2.0),
            psigma=cfg.algo.get("edm_sigma", 2.4),
            sigma_data=cfg.algo.get("data_sigma", 0.5),
            t=t_sd,
        )
        self.snr_inv = torch.from_numpy(snr_inv).to(
            device=self.device, dtype=self.model.dtype
        )
        _, self.prompt_w_interp_anti = self._build_inv_snr_integral(
            self.prompt_w, t_sd
        )
        self.beta_sd = torch.from_numpy(betas_sd).to(
            device=self.device, dtype=self.model.dtype
        )

        self.decreasing_idxs = torch.arange(self.num_diffusion_steps, device=self.device)
        if cfg.exp.num_steps == 2 * cfg.algo.num_diffusion_steps:
            self.decreasing_idxs = torch.repeat_interleave(self.decreasing_idxs, 2)

    def use_prompt_check(self, step):
        return self.cfg.algo.get("prompt_check", False) and (
            ((step + 1) % self.cfg.algo.get("prompt_check_every", 1) == 0)
            or (step == self.num_steps - 1)
        )

    def sample(self, x, y, ts, generator, y_0, results256=False, **unusedkwargs):
        # here batch size is the number of particles
        y_0 = y_0.to(self.device, dtype=self.model_dtype)
        input_img = x

        # optimize prompts
        out_dict = self.setup_prompt()
        text_embeds = torch.autograd.Variable(
            out_dict["text_embed0"].detach().clone(), requires_grad=True
        )
        # prompt project radius
        project_radius = float(self.cfg.algo.get("prompt_project_radius", 5.0))
        ref_dim = 768  # for sd1.5
        embed_dim = int(out_dict["text_embed0"].shape[-1])
        project_radius = project_radius * math.sqrt(float(embed_dim) / float(ref_dim))

        # optimize latents
        latents = self.init_latents(
            x,
            generator,
            y_0,
            init_method=self.cfg.algo.get("init_method", None),
            prompt_embeds=out_dict["text_embed0"],
            neg_embeds=out_dict.get(
                "neg_text_embeds", out_dict.get("uncond_embed", None)
            ),
        )
        latents = torch.autograd.Variable(latents.detach(), requires_grad=True)
        # get extra kwargs
        if self.use_dmd2:
            output_size = (int(self.cfg.model.output_h), int(self.cfg.model.output_w))
            time_ids = self.model._get_add_time_ids(
                original_size=output_size,
                crops_coords_top_left=(0, 0),
                target_size=output_size,
                dtype=self.model_dtype,
                text_encoder_projection_dim=1280,
            ).to(self.device)
        else:
            time_ids = None
        extra_sd_kwargs = {
            "text_embed0": out_dict["text_embed0"],
            "uncond_embed": out_dict.get("uncond_embed", None),
            "neg_text_embeds": out_dict.get("neg_text_embeds", None),
            "pooled_text_embeds": out_dict.get("pooled_text_embeds", None),
            "neg_pooled_text_embeds": out_dict.get("neg_pooled_text_embeds", None),
            "add_time_ids": time_ids,
        }

        counter = 0
        x_list = []
        kl_grad_list = []
        t_list = []

        # gradient flow loop
        for step in range(self.num_steps):
            self.logger.info(f"Step {step+1}/{self.num_steps}")

            prompt_loss = 0.0

            # timestep selection
            num_t_samples = self.cfg.algo.get("num_t_samples", 1)
            if num_t_samples == 0:
                num_t_samples = 1  # special value to use the same timestep idx
            t, idx = self.sample_time(step, num_t_samples=num_t_samples)
            print(f"Step {step}/{self.num_steps}, sampled timestep {t}")
            if torch.is_tensor(t) and t.ndim > 0:
                t_label = "-".join(str(int(ti)) for ti in t.detach().cpu().tolist())
            else:
                t_label = str(int(t.item()))
            alpha_t = self.model.scheduler.alphas_cumprod[t].to(
                device=self.device, dtype=latents.dtype
            )
            alpha_t_log = torch.log1p(-alpha_t.detach().float())
            noise_t = torch.randn(
                latents.shape,
                generator=generator,
                device=latents.device,
                dtype=latents.dtype,
            )

            # prior latent flow
            noise_pred, prev_latents, noise_t, prior_vec = self.prior_step_latents(
                latents,
                text_embeds,
                idx=idx,
                lr_z=self.lr_z,
                noise_t=noise_t,
                extra_sd_kwargs=extra_sd_kwargs,
                alpha_t=alpha_t,
            )

            if self.cfg.algo.get("num_t_samples", 0) == 0:
                prompt_idx = idx

                self.logger.info(f"Prompt optimization timestep idx: {prompt_idx}")
                prompt_t = self.model.scheduler.timesteps[prompt_idx]
                prompt_loss, _ = self.prior_autograd_step_prompt(
                    noise_pred=noise_pred,
                    t=prompt_t,
                    idx=prompt_idx,
                    alpha_t=self.model.scheduler.alphas_cumprod[prompt_t].to(
                        device=self.device, dtype=latents.dtype
                    ),
                    alpha_t_log=math.log1p(
                        -self.model.scheduler.alphas_cumprod[prompt_t]
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    ),
                    latents=prev_latents,  # use previous latents for prompt optimization
                    text_embeds=text_embeds,
                    extra_sd_kwargs=extra_sd_kwargs,
                    noise_t=noise_t,
                    prompt_opt=self.cfg.algo.get("prompt_opt", True),
                    project_radius=project_radius,
                )
            else:
                prompt_loss, _ = self.prior_autograd_step_prompt_multi(
                    latents,
                    text_embeds,
                    extra_sd_kwargs,
                    prompt_opt=self.cfg.algo.get("prompt_opt", True),
                    project_radius=project_radius,
                )
            if self.use_prompt_check(step):
                # prompt_check_y = None 
                prompt_check_y = y_0[0].unsqueeze(0)  # or None
                self.prompt_check(
                    generator,
                    out_dict.get("uncond_embed", None),
                    text_embeds,
                    postfix=f"{counter}",
                    y=prompt_check_y,
                    extra_sd_kwargs=extra_sd_kwargs,
                )
                with torch.no_grad():
                    x_pred_z = self.model.decode_latents(latents, stay_on_device=True)
                    save_image(
                        self.postprocess(x_pred_z.detach().clip(-1, 1)),
                        f"step{counter}_time{t_label}_prior.png",
                    )

            num_kl_steps = 1
            ablate_euler = self.cfg.algo.get("ablate_euler", False)
            kl_input = torch.autograd.Variable(prev_latents.detach().clone(), requires_grad=True) if ablate_euler else latents
            kl_grad_norm, _, _, like_vec = self.kl_autograd_step(
                kl_input,
                y_0,
                idx=idx,
                t_tensor=t,
                alpha_t=alpha_t,
                lr_z=self.lr_z,
                generator=generator,
                num_kl_steps=num_kl_steps,
                curr_step=step,
                return_direc=True
            )
            kl_grad_list.append(kl_grad_norm)

            print(f"Step {counter}, Prompt Loss: {prompt_loss:.6f}\n")
            if ablate_euler:
                # no lie trotter splitting
                print("Ablating Euler step.")
                latents = torch.autograd.Variable(prev_latents.detach() + prior_vec + like_vec, requires_grad=True
                )

            with torch.no_grad():
                x_pred_z = self.model.decode_latents(latents, stay_on_device=True)
                
            should_decode = (
            self.cfg.exp.save_evolution
            or step == self.num_steps - 1
            )
            if should_decode:
                with torch.no_grad():
                    x_pred_z = self.model.decode_latents(
                        latents, stay_on_device=True
                    )

            if self.cfg.exp.save_evolution:
                # save_evo skips (and returns None for) steps off the
                # exp.save_every grid, so the output is taken from x_pred_z
                x_pred_save = x_pred_z.clip(-1, 1)
                self.save_evo(
                    counter,
                    x_pred_save,
                    latents,
                    input_img,
                    x_list,
                    results256,
                )

            counter += 1
            t_list.append(t)

        return x_pred_z.detach().clip(-1, 1), x_pred_z.detach()

    def sample_time(self, step, num_t_samples=1):
        timestep_sampling = self.cfg.algo.get("timestep_sampling", "cyclic")
        num_t_samples = 1
        if timestep_sampling == "cyclic":
            idx = step % self.num_diffusion_steps
            if step >= self.num_steps // 4:
                curr_mod = self.num_diffusion_steps // 2
                idx = step % curr_mod + curr_mod
            t = self.model.scheduler.timesteps[idx]
        elif timestep_sampling == "decreasing":
            idx = self.decreasing_idxs[step]
            t = self.model.scheduler.timesteps[idx]


        elif timestep_sampling == "uniform":
            # uniform sample from all timesteps
            idx = torch.randint(0, self.num_diffusion_steps, (num_t_samples,)).item()
            t = self.model.scheduler.timesteps[idx]

        else:
            raise ValueError(
                f"Unknown cfg.algo.timestep_sampling={timestep_sampling!r} "
                f"expected one of ['cyclic', 'decreasing', 'uniform']."
            )
        return t, idx

    def setup_prompt(self):
        out_dict = {}
        if self.use_dmd2:
            text_embeds, neg_text_embeds, pooled_text_embeds, neg_pooled_text_embeds = (
                self.setup_prompt_sdxl()
            )
            out_dict["text_embed0"] = text_embeds
            out_dict["neg_text_embeds"] = neg_text_embeds
            out_dict["pooled_text_embeds"] = pooled_text_embeds
            out_dict["neg_pooled_text_embeds"] = neg_pooled_text_embeds
        else:
            text_embeds, uncond_embeds = self.setup_prompt_sd15()
            out_dict["text_embed0"] = text_embeds
            out_dict["uncond_embed"] = uncond_embeds
        return out_dict

    def setup_prompt_sd15(self):
        # set up for sd1.5 LCM
        prompt = [self.cfg.exp.prompt]
        text_inputs = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        prefix = "A photo of a"
        prefix_inputs = self.model.tokenizer(
            [prefix],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # attention_mask marks real tokens vs padding
        prefix_len = int(prefix_inputs.attention_mask[0].sum().item())
        # with BOS
        self.prefix_len = prefix_len+1
        if prompt[0] == "":
            self.logger.warning("Empty prompt detected, setting prefix_len to 1 to avoid zeroing out all gradients.")
            self.prefix_len = 1

        self.logger.info(f"Prefix tokenized to {self.prefix_len} tokens: {prefix_inputs.input_ids[0, :prefix_len].tolist()}")
        with torch.no_grad():
            text_embeddings = self.model.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        text_embeddings = text_embeddings.to(self.device, dtype=self.model_dtype)
        self.text_embeddings = text_embeddings

        uncond_inputs = self.model.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.model.text_encoder(
                uncond_inputs.input_ids.to(self.device)
            )[0]
        uncond_embeddings = uncond_embeddings.to(self.device, dtype=self.model_dtype)
        return text_embeddings, uncond_embeddings

    def setup_prompt_sdxl(self):
        # set up for sdxl base model
        prompt = [self.cfg.exp.prompt]
        # get prefix length for sdxl
        prefix = "A photo of a"
        text_inputs = self.model.tokenizer(
                    [prefix],
                    padding="max_length",
                    max_length=self.model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False
                )
        prefix_len = int(text_inputs.attention_mask[0].sum().item())
        self.prefix_len = prefix_len + 1  # add 1 for BOS token
        use_cfg = self.sdxl_guidance_mode == "cfg"
        return self.model.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=use_cfg,
        )

    def lcm_step(
        self,
        latents,
        idx,
        *,
        text_embeds,
        extra_sd_kwargs,
        noise_t: torch.Tensor | None = None,
        return_noise: bool = False,
        **unusedkwargs,
    ):
        # reset since we called scheduler.step
        self._reset_scheduler_state()
        t = self.model.scheduler.timesteps[idx]
        if text_embeds is None or extra_sd_kwargs is None:
            raise ValueError("lcm_step requires text_embeds and extra_sd_kwargs.")

        if noise_t is None:
            noise_t = torch.randn_like(latents)

        batch_size = latents.size(0)
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        t_idx = torch.full((batch_size,), t_int, device=latents.device, dtype=torch.long)
        z_t = self.model.scheduler.add_noise(latents, noise_t, t_idx)

        noise_pred = self.get_noise_pred(
            z_t,
            idx,
            text_embeds,
            extra_sd_kwargs,
        )
        latents, denoised = self.model.scheduler.step(
            noise_pred, t, z_t, return_dict=False
        )
        if return_noise:
            return latents, denoised, noise_pred, noise_t, z_t
        else:
            return latents, denoised

    def prior_step_latents(
        self,
        latents: torch.Tensor,
        text_embeds: torch.Tensor,
        idx: int,
        lr_z: float,
        *unusedargs,
        extra_sd_kwargs=None,
        noise_t: torch.Tensor | None = None,
        bernoulli: bool = False,
        alpha_t: torch.Tensor | None = None,
        **unusedkwargs,
    ):
        """Updates latents and text_embeds in-place."""
        no_prior = self.cfg.algo.get("no_prior", False)
        eff_lr = self.w_t[idx] * lr_z
        _, lcm_pred, noise_pred, noise_t, z_t = self.lcm_step(
            latents,
            idx,
            text_embeds=text_embeds,
            extra_sd_kwargs=extra_sd_kwargs,
            noise_t=noise_t,
            return_noise=True
        )
        prev_latents = latents.detach().clone()
        batch_size = latents.size(0)
        with torch.no_grad():
            # with repulsion term
            latents.copy_(latents.detach().clone() * (1 - eff_lr) + eff_lr * lcm_pred)
            mu_t_flat = (latents * alpha_t.sqrt()).detach().view(batch_size, -1)
            forward_ker = gaussian_iso_logprob(
                z_t.view(batch_size, -1),
                mu_t_flat,
                math.log1p(
                        -alpha_t.detach().float().cpu().item()
                    ),
                sequential=False,  # save memory
            )
            # size (bs, bs) -> (bs,)
            temp = self.temp_func(alpha_t, (1 - alpha_t).sqrt())
            w = torch.softmax(forward_ker.float() / temp, dim=1) # (bs, bs)
            # print(f"size of w in latents update: {w.shape}, {w}")
            latents_copy = latents.detach().clone()
            if not no_prior:
                # for component ablation
                latents.copy_(
                    latents_copy + eff_lr * (latents_copy - (w @ latents_copy.view(batch_size, -1).float()).view_as(latents).to(latents.dtype))
                )
            else:
                latents.copy_(prev_latents)
            update_vec = (latents - prev_latents).detach()
            
        return noise_pred, prev_latents, noise_t, update_vec

    @torch.no_grad()
    def forward_score_est(self, z_t, latents, alpha_t, alpha_t_log):
        batch_size = latents.size(0)
        z_t_grad = torch.autograd.Variable(z_t, requires_grad=False)
        mu_t_flat = (latents * alpha_t.sqrt()).detach().view(batch_size, -1)
        forward_ker = gaussian_iso_logprob(
            z_t_grad.view(batch_size, -1),
            mu_t_flat,
            alpha_t_log,
            sequential=False,  # save memory
        )
        # size (bs, bs) -> (bs,)
        _dtype = latents.dtype
        w = torch.softmax(
            forward_ker.float() / self.temp_func(alpha_t, (1 - alpha_t).sqrt()), dim=1
        ).to(
            dtype=_dtype
        )  # (bs, bs)
        forward_score = (
            -1
            * (math.exp(-alpha_t_log))
            * (
                (z_t_grad.view(batch_size, -1).detach() - w @ mu_t_flat).view(
                    latents.shape
                )
            ).detach()
        )
        return forward_score

    def temp_func(self, alpha, sigma):
        if self.cfg.algo.use_temp:
            temp = 128 * sigma  / alpha
        else:
            temp = 1.0
        return temp

    def prior_autograd_step_prompt(
        self,
        noise_pred,
        t,
        idx,
        alpha_t,
        alpha_t_log,
        latents,
        text_embeds,
        extra_sd_kwargs,
        noise_t,
        prompt_opt=True,
        project_radius: float = 15.0,
    ):
        """
        Version reusing noise pred from the latents update.
        """
        batch_size = latents.size(0)
        t_int = int(t)
        snr_inv = (1 - alpha_t) / alpha_t
        if self.cfg.algo.weighting == "inv_snr":
            scale_prompt = 0.5 * snr_inv * self.beta_sd[t_int]
        else:
            raise ValueError(f"Unknown weighting {self.cfg.algo.weighting}")
        t_idx = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)
        z_t = self.model.scheduler.add_noise(latents, noise_t, t_idx).detach() 
        forward_score = self.forward_score_est(z_t, latents, alpha_t, alpha_t_log)
        score_t = score_t = -1 * noise_pred / (1 - alpha_t).sqrt()
        
        # for prompts
        prompt_loss = (
            (score_t - forward_score).reshape(batch_size, -1).square().sum(dim=1).mean() * scale_prompt
        )       
        if prompt_opt: 
            prompt_grad  = torch.autograd.grad(
                prompt_loss, 
                text_embeds,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )[0]
        else:
            prompt_grad = None
            prompt_grad_norm = torch.tensor(0.0)
        with torch.no_grad():
            if prompt_opt and prompt_grad is not None:
                prompt_grad[0, :self.prefix_len] = 0.0
                prompt_grad_norm = prompt_grad.norm()
                text_embeds -= self.lr_prompt * prompt_grad
                print(f"Prompt embed norm: {text_embeds.norm().item():.6f}")
                text_embed0 = extra_sd_kwargs.get("text_embed0", None)
                print(f"Initial prompt embed norm: {text_embed0.norm().item():.6f}" if text_embed0 is not None else "No initial prompt embed.")
                if text_embed0 is not None:
                    text_embeds.copy_(
                        self.project_onto_ball(
                            text_embeds, text_embed0, radius=project_radius
                        )
                    )
                self.logger.info(f"Prompt grad norm: {prompt_grad_norm.item():.6f}")
            
        return prompt_loss.item(), prompt_grad_norm.item()

    def prior_autograd_step_prompt_multi( 
        self,
        latents,
        text_embeds,
        extra_sd_kwargs,
        prompt_opt=True,
        project_radius: float = 15.0,
        **unusedkwargs,
    ):
        # sampled uniformly
        num_t_samples = self.cfg.algo.get("num_t_samples", 1)
        batch_size = latents.size(0)
        _dtype = latents.dtype
        idx = torch.randint(0, self.num_diffusion_steps, (num_t_samples,))
        t = self.model.scheduler.timesteps[idx]
        alpha_t = self.model.scheduler.alphas_cumprod[t].to(
                device=self.device, dtype=latents.dtype
            )
        alpha_t_log = torch.log1p(-alpha_t.detach().float())
        alpha_t_tensor = torch.as_tensor(
            alpha_t, device=self.device, dtype=_dtype
        ).reshape(-1)
        alpha_t_log_tensor = torch.as_tensor(
            alpha_t_log, device=self.device, dtype=torch.float32
        ).reshape(-1)
        
        t_size = t.numel()
        full_bs = batch_size * t_size
        latent_shape = latents.shape[1:]

        def _repeat_per_particle(values: torch.Tensor) -> torch.Tensor:
            return values.view(t_size, 1).expand(t_size, batch_size).reshape(-1)

        t_idx = _repeat_per_particle(t)
        idx_flat = _repeat_per_particle(idx)
        self.logger.info(
            f"Selected times: {t.cpu().numpy()}"
        )
        alpha_t_flat = _repeat_per_particle(alpha_t_tensor)
        alpha_t_log_flat = _repeat_per_particle(alpha_t_log_tensor)
        lat = (
            latents.unsqueeze(0)
            .repeat(t_size, 1, 1, 1, 1)
            .view(-1, *latent_shape)
            .contiguous()
        )
        noise = torch.randn_like(lat)
        z_t = self.model.scheduler.add_noise(lat, noise, t_idx).detach()
        et = self.get_noise_pred(
            z_t,
            idx.to('cpu') if t_size > 1 else int(idx.item()),
            text_embeds,
            extra_sd_kwargs,
        )
        alpha_t_view = alpha_t_flat.view(-1, 1, 1, 1)
        score_t = -1 * et / (1 - alpha_t_view).sqrt()
        with torch.no_grad():
            z_t_grad = torch.autograd.Variable(z_t, requires_grad=False)
            mu_t_flat = (lat * alpha_t_view.sqrt()).detach().view(full_bs, -1)
            vmapped_gaussian_iso_logprob = torch.vmap(
                gaussian_iso_logprob, in_dims=(0, 0, 0)
            )
            forward_ker = vmapped_gaussian_iso_logprob(
                z_t_grad.view(t_size, batch_size, -1),
                mu_t_flat.view(t_size, batch_size, -1),
                alpha_t_log_tensor,
            )
            temp = self.temp_func(
                alpha_t_tensor.view(t_size, 1, 1),
                (1 - alpha_t_tensor).sqrt().view(t_size, 1, 1),
            )
            w = torch.softmax(forward_ker.float() / temp, dim=-1)
            forward_score = -1 * (
                torch.exp(-alpha_t_log_flat.view(-1, 1, 1, 1))
            ) * (
                (
                    z_t_grad.view(t_size, batch_size, -1).detach()
                    - torch.bmm(
                        w, mu_t_flat.view(t_size, batch_size, -1).float()
                    ).to(_dtype)
                )
            ).view(*lat.shape).detach()
            assert (
                forward_score.shape == score_t.shape
            ), (
                f"forward_score shape {forward_score.shape} does not match "
                f"score_t shape {score_t.shape}"
            )

        if self.cfg.algo.weighting == "inv_snr":
            scale_prompt_flat = (
                0.5 * self.prompt_w_interp_anti[t_idx] * self.beta_sd[t_idx]
            )
        else:
            raise ValueError(f"Unknown weighting {self.cfg.algo.weighting}")
        scale_prompt = scale_prompt_flat.view(t_size, batch_size)
        score_diff_sq_norm = torch.norm(
            (forward_score - score_t).view(t_size, batch_size, -1) ** 2,
            dim=-1,
        )
        prompt_loss = (score_diff_sq_norm * scale_prompt).mean()    
        
        prompt_grad  = torch.autograd.grad(
            prompt_loss, 
            text_embeds,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )[0] 
        
        with torch.no_grad():
            if prompt_grad is not None and prompt_opt:
                # prompt_grad[0, :4] = 0.0  # do not update BOS token
                prompt_grad[0, :self.prefix_len] = 0.0
                # self.lr_prompt = self.lr_prompt * 0.97
                text_embeds -= self.lr_prompt * prompt_grad
                print(f"Prompt embed norm: {text_embeds.norm().item():.6f}")
                text_embed0 = extra_sd_kwargs.get("text_embed0", None)
                print(f"Initial prompt embed norm: {text_embed0.norm().item():.6f}" if text_embed0 is not None else "No initial prompt embed.")
                if text_embed0 is not None:
                    text_embeds.copy_(
                        self.project_onto_ball(
                            text_embeds, text_embed0, radius=project_radius
                        )
                    )
                prompt_grad_norm = torch.norm(prompt_grad)
                self.logger.info(f"Prompt grad norm: {prompt_grad_norm.item():.6f}")
        return prompt_loss.item(), prompt_grad_norm.item()

    @torch.no_grad()
    def project_onto_ball(self, theta, theta_0, radius):
        """Project theta onto a ball of radius `radius` centered at `theta_0`."""
        delta = theta - theta_0
        norm_delta = torch.norm(delta, p=2, dim=-1, keepdim=True)
        scaling_factor = torch.clamp(radius / (norm_delta + 1e-8), max=1.0)
        if (scaling_factor < 1.0).any():
            print("Projected!")
        return theta_0 + scaling_factor * delta

    def kl_autograd_step(
        self,
        latents: torch.Tensor,
        y_0: torch.Tensor,
        *,
        idx: int,
        t_tensor: torch.Tensor,
        alpha_t: torch.Tensor,
        lr_z: float,
        prior_grad_norm: torch.Tensor = None,
        generator: torch.Generator | None = None,
        num_kl_steps: int = 1,
        curr_step=None,
        return_direc: bool = False
    ):
        """
        Returns kl_grad_norm, x_pred_z, decoded_for_vis
        """
        x_pred_z = None
        decoded_for_vis = None
        kl_grad_norm = None
        ablate_decoder = self.cfg.exp.get("ablate_decoder", False)
        prev_latents = latents.detach().clone()
        for kl_step_idx in range(num_kl_steps):
            kl_unscaled, x_pred_z, decoded_for_vis, _encoded = self.kl_loss(
                latents,
                y_0,
                i=idx,
                t_tensor=t_tensor,
                alpha_t=alpha_t,
                generator=generator,
            )
            kl_grad = torch.autograd.grad(
                kl_unscaled,
                latents,
                retain_graph=not ablate_decoder,
                create_graph=False,
            )[0]
            kl_grad_norm = torch.linalg.norm(kl_grad.detach())
            if ablate_decoder:
                (
                    decoder_unscaled,
                    _decoder_x_pred_z,
                    _decoder_for_vis,
                    _decoder_encoded,
                ) = self._kl_loss_dinv_decoder(
                    latents,
                    y_0,
                    i=idx,
                    t_tensor=t_tensor,
                    alpha_t=alpha_t,
                    generator=generator,
                )
                decoder_grad = torch.autograd.grad(
                    decoder_unscaled,
                    latents,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                encoder_grad_flat = (
                    kl_grad.detach()
                ).float().reshape(-1)
                decoder_grad_flat = decoder_grad.detach().float().reshape(-1)
                encoder_grad_norm = torch.linalg.norm(encoder_grad_flat)
                decoder_grad_norm = torch.linalg.norm(decoder_grad_flat)
                eps = 1e-12
                cosine_similarity = torch.dot(
                    encoder_grad_flat, decoder_grad_flat
                ) / (encoder_grad_norm * decoder_grad_norm).clamp_min(eps)
                decoder_to_encoder_norm_ratio = decoder_grad_norm / (
                    encoder_grad_norm.clamp_min(eps)
                )

                metrics = {
                    "cosine_similarity": cosine_similarity.item(),
                    "encoder_grad_norm": encoder_grad_norm.item(),
                    "decoder_grad_norm": decoder_grad_norm.item(),
                    "decoder_to_encoder_norm_ratio": (
                        decoder_to_encoder_norm_ratio.item()
                    ),
                    "t": int(t_tensor.detach().cpu().item()),
                    "idx": (
                        int(idx.detach().cpu().item())
                        if torch.is_tensor(idx)
                        else int(idx)
                    ),
                    "step": int(curr_step) if curr_step is not None else -1,
                    "kl_step": kl_step_idx,
                }

                with torch.no_grad():
                    scaling_fac = kl_grad_norm / decoder_grad_norm.clamp_min(
                        1e-12
                    )
                    # print(f"scaling_fac.item(): {scaling_fac.item():.6f}")
                dec_scale = self.cfg.algo.get("decoder_scale", 1.0)
                kl_grad_step = dec_scale * decoder_grad * scaling_fac
            else:
                kl_loss = kl_unscaled * 0.5
                (kl_grad_step,) = torch.autograd.grad(
                    kl_loss,
                    latents,
                    retain_graph=False,
                    create_graph=False,
                )

            with torch.no_grad():
                #NOTE: with prior regularisation
                if self.cfg.algo.get("no_likelihood", False):
                    kl_coeff = 0.0
                else:
                    kl_coeff = self.get_kl_coeff(idx)
                kl_grad_step *= kl_coeff
                latents -= lr_z * kl_grad_step / num_kl_steps
        with torch.no_grad():
            update_vec = (latents - prev_latents).detach()
            if return_direc:
                return kl_grad_norm.item(), x_pred_z, decoded_for_vis, update_vec
        return kl_grad_norm.item(), x_pred_z, decoded_for_vis

    def get_kl_coeff(self, idx):
        if self.cfg.algo.get("use_kl_coeff", False):
            threshold = self.cfg.algo.get("ablate_kl_threshold", None)
            threshold_int = self.cfg.algo.get("ablate_kl_threshold_int", None)
            if threshold is not None and threshold_int is not None:
                raise ValueError(
                    "Both ablate_kl_threshold and ablate_kl_threshold_int are set. Please set only one."
                )
            if threshold is None and threshold_int is None:
                # Preserve the schedule used before integer thresholds were added.
                threshold = 0.85

            num_timesteps = len(self.model.scheduler.timesteps)
            if threshold_int is not None:
                if (
                    isinstance(threshold_int, bool)
                    or int(threshold_int) != threshold_int
                    or not 0 <= int(threshold_int) <= num_timesteps
                ):
                    raise ValueError(
                        "ablate_kl_threshold_int must be an integer in "
                        f"[0, {num_timesteps}], got {threshold_int!r}."
                    )
                threshold_int = int(threshold_int)

            slope = self.cfg.algo.get("ablate_kl_slope", 0.19)
            u = idx / max(num_timesteps - 1, 1)
            print(idx, u)
            if threshold is not None:
                less_than_threshold = u < threshold
            else:
                less_than_threshold = idx < threshold_int
            if less_than_threshold:
                kl_coeff = 1.0
            else:
                kl_coeff = 1.0 * (1 - u) / slope
            print(f"kl_coeff: {kl_coeff}")
            return kl_coeff
        else:
            return 1.0

    def kl_loss(
        self,
        latents,
        y_0,
        i=None,
        t_tensor=None,
        alpha_t=None,
        *,
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        return self._kl_loss_dinv(
            latents,
            y_0,
            i=i,
            t_tensor=t_tensor,
            alpha_t=alpha_t,
            generator=generator,
            **kwargs
        )

    def _kl_loss_dinv(
        self,
        latents,
        y_0,
        i=None,
        t_tensor=None,
        alpha_t=None,
        *,
        generator: torch.Generator | None = None,
        return_logvar: bool = False,
    ):
        batch_size = latents.size(0)
        decoder_std = self.decoder_std
        # kl_latino_style = bool(self.cfg.algo.get("kl_latino_style", False))
        kl_use_mean = self.cfg.algo.get("kl_use_mean", False)
        kl_include_logvar = self.cfg.algo.get("kl_include_logvar", True)
        var_x_zt = 1.0
        with torch.no_grad():
            x_pred_z = self.model.decode_latents(latents, stay_on_device=True)
            posterior_x0_mu = self.H.prox_l2(
                z=x_pred_z.float(),
                y=y_0.float(),
                # gamma=(decoder_std * var_x_zt) / (self.cfg.exp.sigma_x0**2),
                gamma=(decoder_std**2 * var_x_zt) / (self.cfg.exp.sigma_x0**2),
            )
            encoded_out = self.model.encode_images(
                posterior_x0_mu.clip(-1, 1),
                generator=generator,
                use_mean=kl_use_mean,
                return_logvar=kl_include_logvar,
                enable_grad=False,
            )
            if kl_include_logvar:
                encoded, logvar = encoded_out
            else:
                encoded = encoded_out
                logvar = None
            decoded_for_vis = posterior_x0_mu.clip(-1, 1)

        recon2 = torch.linalg.norm((encoded - latents).view(batch_size, -1), dim=-1) ** 2
        if kl_include_logvar:
            z_norm2 = torch.linalg.norm(
                (torch.exp(0.5 * logvar).detach() * latents).view(batch_size, -1),
                dim=-1,
            ) ** 2
        else:
            z_norm2 = torch.zeros_like(recon2)
        kl_loss = (recon2 - z_norm2).sum()

        if return_logvar:
            return kl_loss, x_pred_z, decoded_for_vis, encoded, logvar
        return kl_loss, x_pred_z, decoded_for_vis, encoded

    def _kl_loss_dinv_decoder(
        self,
        latents,
        y_0,
        i=None,
        t_tensor=None,
        alpha_t=None,
        *,
        generator: torch.Generator | None = None,
        return_logvar: bool = False,
    ):
        """
        Return the unscaled decoder loss used by the DeepInv ablation.
        """
        decoder_std = self.decoder_std
        var_x_zt = 1.0
        x_pred_z = self.model.decode_latents(
            latents,
            stay_on_device=True,
        )
        with torch.no_grad():
            posterior_x0_mu = self.H.prox_l2(
                z=x_pred_z.detach().float(),
                y=y_0.float(),
                gamma=(decoder_std**2 * var_x_zt)
                / (self.cfg.exp.sigma_x0**2),
            )
            decoded_for_vis = posterior_x0_mu.clip(-1, 1)

        decoder_loss = (
            (x_pred_z.float() - decoded_for_vis.float()).pow(2).sum()
        )
        if return_logvar:
            return (
                decoder_loss,
                x_pred_z,
                decoded_for_vis,
                None,
                None,
            )
        return decoder_loss, x_pred_z, decoded_for_vis, None

    def get_noise_pred(
        self, z_t, i: int, text_embeds, extra_sd_kwargs, *args, **kwargs
    ):
        """Gets noise prediction from pretrained models."""
        if self.use_dmd2:
            return self._get_noise_pred_sdxl(
                z_t,
                i,
                text_embeds,
                extra_sd_kwargs,
                *args,
                **kwargs,
            )
        else:
            return self._get_noise_pred_sd15(
                z_t,
                i,
                text_embeds,
                extra_sd_kwargs,
                *args,
                **kwargs,
            )

    def _get_noise_pred_sdxl(
        self, z_t, i: int, text_embeds, extra_sd_kwargs, *args, **kwargs
    ):
        add_time_ids = extra_sd_kwargs["add_time_ids"]
        pooled_text_embeds = extra_sd_kwargs["pooled_text_embeds"]
        neg_pooled_text_embeds = extra_sd_kwargs["neg_pooled_text_embeds"]
        neg_text_embeds = extra_sd_kwargs["neg_text_embeds"]
        use_cfg = self.sdxl_guidance_mode == "cfg"

        batch_size = z_t.size(0)
        if pooled_text_embeds is None or add_time_ids is None:
            raise ValueError("SDXL noise prediction requires pooled_text_embeds and add_time_ids.")
        pooled_text_embeds = repeat_to_batch(pooled_text_embeds, batch_size)
        add_time_ids = repeat_to_batch(add_time_ids, batch_size)

        added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
        if use_cfg:
            if neg_text_embeds is None or neg_pooled_text_embeds is None:
                raise ValueError(
                    "Classifier-free guidance requires neg_text_embeds and neg_pooled_text_embeds."
                )
            neg_pooled_text_embeds = repeat_to_batch(neg_pooled_text_embeds, batch_size)
            added_cond_kwargs = {
                "text_embeds": torch.cat([neg_pooled_text_embeds, pooled_text_embeds], dim=0),
                "time_ids": torch.cat([add_time_ids, add_time_ids], dim=0),
            }

        with torch.set_grad_enabled(self.prompt_opt):
            et = hf_get_noise_pred(
                self.model,
                z_t.detach(),
                i,
                batch_size,
                self.device,
                prompt_embeds=text_embeds,
                neg_embeds=neg_text_embeds,
                guidance_scale=self.cfg_scale,
                do_classifier_free_guidance=use_cfg,
                added_cond_kwargs=added_cond_kwargs,
            )
        return et

    def _get_noise_pred_sd15(
        self, z_t, i: int, text_embeds, extra_sd_kwargs, *args, **kwargs
    ):
        uncond_embed = extra_sd_kwargs["uncond_embed"]
        batch_size = z_t.size(0)
        with torch.set_grad_enabled(self.prompt_opt):
            return hf_get_noise_pred(
                self.model,
                z_t.detach(),
                i,
                batch_size,
                self.device,
                prompt_embeds=text_embeds,
                neg_embeds=uncond_embed,
                guidance_scale=self.cfg_scale,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

    @torch.no_grad()
    def prompt_check(
        self,
        generator,
        uncond_embed,
        text_embeds,
        postfix,
        y=None,
        *,
        extra_sd_kwargs=None,
    ):
        # Reset scheduler state for repeated prompt-check runs.
        self._reset_scheduler_state()

        image_h = int(self.cfg.model.get("output_h", 1024 if self.use_dmd2 else 512))
        image_w = int(self.cfg.model.get("output_w", image_h))
        if y is None:
            latents_test = self.model.prepare_latents(
                batch_size=1,  # number of images to generate
                num_channels_latents=self.model.unet.config.in_channels,  # latent channels
                height=image_h,  # image height
                width=image_w,  # image width
                dtype=self.model_dtype,  # datatype
                device=self.device,
                generator=generator,
            )
        else:
            # add noise to time T
            pinv_y = self.H.A_dagger(y.float()).reshape(-1, 3, image_h, image_w)
            pinv_y = pinv_y.clamp(-1, 1)
            pinv_y = pinv_y.to(device=self.device, dtype=self.model_dtype)
            encoded_pinv = self.model.encode_images(pinv_y, generator=generator)
            noise_T = torch.randn(
                encoded_pinv.shape,
                device=self.device,
                generator=generator,
                dtype=encoded_pinv.dtype,
            )
            t0_int = int(self.model.scheduler.timesteps[0].item())
            t0_batch = torch.full(
                (encoded_pinv.shape[0],),
                t0_int,
                device=self.device,
                dtype=torch.long,
            )
            latents_test = self.model.scheduler.add_noise(
                encoded_pinv, noise_T, t0_batch
            )
            print(f"Prompt check with measurement {latents_test.shape}")
        text_embeds_check = text_embeds[:1].detach().to(
            device=self.device, dtype=self.model_dtype
        )
        prompt_sd_kwargs = dict(extra_sd_kwargs or {})
        if prompt_sd_kwargs.get("uncond_embed", None) is None:
            prompt_sd_kwargs["uncond_embed"] = uncond_embed

        timesteps = self.model.scheduler.timesteps
        for k, timestep in enumerate(timesteps):
            print(
                f"Prior sampling timestep {k+1}/{len(timesteps)}: {timestep.item()}"
            )
            et = self.get_noise_pred(
                latents_test,
                i=k,
                text_embeds=text_embeds_check,
                extra_sd_kwargs=prompt_sd_kwargs,
            )
            latents_test = self.model.scheduler.step(
                et, timestep, latents_test, return_dict=False
            )[0]

        # Decode latents to image
        decoded_image = self.model.decode_latents(latents_test, stay_on_device=True).clip(
        -1, 1
        )
        img_id = getattr(self, "current_img_id", None)
        if img_id is None:
            img_id = self.cfg.exp.img_id.split(".")[0]
        output_dir = os.path.join(self.cfg.exp.root, self.cfg.exp.output_path, self.cfg.algo.name_folder, self.cfg.exp.deg)
        output_path_img = f"{output_dir}/{img_id}"
        ensure_dir(output_path_img)
        save_image(
            self.postprocess(decoded_image),
            os.path.join(output_path_img, f"{img_id}_prior_{postfix}.png"),
        )
        if self.cfg.exp.use_wandb:
            wandb.log({f"prior_image": wandb.Image(os.path.join(output_path_img, f"{img_id}_prior_{postfix}.png"),
                                                   caption=f"prior of {img_id}")})

    def _parse_config_list(self, cfg_key: str, default_value: float, num_steps: int):
        raw = self.cfg.algo.get(cfg_key, None)
        if raw is None:
            return [default_value] * num_steps
        if isinstance(raw, (int, float, bool)):
            return [float(raw)] * num_steps
        if isinstance(raw, list):
            return raw
        if isinstance(raw, omegaconf.listconfig.ListConfig):
            return OmegaConf.to_object(raw)
        parsed = literal_eval(
            raw.replace("true", "True").replace("false", "False")
        )
        if isinstance(parsed, (int, float)):
            return [float(parsed)] * num_steps
        return list(parsed)

    def _expand_stepwise_weights(self, weights, name: str):
        # expand 4 weights to 8 if needed
        if len(weights) == self.num_diffusion_steps // 2:
            weights = np.repeat(weights, 2)
            print(f"Expanded {name} to match num_diffusion_steps: {weights}")
        elif len(weights) != self.num_diffusion_steps:
            raise ValueError(
                f"Length of {name} {len(weights)} does not match num_diffusion_steps "
                f"{self.num_diffusion_steps} or half of it."
            )
        return weights

    def _build_inv_snr_integral(self, stepwise_weights, t_sd):
        # return w_t*inv_snr and its time integral
        timestep_idx = self.model.scheduler.timesteps.detach().cpu().long().numpy()
        interp = torch.from_numpy(
            np.interp(
                x=t_sd,
                xp=list(reversed(t_sd[timestep_idx])),
                fp=list(reversed(stepwise_weights)),
            )
        ).to(device=self.device, dtype=self.model.dtype)
        interp *= self.snr_inv
        anti = torch.from_numpy(
            np.cumsum(interp.detach().cpu().numpy())
        ).to(device=self.device, dtype=self.model.dtype)
        return interp, anti
