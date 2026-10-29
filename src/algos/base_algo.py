# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

from __future__ import annotations

import torch

from utils.degradations_dinv import get_degradation_image
from utils.img_utils import postprocess as utils_postprocess

class AlgoBase:
    def _reset_scheduler_state(self) -> None:
        model = getattr(self, "model", None)
        scheduler = getattr(model, "scheduler", None) if model is not None else None
        if scheduler is None:
            return

        if hasattr(scheduler, "_step_index"):
            scheduler._step_index = None
        if hasattr(scheduler, "_begin_index"):
            scheduler._begin_index = None
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)

        # PNDM-style schedulers keep running buffers across `.step()` calls.
        if hasattr(scheduler, "counter"):
            scheduler.counter = 0
        if hasattr(scheduler, "cur_model_output"):
            scheduler.cur_model_output = 0
        if hasattr(scheduler, "cur_sample"):
            scheduler.cur_sample = None
        if hasattr(scheduler, "ets"):
            scheduler.ets = []

    @torch.no_grad()
    def postprocess(self, x, cfg=None):
        if cfg is None:
            cfg = getattr(self, "cfg", None)

        if cfg is None:
            if isinstance(x, list):
                return [(v + 1) / 2 for v in x]
            return (x + 1) / 2

        return utils_postprocess(x, cfg)

    def _algo_name(self) -> str | None:
        cfg = getattr(self, "cfg", None)
        algo_cfg = getattr(cfg, "algo", None)
        name = getattr(algo_cfg, "name", None)
        if name is None:
            return None
        return str(name)

    def _is_latent_algo(self) -> bool:
        #TODO: will add more algos
        name = self._algo_name()
        if name is None:
            return False
        name = name.lower()
        return (
            ("cwgf" in name)
            or ("ldps" in name)
            or ("psld" in name)
            or ("treg" in name)
            or ("latino" in name)
        )

    def _infer_latent_dtype(self, prompt_embeds=None) -> torch.dtype:
        dtype = getattr(self, "model_dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype

        if prompt_embeds is not None:
            prompt_dtype = getattr(prompt_embeds, "dtype", None)
            if isinstance(prompt_dtype, torch.dtype):
                return prompt_dtype

        model = getattr(self, "model", None)
        model_dtype = getattr(model, "dtype", None)
        if isinstance(model_dtype, torch.dtype):
            return model_dtype

        unet = getattr(model, "unet", None) if model is not None else None
        unet_dtype = getattr(unet, "dtype", None)
        if isinstance(unet_dtype, torch.dtype):
            return unet_dtype

        return torch.float32

    def init_latents(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None,
        y_0: torch.Tensor,
        init_method: str | None = "none",
        prompt_embeds=None,
        neg_embeds=None,
        **unusedkwargs,
    ) -> torch.Tensor:
        """
        Inits latents for sampling.
        """
        if init_method is None:
            init_method = "none"
        init_method = str(init_method).lower()

        cfg = self.cfg
        device = getattr(self, "device", x.device)
        dtype = self._infer_latent_dtype(prompt_embeds=prompt_embeds)

        batch_size = 1 if x.dim() == 3 else x.size(0)

        if init_method == "none":
            return torch.randn(
                (batch_size, int(cfg.model.C), int(cfg.model.H), int(cfg.model.W)),
                device=device,
                dtype=dtype,
                generator=generator,
            )

        

        H = getattr(self, "H", None)
        pinv_y = get_degradation_image(y_0, H, cfg)

        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "encode_images"):
            raise AttributeError

        encoded = model.encode_images(pinv_y.to(device, dtype=dtype), generator=generator)

        if init_method == "pinv":
            #NOTE: ablate here using true x
            # return model.encode_images(x.to(device, dtype=dtype), generator=generator)
            return encoded

        if init_method != "ddim":
            raise ValueError(f"Unknown init_method {init_method!r}; expected 'none', 'pinv', or 'ddim'.")

        ddim_invert = getattr(self, "ddim_invert", None)
        if ddim_invert is None:
            raise NotImplementedError("DDIM inversion not implemented for this algo.")

        from diffusers import StableDiffusionPipeline
        from utils.hf_utils import resolve_repo_id_or_path

        inv_pipe = StableDiffusionPipeline.from_pretrained(
            resolve_repo_id_or_path(
                "runwayml/stable-diffusion-v1-5", cfg.exp.get("sd_local_root", None)
            ),
            torch_dtype=dtype,
        ).to(device)

        algo_cfg = cfg.algo
        return ddim_invert(
            encoded,
            pipe=inv_pipe,
            prompt_embeds=prompt_embeds,
            neg_embeds=neg_embeds,
            save_decoded=algo_cfg.get("save_invert_decoded", False),
            guidance_scale=algo_cfg.get("cfg_scale", 1.2),
            invert_to_t=algo_cfg.get("invert_to", 50),
            num_inference_steps=algo_cfg.get("ddim_steps", 80),
            do_classifier_free_guidance=False,
        )

    @torch.no_grad()
    def save_evo(self, counter, x, latents, input_img, x_list, results256: bool = False):
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            raise AttributeError("`save_evo` requires `self.cfg` to be set.")

        if not cfg.exp.get("load_img_id", False):
            return

        if (not (counter % cfg.exp.save_every == 0)) and (
            not (counter == cfg.exp.num_steps - 1)
        ):
            return

        print(f"Saving evolution at step {counter}")

        from torchvision.utils import make_grid, save_image
        import torch.nn as nn

        if x.dim() == 3:
            x = x.unsqueeze(0)
        if input_img.dim() == 3:
            input_img = input_img.unsqueeze(0)

        if results256 is True:
            downsample = nn.AvgPool2d(2, 2)
            aux = torch.zeros((cfg.algo.n_particles, 3, 256, 256))
            x_256 = torch.zeros((cfg.algo.n_particles, 3, 256, 256))
            xx = self.postprocess(x).clone().detach().cpu()
            for q in range(cfg.algo.n_particles):
                aux[q, :, :, :] = downsample(xx[q, :, :, :].unsqueeze(0)).squeeze()
                x_256[q, :, :, :] = downsample(
                    self.postprocess(input_img[q, :, :, :].unsqueeze(0))
                    .clone()
                    .detach()
                    .cpu()
                ).squeeze()
            xo = aux
        else:
            xo = self.postprocess(x).clone().detach().cpu()
            x_256 = self.postprocess(input_img).cpu()

        x_list.append(x.clone().detach())

        # PSNR
        mse = torch.mean((xo - x_256.cpu()) ** 2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(1 / (mse + 1e-10))
        print(f"Mean PSNR: {psnr.mean()}")
        print(f"PSNR: {psnr}")

        # LPIPS
        lpips_model = getattr(self, "lpips", None)
        if lpips_model is None:
            raise AttributeError(
                "`save_evo` requires `self.lpips` to be set (e.g. lpips.LPIPS(...))."
            )
        LPIPS = lpips_model(xo.cuda(), x_256.cuda(), normalize=True)
        print(f"Mean LPIPS: {LPIPS.mean()}")
        print(f"LPIPS: {LPIPS[:,0,0,0]}")

        image_evol = make_grid((self.postprocess(x).clone().detach().cpu()))
        save_image(image_evol, f"{self.evol_path}/evol_{counter}.png")

        image_z = self.model.decode_latents(latents, stay_on_device=True).detach().clip(
            -1, 1
        )
        image_evol = make_grid((self.postprocess(image_z).clone().detach().cpu()))
        save_image(image_evol, f"{self.evol_path}/evol_{counter}_z.png")

        return LPIPS[:, 0, 0, 0], psnr
