import torch
from contextlib import nullcontext

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

class SDInversePipeline(StableDiffusionPipeline):
    def decode_latents(self, latents, stay_on_device=False, train=False):
        latents = 1 / self.vae.config.scaling_factor * latents
        if train:
            # set self.vae to require grad
            self.vae.train()
        for param in self.vae.parameters():
            param.requires_grad = True
        image = self.vae.decode(latents).sample.clip(-1, 1)
        if not stay_on_device:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def encode_images(
        self,
        images,
        *,
        generator: torch.Generator | None = None,
        use_mean: bool = False,
        enable_grad: bool = False,
        return_logvar: bool = False,
    ):
        if self.dtype == torch.float16:
            images = images.half()
        grad_ctx = nullcontext() if enable_grad else torch.no_grad()
        with grad_ctx:
            latent_dist = self.vae.encode(images).latent_dist
            if use_mean:
                latents = latent_dist.mean
            else:
                try:
                    latents = latent_dist.sample(generator=generator)
                except TypeError:
                    latents = latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            logvar = latent_dist.logvar
            # do not scale and treat scaling as part of the decoder
        out = (latents,logvar) if return_logvar else latents
        return out

class SDInversePipelineDiff(StableDiffusionXLPipeline):
    def decode_latents(self, latents, stay_on_device: bool = False, train: bool = False):
        latents = (1 / self.vae.config.scaling_factor) * latents

        if train:
            self.vae.train()
            for p in self.vae.parameters():
                p.requires_grad = True
        else:
            # keep your original behavior if you want grads even in eval:
            for p in self.vae.parameters():
                p.requires_grad = True

        image = self.vae.decode(latents).sample.clip(-1, 1)
        if not stay_on_device:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()

        return image

    def encode_images(
        self,
        images,
        *,
        generator: torch.Generator | None = None,
        use_mean: bool = False,
        enable_grad: bool = False,
        return_logvar: bool = False,
    ):
        grad_ctx = nullcontext() if enable_grad else torch.no_grad()
        with grad_ctx:
            if self.dtype == torch.float16:
                images = images.half()
            latent_dist = self.vae.encode(images).latent_dist

            if use_mean:
                latents = latent_dist.mean
            else:
                try:
                    latents = latent_dist.sample(generator=generator)
                except TypeError:
                    latents = latent_dist.sample()

            latents = latents * self.vae.config.scaling_factor
            logvar = latent_dist.logvar
        out = (latents, logvar) if return_logvar else latents
        return out
