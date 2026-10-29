from typing import Union
from pathlib import Path

import torch
from models.sd_inverse import SDInversePipeline, SDInversePipelineDiff
from huggingface_hub import hf_hub_download
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DiffusionPipeline,
    LCMScheduler,
    DDIMScheduler,
    AutoPipelineForText2Image,
)


SDXL_MODEL_NAMES = {"dmd2", "sdxllora"}


def is_sdxl_model_name(model_name: str) -> bool:
    return str(model_name) in SDXL_MODEL_NAMES


def resolve_repo_id_or_path(repo_id: str, sd_local_root: str | None) -> str:
    """Return local snapshot path if `sd_local_root` is set, else the HF repo id.

    Local convention (created by `scripts/prefetch_sd_weights.py`):
      ${sd_local_root}/local--<repo_id with / replaced by -->
    """
    if sd_local_root is None or str(sd_local_root).strip() == "":
        return repo_id
    safe = repo_id.replace("/", "--")
    return str(Path(sd_local_root) / f"local--{safe}")


def _set_pipe_scheduler(pipe, scheduler_name: str, num_steps, device):
    if scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "lcm":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name != "default":
        raise ValueError(
            f"Unknown scheduler_name={scheduler_name!r} "
            "(expected 'default', 'ddim', or 'lcm')."
        )
    pipe.scheduler.set_timesteps(num_steps, device=device)
    return pipe, pipe.scheduler.timesteps


def _load_sd_inverse_pipeline(model_id: str, fp16: bool):
    torch_dtype = torch.float16 if fp16 else torch.float32
    if not fp16:
        return SDInversePipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    try:
        return SDInversePipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant="fp16",
        )
    except (OSError, ValueError):
        return SDInversePipeline.from_pretrained(model_id, torch_dtype=torch_dtype)


def init_sd21(
    num_steps,
    sd_local_root,
    device,
    scheduler_name: str = "ddim",
    fp16: bool = False,
):
    model_id = resolve_repo_id_or_path(
        "Manojb/stable-diffusion-2-1-base", sd_local_root
    )
    pipe = _load_sd_inverse_pipeline(model_id, fp16=fp16)
    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return _set_pipe_scheduler(pipe, scheduler_name, num_steps, device)

def init_sd15(
    num_steps,
    sd_local_root,
    device,
    scheduler_name: str = "default",
    fp16: bool = False,
):
    model_id = resolve_repo_id_or_path(
        "runwayml/stable-diffusion-v1-5", sd_local_root
    )
    pipe = _load_sd_inverse_pipeline(model_id, fp16=fp16)
    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return _set_pipe_scheduler(pipe, scheduler_name, num_steps, device)


def init_dmd15(sd_local_root, device, cfg):
    fp16 = cfg.model.use_fp16
    model_id = resolve_repo_id_or_path("runwayml/stable-diffusion-v1-5", sd_local_root)
    adapter_id = resolve_repo_id_or_path(
        "latent-consistency/lcm-lora-sdv1-5", sd_local_root
    )
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        variant="fp16" if fp16 else None,
    )
    lcm_scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    lcm_scheduler.set_timesteps(cfg.algo.num_diffusion_steps)
    pipe.scheduler = lcm_scheduler
    pipe.to(device)
    # explicit weight_name: diffusers cannot infer it with HF_HUB_OFFLINE=1, even from a local folder
    pipe.load_lora_weights(adapter_id, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora()
    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    pipe = SDInversePipeline.from_pretrained(
        model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=lcm_scheduler,
        torch_dtype=unet.dtype,
        variant="fp16" if fp16 else None,
    )
    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe, pipe.scheduler.timesteps


def init_sdxllora(sd_local_root, device, cfg):
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    unet_id = "latent-consistency/lcm-sdxl"
    fp16 = bool(cfg.model.get("use_fp16", False))
    torch_dtype = torch.float16 if fp16 else torch.float32

    base_model = resolve_repo_id_or_path(base_model_id, sd_local_root)
    unet_model = resolve_repo_id_or_path(unet_id, sd_local_root)
    unet = UNet2DConditionModel.from_pretrained(
        unet_model,
        torch_dtype=torch_dtype,
        variant="fp16" if fp16 else None,
    ).to(device)

    if fp16:
        vae_id = "madebyollin/sdxl-vae-fp16-fix"
    else:
        vae_id = "stabilityai/sdxl-vae"
    vae_ckpt = resolve_repo_id_or_path(vae_id, sd_local_root)
    vae = AutoencoderKL.from_pretrained(vae_ckpt, torch_dtype=torch_dtype).to(device)

    pipe = SDInversePipelineDiff.from_pretrained(
        base_model,
        unet=unet,
        vae=vae,
        torch_dtype=torch_dtype,
        variant="fp16" if fp16 else None,
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(cfg.algo.num_diffusion_steps, device=device)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()
    return pipe, pipe.scheduler.timesteps


def init_dmd2(sd_local_root, device, cfg):
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    base_model = resolve_repo_id_or_path(base_model_id, sd_local_root)
    repo_name = "tianweiy/DMD2"
    fp16 = cfg.model.use_fp16
    if fp16:
        ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
    else:
        ckpt_name = "dmd2_sdxl_4step_unet.bin"

    if sd_local_root is None:
        unet_ckpt = hf_hub_download(repo_name, ckpt_name)
    else:
        local_repo = resolve_repo_id_or_path(repo_name, sd_local_root)
        unet_ckpt = str(Path(local_repo) / ckpt_name)

    # Load model.
    unet_config = UNet2DConditionModel.load_config(base_model, subfolder="unet")
    # unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet = UNet2DConditionModel.from_config(unet_config).to(
        device, torch.float16 if fp16 else torch.float32
    )
    unet.load_state_dict(
        torch.load(
            unet_ckpt,
            map_location=device,
            weights_only=True,
        )
    )
    if fp16:
        vae_id = "madebyollin/sdxl-vae-fp16-fix"
    else:
        vae_id = "stabilityai/sdxl-vae"
    vae_ckpt = resolve_repo_id_or_path(vae_id, sd_local_root)
    vae = AutoencoderKL.from_pretrained(vae_ckpt, torch_dtype=unet.dtype).to(device)
    pipe = SDInversePipelineDiff.from_pretrained(
        base_model,
        unet=unet,
        vae=vae,
        torch_dtype=unet.dtype,
        variant="fp16" if fp16 else None,
        guidance_scale=0,
    ).to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # print(pipe.scheduler.timesteps)
    # print(pipe.scheduler.timesteps.device)
    sched_device = pipe.scheduler.timesteps.device
    num_inference_steps = cfg.model.num_diffusion_steps  # or 4 for DMD2
    pipe.scheduler.set_timesteps(num_inference_steps, device=sched_device)

    # Override the scheduler's timesteps with DMD2 values
    if num_inference_steps == 4:
        custom_timesteps = torch.tensor(
            [999, 749, 499, 249], device=sched_device, dtype=torch.long
        )
    else:
        custom_timesteps = torch.tensor(
            [999, 874, 749, 624, 499, 374, 249, 124],
            device=sched_device,
            dtype=torch.long,
        )
    pipe.scheduler.timesteps = custom_timesteps
    return pipe, pipe.scheduler.timesteps


def repeat_to_batch(embeds: torch.Tensor, target_batch: int) -> torch.Tensor:
    if embeds.shape[0] == target_batch:
        return embeds
    if embeds.shape[0] > target_batch:
        return embeds[:target_batch]
    repeat_factor = (target_batch + embeds.shape[0] - 1) // embeds.shape[0]
    return embeds.repeat_interleave(repeat_factor, dim=0)[:target_batch]


def preprocess_unet_input(
    pipe,
    latents,
    i: torch.Tensor,
    batch_size: int,
    device,
    do_classifier_free_guidance: bool,
    prompt_embeds: torch.Tensor,
    neg_embeds: torch.Tensor,
    **kwargs,
):
    t = pipe.scheduler.timesteps[i]
    t_size = t.size(0) if t.ndim > 0 else 1
    if (not torch.is_tensor(t)) or t_size==1:
        t_idx = torch.full((batch_size,), int(t.item()), device=device, dtype=torch.long)
    else:
        t_idx = t.unsqueeze(1).repeat(1, batch_size//t_size).to(device, dtype=torch.long).view(-1)

    # Expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    t_unet = torch.cat([t_idx, t_idx], dim=0) if do_classifier_free_guidance else t_idx
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t_unet)

    # Predict the noise residual
    prompt_embeds = prompt_embeds.to(device=device, dtype=latents.dtype)
    if do_classifier_free_guidance:
        neg_embeds = neg_embeds.to(device=device, dtype=latents.dtype)
        prompt_embeds_b = repeat_to_batch(prompt_embeds, batch_size)
        neg_embeds_b = repeat_to_batch(neg_embeds, batch_size)
        emb_in = torch.cat([neg_embeds_b, prompt_embeds_b], dim=0)
    else:
        emb_in = repeat_to_batch(prompt_embeds, batch_size)

    unet_kwargs = {}
    if "added_cond_kwargs" in kwargs:
        unet_kwargs["added_cond_kwargs"] = kwargs["added_cond_kwargs"]
    if "cross_attention_kwargs" in kwargs:
        unet_kwargs["cross_attention_kwargs"] = kwargs["cross_attention_kwargs"]

    return latent_model_input, t_unet, emb_in, unet_kwargs

def get_noise_pred(
    pipe,
    latents,
    i: Union[int, torch.Tensor],
    batch_size: int,
    device,
    prompt_embeds: torch.Tensor,
    neg_embeds: torch.Tensor,
    guidance_scale: float = 1.2,
    do_classifier_free_guidance: bool = False,
    **kwargs,
):
    latent_model_input, t_unet, emb_in, unet_kwargs = preprocess_unet_input(
                pipe,
                latents,
                i,
                batch_size,
                device,
                do_classifier_free_guidance,
                prompt_embeds,
                neg_embeds,
                **kwargs,
            )
    noise_pred = pipe.unet(
        latent_model_input,
        t_unet,
        encoder_hidden_states=emb_in,
        **unet_kwargs,
    ).sample

    # Perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    return noise_pred
