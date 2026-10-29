#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


# Stable Diffusion repos: only the diffusers components the pipelines load (fp32 and fp16
# safetensors), not the single-file checkpoints, .bin duplicates or non-EMA UNet (~8 GB vs ~47 GB).
SD_PIPELINE_PATTERNS = [
    "model_index.json",
    "feature_extractor/*",
    "scheduler/*",
    "tokenizer/*",
    "*/config.json",
    "safety_checker/model*.safetensors",
    "text_encoder/model*.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model*.safetensors",
]

DMD2_UNET_PATTERNS = [
    "dmd2_sdxl_4step_unet.bin",
    "dmd2_sdxl_4step_unet_fp16.bin",
]

SDXL_BASE_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/config.json",
    "text_encoder/model*.safetensors",
    "text_encoder/model*.safetensors.index.json",
    "text_encoder/pytorch_model*.bin",
    "text_encoder/pytorch_model*.bin.index.json",
    "text_encoder_2/config.json",
    "text_encoder_2/model*.safetensors",
    "text_encoder_2/model*.safetensors.index.json",
    "text_encoder_2/pytorch_model*.bin",
    "text_encoder_2/pytorch_model*.bin.index.json",
    "tokenizer/*",
    "tokenizer_2/*",
    "unet/config.json",
]

SDXL_UNET_PATTERNS = [
    "config.json",
    "diffusion_pytorch_model*.safetensors",
    "diffusion_pytorch_model*.safetensors.index.json",
    "diffusion_pytorch_model*.bin",
    "diffusion_pytorch_model*.bin.index.json",
]

SDXL_VAE_PATTERNS = [
    "config.json",
    "diffusion_pytorch_model*.safetensors",
    "diffusion_pytorch_model*.safetensors.index.json",
    "diffusion_pytorch_model*.bin",
    "diffusion_pytorch_model*.bin.index.json",
]


def _local_dirname(repo_id: str) -> str:
    return f"local--{repo_id.replace('/', '--')}"


def _download_repo(
    repo_id: str,
    out_root: Path,
    *,
    token: str | None,
    allow_patterns: list[str] | None = None,
) -> Path:
    target = out_root / _local_dirname(repo_id)
    target.mkdir(parents=True, exist_ok=True)

    scope = f" ({len(allow_patterns)} include patterns)" if allow_patterns else ""
    print(f"[prefetch] {repo_id} -> {target}{scope}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        token=token,
        allow_patterns=allow_patterns,
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefetch Stable Diffusion / SDXL weights to local directories (cluster-friendly)."
    )
    parser.add_argument(
        "--out",
        default="hf_models",
        help="Output directory to store repos (default: hf_models).",
    )
    parser.add_argument(
        "--sd15",
        default="runwayml/stable-diffusion-v1-5",
        help="SD1.5 repo id (used by cwgf_dmd_s).",
    )
    parser.add_argument(
        "--sd21",
        default="Manojb/stable-diffusion-2-1-base",
        help="SD2.1 repo id (used by model=sd21, only downloaded with --with-sd21).",
    )
    parser.add_argument(
        "--lcm-lora",
        default="latent-consistency/lcm-lora-sdv1-5",
        help="LCM LoRA repo id (used by cwgf_dmd_s).",
    )
    parser.add_argument(
        "--with-dmd2",
        action="store_true",
        help=(
            "Also prefetch the files needed by model=dmd2, without mirroring "
            "the full DMD2/SDXL repos."
        ),
    )
    parser.add_argument(
        "--with-sdxl-lora",
        action="store_true",
        help=(
            "Also prefetch the files needed by model=sdxllora, without "
            "mirroring the full SDXL repos."
        ),
    )
    parser.add_argument(
        "--sdxl-base",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL base repo id (used by DMD2 / SDXL LoRA).",
    )
    parser.add_argument(
        "--sdxl-lora",
        default="latent-consistency/lcm-sdxl",
        help="SDXL LCM UNet repo id (used by model=sdxllora).",
    )
    parser.add_argument(
        "--dmd2-repo",
        default="tianweiy/DMD2",
        help="DMD2 repo id (contains UNet checkpoints).",
    )
    parser.add_argument(
        "--sdxl-vae",
        default="stabilityai/sdxl-vae",
        help="SDXL VAE repo id (used by DMD2 in fp32).",
    )
    parser.add_argument(
        "--sdxl-vae-fp16",
        default="madebyollin/sdxl-vae-fp16-fix",
        help="SDXL VAE repo id (used by DMD2 in fp16).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN / HUGGINGFACE_TOKEN).",
    )
    parser.add_argument("--with-sd21", action="store_true", help="Also prefetch SD2.1 (model=sd21).")
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    token = (
        args.token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

    _download_repo(args.sd15, out_root, token=token, allow_patterns=SD_PIPELINE_PATTERNS)
    if args.with_sd21:
        _download_repo(args.sd21, out_root, token=token, allow_patterns=SD_PIPELINE_PATTERNS)
    if not args.skip_lora:
        _download_repo(args.lcm_lora, out_root, token=token)
    if args.with_dmd2 or args.with_sdxl_lora:
        _download_repo(
            args.sdxl_base,
            out_root,
            token=token,
            allow_patterns=SDXL_BASE_PATTERNS,
        )
        _download_repo(
            args.sdxl_vae,
            out_root,
            token=token,
            allow_patterns=SDXL_VAE_PATTERNS,
        )
        _download_repo(
            args.sdxl_vae_fp16,
            out_root,
            token=token,
            allow_patterns=SDXL_VAE_PATTERNS,
        )
    if args.with_dmd2:
        _download_repo(
            args.dmd2_repo,
            out_root,
            token=token,
            allow_patterns=DMD2_UNET_PATTERNS,
        )
    if args.with_sdxl_lora:
        _download_repo(
            args.sdxl_lora,
            out_root,
            token=token,
            allow_patterns=SDXL_UNET_PATTERNS,
        )

    print("\n[prefetch] Done. Use these in your job scripts:")
    print(f"  export CWGF_SD_LOCAL_ROOT='{out_root}'")
    print("\nOr pass directly via Hydra:")
    print(f"  exp.sd_local_root='{out_root}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
