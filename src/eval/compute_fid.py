import os
import logging

import lpips
import torch
from PIL import Image
from torchvision import transforms

from custom_datasets import build_loader
from utils.img_utils import (
    postprocess,
    preprocess,
    resize_input,
    resize_to_1024_for_dmd2,
)
from utils.hf_utils import is_sdxl_model_name
from utils.save import _get_save_id

from .torchmetrics_fid import compute_fid as compute_fid_torchmetrics
from .torchmetrics_kid import compute_kid as compute_kid_torchmetrics

logger = logging.getLogger(__name__)


def _crop_center(x: torch.Tensor, crop_size: int = 500) -> torch.Tensor:
    h, w = x.shape[2], x.shape[3]
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Expected images at least {crop_size}x{crop_size}, got {(h, w)}."
        )
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return x[:, :, start_h : start_h + crop_size, start_w : start_w + crop_size]


def compute_psnr(xo: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    xo_cropped = _crop_center(xo)
    x_gt_cropped = _crop_center(x_gt).to(xo_cropped.device)
    mse = torch.mean((xo_cropped - x_gt_cropped) ** 2, dim=(1, 2, 3))
    return 10 * torch.log10(1 / (mse + 1e-10))


def compute_lpips(
    lpips_model,
    xo: torch.Tensor,
    x_gt: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if device is None:
        device = next(lpips_model.parameters()).device
    else:
        device = torch.device(device)
    xo_cropped = _crop_center(xo).to(device)
    x_gt_cropped = _crop_center(x_gt).to(device)
    return lpips_model(xo_cropped, x_gt_cropped, normalize=True)


def _get_device(cfg) -> torch.device:
    requested = str(cfg.exp.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _get_fid_device(cfg) -> torch.device:
    requested = str(cfg.exp.get("fid_device", cfg.exp.get("device", "cuda")))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _get_fid_cpu_before_compute(cfg) -> bool:
    return bool(cfg.exp.get("fid_cpu_before_compute", False))


def _get_kid_device(cfg) -> torch.device:
    requested = str(
        cfg.exp.get(
            "kid_device",
            cfg.exp.get("fid_device", cfg.exp.get("device", "cuda")),
        )
    )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _get_kid_cpu_before_compute(cfg) -> bool:
    return bool(
        cfg.exp.get(
            "kid_cpu_before_compute",
            cfg.exp.get("fid_cpu_before_compute", False),
        )
    )


def _compute_kid_from_tensors(
    cfg,
    real_x: torch.Tensor,
    fake_x: torch.Tensor,
) -> tuple[float, float]:
    kid_seed = cfg.exp.get("kid_seed", cfg.exp.seed)
    return compute_kid_torchmetrics(
        real_x,
        fake_x,
        device=_get_kid_device(cfg),
        batch_size=int(cfg.exp.get("kid_batch_size", 10)),
        subsets=int(cfg.exp.get("kid_subsets", 100)),
        subset_size=int(cfg.exp.get("kid_subset_size", 1000)),
        cpu_before_compute=_get_kid_cpu_before_compute(cfg),
        seed=None if kid_seed is None else int(kid_seed),
    )


def is_cwgf_dmd_dmd2(cfg) -> bool:
    algo_name = str(cfg.algo.name)
    return ("latino" in algo_name or "dmd" in algo_name) and is_sdxl_model_name(
        cfg.model.name
    )


def _prepare_eval_input(cfg, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if is_cwgf_dmd_dmd2(cfg):
        x = x.to(device)
        x = resize_to_1024_for_dmd2(cfg, x)
    else:
        x = resize_input(x)
        x = x.to(device)
    return x


def _prepare_real_for_fid(cfg, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = _prepare_eval_input(cfg, x, device)
    x = preprocess(x)
    x = postprocess(x, cfg)
    return preprocess(x).detach().cpu()


def _prepare_eval_targets(
    cfg, x: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _prepare_eval_input(cfg, x, device)
    x_gt = postprocess(preprocess(x), cfg).detach().cpu()
    real_x = preprocess(x_gt).detach().cpu()
    return real_x, x_gt


def _load_generated_batch(img_paths: list[str]) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    images = []
    for img_path in img_paths:
        with Image.open(img_path) as img:
            images.append(to_tensor(img.convert("RGB")))
    return torch.stack(images, dim=0)


# not used now
@torch.no_grad()
def compute_fid(cfg):
    loader = build_loader(cfg)
    device = _get_device(cfg)
    fid_device = _get_fid_device(cfg)
    fid_cpu_before_compute = _get_fid_cpu_before_compute(cfg)
    print(f"length of loader: {len(loader)}")
    real_x = []
    for i, (x, _, _) in enumerate(loader):
        if cfg.exp.max_num_images > 0 and i >= cfg.exp.max_num_images:
            break
        real_x.append(_prepare_real_for_fid(cfg, x, device))
    if not real_x:
        logger.warning("No real images found, cannot compute FID.")
        return None
    real_x = torch.cat(real_x, dim=0)

    output_img = get_output_img(cfg)
    if output_img is None:
        logger.warning("No output images found, cannot compute FID.")
        return None
    fake_x = []
    for img_path in output_img:
        fake_x.append(preprocess(_load_generated_batch([img_path])))
    fake_x = torch.cat(fake_x, dim=0)
    return compute_fid_torchmetrics(
        real_x,
        fake_x,
        device=fid_device,
        cpu_before_compute=fid_cpu_before_compute,
    )


@torch.no_grad()
def compute_kid(cfg) -> tuple[float, float] | None:
    loader = build_loader(cfg)
    device = _get_device(cfg)
    print(f"length of loader: {len(loader)}")
    real_x = []
    for i, (x, _, _) in enumerate(loader):
        if cfg.exp.max_num_images > 0 and i >= cfg.exp.max_num_images:
            break
        real_x.append(_prepare_real_for_fid(cfg, x, device))
    if not real_x:
        logger.warning("No real images found, cannot compute KID.")
        return None

    output_img = get_output_img(cfg)
    if not output_img:
        logger.warning("No output images found, cannot compute KID.")
        return None

    fake_x = [preprocess(_load_generated_batch([img])) for img in output_img]
    return _compute_kid_from_tensors(
        cfg,
        torch.cat(real_x, dim=0),
        torch.cat(fake_x, dim=0),
    )


@torch.no_grad()
def compute_reconstruction_metrics(cfg, *, use_single: bool | None = None):
    loader = build_loader(cfg)
    device = _get_device(cfg)
    fid_device = _get_fid_device(cfg)
    fid_cpu_before_compute = _get_fid_cpu_before_compute(cfg)
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    if use_single is None:
        use_single = cfg.exp.use_single

    real_x = []
    fake_x = []
    psnr_list = []
    lpips_list = []
    missing_ids = []
    start_idx = int(cfg.exp.get("start_idx", 0) or 0)
    max_num_images = int(cfg.exp.max_num_images)
    processed = 0

    for it, (x, _, info) in enumerate(loader):
        if it < start_idx:
            continue
        if max_num_images > 0 and processed >= max_num_images:
            break
        processed += 1
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(
                "compute_reconstruction_metrics expects loader batch_size=1 "
                "so sample ids map to output folders."
            )

        sample_id = _get_save_id(info)
        img_paths = get_output_img_for_sample(cfg, sample_id, use_single=use_single)
        if not img_paths:
            missing_ids.append(sample_id)
            continue

        xo = _load_generated_batch(img_paths)
        real_x_sample, x_gt = _prepare_eval_targets(cfg, x, device)
        if xo.shape[1:] != x_gt.shape[1:]:
            raise ValueError(
                f"Shape mismatch for sample {sample_id}: "
                f"generated={tuple(xo.shape[1:])} vs gt={tuple(x_gt.shape[1:])}"
            )

        x_gt_expanded = x_gt.repeat(xo.shape[0], 1, 1, 1)
        psnr = compute_psnr(xo, x_gt_expanded)
        lpips_score = compute_lpips(lpips_model, xo, x_gt_expanded, device)
        psnr_list.append(psnr.mean().item())
        lpips_list.append(lpips_score.mean().item())

        real_x.append(real_x_sample)
        fake_x.append(preprocess(xo))

    fid = None
    kid_mean = None
    kid_std = None
    if fake_x:
        real_tensor = torch.cat(real_x, dim=0)
        fake_tensor = torch.cat(fake_x, dim=0)
        fid = compute_fid_torchmetrics(
            real_tensor,
            fake_tensor,
            device=fid_device,
            cpu_before_compute=fid_cpu_before_compute,
        )
        if cfg.exp.get("get_kid", False):
            kid_mean, kid_std = _compute_kid_from_tensors(
                cfg,
                real_tensor,
                fake_tensor,
            )

    mean_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else None
    mean_lpips = sum(lpips_list) / len(lpips_list) if lpips_list else None
    return (
        fid,
        kid_mean,
        kid_std,
        mean_psnr,
        mean_lpips,
        len(psnr_list),
        missing_ids,
    )


compute_lpips_score = compute_lpips


def get_output_dir(cfg) -> str:
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path, cfg.algo.name_folder)
    return os.path.join(output_path, cfg.exp.deg)


def _is_saved_sample_image(filename: str) -> bool:
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".png" or not stem.startswith("x_"):
        return False
    return stem[2:].isdigit()


def _sample_image_sort_key(img_path: str):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return int(stem[2:])


def get_output_img_for_sample(
    cfg,
    sample_id: str,
    *,
    use_single: bool | None = None,
) -> list[str]:
    if use_single is None:
        use_single = cfg.exp.use_single

    sample_dir = os.path.join(get_output_dir(cfg), sample_id)
    if not os.path.isdir(sample_dir):
        return []

    item_list = []
    for img in os.listdir(sample_dir):
        if _is_saved_sample_image(img):
            item_list.append(os.path.join(sample_dir, img))
    item_list.sort(key=_sample_image_sort_key)

    if not item_list:
        return []
    if use_single:
        # use single image for fairness when comparing with single-particle baselines
        selected = item_list[torch.randint(0, len(item_list), (1,)).item()]
        return [selected]
    return item_list


def get_output_img(cfg) -> list[str]:
    output_path = get_output_dir(cfg)
    if not os.path.isdir(output_path):
        logger.warning(f"No output directory found at {output_path}")
        return None

    parent_dir = sorted(os.listdir(output_path))
    if len(parent_dir) == 0:
        logger.warning(f"No output images found in {output_path}")
        return None
    output_img = []
    for item in parent_dir:
        item_list = get_output_img_for_sample(cfg, item)
        output_img.extend(item_list)
    return output_img
