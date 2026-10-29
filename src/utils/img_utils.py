import os
from typing import List, Union

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as tvu
import torch.distributed as dist
import deepinv as dinv

def preprocess(x):
    return x * 2 - 1

def save_imagenet_result(x, y, info, samples_root, suffix=""):
    if len(x.shape) == 3:
        n = 1
    else:
        n = x.size(0)

    for i in range(n):
        # print('info["class_id"][i]', info["class_id"][i])
        class_dir = os.path.join(samples_root, info["class_id"][i])
        # print('class_dir', class_dir)
        os.makedirs(class_dir, exist_ok=True)
    for i in range(n):
        if len(suffix) > 0:
            tvu.save_image(
                x[i],
                os.path.join(
                    samples_root, info["class_id"][i], f'{info["name"][i]}_{suffix}.png'
                ),
            )
        else:
            tvu.save_image(
                x[i],
                os.path.join(
                    samples_root, info["class_id"][i], f'{info["name"][i]}.png'
                ),
            )

    dist.barrier()


def save_ffhq_result(x, y, info, samples_root, suffix=""):
    x_list = [torch.zeros_like(x) for i in range(dist.get_world_size())]
    idx = info["index"]
    idx_list = [torch.zeros_like(idx) for i in range(dist.get_world_size())]
    dist.gather(x, x_list, dst=0)
    dist.gather(idx, idx_list, dst=0)

    if len(suffix) == 0:
        lmdb_path = f"{samples_root}.lmdb"
    else:
        lmdb_path = f"{samples_root}_{suffix}.lmdb"

    lmdb_dir = lmdb_path.split("/")[:-1]
    if len(lmdb_dir) > 0:
        lmdb_dir = "/".join(lmdb_dir)
        os.makedirs(lmdb_dir, exist_ok=True)

    if dist.get_rank() == 0:
        x = torch.cat(x_list, dim=0).permute(0, 2, 3, 1).detach().cpu().numpy()
        idx = torch.cat(idx_list, dim=0).detach().cpu().numpy()
        x = (x * 255.0).astype(np.uint8)
        n = x.shape[0]
        env = lmdb.open(lmdb_path, map_size=int(1e12), readonly=False)
        with env.begin(write=True) as txn:
            for i in range(n):
                xi = x[i].copy()
                txn.put(str(int(idx[i])).encode(), xi)

    dist.barrier()


def save_result(name, x, y, info, samples_root, suffix=""):
    if "ImageNet" in name:
        save_imagenet_result(x, y, info, samples_root, suffix)
    elif "FFHQ" in name:
        save_ffhq_result(x, y, info, samples_root, suffix)


def resize_input(x):
    if x.shape[-1] == 256:
        upsample = nn.Upsample(scale_factor=2, mode="nearest")
        return upsample(x.unsqueeze(0)).squeeze() if x.dim() == 3 else upsample(x)
    if x.shape[-1] != 512 or x.shape[-2] != 512:
        resize = transforms.Resize((512, 512))
        if x.dim() == 3:
            return resize(x)
        return torch.stack([resize(img) for img in x])
    return x


def _get_dataset_image_size(cfg) -> int | None:
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        return None
    image_size = getattr(dataset_cfg, "image_size", None)
    if image_size is None:
        return None
    try:
        return int(image_size)
    except (TypeError, ValueError):
        return None


def _should_downsample_model_output(cfg, true_img: bool) -> bool:
    if cfg is None or true_img:
        return False

    model_cfg = getattr(cfg, "model", None)
    model_output_h = getattr(model_cfg, "output_h", None)
    try:
        model_output_h = int(model_output_h)
    except (TypeError, ValueError):
        return False

    dataset_image_size = _get_dataset_image_size(cfg)
    if dataset_image_size is not None and dataset_image_size >= model_output_h:
        return False

    return model_output_h == 1024


def postprocess(x, cfg=None, true_img=False) -> Union[List[torch.Tensor], torch.Tensor]:
    if _should_downsample_model_output(cfg, true_img):
        model_input = x[0] if isinstance(x, list) else x
        if model_input.dim() == 3:
            model_input = model_input.unsqueeze(0)
        noise_model_512_to_1024 = dinv.physics.GaussianNoise(sigma=0)
        model_512_to_1024 = dinv.physics.Downsampling(
        img_size=(3, 1024, 1024),
        factor=2,
        device=model_input.device,
        noise_model=noise_model_512_to_1024,
        filter="bicubic",
        dtype=model_input.dtype,
        )
        def _process_single(v):
            if v.dim() == 3:
                v = v.unsqueeze(0)
            v = v.float()
            v_downsampled = model_512_to_1024.A(v)
            print("DOWNSAMPLING TO:", v_downsampled.shape)
            v_downsampled = (v_downsampled - v_downsampled.min())/(v_downsampled.max() - v_downsampled.min())
            return v_downsampled
    else:
        _process_single = lambda v: (v + 1) / 2    
        
    if type(x) == list:
        return [_process_single(v) for v in x]
    else:
        return _process_single(x)
    
def _resize_to_1024_for_dmd2(cfg, x: torch.Tensor) -> torch.Tensor:
    """
    Returns the upsampled image tensor to 1024x1024 (between 0 and 1).
    Only used for CWGF-DMD with SDXL-family models: map 512x512 inputs to 1024x1024.
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    noise_model_512_to_1024 = dinv.physics.GaussianNoise(sigma=0)
    model_512_to_1024 = dinv.physics.Downsampling(
        img_size=(3, 1024, 1024),
        factor=2,
        device=x.device,
        noise_model=noise_model_512_to_1024,
        filter="bicubic",
    ).A_adjoint
    x_clean = model_512_to_1024(x)
    x_clean = (x_clean - x_clean.min())/(x_clean.max() - x_clean.min())
    return x_clean

def resize_to_1024_for_dmd2(cfg, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[2] != 1024 or x.shape[3] != 1024:
        print(
            f"Input image shape: {x.shape}, resizing to 1024x1024 for CWGF-DMD (SDXL)..."
        )
        x = _resize_to_1024_for_dmd2(cfg, x)
    else:
        print("Image is already 1024x1024, no resizing needed.")
    return x
