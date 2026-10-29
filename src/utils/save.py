import os
import lmdb
import numpy as np
import pandas as pd
import torch
import torchvision.utils as tvu
import torch.distributed as dist

# _LABELS = pd.read_csv("data/validation_labels.csv")

def _get_save_id(info) -> str:
    """
    Works with both:
      - LMDBDataset style: info["index"] is a list of strings like ["000012.png"]
      - Folder dataset style: info["index"] is a tensor like tensor([12])
        and optionally info["filename"] / info["path"] are strings
    """
    if "filename" in info:
        idx0 = info["filename"][0]  # list[str] after DataLoader collation
        return os.path.splitext(str(idx0))[0]
    if "path" in info:
        idx0 = os.path.basename(info["path"][0])
        return os.path.splitext(str(idx0))[0]

    idx = info["index"]
    # idx might be list[str], tensor, int, etc.
    if torch.is_tensor(idx):
        return str(int(idx[0].item()))
    idx0 = idx[0] if isinstance(idx, (list, tuple)) else idx
    return os.path.splitext(str(idx0))[0]


def save_imagenet_result(x, y, info, samples_root, suffix=""):
    save_id = _get_save_id(info)
    output_path = os.path.join(samples_root, f"{save_id}")
    os.makedirs(output_path, exist_ok=True)

    if x.dim() == 3:
        x = x.unsqueeze(0)

    n = x.shape[0]
    for i in range(n):
        tvu.save_image(x[i], os.path.join(output_path, f"x_{i}.png"))
    # for i in range(n):
    #     if len(suffix) > 0:
    #         tvu.save_image(
    #             x[i],
    #             os.path.join(
    #                 samples_root, info["class_id"][i], f'{info["name"][i]}_{suffix}.png'
    #             ),
    #         )
    #     else:
    #         tvu.save_image(
    #             x[i],
    #             os.path.join(
    #                 samples_root, info["class_id"][i], f'{info["name"][i]}.png'
    #             ),
    #         )


def save_imagenet_result_particles(x, y, info, samples_root, cfg, suffix=""):
    # x = x.view(-1, cfg.algo.n_particles, cfg.dataset.n_channels, cfg.dataset.image_size, cfg.dataset.image_size)
    n = x.size(0)

    for i in range(n):
        # print('info["class_id"][i]', info["class_id"][i])
        class_dir = os.path.join(samples_root, info["class_id"][i])
        # print('class_dir', class_dir)
        os.makedirs(class_dir, exist_ok=True)
    for i in range(n):
        for j in range(cfg.algo.n_particles):
            if len(suffix) > 0:
                tvu.save_image(
                    x[i][j],
                    os.path.join(
                        samples_root,
                        info["class_id"][i],
                        f'{info["name"][i]}_{suffix}_{j}.png',
                    ),
                )
            else:
                tvu.save_image(
                    x[i][j],
                    os.path.join(
                        samples_root, info["class_id"][i], f'{info["name"][i]}_{j}.png'
                    ),
                )

    dist.barrier()


def save_ffhq_result(x, y, info, samples_root, suffix=""):
    save_id = _get_save_id(info)
    output_path = os.path.join(samples_root, f"{save_id}")
    os.makedirs(output_path, exist_ok=True)

    if x.dim() == 3:
        x = x.unsqueeze(0)

    n = x.shape[0]
    for i in range(n):
        tvu.save_image(x[i], os.path.join(output_path, f"x_{i}.png"))


# def save_ffhq_result(x, y, info, samples_root, suffix=""):
#     x_list = [torch.zeros_like(x) for i in range(dist.get_world_size())]
#     idx = info['index']
#     idx_list = [torch.zeros_like(idx) for i in range(dist.get_world_size())]
#     dist.gather(x, x_list, dst=0)
#     dist.gather(idx, idx_list, dst=0)

#     if len(suffix) == 0:
#         lmdb_path = f'{samples_root}.lmdb'
#     else:
#         lmdb_path = f'{samples_root}_{suffix}.lmdb'

#     lmdb_dir = lmdb_path.split('/')[:-1]
#     if len(lmdb_dir) > 0:
#         lmdb_dir = '/'.join(lmdb_dir)
#         os.makedirs(lmdb_dir, exist_ok=True)

#     if dist.get_rank() == 0:
#         x = torch.cat(x_list, dim=0).permute(0, 2, 3, 1).detach().cpu().numpy()
#         idx = torch.cat(idx_list, dim=0).detach().cpu().numpy()
#         x = (x * 255.).astype(np.uint8)
#         n = x.shape[0]
#         env = lmdb.open(lmdb_path, map_size=int(1e12), readonly=False)
#         with env.begin(write=True) as txn:
#             for i in range(n):
#                 xi = x[i].copy()
#                 txn.put(str(int(idx[i])).encode(), xi)

#     dist.barrier()


def save_result(name, x, y, info, samples_root, suffix=""):
    if "ImageNet" in name:
        save_imagenet_result(x, y, info, samples_root, suffix)
    elif "FFHQ" in name:
        save_ffhq_result(x, y, info, samples_root, suffix)
