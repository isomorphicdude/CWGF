"""FFHQ dataset loader implementations."""

import os
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, DistributedSampler

from .lmdb_dataset import LMDBDataset


def get_ffhq_dataset(root, split, transform="default", subset=-1, **kwargs):
    use_lmdb = kwargs.get("use_lmdb", False)
    if use_lmdb:
        return _get_ffhq_dataset_lmdb(root, split, transform, subset, **kwargs)
    else:
        return _get_ffhq_dataset(root, split, transform, subset, **kwargs)


def _get_ffhq_dataset_lmdb(root, split, transform="default", subset=-1, **kwargs):
    if transform == "default":
        transform = transforms.ToTensor()
    dset = LMDBDataset(root, "ffhq", split, transform)
    if isinstance(subset, int) and subset > 0:
        dset = Subset(dset, list(range(subset)))
    else:
        assert isinstance(subset, list)
        dset = Subset(dset, subset)
    return dset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageFolderFlat(Dataset):
    """
    Loads images from a directory (recursively), without class subfolders.
    Returns (x, y, info) to match the expected loader output.

    Here y is set equal to x (placeholder), because your code builds the true measurement y_0 on the fly.
    """

    def __init__(self, root_dir, transform=None, recursive=True):
        self.root_dir = Path(root_dir)
        self.transform = transform

        files = []
        if recursive:
            for ext in IMG_EXTS:
                files.extend(self.root_dir.rglob(f"*{ext}"))
        else:
            for ext in IMG_EXTS:
                files.extend(self.root_dir.glob(f"*{ext}"))

        self.files = sorted([p for p in files if p.is_file()])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found in {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(img)
        else:
            # if you ever hit this, you probably want transforms.ToTensor() at least
            raise RuntimeError("No transform provided; expected a tensor transform.")

        # placeholder y; process_dataset will replace it with y_0 (the actual measurement)
        y = x

        info = {
            "index": torch.tensor(idx, dtype=torch.long),
            "path": str(path),
            "filename": path.name,
        }
        return x, y, info


def _get_ffhq_dataset(
    root, split, transform="default", subset=-1, recursive=True, **kwargs
):
    """
    root: path that contains split folders: train/ val/ test/
    split: one of 'train', 'val', 'test'
    """
    if transform == "default":
        transform = transforms.ToTensor()

    split_dir = Path(root) / split
    dset = ImageFolderFlat(split_dir, transform=transform, recursive=recursive)

    if isinstance(subset, int) and subset > 0:
        dset = Subset(dset, list(range(subset)))
    else:
        if subset != -1:  # keep your previous behavior but don't assert on -1
            assert isinstance(subset, list)
            dset = Subset(dset, subset)

    return dset


def get_ffhq_loader(
    dset, *, batch_size, num_workers, shuffle, drop_last, pin_memory, **kwargs
):
    # sampler = DistributedSampler(dset, shuffle=shuffle, drop_last=drop_last)
    loader = DataLoader(
        dset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    return loader
