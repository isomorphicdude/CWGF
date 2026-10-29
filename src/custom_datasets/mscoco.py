"""MS-COCO dataset loader implementations (LMDB-backed)."""

from functools import partial

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from .lmdb_dataset import LMDBDataset
from .imagenet import center_crop_arr


def get_mscoco_dataset(root, split, transform="default", subset=-1, **kwargs):
    image_size = kwargs.get("image_size", 512)
    if isinstance(transform, str) and transform in {"default", "diffusion"}:
        transform = transforms.Compose(
            [partial(center_crop_arr, image_size=image_size), transforms.ToTensor()]
        )
    dset = LMDBDataset(root, "mscoco2017", split, transform)

    if subset is None or (isinstance(subset, int) and subset <= 0):
        return dset
    if isinstance(subset, int):
        return Subset(dset, list(range(min(subset, len(dset)))))
    if isinstance(subset, (list, tuple)):
        return Subset(dset, list(subset))
    raise TypeError(f"subset must be int, list, or None; got {type(subset)}")


def get_mscoco_loader(
    dset, *, batch_size, num_workers, shuffle, drop_last, pin_memory, **kwargs
):
    loader = DataLoader(
        dset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    return loader


# Backwards-compat aliases (historical copy/paste).
get_imagenet_dataset = get_mscoco_dataset
get_imagenet_loader = get_mscoco_loader
