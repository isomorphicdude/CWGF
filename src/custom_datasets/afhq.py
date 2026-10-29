import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from .lmdb_dataset import LMDBDataset


def get_afhq_dataset(root, split, transform="default", subset=-1, **kwargs):
    if transform == "default":
        transform = transforms.ToTensor()
    dset = LMDBDataset(root, "afhq", split, transform)

    if subset is None or (isinstance(subset, int) and subset <= 0):
        return dset
    if isinstance(subset, int):
        return Subset(dset, list(range(min(subset, len(dset)))))
    if isinstance(subset, (list, tuple)):
        return Subset(dset, list(subset))
    raise TypeError(f"subset must be int, list, or None; got {type(subset)}")


def get_afhq_loader(
    dset, *, batch_size, num_workers, shuffle, drop_last, pin_memory, **kwargs
):
    loader = DataLoader(
        dset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader
