"""
Common dataset and dataloader.
"""
from collections.abc import Sequence
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .afhq import get_afhq_dataset
from .ffhq import get_ffhq_dataset
from .imagenet import get_imagenet_dataset
from .mscoco import get_mscoco_dataset


class ZipDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset]):
        if len(datasets) == 0:
            raise ValueError("ZipDataset requires at least one dataset.")
        self.datasets = list(datasets)
        first_len = len(self.datasets[0])
        if not all(len(dataset) == first_len for dataset in self.datasets):
            lengths = [len(dataset) for dataset in self.datasets]
            raise ValueError(f"All datasets must have equal length; got {lengths}.")

    def __getitem__(self, index: int):
        return [dataset[index] for dataset in self.datasets]

    def __len__(self) -> int:
        return len(self.datasets[0])


_DATASET_BUILDERS = {
    "imagenet": get_imagenet_dataset,
    "ffhq": get_ffhq_dataset,
    "afhq": get_afhq_dataset,
    "mscoco": get_mscoco_dataset,
}

_LOADER_DEFAULTS: dict[str, Any] = {
    "batch_size": 1,
    "num_workers": 0,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
}


def _to_dict(cfg_section: Any) -> dict[str, Any]:
    """
    Converts a config section to a plain dict.
    """
    if cfg_section is None:
        return {}
    if isinstance(cfg_section, dict):
        return dict(cfg_section)
    if OmegaConf.is_config(cfg_section):
        container = OmegaConf.to_container(cfg_section, resolve=True)
        if container is None:
            return {}
        if not isinstance(container, dict):
            raise TypeError(f"Expected dict config; got {type(container)}")
        return container
    return dict(cfg_section)


def _get_dataset_key(dataset_cfg: Any) -> str:
    """
    Gets the dataset key.
    """
    name = ""
    if isinstance(dataset_cfg, dict):
        name = str(dataset_cfg.get("name", ""))
    else:
        name = str(getattr(dataset_cfg, "name", ""))

    name_lower = name.casefold()
    if "imagenet" in name_lower:
        return "imagenet"
    if "ffhq" in name_lower:
        return "ffhq"
    if "afhq" in name_lower:
        return "afhq"
    if "mscoco" in name_lower or "coco" in name_lower:
        return "mscoco"

    raise ValueError(
        f"Unknown dataset name {name!r}. Supported: "
        + ", ".join(sorted(_DATASET_BUILDERS.keys()))
    )


def get_dataset(cfg: Any, dataset_attr: str = "dataset") -> Dataset:
    """
    Builds a dataset from config.

    Expects `cfg.<dataset_attr>` to be a mapping-like object with at least:
      - name: str
      - root: str
      - split: str
    """

    dataset_cfg = getattr(cfg, dataset_attr)
    dataset_key = _get_dataset_key(dataset_cfg)
    dataset_kwargs = _to_dict(dataset_cfg)
    transform = dataset_kwargs.get("transform", "default")
    if dataset_key in {"imagenet", "mscoco"} and isinstance(transform, str):
        if transform.casefold() == "diffusion":
            # for compatibility
            dataset_kwargs["transform"] = "default"

    builder = _DATASET_BUILDERS[dataset_key]
    return builder(**dataset_kwargs)


def build_loader(
    cfg: Any,
    dataset_attr: str = "dataset",
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Builds a DataLoader from config with shared defaults."""

    if isinstance(dataset_attr, (list, tuple)):
        datasets = [get_dataset(cfg, da) for da in dataset_attr]
        dataset: Dataset = ZipDataset(datasets)
    else:
        dataset = get_dataset(cfg, dataset_attr)

    loader_cfg = getattr(cfg, "loader", None)
    loader_kwargs = {**_LOADER_DEFAULTS, **_to_dict(loader_cfg)}

    if distributed and world_size > 1:
        shuffle = bool(loader_kwargs.get("shuffle", False))
        drop_last = bool(loader_kwargs.get("drop_last", False))
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False

    num_workers = int(loader_kwargs.get("num_workers", 0))
    loader_kwargs.setdefault("persistent_workers", num_workers > 0)
    if num_workers <= 0:
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("prefetch_factor", None)

    return DataLoader(dataset, **loader_kwargs)
