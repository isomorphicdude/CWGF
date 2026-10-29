# -----------------------------------------------------------
# KID computation using the torchmetrics package.
# -----------------------------------------------------------

import torch
from torchmetrics.image.kid import KernelInceptionDistance

from .torchmetrics_fid import _resolve_device, _to_uint8_fid_input


def _validate_kid_sample_counts(
    n_real: int,
    n_fake: int,
    subset_size: int,
) -> None:
    if subset_size < 2:
        raise ValueError(f"KID subset_size must be at least 2, got {subset_size}.")
    if n_real < subset_size or n_fake < subset_size:
        raise ValueError(
            "KID subset_size exceeds the available samples: "
            f"subset_size={subset_size}, real_samples={n_real}, "
            f"fake_samples={n_fake}. Reduce exp.kid_subset_size or evaluate "
            "more images."
        )


def compute_kid(
    dataset_samples: torch.Tensor,
    model_samples: torch.Tensor,
    device: str | torch.device = "cuda",
    batch_size: int = 10,
    subsets: int = 100,
    subset_size: int = 1000,
    cpu_before_compute: bool = False,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute the Kernel Inception Distance mean and standard deviation."""
    if batch_size <= 0:
        raise ValueError(f"KID batch_size must be positive, got {batch_size}.")

    n_real = int(dataset_samples.shape[0])
    n_fake = int(model_samples.shape[0])
    _validate_kid_sample_counts(n_real, n_fake, subset_size)
    metric_device = _resolve_device(device)

    real_images = _to_uint8_fid_input(dataset_samples)
    fake_images = _to_uint8_fid_input(model_samples)

    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if fake_images.shape[1] == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)

    kid_metric = KernelInceptionDistance(
        feature=2048,
        subsets=subsets,
        subset_size=subset_size,
    ).to(metric_device)

    print(f"Processing {n_real} real images for KID...")
    for i in range(0, n_real, batch_size):
        kid_metric.update(real_images[i : i + batch_size].to(metric_device), real=True)

    print(f"Processing {n_fake} fake images for KID...")
    for i in range(0, n_fake, batch_size):
        kid_metric.update(fake_images[i : i + batch_size].to(metric_device), real=False)

    if cpu_before_compute and metric_device.type != "cpu":
        print("Moving KID metric state to CPU before compute()...")
        kid_metric = kid_metric.cpu()

    print("Computing KID score...")
    if seed is None:
        kid_mean, kid_std = kid_metric.compute()
    else:
        # TorchMetrics samples KID subsets with the CPU RNG. Forking it makes the
        # estimate repeatable without changing the caller's global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            kid_mean, kid_std = kid_metric.compute()

    return kid_mean.item(), kid_std.item()
