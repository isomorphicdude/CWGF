# -----------------------------------------------------------
# Ported from IPLD
# This file contains the computation code for FID using `torchmetrics` package.
# See https://torchmetrics.readthedocs.io/en/stable/ for more details.
# -----------------------------------------------------------

import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def _to_uint8_fid_input(images: torch.Tensor) -> torch.Tensor:
    """
    Converts images to uint8 in [0, 255] for torchmetrics FID.
    Supports inputs in either [0, 1] or [-1, 1].
    """
    images = images.detach().float()
    if images.amin().item() < 0.0:
        print(f"Input images appear to be in [{images.amin().item()}, {images.amax().item()}], converting to [0,1]...   ")
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    return (images * 255.0).round().to(torch.uint8)


def _resolve_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _compute_fid_with_symmetric_eigh(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """
    Work around on cluster for linalg.
    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    # symmetry stability
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(sigma1)
    sqrt_eigvals = eigvals.clamp(min=0).sqrt()
    sqrt_sigma1 = (eigvecs * sqrt_eigvals.unsqueeze(0)) @ eigvecs.T
    # compute sqrt of symmetric stability
    sqrt_product = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    sqrt_product = (sqrt_product + sqrt_product.T) / 2
    c = torch.linalg.eigvalsh(sqrt_product).clamp(min=0).sqrt().sum(dim=-1)

    return a + b - 2 * c


def _compute_fid_from_metric_state(
    fid_metric: FrechetInceptionDistance,
) -> torch.Tensor:
    if (
        fid_metric.real_features_num_samples < 2
        or fid_metric.fake_features_num_samples < 2
    ):
        raise RuntimeError(
            "More than one sample is required for both the real and fake distributed "
            "to compute FID"
        )
    mean_real = (
        fid_metric.real_features_sum / fid_metric.real_features_num_samples
    ).unsqueeze(0)
    mean_fake = (
        fid_metric.fake_features_sum / fid_metric.fake_features_num_samples
    ).unsqueeze(0)

    cov_real_num = (
        fid_metric.real_features_cov_sum
        - fid_metric.real_features_num_samples * mean_real.t().mm(mean_real)
    )
    cov_real = cov_real_num / (fid_metric.real_features_num_samples - 1)
    cov_fake_num = (
        fid_metric.fake_features_cov_sum
        - fid_metric.fake_features_num_samples * mean_fake.t().mm(mean_fake)
    )
    cov_fake = cov_fake_num / (fid_metric.fake_features_num_samples - 1)
    fid = _compute_fid_with_symmetric_eigh(
        mean_real.squeeze(0),
        cov_real,
        mean_fake.squeeze(0),
        cov_fake,
    )
    return fid.to(fid_metric.orig_dtype)


def compute_fid(
    dataset_samples: torch.Tensor,
    model_samples: torch.Tensor,
    device: str | torch.device = "cuda",
    batch_size: int = 10,
    cpu_before_compute: bool = False,
):
    """
    Computes the Fréchet Inception Distance (FID) between two sets of images,
    which can have different numbers of samples.
    """
    metric_device = _resolve_device(device)

    # convert to [0, 255] and uint8, as required by torchmetrics
    real_images = _to_uint8_fid_input(dataset_samples)
    fake_images = _to_uint8_fid_input(model_samples)

    # repeat channels to be RGB
    if real_images.shape[1] == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if fake_images.shape[1] == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)

    fid_metric = FrechetInceptionDistance(feature=2048).to(metric_device)

    n_real = real_images.shape[0]
    print(f"Processing {n_real} real images...")
    for i in range(0, n_real, batch_size):
        real_batch = real_images[i : i + batch_size, ...].to(metric_device)
        fid_metric.update(real_batch, real=True)

    n_fake = fake_images.shape[0]
    print(f"Processing {n_fake} fake images...")
    for i in range(0, n_fake, batch_size):
        fake_batch = fake_images[i : i + batch_size, ...].to(metric_device)
        fid_metric.update(fake_batch, real=False)

    if cpu_before_compute and metric_device.type != "cpu":
        print("Moving FID metric state to CPU before compute()...")
        fid_metric = fid_metric.cpu()

    print("Computing FID score...")
    if fid_metric.real_features_sum.device.type == "cuda":
        print("Using CUDA symmetric eigensolver for FID covariance square root...")
        fid_value = _compute_fid_from_metric_state(fid_metric)
    else:
        fid_value = fid_metric.compute()
    return fid_value.item()
