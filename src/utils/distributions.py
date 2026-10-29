"""Implementations of various probability distributions."""

from contextlib import nullcontext
import math

import torch
import numpy as np
from scipy.stats import Normal, norm

LOG_2PI = math.log(2 * math.pi)


def gaussian_diag_logprob(x, mu, logvar):
    """
    Computes the log probability of a Gaussian distribution with diagonal covariance.
    """
    var = torch.exp(logvar)
    assert (
        x.shape == mu.shape == var.shape
    ), f"Shapes do not match: x {x.shape}, mu {mu.shape}, var {var.shape}"

    return -0.5 * torch.sum(
        (x - mu).pow(2) * torch.exp(-logvar) + logvar + LOG_2PI, dim=-1
    )


def gaussian_std_logprob(x):
    """
    Standard Gaussian log probability.
    """
    return -0.5 * (x.pow(2) + LOG_2PI)


def gaussian_iso_logprob(
    x: torch.Tensor, mu, logvar, sequential=False, scaled=False, pairwise=True, chunk_size=None
):
    """
    Gaussian log prob with isotropic covariance: N(mu, exp(logvar) I).

    Modes:
      - pairwise=True: all-pairs logprob between batches:
            x: [B, D], mu: [M, D] -> returns [B, M]

    sequential=True uses a nested-loop implementation
    chunk_size (only for pairwise vectorized) computes in blocks over mu to reduce peak memory.
    """
    d = x.shape[-1]
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype

    # NOTE: cast to fp32 for fp16/bf16 inputs to avoid overflow/underflow in exp/sums/matmuls.
    scale_const = math.sqrt(d) if scaled else 1.0
    inv_scale_const = 1.0 / scale_const

    logvar_t = torch.as_tensor(logvar, device=x.device, dtype=compute_dtype)
    if logvar_t.numel() != 1:
        raise ValueError(
            f"logvar must be a scalar for isotropic covariance, got shape {tuple(logvar_t.shape)}"
        )
    logvar_t = logvar_t.reshape(())
    # clamp to avoid numerical issues
    logvar_min, logvar_max = -30.0, 20.0
    logvar_t = torch.where(
        torch.isfinite(logvar_t),
        logvar_t,
        logvar_t.new_tensor(logvar_min),
    ).clamp(min=logvar_min, max=logvar_max)

    inv_var = torch.exp(-logvar_t)
    const = d * (logvar_t + logvar_t.new_tensor(LOG_2PI))

    if not pairwise:
        x_ = x.to(compute_dtype)
        mu_ = mu.to(compute_dtype)
        diff = x_ - mu_
        sq = torch.sum(diff * diff, dim=-1)
        return -0.5 * (sq * inv_var * inv_scale_const + const)

    # pairwise mode: x (b,d) and mu (m,d) -> (b,m)
    if x.ndim != 2 or mu.ndim != 2:
        raise ValueError(
            "pairwise=True expects x and mu to be 2D tensors: [B,D] and [M,D]."
        )
    if x.shape[1] != mu.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: x is {x.shape}, mu is {mu.shape}."
        )

    B, D = x.shape
    M = mu.shape[0]

    x_ = x.to(compute_dtype)
    x2 = torch.sum(x_ * x_, dim=1, keepdim=True)

    if sequential:
        mu_ = mu.to(compute_dtype)
        out = x_.new_empty((B, M))
        mu2 = torch.sum(mu_ * mu_, dim=1)
        for i in range(B):
            x2_i = x2[i, 0]
            cross = mu_ @ x_[i]
            dist2 = (x2_i + mu2 - 2.0 * cross).clamp_min(0.0)
            out[i] = -0.5 * (dist2 * inv_var * inv_scale_const + const)
        return out

    def pairwise_block(mu_block):
        mu_block_ = mu_block.to(compute_dtype)
        m2 = torch.sum(mu_block_ * mu_block_, dim=1).unsqueeze(0)
        cross = x_ @ mu_block_.t()
        dist2 = (x2 + m2 - 2.0 * cross).clamp_min(0.0)
        return -0.5 * (dist2 * inv_var * inv_scale_const + const)

    if chunk_size is None:
        return pairwise_block(mu)

    # chunked over mu to reduce memory
    outs = []
    for start in range(0, M, chunk_size):
        outs.append(pairwise_block(mu[start : start + chunk_size]))
    return torch.cat(outs, dim=1)

def inv_cdf_sample(u, weights: np.ndarray):
    """
    Samples from P by finding:
    inf {t: F_n(t)>=u} for empirical cdf F_n of P.
    With F_n(t)=sum_i W_{i=1}^N 1_{X_i <= t}
    Low var version uses u = i/N
    """
    csum_w = np.cumsum(weights)
    csum_w /= csum_w[-1]
    u = np.clip(u, 0.0, np.nextafter(csum_w[-1], -np.inf))
    time_idx = np.searchsorted(csum_w, u, side="left")
    return time_idx

def stratify(time_idx: int, timesteps):
    """
    "Stratified sampling" by converting time_idx
    to closest time in timesteps, breaking ties by rounding down.
    """
    time_idx = int(np.clip(time_idx, 0, 999))
    timesteps = np.asarray(timesteps)
    if timesteps.ndim != 1:
        timesteps = timesteps.reshape(-1)
    if timesteps.size > 1 and timesteps[0] > timesteps[-1]:
        timesteps = timesteps[::-1]

    time_idx = int(np.clip(time_idx, int(timesteps[0]), int(timesteps[-1])))
        
    ins_idx = np.searchsorted(timesteps, time_idx, side="right")
    if ins_idx == 0:
        return int(timesteps[0])
    if ins_idx >= timesteps.size:
        return int(timesteps[-1])
    if timesteps[ins_idx - 1] == time_idx:
        return time_idx

    lower = timesteps[ins_idx - 1]
    upper = timesteps[ins_idx]
    lower_dist = time_idx - lower
    upper_dist = upper - time_idx
    if lower_dist < upper_dist:
        return int(lower)
    if upper_dist < lower_dist:
        return int(upper)
    return int(lower)

def get_idx_from_time(time_idx: int, timesteps):
    """
    Returns the exact index corresponding to the time_idx
    in the consistency steps.
    """
    for idx, t in enumerate(timesteps):
        if t == time_idx:
            return idx
    raise ValueError(f"time_idx {time_idx} not found in timesteps {timesteps}")
    

def normalize_pdf(pdf: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    pdf = np.clip(np.asarray(pdf, dtype=np.float64), 0.0, None)
    z = np.trapezoid(pdf, t_grid)
    if (not np.isfinite(z)) or z <= 0.0:
        raise ValueError("Cannot normalize PDF: non-positive integral")
    return pdf / z, z


EPS = 1e-20
def curves_from_alpha_beta(alpha_bar, betas, t=None, pmean=2.4, psigma=5.0, sigma_data=0.5):
    """
    Returns:
      t:        normalized time in [0,1]
      inv_snr:  (1 - alpha_bar) / alpha_bar
      dwdt:     d/dt [ min(1, gamma * inv_snr) * p_lambda ], where p_lambda=(1-alpha_bar)/beta
    """
    alpha_bar = np.asarray(alpha_bar, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)

    if t is None:
        t = np.linspace(0.0, 1.0, len(alpha_bar), dtype=np.float64)
    sigma2 = np.clip(1.0 - alpha_bar, 0.0, 1.0)
    inv_snr = sigma2 / np.clip(alpha_bar, EPS, None)
    # log snr density
    p_lambda = sigma2 / np.clip(betas, EPS, None)
    #TODO: adjust hyperparams here
    edm_normal = Normal(mu=pmean, sigma=psigma)
    edm_pdf = edm_normal.pdf(np.log(inv_snr))
    w_t = edm_pdf * (inv_snr + sigma_data**2)
    #TODO: use autodiff here
    # numerical derivative w.r.t. t (finite differences not autodiff)
    # dwdt = np.gradient(w_t, t, edge_order=2)

    # dwdt = inv_snr * Normal(
    #     mu=t[502],
    #     sigma=0.23
    # ).pdf(t)
    # return t, inv_snr, dwdt, w_t, edm_pdf
    snr = alpha_bar / np.clip(1.0 - alpha_bar, EPS, None)
    time_gaussian = norm.pdf(t, loc=0.82, scale=0.43) * (
        np.sqrt(np.minimum(snr, 0.61))
    )
    return t, inv_snr, time_gaussian, w_t, edm_pdf
    
