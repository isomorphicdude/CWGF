import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb

from eval.compute_fid import compute_reconstruction_metrics
from utils.torch_utils import init_omega, seed_everything


@hydra.main(
    version_base="1.2", config_path="../configs", config_name="ffhq_stable_diffusion"
)
def main(cfg: DictConfig):
    # Keep the same config init pattern as src/main.py.
    cwd = HydraConfig.get().runtime.output_dir
    cfg = init_omega(cfg, cwd)
    cfg.algo.name_folder = cfg.algo.name
    seed_everything(cfg.exp.seed)

    if cfg.exp.use_wandb:
        wandb.init(
            project=cfg.exp.get("name", "GradientFlowInverse"),
            name=cfg.exp.get(
                "run_name",
                f"{cfg.dataset.name}_{cfg.algo.name_folder}_{cfg.exp.deg}_{cfg.model.name}",
            ),
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.exp.entity,
        )

    (
        fid,
        kid_mean,
        kid_std,
        mean_psnr,
        mean_lpips,
        num_matched,
        missing_ids,
    ) = compute_reconstruction_metrics(cfg, use_single=cfg.exp.use_single)
    if fid is None or mean_psnr is None or mean_lpips is None:
        raise RuntimeError("No matched output images found, cannot compute metrics.")
    if cfg.exp.get("get_kid", False) and (kid_mean is None or kid_std is None):
        raise RuntimeError("KID was enabled but could not be computed.")

    if cfg.exp.use_wandb:
        payload = {
            "fid": fid,
            "mean_psnr": mean_psnr,
            "mean_lpips": mean_lpips,
        }
        if kid_mean is not None:
            payload["kid"] = kid_mean
            payload["kid_std"] = kid_std
        wandb.log(payload)
        wandb.summary["fid"] = fid
        wandb.summary["mean_psnr"] = mean_psnr
        wandb.summary["mean_lpips"] = mean_lpips
        if kid_mean is not None:
            wandb.summary["kid"] = kid_mean
            wandb.summary["kid_std"] = kid_std

    print(f"FID: {fid:.6f}")
    if kid_mean is not None:
        print(f"KID mean: {kid_mean:.6f}")
        print(f"KID std: {kid_std:.6f}")
    print(f"Mean PSNR: {mean_psnr:.6f}")
    print(f"Mean LPIPS: {mean_lpips:.6f}")
    print(f"Matched samples: {num_matched}")
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = " ..." if len(missing_ids) > 10 else ""
        print(f"Missing output folders for {len(missing_ids)} samples: {preview}{suffix}")


if __name__ == "__main__":
    main()
