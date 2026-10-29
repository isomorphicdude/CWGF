import os
import gc
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from PIL import Image
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import lpips
import wandb
import deepinv as dinv
from deepinv.physics import DecomposablePhysics as dinv_H

from custom_datasets import build_loader
from utils.degradations_dinv import get_degradation_image, build_degradation_model
from utils.torch_utils import get_logger, init_omega, seed_everything, ensure_dir
from utils.img_utils import resize_input, resize_to_1024_for_dmd2, postprocess, preprocess
from utils.hf_utils import (
    init_sd15,
    init_sd21,
    init_dmd15,
    init_dmd2,
    init_sdxllora,
    is_sdxl_model_name,
)
from utils.distributed import get_distributed_info, init_distributed
from algos import build_algo
from utils.save import save_result
from eval.compute_fid import compute_lpips, compute_psnr, compute_reconstruction_metrics


def prepare_paths(cfg):
    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, cfg.algo.name_folder)
    output_path = os.path.join(cfg.exp.root, cfg.exp.output_path, cfg.algo.name_folder)
    evol_path = os.path.join(cfg.exp.root, cfg.exp.evol_path, cfg.algo.name_folder)
    for path in (deg_path, output_path, evol_path):
        ensure_dir(path)
    return deg_path, output_path, evol_path


def build_pipeline(cfg):
    sd_local_root = cfg.exp.get("sd_local_root", None)
    model_name = str(cfg.model.name)
    use_fp16 = bool(cfg.model.get("use_fp16", False))
    if "dmd" in cfg.algo.name:
        if model_name == "dmd2":
            return init_dmd2(sd_local_root, "cuda", cfg)
        if model_name == "sdxllora":
            return init_sdxllora(sd_local_root, "cuda", cfg)
        if model_name == "sd15lora":
            return init_dmd15(sd_local_root, "cuda", cfg)
        if model_name == "sd21":
            return init_sd21(
                cfg.algo.num_diffusion_steps,
                sd_local_root,
                "cuda",
                scheduler_name="lcm",
                fp16=use_fp16,
            )
        if model_name == "sd15":
            return init_sd15(
                cfg.algo.num_diffusion_steps,
                sd_local_root,
                "cuda",
                scheduler_name="lcm",
                fp16=use_fp16,
            )
        raise ValueError(
            "DMD-family algorithms currently support model=sd15, model=sd21, "
            "model=sd15lora, model=sdxllora, or model=dmd2."
        )
    if "latino" in cfg.algo.name:
        if model_name == "dmd2":
            return init_dmd2(sd_local_root, "cuda", cfg)
        if model_name == "sdxllora":
            return init_sdxllora(sd_local_root, "cuda", cfg)
        if model_name == "sd15":
            return init_dmd15(sd_local_root, "cuda", cfg)
        raise ValueError(
            "LATINO-family algorithms currently support model=dmd2, "
            "model=sdxllora, or model=sd15."
        )
    else:
        if model_name == "sd21":
            return init_sd21(
                cfg.algo.num_diffusion_steps,
                sd_local_root,
                "cuda",
                fp16=use_fp16,
            )
        if model_name == "sdxllora":
            return init_sdxllora(sd_local_root, "cuda", cfg)
        else:
            return init_sd15(
                cfg.algo.num_diffusion_steps,
                sd_local_root,
                "cuda",
                fp16=use_fp16,
            )



def expand_particles(x, n_particles):
    if n_particles > 1:
        return x.repeat(n_particles, 1, 1, 1)
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x

#NOTE: only use 1024 for CWGF-DMD variants with SDXL-family distilled models
def is_cwgf_dmd_dmd2(cfg) -> bool:
    return (
        ("latino" in cfg.algo.name or "dmd" in cfg.algo.name)
        and is_sdxl_model_name(cfg.model.name)
    )


def build_measurement(x, H: dinv_H, cfg, *, generator: torch.Generator | None = None):
    return H(x.to("cuda"))


def build_y_obs_observation(
    x,
    H: dinv_H,
    cfg,
    *,
    noise_sigma: float,
    generator: torch.Generator | None = None,
):
    x_obs = (x + 1) * 0.5
    y_obs = build_measurement(x_obs, H, cfg, generator=generator)
    return preprocess(y_obs), float(noise_sigma) * 2.0


def save_degraded_(
    cfg,
    y_0,
    H,
    logger,
    *,
    x_gt: torch.Tensor | None = None,
    lpips_model=None,
    wandb_step: int | None = None,
):
    if cfg.exp.save_deg is not True:
        return
    if cfg.exp.deg == "deblur_motion":
        x_deg = postprocess(y_0, cfg)
    else:
        x_deg = postprocess(get_degradation_image(y_0, H, cfg), cfg)

    deg_path = os.path.join(cfg.exp.root, cfg.exp.deg_path, "x_deg.png")
    x_deg_to_save = x_deg.detach().clamp(0, 1)
    save_image(x_deg_to_save[0].cpu(), deg_path)
    logger.info(f"Degradated image y saved in {deg_path}")

    if x_gt is None or lpips_model is None:
        return

    x_gt_eval = x_gt.detach().clamp(0, 1)
    if x_gt_eval.shape != x_deg_to_save.shape:
        logger.warning(
            "Skipping degraded PSNR/LPIPS (shape mismatch): "
            f"x_deg={tuple(x_deg_to_save.shape)} vs x_gt={tuple(x_gt_eval.shape)}"
        )
        return

    x_deg_saved = x_deg_to_save[0:1].cpu()
    x_gt_saved = x_gt_eval[0:1]

    psnr_deg = compute_psnr(x_deg_saved, x_gt_saved)[0].detach().cpu().item()
    lpips_deg = (
        compute_lpips(
            lpips_model,
            x_deg_saved,
            x_gt_saved,
        )[0, 0, 0, 0]
        .detach()
        .cpu()
        .item()
    )

    logger.info(f"Degraded PSNR (x_deg.png): {psnr_deg}")
    logger.info(f"Degraded LPIPS (x_deg.png): {lpips_deg}")
    print(x_deg_saved.shape, x_gt_saved.shape)


def sample_with_algo(cfg, algo, x, y, ts, generator, y_0):
    return algo.sample(x, y, ts, generator, y_0)


def process_single_image(
    cfg, algo, H: dinv_H, ts, generator, lpips_, output_path, logger
):
    img_path = os.path.join(cfg.exp.img_path, cfg.exp.img_id)
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).cuda()

    if is_cwgf_dmd_dmd2(cfg):
        x = resize_to_1024_for_dmd2(cfg, x)
    else:
        logger.info(f"Input image shape: {x.shape}, resizing to 512x512...")
        x = resize_input(x)

    y = None
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = preprocess(x)
    sigma_x0_restore = None
    if "latino" in cfg.algo.name:
        y_0, sigma_y = build_y_obs_observation(
            x,
            H,
            cfg,
            noise_sigma=float(cfg.algo.get("sigma_y", cfg.exp.get("sigma_x0", 1e-2))),
            generator=generator,
        )
        algo.sigma_y = sigma_y
    # elif cfg.algo.name == "cwgf_dmd" and is_sdxl_model_name(cfg.model.name):
    #     y_0, sigma_x0_eff = build_y_obs_observation(
    #         x,
    #         H,
    #         cfg,
    #         noise_sigma=float(cfg.exp.get("sigma_x0", 1e-2)),
    #         generator=generator,
    #     )
    #     sigma_x0_restore = float(cfg.exp.sigma_x0)
    #     cfg.exp.sigma_x0 = sigma_x0_eff
    else:
        y_0 = build_measurement(x, H, cfg, generator=generator)
    x = expand_particles(x, cfg.algo.n_particles)
    y_0 = expand_particles(y_0, cfg.algo.n_particles)
    logger.info(f"Images loaded and measurements generated")
    #TODO: check the psnr and lpips for degraded image, seems too good?
    save_degraded_(cfg, y_0, H, logger, x_gt=postprocess(x, cfg), lpips_model=lpips_)
    print(y_0.shape, x.shape)

    start = time.time()
    try:
        xt_s, _ = sample_with_algo(cfg, algo, x, y, ts, generator, y_0)
    finally:
        if sigma_x0_restore is not None:
            cfg.exp.sigma_x0 = sigma_x0_restore
    end = time.time()
    print(f"Time taken: {end - start}")

    if isinstance(xt_s, list):
        xo = postprocess(xt_s[0], cfg).cpu()
        x_gt = postprocess(x, cfg)
    else:
        xo = postprocess(xt_s, cfg).cpu()
        x_gt = postprocess(x, cfg)
    psnr = compute_psnr(xo, x_gt)
    logger.info(f"Mean PSNR: {psnr.mean()}")
    logger.info(f"PSNR: {psnr}")

    lpips_score = compute_lpips(lpips_, xo, x_gt)
    logger.info(f"Mean LPIPS: {lpips_score.mean()}")
    logger.info(f"LPIPS: {lpips_score[:,0,0,0]}")

    img_id = cfg.exp.img_id.split(".")[0]
    output_path_img = f"{output_path}/{img_id}"
    ensure_dir(output_path_img)

    for i in range(cfg.algo.n_particles):
        save_image(xo[i], f"{output_path_img}/x_hat_{i}.png")
    image_grid = make_grid(xo.cpu())
    save_image(image_grid, f"{output_path_img}/x_hat_grid.png")

    if cfg.exp.use_wandb:
        wandb.log(
            {
                "psnr": psnr.mean().item(),
                "lpips": lpips_score.mean().item(),
            }
        )
        wandb.log(
            {
                "generated_images": wandb.Image(
                    image_grid, caption="Generated Images Grid"
                ),
            }
        )
        
    # save lpips to npz
    import numpy as np
    if os.path.exists(f"{output_path_img}/lpips_score.npz"):
        existing_data = np.load(f"{output_path_img}/lpips_score.npz")
        existing_lpips = existing_data["lpips"]
        new_lpips = np.concatenate([existing_lpips, lpips_score.detach().cpu().numpy()])
        np.savez(f"{output_path_img}/lpips_score.npz", lpips=new_lpips)
    else:
        np.savez(f"{output_path_img}/lpips_score.npz", lpips=lpips_score.detach().cpu().numpy())

    torch.cuda.empty_cache()
    logger.info(f"Done. You can find the generated images in {output_path_img}")


def process_dataset(
    cfg,
    algo,
    H,
    ts,
    generator,
    lpips_,
    output_path,
    logger,
    dataset_name,
    max_num_images,
    rank=0,
    world_size=1,
    is_main_process=True,
    resume=False,
    start_idx=0,
):
    def get_start_idx(output_dir, dataset_length):
        # folder structure
        # root/cwgf/inp_random/00000/x_0.png
        # root/cwgf/inp_random/00000/x_1.png
        max_existing_idx = -1
        if not os.path.isdir(output_dir):
            return 0

        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if not os.path.isdir(item_path):
                continue
            try:
                item_idx = int(item)
            except ValueError:
                continue
            max_existing_idx = max(max_existing_idx, item_idx)

        if max_existing_idx < 0:
            if is_main_process:
                logger.info(
                    "Resume enabled but no numeric output index found in %s; starting from idx 0.",
                    output_dir,
                )
            return 0

        start_idx = max_existing_idx + 1
        if is_main_process:
            logger.info(
                "Resume enabled: max existing idx=%s in %s; dataset length=%s; resuming from idx=%s.",
                max_existing_idx,
                output_dir,
                dataset_length,
                start_idx,
            )
        return start_idx

    use_distributed = world_size > 1
    loader = build_loader(
        cfg,
        distributed=use_distributed,
        rank=rank,
        world_size=world_size,
    )
    loader_len = len(loader)
    dataset_len = len(loader.dataset)
    if use_distributed and hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(cfg.exp.seed)
    if is_main_process:
        logger.info(f"Dataset size is {dataset_len}, loader length is {loader_len}")

    output_path_deg = os.path.join(output_path, cfg.exp.deg)
    ensure_dir(output_path_deg)

    start_idx = int(start_idx or 0)
    if start_idx < 0:
        raise ValueError(f"exp.start_idx must be non-negative, got {start_idx}.")

    psnr_list = []
    lpips_list = []
    max_num_images_per_rank = max_num_images
    if use_distributed and max_num_images > 0:
        max_num_images_per_rank = (max_num_images + world_size - 1) // world_size

    resume_start_idx = 0
    if resume:
        resume_start_idx = get_start_idx(output_path_deg, dataset_len)

    if start_idx >= dataset_len:
        if is_main_process:
            logger.info(
                "Configured start idx %s is >= dataset length %s; skipping dataset processing.",
                start_idx,
                dataset_len,
            )
        return psnr_list, lpips_list

    if start_idx > 0 and is_main_process:
        logger.info("Starting dataset processing from loader idx %s.", start_idx)

    processed = 0
    for it, (x, y, info) in enumerate(loader):
        if it < start_idx:
            continue
        if max_num_images_per_rank > 0 and processed >= max_num_images_per_rank:
            if is_main_process:
                logger.info(f"Reached max number of images: {max_num_images}")
            break
        raw = info["index"]
        if raw is None:
            raise ValueError("No dataloader info.")
        sample_id = raw[0].split(".")[0]  # format "00000.png"
        algo.current_img_id = sample_id
        sample_global_idx = int(sample_id)
        if resume and sample_global_idx < resume_start_idx:
            print(
                f"[rank {rank}] Skipping image index {sample_global_idx} (loader it={it}) "
                f"because resume start index is {resume_start_idx}."
            )
            continue
        if cfg.exp.deg == "deblur_motion":
            # Preserve the original single-GPU behavior while making
            # multi-GPU runs seed by global sample id instead of local rank iteration.
            motion_seed = cfg.exp.seed + (
                sample_global_idx if use_distributed else it
            )
            seed_everything(motion_seed)
            generator = torch.Generator(device="cuda").manual_seed(motion_seed)
            H = build_degradation_model(cfg)
            algo.H = H
        # logger.info(f"Input image:{ info['index'][0]}")

        if is_cwgf_dmd_dmd2(cfg):
            x = x.cuda()
            y = y.cuda()
            x = resize_to_1024_for_dmd2(cfg, x)
        else:
            x = resize_input(x)
        x = x.cuda()
        y = y.cuda()
        x = preprocess(x)
        kwargs = info

        sigma_x0_restore = None
        if "latino" in cfg.algo.name:
            y_0, sigma_y = build_y_obs_observation(
                x,
                H,
                cfg,
                noise_sigma=float(cfg.algo.get("sigma_y", cfg.exp.get("sigma_x0", 1e-2))),
                generator=generator,
            )
            algo.sigma_y = sigma_y
        # elif cfg.algo.name == "cwgf_dmd" and is_sdxl_model_name(cfg.model.name):
        #     y_0, sigma_x0_eff = build_y_obs_observation(
        #         x,
        #         H,
        #         cfg,
        #         noise_sigma=float(cfg.exp.get("sigma_x0", 1e-2)),
        #         generator=generator,
        #     )
        #     sigma_x0_restore = float(cfg.exp.sigma_x0)
        #     cfg.exp.sigma_x0 = sigma_x0_eff
        else:
            y_0 = build_measurement(x, H, cfg, generator=generator)
        x = expand_particles(x, cfg.algo.n_particles)
        y_0 = expand_particles(y_0, cfg.algo.n_particles)
        logger.info(f"Images loaded and measurements generated")

        if is_main_process or not use_distributed:
            save_degraded_(
                cfg,
                y_0,
                H,
                logger,
                x_gt=postprocess(x, cfg),
                lpips_model=lpips_,
                wandb_step=it,
            )

        try:
            xt_s, _ = sample_with_algo(cfg, algo, x, y, ts, generator, y_0)
        finally:
            if sigma_x0_restore is not None:
                cfg.exp.sigma_x0 = sigma_x0_restore

        if isinstance(xt_s, list):
            xo = postprocess(xt_s[0], cfg).cpu()
            x_gt = postprocess(x, cfg)
        else:
            xo = postprocess(xt_s, cfg).cpu()
            x_gt = postprocess(x, cfg)

        save_result(dataset_name, xo, y, info, output_path_deg, "")

        psnr = compute_psnr(xo, x_gt)
        if is_main_process:
            print(f"Mean PSNR: {psnr.mean()}")
            print(f"PSNR: {psnr}")
        # if cfg.exp.use_single:
            # selected_idx = torch.randint(0, psnr.shape[0], (1,)).item()
            # psnr_list.append(psnr[selected_idx].item())
        # else:
            psnr_list.append(psnr.mean().item())

        LPIPS = compute_lpips(lpips_, xo, x_gt)
        if is_main_process:
            print(f"Mean LPIPS: {LPIPS.mean()}")
            print(f"LPIPS: {LPIPS[:,0,0,0]}")
        # if cfg.exp.use_single:
            # LPIPS returns (N, 1, 1, 1); keep a single randomly-selected particle.
            # lpips_list.append(LPIPS[selected_idx].item())
        # else:
            lpips_list.append(LPIPS.mean().item())

        if cfg.exp.use_wandb and is_main_process:
            wandb.log(
                {
                    "psnr": psnr.mean().item(),
                    "lpips": LPIPS.mean().item(),
                },
                step=it,
            )

            # Log restored images
            image_grid = make_grid(xo.cpu())

            if cfg.exp.deg == "deblur_motion":
                x_deg = postprocess(y_0, cfg)
            else:
                x_deg = postprocess(get_degradation_image(y_0, H, cfg), cfg)
            y_grid = make_grid(x_deg.cpu().clip(0, 1))
            image_payload = {
            "restored_images": wandb.Image(
                image_grid, caption=f"Restored Images Grid (idx={it})"
            ),
            "observed_images": wandb.Image(
                y_grid, caption=f"Observation Images Grid (idx={it})"
            ),
            }
            image_payload["image_idx"] = it
            if cfg.exp.log_img > 0 and it % cfg.exp.log_img == 0:
                wandb.log(image_payload)

        torch.cuda.empty_cache()
        processed += 1
        if is_main_process:
            logger.info(f"Done. You can find the generated images in {output_path_deg}")

    return psnr_list, lpips_list


@hydra.main(
    version_base="1.2", config_path="../configs", config_name="ffhq_stable_diffusion"
)
def main(cfg: DictConfig):
    # Load config file
    cwd = HydraConfig.get().runtime.output_dir
    cfg = init_omega(cfg, cwd)
    rank, local_rank, world_size, is_distributed = get_distributed_info()
    init_distributed(local_rank, world_size)
    is_main_process = rank == 0

    if is_distributed:
        cfg.exp.device = f"cuda:{local_rank}"

    seed_everything(cfg.exp.seed + rank)
    generator = torch.Generator(device="cuda").manual_seed(cfg.exp.seed + rank)

    # Set folder name
    cfg.algo.name_folder = cfg.algo.name

    # Build paths
    deg_path, output_path, evol_path = prepare_paths(cfg)
    dataset_name = cfg.dataset.name

    # Get logger
    logger = get_logger(name="main", cfg=cfg)
    if not cfg.exp.load_img_id:
        if cfg.exp.get("save_evolution", False):
            logger.info(
                "exp.load_img_id=False: disabling intermediate evolution save/eval."
            )
        cfg.exp.save_evolution = False

    if cfg.exp.use_wandb and is_main_process:
        wandb.init(
            project=cfg.exp.get("name", "GradientFlowInverse"),
            name=cfg.exp.get("run_name", f"{cfg.dataset.name}_{cfg.algo.name_folder}_{cfg.exp.deg}_{cfg.model.name}"),
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.exp.entity,
        )
    # override if using sdxl
    # if is_cwgf_dmd_dmd2(cfg) and cfg.dataset.image_size == 512:
    #     logger.info(
    #         "CWGF-DMD with DMD2 selected: overriding cfg.dataset.image_size 512 -> 1024"
    #     )
        # cfg.dataset.image_size = 1024

    pipe, ts = build_pipeline(cfg)
    logger.info(f"Model loaded")

    lpips_ = lpips.LPIPS(net="vgg").cuda()  # best forward score

    # Load degradation model
    algo = build_algo(pipe, cfg)
    H = algo.H
    logger.info(
        f"Degradation model loaded. The experiment corresponds to {cfg.exp.deg}"
    )

    ### UP TO THIS POINT, EVERYTHING IS THE SAME EITHER FOR STABLE DIFFUSION OR OTHER DIFFUSION MODELS.
    max_num_images = cfg.exp.max_num_images
    if algo._is_latent_algo():
        if cfg.exp.load_img_id is True:
            if is_main_process:
                process_single_image(
                    cfg, algo, H, ts, generator, lpips_, output_path, logger
                )
        else:
            process_dataset(
                cfg,
                algo,
                H,
                ts,
                generator,
                lpips_,
                output_path,
                logger,
                dataset_name,
                max_num_images,
                rank=rank,
                world_size=world_size,
                is_main_process=is_main_process,
                resume=cfg.exp.get("resume", False),
                start_idx=cfg.exp.get("start_idx", 0),
            )
    else:
        raise NotImplementedError("Wrong method")

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    if (not cfg.exp.load_img_id) and is_main_process and cfg.exp.get("compute_metrics", True):
        # clean up memory
        del pipe, algo, H, ts, lpips_, generator
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        logger.info(
            "Starting reconstruction metrics evaluation on rank 0 "
            "(fid_device=%s, fid_cpu_before_compute=%s, get_kid=%s).",
            cfg.exp.get("fid_device", cfg.exp.get("device", "cuda")),
            cfg.exp.get("fid_cpu_before_compute", False),
            cfg.exp.get("get_kid", False),
        )
        (
            fid_score,
            kid_mean,
            kid_std,
            mean_psnr,
            mean_lpips,
            num_matched,
            missing_ids,
        ) = compute_reconstruction_metrics(cfg, use_single=cfg.exp.use_single)
        if fid_score is None or mean_psnr is None or mean_lpips is None:
            raise RuntimeError("No matched output images found, cannot compute metrics.")
        if cfg.exp.get("get_kid", False) and (
            kid_mean is None or kid_std is None
        ):
            raise RuntimeError("KID was enabled but could not be computed.")
        logger.info(f"FID score: {fid_score}")
        if kid_mean is not None:
            logger.info(f"KID mean: {kid_mean}")
            logger.info(f"KID std: {kid_std}")
        logger.info(f"Mean PSNR: {mean_psnr}")
        logger.info(f"Mean LPIPS: {mean_lpips}")
        logger.info(f"Matched samples: {num_matched}")
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            suffix = " ..." if len(missing_ids) > 10 else ""
            logger.warning(
                "Missing output folders for %s samples: %s%s",
                len(missing_ids),
                preview,
                suffix,
            )

        if cfg.exp.use_wandb:
            payload = {
                "mean_psnr": mean_psnr,
                "mean_lpips": mean_lpips,
                "fid": fid_score,
            }
            if kid_mean is not None:
                payload["kid"] = kid_mean
                payload["kid_std"] = kid_std
            wandb.log(payload)
            wandb.summary["mean_psnr"] = mean_psnr
            wandb.summary["mean_lpips"] = mean_lpips
            wandb.summary["fid"] = fid_score
            if kid_mean is not None:
                wandb.summary["kid"] = kid_mean
                wandb.summary["kid_std"] = kid_std


if __name__ == "__main__":
    main()
