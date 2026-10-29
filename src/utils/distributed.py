import logging
import os
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_distributed_info() -> tuple[int, int, int, bool]:
    rank = _get_env_int("RANK", 0)
    local_rank = _get_env_int("LOCAL_RANK", 0)
    world_size = max(1, _get_env_int("WORLD_SIZE", 1))
    return rank, local_rank, world_size, world_size > 1


def init_distributed(local_rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE > 1.")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        raise RuntimeError("Distributed sampling on CPU is not supported.")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")


def gather_scalar_list(values: list[float]) -> list[float]:
    if not (dist.is_available() and dist.is_initialized()):
        return values

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, values)
    merged = []
    for shard in gathered:
        if shard:
            merged.extend(shard)
    return merged


# def init_processes(rank, size, fn, cfg, cwd):
#     """Initialize the distributed environment."""
#     try:
#         cfg = OmegaConf.create(cfg)
#         OmegaConf.set_struct(cfg, False)
#         cfg.cwd = cwd

#         os.environ["MASTER_ADDR"] = cfg.dist.master_address
#         os.environ["MASTER_PORT"] = str(cfg.dist.port)
#         dist.init_process_group(
#             backend=cfg.dist.backend, init_method="env://", rank=rank, world_size=size
#         )
#         fn(cfg)
#         dist.barrier()
#         dist.destroy_process_group()
#     except Exception:
#         logging.error(traceback.format_exc())
#         dist.destroy_process_group()


# def common_init(rank, seed):
#     # we use different seeds per gpu. But we sync the weights after model initialization.
#     torch.manual_seed(rank + seed)
#     np.random.seed(rank + seed)
#     torch.cuda.manual_seed(rank + seed)
#     torch.cuda.manual_seed_all(rank + seed)
#     torch.backends.cudnn.benchmark = True


# def broadcast_params(params, is_distributed):
#     if is_distributed:
#         for param in params:
#             dist.broadcast(param.data, src=0)


# def get_logger(name=None, cfg=None):
#     load_path = os.path.join(cfg.cwd, ".hydra/hydra.yaml")
#     hydra_conf = OmegaConf.load(load_path)
#     logging.config.dictConfig(
#         OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True)
#     )

#     return logging.getLogger(name)


# def get_results_file(cfg, logger):
#     if dist.get_rank() == 0 or not dist.is_available():
#         results_root = os.path.join(cfg.exp.root, "results")
#         os.makedirs(results_root, exist_ok=True)
#         if "/" in cfg.results:
#             results_dir = "/".join(cfg.results.split("/")[:-1])
#             results_dir = os.path.join(results_root, results_dir)
#             logger.info(f"Creating directory {results_dir}")
#             os.makedirs(results_dir, exist_ok=True)
#         results_file = f"{results_root}/{cfg.results}.yaml"
#     return results_file


# def distributed(func):
#     def wrapper(cfg):
#         cwd = HydraConfig.get().runtime.output_dir
#         if cfg.dist.num_processes_per_node < 0:
#             size = torch.cuda.device_count()
#             cfg.dist.num_processes_per_node = size
#         else:
#             size = cfg.dist.num_processes_per_node
#         if size > 1:
#             num_proc_node = cfg.dist.num_proc_node
#             num_process_per_node = cfg.dist.num_processes_per_node
#             world_size = num_proc_node * num_process_per_node
#             mp.spawn(
#                 init_processes,
#                 args=(world_size, func, cfg, cwd),
#                 nprocs=world_size,
#                 join=True,
#             )
#         else:
#             init_processes(0, size, func, cfg, cwd)

#     return wrapper
