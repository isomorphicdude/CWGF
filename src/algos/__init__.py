# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

from .base_algo import AlgoBase


# Algorithms are imported lazily, so that the extra dependencies of a baseline
# (e.g. OpenAI CLIP for TReg) are only loaded when that baseline is run.
def build_algo(cg_model, cfg):
    if cfg.algo.name == "ldps":
        from .ldps import LDPS
        return LDPS(cg_model, cfg)
    elif cfg.algo.name == "psld":
        from .psld import PSLD
        return PSLD(cg_model, cfg)
    elif cfg.algo.name in ["ldps_p2l", "ldps-p2l"]:
        from .ldps_p2l import LDPSP2L
        return LDPSP2L(cg_model, cfg)
    elif cfg.algo.name == "treg":
        from .treg import TREG
        return TREG(cg_model, cfg)
    elif cfg.algo.name == "cwgf_dmd_s":
        from .cwgf_dmd_simple import CWGFDMDSimple
        return CWGFDMDSimple(cg_model, cfg)
    elif cfg.algo.name == "latino":
        from .latino import LATINO
        return LATINO(cg_model, cfg)
    elif cfg.algo.name == "latino_d":
        from .latino_d import LATINO_D
        return LATINO_D(cg_model, cfg)
    else:
        raise ValueError(f"No algorithm named {cfg.algo.name}")
