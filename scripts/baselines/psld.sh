#!/bin/bash

sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo
algo='psld'

# choose degradation
# deg='sr8'
# deg='inp_large_box'
# deg='bicubic16' # for 1024
deg='bicubic8' # for 512
# deg='deblur_gauss'
# deg='deblur_motion'


num_steps=24
save_every=50
lr_z=1.0
cfg_scale=1.0
n_particles=1
obs_weight=1.0
psld_gamma=0.1
psld_omega=1.0


num_diffusion_steps=200
warm_steps=0
sigma_x0=0.01
# data_name=afhq512_val
data_name=ffhq512_val
# data_name=imagenet512_val
# data_name=mscoco2017_val
max_num_images=100 # for dataset processing

use_model='sd15'
# use_model='dmd2'
fp16=true
use_temp=false
use_dinv=true
init_method='none' # none, pinv, ddim

# PSLD hyperparameters from official github repo (256x256, ffhq):
# omega for measurement error and gamma for inpaint error
# - default: guidance scale=7.5, ddim steps=1000
# - super-resolution: gamma=1e-1, omega=9e-1
# - inpainting (box mask): gamma=1e-2, omega=1e-1
# - gaussian deblur/motion deblur: gamma=1e-1, omega=1e0
if [[ "$deg" == bicubic* || "$deg" == sr* ]]; then
    cfg_scale=1.2
    obs_weight=0.9
    psld_gamma=0.1
    psld_omega=0.9
    num_diffusion_steps=999
elif [[ "$deg" == inp* ]]; then
    cfg_scale=1.2
    obs_weight=0.1
    psld_gamma=0.01
    psld_omega=0.1
    num_diffusion_steps=999
elif [[ "$deg" == "deblur_gauss" || "$deg" == "deblur_motion" ]]; then
    cfg_scale=1.2
    obs_weight=1.0
    psld_gamma=0.1
    psld_omega=1.0
    num_diffusion_steps=999
fi

wandb_logging=${USE_WANDB:-false}
# multi-GPU: NUM_GPUS=2 bash scripts/baselines/psld.sh
torchrun --nproc_per_node=${NUM_GPUS:-1} src/main.py exp.load_img_id=false\
                dataset=$data_name\
                model=$use_model\
                model.use_fp16=$fp16\
                algo=$algo\
                algo.cfg_scale=$cfg_scale\
                algo.lr_z=$lr_z\
                algo.num_diffusion_steps=$num_diffusion_steps\
                algo.obs_weight=$obs_weight\
                algo.psld_gamma=$psld_gamma\
                algo.psld_omega=$psld_omega\
                algo.n_particles=$n_particles\
                ++algo.init_method=$init_method\
                exp.use_wandb=$wandb_logging\
                exp.name="cwgf_baselines" \
                exp.deg=$deg\
                exp.max_num_images=$max_num_images\
                exp.seed=42\
                exp.save_every=$save_every\
                exp.save_evolution=true\
                ++exp.use_dinv=$use_dinv\
                +exp.img_path="data/samples_512" \
                +exp.img_id="00003.png" \
                "${sd_overrides[@]}"
