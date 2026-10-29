#!/bin/bash

sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo
algo='ldps_p2l'

# choose degradation
# deg='sr8'
# deg='inp_large_box'
# deg='bicubic16' # for 1024
deg='bicubic8' # for 512
# deg='deblur_gauss'
# deg='deblur_motion'


num_steps=24
save_every=5
lr_z=1.0
lr_z_mult=0.0
cfg_scale=1.0
n_particles=1
obs_weight=1.0
z_update='adam'
prompt_opt_steps=1
prompt_lr=1e-4
latent_lr=0.05
latent_lr_mult=0.0
adam_beta1=0.9
adam_beta2=0.999
prox_gamma=1.0
prox_every=8
prox_phase=1


num_diffusion_steps=999
warm_steps=0
sigma_x0=0.01
# data_name=afhq512_val
data_name=ffhq512_val
# data_name=imagenet512_val
# data_name=mscoco2017_val
max_num_images=1000 # for dataset processing

use_model='sd15'
# use_model='dmd2'
fp16=true
use_temp=false
use_dinv=true
init_method='none' # none, pinv, ddim

# Table 6 https://arxiv.org/pdf/2310.01110.
# Mapping in this codebase:
# - rho_t = algo.latent_lr + algo.latent_lr_alpha_bar_mult * alpha_bar_t (Adam)
# - rho_t = algo.lr_z + algo.lr_z_alpha_bar_mult * alpha_bar_t (GD)
# - gamma = algo.prox_every
# - lambda = algo.obs_weight
# - K = algo.prompt_opt_steps
# - learning rate = algo.prompt_lr
if [[ "$data_name" == ffhq* ]]; then
    if [[ "$deg" == bicubic* || "$deg" == sr* ]]; then
        cfg_scale=1.2
        obs_weight=1.0
        z_update='gd'
        lr_z=1.0
        lr_z_mult=0.0
        prox_every=4
        prox_phase=0
        prompt_opt_steps=5
        prompt_lr=1e-4
        num_diffusion_steps=999
    elif [[ "$deg" == "deblur_gauss" ]]; then
        cfg_scale=1.2
        obs_weight=1.0
        z_update='adam'
        latent_lr=0.05
        latent_lr_mult=0.0
        prox_every=4
        prox_phase=0
        prompt_opt_steps=5
        prompt_lr=1e-4
        num_diffusion_steps=999
    elif [[ "$deg" == "deblur_motion" ]]; then
        cfg_scale=1.2
        obs_weight=1.0
        z_update='adam'
        latent_lr=0.05
        latent_lr_mult=0.0
        prox_every=5
        prox_phase=0
        prompt_opt_steps=3
        prompt_lr=5e-5
        num_diffusion_steps=999
    elif [[ "$deg" == "inp_large_box" ]]; then
        cfg_scale=1.2
        obs_weight=0.1
        z_update='adam'
        latent_lr=0.05
        latent_lr_mult=0.0
        prox_every=3
        prox_phase=0
        prompt_opt_steps=1
        prompt_lr=1e-4
        num_diffusion_steps=999
    fi
elif [[ "$data_name" == imagenet* ]]; then
    if [[ "$deg" == bicubic* || "$deg" == sr* ]]; then
        cfg_scale=1.2
        obs_weight=1.0
        z_update='gd'
        lr_z=0.0
        lr_z_mult=15.0
        prox_every=4
        prox_phase=0
        prompt_opt_steps=3
        prompt_lr=1e-5
        num_diffusion_steps=999
    elif [[ "$deg" == "deblur_gauss" ]]; then
        cfg_scale=1.2
        obs_weight=1.0
        z_update='gd'
        lr_z=0.0
        lr_z_mult=1.0
        prox_every=4
        prox_phase=0
        prompt_opt_steps=3
        prompt_lr=1e-4
        num_diffusion_steps=999
    elif [[ "$deg" == "deblur_motion" ]]; then
        cfg_scale=1.2
        obs_weight=1.0
        z_update='adam'
        latent_lr=0.1
        latent_lr_mult=0.0
        prox_every=5
        prox_phase=0
        prompt_opt_steps=3
        prompt_lr=1e-5
        num_diffusion_steps=999
    elif [[ "$deg" == "inp_large_box" ]]; then
        cfg_scale=1.2
        obs_weight=0.1
        z_update='gd'
        lr_z=0.5
        lr_z_mult=0.0
        prox_every=3
        prox_phase=0
        prompt_opt_steps=1
        prompt_lr=1e-4
        num_diffusion_steps=999
    fi
fi

if [[ "$data_name" == imagenet* ]]; then
    init_prompt=""
else
    init_prompt="A photo of a cat"
fi

num_diffusion_steps=200
use_wandb=${USE_WANDB:-false}
# Run single experiment (decoder_list is passed as a list literal to Hydra).
# set exp.load_img_id to false to use dataset images
python src/main.py exp.load_img_id=false\
                dataset=$data_name\
                model=$use_model\
                model.use_fp16=$fp16\
                algo=$algo\
                algo.cfg_scale=$cfg_scale\
                algo.lr_z=$lr_z\
                algo.lr_z_mult=$lr_z_mult\
                algo.obs_weight=$obs_weight\
                algo.num_diffusion_steps=$num_diffusion_steps\
                algo.n_particles=$n_particles\
                ++algo.init_method=$init_method\
                algo.prompt_opt_steps=$prompt_opt_steps\
                algo.prompt_lr=$prompt_lr\
                algo.latent_lr=$latent_lr\
                algo.latent_lr_mult=$latent_lr_mult\
                ++algo.z_update=$z_update\
                algo.adam_beta1=$adam_beta1\
                algo.adam_beta2=$adam_beta2\
                algo.prox_gamma=$prox_gamma\
                ++algo.prox_every=$prox_every\
                ++algo.prox_phase=$prox_phase\
                exp.use_wandb=$use_wandb\
                exp.name="cwgf_baselines" \
                exp.deg=$deg\
                exp.max_num_images=$max_num_images\
                exp.root="_exp_p2l_${data_name}_${deg}/"\
                exp.seed=42\
                exp.prompt="$init_prompt" \
                exp.save_every=$save_every\
                exp.save_evolution=true\
                ++exp.use_dinv=$use_dinv\
                +exp.img_path="data/samples_512" \
                +exp.img_id="00003.png" \
                "${sd_overrides[@]}"
