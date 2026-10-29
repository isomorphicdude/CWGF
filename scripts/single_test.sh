#!/bin/bash

sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo
algo='cwgf_dmd_s'

# choose degradation (tuned hyperparameters below for these)
# deg='bicubic8' # for 512
# deg='deblur_motion'
deg='deblur_gauss'


num_steps=16
save_every=1
w_t=1.0
lr_z=1.08

lr_prompt=1.2
n_particles=1 # for linear, 4 particles work much better
num_diffusion_steps=8
num_t_samples=0
warm_steps=0
sigma_x0=0.01
data_name=ffhq512_val
# data_name=imagenet512_val
max_num_images=10 # for dataset processing

use_model='sd15lora'

fp16=true
use_temp=false
use_dinv=true
init_method='none' # ddim, pinv, none
post_method='none' # none, ddim, langevin
prompt_opt=true
init_prompt='A photo of a face'
timestep_sampling='cyclic' # edm or cyclic or decreasing
weighting="inv_snr" # inv_snr or edm

exp_root="_single_exp_${deg}_${n_particles}_${use_model}"

# w_t are spaced equally in [0.1, 0.9] with last value 0.9
if [[ "$deg" == "deblur_gauss" ]]; then
    lr_z=1.0
    lr_prompt=0.66
    w_t='[0.1,0.21162791,0.32325581,0.43488372,0.56511628,0.67674419,0.78837209,0.9]'
    decoder_std=0.08
elif [[ "$deg" == "deblur_motion" ]]; then
    lr_z=1.0
    lr_prompt=0.26
    w_t='[0.1,0.21162791,0.32325581,0.43488372,0.56511628,0.67674419,0.78837209,0.9]'
    decoder_std=0.05
elif [[ "$deg" == "bicubic8" || "$deg" == "sr8" || "$deg" == "bicubic16" ]]; then
    if [[ "$use_model" == "sd15lora" ]]; then
        lr_z=1.0
        lr_prompt=0.03
        w_t='[0.1,0.21162791,0.32325581,0.43488372,0.56511628,0.67674419,0.78837209,0.9]'
        decoder_std=0.25
    elif [[ "$use_model" == "dmd2" || "$use_model" == "sdxllora" ]]; then
        lr_z=1.0
        lr_prompt=0.03
        w_t=1.0
        decoder_std=0.09
    fi
else
    echo "No tuned hyperparameters for deg=${deg} in $0" >&2
    exit 1
fi

prompt_check=false
cfg_scale=1.2 # use 7.5 for nondistilled and 1.2 for distilled
use_wandb=${USE_WANDB:-false}
# set exp.load_img_id to false to use dataset images
python src/main.py exp.load_img_id=true\
                exp.root="${exp_root}_prompt_${prompt_opt}"\
                dataset=$data_name\
                model=$use_model\
                model.use_fp16=$fp16\
                algo=$algo\
                +algo.do_classifier_free_guidance=true\
                algo.decoder_std="$decoder_std"\
                algo.cfg_scale=$cfg_scale\
                algo.w_t="$w_t"\
                algo.lr_z=$lr_z\
                algo.timestep_sampling=$timestep_sampling\
                algo.n_particles=$n_particles\
                +algo.mask_idx=15 \
                ++algo.num_t_samples=$num_t_samples\
                ++algo.num_steps=$num_steps\
                ++algo.num_diffusion_steps=$num_diffusion_steps\
                ++algo.warm_steps=$warm_steps\
                ++algo.prompt_opt=$prompt_opt\
                ++algo.weighting=$weighting \
                ++algo.prompt_check=$prompt_check \
                ++algo.use_temp=$use_temp\
                ++algo.init_method=$init_method\
                ++algo.post_method=$post_method\
                ++algo.lr_prompt=$lr_prompt\
                exp.use_wandb=$use_wandb\
                exp.name="cwgf"\
                exp.num_steps=$num_steps\
                exp.deg=$deg\
                exp.max_num_images=$max_num_images\
                exp.seed=1\
                exp.prompt="${init_prompt}" \
                exp.save_every=$save_every\
                exp.save_evolution=true\
                exp.sigma_x0=$sigma_x0\
                ++exp.use_dinv=$use_dinv\
                +exp.img_path="data/samples_512" \
                +exp.img_id="00003.png" \
                "${sd_overrides[@]}"
