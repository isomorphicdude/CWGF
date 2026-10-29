#!/bin/bash

sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo
algo='cwgf_dmd_s'

data_name='ffhq512_val'
# data_name='imagenet512_val'

# default settings
max_num_images=${MAX_NUM_IMAGES:-1000} # number of dataset images to evaluate
sigma_x0=0.01
warm_steps=0
prompt_opt=true
init_prompt="A photo of a face"
# init_prompt="A photo of a cat"

# algorithm settings
num_diffusion_steps=8
n_particles=1
num_steps=16
timestep_sampling='cyclic' # edm or cyclic
weighting="inv_snr"


deg_list=("deblur_gauss" "deblur_motion" "bicubic8")
n_par_list=(1)

use_model='sd15lora'
fp16=true
use_dinv=true
init_method='none' # ddim, pinv, none
post_method='none' # none, ddim, langevin

use_wandb=${USE_WANDB:-false}
for deg in "${deg_list[@]}"; do
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
    if [[ "$num_diffusion_steps" == 999 ]]; then
        cfg_scale=7.5
        w_t=0.71
        lr_prompt=3.0
        lr_z=1.0
    else
        cfg_scale=1.2
    fi
    study_name=${algo}_${deg}_numdiff${num_diffusion_steps}_${timestep_sampling}_prompt${prompt_opt}
    run_dir="outputs/${study_name}"

    for n_particles in "${n_par_list[@]}"; do
        run_name=${use_model}_${deg}_${n_particles}_${num_diffusion_steps}_${timestep_sampling}_promptopt${prompt_opt}
        python src/main.py \
                hydra.job.chdir=false \
                hydra.run.dir="${run_dir}" \
                dataset=$data_name \
                exp.load_img_id=false \
                exp.root="_exp_eval_${run_name}/" \
                ++algo.kl_use_mean=true \
                exp.name="cwgf_eval" \
                +exp.run_name="${run_name}" \
                model=$use_model \
                model.use_fp16=true \
                exp.sigma_x0=$sigma_x0 \
                exp.prompt="$init_prompt" \
                exp.deg=$deg \
                exp.use_dinv=$use_dinv \
                exp.use_wandb=$use_wandb \
                exp.num_steps=$num_steps \
                exp.save_evolution=false \
                exp.max_num_images=$max_num_images \
                exp.seed=10 \
                algo=$algo \
                algo.n_particles=$n_particles \
                algo.lr_z=$lr_z \
                algo.lr_prompt=$lr_prompt \
                algo.decoder_std=$decoder_std \
                algo.cfg_scale=$cfg_scale \
                algo.w_t="$w_t" \
                algo.weighting=$weighting \
                +algo.mask_idx=15 \
                +algo.prompt_check=false \
                ++algo.num_steps=$num_steps \
                ++algo.num_diffusion_steps=$num_diffusion_steps \
                ++algo.warm_steps=$warm_steps \
                ++algo.prompt_opt=$prompt_opt \
                ++algo.timestep_sampling=$timestep_sampling \
                ++algo.init_method=$init_method \
                ++algo.post_method=$post_method \
                ++algo.num_t_samples=0 \
                "${sd_overrides[@]}"
    done
done
