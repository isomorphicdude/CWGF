#!/bin/bash

sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo / model
algo='latino'
use_model='dmd2'   # alternative: sd15 for the LCM SD1.5 variant
fp16=true

# choose degradation
deg='bicubic16'
# deg='inp_large_box'
# deg='deblur_gauss'
# deg='deblur_motion'

num_diffusion_steps=8
cfg_scale=1.0
sigma_x0=0.01
save_every=1
n_particles=1
init_method='none' # none, pinv

# dataset
data_name=ffhq512_val
# data_name=imagenet512_val
# data_name=mscoco2017_val
max_num_images=1000

use_dinv=true
wandb_logging=${USE_WANDB:-false}
prompt_text="A sharp photo of a face."
exproot="_single_exp_${deg}_1"
# exp.root="_exp_single_new/"

python src/main.py exp.load_img_id=true\
                dataset=$data_name\
                model=$use_model\
                model.use_fp16=$fp16\
                algo=$algo\
                algo.cfg_scale=$cfg_scale\
                algo.num_diffusion_steps=$num_diffusion_steps\
                algo.n_particles=$n_particles\
                algo.sigma_y=$sigma_x0\
                ++algo.init_method=$init_method\
                exp.prompt="$prompt_text"\
                exp.root=$exproot \
                exp.use_wandb=$wandb_logging\
                exp.deg=$deg\
                exp.sigma_x0=$sigma_x0\
                exp.name="cwgf_baselines"\
                exp.max_num_images=$max_num_images\
                exp.seed=42\
                exp.save_every=$save_every\
                exp.save_evolution=true\
                ++exp.use_dinv=$use_dinv\
                "${sd_overrides[@]}"\
                +exp.img_path="data/samples_512" \
                +exp.img_id="00003.png"
