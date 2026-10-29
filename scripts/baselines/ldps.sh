#!/bin/bash

sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo
algo='ldps'

# choose degradation
# deg='sr8'
# deg='inp_large_box'
# deg='bicubic16' # for 1024
deg='bicubic8' # for 512
# deg='deblur_gauss'


num_steps=24
save_every=100
lr_z=1.0
cfg_scale=1.0
n_particles=1
obs_weight=1.0


num_diffusion_steps=200
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

use_wandb=${USE_WANDB:-false}
# runs on the dataset images, then computes FID / PSNR / LPIPS
python src/main.py exp.load_img_id=false\
                dataset=$data_name\
                model=$use_model\
                model.use_fp16=$fp16\
                algo=$algo\
                algo.cfg_scale=$cfg_scale\
                algo.lr_z=$lr_z\
                algo.num_diffusion_steps=$num_diffusion_steps\
                algo.obs_weight=$obs_weight\
                algo.n_particles=$n_particles\
                ++algo.init_method=$init_method\
                exp.use_wandb=$use_wandb\
                exp.name="cwgf_baselines" \
                exp.deg=$deg\
                exp.max_num_images=$max_num_images\
                exp.seed=42\
                exp.save_every=$save_every\
                exp.save_evolution=true\
                exp.resume=true\
                ++exp.use_dinv=$use_dinv\
                +exp.img_path="data/samples_512" \
                +exp.img_id="00003.png" \
                "${sd_overrides[@]}"
