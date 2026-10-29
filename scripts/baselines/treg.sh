#!/bin/bash

sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo
algo='treg'

# choose degradation
# deg='sr8'
# deg='inp_large_box'
deg='bicubic8'
# deg='deblur_gauss'
# deg='deblur_motion'

# TReg defaults from TReg/solve.py
cfg_scale=4.0
cg_lamb=1e-4
use_an=true
an_lr=8e-4
an_steps=10

# TReg defaults from TReg/configs/solver/treg.yaml and sampler code
num_diffusion_steps=200
cg_iters=5
cg_eps=1e-4
cg_damping=0.0

n_particles=1
save_every=5
obs_weight=1.0
init_method='none' # none, pinv, ddim

# dataset / model
# data_name=afhq512_val
data_name=ffhq512_val
# data_name=imagenet512_val
# data_name=mscoco2017_val
max_num_images=1

use_model='sd15'
fp16=true
use_dinv=true
wandb_logging=${USE_WANDB:-false}
prompt_text="A photo of a face"

# set exp.load_img_id=false to evaluate on dataset images
python src/main.py exp.load_img_id=true\
                dataset=$data_name\
                model=$use_model\
                model.use_fp16=$fp16\
                algo=$algo\
                algo.cfg_scale=$cfg_scale\
                algo.num_diffusion_steps=$num_diffusion_steps\
                algo.obs_weight=$obs_weight\
                algo.n_particles=$n_particles\
                ++algo.init_method=$init_method\
                algo.cg_lamb=$cg_lamb\
                algo.cg_iters=$cg_iters\
                algo.cg_eps=$cg_eps\
                algo.cg_damping=$cg_damping\
                algo.use_DPS=false\
                algo.use_an=$use_an\
                algo.an_lr=$an_lr\
                algo.an_steps=$an_steps\
                exp.prompt="$prompt_text"\
                exp.use_wandb=$wandb_logging\
                exp.deg=$deg\
                exp.max_num_images=$max_num_images\
                exp.seed=42\
                exp.save_every=$save_every\
                exp.save_evolution=true\
                ++exp.use_dinv=$use_dinv\
                "${sd_overrides[@]}"\
                +exp.img_path="data/samples_512" \
                +exp.img_id="00003.png"
