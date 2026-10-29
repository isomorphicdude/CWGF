#!/bin/bash
sd_overrides=()
if [ -n "${CWGF_SD_LOCAL_ROOT:-}" ]; then
  sd_overrides+=("exp.sd_local_root=${CWGF_SD_LOCAL_ROOT}")
fi

# choose algo
algo='cwgf_dmd_s'

# choose degradation: pass one as first argument (e.g. `bash scripts/adv_test.sh bicubic8`),
# otherwise deblur_gauss, deblur_motion and bicubic8 are all run
if [ -n "${1:-}" ]; then
    deg_list=("$1")
else
    deg_list=("deblur_gauss" "deblur_motion" "bicubic8")
fi

# default settings
max_num_images=1000 # for dataset processing
sigma_x0=0.01
warm_steps=0
prompt_opt=true

prompt_type="neg"

if [[ "$prompt_type" == "pos" ]]; then
    init_prompt="A photo of a face"
elif [[ "$prompt_type" == "neg" ]]; then
    init_prompt="A photo of a cat"
else
    init_prompt="A photo of a face"
fi

# algorithm settings
num_diffusion_steps=8
num_t_samples=1
n_particles=1
num_steps=16
save_every=1
log_img=100
timestep_sampling='cyclic'
weighting="inv_snr"

# default fallback values
lr_z=1.0
lr_prompt=1.0
decoder_std=0.2
w_t=1.0
project_radius=15.0

data_name='ffhq512_val'

use_model='sd15lora'
# use_model='dmd2'
fp16=true
use_dinv=true


for deg in "${deg_list[@]}"; do
if [[ "$deg" == "deblur_gauss" ]]; then
    lr_z=1.00
    lr_prompt=0.66
    w_t='[0.1,0.21162791,0.32325581,0.43488372,0.56511628,0.67674419,0.78837209,0.9]'
    decoder_std=0.08
elif [[ "$deg" == "deblur_motion" ]]; then
    lr_z=1.0
    lr_prompt=0.26
    w_t='[0.1,0.21162791,0.32325581,0.43488372,0.56511628,0.67674419,0.78837209,0.9]'
    decoder_std=0.05
elif [[ "$deg" == "bicubic8" || "$deg" == "sr8" ]]; then
    lr_z=1.0
    decoder_std=0.25
    w_t='[0.1,0.21162791,0.32325581,0.43488372,0.56511628,0.67674419,0.78837209,0.9]' # linear
    lr_prompt=0.03
fi

cfg_scale=1.2
if [[ "$prompt_type" == "neg" ]]; then
    lr_prompt=3.0
fi

study_name="${algo}_${deg}_cat_test"
run_dir="outputs/${study_name}"
run_name="adv_${num_t_samples}-${algo}_${deg}_${timestep_sampling}_${num_diffusion_steps}_${num_steps}_${n_particles}_${project_radius}"

use_wandb=${USE_WANDB:-false}
prompt_check=false

NUM_REP=1
for seed in $(seq 1 $NUM_REP); do
    seed=$((seed + 9))
    echo "Running seed ${seed}..."
    # python src/eval_fid_from_exp.py \
    python src/main.py \
            exp.load_img_id=true \
            hydra.job.chdir=false \
            ++exp.resume=false \
            hydra.run.dir="${run_dir}" \
            exp.root="_exp_new_${run_name}/" \
            exp.name="cwgf_cat_test" \
            +exp.run_name="${run_name}" \
            dataset=$data_name \
            model=$use_model \
            model.use_fp16=$fp16 \
            exp.sigma_x0=$sigma_x0 \
            exp.prompt="$init_prompt" \
            exp.deg=$deg \
            exp.use_dinv=$use_dinv \
            exp.use_wandb=$use_wandb \
            exp.num_steps=$num_steps \
            exp.save_every=$save_every \
            exp.save_evolution=true \
            exp.max_num_images=$max_num_images \
            exp.seed=$seed \
            algo=$algo \
            algo.n_particles=$n_particles \
            algo.prompt_project_radius=$project_radius \
            algo.lr_z=$lr_z \
            algo.lr_prompt=$lr_prompt \
            algo.decoder_std=$decoder_std \
            algo.w_t="$w_t" \
            algo.weighting=$weighting \
            algo.cfg_scale=$cfg_scale \
            +algo.mask_idx=15 \
            +algo.prompt_check=$prompt_check \
            ++algo.num_steps=$num_steps \
            ++algo.num_diffusion_steps=$num_diffusion_steps \
            ++algo.num_t_samples=$num_t_samples \
            ++algo.warm_steps=$warm_steps \
            ++algo.prompt_opt=$prompt_opt \
            ++algo.timestep_sampling=$timestep_sampling \
            +exp.img_path="data/samples_512" \
            +exp.img_id="00003.png" \
            "${sd_overrides[@]}"
done
done
