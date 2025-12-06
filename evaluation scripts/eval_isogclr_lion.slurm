#!/bin/bash

## SLURM SETTINGS
#SBATCH --job-name=isogclr_lion
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --output=./eval_%x.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-user=yuzhchen@tamu.edu
#SBATCH --mail-type=END,FAIL

echo "Running on: $(hostname)"
echo "Starting at: $(date)"

# Load environment
source ~/.bashrc
conda activate clipenv

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"

# Paths
data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
gamma=0.8
epochs=30

# iSogCLR loss (the working one)
ita_type=isogclr_new

# CHECKPOINT PATH for iSogCLR + AdamW run
ckpt_path="/scratch/user/yuzhchen/633Proj/csce689_iSogCLR/output/isogclr_new_lion_iso_cc3m_g0.8_e30/checkpoint_30.pth"

# OUTPUT PATH FOR EVAL RESULTS
eval_out="/scratch/user/yuzhchen/633Proj/csce689_iSogCLR/output/eval_isogclr_lion"

CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
    --data_path "${data_path}" \
    --ann_path "${ann_path}" \
    --train_file "${data}_train_subset.json" \
    --train_image_root "${train_image_root}" \
    --output_dir "${eval_out}" \
    --use_amp \
    --ita_type "${ita_type}" \
    --tau_init 0.01 \
    --sogclr_gamma "${gamma}" \
    --eta_init 0.03 \
    --sched cosine \
    --no-distributed \
    --epochs "${epochs}" \
    --evaluate \
    --checkpoint "${ckpt_path}" \
    --zs_dataset imagenet \
    --zs_datafolder ./datasets/imagenet/val

echo "Finished at: $(date)"
