#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=clip_adamw
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --output=./eval_%x.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mail-user=yuzhchen@tamu.edu
#SBATCH --mail-type=END,FAIL

#First Executable Line
source ~/.bashrc
conda activate clipenv

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"

data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=30
ita_type=clip

ckpt_path="/scratch/user/yuzhchen/633Proj/csce689_iSogCLR/output/clip_adamw_clip_cc3m_g0.8_e30/checkpoint_30.pth"
eval_out="/scratch/user/yuzhchen/633Proj/csce689_iSogCLR/output/eval_clip_adamw"

CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
    --data_path "${data_path}" \
    --ann_path "${ann_path}" \
    --train_file "${data}_train_subset.json" \
    --train_image_root "${train_image_root}" \
    --output_dir "${eval_out}" \
    --use_amp \
    --ita_type "${ita_type}" \
    --tau_init 0.01 \
    --eta_init 0.03 \
    --sched cosine \
    --no-distributed \
    --epochs "${epochs}" \
    --evaluate \
    --checkpoint "${ckpt_path}" \
    --zs_dataset imagenet \
    --zs_datafolder ./datasets/imagenet/val
