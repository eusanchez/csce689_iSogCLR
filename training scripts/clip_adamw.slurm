#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=clip_adamw_train
#SBATCH --time=15:00:00
#SBATCH --mem=40G
#SBATCH --output=./clip_adamw_output_%x.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1     
#SBATCH --partition=gpu
#SBATCH --mail-user=yuzhchen@tamu.edu
#SBATCH --mail-type=END,FAIL

echo "Running on: $(hostname)"
echo "Starting at: $(date)"

#First Executable Line
source ~/.bashrc
conda activate clipenv

cd /scratch/user/yuzhchen/633Proj/csce689_iSogCLR

export PYTHONPATH="$PYTHONPATH:./bimodal_exps"

data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=30
ita_type=clip
#text_encoder=checkpoints/distilbert-base-uncased 

CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
    --data_path "${data_path}" \
    --ann_path "${ann_path}" \
    --train_file "${train_file}" \
    --train_image_root "${train_image_root}" \
    --output_dir "output/${ita_type}_adamw_clip_${data}_g${gamma}_e${epochs}" \
    --use_amp \
    --ita_type "${ita_type}" \
    --tau_init 0.01 \
   # --sogclr_gamma "${gamma}" \
    --eta_init 0.03 \
    --sched cosine \
    --no-distributed \
    --epochs "${epochs}" \
   # --text_encoder "${text_encoder}"

echo "Finished at: $(date)"
