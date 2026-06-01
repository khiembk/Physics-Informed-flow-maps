#!/usr/bin/env bash
#SBATCH --job-name=pcfm_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job  : $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time : $(date)"
echo "=============================="

python baselines/run_pcfm.py \
    --dataset_location $DATA \
    --output_json      $DATA/results/pcfm_eval.json \
    --n_steps          10 \
    --systems          shallow_water_2d navier_stokes_2d mhd_2d multiphase_2d euler_2d

echo "=============================="
echo "Done : $(date)"
echo "=============================="
