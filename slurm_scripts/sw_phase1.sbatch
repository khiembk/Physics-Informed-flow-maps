#!/usr/bin/env bash
#SBATCH --job-name=sw_phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── paths ──────────────────────────────────────────────────────────────────
REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
CKPT=/scratch/user/u.kt348068/physics_informedPDE/checkpoints

# ── environment ────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

export WANDB_ENTITY=khiembk
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job  : $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node : $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time : $(date)"
echo "=============================="

# ── training ───────────────────────────────────────────────────────────────
# To resume: add --continue_from $CKPT/phase1_checkpoints/sw/phi_stepXXXX.npz

python py/launchers/phase1_learn.py \
    --cfg_path         configs.sw_phase1 \
    --dataset_location $DATA \
    --output_folder    $CKPT

echo "=============================="
echo "Done : $(date)"
echo "=============================="
