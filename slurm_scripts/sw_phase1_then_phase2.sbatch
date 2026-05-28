#!/usr/bin/env bash
#SBATCH --job-name=sw_p1_p2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── paths ──────────────────────────────────────────────────────────────────
REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
RUNS=/scratch/user/u.kt348068/physics_informedPDE/runs
CKPT=/scratch/user/u.kt348068/physics_informedPDE/checkpoints

PHI_CKPT=$CKPT/phase1_checkpoints/shallow_water_2d/phi_final.npz

# ── environment ────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

export WANDB_ENTITY=khiembk
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job  : $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node : $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time : $(date)"
echo "=============================="

# ── Phase 1: 500 steps (early stop before L_sm blowup) ────────────────────
echo ""
echo ">>> Phase 1: 500 steps, phi path encoding"
python py/launchers/phase1_learn.py \
    --cfg_path         configs.sw_phase1 \
    --dataset_location $DATA \
    --output_folder    $CKPT

echo ""
echo ">>> Phase 1 done. Phi checkpoint: $PHI_CKPT"

# ── Phase 2: 10K steps on physics-informed path ───────────────────────────
echo ""
echo ">>> Phase 2: 10K steps, flow map on frozen phi"
python py/launchers/phase2_learn.py \
    --cfg_path         configs.sw_phase2 \
    --dataset_location $DATA \
    --output_folder    $RUNS \
    --phi_ckpt         $PHI_CKPT

echo "=============================="
echo "Done : $(date)"
echo "=============================="
