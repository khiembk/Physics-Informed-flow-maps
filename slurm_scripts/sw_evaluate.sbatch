#!/usr/bin/env bash
#SBATCH --job-name=sw_evaluate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── paths ──────────────────────────────────────────────────────────────────
REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
RUNS=/scratch/user/u.kt348068/physics_informedPDE/runs
RESULTS=/scratch/user/u.kt348068/physics_informedPDE/results

PHASE2_CKPT=$RUNS/sw_phase2_pi_path/params_step6000.npz
BASELINE_CKPT=$RUNS/sw_baseline_linear/params_step10000.npz

# ── environment ────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job  : $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node : $SLURMD_NODENAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Time : $(date)"
echo "Phase2   : $PHASE2_CKPT"
echo "Baseline : $BASELINE_CKPT"
echo "=============================="

mkdir -p $RESULTS

python py/evaluate_sw.py \
    --dataset_location $DATA \
    --phase2_ckpt      $PHASE2_CKPT \
    --baseline_ckpt    $BASELINE_CKPT \
    --n_steps          1 5 10 \
    --output_json      $RESULTS/sw_eval.json

echo "=============================="
echo "Done : $(date)"
echo "=============================="
