#!/usr/bin/env bash
#SBATCH --job-name=pde_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

SYSTEMS=(shallow_water_2d navier_stokes_2d mhd_2d multiphase_2d)
SYSTEM=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
RUNS=/scratch/user/u.kt348068/physics_informedPDE/runs
RESULTS=/scratch/user/u.kt348068/physics_informedPDE/results

PHASE2_CKPT=$RUNS/${SYSTEM}_phase2/params_step6000.npz
BASELINE_CKPT=$RUNS/${SYSTEM}_baseline/params_step10000.npz

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job   : $SLURM_JOB_NAME  [$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID]"
echo "System: $SYSTEM"
echo "=============================="

mkdir -p $RESULTS

python py/evaluate_pde.py \
    --system           $SYSTEM \
    --dataset_location $DATA \
    --phase2_ckpt      $PHASE2_CKPT \
    --baseline_ckpt    $BASELINE_CKPT \
    --n_steps          1 5 10 \
    --output_json      $RESULTS/${SYSTEM}_eval.json

echo "=============================="
echo "Done : $(date)"
echo "=============================="
