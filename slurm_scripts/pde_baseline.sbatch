#!/usr/bin/env bash
#SBATCH --job-name=pde_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

SYSTEMS=(shallow_water_2d navier_stokes_2d mhd_2d multiphase_2d)
CFGS=(configs.sw_baseline configs.ns_baseline configs.mhd_baseline configs.multi_baseline)

SYSTEM=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}
CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
RUNS=/scratch/user/u.kt348068/physics_informedPDE/runs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job   : $SLURM_JOB_NAME  [$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID]"
echo "System: $SYSTEM"
echo "Node  : $SLURMD_NODENAME"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================="

python py/launchers/pde_learn.py \
    --cfg_path         $CFG \
    --dataset_location $DATA \
    --output_folder    $RUNS

echo "=============================="
echo "Done : $(date)"
echo "=============================="
