#!/usr/bin/env bash
#SBATCH --job-name=pde_p1dyn_p2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

SYSTEMS=(shallow_water_2d navier_stokes_2d mhd_2d multiphase_2d euler_2d)
P1_CFGS=(configs.sw_phase1 configs.ns_phase1 configs.mhd_phase1 configs.multi_phase1 configs.euler_phase1)
P2_CFGS=(configs.sw_phase2 configs.ns_phase2 configs.mhd_phase2 configs.multi_phase2 configs.euler_phase2)

SYSTEM=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}
P1_CFG=${P1_CFGS[$SLURM_ARRAY_TASK_ID]}
P2_CFG=${P2_CFGS[$SLURM_ARRAY_TASK_ID]}

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
RUNS=/scratch/user/u.kt348068/physics_informedPDE/runs_dynamic
CKPT=/scratch/user/u.kt348068/physics_informedPDE/checkpoints_dynamic
PHI_CKPT=$CKPT/phase1_dynamic_checkpoints/${SYSTEM}/phi_final.npz

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

echo ">>> Phase 1 DYNAMIC: constraint + dynamics mismatch"
python py/launchers/phase1_dynamic_learn.py \
    --cfg_path         $P1_CFG \
    --dataset_location $DATA \
    --output_folder    $CKPT

echo ">>> Phase 2: 10K steps on frozen dynamic phi"
python py/launchers/phase2_learn.py \
    --cfg_path         $P2_CFG \
    --dataset_location $DATA \
    --output_folder    $RUNS \
    --phi_ckpt         $PHI_CKPT

echo "=============================="
echo "Done : $(date)"
echo "=============================="
