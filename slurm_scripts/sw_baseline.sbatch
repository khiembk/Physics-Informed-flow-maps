#!/bin/bash
#SBATCH --job-name=sw_baseline
#SBATCH --account=156341690590
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/user/u.kt348068/physics_informedPDE/logs/sw_baseline_%j.out
#SBATCH --error=/scratch/user/u.kt348068/physics_informedPDE/logs/sw_baseline_%j.err

mkdir -p /scratch/user/u.kt348068/physics_informedPDE/logs

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
RUNS=/scratch/user/u.kt348068/physics_informedPDE/runs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow

export WANDB_ENTITY=khiembk
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job: sw_baseline (linear interpolant)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Node: $SLURMD_NODENAME"
echo "=============================="

python $REPO/py/launchers/pde_learn.py \
    --cfg_path          configs.sw_baseline \
    --dataset_location  $DATA \
    --output_folder     $RUNS

echo "Done."
