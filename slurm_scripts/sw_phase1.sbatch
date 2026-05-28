#!/bin/bash
#SBATCH --job-name=sw_phase1
#SBATCH --account=156341690590
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/user/u.kt348068/physics_informedPDE/logs/sw_phase1_%j.out
#SBATCH --error=/scratch/user/u.kt348068/physics_informedPDE/logs/sw_phase1_%j.err

mkdir -p /scratch/user/u.kt348068/physics_informedPDE/logs

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
CKPT=/scratch/user/u.kt348068/physics_informedPDE/checkpoints

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow

export WANDB_ENTITY=khiembk
export CUDA_VISIBLE_DEVICES=0

echo "=============================="
echo "Job: sw_phase1 (physics-informed path)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Node: $SLURMD_NODENAME"
echo "=============================="

python $REPO/py/launchers/phase1_learn.py \
    --cfg_path          configs.sw_phase1 \
    --dataset_location  $DATA \
    --output_folder     $CKPT

echo "Done."
