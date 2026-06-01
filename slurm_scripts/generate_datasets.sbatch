#!/usr/bin/env bash
#SBATCH --job-name=gen_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

CONFIGS=(
  py/configs/data/shallow_water_2d.yaml
  py/configs/data/navier_stokes_2d.yaml
  py/configs/data/mhd_2d.yaml
  py/configs/data/multiphase_2d.yaml
  py/configs/data/euler_2d.yaml
)
SYSTEMS=(shallow_water_2d navier_stokes_2d mhd_2d multiphase_2d euler_2d)

CFG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
SYS=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

echo "=============================="
echo "Generating: $SYS"
echo "Config:     $CFG"
echo "=============================="

python py/generate_dataset.py \
    --config    $CFG \
    --output_dir $DATA \
    --batch_size 64

echo "Done: $(date)"
