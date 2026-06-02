#!/usr/bin/env bash
#SBATCH --job-name=pde_eval_dyn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Compare: Trivial | Linear | PCFM+Linear | Phase2-Static | Phase2-Dynamic
SYSTEMS=(shallow_water_2d navier_stokes_2d mhd_2d multiphase_2d euler_2d)
SYSTEM=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}

REPO=/home/u.kt348068/khiemtt/Physics-Informed-flow-maps
DATA=/scratch/user/u.kt348068/physics_informedPDE
RUNS=$DATA/runs
RUNS_DYN=$DATA/runs_dynamic
RESULTS=$DATA/results_dynamic

BL_MAP_0=sw_baseline_linear
BL_MAP=(sw_baseline_linear navier_stokes_2d_baseline mhd_2d_baseline multiphase_2d_baseline euler_2d_baseline)
P2_MAP_0=sw_phase2_pi_path
P2_MAP=(sw_phase2_pi_path navier_stokes_2d_phase2 mhd_2d_phase2 multiphase_2d_phase2 euler_2d_phase2)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phyflow
cd $REPO

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
mkdir -p $RESULTS

echo "=============================="
echo "System: $SYSTEM"
echo "=============================="

# Run PCFM+Linear eval
python py/evaluate_pde.py \
    --system           $SYSTEM \
    --dataset_location $DATA \
    --phase2_ckpt      $RUNS/${P2_MAP[$SLURM_ARRAY_TASK_ID]}/params_step6000.npz \
    --baseline_ckpt    $RUNS/${BL_MAP[$SLURM_ARRAY_TASK_ID]}/params_step10000.npz \
    --n_steps          10 \
    --output_json      $RESULTS/${SYSTEM}_static_eval.json

echo "--- Now Dynamic Phase2 ---"

python py/evaluate_pde.py \
    --system           $SYSTEM \
    --dataset_location $DATA \
    --phase2_ckpt      $RUNS_DYN/${SYSTEM}_phase2/params_step6000.npz \
    --baseline_ckpt    $RUNS/${BL_MAP[$SLURM_ARRAY_TASK_ID]}/params_step10000.npz \
    --n_steps          10 \
    --output_json      $RESULTS/${SYSTEM}_dynamic_eval.json

echo "Done: $(date)"
