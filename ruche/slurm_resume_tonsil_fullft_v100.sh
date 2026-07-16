#!/bin/bash
#SBATCH --job-name=resume_tonsil_fullft_v100
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${REPO}/configs/examples/training_sthelar40x_tonsil_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN.yaml"
CHECKPOINT="${REPO}/run/sthelar40x_tonsil_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN/log/2026-07-01T230338_sthelar40x_tonsil_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN/checkpoints/checkpoint_6.pth"
MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"

cd "$REPO"
mkdir -p logs

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG"
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint file not found: $CHECKPOINT"
  exit 1
fi

module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:       $(hostname)"
echo "Date:       $(date)"
echo "Repo:       $REPO"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Python:     $(which python)"
python -V
echo "===================="

echo "===== START RESUME TRAINING ====="
python cell_segmentation/run_cellvit.py --config "$CONFIG" --checkpoint "$CHECKPOINT"
echo "===== RESUME TRAINING DONE ====="
