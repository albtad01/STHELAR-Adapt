#!/bin/bash
#SBATCH --job-name=infer_sthelar
#SBATCH --time=08:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"

RUN_DIR="${1:-}"
CHECKPOINT_NAME="${2:-checkpoint_10.pth}"

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: missing RUN_DIR"
  echo "Usage:"
  echo "  sbatch ruche/slurm_inference.sh <run_log_dir> [checkpoint_name]"
  exit 1
fi

if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO/$RUN_DIR"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run directory not found: $RUN_DIR"
  exit 1
fi

MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"

cd "$REPO"
mkdir -p logs

module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:                 $(hostname)"
echo "Date:                 $(date)"
echo "Repo:                 $REPO"
echo "Run dir:              $RUN_DIR"
echo "Checkpoint:           $CHECKPOINT_NAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "Python:               $(which python)"
python -V
echo "===== NVIDIA SMI ====="
nvidia-smi || true
echo "===================="

echo "===== CUDA CHECK ====="
python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0 =", torch.cuda.get_device_name(0))
PY
echo "======================"

echo "===== START INFERENCE ====="
python cell_segmentation/inference/inference_cellvit_experiment_pannuke.py \
  --run_dir "$RUN_DIR" \
  --checkpoint_name "$CHECKPOINT_NAME" \
  --gpu "${CUDA_VISIBLE_DEVICES:-0}" \
  --magnification 40
echo "===== INFERENCE DONE ====="
