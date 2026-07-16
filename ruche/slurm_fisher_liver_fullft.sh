#!/bin/bash
#SBATCH --job-name=fisher_liver_fullft
#SBATCH --time=06:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --exclude=ruche-gpu16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Submit this script from the repository root, because Slurm log paths are relative.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"
NUM_BATCHES="${NUM_BATCHES:-5}"

CONFIG="configs/examples/training_sthelar40x_liver_5class_spatial_fullft_lr1e-5_e10_seed42_KEEPALL_FISHER.yaml"
RUN_LOG_ROOT="run/sthelar40x_liver_5class_spatial_fullft_lr1e-5_e10_seed42_KEEPALL_FISHER/log"
OUTPUT_DIR="reports/fisher_drift/liver_fullft_keepall_blocks_b${NUM_BATCHES}"

cd "$REPO"
mkdir -p logs "$OUTPUT_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG"
  exit 1
fi

if [[ ! -d "$RUN_LOG_ROOT" ]]; then
  echo "ERROR: run log root not found: $RUN_LOG_ROOT"
  exit 1
fi

RUN_DIR=$(find "$RUN_LOG_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: could not find latest run directory under $RUN_LOG_ROOT"
  exit 1
fi

CKPT1="$RUN_DIR/checkpoints/checkpoint_1.pth"
CKPT5="$RUN_DIR/checkpoints/checkpoint_5.pth"
CKPT10="$RUN_DIR/checkpoints/checkpoint_10.pth"
for checkpoint in "$CKPT1" "$CKPT5" "$CKPT10"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: required checkpoint not found: $checkpoint"
    exit 1
  fi
done

module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:        $(hostname)"
echo "Date:        $(date)"
echo "Repo:        $REPO"
echo "Config:      $CONFIG"
echo "Run dir:     $RUN_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Num batches: $NUM_BATCHES"
echo "Python:      $(which python)"
python -V
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "WARNING: nvidia-smi failed"
else
  echo "nvidia-smi not available"
fi
python -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}"); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda device")'
echo "===================="

echo "===== PRINT FULLFT BLOCK GROUPS ====="
python utils/fisher_drift_diagnostic.py \
  --print-groups \
  --grouping fullft_blocks \
  --config "$CONFIG" \
  --checkpoint "$CKPT10" \
  2>&1 | tee "$OUTPUT_DIR/group_inventory.txt"

echo "===== START FISHER DRIFT ====="
python utils/fisher_drift_diagnostic.py \
  --grouping fullft_blocks \
  --config "$CONFIG" \
  --checkpoint "$CKPT1" \
  --compare-checkpoints "$CKPT5" "$CKPT10" \
  --dataset-split train \
  --num-batches "$NUM_BATCHES" \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
echo "===== FISHER DRIFT DONE ====="
