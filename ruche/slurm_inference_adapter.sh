#!/bin/bash
#SBATCH --job-name=infer_adapter
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
ADAPTER="${2:-ovary_5}"
BASE_CHECKPOINT="${3:-models/pretrained/CellViT-SAM-H-x40.pth}"

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: missing RUN_DIR"
  echo "Usage:"
  echo "  sbatch ruche/slurm_inference_adapter.sh <run_log_dir> [adapter_alias_or_path] [base_checkpoint]"
  exit 1
fi

if [[ "$RUN_DIR" != /* ]]; then
  RUN_DIR="$REPO/$RUN_DIR"
fi
if [[ "$BASE_CHECKPOINT" != /* ]]; then
  BASE_CHECKPOINT="$REPO/$BASE_CHECKPOINT"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run directory not found: $RUN_DIR"
  exit 1
fi
if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "ERROR: base checkpoint not found: $BASE_CHECKPOINT"
  exit 1
fi

MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"

cd "$REPO"
mkdir -p logs

module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

SAFE_ADAPTER_NAME="${ADAPTER//\//_}"
SAFE_ADAPTER_NAME="${SAFE_ADAPTER_NAME//:/_}"
SAFE_ADAPTER_NAME="${SAFE_ADAPTER_NAME//./_}"
MATERIALIZED_NAME="${SAFE_ADAPTER_NAME}_materialized_for_inference.pth"

echo "===== JOB INFO ====="
echo "Host:                 $(hostname)"
echo "Date:                 $(date)"
echo "Repo:                 $REPO"
echo "Run dir:              $RUN_DIR"
echo "Adapter:              $ADAPTER"
echo "Base checkpoint:      $BASE_CHECKPOINT"
echo "Materialized ckpt:    $MATERIALIZED_NAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "Python:               $(which python)"
python -V
echo "===== NVIDIA SMI ====="
nvidia-smi || true
echo "===================="

echo "===== MATERIALIZE ADAPTER CHECKPOINT ====="
python utils/materialize_adapter_inference_checkpoint.py \
  --run-dir "$RUN_DIR" \
  --adapter "$ADAPTER" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --out-name "$MATERIALIZED_NAME"
echo "===== MATERIALIZATION DONE ====="

echo "===== START INFERENCE ====="
python cell_segmentation/inference/inference_cellvit_experiment_pannuke.py \
  --run_dir "$RUN_DIR" \
  --checkpoint_name "$MATERIALIZED_NAME" \
  --gpu "${CUDA_VISIBLE_DEVICES:-0}" \
  --magnification 40
echo "===== INFERENCE DONE ====="
