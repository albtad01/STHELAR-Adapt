#!/bin/bash
#SBATCH --job-name=eval_klt_frozen
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
CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
  echo "Usage: sbatch ruche/slurm_eval_frozen.sh <frozen-evaluation-config>"
  exit 1
fi
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG"
  exit 1
fi

MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"

cd "$REPO"
mkdir -p logs
module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Config: $CONFIG"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"
nvidia-smi
# SLURM remaps the one allocated GPU into process-local CUDA device 0.
python utils/evaluate_frozen_cellvit.py \
  --config "$CONFIG" \
  --gpu 0
