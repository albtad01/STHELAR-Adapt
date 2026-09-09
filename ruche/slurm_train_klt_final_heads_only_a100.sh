#!/bin/bash
#SBATCH --job-name=klt_final_heads_a100
#SBATCH --time=24:00:00
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
CONFIG="${REPO}/configs/examples/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42.yaml"
MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"

cd "$REPO"
mkdir -p logs

module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "Repo:   $REPO"
echo "Config: $CONFIG"
echo "Python: $(which python)"
python -V
echo "===================="

echo "===== START TRAINING ====="
python cell_segmentation/run_cellvit.py --config "$CONFIG"
echo "===== TRAINING DONE ====="
