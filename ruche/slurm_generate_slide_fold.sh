#!/bin/bash
#SBATCH --job-name=prep_klt_slide
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
  echo "Usage: sbatch ruche/slurm_generate_slide_fold.sh <preprocessing-config>"
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
python utils/generate_slide_independent_fold.py --config "$CONFIG"
