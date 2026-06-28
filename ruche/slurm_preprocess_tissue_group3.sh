#!/bin/bash
#SBATCH --job-name=prep_tissue_g3
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_long
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

cd "$REPO"
mkdir -p logs reports/split_checks

module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "Repo:   $REPO"
echo "Python: $(which python)"
python -V
echo "===================="

run_one() {
  local config="$1"
  if [[ ! -f "$config" ]]; then
    echo "ERROR: config file not found: $config"
    exit 1
  fi

  local output_root
  output_root=$(python -c 'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["output_root"])' "$config")
  local report_prefix
  report_prefix=$(basename "$output_root")

  echo "===== PREPROCESS DATASET ====="
  python -c 'import sys, yaml; c=yaml.safe_load(open(sys.argv[1])); print("Tissue: {}".format(c.get("tissue_name"))); print("Slides: {}".format(", ".join(c.get("slide_ids", [])))); print("Output root: {}".format(c.get("output_root"))); print("Strategy: {}".format(c.get("strategy"))); print("Split axis: {}".format(c.get("split_axis"))); print("Boundary margin: {}".format(c.get("boundary_margin"))); print("Label mode: {}".format(c.get("label_mode"))); print("Max patches per slide: {}".format(c.get("max_patches_per_slide")))' "$config"

  if [[ -d "$output_root" ]]; then
    echo "WARNING: output_root already exists; not overwriting: $output_root"
    echo "         Skipping preprocessing and running read-only sanity check if metadata exists."
  else
    python preprocessing/sthelar/convert_hf_to_cellvit.py --config "$config"
  fi

  if [[ -f "$output_root/patch_info_with_split.csv" && -f "$output_root/split_manifest.yaml" ]]; then
    python utils/check_spatial_split_leakage.py \
      --dataset "$output_root" \
      --config "$config" \
      --output-dir reports/split_checks \
      --report-prefix "$report_prefix"
  else
    echo "WARNING: split metadata not found; skipping split sanity check for $output_root"
  fi

  if [[ -d "$output_root" ]]; then
    du -sh "$output_root"
  fi
  echo "===== DATASET DONE ====="
}

run_one configs/preprocessing_sthelar40x_ovary_5class_spatial_margin128.yaml
run_one configs/preprocessing_sthelar40x_pancreatic_5class_spatial_margin128.yaml
run_one configs/preprocessing_sthelar40x_skin_5class_spatial_margin128.yaml

echo "===== GROUP 3 PREPROCESSING DONE ====="
