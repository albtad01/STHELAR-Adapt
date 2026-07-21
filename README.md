# STHELAR-Adapt: Tissue-Specific Adaptation for Spatial Transcriptomics-Informed Cell Segmentation and Classification

STHELAR-Adapt is a parameter-efficient framework for spatial transcriptomics-informed nuclei instance segmentation and cell-type classification from 40x H&E images.

[COMPAYL 2026 paper (link pending)](#citation) · [Hugging Face adapters](https://huggingface.co/albtad01/STHELAR-Adapt-CellViT-SAM-H-x40) · [STHELAR paper](https://doi.org/10.1038/s41597-026-06937-6) · [STHELAR 40x data](https://huggingface.co/datasets/FelicieGS/STHELAR_40x) · [CellViT](https://github.com/TIO-IKIM/CellViT)

<p align="center">
  <img src="figures/paper_figures/architecture.png" alt="STHELAR-Adapt architecture: pretrained encoder base weights frozen, trainable LoRA and AdaptFormer modules, frozen decoder body, and trainable final heads" width="900">
</p>

STHELAR pairs H&E morphology with Xenium spatial-transcriptomics cell annotations. The task combines nuclei instance segmentation and five-class cell typing across nine tissues, where tissue-dependent morphology and label composition create domain shift.

STHELAR-Adapt evaluates tissue-specific and multi-tissue adaptation of CellViT-SAM-H x40. The selected method trains LoRA Q/V modules, AdaptFormer bottlenecks, and final NP/HV/NT heads while freezing the encoder base weights and decoder body. Approximately 1.12% of parameters are trainable, and selected PEFT recovers 94.96% of FullFT mPQ in the multi-tissue experiment. This repository provides code, preprocessing and training configs, spatial splitting, evaluation, release tools, and verified adapter packages.

## Key results

### Kidney–Liver–Tonsil (KLT) multi-tissue setting

KLT values are test-set means over seeds 42 and 43.

| KLT method | Trainable modules | Trainable | mPQ | bPQ | F1 detection | F1 type |
|---|---|---:|---:|---:|---:|---:|
| Full fine-tuning | All weights | 100% | 0.309 | 0.515 | 0.829 | 0.666 |
| Final-head linear probe | NP/HV/NT heads | <0.01% | 0.204 | 0.446 | 0.818 | 0.430 |
| Selected PEFT | LoRA Q/V + AdaptFormer + NP/HV/NT heads | 1.12% | 0.294 | 0.504 | 0.835 | 0.578 |

### Tissue-specific selected PEFT

These within-dataset results use the released tissue-specific adapters and do not represent cross-patient or cross-site evaluation.

| Tissue | Adapter seed(s) | mPQ | F1 detection | F1 type |
|---|---:|---:|---:|---:|
| Liver | 42 | 0.3925 | 0.8916 | 0.5673 |
| Kidney | 42, 43 mean | 0.2484 | 0.8332 | 0.5013 |
| Ovary | 42 | 0.2999 | 0.8414 | 0.6182 |
| Breast | 42 | 0.3348 | 0.8424 | 0.5873 |
| Colon | 42 | 0.2047 | 0.7904 | 0.6076 |
| Lung | 42 | 0.3015 | 0.8637 | 0.6271 |
| Pancreatic | 43 | 0.2389 | 0.7383 | 0.6682 |
| Skin | 42 | 0.2239 | 0.7250 | 0.6778 |
| Tonsil | 43 | 0.2346 | 0.8226 | 0.5955 |
| **Mean** | — | **0.2755** | **0.8165** | **0.6056** |

See [the final KLT table](reports/paper_tables/klt_ablation_final_with_type_metrics.csv) and [the tissue table](reports/paper_tables/tissue_specific_compayl2026_final.csv) for the complete values and aggregation policy.

## Quick start: CellViT-SAM-H x40 + Lung adapter

The normal workflow combines the official CellViT-SAM-H x40 checkpoint with one released adapter; users do not convert legacy training checkpoints.

### A. Clone and install

```bash
git clone https://github.com/albtad01/STHELAR-Adapt.git
cd STHELAR-Adapt
conda env create -f environment.yml
conda activate sthelar-adapt
python -m pip install torch
python -m pip install -r requirements.txt
```

### B. Obtain the official base checkpoint

Download the [official CellViT-SAM-H x40 checkpoint](https://drive.usercontent.google.com/download?id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV&export=download&authuser=0), place it at `models/pretrained/CellViT-SAM-H-x40.pth`, and verify it:

```bash
sha256sum models/pretrained/CellViT-SAM-H-x40.pth
# b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf
```

The base checkpoint is required but not redistributed by STHELAR-Adapt. Users must comply with its applicable terms.

### C. Download the Lung seed-42 adapter

The repeated `--include` form below was dry-run validated with `huggingface-hub` 1.8.0:

```bash
hf download \
  albtad01/STHELAR-Adapt-CellViT-SAM-H-x40 \
  --include "adapters/tissue_specific/lung/seed42/*" \
  --include "configs/release/compayl2026/tissue_specific/training_sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml" \
  --include "scripts/verify_released_adapter.py" \
  --local-dir sthelar-adapt-release
```

### D. Verify the downloaded adapter

```bash
python sthelar-adapt-release/scripts/verify_released_adapter.py \
  sthelar-adapt-release/adapters/tissue_specific/lung/seed42 \
  --base-checkpoint models/pretrained/CellViT-SAM-H-x40.pth
```

### E. Reconstruct CellViT and load the adapter

```python
from utils.cellvit_adapter_hub import load_cellvit_base, load_sthelar_adapter

base_checkpoint = "models/pretrained/CellViT-SAM-H-x40.pth"
config_path = (
    "sthelar-adapt-release/configs/release/compayl2026/tissue_specific/"
    "training_sthelar40x_lung_5class_spatial_margin128_cap50000_"
    "lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml"
)
adapter_path = "sthelar-adapt-release/adapters/tissue_specific/lung/seed42"

model = load_cellvit_base(
    model_name="cellvit-sam-h-x40",
    base_checkpoint=base_checkpoint,
    config_path=config_path,
    device="cpu",
)
load_sthelar_adapter(model, adapter_path)
model.eval()
```

This reconstructs the released Lung-adapted CellViT model. The released Lung test values are mPQ 0.3015, F1 detection 0.8637, and F1 type 0.6271; they do not guarantee improvement on arbitrary external lung images. See [Training and evaluation](#training-and-evaluation) for dataset inference.

## Qualitative results

![Ground truth, linear probing, and selected PEFT predictions across nine tissues](docs/figures/qualitative.png)

Ground-truth overlays (top), final-head linear-probing predictions (middle), and selected LoRA+AdaptFormer plus final-head predictions (bottom) for one chosen patch from each of kidney, liver, tonsil, ovary, breast, colon, lung, pancreatic, and skin. Examples were selected for qualitative illustration and are not intended to constitute a statistically representative sample.

## Dataset preparation and spatial splitting

Download or materialize the STHELAR 40x Hugging Face Parquet dataset, then set local roots. `${STHELAR_ROOT}` and `${DATA_ROOT}` are expanded by the config loaders and fail if unresolved.

```bash
export STHELAR_ROOT=/path/to/STHELAR_40x
export DATA_ROOT=/path/to/cellvit_ready

python preprocessing/sthelar/convert_hf_to_cellvit.py \
  --config configs/release/compayl2026/preprocessing/preprocessing_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128.yaml

python utils/check_spatial_split_leakage.py \
  --dataset "$DATA_ROOT/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128" \
  --config configs/release/compayl2026/preprocessing/preprocessing_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128.yaml \
  --output-dir reports/split_checks
```

The spatial splitter assigns within-slide train/validation/test regions along a coordinate axis and discards patches in a 128-coordinate-unit band around boundaries. Inspect `split_manifest.yaml`, `patch_info_with_split.csv`, and the leakage-check CSV before training. The other paper preprocessing configs are under `configs/release/compayl2026/preprocessing/`.

The released label mapping contains five foreground types plus background:

| ID | Label |
|---:|---|
| 0 | Background |
| 1 | Immune |
| 2 | Stromal |
| 3 | Epithelial |
| 4 | Melanocyte |
| 5 | Other |

All curated configs set `num_tissue_classes: 1`. CellViT's one-output tissue classifier is retained for compatibility with the inherited architecture, trainer interface, and packaged state. With a single output, its softmax prediction and cross-entropy objective are degenerate; tissue classification is not a reported task. The paper results evaluate the NP, HV, and NT nuclei outputs.

## Training and evaluation

These advanced workflows require the prepared dataset, official base checkpoint, and appropriate compute. They are separate from the normal adapter-user workflow above.

### Training

```bash
# Final NP/HV/NT-head linear probe
python cell_segmentation/run_cellvit.py --config \
  configs/release/compayl2026/klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42.yaml

# Selected PEFT, seed 42 (repeat with the adjacent seed-43 config)
python cell_segmentation/run_cellvit.py --config \
  configs/release/compayl2026/klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml

# Full fine-tuning, seed 42 (repeat with the adjacent seed-43 config)
python cell_segmentation/run_cellvit.py --config \
  configs/release/compayl2026/klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml
```

Tissue-specific FullFT and selected-PEFT configs are in `configs/release/compayl2026/tissue_specific/`. The paper used seed 42 except Pancreatic and Tonsil (seed 43); Kidney PEFT is the mean of seeds 42 and 43.

### Dataset inference and evaluation

Training invokes patch inference after the final epoch. To rerun inference without retraining:

```bash
python utils/rerun_cellvit_inference.py \
  --run_dir /path/to/timestamped/run \
  --checkpoint_name model_best.pth \
  --gpu 0 --magnification 40
```

To rebuild the run summaries and paper tables from local run logs:

```bash
python utils/collect_sthelar_run_tables_v2.py --run-root run \
  --out-summary reports/runs_summary.csv --out-epochs reports/epoch_metrics.csv
python utils/analysis/create_final_type_metric_tables.py
```

Aggregate paper tables are tracked. Some prediction-level artifacts and timestamped runs required for complete figure and QC regeneration are not included; see [the public release audit](reports/public_release_audit.md) for details.

## Verified adapter inventory

All 12 adapter packages passed conversion, exact tensor round-trip, metadata, base-loading, and 256×256 CPU forward checks. Exact source/config/export hashes are frozen in [the verified release manifest](release/huggingface/adapter_manifest_verified.csv); the broader 51-checkpoint audit remains in [the audit manifest](release/adapter_manifest.csv). No full CellViT base weights belong in the adapter release.

| Public adapter ID | Tissue(s) | Seed |
|---|---|---:|
| `klt/seed42` | Kidney + Liver + Tonsil | 42 |
| `klt/seed43` | Kidney + Liver + Tonsil | 43 |
| `tissue_specific/breast/seed42` | Breast | 42 |
| `tissue_specific/colon/seed42` | Colon | 42 |
| `tissue_specific/kidney/seed42` | Kidney | 42 |
| `tissue_specific/kidney/seed43` | Kidney | 43 |
| `tissue_specific/liver/seed42` | Liver | 42 |
| `tissue_specific/lung/seed42` | Lung | 42 |
| `tissue_specific/ovary/seed42` | Ovary | 42 |
| `tissue_specific/pancreatic/seed43` | Pancreatic | 43 |
| `tissue_specific/skin/seed42` | Skin | 42 |
| `tissue_specific/tonsil/seed43` | Tonsil | 43 |

## Repository structure

| Path | Role |
|---|---|
| `preprocessing/sthelar/` | STHELAR Parquet inspection and CellViT conversion |
| `cell_segmentation/`, `models/`, `base_ml/`, `datamodel/` | inherited CellViT framework plus STHELAR/adapter extensions |
| `configs/release/compayl2026/` | curated preprocessing and training configurations |
| `utils/` | evaluation, analysis, checkpoint, and adapter utilities |
| `tools/` | public adapter audit, conversion, and verification tools |
| `reports/`, `figures/paper_figures/`, `docs/figures/` | selected metrics, audits, the publication architecture source, and synchronized figures |
| `ruche/`, `jeanzay/` | optional, cluster-specific SLURM examples; not required locally |
| `release/huggingface/` | Hugging Face adapter-release metadata, configs, scripts, results, and synchronized figures |

### Maintainer utilities

Checkpoint auditing, safetensors export, and package verification are documented in [the adapter release audit](reports/adapter_release_audit.md) and implemented under `tools/`. Normal adapter users do not need the legacy export workflow.

## Limitations

- Evaluation uses few slides per tissue and within-slide spatial splits; spatial separation does not establish cross-site generalization.
- STHELAR cell labels are derived from spatial transcriptomics and retain assignment uncertainty.
- Selected PEFT nearly recovers FullFT mPQ but retains a gap in cell-type F1.
- The work is for research use only and is not validated for clinical diagnosis or treatment.
- The base checkpoint is required but not redistributed; users must comply with its applicable terms and the repository license notices.

## Citation

Please cite the STHELAR-Adapt COMPAYL paper when its final bibliographic record is available, together with the underlying works:

```bibtex
@article{hoerst2024cellvit,
  title   = {CellViT: Vision Transformers for Precise Cell Segmentation and Classification},
  author  = {Hörst, Fabian and others},
  journal = {Medical Image Analysis},
  volume  = {94},
  pages   = {103143},
  year    = {2024},
  doi     = {10.1016/j.media.2024.103143}
}

@article{giraudsauveur2026sthelar,
  title   = {STHELAR, a Multi-Tissue Dataset Linking Spatial Transcriptomics and Histology for Cell Type Annotation},
  author  = {Giraud-Sauveur, Felicie and others},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-06937-6}
}

@inproceedings{kirillov2023segment,
  title  = {Segment Anything},
  author = {Kirillov, Alexander and others},
  booktitle = {ICCV},
  year   = {2023}
}
```

## License and acknowledgements

This repository inherits substantial code and documentation from CellViT. The preserved upstream README is [README_CellViT.md](README_CellViT.md), and the repository carries CellViT's [Apache 2.0 with Commons Clause notice](LICENSE). STHELAR source data remain under CC BY 4.0, which does not automatically govern this code, the adapters, SAM-derived components, or the CellViT base checkpoint. Users must comply with all applicable terms and preserve the CellViT, SAM, and STHELAR citations.
