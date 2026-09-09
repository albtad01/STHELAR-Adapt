# STHELAR-Adapt

STHELAR-Adapt is research software for parameter-efficient adaptation of CellViT to nuclei instance segmentation and spatial-transcriptomics-derived five-class cell typing on STHELAR 40x H&E images. The current reproducibility release uses reciprocal **complete-slide holdouts** across Kidney–Liver–Tonsil (KLT) and nine tissue-specific tasks; older COMPAYL-era within-slide experiments remain in the repository as explicitly historical material.

![STHELAR-Adapt architecture](docs/figures/architecture.png)

## Method

The selected configuration freezes the CellViT-SAM-H-x40 encoder base weights and decoder body, and trains:

- LoRA rank-8 adapters on every encoder attention Q and V projection;
- AdaptFormer bottlenecks with reduction 16 in the encoder MLP blocks;
- the final NP (nuclei probability), HV (horizontal/vertical), and NT (nuclei type) heads.

An adapter archive also stores the mutable normalization buffers required to reconstruct the evaluated state. It does **not** contain frozen CellViT or SAM base weights.

```text
official CellViT-SAM-H-x40 base checkpoint
                  +
STHELAR-Adapt adapter-only safetensors
                  =
adapted CellViT model
```

These composite archives are not standard Hugging Face PEFT packages. Load them with this repository's `utils.cellvit_adapter_hub` API.

## Main complete-slide evaluation

The main tissue-specific comparison uses nine SAM-H tissues, seed 42, and two reciprocal directions per tissue. Each direction trains on one slide and tests on the other; Fold A and Fold B are distinct held-out slides, not independent biological or stochastic replicates. Values below are unweighted means of the two folds within each tissue, followed by an unweighted mean across the nine tissues.

| Metric | Full fine-tuning | Selected PEFT | PEFT − FullFT |
|---|---:|---:|---:|
| bPQ | 0.445141 | 0.441871 | -0.003270 |
| mPQ | 0.186681 | 0.174890 | -0.011791 |
| F1 detection | 0.806061 | 0.808069 | +0.002008 |
| F1 type | 0.417683 | 0.381176 | -0.036507 |

The PEFT/FullFT ratio of the grand tissue means is 0.936837 for mPQ. This is a descriptive recovery ratio, not an equivalence, non-inferiority, or superiority result. The canonical values and aggregation policy are in [the matched tissue table](reports/tissue_peft_vs_fullft_slideind.csv) and [scientific audit](reports/workshop_final_scientific_audit.md). KLT multi-seed, fold-specific results are in [the master CSV](reports/workshop_master_results.csv) and [readable table](reports/workshop_master_results.md).

Historical within-slide configurations and results are preserved under [`configs/release/compayl2026/`](configs/release/compayl2026/) and [`reports/paper_tables/`](reports/paper_tables/). They document an earlier release and are not the current main complete-slide result.

## Quick start

```bash
git clone --branch sthelar-adapt https://github.com/albtad01/STHELAR-Adapt.git
cd STHELAR-Adapt
conda env create -f environment.yml
conda activate sthelar-adapt
python -m pip install -r requirements.txt
```

Obtain the official `CellViT-SAM-H-x40.pth` checkpoint from the [CellViT project](https://github.com/TIO-IKIM/CellViT). It is required but is not redistributed here. The released adapters were verified against exactly:

```text
b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf  CellViT-SAM-H-x40.pth
```

Verify the checkpoint before use:

```bash
echo "b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf  /path/to/CellViT-SAM-H-x40.pth" | sha256sum --check
python tools/verify_released_adapter.py /path/to/adapter-package \
  --base-checkpoint /path/to/CellViT-SAM-H-x40.pth
```

Load a complete-slide KLT Fold-A adapter with its matching architecture config:

```python
from utils.cellvit_adapter_hub import load_cellvit_base, load_sthelar_adapter

model = load_cellvit_base(
    model_name="cellvit-sam-h-x40",
    base_checkpoint="/path/to/CellViT-SAM-H-x40.pth",
    config_path=(
        "configs/slide_exp/training/"
        "training_sthelar40x_klt_5class_slideind_foldA_"
        "lora_adaptformer_heads_seed42.yaml"
    ),
    device="cpu",
)
load_sthelar_adapter(model, "/path/to/adapter-package")
model.eval()
```

The Git repository tracks code, configurations, checksums, and verification metadata—not adapter binaries or base checkpoints. Obtain an adapter package separately from the model release and keep its directory contents together.

## Training and evaluation

Set a portable data root; curated complete-slide configs expand `${DATA_ROOT}` at runtime:

```bash
export DATA_ROOT=/path/to/cellvit_ready
export CELLVIT_SAM_H_X40_CHECKPOINT=/path/to/CellViT-SAM-H-x40.pth
```

Prepare one reciprocal KLT direction and train the selected configuration:

```bash
python preprocessing/sthelar/convert_hf_to_cellvit.py \
  --config configs/slide_exp/preprocessing/preprocessing_sthelar40x_klt_5class_slideind_foldA_margin128.yaml

python cell_segmentation/run_cellvit.py --config \
  configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml
```

The paired Fold-B config reverses the training and held-out slides. Tissue-specific and CellViT-256 configurations are in [`configs/slide_exp/`](configs/slide_exp/), and the exact split assignments are in [`manifests/slide_independent/`](manifests/slide_independent/). Training normally performs final inference; an existing run can be evaluated again with:

```bash
python utils/rerun_cellvit_inference.py \
  --run_dir /path/to/run \
  --checkpoint_name checkpoint_10.pth \
  --gpu 0 --magnification 40
```

These commands require appropriate local data, the upstream checkpoint, and suitable compute. They do not download or embed either asset.

## Adapter release

[`release/huggingface/complete_slide_adapter_manifest.csv`](release/huggingface/complete_slide_adapter_manifest.csv) is the current SAM-H release matrix. It records 30 seed-42 targets: 20 Fold-A/Fold-B complete-slide evaluation adapters have exact state-reconstruction and deterministic-forward verification; 10 separate `all_slides` deployment targets are explicitly marked `MISSING_NEEDS_TRAINING`. The matching public configs are under [`configs/release/complete_slide_v1/`](configs/release/complete_slide_v1/). Binary `safetensors` files are intentionally excluded from Git.

Evaluation adapters are fold-specific: each is trained without its designated held-out slide. The all-slides configs define future deployment artifacts trained on all designated slides; they have no held-out-slide metrics and must not be used to reproduce the evaluation results. The earlier 22-package snapshot, which also includes CellViT-256 and extra KLT seeds, remains in [`release/huggingface/slideind_adapter_manifest_verified.csv`](release/huggingface/slideind_adapter_manifest_verified.csv) for provenance.

## Repository structure

| Path | Contents |
|---|---|
| `cell_segmentation/`, `models/`, `utils/` | inherited CellViT pipeline and STHELAR-Adapt implementation |
| `preprocessing/sthelar/` | STHELAR conversion and split preparation |
| `configs/slide_exp/` | current complete-slide preprocessing and training configs |
| `manifests/slide_independent/` | reciprocal slide assignments and provenance |
| `reports/` | canonical result tables and scientific audits |
| `release/huggingface/` | current and historical adapter-release metadata, verification records, and notices |
| `configs/release/compayl2026/` | historical within-slide release configs |

## Limitations

- STHELAR contains few slides per tissue. Complete-slide holdout does not establish patient independence, cross-site generalization, or clinical validity.
- Spatial-transcriptomics-derived cell labels contain assignment uncertainty and are not interchangeable with morphology-only annotations.
- Typing-sensitive mPQ and F1 type vary more across reciprocal directions than detection F1 in several tissues.
- Performance may change with tissue, stain, scanner, magnification, preprocessing, label mapping, or base-checkpoint version.
- This software and all released adapters are for research use only and are not medical devices or validated for clinical use.

## Citation

Please cite the STHELAR-Adapt manuscript when its bibliographic record is available, together with the upstream resources used in your work:

- Hörst et al., “CellViT: Vision Transformers for Precise Cell Segmentation and Classification,” *Medical Image Analysis* 94 (2024), 103143. [doi:10.1016/j.media.2024.103143](https://doi.org/10.1016/j.media.2024.103143)
- Giraud-Sauveur et al., “STHELAR, a Multi-Tissue Dataset Linking Spatial Transcriptomics and Histology for Cell Type Annotation,” *Scientific Data* (2026). [doi:10.1038/s41597-026-06937-6](https://doi.org/10.1038/s41597-026-06937-6)
- Kirillov et al., “Segment Anything,” ICCV 2023.
- Chen et al., “Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning,” CVPR 2022.

## License and acknowledgements

The distributed code contains components subject to different upstream terms. [`LICENSE`](LICENSE) reproduces the applicable CellViT component terms, including Apache License 2.0 and Commons Clause conditions; [`NOTICE`](NOTICE) records CellViT, Segment Anything, HIPT, STHELAR, and PanNuke provenance and modification statements. STHELAR data and imagery are licensed separately under CC BY 4.0. Neither that data license nor this repository's code terms grant a license to redistribute upstream base checkpoints.

No CellViT/SAM base checkpoint is included. Review the component-specific terms and upstream checkpoint conditions before redistribution or deployment.
