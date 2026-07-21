---
library_name: pytorch
license: other
license_name: apache-2.0-with-commons-clause
license_link: LICENSE
tags:
  - cellvit
  - segment-anything
  - sthelar
  - spatial-transcriptomics
  - nuclei-segmentation
  - peft
  - lora
  - adapter
datasets:
  - FelicieGS/STHELAR_40x
---

# STHELAR-Adapt: Tissue-Specific Adaptation for Spatial Transcriptomics-Informed Cell Segmentation and Classification

STHELAR-Adapt is an adapter-only release for spatial transcriptomics-informed nuclei segmentation and five-class cell typing with CellViT-SAM-H x40. It provides twelve verified KLT and tissue-specific LoRA + AdaptFormer packages across nine STHELAR tissues.

[GitHub repository](https://github.com/albtad01/STHELAR-Adapt)

![STHELAR-Adapt architecture: pretrained encoder base weights frozen, trainable LoRA and AdaptFormer modules, frozen decoder body, and trainable final heads](figures/architecture.png)

## Model summary

The selected method freezes the CellViT-SAM-H x40 encoder base weights and decoder body while training:

- LoRA rank 8, alpha 8, dropout 0 on every encoder attention Q and V projection;
- AdaptFormer bottlenecks with reduction 16 and GELU in encoder MLP blocks;
- final NP (nuclei probability), HV (horizontal/vertical), and NT (nuclei type) output heads.

The selected configuration has 7,908,779 trainable parameters, approximately 1.1176% of the full model. Mutable BatchNorm buffers needed for exact reconstruction are packaged separately from trainable state keys inside each safetensors file; the frozen base weights are not included.

## Exact base requirement

- Architecture/checkpoint name: `CellViT-SAM-H-x40.pth`
- Model family: CellViT-SAM-H x40
- Backbone: SAM-H, 32 blocks, embedding dimension 1280, 16 attention heads
- Input: normalized RGB 256×256 STHELAR 40x patches

Download the [official CellViT-SAM-H x40 checkpoint](https://drive.usercontent.google.com/download?id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV&export=download&authuser=0). The base checkpoint is **required but not included or redistributed**. Verify the file before loading an adapter:

```bash
sha256sum /path/to/CellViT-SAM-H-x40.pth
# b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf
```

Users must comply with the base checkpoint's applicable terms. A filename match alone is insufficient for compatibility.

## Intended uses

- Research reproduction of the COMPAYL 2026 STHELAR-Adapt experiments.
- Research evaluation of nuclei instance segmentation and the declared STHELAR five-class mapping on matching 40x preprocessing/splits.
- Study of tissue-specific parameter-efficient adaptation of CellViT.

## Out-of-scope uses

- Clinical diagnosis, prognosis, treatment selection, or autonomous pathology reporting.
- Patient-level decision making or deployment without independent validation and governance.
- Images from unvalidated tissues, stains, scanners, magnifications, institutions, or preprocessing pipelines.
- Treating spatial-transcriptomics-derived labels as error-free pathology ground truth.
- Loading an adapter with a different base checkpoint, label order, number of classes, or adapter architecture.

## Training data

Adapters were trained on the public [STHELAR 40x Hugging Face dataset](https://huggingface.co/datasets/FelicieGS/STHELAR_40x), derived from Xenium spatial transcriptomics paired with H&E imagery. Experiments cover kidney, liver, tonsil, ovary, breast, colon, lung, pancreatic, and skin tissues. KLT combines kidney, liver, and tonsil.

The release configs use within-slide spatial train/validation/test regions with a 128-coordinate-unit boundary exclusion band. Some tissue configs cap each slide at 50,000 patches before splitting. The exact source Parquet revision and file-level hashes used during the original experiments were not pinned. Users reproducing the experiments should record the STHELAR dataset revision and input checksums they use.

### Label mapping

The task has five foreground classes plus background:

| ID | Label |
|---:|---|
| 0 | Background |
| 1 | Immune |
| 2 | Stromal |
| 3 | Epithelial |
| 4 | Melanocyte |
| 5 | Other |

`num_nuclei_classes` is therefore 6. All release configs set `num_tissue_classes: 1`. The inherited CellViT tissue classifier is retained for architecture/trainer-interface and packaged-state compatibility. A one-output softmax and cross-entropy objective are degenerate, and tissue classification is not part of the reported task; reported metrics evaluate the NP, HV, and NT nuclei outputs.

## Training configuration

Selected PEFT runs use AdamW, learning rate `5e-5`, batch size 4, 10 epochs, mixed precision, and the augmentations recorded in `configs/release/compayl2026/`. Exact configs are included in this repository; see [GitHub](https://github.com/albtad01/STHELAR-Adapt) for preprocessing and training workflows.

## Results

### Kidney–Liver–Tonsil (KLT) multi-tissue setting

KLT values are test-set means over seeds 42 and 43.

| KLT method | Trainable modules | Trainable | mPQ | bPQ | F1 detection | F1 type |
|---|---|---:|---:|---:|---:|---:|
| Full fine-tuning | All weights | 100% | 0.309 | 0.515 | 0.829 | 0.666 |
| Selected PEFT | LoRA Q/V + AdaptFormer + NP/HV/NT heads | 1.12% | 0.294 | 0.504 | 0.835 | 0.578 |
| Final-head linear probe | NP/HV/NT heads | <0.01% | 0.204 | 0.446 | 0.818 | 0.430 |

Selected PEFT recovers 94.96% of FullFT KLT mPQ. [`results/klt_ablation_final_with_type_metrics.csv`](results/klt_ablation_final_with_type_metrics.csv) contains the complete values and ablations.

### Tissue-specific selected PEFT

These within-dataset results do not represent cross-patient or cross-site evaluation.

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

See [`results/tissue_specific_compayl2026_final.csv`](results/tissue_specific_compayl2026_final.csv) for unrounded values and aggregation policy.

## Verified adapter inventory

This release contains 12 adapter-only files: KLT seeds 42/43; Breast 42; Colon 42; Kidney 42/43; Liver 42; Lung 42; Ovary 42; Pancreatic 43; Skin 42; and Tonsil 43. [`adapter_manifest_verified.csv`](adapter_manifest_verified.csv) freezes stable public IDs, exact source/config/export SHA256 values, tensor structure, and verification status; [`verification_summary.json`](verification_summary.json) records the environment and full test outcomes.

No ablation, failed, retest, duplicate-named alternate run, or full base checkpoint belongs in the canonical set.

## Quick start: CellViT-SAM-H x40 + Lung adapter

Clone the code repository and create its public environment:

```bash
git clone https://github.com/albtad01/STHELAR-Adapt.git
cd STHELAR-Adapt
conda env create -f environment.yml
conda activate sthelar-adapt
python -m pip install torch
python -m pip install -r requirements.txt
```

Place the [official base checkpoint](https://drive.usercontent.google.com/download?id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV&export=download&authuser=0) at `models/pretrained/CellViT-SAM-H-x40.pth`, verify the SHA256 shown above, then download the Lung seed-42 package. This repeated-`--include` command was dry-run validated with `huggingface-hub` 1.8.0:

```bash
hf download \
  albtad01/STHELAR-Adapt-CellViT-SAM-H-x40 \
  --include "adapters/tissue_specific/lung/seed42/*" \
  --include "configs/release/compayl2026/tissue_specific/training_sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml" \
  --include "scripts/verify_released_adapter.py" \
  --local-dir sthelar-adapt-release
```

Verify the downloaded adapter against the local base:

```bash
python sthelar-adapt-release/scripts/verify_released_adapter.py \
  sthelar-adapt-release/adapters/tissue_specific/lung/seed42 \
  --base-checkpoint models/pretrained/CellViT-SAM-H-x40.pth
```

Reconstruct CellViT and load the adapter:

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

This reconstructs the released Lung-adapted CellViT model. Its released test values are mPQ 0.3015, F1 detection 0.8637, and F1 type 0.6271; they do not guarantee improvement on arbitrary external lung images. Use the [GitHub inference and evaluation workflow](https://github.com/albtad01/STHELAR-Adapt#training-and-evaluation) for datasets.

## Limitations

- There are limited slides per tissue; within-slide spatial splits do not measure cross-site or cross-patient generalization.
- ST-derived cell-type labels have assignment uncertainty and are not interchangeable with morphology-only annotations.
- A residual F1-type gap remains between selected PEFT and FullFT, even where detection F1 is comparable or higher.
- The heterogeneous `Other` class and absent/rare classes affect macro metrics.
- Ovary aggregate metrics have a documented slide-dominance caveat.
- Performance may shift with scanner, stain, tissue, magnification, patch sampling, QC threshold, or base-checkpoint version.
- Research use only; no clinical validation has been performed.

## Ethical and clinical considerations

Errors can miss nuclei, merge/split instances, or assign incorrect cell types, potentially biasing downstream biological conclusions. Tissue and site imbalance can produce uneven performance across populations and laboratories. Users should inspect per-class/per-slide errors, validate on their intended cohort, retain human oversight, and avoid clinical claims. Dataset privacy/de-identification and intended-use conditions remain the user's responsibility.

## Qualitative results

![Ground truth, linear probing, and selected PEFT predictions across nine tissues](figures/qualitative.png)

Rows show ground truth, final-head linear probing, and selected PEFT predictions for chosen patches across nine tissues. Examples were selected for qualitative illustration and are not intended to constitute a statistically representative sample.

## Citation

Please cite the final STHELAR-Adapt COMPAYL 2026 paper once its bibliographic record is available, as well as:

- Hörst et al., “CellViT: Vision Transformers for Precise Cell Segmentation and Classification,” *Medical Image Analysis* 94 (2024), 103143. DOI: 10.1016/j.media.2024.103143.
- Giraud-Sauveur et al., “STHELAR, a Multi-Tissue Dataset Linking Spatial Transcriptomics and Histology for Cell Type Annotation,” *Scientific Data* (2026). DOI: 10.1038/s41597-026-06937-6.
- Kirillov et al., “Segment Anything,” ICCV 2023.

## License

This release is distributed under the terms in [`LICENSE`](LICENSE):
**Apache License 2.0 with the Commons Clause**. This preserves the
restrictive upstream terms applied to the original CellViT code.

The CellViT-SAM-H x40 base checkpoint is not included and must be obtained
separately under its applicable terms.

STHELAR source data and source imagery remain licensed under CC BY 4.0.
The qualitative figures contain STHELAR-derived imagery with ground-truth
and model-prediction overlays added for this work. Please cite CellViT,
STHELAR, Segment Anything, and STHELAR-Adapt as described above.
