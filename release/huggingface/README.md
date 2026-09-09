---
library_name: pytorch
license: other
license_name: cellvit-component-terms-apache-2.0-with-commons-clause
license_link: LICENSE
tags:
  - cellvit
  - segment-anything
  - sthelar
  - nuclei-segmentation
  - peft
  - lora
  - adapter
datasets:
  - FelicieGS/STHELAR_40x
---

# STHELAR-Adapt complete-slide adapter release metadata

This staging tree documents adapter-only states for STHELAR 40x nuclei instance segmentation and spatial-transcriptomics-derived five-class cell typing. The current experiment uses reciprocal complete-slide holdouts. Older COMPAYL-era within-slide packages are retained under the historical adapter layout and are not the current main result.

## What an adapter contains

The selected CellViT-SAM-H-x40 method trains LoRA Q/V modules (rank 8, alpha 8), AdaptFormer bottlenecks (reduction 16), and final NP/HV/NT heads while freezing the encoder base weights and decoder body. The archive contains those trainable tensors and the mutable normalization buffers required for exact state reconstruction. It contains no frozen CellViT/SAM base weights.

```text
official CellViT-SAM-H-x40 base checkpoint
                  +
STHELAR-Adapt adapter-only safetensors
                  =
adapted CellViT model
```

This is a CellViT-specific composite format, not a standard Hugging Face PEFT package. Use the loader in the source repository.

## Exact base requirement

The SAM-H evaluation adapters require the official `CellViT-SAM-H-x40.pth` checkpoint with SHA256:

```text
b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf
```

The upstream checkpoint is required but not included or redistributed. A matching filename is insufficient; verify its digest before loading an adapter.

## Main complete-slide result

Across nine SAM-H tissues, values are unweighted reciprocal-fold tissue means and then an unweighted mean across tissues:

| Metric | Full fine-tuning | Selected PEFT | PEFT − FullFT |
|---|---:|---:|---:|
| bPQ | 0.445141 | 0.441871 | -0.003270 |
| mPQ | 0.186681 | 0.174890 | -0.011791 |
| F1 detection | 0.806061 | 0.808069 | +0.002008 |
| F1 type | 0.417683 | 0.381176 | -0.036507 |

The canonical direction-level values, pairing checks, and aggregation policy are in the source repository's `reports/tissue_peft_vs_fullft_slideind.csv` and `reports/workshop_final_scientific_audit.md`. These are descriptive results and do not establish equivalence, superiority, patient independence, or clinical validity.

## Current package inventory

`complete_slide_adapter_manifest.csv` is authoritative for the current SAM-H release matrix:

- 20 seed-42 Fold-A/Fold-B evaluation packages are `VERIFIED_EXISTING`, including both reciprocal directions for KLT and all nine tissues;
- 10 `all_slides` deployment targets are `MISSING_NEEDS_TRAINING` and are not released models or result-bearing artifacts.

The versioned metadata, configs, split manifests, results, and verification records are under `complete_slide_v1/`. Binary `safetensors` files are intentionally excluded from the Git source repository and must be obtained only from a separately reviewed model release.

Evaluation adapters are fold-specific and exclude their designated test slide. The all-slides configs define future deployment artifacts, have no held-out-slide result, and must not be substituted when reproducing holdout results.

`slideind_adapter_manifest_verified.csv` preserves an earlier 22-package snapshot containing extra KLT seeds and two CellViT-256 packages. It is retained for provenance and is not the current release matrix.

## Loading

Clone the code repository, obtain the matching adapter package separately, and verify both artifacts:

```bash
sha256sum /path/to/CellViT-SAM-H-x40.pth
python tools/verify_released_adapter.py /path/to/adapter-package \
  --base-checkpoint /path/to/CellViT-SAM-H-x40.pth
```

Then reconstruct the model with the package's matching public training config:

```python
from utils.cellvit_adapter_hub import load_cellvit_base, load_sthelar_adapter

model = load_cellvit_base(
    model_name="cellvit-sam-h-x40",
    base_checkpoint="/path/to/CellViT-SAM-H-x40.pth",
    config_path="/path/to/matching-training-config.yaml",
    device="cpu",
)
load_sthelar_adapter(model, "/path/to/adapter-package")
model.eval()
```

## Intended use and limitations

These artifacts support research reproduction and method comparison on the declared STHELAR 40x preprocessing, label mapping, and reciprocal slides. They are not validated for diagnosis, prognosis, treatment selection, or other clinical use. Validate tissue, stain, scanner, magnification, preprocessing, labels, and base-checkpoint identity before any new-domain study.

## Citation, license, and attribution

Cite the STHELAR-Adapt manuscript when its bibliographic record is available, plus CellViT, STHELAR, Segment Anything, and HIPT as applicable. `LICENSE` contains the component-specific CellViT terms, including Apache License 2.0 and Commons Clause conditions. `NOTICE` records upstream attribution, modifications, the STHELAR CC BY 4.0 data terms, and the fact that upstream base checkpoints are not redistributed.
