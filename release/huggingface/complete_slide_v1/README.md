# STHELAR-Adapt

> Draft release card. The bundle is intentionally incomplete until all 30 rows in
> `complete_slide_adapter_manifest.csv` are verified.

## What this repository contains

This versioned subtree documents an adapter-only CellViT-SAM-H-x40 release. It
contains metadata for 20 verified seed-42 Fold A/Fold B complete-slide evaluation
adapters and separately identifies 10 untrained all-slides deployment targets. It
does not contain CellViT-SAM-H base weights, adapter binaries, raw STHELAR data,
optimizer state, or full training checkpoints.

## Architecture

The frozen CellViT-SAM-H encoder receives LoRA on attention Q/V (rank 8, alpha 8,
dropout 0) and AdaptFormer bottlenecks (reduction 16, GELU). The decoder body stays
frozen; the final NP, HV, and NT prediction heads are trained. Adapter archives also
include mutable normalization/BatchNorm buffers needed for exact reconstruction.

## Base model requirement

You must separately obtain `CellViT-SAM-H-x40.pth` under its applicable upstream
terms. Its required SHA256 is:

```text
b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf
```

Base weights are not included and are not licensed for redistribution by this
adapter release.

## How base + adapter works

The files are not claimed to implement standard Hugging Face PEFT format. From a
checkout of the STHELAR-Adapt code repository, use its tested CellViT loader:

```python
from utils.cellvit_adapter_hub import load_cellvit_base, load_sthelar_adapter

net = load_cellvit_base(
    model_name="cellvit-sam-h-x40",
    base_checkpoint="/path/to/CellViT-SAM-H-x40.pth",
    config_path=(
        "configs/release/complete_slide_v1/training/"
        "tissue_specific/breast/fold_a_seed42.yaml"
    ),
    device="cpu",
)
load_sthelar_adapter(
    net,
    "adapters/complete_slide_v1/samh/tissue_specific/breast/fold_a/seed42",
)
net.eval()
```

The loader injects the exact LoRA and AdaptFormer modules before loading the
adapter parameters and required mutable buffers. Verify the base SHA256 first.

## Adapter inventory

- Fold A and Fold B are reciprocal complete-slide evaluation directions. Their
  held-out slide never participates in training or validation, and evaluation uses
  the fixed epoch-10 checkpoint.
- `all_slides` is a deployment role, not a held-out evaluation model. Every
  designated slide contributes eligible patches, but the current framework needs
  validation: a nominal deterministic spatial-axis 85%/15% split is made inside
  every slide, with a 128-coordinate-unit boundary margin. Actual patch-count
  fractions vary with spatial density and are recorded in each split manifest;
  validation patches do not receive gradient updates.
- All-slides validation outputs are not paper metrics and no Fold A/B metric is
  transferred to an all-slides adapter.

The manifest is authoritative for the 3 KLT and 27 tissue-specific target paths.
No CellViT-256 adapter belongs in this repository.

## Complete-slide results

These are audited complete-slide paper results. All reported models use epoch 10;
no test result selected a checkpoint. KLT entries summarize three seeds per fold
and therefore describe the paper experiment, not only the released seed-42 file.

| Domain | Fold | bPQ | mPQ | F1det | F1type |
|---|:---:|---:|---:|---:|---:|
| KLT Selected PEFT (3 seeds) | A | 0.519 ± 0.002 | 0.219 ± 0.007 | 0.841 ± 0.004 | 0.494 ± 0.018 |
| KLT Selected PEFT (3 seeds) | B | 0.610 ± 0.002 | 0.236 ± 0.005 | 0.865 ± 0.002 | 0.490 ± 0.010 |

Tissue rows are unweighted means of the reciprocal seed-42 Fold A/B results.

| Tissue | bPQ | mPQ | F1det | F1type |
|---|---:|---:|---:|---:|
| Breast | 0.431860 | 0.258258 | 0.822320 | 0.517160 |
| Colon | 0.285615 | 0.134729 | 0.747535 | 0.406590 |
| Kidney | 0.579474 | 0.071632 | 0.874315 | 0.127615 |
| Liver | 0.500980 | 0.208030 | 0.862901 | 0.333961 |
| Lung | 0.535977 | 0.222403 | 0.852452 | 0.437147 |
| Ovary | 0.418095 | 0.188736 | 0.833601 | 0.493001 |
| Pancreatic | 0.383303 | 0.149232 | 0.729409 | 0.347744 |
| Skin | 0.326879 | 0.102916 | 0.702964 | 0.268050 |
| Tonsil | 0.514652 | 0.238071 | 0.847126 | 0.499315 |

Exact fold rows and provenance hashes are in
`results/complete_slide_v1/fold_results_seed42.csv`. There is deliberately no
all-slides result table.

## Dataset / labels

The experiments use STHELAR 40x revision
`e32a8cdd50eff2d38e237f3729e9ac85bbb5203b`, verified from the local Hugging Face
cache metadata. The release predicts Background plus five foreground classes:
Immune, Stromal, Epithelial, Melanocyte, and Other. Exact prepared split manifests
and held-out patch-ID hashes are recorded in the release inventory and results.

## Limitations

STHELAR contains few slides per tissue, these models have no clinical validation,
cell typing is more sensitive than nuclei localization, and performance may shift
under new scanners, stains, laboratories, tissues, or class distributions.
All-slides adapters have no independent held-out-slide performance claim.

## Citation

Please cite STHELAR, CellViT, SAM, and the STHELAR-Adapt manuscript when its
public bibliographic record is available.

## License

See the repository `LICENSE` and `NOTICE`. CellViT code/weights, SAM-derived
components, HIPT-derived components, and STHELAR data retain their respective
upstream terms. STHELAR is CC BY 4.0. Nothing in this adapter-only release grants
rights to redistribute the required CellViT-SAM-H base checkpoint.
