# CellViT STHELAR Tissue Adapter

## Model Description

This repository contains a tissue-specific adapter checkpoint, not a complete
CellViT model. It must be loaded on top of the matching
`CellViT-SAM-H-x40.pth` base checkpoint.

The expected task is STHELAR spatial-transcriptomics-derived nuclei typing and
instance segmentation for the tissue and label taxonomy declared in the
checkpoint metadata.

- **Base model:** CellViT-SAM-H-x40
- **Adapter type:** `<adapter_type>`
- **Decoder scope:** `<decoder_train_scope>`
- **Tissue:** `<tissue>`
- **Split protocol:** `<slide|spatial|patch>`
- **Magnification:** 40x
- **STHELAR label setting:** `<grouped 5-class|detailed 9-class>`

## Intended Use

The adapter provides tissue-specific adaptation of CellViT-SAM-H-x40 to
STHELAR cell labels derived from spatial transcriptomics. It is intended for
research evaluation on data matching the tissue, magnification, preprocessing,
and label mapping documented below.

It is not a standalone segmentation model and is not intended for clinical
use.

## Required Files

1. The original `CellViT-SAM-H-x40.pth` base checkpoint.
2. The adapter-only `.pth` checkpoint from this repository.
3. A compatible CellViT/STHELAR-Adapt configuration.

## Label Mapping

```yaml
nuclei_types:
  Background: 0
  <class_name>: <class_id>
```

Replace this block with the exact `nuclei_types` mapping stored in the adapter
checkpoint metadata. The number and order of classes must match at load time.
The adapter `.pth` embeds this mapping under both `label_mapping` and
`nuclei_types`.

## Loading

```python
from utils.cellvit_adapter_hub import load_cellvit_base, load_sthelar_adapter

cellvit = load_cellvit_base(
    model_name="cellvit-sam-h-x40",
    base_checkpoint="models/pretrained/CellViT-SAM-H-x40.pth",
    config_path="path/to/config.yaml",
)
load_sthelar_adapter(
    cellvit,
    adapter_checkpoint="path/to/adapter.pth",
)
```

This API loads one adapter at a time. Rebuild the base model before loading an
adapter with a different architecture, decoder scope, or label mapping.

To inspect a release without constructing SAM-H:

```bash
python examples/list_adapter_metadata.py path/to/adapter.pth
```

The adapter checkpoint contains trainable LoRA/AdaptFormer/decoder-adapter
parameters, trainable prediction heads, the classifier head when applicable,
and small mutable buffers required for exact reconstruction. Frozen SAM-H and
original decoder weights are supplied by the base checkpoint and are not
duplicated.

## Checkpoint Contents

The release includes only trainable LoRA or VeRA weights, AdaptFormer weights,
configured decoder adapters or trainable decoder stages/heads, the classifier
head when trainable, and mutable BatchNorm buffers required for exact output
reconstruction. It does not include the frozen SAM-H encoder or frozen original
decoder parameters.

Metadata records the base checkpoint, tissue, split, label mode and mapping,
adapter configuration, trainable parameter count and ratio, source run,
validation metrics when available, and git commit.

## Evaluation Context

Report the dataset release, tissue, split strategy, QC metric and threshold,
patch size/stride, random seed, and checkpoint selection criterion alongside
all metrics. QC-filtered results should include both retained patch counts and
the unfiltered result.

## Limitations

Performance may not transfer to other tissues, magnifications, scanners, label
taxonomies, or preprocessing protocols. STHELAR transcriptomics-informed labels
are not directly interchangeable with morphology-derived labels such as the
original PanNuke classes.
