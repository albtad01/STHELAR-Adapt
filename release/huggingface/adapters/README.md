# Adapter staging area

This directory contains metadata for historical adapter-package snapshots:

- `cellvit-sam-h-x40/` and `cellvit-256-x40/` describe the earlier 22-package reciprocal complete-slide snapshot listed in `../slideind_adapter_manifest_verified.csv`;
- `klt/` and `tissue_specific/` preserve the earlier COMPAYL-era within-slide release and are historical.

The current SAM-H matrix and its 20 verified evaluation packages are documented in `../complete_slide_v1/` and `../complete_slide_adapter_manifest.csv`; its 10 all-slides deployment targets remain untrained. Applicable terms are in `../LICENSE` and `../NOTICE`; those files must accompany redistribution.

Do not commit or place a CellViT/SAM base checkpoint in this tree. Adapter `*.safetensors` binaries are also excluded from the Git source repository and belong only in a separately reviewed model release.
