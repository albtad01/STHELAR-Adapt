# Slide-independent split manifests

This directory freezes the lightweight `split_manifest.yaml` and
`split_validation.json` files emitted by the completed KLT and nine-tissue
Fold A/B materializations. The copies preserve slide membership, patch and
class counts, creation timestamps, and disjointness results. Only the old
machine root was replaced with `${DATA_ROOT}`; no scientific membership or
count was changed.

The multi-megabyte `patch_info_with_split.csv` files and packed image/label
payloads remain external. Recreate each fold with its matching config in
`configs/slide_exp/preprocessing/` and
`utils/generate_slide_independent_fold.py`, then compare the newly emitted
manifest and validation record with the corresponding files here.
