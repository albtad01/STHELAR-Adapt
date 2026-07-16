# Type Assignment Analysis

## What Existing Outputs Contain

- Final CellViT inference outputs are `inference_results.json` files under each run log directory.
- `dataset` stores aggregate Dice/Jaccard, bPQ/mPQ, detection F1/precision/recall, DQ/SQ, and tissue accuracy.
- `image_metrics` stores per-patch scalar bPQ/mPQ/DQ/SQ, per-class DQ/SQ/PQ, and `detection_stats`.
- `detection_stats` contains a paired true-vs-predicted type confusion matrix plus unmatched true/predicted type counts.
- The saved files do not contain per-instance masks, contours, centroids, or full true/predicted instance records.
- The small `reports/qualitative/**/{baseline_predictions,predictions}/*.npz` files contain selected prediction `instance_map` and `type_map` arrays only; they are not full-run final outputs and do not include matched GT instances.

## Metric Definitions In This Repo

- Detection F1, precision, and recall are computed from `pair_coordinates` on true/predicted centroids, using radius 12 at 40x and 6 at 20x, then `cell_detection_scores`.
- bPQ is the mean patch-level binary panoptic quality from `get_fast_pq` with IoU threshold 0.5 on remapped binary instance maps.
- mPQ is the mean over patch-level per-class PQ values, each computed with `get_fast_pq` at IoU threshold 0.5 on class-specific instance maps.
- Per-class PQ/DQ/SQ are saved from those class-specific PQ computations.
- Per-class F1 in the inference log is `cell_type_detection_scores`, which uses matched centroid pairs plus unmatched true/predicted nuclei.

## Recomputability

- `reports/runs_summary.csv` rows: 112.
- Completed or inferred rows checked for F1: 96.
- Completed rows missing `test_F1`: 0.
- Matched-nuclei type assignment can be computed from existing `detection_stats` without rerunning inference.
- IoU-based matched type assignment cannot be recomputed from current outputs. Minimal export needed: per patch, save true/pred instance IDs with type labels and either masks/label maps/contours or sufficient geometry to compute pairwise IoU.

## Summary

Top runs by matched-nuclei macro-F1:
- KLT FullFT seed42: macro-F1=0.5345, weighted-F1=0.7783, accuracy=0.7805, matched=205590
- Liver FullFT seed42: macro-F1=0.5329, weighted-F1=0.8333, accuracy=0.8363, matched=53766
- KLT FullFT seed43: macro-F1=0.5313, weighted-F1=0.7784, accuracy=0.7823, matched=204648
- KLT VeRA+AF conv_adapters seed43: macro-F1=0.5286, weighted-F1=0.7773, accuracy=0.7791, matched=203773
- KLT LoRA+AF last_stage seed42: macro-F1=0.5256, weighted-F1=0.7759, accuracy=0.7775, matched=201573

## Slide Or Region Dominance

Runs where one slide/region contributes at least half of patches or matched nuclei:
- Liver FullFT seed42: liver_s0 patch_fraction=0.6241, matched_fraction=0.5802
- Liver LoRA+AF heads_only seed42: liver_s0 patch_fraction=0.6241, matched_fraction=0.5828
- Kidney FullFT seed42: kidney_s1 patch_fraction=0.5719, matched_fraction=0.6691
- Kidney LoRA+AF heads_only seed42: kidney_s1 patch_fraction=0.5719, matched_fraction=0.6649
- Ovary FullFT seed42: ovary_s1 patch_fraction=0.8870, matched_fraction=0.9787
- Ovary LoRA+AF heads_only seed42: ovary_s1 patch_fraction=0.8870, matched_fraction=0.9770

## Selection Notes

- optional unavailable: KLT VeRA+AF heads_only seed42

## Outputs

- `type_f1_summary.csv`: one row per analyzed run.
- Per-run subdirectories contain `confusion_matrix.csv`, `confusion_matrix.png`, `per_class_type_metrics.csv`, and `slide_summary.csv`.
- `klt_frozen_fullft_lora_heads_confusions.png` aggregates KLT Frozen seed42, FullFT seeds 42/43, and LoRA+AF heads_only seeds 42/43 when available.
