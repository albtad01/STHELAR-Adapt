# AUTHORITATIVE PAPER RESULTS — use this file for LaTeX values.

Consolidated read-only from canonical checkpoint-10 TEST artifacts. Fold A and Fold B are reciprocal complete-slide holdouts and are never pooled as n=6.

## Completion inventory

| Family | Canonical complete | Current boundary |
|---|---:|---|
| KLT SAM-H LP | 6/6 seed×fold rows | three seeds per fold |
| KLT CellViT-256 LP | 6/6 seed×fold rows | three seeds per fold |
| KLT Selected PEFT | 12/12 across both backbones | three seeds per fold |
| KLT FullFT | 12/12 across both backbones | three seeds per fold |
| Tissue-specific Selected PEFT | 36/36 | 9 tissues × 2 folds × 2 backbones |
| Matched tissue-specific FullFT | 24/24 | SAM-H nine tissues A/B; CellViT-256 Kidney/Liver/Tonsil A/B |

Provenance: individual canonical rows and scheduler snapshot fields are retained in `reports/workshop_master_results.csv`.

## KLT fold-specific seed master

Values are mean ± sample SD across three stochastic seeds within each fold. Frozen is deterministic.

| Backbone | Method | Fold | Seeds | bPQ | mPQ | F1det | F1type |
|---|---|:---:|:---:|---:|---:|---:|---:|
| CellViT-SAM-H | Frozen | A | 1/1 | 0.476 | -- | 0.829 | -- |
| CellViT-SAM-H | Frozen | B | 1/1 | 0.557 | -- | 0.849 | -- |
| CellViT-SAM-H | LP | A | 3/3 | 0.463 ± 0.001 | 0.199 ± 0.004 | 0.829 ± 0.003 | 0.431 ± 0.014 |
| CellViT-SAM-H | LP | B | 3/3 | 0.558 ± 0.003 | 0.210 ± 0.006 | 0.850 ± 0.001 | 0.420 ± 0.004 |
| CellViT-SAM-H | Selected PEFT | A | 3/3 | 0.519 ± 0.002 | 0.219 ± 0.007 | 0.841 ± 0.004 | 0.494 ± 0.018 |
| CellViT-SAM-H | Selected PEFT | B | 3/3 | 0.610 ± 0.002 | 0.236 ± 0.005 | 0.865 ± 0.002 | 0.490 ± 0.010 |
| CellViT-SAM-H | FullFT | A | 3/3 | 0.525 ± 0.003 | 0.198 ± 0.017 | 0.836 ± 0.004 | 0.475 ± 0.028 |
| CellViT-SAM-H | FullFT | B | 3/3 | 0.604 ± 0.002 | 0.230 ± 0.003 | 0.860 ± 0.004 | 0.541 ± 0.003 |
| CellViT-256 | Frozen | A | 1/1 | 0.475 | -- | 0.834 | -- |
| CellViT-256 | Frozen | B | 1/1 | 0.561 | -- | 0.858 | -- |
| CellViT-256 | LP | A | 3/3 | 0.440 ± 0.006 | 0.171 ± 0.002 | 0.824 ± 0.005 | 0.400 ± 0.003 |
| CellViT-256 | LP | B | 3/3 | 0.540 ± 0.015 | 0.199 ± 0.007 | 0.849 ± 0.009 | 0.414 ± 0.001 |
| CellViT-256 | Selected PEFT | A | 3/3 | 0.469 ± 0.001 | 0.184 ± 0.012 | 0.833 ± 0.009 | 0.475 ± 0.034 |
| CellViT-256 | Selected PEFT | B | 3/3 | 0.564 ± 0.011 | 0.190 ± 0.006 | 0.862 ± 0.003 | 0.467 ± 0.005 |
| CellViT-256 | FullFT | A | 3/3 | 0.510 ± 0.003 | 0.187 ± 0.006 | 0.836 ± 0.002 | 0.465 ± 0.018 |
| CellViT-256 | FullFT | B | 3/3 | 0.595 ± 0.008 | 0.212 ± 0.008 | 0.863 ± 0.005 | 0.499 ± 0.016 |

Provenance: `reports/workshop_master_results.csv` retains every individual seed row, canonical job ID, checkpoint policy, and source `inference_results.json` path.

## Matched tissue-specific Selected PEFT versus FullFT

Every comparison below has an identical train slide, held-out slide, seed, fold, and TEST patch-ID set.

| Backbone | Tissue | Fold | N | PEFT bPQ | FullFT bPQ | PEFT mPQ | FullFT mPQ | ΔmPQ | Recovery | PEFT F1det | FullFT F1det | PEFT F1type | FullFT F1type |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CellViT-SAM-H | Breast | A | 49569 | 0.4113 | 0.3421 | 0.2386 | 0.2031 | +0.0355 | 1.175 | 0.8129 | 0.7819 | 0.5000 | 0.5245 |
| CellViT-SAM-H | Breast | B | 45773 | 0.4525 | 0.4661 | 0.2779 | 0.3026 | -0.0247 | 0.918 | 0.8318 | 0.8231 | 0.5343 | 0.5791 |
| CellViT-SAM-H | Colon | A | 11903 | 0.2847 | 0.2456 | 0.1495 | 0.1277 | +0.0218 | 1.171 | 0.7510 | 0.7560 | 0.4265 | 0.4398 |
| CellViT-SAM-H | Colon | B | 18464 | 0.2865 | 0.3192 | 0.1199 | 0.1242 | -0.0043 | 0.966 | 0.7441 | 0.7407 | 0.3867 | 0.4012 |
| CellViT-SAM-H | Kidney | A | 4449 | 0.5495 | 0.5917 | 0.0904 | 0.1814 | -0.0910 | 0.498 | 0.8478 | 0.8564 | 0.1492 | 0.3071 |
| CellViT-SAM-H | Kidney | B | 6723 | 0.6094 | 0.6087 | 0.0529 | 0.0888 | -0.0359 | 0.596 | 0.9008 | 0.8947 | 0.1060 | 0.2244 |
| CellViT-SAM-H | Liver | A | 9166 | 0.4801 | 0.4934 | 0.1715 | 0.1464 | +0.0251 | 1.171 | 0.8432 | 0.8410 | 0.2649 | 0.2225 |
| CellViT-SAM-H | Liver | B | 20427 | 0.5219 | 0.5075 | 0.2446 | 0.2605 | -0.0160 | 0.939 | 0.8826 | 0.8836 | 0.4030 | 0.4671 |
| CellViT-SAM-H | Lung | A | 20621 | 0.5083 | 0.5180 | 0.1934 | 0.2275 | -0.0341 | 0.850 | 0.8430 | 0.8368 | 0.4042 | 0.4911 |
| CellViT-SAM-H | Lung | B | 10938 | 0.5637 | 0.5860 | 0.2514 | 0.2929 | -0.0415 | 0.858 | 0.8619 | 0.8528 | 0.4701 | 0.5574 |
| CellViT-SAM-H | Ovary | A | 25498 | 0.4488 | 0.4501 | 0.2226 | 0.2473 | -0.0247 | 0.900 | 0.8344 | 0.8225 | 0.5054 | 0.5917 |
| CellViT-SAM-H | Ovary | B | 10039 | 0.3874 | 0.3888 | 0.1548 | 0.1524 | +0.0024 | 1.016 | 0.8328 | 0.8313 | 0.4806 | 0.4581 |
| CellViT-SAM-H | Pancreatic | A | 12647 | 0.3774 | 0.3978 | 0.1239 | 0.1393 | -0.0154 | 0.890 | 0.7338 | 0.7440 | 0.2788 | 0.3323 |
| CellViT-SAM-H | Pancreatic | B | 25526 | 0.3892 | 0.3769 | 0.1745 | 0.1476 | +0.0270 | 1.183 | 0.7250 | 0.6996 | 0.4167 | 0.3745 |
| CellViT-SAM-H | Skin | A | 12790 | 0.4401 | 0.4497 | 0.1342 | 0.1319 | +0.0023 | 1.017 | 0.7432 | 0.7528 | 0.2425 | 0.2103 |
| CellViT-SAM-H | Skin | B | 11209 | 0.2137 | 0.2240 | 0.0717 | 0.0755 | -0.0038 | 0.950 | 0.6628 | 0.7089 | 0.2936 | 0.2720 |
| CellViT-SAM-H | Tonsil | A | 21083 | 0.4522 | 0.4632 | 0.2158 | 0.2390 | -0.0232 | 0.903 | 0.8383 | 0.8270 | 0.4879 | 0.5385 |
| CellViT-SAM-H | Tonsil | B | 23067 | 0.5771 | 0.5837 | 0.2603 | 0.2722 | -0.0119 | 0.956 | 0.8560 | 0.8561 | 0.5108 | 0.5269 |
| CellViT-256 | Kidney | A | 4449 | 0.4681 | 0.5707 | 0.0989 | 0.1578 | -0.0589 | 0.627 | 0.8339 | 0.8549 | 0.2104 | 0.3170 |
| CellViT-256 | Kidney | B | 6723 | 0.5683 | 0.5940 | 0.0965 | 0.0915 | +0.0050 | 1.054 | 0.8938 | 0.8931 | 0.1727 | 0.2060 |
| CellViT-256 | Liver | A | 9166 | 0.4794 | 0.4710 | 0.0737 | 0.0821 | -0.0084 | 0.898 | 0.8325 | 0.8206 | 0.1136 | 0.1327 |
| CellViT-256 | Liver | B | 20427 | 0.5721 | 0.4624 | 0.2513 | 0.1962 | +0.0552 | 1.281 | 0.8653 | 0.8770 | 0.3963 | 0.3894 |
| CellViT-256 | Tonsil | A | 21083 | 0.4091 | 0.4516 | 0.1767 | 0.2115 | -0.0348 | 0.835 | 0.8278 | 0.8207 | 0.4587 | 0.5249 |
| CellViT-256 | Tonsil | B | 23067 | 0.5274 | 0.5710 | 0.2295 | 0.2606 | -0.0312 | 0.880 | 0.8632 | 0.8543 | 0.4998 | 0.5311 |

Provenance: exact paired source paths, TEST patch-ID hashes, resource JSONs, and all metric deltas are in `reports/tissue_peft_vs_fullft_slideind.csv`.

## Nine-tissue Selected PEFT matrices

The complete numerical and per-class matrices are in `reports/nine_tissue_peft_slideind.md` and `.csv`. Matched FullFT covers all nine SAM-H tissues and Kidney/Liver/Tonsil for CellViT-256.

| Backbone | Directions complete | Reciprocal tissues complete |
|---|---:|---:|
| CellViT-SAM-H | 18/18 | 9/9 |
| CellViT-256 | 18/18 | 9/9 |

Provenance: `reports/nine_tissue_peft_slideind.csv` contains each canonical TEST JSON path and all per-class supports/F1 values.

## Matched computational context

The canonical seed42 A100 audit remains separate from predictive seed aggregation. SAM-H Selected PEFT trains 1.1176% of parameters with 9.362 GiB peak allocated VRAM versus FullFT 100% and 15.508 GiB; its verified adapter is about 31.8 MB versus a measured 8.40 GB FullFT training checkpoint. CellViT-256 Selected PEFT trains 0.7942% with 2.040 GiB versus FullFT 2.667 GiB; its verified adapter is about 1.61 MB versus a 561 MB FullFT checkpoint.

Sources: `reports/efficiency_audit/paper_ready_efficiency.csv`, run-specific efficiency JSONs in `reports/tissue_peft_vs_fullft_slideind.csv`, and verified adapter retention metadata.

## Historical within-slide ablation uncertainty

`reports/workshop_ablation_seed_audit.md` shows heterogeneous replication: only five of eleven rows have two comparable canonical seeds; the remaining six are single-seed point estimates, and Frozen has incompatible typing taxonomy. Uncertainty cannot safely be added uniformly to the main ablation table. Mean ± sample SD is defensible only for the five n=2 rows, with explicit row-wise n.

## PAPER-SAFE FINDINGS

- Under complete-slide KLT, Selected PEFT tracks FullFT closely in bPQ/F1det; typing-sensitive mPQ/F1type differences are more direction- and backbone-dependent.
- The lightweight CellViT-256 backbone shows a larger KLT bPQ gap between Selected PEFT and FullFT than SAM-H, while both retain similar F1det.
- Across the complete nine-tissue SAM-H matched block, PEFT sometimes approaches or exceeds FullFT on individual metrics/directions, but the direction-dependent pattern does not support equivalence or superiority.
- F1type and mPQ are more sensitive than F1det to reciprocal slide/tissue changes; detection is descriptively more stable.
- Fold/tissue variation is descriptively larger than seed variation for several typing metrics, but folds are different held-out slides rather than stochastic replicates.
- Class-frequency TVD/JSD, rare classes, density changes, and dominant confusion pairs align descriptively with some typing failures; they do not establish causality.
- Measured adapters provide substantial trainable-parameter and deployable-storage reductions; the matched A100 audit must remain separate from multi-seed predictive aggregation.
- NOT SUPPORTED: statistical superiority/equivalence/non-inferiority, patient-independent replication, causal explanations, or extrapolation of nine-tissue FullFT evidence to CellViT-256.
