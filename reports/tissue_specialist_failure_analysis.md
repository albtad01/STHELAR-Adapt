# Tissue-specialist failure analysis — canonical seed42 evidence

> Canonical policy: checkpoint-10 TEST inference (or exact verified adapter reconstruction) only. Validation curves are shown diagnostically and are never substituted for TEST evidence.

This report reconstructs the exact patch-ID intersections used in `reports/workshop_master_results.csv` for CellViT-SAM-H Selected PEFT seed42. Cell counts below are **patch-level nucleus annotation occurrences**, not unique biological cells: STHELAR patches overlap, so one biological cell can occur in more than one patch. Fold A and Fold B remain reciprocal held-out-slide directions and are not pooled.

### Biological-independence boundary

The local STHELAR README states that the resource contains 27 FFPE slides from 20 cancer patients, but neither that README, `cell_metadata/index.csv`, the per-slide cell metadata, nor the prepared fold manifests provide a slide-to-patient/case/donor/specimen mapping. The `*_s0` and `*_s1` identifiers therefore establish different slides only; they do **not** establish different patients. These experiments support ‘slide-independent’, ‘held-out slide’, or ‘complete-slide holdout’ wording, not patient-independent or specimen-independent wording.

| Protocol | Fold | Train/validation source slide(s) | Held-out TEST slide(s) |
|---|:---:|---|---|
| KLT generalist | A | `kidney_s0`, `liver_s0`, `tonsil_s0` | `kidney_s1`, `liver_s1`, `tonsil_s1` |
| KLT generalist | B | `kidney_s1`, `liver_s1`, `tonsil_s1` | `kidney_s0`, `liver_s0`, `tonsil_s0` |
| Kidney specialist | A / B | `kidney_s0` / `kidney_s1` | `kidney_s1` / `kidney_s0` |
| Liver specialist | A / B | `liver_s0` / `liver_s1` | `liver_s1` / `liver_s0` |
| Tonsil specialist | A / B | `tonsil_s0` / `tonsil_s1` | `tonsil_s1` / `tonsil_s0` |

Source: prepared `split_manifest.yaml` files under `../Datasets/cellvit_ready/sthelar40x_{klt,kidney,liver,tonsil}_5class_slideind_fold*`. Train and validation are spatial partitions of the listed training slide(s); TEST is the complete reciprocal slide payload.

## Summary

| Tissue | Fold | Model | Train / val / raw test patches | Matched test | Train nuclei | TVD train→test | JSD (bits) | bPQ | mPQ | F1det | F1type |
|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kidney | A | Generalist | 24,283 / 2,052 / 23,494 | 4,449 | 1,448,119 | 0.077 | 0.009 | 0.5766 | 0.1826 | 0.8669 | 0.3011 |
| Kidney | A | Specialist | 6,321 / 402 / 4,449 | 4,449 | 188,119 | 0.457 | 0.214 | 0.5495 | 0.0904 | 0.8478 | 0.1492 |
| Kidney | B | Generalist | 20,893 / 2,601 / 26,335 | 6,723 | 1,113,258 | 0.430 | 0.174 | 0.6409 | 0.2372 | 0.9035 | 0.3537 |
| Kidney | B | Specialist | 3,912 / 537 / 6,723 | 6,723 | 92,755 | 0.468 | 0.225 | 0.6094 | 0.0529 | 0.9008 | 0.1060 |
| Liver | A | Generalist | 24,283 / 2,052 / 23,494 | 9,166 | 1,448,119 | 0.354 | 0.107 | 0.5774 | 0.2316 | 0.8722 | 0.3595 |
| Liver | A | Specialist | 18,780 / 1,647 / 9,166 | 9,166 | 449,720 | 0.029 | 0.002 | 0.4801 | 0.1715 | 0.8432 | 0.2649 |
| Liver | B | Generalist | 20,893 / 2,601 / 26,335 | 9,756 | 1,113,258 | 0.342 | 0.107 | 0.6257 | 0.2120 | 0.8977 | 0.3121 |
| Liver | B | Specialist | 8,174 / 992 / 20,427 | 9,756 | 293,190 | 0.032 | 0.003 | 0.5225 | 0.2455 | 0.8836 | 0.4047 |
| Tonsil | A | Generalist | 24,283 / 2,052 / 23,494 | 9,879 | 1,448,119 | 0.143 | 0.031 | 0.4408 | 0.2188 | 0.8279 | 0.5073 |
| Tonsil | A | Specialist | 21,001 / 2,066 / 21,083 | 9,879 | 2,442,229 | 0.071 | 0.012 | 0.4503 | 0.2158 | 0.8379 | 0.4879 |
| Tonsil | B | Generalist | 20,893 / 2,601 / 26,335 | 9,856 | 1,113,258 | 0.166 | 0.025 | 0.5723 | 0.2443 | 0.8493 | 0.4540 |
| Tonsil | B | Specialist | 18,831 / 2,252 / 23,067 | 9,856 | 1,558,660 | 0.062 | 0.010 | 0.5767 | 0.2606 | 0.8560 | 0.5134 |

TVD is total-variation distance on the five foreground-class frequency vectors (0 means identical, 1 means disjoint). JSD is Jensen–Shannon divergence in bits. Test distributions always use the exact matched patch intersection.

## Kidney — Fold A

Exact matched population: **4,449 patches**. Generalist raw test coverage: 23,494; specialist raw test coverage: 4,449.

### Data volume and class distribution

| Model | Split | Patches | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|---:|---:|---:|---:|---:|---:|
| Generalist | Train | 24,283 | 798,175 (55.12%) | 270,672 (18.69%) | 356,338 (24.61%) | 0 (0.00%) | 22,934 (1.58%) |
| Generalist | Validation | 2,052 | 80,971 (59.95%) | 26,407 (19.55%) | 26,653 (19.73%) | 0 (0.00%) | 1,039 (0.77%) |
| Generalist | Matched TEST | 4,449 | 59,273 (55.69%) | 27,084 (25.45%) | 17,944 (16.86%) | 0 (0.00%) | 2,127 (2.00%) |
| Specialist | Train | 6,321 | 18,889 (10.04%) | 52,774 (28.05%) | 106,511 (56.62%) | 0 (0.00%) | 9,945 (5.29%) |
| Specialist | Validation | 402 | 1,141 (17.88%) | 3,214 (50.35%) | 1,919 (30.06%) | 0 (0.00%) | 109 (1.71%) |
| Specialist | Matched TEST | 4,449 | 59,273 (55.69%) | 27,084 (25.45%) | 17,944 (16.86%) | 0 (0.00%) | 2,127 (2.00%) |

| Model | TVD | JSD bits | Specialist/generalist train-patch ratio | Absent, rare, or severely underrepresented training classes |
|---|---:|---:|---:|---|
| Generalist | 0.077 | 0.009 | 1.000 | Melanocyte absent |
| Specialist | 0.457 | 0.214 | 0.260 | Immune severely underrepresented (train 10.04%, test 55.69%); Melanocyte absent |

Declared thresholds: rare = <1% of training annotation occurrences; severe underrepresentation = training frequency <25% of matched-test frequency when the test frequency is at least 1%.

### Canonical matched TEST metrics and paired type performance

| Model | Dice | bPQ | mPQ | F1det | F1type |
|---|---:|---:|---:|---:|---:|
| Generalist | 0.8139 | 0.5766 | 0.1826 | 0.8669 | 0.3011 |
| Specialist | 0.8014 | 0.5495 | 0.0904 | 0.8478 | 0.1492 |

| Model | Class | Matched true support | Total true support | Paired type F1 | Predicted count | Predicted frequency |
|---|---|---:|---:|---:|---:|---:|
| Generalist | Immune | 52,308 | 58,432 | 0.5542 | 29,062 | 29.59% |
| Generalist | Stromal | 21,500 | 26,658 | 0.4968 | 62,599 | 63.74% |
| Generalist | Epithelial | 13,101 | 17,674 | 0.1533 | 6,531 | 6.65% |
| Generalist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Generalist | Other | 1,158 | 2,105 | 0.0000 | 11 | 0.01% |
| Specialist | Immune | 51,254 | 58,422 | 0.0700 | 3,794 | 3.88% |
| Specialist | Stromal | 20,986 | 26,644 | 0.4025 | 87,783 | 89.77% |
| Specialist | Epithelial | 12,660 | 17,669 | 0.1208 | 6,189 | 6.33% |
| Specialist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Specialist | Other | 1,081 | 2,100 | 0.0037 | 25 | 0.03% |

#### Generalist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 5 | 22,017 | 27,204 | 3,086 | 0 | 1 |
| Stromal | 3 | 1,337 | 19,075 | 1,085 | 0 | 3 |
| Epithelial | 7 | 3,417 | 8,248 | 1,436 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 372 | 758 | 28 | 0 | 0 |

Unpaired true counts: `{'Background': 0, 'Immune': 6124, 'Stromal': 5158, 'Epithelial': 4573, 'Melanocyte': 0, 'Other': 947}`. Unpaired predicted counts: `{'Background': 105, 'Immune': 1919, 'Stromal': 7314, 'Epithelial': 896, 'Melanocyte': 0, 'Other': 7}`. Background/untyped predicted instances: `126`.

#### Specialist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 15 | 1,900 | 46,015 | 3,335 | 0 | 4 |
| Stromal | 17 | 270 | 19,838 | 877 | 0 | 1 |
| Epithelial | 12 | 748 | 10,823 | 1,087 | 0 | 2 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 6 | 142 | 903 | 34 | 0 | 2 |

Unpaired true counts: `{'Background': 0, 'Immune': 7168, 'Stromal': 5658, 'Epithelial': 5009, 'Melanocyte': 0, 'Other': 1019}`. Unpaired predicted counts: `{'Background': 227, 'Immune': 734, 'Stromal': 10204, 'Epithelial': 856, 'Melanocyte': 0, 'Other': 16}`. Background/untyped predicted instances: `282`.

### Training and validation curves

Validation values below are diagnostic curves only. Canonical comparison values above come exclusively from checkpoint-10 TEST inference.

| Epoch | Gen train loss | Gen val loss | Gen val bPQ | Gen val mPQ | Spec train loss | Spec val loss | Spec val bPQ | Spec val mPQ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.2801 | 6.0342 | 0.5210 | 0.2431 | 8.5855 | 7.2428 | 0.3199 | 0.1269 |
| 2 | 5.6988 | 5.6242 | 0.5353 | 0.2764 | 6.1326 | 6.0860 | 0.3259 | 0.1705 |
| 3 | 5.3626 | 5.4645 | 0.5446 | 0.2934 | 5.4787 | 5.7996 | 0.3498 | 0.1953 |
| 4 | 4.6986 | 4.5709 | 0.5383 | 0.2965 | 5.2107 | 5.5781 | 0.3755 | 0.2097 |
| 5 | 4.1858 | 4.3902 | 0.5416 | 0.3005 | 5.0466 | 5.4308 | 0.3808 | 0.2156 |
| 6 | 4.0741 | 4.3223 | 0.5365 | 0.2965 | 4.9260 | 5.4127 | 0.3963 | 0.2248 |
| 7 | 4.0084 | 4.1869 | 0.5456 | 0.3101 | 4.8500 | 5.3627 | 0.3912 | 0.2091 |
| 8 | 3.9542 | 4.2390 | 0.5393 | 0.3084 | 4.7786 | 5.3386 | 0.3909 | 0.2207 |
| 9 | 3.9242 | 4.1735 | 0.5394 | 0.3088 | 4.7625 | 5.3061 | 0.3929 | 0.2216 |
| 10 | 3.8766 | 4.2205 | 0.5463 | 0.3133 | 4.7001 | 5.3116 | 0.3678 | 0.2007 |

Sources: `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-19T105648_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; `run/sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T220941_sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; split metadata under `cellvit_ready/sthelar40x_klt_5class_slideind_foldA_margin128` and `cellvit_ready/sthelar40x_kidney_5class_slideind_foldA_margin128`.

## Kidney — Fold B

Exact matched population: **6,723 patches**. Generalist raw test coverage: 26,335; specialist raw test coverage: 6,723.

### Data volume and class distribution

| Model | Split | Patches | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|---:|---:|---:|---:|---:|---:|
| Generalist | Train | 20,893 | 593,561 (53.32%) | 236,384 (21.23%) | 249,467 (22.41%) | 0 (0.00%) | 33,846 (3.04%) |
| Generalist | Validation | 2,601 | 60,652 (47.22%) | 27,793 (21.64%) | 31,487 (24.51%) | 0 (0.00%) | 8,519 (6.63%) |
| Generalist | Matched TEST | 6,723 | 20,030 (10.30%) | 55,988 (28.79%) | 108,430 (55.75%) | 0 (0.00%) | 10,054 (5.17%) |
| Specialist | Train | 3,912 | 52,988 (57.13%) | 24,057 (25.94%) | 13,873 (14.96%) | 0 (0.00%) | 1,837 (1.98%) |
| Specialist | Validation | 537 | 6,285 (45.97%) | 3,027 (22.14%) | 4,071 (29.77%) | 0 (0.00%) | 290 (2.12%) |
| Specialist | Matched TEST | 6,723 | 20,030 (10.30%) | 55,988 (28.79%) | 108,430 (55.75%) | 0 (0.00%) | 10,054 (5.17%) |

| Model | TVD | JSD bits | Specialist/generalist train-patch ratio | Absent, rare, or severely underrepresented training classes |
|---|---:|---:|---:|---|
| Generalist | 0.430 | 0.174 | 1.000 | Melanocyte absent |
| Specialist | 0.468 | 0.225 | 0.187 | Melanocyte absent |

Declared thresholds: rare = <1% of training annotation occurrences; severe underrepresentation = training frequency <25% of matched-test frequency when the test frequency is at least 1%.

### Canonical matched TEST metrics and paired type performance

| Model | Dice | bPQ | mPQ | F1det | F1type |
|---|---:|---:|---:|---:|---:|
| Generalist | 0.8507 | 0.6409 | 0.2372 | 0.9035 | 0.3537 |
| Specialist | 0.8343 | 0.6094 | 0.0529 | 0.9008 | 0.1060 |

| Model | Class | Matched true support | Total true support | Paired type F1 | Predicted count | Predicted frequency |
|---|---|---:|---:|---:|---:|---:|
| Generalist | Immune | 16,649 | 19,734 | 0.2764 | 37,606 | 21.38% |
| Generalist | Stromal | 47,665 | 55,119 | 0.5023 | 82,825 | 47.09% |
| Generalist | Epithelial | 94,510 | 106,830 | 0.6360 | 55,458 | 31.53% |
| Generalist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Generalist | Other | 7,256 | 9,918 | 0.0003 | 2 | 0.00% |
| Specialist | Immune | 16,255 | 19,732 | 0.1724 | 145,562 | 85.62% |
| Specialist | Stromal | 47,028 | 55,116 | 0.1562 | 9,451 | 5.56% |
| Specialist | Epithelial | 92,912 | 106,874 | 0.0953 | 14,991 | 8.82% |
| Specialist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Specialist | Other | 6,729 | 9,921 | 0.0000 | 0 | 0.00% |

#### Generalist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 7 | 7,195 | 7,990 | 1,464 | 0 | 0 |
| Stromal | 20 | 12,434 | 31,391 | 3,839 | 0 | 1 |
| Epithelial | 94 | 14,062 | 33,438 | 47,010 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 9 | 1,727 | 4,512 | 1,016 | 0 | 1 |

Unpaired true counts: `{'Background': 1, 'Immune': 3085, 'Stromal': 7454, 'Epithelial': 12320, 'Melanocyte': 0, 'Other': 2662}`. Unpaired predicted counts: `{'Background': 168, 'Immune': 2188, 'Stromal': 5494, 'Epithelial': 2129, 'Melanocyte': 0, 'Other': 0}`. Background/untyped predicted instances: `305`.

#### Specialist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 9 | 13,537 | 709 | 2,009 | 0 | 0 |
| Stromal | 23 | 38,269 | 4,345 | 4,414 | 0 | 0 |
| Epithelial | 50 | 84,882 | 2,956 | 5,074 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 6 | 4,101 | 600 | 2,028 | 0 | 0 |

Unpaired true counts: `{'Background': 0, 'Immune': 3477, 'Stromal': 8088, 'Epithelial': 13962, 'Melanocyte': 0, 'Other': 3192}`. Unpaired predicted counts: `{'Background': 94, 'Immune': 4773, 'Stromal': 841, 'Epithelial': 1466, 'Melanocyte': 0, 'Other': 0}`. Background/untyped predicted instances: `190`.

### Training and validation curves

Validation values below are diagnostic curves only. Canonical comparison values above come exclusively from checkpoint-10 TEST inference.

| Epoch | Gen train loss | Gen val loss | Gen val bPQ | Gen val mPQ | Spec train loss | Spec val loss | Spec val bPQ | Spec val mPQ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.6933 | 6.2509 | 0.4271 | 0.2228 | 9.4157 | 8.3336 | 0.4848 | 0.1914 |
| 2 | 5.9817 | 5.7965 | 0.4598 | 0.2598 | 7.2600 | 6.5194 | 0.4989 | 0.1888 |
| 3 | 5.6374 | 5.7757 | 0.4705 | 0.2686 | 6.3518 | 6.0194 | 0.5200 | 0.2024 |
| 4 | 5.3565 | 5.1835 | 0.4646 | 0.2650 | 5.9577 | 5.7661 | 0.5115 | 0.1942 |
| 5 | 4.5897 | 4.5181 | 0.4755 | 0.2776 | 5.7527 | 5.6133 | 0.5304 | 0.2192 |
| 6 | 4.3600 | 4.4860 | 0.4707 | 0.2734 | 5.6025 | 5.4634 | 0.5356 | 0.2396 |
| 7 | 4.2619 | 4.4230 | 0.4522 | 0.2712 | 5.5091 | 5.4318 | 0.5212 | 0.2356 |
| 8 | 4.2049 | 4.3043 | 0.4751 | 0.2807 | 5.4302 | 5.3576 | 0.5377 | 0.2443 |
| 9 | 4.1466 | 4.2702 | 0.4784 | 0.2839 | 5.3578 | 5.2826 | 0.5520 | 0.2494 |
| 10 | 4.1199 | 4.2741 | 0.4751 | 0.2813 | 5.3012 | 5.2244 | 0.5543 | 0.2661 |

Sources: `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-20T112527_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; `run/sthelar40x_kidney_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-29T060654_sthelar40x_kidney_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; split metadata under `cellvit_ready/sthelar40x_klt_5class_slideind_foldB_margin128` and `cellvit_ready/sthelar40x_kidney_5class_slideind_foldB_margin128`.

## Liver — Fold A

Exact matched population: **9,166 patches**. Generalist raw test coverage: 23,494; specialist raw test coverage: 9,166.

### Data volume and class distribution

| Model | Split | Patches | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|---:|---:|---:|---:|---:|---:|
| Generalist | Train | 24,283 | 798,175 (55.12%) | 270,672 (18.69%) | 356,338 (24.61%) | 0 (0.00%) | 22,934 (1.58%) |
| Generalist | Validation | 2,052 | 80,971 (59.95%) | 26,407 (19.55%) | 26,653 (19.73%) | 0 (0.00%) | 1,039 (0.77%) |
| Generalist | Matched TEST | 9,166 | 67,212 (20.94%) | 73,797 (22.99%) | 178,802 (55.71%) | 0 (0.00%) | 1,134 (0.35%) |
| Specialist | Train | 18,780 | 99,797 (22.19%) | 90,574 (20.14%) | 254,218 (56.53%) | 0 (0.00%) | 5,131 (1.14%) |
| Specialist | Validation | 1,647 | 8,172 (21.99%) | 7,995 (21.51%) | 20,681 (55.65%) | 0 (0.00%) | 315 (0.85%) |
| Specialist | Matched TEST | 9,166 | 67,212 (20.94%) | 73,797 (22.99%) | 178,802 (55.71%) | 0 (0.00%) | 1,134 (0.35%) |

| Model | TVD | JSD bits | Specialist/generalist train-patch ratio | Absent, rare, or severely underrepresented training classes |
|---|---:|---:|---:|---|
| Generalist | 0.354 | 0.107 | 1.000 | Melanocyte absent |
| Specialist | 0.029 | 0.002 | 0.773 | Melanocyte absent |

Declared thresholds: rare = <1% of training annotation occurrences; severe underrepresentation = training frequency <25% of matched-test frequency when the test frequency is at least 1%.

### Canonical matched TEST metrics and paired type performance

| Model | Dice | bPQ | mPQ | F1det | F1type |
|---|---:|---:|---:|---:|---:|
| Generalist | 0.8406 | 0.5774 | 0.2316 | 0.8722 | 0.3595 |
| Specialist | 0.7956 | 0.4801 | 0.1715 | 0.8432 | 0.2649 |

| Model | Class | Matched true support | Total true support | Paired type F1 | Predicted count | Predicted frequency |
|---|---|---:|---:|---:|---:|---:|
| Generalist | Immune | 56,187 | 66,225 | 0.1973 | 58,472 | 20.13% |
| Generalist | Stromal | 60,899 | 72,718 | 0.5753 | 139,453 | 48.02% |
| Generalist | Epithelial | 146,973 | 176,168 | 0.6655 | 92,475 | 31.84% |
| Generalist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Generalist | Other | 545 | 1,125 | 0.0000 | 9 | 0.00% |
| Specialist | Immune | 53,118 | 66,191 | 0.0173 | 1,030 | 0.36% |
| Specialist | Stromal | 56,751 | 72,684 | 0.4461 | 202,778 | 71.48% |
| Specialist | Epithelial | 142,638 | 175,936 | 0.5962 | 79,874 | 28.16% |
| Specialist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Specialist | Other | 419 | 1,124 | 0.0000 | 5 | 0.00% |

#### Generalist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 4 | 10,686 | 42,358 | 3,143 | 0 | 0 |
| Stromal | 10 | 2,252 | 54,316 | 4,330 | 0 | 1 |
| Epithelial | 10 | 39,102 | 30,834 | 77,033 | 0 | 4 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 76 | 433 | 36 | 0 | 0 |

Unpaired true counts: `{'Background': 1, 'Immune': 10038, 'Stromal': 11819, 'Epithelial': 29195, 'Melanocyte': 0, 'Other': 580}`. Unpaired predicted counts: `{'Background': 113, 'Immune': 6356, 'Stromal': 11512, 'Epithelial': 7933, 'Melanocyte': 0, 'Other': 4}`. Background/untyped predicted instances: `139`.

#### Specialist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 1 | 0 | 0 | 0 | 0 | 0 |
| Immune | 38 | 467 | 50,198 | 2,452 | 0 | 1 |
| Stromal | 44 | 96 | 53,619 | 3,035 | 0 | 1 |
| Epithelial | 242 | 309 | 79,410 | 62,916 | 0 | 3 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 0 | 394 | 25 | 0 | 0 |

Unpaired true counts: `{'Background': 2, 'Immune': 13073, 'Stromal': 15933, 'Epithelial': 33298, 'Melanocyte': 0, 'Other': 705}`. Unpaired predicted counts: `{'Background': 448, 'Immune': 158, 'Stromal': 19157, 'Epithelial': 11446, 'Melanocyte': 0, 'Other': 0}`. Background/untyped predicted instances: `774`.

### Training and validation curves

Validation values below are diagnostic curves only. Canonical comparison values above come exclusively from checkpoint-10 TEST inference.

| Epoch | Gen train loss | Gen val loss | Gen val bPQ | Gen val mPQ | Spec train loss | Spec val loss | Spec val bPQ | Spec val mPQ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.2801 | 6.0342 | 0.5210 | 0.2431 | 6.7392 | 5.3936 | 0.5843 | 0.2944 |
| 2 | 5.6988 | 5.6242 | 0.5353 | 0.2764 | 5.0010 | 4.8004 | 0.6094 | 0.3750 |
| 3 | 5.3626 | 5.4645 | 0.5446 | 0.2934 | 4.6921 | 4.7606 | 0.5948 | 0.3824 |
| 4 | 4.6986 | 4.5709 | 0.5383 | 0.2965 | 4.4613 | 4.2816 | 0.6061 | 0.3935 |
| 5 | 4.1858 | 4.3902 | 0.5416 | 0.3005 | 3.6826 | 3.6134 | 0.6099 | 0.3931 |
| 6 | 4.0741 | 4.3223 | 0.5365 | 0.2965 | 3.4662 | 3.5192 | 0.5974 | 0.3949 |
| 7 | 4.0084 | 4.1869 | 0.5456 | 0.3101 | 3.3947 | 3.4305 | 0.5942 | 0.3930 |
| 8 | 3.9542 | 4.2390 | 0.5393 | 0.3084 | 3.3554 | 3.3316 | 0.6034 | 0.4041 |
| 9 | 3.9242 | 4.1735 | 0.5394 | 0.3088 | 3.3096 | 3.2976 | 0.6235 | 0.4208 |
| 10 | 3.8766 | 4.2205 | 0.5463 | 0.3133 | 3.2723 | 3.2405 | 0.6127 | 0.4176 |

Sources: `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-19T105648_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; `run/sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T220941_sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; split metadata under `cellvit_ready/sthelar40x_klt_5class_slideind_foldA_margin128` and `cellvit_ready/sthelar40x_liver_5class_slideind_foldA_margin128`.

## Liver — Fold B

Exact matched population: **9,756 patches**. Generalist raw test coverage: 26,335; specialist raw test coverage: 20,427.

### Data volume and class distribution

| Model | Split | Patches | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|---:|---:|---:|---:|---:|---:|
| Generalist | Train | 20,893 | 593,561 (53.32%) | 236,384 (21.23%) | 249,467 (22.41%) | 0 (0.00%) | 33,846 (3.04%) |
| Generalist | Validation | 2,601 | 60,652 (47.22%) | 27,793 (21.64%) | 31,487 (24.51%) | 0 (0.00%) | 8,519 (6.63%) |
| Generalist | Matched TEST | 9,756 | 51,384 (22.06%) | 47,080 (20.21%) | 131,891 (56.61%) | 0 (0.00%) | 2,615 (1.12%) |
| Specialist | Train | 8,174 | 63,296 (21.59%) | 68,778 (23.46%) | 160,100 (54.61%) | 0 (0.00%) | 1,016 (0.35%) |
| Specialist | Validation | 992 | 3,916 (14.11%) | 5,019 (18.08%) | 18,702 (67.38%) | 0 (0.00%) | 118 (0.43%) |
| Specialist | Matched TEST | 9,756 | 51,384 (22.06%) | 47,080 (20.21%) | 131,891 (56.61%) | 0 (0.00%) | 2,615 (1.12%) |

| Model | TVD | JSD bits | Specialist/generalist train-patch ratio | Absent, rare, or severely underrepresented training classes |
|---|---:|---:|---:|---|
| Generalist | 0.342 | 0.107 | 1.000 | Melanocyte absent |
| Specialist | 0.032 | 0.003 | 0.391 | Melanocyte absent; Other rare (0.35%) |

Declared thresholds: rare = <1% of training annotation occurrences; severe underrepresentation = training frequency <25% of matched-test frequency when the test frequency is at least 1%.

### Canonical matched TEST metrics and paired type performance

| Model | Dice | bPQ | mPQ | F1det | F1type |
|---|---:|---:|---:|---:|---:|
| Generalist | 0.8320 | 0.6257 | 0.2120 | 0.8977 | 0.3121 |
| Specialist | 0.7771 | 0.5225 | 0.2455 | 0.8836 | 0.4047 |

| Model | Class | Matched true support | Total true support | Paired type F1 | Predicted count | Predicted frequency |
|---|---|---:|---:|---:|---:|---:|
| Generalist | Immune | 43,250 | 50,651 | 0.0777 | 4,360 | 2.08% |
| Generalist | Stromal | 37,102 | 46,356 | 0.3983 | 24,640 | 11.75% |
| Generalist | Epithelial | 115,644 | 130,042 | 0.7725 | 180,698 | 86.17% |
| Generalist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Generalist | Other | 1,299 | 2,569 | 0.0000 | 8 | 0.00% |
| Specialist | Immune | 42,576 | 50,671 | 0.3314 | 13,852 | 6.57% |
| Specialist | Stromal | 36,438 | 46,367 | 0.4682 | 24,011 | 11.39% |
| Specialist | Epithelial | 114,334 | 130,087 | 0.8192 | 172,872 | 82.03% |
| Specialist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Specialist | Other | 1,292 | 2,567 | 0.0000 | 0 | 0.00% |

#### Generalist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 30 | 1,833 | 5,657 | 35,760 | 0 | 0 |
| Stromal | 25 | 1,050 | 11,671 | 24,381 | 0 | 0 |
| Epithelial | 59 | 778 | 3,802 | 111,061 | 0 | 3 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 255 | 368 | 676 | 0 | 0 |

Unpaired true counts: `{'Background': 0, 'Immune': 7401, 'Stromal': 9254, 'Epithelial': 14398, 'Melanocyte': 0, 'Other': 1270}`. Unpaired predicted counts: `{'Background': 239, 'Immune': 444, 'Stromal': 3142, 'Epithelial': 8820, 'Melanocyte': 0, 'Other': 5}`. Background/untyped predicted instances: `355`.

#### Specialist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 10 | 9,171 | 6,538 | 26,867 | 0 | 0 |
| Stromal | 14 | 2,615 | 13,551 | 20,272 | 0 | 0 |
| Epithelial | 14 | 581 | 1,212 | 112,541 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 2 | 399 | 144 | 749 | 0 | 0 |

Unpaired true counts: `{'Background': 0, 'Immune': 8095, 'Stromal': 9929, 'Epithelial': 15753, 'Melanocyte': 0, 'Other': 1275}`. Unpaired predicted counts: `{'Background': 140, 'Immune': 1086, 'Stromal': 2566, 'Epithelial': 12443, 'Melanocyte': 0, 'Other': 0}`. Background/untyped predicted instances: `182`.

### Training and validation curves

Validation values below are diagnostic curves only. Canonical comparison values above come exclusively from checkpoint-10 TEST inference.

| Epoch | Gen train loss | Gen val loss | Gen val bPQ | Gen val mPQ | Spec train loss | Spec val loss | Spec val bPQ | Spec val mPQ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.6933 | 6.2509 | 0.4271 | 0.2228 | 8.4789 | 6.2572 | 0.5360 | 0.2974 |
| 2 | 5.9817 | 5.7965 | 0.4598 | 0.2598 | 6.0902 | 5.5238 | 0.5411 | 0.3270 |
| 3 | 5.6374 | 5.7757 | 0.4705 | 0.2686 | 5.5401 | 5.1400 | 0.5590 | 0.3509 |
| 4 | 5.3565 | 5.1835 | 0.4646 | 0.2650 | 5.2389 | 4.9856 | 0.5681 | 0.3695 |
| 5 | 4.5897 | 4.5181 | 0.4755 | 0.2776 | 5.0850 | 4.8380 | 0.5753 | 0.3811 |
| 6 | 4.3600 | 4.4860 | 0.4707 | 0.2734 | 4.9490 | 4.7376 | 0.5731 | 0.3670 |
| 7 | 4.2619 | 4.4230 | 0.4522 | 0.2712 | 4.8852 | 4.6780 | 0.5802 | 0.3768 |
| 8 | 4.2049 | 4.3043 | 0.4751 | 0.2807 | 4.8106 | 4.6455 | 0.5824 | 0.3860 |
| 9 | 4.1466 | 4.2702 | 0.4784 | 0.2839 | 4.7612 | 4.5991 | 0.5855 | 0.3887 |
| 10 | 4.1199 | 4.2741 | 0.4751 | 0.2813 | 4.6990 | 4.5296 | 0.5848 | 0.3902 |

Sources: `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-20T112527_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; `run/sthelar40x_liver_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-29T060941_sthelar40x_liver_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; split metadata under `cellvit_ready/sthelar40x_klt_5class_slideind_foldB_margin128` and `cellvit_ready/sthelar40x_liver_5class_slideind_foldB_margin128`.

## Tonsil — Fold A

Exact matched population: **9,879 patches**. Generalist raw test coverage: 23,494; specialist raw test coverage: 21,083.

### Data volume and class distribution

| Model | Split | Patches | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|---:|---:|---:|---:|---:|---:|
| Generalist | Train | 24,283 | 798,175 (55.12%) | 270,672 (18.69%) | 356,338 (24.61%) | 0 (0.00%) | 22,934 (1.58%) |
| Generalist | Validation | 2,052 | 80,971 (59.95%) | 26,407 (19.55%) | 26,653 (19.73%) | 0 (0.00%) | 1,039 (0.77%) |
| Generalist | Matched TEST | 9,879 | 527,728 (64.80%) | 163,296 (20.05%) | 84,208 (10.34%) | 0 (0.00%) | 39,104 (4.80%) |
| Specialist | Train | 21,001 | 1,705,169 (69.82%) | 408,278 (16.72%) | 303,671 (12.43%) | 0 (0.00%) | 25,111 (1.03%) |
| Specialist | Validation | 2,066 | 178,844 (68.44%) | 44,368 (16.98%) | 36,111 (13.82%) | 0 (0.00%) | 1,985 (0.76%) |
| Specialist | Matched TEST | 9,879 | 527,728 (64.80%) | 163,296 (20.05%) | 84,208 (10.34%) | 0 (0.00%) | 39,104 (4.80%) |

| Model | TVD | JSD bits | Specialist/generalist train-patch ratio | Absent, rare, or severely underrepresented training classes |
|---|---:|---:|---:|---|
| Generalist | 0.143 | 0.031 | 1.000 | Melanocyte absent |
| Specialist | 0.071 | 0.012 | 0.865 | Melanocyte absent; Other severely underrepresented (train 1.03%, test 4.80%) |

Declared thresholds: rare = <1% of training annotation occurrences; severe underrepresentation = training frequency <25% of matched-test frequency when the test frequency is at least 1%.

### Canonical matched TEST metrics and paired type performance

| Model | Dice | bPQ | mPQ | F1det | F1type |
|---|---:|---:|---:|---:|---:|
| Generalist | 0.7347 | 0.4408 | 0.2188 | 0.8279 | 0.5073 |
| Specialist | 0.7355 | 0.4503 | 0.2158 | 0.8379 | 0.4879 |

| Model | Class | Matched true support | Total true support | Paired type F1 | Predicted count | Predicted frequency |
|---|---|---:|---:|---:|---:|---:|
| Generalist | Immune | 428,001 | 519,982 | 0.8428 | 498,361 | 68.64% |
| Generalist | Stromal | 121,315 | 160,821 | 0.5088 | 114,880 | 15.82% |
| Generalist | Epithelial | 65,389 | 82,911 | 0.6774 | 112,849 | 15.54% |
| Generalist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Generalist | Other | 18,059 | 38,511 | 0.0001 | 5 | 0.00% |
| Specialist | Immune | 429,416 | 520,010 | 0.8088 | 424,471 | 58.77% |
| Specialist | Stromal | 123,092 | 160,887 | 0.5066 | 152,891 | 21.17% |
| Specialist | Epithelial | 65,688 | 82,951 | 0.6361 | 144,906 | 20.06% |
| Specialist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Specialist | Other | 20,589 | 38,541 | 0.0000 | 0 | 0.00% |

#### Generalist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 48 | 363,446 | 30,669 | 33,886 | 0 | 0 |
| Stromal | 78 | 56,473 | 56,116 | 8,724 | 0 | 2 |
| Epithelial | 46 | 6,232 | 3,480 | 55,677 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 34 | 8,354 | 8,995 | 709 | 0 | 1 |

Unpaired true counts: `{'Background': 0, 'Immune': 91981, 'Stromal': 39506, 'Epithelial': 17522, 'Melanocyte': 0, 'Other': 20452}`. Unpaired predicted counts: `{'Background': 430, 'Immune': 63856, 'Stromal': 15620, 'Epithelial': 13853, 'Melanocyte': 0, 'Other': 2}`. Background/untyped predicted instances: `639`.

#### Specialist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 20 | 326,540 | 59,138 | 43,738 | 0 | 0 |
| Stromal | 12 | 41,165 | 64,921 | 17,006 | 0 | 0 |
| Epithelial | 6 | 3,174 | 1,068 | 61,446 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 4 | 7,200 | 8,070 | 5,319 | 0 | 0 |

Unpaired true counts: `{'Background': 0, 'Immune': 90594, 'Stromal': 37795, 'Epithelial': 17263, 'Melanocyte': 0, 'Other': 17952}`. Unpaired predicted counts: `{'Background': 167, 'Immune': 46392, 'Stromal': 19694, 'Epithelial': 17397, 'Melanocyte': 0, 'Other': 0}`. Background/untyped predicted instances: `212`.

### Training and validation curves

Validation values below are diagnostic curves only. Canonical comparison values above come exclusively from checkpoint-10 TEST inference.

| Epoch | Gen train loss | Gen val loss | Gen val bPQ | Gen val mPQ | Spec train loss | Spec val loss | Spec val bPQ | Spec val mPQ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.2801 | 6.0342 | 0.5210 | 0.2431 | 8.8348 | 7.2354 | 0.5142 | 0.2390 |
| 2 | 5.6988 | 5.6242 | 0.5353 | 0.2764 | 6.8927 | 6.4197 | 0.5512 | 0.2562 |
| 3 | 5.3626 | 5.4645 | 0.5446 | 0.2934 | 6.3949 | 6.1889 | 0.5664 | 0.2727 |
| 4 | 4.6986 | 4.5709 | 0.5383 | 0.2965 | 6.1600 | 5.8566 | 0.5625 | 0.2799 |
| 5 | 4.1858 | 4.3902 | 0.5416 | 0.3005 | 5.4782 | 5.1446 | 0.5646 | 0.2837 |
| 6 | 4.0741 | 4.3223 | 0.5365 | 0.2965 | 5.0956 | 5.0198 | 0.5586 | 0.2741 |
| 7 | 4.0084 | 4.1869 | 0.5456 | 0.3101 | 4.9884 | 4.8629 | 0.5686 | 0.2884 |
| 8 | 3.9542 | 4.2390 | 0.5393 | 0.3084 | 4.9077 | 4.8020 | 0.5708 | 0.2845 |
| 9 | 3.9242 | 4.1735 | 0.5394 | 0.3088 | 4.8626 | 4.8448 | 0.5706 | 0.2785 |
| 10 | 3.8766 | 4.2205 | 0.5463 | 0.3133 | 4.8316 | 4.9102 | 0.5567 | 0.2895 |

Sources: `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-19T105648_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; `run/sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T192051_sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; split metadata under `cellvit_ready/sthelar40x_klt_5class_slideind_foldA_margin128` and `cellvit_ready/sthelar40x_tonsil_5class_slideind_foldA_margin128_cap50000`.

## Tonsil — Fold B

Exact matched population: **9,856 patches**. Generalist raw test coverage: 26,335; specialist raw test coverage: 23,067.

### Data volume and class distribution

| Model | Split | Patches | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|---:|---:|---:|---:|---:|---:|
| Generalist | Train | 20,893 | 593,561 (53.32%) | 236,384 (21.23%) | 249,467 (22.41%) | 0 (0.00%) | 33,846 (3.04%) |
| Generalist | Validation | 2,601 | 60,652 (47.22%) | 27,793 (21.64%) | 31,487 (24.51%) | 0 (0.00%) | 8,519 (6.63%) |
| Generalist | Matched TEST | 9,856 | 807,732 (69.89%) | 194,011 (16.79%) | 142,670 (12.34%) | 0 (0.00%) | 11,304 (0.98%) |
| Specialist | Train | 18,831 | 1,024,873 (65.75%) | 305,681 (19.61%) | 160,843 (10.32%) | 0 (0.00%) | 67,263 (4.32%) |
| Specialist | Validation | 2,252 | 106,694 (58.21%) | 40,582 (22.14%) | 18,929 (10.33%) | 0 (0.00%) | 17,099 (9.33%) |
| Specialist | Matched TEST | 9,856 | 807,732 (69.89%) | 194,011 (16.79%) | 142,670 (12.34%) | 0 (0.00%) | 11,304 (0.98%) |

| Model | TVD | JSD bits | Specialist/generalist train-patch ratio | Absent, rare, or severely underrepresented training classes |
|---|---:|---:|---:|---|
| Generalist | 0.166 | 0.025 | 1.000 | Melanocyte absent |
| Specialist | 0.062 | 0.010 | 0.901 | Melanocyte absent |

Declared thresholds: rare = <1% of training annotation occurrences; severe underrepresentation = training frequency <25% of matched-test frequency when the test frequency is at least 1%.

### Canonical matched TEST metrics and paired type performance

| Model | Dice | bPQ | mPQ | F1det | F1type |
|---|---:|---:|---:|---:|---:|
| Generalist | 0.8311 | 0.5723 | 0.2443 | 0.8493 | 0.4540 |
| Specialist | 0.8320 | 0.5767 | 0.2606 | 0.8560 | 0.5134 |

| Model | Class | Matched true support | Total true support | Paired type F1 | Predicted count | Predicted frequency |
|---|---|---:|---:|---:|---:|---:|
| Generalist | Immune | 664,098 | 795,858 | 0.8662 | 946,633 | 87.05% |
| Generalist | Stromal | 158,070 | 191,136 | 0.3265 | 70,705 | 6.50% |
| Generalist | Epithelial | 116,714 | 140,515 | 0.6233 | 70,081 | 6.44% |
| Generalist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Generalist | Other | 6,556 | 11,136 | 0.0000 | 11 | 0.00% |
| Specialist | Immune | 665,146 | 795,780 | 0.8623 | 820,449 | 76.64% |
| Specialist | Stromal | 157,914 | 191,124 | 0.4104 | 160,684 | 15.01% |
| Specialist | Epithelial | 116,267 | 140,460 | 0.6805 | 85,869 | 8.02% |
| Specialist | Melanocyte | 0 | 0 | -- | 0 | 0.00% |
| Specialist | Other | 6,239 | 11,107 | 0.1004 | 3,504 | 0.33% |

#### Generalist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 61 | 644,859 | 14,152 | 5,085 | 0 | 2 |
| Stromal | 43 | 122,056 | 35,500 | 513 | 0 | 1 |
| Epithelial | 75 | 52,458 | 8,796 | 55,459 | 0 | 1 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 4 | 5,413 | 958 | 185 | 0 | 0 |

Unpaired true counts: `{'Background': 0, 'Immune': 131760, 'Stromal': 33066, 'Epithelial': 23801, 'Melanocyte': 0, 'Other': 4580}`. Unpaired predicted counts: `{'Background': 293, 'Immune': 121847, 'Stromal': 11299, 'Epithelial': 8839, 'Melanocyte': 0, 'Other': 7}`. Background/untyped predicted instances: `484`.

#### Specialist full paired confusion matrix

Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 139 | 601,397 | 52,483 | 10,307 | 0 | 959 |
| Stromal | 55 | 96,248 | 60,363 | 653 | 0 | 650 |
| Epithelial | 130 | 28,396 | 21,435 | 65,674 | 0 | 762 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 33 | 3,690 | 1,981 | 113 | 0 | 455 |

Unpaired true counts: `{'Background': 0, 'Immune': 130634, 'Stromal': 33210, 'Epithelial': 24193, 'Melanocyte': 0, 'Other': 4868}`. Unpaired predicted counts: `{'Background': 319, 'Immune': 90718, 'Stromal': 24422, 'Epithelial': 9122, 'Melanocyte': 0, 'Other': 678}`. Background/untyped predicted instances: `684`.

### Training and validation curves

Validation values below are diagnostic curves only. Canonical comparison values above come exclusively from checkpoint-10 TEST inference.

| Epoch | Gen train loss | Gen val loss | Gen val bPQ | Gen val mPQ | Spec train loss | Spec val loss | Spec val bPQ | Spec val mPQ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.6933 | 6.2509 | 0.4271 | 0.2228 | 8.7114 | 7.4852 | 0.2875 | 0.1563 |
| 2 | 5.9817 | 5.7965 | 0.4598 | 0.2598 | 6.7595 | 6.8785 | 0.3228 | 0.1787 |
| 3 | 5.6374 | 5.7757 | 0.4705 | 0.2686 | 6.2855 | 6.6773 | 0.3097 | 0.1755 |
| 4 | 5.3565 | 5.1835 | 0.4646 | 0.2650 | 6.1080 | 6.5464 | 0.3274 | 0.1843 |
| 5 | 4.5897 | 4.5181 | 0.4755 | 0.2776 | 5.7408 | 5.9570 | 0.3391 | 0.1955 |
| 6 | 4.3600 | 4.4860 | 0.4707 | 0.2734 | 5.1082 | 5.5823 | 0.3512 | 0.1990 |
| 7 | 4.2619 | 4.4230 | 0.4522 | 0.2712 | 4.9218 | 5.4401 | 0.3444 | 0.2028 |
| 8 | 4.2049 | 4.3043 | 0.4751 | 0.2807 | 4.8297 | 5.3549 | 0.3382 | 0.2014 |
| 9 | 4.1466 | 4.2702 | 0.4784 | 0.2839 | 4.7630 | 5.3511 | 0.3365 | 0.2001 |
| 10 | 4.1199 | 4.2741 | 0.4751 | 0.2813 | 4.7351 | 5.3488 | 0.3344 | 0.1999 |

Sources: `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-20T112527_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; `run/sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T192051_sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json`; split metadata under `cellvit_ready/sthelar40x_klt_5class_slideind_foldB_margin128` and `cellvit_ready/sthelar40x_tonsil_5class_slideind_foldB_margin128_cap50000`.

## Evidence-weighted interpretation

### What the seed42 evidence supports

- **Training-set size is a plausible but non-isolated factor.** Every specialist has fewer total training patches than the KLT generalist, but Liver/Tonsil specialists can have more same-tissue patches than the KLT contribution from that tissue. Size is therefore confounded with tissue diversity and split-specific coverage.
- **Reduced cross-tissue diversity is structurally present but not independently randomized.** The generalist sees Kidney, Liver and Tonsil; each specialist sees one slide from one tissue. These runs cannot isolate diversity from sample count, class balance, or slide identity.
- **Class-distribution shift is directly measurable.** The TVD/JSD tables and class-frequency deltas identify missing/rare/underrepresented classes. Melanocyte is absent throughout these KLT tissues and is excluded from present-class F1type.
- **Detection is generally more stable than typing.** Across the six directions, specialist F1det differences are smaller than several F1type/mPQ changes. This argues against describing the Kidney loss as a pure detection collapse; paired type confusion and missing/shifted class support are more directly implicated descriptively.
- **Segmentation quality can still contribute.** bPQ changes are reported beside F1det and type metrics. Liver shows direction-dependent mixtures of segmentation/detection and typing behavior, so no single failure mechanism explains every fold.
- **Stochastic optimization cannot be judged from seed42.** The planned seed43/44 replications are required to determine whether Kidney negativity, Liver directionality, Tonsil near-neutrality, and F1det stability persist.

### What must not be claimed

These descriptive data do not identify a causal mechanism and do not support statistical significance, superiority, equivalence, or non-inferiority. Patches and cell occurrences are not independent biological replicates. Fold A/B are reciprocal slide directions, not an n=2 patient cohort estimate and not seed replicates.

## Machine-readable provenance

All distributions, matrices, per-class metrics, predicted frequencies, shift statistics, and epoch curves are in `reports/tissue_specialist_failure_analysis.csv`. Every row includes its canonical inference or training-log source.
