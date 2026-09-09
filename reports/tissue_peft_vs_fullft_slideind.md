# Tissue-specific Selected PEFT versus FullFT — reciprocal complete-slide protocol

All rows are seed42 and use canonical checkpoint-10 TEST evidence. Each pair was required to have identical tissue, fold, train slide, held-out slide, and exact TEST patch-ID hash. Deltas are signed PEFT−FullFT; recovery is PEFT mPQ / FullFT mPQ. Resource measurements are run-specific A100 measurements, not KLT efficiency-audit averages.

| Backbone | Tissue | Fold | N | PEFT bPQ | FullFT bPQ | Δ bPQ | PEFT mPQ | FullFT mPQ | Δ mPQ | mPQ recovery | PEFT F1det | FullFT F1det | Δ F1det | PEFT F1type | FullFT F1type | Δ F1type |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CellViT-SAM-H | Breast | A | 49569 | 0.411262 | 0.342093 | +0.069168 | 0.238573 | 0.203063 | +0.035510 | 1.1749 | 0.812884 | 0.781922 | +0.030963 | 0.500014 | 0.524465 | -0.024451 |
| CellViT-SAM-H | Breast | B | 45773 | 0.452459 | 0.466097 | -0.013637 | 0.277942 | 0.302617 | -0.024674 | 0.9185 | 0.831757 | 0.823098 | +0.008658 | 0.534307 | 0.579054 | -0.044747 |
| CellViT-SAM-H | Colon | A | 11903 | 0.284721 | 0.245586 | +0.039134 | 0.149549 | 0.127709 | +0.021841 | 1.1710 | 0.750952 | 0.755959 | -0.005007 | 0.426503 | 0.439816 | -0.013313 |
| CellViT-SAM-H | Colon | B | 18464 | 0.286509 | 0.319177 | -0.032668 | 0.119909 | 0.124182 | -0.004273 | 0.9656 | 0.744118 | 0.740695 | +0.003423 | 0.386678 | 0.401244 | -0.014566 |
| CellViT-SAM-H | Kidney | A | 4449 | 0.549529 | 0.591683 | -0.042154 | 0.090364 | 0.181413 | -0.091049 | 0.4981 | 0.847800 | 0.856378 | -0.008578 | 0.149249 | 0.307103 | -0.157854 |
| CellViT-SAM-H | Kidney | B | 6723 | 0.609419 | 0.608680 | +0.000739 | 0.052900 | 0.088821 | -0.035921 | 0.5956 | 0.900830 | 0.894684 | +0.006145 | 0.105982 | 0.224373 | -0.118391 |
| CellViT-SAM-H | Liver | A | 9166 | 0.480070 | 0.493432 | -0.013362 | 0.171506 | 0.146406 | +0.025100 | 1.1714 | 0.843156 | 0.841040 | +0.002116 | 0.264902 | 0.222512 | +0.042390 |
| CellViT-SAM-H | Liver | B | 20427 | 0.521889 | 0.507493 | +0.014396 | 0.244555 | 0.260521 | -0.015966 | 0.9387 | 0.882647 | 0.883564 | -0.000917 | 0.403020 | 0.467050 | -0.064030 |
| CellViT-SAM-H | Lung | A | 20621 | 0.508304 | 0.518004 | -0.009701 | 0.193408 | 0.227549 | -0.034141 | 0.8500 | 0.842963 | 0.836796 | +0.006167 | 0.404211 | 0.491063 | -0.086852 |
| CellViT-SAM-H | Lung | B | 10938 | 0.563651 | 0.585981 | -0.022330 | 0.251397 | 0.292869 | -0.041472 | 0.8584 | 0.861942 | 0.852778 | +0.009164 | 0.470082 | 0.557425 | -0.087343 |
| CellViT-SAM-H | Ovary | A | 25498 | 0.448760 | 0.450118 | -0.001359 | 0.222645 | 0.247301 | -0.024657 | 0.9003 | 0.834355 | 0.822513 | +0.011842 | 0.505403 | 0.591676 | -0.086273 |
| CellViT-SAM-H | Ovary | B | 10039 | 0.387430 | 0.388803 | -0.001373 | 0.154828 | 0.152397 | +0.002431 | 1.0160 | 0.832848 | 0.831309 | +0.001539 | 0.480599 | 0.458089 | +0.022510 |
| CellViT-SAM-H | Pancreatic | A | 12647 | 0.377441 | 0.397845 | -0.020404 | 0.123931 | 0.139285 | -0.015354 | 0.8898 | 0.733776 | 0.743969 | -0.010193 | 0.278763 | 0.332267 | -0.053505 |
| CellViT-SAM-H | Pancreatic | B | 25526 | 0.389166 | 0.376939 | +0.012227 | 0.174534 | 0.147552 | +0.026982 | 1.1829 | 0.725043 | 0.699560 | +0.025483 | 0.416725 | 0.374501 | +0.042224 |
| CellViT-SAM-H | Skin | A | 12790 | 0.440098 | 0.449718 | -0.009620 | 0.134163 | 0.131896 | +0.002266 | 1.0172 | 0.743151 | 0.752794 | -0.009643 | 0.242519 | 0.210268 | +0.032251 |
| CellViT-SAM-H | Skin | B | 11209 | 0.213661 | 0.223973 | -0.010311 | 0.071668 | 0.075462 | -0.003793 | 0.9497 | 0.662776 | 0.708927 | -0.046151 | 0.293581 | 0.272010 | +0.021571 |
| CellViT-SAM-H | Tonsil | A | 21083 | 0.452244 | 0.463236 | -0.010992 | 0.215840 | 0.239026 | -0.023186 | 0.9030 | 0.838259 | 0.826966 | +0.011293 | 0.487866 | 0.538523 | -0.050657 |
| CellViT-SAM-H | Tonsil | B | 23067 | 0.577060 | 0.583670 | -0.006610 | 0.260302 | 0.272190 | -0.011887 | 0.9563 | 0.855994 | 0.856145 | -0.000152 | 0.510764 | 0.526859 | -0.016095 |
| CellViT-256 | Kidney | A | 4449 | 0.468128 | 0.570719 | -0.102592 | 0.098889 | 0.157795 | -0.058905 | 0.6267 | 0.833860 | 0.854946 | -0.021086 | 0.210434 | 0.316974 | -0.106540 |
| CellViT-256 | Kidney | B | 6723 | 0.568279 | 0.594013 | -0.025733 | 0.096474 | 0.091515 | +0.004958 | 1.0542 | 0.893834 | 0.893109 | +0.000725 | 0.172681 | 0.206014 | -0.033333 |
| CellViT-256 | Liver | A | 9166 | 0.479362 | 0.470952 | +0.008410 | 0.073691 | 0.082082 | -0.008392 | 0.8978 | 0.832494 | 0.820610 | +0.011883 | 0.113617 | 0.132705 | -0.019088 |
| CellViT-256 | Liver | B | 20427 | 0.572147 | 0.462403 | +0.109745 | 0.251328 | 0.196174 | +0.055154 | 1.2811 | 0.865266 | 0.877002 | -0.011736 | 0.396318 | 0.389351 | +0.006967 |
| CellViT-256 | Tonsil | A | 21083 | 0.409102 | 0.451610 | -0.042508 | 0.176689 | 0.211481 | -0.034793 | 0.8355 | 0.827759 | 0.820673 | +0.007086 | 0.458659 | 0.524881 | -0.066222 |
| CellViT-256 | Tonsil | B | 23067 | 0.527360 | 0.570982 | -0.043622 | 0.229453 | 0.260605 | -0.031152 | 0.8805 | 0.863211 | 0.854308 | +0.008903 | 0.499848 | 0.531130 | -0.031282 |

## Descriptive summary over 18 matched SAM-H tissue×fold conditions

These are descriptive condition-level summaries. Tissue×fold conditions are not treated as independent biological replicates; no significance, equivalence, or non-inferiority test is performed.

| Metric | PEFT mean | FullFT mean | Mean absolute delta | PEFT higher (of 18) |
|---|---:|---:|---:|---:|
| bPQ | 0.441871 | 0.445141 | 0.018344 | 5 |
| mPQ | 0.174890 | 0.186681 | 0.024472 | 6 |
| F1det | 0.808069 | 0.806061 | 0.010969 | 11 |
| F1type | 0.381176 | 0.417683 | 0.054390 | 5 |

mPQ recovery PEFT/FullFT: minimum 0.4981, median 0.9442, maximum 1.1829.

## Run-specific resource measurements

| Backbone | Tissue | Fold | Method | Trainable % | Train h | Peak train VRAM GiB | Model state bytes | State kind |
|---|---|:---:|---|---:|---:|---:|---:|---|
| CellViT-SAM-H | Breast | A | Selected PEFT | 1.117621 | 11.3217 | 9.362 | -- | not available |
| CellViT-SAM-H | Breast | A | FullFT | 100.000000 | 12.3404 | 15.508 | 8,397,780,609 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Breast | B | Selected PEFT | 1.117621 | 11.6076 | 9.362 | -- | not available |
| CellViT-SAM-H | Breast | B | FullFT | 100.000000 | 12.7364 | 15.508 | 8,397,780,609 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Colon | A | Selected PEFT | 1.117621 | 5.0504 | 9.362 | -- | not available |
| CellViT-SAM-H | Colon | A | FullFT | 100.000000 | 5.7282 | 15.508 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Colon | B | Selected PEFT | 1.117621 | 3.6736 | 9.362 | -- | not available |
| CellViT-SAM-H | Colon | B | FullFT | 100.000000 | 3.7505 | 15.502 | 8,397,780,609 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Kidney | A | Selected PEFT | 1.117621 | 1.7169 | 9.362 | -- | not available |
| CellViT-SAM-H | Kidney | A | FullFT | 100.000000 | 2.1223 | 15.508 | 8,397,788,628 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Kidney | B | Selected PEFT | 1.117621 | 1.1508 | 9.362 | 31,812,052 | adapter safetensors |
| CellViT-SAM-H | Kidney | B | FullFT | 100.000000 | 1.3376 | 15.508 | 8,397,788,628 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Liver | A | Selected PEFT | 1.117621 | 5.2774 | 9.362 | -- | not available |
| CellViT-SAM-H | Liver | A | FullFT | 100.000000 | 5.7421 | 15.508 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Liver | B | Selected PEFT | 1.117621 | 2.4663 | 9.362 | 31,812,036 | adapter safetensors |
| CellViT-SAM-H | Liver | B | FullFT | 100.000000 | 2.6417 | 15.502 | 8,397,788,564 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Lung | A | Selected PEFT | 1.117621 | 2.7405 | 9.362 | -- | not available |
| CellViT-SAM-H | Lung | A | FullFT | 100.000000 | 3.1423 | 15.508 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Lung | B | Selected PEFT | 1.117621 | 5.1444 | 9.362 | -- | not available |
| CellViT-SAM-H | Lung | B | FullFT | 100.000000 | 5.8451 | 15.508 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Ovary | A | Selected PEFT | 1.117621 | 2.5782 | 9.362 | -- | not available |
| CellViT-SAM-H | Ovary | A | FullFT | 100.000000 | 3.1630 | 15.508 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Ovary | B | Selected PEFT | 1.117621 | 6.6976 | 9.362 | -- | not available |
| CellViT-SAM-H | Ovary | B | FullFT | 100.000000 | 7.3482 | 15.502 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Pancreatic | A | Selected PEFT | 1.117621 | 6.0294 | 9.362 | -- | not available |
| CellViT-SAM-H | Pancreatic | A | FullFT | 100.000000 | 6.9337 | 15.502 | 8,397,788,692 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Pancreatic | B | Selected PEFT | 1.117621 | 3.4037 | 9.362 | -- | not available |
| CellViT-SAM-H | Pancreatic | B | FullFT | 100.000000 | 3.7502 | 15.508 | 8,397,788,692 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Skin | A | Selected PEFT | 1.117621 | 2.7142 | 9.362 | -- | not available |
| CellViT-SAM-H | Skin | A | FullFT | 100.000000 | 3.3630 | 15.502 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Skin | B | Selected PEFT | 1.117621 | 3.1934 | 9.362 | -- | not available |
| CellViT-SAM-H | Skin | B | FullFT | 100.000000 | 3.7100 | 15.502 | 8,397,780,545 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Tonsil | A | Selected PEFT | 1.117621 | 8.2469 | 9.362 | -- | not available |
| CellViT-SAM-H | Tonsil | A | FullFT | 100.000000 | 9.3418 | 15.508 | 8,397,780,609 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-SAM-H | Tonsil | B | Selected PEFT | 1.117621 | 6.4804 | 9.362 | -- | not available |
| CellViT-SAM-H | Tonsil | B | FullFT | 100.000000 | 7.0422 | 15.508 | 8,397,780,609 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-256 | Kidney | A | Selected PEFT | 0.794189 | 1.4596 | 2.040 | 1,613,512 | adapter safetensors |
| CellViT-256 | Kidney | A | FullFT | 100.000000 | 1.3759 | 2.667 | 561,447,040 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-256 | Kidney | B | Selected PEFT | 0.794189 | 0.8411 | 2.040 | 1,613,440 | adapter safetensors |
| CellViT-256 | Kidney | B | FullFT | 100.000000 | 0.9065 | 2.667 | 561,442,705 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-256 | Liver | A | Selected PEFT | 0.794189 | 3.9434 | 2.040 | 1,613,512 | adapter safetensors |
| CellViT-256 | Liver | A | FullFT | 100.000000 | 3.8593 | 2.667 | 561,442,705 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-256 | Liver | B | Selected PEFT | 0.794189 | 2.0001 | 2.040 | 1,613,456 | adapter safetensors |
| CellViT-256 | Liver | B | FullFT | 100.000000 | 1.7880 | 2.676 | 561,447,040 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-256 | Tonsil | A | Selected PEFT | 0.794189 | 7.3619 | 2.040 | 1,613,456 | adapter safetensors |
| CellViT-256 | Tonsil | A | FullFT | 100.000000 | 5.9238 | 2.667 | 561,447,104 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |
| CellViT-256 | Tonsil | B | Selected PEFT | 0.794189 | 5.3035 | 2.040 | 1,613,520 | adapter safetensors |
| CellViT-256 | Tonsil | B | FullFT | 100.000000 | 5.3724 | 2.667 | 561,442,769 | canonical FullFT checkpoint size (provenance; weight may be retention-deleted) |

## Provenance

Exact inference and efficiency paths, TEST patch-ID hashes, signed and absolute deltas, parameter counts, inference VRAM, and throughput are retained in `reports/tissue_peft_vs_fullft_slideind.csv`.
