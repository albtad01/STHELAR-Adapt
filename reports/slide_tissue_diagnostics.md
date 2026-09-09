# Workshop slide- and tissue-dependent diagnostics

Generated from repository-local manifests and canonical inference artifacts on 2026-09-01. This is a read-only, descriptive scientific analysis. No run, checkpoint, configuration, adapter, or scheduler state was changed.

## Scope and definitions

Canonical completion requires an `inference_results.json` whose `inference.log` explicitly loads `checkpoint_10.pth` and whose `image_metrics` keys cover the complete manifest TEST set. Cell counts are **annotation occurrences in patches**, not unique cells: overlapping patches can repeat a biological cell. The train partition excludes the spatial validation partition. Entropy is Shannon entropy in bits over the five STHELAR classes (maximum log2(5)=2.322). ‘Rare’ is declared as <1% of training annotation occurrences. TVD and JSD (base 2) are descriptive distances on foreground-class frequency vectors; neither is a significance test.

Slide identifiers establish held-out slides, not patient independence: the repository does not provide a slide-to-patient mapping. Folds are reciprocal directions, not biological replicates, and seeds are stochastic repeats within a fixed fold.

## Part A — KLT Fold A versus Fold B

### KLT composition: every train and held-out slide plus pooled KLT

| Tissue/scope | Fold | Split | Slide(s) | Patches | Annotated cells | Nuclei/patch | Entropy (bits) | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kidney | A | TRAIN | `kidney_s0` | 6,321 | 188,119 | 29.76 | 1.536 | 18,889 (10.04%) | 52,774 (28.05%) | 106,511 (56.62%) | 0 (0.00%) | 9,945 (5.29%) |
| Liver | A | TRAIN | `liver_s0` | 8,980 | 215,196 | 23.96 | 1.484 | 47,576 (22.11%) | 43,173 (20.06%) | 122,007 (56.70%) | 0 (0.00%) | 2,440 (1.13%) |
| Tonsil | A | TRAIN | `tonsil_s0` | 8,982 | 1,044,804 | 116.32 | 1.229 | 731,710 (70.03%) | 174,725 (16.72%) | 127,820 (12.23%) | 0 (0.00%) | 10,549 (1.01%) |
| Pooled KLT | A | TRAIN | `kidney_s0+liver_s0+tonsil_s0` | 24,283 | 1,448,119 | 59.64 | 1.518 | 798,175 (55.12%) | 270,672 (18.69%) | 356,338 (24.61%) | 0 (0.00%) | 22,934 (1.58%) |
| Kidney | A | TEST | `kidney_s1` | 4,449 | 106,428 | 23.92 | 1.519 | 59,273 (55.69%) | 27,084 (25.45%) | 17,944 (16.86%) | 0 (0.00%) | 2,127 (2.00%) |
| Liver | A | TEST | `liver_s1` | 9,166 | 320,945 | 35.01 | 1.459 | 67,212 (20.94%) | 73,797 (22.99%) | 178,802 (55.71%) | 0 (0.00%) | 1,134 (0.35%) |
| Tonsil | A | TEST | `tonsil_s1` | 9,879 | 814,336 | 82.43 | 1.419 | 527,728 (64.80%) | 163,296 (20.05%) | 84,208 (10.34%) | 0 (0.00%) | 39,104 (4.80%) |
| Pooled KLT | A | TEST | `kidney_s1+liver_s1+tonsil_s1` | 23,494 | 1,241,709 | 52.85 | 1.613 | 654,213 (52.69%) | 264,177 (21.28%) | 280,954 (22.63%) | 0 (0.00%) | 42,365 (3.41%) |
| Kidney | B | TRAIN | `kidney_s1` | 3,912 | 92,755 | 23.71 | 1.488 | 52,988 (57.13%) | 24,057 (25.94%) | 13,873 (14.96%) | 0 (0.00%) | 1,837 (1.98%) |
| Liver | B | TRAIN | `liver_s1` | 8,174 | 293,190 | 35.87 | 1.473 | 63,296 (21.59%) | 68,778 (23.46%) | 160,100 (54.61%) | 0 (0.00%) | 1,016 (0.35%) |
| Tonsil | B | TRAIN | `tonsil_s1` | 8,807 | 727,313 | 82.58 | 1.394 | 477,277 (65.62%) | 143,549 (19.74%) | 75,494 (10.38%) | 0 (0.00%) | 30,993 (4.26%) |
| Pooled KLT | B | TRAIN | `kidney_s1+liver_s1+tonsil_s1` | 20,893 | 1,113,258 | 53.28 | 1.595 | 593,561 (53.32%) | 236,384 (21.23%) | 249,467 (22.41%) | 0 (0.00%) | 33,846 (3.04%) |
| Kidney | B | TEST | `kidney_s0` | 6,723 | 194,502 | 28.93 | 1.546 | 20,030 (10.30%) | 55,988 (28.79%) | 108,430 (55.75%) | 0 (0.00%) | 10,054 (5.17%) |
| Liver | B | TEST | `liver_s0` | 9,756 | 232,970 | 23.88 | 1.485 | 51,384 (22.06%) | 47,080 (20.21%) | 131,891 (56.61%) | 0 (0.00%) | 2,615 (1.12%) |
| Tonsil | B | TEST | `tonsil_s0` | 9,856 | 1,155,717 | 117.26 | 1.231 | 807,732 (69.89%) | 194,011 (16.79%) | 142,670 (12.34%) | 0 (0.00%) | 11,304 (0.98%) |
| Pooled KLT | B | TEST | `kidney_s0+liver_s0+tonsil_s0` | 26,335 | 1,583,189 | 60.12 | 1.511 | 879,146 (55.53%) | 297,079 (18.76%) | 382,991 (24.19%) | 0 (0.00%) | 23,973 (1.51%) |

| Tissue/scope | Fold | TVD | JSD (bits) | Absent/rare in train but present in test |
|---|:---:|---:|---:|---|
| Kidney | A | 0.457 | 0.214 | none |
| Liver | A | 0.029 | 0.002 | none |
| Tonsil | A | 0.071 | 0.012 | none |
| Pooled KLT | A | 0.044 | 0.004 | none |
| Kidney | B | 0.468 | 0.225 | none |
| Liver | B | 0.032 | 0.003 | Other rare in train (0.35%) |
| Tonsil | B | 0.062 | 0.010 | none |
| Pooled KLT | B | 0.040 | 0.003 | none |

### Canonical SAM-H seed-level TEST metrics

| Test scope | Method | Fold | Seed | bPQ | mPQ | F1det | F1type |
|---|---|:---:|---:|---:|---:|---:|---:|
| Kidney | Selected PEFT | A | 42 | 0.5766 | 0.1826 | 0.8669 | 0.3011 |
| Kidney | Selected PEFT | A | 43 | 0.5928 | 0.1738 | 0.8717 | 0.2897 |
| Kidney | Selected PEFT | A | 44 | 0.5801 | 0.1700 | 0.8627 | 0.3251 |
| Kidney | Selected PEFT | B | 42 | 0.6409 | 0.2372 | 0.9035 | 0.3537 |
| Kidney | Selected PEFT | B | 43 | 0.6479 | 0.2653 | 0.9064 | 0.3854 |
| Kidney | Selected PEFT | B | 44 | 0.6325 | 0.2239 | 0.9012 | 0.3368 |
| Kidney | FullFT | A | 42 | 0.5854 | 0.1599 | 0.8574 | 0.2940 |
| Kidney | FullFT | A | 43 | 0.5883 | 0.1724 | 0.8628 | 0.2826 |
| Kidney | FullFT | A | 44 | 0.6017 | 0.1827 | 0.8656 | 0.3274 |
| Kidney | FullFT | B | 42 | 0.6475 | 0.2267 | 0.8996 | 0.3829 |
| Kidney | FullFT | B | 43 | 0.6503 | 0.2411 | 0.8979 | 0.3970 |
| Kidney | FullFT | B | 44 | 0.6486 | 0.2458 | 0.8974 | 0.4136 |
| Liver | Selected PEFT | A | 42 | 0.5774 | 0.2316 | 0.8722 | 0.3595 |
| Liver | Selected PEFT | A | 43 | 0.5743 | 0.2262 | 0.8737 | 0.3391 |
| Liver | Selected PEFT | A | 44 | 0.5720 | 0.2619 | 0.8652 | 0.4212 |
| Liver | Selected PEFT | B | 42 | 0.6257 | 0.2120 | 0.8977 | 0.3121 |
| Liver | Selected PEFT | B | 43 | 0.6292 | 0.2099 | 0.9029 | 0.3212 |
| Liver | Selected PEFT | B | 44 | 0.6247 | 0.2249 | 0.8993 | 0.3312 |
| Liver | FullFT | A | 42 | 0.5856 | 0.1373 | 0.8654 | 0.2170 |
| Liver | FullFT | A | 43 | 0.5812 | 0.2019 | 0.8721 | 0.3083 |
| Liver | FullFT | A | 44 | 0.5816 | 0.1879 | 0.8743 | 0.2996 |
| Liver | FullFT | B | 42 | 0.6121 | 0.1891 | 0.8979 | 0.3494 |
| Liver | FullFT | B | 43 | 0.6021 | 0.1842 | 0.8960 | 0.3120 |
| Liver | FullFT | B | 44 | 0.6032 | 0.1824 | 0.8979 | 0.3098 |
| Tonsil | Selected PEFT | A | 42 | 0.4408 | 0.2188 | 0.8279 | 0.5073 |
| Tonsil | Selected PEFT | A | 43 | 0.4365 | 0.2220 | 0.8273 | 0.5171 |
| Tonsil | Selected PEFT | A | 44 | 0.4379 | 0.2203 | 0.8205 | 0.5079 |
| Tonsil | Selected PEFT | B | 42 | 0.5723 | 0.2443 | 0.8493 | 0.4540 |
| Tonsil | Selected PEFT | B | 43 | 0.5689 | 0.2540 | 0.8522 | 0.4747 |
| Tonsil | Selected PEFT | B | 44 | 0.5751 | 0.2618 | 0.8554 | 0.4900 |
| Tonsil | FullFT | A | 42 | 0.4358 | 0.2263 | 0.8147 | 0.5582 |
| Tonsil | FullFT | A | 43 | 0.4451 | 0.2357 | 0.8202 | 0.5740 |
| Tonsil | FullFT | A | 44 | 0.4443 | 0.2325 | 0.8219 | 0.5742 |
| Tonsil | FullFT | B | 42 | 0.5710 | 0.2661 | 0.8491 | 0.5286 |
| Tonsil | FullFT | B | 43 | 0.5725 | 0.2752 | 0.8399 | 0.5513 |
| Tonsil | FullFT | B | 44 | 0.5676 | 0.2663 | 0.8505 | 0.5486 |
| Pooled KLT | Selected PEFT | A | 42 | 0.5198 | 0.2170 | 0.8428 | 0.4836 |
| Pooled KLT | Selected PEFT | A | 43 | 0.5199 | 0.2145 | 0.8432 | 0.4830 |
| Pooled KLT | Selected PEFT | A | 44 | 0.5171 | 0.2270 | 0.8357 | 0.5147 |
| Pooled KLT | Selected PEFT | B | 42 | 0.6096 | 0.2305 | 0.8629 | 0.4779 |
| Pooled KLT | Selected PEFT | B | 43 | 0.6114 | 0.2406 | 0.8661 | 0.4960 |
| Pooled KLT | Selected PEFT | B | 44 | 0.6081 | 0.2384 | 0.8672 | 0.4947 |
| Pooled KLT | FullFT | A | 42 | 0.5226 | 0.1790 | 0.8316 | 0.4432 |
| Pooled KLT | FullFT | A | 43 | 0.5253 | 0.2105 | 0.8372 | 0.4894 |
| Pooled KLT | FullFT | A | 44 | 0.5277 | 0.2057 | 0.8391 | 0.4926 |
| Pooled KLT | FullFT | B | 42 | 0.6058 | 0.2275 | 0.8622 | 0.5381 |
| Pooled KLT | FullFT | B | 43 | 0.6033 | 0.2328 | 0.8549 | 0.5434 |
| Pooled KLT | FullFT | B | 44 | 0.6015 | 0.2300 | 0.8630 | 0.5426 |

### Within-fold seed variation and reciprocal-fold difference

| Test scope | Method | Fold | bPQ mean ± sample SD | mPQ mean ± sample SD | F1det mean ± sample SD | F1type mean ± sample SD |
|---|---|:---:|---:|---:|---:|---:|
| Kidney | Selected PEFT | A | 0.5832 ± 0.0085 | 0.1755 ± 0.0065 | 0.8671 ± 0.0045 | 0.3053 ± 0.0181 |
| Kidney | Selected PEFT | B | 0.6405 ± 0.0077 | 0.2421 ± 0.0212 | 0.9037 ± 0.0026 | 0.3586 ± 0.0247 |
| Kidney | Selected PEFT | **B−A** | +0.0573 | +0.0667 | +0.0366 | +0.0534 |
| Kidney | FullFT | A | 0.5918 ± 0.0087 | 0.1717 ± 0.0114 | 0.8619 ± 0.0042 | 0.3013 ± 0.0233 |
| Kidney | FullFT | B | 0.6488 ± 0.0014 | 0.2379 ± 0.0100 | 0.8983 ± 0.0012 | 0.3978 ± 0.0153 |
| Kidney | FullFT | **B−A** | +0.0570 | +0.0662 | +0.0364 | +0.0965 |
| Liver | Selected PEFT | A | 0.5746 ± 0.0027 | 0.2399 ± 0.0192 | 0.8704 ± 0.0045 | 0.3733 ± 0.0428 |
| Liver | Selected PEFT | B | 0.6265 ± 0.0024 | 0.2156 ± 0.0081 | 0.9000 ± 0.0027 | 0.3215 ± 0.0095 |
| Liver | Selected PEFT | **B−A** | +0.0519 | -0.0243 | +0.0296 | -0.0518 |
| Liver | FullFT | A | 0.5828 ± 0.0024 | 0.1757 ± 0.0340 | 0.8706 ± 0.0046 | 0.2750 ± 0.0504 |
| Liver | FullFT | B | 0.6058 ± 0.0055 | 0.1852 ± 0.0034 | 0.8973 ± 0.0011 | 0.3237 ± 0.0222 |
| Liver | FullFT | **B−A** | +0.0230 | +0.0095 | +0.0267 | +0.0488 |
| Tonsil | Selected PEFT | A | 0.4384 ± 0.0022 | 0.2204 ± 0.0016 | 0.8252 ± 0.0041 | 0.5108 ± 0.0055 |
| Tonsil | Selected PEFT | B | 0.5721 ± 0.0031 | 0.2534 ± 0.0087 | 0.8523 ± 0.0030 | 0.4729 ± 0.0181 |
| Tonsil | Selected PEFT | **B−A** | +0.1337 | +0.0330 | +0.0271 | -0.0379 |
| Tonsil | FullFT | A | 0.4418 ± 0.0051 | 0.2315 ± 0.0048 | 0.8190 ± 0.0038 | 0.5688 ± 0.0092 |
| Tonsil | FullFT | B | 0.5704 ± 0.0025 | 0.2692 ± 0.0052 | 0.8465 ± 0.0057 | 0.5428 ± 0.0124 |
| Tonsil | FullFT | **B−A** | +0.1286 | +0.0377 | +0.0276 | -0.0260 |
| Pooled KLT | Selected PEFT | A | 0.5189 ± 0.0016 | 0.2195 ± 0.0066 | 0.8406 ± 0.0042 | 0.4937 ± 0.0181 |
| Pooled KLT | Selected PEFT | B | 0.6097 ± 0.0016 | 0.2365 ± 0.0053 | 0.8654 ± 0.0022 | 0.4895 ± 0.0101 |
| Pooled KLT | Selected PEFT | **B−A** | +0.0908 | +0.0170 | +0.0249 | -0.0042 |
| Pooled KLT | FullFT | A | 0.5252 ± 0.0026 | 0.1984 ± 0.0170 | 0.8360 ± 0.0039 | 0.4751 ± 0.0277 |
| Pooled KLT | FullFT | B | 0.6035 ± 0.0022 | 0.2301 ± 0.0027 | 0.8601 ± 0.0044 | 0.5414 ± 0.0029 |
| Pooled KLT | FullFT | **B−A** | +0.0783 | +0.0317 | +0.0241 | +0.0663 |

### Per-class F1 and support

Support is shown as matched true / matched-plus-unmatched true. F1 uses the paired foreground confusion matrix; zero matched-support classes are omitted from F1type.

| Test scope | Method | Fold | Seed | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| Kidney | Selected PEFT | A | 42 | 0.5542 (52,308/58,432) | 0.4968 (21,500/26,658) | 0.1533 (13,101/17,674) | -- (0/0) | 0.0000 (1,158/2,105) |
| Kidney | Selected PEFT | A | 43 | 0.4765 (52,415/58,365) | 0.4780 (21,512/26,613) | 0.1990 (12,756/17,621) | -- (0/0) | 0.0052 (1,116/2,096) |
| Kidney | Selected PEFT | A | 44 | 0.5461 (52,006/58,414) | 0.5158 (21,600/26,650) | 0.2321 (12,756/17,642) | -- (0/0) | 0.0063 (1,169/2,094) |
| Kidney | Selected PEFT | B | 42 | 0.2764 (16,649/19,734) | 0.5023 (47,665/55,119) | 0.6360 (94,510/106,830) | -- (0/0) | 0.0003 (7,256/9,918) |
| Kidney | Selected PEFT | B | 43 | 0.3020 (16,624/19,727) | 0.5181 (47,786/55,093) | 0.7049 (94,926/106,757) | -- (0/0) | 0.0168 (7,273/9,888) |
| Kidney | Selected PEFT | B | 44 | 0.2635 (16,255/19,709) | 0.4867 (46,426/55,051) | 0.5838 (92,603/106,291) | -- (0/0) | 0.0131 (6,677/9,870) |
| Kidney | FullFT | A | 42 | 0.3935 (51,898/58,424) | 0.4625 (21,147/26,659) | 0.1575 (13,077/17,672) | -- (0/0) | 0.1625 (1,210/2,105) |
| Kidney | FullFT | A | 43 | 0.4311 (51,835/58,427) | 0.4692 (20,744/26,656) | 0.1224 (12,765/17,659) | -- (0/0) | 0.1078 (1,128/2,106) |
| Kidney | FullFT | A | 44 | 0.5114 (51,804/58,435) | 0.4895 (20,566/26,657) | 0.1743 (12,588/17,670) | -- (0/0) | 0.1345 (1,084/2,106) |
| Kidney | FullFT | B | 42 | 0.2619 (16,371/19,732) | 0.4806 (46,619/55,113) | 0.5619 (93,871/106,807) | -- (0/0) | 0.2272 (6,958/9,916) |
| Kidney | FullFT | B | 43 | 0.2663 (16,489/19,736) | 0.4621 (46,863/55,122) | 0.6131 (93,983/106,790) | -- (0/0) | 0.2463 (7,250/9,919) |
| Kidney | FullFT | B | 44 | 0.3044 (16,409/19,741) | 0.4548 (46,463/55,139) | 0.6602 (94,286/106,920) | -- (0/0) | 0.2347 (7,091/9,927) |
| Liver | Selected PEFT | A | 42 | 0.1973 (56,187/66,225) | 0.5753 (60,899/72,718) | 0.6655 (146,973/176,168) | -- (0/0) | 0.0000 (545/1,125) |
| Liver | Selected PEFT | A | 43 | 0.1235 (56,309/66,192) | 0.5692 (60,959/72,657) | 0.6635 (146,204/175,616) | -- (0/0) | 0.0000 (561/1,125) |
| Liver | Selected PEFT | A | 44 | 0.2639 (56,096/66,204) | 0.5775 (61,049/72,697) | 0.8347 (146,499/175,987) | -- (0/0) | 0.0087 (585/1,123) |
| Liver | Selected PEFT | B | 42 | 0.0777 (43,250/50,651) | 0.3983 (37,102/46,356) | 0.7725 (115,644/130,042) | -- (0/0) | 0.0000 (1,299/2,569) |
| Liver | Selected PEFT | B | 43 | 0.0688 (43,079/50,656) | 0.3726 (36,806/46,353) | 0.7795 (115,371/130,032) | -- (0/0) | 0.0638 (1,246/2,564) |
| Liver | Selected PEFT | B | 44 | 0.0535 (42,590/50,625) | 0.4452 (36,224/46,347) | 0.7892 (114,448/130,011) | -- (0/0) | 0.0369 (1,199/2,568) |
| Liver | FullFT | A | 42 | 0.0797 (56,205/66,216) | 0.5636 (60,149/72,713) | 0.1905 (145,678/176,015) | -- (0/0) | 0.0342 (575/1,125) |
| Liver | FullFT | A | 43 | 0.1439 (55,848/66,219) | 0.5564 (59,751/72,705) | 0.5104 (144,782/175,840) | -- (0/0) | 0.0224 (555/1,125) |
| Liver | FullFT | A | 44 | 0.1353 (55,691/66,222) | 0.5794 (58,995/72,688) | 0.4557 (143,759/176,027) | -- (0/0) | 0.0282 (523/1,124) |
| Liver | FullFT | B | 42 | 0.0584 (42,808/50,661) | 0.3135 (36,230/46,345) | 0.7646 (114,508/130,067) | -- (0/0) | 0.2610 (1,291/2,567) |
| Liver | FullFT | B | 43 | 0.0388 (42,544/50,671) | 0.2432 (35,849/46,351) | 0.7632 (113,469/130,073) | -- (0/0) | 0.2029 (1,287/2,568) |
| Liver | FullFT | B | 44 | 0.0262 (42,710/50,679) | 0.2307 (36,117/46,378) | 0.7611 (113,780/130,094) | -- (0/0) | 0.2214 (1,274/2,569) |
| Tonsil | Selected PEFT | A | 42 | 0.8428 (428,001/519,982) | 0.5088 (121,315/160,821) | 0.6774 (65,389/82,911) | -- (0/0) | 0.0001 (18,059/38,511) |
| Tonsil | Selected PEFT | A | 43 | 0.8436 (428,807/519,928) | 0.5165 (121,265/160,804) | 0.7073 (65,551/82,862) | -- (0/0) | 0.0011 (18,036/38,510) |
| Tonsil | Selected PEFT | A | 44 | 0.8419 (427,343/519,815) | 0.5008 (120,998/160,763) | 0.6871 (65,890/82,883) | -- (0/0) | 0.0018 (18,428/38,488) |
| Tonsil | Selected PEFT | B | 42 | 0.8662 (664,098/795,858) | 0.3265 (158,070/191,136) | 0.6233 (116,714/140,515) | -- (0/0) | 0.0000 (6,556/11,136) |
| Tonsil | Selected PEFT | B | 43 | 0.8643 (665,133/795,858) | 0.3799 (158,144/191,127) | 0.6455 (116,410/140,466) | -- (0/0) | 0.0090 (6,399/11,130) |
| Tonsil | Selected PEFT | B | 44 | 0.8620 (664,525/795,021) | 0.4139 (157,523/190,934) | 0.6709 (115,594/140,361) | -- (0/0) | 0.0133 (6,209/11,094) |
| Tonsil | FullFT | A | 42 | 0.8486 (423,518/519,891) | 0.5243 (116,661/160,792) | 0.7176 (64,307/82,875) | -- (0/0) | 0.1422 (15,777/38,527) |
| Tonsil | FullFT | A | 43 | 0.8261 (424,976/519,787) | 0.5343 (119,828/160,694) | 0.7237 (64,770/82,756) | -- (0/0) | 0.2121 (18,409/38,419) |
| Tonsil | FullFT | A | 44 | 0.8546 (425,768/519,959) | 0.5171 (118,747/160,799) | 0.7239 (64,846/82,894) | -- (0/0) | 0.2012 (17,409/38,465) |
| Tonsil | FullFT | B | 42 | 0.8712 (662,878/795,897) | 0.4274 (157,503/191,173) | 0.6948 (115,829/140,492) | -- (0/0) | 0.1209 (6,557/11,132) |
| Tonsil | FullFT | B | 43 | 0.8691 (658,338/795,861) | 0.4357 (156,835/191,160) | 0.7269 (115,248/140,467) | -- (0/0) | 0.1736 (6,726/11,136) |
| Tonsil | FullFT | B | 44 | 0.8744 (661,019/795,892) | 0.4109 (156,880/191,170) | 0.7433 (115,144/140,571) | -- (0/0) | 0.1659 (6,415/11,139) |
| Pooled KLT | Selected PEFT | A | 42 | 0.7544 (536,496/644,639) | 0.5327 (203,714/260,197) | 0.6471 (225,463/276,753) | -- (0/0) | 0.0001 (19,762/41,741) |
| Pooled KLT | Selected PEFT | A | 43 | 0.7448 (537,531/644,485) | 0.5304 (203,736/260,074) | 0.6555 (224,511/276,099) | -- (0/0) | 0.0013 (19,713/41,731) |
| Pooled KLT | Selected PEFT | A | 44 | 0.7757 (535,445/644,433) | 0.5320 (203,647/260,110) | 0.7485 (225,145/276,512) | -- (0/0) | 0.0024 (20,182/41,705) |
| Pooled KLT | Selected PEFT | B | 42 | 0.8235 (723,997/866,243) | 0.3918 (242,837/292,611) | 0.6963 (326,868/377,387) | -- (0/0) | 0.0001 (15,111/23,623) |
| Pooled KLT | Selected PEFT | B | 43 | 0.8236 (724,836/866,241) | 0.4189 (242,736/292,573) | 0.7222 (326,707/377,255) | -- (0/0) | 0.0191 (14,918/23,582) |
| Pooled KLT | Selected PEFT | B | 44 | 0.8189 (723,370/865,355) | 0.4383 (240,173/292,332) | 0.7060 (322,645/376,663) | -- (0/0) | 0.0155 (14,085/23,532) |
| Pooled KLT | FullFT | A | 42 | 0.7039 (531,621/644,531) | 0.5290 (197,957/260,164) | 0.4149 (223,062/276,562) | -- (0/0) | 0.1250 (17,562/41,757) |
| Pooled KLT | FullFT | A | 43 | 0.7176 (532,659/644,433) | 0.5328 (200,323/260,055) | 0.5749 (222,317/276,255) | -- (0/0) | 0.1324 (20,092/41,650) |
| Pooled KLT | FullFT | A | 44 | 0.7357 (533,263/644,616) | 0.5358 (198,308/260,144) | 0.5473 (221,193/276,591) | -- (0/0) | 0.1517 (19,016/41,695) |
| Pooled KLT | FullFT | B | 42 | 0.8313 (722,057/866,290) | 0.4318 (240,352/292,631) | 0.6984 (324,208/377,366) | -- (0/0) | 0.1909 (14,806/23,615) |
| Pooled KLT | FullFT | B | 43 | 0.8226 (717,371/866,268) | 0.4226 (239,547/292,633) | 0.7175 (322,700/377,330) | -- (0/0) | 0.2110 (15,263/23,623) |
| Pooled KLT | FullFT | B | 44 | 0.8339 (720,138/866,312) | 0.4047 (239,460/292,687) | 0.7318 (323,210/377,585) | -- (0/0) | 0.1999 (14,780/23,635) |

| Test scope | Method | Fold | Class | F1 mean ± sample SD | B−A mean F1 |
|---|---|:---:|---|---:|---:|
| Kidney | Selected PEFT | A | Immune | 0.5256 ± 0.0427 | -- |
| Kidney | Selected PEFT | A | Stromal | 0.4969 ± 0.0189 | -- |
| Kidney | Selected PEFT | A | Epithelial | 0.1948 ± 0.0396 | -- |
| Kidney | Selected PEFT | A | Melanocyte | -- ± -- | -- |
| Kidney | Selected PEFT | A | Other | 0.0038 ± 0.0034 | -- |
| Kidney | Selected PEFT | B | Immune | 0.2806 ± 0.0196 | -0.2450 |
| Kidney | Selected PEFT | B | Stromal | 0.5023 ± 0.0157 | 0.0055 |
| Kidney | Selected PEFT | B | Epithelial | 0.6416 ± 0.0607 | 0.4468 |
| Kidney | Selected PEFT | B | Melanocyte | -- ± -- | -- |
| Kidney | Selected PEFT | B | Other | 0.0100 ± 0.0087 | 0.0062 |
| Kidney | FullFT | A | Immune | 0.4453 ± 0.0602 | -- |
| Kidney | FullFT | A | Stromal | 0.4737 ± 0.0141 | -- |
| Kidney | FullFT | A | Epithelial | 0.1514 ± 0.0265 | -- |
| Kidney | FullFT | A | Melanocyte | -- ± -- | -- |
| Kidney | FullFT | A | Other | 0.1349 ± 0.0274 | -- |
| Kidney | FullFT | B | Immune | 0.2775 ± 0.0234 | -0.1678 |
| Kidney | FullFT | B | Stromal | 0.4659 ± 0.0133 | -0.0078 |
| Kidney | FullFT | B | Epithelial | 0.6117 ± 0.0492 | 0.4604 |
| Kidney | FullFT | B | Melanocyte | -- ± -- | -- |
| Kidney | FullFT | B | Other | 0.2361 ± 0.0096 | 0.1012 |
| Liver | Selected PEFT | A | Immune | 0.1949 ± 0.0702 | -- |
| Liver | Selected PEFT | A | Stromal | 0.5740 ± 0.0043 | -- |
| Liver | Selected PEFT | A | Epithelial | 0.7212 ± 0.0983 | -- |
| Liver | Selected PEFT | A | Melanocyte | -- ± -- | -- |
| Liver | Selected PEFT | A | Other | 0.0029 ± 0.0050 | -- |
| Liver | Selected PEFT | B | Immune | 0.0667 ± 0.0123 | -0.1282 |
| Liver | Selected PEFT | B | Stromal | 0.4054 ± 0.0368 | -0.1686 |
| Liver | Selected PEFT | B | Epithelial | 0.7804 ± 0.0084 | 0.0592 |
| Liver | Selected PEFT | B | Melanocyte | -- ± -- | -- |
| Liver | Selected PEFT | B | Other | 0.0336 ± 0.0321 | 0.0307 |
| Liver | FullFT | A | Immune | 0.1196 ± 0.0348 | -- |
| Liver | FullFT | A | Stromal | 0.5665 ± 0.0118 | -- |
| Liver | FullFT | A | Epithelial | 0.3856 ± 0.1711 | -- |
| Liver | FullFT | A | Melanocyte | -- ± -- | -- |
| Liver | FullFT | A | Other | 0.0283 ± 0.0059 | -- |
| Liver | FullFT | B | Immune | 0.0411 ± 0.0162 | -0.0785 |
| Liver | FullFT | B | Stromal | 0.2624 ± 0.0446 | -0.3040 |
| Liver | FullFT | B | Epithelial | 0.7630 ± 0.0018 | 0.3774 |
| Liver | FullFT | B | Melanocyte | -- ± -- | -- |
| Liver | FullFT | B | Other | 0.2284 ± 0.0297 | 0.2002 |
| Tonsil | Selected PEFT | A | Immune | 0.8428 ± 0.0008 | -- |
| Tonsil | Selected PEFT | A | Stromal | 0.5087 ± 0.0079 | -- |
| Tonsil | Selected PEFT | A | Epithelial | 0.6906 ± 0.0153 | -- |
| Tonsil | Selected PEFT | A | Melanocyte | -- ± -- | -- |
| Tonsil | Selected PEFT | A | Other | 0.0010 ± 0.0009 | -- |
| Tonsil | Selected PEFT | B | Immune | 0.8642 ± 0.0021 | 0.0214 |
| Tonsil | Selected PEFT | B | Stromal | 0.3734 ± 0.0441 | -0.1353 |
| Tonsil | Selected PEFT | B | Epithelial | 0.6466 ± 0.0238 | -0.0440 |
| Tonsil | Selected PEFT | B | Melanocyte | -- ± -- | -- |
| Tonsil | Selected PEFT | B | Other | 0.0074 ± 0.0068 | 0.0064 |
| Tonsil | FullFT | A | Immune | 0.8431 ± 0.0151 | -- |
| Tonsil | FullFT | A | Stromal | 0.5252 ± 0.0086 | -- |
| Tonsil | FullFT | A | Epithelial | 0.7217 ± 0.0036 | -- |
| Tonsil | FullFT | A | Melanocyte | -- ± -- | -- |
| Tonsil | FullFT | A | Other | 0.1852 ± 0.0377 | -- |
| Tonsil | FullFT | B | Immune | 0.8716 ± 0.0027 | 0.0285 |
| Tonsil | FullFT | B | Stromal | 0.4246 ± 0.0126 | -0.1006 |
| Tonsil | FullFT | B | Epithelial | 0.7217 ± 0.0246 | -0.0000 |
| Tonsil | FullFT | B | Melanocyte | -- ± -- | -- |
| Tonsil | FullFT | B | Other | 0.1535 ± 0.0285 | -0.0317 |
| Pooled KLT | Selected PEFT | A | Immune | 0.7583 ± 0.0158 | -- |
| Pooled KLT | Selected PEFT | A | Stromal | 0.5317 ± 0.0012 | -- |
| Pooled KLT | Selected PEFT | A | Epithelial | 0.6837 ± 0.0563 | -- |
| Pooled KLT | Selected PEFT | A | Melanocyte | -- ± -- | -- |
| Pooled KLT | Selected PEFT | A | Other | 0.0013 ± 0.0012 | -- |
| Pooled KLT | Selected PEFT | B | Immune | 0.8220 ± 0.0027 | 0.0637 |
| Pooled KLT | Selected PEFT | B | Stromal | 0.4163 ± 0.0234 | -0.1154 |
| Pooled KLT | Selected PEFT | B | Epithelial | 0.7081 ± 0.0131 | 0.0245 |
| Pooled KLT | Selected PEFT | B | Melanocyte | -- ± -- | -- |
| Pooled KLT | Selected PEFT | B | Other | 0.0116 ± 0.0101 | 0.0103 |
| Pooled KLT | FullFT | A | Immune | 0.7191 ± 0.0160 | -- |
| Pooled KLT | FullFT | A | Stromal | 0.5326 ± 0.0034 | -- |
| Pooled KLT | FullFT | A | Epithelial | 0.5123 ± 0.0855 | -- |
| Pooled KLT | FullFT | A | Melanocyte | -- ± -- | -- |
| Pooled KLT | FullFT | A | Other | 0.1364 ± 0.0138 | -- |
| Pooled KLT | FullFT | B | Immune | 0.8293 ± 0.0059 | 0.1102 |
| Pooled KLT | FullFT | B | Stromal | 0.4197 ± 0.0138 | -0.1129 |
| Pooled KLT | FullFT | B | Epithelial | 0.7159 ± 0.0168 | 0.2035 |
| Pooled KLT | FullFT | B | Melanocyte | -- ± -- | -- |
| Pooled KLT | FullFT | B | Other | 0.2006 ± 0.0101 | 0.0642 |

### Predicted foreground-class frequencies

Counts include paired and unmatched foreground predictions; background/untyped predictions are excluded from the denominator.

| Test scope | Method | Fold | Seed | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| Kidney | Selected PEFT | A | 42 | 29,062 (29.59%) | 62,599 (63.74%) | 6,531 (6.65%) | 0 (0.00%) | 11 (0.01%) |
| Kidney | Selected PEFT | A | 43 | 21,540 (22.31%) | 64,838 (67.15%) | 10,113 (10.47%) | 0 (0.00%) | 68 (0.07%) |
| Kidney | Selected PEFT | A | 44 | 29,046 (29.66%) | 52,539 (53.65%) | 16,143 (16.48%) | 0 (0.00%) | 202 (0.21%) |
| Kidney | Selected PEFT | B | 42 | 37,606 (21.38%) | 82,825 (47.09%) | 55,458 (31.53%) | 0 (0.00%) | 2 (0.00%) |
| Kidney | Selected PEFT | B | 43 | 32,446 (18.44%) | 79,488 (45.18%) | 63,665 (36.18%) | 0 (0.00%) | 345 (0.20%) |
| Kidney | Selected PEFT | B | 44 | 33,309 (19.79%) | 87,911 (52.22%) | 46,829 (27.82%) | 0 (0.00%) | 299 (0.18%) |
| Kidney | FullFT | A | 42 | 17,110 (17.33%) | 71,165 (72.08%) | 6,531 (6.62%) | 0 (0.00%) | 3,921 (3.97%) |
| Kidney | FullFT | A | 43 | 18,515 (19.39%) | 59,020 (61.80%) | 5,387 (5.64%) | 0 (0.00%) | 12,572 (13.17%) |
| Kidney | FullFT | A | 44 | 24,138 (25.72%) | 55,644 (59.28%) | 7,913 (8.43%) | 0 (0.00%) | 6,170 (6.57%) |
| Kidney | FullFT | B | 42 | 23,629 (13.69%) | 98,019 (56.81%) | 44,855 (26.00%) | 0 (0.00%) | 6,040 (3.50%) |
| Kidney | FullFT | B | 43 | 40,244 (23.00%) | 69,714 (39.85%) | 54,230 (31.00%) | 0 (0.00%) | 10,749 (6.14%) |
| Kidney | FullFT | B | 44 | 26,669 (15.30%) | 78,262 (44.91%) | 62,323 (35.76%) | 0 (0.00%) | 7,005 (4.02%) |
| Liver | Selected PEFT | A | 42 | 58,472 (20.13%) | 139,453 (48.02%) | 92,475 (31.84%) | 0 (0.00%) | 9 (0.00%) |
| Liver | Selected PEFT | A | 43 | 55,255 (19.17%) | 139,014 (48.23%) | 93,886 (32.57%) | 0 (0.00%) | 90 (0.03%) |
| Liver | Selected PEFT | A | 44 | 23,941 (8.14%) | 119,318 (40.58%) | 150,108 (51.05%) | 0 (0.00%) | 666 (0.23%) |
| Liver | Selected PEFT | B | 42 | 4,360 (2.08%) | 24,640 (11.75%) | 180,698 (86.17%) | 0 (0.00%) | 8 (0.00%) |
| Liver | Selected PEFT | B | 43 | 3,066 (1.49%) | 21,383 (10.41%) | 180,168 (87.68%) | 0 (0.00%) | 863 (0.42%) |
| Liver | Selected PEFT | B | 44 | 1,986 (0.98%) | 26,879 (13.26%) | 173,281 (85.51%) | 0 (0.00%) | 488 (0.24%) |
| Liver | FullFT | A | 42 | 118,314 (40.72%) | 146,975 (50.58%) | 18,744 (6.45%) | 0 (0.00%) | 6,552 (2.25%) |
| Liver | FullFT | A | 43 | 67,705 (23.99%) | 132,336 (46.88%) | 59,799 (21.19%) | 0 (0.00%) | 22,427 (7.95%) |
| Liver | FullFT | A | 44 | 93,339 (33.79%) | 119,442 (43.24%) | 53,877 (19.50%) | 0 (0.00%) | 9,592 (3.47%) |
| Liver | FullFT | B | 42 | 2,811 (1.38%) | 13,629 (6.67%) | 185,461 (90.80%) | 0 (0.00%) | 2,344 (1.15%) |
| Liver | FullFT | B | 43 | 2,206 (1.10%) | 8,509 (4.23%) | 185,455 (92.10%) | 0 (0.00%) | 5,198 (2.58%) |
| Liver | FullFT | B | 44 | 1,167 (0.58%) | 7,775 (3.85%) | 188,555 (93.32%) | 0 (0.00%) | 4,550 (2.25%) |
| Tonsil | Selected PEFT | A | 42 | 498,361 (68.64%) | 114,880 (15.82%) | 112,849 (15.54%) | 0 (0.00%) | 5 (0.00%) |
| Tonsil | Selected PEFT | A | 43 | 499,152 (68.44%) | 125,722 (17.24%) | 104,378 (14.31%) | 0 (0.00%) | 44 (0.01%) |
| Tonsil | Selected PEFT | A | 44 | 514,075 (69.52%) | 107,894 (14.59%) | 117,352 (15.87%) | 0 (0.00%) | 114 (0.02%) |
| Tonsil | Selected PEFT | B | 42 | 946,633 (87.05%) | 70,705 (6.50%) | 70,081 (6.44%) | 0 (0.00%) | 11 (0.00%) |
| Tonsil | Selected PEFT | B | 43 | 907,902 (83.95%) | 99,768 (9.22%) | 73,407 (6.79%) | 0 (0.00%) | 431 (0.04%) |
| Tonsil | Selected PEFT | B | 44 | 848,607 (79.38%) | 137,784 (12.89%) | 81,491 (7.62%) | 0 (0.00%) | 1,109 (0.10%) |
| Tonsil | FullFT | A | 42 | 487,284 (67.68%) | 142,512 (19.79%) | 84,134 (11.69%) | 0 (0.00%) | 6,021 (0.84%) |
| Tonsil | FullFT | A | 43 | 444,229 (60.91%) | 181,922 (24.94%) | 87,675 (12.02%) | 0 (0.00%) | 15,478 (2.12%) |
| Tonsil | FullFT | A | 44 | 505,055 (69.88%) | 119,967 (16.60%) | 89,520 (12.39%) | 0 (0.00%) | 8,192 (1.13%) |
| Tonsil | FullFT | B | 42 | 884,714 (81.78%) | 110,165 (10.18%) | 83,358 (7.71%) | 0 (0.00%) | 3,614 (0.33%) |
| Tonsil | FullFT | B | 43 | 864,297 (79.10%) | 124,081 (11.36%) | 94,112 (8.61%) | 0 (0.00%) | 10,175 (0.93%) |
| Tonsil | FullFT | B | 44 | 858,850 (80.24%) | 97,138 (9.08%) | 101,901 (9.52%) | 0 (0.00%) | 12,442 (1.16%) |
| Pooled KLT | Selected PEFT | A | 42 | 585,895 (52.56%) | 316,932 (28.43%) | 211,855 (19.01%) | 0 (0.00%) | 25 (0.00%) |
| Pooled KLT | Selected PEFT | A | 43 | 575,947 (51.70%) | 329,574 (29.58%) | 208,377 (18.70%) | 0 (0.00%) | 202 (0.02%) |
| Pooled KLT | Selected PEFT | A | 44 | 567,062 (50.12%) | 279,751 (24.73%) | 283,603 (25.07%) | 0 (0.00%) | 982 (0.09%) |
| Pooled KLT | Selected PEFT | B | 42 | 988,599 (67.11%) | 178,170 (12.10%) | 306,237 (20.79%) | 0 (0.00%) | 21 (0.00%) |
| Pooled KLT | Selected PEFT | B | 43 | 943,414 (64.49%) | 200,639 (13.71%) | 317,240 (21.69%) | 0 (0.00%) | 1,639 (0.11%) |
| Pooled KLT | Selected PEFT | B | 44 | 883,902 (61.38%) | 252,574 (17.54%) | 301,601 (20.94%) | 0 (0.00%) | 1,896 (0.13%) |
| Pooled KLT | FullFT | A | 42 | 622,708 (56.14%) | 360,652 (32.51%) | 109,409 (9.86%) | 0 (0.00%) | 16,494 (1.49%) |
| Pooled KLT | FullFT | A | 43 | 530,449 (47.91%) | 373,278 (33.72%) | 152,861 (13.81%) | 0 (0.00%) | 50,477 (4.56%) |
| Pooled KLT | FullFT | A | 44 | 622,532 (56.96%) | 295,053 (27.00%) | 151,310 (13.85%) | 0 (0.00%) | 23,954 (2.19%) |
| Pooled KLT | FullFT | B | 42 | 911,154 (62.47%) | 221,813 (15.21%) | 313,674 (21.50%) | 0 (0.00%) | 11,998 (0.82%) |
| Pooled KLT | FullFT | B | 43 | 906,747 (61.73%) | 202,304 (13.77%) | 333,797 (22.72%) | 0 (0.00%) | 26,122 (1.78%) |
| Pooled KLT | FullFT | B | 44 | 886,686 (61.29%) | 183,175 (12.66%) | 352,779 (24.39%) | 0 (0.00%) | 23,997 (1.66%) |

### Paired confusion matrices

Rows are true labels, columns predicted labels. Background is retained exactly as stored; F1type excludes its row and column.

#### Kidney — Selected PEFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 5 | 22,017 | 27,204 | 3,086 | 0 | 1 |
| Stromal | 3 | 1,337 | 19,075 | 1,085 | 0 | 3 |
| Epithelial | 7 | 3,417 | 8,248 | 1,436 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 372 | 758 | 28 | 0 | 0 |

#### Kidney — Selected PEFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 72 | 17,370 | 29,865 | 5,164 | 0 | 16 |
| Stromal | 48 | 759 | 19,051 | 1,688 | 0 | 14 |
| Epithelial | 60 | 2,146 | 8,426 | 2,171 | 0 | 13 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 10 | 224 | 851 | 38 | 0 | 3 |

#### Kidney — Selected PEFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 23 | 21,570 | 21,828 | 8,589 | 0 | 19 |
| Stromal | 11 | 1,649 | 17,470 | 2,465 | 0 | 16 |
| Epithelial | 39 | 3,376 | 6,178 | 3,139 | 0 | 63 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 12 | 400 | 662 | 103 | 0 | 4 |

#### Kidney — Selected PEFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 7 | 7,195 | 7,990 | 1,464 | 0 | 0 |
| Stromal | 20 | 12,434 | 31,391 | 3,839 | 0 | 1 |
| Epithelial | 94 | 14,062 | 33,438 | 47,010 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 9 | 1,727 | 4,512 | 1,016 | 0 | 1 |

#### Kidney — Selected PEFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 14 | 7,190 | 7,974 | 1,435 | 0 | 25 |
| Stromal | 46 | 12,258 | 31,522 | 3,965 | 0 | 41 |
| Epithelial | 167 | 9,597 | 30,098 | 55,126 | 0 | 105 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 39 | 1,946 | 4,308 | 956 | 0 | 63 |

#### Kidney — Selected PEFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 32 | 6,406 | 8,568 | 1,247 | 0 | 34 |
| Stromal | 88 | 11,595 | 31,693 | 3,077 | 0 | 61 |
| Epithelial | 633 | 13,202 | 38,996 | 40,334 | 0 | 71 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 57 | 1,163 | 4,556 | 913 | 0 | 45 |

#### Kidney — FullFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 13 | 13,336 | 33,881 | 3,503 | 0 | 1,178 |
| Stromal | 2 | 754 | 19,414 | 674 | 0 | 305 |
| Epithelial | 9 | 1,638 | 8,822 | 1,477 | 0 | 1,140 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 154 | 694 | 23 | 0 | 339 |

#### Kidney — FullFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 10 | 14,970 | 29,145 | 3,118 | 0 | 4,602 |
| Stromal | 5 | 691 | 17,367 | 708 | 0 | 1,978 |
| Epithelial | 22 | 1,810 | 6,436 | 1,082 | 0 | 3,437 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 142 | 338 | 13 | 0 | 635 |

#### Kidney — FullFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 2 | 19,133 | 26,247 | 4,234 | 0 | 2,190 |
| Stromal | 4 | 1,255 | 17,429 | 1,171 | 0 | 711 |
| Epithelial | 11 | 2,451 | 6,520 | 1,719 | 0 | 1,898 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 190 | 453 | 17 | 0 | 424 |

#### Kidney — FullFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 9 | 5,106 | 8,567 | 1,322 | 0 | 1,376 |
| Stromal | 26 | 8,539 | 33,410 | 2,917 | 0 | 1,753 |
| Epithelial | 117 | 8,164 | 46,350 | 38,599 | 0 | 758 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 11 | 814 | 4,083 | 671 | 0 | 1,390 |

#### Kidney — FullFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 5 | 7,299 | 5,716 | 1,703 | 0 | 1,771 |
| Stromal | 17 | 13,341 | 25,871 | 4,637 | 0 | 3,014 |
| Epithelial | 134 | 16,188 | 30,812 | 44,809 | 0 | 2,174 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 8 | 1,506 | 2,698 | 1,050 | 0 | 1,996 |

#### Kidney — FullFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 0 | 6,351 | 6,529 | 2,173 | 0 | 1,356 |
| Stromal | 0 | 11,587 | 27,184 | 5,557 | 0 | 2,135 |
| Epithelial | 4 | 6,210 | 36,224 | 50,891 | 0 | 961 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 1,168 | 3,133 | 1,255 | 0 | 1,535 |

#### Liver — Selected PEFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 4 | 10,686 | 42,358 | 3,143 | 0 | 0 |
| Stromal | 10 | 2,252 | 54,316 | 4,330 | 0 | 1 |
| Epithelial | 10 | 39,102 | 30,834 | 77,033 | 0 | 4 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 76 | 433 | 36 | 0 | 0 |

#### Liver — Selected PEFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 37 | 6,538 | 45,974 | 3,789 | 0 | 8 |
| Stromal | 71 | 1,785 | 53,787 | 5,371 | 0 | 16 |
| Epithelial | 562 | 41,222 | 27,789 | 77,152 | 0 | 41 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 51 | 471 | 39 | 0 | 0 |

#### Liver — Selected PEFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 1 | 0 | 0 | 0 | 0 | 0 |
| Immune | 25 | 10,133 | 38,858 | 7,019 | 0 | 86 |
| Stromal | 31 | 1,895 | 49,129 | 9,889 | 0 | 136 |
| Epithelial | 191 | 8,599 | 20,682 | 117,113 | 0 | 105 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 2 | 81 | 422 | 78 | 0 | 4 |

#### Liver — Selected PEFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 30 | 1,833 | 5,657 | 35,760 | 0 | 0 |
| Stromal | 25 | 1,050 | 11,671 | 24,381 | 0 | 0 |
| Epithelial | 59 | 778 | 3,802 | 111,061 | 0 | 3 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 255 | 368 | 676 | 0 | 0 |

#### Liver — Selected PEFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 25 | 1,577 | 6,142 | 35,077 | 0 | 283 |
| Stromal | 28 | 684 | 10,386 | 25,560 | 0 | 176 |
| Epithelial | 69 | 360 | 2,037 | 112,829 | 0 | 145 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 5 | 166 | 372 | 647 | 0 | 61 |

#### Liver — Selected PEFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 56 | 1,190 | 7,831 | 33,453 | 0 | 116 |
| Stromal | 34 | 344 | 13,521 | 22,245 | 0 | 114 |
| Epithelial | 90 | 318 | 2,730 | 111,340 | 0 | 60 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 63 | 442 | 666 | 0 | 28 |

#### Liver — FullFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 13 | 6,462 | 47,462 | 697 | 0 | 1,584 |
| Stromal | 15 | 3,215 | 54,785 | 1,143 | 0 | 1,006 |
| Epithelial | 163 | 96,134 | 31,604 | 15,535 | 0 | 2,405 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 63 | 408 | 7 | 0 | 97 |

#### Liver — FullFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 10 | 8,518 | 39,941 | 1,889 | 0 | 5,500 |
| Stromal | 23 | 1,679 | 50,922 | 2,497 | 0 | 4,653 |
| Epithelial | 338 | 52,306 | 32,160 | 51,119 | 0 | 9,197 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 50 | 261 | 19 | 0 | 225 |

#### Liver — FullFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 7 | 9,680 | 40,894 | 2,516 | 0 | 2,601 |
| Stromal | 40 | 3,367 | 49,601 | 3,983 | 0 | 2,044 |
| Epithelial | 151 | 74,330 | 21,423 | 44,352 | 0 | 3,654 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 66 | 302 | 29 | 0 | 126 |

#### Liver — FullFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 1 | 0 | 0 | 0 | 0 | 0 |
| Immune | 20 | 1,325 | 2,155 | 38,431 | 0 | 897 |
| Stromal | 36 | 746 | 7,528 | 27,515 | 0 | 441 |
| Epithelial | 34 | 403 | 1,937 | 112,052 | 0 | 116 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 2 | 115 | 178 | 586 | 0 | 412 |

#### Liver — FullFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 1 | 0 | 0 | 0 | 0 | 0 |
| Immune | 10 | 864 | 1,146 | 39,130 | 0 | 1,404 |
| Stromal | 30 | 785 | 5,279 | 27,865 | 0 | 1,920 |
| Epithelial | 28 | 343 | 1,099 | 111,748 | 0 | 279 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 53 | 43 | 639 | 0 | 552 |

#### Liver — FullFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 2 | 0 | 0 | 0 | 0 | 0 |
| Immune | 2 | 573 | 1,029 | 39,444 | 0 | 1,664 |
| Stromal | 3 | 356 | 4,947 | 29,499 | 0 | 1,315 |
| Epithelial | 7 | 140 | 743 | 112,644 | 0 | 253 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 31 | 54 | 628 | 0 | 561 |

#### Tonsil — Selected PEFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 48 | 363,446 | 30,669 | 33,886 | 0 | 0 |
| Stromal | 78 | 56,473 | 56,116 | 8,724 | 0 | 2 |
| Epithelial | 46 | 6,232 | 3,480 | 55,677 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 34 | 8,354 | 8,995 | 709 | 0 | 1 |

#### Tonsil — Selected PEFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 102 | 364,396 | 37,757 | 26,647 | 0 | 7 |
| Stromal | 95 | 54,501 | 59,091 | 7,653 | 0 | 20 |
| Epithelial | 95 | 7,424 | 2,782 | 55,343 | 0 | 2 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 35 | 8,798 | 7,930 | 1,298 | 0 | 10 |

#### Tonsil — Selected PEFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 215 | 365,175 | 29,783 | 32,372 | 0 | 13 |
| Stromal | 136 | 57,714 | 53,261 | 9,993 | 0 | 30 |
| Epithelial | 74 | 6,730 | 1,902 | 57,251 | 0 | 7 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 57 | 10,489 | 6,771 | 1,151 | 0 | 17 |

#### Tonsil — Selected PEFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 61 | 644,859 | 14,152 | 5,085 | 0 | 2 |
| Stromal | 43 | 122,056 | 35,500 | 513 | 0 | 1 |
| Epithelial | 75 | 52,458 | 8,796 | 55,459 | 0 | 1 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 4 | 5,413 | 958 | 185 | 0 | 0 |

#### Tonsil — Selected PEFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 61 | 631,473 | 27,992 | 5,610 | 0 | 58 |
| Stromal | 52 | 111,261 | 46,108 | 734 | 0 | 41 |
| Epithelial | 124 | 48,324 | 9,324 | 58,601 | 0 | 161 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 10 | 4,981 | 1,187 | 201 | 0 | 30 |

#### Tonsil — Selected PEFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 898 | 610,872 | 45,006 | 8,347 | 0 | 300 |
| Stromal | 245 | 99,599 | 56,952 | 776 | 0 | 196 |
| Epithelial | 229 | 38,236 | 14,045 | 63,132 | 0 | 181 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 46 | 4,151 | 1,650 | 362 | 0 | 46 |

#### Tonsil — FullFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 139 | 358,241 | 45,019 | 18,794 | 0 | 1,464 |
| Stromal | 107 | 48,415 | 62,568 | 4,051 | 0 | 1,627 |
| Epithelial | 82 | 8,624 | 6,137 | 49,071 | 0 | 475 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 18 | 5,476 | 8,281 | 540 | 0 | 1,480 |

#### Tonsil — FullFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 243 | 334,505 | 66,584 | 19,635 | 0 | 4,252 |
| Stromal | 205 | 37,396 | 73,197 | 4,798 | 0 | 4,437 |
| Epithelial | 201 | 7,729 | 5,152 | 50,890 | 0 | 999 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 126 | 5,269 | 9,251 | 555 | 0 | 3,334 |

#### Tonsil — FullFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 71 | 369,205 | 36,511 | 18,492 | 0 | 1,560 |
| Stromal | 100 | 52,689 | 57,537 | 6,371 | 0 | 2,150 |
| Epithelial | 63 | 9,778 | 2,585 | 51,570 | 0 | 913 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 80 | 6,590 | 7,163 | 1,191 | 0 | 2,465 |

#### Tonsil — FullFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 22 | 625,767 | 28,854 | 6,943 | 0 | 1,314 |
| Stromal | 6 | 103,006 | 53,483 | 499 | 0 | 515 |
| Epithelial | 98 | 40,321 | 9,256 | 65,698 | 0 | 554 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 8 | 4,672 | 1,176 | 134 | 0 | 575 |

#### Tonsil — FullFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 58 | 610,792 | 34,961 | 9,498 | 0 | 3,087 |
| Stromal | 19 | 98,416 | 56,054 | 639 | 0 | 1,726 |
| Epithelial | 123 | 33,654 | 8,511 | 71,725 | 0 | 1,358 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 4 | 4,303 | 974 | 223 | 0 | 1,226 |

#### Tonsil — FullFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 27 | 620,296 | 24,003 | 12,353 | 0 | 4,367 |
| Stromal | 9 | 104,581 | 48,978 | 1,181 | 0 | 2,140 |
| Epithelial | 19 | 28,890 | 7,700 | 76,224 | 0 | 2,330 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 3,983 | 853 | 199 | 0 | 1,380 |

#### Pooled KLT — Selected PEFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 10 | 0 | 0 | 0 | 0 | 0 |
| Immune | 57 | 396,149 | 100,231 | 40,115 | 0 | 1 |
| Stromal | 91 | 60,062 | 129,507 | 14,139 | 0 | 6 |
| Epithelial | 63 | 48,751 | 42,562 | 134,146 | 0 | 4 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 35 | 8,802 | 10,186 | 773 | 0 | 1 |

#### Pooled KLT — Selected PEFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 10 | 0 | 0 | 0 | 0 | 0 |
| Immune | 211 | 388,304 | 113,596 | 35,600 | 0 | 31 |
| Stromal | 214 | 57,045 | 131,929 | 14,712 | 0 | 50 |
| Epithelial | 717 | 50,792 | 38,997 | 134,666 | 0 | 56 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 45 | 9,073 | 9,252 | 1,375 | 0 | 13 |

#### Pooled KLT — Selected PEFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 9 | 0 | 0 | 0 | 0 | 0 |
| Immune | 263 | 396,878 | 90,469 | 47,980 | 0 | 118 |
| Stromal | 178 | 61,258 | 119,860 | 22,347 | 0 | 182 |
| Epithelial | 304 | 18,705 | 28,762 | 177,503 | 0 | 175 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 71 | 10,970 | 7,855 | 1,332 | 0 | 25 |

#### Pooled KLT — Selected PEFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 17 | 0 | 0 | 0 | 0 | 0 |
| Immune | 98 | 653,887 | 27,799 | 42,309 | 0 | 2 |
| Stromal | 88 | 135,540 | 78,562 | 28,733 | 0 | 2 |
| Epithelial | 228 | 67,298 | 46,036 | 213,530 | 0 | 4 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 13 | 7,395 | 5,838 | 1,877 | 0 | 1 |

#### Pooled KLT — Selected PEFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 18 | 0 | 0 | 0 | 0 | 0 |
| Immune | 100 | 640,240 | 42,108 | 42,122 | 0 | 366 |
| Stromal | 126 | 124,203 | 88,016 | 30,259 | 0 | 258 |
| Epithelial | 360 | 58,281 | 41,459 | 226,556 | 0 | 411 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 54 | 7,093 | 5,867 | 1,804 | 0 | 154 |

#### Pooled KLT — Selected PEFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 17 | 0 | 0 | 0 | 0 | 0 |
| Immune | 986 | 618,468 | 61,405 | 43,047 | 0 | 450 |
| Stromal | 367 | 111,538 | 102,166 | 26,098 | 0 | 371 |
| Epithelial | 952 | 51,756 | 55,771 | 214,806 | 0 | 312 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 104 | 5,377 | 6,648 | 1,941 | 0 | 119 |

#### Pooled KLT — FullFT — Fold A — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 10 | 0 | 0 | 0 | 0 | 0 |
| Immune | 165 | 378,039 | 126,362 | 22,994 | 0 | 4,226 |
| Stromal | 124 | 52,384 | 136,767 | 5,868 | 0 | 2,938 |
| Epithelial | 254 | 106,396 | 46,563 | 66,083 | 0 | 4,020 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 19 | 5,693 | 9,383 | 570 | 0 | 1,916 |

#### Pooled KLT — FullFT — Fold A — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 10 | 0 | 0 | 0 | 0 | 0 |
| Immune | 263 | 357,993 | 135,670 | 24,642 | 0 | 14,354 |
| Stromal | 233 | 39,766 | 141,486 | 8,003 | 0 | 11,068 |
| Epithelial | 561 | 61,845 | 43,748 | 103,091 | 0 | 13,633 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 126 | 5,461 | 9,850 | 587 | 0 | 4,194 |

#### Pooled KLT — FullFT — Fold A — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 80 | 398,018 | 103,652 | 25,242 | 0 | 6,351 |
| Stromal | 144 | 57,311 | 124,567 | 11,525 | 0 | 4,905 |
| Epithelial | 225 | 86,559 | 30,528 | 97,641 | 0 | 6,465 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 81 | 6,846 | 7,918 | 1,237 | 0 | 3,015 |

#### Pooled KLT — FullFT — Fold B — seed 42

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 13 | 0 | 0 | 0 | 0 | 0 |
| Immune | 51 | 632,198 | 39,576 | 46,696 | 0 | 3,587 |
| Stromal | 68 | 112,291 | 94,421 | 30,931 | 0 | 2,709 |
| Epithelial | 249 | 48,888 | 57,543 | 216,349 | 0 | 1,428 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 21 | 5,601 | 5,437 | 1,391 | 0 | 2,377 |

#### Pooled KLT — FullFT — Fold B — seed 43

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 15 | 0 | 0 | 0 | 0 | 0 |
| Immune | 73 | 618,955 | 41,823 | 50,331 | 0 | 6,262 |
| Stromal | 66 | 112,542 | 87,204 | 33,141 | 0 | 6,660 |
| Epithelial | 285 | 50,185 | 40,422 | 228,282 | 0 | 3,811 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 13 | 5,862 | 3,715 | 1,912 | 0 | 3,774 |

#### Pooled KLT — FullFT — Fold B — seed 44

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 17 | 0 | 0 | 0 | 0 | 0 |
| Immune | 29 | 627,220 | 31,561 | 53,970 | 0 | 7,387 |
| Stromal | 12 | 116,524 | 81,109 | 36,237 | 0 | 5,590 |
| Epithelial | 30 | 35,240 | 44,667 | 239,759 | 0 | 3,544 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 5,182 | 4,040 | 2,082 | 0 | 3,476 |

### Descriptive Fold A/B correspondence

| Test scope | Train→test TVD A / B | TEST nuclei/patch A / B | Method | B−A mPQ | B−A F1type | Dominant paired off-diagonal A / B |
|---|---:|---:|---|---:|---:|---|
| Kidney | 0.457 / 0.468 | 23.92 / 28.93 | Selected PEFT | +0.0667 | +0.0534 | Immune→Stromal 78,897 (50.3%) / Epithelial→Stromal 102,532 (36.4%) |
| Kidney | 0.457 / 0.468 | 23.92 / 28.93 | FullFT | +0.0662 | +0.0965 | Immune→Stromal 89,273 (57.4%) / Epithelial→Stromal 113,386 (40.2%) |
| Liver | 0.029 / 0.032 | 35.01 / 23.88 | Selected PEFT | -0.0243 | -0.0518 | Immune→Stromal 127,190 (75.4%) / Immune→Epithelial 104,290 (80.9%) |
| Liver | 0.029 / 0.032 | 35.01 / 23.88 | FullFT | +0.0095 | +0.0488 | Epithelial→Immune 222,770 (51.3%) / Immune→Epithelial 117,005 (91.4%) |
| Tonsil | 0.071 / 0.062 | 82.43 / 117.26 | Selected PEFT | +0.0330 | -0.0379 | Stromal→Immune 168,688 (46.4%) / Stromal→Immune 332,916 (70.3%) |
| Tonsil | 0.071 / 0.062 | 82.43 / 117.26 | FullFT | +0.0377 | -0.0260 | Immune→Stromal 148,114 (11.6%) / Stromal→Immune 306,003 (64.9%) |
| Pooled KLT | 0.044 / 0.040 | 52.85 / 60.12 | Selected PEFT | +0.0170 | -0.0042 | Immune→Stromal 304,296 (18.9%) / Stromal→Immune 371,281 (51.2%) |
| Pooled KLT | 0.044 / 0.040 | 52.85 / 60.12 | FullFT | +0.0317 | +0.0663 | Immune→Stromal 365,684 (22.9%) / Stromal→Immune 341,357 (47.5%) |

- Fold A has pooled train→test TVD 0.044 and JSD 0.004 bits; its held-out TEST density is 52.85 nuclei/patch and entropy is 1.613 bits.
- Fold B has pooled train→test TVD 0.040 and JSD 0.003 bits; its held-out TEST density is 60.12 nuclei/patch and entropy is 1.511 bits.
- Melanocyte is absent from both KLT training and TEST in both directions, so it does not explain an A/B difference and is excluded from present-class F1type. Other is present but is the least prevalent pooled class in both directions.
- No KLT test class is wholly missing from its reciprocal training set. The only declared rare→present case is Other in Liver Fold B training (0.35%); therefore a newly appearing class is not a general explanation for the fold changes.
- FullFT’s mean F1type rises from 0.4751 in A to 0.5414 in B (B−A +0.0663); Selected PEFT changes from 0.4937 to 0.4895 (B−A -0.0042). These changes occur alongside reciprocal class-prevalence and density changes, but two slides per tissue do not isolate a cause.
- The largest mean per-class F1 shifts for Selected PEFT are Stromal -0.115, Immune +0.064, Epithelial +0.024. The seed-level matrices above show the corresponding confusion pairs; this is descriptive coincidence, not attribution.
- The largest mean per-class F1 shifts for FullFT are Epithelial +0.204, Stromal -0.113, Immune +0.110. The seed-level matrices above show the corresponding confusion pairs; this is descriptive coincidence, not attribution.

## Part B — tissue-specific difficulty

### Tissue-specific slide composition (shared by both backbones)

| Tissue/scope | Fold | Split | Slide(s) | Patches | Annotated cells | Nuclei/patch | Entropy (bits) | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast | A | TRAIN | `breast_s0` | 43,105 | 1,204,376 | 27.94 | 1.413 | 224,746 (18.66%) | 224,994 (18.68%) | 740,047 (61.45%) | 0 (0.00%) | 14,589 (1.21%) |
| Breast | A | TEST | `breast_s1` | 49,569 | 831,391 | 16.77 | 1.654 | 196,107 (23.59%) | 208,801 (25.11%) | 400,950 (48.23%) | 0 (0.00%) | 25,533 (3.07%) |
| Breast | B | TRAIN | `breast_s1` | 42,794 | 738,084 | 17.25 | 1.675 | 174,792 (23.68%) | 192,031 (26.02%) | 346,255 (46.91%) | 0 (0.00%) | 25,006 (3.39%) |
| Breast | B | TEST | `breast_s0` | 45,773 | 1,271,180 | 27.77 | 1.421 | 242,774 (19.10%) | 238,713 (18.78%) | 774,463 (60.92%) | 0 (0.00%) | 15,230 (1.20%) |
| Colon | A | TRAIN | `colon_s1` | 16,435 | 1,090,357 | 66.34 | 1.531 | 188,427 (17.28%) | 317,586 (29.13%) | 569,210 (52.20%) | 0 (0.00%) | 15,134 (1.39%) |
| Colon | A | TEST | `colon_s2` | 11,903 | 764,721 | 64.25 | 1.180 | 199,011 (26.02%) | 31,033 (4.06%) | 520,139 (68.02%) | 0 (0.00%) | 14,538 (1.90%) |
| Colon | B | TRAIN | `colon_s2` | 10,690 | 683,428 | 63.93 | 1.181 | 181,127 (26.50%) | 26,865 (3.93%) | 462,478 (67.67%) | 0 (0.00%) | 12,958 (1.90%) |
| Colon | B | TEST | `colon_s1` | 18,464 | 1,202,814 | 65.14 | 1.554 | 211,410 (17.58%) | 374,228 (31.11%) | 599,872 (49.87%) | 0 (0.00%) | 17,304 (1.44%) |
| Kidney | A | TRAIN | `kidney_s0` | 6,321 | 188,119 | 29.76 | 1.536 | 18,889 (10.04%) | 52,774 (28.05%) | 106,511 (56.62%) | 0 (0.00%) | 9,945 (5.29%) |
| Kidney | A | TEST | `kidney_s1` | 4,449 | 106,428 | 23.92 | 1.519 | 59,273 (55.69%) | 27,084 (25.45%) | 17,944 (16.86%) | 0 (0.00%) | 2,127 (2.00%) |
| Kidney | B | TRAIN | `kidney_s1` | 3,912 | 92,755 | 23.71 | 1.488 | 52,988 (57.13%) | 24,057 (25.94%) | 13,873 (14.96%) | 0 (0.00%) | 1,837 (1.98%) |
| Kidney | B | TEST | `kidney_s0` | 6,723 | 194,502 | 28.93 | 1.546 | 20,030 (10.30%) | 55,988 (28.79%) | 108,430 (55.75%) | 0 (0.00%) | 10,054 (5.17%) |
| Liver | A | TRAIN | `liver_s0` | 18,780 | 449,720 | 23.95 | 1.486 | 99,797 (22.19%) | 90,574 (20.14%) | 254,218 (56.53%) | 0 (0.00%) | 5,131 (1.14%) |
| Liver | A | TEST | `liver_s1` | 9,166 | 320,945 | 35.01 | 1.459 | 67,212 (20.94%) | 73,797 (22.99%) | 178,802 (55.71%) | 0 (0.00%) | 1,134 (0.35%) |
| Liver | B | TRAIN | `liver_s1` | 8,174 | 293,190 | 35.87 | 1.473 | 63,296 (21.59%) | 68,778 (23.46%) | 160,100 (54.61%) | 0 (0.00%) | 1,016 (0.35%) |
| Liver | B | TEST | `liver_s0` | 20,427 | 486,883 | 23.84 | 1.487 | 107,969 (22.18%) | 98,569 (20.24%) | 274,899 (56.46%) | 0 (0.00%) | 5,446 (1.12%) |
| Lung | A | TRAIN | `lung_s1` | 9,544 | 287,894 | 30.16 | 1.825 | 131,813 (45.79%) | 62,222 (21.61%) | 61,096 (21.22%) | 0 (0.00%) | 32,763 (11.38%) |
| Lung | A | TEST | `lung_s3` | 20,621 | 563,193 | 27.31 | 1.836 | 197,303 (35.03%) | 177,551 (31.53%) | 147,577 (26.20%) | 0 (0.00%) | 40,762 (7.24%) |
| Lung | B | TRAIN | `lung_s3` | 19,547 | 540,776 | 27.67 | 1.835 | 192,842 (35.66%) | 168,226 (31.11%) | 140,379 (25.96%) | 0 (0.00%) | 39,329 (7.27%) |
| Lung | B | TEST | `lung_s1` | 10,938 | 325,329 | 29.74 | 1.825 | 147,593 (45.37%) | 72,882 (22.40%) | 69,312 (21.31%) | 0 (0.00%) | 35,542 (10.92%) |
| Ovary | A | TRAIN | `ovary_s0` | 9,770 | 503,209 | 51.51 | 1.464 | 105,133 (20.89%) | 62,037 (12.33%) | 314,259 (62.45%) | 0 (0.00%) | 21,780 (4.33%) |
| Ovary | A | TEST | `ovary_s1` | 25,498 | 831,152 | 32.60 | 1.679 | 91,824 (11.05%) | 309,105 (37.19%) | 368,980 (44.39%) | 0 (0.00%) | 61,243 (7.37%) |
| Ovary | B | TRAIN | `ovary_s1` | 23,387 | 737,113 | 31.52 | 1.693 | 82,059 (11.13%) | 286,086 (38.81%) | 311,964 (42.32%) | 0 (0.00%) | 57,004 (7.73%) |
| Ovary | B | TEST | `ovary_s0` | 10,039 | 506,647 | 50.47 | 1.473 | 106,101 (20.94%) | 62,845 (12.40%) | 314,942 (62.16%) | 0 (0.00%) | 22,759 (4.49%) |
| Pancreatic | A | TRAIN | `pancreatic_s0` | 23,219 | 406,381 | 17.50 | 1.733 | 128,120 (31.53%) | 67,935 (16.72%) | 183,522 (45.16%) | 0 (0.00%) | 26,804 (6.60%) |
| Pancreatic | A | TEST | `pancreatic_s1` | 12,647 | 476,553 | 37.68 | 1.964 | 158,384 (33.24%) | 83,635 (17.55%) | 119,554 (25.09%) | 0 (0.00%) | 114,980 (24.13%) |
| Pancreatic | B | TRAIN | `pancreatic_s1` | 12,348 | 469,466 | 38.02 | 1.968 | 153,346 (32.66%) | 83,628 (17.81%) | 119,550 (25.47%) | 0 (0.00%) | 112,942 (24.06%) |
| Pancreatic | B | TEST | `pancreatic_s0` | 25,526 | 446,066 | 17.47 | 1.729 | 140,789 (31.56%) | 74,528 (16.71%) | 201,973 (45.28%) | 0 (0.00%) | 28,776 (6.45%) |
| Skin | A | TRAIN | `skin_s1` | 10,561 | 173,818 | 16.46 | 1.922 | 7,682 (4.42%) | 47,713 (27.45%) | 60,192 (34.63%) | 5,569 (3.20%) | 52,662 (30.30%) |
| Skin | A | TEST | `skin_s2` | 12,790 | 202,594 | 15.84 | 1.749 | 78,839 (38.91%) | 38,892 (19.20%) | 10,866 (5.36%) | 73,910 (36.48%) | 87 (0.04%) |
| Skin | B | TRAIN | `skin_s2` | 11,308 | 191,282 | 16.92 | 1.724 | 74,229 (38.81%) | 33,334 (17.43%) | 9,822 (5.13%) | 73,824 (38.59%) | 73 (0.04%) |
| Skin | B | TEST | `skin_s1` | 11,209 | 179,561 | 16.02 | 1.917 | 7,763 (4.32%) | 48,251 (26.87%) | 61,032 (33.99%) | 5,633 (3.14%) | 56,882 (31.68%) |
| Tonsil | A | TRAIN | `tonsil_s0` | 21,001 | 2,442,229 | 116.29 | 1.235 | 1,705,169 (69.82%) | 408,278 (16.72%) | 303,671 (12.43%) | 0 (0.00%) | 25,111 (1.03%) |
| Tonsil | A | TEST | `tonsil_s1` | 21,083 | 1,741,964 | 82.62 | 1.417 | 1,131,567 (64.96%) | 346,263 (19.88%) | 179,772 (10.32%) | 0 (0.00%) | 84,362 (4.84%) |
| Tonsil | B | TRAIN | `tonsil_s1` | 18,831 | 1,558,660 | 82.77 | 1.392 | 1,024,873 (65.75%) | 305,681 (19.61%) | 160,843 (10.32%) | 0 (0.00%) | 67,263 (4.32%) |
| Tonsil | B | TEST | `tonsil_s0` | 23,067 | 2,703,537 | 117.20 | 1.237 | 1,884,013 (69.69%) | 452,646 (16.74%) | 339,782 (12.57%) | 0 (0.00%) | 27,096 (1.00%) |

| Tissue/scope | Fold | TVD | JSD (bits) | Absent/rare in train but present in test |
|---|:---:|---:|---:|---|
| Breast | A | 0.132 | 0.014 | none |
| Breast | B | 0.140 | 0.017 | none |
| Colon | A | 0.251 | 0.091 | none |
| Colon | B | 0.272 | 0.103 | none |
| Kidney | A | 0.457 | 0.214 | none |
| Kidney | B | 0.468 | 0.225 | none |
| Liver | A | 0.029 | 0.002 | none |
| Liver | B | 0.032 | 0.003 | Other rare in train (0.35%) |
| Lung | A | 0.149 | 0.017 | none |
| Lung | B | 0.134 | 0.014 | none |
| Ovary | A | 0.279 | 0.072 | none |
| Ovary | B | 0.296 | 0.079 | none |
| Pancreatic | A | 0.201 | 0.060 | none |
| Pancreatic | B | 0.198 | 0.059 | none |
| Skin | A | 0.678 | 0.473 | none |
| Skin | B | 0.699 | 0.492 | Other rare in train (0.04%) |
| Tonsil | A | 0.070 | 0.012 | none |
| Tonsil | B | 0.062 | 0.010 | none |

### Complete seed42 Selected PEFT results and completed-direction ranks

Ranks are descending within each backbone over currently complete tissue×fold directions. They are descriptive; incomplete directions are not assigned a rank.

| Backbone | Tissue | Fold | bPQ (rank) | mPQ (rank) | F1det (rank) | F1type (rank) | Type F1/support by class | Dominant paired confusion | Strong detection / weak typing flag |
|---|---|:---:|---:|---:|---:|---:|---|---|:---:|
| CellViT-SAM-H | Breast | A | 0.4113 (12) | 0.2386 (5) | 0.8129 (12) | 0.5000 (4) | Immune 0.509 (193,476); Stromal 0.624 (205,990); Epithelial 0.838 (395,725); Other 0.030 (25,099) | Immune→Stromal (57,086) | no |
| CellViT-SAM-H | Breast | B | 0.4525 (8) | 0.2779 (1) | 0.8318 (11) | 0.5343 (1) | Immune 0.692 (238,925); Stromal 0.493 (234,604); Epithelial 0.930 (762,687); Other 0.023 (15,016) | Stromal→Immune (95,534) | no |
| CellViT-SAM-H | Colon | A | 0.2847 (17) | 0.1495 (12) | 0.7510 (13) | 0.4265 (8) | Immune 0.569 (195,875); Stromal 0.276 (30,566); Epithelial 0.850 (511,629); Other 0.011 (14,283) | Epithelial→Immune (56,422) | no |
| CellViT-SAM-H | Colon | B | 0.2865 (16) | 0.1199 (15) | 0.7441 (14) | 0.3867 (12) | Immune 0.384 (208,114); Stromal 0.296 (368,241); Epithelial 0.866 (590,393); Other 0.000 (17,032) | Stromal→Immune (156,281) | no |
| CellViT-SAM-H | Kidney | A | 0.5495 (4) | 0.0904 (16) | 0.8478 (5) | 0.1492 (17) | Immune 0.070 (58,422); Stromal 0.403 (26,644); Epithelial 0.121 (17,669); Other 0.004 (2,100) | Immune→Stromal (46,015) | yes |
| CellViT-SAM-H | Kidney | B | 0.6094 (1) | 0.0529 (18) | 0.9008 (1) | 0.1060 (18) | Immune 0.172 (19,732); Stromal 0.156 (55,116); Epithelial 0.095 (106,874); Other 0.000 (9,921) | Epithelial→Immune (84,882) | yes |
| CellViT-SAM-H | Liver | A | 0.4801 (7) | 0.1715 (10) | 0.8432 (6) | 0.2649 (15) | Immune 0.017 (66,191); Stromal 0.446 (72,684); Epithelial 0.596 (175,936); Other 0.000 (1,124) | Epithelial→Stromal (79,410) | yes |
| CellViT-SAM-H | Liver | B | 0.5219 (5) | 0.2446 (4) | 0.8826 (2) | 0.4030 (11) | Immune 0.329 (106,478); Stromal 0.465 (97,143); Epithelial 0.818 (271,122); Other 0.000 (5,350) | Immune→Epithelial (56,418) | yes |
| CellViT-SAM-H | Lung | A | 0.5083 (6) | 0.1934 (8) | 0.8430 (7) | 0.4042 (10) | Immune 0.622 (194,173); Stromal 0.639 (174,761); Epithelial 0.355 (145,102); Other 0.001 (40,119) | Epithelial→Immune (61,435) | yes |
| CellViT-SAM-H | Lung | B | 0.5637 (3) | 0.2514 (3) | 0.8619 (3) | 0.4701 (7) | Immune 0.726 (145,454); Stromal 0.434 (71,716); Epithelial 0.719 (68,294); Other 0.001 (34,934) | Stromal→Immune (39,038) | no |
| CellViT-SAM-H | Ovary | A | 0.4488 (10) | 0.2226 (6) | 0.8344 (9) | 0.5054 (3) | Immune 0.362 (90,259); Stromal 0.775 (304,120); Epithelial 0.884 (360,874); Other 0.000 (60,217) | Stromal→Immune (33,012) | no |
| CellViT-SAM-H | Ovary | B | 0.3874 (14) | 0.1548 (11) | 0.8328 (10) | 0.4806 (6) | Immune 0.345 (104,483); Stromal 0.647 (61,888); Epithelial 0.713 (310,152); Other 0.218 (22,354) | Epithelial→Immune (67,466) | no |
| CellViT-SAM-H | Pancreatic | A | 0.3774 (15) | 0.1239 (14) | 0.7338 (16) | 0.2788 (14) | Immune 0.614 (155,985); Stromal 0.381 (82,287); Epithelial 0.119 (117,706); Other 0.002 (113,164) | Other→Immune (46,962) | no |
| CellViT-SAM-H | Pancreatic | B | 0.3892 (13) | 0.1745 (9) | 0.7250 (17) | 0.4167 (9) | Immune 0.574 (138,877); Stromal 0.388 (73,507); Epithelial 0.695 (199,217); Other 0.010 (28,420) | Epithelial→Stromal (46,219) | no |
| CellViT-SAM-H | Skin | A | 0.4401 (11) | 0.1342 (13) | 0.7432 (15) | 0.2425 (16) | Immune 0.455 (77,761); Stromal 0.283 (38,319); Epithelial 0.472 (10,718); Melanocyte 0.001 (72,962); Other 0.002 (87) | Melanocyte→Stromal (39,468) | no |
| CellViT-SAM-H | Skin | B | 0.2137 (18) | 0.0717 (17) | 0.6628 (18) | 0.2936 (13) | Immune 0.176 (7,656); Stromal 0.497 (47,505); Epithelial 0.761 (60,109); Melanocyte 0.034 (5,534); Other 0.000 (55,988) | Stromal→Epithelial (8,991) | no |
| CellViT-SAM-H | Tonsil | A | 0.4522 (9) | 0.2158 (7) | 0.8383 (8) | 0.4879 (5) | Immune 0.810 (1,115,096); Stromal 0.506 (341,054); Epithelial 0.636 (177,099); Other 0.000 (83,059) | Immune→Stromal (125,211) | no |
| CellViT-SAM-H | Tonsil | B | 0.5771 (2) | 0.2603 (2) | 0.8560 (4) | 0.5108 (2) | Immune 0.861 (1,855,942); Stromal 0.410 (445,916); Epithelial 0.677 (334,497); Other 0.095 (26,622) | Stromal→Immune (224,338) | no |
| CellViT-256 | Breast | A | 0.4537 (8) | 0.2366 (3) | 0.7918 (11) | 0.4639 (5) | Immune 0.426 (192,844); Stromal 0.598 (205,192); Epithelial 0.814 (394,395); Other 0.018 (25,045) | Immune→Stromal (71,867) | no |
| CellViT-256 | Breast | B | 0.5101 (4) | 0.3023 (1) | 0.8349 (5) | 0.5227 (1) | Immune 0.653 (238,576); Stromal 0.490 (233,889); Epithelial 0.877 (759,164); Other 0.071 (14,928) | Stromal→Immune (85,573) | no |
| CellViT-256 | Colon | A | 0.3346 (15) | 0.1151 (10) | 0.7513 (13) | 0.3301 (11) | Immune 0.476 (195,544); Stromal 0.229 (30,511); Epithelial 0.569 (510,930); Other 0.047 (14,183) | Epithelial→Immune (166,212) | no |
| CellViT-256 | Colon | B | 0.2395 (17) | 0.1079 (11) | 0.7105 (15) | 0.4724 (4) | Immune 0.435 (205,560); Stromal 0.502 (362,553); Epithelial 0.882 (585,589); Other 0.071 (16,842) | Stromal→Immune (96,174) | no |
| CellViT-256 | Kidney | A | 0.4681 (7) | 0.0989 (13) | 0.8339 (6) | 0.2104 (14) | Immune 0.281 (58,172); Stromal 0.345 (26,332); Epithelial 0.190 (17,483); Other 0.025 (2,066) | Immune→Stromal (30,095) | yes |
| CellViT-256 | Kidney | B | 0.5683 (2) | 0.0965 (15) | 0.8938 (1) | 0.1727 (15) | Immune 0.120 (19,709); Stromal 0.135 (54,995); Epithelial 0.401 (106,277); Other 0.035 (9,885) | Epithelial→Immune (44,356) | yes |
| CellViT-256 | Liver | A | 0.4794 (6) | 0.0737 (17) | 0.8325 (7) | 0.1136 (18) | Immune 0.006 (66,172); Stromal 0.368 (72,592); Epithelial 0.080 (175,685); Other 0.000 (1,124) | Epithelial→Stromal (132,257) | yes |
| CellViT-256 | Liver | B | 0.5721 (1) | 0.2513 (2) | 0.8653 (2) | 0.3963 (10) | Immune 0.430 (104,556); Stromal 0.354 (95,611); Epithelial 0.801 (266,639); Other 0.000 (5,311) | Immune→Epithelial (42,113) | yes |
| CellViT-256 | Lung | A | 0.4125 (11) | 0.1481 (8) | 0.8297 (8) | 0.4050 (9) | Immune 0.572 (193,054); Stromal 0.589 (173,776); Epithelial 0.320 (143,632); Other 0.139 (39,652) | Immune→Stromal (51,043) | no |
| CellViT-256 | Lung | B | 0.4963 (5) | 0.2168 (5) | 0.8282 (9) | 0.4546 (7) | Immune 0.702 (144,815); Stromal 0.376 (70,978); Epithelial 0.621 (67,669); Other 0.119 (34,686) | Stromal→Immune (34,229) | no |
| CellViT-256 | Ovary | A | 0.4126 (10) | 0.1994 (6) | 0.7846 (12) | 0.5158 (2) | Immune 0.341 (89,958); Stromal 0.751 (303,337); Epithelial 0.845 (360,012); Other 0.126 (60,005) | Epithelial→Immune (36,525) | no |
| CellViT-256 | Ovary | B | 0.4139 (9) | 0.1477 (9) | 0.8385 (4) | 0.4263 (8) | Immune 0.357 (104,163); Stromal 0.465 (60,936); Epithelial 0.654 (309,004); Other 0.230 (22,182) | Epithelial→Immune (81,463) | no |
| CellViT-256 | Pancreatic | A | 0.3500 (14) | 0.0970 (14) | 0.7363 (14) | 0.2328 (13) | Immune 0.573 (155,402); Stromal 0.284 (81,687); Epithelial 0.067 (117,095); Other 0.006 (112,225) | Epithelial→Immune (54,911) | no |
| CellViT-256 | Pancreatic | B | 0.2998 (16) | 0.1067 (12) | 0.6303 (18) | 0.3151 (12) | Immune 0.452 (138,213); Stromal 0.321 (73,108); Epithelial 0.468 (197,644); Other 0.020 (28,187) | Epithelial→Stromal (77,991) | no |
| CellViT-256 | Skin | A | 0.3944 (13) | 0.0832 (16) | 0.7087 (16) | 0.1309 (17) | Immune 0.193 (77,698); Stromal 0.262 (38,263); Epithelial 0.189 (10,678); Melanocyte 0.009 (72,678); Other 0.001 (87) | Immune→Other (38,620) | no |
| CellViT-256 | Skin | B | 0.2221 (18) | 0.0306 (18) | 0.6782 (17) | 0.1505 (16) | Immune 0.016 (7,633); Stromal 0.106 (47,418); Epithelial 0.630 (60,114); Melanocyte 0.000 (5,532); Other 0.000 (55,877) | Stromal→Epithelial (24,057) | no |
| CellViT-256 | Tonsil | A | 0.4091 (12) | 0.1767 (7) | 0.8278 (10) | 0.4587 (6) | Immune 0.777 (1,114,971); Stromal 0.479 (340,968); Epithelial 0.579 (177,080); Other 0.000 (83,047) | Immune→Stromal (156,900) | no |
| CellViT-256 | Tonsil | B | 0.5274 (3) | 0.2295 (4) | 0.8632 (3) | 0.4998 (3) | Immune 0.839 (1,852,833); Stromal 0.408 (445,134); Epithelial 0.690 (333,755); Other 0.062 (26,588) | Stromal→Immune (192,828) | no |

### Matched Kidney/Liver/Tonsil PEFT versus FullFT diagnostics

Only exact tissue/fold/seed42 pairs with identical TEST patch-ID sets are included. Per-class support is matched-plus-unmatched true support; predicted frequencies include paired and unmatched foreground predictions.

| Backbone | Tissue | Fold | Method | bPQ | mPQ | F1det | F1type | Per-class F1 / true support | Predicted foreground frequency | Dominant paired confusion |
|---|---|:---:|---|---:|---:|---:|---:|---|---|---|
| CellViT-SAM-H | Kidney | A | Selected PEFT | 0.5495 | 0.0904 | 0.8478 | 0.1492 | Immune 0.070 / 58,422; Stromal 0.403 / 26,644; Epithelial 0.121 / 17,669; Melanocyte -- / 0; Other 0.004 / 2,100 | Immune 3.9%; Stromal 89.8%; Epithelial 6.3%; Melanocyte 0.0%; Other 0.0% | Immune->Stromal (46,015) |
| CellViT-SAM-H | Kidney | A | FullFT | 0.5917 | 0.1814 | 0.8564 | 0.3071 | Immune 0.495 / 58,398; Stromal 0.481 / 26,630; Epithelial 0.128 / 17,620; Melanocyte -- / 0; Other 0.125 / 2,102 | Immune 30.0%; Stromal 59.8%; Epithelial 9.0%; Melanocyte 0.0%; Other 1.2% | Immune->Stromal (26,532) |
| CellViT-SAM-H | Kidney | B | Selected PEFT | 0.6094 | 0.0529 | 0.9008 | 0.1060 | Immune 0.172 / 19,732; Stromal 0.156 / 55,116; Epithelial 0.095 / 106,874; Melanocyte -- / 0; Other 0.000 / 9,921 | Immune 85.6%; Stromal 5.6%; Epithelial 8.8%; Melanocyte 0.0%; Other 0.0% | Epithelial->Immune (84,882) |
| CellViT-SAM-H | Kidney | B | FullFT | 0.6087 | 0.0888 | 0.8947 | 0.2244 | Immune 0.112 / 19,736; Stromal 0.347 / 55,104; Epithelial 0.174 / 106,591; Melanocyte -- / 0; Other 0.264 / 9,917 | Immune 55.0%; Stromal 17.8%; Epithelial 21.7%; Melanocyte 0.0%; Other 5.5% | Epithelial->Immune (66,123) |
| CellViT-SAM-H | Liver | A | Selected PEFT | 0.4801 | 0.1715 | 0.8432 | 0.2649 | Immune 0.017 / 66,191; Stromal 0.446 / 72,684; Epithelial 0.596 / 175,936; Melanocyte -- / 0; Other 0.000 / 1,124 | Immune 0.4%; Stromal 71.5%; Epithelial 28.2%; Melanocyte 0.0%; Other 0.0% | Epithelial->Stromal (79,410) |
| CellViT-SAM-H | Liver | A | FullFT | 0.4934 | 0.1464 | 0.8410 | 0.2225 | Immune 0.037 / 66,226; Stromal 0.418 / 72,721; Epithelial 0.411 / 176,089; Melanocyte -- / 0; Other 0.024 / 1,125 | Immune 1.3%; Stromal 82.7%; Epithelial 15.3%; Melanocyte 0.0%; Other 0.8% | Epithelial->Stromal (101,689) |
| CellViT-SAM-H | Liver | B | Selected PEFT | 0.5219 | 0.2446 | 0.8826 | 0.4030 | Immune 0.329 / 106,478; Stromal 0.465 / 97,143; Epithelial 0.818 / 271,122; Melanocyte -- / 0; Other 0.000 / 5,350 | Immune 6.6%; Stromal 11.4%; Epithelial 82.0%; Melanocyte 0.0%; Other 0.0% | Immune->Epithelial (56,418) |
| CellViT-SAM-H | Liver | B | FullFT | 0.5075 | 0.2605 | 0.8836 | 0.4671 | Immune 0.395 / 105,407; Stromal 0.527 / 96,415; Epithelial 0.845 / 269,834; Melanocyte -- / 0; Other 0.101 / 5,274 | Immune 9.5%; Stromal 13.9%; Epithelial 76.1%; Melanocyte 0.0%; Other 0.5% | Immune->Epithelial (45,786) |
| CellViT-SAM-H | Tonsil | A | Selected PEFT | 0.4522 | 0.2158 | 0.8383 | 0.4879 | Immune 0.810 / 1,115,096; Stromal 0.506 / 341,054; Epithelial 0.636 / 177,099; Melanocyte -- / 0; Other 0.000 / 83,059 | Immune 59.0%; Stromal 21.0%; Epithelial 20.0%; Melanocyte 0.0%; Other 0.0% | Immune->Stromal (125,211) |
| CellViT-SAM-H | Tonsil | A | FullFT | 0.4632 | 0.2390 | 0.8270 | 0.5385 | Immune 0.816 / 1,115,079; Stromal 0.531 / 341,055; Epithelial 0.714 / 177,095; Melanocyte -- / 0; Other 0.094 / 83,061 | Immune 58.6%; Stromal 25.5%; Epithelial 15.4%; Melanocyte 0.0%; Other 0.5% | Immune->Stromal (155,272) |
| CellViT-SAM-H | Tonsil | B | Selected PEFT | 0.5771 | 0.2603 | 0.8560 | 0.5108 | Immune 0.861 / 1,855,942; Stromal 0.410 / 445,916; Epithelial 0.677 / 334,497; Melanocyte -- / 0; Other 0.095 / 26,622 | Immune 76.5%; Stromal 15.1%; Epithelial 8.1%; Melanocyte 0.0%; Other 0.3% | Stromal->Immune (224,338) |
| CellViT-SAM-H | Tonsil | B | FullFT | 0.5837 | 0.2722 | 0.8561 | 0.5269 | Immune 0.867 / 1,856,097; Stromal 0.446 / 445,973; Epithelial 0.640 / 334,702; Melanocyte -- / 0; Other 0.155 / 26,691 | Immune 76.4%; Stromal 15.4%; Epithelial 7.2%; Melanocyte 0.0%; Other 1.0% | Stromal->Immune (207,518) |
| CellViT-256 | Kidney | A | Selected PEFT | 0.4681 | 0.0989 | 0.8339 | 0.2104 | Immune 0.281 / 58,172; Stromal 0.345 / 26,332; Epithelial 0.190 / 17,483; Melanocyte -- / 0; Other 0.025 / 2,066 | Immune 12.6%; Stromal 58.7%; Epithelial 28.4%; Melanocyte 0.0%; Other 0.3% | Immune->Stromal (30,095) |
| CellViT-256 | Kidney | A | FullFT | 0.5707 | 0.1578 | 0.8549 | 0.3170 | Immune 0.494 / 58,361; Stromal 0.433 / 26,578; Epithelial 0.190 / 17,556; Melanocyte -- / 0; Other 0.150 / 2,094 | Immune 28.7%; Stromal 53.9%; Epithelial 15.3%; Melanocyte 0.0%; Other 2.1% | Immune->Stromal (25,345) |
| CellViT-256 | Kidney | B | Selected PEFT | 0.5683 | 0.0965 | 0.8938 | 0.1727 | Immune 0.120 / 19,709; Stromal 0.135 / 54,995; Epithelial 0.401 / 106,277; Melanocyte -- / 0; Other 0.035 / 9,885 | Immune 37.3%; Stromal 10.9%; Epithelial 50.4%; Melanocyte 0.0%; Other 1.3% | Epithelial->Immune (44,356) |
| CellViT-256 | Kidney | B | FullFT | 0.5940 | 0.0915 | 0.8931 | 0.2060 | Immune 0.094 / 19,708; Stromal 0.211 / 55,000; Epithelial 0.351 / 106,087; Melanocyte -- / 0; Other 0.168 / 9,891 | Immune 43.0%; Stromal 11.7%; Epithelial 43.4%; Melanocyte 0.0%; Other 1.9% | Epithelial->Immune (52,339) |
| CellViT-256 | Liver | A | Selected PEFT | 0.4794 | 0.0737 | 0.8325 | 0.1136 | Immune 0.006 / 66,172; Stromal 0.368 / 72,592; Epithelial 0.080 / 175,685; Melanocyte -- / 0; Other 0.000 / 1,124 | Immune 0.1%; Stromal 97.0%; Epithelial 2.8%; Melanocyte 0.0%; Other 0.0% | Epithelial->Stromal (132,257) |
| CellViT-256 | Liver | A | FullFT | 0.4710 | 0.0821 | 0.8206 | 0.1327 | Immune 0.015 / 66,223; Stromal 0.364 / 72,714; Epithelial 0.143 / 176,113; Melanocyte -- / 0; Other 0.009 / 1,125 | Immune 0.5%; Stromal 94.1%; Epithelial 4.6%; Melanocyte 0.0%; Other 0.7% | Epithelial->Stromal (126,702) |
| CellViT-256 | Liver | B | Selected PEFT | 0.5721 | 0.2513 | 0.8653 | 0.3963 | Immune 0.430 / 104,556; Stromal 0.354 / 95,611; Epithelial 0.801 / 266,639; Melanocyte -- / 0; Other 0.000 / 5,311 | Immune 15.4%; Stromal 8.9%; Epithelial 75.6%; Melanocyte 0.0%; Other 0.0% | Immune->Epithelial (42,113) |
| CellViT-256 | Liver | B | FullFT | 0.4624 | 0.1962 | 0.8770 | 0.3894 | Immune 0.308 / 105,428; Stromal 0.380 / 96,347; Epithelial 0.806 / 269,577; Melanocyte -- / 0; Other 0.063 / 5,310 | Immune 6.9%; Stromal 9.0%; Epithelial 83.7%; Melanocyte 0.0%; Other 0.4% | Immune->Epithelial (55,682) |
| CellViT-256 | Tonsil | A | Selected PEFT | 0.4091 | 0.1767 | 0.8278 | 0.4587 | Immune 0.777 / 1,114,971; Stromal 0.479 / 340,968; Epithelial 0.579 / 177,080; Melanocyte -- / 0; Other 0.000 / 83,047 | Immune 54.1%; Stromal 23.8%; Epithelial 22.0%; Melanocyte 0.0%; Other 0.0% | Immune->Stromal (156,900) |
| CellViT-256 | Tonsil | A | FullFT | 0.4516 | 0.2115 | 0.8207 | 0.5249 | Immune 0.816 / 1,114,169; Stromal 0.492 / 340,401; Epithelial 0.618 / 176,927; Melanocyte -- / 0; Other 0.174 / 82,849 | Immune 59.0%; Stromal 18.2%; Epithelial 21.2%; Melanocyte 0.0%; Other 1.6% | Immune->Epithelial (103,365) |
| CellViT-256 | Tonsil | B | Selected PEFT | 0.5274 | 0.2295 | 0.8632 | 0.4998 | Immune 0.839 / 1,852,833; Stromal 0.408 / 445,134; Epithelial 0.690 / 333,755; Melanocyte -- / 0; Other 0.062 / 26,588 | Immune 71.2%; Stromal 18.3%; Epithelial 9.9%; Melanocyte 0.0%; Other 0.5% | Stromal->Immune (192,828) |
| CellViT-256 | Tonsil | B | FullFT | 0.5710 | 0.2606 | 0.8543 | 0.5311 | Immune 0.853 / 1,855,413; Stromal 0.441 / 445,847; Epithelial 0.691 / 334,504; Melanocyte -- / 0; Other 0.139 / 26,666 | Immune 73.0%; Stromal 17.6%; Epithelial 8.6%; Melanocyte 0.0%; Other 0.8% | Stromal->Immune (196,057) |

### Matched tissue paired confusion matrices

Rows are true labels and columns predicted labels; counts are paired matches only.

#### CellViT-SAM-H — Kidney — Selected PEFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 15 | 1,900 | 46,015 | 3,335 | 0 | 4 |
| Stromal | 17 | 270 | 19,838 | 877 | 0 | 1 |
| Epithelial | 12 | 748 | 10,823 | 1,087 | 0 | 2 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 6 | 142 | 903 | 34 | 0 | 2 |

#### CellViT-SAM-H — Kidney — FullFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 39 | 19,404 | 26,532 | 5,299 | 0 | 304 |
| Stromal | 31 | 2,280 | 17,254 | 1,253 | 0 | 131 |
| Epithelial | 61 | 4,771 | 6,522 | 1,335 | 0 | 317 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 4 | 445 | 566 | 29 | 0 | 128 |

#### CellViT-SAM-H — Kidney — Selected PEFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 9 | 13,537 | 709 | 2,009 | 0 | 0 |
| Stromal | 23 | 38,269 | 4,345 | 4,414 | 0 | 0 |
| Epithelial | 50 | 84,882 | 2,956 | 5,074 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 6 | 4,101 | 600 | 2,028 | 0 | 0 |

#### CellViT-SAM-H — Kidney — FullFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 5 | 5,917 | 2,337 | 6,512 | 0 | 1,494 |
| Stromal | 35 | 16,472 | 12,928 | 13,845 | 0 | 3,049 |
| Epithelial | 333 | 66,123 | 12,051 | 10,880 | 0 | 1,756 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 10 | 999 | 834 | 2,859 | 0 | 1,972 |

#### CellViT-SAM-H — Liver — Selected PEFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 1 | 0 | 0 | 0 | 0 | 0 |
| Immune | 38 | 467 | 50,198 | 2,452 | 0 | 1 |
| Stromal | 44 | 96 | 53,619 | 3,035 | 0 | 1 |
| Epithelial | 242 | 309 | 79,410 | 62,916 | 0 | 3 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 0 | 394 | 25 | 0 | 0 |

#### CellViT-SAM-H — Liver — FullFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 3 | 1,064 | 51,650 | 860 | 0 | 466 |
| Stromal | 7 | 399 | 55,757 | 971 | 0 | 320 |
| Epithelial | 89 | 1,961 | 101,689 | 37,169 | 0 | 1,038 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 1 | 415 | 10 | 0 | 28 |

#### CellViT-SAM-H — Liver — Selected PEFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 17 | 19,085 | 13,847 | 56,418 | 0 | 0 |
| Stromal | 27 | 5,662 | 28,105 | 42,376 | 0 | 0 |
| Epithelial | 34 | 1,260 | 2,462 | 234,453 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 2 | 809 | 294 | 1,546 | 0 | 0 |

#### CellViT-SAM-H — Liver — FullFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 4 | 0 | 0 | 0 | 0 | 0 |
| Immune | 1,088 | 24,470 | 16,475 | 45,786 | 0 | 517 |
| Stromal | 755 | 8,418 | 33,617 | 31,423 | 0 | 606 |
| Epithelial | 1,322 | 2,472 | 3,137 | 228,551 | 0 | 113 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 78 | 1,294 | 280 | 753 | 0 | 200 |

#### CellViT-SAM-H — Tonsil — Selected PEFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 11 | 0 | 0 | 0 | 0 | 0 |
| Immune | 38 | 702,108 | 125,211 | 94,099 | 0 | 0 |
| Stromal | 35 | 87,652 | 137,221 | 36,354 | 0 | 0 |
| Epithelial | 18 | 6,839 | 2,246 | 131,783 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 14 | 15,982 | 16,921 | 11,261 | 0 | 0 |

#### CellViT-SAM-H — Tonsil — FullFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 11 | 0 | 0 | 0 | 0 | 0 |
| Immune | 55 | 698,947 | 155,272 | 60,543 | 0 | 1,700 |
| Stromal | 34 | 77,420 | 158,796 | 18,882 | 0 | 2,085 |
| Epithelial | 22 | 10,562 | 4,153 | 124,916 | 0 | 261 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 12 | 10,745 | 23,172 | 5,438 | 0 | 2,239 |

#### CellViT-SAM-H — Tonsil — Selected PEFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 18 | 0 | 0 | 0 | 0 | 0 |
| Immune | 347 | 1,401,131 | 123,015 | 24,362 | 0 | 2,375 |
| Stromal | 111 | 224,338 | 141,249 | 1,835 | 0 | 1,671 |
| Epithelial | 298 | 68,778 | 51,101 | 155,440 | 0 | 1,781 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 82 | 8,800 | 4,712 | 310 | 0 | 1,028 |

#### CellViT-SAM-H — Tonsil — FullFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 19 | 0 | 0 | 0 | 0 | 0 |
| Immune | 192 | 1,409,477 | 111,984 | 19,033 | 0 | 9,847 |
| Stromal | 54 | 207,518 | 155,858 | 876 | 0 | 3,967 |
| Epithelial | 93 | 75,350 | 59,085 | 139,761 | 0 | 2,642 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 13 | 8,193 | 4,122 | 298 | 0 | 2,658 |

#### CellViT-256 — Kidney — Selected PEFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 265 | 8,642 | 30,095 | 11,255 | 0 | 107 |
| Stromal | 329 | 874 | 12,253 | 7,564 | 0 | 62 |
| Epithelial | 198 | 1,657 | 7,186 | 3,265 | 0 | 42 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 40 | 140 | 724 | 96 | 0 | 15 |

#### CellViT-256 — Kidney — FullFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 76 | 19,035 | 25,345 | 6,491 | 0 | 536 |
| Stromal | 83 | 2,483 | 14,467 | 3,527 | 0 | 258 |
| Epithelial | 125 | 3,746 | 5,772 | 2,368 | 0 | 573 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 12 | 320 | 463 | 60 | 0 | 195 |

#### CellViT-256 — Kidney — Selected PEFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 8 | 0 | 0 | 0 | 0 | 0 |
| Immune | 32 | 4,566 | 886 | 10,134 | 0 | 344 |
| Stromal | 144 | 10,975 | 4,279 | 29,987 | 0 | 1,049 |
| Epithelial | 647 | 44,356 | 11,589 | 34,158 | 0 | 544 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 42 | 564 | 361 | 5,247 | 0 | 146 |

#### CellViT-256 — Kidney — FullFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 7 | 0 | 0 | 0 | 0 | 0 |
| Immune | 33 | 4,016 | 1,404 | 10,302 | 0 | 377 |
| Stromal | 139 | 12,786 | 6,794 | 25,551 | 0 | 951 |
| Epithelial | 837 | 52,339 | 9,462 | 27,807 | 0 | 638 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 36 | 612 | 537 | 4,668 | 0 | 788 |

#### CellViT-256 — Liver — Selected PEFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 57 | 165 | 52,458 | 335 | 0 | 25 |
| Stromal | 136 | 53 | 54,246 | 862 | 0 | 10 |
| Epithelial | 493 | 89 | 132,257 | 5,815 | 0 | 56 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 1 | 0 | 526 | 2 | 0 | 0 |

#### CellViT-256 — Liver — FullFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 3 | 0 | 0 | 0 | 0 | 0 |
| Immune | 6 | 393 | 50,522 | 190 | 0 | 172 |
| Stromal | 14 | 122 | 51,060 | 262 | 0 | 97 |
| Epithelial | 65 | 753 | 126,702 | 10,742 | 0 | 1,010 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 0 | 0 | 343 | 6 | 0 | 7 |

#### CellViT-256 — Liver — Selected PEFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 1,939 | 29,292 | 8,609 | 42,113 | 0 | 21 |
| Stromal | 1,559 | 12,601 | 17,539 | 38,327 | 0 | 11 |
| Epithelial | 4,517 | 13,264 | 4,304 | 199,342 | 0 | 17 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 41 | 904 | 286 | 862 | 0 | 0 |

#### CellViT-256 — Liver — FullFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 5 | 0 | 0 | 0 | 0 | 0 |
| Immune | 1,067 | 16,804 | 10,853 | 55,682 | 0 | 389 |
| Stromal | 823 | 5,697 | 19,677 | 44,920 | 0 | 479 |
| Epithelial | 1,579 | 2,217 | 1,900 | 220,493 | 0 | 130 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 42 | 592 | 346 | 1,161 | 0 | 104 |

#### CellViT-256 — Tonsil — Selected PEFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 10 | 0 | 0 | 0 | 0 | 0 |
| Immune | 163 | 626,765 | 156,900 | 119,269 | 0 | 0 |
| Stromal | 121 | 67,032 | 131,238 | 41,450 | 0 | 0 |
| Epithelial | 37 | 4,704 | 5,042 | 121,799 | 0 | 0 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 26 | 11,592 | 15,477 | 6,757 | 0 | 0 |

#### CellViT-256 — Tonsil — FullFT — Fold A

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 10 | 0 | 0 | 0 | 0 | 0 |
| Immune | 965 | 704,799 | 99,606 | 103,365 | 0 | 5,797 |
| Stromal | 688 | 90,572 | 124,315 | 40,263 | 0 | 7,133 |
| Epithelial | 190 | 6,205 | 1,747 | 132,177 | 0 | 303 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 224 | 12,726 | 16,969 | 11,838 | 0 | 5,759 |

#### CellViT-256 — Tonsil — Selected PEFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 20 | 0 | 0 | 0 | 0 | 0 |
| Immune | 3,456 | 1,280,797 | 189,661 | 41,843 | 0 | 4,986 |
| Stromal | 893 | 192,828 | 149,761 | 7,657 | 0 | 1,880 |
| Epithelial | 1,040 | 54,621 | 39,838 | 164,752 | 0 | 3,456 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 116 | 6,372 | 3,458 | 366 | 0 | 679 |

#### CellViT-256 — Tonsil — FullFT — Fold B

| True \ Pred | Background | Immune | Stromal | Epithelial | Melanocyte | Other |
|---|---:|---:|---:|---:|---:|---:|
| Background | 19 | 0 | 0 | 0 | 0 | 0 |
| Immune | 876 | 1,350,504 | 161,344 | 27,497 | 0 | 8,470 |
| Stromal | 180 | 196,057 | 163,813 | 2,842 | 0 | 3,416 |
| Epithelial | 291 | 63,693 | 46,818 | 160,887 | 0 | 2,487 |
| Melanocyte | 0 | 0 | 0 | 0 | 0 | 0 |
| Other | 38 | 7,813 | 4,344 | 343 | 0 | 2,165 |


### Reciprocal-fold tissue mean ranks

Only tissues with both A and B complete are included; means are unweighted across the two directions.

| Backbone | Tissue | bPQ mean (rank) | mPQ mean (rank) | F1det mean (rank) | F1type mean (rank) |
|---|---|---:|---:|---:|---:|
| CellViT-SAM-H | Breast | 0.4319 (5) | 0.2583 (1) | 0.8223 (6) | 0.5172 (1) |
| CellViT-SAM-H | Tonsil | 0.5147 (3) | 0.2381 (2) | 0.8471 (4) | 0.4993 (2) |
| CellViT-SAM-H | Lung | 0.5360 (2) | 0.2224 (3) | 0.8525 (3) | 0.4371 (4) |
| CellViT-SAM-H | Liver | 0.5010 (4) | 0.2080 (4) | 0.8629 (2) | 0.3340 (7) |
| CellViT-SAM-H | Ovary | 0.4181 (6) | 0.1887 (5) | 0.8336 (5) | 0.4930 (3) |
| CellViT-SAM-H | Pancreatic | 0.3833 (7) | 0.1492 (6) | 0.7294 (8) | 0.3477 (6) |
| CellViT-SAM-H | Colon | 0.2856 (9) | 0.1347 (7) | 0.7475 (7) | 0.4066 (5) |
| CellViT-SAM-H | Skin | 0.3269 (8) | 0.1029 (8) | 0.7030 (9) | 0.2680 (8) |
| CellViT-SAM-H | Kidney | 0.5795 (1) | 0.0716 (9) | 0.8743 (1) | 0.1276 (9) |
| CellViT-256 | Breast | 0.4819 (3) | 0.2694 (1) | 0.8134 (5) | 0.4933 (1) |
| CellViT-256 | Tonsil | 0.4682 (4) | 0.2031 (2) | 0.8455 (3) | 0.4793 (2) |
| CellViT-256 | Lung | 0.4544 (5) | 0.1824 (3) | 0.8289 (4) | 0.4298 (4) |
| CellViT-256 | Ovary | 0.4133 (6) | 0.1736 (4) | 0.8115 (6) | 0.4711 (3) |
| CellViT-256 | Liver | 0.5258 (1) | 0.1625 (5) | 0.8489 (2) | 0.2550 (7) |
| CellViT-256 | Colon | 0.2871 (9) | 0.1115 (6) | 0.7309 (7) | 0.4013 (5) |
| CellViT-256 | Pancreatic | 0.3249 (7) | 0.1019 (7) | 0.6833 (9) | 0.2739 (6) |
| CellViT-256 | Kidney | 0.5182 (2) | 0.0977 (8) | 0.8638 (1) | 0.1916 (8) |
| CellViT-256 | Skin | 0.3083 (8) | 0.0569 (9) | 0.6935 (8) | 0.1407 (9) |

### Detection–typing separation and class contribution

‘Strong detection / weak typing’ is operationalized only for triage: F1det at or above the median and F1type below the median among complete directions of the same backbone. It is not a biological or statistical threshold.

- CellViT-SAM-H: Kidney A (F1det 0.8478, F1type 0.1492); Kidney B (F1det 0.9008, F1type 0.1060); Liver A (F1det 0.8432, F1type 0.2649); Liver B (F1det 0.8826, F1type 0.4030); Lung A (F1det 0.8430, F1type 0.4042).
- CellViT-256: Kidney A (F1det 0.8339, F1type 0.2104); Kidney B (F1det 0.8938, F1type 0.1727); Liver A (F1det 0.8325, F1type 0.1136); Liver B (F1det 0.8653, F1type 0.3963).
- The complete-result table reports every present class’s F1 and support. Low typing values are usually associated with one or more low-F1 classes rather than uniform failure across all classes; the dominant off-diagonal pair identifies the largest paired confusion by count. Other has low support and often low F1; Kidney additionally shows pronounced direction-specific Immune/Stromal/Epithelial confusion. Skin is the only tissue here with nonzero Melanocyte annotations, and its available rows must be interpreted with that distinct class composition.
- Train→test TVD/JSD are properties of the tissue/fold data and therefore shared by SAM-H and CellViT-256. They should not be duplicated as independent observations across backbones.

### Completion boundary and FullFT-ready schema

The current artifact audit includes 18/18 SAM-H and 18/18 CellViT-256 Selected PEFT directions. No Selected PEFT direction is missing. The frozen `workshop_master_results.csv` predates several CellViT-256 completions, so completion was re-established from each inference log and exact TEST coverage rather than copied from its older status cells.

Matched Kidney/Liver/Tonsil FullFT rows are included through the same checkpoint-10 aggregation and confusion schema. FullFT is not extrapolated to the other six tissues.

## Paper-safe interpretations

- Reciprocal Fold A/B directions differ in class prevalence, train→test distribution distance, and nuclei density; the observed metric changes occur alongside those measured data changes.
- KLT Melanocyte is absent in train and TEST in both folds and is excluded from present-class F1type. Other is present but least prevalent and frequently has low typing F1.
- FullFT has a larger observed B−A mean F1type change than Selected PEFT in the three-seed KLT results. The largest per-class shifts and confusion matrices are reported above; they do not establish why the change occurred.
- Several tissue directions retain comparatively strong detection while typing is weak under the declared within-backbone median rule. Their per-class F1/support and dominant confusions localize the descriptive typing loss.
- Tissue difficulty is direction-dependent: reciprocal slides can change both class composition and metric ranking. Fold directions are not interchangeable replicates.
- These results support slide-held-out wording only. They do not support causal, statistical-significance, patient-level, equivalence, or non-inferiority claims.

## Machine-readable record types

`slide_tissue_diagnostics.csv` is long-form and append-safe. `slide_composition` and `distribution_shift` store data composition; `klt_seed_metric`, `klt_per_class`, `klt_confusion`, `klt_fold_summary`, and `klt_fold_difference` store Part A; `tissue_result`, `tissue_per_class`, `tissue_reciprocal_fold_mean`, and `tissue_missing` store Part B. Every evidence row carries a source path where applicable.
