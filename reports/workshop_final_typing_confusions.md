# Final matched SAM-H tissue typing-confusion analysis

**READ-ONLY DERIVED PAPER REPORT.** Canonical scientific artifacts were not modified.

## Scope and method

This audit uses exactly 18 matched tissue-specific reciprocal complete-slide TEST directions (nine tissues × Fold A/B), CellViT-SAM-H, seed 42. Every source was verified to contain the expected TEST patch count and patch-ID hash, the PEFT and FullFT patch-ID lists were identical within each direction, the saved config used seed 42 and the five-class STHELAR taxonomy, and `inference.log` explicitly loaded `checkpoint_10.pth`.

Each raw matrix contains paired/matched nuclei only; unmatched detections are excluded. Rows are true classes and columns are predicted classes in the order I=Immune, S=Stromal, E=Epithelial, M=Melanocyte, O=Other. Each direction was row-normalized first and then directions were averaged equally, preventing large slides from dominating. A zero-support row is undefined and is omitted from that class-row average rather than treated as an all-zero error row. Thus I/S/E/O use n=18 directions and Melanocyte uses only the two Skin directions (n=2). Per-class F1 follows the same matched-nuclei and nonzero-true-support convention. Results are descriptive; no significance test is performed.

Source matrix: `reports/tissue_peft_vs_fullft_slideind.csv`.

## Macro-averaged row-normalized matrices

### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 0.533 | 0.262 | 0.187 | 0.001 | 0.018 |
| S | 0.333 | 0.520 | 0.133 | 0.001 | 0.013 |
| E | 0.182 | 0.177 | 0.629 | 0.002 | 0.011 |
| M | 0.063 | 0.376 | 0.436 | 0.014 | 0.110 |
| O | 0.396 | 0.319 | 0.231 | 0.002 | 0.053 |

Row coverage: Immune n=18, Stromal n=18, Epithelial n=18, Melanocyte n=2, Other n=18.

### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 0.537 | 0.245 | 0.157 | 0.003 | 0.058 |
| S | 0.296 | 0.535 | 0.113 | 0.002 | 0.054 |
| E | 0.195 | 0.171 | 0.602 | 0.003 | 0.030 |
| M | 0.121 | 0.130 | 0.620 | 0.011 | 0.117 |
| O | 0.331 | 0.305 | 0.163 | 0.004 | 0.198 |

Row coverage: Immune n=18, Stromal n=18, Epithelial n=18, Melanocyte n=2, Other n=18.

### PEFT minus FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | -0.004 | +0.016 | +0.030 | -0.002 | -0.040 |
| S | +0.037 | -0.015 | +0.020 | -0.001 | -0.040 |
| E | -0.013 | +0.006 | +0.027 | -0.001 | -0.019 |
| M | -0.058 | +0.246 | -0.184 | +0.003 | -0.006 |
| O | +0.065 | +0.014 | +0.068 | -0.002 | -0.145 |

## Per-class matched-nuclei F1

These are equal-direction means of the direction-specific class F1 values.

| Class | PEFT | FullFT | PEFT−FullFT | Directions |
|---|---:|---:|---:|---:|
| Immune | 0.460 | 0.496 | -0.036 | 18 |
| Stromal | 0.451 | 0.472 | -0.021 | 18 |
| Epithelial | 0.619 | 0.607 | +0.012 | 18 |
| Melanocyte | 0.018 | 0.020 | -0.002 | 2 |
| Other | 0.022 | 0.120 | -0.098 | 18 |

## Dominant off-diagonal errors

For comparability with the previously reported statement, the dominant error is the unique largest off-diagonal cell by raw matched-nucleus count within each direction. There were no ties. PEFT and FullFT share that pair in **16/18 directions**; the exceptions are Skin A and Skin B.

| True→predicted pair | PEFT directions | FullFT directions |
|---|---:|---:|
| Immune→Stromal | 3 | 3 |
| Immune→Epithelial | 1 | 1 |
| Immune→Melanocyte | 0 | 0 |
| Immune→Other | 0 | 0 |
| Stromal→Immune | 5 | 5 |
| Stromal→Epithelial | 1 | 0 |
| Stromal→Melanocyte | 0 | 0 |
| Stromal→Other | 0 | 0 |
| Epithelial→Immune | 4 | 4 |
| Epithelial→Stromal | 2 | 2 |
| Epithelial→Melanocyte | 0 | 0 |
| Epithelial→Other | 0 | 0 |
| Melanocyte→Immune | 0 | 0 |
| Melanocyte→Stromal | 1 | 0 |
| Melanocyte→Epithelial | 0 | 1 |
| Melanocyte→Other | 0 | 0 |
| Other→Immune | 1 | 1 |
| Other→Stromal | 0 | 0 |
| Other→Epithelial | 0 | 1 |
| Other→Melanocyte | 0 | 0 |

### Direction-level agreement

| Tissue | Fold | PEFT dominant pair (count) | FullFT dominant pair (count) | Same |
|---|:---:|---|---|:---:|
| Breast | A | Immune→Stromal (57,086) | Immune→Stromal (57,186) | yes |
| Breast | B | Stromal→Immune (95,534) | Stromal→Immune (59,332) | yes |
| Colon | A | Epithelial→Immune (56,422) | Epithelial→Immune (79,278) | yes |
| Colon | B | Stromal→Immune (156,281) | Stromal→Immune (191,059) | yes |
| Kidney | A | Immune→Stromal (46,015) | Immune→Stromal (26,532) | yes |
| Kidney | B | Epithelial→Immune (84,882) | Epithelial→Immune (66,123) | yes |
| Liver | A | Epithelial→Stromal (79,410) | Epithelial→Stromal (101,689) | yes |
| Liver | B | Immune→Epithelial (56,418) | Immune→Epithelial (45,786) | yes |
| Lung | A | Epithelial→Immune (61,435) | Epithelial→Immune (40,932) | yes |
| Lung | B | Stromal→Immune (39,038) | Stromal→Immune (27,704) | yes |
| Ovary | A | Stromal→Immune (33,012) | Stromal→Immune (32,675) | yes |
| Ovary | B | Epithelial→Immune (67,466) | Epithelial→Immune (89,970) | yes |
| Pancreatic | A | Other→Immune (46,962) | Other→Immune (47,319) | yes |
| Pancreatic | B | Epithelial→Stromal (46,219) | Epithelial→Stromal (73,976) | yes |
| Skin | A | Melanocyte→Stromal (39,468) | Melanocyte→Epithelial (18,175) | no |
| Skin | B | Stromal→Epithelial (8,991) | Other→Epithelial (12,154) | no |
| Tonsil | A | Immune→Stromal (125,211) | Immune→Stromal (155,272) | yes |
| Tonsil | B | Stromal→Immune (224,338) | Stromal→Immune (207,518) | yes |

## Paper-ready wording

### Main Discussion sentence

Across the 18 matched complete-slide directions, PEFT and FullFT shared the largest matched-nucleus off-diagonal confusion in 16 cases, indicating that their typing errors largely followed the same slide-dependent class ambiguities even when class-wise F1 differed.

### Appendix paragraph

We row-normalized each direction before averaging, so every complete-slide direction contributed equally rather than in proportion to its matched-nucleus count. PEFT and FullFT had the same dominant off-diagonal error in 16 of 18 directions, with Stromal→Immune, Epithelial→Immune, Immune→Stromal, and Epithelial→Stromal recurring most often. Mean matched-nuclei F1 was similar for Epithelial (0.619 PEFT; 0.607 FullFT), while the largest difference occurred for Other (0.022; 0.120); Melanocyte remained weak for both methods and was evaluable only in the two Skin directions. These results show substantial overlap in the principal typing-error modes, while the class-level differences indicate that shared dominant confusions do not imply identical typing behavior.

## Canonical source provenance

| Tissue | Fold | TEST patches | Patch-ID SHA256 | PEFT source | FullFT source |
|---|:---:|---:|---|---|---|
| Breast | A | 49,569 | `6919f9d362da83fdab6869db214f1149625570ce8b9e2d81ed9651cf98f0fe57` | `run/sthelar40x_breast_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T105512_sthelar40x_breast_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_breast_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-09-01T195014_sthelar40x_breast_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Breast | B | 45,773 | `d41bd81521359b3f03b1832ffa33157a6ad1b2fd89819853a91a8a9835dc7d73` | `run/sthelar40x_breast_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T225723_sthelar40x_breast_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_breast_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-09-01T200157_sthelar40x_breast_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Colon | A | 11,903 | `95091217911927e0dbf677070725bf1795fe5191566c82602778576ae497686a` | `run/sthelar40x_colon_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T110313_sthelar40x_colon_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_colon_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-09-01T211611_sthelar40x_colon_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Colon | B | 18,464 | `c2b49d03ad766c3afc07938621d46e2f5c85d2a631a4aab9b7b8b26e804887d3` | `run/sthelar40x_colon_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-27T154219_sthelar40x_colon_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_colon_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-09-02T033103_sthelar40x_colon_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Kidney | A | 4,449 | `b294169ffe32e16f2d9a571f70ab523bd530581e1f64bbc0e475f24dbba448ad` | `run/sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T220941_sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_kidney_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-08-31T202148_sthelar40x_kidney_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Kidney | B | 6,723 | `b20a4b6f38be145511cdb41d76c9e6d2830968b56af08084050986485b1e3cb5` | `run/sthelar40x_kidney_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-29T060654_sthelar40x_kidney_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_kidney_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-08-31T215922_sthelar40x_kidney_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Liver | A | 9,166 | `804074f4d9746fe34711ab8329cd95971876a36acf8f6fa8a4c397d23944bd17` | `run/sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T220941_sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_liver_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-08-31T221828_sthelar40x_liver_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Liver | B | 20,427 | `c190501712ee96da9433ec72d5f100102a64473809a4f9c0f7644314e577f11c` | `run/sthelar40x_liver_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-29T060941_sthelar40x_liver_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_liver_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-08-31T223807_sthelar40x_liver_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Lung | A | 20,621 | `5db06a5c053f1e5e1cf24db539858410b34c8bfc91c24b2ae46ccf93e461091d` | `run/sthelar40x_lung_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T141800_sthelar40x_lung_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_lung_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-09-02T080128_sthelar40x_lung_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Lung | B | 10,938 | `ec3ab11027a24760ba6aaea75d3a6b61bd6113cdc815385289d049041b754bb1` | `run/sthelar40x_lung_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-27T154613_sthelar40x_lung_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_lung_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-09-02T091426_sthelar40x_lung_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Ovary | A | 25,498 | `e501c6aab1225b8b8a96a9e00d89d1b12dc0f10ecd006e9ff4fcd0ce70334617` | `run/sthelar40x_ovary_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T105512_sthelar40x_ovary_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_ovary_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-09-02T100058_sthelar40x_ovary_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Ovary | B | 10,039 | `0792454f7561464ffbbad200e7863074519bf3f66522fb5e97ed0aa2b4dd9189` | `run/sthelar40x_ovary_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-27T154219_sthelar40x_ovary_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_ovary_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-09-02T114048_sthelar40x_ovary_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Pancreatic | A | 12,647 | `6c5b9bfb903b0e1be01f4c8af47025b2de17cfc067b4756773b74d84bbfc7ed4` | `run/sthelar40x_pancreatic_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T105512_sthelar40x_pancreatic_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_pancreatic_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-09-02T135333_sthelar40x_pancreatic_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Pancreatic | B | 25,526 | `131dba6a8406983810271632caff4321d009223660b4eac06350afd1f06466a3` | `run/sthelar40x_pancreatic_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T225723_sthelar40x_pancreatic_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_pancreatic_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-09-02T152702_sthelar40x_pancreatic_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Skin | A | 12,790 | `40d1b93990a1e94d7bc5cc7bca9bf05e0c8537e4aa3fbf7caa58e2bd26163bee` | `run/sthelar40x_skin_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T164216_sthelar40x_skin_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_skin_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-09-02T192405_sthelar40x_skin_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Skin | B | 11,209 | `f3031bff17e940b3fe42adbb4bfbd4f26484bb7227111655332a7632620a51d7` | `run/sthelar40x_skin_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T230626_sthelar40x_skin_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_skin_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-09-02T194550_sthelar40x_skin_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Tonsil | A | 21,083 | `e6c67abea5f71f64b93f484dca1b8271dc1a36351cb2f69ae542e13e9105452b` | `run/sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T192051_sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_tonsil_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-08-31T225424_sthelar40x_tonsil_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/inference_results.json` |
| Tonsil | B | 23,067 | `1504d6aa6fde1cfdc9c35ba08d98d6c14e3a97208b3b0adf12801766c596de3c` | `run/sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T192051_sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/inference_results.json` | `run/sthelar40x_tonsil_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-08-31T233216_sthelar40x_tonsil_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/inference_results.json` |

## Direction-level raw matched-nuclei matrices

Rows are true labels; columns are predictions. These matrices provide the complete reconstruction requested for all 18 directions and both methods.

### Breast — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 64,152 | 57,086 | 33,360 | 0 | 1,270 |
| S | 16,263 | 111,182 | 36,313 | 0 | 646 |
| E | 14,373 | 21,139 | 275,443 | 0 | 654 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 1,638 | 2,710 | 979 | 0 | 123 |

Dominant off-diagonal: Immune→Stromal (57,086).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 67,783 | 57,186 | 23,389 | 0 | 748 |
| S | 16,465 | 113,748 | 21,717 | 0 | 369 |
| E | 17,411 | 27,488 | 243,694 | 0 | 915 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 1,403 | 2,465 | 719 | 0 | 266 |

Dominant off-diagonal: Immune→Stromal (57,186).

### Breast — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 173,749 | 6,454 | 17,345 | 0 | 431 |
| S | 95,534 | 68,193 | 26,447 | 0 | 393 |
| E | 28,740 | 10,605 | 579,807 | 0 | 315 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 6,363 | 1,036 | 3,722 | 0 | 143 |

Dominant off-diagonal: Stromal→Immune (95,534).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 148,903 | 14,208 | 7,844 | 0 | 25,467 |
| S | 59,332 | 95,270 | 15,810 | 0 | 11,967 |
| E | 17,063 | 22,344 | 548,850 | 0 | 27,824 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 3,637 | 1,768 | 1,905 | 0 | 3,632 |

Dominant off-diagonal: Stromal→Immune (59,332).

### Colon — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 87,540 | 21,676 | 39,193 | 0 | 46 |
| S | 11,665 | 7,914 | 4,219 | 0 | 2 |
| E | 56,422 | 3,149 | 301,557 | 0 | 136 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 3,685 | 733 | 3,303 | 0 | 43 |

Dominant off-diagonal: Epithelial→Immune (56,422).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 94,190 | 14,954 | 10,655 | 0 | 29,420 |
| S | 11,207 | 7,373 | 1,139 | 0 | 3,814 |
| E | 79,278 | 2,491 | 236,058 | 0 | 36,287 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 2,330 | 488 | 590 | 0 | 4,670 |

Dominant off-diagonal: Epithelial→Immune (79,278).

### Colon — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 79,306 | 11,663 | 59,077 | 0 | 1 |
| S | 156,281 | 41,623 | 27,457 | 0 | 1 |
| E | 22,333 | 2,307 | 369,133 | 0 | 1 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 4,741 | 214 | 2,746 | 0 | 0 |

Dominant off-diagonal: Stromal→Immune (156,281).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 97,305 | 3,932 | 49,998 | 0 | 1,086 |
| S | 191,059 | 19,917 | 21,553 | 0 | 3,536 |
| E | 27,099 | 678 | 380,986 | 0 | 957 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 5,506 | 84 | 1,616 | 0 | 1,208 |

Dominant off-diagonal: Stromal→Immune (191,059).

### Kidney — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 1,900 | 46,015 | 3,335 | 0 | 4 |
| S | 270 | 19,838 | 877 | 0 | 1 |
| E | 748 | 10,823 | 1,087 | 0 | 2 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 142 | 903 | 34 | 0 | 2 |

Dominant off-diagonal: Immune→Stromal (46,015).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 19,404 | 26,532 | 5,299 | 0 | 304 |
| S | 2,280 | 17,254 | 1,253 | 0 | 131 |
| E | 4,771 | 6,522 | 1,335 | 0 | 317 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 445 | 566 | 29 | 0 | 128 |

Dominant off-diagonal: Immune→Stromal (26,532).

### Kidney — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 13,537 | 709 | 2,009 | 0 | 0 |
| S | 38,269 | 4,345 | 4,414 | 0 | 0 |
| E | 84,882 | 2,956 | 5,074 | 0 | 0 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 4,101 | 600 | 2,028 | 0 | 0 |

Dominant off-diagonal: Epithelial→Immune (84,882).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 5,917 | 2,337 | 6,512 | 0 | 1,494 |
| S | 16,472 | 12,928 | 13,845 | 0 | 3,049 |
| E | 66,123 | 12,051 | 10,880 | 0 | 1,756 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 999 | 834 | 2,859 | 0 | 1,972 |

Dominant off-diagonal: Epithelial→Immune (66,123).

### Liver — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 467 | 50,198 | 2,452 | 0 | 1 |
| S | 96 | 53,619 | 3,035 | 0 | 1 |
| E | 309 | 79,410 | 62,916 | 0 | 3 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 0 | 394 | 25 | 0 | 0 |

Dominant off-diagonal: Epithelial→Stromal (79,410).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 1,064 | 51,650 | 860 | 0 | 466 |
| S | 399 | 55,757 | 971 | 0 | 320 |
| E | 1,961 | 101,689 | 37,169 | 0 | 1,038 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 1 | 415 | 10 | 0 | 28 |

Dominant off-diagonal: Epithelial→Stromal (101,689).

### Liver — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 19,085 | 13,847 | 56,418 | 0 | 0 |
| S | 5,662 | 28,105 | 42,376 | 0 | 0 |
| E | 1,260 | 2,462 | 234,453 | 0 | 0 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 809 | 294 | 1,546 | 0 | 0 |

Dominant off-diagonal: Immune→Epithelial (56,418).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 24,470 | 16,475 | 45,786 | 0 | 517 |
| S | 8,418 | 33,617 | 31,423 | 0 | 606 |
| E | 2,472 | 3,137 | 228,551 | 0 | 113 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 1,294 | 280 | 753 | 0 | 200 |

Dominant off-diagonal: Immune→Epithelial (45,786).

### Lung — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 121,677 | 38,022 | 2,842 | 0 | 33 |
| S | 33,278 | 106,820 | 2,354 | 0 | 17 |
| E | 61,435 | 32,124 | 27,413 | 0 | 36 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 12,267 | 15,153 | 896 | 0 | 21 |

Dominant off-diagonal: Epithelial→Immune (61,435).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 93,101 | 39,783 | 4,784 | 0 | 24,506 |
| S | 17,131 | 98,059 | 4,261 | 0 | 19,992 |
| E | 40,932 | 27,192 | 44,786 | 0 | 6,363 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 4,370 | 11,979 | 797 | 0 | 10,937 |

Dominant off-diagonal: Epithelial→Immune (40,932).

### Lung — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 115,831 | 2,787 | 7,420 | 0 | 4 |
| S | 39,038 | 17,276 | 1,658 | 0 | 19 |
| E | 16,735 | 778 | 35,802 | 0 | 8 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 21,327 | 744 | 1,397 | 0 | 11 |

Dominant off-diagonal: Stromal→Immune (39,038).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 105,406 | 4,051 | 10,968 | 0 | 5,059 |
| S | 27,704 | 21,678 | 1,525 | 0 | 5,617 |
| E | 12,245 | 1,309 | 35,774 | 0 | 1,270 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 15,971 | 618 | 1,750 | 0 | 5,559 |

Dominant off-diagonal: Stromal→Immune (27,704).

### Ovary — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 26,507 | 20,537 | 27,921 | 0 | 5 |
| S | 33,012 | 184,468 | 30,393 | 0 | 10 |
| E | 4,018 | 4,196 | 302,978 | 0 | 2 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 7,989 | 18,833 | 12,893 | 0 | 8 |

Dominant off-diagonal: Stromal→Immune (33,012).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 38,284 | 18,530 | 12,159 | 0 | 5,267 |
| S | 32,675 | 177,462 | 19,044 | 0 | 9,596 |
| E | 15,956 | 5,679 | 285,322 | 0 | 177 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 6,926 | 16,624 | 8,280 | 0 | 7,065 |

Dominant off-diagonal: Stromal→Immune (32,675).

### Ovary — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 34,111 | 6,212 | 42,439 | 0 | 2,676 |
| S | 7,972 | 33,649 | 5,992 | 0 | 1,678 |
| E | 67,466 | 9,872 | 165,101 | 0 | 4,125 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 2,792 | 5,060 | 3,042 | 0 | 2,699 |

Dominant off-diagonal: Epithelial→Immune (67,466).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 42,314 | 5,937 | 34,764 | 0 | 1,666 |
| S | 11,082 | 30,774 | 4,010 | 0 | 1,358 |
| E | 89,970 | 9,999 | 142,955 | 0 | 3,515 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 3,457 | 5,099 | 2,607 | 0 | 1,964 |

Dominant off-diagonal: Epithelial→Immune (89,970).

### Pancreatic — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 96,703 | 6,557 | 125 | 0 | 17 |
| S | 33,742 | 24,665 | 106 | 0 | 33 |
| E | 34,326 | 36,620 | 4,862 | 0 | 761 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 46,962 | 3,069 | 226 | 0 | 39 |

Dominant off-diagonal: Other→Immune (46,962).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 101,214 | 6,434 | 178 | 0 | 364 |
| S | 31,553 | 24,226 | 233 | 0 | 770 |
| E | 33,091 | 25,328 | 11,271 | 0 | 7,497 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 47,319 | 2,555 | 298 | 0 | 792 |

Dominant off-diagonal: Other→Immune (47,319).

### Pancreatic — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 46,905 | 38,879 | 13,363 | 0 | 1,722 |
| S | 6,843 | 34,249 | 7,707 | 0 | 257 |
| E | 6,124 | 46,219 | 96,255 | 0 | 436 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 2,777 | 7,961 | 10,806 | 0 | 124 |

Dominant off-diagonal: Epithelial→Stromal (46,219).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 47,418 | 42,782 | 4,997 | 0 | 1,183 |
| S | 6,890 | 35,680 | 3,248 | 0 | 182 |
| E | 6,392 | 73,976 | 59,754 | 0 | 384 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 3,448 | 9,956 | 6,743 | 0 | 106 |

Dominant off-diagonal: Epithelial→Stromal (73,976).

### Skin — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 23,334 | 24,928 | 1,066 | 11 | 17,425 |
| S | 8,809 | 16,264 | 664 | 7 | 6,116 |
| E | 568 | 2,535 | 4,122 | 1 | 1,393 |
| M | 3,141 | 39,468 | 2,994 | 43 | 12,943 |
| O | 8 | 10 | 1 | 0 | 32 |

Dominant off-diagonal: Melanocyte→Stromal (39,468).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 36,856 | 3,439 | 1,972 | 3,934 | 18,137 |
| S | 13,618 | 5,306 | 1,360 | 1,165 | 8,591 |
| E | 1,371 | 1,139 | 3,721 | 429 | 995 |
| M | 13,168 | 9,878 | 18,175 | 1,260 | 12,939 |
| O | 8 | 3 | 0 | 3 | 33 |

Dominant off-diagonal: Melanocyte→Epithelial (18,175).

### Skin — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 1,344 | 2,148 | 1,693 | 115 | 0 |
| S | 3,915 | 10,710 | 8,991 | 409 | 0 |
| E | 1,033 | 467 | 39,050 | 1,213 | 0 |
| M | 293 | 312 | 3,282 | 109 | 0 |
| O | 3,431 | 5,439 | 7,833 | 512 | 0 |

Dominant off-diagonal: Stromal→Epithelial (8,991).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 234 | 3,576 | 1,836 | 1 | 0 |
| S | 292 | 14,238 | 11,760 | 31 | 0 |
| E | 38 | 135 | 45,947 | 1 | 0 |
| M | 22 | 363 | 4,017 | 0 | 0 |
| O | 318 | 8,657 | 12,154 | 9 | 0 |

Dominant off-diagonal: Other→Epithelial (12,154).

### Tonsil — Fold A

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 702,108 | 125,211 | 94,099 | 0 | 0 |
| S | 87,652 | 137,221 | 36,354 | 0 | 0 |
| E | 6,839 | 2,246 | 131,783 | 0 | 0 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 15,982 | 16,921 | 11,261 | 0 | 0 |

Dominant off-diagonal: Immune→Stromal (125,211).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 698,947 | 155,272 | 60,543 | 0 | 1,700 |
| S | 77,420 | 158,796 | 18,882 | 0 | 2,085 |
| E | 10,562 | 4,153 | 124,916 | 0 | 261 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 10,745 | 23,172 | 5,438 | 0 | 2,239 |

Dominant off-diagonal: Immune→Stromal (155,272).

### Tonsil — Fold B

#### PEFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 1,401,131 | 123,015 | 24,362 | 0 | 2,375 |
| S | 224,338 | 141,249 | 1,835 | 0 | 1,671 |
| E | 68,778 | 51,101 | 155,440 | 0 | 1,781 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 8,800 | 4,712 | 310 | 0 | 1,028 |

Dominant off-diagonal: Stromal→Immune (224,338).

#### FullFT

| True \ Pred. | I | S | E | M | O |
|---|---:|---:|---:|---:|---:|
| I | 1,409,477 | 111,984 | 19,033 | 0 | 9,847 |
| S | 207,518 | 155,858 | 876 | 0 | 3,967 |
| E | 75,350 | 59,085 | 139,761 | 0 | 2,642 |
| M | 0 | 0 | 0 | 0 | 0 |
| O | 8,193 | 4,122 | 298 | 0 | 2,658 |

Dominant off-diagonal: Stromal→Immune (207,518).
