# FINAL READ-ONLY SCIENTIFIC AND MANUSCRIPT AUDIT

Audit date: 2026-09-03. No experiment, scheduler state, configuration, run artifact, checkpoint, adapter, or canonical numerical result was changed. This report is a derived paper-facing audit only.

## Evidence boundary

Primary numerical sources:

- `reports/tissue_peft_vs_fullft_slideind.csv` and the paired canonical `inference_results.json`/`inference.log` paths recorded there;
- `reports/nine_tissue_peft_slideind.csv`;
- `reports/workshop_master_results.csv`;
- `reports/slide_tissue_diagnostics.csv` plus the slide-independent split manifests;
- `reports/workshop_ablation_seed_audit.csv`, supplemented only where explicitly noted by additional canonical checkpoint-10 TEST rows already catalogued in `reports/klt_typing_peft_strategy_analysis.md`;
- `reports/efficiency_audit/paper_ready_efficiency.csv` and the verified serialized-size audit.

The current main manuscript `.tex` is not present in the repository. The accessible material comprised an external appendix attachment, generated paper tables in `reports/workshop_latex_tables.tex`, and older manuscript drafts. Consequently, numerical and structural checks on those accessible sources are complete, but a claim-by-claim, citation-order, and body-reference audit of the unavailable current main `.tex` cannot be certified.

## 1. Nine-tissue matched SAM-H PEFT versus FullFT

### Pairing verification

All 18 pairs pass the following checks:

- CellViT-SAM-H x40, tissue and reciprocal fold agree;
- seed is 42 for both methods;
- PEFT and FullFT configs resolve to the same prepared dataset directory;
- that common directory resolves to one split manifest, so train and validation-side provenance are identical;
- the validation region comes from the training-side slide and not the held-out slide;
- the held-out TEST slide agrees;
- sorted `image_metrics` TEST patch IDs have the same SHA-256 in the two results;
- both inference logs explicitly resolve `checkpoint_10.pth`, and TEST coverage equals the manifest count.

| Tissue | Fold | Train slide | Validation source | TEST slide | TEST patches | TEST-ID SHA-256 prefix |
|---|:---:|---|---|---|---:|---|
| Breast | A | `breast_s0` | `breast_s0` | `breast_s1` | 49,569 | `6919f9d362da` |
| Breast | B | `breast_s1` | `breast_s1` | `breast_s0` | 45,773 | `d41bd8152135` |
| Colon | A | `colon_s1` | `colon_s1` | `colon_s2` | 11,903 | `950912179119` |
| Colon | B | `colon_s2` | `colon_s2` | `colon_s1` | 18,464 | `c2b49d03ad76` |
| Kidney | A | `kidney_s0` | `kidney_s0` | `kidney_s1` | 4,449 | `b294169ffe32` |
| Kidney | B | `kidney_s1` | `kidney_s1` | `kidney_s0` | 6,723 | `b20a4b6f38be` |
| Liver | A | `liver_s0` | `liver_s0` | `liver_s1` | 9,166 | `804074f4d974` |
| Liver | B | `liver_s1` | `liver_s1` | `liver_s0` | 20,427 | `c190501712ee` |
| Lung | A | `lung_s1` | `lung_s1` | `lung_s3` | 20,621 | `5db06a5c053f` |
| Lung | B | `lung_s3` | `lung_s3` | `lung_s1` | 10,938 | `ec3ab11027a2` |
| Ovary | A | `ovary_s0` | `ovary_s0` | `ovary_s1` | 25,498 | `e501c6aab122` |
| Ovary | B | `ovary_s1` | `ovary_s1` | `ovary_s0` | 10,039 | `0792454f7561` |
| Pancreatic | A | `pancreatic_s0` | `pancreatic_s0` | `pancreatic_s1` | 12,647 | `6c5b9bfb903b` |
| Pancreatic | B | `pancreatic_s1` | `pancreatic_s1` | `pancreatic_s0` | 25,526 | `131dba6a8406` |
| Skin | A | `skin_s1` | `skin_s1` | `skin_s2` | 12,790 | `40d1b93990a1` |
| Skin | B | `skin_s2` | `skin_s2` | `skin_s1` | 11,209 | `f3031bff17e9` |
| Tonsil | A | `tonsil_s0` | `tonsil_s0` | `tonsil_s1` | 21,083 | `e6c67abea5f7` |
| Tonsil | B | `tonsil_s1` | `tonsil_s1` | `tonsil_s0` | 23,067 | `1504d6aa6fde` |

These identifiers establish complete-slide holdout, not patient independence. No repository-local slide-to-patient mapping is available.

### Unweighted reciprocal-fold tissue means

Delta is PEFT minus FullFT. Every tissue value is the unweighted arithmetic mean of Fold A and Fold B.

| Tissue | bPQ FullFT | bPQ PEFT | Delta | mPQ FullFT | mPQ PEFT | Delta | F1det FullFT | F1det PEFT | Delta | F1type FullFT | F1type PEFT | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast | 0.404095 | 0.431860 | +0.027766 | 0.252840 | 0.258258 | +0.005418 | 0.802510 | 0.822320 | +0.019811 | 0.551759 | 0.517160 | -0.034599 |
| Colon | 0.282382 | 0.285615 | +0.003233 | 0.125946 | 0.134729 | +0.008784 | 0.748327 | 0.747535 | -0.000792 | 0.420530 | 0.406590 | -0.013940 |
| Kidney | 0.600182 | 0.579474 | -0.020708 | 0.135117 | 0.071632 | -0.063485 | 0.875531 | 0.874315 | -0.001216 | 0.265738 | 0.127615 | -0.138123 |
| Liver | 0.500462 | 0.500980 | +0.000517 | 0.203463 | 0.208030 | +0.004567 | 0.862302 | 0.862901 | +0.000599 | 0.344781 | 0.333961 | -0.010820 |
| Lung | 0.551993 | 0.535977 | -0.016015 | 0.260209 | 0.222403 | -0.037806 | 0.844787 | 0.852452 | +0.007665 | 0.524244 | 0.437147 | -0.087098 |
| Ovary | 0.419461 | 0.418095 | -0.001366 | 0.199849 | 0.188736 | -0.011113 | 0.826911 | 0.833601 | +0.006690 | 0.524883 | 0.493001 | -0.031881 |
| Pancreatic | 0.387392 | 0.383303 | -0.004089 | 0.143419 | 0.149232 | +0.005814 | 0.721764 | 0.729409 | +0.007645 | 0.353384 | 0.347744 | -0.005640 |
| Skin | 0.336845 | 0.326879 | -0.009966 | 0.103679 | 0.102916 | -0.000763 | 0.730861 | 0.702964 | -0.027897 | 0.241139 | 0.268050 | +0.026911 |
| Tonsil | 0.523453 | 0.514652 | -0.008801 | 0.255608 | 0.238071 | -0.017537 | 0.841556 | 0.847126 | +0.005571 | 0.532691 | 0.499315 | -0.033376 |

### Across nine tissue-level means

| Metric | FullFT mean | PEFT mean | Signed delta | Absolute difference of means | Median tissue delta | Tissue-delta range | PEFT higher |
|---|---:|---:|---:|---:|---:|---:|---:|
| bPQ | 0.445141 | 0.441871 | -0.003270 | 0.003270 | -0.004089 | [-0.020708, +0.027766] | 3/9 |
| mPQ | 0.186681 | 0.174890 | -0.011791 | 0.011791 | -0.000763 | [-0.063485, +0.008784] | 4/9 |
| F1det | 0.806061 | 0.808069 | +0.002008 | 0.002008 | +0.005571 | [-0.027897, +0.019811] | 6/9 |
| F1type | 0.417683 | 0.381176 | -0.036507 | 0.036507 | -0.031881 | [-0.138123, +0.026911] | 1/9 |

The ratio of the PEFT and FullFT grand tissue means for mPQ is 0.936837, i.e. 93.68% recovery. Mean absolute tissue-level differences are 0.010273 bPQ, 0.017254 mPQ, 0.008654 F1det, and 0.042487 F1type.

### Across 18 reciprocal directions

These are descriptive direction-level summaries only. The 18 rows are not independent biological or stochastic replicates.

| Metric | FullFT mean | PEFT mean | Signed delta | Mean absolute delta | Median absolute delta | Absolute-delta range | Signed-delta range | PEFT higher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bPQ | 0.445141 | 0.441871 | -0.003270 | 0.018344 | 0.012794 | [0.000739, 0.069168] | [-0.042154, +0.069168] | 5/18 |
| mPQ | 0.186681 | 0.174890 | -0.011791 | 0.024472 | 0.023921 | [0.002266, 0.091049] | [-0.091049, +0.035510] | 6/18 |
| F1det | 0.806061 | 0.808069 | +0.002008 | 0.010969 | 0.008618 | [0.000152, 0.046151] | [-0.046151, +0.030963] | 11/18 |
| F1type | 0.417683 | 0.381176 | -0.036507 | 0.054390 | 0.043568 | [0.013313, 0.157854] | [-0.157854, +0.042390] | 5/18 |

## 2. Detection versus typing

The aggregate finding holds descriptively. The mean absolute F1type difference across directions (0.0544) is almost five times the F1det difference (0.0110), and the mPQ difference is intermediate (0.0245). The medians are 0.0436, 0.0086, and 0.0239 respectively. F1det differs by at most 0.0462, whereas F1type spans differences up to 0.1579.

Using the explicit descriptive rule F1det >= 0.80 and F1type < 0.30, high detection with weak typing occurs for PEFT in Kidney A/B and Liver A, and for FullFT in Kidney B and Liver A. At the tissue level, Kidney is the clearest separation: F1det is 0.874 PEFT/0.876 FullFT while F1type is 0.128/0.266.

PEFT exceeds FullFT in direction-level mPQ for Breast A, Colon A, Liver A, Ovary B, Pancreatic B, and Skin A. It exceeds FullFT in F1type for Liver A, Ovary B, Pancreatic B, and Skin A/B. At the tissue-mean level, PEFT is higher in mPQ for Breast, Colon, Liver, and Pancreatic, but only Skin is higher in F1type.

FullFT's clearest advantages occur in Kidney A (mPQ +0.0910 and F1type +0.1579 over PEFT), Kidney B (F1type +0.1184), and Lung A/B (mPQ +0.0341/+0.0415 and F1type +0.0869/+0.0873). These examples agree with, rather than replace, the aggregate result.

## 3. Slide-shift diagnostic synthesis

TVD/JSD are properties of the common tissue/fold data and are not duplicated as observations across methods. Descriptive rank correlations between TVD and F1type are -0.439 for PEFT and -0.451 for FullFT; the corresponding JSD correlations are -0.478 and -0.503. This is compatible with a broad inverse association, but it is neither uniform nor causal. TVD/JSD have essentially no descriptive rank relation to the absolute PEFT-FullFT F1type gap (-0.018/-0.104).

Paper-safe conclusions, limited to five:

1. Larger class-frequency shift coincides with weaker typing in several directions, most clearly Kidney, but does not fully explain performance.
2. High shift does not uniformly favor FullFT: Skin A/B have the largest TVD/JSD (0.678/0.473 and 0.699/0.492), yet PEFT and FullFT have nearly equal mPQ and PEFT has higher F1type in both directions. This disproves a simple “high shift implies a PEFT typing deficit” explanation.
3. Low shift does not guarantee stable typing: Liver A/B have the smallest TVD/JSD (0.029/0.002 and 0.032/0.003), but the F1type delta changes from +0.042 for PEFT to -0.064. Tonsil also has low shift while retaining a PEFT F1type deficit in both directions.
4. Nuclei-density change does not track the between-method gap monotonically. Large reciprocal density changes in Tonsil and Pancreatic coexist with small, mixed-sign deltas; the rank relation between absolute density change and absolute F1type delta is approximately 0.04.
5. Typing errors concentrate in recurring class confusions. PEFT and FullFT share the same dominant off-diagonal pair in 16/18 directions; Stromal->Immune, Epithelial->Immune, Immune->Stromal, and Epithelial->Stromal recur most often. Averaged over directions, the per-class PEFT-FullFT F1 difference is largest for Other (-0.098), followed by Immune (-0.036) and Stromal (-0.021), while Epithelial is +0.012. Rare classes contribute in some directions, but rarity is not sufficient: the largest loss occurs in Kidney A despite no class being absent or <1% in training, while rare Other in Liver B and Skin B accompanies opposite-signed F1type outcomes.

## 4. Historical within-slide ablation selection audit

The table below is a canonical checkpoint-10 TEST superset of the paper ablation rows visible in the accessible July manuscript and the current workshop ablation audit. F1type is recomputed consistently as present-class macro-F1 on paired nuclei. A bare value means one run; no uncertainty is imputed.

| Method | Exact trainable components | n | mPQ | bPQ | F1det | F1type | Trainable % | Decoder body modified? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Frozen | none | 1 | -- | 0.446763 | 0.816732 | -- | 0 | No; taxonomy incompatible for STHELAR metrics |
| Final-head LP | final NP/HV/NT heads | 1 | 0.204322 | 0.445691 | 0.818139 | 0.429935 | 0.000093 | No |
| LoRA | encoder attention Q/V, r=8, alpha=8 | 1 | 0.112672 | 0.430256 | 0.830822 | 0.280017 | 0.1870 | No |
| AdaptFormer | encoder MLP bottlenecks, reduction 16 | 1 | 0.118525 | 0.418378 | 0.805933 | 0.296781 | 0.9340 | No |
| VeRA | encoder attention Q/V scaling, r=16 | 1 | 0.064683 | 0.456896 | 0.827227 | 0.181212 | 0.0119 | No |
| Decoder conv adapters | decoder residual conv adapters + final heads/classifier | 1 | 0.231714 | 0.458723 | 0.828193 | 0.561672 | 0.0790 | Yes, by inserted adapters; original conv weights frozen |
| AdaptFormer + heads | AdaptFormer + final NP/HV/NT heads/classifier | 2 | 0.287710 +/- 0.003574 | 0.498686 +/- 0.009294 | 0.832820 +/- 0.002372 | 0.596963 +/- 0.028204 | 0.9341 | No |
| VeRA + AdaptFormer + heads | VeRA + AdaptFormer + final heads/classifier | 1 | 0.287031 | 0.493231 | 0.832649 | 0.579022 | 0.9458 | No |
| Selected LoRA + AdaptFormer + heads | LoRA Q/V r=8/alpha=8 + AdaptFormer reduction 16/GELU + final NP/HV/NT heads/classifier | 2 | 0.293647 +/- 0.004668 | 0.504263 +/- 0.004687 | 0.834541 +/- 0.001353 | 0.578286 +/- 0.024103 | 1.1176 | No |
| LoRA + AdaptFormer + last stage | selected encoder adapters + trainable late decoder stage + final heads/classifier | 3 | 0.295497 +/- 0.002051 | 0.503711 +/- 0.004254 | 0.824539 +/- 0.003092 | 0.649572 +/- 0.008351 | 1.1674 | Yes; late original decoder stage trained |
| LoRA + AdaptFormer + conv adapters | selected encoder adapters + decoder residual conv adapters + final heads/classifier | 2 | 0.288389 +/- 0.002882 | 0.493081 +/- 0.005789 | 0.829267 +/- 0.000418 | 0.651441 +/- 0.002679 | 1.1947 | Yes, by inserted adapters; original conv weights frozen |
| VeRA + AdaptFormer + conv adapters | VeRA + AdaptFormer + decoder residual conv adapters + final heads/classifier | 2 | 0.284855 +/- 0.001254 | 0.491741 +/- 0.000096 | 0.828732 +/- 0.005746 | 0.652165 +/- 0.012171 | 1.0231 | Yes, by inserted adapters; original conv weights frozen |
| FullFT | complete encoder, decoder, and classifier | 2 | 0.309216 +/- 0.004826 | 0.514735 +/- 0.008973 | 0.828604 +/- 0.002152 | 0.666112 +/- 0.002794 | 100 | Yes; all base parameters trained |

The third last-stage run is seed44. Its inference log explicitly loads checkpoint 10, its TEST set has the same 4,653 patch IDs and SHA-256 as seeds42/43, and its data/config differ only in the intended stochastic seed. Therefore `reports/workshop_ablation_seed_audit.md` and `reports/workshop_latex_tables.tex`, which currently report n=2 for this row, are stale on replication depth and summary values.

Answers to the selection questions:

- **A. Best on any PEFT metric?** Yes. Within the audited paper rows, selected heads-only has the highest PEFT bPQ (0.504263) and F1det (0.834541). It is not best on mPQ or F1type.
- **B. Near-best in mPQ?** Yes. It is second among the audited PEFT rows: 0.293647 versus 0.295497 for last-stage, a difference of 0.001850.
- **C. Empirical selection reason?** It combines the best audited PEFT bPQ and detection F1 with near-best mPQ, only 1.1176% trainable parameters, a frozen decoder body, and a verified compact modular state. The choice was frozen before the complete-slide campaign.
- **D. Misleading wording?** “Best PEFT configuration,” “best typing configuration,” or “optimal adapter” would be misleading. Last-stage is numerically higher in mPQ, and last-stage/conv variants are substantially higher in F1type. Use “selected frozen-decoder configuration” or “deployment-oriented selected configuration.”

Additional ablation consistency warnings:

- `reports/paper_tables/klt_ablation_f1_type_table.tex` imputes `+/-` values for single-seed rows. Those are not sample SDs and are not paper-safe.
- The older manuscript table's LoRA/AdaptFormer/VeRA F1type values 0.224/0.237/0.145 do not use the same present-class aggregation as the combined rows. Consistent canonical present-class values are 0.280/0.297/0.181.
- Frozen mPQ/F1type must remain blank because the pretrained PanNuke taxonomy is incompatible with STHELAR typing.

## 5. CellViT-256 interpretation and computational audit

### Predictive evidence

In three-seed KLT, CellViT-256 retains detection relatively well but loses more class-aware performance than SAM-H. For selected PEFT, SAM-H versus CellViT-256 is:

| Fold | Backbone | bPQ | mPQ | F1det | F1type |
|:---:|---|---:|---:|---:|---:|
| A | SAM-H | 0.518942 | 0.219499 | 0.840557 | 0.493738 |
| A | CellViT-256 | 0.469383 | 0.183533 | 0.833232 | 0.474964 |
| B | SAM-H | 0.609713 | 0.236494 | 0.865418 | 0.489520 |
| B | CellViT-256 | 0.564433 | 0.190065 | 0.861685 | 0.466681 |

Across all nine tissue-specific PEFT tissues and directions, SAM-H/CellViT-256 means are bPQ 0.441871/0.420214, mPQ 0.174890/0.150992, F1det 0.808069/0.791081, and F1type 0.381176/0.348425.

For the six matched CellViT-256 Kidney/Liver/Tonsil tissue-fold controls, PEFT minus FullFT is -0.016050 bPQ, -0.012188 mPQ, -0.000704 F1det, and -0.041583 F1type on average. Thus detection remains nearly matched, while type-aware metrics show a clearer gap; individual directions remain mixed.

### Canonical matched seed42 A100 cost measurements

| Backbone | Method | Train h | Peak allocated VRAM GiB | Verified adapter MiB | N=9 deployment GiB |
|---|---|---:|---:|---:|---:|
| SAM-H | Selected PEFT | 7.398 | 9.362 | 30.369 | 2.874 |
| SAM-H | FullFT | 7.986 | 15.508 | -- | 23.463 |
| CellViT-256 | Selected PEFT | 6.044 | 2.040 | 1.548 | 0.188 |
| CellViT-256 | FullFT | 5.577 | 2.667 | -- | 1.569 |

The canonical training times are therefore SAM-H PEFT/FullFT = 7.398/7.986 h and CellViT-256 PEFT/FullFT = 6.044/5.577 h. Accessible generated tables preserve this ordering; no swap was found. CellViT-256 PEFT is 8.37% slower than its FullFT control despite training 0.7942% of parameters, whereas SAM-H PEFT is 7.36% faster and uses 39.63% less allocated training VRAM than SAM-H FullFT. Parameter efficiency must not be equated with wall-clock efficiency.

Paper-safe sentence: **PEFT is most attractive at SAM-H scale, where it preserves or improves KLT detection and closely tracks class-aware performance while reducing memory and multi-domain state, whereas on CellViT-256 the smaller absolute resource base leaves less benefit and PEFT shows clearer bPQ/mPQ/type gaps without a wall-clock saving.**

## 6. Qualitative figure audit

`figures/qualitative_frozen_fullft_peft.png` does not exist. The filename therefore does not describe any auditable current asset. The existing candidates `figures/qualitative_grid_3x6_main.{png,pdf}` and `figures/qualitative_grid_3x9_supplement.{png,pdf}` contain:

1. Ground truth;
2. Frozen CellViT prediction;
3. tissue-specific selected PEFT prediction.

They do **not** contain LP or FullFT rows. The backbone is CellViT-SAM-H x40. Frozen uses seed42; PEFT uses seed42 for Breast and seed43 for the other displayed tissues. These are historical within-slide spatial TEST examples, have no reciprocal complete-slide Fold A/B designation, and were reconstructed from `model_best.pth`, not canonical checkpoint 10. They are therefore not legitimate held-out-slide examples for the current experiment.

| Figure column | PEFT seed | Patch ID | Slide | Protocol status |
|---|---:|---|---|---|
| Kidney (supplement only) | 43 | `kidney_s0__x8832_y25152__kidney_s0_8168.png` | `kidney_s0` | within-slide TEST region |
| Liver | 43 | `liver_s1__x13056_y13632__liver_s1_5748.png` | `liver_s1` | within-slide TEST region |
| Tonsil | 43 | `tonsil_s0__x27456_y37632__tonsil_s0_32287.png` | `tonsil_s0` | within-slide TEST region |
| Ovary | 43 | `ovary_s1__x26880_y22080__ovary_s1_18655.png` | `ovary_s1` | within-slide TEST region |
| Breast | 42 | `breast_s1__x43584_y71808__breast_s1_100459.png` | `breast_s1` | within-slide TEST region |
| Colon | 43 | `colon_s2__x18048_y26304__colon_s2_15438.png` | `colon_s2` | within-slide TEST region |
| Lung (supplement only) | 43 | `lung_s1__x13248_y12096__lung_s1_5172.png` | `lung_s1` | within-slide TEST region |
| Pancreatic (supplement only) | 43 | `pancreatic_s0__x16896_y55296__pancreatic_s0_30904.png` | `pancreatic_s0` | within-slide TEST region |
| Skin | 43 | `skin_s2__x19392_y1152__skin_s2_797.png` | `skin_s2` | within-slide TEST region |

The selection procedure is explicitly favorable to PEFT: it first ranks PEFT TEST patches by mPQ/bPQ/Jaccard and image-content constraints, then chooses among shortlisted patches using PEFT-versus-Frozen matched-nucleus type-accuracy improvement. The examples are illustrative and not representative.

The sentence “differences are most apparent in cell-type assignment rather than nuclei localization” is consistent with the aggregate quantitative analysis, but the current figure alone cannot establish it because patches were selected on typing improvement. It is defensible only as a non-quantitative visual description accompanied by an explicit selection disclaimer. Any caption claiming LP, FullFT, checkpoint-10, or held-out-slide content for these existing grids is incorrect.

## 7. Manuscript consistency findings

### Passed on accessible material

- No `XX.X`/`X.XXX` placeholders were found in the accessible appendix or generated workshop tables.
- The final tissue Table 3 values below equal unweighted Fold A/B means; its Mean row is exactly the arithmetic mean of the nine tissue means (equivalently, because every tissue has two folds, the 18 direction values).
- The computational-table training times are in the correct PEFT/FullFT order.
- Exact `1.1176%` in technical tables and rounded `1.12%` in prose are consistent precision choices.
- The appendix correctly states that folds are reciprocal directions, not stochastic or biological replicates, and does not claim patient independence.
- The appendix correctly identifies the within-slide KLT benchmark as a selection benchmark rather than complete-slide evidence.
- Evaluation adapters and future both-slide deployment adapters are explicitly distinguished.
- All labels referenced locally inside the accessible appendix resolve inside that appendix.

### Must be corrected in the manuscript source before submission

- The accessible appendix still contains stale LP `\NA` rows and the comment “Replace LP rows after seed44 completion.” LP is now 3/3 for both backbones and folds in `reports/workshop_latex_tables.tex`.
- The accessible appendix tissue table still contains `\NA` for Breast, Colon, Lung, Ovary, Pancreatic, and Skin FullFT/PEFT rows. Replace it with the completed fold-specific table generated from the canonical matched CSV.
- The appendix comment requesting later CellViT-256 and K/L/T insertion is stale.
- `reports/workshop_latex_tables.tex` says LP is n=2/3 because seed44 is running, while its own rows and canonical master show 3/3. The caption is stale.
- `reports/slide_tissue_diagnostics.md` says SAM-H FullFT exists only for Kidney/Liver/Tonsil; that completion boundary predates the final six-tissue FullFT block. Use it for composition diagnostics, not final FullFT coverage.
- The appendix adapter sizes `30.34` and `1.54` MiB are inconsistent with measured 30.369 and 1.548 MiB and are not correct two-decimal rounding. Use either 30.369/1.548 everywhere or 30.37/1.55 everywhere.
- The historical ablation last-stage row is stale at n=2; three comparable canonical seeds exist. Do not use the imputed single-seed uncertainties in `reports/paper_tables/klt_ablation_f1_type_table.tex`.
- The requested qualitative filename is missing, and existing qualitative grids do not match the presumed filename/caption/protocol.

### Not certifiable without the current main `.tex`

- exhaustive comparison of every prose number and table against canonical sources;
- confirmation that every main-text table and figure is cited;
- resolution of all main-text appendix references;
- resolution and first-citation ordering of all bibliography keys;
- global searches for unsupported superiority/equivalence language.

The accessible appendix cites `hovernet`, `cellvit`, `sam`, and `metricsreloaded`; no bibliography accompanies that attachment, so key resolution depends on the unavailable main source. The older July sanity audit found its then-current manual bibliography order valid, but that does not certify the final manuscript.

## Final paper-ready nine-tissue Table 3

Bold is decided from unrounded values; Colon F1det rounds to a displayed tie, but PEFT is slightly lower and is therefore not bold.

```latex
\begin{table}[t]
\caption{Tissue-specific \SAMH{} results under reciprocal complete-slide holdout (seed 42), averaged equally over Fold A/B. Fold-specific values are reported in Appendix~\ref{app:tissue}. Bold indicates higher PEFT performance.}
\label{tab:tissue}
\centering
\fontsize{8}{9}\selectfont
\begin{tabular}{lcccccc}
\toprule
Tissue &
\multicolumn{2}{c}{\mPQ{}} &
\multicolumn{2}{c}{\Fdet{}} &
\multicolumn{2}{c}{\Ftype{}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
& \FullFT{} & PEFT & \FullFT{} & PEFT & \FullFT{} & PEFT \\
\midrule
Breast     & .253 & \textbf{.258} & .803 & \textbf{.822} & .552 & .517 \\
Colon      & .126 & \textbf{.135} & .748 & .748           & .421 & .407 \\
Kidney     & .135 & .072           & .876 & .874           & .266 & .128 \\
Liver      & .203 & \textbf{.208} & .862 & \textbf{.863} & .345 & .334 \\
Lung       & .260 & .222           & .845 & \textbf{.852} & .524 & .437 \\
Ovary      & .200 & .189           & .827 & \textbf{.834} & .525 & .493 \\
Pancreatic & .143 & \textbf{.149} & .722 & \textbf{.729} & .353 & .348 \\
Skin       & .104 & .103           & .731 & .703           & .241 & \textbf{.268} \\
Tonsil     & .256 & .238           & .842 & \textbf{.847} & .533 & .499 \\
\midrule
Mean       & .187 & .175           & .806 & \textbf{.808} & .418 & .381 \\
\bottomrule
\end{tabular}
\end{table}
```

## A. STRONGEST SUPPORTED CLAIMS

1. In 18 seed42 SAM-H comparisons matched on tissue, reciprocal fold, train/validation-side data, held-out slide, and exact TEST patch IDs, PEFT and FullFT have nearly identical mean detection F1 (0.808 versus 0.806), while FullFT has higher mean mPQ (0.187 versus 0.175) and F1type (0.418 versus 0.381).
2. PEFT has higher F1det in 6/9 tissue means and 11/18 directions, but higher mPQ in only 4/9 tissues and F1type in 1/9; this supports robust detection preservation, not overall PEFT superiority.
3. Direction-level PEFT-FullFT differences are markedly wider for F1type than F1det (mean absolute 0.054 versus 0.011), locating the main adaptation gap in typing rather than nuclei localization.
4. Slide class-frequency shift is descriptively associated with typing difficulty, but high TVD/JSD neither guarantees poor relative PEFT performance nor fully explains the observed gaps.
5. The selected historical PEFT configuration is a frozen-decoder design choice with the best audited PEFT bPQ/F1det and near-best mPQ, not the best typing configuration.
6. Backbone scale matters: CellViT-256 preserves detection reasonably but shows larger PEFT gaps in bPQ/mPQ/type and no PEFT wall-time saving, whereas SAM-H obtains the stronger predictive/resource trade-off.
7. Verified adapters reduce trainable parameters and multi-domain deployment state substantially: at N=9, measured deployment storage is 2.874 versus 23.463 GiB for SAM-H PEFT versus FullFT, and 0.188 versus 1.569 GiB for CellViT-256.

## B. CLAIMS TO AVOID

- PEFT is statistically superior, equivalent, non-inferior, or indistinguishable from FullFT.
- PEFT is the best method or the selected heads-only configuration is the best PEFT configuration on all metrics.
- PEFT matches FullFT typing across tissues; the aggregate F1type gap points in the opposite direction.
- The nine tissue-fold rows are independent biological replicates, or patches are biological replicates.
- Fold A/B are random seeds; they are reciprocal complete-slide directions.
- Nine-tissue robustness across optimization seeds; the tissue-specific matched block uses seed42 only.
- Patient-independent, specimen-independent, donor-independent, or site-independent validation; only slide identifiers and complete-slide holdout are established.
- Within-slide spatial splitting is leakage-safe at slide level or constitutes slide-independent validation.
- High TVD/JSD causes typing failure, low TVD/JSD guarantees stable typing, or density shift causes the PEFT-FullFT gap.
- Rare or missing training classes alone explain the typing losses.
- Fewer trainable parameters imply proportional training-time or throughput savings; CellViT-256 is a direct counterexample.
- Frozen class-aware STHELAR performance, because its taxonomy is incompatible.
- The existing qualitative grids are held-out-slide, checkpoint-10, LP, or FullFT evidence, or are a representative sample.
- The verified evaluation adapters and future both-slide deployment adapters are interchangeable evidence sources.

## C. RECOMMENDED DISCUSSION

Across complete-slide holdout, PEFT is most useful when adaptation state must remain modular: on SAM-H it preserves detection almost exactly while reducing trainable parameters and multi-domain storage, although FullFT retains an average advantage in class-aware mPQ and typing. The smaller CellViT-256 backbone keeps detection competitive but shows larger bPQ, mPQ, and typing losses under PEFT, while also providing no wall-clock advantage over FullFT, indicating that the practical trade-off depends on backbone scale. Across tissues, PEFT-FullFT detection differences are small, whereas typing differences are broader and direction-dependent. Class-frequency divergence coincides descriptively with some difficult directions, but low divergence does not guarantee stable typing and high divergence does not uniformly favor FullFT. Recurrent immune-stromal and epithelial confusions, together with weak Other-class F1, further localize the gap without establishing causality. Future work should evaluate typing-aware adapters across independently mapped patients and sites.
