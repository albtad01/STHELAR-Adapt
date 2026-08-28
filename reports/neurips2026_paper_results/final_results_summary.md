# Final STHELAR-Adapt paper-results summary

This package uses completed, fixed-final-epoch (`checkpoint_10.pth`) test outputs only. Validation metrics are excluded. Fold A and Fold B are reported separately; seed means/SDs are calculated inside each fold, followed by the mean of the two fold means. Folds and seeds are never pooled as interchangeable replicates. `F1type` follows the existing paper definition: unweighted macro-F1 over foreground classes with nonzero matched true support, computed on matched nuclei only (four KLT classes; Melanocyte is excluded because its matched true support is zero). Per-class support columns distinguish matched support from matched+unmatched total true support. No statistical significance is claimed.

## A. Facts safe to state strongly

- The new KLT evaluation is slide-independent: Fold A holds out all s1 slides and Fold B holds out all s0 slides for kidney, liver and tonsil.
- Frozen results are strictly class-agnostic. Dice, Jaccard, bPQ and F1 detection are valid; mPQ, matched-only F1 type and per-class values are `NA`.
- SAM-H Selected PEFT trains 7,908,779/707,644,395 parameters (1.1176%); CellViT-256 Selected PEFT trains 374,198/47,116,967 (0.7942%).
- On matched seed-42 A100 runs, SAM-H PEFT uses less training allocated VRAM than SAM-H FullFT (9.362 vs 15.508 GiB) and a 31,844,024-byte verified adapter instead of a complete model.
- CellViT-256 PEFT uses less training allocated VRAM than CellViT-256 FullFT (2.040 vs 2.667 GiB), with a 1,623,604-byte verified adapter.
- Adapter sizes above are genuine serialized deployable state dictionaries and are distinct from training checkpoint and complete inference-model sizes. Base+adapter loading was previously verified against each canonical checkpoint with exact state equality and deterministic forward equality.

## B. Descriptive findings requiring cautious wording

- SAM-H Fold A/B asymmetry remains visible: seed-42 Selected PEFT/FullFT mean bPQ is 0.521 in Fold A and 0.608 in Fold B. This is a held-out-slide composition effect as well as a possible model effect.
- Across the four seed-42 PEFT/FullFT backbone-method combinations, per-slide mean bPQ is Kidney 0.589, Liver 0.583, and Tonsil 0.498. Tonsil is descriptively the lowest of the three tissues in this aggregation.
- NT-header1 changes the matched-only typing balance descriptively: the mean of the two within-fold seed means is Epithelial F1 0.615 vs 0.680 for Selected PEFT, and Other F1 0.149 vs 0.005. This is a trade-off description, not an inferential claim.
- Other-class F1 is low relative to the stronger foreground classes in several fold/seed matrices; inspect `per_class_metrics.csv` and the preserved confusion matrices when wording this point.
- Within-slide versus slide-independent deltas are descriptive only because test composition changes from 4,653 spatial patches sampled from all six slides to complete held-out slides (23,494/26,335 patches by fold).

## C. Claims not supported by these data

- No claim of statistical significance, equivalence, non-inferiority, or superiority.
- No claim that slide independence improves a model, even where class-agnostic metrics increase; the test sets differ in both slide identity and patch composition.
- No interpretation of validation metrics and no assertion that epoch 10 is optimal.
- No use of folds as independent seed replicates or of patches as independent samples for significance testing.
- No claim that Frozen has taxonomy-dependent performance.
- No claim that parameter efficiency guarantees wall-clock efficiency. CellViT-256 PEFT is slower than CellViT-256 FullFT on the matched runs (6.044 vs 5.577 h), despite training only 0.7942% of its parameters. SAM-H PEFT is faster than FullFT (7.398 vs 7.986 h), but the wall-time reduction is much smaller than the parameter reduction.
- No claim that the currently public adapters reproduce the new slide-independent results; the public Hugging Face packages are from the earlier within-slide release.

## D. Exact suggested numerical statements

1. “Across the two slide-independent folds at seed 42, SAM-H Selected PEFT obtained Dice 0.814, bPQ 0.565, mPQ 0.224, F1 detection 0.853, and matched-only present-class macro-F1 type 0.481 (mean of Fold A/B values).”
2. “The corresponding SAM-H FullFT values were Dice 0.815, bPQ 0.564, mPQ 0.203, F1 detection 0.847, and matched-only present-class macro-F1 type 0.491.”
3. “SAM-H Selected PEFT trained 1.118% of parameters and reduced peak allocated training memory by 6.146 GiB (39.6%) relative to FullFT; wall time decreased by 0.588 h (7.4%).”
4. “Across matched seed-42 folds, CellViT-256 Selected PEFT obtained bPQ 0.518 versus 0.554 for CellViT-256 FullFT, while using 2.040 versus 2.667 GiB peak allocated training memory.”
5. “CellViT-256 PEFT did not provide wall-clock savings: its mean training time was 6.044 h versus 5.577 h for FullFT (+8.4%).”
6. “The verified adapter-only artifacts occupied 31,844,024 bytes (30.369 MiB) for SAM-H and 1,623,604 bytes (1.548 MiB) for CellViT-256; these values do not include the shared pretrained base.”

## File guide

- `main_samh_slideind.csv`: all valid SAM-H seeds/folds plus within-fold seed mean/SD and mean of fold means.
- `cellvit256_slideind.csv`: CellViT-256 seed-42 Fold A/B, with Frozen taxonomy fields as `NA`.
- `matched_backbone_comparison.csv`: exactly matched seed-42 Fold A/B test and pre-audited A100 efficiency values.
- `within_vs_slideind.csv`: seed-matched protocol values and deltas with composition warnings.
- `per_slide_metrics.csv`, `per_class_metrics.csv`: descriptive held-out-slide and matched-nuclei typing audit.
- `confusion_matrices/`: raw, row-normalized, and PNG individual matrices plus explicitly descriptive aggregates.

## Audit reconciliation — 2026-08-27

This package remains the canonical result source for the revised KLT analysis.
The experiment registry was reconciled against the run directories, final JSONs,
retained checkpoints, efficiency records, fold manifests, and Slurm accounting.
No validation value was substituted for a test value.

| Experiment family | Current scientific status | Seeds/directions | Paper role |
|---|---|---|---|
| Historical within-slide KLT PEFT selection | COMPLETED VALID for the reported supervised methods; historical Frozen is valid only for class-agnostic quantities | Mostly seeds 42/43; individual exploratory scopes vary | HISTORICAL ONLY / APPENDIX; establishes pre-selection of LoRA(Q,V)+AdaptFormer+final heads |
| Historical nine-tissue within-slide Selected PEFT/FullFT | COMPLETED VALID, with explicitly heterogeneous PEFT seed/checkpoint provenance | FullFT seed 42 throughout; PEFT mostly seed 42, Kidney also seed 43, Pancreatic/Tonsil seed 43 | APPENDIX / baseline for tissue-shift deltas |
| SAM-H held-out-slide KLT | COMPLETED VALID | Frozen deterministic; LP, Selected PEFT, NT-header1 and FullFT seeds 42/43, reciprocal A/B | MAIN PAPER |
| CellViT-256 held-out-slide KLT | COMPLETED VALID | Frozen deterministic; LP, Selected PEFT and FullFT seed 42, reciprocal A/B | SUPPORTING ANALYSIS / backbone-scale context |
| Matched A100 efficiency and deployment storage | COMPLETED VALID | Seed-42 A/B matched measurements | MAIN PAPER / SUPPORTING ANALYSIS |
| SAM-H typing/error analysis | COMPLETED VALID from canonical test outputs | Selected PEFT, NT-header1 and FullFT seeds 42/43, reciprocal A/B | MAIN PAPER / APPENDIX |
| Nine-tissue specialist held-out-slide Fold A | COMPLETED VALID for all nine tissues | Selected PEFT seed 42, one direction per tissue | MAIN PAPER / SUPPORTING ANALYSIS |
| Reciprocal tissue-specific Fold B | PARTIAL, supplementary rather than required by the nine-tissue Fold-A design | Breast, Pancreatic, Skin, Tonsil, Ovary, Colon and Lung complete; Kidney/Liver missing after infrastructure failures | SUPPORTING ANALYSIS |

The historical nine-tissue within-slide Selected PEFT/FullFT table is complete,
but it is not a perfectly matched seed/checkpoint experiment:

| Tissue | FullFT seed/checkpoint | FullFT mPQ / F1det / F1type | Selected-PEFT seed/checkpoint | PEFT mPQ / F1det / F1type |
|---|---|---|---|---|
| Breast | 42 / final epoch alias | 0.3481 / 0.8176 / 0.6232 | 42 / final epoch alias | 0.3348 / 0.8424 / 0.5873 |
| Colon | 42 / final epoch alias | 0.2057 / 0.7876 / 0.6910 | 42 / final epoch alias | 0.2047 / 0.7904 / 0.6076 |
| Kidney | 42 / final epoch alias | 0.2592 / 0.8285 / 0.5222 | mean of seed 42 final epoch and seed 43 model-best | 0.2484 / 0.8332 / 0.5013 |
| Liver | 42 / final epoch alias | 0.4124 / 0.8839 / 0.6661 | 42 / final epoch alias | 0.3925 / 0.8916 / 0.5673 |
| Lung | 42 / final epoch alias | 0.3225 / 0.8511 / 0.6683 | 42 / final epoch alias | 0.3015 / 0.8637 / 0.6271 |
| Ovary | 42 / final epoch alias | 0.3195 / 0.8311 / 0.6455 | 42 / final epoch alias | 0.2999 / 0.8414 / 0.6182 |
| Pancreatic | 42 / final epoch alias | 0.2118 / 0.6802 / 0.6965 | 43 / model-best | 0.2389 / 0.7383 / 0.6682 |
| Skin | 42 / final epoch alias | 0.2107 / 0.7006 / 0.6892 | 42 / model-best | 0.2239 / 0.7250 / 0.6778 |
| Tonsil | 42 / final epoch alias | 0.2498 / 0.8203 / 0.6752 | 43 / model-best | 0.2346 / 0.8226 / 0.5955 |

The full-precision source table is
`reports/paper_tables/tissue_specific_compayl2026_final.csv`; Dice, Jaccard and
bPQ remain in the corresponding canonical `inference_results.json` files. The
new tissue Fold-A campaign instead fixes seed 42 and `checkpoint_10.pth` for
every tissue, so Pancreatic/Tonsil historical deltas carry both seed and
checkpoint caveats, Skin carries a checkpoint caveat, and Kidney's published
mean combines two checkpoint policies.

Every trainable SAM-H and CellViT-256 KLT row in the two canonical CSVs has a
readable `checkpoint_10.pth`, final `inference_results.json`, completed training
and inference efficiency JSONs, and checkpoint-retention metadata. Frozen has
no training checkpoint by definition and has completed class-agnostic inference
plus inference-efficiency metadata.

The numerical typing claims are supported with two qualifications:

- Relative to the historical within-slide KLT result, slide-independent
  detection F1 changes by about +0.016 to +0.020 for Selected PEFT/FullFT,
  whereas F1 type changes by -0.080 to -0.177 across matched method/seed rows.
  Typing is therefore less transferable than detection in these comparisons.
- Fold effects dominate seed effects for bPQ and detection F1 for every
  trainable SAM-H method, and for NT-header1/FullFT typing. This is not universal:
  Selected-PEFT and LP F1-type fold gaps are smaller than their largest within-fold
  seed SD. Use “fold effects are often larger,” not an unqualified statement.

The `Other` failure and NT-header1 trade-off are numerically supported by the
saved per-class tables and matched-nuclei confusion matrices. Across seeds and
folds, NT-header1 raises mean Other F1 from approximately 0.005 to 0.149, while
mean Epithelial F1 falls from approximately 0.680 to 0.615. This is a descriptive
class-dependent trade-off, not a reason to redefine the pre-selected primary
method after seeing the held-out test results.

Per-slide and per-class rows are already materialized for SAM-H Selected PEFT,
NT-header1 and FullFT and for CellViT-256 Selected PEFT/FullFT. Individual and
descriptively aggregated confusion-matrix CSV/PNG files are materialized for
the three SAM-H typing methods. CellViT-256 and tissue-specific inference JSONs
contain the matched-pair confusion and unmatched-count sufficient statistics,
but their final standalone confusion plots/tables have not yet been materialized;
that is an existing-output analysis, not a missing training experiment.

### Generalist versus specialist status

The seed-42 Fold-A generalist/specialist comparison is complete for Kidney and
Liver on exactly the same held-out slide and exactly the same patch set. Tonsil
uses the same held-out slide (`tonsil_s1`) but different packed coverage: the KLT
source contains 9,879 test patches and the tissue-specific source contains 21,083.
The existing per-patch results permit a composition-matched descriptive
reaggregation on the common identifiers without retraining; until then, the
whole-slide Tonsil aggregates are not perfectly composition matched.

### Biological-independence wording

The local STHELAR README identifies slide IDs and states that the dataset has 27
slides from 20 cancer patients, but it provides no slide-to-patient, donor,
specimen, or section mapping. The per-cell and patch metadata also identify
slides rather than biological subjects. Consequently, the strongest supported
wording is **slide-independent evaluation**. Patient-independent or
specimen-independent evaluation must not be claimed from these identifiers.
