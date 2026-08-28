# CellViT-256 x40 Baseline Audit

Audit date: 2026-08-21 (Europe/Paris). The audit was performed from the current Ruche filesystem before submission. No model was downloaded and no fold dataset was regenerated or modified.

## Gate decision

**PASS.** The repository already contained both the `CellViT256` implementation and a usable full pretrained CellViT-256 x40 checkpoint. The checkpoint strictly loads into the current original-taxonomy constructor and completes a 256 x 256 forward pass. The existing architecture also supports STHELAR training and inference. A localized legacy control-flow issue that left ViT256 decoders trainable regardless of the requested SAM-scoped adapter mode was found by the integration smoke test; the current factory now explicitly applies only the two requested ViT256 modes (`final_heads_only` and `fullft`) and fails closed for any other strict-campaign mode. This is trainability dispatch, not a new model architecture.

## Implementation evidence

- Construction: `models/segmentation/cell_segmentation/cellvit.py`, class `CellViT256`, uses a 384-dimensional, 12-block, 6-head ViT with extract layers 3/6/9/12 and patch size 16.
- Training factory: `cell_segmentation/experiments/experiment_cellvit_pannuke.py` accepts `model.backbone: ViT256`, constructs `CellViT256`, loads the full pretrained model, and now applies exact ViT256 LP or FullFT trainability.
- Inference factory: `cell_segmentation/inference/inference_cellvit_experiment_pannuke.py` recognizes checkpoint architecture `CellViT256`, reconstructs the model, and strictly loads trained or untouched checkpoint state.
- STHELAR output adaptation: the STHELAR constructor has six nucleus channels (background plus five foreground targets) and one tissue channel. All 437 shape-compatible pretrained tensors load; only `encoder.head.weight` and `encoder.head.bias` are deliberately omitted because the pretrained tissue classifier is 19-way and STHELAR KLT is one-way. The strict campaign loader asserts that there are no other omissions.
- Live smoke results:
  - untouched Frozen model: strict load, zero trainable parameters, output shapes tissue `[1,19]`, NP `[1,2,256,256]`, HV `[1,2,256,256]`, NT `[1,6,256,256]`;
  - STHELAR LP model: strict-compatible base load, 650 trainable parameters, output shapes tissue `[1,1]`, NP `[1,2,256,256]`, HV `[1,2,256,256]`, NT `[1,6,256,256]`;
  - STHELAR FullFT model: strict-compatible base load, all 46,743,419 parameters trainable with the same output shapes.
- The class-agnostic Frozen evaluator completed a two-real-patch Fold A smoke test and emitted only Dice, binary Jaccard, bPQ/bDQ/bSQ, and detection precision/recall/F1. It did not emit STHELAR class-aware results.

## Pretrained checkpoint identity

| Field | Value |
|---|---|
| Exact path | `${PROJECT_ROOT}/models/pretrained/CellViT-256-x40.pth` |
| Logical size | 187,224,155 bytes (178.551 MiB; 0.17437 GiB) |
| SHA256 | `ee3986922fc500353db3d7692c566e19e1c694c8f10e47cf49dfd90160fc3b2b` |
| Checkpoint architecture | `CellViT256` |
| Checkpoint epoch | 129 |
| Magnification | x40 is declared by the repository checkpoint identity `CellViT-256-x40.pth`; the embedded flattened config does not contain a magnification field. All new configs independently require `data.magnification: 40`. No x20 checkpoint is used. |
| Pretraining dataset | PanNuke |
| Nucleus taxonomy | Background, Neoplastic, Inflammatory, Connective, Dead, Epithelial |
| Tissue taxonomy | 19 PanNuke tissues |
| State tensors | 439 |
| Strict load | All 439 tensors load strictly into `CellViT256(num_nuclei_classes=6, num_tissue_classes=19)` with no missing or unexpected keys. |
| Untouched total parameters | 46,750,349 |
| STHELAR-constructor total parameters | 46,743,419 (19-way tissue classifier replaced by a 1-way classifier) |

## Taxonomy constraint

The PanNuke nucleus labels are not equivalent to STHELAR's Background, Immune, Stromal, Epithelial, Melanocyte, and Other labels. Equal channel counts do not establish a mapping. Frozen therefore retains the untouched PanNuke head, trains zero parameters, and reports class-agnostic metrics only. No STHELAR mPQ, per-class PQ/DQ/SQ, F1-type, confusion matrix, or tissue-classifier accuracy is valid for Frozen.

For LP and FullFT, the six-channel head is optimized against the explicit STHELAR dataset mapping; this is supervised adaptation, not a claimed PanNuke-to-STHELAR semantic mapping.

## Exact LP and FullFT scopes

LP trains exactly these tensors:

1. `nuclei_binary_map_decoder.decoder0_header.2.weight`
2. `nuclei_binary_map_decoder.decoder0_header.2.bias`
3. `hv_map_decoder.decoder0_header.2.weight`
4. `hv_map_decoder.decoder0_header.2.bias`
5. `nuclei_type_maps_decoder.decoder0_header.2.weight`
6. `nuclei_type_maps_decoder.decoder0_header.2.bias`

Their counts are 130 NP + 130 HV + 390 NT = **650 trainable parameters**, or 0.00139057% of 46,743,419. The count happens to equal SAM-H LP because both architectures end in the same three 64-channel 1 x 1 prediction convolutions; it was measured from the ViT256 model and not asserted from the SAM-H result. The encoder, shared decoder helpers, branch bodies, and tissue classifier remain frozen.

FullFT trains **46,743,419 / 46,743,419 parameters (100%)**.

## Difference from CellViT-SAM-H x40

The matched STHELAR CellViT-SAM-H model has 699,736,523 parameters. The matched CellViT-256 model has 46,743,419: **6.68015% as many parameters, 652,993,104 fewer parameters, or 14.9697 times smaller by parameter count**. CellViT-256 uses a 384-dimensional 12-block ViT; SAM-H uses a 1,280-dimensional 32-block encoder. Both use the same non-shared CellViT NP/HV/NT decoder family and the same final-head LP definition.

## Fold revalidation

The existing Fold A/B materializations were read only. Independent CSV checks confirmed unique image and label patch identifiers within every split and pairwise disjoint identifiers across train/valid/test.

| Fold | Train | Validation | Test | Train/validation slides | Test-only slides | Payload links |
|---|---:|---:|---:|---|---|---|
| A | 24,283 | 2,052 | 23,494 | kidney_s0, liver_s0, tonsil_s0 | kidney_s1, liver_s1, tonsil_s1 | exact symlinks to source `images.zip`, `labels.zip`, `types.csv`, and `dataset_config.yaml`; targets exist and share the source inodes |
| B | 20,893 | 2,601 | 26,335 | kidney_s1, liver_s1, tonsil_s1 | kidney_s0, liver_s0, tonsil_s0 | exact symlinks to the same immutable source payloads; targets exist and share the source inodes |

Both `split_validation.json` files report all checks passed, exact slide assignments, train/test and validation/test slide disjointness, and patch-identifier disjointness.

## Historical CellViT-256 run

The historical run is:

`run/sthelar40x_liver_5class_spatial_vit256_lora_adaptformer_r8_a8_gelu_red16_lr5e-5_e10_seed42_CLEAN/log/2026-06-18T172812_sthelar40x_liver_5class_spatial_vit256_lora_adaptformer_r8_a8_gelu_red16_lr5e-5_e10_seed42_CLEAN`

It completed ten training epochs and demonstrates that the constructor, checkpoint loading, STHELAR dataloader, losses, and GPU training path can execute. It is not scientifically comparable because it used only liver, the older pre-margin dataset `sthelar40x_liver_5class_spatial`, not the current two-fold slide-independent KLT protocol, and requested LoRA+AdaptFormer rather than Frozen/LP/FullFT. More importantly, its log reports 25,077,755 trainable parameters—the frozen encoder plus entire decoder—showing that the requested adapters were not actually inserted because of the legacy SAM-only dispatch. Its automatic final inference then failed strict loading for the absent adapters, and no completed inference result/checkpoint remains. It is engineering evidence only and is not a primary comparison.

## Storage preflight

`ruche-quota` reported workdir usage of 426/500 GB (85%), approximately 74 GB free. Estimates use observed SAM-H checkpoint-10 overheads and CellViT-256's exact FP32 parameter payload, rather than assuming file size scales perfectly:

| Quantity | Estimate |
|---|---:|
| LP checkpoint | 187,352,762 bytes (178.674 MiB; 0.17449 GiB) |
| FullFT checkpoint with AdamW state | 561,871,124 bytes (535.842 MiB; 0.52328 GiB) |
| LP `model_best` + `checkpoint_10` high-water per run | 374,705,524 bytes (0.34897 GiB) |
| FullFT `model_best` + `checkpoint_10` high-water per run | 1,123,742,248 bytes (1.04657 GiB) |
| Four-run concurrent high-water (two LP + two FullFT) | about 2.791 GiB |
| Final persistent usage (one checkpoint for each of four runs) | about 1.396 GiB |

The retention wrapper keeps `checkpoint_10.pth` only after checkpoint readability, successful checkpoint-10 inference, saved results, completed training/inference efficiency records, and retention metadata containing all ten validation scores, best validation epoch/score, and final validation score.

## Efficiency and aggregation

The existing `EfficiencyRecorder` remains the only measurement pipeline. It records parameter counts, trainability percentage, train/validation/test patch counts, epoch and total training times, training/inference peak CUDA allocated and reserved memory, checkpoint size, GPU, batch size, AMP, inference wall time, patch count, and patches/s. Records now carry an explicit magnification-qualified `backbone` field. The aggregation key and `reports/slide_exp_efficiency.csv` include that field, preventing `CellViT-SAM-H x40` and `CellViT-256 x40` conditions from merging.

## Final held-out-slide status — 2026-08-27

This section supersedes the pre-submission limitation to LP/FullFT above. A
strict CellViT-256 Selected-PEFT mode was subsequently implemented and validated
without changing the conceptual method: rank-8/alpha-8 LoRA modifies only the Q
and V slices of each of the 12 fused QKV projections; reduction-16 GELU
AdaptFormer modules are inserted in the 12 MLP residual paths; and only the
final NP/HV/NT heads are trainable in the decoder. All decoder-body and encoder
base parameters remain frozen.

The final scope is 374,198 trainable parameters out of 47,116,967
(0.7941894902%): 147,456 LoRA, 226,092 AdaptFormer, and 650 final-head
parameters. The corresponding SAM-H encoder has depth 32, embedding width
1,280 and 16 heads; CellViT-256 has depth 12, width 384 and 6 heads. This is a
backbone-scale contextualization of the same adaptation strategy, not a
replacement for the primary SAM-H method.

All eight requested CellViT-256 Fold-A/B conditions are COMPLETED VALID: Frozen
inference plus LP, Selected PEFT, and FullFT at seed 42. Trainable rows retain a
readable `checkpoint_10.pth` and have final inference, training/inference
efficiency, and retention metadata. Frozen retains the original PanNuke taxonomy
and reports only class-agnostic metrics.

| Method | Fold A bPQ / mPQ / F1det / F1type | Fold B bPQ / mPQ / F1det / F1type |
|---|---|---|
| Frozen | 0.4745 / NA / 0.8336 / NA | 0.5614 / NA / 0.8578 / NA |
| LP | 0.4470 / 0.1736 / 0.8284 / 0.3994 | 0.5483 / 0.1994 / 0.8531 / 0.4140 |
| Selected PEFT | 0.4684 / 0.1905 / 0.8245 / 0.4976 | 0.5676 / 0.1952 / 0.8593 / 0.4648 |
| FullFT | 0.5110 / 0.1927 / 0.8359 / 0.4674 | 0.5964 / 0.2046 / 0.8659 / 0.4812 |

Canonical paths and full precision are in
`reports/neurips2026_paper_results/cellvit256_slideind.csv`. The historical
Liver run described earlier remains **FAILED SCIENTIFIC** for PEFT comparison:
its requested adapters were never inserted, its trainable scope was wrong, and
its final inference failed. It is superseded by the valid Fold-A/B campaign and
must not enter a paper table.
