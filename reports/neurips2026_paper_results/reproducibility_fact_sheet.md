# Authoritative training and reproducibility fact sheet

Status date: 2026-08-27 (public endpoints checked 2026-08-25). Values below were traced to the completed timestamped configs, executable code, saved efficiency metadata, materialization manifests, or immutable checkpoint metadata. `UNKNOWN` means the requested value could not be verified; it is not inferred.

## Methods values

| Field | Exact verified value | Provenance |
|---|---|---|
| Input patch size | 256 × 256 RGB pixels | Timestamped run `config.yaml`, `data.input_shape` |
| Magnification | 40× | Timestamped run `config.yaml`, `data.magnification` |
| Slide-independent folds | Fold A trains/validates on kidney/liver/tonsil s0 and tests on all s1 patches; Fold B reverses s0/s1 | Fold `split_manifest.yaml` and `split_validation.json` |
| Train/validation policy | Within training slides only: spatial x-axis 85% train / 15% validation | `utils/generate_slide_independent_fold.py`; fold manifests |
| Test policy | Whole held-out slides; Fold A 23,494 test patches, Fold B 26,335; primary unfiltered `dataset`/`image_metrics` outputs only | Fold manifests and final `inference_results.json` |
| 128-pixel margin | Inherited from the earlier source materialization's spatial x-boundary exclusion bands. The new materialization reuses the packed payload and applies the margin to the within-training-slide train/validation boundary; train/test independence is by slide ID, not by this margin. | `utils/generate_slide_independent_fold.py`; source/fold manifests |
| Optimizer | AdamW, betas=(0.85, 0.85) | Timestamped configs; `base_ml/base_experiment.py` |
| Learning rate | LP 5×10⁻⁵; Selected PEFT/NT-header1 5×10⁻⁵; FullFT 1×10⁻⁵ | Timestamped configs |
| Weight decay | 0.01, inherited from the PyTorch 2.5.1 `torch.optim.AdamW` default because configs omit `weight_decay` and code forwards only configured kwargs | Configs; `base_ml/base_experiment.py`; verified local PyTorch signature |
| Scheduler | ExponentialLR, gamma=0.95 (code default; configs select `exponential` and omit gamma) | `experiment_cellvit_pannuke.py`, `get_scheduler` |
| Training batch size | 4 | Timestamped configs / efficiency audit |
| Inference batch size | 16 | Timestamped configs / efficiency audit |
| Augmentations | Each p=0.5: RandomRotate90, horizontal flip, vertical flip, downscale(scale=0.2), blur(limit=10), Gaussian noise(var_limit=10), color jitter(scale_setting=0.25, scale_color=0.1), superpixels, zoom blur, random-sized crop, elastic transform | Timestamped configs |
| Normalization | Per RGB channel mean=(0.5,0.5,0.5), SD=(0.5,0.5,0.5) | Timestamped configs |
| NP losses | xentropy_loss weight 1 + dice_loss weight 1, static | Default loss code; configs contain no override |
| HV losses | mse_loss_maps weight 1 + msge_loss_maps weight 1, static | Default loss code; configs contain no override |
| NT losses | xentropy_loss weight 1 + dice_loss weight 1, static | Default loss code; configs contain no override |
| Tissue loss | CrossEntropyLoss weight 1, static, one KLT tissue output | Default loss code and configs |
| Regression loss | None | Config/code branch not enabled |
| Class imbalance handling | Cell-presence WeightedRandomSampler, gamma=0.85, replacement=True, `num_samples=len(train_dataset)`, seeded generator. Per-class presence weight `k/(gamma*n_c+(1-gamma)*k)` and image weights from the gamma-weighted sum; absent-class-free patches receive the smallest nonzero image weight. | `cell_segmentation/datasets/pannuke.py`; `experiment_cellvit_pannuke.py`; configs |
| AMP policy | CUDA autocast FP16 for train/inference; GradScaler during training | Trainer code and saved efficiency audit |
| Epochs | 10 | Timestamped configs |
| Checkpoint policy | Save best and last, evaluate every epoch, retain final epoch under retention policy. Canonical tests here explicitly used fixed `checkpoint_10.pth`; no epoch was selected using test performance. | Configs, inference logs, `reports/checkpoint_selection_policy_audit.md` |
| Seed protocol | SAM-H Frozen seed 42; SAM-H LP/Selected PEFT/NT-header1/FullFT seeds 42 and 43; CellViT-256 all requested methods seed 42. Seed effects are summarized within each fold; folds are not seed replicates. | Completed-run inventory |
| Hardware | Matched runs: NVIDIA A100-SXM4-40GB | Saved efficiency metadata |
| Software | PyTorch 2.5.1+cu121; CUDA 12.1; Python used by project environment 3.9.25 | Saved efficiency metadata / verified environment |

## Parameter counts

| Backbone/method | Total | Trainable | Trainable % |
|---|---:|---:|---:|
| SAM-H Frozen | 699,741,149 | 0 | 0 |
| SAM-H LP | 699,736,523 | 650 | 0.0000928921 |
| SAM-H Selected PEFT | 707,644,395 | 7,908,779 | 1.117620525 |
| SAM-H NT-header1 | 707,644,395 | 7,945,835 | 1.122857053 |
| SAM-H FullFT | 699,736,523 | 699,736,523 | 100 |
| CellViT-256 Frozen | 46,750,349 | 0 | 0 |
| CellViT-256 LP | 46,743,419 | 650 | 0.0013905701 |
| CellViT-256 Selected PEFT | 47,116,967 | 374,198 | 0.7941894902 |
| CellViT-256 FullFT | 46,743,419 | 46,743,419 | 100 |

## Exact Selected PEFT definition

- LoRA rank 8, alpha 8, dropout 0, applied to Q and V slices of the fused QKV projection in every transformer encoder block.
- AdaptFormer after each encoder-block MLP, bottleneck reduction 16, GELU activation.
- Trainable final 1×1 NP, HV and NT decoder heads. SAM-H also has the one-output tissue classifier trainable under the canonical policy; CellViT-256 does not train its encoder classification head.
- Decoder bodies and pretrained backbone weights remain frozen. The verified deployable adapter additionally carries the 105 mutable normalization buffers required for exact reconstruction, although buffers are not trainable parameters.
- NT-header1 is a prespecified typing variant that additionally trains the preceding NT header convolution; it is reported separately and is not relabeled as Selected PEFT.

## Reproducibility and release status

| Item | Current factual status | Anonymous-PDF treatment |
|---|---|---|
| Git repository | The migration snapshot records its source revision in `reports/cluster_migration_snapshot.md`. Public code includes PEFT modules and within-slide release configs; the completed slide-independent seed/fold work is carried by this snapshot. | Keep repository ownership and author history outside anonymous manuscripts when required by venue policy. |
| Code required for PEFT | Present locally in `models/adapters/`, adapter application/loading utilities, training experiment code, and release verification scripts. A prior PEFT-capable revision is public; the exact new slide-independent configs/results package is not public. | Public repository link reveals identity; do not link anonymously. |
| Configs | Completed-run timestamped configs and local `configs/slide_exp/` exist. Public GitHub/Hugging Face configs cover the earlier within-slide release, not this final slide-independent package. | Public links reveal account identity; omit. |
| External adapter release | An earlier public, ungated release contains 12 within-slide KLT/tissue-specific adapter packages. **The new slide-independent Fold A/B adapters audited here are not in that remote release.** | Keep account namespaces outside anonymous manuscripts when required by venue policy. |
| Pretrained checkpoint provenance | Local required base `models/pretrained/CellViT-SAM-H-x40.pth`, 2,799,315,941 bytes, SHA256 `b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf`; README attributes it to official CellViT-SAM-H x40 and records a Google Drive download. Base is not redistributed. Exact immutable upstream release identifier/version: **UNKNOWN**. | Upstream CellViT link is identity-safe; the project README link is not. Cite CellViT normally, but do not expose the author-owned repository/HF namespace. |

## Biological-independence audit

| Tissue | Paired slide IDs used in this paper | Patient/donor relationship | Specimen/section relationship | Evidence |
|---|---|---|---|---|
| Breast | `breast_s0`, `breast_s1` | UNKNOWN | UNKNOWN | Local `Datasets/STHELAR_40x/README.md`; fold manifests |
| Colon | `colon_s1`, `colon_s2` | UNKNOWN | UNKNOWN | same |
| Kidney | `kidney_s0`, `kidney_s1` | UNKNOWN | UNKNOWN | same |
| Liver | `liver_s0`, `liver_s1` | UNKNOWN | UNKNOWN | same |
| Lung | `lung_s1`, `lung_s3` | UNKNOWN | UNKNOWN | same |
| Ovary | `ovary_s0`, `ovary_s1` | UNKNOWN | UNKNOWN | same |
| Pancreatic | `pancreatic_s0`, `pancreatic_s1` | UNKNOWN | UNKNOWN | same |
| Skin | `skin_s1`, `skin_s2` | UNKNOWN | UNKNOWN | same |
| Tonsil | `tonsil_s0`, `tonsil_s1` | UNKNOWN | UNKNOWN | same |

The dataset README states that STHELAR contains 27 human FFPE slides and samples
from 20 cancer patients, but it does not map individual slide IDs to patients,
donors, specimens, or sections. `patches_overview_sthelar40x.parquet`, the
per-slide cell-metadata index, and the derived `patch_info_with_split.csv` files
retain `slide_id` but no biological-subject identifier. Therefore the paper may
say **slide-independent** and **completely held-out slide**, but it must not say
patient-independent, donor-independent, or specimen-independent without a new
authoritative metadata source.

No visibility, publication, checkpoint, or remote state was changed during this audit.
