# KLT typing and PEFT strategy analysis

## Scope and conclusion

This is a read-only analysis of completed inference artifacts. No job was launched and no experimental asset was changed. The only new artifact is this requested report.

All quantitative comparisons use the same KLT test set: `sthelar40x_kidney_liver_tonsil_5class_spatial_margin128`, split `test`, 4,653 patches, 40x, six output IDs (`Background` plus five reported classes). Seed 42 is the primary like-for-like comparison; seed 43 is used wherever an equivalent completed run exists. Seed 44 is shown only as additional last-stage sensitivity evidence.

Main conclusion: the selected LoRA+AdaptFormer `heads_only` method preserves class-agnostic detection extremely well, but its two-seed present-class typing gap to FullFT is almost entirely an `Other`-class gap. Existing decoder adaptation recovers `Other`. The best single-seed `Other` result is the full-NT-branch scope, but the best balanced already-tested operating point is the much smaller `nt_header1_np_hv_heads` scope. The latter needs a seed-43 replication before it should replace the selected method in a revised paper.

## Metric and support definitions

- `mPQ`, `bPQ`, and detection F1 are copied from each final `inference_results.json` dataset block.
- `F1 type` is the macro-F1 over classes with nonzero **matched true support**, computed from the saved centroid-pair `detection_stats.paired_confusion`. Background rows/columns are excluded. In KLT, the present classes are Immune, Stromal, Epithelial, and Other; Melanocyte has zero support, so this is a four-class macro average.
- Per-class precision, recall, and F1 are **matched-nuclei type-assignment** metrics: precision = diagonal / matched predicted support; recall = diagonal / matched true support. The `n` in the per-class table is matched true support after excluding pairs assigned Background. It is not raw dataset frequency and does not include unmatched detections.
- This differs from `nuclei_metrics_d`, whose per-class score also incorporates unmatched nuclei. The matched-only definition is used here because it isolates typing among detected/matched nuclei and is the definition used by the existing F1-type paper tables.
- The raw converted test metadata (`cell_count_test.csv`) contains 141,623 Immune, 54,200 Stromal, 58,140 Epithelial, 0 Melanocyte, and 9,558 Other per-patch nucleus observations. The inference coordinate statistics contain 9,421 true Other observations after their own extraction/counting path. These two support systems should not be mixed with the matched support `n` in the table.
- Frozen pretrained CellViT-SAM-H x40 is reported only for semantically comparable class-agnostic `bPQ` and detection F1. Its PanNuke type outputs are not an independently justified mapping to STHELAR classes; therefore frozen mPQ, F1 type, and per-class type metrics are deliberately `NA`.

Values are printed to six decimal places from the stored floating-point results; integer supports are exact.

## Exact comparable results

### Aggregate comparison

| Method | Decoder scope | Seed | Trainable params | Trainable % | mPQ | bPQ | F1 detection | Present-class F1 type |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen pretrained | frozen | 42 | 0 | 0.0000 | NA | 0.446763 | 0.816732 | NA |
| Final-head LP | NP/HV/NT final heads | 42 | 650 | 0.0001 | 0.204322 | 0.445691 | 0.818139 | 0.429935 |
| AdaptFormer only | decoder frozen | 42 | 6,597,152 | 0.9340 | 0.118525 | 0.418378 | 0.805933 | 0.296781 |
| LoRA only | decoder frozen | 42 | 1,310,720 | 0.1870 | 0.112672 | 0.430256 | 0.830822 | 0.280017 |
| VeRA only | decoder frozen | 42 | 82,976 | 0.0119 | 0.064683 | 0.456896 | 0.827227 | 0.181212 |
| Decoder conv only | conv adapters + final heads | 42 | 553,002 | 0.0790 | 0.231714 | 0.458723 | 0.828193 | 0.561672 |
| AdaptFormer + heads | final heads | 42 | 6,598,059 | 0.9341 | 0.290237 | 0.505257 | 0.834497 | 0.616906 |
| AdaptFormer + heads | final heads | 43 | 6,598,059 | 0.9341 | 0.285183 | 0.492114 | 0.831143 | 0.577020 |
| AdaptFormer + last stage | all branch headers + shared `decoder0` | 42 | 6,950,571 | 0.9840 | 0.290929 | 0.498127 | 0.828122 | 0.647072 |
| **Selected PEFT: LoRA+AF** | final heads | 42 | 7,908,779 | 1.1176 | 0.296948 | 0.507577 | 0.833585 | 0.561243 |
| **Selected PEFT: LoRA+AF** | final heads | 43 | 7,908,779 | 1.1176 | 0.290346 | 0.500949 | 0.835498 | 0.595330 |
| LoRA+AF | last stage | 42 | 8,261,291 | 1.1674 | 0.297845 | 0.506423 | 0.827868 | 0.657020 |
| LoRA+AF | last stage | 43 | 8,261,291 | 1.1674 | 0.294060 | 0.498808 | 0.823991 | 0.651152 |
| LoRA+AF | last stage | 44 | 8,261,291 | 1.1674 | 0.294585 | 0.505903 | 0.821758 | 0.640543 |
| LoRA+AF | conv adapters + final heads | 42 | 8,460,874 | 1.1947 | 0.286352 | 0.488987 | 0.828971 | 0.649546 |
| LoRA+AF | conv adapters + final heads | 43 | 8,460,874 | 1.1947 | 0.290427 | 0.497174 | 0.829562 | 0.653335 |
| **LoRA+AF NT header1** | NP/HV/NT final heads + pre-final NT block | 42 | 7,945,835 | 1.1229 | 0.292351 | 0.499539 | 0.832403 | 0.662082 |
| **LoRA+AF full NT** | NP/HV final heads + full NT branch | 42 | 22,986,219 | 3.2483 | 0.293824 | 0.498408 | 0.831731 | 0.668528 |
| VeRA+AF | final heads | 43 | 6,681,035 | 0.9458 | 0.287031 | 0.493231 | 0.832649 | 0.579022 |
| VeRA+AF | conv adapters + final heads | 42 | 7,233,130 | 1.0231 | 0.283968 | 0.491809 | 0.832795 | 0.643559 |
| VeRA+AF | conv adapters + final heads | 43 | 7,233,130 | 1.0231 | 0.285742 | 0.491673 | 0.824669 | 0.660772 |
| FullFT | all | 42 | 699,736,523 | 100.0000 | 0.305804 | 0.508390 | 0.827083 | 0.668088 |
| FullFT | all | 43 | 699,736,523 | 100.0000 | 0.312629 | 0.521080 | 0.830126 | 0.664137 |

### Exact per-class typing comparison

Each cell is `precision / recall / F1 (matched true support n)`.

| Method | Seed | Immune P/R/F1 (n) | Stromal P/R/F1 (n) | Epithelial P/R/F1 (n) | Melanocyte P/R/F1 (n) | Other P/R/F1 (n) |
|---|---:|---|---|---|---|---|
| Frozen pretrained | 42 | NA | NA | NA | NA | NA |
| Final-head LP | 42 | 0.756407/0.835783/0.794117 (107,492) | 0.383996/0.239993/0.295378 (37,972) | 0.586157/0.681503/0.630245 (40,710) | 0.000000/0.000000/0.000000 (0) | 0.000000/0.000000/0.000000 (3,662) |
| AdaptFormer only / decoder frozen | 42 | 0.738727/0.903362/0.812791 (101,233) | 0.404757/0.233980/0.296538 (38,187) | 0.918842/0.036462/0.070140 (44,403) | 0.000000/0.000000/0.000000 (0) | 0.004044/0.071327/0.007654 (2,103) |
| LoRA only / decoder frozen | 42 | 0.747174/0.864853/0.801718 (109,747) | 0.437116/0.176076/0.251032 (40,051) | 0.845353/0.025271/0.049074 (44,993) | 0.000000/0.000000/0.000000 (0) | 0.009841/0.124823/0.018243 (3,533) |
| VeRA only / decoder frozen | 42 | 0.732957/0.267894/0.392376 (112,578) | 0.210014/0.716207/0.324789 (41,037) | 0.000000/0.000000/0.000000 (44,760) | 0.000000/0.000000/0.000000 (0) | 0.004736/0.020341/0.007683 (4,572) |
| Decoder conv only | 42 | 0.779189/0.848327/0.812290 (112,756) | 0.453125/0.464025/0.458510 (40,862) | 0.860858/0.698208/0.771049 (45,369) | 0.000000/0.000000/0.000000 (0) | 0.355447/0.143878/0.204840 (4,059) |
| AdaptFormer + heads | 42 | 0.792239/0.889456/0.838038 (113,575) | 0.602835/0.442590/0.510431 (41,517) | 0.891135/0.845498/0.867717 (46,200) | 0.000000/0.000000/0.000000 (0) | 0.270839/0.234633/0.251439 (4,002) |
| AdaptFormer + heads | 43 | 0.807251/0.854947/0.830415 (113,414) | 0.589792/0.511810/0.548041 (41,703) | 0.835416/0.880460/0.857346 (46,478) | 0.000000/0.000000/0.000000 (0) | 0.422572/0.039519/0.072278 (4,074) |
| AdaptFormer + last stage | 42 | 0.821420/0.847744/0.834375 (112,147) | 0.609295/0.479148/0.536440 (40,524) | 0.878486/0.878409/0.878448 (45,612) | 0.000000/0.000000/0.000000 (0) | 0.237362/0.593020/0.339026 (3,381) |
| Selected PEFT | 42 | 0.789189/0.900380/0.841126 (113,391) | 0.633379/0.447338/0.524345 (41,358) | 0.878258/0.876474/0.877365 (46,290) | 0.000000/0.000000/0.000000 (0) | 0.571429/0.001069/0.002134 (3,741) |
| Selected PEFT | 43 | 0.808411/0.859122/0.832995 (113,488) | 0.584751/0.519691/0.550305 (41,542) | 0.864711/0.875491/0.870068 (46,366) | 0.000000/0.000000/0.000000 (0) | 0.433904/0.075040/0.127952 (3,718) |
| LoRA+AF last stage | 42 | 0.811629/0.853837/0.832198 (112,183) | 0.569450/0.519043/0.543079 (40,409) | 0.902770/0.846245/0.873594 (45,566) | 0.000000/0.000000/0.000000 (0) | 0.351035/0.412299/0.379208 (3,415) |
| LoRA+AF last stage | 43 | 0.821935/0.843728/0.832689 (111,786) | 0.597346/0.509856/0.550144 (40,433) | 0.889626/0.865737/0.877519 (45,545) | 0.000000/0.000000/0.000000 (0) | 0.248773/0.558701/0.344258 (3,356) |
| LoRA+AF last stage | 44 | 0.775280/0.921765/0.842200 (112,149) | 0.689255/0.389524/0.497751 (40,760) | 0.902227/0.837696/0.868765 (45,649) | 0.000000/0.000000/0.000000 (0) | 0.363769/0.343716/0.353458 (3,628) |
| LoRA+AF conv adapters | 42 | 0.818415/0.835214/0.826729 (112,977) | 0.568509/0.526663/0.546786 (40,825) | 0.904739/0.834011/0.867937 (45,756) | 0.000000/0.000000/0.000000 (0) | 0.261268/0.562129/0.356733 (3,702) |
| LoRA+AF conv adapters | 43 | 0.819512/0.846357/0.832718 (113,106) | 0.606157/0.488508/0.541010 (41,115) | 0.844531/0.895721/0.869373 (45,848) | 0.000000/0.000000/0.000000 (0) | 0.320367/0.438501/0.370239 (4,057) |
| LoRA+AF NT header1 | 42 | 0.807044/0.863920/0.834514 (112,963) | 0.594921/0.502371/0.544743 (41,127) | 0.893738/0.865160/0.879217 (45,973) | 0.000000/0.000000/0.000000 (0) | 0.394653/0.385172/0.389855 (3,871) |
| LoRA+AF full NT | 42 | 0.811427/0.863153/0.836491 (113,448) | 0.618611/0.491630/0.547859 (41,161) | 0.882826/0.872816/0.877792 (46,122) | 0.000000/0.000000/0.000000 (0) | 0.349991/0.500618/0.411968 (4,043) |
| VeRA+AF heads | 43 | 0.805516/0.848207/0.826311 (113,345) | 0.598316/0.491358/0.539588 (41,658) | 0.804802/0.889829/0.845182 (46,446) | 0.000000/0.000000/0.000000 (0) | 0.459144/0.059282/0.105006 (3,981) |
| VeRA+AF conv adapters | 42 | 0.806236/0.869564/0.836704 (113,320) | 0.602490/0.453234/0.517311 (41,109) | 0.892153/0.848630/0.869847 (45,630) | 0.000000/0.000000/0.000000 (0) | 0.268997/0.502349/0.350375 (4,045) |
| VeRA+AF conv adapters | 43 | 0.815737/0.855148/0.834977 (112,846) | 0.608282/0.493048/0.544636 (40,994) | 0.887202/0.868302/0.877650 (45,908) | 0.000000/0.000000/0.000000 (0) | 0.299030/0.543602/0.385823 (4,025) |
| FullFT | 42 | 0.810046/0.867486/0.837782 (113,399) | 0.647656/0.473359/0.546957 (41,383) | 0.899137/0.857643/0.877900 (46,025) | 0.000000/0.000000/0.000000 (0) | 0.302809/0.633285/0.409712 (4,783) |
| FullFT | 43 | 0.813025/0.871958/0.841461 (113,346) | 0.669195/0.441674/0.532135 (41,148) | 0.885209/0.877468/0.881322 (45,629) | 0.000000/0.000000/0.000000 (0) | 0.285754/0.675580/0.401629 (4,525) |

### Exact run and checkpoint ledger

Every completed inference log resolves to `checkpoint_10.pth`. The exact checkpoint is `<run directory>/checkpoints/checkpoint_10.pth` for each entry below.

| Method | Seed | Exact timestamped run directory |
|---|---:|---|
| Frozen pretrained | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN/log/2026-06-24T195646_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN` |
| Final-head LP | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42/log/2026-07-02T153245_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42` |
| AdaptFormer only / decoder frozen | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_adaptformer_red16_decoder_frozen_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T001346_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_adaptformer_red16_decoder_frozen_lr5e-5_e10_seed42_CLEAN` |
| LoRA only / decoder frozen | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_r8_a8_decoder_frozen_lr5e-5_e10_seed42_CLEAN/log/2026-06-24T220904_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_r8_a8_decoder_frozen_lr5e-5_e10_seed42_CLEAN` |
| VeRA only / decoder frozen | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_r16_a16_decoder_frozen_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T070650_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_r16_a16_decoder_frozen_lr5e-5_e10_seed42_CLEAN` |
| Decoder conv only | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_frozen_encoder_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-24T201215_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_frozen_encoder_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN` |
| AdaptFormer + heads | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_adaptformer_red16_heads_e10_seed42/log/2026-07-03T190728_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_adaptformer_red16_heads_e10_seed42` |
| AdaptFormer + heads | 43 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_adaptformer_red16_heads_e10_seed43/log/2026-07-05T100313_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_adaptformer_red16_heads_e10_seed43` |
| AdaptFormer + last stage | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_adaptformer_red16_heads_laststage_e10_seed42/log/2026-07-03T190728_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_adaptformer_red16_heads_laststage_e10_seed42` |
| Selected PEFT | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T113302_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN` |
| Selected PEFT | 43 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T002217_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN` |
| LoRA+AF last stage | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T185008_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN` |
| LoRA+AF last stage | 43 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN/log/2026-06-27T201436_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN` |
| LoRA+AF last stage | 44 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed44_CLEAN/log/2026-07-03T220834_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed44_CLEAN` |
| LoRA+AF conv adapters | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T102125_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN` |
| LoRA+AF conv adapters | 43 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T002217_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN` |
| LoRA+AF NT header1 | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_nt_header1_np_hv_heads_lr5e-5_e10_seed42_CLEAN/log/2026-07-04T151400_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_nt_header1_np_hv_heads_lr5e-5_e10_seed42_CLEAN` |
| LoRA+AF full NT | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_np_hv_heads_nt_all_lr5e-5_e10_seed42_CLEAN/log/2026-07-03T221049_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_np_hv_heads_nt_all_lr5e-5_e10_seed42_CLEAN` |
| VeRA+AF heads | 43 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-28T022341_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN` |
| VeRA+AF conv adapters | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T102125_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN` |
| VeRA+AF conv adapters | 43 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/log/2026-06-27T235829_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN` |
| FullFT | 42 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN/log/2026-06-24T195748_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN` |
| FullFT | 43 | `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed43_KEEPALL_FISHER/log/2026-06-26T002217_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed43_KEEPALL_FISHER` |

## Why detection is preserved while typing remains below FullFT

Across seeds 42/43, selected PEFT has mean detection F1 0.834541 versus 0.828604 for FullFT: selected is **+0.005937**. Its mean bPQ is 0.504263 versus 0.514735 (**-0.010473**) and mean mPQ is 0.293647 versus 0.309216 (**-0.015570**). Thus detection is not the bottleneck; class-aware panoptic quality and matched-nuclei macro typing remain lower.

Mean present-class F1 type is 0.578286 for selected PEFT versus 0.666113 for FullFT, a gap of **0.087826**. Yet Immune and Epithelial are nearly preserved: selected versus FullFT two-seed means are 0.837060 versus 0.839622 for Immune (**-0.002561**) and 0.873716 versus 0.879611 for Epithelial (**-0.005894**). Stromal is also close on average, 0.537325 versus 0.539546 (**-0.002221**). Other is not: 0.065043 versus 0.405671 (**-0.340627**).

This explains how class-agnostic detection and even majority-weighted typing can look strong while macro type-F1 is lower: the selected scope finds nuclei and classifies common types, but nearly stops emitting the minority/heterogeneous `Other` label at seed 42.

## `Other` and Stromal failure analysis

### Attribution of the macro-F1 gap

Because present-class macro-F1 is an arithmetic mean of four class F1s, the class-wise gap can be decomposed exactly (this is arithmetic attribution, not a causal claim).

- Seed 42: FullFT minus selected F1 type is 0.106845. The Other F1 difference is 0.407577; divided by four it contributes **0.101894**, or **95.37%** of the macro gap.
- Seed 43: the macro gap is 0.068808. Other contributes **0.068419**, or **99.44%**.
- Two-seed means: the macro gap is 0.087826. Other contributes **0.085157**, or **96.97%**.

Therefore almost all of the selected-PEFT versus FullFT typing gap is attributable to Other.

### Is Other sufficiently represented?

Yes for a test-set class-level difference, with an important clustering caveat. The raw test metadata has 9,558 Other observations (3.63% of 263,521 non-background observations). The inference coordinate accounting has 9,421 true Other observations. Matched true support is still in the thousands: 3,741/3,718 for selected seeds 42/43 and 4,783/4,525 for FullFT. This is not a one- or two-example rare-class effect.

However, 8,111 of the 9,558 raw Other observations (84.86%) occur in `tonsil_s1`; the remaining five slides contribute 109, 290, 175, 118, and 755. Thus the difference is meaningful for this KLT test set but is not strong multi-slide evidence about every biological subtype inside Other.

### Main Other confusions

Rows below are matched pairs with true Other; columns are predicted labels. Background-paired cases are excluded consistently with F1 type.

| Method | Seed | Immune | Stromal | Epithelial | Other | Other behavior |
|---|---:|---:|---:|---:|---:|---|
| Final-head LP | 42 | 1,954 | 1,024 | 684 | 0 | Never predicts Other. |
| Selected PEFT | 42 | 2,385 | 1,255 | 97 | 4 | 0.11% matched recall; collapse is mainly to Immune and Stromal. |
| Selected PEFT | 43 | 1,924 | 1,426 | 89 | 279 | Some recovery, but only 7.50% recall. |
| LoRA+AF last stage | 42 | 1,369 | 589 | 49 | 1,408 | 41.23% recall; main residual confusion is Immune. |
| LoRA+AF conv adapters | 42 | 912 | 643 | 66 | 2,081 | 56.21% recall, but precision falls to 0.261268 because many Immune/Stromal cells are predicted Other. |
| LoRA+AF NT header1 | 42 | 1,471 | 848 | 61 | 1,491 | Balanced 0.394653 precision / 0.385172 recall. |
| LoRA+AF full NT | 42 | 1,573 | 370 | 76 | 2,024 | Best F1 (0.411968); residual confusion is mainly Immune. |
| FullFT | 42 | 1,216 | 463 | 75 | 3,029 | Highest recall (0.633285), with lower precision (0.302809) from Other false positives. |

The selected seed-42 problem is overwhelmingly **failure to predict Other**, not merely noisy false-positive calibration: only seven predictions in the entire non-background matched matrix are Other, four of which are correct.

### Stromal, common types, Melanocyte, and LP

1. **Stromal is not consistently harmed.** Selected is below FullFT at seed 42 (0.524345 versus 0.546957) but above it at seed 43 (0.550305 versus 0.532135). The two-seed mean difference is only -0.002221. Decoder adaptation improves Stromal at seed 42 (last stage 0.543079, conv adapters 0.546786, NT header1 0.544743, full NT 0.547859) but not uniformly at seed 43.
2. **Immune and Epithelial are essentially preserved.** Their selected two-seed deficits are 0.002561 and 0.005894 respectively. No evidence supports a broad common-class collapse.
3. **Melanocyte is absent, not merely rare.** Raw test support and every matched confusion matrix give zero. KLT (Kidney, Liver, Tonsil) cannot support a Melanocyte conclusion.
4. **LP already fails on Other.** Final-head LP has Other F1 = 0 with 3,662 matched true examples and emits no matched Other predictions. Selected PEFT does not introduce a previously working Other classifier; it fails to resolve a failure already present under final-head-only adaptation. The selected seed-42 result is only marginally above zero, while seed 43 recovers to 0.127952.

### Which tested decoder scope recovers Other most?

At seed 42 the largest absolute Other F1 is **LoRA+AF with full NT branch**: 0.411968, a +0.409834 recovery over selected. It slightly exceeds seed-42 FullFT Other F1 (0.409712), but it is a single-seed observation.

The strongest efficiency result is **LoRA+AF NT header1**: Other F1 0.389855, +0.387721 over selected, for only 37,056 extra trainable parameters (+0.0053 percentage points). Relative to selected it changes detection F1 by -0.001182, mPQ by -0.004597, and bPQ by -0.008037. Epithelial improves +0.001851, Stromal improves +0.020397, and Immune falls -0.006612.

The full-NT scope changes detection F1 by -0.001854, mPQ by -0.003124, and bPQ by -0.009168 versus selected; Epithelial is +0.000427, Stromal +0.023514, and Immune -0.004635. Its cost is much larger: 22,986,219 trainable parameters (3.2483%), versus 7,908,779 (1.1176%) for selected.

### Seed-43 reproduction

The general decoder-recovery pattern reproduces, but the two typing-focused NT scopes have no seed-43 result.

- Last stage versus selected: Other improves from 0.002134 to 0.379208 at seed 42 and from 0.127952 to 0.344258 at seed 43. Present F1 improves by +0.095777 and +0.055823. Detection F1 costs are -0.005717 and -0.011506.
- Conv adapters versus selected: Other improves to 0.356733 and 0.370239; present F1 improves by +0.088303 and +0.058005. Detection F1 costs are -0.004614 and -0.005936.
- Across seeds 42/43, last stage mean Other F1 is 0.361733 and conv-adapter mean is 0.363486, versus 0.065043 selected. Thus the qualitative result is reproducible even though which generic decoder scope is best changes by seed.
- `nt_header1_np_hv_heads` and `np_hv_heads_nt_all` are seed-42 only. Their stronger balance is provisional until replicated.

## Encoder versus decoder interpretation

### Directly observed evidence

- Selected encoder LoRA Q/V + AdaptFormer + final heads preserves detection: its two-seed F1 detection exceeds FullFT by 0.005937, while common-class typing is close.
- Final-head LP is substantially weaker than selected at seed 42 (mPQ 0.204322 versus 0.296948; F1 type 0.429935 versus 0.561243), so pretrained features plus final linear heads alone are not enough for the full task.
- Encoder-only adapter controls with frozen decoder/heads retain detection but perform poorly in class-aware metrics. Because their STHELAR output heads are frozen, they do not isolate representation quality cleanly; they chiefly show that encoder adaptation without target-head adaptation is insufficient.
- Decoder-only convolutional adapters reach F1 detection 0.828193 and F1 type 0.561672 with 0.0790% trainable parameters, including Other F1 0.204840. Decoder-local adaptation alone has real typing value, but its mPQ 0.231714 remains far below the combined encoder+decoder methods.
- Controlled decoder-scope comparisons support extra decoder capacity for Other. With the same LoRA+AF encoder at seed 42, last-stage tuning changes Other from 0.002134 to 0.379208 and F1 type from 0.561243 to 0.657020, while mPQ is +0.000897. Seed 43 reproduces the Other gain.
- The tighter NT-header1 scope is even more informative at seed 42: adding only the pre-final NT block to the selected trainable set raises Other to 0.389855 and F1 type to 0.662082 for 37,056 extra parameters. The code and logs confirm NP/HV decoder bodies remain frozen.
- AdaptFormer-only encoder PEFT shows the same direction: heads to last-stage at seed 42 raises Other from 0.251439 to 0.339026 and F1 type from 0.616906 to 0.647072.

Taken together, existing results support a qualified version of the hypothesis: **encoder PEFT plus trained output heads is sufficient for strong detection and common-class typing on KLT, while additional decoder adaptation substantially improves the difficult Other class.** The word “requires” is too strong because the strongest branch-specific evidence has one seed and no causal mechanism was directly measured.

### Plausible interpretation

- Frozen decoder features may already encode localization and common morphology well enough for NP/HV detection and abundant Immune/Epithelial discrimination.
- A final 1x1 NT head may have insufficient task-specific transformation capacity to separate a heterogeneous catch-all class under imbalance; adapting a pre-final NT convolution can reshape features before the class logits without perturbing NP/HV bodies.
- Other’s heterogeneity and slide concentration could make its decision boundary more variable and encourage a head-only optimizer to favor common labels. The seed-42 selected confusion matrix is consistent with this: Other is almost always absorbed into Immune or Stromal.

These are interpretations consistent with the ablations, not experimentally established mechanisms.

### Unsupported claims

- That biological heterogeneity itself causes the architectural gap.
- That decoder adaptation is universally necessary for rare or heterogeneous classes, or that the result generalizes beyond KLT.
- That LoRA Q/V specifically preserves detection better than AdaptFormer; the available scope matrix is not a full factorial causal decomposition.
- That NT-header1 will reproduce at seed 43 or on tissue-specific datasets.
- That a two-stage training schedule, class-balanced loss change, or a new branch-specific adapter would outperform the already-tested joint NT-header1 run.
- Any Melanocyte claim.

## What composes grouped `Other`?

The preprocessing config uses source column `cells_final_label_group`. It maps `Specialized -> Other` and `Other -> Other`, with any unmapped source label also sent to fallback `Other`.

The local STHELAR dataset README defines source `Specialized` as tissue-specific cells such as cardiomyocytes, osteoblasts, osteoclasts, and some endocrine cells. It defines source grouped `Other` as the collapse of low-information `Less10`, `Unknown`, and `Stem_like` labels; its prose describes cells without marker genes or with fewer than 10 RNAs. Thus the model’s KLT `Other` target combines at least:

- source `Specialized` tissue-specific biology;
- source low-information `Other` (`Less10`, `Unknown`, `Stem_like`);
- any unexpected label caught by the preprocessing fallback.

This is a heterogeneous target by construction. That could plausibly contribute to difficulty, but the current experiments do not separate its source subcategories, so they cannot prove that heterogeneity rather than imbalance, slide concentration, optimization, or feature capacity causes the observed gap.

## Pareto trade-off

### Primary seed-42 LoRA+AdaptFormer frontier

| Scope | Trainable % | mPQ | F1 detection | F1 type | Other F1 | Versus selected | Pareto status |
|---|---:|---:|---:|---:|---:|---|---|
| Selected heads only | 1.1176 | 0.296948 | **0.833585** | 0.561243 | 0.002134 | Detection/bPQ-oriented baseline | Non-dominated |
| Last stage | 1.1674 | **0.297845** | 0.827868 | 0.657020 | 0.379208 | +0.000897 mPQ, -0.005717 detection, +0.095777 type | Non-dominated |
| Conv adapters | 1.1947 | 0.286352 | 0.828971 | 0.649546 | 0.356733 | Worse than NT header1 on every listed objective and more parameters | Dominated by NT header1 |
| NT header1 | **1.1229** | 0.292351 | 0.832403 | 0.662082 | 0.389855 | -0.004597 mPQ, -0.001182 detection, +0.100839 type | **Best balanced provisional point** |
| Full NT branch | 3.2483 | 0.293824 | 0.831731 | **0.668528** | **0.411968** | -0.003124 mPQ, -0.001854 detection, +0.107285 type | Non-dominated, but parameter-heavy |
| FullFT | 100.0000 | 0.305804 | 0.827083 | 0.668088 | 0.409712 | Reference | Non-dominated reference |

The currently selected strategy is no longer the unqualified best operating point once macro typing and Other matter. It remains the best two-seed detection-F1/bPQ-oriented LoRA+AF scope and the safest compact default. For a typing-aware paper claim:

- **NT header1** is the best observed overall balance, but only at seed 42.
- **Last stage** is the best already replicated typing-aware LoRA+AF choice: across seeds 42/43 it moves mean F1 type 0.578286 -> 0.654086 and Other 0.065043 -> 0.361733, while mean mPQ improves 0.293647 -> 0.295953. Costs are mean detection F1 0.834541 -> 0.825930, bPQ 0.504263 -> 0.502615, and trainable percentage 1.1176% -> 1.1674%.
- Conv adapters recover Other reproducibly but have a less attractive segmentation/parameter trade-off.
- Full NT gives the highest observed Other and type F1, but increasing trainable parameters from 1.1176% to 3.2483% is not justified by its small advantage over NT header1 without a second seed.

## Candidate follow-up strategies

### 1. LoRA+AdaptFormer + pre-final NT header block

- Classification: `EXISTING_TESTED_CONFIGURATION`
- Encoder: LoRA rank 8, alpha 8 on Q/V; AdaptFormer reduction 16 in encoder MLPs; encoder base frozen.
- Decoder: NP/HV bodies frozen; shared decoder frozen; train `nuclei_type_maps_decoder.decoder0_header.1` plus NP/HV/NT final heads.
- Heads: NP, HV, NT final heads and classifier head trainable.
- Known fraction: 7,945,835 / 707,644,395 = **1.1229%**.
- Rationale/failure targeted: concentrates 37,056 additional parameters immediately before NT logits; observed Other F1 0.389855 and F1 type 0.662082 while preserving F1 detection 0.832403.
- Compute relative to selected: trainable count **1.0047x** selected; wall-time/memory were not measured and must not be claimed equivalent.
- Exact test: repeat the existing KLT configuration unchanged except `random_seed: 43`, train ten epochs under the same sampling/optimizer/split, infer `checkpoint_10.pth` on the same test set, and predeclare comparison of mPQ, bPQ, detection F1, present F1 type, Other F1, and the Other confusion row against selected and FullFT seed 43.

### 2. LoRA+AdaptFormer + decoder last stage

- Classification: `EXISTING_TESTED_CONFIGURATION`
- Encoder: LoRA Q/V r8/a8 plus AdaptFormer reduction 16; encoder base frozen.
- Decoder: shared `decoder0` and all branch `decoder0_header` modules trainable; earlier decoder stages frozen.
- Heads: NP/HV/NT final heads and classifier trainable.
- Known fraction: 8,261,291 / 707,644,395 = **1.1674%**.
- Rationale/failure targeted: reproducibly recovers Other at seeds 42/43 and improves mean mPQ slightly; this is the strongest two-seed typing-aware evidence.
- Compute relative to selected: trainable count 1.0446x; observed detection cost is material (-0.008612 mean).
- Exact test needed: none to establish the existing KLT seed-42/43 result. If promoted in the paper, report both seeds and the detection trade-off; do not present it as preserving detection identically.

### 3. LoRA+AdaptFormer + full NT branch

- Classification: `EXISTING_TESTED_CONFIGURATION`
- Encoder: LoRA Q/V r8/a8 plus AdaptFormer reduction 16; encoder base frozen.
- Decoder: full NT branch trainable; NP/HV bodies and shared decoder frozen.
- Heads: NP/HV final heads, the NT head as part of the full NT branch, and classifier trainable.
- Known fraction: 22,986,219 / 707,644,395 = **3.2483%**.
- Rationale/failure targeted: highest observed Other F1 (0.411968) and present F1 type (0.668528) among PEFT, essentially matching seed-42 FullFT typing.
- Compute relative to selected: trainable count 2.9065x, still far below FullFT but much larger than NT header1.
- Exact test: seed-43 replication on the unchanged KLT protocol. It is lower priority than NT-header1 replication because the observed typing advantage over NT header1 is only 0.006446 macro-F1 and 0.022113 Other F1 for +2.1254 percentage points trainable.

No new untested architectural strategy is prioritized: the existing NT-header1 experiment already tests the most direct minimal typing-specific extension.

## One recommended experiment for the revised paper

**Run one seed-43 replication of the existing LoRA+AdaptFormer `nt_header1_np_hv_heads` KLT configuration.**

This is the highest-value experiment because the seed-42 result recovers Other from 0.002134 to 0.389855 and present F1 type from 0.561243 to 0.662082, while changing detection F1 by only -0.001182 and adding only 37,056 trainable parameters. It uses an implemented scope and requires no new architecture engineering. A second seed directly answers whether the exceptionally favorable typing/segmentation balance is reproducible.

Status: the **configuration is already tested at seed 42**, but the **recommended seed-43 experiment is untested and requires a new run**. Until that run exists, retain selected heads-only as the conservative detection-oriented method, report last-stage as the replicated typing-aware alternative, and frame NT header1 as the strongest provisional operating point rather than as a validated replacement.

## Source artifacts inspected

- Run configs, logs, inference logs, and `inference_results.json` under the exact run directories above.
- `reports/type_assignment/type_assignment_report.md` and its saved confusion/per-class CSVs, cross-checked by direct aggregation of each `image_metrics[*].detection_stats`.
- `models/adapters/utils.py` for exact scope feasibility and parameter-name selection.
- `cell_segmentation/experiments/experiment_cellvit_pannuke.py` for scope application.
- `configs/preprocessing_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128.yaml` and the converted dataset `dataset_config.yaml`, `cell_count_test.csv`.
- Local source dataset `STHELAR_40x/README.md` for label provenance and grouping semantics.

## Slide-independent reconciliation — 2026-08-27

The seed-43 recommendation above referred specifically to the historical
**within-slide** NT-header1 condition and is retained as provenance. It must not
be read as a current request to reopen the architecture search. The reciprocal
held-out-slide campaign now has valid NT-header1 seeds 42 and 43 on both folds,
with fixed epoch-10 inference, per-class supports, and saved confusion matrices.
The pre-selected heads-only method remains primary; NT-header1 is a prespecified
typing/error-analysis condition.

Across the two within-fold seed means, NT-header1 has Other F1 approximately
0.149 versus 0.005 for Selected PEFT, while Epithelial F1 is approximately
0.615 versus 0.680. Its overall F1 type is 0.500 versus 0.485, with lower mean
mPQ (0.218 versus 0.226) and slightly lower detection F1 (0.851 versus 0.854).
This supports the narrower claim that a small amount of NT-decoder capacity
changes the difficult Other-class trade-off; it does not support post-hoc
replacement of the selected method.

The held-out-slide shift claim is also supported: relative to the corresponding
historical within-slide rows, Selected-PEFT F1 detection changes by about +0.019
for each seed while F1 type decreases by 0.080/0.106; FullFT detection changes
by +0.016/+0.020 while F1 type decreases by 0.148/0.177. Fold effects are often
larger than seed effects for segmentation and detection, but that wording must
remain metric-specific: the Selected-PEFT and LP F1-type fold gaps are not
larger than their largest within-fold seed SD.
