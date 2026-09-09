# Storage cleanup record — 2026-08-19

This is the execution record for the narrow deletion set approved after the
read-only audit in `reports/storage_audit.md`. No other item in `run/`, no
canonical run, adapter, release artifact, 40x dataset, `cellvit_ready` payload,
log tree, or brain data was deleted or altered.

## Before and after

| Measurement | Before cleanup | After cleanup |
|---|---:|---:|
| Ruche workdir quota | 420 GiB / 500 GiB (83%) | 388 GiB / 500 GiB (77%) |
| Ruche workdir files | 213,265 / 400,000 (53%) | 180,129 / 400,000 (45%) |
| `STHELAR-Adapt/run` | 282,053,447,168 bytes (about 263 GiB) | 255 GiB (`du -sh`, after new slide-experiment jobs had begun writing small outputs) |

The exact approved deletion set totalled **34,264,066,657 bytes (31.911
GiB)**: 8,678,543,969 bytes (8.083 GiB) inside `run/` and 25,585,522,688
bytes (23.828 GiB) in the three separately approved cluster directories.

## Deleted paths

| Deleted path | Pre-deletion bytes | Recorded modification time | Safety verification |
|---|---:|---|---|
| `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T070649_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN/checkpoints/model_best.pth` | 2,897,444,952 | 2026-06-26 14:38:35.433902474 +02:00 | Replacement attempt `2026-06-27T201436` rechecked: `model_best.pth`, `config.yaml`, `logs.log`, and `inference_results.json` all present. |
| `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T150300_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/checkpoints/model_best.pth` | 2,879,891,064 | 2026-06-26 17:36:29.918943078 +02:00 | Replacement attempt `2026-06-28T022341` rechecked with all four required files present. |
| `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T150300_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/checkpoints/model_best.pth` | 2,886,744,412 | 2026-06-26 17:43:43.990769470 +02:00 | Replacement attempt `2026-06-27T235829` rechecked with all four required files present. |
| `run/sthelar40x_bps_9class_slide_adaptformer_gelu_r16_e3` | 21,326 | 2026-06-10 17:42:21 +00:00 | Exact failed/debug directory identified by the audit. |
| `run/sthelar40x_tonsil_9class_slide_adaptformer_gelu_red16_lr5e-5_e15_seed42_CLEAN` | 21,471 | 2026-06-10 17:42:19 +00:00 | Exact failed/incomplete directory identified by the audit. |
| `run/sthelar40x_tonsil_9class_slide_fullft_lr5e-5_e1_seed42_TEST` | 14,420,744 | 2026-06-17 12:59:23 +00:00 | Exact one-epoch TEST/debug directory identified by the audit. |
| `${RUCHE_WORKDIR}/workspace/Datasets/STHELAR_20x` | 17,840,127,488 | 2026-04-11 17:21:34.280 +02:00 | Confirmed ordinary directory at the exact path, not a symlink; not referenced by the release KLT/40x configs. |
| `${RUCHE_WORKDIR}/workspace/Datasets/BioImageArchive` | 7,199,483,904 | 2026-04-03 11:28:21.590 +02:00 | Confirmed ordinary directory, not a symlink; no `brain_s0` entry; not referenced by release KLT/40x configs. |
| `${RUCHE_WORKDIR}/workspace/sthelar-release-smoke-<temporary>` | 545,911,296 | 2026-07-21 15:53:32.269411 +02:00 | Confirmed ordinary temporary clone at the exact approved path, not a symlink and not referenced by release KLT/40x configs. |

All nine paths were rechecked after the interrupted session and remain absent.

## Current dataset sizes

The required post-cleanup `du` check reported:

```text
310M  ${RUCHE_WORKDIR}/workspace/Datasets/CellViT_for_STHELAR_debug
51G   ${RUCHE_WORKDIR}/workspace/Datasets/STHELAR_40x
56G   ${RUCHE_WORKDIR}/workspace/Datasets/cellvit_ready
```

`STHELAR_20x` and `BioImageArchive` are absent as intended. The raw brain
SpatialData was not present and was not touched.
