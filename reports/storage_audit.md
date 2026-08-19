# Read-only storage audit

Audit date: 2026-08-18. No file or directory was deleted, moved, renamed, or modified during inspection. The only new artifacts are this report, `reviewer_experiment_plan.md`, and `run_storage_inventory.csv`.

## Summary

`run/` currently contains 131 top-level run directories and occupies 282,053,447,168 physical bytes by `du` (262.683 GiB; 282.053 GB decimal). The generated per-directory inventory totals 282,049,644,429 apparent bytes (262.679 GiB); the small difference is filesystem allocation/directory accounting.

The high-confidence proposed reclaim set is deliberately narrow: **8,678,543,969 bytes (8.083 GiB; 8.679 GB decimal)**. It consists of three superseded checkpoints from interrupted/partial attempts and three failed/debug run directories. The three parent KLT runs containing superseded checkpoints remain `KEEP_CRITICAL` because they also contain canonical completed attempts. Nothing was classified `CANDIDATE_DELETE` merely because of age.

## Measured storage footprint

These are read-only `du -sB1` measurements taken during this audit.

| Path | Bytes | GiB | Notes |
|---|---:|---:|---|
| `STHELAR-Adapt/run` | 282,053,447,168 | 262.683 | Dominant repository storage; inventoried row by row |
| `Datasets/cellvit_ready` | 59,223,767,552 | 55.156 | Generated packed datasets and split metadata |
| `Datasets/STHELAR_40x` | 53,856,258,560 | 50.158 | Source 40x data/metadata |
| `Datasets/STHELAR_20x` | 17,840,127,488 | 16.615 | Source 20x data |
| `Datasets/BioImageArchive` | 7,199,483,904 | 6.705 | Source/archive data |
| `CellViT_for_STHELAR` | 7,796,354,048 | 7.261 | Base project/checkpoints |
| `STHELAR-Adapt/logs` | 4,375,128,064 | 4.075 | SLURM stdout/stderr; retained for provenance and timing reconstruction |
| `STHELAR-Adapt/adapters` | 1,510,545,920 | 1.407 | Adapter archive; release manifest sources |
| `STHELAR-Adapt/release` | 391,539,712 | 0.365 | Public release, including 12 verified Hugging Face adapters |

The raw brain SpatialData was not found or assumed to be present on Ruche. Temporary `sthelar-release-smoke-*` clones were not included in `run/` and were not altered.

## Inventory method and classification policy

`run_storage_inventory.csv` has one row for every immediate child directory of `run/`. Size is the current apparent byte total of files and directories; modification date is the latest observed mtime in the run. Checkpoint and inference-result paths are enumerated. Experiment family, method, tissue, and seed are inferred from the run name and saved configs, then cross-checked against:

- `release/adapter_manifest.csv`;
- `release/huggingface/adapter_manifest_verified.csv`;
- final paper tables and type-assignment summaries under `reports/`;
- qualitative selection reports and utilities.

The policy was conservative:

- `KEEP_CRITICAL`: any release-manifest source, any of the 12 public adapter sources, canonical KLT runs, canonical tissue-specific PEFT/FullFT runs, paper-table sources, or qualitative/QC sources.
- `KEEP_USEFUL`: noncanonical checkpoints that are expensive to reproduce, or inference results whose checkpoint is no longer present.
- `REGENERABLE`: only for an artifact proven reproducible from a preserved canonical source. No whole run directory met that standard strongly enough, so there are zero directory-level rows in this class.
- `CANDIDATE_DELETE`: only an explicit failed/debug directory or a superseded partial checkpoint with a later completed attempt in the same run.
- `UNKNOWN`: configuration/log-only cases for which the audit could not establish safety.

### Directory-level totals

| Classification | Directories | Apparent bytes | GiB |
|---|---:|---:|---:|
| `KEEP_CRITICAL` | 69 | 244,067,528,502 | 227.306 |
| `KEEP_USEFUL` | 56 | 37,941,042,067 | 35.335 |
| `REGENERABLE` | 0 | 0 | 0.000 |
| `CANDIDATE_DELETE` | 3 | 14,463,541 | 0.013 |
| `UNKNOWN` | 3 | 26,610,319 | 0.025 |
| **Total** | **131** | **282,049,644,429** | **262.679** |

The `KEEP_CRITICAL` directory total includes 8,664,080,428 bytes in three superseded partial checkpoints because their parent directories must remain critical. After carving out only those proposed files, the preservation-adjusted critical payload is **235,403,448,074 bytes (219.237 GiB)**. The unadjusted 227.306 GiB figure is the direct sum of `KEEP_CRITICAL` rows in the CSV and is the appropriate conservative “keep these directories intact” total.

## Exact proposed reclaim set

These are proposals for a later, separately approved cleanup. No deletion was performed.

### Superseded partial-attempt checkpoints inside critical runs

1. `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T070649_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN/checkpoints/model_best.pth` — 2,897,444,952 bytes. The attempt stopped before final inference; the same parent contains the completed `2026-06-27T201436` attempt used by the type summary. Preserve both attempts' configs/logs and the completed checkpoint/results.
2. `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T150300_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/checkpoints/model_best.pth` — 2,879,891,064 bytes. This partial attempt reached only early epochs and has no final inference; the `2026-06-28T022341` attempt completed and supplies the preserved result.
3. `run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T150300_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/checkpoints/model_best.pth` — 2,886,744,412 bytes. This partial attempt has no final inference; the `2026-06-27T235829` attempt completed and supplies the preserved result.

Subtotal: **8,664,080,428 bytes (8.069 GiB)**.

Safety condition: before any later deletion, verify again that the named completed attempt, its `model_best.pth`, `config.yaml`, `logs.log`, and `inference_results.json` are present, and archive the partial logs/configs. The proposed checkpoint files are not byte-identical copies of the completed checkpoints; they are regenerable/superseded states, not deduplicated aliases.

### Failed/debug run directories

1. `run/sthelar40x_bps_9class_slide_adaptformer_gelu_r16_e3` — 21,326 bytes. Only a config, dataset manifest, and short failed/incomplete log; no checkpoint or inference result; not referenced by either adapter manifest or a paper table.
2. `run/sthelar40x_tonsil_9class_slide_adaptformer_gelu_red16_lr5e-5_e15_seed42_CLEAN` — 21,471 bytes. Only a config, dataset manifest, and short failed/incomplete log; no checkpoint or inference result; not a canonical tissue-specific 5-class run.
3. `run/sthelar40x_tonsil_9class_slide_fullft_lr5e-5_e1_seed42_TEST` — 14,420,744 bytes. Explicit one-epoch `TEST`/debug run with no checkpoint; its TensorBoard/inference artifacts are noncanonical and regenerable.

Subtotal: **14,463,541 bytes (0.013 GiB)**.

Combined proposed reclaim: **8,678,543,969 bytes (8.083 GiB)**.

## Preserved critical material

The following categories are explicitly protected as `KEEP_CRITICAL` in the CSV:

- all KLT FullFT, final-head LP, frozen/raw-style, PEFT, and final ablation runs with canonical checkpoint/results;
- all nine canonical tissue-specific FullFT runs and all canonical selected LoRA Q/V + AdaptFormer + final-head runs, including seeds/retests needed to explain published selection;
- every run name referenced by `release/adapter_manifest.csv`;
- all 12 source runs identified by `release/huggingface/adapter_manifest_verified.csv` (KLT seeds 42/43 and the ten listed tissue/seed releases, including both Kidney seeds);
- checkpoints and prediction/QC artifacts referenced by paper tables, type-assignment summaries, or qualitative selection code;
- all configs/logs needed to reconstruct the submitted paper and public adapter provenance.

Equal file size was not treated as duplication. In particular, Kidney seed-43, Colon seed-42, Lung seed-42, and V100/A100 retest checkpoints are distinct trained states or have different publication/qualitative provenance and remain protected. The existing adapter audit reports no exact file or normalized tensor-state duplicates among the 51 adapter archives.

## Intermediate checkpoints, duplicates, caches, and temporary artifacts

- The current `run/` tree contains `model_best.pth` files but no remaining `checkpoint_*.pth` files. Older `reports/checkpoint_cleanup/pth_inventory_current.txt` and `reports/deleted_checkpoints_list.txt` describe a prior state and must not be used as a current deletion list.
- No current checkpoint hardlinks were found in the live inventory. Same-sized checkpoints were not assumed to be duplicates; a complete cryptographic hash of all full training checkpoints was not rerun because it would scan hundreds of gigabytes and would not, by itself, establish scientific redundancy.
- The only high-confidence redundant model states found are the three partial-attempt checkpoints itemized above.
- Inference JSON/CSV files are generally tiny relative to checkpoints and are often the only surviving evidence for runs whose checkpoint has already been removed. They were retained unless part of the explicit one-epoch TEST directory.
- TensorBoard event files and progress-heavy SLURM stderr logs may be mechanically regenerable, but `logs/` is only 4.075 GiB and provides job/timing/failure provenance needed for the reviewer audit. No log-wide deletion is proposed.
- No cache or temporary subtree inside `run/` was both large and provably detached from a canonical run. Temporary smoke clones outside `run/` require a separate path-specific audit before cleanup.

## Explicit uncertainty warnings

Three runs remain `UNKNOWN` rather than deletion candidates:

- `run/sthelar40x_liver_5class_spatial_vit256_lora_adaptformer_r8_a8_gelu_red16_lr5e-5_e10_seed42_CLEAN` (5,113,986 bytes): no current checkpoint, but its completed log is important evidence that the CellViT-256 adapter path works and is directly relevant to the requested baseline extension.
- `run/sthelar40x_tonsil_9class_slide_lora_ntonly_r4_a4_lr5e-5_e15_seed42_CLEAN` (8,151,023 bytes): config/log-only state with unclear historical use.
- `run/sthelar40x_tonsil_9class_slide_lora_r8_a8_lr5e-5_e15_seed42_CLEAN` (13,345,310 bytes): config/log-only state with unclear historical use.

Do not delete these without the experiment owner's confirmation. Likewise, the 56 `KEEP_USEFUL` exploratory runs are not paper-critical by the evidence located, but their checkpoints or unique inference results are costly or impossible to recreate exactly. A future cleanup should first obtain an owner-approved canonical-run registry, verify external backups and adapter export hashes, and only then consider promoting individual rows to `REGENERABLE` or `CANDIDATE_DELETE`.

## Recommended cleanup procedure (not executed)

1. Freeze this CSV and the two adapter manifests as the audit snapshot.
2. Obtain explicit owner approval for each of the six paths above.
3. Recheck completed replacement checkpoints/results and public adapter hashes.
4. Copy the partial attempt's small config/log provenance to an archival bundle if desired.
5. Measure free space before and after any future removal and record every exact path.

Until those checks are approved and performed, the reclaim figure is a proposal, not available space.
