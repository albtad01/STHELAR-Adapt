# STHELAR-Adapt cluster migration snapshot

Snapshot date: 2026-08-28 (Europe/Paris)

## Source and purpose

- Source branch: `sthelar-adapt`
- Source commit: `579cc5e4a0e7d84789d83609a757c3c666b7d726`
- Migration branch: `slideind-repro`
- Purpose: preserve a lightweight, reproducible source/config/manifest/report
  snapshot for continuing the revised STHELAR-Adapt experiments on another
  cluster. No job was launched and no scientific result was rewritten while
  preparing the snapshot.

The snapshot represents the historical within-slide KLT PEFT selection,
historical nine-tissue specialization, reciprocal slide-independent KLT Fold
A/B, CellViT-SAM-H Frozen/LP/Selected PEFT/NT-header1/FullFT, CellViT-256
Frozen/LP/Selected PEFT/FullFT, the matched A100 efficiency/deployment audit,
typing/per-class/per-slide/confusion analyses, the nine-tissue Fold-A
specialist campaign, and the available reciprocal tissue-specific Fold-B
runs.

The primary method is CellViT-SAM-H Selected PEFT: LoRA on Q and V with rank 8
and alpha 8, AdaptFormer reduction 16 with GELU, and the final NP/HV/NT heads.

## External assets not stored in Git

Git intentionally excludes all model checkpoints, dataset payloads, run
directories, optimizer state, prediction JSON dumps, logs, WandB/cache files,
and generated confusion-matrix PNGs. In particular, copy or lawfully recover:

| External asset | Expected location relative to an environment root | Size / identity |
|---|---|---|
| CellViT-SAM-H x40 base | `${PROJECT_ROOT}/models/pretrained/CellViT-SAM-H-x40.pth` | 2,799,315,941 bytes; SHA256 `b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf` |
| CellViT-256 x40 base | `${PROJECT_ROOT}/models/pretrained/CellViT-256-x40.pth` | 187,224,155 bytes; SHA256 `ee3986922fc500353db3d7692c566e19e1c694c8f10e47cf49dfd90160fc3b2b` |
| STHELAR 40x source data | `${STHELAR_ROOT}` | External Hugging Face/authorized source payload; not redistributed |
| Packed CellViT datasets | `${DATA_ROOT}/sthelar40x_*` | ZIP/CSV payloads and materialized split tables; not stored in Git |
| Canonical training state | `${PROJECT_ROOT}/run/.../checkpoints/checkpoint_10.pth` | Must be copied from retained runs if training is to be resumed rather than restarted |

The SAM-H retrieval instructions and checksum command are in `README.md`.
The exact immutable upstream release identifier for that checkpoint is not
known. A public retrieval identifier for the audited CellViT-256 checkpoint is
also not recorded; obtain it from an authorized official source or transfer
the checksum-verified local file. Do not substitute a checkpoint solely on
the basis of its filename.

The following audited local files were deliberately preserved but not included
in the snapshot:

| Local path or group | Classification | Reason |
|---|---|---|
| `reports/storage_cleanup_*.md` | C — cluster-specific | Old-cluster storage-operation journals; not scientific interpretation |
| `ruche/slurm_train_checkpoint10_only.sh` and `ruche/slurm_inference_checkpoint10_retention.sh` | C — cluster-specific | Hard-code the old partition/module environment; the portable retention implementation is committed in `utils/finalize_checkpoint10_retention.py` |
| `reports/neurips2026_paper_results/confusion_matrices/**/*.png` | D — generated binary | Reproducible from the committed raw/normalized CSVs and intentionally ignored |
| `reports/slide_exp_efficiency.csv` | F — superseded intermediate | Contains an earlier partial seed-43 collection; the reconciled `reports/efficiency_audit/` tables are canonical |
| `reports/tissue_specific_foldA_within_vs_slideind.csv` | F — superseded intermediate | Earlier snapshot omits the final Breast row; use `reports/tissue_specific_foldA_results_20260826.md` |
| Old-cluster checkpoint symlinks below ignored `models/pretrained/` and `run/` | C/D — cluster-specific heavy asset | Targets resolve to old account paths and are not portable Git content |

No candidate credential, private key, API token, or other category-E secret was
identified. No manuscript LaTeX source was added.

## Dataset layout and split reconstruction

Set `${DATA_ROOT}` to a cluster-local `cellvit_ready` directory. Each packed
source dataset used by the metadata-only fold generator must contain:

- `images.zip`, `labels.zip`, `types.csv`, and `dataset_config.yaml`;
- `patch_info_with_split.csv`;
- `cell_count_train.csv`, `cell_count_valid.csv`, and `cell_count_test.csv`.

The fold generator reuses the four packed payload files by symlink and creates
new train/validation/test metadata. Symlinks copied verbatim from the old
cluster will be broken; regenerate them on the destination cluster. For
example:

```bash
export PROJECT_ROOT="$PWD"
export DATA_ROOT=/path/to/cellvit_ready
export STHELAR_ROOT=/path/to/STHELAR_40x

python utils/generate_slide_independent_fold.py \
  --config configs/slide_exp/preprocessing/preprocessing_sthelar40x_klt_5class_slideind_foldA_margin128.yaml
python utils/generate_slide_independent_fold.py \
  --config configs/slide_exp/preprocessing/preprocessing_sthelar40x_klt_5class_slideind_foldB_margin128.yaml
```

Run the analogous committed preprocessing config for each tissue direction.
Compare the emitted `split_manifest.yaml` and `split_validation.json` with
`manifests/slide_independent/`. All 20 recorded validations (KLT plus nine
tissues, two directions each) report exact slide assignments, disjoint
train/test and validation/test slides, unique patch identifiers across splits,
and `all_checks_passed: true`. Do not edit membership to reconcile a mismatch.

## Experiment status at snapshot

| Family | Status on 2026-08-28 |
|---|---|
| Historical within-slide KLT and nine-tissue experiments | Completed for the canonical reported rows; heterogeneous historical seed/checkpoint caveats are documented in the result summary |
| CellViT-SAM-H KLT Fold A/B | Completed: Frozen; LP, Selected PEFT, NT-header1, and FullFT seeds 42/43 |
| CellViT-256 KLT Fold A/B | Completed: Frozen, LP, Selected PEFT, and FullFT seed 42 |
| Matched A100 efficiency/deployment audit | Completed for seed-42 Fold A/B |
| Typing/per-class/per-slide/confusion analysis | Completed for the canonical SAM-H analysis; additional lightweight-backbone/tissue tables remain optional existing-output analyses |
| Nine-tissue specialist Fold A | Completed for all nine tissues, Selected PEFT seed 42 |
| Reciprocal tissue Fold B | Completed for Breast, Pancreatic, Skin, Tonsil, Ovary, Colon, and Lung; Kidney and Liver remain missing after infrastructure failures |
| SAM-H seed 44 | Missing; no validated seed-44 config existed in the audited worktree, so none was fabricated |
| CellViT-256 seeds 43/44 | Missing; no validated configs existed in the audited worktree, so none was fabricated |

Ovary job 1501898, Colon job 1501899, and Lung job 1501917 changed from the
2026-08-27 registry's RUNNING state to Slurm `COMPLETED`/exit `0:0`. Read-only
artifact checks found the retained epoch-10 checkpoint, final inference JSON,
both completed efficiency records, and retention metadata for each. No job was
submitted during this migration audit.

See `reports/paper_future_experiments.md` for the prioritized missing work and
`reports/neurips2026_paper_results/final_results_summary.md` for canonical
result interpretation.

## Bootstrap assumptions

```bash
git clone <repository-url> STHELAR-Adapt
cd STHELAR-Adapt
git switch slideind-repro

conda env create -f environment.yml
conda activate sthelar-adapt
python -m pip install -r requirements.txt

export PROJECT_ROOT="$PWD"
export DATA_ROOT=/path/to/cellvit_ready
export STHELAR_ROOT=/path/to/STHELAR_40x
mkdir -p models/pretrained
sha256sum models/pretrained/CellViT-SAM-H-x40.pth \
  models/pretrained/CellViT-256-x40.pth
```

Install PyTorch separately for the destination CUDA stack before the remaining
requirements. Canonical completed runs recorded Python 3.9.25, PyTorch
2.5.1+cu121, CUDA 12.1, FP16 autocast, and NVIDIA A100-SXM4-40GB. The checked-in
`environment.yml` pins the project to Python 3.9.7; use a compatible Python 3.9
environment if that exact build is unavailable. Validate imports, YAML
expansion, checkpoint hashes, dataset manifests, GPU type/memory, and scheduler
resource directives before any submission. The old Ruche partition/module
wrappers are intentionally not part of this portable snapshot.

All `${PROJECT_ROOT}`, `${DATA_ROOT}`, and `${STHELAR_ROOT}` placeholders are
expanded by the project config loaders and fail when unresolved. Keep raw data,
materialized datasets, runs, logs, and checkpoints outside Git.

## Public provenance note

Newly committed reports contain no personal account path or direct personal
repository/model namespace. Existing repository history, remote configuration,
and upstream source attribution predate this migration and can identify
contributors; history was intentionally not rewritten. The result-package
directory and generator retain the historical venue-oriented name
`neurips2026_paper_results` for provenance, and job registries retain scheduler
IDs and old cluster/node names. These names are preserved to avoid breaking
recorded paths and do not identify a manuscript submission.
