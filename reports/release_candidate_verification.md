# STHELAR-Adapt release-candidate verification

Date: 2026-07-20  
Branch: `release/public-audit`  
Status: **technical release-candidate checks passed; publication remains blocked on manual licensing review**

No training, dataset-scale inference, GPU job, GitHub upload, Hugging Face upload, checkpoint deletion, or Git-history rewrite was performed. The `V100_RETEST` checkpoint was not opened or otherwise investigated during this phase. The 38 checkpoints marked `archive locally` in the audit manifest were not modified, moved, archived, or deleted.

## Outcome

Exactly the 12 rows marked `release` in `release/adapter_manifest.csv` were exported under stable public IDs. All 12 passed:

- CPU, weights-only `.pth` loading;
- strict top-level checkpoint-schema validation;
- required LoRA, AdaptFormer, NP, HV, and NT component checks;
- safetensors conversion and exact tensor-by-tensor round-trip equality;
- expected-key and expected-shape validation for 296 trainable state tensors and 105 mutable buffers;
- the declared trainable count of 7,908,779 parameters;
- the six-entry ordered mapping `Background=0`, `Immune=1`, `Stromal=2`, `Epithelial=3`, `Melanocyte=4`, `Other=5`;
- sanitization checks for absolute paths, personal paths, and cluster/user markers;
- reconstruction against the declared CellViT-SAM-H-x40 base checkpoint;
- zero missing and zero unexpected adapter parameter keys, and zero missing and zero unexpected declared mutable-buffer keys;
- one 256×256 CPU forward smoke test per adapter.

The frozen per-file hashes and results are in `release/huggingface/adapter_manifest_verified.csv` (SHA256 `3dff12b9d7c6f79ed99603b529b13d1cbb0e0172fc8b7b54bc61a56cf5b7ef2f`). Machine-readable details are in `release/huggingface/verification_summary.json`.

## Environment and base checkpoint

The existing local Python 3.9 project environment was used. Its local environment directory is named `cellvit39`; the public `environment.yml` now declares `name: sthelar-adapt`, so the documented activation command is `conda activate sthelar-adapt`.

| Component | Verified value |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu121 |
| safetensors | 0.7.0 |
| PyYAML | 6.0 |
| CUDA available in verification process | false |
| Forward device | CPU |

`safetensors` was initially absent from the environment even though it is now declared in `requirements.txt`; version 0.7.0 was installed into the existing environment before conversion. The README therefore installs it only through `requirements.txt`, not through a redundant standalone command.

The existing repository symlink resolved to a readable local base checkpoint:

| Item | Value |
|---|---|
| Required filename | `CellViT-SAM-H-x40.pth` |
| Size | 2,799,315,941 bytes |
| SHA256 | `b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf` |
| Copied into release tree | no |

Base loading finds 744 shape-compatible tensors. The upstream base has a 19-output tissue classifier, while these adapters declare one tissue output; `classifier_head.weight` and `classifier_head.bias` are therefore deliberately skipped and deterministically reinitialized for the transfer task. This expected base-head conversion is distinct from adapter loading: all 296 adapter parameter keys and all 105 declared mutable-buffer keys match exactly.

Every forward returned:

| Output | Shape |
|---|---|
| `tissue_types` | `[1, 1]` |
| `nuclei_binary_map` | `[1, 2, 256, 256]` |
| `hv_map` | `[1, 2, 256, 256]` |
| `nuclei_type_map` | `[1, 6, 256, 256]` |

## Canonical adapter set

| Public ID | Source selection | Result |
|---|---|---|
| `klt/seed42` | KLT selected PEFT, seed 42 | pass |
| `klt/seed43` | KLT selected PEFT, seed 43 | pass |
| `tissue_specific/breast/seed42` | Breast selected PEFT, seed 42 | pass |
| `tissue_specific/colon/seed42` | Colon paper-matching timestamped run, seed 42 | pass |
| `tissue_specific/kidney/seed42` | Kidney selected PEFT, seed 42 | pass |
| `tissue_specific/kidney/seed43` | Kidney selected PEFT, seed 43 | pass |
| `tissue_specific/liver/seed42` | Liver selected PEFT, seed 42 | pass |
| `tissue_specific/lung/seed42` | Lung paper-matching timestamped run, seed 42 | pass |
| `tissue_specific/ovary/seed42` | Ovary selected PEFT, seed 42 | pass |
| `tissue_specific/pancreatic/seed43` | Pancreatic selected PEFT, seed 43 | pass |
| `tissue_specific/skin/seed42` | Skin selected PEFT, seed 42 | pass |
| `tissue_specific/tonsil/seed43` | Tonsil selected PEFT, seed 43 | pass |

The exact source relative path, source size, source SHA256, matching config path and SHA256, generated `adapter_config.json` SHA256, safetensors size and SHA256, dtype distribution, and test state are frozen in the verified manifest. No validity decision relied on a filename alone.

## KLT final-head linear-probe config recovery

The recovered public config was compared recursively, field by field, with this exact original timestamped config:

`run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42/log/2026-07-02T153245_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42/config.yaml`

Original SHA256: `4831771b975aadbcc8be0a4ed89d52c21e7c41c78633da6316ad8f2c3b6ce58c`  
Recovered config SHA256: `7c9cdb184bc3a3e13735b507777ec568d70a761fcbe9c8cbb06c12f6b30191ff`

All scientific model, optimizer, scheduler, sampling, augmentation, checkpointing, seed, class-count, split, magnification, inference-threshold, and epoch fields are equal. There are no inferred scientific fields and no missing scientific fields.

The complete intentional difference set is:

| Field | Public-release treatment |
|---|---|
| `data.dataset_path` | absolute private path replaced by `${DATA_ROOT}/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128` |
| `logging.log_dir` | absolute timestamped runtime directory replaced by a portable relative log root |
| `logging.wandb_dir` | absolute runtime directory replaced by a portable relative directory |
| `logging.run_id` | removed; generated W&B runtime identity, not a training choice |
| `logging.wandb_file` | removed; generated W&B runtime identity, not a training choice |
| `agent` | removed; runtime-injected `null` value |
| `run_sweep` | removed; runtime-injected `false` value |
| `dataset_config` | removed; runtime-resolved dataset metadata, not an input field |

## Documentation and assets

- All references now use `configs/release/compayl2026/`.
- The root installation instructions use `conda activate sthelar-adapt`, matching `environment.yml`.
- The redundant standalone safetensors installation was removed.
- F1 type was added to the root KLT headline table: 0.430 final-head LP, 0.666 FullFT, and 0.578 selected PEFT.
- Both qualitative captions use: “Examples were selected for qualitative illustration and are not intended to constitute a statistically representative sample.”
- The publication architecture source is `figures/paper_figures/architecture.png`; byte-identical destinations live at `docs/figures/architecture.png` and `release/huggingface/figures/architecture.png`. The qualitative figure remains synchronized between `docs/figures/qualitative.png` and `release/huggingface/figures/qualitative.png`.
- All filesystem-relative links in the root README and standalone model card passed local-link validation.

## Existing diff and recommended commit split

The current worktree is intentionally not committed and history was not rewritten. Conceptually, the diff falls into three requested groups:

1. **Documentation, audit, and release metadata:** root/model-card READMEs, reports, curated release configs, figures, manifests, cluster-specific README notices, license/citation copies, results copies, and the non-binary metadata for the 12-package release tree.
2. **New adapter tools:** `tools/audit_adapter_checkpoints.py`, `tools/export_adapter_safetensors.py`, `tools/verify_released_adapter.py`, plus identical release-tree script copies.
3. **Runtime behavior changes:** environment-variable expansion and unresolved-placeholder errors in config loading/conversion; explicit leakage-check dataset arguments; safetensors package loading in the adapter hub; portable example/release config paths; and the declared safetensors dependency.

A clean commit sequence would be:

1. `fix(runtime): make STHELAR paths and adapter loading portable` — runtime Python changes, all portable-path YAML substitutions, and `requirements.txt`.
2. `feat(release-tools): add safe adapter audit export and verification` — the three canonical tools only.
3. `docs(release): add curated COMPAYL configs audits and figures` — GitHub README, reports, canonical figures, internal audit manifest, and cluster notices.
4. `chore(release): add verified Hugging Face release metadata` — `.gitignore` plus the standalone model-card tree's non-binary metadata, configs, scripts, results, and synchronized figures. Adapter safetensors, original PTH files, base checkpoints, caches, and the complete internal manifest are explicitly excluded.

The exact pathspecs, exclusions, commands, and expected status after each commit are recorded in `reports/public_release_commit_plan.md`. No commit has been created.

## Commands

The verification interpreter was the `python3.9` executable from the local `cellvit39` environment. Its private installation prefix is intentionally omitted from this public report; the invoked command bodies were:

```bash
python3.9 --version
python3.9 -c 'import torch, safetensors, yaml; print(torch.__version__, safetensors.__version__, yaml.__version__, torch.cuda.is_available())'
sha256sum models/pretrained/CellViT-SAM-H-x40.pth

python3.9 tools/export_adapter_safetensors.py \
  <source-adapter.pth> release/huggingface/adapters/<public-id> \
  --adapter-id <public-id> \
  --source-config <matching-config.yaml> \
  --base-checkpoint-sha256 b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf

python3.9 tools/verify_released_adapter.py \
  release/huggingface/adapters/<public-id> \
  --base-checkpoint models/pretrained/CellViT-SAM-H-x40.pth
```

The export and verification commands were run once for each public ID in the canonical table above. The converter refuses existing output files, so reruns must use a new empty destination rather than overwrite a frozen package.

For a new user, the portable setup and verification sequence is:

```bash
conda env create -f environment.yml
conda activate sthelar-adapt
python -m pip install torch
python -m pip install -r requirements.txt

python tools/verify_released_adapter.py release/huggingface/adapters/klt/seed42
python tools/verify_released_adapter.py release/huggingface/adapters/klt/seed42 \
  --base-checkpoint /path/to/CellViT-SAM-H-x40.pth
```

Dataset preparation, leakage checking, LP/PEFT/FullFT training, inference, and table-generation commands are documented in the root README. They were not run during release verification.

## Tests: passed, failed, and skipped

### Passed

- Python compilation of the canonical export and verification tools.
- Strict export of 12/12 release-marked PTH files; exactly 12 safetensors files exist in the candidate tree.
- Exact safetensors round-trip equality for 12/12 packages.
- Independent source, source-config, adapter-config, and safetensors SHA256 validation for 12/12 packages.
- Required component, key-set, shape, dtype, trainable-count, label-order, and path-sanitization checks for 12/12 packages.
- Independent scan of all 12 safetensors headers found zero private/absolute-path marker matches.
- Base checksum, reconstruction, exact adapter-state load, and 256×256 CPU forward for 12/12 packages.
- Recursive field comparison of the recovered LP YAML against its exact timestamped source.
- Root/Hugging Face README local-link resolution.
- Root/Hugging Face config-tree equality.
- Adapter-hub resolution of a packaged safetensors adapter and public metadata loading.

### Failed, then corrected and rerun

- The initial environment version probe failed because `safetensors` was not installed. Installing the already-declared dependency resolved it; version 0.7.0 was then recorded.
- The first KLT seed42 full verification completed model loading and forward execution but failed while formatting its result because `load_info` had been discarded in the verifier. The reporting bug was fixed; the complete KLT seed42 verification was rerun and passed. No adapter content changed because of this tool fix.

There are no unresolved adapter verification failures.

### Skipped by design

- Training of any kind.
- Full-dataset or dataset-scale inference and evaluation.
- Dataset-dependent preprocessing or leakage validation.
- GPU jobs and GPU forward tests (CUDA was unavailable in the verification process).
- Clinical, cross-site, or external-dataset validation.
- Uploads to GitHub or Hugging Face.
- Git commits, Git-history rewriting, checkpoint deletion, or cleanup of the 38 non-release checkpoints.

## Remaining release blockers

- Manually verify CellViT code/checkpoint, SAM-derived component, adapter-weight, and combined-work licensing with the relevant rights holders. STHELAR data being CC BY 4.0 does not determine the adapter license.
- Add the authoritative base-checkpoint download/release identifier and confirm that the recorded SHA256 corresponds to the distributable checkpoint users can lawfully obtain.
- Insert the final COMPAYL bibliographic record when public.
- Pin or hash the exact STHELAR dataset revision and source Parquet files if they can be recovered.
- Decide the publication mechanism for the 365 MiB adapter release tree. Do not put the full base checkpoint in either release.
