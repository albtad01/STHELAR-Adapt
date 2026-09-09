# Public release audit

> Follow-up (2026-07-20): the adapter conversion, base reconstruction, and CPU forward checks that were pending during this audit are now complete for all 12 release candidates. See `reports/release_candidate_verification.md` and `release/huggingface/adapter_manifest_verified.csv`. Statements below about checks not yet run describe the original audit phase.

Audit date: 2026-07-20  
Branch: `release/public-audit`  
Scope: all 672 files tracked at audit start, the 66 reachable Git commits, and the local ignored `adapters/` tree. No training, dataset-scale inference, GPU job, large download, upload, deletion, or history rewrite was performed.

## Executive findings

The selected method is implemented and is now represented by curated KLT and tissue-specific configs: frozen CellViT-SAM-H x40 base weights, LoRA r8/a8 on Q and V, AdaptFormer reduction 16/GELU in encoder MLPs, trainable final NP/HV/NT heads, and a frozen decoder body (7,908,779 trainable parameters; 1.1176%). A missing KLT final-head linear-probe config was recovered from the exact local run and sanitized into `configs/release/compayl2026/klt/`.

The repository is not yet a fully self-contained reproduction package. The raw STHELAR data and CellViT base checkpoint are intentionally external, but their exact revisions/SHA256 values are not pinned. More importantly, ignored timestamped run directories and prediction-level outputs are needed to regenerate several final tables and figures. The checked-in aggregate tables are usable, but not every number can be regenerated from tracked inputs alone.

The local adapter audit found 51 structurally readable `.pth` files, including two same-named but tensor-different Colon seed-42 runs and two same-named but tensor-different Lung seed-42 runs. The paper values identify one of each; filename/config/seed alone do not. Twelve checkpoints form the proposed release set. See `reports/adapter_release_audit.md` and `release/adapter_manifest.csv`.

No token, API-key, password, private-key, credential filename, private service URL, or `.env` match was detected in the current tracked tree or across all 66 commits using the patterns recorded below. This is a pattern audit, not a guarantee. Personal and cluster filesystem paths do remain in inherited notebooks, historical scripts, generated analysis code, and embedded adapter metadata.

## 1. Repository structure

| Top-level path | Role and classification |
|---|---|
| `base_ml/` | Core inherited CellViT experiment/CLI/trainer infrastructure. Retain with upstream attribution. |
| `cell_segmentation/` | Core training and patch inference, mixed with inherited PanNuke/CoNSeP/MoNuSeg/StarDist/CPP-Net code and exploratory notebooks. STHELAR reuses the PanNuke experiment/dataloader path. |
| `configs/` | `release/compayl2026/` is the canonical public config set; `examples/` is mixed current/legacy material; `in_use/` is stale upstream/local material with misleading status naming. |
| `datamodel/` | Inherited CellViT WSI/patient data models; not used by the documented STHELAR patch workflow but potentially needed by upstream WSI inference. |
| `docs/` | Inherited CellViT documentation/assets plus the STHELAR architecture figure. Several assets are unused by the new root README. |
| `examples/` | Adapter metadata/loading examples. Useful, though three tissue-specific loaders duplicate the generic hub API. |
| `figures/` | Three selected tracked publication figures plus many ignored local variants. The tracked 3x9 qualitative grid is the nine-tissue figure requested for release. |
| `jeanzay/` | Cluster-specific SLURM examples. Newly marked optional and site-specific. |
| `logs_paper/` | 352 tracked generated text/log artifacts (over half the tracked tree). Valuable provenance, but unsuitable as the only machine-readable run ledger and likely better as an archival release artifact. |
| `models/` | Core CellViT/SAM architecture and PEFT modules. `models/pretrained/` contains no redistributed base checkpoint. Substantial portions are inherited. |
| `preprocessing/` | Inherited WSI patch extraction plus the core `preprocessing/sthelar/` Parquet-to-CellViT converter and inspection tools. |
| `reports/` | Selected metrics/tables and many generated or ignored diagnostic products. Mixes durable evidence with generated HTML, intermediate analysis, and stale reports. |
| `ruche/` | Ruche-specific SLURM launch scripts; newly marked optional. `legacy/` is provenance only. |
| `shell_commands/` | Ad hoc path-specific dataset transfer, symlink, and archive commands. Not part of a portable workflow; recommend archive/documentation rather than execution. |
| `utils/` | Core adapter checkpoint/loading code mixed with one-off experiment generation, analyses, plotting, rerun, and duplicated table/grid tools. |
| `adapters/` | Only `README_TEMPLATE.md` is tracked; 51 local `.pth` files and the local checksum file are ignored. Never commit the unreviewed `.pth` set wholesale. |
| `run/`, `logs/`, caches | Ignored generated/local state. Required for some provenance scripts but absent from a fresh clone. |

Root files: `README.md` is now the STHELAR-Adapt entry point; `README_CellViT.md` preserves upstream documentation; `LICENSE` is the inherited CellViT Apache-2.0-with-Commons-Clause notice; dependency definitions are split among `environment.yml`, `requirements.txt`, `requirements_complete.txt`, and `optional_dependencies.txt`; `hub.py` and `makefile` are inherited/maintenance utilities.

### Core, experimental, generated, obsolete, duplicated

- Core release paths: `preprocessing/sthelar/convert_hf_to_cellvit.py`, `cell_segmentation/run_cellvit.py`, the PanNuke experiment/trainer/inference path it invokes, `models/adapters/`, `utils/adapter_checkpoint.py`, `utils/cellvit_adapter_hub.py`, `utils/rerun_cellvit_inference.py`, leakage checking, curated release configs, final paper CSVs, and the three public adapter tools.
- Experiment-specific but reproducibility-relevant: Fisher diagnostics, QC sweep, pooled-three-class evaluation, type-assignment analysis, slide dominance, qualitative-grid generation, and ablation adapters/configs. Retain with an explicit `research/` or `experiments/` classification rather than calling them dead.
- Upstream CellViT: most of `base_ml/`, `datamodel/`, `models/segmentation/`, SAM encoder files, non-STHELAR datasets/inference, WSI patch extraction, upstream docs/assets, `hub.py`, and `README_CellViT.md`.
- Generated: `logs_paper/`, `reports/flake8/`, most CSV/TeX tables, plots, ignored `run/`, caches, local checkpoints, and local checksums. Keep only release evidence; archive the remainder outside the code tree or generate it from a manifest.
- Likely obsolete/stale: `configs/in_use/`, `configs/examples/legacy/`, `ruche/legacy/`, old PanNuke notebooks, `shell_commands/`, and root `configs/preprocessing_sthelar.yaml` / `configs/training_sthelar.yaml` now superseded by release configs. Do not delete until an owner confirms no unpublished workflow depends on them.
- Duplicated families: `collect_sthelar_run_tables.py` and `_v2.py`; `lora.py`, `lora2.py`, and `lora_ntonly.py`; Macenko v1/v2; `cell_detection.py`, `_256.py`, and `_mp.py`; several adapter exporters/load tests; `make_qualitative_grid.py` and `make_qualitative_grid_3x6.py`; three nearly identical tissue loader examples. Some variants encode experiments, so consolidate only after tests.

### Undocumented/unreachable paths

Before this audit the root README contained only a title and image, so nearly every executable was unreachable from a documented workflow. The revised README documents the release converter, leakage checker, KLT LP/PEFT/FullFT training, tissue config location, inference rerun, run collection, table generation, adapter conversion, verification, and loading.

Still outside the public workflow: all `shell_commands/`; `configs/in_use/`; legacy configs/SLURM; non-CellViT training entry points; dataset analysis notebooks; Fisher/pooled/QC/slide-dominance plotting; config-generator scripts; and WSI inference. Treat these as upstream or research extras. Several analysis scripts still contain fixed local roots and cannot run on a new checkout without editing.

### Misleading names

`CLEAN`, `CLEAN_TRUE`, `KEEPALL_FISHER`, `V100_RETEST`, `final`, `test`, `in_use`, `legacy`, and version suffixes occur across configs/run names/scripts. `CLEAN` is historical run nomenclature, not a cleanliness or release guarantee. `final` is especially unsafe because later regenerated tables differ. Public assets should use immutable method/tissue/seed IDs plus hashes; preserve old names only in provenance fields.

## 2. Reproducibility

| Goal | Status | Evidence and gap |
|---|---|---|
| STHELAR preprocessing | Partial | Converter and ten curated 40x configs exist; commands are now documented and environment placeholders expand. Missing exact Hugging Face dataset revision, input checksums, and a small fixture test. |
| Leakage-safe spatial splits | Partial/strong | Split algorithm, margin-128 configs, manifests produced by the converter, checker, and checked-in summary CSVs exist. A fresh clone lacks the datasets/manifests needed to rerun validation. “Leakage-safe” here means coordinate-separated within-slide patches, not patient/site-independent validation. |
| KLT LP / PEFT / FullFT | Partial | Exact LP seed-42 config added; selected PEFT and FullFT seed-42/43 configs exist. Requires external data/base weights/GPU. Environment is not fully locked, and exact base SHA is absent. No training was run in this audit. |
| Tissue-specific training | Partial | All nine preprocessing, FullFT, and selected PEFT config families exist. Published policy is seed42 except Pancreatic/Tonsil seed43, with Kidney PEFT averaged over 42/43. Exact Colon/Lung timestamped rerun identity was previously not encoded in configs. |
| Inference/evaluation | Partial | Training triggers inference and `rerun_cellvit_inference.py` is documented. Adapter reconstruction code exists. No tracked lightweight prediction fixture or model/base checkpoint means end-to-end inference cannot be tested here. |
| Reported tables/figures | Not fully reproducible from clone | Aggregate CSV/TeX/figures are tracked, but ignored `run/` directories, some prediction arrays, exact QC aggregation, and qualitative selections/inference are required. Table scripts partly rely on local generated inputs and some only cover subsets of tissues. |

The selected configuration is identifiable at the method level. It is not always unique at the run level: Colon seed42 and Lung seed42 each have two completed timestamped runs with identical run/config names but different tensors and test metrics. The paper values trace to Colon `2026-06-30T092331` (mPQ 0.204747...) and Lung `2026-07-01T025800` (mPQ 0.301539...). A public `results/run_provenance.csv` keyed by run ID, source config SHA, checkpoint SHA, dataset manifest SHA, selection criterion, and reported table cell is still needed.

The final KLT CSV traces seed-averaged method rows, and the tissue CSV records the seed policy. However, older notes/reports are stale: `reports/paper_tables/paper_table_notes.md` states an outdated Kidney recovery result; `reports/final_submission_sanity_check.md` itself records stale tissue CSVs and a possible two-vs-three-seed last-stage mismatch. Do not use filenames containing `final` as provenance without comparing content.

Dependency issues: PyTorch is intentionally installed separately; `safetensors` was missing from `requirements.txt`; both `opencv-python` and `opencv-python-headless` are pinned together; `environment.yml` pins Python/system libraries but comments out Python packages; and `requirements_complete.txt` represents a different, larger environment. Create one tested lock file per supported CPU/CUDA environment before claiming turnkey reproduction.

## 3. Hard-coded paths and personal information

The initial tracked scan found path/user matches across cluster and workstation prefixes and contributor usernames (overlapping sets). Author names/citations and source comments containing author names are legitimate attribution and should remain. Git author emails are normal repository metadata.

Low-risk fixes made:

- Curated release and STHELAR example configs now use `${STHELAR_ROOT}` / `${DATA_ROOT}`.
- Training and STHELAR preprocessing config loaders recursively expand variables and fail on unresolved `${...}` values.
- The leakage checker now requires `--dataset`; the STHELAR inspection help no longer advertises a personal volume.
- Ruche/Jean Zay directories are explicitly documented as optional site-specific examples.
- Safetensors export strips absolute local path values to basenames before writing public JSON/header metadata.

Remaining path-bearing categories:

- inherited notebooks and PanNuke/CoNSeP/MoNuSeg scripts with workstation examples/output;
- `configs/in_use/train_cellvit_sweep.yaml`;
- inherited one-off analysis utilities outside the supported complete-slide workflow;
- Ruche and `shell_commands/` scripts, where cluster paths are historical examples;
- local ignored adapter `.pth` metadata, which embeds `/gpfs/workdir/<user>/...` source/base/run paths.

Do not publish original `.pth` files without conversion/sanitization. Refactor remaining analysis tools to required CLI arguments if they are selected for the supported public workflow; otherwise move them to `docs/archive/` or an archival branch.

## 4. Security and privacy

Scans covered tracked Python, YAML, shell, Markdown, CSV, TeX, notebook JSON, logs/reports, filenames, and all reachable revisions. Patterns included AWS-style keys, GitHub classic/fine-grained tokens, Hugging Face tokens, OpenAI-style keys, W&B/HF environment assignments with literal values, password/API-key assignments, and private-key headers. Sensitive filenames included `.env`, credentials, secrets, tokens, passwords, and common SSH private-key names.

Result: **zero current-tree matches, zero history matches, and zero sensitive filenames** for those patterns. No tracked W&B URL, obvious internal/private host URL, or `.env` file was found. The configured Git remote is the public GitHub SSH URL. No actual patient identifier was identified; notebook matches concern generic patient/case field names or public benchmark data. STHELAR slide IDs such as `kidney_s0` appear to be dataset identifiers, not direct patient identifiers, but dataset maintainers must confirm de-identification and redistribution terms.

Important residual risks:

- Local ignored `.pth` metadata contains personal username/cluster paths. The exporter now sanitizes public output, but each exported JSON/header must be rescanned.
- Pattern scans cannot detect arbitrary high-entropy or unknown credential formats. Run a dedicated secret scanner (for example, Gitleaks) in CI before publication.
- Notebooks contain saved outputs and workstation paths. Clear outputs or archive notebooks before a clean public tag.
- If any later manual review finds a secret, record only a redacted prefix/location, rotate it first, then plan history cleanup. No history was rewritten here.

## 5. Licensing and attribution

The root `LICENSE` states CellViT's “Apache 2.0 with Commons Clause” terms and mandatory CellViT citation. `README_CellViT.md` is the upstream README and preserves CellViT attribution, checkpoint warnings, citations, and license link. SAM-derived encoder files retain Meta copyright/license headers. The revised root README preserves CellViT, SAM, and STHELAR citations.

Recommendation: retain `README_CellViT.md` at the root for the first public release because it is a conspicuous preserved upstream notice and current links refer to it. Later, it may move unchanged to `docs/upstream_cellvit.md` if the root README and license clearly link to it and Git history preserves the move. Do not replace it with only a short summary until legal/manual review confirms which notices must accompany the inherited files.

Manual questions before publishing adapters:

1. Does the CellViT Apache-2.0-with-Commons-Clause grant/condition permit distribution of adapter weights trained against CellViT-SAM-H x40, and what exact notices must accompany them?
2. Which license and terms govern the CellViT-SAM-H x40 checkpoint itself, and may users obtain/use it for this adapter workflow? Record its authoritative URL, version, and SHA256; do not redistribute it here.
3. Do SAM's code/model terms impose requirements on the combined architecture or adapter weights?
4. Do STHELAR data terms permit trained-weight publication and the shown qualitative patches? CC BY 4.0 for data does **not** automatically license code or weights.
5. Who owns the new adaptation code and adapter weights, and who can choose their license? The current inherited `LICENSE` cannot answer that.
6. Does the COMPAYL paper/figure license permit redistributing all figures in the model card?

Do not label the Hugging Face adapter repository CC BY 4.0 by inference from the dataset license.

## 6. Code quality

High-priority issues:

- Run identity is encoded in long mutable names rather than a stable manifest; same config/name/seed produced different final checkpoints.
- The adapter loading API previously recognized `adapter_config.yaml` only; it now recognizes the new JSON export metadata too.
- Existing `.pth` loaders outside the new exporter often call `torch.load` without `weights_only=True`. Restrict untrusted checkpoint loading; `.pth` is executable pickle.
- Several scripts import/write at module-level constants and fixed paths, especially table/config generators.
- There is no focused unit-test suite for split assignment, adapter insertion/trainability, adapter schema, metadata sanitization, or model loading.
- `run_cellvit.py` maps STHELAR to a class named `ExperimentCellVitPanNuke`, which is functional reuse but confusing public naming.
- Generated logs/reports dominate the tracked tree and make stale/authoritative artifacts hard to distinguish.

Recommended disposition:

- Retain core upstream code and research-specific ablations until regression tests exist.
- Consolidate exporter/load utilities around `tools/export_adapter_safetensors.py`, `tools/verify_released_adapter.py`, and `utils/cellvit_adapter_hub.py` after the release format is tested.
- Move `configs/in_use`, legacy cluster/config files, notebooks with outputs, `shell_commands`, old generated flake8 HTML, and superseded reports to `docs/archive/` or a separate provenance archive after manual approval.
- Keep only immutable final CSVs/figures in Git; publish full logs/predictions as a versioned artifact with hashes if required for paper reproducibility.
- Rename future runs using compact IDs and a manifest; do not rename historical sources before recording mappings.

No uncertain file was deleted.

## 7. Lightweight checks

Performed without dataset, GPU, base checkpoint, network, or large download:

```text
git status --short --branch
git ls-files
git grep path/privacy/security patterns
git rev-list --all  # 66 commits
git grep per revision for secret patterns
python3 tools/audit_adapter_checkpoints.py adapters --output release/adapter_manifest.csv
python3.11 -m compileall -q base_ml cell_segmentation configs datamodel models preprocessing utils examples tools
bash -n <each tracked .sh file>
git diff --check
```

Results: Python 3.11 compilation passed; all discovered shell scripts passed `bash -n` (0 failures); 266 local YAML files parsed with Ruby's standard YAML parser; all 12 selected release configs passed method/rank/target/reduction/freeze/class-count invariants; root/model-card local Markdown links resolved; manifest counts/actions passed (51 audited, 12 candidates); no bare `DATA_ROOT/...` or `STHELAR_ROOT/...` placeholders remain in release/STHELAR example configs; and `git diff --check` passed. The structural adapter audit succeeded for all 51 `.pth` files using a restricted metadata-only parser; it did not execute checkpoint pickle globals or compare floating-point tensors. Full PyTorch/safetensors export, round-trip equality, base-model state loading, forward inference, training, dataset preprocessing, and dataset-backed leakage checks were not run because this environment lacks PyTorch/safetensors/data/base weights and those operations are outside the audit scope.

## Release gates

Before publication: resolve licensing; pin dataset/base hashes; convert and verify the 12 canonical adapters in a trusted PyTorch environment; add their exported hashes to the manifest; rescan exports; create a run-provenance ledger; add fixture tests/CI; correct the architecture figure wording/taxonomy; and regenerate every claimed table from versioned source artifacts.
