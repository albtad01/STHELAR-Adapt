# Public release commit plan

Prepared: 2026-07-21  
Branch: `release/public-audit`  
State: plan only—no files have been staged or committed

This plan deliberately separates runtime behavior, release tooling, GitHub documentation/audit material, and the standalone Hugging Face metadata tree. Run each status/diff review before its commit. Do not replace the exact pathspecs with `git add .`, `git add -A`, or `git add release/`.

## Global exclusions

Every commit excludes:

- all `*.safetensors`, `*.pth`, `*.pt`, and `*.ckpt` files;
- `models/pretrained/` and every full CellViT base checkpoint;
- `release/huggingface/.cache/` and `release/huggingface/cache/`;
- local `adapters/`, `run/`, `wandb/`, checkpoints, logs, and dataset files;
- `figures/paper_figures/architecture.png` and `figures/paper_figures/qualitative.png` (duplicate/superseded source copies pending manual archival approval);
- `release/huggingface/figures/sthelar-adapt.png` and `release/huggingface/figures/qualitative_grid_3x9_supplement.png` (superseded HF-tree copies pending manual cleanup approval);
- the removed redundant `release/huggingface/adapter_manifest.csv`;
- any unrelated pre-existing worktree change not named below.

The complete 51-checkpoint audit manifest stays in the GitHub repository as `release/adapter_manifest.csv`; it must not be copied into the standalone Hugging Face tree.

## Commit 1

### `fix(runtime): make STHELAR paths and adapter loading portable`

Why: these files implement environment-variable expansion and validation, replace machine-specific paths with `${DATA_ROOT}`/`${STHELAR_ROOT}`, make CLI dataset paths explicit, support JSON/safetensors adapter packages, and declare the required safetensors dependency. All YAML substitutions that depend on runtime expansion belong here, including the recovered LP config.

Exact files:

```text
base_ml/base_cli.py
preprocessing/sthelar/convert_hf_to_cellvit.py
preprocessing/sthelar/inspect_dataset.py
utils/cellvit_adapter_hub.py
utils/check_spatial_split_leakage.py
requirements.txt
configs/examples/preprocessing_sthelar20x_5class.yaml
configs/examples/preprocessing_sthelar20x_9class.yaml
configs/examples/preprocessing_sthelar20x_cancer_normal.yaml
configs/examples/training_sthelar20x_5class.yaml
configs/examples/training_sthelar20x_9class.yaml
configs/examples/training_sthelar20x_cancer_normal.yaml
configs/examples/training_sthelar40x_tonsil_9class_slide_lora_r8_a8_lr5e-5_e15_seed42_CLEAN.yaml
```

Also include all 34 paths formed by prefixing the [curated config inventory](#curated-config-inventory) with `configs/release/compayl2026/`.

Exact staging commands:

```bash
git add -- \
  base_ml/base_cli.py \
  preprocessing/sthelar/convert_hf_to_cellvit.py \
  preprocessing/sthelar/inspect_dataset.py \
  utils/cellvit_adapter_hub.py \
  utils/check_spatial_split_leakage.py \
  requirements.txt

git add -- \
  configs/examples/preprocessing_sthelar20x_5class.yaml \
  configs/examples/preprocessing_sthelar20x_9class.yaml \
  configs/examples/preprocessing_sthelar20x_cancer_normal.yaml \
  configs/examples/training_sthelar20x_5class.yaml \
  configs/examples/training_sthelar20x_9class.yaml \
  configs/examples/training_sthelar20x_cancer_normal.yaml \
  configs/examples/training_sthelar40x_tonsil_9class_slide_lora_r8_a8_lr5e-5_e15_seed42_CLEAN.yaml

git add -- configs/release/compayl2026
git diff --cached --check
git diff --cached --name-status
git commit -m "fix(runtime): make STHELAR paths and adapter loading portable"
```

Explicitly excluded: root/HF READMEs, audits, figures, tools, `.gitignore`, `release/adapter_manifest.csv`, and the entire `release/huggingface/` tree.

Expected status afterward: the runtime Python files, `requirements.txt`, seven example YAMLs, and all root curated release YAMLs are clean. Documentation, tools, `.gitignore`, GitHub audit/release files, HF metadata, and the intentionally untracked duplicate figures remain.

## Commit 2

### `feat(release-tools): add safe adapter audit export and verification`

Why: these are the canonical repository implementations for restricted structural auditing, CPU/weights-only conversion to safetensors, metadata sanitization, hash generation, strict package validation, model compatibility, and optional forward smoke testing.

Exact files:

```text
tools/audit_adapter_checkpoints.py
tools/export_adapter_safetensors.py
tools/verify_released_adapter.py
```

Exact staging and commit commands:

```bash
git add -- \
  tools/audit_adapter_checkpoints.py \
  tools/export_adapter_safetensors.py \
  tools/verify_released_adapter.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat(release-tools): add safe adapter audit export and verification"
```

Explicitly excluded: the duplicate standalone-HF script copies, every adapter package/weight, all manifests, configs, reports, and documentation.

Expected status afterward: `tools/` is clean. Only documentation/audit assets, `.gitignore`, HF metadata, and intentionally untracked duplicate figures remain.

## Commit 3

### `docs(release): add curated COMPAYL configs audits and figures`

Why: this is the GitHub-facing documentation/provenance layer: concise user guidance, canonical GitHub figures, audit findings, frozen internal checkpoint inventory, cluster-context notices, verification record, and this reviewable commit plan. The curated root configs were committed with runtime portability in Commit 1 because their placeholders depend on that code.

Exact files:

```text
README.md
docs/figures/architecture.png
docs/figures/qualitative.png
jeanzay/README.md
ruche/README.md
release/adapter_manifest.csv
reports/adapter_release_audit.md
reports/public_release_audit.md
reports/release_candidate_verification.md
reports/public_release_commit_plan.md
```

Exact staging and commit commands:

```bash
git add -- \
  README.md \
  docs/figures/architecture.png \
  docs/figures/qualitative.png \
  jeanzay/README.md \
  ruche/README.md \
  release/adapter_manifest.csv \
  reports/adapter_release_audit.md \
  reports/public_release_audit.md \
  reports/release_candidate_verification.md \
  reports/public_release_commit_plan.md
git diff --cached --check
git diff --cached --name-status
git commit -m "docs(release): add curated COMPAYL configs audits and figures"
```

Explicitly excluded: `.gitignore`, all `release/huggingface/` contents, `figures/paper_figures/`, legacy/superseded figure copies, binary weights, and caches.

Expected status afterward: GitHub README/audits/canonical figures and the internal audit manifest are clean. Only `.gitignore`, the intended HF public tree, and intentionally untracked duplicate figures remain visible; ignored local weights remain unstaged.

## Commit 4

### `chore(release): add verified Hugging Face release metadata`

Why: this is the complete non-binary standalone model-repository draft: release-candidate model card, pending-license notice, citations, verified manifest/summary, public configs/results/figures, loader/verifier scripts, and per-adapter public metadata/checksums. `.gitignore` is included here because it enforces the boundary between GitHub metadata and binary weights.

Exact top-level and fixed files:

```text
.gitignore
release/huggingface/README.md
release/huggingface/LICENSE
release/huggingface/CITATION.cff
release/huggingface/adapter_manifest_verified.csv
release/huggingface/verification_summary.json
release/huggingface/adapters/README.md
release/huggingface/figures/architecture.png
release/huggingface/figures/qualitative.png
release/huggingface/results/klt_ablation_final_with_type_metrics.csv
release/huggingface/results/tissue_specific_compayl2026_final.csv
release/huggingface/scripts/export_adapter_safetensors.py
release/huggingface/scripts/verify_released_adapter.py
```

Also include:

- all 34 paths formed by prefixing the [curated config inventory](#curated-config-inventory) with `release/huggingface/configs/release/compayl2026/`;
- exactly `adapter_config.json` and `checksums.json` in each of these 12 directories:

```text
release/huggingface/adapters/klt/seed42/
release/huggingface/adapters/klt/seed43/
release/huggingface/adapters/tissue_specific/breast/seed42/
release/huggingface/adapters/tissue_specific/colon/seed42/
release/huggingface/adapters/tissue_specific/kidney/seed42/
release/huggingface/adapters/tissue_specific/kidney/seed43/
release/huggingface/adapters/tissue_specific/liver/seed42/
release/huggingface/adapters/tissue_specific/lung/seed42/
release/huggingface/adapters/tissue_specific/ovary/seed42/
release/huggingface/adapters/tissue_specific/pancreatic/seed43/
release/huggingface/adapters/tissue_specific/skin/seed42/
release/huggingface/adapters/tissue_specific/tonsil/seed43/
```

Exact staging commands:

```bash
git add -- \
  .gitignore \
  release/huggingface/README.md \
  release/huggingface/LICENSE \
  release/huggingface/CITATION.cff \
  release/huggingface/adapter_manifest_verified.csv \
  release/huggingface/verification_summary.json \
  release/huggingface/adapters/README.md \
  release/huggingface/figures/architecture.png \
  release/huggingface/figures/qualitative.png \
  release/huggingface/results/klt_ablation_final_with_type_metrics.csv \
  release/huggingface/results/tissue_specific_compayl2026_final.csv \
  release/huggingface/scripts/export_adapter_safetensors.py \
  release/huggingface/scripts/verify_released_adapter.py

git add -- release/huggingface/configs/release/compayl2026

git add -- \
  ':(glob)release/huggingface/adapters/**/adapter_config.json' \
  ':(glob)release/huggingface/adapters/**/checksums.json'

git diff --cached --check
git diff --cached --name-status
if git diff --cached --name-only | grep -Eq '\.(safetensors|pth|pt|ckpt)$'; then
  echo "ERROR: binary checkpoint staged" >&2
  exit 1
fi
git commit -m "chore(release): add verified Hugging Face release metadata"
```

Explicitly excluded from Commit 4:

```text
release/adapter_manifest.csv
release/huggingface/adapter_manifest.csv
release/huggingface/figures/sthelar-adapt.png
release/huggingface/figures/qualitative_grid_3x9_supplement.png
release/huggingface/adapters/**/*.safetensors
release/huggingface/adapters/**/*.pth
release/huggingface/.cache/
release/huggingface/cache/
models/pretrained/**
```

Expected status afterward: none of the four commit scopes remains modified/untracked. The only expected visible untracked paths are the two `figures/paper_figures/` copies and the two superseded HF figure copies. Safetensors packages, local `.pth` adapters, base checkpoints, and caches remain ignored and unstaged. Before any future push, run `git status --short --ignored` and inspect the four commits manually.

## Curated config inventory

The following 34 relative paths are the exact config set used twice in this plan: with prefix `configs/release/compayl2026/` in Commit 1, and with prefix `release/huggingface/configs/release/compayl2026/` in Commit 4.

```text
klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml
klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed43_KEEPALL_FISHER.yaml
klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42.yaml
klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
klt/training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN.yaml
preprocessing/preprocessing_sthelar40x_breast_5class_spatial_margin128_cap50000.yaml
preprocessing/preprocessing_sthelar40x_colon_5class_spatial_margin128_cap50000.yaml
preprocessing/preprocessing_sthelar40x_kidney_5class_spatial_margin128.yaml
preprocessing/preprocessing_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128.yaml
preprocessing/preprocessing_sthelar40x_liver_5class_spatial_margin128.yaml
preprocessing/preprocessing_sthelar40x_lung_5class_spatial_margin128_cap50000.yaml
preprocessing/preprocessing_sthelar40x_ovary_5class_spatial_margin128.yaml
preprocessing/preprocessing_sthelar40x_pancreatic_5class_spatial_margin128_cap50000.yaml
preprocessing/preprocessing_sthelar40x_skin_5class_spatial_margin128_cap50000.yaml
preprocessing/preprocessing_sthelar40x_tonsil_5class_spatial_margin128_cap50000.yaml
tissue_specific/training_sthelar40x_breast_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_colon_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_colon_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_kidney_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN.yaml
tissue_specific/training_sthelar40x_liver_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_lung_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_ovary_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_pancreatic_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_pancreatic_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN.yaml
tissue_specific/training_sthelar40x_skin_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_skin_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_tonsil_5class_spatial_margin128_cap50000_fullft_lr1e-5_e10_seed42_CLEAN.yaml
tissue_specific/training_sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN.yaml
```

## Plan validation

The four path sets were expanded without staging and checked directly against the worktree:

- Commit 1: 47 files;
- Commit 2: 3 files;
- Commit 3: 10 files;
- Commit 4: 71 files;
- total: 131 unique proposed files, with no overlap between commits;
- no proposed path ends in `.safetensors`, `.pth`, `.pt`, or `.ckpt`, names a base checkpoint, enters a cache directory, or includes `release/huggingface/adapter_manifest.csv`;
- the largest proposed file is `release/huggingface/figures/qualitative.png` at 7,338,332 bytes; no proposed file exceeds 10 MiB.

`git add --dry-run` was not used because this review workspace exposes `.git` read-only and Git cannot create `.git/index.lock`, even for a dry run. The direct path-set checks above did not modify the index. `git diff --cached --quiet` confirms that the index remains clean.

## Manual review gates before running this plan

1. Confirm that public GitHub history should contain the internal 51-checkpoint audit manifest; it contains relative local source paths and hashes but no weights.
2. Confirm whether per-adapter `checksums.json` files should be public. They are useful provenance and contain only filenames/hashes, but source `.pth` filenames remain visible.
3. Confirm that the copied inherited `LICENSE` is the correct notice to include while the model-card front matter remains `license: other` / pending rights-holder approval.
4. Confirm archival/removal of duplicate figure sources separately; this plan does not delete them.
5. Replace the placeholder Hugging Face repository ID only after the publication destination is approved.
