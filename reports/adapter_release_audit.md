# Adapter release audit

> Follow-up (2026-07-20): all 12 release-marked adapters were subsequently converted and passed exact round-trip, metadata, base reconstruction, strict adapter-key loading, and 256×256 CPU forward checks. See `reports/release_candidate_verification.md` and `release/huggingface/adapter_manifest_verified.csv`. The pending-language below records the state of the original structural audit.

Audit source: local ignored `adapters/` directory (used as the intended directory because the request contained the unresolved placeholder `<REPLACE_WITH_LOCAL_ADAPTER_DIRECTORY>`). Files were read only and not modified. Inventory: 51 `.pth` files, 0 structural parse failures, 12 proposed release checkpoints, 38 local archives, and 1 explicit investigation (`V100_RETEST`). No byte-for-byte or normalized tensor-state exact duplicates were found.

## Method

`tools/audit_adapter_checkpoints.py` computes file SHA256 and reads PyTorch ZIP `data.pkl` through a restricted unpickler that accepts only tensor rebuild/storage descriptors, ordered dictionaries, and the small NumPy scalar encoding used by these files. It does not import PyTorch, materialize values, or execute arbitrary checkpoint classes. Raw tensor-storage bytes are hashed with state keys/shapes to detect content duplicates independent of ZIP prefixes. This is a structural audit, not a substitute for `torch.load(weights_only=True)`, safetensors round-trip equality, actual model loading, or inference.

Every file has the expected top-level adapter-only schema:

`format; format_version; metadata; adapter_state_dict; mutable_buffer_state_dict`

The selected heads-only LoRA+AdaptFormer files each contain 401 tensors total (296 state tensors and 105 mutable buffers), 7,927,950 tensor elements including buffers, 7,908,779 trainable parameters in metadata, float32 parameters plus int64 BatchNorm counters, LoRA Q/V keys, AdaptFormer keys, and NP/HV/NT final-head keys. They report `exact_state_match` at their original export. The embedded paths are private/cluster-specific and must be sanitized.

## Size/component interpretation

- Approximately 2.3 MiB: decoder-conv-adapter-only controls. Small size is explained by scope, not proof of validity.
- Approximately 7.4 MiB: LoRA plus decoder-conv adapters without AdaptFormer.
- Approximately 25.4–28.0 MiB: AdaptFormer and/or VeRA families, with scope-dependent heads/adapters.
- Approximately 30.4–30.6 MiB: selected LoRA+AdaptFormer heads-only files.
- Approximately 31.8–32.7 MiB: LoRA+AdaptFormer last-stage/conv-adapter variants.
- Approximately 88.0 MiB: `decoder_np_hv_heads_nt_all`, much larger because its decoder scope is not the selected final-head-only method; archive as an ablation.
- Approximately 0.5 MiB and 5.2 MiB: compact VeRA-only and LoRA-only controls. They are structurally coherent but not selected releases.

File size alone was not used to approve any checkpoint.

## Canonical release set

Exact paths and SHA256 values are in `release/adapter_manifest.csv`; only rows whose `recommended_action` is `release` are canonical.

1. `adapters/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
2. `adapters/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN_adapter.pth`
3. `adapters/sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
4. `adapters/sthelar40x_colon_5class_spatial_margin128_lora_adaptformer_heads_only_seed42/sthelar40x_colon_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
5. `adapters/sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
6. `adapters/sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_heads_only_seed43/sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN_adapter.pth`
7. `adapters/sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
8. `adapters/sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
9. `adapters/sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
10. `adapters/sthelar40x_pancreatic_5class_spatial_margin128_cap50000_lora_adaptformer_heads_only_seed43/sthelar40x_pancreatic_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN_adapter.pth`
11. `adapters/sthelar40x_skin_5class_spatial_margin128_lora_adaptformer_heads_only_seed42/sthelar40x_skin_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth`
12. `adapters/sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_heads_only_seed43/sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN_adapter.pth`

This matches the reported policy: KLT seeds 42/43; tissue seed42 except Pancreatic/Tonsil seed43; Kidney PEFT averaged over seeds 42/43.

## Ambiguous reruns and duplicates

There are no exact file or normalized tensor-state duplicates. Several equal-sized files are different trained states.

- Colon seed42: the nested file above traces to timestamp `2026-06-30T092331`, test mPQ `0.2047473297`, matching the paper. The same-named root file traces to `2026-06-30T203052`, test mPQ `0.2018031368`; archive locally.
- Lung seed42: the root file above traces to timestamp `2026-07-01T025800`, test mPQ `0.3015391203`, matching the paper. The same-named nested file traces to `2026-06-30T092331`, test mPQ `0.2963457958`; archive locally.
- Liver `V100_RETEST` is a separately named/tensor-different run. Its `RETEST` status and absent matching release config require investigation; do not release it.

All ablation, 9-class BPS, VeRA, conv-adapter, last-stage, broad-decoder, and compact-control checkpoints should remain local or be published later only as explicitly labelled supplementary artifacts. “Archive locally” does not mean invalid; it means outside the canonical paper release.

## Conversion and verification

`tools/export_adapter_safetensors.py`:

- CPU-loads a strict known `.pth` schema and prefers `weights_only=True` when supported;
- rejects missing/unexpected top-level keys and non-tensors in state sections;
- exports metadata separately to `adapter_config.json`, replaces absolute paths with basenames, and embeds the JSON in the safetensors header for loader compatibility;
- refuses to overwrite output files;
- checks all required LoRA/AdaptFormer/NP/HV/NT components by default;
- reloads safetensors and checks every tensor with `torch.equal`;
- writes SHA256 values for source, weights, and config.

`tools/verify_released_adapter.py` checks exact keys/shapes against export metadata, component presence, the five-foreground-class-plus-background mapping, declared trainable parameter count, declared `CellViT-SAM-H-x40.pth` compatibility, and exported SHA256. With `--base-checkpoint`, it reconstructs the model, loads every adapter/buffer tensor, and performs a 256×256 CPU forward smoke test unless `--skip-forward` is passed.

Neither utility was executed end-to-end here because PyTorch and safetensors are absent from the available environment. Before release, run both on every canonical file in a trusted Python environment, record exported hashes in a frozen manifest, and rescan JSON/header metadata for local paths/secrets. The full base checkpoint must not be included.

## Remaining decisions

- Obtain and record the authoritative base checkpoint SHA256. Current adapter metadata declares only the name/path.
- Decide stable public adapter IDs and whether KLT/tissue weights live in one repository or separate subdirectories.
- Confirm that mutable BatchNorm buffers are intended/required; the original exporter includes them for exact reconstruction.
- Confirm weight licensing with CellViT/SAM/STHELAR rights holders before upload.
- Do not mark any file `discard` without manual approval; the manifest uses `archive locally` or `investigate` for all non-release files.
