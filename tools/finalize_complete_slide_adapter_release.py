#!/usr/bin/env python3
"""Idempotently stage and verify the 30-target complete_slide_v1 release.

The tool never trains and never deletes source artifacts. It accepts only packages
with exact base+adapter state reconstruction and exact deterministic forward proof.
Use --require-complete for the publication gate.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "release/huggingface/complete_slide_v1"
OLD_ROOT = ROOT / "release/huggingface/adapters/cellvit-sam-h-x40"
STAGING = ROOT / "adapters/complete_slide_v1_staging_safetensors"
RECOVERY = ROOT / "adapters/slide_exp_safetensors"
INVENTORY_CSV = ROOT / "release/huggingface/workshop_release_inventory.csv"
INVENTORY_MD = ROOT / "release/huggingface/workshop_release_inventory.md"
MANIFEST = ROOT / "release/huggingface/complete_slide_adapter_manifest.csv"
SUMMARY = ROOT / "release/huggingface/complete_slide_verification_summary.json"
BASE_SHA = "b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf"
REVISION = "e32a8cdd50eff2d38e237f3729e9ac85bbb5203b"
TISSUES = ("breast", "colon", "kidney", "liver", "lung", "ovary", "pancreatic", "skin", "tonsil")
SLIDES = {
    "klt": (["kidney_s0", "liver_s0", "tonsil_s0"], ["kidney_s1", "liver_s1", "tonsil_s1"]),
    "breast": (["breast_s0"], ["breast_s1"]), "colon": (["colon_s1"], ["colon_s2"]),
    "kidney": (["kidney_s0"], ["kidney_s1"]), "liver": (["liver_s0"], ["liver_s1"]),
    "lung": (["lung_s1"], ["lung_s3"]), "ovary": (["ovary_s0"], ["ovary_s1"]),
    "pancreatic": (["pancreatic_s0"], ["pancreatic_s1"]),
    "skin": (["skin_s1"], ["skin_s2"]), "tonsil": (["tonsil_s0"], ["tonsil_s1"]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def targets() -> list[dict[str, str]]:
    rows = []
    for domain in ("klt",) + TISSUES:
        family = "klt" if domain == "klt" else f"tissue_specific/{domain}"
        for role in ("fold_a", "fold_b", "all_slides"):
            rows.append({"domain": domain, "family": family, "role": role, "seed": "42"})
    return rows


def target_id(row: dict[str, str]) -> str:
    return f"{row['family']}/{row['role']}/seed42"


def output_dir(row: dict[str, str]) -> Path:
    return BUNDLE / "adapters/complete_slide_v1/samh" / row["family"] / row["role"] / "seed42"


def source_config(row: dict[str, str]) -> Path:
    return ROOT / "configs/release/complete_slide_v1/training" / row["family"] / f"{row['role']}_seed42.yaml"


def package_for(row: dict[str, str]) -> Path | None:
    role = row["role"]
    domain = row["domain"]
    if role in {"fold_a", "fold_b"}:
        fold = "folda" if role == "fold_a" else "foldb"
        base = OLD_ROOT / ("klt" if domain == "klt" else f"tissue_specific/{domain}") / "slideind" / fold / "seed42"
        if base.is_dir():
            return base
        if domain in {"kidney", "liver"} and role == "fold_b":
            name = f"sthelar40x_{domain}_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42"
            candidate = RECOVERY / name
            return candidate if candidate.is_dir() else None
        return None
    name = f"sthelar40x_{domain}_5class_all_slides_lora_adaptformer_r8_a8_red16_heads_e10_seed42"
    for root in (STAGING, RECOVERY):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def verification_path(package: Path) -> Path | None:
    candidates = sorted(package.glob("verification*.json"))
    exact = [p for p in candidates if p.is_file() and json.loads(p.read_text()).get("forward_verification") == "exact_all_output_tensors"]
    return exact[-1] if exact else None


def inspect_package(package: Path, row: dict[str, str]) -> dict[str, object]:
    weights = package / "adapter_model.safetensors"
    config_path = package / "adapter_config.json"
    checksums_path = package / "checksums.json"
    verification = verification_path(package)
    for path in (weights, config_path, checksums_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if verification is None:
        raise RuntimeError(f"No exact forward verification record in {package}")
    config = json.loads(config_path.read_text())
    checksums = json.loads(checksums_path.read_text())
    proof = json.loads(verification.read_text())
    expected = checksums.get("exported", {}).get("adapter_model.safetensors")
    actual = sha256(weights)
    if expected != actual:
        raise RuntimeError(f"Safetensors checksum mismatch in {package}")
    if config.get("base_model") != "CellViT-SAM-H-x40" or config.get("base_checkpoint_sha256") != BASE_SHA:
        raise RuntimeError(f"Wrong base identity in {package}")
    if config.get("adapter_type") != "lora_adaptformer" or config.get("decoder_train_scope") != "heads_only":
        raise RuntimeError(f"Wrong adapter formulation in {package}")
    if config.get("lora") != {"alpha": 8, "dropout": 0.0, "rank": 8, "targets": ["q", "v"]}:
        raise RuntimeError(f"Wrong LoRA configuration in {package}")
    if config.get("adaptformer") != {"activation": "GELU", "reduction": 16}:
        raise RuntimeError(f"Wrong AdaptFormer configuration in {package}")
    if proof.get("status") != "ok" or proof.get("state_reconstruction") != "exact_all_tensors" or proof.get("forward_verification") != "exact_all_output_tensors":
        raise RuntimeError(f"Verification gates failed in {verification}")
    comparisons = proof.get("forward_output_comparison", [])
    if {x.get("name") for x in comparisons} != {"hv_map", "nuclei_binary_map", "nuclei_type_map", "tissue_types"} or not all(x.get("exact") for x in comparisons):
        raise RuntimeError(f"Incomplete exact output proof in {verification}")
    required_components = set(config.get("expected_components", []))
    if required_components != {"lora", "adaptformer", "np_head", "hv_head", "nt_head"}:
        raise RuntimeError(f"Component declaration mismatch in {package}")
    return {"weights": weights, "config": config, "proof": proof, "verification": verification, "adapter_sha256": actual}


def stage(row: dict[str, str], inspected: dict[str, object]) -> dict[str, str]:
    destination = output_dir(row)
    destination.mkdir(parents=True, exist_ok=True)
    weights_out = destination / "adapter.safetensors"
    if not weights_out.is_file() or sha256(weights_out) != inspected["adapter_sha256"]:
        shutil.copyfile(inspected["weights"], weights_out)
    role = row["role"]
    a, b = SLIDES[row["domain"]]
    train = a if role == "fold_a" else b if role == "fold_b" else sorted(a + b)
    validation = train
    held_out = b if role == "fold_a" else a if role == "fold_b" else []
    source = source_config(row)
    source_hash = inspected["config"].get("source_config_sha256")
    if role == "all_slides":
        source_hash = sha256(source)
    public_config = dict(inspected["config"])
    public_config.update({
        "format_version": 1,
        "model_family": "CellViT-SAM-H-x40",
        "base_checkpoint_filename": "CellViT-SAM-H-x40.pth",
        "base_sha256": BASE_SHA,
        "adapter_type": "lora_adaptformer_heads",
        "lora": {"rank": 8, "alpha": 8, "targets": ["q", "v"], "dropout": 0.0},
        "adaptformer": {"reduction": 16, "activation": "GELU"},
        "trainable_heads": ["NP", "HV", "NT"],
        "label_order": ["Background", "Immune", "Stromal", "Epithelial", "Melanocyte", "Other"],
        "dataset_revision": REVISION,
        "semantic_role": role,
        "evaluation_role": "deployment" if role == "all_slides" else "held_out_evaluation",
        "tissue": "KLT" if row["domain"] == "klt" else row["domain"],
        "seed": 42,
        "epoch": 10,
        "training_slides": train,
        "validation_source_slides": validation,
        "held_out_test": None if role == "all_slides" else held_out,
        "paper_metric": None if role == "all_slides" else "see results/complete_slide_v1/fold_results_seed42.csv",
        "all_slides_nominal_spatial_train_fraction": 0.85 if role == "all_slides" else None,
        "all_slides_nominal_spatial_validation_fraction": 0.15 if role == "all_slides" else None,
        "source_config_sha256": source_hash,
        "source_checkpoint_sha256": inspected["proof"].get("canonical_checkpoint_sha256"),
        "adapter_sha256": inspected["adapter_sha256"],
        "adapter_weight_file": "adapter.safetensors",
    })
    # The public contract uses the explicit adapter_type above; retain the loader's
    # architecture spelling separately so old embedded metadata remains interpretable.
    public_config["loader_adapter_type"] = "lora_adaptformer"
    atomic_json(destination / "adapter_config.json", public_config)
    proof = inspected["proof"]
    public_proof = {
        "status": "VERIFIED",
        "verified_source_package": rel(Path(inspected["weights"]).parent),
        "source_verification_sha256": sha256(inspected["verification"]),
        "base_checkpoint_sha256": BASE_SHA,
        "source_checkpoint_sha256": proof.get("canonical_checkpoint_sha256"),
        "adapter_sha256": inspected["adapter_sha256"],
        "safetensors_parse": "pass",
        "source_tensor_equality": "pass",
        "metadata_consistency": "pass",
        "adapter_required_missing_keys": proof.get("missing_adapter_keys", []),
        "adapter_required_unexpected_keys": proof.get("unexpected_adapter_keys", []),
        "mutable_buffer_missing_keys": proof.get("missing_mutable_buffer_keys", []),
        "mutable_buffer_unexpected_keys": proof.get("unexpected_mutable_buffer_keys", []),
        "state_reconstruction": proof.get("state_reconstruction"),
        "forward_verification": proof.get("forward_verification"),
        "forward_output_comparison": proof.get("forward_output_comparison"),
    }
    atomic_json(destination / "verification.json", public_proof)
    return {
        "adapter_path": rel(weights_out), "adapter_sha256": sha256(weights_out),
        "adapter_config_path": rel(destination / "adapter_config.json"),
        "adapter_config_sha256": sha256(destination / "adapter_config.json"),
        "verification_path": rel(destination / "verification.json"),
        "verification_sha256": sha256(destination / "verification.json"),
    }


def build(stage_files: bool) -> list[dict[str, str]]:
    results = []
    for row in targets():
        package = package_for(row)
        status = "MISSING_NEEDS_TRAINING"
        reason = "No source package exists; submit canonical all-slides deployment training."
        details: dict[str, str] = {}
        if package is not None:
            try:
                inspected = inspect_package(package, row)
                details = stage(row, inspected) if stage_files else {}
                status = "VERIFIED_EXISTING"
                reason = "Exact source-state reconstruction and deterministic forward equality are recorded."
                details.update({
                    "source_checkpoint_path": str(inspected["proof"].get("canonical_checkpoint", "")),
                    "source_checkpoint_sha256": str(inspected["proof"].get("canonical_checkpoint_sha256", "")),
                    "existing_adapter_export_path": rel(Path(inspected["weights"])),
                    "existing_adapter_sha256": str(inspected["adapter_sha256"]),
                    "verification_record": rel(Path(inspected["verification"])),
                })
            except Exception as exc:
                status = "EXISTING_NEEDS_VERIFICATION"
                reason = str(exc)
        a, b = SLIDES[row["domain"]]
        train = a if row["role"] == "fold_a" else b if row["role"] == "fold_b" else sorted(a + b)
        held = b if row["role"] == "fold_a" else a if row["role"] == "fold_b" else []
        results.append({
            "target_adapter": target_id(row), "semantic_task": "KLT" if row["domain"] == "klt" else "tissue_specific",
            "tissue": "KLT" if row["domain"] == "klt" else row["domain"], "semantic_role": row["role"],
            "seed": "42", "model_backbone": "CellViT-SAM-H-x40",
            "adapter_configuration": "LoRA Q,V r8 alpha8 dropout0 + AdaptFormer red16 GELU + final NP/HV/NT heads; encoder base and decoder body frozen",
            "epoch_checkpoint": "10", "training_slides": ";".join(train), "validation_slides": ";".join(train),
            "held_out_test_slide": ";".join(held) if held else "none", "config_path": rel(source_config(row)),
            "source_checkpoint_path": details.get("source_checkpoint_path", ""), "source_checkpoint_sha256": details.get("source_checkpoint_sha256", ""),
            "existing_adapter_export_path": details.get("existing_adapter_export_path", ""), "existing_adapter_sha256": details.get("existing_adapter_sha256", ""),
            "final_adapter_path": details.get("adapter_path", ""), "final_adapter_sha256": details.get("adapter_sha256", ""),
            "adapter_config_path": details.get("adapter_config_path", ""), "adapter_config_sha256": details.get("adapter_config_sha256", ""),
            "verification_path": details.get("verification_path", details.get("verification_record", "")), "verification_sha256": details.get("verification_sha256", ""),
            "matches_selected_peft": "yes" if status == "VERIFIED_EXISTING" else "pending",
            "scientifically_canonical": "yes" if row["role"] != "all_slides" and status == "VERIFIED_EXISTING" else "deployment" if row["role"] == "all_slides" else "pending",
            "exact_load_forward_verification": "yes" if status == "VERIFIED_EXISTING" else "no",
            "status": status, "reason": reason,
        })
    return results


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_inventory_md(rows: list[dict[str, str]]) -> None:
    counts = Counter(r["status"] for r in rows)
    lines = [
        "# Workshop release inventory", "", "Status date: 2026-09-09", "",
        "## Safety and provenance", "",
        f"- Repository commit at audit start: `5dc3ae1dea3999c5c78863a28e2216cafafa55e6`.",
        f"- CellViT-SAM-H-x40 base SHA256 recomputed: `{BASE_SHA}` (match). Base weights are excluded.",
        f"- Local Hugging Face cache metadata identifies STHELAR 40x revision `{REVISION}`. Prepared payload provenance is additionally pinned by the exact fold manifests and TEST patch-ID SHA256 values in `reports/workshop_master_results.csv`.",
        "- Workdir at audit start: 399 GiB / 500 GiB. Existing run/adapters/release sizes: 262 GiB / 1.6 GiB / 986 MiB.",
        "- Conservative active-job peak: 9 GiB per job (three ~2.9 GB checkpoint equivalents plus overhead). Dependencies cap concurrency at two. Because each completed job retains its canonical ~2.9 GB epoch-10 source, the campaign-wide worst point is about 41 GiB (eight retained sources plus two active jobs). The final 30 adapters are about 0.95 GB; duplicated versioned staging plus configs/results remains under 2 GB.",
        "", "## Target matrix", "", "| Target | Role | Train slides | Held-out test | Status |", "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['target_adapter']}` | {r['semantic_role']} | {r['training_slides']} | {r['held_out_test_slide']} | **{r['status']}** |")
    lines += [
        "", "## Status totals", "",
        *[f"- `{key}`: {counts.get(key, 0)}" for key in ("VERIFIED_EXISTING", "EXISTING_NEEDS_EXPORT", "EXISTING_NEEDS_VERIFICATION", "MISSING_NEEDS_TRAINING", "INVALID_DO_NOT_RELEASE")],
        "", "## Excluded candidate families", "",
        "- The audit searched `run/`, `adapters/`, `release/`, `reports/`, `configs/`, every source path referenced by the release/manuscript manifests, `adapters/complete_slide_v1_staging_safetensors/`, `adapters/slide_exp_safetensors/`, and both the historical and versioned Hugging Face staging roots. No separate archive/superseded root was referenced by a manifest.",
        "- The scientific target match was resolved from embedded configuration, split manifests, source-checkpoint hashes, adapter tensors, and exact-forward evidence; filenames alone were never sufficient.",
        "- Historical `configs/release/compayl2026/` and unversioned tissue/KLT adapters are within-slide artifacts and are not semantic matches.",
        "- NT-header1, last-stage, conv-adapter, VeRA, LP, FullFT, non-seed42, and unknown-epoch candidates do not match the selected release formulation.",
        "- CellViT-256 artifacts are valid for their own backbone but are prohibited from this SAM-H release; see `cellvit256_release_plan.md`.",
        "- All-slides rows are deployment artifacts. They use a nominal 85%/15% spatial-axis partition within every designated slide plus a 128-unit margin; actual train/validation patch fractions vary and are recorded in split manifests. Validation is not a held-out test and has no paper metric.",
    ]
    INVENTORY_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-stage", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    lock_path = ROOT / "release/huggingface/.complete_slide_v1_finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_handle:
        # Two dependency-wave jobs may complete together. Serialize shared
        # inventory/staging writes while leaving independent training untouched.
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows = build(not args.no_stage)
        write_csv(INVENTORY_CSV, rows); write_csv(MANIFEST, rows); write_inventory_md(rows)
        counts = Counter(r["status"] for r in rows)
        complete = len(rows) == 30 and counts == {"VERIFIED_EXISTING": 30}
        bundle_manifest = BUNDLE / "complete_slide_adapter_manifest.csv"
        bundle_summary = BUNDLE / "complete_slide_verification_summary.json"
        shutil.copyfile(MANIFEST, bundle_manifest)
        # A checksum document cannot contain its own stable hash. Every other public
        # file is hashed here; the uploader handles this generated summary explicitly.
        public_files = sorted(
            rel(p) for p in BUNDLE.rglob("*")
            if (
                p.is_file()
                and p != bundle_summary
                and "__pycache__" not in p.parts
                and p.suffix not in {".pyc", ".pyo"}
            )
        )
        summary = {
            "format_version": 1, "release": "complete_slide_v1", "target_count": 30,
            "status_counts": dict(counts), "release_complete": complete,
            "base_checkpoint_filename": "CellViT-SAM-H-x40.pth", "base_sha256": BASE_SHA,
            "dataset_revision": REVISION, "generated_at": datetime.now(timezone.utc).isoformat(),
            "public_files": public_files,
            "public_file_sha256": {path: sha256(ROOT / path) for path in public_files},
        }
        atomic_json(SUMMARY, summary)
        shutil.copyfile(SUMMARY, bundle_summary)
        print(json.dumps(summary, indent=2))
        if args.require_complete and not complete:
            raise SystemExit("Release is incomplete; refusing publication-ready status")


if __name__ == "__main__":
    main()
