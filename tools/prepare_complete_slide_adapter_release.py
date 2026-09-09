#!/usr/bin/env python3
"""Create curated configs and static documentation for complete_slide_v1.

This script is deterministic and never edits historical configs. Fold configs are
byte-for-byte copies of the canonical sources; deployment configs are derived
from those sources with explicit all-slides metadata and no held-out test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/release/complete_slide_v1"
BUNDLE = ROOT / "release/huggingface/complete_slide_v1"
DATA_ROOT = "${DATA_ROOT}"
REVISION = "e32a8cdd50eff2d38e237f3729e9ac85bbb5203b"
BASE_SHA = "b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf"
TISSUES = (
    "breast", "colon", "kidney", "liver", "lung", "ovary",
    "pancreatic", "skin", "tonsil",
)
SLIDES = {
    "klt": ["kidney_s0", "kidney_s1", "liver_s0", "liver_s1", "tonsil_s0", "tonsil_s1"],
    "breast": ["breast_s0", "breast_s1"],
    "colon": ["colon_s1", "colon_s2"],
    "kidney": ["kidney_s0", "kidney_s1"],
    "liver": ["liver_s0", "liver_s1"],
    "lung": ["lung_s1", "lung_s3"],
    "ovary": ["ovary_s0", "ovary_s1"],
    "pancreatic": ["pancreatic_s0", "pancreatic_s1"],
    "skin": ["skin_s1", "skin_s2"],
    "tonsil": ["tonsil_s0", "tonsil_s1"],
}
SUFFIX = {
    "klt": "",
    "breast": "_cap50000", "colon": "_cap50000", "kidney": "",
    "liver": "", "lung": "_cap50000", "ovary": "",
    "pancreatic": "_cap50000", "skin": "_cap50000", "tonsil": "_cap50000",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def canonical_source(domain: str, fold: str) -> Path:
    letter = fold[-1].upper()
    if domain == "klt":
        name = f"training_sthelar40x_klt_5class_slideind_fold{letter}_lora_adaptformer_heads_seed42.yaml"
    else:
        name = f"training_sthelar40x_{domain}_5class_slideind_fold{letter}_lora_adaptformer_heads_seed42.yaml"
        if domain in {"kidney", "liver"} and letter == "B":
            retry = ROOT / "configs/slide_exp/training" / name.replace(".yaml", "_retry_adapterexport.yaml")
            if retry.is_file():
                return retry
    return ROOT / "configs/slide_exp/training" / name


def make_configs() -> None:
    training_root = CONFIG_ROOT / "training"
    prep_root = CONFIG_ROOT / "preprocessing"
    for domain in ("klt",) + TISSUES:
        family = "klt" if domain == "klt" else f"tissue_specific/{domain}"
        for role in ("fold_a", "fold_b"):
            source = canonical_source(domain, role)
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = training_root / family / f"{role}_seed42.yaml"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.read_bytes() != source.read_bytes():
                shutil.copyfile(source, destination)

        suffix = SUFFIX[domain]
        source_name = (
            "sthelar40x_kidney_liver_tonsil_5class_spatial_margin128"
            if domain == "klt"
            else f"sthelar40x_{domain}_5class_spatial_margin128{suffix}"
        )
        output_name = f"sthelar40x_{domain}_5class_all_slides_margin128{suffix}"
        prep = {
            "fold": "all_slides",
            "strategy": "slide",
            "reuse_packed_data": True,
            "source_dataset_root": f"{DATA_ROOT}/{source_name}",
            "output_root": f"{DATA_ROOT}/{output_name}",
            "slide_ids": SLIDES[domain],
            "train_slides": SLIDES[domain],
            "valid_slides": [],
            "test_slides": [],
            "split_axis": "x",
            "boundary_margin": 128,
            "train_frac": 0.70,
            "valid_frac": 0.15,
            "test_frac": 0.15,
            "train_frac_inside_train_slide": 0.85,
            "valid_frac_inside_train_slide": 0.15,
            "random_seed": 42,
            "release_metadata": {
                "evaluation_role": "deployment",
                "held_out_test": None,
                "paper_metric": None,
                "all_slides_definition": "All designated slides are eligible; a nominal 85%/15% spatial-axis train partition with a 128-unit boundary margin is applied within every slide. Patch-count fractions vary with spatial density.",
            },
        }
        prep_path = prep_root / family / "all_slides_seed42.yaml"
        write_text(prep_path, yaml.safe_dump(prep, sort_keys=False))

        base = yaml.safe_load(canonical_source(domain, "fold_a").read_text())
        run_name = f"sthelar40x_{domain}_5class_all_slides_lora_adaptformer_r8_a8_red16_heads_e10_seed42"
        base["logging"].update({
            "project": "sthelar40x_complete_slide_v1_deployment",
            "notes": "Deployment-only all-slides adapter. No held-out test claim; 15% spatial validation is retained within every designated slide.",
            "log_comment": run_name,
            "tags": ["sthelar", "40x", domain, "all_slides", "deployment", "selected_peft", "seed42"],
            "wandb_dir": f"run/{run_name}/wandb",
            "log_dir": f"run/{run_name}/log",
        })
        base["data"].update({
            "dataset_path": f"{DATA_ROOT}/{output_name}",
            "split": "all_slides_deployment",
            "train_folds": ["train"],
            "val_folds": ["valid"],
            "test_folds": None,
        })
        base["adapters"] = {
            "adapter_type": "lora_adaptformer",
            "decoder_train_scope": "heads_only",
            "lora": {"rank": 8, "alpha": 8, "targets": ["q", "v"], "dropout": 0.0},
            "adaptformer": {"activation": "GELU", "reduction": 16},
        }
        base["training"].update({"epochs": 10, "optimizer": "AdamW", "mixed_precision": True})
        base["checkpointing"] = {
            "save_best": True, "save_last": True, "keep_last_n": 1,
            "save_every": 1, "delete_intermediate_checkpoints": True,
        }
        base["adapter_export"] = {
            "enabled": True, "format": "pth",
            "output_dir": "adapters/complete_slide_v1_staging",
            "export_best": False, "export_last": True, "verify_load": True,
        }
        base["eval_checkpoint"] = "checkpoint_10.pth"
        base["release_metadata"] = {
            "release": "complete_slide_v1",
            "semantic_role": "all_slides",
            "evaluation_role": "deployment",
            "held_out_test": None,
            "paper_metric": None,
            "designated_slides": SLIDES[domain],
            "nominal_spatial_train_fraction": 0.85,
            "nominal_spatial_validation_fraction": 0.15,
            "validation_strategy": "deterministic spatial x-axis split within every designated slide with a 128-coordinate-unit boundary margin",
            "dataset_revision": REVISION,
            "base_sha256": BASE_SHA,
        }
        out = training_root / family / "all_slides_seed42.yaml"
        write_text(out, yaml.safe_dump(base, sort_keys=False))


def copy_public_configs() -> None:
    destination = BUNDLE / "configs/complete_slide_v1"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(CONFIG_ROOT, destination)


def make_results() -> None:
    source = ROOT / "reports/workshop_master_results.csv"
    rows = []
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["protocol"] in {"held_out_slide_KLT", "tissue_specific_held_out_slide"}
                and row["backbone"] == "CellViT-SAM-H"
                and row["method"] == "Selected PEFT"
                and row["seed"] == "42"
                and row["fold"] in {"A", "B"}
            ):
                rows.append({key: row[key] for key in (
                    "protocol", "domain", "tissue", "seed", "fold", "checkpoint_policy",
                    "bPQ", "mPQ", "F1det", "F1type", "test_patches",
                    "train_slide", "test_slide", "test_patch_id_sha256",
                )})
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 audited seed-42 fold rows, found {len(rows)}")
    out = BUNDLE / "results/complete_slide_v1/fold_results_seed42.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_text(BUNDLE / "results/complete_slide_v1/README.md", """# Complete-slide results\n\n`fold_results_seed42.csv` contains only audited fixed epoch-10 held-out-slide results copied from `reports/workshop_master_results.csv`. The all-slides deployment adapters have no held-out test metrics and do not appear in this table.\n""")


def copy_split_manifests() -> None:
    data_root = Path(os.environ.get("DATA_ROOT", "data/cellvit_ready")).expanduser()
    destination_root = BUNDLE / "results/complete_slide_v1/split_manifests"
    for domain in ("klt",) + TISSUES:
        family = "klt" if domain == "klt" else f"tissue_specific/{domain}"
        suffix = SUFFIX[domain]
        for role in ("fold_a", "fold_b", "all_slides"):
            if role == "all_slides":
                dataset = f"sthelar40x_{domain}_5class_all_slides_margin128{suffix}"
            else:
                letter = role[-1].upper()
                dataset = f"sthelar40x_{domain}_5class_slideind_fold{letter}_margin128{suffix}"
            source = data_root / dataset
            out = destination_root / family / role
            out.mkdir(parents=True, exist_ok=True)
            manifest = yaml.safe_load((source / "split_manifest.yaml").read_text())
            for key in ("source_dataset_root", "output_root"):
                if key in manifest:
                    manifest[key] = f"${{DATA_ROOT}}/{Path(manifest[key]).name}"
            write_text(out / "split_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
            validation = source / "split_validation.json"
            if validation.is_file():
                shutil.copyfile(validation, out / "split_validation.json")


def make_static_docs() -> None:
    write_text(CONFIG_ROOT / "README.md", """# complete_slide_v1 configs\n\nFold A/B files are byte-for-byte curated copies of the canonical complete-slide seed-42 experiment configs. `all_slides` uses all designated slides, with a nominal deterministic spatial-axis 85% training / 15% validation partition and a 128-unit boundary margin inside every slide. Actual patch fractions vary with spatial patch density and are recorded in each materialized split manifest. Therefore “all slides” does not mean every eligible patch receives gradient updates. All-slides configs set `test_folds: null`; framework post-training inference falls back to validation and is not a held-out evaluation.\n""")
    copy_public_configs()
    make_results()
    copy_split_manifests()
    scripts = BUNDLE / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for source in (
        ROOT / "utils/cellvit_adapter_hub.py",
        ROOT / "utils/adapter_checkpoint.py",
        ROOT / "tools/verify_released_adapter.py",
    ):
        shutil.copyfile(source, scripts / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Regenerate then print config hashes")
    args = parser.parse_args()
    make_configs()
    make_static_docs()
    paths = sorted(CONFIG_ROOT.rglob("*.yaml"))
    print(json.dumps({"configs": len(paths), "hashes": {str(p.relative_to(ROOT)): sha256(p) for p in paths}}, indent=2))


if __name__ == "__main__":
    main()
