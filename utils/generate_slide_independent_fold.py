#!/usr/bin/env python3
"""Materialize slide-independent split metadata while reusing packed KLT data."""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from preprocessing.sthelar.convert_hf_to_cellvit import assign_slide_split


REUSED_FILES = ("images.zip", "labels.zip", "types.csv", "dataset_config.yaml")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_counts(source: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "valid", "test"):
        path = source / "cell_count_{}.csv".format(split)
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    counts = pd.concat(frames, ignore_index=True)
    if counts["Image"].duplicated().any():
        duplicates = counts.loc[counts["Image"].duplicated(), "Image"].tolist()
        raise ValueError("Duplicate source cell-count identifiers: {}".format(duplicates[:5]))
    return counts.set_index("Image", drop=False)


def _split_sets(frame: pd.DataFrame) -> Dict[str, set]:
    return {
        split: set(frame.loc[frame["split"] == split, "packed_file_name"].astype(str))
        for split in ("train", "valid", "test")
    }


def validate_dataset(output: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "patch_info_with_split.csv",
        "cell_count_train.csv",
        "cell_count_valid.csv",
        "cell_count_test.csv",
        "split_manifest.yaml",
        "split_validation.json",
    ]
    for name in required:
        if not (output / name).is_file():
            raise FileNotFoundError(output / name)
    for name in REUSED_FILES:
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_target = (Path(config["source_dataset_root"]).resolve() / name).resolve()
        if path.resolve() != expected_target:
            raise AssertionError(
                "Reused payload points to the wrong source: {} -> {}".format(
                    path, path.resolve()
                )
            )

    manifest = yaml.safe_load((output / "split_manifest.yaml").read_text())
    expected_manifest = {
        "fold": str(config["fold"]),
        "split_strategy": "slide",
        "spatial_margin": int(config.get("boundary_margin", 128)),
        "train_slide_ids": sorted(str(item) for item in config["train_slides"]),
        "validation_source_slide_ids": sorted(
            str(item) for item in config["train_slides"]
        ),
        "test_slide_ids": sorted(str(item) for item in config["test_slides"]),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise AssertionError(
                "Existing manifest mismatch for {}: expected={} observed={}".format(
                    key, expected, manifest.get(key)
                )
            )

    frame = pd.read_csv(output / "patch_info_with_split.csv")
    sets = _split_sets(frame)
    overlaps = {
        "train_valid": sorted(sets["train"] & sets["valid"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "valid_test": sorted(sets["valid"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError("Patch identifier leakage: {}".format(overlaps))
    if frame["packed_file_name"].duplicated().any():
        raise AssertionError("A packed patch identifier occurs more than once")

    train_slide_ids = set(config["train_slides"])
    test_slide_ids = set(config["test_slides"])
    train_actual = set(frame.loc[frame["split"] == "train", "slide_id"].astype(str))
    valid_actual = set(frame.loc[frame["split"] == "valid", "slide_id"].astype(str))
    test_actual = set(frame.loc[frame["split"] == "test", "slide_id"].astype(str))
    checks = {
        "fold": str(config["fold"]),
        "train_slide_ids": sorted(train_actual),
        "validation_source_slide_ids": sorted(valid_actual),
        "test_slide_ids": sorted(test_actual),
        "train_test_disjoint": not bool(train_actual & test_actual),
        "validation_test_disjoint": not bool(valid_actual & test_actual),
        "patch_identifiers_unique_across_splits": not any(overlaps.values()),
        "train_slides_exact": train_actual == train_slide_ids,
        "validation_sources_exact": valid_actual == train_slide_ids,
        "test_slides_exact": test_actual == test_slide_ids,
        "all_checks_passed": False,
    }
    checks["all_checks_passed"] = all(
        checks[key]
        for key in (
            "train_test_disjoint",
            "validation_test_disjoint",
            "patch_identifiers_unique_across_splits",
            "train_slides_exact",
            "validation_sources_exact",
            "test_slides_exact",
        )
    )
    if not checks["all_checks_passed"]:
        raise AssertionError("Slide-independent validation failed: {}".format(checks))

    saved = json.loads((output / "split_validation.json").read_text())
    if not saved.get("all_checks_passed", False):
        raise AssertionError("Saved split validation is not successful")
    for key in (
        "fold",
        "train_slide_ids",
        "validation_source_slide_ids",
        "test_slide_ids",
        "train_test_disjoint",
        "validation_test_disjoint",
        "patch_identifiers_unique_across_splits",
    ):
        if saved.get(key) != checks.get(key):
            raise AssertionError(
                "Saved validation mismatch for {}: saved={} actual={}".format(
                    key, saved.get(key), checks.get(key)
                )
            )
    return checks


def materialize(config_path: Path) -> Path:
    config = _expand(yaml.safe_load(config_path.read_text()))
    if config.get("strategy") != "slide":
        raise ValueError("Metadata-only fold generation requires strategy: slide")
    if config.get("reuse_packed_data") is not True:
        raise ValueError("reuse_packed_data must be true")

    source = Path(config["source_dataset_root"]).resolve()
    output = Path(config["output_root"]).resolve()
    if source == output:
        raise ValueError("Output must differ from the immutable source dataset")
    if output.exists():
        checks = validate_dataset(output, config)
        print("Validated existing fold dataset: {}".format(output))
        print(json.dumps(checks, indent=2, sort_keys=True))
        return output

    for name in REUSED_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    selected = sorted(str(item) for item in config["slide_ids"])
    train_slides = [str(item) for item in config["train_slides"]]
    test_slides = [str(item) for item in config["test_slides"]]
    if set(train_slides) & set(test_slides):
        raise ValueError("Configured train and test slides overlap")
    if set(train_slides) | set(test_slides) != set(selected):
        raise ValueError("Explicit train/test slides must cover selected slide_ids")

    patch_info = pd.read_csv(source / "patch_info_with_split.csv")
    patch_info = patch_info[patch_info["slide_id"].astype(str).isin(selected)].copy()
    actual_selected = sorted(patch_info["slide_id"].astype(str).unique().tolist())
    if actual_selected != selected:
        raise ValueError(
            "Source selected slides differ: expected={} actual={}".format(
                selected, actual_selected
            )
        )

    split_frame, helper_manifest = assign_slide_split(
        patch_info=patch_info,
        axis=str(config.get("split_axis", "x")),
        train_frac=float(config.get("train_frac", 0.70)),
        valid_frac=float(config.get("valid_frac", 0.15)),
        test_frac=float(config.get("test_frac", 0.15)),
        train_frac_inside_train_slide=float(
            config.get("train_frac_inside_train_slide", 0.85)
        ),
        valid_frac_inside_train_slide=float(
            config.get("valid_frac_inside_train_slide", 0.15)
        ),
        boundary_margin=int(config.get("boundary_margin", 128)),
        train_slides=train_slides,
        valid_slides=[],
        test_slides=test_slides,
        random_seed=int(config.get("random_seed", 42)),
    )
    split_frame = split_frame[
        split_frame["split"].isin(["train", "valid", "test"])
    ].copy()
    split_frame = split_frame.sort_values(
        ["split", "slide_id", "packed_file_name"]
    ).reset_index(drop=True)

    source_counts = _load_counts(source)
    missing = sorted(set(split_frame["packed_file_name"]) - set(source_counts.index))
    if missing:
        raise KeyError("Missing source cell counts: {}".format(missing[:5]))

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}-tmp-".format(output.name), dir=str(output.parent))
    )
    try:
        for name in REUSED_FILES:
            os.symlink(str((source / name).resolve()), str(staging / name))

        split_frame.to_csv(staging / "patch_info_with_split.csv", index=False)
        class_columns = [column for column in source_counts.columns if column != "Image"]
        class_counts: Dict[str, Dict[str, int]] = {}
        patch_counts: Dict[str, int] = {}
        counts_per_slide: Dict[str, Dict[str, int]] = {}
        for split in ("train", "valid", "test"):
            names = split_frame.loc[
                split_frame["split"] == split, "packed_file_name"
            ].astype(str)
            fold_counts = source_counts.loc[names.tolist(), ["Image"] + class_columns]
            fold_counts.to_csv(staging / "cell_count_{}.csv".format(split), index=False)
            patch_counts[split] = int(len(names))
            class_counts[split] = {
                column: int(fold_counts[column].sum()) for column in class_columns
            }
            per_slide = (
                split_frame[split_frame["split"] == split]
                .groupby("slide_id")
                .size()
                .to_dict()
            )
            counts_per_slide[split] = {
                str(slide): int(count) for slide, count in per_slide.items()
            }

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fold": str(config["fold"]),
            "source_dataset_root": str(source),
            "output_root": str(output),
            "split_strategy": "slide",
            "split_axis": str(config.get("split_axis", "x")),
            "spatial_margin": int(config.get("boundary_margin", 128)),
            "train_fraction_inside_training_slides": float(
                config.get("train_frac_inside_train_slide", 0.85)
            ),
            "validation_fraction_inside_training_slides": float(
                config.get("valid_frac_inside_train_slide", 0.15)
            ),
            "train_slide_ids": sorted(train_slides),
            "validation_source_slide_ids": sorted(train_slides),
            "test_slide_ids": sorted(test_slides),
            "patch_counts": patch_counts,
            "counts_per_slide": counts_per_slide,
            "class_counts_per_split": class_counts,
            "reused_packed_payload": True,
            "reused_files": list(REUSED_FILES),
            "helper_manifest": helper_manifest,
        }
        with (staging / "split_manifest.yaml").open("w") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False)

        sets = _split_sets(split_frame)
        validation = {
            "fold": str(config["fold"]),
            "train_slide_ids": sorted(train_slides),
            "validation_source_slide_ids": sorted(train_slides),
            "test_slide_ids": sorted(test_slides),
            "train_test_disjoint": not bool(set(train_slides) & set(test_slides)),
            "validation_test_disjoint": not bool(
                set(train_slides) & set(test_slides)
            ),
            "patch_identifiers_unique_across_splits": not bool(
                (sets["train"] & sets["valid"])
                or (sets["train"] & sets["test"])
                or (sets["valid"] & sets["test"])
            ),
            "train_slides_exact": True,
            "validation_sources_exact": True,
            "test_slides_exact": True,
            "all_checks_passed": True,
        }
        _atomic_json(staging / "split_validation.json", validation)
        os.replace(str(staging), str(output))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    checks = validate_dataset(output, config)
    print("Created metadata-only fold dataset: {}".format(output))
    print(json.dumps(checks, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    materialize(args.config)


if __name__ == "__main__":
    main()
