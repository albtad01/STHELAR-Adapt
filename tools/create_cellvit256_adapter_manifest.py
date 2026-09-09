#!/usr/bin/env python3
"""Build the local verified CellViT-256 slide-experiment adapter manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "adapters" / "slide_exp_safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def previous_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["adapter_id"]: row for row in csv.DictReader(handle)}


def collect(root: Path, previous: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for verification_path in sorted(root.glob("*/verification.json")):
        package = verification_path.parent
        config_path = package / "adapter_config.json"
        checksums_path = package / "checksums.json"
        adapter_path = package / "adapter_model.safetensors"
        if not all(path.is_file() for path in (config_path, checksums_path, adapter_path)):
            continue
        config = json.loads(config_path.read_text())
        if config.get("base_model") != "CellViT-256-x40":
            continue
        checksums = json.loads(checksums_path.read_text())
        verification = json.loads(verification_path.read_text())
        adapter_id = str(config["adapter_id"])
        canonical = repo_path(str(verification["canonical_checkpoint"]))
        prior = previous.get(adapter_id, {})
        adapter_sha = sha256(adapter_path)
        expected_sha = checksums["exported"]["adapter_model.safetensors"]
        if adapter_sha != expected_sha:
            raise RuntimeError(f"Adapter checksum mismatch: {adapter_id}")
        outputs = verification.get("forward_output_comparison", [])
        output_exact = (
            {row.get("name") for row in outputs}
            == {"hv_map", "nuclei_binary_map", "nuclei_type_map", "tissue_types"}
            and all(row.get("exact") is True for row in outputs)
        )
        rows.append(
            {
                "adapter_id": adapter_id,
                "run_name": package.name,
                "base_model": config["base_model"],
                "base_checkpoint_sha256": str(config["base_checkpoint_sha256"]),
                "training_config": str(verification["training_config"]),
                "source_config_sha256": str(config["source_config_sha256"]),
                "canonical_checkpoint": str(verification["canonical_checkpoint"]),
                "canonical_checkpoint_exists": str(canonical.is_file()).lower(),
                "canonical_checkpoint_bytes": str(
                    canonical.stat().st_size
                    if canonical.is_file()
                    else prior.get("canonical_checkpoint_bytes", "")
                ),
                "canonical_checkpoint_sha256": str(
                    verification["canonical_checkpoint_sha256"]
                ),
                "adapter_path": relative(adapter_path),
                "adapter_bytes": str(adapter_path.stat().st_size),
                "adapter_sha256": adapter_sha,
                "trainable_parameters": str(verification["trainable_parameter_count"]),
                "state_reconstruction": str(verification["state_reconstruction"]),
                "forward_verification": str(verification["forward_verification"]),
                "all_required_outputs_exact": str(output_exact).lower(),
                "verification_status": str(verification["status"]),
                "verification_timestamp": str(verification["verification_timestamp"]),
                "safe_to_delete_checkpoint": str(
                    verification.get("status") == "ok"
                    and verification.get("state_reconstruction") == "exact_all_tensors"
                    and verification.get("forward_verification")
                    == "exact_all_output_tensors"
                    and output_exact
                ).lower(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "reports" / "cellvit256_slide_exp_adapter_manifest_verified.csv",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    rows = collect(args.root.expanduser().resolve(), previous_rows(output))
    if not rows:
        raise RuntimeError("No CellViT-256 adapter packages found")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps({"rows": len(rows), "output": relative(output)}, indent=2))


if __name__ == "__main__":
    main()
