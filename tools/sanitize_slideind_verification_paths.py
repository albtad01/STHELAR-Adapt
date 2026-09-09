#!/usr/bin/env python3
"""Make canonical slide-independent verification records release-portable."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO / "release" / "huggingface" / "adapters"
PATH_FIELDS = (
    "release_dir",
    "adapter_path",
    "base_checkpoint",
    "canonical_checkpoint",
    "training_config",
)


def portable(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        if field == "base_checkpoint":
            return f"models/pretrained/{path.name}"
        raise ValueError(f"Unexpected external path in {field}: {value}")


def main() -> None:
    records = sorted(
        path
        for path in ADAPTER_ROOT.rglob("verification.json")
        if "/slideind/" in path.as_posix()
        and (
            "/cellvit-sam-h-x40/" in path.as_posix()
            or "/cellvit-256-x40/" in path.as_posix()
        )
    )
    if len(records) != 22:
        raise RuntimeError(f"Expected 22 canonical records, found {len(records)}")
    changed = 0
    for record in records:
        data = json.loads(record.read_text())
        for field in PATH_FIELDS:
            data[field] = portable(data.get(field), field)
        rendered = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if record.read_text() != rendered:
            record.write_text(rendered)
            changed += 1
    print(json.dumps({"records": len(records), "changed": changed}, indent=2))


if __name__ == "__main__":
    main()
