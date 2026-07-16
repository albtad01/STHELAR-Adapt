"""Convenience entrypoint for STHELAR CellViT adapter loading.

Example:

    import hub

    cellvit = hub.model("cellvit-sam-h-x40")
    hub.load_sthelar_adapter(cellvit, "ovary_5")
"""

from utils.cellvit_adapter_hub import (
    add_liver_adapter,
    add_ovary_adapter,
    add_tonsil_adapter,
    get_adapter_metadata,
    load_cellvit_base,
    load_sthelar_adapter,
    model,
    register_adapter_alias,
    resolve_adapter_checkpoint,
    verify_adapter_equivalence,
)

__all__ = [
    "add_liver_adapter",
    "add_ovary_adapter",
    "add_tonsil_adapter",
    "get_adapter_metadata",
    "load_cellvit_base",
    "load_sthelar_adapter",
    "model",
    "register_adapter_alias",
    "resolve_adapter_checkpoint",
    "verify_adapter_equivalence",
]
