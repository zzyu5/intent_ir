"""
IntentIR primitive catalog for architecture-quality checks.

This catalog defines which ops are valid shared primitives. Provider-specific
names must not be introduced into core IntentIR mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .specs import OP_SPEC_INDEX


@dataclass(frozen=True)
class PrimitiveInfo:
    name: str
    tier: str
    kind: str
    reusable: bool


PRIMITIVE_CATALOG: dict[str, PrimitiveInfo] = {
    name: PrimitiveInfo(
        name=name,
        tier=str(spec.tier),
        kind=str(spec.kind),
        reusable=(str(spec.tier) != "macro"),
    )
    for name, spec in OP_SPEC_INDEX.items()
}


def primitive_names(*, include_macro: bool = False) -> set[str]:
    if include_macro:
        return set(PRIMITIVE_CATALOG.keys())
    return {name for name, info in PRIMITIVE_CATALOG.items() if bool(info.reusable)}


def primitive_info(op_name: str) -> PrimitiveInfo | None:
    return PRIMITIVE_CATALOG.get(str(op_name))


def is_allowed_primitive(op_name: str, *, include_macro: bool = False) -> bool:
    info = primitive_info(op_name)
    if info is None:
        return False
    return bool(info.reusable or include_macro)


def catalog_summary() -> Mapping[str, int]:
    by_tier: dict[str, int] = {}
    for info in PRIMITIVE_CATALOG.values():
        by_tier[info.tier] = int(by_tier.get(info.tier, 0)) + 1
    return {"total": len(PRIMITIVE_CATALOG), **{f"tier_{k}": v for k, v in sorted(by_tier.items())}}


__all__ = [
    "PrimitiveInfo",
    "PRIMITIVE_CATALOG",
    "primitive_names",
    "primitive_info",
    "is_allowed_primitive",
    "catalog_summary",
]
