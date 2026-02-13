"""
Canonical operator set for IntentIR.

Derived from declarative OpSpec metadata in `intent_ir.ops.specs` so that
validation/interpreter/backend capability queries can share one source.
"""

from __future__ import annotations

from .specs import OP_SPEC_INDEX, all_op_specs


CORE_OPS: set[str] = {spec.name for spec in all_op_specs() if spec.tier == "core"}
EXPERIMENTAL_OPS: set[str] = {spec.name for spec in all_op_specs() if spec.tier == "experimental"}
MACRO_OPS: set[str] = {spec.name for spec in all_op_specs() if spec.tier == "macro"}
SUPPORTED_OPS: set[str] = set().union(CORE_OPS, EXPERIMENTAL_OPS, MACRO_OPS)


__all__ = [
    "SUPPORTED_OPS",
    "CORE_OPS",
    "EXPERIMENTAL_OPS",
    "MACRO_OPS",
    "OP_SPEC_INDEX",
]
