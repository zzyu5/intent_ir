"""
Macro expansion pass: semantic macro ops -> primitive IntentIR ops.

Why this exists:
- LLMs should emit compact, semantic macro ops when appropriate.
- The compiler lowers/expands them into the primitive IntentIR op set before:
  - Task5 interpreter diff
  - Task6 RVV lowering

Extensibility:
- Each macro op has a dedicated lowering implementation under `intent_ir/macro_lowering/`.
- Shared "micro-op" builders live in `intent_ir/macro_lowering/common.py`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List

from ..ir.ir_types import IntentFunction, TensorType
from .macro_lowering.common import LoweringBuilder
from .macro_lowering.registry import lower_macro_op, supports_macro


def expand_macros(intent: IntentFunction) -> IntentFunction:
    """
    Expand all supported macro ops in `intent` and return a new IntentFunction.
    """
    if not any(supports_macro(op.op) for op in intent.ops):
        return intent

    ops_out: List = []
    tensors: Dict[str, TensorType] = dict(intent.tensors)
    b = LoweringBuilder(tensors=tensors, ops_out=ops_out)

    for op in intent.ops:
        if supports_macro(op.op):
            lower_macro_op(b, op)
        else:
            b.append_existing_op(op)

    expanded = replace(intent, tensors=tensors, ops=ops_out)
    expanded.validate()
    return expanded


def expand_macros_json(intent_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON-level macro expansion helper used by pipeline/runtime tooling.

    This keeps script callers on payload-level data flow and avoids ad-hoc
    IntentFunction reconstruction in each script.
    """
    if not isinstance(intent_json, dict):
        raise TypeError("intent_json must be an object")
    intent = IntentFunction.from_json_dict(dict(intent_json))
    return expand_macros(intent).to_json_dict()


__all__ = ["expand_macros", "expand_macros_json"]
