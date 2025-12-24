from __future__ import annotations

from typing import Callable, Dict

from ..ir_types import IntentIRValidationError, Op
from .common import LoweringBuilder
from .upsample_bicubic2d_aa import lower_upsample_bicubic2d_aa


MacroLowerFn = Callable[[LoweringBuilder, Op], None]

_REGISTRY: Dict[str, MacroLowerFn] = {
    "upsample_bicubic2d_aa": lower_upsample_bicubic2d_aa,
}


def supports_macro(op_name: str) -> bool:
    return op_name in _REGISTRY


def lower_macro_op(b: LoweringBuilder, op: Op) -> None:
    fn = _REGISTRY.get(op.op)
    if fn is None:
        raise IntentIRValidationError(f"macro_lowering: unsupported macro op: {op.op}")
    fn(b, op)


__all__ = ["supports_macro", "lower_macro_op"]

