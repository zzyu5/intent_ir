from __future__ import annotations

from intent_ir.macros import expand_macros
from intent_ir.mlir.convert_from_intent import to_mlir
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def expand_macros_intent(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    """
    Expand macro ops into primitive compositions on the dual-track bridge.
    """
    intent = to_intent(module)
    expanded = expand_macros(intent)
    out = to_mlir(expanded, provenance=dict(module.provenance or {}))
    out.meta = dict(out.meta or {})
    out.meta["macro_expanded"] = True
    out.meta["macro_expand_source"] = "intent_ir.macros.expand_macros"
    return out

