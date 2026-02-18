from __future__ import annotations

from intent_ir.ir.canonicalize import canonicalize_for_consistency
from intent_ir.mlir.convert_from_intent import to_mlir
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def canonicalize_intent(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    intent = to_intent(module)
    canonical = canonicalize_for_consistency(intent)
    out = to_mlir(canonical, provenance=dict(module.provenance or {}))
    out.meta = dict(out.meta or {})
    out.meta["canonicalized"] = True
    return out

