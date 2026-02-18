from __future__ import annotations

from intent_ir.mlir.convert_from_intent import to_mlir
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def normalize_symbols(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    intent = to_intent(module)
    symbols = list(dict.fromkeys([str(s).strip() for s in list(module.symbols or []) if str(s).strip()]))
    intent.meta = dict(intent.meta or {})
    intent.meta["normalized_symbols"] = list(symbols)
    out = to_mlir(intent, provenance=dict(module.provenance or {}))
    out.meta = dict(out.meta or {})
    out.meta["normalized_symbols"] = list(symbols)
    return out

