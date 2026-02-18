from __future__ import annotations

from intent_ir.ir.canonicalize import canonicalize_for_consistency
from intent_ir.mlir.convert_from_intent import to_mlir
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def backend_legalize(module: IntentMLIRModule, *, backend: str | None = None, **_: object) -> IntentMLIRModule:
    """
    Backend-facing legalize pass on Intent dialect bridge representation.
    """
    backend_name = str(backend or "").strip() or "generic"
    intent = to_intent(module)
    legalized = canonicalize_for_consistency(intent)
    meta = dict(legalized.meta or {})
    meta["backend_legalized"] = True
    meta["backend_target"] = backend_name
    legalized.meta = meta

    out = to_mlir(legalized, provenance=dict(module.provenance or {}))
    out.meta = dict(out.meta or {})
    out.meta["backend_legalized"] = True
    out.meta["backend_target"] = backend_name
    return out

