from __future__ import annotations

from intent_ir.mlir.convert_from_intent import to_mlir
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def attach_provider_meta(module: IntentMLIRModule, **kwargs: object) -> IntentMLIRModule:
    provider = str(kwargs.get("provider") or "").strip()
    backend = str(kwargs.get("backend") or "").strip()
    intent = to_intent(module)
    meta = dict(intent.meta or {})
    if provider:
        meta["provider"] = provider
    if backend:
        meta["backend_target"] = backend
    intent.meta = meta
    out = to_mlir(intent, provenance=dict(module.provenance or {}))
    out.meta = dict(out.meta or {})
    if provider:
        out.meta["provider"] = provider
    if backend:
        out.meta["backend_target"] = backend
    return out

