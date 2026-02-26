from __future__ import annotations

from intent_ir.mlir.convert_from_intent import to_mlir
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def attach_provider_meta(module: IntentMLIRModule, **kwargs: object) -> IntentMLIRModule:
    provider = str(kwargs.get("provider") or "").strip()
    backend = str(kwargs.get("backend") or "").strip()
    incoming_meta = dict(module.meta or {})
    intent = to_intent(module)
    meta = dict(incoming_meta)
    meta.update(dict(intent.meta or {}))
    if provider:
        meta["provider"] = provider
    if backend:
        meta["backend_target"] = backend
    intent.meta = meta
    out = to_mlir(intent, provenance=dict(module.provenance or {}))
    merged_meta = dict(incoming_meta)
    merged_meta.update(dict(out.meta or {}))
    out.meta = merged_meta
    if provider:
        out.meta["provider"] = provider
    if backend:
        out.meta["backend_target"] = backend
    return out
