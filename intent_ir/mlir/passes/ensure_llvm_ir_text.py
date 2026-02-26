from __future__ import annotations

from ..module import IntentMLIRModule

def ensure_llvm_ir_text(module: IntentMLIRModule, *, backend: str | None = None) -> IntentMLIRModule:
    """
    Ensure downstream LLVM tools always receive textual LLVM IR.

    Stub synthesis has been removed. Non-LLVM payloads fail fast.
    """
    text = str(module.module_text or "")
    if _looks_like_llvm_ir(text):
        out = _clone(module)
        out.meta["llvm_text_origin"] = "translated"
        if backend:
            out.meta["llvm_text_backend"] = str(backend)
        return out

    raise RuntimeError("ensure_llvm_ir_text: textual LLVM IR missing and stub synthesis disabled")


def _looks_like_llvm_ir(text: str) -> bool:
    s = str(text or "")
    return ("define " in s and "{" in s and "}" in s) or ("; ModuleID" in s)


def _clone(module: IntentMLIRModule) -> IntentMLIRModule:
    return IntentMLIRModule(
        module_text=str(module.module_text or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
