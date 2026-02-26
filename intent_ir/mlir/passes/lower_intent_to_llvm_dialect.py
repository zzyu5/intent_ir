from __future__ import annotations

from ..module import IntentMLIRModule


def lower_intent_to_llvm_dialect(module: IntentMLIRModule, *, backend: str | None = None) -> IntentMLIRModule:
    """
    Strict hard-cut behavior:
    - keep textual LLVM IR as-is
    - reject non-LLVM input (legacy C/CUDA-C lowering removed)
    """
    text = str(module.module_text or "")
    if _looks_like_llvm_ir(text):
        out = _clone(module, module_text=text)
        out.meta["llvm_dialect_origin"] = "already_llvm_ir"
        if backend:
            out.meta["llvm_dialect_backend"] = str(backend)
        return out

    backend_tag = str(backend or "").strip().lower() or "generic"
    raise RuntimeError(
        "lower_intent_to_llvm_dialect: legacy C/CUDA-C fallback has been removed; "
        f"backend={backend_tag} requires textual LLVM IR input"
    )


def _looks_like_llvm_ir(text: str) -> bool:
    s = str(text or "")
    return ("; ModuleID" in s) or ("define " in s and "{" in s and "}" in s)


def _clone(module: IntentMLIRModule, *, module_text: str) -> IntentMLIRModule:
    return IntentMLIRModule(
        module_text=str(module_text or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
