from __future__ import annotations

import re

from intent_ir.mlir.module import IntentMLIRModule


_CALL_RE = re.compile(r"llvm\.call\s+@__nv_exp2f\b")


def rewrite_cuda_nv_exp2f_calls(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    """
    Post-lowering cleanup for CUDA device IR.

    Some MLIR lowering paths materialize `math.exp2` as a libdevice call
    (`__nv_exp2f`). This is correct but can regress performance. Replace such
    calls with the LLVM intrinsic form (`llvm.intr.exp2`) so `llc` can lower it
    directly to `ex2.approx.f32` in PTX.

    This is a text-level rewrite because the repository intentionally avoids
    MLIR Python bindings.
    """
    text = str(module.module_text or "")
    if "@__nv_exp2f" not in text:
        return module

    rewritten = _CALL_RE.sub("llvm.intr.exp2", text)
    if rewritten == text:
        return module

    out = IntentMLIRModule(
        module_text=rewritten,
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    out.meta["rewrite_cuda_nv_exp2f_calls"] = True
    return out

