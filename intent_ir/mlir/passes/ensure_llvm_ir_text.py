from __future__ import annotations

import re
from typing import Any

from ..module import IntentMLIRModule

_INVALID_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


def ensure_llvm_ir_text(module: IntentMLIRModule, *, backend: str | None = None) -> IntentMLIRModule:
    """
    Ensure downstream LLVM tools always receive textual LLVM IR.

    Migration note:
    - If `mlir-translate` already produced LLVM IR text, keep it untouched.
    - Otherwise synthesize a minimal, valid LLVM IR wrapper using intent metadata.
    """
    text = str(module.module_text or "")
    if _looks_like_llvm_ir(text):
        out = _clone(module)
        out.meta["llvm_text_origin"] = "translated"
        return out

    kernel_name = _kernel_name(module)
    stub_ir = _emit_stub_llvm_ir(kernel_name)
    out = _clone(module)
    out.module_text = stub_ir
    out.meta["llvm_text_origin"] = "intent_stub"
    out.meta["llvm_stub_kernel"] = kernel_name
    if backend:
        out.meta["llvm_stub_backend"] = str(backend)
    return out


def _looks_like_llvm_ir(text: str) -> bool:
    s = str(text or "")
    return ("define " in s and "{" in s and "}" in s) or ("; ModuleID" in s)


def _kernel_name(module: IntentMLIRModule) -> str:
    name = ""
    payload = module.intent_json if isinstance(module.intent_json, dict) else {}
    if isinstance(payload, dict):
        name = str(payload.get("name") or "").strip()
    if not name:
        name = str((module.meta or {}).get("kernel_name") or "").strip()
    if not name:
        name = "intentir_kernel"
    name = _INVALID_IDENT_RE.sub("_", name).strip("_")
    if not name:
        name = "intentir_kernel"
    if name[0].isdigit():
        name = f"kernel_{name}"
    return name


def _emit_stub_llvm_ir(kernel_name: str) -> str:
    return (
        f'; ModuleID = "intentir::{kernel_name}"\n'
        'source_filename = "intentir"\n\n'
        f"define void @{kernel_name}() {{\n"
        "entry:\n"
        "  ret void\n"
        "}\n"
    )


def _clone(module: IntentMLIRModule) -> IntentMLIRModule:
    return IntentMLIRModule(
        module_text=str(module.module_text or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )

