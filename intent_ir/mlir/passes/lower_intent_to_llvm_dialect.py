from __future__ import annotations

import re
from typing import Any

from ..module import IntentMLIRModule

_INVALID_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


def lower_intent_to_llvm_dialect(module: IntentMLIRModule, *, backend: str | None = None) -> IntentMLIRModule:
    """
    Lower Intent dialect text to a minimal LLVM dialect module.

    This pass is a migration bridge for the LLVM pipeline:
    - keep intent_json/meta provenance intact
    - produce LLVM-dialect MLIR so `mlir-translate --mlir-to-llvmir` can run
    """
    kernel = _kernel_name(module)
    llvm_mod = _emit_min_llvm_dialect(kernel)
    out = _clone(module)
    out.module_text = llvm_mod
    out.meta["llvm_dialect_origin"] = "intent_bridge"
    out.meta["llvm_dialect_kernel"] = kernel
    if backend:
        out.meta["llvm_dialect_backend"] = str(backend)
    return out


def _emit_min_llvm_dialect(kernel_name: str) -> str:
    return (
        "module {\n"
        f"  llvm.func @{kernel_name}() {{\n"
        "    llvm.return\n"
        "  }\n"
        "}\n"
    )


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


def _clone(module: IntentMLIRModule) -> IntentMLIRModule:
    return IntentMLIRModule(
        module_text=str(module.module_text or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )

