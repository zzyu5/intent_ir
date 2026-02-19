from __future__ import annotations

import shutil
import subprocess
from typing import Any


def _probe_tool(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"available": False, "path": "", "version": ""}
    version = ""
    try:
        p = subprocess.run([path, "--version"], capture_output=True, text=True)
        if p.returncode == 0:
            version = str((p.stdout or p.stderr or "").splitlines()[0]).strip()
    except Exception:
        version = ""
    return {"available": True, "path": str(path), "version": str(version)}


def detect_mlir_toolchain() -> dict[str, Any]:
    mlir_opt = _probe_tool("mlir-opt")
    mlir_translate = _probe_tool("mlir-translate")
    llvm_as = _probe_tool("llvm-as")
    llvm_opt = _probe_tool("opt")
    tools = {
        "mlir-opt": mlir_opt,
        "mlir-translate": mlir_translate,
        "llvm-as": llvm_as,
        "opt": llvm_opt,
    }
    required_tools = ("mlir-opt", "mlir-translate", "llvm-as", "opt")
    missing_required = [name for name in required_tools if not bool((tools.get(name) or {}).get("available"))]
    return {
        "schema_version": "intent_mlir_toolchain_probe_v1",
        # `ok` is the hard requirement used by migration gates.
        "ok": bool(len(missing_required) == 0),
        # Keep an explicit transitional signal for old two-tool checks.
        "mlir_core_ok": bool(mlir_opt.get("available") and mlir_translate.get("available")),
        "required_tools": list(required_tools),
        "missing_required_tools": list(missing_required),
        "tools": tools,
    }
