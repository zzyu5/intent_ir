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
    return {
        "schema_version": "intent_mlir_toolchain_probe_v1",
        "ok": bool(mlir_opt.get("available") and mlir_translate.get("available")),
        "tools": {
            "mlir-opt": mlir_opt,
            "mlir-translate": mlir_translate,
        },
    }

