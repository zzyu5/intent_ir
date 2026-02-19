from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


_LLVM_VERSIONS = (19, 18, 17, 16, 15, 14, 13)


def _llvm_bindirs() -> list[Path]:
    out: list[Path] = []
    cfg_names = ["llvm-config", *[f"llvm-config-{v}" for v in _LLVM_VERSIONS]]
    seen: set[str] = set()
    for cfg in cfg_names:
        cfg_path = shutil.which(cfg)
        if not cfg_path:
            continue
        try:
            p = subprocess.run([cfg_path, "--bindir"], capture_output=True, text=True)
            if p.returncode != 0:
                continue
            bindir = str(p.stdout or "").strip()
            if not bindir:
                continue
            key = str(Path(bindir).resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(key))
        except Exception:
            continue
    return out


def _candidate_names(base: str, env_var: str) -> list[str]:
    out: list[str] = []
    env_val = str(os.getenv(env_var, "") or "").strip()
    if env_val:
        out.append(env_val)
    out.append(base)
    out.extend([f"{base}-{v}" for v in _LLVM_VERSIONS])
    dedup: list[str] = []
    seen: set[str] = set()
    for x in out:
        k = str(x).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return dedup


def _probe_tool(base: str, *, env_var: str) -> dict[str, Any]:
    candidates = _candidate_names(base, env_var)
    extra_bindirs = _llvm_bindirs()
    checked: list[str] = []
    chosen_path = ""
    chosen_name = ""
    for cand in candidates:
        cand_path = Path(cand)
        if cand_path.is_file() and os.access(str(cand_path), os.X_OK):
            chosen_path = str(cand_path)
            chosen_name = cand
            checked.append(cand)
            break
        w = shutil.which(cand)
        if w:
            chosen_path = str(w)
            chosen_name = cand
            checked.append(cand)
            break
        # Try llvm-config discovered bindirs for versioned binary layouts.
        found = ""
        for bindir in extra_bindirs:
            p = bindir / cand
            if p.is_file() and os.access(str(p), os.X_OK):
                found = str(p)
                break
        checked.append(cand)
        if found:
            chosen_path = found
            chosen_name = cand
            break
    if not chosen_path:
        return {
            "available": False,
            "path": "",
            "version": "",
            "resolved_name": "",
            "env_var": str(env_var),
            "candidates_checked": checked,
        }
    version = ""
    try:
        p = subprocess.run([chosen_path, "--version"], capture_output=True, text=True)
        if p.returncode == 0:
            version = str((p.stdout or p.stderr or "").splitlines()[0]).strip()
    except Exception:
        version = ""
    return {
        "available": True,
        "path": str(chosen_path),
        "version": str(version),
        "resolved_name": str(chosen_name),
        "env_var": str(env_var),
        "candidates_checked": checked,
    }


def detect_mlir_toolchain() -> dict[str, Any]:
    mlir_opt = _probe_tool("mlir-opt", env_var="INTENTIR_MLIR_OPT")
    mlir_translate = _probe_tool("mlir-translate", env_var="INTENTIR_MLIR_TRANSLATE")
    llvm_as = _probe_tool("llvm-as", env_var="INTENTIR_LLVM_AS")
    llvm_opt = _probe_tool("opt", env_var="INTENTIR_LLVM_OPT")
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
        "required_env_vars": {
            "mlir-opt": "INTENTIR_MLIR_OPT",
            "mlir-translate": "INTENTIR_MLIR_TRANSLATE",
            "llvm-as": "INTENTIR_LLVM_AS",
            "opt": "INTENTIR_LLVM_OPT",
        },
        "install_hint": (
            ""
            if len(missing_required) == 0
            else (
                "Missing required MLIR/LLVM tools: "
                + ", ".join(missing_required)
                + ". Install toolchain packages or set INTENTIR_MLIR_OPT / INTENTIR_MLIR_TRANSLATE / "
                + "INTENTIR_LLVM_AS / INTENTIR_LLVM_OPT to executable paths."
            )
        ),
        "tools": tools,
    }
