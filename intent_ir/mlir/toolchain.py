from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


_LLVM_VERSIONS = (19, 18, 17, 16, 15, 14, 13)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TOOLCHAIN_ROOT = _REPO_ROOT / "artifacts" / "toolchains"


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


def _intentir_local_bindirs() -> list[Path]:
    """
    Probe repository-local toolchain locations so MLIR tools can be used
    without system-wide installation.
    """
    roots: list[Path] = []
    env_root = str(os.getenv("INTENTIR_MLIR_TOOLCHAIN_ROOT", "")).strip()
    if env_root:
        roots.append(Path(env_root))
    roots.append(_DEFAULT_TOOLCHAIN_ROOT / "mlir-current")
    roots.extend(sorted(_DEFAULT_TOOLCHAIN_ROOT.glob("mlir-*")))

    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        candidates: list[Path] = []
        candidates.append(root / "bin")
        candidates.append(root / "usr" / "bin")
        candidates.extend(sorted((root / "usr" / "lib").glob("llvm-*/bin")))
        candidates.extend(sorted((root / "lib").glob("llvm-*/bin")))
        for p in candidates:
            if not p.is_dir():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p.resolve())
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
    extra_bindirs = _llvm_bindirs() + _intentir_local_bindirs()
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


def _probe_tool_aliases(aliases: list[str], *, env_var: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "path": "",
        "version": "",
        "resolved_name": "",
        "env_var": str(env_var),
        "candidates_checked": [],
    }
    for base in list(aliases or []):
        probe = _probe_tool(str(base), env_var=env_var)
        out["candidates_checked"] = list(out.get("candidates_checked") or []) + list(probe.get("candidates_checked") or [])
        if bool(probe.get("available")):
            return probe
    return out


def detect_mlir_toolchain() -> dict[str, Any]:
    mlir_opt = _probe_tool("mlir-opt", env_var="INTENTIR_MLIR_OPT")
    mlir_translate = _probe_tool("mlir-translate", env_var="INTENTIR_MLIR_TRANSLATE")
    llvm_as = _probe_tool("llvm-as", env_var="INTENTIR_LLVM_AS")
    llvm_opt = _probe_tool("opt", env_var="INTENTIR_LLVM_OPT")
    llc = _probe_tool("llc", env_var="INTENTIR_LLC")
    ptxas = _probe_tool("ptxas", env_var="INTENTIR_PTXAS")
    clang = _probe_tool("clang", env_var="INTENTIR_CLANG")
    rvv_cc = _probe_tool_aliases(
        ["riscv64-linux-gnu-gcc", "riscv64-unknown-linux-gnu-gcc", "clang"],
        env_var="INTENTIR_RVV_CC",
    )
    rvv_ld = _probe_tool_aliases(
        ["riscv64-linux-gnu-ld", "ld.lld", "ld"],
        env_var="INTENTIR_RVV_LD",
    )
    tools = {
        "mlir-opt": mlir_opt,
        "mlir-translate": mlir_translate,
        "llvm-as": llvm_as,
        "opt": llvm_opt,
        "llc": llc,
        "ptxas": ptxas,
        "clang": clang,
        "rvv_cc": rvv_cc,
        "rvv_ld": rvv_ld,
    }
    required_tools = ("mlir-opt", "mlir-translate", "llvm-as", "opt")
    missing_required = [name for name in required_tools if not bool((tools.get(name) or {}).get("available"))]
    cuda_required_tools = ("llc",)
    cuda_optional_tools = ("ptxas",)
    rvv_required_tools = ("llc", "rvv_cc", "rvv_ld")
    rvv_optional_tools = ("clang",)
    missing_cuda_required = [name for name in cuda_required_tools if not bool((tools.get(name) or {}).get("available"))]
    missing_rvv_required = [name for name in rvv_required_tools if not bool((tools.get(name) or {}).get("available"))]
    backend_install_hints: dict[str, str] = {}
    if missing_cuda_required:
        backend_install_hints["cuda"] = (
            "Missing CUDA downstream LLVM tools: "
            + ", ".join(missing_cuda_required)
            + ". Set INTENTIR_LLC (and optionally INTENTIR_PTXAS)."
        )
    if missing_rvv_required:
        backend_install_hints["rvv"] = (
            "Missing RVV downstream LLVM tools: "
            + ", ".join(missing_rvv_required)
            + ". Set INTENTIR_LLC / INTENTIR_RVV_CC / INTENTIR_RVV_LD."
        )
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
            "llc": "INTENTIR_LLC",
            "ptxas": "INTENTIR_PTXAS",
            "clang": "INTENTIR_CLANG",
            "rvv_cc": "INTENTIR_RVV_CC",
            "rvv_ld": "INTENTIR_RVV_LD",
        },
        "cuda_required_tools": list(cuda_required_tools),
        "cuda_optional_tools": list(cuda_optional_tools),
        "cuda_toolchain_ok": bool(len(missing_cuda_required) == 0),
        "missing_cuda_required_tools": list(missing_cuda_required),
        "rvv_required_tools": list(rvv_required_tools),
        "rvv_optional_tools": list(rvv_optional_tools),
        "rvv_toolchain_ok": bool(len(missing_rvv_required) == 0),
        "missing_rvv_required_tools": list(missing_rvv_required),
        "backend_install_hints": backend_install_hints,
        "local_toolchain_roots": [
            str(_DEFAULT_TOOLCHAIN_ROOT / "mlir-current"),
            str(_DEFAULT_TOOLCHAIN_ROOT / "mlir-*"),
        ],
        "install_hint": (
            ""
            if len(missing_required) == 0
            else (
                "Missing required MLIR/LLVM tools: "
                + ", ".join(missing_required)
                + ". Install toolchain packages or set INTENTIR_MLIR_OPT / INTENTIR_MLIR_TRANSLATE / "
                + "INTENTIR_LLVM_AS / INTENTIR_LLVM_OPT to executable paths. "
                + "You can also run `python scripts/intentir.py mlir provision-toolchain --version 14 --force`."
            )
        ),
        "tools": tools,
    }
