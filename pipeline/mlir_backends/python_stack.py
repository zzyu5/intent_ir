from __future__ import annotations

import json
import os
from pathlib import Path


_CUDA_WAVE_KERNELS: dict[tuple[str, str], set[str]] = {}
_RVV_WAVE_KERNELS: dict[tuple[str, str], set[str]] = {}


def cuda_real_mlir_wave_name(*, real_mlir_enabled: bool) -> str:
    raw = str(os.getenv("INTENTIR_CUDA_REAL_MLIR_WAVE", "")).strip().lower()
    if raw:
        return raw
    return "wave25" if real_mlir_enabled else ""


def rvv_real_mlir_wave_name(*, real_mlir_enabled: bool) -> str:
    raw = str(os.getenv("INTENTIR_RVV_REAL_MLIR_WAVE", "")).strip().lower()
    if raw:
        return raw
    return "wave22" if real_mlir_enabled else ""


def cuda_real_mlir_wave_kernels(*, root: Path, wave: str) -> set[str]:
    wave_name = str(wave or "").strip().lower()
    if not wave_name:
        return set()
    key = (str(root), wave_name)
    cached = _CUDA_WAVE_KERNELS.get(key)
    if cached is not None:
        return cached
    path = Path(root) / "workflow" / "flaggems" / "state" / f"cuda_real_mlir_{wave_name}_kernels.json"
    kernels: set[str] = set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("kernels")
            if isinstance(rows, list):
                for item in rows:
                    name = str(item).strip()
                    if name:
                        kernels.add(name)
    except Exception:
        kernels = set()
    _CUDA_WAVE_KERNELS[key] = kernels
    return kernels


def rvv_real_mlir_wave_kernels(*, root: Path, wave: str) -> set[str]:
    wave_name = str(wave or "").strip().lower()
    if not wave_name:
        return set()
    key = (str(root), wave_name)
    cached = _RVV_WAVE_KERNELS.get(key)
    if cached is not None:
        return cached
    path = Path(root) / "workflow" / "flaggems" / "state" / f"rvv_real_mlir_{wave_name}_kernels.json"
    kernels: set[str] = set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("kernels")
            if isinstance(rows, list):
                for item in rows:
                    name = str(item).strip()
                    if name:
                        kernels.add(name)
    except Exception:
        kernels = set()
    _RVV_WAVE_KERNELS[key] = kernels
    return kernels
