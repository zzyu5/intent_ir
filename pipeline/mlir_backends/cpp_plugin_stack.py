from __future__ import annotations

import json
import os
from pathlib import Path


_CPP_WAVE_KERNELS: dict[tuple[str, str], set[str]] = {}


def compiler_cpp_wave_name() -> str:
    return str(os.getenv("INTENTIR_COMPILER_CPP_WAVE", "wave2")).strip().lower()


def compiler_cpp_miss_policy() -> str:
    return str(os.getenv("INTENTIR_COMPILER_CPP_MISS_POLICY", "skip")).strip().lower()


def compiler_cpp_wave_kernels(*, root: Path, wave: str | None = None) -> set[str]:
    wave_name = str(wave or compiler_cpp_wave_name()).strip().lower()
    if not wave_name:
        return set()
    key = (str(root), wave_name)
    cached = _CPP_WAVE_KERNELS.get(key)
    if cached is not None:
        return cached
    path = Path(root) / "workflow" / "flaggems" / "state" / f"compiler_cpp_{wave_name}_kernels.json"
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
    _CPP_WAVE_KERNELS[key] = kernels
    return kernels
