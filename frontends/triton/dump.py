"""
Triton kernel dumping helpers.

We rely on Triton's `TRITON_KERNEL_DUMP=1` mechanism to materialize TTIR/LLVM IR
artifacts during JIT compilation, and isolate dump/cache dirs per-kernel to
avoid cache hits suppressing dumps.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple


def prepare_dump_and_cache_dirs(base_dir: Path, kernel_name: str, *, clean: bool = True) -> Tuple[Path, Path]:
    dump_dir = base_dir / "_triton_dump" / kernel_name
    cache_dir = base_dir / "_triton_cache" / kernel_name
    if clean:
        shutil.rmtree(dump_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(dump_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    return dump_dir, cache_dir


def _find_latest_artifact(dump_dir: Path, *, suffix: str, name_hint: str) -> Optional[Path]:
    hits = sorted(dump_dir.rglob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in hits:
        if name_hint in p.name:
            return p
    return hits[0] if hits else None


def find_latest_ttir(dump_dir: Path, name_hint: str) -> Optional[Path]:
    return _find_latest_artifact(dump_dir, suffix=".ttir", name_hint=name_hint)


def find_latest_ttgir(dump_dir: Path, name_hint: str) -> Optional[Path]:
    return _find_latest_artifact(dump_dir, suffix=".ttgir", name_hint=name_hint)


def find_latest_ptx(dump_dir: Path, name_hint: str) -> Optional[Path]:
    return _find_latest_artifact(dump_dir, suffix=".ptx", name_hint=name_hint)


def find_latest_llir(dump_dir: Path, name_hint: str) -> Optional[Path]:
    return _find_latest_artifact(dump_dir, suffix=".llir", name_hint=name_hint)


def find_latest_cubin(dump_dir: Path, name_hint: str) -> Optional[Path]:
    return _find_latest_artifact(dump_dir, suffix=".cubin", name_hint=name_hint)


__all__ = [
    "prepare_dump_and_cache_dirs",
    "find_latest_ttir",
    "find_latest_ttgir",
    "find_latest_ptx",
    "find_latest_llir",
    "find_latest_cubin",
]
