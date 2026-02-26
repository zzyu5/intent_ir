from __future__ import annotations

from pathlib import Path
import re
from threading import Lock


ROOT = Path(__file__).resolve().parents[2]
_INDEX_LOCK = Lock()
_LLVM_FILE_INDEX: dict[str, list[Path]] | None = None


def _is_textual_llvm_ir(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    s = str(text or "")
    return ("target triple" in s and "define " in s) or ("; ModuleID" in s)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _llvm_target_triple(text: str) -> str:
    m = re.search(r'target\s+triple\s*=\s*"([^"]+)"', str(text or ""))
    return str(m.group(1) if m else "").strip().lower()


def _is_valid_cached_llvm_for_pipeline(path: Path, *, llvm_pipeline: str) -> bool:
    if not path.is_file() or not _is_textual_llvm_ir(path):
        return False
    try:
        text = _read_text(path)
    except Exception:
        return False
    triple = _llvm_target_triple(text)
    p = str(llvm_pipeline or "").strip().lower()
    if p.endswith("cuda_llvm"):
        if "nvptx" not in triple:
            return False
        if "define dso_local i32 @main(" in text:
            return False
        if "@intentir_runtime_init" in text:
            return False
        return True
    if p.endswith("rvv_llvm"):
        return "riscv" in triple
    return True


def _cache_roots() -> list[Path]:
    return [
        ROOT / "artifacts" / "flaggems_matrix" / "daily",
        ROOT / "artifacts" / "flaggems_triton_full_pipeline",
        ROOT / "artifacts" / "triton_full_pipeline",
        ROOT / "artifacts" / "tilelang_full_pipeline",
        ROOT / "artifacts" / "cuda_full_pipeline",
    ]


def _build_llvm_file_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    patterns = [
        "*.intentir.intentdialect.*_llvm.mlir",
        "*.intentir.intentdialect.*_llvm.ll",
    ]
    for base in _cache_roots():
        if not base.is_dir():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if not path.is_file():
                    continue
                name = path.name
                bucket = index.setdefault(name, [])
                bucket.append(path)
    for name, bucket in index.items():
        bucket.sort(key=lambda p: float(p.stat().st_mtime), reverse=True)
    return index


def _llvm_file_index() -> dict[str, list[Path]]:
    global _LLVM_FILE_INDEX
    with _INDEX_LOCK:
        if _LLVM_FILE_INDEX is None:
            _LLVM_FILE_INDEX = _build_llvm_file_index()
        return {k: list(v) for k, v in _LLVM_FILE_INDEX.items()}


def discover_cached_downstream_llvm_module_path(
    *,
    spec_name: str,
    llvm_pipeline: str,
    current_out_dir: Path,
) -> str | None:
    prefix = f"{str(spec_name)}.intentir.intentdialect.{str(llvm_pipeline)}"
    local_candidates = [
        Path(current_out_dir) / f"{prefix}.mlir",
        Path(current_out_dir) / f"{prefix}.ll",
    ]
    for path in local_candidates:
        if _is_valid_cached_llvm_for_pipeline(path, llvm_pipeline=str(llvm_pipeline)):
            return str(path)

    idx = _llvm_file_index()
    for name in (f"{prefix}.mlir", f"{prefix}.ll"):
        paths = list(idx.get(name) or [])
        for path in paths:
            if _is_valid_cached_llvm_for_pipeline(path, llvm_pipeline=str(llvm_pipeline)):
                return str(path)
    return None


def _reset_llvm_cache_index_for_tests() -> None:
    global _LLVM_FILE_INDEX
    with _INDEX_LOCK:
        _LLVM_FILE_INDEX = None
