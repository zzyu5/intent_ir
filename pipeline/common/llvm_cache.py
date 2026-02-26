from __future__ import annotations

from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parents[2]
_INDEX_LOCK = Lock()
_LLVM_FILE_INDEX: dict[str, Path] | None = None


def _is_textual_llvm_ir(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    s = str(text or "")
    return ("target triple" in s and "define " in s) or ("; ModuleID" in s)


def _cache_roots() -> list[Path]:
    return [
        ROOT / "artifacts" / "flaggems_matrix" / "daily",
        ROOT / "artifacts" / "flaggems_triton_full_pipeline",
        ROOT / "artifacts" / "triton_full_pipeline",
        ROOT / "artifacts" / "tilelang_full_pipeline",
        ROOT / "artifacts" / "cuda_full_pipeline",
    ]


def _build_llvm_file_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
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
                prev = index.get(name)
                if prev is None:
                    index[name] = path
                    continue
                try:
                    path_mtime = float(path.stat().st_mtime)
                except Exception:
                    path_mtime = 0.0
                try:
                    prev_mtime = float(prev.stat().st_mtime)
                except Exception:
                    prev_mtime = 0.0
                if path_mtime >= prev_mtime:
                    index[name] = path
    return index


def _llvm_file_index() -> dict[str, Path]:
    global _LLVM_FILE_INDEX
    with _INDEX_LOCK:
        if _LLVM_FILE_INDEX is None:
            _LLVM_FILE_INDEX = _build_llvm_file_index()
        return dict(_LLVM_FILE_INDEX)


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
        if path.is_file() and _is_textual_llvm_ir(path):
            return str(path)

    idx = _llvm_file_index()
    for name in (f"{prefix}.mlir", f"{prefix}.ll"):
        path = idx.get(name)
        if path is not None and path.is_file() and _is_textual_llvm_ir(path):
            return str(path)
    return None


def _reset_llvm_cache_index_for_tests() -> None:
    global _LLVM_FILE_INDEX
    with _INDEX_LOCK:
        _LLVM_FILE_INDEX = None

