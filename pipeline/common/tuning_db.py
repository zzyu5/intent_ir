from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CUDA_TUNING_DB = ROOT / "workflow" / "flaggems" / "state" / "tuning_db" / "cuda.jsonl"
_CACHE: dict[tuple[str, str], tuple[dict[tuple[str, str], "TuningDBEntry"], str]] = {}


@dataclass(frozen=True)
class TuningDBEntry:
    bindings: dict[str, Any]
    kernel_kind: str = ""


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _default_db_path_for_backend(backend: str) -> Path | None:
    b = str(backend or "").strip().lower()
    if b == "cuda":
        return _DEFAULT_CUDA_TUNING_DB if _DEFAULT_CUDA_TUNING_DB.is_file() else None
    return None


def resolve_tuning_db_path(*, path: Path | None, backend: str) -> Path | None:
    p = Path(path) if path is not None else None
    if p is not None:
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p if p.is_file() else None
    return _default_db_path_for_backend(backend)


def load_tuning_db_jsonl(*, path: Path, backend: str | None = None) -> dict[tuple[str, str], TuningDBEntry]:
    """
    Load tuning DB entries keyed by (kernel, arch).

    Schema (per line JSON):
      {"backend":"cuda","kernel":"rms_norm2d","arch":"sm89","bindings":{"tile_n":768},"kernel_kind":"..."}

    Notes:
    - `kernel_kind` and `variant` are accepted synonyms (variant wins if both set).
    - Extra fields are ignored for forward compatibility.
    """

    p = Path(path)
    if not p.is_file():
        return {}
    backend_filter = str(backend or "").strip().lower()
    out: dict[tuple[str, str], TuningDBEntry] = {}
    for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = str(raw).strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"invalid tuning_db jsonl at {p}:{i}: {type(e).__name__}: {e}") from e
        if not isinstance(row, dict):
            continue
        row_backend = str(row.get("backend") or "").strip().lower()
        if backend_filter and row_backend and row_backend != backend_filter:
            continue
        kernel = str(row.get("kernel") or "").strip()
        arch = str(row.get("arch") or "").strip()
        bindings = row.get("bindings")
        if not kernel or not arch or not isinstance(bindings, dict):
            continue
        kernel_kind = str(row.get("kernel_kind") or "").strip()
        variant = str(row.get("variant") or "").strip()
        if variant:
            kernel_kind = variant
        out[(kernel, arch)] = TuningDBEntry(bindings=dict(bindings), kernel_kind=str(kernel_kind))
    return out


def load_tuning_db(*, path: Path | None = None, backend: str = "cuda") -> tuple[dict[tuple[str, str], TuningDBEntry], str]:
    """
    Cached loader. Returns (mapping, repo_rel_path_or_empty).
    """

    b = str(backend or "").strip().lower() or "cuda"
    resolved = resolve_tuning_db_path(path=path, backend=b)
    if resolved is None:
        return {}, ""
    key = (str(resolved.resolve()), b)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    mapping = load_tuning_db_jsonl(path=resolved, backend=b)
    payload = (mapping, _to_repo_rel(resolved))
    _CACHE[key] = payload
    return payload


__all__ = [
    "TuningDBEntry",
    "load_tuning_db_jsonl",
    "load_tuning_db",
    "resolve_tuning_db_path",
]
