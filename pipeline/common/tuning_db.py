from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CUDA_TUNING_DB = ROOT / "workflow" / "flaggems" / "state" / "tuning_db" / "cuda.jsonl"
_CACHE: dict[tuple[str, str], tuple[dict[tuple[str, str], list["TuningDBEntry"]], str]] = {}


@dataclass(frozen=True)
class TuningDBEntry:
    bindings: dict[str, Any]
    kernel_kind: str = ""
    when: dict[str, Any] = field(default_factory=dict)
    compiler_stacks: tuple[str, ...] = field(default_factory=tuple)


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

def tuning_db_path_from_env(*, backend: str = "cuda") -> Path | None:
    b = str(backend or "").strip().lower() or "cuda"
    raw = ""
    if b == "cuda":
        raw = str(os.getenv("INTENTIR_CUDA_TUNING_DB", "") or "").strip()
        if not raw:
            raw = str(os.getenv("INTENTIR_TUNING_DB", "") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else p

def _match_when(when: dict[str, Any], shape_bindings: dict[str, int]) -> bool:
    if not when:
        return True
    for raw_key, cond in dict(when).items():
        key = str(raw_key).strip()
        if not key:
            return False
        if key not in shape_bindings:
            return False
        v = int(shape_bindings[key])
        if isinstance(cond, (int, float)):
            if v != int(cond):
                return False
            continue
        if isinstance(cond, (list, tuple, set)):
            allowed = {int(x) for x in list(cond)}
            if v not in allowed:
                return False
            continue
        if isinstance(cond, dict):
            c = dict(cond)
            if "eq" in c and v != int(c["eq"]):
                return False
            if "ne" in c and v == int(c["ne"]):
                return False
            if "lt" in c and v >= int(c["lt"]):
                return False
            if "le" in c and v > int(c["le"]):
                return False
            if "gt" in c and v <= int(c["gt"]):
                return False
            if "ge" in c and v < int(c["ge"]):
                return False
            if "in" in c:
                allowed = {int(x) for x in list(c["in"] or [])}
                if v not in allowed:
                    return False
            if "not_in" in c:
                blocked = {int(x) for x in list(c["not_in"] or [])}
                if v in blocked:
                    return False
            if "divisible_by" in c:
                d = int(c["divisible_by"])
                if d <= 0 or (v % d) != 0:
                    return False
            if "mod" in c:
                m = int(c["mod"])
                eq = int(c.get("mod_eq", c.get("eq", 0)))
                if m <= 0 or (v % m) != int(eq):
                    return False
            continue
        return False
    return True


def resolve_tuning_entries(
    entries: list[TuningDBEntry],
    *,
    shape_bindings: dict[str, int],
    compiler_stack: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Apply all matching entries in file order (last-match wins).

    Returns (merged_bindings, kernel_kind_override_or_empty).
    """

    stack = str(compiler_stack or "").strip().lower()
    if stack in {"cpp", "c++"}:
        stack = "cpp_plugin"

    merged: dict[str, Any] = {}
    kernel_kind = ""
    for e in list(entries or []):
        if not isinstance(e, TuningDBEntry):
            continue
        if e.compiler_stacks:
            allowed = {str(x).strip().lower() for x in e.compiler_stacks if str(x).strip()}
            if stack not in allowed:
                continue
        if not _match_when(dict(e.when or {}), shape_bindings=shape_bindings):
            continue
        merged.update(dict(e.bindings or {}))
        kk = str(e.kernel_kind or "").strip()
        if kk:
            kernel_kind = kk
    return merged, kernel_kind


def load_tuning_db_jsonl(*, path: Path, backend: str | None = None) -> dict[tuple[str, str], list[TuningDBEntry]]:
    """
    Load tuning DB entries keyed by (kernel, arch). Entries are stored in file order.

    Schema (per line JSON):
      {"backend":"cuda","kernel":"rms_norm2d","arch":"sm89","bindings":{"tile_n":768},"kernel_kind":"..."}
      Optional:
        - "when": {"HEAD_DIM": 16}  (shape-bucket constraints)

    Notes:
    - `kernel_kind` and `variant` are accepted synonyms (variant wins if both set).
    - Extra fields are ignored for forward compatibility.
    """

    p = Path(path)
    if not p.is_file():
        return {}
    backend_filter = str(backend or "").strip().lower()
    out: dict[tuple[str, str], list[TuningDBEntry]] = {}
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
        when = row.get("when")
        if when is None:
            when = row.get("shape")  # legacy/alt spelling
        when_dict = dict(when) if isinstance(when, dict) else {}

        stacks_raw = row.get("compiler_stack")
        if stacks_raw is None:
            stacks_raw = row.get("stack")
        stacks: list[str] = []
        if isinstance(stacks_raw, str):
            s = str(stacks_raw).strip()
            if s:
                stacks = [s]
        elif isinstance(stacks_raw, (list, tuple, set)):
            for x in list(stacks_raw):
                s = str(x).strip()
                if s:
                    stacks.append(s)

        out.setdefault((kernel, arch), []).append(
            TuningDBEntry(
                bindings=dict(bindings),
                kernel_kind=str(kernel_kind),
                when=when_dict,
                compiler_stacks=tuple(stacks),
            )
        )
    return out


def load_tuning_db(
    *, path: Path | None = None, backend: str = "cuda"
) -> tuple[dict[tuple[str, str], list[TuningDBEntry]], str]:
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
    "resolve_tuning_entries",
    "tuning_db_path_from_env",
]
