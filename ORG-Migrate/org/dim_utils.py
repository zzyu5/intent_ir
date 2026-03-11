from __future__ import annotations

from typing import Any

from .schema import OrgDim, OrgDoc


def _coerce_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def normalize_dim_name(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    s = s.replace("-", "_").replace(" ", "_")
    out = []
    for ch in s:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out)
    return "_".join(part for part in s2.split("_") if part)


def collect_dim_candidates(org: OrgDoc) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for dim in list(getattr(org, "dims", []) or []):
        name = str(getattr(dim, "name", "") or "").strip()
        if not name:
            continue
        out[name] = list(getattr(dim, "candidates", []) or [])
    return out


def collect_dim_candidates_normalized(org: OrgDoc) -> dict[str, list[Any]]:
    raw = collect_dim_candidates(org)
    out: dict[str, list[Any]] = {}
    for name, candidates in raw.items():
        key = normalize_dim_name(name)
        if not key:
            continue
        out[key] = list(candidates or [])
    return out


def collect_dim_candidate_ints_normalized(org: OrgDoc) -> dict[str, list[int]]:
    raw = collect_dim_candidates_normalized(org)
    out: dict[str, list[int]] = {}
    for key, candidates in raw.items():
        ints = [v for x in candidates if (v := _coerce_int(x)) is not None]
        if ints:
            out[key] = ints
    return out


def union_dim_candidates(dim_candidates_norm: dict[str, list[Any]], *names: str) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for name in list(names or []):
        key = normalize_dim_name(str(name))
        if not key:
            continue
        for value in list((dim_candidates_norm or {}).get(key) or []):
            marker = repr(value)
            if marker in seen:
                continue
            seen.add(marker)
            out.append(value)
    return out


def union_dim_candidate_ints(dim_candidates_norm: dict[str, list[int]], *names: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for name in list(names or []):
        key = normalize_dim_name(str(name))
        if not key:
            continue
        for value in list((dim_candidates_norm or {}).get(key) or []):
            iv = int(value)
            if iv in seen:
                continue
            seen.add(iv)
            out.append(iv)
    return out


def find_dim(org: OrgDoc, *names: str) -> OrgDim | None:
    wanted = {normalize_dim_name(x) for x in names if normalize_dim_name(x)}
    for dim in list(getattr(org, "dims", []) or []):
        if normalize_dim_name(getattr(dim, "name", "")) in wanted:
            return dim
    return None


__all__ = [
    "collect_dim_candidate_ints_normalized",
    "collect_dim_candidates",
    "collect_dim_candidates_normalized",
    "find_dim",
    "normalize_dim_name",
    "union_dim_candidate_ints",
    "union_dim_candidates",
]
