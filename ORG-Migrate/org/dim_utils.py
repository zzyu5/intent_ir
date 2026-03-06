from __future__ import annotations

from typing import Any

from .schema import OrgDoc


def _coerce_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def collect_dim_allowed_ints(org: OrgDoc) -> dict[str, set[int]]:
    """
    Collect `dims[].allowed` (when present) from an ORGDoc.

    Returns a mapping: dim_name -> set[int].
    Non-integer entries are ignored.
    """

    allowed: dict[str, set[int]] = {}
    for n in list(getattr(org, "nodes", []) or []):
        for d in list(getattr(n, "dims", []) or []):
            name = str(getattr(d, "name", "") or "").strip()
            if not name:
                continue
            raw_allowed = list(getattr(d, "allowed", []) or [])
            vals = {v for x in raw_allowed if (v := _coerce_int(x)) is not None}
            if not vals:
                continue
            allowed.setdefault(name, set()).update(vals)
    return allowed


def normalize_dim_name(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    s = s.replace("-", "_").replace(" ", "_")
    # Keep only [a-z0-9_], collapsing runs of separators.
    out = []
    for ch in s:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out)
    return "_".join(part for part in s2.split("_") if part)


def collect_dim_allowed_ints_normalized(org: OrgDoc) -> dict[str, set[int]]:
    """
    Like `collect_dim_allowed_ints`, but keyed by a normalized dim name.
    Useful when ORG LLM emits source-level names (e.g. BLOCK_KV) instead of
    backend binding names (e.g. ATTN_BLOCK_KV).
    """

    raw = collect_dim_allowed_ints(org)
    out: dict[str, set[int]] = {}
    for k, vals in dict(raw or {}).items():
        nk = normalize_dim_name(str(k))
        if not nk:
            continue
        out.setdefault(nk, set()).update({int(v) for v in set(vals or set())})
    return out


def union_dim_allowed(dim_allowed_norm: dict[str, set[int]], *names: str) -> set[int]:
    out: set[int] = set()
    for n in list(names or []):
        key = normalize_dim_name(str(n))
        if not key:
            continue
        out.update({int(v) for v in set((dim_allowed_norm or {}).get(key) or set())})
    return out


__all__ = [
    "collect_dim_allowed_ints",
    "collect_dim_allowed_ints_normalized",
    "normalize_dim_name",
    "union_dim_allowed",
]
