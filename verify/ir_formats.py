from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from intent_ir.llm import strip_code_fence


def _balanced_delims(text: str) -> List[str]:
    """
    Lightweight delimiter balance check for MLIR-like textual IR.

    We intentionally only check (), [], {} because MLIR types use <> heavily and
    "->" confounds a naive angle-bracket matcher.
    """
    opens = {"(": ")", "[": "]", "{": "}"}
    closes = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in str(text):
        if ch in opens:
            stack.append(ch)
            continue
        if ch in closes:
            if not stack or stack[-1] != closes[ch]:
                return [f"unbalanced delimiter: unexpected '{ch}'"]
            stack.pop()
    if stack:
        return [f"unbalanced delimiter: missing '{opens[stack[-1]]}'"]
    return []


def _io_spec_tensor_ranks(io_spec: Any) -> Optional[List[int]]:
    if not isinstance(io_spec, dict):
        return None
    tensors = io_spec.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        return None
    ranks: List[int] = []
    for t in tensors.values():
        if not isinstance(t, dict):
            continue
        shape = t.get("shape")
        if not isinstance(shape, list):
            continue
        ranks.append(int(len(shape)))
    return ranks if ranks else None


def _io_spec_symbols(io_spec: Any) -> List[str]:
    if not isinstance(io_spec, dict):
        return []
    tensors = io_spec.get("tensors")
    if not isinstance(tensors, dict):
        return []
    out: List[str] = []
    for t in tensors.values():
        if not isinstance(t, dict):
            continue
        shape = t.get("shape")
        if not isinstance(shape, list):
            continue
        for d in shape:
            if isinstance(d, str) and d.strip() and d not in out:
                out.append(d)
    return out


def _extract_tensor_type_ranks_from_mlir(text: str) -> List[int]:
    """
    Extract ranks of tensor/memref types from the *function signature*.

    We use a heuristic regex and compute rank as (#segments split by 'x') - 1.
    """
    ranks: List[int] = []

    t = str(text)
    sig_start = t.find("func.func")
    if sig_start < 0:
        sig_start = t.find("func @")
    if sig_start < 0:
        return ranks
    sig_end = t.find("{", sig_start)
    if sig_end < 0:
        # Fallback: just take the first line.
        nl = t.find("\n", sig_start)
        sig_end = nl if nl >= 0 else len(t)
    sig = t[sig_start:sig_end]

    for m in re.finditer(r"(tensor|memref)<([^>]+)>", sig):
        body = m.group(2)
        parts = [p for p in body.split("x") if p]
        if len(parts) >= 2:
            ranks.append(int(len(parts) - 1))
    return ranks


def validate_mlir_linalg_text(text: str, *, io_spec: Any | None = None) -> List[str]:
    """
    Validate that the output looks like *syntactically plausible* MLIR (Linalg).

    This is intentionally NOT a full MLIR parser; E6 only needs a stable,
    deterministic "legal-looking IR" gate for comparing LLM output formats.
    """
    t = strip_code_fence(str(text or "")).strip()
    if not t:
        return ["empty output"]

    errs: List[str] = []
    errs += _balanced_delims(t)

    # Basic structure anchors.
    if "module" not in t:
        errs.append("missing 'module'")
    if "func.func" not in t and "func @" not in t:
        errs.append("missing 'func.func' or 'func @'")
    if "linalg." not in t:
        errs.append("missing any 'linalg.' op")

    # Common non-IR artifacts / placeholders.
    bad_markers = ["<TODO", "TODO", "…", "...", "PLACEHOLDER", "fill_me", "TBD"]
    if any(x in t for x in bad_markers):
        errs.append("contains placeholder markers")
    if "```" in t:
        errs.append("contains markdown code fences")

    # Optional structure completeness check against io_spec:
    # - total tensor/memref type count and rank multiset must match.
    exp_ranks = _io_spec_tensor_ranks(io_spec)
    if exp_ranks:
        got_ranks = _extract_tensor_type_ranks_from_mlir(t)
        # Accept duplicates (e.g., output both as argument and return); require
        # that the signature covers at least all expected tensor ranks.
        if len(got_ranks) < len(exp_ranks):
            errs.append(f"tensor type count mismatch: got={len(got_ranks)} expected_at_least={len(exp_ranks)}")
        else:
            exp_counts: Dict[int, int] = {}
            got_counts: Dict[int, int] = {}
            for r in exp_ranks:
                exp_counts[int(r)] = int(exp_counts.get(int(r), 0)) + 1
            for r in got_ranks:
                got_counts[int(r)] = int(got_counts.get(int(r), 0)) + 1
            missing: List[Tuple[int, int, int]] = []
            for rk, need in exp_counts.items():
                have = int(got_counts.get(int(rk), 0))
                if have < int(need):
                    missing.append((int(rk), int(have), int(need)))
            if missing:
                errs.append(f"tensor rank coverage mismatch: missing={missing} got={sorted(got_ranks)} expected={sorted(exp_ranks)}")

    return errs


@dataclass(frozen=True)
class TileDslValidation:
    ok: bool
    errors: List[str]


def validate_tile_dsl_json(obj: Any, *, io_spec: Any | None = None) -> List[str]:
    """
    Validate a minimal tile-centric schedule JSON.

    Expected shape (loose):
      {
        "schema_version": "tile_dsl_v0",
        "kernel": "...",
        "schedule": {
          "tile": {"M": 128, "N": 128, ...},
          "vec_width": 8,
          "num_threads": 1
        }
      }
    """
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    d: Dict[str, Any] = obj
    errs: List[str] = []

    sv = d.get("schema_version")
    if not isinstance(sv, str) or not sv.strip():
        errs.append("missing schema_version")
    kernel = d.get("kernel")
    if not isinstance(kernel, str) or not kernel.strip():
        errs.append("missing kernel name")
    sched = d.get("schedule")
    if not isinstance(sched, dict):
        errs.append("missing schedule dict")
        return errs

    tile = sched.get("tile")
    if not isinstance(tile, dict) or not tile:
        errs.append("missing schedule.tile (non-empty dict)")
    else:
        for k, v in tile.items():
            if not isinstance(k, str) or not k.strip():
                errs.append("schedule.tile has non-string axis key")
                break
            if not isinstance(v, int) or int(v) <= 0:
                errs.append(f"schedule.tile.{k} must be a positive int")
                break

    vw = sched.get("vec_width")
    if vw is not None and (not isinstance(vw, int) or int(vw) <= 0):
        errs.append("schedule.vec_width must be a positive int if present")
    nt = sched.get("num_threads")
    if nt is not None and (not isinstance(nt, int) or int(nt) <= 0):
        errs.append("schedule.num_threads must be a positive int if present")

    # Optional structure completeness check: schedule axes should be drawn from
    # known symbolic dimensions in io_spec (when available).
    if not errs and io_spec is not None:
        syms = set(_io_spec_symbols(io_spec))
        if syms:
            bad = [str(k) for k in (tile or {}).keys() if str(k) not in syms]
            if bad:
                errs.append(f"schedule.tile has unknown axes: {bad} (known={sorted(syms)})")

    return errs


__all__ = ["validate_mlir_linalg_text", "validate_tile_dsl_json"]
