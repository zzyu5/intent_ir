from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

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


def validate_mlir_linalg_text(text: str) -> List[str]:
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

    return errs


@dataclass(frozen=True)
class TileDslValidation:
    ok: bool
    errors: List[str]


def validate_tile_dsl_json(obj: Any) -> List[str]:
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

    return errs


__all__ = ["validate_mlir_linalg_text", "validate_tile_dsl_json"]

