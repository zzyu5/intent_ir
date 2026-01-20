"""
IntentIR deterministic repair passes.

These passes are *not* intended to change semantics; they only repair common
LLM/schema slips so downstream stages (static_validate / diff / backend) can
operate on a well-formed IntentFunction.
"""

from __future__ import annotations

from typing import Dict, List, Set

from .ir_types import IntentFunction, Op, TensorType


def _strip_trailing_ones(shape: List[object]) -> List[object]:
    out = list(shape)
    while out:
        d = out[-1]
        if getattr(d, "kind", None) != "const":
            break
        try:
            if int(getattr(d, "value", -1)) != 1:
                break
        except Exception:
            break
        out.pop()
    return out


def _tensor_compatible(a: TensorType, b: TensorType) -> bool:
    # Treat i1/bool as compatible.
    da = "bool" if str(a.dtype) == "i1" else str(a.dtype)
    db = "bool" if str(b.dtype) == "i1" else str(b.dtype)
    if da != db:
        return False
    la = a.layout.name if hasattr(a.layout, "name") else str(a.layout)
    lb = b.layout.name if hasattr(b.layout, "name") else str(b.layout)
    if str(la) != str(lb):
        return False

    sa = _strip_trailing_ones(list(a.shape))
    sb = _strip_trailing_ones(list(b.shape))
    if len(sa) != len(sb):
        return False
    for da_, db_ in zip(sa, sb):
        ka = getattr(da_, "kind", None)
        kb = getattr(db_, "kind", None)
        va = getattr(da_, "value", None)
        vb = getattr(db_, "value", None)
        if ka == "const" and kb == "const" and str(va) != str(vb):
            return False
        # sym matches anything (specialization tolerant)
    return True


def _candidate_output_aliases(name: str) -> List[str]:
    """
    Heuristic aliases for missing outputs.

    Examples:
      out_ptr -> out
      out_mean_ptr -> mean
      out_rstd_ptr -> rstd
    """
    raw = str(name)
    s = raw.strip()
    cands: List[str] = []

    def add(x: str) -> None:
        x = str(x)
        if x and x not in cands:
            cands.append(x)

    add(s)
    lower = s.lower()

    # Strip common pointer suffixes.
    for suf in ("_ptr", "ptr"):
        if lower.endswith(suf):
            add(s[: -len(suf)])

    # Strip leading out_/output_ conventions.
    if lower.startswith("out_"):
        add(s[4:])
        base = s[4:]
        base_l = base.lower()
        if base_l.endswith("_ptr"):
            add(base[:-4])
        if base_l.endswith("ptr"):
            add(base[:-3])

    # Norm-specific shorthands.
    if "mean" in lower:
        add("mean")
        add("mean_val")
        add("mean_computed")
    if "rstd" in lower:
        add("rstd")
        add("rstd_val")
        add("rstd_computed")
    if lower in {"out", "output"} or "out" in lower:
        add("out")
        add("output")
        add("y")

    return cands


def repair_missing_outputs(intent: IntentFunction) -> List[str]:
    """
    Repair common LLM slips where declared outputs are not produced by any op.

    Strategy:
      - If an output tensor `out_ptr` is missing, and a produced value `out`
        exists with a compatible tensor type, insert `identity(out) -> out_ptr`.
      - Only perform name-based repairs; do not guess among multiple candidates.
    """
    if not intent.outputs:
        return []
    produced: Set[str] = {op.output for op in intent.ops if op.output}
    actions: List[str] = []

    # Precompute candidate -> type for fast checks.
    types: Dict[str, TensorType] = {k: v for k, v in intent.tensors.items()}

    for out in list(intent.outputs):
        if out in produced:
            continue
        out_tt = types.get(out)
        if out_tt is None:
            continue
        chosen = None
        for cand in _candidate_output_aliases(out):
            if cand not in produced:
                continue
            cand_tt = types.get(cand)
            if cand_tt is None:
                # Many LLM outputs only declare interface tensors; allow a small
                # set of extremely common aliases without relying on types.
                if cand in {"out", "mean", "rstd", "output", "y"}:
                    chosen = cand
                    break
                continue
            if _tensor_compatible(out_tt, cand_tt):
                chosen = cand
                break
        if chosen is None:
            continue
        intent.ops.append(Op(op="identity", inputs=[chosen], output=out, attrs={}))
        produced.add(out)
        actions.append(f"repair_output:{out}<-{chosen}")

    return actions


__all__ = ["repair_missing_outputs"]
