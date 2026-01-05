"""
Diff debugging utilities (P0 gap fix).

When a final output mismatch happens, users want a quick answer to:
  - which op/tensor first diverged (when reference also exposes intermediates)
  - what are the shapes/dtypes/stats of intermediates around that op

This module re-runs the interpreter with tracing and compares any tensors that
exist in both the interpreter env and the reference output dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from intent_ir.ir import IntentFunction
from intent_ir.macros import expand_macros
from verify.gen_cases import TestCase
from verify.interpreter import execute_intent_with_trace, InterpreterTrace
from verify.diff_runner import DiffResult, _with_io_aliases  # type: ignore
from verify.tolerances import infer_tolerances


def _compare_arrays(name: str, pred: np.ndarray, ref: np.ndarray, tol: Dict[str, float]) -> DiffResult:
    atol = float(tol.get("atol", 1e-3))
    rtol = float(tol.get("rtol", 1e-3))
    p = np.asarray(pred)
    r = np.asarray(ref)
    if p.shape != r.shape:
        return DiffResult(False, 0.0, 0.0, None, f"shape mismatch for {name}: {p.shape} vs {r.shape}")
    if r.dtype == bool or p.dtype == bool:
        bad = np.not_equal(p.astype(bool), r.astype(bool))
        if not bool(bad.any()):
            return DiffResult(True, 0.0, 0.0, None, "ok")
        idx = tuple(int(x) for x in np.argwhere(bad)[0])
        return DiffResult(False, 1.0, 1.0, idx, f"bool mismatch at {idx}")
    p32 = p.astype(np.float64, copy=False)
    r32 = r.astype(np.float64, copy=False)
    diff = np.abs(p32 - r32)
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    denom = np.abs(r32) + 1e-8
    rel = diff / denom
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    ok = (max_abs <= atol) or (max_rel <= rtol)
    if ok:
        return DiffResult(True, max_abs, max_rel, None, "ok")
    bad_mask = diff > atol
    if not bool(bad_mask.any()):
        bad_mask = rel > rtol
    idx = tuple(int(x) for x in np.argwhere(bad_mask)[0]) if bool(bad_mask.any()) else None
    if idx is None:
        return DiffResult(False, max_abs, max_rel, None, f"mismatch max_abs={max_abs} max_rel={max_rel}")
    return DiffResult(False, max_abs, max_rel, idx, f"mismatch at {idx}: ref={float(r32[idx])} pred={float(p32[idx])}")


@dataclass(frozen=True)
class Divergence:
    op_index: int
    tensor: str
    diff: DiffResult


def debug_mismatch(
    intent: IntentFunction,
    run_ref_fn: Callable[[TestCase], Dict[str, np.ndarray]],
    case: TestCase,
    *,
    tolerances: Optional[Dict[str, float]] = None,
    sample_elems: int = 16,
) -> Dict[str, Any]:
    """
    Return a JSON-serializable debug report.
    """
    tol = dict(tolerances) if tolerances is not None else None

    try:
        intent_exec = expand_macros(intent)
    except Exception as e:
        return {
            "ok": False,
            "error": f"macro expansion error: {type(e).__name__}: {e}",
            "case": {"shapes": dict(case.shapes), "seed": int(case.seed)},
        }

    ref_out = run_ref_fn(case)
    ref_out = _with_io_aliases(intent_exec, ref_out)
    if tol is None:
        tol = infer_tolerances(intent_exec, ref_out=ref_out).to_dict()

    bindings = dict(case.shapes)
    pred_out, trace, env = execute_intent_with_trace(intent_exec, ref_out, shape_bindings=bindings, sample_elems=sample_elems)

    # Compare final outputs.
    final: Dict[str, Any] = {"ok": True, "outputs": {}}
    ok_all = True
    for name in intent_exec.outputs:
        if name not in ref_out or name not in pred_out:
            final["outputs"][name] = {"ok": False, "summary": "missing output in ref/pred"}
            ok_all = False
            continue
        d = _compare_arrays(name, pred_out[name], ref_out[name], tol)
        final["outputs"][name] = {
            "ok": bool(d.ok),
            "summary": str(d.summary),
            "max_abs": float(d.max_abs_err),
            "max_rel": float(d.max_rel_err),
            "first_bad_index": list(d.first_bad_index) if d.first_bad_index is not None else None,
        }
        ok_all = ok_all and bool(d.ok)
    final["ok"] = bool(ok_all)

    # Find first op that produces a tensor also present in ref_out, and mismatches.
    first: Optional[Divergence] = None
    for ot in trace.op_traces:
        tname = ot.output.name
        if tname not in ref_out:
            continue
        try:
            d = _compare_arrays(tname, env[tname], ref_out[tname], tol)
        except Exception as e:
            d = DiffResult(False, 0.0, 0.0, None, f"compare error: {type(e).__name__}: {e}")
        if not d.ok:
            first = Divergence(op_index=int(ot.op_index), tensor=tname, diff=d)
            break

    report: Dict[str, Any] = {
        "ok": bool(final["ok"]),
        "case": {"shapes": dict(case.shapes), "seed": int(case.seed)},
        "final": final,
        "divergence": None,
        "trace": {
            "ops": [
                {
                    "op_index": int(ot.op_index),
                    "op": dict(ot.op),
                    "inputs": {k: vars(v) for k, v in ot.inputs.items()},
                    "output": vars(ot.output),
                }
                for ot in trace.op_traces[: min(64, len(trace.op_traces))]
            ],
            "truncated": bool(len(trace.op_traces) > 64),
        },
    }
    if first is not None:
        report["divergence"] = {
            "op_index": int(first.op_index),
            "tensor": str(first.tensor),
            "diff": {
                "summary": str(first.diff.summary),
                "max_abs": float(first.diff.max_abs_err),
                "max_rel": float(first.diff.max_rel_err),
                "first_bad_index": list(first.diff.first_bad_index) if first.diff.first_bad_index is not None else None,
            },
        }
    return report


__all__ = ["debug_mismatch"]
