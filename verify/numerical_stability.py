"""
Numerical stability checks (P0 gap fix).

Stage B diff focuses on random inputs; this module adds a tiny set of
"edge-value" probes to catch NaN/Inf/overflow corner cases early.

Design constraints:
- Must reuse the real reference runner (Triton/TileLang) as-is.
- Must be deterministic and cheap (a handful of extra launches).
- Must not require LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from intent_ir.ir import IntentFunction
from verify.diff_runner import run_diff, _normalize_io_name, _with_io_aliases  # type: ignore
from verify.gen_cases import TestCase
from verify.tolerances import infer_tolerances


_Runner = Callable[[TestCase], Dict[str, np.ndarray]]


@dataclass(frozen=True)
class NumericalTestResult:
    name: str
    ok: bool
    summary: str


@dataclass(frozen=True)
class NumericalStabilityReport:
    ok: bool
    input: Optional[str]
    tolerances: Dict[str, float]
    results: List[NumericalTestResult]
    skipped: bool = False
    reason: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "skipped": bool(self.skipped),
            "reason": self.reason,
            "input": self.input,
            "tolerances": dict(self.tolerances),
            "results": [{"name": r.name, "ok": bool(r.ok), "summary": str(r.summary)} for r in self.results],
        }


def _external_inputs(intent: IntentFunction) -> List[str]:
    produced = {op.output for op in intent.ops if op.output}
    used: set[str] = set()
    for op in intent.ops:
        used.update(op.inputs)
    external = [n for n in used if (n in intent.tensors and n not in produced)]
    return sorted(set(external))


def _pick_primary_input(intent: IntentFunction, aliased_io: Dict[str, np.ndarray]) -> str | None:
    """
    Pick one "primary" float input tensor for stress testing.
    """
    for name in _external_inputs(intent):
        arr = aliased_io.get(name)
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size == 0:
            continue
        if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
            continue
        # skip scalar meta-values like eps (they are still useful, but less informative)
        if a.ndim == 0:
            continue
        return str(name)
    return None


def _mutate_input_dict(base_inputs: Dict[str, np.ndarray], *, target_name: str, mutated: np.ndarray) -> Dict[str, np.ndarray]:
    out = dict(base_inputs)
    norm = _normalize_io_name(target_name)
    # Update all keys that correspond to this input (runner name + intent alias).
    for k in list(out.keys()):
        if _normalize_io_name(k) == norm:
            out[k] = mutated
    out[target_name] = mutated
    return out


def run_numerical_stability_suite(
    kernel_name: str,
    intent: IntentFunction,
    run_ref_fn: _Runner,
    *,
    base_case: TestCase,
    tolerances: Optional[Dict[str, float]] = None,
) -> NumericalStabilityReport:
    tol = dict(tolerances) if tolerances is not None else infer_tolerances(intent).to_dict()

    try:
        sample = run_ref_fn(base_case)
    except Exception as e:
        return NumericalStabilityReport(
            ok=False,
            skipped=True,
            reason=f"reference runner error: {type(e).__name__}: {e}",
            input=None,
            tolerances=tol,
            results=[],
        )

    aliased = _with_io_aliases(intent, sample)
    primary = _pick_primary_input(intent, aliased)
    if primary is None:
        return NumericalStabilityReport(
            ok=True,
            skipped=True,
            reason="no suitable float external input found",
            input=None,
            tolerances=tol,
            results=[],
        )

    base_arr = np.asarray(aliased[primary])
    base_inputs = dict(sample)
    base_inputs.update({primary: base_arr})

    results: List[NumericalTestResult] = []

    def _check(name: str, mutated_arr: np.ndarray) -> None:
        case = TestCase(shapes=dict(base_case.shapes), dtypes=dict(base_case.dtypes or {}), seed=int(base_case.seed))
        case.inputs = _mutate_input_dict(base_inputs, target_name=primary, mutated=mutated_arr)
        diffs, _ = run_diff(intent, run_ref_fn, [case], tolerances=tol)
        d0 = diffs[0] if diffs else None
        ok = bool(d0.ok) if d0 is not None else False
        results.append(NumericalTestResult(name=name, ok=ok, summary=(d0.summary if d0 else "no diff result")))

    # 1) Inject NaN.
    if np.issubdtype(base_arr.dtype, np.floating) and base_arr.size:
        m = np.array(base_arr, copy=True)
        m.reshape(-1)[0] = np.nan
        _check("nan_injection", m)

    # 2) Inject +/- Inf.
    if np.issubdtype(base_arr.dtype, np.floating) and base_arr.size:
        m = np.array(base_arr, copy=True)
        flat = m.reshape(-1)
        flat[0] = np.inf
        if flat.size > 1:
            flat[1] = -np.inf
        _check("inf_injection", m)

    # 3) Overflow-ish scale (keep finite).
    if np.issubdtype(base_arr.dtype, np.floating) and base_arr.size:
        m = np.array(base_arr, copy=True).astype(np.float32, copy=False)
        m = m * np.float32(1e4)
        m = m.astype(base_arr.dtype, copy=False)
        _check("scale_1e4", m)

    ok_all = bool(results and all(r.ok for r in results))
    return NumericalStabilityReport(ok=ok_all, input=primary, tolerances=tol, results=results)


__all__ = ["NumericalStabilityReport", "run_numerical_stability_suite"]

