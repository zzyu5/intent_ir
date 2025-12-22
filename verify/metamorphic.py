"""
Task5 Stage C (v1.2): metamorphic + bounded exhaustive verification helpers.

This module is intentionally independent from Triton/torch; it operates on:
- IntentFunction (our IR)
- a reference runner `run_ref_fn(TestCase) -> Dict[str, np.ndarray]`
  that returns both inputs (for interpreter) and outputs (for comparison).

Stage C strengthens falsification power beyond random differential testing:
- Metamorphic: check operator invariants (e.g., permutation/shift invariance).
- Bounded exhaustive: enumerate tiny shapes + small value domains to kill subtle bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from intent_ir.ir_types import IntentFunction
from verify.diff_runner import DiffResult, run_diff
from verify.gen_cases import TestCase


@dataclass
class MetamorphicResult:
    relation: str
    ok: bool
    detail: str
    base_diff: Optional[DiffResult] = None
    transformed_diff: Optional[DiffResult] = None


@dataclass
class MetamorphicSuiteReport:
    ok: bool
    results: List[MetamorphicResult]


@dataclass
class BoundedExhaustiveReport:
    ok: bool
    checked: int
    total: int
    detail: str
    first_failure_case: Optional[TestCase] = None
    first_failure_summary: Optional[str] = None


_Runner = Callable[[TestCase], Dict[str, np.ndarray]]


def _derive_bindings(case: TestCase) -> Dict[str, int]:
    bindings = dict(case.shapes)
    if "group" in bindings and "num_groups" not in bindings:
        bindings["num_groups"] = int(bindings["group"])
    if "num_groups" in bindings and "C" in bindings and "group_size" not in bindings:
        g = int(bindings["num_groups"])
        c = int(bindings["C"])
        bindings["group_size"] = c // g if (g > 0 and c % g == 0) else (c + g - 1) // max(g, 1)
    if "group_size" in bindings and "HW" in bindings and "num_elements" not in bindings:
        try:
            bindings["num_elements"] = int(bindings["group_size"]) * int(bindings["HW"])
        except Exception:
            pass
    return bindings


def _as_inputs_for_intent(intent: IntentFunction, ref_io: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # Keep the same convention as `verify.diff_runner.run_diff`:
    # everything that is not a declared output is considered an input.
    return {k: v for k, v in ref_io.items() if k not in intent.outputs}


def _allclose(a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float) -> bool:
    if a.dtype == bool or b.dtype == bool:
        return np.array_equal(a.astype(bool), b.astype(bool))
    return np.allclose(a, b, atol=atol, rtol=rtol)


def run_metamorphic_suite(
    kernel_name: str,
    intent: IntentFunction,
    run_ref_fn: _Runner,
    *,
    base_case: TestCase,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    rng_seed: int = 0,
) -> MetamorphicSuiteReport:
    """
    Run metamorphic relations for a kernel.

    This checks BOTH:
    - translation correctness: (ref == interpreter) for base + transformed inputs
    - metamorphic invariant: (out_base relation out_transformed) for both ref and interpreter
    """
    rng = np.random.default_rng(rng_seed)
    results: List[MetamorphicResult] = []

    # 1) Run once to sample a concrete input, then "freeze" it into case.inputs.
    sampled = run_ref_fn(base_case)
    base_inputs = _as_inputs_for_intent(intent, sampled)
    base_case_fixed = TestCase(shapes=dict(base_case.shapes), dtypes=base_case.dtypes or {}, seed=base_case.seed, inputs=base_inputs)

    relations = _default_relations(kernel_name)
    if not relations:
        return MetamorphicSuiteReport(ok=True, results=[MetamorphicResult(relation="(none)", ok=True, detail="no relations registered")])

    for rel_id, transform_inputs, check_outputs in relations:
        try:
            bindings = _derive_bindings(base_case_fixed)
            transformed_inputs = transform_inputs(base_inputs, bindings, rng)
            transformed_case = TestCase(
                shapes=dict(base_case_fixed.shapes),
                dtypes=base_case_fixed.dtypes or {},
                seed=base_case_fixed.seed,
                inputs=transformed_inputs,
            )

            base_diffs, _ = run_diff(intent, run_ref_fn, [base_case_fixed], tolerances={"atol": atol, "rtol": rtol})
            tr_diffs, _ = run_diff(intent, run_ref_fn, [transformed_case], tolerances={"atol": atol, "rtol": rtol})
            base_diff = base_diffs[0] if base_diffs else None
            tr_diff = tr_diffs[0] if tr_diffs else None
            if base_diff is None or tr_diff is None:
                results.append(
                    MetamorphicResult(
                        relation=rel_id,
                        ok=False,
                        detail="internal error: missing diff results",
                        base_diff=base_diff,
                        transformed_diff=tr_diff,
                    )
                )
                continue

            # Metamorphic invariant check against reference outputs, and interpreter outputs.
            ref_base = run_ref_fn(base_case_fixed)
            ref_tr = run_ref_fn(transformed_case)

            from verify.interpreter import execute_intent

            with np.errstate(all="ignore"):
                pred_base = execute_intent(
                    intent, _as_inputs_for_intent(intent, ref_base), shape_bindings=_derive_bindings(base_case_fixed)
                )
                pred_tr = execute_intent(
                    intent, _as_inputs_for_intent(intent, ref_tr), shape_bindings=_derive_bindings(transformed_case)
                )

            ok_ref, msg_ref = check_outputs(ref_base, ref_tr, bindings, atol=atol, rtol=rtol)
            ok_pred, msg_pred = check_outputs(pred_base, pred_tr, bindings, atol=atol, rtol=rtol)

            ok = bool(base_diff.ok and tr_diff.ok and ok_ref and ok_pred)
            detail = "; ".join(
                [
                    f"base_diff={base_diff.summary}",
                    f"trans_diff={tr_diff.summary}",
                    f"ref={msg_ref}",
                    f"pred={msg_pred}",
                ]
            )
            results.append(MetamorphicResult(relation=rel_id, ok=ok, detail=detail, base_diff=base_diff, transformed_diff=tr_diff))
        except Exception as e:
            results.append(MetamorphicResult(relation=rel_id, ok=False, detail=f"error: {type(e).__name__}: {e}"))

    return MetamorphicSuiteReport(ok=all(r.ok for r in results), results=results)


def run_bounded_exhaustive(
    kernel_name: str,
    intent: IntentFunction,
    run_ref_fn: _Runner,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    max_cases: int | None = None,
    rng_seed: int = 0,
) -> BoundedExhaustiveReport:
    """
    Bounded exhaustive differential testing on tiny shapes + small input domains.
    """
    rng = np.random.default_rng(rng_seed)
    spec = _bounded_specs(kernel_name)
    if spec is None:
        return BoundedExhaustiveReport(ok=True, checked=0, total=0, detail="skipped: no bounded spec for this kernel")

    shape_bindings, input_names, domain = spec
    total_elems = sum(int(np.prod(shape_bindings[name])) for name in input_names)
    total = len(domain) ** total_elems
    if max_cases is not None:
        total = min(total, max_cases)

    checked = 0
    # Enumerate in a deterministic order. For very large totals, fall back to sampling.
    exhaustive = max_cases is None
    if not exhaustive:
        # Sample `max_cases` random assignments.
        def iter_assignments():
            for _ in range(max_cases):
                yield [rng.choice(domain) for _ in range(total_elems)]

    else:
        def iter_assignments():
            return product(domain, repeat=total_elems)

    # Build all cases one-by-one to allow early stop on first failure.
    for flat_vals in iter_assignments():
        flat_vals = list(flat_vals)
        inputs: Dict[str, np.ndarray] = {}
        cursor = 0
        for name in input_names:
            shape = shape_bindings[name]
            n = int(np.prod(shape))
            arr = np.asarray(flat_vals[cursor : cursor + n], dtype=np.float32).reshape(shape)
            inputs[name] = arr
            cursor += n

        # Fill fixed inputs if needed.
        if kernel_name == "group_norm_kernel":
            C = int(shape_bindings["W"][0])
            inputs.setdefault("W", np.ones((C,), dtype=np.float32))
            inputs.setdefault("B", np.zeros((C,), dtype=np.float32))

        case_shapes = _case_shapes_from_input_shapes(kernel_name, shape_bindings)
        case = TestCase(shapes=case_shapes, dtypes={}, seed=0, inputs=inputs)
        diffs, _ = run_diff(intent, run_ref_fn, [case], tolerances={"atol": atol, "rtol": rtol})
        checked += 1
        if not diffs or not diffs[0].ok:
            summary = diffs[0].summary if diffs else "no diff result"
            return BoundedExhaustiveReport(
                ok=False,
                checked=checked,
                total=total,
                detail="first failure",
                first_failure_case=case,
                first_failure_summary=summary,
            )
        if max_cases is not None and checked >= max_cases:
            break

    return BoundedExhaustiveReport(ok=True, checked=checked, total=total, detail="ok")


def _default_relations(
    kernel_name: str,
) -> List[
    Tuple[
        str,
        Callable[[Dict[str, np.ndarray], Dict[str, int], np.random.Generator], Dict[str, np.ndarray]],
        Callable[[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]], Tuple[bool, str]],
    ]
]:
    relations = []

    if kernel_name == "any_kernel_dim":
        def _transform(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            inp = np.asarray(inputs["inp"])
            if inp.ndim != 2:
                raise ValueError(f"expected inp rank-2, got {inp.shape}")
            perm = rng.permutation(inp.shape[1])
            out = dict(inputs)
            out["inp"] = inp[:, perm]
            return out

        def _check(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            ok = np.array_equal(np.asarray(base_out["out"]).astype(bool), np.asarray(tr_out["out"]).astype(bool))
            return ok, "out invariant under column perm" if ok else "out changed under column perm"

        relations.append(("MR_any_perm_columns", _transform, _check))

    if kernel_name == "group_norm_kernel":
        def _transform(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            x = np.asarray(inputs["X"], dtype=np.float32)
            c = np.float32(0.37)
            out = dict(inputs)
            out["X"] = x + c
            return out

        def _check(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            # Shift invariance:
            # Y unchanged; Mean shifts by +c; Rstd unchanged.
            c = np.float32(0.37)
            y0, y1 = np.asarray(base_out["Y"]), np.asarray(tr_out["Y"])
            m0, m1 = np.asarray(base_out["Mean"]), np.asarray(tr_out["Mean"])
            r0, r1 = np.asarray(base_out["Rstd"]), np.asarray(tr_out["Rstd"])
            ok_y = _allclose(y0, y1, atol=atol, rtol=rtol)
            ok_m = _allclose(m0 + c, m1, atol=atol, rtol=rtol)
            ok_r = _allclose(r0, r1, atol=atol, rtol=rtol)
            ok = bool(ok_y and ok_m and ok_r)
            if ok:
                return True, "shift invariance holds"
            return False, f"shift invariance violated (Y={ok_y}, Mean={ok_m}, Rstd={ok_r})"

        relations.append(("MR_groupnorm_shift", _transform, _check))

    if kernel_name == "_attn_fwd":
        def _transform(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            k = np.asarray(inputs["K"], dtype=np.float32)
            v = np.asarray(inputs["V"], dtype=np.float32)
            mask = np.asarray(inputs.get("attn_mask", 0.0), dtype=np.float32)
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError(f"expected K/V rank-4, got K={k.shape} V={v.shape}")
            kv = k.shape[2]
            perm = rng.permutation(kv)
            out = dict(inputs)
            out["K"] = k[:, :, perm, :]
            out["V"] = v[:, :, perm, :]
            if mask.ndim == 4:
                out["attn_mask"] = mask[:, :, :, perm]
            return out

        def _check(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            o0, o1 = np.asarray(base_out["Out"]), np.asarray(tr_out["Out"])
            ok = _allclose(o0, o1, atol=atol, rtol=rtol)
            return ok, "Out invariant under KV perm" if ok else "Out changed under KV perm"

        relations.append(("MR_attn_perm_kv", _transform, _check))

    return relations


def _bounded_specs(
    kernel_name: str,
) -> Optional[Tuple[Dict[str, Tuple[int, ...]], List[str], Sequence[float]]]:
    """
    Returns:
      - input_shapes: {input_name: shape_tuple}
      - input_names: which inputs are enumerated from the domain
      - domain: scalar domain values
    """
    if kernel_name == "any_kernel_dim":
        # 2x3 boolean-ish domain => 64 total combinations.
        return ({"inp": (2, 3)}, ["inp"], [0.0, 1.0])
    if kernel_name == "group_norm_kernel":
        # Tiny groupnorm: N=1, C=2, HW=2, num_groups=2 => X has 4 elems, domain 3^4=81.
        return ({"X": (1, 2, 2), "W": (2,), "B": (2,)}, ["X"], [-1.0, 0.0, 1.0])
    return None


def _case_shapes_from_input_shapes(kernel_name: str, input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, int]:
    if kernel_name == "any_kernel_dim":
        m, n = input_shapes["inp"]
        return {"M": int(m), "N": int(n)}
    if kernel_name == "group_norm_kernel":
        n, c, hw = input_shapes["X"]
        # bounded spec chooses num_groups=2, group_size=1 for C=2
        return {"N": int(n), "C": int(c), "HW": int(hw), "num_groups": 2, "group_size": int(c) // 2}
    raise ValueError(f"unknown kernel_name for bounded case shapes: {kernel_name}")


__all__ = [
    "MetamorphicResult",
    "MetamorphicSuiteReport",
    "BoundedExhaustiveReport",
    "run_metamorphic_suite",
    "run_bounded_exhaustive",
]
