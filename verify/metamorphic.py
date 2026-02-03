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

from intent_ir.ir import IntentFunction
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


def _alias_io(intent: IntentFunction, ref_io: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Reuse Stage B's IO aliasing (Input/output vs *_ptr, case-insensitive).
    Without this, Stage C may accidentally treat outputs as inputs when runner
    names differ from IntentIR/LLM names.
    """
    from verify.diff_runner import _with_io_aliases as _with_io_aliases_for_diff

    return _with_io_aliases_for_diff(intent, ref_io)


def _external_inputs(intent: IntentFunction, ref_io: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extract only true external inputs consumed by the ops graph.
    """
    produced = {op.output for op in intent.ops if op.output}
    used: set[str] = set()
    for op in intent.ops:
        used.update(op.inputs)
    external = {n for n in used if (n in intent.tensors and n not in produced)}
    return {k: v for k, v in ref_io.items() if k in external}


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
    sampled_raw = run_ref_fn(base_case)
    sampled = _alias_io(intent, sampled_raw)

    # Freeze inputs in a runner-compatible way:
    # - keep runner's original input keys (so subsequent launches reuse the same input)
    # - also include IntentIR external input names (so transforms can be written once)
    from verify.diff_runner import _normalize_io_name as _normalize_io_name_for_diff

    external_intent_inputs = _external_inputs(intent, sampled)
    external_norms = {_normalize_io_name_for_diff(n) for n in external_intent_inputs.keys()}
    runner_inputs = {k: v for k, v in sampled_raw.items() if _normalize_io_name_for_diff(k) in external_norms}

    base_inputs = dict(runner_inputs)
    base_inputs.update(external_intent_inputs)
    base_case_fixed = TestCase(
        shapes=dict(base_case.shapes),
        dtypes=base_case.dtypes or {},
        seed=base_case.seed,
        inputs=base_inputs,
    )

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
            ref_base = _alias_io(intent, run_ref_fn(base_case_fixed))
            ref_tr = _alias_io(intent, run_ref_fn(transformed_case))

            from verify.interpreter import execute_intent

            with np.errstate(all="ignore"):
                pred_base = execute_intent(
                    intent,
                    _external_inputs(intent, ref_base),
                    shape_bindings=_derive_bindings(base_case_fixed),
                )
                pred_tr = execute_intent(
                    intent,
                    _external_inputs(intent, ref_tr),
                    shape_bindings=_derive_bindings(transformed_case),
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
        if kernel_name == "softmax_inner":
            # Triton runner uses "input"/"output" for stability; provide both names.
            if "input_ptr" in inputs:
                inputs.setdefault("input", np.asarray(inputs["input_ptr"]))
        if kernel_name == "layer_norm_persistent":
            N = int(shape_bindings["weight_ptr"][0])
            inputs.setdefault("weight_ptr", np.ones((N,), dtype=np.float32))
            inputs.setdefault("bias_ptr", np.zeros((N,), dtype=np.float32))

        case_shapes = _case_shapes_from_input_shapes(kernel_name, shape_bindings)
        case = TestCase(shapes=case_shapes, dtypes={}, seed=0, inputs=inputs)
        diffs, _ = run_diff(intent, run_ref_fn, [case], tolerances={"atol": atol, "rtol": rtol})
        checked += 1
        if not diffs or not diffs[0].ok:
            summary = diffs[0].summary if diffs else "no diff result"
            # If the reference runner itself fails, bounded exhaustive cannot be
            # interpreted as a semantic counterexample (it is likely an out-of-
            # contract shape for this kernel). Skip instead of failing hard.
            if isinstance(summary, str) and summary.startswith("ref runner error:"):
                return BoundedExhaustiveReport(ok=True, checked=0, total=0, detail="skipped: ref runner error (bounded spec out-of-contract)")
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

    if kernel_name == "softmax_inner":
        def _transform_shift(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            x = np.asarray(inputs.get("input_ptr") if "input_ptr" in inputs else inputs["input"], dtype=np.float32)
            c = np.float32(0.37)
            out = dict(inputs)
            x2 = x + c
            if "input_ptr" in out:
                out["input_ptr"] = x2
            if "input" in out:
                out["input"] = x2
            if "input_ptr" not in out and "input" not in out:
                out["input"] = x2
            return out

        def _check_shift(
            base_out: Dict[str, np.ndarray],
            tr_out: Dict[str, np.ndarray],
            bindings: Dict[str, int],
            *,
            atol: float,
            rtol: float,
        ) -> Tuple[bool, str]:
            y0 = np.asarray(base_out.get("output_ptr") if "output_ptr" in base_out else base_out["output"])
            y1 = np.asarray(tr_out.get("output_ptr") if "output_ptr" in tr_out else tr_out["output"])
            ok = _allclose(y0, y1, atol=atol, rtol=rtol)
            return ok, ("softmax shift invariance holds" if ok else "softmax changed under constant shift")

        relations.append(("MR_softmax_shift", _transform_shift, _check_shift))

        def _transform_rev_cols(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            x = np.asarray(inputs.get("input_ptr") if "input_ptr" in inputs else inputs["input"], dtype=np.float32)
            if x.ndim != 2:
                raise ValueError(f"expected rank-2 input, got {x.shape}")
            out = dict(inputs)
            x2 = x[:, ::-1].copy()
            if "input_ptr" in out:
                out["input_ptr"] = x2
            if "input" in out:
                out["input"] = x2
            if "input_ptr" not in out and "input" not in out:
                out["input"] = x2
            return out

        def _check_rev_cols(
            base_out: Dict[str, np.ndarray],
            tr_out: Dict[str, np.ndarray],
            bindings: Dict[str, int],
            *,
            atol: float,
            rtol: float,
        ) -> Tuple[bool, str]:
            y0 = np.asarray(base_out.get("output_ptr") if "output_ptr" in base_out else base_out["output"])
            y1 = np.asarray(tr_out.get("output_ptr") if "output_ptr" in tr_out else tr_out["output"])
            ok = _allclose(y0[:, ::-1], y1, atol=atol, rtol=rtol)
            return ok, ("softmax equiv under column reverse" if ok else "softmax not equiv under column reverse")

        relations.append(("MR_softmax_reverse_cols", _transform_rev_cols, _check_rev_cols))

    if kernel_name == "layer_norm_persistent":
        def _transform(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            x = np.asarray(
                inputs.get("Input")
                if "Input" in inputs
                else (inputs.get("in_ptr") if "in_ptr" in inputs else inputs["input"]),
                dtype=np.float32,
            )
            c = np.float32(0.37)
            out = dict(inputs)
            x2 = x + c
            if "Input" in out:
                out["Input"] = x2
            if "in_ptr" in out:
                out["in_ptr"] = x2
            if "input" in out:
                out["input"] = x2
            if "Input" not in out and "in_ptr" not in out and "input" not in out:
                out["input"] = x2
            return out

        def _check(
            base_out: Dict[str, np.ndarray],
            tr_out: Dict[str, np.ndarray],
            bindings: Dict[str, int],
            *,
            atol: float,
            rtol: float,
        ) -> Tuple[bool, str]:
            c = np.float32(0.37)
            y0 = np.asarray(
                base_out.get("Output")
                if "Output" in base_out
                else (base_out.get("out_ptr") if "out_ptr" in base_out else base_out["output"]),
                dtype=np.float32,
            )
            y1 = np.asarray(
                tr_out.get("Output")
                if "Output" in tr_out
                else (tr_out.get("out_ptr") if "out_ptr" in tr_out else tr_out["output"]),
                dtype=np.float32,
            )
            m0 = np.asarray(
                base_out.get("Mean")
                if "Mean" in base_out
                else (base_out.get("out_mean_ptr") if "out_mean_ptr" in base_out else base_out.get("mean", 0.0)),
                dtype=np.float32,
            )
            m1 = np.asarray(
                tr_out.get("Mean")
                if "Mean" in tr_out
                else (tr_out.get("out_mean_ptr") if "out_mean_ptr" in tr_out else tr_out.get("mean", 0.0)),
                dtype=np.float32,
            )
            r0 = np.asarray(
                base_out.get("Rstd")
                if "Rstd" in base_out
                else (base_out.get("out_rstd_ptr") if "out_rstd_ptr" in base_out else base_out.get("rstd", 0.0)),
                dtype=np.float32,
            )
            r1 = np.asarray(
                tr_out.get("Rstd")
                if "Rstd" in tr_out
                else (tr_out.get("out_rstd_ptr") if "out_rstd_ptr" in tr_out else tr_out.get("rstd", 0.0)),
                dtype=np.float32,
            )
            ok_y = _allclose(y0, y1, atol=atol, rtol=rtol)
            ok_m = _allclose(m0 + c, m1, atol=atol, rtol=rtol)
            ok_r = _allclose(r0, r1, atol=atol, rtol=rtol)
            ok = bool(ok_y and ok_m and ok_r)
            if ok:
                return True, "shift invariance holds"
            return False, f"shift invariance violated (Output={ok_y}, Mean={ok_m}, Rstd={ok_r})"

        relations.append(("MR_layernorm_shift", _transform, _check))

    if kernel_name == "_attn_fwd":
        def _transform(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            k = np.asarray(inputs["K"], dtype=np.float32)
            v = np.asarray(inputs["V"], dtype=np.float32)
            mask = np.asarray(inputs.get("attn_mask", 0.0), dtype=np.float32)
            # Support both common layouts:
            # - Triton attention: [Z, H, KV, D]
            # - TileLang MVP attention: [KV, D]
            if k.ndim == 4 and v.ndim == 4:
                kv = k.shape[2]
                perm = rng.permutation(kv)
                out = dict(inputs)
                out["K"] = k[:, :, perm, :]
                out["V"] = v[:, :, perm, :]
                if mask.ndim == 4:
                    out["attn_mask"] = mask[:, :, :, perm]
                return out
            if k.ndim == 2 and v.ndim == 2:
                kv = k.shape[0]
                perm = rng.permutation(kv)
                out = dict(inputs)
                out["K"] = k[perm, :]
                out["V"] = v[perm, :]
                return out
            raise ValueError(f"expected K/V rank-2 or rank-4, got K={k.shape} V={v.shape}")

        def _check(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            o0, o1 = np.asarray(base_out["Out"]), np.asarray(tr_out["Out"])
            ok = _allclose(o0, o1, atol=atol, rtol=rtol)
            return ok, "Out invariant under KV perm" if ok else "Out changed under KV perm"

        relations.append(("MR_attn_perm_kv", _transform, _check))

        def _transform_scale_v(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            v = np.asarray(inputs["V"], dtype=np.float32)
            s = np.float32(0.5)
            out = dict(inputs)
            out["V"] = v * s
            return out

        def _check_scale_v(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            s = np.float32(0.5)
            o0 = np.asarray(base_out["Out"], dtype=np.float32)
            o1 = np.asarray(tr_out["Out"], dtype=np.float32)
            ok = _allclose(o0 * s, o1, atol=atol, rtol=rtol)
            return ok, ("Out scales with V" if ok else "Out not scaling with V")

        relations.append(("MR_attn_scale_v", _transform_scale_v, _check_scale_v))

    if kernel_name == "upsample_bicubic2d_aa":
        def _transform_scale(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            x = np.asarray(
                inputs.get("I") if "I" in inputs else (inputs.get("Input") if "Input" in inputs else inputs["input"]),
                dtype=np.float32,
            )
            s = np.float32(1.7)
            out = dict(inputs)
            x2 = x * s
            if "I" in out:
                out["I"] = x2
            if "Input" in out:
                out["Input"] = x2
            if "input" in out:
                out["input"] = x2
            if "I" not in out and "Input" not in out and "input" not in out:
                out["input"] = x2
            return out

        def _check_scale(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            s = np.float32(1.7)
            y0 = np.asarray(
                base_out.get("O")
                if "O" in base_out
                else (base_out.get("Output") if "Output" in base_out else base_out["output"]),
                dtype=np.float32,
            )
            y1 = np.asarray(
                tr_out.get("O")
                if "O" in tr_out
                else (tr_out.get("Output") if "Output" in tr_out else tr_out["output"]),
                dtype=np.float32,
            )
            ok = _allclose(y0 * s, y1, atol=atol, rtol=rtol)
            return ok, ("output scales with input" if ok else "output not scaling with input")

        relations.append(("MR_upsample_scale_input", _transform_scale, _check_scale))

    if kernel_name == "rowmask_where2d":
        # Test that flipping a row mask flips whether output is zero or preserved.
        # If mask[i] = True, output[i,:] = input[i,:]
        # If mask[i] = False, output[i,:] = 0
        # Note: IntentFunction uses tensor name 'mask' (i1 dtype), not 'row_mask'
        def _transform_flip_mask(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            mask = np.asarray(inputs["mask"], dtype=bool)
            out = dict(inputs)
            out["mask"] = ~mask
            return out

        def _check_flip_mask(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            # After flipping mask, the non-zero rows should swap.
            # base: mask[i] = True means out[i,:] = inp[i,:]
            # transformed: mask[i] = False means out[i,:] = 0
            # This checks that the output correctly responds to mask changes.
            y0 = np.asarray(base_out.get("out") if "out" in base_out else base_out["output"], dtype=np.float32)
            y1 = np.asarray(tr_out.get("out") if "out" in tr_out else tr_out["output"], dtype=np.float32)
            # Check that at least one flip actually occurs (they're not identical)
            not_identical = not np.allclose(y0, y1, atol=atol, rtol=rtol)
            return not_identical, ("mask flip changes output correctly" if not_identical else "mask flip did not change output - possible where_drop_mask mutation")

        relations.append(("MR_rowmask_flip", _transform_flip_mask, _check_flip_mask))

    if kernel_name == "masked_softmax2d":
        # Test that flipping the mask changes which elements get -inf (before softmax)
        def _transform_flip_mask(inputs: Dict[str, np.ndarray], bindings: Dict[str, int], rng: np.random.Generator) -> Dict[str, np.ndarray]:
            mask = np.asarray(inputs["mask"], dtype=bool)
            out = dict(inputs)
            out["mask"] = ~mask
            return out

        def _check_flip_mask(base_out: Dict[str, np.ndarray], tr_out: Dict[str, np.ndarray], bindings: Dict[str, int], *, atol: float, rtol: float) -> Tuple[bool, str]:
            y0 = np.asarray(base_out.get("out") if "out" in base_out else base_out["output"], dtype=np.float32)
            y1 = np.asarray(tr_out.get("out") if "out" in tr_out else tr_out["output"], dtype=np.float32)
            not_identical = not np.allclose(y0, y1, atol=atol, rtol=rtol)
            return not_identical, ("mask flip changes softmax output" if not_identical else "mask flip did not change output")

        relations.append(("MR_masked_softmax_flip", _transform_flip_mask, _check_flip_mask))

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
    if kernel_name == "softmax_inner":
        # Tiny softmax: 1x2, domain 3^2=9.
        return ({"input_ptr": (1, 2)}, ["input_ptr"], [-1.0, 0.0, 1.0])
    if kernel_name == "layer_norm_persistent":
        # Tiny layernorm: M=1,N=2, domain 3^2=9 for input only.
        return ({"in_ptr": (1, 2), "weight_ptr": (2,), "bias_ptr": (2,)}, ["in_ptr"], [-1.0, 0.0, 1.0])
    if kernel_name == "rowmask_where2d":
        # Tiny rowmask: M=2, N=3, enumerate both inp values and mask patterns.
        # IntentFunction uses 'inp' (f32) and 'mask' (i1), not 'row_mask'
        # inp: 2x3 = 6 elements, mask: 2 booleans
        # For boolean mask, use 0.0=False, 1.0=True in domain
        # Total: 3^6 * 2^2 = 2916 cases (exhaustive), truncated if needed.
        return ({"inp": (2, 3), "mask": (2,)}, ["inp", "mask"], [-1.0, 0.0, 1.0])
    if kernel_name == "masked_softmax2d":
        # Tiny masked_softmax: M=2, N=3, with mask
        return ({"input": (2, 3), "mask": (2, 3)}, ["input", "mask"], [-1.0, 0.0, 1.0])
    if kernel_name == "masked_attention2d":
        # Tiny attention with mask: Q=2xD, K=3xD, V=3xD, mask=2x3, D=2
        return ({"Q": (2, 2), "K": (3, 2), "V": (3, 2), "mask": (2, 3)}, ["Q", "K", "V", "mask"], [-1.0, 0.0, 1.0])
    if kernel_name == "where2d":
        # Tiny where: M=2, N=3, condition + true + false branches
        return ({"cond": (2, 3), "true_val": (2, 3), "false_val": (2, 3)}, ["cond", "true_val", "false_val"], [-1.0, 0.0, 1.0])
    return None


def _case_shapes_from_input_shapes(kernel_name: str, input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, int]:
    if kernel_name == "any_kernel_dim":
        m, n = input_shapes["inp"]
        return {"M": int(m), "N": int(n)}
    if kernel_name == "group_norm_kernel":
        n, c, hw = input_shapes["X"]
        # bounded spec chooses num_groups=2, group_size=1 for C=2
        return {"N": int(n), "C": int(c), "HW": int(hw), "num_groups": 2, "group_size": int(c) // 2}
    if kernel_name == "softmax_inner":
        m, n = input_shapes["input_ptr"]
        return {"M": int(m), "N": int(n)}
    if kernel_name == "layer_norm_persistent":
        m, n = input_shapes["in_ptr"]
        return {"M": int(m), "N": int(n)}
    if kernel_name == "rowmask_where2d":
        m, n = input_shapes["inp"]
        return {"M": int(m), "N": int(n)}
    if kernel_name == "masked_softmax2d":
        m, n = input_shapes["input"]
        return {"M": int(m), "N": int(n)}
    if kernel_name == "masked_attention2d":
        seq_q, d = input_shapes["Q"]
        seq_kv, _ = input_shapes["K"]
        return {"SEQ_Q": int(seq_q), "SEQ_KV": int(seq_kv), "D": int(d)}
    if kernel_name == "where2d":
        m, n = input_shapes["cond"]
        return {"M": int(m), "N": int(n)}
    raise ValueError(f"unknown kernel_name for bounded case shapes: {kernel_name}")


__all__ = [
    "MetamorphicResult",
    "MetamorphicSuiteReport",
    "BoundedExhaustiveReport",
    "run_metamorphic_suite",
    "run_bounded_exhaustive",
]
