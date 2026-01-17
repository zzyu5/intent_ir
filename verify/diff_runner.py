"""
Differential runner comparing intent interpreter vs reference runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

from intent_ir.ir import IntentFunction
from intent_ir.macros import expand_macros
from verify.gen_cases import TestCase
from verify.interpreter import execute_intent
from verify.tolerances import infer_tolerances


@dataclass
class DiffResult:
    ok: bool
    max_abs_err: float
    max_rel_err: float
    first_bad_index: Tuple[int, ...] | None
    summary: str


@dataclass
class Counterexample:
    case: TestCase
    diff: DiffResult
    intent_json: Dict[str, Any]
    facts_summary: Dict[str, Any] | None
    hints: List[str]


def run_diff(
    intent: IntentFunction,
    run_ref_fn: Callable[[TestCase], Dict[str, np.ndarray]],
    cases: Iterable[TestCase],
    constraints=None,
    tolerances=None,
) -> Tuple[List[DiffResult], List[Counterexample]]:
    inferred_tol = None if tolerances is None else dict(tolerances)
    cases_list = list(cases)
    try:
        intent_exec = expand_macros(intent)
    except Exception as e:
        diff = DiffResult(
            ok=False,
            max_abs_err=0.0,
            max_rel_err=0.0,
            first_bad_index=None,
            summary=f"macro expansion error: {type(e).__name__}: {e}",
        )
        return [diff for _ in cases_list], [
            Counterexample(
                case=cases_list[0] if cases_list else TestCase(shapes={}, dtypes={}, seed=0),
                diff=diff,
                intent_json=intent.to_json_dict(),
                facts_summary=None,
                hints=["fix compiler macro expansion (not an LLM issue)"],
            )
        ]
    diffs: List[DiffResult] = []
    counterexamples: List[Counterexample] = []
    for case in cases_list:
        ref_out = run_ref_fn(case)
        # Make IO naming robust: LLM may emit Input/Output while runner returns input/output
        # (or *_ptr variants). Add non-destructive aliases so interpreter can resolve names.
        ref_out = _with_io_aliases(intent_exec, ref_out)
        if inferred_tol is None:
            inferred_tol = infer_tolerances(intent_exec, ref_out=ref_out).to_dict()
        # derive bindings and align input shapes
        bindings = dict(case.shapes)
        # Common axis aliases (align kernel-signature symbols with user-friendly names).
        if "batch" in bindings and "Z" not in bindings:
            bindings["Z"] = bindings["batch"]
        if "Z" in bindings and "batch" not in bindings:
            bindings["batch"] = bindings["Z"]
        if "group" in bindings and "num_groups" not in bindings:
            bindings["num_groups"] = bindings["group"]
        if "num_groups" in bindings and "C" in bindings and "group_size" not in bindings:
            g = int(bindings["num_groups"])
            c = int(bindings["C"])
            if g <= 0:
                raise ValueError(f"invalid num_groups: {g}")
            bindings["group_size"] = c // g if (c % g == 0) else (c + g - 1) // g
        if "group_size" in bindings and "HW" in bindings and "num_elements" not in bindings:
            try:
                bindings["num_elements"] = int(bindings["group_size"]) * int(bindings["HW"])
            except Exception:
                pass
        # Generic grouped reductions: if a kernel uses [M, N] reshaped into [M, G, group_size],
        # bind G = N / group_size when possible (LLM often emits a symbolic G).
        if "N" in bindings and "group_size" in bindings and "G" not in bindings:
            try:
                n = int(bindings["N"])
                gs = int(bindings["group_size"])
                if gs > 0 and n % gs == 0:
                    bindings["G"] = n // gs
            except Exception:
                pass
        # Common derived symbols used by some frontends/LLM outputs (avoid "unbound symbol" failures).
        if "HEAD_DIM" in bindings:
            try:
                hd = int(bindings["HEAD_DIM"])
                if hd > 0:
                    bindings.setdefault("HEAD_DIM_DIV2", hd // 2)
                    bindings.setdefault("HEAD_DIM_DIV_2", hd // 2)
                    bindings.setdefault("HEAD_DIM_HALF", hd // 2)
                    bindings.setdefault("HEAD_DIM_MID", hd // 2)
            except Exception:
                pass
        # Only feed/validate true external inputs (values consumed by the ops graph).
        produced = {op.output for op in intent_exec.ops if op.output}
        used: set[str] = set()
        for op in intent_exec.ops:
            used.update(op.inputs)
        external_inputs = {n for n in used if (n in intent_exec.tensors and n not in produced)}
        inputs = {k: v for k, v in ref_out.items() if k in external_inputs}
        # Some frontends expose semantic scalar inputs (e.g., sm_scale) that may be
        # compiled away in the runtime kernel signature (so the baseline runner
        # cannot return them). If IntentIR models them as scalar tensors, inject
        # derived 0-d arrays so the interpreter can evaluate the graph.
        missing = sorted([n for n in external_inputs if n not in inputs])
        for name in missing:
            tt = intent_exec.tensors.get(name)
            if tt is None:
                continue
            # Only inject scalars (rank-0 tensors).
            if getattr(tt, "shape", None):
                continue
            if case.inputs and name in case.inputs:
                try:
                    inputs[name] = np.asarray(case.inputs[name])
                    continue
                except Exception:
                    pass
            # Derived semantic constants.
            if name == "sm_scale":
                hd = bindings.get("HEAD_DIM")
                if hd is not None and int(hd) > 0:
                    inputs[name] = np.array(1.0 / np.sqrt(float(hd)), dtype=np.float32)
                    continue
            if name in bindings:
                dt = str(getattr(tt, "dtype", "f32"))
                np_dt = np.float32
                if dt == "f16":
                    np_dt = np.float16
                elif dt == "f64":
                    np_dt = np.float64
                elif dt == "i32":
                    np_dt = np.int32
                elif dt == "i64":
                    np_dt = np.int64
                elif dt == "i8":
                    np_dt = np.int8
                elif dt == "u8":
                    np_dt = np.uint8
                elif dt == "bool":
                    np_dt = np.bool_
                inputs[name] = np.array(bindings[name], dtype=np_dt)
        # Strict "original view" check: inputs must match declared tensor shapes exactly.
        # If the LLM needs a grouped/view shape, it must introduce explicit reshape ops
        # inside IntentIR, rather than redefining the input tensor shape.
        for name, arr in list(inputs.items()):
            if name not in intent_exec.tensors:
                continue
            expected_shape = _resolve_tensor_shape(intent_exec.tensors[name], bindings)
            if expected_shape is None:
                # If an input tensor shape contains unbound symbols, we cannot reliably
                # check semantics; treat as failure so the LLM must align with runtime axes.
                diff = DiffResult(
                    ok=False,
                    max_abs_err=0.0,
                    max_rel_err=0.0,
                    first_bad_index=None,
                    summary=f"unbound symbols in input shape for {name}: intent shape={intent_exec.tensors[name].shape}",
                )
                diffs.append(diff)
                counterexamples.append(
                    Counterexample(
                        case=case,
                        diff=diff,
                        intent_json=intent.to_json_dict(),
                        facts_summary=None,
                        hints=["ensure all input tensor dims are bound by case shapes (e.g., N/C/HW), and keep original view"],
                    )
                )
                break
            if tuple(arr.shape) != tuple(expected_shape):
                diff = DiffResult(
                    ok=False,
                    max_abs_err=0.0,
                    max_rel_err=0.0,
                    first_bad_index=None,
                    summary=f"input shape mismatch for {name}: ref {arr.shape} vs intent {expected_shape}",
                )
                diffs.append(diff)
                counterexamples.append(
                    Counterexample(
                        case=case,
                        diff=diff,
                        intent_json=intent.to_json_dict(),
                        facts_summary=None,
                        hints=[
                            "keep original input view; if you need grouped/view shapes, add intent.reshape/broadcast_in_dim ops",
                        ],
                    )
                )
                break
        else:
            # Feed symbolic shape bindings from TestCase for broadcast/reshape
            try:
                with np.errstate(all="ignore"):
                    pred = execute_intent(intent_exec, inputs, shape_bindings=bindings)
                diff = _compare_outputs(pred, ref_out, intent_exec.outputs, inferred_tol)
            except Exception as e:
                diff = DiffResult(
                    ok=False,
                    max_abs_err=0.0,
                    max_rel_err=0.0,
                    first_bad_index=None,
                    summary=f"interpreter error: {type(e).__name__}: {e}",
                )
                pred = {}
            diffs.append(diff)
            if not diff.ok:
                counterexamples.append(
                    Counterexample(
                        case=case,
                        diff=diff,
                        intent_json=intent.to_json_dict(),
                        facts_summary=None,
                        hints=["check missing ops/reshape/broadcast or wrong reduce axes"],
                    )
                )
    return diffs, counterexamples


def _normalize_io_name(name: str) -> str:
    s = str(name).strip()
    s = s.strip("_")
    s = s.lower()
    # common ptr naming patterns
    if s.startswith("ptr_"):
        s = s[4:]
    if s.endswith("_ptr"):
        s = s[:-4]
    if s.endswith("ptr") and len(s) > 3:
        # e.g. inputptr -> input
        s = s[:-3]
    # Common shorthand aliases from copied kernels.
    if s == "i":
        s = "input"
    if s == "o":
        s = "output"
    if s == "x":
        s = "input"
    if s == "y":
        s = "output"
    if s == "w":
        s = "weight"
    if s == "b":
        s = "bias"
    if s == "in":
        s = "input"
    if s == "out":
        s = "output"
    # Make common style differences alias-friendly (row_mask vs RowMask).
    # Do this late so *_ptr handling above can still match.
    s = s.replace("_", "")
    return s


def _with_io_aliases(intent: IntentFunction, ref_io: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Return a shallow-copied dict that contains the original ref_io plus aliases
    for intent tensor names (case-insensitive + *_ptr tolerant).
    """
    out = dict(ref_io)
    # build normalized->original ref keys (prefer exact if collision)
    norm_to_keys: Dict[str, List[str]] = {}
    for k in ref_io.keys():
        norm_to_keys.setdefault(_normalize_io_name(k), []).append(k)

    # Add aliases for all intent tensor names and outputs.
    wanted = set(intent.tensors.keys()) | set(intent.outputs)
    for name in wanted:
        if name in out:
            continue
        norm = _normalize_io_name(name)
        keys = norm_to_keys.get(norm) or []
        if keys:
            if len(keys) == 1:
                out[name] = ref_io[keys[0]]
                continue
            # Collision: pick the best match (prefer exact case-insensitive match).
            lower_name = str(name).lower()
            preferred = None
            for k in keys:
                if str(k).lower() == lower_name:
                    preferred = k
                    break
            if preferred is None and norm in ref_io:
                preferred = norm
            if preferred is None:
                preferred = keys[0]
            out[name] = ref_io[preferred]
            continue
        # Fallback: substring match on normalized keys (e.g., out_mean_ptr -> Mean).
        # Avoid overly-short names (e.g., "N", "C") accidentally matching "input".
        if len(norm) >= 3:
            candidates = [k for k in ref_io.keys() if norm and (norm in _normalize_io_name(k))]
            if len(candidates) == 1:
                out[name] = ref_io[candidates[0]]
    return out


def _compare_outputs(pred: Dict[str, np.ndarray], ref: Dict[str, np.ndarray], outputs: list[str], tol) -> DiffResult:
    max_abs = 0.0
    max_rel = 0.0
    bad_idx = None
    for name in outputs:
        if name not in ref:
            return DiffResult(False, max_abs, max_rel, bad_idx, f"reference missing output {name}")
        ref_arr = ref[name]
        if name not in pred:
            return DiffResult(False, max_abs, max_rel, bad_idx, f"missing output {name}")
        p = pred[name]
        if p.shape != ref_arr.shape:
            # Allow only "reshape-equivalent" alignments that preserve structure:
            # squeeze/unsqueeze extra unit (size=1) dims to match reference.
            p2 = _squeeze_unit_dims_to_match(p, ref_arr.shape)
            if p2 is None:
                p2 = _unsqueeze_unit_dims_to_match(p, ref_arr.shape)
            if p2 is None:
                return DiffResult(False, max_abs, max_rel, bad_idx, f"shape mismatch for {name}: {p.shape} vs {ref_arr.shape}")
            p = p2
        if ref_arr.dtype == bool or p.dtype == bool:
            neq = np.not_equal(p.astype(bool), ref_arr.astype(bool))
            if np.any(neq):
                bi = tuple(np.unravel_index(np.argmax(neq.astype(np.int32)), neq.shape))
                return DiffResult(False, float("inf"), float("inf"), bi, f"mismatch in {name} (bool)")
            continue
        else:
            # Be strict about non-finite values: allow them only if the pattern AND
            # values match (e.g., NaN vs NaN at the same indices).
            p_nonfinite = ~np.isfinite(p)
            r_nonfinite = ~np.isfinite(ref_arr)
            if np.any(p_nonfinite) or np.any(r_nonfinite):
                if not np.array_equal(p_nonfinite, r_nonfinite):
                    return DiffResult(False, float("inf"), float("inf"), None, f"non-finite pattern mismatch in {name}")
                if not np.array_equal(p[p_nonfinite], ref_arr[r_nonfinite]):
                    return DiffResult(False, float("inf"), float("inf"), None, f"non-finite values mismatch in {name}")
            finite = ~(p_nonfinite | r_nonfinite)
            abs_err = np.zeros_like(ref_arr, dtype=np.float64)
            rel_err = np.zeros_like(ref_arr, dtype=np.float64)
            if np.any(finite):
                # Avoid integer overflow in abs()/sub (e.g., abs(int8(-128))).
                p_f = p.astype(np.float64, copy=False)
                r_f = ref_arr.astype(np.float64, copy=False)
                abs_err[finite] = np.abs(p_f[finite] - r_f[finite])
                rel_err[finite] = abs_err[finite] / (np.abs(r_f[finite]) + 1e-8)
        max_abs_err = float(abs_err.max()) if abs_err.size else 0.0
        max_rel_err = float(rel_err.max()) if rel_err.size else 0.0
        max_abs = max(max_abs, max_abs_err)
        max_rel = max(max_rel, max_rel_err)

        # Pass/fail uses a per-element tolerance (np.allclose-style).
        # This avoids false negatives where the max-abs and max-rel occur at different indices.
        atol = float(tol.get("atol", 1e-3))
        rtol = float(tol.get("rtol", 1e-3))
        # Cast before abs() to avoid integer overflow (e.g., abs(int8(-128)) -> -128).
        thresh = atol + rtol * np.abs(ref_arr.astype(np.float64, copy=False))
        viol = (abs_err > thresh) & finite
        if np.any(viol):
            margin = np.where(viol, abs_err - thresh, -np.inf)
            bad_idx = tuple(np.unravel_index(int(np.argmax(margin)), margin.shape))
            return DiffResult(False, max_abs, max_rel, bad_idx, f"mismatch in {name}")
    return DiffResult(True, max_abs, max_rel, bad_idx, "ok")


def _squeeze_unit_dims_to_match(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray | None:
    """
    Try to drop ONLY size=1 dimensions from `arr` so that it matches `target_shape`.
    This allows [N,G] <-> [N,G,1] etc, but will never allow flattening like [N,G] <-> [NG].
    """
    if tuple(arr.shape) == tuple(target_shape):
        return arr
    pred_shape = tuple(arr.shape)
    ref_shape = tuple(target_shape)
    if len(pred_shape) < len(ref_shape):
        return None

    drop_axes: List[int] = []
    i = 0
    j = 0
    while i < len(pred_shape) and j < len(ref_shape):
        if pred_shape[i] == ref_shape[j]:
            i += 1
            j += 1
            continue
        if pred_shape[i] == 1:
            drop_axes.append(i)
            i += 1
            continue
        return None
    # remaining pred dims must all be unit dims
    while i < len(pred_shape):
        if pred_shape[i] == 1:
            drop_axes.append(i)
            i += 1
            continue
        return None
    if j != len(ref_shape):
        return None
    if not drop_axes:
        return None
    squeezed = np.squeeze(arr, axis=tuple(drop_axes))
    return squeezed if tuple(squeezed.shape) == ref_shape else None


def _unsqueeze_unit_dims_to_match(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray | None:
    """
    Try to insert ONLY size=1 dimensions into `arr` so that it matches `target_shape`.
    This allows [OH,OW] <-> [1,1,OH,OW] etc, but never allows flattening or reordering.
    """
    if tuple(arr.shape) == tuple(target_shape):
        return arr
    pred_shape = tuple(arr.shape)
    ref_shape = tuple(target_shape)
    if len(pred_shape) > len(ref_shape):
        return None

    new_shape: List[int] = []
    i = 0
    for d in ref_shape:
        if i < len(pred_shape) and pred_shape[i] == d:
            new_shape.append(int(d))
            i += 1
            continue
        if d == 1:
            new_shape.append(1)
            continue
        return None
    if i != len(pred_shape):
        return None
    try:
        reshaped = np.reshape(arr, tuple(new_shape))
    except Exception:
        return None
    return reshaped if tuple(reshaped.shape) == ref_shape else None


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _resolve_tensor_shape(tensor, bindings: Dict[str, int]):
    shape = []
    for d in tensor.shape:
        if hasattr(d, "kind") and getattr(d, "kind") == "sym":
            val = bindings.get(d.value)
            if val is None:
                return None
            shape.append(val)
        elif hasattr(d, "kind") and getattr(d, "kind") == "const":
            shape.append(int(d.value))
        elif isinstance(d, str) and d in bindings:
            shape.append(bindings[d])
        elif isinstance(d, (int, float)):
            shape.append(int(d))
        else:
            return None
    return tuple(shape)


__all__ = ["DiffResult", "Counterexample", "run_diff"]
