"""
Task5 v1.2: Mutation-kill harness.

Goal: quantify "falsification power" of our verification pipeline by generating
synthetic wrong IntentIR variants ("mutants") from a passing intent, then
measuring how many get rejected by:
- Stage A: static validation (IntentIR vs TTIR certificate)
- Stage B: dynamic differential testing (ref vs interpreter)
- Stage C: metamorphic relations (operator invariants)

This module never calls the LLM.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from intent_ir.ir import IntentFunction, Op
from verify.diff_runner import DiffResult, run_diff
from verify.gen_cases import TestCase
from verify.metamorphic import MetamorphicSuiteReport, run_bounded_exhaustive, run_metamorphic_suite


@dataclass
class MutationOutcome:
    mutant_id: int
    killed_by: str  # "A_static" | "B_diff" | "C_metamorphic" | "C_bounded" | "survived" | "invalid"
    detail: str
    diff_summary: Optional[str] = None


@dataclass
class MutationReport:
    total: int
    killed: int
    survived: int
    killed_by_stage: Dict[str, int] = field(default_factory=dict)
    outcomes: List[MutationOutcome] = field(default_factory=list)

    @property
    def kill_rate(self) -> float:
        return 0.0 if self.total == 0 else float(self.killed) / float(self.total)


_Runner = Callable[[TestCase], Dict[str, np.ndarray]]


def generate_mutants(intent: IntentFunction, *, n: int, seed: int = 0) -> List[IntentFunction]:
    rng = random.Random(seed)
    base_fp = _fingerprint(intent)
    mutants: List[IntentFunction] = []
    attempts = 0
    while len(mutants) < n and attempts < n * 20:
        attempts += 1
        m = copy.deepcopy(intent)
        if _apply_one_mutation(m, rng):
            try:
                m.validate()
            except Exception:
                # Keep invalid mutants too; they count as killed early.
                pass
            if _fingerprint(m) == base_fp:
                # Avoid "no-op" mutants (e.g., 0.0 -> 0.0).
                continue
            mutants.append(m)
    return mutants


def run_mutation_kill(
    kernel_name: str,
    *,
    intent: IntentFunction,
    run_ref_fn: _Runner,
    diff_cases: Sequence[TestCase],
    metamorphic_base_case: TestCase,
    static_validate_fn: Callable[[IntentFunction], object] | None = None,
    n_mutants: int = 16,
    seed: int = 0,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    include_bounded: bool = True,
) -> MutationReport:
    mutants = generate_mutants(intent, n=n_mutants, seed=seed)
    killed_by: Dict[str, int] = {"A_static": 0, "B_diff": 0, "C_metamorphic": 0, "C_bounded": 0, "invalid": 0}
    outcomes: List[MutationOutcome] = []

    for mid, m in enumerate(mutants):
        # Stage A: optional frontend static validation (certificate/obligations).
        if static_validate_fn is not None:
            try:
                sv = static_validate_fn(m)
                ok = bool(getattr(sv, "ok", False))
                reasons = getattr(sv, "reasons", None)
                reasons_list = reasons if isinstance(reasons, list) else []
            except Exception as e:
                killed_by["invalid"] += 1
                outcomes.append(
                    MutationOutcome(mutant_id=mid, killed_by="invalid", detail=f"static_validate error: {type(e).__name__}: {e}")
                )
                continue
            if not ok:
                killed_by["A_static"] += 1
                outcomes.append(
                    MutationOutcome(
                        mutant_id=mid,
                        killed_by="A_static",
                        detail="; ".join(str(x) for x in reasons_list) or "static obligations failed",
                    )
                )
                continue

        # Stage B: dynamic diff (reuse the same cases as the pipeline, but keep it small)
        try:
            diffs, _ = run_diff(m, run_ref_fn, diff_cases, tolerances={"atol": atol, "rtol": rtol})
            if not all(d.ok for d in diffs):
                worst = max(diffs, key=lambda d: (not d.ok, d.max_abs_err))
                killed_by["B_diff"] += 1
                outcomes.append(MutationOutcome(mutant_id=mid, killed_by="B_diff", detail="dynamic diff mismatch", diff_summary=worst.summary))
                continue
        except Exception as e:
            killed_by["B_diff"] += 1
            outcomes.append(MutationOutcome(mutant_id=mid, killed_by="B_diff", detail=f"diff error: {type(e).__name__}: {e}"))
            continue

        # Stage C: metamorphic invariants (only for survivors so far)
        meta: MetamorphicSuiteReport = run_metamorphic_suite(
            kernel_name,
            m,
            run_ref_fn,
            base_case=metamorphic_base_case,
            atol=atol,
            rtol=rtol,
            rng_seed=seed + mid + 17,
        )
        if not meta.ok:
            killed_by["C_metamorphic"] += 1
            bad = next((r for r in meta.results if not r.ok), None)
            outcomes.append(MutationOutcome(mutant_id=mid, killed_by="C_metamorphic", detail=(bad.detail if bad else "metamorphic failed")))
            continue

        if include_bounded:
            bounded = run_bounded_exhaustive(kernel_name, m, run_ref_fn, atol=atol, rtol=rtol, max_cases=None, rng_seed=seed + mid + 101)
            if bounded.total > 0 and not bounded.ok:
                killed_by["C_bounded"] += 1
                outcomes.append(
                    MutationOutcome(
                        mutant_id=mid,
                        killed_by="C_bounded",
                        detail=f"bounded exhaustive failed at {bounded.checked}/{bounded.total}: {bounded.first_failure_summary}",
                    )
                )
                continue

        outcomes.append(MutationOutcome(mutant_id=mid, killed_by="survived", detail="passed A+B+C"))

    total = len(mutants)
    survived = sum(1 for o in outcomes if o.killed_by == "survived")
    killed = total - survived
    # drop zero entries for nicer printing
    killed_by_stage = {k: v for k, v in killed_by.items() if v}
    return MutationReport(total=total, killed=killed, survived=survived, killed_by_stage=killed_by_stage, outcomes=outcomes)


def _apply_one_mutation(intent: IntentFunction, rng: random.Random) -> bool:
    # Pick a mutation operator that can apply.
    ops = intent.ops
    candidates: List[Callable[[], bool]] = []

    def pick_indices(pred):
        return [i for i, op in enumerate(ops) if pred(op)]

    reduce_ids = pick_indices(lambda o: o.op in {"reduce_sum", "reduce_max", "reduce_any"})
    softmax_ids = pick_indices(lambda o: o.op == "softmax")
    transpose_ids = pick_indices(lambda o: o.op == "transpose")
    elemwise_ids = pick_indices(lambda o: o.op in {"add", "sub", "mul", "div", "max", "min", "ne"})
    const_ids = pick_indices(lambda o: o.op == "const")
    bcast_ids = pick_indices(lambda o: o.op == "broadcast_in_dim")

    if reduce_ids:
        def mut_reduce_axes() -> bool:
            idx = rng.choice(reduce_ids)
            op = ops[idx]
            axes = _get_axes_list(op)
            if axes is None:
                return False
            input_rank = _infer_input_rank(intent, op.inputs[0])
            if input_rank is None or input_rank < 1:
                return False
            # Pick a new axis in range that differs.
            axis_pool = list(range(input_rank))
            if not axis_pool:
                return False
            old = list(axes)
            # mutate one position
            pos = rng.randrange(len(old))
            new_ax = (old[pos] + 1) % input_rank
            old[pos] = new_ax
            _set_axes_list(op, old)
            return True

        candidates.append(mut_reduce_axes)

    if softmax_ids:
        def mut_softmax_axis() -> bool:
            idx = rng.choice(softmax_ids)
            op = ops[idx]
            axis = int(op.attrs.get("axis", -1))
            input_rank = _infer_input_rank(intent, op.inputs[0])
            if input_rank is None or input_rank < 1:
                return False
            axis_mod = axis % input_rank
            new_axis = (axis_mod - 1) % input_rank
            op.attrs["axis"] = int(new_axis)
            return True

        candidates.append(mut_softmax_axis)

    if transpose_ids:
        def mut_transpose_perm() -> bool:
            idx = rng.choice(transpose_ids)
            op = ops[idx]
            perm = list(op.attrs.get("perm") or [])
            if len(perm) < 2:
                return False
            i, j = rng.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
            op.attrs["perm"] = perm
            return True

        candidates.append(mut_transpose_perm)

    if bcast_ids:
        def mut_broadcast_dims() -> bool:
            idx = rng.choice(bcast_ids)
            op = ops[idx]
            out_shape = op.attrs.get("out_shape") or []
            bcast_dims = list(op.attrs.get("broadcast_dims") or [])
            if not out_shape or not bcast_dims:
                return False
            out_rank = len(out_shape)
            # shift one dim
            k = rng.randrange(len(bcast_dims))
            bcast_dims[k] = int((int(bcast_dims[k]) + 1) % out_rank)
            op.attrs["broadcast_dims"] = bcast_dims
            return True

        candidates.append(mut_broadcast_dims)

    if elemwise_ids:
        def mut_elemwise_op() -> bool:
            idx = rng.choice(elemwise_ids)
            op = ops[idx]
            pool = ["add", "sub", "mul", "div", "max", "min"]
            if op.op not in pool:
                return False
            new = rng.choice([p for p in pool if p != op.op])
            op.op = new
            return True

        candidates.append(mut_elemwise_op)

    if const_ids:
        def mut_const_value() -> bool:
            idx = rng.choice(const_ids)
            op = ops[idx]
            v = op.attrs.get("value")
            # Keep placeholders untouched (we want semantic mutants, not format-only).
            if isinstance(v, str) and v.startswith("placeholder:"):
                return False
            if isinstance(v, (int, float, np.number)):
                op.attrs["value"] = float(v) + 1.0
                return True
            if isinstance(v, str):
                if v == "eps":
                    op.attrs["value"] = 1e-3
                    return True
                # try simple numeric strings
                try:
                    op.attrs["value"] = float(v) + 1.0
                    return True
                except Exception:
                    return False
            return False

        candidates.append(mut_const_value)

    if not candidates:
        return False

    rng.shuffle(candidates)
    for f in candidates:
        try:
            if f():
                return True
        except Exception:
            continue
    return False


def _get_axes_list(op: Op) -> Optional[List[int]]:
    raw = op.attrs.get("axes", op.attrs.get("dims", op.attrs.get("axis")))
    if raw is None:
        return None
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, list) and raw:
        out: List[int] = []
        for x in raw:
            if isinstance(x, int):
                out.append(int(x))
            elif isinstance(x, str) and x.isdigit():
                out.append(int(x))
        return out or None
    return None


def _set_axes_list(op: Op, axes: List[int]) -> None:
    # Normalize into attrs.axes for simplicity.
    op.attrs["axes"] = [int(x) for x in axes]


def _infer_input_rank(intent: IntentFunction, value_name: str) -> Optional[int]:
    # Prefer declared tensor ranks.
    if value_name in intent.tensors:
        return len(intent.tensors[value_name].shape)
    # Fall back: find producing op and infer via its attrs.
    prod = next((op for op in intent.ops if op.output == value_name), None)
    if prod is None:
        return None
    if prod.op == "broadcast_in_dim":
        out_shape = prod.attrs.get("out_shape") or []
        return len(out_shape)
    if prod.op == "transpose":
        return _infer_input_rank(intent, prod.inputs[0])
    if prod.op == "reshape":
        shape = prod.attrs.get("shape") or []
        return len(shape)
    if prod.op in {"reduce_sum", "reduce_max", "reduce_any"}:
        return _infer_input_rank(intent, prod.inputs[0])
    if prod.op == "softmax":
        return _infer_input_rank(intent, prod.inputs[0])
    if prod.op in {"add", "sub", "mul", "div", "max", "min", "ne"}:
        return _infer_input_rank(intent, prod.inputs[0])
    if prod.op == "matmul":
        return 2
    return None


def _fingerprint(intent: IntentFunction) -> str:
    # Stable JSON fingerprint to dedup mutants.
    try:
        payload = intent.to_json_dict()
    except Exception:
        # If to_json_dict fails, fall back to repr.
        return repr(intent)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


__all__ = [
    "MutationOutcome",
    "MutationReport",
    "generate_mutants",
    "run_mutation_kill",
]
