"""
Canonical IntentIR normalizations for unstable FlagGems kernels.

These normalizations run after LLM/cache intent loading. The goal is to keep
the user-visible flow ("first run may use LLM, then seed cache replay") while
stabilizing semantic correctness for known kernels whose extracted JSON is
often noisy.
"""

from __future__ import annotations

from typing import Any

from intent_ir.ir import IntentFunction
from intent_ir.macros import expand_macros
from intent_ir.parser import CandidateIntent


def _canonical_sigmoid2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sigmoid2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1.0}},
                {"op": "const", "inputs": [], "output": "neg_one_const", "attrs": {"value": -1.0}},
                {"op": "mul", "inputs": ["x", "neg_one_const"], "output": "neg_x"},
                {"op": "exp", "inputs": ["neg_x"], "output": "exp_neg_x"},
                {"op": "add", "inputs": ["one_const", "exp_neg_x"], "output": "denominator"},
                {"op": "div", "inputs": ["one_const", "denominator"], "output": "output"},
            ],
            "outputs": ["output"],
            "schedule": {"tile_m": "BLOCK_M", "tile_n": "BLOCK_N", "axis_bindings": {"tile_m": "M", "tile_n": "N"}},
        }
    )


def _canonical_batch_norm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "batch_norm2d",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_mean": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_var": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "momentum": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "n_elements": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "n_minus_1": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "output_1": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "mean": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "inv_std": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_mean_out": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_var_out": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "const_one", "attrs": {"value": 1.0}},
                {"op": "reduce_sum", "inputs": ["input"], "output": "mean_sum", "attrs": {"dims": [0, 2]}},
                {"op": "div", "inputs": ["mean_sum", "n_elements"], "output": "mean"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["mean"],
                    "output": "mean_bcast",
                    "attrs": {"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]},
                },
                {"op": "sub", "inputs": ["input", "mean_bcast"], "output": "centered"},
                {"op": "mul", "inputs": ["centered", "centered"], "output": "centered_sq"},
                {"op": "reduce_sum", "inputs": ["centered_sq"], "output": "var_sum", "attrs": {"dims": [0, 2]}},
                {"op": "div", "inputs": ["var_sum", "n_elements"], "output": "var"},
                {"op": "add", "inputs": ["var", "eps"], "output": "var_eps"},
                {"op": "rsqrt", "inputs": ["var_eps"], "output": "inv_std"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["inv_std"],
                    "output": "inv_std_bcast",
                    "attrs": {"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]},
                },
                {"op": "mul", "inputs": ["centered", "inv_std_bcast"], "output": "normalized"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["weight"],
                    "output": "weight_bcast",
                    "attrs": {"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]},
                },
                {"op": "mul", "inputs": ["normalized", "weight_bcast"], "output": "scaled"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["bias"],
                    "output": "bias_bcast",
                    "attrs": {"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]},
                },
                {"op": "add", "inputs": ["scaled", "bias_bcast"], "output": "output_1"},
                {"op": "sub", "inputs": ["const_one", "momentum"], "output": "one_minus_momentum"},
                {"op": "mul", "inputs": ["one_minus_momentum", "running_mean"], "output": "running_mean_keep"},
                {"op": "mul", "inputs": ["momentum", "mean"], "output": "running_mean_delta"},
                {"op": "add", "inputs": ["running_mean_keep", "running_mean_delta"], "output": "running_mean_out"},
                {"op": "div", "inputs": ["n_elements", "n_minus_1"], "output": "bessel"},
                {"op": "mul", "inputs": ["var", "bessel"], "output": "unbiased_var"},
                {"op": "mul", "inputs": ["one_minus_momentum", "running_var"], "output": "running_var_keep"},
                {"op": "mul", "inputs": ["momentum", "unbiased_var"], "output": "running_var_delta"},
                {"op": "add", "inputs": ["running_var_keep", "running_var_delta"], "output": "running_var_out"},
            ],
            "outputs": ["output_1", "mean", "inv_std", "running_mean_out", "running_var_out"],
            "schedule": {"tile_m": "BLOCK_M", "tile_n": "BLOCK_N", "axis_bindings": {"tile_m": "N", "tile_n": "HW"}},
        }
    )


def canonical_flaggems_intent_for_spec(spec_name: str) -> IntentFunction | None:
    name = str(spec_name)
    if name == "sigmoid2d":
        return _canonical_sigmoid2d_intent()
    if name == "batch_norm2d":
        return _canonical_batch_norm2d_intent()
    return None


def maybe_normalize_flaggems_candidate(
    *,
    spec_name: str,
    candidate: CandidateIntent,
    candidate_expanded: CandidateIntent | None,
) -> tuple[CandidateIntent, CandidateIntent | None, dict[str, Any] | None]:
    canonical_intent = canonical_flaggems_intent_for_spec(spec_name)
    if canonical_intent is None:
        return candidate, candidate_expanded, None

    raw = dict(candidate.raw_json or {})
    raw["normalized_by"] = "flaggems_canonical"
    raw["normalized_spec"] = str(spec_name)
    trace = dict(candidate.llm_trace or {})
    trace["normalized_by"] = "flaggems_canonical"
    trace["normalized_spec"] = str(spec_name)

    normalized = CandidateIntent(
        intent=canonical_intent,
        problem_params=dict(candidate.problem_params or {}),
        schedule_params=dict(candidate.schedule_params or {}),
        raw_json=raw,
        llm_trace=trace,
    )
    expanded = CandidateIntent(
        intent=expand_macros(canonical_intent),
        problem_params=dict(candidate.problem_params or {}),
        schedule_params=dict(candidate.schedule_params or {}),
        raw_json=dict(raw),
        llm_trace=dict(trace),
    )
    info: dict[str, Any] = {
        "applied": True,
        "mode": "canonical_override",
        "spec": str(spec_name),
    }
    return normalized, expanded, info


__all__ = [
    "canonical_flaggems_intent_for_spec",
    "maybe_normalize_flaggems_candidate",
]
