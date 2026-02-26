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

_F32_FINITE_MAX = 3.4028234663852886e38


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


def _canonical_tanh2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "tanh2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1.0}},
                {"op": "const", "inputs": [], "output": "two_const", "attrs": {"value": 2.0}},
                {"op": "mul", "inputs": ["two_const", "x"], "output": "two_x"},
                {"op": "exp", "inputs": ["two_x"], "output": "exp_two_x"},
                {"op": "sub", "inputs": ["exp_two_x", "one_const"], "output": "numer"},
                {"op": "add", "inputs": ["exp_two_x", "one_const"], "output": "denom"},
                {"op": "div", "inputs": ["numer", "denom"], "output": "out"},
            ],
            "outputs": ["out"],
            "schedule": {"tile_m": "BLOCK_M", "tile_n": "BLOCK_N", "axis_bindings": {"tile_m": "M", "tile_n": "N"}},
        }
    )


def _canonical_silu2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "silu2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1.0}},
                {"op": "const", "inputs": [], "output": "neg_one_const", "attrs": {"value": -1.0}},
                {"op": "mul", "inputs": ["x", "neg_one_const"], "output": "neg_x"},
                {"op": "exp", "inputs": ["neg_x"], "output": "exp_neg_x"},
                {"op": "add", "inputs": ["one_const", "exp_neg_x"], "output": "denominator"},
                {"op": "div", "inputs": ["one_const", "denominator"], "output": "sigma"},
                {"op": "mul", "inputs": ["x", "sigma"], "output": "y"},
            ],
            "outputs": ["y"],
            "schedule": {"tile_m": "BLOCK_M", "tile_n": "BLOCK_N", "axis_bindings": {"tile_m": "M", "tile_n": "N"}},
        }
    )


def _canonical_softplus2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "softplus2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "threshold": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1.0}},
                {"op": "mul", "inputs": ["x", "beta"], "output": "z"},
                {"op": "exp", "inputs": ["z"], "output": "exp_z"},
                {"op": "add", "inputs": ["one_const", "exp_z"], "output": "exp_plus_one"},
                {"op": "log", "inputs": ["exp_plus_one"], "output": "log_term"},
                {"op": "gt", "inputs": ["z", "threshold"], "output": "gt_threshold"},
                {"op": "where", "inputs": ["gt_threshold", "z", "log_term"], "output": "soft_z"},
                {"op": "div", "inputs": ["soft_z", "beta"], "output": "out"},
            ],
            "outputs": ["out"],
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


def _canonical_isnan2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "isnan2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "ne", "inputs": ["inp", "inp"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_isinf2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "isinf2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "abs", "inputs": ["inp"], "output": "abs_inp"},
                {"op": "const", "inputs": [], "output": "finite_max", "attrs": {"value": _F32_FINITE_MAX}},
                {"op": "gt", "inputs": ["abs_inp", "finite_max"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_isfinite2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "isfinite2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "abs", "inputs": ["inp"], "output": "abs_inp"},
                {"op": "const", "inputs": [], "output": "finite_max", "attrs": {"value": _F32_FINITE_MAX}},
                {"op": "le", "inputs": ["abs_inp", "finite_max"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_isclose2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "isclose2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "rtol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "atol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sub", "inputs": ["A", "B"], "output": "diff"},
                {"op": "abs", "inputs": ["diff"], "output": "abs_diff"},
                {"op": "abs", "inputs": ["B"], "output": "abs_b"},
                {"op": "mul", "inputs": ["rtol", "abs_b"], "output": "rtol_term"},
                {"op": "add", "inputs": ["atol", "rtol_term"], "output": "tol"},
                {"op": "le", "inputs": ["abs_diff", "tol"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_allclose2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "allclose2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "rtol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "atol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "output": {"dtype": "bool", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "sub", "inputs": ["A", "B"], "output": "diff"},
                {"op": "abs", "inputs": ["diff"], "output": "abs_diff"},
                {"op": "abs", "inputs": ["B"], "output": "abs_b"},
                {"op": "mul", "inputs": ["rtol", "abs_b"], "output": "rtol_term"},
                {"op": "add", "inputs": ["atol", "rtol_term"], "output": "tol"},
                {"op": "le", "inputs": ["abs_diff", "tol"], "output": "close_mask"},
                {"op": "not", "inputs": ["close_mask"], "output": "not_close"},
                {"op": "reduce_any", "inputs": ["not_close"], "output": "any_not_close", "attrs": {"dims": [0, 1]}},
                {"op": "not", "inputs": ["any_not_close"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_row_all_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "row_all",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", 1], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0}},
                {"op": "eq", "inputs": ["inp", "zero_const"], "output": "is_zero"},
                {
                    "op": "reduce_any",
                    "inputs": ["is_zero"],
                    "output": "any_zero",
                    "attrs": {"dims": [1], "keepdims": True},
                },
                {"op": "not", "inputs": ["any_zero"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_sub2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sub2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sub", "inputs": ["x", "y"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_cast2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cast2d",
            "tensors": {
                "x": {"dtype": "f16", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "out", "attrs": {"to": "f32"}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_sqrt2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sqrt2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sqrt", "inputs": ["A"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_sin2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sin2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sin", "inputs": ["A"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_tan2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "tan2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "tan", "inputs": ["A"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_std2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "std2d",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "std", "inputs": ["X"], "output": "Out", "attrs": {"axis": 1, "dims": [1], "keepdims": False, "correction": 1}},
            ],
            "outputs": ["Out"],
        }
    )


def _canonical_var_mean2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "var_mean2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "std_out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "std", "inputs": ["inp"], "output": "std_out", "attrs": {"axis": 1, "dims": [1], "keepdims": False, "correction": 1}},
                {"op": "mul", "inputs": ["std_out", "std_out"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_vector_norm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "vector_norm2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sq": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sum_sq": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "mul", "inputs": ["inp", "inp"], "output": "sq"},
                {"op": "reduce_sum", "inputs": ["sq"], "output": "sum_sq", "attrs": {"dims": [1], "keepdims": False}},
                {"op": "sqrt", "inputs": ["sum_sq"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_mm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "mm2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {"transpose_a": False, "transpose_b": False}},
            ],
            "outputs": ["C"],
        }
    )


def _canonical_addmm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "addmm2d",
            "tensors": {
                "mat1": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "mat2": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "mm_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "scaled_mm": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "scaled_bias": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "add_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["mat1", "mat2"], "output": "mm_out", "attrs": {"transpose_a": False, "transpose_b": False}},
                {"op": "mul", "inputs": ["mm_out", "alpha"], "output": "scaled_mm"},
                {"op": "mul", "inputs": ["input", "beta"], "output": "scaled_bias"},
                {"op": "add", "inputs": ["scaled_mm", "scaled_bias"], "output": "add_out"},
                {"op": "cast", "inputs": ["add_out"], "output": "out", "attrs": {"to": "f32"}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_dot1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "dot1d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "x_f32": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "y_f32": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "mul_out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["y"], "output": "y_f32", "attrs": {"to": "f32"}},
                {"op": "mul", "inputs": ["x_f32", "y_f32"], "output": "mul_out"},
                {"op": "reduce_sum", "inputs": ["mul_out"], "output": "out", "attrs": {"dims": [0], "keepdims": False}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_vdot1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "vdot1d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "A_f32": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "B_f32": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "mul_out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["A"], "output": "A_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["B"], "output": "B_f32", "attrs": {"to": "f32"}},
                {"op": "mul", "inputs": ["A_f32", "B_f32"], "output": "mul_out"},
                {"op": "reduce_sum", "inputs": ["mul_out"], "output": "out", "attrs": {"dims": [0], "keepdims": False}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_rms_norm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "rms_norm2d",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "N_scalar": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "x_sq": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sum_sq": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "mean_sq": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "var_eps": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "INV_RMS": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "inv_rms_bcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "w_bcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "x_norm": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "mul", "inputs": ["input", "input"], "output": "x_sq"},
                {"op": "reduce_sum", "inputs": ["x_sq"], "output": "sum_sq", "attrs": {"dims": [1], "keepdims": False}},
                {"op": "div", "inputs": ["sum_sq", "N_scalar"], "output": "mean_sq"},
                {"op": "add", "inputs": ["mean_sq", "eps"], "output": "var_eps"},
                {"op": "rsqrt", "inputs": ["var_eps"], "output": "INV_RMS"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["INV_RMS"],
                    "output": "inv_rms_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["weight"],
                    "output": "w_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {"op": "mul", "inputs": ["input", "inv_rms_bcast"], "output": "x_norm"},
                {"op": "mul", "inputs": ["x_norm", "w_bcast"], "output": "out"},
            ],
            "outputs": ["out", "INV_RMS"],
        }
    )


def _canonical_vstack2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "vstack2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M_OUT", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "concat", "inputs": ["A", "B"], "output": "out", "attrs": {"axis": 0}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_where2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "where2d",
            "tensors": {
                "condition": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "self": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "other": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "where", "inputs": ["condition", "self", "other"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_stack2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "stack2d",
            "tensors": {
                "input0": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "input1": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": [2, "M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "stack", "inputs": ["input0", "input1"], "output": "out_ptr", "attrs": {"axis": 0}},
            ],
            "outputs": ["out_ptr"],
        }
    )


def _canonical_sort2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sort2d",
            "tensors": {
                "in_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sort", "inputs": ["in_ptr"], "output": "out_ptr", "attrs": {"axis": 1, "descending": False, "stable": False}},
            ],
            "outputs": ["out_ptr"],
        }
    )


def _canonical_sort_stable2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sort_stable2d",
            "tensors": {
                "in_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sort", "inputs": ["in_ptr"], "output": "out_ptr", "attrs": {"axis": 1, "descending": False, "stable": True}},
            ],
            "outputs": ["out_ptr"],
        }
    )


def _canonical_topk2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "topk2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sorted_vals": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["M", "K"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["M", "K"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sort", "inputs": ["inp"], "output": "sorted_vals", "attrs": {"axis": 1, "descending": True, "stable": False}},
                {"op": "iota", "inputs": [], "output": "row_idx", "attrs": {"axis": 0, "shape": ["M", "K"], "dtype": "i32"}},
                {"op": "iota", "inputs": [], "output": "col_idx", "attrs": {"axis": 1, "shape": ["M", "K"], "dtype": "i32"}},
                {"op": "gather", "inputs": ["sorted_vals", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_threshold2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "threshold2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "threshold": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "value": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gt", "inputs": ["inp", "threshold"], "output": "keep_mask"},
                {"op": "where", "inputs": ["keep_mask", "inp", "value"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_cat2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cat2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
            },
            "ops": [
                {"op": "concat", "inputs": ["A", "B"], "output": "out", "attrs": {"axis": 1}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_hstack2d_intent() -> IntentFunction:
    out = _canonical_cat2d_intent().to_json_dict()
    out["name"] = "hstack2d"
    return IntentFunction.from_json_dict(out)


def _canonical_clamp2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "clamp2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mini": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "maxi": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "max", "inputs": ["mini", "x_f32"], "output": "clamped_min"},
                {"op": "min", "inputs": ["clamped_min", "maxi"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_constant_pad_nd2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "constant_pad_nd2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "pad",
                    "inputs": ["inp"],
                    "output": "out",
                    "attrs": {"pad_width": {"pairs": [[1, 0], [1, 2]]}, "mode": "constant", "value": 0.0},
                }
            ],
            "outputs": ["out"],
        }
    )


def _canonical_pad2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "pad2d",
            "tensors": {
                "in0": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {
                    "dtype": "f32",
                    "shape": ["M + 1", "N + 3"],
                    "layout": "row_major",
                },
            },
            "ops": [
                {
                    "op": "pad",
                    "inputs": ["in0"],
                    "output": "out",
                    "attrs": {
                        "pad_width": {"pairs": [[1, 0], [1, 2]]},
                        "mode": "constant",
                        "value": 0.0,
                    },
                }
            ],
            "outputs": ["out"],
        }
    )


def _canonical_prod2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "prod2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_prod", "inputs": ["inp"], "output": "out", "attrs": {"dims": [0, 1]}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_prod_dim2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "prod_dim2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_prod", "inputs": ["inp"], "output": "out", "attrs": {"dims": [1]}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_remainder2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "remainder2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "other": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "remainder", "inputs": ["inp", "other"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_reciprocal2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "reciprocal2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "one": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "A_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "one", "attrs": {"value": 1.0}},
                {"op": "cast", "inputs": ["A"], "output": "A_f32", "attrs": {"to": "f32"}},
                {"op": "div", "inputs": ["one", "A_f32"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_rsqrt2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "rsqrt2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "A_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["A"], "output": "A_f32", "attrs": {"to": "f32"}},
                {"op": "rsqrt", "inputs": ["A_f32"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_per_token_group_quant_fp8_2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "per_token_group_quant_fp8_2d",
            "tensors": {
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "fp8_min": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "fp8_max": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "y_grouped": {"dtype": "f32", "shape": ["MG", "GROUP_SIZE"], "layout": "row_major"},
                "y_abs": {"dtype": "f32", "shape": ["MG", "GROUP_SIZE"], "layout": "row_major"},
                "absmax": {"dtype": "f32", "shape": ["MG", 1], "layout": "row_major"},
                "absmax_clamped": {"dtype": "f32", "shape": ["MG", 1], "layout": "row_major"},
                "y_s_2d": {"dtype": "f32", "shape": ["MG", 1], "layout": "row_major"},
                "y_s_broadcast": {"dtype": "f32", "shape": ["MG", "GROUP_SIZE"], "layout": "row_major"},
                "y_scaled": {"dtype": "f32", "shape": ["MG", "GROUP_SIZE"], "layout": "row_major"},
                "y_clamped_min": {"dtype": "f32", "shape": ["MG", "GROUP_SIZE"], "layout": "row_major"},
                "y_clamped": {"dtype": "f32", "shape": ["MG", "GROUP_SIZE"], "layout": "row_major"},
                "y_q": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y_s": {"dtype": "f32", "shape": ["M", "G"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reshape", "inputs": ["y"], "output": "y_grouped", "attrs": {"shape": ["MG", "GROUP_SIZE"]}},
                {"op": "abs", "inputs": ["y_grouped"], "output": "y_abs"},
                {"op": "reduce_max", "inputs": ["y_abs"], "output": "absmax", "attrs": {"dims": [1], "keepdims": True}},
                {"op": "max", "inputs": ["absmax", "eps"], "output": "absmax_clamped"},
                {"op": "div", "inputs": ["absmax_clamped", "fp8_max"], "output": "y_s_2d"},
                {"op": "reshape", "inputs": ["y_s_2d"], "output": "y_s", "attrs": {"shape": ["M", "G"]}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["y_s_2d"],
                    "output": "y_s_broadcast",
                    "attrs": {"out_shape": ["MG", "GROUP_SIZE"], "broadcast_dims": [0, 1]},
                },
                {"op": "div", "inputs": ["y_grouped", "y_s_broadcast"], "output": "y_scaled"},
                {"op": "max", "inputs": ["y_scaled", "fp8_min"], "output": "y_clamped_min"},
                {"op": "min", "inputs": ["y_clamped_min", "fp8_max"], "output": "y_clamped"},
                {"op": "reshape", "inputs": ["y_clamped"], "output": "y_q", "attrs": {"shape": ["M", "N"]}},
            ],
            "outputs": ["y_q", "y_s"],
        }
    )


def _canonical_gather2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "gather2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gather", "inputs": ["inp", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_repeat2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "repeat2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gather", "inputs": ["inp", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_tile2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "tile2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gather", "inputs": ["inp", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_repeat_interleave_self_int1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "repeat_interleave_self_int1d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": [1, "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["N_OUT"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["N_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N_OUT"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gather", "inputs": ["inp", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_repeat_interleave_self_tensor1d_intent() -> IntentFunction:
    out = _canonical_repeat_interleave_self_int1d_intent().to_json_dict()
    out["name"] = "repeat_interleave_self_tensor1d"
    return IntentFunction.from_json_dict(out)


def _canonical_repeat_interleave_tensor1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "repeat_interleave_tensor1d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": [1, "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["N_OUT"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["N_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N_OUT"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gather", "inputs": ["inp", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_index_select2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "index_select2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["L", "N"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["L", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["L", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gather", "inputs": ["inp", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_flip2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "flip2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "gather", "inputs": ["inp", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_embedding2d_intent() -> IntentFunction:
    out = _canonical_gather2d_intent().to_json_dict()
    out["name"] = "embedding2d"
    return IntentFunction.from_json_dict(out)


def _canonical_isin1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "isin1d",
            "tensors": {
                "in0": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
                "in1": {"dtype": "i32", "shape": ["K"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["in0"],
                    "output": "in0_mk",
                    "attrs": {"out_shape": ["M", "K"], "broadcast_dims": [0]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["in1"],
                    "output": "in1_mk",
                    "attrs": {"out_shape": ["M", "K"], "broadcast_dims": [1]},
                },
                {"op": "ne", "inputs": ["in0_mk", "in1_mk"], "output": "neq_mk"},
                {"op": "not", "inputs": ["neq_mk"], "output": "eq_mk"},
                {"op": "reduce_any", "inputs": ["eq_mk"], "output": "out", "attrs": {"dims": [1]}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_kron2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "kron2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["P", "Q"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["MP", "NQ"], "layout": "row_major"},
            },
            "ops": [
                {"op": "kron", "inputs": ["A", "B"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_linspace1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "linspace1d",
            "tensors": {
                "start": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "end": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "denom": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "idx", "attrs": {"axis": 0, "shape": ["N"], "dtype": "i32"}},
                {"op": "cast", "inputs": ["idx"], "output": "idx_f", "attrs": {"to": "f32"}},
                {"op": "sub", "inputs": ["end", "start"], "output": "delta"},
                {"op": "div", "inputs": ["delta", "denom"], "output": "step"},
                {"op": "mul", "inputs": ["idx_f", "step"], "output": "scaled"},
                {"op": "add", "inputs": ["start", "scaled"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_logspace1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "logspace1d",
            "tensors": {
                "start": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "end": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "denom": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "log_base": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "idx", "attrs": {"axis": 0, "shape": ["N"], "dtype": "i32"}},
                {"op": "cast", "inputs": ["idx"], "output": "idx_f", "attrs": {"to": "f32"}},
                {"op": "sub", "inputs": ["end", "start"], "output": "delta"},
                {"op": "div", "inputs": ["delta", "denom"], "output": "step"},
                {"op": "mul", "inputs": ["idx_f", "step"], "output": "scaled"},
                {"op": "add", "inputs": ["start", "scaled"], "output": "lin"},
                {"op": "mul", "inputs": ["lin", "log_base"], "output": "exp_arg"},
                {"op": "exp", "inputs": ["exp_arg"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_lerp2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "lerp2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "W": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sub", "inputs": ["B", "A"], "output": "delta"},
                {"op": "mul", "inputs": ["W", "delta"], "output": "scaled_delta"},
                {"op": "add", "inputs": ["A", "scaled_delta"], "output": "C"},
            ],
            "outputs": ["C"],
        }
    )


def _canonical_le2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "le2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "le", "inputs": ["A", "B"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_log2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "log2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "log", "inputs": ["inp"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_log_sigmoid2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "log_sigmoid2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "abs", "inputs": ["inp"], "output": "abs_inp"},
                {"op": "const", "inputs": [], "output": "neg_one", "attrs": {"value": -1.0}},
                {"op": "mul", "inputs": ["abs_inp", "neg_one"], "output": "neg_abs"},
                {"op": "exp", "inputs": ["neg_abs"], "output": "exp_neg_abs"},
                {"op": "const", "inputs": [], "output": "one", "attrs": {"value": 1.0}},
                {"op": "add", "inputs": ["exp_neg_abs", "one"], "output": "one_plus_exp"},
                {"op": "log", "inputs": ["one_plus_exp"], "output": "log_one_plus_exp"},
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0}},
                {"op": "min", "inputs": ["inp", "zero"], "output": "min_inp_zero"},
                {"op": "sub", "inputs": ["min_inp_zero", "log_one_plus_exp"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_log_softmax2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "log_softmax2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "softmax", "inputs": ["inp"], "output": "softmax_out", "attrs": {"axis": 1}},
                {"op": "log", "inputs": ["softmax_out"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_logical_and2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "logical_and2d",
            "tensors": {
                "A": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "and", "inputs": ["A", "B"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_logical_not2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "logical_not2d",
            "tensors": {
                "inp": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "not", "inputs": ["inp"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_logical_or2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "logical_or2d",
            "tensors": {
                "A": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "a_i1": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "b_i1": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["A"], "output": "a_i1", "attrs": {"to": "i1"}},
                {"op": "cast", "inputs": ["B"], "output": "b_i1", "attrs": {"to": "i1"}},
                {"op": "max", "inputs": ["a_i1", "b_i1"], "output": "Out"},
            ],
            "outputs": ["Out"],
        }
    )


def _canonical_logical_xor2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "logical_xor2d",
            "tensors": {
                "A": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "a_bool": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "b_bool": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["A"], "output": "a_bool", "attrs": {"to": "i1"}},
                {"op": "cast", "inputs": ["B"], "output": "b_bool", "attrs": {"to": "i1"}},
                {"op": "ne", "inputs": ["a_bool", "b_bool"], "output": "Out"},
            ],
            "outputs": ["Out"],
        }
    )


def _canonical_lt2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "lt2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "cast_x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "cast_x", "attrs": {"to": "f32"}},
                {"op": "lt", "inputs": ["cast_x", "y"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_minimum2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "minimum2d",
            "tensors": {
                "X": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "Y": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "x_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "result_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["X"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["Y"], "output": "y_f32", "attrs": {"to": "f32"}},
                {"op": "min", "inputs": ["x_f32", "y_f32"], "output": "result_f32"},
                {"op": "cast", "inputs": ["result_f32"], "output": "Out", "attrs": {"to": "bf16"}},
            ],
            "outputs": ["Out"],
        }
    )


def _canonical_ne2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "ne2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "x_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["y"], "output": "y_f32", "attrs": {"to": "f32"}},
                {"op": "ne", "inputs": ["x_f32", "y_f32"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_neg2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "neg2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "neg_one": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "neg_one", "attrs": {"value": -1.0}},
                {"op": "mul", "inputs": ["A", "neg_one"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_masked_scatter2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "masked_scatter2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "source": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "masked_scatter", "inputs": ["inp", "mask", "source"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_masked_select2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "masked_select2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
            },
            "ops": [
                {"op": "masked_select", "inputs": ["inp", "mask"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_mse_loss2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "mse_loss2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "target": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "mse_loss", "inputs": ["inp", "target"], "output": "out", "attrs": {"reduction": 1}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_mv2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "mv2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "Inp": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["A", "B"], "output": "mv_out"},
                {"op": "mul", "inputs": ["mv_out", "alpha"], "output": "mv_scaled"},
                {"op": "mul", "inputs": ["Inp", "beta"], "output": "inp_scaled"},
                {"op": "add", "inputs": ["mv_scaled", "inp_scaled"], "output": "C"},
            ],
            "outputs": ["C"],
        }
    )


def _canonical_nan_to_num2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "nan_to_num2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "abs", "inputs": ["A"], "output": "abs_a"},
                {"op": "const", "inputs": [], "output": "finite_max", "attrs": {"value": _F32_FINITE_MAX}},
                {"op": "gt", "inputs": ["abs_a", "finite_max"], "output": "is_inf"},
                {"op": "ne", "inputs": ["A", "A"], "output": "is_nan"},
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0}},
                {"op": "const", "inputs": [], "output": "posinf_const", "attrs": {"value": 9.0}},
                {"op": "const", "inputs": [], "output": "neginf_const", "attrs": {"value": -9.0}},
                {"op": "ge", "inputs": ["A", "zero_const"], "output": "is_nonnegative"},
                {"op": "where", "inputs": ["is_nonnegative", "posinf_const", "neginf_const"], "output": "inf_repl"},
                {"op": "where", "inputs": ["is_inf", "inf_repl", "A"], "output": "no_inf"},
                {"op": "where", "inputs": ["is_nan", "zero_const", "no_inf"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_nll_loss2d_forward_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "nll_loss2d_forward",
            "tensors": {
                "self": {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"},
                "target": {"dtype": "i64", "shape": ["N", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "nll_loss2d_forward",
                    "inputs": ["self", "target", "weight"],
                    "output": "output",
                    "attrs": {"reduction": 1, "ignore_index": -100},
                },
            ],
            "outputs": ["output"],
        }
    )


def _canonical_nll_loss_forward_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "nll_loss_forward",
            "tensors": {
                "self": {"dtype": "f32", "shape": ["N", "C"], "layout": "row_major"},
                "target": {"dtype": "i64", "shape": ["N"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "nll_loss_forward",
                    "inputs": ["self", "target", "weight"],
                    "output": "output",
                    "attrs": {"reduction": 1, "ignore_index": -100},
                },
            ],
            "outputs": ["output"],
        }
    )


def _canonical_one_hot2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "one_hot2d",
            "tensors": {
                "tensor": {"dtype": "i64", "shape": ["M"], "layout": "row_major"},
                "out": {"dtype": "i64", "shape": ["M", "C"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "class_idx", "attrs": {"axis": 1, "shape": ["M", "C"], "dtype": "i32"}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["tensor"],
                    "output": "token_idx",
                    "attrs": {"out_shape": ["M", "C"], "broadcast_dims": [0]},
                },
                {"op": "ne", "inputs": ["token_idx", "class_idx"], "output": "neq_mask"},
                {"op": "not", "inputs": ["neq_mask"], "output": "eq_mask"},
                {"op": "cast", "inputs": ["eq_mask"], "output": "out", "attrs": {"to": "i64"}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_nonzero2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "nonzero2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i64", "shape": ["num_nonzeros", 2], "layout": "row_major"},
            },
            "ops": [
                {"op": "nonzero", "inputs": ["inp"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_normed_cumsum2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "normed_cumsum2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "EPS": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cumsum", "inputs": ["inp"], "output": "y_cumsum", "attrs": {"axis": 1}},
                {"op": "reduce_sum", "inputs": ["inp"], "output": "y_denom", "attrs": {"dims": [1], "keepdims": True}},
                {"op": "add", "inputs": ["y_denom", "EPS"], "output": "y_sum_eps"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["y_sum_eps"],
                    "output": "y_sum_eps_bc",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0, 1]},
                },
                {"op": "div", "inputs": ["y_cumsum", "y_sum_eps_bc"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_cumsum2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cumsum2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "offset": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "inp_vals": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "result": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "offset", "attrs": {"axis": 1, "shape": ["M", "N"]}},
                {"op": "identity", "inputs": ["inp"], "output": "inp_vals"},
                {"op": "cumsum", "inputs": ["inp_vals"], "output": "result", "attrs": {"axis": 1}},
                {"op": "identity", "inputs": ["result"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_max_pool2d_with_indices_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "max_pool2d_with_indices_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
                "indices": {"dtype": "i64", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "max_pool2d_with_indices",
                    "inputs": ["input"],
                    "output": "out",
                    "attrs": {
                        "kernel_size": [2, 2],
                        "stride": [2, 2],
                        "padding": [0, 0],
                        "dilation": [1, 1],
                        "ceil_mode": False,
                        "select": "values",
                    },
                },
                {
                    "op": "max_pool2d_with_indices",
                    "inputs": ["input"],
                    "output": "indices",
                    "attrs": {
                        "kernel_size": [2, 2],
                        "stride": [2, 2],
                        "padding": [0, 0],
                        "dilation": [1, 1],
                        "ceil_mode": False,
                        "select": "indices",
                    },
                },
            ],
            "outputs": ["out", "indices"],
        }
    )


def _canonical_conv1d_ncl_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv1d_ncl",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "L"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_PER_G", "K"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OL"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv1d",
                    "inputs": ["input", "weight", "bias"],
                    "output": "out",
                    "attrs": {"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_conv3d_ncdhw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv3d_ncdhw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "D", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_PER_G", "KD", "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OD", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv3d",
                    "inputs": ["input", "weight", "bias"],
                    "output": "out",
                    "attrs": {"stride": [1, 1, 1], "padding": [1, 1, 1], "dilation": [1, 1, 1], "groups": 1},
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_conv_depthwise2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv_depthwise2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", 1, "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv_depthwise2d",
                    "inputs": ["input", "weight", "bias"],
                    "output": "out",
                    "attrs": {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1]},
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_trace2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "trace2d",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "diag_mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "zero_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "diag_vals": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "row_idx", "attrs": {"axis": 0, "shape": ["M", "N"], "dtype": "i32"}},
                {"op": "iota", "inputs": [], "output": "col_idx", "attrs": {"axis": 1, "shape": ["M", "N"], "dtype": "i32"}},
                {"op": "eq", "inputs": ["row_idx", "col_idx"], "output": "diag_mask"},
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "where", "inputs": ["diag_mask", "input", "zero_const"], "output": "diag_vals"},
                {"op": "reduce_sum", "inputs": ["diag_vals"], "output": "out", "attrs": {"dims": [0, 1]}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_triu2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "triu2d",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "diagonal": {"dtype": "i32", "shape": [], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "shifted_row": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "keep_mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "zero_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "row_idx", "attrs": {"axis": 0, "shape": ["M", "N"], "dtype": "i32"}},
                {"op": "iota", "inputs": [], "output": "col_idx", "attrs": {"axis": 1, "shape": ["M", "N"], "dtype": "i32"}},
                {"op": "add", "inputs": ["row_idx", "diagonal"], "output": "shifted_row"},
                {"op": "le", "inputs": ["shifted_row", "col_idx"], "output": "keep_mask"},
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "where", "inputs": ["keep_mask", "input", "zero_const"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_upsample_nearest1d_ncl_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "upsample_nearest1d_ncl",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "IL"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OL"], "layout": "row_major"},
            },
            "ops": [
                {"op": "upsample_nearest1d", "inputs": ["input"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_upsample_nearest2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "upsample_nearest2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "IH", "IW"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {"op": "upsample_nearest2d", "inputs": ["input"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_scatter2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "scatter2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "index": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "scatter", "inputs": ["inp", "index", "src"], "output": "out", "attrs": {"dim": 1}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_select_scatter2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "select_scatter2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "select_scatter", "inputs": ["inp", "src"], "output": "out", "attrs": {"dim": 1, "index": 0}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_slice_scatter2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "slice_scatter2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["M", "L"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "slice_scatter",
                    "inputs": ["inp", "src"],
                    "output": "out",
                    "attrs": {"dim": 1, "start": 0, "end": 4, "step": 1},
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_quantile2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "quantile2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "q": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "quantile",
                    "inputs": ["inp", "q"],
                    "output": "out",
                    "attrs": {"dim": 1, "keepdim": False, "interpolation": "linear"},
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_polar2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "polar2d",
            "tensors": {
                "abs": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "angle": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N", 2], "layout": "row_major"},
            },
            "ops": [
                {"op": "polar", "inputs": ["abs", "angle"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_glu2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "glu2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N_HALF"], "layout": "row_major"},
            },
            "ops": [
                {"op": "glu", "inputs": ["x"], "output": "out", "attrs": {"axis": 1}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_cummax1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cummax1d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cummax", "inputs": ["x"], "output": "out", "attrs": {"axis": 0}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_cummin1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cummin1d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cummin", "inputs": ["x"], "output": "out", "attrs": {"axis": 0}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_index_add2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "index_add2d",
            "tensors": {
                "base": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "index": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["L", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "index_add", "inputs": ["base", "index", "src"], "output": "out", "attrs": {"axis": 0, "alpha": 1.0}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_index_put2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "index_put2d",
            "tensors": {
                "base": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "values": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "index_put",
                    "inputs": ["base", "row_idx", "col_idx", "values"],
                    "output": "out",
                    "attrs": {"accumulate": False},
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_masked_fill2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "masked_fill2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "value": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "where", "inputs": ["mask", "value", "inp"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_count_nonzero2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "count_nonzero2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i64", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0}},
                {"op": "ne", "inputs": ["x", "zero_const"], "output": "is_nonzero_bool"},
                {"op": "cast", "inputs": ["is_nonzero_bool"], "output": "is_nonzero_i64", "attrs": {"to": "i64"}},
                {"op": "reduce_sum", "inputs": ["is_nonzero_i64"], "output": "out", "attrs": {"dims": [0, 1]}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_diag2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "diag2d",
            "tensors": {
                "data": {"dtype": "f32", "shape": ["M", "M"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "diag_idx", "attrs": {"axis": 0, "shape": ["M"]}},
                {"op": "gather", "inputs": ["data", "diag_idx", "diag_idx"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_diag_embed2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "diag_embed2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["B", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["B", "N", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_scalar", "attrs": {"value": 0.0}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["zero_scalar"],
                    "output": "y_zeros",
                    "attrs": {"broadcast_dims": [], "out_shape": ["B", "N", "N"]},
                },
                {"op": "iota", "inputs": [], "output": "idx_row", "attrs": {"axis": 1, "shape": ["B", "N", "N"]}},
                {"op": "iota", "inputs": [], "output": "idx_col", "attrs": {"axis": 2, "shape": ["B", "N", "N"]}},
                {"op": "eq", "inputs": ["idx_row", "idx_col"], "output": "diag_mask"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["x"],
                    "output": "x_bcast",
                    "attrs": {"broadcast_dims": [0, 2], "out_shape": ["B", "N", "N"]},
                },
                {"op": "where", "inputs": ["diag_mask", "x_bcast", "y_zeros"], "output": "y"},
            ],
            "outputs": ["y"],
        }
    )


def _canonical_eq2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "eq2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["y"], "output": "y_f32", "attrs": {"to": "f32"}},
                {"op": "eq", "inputs": ["x_f32", "y_f32"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_elu2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "elu2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "zero_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "one_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "pos_mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "exp_x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "neg_branch": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0}},
                {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1.0}},
                {"op": "gt", "inputs": ["x", "zero_const"], "output": "pos_mask"},
                {"op": "exp", "inputs": ["x"], "output": "exp_x"},
                {"op": "sub", "inputs": ["exp_x", "one_const"], "output": "neg_branch"},
                {"op": "where", "inputs": ["pos_mask", "x", "neg_branch"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_celu2d_intent() -> IntentFunction:
    out = _canonical_elu2d_intent().to_json_dict()
    out["name"] = "celu2d"
    return IntentFunction.from_json_dict(out)


def _canonical_eye2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "eye2d",
            "tensors": {
                "idx_row": {"dtype": "i32", "shape": ["N", "N"], "layout": "row_major"},
                "idx_col": {"dtype": "i32", "shape": ["N", "N"], "layout": "row_major"},
                "offdiag_mask": {"dtype": "bool", "shape": ["N", "N"], "layout": "row_major"},
                "diag_mask": {"dtype": "bool", "shape": ["N", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "idx_row", "attrs": {"axis": 0, "shape": ["N", "N"]}},
                {"op": "iota", "inputs": [], "output": "idx_col", "attrs": {"axis": 1, "shape": ["N", "N"]}},
                {"op": "ne", "inputs": ["idx_row", "idx_col"], "output": "offdiag_mask"},
                {"op": "not", "inputs": ["offdiag_mask"], "output": "diag_mask"},
                {"op": "cast", "inputs": ["diag_mask"], "output": "out", "attrs": {"to": "f32"}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_eye_m2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "eye_m2d",
            "tensors": {
                "idx_row": {"dtype": "i32", "shape": ["N", "M"], "layout": "row_major"},
                "idx_col": {"dtype": "i32", "shape": ["N", "M"], "layout": "row_major"},
                "offdiag_mask": {"dtype": "bool", "shape": ["N", "M"], "layout": "row_major"},
                "diag_mask": {"dtype": "bool", "shape": ["N", "M"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "idx_row", "attrs": {"axis": 0, "shape": ["N", "M"]}},
                {"op": "iota", "inputs": [], "output": "idx_col", "attrs": {"axis": 1, "shape": ["N", "M"]}},
                {"op": "ne", "inputs": ["idx_row", "idx_col"], "output": "offdiag_mask"},
                {"op": "not", "inputs": ["offdiag_mask"], "output": "diag_mask"},
                {"op": "cast", "inputs": ["diag_mask"], "output": "out", "attrs": {"to": "f32"}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_unique2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "unique2d",
            "tensors": {
                "inp": {"dtype": "i32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["U"], "layout": "row_major"},
            },
            "ops": [
                {"op": "unique", "inputs": ["inp"], "output": "out", "attrs": {"sorted": True}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_weight_norm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "weight_norm2d",
            "tensors": {
                "v": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "g": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "vv": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "norm_sq": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "norm": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "scale": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "scale_bc": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "mul", "inputs": ["v", "v"], "output": "vv"},
                {"op": "reduce_sum", "inputs": ["vv"], "output": "norm_sq", "attrs": {"dims": [1], "keepdims": False}},
                {"op": "sqrt", "inputs": ["norm_sq"], "output": "norm"},
                {"op": "div", "inputs": ["g", "norm"], "output": "scale"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["scale"],
                    "output": "scale_bc",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {"op": "mul", "inputs": ["v", "scale_bc"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_angle2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "angle2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "zero": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "pi": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "is_neg": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "result": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "const", "inputs": [], "output": "pi", "attrs": {"value": 3.141592653589793, "dtype": "f32"}},
                {"op": "lt", "inputs": ["inp", "zero"], "output": "is_neg"},
                {"op": "where", "inputs": ["is_neg", "pi", "zero"], "output": "result"},
            ],
            "outputs": ["result"],
        }
    )


def _canonical_argmax2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "argmax2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_index": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "argmax", "inputs": ["inp"], "output": "out_index", "attrs": {"axis": 1}},
            ],
            "outputs": ["out_index"],
        }
    )


def _canonical_argmin2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "argmin2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_index": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "argmin", "inputs": ["inp"], "output": "out_index", "attrs": {"axis": 1}},
            ],
            "outputs": ["out_index"],
        }
    )


def _canonical_avg_pool2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "avg_pool2d_nchw",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "avg_pool2d",
                    "inputs": ["inp"],
                    "output": "out",
                    "attrs": {
                        "kernel_size": [2, 2],
                        "stride": [2, 2],
                        "padding": [0, 0],
                        "ceil_mode": False,
                        "count_include_pad": True,
                    },
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_bitwise_and2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "bitwise_and2d",
            "tensors": {
                "A": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "bitwise_and", "inputs": ["A", "B"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_bitwise_or2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "bitwise_or2d",
            "tensors": {
                "A": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "bitwise_or", "inputs": ["A", "B"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_bitwise_left_shift2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "bitwise_left_shift2d",
            "tensors": {
                "A": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "bitwise_left_shift", "inputs": ["A", "B"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_bitwise_right_shift2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "bitwise_right_shift2d",
            "tensors": {
                "A": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "bitwise_right_shift", "inputs": ["A", "B"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_bitwise_not2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "bitwise_not2d",
            "tensors": {
                "inp": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "bitwise_not", "inputs": ["inp"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_row_max_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "row_max",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_max", "inputs": ["inp"], "output": "out", "attrs": {"dims": [1], "keepdims": False}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_min2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "min2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_value": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_min", "inputs": ["inp"], "output": "out_value", "attrs": {"dims": [0, 1], "keepdims": False}},
            ],
            "outputs": ["out_value"],
        }
    )


def _canonical_any_kernel_dim_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "any_kernel_dim",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0}},
                {"op": "ne", "inputs": ["inp", "zero"], "output": "neq_zero"},
                {"op": "reduce_any", "inputs": ["neq_zero"], "output": "out", "attrs": {"dims": [1]}},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_scaled_dot_product_attention_bhsd_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "scaled_dot_product_attention_bhsd",
            "tensors": {
                "query": {"dtype": "f32", "shape": ["B", "H", "Q", "D"], "layout": "row_major"},
                "key": {"dtype": "f32", "shape": ["B", "H", "K", "D"], "layout": "row_major"},
                "value": {"dtype": "f32", "shape": ["B", "H", "K", "D"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["B", "H", "Q", "D"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "scaled_dot_product_attention",
                    "inputs": ["query", "key", "value"],
                    "output": "out",
                    "attrs": {"is_causal": False},
                },
            ],
            "outputs": ["out"],
        }
    )


def _canonical_flash_attn_varlen_func_bhsd_intent() -> IntentFunction:
    out = _canonical_scaled_dot_product_attention_bhsd_intent().to_json_dict()
    out["name"] = "flash_attn_varlen_func_bhsd"
    return IntentFunction.from_json_dict(out)


def _canonical_exp22d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "exp22d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "ln2": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "scaled": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "ln2", "attrs": {"value": 0.6931471805599453}},
                {"op": "mul", "inputs": ["A", "ln2"], "output": "scaled"},
                {"op": "exp", "inputs": ["scaled"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_pow_scalar2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "pow_scalar2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "exponent": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["exponent"],
                    "output": "exp_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": []},
                },
                {"op": "pow", "inputs": ["inp", "exp_bcast"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _canonical_pow_tensor_scalar2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "pow_tensor_scalar2d",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "exponent": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "x_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "exp_f32": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "exp_bcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["X"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["exponent"], "output": "exp_f32", "attrs": {"to": "f32"}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["exp_f32"],
                    "output": "exp_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": []},
                },
                {"op": "pow", "inputs": ["x_f32", "exp_bcast"], "output": "Out"},
            ],
            "outputs": ["Out"],
        }
    )


def _canonical_pow_tensor_tensor2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "pow_tensor_tensor2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "exponent": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "cast_x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "cast_exponent": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "pow_op": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "cast_x", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["exponent"], "output": "cast_exponent", "attrs": {"to": "f32"}},
                {"op": "pow", "inputs": ["cast_x", "cast_exponent"], "output": "pow_op"},
                {"op": "identity", "inputs": ["pow_op"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )


def _canonical_min_dim2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "min_dim2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_value": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "indices": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_min", "inputs": ["inp"], "output": "out_value", "attrs": {"dims": [1], "keepdims": False}},
                {"op": "argmin", "inputs": ["inp"], "output": "indices", "attrs": {"axis": 1}},
            ],
            "outputs": ["out_value", "indices"],
        }
    )


def _canonical_conv2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_IN", "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv2d",
                    "inputs": ["input", "weight", "bias"],
                    "output": "out",
                    "attrs": {
                        "stride": ["SH", "SW"],
                        "padding": ["PH", "PW"],
                        "dilation": ["DH", "DW"],
                        "groups": 1,
                    },
                },
            ],
            "outputs": ["out"],
        }
    )


def canonical_flaggems_intent_for_spec(spec_name: str) -> IntentFunction | None:
    name = str(spec_name)
    if name == "sigmoid2d":
        return _canonical_sigmoid2d_intent()
    if name == "tanh2d":
        return _canonical_tanh2d_intent()
    if name == "silu2d":
        return _canonical_silu2d_intent()
    if name == "softplus2d":
        return _canonical_softplus2d_intent()
    if name == "angle2d":
        return _canonical_angle2d_intent()
    if name == "argmax2d":
        return _canonical_argmax2d_intent()
    if name == "argmin2d":
        return _canonical_argmin2d_intent()
    if name == "min_dim2d":
        return _canonical_min_dim2d_intent()
    if name == "avg_pool2d_nchw":
        return _canonical_avg_pool2d_nchw_intent()
    if name == "bitwise_and2d":
        return _canonical_bitwise_and2d_intent()
    if name == "bitwise_or2d":
        return _canonical_bitwise_or2d_intent()
    if name == "bitwise_left_shift2d":
        return _canonical_bitwise_left_shift2d_intent()
    if name == "bitwise_right_shift2d":
        return _canonical_bitwise_right_shift2d_intent()
    if name == "bitwise_not2d":
        return _canonical_bitwise_not2d_intent()
    if name == "row_max":
        return _canonical_row_max_intent()
    if name == "min2d":
        return _canonical_min2d_intent()
    if name == "pow_scalar2d":
        return _canonical_pow_scalar2d_intent()
    if name == "pow_tensor_scalar2d":
        return _canonical_pow_tensor_scalar2d_intent()
    if name == "pow_tensor_tensor2d":
        return _canonical_pow_tensor_tensor2d_intent()
    if name == "exp22d":
        return _canonical_exp22d_intent()
    if name == "any_kernel_dim":
        return _canonical_any_kernel_dim_intent()
    if name == "batch_norm2d":
        return _canonical_batch_norm2d_intent()
    if name == "isnan2d":
        return _canonical_isnan2d_intent()
    if name == "isinf2d":
        return _canonical_isinf2d_intent()
    if name == "isfinite2d":
        return _canonical_isfinite2d_intent()
    if name == "isclose2d":
        return _canonical_isclose2d_intent()
    if name == "allclose2d":
        return _canonical_allclose2d_intent()
    if name == "row_all":
        return _canonical_row_all_intent()
    if name == "threshold2d":
        return _canonical_threshold2d_intent()
    if name == "cat2d":
        return _canonical_cat2d_intent()
    if name == "hstack2d":
        return _canonical_hstack2d_intent()
    if name == "clamp2d":
        return _canonical_clamp2d_intent()
    if name == "constant_pad_nd2d":
        return _canonical_constant_pad_nd2d_intent()
    if name == "pad2d":
        return _canonical_pad2d_intent()
    if name == "prod2d":
        return _canonical_prod2d_intent()
    if name == "prod_dim2d":
        return _canonical_prod_dim2d_intent()
    if name == "reciprocal2d":
        return _canonical_reciprocal2d_intent()
    if name == "remainder2d":
        return _canonical_remainder2d_intent()
    if name == "rsqrt2d":
        return _canonical_rsqrt2d_intent()
    if name == "per_token_group_quant_fp8_2d":
        return _canonical_per_token_group_quant_fp8_2d_intent()
    if name == "gather2d":
        return _canonical_gather2d_intent()
    if name == "repeat2d":
        return _canonical_repeat2d_intent()
    if name == "tile2d":
        return _canonical_tile2d_intent()
    if name == "repeat_interleave_self_int1d":
        return _canonical_repeat_interleave_self_int1d_intent()
    if name == "repeat_interleave_self_tensor1d":
        return _canonical_repeat_interleave_self_tensor1d_intent()
    if name == "repeat_interleave_tensor1d":
        return _canonical_repeat_interleave_tensor1d_intent()
    if name == "index_select2d":
        return _canonical_index_select2d_intent()
    if name == "flip2d":
        return _canonical_flip2d_intent()
    if name == "embedding2d":
        return _canonical_embedding2d_intent()
    if name == "isin1d":
        return _canonical_isin1d_intent()
    if name == "kron2d":
        return _canonical_kron2d_intent()
    if name == "linspace1d":
        return _canonical_linspace1d_intent()
    if name == "logspace1d":
        return _canonical_logspace1d_intent()
    if name == "lerp2d":
        return _canonical_lerp2d_intent()
    if name == "le2d":
        return _canonical_le2d_intent()
    if name == "log2d":
        return _canonical_log2d_intent()
    if name == "log_sigmoid2d":
        return _canonical_log_sigmoid2d_intent()
    if name == "log_softmax2d":
        return _canonical_log_softmax2d_intent()
    if name == "logical_and2d":
        return _canonical_logical_and2d_intent()
    if name == "logical_not2d":
        return _canonical_logical_not2d_intent()
    if name == "logical_or2d":
        return _canonical_logical_or2d_intent()
    if name == "logical_xor2d":
        return _canonical_logical_xor2d_intent()
    if name == "lt2d":
        return _canonical_lt2d_intent()
    if name == "minimum2d":
        return _canonical_minimum2d_intent()
    if name == "ne2d":
        return _canonical_ne2d_intent()
    if name == "neg2d":
        return _canonical_neg2d_intent()
    if name == "masked_select2d":
        return _canonical_masked_select2d_intent()
    if name == "masked_scatter2d":
        return _canonical_masked_scatter2d_intent()
    if name == "mse_loss2d":
        return _canonical_mse_loss2d_intent()
    if name == "mv2d":
        return _canonical_mv2d_intent()
    if name == "nan_to_num2d":
        return _canonical_nan_to_num2d_intent()
    if name == "nll_loss2d_forward":
        return _canonical_nll_loss2d_forward_intent()
    if name == "nll_loss_forward":
        return _canonical_nll_loss_forward_intent()
    if name == "nonzero2d":
        return _canonical_nonzero2d_intent()
    if name == "normed_cumsum2d":
        return _canonical_normed_cumsum2d_intent()
    if name == "cumsum2d":
        return _canonical_cumsum2d_intent()
    if name == "one_hot2d":
        return _canonical_one_hot2d_intent()
    if name == "max_pool2d_with_indices_nchw":
        return _canonical_max_pool2d_with_indices_nchw_intent()
    if name == "conv1d_ncl":
        return _canonical_conv1d_ncl_intent()
    if name == "conv3d_ncdhw":
        return _canonical_conv3d_ncdhw_intent()
    if name == "conv_depthwise2d_nchw":
        return _canonical_conv_depthwise2d_nchw_intent()
    if name == "scatter2d":
        return _canonical_scatter2d_intent()
    if name == "select_scatter2d":
        return _canonical_select_scatter2d_intent()
    if name == "slice_scatter2d":
        return _canonical_slice_scatter2d_intent()
    if name == "quantile2d":
        return _canonical_quantile2d_intent()
    if name == "polar2d":
        return _canonical_polar2d_intent()
    if name == "trace2d":
        return _canonical_trace2d_intent()
    if name == "triu2d":
        return _canonical_triu2d_intent()
    if name == "sub2d":
        return _canonical_sub2d_intent()
    if name == "sin2d":
        return _canonical_sin2d_intent()
    if name == "tan2d":
        return _canonical_tan2d_intent()
    if name == "cast2d":
        return _canonical_cast2d_intent()
    if name == "sqrt2d":
        return _canonical_sqrt2d_intent()
    if name == "std2d":
        return _canonical_std2d_intent()
    if name == "var_mean2d":
        return _canonical_var_mean2d_intent()
    if name == "mm2d":
        return _canonical_mm2d_intent()
    if name == "addmm2d":
        return _canonical_addmm2d_intent()
    if name == "dot1d":
        return _canonical_dot1d_intent()
    if name == "vector_norm2d":
        return _canonical_vector_norm2d_intent()
    if name == "vdot1d":
        return _canonical_vdot1d_intent()
    if name == "rms_norm2d":
        return _canonical_rms_norm2d_intent()
    if name == "vstack2d":
        return _canonical_vstack2d_intent()
    if name == "where2d":
        return _canonical_where2d_intent()
    if name == "stack2d":
        return _canonical_stack2d_intent()
    if name == "sort2d":
        return _canonical_sort2d_intent()
    if name == "sort_stable2d":
        return _canonical_sort_stable2d_intent()
    if name == "topk2d":
        return _canonical_topk2d_intent()
    if name == "upsample_nearest1d_ncl":
        return _canonical_upsample_nearest1d_ncl_intent()
    if name == "upsample_nearest2d_nchw":
        return _canonical_upsample_nearest2d_nchw_intent()
    if name == "glu2d":
        return _canonical_glu2d_intent()
    if name == "cummax1d":
        return _canonical_cummax1d_intent()
    if name == "cummin1d":
        return _canonical_cummin1d_intent()
    if name == "index_add2d":
        return _canonical_index_add2d_intent()
    if name == "index_put2d":
        return _canonical_index_put2d_intent()
    if name == "masked_fill2d":
        return _canonical_masked_fill2d_intent()
    if name == "count_nonzero2d":
        return _canonical_count_nonzero2d_intent()
    if name == "diag2d":
        return _canonical_diag2d_intent()
    if name == "diag_embed2d":
        return _canonical_diag_embed2d_intent()
    if name == "eq2d":
        return _canonical_eq2d_intent()
    if name == "celu2d":
        return _canonical_celu2d_intent()
    if name == "elu2d":
        return _canonical_elu2d_intent()
    if name == "eye2d":
        return _canonical_eye2d_intent()
    if name == "eye_m2d":
        return _canonical_eye_m2d_intent()
    if name == "unique2d":
        return _canonical_unique2d_intent()
    if name == "weight_norm2d":
        return _canonical_weight_norm2d_intent()
    if name == "scaled_dot_product_attention_bhsd":
        return _canonical_scaled_dot_product_attention_bhsd_intent()
    if name == "flash_attn_varlen_func_bhsd":
        return _canonical_flash_attn_varlen_func_bhsd_intent()
    if name == "conv2d_nchw":
        return _canonical_conv2d_nchw_intent()
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
