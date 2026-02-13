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
                {"op": "iota", "inputs": [], "output": "idx_b", "attrs": {"axis": 0, "shape": ["B", "N", "N"]}},
                {"op": "iota", "inputs": [], "output": "idx_row", "attrs": {"axis": 1, "shape": ["B", "N", "N"]}},
                {"op": "iota", "inputs": [], "output": "idx_col", "attrs": {"axis": 2, "shape": ["B", "N", "N"]}},
                {"op": "ne", "inputs": ["idx_row", "idx_col"], "output": "offdiag_mask"},
                {"op": "not", "inputs": ["offdiag_mask"], "output": "diag_mask"},
                {"op": "gather", "inputs": ["x", "idx_b", "idx_col"], "output": "diag_values"},
                {"op": "where", "inputs": ["diag_mask", "diag_values", "y_zeros"], "output": "y"},
            ],
            "outputs": ["y"],
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


def canonical_flaggems_intent_for_spec(spec_name: str) -> IntentFunction | None:
    name = str(spec_name)
    if name == "sigmoid2d":
        return _canonical_sigmoid2d_intent()
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
    if name == "threshold2d":
        return _canonical_threshold2d_intent()
    if name == "gather2d":
        return _canonical_gather2d_intent()
    if name == "index_select2d":
        return _canonical_index_select2d_intent()
    if name == "flip2d":
        return _canonical_flip2d_intent()
    if name == "embedding2d":
        return _canonical_embedding2d_intent()
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
    if name == "celu2d":
        return _canonical_celu2d_intent()
    if name == "elu2d":
        return _canonical_elu2d_intent()
    if name == "eye2d":
        return _canonical_eye2d_intent()
    if name == "eye_m2d":
        return _canonical_eye_m2d_intent()
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
