from __future__ import annotations

import subprocess

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, to_mlir
from intent_ir.mlir.passes.lower_intent_to_cuda_gpu_kernel import lower_intent_to_cuda_gpu_kernel


def _verify_with_mlir_opt(module_text: str) -> None:
    toolchain = detect_mlir_toolchain()
    tools = toolchain.get("tools") if isinstance(toolchain.get("tools"), dict) else {}
    mlir_opt = tools.get("mlir-opt") if isinstance(tools.get("mlir-opt"), dict) else {}
    if not bool(mlir_opt.get("available")):
        pytest.skip("mlir-opt unavailable; cannot verify emitted MLIR")
    mlir_opt_path = str(mlir_opt.get("path") or "").strip()
    if not mlir_opt_path:
        pytest.skip("mlir-opt path missing; cannot verify emitted MLIR")
    proc = subprocess.run(
        [mlir_opt_path, "--verify-each"],
        input=str(module_text),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def _argmax2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "argmax2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_index": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "argmax", "inputs": ["inp"], "output": "out_index", "attrs": {"axis": 1}}],
            "outputs": ["out_index"],
        }
    )


def _argmin2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "argmin2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_index": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "argmin", "inputs": ["inp"], "output": "out_index", "attrs": {"axis": 1}}],
            "outputs": ["out_index"],
        }
    )


def _prod_dim2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "prod_dim2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "reduce_prod", "inputs": ["inp"], "output": "out", "attrs": {"dims": [1]}}],
            "outputs": ["out"],
        }
    )


def _min2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "min2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_value": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [{"op": "reduce_min", "inputs": ["inp"], "output": "out_value", "attrs": {"dims": [0, 1]}}],
            "outputs": ["out_value"],
        }
    )


def _min_dim2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "min_dim2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_value": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "indices": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_min", "inputs": ["inp"], "output": "out_value", "attrs": {"dims": [1]}},
                {"op": "argmin", "inputs": ["inp"], "output": "indices", "attrs": {"axis": 1}},
            ],
            "outputs": ["out_value", "indices"],
        }
    )


def _count_nonzero2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "count_nonzero2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "zero_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "is_nonzero_bool": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "is_nonzero_i64": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i64", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "ne", "inputs": ["x", "zero_const"], "output": "is_nonzero_bool"},
                {"op": "cast", "inputs": ["is_nonzero_bool"], "output": "is_nonzero_i64", "attrs": {"to": "i64"}},
                {"op": "reduce_sum", "inputs": ["is_nonzero_i64"], "output": "out", "attrs": {"dims": [0, 1]}},
            ],
            "outputs": ["out"],
        }
    )


def _trace2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "trace2d",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "diag_mask": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
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


def _allclose2d_intent() -> IntentFunction:
    # allclose2d output is modeled as scalar i8 (0/1) in FlagGems.
    return IntentFunction.from_json_dict(
        {
            "name": "allclose2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "rtol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "atol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "diff": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "abs_diff": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "abs_b": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "rtol_abs_b": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "tol": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "close": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "not_close": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "any_not_close": {"dtype": "i1", "shape": [], "layout": "row_major"},
                "all_close": {"dtype": "i1", "shape": [], "layout": "row_major"},
                "output": {"dtype": "i8", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "sub", "inputs": ["A", "B"], "output": "diff"},
                {"op": "abs", "inputs": ["diff"], "output": "abs_diff"},
                {"op": "abs", "inputs": ["B"], "output": "abs_b"},
                {"op": "mul", "inputs": ["abs_b", "rtol"], "output": "rtol_abs_b"},
                {"op": "add", "inputs": ["rtol_abs_b", "atol"], "output": "tol"},
                {"op": "le", "inputs": ["abs_diff", "tol"], "output": "close"},
                {"op": "not", "inputs": ["close"], "output": "not_close"},
                {"op": "reduce_any", "inputs": ["not_close"], "output": "any_not_close", "attrs": {"dims": [0, 1]}},
                {"op": "not", "inputs": ["any_not_close"], "output": "all_close"},
                {"op": "cast", "inputs": ["all_close"], "output": "output", "attrs": {"to": "i8"}},
            ],
            "outputs": ["output"],
        }
    )


def _prod2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "prod2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [{"op": "reduce_prod", "inputs": ["inp"], "output": "out", "attrs": {"dims": [0, 1]}}],
            "outputs": ["out"],
        }
    )


def _stack2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "stack2d",
            "tensors": {
                "input0": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "input1": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": [2, "M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "stack", "inputs": ["input0", "input1"], "output": "out_ptr", "attrs": {"axis": 0}}],
            "outputs": ["out_ptr"],
        }
    )


def _polar2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "polar2d",
            "tensors": {
                "abs": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "angle": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N", 2], "layout": "row_major"},
            },
            "ops": [{"op": "polar", "inputs": ["abs", "angle"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _diag_embed2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "diag_embed2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["B", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["B", "N", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["x"],
                    "output": "y",
                    "attrs": {"broadcast_dims": [0, 2], "out_shape": ["B", "N", "N"]},
                }
            ],
            "outputs": ["y"],
        }
    )


def _upsample_nearest1d_ncl_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "upsample_nearest1d_ncl",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "IL"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OL"], "layout": "row_major"},
            },
            "ops": [{"op": "upsample_nearest1d", "inputs": ["input"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _upsample_nearest2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "upsample_nearest2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "IH", "IW"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [{"op": "upsample_nearest2d", "inputs": ["input"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _glu2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "glu2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N_HALF"], "layout": "row_major"},
            },
            "ops": [{"op": "glu", "inputs": ["x"], "output": "out", "attrs": {"axis": 1}}],
            "outputs": ["out"],
        }
    )


def _log_softmax2d_intent() -> IntentFunction:
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


def _weight_norm2d_intent() -> IntentFunction:
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


def _mse_loss2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "mse_loss2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "target": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [{"op": "mse_loss", "inputs": ["inp", "target"], "output": "out", "attrs": {"reduction": 1}}],
            "outputs": ["out"],
        }
    )


def _std2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "std2d",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "std",
                    "inputs": ["X"],
                    "output": "Out",
                    "attrs": {"axis": 1, "dims": [1], "keepdims": False, "correction": 1},
                }
            ],
            "outputs": ["Out"],
        }
    )


def _var_mean2d_intent() -> IntentFunction:
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


def _bmm3d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "bmm3d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["BATCH", "M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["BATCH", "K", "N"], "layout": "row_major"},
                "O": {"dtype": "f32", "shape": ["BATCH", "M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "O"}],
            "outputs": ["O"],
        }
    )


def _baddbmm3d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "baddbmm3d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["BATCH", "M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["BATCH", "K", "N"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["BATCH", "M", "N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "matmul_out": {"dtype": "f32", "shape": ["BATCH", "M", "N"], "layout": "row_major"},
                "scaled_matmul": {"dtype": "f32", "shape": ["BATCH", "M", "N"], "layout": "row_major"},
                "scaled_bias": {"dtype": "f32", "shape": ["BATCH", "M", "N"], "layout": "row_major"},
                "O": {"dtype": "f32", "shape": ["BATCH", "M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["A", "B"], "output": "matmul_out"},
                {"op": "mul", "inputs": ["matmul_out", "alpha"], "output": "scaled_matmul"},
                {"op": "mul", "inputs": ["bias", "beta"], "output": "scaled_bias"},
                {"op": "add", "inputs": ["scaled_matmul", "scaled_bias"], "output": "O"},
            ],
            "outputs": ["O"],
        }
    )


def _kron2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "kron2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["P", "Q"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["MP", "NQ"], "layout": "row_major"},
            },
            "ops": [{"op": "kron", "inputs": ["A", "B"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _cumsum2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cumsum2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "offset": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "inp_vals": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "result": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "offset", "attrs": {"axis": 1, "shape": ["M", "N"], "dtype": "i32"}},
                {"op": "identity", "inputs": ["inp"], "output": "inp_vals"},
                {"op": "cumsum", "inputs": ["inp_vals"], "output": "result", "attrs": {"axis": 1}},
                {"op": "identity", "inputs": ["result"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _normed_cumsum2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "normed_cumsum2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "EPS": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "y_cumsum": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y_denom": {"dtype": "f32", "shape": ["M", 1], "layout": "row_major"},
                "y_sum_eps": {"dtype": "f32", "shape": ["M", 1], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cumsum", "inputs": ["inp"], "output": "y_cumsum", "attrs": {"axis": 1}},
                {"op": "reduce_sum", "inputs": ["inp"], "output": "y_denom", "attrs": {"dims": [1], "keepdims": True}},
                {"op": "add", "inputs": ["y_denom", "EPS"], "output": "y_sum_eps"},
                {"op": "div", "inputs": ["y_cumsum", "y_sum_eps"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _cummax1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cummax1d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [{"op": "cummax", "inputs": ["x"], "output": "out", "attrs": {"axis": 0}}],
            "outputs": ["out"],
        }
    )


def _cummin1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cummin1d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [{"op": "cummin", "inputs": ["x"], "output": "out", "attrs": {"axis": 0}}],
            "outputs": ["out"],
        }
    )


def _sort2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sort2d",
            "tensors": {
                "in_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "sort",
                    "inputs": ["in_ptr"],
                    "output": "out_ptr",
                    "attrs": {"axis": 1, "descending": False, "stable": False},
                }
            ],
            "outputs": ["out_ptr"],
        }
    )


def _sort_stable2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "sort_stable2d",
            "tensors": {
                "in_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "sort",
                    "inputs": ["in_ptr"],
                    "output": "out_ptr",
                    "attrs": {"axis": 1, "descending": False, "stable": True},
                }
            ],
            "outputs": ["out_ptr"],
        }
    )


def _topk2d_intent() -> IntentFunction:
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
                {"op": "sort", "inputs": ["inp"], "output": "sorted_vals", "attrs": {"axis": 1, "descending": True}},
                {"op": "iota", "inputs": [], "output": "row_idx", "attrs": {"axis": 0, "shape": ["M", "K"], "dtype": "i32"}},
                {"op": "iota", "inputs": [], "output": "col_idx", "attrs": {"axis": 1, "shape": ["M", "K"], "dtype": "i32"}},
                {"op": "gather", "inputs": ["sorted_vals", "row_idx", "col_idx"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


def _quantile2d_intent() -> IntentFunction:
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
                }
            ],
            "outputs": ["out"],
        }
    )


def _avg_pool2d_nchw_intent() -> IntentFunction:
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
                    "attrs": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": False},
                }
            ],
            "outputs": ["out"],
        }
    )


def _max_pool2d_with_indices_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "max_pool2d_with_indices_nchw",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
                "indices": {"dtype": "i64", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "max_pool2d_with_indices",
                    "inputs": ["inp"],
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
                    "inputs": ["inp"],
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


def _index_add2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "index_add2d",
            "tensors": {
                "base": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "index": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["L", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "index_add", "inputs": ["base", "index", "src"], "output": "out", "attrs": {"axis": 0, "alpha": 1.0}}],
            "outputs": ["out"],
        }
    )


def _index_put2d_intent() -> IntentFunction:
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
            "ops": [{"op": "index_put", "inputs": ["base", "row_idx", "col_idx", "values"], "output": "out", "attrs": {"accumulate": False}}],
            "outputs": ["out"],
        }
    )


def _scatter2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "scatter2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "index": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "scatter", "inputs": ["inp", "index", "src"], "output": "out", "attrs": {"dim": 1}}],
            "outputs": ["out"],
        }
    )


def _select_scatter2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "select_scatter2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "select_scatter", "inputs": ["inp", "src"], "output": "out", "attrs": {"dim": 1, "index": 0}}],
            "outputs": ["out"],
        }
    )


def _slice_scatter2d_intent() -> IntentFunction:
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
                }
            ],
            "outputs": ["out"],
        }
    )


def _masked_select2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "masked_select2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
            },
            "ops": [{"op": "masked_select", "inputs": ["inp", "mask"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _masked_scatter2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "masked_scatter2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "source": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "masked_scatter", "inputs": ["inp", "mask", "source"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _nll_loss_forward_intent() -> IntentFunction:
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
                }
            ],
            "outputs": ["output"],
        }
    )


def _nll_loss2d_forward_intent() -> IntentFunction:
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
                }
            ],
            "outputs": ["output"],
        }
    )


def _per_token_group_quant_fp8_2d_intent() -> IntentFunction:
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


def _batch_norm2d_intent() -> IntentFunction:
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
                "const_one": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "mean_sum": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "mean_bcast": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "centered": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "centered_sq": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "var_sum": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "var": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "var_eps": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "inv_std_bcast": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "normalized": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "weight_bcast": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "scaled": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "bias_bcast": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "one_minus_momentum": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "running_mean_keep": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_mean_delta": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "bessel": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "unbiased_var": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_var_keep": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_var_delta": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
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
        }
    )


def _scaled_dot_product_attention_bhsd_intent() -> IntentFunction:
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
                }
            ],
            "outputs": ["out"],
        }
    )


def _flash_attn_varlen_func_bhsd_intent() -> IntentFunction:
    out = _scaled_dot_product_attention_bhsd_intent().to_json_dict()
    out["name"] = "flash_attn_varlen_func_bhsd"
    return IntentFunction.from_json_dict(out)


def _conv1d_ncl_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv1d_ncl",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "L"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_PER_G", "K"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OL"], "layout": "row_major"},
            },
            "ops": [{"op": "conv1d", "inputs": ["input", "weight", "bias"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _conv2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_PER_G", "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [{"op": "conv2d", "inputs": ["input", "weight", "bias"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _conv3d_ncdhw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv3d_ncdhw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "D", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_PER_G", "KD", "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OD", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [{"op": "conv3d", "inputs": ["input", "weight", "bias"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _conv_depthwise2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "conv_depthwise2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", 1, "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [{"op": "conv_depthwise2d", "inputs": ["input", "weight", "bias"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _unique2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "unique2d",
            "tensors": {
                "inp": {"dtype": "i32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["U"], "layout": "row_major"},
            },
            "ops": [{"op": "unique", "inputs": ["inp"], "output": "out", "attrs": {"sorted": True}}],
            "outputs": ["out"],
        }
    )


def _nonzero2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "nonzero2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i64", "shape": ["num_nonzeros", 2], "layout": "row_major"},
            },
            "ops": [{"op": "nonzero", "inputs": ["inp"], "output": "out"}],
            "outputs": ["out"],
        }
    )


@pytest.mark.parametrize(
    "intent_fn,shape_bindings,expected_kind,needle",
    [
        # wave12
        (_argmax2d_intent, {"M": 4, "N": 64}, "row_argmax_axis1_v1", "arith.cmpf"),
        (_argmin2d_intent, {"M": 4, "N": 64}, "row_argmin_axis1_v1", "arith.cmpf"),
        (_prod_dim2d_intent, {"M": 4, "N": 64}, "row_reduce_prod_axis1_v1", "arith.mulf"),
        (_min_dim2d_intent, {"M": 4, "N": 64}, "row_reduce_min_argmin_axis1_v1", "memref.get_global"),
        (_min2d_intent, {"M": 4, "N": 64}, "reduce_min_all_v1", "arith.min"),
        (_trace2d_intent, {"M": 16, "N": 16}, "trace2d_v1", "gpu.thread_id x"),
        (_count_nonzero2d_intent, {"M": 4, "N": 64}, "count_nonzero2d_v1", "arith.addi"),
        (_allclose2d_intent, {"M": 4, "N": 64}, "allclose2d_v1", "math.absf"),
        # wave13
        (_prod2d_intent, {"M": 4, "N": 64}, "reduce_prod_all_v1", "cS_prodall_128"),
        (_stack2d_intent, {"M": 4, "N": 16}, "stack2d_v1", "arith.divui %lin, %cPlane"),
        (_polar2d_intent, {"M": 8, "N": 16}, "polar2d_v1", "math.cos"),
        (_diag_embed2d_intent, {"B": 2, "N": 8}, "diag_embed2d_v1", "arith.cmpi eq, %ii, %jj"),
        (_upsample_nearest1d_ncl_intent, {"N": 2, "C": 3, "IL": 8, "OL": 16}, "upsample_nearest1d_ncl_v1", "%ol_mul"),
        (_upsample_nearest2d_nchw_intent, {"N": 1, "C": 2, "IH": 8, "IW": 8, "OH": 16, "OW": 16}, "upsample_nearest2d_nchw_v1", "%cIHW"),
        (_glu2d_intent, {"M": 4, "N": 64, "N_HALF": 32}, "glu2d_v1", "arith.divf %c1f, %den"),
        (_log_softmax2d_intent, {"M": 4, "N": 64}, "row_log_softmax_axis1_v1", "math.log"),
        # wave14
        (_weight_norm2d_intent, {"M": 4, "N": 64}, "weight_norm2d_v1", "math.sqrt %sumsq"),
        (_mse_loss2d_intent, {"M": 4, "N": 64}, "mse_loss2d_v1", "arith.mulf %d, %d"),
        (_std2d_intent, {"M": 4, "N": 64}, "row_std_axis1_v1", "memref.store %outv"),
        (_var_mean2d_intent, {"M": 4, "N": 64}, "row_var_axis1_v1", "memref.store %var"),
        # wave15
        (_bmm3d_intent, {"BATCH": 2, "M": 8, "K": 16, "N": 8}, "bmm_tile_v2", "batch_m = arith.muli %bid_b"),
        (_baddbmm3d_intent, {"BATCH": 2, "M": 8, "K": 16, "N": 8}, "baddbmm_tile_v2", "batch_m = arith.muli %bid_b"),
        # wave16
        (_kron2d_intent, {"M": 4, "N": 8, "P": 2, "Q": 3, "MP": 8, "NQ": 24}, "kron2d_v1", "%out_row = arith.divui %lin, %cNQ"),
        (_cumsum2d_intent, {"M": 4, "N": 64}, "cumsum2d_axis1_v1", "gpu.shuffle up"),
        (_normed_cumsum2d_intent, {"M": 4, "N": 64}, "normed_cumsum2d_axis1_v1", "arith.divf %full_scan, %denom"),
        (_cummax1d_intent, {"N": 64}, "cummax1d_axis0_v1", "arith.maximumf %acc, %xv"),
        (_cummin1d_intent, {"N": 64}, "cummin1d_axis0_v1", "arith.minimumf %acc, %xv"),
        # wave17
        (_sort2d_intent, {"M": 4, "N": 64}, "row_sort_axis1_bitonic_v1", "arith.xori %tid_i32"),
        (_sort_stable2d_intent, {"M": 4, "N": 64}, "row_sort_axis1_bitonic_stable_v1", "memref<512xf32, 3>"),
        (_topk2d_intent, {"M": 4, "N": 64, "K": 8}, "row_topk_axis1_bitonic_v1", "0xFF800000"),
        (_quantile2d_intent, {"M": 8, "N": 32}, "row_quantile_axis1_sort_v1", "arith.maximumf"),
        # wave18
        (_avg_pool2d_nchw_intent, {"N": 1, "C": 3, "H": 8, "W": 8, "OH": 4, "OW": 4}, "avg_pool2d_nchw_v1", "0.25 : f32"),
        (
            _max_pool2d_with_indices_nchw_intent,
            {"N": 1, "C": 1, "H": 8, "W": 8, "OH": 4, "OW": 4},
            "max_pool2d_with_indices_nchw_v1",
            "best_i3_i64",
        ),
        # wave19
        (_index_add2d_intent, {"M": 4, "N": 64, "L": 4}, "index_add2d_axis0_v1", "iter_args(%a = %base_v)"),
        (_index_put2d_intent, {"M": 4, "N": 64, "L": 8}, "index_put2d_v1", "%match_row"),
        (_scatter2d_intent, {"M": 4, "N": 64}, "scatter2d_dim1_v1", "%dst_i32 = memref.load"),
        (_select_scatter2d_intent, {"M": 4, "N": 64}, "select_scatter2d_dim1_v1", "memref.load %src[%bid]"),
        (_slice_scatter2d_intent, {"M": 4, "N": 64, "L": 4}, "slice_scatter2d_dim1_v1", "%dst_col = arith.addi"),
        # wave20
        (_masked_select2d_intent, {"M": 4, "N": 64, "L": 128}, "masked_select2d_prefixsum_v1", "scan_i32"),
        (_masked_scatter2d_intent, {"M": 4, "N": 64, "L": 128}, "masked_scatter2d_prefixsum_v1", "scan_i32"),
        # wave21
        (_nll_loss_forward_intent, {"N": 16, "C": 8}, "nll_loss_forward_v1", "arith.constant -100 : i64"),
        (_nll_loss2d_forward_intent, {"N": 2, "C": 4, "H": 4, "W": 4}, "nll_loss2d_forward_v1", "arith.constant -100 : i64"),
        # wave22
        (
            _per_token_group_quant_fp8_2d_intent,
            {"M": 4, "N": 64, "G": 4, "GROUP_SIZE": 16},
            "per_token_group_quant_fp8_2d_v1",
            "memref.store %scale, %y_s[%bid]",
        ),
        (_batch_norm2d_intent, {"N": 2, "C": 4, "HW": 4}, "batch_norm2d_v1", "memref.store %inv_std_v, %inv_std[%bid]"),
        # wave23
        (
            _scaled_dot_product_attention_bhsd_intent,
            {"B": 1, "H": 2, "Q": 8, "K": 8, "D": 16, "IS_CAUSAL": 0},
            "sdpa_bhsd_v1",
            "gpu.func @scaled_dot_product_attention_bhsd",
        ),
        (
            _flash_attn_varlen_func_bhsd_intent,
            {"B": 1, "H": 2, "Q": 8, "K": 8, "D": 16, "IS_CAUSAL": 0},
            "sdpa_bhsd_v1",
            "gpu.func @flash_attn_varlen_func_bhsd",
        ),
        # wave24
        (
            _conv1d_ncl_intent,
            {
                "N": 1,
                "C_IN": 4,
                "C_OUT": 4,
                "C_PER_G": 2,
                "L": 8,
                "K": 3,
                "OL": 8,
                "STRIDE": 1,
                "PADDING": 1,
                "DILATION": 1,
                "GROUPS": 2,
            },
            "conv1d_ncl_v1",
            "scf.for %ic = %c0 to %cC_PER_G",
        ),
        (
            _conv2d_nchw_intent,
            {
                "N": 1,
                "C_IN": 4,
                "C_OUT": 4,
                "C_PER_G": 2,
                "H": 8,
                "W": 8,
                "KH": 3,
                "KW": 3,
                "OH": 8,
                "OW": 8,
                "SH": 1,
                "SW": 1,
                "PH": 1,
                "PW": 1,
                "DH": 1,
                "DW": 1,
                "GROUPS": 2,
            },
            "conv2d_nchw_v1",
            "scf.for %kh_i = %c0 to %cKH",
        ),
        (
            _conv3d_ncdhw_intent,
            {
                "N": 1,
                "C_IN": 4,
                "C_OUT": 4,
                "C_PER_G": 2,
                "D": 8,
                "H": 8,
                "W": 8,
                "KD": 3,
                "KH": 3,
                "KW": 3,
                "OD": 8,
                "OH": 8,
                "OW": 8,
                "SD": 1,
                "SH": 1,
                "SW": 1,
                "PD": 1,
                "PH": 1,
                "PW": 1,
                "DD": 1,
                "DH": 1,
                "DW": 1,
                "GROUPS": 2,
            },
            "conv3d_ncdhw_v1",
            "scf.for %kd_i = %c0 to %cKD",
        ),
        (
            _conv_depthwise2d_nchw_intent,
            {
                "N": 1,
                "C_IN": 4,
                "C_OUT": 8,
                "H": 8,
                "W": 8,
                "KH": 3,
                "KW": 3,
                "OH": 8,
                "OW": 8,
                "SH": 1,
                "SW": 1,
                "PH": 1,
                "PW": 1,
                "DH": 1,
                "DW": 1,
                "MULT": 2,
            },
            "conv_depthwise2d_nchw_v1",
            "arith.divui %oc, %cMULT",
        ),
        # wave25
        (
            _unique2d_intent,
            {"N": 16, "U": 8},
            "unique2d_v1",
            "arith.cmpi sgt",
        ),
        (
            _nonzero2d_intent,
            {"M": 4, "N": 64, "num_nonzeros": 128},
            "nonzero2d_v1",
            "arith.cmpf one",
        ),
    ],
)
def test_cuda_real_mlir_wave_codegen_and_is_parseable(
    monkeypatch: pytest.MonkeyPatch,
    intent_fn,
    shape_bindings: dict[str, int],
    expected_kind: str,
    needle: str,
) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = intent_fn()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = dict(shape_bindings)
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == expected_kind
    assert str(needle) in out.module_text
    _verify_with_mlir_opt(out.module_text)
