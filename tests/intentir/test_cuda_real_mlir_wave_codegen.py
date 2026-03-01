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
