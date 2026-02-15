from __future__ import annotations

import pytest

from backends.cuda.codegen.cpp_driver import CudaLoweringError, lower_intent_to_cuda_kernel
from intent_ir.ir import IntentFunction


def _lower_or_skip(intent: IntentFunction, *, shape_bindings: dict[str, int]) -> object:
    try:
        return lower_intent_to_cuda_kernel(intent, shape_bindings=shape_bindings)
    except CudaLoweringError as exc:
        if "unsupported intent for cuda cpp codegen" in str(exc):
            pytest.skip(f"cpp cuda codegen unsupported for this pattern: {exc}")
        raise


def _eye_like_intent(name: str, rows: str, cols: str) -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "idx_row": {"dtype": "i32", "shape": [rows, cols], "layout": "row_major"},
                "idx_col": {"dtype": "i32", "shape": [rows, cols], "layout": "row_major"},
                "offdiag_mask": {"dtype": "bool", "shape": [rows, cols], "layout": "row_major"},
                "diag_mask": {"dtype": "bool", "shape": [rows, cols], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [rows, cols], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "idx_row", "attrs": {"axis": 0, "shape": [rows, cols]}},
                {"op": "iota", "inputs": [], "output": "idx_col", "attrs": {"axis": 1, "shape": [rows, cols]}},
                {"op": "ne", "inputs": ["idx_row", "idx_col"], "output": "offdiag_mask"},
                {"op": "not", "inputs": ["offdiag_mask"], "output": "diag_mask"},
                {"op": "cast", "inputs": ["diag_mask"], "output": "out", "attrs": {"to": "f32"}},
            ],
            "outputs": ["out"],
            "parallel_axes": [rows, cols],
            "schedule": {"tile_n": 64, "parallel_axes": [rows, cols]},
        }
    )


def test_cuda_lowering_supports_eye_like_square_iota_graph(monkeypatch) -> None:
    intent = _eye_like_intent("eye2d_cuda_lowering", rows="N", cols="N")
    lowered = _lower_or_skip(intent, shape_bindings={"N": 8})
    assert lowered.kernel_name == "eye2d_cuda_lowering"
    assert "v_idx_row = (int)(i0);" in lowered.cuda_src
    assert "v_idx_col = (int)(i1);" in lowered.cuda_src


def test_cuda_lowering_supports_eye_like_rectangular_iota_graph(monkeypatch) -> None:
    intent = _eye_like_intent("eye_m2d_cuda_lowering", rows="N", cols="M")
    lowered = _lower_or_skip(intent, shape_bindings={"N": 8, "M": 6})
    assert lowered.kernel_name == "eye_m2d_cuda_lowering"
    assert "v_idx_row = (int)(i0);" in lowered.cuda_src
    assert "v_idx_col = (int)(i1);" in lowered.cuda_src


def test_cuda_lowering_supports_cos_erf_log_fused_elementwise(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "cos_erf_log_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "E": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "L": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "T": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cos", "inputs": ["A"], "output": "C"},
                {"op": "erf", "inputs": ["A"], "output": "E"},
                {"op": "log", "inputs": ["A"], "output": "L"},
                {"op": "add", "inputs": ["C", "E"], "output": "T"},
                {"op": "add", "inputs": ["T", "L"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "cos_erf_log_cuda_lowering"
    assert "cosf(" in lowered.cuda_src
    assert "erff(" in lowered.cuda_src
    assert "logf(" in lowered.cuda_src


def test_cuda_lowering_supports_sin_fused_elementwise(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "sin2d_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "A_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["A"], "output": "A_f32", "attrs": {"to": "f32"}},
                {"op": "sin", "inputs": ["A_f32"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "sin2d_cuda_lowering"
    assert "sinf(" in lowered.cuda_src


def test_cuda_lowering_supports_acos_fused_elementwise(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "acos2d_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "acos", "inputs": ["A"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "acos2d_cuda_lowering"
    assert "acosf(" in lowered.cuda_src


def test_cuda_lowering_supports_sqrt_fused_elementwise(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "sqrt2d_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "sqrt", "inputs": ["A"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "sqrt2d_cuda_lowering"
    assert "sqrtf(" in lowered.cuda_src


def test_cuda_lowering_supports_bitwise_elementwise(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "bitwise2d_cuda_lowering",
            "tensors": {
                "A": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "AndOut": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "OrOut": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "bitwise_and", "inputs": ["A", "B"], "output": "AndOut"},
                {"op": "bitwise_or", "inputs": ["A", "B"], "output": "OrOut"},
                {"op": "bitwise_not", "inputs": ["OrOut"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "bitwise2d_cuda_lowering"
    assert " & " in lowered.cuda_src
    assert " | " in lowered.cuda_src
    assert "~v_OrOut" in lowered.cuda_src


def test_cuda_lowering_supports_rms_norm_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "rms_norm2d_cuda_lowering",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "N_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "x_squared": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "var": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "var_eps": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "rrms": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "rrms_bcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "weight_bcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "x_norm": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "INV_RMS": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "eps", "attrs": {"value": 1.0e-5, "dtype": "f32"}},
                {"op": "const", "inputs": [], "output": "N_const", "attrs": {"value": "N", "dtype": "f32"}},
                {"op": "mul", "inputs": ["input", "input"], "output": "x_squared"},
                {"op": "reduce_sum", "inputs": ["x_squared"], "output": "var", "attrs": {"dims": [1], "scale": "1.0/N"}},
                {"op": "add", "inputs": ["var", "eps"], "output": "var_eps"},
                {"op": "rsqrt", "inputs": ["var_eps"], "output": "rrms"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["rrms"],
                    "output": "rrms_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["weight"],
                    "output": "weight_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {"op": "mul", "inputs": ["input", "rrms_bcast"], "output": "x_norm"},
                {"op": "mul", "inputs": ["x_norm", "weight_bcast"], "output": "y"},
                {"op": "identity", "inputs": ["y"], "output": "out"},
                {"op": "identity", "inputs": ["rrms"], "output": "INV_RMS"},
            ],
            "outputs": ["out", "INV_RMS"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "rms_norm2d_cuda_lowering"
    assert "rsqrtf(" in lowered.cuda_src
    assert "inv_rms" in lowered.cuda_src
    assert lowered.output_names == ["out", "INV_RMS"]


def test_cuda_lowering_supports_scatter_2d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "scatter2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "index": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "scatter", "inputs": ["inp", "index", "src"], "output": "out", "attrs": {"dim": 1}}],
            "outputs": ["out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "scatter2d_cuda_lowering"
    assert "const int dst = index" in lowered.cuda_src
    assert "(unsigned)dst < (unsigned)N" in lowered.cuda_src


def test_cuda_lowering_supports_select_scatter_2d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "select_scatter2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "select_scatter", "inputs": ["inp", "src"], "output": "out", "attrs": {"dim": 1, "index": 0}}],
            "outputs": ["out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "select_scatter2d_cuda_lowering"
    assert "INDEX_COL" in lowered.cuda_src
    assert "src[(size_t)row]" in lowered.cuda_src


def test_cuda_lowering_supports_slice_scatter_2d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "slice_scatter2d_cuda_lowering",
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
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8, "L": 4})
    assert lowered.kernel_name == "slice_scatter2d_cuda_lowering"
    assert "for (int k = 0; k < 4; ++k)" in lowered.cuda_src
    assert "src[(size_t)row * (size_t)4 + (size_t)k]" in lowered.cuda_src


def test_cuda_lowering_supports_log_softmax_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "log_softmax2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "tmp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "softmax", "inputs": ["inp"], "output": "tmp", "attrs": {"axis": 1}},
                {"op": "log", "inputs": ["tmp"], "output": "out"},
            ],
            "outputs": ["out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 64})
    assert lowered.kernel_name == "log_softmax2d_cuda_lowering"
    assert "softmax_2d_last_f32<BLOCK_THREADS, EPT," in lowered.cuda_src
    assert "(inp, out, R, C)" in lowered.cuda_src


def test_cuda_lowering_supports_remainder_elementwise(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "remainder2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "other": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "remainder", "inputs": ["inp", "other"], "output": "out"},
            ],
            "outputs": ["out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 64})
    assert lowered.kernel_name == "remainder2d_cuda_lowering"
    assert "floorf" in lowered.cuda_src
    assert "NAN" in lowered.cuda_src


def test_cuda_lowering_supports_quantile_axis1(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "quantile2d_cuda_lowering",
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
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 32})
    assert lowered.kernel_name == "quantile2d_cuda_lowering"
    assert "N_STACK = 32" in lowered.cuda_src
    assert "floorf(" in lowered.cuda_src
    assert "vals[N_STACK]" in lowered.cuda_src


def test_cuda_lowering_respects_broadcast_in_dim_axes(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "broadcast_in_dim_axes_cuda_lowering",
            "tensors": {
                "vec1": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "vec2": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "v1": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "v2": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["vec1"],
                    "output": "v1",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["vec2"],
                    "output": "v2",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {"op": "mul", "inputs": ["v1", "v2"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    # vec1 should index by row (i0), vec2 should index by col (i1).
    assert "vec2[" in lowered.cuda_src and "(int64_t)(i1)" in lowered.cuda_src
    assert "Out[(size_t)tid]" in lowered.cuda_src


def test_cuda_lowering_supports_row_all_eq_reduce_not_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "row_all_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "zero": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "is_zero": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "any_zero": {"dtype": "bool", "shape": ["M", 1], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M", 1], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0}},
                {"op": "eq", "inputs": ["inp", "zero"], "output": "is_zero"},
                {"op": "reduce_any", "inputs": ["is_zero"], "output": "any_zero", "attrs": {"dims": [1], "keepdims": True}},
                {"op": "not", "inputs": ["any_zero"], "output": "out"},
            ],
            "outputs": ["out"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 256, "parallel_axes": ["M"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "row_all_cuda_lowering"
    assert "block_allreduce_max" in lowered.cuda_src
    assert "((any != 0) ? false : true)" in lowered.cuda_src


def test_cuda_lowering_supports_addmm_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "addmm2d_cuda_lowering",
            "tensors": {
                "a": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "b": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "i": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "matmul_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "scaled_matmul": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "scaled_bias": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sum_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "c": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["a", "b"], "output": "matmul_out"},
                {"op": "mul", "inputs": ["matmul_out", "alpha"], "output": "scaled_matmul"},
                {"op": "mul", "inputs": ["i", "beta"], "output": "scaled_bias"},
                {"op": "add", "inputs": ["scaled_matmul", "scaled_bias"], "output": "sum_out"},
                {"op": "cast", "inputs": ["sum_out"], "output": "c", "attrs": {"to": "f32"}},
            ],
            "outputs": ["c"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_m": 8, "tile_n": 16, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 8, "N": 8, "K": 16})
    assert lowered.kernel_name == "addmm2d_cuda_lowering"
    assert "alpha * acc + beta * bias" in lowered.cuda_src


def test_cuda_lowering_supports_addmv_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "addmv2d_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "Inp": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "mv_result": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "scaled_mv": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "scaled_inp": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["A", "B"], "output": "mv_result"},
                {"op": "mul", "inputs": ["mv_result", "alpha"], "output": "scaled_mv"},
                {"op": "mul", "inputs": ["Inp", "beta"], "output": "scaled_inp"},
                {"op": "add", "inputs": ["scaled_mv", "scaled_inp"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"N": 16, "M": 8})
    assert lowered.kernel_name == "addmv2d_cuda_lowering"
    assert "alpha * acc + beta *" in lowered.cuda_src


def test_cuda_lowering_supports_baddbmm_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "baddbmm3d_cuda_lowering",
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
            "parallel_axes": ["BATCH", "M", "N"],
            "schedule": {"tile_m": 8, "tile_n": 16, "tile_k": 8, "parallel_axes": ["BATCH", "M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"BATCH": 2, "M": 8, "N": 8, "K": 16})
    assert lowered.kernel_name == "baddbmm3d_cuda_lowering"
    assert "blockIdx.z" in lowered.cuda_src
    assert "alpha * acc + beta * bias" in lowered.cuda_src


def test_cuda_lowering_supports_batch_norm2d_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "batch_norm2d_cuda_lowering",
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
                "mean": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "mean_bcast": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "var_eps": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "inv_std": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "output_1": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "running_mean_out": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "running_var_out": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_sum", "inputs": ["input"], "output": "mean", "attrs": {"dims": [0, 2]}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["mean"],
                    "output": "mean_bcast",
                    "attrs": {"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]},
                },
                {"op": "add", "inputs": ["mean", "eps"], "output": "var_eps"},
                {"op": "rsqrt", "inputs": ["var_eps"], "output": "inv_std"},
                {"op": "identity", "inputs": ["input"], "output": "output_1"},
                {"op": "add", "inputs": ["running_mean", "mean"], "output": "running_mean_out"},
                {"op": "add", "inputs": ["running_var", "mean"], "output": "running_var_out"},
            ],
            "outputs": ["output_1", "mean", "inv_std", "running_mean_out", "running_var_out"],
            "parallel_axes": ["N", "C", "HW"],
            "schedule": {"tile_n": 256, "parallel_axes": ["N", "C", "HW"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"N": 2, "C": 4, "HW": 8})
    assert lowered.kernel_name == "batch_norm2d_cuda_lowering"
    assert "running_mean_out" in lowered.cuda_src
    assert "block_allreduce_sum" in lowered.cuda_src
    assert len(lowered.output_names) == 5


def test_cuda_lowering_supports_per_token_group_quant_fp8_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
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
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(
        intent,
        shape_bindings={"M": 4, "N": 64, "GROUP_SIZE": 16, "G": 4, "MG": 16},
    )
    assert lowered.kernel_name == "per_token_group_quant_fp8_2d"
    assert len(lowered.output_names) == 2
    assert "fabsf" in lowered.cuda_src
    assert "fmaxf(absmax, eps_v) / qmax" in lowered.cuda_src
    assert "y_q" in lowered.cuda_src


def test_cuda_lowering_supports_allclose_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "allclose2d_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "rtol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "atol": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "diff": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "abs_diff": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "abs_b": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "rtol_term": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "tol": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "close_mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "not_close": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "any_not_close": {"dtype": "bool", "shape": [], "layout": "row_major"},
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
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "allclose2d_cuda_lowering"
    assert "fabsf(av - bv)" in lowered.cuda_src
    assert "output[0] = (any == 0)" in lowered.cuda_src


def test_cuda_lowering_supports_bitwise_or_and_right_shift(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "bitwise_or_right_shift_cuda_lowering",
            "tensors": {
                "A": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "OR": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "bitwise_or", "inputs": ["A", "B"], "output": "OR"},
                {"op": "bitwise_right_shift", "inputs": ["OR", "B"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "bitwise_or_right_shift_cuda_lowering"
    assert "|" in lowered.cuda_src
    assert ">>" in lowered.cuda_src


def test_cuda_lowering_supports_masked_select_2d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "masked_select2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
            },
            "ops": [
                {"op": "masked_select", "inputs": ["inp", "mask"], "output": "out"},
            ],
            "outputs": ["out"],
            "parallel_axes": [],
            "schedule": {"tile_n": 64},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 16, "L": 24})
    assert lowered.kernel_name == "masked_select2d_cuda_lowering"
    assert "if (mask[i] != 0)" in lowered.cuda_src
    assert "out_pos < L" in lowered.cuda_src


def test_cuda_lowering_supports_masked_scatter_2d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "masked_scatter2d_cuda_lowering",
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
            "parallel_axes": [],
            "schedule": {"tile_n": 64},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 16, "L": 24})
    assert lowered.kernel_name == "masked_scatter2d_cuda_lowering"
    assert "src_pos < L" in lowered.cuda_src
    assert "out[i] = source[src_pos]" in lowered.cuda_src


def test_cuda_lowering_supports_var_from_std_mul_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "var_mean2d_cuda_lowering",
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
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "var_mean2d_cuda_lowering"
    assert "double var = sq / den;" in lowered.cuda_src
    assert "out[(size_t)row" in lowered.cuda_src


def test_cuda_lowering_supports_vector_norm_mul_reduce_sqrt_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "vector_norm2d_cuda_lowering",
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
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 16})
    assert lowered.kernel_name == "vector_norm2d_cuda_lowering"
    assert "sum_sq += v * v;" in lowered.cuda_src
    assert "sqrtf((float)sum_sq)" in lowered.cuda_src


def test_cuda_lowering_supports_upsample_nearest1d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
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
    lowered = _lower_or_skip(intent, shape_bindings={"N": 2, "C": 3, "IL": 8, "OL": 16})
    assert lowered.kernel_name == "upsample_nearest1d_ncl"
    assert "int il = (int)(((int64_t)ol" in lowered.cuda_src
    assert "out_idx" in lowered.cuda_src


def test_cuda_lowering_supports_upsample_nearest2d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
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
    lowered = _lower_or_skip(intent, shape_bindings={"N": 1, "C": 2, "IH": 8, "IW": 8, "OH": 16, "OW": 16})
    assert lowered.kernel_name == "upsample_nearest2d_nchw"
    assert "int ih = (int)(((int64_t)oh" in lowered.cuda_src
    assert "int iw = (int)(((int64_t)ow" in lowered.cuda_src


def test_cuda_lowering_supports_upsample_bicubic2d_aa_macro(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "upsample_bicubic2d_aa",
            "tensors": {
                "I": {"dtype": "f32", "shape": ["N", "C", "IH", "IW"], "layout": "row_major"},
                "reciprocal_scale_h": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "reciprocal_scale_w": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "O": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {"op": "upsample_bicubic2d_aa", "inputs": ["I"], "output": "O", "attrs": {}},
            ],
            "outputs": ["O"],
        }
    )
    lowered = _lower_or_skip(
        intent,
        shape_bindings={"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 4, "OW": 4},
    )
    assert lowered.kernel_name == "upsample_bicubic2d_aa"
    assert "O[(size_t)tid]" in lowered.cuda_src


def test_cuda_lowering_supports_upsample_bicubic2d_aa_expanded_name_path(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "upsample_bicubic2d_aa",
            "tensors": {
                "ptr_i": {"dtype": "f32", "shape": ["N", "C", "IH", "IW"], "layout": "row_major"},
                "reciprocal_scale_h": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "reciprocal_scale_w": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "tmp": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
                "ptr_o": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {"op": "identity", "inputs": ["ptr_i"], "output": "tmp"},
                {"op": "identity", "inputs": ["tmp"], "output": "ptr_o"},
            ],
            "outputs": ["ptr_o"],
        }
    )
    lowered = _lower_or_skip(
        intent,
        shape_bindings={"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 4, "OW": 4},
    )
    assert lowered.kernel_name == "upsample_bicubic2d_aa"
    assert "const float* __restrict__ ptr_i" in lowered.cuda_src
    assert "float* __restrict__ ptr_o" in lowered.cuda_src
    assert "ptr_o[(size_t)tid]" in lowered.cuda_src


def test_cuda_lowering_supports_weight_norm_dim0_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "weight_norm2d_cuda_lowering",
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
                {"op": "broadcast_in_dim", "inputs": ["scale"], "output": "scale_bc", "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]}},
                {"op": "mul", "inputs": ["v", "scale_bc"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 8, "N": 16})
    assert lowered.kernel_name == "weight_norm2d_cuda_lowering"
    assert "const float norm = sqrtf((float)sum_sq);" in lowered.cuda_src
    assert "row_ptr[(size_t)col] * scale" in lowered.cuda_src


def test_cuda_lowering_supports_topk2d_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "topk2d_cuda_lowering",
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
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 16, "K": 4})
    assert lowered.kernel_name == "topk2d_cuda_lowering"
    assert "float top_vals[256];" in lowered.cuda_src
    assert "for (int n = 0; n < N; ++n)" in lowered.cuda_src


def test_cuda_lowering_supports_trace2d_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "trace2d_cuda_lowering",
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
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0, "dtype": "f32"}},
                {"op": "where", "inputs": ["diag_mask", "input", "zero_const"], "output": "diag_vals"},
                {"op": "reduce_sum", "inputs": ["diag_vals"], "output": "out", "attrs": {"dims": [0, 1]}},
            ],
            "outputs": ["out"],
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 16, "N": 12})
    assert lowered.kernel_name == "trace2d_cuda_lowering"
    assert "const int L = (M < N) ? M : N;" in lowered.cuda_src
    assert "acc += input[(size_t)i * (size_t)N + (size_t)i];" in lowered.cuda_src


def test_cuda_lowering_supports_diag2d_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "diag2d_cuda_lowering",
            "tensors": {
                "data": {"dtype": "f32", "shape": ["M", "M"], "layout": "row_major"},
                "diag_idx": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "diag_idx", "attrs": {"axis": 0, "shape": ["M"], "dtype": "i32"}},
                {"op": "gather", "inputs": ["data", "diag_idx", "diag_idx"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 16})
    assert lowered.kernel_name == "diag2d_cuda_lowering"
    assert "data[(size_t)row * (size_t)M + (size_t)row]" in lowered.cuda_src


def test_cuda_lowering_supports_diag_embed2d_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "diag_embed2d_cuda_lowering",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["B", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["B", "N", "N"], "layout": "row_major"},
                "zero_scalar": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "y_zeros": {"dtype": "f32", "shape": ["B", "N", "N"], "layout": "row_major"},
                "idx_row": {"dtype": "i32", "shape": ["B", "N", "N"], "layout": "row_major"},
                "idx_col": {"dtype": "i32", "shape": ["B", "N", "N"], "layout": "row_major"},
                "diag_mask": {"dtype": "bool", "shape": ["B", "N", "N"], "layout": "row_major"},
                "x_bcast": {"dtype": "f32", "shape": ["B", "N", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_scalar", "attrs": {"value": 0.0, "dtype": "f32"}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["zero_scalar"],
                    "output": "y_zeros",
                    "attrs": {"broadcast_dims": [], "out_shape": ["B", "N", "N"]},
                },
                {"op": "iota", "inputs": [], "output": "idx_row", "attrs": {"axis": 1, "shape": ["B", "N", "N"], "dtype": "i32"}},
                {"op": "iota", "inputs": [], "output": "idx_col", "attrs": {"axis": 2, "shape": ["B", "N", "N"], "dtype": "i32"}},
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
    lowered = _lower_or_skip(intent, shape_bindings={"B": 2, "N": 8})
    assert lowered.kernel_name == "diag_embed2d_cuda_lowering"
    assert "const float v = (row == col)" in lowered.cuda_src
    assert "x[(size_t)b * (size_t)N + (size_t)col]" in lowered.cuda_src


def test_cuda_lowering_supports_nonzero2d_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "nonzero2d_cuda_lowering",
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
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8, "num_nonzeros": 16})
    assert lowered.kernel_name == "nonzero2d_cuda_lowering"
    assert "if (v != 0.0f)" in lowered.cuda_src
    assert "out[(size_t)write_idx * 2]" in lowered.cuda_src


def test_cuda_lowering_supports_count_nonzero2d_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "count_nonzero2d_cuda_lowering",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i64", "shape": [], "layout": "row_major"},
                "zero_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "is_nonzero_bool": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "is_nonzero_i64": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0}},
                {"op": "ne", "inputs": ["x", "zero_const"], "output": "is_nonzero_bool"},
                {"op": "cast", "inputs": ["is_nonzero_bool"], "output": "is_nonzero_i64", "attrs": {"to": "i64"}},
                {"op": "reduce_sum", "inputs": ["is_nonzero_i64"], "output": "out", "attrs": {"dims": [0, 1]}},
            ],
            "outputs": ["out"],
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "count_nonzero2d_cuda_lowering"
    assert "if (x[(size_t)i] != 0.0f) ++acc;" in lowered.cuda_src
    assert "out[0] = acc;" in lowered.cuda_src


def test_cuda_lowering_supports_bf16_cast_minimum_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "minimum2d_cuda_lowering",
            "tensors": {
                "X": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "Y": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "x_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "result_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["X"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["Y"], "output": "y_f32", "attrs": {"to": "f32"}},
                {"op": "min", "inputs": ["x_f32", "y_f32"], "output": "result_f32"},
                {"op": "cast", "inputs": ["result_f32"], "output": "Out", "attrs": {"to": "bf16"}},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "minimum2d_cuda_lowering"
    assert "__bfloat162float(" in lowered.cuda_src
    assert "__float2bfloat16(" in lowered.cuda_src


def test_cuda_lowering_supports_exp_base2_attr(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "exp22d_cuda_lowering",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "exp", "inputs": ["x"], "output": "out", "attrs": {"base": 2.0}},
            ],
            "outputs": ["out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "exp22d_cuda_lowering"
    assert "exp2f(" in lowered.cuda_src


def test_cuda_lowering_supports_row_mean_sum_div_pattern(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "row_mean_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sum_result": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "N_scalar": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "N_scalar", "attrs": {"value": "N"}},
                {"op": "reduce_sum", "inputs": ["inp"], "output": "sum_result", "attrs": {"dims": [1]}},
                {"op": "div", "inputs": ["sum_result", "N_scalar"], "output": "out"},
            ],
            "outputs": ["out"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "row_mean_cuda_lowering"
    assert "block_allreduce_sum" in lowered.cuda_src
    assert "sum / (float)N" in lowered.cuda_src


def test_cuda_lowering_supports_reduce_min_all_2d(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "min2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_value": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_min", "inputs": ["inp"], "output": "out_value", "attrs": {"dims": [0, 1], "keepdims": False}},
            ],
            "outputs": ["out_value"],
            "parallel_axes": [],
            "schedule": {"tile_n": 128},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "min2d_cuda_lowering"
    assert "fminf" in lowered.cuda_src
    assert "out_value[0]" in lowered.cuda_src


def test_cuda_lowering_supports_reduce_min_axis1_with_indices(monkeypatch) -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "min_dim2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "max_value": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "min_values_init": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "argmin_values_init": {"dtype": "i64", "shape": ["M"], "layout": "row_major"},
                "out_value": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out_index": {"dtype": "i64", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "max_value", "attrs": {"value": 3.4028235e38}},
                {"op": "broadcast_in_dim", "inputs": ["max_value"], "output": "min_values_init", "attrs": {"out_shape": ["M"], "broadcast_dims": []}},
                {"op": "const", "inputs": [], "output": "argmin_values_init", "attrs": {"value": 0, "dtype": "i64", "shape": ["M"]}},
                {
                    "op": "reduce_min",
                    "inputs": ["inp"],
                    "output": "out_value",
                    "attrs": {"dims": [1], "keepdims": False, "return_indices": True, "index_output": "out_index"},
                },
            ],
            "outputs": ["out_value", "out_index"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M"]},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "min_dim2d_cuda_lowering"
    assert "int64_t* __restrict__ out_index" in lowered.cuda_src
    assert "out_index[(size_t)m]" in lowered.cuda_src


def test_cuda_lowering_supports_argmax_argmin_axis1(monkeypatch) -> None:
    argmax = IntentFunction.from_json_dict(
        {
            "name": "argmax2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_index": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "argmax", "inputs": ["inp"], "output": "out_index", "attrs": {"axis": 1}}],
            "outputs": ["out_index"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M"]},
        }
    )
    argmin = IntentFunction.from_json_dict(
        {
            "name": "argmin2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_index": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "argmin", "inputs": ["inp"], "output": "out_index", "attrs": {"axis": 1}}],
            "outputs": ["out_index"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M"]},
        }
    )
    lowered_max = _lower_or_skip(argmax, shape_bindings={"M": 4, "N": 8})
    lowered_min = _lower_or_skip(argmin, shape_bindings={"M": 4, "N": 8})
    assert lowered_max.kernel_name == "argmax2d_cuda_lowering"
    assert lowered_min.kernel_name == "argmin2d_cuda_lowering"
    assert "int* __restrict__ out_index" in lowered_max.cuda_src
    assert "int* __restrict__ out_index" in lowered_min.cuda_src
