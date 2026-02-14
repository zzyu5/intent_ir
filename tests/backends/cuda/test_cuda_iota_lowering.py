from __future__ import annotations

from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
from intent_ir.ir import IntentFunction


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
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
    intent = _eye_like_intent("eye2d_cuda_lowering", rows="N", cols="N")
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"N": 8})
    assert lowered.kernel_name == "eye2d_cuda_lowering"
    assert "v_idx_row = (int)(i0);" in lowered.cuda_src
    assert "v_idx_col = (int)(i1);" in lowered.cuda_src


def test_cuda_lowering_supports_eye_like_rectangular_iota_graph(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
    intent = _eye_like_intent("eye_m2d_cuda_lowering", rows="N", cols="M")
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"N": 8, "M": 6})
    assert lowered.kernel_name == "eye_m2d_cuda_lowering"
    assert "v_idx_row = (int)(i0);" in lowered.cuda_src
    assert "v_idx_col = (int)(i1);" in lowered.cuda_src


def test_cuda_lowering_supports_cos_erf_fused_elementwise(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
    intent = IntentFunction.from_json_dict(
        {
            "name": "cos_erf_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "E": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cos", "inputs": ["A"], "output": "C"},
                {"op": "erf", "inputs": ["A"], "output": "E"},
                {"op": "add", "inputs": ["C", "E"], "output": "Out"},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "cos_erf_cuda_lowering"
    assert "cosf(" in lowered.cuda_src
    assert "erff(" in lowered.cuda_src


def test_cuda_lowering_respects_broadcast_in_dim_axes(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
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
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"M": 4, "N": 8})
    # vec1 should index by row (i0), vec2 should index by col (i1).
    assert "vec1[(size_t)(((int64_t)i0)" in lowered.cuda_src
    assert "vec2[(size_t)(((int64_t)i1)" in lowered.cuda_src


def test_cuda_lowering_supports_row_all_eq_reduce_not_pattern(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
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
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "row_all_cuda_lowering"
    assert "block_allreduce_max" in lowered.cuda_src
    assert "((any != 0) ? false : true)" in lowered.cuda_src


def test_cuda_lowering_supports_addmm_pattern(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
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
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"M": 8, "N": 8, "K": 16})
    assert lowered.kernel_name == "addmm2d_cuda_lowering"
    assert "alpha * acc + beta * bias" in lowered.cuda_src


def test_cuda_lowering_supports_addmv_pattern(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
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
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"N": 16, "M": 8})
    assert lowered.kernel_name == "addmv2d_cuda_lowering"
    assert "alpha * acc + beta *" in lowered.cuda_src


def test_cuda_lowering_supports_allclose_pattern(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")
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
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "allclose2d_cuda_lowering"
    assert "fabsf(av - bv)" in lowered.cuda_src
    assert "output[0] = (any == 0)" in lowered.cuda_src
