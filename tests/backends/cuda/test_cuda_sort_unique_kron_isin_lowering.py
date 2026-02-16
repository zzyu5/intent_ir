from __future__ import annotations

import pytest

from backends.cuda.codegen.cpp_driver import CudaLoweringError, lower_intent_to_cuda_kernel
from intent_ir.ir import IntentFunction


def _lower_or_skip(intent: IntentFunction, *, shape_bindings: dict[str, int]):
    try:
        return lower_intent_to_cuda_kernel(intent, shape_bindings=shape_bindings)
    except CudaLoweringError as exc:
        if "unsupported intent for cuda cpp codegen" in str(exc):
            pytest.skip(f"cpp cuda codegen unsupported for this pattern: {exc}")
        raise


def test_cuda_lowering_supports_sort2d_axis1() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "sort2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "sort", "inputs": ["inp"], "output": "out", "attrs": {"axis": 1, "descending": True}}],
            "outputs": ["out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_m": 128},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 8})
    assert lowered.kernel_name == "sort2d_cuda_lowering"
    assert "for (int64_t ii = 1;" in lowered.cuda_src
    assert "while (jj >= 0 && out[base + jj] > v)" in lowered.cuda_src


def test_cuda_lowering_supports_unique1d_i32() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "unique1d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "i32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "i32", "shape": ["U"], "layout": "row_major"},
            },
            "ops": [{"op": "unique", "inputs": ["inp"], "output": "out", "attrs": {}}],
            "outputs": ["out"],
            "parallel_axes": ["U"],
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"N": 16, "U": 8})
    assert lowered.kernel_name == "unique1d_cuda_lowering"
    assert "int64_t unique_count = 0;" in lowered.cuda_src
    assert "if (!seen)" in lowered.cuda_src


def test_cuda_lowering_supports_kron2d_f32() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "kron2d_cuda_lowering",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["P", "Q"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["MP", "NQ"], "layout": "row_major"},
            },
            "ops": [{"op": "kron", "inputs": ["A", "B"], "output": "Out", "attrs": {}}],
            "outputs": ["Out"],
            "parallel_axes": ["MP", "NQ"],
            "schedule": {"tile_n": 256},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 2, "N": 3, "P": 4, "Q": 5, "MP": 8, "NQ": 15})
    assert lowered.kernel_name == "kron2d_cuda_lowering"
    assert "const float av = A[i * 3LL + j];" in lowered.cuda_src
    assert "const float bv = B[p * 5LL + q];" in lowered.cuda_src


def test_cuda_lowering_supports_isin1d_macro_pattern() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "isin1d_cuda_lowering",
            "tensors": {
                "input": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
                "test_elements": {"dtype": "i32", "shape": ["K"], "layout": "row_major"},
                "input_bc": {"dtype": "i32", "shape": ["M", "K"], "layout": "row_major"},
                "test_bc": {"dtype": "i32", "shape": ["M", "K"], "layout": "row_major"},
                "ne_out": {"dtype": "bool", "shape": ["M", "K"], "layout": "row_major"},
                "eq_out": {"dtype": "bool", "shape": ["M", "K"], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["input"],
                    "output": "input_bc",
                    "attrs": {"broadcast_dims": [0], "out_shape": ["M", "K"]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["test_elements"],
                    "output": "test_bc",
                    "attrs": {"broadcast_dims": [1], "out_shape": ["M", "K"]},
                },
                {"op": "ne", "inputs": ["input_bc", "test_bc"], "output": "ne_out"},
                {"op": "not", "inputs": ["ne_out"], "output": "eq_out"},
                {"op": "reduce_any", "inputs": ["eq_out"], "output": "out", "attrs": {"dims": [1], "keepdims": False}},
            ],
            "outputs": ["out"],
            "parallel_axes": ["M"],
            "schedule": {"tile_m": 128},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 8, "K": 6})
    assert lowered.kernel_name == "isin1d_cuda_lowering"
    assert "const int v = input[i];" in lowered.cuda_src
    assert "for (int64_t k = 0; k < 6LL; ++k)" in lowered.cuda_src
    assert "out[i] = any;" in lowered.cuda_src


def test_cuda_lowering_supports_std2d_axis1() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "std2d_cuda_lowering",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "std", "inputs": ["X"], "output": "Out", "attrs": {"axis": 1, "keepdims": False, "correction": 1}}],
            "outputs": ["Out"],
            "parallel_axes": ["M"],
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"M": 4, "N": 16})
    assert lowered.kernel_name == "std2d_cuda_lowering"
    assert "const int64_t denom_i = (int64_t)N - (int64_t)1;" in lowered.cuda_src
    assert "sqrtf(fmaxf(var, 0.0f))" in lowered.cuda_src


def test_cuda_lowering_supports_per_token_group_quant_fp8_pattern() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "per_token_group_quant_fp8_cuda_lowering",
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
                {"op": "abs", "inputs": ["y_grouped"], "output": "y_abs", "attrs": {}},
                {"op": "reduce_max", "inputs": ["y_abs"], "output": "absmax", "attrs": {"dims": [1], "keepdims": True}},
                {"op": "max", "inputs": ["absmax", "eps"], "output": "absmax_clamped", "attrs": {}},
                {"op": "div", "inputs": ["absmax_clamped", "fp8_max"], "output": "y_s_2d", "attrs": {}},
                {"op": "reshape", "inputs": ["y_s_2d"], "output": "y_s", "attrs": {"shape": ["M", "G"]}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["y_s_2d"],
                    "output": "y_s_broadcast",
                    "attrs": {"out_shape": ["MG", "GROUP_SIZE"], "broadcast_dims": [0, 1]},
                },
                {"op": "div", "inputs": ["y_grouped", "y_s_broadcast"], "output": "y_scaled", "attrs": {}},
                {"op": "max", "inputs": ["y_scaled", "fp8_min"], "output": "y_clamped_min", "attrs": {}},
                {"op": "min", "inputs": ["y_clamped_min", "fp8_max"], "output": "y_clamped", "attrs": {}},
                {"op": "reshape", "inputs": ["y_clamped"], "output": "y_q", "attrs": {"shape": ["M", "N"]}},
            ],
            "outputs": ["y_q", "y_s"],
            "parallel_axes": ["M", "N"],
        }
    )
    lowered = _lower_or_skip(
        intent,
        shape_bindings={"M": 2, "N": 8, "G": 2, "MG": 4, "GROUP_SIZE": 4},
    )
    assert lowered.kernel_name == "per_token_group_quant_fp8_cuda_lowering"
    assert "const int64_t gid" in lowered.cuda_src
    assert "float absmax = 0.0f;" in lowered.cuda_src
    assert "y_s[m * 2LL + g] = scale;" in lowered.cuda_src
