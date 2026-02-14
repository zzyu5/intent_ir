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
