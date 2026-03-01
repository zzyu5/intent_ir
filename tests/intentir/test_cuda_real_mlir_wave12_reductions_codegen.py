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


@pytest.mark.parametrize(
    "intent_fn,shape_bindings,expected_kind,needle",
    [
        (_argmax2d_intent, {"M": 4, "N": 64}, "row_argmax_axis1_v1", "arith.cmpf"),
        (_argmin2d_intent, {"M": 4, "N": 64}, "row_argmin_axis1_v1", "arith.cmpf"),
        (_prod_dim2d_intent, {"M": 4, "N": 64}, "row_reduce_prod_axis1_v1", "arith.mulf"),
        (_min_dim2d_intent, {"M": 4, "N": 64}, "row_reduce_min_argmin_axis1_v1", "memref.get_global"),
        (_min2d_intent, {"M": 4, "N": 64}, "reduce_min_all_v1", "arith.min"),
        (_trace2d_intent, {"M": 16, "N": 16}, "trace2d_v1", "gpu.thread_id x"),
        (_count_nonzero2d_intent, {"M": 4, "N": 64}, "count_nonzero2d_v1", "arith.addi"),
        (_allclose2d_intent, {"M": 4, "N": 64}, "allclose2d_v1", "math.absf"),
    ],
)
def test_cuda_real_mlir_wave12_codegen_and_is_parseable(
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

