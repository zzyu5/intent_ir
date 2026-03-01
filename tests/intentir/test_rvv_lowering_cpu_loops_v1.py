from __future__ import annotations

import subprocess

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, to_mlir
from intent_ir.mlir.passes.lower_intent_to_rvv_cpu_kernel import lower_intent_to_rvv_cpu_kernel


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


def _add2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}}],
            "outputs": ["z"],
        }
    )


def _add_bias2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add_bias2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["x", "bias"], "output": "out", "attrs": {}}],
            "outputs": ["out"],
        }
    )


def _transpose2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "transpose2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
            },
            "ops": [{"op": "transpose", "inputs": ["inp"], "output": "out", "attrs": {"perm": [1, 0]}}],
            "outputs": ["out"],
        }
    )


def _copy2d_divmod_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "copy2d_divmod",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "identity", "inputs": ["inp"], "output": "out", "attrs": {}}],
            "outputs": ["out"],
        }
    )


def _rowmask_where2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "rowmask_where2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "i32", "shape": ["M"], "layout": "row_major"},
                "mask_2d": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "zero": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "zero_2d": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["mask"],
                    "output": "mask_2d",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["zero"],
                    "output": "zero_2d",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": []},
                },
                {"op": "where", "inputs": ["mask_2d", "inp", "zero_2d"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out"],
        }
    )


def _relu2d_where_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "relu2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "zero": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "gt", "inputs": ["x", "zero"], "output": "mask", "attrs": {}},
                {"op": "where", "inputs": ["mask", "x", "zero"], "output": "output", "attrs": {}},
            ],
            "outputs": ["output"],
        }
    )


def _clamp2d_cast_intent() -> IntentFunction:
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
                {"op": "max", "inputs": ["mini", "x_f32"], "output": "clamped_min", "attrs": {}},
                {"op": "min", "inputs": ["clamped_min", "maxi"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out"],
        }
    )


def _cast2d_f16_to_f32_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "cast2d",
            "tensors": {
                "x": {"dtype": "f16", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "cast", "inputs": ["x"], "output": "out", "attrs": {"to": "f32"}}],
            "outputs": ["out"],
        }
    )


def _where2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "where2d",
            "tensors": {
                "condition": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "self": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "other": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "where", "inputs": ["condition", "self", "other"], "output": "out", "attrs": {}}],
            "outputs": ["out"],
        }
    )


def _neg2d_const_intent() -> IntentFunction:
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
                {"op": "mul", "inputs": ["A", "neg_one"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out"],
        }
    )


def _maximum2d_bf16_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "maximum2d",
            "tensors": {
                "X": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "Y": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "X_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Y_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "result_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["X"], "output": "X_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["Y"], "output": "Y_f32", "attrs": {"to": "f32"}},
                {"op": "max", "inputs": ["X_f32", "Y_f32"], "output": "result_f32", "attrs": {}},
                {"op": "cast", "inputs": ["result_f32"], "output": "Out", "attrs": {"to": "bf16"}},
            ],
            "outputs": ["Out"],
        }
    )


def _unary2d_intent(op: str, *, name: str) -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": str(op), "inputs": ["x"], "output": "out", "attrs": {}}],
            "outputs": ["out"],
        }
    )


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
            "ops": [
                {
                    "op": "reduce_min",
                    "inputs": ["inp"],
                    "output": "out_value",
                    "attrs": {"dims": [0, 1], "keepdims": False},
                }
            ],
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
                {
                    "op": "reduce_min",
                    "inputs": ["inp"],
                    "output": "out_value",
                    "attrs": {"dims": [1], "keepdims": False},
                },
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
                "is_nonzero_bool": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "is_nonzero_i64": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "i64", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0}},
                {"op": "ne", "inputs": ["x", "zero_const"], "output": "is_nonzero_bool", "attrs": {}},
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
                "diag_mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "zero_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "diag_vals": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "row_idx", "attrs": {"axis": 0, "shape": ["M", "N"], "dtype": "i32"}},
                {"op": "iota", "inputs": [], "output": "col_idx", "attrs": {"axis": 1, "shape": ["M", "N"], "dtype": "i32"}},
                {"op": "eq", "inputs": ["row_idx", "col_idx"], "output": "diag_mask", "attrs": {}},
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "where", "inputs": ["diag_mask", "input", "zero_const"], "output": "diag_vals", "attrs": {}},
                {"op": "reduce_sum", "inputs": ["diag_vals"], "output": "out", "attrs": {"dims": [0, 1], "keepdims": False}},
            ],
            "outputs": ["out"],
        }
    )


def _allclose2d_intent() -> IntentFunction:
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
                "rtol_term": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "tol": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "close_mask": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "not_close": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                "any_not_close": {"dtype": "bool", "shape": [], "layout": "row_major"},
                "output": {"dtype": "bool", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "sub", "inputs": ["A", "B"], "output": "diff", "attrs": {}},
                {"op": "abs", "inputs": ["diff"], "output": "abs_diff", "attrs": {}},
                {"op": "abs", "inputs": ["B"], "output": "abs_b", "attrs": {}},
                {"op": "mul", "inputs": ["rtol", "abs_b"], "output": "rtol_term", "attrs": {}},
                {"op": "add", "inputs": ["atol", "rtol_term"], "output": "tol", "attrs": {}},
                {"op": "le", "inputs": ["abs_diff", "tol"], "output": "close_mask", "attrs": {}},
                {"op": "not", "inputs": ["close_mask"], "output": "not_close", "attrs": {}},
                {"op": "reduce_any", "inputs": ["not_close"], "output": "any_not_close", "attrs": {"dims": [0, 1]}},
                {"op": "not", "inputs": ["any_not_close"], "output": "output", "attrs": {}},
            ],
            "outputs": ["output"],
        }
    )


def _softmax_inner_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "softmax_inner",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "m_max": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "m_max_bcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "centered": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "exp_vals": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sum_exp": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "sum_exp_bcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_max", "inputs": ["input"], "output": "m_max", "attrs": {"dims": [1]}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["m_max"],
                    "output": "m_max_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {"op": "sub", "inputs": ["input", "m_max_bcast"], "output": "centered", "attrs": {}},
                {"op": "exp", "inputs": ["centered"], "output": "exp_vals", "attrs": {}},
                {"op": "reduce_sum", "inputs": ["exp_vals"], "output": "sum_exp", "attrs": {"dims": [1]}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["sum_exp"],
                    "output": "sum_exp_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {"op": "div", "inputs": ["exp_vals", "sum_exp_bcast"], "output": "output", "attrs": {}},
            ],
            "outputs": ["output"],
        }
    )


def _log_softmax2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "log_softmax2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "softmax_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "softmax", "inputs": ["inp"], "output": "softmax_out", "attrs": {"axis": 1}},
                {"op": "log", "inputs": ["softmax_out"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out"],
        }
    )


def test_rvv_cpu_loops_v1_lowering_emits_scf_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_add2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_v1"
    assert "scf.for" in str(out.module_text or "")
    assert "memref.load" in str(out.module_text or "")
    _verify_with_mlir_opt(str(out.module_text or ""))


def test_rvv_cpu_loops_v1_supports_rank1_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_add_bias2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    text = str(out.module_text or "")
    assert "memref<64xf32>" in text
    assert "memref.load %bias[%n]" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_transpose2d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_transpose2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_transpose2d_v1"
    text = str(out.module_text or "")
    assert "memref.load %inp[%in_idx]" in text
    assert "memref.store %v, %out[%out_idx]" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_identity_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_copy2d_divmod_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_v1"
    text = str(out.module_text or "")
    assert "memref.load %inp[%i]" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_rowmask_where2d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_rowmask_where2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_v1"
    text = str(out.module_text or "")
    assert "memref<4xi32>" in text
    assert "%c0i32 = arith.constant 0 : i32" in text
    assert "arith.cmpi ne" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_rewrites_relu_where_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_relu2d_where_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_v1"
    text = str(out.module_text or "")
    assert "arith.maximumf" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_cast_to_f32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_clamp2d_cast_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    text = str(out.module_text or "")
    assert "arith.maximumf" in text
    assert "arith.minimumf" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_cast_f16_to_f32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_cast2d_f16_to_f32_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    text = str(out.module_text or "")
    assert "arith.extf" in text
    assert "xf16" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_where2d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_where2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    text = str(out.module_text or "")
    assert "arith.select" in text
    assert "arith.cmpi" in text
    assert "xi8" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_const_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_neg2d_const_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    text = str(out.module_text or "")
    assert "arith.constant -1.0" in text
    assert "arith.mulf" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_normalizes_bf16_to_f32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_maximum2d_bf16_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert bool(out.meta.get("rvv_dtype_normalized_bf16_to_f32")) is True
    text = str(out.module_text or "")
    assert "bf16" not in text
    assert "arith.maximumf" in text
    _verify_with_mlir_opt(text)


@pytest.mark.parametrize(
    ("op_name", "expected_mlir"),
    [
        ("floor", "math.floor"),
        ("ceil", "math.ceil"),
        ("log", "math.log"),
        ("sin", "math.sin"),
    ],
)
def test_rvv_cpu_loops_v1_supports_more_unary_ops(
    monkeypatch: pytest.MonkeyPatch, op_name: str, expected_mlir: str
) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_unary2d_intent(op_name, name=f"{op_name}2d"))
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    text = str(out.module_text or "")
    assert expected_mlir in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_argmax2d_i32_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_argmax2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_row_argmax_axis1_v1"
    text = str(out.module_text or "")
    assert "memref<4xi32>" in text
    assert "arith.cmpf ogt" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_argmin2d_i32_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_argmin2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_row_argmin_axis1_v1"
    text = str(out.module_text or "")
    assert "memref<4xi32>" in text
    assert "arith.cmpf olt" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_prod_dim2d_row_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_prod_dim2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_row_reduce_prod_axis1_v1"
    text = str(out.module_text or "")
    assert "arith.mulf" in text
    assert "memref<4xf32>" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_min2d_scalar_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_min2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_reduce_min_all_v1"
    text = str(out.module_text or "")
    assert "memref<1xf32>" in text
    assert "arith.minimumf" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_min_dim2d_value_and_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_min_dim2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_row_reduce_min_argmin_axis1_v1"
    text = str(out.module_text or "")
    assert "memref<4xf32>" in text
    assert "memref<4xi32>" in text
    assert "arith.cmpf olt" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_count_nonzero2d_i64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_count_nonzero2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_count_nonzero2d_v1"
    text = str(out.module_text or "")
    assert "memref<1xi64>" in text
    assert "cmpf une" in text
    assert "arith.addi" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_trace2d_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_trace2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_trace2d_v1"
    text = str(out.module_text or "")
    assert "memref<1xf32>" in text
    assert "memref.load %input[%idx]" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_allclose2d_bool_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_allclose2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_allclose2d_v1"
    text = str(out.module_text or "")
    assert "memref<1xi8>" in text
    assert "math.absf" in text
    assert "arith.cmpf ole" in text
    assert "arith.extui" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_softmax_inner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_softmax_inner_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_softmax_inner_v1"
    text = str(out.module_text or "")
    assert "arith.maximumf" in text
    assert "math.exp" in text
    assert "arith.divf" in text
    _verify_with_mlir_opt(text)


def test_rvv_cpu_loops_v1_supports_log_softmax2d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_log_softmax2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_log_softmax2d_v1"
    text = str(out.module_text or "")
    assert "arith.maximumf" in text
    assert "math.exp" in text
    assert "math.log" in text
    _verify_with_mlir_opt(text)
