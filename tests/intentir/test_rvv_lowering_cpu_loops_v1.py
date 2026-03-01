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


def test_rvv_cpu_loops_v1_lowering_emits_scf_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    mod = to_mlir(_add2d_intent())
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str(out.meta.get("rvv_real_mlir_kernel_kind") or "") == "cpu_loops_v1"
    assert "scf.for" in str(out.module_text or "")
    assert "memref.load" in str(out.module_text or "")
    _verify_with_mlir_opt(str(out.module_text or ""))


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
