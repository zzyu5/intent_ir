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


def _dot1d_intent(*, name: str) -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": str(name),
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "x_f32": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "y_f32": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "mul_out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "cast", "inputs": ["y"], "output": "y_f32", "attrs": {"to": "f32"}},
                {"op": "mul", "inputs": ["x_f32", "y_f32"], "output": "mul_out"},
                {"op": "reduce_sum", "inputs": ["mul_out"], "output": "out", "attrs": {"dims": [0], "keepdims": False}},
            ],
            "outputs": ["out"],
        }
    )


def _mv2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "mv2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "Inp": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["A", "B"], "output": "mv_out"},
                {"op": "mul", "inputs": ["mv_out", "alpha"], "output": "mv_scaled"},
                {"op": "mul", "inputs": ["Inp", "beta"], "output": "inp_scaled"},
                {"op": "add", "inputs": ["mv_scaled", "inp_scaled"], "output": "C"},
            ],
            "outputs": ["C"],
        }
    )


@pytest.mark.parametrize("name", ["dot1d", "vdot1d"])
def test_cuda_real_mlir_dot1d_vdot1d_codegen(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _dot1d_intent(name=name)
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"N": 256}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "row_reduce_sum_axis1_v1"
    _verify_with_mlir_opt(out.module_text)


def test_cuda_real_mlir_mv2d_codegen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _mv2d_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"N": 16, "M": 32}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "matvec_v1"
    assert "gpu.thread_id x" in out.module_text
    _verify_with_mlir_opt(out.module_text)

