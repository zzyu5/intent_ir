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


def _mm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "mm2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {"transpose_a": False, "transpose_b": False}},
            ],
            "outputs": ["C"],
        }
    )


def _addmm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "addmm2d",
            "tensors": {
                "mat1": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "mat2": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "mm_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "scaled_mm": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "scaled_bias": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "add_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "matmul",
                    "inputs": ["mat1", "mat2"],
                    "output": "mm_out",
                    "attrs": {"transpose_a": False, "transpose_b": False},
                },
                {"op": "mul", "inputs": ["mm_out", "alpha"], "output": "scaled_mm"},
                {"op": "mul", "inputs": ["input", "beta"], "output": "scaled_bias"},
                {"op": "add", "inputs": ["scaled_mm", "scaled_bias"], "output": "add_out"},
                {"op": "cast", "inputs": ["add_out"], "output": "out", "attrs": {"to": "f32"}},
            ],
            "outputs": ["out"],
        }
    )


def test_cuda_real_mlir_mm2d_lowering_uses_matmul_tile_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _mm2d_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 16, "K": 32, "N": 16}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "matmul_tile_v2"
    assert "vector<4xf32>" in out.module_text
    assert "llvm.intr.fma" in out.module_text
    _verify_with_mlir_opt(out.module_text)


def test_cuda_real_mlir_addmm2d_lowering_fuses_alpha_beta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _addmm2d_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 16, "K": 32, "N": 16}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "matmul_tile_v2"
    assert "memref.load %alpha[%c0]" in out.module_text
    assert "memref.load %beta[%c0]" in out.module_text
    assert "vector.load %input[" in out.module_text
    _verify_with_mlir_opt(out.module_text)

