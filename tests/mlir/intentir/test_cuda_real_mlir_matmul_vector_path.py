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


def _ai_bench_matmul_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "ai_bench_matmul",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "Out", "attrs": {}}],
            "outputs": ["Out"],
        }
    )


def _matmul_fused_epilogue2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "matmul_fused_epilogue2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "Bias": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "RowMask": {"dtype": "i1", "shape": ["M"], "layout": "row_major"},
                "ColMask": {"dtype": "i1", "shape": ["N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "matmul_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "add_bias_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row_mask_bcast": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "mask_combined": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "zero_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "matmul", "inputs": ["A", "B"], "output": "matmul_out"},
                {"op": "add", "inputs": ["matmul_out", "Bias"], "output": "add_bias_out"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["RowMask"],
                    "output": "row_mask_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {"op": "and", "inputs": ["row_mask_bcast", "ColMask"], "output": "mask_combined"},
                {"op": "const", "inputs": [], "output": "zero_const", "attrs": {"value": 0.0}},
                {"op": "where", "inputs": ["mask_combined", "add_bias_out", "zero_const"], "output": "C"},
            ],
            "outputs": ["C"],
        }
    )


def test_cuda_real_mlir_matmul_tile_v2_vectorizes_4cols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _ai_bench_matmul_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 64, "N": 16, "K": 16}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "matmul_tile_v2"
    assert "vector<4xf32>" in out.module_text
    assert "vector.load" in out.module_text
    _verify_with_mlir_opt(out.module_text)


def test_cuda_real_mlir_matmul_fused_epilogue2d_vector_mask_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _matmul_fused_epilogue2d_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 32, "N": 32, "K": 32}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "matmul_tile_v2"
    assert "vector<4xf32>" in out.module_text
    _verify_with_mlir_opt(out.module_text)
