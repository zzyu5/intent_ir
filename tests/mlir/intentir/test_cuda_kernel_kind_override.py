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


def _attn_fwd_intent() -> IntentFunction:
    # Keep `attn_mask` as an ABI arg (no-op in lowering), matching repository behavior.
    return IntentFunction.from_json_dict(
        {
            "name": "_attn_fwd",
            "tensors": {
                "Q": {"dtype": "f32", "shape": ["Z", "q_numhead", "Q_CTX", "HEAD_DIM"], "layout": "row_major"},
                "K": {"dtype": "f32", "shape": ["Z", "kv_numhead", "KV_CTX", "HEAD_DIM"], "layout": "row_major"},
                "V": {"dtype": "f32", "shape": ["Z", "kv_numhead", "KV_CTX", "HEAD_DIM"], "layout": "row_major"},
                "attn_mask": {"dtype": "f32", "shape": ["Z", "q_numhead", "Q_CTX", "KV_CTX"], "layout": "row_major"},
                "sm_scale": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["Z", "q_numhead", "Q_CTX", "HEAD_DIM"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["Q", "K"], "output": "tmp0", "attrs": {}},
                {"op": "add", "inputs": ["tmp0", "V"], "output": "tmp1", "attrs": {}},
                {"op": "mul", "inputs": ["tmp1", "sm_scale"], "output": "tmp2", "attrs": {}},
                {"op": "add", "inputs": ["tmp2", "attn_mask"], "output": "Out", "attrs": {}},
            ],
            "outputs": ["Out"],
        }
    )


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


def _softmax_inner_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "softmax_inner",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "maxv": {"dtype": "f32", "shape": ["M", 1], "layout": "row_major"},
                "shifted": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "expv": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sumv": {"dtype": "f32", "shape": ["M", 1], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_max", "inputs": ["input"], "output": "maxv", "attrs": {"dims": [1]}},
                {"op": "sub", "inputs": ["input", "maxv"], "output": "shifted", "attrs": {}},
                {"op": "exp", "inputs": ["shifted"], "output": "expv", "attrs": {}},
                {"op": "reduce_sum", "inputs": ["expv"], "output": "sumv", "attrs": {"dims": [1]}},
                {"op": "div", "inputs": ["expv", "sumv"], "output": "output", "attrs": {}},
            ],
            "outputs": ["output"],
        }
    )


def _masked_softmax2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "masked_softmax2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "i32", "shape": ["N"], "layout": "row_major"},
                "maxv": {"dtype": "f32", "shape": ["M", 1], "layout": "row_major"},
                "sumv": {"dtype": "f32", "shape": ["M", 1], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_max", "inputs": ["inp"], "output": "maxv", "attrs": {"dims": [1]}},
                {"op": "reduce_sum", "inputs": ["inp"], "output": "sumv", "attrs": {"dims": [1]}},
                {"op": "identity", "inputs": ["inp"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out"],
        }
    )


def test_cuda_kernel_kind_override_attn_fwd_forces_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    monkeypatch.delenv("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V1", raising=False)
    monkeypatch.delenv("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V2", raising=False)

    intent = _attn_fwd_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64}
    mod.meta["intentir_kernel_kind_override"] = "attn_fwd_softmax_v2"

    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "attn_fwd_softmax_v2"
    assert "gpu.barrier" not in out.module_text
    assert "gpu.shuffle xor" in out.module_text
    _verify_with_mlir_opt(out.module_text)


def test_cuda_kernel_kind_override_matmul_forces_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")

    intent = _ai_bench_matmul_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 64, "N": 16, "K": 16}
    mod.meta["intentir_kernel_kind_override"] = "matmul_tile_v1"

    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "matmul_tile_v1"
    assert "vector<4xf32>" not in out.module_text
    _verify_with_mlir_opt(out.module_text)


def test_cuda_kernel_kind_override_softmax_inner_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _softmax_inner_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 4, "N": 64, "SOFTMAX_BLOCK_THREADS": 64}
    mod.meta["intentir_kernel_kind_override"] = "row_softmax_axis1_triton_v1"
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "row_softmax_axis1_triton_v1"
    assert "vector.from_elements" in out.module_text


def test_cuda_kernel_kind_override_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")

    intent = _ai_bench_matmul_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 64, "N": 16, "K": 16}
    mod.meta["intentir_kernel_kind_override"] = "matmul_tile_v999"

    with pytest.raises(RuntimeError, match="invalid intentir_kernel_kind_override"):
        _ = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
