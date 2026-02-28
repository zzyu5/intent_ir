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


def _flash_attention2d_intent() -> IntentFunction:
    # Minimal intent: only needs to surface the expected tensor interface so the
    # lowering pattern matcher triggers.
    return IntentFunction.from_json_dict(
        {
            "name": "flash_attention2d",
            "tensors": {
                "Q": {"dtype": "f32", "shape": ["Q_CTX", "HEAD_DIM"], "layout": "row_major"},
                "K": {"dtype": "f32", "shape": ["KV_CTX", "HEAD_DIM"], "layout": "row_major"},
                "V": {"dtype": "f32", "shape": ["KV_CTX", "HEAD_DIM"], "layout": "row_major"},
                "sm_scale": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["Q_CTX", "HEAD_DIM"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["Q", "K"], "output": "tmp0", "attrs": {}},
                {"op": "mul", "inputs": ["tmp0", "sm_scale"], "output": "tmp1", "attrs": {}},
                {"op": "add", "inputs": ["tmp1", "V"], "output": "Out", "attrs": {}},
            ],
            "outputs": ["Out"],
        }
    )


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


@pytest.mark.parametrize(
    ("intent_factory", "shape_bindings"),
    [
        (_flash_attention2d_intent, {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64}),
        (_attn_fwd_intent, {"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64}),
    ],
)
def test_cuda_real_mlir_attention_emits_shuffle_and_no_barrier(
    monkeypatch: pytest.MonkeyPatch, intent_factory, shape_bindings: dict[str, int]
) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    monkeypatch.delenv("INTENTIR_CUDA_REAL_MLIR_ATTN_V1", raising=False)
    monkeypatch.delenv("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V1", raising=False)
    # `_attn_fwd` defaults to a tiled v3 implementation for perf; keep v2 (warp-only)
    # as a debug toggle and exercise it here to assert "no barrier" + "shuffle xor".
    monkeypatch.setenv("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V2", "1")
    intent = intent_factory()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = dict(shape_bindings)
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert "gpu.barrier" not in out.module_text
    assert "gpu.shuffle xor" in out.module_text
    _verify_with_mlir_opt(out.module_text)
