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


def _logical_or2d_intent() -> IntentFunction:
    # Many FlagGems logical_or extracted intents represent OR as:
    #   cast -> max(i1, i1)  (max == OR)
    # This test exercises i1 max lowering under real-MLIR CUDA.
    return IntentFunction.from_json_dict(
        {
            "name": "logical_or2d",
            "tensors": {
                "A": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "max", "inputs": ["A", "B"], "output": "Out", "attrs": {}}],
            "outputs": ["Out"],
        }
    )


def test_cuda_real_mlir_logical_or_i1_max_lowers_to_ori(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _logical_or2d_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert "arith.ori" in out.module_text
    _verify_with_mlir_opt(out.module_text)

