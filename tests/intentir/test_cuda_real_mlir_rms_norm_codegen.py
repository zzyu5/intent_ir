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


def _rms_norm2d_flaggems_style_intent() -> IntentFunction:
    # FlagGems rms_norm commonly uses tensor names like input/weight/Y/INV_RMS,
    # and provides eps as a scalar ABI tensor (shape=[]).
    return IntentFunction.from_json_dict(
        {
            "name": "rms_norm2d",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "tmp0": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "INV_RMS": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["input", "eps"], "output": "tmp0", "attrs": {}},
                {"op": "mul", "inputs": ["tmp0", "weight"], "output": "Y", "attrs": {}},
            ],
            "outputs": ["Y", "INV_RMS"],
        }
    )


def test_cuda_real_mlir_rms_norm_accepts_eps_tensor_and_emits_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _rms_norm2d_flaggems_style_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "rms_norm_axis1_v3"
    assert "memref.load %eps[%c0]" in out.module_text
    _verify_with_mlir_opt(out.module_text)
