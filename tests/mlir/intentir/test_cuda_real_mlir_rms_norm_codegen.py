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


def _liger_rms_norm_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "liger_rms_norm",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "W": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "offset": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "tmp0": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "RSTD": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "eps", "attrs": {"dtype": "f32", "value": 1e-5}},
                {"op": "const", "inputs": [], "output": "offset", "attrs": {"dtype": "f32", "value": 0.0}},
                {"op": "add", "inputs": ["X", "eps"], "output": "tmp0", "attrs": {}},
                {"op": "mul", "inputs": ["tmp0", "W"], "output": "Y", "attrs": {}},
            ],
            "outputs": ["Y", "RSTD"],
        }
    )


def _liger_fused_add_rms_norm_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "liger_fused_add_rms_norm",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "R": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "W": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "offset": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "S": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "S_squared": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "sum_squares": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "n_cols_const": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "mean_square": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "variance_eps": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "RSTD": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "rstd_broadcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "S_normalized": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "W_broadcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "offset_broadcast": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "weight_offset": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "eps", "attrs": {"dtype": "f32", "value": 1e-5}},
                {"op": "const", "inputs": [], "output": "offset", "attrs": {"dtype": "f32", "value": 0.0}},
                {"op": "add", "inputs": ["X", "R"], "output": "S", "attrs": {}},
                {"op": "mul", "inputs": ["S", "S"], "output": "S_squared", "attrs": {}},
                {"op": "reduce_sum", "inputs": ["S_squared"], "output": "sum_squares", "attrs": {"dims": [1]}},
                {"op": "const", "inputs": [], "output": "n_cols_const", "attrs": {"dtype": "f32", "value": "N"}},
                {"op": "div", "inputs": ["sum_squares", "n_cols_const"], "output": "mean_square", "attrs": {}},
                {"op": "add", "inputs": ["mean_square", "eps"], "output": "variance_eps", "attrs": {}},
                {"op": "rsqrt", "inputs": ["variance_eps"], "output": "RSTD", "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["RSTD"],
                    "output": "rstd_broadcast",
                    "attrs": {"broadcast_dims": [0], "out_shape": ["M", "N"]},
                },
                {"op": "mul", "inputs": ["S", "rstd_broadcast"], "output": "S_normalized", "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["W"],
                    "output": "W_broadcast",
                    "attrs": {"broadcast_dims": [1], "out_shape": ["M", "N"]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["offset"],
                    "output": "offset_broadcast",
                    "attrs": {"broadcast_dims": [], "out_shape": ["M", "N"]},
                },
                {"op": "add", "inputs": ["offset_broadcast", "W_broadcast"], "output": "weight_offset", "attrs": {}},
                {"op": "mul", "inputs": ["S_normalized", "weight_offset"], "output": "Y", "attrs": {}},
            ],
            "outputs": ["Y", "S", "RSTD"],
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


def test_cuda_real_mlir_unknown_rms_norm_accepts_kernel_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _liger_rms_norm_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    mod.meta["intentir_kernel_kind_override"] = "rms_norm_axis1_v3"
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "rms_norm_axis1_v3"
    _verify_with_mlir_opt(out.module_text)


def test_cuda_real_mlir_unknown_fused_add_rms_norm_accepts_kernel_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _liger_fused_add_rms_norm_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"M": 4, "N": 64}
    mod.meta["intentir_kernel_kind_override"] = "rms_norm_axis1_v3"
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "rms_norm_axis1_v3"
    assert "memref.store %s0" in out.module_text
    assert "memref.store %rstd_v" in out.module_text
    _verify_with_mlir_opt(out.module_text)


def test_cuda_real_mlir_unknown_rms_norm_accepts_full_row_kernel_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _liger_rms_norm_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {
        "M": 4,
        "N": 256,
        "RMS_NORM_BLOCK_THREADS": 64,
        "RMS_NORM_VECTOR_WIDTH": 4,
        "RMS_NORM_FULL_ROW_VECTOR": 1,
    }
    mod.meta["intentir_kernel_kind_override"] = "rms_norm_axis1_v4"
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "rms_norm_axis1_v4"
    assert "vector.load" in out.module_text
    assert "full_row_x_" in out.module_text
    assert "scf.for %jb =" not in out.module_text
    _verify_with_mlir_opt(out.module_text)


def test_cuda_real_mlir_unknown_fused_add_rms_norm_accepts_full_row_kernel_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _liger_fused_add_rms_norm_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {
        "M": 4,
        "N": 256,
        "RMS_NORM_BLOCK_THREADS": 64,
        "RMS_NORM_VECTOR_WIDTH": 4,
        "RMS_NORM_FULL_ROW_VECTOR": 1,
    }
    mod.meta["intentir_kernel_kind_override"] = "rms_norm_axis1_v4"
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == "rms_norm_axis1_v4"
    assert "full_row_s_" in out.module_text
    assert "offsetv = vector.broadcast %offset" in out.module_text
    assert "scf.for %jb =" not in out.module_text
    _verify_with_mlir_opt(out.module_text)
