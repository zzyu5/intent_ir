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


def _prod2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "prod2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [{"op": "reduce_prod", "inputs": ["inp"], "output": "out", "attrs": {"dims": [0, 1]}}],
            "outputs": ["out"],
        }
    )


def _stack2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "stack2d",
            "tensors": {
                "input0": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "input1": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": [2, "M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "stack", "inputs": ["input0", "input1"], "output": "out_ptr", "attrs": {"axis": 0}}],
            "outputs": ["out_ptr"],
        }
    )


def _polar2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "polar2d",
            "tensors": {
                "abs": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "angle": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N", 2], "layout": "row_major"},
            },
            "ops": [{"op": "polar", "inputs": ["abs", "angle"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _diag_embed2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "diag_embed2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["B", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["B", "N", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["x"],
                    "output": "y",
                    "attrs": {"broadcast_dims": [0, 2], "out_shape": ["B", "N", "N"]},
                }
            ],
            "outputs": ["y"],
        }
    )


def _upsample_nearest1d_ncl_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "upsample_nearest1d_ncl",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "IL"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OL"], "layout": "row_major"},
            },
            "ops": [{"op": "upsample_nearest1d", "inputs": ["input"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _upsample_nearest2d_nchw_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "upsample_nearest2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "IH", "IW"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [{"op": "upsample_nearest2d", "inputs": ["input"], "output": "out"}],
            "outputs": ["out"],
        }
    )


def _glu2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "glu2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N_HALF"], "layout": "row_major"},
            },
            "ops": [{"op": "glu", "inputs": ["x"], "output": "out", "attrs": {"axis": 1}}],
            "outputs": ["out"],
        }
    )


def _log_softmax2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "log_softmax2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "softmax", "inputs": ["inp"], "output": "softmax_out", "attrs": {"axis": 1}},
                {"op": "log", "inputs": ["softmax_out"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )


@pytest.mark.parametrize(
    "intent_fn,shape_bindings,expected_kind,needle",
    [
        (_prod2d_intent, {"M": 4, "N": 64}, "reduce_prod_all_v1", "cS_prodall_128"),
        (_stack2d_intent, {"M": 4, "N": 16}, "stack2d_v1", "arith.divui %lin, %cPlane"),
        (_polar2d_intent, {"M": 8, "N": 16}, "polar2d_v1", "math.cos"),
        (_diag_embed2d_intent, {"B": 2, "N": 8}, "diag_embed2d_v1", "arith.cmpi eq, %ii, %jj"),
        (_upsample_nearest1d_ncl_intent, {"N": 2, "C": 3, "IL": 8, "OL": 16}, "upsample_nearest1d_ncl_v1", "%ol_mul"),
        (_upsample_nearest2d_nchw_intent, {"N": 1, "C": 2, "IH": 8, "IW": 8, "OH": 16, "OW": 16}, "upsample_nearest2d_nchw_v1", "%cIHW"),
        (_glu2d_intent, {"M": 4, "N": 64, "N_HALF": 32}, "glu2d_v1", "arith.divf %c1f, %den"),
        (_log_softmax2d_intent, {"M": 4, "N": 64}, "row_log_softmax_axis1_v1", "math.log"),
    ],
)
def test_cuda_real_mlir_wave13_codegen_and_is_parseable(
    monkeypatch: pytest.MonkeyPatch,
    intent_fn,
    shape_bindings: dict[str, int],
    expected_kind: str,
    needle: str,
) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = intent_fn()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = dict(shape_bindings)
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    assert str(out.meta.get("cuda_real_mlir_kernel_kind") or "") == expected_kind
    assert str(needle) in out.module_text
    _verify_with_mlir_opt(out.module_text)

