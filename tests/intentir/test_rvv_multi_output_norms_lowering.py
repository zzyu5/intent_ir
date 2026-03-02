from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_mlir
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


def _rms_norm2d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "rms_norm2d",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "N_scalar": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "INV_RMS": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "mul", "inputs": ["input", "input"], "output": "x_sq", "attrs": {}},
                {"op": "reduce_sum", "inputs": ["x_sq"], "output": "sum_sq", "attrs": {"dims": [1], "keepdims": False}},
                {"op": "div", "inputs": ["sum_sq", "N_scalar"], "output": "mean_sq", "attrs": {}},
                {"op": "add", "inputs": ["mean_sq", "eps"], "output": "var_eps", "attrs": {}},
                {"op": "rsqrt", "inputs": ["var_eps"], "output": "INV_RMS", "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["INV_RMS"],
                    "output": "inv_rms_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["weight"],
                    "output": "w_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {"op": "mul", "inputs": ["input", "inv_rms_bcast"], "output": "x_norm", "attrs": {}},
                {"op": "mul", "inputs": ["x_norm", "w_bcast"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out", "INV_RMS"],
        }
    )


def _layer_norm_persistent_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "layer_norm_persistent",
            "tensors": {
                "in_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "weight_ptr": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "bias_ptr": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out_ptr": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out_mean_ptr": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out_rstd_ptr": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "eps", "attrs": {"value": 1e-05, "dtype": "f32"}},
                {"op": "const", "inputs": [], "output": "N_scalar", "attrs": {"value": "N"}},
                {"op": "reduce_sum", "inputs": ["in_ptr"], "output": "sum_x", "attrs": {"dims": [1]}},
                {"op": "div", "inputs": ["sum_x", "N_scalar"], "output": "mean", "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["mean"],
                    "output": "mean_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {"op": "sub", "inputs": ["in_ptr", "mean_bcast"], "output": "deviation", "attrs": {}},
                {"op": "mul", "inputs": ["deviation", "deviation"], "output": "deviation_sq", "attrs": {}},
                {"op": "reduce_sum", "inputs": ["deviation_sq"], "output": "sum_sq", "attrs": {"dims": [1]}},
                {"op": "div", "inputs": ["sum_sq", "N_scalar"], "output": "variance", "attrs": {}},
                {"op": "add", "inputs": ["variance", "eps"], "output": "var_eps", "attrs": {}},
                {"op": "rsqrt", "inputs": ["var_eps"], "output": "rstd", "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["rstd"],
                    "output": "rstd_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {"op": "mul", "inputs": ["deviation", "rstd_bcast"], "output": "normalized", "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["weight_ptr"],
                    "output": "weight_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {"op": "mul", "inputs": ["normalized", "weight_bcast"], "output": "scaled", "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["bias_ptr"],
                    "output": "bias_bcast",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {"op": "add", "inputs": ["scaled", "bias_bcast"], "output": "out_ptr", "attrs": {}},
                {"op": "identity", "inputs": ["mean"], "output": "out_mean_ptr", "attrs": {}},
                {"op": "identity", "inputs": ["rstd"], "output": "out_rstd_ptr", "attrs": {}},
            ],
            "outputs": ["out_ptr", "out_mean_ptr", "out_rstd_ptr"],
        }
    )


def _group_norm_kernel_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "group_norm_kernel",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "W": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "Y": {"dtype": "f32", "shape": ["N", "C", "HW"], "layout": "row_major"},
                "Mean": {"dtype": "f32", "shape": ["N", "num_groups"], "layout": "row_major"},
                "Rstd": {"dtype": "f32", "shape": ["N", "num_groups"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "output": "group_size", "inputs": [], "attrs": {"value": "group_size", "dtype": "i32"}},
                {"op": "const", "output": "HW", "inputs": [], "attrs": {"value": "HW", "dtype": "i32"}},
                {"op": "const", "output": "eps", "inputs": [], "attrs": {"value": 1e-05, "dtype": "f32"}},
                {"op": "reshape", "output": "X_reshaped", "inputs": ["X"], "attrs": {"shape": ["N", "num_groups", "group_size", "HW"]}},
                {"op": "mul", "output": "num_elements", "inputs": ["group_size", "HW"], "attrs": {}},
                {"op": "cast", "output": "X_f32", "inputs": ["X_reshaped"], "attrs": {"to": "f32"}},
                {"op": "reduce_sum", "output": "sum_X", "inputs": ["X_f32"], "attrs": {"dims": [2, 3]}},
                {"op": "div", "output": "mean", "inputs": ["sum_X", "num_elements"], "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "output": "mean_bcast",
                    "inputs": ["mean"],
                    "attrs": {"out_shape": ["N", "num_groups", "group_size", "HW"], "broadcast_dims": [0, 1]},
                },
                {"op": "sub", "output": "x_centered", "inputs": ["X_f32", "mean_bcast"], "attrs": {}},
                {"op": "mul", "output": "x_sq", "inputs": ["x_centered", "x_centered"], "attrs": {}},
                {"op": "reduce_sum", "output": "sum_x_sq", "inputs": ["x_sq"], "attrs": {"dims": [2, 3]}},
                {"op": "div", "output": "var", "inputs": ["sum_x_sq", "num_elements"], "attrs": {}},
                {"op": "add", "output": "var_eps", "inputs": ["var", "eps"], "attrs": {}},
                {"op": "rsqrt", "output": "rstd_computed", "inputs": ["var_eps"], "attrs": {}},
                {
                    "op": "broadcast_in_dim",
                    "output": "rstd_bcast",
                    "inputs": ["rstd_computed"],
                    "attrs": {"out_shape": ["N", "num_groups", "group_size", "HW"], "broadcast_dims": [0, 1]},
                },
                {"op": "mul", "output": "x_hat", "inputs": ["x_centered", "rstd_bcast"], "attrs": {}},
                {"op": "reshape", "output": "W_reshaped", "inputs": ["W"], "attrs": {"shape": ["num_groups", "group_size"]}},
                {
                    "op": "broadcast_in_dim",
                    "output": "W_bcast",
                    "inputs": ["W_reshaped"],
                    "attrs": {"out_shape": ["N", "num_groups", "group_size", "HW"], "broadcast_dims": [1, 2]},
                },
                {"op": "mul", "output": "x_hat_scaled", "inputs": ["x_hat", "W_bcast"], "attrs": {}},
                {"op": "reshape", "output": "B_reshaped", "inputs": ["B"], "attrs": {"shape": ["num_groups", "group_size"]}},
                {
                    "op": "broadcast_in_dim",
                    "output": "B_bcast",
                    "inputs": ["B_reshaped"],
                    "attrs": {"out_shape": ["N", "num_groups", "group_size", "HW"], "broadcast_dims": [1, 2]},
                },
                {"op": "add", "output": "Y_reshaped", "inputs": ["x_hat_scaled", "B_bcast"], "attrs": {}},
                {"op": "reshape", "output": "Y", "inputs": ["Y_reshaped"], "attrs": {"shape": ["N", "C", "HW"]}},
                {"op": "identity", "output": "Mean", "inputs": ["mean"], "attrs": {}},
                {"op": "identity", "output": "Rstd", "inputs": ["rstd_computed"], "attrs": {}},
            ],
            "outputs": ["Y", "Mean", "Rstd"],
        }
    )


@pytest.mark.parametrize(
    ("intent_factory", "shape_bindings", "expected_kind"),
    [
        (_rms_norm2d_intent, {"M": 4, "N": 64}, "rms_norm2d_rvv_v1"),
        (_layer_norm_persistent_intent, {"M": 4, "N": 64}, "layer_norm_rvv_v1"),
        (_group_norm_kernel_intent, {"N": 2, "C": 64, "HW": 16, "num_groups": 8, "group_size": 8}, "group_norm_rvv_v1"),
    ],
)
def test_rvv_multi_output_norms_lowering_emits_parseable_mlir(
    tmp_path: Path, intent_factory, shape_bindings: dict[str, int], expected_kind: str
) -> None:
    intent = intent_factory()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = dict(shape_bindings)

    lowered = lower_intent_to_rvv_cpu_kernel(mod, backend="rvv")
    assert str((lowered.meta or {}).get("rvv_real_mlir_kernel_kind") or "") == expected_kind
    _verify_with_mlir_opt(str(lowered.module_text or ""))

    out_dir = tmp_path / str(intent.name)
    out, trace = run_pipeline(mod, "downstream_rvv_std_llvm", backend="rvv", out_dir=out_dir, fail_on_error=True)
    assert bool(trace.get("ok")) is True
    text = str(out.module_text or "")
    assert "target triple = \"riscv64-unknown-linux-gnu\"" in text
    assert str((out.meta or {}).get("llvm_dialect_origin") or "") == "mlir_translate"

