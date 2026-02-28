from __future__ import annotations

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir
from intent_ir.mlir.passes.lower_intent_to_cuda_gpu_kernel import lower_intent_to_cuda_gpu_kernel


def _logspace1d_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "logspace1d",
            "tensors": {
                "start": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "end": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "denom": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "log_base": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "idx": {"dtype": "i32", "shape": ["N"], "layout": "row_major"},
                "idx_f": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "delta": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "step": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "scaled": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "lin": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "exp_arg": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "iota", "inputs": [], "output": "idx", "attrs": {"axis": 0, "shape": ["N"], "dtype": "i32"}},
                {"op": "cast", "inputs": ["idx"], "output": "idx_f", "attrs": {"to": "f32"}},
                {"op": "sub", "inputs": ["end", "start"], "output": "delta", "attrs": {}},
                {"op": "div", "inputs": ["delta", "denom"], "output": "step", "attrs": {}},
                {"op": "mul", "inputs": ["idx_f", "step"], "output": "scaled", "attrs": {}},
                {"op": "add", "inputs": ["start", "scaled"], "output": "lin", "attrs": {}},
                {"op": "mul", "inputs": ["lin", "log_base"], "output": "exp_arg", "attrs": {}},
                {"op": "exp", "inputs": ["exp_arg"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out"],
        }
    )


def test_cuda_real_mlir_logspace1d_sets_small_launch_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    intent = _logspace1d_intent()
    mod = to_mlir(intent)
    mod.meta["shape_bindings"] = {"N": 64}
    out = lower_intent_to_cuda_gpu_kernel(mod, backend="cuda")
    launch = dict(out.meta.get("cuda_real_mlir_launch_override") or {})
    assert launch.get("block") == [64, 1, 1]
    assert launch.get("grid") == [1, 1, 1]

