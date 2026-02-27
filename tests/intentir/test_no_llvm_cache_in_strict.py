from __future__ import annotations

import importlib

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir

_LOWER = importlib.import_module("intent_ir.mlir.passes.lower_intent_to_llvm_dialect")


def _sample_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}}],
            "outputs": ["z"],
        }
    )


def test_lower_intent_to_llvm_dialect_forbids_cached_llvm_in_real_mlir_strict(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "strict")

    mod = to_mlir(_sample_intent())
    mod.meta = dict(mod.meta or {})
    mod.meta["prelowered_llvm_ir_text"] = '; ModuleID = "cached"\ndefine void @k() { ret void }\n'

    with pytest.raises(RuntimeError, match="cached LLVM IR is forbidden"):
        _ = _LOWER.lower_intent_to_llvm_dialect(mod, backend="cuda")

