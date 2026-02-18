from __future__ import annotations

from backends.common.mlir_bridge import resolve_intent_payload
from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir


def _intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "relu1d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [{"op": "relu", "inputs": ["x"], "output": "y", "attrs": {}}],
            "outputs": ["y"],
        }
    )


def test_resolve_intent_payload_accepts_mlir_module() -> None:
    intent = _intent()
    module = to_mlir(intent)
    resolved = resolve_intent_payload(module)
    assert isinstance(resolved, IntentFunction)
    assert resolved.name == intent.name


def test_resolve_intent_payload_accepts_mlir_text_payload() -> None:
    intent = _intent()
    module = to_mlir(intent)
    resolved = resolve_intent_payload(module.module_text)
    assert isinstance(resolved, IntentFunction)
    assert resolved.outputs == intent.outputs

