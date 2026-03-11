from __future__ import annotations

from intent_ir.ir import IntentFunction


def test_intent_regions_roundtrip() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "cfg_blueprint",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [{"op": "identity", "inputs": ["x"], "output": "out"}],
            "outputs": ["out"],
            "regions": [
                {
                    "id": "r_entry",
                    "kind": "if",
                    "predicate": "x[i] > 0",
                    "path_id": "pi_pos",
                    "inputs": ["x", "y"],
                    "outputs": ["out"],
                    "ops": [{"op": "identity", "inputs": ["x"], "output": "out"}],
                    "regions": [
                        {
                            "id": "r_then",
                            "kind": "then",
                            "path_id": "pi_pos",
                            "inputs": ["x"],
                            "outputs": ["out"],
                            "ops": [{"op": "identity", "inputs": ["x"], "output": "out"}],
                        },
                        {
                            "id": "r_else",
                            "kind": "else",
                            "path_id": "pi_neg",
                            "inputs": ["y"],
                            "outputs": ["out"],
                            "ops": [{"op": "identity", "inputs": ["y"], "output": "out"}],
                        },
                    ],
                }
            ],
        }
    )
    out = intent.to_json_dict()
    assert out["regions"][0]["kind"] == "if"
    assert out["regions"][0]["regions"][0]["path_id"] == "pi_pos"
    assert out["regions"][0]["regions"][1]["kind"] == "else"
