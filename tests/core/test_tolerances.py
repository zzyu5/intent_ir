import numpy as np

from intent_ir.ir import IntentFunction
from verify.tolerances import infer_tolerances, infer_tolerances_from_intent_json


def test_infer_tolerances_softmax_is_tighter_than_legacy():
    intent = IntentFunction.from_json_dict(
        {
            "name": "softmax_only",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "softmax", "inputs": ["x"], "output": "y", "attrs": {"axis": 1, "stable": True}}],
            "outputs": ["y"],
        }
    )
    tol = infer_tolerances(intent).to_dict()
    assert tol["atol"] < 1e-3
    assert tol["rtol"] < 1e-3


def test_infer_tolerances_matmul_is_looser_than_legacy():
    intent = IntentFunction.from_json_dict(
        {
            "name": "matmul_only",
            "tensors": {
                "A": {"dtype": "f16", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f16", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {"accum_dtype": "f32"}}],
            "outputs": ["C"],
        }
    )
    tol = infer_tolerances(intent).to_dict()
    assert tol["atol"] > 1e-3
    assert tol["rtol"] > 1e-3


def test_infer_tolerances_bf16_minimum_is_looser_than_legacy():
    intent = IntentFunction.from_json_dict(
        {
            "name": "bf16_minimum",
            "tensors": {
                "X": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "Y": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "bf16", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "min", "inputs": ["X", "Y"], "output": "Out"}],
            "outputs": ["Out"],
        }
    )
    tol = infer_tolerances(intent).to_dict()
    assert tol["atol"] >= 1e-2
    assert tol["rtol"] >= 1e-2


def test_infer_tolerances_from_intent_json_matches_intent_path():
    intent_json = {
        "name": "softmax_only_json",
        "tensors": {
            "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
        },
        "ops": [{"op": "softmax", "inputs": ["x"], "output": "y", "attrs": {"axis": 1, "stable": True}}],
        "outputs": ["y"],
    }
    ref = {"y": np.ones((2, 2), dtype=np.float32)}

    tol_json = infer_tolerances_from_intent_json(intent_json, ref_out=ref).to_dict()
    tol_intent = infer_tolerances(IntentFunction.from_json_dict(dict(intent_json)), ref_out=ref).to_dict()

    assert tol_json == tol_intent
