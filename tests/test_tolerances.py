import numpy as np

from intent_ir.ir import IntentFunction
from verify.tolerances import infer_tolerances


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


def test_infer_tolerances_matmul_stays_legacy():
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
    assert np.isclose(tol["atol"], 1e-3)
    assert np.isclose(tol["rtol"], 1e-3)

