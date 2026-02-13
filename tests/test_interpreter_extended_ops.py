from __future__ import annotations

import math
import numpy as np

from intent_ir.ir import IntentFunction
from verify.interpreter import execute_intent


def test_extended_unary_ops_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "extended_unary",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "a": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "b": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "c": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "neg", "inputs": ["x"], "output": "x_neg"},
                {"op": "ceil", "inputs": ["x_neg"], "output": "x_ceil"},
                {"op": "sqrt", "inputs": ["x"], "output": "x_sqrt"},
                {"op": "log", "inputs": ["x"], "output": "x_log"},
                {"op": "sin", "inputs": ["x"], "output": "x_sin"},
                {"op": "cos", "inputs": ["x"], "output": "x_cos"},
                {"op": "tan", "inputs": ["x"], "output": "x_tan"},
                {"op": "erf", "inputs": ["x"], "output": "x_erf"},
                {"op": "add", "inputs": ["x_sqrt", "x_log"], "output": "a"},
                {"op": "add", "inputs": ["x_sin", "x_cos"], "output": "b"},
                {"op": "add", "inputs": ["x_tan", "x_erf"], "output": "c"},
            ],
            "outputs": ["a", "b", "c"],
        }
    )
    x = np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32)
    out = execute_intent(intent, {"x": x}, shape_bindings={"M": 2, "N": 2})

    assert np.allclose(out["a"], np.sqrt(x) + np.log(x), atol=1e-6)
    assert np.allclose(out["b"], np.sin(x) + np.cos(x), atol=1e-6)
    assert np.allclose(out["c"], np.tan(x) + np.vectorize(math.erf)(x), atol=1e-6)


def test_extended_reduction_ops_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "extended_reduce",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mn": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "pd": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "am": {"dtype": "i64", "shape": ["M"], "layout": "row_major"},
                "cs": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_min", "inputs": ["x"], "output": "mn", "attrs": {"dims": [1]}},
                {"op": "reduce_prod", "inputs": ["x"], "output": "pd", "attrs": {"dims": [1]}},
                {"op": "argmax", "inputs": ["x"], "output": "am", "attrs": {"axis": 1}},
                {"op": "cumsum", "inputs": ["x"], "output": "cs", "attrs": {"axis": 1}},
            ],
            "outputs": ["mn", "pd", "am", "cs"],
        }
    )
    x = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 8.0]], dtype=np.float32)
    out = execute_intent(intent, {"x": x}, shape_bindings={"M": 2, "N": 3})

    assert np.allclose(out["mn"], np.min(x, axis=1), atol=1e-6)
    assert np.allclose(out["pd"], np.prod(x, axis=1), atol=1e-6)
    assert np.array_equal(out["am"], np.argmax(x, axis=1))
    assert np.allclose(out["cs"], np.cumsum(x, axis=1), atol=1e-6)
