from __future__ import annotations

import numpy as np

from intent_ir.ir import IntentFunction
from intent_ir.parser import normalize_candidate_json
from verify.interpreter import execute_intent


def test_normalize_candidate_json_materializes_symbolic_and_optional_scalars() -> None:
    normalized = normalize_candidate_json(
        {
            "name": "liger_rms_norm",
            "tensors": {
                "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "W": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "RSTD": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "eps": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "n_cols": {"dtype": "i32", "shape": [], "layout": "row_major"},
                "offset": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "mul", "inputs": ["X", "X"], "output": "X_squared"},
                {"op": "reduce_sum", "inputs": ["X_squared"], "output": "sum_squares", "attrs": {"dims": [1]}},
                {"op": "cast", "inputs": ["n_cols"], "output": "n_cols_f32", "attrs": {"to": "f32"}},
                {"op": "div", "inputs": ["sum_squares", "n_cols_f32"], "output": "mean_square"},
                {"op": "add", "inputs": ["mean_square", "eps"], "output": "variance_eps"},
                {"op": "rsqrt", "inputs": ["variance_eps"], "output": "rstd_computed"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["offset"],
                    "output": "offset_bc",
                    "attrs": {"out_shape": ["N"], "broadcast_dims": []},
                },
                {"op": "add", "inputs": ["offset_bc", "W"], "output": "weight_scaled"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["weight_scaled"],
                    "output": "weight_bc",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["rstd_computed"],
                    "output": "rstd_bc",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [0]},
                },
                {"op": "mul", "inputs": ["X", "rstd_bc"], "output": "X_norm"},
                {"op": "mul", "inputs": ["X_norm", "weight_bc"], "output": "Y"},
                {"op": "identity", "inputs": ["rstd_computed"], "output": "RSTD"},
            ],
            "outputs": ["Y", "RSTD"],
            "meta": {
                "access_witness": {
                    "axis_contig_len": {"n_cols": 1024},
                }
            },
        }
    )

    prefix_consts = [op for op in normalized["ops"] if op.get("op") == "const"]
    const_by_output = {str(op["output"]): op for op in prefix_consts}
    assert const_by_output["n_cols"]["attrs"]["value"] == "N"
    assert const_by_output["offset"]["attrs"]["value"] == 0.0


def test_normalize_candidate_json_canonicalizes_mul_scale_attr_for_execution() -> None:
    normalized = normalize_candidate_json(
        {
            "name": "liger_swiglu",
            "tensors": {
                "a": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "b": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "c": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "const_one", "attrs": {"value": 1.0, "dtype": "f32"}},
                {"op": "mul", "inputs": ["a", "const_one"], "output": "neg_a", "attrs": {"scale": -1.0}},
                {"op": "exp", "inputs": ["neg_a"], "output": "exp_neg_a"},
                {"op": "add", "inputs": ["exp_neg_a", "const_one"], "output": "exp_plus_one"},
                {"op": "div", "inputs": ["const_one", "exp_plus_one"], "output": "sigmoid_a"},
                {"op": "mul", "inputs": ["a", "sigmoid_a"], "output": "silu_a"},
                {"op": "mul", "inputs": ["silu_a", "b"], "output": "c"},
            ],
            "outputs": ["c"],
        }
    )

    assert all("scale" not in dict(op.get("attrs") or {}) for op in normalized["ops"] if op.get("op") == "mul")
    intent = IntentFunction.from_json_dict(normalized)
    a = np.array([[1.0, -2.0], [0.5, -0.25]], dtype=np.float32)
    b = np.array([[3.0, 4.0], [2.0, -1.0]], dtype=np.float32)
    out = execute_intent(intent, {"a": a, "b": b}, shape_bindings={"M": 2, "N": 2})
    expected = (a * (1.0 / (1.0 + np.exp(-a)))) * b
    assert np.allclose(out["c"], expected, atol=1e-6)
