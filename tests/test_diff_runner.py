import numpy as np
import pytest

from intent_ir.ir import IntentFunction
from verify.interpreter import execute_intent
from verify.gen_cases import generate_cases, TestCase
from verify.diff_runner import run_diff


def _intent_matmul_bias_relu():
    js = {
        "name": "mm",
        "tensors": {
            "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
            "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
            "bias": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
        },
        "ops": [
            {"op": "matmul", "inputs": ["A", "B"], "output": "Y"},
            {
                "op": "broadcast_in_dim",
                "inputs": ["bias"],
                "output": "bias2d",
                "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
            },
            {"op": "add", "inputs": ["Y", "bias2d"], "output": "Z"},
            {"op": "relu", "inputs": ["Z"], "output": "C"},
        ],
        "outputs": ["C"],
    }
    return IntentFunction.from_json_dict(js)


def test_interpreter_gemm_basic():
    intent = _intent_matmul_bias_relu()
    M, K, N = 2, 3, 4
    A = np.ones((M, K), dtype=np.float32)
    B = np.ones((K, N), dtype=np.float32)
    bias = np.ones((N,), dtype=np.float32)
    out = execute_intent(intent, {"A": A, "B": B, "bias": bias})
    assert out["C"].shape == (M, N)
    assert np.all(out["C"] == (K + 1.0))


def test_casegen_produces_edge_cases_when_mask_needed():
    intent = _intent_matmul_bias_relu()
    cases = generate_cases(intent, constraints=None, limit=3, seed=0)
    assert len(cases) == 3
    # shapes should contain symbolic dims
    for c in cases:
        assert "M" in c.shapes and "N" in c.shapes and "K" in c.shapes


def test_diff_runner_reports_failure():
    intent = _intent_matmul_bias_relu()

    def ref_fn(case: TestCase):
        M, K, N = case.shapes["M"], case.shapes["K"], case.shapes["N"]
        A = np.ones((M, K), dtype=np.float32)
        B = np.ones((K, N), dtype=np.float32)
        bias = np.ones((N,), dtype=np.float32)
        C = np.full((M, N), fill_value=999.0, dtype=np.float32)
        return {"A": A, "B": B, "bias": bias, "C": C}

    cases = [TestCase(shapes={"M": 2, "K": 2, "N": 2}, seed=0, dtypes={})]
    diffs, cex = run_diff(intent, ref_fn, cases)
    assert len(diffs) == 1
    assert not diffs[0].ok
    assert cex, "should produce counterexample when mismatch occurs"


def test_interpreter_softmax_row_broadcast_from_reduce_vector():
    # Regression: disambiguate broadcasting of a reduce result [M] with a 2D tensor [M,N].
    # NumPy aligns 1D on the trailing axis; IntentIR uses shape symbols to mean "row-wise".
    js = {
        "name": "softmax_like",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "mx": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            "Xc": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "E": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "sm": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
        },
        "ops": [
            {"op": "reduce_max", "inputs": ["X"], "output": "mx", "attrs": {"axis": [1], "keepdims": False}},
            {"op": "sub", "inputs": ["X", "mx"], "output": "Xc"},
            {"op": "exp", "inputs": ["Xc"], "output": "E"},
            {"op": "reduce_sum", "inputs": ["E"], "output": "sm", "attrs": {"axis": [1], "keepdims": False}},
            {"op": "div", "inputs": ["E", "sm"], "output": "out"},
        ],
        "outputs": ["out"],
    }
    intent = IntentFunction.from_json_dict(js)
    X = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = execute_intent(intent, {"X": X})["out"]
    mx = np.max(X, axis=1, keepdims=True)
    E = np.exp(X - mx)
    expected = E / np.sum(E, axis=1, keepdims=True)
    assert np.allclose(out, expected, atol=1e-6, rtol=1e-6)
