import json

import numpy as np

from intent_ir.ir import IntentFunction
from verify.diff_debugger import debug_mismatch
from verify.gen_cases import TestCase
from verify.interpreter import execute_intent, execute_intent_with_trace


def _intent_json():
    return {
        "name": "gemm_bias_relu",
        "tensors": {
            "A": {"dtype": "f16", "shape": ["M", "K"], "layout": "row_major"},
            "B": {"dtype": "f16", "shape": ["K", "N"], "layout": "row_major"},
            "bias": {"dtype": "f16", "shape": ["N"], "layout": "row_major"},
            "C": {"dtype": "f16", "shape": ["M", "N"], "layout": "row_major"},
        },
        "ops": [
            {"op": "matmul", "inputs": ["A", "B"], "output": "Y", "attrs": {"accum_dtype": "f32"}},
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
        "parallel_axes": ["M", "N"],
    }


def test_execute_intent_with_trace_matches_execute_intent():
    intent = IntentFunction.from_json_dict(_intent_json())
    shapes = {"M": 2, "N": 3, "K": 4}
    rng = np.random.default_rng(0)
    inputs = {
        "A": rng.standard_normal((shapes["M"], shapes["K"]), dtype=np.float32).astype(np.float16),
        "B": rng.standard_normal((shapes["K"], shapes["N"]), dtype=np.float32).astype(np.float16),
        "bias": rng.standard_normal((shapes["N"],), dtype=np.float32).astype(np.float16),
    }
    out0 = execute_intent(intent, inputs, shape_bindings=shapes)
    out1, trace, env = execute_intent_with_trace(intent, inputs, shape_bindings=shapes, sample_elems=8)
    assert np.allclose(out0["C"], out1["C"])
    assert len(trace.op_traces) == len(intent.ops)
    assert env["C"].shape == out0["C"].shape


def test_debug_mismatch_report_is_json_serializable():
    intent = IntentFunction.from_json_dict(_intent_json())
    case = TestCase(shapes={"M": 2, "N": 3, "K": 4}, dtypes={}, seed=0)

    def run_ref_fn(c: TestCase):
        rng = np.random.default_rng(int(c.seed))
        M, N, K = int(c.shapes["M"]), int(c.shapes["N"]), int(c.shapes["K"])
        A = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
        B = rng.standard_normal((K, N), dtype=np.float32).astype(np.float16)
        bias = rng.standard_normal((N,), dtype=np.float32).astype(np.float16)
        ref = execute_intent(intent, {"A": A, "B": B, "bias": bias}, shape_bindings=dict(c.shapes))
        # Intentionally perturb the final output to simulate a mismatch.
        ref["C"] = ref["C"] + np.float16(1.0)
        return {"A": A, "B": B, "bias": bias, **ref}

    report = debug_mismatch(intent, run_ref_fn, case, tolerances={"atol": 0.0, "rtol": 0.0}, sample_elems=4)
    assert report["ok"] is False
    json.dumps(report)

