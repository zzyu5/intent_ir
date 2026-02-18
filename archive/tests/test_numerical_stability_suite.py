import numpy as np

from intent_ir.ir import IntentFunction
from verify.gen_cases import TestCase
from verify.interpreter import execute_intent
from verify.numerical_stability import run_numerical_stability_suite


def test_numerical_stability_suite_runs_on_simple_kernel():
    intent = IntentFunction.from_json_dict(
        {
            "name": "relu1d",
            "tensors": {"x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"}, "y": {"dtype": "f32", "shape": ["N"], "layout": "row_major"}},
            "ops": [{"op": "relu", "inputs": ["x"], "output": "y"}],
            "outputs": ["y"],
        }
    )

    def run_ref(case: TestCase):
        x = np.zeros((int(case.shapes["N"]),), dtype=np.float32)
        out = execute_intent(intent, {"x": x}, shape_bindings=dict(case.shapes))
        # return both inputs + outputs
        return {"x": x, **out}

    rep = run_numerical_stability_suite("relu1d", intent, run_ref_fn=run_ref, base_case=TestCase(shapes={"N": 8}, dtypes={}, seed=0))
    assert rep.skipped is False
    assert rep.results
    assert rep.ok is True

