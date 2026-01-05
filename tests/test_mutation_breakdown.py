import numpy as np

from intent_ir.ir import IntentFunction
from verify.gen_cases import TestCase
from verify.interpreter import execute_intent
from verify.mutation import run_mutation_kill


def test_mutation_breakdown_has_totals():
    intent = IntentFunction.from_json_dict(
        {
            "name": "add_const",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "c", "attrs": {"value": 1.0, "dtype": "f32"}},
                {"op": "add", "inputs": ["x", "c"], "output": "y"},
            ],
            "outputs": ["y"],
        }
    )

    def run_ref(case: TestCase):
        rng = np.random.default_rng(int(case.seed))
        x = rng.standard_normal((int(case.shapes["N"]),), dtype=np.float32)
        out = execute_intent(intent, {"x": x}, shape_bindings=dict(case.shapes))
        return {"x": x, **out}

    cases = [TestCase(shapes={"N": 8}, dtypes={}, seed=0)]
    rep = run_mutation_kill(
        "unknown_kernel",
        intent=intent,
        run_ref_fn=run_ref,
        diff_cases=cases,
        metamorphic_base_case=cases[0],
        static_validate_fn=None,
        n_mutants=6,
        seed=0,
        include_bounded=False,
    )
    assert rep.total == len(rep.outcomes)
    assert rep.mutation_breakdown
    assert sum(int(v.get("total", 0)) for v in rep.mutation_breakdown.values()) == rep.total

