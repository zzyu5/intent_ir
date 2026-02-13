from __future__ import annotations

from intent_ir.ir import IntentFunction
from intent_ir.parser import CandidateIntent
from pipeline.triton.flaggems_intent_normalize import (
    canonical_flaggems_intent_for_spec,
    maybe_normalize_flaggems_candidate,
)


def _dummy_candidate(name: str = "dummy") -> CandidateIntent:
    intent = IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "identity", "inputs": ["x"], "output": "y"}],
            "outputs": ["y"],
        }
    )
    return CandidateIntent(intent=intent, problem_params={}, schedule_params={}, raw_json={}, llm_trace={})


def test_canonical_intent_templates_exist_for_blocked_kernels() -> None:
    sigmoid = canonical_flaggems_intent_for_spec("sigmoid2d")
    assert sigmoid is not None
    assert sigmoid.name == "sigmoid2d"
    assert sigmoid.outputs == ["output"]
    assert any(op.op == "exp" for op in sigmoid.ops)
    assert not any(("base" in (op.attrs or {})) for op in sigmoid.ops if op.op == "exp")

    batch_norm = canonical_flaggems_intent_for_spec("batch_norm2d")
    assert batch_norm is not None
    assert batch_norm.name == "batch_norm2d"
    assert batch_norm.outputs == ["output_1", "mean", "inv_std", "running_mean_out", "running_var_out"]
    assert any(op.op == "reduce_sum" for op in batch_norm.ops)
    assert any(op.op == "rsqrt" for op in batch_norm.ops)


def test_maybe_normalize_flaggems_candidate_overrides_known_spec() -> None:
    cand = _dummy_candidate("old")
    out, out_expanded, info = maybe_normalize_flaggems_candidate(
        spec_name="sigmoid2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None and info.get("applied") is True
    assert out.intent.name == "sigmoid2d"
    assert out_expanded is not None
    assert out.raw_json.get("normalized_by") == "flaggems_canonical"


def test_maybe_normalize_flaggems_candidate_noop_for_other_specs() -> None:
    cand = _dummy_candidate("keep")
    out, out_expanded, info = maybe_normalize_flaggems_candidate(
        spec_name="relu2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is None
    assert out.intent.name == "keep"
    assert out_expanded is None
