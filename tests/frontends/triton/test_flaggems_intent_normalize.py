from __future__ import annotations

import math

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

    isnan = canonical_flaggems_intent_for_spec("isnan2d")
    assert isnan is not None
    assert [op.op for op in isnan.ops] == ["ne"]

    isinf = canonical_flaggems_intent_for_spec("isinf2d")
    assert isinf is not None
    assert [op.op for op in isinf.ops] == ["abs", "const", "gt"]
    const_val = [op.attrs["value"] for op in isinf.ops if op.op == "const"][0]
    assert math.isfinite(float(const_val))

    isfinite = canonical_flaggems_intent_for_spec("isfinite2d")
    assert isfinite is not None
    assert [op.op for op in isfinite.ops] == ["abs", "const", "le"]
    const_val = [op.attrs["value"] for op in isfinite.ops if op.op == "const"][0]
    assert math.isfinite(float(const_val))

    masked_fill = canonical_flaggems_intent_for_spec("masked_fill2d")
    assert masked_fill is not None
    assert [op.op for op in masked_fill.ops] == ["where"]
    assert "eq" not in [op.op for op in masked_fill.ops]

    gather = canonical_flaggems_intent_for_spec("gather2d")
    assert gather is not None
    assert [op.op for op in gather.ops] == ["gather"]
    assert gather.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    index_select = canonical_flaggems_intent_for_spec("index_select2d")
    assert index_select is not None
    assert [op.op for op in index_select.ops] == ["gather"]
    assert index_select.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    flip = canonical_flaggems_intent_for_spec("flip2d")
    assert flip is not None
    assert [op.op for op in flip.ops] == ["gather"]
    assert flip.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    embedding = canonical_flaggems_intent_for_spec("embedding2d")
    assert embedding is not None
    assert [op.op for op in embedding.ops] == ["gather"]
    assert embedding.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    glu = canonical_flaggems_intent_for_spec("glu2d")
    assert glu is not None
    assert [op.op for op in glu.ops] == ["glu"]

    cummax = canonical_flaggems_intent_for_spec("cummax1d")
    assert cummax is not None
    assert [op.op for op in cummax.ops] == ["cummax"]

    cummin = canonical_flaggems_intent_for_spec("cummin1d")
    assert cummin is not None
    assert [op.op for op in cummin.ops] == ["cummin"]

    index_add = canonical_flaggems_intent_for_spec("index_add2d")
    assert index_add is not None
    assert [op.op for op in index_add.ops] == ["index_add"]

    index_put = canonical_flaggems_intent_for_spec("index_put2d")
    assert index_put is not None
    assert [op.op for op in index_put.ops] == ["index_put"]

    count_nonzero = canonical_flaggems_intent_for_spec("count_nonzero2d")
    assert count_nonzero is not None
    assert [op.op for op in count_nonzero.ops] == ["const", "ne", "cast", "reduce_sum"]

    diag = canonical_flaggems_intent_for_spec("diag2d")
    assert diag is not None
    assert [op.op for op in diag.ops] == ["iota", "gather"]

    diag_embed = canonical_flaggems_intent_for_spec("diag_embed2d")
    assert diag_embed is not None
    assert [op.op for op in diag_embed.ops] == ["const", "broadcast_in_dim", "iota", "iota", "iota", "ne", "not", "gather", "where"]

    elu = canonical_flaggems_intent_for_spec("elu2d")
    assert elu is not None
    assert [op.op for op in elu.ops] == ["const", "const", "gt", "exp", "sub", "where"]

    celu = canonical_flaggems_intent_for_spec("celu2d")
    assert celu is not None
    assert [op.op for op in celu.ops] == ["const", "const", "gt", "exp", "sub", "where"]

    eye = canonical_flaggems_intent_for_spec("eye2d")
    assert eye is not None
    assert [op.op for op in eye.ops] == ["iota", "iota", "ne", "not", "cast"]

    eye_m = canonical_flaggems_intent_for_spec("eye_m2d")
    assert eye_m is not None
    assert [op.op for op in eye_m.ops] == ["iota", "iota", "ne", "not", "cast"]


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

    out2, out2_expanded, info2 = maybe_normalize_flaggems_candidate(
        spec_name="masked_fill2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info2 is not None and info2.get("applied") is True
    assert out2.intent.name == "masked_fill2d"
    assert out2_expanded is not None
    assert [op.op for op in out2.intent.ops] == ["where"]


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
