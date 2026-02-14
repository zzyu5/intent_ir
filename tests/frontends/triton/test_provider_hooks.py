from __future__ import annotations

from intent_ir.ir import IntentFunction
from intent_ir.parser import CandidateIntent
from pipeline.triton.provider_hooks import (
    annotate_provider_intent_meta,
    maybe_normalize_provider_candidate,
    validate_provider_intent_meta,
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


def test_flaggems_provider_normalization_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("keep")
    out, out_expanded, info = maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="sigmoid2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is None
    assert out.intent.name == "keep"
    assert out_expanded is None


def test_flaggems_provider_normalization_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", "1")
    cand = _dummy_candidate("old")
    out, out_expanded, info = maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="sigmoid2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("provider") == "flaggems"
    assert out.intent.name == "sigmoid2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_required_specs(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="row_all",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "row_all"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_bitwise_or(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="bitwise_or2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "bitwise_or2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_batch_norm(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="batch_norm2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "batch_norm2d"
    assert out_expanded is not None


def test_annotate_and_validate_provider_meta_generic() -> None:
    cand = _dummy_candidate("plain")
    annotate_provider_intent_meta(
        cand.intent,
        provider="triton_custom",
        source_op=None,
        capability_state=None,
        backend_target="rvv",
    )
    report = validate_provider_intent_meta(cand.intent, provider="triton_custom", require_source_and_state=False)
    assert report["ok"] is True
    assert report["provider"] == "triton_custom"


def test_validate_provider_meta_requires_source_state_for_flaggems() -> None:
    cand = _dummy_candidate("plain")
    annotate_provider_intent_meta(
        cand.intent,
        provider="flaggems",
        source_op="add",
        capability_state="dual_pass",
        backend_target=None,
    )
    ok = validate_provider_intent_meta(cand.intent, provider="flaggems", require_source_and_state=True)
    assert ok["ok"] is True

    cand2 = _dummy_candidate("plain")
    annotate_provider_intent_meta(
        cand2.intent,
        provider="flaggems",
        source_op=None,
        capability_state=None,
        backend_target=None,
    )
    bad = validate_provider_intent_meta(cand2.intent, provider="flaggems", require_source_and_state=True)
    assert bad["ok"] is False
