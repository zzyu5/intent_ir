from __future__ import annotations

from intent_ir.ir import IntentFunction
from intent_ir.parser import CandidateIntent
from pipeline.triton.providers import get_provider_plugin


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


def _maybe_normalize_provider_candidate(
    *,
    provider: str,
    spec_name: str,
    candidate: CandidateIntent,
    candidate_expanded: CandidateIntent | None,
) -> tuple[CandidateIntent, CandidateIntent | None, dict]:
    plugin = get_provider_plugin(provider)
    out, out_expanded, info = plugin.maybe_normalize_candidate(
        spec_name=str(spec_name),
        candidate=candidate,
        candidate_expanded=candidate_expanded,
    )
    return out, out_expanded, dict(info or {})


def _annotate_provider_intent_meta(
    intent,
    *,
    provider: str,
    source_op: str | None,
    capability_state: str | None,
    backend_target: str | None,
) -> None:
    plugin = get_provider_plugin(provider)
    plugin.annotate_intent_meta(
        intent,
        source_op=source_op,
        capability_state=capability_state,
        backend_target=backend_target,
    )


def _validate_provider_intent_meta(
    intent,
    *,
    provider: str,
) -> dict:
    plugin = get_provider_plugin(provider)
    return dict(plugin.validate_intent_meta(intent))


def test_flaggems_provider_normalization_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("keep")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="sigmoid2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info == {}
    assert out.intent.name == "keep"
    assert out_expanded is None


def test_flaggems_provider_normalization_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", "1")
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
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
    out, out_expanded, info = _maybe_normalize_provider_candidate(
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
    out, out_expanded, info = _maybe_normalize_provider_candidate(
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
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="batch_norm2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "batch_norm2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_clamp(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="clamp2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "clamp2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_hstack(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="hstack2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "hstack2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_per_token(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="per_token_group_quant_fp8_2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "per_token_group_quant_fp8_2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_remainder(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="remainder2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "remainder2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_repeat_interleave(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="repeat_interleave_tensor1d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "repeat_interleave_tensor1d"
    assert out_expanded is not None


def test_annotate_and_validate_provider_meta_generic() -> None:
    cand = _dummy_candidate("plain")
    _annotate_provider_intent_meta(
        cand.intent,
        provider="triton_custom",
        source_op=None,
        capability_state=None,
        backend_target="rvv",
    )
    report = _validate_provider_intent_meta(cand.intent, provider="triton_custom")
    assert report["ok"] is True
    assert report["provider"] == "triton_custom"


def test_validate_provider_meta_requires_source_state_for_flaggems() -> None:
    cand = _dummy_candidate("plain")
    _annotate_provider_intent_meta(
        cand.intent,
        provider="flaggems",
        source_op="add",
        capability_state="dual_pass",
        backend_target=None,
    )
    ok = _validate_provider_intent_meta(cand.intent, provider="flaggems")
    assert ok["ok"] is True

    cand2 = _dummy_candidate("plain")
    _annotate_provider_intent_meta(
        cand2.intent,
        provider="flaggems",
        source_op=None,
        capability_state=None,
        backend_target=None,
    )
    bad = _validate_provider_intent_meta(cand2.intent, provider="flaggems")
    assert bad["ok"] is False
