from __future__ import annotations

from intent_ir.ir import IntentFunction
from intent_ir.parser import CandidateIntent
from pipeline.triton.providers import get_provider_plugin, registered_providers


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


def test_provider_registry_contains_native_and_flaggems() -> None:
    names = set(registered_providers())
    assert "native" in names
    assert "flaggems" in names


def test_unknown_provider_uses_generic_plugin_behavior() -> None:
    plugin = get_provider_plugin("custom_provider")
    cand = _dummy_candidate()
    plugin.annotate_intent_meta(cand.intent, source_op=None, capability_state=None, backend_target="rvv")
    out = plugin.validate_intent_meta(cand.intent)
    assert out["ok"] is True
    assert out["provider"] == "custom_provider"


def test_flaggems_plugin_opt_in_normalization(monkeypatch) -> None:
    plugin = get_provider_plugin("flaggems")
    cand = _dummy_candidate("old")

    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    out0, out0_exp, info0 = plugin.maybe_normalize_candidate(
        spec_name="sigmoid2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info0 is None
    assert out0.intent.name == "old"
    assert out0_exp is None

    monkeypatch.setenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", "1")
    out1, out1_exp, info1 = plugin.maybe_normalize_candidate(
        spec_name="sigmoid2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info1 is not None
    assert out1.intent.name == "sigmoid2d"
    assert out1_exp is not None


def test_flaggems_plugin_forces_deterministic_overrides_for_known_unstable_specs(monkeypatch) -> None:
    plugin = get_provider_plugin("flaggems")
    cand = _dummy_candidate("old")

    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    out, out_exp, info = plugin.maybe_normalize_candidate(
        spec_name="diag2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "diag2d"
    assert out_exp is not None
