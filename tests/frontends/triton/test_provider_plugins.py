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

    out2, out2_exp, info2 = plugin.maybe_normalize_candidate(
        spec_name="elu2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info2 is not None
    assert info2.get("enabled_by") == "provider_required_deterministic_override"
    assert out2.intent.name == "elu2d"
    assert out2_exp is not None

    out3, out3_exp, info3 = plugin.maybe_normalize_candidate(
        spec_name="eye2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info3 is not None
    assert info3.get("enabled_by") == "provider_required_deterministic_override"
    assert out3.intent.name == "eye2d"
    assert out3_exp is not None

    out4, out4_exp, info4 = plugin.maybe_normalize_candidate(
        spec_name="flip2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info4 is not None
    assert info4.get("enabled_by") == "provider_required_deterministic_override"
    assert out4.intent.name == "flip2d"
    assert out4_exp is not None

    out5, out5_exp, info5 = plugin.maybe_normalize_candidate(
        spec_name="glu2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info5 is not None
    assert info5.get("enabled_by") == "provider_required_deterministic_override"
    assert out5.intent.name == "glu2d"
    assert out5_exp is not None

    out6, out6_exp, info6 = plugin.maybe_normalize_candidate(
        spec_name="linspace1d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info6 is not None
    assert info6.get("enabled_by") == "provider_required_deterministic_override"
    assert out6.intent.name == "linspace1d"
    assert out6_exp is not None

    out7, out7_exp, info7 = plugin.maybe_normalize_candidate(
        spec_name="nll_loss2d_forward",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info7 is not None
    assert info7.get("enabled_by") == "provider_required_deterministic_override"
    assert out7.intent.name == "nll_loss2d_forward"
    assert out7_exp is not None

    out8, out8_exp, info8 = plugin.maybe_normalize_candidate(
        spec_name="one_hot2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info8 is not None
    assert info8.get("enabled_by") == "provider_required_deterministic_override"
    assert out8.intent.name == "one_hot2d"
    assert out8_exp is not None

    out9, out9_exp, info9 = plugin.maybe_normalize_candidate(
        spec_name="max_pool2d_with_indices_nchw",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info9 is not None
    assert info9.get("enabled_by") == "provider_required_deterministic_override"
    assert out9.intent.name == "max_pool2d_with_indices_nchw"
    assert out9_exp is not None

    out10, out10_exp, info10 = plugin.maybe_normalize_candidate(
        spec_name="conv1d_ncl",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info10 is not None
    assert info10.get("enabled_by") == "provider_required_deterministic_override"
    assert out10.intent.name == "conv1d_ncl"
    assert out10_exp is not None

    out11, out11_exp, info11 = plugin.maybe_normalize_candidate(
        spec_name="conv3d_ncdhw",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info11 is not None
    assert info11.get("enabled_by") == "provider_required_deterministic_override"
    assert out11.intent.name == "conv3d_ncdhw"
    assert out11_exp is not None

    out12, out12_exp, info12 = plugin.maybe_normalize_candidate(
        spec_name="conv_depthwise2d_nchw",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info12 is not None
    assert info12.get("enabled_by") == "provider_required_deterministic_override"
    assert out12.intent.name == "conv_depthwise2d_nchw"
    assert out12_exp is not None

    out13, out13_exp, info13 = plugin.maybe_normalize_candidate(
        spec_name="trace2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info13 is not None
    assert info13.get("enabled_by") == "provider_required_deterministic_override"
    assert out13.intent.name == "trace2d"
    assert out13_exp is not None

    out14, out14_exp, info14 = plugin.maybe_normalize_candidate(
        spec_name="triu2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info14 is not None
    assert info14.get("enabled_by") == "provider_required_deterministic_override"
    assert out14.intent.name == "triu2d"
    assert out14_exp is not None

    out15, out15_exp, info15 = plugin.maybe_normalize_candidate(
        spec_name="upsample_nearest1d_ncl",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info15 is not None
    assert info15.get("enabled_by") == "provider_required_deterministic_override"
    assert out15.intent.name == "upsample_nearest1d_ncl"
    assert out15_exp is not None

    out16, out16_exp, info16 = plugin.maybe_normalize_candidate(
        spec_name="upsample_nearest2d_nchw",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info16 is not None
    assert info16.get("enabled_by") == "provider_required_deterministic_override"
    assert out16.intent.name == "upsample_nearest2d_nchw"
    assert out16_exp is not None

    out17, out17_exp, info17 = plugin.maybe_normalize_candidate(
        spec_name="scatter2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info17 is not None
    assert info17.get("enabled_by") == "provider_required_deterministic_override"
    assert out17.intent.name == "scatter2d"
    assert out17_exp is not None

    out18, out18_exp, info18 = plugin.maybe_normalize_candidate(
        spec_name="select_scatter2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info18 is not None
    assert info18.get("enabled_by") == "provider_required_deterministic_override"
    assert out18.intent.name == "select_scatter2d"
    assert out18_exp is not None

    out19, out19_exp, info19 = plugin.maybe_normalize_candidate(
        spec_name="slice_scatter2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info19 is not None
    assert info19.get("enabled_by") == "provider_required_deterministic_override"
    assert out19.intent.name == "slice_scatter2d"
    assert out19_exp is not None

    out20, out20_exp, info20 = plugin.maybe_normalize_candidate(
        spec_name="quantile2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info20 is not None
    assert info20.get("enabled_by") == "provider_required_deterministic_override"
    assert out20.intent.name == "quantile2d"
    assert out20_exp is not None

    out21, out21_exp, info21 = plugin.maybe_normalize_candidate(
        spec_name="polar2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info21 is not None
    assert info21.get("enabled_by") == "provider_required_deterministic_override"
    assert out21.intent.name == "polar2d"
    assert out21_exp is not None

    out22, out22_exp, info22 = plugin.maybe_normalize_candidate(
        spec_name="unique2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info22 is not None
    assert info22.get("enabled_by") == "provider_required_deterministic_override"
    assert out22.intent.name == "unique2d"
    assert out22_exp is not None

    out23, out23_exp, info23 = plugin.maybe_normalize_candidate(
        spec_name="weight_norm2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info23 is not None
    assert info23.get("enabled_by") == "provider_required_deterministic_override"
    assert out23.intent.name == "weight_norm2d"
    assert out23_exp is not None

    out24, out24_exp, info24 = plugin.maybe_normalize_candidate(
        spec_name="scaled_dot_product_attention_bhsd",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info24 is not None
    assert info24.get("enabled_by") == "provider_required_deterministic_override"
    assert out24.intent.name == "scaled_dot_product_attention_bhsd"
    assert out24_exp is not None
