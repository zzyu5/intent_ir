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


def test_flaggems_provider_forces_canonical_for_tanh(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="tanh2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "tanh2d"
    assert out_expanded is not None


def test_flaggems_provider_forces_canonical_for_silu(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="silu2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "silu2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["y"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["const", "const", "mul", "exp", "add", "div", "mul"]


def test_flaggems_provider_forces_canonical_for_softplus(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="softplus2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "softplus2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == [
        "const",
        "mul",
        "exp",
        "add",
        "log",
        "gt",
        "where",
        "div",
    ]


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


def test_flaggems_provider_forces_canonical_for_cast_with_correct_dtype_direction(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="cast2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "cast2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("tensors", {}).get("x", {}).get("dtype") == "f16"
    assert payload.get("tensors", {}).get("out", {}).get("dtype") == "f32"
    assert payload.get("ops", [{}])[0].get("attrs", {}).get("to") == "f32"


def test_flaggems_provider_forces_canonical_for_lerp(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="lerp2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "lerp2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["C"]
    assert payload.get("tensors", {}).get("W", {}).get("shape") == []
    ops = payload.get("ops", [])
    assert [str(op.get("op")) for op in ops] == ["sub", "mul", "add"]


def test_flaggems_provider_forces_canonical_for_logical_or(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="logical_or2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "logical_or2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["Out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "cast", "max"]


def test_flaggems_provider_forces_canonical_for_logical_xor(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="logical_xor2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "logical_xor2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["Out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "cast", "ne"]


def test_flaggems_provider_forces_canonical_for_lt(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="lt2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "lt2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "lt"]


def test_flaggems_provider_forces_canonical_for_minimum(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="minimum2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "minimum2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["Out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "cast", "min", "cast"]


def test_flaggems_provider_forces_canonical_for_ne(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="ne2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "ne2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["output"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "cast", "ne"]


def test_flaggems_provider_forces_canonical_for_cumsum(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="cumsum2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "cumsum2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["iota", "identity", "cumsum", "identity"]


def test_flaggems_provider_forces_canonical_for_normed_cumsum_uses_eps_symbol(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="normed_cumsum2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "normed_cumsum2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    tensors = dict(payload.get("tensors", {}))
    assert "EPS" in tensors
    assert "eps" not in tensors
    assert any(str(op.get("op")) == "reduce_sum" for op in payload.get("ops", []))
    assert any(str(op.get("op")) == "broadcast_in_dim" for op in payload.get("ops", []))
    add_ops = [op for op in payload.get("ops", []) if str(op.get("op")) == "add"]
    assert add_ops
    assert add_ops[0].get("inputs") == ["y_denom", "EPS"]


def test_flaggems_provider_forces_canonical_for_pow_tensor_scalar(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="pow_tensor_scalar2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "pow_tensor_scalar2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["Out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "cast", "broadcast_in_dim", "pow"]


def test_flaggems_provider_forces_canonical_for_pow_tensor_tensor(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="pow_tensor_tensor2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "pow_tensor_tensor2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["output"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "cast", "pow", "identity"]


def test_flaggems_provider_forces_canonical_for_reciprocal(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="reciprocal2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "reciprocal2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["const", "cast", "div"]


def test_flaggems_provider_forces_canonical_for_rsqrt(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="rsqrt2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "rsqrt2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["cast", "rsqrt"]


def test_flaggems_provider_forces_canonical_for_sin(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="sin2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "sin2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["sin"]


def test_flaggems_provider_forces_canonical_for_tan(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="tan2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "tan2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["tan"]


def test_flaggems_provider_forces_canonical_for_neg(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", raising=False)
    cand = _dummy_candidate("old")
    out, out_expanded, info = _maybe_normalize_provider_candidate(
        provider="flaggems",
        spec_name="neg2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None
    assert info.get("enabled_by") == "provider_required_deterministic_override"
    assert out.intent.name == "neg2d"
    assert out_expanded is not None
    payload = out.intent.to_json_dict()
    assert payload.get("outputs") == ["out"]
    assert [str(op.get("op")) for op in payload.get("ops", [])] == ["const", "mul"]


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
