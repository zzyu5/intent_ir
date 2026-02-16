from __future__ import annotations

from dataclasses import dataclass

from intent_ir.ir import IntentFunction

from backends.common.pipeline_utils import (
    collect_intent_info,
    has_symbolic_dims,
    legalize_rewrite_counts,
    normalize_bindings,
    op_family,
    resolve_dim_int,
    run_stage,
    schedule_overrides_from_env,
)


@dataclass
class _Stage:
    name: str
    ok: bool
    ms: float
    detail: str
    artifacts: dict


def test_run_stage_success_with_artifacts() -> None:
    stage = run_stage(
        "s",
        lambda: ("ok", {"x": 1}),
        stage_factory=_Stage,
    )
    assert stage.ok is True
    assert stage.detail == "ok"
    assert stage.artifacts == {"x": 1}
    assert stage.ms >= 0.0


def test_collect_intent_info_extracts_ops_tensors_and_schedule() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "u",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["A", "B"], "output": "B"}],
            "outputs": ["B"],
            "schedule": {"tile_m": 32, "tile_n": 64},
        }
    )
    name, ops, tensor_shapes, sched = collect_intent_info(intent)
    assert name == "u"
    assert ops == ["add"]
    assert tensor_shapes["A"] == ["M", "N"]
    assert sched["tile_m"] == 32
    assert sched["tile_n"] == 64


def test_common_mapping_helpers() -> None:
    assert legalize_rewrite_counts(["identity", "reshape"])["total_rewrite_candidates"] == 2
    assert op_family(["matmul"]) == "matmul_conv"
    assert op_family(["reduce_sum"]) == "elementwise_reduction"
    assert has_symbolic_dims({"A": ["M", 4]}) is True
    assert has_symbolic_dims({"A": [2, 4]}) is False
    assert normalize_bindings({"M": 2.0, "N": True, "X": "4"}) == {"M": 2, "N": 1, "X": 4}


class _Sym:
    def __init__(self, value: str):
        self.value = value


def test_resolve_dim_int_supports_value_attr_symbol() -> None:
    assert resolve_dim_int(_Sym("M"), {"M": 16}) == 16


def test_schedule_overrides_from_env_backend_precedence(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_TILE_M", "8")
    monkeypatch.setenv("INTENTIR_CUDA_TILE_M", "16")
    monkeypatch.setenv("INTENTIR_CUDA_TILE_N", "32")
    monkeypatch.setenv("INTENTIR_CUDA_SCHEDULE_PROFILE_TAG", "fast")
    overrides, tag = schedule_overrides_from_env(backend_prefix="cuda")
    assert overrides == {"tile_m": 16, "tile_n": 32}
    assert tag == "fast"
