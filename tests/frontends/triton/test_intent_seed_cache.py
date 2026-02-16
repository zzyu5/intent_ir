from __future__ import annotations

from pathlib import Path

from intent_ir.ir import IntentFunction
from intent_ir.parser import CandidateIntent
from pipeline.triton.core import (
    _intent_seed_path,
    _load_intent_seed,
    _provider_deterministic_intent_for,
    _save_intent_seed,
)


def _simple_intent() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["A", "B"], "output": "C"}],
            "outputs": ["C"],
        }
    )


def test_intent_seed_path_suffix(tmp_path: Path) -> None:
    p = _intent_seed_path(tmp_path, "add2d")
    assert str(p).endswith("add2d.intent_seed.json")


def test_save_and_load_intent_seed_roundtrip(tmp_path: Path) -> None:
    intent = _simple_intent()
    cand = CandidateIntent(
        intent=intent,
        problem_params={"M": 4, "N": 8},
        schedule_params={"tile_m": 16},
        raw_json={"ops": [{"op": "add"}]},
        llm_trace={"provider": "mock"},
    )
    seed_path = tmp_path / "add2d.intent_seed.json"

    _save_intent_seed(
        seed_path=seed_path,
        kernel_name="add2d",
        triton_provider="flaggems",
        backend_target="rvv",
        candidate=cand,
        candidate_expanded=None,
    )

    loaded, loaded_expanded = _load_intent_seed(seed_path)
    assert loaded.intent.to_json_dict() == intent.to_json_dict()
    assert loaded.problem_params == {"M": 4, "N": 8}
    assert loaded.schedule_params == {"tile_m": 16}
    assert loaded.raw_json == {"ops": [{"op": "add"}]}
    assert loaded.llm_trace == {"provider": "mock"}
    assert loaded_expanded is not None
    assert loaded_expanded.intent.outputs == ["C"]


def test_provider_deterministic_intent_for_flaggems_known_kernel() -> None:
    intent = _provider_deterministic_intent_for(
        kernel_name="bitwise_right_shift2d",
        triton_provider="flaggems",
    )
    assert intent is not None
    assert intent.name == "bitwise_right_shift2d"


def test_provider_deterministic_intent_for_non_flaggems_provider_returns_none() -> None:
    intent = _provider_deterministic_intent_for(
        kernel_name="bitwise_right_shift2d",
        triton_provider="native",
    )
    assert intent is None
