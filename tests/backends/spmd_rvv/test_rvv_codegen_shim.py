from __future__ import annotations

from typing import Any

import pytest

from backends.spmd_rvv.codegen import intentir_to_c
from backends.spmd_rvv.pipeline.stages import RvvPipelineResult, RvvPipelineStage
from intent_ir.ir import IntentFunction


def _add_intent(name: str = "rvv_add_shim") -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "add", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
            "outputs": ["C"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )


def test_rvv_legacy_codegen_entry_calls_pipeline_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline_calls: list[str] = []
    cpp_calls: list[str] = []

    def _fake_run_pipeline(intent_payload: Any) -> RvvPipelineResult:
        pipeline_calls.append(str(getattr(intent_payload, "name", "unknown")))
        return RvvPipelineResult(
            ok=True,
            stages=[RvvPipelineStage(name="legalize", ok=True, ms=0.1, detail="ok")],
            reason_code="ok",
        )

    def _fake_cpp_lower(
        intent: IntentFunction,
        *,
        shape_bindings,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        mode: str = "verify",
    ) -> str:
        del shape_bindings, atol, rtol, mode
        cpp_calls.append(str(intent.name))
        return "ok"

    monkeypatch.setattr("backends.spmd_rvv.pipeline.driver.run_rvv_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(intentir_to_c, "lower_intent_to_c_with_files_cpp", _fake_cpp_lower)

    got = intentir_to_c.lower_intent_to_c_with_files(_add_intent(), shape_bindings={"M": 2, "N": 2})
    assert got == "ok"
    assert pipeline_calls == ["rvv_add_shim"]
    assert cpp_calls == ["rvv_add_shim"]


def test_rvv_legacy_codegen_entry_bubbles_pipeline_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_pipeline(_intent_payload: Any) -> RvvPipelineResult:
        return RvvPipelineResult(
            ok=False,
            stages=[RvvPipelineStage(name="compile", ok=False, ms=1.0, detail="timeout")],
            reason_code="compile_timeout",
            reason_detail="rvv compile stage exceeded budget",
        )

    monkeypatch.setattr("backends.spmd_rvv.pipeline.driver.run_rvv_pipeline", _fake_run_pipeline)

    with pytest.raises(ValueError, match="compile_timeout"):
        intentir_to_c.lower_intent_to_c_with_files(_add_intent("rvv_add_fail"), shape_bindings={"M": 2, "N": 2})

