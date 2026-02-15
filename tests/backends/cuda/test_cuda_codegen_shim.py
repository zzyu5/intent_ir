from __future__ import annotations

from typing import Any

import pytest

from backends.cuda.codegen import intentir_to_cuda
from backends.cuda.pipeline.stages import CudaPipelineResult, CudaPipelineStage
from intent_ir.ir import IntentFunction


def _add_intent(name: str = "cuda_add_shim") -> IntentFunction:
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


def test_cuda_legacy_codegen_entry_calls_pipeline_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_run_pipeline(intent_payload: Any) -> CudaPipelineResult:
        name = str(getattr(intent_payload, "name", "unknown"))
        calls.append(name)
        return CudaPipelineResult(
            ok=True,
            stages=[CudaPipelineStage(name="legalize", ok=True, ms=0.1, detail="ok")],
            reason_code="ok",
        )

    monkeypatch.setattr("backends.cuda.pipeline.driver.run_cuda_pipeline", _fake_run_pipeline)
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")

    lowered = intentir_to_cuda.lower_intent_to_cuda_kernel(_add_intent(), shape_bindings={"M": 2, "N": 2})
    assert lowered.kernel_name == "cuda_add_shim"
    assert calls == ["cuda_add_shim"]


def test_cuda_legacy_codegen_entry_bubbles_pipeline_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_pipeline(_intent_payload: Any) -> CudaPipelineResult:
        return CudaPipelineResult(
            ok=False,
            stages=[CudaPipelineStage(name="compile", ok=False, ms=1.0, detail="timeout")],
            reason_code="compile_timeout",
            reason_detail="compile stage exceeded budget",
        )

    monkeypatch.setattr("backends.cuda.pipeline.driver.run_cuda_pipeline", _fake_run_pipeline)
    monkeypatch.setenv("INTENTIR_CUDA_CODEGEN", "py")

    with pytest.raises(intentir_to_cuda.CudaLoweringError, match="compile_timeout"):
        intentir_to_cuda.lower_intent_to_cuda_kernel(_add_intent("cuda_add_fail"), shape_bindings={"M": 2, "N": 2})

