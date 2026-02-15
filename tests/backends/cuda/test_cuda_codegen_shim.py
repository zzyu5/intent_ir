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
    lowered_calls: list[tuple[str, dict[str, int]]] = []

    def _fake_run_pipeline(intent_payload: Any, *, shape_bindings: dict[str, int] | None = None) -> CudaPipelineResult:
        name = str(getattr(intent_payload, "name", "unknown"))
        calls.append(f"{name}:{sorted((shape_bindings or {}).items())}")
        return CudaPipelineResult(
            ok=True,
            stages=[CudaPipelineStage(name="legalize", ok=True, ms=0.1, detail="ok")],
            reason_code="ok",
        )

    monkeypatch.setattr("backends.cuda.pipeline.driver.run_cuda_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(
        "backends.cuda.codegen.cpp_driver.lower_intent_to_cuda_kernel_cpp",
        lambda intent_payload, *, bindings: (
            lowered_calls.append((str(intent_payload.name), dict(bindings)))
            or {
                "kernel_name": str(intent_payload.name),
                "cuda_src": "__global__ void k() {}",
                "io_spec": {},
                "launch": {"grid": [1, 1, 1], "block": [1, 1, 1], "shared_mem": 0},
                "output_names": ["C"],
                "bindings": dict(bindings),
            }
        ),
    )

    lowered = intentir_to_cuda.lower_intent_to_cuda_kernel(_add_intent(), shape_bindings={"M": 2, "N": 2})
    assert lowered.kernel_name == "cuda_add_shim"
    assert calls == ["cuda_add_shim:[('M', 2), ('N', 2)]"]
    assert lowered_calls == [("cuda_add_shim", {"M": 2, "N": 2, "CUDA_RESPECT_SCHEDULE": 0})]


def test_cuda_legacy_codegen_entry_bubbles_pipeline_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_pipeline(_intent_payload: Any, *, shape_bindings: dict[str, int] | None = None) -> CudaPipelineResult:
        _ = shape_bindings
        return CudaPipelineResult(
            ok=False,
            stages=[CudaPipelineStage(name="compile", ok=False, ms=1.0, detail="timeout")],
            reason_code="compile_timeout",
            reason_detail="compile stage exceeded budget",
        )

    monkeypatch.setattr("backends.cuda.pipeline.driver.run_cuda_pipeline", _fake_run_pipeline)

    with pytest.raises(intentir_to_cuda.CudaLoweringError, match="compile_timeout"):
        intentir_to_cuda.lower_intent_to_cuda_kernel(_add_intent("cuda_add_fail"), shape_bindings={"M": 2, "N": 2})
