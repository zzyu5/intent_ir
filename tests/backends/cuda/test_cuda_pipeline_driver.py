from __future__ import annotations

from backends.cuda.pipeline.driver import run_cuda_pipeline
from backends.cuda.pipeline.stages import CUDA_PIPELINE_STAGES
from intent_ir.ir import IntentFunction


def _add_intent(name: str = "cuda_pipeline_add") -> IntentFunction:
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


def test_run_cuda_pipeline_reports_stage_artifacts_for_valid_intent() -> None:
    result = run_cuda_pipeline(_add_intent())
    assert result.ok is True
    assert result.reason_code == "ok"
    assert [s.name for s in result.stages] == list(CUDA_PIPELINE_STAGES)
    legalize = result.stages[0]
    assert legalize.ok is True
    assert legalize.artifacts["op_count"] == 1
    assert legalize.artifacts["tensor_count"] == 3
    emit = next(s for s in result.stages if s.name == "emit")
    assert emit.artifacts.get("codegen_mode") in {"cpp", "py"}


def test_run_cuda_pipeline_rejects_invalid_payload() -> None:
    result = run_cuda_pipeline({})
    assert result.ok is False
    assert result.reason_code == "invalid_intent"
    assert any((not s.ok) for s in result.stages)

