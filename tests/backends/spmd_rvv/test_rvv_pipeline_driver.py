from __future__ import annotations

from backends.spmd_rvv.pipeline.driver import run_rvv_pipeline
from backends.spmd_rvv.pipeline.stages import RVV_PIPELINE_STAGES
from intent_ir.ir import IntentFunction


def _intent(op: str, *, name: str) -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": op, "inputs": ["A", "B"], "output": "C", "attrs": {}}],
            "outputs": ["C"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 64, "parallel_axes": ["M", "N"]},
        }
    )


def test_run_rvv_pipeline_reports_stage_artifacts_for_supported_intent() -> None:
    result = run_rvv_pipeline(_intent("add", name="rvv_pipeline_add"))
    assert result.ok is True
    assert result.reason_code == "ok"
    assert [s.name for s in result.stages] == list(RVV_PIPELINE_STAGES)
    legalize = result.stages[0]
    assert legalize.ok is True
    assert legalize.artifacts["op_count"] == 1
    assert legalize.artifacts["tensor_count"] == 3
    assert "rewrite_counts" in legalize.artifacts
    assert "total_rewrite_candidates" in legalize.artifacts["rewrite_counts"]


def test_run_rvv_pipeline_marks_schedule_rewrite_aware_for_transform_heavy_intent() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "rvv_pipeline_transform_heavy",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "identity", "inputs": ["A"], "output": "B", "attrs": {}},
            ],
            "outputs": ["B"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 128, "parallel_axes": ["M", "N"]},
        }
    )
    result = run_rvv_pipeline(intent)
    assert result.ok is True
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("rewrite_aware") is True


def test_run_rvv_pipeline_flags_unsupported_ops() -> None:
    result = run_rvv_pipeline(_intent("weight_norm_interface", name="rvv_pipeline_unsupported"))
    assert result.ok is False
    assert result.reason_code == "lowering_missing_op"
    assert any((not s.ok) for s in result.stages)
