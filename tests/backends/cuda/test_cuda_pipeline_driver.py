from __future__ import annotations

from backends.cuda.codegen.cpp_driver import lower_intent_to_cuda_kernel_cpp
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
    assert "rewrite_counts" in legalize.artifacts
    assert "total_rewrite_candidates" in legalize.artifacts["rewrite_counts"]
    emit = next(s for s in result.stages if s.name == "emit")
    assert emit.artifacts.get("emit_backend") == "cpp_pybind"
    assert emit.artifacts.get("emit_mode") in {"kernel_generated", "skipped_missing_bindings"}
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("op_family") == "elementwise_reduction"
    assert str(schedule.artifacts.get("schedule_profile") or "").startswith("cuda_")


def test_run_cuda_pipeline_marks_schedule_rewrite_aware_for_transform_heavy_intent() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "cuda_pipeline_transform_heavy",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "identity", "inputs": ["A"], "output": "B", "attrs": {}},
            ],
            "outputs": ["B"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 256, "parallel_axes": ["M", "N"]},
        }
    )
    result = run_cuda_pipeline(intent)
    assert result.ok is True
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("rewrite_aware") is True


def test_run_cuda_pipeline_matmul_conv_profile_exported() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "cuda_pipeline_mm",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
            "outputs": ["C"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_m": 32, "tile_n": 64, "tile_k": 16, "parallel_axes": ["M", "N"]},
        }
    )
    result = run_cuda_pipeline(intent)
    assert result.ok is True
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("op_family") == "matmul_conv"
    assert schedule.artifacts.get("schedule_profile") == "cuda_matmul_conv_v1"


def test_run_cuda_pipeline_rejects_invalid_payload() -> None:
    result = run_cuda_pipeline({})
    assert result.ok is False
    assert result.reason_code == "invalid_intent"
    assert any((not s.ok) for s in result.stages)


def test_run_cuda_pipeline_schedule_only_mode_marks_compile_launch_as_schedule_only() -> None:
    result = run_cuda_pipeline(_add_intent(), pipeline_mode="schedule_only")
    assert result.ok is True
    compile_stage = next(s for s in result.stages if s.name == "compile")
    launch_stage = next(s for s in result.stages if s.name == "launch")
    assert compile_stage.artifacts.get("compile_mode") in {"skipped_schedule_only", "skipped_missing_bindings"}
    assert launch_stage.artifacts.get("launch_mode") in {"skipped_schedule_only", "skipped_missing_bindings"}


def test_var_mean_codegen_signature_has_no_trailing_comma() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "var_mean2d",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "std_out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "std", "inputs": ["inp"], "output": "std_out", "attrs": {"axis": 1, "dims": [1], "keepdims": False, "correction": 1}},
                {"op": "mul", "inputs": ["std_out", "std_out"], "output": "out", "attrs": {}},
            ],
            "outputs": ["out"],
        }
    )
    lowered = lower_intent_to_cuda_kernel_cpp(intent, bindings={"M": 4, "N": 64})
    src = str(lowered.get("cuda_src") or "")
    assert "float* __restrict__ out,\n)" not in src
    assert "float* __restrict__ out\n)" in src
