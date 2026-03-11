from __future__ import annotations

from pathlib import Path

from backends.common.mlir_contract import MlirBackendContract
from backends.spmd_rvv.pipeline.driver import run_rvv_pipeline
from backends.spmd_rvv.pipeline.stages import RVV_PIPELINE_STAGES
from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir
from intent_ir.mlir.passes.emit_rvv_contract import build_rvv_contract


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
    result = run_rvv_pipeline(to_mlir(_intent("add", name="rvv_pipeline_add")))
    assert result.ok is True
    assert result.reason_code == "ok"
    assert [s.name for s in result.stages] == list(RVV_PIPELINE_STAGES)
    legalize = result.stages[0]
    assert legalize.ok is True
    assert legalize.artifacts["op_count"] == 1
    assert legalize.artifacts["tensor_count"] == 3
    assert "rewrite_counts" in legalize.artifacts
    assert "total_rewrite_candidates" in legalize.artifacts["rewrite_counts"]
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("op_family") == "elementwise_reduction"
    assert str(schedule.artifacts.get("schedule_profile") or "").startswith("rvv_")


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
    result = run_rvv_pipeline(to_mlir(intent))
    assert result.ok is True
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("rewrite_aware") is True


def test_run_rvv_pipeline_matmul_conv_profile_exported() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "rvv_pipeline_mm",
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
    result = run_rvv_pipeline(to_mlir(intent))
    assert result.ok is True
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("op_family") == "matmul_conv"
    assert schedule.artifacts.get("schedule_profile") == "rvv_matmul_conv_v1"


def test_run_rvv_pipeline_flags_unsupported_ops() -> None:
    result = run_rvv_pipeline(to_mlir(_intent("weight_norm_interface", name="rvv_pipeline_unsupported")))
    assert result.ok is False
    assert result.reason_code == "lowering_missing_op"
    assert any((not s.ok) for s in result.stages)


def test_run_rvv_pipeline_rejects_legacy_intent_payload() -> None:
    result = run_rvv_pipeline(_intent("add", name="rvv_pipeline_legacy_intent"))
    assert result.ok is False
    assert result.reason_code == "invalid_intent"
    assert any((not s.ok) for s in result.stages)


def test_run_rvv_pipeline_schedule_only_mode_marks_compile_run_as_schedule_only() -> None:
    result = run_rvv_pipeline(
        to_mlir(_intent("add", name="rvv_pipeline_add_schedule_only")),
        pipeline_mode="schedule_only",
    )
    assert result.ok is True
    compile_stage = next(s for s in result.stages if s.name == "compile")
    run_stage = next(s for s in result.stages if s.name == "run")
    assert compile_stage.artifacts.get("compile_mode") in {"skipped_schedule_only", "skipped_missing_bindings"}
    assert run_stage.artifacts.get("run_mode") in {"skipped_schedule_only", "skipped_missing_bindings"}


def test_run_rvv_pipeline_accepts_mlir_module_payload() -> None:
    mod = to_mlir(_intent("add", name="rvv_pipeline_add_mlir"))
    result = run_rvv_pipeline(mod, pipeline_mode="schedule_only")
    assert result.ok is True
    assert result.input_ir_kind == "mlir_module"
    assert float(result.mlir_parse_ms) >= 0.0
    assert result.mlir_backend_contract_used is True
    legalize = next(s for s in result.stages if s.name == "legalize")
    assert legalize.artifacts.get("input_ir_kind") == "mlir_module"
    assert float(legalize.artifacts.get("mlir_parse_ms") or 0.0) >= 0.0
    assert legalize.artifacts.get("mlir_backend_contract_used") is True


def test_run_rvv_pipeline_accepts_mlir_backend_contract_payload() -> None:
    mod = to_mlir(_intent("add", name="rvv_pipeline_contract"))
    contract = build_rvv_contract(mod, source_kind="mlir_module")
    result = run_rvv_pipeline(contract, pipeline_mode="schedule_only")
    assert result.ok is True
    assert result.input_ir_kind == "mlir_contract"
    assert result.mlir_backend_contract_used is True
    legalize = next(s for s in result.stages if s.name == "legalize")
    assert legalize.artifacts.get("mlir_backend_contract_used") is True
    assert legalize.artifacts.get("contract_backend") == "rvv"


def test_run_rvv_pipeline_accepts_contract_json_mapping() -> None:
    mod = to_mlir(_intent("add", name="rvv_pipeline_contract_json"))
    contract = build_rvv_contract(mod, source_kind="mlir_module")
    payload = contract.to_json_dict()
    result = run_rvv_pipeline(payload, pipeline_mode="schedule_only")
    assert result.ok is True
    assert result.input_ir_kind == "mlir_contract"
    assert result.mlir_backend_contract_used is True
    assert isinstance(MlirBackendContract.from_json_dict(payload), MlirBackendContract)


def test_run_rvv_pipeline_supports_prebuilt_rvv_elf(monkeypatch, tmp_path: Path) -> None:
    mod = to_mlir(_intent("add", name="rvv_pipeline_prebuilt_elf"))
    contract = build_rvv_contract(mod, source_kind="mlir_module")
    elf_path = tmp_path / "run.elf"
    elf_path.write_bytes(b"\x7fELFfake")
    contract.executable.format = "rvv_elf"
    contract.executable.path = str(elf_path)
    contract.executable.target = "rvv"

    class _Proc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(*_args, **_kwargs):
        return _Proc()

    monkeypatch.setattr("backends.spmd_rvv.pipeline.driver.subprocess.run", _fake_run)
    result = run_rvv_pipeline(contract.to_json_dict(), shape_bindings={"M": 2, "N": 2})
    assert result.ok is True
    emit_stage = next(s for s in result.stages if s.name == "emit_cpp")
    compile_stage = next(s for s in result.stages if s.name == "compile")
    run_stage = next(s for s in result.stages if s.name == "run")
    assert emit_stage.artifacts.get("executable_format") == "rvv_elf"
    assert compile_stage.artifacts.get("compile_mode") == "prebuilt_elf_staged"
    assert run_stage.artifacts.get("run_mode") == "executed"
