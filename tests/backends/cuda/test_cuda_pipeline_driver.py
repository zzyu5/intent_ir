from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from backends.cuda.pipeline.driver import lower_cuda_contract_to_kernel, run_cuda_pipeline
from backends.cuda.pipeline.stages import CUDA_PIPELINE_STAGES
from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir
from intent_ir.mlir.passes.emit_cuda_contract import build_cuda_contract


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


def _sdpa_intent(name: str = "cuda_pipeline_sdpa") -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "query": {"dtype": "f32", "shape": ["B", "H", "Q", "D"], "layout": "row_major"},
                "key": {"dtype": "f32", "shape": ["B", "H", "K", "D"], "layout": "row_major"},
                "value": {"dtype": "f32", "shape": ["B", "H", "K", "D"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["B", "H", "Q", "D"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "scaled_dot_product_attention",
                    "inputs": ["query", "key", "value"],
                    "output": "out",
                    "attrs": {"is_causal": False},
                }
            ],
            "outputs": ["out"],
            "parallel_axes": ["B", "H", "Q", "D"],
            "schedule": {"tile_n": 128, "parallel_axes": ["B", "H", "Q", "D"]},
        }
    )


def _addmv_reduce_intent(name: str = "cuda_pipeline_addmv_reduce") -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "A": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "Inp": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "mul_AB": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
                "reduce_M": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "mul_alpha": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "mul_beta": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "beta": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "mul", "inputs": ["A", "B"], "output": "mul_AB", "attrs": {}},
                {"op": "reduce_sum", "inputs": ["mul_AB"], "output": "reduce_M", "attrs": {"dims": [1]}},
                {"op": "mul", "inputs": ["reduce_M", "alpha"], "output": "mul_alpha", "attrs": {}},
                {"op": "mul", "inputs": ["Inp", "beta"], "output": "mul_beta", "attrs": {}},
                {"op": "add", "inputs": ["mul_alpha", "mul_beta"], "output": "Out", "attrs": {}},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["N", "M"],
            "schedule": {"parallel_axes": ["N", "M"]},
        }
    )


def test_run_cuda_pipeline_reports_stage_artifacts_for_valid_intent() -> None:
    result = run_cuda_pipeline(to_mlir(_add_intent()))
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
    assert emit.artifacts.get("emit_backend") == "mlir_contract"
    assert emit.artifacts.get("emit_mode") in {"executed", "skipped_missing_bindings"}
    if emit.artifacts.get("emit_mode") == "executed":
        assert emit.artifacts.get("execution_engine") == "mlir_native"
        assert emit.artifacts.get("contract_schema_version") == "intent_mlir_backend_contract_v2"
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
    result = run_cuda_pipeline(to_mlir(intent))
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
    result = run_cuda_pipeline(to_mlir(intent))
    assert result.ok is True
    schedule = next(s for s in result.stages if s.name == "schedule")
    assert schedule.artifacts.get("op_family") == "matmul_conv"
    assert schedule.artifacts.get("schedule_profile") == "cuda_matmul_conv_v1"


def test_run_cuda_pipeline_rejects_invalid_payload() -> None:
    result = run_cuda_pipeline({})
    assert result.ok is False
    assert result.reason_code == "invalid_intent"
    assert any((not s.ok) for s in result.stages)


def test_run_cuda_pipeline_rejects_legacy_intent_payload() -> None:
    result = run_cuda_pipeline(_add_intent())
    assert result.ok is False
    assert result.reason_code == "invalid_intent"
    assert any((not s.ok) for s in result.stages)


def test_run_cuda_pipeline_schedule_only_mode_marks_compile_launch_as_schedule_only() -> None:
    result = run_cuda_pipeline(to_mlir(_add_intent()), pipeline_mode="schedule_only")
    assert result.ok is True
    compile_stage = next(s for s in result.stages if s.name == "compile")
    launch_stage = next(s for s in result.stages if s.name == "launch")
    assert compile_stage.artifacts.get("compile_mode") in {"skipped_schedule_only", "skipped_missing_bindings"}
    assert launch_stage.artifacts.get("launch_mode") in {"skipped_schedule_only", "skipped_missing_bindings"}


def test_run_cuda_pipeline_accepts_mlir_module_payload() -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_add_mlir"))
    result = run_cuda_pipeline(mod, pipeline_mode="schedule_only")
    assert result.ok is True
    assert result.input_ir_kind == "mlir_module"
    assert float(result.mlir_parse_ms) >= 0.0
    assert result.mlir_backend_contract_used is True
    legalize = next(s for s in result.stages if s.name == "legalize")
    assert legalize.artifacts.get("input_ir_kind") == "mlir_module"
    assert float(legalize.artifacts.get("mlir_parse_ms") or 0.0) >= 0.0


def test_run_cuda_pipeline_accepts_mlir_backend_contract_payload() -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_add_contract"))
    contract = build_cuda_contract(mod)
    result = run_cuda_pipeline(contract, pipeline_mode="schedule_only")
    assert result.ok is True
    assert result.input_ir_kind == "mlir_contract"
    assert result.mlir_backend_contract_used is True
    legalize = next(s for s in result.stages if s.name == "legalize")
    assert legalize.artifacts.get("mlir_backend_contract_used") is True
    assert legalize.artifacts.get("contract_backend") == "cuda"


def test_run_cuda_pipeline_accepts_contract_json_mapping() -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_add_contract_json"))
    contract = build_cuda_contract(mod)
    payload: dict[str, object] = dict(contract.to_json_dict())
    result = run_cuda_pipeline(payload, pipeline_mode="schedule_only")
    assert result.ok is True
    assert result.input_ir_kind == "mlir_contract"


def test_cuda_cpp_codegen_entrypoints_are_removed_for_hard_cut() -> None:
    with pytest.raises(ModuleNotFoundError):
        _ = importlib.import_module("backends.cuda.codegen")
    with pytest.raises(ModuleNotFoundError):
        _ = importlib.import_module("backends.cuda.codegen.cpp_driver")


def test_lower_cuda_contract_to_kernel_rejects_cuda_kernel_json_executable(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_kernel_json"))
    contract = build_cuda_contract(mod)
    src_path = tmp_path / "k.cu"
    src_path.write_text('extern "C" __global__ void k() {}', encoding="utf-8")
    kernel_payload = {
        "schema_version": "intent_cuda_lowered_kernel_v1",
        "kernel_name": "k",
        "io_spec": dict(contract.io_spec or {}),
        "launch": {"grid": [1, 1, 1], "block": [1, 1, 1], "shared_mem": 0},
        "output_names": ["C"],
        "cuda_src_path": str(src_path),
    }
    kernel_json = tmp_path / "k.json"
    kernel_json.write_text(json.dumps(kernel_payload), encoding="utf-8")
    contract.executable.format = "cuda_kernel_json"
    contract.executable.path = str(kernel_json)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    try:
        _ = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 4, "N": 4})
    except ValueError as e:
        msg = str(e)
        assert "unsupported or missing" in msg
        assert "cuda_ptx" in msg
    else:
        raise AssertionError("expected ValueError for cuda_kernel_json executable")


def test_lower_cuda_contract_to_kernel_supports_cuda_ptx_executable(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_ptx_exec"))
    contract = build_cuda_contract(mod)
    ptx_path = tmp_path / "k.ptx"
    ptx_path.write_text("// fake ptx\n.visible .entry k() { ret; }\n", encoding="utf-8")
    contract.executable.format = "cuda_ptx"
    contract.executable.path = str(ptx_path)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    contract.launch = {"grid": [1, 1, 1], "block": [1, 1, 1], "shared_mem": 0}
    lowered = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 4, "N": 4})
    assert str(lowered.get("kernel_name") or "") == "k"
    assert str(lowered.get("execution_engine") or "") == "mlir_native"
    assert isinstance(lowered.get("cuda_ptx"), (bytes, bytearray))


def test_lower_cuda_contract_to_kernel_uses_ptx_invocation_launch_when_contract_launch_empty(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_ptx_invocation_launch"))
    contract = build_cuda_contract(mod)
    ptx_path = tmp_path / "k.ptx"
    ptx_path.write_text("// fake ptx\n.visible .entry k() { ret; }\n", encoding="utf-8")
    contract.executable.format = "cuda_ptx"
    contract.executable.path = str(ptx_path)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    contract.executable.invocation = {
        "launch": {"grid": [2, 1, 1], "block": [64, 1, 1], "shared_mem": 0},
        "output_names": ["C"],
    }
    contract.launch = {}
    lowered = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 4, "N": 4})
    launch = dict(lowered.get("launch") or {})
    assert launch.get("grid") == [2, 1, 1]
    assert launch.get("block") == [64, 1, 1]


def test_lower_cuda_contract_to_kernel_uses_ptx_invocation_io_spec_when_contract_io_semantic_only(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_ptx_invocation_io"))
    contract = build_cuda_contract(mod)
    ptx_path = tmp_path / "k.ptx"
    ptx_path.write_text("// fake ptx\n.visible .entry k() { ret; }\n", encoding="utf-8")
    contract.executable.format = "cuda_ptx"
    contract.executable.path = str(ptx_path)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    contract.executable.invocation = {
        "io_spec": {
            "arg_names": ["x", "y", "out"],
            "tensors": {
                "x": {"dtype": "f32", "shape": [4, 4]},
                "y": {"dtype": "f32", "shape": [4, 4]},
                "out": {"dtype": "f32", "shape": [4, 4]},
            },
            "outputs": ["out"],
        },
        "launch": {"grid": [1, 1, 1], "block": [64, 1, 1], "shared_mem": 0},
        "output_names": ["out"],
    }
    contract.launch = {}
    lowered = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 4, "N": 4})
    io_spec = dict(lowered.get("io_spec") or {})
    assert io_spec.get("arg_names") == ["x", "y", "out"]


def test_lower_cuda_contract_to_kernel_uses_ptx_invocation_shape_bindings_for_missing_scalars(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_ptx_invocation_bindings"))
    contract = build_cuda_contract(mod)
    ptx_path = tmp_path / "k.ptx"
    ptx_path.write_text("// fake ptx\n.visible .entry k() { ret; }\n", encoding="utf-8")
    contract.executable.format = "cuda_ptx"
    contract.executable.path = str(ptx_path)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    contract.executable.invocation = {
        "shape_bindings": {"M": 4, "N": 4, "T": 16},
        "io_spec": {
            "arg_names": ["x", "out", "M", "N", "T"],
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"]},
                "out": {"dtype": "f32", "shape": ["T"]},
            },
            "scalars": {"M": "i32", "N": "i32", "T": "i32"},
            "outputs": ["out"],
        },
        "launch": {"grid": [1, 1, 1], "block": [64, 1, 1], "shared_mem": 0},
        "output_names": ["out"],
    }
    contract.launch = {}
    lowered = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 4, "N": 4})
    bindings = dict(lowered.get("bindings") or {})
    assert bindings.get("T") == 16


def test_lower_cuda_contract_to_kernel_augments_ptx_invocation_arg_names_from_signature(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_ptx_invocation_signature_augment"))
    contract = build_cuda_contract(mod)
    ptx_path = tmp_path / "k.ptx"
    ptx_path.write_text(
        (
            "// fake ptx\n"
            ".visible .entry k(\n"
            "    .param .u64 k_param_0,\n"
            "    .param .u64 k_param_1,\n"
            "    .param .u32 k_param_2,\n"
            "    .param .u32 k_param_3\n"
            ") { ret; }\n"
        ),
        encoding="utf-8",
    )
    contract.executable.format = "cuda_ptx"
    contract.executable.path = str(ptx_path)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    contract.executable.invocation = {
        "shape_bindings": {"M": 4, "N": 8},
        "io_spec": {
            "arg_names": ["A", "C"],
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"]},
                "C": {"dtype": "f32", "shape": ["M", "N"]},
            },
            "outputs": ["C"],
            "scalars": {},
        },
        "launch": {"grid": [1, 1, 1], "block": [64, 1, 1], "shared_mem": 0},
        "output_names": ["C"],
    }
    contract.launch = {}
    lowered = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 4, "N": 8})
    io_spec = dict(lowered.get("io_spec") or {})
    assert io_spec.get("arg_names") == ["A", "C", "M", "N"]
    scalars = dict(io_spec.get("scalars") or {})
    assert scalars.get("M") == "i32"
    assert scalars.get("N") == "i32"


def test_lower_cuda_contract_to_kernel_ptx_augments_scalar_aliases_from_shape_bindings(tmp_path: Path) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_ptx_alias_augment"))
    contract = build_cuda_contract(mod)
    ptx_path = tmp_path / "k.ptx"
    ptx_path.write_text("// fake ptx\n.visible .entry k() { ret; }\n", encoding="utf-8")
    contract.executable.format = "cuda_ptx"
    contract.executable.path = str(ptx_path)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    contract.io_spec = {
        "arg_names": ["A", "B", "out", "M0", "N0", "M1", "N1", "M_OUT", "N_OUT"],
        "tensors": {
            "A": {"dtype": "f32", "shape": ["M", "N"]},
            "B": {"dtype": "f32", "shape": ["M", "N"]},
            "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"]},
        },
        "scalars": {
            "M0": "i32",
            "N0": "i32",
            "M1": "i32",
            "N1": "i32",
            "M_OUT": "i32",
            "N_OUT": "i32",
        },
        "outputs": ["out"],
    }
    lowered = lower_cuda_contract_to_kernel(
        contract.to_json_dict(),
        shape_bindings={"M": 4, "N": 8, "M_OUT": 4, "N_OUT": 16},
    )
    bindings = dict(lowered.get("bindings") or {})
    assert bindings.get("M0") == 4
    assert bindings.get("N0") == 8
    assert bindings.get("M1") == 4
    assert bindings.get("N1") == 8
    assert bindings.get("M_OUT") == 4
    assert bindings.get("N_OUT") == 16


def test_lower_cuda_contract_to_kernel_strict_llvm_ptx_rejects_nvrtc_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_ptx_strict"))
    contract = build_cuda_contract(mod)
    ptx_path = tmp_path / "k.ptx"
    ptx_path.write_text("// fake ptx\n.visible .entry k() { ret; }\n", encoding="utf-8")
    contract.executable.format = "cuda_ptx"
    contract.executable.path = str(ptx_path)
    contract.executable.entry = "k"
    contract.executable.target = "cuda"
    artifacts = dict(contract.artifacts or {})
    artifacts["cuda_ptx_origin"] = "nvrtc_fallback_from_llvm"
    contract.artifacts = artifacts
    monkeypatch.setenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", "1")
    with pytest.raises(ValueError, match="strict LLVM PTX mode"):
        _ = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 4, "N": 4})


def test_lower_cuda_contract_to_kernel_rejects_mlir_module_contract_even_without_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_no_exec"))
    contract = build_cuda_contract(mod)
    module_path = tmp_path / "mod.mlir"
    module_path.write_text(str(mod.module_text), encoding="utf-8")
    contract.executable.format = "cuda_mlir_module"
    contract.executable.path = str(module_path)
    contract.executable.target = "cuda"
    monkeypatch.delenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", raising=False)
    with pytest.raises(ValueError, match="mlir_module executable fallback is removed"):
        _ = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 2, "N": 2})


def test_lower_cuda_contract_to_kernel_rejects_mlir_module_contract_when_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = to_mlir(_add_intent("cuda_pipeline_no_exec_strict"))
    contract = build_cuda_contract(mod)
    module_path = tmp_path / "mod.mlir"
    module_path.write_text(str(mod.module_text), encoding="utf-8")
    contract.executable.format = "cuda_mlir_module"
    contract.executable.path = str(module_path)
    contract.executable.target = "cuda"
    monkeypatch.setenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", "1")
    with pytest.raises(ValueError, match="mlir_module executable fallback is removed"):
        _ = lower_cuda_contract_to_kernel(contract.to_json_dict(), shape_bindings={"M": 2, "N": 2})
