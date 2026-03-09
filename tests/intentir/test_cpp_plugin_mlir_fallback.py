from __future__ import annotations

from intent_ir.mlir.module import IntentMLIRModule
from intent_ir.mlir.pass_manager import _run_one_pass


def test_cpp_plugin_mlir_fallback_runs_python_cuda_passes(monkeypatch) -> None:
    module = IntentMLIRModule(module_text="module {}", dialect_version="std_mlir_v1", meta={})

    def _apply_tuning_db(mod: IntentMLIRModule, **_: object) -> IntentMLIRModule:
        mod = IntentMLIRModule(
            module_text=mod.module_text,
            dialect_version=mod.dialect_version,
            provenance=dict(mod.provenance or {}),
            symbols=list(mod.symbols or []),
            meta=dict(mod.meta or {}),
            intent_json=(dict(mod.intent_json) if isinstance(mod.intent_json, dict) else None),
        )
        mod.meta["apply_tuning_db_seen"] = True
        return mod

    def _lower_cuda(mod: IntentMLIRModule, **_: object) -> IntentMLIRModule:
        mod = IntentMLIRModule(
            module_text=mod.module_text,
            dialect_version=mod.dialect_version,
            provenance=dict(mod.provenance or {}),
            symbols=list(mod.symbols or []),
            meta=dict(mod.meta or {}),
            intent_json=(dict(mod.intent_json) if isinstance(mod.intent_json, dict) else None),
        )
        mod.meta["lower_cuda_seen"] = True
        return mod

    monkeypatch.delenv("INTENTIR_MLIR_PASS_PLUGIN", raising=False)
    monkeypatch.setenv("INTENTIR_AUTO_MLIR_PASS_PLUGIN", "0")
    monkeypatch.setitem(__import__("intent_ir.mlir.pass_manager", fromlist=["PASS_REGISTRY"]).PASS_REGISTRY, "apply_tuning_db", _apply_tuning_db)
    monkeypatch.setitem(__import__("intent_ir.mlir.pass_manager", fromlist=["PASS_REGISTRY"]).PASS_REGISTRY, "lower_intent_to_cuda_gpu_kernel", _lower_cuda)

    result = _run_one_pass(
        module,
        "mlir-opt:pass-pipeline=builtin.module(intentir-apply-tuning-db-cuda-v1,intentir-lower-cuda-focus-v1)",
        backend="cuda",
        toolchain={"tools": {"mlir-opt": {"path": ""}}},
    )
    assert result.kind == "python"
    assert "python_fallback:intentir_mlir_plugin_unavailable" in result.detail
    assert result.module.meta["apply_tuning_db_seen"] is True
    assert result.module.meta["lower_cuda_seen"] is True
    assert result.module.meta["intentir_mlir_opt_fallback_passes"] == [
        "apply_tuning_db",
        "lower_intent_to_cuda_gpu_kernel",
    ]


def test_cpp_plugin_mlir_fallback_runs_extract_gpu_module(monkeypatch) -> None:
    module = IntentMLIRModule(module_text="module { gpu.module @kernels {} }", dialect_version="std_mlir_v1", meta={})

    def _extract_gpu(mod: IntentMLIRModule, **_: object) -> IntentMLIRModule:
        mod = IntentMLIRModule(
            module_text="module {}",
            dialect_version=mod.dialect_version,
            provenance=dict(mod.provenance or {}),
            symbols=list(mod.symbols or []),
            meta=dict(mod.meta or {}),
            intent_json=(dict(mod.intent_json) if isinstance(mod.intent_json, dict) else None),
        )
        mod.meta["extract_gpu_seen"] = True
        return mod

    monkeypatch.delenv("INTENTIR_MLIR_PASS_PLUGIN", raising=False)
    monkeypatch.setenv("INTENTIR_AUTO_MLIR_PASS_PLUGIN", "0")
    monkeypatch.setitem(__import__("intent_ir.mlir.pass_manager", fromlist=["PASS_REGISTRY"]).PASS_REGISTRY, "extract_gpu_module_llvm", _extract_gpu)

    result = _run_one_pass(
        module,
        "mlir-opt:pass-pipeline=builtin.module(intentir-extract-gpu-module-llvm-v1)",
        backend="cuda",
        toolchain={"tools": {"mlir-opt": {"path": ""}}},
    )
    assert result.kind == "python"
    assert result.module.meta["extract_gpu_seen"] is True
    assert result.module.meta["intentir_mlir_opt_fallback_passes"] == ["extract_gpu_module_llvm"]
