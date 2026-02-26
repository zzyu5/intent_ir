from __future__ import annotations

import json
from pathlib import Path

from pipeline.common import strict_policy as sp


def test_strict_fallback_enabled_defaults_to_true(monkeypatch) -> None:
    monkeypatch.delenv("INTENTIR_FALLBACK_POLICY", raising=False)
    assert sp.strict_fallback_enabled() is True


def test_strict_fallback_enabled_supports_legacy_compat(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "legacy")
    assert sp.strict_fallback_enabled() is False
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "compat")
    assert sp.strict_fallback_enabled() is False
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "strict")
    assert sp.strict_fallback_enabled() is True


def test_cuda_require_llvm_ptx_respects_explicit_env_override(monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "legacy")
    monkeypatch.delenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", raising=False)
    assert sp.cuda_require_llvm_ptx() is False

    monkeypatch.setenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", "1")
    assert sp.cuda_require_llvm_ptx() is True

    monkeypatch.setenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", "0")
    assert sp.cuda_require_llvm_ptx() is False


def test_runtime_fallback_from_artifacts_detects_cuda_and_rvv_tags() -> None:
    cuda_fb, cuda_detail = sp.runtime_fallback_from_artifacts(
        {"cuda_ptx_origin": "nvrtc_fallback_from_llvm"},
        backend="cuda",
    )
    assert cuda_fb is True
    assert "cuda_ptx_origin=nvrtc_fallback_from_llvm" in cuda_detail

    rvv_fb, rvv_detail = sp.runtime_fallback_from_artifacts(
        {"rvv_kernel_src_origin": "compat_cpp_codegen"},
        backend="rvv",
    )
    assert rvv_fb is True
    assert "rvv_kernel_src_origin=compat_cpp_codegen" in rvv_detail


def test_enrich_frontend_report_with_strict_fields_reads_contract_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "strict")
    contract_path = tmp_path / "downstream_cuda_llvm.contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "cuda",
                "artifacts": {
                    "cuda_ptx_origin": "llvm_llc",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {}
    mlir_report = {"downstream_llvm_contract_path": str(contract_path)}
    out = sp.enrich_frontend_report_with_strict_fields(report, mlir_report=mlir_report)
    assert report["execution_ir"] == "mlir"
    assert report["execution_engine"] == "mlir_native"
    assert report["strict_mode"] is True
    assert report["fallback_policy"] == "strict"
    assert report["contract_schema_version"] == "intent_mlir_backend_contract_v2"
    assert report["runtime_fallback"] is False
    assert isinstance(out.get("downstream_llvm_contract_path"), str)
