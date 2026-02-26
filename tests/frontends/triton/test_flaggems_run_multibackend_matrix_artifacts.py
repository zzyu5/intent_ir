from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "flaggems" / "run_multibackend_matrix.py"
    spec = importlib.util.spec_from_file_location("run_multibackend_matrix", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_mlir_llvm_artifacts_prefers_cuda_contract_for_cuda_pipeline(tmp_path: Path) -> None:
    mod = _load_module()

    provider_report_dir = tmp_path / "pipeline_reports"
    provider_report_dir.mkdir(parents=True, exist_ok=True)

    llvm_path = provider_report_dir / "k.intentir.intentdialect.downstream_cuda_llvm.ll"
    llvm_path.write_text(
        '; ModuleID = "k"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "define void @k() { ret void }\n",
        encoding="utf-8",
    )

    cuda_contract = provider_report_dir / "k.intentir.intentdialect.downstream_cuda_llvm.contract.json"
    cuda_contract.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "cuda",
                "artifacts": {"cuda_ptx_origin": "llvm_llc"},
                "executable": {"invocation": {}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rvv_contract = provider_report_dir / "k.intentir.intentdialect.downstream_rvv_llvm.contract.json"
    rvv_contract.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "rvv",
                "artifacts": {"rvv_kernel_src_origin": "compat_cpp_codegen"},
                "executable": {"invocation": {}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = provider_report_dir / "k.json"
    report_path.write_text(
        json.dumps(
            {
                "mlir": {
                    "llvm_pipeline": "downstream_cuda_llvm",
                    "llvm_emit_ok": True,
                    "llvm_ir_path": str(llvm_path),
                    "downstream_cuda_llvm_contract_path": str(cuda_contract),
                    # Keep a conflicting generic pointer to RVV path to verify we do
                    # not use it for CUDA llvm pipeline evidence.
                    "downstream_llvm_contract_path": str(rvv_contract),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "mlir_llvm_artifacts.json"
    payload = mod._collect_mlir_llvm_artifacts(
        provider_report_dir=provider_report_dir,
        kernels=["k"],
        out_path=out_path,
    )

    assert bool(payload.get("artifact_complete")) is True
    rows = list(payload.get("entries") or [])
    assert len(rows) == 1
    row = dict(rows[0])
    assert bool(row.get("ok")) is True
    assert bool(row.get("runtime_fallback")) is False
    assert str(row.get("reason_code") or "") == "ok"


def test_collect_mlir_llvm_artifacts_requires_pipeline_specific_contract(tmp_path: Path) -> None:
    mod = _load_module()

    provider_report_dir = tmp_path / "pipeline_reports"
    provider_report_dir.mkdir(parents=True, exist_ok=True)

    llvm_path = provider_report_dir / "k.intentir.intentdialect.downstream_cuda_llvm.ll"
    llvm_path.write_text(
        '; ModuleID = "k"\n'
        'target triple = "nvptx64-nvidia-cuda"\n'
        "define void @k() { ret void }\n",
        encoding="utf-8",
    )

    # Intentional: downstream_cuda_llvm_contract_path is absent.
    rvv_contract = provider_report_dir / "k.intentir.intentdialect.downstream_rvv_llvm.contract.json"
    rvv_contract.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "rvv",
                "artifacts": {"rvv_kernel_src_origin": "compat_cpp_codegen"},
                "executable": {"invocation": {}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = provider_report_dir / "k.json"
    report_path.write_text(
        json.dumps(
            {
                "mlir": {
                    "llvm_pipeline": "downstream_cuda_llvm",
                    "llvm_emit_ok": True,
                    "llvm_ir_path": str(llvm_path),
                    "downstream_llvm_contract_path": str(rvv_contract),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "mlir_llvm_artifacts.json"
    payload = mod._collect_mlir_llvm_artifacts(
        provider_report_dir=provider_report_dir,
        kernels=["k"],
        out_path=out_path,
    )
    assert bool(payload.get("artifact_complete")) is False
    rows = list(payload.get("entries") or [])
    assert len(rows) == 1
    row = dict(rows[0])
    assert str(row.get("reason_code") or "") == "llvm_contract_missing"


def test_collect_mlir_llvm_artifacts_strict_cuda_prefers_cuda_contract_even_if_primary_rvv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    monkeypatch.setenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", "1")

    provider_report_dir = tmp_path / "pipeline_reports"
    provider_report_dir.mkdir(parents=True, exist_ok=True)

    rvv_llvm_path = provider_report_dir / "k.intentir.intentdialect.downstream_rvv_llvm.ll"
    rvv_llvm_path.write_text(
        '; ModuleID = "k"\n'
        'target triple = "riscv64-unknown-linux-gnu"\n'
        "define void @k() { ret void }\n",
        encoding="utf-8",
    )

    rvv_contract = provider_report_dir / "k.intentir.intentdialect.downstream_rvv_llvm.contract.json"
    rvv_contract.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "rvv",
                "artifacts": {"rvv_kernel_src_origin": "compat_cpp_codegen"},
                "executable": {"invocation": {}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cuda_contract = provider_report_dir / "k.intentir.intentdialect.downstream_cuda_llvm.contract.json"
    cuda_contract.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "cuda",
                "artifacts": {"cuda_ptx_origin": "llvm_llc"},
                "executable": {"invocation": {}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = provider_report_dir / "k.json"
    report_path.write_text(
        json.dumps(
            {
                "mlir": {
                    "llvm_pipeline": "downstream_rvv_llvm",
                    "llvm_emit_ok": True,
                    "llvm_ir_path": str(rvv_llvm_path),
                    "downstream_rvv_llvm_contract_path": str(rvv_contract),
                    "downstream_cuda_llvm_contract_path": str(cuda_contract),
                    "downstream_cuda_llvm": {"ok": True, "passes": []},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "mlir_llvm_artifacts.json"
    payload = mod._collect_mlir_llvm_artifacts(
        provider_report_dir=provider_report_dir,
        kernels=["k"],
        out_path=out_path,
    )
    assert bool(payload.get("artifact_complete")) is True
    rows = list(payload.get("entries") or [])
    assert len(rows) == 1
    row = dict(rows[0])
    assert str(row.get("llvm_pipeline") or "") == "downstream_cuda_llvm"
    assert bool(row.get("ok")) is True
    assert bool(row.get("runtime_fallback")) is False
    assert str(row.get("reason_code") or "") == "ok"


def test_collect_mlir_llvm_artifacts_strict_cuda_requires_cuda_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _load_module()
    monkeypatch.setenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", "1")

    provider_report_dir = tmp_path / "pipeline_reports"
    provider_report_dir.mkdir(parents=True, exist_ok=True)

    rvv_llvm_path = provider_report_dir / "k.intentir.intentdialect.downstream_rvv_llvm.ll"
    rvv_llvm_path.write_text(
        '; ModuleID = "k"\n'
        'target triple = "riscv64-unknown-linux-gnu"\n'
        "define void @k() { ret void }\n",
        encoding="utf-8",
    )

    rvv_contract = provider_report_dir / "k.intentir.intentdialect.downstream_rvv_llvm.contract.json"
    rvv_contract.write_text(
        json.dumps(
            {
                "schema_version": "intent_mlir_backend_contract_v2",
                "backend": "rvv",
                "artifacts": {"rvv_kernel_src_origin": "llvm_llc"},
                "executable": {"invocation": {}},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = provider_report_dir / "k.json"
    report_path.write_text(
        json.dumps(
            {
                "mlir": {
                    "llvm_pipeline": "downstream_rvv_llvm",
                    "llvm_emit_ok": True,
                    "llvm_ir_path": str(rvv_llvm_path),
                    "downstream_rvv_llvm_contract_path": str(rvv_contract),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "mlir_llvm_artifacts.json"
    payload = mod._collect_mlir_llvm_artifacts(
        provider_report_dir=provider_report_dir,
        kernels=["k"],
        out_path=out_path,
    )
    assert bool(payload.get("artifact_complete")) is False
    rows = list(payload.get("entries") or [])
    assert len(rows) == 1
    row = dict(rows[0])
    assert str(row.get("reason_code") or "") == "llvm_contract_missing"


def test_project_status_converged_to_scope_rewrites_primary_counts(tmp_path: Path) -> None:
    mod = _load_module()

    status_path = tmp_path / "status_converged.json"
    status_path.write_text(
        json.dumps(
            {
                "scope_enabled": True,
                "counts_global": {"blocked_backend": 10, "dual_pass": 2},
                "counts_by_status": {"blocked_backend": 10, "dual_pass": 2},
                "global_entries_count": 12,
                "determinability_ok_count": 2,
                "artifact_complete_count": 2,
                "runtime_fallback_kernel_count": 3,
                "runtime_fallback_kernels": ["x", "y", "z"],
                "entries": [{"semantic_op": "old"}],
                "scoped_entries": [
                    {
                        "semantic_op": "row_max",
                        "e2e_spec": "row_max",
                        "status": "dual_pass",
                        "determinability": True,
                        "artifact_complete": True,
                        "runtime_fallback": False,
                    },
                    {
                        "semantic_op": "row_all",
                        "e2e_spec": "row_all",
                        "status": "dual_pass",
                        "determinability": True,
                        "artifact_complete": True,
                        "runtime_fallback": False,
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    changed = mod._project_status_converged_to_scope(status_converged_path=status_path)
    assert changed is True

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["scope_projection_applied"] is True
    assert payload["scope_projection_source"] == "scoped_entries"
    assert payload["registry_counts_global"] == {"blocked_backend": 10, "dual_pass": 2}
    assert payload["registry_entries_count"] == 12
    assert payload["counts_global"] == {"dual_pass": 2}
    assert payload["counts_by_status"] == {"dual_pass": 2}
    assert payload["global_entries_count"] == 2
    assert payload["runtime_fallback_kernel_count"] == 0
    assert payload["runtime_fallback_kernels"] == []
    assert len(payload["entries"]) == 2
