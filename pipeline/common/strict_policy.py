from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from pipeline.common.evidence_mode import evidence_mode


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SCHEMA_VERSION = "intent_mlir_backend_contract_v2"


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(str(name), "")
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if not s:
        return bool(default)
    return s in {"1", "true", "yes", "y", "on"}


def strict_fallback_enabled() -> bool:
    policy = str(os.getenv("INTENTIR_FALLBACK_POLICY", "strict")).strip().lower()
    if policy in {"legacy", "compat", "deterministic"}:
        return False
    return True


def cuda_require_llvm_ptx() -> bool:
    raw = os.getenv("INTENTIR_CUDA_REQUIRE_LLVM_PTX", "")
    if str(raw).strip():
        return _env_flag("INTENTIR_CUDA_REQUIRE_LLVM_PTX", default=False)
    return bool(strict_fallback_enabled())


def runtime_fallback_from_artifacts(
    artifacts: Mapping[str, Any] | None,
    *,
    backend: str | None = None,
) -> tuple[bool, str]:
    if not isinstance(artifacts, Mapping):
        return False, ""
    tags: list[str] = []
    explicit = str(artifacts.get("runtime_fallback") or "").strip()
    if explicit:
        tags.append(explicit)
    ctx_fallback = str(artifacts.get("intent_recovery_fallback") or "").strip()
    if ctx_fallback:
        tags.append(f"intent_recovery_fallback={ctx_fallback}")
    backend_norm = str(backend or "").strip().lower()
    ptx_origin = str(artifacts.get("cuda_ptx_origin") or "").strip().lower()
    if (backend_norm == "cuda" or ptx_origin) and ptx_origin and ptx_origin != "llvm_llc":
        tags.append(f"cuda_ptx_origin={ptx_origin}")
    rvv_src_origin = str(artifacts.get("rvv_kernel_src_origin") or "").strip()
    if (backend_norm == "rvv" or rvv_src_origin) and rvv_src_origin:
        if "compat_cpp_codegen" in rvv_src_origin or "cpp_codegen" in rvv_src_origin:
            tags.append(f"rvv_kernel_src_origin={rvv_src_origin}")
    if not tags:
        return False, ""
    dedup = sorted(set([str(x) for x in tags if str(x).strip()]))
    return bool(dedup), ",".join(dedup)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_contract_path(mlir_report: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(mlir_report, Mapping):
        return None
    cand = [
        str(mlir_report.get("downstream_llvm_contract_path") or ""),
        str(mlir_report.get("downstream_cuda_llvm_contract_path") or ""),
        str(mlir_report.get("downstream_rvv_llvm_contract_path") or ""),
        str(mlir_report.get("downstream_contract_path") or ""),
    ]
    for raw in cand:
        s = str(raw).strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.is_file():
            return p
    return None


def enrich_frontend_report_with_strict_fields(
    report: dict[str, Any],
    *,
    mlir_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    strict_mode = bool(strict_fallback_enabled())
    fallback_policy = "strict" if strict_mode else "legacy_compatible"
    report["execution_ir"] = "mlir"
    report["execution_engine"] = "mlir_native"
    report["fallback_policy"] = str(fallback_policy)
    report["strict_mode"] = bool(strict_mode)
    report["intentir_evidence_mode"] = str(evidence_mode())

    contract_schema = CONTRACT_SCHEMA_VERSION
    runtime_fallback = False
    runtime_fallback_detail = ""

    contract_path = _resolve_contract_path(mlir_report)
    if contract_path is not None:
        payload = _load_json(contract_path)
        contract_schema = str(payload.get("schema_version") or contract_schema)
        backend = str(payload.get("backend") or "").strip().lower()
        art = payload.get("artifacts")
        runtime_fallback, runtime_fallback_detail = runtime_fallback_from_artifacts(art, backend=backend)
        out["downstream_llvm_contract_path"] = str(contract_path)
        out["contract_backend"] = str(backend)
    else:
        if isinstance(report.get("runtime_fallback"), bool):
            runtime_fallback = bool(report.get("runtime_fallback"))
        runtime_fallback_detail = str(report.get("runtime_fallback_detail") or "")
    report["contract_schema_version"] = str(contract_schema or CONTRACT_SCHEMA_VERSION)
    report["runtime_fallback"] = bool(runtime_fallback)
    report["runtime_fallback_detail"] = str(runtime_fallback_detail or "")
    return out
