from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from intent_ir.mlir.module import IntentMLIRModule
from intent_ir.mlir.passes.emit_cuda_contract import build_cuda_contract
from intent_ir.mlir.passes.emit_rvv_contract import build_rvv_contract


def _dump_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _emit_contract(
    *,
    backend: str,
    spec_name: str,
    out_dir: Path,
    module: IntentMLIRModule,
    suffix: str,
) -> str:
    if backend == "cuda":
        contract = build_cuda_contract(module, source_kind="mlir_module")
    elif backend == "rvv":
        contract = build_rvv_contract(module, source_kind="mlir_module")
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported backend: {backend}")
    contract_path = out_dir / f"{spec_name}.intentir.intentdialect.{suffix}.contract.json"
    return _dump_json(contract_path, contract.to_json_dict())


def emit_backend_contract_artifacts(
    *,
    spec_name: str,
    out_dir: Path,
    midend_module: IntentMLIRModule,
    mlir_report: dict[str, Any],
    downstream_name: str | None = None,
    downstream_module: IntentMLIRModule | None = None,
) -> dict[str, Any]:
    """
    Emit MLIR backend contracts into stable JSON artifacts and annotate report fields.

    Fields written into `mlir_report`:
      - `midend_cuda_contract_path`
      - `midend_rvv_contract_path`
      - `downstream_cuda_contract_path` / `downstream_rvv_contract_path` (if available)
      - `downstream_contract_path`, `downstream_contract_backend`
      - `mlir_backend_contract_used`
    """
    emitted: dict[str, Any] = {}

    # Always emit contracts from midend so tools can recover even when a specific
    # downstream pipeline is skipped.
    for backend in ("cuda", "rvv"):
        try:
            p = _emit_contract(
                backend=backend,
                spec_name=spec_name,
                out_dir=out_dir,
                module=midend_module,
                suffix=f"midend_{backend}",
            )
            emitted[f"midend_{backend}_contract_path"] = p
        except Exception as e:  # pragma: no cover - defensive
            emitted[f"midend_{backend}_contract_error"] = f"{type(e).__name__}: {e}"

    if downstream_module is not None and str(downstream_name or "") in {"downstream_cuda", "downstream_rvv"}:
        backend = "cuda" if str(downstream_name) == "downstream_cuda" else "rvv"
        try:
            p = _emit_contract(
                backend=backend,
                spec_name=spec_name,
                out_dir=out_dir,
                module=downstream_module,
                suffix=str(downstream_name),
            )
            emitted[f"{downstream_name}_contract_path"] = p
            emitted["downstream_contract_path"] = p
            emitted["downstream_contract_backend"] = backend
        except Exception as e:  # pragma: no cover - defensive
            emitted[f"{downstream_name}_contract_error"] = f"{type(e).__name__}: {e}"

    if any(str(k).endswith("_contract_path") for k in emitted):
        emitted["mlir_backend_contract_used"] = True

    mlir_report.update(emitted)
    return emitted


__all__ = ["emit_backend_contract_artifacts"]
