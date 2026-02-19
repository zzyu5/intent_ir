"""
Backend-neutral MLIR downstream execution contract.

This contract is the transitional runtime boundary between Intent MLIR passes
and backend compiler pipelines. It is intentionally JSON-serializable so it can
be persisted in run artifacts and reused by remote executors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MlirBackendContract:
    schema_version: str = "intent_mlir_backend_contract_v1"
    backend: str = ""
    kernel_name: str = ""
    io_spec: dict[str, Any] = field(default_factory=dict)
    launch: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    reason_context: dict[str, Any] = field(default_factory=dict)
    op_names: list[str] = field(default_factory=list)
    tensor_shapes: dict[str, list[Any]] = field(default_factory=dict)
    intent_json: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "backend": str(self.backend),
            "kernel_name": str(self.kernel_name),
            "io_spec": dict(self.io_spec or {}),
            "launch": dict(self.launch or {}),
            "schedule": dict(self.schedule or {}),
            "artifacts": dict(self.artifacts or {}),
            "reason_context": dict(self.reason_context or {}),
            "op_names": [str(x) for x in list(self.op_names or []) if str(x).strip()],
            "tensor_shapes": {str(k): list(v or []) for k, v in dict(self.tensor_shapes or {}).items()},
            "intent_json": (dict(self.intent_json) if isinstance(self.intent_json, dict) else None),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "MlirBackendContract":
        if not isinstance(payload, dict):
            raise TypeError("MlirBackendContract payload must be an object")
        schema = str(payload.get("schema_version") or "")
        if schema and schema != "intent_mlir_backend_contract_v1":
            raise ValueError(f"unsupported MlirBackendContract schema_version: {schema}")
        tensor_shapes: dict[str, list[Any]] = {}
        for k, v in dict(payload.get("tensor_shapes") or {}).items():
            if isinstance(v, list):
                tensor_shapes[str(k)] = list(v)
        return cls(
            schema_version="intent_mlir_backend_contract_v1",
            backend=str(payload.get("backend") or ""),
            kernel_name=str(payload.get("kernel_name") or ""),
            io_spec=dict(payload.get("io_spec") or {}),
            launch=dict(payload.get("launch") or {}),
            schedule=dict(payload.get("schedule") or {}),
            artifacts=dict(payload.get("artifacts") or {}),
            reason_context=dict(payload.get("reason_context") or {}),
            op_names=[str(x) for x in list(payload.get("op_names") or []) if str(x).strip()],
            tensor_shapes=tensor_shapes,
            intent_json=(dict(payload.get("intent_json")) if isinstance(payload.get("intent_json"), dict) else None),
        )


__all__ = ["MlirBackendContract"]
