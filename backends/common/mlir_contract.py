"""
Backend-neutral MLIR downstream execution contract.

This contract is the transitional runtime boundary between Intent MLIR passes
and backend compiler pipelines. It is intentionally JSON-serializable so it can
be persisted in run artifacts and reused by remote executors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

CONTRACT_SCHEMA_V1 = "intent_mlir_backend_contract_v1"
CONTRACT_SCHEMA_V2 = "intent_mlir_backend_contract_v2"


@dataclass
class MlirExecutableArtifact:
    format: str = ""
    path: str = ""
    entry: str = ""
    target: str = ""
    toolchain_fingerprint: str = ""
    invocation: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format": str(self.format or ""),
            "path": str(self.path or ""),
            "entry": str(self.entry or ""),
            "target": str(self.target or ""),
            "toolchain_fingerprint": str(self.toolchain_fingerprint or ""),
            "invocation": dict(self.invocation or {}),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any] | None) -> "MlirExecutableArtifact":
        raw = dict(payload or {})
        return cls(
            format=str(raw.get("format") or ""),
            path=str(raw.get("path") or ""),
            entry=str(raw.get("entry") or ""),
            target=str(raw.get("target") or ""),
            toolchain_fingerprint=str(raw.get("toolchain_fingerprint") or ""),
            invocation=dict(raw.get("invocation") or {}),
        )


@dataclass
class MlirBackendContract:
    schema_version: str = CONTRACT_SCHEMA_V2
    backend: str = ""
    kernel_name: str = ""
    io_spec: dict[str, Any] = field(default_factory=dict)
    launch: dict[str, Any] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    reason_context: dict[str, Any] = field(default_factory=dict)
    op_names: list[str] = field(default_factory=list)
    tensor_shapes: dict[str, list[Any]] = field(default_factory=dict)
    executable: MlirExecutableArtifact = field(default_factory=MlirExecutableArtifact)

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
            "executable": self.executable.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "MlirBackendContract":
        if not isinstance(payload, dict):
            raise TypeError("MlirBackendContract payload must be an object")
        schema = str(payload.get("schema_version") or "")
        if schema in {"", CONTRACT_SCHEMA_V2}:
            normalized_schema = CONTRACT_SCHEMA_V2
        elif schema == CONTRACT_SCHEMA_V1:
            raise ValueError("unsupported legacy MlirBackendContract schema_version: intent_mlir_backend_contract_v1")
        else:
            raise ValueError(f"unsupported MlirBackendContract schema_version: {schema}")
        tensor_shapes: dict[str, list[Any]] = {}
        for k, v in dict(payload.get("tensor_shapes") or {}).items():
            if isinstance(v, list):
                tensor_shapes[str(k)] = list(v)
        return cls(
            schema_version=normalized_schema,
            backend=str(payload.get("backend") or ""),
            kernel_name=str(payload.get("kernel_name") or ""),
            io_spec=dict(payload.get("io_spec") or {}),
            launch=dict(payload.get("launch") or {}),
            schedule=dict(payload.get("schedule") or {}),
            artifacts=dict(payload.get("artifacts") or {}),
            reason_context=dict(payload.get("reason_context") or {}),
            op_names=[str(x) for x in list(payload.get("op_names") or []) if str(x).strip()],
            tensor_shapes=tensor_shapes,
            executable=MlirExecutableArtifact.from_json_dict(
                payload.get("executable") if isinstance(payload.get("executable"), Mapping) else {}
            ),
        )


__all__ = [
    "CONTRACT_SCHEMA_V1",
    "CONTRACT_SCHEMA_V2",
    "MlirBackendContract",
    "MlirExecutableArtifact",
]
