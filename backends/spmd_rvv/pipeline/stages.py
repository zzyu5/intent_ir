"""
RVV pipeline stage data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RVV_PIPELINE_STAGES: tuple[str, ...] = (
    "legalize",
    "shape_infer",
    "schedule",
    "emit_cpp",
    "compile",
    "run",
)


@dataclass
class RvvPipelineStage:
    name: str
    ok: bool
    ms: float = 0.0
    detail: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RvvPipelineResult:
    ok: bool
    stages: list[RvvPipelineStage]
    reason_code: str = "ok"
    reason_detail: str = ""
    input_ir_kind: str = "intent"
    mlir_parse_ms: float = 0.0
    mlir_backend_contract_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reason_code": str(self.reason_code),
            "reason_detail": str(self.reason_detail),
            "input_ir_kind": str(self.input_ir_kind),
            "mlir_parse_ms": float(self.mlir_parse_ms),
            "mlir_backend_contract_used": bool(self.mlir_backend_contract_used),
            "stages": [
                {
                    "name": s.name,
                    "ok": bool(s.ok),
                    "ms": float(s.ms),
                    "detail": str(s.detail),
                    "artifacts": dict(s.artifacts),
                }
                for s in self.stages
            ],
        }
