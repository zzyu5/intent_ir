from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


@dataclass(frozen=True)
class MLIRBridgeMeta:
    source_kind: str
    mlir_parse_ms: float
    used_mlir_bridge: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_kind": str(self.source_kind),
            "mlir_parse_ms": float(self.mlir_parse_ms),
            "used_mlir_bridge": bool(self.used_mlir_bridge),
        }


def resolve_intent_payload_with_meta(payload: Any) -> tuple[Any, MLIRBridgeMeta]:
    t0 = time.perf_counter()
    if isinstance(payload, IntentMLIRModule):
        out = to_intent(payload)
        dt = float((time.perf_counter() - t0) * 1000.0)
        return out, MLIRBridgeMeta(source_kind="mlir_module", mlir_parse_ms=dt, used_mlir_bridge=True)
    if isinstance(payload, str):
        text = str(payload)
        if "intentir_json_begin" in text and "intentir_json_end" in text:
            out = to_intent(text)
            dt = float((time.perf_counter() - t0) * 1000.0)
            return out, MLIRBridgeMeta(source_kind="mlir_text", mlir_parse_ms=dt, used_mlir_bridge=True)
    dt = float((time.perf_counter() - t0) * 1000.0)
    return payload, MLIRBridgeMeta(source_kind="intent", mlir_parse_ms=dt, used_mlir_bridge=False)


def resolve_intent_payload(payload: Any):
    """
    Bridge helper for backend drivers during dual-track migration.

    Accepts either:
    - current IntentFunction payload (unchanged),
    - IntentMLIRModule,
    - MLIR text carrying intentir_json payload block.
    """
    if isinstance(payload, IntentMLIRModule):
        return to_intent(payload)
    if isinstance(payload, str):
        text = str(payload)
        if "intentir_json_begin" in text and "intentir_json_end" in text:
            return to_intent(text)
    return payload
