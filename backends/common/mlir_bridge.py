from __future__ import annotations

from typing import Any

from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


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

