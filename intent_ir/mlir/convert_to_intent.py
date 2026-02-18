from __future__ import annotations

import base64
import json
import re
from typing import Any

from intent_ir.ir import IntentFunction

from .module import IntentMLIRModule

_JSON_BLOCK_RE = re.compile(
    r"intentir_json_begin\s*\n\s*//\s*(?P<payload>[A-Za-z0-9+/=]+)\s*\n\s*//\s*intentir_json_end",
    re.S,
)


def to_intent(module_or_text: IntentMLIRModule | str) -> IntentFunction:
    if isinstance(module_or_text, IntentMLIRModule):
        if isinstance(module_or_text.intent_json, dict):
            return IntentFunction.from_json_dict(module_or_text.intent_json)
        text = str(module_or_text.module_text or "")
    else:
        text = str(module_or_text or "")

    payload = _extract_intent_json_payload(text)
    return IntentFunction.from_json_dict(payload)


def _extract_intent_json_payload(text: str) -> dict[str, Any]:
    m = _JSON_BLOCK_RE.search(str(text or ""))
    if not m:
        raise ValueError(
            "cannot recover IntentFunction from MLIR text: missing intentir_json payload block; "
            "run conversion via intent_ir.mlir.convert_from_intent.to_mlir first"
        )
    raw = str(m.group("payload") or "").strip()
    try:
        decoded = base64.b64decode(raw.encode("ascii")).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as e:
        raise ValueError(f"invalid intentir_json payload block: {type(e).__name__}: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError("decoded intent payload is not a JSON object")
    return payload

