from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentMLIRModule:
    """
    Transitional MLIR carrier for IntentIR migration.

    During dual-track migration we keep both:
    - `module_text`: textual IR (intent dialect flavored)
    - `intent_json`: lossless bridge back to IntentFunction
    """

    module_text: str
    dialect_version: str = "intent_dialect_v0"
    provenance: dict[str, Any] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    intent_json: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "intent_mlir_module_v1",
            "dialect_version": str(self.dialect_version),
            "module_text": str(self.module_text),
            "provenance": dict(self.provenance or {}),
            "symbols": list(self.symbols or []),
            "meta": dict(self.meta or {}),
            "intent_json": (dict(self.intent_json) if isinstance(self.intent_json, dict) else None),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "IntentMLIRModule":
        if not isinstance(payload, dict):
            raise TypeError("IntentMLIRModule payload must be an object")
        return cls(
            module_text=str(payload.get("module_text") or ""),
            dialect_version=str(payload.get("dialect_version") or "intent_dialect_v0"),
            provenance=dict(payload.get("provenance") or {}),
            symbols=[str(x) for x in list(payload.get("symbols") or []) if str(x).strip()],
            meta=dict(payload.get("meta") or {}),
            intent_json=(dict(payload.get("intent_json")) if isinstance(payload.get("intent_json"), dict) else None),
        )

