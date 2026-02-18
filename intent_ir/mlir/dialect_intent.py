from __future__ import annotations

from typing import Any

from intent_ir.ops.specs import op_spec_for


def intent_dialect_op_name(op_name: str) -> str:
    """
    Map IntentIR op name to intent dialect op symbol.
    """
    name = str(op_name or "").strip()
    return f"intent.{name}" if name else "intent.unknown"


def op_kind(op_name: str) -> str:
    spec = op_spec_for(str(op_name))
    if spec is None:
        return "unknown"
    return str(spec.kind or "unknown")


def op_tier(op_name: str) -> str:
    spec = op_spec_for(str(op_name))
    if spec is None:
        return "unknown"
    return str(spec.tier or "unknown")


def op_meta(op_name: str) -> dict[str, Any]:
    return {
        "dialect_op": intent_dialect_op_name(op_name),
        "kind": op_kind(op_name),
        "tier": op_tier(op_name),
    }

