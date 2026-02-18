from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from intent_ir.ir import IntentFunction
from intent_ir.utils.repo_state import repo_state

from .dialect_intent import intent_dialect_op_name, op_meta
from .module import IntentMLIRModule
from .types import attrs_to_mlir_dict, tensor_type_to_mlir

ROOT = Path(__file__).resolve().parents[2]


def to_mlir(
    intent: IntentFunction,
    *,
    provenance: dict[str, Any] | None = None,
    include_json_payload: bool = True,
) -> IntentMLIRModule:
    payload = intent.to_json_dict()
    symbols = _collect_symbols(payload)
    prov = dict(provenance or {})
    prov.setdefault("repo", repo_state(root=ROOT))
    prov.setdefault("source", "intent_function")

    lines: list[str] = []
    lines.append("module attributes {intent.dialect_version = \"intent_dialect_v0\"} {")
    lines.append(f"  intent.func @{intent.name}() {{")

    for op in list(intent.ops or []):
        opn = str(op.op)
        opn_dialect = intent_dialect_op_name(opn)
        meta = op_meta(opn)
        inps = ", ".join(f"%{x}" for x in list(op.inputs or []))
        attrs = dict(op.attrs or {})
        attrs["kind"] = meta.get("kind")
        attrs["tier"] = meta.get("tier")
        attrs_txt = attrs_to_mlir_dict(attrs)
        out_name = str(op.output)
        out_type = tensor_type_to_mlir(intent.tensors[out_name]) if out_name in intent.tensors else "!intent.unknown"
        if inps:
            lines.append(f"    %{out_name} = {opn_dialect}({inps}) {attrs_txt} : {out_type}")
        else:
            lines.append(f"    %{out_name} = {opn_dialect}() {attrs_txt} : {out_type}")

    if list(intent.outputs or []):
        outs = ", ".join(f"%{x}" for x in list(intent.outputs or []))
        lines.append(f"    intent.return {outs}")
    else:
        lines.append("    intent.return")
    lines.append("  }")

    if include_json_payload:
        encoded = base64.b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        lines.append("  // intentir_json_begin")
        lines.append(f"  // {encoded}")
        lines.append("  // intentir_json_end")

    lines.append("}")
    text = "\n".join(lines) + "\n"
    return IntentMLIRModule(
        module_text=text,
        dialect_version="intent_dialect_v0",
        provenance=prov,
        symbols=symbols,
        meta={"bridge_format": "intent_json_base64_comment_v1"},
        intent_json=(payload if include_json_payload else None),
    )


def _collect_symbols(intent_json: dict[str, Any]) -> list[str]:
    out: list[str] = []
    tensors = dict(intent_json.get("tensors") or {})
    for t in tensors.values():
        if not isinstance(t, dict):
            continue
        for d in list(t.get("shape") or []):
            if isinstance(d, dict):
                if str(d.get("kind") or "") == "sym":
                    v = str(d.get("value") or "").strip()
                    if v and v not in out:
                        out.append(v)
                continue
            if isinstance(d, str) and d.strip() and d.strip() not in out:
                out.append(d.strip())
    return out

