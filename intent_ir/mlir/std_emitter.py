from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from intent_ir.ir import IntentFunction
from intent_ir.utils.repo_state import repo_state

from .module import IntentMLIRModule

ROOT = Path(__file__).resolve().parents[2]

_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$.]*$")


def emit_std_mlir(
    intent: IntentFunction,
    *,
    provenance: dict[str, Any] | None = None,
    include_json_payload: bool = True,
) -> IntentMLIRModule:
    """
    Emit a parseable, toolchain-friendly MLIR module using only registered
    upstream dialects (builtin/func/arith/tensor/linalg).

    This is intentionally a *carrier* representation: it embeds the full
    IntentFunction JSON payload in module attributes (and optionally as a
    comment block) so Python passes can round-trip losslessly during
    migration, while keeping the textual IR parsable by `mlir-opt` without
    `--allow-unregistered-dialect`.
    """
    payload = intent.to_json_dict()
    symbols = _collect_symbols(payload)
    prov = dict(provenance or {})
    prov.setdefault("repo", repo_state(root=ROOT))
    prov.setdefault("source", "intent_function")

    encoded = ""
    if include_json_payload:
        encoded = base64.b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).decode("ascii")

    # Minimal stub body: keep it valid + parsable, but do not claim semantics.
    # Later phases will replace cached LLVM IR with real MLIR lowering.
    fn_sym = _mlir_symbol(intent.name)
    attrs: list[str] = []
    attrs.append('intentir.format = "std_mlir_v1"')
    attrs.append(f'intentir.intent_name = "{_escape_mlir_string(intent.name)}"')
    # Shape bindings are required for backend lowering and should be available
    # to non-Python MLIR passes via module attributes.
    bindings_raw = (intent.meta or {}).get("shape_bindings")
    if isinstance(bindings_raw, dict) and bindings_raw:
        bindings: dict[str, int] = {}
        for k, v in dict(bindings_raw).items():
            key = str(k).strip()
            if not key:
                continue
            try:
                bindings[key] = int(v)
            except Exception:
                continue
        if bindings:
            b64 = base64.b64encode(
                json.dumps(bindings, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).decode("ascii")
            attrs.append(f'intentir.shape_bindings_b64 = "{b64}"')
    if include_json_payload:
        attrs.append(f'intentir.intent_json_b64 = "{encoded}"')
    if symbols:
        sym_items = ", ".join(f"\"{_escape_mlir_string(s)}\"" for s in symbols)
        attrs.append(f"intentir.symbols = [{sym_items}]")

    lines: list[str] = []
    if attrs:
        lines.append("module attributes {")
        for i, a in enumerate(attrs):
            suffix = "," if i < (len(attrs) - 1) else ""
            lines.append(f"  {a}{suffix}")
        lines.append("} {")
    else:
        lines.append("module {")

    lines.append(f"  func.func @{fn_sym}() {{")
    lines.append("    %c0 = arith.constant 0 : i32")
    lines.append("    %init = tensor.from_elements %c0 : tensor<1xi32>")
    lines.append("    %t1 = linalg.fill ins(%c0 : i32) outs(%init : tensor<1xi32>) -> tensor<1xi32>")
    lines.append("    return")
    lines.append("  }")

    if include_json_payload:
        lines.append("  // intentir_json_begin")
        lines.append(f"  // {encoded}")
        lines.append("  // intentir_json_end")

    lines.append("}")
    text = "\n".join(lines) + "\n"

    return IntentMLIRModule(
        module_text=text,
        dialect_version="std_mlir_v1",
        provenance=prov,
        symbols=symbols,
        meta={"bridge_format": "std_mlir_v1"},
        intent_json=(payload if include_json_payload else None),
    )


def _mlir_symbol(name: str) -> str:
    n = str(name or "").strip() or "intent_fn"
    if _SYMBOL_RE.match(n):
        return n
    return f"\"{_escape_mlir_string(n)}\""


def _escape_mlir_string(s: str) -> str:
    # Conservative escaping for MLIR string attributes / quoted symbols.
    return str(s).replace("\\", "\\\\").replace("\"", "\\\"")


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
