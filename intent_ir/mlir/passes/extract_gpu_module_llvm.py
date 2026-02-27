from __future__ import annotations

from intent_ir.mlir.module import IntentMLIRModule


def extract_gpu_module_llvm(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    """
    Text-level helper: move LLVM/NVVM IR out of `gpu.module` so `mlir-translate`
    can emit textual LLVM IR.

    `mlir-translate --mlir-to-llvmir` ignores nested `gpu.module` bodies; we keep
    the surrounding `module attributes { ... }` and splice the `gpu.module`
    contents into the parent.

    This is intentionally simple and conservative: if the input does not
    contain a `gpu.module`, return it unchanged.
    """
    text = str(module.module_text or "")
    if "gpu.module" not in text:
        return module

    lines = text.splitlines()
    # Drop trailing empty lines to make "all but last N" slicing stable.
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return module

    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("gpu.module ") and "{" in ln:
            start = i
            break
    if start is None:
        return module

    # Brace-count scan from the gpu.module line. This counts all `{`/`}` chars,
    # including those in nested attribute dicts, which are properly nested too.
    depth = 0
    end = None
    for i in range(start, len(lines)):
        ln = lines[i]
        depth += ln.count("{")
        depth -= ln.count("}")
        if i > start and depth == 0:
            end = i
            break
    if end is None or end <= start:
        return module

    indent = len(lines[start]) - len(lines[start].lstrip(" "))
    inner_indent = " " * (indent + 2)
    inner: list[str] = []
    for ln in lines[start + 1 : end]:
        if ln.startswith(inner_indent):
            inner.append(ln[len(inner_indent) :])
        else:
            inner.append(ln.lstrip("\n"))

    out_lines = list(lines[:start]) + inner + list(lines[end + 1 :])
    out = IntentMLIRModule(
        module_text="\n".join(out_lines) + "\n",
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    out.meta["gpu_module_extracted"] = True
    return out

