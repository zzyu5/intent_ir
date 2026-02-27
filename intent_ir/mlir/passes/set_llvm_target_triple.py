from __future__ import annotations

from intent_ir.mlir.module import IntentMLIRModule


def set_llvm_target_triple(module: IntentMLIRModule, *, backend: str | None = None, **_: object) -> IntentMLIRModule:
    """
    Inject `llvm.target_triple` module attribute for real-MLIR downstream pipelines.

    We keep this as a tiny text-level transform because the rest of the
    conversion pipeline is driven by mlir-opt/mlir-translate (no Python
    lowering). This attribute is required so downstream LLVM IR has a
    `target triple = "..."` line, which RVV remote execution uses to decide
    whether to compile remotely.
    """
    b = str(backend or "").strip().lower()
    if b.startswith("rvv") or b == "riscv":
        triple = "riscv64-unknown-linux-gnu"
    elif b.startswith("cuda"):
        triple = "nvptx64-nvidia-cuda"
    else:
        return module

    text = str(module.module_text or "")
    if "llvm.target_triple" in text:
        return module

    lines = text.splitlines()
    # Find the "module attributes {" block emitted by std_emitter.
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "module attributes {")
    except StopIteration:
        return module
    try:
        end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "} {")
    except StopIteration:
        return module

    attrs = [str(ln) for ln in lines[start + 1 : end]]
    attrs = [ln for ln in attrs if ln.strip()]
    # Strip trailing commas so we can re-render deterministically.
    stripped = [ln.rstrip().rstrip(",") for ln in attrs]
    stripped.append(f'  llvm.target_triple = "{triple}"')
    rendered: list[str] = []
    for i, ln in enumerate(stripped):
        suffix = "," if i < (len(stripped) - 1) else ""
        rendered.append(f"{ln}{suffix}")

    out_lines = list(lines[: start + 1]) + rendered + list(lines[end:])
    out = IntentMLIRModule(
        module_text="\n".join(out_lines) + ("\n" if text.endswith("\n") else ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    out.meta["llvm_target_triple_injected"] = str(triple)
    return out

