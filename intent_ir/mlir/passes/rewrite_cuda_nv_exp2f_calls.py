from __future__ import annotations

import re

from intent_ir.mlir.module import IntentMLIRModule


_CALL_EXP2_RE = re.compile(r"llvm\.call\s+@__nv_exp2f\b")
_CALL_EXP_RE = re.compile(r"llvm\.call\s+@__nv_expf\b")
_ARITH_MAXF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dst>%[A-Za-z0-9_$.]+)\s*=\s*arith\.maxf\s+(?P<a>%[A-Za-z0-9_$.]+),\s*(?P<b>%[A-Za-z0-9_$.]+)\s*:\s*(?P<ty>[^\n]+)$",
    re.MULTILINE,
)
_ARITH_MINF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dst>%[A-Za-z0-9_$.]+)\s*=\s*arith\.minf\s+(?P<a>%[A-Za-z0-9_$.]+),\s*(?P<b>%[A-Za-z0-9_$.]+)\s*:\s*(?P<ty>[^\n]+)$",
    re.MULTILINE,
)
_ARITH_MAXSI_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dst>%[A-Za-z0-9_$.]+)\s*=\s*arith\.maxsi\s+(?P<a>%[A-Za-z0-9_$.]+),\s*(?P<b>%[A-Za-z0-9_$.]+)\s*:\s*(?P<ty>[^\n]+)$",
    re.MULTILINE,
)
_ARITH_MINSI_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dst>%[A-Za-z0-9_$.]+)\s*=\s*arith\.minsi\s+(?P<a>%[A-Za-z0-9_$.]+),\s*(?P<b>%[A-Za-z0-9_$.]+)\s*:\s*(?P<ty>[^\n]+)$",
    re.MULTILINE,
)
_ARITH_MAXUI_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dst>%[A-Za-z0-9_$.]+)\s*=\s*arith\.maxui\s+(?P<a>%[A-Za-z0-9_$.]+),\s*(?P<b>%[A-Za-z0-9_$.]+)\s*:\s*(?P<ty>[^\n]+)$",
    re.MULTILINE,
)
_ARITH_MINUI_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dst>%[A-Za-z0-9_$.]+)\s*=\s*arith\.minui\s+(?P<a>%[A-Za-z0-9_$.]+),\s*(?P<b>%[A-Za-z0-9_$.]+)\s*:\s*(?P<ty>[^\n]+)$",
    re.MULTILINE,
)
_MATH_ERF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<dst>%[A-Za-z0-9_$.]+)\s*=\s*math\.erf\s+(?P<a>%[A-Za-z0-9_$.]+)\s*:\s*f32\s*$",
    re.MULTILINE,
)
_ERFF_DECL_RE = re.compile(r"^\s*llvm\.func\s+@erff\b", re.MULTILINE)
_FIRST_LLVM_FUNC_RE = re.compile(r"^(?P<indent>\s*)llvm\.func\b", re.MULTILINE)


def rewrite_cuda_nv_exp2f_calls(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    """
    Post-lowering cleanup for CUDA device IR.

    Some MLIR lowering paths materialize `math.exp2` as a libdevice call
    (`__nv_exp2f`). This is correct but can regress performance. Replace such
    calls with the LLVM intrinsic form (`"llvm.intr.exp2"`) so `llc` can lower
    it directly to `ex2.approx.f32` in PTX.

    MLIR 14's `convert-arith-to-llvm` does not lower `arith.{maxf,minf,maxsi,minsi,maxui,minui}`.
    Rewrite these ops to the corresponding LLVM intrinsics so `mlir-translate`
    can produce LLVM IR.

    MLIR 14's `convert-math-to-llvm` leaves `math.erf` behind; rewrite it to a
    libm call (`erff`) and insert a declaration if needed.

    This is a text-level rewrite because the repository intentionally avoids
    MLIR Python bindings.
    """
    text = str(module.module_text or "")
    touched = False
    touched_exp2 = False
    touched_minmax = False
    touched_erf = False
    rewritten = text
    if "@__nv_exp2f" in rewritten:
        next_text = _CALL_EXP2_RE.sub("\"llvm.intr.exp2\"", rewritten)
        if next_text != rewritten:
            rewritten = next_text
            touched = True
            touched_exp2 = True
    if "@__nv_expf" in rewritten:
        next_text = _CALL_EXP_RE.sub("\"llvm.intr.exp\"", rewritten)
        if next_text != rewritten:
            rewritten = next_text
            touched = True
            touched_exp2 = True

    def _rewrite_binop(pat: re.Pattern[str], intr_name: str, payload: str) -> str:
        return pat.sub(
            rf'\g<indent>\g<dst> = "{intr_name}"(\g<a>, \g<b>) : (\g<ty>, \g<ty>) -> \g<ty>',
            payload,
        )

    for pat, intr in (
        (_ARITH_MAXF_RE, "llvm.intr.maximum"),
        (_ARITH_MINF_RE, "llvm.intr.minimum"),
        (_ARITH_MAXSI_RE, "llvm.intr.smax"),
        (_ARITH_MINSI_RE, "llvm.intr.smin"),
        (_ARITH_MAXUI_RE, "llvm.intr.umax"),
        (_ARITH_MINUI_RE, "llvm.intr.umin"),
    ):
        next_text = _rewrite_binop(pat, intr, rewritten)
        if next_text != rewritten:
            rewritten = next_text
            touched = True
            touched_minmax = True

    if "math.erf" in rewritten:
        next_text = _MATH_ERF_RE.sub(r"\g<indent>\g<dst> = llvm.call @erff(\g<a>) : (f32) -> f32", rewritten)
        if next_text != rewritten:
            rewritten = next_text
            touched = True
            touched_erf = True
            if not _ERFF_DECL_RE.search(rewritten):
                m = _FIRST_LLVM_FUNC_RE.search(rewritten)
                if m:
                    indent = m.group("indent") or ""
                    decl = f"{indent}llvm.func @erff(f32) -> f32 attributes {{sym_visibility = \"private\"}}"
                    rewritten = rewritten[: m.start()] + decl + "\n" + rewritten[m.start() :]

    if not touched or rewritten == text:
        return module

    out = IntentMLIRModule(
        module_text=rewritten,
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    if touched_exp2:
        out.meta["rewrite_cuda_nv_exp2f_calls"] = True
    if touched_minmax:
        out.meta["rewrite_cuda_llvm_intr_minmax"] = True
    if touched_erf:
        out.meta["rewrite_cuda_math_erf_to_libm_call"] = True
    return out
