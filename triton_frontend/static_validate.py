"""
Static validation of Intent IR against a SemanticCertificate (Task5 Stage A).
This is deliberately lightweight: it checks presence of expected anchors and basic shape/axis roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from intent_ir.ir_types import IntentFunction
from .certificate import SemanticCertificate, Obligation


@dataclass
class StaticValidationResult:
    ok: bool
    obligations: List[Obligation]
    reasons: List[str]


def static_validate(intent: IntentFunction, cert: SemanticCertificate) -> StaticValidationResult:
    reasons: List[str] = []
    obligations = list(cert.obligations)
    if cert.contract and cert.contract.level == "OUT_OF_SCOPE":
        obligations.append(Obligation(id="SV_contract_out_of_scope", status="FAIL", detail="contract OUT_OF_SCOPE"))
        reasons.append("contract OUT_OF_SCOPE")
    # Outputs must be produced by ops (not just declared in tensors).
    produced = {op.output: op for op in intent.ops}
    for out in intent.outputs:
        if out not in produced:
            obligations.append(Obligation(id=f"SV_output_not_produced_{out}", status="FAIL", detail="no producing op"))
            reasons.append(f"output {out} not produced by any op")
        else:
            op = produced[out]
            if op.op == "const":
                val = (op.attrs or {}).get("value")
                if isinstance(val, str) and val.startswith("placeholder:"):
                    obligations.append(
                        Obligation(id=f"SV_output_placeholder_{out}", status="FAIL", detail=str(val))
                    )
                    reasons.append(f"output {out} is placeholder const; must be produced by real ops")
    # Anchor check: intent ops should contain matmul/reduce/softmax depending on kernel_kind
    ops = [op.op for op in intent.ops]
    if cert.kernel_kind == "matmul":
        if "matmul" not in ops:
            obligations.append(Obligation(id="SV_matmul_missing", status="FAIL", detail="matmul op absent"))
            reasons.append("matmul op missing")
        else:
            obligations.append(Obligation(id="SV_matmul_present", status="PASS", detail=None))
    if cert.kernel_kind in {"reduce", "attention"}:
        # TTIR uses reduce ops for many patterns (e.g., softmax). In IntentIR, a single
        # `softmax` op is allowed to represent the internal reduce_max/reduce_sum.
        has_reduce = any(op.op.startswith("reduce") for op in intent.ops)
        if cert.kernel_kind == "attention" and "softmax" in ops:
            has_reduce = True
        if not has_reduce:
            obligations.append(Obligation(id="SV_reduce_missing", status="FAIL", detail="reduce op absent"))
            reasons.append("reduce op missing")
        else:
            obligations.append(Obligation(id="SV_reduce_present", status="PASS", detail=None))
    if cert.kernel_kind == "attention":
        if "softmax" not in ops:
            obligations.append(Obligation(id="SV_softmax_missing", status="FAIL", detail="softmax op missing"))
            reasons.append("softmax op missing")
        else:
            obligations.append(Obligation(id="SV_softmax_present", status="PASS", detail=None))
    for out in intent.outputs:
        if out not in intent.tensors:
            obligations.append(Obligation(id=f"SV_output_tensor_{out}", status="FAIL", detail="output not declared"))
            reasons.append(f"output {out} not declared in tensors")

    # Shape symbol sanity: do not allow invented/unbound shape symbols in reshape/broadcast/iota shapes.
    allowed_syms = set(intent.parallel_axes or [])
    # Symbols from tensor shapes
    for t in intent.tensors.values():
        for d in t.shape:
            if getattr(d, "kind", None) == "sym":
                allowed_syms.add(str(d.value))
    # Also allow scalar tensor names (commonly used as derived symbols like group_size/C/HW).
    for name, t in intent.tensors.items():
        if len(t.shape) == 0:
            allowed_syms.add(name)

    unknown: set[str] = set()
    for op in intent.ops:
        attrs = op.attrs or {}
        if op.op in {"reshape", "broadcast_in_dim", "iota"}:
            shape_list = None
            if op.op == "broadcast_in_dim":
                shape_list = attrs.get("out_shape")
            else:
                shape_list = attrs.get("shape")
            if isinstance(shape_list, list):
                for dim in shape_list:
                    if isinstance(dim, str) and dim and (not dim.isdigit()) and (dim not in allowed_syms):
                        unknown.add(dim)
    if unknown:
        obligations.append(
            Obligation(
                id="SV_unknown_shape_symbols",
                status="FAIL",
                detail=f"unknown symbols in shape attrs: {sorted(unknown)}",
            )
        )
        reasons.append(f"unknown symbols used in reshape/broadcast/iota shapes: {sorted(unknown)}")

    # Structural witness: TTIR distinct store pointer groups should not exceed
    # the number of declared outputs (helps catch missing Mean/Rstd etc).
    if cert.pointer_groups:
        num_store_groups = sum(1 for g in cert.pointer_groups.values() if g.stores)
        if num_store_groups > len(intent.outputs):
            obligations.append(
                Obligation(
                    id="SV_outputs_lt_store_groups",
                    status="FAIL",
                    detail=f"store_groups={num_store_groups} outputs={len(intent.outputs)}",
                )
            )
            reasons.append(f"intent outputs ({len(intent.outputs)}) fewer than TTIR store groups ({num_store_groups})")
        else:
            obligations.append(
                Obligation(
                    id="SV_outputs_ge_store_groups",
                    status="PASS",
                    detail=f"store_groups={num_store_groups} outputs={len(intent.outputs)}",
                )
            )
    if cert.needs_mask and not any(op.op.startswith("reduce") for op in intent.ops):
        obligations.append(Obligation(id="SV_mask_without_reduce", status="UNKNOWN", detail="needs_mask but no reduce op"))
    ok = all(ob.status != "FAIL" for ob in obligations)
    return StaticValidationResult(ok=ok, obligations=obligations, reasons=reasons)


__all__ = ["StaticValidationResult", "static_validate"]
