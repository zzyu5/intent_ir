"""
Static validation of IntentIR against a SemanticCertificate (Task5 Stage A).

This module is frontend-agnostic: it operates on the shared CertificateV2
(`frontends/common/certificate_v2.py`) and does not depend on TTIR/Triton.

Historically this lived under `frontends/triton/static_validate.py`; it was
moved to `frontends/common/` so TileLang and future frontends can reuse the
same Stage-A checks without importing Triton-specific modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from intent_ir.ir import IntentFunction
from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.evidence import CanonicalEvidence


Status = Literal["PASS", "FAIL", "UNKNOWN"]


@dataclass(frozen=True)
class StaticObligation:
    id: str
    status: Status
    detail: Optional[str] = None


@dataclass
class StaticValidationResult:
    ok: bool
    obligations: List[StaticObligation]
    reasons: List[str]


def _is_legacy_cert(cert: object) -> bool:
    return hasattr(cert, "kernel_kind") and hasattr(cert, "obligations")


def _kernel_kind_from_cert(cert: object) -> str:
    if _is_legacy_cert(cert):
        try:
            return str(getattr(cert, "kernel_kind"))
        except Exception:
            return "unknown"
    anchors = (cert.semantic_facts or {}).get("anchors") or {}
    if isinstance(anchors, dict):
        # NOTE: some frontends historically stuffed a *frontend* identifier into
        # `kernel_kind_hint` (e.g., "cuda_ptx", "tilelang_tileop"). For static
        # validation we want a semantic kind, otherwise anchor checks never run.
        k = anchors.get("kernel_kind_hint") or anchors.get("kernel_kind")
        if isinstance(k, str) and k.strip() in {"matmul", "reduce", "attention", "copy"}:
            return str(k).strip()
        # derive (best-effort)
        if anchors.get("has_atomic"):
            return "unknown"
        if anchors.get("has_dot") and anchors.get("has_reduce"):
            return "attention"
        if anchors.get("has_dot"):
            return "matmul"
        if anchors.get("has_reduce"):
            return "reduce"
    return "unknown"


def _iter_accesses_from_cert(cert: SemanticCertificateV2) -> list[dict[str, Any]]:
    """
    Return a list of access dicts from cert_v2, supporting both in-memory objects
    (CanonicalEvidence) and JSON-like dict payloads.
    """
    ce = (cert.semantic_facts or {}).get("canonical_evidence")
    if isinstance(ce, CanonicalEvidence):
        return [a.to_json_dict() for a in ce.accesses]
    if isinstance(ce, dict):
        acc = ce.get("accesses")
        if isinstance(acc, list):
            return [a for a in acc if isinstance(a, dict)]
    return []


def _needs_mask_from_cert(cert: object) -> bool:
    if _is_legacy_cert(cert):
        try:
            return bool(getattr(cert, "needs_mask", False))
        except Exception:
            return False
    for a in _iter_accesses_from_cert(cert):  # type: ignore[arg-type]
        pred = a.get("predicate")
        if isinstance(pred, dict) and pred.get("clauses"):
            return True
    return False


def _store_group_count(cert: object) -> int | None:
    if _is_legacy_cert(cert):
        try:
            groups = getattr(cert, "pointer_groups", None)
            if not isinstance(groups, dict) or not groups:
                return None
            return sum(1 for g in groups.values() if bool(getattr(g, "stores", False)))
        except Exception:
            return None
    store_tensors: set[str] = set()
    for a in _iter_accesses_from_cert(cert):  # type: ignore[arg-type]
        if a.get("kind") == "store":
            t = a.get("tensor")
            if isinstance(t, str) and t:
                store_tensors.add(t)
    return len(store_tensors) if store_tensors else None


def _contract_level_from_cert(cert: object) -> str | None:
    if _is_legacy_cert(cert):
        try:
            c = getattr(cert, "contract", None)
            if c is None:
                return None
            lvl = getattr(c, "level", None)
            return str(lvl) if lvl is not None else None
        except Exception:
            return None
    c = (cert.semantic_facts or {}).get("contract")
    if isinstance(c, dict):
        lvl = c.get("level")
        return str(lvl) if isinstance(lvl, str) else None
    # Pipelines may attach a contract summary to cert.meta (to avoid perturbing
    # CertificateV2.semantic_facts golden locks). Prefer this as a fallback.
    meta = getattr(cert, "meta", None)
    if isinstance(meta, dict):
        c2 = meta.get("contract")
        if isinstance(c2, dict):
            lvl = c2.get("level")
            return str(lvl) if isinstance(lvl, str) else None
    return None


def _contract_reasons_from_cert(cert: object) -> list[str]:
    if _is_legacy_cert(cert):
        try:
            c = getattr(cert, "contract", None)
            rs = getattr(c, "reasons", None) if c is not None else None
            if isinstance(rs, list):
                return [str(x) for x in rs if isinstance(x, str) and x.strip()]
        except Exception:
            return []
        return []
    # Prefer meta["contract"] (pipeline-attached) but allow semantic_facts["contract"].
    for container in (getattr(cert, "meta", None), getattr(cert, "semantic_facts", None)):
        if not isinstance(container, dict):
            continue
        c = container.get("contract")
        if isinstance(c, dict):
            rs = c.get("reasons")
            if isinstance(rs, list):
                return [str(x) for x in rs if isinstance(x, str) and x.strip()]
    return []


def _seed_obligations(cert: object) -> list[StaticObligation]:
    if _is_legacy_cert(cert):
        out0: list[StaticObligation] = []
        try:
            for o in list(getattr(cert, "obligations", []) or []):
                out0.append(
                    StaticObligation(
                        id=str(getattr(o, "id", "O?")),
                        status=str(getattr(o, "status", "UNKNOWN")),  # type: ignore[arg-type]
                        detail=(None if getattr(o, "detail", None) is None else str(getattr(o, "detail"))),
                    )
                )
        except Exception:
            return []
        return out0
    out: list[StaticObligation] = []
    obs = (getattr(cert, "semantic_facts", None) or {}).get("obligations") if hasattr(cert, "semantic_facts") else None
    if isinstance(obs, list):
        for o in obs:
            if isinstance(o, dict):
                out.append(
                    StaticObligation(
                        id=str(o.get("id", "O?")),
                        status=str(o.get("status", "UNKNOWN")),  # type: ignore[arg-type]
                        detail=(None if o.get("reason") is None else str(o.get("reason"))),
                    )
                )
    return out


def static_validate(intent: IntentFunction, cert: object) -> StaticValidationResult:
    reasons: List[str] = []
    obligations = _seed_obligations(cert)
    contract_level = _contract_level_from_cert(cert)
    if contract_level == "OUT_OF_SCOPE":
        obligations.append(StaticObligation(id="SV_contract_out_of_scope", status="FAIL", detail="contract OUT_OF_SCOPE"))
        reasons.append("contract OUT_OF_SCOPE")
        for r in _contract_reasons_from_cert(cert):
            if r not in reasons:
                reasons.append(r)
    # Outputs must be produced by ops (not just declared in tensors).
    produced = {op.output: op for op in intent.ops}
    for out in intent.outputs:
        if out not in produced:
            obligations.append(StaticObligation(id=f"SV_output_not_produced_{out}", status="FAIL", detail="no producing op"))
            reasons.append(f"output {out} not produced by any op")
        else:
            op = produced[out]
            if op.op == "const":
                val = (op.attrs or {}).get("value")
                if isinstance(val, str) and val.startswith("placeholder:"):
                    obligations.append(
                        StaticObligation(id=f"SV_output_placeholder_{out}", status="FAIL", detail=str(val))
                    )
                    reasons.append(f"output {out} is placeholder const; must be produced by real ops")
    # Anchor check: intent ops should contain matmul/reduce/softmax depending on kernel_kind
    ops = [op.op for op in intent.ops]
    kernel_kind = _kernel_kind_from_cert(cert)
    if kernel_kind == "matmul":
        if "matmul" not in ops:
            obligations.append(StaticObligation(id="SV_matmul_missing", status="FAIL", detail="matmul op absent"))
            reasons.append("matmul op missing")
        else:
            obligations.append(StaticObligation(id="SV_matmul_present", status="PASS", detail=None))
    if kernel_kind in {"reduce", "attention"}:
        # TTIR uses reduce ops for many patterns (e.g., softmax). In IntentIR, a single
        # `softmax` op is allowed to represent the internal reduce_max/reduce_sum.
        has_reduce = any(op.op.startswith("reduce") for op in intent.ops)
        if "softmax" in ops:
            has_reduce = True
        if not has_reduce:
            obligations.append(StaticObligation(id="SV_reduce_missing", status="FAIL", detail="reduce op absent"))
            reasons.append("reduce op missing")
        else:
            obligations.append(StaticObligation(id="SV_reduce_present", status="PASS", detail=None))
    if kernel_kind == "attention":
        if "softmax" not in ops:
            obligations.append(StaticObligation(id="SV_softmax_missing", status="FAIL", detail="softmax op missing"))
            reasons.append("softmax op missing")
        else:
            obligations.append(StaticObligation(id="SV_softmax_present", status="PASS", detail=None))
    for out in intent.outputs:
        if out not in intent.tensors:
            obligations.append(StaticObligation(id=f"SV_output_tensor_{out}", status="FAIL", detail="output not declared"))
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
            StaticObligation(
                id="SV_unknown_shape_symbols",
                status="FAIL",
                detail=f"unknown symbols in shape attrs: {sorted(unknown)}",
            )
        )
        reasons.append(f"unknown symbols used in reshape/broadcast/iota shapes: {sorted(unknown)}")

    # Structural witness: TTIR distinct store pointer groups should not exceed
    # the number of declared outputs (helps catch missing Mean/Rstd etc).
    num_store_groups = _store_group_count(cert)
    if num_store_groups is not None:
        if num_store_groups > len(intent.outputs):
            obligations.append(
                StaticObligation(
                    id="SV_outputs_lt_store_groups",
                    status="FAIL",
                    detail=f"store_groups={num_store_groups} outputs={len(intent.outputs)}",
                )
            )
            reasons.append(f"intent outputs ({len(intent.outputs)}) fewer than TTIR store groups ({num_store_groups})")
        else:
            obligations.append(
                StaticObligation(
                    id="SV_outputs_ge_store_groups",
                    status="PASS",
                    detail=f"store_groups={num_store_groups} outputs={len(intent.outputs)}",
                )
            )
    if _needs_mask_from_cert(cert) and not any(op.op.startswith("reduce") for op in intent.ops):
        obligations.append(StaticObligation(id="SV_mask_without_reduce", status="UNKNOWN", detail="needs_mask but no reduce op"))
    # Make `reasons` actionable for LLM repair loops: include FAIL obligation details
    # even when they originate from frontend-provided obligations.
    seen: set[str] = set(reasons)
    for ob in obligations:
        if ob.status != "FAIL":
            continue
        msg = None
        if ob.detail:
            msg = f"{ob.id}: {ob.detail}"
        else:
            msg = f"{ob.id}: FAIL"
        if msg not in seen:
            reasons.append(msg)
            seen.add(msg)

    ok = all(ob.status != "FAIL" for ob in obligations)
    return StaticValidationResult(ok=ok, obligations=obligations, reasons=reasons)


__all__ = ["StaticObligation", "StaticValidationResult", "static_validate"]
