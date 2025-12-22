"""
Semantic certificate for TTIR, derived from Task4 facts (v1.2 sketch).
This is intentionally lightweight: it records kernel_kind anchors, masks/atomic flags,
and schedule/tile hints. It is not a full proof, but provides structured obligations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Literal

from .facts import TTIRFacts, extract_facts
from .contract import evaluate_contract, ContractReport
from .ttir_witness import MaskWitness, PointerGroup, build_mask_witnesses, build_pointer_groups
from .ttir_witness import parse_function_args, parse_ssa_defs
from .affine_expr import build_aliases, expr_to_str
from .smt_check import check_mask_constraints_inbounds


@dataclass
class Obligation:
    id: str
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    detail: str | None = None


@dataclass
class SemanticCertificate:
    kernel_kind: Literal["matmul", "reduce", "softmax", "attention", "unknown"]
    tile_hints: List[int] = field(default_factory=list)
    needs_mask: bool = False
    contract: ContractReport | None = None
    obligations: List[Obligation] = field(default_factory=list)
    pointer_groups: Dict[str, PointerGroup] = field(default_factory=dict)
    mask_witnesses: Dict[str, MaskWitness] = field(default_factory=dict)
    mask_constraints: Dict[str, List[str]] = field(default_factory=dict)
    mask_formulas: Dict[str, str] = field(default_factory=dict)
    mask_accesses: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    index_maps: Dict[str, List[str]] = field(default_factory=dict)
    index_symbols: Dict[str, Any] = field(default_factory=dict)
    raw_facts: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "kernel_kind": self.kernel_kind,
            "tile_hints": list(self.tile_hints),
            "needs_mask": bool(self.needs_mask),
            "contract": asdict(self.contract) if self.contract is not None else None,
            "obligations": [asdict(o) for o in self.obligations],
            "pointer_groups": {k: asdict(v) for k, v in self.pointer_groups.items()},
            "mask_witnesses": {k: asdict(v) for k, v in self.mask_witnesses.items()},
            "mask_constraints": {k: list(v) for k, v in self.mask_constraints.items()},
            "mask_formulas": dict(self.mask_formulas),
            "mask_accesses": {k: list(v) for k, v in self.mask_accesses.items()},
            "index_maps": {k: list(v) for k, v in self.index_maps.items()},
            "index_symbols": dict(self.index_symbols),
            "raw_facts": dict(self.raw_facts),
        }


def _infer_kernel_kind(facts: TTIRFacts) -> Literal["matmul", "reduce", "softmax", "attention", "unknown"]:
    if facts.has_atomic:
        return "unknown"
    if facts.has_dot and facts.has_reduce:
        return "attention"  # heuristic
    if facts.has_dot:
        return "matmul"
    if facts.has_reduce:
        return "reduce"
    return "unknown"


BLOCK_RE = re.compile(r"BLOCK_[MNK]\s*=\s*(\d+)")
MAKE_RANGE_END_RE = re.compile(r"tt\\.make_range\\b[^\\n]*?end\\s*=\\s*(\\d+)")
ARITH_CONST_I_RE = re.compile(r"arith\\.constant\\s+(\\d+)\\s*:\\s*i(?:32|64)")
TENSOR_SHAPE_RE = re.compile(r"tensor<([0-9]+(?:x[0-9]+)+)x")


def _extract_tile_hints(ttir: str, facts: TTIRFacts) -> List[int]:
    """
    Best-effort extraction of "tile-ish" constants from TTIR.

    Triton TTIR often contains `tt.make_range ... end = <BLOCK>` or `arith.constant <BLOCK> : i32`
    values that correspond to block sizes. We only use these as hints for case generation.
    """
    hints: set[int] = set()

    for m in BLOCK_RE.finditer(ttir):
        try:
            hints.add(int(m.group(1)))
        except Exception:
            pass

    for m in MAKE_RANGE_END_RE.finditer(ttir):
        try:
            hints.add(int(m.group(1)))
        except Exception:
            pass

    for m in ARITH_CONST_I_RE.finditer(ttir):
        try:
            hints.add(int(m.group(1)))
        except Exception:
            pass

    # Also treat tensor block shapes as tile hints (common in TTIR: tensor<8x16x!tt.ptr<...>>).
    for m in TENSOR_SHAPE_RE.finditer(ttir):
        dims = m.group(1).split("x")
        for d in dims:
            try:
                hints.add(int(d))
            except Exception:
                pass

    # Filter to plausible tile sizes.
    filtered = [v for v in hints if 2 <= v <= 2048]
    # Prefer powers of two, but keep a couple of common non-pow2 sentinels.
    preferred = [v for v in filtered if (v & (v - 1) == 0)]
    out = preferred if preferred else filtered
    return sorted(set(out))


def build_certificate(ttir: str, facts: TTIRFacts | None = None) -> SemanticCertificate:
    facts = facts or extract_facts(ttir)
    contract = evaluate_contract(facts)
    kernel_kind = _infer_kernel_kind(facts)
    tile_hints = _extract_tile_hints(ttir, facts)
    obligations: List[Obligation] = []
    # O1: anchor present
    anchor_ok = (kernel_kind == "matmul" and facts.has_dot) or (kernel_kind in {"reduce", "attention"} and facts.has_reduce) or (
        kernel_kind == "unknown"
    )
    obligations.append(Obligation(id="O1_anchor_present", status="PASS" if anchor_ok else "FAIL", detail=str(facts.op_counts)))
    # O2: no atomic
    obligations.append(Obligation(id="O2_no_atomic", status="FAIL" if facts.has_atomic else "PASS", detail=None))
    # O3: mask if needed
    mask_present = any(site.has_mask for site in facts.load_sites + facts.store_sites)
    obligations.append(Obligation(id="O3_mask_present", status="PASS" if mask_present else "UNKNOWN", detail=None))
    pointer_groups = build_pointer_groups(ttir, facts)
    mask_witnesses = build_mask_witnesses(ttir, facts)
    index_maps = _extract_index_maps(ttir, facts)
    mask_constraints = _extract_mask_constraints(ttir, mask_witnesses)
    mask_formulas = {k: v.formula for k, v in mask_witnesses.items() if getattr(v, "formula", None)}
    mask_accesses = _collect_mask_accesses(facts)

    # Canonicalize pid/range symbols across witnesses to make reports comparable.
    symbols: Dict[str, Any] = {}
    index_maps, symbols = _canonicalize_index_maps(index_maps, symbols)
    mask_constraints, symbols = _canonicalize_mask_constraints(mask_constraints, symbols)
    # O4: pointer grouping best-effort witness
    obligations.append(
        Obligation(
            id="O4_pointer_grouping",
            status="PASS" if pointer_groups else "UNKNOWN",
            detail=f"groups={len(pointer_groups)}",
        )
    )
    # O5: mask witness coverage (only checks masked access sites)
    masked_sites = [s for s in (facts.load_sites + facts.store_sites) if s.has_mask and getattr(s, "mask", None)]
    if not masked_sites:
        obligations.append(Obligation(id="O5_mask_witness", status="UNKNOWN", detail="no masked sites"))
    else:
        ok = True
        for s in masked_sites:
            m = getattr(s, "mask", None)
            if not m or m not in mask_witnesses or not mask_witnesses[m].cmps:
                ok = False
                break
        obligations.append(Obligation(id="O5_mask_witness", status="PASS" if ok else "UNKNOWN", detail=f"masks={len(mask_witnesses)}"))
    # O6: index maps extracted
    obligations.append(
        Obligation(
            id="O6_index_maps",
            status="PASS" if any(index_maps.values()) else "UNKNOWN",
            detail=f"bases={len(index_maps)}",
        )
    )
    # O7: minimal formal-ish check: prove masked indices are non-negative (so cmp gives 0<=idx<Dim).
    smt = check_mask_constraints_inbounds(mask_constraints, index_symbols=symbols)
    obligations.append(
        Obligation(
            id="O7_mask_inbounds_nonneg",
            status=smt.status,
            detail=f"passed={smt.passed} unknown={smt.unknown} total={smt.total}",
        )
    )
    return SemanticCertificate(
        kernel_kind=kernel_kind,
        tile_hints=tile_hints,
        needs_mask=mask_present,
        contract=contract,
        obligations=obligations,
        pointer_groups=pointer_groups,
        mask_witnesses=mask_witnesses,
        mask_constraints=mask_constraints,
        mask_formulas=mask_formulas,
        mask_accesses=mask_accesses,
        index_maps=index_maps,
        index_symbols=symbols,
        raw_facts={"op_counts": facts.op_counts, "has_dot": facts.has_dot, "has_reduce": facts.has_reduce},
    )


def _extract_index_maps(ttir: str, facts: TTIRFacts) -> Dict[str, List[str]]:
    """
    Best-effort index expression witness for pointer groups.

    Output format: {base_arg: ["load@L37: base + (offset_expr)", ...]}.
    """
    arg_types = parse_function_args(ttir)
    func_args = set(arg_types.keys())
    defs = parse_ssa_defs(ttir)
    aliases = build_aliases(defs)

    maps: Dict[str, List[str]] = {}
    for site in list(facts.load_sites) + list(facts.store_sites):
        ptr = getattr(site, "ptr", None)
        if not ptr:
            continue
        base, offsets = _trace_addptr_chain(ptr, defs, func_args)
        if base is None:
            continue
        if not offsets:
            offset_s = "0"
        else:
            parts = [expr_to_str(off, defs, aliases=aliases) for off in offsets]
            offset_s = " + ".join(parts) if len(parts) > 1 else parts[0]
        entry = f"{site.kind}@L{site.line_no}: {base} + ({offset_s})"
        maps.setdefault(base, [])
        if entry not in maps[base]:
            maps[base].append(entry)
    # Keep output compact: at most a few examples per base.
    for k in list(maps.keys()):
        maps[k] = maps[k][:6]
    return maps


_RANGE_RE = re.compile(r"\brange\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def _canonicalize_text(text: str, symbols: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    Canonicalize pid/range tokens inside a witness string.

    - pid_x/y/z -> pid0/pid1/pid2
    - range(a,b) -> rK with symbol table rK:[a,b)
    """
    # pid canonicalization
    pid_map = symbols.setdefault("pids", {"pid_x": "pid0", "pid_y": "pid1", "pid_z": "pid2"})
    for src, dst in pid_map.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    ranges = symbols.setdefault("ranges", {})
    # Stable ordering: assign by first appearance.
    def repl(m: re.Match) -> str:
        a = int(m.group(1))
        b = int(m.group(2))
        key = f"range({a},{b})"
        # find existing
        for name, spec in ranges.items():
            if spec.get("start") == a and spec.get("end") == b:
                return name
        new_name = f"r{len(ranges)}"
        ranges[new_name] = {"start": a, "end": b}
        return new_name

    text = _RANGE_RE.sub(repl, text)
    return text, symbols


def _canonicalize_index_maps(index_maps: Dict[str, List[str]], symbols: Dict[str, Any]) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
    out: Dict[str, List[str]] = {}
    for base, entries in index_maps.items():
        new_entries = []
        for e in entries:
            ee, symbols = _canonicalize_text(e, symbols)
            new_entries.append(ee)
        out[base] = new_entries
    return out, symbols


def _canonicalize_mask_constraints(mask_constraints: Dict[str, List[str]], symbols: Dict[str, Any]) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
    out: Dict[str, List[str]] = {}
    for mask, cs in mask_constraints.items():
        new_cs = []
        for c in cs:
            cc, symbols = _canonicalize_text(c, symbols)
            new_cs.append(cc)
        out[mask] = new_cs
    return out, symbols


def _extract_mask_constraints(ttir: str, mask_witnesses: Dict[str, MaskWitness]) -> Dict[str, List[str]]:
    """
    Format mask witness cmp nodes into human-readable constraints.
    """
    defs = parse_ssa_defs(ttir)
    aliases = build_aliases(defs)
    constraints: Dict[str, List[str]] = {}

    kind_to_op = {
        "slt": "<",
        "ult": "<",
        "sle": "<=",
        "ule": "<=",
        "sgt": ">",
        "ugt": ">",
        "sge": ">=",
        "uge": ">=",
        "eq": "==",
        "ne": "!=",
    }

    for mask, w in mask_witnesses.items():
        cs: List[str] = []
        for c in w.cmps:
            if not c.lhs or not c.rhs:
                continue
            lhs = expr_to_str(c.lhs, defs, aliases=aliases)
            rhs = expr_to_str(c.rhs, defs, aliases=aliases)
            op = kind_to_op.get(c.kind or "", c.kind or "?")
            cs.append(f"{lhs} {op} {rhs}")
        constraints[mask] = cs
    return constraints


def _collect_mask_accesses(facts: TTIRFacts) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for site in list(facts.load_sites) + list(facts.store_sites):
        if not site.has_mask or not getattr(site, "mask", None):
            continue
        m = getattr(site, "mask")
        if not isinstance(m, str):
            continue
        out.setdefault(m, []).append(
            {
                "kind": site.kind,
                "line": site.line_no,
                "ptr": getattr(site, "ptr", None),
                "tensor_hint": site.tensor_hint,
            }
        )
    return out


def _trace_addptr_chain(
    ptr: str,
    defs: Dict[str, Any],
    func_args: set[str],
    *,
    max_depth: int = 64,
) -> tuple[str | None, list[str]]:
    """
    Trace a pointer value through tt.addptr / shape ops back to a function arg pointer.
    Returns (base_arg, [offset_ssa...]) where offsets are in "nearest-first" order.
    """
    cur = ptr
    offsets: list[str] = []
    for _ in range(max_depth):
        if cur in func_args:
            return cur, list(reversed(offsets))
        d = defs.get(cur)
        if d is None:
            return None, []
        if getattr(d, "op", None) == "scf.iter_arg":
            cur = d.operands[0]
            continue
        op = getattr(d, "op", "")
        ops = getattr(d, "operands", ())
        if op in {"tt.bitcast", "tt.broadcast", "tt.splat", "tt.reshape", "tt.expand_dims"} and ops:
            cur = ops[0]
            continue
        if op == "tt.addptr" and len(ops) >= 2:
            offsets.append(ops[1])
            cur = ops[0]
            continue
        return None, []
    return None, []


__all__ = ["SemanticCertificate", "Obligation", "build_certificate"]
