"""
Task4 v1.2 (partial): TTIR witness extraction.

This upgrades the "facts/booleans" style TTIR analysis into a more explainable
certificate payload:
- pointer grouping: which function-arg pointer a load/store ultimately aliases
- mask witness: which cmp operations contribute to a given mask SSA value

This is intentionally best-effort and text-based (no full MLIR parser).
It targets common Triton TTIR patterns (tt.addptr/splat/broadcast/bitcast).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .facts import AccessSite, TTIRFacts


@dataclass(frozen=True)
class SSAValueDef:
    name: str
    line_no: int
    op: str
    operands: Tuple[str, ...]
    line: str


@dataclass
class PointerGroup:
    base_arg: str
    element_type: str | None = None
    loads: List[int] = field(default_factory=list)
    stores: List[int] = field(default_factory=list)
    masked_loads: List[int] = field(default_factory=list)
    masked_stores: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class MaskCmp:
    line_no: int
    kind: str | None
    lhs: str | None
    rhs: str | None
    line: str


@dataclass
class MaskWitness:
    mask: str
    cmps: List[MaskCmp] = field(default_factory=list)
    visited: List[str] = field(default_factory=list)
    formula: str | None = None


FUNC_RE = re.compile(r"tt\.func\b[^(]*\((.*)\)\s+attributes")
# Triton TTIR argument names may be either positional (%arg0) or symbolic (%x_ptr).
# Accept both so downstream pointer/mask witnesses stay robust across TTIR versions.
ARG_RE = re.compile(r"(%[A-Za-z0-9_]+)\s*:\s*([^,)]+)")
DEF_RE = re.compile(r"^\s*(%[A-Za-z0-9_]+)(?::\d+)?\s*=\s*(.+?)\s*$")


def parse_function_args(ttir: str) -> Dict[str, str]:
    """
    Return {arg_name: type_str} for function arguments.
    """
    for line in ttir.splitlines():
        if "tt.func" not in line:
            continue
        m = FUNC_RE.search(line)
        if not m:
            # some TTIR may wrap args across multiple lines; fall back to regex scan
            continue
        arg_blob = m.group(1)
        out: Dict[str, str] = {}
        for am in ARG_RE.finditer(arg_blob):
            out[am.group(1)] = am.group(2).strip()
        return out
    # fallback: pick any %argN occurrences in the first ~20 lines
    out: Dict[str, str] = {}
    for line in ttir.splitlines()[:25]:
        for am in ARG_RE.finditer(line):
            out[am.group(1)] = am.group(2).strip()
    return out


def parse_ssa_defs(ttir: str) -> Dict[str, SSAValueDef]:
    """
    Parse simple SSA definitions of the form:
      %29 = tt.addptr %17, %28 : ...
      %19 = "tt.reduce"(%18) <{...}> ({ ...
      %63:5 = scf.for ...
    """
    defs: Dict[str, SSAValueDef] = {}
    lines = ttir.splitlines()
    for idx, line in enumerate(lines, start=1):
        m = DEF_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        # TTIR/MLIR may reuse SSA names in nested regions (e.g., reduce bodies).
        # For our witness extraction we want the "outermost" binding, so keep the
        # first definition and ignore later duplicates.
        if name in defs:
            continue
        rhs = m.group(2).strip()
        op_token = rhs.split(None, 1)[0]
        op = _normalize_op_token(op_token)
        operands = tuple(_extract_all_ssa_operands(rhs))
        defs[name] = SSAValueDef(name=name, line_no=idx, op=op, operands=operands, line=line.strip())
        if op == "scf.for":
            # Record iter_args mapping: %arg28 -> %54 etc (block args are not SSA defs).
            for arg_name, init in _extract_scf_iter_args(rhs):
                if arg_name not in defs:
                    defs[arg_name] = SSAValueDef(name=arg_name, line_no=idx, op="scf.iter_arg", operands=(init,), line=line.strip())
    return defs


def build_pointer_groups(ttir: str, facts: TTIRFacts) -> Dict[str, PointerGroup]:
    arg_types = parse_function_args(ttir)
    func_args: Set[str] = set(arg_types.keys())
    defs = parse_ssa_defs(ttir)
    groups: Dict[str, PointerGroup] = {}
    for site in list(facts.load_sites) + list(facts.store_sites):
        ptr = getattr(site, "ptr", None)
        if not ptr:
            continue
        base = trace_base_pointer(ptr, defs, func_args)
        if base is None:
            continue
        g = groups.get(base)
        if g is None:
            g = PointerGroup(base_arg=base, element_type=_infer_ptr_element_type(site.line))
            groups[base] = g
        if site.kind == "load":
            g.loads.append(site.line_no)
            if site.has_mask:
                g.masked_loads.append(site.line_no)
        else:
            g.stores.append(site.line_no)
            if site.has_mask:
                g.masked_stores.append(site.line_no)
    return groups


def build_mask_witnesses(ttir: str, facts: TTIRFacts) -> Dict[str, MaskWitness]:
    arg_types = parse_function_args(ttir)
    func_args: Set[str] = set(arg_types.keys())
    defs = parse_ssa_defs(ttir)
    out: Dict[str, MaskWitness] = {}
    # Collect masks actually used by memory ops.
    masks: Set[str] = set()
    for site in list(facts.load_sites) + list(facts.store_sites):
        m = getattr(site, "mask", None)
        if site.has_mask and isinstance(m, str) and m.startswith("%"):
            masks.add(m)
    for m in sorted(masks):
        out[m] = extract_mask_witness(m, defs, func_args)
    return out


PTR_PRESERVING_OPS = {
    "tt.addptr",
    "tt.bitcast",
    "tt.broadcast",
    "tt.splat",
    "tt.reshape",
    "tt.expand_dims",
}


def trace_base_pointer(value: str, defs: Dict[str, SSAValueDef], func_args: Set[str], *, max_depth: int = 64) -> str | None:
    """
    Trace a pointer SSA value back to a function argument base pointer.
    """
    cur = value
    for _ in range(max_depth):
        if cur in func_args:
            return cur
        d = defs.get(cur)
        if d is None:
            return None
        if d.op == "scf.iter_arg":
            cur = d.operands[0]
            continue
        if d.op not in PTR_PRESERVING_OPS:
            return None
        if not d.operands:
            return None
        # For tt.addptr/bitcast/broadcast/etc, the first operand is the base pointer-like value.
        cur = d.operands[0]
    return None


CMP_DEF_RE = re.compile(r"\barith\.(cmpi|cmpf)\s+([a-z]+),\s*(%[A-Za-z0-9_]+),\s*(%[A-Za-z0-9_]+)")
AND_RE = re.compile(r"\barith\.and[io]\b")
OR_RE = re.compile(r"\barith\.or[io]\b")


def extract_mask_witness(mask: str, defs: Dict[str, SSAValueDef], func_args: Set[str], *, max_nodes: int = 128) -> MaskWitness:
    """
    Follow SSA dependencies from a mask value and collect cmp sites.
    """
    cmps: List[MaskCmp] = []
    visited: List[str] = []
    expr_cache: Dict[str, str] = {}

    def build_expr(v: str, depth: int = 0) -> str:
        if depth > max_nodes:
            return v
        if v in expr_cache:
            return expr_cache[v]
        visited.append(v)
        if v in func_args:
            expr_cache[v] = v
            return v
        d = defs.get(v)
        if d is None:
            expr_cache[v] = v
            return v
        m = CMP_DEF_RE.search(d.line)
        if m:
            kind = m.group(2)
            lhs = m.group(3)
            rhs = m.group(4)
            cmps.append(MaskCmp(line_no=d.line_no, kind=kind, lhs=lhs, rhs=rhs, line=d.line))
            out = f"({lhs} {kind} {rhs})"
            expr_cache[v] = out
            return out
        # boolean combines
        if d.op in {"arith.andi"} or AND_RE.search(d.line):
            parts = [build_expr(opnd, depth + 1) for opnd in d.operands if opnd.startswith("%")]
            expr = " and ".join(p for p in parts if p)
            expr_cache[v] = f"({expr})" if expr else v
            return expr_cache[v]
        if d.op in {"arith.ori"} or OR_RE.search(d.line):
            parts = [build_expr(opnd, depth + 1) for opnd in d.operands if opnd.startswith("%")]
            expr = " or ".join(p for p in parts if p)
            expr_cache[v] = f"({expr})" if expr else v
            return expr_cache[v]
        # shape/broadcast ops just forward
        if d.op in {"tt.broadcast", "tt.expand_dims", "tt.splat"} and d.operands:
            expr_cache[v] = build_expr(d.operands[0], depth + 1)
            return expr_cache[v]
        expr_cache[v] = v
        return v

    formula = build_expr(mask)
    uniq: Dict[int, MaskCmp] = {c.line_no: c for c in cmps}
    return MaskWitness(mask=mask, cmps=sorted(uniq.values(), key=lambda c: c.line_no), visited=visited, formula=formula)


def _normalize_op_token(tok: str) -> str:
    # Handle quoted ops like "tt.reduce"(%x)
    if tok.startswith('"'):
        # token can be like: "tt.reduce"(%18)
        base = tok.split("(", 1)[0].strip()
        return base.strip('"')
    return tok


SSA_TOKEN_RE = re.compile(r"(%[A-Za-z0-9_]+)")


def _extract_all_ssa_operands(rhs: str) -> List[str]:
    # Take every %foo token on the RHS, in order.
    return SSA_TOKEN_RE.findall(rhs)


ITER_ARGS_RE = re.compile(r"iter_args\(([^)]*)\)")
ITER_ARG_PAIR_RE = re.compile(r"(%arg\d+)\s*=\s*(%[A-Za-z0-9_]+)")


def _extract_scf_iter_args(rhs: str) -> List[Tuple[str, str]]:
    m = ITER_ARGS_RE.search(rhs)
    if not m:
        return []
    blob = m.group(1)
    pairs = []
    for pm in ITER_ARG_PAIR_RE.finditer(blob):
        pairs.append((pm.group(1), pm.group(2)))
    return pairs


PTR_ELEM_RE = re.compile(r"!tt\.ptr<([^>]+)>")


def _infer_ptr_element_type(line: str) -> str | None:
    m = PTR_ELEM_RE.search(line)
    if not m:
        return None
    return m.group(1).strip()


__all__ = [
    "SSAValueDef",
    "PointerGroup",
    "MaskCmp",
    "MaskWitness",
    "parse_function_args",
    "parse_ssa_defs",
    "trace_base_pointer",
    "build_pointer_groups",
    "build_mask_witnesses",
    "extract_mask_witness",
]
