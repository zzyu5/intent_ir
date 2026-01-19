from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from frontends.common.evidence import AccessSummary, IndexExpr, Predicate

from .expr import AffineExpr, parse_int_literal, render_affine


@dataclass(frozen=True)
class Flatten2DExpr:
    """
    Row-major flatten: idx = row * STRIDE + col.

    STRIDE is kept as a symbol name (e.g., "N") so we can recover (row,col)
    even when STRIDE is dynamic and would break strict affine recovery.
    """

    row: AffineExpr
    col: AffineExpr
    stride_sym: str


@dataclass(frozen=True)
class _PtrVal:
    tensor: str


@dataclass(frozen=True)
class _OffsetVal:
    # Element index (not bytes). For 2D flatten we keep the structured form.
    idx: AffineExpr | Flatten2DExpr
    elem_bytes: int


@dataclass(frozen=True)
class _AddrVal:
    tensor: str
    idx: AffineExpr | Flatten2DExpr
    elem_bytes: int


@dataclass
class ParsedPTX:
    anchors: Dict[str, Any]
    accesses: List[AccessSummary]
    predicate_clauses: List[str]
    symbol_ranges: Dict[str, Dict[str, int]]
    tile_hints: List[int]
    raw: Dict[str, Any] = field(default_factory=dict)


_ENTRY_RE = re.compile(r"^\s*\.visible\s+\.entry\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PARAM_RE = re.compile(r"^\s*\.param\s+\.[A-Za-z0-9]+\s+(?P<name>[A-Za-z0-9_]+)\s*,?\s*$")

_LD_PARAM_RE = re.compile(
    r"^\s*ld\.param\.(?P<ty>u64|u32|s32)\s+(?P<dst>%[a-z0-9]+)\s*,\s*\[(?P<param>[A-Za-z0-9_]+)\]\s*;\s*$"
)
_CVTA_RE = re.compile(r"^\s*cvta\.to\.global\.u64\s+(?P<dst>%rd\d+)\s*,\s*(?P<src>%rd\d+)\s*;\s*$")
_MOV_SPECIAL_RE = re.compile(
    r"^\s*mov\.u32\s+(?P<dst>%r\d+)\s*,\s*%(?P<spec>ctaid|tid|ntid)\.(?P<axis>x|y|z)\s*;\s*$"
)
_MOV_IMM_RE = re.compile(r"^\s*mov\.(?P<ty>u32|s32|u64|b64)\s+(?P<dst>%[a-z0-9]+)\s*,\s*(?P<imm>[-0-9a-fA-FxX]+)\s*;\s*$")

_ADD_S32_RE = re.compile(r"^\s*add\.(?P<ty>s32|u32)\s+(?P<dst>%r\d+)\s*,\s*(?P<a>%r\d+)\s*,\s*(?P<b>%r\d+)\s*;\s*$")
_SUB_S32_RE = re.compile(r"^\s*sub\.(?P<ty>s32|u32)\s+(?P<dst>%r\d+)\s*,\s*(?P<a>%r\d+)\s*,\s*(?P<b>%r\d+)\s*;\s*$")
_MUL_LO_RE = re.compile(
    r"^\s*mul\.lo\.s32\s+(?P<dst>%r\d+)\s*,\s*(?P<a>%r\d+)\s*,\s*(?P<b>%r\d+|[-0-9a-fA-FxX]+)\s*;\s*$"
)
_MAD_LO_RE = re.compile(r"^\s*mad\.lo\.s32\s+(?P<dst>%r\d+)\s*,\s*(?P<a>%r\d+)\s*,\s*(?P<b>%r\d+)\s*,\s*(?P<c>%r\d+)\s*;\s*$")

_MUL_WIDE_RE = re.compile(
    r"^\s*mul\.wide\.(?P<ty>s32|u32)\s+(?P<dst>%rd\d+)\s*,\s*(?P<src>%r\d+)\s*,\s*(?P<imm>[-0-9a-fA-FxX]+)\s*;\s*$"
)
_ADD_S64_RE = re.compile(r"^\s*add\.(?:s64|u64)\s+(?P<dst>%rd\d+)\s*,\s*(?P<a>%rd\d+)\s*,\s*(?P<b>%rd\d+)\s*;\s*$")

_CVT_S64_S32_RE = re.compile(r"^\s*cvt\.(?:s64|u64)\.(?:s32|u32)\s+(?P<dst>%rd\d+)\s*,\s*(?P<src>%r\d+)\s*;\s*$")
_SHL_B64_RE = re.compile(r"^\s*shl\.b64\s+(?P<dst>%rd\d+)\s*,\s*(?P<src>%rd\d+)\s*,\s*(?P<imm>\d+)\s*;\s*$")

_SETP_RE = re.compile(r"^\s*setp\.(?P<cmp>ge|gt|le|lt)\.s32\s+(?P<pred>%p\d+)\s*,\s*(?P<lhs>%r\d+)\s*,\s*(?P<rhs>%r\d+)\s*;\s*$")
_BRA_PRED_RE = re.compile(r"^\s*@(?P<pred>%p\d+)\s+bra\s+\$L.*;\s*$")

_LD_GLOBAL_RE = re.compile(
    r"^\s*(?:@(?P<pred>%p\d+)\s+)?"
    r"ld\.global(?:\.[A-Za-z0-9]+)*\.(?P<ty>f16|f32|u8|s8|u16|u32|s32)\s+"
    r"(?:%[a-z0-9]+|\{[^\}]*\})\s*,\s*"
    r"\[(?P<addr>%rd\d+)(?P<off>(?:\+\-?\d+|-\d+))?\]\s*;\s*$"
)
_ST_GLOBAL_RE = re.compile(
    r"^\s*(?:@(?P<pred>%p\d+)\s+)?"
    r"st\.global(?:\.[A-Za-z0-9]+)*\.(?P<ty>f16|f32|u8|s8|u16|u32|s32)\s+"
    r"\[(?P<addr>%rd\d+)(?P<off>(?:\+\-?\d+|-\d+))?\]\s*,\s*"
    r"(?:%[a-z0-9]+|\{[^\}]*\})\s*;\s*$"
)


def _dtype_from_ptx_suffix(s: str) -> Tuple[str, int]:
    suf = str(s)
    if suf == "f16":
        return ("f16", 2)
    if suf == "f32":
        return ("f32", 4)
    if suf in {"u8", "s8"}:
        return ("u8" if suf == "u8" else "i8", 1)
    if suf == "u16":
        return ("u16", 2)
    if suf == "u32":
        return ("u32", 4)
    if suf == "s32":
        return ("i32", 4)
    return ("unknown", 4)


def _shape_syms(shape: List[object]) -> set[str]:
    out: set[str] = set()
    for d in shape:
        if isinstance(d, str) and d:
            out.add(str(d))
    return out


def _shape_num_elems(shape: List[object], canonical_shapes: Dict[str, Any]) -> Optional[int]:
    prod = 1
    for d in shape:
        if isinstance(d, int):
            v = d
        elif isinstance(d, str):
            if d in canonical_shapes:
                try:
                    v = int(canonical_shapes[d])
                except Exception:
                    return None
            else:
                # Unknown symbolic dim.
                return None
        else:
            return None
        if v < 0:
            return None
        prod *= int(v)
    return int(prod)


def _block_dim_from_launch(launch: Dict[str, Any], axis: str) -> Optional[int]:
    blk = launch.get("block")
    if isinstance(blk, (list, tuple)) and len(blk) >= 3:
        idx = {"x": 0, "y": 1, "z": 2}[str(axis)]
        try:
            v = int(blk[idx])
        except Exception:
            return None
        return v if v > 0 else None
    return None


def _symbol_for_special(spec: str, axis: str) -> Optional[AffineExpr]:
    if spec == "ctaid":
        return AffineExpr.sym({"x": "pid0", "y": "pid1", "z": "pid2"}[axis])
    if spec == "tid":
        return AffineExpr.sym({"x": "r0", "y": "r1", "z": "r2"}[axis])
    return None


def _mul_affine(a: AffineExpr, b: AffineExpr) -> AffineExpr:
    if not (a.ok and b.ok):
        return AffineExpr(ok=False)
    ca = a.const_int()
    cb = b.const_int()
    if ca is not None:
        return b.mul_const(ca)
    if cb is not None:
        return a.mul_const(cb)
    return AffineExpr(ok=False)


def _maybe_flatten2d(a: AffineExpr, b: AffineExpr, c: AffineExpr) -> Optional[Flatten2DExpr]:
    """
    Detect row-major flatten pattern: row * STRIDE + col, where STRIDE is a scalar symbol.
    """
    stride = b.is_single_symbol()
    if stride is None:
        return None
    if not (a.ok and c.ok):
        return None
    return Flatten2DExpr(row=a, col=c, stride_sym=stride)


def parse_ptx_kernel(
    ptx_text: str,
    *,
    kernel_name: str,
    io_spec: Dict[str, Any],
    launch: Dict[str, Any],
) -> ParsedPTX:
    """
    Parse a PTX module text and extract Tier-A evidence for a single kernel entry.
    """
    lines = [ln.rstrip() for ln in str(ptx_text).splitlines()]

    # 1) Find entry header and param list.
    params: List[str] = []
    in_entry = False
    for ln in lines:
        m = _ENTRY_RE.match(ln)
        if m and str(m.group("name")) == str(kernel_name):
            in_entry = True
            continue
        if in_entry:
            if ln.strip().startswith(")"):
                break
            pm = _PARAM_RE.match(ln)
            if pm:
                params.append(str(pm.group("name")))
    arg_names = io_spec.get("arg_names") if isinstance(io_spec.get("arg_names"), list) else []
    arg_names = [str(x) for x in arg_names]
    param_to_arg: Dict[str, str] = {}
    for i, p in enumerate(params):
        if i < len(arg_names):
            param_to_arg[p] = arg_names[i]
        else:
            param_to_arg[p] = f"arg{i}"

    tensors_spec = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), dict) else {}
    scalars_spec = io_spec.get("scalars") if isinstance(io_spec.get("scalars"), dict) else {}

    # 2) Anchors (MVP hard signals).
    ptx_all = "\n".join(lines)
    anchors: Dict[str, Any] = {
        "kernel_kind_hint": "cuda_ptx",
        # `has_copy` is refined after access recovery; initialize to True so we
        # don't accidentally regress older golden tests on partial parsers.
        "has_copy": True,
        "has_dot": ("wmma." in ptx_all) or ("mma." in ptx_all),
        "has_reduce": False,
        "has_atomic": ("atom." in ptx_all) or ("atom.global" in ptx_all),
        "has_barrier": ("bar.sync" in ptx_all) or ("barrier" in ptx_all),
        "has_async": ("cp.async" in ptx_all) or ("ldmatrix" in ptx_all),
    }

    # 3) Track regs.
    reg_aff: Dict[str, AffineExpr] = {}
    pred_cmps: List[Tuple[str, str, str]] = []  # (cmp, lhs_reg, rhs_reg)
    saw_branch = False

    ptr_base: Dict[str, _PtrVal] = {}
    byte_offsets: Dict[str, _OffsetVal] = {}
    addr_regs: Dict[str, _AddrVal] = {}

    def get_aff(reg: str) -> AffineExpr:
        return reg_aff.get(reg, AffineExpr(ok=False))

    for ln in lines:
        # scalar/ptr param loads
        m = _LD_PARAM_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            param = str(m.group("param"))
            arg = str(param_to_arg.get(param, param))
            if dst.startswith("%rd"):
                if arg in tensors_spec:
                    ptr_base[dst] = _PtrVal(tensor=arg)
                else:
                    reg_aff[dst] = AffineExpr.sym(arg)
            else:
                reg_aff[dst] = AffineExpr.sym(arg)
            continue

        # cvta: pointer cast
        m = _CVTA_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            src = str(m.group("src"))
            if src in ptr_base:
                ptr_base[dst] = ptr_base[src]
            elif src in addr_regs:
                addr_regs[dst] = addr_regs[src]
            continue

        # mov special regs
        m = _MOV_SPECIAL_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            spec = str(m.group("spec"))
            axis = str(m.group("axis"))
            if spec == "ntid":
                v = _block_dim_from_launch(launch, axis)
                if v is None:
                    reg_aff[dst] = AffineExpr(ok=False)
                else:
                    reg_aff[dst] = AffineExpr.const_val(v)
            else:
                sym = _symbol_for_special(spec, axis)
                reg_aff[dst] = sym if sym is not None else AffineExpr(ok=False)
            continue

        # mov imm
        m = _MOV_IMM_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            imm = parse_int_literal(m.group("imm"))
            if imm is not None:
                reg_aff[dst] = AffineExpr.const_val(int(imm))
            continue

        # cvt s64/u64 from s32/u32 (propagate affine)
        m = _CVT_S64_S32_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            src = str(m.group("src"))
            reg_aff[dst] = get_aff(src)
            continue

        # add/sub s32
        m = _ADD_S32_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            reg_aff[dst] = get_aff(str(m.group("a"))).add(get_aff(str(m.group("b"))))
            continue
        m = _SUB_S32_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            reg_aff[dst] = get_aff(str(m.group("a"))).sub(get_aff(str(m.group("b"))))
            continue

        # mul.lo (const or reg)
        m = _MUL_LO_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            a = get_aff(str(m.group("a")))
            b_tok = str(m.group("b"))
            if b_tok.startswith("%r"):
                b = get_aff(b_tok)
            else:
                imm = parse_int_literal(b_tok)
                b = AffineExpr.const_val(int(imm)) if imm is not None else AffineExpr(ok=False)
            reg_aff[dst] = _mul_affine(a, b)
            continue

        # mad.lo
        m = _MAD_LO_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            a = get_aff(str(m.group("a")))
            b = get_aff(str(m.group("b")))
            c = get_aff(str(m.group("c")))
            flat = _maybe_flatten2d(a, b, c)
            if flat is not None:
                # Represent flatten2d as unresolved affine, but keep structured form
                # via an auxiliary map keyed by dst register.
                # We store it in meta by encoding in reg_aff as <unresolved> and
                # keeping the real structure in byte_offsets when needed.
                reg_aff[dst] = AffineExpr(ok=False)
                # Store a synthetic tag for later lookup.
                # (We reuse byte_offsets for capturing flatten2d produced regs.)
                byte_offsets[f"__flat2d__{dst}"] = _OffsetVal(idx=flat, elem_bytes=1)
            else:
                reg_aff[dst] = _mul_affine(a, b).add(c)
            continue

        # setp comparisons (collect until first early-exit branch)
        if not saw_branch:
            m = _SETP_RE.match(ln)
            if m:
                pred_cmps.append((str(m.group("cmp")), str(m.group("lhs")), str(m.group("rhs"))))
                continue
            if _BRA_PRED_RE.match(ln):
                saw_branch = True
                continue

        # shift-left b64 (common byte-offset pattern: idx << log2(elem_bytes))
        m = _SHL_B64_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            src = str(m.group("src"))
            imm = parse_int_literal(str(m.group("imm")))
            if imm is None:
                continue
            shift = int(imm)
            if shift < 0 or shift > 63:
                continue
            mul = 1 << shift
            src_aff = get_aff(src)
            if src_aff.ok:
                reg_aff[dst] = src_aff.mul_const(mul)
                if mul in {1, 2, 4, 8}:
                    byte_offsets[dst] = _OffsetVal(idx=src_aff, elem_bytes=int(mul))
            continue

        # offset bytes (mul.wide idx, elem_bytes) — only treat {1,2,4,8} as element-bytes.
        m = _MUL_WIDE_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            src = str(m.group("src"))
            imm = parse_int_literal(str(m.group("imm")))
            if imm is None:
                continue
            imm_i = int(imm)
            # Recover flatten2d form if src was marked as such.
            key = f"__flat2d__{src}"
            if key in byte_offsets:
                idx = byte_offsets[key].idx
            else:
                idx = get_aff(src)
            if isinstance(idx, AffineExpr) and idx.ok:
                reg_aff[dst] = idx.mul_const(imm_i)
            else:
                reg_aff[dst] = AffineExpr(ok=False)
            if imm_i in {1, 2, 4, 8}:
                byte_offsets[dst] = _OffsetVal(idx=idx, elem_bytes=int(imm_i))
            continue

        # ptr add (base + offset)
        m = _ADD_S64_RE.match(ln)
        if m:
            dst = str(m.group("dst"))
            a = str(m.group("a"))
            b = str(m.group("b"))
            base = None
            off_reg = None
            if a in ptr_base:
                base = ptr_base[a].tensor
                off_reg = b
            elif a in addr_regs:
                base = addr_regs[a].tensor
                off_reg = b
            elif b in ptr_base:
                base = ptr_base[b].tensor
                off_reg = a
            elif b in addr_regs:
                base = addr_regs[b].tensor
                off_reg = a
            if base is not None and off_reg is not None:
                if off_reg in byte_offsets:
                    off = byte_offsets[off_reg]
                    addr_regs[dst] = _AddrVal(tensor=base, idx=off.idx, elem_bytes=int(off.elem_bytes))
                elif off_reg in reg_aff and reg_aff[off_reg].ok:
                    # Unknown unit (bytes vs elems). Use elem_bytes=1 and mark unresolved later.
                    addr_regs[dst] = _AddrVal(tensor=base, idx=reg_aff[off_reg], elem_bytes=1)
                else:
                    addr_regs[dst] = _AddrVal(tensor=base, idx=AffineExpr(ok=False), elem_bytes=1)
            continue

    # 4) Predicate clauses (MVP): invert early-exit setp.ge into "<" clauses.
    predicate_clauses: List[str] = []
    for cmp_, lhs_r, rhs_r in pred_cmps:
        lhs = get_aff(lhs_r)
        rhs = get_aff(rhs_r)
        if not (lhs.ok and rhs.ok):
            continue
        if cmp_ == "ge":
            predicate_clauses.append(f"{render_affine(lhs)} < {render_affine(rhs)}")
        elif cmp_ == "gt":
            predicate_clauses.append(f"{render_affine(lhs)} <= {render_affine(rhs)}")
        elif cmp_ == "lt":
            predicate_clauses.append(f"{render_affine(lhs)} >= {render_affine(rhs)}")
        elif cmp_ == "le":
            predicate_clauses.append(f"{render_affine(lhs)} > {render_affine(rhs)}")

    pred_obj: Optional[Predicate] = Predicate(clauses=sorted(set(predicate_clauses))) if predicate_clauses else None

    # 5) Accesses.
    accesses: List[AccessSummary] = []
    for ln in lines:
        m = _LD_GLOBAL_RE.match(ln)
        kind = None
        ty = None
        addr = None
        off_raw = None
        if m:
            kind = "load"
            ty = str(m.group("ty"))
            addr = str(m.group("addr"))
            off_raw = m.group("off")
        else:
            m2 = _ST_GLOBAL_RE.match(ln)
            if m2:
                kind = "store"
                ty = str(m2.group("ty"))
                addr = str(m2.group("addr"))
                off_raw = m2.group("off")
        if not kind:
            continue

        dt, elem_bytes = _dtype_from_ptx_suffix(str(ty))
        tensor = "<unresolved>"
        idx_exprs: List[IndexExpr] = [IndexExpr(terms={}, const=0)]
        meta: Dict[str, Any] = {}
        if off_raw:
            s = str(off_raw).strip()
            # PTX sometimes prints "+-32" instead of "-32".
            if s.startswith("+-"):
                s = "-" + s[2:]
            try:
                meta["addr_off_bytes"] = int(s)
            except Exception:
                pass

        ai = addr_regs.get(str(addr))
        if ai is not None:
            tensor = str(ai.tensor)
            rank = int((tensors_spec.get(tensor) or {}).get("rank", 1)) if isinstance(tensors_spec, dict) else 1
            idx = ai.idx
            if isinstance(idx, Flatten2DExpr) and rank == 2:
                idx_exprs = [
                    IndexExpr(terms=dict(idx.row.terms), const=int(idx.row.const)),
                    IndexExpr(terms=dict(idx.col.terms), const=int(idx.col.const)),
                ]
            elif isinstance(idx, AffineExpr) and idx.ok:
                idx_exprs = [IndexExpr(terms=dict(idx.terms), const=int(idx.const))]
            else:
                meta["unresolved"] = True
        else:
            # Fall back to ptr base mapping when add-tree recovery fails.
            if str(addr) in ptr_base:
                tensor = str(ptr_base[str(addr)].tensor)
            meta["unresolved"] = True

        rank = int((tensors_spec.get(tensor) or {}).get("rank", 1)) if isinstance(tensors_spec, dict) else 1
        accesses.append(
            AccessSummary(
                kind=str(kind),  # type: ignore[arg-type]
                tensor=str(tensor),
                dtype=str(dt),
                rank=int(rank),
                index_exprs=list(idx_exprs),
                predicate=pred_obj,
                address_space="global",
                meta=meta,
            )
        )

    # 5.1) Infer has_reduce (anchor) from IO shape relations + PTX accesses.
    canonical_shapes = launch.get("canonical_shapes") if isinstance(launch.get("canonical_shapes"), dict) else {}

    def tensor_shape(t: str) -> Optional[List[object]]:
        spec = tensors_spec.get(t) if isinstance(tensors_spec, dict) else None
        if isinstance(spec, dict) and isinstance(spec.get("shape"), list):
            return list(spec.get("shape"))  # type: ignore[return-value]
        return None

    # `has_copy` is a weak-but-useful semantic anchor: any global load+store.
    # Do NOT require tensor name resolution here; otherwise Tier-A kernels with
    # unresolved ptr mapping would become OUT_OF_SCOPE (O1 FAIL) too easily.
    anchors["has_copy"] = any(a.kind == "load" for a in accesses) and any(a.kind == "store" for a in accesses)

    read_tensors = {a.tensor for a in accesses if a.kind == "load" and a.tensor in tensors_spec}
    write_tensors = {a.tensor for a in accesses if a.kind == "store" and a.tensor in tensors_spec}

    def _is_int_index_tensor(t: str) -> bool:
        spec = tensors_spec.get(t) if isinstance(tensors_spec, dict) else None
        if not isinstance(spec, dict):
            return False
        dt = spec.get("dtype")
        if not isinstance(dt, str):
            return False
        d = dt.lower().strip()
        return d in {"i1", "i8", "i16", "i32", "i64", "u1", "u8", "u16", "u32", "u64"}

    def _is_gather_like_write(wt: str) -> bool:
        """
        Heuristic guard against false-positive `has_reduce` when the output rank
        is smaller than some inputs purely due to *indexing* (gather), not due to
        a reduction.

        Pattern:
          - write tensor shape == index tensor shape (int dtype)
          - there exists another read tensor with higher rank (data source)
        """
        wshape = tensor_shape(wt)
        if not wshape:
            return False
        wrank = len(wshape)
        # Any int index tensor with the same shape as the output.
        has_index = False
        for rt in sorted(read_tensors):
            if not _is_int_index_tensor(rt):
                continue
            rshape = tensor_shape(rt)
            if rshape and list(rshape) == list(wshape):
                has_index = True
                break
        if not has_index:
            return False
        # Any higher-rank non-index read tensor.
        for rt in sorted(read_tensors):
            if _is_int_index_tensor(rt):
                continue
            rshape = tensor_shape(rt)
            if rshape and len(rshape) > wrank:
                return True
        return False

    has_reduce = False
    gather_like_writes = {wt for wt in sorted(write_tensors) if _is_gather_like_write(wt)}

    # Strong signals: any written tensor is lower-rank or has fewer elements than some read tensor.
    for wt in sorted(write_tensors):
        if wt in gather_like_writes:
            continue
        wshape = tensor_shape(wt)
        if not wshape:
            continue
        wrank = len(wshape)
        wne = _shape_num_elems(wshape, canonical_shapes)
        for rt in sorted(read_tensors):
            rshape = tensor_shape(rt)
            if not rshape:
                continue
            rrank = len(rshape)
            if wrank < rrank:
                has_reduce = True
                break
            rne = _shape_num_elems(rshape, canonical_shapes)
            if wne is not None and rne is not None and wne < rne:
                has_reduce = True
                break
        if has_reduce:
            break

    # Fallback signal: symbols used by reads form a strict superset of symbols used by writes,
    # with no new symbols introduced by writes (typical "reduce over K" / "reduce over N").
    if not has_reduce and not gather_like_writes:
        read_syms: set[str] = set()
        write_syms: set[str] = set()
        for t in sorted(read_tensors):
            shp = tensor_shape(t)
            if shp:
                read_syms |= _shape_syms(shp)
        for t in sorted(write_tensors):
            shp = tensor_shape(t)
            if shp:
                write_syms |= _shape_syms(shp)
        missing = read_syms - write_syms
        new = write_syms - read_syms
        if missing and not new:
            has_reduce = True

    anchors["has_reduce"] = bool(has_reduce)

    # Infer has_dot (matmul-like) from tensor shape symbols:
    #   A:[M,K], B:[K,N], C:[M,N]  -> has_dot
    #
    # This is intentionally conservative: it requires two distinct read tensors
    # and one written tensor, all rank-2, sharing a single reduction symbol.
    def _infer_has_dot() -> bool:
        for wt in sorted(write_tensors):
            wshape = tensor_shape(wt)
            if not wshape or len(wshape) != 2:
                continue
            osyms = _shape_syms(wshape)
            if len(osyms) != 2:
                continue
            for a in sorted(read_tensors):
                if a == wt:
                    continue
                ashp = tensor_shape(a)
                if not ashp or len(ashp) != 2:
                    continue
                asyms = _shape_syms(ashp)
                for b in sorted(read_tensors):
                    if b == wt or b == a:
                        continue
                    bshp = tensor_shape(b)
                    if not bshp or len(bshp) != 2:
                        continue
                    bsyms = _shape_syms(bshp)
                    # Candidate reduction symbols are those present in reads but not in output.
                    red = (asyms | bsyms) - osyms
                    if len(red) != 1:
                        continue
                    (k_sym,) = tuple(red)
                    if (k_sym not in asyms) or (k_sym not in bsyms):
                        continue
                    a_out = asyms & osyms
                    b_out = bsyms & osyms
                    if len(a_out) != 1 or len(b_out) != 1:
                        continue
                    if a_out == b_out:
                        continue
                    # No extra symbols besides M/N/K.
                    if (asyms | bsyms | osyms) != (osyms | {k_sym}):
                        continue
                    return True
        return False

    if not bool(anchors.get("has_dot")):
        anchors["has_dot"] = bool(_infer_has_dot())

    # 6) Symbol ranges (best-effort, for SMT domains).
    # Use canonical shapes to approximate pid domains for deterministic checks.
    bx = _block_dim_from_launch(launch, "x") or 1
    by = _block_dim_from_launch(launch, "y") or 1
    bz = _block_dim_from_launch(launch, "z") or 1
    n = canonical_shapes.get("N") if isinstance(canonical_shapes, dict) else None
    m_ = canonical_shapes.get("M") if isinstance(canonical_shapes, dict) else None
    try:
        n_i = int(n) if isinstance(n, (int, float, str)) and str(n).isdigit() else None
    except Exception:
        n_i = None
    try:
        m_i = int(m_) if isinstance(m_, (int, float, str)) and str(m_).isdigit() else None
    except Exception:
        m_i = None

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b if b > 0 else 1

    pid0_end = ceil_div(n_i, bx) if n_i is not None else 1
    pid1_end = ceil_div(m_i, by) if m_i is not None else 1

    symbol_ranges: Dict[str, Dict[str, int]] = {
        "r0": {"start": 0, "end": int(bx)},
        "r1": {"start": 0, "end": int(by)},
        "r2": {"start": 0, "end": int(bz)},
        "pid0": {"start": 0, "end": int(max(1, pid0_end))},
        "pid1": {"start": 0, "end": int(max(1, pid1_end))},
        "pid2": {"start": 0, "end": 1},
    }
    tile_hints = sorted({int(bx), int(by), int(bz)})

    return ParsedPTX(
        anchors=anchors,
        accesses=accesses,
        predicate_clauses=list(sorted(set(predicate_clauses))),
        symbol_ranges=symbol_ranges,
        tile_hints=tile_hints,
        raw={
            "ptx_params": params,
            "param_to_arg": dict(param_to_arg),
            "num_accesses": int(len(accesses)),
        },
    )


__all__ = ["Flatten2DExpr", "ParsedPTX", "parse_ptx_kernel"]
