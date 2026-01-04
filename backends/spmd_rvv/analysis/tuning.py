"""
Backend tuning interface for SPMD+RVV (Task6).

This module follows the A+B+C split:
- A/B (semantics + portable structure) are in IntentIR + certificates/contract.
- C (hardware-specific schedule) is treated as *tunable*.

We expose a small "Auto / Guided / Locked" interface so users can:
  - run fully automatic schedule selection (default)
  - reuse frontend schedule hints as priors (guided)
  - lock specific schedule fields while auto-tuning the rest (locked)

MVP scope: choose a ScheduleSketch for matmul-heavy kernels using the analytical
GEMM cost model. Other ops currently ignore schedule knobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

from intent_ir.ir import IntentFunction, ScheduleSketch

from .cost_model import GEMMCostModel, estimate_program_cost
from .hardware_profile import RVVHardwareProfile
from frontends.common.access_witness import build_stride_summary


TuningMode = Literal["auto", "guided", "locked"]

_INT_RE = re.compile(r"^-?\d+$")
_IN_SET_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\\s*in\\s*\\((?P<vals>[^)]*)\\)\\s*$")
_EQ_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*(?P<val>-?\\d+)\\s*$")


@dataclass(frozen=True)
class TuningRequest:
    mode: TuningMode = "auto"
    budget: int = 0
    locks: Dict[str, int] = field(default_factory=dict)
    constraints: Dict[str, Set[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class TuningResult:
    schedule: ScheduleSketch
    notes: List[str] = field(default_factory=list)
    model_op: Optional[str] = None
    model_mnk: Optional[Tuple[int, int, int]] = None


@dataclass(frozen=True)
class ScheduleCandidate:
    schedule: ScheduleSketch
    score: float
    tile_mnk: Optional[Tuple[int, int, int]] = None
    notes: List[str] = field(default_factory=list)


def _normalize_int_hints(hints: Iterable[int] | None) -> List[int]:
    out: set[int] = set()
    for x in hints or []:
        try:
            v = int(x)
        except Exception:
            continue
        if v > 0:
            out.add(v)
    return sorted(out)


def _expand_hint_values(hints: Iterable[int], *, max_val: int, align: int = 1) -> List[int]:
    """
    Expand a hint list into a small set of candidate values:
      - include h, h/2, h*2
      - snap to multiples of `align` (vector lanes) when requested
      - clamp to [1, max_val]
    """
    max_v = int(max_val)
    if max_v <= 0:
        return []
    a = max(1, int(align))
    cand: set[int] = set()
    for h0 in _normalize_int_hints(hints):
        for v0 in (h0, h0 // 2, h0 * 2):
            if v0 <= 0:
                continue
            if a > 1:
                down = (v0 // a) * a
                up = ((v0 + a - 1) // a) * a
                for v in (v0, down, up):
                    if 1 <= v <= max_v:
                        cand.add(int(v))
            else:
                if 1 <= v0 <= max_v:
                    cand.add(int(v0))
    return sorted(cand)


def _parse_kv_int(s: str) -> Tuple[str, int]:
    m = _EQ_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"expected KEY=INT, got: {s!r}")
    return m.group("key"), int(m.group("val"))


def parse_locks(items: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        k, v = _parse_kv_int(it)
        out[k] = int(v)
    return out


def parse_constraints(items: Iterable[str]) -> Dict[str, Set[int]]:
    """
    Parse a minimal constraint syntax for schedule fields:
      - "tile_n in (64,128)"
      - "tile_k=32" (treated as singleton set)

    This intentionally does not try to parse general boolean expressions yet.
    """
    out: Dict[str, Set[int]] = {}
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        m = _IN_SET_RE.match(s)
        if m:
            key = m.group("key")
            vals_s = m.group("vals")
            vals: Set[int] = set()
            for part in vals_s.split(","):
                p = part.strip()
                if not p:
                    continue
                if not _INT_RE.match(p):
                    raise ValueError(f"constraint expects integers only, got {p!r} in {s!r}")
                vals.add(int(p))
            if not vals:
                raise ValueError(f"constraint set is empty: {s!r}")
            out[key] = vals
            continue
        m = _EQ_RE.match(s)
        if m:
            key = m.group("key")
            out[key] = {int(m.group("val"))}
            continue
        raise ValueError(f"unsupported constraint syntax: {s!r}")
    return out


def _resolve_dim(d, shape_bindings: Dict[str, int]) -> Optional[int]:
    if getattr(d, "kind", None) == "const":
        try:
            return int(d.value)
        except Exception:
            return None
    if getattr(d, "kind", None) == "sym":
        key = str(d.value)
        v = shape_bindings.get(key)
        return int(v) if isinstance(v, int) else None
    return None


def _resolve_shape(intent: IntentFunction, tensor: str, shape_bindings: Dict[str, int]) -> Optional[List[int]]:
    t = intent.tensors.get(tensor)
    if t is None:
        return None
    out: List[int] = []
    for d in t.shape:
        v = _resolve_dim(d, shape_bindings)
        if v is None:
            return None
        out.append(int(v))
    return out


def _extract_mnk_for_matmul(intent: IntentFunction, op_idx: int, shape_bindings: Dict[str, int]) -> Optional[Tuple[int, int, int]]:
    op = intent.ops[op_idx]
    if op.op != "matmul" or len(op.inputs) < 2:
        return None
    a = op.inputs[0]
    b = op.inputs[1]
    a_shape = _resolve_shape(intent, a, shape_bindings)
    b_shape = _resolve_shape(intent, b, shape_bindings)
    if not a_shape or not b_shape:
        return None
    ta = bool(op.attrs.get("transpose_a", False))
    tb = bool(op.attrs.get("transpose_b", False))
    if len(a_shape) == 2 and len(b_shape) == 2:
        m = a_shape[1] if ta else a_shape[0]
        k = a_shape[0] if ta else a_shape[1]
        k2 = b_shape[1] if tb else b_shape[0]
        n = b_shape[0] if tb else b_shape[1]
        if k2 != k:
            return None
        return int(m), int(n), int(k)
    if len(a_shape) == 4 and len(b_shape) == 4:
        # [B,H,M,K] x [B,H,K,N] (or transpose_b variant)
        m = a_shape[3] if ta else a_shape[2]
        k = a_shape[2] if ta else a_shape[3]
        k2 = b_shape[3] if tb else b_shape[2]
        n = b_shape[2] if tb else b_shape[3]
        if k2 != k:
            return None
        return int(m), int(n), int(k)
    return None


def _pick_model_matmul(intent: IntentFunction, shape_bindings: Dict[str, int]) -> Tuple[Optional[int], Optional[Tuple[int, int, int]]]:
    best_idx: Optional[int] = None
    best_score = -1
    best_mnk: Optional[Tuple[int, int, int]] = None
    for i, op in enumerate(intent.ops):
        if op.op != "matmul":
            continue
        mnk = _extract_mnk_for_matmul(intent, i, shape_bindings)
        if not mnk:
            continue
        m, n, k = mnk
        score = int(m) * int(n) * int(k)
        if score > best_score:
            best_score = score
            best_idx = i
            best_mnk = mnk
    return best_idx, best_mnk


def _allowed_set(req: TuningRequest, key: str) -> Optional[Set[int]]:
    # Locked wins over constraints (singleton set).
    if key in req.locks:
        return {int(req.locks[key])}
    return req.constraints.get(key)


def _candidate_tiles(profile: RVVHardwareProfile, *, allowed: Dict[str, Optional[Set[int]]], M: int, N: int, K: int) -> List[Tuple[int, int, int]]:
    vec_lanes = max(1, int(profile.rvv_vlen_bits) // 32)
    tm_list = [16, 32, 64]
    tn_list = [vec_lanes * x for x in (1, 2, 4)]
    tk_list = [16, 32, 64]

    def filt(vals: List[int], key: str) -> List[int]:
        aset = allowed.get(key)
        if not aset:
            return vals
        return [v for v in vals if v in aset]

    tm_list = filt(tm_list, "tile_m")
    tn_list = filt(tn_list, "tile_n")
    tk_list = filt(tk_list, "tile_k")

    # Clamp to problem sizes (avoid obviously-wasted oversize tiles).
    tm_list = [v for v in tm_list if v > 0 and v <= int(M)] or [max(1, min(16, int(M)))]
    tn_list = [v for v in tn_list if v > 0 and v <= int(N)] or [max(1, min(vec_lanes, int(N)))]
    tk_list = [v for v in tk_list if v > 0 and v <= int(K)] or [max(1, min(16, int(K)))]

    out: List[Tuple[int, int, int]] = []
    for tm in tm_list:
        for tn in tn_list:
            for tk in tk_list:
                # Simple cache-fit filter.
                ws_bytes = (tm * tk + tk * tn + tm * tn) * 4
                if ws_bytes <= int(profile.l2_cache_kb) * 1024:
                    out.append((tm, tn, tk))
    return out or [(tm_list[0], tn_list[0], tk_list[0])]


def propose_schedule_candidates(
    intent: IntentFunction,
    *,
    shape_bindings: Dict[str, int],
    profile: RVVHardwareProfile,
    request: TuningRequest | None = None,
    tile_hints: Iterable[int] | None = None,
    limit: int | None = None,
    evidence: object | None = None,
) -> List[ScheduleCandidate]:
    """
    Propose ranked schedule candidates (MVP: matmul tiling).

    This is used by higher-level tools (e.g. remote autotune) to:
    - enumerate a small candidate set from constraints/locks
    - rank by analytical cost model (predicted GFLOPs)
    - optionally benchmark top-K candidates on real hardware
    """
    req = request or TuningRequest()
    hints = _normalize_int_hints(tile_hints)

    # Evidence-derived access witness (optional): used to pick vec_width and to add guided priors.
    summary = None
    try:
        summary = build_stride_summary(
            evidence,
            shape_bindings=shape_bindings,
            cache_line_bytes=int(getattr(profile, "cache_line_bytes", 64) or 64),
        )
    except Exception:
        summary = None
    if summary is not None and summary.dominant_range_len:
        hints = sorted(set(hints + [int(summary.dominant_range_len)]))

    # vec_width: default to vector lanes; can be constrained/locked.
    vec_lanes = max(1, int(profile.rvv_vlen_bits) // 32)
    vec_allowed = _allowed_set(req, "vec_width")

    # Derive candidate vec_width values:
    # - user lock/constraint wins
    # - else, if evidence suggests a contiguous range axis, try min(lanes, range_len)
    # - else fall back to scalar (1 lane) to be conservative on irregular accesses
    vec_candidates: List[int] = []
    if "vec_width" in req.locks:
        vec_candidates = [int(req.locks["vec_width"])]
        vec_notes = [f"lock vec_width={vec_candidates[0]}"]
    elif vec_allowed:
        vec_candidates = sorted((int(v) for v in vec_allowed if int(v) > 0), reverse=True)
        vec_notes = [f"vec_width constrained={sorted(vec_allowed)}"]
    else:
        if summary is not None and summary.has_contiguous_range:
            vw = vec_lanes
            if summary.dominant_range_len and int(summary.dominant_range_len) > 0:
                vw = min(vec_lanes, int(summary.dominant_range_len))
            vec_candidates = [vw]
            if vw >= 2:
                vec_candidates.append(max(1, vw // 2))
            # Preserve preference order (larger first) while de-duplicating.
            vec_candidates = list(dict.fromkeys(int(x) for x in vec_candidates if int(x) > 0))
            vec_notes = [f"vec_width from evidence={vec_candidates} lanes={vec_lanes}"]
            if summary.dominant_axis:
                vec_notes.append(f"vec_axis_hint={summary.dominant_axis}")
        else:
            vec_candidates = [1]
            vec_notes = [f"vec_width conservative=1 lanes={vec_lanes}"]

    pipeline_depth = (intent.schedule.pipeline_depth if intent.schedule else None)
    if "pipeline_depth" in req.locks:
        pipeline_depth = int(req.locks["pipeline_depth"])
        vec_notes.append(f"lock pipeline_depth={pipeline_depth}")

    model_op_idx, mnk = _pick_model_matmul(intent, shape_bindings)
    if mnk and model_op_idx is not None:
        M, N, K = mnk
        allowed = {
            "tile_m": _allowed_set(req, "tile_m"),
            "tile_n": _allowed_set(req, "tile_n"),
            "tile_k": _allowed_set(req, "tile_k"),
        }
        tiles = _candidate_tiles(profile, allowed=allowed, M=M, N=N, K=K)
        if req.mode == "guided" and hints:
            vec_lanes = max(1, int(profile.rvv_vlen_bits) // 32)
            hm = _expand_hint_values(hints, max_val=int(M), align=1)
            hn = _expand_hint_values(hints, max_val=int(N), align=max(1, min(vec_lanes, max(vec_candidates) if vec_candidates else vec_lanes)))
            hk = _expand_hint_values(hints, max_val=int(K), align=1)
            guided_tiles: set[Tuple[int, int, int]] = set()
            for tm in hm or []:
                for tn in hn or []:
                    for tk in hk or []:
                        guided_tiles.add((int(tm), int(tn), int(tk)))
            if guided_tiles:
                tiles = list(dict.fromkeys(list(tiles) + sorted(guided_tiles)))
                vec_notes.append(f"guided tile_hints={hints[:8]}{'...' if len(hints) > 8 else ''}")
        model = GEMMCostModel(profile, M=M, N=N, K=K)

        base = intent.schedule or ScheduleSketch()
        out: List[ScheduleCandidate] = []
        for vw in vec_candidates or [vec_lanes]:
            vw_i = max(1, min(int(vw), vec_lanes))
            for tm, tn, tk in tiles:
                est = model.evaluate_tile(int(tm), int(tn), int(tk))
                sched = ScheduleSketch(
                    tile_m=int(tm),
                    tile_n=int(tn),
                    tile_k=int(tk),
                    vec_width=int(vw_i),
                    pipeline_depth=pipeline_depth,
                    axis_bindings=dict(base.axis_bindings) if base else {},
                    vec_axis=(base.vec_axis if base else (summary.dominant_axis if summary is not None else None)),
                    parallel_axes=list(base.parallel_axes) if base else [],
                    memory_hint=dict(base.memory_hint) if base else {},
                )
                notes = (
                    list(vec_notes)
                    + [f"vec_width={vw_i}"]
                    + [f"cache={est.cache_level}", f"ai={est.intensity:.2f}", f"pred_gflops={est.gflops:.2f}"]
                )
                out.append(ScheduleCandidate(schedule=sched, score=float(est.gflops), tile_mnk=(int(tm), int(tn), int(tk)), notes=notes))
        out.sort(key=lambda c: c.score, reverse=True)
        if limit is not None:
            out = out[: max(0, int(limit))]
        return out

    # No matmul (or unbound): Stage-0 heuristics for generic tile/vec knobs.
    base = intent.schedule or ScheduleSketch()
    tile_m = base.tile_m if isinstance(base.tile_m, int) else None
    tile_n = base.tile_n if isinstance(base.tile_n, int) else None
    tile_k = base.tile_k if isinstance(base.tile_k, int) else None
    if "tile_m" in req.locks:
        tile_m = int(req.locks["tile_m"])
    if "tile_n" in req.locks:
        tile_n = int(req.locks["tile_n"])
    if "tile_k" in req.locks:
        tile_k = int(req.locks["tile_k"])

    # If the schedule uses symbolic tile knobs (strings), try to pick concrete ints
    # from shape bindings + tile hints. This does not attempt to interpret axis roles.
    shape_M = shape_bindings.get("M")
    shape_N = shape_bindings.get("N")
    shape_K = shape_bindings.get("K")
    vec_lanes = max(1, int(profile.rvv_vlen_bits) // 32)

    def _pick_tile(existing: int | None, dim: int | None, key: str, *, align: int = 1) -> List[int | None]:
        aset = _allowed_set(req, key)
        if existing is not None:
            return [existing]
        if key in req.locks:
            return [int(req.locks[key])]
        if aset:
            vals = sorted(int(v) for v in aset if int(v) > 0)
            if dim is not None:
                vals = [v for v in vals if v <= int(dim)]
            return vals or [None]
        if dim is None or int(dim) <= 0:
            return [None]
        # Heuristic grid + guided hints.
        grid = [16, 32, 64, 128]
        grid = [v for v in grid if v <= int(dim)]
        if req.mode == "guided" and hints:
            grid = sorted(set(grid) | set(_expand_hint_values(hints, max_val=int(dim), align=align)))
        if align > 1:
            grid = [v for v in grid if v % align == 0]
            if not grid:
                # If the dimension is smaller than the alignment unit, keep the dim itself.
                if int(dim) < int(align):
                    grid = [int(dim)]
                else:
                    grid = [int(align)]
        return grid[:8] or [None]

    tm_vals = _pick_tile(tile_m, int(shape_M) if isinstance(shape_M, int) else None, "tile_m", align=1)
    # Align the innermost tile to the candidate vector width (or lanes if unspecified).
    tn_align = max(1, min(vec_lanes, max(vec_candidates) if vec_candidates else vec_lanes))
    tn_vals = _pick_tile(tile_n, int(shape_N) if isinstance(shape_N, int) else None, "tile_n", align=tn_align)
    tk_vals = _pick_tile(tile_k, int(shape_K) if isinstance(shape_K, int) else None, "tile_k", align=1)

    # Keep candidate set small: vary only the most-relevant axis if possible.
    candidates: List[ScheduleCandidate] = []
    for vw in vec_candidates or [1]:
        vw_i = max(1, min(int(vw), vec_lanes))
        program_est = estimate_program_cost(
            intent,
            shape_bindings=shape_bindings,
            profile=profile,
            vec_width=vw_i,
            evidence=evidence,
        )
        for tm in tm_vals[:1]:
            for tk in tk_vals[:1]:
                for tn in tn_vals[:8]:
                    sched = ScheduleSketch(
                        tile_m=tm,
                        tile_n=tn,
                        tile_k=tk,
                        vec_width=int(vw_i),
                        pipeline_depth=pipeline_depth,
                        axis_bindings=dict(base.axis_bindings) if base else {},
                        vec_axis=(base.vec_axis if base else (summary.dominant_axis if summary is not None else None)),
                        parallel_axes=list(base.parallel_axes) if base else [],
                        memory_hint=dict(base.memory_hint) if base else {},
                    )
                    # Use coarse roofline estimate as the primary ranking signal.
                    score = -float(program_est.ms)
                    notes = (
                        list(vec_notes)
                        + [f"vec_width={vw_i}"]
                        + ["no bound matmul"]
                        + [f"pred_ms={program_est.ms:.3f}", f"pred_gflops={program_est.gflops:.3f}"]
                        + list(program_est.notes)
                    )
                    if req.mode == "guided" and hints:
                        notes.append(f"guided tile_hints={hints[:8]}{'...' if len(hints) > 8 else ''}")
                    candidates.append(ScheduleCandidate(schedule=sched, score=score, tile_mnk=None, notes=notes))
    candidates.sort(key=lambda c: c.score, reverse=True)
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]
    return candidates or [ScheduleCandidate(schedule=(intent.schedule or ScheduleSketch()), score=0.0, tile_mnk=None, notes=list(vec_notes) + ["no candidates"])]


def select_schedule(
    intent: IntentFunction,
    *,
    shape_bindings: Dict[str, int],
    profile: RVVHardwareProfile,
    request: TuningRequest | None = None,
    tile_hints: Iterable[int] | None = None,
    evidence: object | None = None,
) -> TuningResult:
    """
    Select a ScheduleSketch for the backend.

    MVP strategy:
    - If the kernel contains matmul and M/N/K are bound, run the GEMM cost model
      on a small candidate grid (Auto).
    - Guided: if existing schedule has int tiles, include them in the candidate
      neighborhood; otherwise behaves like Auto.
    - Locked: apply locks and only tune remaining fields.
    """
    req = request or TuningRequest()
    notes: List[str] = [f"mode={req.mode}"]

    candidates = propose_schedule_candidates(
        intent,
        shape_bindings=shape_bindings,
        profile=profile,
        request=req,
        tile_hints=tile_hints,
        evidence=evidence,
    )
    best = candidates[0] if candidates else None
    if best is None:
        return TuningResult(schedule=(intent.schedule or ScheduleSketch()), notes=notes + ["no candidates"], model_op=None, model_mnk=None)

    schedule = best.schedule
    notes.extend(list(best.notes))
    if best.tile_mnk is not None:
        tm, tn, tk = best.tile_mnk
        notes.append(f"picked_tile=({tm},{tn},{tk})")

    model_op_idx, mnk = _pick_model_matmul(intent, shape_bindings)
    return TuningResult(
        schedule=schedule,
        notes=notes,
        model_op=(f"ops[{model_op_idx}].matmul" if model_op_idx is not None else None),
        model_mnk=mnk,
    )


__all__ = [
    "TuningMode",
    "TuningRequest",
    "TuningResult",
    "parse_locks",
    "parse_constraints",
    "ScheduleCandidate",
    "propose_schedule_candidates",
    "select_schedule",
]
