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

from .cost_model import GEMMCostModel
from .hardware_profile import RVVHardwareProfile


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


def select_schedule(
    intent: IntentFunction,
    *,
    shape_bindings: Dict[str, int],
    profile: RVVHardwareProfile,
    request: TuningRequest | None = None,
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

    model_op_idx, mnk = _pick_model_matmul(intent, shape_bindings)
    tile_m = tile_n = tile_k = None

    # vec_width: default to vector lanes; can be constrained/locked.
    vec_lanes = max(1, int(profile.rvv_vlen_bits) // 32)
    vec_allowed = _allowed_set(req, "vec_width")
    if vec_allowed:
        vec_width = sorted(vec_allowed)[0]
        notes.append(f"vec_width constrained={sorted(vec_allowed)}")
    else:
        vec_width = vec_lanes
        notes.append(f"vec_width default=lanes({vec_lanes})")

    if mnk and model_op_idx is not None:
        M, N, K = mnk
        allowed = {
            "tile_m": _allowed_set(req, "tile_m"),
            "tile_n": _allowed_set(req, "tile_n"),
            "tile_k": _allowed_set(req, "tile_k"),
        }
        candidates = _candidate_tiles(profile, allowed=allowed, M=M, N=N, K=K)

        # Guided: add neighborhood around existing schedule if it is concrete.
        if req.mode == "guided" and intent.schedule:
            for key in ("tile_m", "tile_n", "tile_k"):
                v = getattr(intent.schedule, key, None)
                if isinstance(v, int) and v > 0:
                    for cand in (v // 2, v, v * 2):
                        if cand > 0:
                            # We only inject candidates for the matching axis, keeping others from the grid.
                            pass
            notes.append("guided: existing schedule ints are noted (neighborhood search TODO)")

        model = GEMMCostModel(profile, M=M, N=N, K=K)
        best = model.search_best_tile(candidates)
        tile_m, tile_n, tile_k = best.tile_m, best.tile_n, best.tile_k
        notes.extend(list(best.notes))
        notes.append(f"picked_tile=({tile_m},{tile_n},{tile_k}) for M,N,K=({M},{N},{K})")
    else:
        # No matmul (or no concrete bindings): keep existing schedule tiles if present.
        if intent.schedule:
            if isinstance(intent.schedule.tile_m, int):
                tile_m = intent.schedule.tile_m
            if isinstance(intent.schedule.tile_n, int):
                tile_n = intent.schedule.tile_n
            if isinstance(intent.schedule.tile_k, int):
                tile_k = intent.schedule.tile_k
        notes.append("no bound matmul; keep existing tile hints if any")

    # Apply locks last (strongest).
    if "tile_m" in req.locks:
        tile_m = int(req.locks["tile_m"])
    if "tile_n" in req.locks:
        tile_n = int(req.locks["tile_n"])
    if "tile_k" in req.locks:
        tile_k = int(req.locks["tile_k"])

    schedule = ScheduleSketch(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        vec_width=vec_width,
        pipeline_depth=(intent.schedule.pipeline_depth if intent.schedule else None),
        axis_bindings=(dict(intent.schedule.axis_bindings) if intent.schedule else {}),
        vec_axis=(intent.schedule.vec_axis if intent.schedule else None),
        parallel_axes=(list(intent.schedule.parallel_axes) if intent.schedule else []),
        memory_hint=(dict(intent.schedule.memory_hint) if intent.schedule else {}),
    )
    return TuningResult(schedule=schedule, notes=notes, model_op=(f"ops[{model_op_idx}].matmul" if model_op_idx is not None else None), model_mnk=mnk)


__all__ = [
    "TuningMode",
    "TuningRequest",
    "TuningResult",
    "parse_locks",
    "parse_constraints",
    "select_schedule",
]

