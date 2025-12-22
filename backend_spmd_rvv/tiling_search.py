"""
Heuristic tiling search for SPMD + RVV backend (Task6 MVP).

Given an IntentFunction (focus: matmul) and a simple SPMD profile, pick tile
sizes for outer blocking and optional vector width. This is intentionally
lightweight and deterministic to keep tests stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from intent_ir.ir_types import IntentFunction, IntentIRValidationError, Op
from triton_frontend.facts import TTIRConstraints
from .hardware_profile import RVVHardwareProfile
from .cost_model import GEMMCostModel


@dataclass
class SPMDProfile:
    num_cores: int
    cache_bytes: int | None = None
    scratchpad_bytes: int | None = None
    rvv_enabled: bool = False
    rvv_vlen_bits: int | None = None
    mem_bandwidth_gbps: float | None = None


@dataclass
class TileChoice:
    tile_m: int
    tile_n: int
    tile_k: int
    vec_width: int | None = None
    notes: List[str] = field(default_factory=list)


def _infer_matmul_axes(op: Op, intent: IntentFunction) -> tuple[str | None, str | None, str | None]:
    """
    Try to infer (M, N, K) symbolic axes from a matmul op's input tensor shapes.
    """
    if op.op != "matmul" or len(op.inputs) < 2:
        return None, None, None
    a = intent.tensors.get(op.inputs[0])
    b = intent.tensors.get(op.inputs[1])
    if not a or not b:
        return None, None, None
    try:
        m = a.shape[0].value if hasattr(a.shape[0], "value") else None
        k_a = a.shape[1].value if len(a.shape) > 1 and hasattr(a.shape[1], "value") else None
        k_b = b.shape[0].value if hasattr(b.shape[0], "value") else None
        n = b.shape[1].value if len(b.shape) > 1 and hasattr(b.shape[1], "value") else None
    except Exception:
        return None, None, None
    if k_a is not None and k_b is not None and k_a != k_b:
        return m, n, None
    return m, n, k_a or k_b


def choose_tiles(intent: IntentFunction, profile: SPMDProfile, constraints: TTIRConstraints | None = None) -> TileChoice:
    return choose_tiles_with_bindings(intent, profile, constraints=constraints, shape_bindings=None)


def _candidate_tiles(vec_lanes: int, cache_kb: int) -> List[Tuple[int, int, int]]:
    tn_list = [vec_lanes * k for k in (1, 2, 4)]
    tm_list = [16, 32, 64]
    tk_list = [16, 32, 64]
    out: List[Tuple[int, int, int]] = []
    for tm in tm_list:
        for tn in tn_list:
            for tk in tk_list:
                ws = (tm * tk + tk * tn + tm * tn) * 4
                if ws <= cache_kb * 1024:
                    out.append((tm, tn, tk))
    return out or [(16, 16, 16)]


def choose_tiles_with_bindings(
    intent: IntentFunction,
    profile: SPMDProfile | RVVHardwareProfile,
    constraints: TTIRConstraints | None = None,
    shape_bindings: Optional[dict] = None,
) -> TileChoice:
    """
    Deterministic heuristic tile chooser.

    - Prefer existing schedule.tile_* if they are positive ints.
    - Otherwise choose from a small grid {16,32,64} for M/N and {16,32} for K.
    - If scratchpad_bytes is set, enforce a simple footprint cap for matmul tiles.
    - Record notes for mask handling or fallback paths.
    """
    notes: List[str] = []
    # Shape-aware path for GEMM if bindings available and RVV profile provided.
    if shape_bindings and isinstance(profile, RVVHardwareProfile):
        M = shape_bindings.get("M")
        N = shape_bindings.get("N")
        K = shape_bindings.get("K")
        if all(isinstance(x, int) and x > 0 for x in (M, N, K)):
            vec_lanes = max(1, profile.rvv_vlen_bits // 32)
            candidates = _candidate_tiles(vec_lanes, profile.l2_cache_kb)
            model = GEMMCostModel(profile, M, N, K)
            choice = model.search_best_tile(candidates)
            return choice

    # Fallback heuristic path (original).
    sched = intent.schedule
    tile_m = 16
    tile_n = 16
    tile_k = 16
    vec_width = None

    def _take_int(x):
        return int(x) if isinstance(x, int) and x > 0 else None

    if sched:
        tm = _take_int(sched.tile_m)
        tn = _take_int(sched.tile_n)
        tk = _take_int(sched.tile_k)
        vw = _take_int(sched.vec_width)
        if tm:
            tile_m = tm
        if tn:
            tile_n = tn
        if tk:
            tile_k = tk
        if vw:
            vec_width = vw

    if getattr(profile, "scratchpad_bytes", None):
        footprint = lambda tm, tn, tk: (tm * tk + tk * tn + tm * tn) * 4
        candidates = [(tm, tn, tk) for tm in (64, 32, 16) for tn in (64, 32, 16) for tk in (32, 16)]
        for tm, tn, tk in candidates:
            if footprint(tm, tn, tk) <= profile.scratchpad_bytes:  # type: ignore[attr-defined]
                tile_m, tile_n, tile_k = tm, tn, tk
                notes.append(f"fit scratchpad<= {profile.scratchpad_bytes}")  # type: ignore[attr-defined]
                break
        else:
            notes.append("scratchpad_bytes given but no tile fits; using default")

    if constraints and getattr(constraints, "needs_mask", False):
        notes.append("mask expected (edge tiles allowed)")

    if getattr(profile, "rvv_enabled", False) and vec_width is None and getattr(profile, "rvv_vlen_bits", None):
        lanes = max(1, int(profile.rvv_vlen_bits) // 32)  # type: ignore[attr-defined]
        vec_width = lanes
        notes.append(f"rvv lanes={lanes}")

    for v, name in [(tile_m, "tile_m"), (tile_n, "tile_n"), (tile_k, "tile_k")]:
        if v <= 0:
            raise IntentIRValidationError(f"{name} must be positive, got {v}")

    return TileChoice(tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, vec_width=vec_width, notes=notes)


__all__ = ["SPMDProfile", "TileChoice", "choose_tiles"]
