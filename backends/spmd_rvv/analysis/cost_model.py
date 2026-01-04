"""
Analytical cost model (roofline-style) for GEMM tiling (Task6).

This is intentionally simple: estimates GFLOPs for a (tm, tn, tk) tile using
cache-fit heuristics and vector lane alignment. It is not cycle-accurate but
gives a ranking signal for tile search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .hardware_profile import RVVHardwareProfile
from intent_ir.ir import IntentFunction


@dataclass(frozen=True)
class CostEstimate:
    gflops: float
    cache_level: str
    intensity: float


def _working_set_bytes(tm: int, tn: int, tk: int, dtype_size: int = 4) -> int:
    return (tm * tk + tk * tn + tm * tn) * dtype_size


@dataclass(frozen=True)
class ProgramCostEstimate:
    """
    Coarse roofline-style estimate for non-matmul-heavy programs.

    This is used to reason about reduce/elementwise kernels and to drive
    schedule selection when no matmul is available for the GEMM model.
    """

    gflops: float
    ms: float
    intensity: float
    flops: float
    bytes: float
    notes: Tuple[str, ...] = ()


def _dtype_bytes(dt: str) -> int:
    if dt in {"bool", "i1", "u8"}:
        return 1
    if dt in {"i8"}:
        return 1
    if dt in {"i32", "f32"}:
        return 4
    if dt in {"i64", "f64"}:
        return 8
    # Conservative default (treat unknown as f32).
    return 4


def _resolve_dim_token(tok, bindings: Dict[str, int]) -> Optional[int]:
    if isinstance(tok, int):
        return int(tok)
    if isinstance(tok, float):
        return int(tok)
    if isinstance(tok, str):
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            return int(tok)
        v = bindings.get(tok)
        return int(v) if isinstance(v, int) else None
    return None


def _resolve_numel(intent: IntentFunction, name: str, bindings: Dict[str, int]) -> Optional[int]:
    t = intent.tensors.get(name)
    if t is None:
        return None
    n = 1
    for d in t.shape:
        v = _resolve_dim_token(getattr(d, "value", d), bindings)
        if v is None:
            return None
        if v <= 0:
            return None
        n *= int(v)
    return int(n)


def estimate_program_cost(
    intent: IntentFunction,
    *,
    shape_bindings: Dict[str, int],
    profile: RVVHardwareProfile,
    vec_width: Optional[int] = None,
) -> ProgramCostEstimate:
    """
    Estimate cost for reduce/elementwise-heavy kernels (matmul is treated as "unknown" here).

    The model is intentionally coarse but stable:
    - bytes: sum of input/output tensor volumes per op (very rough upper bound)
    - flops: approximate scalar op-count (1 flop per elementwise element; reduce counts input elements)
    - time: roofline max(compute, memory) scaled by a vector-efficiency factor
    """

    lanes = max(1, int(profile.rvv_vlen_bits) // 32)
    vw = int(vec_width) if isinstance(vec_width, int) and vec_width > 0 else lanes
    vw = max(1, min(vw, lanes))

    # Per-core peaks (the generated C is single-threaded today).
    compute_peak_vec = float(profile.fma_units_per_core) * float(lanes) * float(profile.frequency_ghz) * 2.0  # GFLOPs/core @ full lanes
    bw_core = float(profile.mem_bandwidth_gbps) / max(1, int(profile.num_cores))  # GB/s per core (rough)

    # "Equivalent flops" weights per element. This is a pragmatic knob, not physics.
    # - exp/floor currently lower to scalar libm on RVV -> treat as expensive + scalar-lane.
    weights = {
        "add": 1.0,
        "sub": 1.0,
        "mul": 1.0,
        "div": 4.0,
        "max": 1.0,
        "min": 1.0,
        "abs": 1.0,
        "relu": 1.0,
        "rsqrt": 8.0,
        "exp": 25.0,
        "floor": 3.0,
        "cast": 1.0,
        "where": 1.0,
        "reduce_sum": 1.0,
        "reduce_max": 1.0,
        "reduce_any": 0.5,
        "broadcast_in_dim": 0.1,
        "transpose": 0.0,
        "gather": 0.0,
    }

    vec_ops = {
        # Vectorized runtime paths today.
        "add",
        "sub",
        "mul",
        "div",
        "max",
        "min",
        "abs",
        "relu",
        "rsqrt",
        "where",
        "reduce_sum",
        "reduce_max",
        "reduce_any",
        "broadcast_in_dim",
        "cast",  # common u8<->f32 paths are vectorized
        "floor",
    }
    scalar_ops = {"exp"}

    total_flops = 0.0
    total_bytes = 0.0
    total_time_s = 0.0
    notes: List[str] = []
    scalar_hit: Dict[str, int] = {}

    for op in intent.ops:
        if op.op in {"const", "reshape", "identity", "layout_cast"}:
            continue
        if op.op == "matmul":
            notes.append("matmul ignored by generic cost model")
            continue

        out = op.output
        out_n = _resolve_numel(intent, out, shape_bindings)
        if out_n is None or out_n <= 0:
            continue

        out_dt = intent.tensors[out].dtype
        bytes_out = float(out_n * _dtype_bytes(out_dt))
        bytes_in = 0.0

        # Determine input volume (broadcast-aware because tensors have their own shapes).
        for nm in op.inputs:
            in_n = _resolve_numel(intent, nm, shape_bindings)
            if in_n is None or in_n <= 0:
                continue
            bytes_in += float(in_n * _dtype_bytes(intent.tensors[nm].dtype))

        bytes_ = bytes_in + bytes_out

        # Flops-like count.
        if op.op in {"reduce_sum", "reduce_max", "reduce_any"} and op.inputs:
            in_n = _resolve_numel(intent, op.inputs[0], shape_bindings)
            elems = float(in_n) if in_n is not None and in_n > 0 else float(out_n)
        else:
            elems = float(out_n)

        w = float(weights.get(op.op, 0.0))
        flops = elems * w

        # Vectorization factor: scalar ops effectively use 1 lane.
        if op.op in scalar_ops:
            eff_lanes = 1
            scalar_hit[op.op] = scalar_hit.get(op.op, 0) + 1
        elif op.op in vec_ops:
            eff_lanes = vw
        else:
            # Unknown ops: assume scalar to be conservative.
            eff_lanes = 1
            scalar_hit[op.op] = scalar_hit.get(op.op, 0) + 1

        compute_peak = compute_peak_vec * (float(eff_lanes) / float(lanes))
        compute_s = (flops / (compute_peak * 1e9)) if compute_peak > 0 else 0.0
        mem_s = (bytes_ / (bw_core * 1e9)) if bw_core > 0 else 0.0
        op_s = max(compute_s, mem_s)

        total_flops += flops
        total_bytes += bytes_
        total_time_s += op_s

    intensity = (total_flops / total_bytes) if total_bytes > 0 else 0.0
    achieved = (total_flops / (total_time_s * 1e9)) if total_time_s > 0 else 0.0
    ms = total_time_s * 1e3

    notes.append(f"bw_core={bw_core:.3f}GB/s")
    notes.append(f"vec_width={vw} lanes={lanes}")
    notes.append(f"intensity={intensity:.3f} flop/byte")
    if scalar_hit:
        short = ", ".join(f"{k}={v}" for k, v in sorted(scalar_hit.items()))
        notes.append(f"scalar_ops={short}")

    return ProgramCostEstimate(
        gflops=float(achieved),
        ms=float(ms),
        intensity=float(intensity),
        flops=float(total_flops),
        bytes=float(total_bytes),
        notes=tuple(notes),
    )


class GEMMCostModel:
    def __init__(self, profile: RVVHardwareProfile, M: int, N: int, K: int, dtype_size: int = 4):
        self.profile = profile
        self.M = M
        self.N = N
        self.K = K
        self.dtype_size = dtype_size

    def _cache_level(self, tm: int, tn: int, tk: int) -> Tuple[str, int]:
        ws = _working_set_bytes(tm, tn, tk, self.dtype_size)
        if ws < 0.7 * self.profile.l1d_cache_kb * 1024:
            return "L1", self.profile.l1_latency_cycles
        if ws < 0.7 * self.profile.l2_cache_kb * 1024:
            return "L2", self.profile.l2_latency_cycles
        if self.profile.l3_cache_kb and ws < 0.7 * self.profile.l3_cache_kb * 1024:
            return "L3", self.profile.mem_latency_cycles // 2
        return "DRAM", self.profile.mem_latency_cycles

    def _compute_intensity(self, tm: int, tn: int, tk: int) -> float:
        flops = 2.0 * tm * tn * tk
        # bytes per tile (assuming perfect reuse inside tile)
        bytes_moved = (tm * tk + tk * tn + tm * tn) * self.dtype_size
        return flops / bytes_moved if bytes_moved > 0 else 0.0

    def evaluate_tile(self, tm: int, tn: int, tk: int) -> CostEstimate:
        ai = self._compute_intensity(tm, tn, tk)
        cache_level, _ = self._cache_level(tm, tn, tk)

        vector_lanes = max(1, self.profile.rvv_vlen_bits // 32)
        compute_peak = (
            self.profile.num_cores
            * self.profile.fma_units_per_core
            * vector_lanes
            * self.profile.frequency_ghz
            * 2.0
        )  # approximate GFLOPs peak
        # mem_bandwidth_gbps is treated as GB/s; AI is FLOPs/byte -> memory roofline in GFLOPs is GB/s * FLOPs/byte.
        memory_peak = self.profile.mem_bandwidth_gbps * ai
        achievable = min(compute_peak, memory_peak)
        if cache_level == "DRAM":
            achievable *= 0.5
        elif cache_level == "L2":
            achievable *= 0.8
        # vector alignment efficiency
        achievable *= min(1.0, tn / vector_lanes)
        return CostEstimate(gflops=achievable, cache_level=cache_level, intensity=ai)

    def search_best_tile(self, candidates: List[Tuple[int, int, int]]) -> TileChoice:
        # Local import to avoid circular dep at module load time.
        from .tiling_search import TileChoice

        best = None
        best_perf = -1.0
        for tm, tn, tk in candidates:
            est = self.evaluate_tile(tm, tn, tk)
            if est.gflops > best_perf:
                best_perf = est.gflops
                best = (tm, tn, tk, est)
        if best is None:
            best = (16, 16, 16, CostEstimate(0.0, "L2", 0.0))
        tm, tn, tk, est = best
        notes = [f"cache={est.cache_level}", f"ai={est.intensity:.2f}", f"pred_gflops={est.gflops:.2f}"]
        return TileChoice(tile_m=tm, tile_n=tn, tile_k=tk, vec_width=None, notes=notes)


__all__ = ["GEMMCostModel", "CostEstimate", "ProgramCostEstimate", "estimate_program_cost"]
