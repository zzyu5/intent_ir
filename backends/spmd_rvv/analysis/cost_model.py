"""
Analytical cost model (roofline-style) for GEMM tiling (Task6).

This is intentionally simple: estimates GFLOPs for a (tm, tn, tk) tile using
cache-fit heuristics and vector lane alignment. It is not cycle-accurate but
gives a ranking signal for tile search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .hardware_profile import RVVHardwareProfile


@dataclass(frozen=True)
class CostEstimate:
    gflops: float
    cache_level: str
    intensity: float


def _working_set_bytes(tm: int, tn: int, tk: int, dtype_size: int = 4) -> int:
    return (tm * tk + tk * tn + tm * tn) * dtype_size


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


__all__ = ["GEMMCostModel", "CostEstimate"]
