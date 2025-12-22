"""
Hardware profile helpers for RVV devices (Task6).

We keep this lightweight: profiles can be loaded from JSON configs instead of
probing hardware directly. Probing remote hosts is left to device_query.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RVVHardwareProfile:
    # Base config
    num_cores: int
    rvv_vlen_bits: int
    frequency_ghz: float
    mem_bandwidth_gbps: float

    # Caches
    l1d_cache_kb: int = 32
    l2_cache_kb: int = 512
    l3_cache_kb: Optional[int] = None
    cache_line_bytes: int = 64

    # Latencies (cycles)
    l1_latency_cycles: int = 4
    l2_latency_cycles: int = 12
    mem_latency_cycles: int = 100

    # Compute
    fma_units_per_core: int = 2
    lmul_max: int = 8
    vtype_switch_penalty: int = 2


def load_profile_from_json(path: str | Path) -> RVVHardwareProfile:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return RVVHardwareProfile(**data)


__all__ = ["RVVHardwareProfile", "load_profile_from_json"]
