"""
Case generation for IntentFunction based on constraints.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from intent_ir.ir_types import IntentFunction
from triton_frontend.facts import TTIRConstraints
from typing import Sequence


@dataclass
class TestCase:
    shapes: Dict[str, int]
    dtypes: Dict[str, str] = None
    seed: int = 0
    inputs: Dict[str, np.ndarray] | None = None
    __test__ = False  # prevent pytest from treating this as a test container


EDGE_VALUES = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33]


def generate_cases(
    intent: IntentFunction,
    constraints: Optional[TTIRConstraints] = None,
    limit: int = 10,
    seed: int = 0,
    tile_hints: Sequence[int] | None = None,
    axes: Sequence[str] | None = None,
    exclude_axes: Sequence[str] | None = None,
    extra_sizes: Sequence[int] | None = None,
) -> List[TestCase]:
    """
    Deterministic, constraint-aware case generation.

    Rules:
    - Include small values {1,2,3} for all axes.
    - Use schedule/TTIR tile hints to craft divisible and non-divisible sizes: tile-1, tile, tile+1, 2*tile+1.
    - If needs_mask, ensure at least one non-divisible per axis.
    - Produce Cartesian combinations truncated to `limit`, but preserve diversity order.
    """
    axes_list = list(axes) if axes is not None else _collect_axes(intent)
    if exclude_axes:
        exclude = set(exclude_axes)
        axes_list = [ax for ax in axes_list if ax not in exclude]
    needs_mask = constraints.needs_mask if constraints else False
    tile_list: List[int] = []
    if tile_hints:
        tile_list.extend(int(t) for t in tile_hints if isinstance(t, (int, float)))
    tile_list.extend(_collect_tile_hints(intent))
    if not tile_list:
        tile_list = [16]

    per_axis_vals = []
    extra_set = set(int(x) for x in (extra_sizes or []) if isinstance(x, (int, float)) and int(x) > 0)
    for ax in axes_list:
        vals = set(EDGE_VALUES[:3])  # 1,2,3
        for t in tile_list:
            if t <= 0:
                continue
            vals.update([max(1, t - 1), t, t + 1, 2 * t + 1])
        vals.update(extra_set)
        # needs_mask: prefer non-divisible option first
        ordered = sorted(vals)
        if needs_mask:
            ordered = sorted(ordered, key=lambda v: all(v % t == 0 for t in tile_list if t > 1))
        per_axis_vals.append(ordered)

    cases: List[TestCase] = []
    if not axes_list:
        return [TestCase(shapes={}, dtypes={}, seed=seed)]

    # Deterministic Cartesian product, truncated to limit
    from itertools import product

    for combo in product(*per_axis_vals):
        shapes = {ax: combo[i] for i, ax in enumerate(axes_list)}
        # needs_mask: ensure at least one axis non-divisible by any tile
        if needs_mask:
            if all(all(shapes[ax] % t == 0 for t in tile_list if t > 1) for ax in axes_list):
                continue
        cases.append(TestCase(shapes=shapes, dtypes={}, seed=seed))
        if len(cases) >= limit:
            break
    return cases


def _collect_axes(intent: IntentFunction) -> List[str]:
    axes = set()
    for t in intent.tensors.values():
        for d in t.shape:
            if d.kind == "sym":
                axes.add(d.value)
    return sorted(list(axes))


def _collect_tile_hints(intent: IntentFunction) -> List[int]:
    tiles = []
    if intent.schedule:
        for k in ("tile_m", "tile_n", "tile_k", "vec_width"):
            v = getattr(intent.schedule, k)
            if isinstance(v, int):
                tiles.append(v)
    return tiles or [16]


__all__ = ["TestCase", "generate_cases"]
