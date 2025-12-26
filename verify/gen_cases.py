"""
Case generation for IntentFunction based on constraints.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from intent_ir.ir import IntentFunction
from pipeline.interfaces import FrontendConstraints
from typing import Sequence
import re


@dataclass
class TestCase:
    shapes: Dict[str, int]
    dtypes: Dict[str, str] = None
    seed: int = 0
    inputs: Dict[str, np.ndarray] | None = None
    __test__ = False  # prevent pytest from treating this as a test container


EDGE_VALUES = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33]


@dataclass
class GeneratedCases:
    """
    PR#6: split cases by contract boundary.

    - `in_contract`: must satisfy all assumptions; used for correctness gating.
    - `out_of_contract`: violates exactly one assumption; used only for behavior probing.
    """

    in_contract: List[TestCase]
    out_of_contract: List[TestCase]


_ASSUMP_MOD_EQ0_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*%\s*(\d+)\s*==\s*0\s*$")


def _parse_assumptions(assumptions: Sequence[str] | None) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for a in assumptions or []:
        if not isinstance(a, str):
            continue
        m = _ASSUMP_MOD_EQ0_RE.match(a)
        if not m:
            continue
        axis = m.group(1)
        try:
            mod = int(m.group(2))
        except Exception:
            continue
        if mod > 0:
            out.append((axis, mod))
    return out


def _satisfies_assumptions(shapes: Dict[str, int], assumptions: List[Tuple[str, int]]) -> bool:
    for axis, mod in assumptions:
        if axis not in shapes:
            continue
        try:
            v = int(shapes[axis])
        except Exception:
            return False
        if v % mod != 0:
            return False
    return True


def generate_cases_split(
    intent: IntentFunction,
    constraints: Optional[FrontendConstraints] = None,
    *,
    limit: int = 10,
    seed: int = 0,
    tile_hints: Sequence[int] | None = None,
    axes: Sequence[str] | None = None,
    exclude_axes: Sequence[str] | None = None,
    extra_sizes: Sequence[int] | None = None,
    predicate_clauses: Sequence[str] | None = None,
    assumptions: Sequence[str] | None = None,
    base_shapes: Dict[str, int] | None = None,
) -> GeneratedCases:
    """
    Deterministic case generation with a contract boundary.

    - in_contract: filters cases to satisfy `assumptions` (simple `X % k == 0` today).
    - out_of_contract: for each assumption, emit one case that violates ONLY that assumption.
    """
    parsed_assumps = _parse_assumptions(assumptions)

    # Reuse the existing generator to produce a diverse candidate pool, then split.
    candidates = generate_cases(
        intent,
        constraints=constraints,
        limit=max(50, limit * 8),
        seed=seed,
        tile_hints=tile_hints,
        axes=axes,
        exclude_axes=exclude_axes,
        extra_sizes=extra_sizes,
        predicate_clauses=predicate_clauses,
    )

    in_contract: List[TestCase] = []
    # Always try to include the provided base shape as an in-contract anchor when possible.
    # This prevents empty in-contract sets when assumptions are strict (e.g., M%16==0,N%16==0,K%16==0)
    # but the Cartesian product order hits many small non-divisible shapes first.
    if base_shapes:
        base = dict(base_shapes)
        if _satisfies_assumptions(base, parsed_assumps):
            in_contract.append(TestCase(shapes=base, dtypes={}, seed=seed))
    for c in candidates:
        if _satisfies_assumptions(c.shapes, parsed_assumps):
            in_contract.append(c)
        if len(in_contract) >= limit:
            break

    # Ensure we have at least one in-contract shape to anchor out-of-contract probing.
    anchor_shapes = dict(base_shapes or (in_contract[0].shapes if in_contract else {}))
    if parsed_assumps and anchor_shapes and not _satisfies_assumptions(anchor_shapes, parsed_assumps):
        # Try to find any satisfying candidate to use as anchor.
        for c in candidates:
            if _satisfies_assumptions(c.shapes, parsed_assumps):
                anchor_shapes = dict(c.shapes)
                break

    out_of_contract: List[TestCase] = []
    seen = set(tuple(sorted(c.shapes.items())) for c in in_contract)
    # One violation at a time.
    for axis, mod in parsed_assumps:
        if axis not in anchor_shapes:
            continue
        shapes = dict(anchor_shapes)
        v = int(shapes[axis])
        # Pick a nearby violating value deterministically.
        cand_vals = [max(1, v - 1), v + 1, max(1, mod - 1), mod + 1, 2 * mod + 1]
        bad = None
        for vv in cand_vals:
            if vv % mod != 0:
                bad = vv
                break
        if bad is None:
            continue
        shapes[axis] = int(bad)
        # Must violate exactly this assumption, and satisfy all others.
        ok_other = True
        for ax2, mod2 in parsed_assumps:
            if ax2 == axis:
                continue
            if ax2 in shapes and int(shapes[ax2]) % int(mod2) != 0:
                ok_other = False
                break
        if not ok_other:
            continue
        key = tuple(sorted(shapes.items()))
        if key in seen:
            continue
        seen.add(key)
        out_of_contract.append(TestCase(shapes=shapes, dtypes={}, seed=seed))
        if len(out_of_contract) >= min(limit, len(parsed_assumps)):
            break

    return GeneratedCases(in_contract=in_contract, out_of_contract=out_of_contract)


def generate_cases(
    intent: IntentFunction,
    constraints: Optional[FrontendConstraints] = None,
    limit: int = 10,
    seed: int = 0,
    tile_hints: Sequence[int] | None = None,
    axes: Sequence[str] | None = None,
    exclude_axes: Sequence[str] | None = None,
    extra_sizes: Sequence[int] | None = None,
    predicate_clauses: Sequence[str] | None = None,
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
    needs_mask = bool(constraints.needs_mask) if constraints else False
    if predicate_clauses:
        needs_mask = True
    tile_list: List[int] = []
    if tile_hints:
        tile_list.extend(int(t) for t in tile_hints if isinstance(t, (int, float)))
    tile_list.extend(_collect_tile_hints(intent))
    if not tile_list:
        tile_list = [16]

    per_axis_vals = []
    extra_set = set(int(x) for x in (extra_sizes or []) if isinstance(x, (int, float)) and int(x) > 0)
    if predicate_clauses:
        import re

        num_re = re.compile(r"(-?\d+)")
        for c in predicate_clauses:
            if not isinstance(c, str):
                continue
            for m in num_re.findall(c):
                try:
                    v = int(m)
                except Exception:
                    continue
                if 0 < v <= 2048:
                    extra_set.update({v, max(1, v - 1), v + 1})
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


__all__ = ["TestCase", "GeneratedCases", "generate_cases", "generate_cases_split"]
