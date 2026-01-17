"""
E5 freeze-tile baseline helpers.

Motivation (paper RQ3 / performance decoupling):
  - A tile-centric frontend (e.g., Triton) exposes compile-time tiling parameters
    such as BLOCK_M/BLOCK_N/BLOCK_K.
  - On a different target (e.g., RVV CPU), reusing those tiles can be suboptimal.
  - We therefore compare:
      (1) Freeze: reuse frontend tiles as-is
      (2) Retune: re-select tiles on the target via cost model / tuning

This module turns frontend launch meta (constexpr bindings) into a concrete
`ScheduleSketch` that can be fed to the backend without any retuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from intent_ir.ir import IntentFunction, ScheduleSketch
from pipeline.interfaces import KernelDescriptor


@dataclass(frozen=True)
class FrozenSchedule:
    schedule: ScheduleSketch
    notes: list[str]


def _load_constexpr_from_descriptor(desc: KernelDescriptor | Mapping[str, Any] | None) -> Dict[str, int]:
    if desc is None:
        return {}
    launch: Any
    if isinstance(desc, KernelDescriptor):
        launch = desc.launch
    else:
        launch = desc.get("launch") if isinstance(desc, Mapping) else None
    if not isinstance(launch, Mapping):
        return {}
    raw = launch.get("constexpr")
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, int] = {}
    for k, v in raw.items():
        if v is None:
            continue
        try:
            out[str(k).upper()] = int(v)
        except Exception:
            continue
    return out


def _resolve_symbol(v: str | int | None, constexpr: Dict[str, int], *, field: str, notes: list[str]) -> str | int | None:
    if isinstance(v, str):
        key = v.strip()
        if not key:
            return v
        hit = constexpr.get(key.upper())
        if hit is not None:
            notes.append(f"freeze_tile: resolved {field}={key} -> {hit}")
            return int(hit)
        return v
    return v


def resolve_schedule_symbols(schedule: ScheduleSketch, constexpr: Mapping[str, Any] | None) -> Tuple[ScheduleSketch, list[str]]:
    """
    Resolve schedule fields like tile_m="BLOCK_M" using `constexpr` bindings.
    Returns (resolved_schedule, notes).
    """
    notes: list[str] = []
    c = _load_constexpr_from_descriptor({"launch": {"constexpr": constexpr}})  # reuse normalization
    tm = _resolve_symbol(schedule.tile_m, c, field="tile_m", notes=notes)
    tn = _resolve_symbol(schedule.tile_n, c, field="tile_n", notes=notes)
    tk = _resolve_symbol(schedule.tile_k, c, field="tile_k", notes=notes)
    vw = _resolve_symbol(schedule.vec_width, c, field="vec_width", notes=notes)
    pd = _resolve_symbol(schedule.pipeline_depth, c, field="pipeline_depth", notes=notes)
    return (
        ScheduleSketch(
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            vec_width=vw,
            pipeline_depth=pd,
            axis_bindings=dict(schedule.axis_bindings or {}),
            vec_axis=schedule.vec_axis,
            parallel_axes=list(schedule.parallel_axes or []),
            memory_hint=dict(schedule.memory_hint or {}),
        ),
        notes,
    )


def freeze_tile_schedule(intent: IntentFunction, *, desc: KernelDescriptor | Mapping[str, Any] | None = None) -> FrozenSchedule:
    """
    Build a concrete schedule by freezing frontend tile constexpr values.

    This prefers resolving existing schedule symbols (BLOCK_*) and falls back to
    common Triton names when schedule is missing.
    """
    notes: list[str] = []
    constexpr = _load_constexpr_from_descriptor(desc)

    base = intent.schedule or ScheduleSketch()
    resolved, n0 = resolve_schedule_symbols(base, constexpr)
    notes.extend(n0)

    # If the schedule still carries symbolic tiles, try a minimal heuristic from launch meta.
    def _need_int(v: str | int | None) -> bool:
        return not isinstance(v, int)

    if _need_int(resolved.tile_m) and "BLOCK_M" in constexpr:
        resolved.tile_m = int(constexpr["BLOCK_M"])
        notes.append("freeze_tile: inferred tile_m=BLOCK_M from launch constexpr")
    if _need_int(resolved.tile_n) and "BLOCK_N" in constexpr:
        resolved.tile_n = int(constexpr["BLOCK_N"])
        notes.append("freeze_tile: inferred tile_n=BLOCK_N from launch constexpr")
    if _need_int(resolved.tile_k) and "BLOCK_K" in constexpr:
        resolved.tile_k = int(constexpr["BLOCK_K"])
        notes.append("freeze_tile: inferred tile_k=BLOCK_K from launch constexpr")

    return FrozenSchedule(schedule=resolved, notes=notes)


__all__ = ["FrozenSchedule", "freeze_tile_schedule", "resolve_schedule_symbols"]

