"""
IntentIR composition policy for mapping-quality gates.
"""

from __future__ import annotations

from typing import Mapping


COMPLEX_FAMILIES: tuple[str, ...] = (
    "index_scatter_gather",
    "conv_pool_interp",
    "matmul_linear",
    "attention_sequence",
    "reduction",
    "norm_activation",
)

COMPLEX_FAMILY_SINGLE_INTENT_RATIO_TARGETS: Mapping[str, float] = {
    "m1": 0.40,
    "m2": 0.20,
}


def complex_families() -> set[str]:
    return set(COMPLEX_FAMILIES)


def single_intent_ratio_target(stage: str = "m1") -> float:
    key = str(stage or "").strip().lower()
    if key not in COMPLEX_FAMILY_SINGLE_INTENT_RATIO_TARGETS:
        key = "m1"
    return float(COMPLEX_FAMILY_SINGLE_INTENT_RATIO_TARGETS[key])


def evaluate_complex_family_ratio(*, ratio: float, stage: str = "m1") -> dict[str, object]:
    threshold = single_intent_ratio_target(stage)
    ok = float(ratio) <= float(threshold)
    return {
        "stage": str(stage),
        "ok": bool(ok),
        "ratio": float(ratio),
        "threshold": float(threshold),
        "detail": (
            f"complex_family_single_intent_ratio {float(ratio):.4f} <= {float(threshold):.4f}"
            if ok
            else f"complex_family_single_intent_ratio {float(ratio):.4f} > {float(threshold):.4f}"
        ),
    }


__all__ = [
    "COMPLEX_FAMILIES",
    "COMPLEX_FAMILY_SINGLE_INTENT_RATIO_TARGETS",
    "complex_families",
    "single_intent_ratio_target",
    "evaluate_complex_family_ratio",
]

