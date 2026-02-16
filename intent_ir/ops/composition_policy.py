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

# Complex families where composition is expected for higher-level semantics.
_COMPOSITION_REQUIRED_BY_FAMILY: Mapping[str, set[str]] = {
    "attention_sequence": {
        "scaled_dot_product_attention",
        "scaled_dot_product_attention_forward",
        "flash_attention_forward",
        "scaled_softmax_forward",
        "ScaleDotProductAttention",
    },
    "conv_pool_interp": {
        "upsample_bicubic2d_aa",
        "upsample_nearest1d",
        "upsample_nearest2d",
    },
    "index_scatter_gather": {
        "where_self",
        "where_scalar_self",
        "where_scalar_other",
        "index_put",
        "select_scatter",
        "slice_scatter",
        "masked_scatter",
        "masked_select",
    },
    "matmul_linear": {
        "vdot",
        "dot",
        "bmm",
    },
    "norm_activation": {
        "layer_norm",
        "group_norm",
        "batch_norm",
        "softmax",
        "log_softmax",
        "weight_norm_interface",
    },
    "reduction": {
        "mean",
        "var",
        "std",
        "var_mean",
        "all",
    },
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


def composition_required(
    *,
    semantic_op: str,
    family: str,
    mapping_kind: str,
) -> bool:
    if str(family) not in complex_families():
        return False
    kind = str(mapping_kind or "").strip()
    if kind == "macro_template":
        return True
    required = _COMPOSITION_REQUIRED_BY_FAMILY.get(str(family), set())
    return str(semantic_op) in required


__all__ = [
    "COMPLEX_FAMILIES",
    "COMPLEX_FAMILY_SINGLE_INTENT_RATIO_TARGETS",
    "complex_families",
    "single_intent_ratio_target",
    "evaluate_complex_family_ratio",
    "composition_required",
]
