"""
Macro op spec helpers.

We keep "semantic macro ops" compact (LLM-friendly) but still rich enough to
support deterministic compiler-like expansion into the primitive IntentIR op set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..ir.ir_types import IntentFunction, IntentIRValidationError


def normalize_upsample_bicubic2d_aa_attrs(attrs: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Normalize/enrich attrs for `upsample_bicubic2d_aa` macro op.

    Goals:
    - Prefer a structured `attrs.impl` (machine-usable).
    - Keep a few flat shortcuts for readability/backward compatibility.
    - Provide enough detail to deterministically expand into primitive ops.
    """
    attrs = {} if attrs is None else dict(attrs)
    impl: Dict[str, Any] = dict(attrs.get("impl") or {}) if isinstance(attrs.get("impl"), dict) else {}
    kernel: Dict[str, Any] = dict(impl.get("kernel") or {}) if isinstance(impl.get("kernel"), dict) else {}
    index_plan: Dict[str, Any] = dict(impl.get("index_plan") or {}) if isinstance(impl.get("index_plan"), dict) else {}
    comp: Dict[str, Any] = dict(impl.get("composition") or {}) if isinstance(impl.get("composition"), dict) else {}

    def pick_number(*cands: Any, default: float) -> float:
        for v in cands:
            if isinstance(v, (int, float)):
                return float(v)
        return float(default)

    def pick_bool(*cands: Any, default: bool) -> bool:
        for v in cands:
            if isinstance(v, bool):
                return bool(v)
        return bool(default)

    def pick_str(*cands: Any, default: str) -> str:
        for v in cands:
            if isinstance(v, str) and v:
                return str(v)
        return str(default)

    # Canonical numeric params.
    a = pick_number(attrs.get("a"), kernel.get("a"), default=-0.5)
    support = pick_number(attrs.get("support"), index_plan.get("support"), default=2.0)
    invscale = pick_number(attrs.get("invscale"), kernel.get("invscale"), default=1.0)
    other_value = pick_number(attrs.get("other_value"), comp.get("other_value"), default=0.0)

    # Canonical composition.
    separable = pick_bool(attrs.get("separable"), comp.get("separable"), default=True)
    compute_order = pick_str(attrs.get("compute_order"), comp.get("compute_order"), default="x_then_y")
    normalize_weights = pick_bool(attrs.get("normalize_weights"), comp.get("normalize_weights"), default=True)
    mask_policy = pick_str(attrs.get("mask_policy"), comp.get("mask_policy"), default="mask_y(dy) & mask_x(dx)")

    # Canonical index plan (rounding/clamp policies).
    center_offset = pick_number(index_plan.get("center_offset"), default=0.5)
    start_offset = pick_number(index_plan.get("start_offset"), default=0.5)
    span_end_offset = pick_number(index_plan.get("span_end_offset"), default=0.5)
    tap_offset = pick_number(index_plan.get("tap_offset"), default=start_offset)
    clamp_low = pick_number(index_plan.get("clamp_low"), default=0.0)
    tap_enable = pick_str(index_plan.get("tap_enable"), default="k < span_size")
    taps = None
    taps_raw = index_plan.get("taps")
    if isinstance(taps_raw, int):
        taps = int(taps_raw)
    elif isinstance(taps_raw, float) and float(taps_raw).is_integer():
        taps = int(taps_raw)
    else:
        # Derive taps from support if it is close to an integer (common stencil pattern).
        try:
            si = int(round(float(support)))
            if abs(float(support) - float(si)) <= 1e-6:
                taps = 2 * si + 1
        except Exception:
            taps = None
    if taps is None or taps <= 0:
        taps = 5

    kernel_name = pick_str(attrs.get("kernel"), kernel.get("name"), default="keys_cubic")
    kernel["name"] = kernel_name
    kernel["a"] = a
    kernel["invscale"] = invscale

    # Provide an explicit numeric piecewise poly spec for keys_cubic with parameter `a`.
    # Each segment is `poly(t)=c0 + c1*t + c2*t^2 + c3*t^3` over a range, with outside=0.
    # This is a compact, machine-usable representation for deterministic lowering.
    segments: List[Dict[str, Any]] = []
    raw_segments = kernel.get("segments")
    if isinstance(raw_segments, list):
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            t_max = seg.get("t_max")
            coeffs = seg.get("coeffs")
            if isinstance(t_max, (int, float)) and isinstance(coeffs, list) and len(coeffs) == 4 and all(
                isinstance(x, (int, float)) for x in coeffs
            ):
                segments.append({"t_max": float(t_max), "coeffs": [float(x) for x in coeffs]})
    if not segments and kernel_name == "keys_cubic":
        # segment1 (t < 1): (a+2)*t^3 - (a+3)*t^2 + 1
        segments.append({"t_max": 1.0, "coeffs": [1.0, 0.0, -(a + 3.0), (a + 2.0)]})
        # segment2 (1 <= t < 2): a*(t^3 - 5*t^2 + 8*t - 4)
        segments.append({"t_max": 2.0, "coeffs": [(-4.0 * a), (8.0 * a), (-5.0 * a), a]})
    if segments:
        segments = sorted(segments, key=lambda s: float(s.get("t_max", 0.0)))
        kernel["segments"] = segments
    kernel.setdefault("outside", 0.0)

    # Hoist/reuse list.
    hoist = impl.get("hoist")
    if not (isinstance(hoist, list) and all(isinstance(x, str) for x in hoist)):
        hoist = attrs.get("hoist")
    if not (isinstance(hoist, list) and all(isinstance(x, str) for x in hoist)):
        hoist = ["span_start", "span_size", "weight_x", "weight_y", "mask_x", "mask_y"]

    index_plan = dict(index_plan)
    index_plan.update(
        {
            "center_offset": center_offset,
            "support": support,
            "start_offset": start_offset,
            "span_end_offset": span_end_offset,
            "tap_offset": tap_offset,
            "clamp_low": clamp_low,
            "taps": int(taps),
            "tap_enable": tap_enable,
        }
    )
    comp = dict(comp)
    comp.update(
        {
            "separable": separable,
            "compute_order": compute_order,
            "normalize_weights": normalize_weights,
            "other_value": other_value,
            "mask_policy": mask_policy,
        }
    )
    impl = dict(impl)
    impl.update({"kernel": kernel, "index_plan": index_plan, "composition": comp, "hoist": hoist})

    attrs["impl"] = impl

    # Keep flat shortcuts (for readability + existing code paths).
    attrs.setdefault("a", a)
    attrs.setdefault("support", support)
    attrs.setdefault("invscale", invscale)
    attrs.setdefault("kernel", kernel_name)
    attrs.setdefault("separable", separable)
    attrs.setdefault("compute_order", compute_order)
    attrs.setdefault("normalize_weights", normalize_weights)
    attrs.setdefault("other_value", other_value)
    attrs.setdefault("mask_policy", mask_policy)
    attrs.setdefault("hoist", hoist)

    # Optional human-readable strings (best-effort; do not overwrite if provided).
    attrs.setdefault("center_formula", "(o+center_offset)*reciprocal_scale")
    attrs.setdefault("start_formula", "floor(center-support+start_offset) clamped to >=clamp_low")
    attrs.setdefault("span_size_formula", "min(center+support+span_end_offset, I_dim) - span_start")
    attrs.setdefault("tap_enable_policy", tap_enable)
    attrs.setdefault("abs_arg", "abs((k+start_minus_center+tap_offset)*invscale)")
    attrs.setdefault("piecewise", "t<1 ? poly1 : (t<2 ? poly2 : 0)")
    attrs.setdefault("reuse", "weight_x reused across dy; weight_y reused across dx-accums")
    return attrs


def enrich_intent_macros(intent: IntentFunction) -> IntentFunction:
    """
    Enrich macro ops in-place (keeps IR more compiler-like and debuggable).
    """
    changed = False
    for op in intent.ops:
        if op.op == "upsample_bicubic2d_aa":
            op.attrs = normalize_upsample_bicubic2d_aa_attrs(op.attrs)
            changed = True
    if changed:
        intent.validate()
    return intent


__all__ = ["normalize_upsample_bicubic2d_aa_attrs", "enrich_intent_macros"]
