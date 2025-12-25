from __future__ import annotations

from typing import Any, Dict, List

from ...ir.ir_types import IntentIRValidationError, Op
from ..macro_spec import normalize_upsample_bicubic2d_aa_attrs
from .common import (
    LoweringBuilder,
    emit_normalize_weights,
    emit_piecewise_poly,
    parse_index_plan,
    parse_piecewise_poly,
)


def lower_upsample_bicubic2d_aa(b: LoweringBuilder, op: Op) -> None:
    """
    Lower `upsample_bicubic2d_aa` into primitive ops.

    This lowering is intentionally structured (index plan -> weights/masks -> gathers -> accum),
    and relies on the macro spec in attrs.impl for reproducibility and extensibility.
    """
    if len(op.inputs) != 1:
        raise IntentIRValidationError("macro_lowering: upsample_bicubic2d_aa requires exactly 1 input")
    inp = op.inputs[0]
    out = op.output

    attrs = normalize_upsample_bicubic2d_aa_attrs(op.attrs)
    impl = attrs.get("impl") if isinstance(attrs.get("impl"), dict) else {}
    comp = impl.get("composition") if isinstance(impl, dict) and isinstance(impl.get("composition"), dict) else {}

    # Shape symbols as scalar consts for index math.
    N = b.shape_const("N", dtype="i32")
    C = b.shape_const("C", dtype="i32")
    IH = b.shape_const("IH", dtype="i32")
    IW = b.shape_const("IW", dtype="i32")
    OH = b.shape_const("OH", dtype="i32")
    OW = b.shape_const("OW", dtype="i32")

    # Scalars from kernel signature: reciprocal_scale_h, reciprocal_scale_w
    rs_h = "reciprocal_scale_h"
    rs_w = "reciprocal_scale_w"
    b.ensure_tensor(rs_h, dtype="f32", shape=[])
    b.ensure_tensor(rs_w, dtype="f32", shape=[])

    # Common constants.
    c0f = b.const(0.0, dtype="f32", name=b.fresh("c0"))
    c1f = b.const(1.0, dtype="f32", name=b.fresh("c1"))
    z_i32 = b.const(0, dtype="i32", name=b.fresh("z_i32"))

    # Parse spec blocks.
    a_param = float(attrs.get("a", -0.5)) if isinstance(attrs.get("a", -0.5), (int, float)) else -0.5
    index_plan = parse_index_plan(impl if isinstance(impl, dict) else None)
    poly = parse_piecewise_poly(impl if isinstance(impl, dict) else None, a_fallback=a_param)

    # Tap count is part of the implementation spec (derived from support when possible).
    # We still *unroll* it into primitive ops (compiler-like), rather than executing a runtime loop.
    taps = int(getattr(index_plan, "taps", 5))
    if taps <= 0:
        taps = 5
    if taps > 33:
        # Prevent pathological IR explosion; keep this conservative for now.
        raise IntentIRValidationError(f"macro_lowering: taps too large for upsample_bicubic2d_aa: {taps}")

    separable = bool(attrs.get("separable", comp.get("separable", True)))
    normalize_taps = bool(attrs.get("normalize_weights", comp.get("normalize_weights", True)))
    other_value = float(attrs.get("other_value", comp.get("other_value", 0.0)))

    support = b.const(index_plan.support, dtype="f32", name=b.fresh("support"))
    invscale = b.const(float(attrs.get("invscale", 1.0)), dtype="f32", name=b.fresh("invscale"))
    c_center_off = b.const(index_plan.center_offset, dtype="f32", name=b.fresh("c_center_off"))
    c_start_off = b.const(index_plan.start_offset, dtype="f32", name=b.fresh("c_start_off"))
    c_span_end_off = b.const(index_plan.span_end_offset, dtype="f32", name=b.fresh("c_span_end_off"))
    c_tap_off = b.const(index_plan.tap_offset, dtype="f32", name=b.fresh("c_tap_off"))
    c_clamp_low = b.const(index_plan.clamp_low, dtype="f32", name=b.fresh("c_clamp_low"))
    other_v = b.const(other_value, dtype="f32", name=b.fresh("other"))

    # Strategy (compiler-like):
    # - Build "stencil plan" in 2D output space [OH,OW]
    # - Broadcast across [N,C] by relying on numpy-style broadcasting semantics.
    grid4_shape = ["N", "C", "OH", "OW"]
    n_i = b.emit("iota", [], b.fresh("n_i"), {"shape": grid4_shape, "axis": 0, "dtype": "i32"})
    c_i = b.emit("iota", [], b.fresh("c_i"), {"shape": grid4_shape, "axis": 1, "dtype": "i32"})

    grid2_shape = ["OH", "OW"]
    oh_2d = b.emit("iota", [], b.fresh("oh_2d"), {"shape": grid2_shape, "axis": 0, "dtype": "i32"})
    ow_2d = b.emit("iota", [], b.fresh("ow_2d"), {"shape": grid2_shape, "axis": 1, "dtype": "i32"})

    # center = (o + center_offset) * reciprocal_scale
    oh_f = b.emit("cast", [oh_2d], b.fresh("oh_f"), {"to": "f32"})
    ow_f = b.emit("cast", [ow_2d], b.fresh("ow_f"), {"to": "f32"})
    center_h = b.emit("mul", [b.emit("add", [oh_f, c_center_off], b.fresh("oh_p")), rs_h], b.fresh("center_h"))
    center_w = b.emit("mul", [b.emit("add", [ow_f, c_center_off], b.fresh("ow_p")), rs_w], b.fresh("center_w"))

    # span_start = max(center - support + start_offset, clamp_low).to i32
    tmp_h = b.emit("sub", [center_h, support], b.fresh("tmp_h"))
    tmp_h = b.emit("add", [tmp_h, c_start_off], b.fresh("tmp_h2"))
    tmp_h = b.emit("max", [tmp_h, c_clamp_low], b.fresh("tmp_h3"))
    span_start_h = b.emit("cast", [tmp_h], b.fresh("span_start_h"), {"to": "i32"})

    tmp_w = b.emit("sub", [center_w, support], b.fresh("tmp_w"))
    tmp_w = b.emit("add", [tmp_w, c_start_off], b.fresh("tmp_w2"))
    tmp_w = b.emit("max", [tmp_w, c_clamp_low], b.fresh("tmp_w3"))
    span_start_w = b.emit("cast", [tmp_w], b.fresh("span_start_w"), {"to": "i32"})

    # span_size = (min(center + support + span_end_offset, I_dim) - span_start).to i32
    IH_f = b.emit("cast", [IH], b.fresh("IH_f"), {"to": "f32"})
    IW_f = b.emit("cast", [IW], b.fresh("IW_f"), {"to": "f32"})

    tmp_h2 = b.emit("add", [b.emit("add", [center_h, support], b.fresh("tmp_h4")), c_span_end_off], b.fresh("tmp_h5"))
    tmp_h2 = b.emit("min", [tmp_h2, IH_f], b.fresh("tmp_h6"))
    ssh_f = b.emit("cast", [span_start_h], b.fresh("ssh_f"), {"to": "f32"})
    span_size_h = b.emit("cast", [b.emit("sub", [tmp_h2, ssh_f], b.fresh("span_size_h_f"))], b.fresh("span_size_h"), {"to": "i32"})

    tmp_w2 = b.emit("add", [b.emit("add", [center_w, support], b.fresh("tmp_w4")), c_span_end_off], b.fresh("tmp_w5"))
    tmp_w2 = b.emit("min", [tmp_w2, IW_f], b.fresh("tmp_w6"))
    ssw_f = b.emit("cast", [span_start_w], b.fresh("ssw_f"), {"to": "f32"})
    span_size_w = b.emit("cast", [b.emit("sub", [tmp_w2, ssw_f], b.fresh("span_size_w_f"))], b.fresh("span_size_w"), {"to": "i32"})

    # start_minus_center = span_start - center
    start_minus_center_h = b.emit("sub", [ssh_f, center_h], b.fresh("start_minus_center_h"))
    start_minus_center_w = b.emit("sub", [ssw_f, center_w], b.fresh("start_minus_center_w"))

    def make_weight_and_mask(axis: str, k: int):
        if axis == "y":
            span_start = span_start_h
            span_size = span_size_h
            start_minus_center = start_minus_center_h
            dim_i32 = IH
        else:
            span_start = span_start_w
            span_size = span_size_w
            start_minus_center = start_minus_center_w
            dim_i32 = IW

        k_i32 = b.const(k, dtype="i32", name=b.fresh(f"k{axis}{k}_i32"))
        k_f = b.emit("cast", [k_i32], b.fresh(f"k{axis}{k}_f"), {"to": "f32"})
        t = b.emit("add", [k_f, start_minus_center], b.fresh(f"t{axis}{k}_0"))
        t = b.emit("add", [t, c_tap_off], b.fresh(f"t{axis}{k}_1"))
        t = b.emit("mul", [t, invscale], b.fresh(f"t{axis}{k}_2"))
        t = b.emit("abs", [t], b.fresh(f"t{axis}{k}_abs"))

        w = emit_piecewise_poly(b, t, poly, tag=f"w_{axis}{k}")

        # gate by k < span_size
        k_lt_span = b.emit("lt", [k_i32, span_size], b.fresh(f"klt_{axis}{k}"))
        w = b.emit("where", [k_lt_span, w, c0f], b.fresh(f"w_{axis}{k}_gated"))

        idx = b.emit("add", [span_start, k_i32], b.fresh(f"idx_{axis}{k}"))
        mask = b.emit("lt", [idx, dim_i32], b.fresh(f"mask_{axis}{k}"))
        return w, mask, idx

    wy: List[str] = []
    my: List[str] = []
    iy: List[str] = []
    wx: List[str] = []
    mx: List[str] = []
    ix: List[str] = []
    for k in range(taps):
        w, m, idx = make_weight_and_mask("y", k)
        wy.append(w)
        my.append(m)
        iy.append(idx)
    for k in range(taps):
        w, m, idx = make_weight_and_mask("x", k)
        wx.append(w)
        mx.append(m)
        ix.append(idx)

    if normalize_taps:
        wy_n = emit_normalize_weights(b, wy, tag="wy", zero_f=c0f, one_f=c1f)
        wx_n = emit_normalize_weights(b, wx, tag="wx", zero_f=c0f, one_f=c1f)
    else:
        wy_n = wy
        wx_n = wx

    acc = b.const(0.0, dtype="f32", name=b.fresh("acc0"))
    if separable:
        ix_safe2_list: List[str] = []
        for kx in range(taps):
            ix_safe2_list.append(b.emit("where", [mx[kx], ix[kx], z_i32], b.fresh(f"ix_s2_{kx}")))
        for ky in range(taps):
            iy_s2 = b.emit("where", [my[ky], iy[ky], z_i32], b.fresh(f"iy_s2_{ky}"))
            row = b.const(0.0, dtype="f32", name=b.fresh(f"row_{ky}_0"))
            for kx in range(taps):
                mask2 = b.emit("and", [my[ky], mx[kx]], b.fresh(f"m2_{ky}_{kx}"))
                val = b.emit("gather", [inp, n_i, c_i, iy_s2, ix_safe2_list[kx]], b.fresh(f"g_{ky}_{kx}"))
                val = b.emit("where", [mask2, val, other_v], b.fresh(f"gv_{ky}_{kx}"))
                contrib = b.emit("mul", [val, wx_n[kx]], b.fresh(f"cx_{ky}_{kx}"))
                row = b.emit("add", [row, contrib], b.fresh(f"row_{ky}_{kx}"))
            row = b.emit("mul", [row, wy_n[ky]], b.fresh(f"rowy_{ky}"))
            acc = b.emit("add", [acc, row], b.fresh(f"acc_y_{ky}"))
    else:
        for ky in range(taps):
            for kx in range(taps):
                iy_s2 = b.emit("where", [my[ky], iy[ky], z_i32], b.fresh(f"iy_s2_{ky}"))
                ix_s2 = b.emit("where", [mx[kx], ix[kx], z_i32], b.fresh(f"ix_s2_{kx}"))
                mask = b.emit("and", [my[ky], mx[kx]], b.fresh(f"m_{ky}_{kx}"))
                val = b.emit("gather", [inp, n_i, c_i, iy_s2, ix_s2], b.fresh(f"g_{ky}_{kx}"))
                val = b.emit("where", [mask, val, other_v], b.fresh(f"gv_{ky}_{kx}"))
                wtap = b.emit("mul", [wy_n[ky], wx_n[kx]], b.fresh(f"w_{ky}_{kx}"))
                contrib = b.emit("mul", [val, wtap], b.fresh(f"c_{ky}_{kx}"))
                acc = b.emit("add", [acc, contrib], b.fresh(f"acc_{ky}_{kx}"))

    # Final write: bind to macro's declared output name.
    b.emit("identity", [acc], out)


__all__ = ["lower_upsample_bicubic2d_aa"]
