from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ...ir.ir_types import IntentIRValidationError, Op, TensorLayout, TensorType


@dataclass(frozen=True)
class PolySegment:
    # Valid for t < t_max
    t_max: float
    # poly(t) = c0 + c1*t + c2*t^2 + c3*t^3
    coeffs: Tuple[float, float, float, float]


@dataclass(frozen=True)
class PiecewisePoly:
    segments: List[PolySegment]
    outside: float = 0.0


@dataclass(frozen=True)
class IndexPlan:
    center_offset: float = 0.5
    start_offset: float = 0.5
    span_end_offset: float = 0.5
    tap_offset: float = 0.5
    clamp_low: float = 0.0
    support: float = 2.0
    taps: int = 5
    tap_enable: str = "k < span_size"


class LoweringBuilder:
    """
    A small "micro-op builder" that produces primitive IntentIR ops.

    This is the shared, extensible core: new macro ops should reuse these helpers
    rather than reimplementing ad-hoc emit/const/name logic.
    """

    def __init__(self, *, tensors: Dict[str, TensorType], ops_out: List[Op]):
        self.tensors = tensors
        self.ops_out = ops_out
        self.defined: set[str] = set(tensors.keys())
        self.produced: set[str] = set()
        self.default_layout = self._infer_default_layout(tensors)

    @staticmethod
    def _infer_default_layout(tensors: Dict[str, TensorType]) -> TensorLayout:
        for t in tensors.values():
            return t.layout
        raise IntentIRValidationError("macro_lowering: cannot infer default layout")

    def fresh(self, base: str) -> str:
        base = str(base)
        name = base
        i = 0
        while name in self.defined:
            i += 1
            name = f"{base}_{i}"
        self.defined.add(name)
        return name

    def emit(self, op: str, inputs: List[str], output: str, attrs: dict | None = None) -> str:
        # Allow reusing existing names for "final" outputs (macro op declared outputs).
        attrs = {} if attrs is None else dict(attrs)
        self.ops_out.append(Op(op=op, inputs=list(inputs), output=output, attrs=attrs))
        self.defined.add(output)
        self.produced.add(output)
        return output

    def append_existing_op(self, op: Op) -> None:
        self.ops_out.append(op)
        self.defined.add(op.output)
        self.produced.add(op.output)

    def ensure_tensor(self, name: str, *, dtype: str, shape: list, layout: TensorLayout | None = None) -> None:
        if name in self.tensors:
            return
        self.tensors[name] = TensorType(dtype=dtype, shape=shape, layout=layout or self.default_layout)
        self.defined.add(name)

    def const(self, value: Any, *, dtype: str = "f32", name: str | None = None) -> str:
        out = name or self.fresh("c")
        self.emit("const", [], out, {"value": value, "dtype": dtype})
        self.tensors.setdefault(out, TensorType(dtype=dtype, shape=[], layout=self.default_layout))
        return out

    def shape_const(self, sym: str, *, dtype: str = "i32") -> str:
        """
        Emit a scalar const with `attrs.value` equal to the symbol name.

        IMPORTANT: `sym` may already exist in `tensors` (declared scalar) but still not
        have a runtime value. We only skip emission if it has already been produced.
        """
        if sym in self.produced:
            return sym
        self.tensors.setdefault(sym, TensorType(dtype=dtype, shape=[], layout=self.default_layout))
        self.defined.add(sym)
        self.emit("const", [], sym, {"value": sym, "dtype": dtype})
        return sym


def parse_index_plan(impl: Dict[str, Any] | None) -> IndexPlan:
    impl = impl or {}
    ip = impl.get("index_plan")
    if not isinstance(ip, dict):
        return IndexPlan()

    def num(key: str, default: float) -> float:
        v = ip.get(key, default)
        return float(v) if isinstance(v, (int, float)) else float(default)

    def s(key: str, default: str) -> str:
        v = ip.get(key, default)
        return str(v) if isinstance(v, str) else str(default)

    start_off = num("start_offset", 0.5)
    support = num("support", 2.0)
    # Prefer explicit taps, otherwise derive from support when it is close to an integer.
    taps_raw = ip.get("taps")
    taps = None
    if isinstance(taps_raw, int):
        taps = int(taps_raw)
    elif isinstance(taps_raw, float) and float(taps_raw).is_integer():
        taps = int(taps_raw)
    else:
        try:
            si = int(round(float(support)))
            if abs(float(support) - float(si)) <= 1e-6:
                taps = 2 * si + 1
        except Exception:
            taps = None
    if taps is None or taps <= 0:
        taps = 5
    return IndexPlan(
        center_offset=num("center_offset", 0.5),
        start_offset=start_off,
        span_end_offset=num("span_end_offset", 0.5),
        tap_offset=num("tap_offset", start_off),
        clamp_low=num("clamp_low", 0.0),
        support=support,
        taps=int(taps),
        tap_enable=s("tap_enable", "k < span_size"),
    )


def parse_piecewise_poly(impl: Dict[str, Any] | None, *, a_fallback: float = -0.5) -> PiecewisePoly:
    impl = impl or {}
    k = impl.get("kernel")
    if not isinstance(k, dict):
        # Default: keys cubic
        a = float(a_fallback)
        return PiecewisePoly(
            segments=[
                PolySegment(t_max=1.0, coeffs=(1.0, 0.0, -(a + 3.0), (a + 2.0))),
                PolySegment(t_max=2.0, coeffs=((-4.0 * a), (8.0 * a), (-5.0 * a), a)),
            ],
            outside=0.0,
        )

    outside = k.get("outside", 0.0)
    outside_f = float(outside) if isinstance(outside, (int, float)) else 0.0

    segs: List[PolySegment] = []
    raw = k.get("segments")
    if isinstance(raw, list):
        for seg in raw:
            if not isinstance(seg, dict):
                continue
            t_max = seg.get("t_max")
            coeffs = seg.get("coeffs")
            if not isinstance(t_max, (int, float)):
                continue
            if not (isinstance(coeffs, list) and len(coeffs) == 4 and all(isinstance(x, (int, float)) for x in coeffs)):
                continue
            c0, c1, c2, c3 = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3]))
            segs.append(PolySegment(t_max=float(t_max), coeffs=(c0, c1, c2, c3)))
    if not segs:
        a = float(k.get("a", a_fallback)) if isinstance(k.get("a", a_fallback), (int, float)) else float(a_fallback)
        segs = [
            PolySegment(t_max=1.0, coeffs=(1.0, 0.0, -(a + 3.0), (a + 2.0))),
            PolySegment(t_max=2.0, coeffs=((-4.0 * a), (8.0 * a), (-5.0 * a), a)),
        ]
    segs = sorted(segs, key=lambda s: s.t_max)
    return PiecewisePoly(segments=segs, outside=outside_f)


def emit_poly_horner(b: LoweringBuilder, t: str, *, coeffs: Sequence[float], tag: str) -> str:
    if len(coeffs) != 4:
        raise IntentIRValidationError("poly coeffs must have 4 items (c0..c3)")
    c0, c1, c2, c3 = coeffs
    c0v = b.const(c0, dtype="f32", name=b.fresh(f"{tag}_c0"))
    c1v = b.const(c1, dtype="f32", name=b.fresh(f"{tag}_c1"))
    c2v = b.const(c2, dtype="f32", name=b.fresh(f"{tag}_c2"))
    c3v = b.const(c3, dtype="f32", name=b.fresh(f"{tag}_c3"))
    p = b.emit("mul", [c3v, t], b.fresh(f"{tag}_p0"))
    p = b.emit("add", [p, c2v], b.fresh(f"{tag}_p1"))
    p = b.emit("mul", [p, t], b.fresh(f"{tag}_p2"))
    p = b.emit("add", [p, c1v], b.fresh(f"{tag}_p3"))
    p = b.emit("mul", [p, t], b.fresh(f"{tag}_p4"))
    p = b.emit("add", [p, c0v], b.fresh(f"{tag}_p5"))
    return p


def emit_piecewise_poly(b: LoweringBuilder, t: str, poly: PiecewisePoly, *, tag: str) -> str:
    """
    Emit a piecewise cubic polynomial:
      for i: if t < seg[i].t_max -> poly_i(t)
      else -> outside
    """
    outside = b.const(poly.outside, dtype="f32", name=b.fresh(f"{tag}_outside"))
    acc = outside
    # Apply in reverse to make nested where stable.
    for i, seg in enumerate(reversed(poly.segments)):
        seg_tag = f"{tag}_seg{len(poly.segments) - 1 - i}"
        tmax = b.const(seg.t_max, dtype="f32", name=b.fresh(f"{seg_tag}_tmax"))
        val = emit_poly_horner(b, t, coeffs=seg.coeffs, tag=seg_tag)
        lt = b.emit("lt", [t, tmax], b.fresh(f"{seg_tag}_lt"))
        acc = b.emit("where", [lt, val, acc], b.fresh(f"{seg_tag}_pw"))
    return acc


def emit_normalize_weights(b: LoweringBuilder, weights: List[str], *, tag: str, zero_f: str, one_f: str) -> List[str]:
    if not weights:
        raise IntentIRValidationError("normalize_weights expects at least 1 weight")
    total = weights[0]
    for i in range(1, len(weights)):
        total = b.emit("add", [total, weights[i]], b.fresh(f"{tag}_sum_{i}"))
    nz = b.emit("ne", [total, zero_f], b.fresh(f"{tag}_nz"))
    total_safe = b.emit("where", [nz, total, one_f], b.fresh(f"{tag}_tot_safe"))
    out: List[str] = []
    for i, w in enumerate(weights):
        out.append(b.emit("div", [w, total_safe], b.fresh(f"{tag}_w_norm_{i}")))
    return out


__all__ = [
    "LoweringBuilder",
    "IndexPlan",
    "PiecewisePoly",
    "PolySegment",
    "parse_index_plan",
    "parse_piecewise_poly",
    "emit_piecewise_poly",
    "emit_normalize_weights",
]
