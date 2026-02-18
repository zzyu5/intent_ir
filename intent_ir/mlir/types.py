from __future__ import annotations

from typing import Any

from intent_ir.ir.ir_types import Dim, TensorLayout, TensorType


def dim_to_mlir(dim: Dim) -> str:
    if dim.kind == "const":
        return str(int(dim.value))
    return f"?{str(dim.value)}"


def tensor_type_to_mlir(tp: TensorType) -> str:
    shape = "x".join(dim_to_mlir(d) for d in list(tp.shape or []))
    if not shape:
        shape = "1"
    layout = layout_to_attr(tp.layout)
    return f"!intent.tensor<{shape}x{tp.dtype}>{{layout={layout}}}"


def layout_to_attr(layout: TensorLayout) -> str:
    kind = str(layout.kind)
    params = dict(layout.params or {})
    if not params:
        return f"\"{kind}\""
    items = ", ".join(f"{k}:{repr(v)}" for k, v in sorted(params.items(), key=lambda kv: kv[0]))
    return f"\"{kind}({items})\""


def attrs_to_mlir_dict(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "{}"
    items = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(attrs.items(), key=lambda kv: kv[0]))
    return "{" + items + "}"


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f"\"{v}\""
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        items = ", ".join(f"{k}:{_fmt(vv)}" for k, vv in sorted(v.items(), key=lambda kv: kv[0]))
        return "{" + items + "}"
    return str(v)

