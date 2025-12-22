"""
MLIR-like pretty printer for Intent-IR v1.1 (pure Python, no MLIR deps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .ir_types import IntentFunction, TensorLayout, TensorType


@dataclass
class MLIRLikePrinterOptions:
    indent: int = 2
    deterministic: bool = True
    include_meta: bool = False
    emit_type_annotations: bool = True
    emit_attrs: bool = True
    prefer_elemwise_sugar: bool = False


def print_mlir_like(intent: IntentFunction, opts: MLIRLikePrinterOptions | None = None) -> str:
    opts = opts or MLIRLikePrinterOptions()
    lines: List[str] = []
    ind = " " * opts.indent

    input_names = _sorted_inputs(intent) if opts.deterministic else list(_inputs(intent))
    output_names = list(intent.outputs)
    ssa_produced: set[str] = set()

    # func signature
    lines.append(f"func @{intent.name}(")
    sig_lines = []
    for name in input_names:
        t = intent.tensors[name]
        sig_lines.append(f"{ind}{name}: {_fmt_tensor_type(t)}")
    lines.append(",\n".join(sig_lines))
    lines.append(") -> (")
    ret_lines = []
    for name in output_names:
        t = intent.tensors[name]
        ret_lines.append(f"{ind}{name}: {_fmt_tensor_type(t)}")
    lines.append(",\n".join(ret_lines))
    lines.append(") {")

    # body ops
    for op in intent.ops:
        op_inputs = [_fmt_value(x, ssa_produced, input_names) for x in op.inputs]
        out_ssa = f"%{op.output}"
        ssa_produced.add(op.output)
        op_line = _fmt_op_line(op.op, out_ssa, op_inputs, op.attrs, intent, opts)
        lines.append(ind + op_line)

    # schedule
    if intent.schedule is not None:
        target = f"%{output_names[0]}"
        lines.append("")
        lines.append(f"{ind}attach_schedule {target} {{")
        for key, val in _iter_schedule_fields(intent, opts.deterministic):
            lines.append(f"{ind*2}{key} = {val}")
        lines.append(f"{ind}}}")

    # return
    ret_vals = ", ".join(f"%{name}" for name in output_names)
    lines.append(f"\n{ind}return {ret_vals}")
    lines.append("}")

    if opts.include_meta and intent.meta:
        lines.append("\n// meta: " + _fmt_dict(intent.meta))

    return "\n".join(lines)


def write_mlir_like(intent: IntentFunction, path: str | Path, opts: MLIRLikePrinterOptions | None = None) -> None:
    text = print_mlir_like(intent, opts)
    Path(path).write_text(text, encoding="utf-8")


def _inputs(intent: IntentFunction) -> Iterable[str]:
    outputs = set(intent.outputs)
    for name in intent.tensors:
        if name not in outputs:
            yield name


def _sorted_inputs(intent: IntentFunction) -> List[str]:
    return sorted(_inputs(intent))


def _fmt_tensor_type(t: TensorType) -> str:
    dims = ",".join(str(d.value) for d in t.shape)
    layout = _fmt_layout(t.layout)
    return f"tensor<{t.dtype},[{dims}],{layout}>"


def _fmt_layout(layout: TensorLayout) -> str:
    if layout.kind in {"row_major", "col_major"} and not layout.params:
        return layout.kind
    if layout.kind == "blocked":
        params = ", ".join(f"{k}={layout.params[k]}" for k in sorted(layout.params))
        return f"blocked({params})"
    if layout.kind == "custom":
        if layout.params:
            params = ", ".join(
                f'{k}="{layout.params[k]}"' if isinstance(layout.params[k], str) else f"{k}={layout.params[k]}"
                for k in sorted(layout.params)
            )
            return f"custom({params})"
        return "custom"
    return layout.kind


def _fmt_value(name: str, produced: set[str], inputs: Iterable[str]) -> str:
    if name in produced:
        return f"%{name}"
    if name in inputs:
        return name
    # default to SSA if it looks like intermediate
    return f"%{name}"


def _fmt_op_line(op: str, out: str, inputs: List[str], attrs: Dict[str, Any], intent: IntentFunction, opts: MLIRLikePrinterOptions) -> str:
    attr_str = ""
    if opts.emit_attrs:
        attr_str = _fmt_attrs(op, attrs)
    if op in {"add", "sub", "mul", "div", "max", "min", "exp", "relu", "rsqrt"}:
        if opts.prefer_elemwise_sugar:
            return f"{out} = intent.{op}({', '.join(inputs)}){attr_str}"
        return f'{out} = intent.elemwise("{op}", {", ".join(inputs)}){attr_str}'
    if op == "identity":
        return f"{out} = intent.identity({', '.join(inputs)}){attr_str}"
    if op == "const":
        val = attrs.get("value")
        return f"{out} = intent.const() {{value={_fmt_value_any(val)}}}"
    if op == "matmul":
        return f"{out} = intent.matmul({', '.join(inputs)}){attr_str}"
    if op == "reduce_sum":
        return f"{out} = intent.reduce_sum({', '.join(inputs)}){attr_str}"
    if op == "softmax":
        return f"{out} = intent.softmax({', '.join(inputs)}){attr_str}"
    if op == "broadcast_in_dim":
        return f"{out} = intent.broadcast_in_dim({', '.join(inputs)}){attr_str}"
    if op == "transpose":
        return f"{out} = intent.transpose({', '.join(inputs)}){attr_str}"
    if op == "reshape":
        return f"{out} = intent.reshape({', '.join(inputs)}){attr_str}"
    if op == "layout_cast":
        return f"{out} = intent.layout_cast({', '.join(inputs)}){attr_str}"
    if op == "conv2d":
        return f"{out} = intent.conv2d({', '.join(inputs)}){attr_str}"
    if op == "custom_call":
        callee = (attrs or {}).get("callee", "")
        callee_s = _fmt_value_any(callee) if callee else "\"\""
        # Keep attrs printing stable; include callee in the call head as well.
        return f"{out} = intent.custom_call({callee_s}, {', '.join(inputs)}){attr_str}"
    return f"{out} = intent.{op}({', '.join(inputs)}){attr_str}"


def _fmt_attrs(op: str, attrs: Dict[str, Any]) -> str:
    if not attrs:
        return ""
    # normalize some common attrs
    normalized: Dict[str, Any] = dict(attrs)
    if op == "reduce_sum":
        if "dims" not in normalized and "axis" in normalized:
            normalized["dims"] = [normalized.pop("axis")]
    return f" {{{_fmt_dict(normalized)}}}"


def _fmt_dict(d: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(d.keys()):
        parts.append(f"{k}={_fmt_value_any(d[k])}")
    return ", ".join(parts)


def _fmt_value_any(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, dict):
        return "{" + _fmt_dict(v) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_value_any(x) for x in v) + "]"
    return str(v)


def _iter_schedule_fields(intent: IntentFunction, deterministic: bool) -> Iterable[tuple[str, str]]:
    s = intent.schedule
    if s is None:
        return []
    fields: List[tuple[str, str]] = []

    def fmt_scalar(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, str):
            return f'sym("{val}")'
        return str(val)

    for key in ("tile_m", "tile_n", "tile_k", "vec_width", "pipeline_depth"):
        val = getattr(s, key)
        if val is not None:
            fields.append((key, fmt_scalar(val)))

    if s.axis_bindings:
        bindings = ", ".join(f"{k}={s.axis_bindings[k]}" for k in sorted(s.axis_bindings))
        fields.append(("axis_bindings", f"{{{bindings}}}"))

    if s.vec_axis is not None:
        fields.append(("vec_axis", s.vec_axis))

    if s.parallel_axes:
        arr = ", ".join(s.parallel_axes)
        fields.append(("parallel_axes", f"[{arr}]"))

    if s.memory_hint:
        fields.append(("memory_hint", _fmt_value_any(_sort_dict(s.memory_hint))))

    return fields


def _sort_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return d  # type: ignore[return-value]
    return {k: _sort_dict(d[k]) for k in sorted(d.keys())}


__all__ = ["MLIRLikePrinterOptions", "print_mlir_like", "write_mlir_like"]
