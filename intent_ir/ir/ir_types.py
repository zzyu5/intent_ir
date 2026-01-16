"""
Intent-IR v1.1 core data structures, JSON round-trip, and validation.

This file defines the canonical Python AST (dataclasses) for Intent-IR and
provides helpers to parse from/to JSON plus semantic validation. It covers
the v1.1 patches: axis_roles, broadcast/shape/layout ops, matmul epilogue,
and extended schedule fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Literal, Optional

from intent_ir.diagnostics import closest_match, format_op_snippet
from intent_ir.ops import SUPPORTED_OPS


__all__ = [
    "IntentIRValidationError",
    "Dim",
    "TensorLayout",
    "TensorType",
    "Op",
    "ScheduleSketch",
    "IntentFunction",
    "parse_dim",
    "parse_layout",
]


class IntentIRValidationError(Exception):
    """Raised when Intent-IR validation fails."""


AllowedDType = Literal[
    "f16",
    "bf16",
    "f32",
    "f64",
    "i8",
    "u8",
    "i1",
    "i32",
    "i64",
    "bool",
]

SUPPORTED_DTYPES: set[str] = {
    "f16",
    "bf16",
    "f32",
    "f64",
    "i8",
    "u8",
    "i1",
    "i32",
    "i64",
    "bool",
}

AXIS_ROLE_VALUES = {"spatial", "reduction", "batch", "channel"}

EPILOGUE_ELEMWISE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "exp",
    "max",
    "min",
    "ne",
    "sigmoid",
    "tanh",
    "gelu",
    "clip",
    "identity",
    "cast",
    "rsqrt",
}


@dataclass(frozen=True)
class Dim:
    kind: Literal["sym", "const"]
    value: str | int


@dataclass(frozen=True)
class TensorLayout:
    kind: Literal["row_major", "col_major", "blocked", "custom"]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorType:
    dtype: AllowedDType
    shape: List[Dim]
    layout: TensorLayout


@dataclass
class Op:
    op: str
    inputs: List[str]
    output: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleSketch:
    tile_m: str | int | None = None
    tile_n: str | int | None = None
    tile_k: str | int | None = None
    vec_width: str | int | None = None
    pipeline_depth: str | int | None = None
    axis_bindings: Dict[str, str] = field(default_factory=dict)
    vec_axis: Optional[str] = None
    parallel_axes: List[str] = field(default_factory=list)
    memory_hint: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentFunction:
    name: str
    tensors: Dict[str, TensorType]
    ops: List[Op]
    outputs: List[str]
    parallel_axes: List[str] = field(default_factory=list)
    schedule: Optional[ScheduleSketch] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    axis_roles: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "IntentFunction":
        name = data.get("name") or data.get("kernel_type") or "intent_fn"
        tensors_json = data.get("tensors") or {}
        if not isinstance(tensors_json, dict):
            raise IntentIRValidationError("tensors must be an object")
        tensors = {k: _tensor_from_json(k, v) for k, v in tensors_json.items()}

        ops_json = data.get("ops") or []
        if not isinstance(ops_json, list):
            raise IntentIRValidationError("ops must be a list")
        ops = [_op_from_json(o) for o in ops_json]

        outputs = data.get("outputs")
        if outputs is None:
            outputs = [ops[-1].output] if ops else []
        if not isinstance(outputs, list):
            raise IntentIRValidationError("outputs must be a list if provided")

        schedule = None
        if data.get("schedule") is not None:
            schedule = _schedule_from_json(data["schedule"])

        parallel_axes = data.get("parallel_axes") or []
        if schedule and schedule.parallel_axes:
            parallel_axes = schedule.parallel_axes

        axis_roles = data.get("axis_roles") or {}
        meta = data.get("meta") or {}
        inst = cls(
            name=name,
            tensors=tensors,
            ops=ops,
            outputs=outputs,
            parallel_axes=parallel_axes,
            schedule=schedule,
            meta=meta,
            axis_roles=axis_roles,
        )
        inst.validate()
        return inst

    def to_json_dict(self) -> Dict[str, Any]:
        tensors_json = {k: _tensor_to_json(v) for k, v in self.tensors.items()}
        ops_json = [_op_to_json(o) for o in self.ops]
        result: Dict[str, Any] = {
            "name": self.name,
            "tensors": tensors_json,
            "ops": ops_json,
            "outputs": list(self.outputs),
            "parallel_axes": list(self.parallel_axes),
        }
        if self.schedule is not None:
            result["schedule"] = _schedule_to_json(self.schedule)
        if self.axis_roles:
            result["axis_roles"] = dict(self.axis_roles)
        if self.meta:
            result["meta"] = dict(self.meta)
        return result

    def validate(self) -> None:
        _validate_tensors(self.tensors)
        _validate_axis_roles(self.axis_roles, self.tensors, self.parallel_axes)
        _validate_ops(self.ops, self.tensors)
        _validate_outputs(self.outputs, self.tensors, self.ops)
        _validate_parallel_axes(self.parallel_axes, self.tensors)
        if self.schedule:
            _validate_schedule(self.schedule)
            if self.schedule.parallel_axes and self.schedule.parallel_axes != self.parallel_axes:
                raise IntentIRValidationError(
                    "schedule.parallel_axes must match function.parallel_axes when both are set"
                )


def parse_dim(x: int | str) -> Dim:
    if isinstance(x, int):
        return Dim(kind="const", value=x)
    if isinstance(x, str):
        return Dim(kind="sym", value=x)
    raise IntentIRValidationError(f"invalid dimension type: {type(x)}")


def parse_layout(x: str | Dict[str, Any]) -> TensorLayout:
    if isinstance(x, str):
        # LLMs sometimes emit "scalar" for 0-rank tensors. Layout is irrelevant for scalars;
        # treat it as row_major for compatibility.
        if x == "scalar":
            return TensorLayout(kind="row_major", params={})
        if x not in {"row_major", "col_major"}:
            raise IntentIRValidationError(f"unsupported layout: {x}")
        return TensorLayout(kind=x, params={})
    if not isinstance(x, dict):
        raise IntentIRValidationError("layout must be string or object")
    kind = x.get("kind")
    params = x.get("params", {})
    if kind not in {"row_major", "col_major", "blocked", "custom"}:
        raise IntentIRValidationError(f"unsupported layout kind: {kind}")
    if not isinstance(params, dict):
        raise IntentIRValidationError("layout.params must be an object")
    return TensorLayout(kind=kind, params=params)


def _tensor_from_json(name: str, data: Dict[str, Any]) -> TensorType:
    if not isinstance(data, dict):
        raise IntentIRValidationError(f"tensor {name} must be an object")
    dtype = data.get("dtype")
    if dtype not in SUPPORTED_DTYPES:
        raise IntentIRValidationError(f"tensors.{name}.dtype unsupported: {dtype}")
    shape_raw = data.get("shape")
    if not isinstance(shape_raw, list):
        raise IntentIRValidationError(f"tensors.{name}.shape must be a list")
    shape = [parse_dim(d) for d in shape_raw]
    layout_raw = data.get("layout", "row_major")
    layout = parse_layout(layout_raw)
    return TensorType(dtype=dtype, shape=shape, layout=layout)


def _tensor_to_json(t: TensorType) -> Dict[str, Any]:
    return {
        "dtype": t.dtype,
        "shape": [d.value for d in t.shape],
        "layout": _layout_to_json(t.layout),
    }


def _layout_to_json(layout: TensorLayout) -> Any:
    if layout.kind in {"row_major", "col_major"} and not layout.params:
        return layout.kind
    return {"kind": layout.kind, "params": dict(layout.params)}


def _op_from_json(data: Dict[str, Any]) -> Op:
    if not isinstance(data, dict):
        raise IntentIRValidationError("each op must be an object")
    op = data.get("op")
    inputs = data.get("inputs") or []
    output = data.get("output")
    attrs = data.get("attrs", {})
    if not isinstance(op, str):
        raise IntentIRValidationError("op.op must be a string")
    if not isinstance(inputs, list):
        raise IntentIRValidationError(f"op.inputs must be a list for op {op}")
    if not isinstance(output, str):
        raise IntentIRValidationError(f"op.output must be a string for op {op}")
    if attrs is None:
        attrs = {}
    if not isinstance(attrs, dict):
        raise IntentIRValidationError(f"op.attrs must be an object for op {op}")
    return Op(op=op, inputs=inputs, output=output, attrs=attrs)


def _op_to_json(op: Op) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "op": op.op,
        "inputs": list(op.inputs),
        "output": op.output,
    }
    if op.attrs:
        data["attrs"] = op.attrs
    return data


def _schedule_from_json(data: Dict[str, Any]) -> ScheduleSketch:
    if not isinstance(data, dict):
        raise IntentIRValidationError("schedule must be an object")
    axis_bindings = data.get("axis_bindings") or {}
    memory_hint = data.get("memory_hint") or {}
    parallel_axes = data.get("parallel_axes") or []
    return ScheduleSketch(
        tile_m=data.get("tile_m"),
        tile_n=data.get("tile_n"),
        tile_k=data.get("tile_k"),
        vec_width=data.get("vec_width"),
        pipeline_depth=data.get("pipeline_depth"),
        axis_bindings=axis_bindings,
        vec_axis=data.get("vec_axis"),
        parallel_axes=parallel_axes,
        memory_hint=memory_hint,
    )


def _schedule_to_json(s: ScheduleSketch) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "tile_m": s.tile_m,
        "tile_n": s.tile_n,
        "tile_k": s.tile_k,
        "vec_width": s.vec_width,
        "pipeline_depth": s.pipeline_depth,
        "axis_bindings": dict(s.axis_bindings),
        "vec_axis": s.vec_axis,
        "parallel_axes": list(s.parallel_axes),
        "memory_hint": dict(s.memory_hint),
    }
    return data


def _validate_tensors(tensors: Dict[str, TensorType]) -> None:
    for name, t in tensors.items():
        if t.dtype not in SUPPORTED_DTYPES:
            raise IntentIRValidationError(f"tensors.{name}.dtype unsupported: {t.dtype}")
        for idx, d in enumerate(t.shape):
            if not isinstance(d, Dim):
                raise IntentIRValidationError(f"tensors.{name}.shape[{idx}] is not a Dim")
            if d.kind == "const" and not isinstance(d.value, int):
                raise IntentIRValidationError(f"tensors.{name}.shape[{idx}] const must be int")
            if d.kind == "sym" and not isinstance(d.value, str):
                raise IntentIRValidationError(f"tensors.{name}.shape[{idx}] sym must be str")
        layout = t.layout
        if layout.kind not in {"row_major", "col_major", "blocked", "custom"}:
            raise IntentIRValidationError(f"tensors.{name}.layout unsupported: {layout.kind}")
        if layout.kind == "blocked":
            if not layout.params:
                raise IntentIRValidationError(f"tensors.{name}.layout blocked requires params")
            for k, v in layout.params.items():
                if not isinstance(v, int):
                    raise IntentIRValidationError(
                        f"tensors.{name}.layout.params[{k}] must be int for blocked layout"
                    )
        if layout.kind == "custom":
            if not isinstance(layout.params, dict):
                raise IntentIRValidationError(
                    f"tensors.{name}.layout.params must be object for custom layout"
                )


def _validate_axis_roles(axis_roles: Dict[str, str], tensors: Dict[str, TensorType], parallel_axes: List[str]) -> None:
    if not axis_roles:
        return
    known_axes = set(parallel_axes)
    for t in tensors.values():
        for d in t.shape:
            if d.kind == "sym":
                known_axes.add(d.value)
    for axis, role in axis_roles.items():
        if role not in AXIS_ROLE_VALUES:
            raise IntentIRValidationError(f"axis_roles.{axis} role unsupported: {role}")
        # Allow axis_roles that refer to derived/implicit axes (e.g., group_size)
        # even if they are not explicit shape symbols; keep the information
        # without failing validation.


def _validate_ops(ops: List[Op], tensors: Dict[str, TensorType]) -> None:
    if not ops:
        raise IntentIRValidationError("ops must not be empty")
    available_names = set(tensors.keys())
    produced = set()
    # Heuristics for common IR mistakes (LLM outputs).
    _suffix_2d = re.compile(r"^(?P<base>.+?)(?:2d|_2d)$", re.IGNORECASE)
    for idx, op in enumerate(ops):
        if op.op not in SUPPORTED_OPS:
            raise IntentIRValidationError(f"op[{idx}].op unsupported: {op.op}")
        for i, inp in enumerate(op.inputs):
            if inp not in available_names:
                # Rich, user-friendly diagnostic text (Clang-like).
                snippet = format_op_snippet(op, idx=idx)
                caret = None
                try:
                    # place a caret roughly under the undefined input token (best-effort)
                    s = f"{op.op}({', '.join(op.inputs)}) -> {op.output}"
                    needle = str(inp)
                    pos = s.find(needle)
                    if pos >= 0:
                        caret = " " * pos + "^" * max(1, len(needle))
                except Exception:
                    caret = None

                avail = sorted(list(available_names))
                sugg = closest_match(str(inp), avail, n=1)
                notes: List[str] = []
                if sugg:
                    notes.append(f"Did you mean '{sugg[0]}'?")
                notes.append(f"Available tensors at this point: {avail}")

                hints: List[str] = []
                m = _suffix_2d.match(str(inp))
                if m:
                    base = m.group("base")
                    if base in available_names:
                        hints.append(
                            f"If you intended to broadcast '{base}' to match a 2D tensor, insert a broadcast op, e.g.: "
                            f"broadcast_in_dim({base}, out_shape=[M,N], broadcast_dims=[1]) -> {inp}"
                        )

                msg_lines = [
                    f"Error: op[{idx}] '{op.op}' references undefined tensor '{inp}'",
                    f"  -> {snippet}",
                ]
                if caret:
                    msg_lines.append(f"     {caret}")
                for n in notes:
                    msg_lines.append(f"Note: {n}")
                for h in hints:
                    msg_lines.append(f"Hint: {h}")
                raise IntentIRValidationError("\n".join(msg_lines))
        _validate_op_attrs(op, idx)
        if op.output in produced:
            raise IntentIRValidationError(f"op[{idx}].output duplicates previous op output: {op.output}")
        produced.add(op.output)
        available_names.add(op.output)


def _validate_op_attrs(op: Op, idx: int) -> None:
    attrs = op.attrs or {}
    if op.op == "matmul":
        if len(op.inputs) != 2:
            raise IntentIRValidationError(f"op[{idx}] matmul requires 2 inputs (A, B)")
        accum = attrs.get("accum_dtype")
        if accum is not None and accum not in SUPPORTED_DTYPES:
            raise IntentIRValidationError(f"op[{idx}].attrs.accum_dtype unsupported: {accum}")
        if "epilogue" in attrs:
            _validate_epilogue(attrs["epilogue"], f"op[{idx}].attrs.epilogue")
    elif op.op == "const":
        if op.inputs:
            raise IntentIRValidationError(f"op[{idx}] const requires 0 inputs")
        if "value" not in attrs:
            raise IntentIRValidationError(f"op[{idx}] const requires attrs.value")
        dtype = attrs.get("dtype")
        if dtype is not None and dtype not in SUPPORTED_DTYPES:
            raise IntentIRValidationError(f"op[{idx}].attrs.dtype unsupported: {dtype}")
    elif op.op == "cast":
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] cast requires 1 input")
        to = attrs.get("to")
        if to is None and "dtype" in attrs:
            raise IntentIRValidationError(f"op[{idx}] cast requires attrs.to (not attrs.dtype)")
        if to is None:
            raise IntentIRValidationError(f"op[{idx}] cast requires attrs.to")
        if to not in SUPPORTED_DTYPES:
            raise IntentIRValidationError(f"op[{idx}] cast.to unsupported: {to}")
    elif op.op == "iota":
        if op.inputs:
            raise IntentIRValidationError(f"op[{idx}] iota requires 0 inputs")
        shape = attrs.get("shape")
        axis = attrs.get("axis")
        if axis is None and "dimension" in attrs:
            raise IntentIRValidationError(f"op[{idx}] iota requires attrs.axis (not attrs.dimension)")
        dtype = attrs.get("dtype", "i32")
        if not isinstance(shape, list) or not shape:
            raise IntentIRValidationError(f"op[{idx}] iota.shape must be non-empty list")
        for d in shape:
            parse_dim(d)
        if axis is None:
            raise IntentIRValidationError(f"op[{idx}] iota requires attrs.axis")
        if not isinstance(axis, int):
            raise IntentIRValidationError(f"op[{idx}] iota.axis must be int, got {type(axis).__name__}")
        if dtype not in SUPPORTED_DTYPES:
            raise IntentIRValidationError(f"op[{idx}] iota.dtype unsupported: {dtype}")
    elif op.op == "gather":
        # gather(data, i0, i1, ...) where number of indices equals data rank.
        if len(op.inputs) < 2:
            raise IntentIRValidationError(f"op[{idx}] gather requires data + index tensors")
    elif op.op == "where":
        if len(op.inputs) != 3:
            raise IntentIRValidationError(f"op[{idx}] where requires 3 inputs (cond, x, y)")
    elif op.op == "dropout":
        # Triton tl.rand(seed, offsets) is modeled as a single semantic op.
        # Prefer dropout(X, p, seed) where p/seed are scalar tensors (rank-0).
        if len(op.inputs) != 3:
            raise IntentIRValidationError(f"op[{idx}] dropout requires 3 inputs (X, p, seed)")
        # Optional attrs for future extensions (keep lightweight to avoid over-constraining).
        # - n_rounds: Philox rounds (Triton default is 10).
        n_rounds = attrs.get("n_rounds")
        if n_rounds is not None and not isinstance(n_rounds, int):
            raise IntentIRValidationError(f"op[{idx}] dropout.attrs.n_rounds must be int when provided")
    elif op.op in {"add", "sub", "mul", "div", "max", "min"}:
        # Canonical form is binary; allow a small set of legacy shorthands
        # (1 input + scalar attr) for compatibility with older LLM outputs.
        if len(op.inputs) == 2:
            return
        if len(op.inputs) == 1:
            if op.op == "div" and ("divisor" in attrs):
                return
            if op.op == "add" and ("addend" in attrs):
                return
            if op.op == "sub" and ("subtract" in attrs):
                return
            if op.op == "mul" and ("mul_factor" in attrs):
                return
            if op.op in {"max", "min"} and ("other" in attrs):
                return
        raise IntentIRValidationError(f"op[{idx}] {op.op} requires 2 inputs (or 1+scalar attr)")
    elif op.op == "ne":
        if len(op.inputs) != 2:
            raise IntentIRValidationError(f"op[{idx}] ne requires 2 inputs")
    elif op.op in {"abs", "floor", "not", "exp", "relu", "rsqrt", "identity"}:
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] {op.op} requires 1 input")
    elif op.op in {"lt", "le", "gt", "ge", "and", "or"}:
        if len(op.inputs) != 2:
            raise IntentIRValidationError(f"op[{idx}] {op.op} requires 2 inputs")
    elif op.op in {"reduce_sum", "reduce_max", "reduce_any"}:
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] {op.op} requires 1 input")
        dims = attrs.get("dims")
        if dims is None:
            dims = attrs.get("axis")
            if isinstance(dims, int):
                dims = [dims]
        if dims is None:
            raise IntentIRValidationError(f"op[{idx}] {op.op} requires dims or axis")
        if not isinstance(dims, list) or not dims or not all(isinstance(x, int) for x in dims):
            raise IntentIRValidationError(f"op[{idx}] {op.op}.dims must be list[int]")
    elif op.op == "softmax":
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] softmax requires 1 input")
        axis = attrs.get("axis")
        if axis is None:
            raise IntentIRValidationError(f"op[{idx}] softmax requires axis")
        if not isinstance(axis, int):
            raise IntentIRValidationError(f"op[{idx}] softmax.axis must be int, got {type(axis).__name__}")
        stable = attrs.get("stable")
        if stable is not None and not isinstance(stable, bool):
            raise IntentIRValidationError(f"op[{idx}].attrs.stable must be bool when provided")
    elif op.op == "broadcast_in_dim":
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] broadcast_in_dim requires 1 input")
        out_shape = attrs.get("out_shape")
        bcast_dims = attrs.get("broadcast_dims")
        if not isinstance(out_shape, list) or not out_shape:
            raise IntentIRValidationError(f"op[{idx}] broadcast_in_dim.out_shape must be non-empty list")
        for j, d in enumerate(out_shape):
            parse_dim(d)  # validate type only
        if not isinstance(bcast_dims, list):
            raise IntentIRValidationError(f"op[{idx}] broadcast_in_dim.broadcast_dims must be a list")
        for j, v in enumerate(bcast_dims):
            if not isinstance(v, int):
                raise IntentIRValidationError(
                    f"op[{idx}] broadcast_in_dim.broadcast_dims[{j}] must be int"
                )
    elif op.op == "transpose":
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] transpose requires 1 input")
        perm = attrs.get("perm")
        if not isinstance(perm, list) or not perm or not all(isinstance(p, int) for p in perm):
            raise IntentIRValidationError(f"op[{idx}] transpose.perm must be list of int")
    elif op.op == "reshape":
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] reshape requires 1 input")
        shape = attrs.get("shape")
        if not isinstance(shape, list) or not shape:
            raise IntentIRValidationError(f"op[{idx}] reshape.shape must be non-empty list")
        for j, d in enumerate(shape):
            parse_dim(d)
    elif op.op == "layout_cast":
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] layout_cast requires 1 input")
        target = attrs.get("to")
        if not isinstance(target, str):
            raise IntentIRValidationError(f"op[{idx}] layout_cast.to must be string")
    elif op.op == "upsample_bicubic2d_aa":
        # Macro op: output = upsample_bicubic2d_aa(input) with OH/OW derived from output tensor shape.
        # Inputs: [input_tensor]
        if len(op.inputs) != 1:
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa requires 1 input (the input tensor)")
        # Optional attrs (macro impl spec / lowering hints). Keep type checks lightweight:
        # - Allow both "flat" attrs (legacy) and a structured "impl" object (preferred).
        impl = attrs.get("impl")
        if impl is not None and not isinstance(impl, dict):
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.impl must be object when provided")

        a = attrs.get("a")
        if a is not None and not isinstance(a, (int, float)):
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.a must be number when provided")
        support = attrs.get("support")
        if support is not None and not isinstance(support, (int, float)):
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.support must be number when provided")
        invscale = attrs.get("invscale")
        if invscale is not None and not isinstance(invscale, (int, float)):
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.invscale must be number when provided")
        for key in ("kernel", "center_formula", "start_formula", "clamp_policy", "span_size_formula", "tap_enable_policy", "abs_arg", "normalize_avoid_div0", "compute_order", "mask_policy", "reuse"):
            v = attrs.get(key)
            if v is not None and not isinstance(v, str):
                raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.{key} must be string when provided")
        for key in ("normalize_weights", "separable"):
            v = attrs.get(key)
            if v is not None and not isinstance(v, bool):
                raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.{key} must be bool when provided")
        other = attrs.get("other_value")
        if other is not None and not isinstance(other, (int, float)):
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.other_value must be number when provided")
        hoist = attrs.get("hoist")
        if hoist is not None and not (isinstance(hoist, list) and all(isinstance(x, str) for x in hoist)):
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.hoist must be list[str] when provided")
        piecewise = attrs.get("piecewise")
        if piecewise is not None and not isinstance(piecewise, (dict, str)):
            raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.piecewise must be object/string when provided")

        # Validate a few common structured impl keys (do not over-constrain; this evolves).
        if isinstance(impl, dict):
            for key in ("kernel", "index_plan", "composition"):
                v = impl.get(key)
                if v is not None and not isinstance(v, dict):
                    raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.impl.{key} must be object when provided")
            h = impl.get("hoist")
            if h is not None and not (isinstance(h, list) and all(isinstance(x, str) for x in h)):
                raise IntentIRValidationError(f"op[{idx}] upsample_bicubic2d_aa.attrs.impl.hoist must be list[str] when provided")


def _validate_epilogue(node: Any, path: str) -> None:
    if isinstance(node, str):
        return
    if not isinstance(node, dict):
        raise IntentIRValidationError(f"{path} must be string or object")
    op_name = node.get("op")
    inputs = node.get("inputs")
    if op_name not in EPILOGUE_ELEMWISE_OPS:
        raise IntentIRValidationError(f"{path}.op unsupported: {op_name}")
    if not isinstance(inputs, list):
        raise IntentIRValidationError(f"{path}.inputs must be list")
    for idx, child in enumerate(inputs):
        _validate_epilogue(child, f"{path}.inputs[{idx}]")


def _validate_outputs(outputs: List[str], tensors: Dict[str, TensorType], ops: List[Op]) -> None:
    if not outputs:
        raise IntentIRValidationError("outputs must not be empty")
    tensor_names = set(tensors.keys())
    for i, out in enumerate(outputs):
        if out not in tensor_names:
            raise IntentIRValidationError(f"outputs[{i}] not found in tensors: {out}")


def _validate_parallel_axes(parallel_axes: List[str], tensors: Dict[str, TensorType]) -> None:
    if not parallel_axes:
        return
    shape_axes = {d.value for t in tensors.values() for d in t.shape if d.kind == "sym"}
    for i, ax in enumerate(parallel_axes):
        if not isinstance(ax, str):
            raise IntentIRValidationError(f"parallel_axes[{i}] must be string")
        if shape_axes and ax not in shape_axes:
            raise IntentIRValidationError(f"parallel_axes[{i}] not found in any tensor shape: {ax}")


def _validate_schedule(schedule: ScheduleSketch) -> None:
    for name in ("tile_m", "tile_n", "tile_k", "vec_width", "pipeline_depth"):
        val = getattr(schedule, name)
        if val is not None and not isinstance(val, (int, str)):
            raise IntentIRValidationError(f"schedule.{name} must be int/str/None")
    if schedule.vec_axis is not None and not isinstance(schedule.vec_axis, str):
        raise IntentIRValidationError("schedule.vec_axis must be string when provided")
    if not isinstance(schedule.axis_bindings, dict):
        raise IntentIRValidationError("schedule.axis_bindings must be an object")
    for k, v in schedule.axis_bindings.items():
        if k not in {"tile_m", "tile_n", "tile_k", "vec_width"}:
            raise IntentIRValidationError(f"schedule.axis_bindings key unsupported: {k}")
        if not isinstance(v, str):
            raise IntentIRValidationError(f"schedule.axis_bindings[{k}] must be string")
    for i, ax in enumerate(schedule.parallel_axes):
        if not isinstance(ax, str):
            raise IntentIRValidationError(f"schedule.parallel_axes[{i}] must be string")
    if not isinstance(schedule.memory_hint, dict):
        raise IntentIRValidationError("schedule.memory_hint must be an object")
    residency = schedule.memory_hint.get("residency")
    if residency is not None:
        if not isinstance(residency, dict):
            raise IntentIRValidationError("schedule.memory_hint.residency must be object")
        for k, v in residency.items():
            if not isinstance(v, str):
                raise IntentIRValidationError(
                    f"schedule.memory_hint.residency[{k}] must be string"
                )
