"""
Declarative operator specifications for IntentIR.

This module is the single source of truth for:
- legal op names
- tiering (core/experimental/macro)
- lightweight arity and attr-schema metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


OpTier = Literal["core", "experimental", "macro"]
OpKind = Literal["elementwise", "reduction", "transform", "index", "semantic", "macro"]
ArityMode = Literal["exact", "min"]


@dataclass(frozen=True)
class OpSpec:
    name: str
    tier: OpTier
    kind: OpKind
    arity: tuple[int, ...]
    arity_mode: ArityMode = "exact"
    attr_schema: Mapping[str, str] = field(default_factory=dict)
    dtype_rules: str = ""
    shape_rules: str = ""
    backend_expectation: Mapping[str, str] = field(default_factory=dict)

    def allows_input_count(self, n_inputs: int) -> bool:
        if not self.arity:
            return True
        if self.arity_mode == "min":
            return int(n_inputs) >= int(min(self.arity))
        return int(n_inputs) in set(self.arity)


_OP_SPECS: tuple[OpSpec, ...] = (
    # Core compute.
    OpSpec("matmul", "core", "semantic", (2,), attr_schema={"accum_dtype": "dtype", "epilogue": "object"}),
    OpSpec("softmax", "core", "reduction", (1,), attr_schema={"axis": "int!", "stable": "bool"}),
    OpSpec("dropout", "core", "semantic", (3,), attr_schema={"n_rounds": "int"}),
    OpSpec("correlation", "core", "semantic", (3,)),
    OpSpec("resize", "core", "semantic", (1,), attr_schema={"hw_fl": "int"}),
    OpSpec("warp", "core", "semantic", (2,)),
    OpSpec("rope", "core", "semantic", (3,)),
    # Elementwise arithmetic/comparisons/boolean.
    OpSpec("add", "core", "elementwise", (1, 2)),
    OpSpec("sub", "core", "elementwise", (1, 2)),
    OpSpec("mul", "core", "elementwise", (1, 2)),
    OpSpec("div", "core", "elementwise", (1, 2)),
    OpSpec("max", "core", "elementwise", (1, 2)),
    OpSpec("min", "core", "elementwise", (1, 2)),
    OpSpec("ne", "core", "elementwise", (2,)),
    OpSpec("lt", "core", "elementwise", (2,)),
    OpSpec("le", "core", "elementwise", (2,)),
    OpSpec("gt", "core", "elementwise", (2,)),
    OpSpec("ge", "core", "elementwise", (2,)),
    OpSpec("and", "core", "elementwise", (2,)),
    OpSpec("or", "core", "elementwise", (2,)),
    OpSpec("not", "core", "elementwise", (1,)),
    OpSpec("exp", "core", "elementwise", (1,)),
    OpSpec("relu", "core", "elementwise", (1,)),
    OpSpec("rsqrt", "core", "elementwise", (1,)),
    OpSpec("abs", "core", "elementwise", (1,)),
    OpSpec("floor", "core", "elementwise", (1,)),
    OpSpec("cast", "core", "elementwise", (1,), attr_schema={"to": "dtype!"}),
    OpSpec("where", "core", "elementwise", (3,)),
    # Indexing / shape / layout.
    OpSpec("iota", "core", "index", (0,), attr_schema={"shape": "shape!", "axis": "int!", "dtype": "dtype"}),
    OpSpec("gather", "core", "index", (2,), arity_mode="min"),
    OpSpec("identity", "core", "transform", (1,)),
    OpSpec("const", "core", "transform", (0,), attr_schema={"value": "any!", "dtype": "dtype"}),
    OpSpec("reduce_sum", "core", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("reduce_any", "core", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("reduce_max", "core", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("broadcast_in_dim", "core", "transform", (1,), attr_schema={"out_shape": "shape!", "broadcast_dims": "int_list!"}),
    OpSpec("transpose", "core", "transform", (1,), attr_schema={"perm": "int_list!"}),
    OpSpec("reshape", "core", "transform", (1,), attr_schema={"shape": "shape!"}),
    OpSpec("layout_cast", "core", "transform", (1,), attr_schema={"to": "str!"}),
    # Allowed but not fully covered end-to-end.
    OpSpec("conv2d", "experimental", "semantic", (2,), arity_mode="min"),
    OpSpec("log", "experimental", "elementwise", (1,)),
    OpSpec("sin", "experimental", "elementwise", (1,)),
    OpSpec("cos", "experimental", "elementwise", (1,)),
    OpSpec("acos", "experimental", "elementwise", (1,)),
    OpSpec("atan", "experimental", "elementwise", (1,)),
    OpSpec("tan", "experimental", "elementwise", (1,)),
    OpSpec("erf", "experimental", "elementwise", (1,)),
    OpSpec("sqrt", "experimental", "elementwise", (1,)),
    OpSpec("ceil", "experimental", "elementwise", (1,)),
    OpSpec("neg", "experimental", "elementwise", (1,)),
    OpSpec("eq", "experimental", "elementwise", (2,)),
    OpSpec("remainder", "experimental", "elementwise", (2,)),
    OpSpec("pow", "experimental", "elementwise", (2,)),
    OpSpec("reduce_min", "experimental", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("reduce_prod", "experimental", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("mean", "experimental", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("var", "experimental", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("std", "experimental", "reduction", (1,), attr_schema={"dims": "int_list", "axis": "int_or_int_list"}),
    OpSpec("argmax", "experimental", "reduction", (1,), attr_schema={"axis": "int"}),
    OpSpec("argmin", "experimental", "reduction", (1,), attr_schema={"axis": "int"}),
    OpSpec("cumsum", "experimental", "reduction", (1,), attr_schema={"axis": "int"}),
    # Structure / indexing family (staged backend support).
    OpSpec("concat", "experimental", "transform", (1,), arity_mode="min", attr_schema={"axis": "int"}),
    OpSpec("stack", "experimental", "transform", (1,), arity_mode="min", attr_schema={"axis": "int"}),
    OpSpec("tile", "experimental", "transform", (1,), attr_schema={"repeats": "int_or_int_list!"}),
    OpSpec("repeat", "experimental", "transform", (1,), attr_schema={"repeats": "int_or_int_list!", "axis": "int"}),
    OpSpec(
        "repeat_interleave",
        "experimental",
        "transform",
        (1,),
        attr_schema={"repeats": "int_or_int_list!", "axis": "int"},
    ),
    OpSpec("pad", "experimental", "transform", (1,), attr_schema={"pad_width": "object!", "mode": "str", "value": "number"}),
    OpSpec("sort", "experimental", "transform", (1,), attr_schema={"axis": "int", "descending": "bool", "stable": "bool"}),
    OpSpec("topk", "experimental", "index", (1,), attr_schema={"k": "int!", "axis": "int", "largest": "bool", "sorted": "bool"}),
    OpSpec("unique", "experimental", "index", (1,), attr_schema={"axis": "int", "sorted": "bool"}),
    OpSpec("nonzero", "experimental", "index", (1,)),
    # Macro ops.
    OpSpec("upsample_bicubic2d_aa", "macro", "macro", (1,), attr_schema={"impl": "object"}),
)

OP_SPEC_INDEX: dict[str, OpSpec] = {spec.name: spec for spec in _OP_SPECS}


def all_op_specs() -> tuple[OpSpec, ...]:
    return _OP_SPECS


def op_spec_index() -> dict[str, OpSpec]:
    return dict(OP_SPEC_INDEX)


def op_spec_for(op_name: str) -> OpSpec | None:
    return OP_SPEC_INDEX.get(str(op_name))


def ops_by_tier(tier: OpTier) -> set[str]:
    return {spec.name for spec in _OP_SPECS if spec.tier == tier}


__all__ = [
    "OpSpec",
    "OpTier",
    "OpKind",
    "ArityMode",
    "OP_SPEC_INDEX",
    "all_op_specs",
    "op_spec_index",
    "op_spec_for",
    "ops_by_tier",
]
