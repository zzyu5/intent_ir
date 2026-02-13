"""
Rule-based semantic mapping for FlagGems -> IntentIR.

This keeps mapping logic maintainable and explainable by attaching
`mapping_kind` and `intent_pattern_id` for each semantic op.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intent_ir.ops import SUPPORTED_OPS


@dataclass(frozen=True)
class SemanticMapping:
    semantic_op: str
    intent_ops: tuple[str, ...]
    mapping_kind: str
    intent_pattern_id: str
    status_reason_detail: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "semantic_op": self.semantic_op,
            "intent_ops": list(self.intent_ops),
            "mapping_kind": self.mapping_kind,
            "intent_pattern_id": self.intent_pattern_id,
            "status_reason_detail": self.status_reason_detail,
        }


def _mk(
    semantic_op: str,
    intent_ops: tuple[str, ...],
    *,
    mapping_kind: str,
    pattern_id: str,
    detail: str,
) -> SemanticMapping:
    return SemanticMapping(
        semantic_op=str(semantic_op),
        intent_ops=tuple(str(x) for x in intent_ops),
        mapping_kind=str(mapping_kind),
        intent_pattern_id=str(pattern_id),
        status_reason_detail=str(detail),
    )


_UNARY_TEMPLATE: dict[str, SemanticMapping] = {
    "abs": _mk("abs", ("abs",), mapping_kind="unary_template", pattern_id="unary.abs", detail="mapped by unary template"),
    "exp": _mk("exp", ("exp",), mapping_kind="unary_template", pattern_id="unary.exp", detail="mapped by unary template"),
    "relu": _mk("relu", ("relu",), mapping_kind="unary_template", pattern_id="unary.relu", detail="mapped by unary template"),
    "rsqrt": _mk("rsqrt", ("rsqrt",), mapping_kind="unary_template", pattern_id="unary.rsqrt", detail="mapped by unary template"),
    "floor": _mk("floor", ("floor",), mapping_kind="unary_template", pattern_id="unary.floor", detail="mapped by unary template"),
    "sqrt": _mk(
        "sqrt",
        ("rsqrt", "mul"),
        mapping_kind="unary_template",
        pattern_id="unary.sqrt_via_rsqrt",
        detail="mapped as x * rsqrt(x)",
    ),
    "neg": _mk(
        "neg",
        ("const", "mul"),
        mapping_kind="unary_template",
        pattern_id="unary.neg_via_mul",
        detail="mapped as x * (-1)",
    ),
    "ceil": _mk(
        "ceil",
        ("const", "mul", "floor", "mul"),
        mapping_kind="unary_template",
        pattern_id="unary.ceil_via_floor",
        detail="mapped as -floor(-x)",
    ),
    "sigmoid": _mk(
        "sigmoid",
        ("const", "mul", "exp", "add", "div"),
        mapping_kind="unary_template",
        pattern_id="unary.sigmoid_via_exp",
        detail="mapped as 1 / (1 + exp(-x))",
    ),
    "tanh": _mk(
        "tanh",
        ("const", "mul", "exp", "sub", "add", "div"),
        mapping_kind="unary_template",
        pattern_id="unary.tanh_via_exp",
        detail="mapped as (exp(2x)-1)/(exp(2x)+1)",
    ),
    "silu": _mk(
        "silu",
        ("const", "mul", "exp", "add", "div", "mul"),
        mapping_kind="unary_template",
        pattern_id="unary.silu_via_sigmoid",
        detail="mapped as x * sigmoid(x)",
    ),
    "exp2": _mk(
        "exp2",
        ("const", "mul", "exp"),
        mapping_kind="unary_template",
        pattern_id="unary.exp2_via_exp",
        detail="mapped as exp(x * ln(2))",
    ),
}

_BINARY_TEMPLATE: dict[str, SemanticMapping] = {
    "add": _mk("add", ("add",), mapping_kind="binary_template", pattern_id="binary.add", detail="mapped by binary template"),
    "sub": _mk("sub", ("sub",), mapping_kind="binary_template", pattern_id="binary.sub", detail="mapped by binary template"),
    "mul": _mk("mul", ("mul",), mapping_kind="binary_template", pattern_id="binary.mul", detail="mapped by binary template"),
    "div": _mk("div", ("div",), mapping_kind="binary_template", pattern_id="binary.div", detail="mapped by binary template"),
    "max": _mk("max", ("reduce_max",), mapping_kind="reduce_template", pattern_id="reduce.max", detail="mapped to reduce_max over explicit axes"),
    "min": _mk("min", ("min",), mapping_kind="binary_template", pattern_id="binary.min", detail="mapped by binary template"),
}

_CMP_TEMPLATE: dict[str, SemanticMapping] = {
    "gt": _mk("gt", ("gt",), mapping_kind="cmp_template", pattern_id="cmp.gt", detail="mapped by comparison template"),
    "ge": _mk("ge", ("ge",), mapping_kind="cmp_template", pattern_id="cmp.ge", detail="mapped by comparison template"),
    "lt": _mk("lt", ("lt",), mapping_kind="cmp_template", pattern_id="cmp.lt", detail="mapped by comparison template"),
    "le": _mk("le", ("le",), mapping_kind="cmp_template", pattern_id="cmp.le", detail="mapped by comparison template"),
    "ne": _mk("ne", ("ne",), mapping_kind="cmp_template", pattern_id="cmp.ne", detail="mapped by comparison template"),
    "eq": _mk(
        "eq",
        ("ne", "not"),
        mapping_kind="cmp_template",
        pattern_id="cmp.eq_via_not_ne",
        detail="mapped as not(ne(x, y))",
    ),
    "equal": _mk(
        "equal",
        ("ne", "not"),
        mapping_kind="cmp_template",
        pattern_id="cmp.equal_via_not_ne",
        detail="mapped as not(ne(x, y))",
    ),
}

_REDUCE_TEMPLATE: dict[str, SemanticMapping] = {
    "any": _mk("any", ("reduce_any",), mapping_kind="reduce_template", pattern_id="reduce.any", detail="mapped to reduce_any"),
    "sum": _mk("sum", ("reduce_sum",), mapping_kind="reduce_template", pattern_id="reduce.sum", detail="mapped to reduce_sum"),
    "sum_dim": _mk("sum_dim", ("reduce_sum",), mapping_kind="reduce_template", pattern_id="reduce.sum_dim", detail="mapped to reduce_sum with dims"),
    "amax": _mk("amax", ("reduce_max",), mapping_kind="reduce_template", pattern_id="reduce.amax", detail="mapped to reduce_max"),
    "max_dim": _mk("max_dim", ("reduce_max",), mapping_kind="reduce_template", pattern_id="reduce.max_dim", detail="mapped to reduce_max with dims"),
    # all(x) == not(any(not(x)))
    "all": _mk(
        "all",
        ("not", "reduce_any", "not"),
        mapping_kind="reduce_template",
        pattern_id="reduce.all_via_not_any",
        detail="mapped as not(reduce_any(not(x)))",
    ),
}

_INDEX_TEMPLATE: dict[str, SemanticMapping] = {
    "gather": _mk("gather", ("gather",), mapping_kind="index_template", pattern_id="index.gather", detail="mapped by index template"),
    "where_self": _mk("where_self", ("gt", "where"), mapping_kind="index_template", pattern_id="index.where_self", detail="mapped via compare+where"),
    "where_scalar_self": _mk(
        "where_scalar_self",
        ("where",),
        mapping_kind="index_template",
        pattern_id="index.where_scalar_self",
        detail="mapped to where with scalar broadcast",
    ),
    "where_scalar_other": _mk(
        "where_scalar_other",
        ("where",),
        mapping_kind="index_template",
        pattern_id="index.where_scalar_other",
        detail="mapped to where with scalar broadcast",
    ),
}

_MACRO_TEMPLATE: dict[str, SemanticMapping] = {
    "layer_norm": _mk(
        "layer_norm",
        ("reduce_sum", "sub", "mul", "add", "rsqrt", "broadcast_in_dim", "div"),
        mapping_kind="macro_template",
        pattern_id="macro.layer_norm",
        detail="mapped as normalized arithmetic decomposition",
    ),
    "softmax": _mk("softmax", ("softmax",), mapping_kind="macro_template", pattern_id="macro.softmax", detail="mapped to softmax primitive"),
    "clamp": _mk("clamp", ("max", "min"), mapping_kind="macro_template", pattern_id="macro.clamp", detail="mapped as max/min composition"),
    "maximum": _mk(
        "maximum",
        ("max",),
        mapping_kind="macro_template",
        pattern_id="macro.maximum",
        detail="mapped to max primitive",
    ),
    "minimum": _mk(
        "minimum",
        ("min",),
        mapping_kind="macro_template",
        pattern_id="macro.minimum",
        detail="mapped to min primitive",
    ),
    "upsample_bicubic2d_aa": _mk(
        "upsample_bicubic2d_aa",
        ("upsample_bicubic2d_aa",),
        mapping_kind="macro_template",
        pattern_id="macro.upsample_bicubic2d_aa",
        detail="mapped to macro op (expanded before backend lowering)",
    ),
}


def _direct_supported_mapping(semantic_op: str) -> SemanticMapping:
    return _mk(
        semantic_op,
        (semantic_op,),
        mapping_kind="direct_supported_op",
        pattern_id=f"direct.{semantic_op}",
        detail="semantic op equals IntentIR op name",
    )


def resolve_semantic_mapping(semantic_op: str) -> SemanticMapping:
    s = str(semantic_op)
    if s in _UNARY_TEMPLATE:
        return _UNARY_TEMPLATE[s]
    if s in _BINARY_TEMPLATE:
        return _BINARY_TEMPLATE[s]
    if s in _CMP_TEMPLATE:
        return _CMP_TEMPLATE[s]
    if s in _REDUCE_TEMPLATE:
        return _REDUCE_TEMPLATE[s]
    if s in _INDEX_TEMPLATE:
        return _INDEX_TEMPLATE[s]
    if s in _MACRO_TEMPLATE:
        return _MACRO_TEMPLATE[s]
    if s in SUPPORTED_OPS:
        return _direct_supported_mapping(s)
    return _mk(
        s,
        (),
        mapping_kind="unmapped",
        pattern_id="none",
        detail="no semantic rule matched and no direct IntentIR op",
    )


def semantic_to_intent_ops(semantic_op: str) -> list[str]:
    return list(resolve_semantic_mapping(semantic_op).intent_ops)


__all__ = [
    "SemanticMapping",
    "resolve_semantic_mapping",
    "semantic_to_intent_ops",
]
