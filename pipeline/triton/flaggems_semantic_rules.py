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


_ALIAS_TO_BASE: dict[str, str] = {
    # Scalar/tensor alias forms.
    "eq_scalar": "eq",
    "ge_scalar": "ge",
    "gt_scalar": "gt",
    "le_scalar": "le",
    "lt_scalar": "lt",
    "ne_scalar": "ne",
    # Logical op naming aliases.
    "logical_and": "and",
    "logical_or": "or",
    "logical_not": "not",
    # Arithmetic alias forms.
    "true_divide": "div",
    "div_mode": "div",
    "floor_divide": "div",
    "clamp_min": "maximum",
    "clamp_tensor": "clamp",
    # Shape/layout alias-like ops.
    "copy": "identity",
    "contiguous": "identity",
    "resolve_conj": "identity",
    "resolve_neg": "identity",
    "to_copy": "cast",
    "cat": "concat",
    "hstack": "concat",
    "vstack": "concat",
    "constant_pad_nd": "pad",
    "repeat_interleave_self_int": "repeat_interleave",
    "repeat_interleave_self_tensor": "repeat_interleave",
    "repeat_interleave_tensor": "repeat_interleave",
    "sort_stable": "sort",
    # Bitwise aliases.
    "bitwise_and_scalar": "bitwise_and",
    "bitwise_and_scalar_tensor": "bitwise_and",
    "bitwise_and_tensor": "bitwise_and",
    "bitwise_or_scalar": "bitwise_or",
    "bitwise_or_scalar_tensor": "bitwise_or",
    "bitwise_or_tensor": "bitwise_or",
    # Reduction aliases.
    "mean_dim": "mean",
    "prod_dim": "prod",
    "min_dim": "reduce_min",
    "lerp_scalar": "lerp",
    "lerp_tensor": "lerp",
    "pow_scalar": "pow",
    "pow_tensor_scalar": "pow",
    "pow_tensor_tensor": "pow",
    # Creation aliases.
    "fill_scalar": "full",
    "fill_tensor": "full",
    "ones_like": "ones",
    "zeros_like": "zeros",
    "full_like": "full",
    # Attention/reduction aliases.
    "scaled_softmax_forward": "softmax",
}


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
    "reciprocal": _mk(
        "reciprocal",
        ("const", "div"),
        mapping_kind="unary_template",
        pattern_id="unary.reciprocal_via_div",
        detail="mapped as 1 / x",
    ),
    "log": _mk("log", ("log",), mapping_kind="unary_template", pattern_id="unary.log", detail="mapped to log primitive"),
    "sin": _mk("sin", ("sin",), mapping_kind="unary_template", pattern_id="unary.sin", detail="mapped to sin primitive"),
    "cos": _mk("cos", ("cos",), mapping_kind="unary_template", pattern_id="unary.cos", detail="mapped to cos primitive"),
    "acos": _mk("acos", ("acos",), mapping_kind="unary_template", pattern_id="unary.acos", detail="mapped to acos primitive"),
    "atan": _mk("atan", ("atan",), mapping_kind="unary_template", pattern_id="unary.atan", detail="mapped to atan primitive"),
    "angle": _mk("angle", ("angle",), mapping_kind="unary_template", pattern_id="unary.angle", detail="mapped to angle primitive"),
    "tan": _mk("tan", ("tan",), mapping_kind="unary_template", pattern_id="unary.tan", detail="mapped to tan primitive"),
    "erf": _mk("erf", ("erf",), mapping_kind="unary_template", pattern_id="unary.erf", detail="mapped to erf primitive"),
    "isnan": _mk(
        "isnan",
        ("ne",),
        mapping_kind="unary_template",
        pattern_id="unary.isnan_via_ne_self",
        detail="mapped as ne(x, x)",
    ),
    "isinf": _mk(
        "isinf",
        ("abs", "const", "gt"),
        mapping_kind="unary_template",
        pattern_id="unary.isinf_via_abs_gt_finite_max",
        detail="mapped as abs(x) > f32_max",
    ),
    "isfinite": _mk(
        "isfinite",
        ("abs", "const", "le"),
        mapping_kind="unary_template",
        pattern_id="unary.isfinite_via_abs_le_finite_max",
        detail="mapped as abs(x) <= f32_max",
    ),
}

_BINARY_TEMPLATE: dict[str, SemanticMapping] = {
    "add": _mk("add", ("add",), mapping_kind="binary_template", pattern_id="binary.add", detail="mapped by binary template"),
    "sub": _mk("sub", ("sub",), mapping_kind="binary_template", pattern_id="binary.sub", detail="mapped by binary template"),
    "mul": _mk("mul", ("mul",), mapping_kind="binary_template", pattern_id="binary.mul", detail="mapped by binary template"),
    "div": _mk("div", ("div",), mapping_kind="binary_template", pattern_id="binary.div", detail="mapped by binary template"),
    "bitwise_and": _mk("bitwise_and", ("bitwise_and",), mapping_kind="binary_template", pattern_id="binary.bitwise_and", detail="mapped to bitwise_and primitive"),
    "bitwise_or": _mk("bitwise_or", ("bitwise_or",), mapping_kind="binary_template", pattern_id="binary.bitwise_or", detail="mapped to bitwise_or primitive"),
    "bitwise_left_shift": _mk(
        "bitwise_left_shift",
        ("bitwise_left_shift",),
        mapping_kind="binary_template",
        pattern_id="binary.bitwise_left_shift",
        detail="mapped to bitwise_left_shift primitive",
    ),
    "bitwise_right_shift": _mk(
        "bitwise_right_shift",
        ("bitwise_right_shift",),
        mapping_kind="binary_template",
        pattern_id="binary.bitwise_right_shift",
        detail="mapped to bitwise_right_shift primitive",
    ),
    "min": _mk("min", ("reduce_min",), mapping_kind="reduce_template", pattern_id="reduce.min", detail="mapped to reduce_min"),
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
    "logical_xor": _mk(
        "logical_xor",
        ("or", "and", "not", "and"),
        mapping_kind="cmp_template",
        pattern_id="cmp.logical_xor_via_or_and_not",
        detail="mapped as (a or b) and not(a and b)",
    ),
}

_REDUCE_TEMPLATE: dict[str, SemanticMapping] = {
    "any": _mk("any", ("reduce_any",), mapping_kind="reduce_template", pattern_id="reduce.any", detail="mapped to reduce_any"),
    "sum": _mk("sum", ("reduce_sum",), mapping_kind="reduce_template", pattern_id="reduce.sum", detail="mapped to reduce_sum"),
    "sum_dim": _mk("sum_dim", ("reduce_sum",), mapping_kind="reduce_template", pattern_id="reduce.sum_dim", detail="mapped to reduce_sum with dims"),
    "amax": _mk("amax", ("reduce_max",), mapping_kind="reduce_template", pattern_id="reduce.amax", detail="mapped to reduce_max"),
    "max": _mk("max", ("reduce_max",), mapping_kind="reduce_template", pattern_id="reduce.max", detail="mapped to reduce_max"),
    "max_dim": _mk("max_dim", ("reduce_max",), mapping_kind="reduce_template", pattern_id="reduce.max_dim", detail="mapped to reduce_max with dims"),
    "mean": _mk("mean", ("reduce_sum", "div"), mapping_kind="reduce_template", pattern_id="reduce.mean_via_sum", detail="mapped as reduce_sum / num_elements"),
    "prod": _mk("prod", ("reduce_prod",), mapping_kind="reduce_template", pattern_id="reduce.prod", detail="mapped to reduce_prod"),
    "argmax": _mk("argmax", ("argmax",), mapping_kind="reduce_template", pattern_id="reduce.argmax", detail="mapped to argmax primitive"),
    "argmin": _mk("argmin", ("argmin",), mapping_kind="reduce_template", pattern_id="reduce.argmin", detail="mapped to argmin primitive"),
    "cumsum": _mk("cumsum", ("cumsum",), mapping_kind="reduce_template", pattern_id="reduce.cumsum", detail="mapped to cumsum primitive"),
    "var": _mk("var", ("var",), mapping_kind="reduce_template", pattern_id="reduce.var", detail="mapped to var primitive"),
    "std": _mk("std", ("std",), mapping_kind="reduce_template", pattern_id="reduce.std", detail="mapped to std primitive"),
    "count_nonzero": _mk(
        "count_nonzero",
        ("count_nonzero",),
        mapping_kind="reduce_template",
        pattern_id="reduce.count_nonzero",
        detail="mapped to count_nonzero primitive",
    ),
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
    "index": _mk("index", ("gather",), mapping_kind="index_template", pattern_id="index.index_via_gather", detail="mapped to gather primitive"),
    "index_select": _mk("index_select", ("gather",), mapping_kind="index_template", pattern_id="index.index_select_via_gather", detail="mapped to gather primitive"),
    "embedding": _mk(
        "embedding",
        ("gather",),
        mapping_kind="index_template",
        pattern_id="index.embedding_via_gather",
        detail="mapped to gather primitive with row/col index expansion",
    ),
    "flip": _mk(
        "flip",
        ("gather",),
        mapping_kind="index_template",
        pattern_id="index.flip_via_gather",
        detail="mapped to gather primitive with reversed column indices",
    ),
    "diag": _mk("diag", ("diag",), mapping_kind="index_template", pattern_id="index.diag", detail="mapped to diag primitive"),
    "diag_embed": _mk(
        "diag_embed",
        ("diag_embed",),
        mapping_kind="index_template",
        pattern_id="index.diag_embed",
        detail="mapped to diag_embed primitive",
    ),
    "masked_scatter": _mk(
        "masked_scatter",
        ("masked_scatter",),
        mapping_kind="index_template",
        pattern_id="index.masked_scatter",
        detail="mapped to masked_scatter primitive",
    ),
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
    "group_norm": _mk(
        "group_norm",
        ("reduce_sum", "sub", "mul", "add", "rsqrt", "broadcast_in_dim", "div"),
        mapping_kind="macro_template",
        pattern_id="macro.group_norm",
        detail="mapped as grouped normalized arithmetic decomposition",
    ),
    "batch_norm": _mk(
        "batch_norm",
        ("reduce_sum", "sub", "mul", "add", "rsqrt", "broadcast_in_dim", "div"),
        mapping_kind="macro_template",
        pattern_id="macro.batch_norm",
        detail="mapped as batch normalized arithmetic decomposition",
    ),
    "softmax": _mk("softmax", ("softmax",), mapping_kind="macro_template", pattern_id="macro.softmax", detail="mapped to softmax primitive"),
    "log_softmax": _mk(
        "log_softmax",
        ("softmax", "log"),
        mapping_kind="macro_template",
        pattern_id="macro.log_softmax",
        detail="mapped as log(softmax(x))",
    ),
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
    "lerp": _mk(
        "lerp",
        ("sub", "mul", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.lerp",
        detail="mapped as a + w * (b - a)",
    ),
    "isclose": _mk(
        "isclose",
        ("sub", "abs", "abs", "mul", "add", "le"),
        mapping_kind="macro_template",
        pattern_id="macro.isclose",
        detail="mapped as abs(a-b) <= atol + rtol * abs(b)",
    ),
    "allclose": _mk(
        "allclose",
        ("sub", "abs", "abs", "mul", "add", "le", "not", "reduce_any", "not"),
        mapping_kind="macro_template",
        pattern_id="macro.allclose",
        detail="mapped as not(reduce_any(not(isclose(a,b))))",
    ),
    "threshold": _mk(
        "threshold",
        ("const", "gt", "where"),
        mapping_kind="macro_template",
        pattern_id="macro.threshold",
        detail="mapped as where(x > threshold, x, value)",
    ),
    "masked_fill": _mk(
        "masked_fill",
        ("where",),
        mapping_kind="macro_template",
        pattern_id="macro.masked_fill",
        detail="mapped as where(mask, value, input)",
    ),
    "softplus": _mk(
        "softplus",
        ("exp", "add", "log"),
        mapping_kind="macro_template",
        pattern_id="macro.softplus",
        detail="mapped as log(1 + exp(x))",
    ),
    "celu": _mk(
        "celu",
        ("const", "gt", "exp", "const", "sub", "where"),
        mapping_kind="macro_template",
        pattern_id="macro.celu",
        detail="mapped as where(x > 0, x, alpha * (exp(x/alpha) - 1)) with alpha=1",
    ),
    "elu": _mk(
        "elu",
        ("const", "gt", "exp", "const", "sub", "where"),
        mapping_kind="macro_template",
        pattern_id="macro.elu",
        detail="mapped as where(x > 0, x, alpha * (exp(x) - 1)) with alpha=1",
    ),
    "eye": _mk(
        "eye",
        ("iota", "iota", "ne", "not", "cast"),
        mapping_kind="macro_template",
        pattern_id="macro.eye",
        detail="mapped as cast(not(ne(iota_row, iota_col)))",
    ),
    "eye_m": _mk(
        "eye_m",
        ("iota", "iota", "ne", "not", "cast"),
        mapping_kind="macro_template",
        pattern_id="macro.eye_m",
        detail="mapped as cast(not(ne(iota_row, iota_col))) on rectangular shape",
    ),
    "gelu": _mk(
        "gelu",
        ("const", "mul", "erf", "add", "mul", "mul"),
        mapping_kind="macro_template",
        pattern_id="macro.gelu",
        detail="mapped as 0.5 * x * (1 + erf(x / sqrt(2)))",
    ),
    "log_sigmoid": _mk(
        "log_sigmoid",
        ("sigmoid", "log"),
        mapping_kind="macro_template",
        pattern_id="macro.log_sigmoid",
        detail="mapped as log(sigmoid(x))",
    ),
    "rms_norm": _mk(
        "rms_norm",
        ("mul", "reduce_sum", "add", "rsqrt", "mul"),
        mapping_kind="macro_template",
        pattern_id="macro.rms_norm",
        detail="mapped as x * rsqrt(mean(x*x)+eps)",
    ),
    "rms_norm_forward": _mk(
        "rms_norm_forward",
        ("mul", "reduce_sum", "add", "rsqrt", "mul"),
        mapping_kind="macro_template",
        pattern_id="macro.rms_norm_forward",
        detail="mapped as x * rsqrt(mean(x*x)+eps)",
    ),
    "vector_norm": _mk(
        "vector_norm",
        ("mul", "reduce_sum", "sqrt"),
        mapping_kind="macro_template",
        pattern_id="macro.vector_norm",
        detail="mapped as sqrt(reduce_sum(x*x))",
    ),
    "var_mean": _mk(
        "var_mean",
        ("var", "mean"),
        mapping_kind="macro_template",
        pattern_id="macro.var_mean",
        detail="mapped to var/mean primitives",
    ),
    "normed_cumsum": _mk(
        "normed_cumsum",
        ("cumsum", "div"),
        mapping_kind="macro_template",
        pattern_id="macro.normed_cumsum",
        detail="mapped as cumsum normalized by scalar divisor",
    ),
    "flash_attn_varlen_func": _mk(
        "flash_attn_varlen_func",
        ("matmul", "softmax", "matmul"),
        mapping_kind="macro_template",
        pattern_id="macro.flash_attn_varlen",
        detail="mapped as attention core decomposition",
    ),
    "mm": _mk(
        "mm",
        ("matmul",),
        mapping_kind="macro_template",
        pattern_id="macro.mm",
        detail="mapped to matmul primitive",
    ),
    "bmm": _mk(
        "bmm",
        ("matmul",),
        mapping_kind="macro_template",
        pattern_id="macro.bmm",
        detail="mapped to batched matmul primitive",
    ),
    "dot": _mk(
        "dot",
        ("matmul",),
        mapping_kind="macro_template",
        pattern_id="macro.dot",
        detail="mapped to vector matmul primitive",
    ),
    "vdot": _mk(
        "vdot",
        ("matmul",),
        mapping_kind="macro_template",
        pattern_id="macro.vdot",
        detail="mapped to vector matmul primitive",
    ),
    "mv": _mk(
        "mv",
        ("matmul",),
        mapping_kind="macro_template",
        pattern_id="macro.mv",
        detail="mapped to matrix-vector matmul primitive",
    ),
    "addmm": _mk(
        "addmm",
        ("matmul", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.addmm",
        detail="mapped as add(input, matmul(mat1, mat2))",
    ),
    "baddbmm": _mk(
        "baddbmm",
        ("matmul", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.baddbmm",
        detail="mapped as add(input, matmul(batch1, batch2))",
    ),
    "addmv": _mk(
        "addmv",
        ("matmul", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.addmv",
        detail="mapped as add(input, matmul(mat, vec))",
    ),
    "addcmul": _mk(
        "addcmul",
        ("const", "mul", "mul", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.addcmul",
        detail="mapped as input + value * tensor1 * tensor2",
    ),
    "addcdiv": _mk(
        "addcdiv",
        ("const", "mul", "div", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.addcdiv",
        detail="mapped as input + value * tensor1 / tensor2",
    ),
    "addr": _mk(
        "addr",
        ("const", "mul", "matmul", "const", "mul", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.addr",
        detail="mapped as beta*input + alpha*outer(vec1, vec2) via matmul decomposition",
    ),
    "upsample_bicubic2d_aa": _mk(
        "upsample_bicubic2d_aa",
        ("upsample_bicubic2d_aa",),
        mapping_kind="macro_template",
        pattern_id="macro.upsample_bicubic2d_aa",
        detail="mapped to macro op (expanded before backend lowering)",
    ),
    "zeros": _mk(
        "zeros",
        ("const", "broadcast_in_dim"),
        mapping_kind="macro_template",
        pattern_id="macro.zeros",
        detail="mapped as scalar const(0) + broadcast",
    ),
    "ones": _mk(
        "ones",
        ("const", "broadcast_in_dim"),
        mapping_kind="macro_template",
        pattern_id="macro.ones",
        detail="mapped as scalar const(1) + broadcast",
    ),
    "full": _mk(
        "full",
        ("const", "broadcast_in_dim"),
        mapping_kind="macro_template",
        pattern_id="macro.full",
        detail="mapped as scalar const(value) + broadcast",
    ),
    "bitwise_not": _mk(
        "bitwise_not",
        ("bitwise_not",),
        mapping_kind="macro_template",
        pattern_id="macro.bitwise_not",
        detail="mapped to bitwise_not primitive",
    ),
    "avg_pool2d": _mk(
        "avg_pool2d",
        ("avg_pool2d",),
        mapping_kind="macro_template",
        pattern_id="macro.avg_pool2d",
        detail="mapped to avg_pool2d primitive",
    ),
    "arange": _mk(
        "arange",
        ("iota",),
        mapping_kind="macro_template",
        pattern_id="macro.arange_via_iota",
        detail="mapped to iota primitive (range parameters normalized in attrs)",
    ),
    "linspace": _mk(
        "linspace",
        ("iota", "cast", "sub", "div", "mul", "add"),
        mapping_kind="macro_template",
        pattern_id="macro.linspace_via_iota",
        detail="mapped as start + cast(iota) * ((end-start)/denom)",
    ),
    "logspace": _mk(
        "logspace",
        ("iota", "cast", "sub", "div", "mul", "add", "mul", "exp"),
        mapping_kind="macro_template",
        pattern_id="macro.logspace_via_exp_linspace",
        detail="mapped as exp((start + cast(iota)*step) * log_base)",
    ),
    "isin": _mk(
        "isin",
        ("broadcast_in_dim", "broadcast_in_dim", "ne", "not", "reduce_any"),
        mapping_kind="macro_template",
        pattern_id="macro.isin_via_broadcast_eq_any",
        detail="mapped as reduce_any(not(ne(broadcast(x), broadcast(values))))",
    ),
    "kron": _mk(
        "kron",
        ("kron",),
        mapping_kind="macro_template",
        pattern_id="macro.kron",
        detail="mapped to kron primitive",
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
    if s in _ALIAS_TO_BASE:
        base = str(_ALIAS_TO_BASE[s])
        m = resolve_semantic_mapping(base)
        return _mk(
            s,
            m.intent_ops,
            mapping_kind="alias",
            pattern_id=f"alias.{s}->{base}",
            detail=f"alias to {base}; {m.status_reason_detail}",
        )
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
