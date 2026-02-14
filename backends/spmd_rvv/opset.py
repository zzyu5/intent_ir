"""
SPMD+RVV backend supported op set.

This is used for:
- preflight checks (better errors than "unsupported op lowering")
- completeness tests (PROJECT_CRITICAL_GAPS_ANALYSIS_2025.md §3.1)

Source of truth for actual lowering is the C++ codegen tool under
`backends/spmd_rvv/cpp_codegen/intentir_codegen.cpp`.
"""

from __future__ import annotations


SPMD_RVV_SUPPORTED_OPS: set[str] = {
    # Constants / identity-like ops.
    "const",
    "identity",
    "layout_cast",
    "reshape",
    "transpose",
    "broadcast_in_dim",
    "concat",
    "pad",
    # Elementwise.
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "ne",
    "eq",
    "lt",
    "le",
    "gt",
    "ge",
    "bitwise_and",
    "bitwise_or",
    "bitwise_not",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "and",
    "or",
    "not",
    "exp",
    "acos",
    "atan",
    "cos",
    "erf",
    "relu",
    "rsqrt",
    "abs",
    "floor",
    "ceil",
    "cast",
    "where",
    # Indexing / reductions.
    "iota",
    "gather",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_any",
    "argmax",
    "argmin",
    "cumsum",
    "cummax",
    "cummin",
    "avg_pool2d",
    "scaled_dot_product_attention",
    # Higher-level kernels.
    "softmax",
    "matmul",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_depthwise2d",
    "dropout",
    "correlation",
    "resize",
    "warp",
    "rope",
}


__all__ = ["SPMD_RVV_SUPPORTED_OPS"]
