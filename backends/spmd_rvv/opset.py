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
    # Elementwise.
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
    "not",
    "exp",
    "relu",
    "rsqrt",
    "abs",
    "floor",
    "cast",
    "where",
    # Indexing / reductions.
    "iota",
    "gather",
    "reduce_sum",
    "reduce_max",
    "reduce_any",
    # Higher-level kernels.
    "softmax",
    "matmul",
    "dropout",
    "correlation",
    "resize",
    "warp",
}


__all__ = ["SPMD_RVV_SUPPORTED_OPS"]
