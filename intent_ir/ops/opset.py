"""
Canonical operator set for IntentIR.

Motivation (PROJECT_CRITICAL_GAPS_ANALYSIS_2025.md §3.1):
- Adding a new op should not require hunting across multiple files.
- We need a single "source of truth" for what op names are legal in IntentIR JSON.
- We also need a clear notion of "core" vs "experimental/out-of-scope" ops for
  end-to-end verification and backend lowering.

This module only defines op names and tiers. It intentionally does NOT import:
- `intent_ir.ir` (to avoid cycles)
- `verify/` or `backends/` (frontends/backends own their support sets)
"""

from __future__ import annotations


# Ops that are intended to be supported end-to-end (LLM → validate → macro-expand → interpreter → RVV lowering).
CORE_OPS: set[str] = {
    # Core compute.
    "matmul",
    "softmax",
    "dropout",
    # Elementwise arithmetic / comparisons / boolean.
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
    # Indexing / shape / layout.
    "iota",
    "gather",
    "identity",
    "const",
    "reduce_sum",
    "reduce_any",
    "reduce_max",
    "broadcast_in_dim",
    "transpose",
    "reshape",
    "layout_cast",
}

# Ops that are allowed by the IR schema but are not yet covered end-to-end.
# They should trigger graceful degradation (PARTIAL/OUT_OF_SCOPE) rather than
# silently producing incorrect results.
EXPERIMENTAL_OPS: set[str] = {
    "conv2d",
}

# Semantic macro ops: allowed in IntentIR JSON, but must be expanded into CORE_OPS
# before interpreter / backend lowering.
MACRO_OPS: set[str] = {
    "upsample_bicubic2d_aa",
}

SUPPORTED_OPS: set[str] = set().union(CORE_OPS, EXPERIMENTAL_OPS, MACRO_OPS)


__all__ = ["SUPPORTED_OPS", "CORE_OPS", "EXPERIMENTAL_OPS", "MACRO_OPS"]
