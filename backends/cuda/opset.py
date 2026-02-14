"""
CUDA backend supported op set (WIP).

This reflects currently implemented lowering paths in `intentir_to_cuda.py`
(direct kernels + fused patterns + generic fused elementwise).
"""

from __future__ import annotations


CUDA_SUPPORTED_OPS: set[str] = {
    # Single-op semantic kernels.
    "matmul",
    "conv1d",
    "dropout",
    "softmax",
    "correlation",
    "resize",
    "warp",
    "rope",
    # Tensor transforms / indexing.
    "transpose",
    "gather",
    "concat",
    "pad",
    # Core primitives covered by pattern + fused elementwise lowerings.
    "const",
    "iota",
    "identity",
    "broadcast_in_dim",
    "cast",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_any",
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "min",
    "relu",
    "abs",
    "exp",
    "acos",
    "atan",
    "cos",
    "erf",
    "floor",
    "ceil",
    "rsqrt",
    "ne",
    "eq",
    "lt",
    "le",
    "gt",
    "ge",
    "bitwise_and",
    "bitwise_or",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "bitwise_not",
    "and",
    "or",
    "not",
    "where",
    "argmax",
    "argmin",
    "avg_pool2d",
    "scaled_dot_product_attention",
}

__all__ = ["CUDA_SUPPORTED_OPS"]
