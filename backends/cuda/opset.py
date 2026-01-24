"""
CUDA backend supported op set (WIP).

MVP target: AI-Bench8 suite + core primitives (const/elemwise/reduce/matmul).
"""

from __future__ import annotations


CUDA_SUPPORTED_OPS: set[str] = {
    # Single-op macro kernels (AI-Bench8).
    "matmul",
    "dropout",
    "correlation",
    "resize",
    "warp",
    "rope",
    # Patterns we currently lower as fused CUDA kernels (softmax / layernorm).
    # These appear as primitive ops in IntentIR today.
    "const",
    "identity",
    "reduce_sum",
    "reduce_max",
    "broadcast_in_dim",
    "add",
    "sub",
    "mul",
    "div",
    "exp",
    "rsqrt",
}

__all__ = ["CUDA_SUPPORTED_OPS"]
