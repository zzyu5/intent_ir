"""
CUDA backend supported op set (WIP).

MVP target: AI-Bench8 suite + core primitives (const/elemwise/reduce/matmul).
"""

from __future__ import annotations


CUDA_SUPPORTED_OPS: set[str] = {
    # (filled in as CUDA lowering lands)
}

__all__ = ["CUDA_SUPPORTED_OPS"]

