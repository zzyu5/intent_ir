"""
Compiler-style CUDA backend pipeline (stage skeleton).

This package is intentionally introduced as a non-breaking layer for
incremental migration from script-style codegen flow.
"""

from .stages import CudaPipelineResult, CudaPipelineStage
from .driver import run_cuda_pipeline

__all__ = ["CudaPipelineStage", "CudaPipelineResult", "run_cuda_pipeline"]
