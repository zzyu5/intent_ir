"""
CUDA backend entrypoints (WIP).

This backend will lower IntentIR into CUDA kernels runnable on NVIDIA GPUs.
"""

from .opset import CUDA_SUPPORTED_OPS  # noqa: F401
from .pipeline import CudaPipelineResult, CudaPipelineStage, run_cuda_pipeline  # noqa: F401
from .runtime import CudaLaunch, CudaRuntimeError, compile_cuda_extension, run_cuda_kernel  # noqa: F401

__all__ = [
    "CUDA_SUPPORTED_OPS",
    "CudaPipelineStage",
    "CudaPipelineResult",
    "run_cuda_pipeline",
    "CudaLaunch",
    "CudaRuntimeError",
    "compile_cuda_extension",
    "run_cuda_kernel",
]
