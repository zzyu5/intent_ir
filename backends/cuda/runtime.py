"""
CUDA backend runtime helpers (MVP).

We reuse the Torch CUDA extension runner from `frontends/cuda/runtime.py` to keep
the backend dependency-light (torch + nvcc only).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from frontends.cuda.runtime import (  # noqa: F401
    CudaLaunch,
    CudaRuntimeError,
    compile_cuda_extension,
    run_cuda_kernel_io,
)


def run_cuda_kernel(
    *,
    kernel_name: str,
    cuda_src: str,
    io_spec: Dict[str, Any],
    launch: CudaLaunch,
    bindings: Dict[str, Any],
    inputs_np: Dict[str, np.ndarray],
    output_names: Iterable[str],
    extra_cuda_cflags: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Thin wrapper around `frontends.cuda.runtime.run_cuda_kernel_io`.
    """
    return run_cuda_kernel_io(
        kernel_name=kernel_name,
        cuda_src=cuda_src,
        io_spec=io_spec,
        launch=launch,
        bindings=bindings,
        inputs_np=inputs_np,
        output_names=output_names,
        extra_cuda_cflags=extra_cuda_cflags,
    )


__all__ = ["CudaLaunch", "CudaRuntimeError", "compile_cuda_extension", "run_cuda_kernel"]
