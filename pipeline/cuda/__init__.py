"""
CUDA full-pipeline orchestration (frontend: cuda).

This package mirrors `pipeline/triton` and `pipeline/tilelang`:
- `pipeline/cuda/core.py` holds kernel specs + stage1..7 orchestration
- scripts/workflow call into it indirectly via backend runners (e.g., `scripts/cuda_backend_smoke.py`)
"""

from .core import coverage_kernel_specs, default_kernel_specs, run_pipeline_for_spec

__all__ = ["coverage_kernel_specs", "default_kernel_specs", "run_pipeline_for_spec"]
