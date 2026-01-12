"""
CUDA full-pipeline orchestration (frontend: cuda).

This package mirrors `pipeline/triton` and `pipeline/tilelang`:
- `pipeline/cuda/core.py` holds kernel specs + stage1..7 orchestration
- scripts call into it (e.g., `scripts/full_pipeline_verify.py --frontend cuda`)
"""

from .core import coverage_kernel_specs, default_kernel_specs, run_pipeline_for_spec

__all__ = ["coverage_kernel_specs", "default_kernel_specs", "run_pipeline_for_spec"]

