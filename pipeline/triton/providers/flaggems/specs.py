"""
Provider-scoped re-export for FlagGems kernel spec factories.
"""

from pipeline.triton.flaggems_specs import (
    coverage_flaggems_kernel_specs,
    default_flaggems_kernel_specs,
)

__all__ = ["coverage_flaggems_kernel_specs", "default_flaggems_kernel_specs"]
