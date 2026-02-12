"""
Backend implementations.

A backend lowers (expanded) IntentIR into target code and/or binaries.
"""

from .capability import check_dual_backend_support, check_target_support, supported_ops_for_target  # noqa: F401

__all__ = [
    "spmd_rvv",
    "cuda",
    "supported_ops_for_target",
    "check_target_support",
    "check_dual_backend_support",
]
