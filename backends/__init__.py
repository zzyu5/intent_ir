"""
Backend implementations.

A backend lowers (expanded) IntentIR into target code and/or binaries.
"""

__all__ = ["spmd_rvv", "cuda"]
