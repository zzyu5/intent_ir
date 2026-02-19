"""
Backend implementations.

A backend lowers (expanded) IntentIR into target code and/or binaries.
"""

from typing import Any


def supported_ops_for_target(target: str) -> set[str]:
    from .capability import supported_ops_for_target as _impl

    return _impl(target)


def check_target_support(intent: Any, target: str) -> tuple[bool, list[str]]:
    from .capability import check_target_support as _impl

    return _impl(intent, target)


def check_dual_backend_support(intent: Any) -> tuple[bool, list[str], list[str]]:
    from .capability import check_dual_backend_support as _impl

    return _impl(intent)

__all__ = [
    "spmd_rvv",
    "cuda",
    "supported_ops_for_target",
    "check_target_support",
    "check_dual_backend_support",
]
