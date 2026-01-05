"""
Backward-compatible import path.

The frontend-agnostic implementation lives in `frontends/common/static_validate.py`.
"""

from frontends.common.static_validate import StaticObligation, StaticValidationResult, static_validate

__all__ = ["StaticObligation", "StaticValidationResult", "static_validate"]
