"""
Provider-scoped re-export for FlagGems intent normalization utilities.

Core pipeline code should import through provider package boundaries.
"""

from pipeline.triton.flaggems_intent_normalize import maybe_normalize_flaggems_candidate

__all__ = ["maybe_normalize_flaggems_candidate"]
