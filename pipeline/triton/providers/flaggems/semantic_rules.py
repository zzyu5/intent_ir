"""
Provider-scoped re-export for FlagGems semantic mapping rules.

This keeps the provider boundary stable while migration away from legacy
module layout is ongoing.
"""

from pipeline.triton.flaggems_semantic_rules import (
    SemanticMapping,
    explain_mapping,
    mapping_for_semantic_op,
    resolve_semantic_mapping,
)

__all__ = [
    "SemanticMapping",
    "explain_mapping",
    "mapping_for_semantic_op",
    "resolve_semantic_mapping",
]
