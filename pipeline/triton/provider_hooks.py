"""
Compatibility shim for legacy provider hook imports.

New code should import provider plugins from `pipeline.triton.providers`.
"""

from __future__ import annotations

from typing import Any

from intent_ir.parser import CandidateIntent
from pipeline.triton.providers import (
    flaggems_canonical_normalization_enabled,
    get_provider_plugin,
)


def maybe_normalize_provider_candidate(
    *,
    provider: str,
    spec_name: str,
    candidate: CandidateIntent,
    candidate_expanded: CandidateIntent | None,
) -> tuple[CandidateIntent, CandidateIntent | None, dict[str, Any] | None]:
    plugin = get_provider_plugin(provider)
    return plugin.maybe_normalize_candidate(
        spec_name=str(spec_name),
        candidate=candidate,
        candidate_expanded=candidate_expanded,
    )


def annotate_provider_intent_meta(
    intent,
    *,
    provider: str,
    source_op: str | None,
    capability_state: str | None,
    backend_target: str | None,
) -> None:
    plugin = get_provider_plugin(provider)
    plugin.annotate_intent_meta(
        intent,
        source_op=source_op,
        capability_state=capability_state,
        backend_target=backend_target,
    )


def validate_provider_intent_meta(
    intent,
    *,
    provider: str,
    require_source_and_state: bool = False,
) -> dict[str, Any]:
    plugin = get_provider_plugin(provider)
    if bool(require_source_and_state) and not bool(plugin.require_source_and_state):
        # Preserve legacy behavior where callers could force strict checks.
        from pipeline.triton.providers.base import TritonProviderPlugin  # noqa: PLC0415

        plugin = TritonProviderPlugin(name=str(plugin.name), require_source_and_state=True)
    return plugin.validate_intent_meta(intent)


__all__ = [
    "annotate_provider_intent_meta",
    "flaggems_canonical_normalization_enabled",
    "maybe_normalize_provider_candidate",
    "validate_provider_intent_meta",
]
