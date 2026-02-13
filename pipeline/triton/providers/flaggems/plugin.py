from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from intent_ir.parser import CandidateIntent
from pipeline.triton.providers.base import TritonProviderPlugin
from pipeline.triton.providers.flaggems.intent_normalize import maybe_normalize_flaggems_candidate

_ALWAYS_CANONICAL_SPECS = frozenset(
    {
        "count_nonzero2d",
        "diag2d",
        "diag_embed2d",
        "celu2d",
        "elu2d",
        "eye2d",
        "eye_m2d",
    }
)


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def flaggems_canonical_normalization_enabled() -> bool:
    """
    Canonical override is disabled by default.
    Enable only when explicitly debugging unstable extraction quality.
    """
    return _truthy_env("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", "0")


@dataclass(frozen=True)
class FlaggemsProviderPlugin(TritonProviderPlugin):
    name: str = "flaggems"
    require_source_and_state: bool = True

    def maybe_normalize_candidate(
        self,
        *,
        spec_name: str,
        candidate: CandidateIntent,
        candidate_expanded: CandidateIntent | None,
    ) -> tuple[CandidateIntent, CandidateIntent | None, dict[str, Any] | None]:
        force_canonical = str(spec_name) in _ALWAYS_CANONICAL_SPECS
        if not force_canonical and not flaggems_canonical_normalization_enabled():
            return candidate, candidate_expanded, None

        out, out_expanded, info = maybe_normalize_flaggems_candidate(
            spec_name=str(spec_name),
            candidate=candidate,
            candidate_expanded=candidate_expanded,
        )
        if info is None:
            return out, out_expanded, None
        wrapped = dict(info)
        wrapped["provider"] = "flaggems"
        wrapped["enabled_by"] = (
            "provider_required_deterministic_override"
            if force_canonical
            else "INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE"
        )
        return out, out_expanded, wrapped


FLAGGEMS_PROVIDER = FlaggemsProviderPlugin()


__all__ = [
    "FLAGGEMS_PROVIDER",
    "FlaggemsProviderPlugin",
    "flaggems_canonical_normalization_enabled",
]
