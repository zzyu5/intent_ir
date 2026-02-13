"""
Provider hooks for Triton pipeline integration.

This module keeps provider-specific behavior out of `pipeline.triton.core`.
`flaggems` is treated as a Triton provider, and all provider custom logic is
isolated here so IntentIR generation stays provider-agnostic by default.
"""

from __future__ import annotations

import os
from typing import Any

from intent_ir.parser import CandidateIntent


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def flaggems_canonical_normalization_enabled() -> bool:
    """
    Canonical override is disabled by default.

    Rationale:
    - FlagGems is a Triton provider subset.
    - IntentIR path should not silently diverge into provider-specific rewrites.
    - Keep this as an opt-in debugging/resilience tool.
    """
    return _truthy_env("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", "0")


def maybe_normalize_provider_candidate(
    *,
    provider: str,
    spec_name: str,
    candidate: CandidateIntent,
    candidate_expanded: CandidateIntent | None,
) -> tuple[CandidateIntent, CandidateIntent | None, dict[str, Any] | None]:
    p = str(provider).strip().lower()
    if p != "flaggems":
        return candidate, candidate_expanded, None
    if not flaggems_canonical_normalization_enabled():
        return candidate, candidate_expanded, None
    from pipeline.triton.flaggems_intent_normalize import maybe_normalize_flaggems_candidate  # noqa: PLC0415

    out, out_expanded, info = maybe_normalize_flaggems_candidate(
        spec_name=str(spec_name),
        candidate=candidate,
        candidate_expanded=candidate_expanded,
    )
    if info is None:
        return out, out_expanded, None
    wrapped = dict(info)
    wrapped["provider"] = "flaggems"
    wrapped["enabled_by"] = "INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE"
    return out, out_expanded, wrapped


def annotate_provider_intent_meta(
    intent,
    *,
    provider: str,
    source_op: str | None,
    capability_state: str | None,
    backend_target: str | None,
) -> None:
    p = str(provider).strip()
    if not p or p == "native":
        return
    meta = dict(getattr(intent, "meta", {}) or {})
    meta["provider"] = p
    if source_op is not None:
        meta["source_op"] = str(source_op)
    if capability_state is not None:
        meta["capability_state"] = str(capability_state)
    if backend_target is not None:
        meta["backend_target"] = str(backend_target)
    intent.meta = meta

    for op in (getattr(intent, "ops", []) or []):
        op_meta = dict(getattr(op, "meta", {}) or {})
        op_meta["provider"] = p
        if source_op is not None:
            op_meta["source_op"] = str(source_op)
        if capability_state is not None:
            op_meta["capability_state"] = str(capability_state)
        if backend_target is not None:
            op_meta["backend_target"] = str(backend_target)
        setattr(op, "meta", op_meta)


def validate_provider_intent_meta(
    intent,
    *,
    provider: str,
    require_source_and_state: bool = False,
) -> dict[str, Any]:
    p = str(provider).strip()
    if not p or p == "native":
        return {"ok": True, "skipped": True, "provider": p or "native"}

    fn_meta = dict(getattr(intent, "meta", {}) or {})
    fn_ok = fn_meta.get("provider") == p
    if require_source_and_state:
        fn_ok = fn_ok and isinstance(fn_meta.get("source_op"), str) and isinstance(fn_meta.get("capability_state"), str)
    missing_ops: list[int] = []
    for i, op in enumerate(getattr(intent, "ops", []) or []):
        op_meta = dict(getattr(op, "meta", {}) or {})
        op_ok = op_meta.get("provider") == p
        if require_source_and_state:
            op_ok = op_ok and isinstance(op_meta.get("source_op"), str) and isinstance(op_meta.get("capability_state"), str)
        if not op_ok:
            missing_ops.append(i)
    return {
        "ok": bool(fn_ok and not missing_ops),
        "skipped": False,
        "provider": p,
        "require_source_and_state": bool(require_source_and_state),
        "function_meta_ok": bool(fn_ok),
        "missing_op_meta_indices": list(missing_ops),
    }


__all__ = [
    "annotate_provider_intent_meta",
    "flaggems_canonical_normalization_enabled",
    "maybe_normalize_provider_candidate",
    "validate_provider_intent_meta",
]
