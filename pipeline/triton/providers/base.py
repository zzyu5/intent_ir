"""
Provider plugin base for Triton pipeline integration.

The core pipeline should depend on this generic plugin contract instead of
hardcoding provider-specific branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intent_ir.parser import CandidateIntent


@dataclass(frozen=True)
class TritonProviderPlugin:
    name: str
    require_source_and_state: bool = False

    def maybe_normalize_candidate(
        self,
        *,
        spec_name: str,
        candidate: CandidateIntent,
        candidate_expanded: CandidateIntent | None,
    ) -> tuple[CandidateIntent, CandidateIntent | None, dict[str, Any] | None]:
        return candidate, candidate_expanded, None

    def annotate_intent_meta(
        self,
        intent,
        *,
        source_op: str | None,
        capability_state: str | None,
        backend_target: str | None,
    ) -> None:
        p = str(self.name).strip()
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

    def validate_intent_meta(self, intent) -> dict[str, Any]:
        p = str(self.name).strip()
        if not p or p == "native":
            return {"ok": True, "skipped": True, "provider": p or "native"}

        fn_meta = dict(getattr(intent, "meta", {}) or {})
        fn_ok = fn_meta.get("provider") == p
        if self.require_source_and_state:
            fn_ok = fn_ok and isinstance(fn_meta.get("source_op"), str) and isinstance(fn_meta.get("capability_state"), str)
        missing_ops: list[int] = []
        for i, op in enumerate(getattr(intent, "ops", []) or []):
            op_meta = dict(getattr(op, "meta", {}) or {})
            op_ok = op_meta.get("provider") == p
            if self.require_source_and_state:
                op_ok = op_ok and isinstance(op_meta.get("source_op"), str) and isinstance(op_meta.get("capability_state"), str)
            if not op_ok:
                missing_ops.append(i)
        return {
            "ok": bool(fn_ok and not missing_ops),
            "skipped": False,
            "provider": p,
            "require_source_and_state": bool(self.require_source_and_state),
            "function_meta_ok": bool(fn_ok),
            "missing_op_meta_indices": list(missing_ops),
        }


__all__ = ["TritonProviderPlugin"]
