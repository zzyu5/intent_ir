"""
LLMIntentHub: unified "KernelDescriptor -> CandidateIntent" entrypoint.

This is the place where we:
- inject structured frontend evidence (facts/constraints) into the prompt
- record an execution trace (model/provider/cache/prompt hash)

The hub does NOT hardcode any particular frontend IR; it consumes the generic
KernelDescriptor and selects frontend-specific prompt builders when needed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pipeline.interfaces import KernelDescriptor

from intent_ir.ir import IntentIRValidationError
from intent_ir.llm import DEFAULT_MODEL, LLMClientError, extract_json_object_with_trace
from intent_ir.parser import CandidateIntent, LLMJsonParseError, parse_candidate_json


def _hash_messages(messages: List[Dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evidence_blob(descriptor: KernelDescriptor) -> str:
    ev = {
        "kernel": descriptor.name,
        "frontend": descriptor.frontend,
        "io_spec": descriptor.io_spec,
        "launch": descriptor.launch,
        "frontend_facts": descriptor.frontend_facts,
        "frontend_constraints": descriptor.frontend_constraints,
        "meta": {
            "versions": {k: descriptor.meta.get(k) for k in ("triton", "torch", "tilelang") if descriptor.meta.get(k) is not None}
        },
    }
    return json.dumps(ev, indent=2, ensure_ascii=False, sort_keys=True)


@dataclass
class LLMIntentHub:
    default_model: str = DEFAULT_MODEL
    timeout_s: int = 600
    max_parse_retries: int = 2
    max_attempts: int = 2
    extra_chat_kwargs: Dict[str, Any] = field(default_factory=dict)

    def lift(self, descriptor: KernelDescriptor, *, feedback: Optional[List[str]] = None, model: Optional[str] = None) -> CandidateIntent:
        """
        Produce a CandidateIntent from a KernelDescriptor.

        Retries are limited (max_attempts) to respect provider rate limits.
        """
        fb = [str(x) for x in (feedback or []) if str(x).strip()]
        last_err: Exception | None = None
        for attempt in range(max(1, int(self.max_attempts))):
            try:
                messages = self._build_messages(descriptor, feedback=fb, attempt=attempt, last_error=last_err)
                prompt_hash = _hash_messages(messages)
                js, trace = extract_json_object_with_trace(
                    messages,
                    model=(model or self.default_model),
                    timeout=int(self.timeout_s),
                    max_parse_retries=int(self.max_parse_retries),
                    **dict(self.extra_chat_kwargs),
                )
                cand = parse_candidate_json(js)
                cand.llm_trace = {
                    "prompt_hash": prompt_hash,
                    "frontend": descriptor.frontend,
                    "kernel": descriptor.name,
                    "extract_trace": trace,
                }
                return cand
            except (LLMClientError, LLMJsonParseError, IntentIRValidationError) as e:
                last_err = e
                fb = fb or []
                fb = fb + [f"Previous failure: {type(e).__name__}: {e}"]
                continue
        raise last_err or RuntimeError("LLMIntentHub.lift failed without exception")

    def _build_messages(
        self,
        descriptor: KernelDescriptor,
        *,
        feedback: List[str],
        attempt: int,
        last_error: Exception | None,
    ) -> List[Dict[str, str]]:
        evidence = _evidence_blob(descriptor)
        extra_lines: List[str] = [
            "Evidence appendix (JSON):",
            evidence,
            "",
            "Use the evidence to align output tensors, masks, and reduce axes; do not copy TTIR lines verbatim.",
        ]
        if feedback:
            extra_lines += ["", "Feedback from previous failures:", *[f"- {x}" for x in feedback]]
        if attempt > 0 and last_error is not None:
            extra_lines += ["", f"Retry attempt={attempt} after error: {type(last_error).__name__}: {last_error}"]
        extra_instruction = "\n".join(extra_lines).strip()

        if descriptor.frontend == "triton":
            from frontends.triton.llm_intent import build_messages

            return build_messages(descriptor.source_text, kernel_name=descriptor.name, extra_instruction=extra_instruction)
        if descriptor.frontend == "tilelang":
            from frontends.tilelang.llm_intent import build_messages

            return build_messages(descriptor.source_text, kernel_name=descriptor.name, extra_instruction=extra_instruction)
        raise NotImplementedError(f"LLMIntentHub does not support frontend={descriptor.frontend}")


__all__ = ["LLMIntentHub"]
