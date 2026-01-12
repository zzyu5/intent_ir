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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pipeline.interfaces import KernelDescriptor

from intent_ir.ir import IntentIRValidationError
from intent_ir.llm import DEFAULT_MODEL, LLMClientError, candidate_models, chat_completion, parse_json_block
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
            messages = self._build_messages(descriptor, feedback=fb, attempt=attempt, last_error=last_err)
            prompt_hash = _hash_messages(messages)
            requested = model or self.default_model
            extra = dict(self.extra_chat_kwargs)
            extra.setdefault("max_tokens", 800)

            trace: Dict[str, Any] = {
                "requested_model": requested,
                "candidates": list(candidate_models(requested)),
                "attempts": [],
            }

            for m in trace["candidates"]:
                try:
                    resp = chat_completion(
                        messages,
                        model=m,
                        stream=False,
                        allow_fallback=False,
                        timeout=int(self.timeout_s),
                        max_retries=2,
                        **extra,
                    )
                except LLMClientError as e:
                    last_err = e
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": False, "stage": "http", "error": str(e)})
                    continue

                raw_text = resp.first_message()
                cache_hit = bool(resp.meta.get("cache_hit"))
                try:
                    js = parse_json_block(raw_text)
                except Exception as e:
                    last_err = e
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": cache_hit, "stage": "json", "error": str(e)})
                    continue

                try:
                    cand = parse_candidate_json(js)
                except (LLMJsonParseError, IntentIRValidationError) as e:
                    # Semantic parse failed; try the next provider/model candidate
                    # instead of retrying the same broken completion.
                    # If the response came from the on-disk cache, it can lock us
                    # into a permanently-bad completion. Bust that cache entry once
                    # and re-fetch for the same model.
                    if cache_hit:
                        cache_path = resp.meta.get("cache_path")
                        if isinstance(cache_path, str) and cache_path:
                            try:
                                Path(cache_path).unlink(missing_ok=True)
                                resp2 = chat_completion(
                                    messages,
                                    model=m,
                                    stream=False,
                                    allow_fallback=False,
                                    timeout=int(self.timeout_s),
                                    max_retries=2,
                                    **extra,
                                )
                                raw2 = resp2.first_message()
                                js2 = parse_json_block(raw2)
                                cand2 = parse_candidate_json(js2)
                                cache_hit2 = bool(resp2.meta.get("cache_hit"))
                                trace["ok"] = True
                                trace["chosen"] = {
                                    "model": resp2.meta.get("response_model") or resp2.meta.get("model") or m,
                                    "base_url": resp2.meta.get("base_url"),
                                    "cache_hit": cache_hit2,
                                }
                                trace["attempts"].append(
                                    {"model": m, "ok": True, "cache_hit": cache_hit2, "stage": "semantic", "note": "cache_bust_retry"}
                                )
                                cand2.llm_trace = {
                                    "prompt_hash": prompt_hash,
                                    "frontend": descriptor.frontend,
                                    "kernel": descriptor.name,
                                    "extract_trace": trace,
                                }
                                return cand2
                            except Exception:
                                pass
                    last_err = e
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": cache_hit, "stage": "semantic", "error": str(e)})
                    continue

                trace["ok"] = True
                trace["chosen"] = {
                    "model": resp.meta.get("response_model") or resp.meta.get("model") or m,
                    "base_url": resp.meta.get("base_url"),
                    "cache_hit": cache_hit,
                }
                trace["attempts"].append({"model": m, "ok": True, "cache_hit": cache_hit, "stage": "semantic"})
                cand.llm_trace = {
                    "prompt_hash": prompt_hash,
                    "frontend": descriptor.frontend,
                    "kernel": descriptor.name,
                    "extract_trace": trace,
                }
                return cand

            # If all model candidates failed, append the last error as feedback and retry.
            if last_err is not None:
                fb = fb or []
                fb = fb + [f"Previous failure: {type(last_err).__name__}: {last_err}"]
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
        if descriptor.frontend == "cuda":
            from frontends.cuda.llm_intent import build_messages

            return build_messages(descriptor.source_text, kernel_name=descriptor.name, extra_instruction=extra_instruction)
        raise NotImplementedError(f"LLMIntentHub does not support frontend={descriptor.frontend}")


__all__ = ["LLMIntentHub"]
