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
import time
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


def _maybe_truncate_source(source_text: str) -> str:
    """
    Provider-facing safeguard: truncate very long kernel sources.

    Some proxy providers become unstable (5xx) on large prompts for complex kernels
    (e.g., bicubic upsample with many repeated loads). For such cases, the evidence
    appendix + kernel name is usually sufficient for the LLM to emit a macro op.
    """
    text = str(source_text)
    lines = text.splitlines()
    # Conservative-but-not-overzealous defaults:
    # - do NOT truncate normal kernels (~100–600 LOC), since this breaks cache
    #   locality and can reduce LLM quality for non-macro kernels.
    # - only truncate very large sources that are likely to trigger proxy 5xx.
    max_lines = 1200
    max_chars = 60000
    head = 400
    tail = 120
    try:
        if len(text) <= max_chars and len(lines) <= max_lines:
            return text
    except Exception:
        return text
    head_lines = lines[: max(0, int(head))]
    tail_lines = lines[-max(0, int(tail)) :] if int(tail) > 0 else []
    banner = f"[IntentIR] SOURCE TRUNCATED: original_lines={len(lines)} kept_head={len(head_lines)} kept_tail={len(tail_lines)}"
    return "\n".join([banner, *head_lines, "[IntentIR] ... TRUNCATED ...", *tail_lines])


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
    http_max_retries: int = 4
    http_max_total_wait_s: int = 180
    max_parse_retries: int = 2
    max_attempts: int = 2
    extra_chat_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Provider health state:
    # - Quota exhaustion -> hard disable for this process (until=+inf).
    # - Transient 5xx/proxy issues -> short cooldown (until=now+cooldown_s),
    #   only after repeated failures to avoid flaking out a generally-working provider.
    disabled_models: Dict[str, float] = field(default_factory=dict)  # model -> disabled_until (epoch seconds)
    model_fail_streak: Dict[str, int] = field(default_factory=dict)
    server_error_disable_after: int = 2
    server_error_cooldown_s: int = 180
    # When True, try multiple configured provider/model candidates (in order).
    # For paper experiments, it can be useful to disable fallback to measure
    # raw reliability/cost of a single provider.
    allow_model_fallback: bool = True

    def _maybe_disable_model(self, model: str, err: Exception) -> None:
        """
        Disable a provider/model for the lifetime of this process when we detect
        hard failures (quota exhausted, repeated 5xx), so large suites don't get
        stuck retrying a dead endpoint.
        """
        m = str(model)
        msg = str(err)
        now = time.time()
        # Quota/credit exhaustion: these won't recover without user action.
        hard_markers = [
            "pre_consume_token_quota_failed",
            "insufficient_quota",
            "quota",
            "余额",
            "令牌总使用次数已达到限制",
        ]
        if any(x in msg for x in hard_markers):
            self.disabled_models[m] = float("inf")
            return
        # Transient 5xx from a proxy is often recoverable; only disable after a
        # short streak to avoid one-off flakiness making the suite brittle.
        if "server error" in msg or " 520 " in msg or " 502 " in msg or " 503 " in msg or " 504 " in msg:
            streak = int(self.model_fail_streak.get(m, 0)) + 1
            self.model_fail_streak[m] = streak
            if streak >= max(1, int(self.server_error_disable_after)):
                self.disabled_models[m] = now + float(max(1, int(self.server_error_cooldown_s)))
            return

    def _is_model_disabled(self, model: str) -> bool:
        m = str(model)
        until = self.disabled_models.get(m)
        if until is None:
            return False
        if until == float("inf"):
            return True
        now = time.time()
        if until > now:
            return True
        # cooldown expired
        try:
            del self.disabled_models[m]
        except KeyError:
            pass
        return False

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
            # Complex kernels (e.g., attention with masks) can exceed 1600 tokens.
            # Truncation often manifests as invalid JSON; prefer a larger cap.
            extra.setdefault("max_tokens", 4096)
            # Reduce non-determinism; helps providers obey "JSON only" prompts.
            extra.setdefault("temperature", 0)

            trace: Dict[str, Any] = {
                "requested_model": requested,
                "candidates": (list(candidate_models(requested)) if bool(self.allow_model_fallback) else [requested]),
                "attempts": [],
            }

            for m in trace["candidates"]:
                if self._is_model_disabled(m):
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": False, "stage": "skip", "error": "disabled"})
                    continue
                try:
                    resp = chat_completion(
                        messages,
                        model=m,
                        stream=False,
                        allow_fallback=False,
                        timeout=int(self.timeout_s),
                        max_retries=int(self.http_max_retries),
                        max_total_wait_s=int(self.http_max_total_wait_s),
                        **extra,
                    )
                except LLMClientError as e:
                    last_err = e
                    self._maybe_disable_model(m, e)
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
                                    max_retries=int(self.http_max_retries),
                                    max_total_wait_s=int(self.http_max_total_wait_s),
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
                # Reset transient failure streak on success.
                try:
                    self.model_fail_streak.pop(str(m), None)
                except Exception:
                    pass
                cand.llm_trace = {
                    "prompt_hash": prompt_hash,
                    "frontend": descriptor.frontend,
                    "kernel": descriptor.name,
                    "extract_trace": trace,
                }
                return cand

            # If all model candidates failed, append the last error as feedback and retry.
            if trace.get("attempts"):
                # Preserve multi-provider failure context: by default we would only
                # raise the *last* LLMClientError, losing earlier provider errors.
                # This aggregated message is safe (no API keys) and makes regressions
                # debuggable without rerunning with verbose logs.
                try:
                    attempts = trace.get("attempts") or []
                    errs: List[str] = []
                    for a in attempts:
                        if not isinstance(a, dict) or a.get("ok") is True:
                            continue
                        m = a.get("model")
                        st = a.get("stage")
                        er = a.get("error")
                        if isinstance(m, str) and isinstance(st, str) and isinstance(er, str) and er.strip():
                            errs.append(f"{m}[{st}]: {er}")
                    if errs:
                        # Keep the exception string compact but informative.
                        head = errs[:6]
                        tail = f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""
                        last_err = LLMClientError("all candidates failed: " + " | ".join(head) + tail)
                        # Attach the per-attempt trace so callers (e.g., E3 regression)
                        # can report accurate cache/API usage and failure breakdown.
                        try:
                            setattr(last_err, "intentir_trace", trace)
                            setattr(last_err, "intentir_prompt_hash", prompt_hash)
                            setattr(last_err, "intentir_frontend", descriptor.frontend)
                            setattr(last_err, "intentir_kernel", descriptor.name)
                            setattr(last_err, "intentir_attempt", int(attempt))
                        except Exception:
                            pass
                except Exception:
                    pass
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

        src = _maybe_truncate_source(descriptor.source_text)
        compact = bool(src.startswith("[IntentIR] SOURCE TRUNCATED"))
        if descriptor.frontend == "triton":
            from frontends.triton.llm_intent import build_messages

            return build_messages(src, kernel_name=descriptor.name, extra_instruction=extra_instruction, compact=compact)
        if descriptor.frontend == "tilelang":
            from frontends.tilelang.llm_intent import build_messages

            return build_messages(src, kernel_name=descriptor.name, extra_instruction=extra_instruction, compact=compact)
        if descriptor.frontend == "cuda":
            from frontends.cuda.llm_intent import build_messages

            return build_messages(src, kernel_name=descriptor.name, extra_instruction=extra_instruction, compact=compact)
        raise NotImplementedError(f"LLMIntentHub does not support frontend={descriptor.frontend}")


__all__ = ["LLMIntentHub"]
