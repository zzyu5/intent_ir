"""
Frontend-agnostic helpers for extracting structured JSON from LLM responses.

IntentIR core should not hardcode any particular DSL (Triton/CUDA/TileLang).
Frontends are responsible for building prompts/messages; this module only:
- calls the configured provider(s) via `intent_ir.llm.llm_client.chat_completion`
- parses the first assistant message as a JSON object
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .llm_client import DEFAULT_MODEL, LLMClientError, LLMResponse, candidate_models, chat_completion


def strip_code_fence(text: str) -> str:
    fence = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
    return fence.sub("", text).strip()


def parse_json_block(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: grab substring between first { and last } and retry.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                # Remove trailing commas before } or ].
                cleaned2 = re.sub(r",\\s*([}\\]])", r"\\1", snippet)
                return json.loads(cleaned2)
        raise


def extract_json_object(
    messages: List[Dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    max_parse_retries: int = 2,
    **chat_kwargs: Any,
) -> Dict[str, Any]:
    """
    Call the chat completion endpoint and parse the response into a JSON object.

    - HTTP/provider fallback is handled by `intent_ir.llm.llm_client` itself.
    - This adds content-level fallback: if a provider returns non-JSON prose, try
      other providers/models; if still failing, retry once with a STRICT JSON hint.
    """
    last_err: Exception | None = None
    chat_kwargs = dict(chat_kwargs)
    chat_kwargs.setdefault("max_tokens", 800)
    model_candidates = candidate_models(model)

    for attempt in range(max(1, int(max_parse_retries))):
        for m in model_candidates:
            try:
                response: LLMResponse = chat_completion(messages, model=m, stream=False, allow_fallback=False, **chat_kwargs)
            except LLMClientError as e:
                last_err = e
                continue
            raw_text = response.first_message()
            try:
                return parse_json_block(raw_text)
            except json.JSONDecodeError as e:
                last_err = e
                continue
        # Retry with a stronger hint (append to the user message).
        if last_err is not None and messages and messages[-1].get("role") == "user":
            messages[-1]["content"] += (
                f"\nPrevious attempt failed to parse JSON: {last_err}. "
                "Please return STRICT JSON (no prose/code fences), and ensure arrays/objects have valid commas."
            )

    snippet = "" if last_err is None else str(last_err)
    raise LLMClientError(f"Failed to parse LLM JSON after retries: {snippet}")


def extract_json_object_with_trace(
    messages: List[Dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    max_parse_retries: int = 2,
    **chat_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Like extract_json_object, but also returns a small trace dict for reporting.

    Trace includes which model succeeded, whether cache was hit, and parse errors
    encountered during retries.
    """
    last_err: Exception | None = None
    chat_kwargs = dict(chat_kwargs)
    chat_kwargs.setdefault("max_tokens", 800)
    model_candidates = candidate_models(model)

    trace: Dict[str, Any] = {
        "requested_model": model,
        "candidates": list(model_candidates),
        "attempts": [],
    }

    for attempt in range(max(1, int(max_parse_retries))):
        for m in model_candidates:
            try:
                response: LLMResponse = chat_completion(messages, model=m, stream=False, allow_fallback=False, **chat_kwargs)
            except LLMClientError as e:
                last_err = e
                trace["attempts"].append({"model": m, "ok": False, "cache_hit": False, "error": str(e)})
                continue
            raw_text = response.first_message()
            try:
                obj = parse_json_block(raw_text)
                trace["ok"] = True
                trace["chosen"] = {
                    "model": response.meta.get("model") or response.meta.get("response_model") or m,
                    "base_url": response.meta.get("base_url"),
                    "cache_hit": bool(response.meta.get("cache_hit")),
                }
                trace["attempts"].append({"model": m, "ok": True, "cache_hit": bool(response.meta.get("cache_hit"))})
                return obj, trace
            except json.JSONDecodeError as e:
                last_err = e
                trace["attempts"].append({"model": m, "ok": False, "cache_hit": bool(response.meta.get("cache_hit")), "error": str(e)})
                continue
        # Retry with a stronger hint (append to the user message).
        if last_err is not None and messages and messages[-1].get("role") == "user":
            messages[-1]["content"] += (
                f"\nPrevious attempt failed to parse JSON: {last_err}. "
                "Please return STRICT JSON (no prose/code fences), and ensure arrays/objects have valid commas."
            )

    trace["ok"] = False
    trace["last_error"] = "" if last_err is None else str(last_err)
    snippet = "" if last_err is None else str(last_err)
    raise LLMClientError(f"Failed to parse LLM JSON after retries: {snippet}")


__all__ = ["strip_code_fence", "parse_json_block", "extract_json_object", "extract_json_object_with_trace"]
