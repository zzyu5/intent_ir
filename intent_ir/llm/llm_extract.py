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
from typing import Any, Dict, List

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
            response: LLMResponse = chat_completion(messages, model=m, stream=False, **chat_kwargs)
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


__all__ = ["strip_code_fence", "parse_json_block", "extract_json_object"]
