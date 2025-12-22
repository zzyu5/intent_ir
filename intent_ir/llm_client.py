"""
Lightweight wrapper around OpenAI-compatible chat completions endpoints.

Design goals:
- Provider fallback (try requested model, then other configured providers)
- Local on-disk response cache (avoid repeated LLM calls)
- No secrets committed to the repo: API keys are loaded from a local JSON file
  (gitignored) or environment variables.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

import requests


DEFAULT_MODEL = "claude-sonnet-4.5"

# Where to load provider configuration from (gitignored).
_LOCAL_PROVIDERS_PATH = Path(__file__).resolve().parent / "llm_providers.local.json"


def _default_providers_skeleton() -> Dict[str, Dict[str, str]]:
    # Keep this file key-free; real keys should come from llm_providers.local.json.
    return {
        "claude-sonnet-4-5": {"base_url": "https://x666.me/v1", "api_key": ""},
        "claude-sonnet-4.5": {"base_url": "https://ai.hybgzs.com/v1", "api_key": ""},
    }


def _load_providers() -> Dict[str, Dict[str, str]]:
    providers = _default_providers_skeleton()
    # Local file overrides (preferred)
    if _LOCAL_PROVIDERS_PATH.exists():
        try:
            data = json.loads(_LOCAL_PROVIDERS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        providers.setdefault(str(k), {})
                        if "base_url" in v:
                            providers[str(k)]["base_url"] = str(v["base_url"])
                        if "api_key" in v:
                            providers[str(k)]["api_key"] = str(v["api_key"])
        except Exception:
            # If local file is malformed, keep skeleton and allow env fallback below.
            pass

    # Optional environment fallback for CI/automation (kept as last resort).
    for name, cfg in list(providers.items()):
        env_key = os.getenv(f"INTENTIR_{name.upper().replace('-','_').replace('.','_')}_API_KEY")
        if env_key and not cfg.get("api_key"):
            cfg["api_key"] = env_key
        env_url = os.getenv(f"INTENTIR_{name.upper().replace('-','_').replace('.','_')}_BASE_URL")
        if env_url:
            cfg["base_url"] = env_url
    return providers


def _default_base_url(providers: Dict[str, Dict[str, str]]) -> str:
    cfg = providers.get(DEFAULT_MODEL) or next(iter(providers.values()))
    return cfg.get("base_url") or "https://api.openai.com/v1"

# Store cached responses locally to reduce repeated calls (esp. during retries).
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "intentir" / "llm"


class LLMClientError(Exception):
    """Raised when the LLM API call fails."""


@dataclass
class LLMResponse:
    raw: Dict[str, Any]

    def first_message(self) -> str:
        """Return the first message content if present, else empty string."""
        choices = self.raw.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""


def _select_provider(model: str) -> Dict[str, str]:
    providers = _load_providers()
    if model in providers:
        return providers[model]
    return providers.get(DEFAULT_MODEL) or next(iter(providers.values()))


def _candidate_models(requested_model: str) -> List[str]:
    # Try requested model first; if it fails, fall back to the other provider/model.
    models: List[str] = []
    providers = _load_providers()
    if requested_model in providers:
        models.append(requested_model)
    for m in ("claude-sonnet-4.5", "claude-sonnet-4-5"):
        if m in providers and m not in models:
            models.append(m)
    if not models:
        models = [DEFAULT_MODEL]
    return models


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    stream: bool = False,
    timeout: int = 600,
    use_cache: bool = True,
    cache_dir: Optional[str | Path] = None,
    **extra: Any,
) -> LLMResponse:
    """
    Call the chat completions endpoint and return an LLMResponse.
    Provider and key are selected from local config with automatic fallback.
    """
    providers = _load_providers()
    if base_url is None:
        base_url = _default_base_url(providers)
    cache_path: Path | None = None
    if use_cache:
        cd = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        cd.mkdir(parents=True, exist_ok=True)
        cache_key_obj = {"messages": messages, "model": model, "stream": stream, "extra": extra}
        cache_key = hashlib.sha256(json.dumps(cache_key_obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        cache_path = cd / f"{cache_key}.json"
        if cache_path.exists():
            try:
                return LLMResponse(raw=json.loads(cache_path.read_text(encoding="utf-8")))
            except Exception:
                # Corrupt cache; ignore and re-fetch.
                pass

    errors: List[str] = []
    for m in _candidate_models(model):
        provider = _select_provider(m)
        url_base = provider["base_url"]
        api_key = provider["api_key"]
        if not api_key:
            errors.append(
                f"{m}@{url_base}: missing api_key. Create `{_LOCAL_PROVIDERS_PATH}` "
                "or set INTENTIR_<MODEL>_API_KEY env var."
            )
            continue
        url = f"{url_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload: Dict[str, Any] = {
            "model": m,
            "messages": messages,
            "stream": stream,
            **extra,
        }
        # Retry a little on transient failures or rate limits.
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.Timeout:
                errors.append(f"{m}@{url_base}: timeout after {timeout}s (attempt {attempt+1}/3)")
                continue
            except requests.RequestException as e:
                errors.append(f"{m}@{url_base}: request failed: {e} (attempt {attempt+1}/3)")
                continue

            if resp.status_code == 429:
                # Respect provider throttling; user asked us to wait instead of skipping.
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait_s = int(retry_after) if retry_after is not None else 15
                except Exception:
                    wait_s = 15
                errors.append(f"{m}@{url_base}: 429 rate limited, waiting {wait_s}s (attempt {attempt+1}/3)")
                time.sleep(max(1, min(wait_s, 60)))
                continue

            if resp.status_code >= 500:
                errors.append(f"{m}@{url_base}: {resp.status_code} server error (attempt {attempt+1}/3)")
                time.sleep(1.0 + attempt)
                continue

            if resp.status_code != 200:
                errors.append(f"{m}@{url_base}: {resp.status_code} {resp.text[:160]}")
                break

            try:
                data = resp.json()
            except ValueError:
                errors.append(f"{m}@{url_base}: invalid JSON response {resp.text[:160]}")
                break

            if cache_path is not None:
                try:
                    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            return LLMResponse(raw=data)

    raise LLMClientError(" | ".join(errors))


__all__ = ["LLMClientError", "LLMResponse", "chat_completion", "DEFAULT_MODEL"]
