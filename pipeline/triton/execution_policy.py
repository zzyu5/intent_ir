"""
Execution-path policy for Triton frontend runs.

This module centralizes IntentIR-vs-traditional path semantics and legacy
bridge behavior so callers do not need to coordinate scattered flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXECUTION_PATH_VALUES = ("traditional", "intentir")
INTENTIR_MODE_VALUES = ("auto", "force_compile", "force_cache")
FALLBACK_POLICY_VALUES = ("deterministic", "strict")


@dataclass(frozen=True)
class ExecutionPathPolicy:
    path: str = "intentir"
    intentir_mode: str = "auto"
    seed_cache_dir: Path | None = None
    fallback_policy: str = "deterministic"

    @property
    def use_intent_ir(self) -> bool:
        return str(self.path) == "intentir"

    @property
    def intentir_seed_policy(self) -> str:
        if not self.use_intent_ir:
            return "auto"
        return {
            "auto": "auto",
            "force_compile": "force_llm",
            "force_cache": "force_cache",
        }[str(self.intentir_mode)]

    @property
    def use_llm(self) -> bool:
        if not self.use_intent_ir:
            return False
        return self.intentir_seed_policy != "force_cache"

    @property
    def allow_deterministic_fallback(self) -> bool:
        return str(self.fallback_policy) == "deterministic"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "intentir_mode": str(self.intentir_mode),
            "intentir_seed_policy": str(self.intentir_seed_policy),
            "seed_cache_dir": (str(self.seed_cache_dir) if self.seed_cache_dir is not None else None),
            "fallback_policy": str(self.fallback_policy),
            "use_intent_ir": bool(self.use_intent_ir),
            "use_llm": bool(self.use_llm),
            "allow_deterministic_fallback": bool(self.allow_deterministic_fallback),
        }


def make_execution_policy(
    *,
    path: str,
    intentir_mode: str,
    seed_cache_dir: Path | None = None,
    fallback_policy: str = "deterministic",
) -> ExecutionPathPolicy:
    p = str(path).strip().lower()
    m = str(intentir_mode).strip().lower()
    fb = str(fallback_policy).strip().lower()
    if p not in EXECUTION_PATH_VALUES:
        raise ValueError(f"unsupported execution path: {path}")
    if m not in INTENTIR_MODE_VALUES:
        raise ValueError(f"unsupported intentir mode: {intentir_mode}")
    if fb not in FALLBACK_POLICY_VALUES:
        raise ValueError(f"unsupported fallback policy: {fallback_policy}")
    if p == "traditional" and m != "auto":
        raise ValueError("intentir_mode is only valid when execution path is intentir")
    return ExecutionPathPolicy(
        path=p,
        intentir_mode=m,
        seed_cache_dir=seed_cache_dir,
        fallback_policy=fb,
    )


def make_policy_from_legacy_flags(
    *,
    use_intent_ir: bool,
    use_llm: bool,
    intentir_seed_policy: str,
    allow_deterministic_fallback: bool,
) -> ExecutionPathPolicy:
    if not bool(use_intent_ir):
        return make_execution_policy(
            path="traditional",
            intentir_mode="auto",
            fallback_policy=("deterministic" if allow_deterministic_fallback else "strict"),
        )
    seed_policy = str(intentir_seed_policy).strip().lower()
    if seed_policy not in {"auto", "force_llm", "force_cache"}:
        raise ValueError(f"unsupported intentir_seed_policy: {intentir_seed_policy}")
    mode = {
        "auto": "auto",
        "force_llm": "force_compile",
        "force_cache": "force_cache",
    }[seed_policy]
    # Legacy override: use_llm=False + auto historically meant force_cache.
    if mode == "auto" and not bool(use_llm):
        mode = "force_cache"
    return make_execution_policy(
        path="intentir",
        intentir_mode=mode,
        fallback_policy=("deterministic" if allow_deterministic_fallback else "strict"),
    )


__all__ = [
    "EXECUTION_PATH_VALUES",
    "INTENTIR_MODE_VALUES",
    "FALLBACK_POLICY_VALUES",
    "ExecutionPathPolicy",
    "make_execution_policy",
    "make_policy_from_legacy_flags",
]
