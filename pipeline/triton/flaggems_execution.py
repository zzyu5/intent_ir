"""
Shared execution-path helpers for FlagGems scripts.

These helpers keep FlagGems-specific CLI semantics out of generic project
entrypoints while still providing a single source of truth for:
- execution path selection (original vs intentir)
- IntentIR seed policy mapping
- seed cache synchronization
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from intent_ir.macros import expand_macros
from pipeline.triton.execution_policy import (
    ExecutionPathPolicy,
    INTENTIR_MODE_VALUES as POLICY_INTENTIR_MODE_VALUES,
    make_execution_policy,
)
from pipeline.triton.flaggems_intent_normalize import canonical_flaggems_intent_for_spec


FLAGGEMS_PATH_VALUES = ("original", "intentir")
INTENTIR_MODE_VALUES = POLICY_INTENTIR_MODE_VALUES


@dataclass(frozen=True)
class FlaggemsExecutionConfig:
    flaggems_path: str
    intentir_mode: str
    use_intent_ir: bool
    intentir_seed_policy: str
    execution_policy: ExecutionPathPolicy


def resolve_flaggems_execution(
    *,
    flaggems_path: str,
    intentir_mode: str,
    seed_cache_dir: Path | None = None,
    fallback_policy: str = "deterministic",
) -> FlaggemsExecutionConfig:
    path = str(flaggems_path).strip().lower()
    mode = str(intentir_mode).strip().lower()
    if path not in FLAGGEMS_PATH_VALUES:
        raise ValueError(f"unsupported --flaggems-path: {flaggems_path}")
    if mode not in INTENTIR_MODE_VALUES:
        raise ValueError(f"unsupported --intentir-mode: {intentir_mode}")

    if path == "original":
        if mode != "auto":
            raise ValueError("--intentir-mode is only valid when --flaggems-path=intentir")
        policy = make_execution_policy(
            path="traditional",
            intentir_mode="auto",
            seed_cache_dir=seed_cache_dir,
            fallback_policy=fallback_policy,
        )
        return FlaggemsExecutionConfig(
            flaggems_path=path,
            intentir_mode=mode,
            use_intent_ir=False,
            intentir_seed_policy="auto",
            execution_policy=policy,
        )

    # path == intentir
    seed_policy = {
        "auto": "auto",
        "force_compile": "force_llm",
        "force_cache": "force_cache",
    }[mode]
    policy = make_execution_policy(
        path="intentir",
        intentir_mode=mode,
        seed_cache_dir=seed_cache_dir,
        fallback_policy=fallback_policy,
    )
    return FlaggemsExecutionConfig(
        flaggems_path=path,
        intentir_mode=mode,
        use_intent_ir=True,
        intentir_seed_policy=seed_policy,
        execution_policy=policy,
    )


def sync_seed_into_run_dir(*, spec_name: str, seed_cache_dir: Path, run_out_dir: Path, intentir_mode: str) -> str:
    """
    Prepare per-kernel seed file in the current run directory.

    Returns one of: "skipped", "hit", "miss", "synthesized".
    """
    mode = str(intentir_mode)
    if mode not in INTENTIR_MODE_VALUES:
        raise ValueError(f"unsupported intentir_mode: {intentir_mode}")
    if mode == "force_compile":
        return "skipped"

    cache_seed = Path(seed_cache_dir) / f"{spec_name}.intent_seed.json"
    run_seed = Path(run_out_dir) / f"{spec_name}.intent_seed.json"
    if cache_seed.is_file():
        run_seed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_seed, run_seed)
        return "hit"
    if mode == "force_cache":
        raise RuntimeError(f"missing seed cache for {spec_name}: {cache_seed}")
    synthesized = _write_canonical_seed_if_available(spec_name=spec_name, cache_seed=cache_seed, run_seed=run_seed)
    if synthesized:
        return "synthesized"
    return "miss"


def sync_seed_back_to_cache(*, spec_name: str, seed_cache_dir: Path, run_out_dir: Path, intentir_mode: str) -> bool:
    """
    Persist generated seed back to centralized cache.

    Returns True when a seed file was written to cache.
    """
    mode = str(intentir_mode)
    if mode not in INTENTIR_MODE_VALUES:
        raise ValueError(f"unsupported intentir_mode: {intentir_mode}")
    if mode not in {"auto", "force_compile"}:
        return False

    run_seed = Path(run_out_dir) / f"{spec_name}.intent_seed.json"
    if not run_seed.is_file():
        return False
    cache_seed = Path(seed_cache_dir) / f"{spec_name}.intent_seed.json"
    cache_seed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_seed, cache_seed)
    return True


def _write_canonical_seed_if_available(*, spec_name: str, cache_seed: Path, run_seed: Path) -> bool:
    canonical = canonical_flaggems_intent_for_spec(str(spec_name))
    if canonical is None:
        return False

    expanded = expand_macros(canonical)
    payload = {
        "schema_version": "intent_seed_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kernel": str(spec_name),
        "triton_provider": "flaggems",
        "backend_target": None,
        "intent": canonical.to_json_dict(),
        "intent_expanded": expanded.to_json_dict(),
        "problem_params": {},
        "schedule_params": {},
        "raw_json": {"fallback": True, "source": "provider_canonical_seed"},
        "llm_trace": {"fallback": True, "source": "provider_canonical_seed"},
    }
    txt = json.dumps(payload, indent=2, ensure_ascii=False)
    run_seed.parent.mkdir(parents=True, exist_ok=True)
    run_seed.write_text(txt, encoding="utf-8")
    cache_seed.parent.mkdir(parents=True, exist_ok=True)
    cache_seed.write_text(txt, encoding="utf-8")
    return True
