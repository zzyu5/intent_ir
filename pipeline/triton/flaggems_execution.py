"""
Shared execution-path helpers for FlagGems scripts.

These helpers keep FlagGems-specific CLI semantics out of generic project
entrypoints while still providing a single source of truth for:
- execution path selection (original vs intentir)
- IntentIR seed policy mapping
- seed cache synchronization
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


FLAGGEMS_PATH_VALUES = ("original", "intentir")
INTENTIR_MODE_VALUES = ("auto", "force_compile", "force_cache")


@dataclass(frozen=True)
class FlaggemsExecutionConfig:
    flaggems_path: str
    intentir_mode: str
    use_intent_ir: bool
    intentir_seed_policy: str


def resolve_flaggems_execution(*, flaggems_path: str, intentir_mode: str) -> FlaggemsExecutionConfig:
    path = str(flaggems_path).strip().lower()
    mode = str(intentir_mode).strip().lower()
    if path not in FLAGGEMS_PATH_VALUES:
        raise ValueError(f"unsupported --flaggems-path: {flaggems_path}")
    if mode not in INTENTIR_MODE_VALUES:
        raise ValueError(f"unsupported --intentir-mode: {intentir_mode}")

    if path == "original":
        if mode != "auto":
            raise ValueError("--intentir-mode is only valid when --flaggems-path=intentir")
        return FlaggemsExecutionConfig(
            flaggems_path=path,
            intentir_mode=mode,
            use_intent_ir=False,
            intentir_seed_policy="auto",
        )

    # path == intentir
    seed_policy = {
        "auto": "auto",
        "force_compile": "force_llm",
        "force_cache": "force_cache",
    }[mode]
    return FlaggemsExecutionConfig(
        flaggems_path=path,
        intentir_mode=mode,
        use_intent_ir=True,
        intentir_seed_policy=seed_policy,
    )


def sync_seed_into_run_dir(*, spec_name: str, seed_cache_dir: Path, run_out_dir: Path, intentir_mode: str) -> str:
    """
    Prepare per-kernel seed file in the current run directory.

    Returns one of: "skipped", "hit", "miss".
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

