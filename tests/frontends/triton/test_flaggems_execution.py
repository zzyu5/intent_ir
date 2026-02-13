from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.triton.flaggems_execution import (
    resolve_flaggems_execution,
    sync_seed_back_to_cache,
    sync_seed_into_run_dir,
)


def test_resolve_flaggems_execution_original_path() -> None:
    cfg = resolve_flaggems_execution(flaggems_path="original", intentir_mode="auto")
    assert cfg.flaggems_path == "original"
    assert cfg.intentir_mode == "auto"
    assert cfg.use_intent_ir is False
    assert cfg.intentir_seed_policy == "auto"
    assert cfg.execution_policy.path == "traditional"
    assert cfg.execution_policy.use_llm is False


def test_resolve_flaggems_execution_intentir_modes() -> None:
    auto = resolve_flaggems_execution(flaggems_path="intentir", intentir_mode="auto")
    force_compile = resolve_flaggems_execution(flaggems_path="intentir", intentir_mode="force_compile")
    force_cache = resolve_flaggems_execution(flaggems_path="intentir", intentir_mode="force_cache")
    assert auto.use_intent_ir is True and auto.intentir_seed_policy == "auto"
    assert force_compile.use_intent_ir is True and force_compile.intentir_seed_policy == "force_llm"
    assert force_cache.use_intent_ir is True and force_cache.intentir_seed_policy == "force_cache"
    assert auto.execution_policy.path == "intentir"
    assert force_compile.execution_policy.intentir_mode == "force_compile"
    assert force_cache.execution_policy.use_llm is False


def test_resolve_flaggems_execution_rejects_invalid_combo() -> None:
    with pytest.raises(ValueError, match="--intentir-mode is only valid"):
        resolve_flaggems_execution(flaggems_path="original", intentir_mode="force_cache")


def test_sync_seed_into_run_dir_modes(tmp_path: Path) -> None:
    seed_cache_dir = tmp_path / "seed_cache"
    run_out_dir = tmp_path / "run_out"
    seed_cache_dir.mkdir()
    run_out_dir.mkdir()

    assert (
        sync_seed_into_run_dir(
            spec_name="add2d",
            seed_cache_dir=seed_cache_dir,
            run_out_dir=run_out_dir,
            intentir_mode="auto",
        )
        == "miss"
    )
    assert (
        sync_seed_into_run_dir(
            spec_name="add2d",
            seed_cache_dir=seed_cache_dir,
            run_out_dir=run_out_dir,
            intentir_mode="force_compile",
        )
        == "skipped"
    )
    with pytest.raises(RuntimeError, match="missing seed cache"):
        sync_seed_into_run_dir(
            spec_name="add2d",
            seed_cache_dir=seed_cache_dir,
            run_out_dir=run_out_dir,
            intentir_mode="force_cache",
        )

    cache_seed = seed_cache_dir / "add2d.intent_seed.json"
    cache_seed.write_text('{"kernel":"add2d"}', encoding="utf-8")
    assert (
        sync_seed_into_run_dir(
            spec_name="add2d",
            seed_cache_dir=seed_cache_dir,
            run_out_dir=run_out_dir,
            intentir_mode="auto",
        )
        == "hit"
    )
    run_seed = run_out_dir / "add2d.intent_seed.json"
    assert run_seed.is_file()
    assert run_seed.read_text(encoding="utf-8") == '{"kernel":"add2d"}'


def test_sync_seed_back_to_cache(tmp_path: Path) -> None:
    seed_cache_dir = tmp_path / "seed_cache"
    run_out_dir = tmp_path / "run_out"
    seed_cache_dir.mkdir()
    run_out_dir.mkdir()
    run_seed = run_out_dir / "relu2d.intent_seed.json"

    assert (
        sync_seed_back_to_cache(
            spec_name="relu2d",
            seed_cache_dir=seed_cache_dir,
            run_out_dir=run_out_dir,
            intentir_mode="force_cache",
        )
        is False
    )
    run_seed.write_text('{"kernel":"relu2d"}', encoding="utf-8")
    assert (
        sync_seed_back_to_cache(
            spec_name="relu2d",
            seed_cache_dir=seed_cache_dir,
            run_out_dir=run_out_dir,
            intentir_mode="auto",
        )
        is True
    )
    assert (seed_cache_dir / "relu2d.intent_seed.json").read_text(encoding="utf-8") == '{"kernel":"relu2d"}'
