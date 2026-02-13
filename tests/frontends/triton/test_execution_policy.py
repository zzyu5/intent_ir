from __future__ import annotations

import pytest

from pipeline.triton.execution_policy import make_execution_policy, make_policy_from_legacy_flags


def test_make_execution_policy_intentir_modes() -> None:
    auto = make_execution_policy(path="intentir", intentir_mode="auto")
    force_compile = make_execution_policy(path="intentir", intentir_mode="force_compile")
    force_cache = make_execution_policy(path="intentir", intentir_mode="force_cache")
    assert auto.intentir_seed_policy == "auto"
    assert force_compile.intentir_seed_policy == "force_llm"
    assert force_cache.intentir_seed_policy == "force_cache"
    assert force_cache.use_llm is False


def test_make_execution_policy_rejects_invalid_combo() -> None:
    with pytest.raises(ValueError, match="only valid when execution path is intentir"):
        make_execution_policy(path="traditional", intentir_mode="force_cache")


def test_make_policy_from_legacy_flags_bridge() -> None:
    p = make_policy_from_legacy_flags(
        use_intent_ir=True,
        use_llm=False,
        intentir_seed_policy="auto",
        allow_deterministic_fallback=False,
    )
    assert p.path == "intentir"
    assert p.intentir_mode == "force_cache"
    assert p.fallback_policy == "strict"
