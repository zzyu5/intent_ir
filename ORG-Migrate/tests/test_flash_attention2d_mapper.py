from __future__ import annotations

import pytest

from org.mapping.cuda.flash_attention2d import plan_flash_attention2d
from org.schema import validate_org_doc


def _make_org(*, why: list[str] | None = None, how: list[str] | None = None):
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "overlap_pipeline",
                    "why": list(why or []),
                    "how": list(how or []),
                    "dims": ["ATTN_BLOCK_KV", "ATTN_SCORE_WARPS"],
                    "constraints": ["threads <= 1024"],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )


def _candidate_keys(plan) -> set[tuple[str, tuple[tuple[str, int], ...]]]:
    keys: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    for c in list(plan.candidates or []):
        k = (str(c.kernel_kind), tuple(sorted((str(x), int(y)) for x, y in dict(c.bindings or {}).items())))
        keys.add(k)
    return keys


def test_mapper_budget_and_dedupe_stable() -> None:
    org = _make_org()
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=4,
        enable_cpp_extras=False,
    )
    assert len(plan.candidates) <= 4
    assert len(_candidate_keys(plan)) == len(plan.candidates)


def test_mapper_prefer_v7_when_avoid_recompute() -> None:
    org = _make_org(why=["avoid_recompute"])
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        enable_cpp_extras=False,
    )
    assert any(c.kernel_kind == "attn2d_causal_softmax_v7" for c in plan.candidates)
    assert plan.candidates[0].kernel_kind == "attn2d_causal_softmax_v7"
    assert plan.candidates[0].bindings.get("ATTN_BLOCK_KV") == 32


def test_mapper_defaults_to_v6_first_without_tags() -> None:
    org = _make_org()
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        enable_cpp_extras=False,
    )
    assert plan.candidates[0].kernel_kind == "attn2d_causal_softmax_v6"
    assert plan.candidates[0].bindings.get("ATTN_BLOCK_KV") == 32
    assert plan.candidates[0].bindings.get("ATTN_SCORE_WARPS") == 6


def test_mapper_async_copy_candidate_when_guardrails_pass() -> None:
    # node_type=overlap_pipeline should be sufficient to trigger want_async.
    org = _make_org(how=[])
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        enable_cpp_extras=True,
    )
    assert any(int(c.bindings.get("FLASH_ATTN_ASYNC_COPY") or 0) == 1 for c in plan.candidates)


def test_mapper_async_copy_substitution_when_guardrails_fail() -> None:
    org = _make_org(how=[])
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 96, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        enable_cpp_extras=True,
    )
    subs = list((plan.trace or {}).get("substitutions") or [])
    assert any(str(s.get("from") or "") == "abstract.async_copy" for s in subs)


def test_mapper_async_copy_is_prioritized_with_prefer_v7() -> None:
    org = _make_org(why=["avoid_recompute"], how=[])
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        enable_cpp_extras=True,
    )
    assert plan.candidates
    c0 = plan.candidates[0]
    assert c0.kernel_kind == "attn2d_causal_softmax_v7"
    assert int(c0.bindings.get("FLASH_ATTN_ASYNC_COPY") or 0) == 1


def test_mapper_empty_candidates_when_head_dim_not_64() -> None:
    org = _make_org(why=["avoid_recompute"])
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 128},
        target="cuda_5090d",
        budget=32,
        enable_cpp_extras=True,
    )
    assert plan.candidates == []


def test_mapper_respects_org_dim_allowed_sets() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "tiling",
                    "why": [],
                    "how": [],
                    "dims": [
                        {"name": "ATTN_BLOCK_KV", "allowed": [64]},
                        {"name": "ATTN_SCORE_WARPS", "allowed": [4]},
                    ],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        enable_cpp_extras=False,
    )
    assert plan.candidates
    assert list(plan.param_space.get("ATTN_BLOCK_KV") or []) == [64]
    assert list(plan.param_space.get("ATTN_SCORE_WARPS") or []) == [4]
    for c in plan.candidates:
        assert int(c.bindings.get("ATTN_BLOCK_KV") or 0) == 64
        if c.kernel_kind == "attn2d_causal_softmax_v6":
            assert int(c.bindings.get("ATTN_SCORE_WARPS") or 0) == 4


def test_mapper_emits_module_graph_and_passes() -> None:
    org = _make_org(why=["avoid_recompute"])
    plan = plan_flash_attention2d(
        org,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=8,
        enable_cpp_extras=False,
    )
    assert plan.modules
    assert any(getattr(m, "id", "") == "template_v6" for m in plan.modules)
    assert any(getattr(m, "id", "") == "template_v7" for m in plan.modules)
    assert plan.passes
    assert "dedupe_clip" in set(plan.passes)
    assert plan.module_edges
    assert any(getattr(e, "src", "") == "template_v6" for e in plan.module_edges)
