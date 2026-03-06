from __future__ import annotations

from org.mapping.cuda.attn_fwd import plan_attn_fwd
from org.schema import validate_org_doc


def _make_org(*, dims: list[object] | None = None) -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "_attn_fwd",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "tiling",
                    "why": [],
                    "how": [],
                    "dims": list(dims or ["ATTN_FWD_BLOCK_M", "ATTN_FWD_BLOCK_KV"]),
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )


def test_mapper_python_generates_only_supported_block_kv_values() -> None:
    org = _make_org()
    plan = plan_attn_fwd(
        org,
        shape_bindings={"Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        compiler_stack="python",
    )
    assert plan.candidates
    assert all(c.kernel_kind == "attn_fwd_tiled_v3" for c in plan.candidates)
    assert {int(c.bindings.get("ATTN_FWD_BLOCK_M") or 0) for c in plan.candidates}.issubset({4, 8})
    assert {int(c.bindings.get("ATTN_FWD_BLOCK_KV") or 0) for c in plan.candidates}.issubset({16, 32})


def test_mapper_cpp_plugin_uses_softmax_v7_kind() -> None:
    org = _make_org()
    plan = plan_attn_fwd(
        org,
        shape_bindings={"Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        compiler_stack="cpp_plugin",
    )
    assert plan.candidates
    assert all(c.kernel_kind == "attn_fwd_softmax_v7" for c in plan.candidates)


def test_mapper_respects_org_dim_allowed_sets() -> None:
    org = _make_org(
        dims=[
            {"name": "ATTN_FWD_BLOCK_M", "allowed": [8]},
            {"name": "ATTN_FWD_BLOCK_KV", "allowed": [32]},
        ]
    )
    plan = plan_attn_fwd(
        org,
        shape_bindings={"Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        target="cuda_5090d",
        budget=32,
        compiler_stack="python",
    )
    assert plan.candidates
    assert list(plan.param_space.get("ATTN_FWD_BLOCK_M") or []) == [8]
    assert list(plan.param_space.get("ATTN_FWD_BLOCK_KV") or []) == [32]
    for c in plan.candidates:
        assert int(c.bindings.get("ATTN_FWD_BLOCK_M") or 0) == 8
        assert int(c.bindings.get("ATTN_FWD_BLOCK_KV") or 0) == 32


def test_mapper_empty_candidates_when_head_dim_not_64() -> None:
    org = _make_org()
    plan = plan_attn_fwd(
        org,
        shape_bindings={"Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 128},
        target="cuda_5090d",
        budget=32,
        compiler_stack="python",
    )
    assert plan.candidates == []

