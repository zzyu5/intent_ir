from __future__ import annotations

from org.mapping.cuda.masked_attention2d import plan_masked_attention2d
from org.schema import validate_org_doc


def _make_org(*, node_type: str = "parallel_mapping") -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "masked_attention2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": str(node_type),
                    "why": [],
                    "how": [],
                    "dims": [],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )


def test_mapper_python_prefers_v18_for_canonical_tiny() -> None:
    org = _make_org(node_type="parallel_mapping")
    plan = plan_masked_attention2d(
        org,
        shape_bindings={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
        target="cuda_5090d",
        budget=32,
        compiler_stack="python",
    )
    assert [c.kernel_kind for c in plan.candidates] == ["attn2d_causal_softmax_v18", "attn2d_causal_softmax_v4"]


def test_mapper_cpp_plugin_uses_masked_hd16_keys_variant() -> None:
    org = _make_org(node_type="parallel_mapping")
    plan = plan_masked_attention2d(
        org,
        shape_bindings={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
        target="cuda_5090d",
        budget=32,
        compiler_stack="cpp_plugin",
    )
    assert len(plan.candidates) == 1
    assert plan.candidates[0].kernel_kind == "attn2d_causal_softmax_masked_hd16_keys_v1"


def test_mapper_budget_clips() -> None:
    org = _make_org()
    plan = plan_masked_attention2d(
        org,
        shape_bindings={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
        target="cuda_5090d",
        budget=1,
        compiler_stack="python",
    )
    assert len(plan.candidates) == 1

