from __future__ import annotations

from org.mapping.cuda.matmul_fused_epilogue2d import plan_matmul_fused_epilogue2d
from org.schema import validate_org_doc


def test_mapper_emits_mma_async_candidate_and_tiled_baseline() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "matmul_fused_epilogue2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "overlap_pipeline",
                    "why": ["hide_memory_latency"],
                    "how": ["double_buffering"],
                    "dims": [
                        {"name": "MMA_BM", "allowed": [32]},
                        {"name": "MMA_BN", "allowed": [32]},
                        {"name": "MMA_BK", "allowed": [32]},
                    ],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )
    plan = plan_matmul_fused_epilogue2d(
        org,
        shape_bindings={"M": 32, "N": 32, "K": 32},
        target="cuda_5090d",
        budget=32,
    )
    assert plan.candidates
    assert any(c.kernel_kind == "matmul_tile_v2" for c in plan.candidates)
    assert any(
        c.kernel_kind == "matmul_mma_tf32_v1" and int(c.bindings.get("MMA_ASYNC_COPY") or 0) == 1
        for c in plan.candidates
    )


def test_mapper_respects_dim_allowed_sets() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "matmul_fused_epilogue2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "special_primitive",
                    "why": [],
                    "how": [],
                    "dims": [
                        {"name": "MMA_BM", "allowed": [32]},
                        {"name": "MMA_BN", "allowed": [32]},
                        {"name": "MMA_BK", "allowed": [16]},
                    ],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )
    plan = plan_matmul_fused_epilogue2d(
        org,
        shape_bindings={"M": 32, "N": 32, "K": 32},
        target="cuda_5090d",
        budget=32,
    )
    assert list(plan.param_space.get("MMA_BM") or []) == [32]
    assert list(plan.param_space.get("MMA_BN") or []) == [32]
    assert list(plan.param_space.get("MMA_BK") or []) == [16]
    for c in plan.candidates:
        if c.kernel_kind != "matmul_mma_tf32_v1":
            continue
        assert int(c.bindings.get("MMA_BM") or 0) == 32
        assert int(c.bindings.get("MMA_BN") or 0) == 32
        assert int(c.bindings.get("MMA_BK") or 0) == 16

