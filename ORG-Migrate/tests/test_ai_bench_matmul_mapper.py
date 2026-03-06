from __future__ import annotations

from org.mapping.cuda.ai_bench_matmul import plan_ai_bench_matmul
from org.schema import validate_org_doc


def _make_org(*, node_type: str, why: list[str] | None = None, how: list[str] | None = None):
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "ai_bench_matmul",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": str(node_type),
                    "why": list(why or []),
                    "how": list(how or []),
                    "dims": ["MMA_BM", "MMA_BN", "MMA_BK"],
                    "constraints": ["MMA_BM%16==0", "MMA_BN%16==0", "MMA_BK%8==0"],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )


def test_matmul_mapper_budget_and_dedupe_stable() -> None:
    org = _make_org(node_type="special_primitive")
    plan = plan_ai_bench_matmul(
        org,
        shape_bindings={"M": 256, "N": 512, "K": 256},
        target="cuda_5090d",
        budget=5,
    )
    assert len(plan.candidates) <= 5
    keys = {
        (c.kernel_kind, tuple(sorted((k, int(v)) for k, v in dict(c.bindings or {}).items())))
        for c in plan.candidates
    }
    assert len(keys) == len(plan.candidates)


def test_matmul_mapper_want_async_prioritizes_v2() -> None:
    org = _make_org(node_type="overlap_pipeline", how=[])
    plan = plan_ai_bench_matmul(
        org,
        shape_bindings={"M": 256, "N": 512, "K": 256},
        target="cuda_5090d",
        budget=32,
    )
    assert plan.candidates
    c0 = plan.candidates[0]
    assert c0.kernel_kind == "matmul_mma_tf32_v2"
    assert int(c0.bindings.get("MMA_ASYNC_COPY") or 0) == 1


def test_matmul_mapper_without_async_still_emits_v2() -> None:
    org = _make_org(node_type="special_primitive")
    plan = plan_ai_bench_matmul(
        org,
        shape_bindings={"M": 256, "N": 512, "K": 256},
        target="cuda_5090d",
        budget=2,
    )
    assert len(plan.candidates) == 2
    assert plan.candidates[0].kernel_kind == "matmul_mma_tf32_global_v1"
    assert plan.candidates[1].kernel_kind == "matmul_mma_tf32_v2"


def test_matmul_mapper_empty_when_k_not_multiple_of_8() -> None:
    org = _make_org(node_type="special_primitive")
    plan = plan_ai_bench_matmul(
        org,
        shape_bindings={"M": 256, "N": 512, "K": 250},
        target="cuda_5090d",
        budget=32,
    )
    assert plan.candidates == []
    subs = list((plan.trace or {}).get("substitutions") or [])
    assert any(str(s.get("to") or "") == "backend.skip" for s in subs)


def test_matmul_mapper_respects_org_dim_allowed_sets() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "ai_bench_matmul",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "special_primitive",
                    "why": [],
                    "how": [],
                    "dims": [{"name": "MMA_BK", "allowed": [32]}],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )
    plan = plan_ai_bench_matmul(
        org,
        shape_bindings={"M": 256, "N": 512, "K": 256},
        target="cuda_5090d",
        budget=32,
    )
    assert plan.candidates
    assert list(plan.param_space.get("MMA_BK") or []) == [32]
    for c in plan.candidates:
        assert int(c.bindings.get("MMA_BK") or 0) == 32
