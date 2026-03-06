from __future__ import annotations

from org.mapping.cuda.ai_bench_softmax import plan_ai_bench_softmax
from org.schema import validate_org_doc


def _make_org(*, dims) -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "ai_bench_softmax",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "tiling",
                    "why": [],
                    "how": [],
                    "dims": dims,
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        }
    )


def test_mapper_includes_thread_overrides_and_triton_like_when_c_le_1024() -> None:
    org = _make_org(dims=["SOFTMAX_BLOCK_THREADS"])
    plan = plan_ai_bench_softmax(
        org,
        shape_bindings={"R": 1823, "C": 781},
        target="cuda_5090d",
        budget=32,
    )
    assert plan.candidates
    assert any(c.kernel_kind == "row_softmax_axis1_triton_v1" for c in plan.candidates)
    assert any(c.kernel_kind == "row_softmax_axis1_v1" and int(c.bindings.get("SOFTMAX_BLOCK_THREADS") or 0) == 256 for c in plan.candidates)


def test_mapper_emits_vec4_candidate_when_requested_and_feasible() -> None:
    org = _make_org(
        dims=[
            {"name": "SOFTMAX_BLOCK_THREADS", "allowed": [256]},
            {"name": "SOFTMAX_VEC4", "allowed": [1]},
        ]
    )
    plan = plan_ai_bench_softmax(
        org,
        shape_bindings={"R": 1823, "C": 781},
        target="cuda_5090d",
        budget=32,
    )
    assert plan.candidates
    c0 = plan.candidates[0]
    assert c0.kernel_kind == "row_softmax_axis1_v1"
    assert int(c0.bindings.get("SOFTMAX_BLOCK_THREADS") or 0) == 256
    assert int(c0.bindings.get("SOFTMAX_VEC4") or 0) == 1

