from __future__ import annotations

from org.mapping.cuda.row_softmax import plan_softmax_inner
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_softmax_inner() -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "softmax_inner",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "softmax_inner.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row tile resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "row reduction", "scope": "softmax", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid extra buffer", "scope": "softmax", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "vector row path", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [64, 128], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "row_softmax_axis1_triton_v1",
                "bindings": {"SOFTMAX_BLOCK_THREADS": 64},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "softmax_inner.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_softmax_inner_chain() -> None:
    plan = plan_softmax_inner(
        _org_softmax_inner(),
        shape_bindings={"M": 4, "N": 64},
        source_oracle={"kernel_kind": "row_softmax_axis1_triton_v1", "bindings": {"SOFTMAX_BLOCK_THREADS": 64}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "staging.row_tile_resident": {"present": True, "attrs": {"resident_bytes_hint": 256}},
                "communication.row_reduction": {"present": True, "attrs": {"reduction_scope": "warp"}},
                "layout.vector_row_path": {"present": True, "attrs": {"vector_row_path": True}},
            }
        },
        budget=4,
    )
    assert plan.candidates[0].kernel_kind == "row_softmax_axis1_triton_v1"
    assert plan.candidates[0].bindings == {"SOFTMAX_BLOCK_THREADS": 64}
    assert any(c.kernel_kind == "row_softmax_axis1_v1" for c in plan.candidates)
