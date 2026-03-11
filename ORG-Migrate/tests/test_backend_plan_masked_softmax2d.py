from __future__ import annotations

from org.mapping.cuda.row_softmax import plan_masked_softmax2d
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_masked_softmax2d() -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "masked_softmax2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "masked_softmax2d.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "row reduction", "scope": "softmax", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "apply mask inline", "scope": "mask", "tensors": ["mask"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "vector row path", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "mask_apply", "category": "communication", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
            ],
            "dims": [],
            "source_oracle": {
                "kernel_kind": "row_masked_softmax_axis1_v1",
                "bindings": {},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "masked_softmax2d.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_masked_softmax2d_chain() -> None:
    plan = plan_masked_softmax2d(
        _org_masked_softmax2d(),
        shape_bindings={"M": 4, "N": 64},
        source_oracle={"kernel_kind": "row_masked_softmax_axis1_v1", "bindings": {}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "staging.row_tile_resident": {"present": True, "attrs": {"resident_bytes_hint": 1024}},
                "communication.row_reduction": {"present": True, "attrs": {"reduction_scope": "warp"}},
                "communication.mask_apply": {"present": True, "attrs": {"mask_apply": True}},
            }
        },
        budget=2,
    )
    assert plan.candidates[0].kernel_kind == "row_masked_softmax_axis1_v1"
