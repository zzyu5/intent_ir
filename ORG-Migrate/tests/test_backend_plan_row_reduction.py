from __future__ import annotations

from org.mapping.cuda.row_reduction import plan_row_max, plan_row_sum
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_row_sum() -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "row_sum",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 256},
                "artifacts": {"ttgir_path": "row_sum.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row tile resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "reduction_tree_balance", "summary": "balanced reduction tree", "scope": "reduce", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector row loads", "scope": "load", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g2"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "shared_staging", "category": "staging", "supports_goals": ["g0", "g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [128, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "row_sum_axis1_v2",
                "bindings": {"ROW_REDUCE_BLOCK_THREADS": 128, "ROW_REDUCE_VECTOR_WIDTH": 2, "ROW_REDUCE_SHARED_STAGE": 1},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "row_sum.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def _org_row_max() -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "row_max",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 256},
                "artifacts": {"ttgir_path": "row_max.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row tile resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "reduction_tree_balance", "summary": "balanced warp max tree", "scope": "reduce", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector row loads", "scope": "load", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "tile_load_stage", "category": "mapping", "supports_goals": ["g2"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "warp_reduction_tree", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "block_synchronization", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [128, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "row_max_axis1_v2",
                "bindings": {"ROW_REDUCE_BLOCK_THREADS": 128, "ROW_REDUCE_VECTOR_WIDTH": 2, "ROW_REDUCE_SHARED_STAGE": 1},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "row_max.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_row_sum_prefers_shared_vector_tree() -> None:
    plan = plan_row_sum(
        _org_row_sum(),
        shape_bindings={"M": 4, "N": 256},
        source_oracle={"kernel_kind": "row_sum_axis1_v2", "bindings": {"ROW_REDUCE_BLOCK_THREADS": 128, "ROW_REDUCE_VECTOR_WIDTH": 2, "ROW_REDUCE_SHARED_STAGE": 1}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "tiling.blocked_layout": {
                    "present": True,
                    "attrs": {"layouts": [{"size_per_thread": [2], "threads_per_warp_layout": [32], "warps_per_cta": [4]}]},
                },
                "communication.reduction": {"present": True, "attrs": {"reduction_scope": "warp"}},
            }
        },
        budget=4,
    )
    assert plan.candidates[0].kernel_kind == "row_sum_axis1_v2"
    assert plan.candidates[0].bindings == {
        "ROW_REDUCE_BLOCK_THREADS": 128,
        "ROW_REDUCE_VECTOR_WIDTH": 2,
        "ROW_REDUCE_SHARED_STAGE": 1,
    }
    assert any(module.id == "row_sum_shared_warp_exchange" for module in plan.selected_modules)


def test_backend_plan_row_max_prefers_shared_vector_tree() -> None:
    plan = plan_row_max(
        _org_row_max(),
        shape_bindings={"M": 4, "N": 256},
        source_oracle={"kernel_kind": "row_max_axis1_v2", "bindings": {"ROW_REDUCE_BLOCK_THREADS": 128, "ROW_REDUCE_VECTOR_WIDTH": 2, "ROW_REDUCE_SHARED_STAGE": 1}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "tiling.blocked_layout": {
                    "present": True,
                    "attrs": {"layouts": [{"size_per_thread": [2], "threads_per_warp_layout": [32], "warps_per_cta": [4]}]},
                },
                "communication.reduction": {"present": True, "attrs": {"reduction_scope": "warp"}},
            }
        },
        budget=4,
    )
    assert plan.candidates[0].kernel_kind == "row_max_axis1_v2"
    assert plan.candidates[0].bindings == {
        "ROW_REDUCE_BLOCK_THREADS": 128,
        "ROW_REDUCE_VECTOR_WIDTH": 2,
        "ROW_REDUCE_SHARED_STAGE": 1,
    }
    assert any(module.id == "row_max_shared_warp_exchange" for module in plan.selected_modules)
