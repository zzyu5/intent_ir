from __future__ import annotations

from org.mapping.cuda.group_norm_kernel import plan_group_norm_kernel
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_group_norm(*, group_size: int) -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "group_norm_kernel",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"N": 1024, "C": 128, "HW": 256, "num_groups": 128, "group_size": group_size},
                "artifacts": {"ttgir_path": "group_norm_kernel.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep group tile resident", "scope": "tile", "tensors": ["X"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "reduction_tree_balance", "summary": "warp reduction", "scope": "reduce", "tensors": ["Mean", "Rstd"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector group io", "scope": "load_store", "tensors": ["X", "Y"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "fused_epilogue_avoid_writeback", "summary": "fuse affine epilogue", "scope": "epilogue", "tensors": ["W", "B"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "group_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "warp_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "online_normalization", "category": "fusion", "supports_goals": ["g1"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "affine_fused_epilogue", "category": "fusion", "supports_goals": ["g3"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "blocked_layout", "category": "tiling", "supports_goals": ["g0", "g2"], "attrs": {}, "dims": ["block_threads", "vector_width"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [256, 128], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [4, 2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "group_norm_v1",
                "bindings": {"GROUP_NORM_BLOCK_THREADS": 256, "GROUP_NORM_VECTOR_WIDTH": 4 if group_size == 1 else 1},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "group_norm_kernel.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_group_norm_prefers_vector_when_single_channel_group() -> None:
    plan = plan_group_norm_kernel(
        _org_group_norm(group_size=1),
        shape_bindings={"N": 1024, "C": 128, "HW": 256, "num_groups": 128, "group_size": 1},
        source_oracle={"kernel_kind": "group_norm_v1", "bindings": {"GROUP_NORM_BLOCK_THREADS": 256, "GROUP_NORM_VECTOR_WIDTH": 4}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "tiling.blocked_layout": {
                    "present": True,
                    "attrs": {"layouts": [{"size_per_thread": [4], "threads_per_warp_layout": [32], "warps_per_cta": [8]}]},
                }
            }
        },
        budget=4,
    )
    assert plan.candidates[0].bindings == {
        "GROUP_NORM_BLOCK_THREADS": 256,
        "GROUP_NORM_VECTOR_WIDTH": 4,
    }
    assert any(module.id == "group_norm_vector_group_io" for module in plan.selected_modules)


def test_backend_plan_group_norm_scalarizes_multi_channel_group() -> None:
    plan = plan_group_norm_kernel(
        _org_group_norm(group_size=2),
        shape_bindings={"N": 1024, "C": 128, "HW": 256, "num_groups": 64, "group_size": 2},
        source_oracle={"kernel_kind": "group_norm_v1", "bindings": {"GROUP_NORM_BLOCK_THREADS": 256, "GROUP_NORM_VECTOR_WIDTH": 1}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "tiling.blocked_layout": {
                    "present": True,
                    "attrs": {"layouts": [{"size_per_thread": [4], "threads_per_warp_layout": [32], "warps_per_cta": [8]}]},
                }
            }
        },
        budget=4,
    )
    assert plan.candidates[0].bindings["GROUP_NORM_VECTOR_WIDTH"] == 1
