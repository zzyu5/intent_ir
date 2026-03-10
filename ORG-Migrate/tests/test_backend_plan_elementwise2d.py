from __future__ import annotations

from org.mapping.cuda.elementwise2d import plan_add2d, plan_exp2d
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org(kernel: str, primitive_tag: str) -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": kernel,
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 256, "N": 2097152},
                "artifacts": {"ttgir_path": f"{kernel}.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "tile resident", "scope": "tile", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "memory_coalescing", "summary": "vector io", "scope": "load_store", "tensors": ["input", "output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "register compute", "scope": "compute", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "wide cta", "scope": "grid", "tensors": ["output"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "blocked_register_layout", "category": "tiling", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads", "vector_width"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "vector_global_io", "category": "mapping", "supports_goals": ["g1"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": primitive_tag, "category": "primitive", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "two_axis_grid_mapping", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [256, 512, 128], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [4, 2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "elementwise_v1",
                "bindings": {"ELEMENTWISE_BLOCK_THREADS": 256, "ELEMENTWISE_VECTOR_WIDTH": 4},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": f"{kernel}.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_add2d_prefers_vectorized_elementwise() -> None:
    plan = plan_add2d(
        _org("add2d", "elementwise_add_primitive"),
        shape_bindings={"M": 256, "N": 2097152},
        source_oracle={"kernel_kind": "elementwise_v1", "bindings": {"ELEMENTWISE_BLOCK_THREADS": 256, "ELEMENTWISE_VECTOR_WIDTH": 4}},
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
    assert plan.candidates[0].kernel_kind == "elementwise_v1"
    assert plan.candidates[0].bindings == {
        "ELEMENTWISE_BLOCK_THREADS": 256,
        "ELEMENTWISE_VECTOR_WIDTH": 4,
    }
    assert any(module.id == "elementwise_add_vector_global_io" for module in plan.selected_modules)


def test_backend_plan_exp2d_prefers_vectorized_elementwise() -> None:
    plan = plan_exp2d(
        _org("exp2d", "elementwise_exp_primitive"),
        shape_bindings={"M": 256, "N": 2097152},
        source_oracle={"kernel_kind": "elementwise_v1", "bindings": {"ELEMENTWISE_BLOCK_THREADS": 256, "ELEMENTWISE_VECTOR_WIDTH": 4}},
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
    assert plan.candidates[0].kernel_kind == "elementwise_v1"
    assert plan.candidates[0].bindings == {
        "ELEMENTWISE_BLOCK_THREADS": 256,
        "ELEMENTWISE_VECTOR_WIDTH": 4,
    }
    assert any(module.id == "elementwise_exp_vector_global_io" for module in plan.selected_modules)
