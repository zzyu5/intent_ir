from __future__ import annotations

from org.mapping.cuda.universal_planner import plan_cuda_kernel
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def test_universal_planner_unknown_kernel_uses_graph_fallback() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "liger_swiglu",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 128, "N": 1024},
                "artifacts": {"ttgir_path": "liger_swiglu.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "cta tiles stay resident", "scope": "tile", "tensors": ["A", "B", "C"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "memory_coalescing", "summary": "vector io", "scope": "load_store", "tensors": ["A", "B", "C"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "blocked_register_layout", "category": "tiling", "supports_goals": ["g0"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "vector_global_io", "category": "mapping", "supports_goals": ["g1"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "vector_width", "role": "vector_width", "candidates": [4, 2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "",
                "bindings": {},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e0"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "liger_swiglu.ttgir:1", "summary": "elementwise graph"},
            ],
        }
    )
    plan = plan_cuda_kernel(
        "liger_swiglu",
        org,
        shape_bindings={"M": 128, "N": 1024},
        source_oracle={"kernel_kind": "", "bindings": {}},
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
    assert plan.candidates
    assert plan.candidates[0].kernel_kind == "elementwise_v1"
    assert any("family_inferred=add2d" in str(note) for note in list(plan.notes or []))


def test_universal_planner_unknown_rms_norm_uses_graph_fallback() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "liger_rms_norm",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 128, "N": 64},
                "artifacts": {"ttgir_path": "liger_rms_norm.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row state resident", "scope": "row", "tensors": ["X", "RSTD", "Y"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "affine_epilogue_fusion", "summary": "fuse row normalization with affine scale", "scope": "epilogue", "tensors": ["Y"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["row_width"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "warp_statistics", "category": "communication", "supports_goals": ["g0"], "attrs": {}, "dims": ["row_width"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "affine_epilogue", "category": "fusion", "supports_goals": ["g1"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "row_width", "role": "row_width", "candidates": [64], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "X", "role": "input_row", "shape_refs": ["M", "N"], "evidence_refs": ["e0"]},
                {"id": "t1", "name": "RSTD", "role": "rstd", "shape_refs": ["M"], "evidence_refs": ["e0"]},
                {"id": "t2", "name": "Y", "role": "affine_out", "shape_refs": ["M", "N"], "evidence_refs": ["e0"]},
            ],
            "tensor_lifetimes": [
                {
                    "id": "lt0",
                    "tensor": "t0",
                    "region": "row_reduce",
                    "storage": "register",
                    "start": "load_x",
                    "end": "epilogue",
                    "producer_mechanisms": ["m0"],
                    "consumer_mechanisms": ["m1", "m2"],
                    "supports_goals": ["g0"],
                    "dims": ["row_width"],
                    "bytes_hint": 256,
                    "reuse_window": "full_row",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t1",
                    "region": "row_reduce",
                    "storage": "register",
                    "start": "rsqrt",
                    "end": "epilogue",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g0"],
                    "dims": ["row_width"],
                    "bytes_hint": 4,
                    "reuse_window": "row_epilogue",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt2",
                    "tensor": "t2",
                    "region": "affine_epilogue",
                    "storage": "register",
                    "start": "mul_weight",
                    "end": "store",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g1"],
                    "dims": ["row_width"],
                    "bytes_hint": 256,
                    "reuse_window": "row_epilogue",
                    "evidence_refs": ["e0"],
                },
            ],
            "source_oracle": {
                "kernel_kind": "",
                "bindings": {},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e0"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "liger_rms_norm.ttgir:1", "summary": "rowwise rms norm graph"},
            ],
        }
    )
    plan = plan_cuda_kernel(
        "liger_rms_norm",
        org,
        shape_bindings={"M": 128, "N": 64},
        source_oracle={"kernel_kind": "", "bindings": {}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={},
        budget=4,
    )
    assert plan.candidates
    assert plan.candidates[0].kernel_kind == "rms_norm_axis1_v3"
    assert any("family_inferred=rms_norm2d" in str(note) for note in list(plan.notes or []))
