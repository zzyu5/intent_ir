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


def test_universal_planner_full_row_rms_norm_prefers_vector_resident_candidate() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "liger_rms_norm",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 2048, "N": 32768},
                "artifacts": {"ttgir_path": "liger_rms_norm.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep the full row live in registers", "scope": "row", "tensors": ["X", "Y"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "affine_epilogue_fusion", "summary": "fuse scaling into writeback", "scope": "epilogue", "tensors": ["Y"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector row io", "scope": "load_store", "tensors": ["X", "W", "Y"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["RMS_NORM_BLOCK_THREADS"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "warp_statistics", "category": "communication", "supports_goals": ["g0"], "attrs": {}, "dims": ["RMS_NORM_BLOCK_THREADS"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "register_staging", "category": "primitive", "supports_goals": ["g0"], "attrs": {}, "dims": ["RMS_NORM_BLOCK_THREADS", "RMS_NORM_VECTOR_WIDTH"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "affine_epilogue", "category": "fusion", "supports_goals": ["g1"], "attrs": {}, "dims": ["RMS_NORM_VECTOR_WIDTH"], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g2"], "attrs": {}, "dims": ["RMS_NORM_VECTOR_WIDTH"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "RMS_NORM_BLOCK_THREADS", "role": "threads", "candidates": [256, 128, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "RMS_NORM_VECTOR_WIDTH", "role": "vector_width", "candidates": [4, 2, 1], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "X", "role": "input_row", "shape_refs": ["M", "N"], "layout": "blocked", "evidence_refs": ["e0"]},
                {"id": "t1", "name": "W", "role": "weight_row", "shape_refs": ["N"], "layout": "blocked", "evidence_refs": ["e0"]},
                {"id": "t2", "name": "RSTD", "role": "rstd", "shape_refs": ["M"], "layout": "scalar", "evidence_refs": ["e0"]},
                {"id": "t3", "name": "Y", "role": "affine_out", "shape_refs": ["M", "N"], "layout": "blocked", "evidence_refs": ["e0"]},
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
                    "consumer_mechanisms": ["m1", "m2", "m3"],
                    "supports_goals": ["g0", "g2"],
                    "dims": ["RMS_NORM_BLOCK_THREADS", "RMS_NORM_VECTOR_WIDTH"],
                    "bytes_hint": 16384,
                    "reuse_window": "full_row",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t1",
                    "region": "affine_epilogue",
                    "storage": "register",
                    "start": "load_w",
                    "end": "epilogue",
                    "producer_mechanisms": ["m0"],
                    "consumer_mechanisms": ["m3"],
                    "supports_goals": ["g1", "g2"],
                    "dims": ["RMS_NORM_VECTOR_WIDTH"],
                    "bytes_hint": 16384,
                    "reuse_window": "row_epilogue",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt2",
                    "tensor": "t2",
                    "region": "row_reduce",
                    "storage": "register",
                    "start": "rsqrt",
                    "end": "store_rstd",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": ["m3"],
                    "supports_goals": ["g0"],
                    "bytes_hint": 4,
                    "reuse_window": "row_epilogue",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt3",
                    "tensor": "t3",
                    "region": "affine_epilogue",
                    "storage": "register",
                    "start": "mul_weight",
                    "end": "store",
                    "producer_mechanisms": ["m3"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g1", "g2"],
                    "dims": ["RMS_NORM_VECTOR_WIDTH"],
                    "bytes_hint": 16384,
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
                {"id": "e0", "kind": "ttgir_line", "path": "liger_rms_norm.ttgir:1", "summary": "full-row rms norm graph"},
            ],
        }
    )
    plan = plan_cuda_kernel(
        "liger_rms_norm",
        org,
        shape_bindings={"M": 2048, "N": 32768},
        source_oracle={"kernel_kind": "", "bindings": {}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={},
        budget=8,
    )
    assert plan.candidates
    assert plan.candidates[0].kernel_kind == "rms_norm_axis1_v4"
    assert int(plan.candidates[0].bindings.get("RMS_NORM_FULL_ROW_VECTOR") or 0) == 1
    assert int(plan.candidates[0].bindings.get("RMS_NORM_BLOCK_THREADS") or 0) == 256


def test_universal_planner_cfg_cross_entropy_uses_region_max_path() -> None:
    org = validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "liger_cross_entropy",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"BT": 128, "V": 256},
                "artifacts": {"ttgir_path": "liger_cross_entropy.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "streaming_softmax_state", "summary": "online row state", "scope": "row", "tensors": ["input", "state"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "reduction_tree_balance", "summary": "stable row reductions", "scope": "row", "tensors": ["input", "loss"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "resident_working_set", "summary": "keep row tile resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0", "g2"], "attrs": {}, "dims": ["CE_BLOCK_THREADS"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "row_reduction", "category": "communication", "supports_goals": ["g0", "g1"], "attrs": {}, "dims": ["CE_BLOCK_THREADS"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "label_gather", "category": "primitive", "supports_goals": ["g1"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "branch_mask", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "loss_finalize", "category": "fusion", "supports_goals": ["g1"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "CE_BLOCK_THREADS", "role": "threads", "candidates": [128], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "input", "role": "input_row", "shape_refs": ["BT", "V"], "evidence_refs": ["e0"]},
                {"id": "t1", "name": "target", "role": "target", "shape_refs": ["BT"], "evidence_refs": ["e0"]},
                {"id": "t2", "name": "state", "role": "max_state", "shape_refs": ["BT"], "evidence_refs": ["e0"]},
                {"id": "t3", "name": "picked", "role": "picked", "shape_refs": ["BT"], "evidence_refs": ["e0"]},
                {"id": "t4", "name": "loss", "role": "output", "shape_refs": [], "evidence_refs": ["e0"]},
            ],
            "tensor_lifetimes": [
                {
                    "id": "lt0",
                    "tensor": "t0",
                    "region": "row",
                    "storage": "shared",
                    "start": "load",
                    "end": "reduce",
                    "producer_mechanisms": ["m0"],
                    "consumer_mechanisms": ["m1", "m2"],
                    "supports_goals": ["g0", "g2"],
                    "dims": ["CE_BLOCK_THREADS"],
                    "bytes_hint": 1024,
                    "reuse_window": "row_tile",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t2",
                    "region": "row",
                    "storage": "register",
                    "start": "reduce",
                    "end": "finalize",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g0"],
                    "bytes_hint": 8,
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt2",
                    "tensor": "t3",
                    "region": "row",
                    "storage": "register",
                    "start": "gather",
                    "end": "finalize",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g1"],
                    "bytes_hint": 4,
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt3",
                    "tensor": "t4",
                    "region": "cfg_ignore",
                    "storage": "register",
                    "start": "branch_ignore",
                    "end": "join",
                    "producer_mechanisms": ["m3"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g1"],
                    "bytes_hint": 4,
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt4",
                    "tensor": "t4",
                    "region": "cfg_active",
                    "storage": "register",
                    "start": "branch_active",
                    "end": "join",
                    "producer_mechanisms": ["m4"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g1"],
                    "bytes_hint": 12,
                    "evidence_refs": ["e0"],
                },
            ],
            "dataflow_edges": [],
            "mechanism_topology": [],
            "schedule_edges": [],
            "region_graph": {
                "regions": [
                    {"id": "r0", "kind": "if", "path_id": "pi_ignore", "predicate": "target == ignore_index", "entry_mechanisms": ["m3"], "exit_mechanisms": ["m4"], "evidence_refs": ["e0"]},
                    {"id": "r1", "kind": "else", "path_id": "pi_active", "predicate": "target != ignore_index", "entry_mechanisms": ["m1", "m2"], "exit_mechanisms": ["m4"], "evidence_refs": ["e0"]},
                ],
                "edges": [
                    {"id": "re0", "src": "r0", "dst": "r1", "relation": "joins", "path_id": "pi_ignore", "lifetimes": ["lt3"], "mechanisms": ["m3"], "evidence_refs": ["e0"]},
                    {"id": "re1", "src": "r1", "dst": "r0", "relation": "joins", "path_id": "pi_active", "lifetimes": ["lt0", "lt1", "lt2", "lt4"], "mechanisms": ["m1", "m2", "m4"], "evidence_refs": ["e0"]},
                ],
            },
            "source_oracle": {
                "kernel_kind": "",
                "bindings": {},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e0"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "liger_cross_entropy.ttgir:1", "summary": "cfg loss graph"},
            ],
        }
    )
    plan = plan_cuda_kernel(
        "liger_cross_entropy",
        org,
        shape_bindings={"BT": 128, "V": 256},
        source_oracle={"kernel_kind": "", "bindings": {}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={},
        budget=4,
    )
    assert plan.candidates
    assert plan.candidates[0].kernel_kind == "cfg_masked_row_reduce_v1"
    assert int(plan.candidates[0].bindings.get("CFG_ROW_BLOCK_THREADS") or 0) in {128, 256, 512, 1024}
    assert int(plan.candidates[0].bindings.get("CFG_ROW_VECTOR_WIDTH") or 0) in {1, 2, 4}
    assert any("family_inferred=cfg_masked_row_reduce2d" in str(note) for note in list(plan.notes or []))
    assert any("topology_cfg_max_path_bytes=1048" in str(note) for note in list(plan.notes or []))
