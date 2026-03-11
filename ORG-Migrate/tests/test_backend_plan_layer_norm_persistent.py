from __future__ import annotations

from org.mapping.cuda.layer_norm_persistent import plan_layer_norm_persistent
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_layer_norm() -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "layer_norm_persistent",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "layer_norm_persistent.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "persistent_row_state", "summary": "persist row statistics", "scope": "norm", "tensors": ["mean", "rstd"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "memory_coalescing", "summary": "vector row load", "scope": "load", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "affine_epilogue_fusion", "summary": "fuse affine writeback", "scope": "epilogue", "tensors": ["weight", "bias"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "warp_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "persistent_row_cache", "category": "staging", "supports_goals": ["g1"], "attrs": {}, "dims": ["persistent_row"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "tile_load_stage", "category": "mapping", "supports_goals": ["g2"], "attrs": {}, "dims": ["vector_width"], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "affine_epilogue", "category": "fusion", "supports_goals": ["g3"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m5", "tag": "row_parallel_axis", "category": "mapping", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m6", "tag": "block_synchronization", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_threads", "role": "thread_block", "candidates": [32, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "vector_width", "role": "vector_width", "candidates": [2, 1], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "persistent_row", "role": "persistent_row", "candidates": [1], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "resident_bytes", "role": "resident_bytes", "candidates": [256], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "input", "role": "input_row", "aliases": ["x"], "shape_refs": ["N"], "evidence_refs": ["e0"]},
                {"id": "t1", "name": "row_stats", "role": "row_stats", "aliases": ["mean", "rstd"], "evidence_refs": ["e0"]},
                {"id": "t2", "name": "out", "role": "affine_out", "evidence_refs": ["e0"]},
            ],
            "tensor_lifetimes": [
                {
                    "id": "lt0",
                    "tensor": "t0",
                    "region": "row_reduce",
                    "storage": "shared",
                    "start": "row_load",
                    "end": "row_reduce",
                    "producer_mechanisms": ["m3"],
                    "consumer_mechanisms": ["m0", "m1"],
                    "supports_goals": ["g0", "g2"],
                    "dims": ["vector_width", "resident_bytes"],
                    "bytes_hint": 256,
                    "reuse_window": "row_reduce",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t1",
                    "region": "row_stats",
                    "storage": "shared",
                    "start": "row_reduce",
                    "end": "affine_epilogue",
                    "producer_mechanisms": ["m1", "m2"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g1"],
                    "dims": ["persistent_row"],
                    "bytes_hint": 8,
                    "reuse_window": "row_epilogue",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt2",
                    "tensor": "t2",
                    "region": "affine_epilogue",
                    "storage": "register",
                    "start": "affine_epilogue",
                    "end": "store",
                    "producer_mechanisms": ["m4"],
                    "consumer_mechanisms": [],
                    "supports_goals": ["g3"],
                    "dims": [],
                    "evidence_refs": ["e0"],
                },
            ],
            "dataflow_edges": [
                {"id": "df0", "src": "lt0", "dst": "lt1", "tensor": "t1", "kind": "reduce", "order": 0, "mechanisms": ["m0", "m1"], "evidence_refs": ["e0"]},
                {"id": "df1", "src": "lt1", "dst": "lt2", "tensor": "t2", "kind": "epilogue", "order": 1, "mechanisms": ["m2", "m4"], "evidence_refs": ["e0"]},
            ],
            "mechanism_topology": [
                {"id": "mt0", "src": "m3", "dst": "m0", "relation": "vectorizes", "tensors": ["t0"], "lifetimes": ["lt0"], "evidence_refs": ["e0"]},
                {"id": "mt1", "src": "m0", "dst": "m2", "relation": "feeds", "tensors": ["t0", "t1"], "lifetimes": ["lt0", "lt1"], "evidence_refs": ["e0"]},
                {"id": "mt2", "src": "m1", "dst": "m2", "relation": "feeds", "tensors": ["t1"], "lifetimes": ["lt1"], "evidence_refs": ["e0"]},
                {"id": "mt3", "src": "m2", "dst": "m4", "relation": "feeds", "tensors": ["t1", "t2"], "lifetimes": ["lt1", "lt2"], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "layer_norm_axis1_v1",
                "bindings": {"LAYER_NORM_BLOCK_THREADS": 32, "LAYER_NORM_VECTOR_WIDTH": 2, "LAYER_NORM_PERSISTENT_ROW": 1},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "layer_norm_persistent.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_layer_norm_prefers_persistent_vector_row() -> None:
    plan = plan_layer_norm_persistent(
        _org_layer_norm(),
        shape_bindings={"M": 4, "N": 64},
        source_oracle={"kernel_kind": "layer_norm_axis1_v1", "bindings": {"LAYER_NORM_BLOCK_THREADS": 32, "LAYER_NORM_VECTOR_WIDTH": 2, "LAYER_NORM_PERSISTENT_ROW": 1}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "tiling.blocked_layout": {
                    "present": True,
                    "attrs": {"layouts": [{"size_per_thread": [4], "threads_per_warp_layout": [32], "warps_per_cta": [4]}]},
                }
            }
        },
        budget=4,
    )
    assert plan.candidates[0].bindings == {
        "LAYER_NORM_BLOCK_THREADS": 32,
        "LAYER_NORM_VECTOR_WIDTH": 2,
        "LAYER_NORM_PERSISTENT_ROW": 1,
    }
    selected_ids = {module.id for module in plan.selected_modules}
    assert "layer_norm_persistent_row_cache" in selected_ids
    assert "layer_norm_register_stage" in selected_ids
    assert "topology_mode=graph" in plan.notes


def test_backend_plan_layer_norm_disables_persistent_when_topology_exceeds_shared_budget() -> None:
    payload = _org_layer_norm().to_json_dict()
    payload["source_context"]["shape_bindings"]["N"] = 2097152
    payload["dims"] = [
        {"name": "block_threads", "role": "thread_block", "candidates": [128, 256], "constraints": [], "evidence_refs": ["e0"]},
        {"name": "vector_width", "role": "vector_width", "candidates": [2, 4], "constraints": [], "evidence_refs": ["e0"]},
        {"name": "persistent_row", "role": "persistent_row", "candidates": [1], "constraints": [], "evidence_refs": ["e0"]},
        {"name": "resident_bytes", "role": "resident_bytes", "candidates": [8388608], "constraints": [], "evidence_refs": ["e0"]},
    ]
    payload["tensor_lifetimes"][0]["bytes_hint"] = 8388608
    org = validate_org_doc(payload)
    plan = plan_layer_norm_persistent(
        org,
        shape_bindings={"M": 256, "N": 2097152},
        source_oracle={"kernel_kind": "layer_norm_axis1_v1", "bindings": {"LAYER_NORM_BLOCK_THREADS": 32, "LAYER_NORM_VECTOR_WIDTH": 2, "LAYER_NORM_PERSISTENT_ROW": 1}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "tiling.blocked_layout": {
                    "present": True,
                    "attrs": {"layouts": [{"size_per_thread": [4], "threads_per_warp_layout": [32], "warps_per_cta": [4]}]},
                }
            }
        },
        budget=4,
    )
    assert plan.candidates[0].bindings["LAYER_NORM_PERSISTENT_ROW"] == 0
    selected_ids = {module.id for module in plan.selected_modules}
    assert "layer_norm_persistent_row_cache" not in selected_ids


def test_backend_plan_layer_norm_allows_tile_scoped_persistent_when_reuse_window_caps_shared_fit() -> None:
    payload = _org_layer_norm().to_json_dict()
    payload["source_context"]["shape_bindings"]["N"] = 16384
    payload["dims"] = [
        {"name": "block_threads", "role": "thread_block", "candidates": [128, 256], "constraints": [], "evidence_refs": ["e0"]},
        {"name": "vector_width", "role": "vector_width", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
        {"name": "persistent_row", "role": "persistent_row", "candidates": [1], "constraints": [], "evidence_refs": ["e0"]},
        {"name": "resident_bytes", "role": "resident_bytes", "candidates": [8388608], "constraints": [], "evidence_refs": ["e0"]},
    ]
    payload["tensor_lifetimes"][0]["bytes_hint"] = 8388608
    payload["tensor_lifetimes"][0]["reuse_window"] = "cta_tile"
    org = validate_org_doc(payload)
    plan = plan_layer_norm_persistent(
        org,
        shape_bindings={"M": 256, "N": 16384},
        source_oracle={"kernel_kind": "layer_norm_axis1_v1", "bindings": {"LAYER_NORM_BLOCK_THREADS": 32, "LAYER_NORM_VECTOR_WIDTH": 2, "LAYER_NORM_PERSISTENT_ROW": 1}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "tiling.blocked_layout": {
                    "present": True,
                    "attrs": {"layouts": [{"size_per_thread": [4], "threads_per_warp_layout": [32], "warps_per_cta": [4]}]},
                }
            }
        },
        budget=4,
    )
    assert plan.candidates[0].bindings["LAYER_NORM_PERSISTENT_ROW"] == 1
    selected_ids = {module.id for module in plan.selected_modules}
    assert "layer_norm_persistent_row_cache" in selected_ids
    assert "topology_resident_window_scope=tile" in plan.notes
