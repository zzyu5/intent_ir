from __future__ import annotations

from org.mapping.cuda.flash_attention2d import plan_flash_attention2d
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_flash(*, kv_storage: str = "shared", kv_bytes_hint: int = 16384, pipeline: bool = True) -> object:
    mechanism_topology = [
        {"id": "mt0", "src": "m0", "dst": "m2", "relation": "feeds", "tensors": ["t0", "t3", "t4", "t5"], "lifetimes": ["lt0", "lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
        {"id": "mt1", "src": "m1", "dst": "m2", "relation": "feeds", "tensors": ["t1", "t2", "t3", "t4", "t5"], "lifetimes": ["lt1", "lt2", "lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
        {"id": "mt2", "src": "m2", "dst": "m4", "relation": "feeds", "tensors": ["t3", "t4", "t5"], "lifetimes": ["lt3", "lt4", "lt5"], "evidence_refs": ["e0"]},
    ]
    if pipeline:
        mechanism_topology.extend(
            [
                {"id": "mt3", "src": "m1", "dst": "m3", "relation": "gates", "tensors": ["t1", "t2"], "lifetimes": ["lt1", "lt2"], "evidence_refs": ["e0"]},
                {"id": "mt4", "src": "m3", "dst": "m1", "relation": "feeds", "tensors": ["t1", "t2"], "lifetimes": ["lt1", "lt2"], "evidence_refs": ["e0"]},
            ]
        )
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
                "artifacts": {"ttgir_path": "flash.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep Q and output accumulator resident through the kv loop", "scope": "kv_loop", "tensors": ["Q", "Out"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "carry online max/sum state across streamed K/V tiles", "scope": "softmax", "tensors": ["max_state", "sum_state"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid materializing a full score matrix", "scope": "scores", "tensors": ["max_state", "sum_state"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "prefetch the next streamed K/V tile", "scope": "kv_loop", "tensors": ["K", "V"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "q_resident_state", "category": "staging", "supports_goals": ["g0"], "attrs": {"communication_scope": "cta"}, "dims": ["resident_bytes"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "kv_streamed_tiles", "category": "staging", "supports_goals": ["g0", "g3"], "attrs": {"communication_scope": "kv_loop"}, "dims": ["tile_kv", "resident_bytes", "pipeline_stages"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {"communication_scope": "warp"}, "dims": ["score_warps"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {"pipeline_depth": 2}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
                {"id": "m4", "tag": "output_layout_convert", "category": "fusion", "supports_goals": ["g0"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "tile_kv", "role": "kv_tile", "candidates": [32, 64], "constraints": ["tile_kv <= KV_CTX"], "evidence_refs": ["e0"]},
                {"name": "score_warps", "role": "score_reduce", "candidates": [4, 6], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": ([2] if pipeline else [1]), "constraints": [], "evidence_refs": ["e0"]},
                {"name": "resident_bytes", "role": "resident_budget", "candidates": [33024], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "tensors": [
                {"id": "t0", "name": "Q", "role": "query_state", "shape_refs": ["HEAD_DIM"], "evidence_refs": ["e0"]},
                {"id": "t1", "name": "K", "role": "key_tile", "shape_refs": ["tile_kv", "HEAD_DIM"], "evidence_refs": ["e0"]},
                {"id": "t2", "name": "V", "role": "value_tile", "shape_refs": ["tile_kv", "HEAD_DIM"], "evidence_refs": ["e0"]},
                {"id": "t3", "name": "max_state", "role": "softmax_max", "evidence_refs": ["e0"]},
                {"id": "t4", "name": "sum_state", "role": "softmax_sum", "evidence_refs": ["e0"]},
                {"id": "t5", "name": "Out", "role": "output_accumulator", "shape_refs": ["HEAD_DIM"], "evidence_refs": ["e0"]},
            ],
            "tensor_lifetimes": [
                {
                    "id": "lt0",
                    "tensor": "t0",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "load_q",
                    "end": "kv_loop_exit",
                    "producer_mechanisms": ["m0"],
                    "consumer_mechanisms": ["m2"],
                    "supports_goals": ["g0"],
                    "dims": ["resident_bytes"],
                    "bytes_hint": 256,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt1",
                    "tensor": "t1",
                    "region": "kv_loop",
                    "storage": str(kv_storage),
                    "start": "load_k_tile",
                    "end": "softmax_update",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": ["m2", "m3"],
                    "supports_goals": ["g0", "g3"],
                    "dims": ["tile_kv", "resident_bytes", "pipeline_stages"],
                    "bytes_hint": int(kv_bytes_hint),
                    "reuse_window": "cta_tile",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt2",
                    "tensor": "t2",
                    "region": "kv_loop",
                    "storage": str(kv_storage),
                    "start": "load_v_tile",
                    "end": "softmax_update",
                    "producer_mechanisms": ["m1"],
                    "consumer_mechanisms": ["m2", "m3"],
                    "supports_goals": ["g0", "g3"],
                    "dims": ["tile_kv", "resident_bytes", "pipeline_stages"],
                    "bytes_hint": int(kv_bytes_hint),
                    "reuse_window": "cta_tile",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt3",
                    "tensor": "t3",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "reduce_max",
                    "end": "softmax_update",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": ["m2", "m4"],
                    "supports_goals": ["g1", "g2"],
                    "dims": ["score_warps"],
                    "bytes_hint": 4,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt4",
                    "tensor": "t4",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "reduce_sum",
                    "end": "normalize",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": ["m2", "m4"],
                    "supports_goals": ["g1", "g2"],
                    "dims": ["score_warps"],
                    "bytes_hint": 4,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
                {
                    "id": "lt5",
                    "tensor": "t5",
                    "region": "kv_loop",
                    "storage": "register",
                    "start": "softmax_update",
                    "end": "store",
                    "producer_mechanisms": ["m2"],
                    "consumer_mechanisms": ["m4"],
                    "supports_goals": ["g0", "g1"],
                    "dims": ["resident_bytes"],
                    "bytes_hint": 256,
                    "reuse_window": "kv_loop",
                    "evidence_refs": ["e0"],
                },
            ],
            "dataflow_edges": [
                {"id": "df0", "src": "lt0", "dst": "lt3", "tensor": "t3", "kind": "reduce", "order": 0, "mechanisms": ["m2"], "evidence_refs": ["e0"]},
                {"id": "df1", "src": "lt1", "dst": "lt3", "tensor": "t3", "kind": "stage", "order": 1, "mechanisms": ["m1", "m2"], "evidence_refs": ["e0"]},
                {"id": "df2", "src": "lt2", "dst": "lt5", "tensor": "t5", "kind": "update", "order": 2, "mechanisms": ["m1", "m2"], "evidence_refs": ["e0"]},
                {"id": "df3", "src": "lt3", "dst": "lt4", "tensor": "t4", "kind": "normalize", "order": 3, "mechanisms": ["m2"], "evidence_refs": ["e0"]},
                {"id": "df4", "src": "lt4", "dst": "lt5", "tensor": "t5", "kind": "update", "order": 4, "mechanisms": ["m2"], "evidence_refs": ["e0"]},
            ],
            "mechanism_topology": mechanism_topology,
            "source_oracle": {
                "kernel_kind": "attn2d_causal_softmax_v6",
                "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_flash_attention2d_prefers_shared_stage_graph_on_sm120() -> None:
    plan = plan_flash_attention2d(
        _org_flash(),
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        source_oracle={"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "staging.q_resident_state": {"present": True, "attrs": {"resident_bytes_hint": 256}},
                "staging.kv_streamed_tiles": {"present": True, "attrs": {"resident_bytes_hint": 16384}},
                "pipeline.stage_hint": {"present": True, "attrs": {"pipeline_depth_hint": 2}},
            }
        },
        ptx_facts={"mechanisms": {"pipeline.async_copy": {"present": True, "attrs": {"complete_async_pipeline": True}}}},
        toolchain_model={"effective_sm": "sm_120", "downleveled": False},
        budget=12,
    )
    assert plan.selected_modules
    assert plan.candidates[0].kernel_kind == "attn2d_causal_softmax_v8"
    assert plan.candidates[0].bindings == {"ATTN_BLOCK_KV": 32, "FLASH_KV_SHARED_STAGE": 1}
    selected_ids = {module.id for module in plan.selected_modules}
    assert "kv_shared_stage" in selected_ids
    assert "q_resident_state" in selected_ids
    assert "kv_tile_stage" in selected_ids
    assert "online_softmax_reduce" in selected_ids
    assert "topology_mode=graph" in plan.notes
    assert "topology_shared_stage_path=True" in plan.notes
    assert "topology_pipeline_depth=2" in plan.notes
    assert any("topology_shared_stage_fit" in str(c.score_reason) for c in plan.candidates if c.kernel_kind == "attn2d_causal_softmax_v8")


def test_backend_plan_flash_attention2d_disables_shared_stage_when_topology_exceeds_budget() -> None:
    plan = plan_flash_attention2d(
        _org_flash(kv_bytes_hint=65536),
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        source_oracle={"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "staging.q_resident_state": {"present": True, "attrs": {"resident_bytes_hint": 256}},
                "staging.kv_streamed_tiles": {"present": True, "attrs": {"resident_bytes_hint": 16384}},
                "pipeline.stage_hint": {"present": True, "attrs": {"pipeline_depth_hint": 2}},
            }
        },
        ptx_facts={"mechanisms": {"pipeline.async_copy": {"present": True, "attrs": {"complete_async_pipeline": True}}}},
        toolchain_model={"effective_sm": "sm_120", "downleveled": False},
        budget=12,
    )
    selected_ids = {module.id for module in plan.selected_modules}
    assert "kv_shared_stage" not in selected_ids
    assert plan.candidates[0].bindings.get("FLASH_KV_SHARED_STAGE", 0) == 0
    assert "topology_any_shared_stage_fit=False" in plan.notes


def test_backend_plan_flash_attention2d_large_smem_prefers_async_v7_front() -> None:
    plan = plan_flash_attention2d(
        _org_flash(),
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        source_oracle={"kernel_kind": "attn2d_causal_softmax_v7", "bindings": {"ATTN_BLOCK_KV": 64}},
        hardware_model=build_hardware_model(target="cuda_h100", arch="sm90"),
        ttgir_facts={
            "mechanisms": {
                "staging.q_resident_state": {"present": True, "attrs": {"resident_bytes_hint": 256}},
                "staging.kv_streamed_tiles": {"present": True, "attrs": {"resident_bytes_hint": 16384}},
                "pipeline.stage_hint": {"present": True, "attrs": {"pipeline_depth_hint": 2}},
            }
        },
        ptx_facts={"mechanisms": {"pipeline.async_copy": {"present": True, "attrs": {"complete_async_pipeline": True}}}},
        toolchain_model={"effective_sm": "sm_90", "downleveled": False},
        budget=12,
    )
    assert plan.hardware_model["arch_cluster"] == "cuda_tc_large_smem"
    assert plan.candidates[0].kernel_kind == "attn2d_causal_softmax_v7"
    assert plan.candidates[0].bindings.get("FLASH_ATTN_ASYNC_COPY") == 1
    assert "topology_pipeline_path=True" in plan.notes
