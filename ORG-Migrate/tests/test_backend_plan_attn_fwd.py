from __future__ import annotations

from org.mapping.cuda.attn_fwd import plan_attn_fwd
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_attn_fwd() -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "_attn_fwd",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
                "artifacts": {"ttgir_path": "attn_fwd.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep q/state resident", "scope": "q_state", "tensors": ["Q"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "online reduce", "scope": "softmax", "tensors": ["Out"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid full score matrix", "scope": "scores", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "pipeline kv tiles", "scope": "kv_loop", "tensors": ["K", "V"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "qkv_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_m", "block_kv"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {}, "dims": ["block_kv"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "mask_causal_apply", "category": "communication", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_m", "role": "query_tile", "candidates": [8, 4], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "block_kv", "role": "kv_tile", "candidates": [32, 16], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "attn_fwd_tiled_v3",
                "bindings": {"ATTN_FWD_BLOCK_M": 8, "ATTN_FWD_BLOCK_KV": 32},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "attn_fwd.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_attn_fwd_chain() -> None:
    plan = plan_attn_fwd(
        _org_attn_fwd(),
        shape_bindings={"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        source_oracle={"kernel_kind": "attn_fwd_tiled_v3", "bindings": {"ATTN_FWD_BLOCK_M": 8, "ATTN_FWD_BLOCK_KV": 32}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "staging.q_resident_state": {"present": True, "attrs": {"resident_bytes_hint": 4096}},
                "staging.kv_streamed_tiles": {"present": True, "attrs": {"resident_bytes_hint": 8192}},
                "communication.streaming_softmax": {"present": True, "attrs": {"reduction_scope": "warp"}},
                "communication.mask_causal": {"present": True, "attrs": {"mask_or_causal": True}},
                "pipeline.stage_hint": {"present": True, "attrs": {"pipeline_depth_hint": 2}},
            }
        },
        ptx_facts={"mechanisms": {"pipeline.async_copy": {"present": True, "attrs": {"complete_async_pipeline": True}}}},
        budget=6,
    )
    assert plan.hardware_model["arch_cluster"] == "cuda_tc_mid_smem"
    assert plan.candidates[0].kernel_kind == "attn_fwd_tiled_v3"
    assert plan.candidates[0].bindings == {"ATTN_FWD_BLOCK_M": 8, "ATTN_FWD_BLOCK_KV": 32}
    assert any(candidate.kernel_kind == "attn_fwd_softmax_v2" for candidate in plan.candidates)
    assert "ATTN_FWD_BLOCK_M" in plan.param_space
    assert "source_exact" in str(plan.candidates[0].score_reason)
