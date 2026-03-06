from __future__ import annotations

from org.mapping.cuda.flash_attention2d import plan_flash_attention2d
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_flash() -> object:
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
                {"id": "g0", "tag": "resident_working_set", "summary": "keep q/state resident", "scope": "kv_loop", "tensors": ["Q", "Out"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "online reduce", "scope": "softmax", "tensors": ["Out"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "no score matrix", "scope": "softmax", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "prefetch next tile", "scope": "kv_loop", "tensors": ["K", "V"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "kv_tile_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["tile_kv"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {}, "dims": ["score_warps"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "tile_kv", "role": "kv_tile", "candidates": [32, 64], "constraints": ["tile_kv <= KV_CTX"], "evidence_refs": ["e0"]},
                {"name": "score_warps", "role": "score_reduce", "candidates": [6, 4], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
            ],
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


def test_backend_plan_flash_attention2d_chain() -> None:
    plan = plan_flash_attention2d(
        _org_flash(),
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        source_oracle={"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        ttgir_facts={
            "mechanisms": {
                "staging.q_resident_state": {"present": True, "attrs": {"resident_bytes_hint": 256}},
                "staging.kv_streamed_tiles": {"present": True, "attrs": {"resident_bytes_hint": 32768}},
                "pipeline.stage_hint": {"present": False, "attrs": {"pipeline_depth_hint": None}},
            }
        },
        ptx_facts={"mechanisms": {"pipeline.async_copy": {"present": True, "attrs": {"complete_async_pipeline": False}}}},
        budget=8,
    )
    assert plan.selected_modules
    assert plan.hardware_model["arch_cluster"] == "cuda_tc_mid_smem"
    assert plan.param_space["ATTN_BLOCK_KV"][0] == 64
    assert plan.candidates
    assert [c.kernel_kind for c in plan.candidates[:4]] == [
        "attn2d_causal_softmax_v6",
        "attn2d_causal_softmax_v6",
        "attn2d_causal_softmax_v6",
        "attn2d_causal_softmax_v6",
    ]
    assert "attn2d_causal_softmax_v8" in plan.param_space["kernel_kind"]
    assert any(c.kernel_kind == "attn2d_causal_softmax_v8" for c in plan.candidates)
    assert plan.candidates[0].bindings == {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}
    assert plan.candidates[0].score is not None
    assert "cluster=cuda_tc_mid_smem" in str(plan.candidates[0].score_reason)
    assert "source_exact" in str(plan.candidates[0].score_reason)
    assert any(str(x).startswith("preserve:") for x in plan.notes)
    assert any(item.get("reason") == "incomplete async evidence" for item in plan.substitutions)


def test_backend_plan_flash_attention2d_large_smem_allows_v7_front() -> None:
    plan = plan_flash_attention2d(
        _org_flash(),
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        source_oracle={"kernel_kind": "attn2d_causal_softmax_v7", "bindings": {"ATTN_BLOCK_KV": 64}},
        hardware_model=build_hardware_model(target="cuda_h100", arch="sm90"),
        ttgir_facts={
            "mechanisms": {
                "staging.q_resident_state": {"present": True, "attrs": {"resident_bytes_hint": 256}},
                "staging.kv_streamed_tiles": {"present": True, "attrs": {"resident_bytes_hint": 32768}},
                "pipeline.stage_hint": {"present": True, "attrs": {"pipeline_depth_hint": 2}},
            }
        },
        ptx_facts={"mechanisms": {"pipeline.async_copy": {"present": True, "attrs": {"complete_async_pipeline": True}}}},
        budget=8,
    )
    assert plan.hardware_model["arch_cluster"] == "cuda_tc_large_smem"
    assert plan.candidates[0].kernel_kind == "attn2d_causal_softmax_v7"
    assert plan.candidates[0].bindings.get("ATTN_BLOCK_KV") == 64
