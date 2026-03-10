from __future__ import annotations

from types import SimpleNamespace

from org.llm_hub import LLMOrgHub


def test_llm_org_hub_sanitizes_unknown_dim_refs(monkeypatch) -> None:
    payload = {
        "schema_version": "intentir_org_v1",
        "kernel": "flash_attention2d",
        "goals": [
            {"id": "g0", "tag": "resident_working_set", "summary": "keep resident", "scope": "kv_loop", "tensors": ["Q"], "evidence_refs": ["e0"]},
        ],
        "mechanisms": [
            {"id": "m0", "tag": "q_resident_state", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["resident_bytes", "ghost_dim"], "evidence_refs": ["e0"]},
        ],
        "dims": [
            {"name": "resident_bytes", "role": "resident_budget", "candidates": [256], "constraints": [], "evidence_refs": ["e0"]},
        ],
        "tensors": [
            {"id": "t0", "name": "Q", "role": "query_state", "evidence_refs": ["e0"]},
        ],
        "tensor_lifetimes": [
            {
                "id": "lt0",
                "tensor": "t0",
                "region": "kv_loop",
                "storage": "register",
                "start": "load_q",
                "end": "softmax",
                "producer_mechanisms": ["m0", "ghost_mech"],
                "consumer_mechanisms": ["m0"],
                "supports_goals": ["g0", "ghost_goal"],
                "dims": ["resident_bytes", "ghost_dim"],
                "evidence_refs": ["e0"],
            }
        ],
        "dataflow_edges": [
            {"id": "df0", "src": "lt0", "dst": "ghost_lt", "tensor": "t0", "kind": "stage", "order": 0, "mechanisms": ["m0"], "evidence_refs": ["e0"]},
        ],
        "mechanism_topology": [
            {"id": "mt0", "src": "m0", "dst": "ghost_mech", "relation": "feeds", "tensors": ["t0"], "lifetimes": ["lt0", "ghost_lt"], "evidence_refs": ["e0"]},
        ],
        "evidence": [
            {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "evidence"},
        ],
    }

    def fake_extract(messages, model, max_parse_retries, **kwargs):
        return payload, {"ok": True, "chosen": {"model": model}}

    monkeypatch.setattr("org.llm_hub.extract_json_object_with_trace", fake_extract)
    hub = LLMOrgHub()
    desc = SimpleNamespace(
        name="flash_attention2d",
        frontend="triton",
        source_text="def kernel(): pass",
        io_spec={},
        launch={},
        frontend_facts={},
        frontend_constraints={},
        artifacts=SimpleNamespace(ttir_path=None, ttgir_path=None, ptx_text=None),
        meta={},
    )
    candidate = hub.lift(
        desc,
        intent_summary={"name": "flash_attention2d"},
        extra_evidence={
            "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
            "source_arch": "sm90",
            "target_arch": "sm120",
            "source_oracle_facts": {"oracle": {"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64}, "arch": "sm90", "compiler_stack": "python"}},
        },
    )
    mechanism = candidate.org.mechanisms[0]
    assert mechanism.dims == ["resident_bytes"]
    assert candidate.org.tensor_lifetimes[0].producer_mechanisms == ["m0"]
    assert candidate.org.tensor_lifetimes[0].supports_goals == ["g0"]
    assert candidate.org.dataflow_edges == []
    assert candidate.org.mechanism_topology == []
