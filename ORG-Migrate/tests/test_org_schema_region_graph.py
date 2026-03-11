from __future__ import annotations

from org.schema import validate_org_doc


def test_org_schema_accepts_region_graph() -> None:
    doc = validate_org_doc(
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
                {"id": "g0", "tag": "resident_working_set", "summary": "resident q", "scope": "kv_loop", "tensors": ["Q"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "q_resident_state", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
            ],
            "dims": [],
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
                    "end": "exit",
                    "producer_mechanisms": ["m0"],
                    "consumer_mechanisms": ["m0"],
                    "supports_goals": ["g0"],
                    "bytes_hint": 256,
                    "evidence_refs": ["e0"],
                }
            ],
            "dataflow_edges": [],
            "mechanism_topology": [],
            "schedule_edges": [],
            "region_graph": {
                "regions": [
                    {
                        "id": "r0",
                        "kind": "if",
                        "path_id": "pi0",
                        "predicate": "q_idx < kv_idx",
                        "entry_mechanisms": ["m0"],
                        "exit_mechanisms": ["m0"],
                        "evidence_refs": ["e0"],
                    },
                    {
                        "id": "r1",
                        "kind": "then",
                        "parent": "r0",
                        "path_id": "pi0",
                        "entry_mechanisms": ["m0"],
                        "exit_mechanisms": ["m0"],
                        "evidence_refs": ["e0"],
                    },
                ],
                "edges": [
                    {
                        "id": "re0",
                        "src": "r0",
                        "dst": "r1",
                        "relation": "branch_then",
                        "path_id": "pi0",
                        "lifetimes": ["lt0"],
                        "mechanisms": ["m0"],
                        "evidence_refs": ["e0"],
                    }
                ],
            },
            "source_oracle": {
                "kernel_kind": "attn2d_causal_softmax_v8",
                "bindings": {"ATTN_BLOCK_KV": 32},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e0"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "cfg evidence"},
            ],
        }
    )
    assert doc.region_graph is not None
    assert doc.region_graph.regions[0].predicate == "q_idx < kv_idx"
    assert doc.region_graph.edges[0].path_id == "pi0"
