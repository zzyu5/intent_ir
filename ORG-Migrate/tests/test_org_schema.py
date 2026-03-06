import pytest

from org.schema import OrgValidationError, validate_org_doc


def test_validate_org_doc_minimal_ok() -> None:
    payload = {
        "schema_version": "intentir_org_v1",
        "kernel": "flash_attention2d",
        "nodes": [
            {
                "id": "n0",
                "node_type": "tiling",
                "why": ["avoid_recompute"],
                "how": ["online_softmax"],
                "dims": ["ATTN_BLOCK_KV"],
                "constraints": ["ATTN_BLOCK_KV in {16,32,64}"],
                "evidence": [{"kind": "intent_summary", "path": "intent_summary.op_names"}],
            }
        ],
        "edges": [],
    }
    doc = validate_org_doc(payload)
    assert doc.schema_version == "intentir_org_v1"
    assert doc.kernel == "flash_attention2d"
    assert len(doc.nodes) == 1
    assert doc.nodes[0].node_type == "tiling"
    assert doc.nodes[0].dims[0].name == "ATTN_BLOCK_KV"


def test_validate_org_doc_missing_nodes_raises() -> None:
    payload = {"schema_version": "intentir_org_v1", "kernel": "flash_attention2d", "edges": []}
    with pytest.raises(OrgValidationError):
        validate_org_doc(payload)


def test_validate_org_doc_unknown_node_type_raises() -> None:
    payload = {
        "schema_version": "intentir_org_v1",
        "kernel": "flash_attention2d",
        "nodes": [
            {
                "id": "n0",
                "node_type": "not_a_real_type",
                "why": [],
                "how": [],
                "dims": [],
                "constraints": [],
                "evidence": [],
            }
        ],
        "edges": [],
    }
    with pytest.raises(OrgValidationError):
        validate_org_doc(payload)


def test_validate_org_doc_duplicate_node_ids_raises() -> None:
    payload = {
        "schema_version": "intentir_org_v1",
        "kernel": "flash_attention2d",
        "nodes": [
            {
                "id": "n0",
                "node_type": "tiling",
                "why": [],
                "how": [],
                "dims": [],
                "constraints": [],
                "evidence": [],
            },
            {
                "id": "n0",
                "node_type": "staging",
                "why": [],
                "how": [],
                "dims": [],
                "constraints": [],
                "evidence": [],
            },
        ],
        "edges": [],
    }
    with pytest.raises(OrgValidationError):
        validate_org_doc(payload)


def test_validate_org_doc_allows_int_ids_for_llm_tolerance() -> None:
    payload = {
        "schema_version": "intentir_org_v1",
        "kernel": "flash_attention2d",
        "nodes": [
            {
                "id": 0,
                "node_type": "tiling",
                "why": [],
                "how": [],
                "dims": [],
                "constraints": [],
                "evidence": [],
            },
            {
                "id": 1,
                "node_type": "staging",
                "why": [],
                "how": [],
                "dims": [],
                "constraints": [],
                "evidence": [],
            },
        ],
        "edges": [{"src": 0, "dst": 1, "edge_type": "depends_on"}],
    }
    doc = validate_org_doc(payload)
    assert [n.id for n in doc.nodes] == ["0", "1"]
    assert doc.edges[0].src == "0"
    assert doc.edges[0].dst == "1"


def test_validate_org_doc_drops_invalid_edges_with_warning() -> None:
    payload = {
        "schema_version": "intentir_org_v1",
        "kernel": "flash_attention2d",
        "nodes": [
            {
                "id": "n0",
                "node_type": "tiling",
                "why": [],
                "how": [],
                "dims": [],
                "constraints": [],
                "evidence": [],
            },
            {
                "id": "n1",
                "node_type": "staging",
                "why": [],
                "how": [],
                "dims": [],
                "constraints": [],
                "evidence": [],
            },
        ],
        "edges": [
            # src is not coercible -> should be dropped and recorded as warning.
            {"src": ["n0"], "dst": "n1", "edge_type": "depends_on"}
        ],
        "meta": {"note": "keep"},
    }
    doc = validate_org_doc(payload)
    assert doc.meta.get("note") == "keep"
    assert doc.edges == []
    vw = doc.meta.get("validation_warnings")
    assert isinstance(vw, dict)
    inv = vw.get("invalid_edges")
    assert isinstance(inv, dict)
    assert inv.get("count") == 1
