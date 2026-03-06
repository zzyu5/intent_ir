from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import ORG_NODE_TYPES, ORG_SCHEMA_VERSION_V1


class OrgValidationError(ValueError):
    def __init__(self, msg: str, *, path: str = "") -> None:
        super().__init__(f"{msg}{(' (path=' + path + ')') if path else ''}")
        self.path = str(path or "")


def _as_dict(x: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(x, Mapping):
        raise OrgValidationError("expected object", path=path)
    return dict(x)


def _as_list(x: Any, *, path: str) -> list[Any]:
    if not isinstance(x, list):
        raise OrgValidationError("expected list", path=path)
    return list(x)


def _as_str(x: Any, *, path: str) -> str:
    if not isinstance(x, str):
        raise OrgValidationError("expected string", path=path)
    s = str(x).strip()
    if not s:
        raise OrgValidationError("expected non-empty string", path=path)
    return s


def _as_id_str(x: Any, *, path: str) -> str:
    """
    Be slightly tolerant for id-like fields produced by LLMs:
    - allow integers (coerce to string)
    - allow objects like {"id": "..."} (extract id)

    Still rejects lists/objects without an id and enforces non-empty.
    """

    if isinstance(x, Mapping) and "id" in x:
        x = x.get("id")
    if isinstance(x, str):
        s = str(x).strip()
    elif isinstance(x, int) and not isinstance(x, bool):
        s = str(int(x))
    else:
        raise OrgValidationError("expected string", path=path)
    if not s:
        raise OrgValidationError("expected non-empty string", path=path)
    return s


def _as_str_list(x: Any, *, path: str) -> list[str]:
    items = _as_list(x, path=path)
    out: list[str] = []
    for i, raw in enumerate(items):
        if not isinstance(raw, str):
            raise OrgValidationError("expected string", path=f"{path}[{i}]")
        s = str(raw).strip()
        if not s:
            continue
        out.append(s)
    return out


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    path: str
    detail: str = ""

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "EvidenceRef":
        obj = _as_dict(raw, path=path)
        kind = str(obj.get("kind") or "").strip()
        p = str(obj.get("path") or "").strip()
        if not kind:
            raise OrgValidationError("evidence.kind must be non-empty string", path=f"{path}.kind")
        if not p:
            raise OrgValidationError("evidence.path must be non-empty string", path=f"{path}.path")
        detail = str(obj.get("detail") or "").strip()
        return cls(kind=kind, path=p, detail=detail)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": str(self.kind), "path": str(self.path)}
        if str(self.detail or "").strip():
            out["detail"] = str(self.detail)
        return out


@dataclass(frozen=True)
class OrgDim:
    name: str
    allowed: list[Any] = field(default_factory=list)
    note: str = ""

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgDim":
        if isinstance(raw, str):
            name = str(raw).strip()
            if not name:
                raise OrgValidationError("dim must be non-empty string", path=path)
            return cls(name=name)
        obj = _as_dict(raw, path=path)
        name = str(obj.get("name") or obj.get("dim") or "").strip()
        if not name:
            raise OrgValidationError("dim.name must be non-empty string", path=f"{path}.name")
        allowed_raw = obj.get("allowed")
        allowed: list[Any] = []
        if isinstance(allowed_raw, list):
            allowed = list(allowed_raw)
        note = str(obj.get("note") or "").strip()
        return cls(name=name, allowed=allowed, note=note)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": str(self.name)}
        if self.allowed:
            out["allowed"] = list(self.allowed)
        if str(self.note or "").strip():
            out["note"] = str(self.note)
        return out


@dataclass(frozen=True)
class OrgNode:
    id: str
    node_type: str
    why: list[str] = field(default_factory=list)
    how: list[str] = field(default_factory=list)
    dims: list[OrgDim] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgNode":
        obj = _as_dict(raw, path=path)
        node_id_raw = obj.get("id") if obj.get("id") is not None else obj.get("node_id")
        node_id = _as_id_str(node_id_raw, path=f"{path}.id")
        node_type = _as_str(obj.get("node_type") or obj.get("type"), path=f"{path}.node_type")
        if node_type not in ORG_NODE_TYPES:
            raise OrgValidationError(
                f"unsupported node_type={node_type!r}; allowed={list(ORG_NODE_TYPES)}",
                path=f"{path}.node_type",
            )
        if "why" not in obj:
            raise OrgValidationError("missing required field", path=f"{path}.why")
        if "how" not in obj:
            raise OrgValidationError("missing required field", path=f"{path}.how")
        if "dims" not in obj:
            raise OrgValidationError("missing required field", path=f"{path}.dims")
        if "constraints" not in obj:
            raise OrgValidationError("missing required field", path=f"{path}.constraints")
        if "evidence" not in obj:
            raise OrgValidationError("missing required field", path=f"{path}.evidence")

        why = _as_str_list(obj.get("why"), path=f"{path}.why")
        how = _as_str_list(obj.get("how"), path=f"{path}.how")
        dims_list = _as_list(obj.get("dims"), path=f"{path}.dims")
        dims: list[OrgDim] = [OrgDim.from_json(x, path=f"{path}.dims[{i}]") for i, x in enumerate(dims_list)]
        constraints = _as_str_list(obj.get("constraints"), path=f"{path}.constraints")
        evidence_list = _as_list(obj.get("evidence"), path=f"{path}.evidence")
        evidence = [EvidenceRef.from_json(x, path=f"{path}.evidence[{i}]") for i, x in enumerate(evidence_list)]
        attrs_raw = obj.get("attrs") or {}
        attrs = dict(attrs_raw) if isinstance(attrs_raw, Mapping) else {}
        return cls(
            id=node_id,
            node_type=node_type,
            why=why,
            how=how,
            dims=dims,
            constraints=constraints,
            evidence=evidence,
            attrs=attrs,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "node_type": str(self.node_type),
            "why": list(self.why or []),
            "how": list(self.how or []),
            "dims": [d.to_json_dict() for d in list(self.dims or [])],
            "constraints": list(self.constraints or []),
            "evidence": [e.to_json_dict() for e in list(self.evidence or [])],
            "attrs": dict(self.attrs or {}),
        }


@dataclass(frozen=True)
class OrgEdge:
    src: str
    dst: str
    edge_type: str = "depends_on"
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgEdge":
        obj = _as_dict(raw, path=path)
        src_raw = obj.get("src") if obj.get("src") is not None else obj.get("source")
        dst_raw = obj.get("dst") if obj.get("dst") is not None else obj.get("target")
        src = _as_id_str(src_raw, path=f"{path}.src")
        dst = _as_id_str(dst_raw, path=f"{path}.dst")
        edge_type = str(obj.get("edge_type") or obj.get("type") or "depends_on").strip() or "depends_on"
        attrs_raw = obj.get("attrs") or {}
        attrs = dict(attrs_raw) if isinstance(attrs_raw, Mapping) else {}
        return cls(src=src, dst=dst, edge_type=edge_type, attrs=attrs)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"src": str(self.src), "dst": str(self.dst)}
        if str(self.edge_type or "").strip():
            out["edge_type"] = str(self.edge_type)
        if self.attrs:
            out["attrs"] = dict(self.attrs)
        return out


@dataclass
class OrgDoc:
    schema_version: str = ORG_SCHEMA_VERSION_V1
    kernel: str = ""
    nodes: list[OrgNode] = field(default_factory=list)
    edges: list[OrgEdge] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json_dict(cls, raw: Any) -> "OrgDoc":
        obj = _as_dict(raw, path="$")
        schema_version = str(obj.get("schema_version") or ORG_SCHEMA_VERSION_V1).strip()
        if schema_version != ORG_SCHEMA_VERSION_V1:
            raise OrgValidationError(
                f"unsupported schema_version={schema_version!r}; expected {ORG_SCHEMA_VERSION_V1!r}",
                path="schema_version",
            )
        kernel = _as_str(obj.get("kernel") or obj.get("kernel_name") or "unknown", path="kernel")
        nodes_raw = obj.get("nodes")
        if nodes_raw is None:
            raise OrgValidationError("missing required field", path="nodes")
        node_list = _as_list(nodes_raw, path="nodes")
        nodes = [OrgNode.from_json(x, path=f"nodes[{i}]") for i, x in enumerate(node_list)]
        edges_raw = obj.get("edges")
        if edges_raw is None:
            raise OrgValidationError("missing required field", path="edges")
        edge_list = _as_list(edges_raw, path="edges")
        meta_raw = obj.get("meta") or {}
        meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}

        # Edges are useful for humans, but not required for Phase-1 mappers.
        # Be tolerant to minor LLM formatting issues (e.g., non-string ids) by
        # dropping invalid edges and recording a warning.
        edges: list[OrgEdge] = []
        edge_warnings: list[dict[str, Any]] = []
        for i, x in enumerate(edge_list):
            try:
                edges.append(OrgEdge.from_json(x, path=f"edges[{i}]"))
            except OrgValidationError as e:
                edge_warnings.append({"index": int(i), "error": str(e)})
        if edge_warnings:
            vw = meta.get("validation_warnings")
            if not isinstance(vw, dict):
                vw = {}
            vw["invalid_edges"] = {"count": int(len(edge_warnings)), "examples": edge_warnings[:8]}
            meta["validation_warnings"] = vw
        inst = cls(schema_version=schema_version, kernel=kernel, nodes=nodes, edges=edges, meta=meta)
        inst.validate()
        return inst

    def validate(self) -> None:
        if str(self.schema_version or "").strip() != ORG_SCHEMA_VERSION_V1:
            raise OrgValidationError(
                f"schema_version must be {ORG_SCHEMA_VERSION_V1!r}",
                path="schema_version",
            )
        if not str(self.kernel or "").strip():
            raise OrgValidationError("kernel must be non-empty string", path="kernel")
        ids = [n.id for n in list(self.nodes or [])]
        if len(set(ids)) != len(ids):
            raise OrgValidationError("node ids must be unique", path="nodes")
        # Edge sanity: allow dangling edges but enforce type correctness.
        for i, e in enumerate(list(self.edges or [])):
            if not isinstance(e, OrgEdge):
                raise OrgValidationError("invalid edge object", path=f"edges[{i}]")

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": ORG_SCHEMA_VERSION_V1,
            "kernel": str(self.kernel),
            "nodes": [n.to_json_dict() for n in list(self.nodes or [])],
            "edges": [e.to_json_dict() for e in list(self.edges or [])],
        }
        if self.meta:
            out["meta"] = dict(self.meta)
        return out


def validate_org_doc(payload: Any) -> OrgDoc:
    """
    Validate payload as `intentir_org_v1` and return a parsed `OrgDoc`.
    """

    return OrgDoc.from_json_dict(payload)


__all__ = [
    "EvidenceRef",
    "OrgDim",
    "OrgDoc",
    "OrgEdge",
    "OrgNode",
    "OrgValidationError",
    "validate_org_doc",
]
