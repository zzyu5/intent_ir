from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import ORG_GOAL_TAGS, ORG_GOAL_TAGS_BY_KERNEL, ORG_MECHANISM_CATEGORIES, ORG_SCHEMA_VERSION_V1


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


def _as_optional_str(x: Any, *, path: str) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        raise OrgValidationError("expected string", path=path)
    return str(x).strip()


def _as_str_list(x: Any, *, path: str) -> list[str]:
    if isinstance(x, str):
        s = str(x).strip()
        return ([s] if s else [])
    items = _as_list(x, path=path)
    out: list[str] = []
    for i, raw in enumerate(items):
        s = _as_str(raw, path=f"{path}[{i}]")
        out.append(s)
    return out


def _coerce_int(x: Any, *, path: str) -> int:
    try:
        return int(x)
    except Exception as exc:
        raise OrgValidationError("expected int", path=path) from exc


def _as_optional_int(x: Any, *, path: str) -> int | None:
    if x is None:
        return None
    return _coerce_int(x, path=path)


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    kind: str
    path: str
    summary: str = ""
    text: str = ""

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "EvidenceItem":
        obj = _as_dict(raw, path=path)
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            kind=_as_str(obj.get("kind"), path=f"{path}.kind"),
            path=_as_str(obj.get("path"), path=f"{path}.path"),
            summary=_as_optional_str(obj.get("summary"), path=f"{path}.summary"),
            text=_as_optional_str(obj.get("text"), path=f"{path}.text"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "path": self.path}
        if self.summary:
            out["summary"] = self.summary
        if self.text:
            out["text"] = self.text
        return out


@dataclass(frozen=True)
class SourceContext:
    frontend: str
    source_arch: str
    target_arch: str
    shape_bindings: dict[str, int] = field(default_factory=dict)
    artifacts: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "SourceContext":
        obj = _as_dict(raw, path=path)
        shape_bindings_raw = obj.get("shape_bindings") or {}
        shape_bindings: dict[str, int] = {}
        if isinstance(shape_bindings_raw, Mapping):
            for k, v in dict(shape_bindings_raw).items():
                key = str(k).strip()
                if not key:
                    continue
                shape_bindings[key] = _coerce_int(v, path=f"{path}.shape_bindings.{key}")
        artifacts_raw = obj.get("artifacts") or {}
        artifacts: dict[str, str | None] = {}
        if isinstance(artifacts_raw, Mapping):
            for k, v in dict(artifacts_raw).items():
                key = str(k).strip()
                if not key:
                    continue
                artifacts[key] = (None if v is None else str(v).strip())
        return cls(
            frontend=_as_str(obj.get("frontend"), path=f"{path}.frontend"),
            source_arch=_as_optional_str(obj.get("source_arch"), path=f"{path}.source_arch"),
            target_arch=_as_optional_str(obj.get("target_arch"), path=f"{path}.target_arch"),
            shape_bindings=shape_bindings,
            artifacts=artifacts,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "frontend": self.frontend,
            "source_arch": self.source_arch,
            "target_arch": self.target_arch,
            "shape_bindings": {str(k): int(v) for k, v in self.shape_bindings.items()},
            "artifacts": {str(k): (None if v is None else str(v)) for k, v in self.artifacts.items()},
        }


@dataclass(frozen=True)
class OrgGoal:
    id: str
    tag: str
    summary: str
    scope: str
    tensors: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgGoal":
        obj = _as_dict(raw, path=path)
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            tag=_as_str(obj.get("tag"), path=f"{path}.tag"),
            summary=_as_str(obj.get("summary"), path=f"{path}.summary"),
            scope=_as_str(obj.get("scope"), path=f"{path}.scope"),
            tensors=_as_str_list(obj.get("tensors") or [], path=f"{path}.tensors"),
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tag": self.tag,
            "summary": self.summary,
            "scope": self.scope,
            "tensors": list(self.tensors),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class OrgMechanism:
    id: str
    tag: str
    category: str
    supports_goals: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    dims: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgMechanism":
        obj = _as_dict(raw, path=path)
        attrs = dict(obj.get("attrs") or {}) if isinstance(obj.get("attrs"), Mapping) else {}
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            tag=_as_str(obj.get("tag"), path=f"{path}.tag"),
            category=_as_str(obj.get("category"), path=f"{path}.category"),
            supports_goals=_as_str_list(obj.get("supports_goals") or [], path=f"{path}.supports_goals"),
            attrs=attrs,
            dims=_as_str_list(obj.get("dims") or [], path=f"{path}.dims"),
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tag": self.tag,
            "category": self.category,
            "supports_goals": list(self.supports_goals),
            "attrs": dict(self.attrs),
            "dims": list(self.dims),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class OrgDim:
    name: str
    role: str
    candidates: list[Any] = field(default_factory=list)
    range: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgDim":
        obj = _as_dict(raw, path=path)
        candidates_raw = obj.get("candidates") or []
        candidates = list(candidates_raw) if isinstance(candidates_raw, list) else []
        range_raw = obj.get("range") or {}
        range_v = dict(range_raw) if isinstance(range_raw, Mapping) else {}
        constraints = _as_str_list(obj.get("constraints") or [], path=f"{path}.constraints")
        return cls(
            name=_as_str(obj.get("name"), path=f"{path}.name"),
            role=_as_str(obj.get("role"), path=f"{path}.role"),
            candidates=candidates,
            range=range_v,
            constraints=constraints,
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "constraints": list(self.constraints),
            "evidence_refs": list(self.evidence_refs),
        }
        if self.candidates:
            out["candidates"] = list(self.candidates)
        if self.range:
            out["range"] = dict(self.range)
        return out


@dataclass(frozen=True)
class OrgTensor:
    id: str
    name: str
    role: str
    dtype: str = ""
    layout: str = ""
    view_of: str = ""
    alias_group: str = ""
    aliases: list[str] = field(default_factory=list)
    shape_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgTensor":
        obj = _as_dict(raw, path=path)
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            name=_as_str(obj.get("name"), path=f"{path}.name"),
            role=_as_str(obj.get("role"), path=f"{path}.role"),
            dtype=_as_optional_str(obj.get("dtype"), path=f"{path}.dtype"),
            layout=_as_optional_str(obj.get("layout"), path=f"{path}.layout"),
            view_of=_as_optional_str(obj.get("view_of"), path=f"{path}.view_of"),
            alias_group=_as_optional_str(obj.get("alias_group"), path=f"{path}.alias_group"),
            aliases=_as_str_list(obj.get("aliases") or [], path=f"{path}.aliases"),
            shape_refs=_as_str_list(obj.get("shape_refs") or [], path=f"{path}.shape_refs"),
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "aliases": list(self.aliases),
            "shape_refs": list(self.shape_refs),
            "evidence_refs": list(self.evidence_refs),
        }
        if self.dtype:
            out["dtype"] = self.dtype
        if self.layout:
            out["layout"] = self.layout
        if self.view_of:
            out["view_of"] = self.view_of
        if self.alias_group:
            out["alias_group"] = self.alias_group
        return out


@dataclass(frozen=True)
class OrgTensorLifetime:
    id: str
    tensor: str
    region: str
    storage: str
    start: str
    end: str
    scope: str = ""
    layout: str = ""
    producer_mechanisms: list[str] = field(default_factory=list)
    consumer_mechanisms: list[str] = field(default_factory=list)
    supports_goals: list[str] = field(default_factory=list)
    dims: list[str] = field(default_factory=list)
    bytes_hint: int | None = None
    pipeline_stage: int | None = None
    reuse_window: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgTensorLifetime":
        obj = _as_dict(raw, path=path)
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            tensor=_as_str(obj.get("tensor"), path=f"{path}.tensor"),
            region=_as_str(obj.get("region"), path=f"{path}.region"),
            storage=_as_str(obj.get("storage"), path=f"{path}.storage"),
            start=_as_str(obj.get("start"), path=f"{path}.start"),
            end=_as_str(obj.get("end"), path=f"{path}.end"),
            scope=_as_optional_str(obj.get("scope"), path=f"{path}.scope"),
            layout=_as_optional_str(obj.get("layout"), path=f"{path}.layout"),
            producer_mechanisms=_as_str_list(obj.get("producer_mechanisms") or [], path=f"{path}.producer_mechanisms"),
            consumer_mechanisms=_as_str_list(obj.get("consumer_mechanisms") or [], path=f"{path}.consumer_mechanisms"),
            supports_goals=_as_str_list(obj.get("supports_goals") or [], path=f"{path}.supports_goals"),
            dims=_as_str_list(obj.get("dims") or [], path=f"{path}.dims"),
            bytes_hint=_as_optional_int(obj.get("bytes_hint"), path=f"{path}.bytes_hint"),
            pipeline_stage=_as_optional_int(obj.get("pipeline_stage"), path=f"{path}.pipeline_stage"),
            reuse_window=_as_optional_str(obj.get("reuse_window"), path=f"{path}.reuse_window"),
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "tensor": self.tensor,
            "region": self.region,
            "storage": self.storage,
            "start": self.start,
            "end": self.end,
            "producer_mechanisms": list(self.producer_mechanisms),
            "consumer_mechanisms": list(self.consumer_mechanisms),
            "supports_goals": list(self.supports_goals),
            "dims": list(self.dims),
            "evidence_refs": list(self.evidence_refs),
        }
        if self.scope:
            out["scope"] = self.scope
        if self.layout:
            out["layout"] = self.layout
        if self.bytes_hint is not None:
            out["bytes_hint"] = int(self.bytes_hint)
        if self.pipeline_stage is not None:
            out["pipeline_stage"] = int(self.pipeline_stage)
        if self.reuse_window:
            out["reuse_window"] = self.reuse_window
        return out


@dataclass(frozen=True)
class OrgDataflowEdge:
    id: str
    src: str
    dst: str
    tensor: str
    kind: str
    order: int = 0
    mechanisms: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgDataflowEdge":
        obj = _as_dict(raw, path=path)
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            src=_as_str(obj.get("src"), path=f"{path}.src"),
            dst=_as_str(obj.get("dst"), path=f"{path}.dst"),
            tensor=_as_str(obj.get("tensor"), path=f"{path}.tensor"),
            kind=_as_str(obj.get("kind"), path=f"{path}.kind"),
            order=_coerce_int(obj.get("order", 0), path=f"{path}.order"),
            mechanisms=_as_str_list(obj.get("mechanisms") or [], path=f"{path}.mechanisms"),
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "src": self.src,
            "dst": self.dst,
            "tensor": self.tensor,
            "kind": self.kind,
            "order": int(self.order),
            "mechanisms": list(self.mechanisms),
            "evidence_refs": list(self.evidence_refs),
        }
        return out


@dataclass(frozen=True)
class OrgMechanismTopologyEdge:
    id: str
    src: str
    dst: str
    relation: str
    tensors: list[str] = field(default_factory=list)
    lifetimes: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgMechanismTopologyEdge":
        obj = _as_dict(raw, path=path)
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            src=_as_str(obj.get("src"), path=f"{path}.src"),
            dst=_as_str(obj.get("dst"), path=f"{path}.dst"),
            relation=_as_str(obj.get("relation"), path=f"{path}.relation"),
            tensors=_as_str_list(obj.get("tensors") or [], path=f"{path}.tensors"),
            lifetimes=_as_str_list(obj.get("lifetimes") or [], path=f"{path}.lifetimes"),
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "tensors": list(self.tensors),
            "lifetimes": list(self.lifetimes),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class OrgScheduleEdge:
    id: str
    src: str
    dst: str
    relation: str
    scope: str = ""
    resources: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "OrgScheduleEdge":
        obj = _as_dict(raw, path=path)
        attrs = dict(obj.get("attrs") or {}) if isinstance(obj.get("attrs"), Mapping) else {}
        return cls(
            id=_as_str(obj.get("id"), path=f"{path}.id"),
            src=_as_str(obj.get("src"), path=f"{path}.src"),
            dst=_as_str(obj.get("dst"), path=f"{path}.dst"),
            relation=_as_str(obj.get("relation"), path=f"{path}.relation"),
            scope=_as_optional_str(obj.get("scope"), path=f"{path}.scope"),
            resources=_as_str_list(obj.get("resources") or [], path=f"{path}.resources"),
            attrs=attrs,
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "resources": list(self.resources),
            "evidence_refs": list(self.evidence_refs),
        }
        if self.scope:
            out["scope"] = self.scope
        if self.attrs:
            out["attrs"] = dict(self.attrs)
        return out


@dataclass(frozen=True)
class SourceOracle:
    kernel_kind: str
    bindings: dict[str, int] = field(default_factory=dict)
    arch: str = ""
    compiler_stack: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, raw: Any, *, path: str) -> "SourceOracle":
        obj = _as_dict(raw, path=path)
        bindings_raw = obj.get("bindings") or {}
        bindings: dict[str, int] = {}
        if isinstance(bindings_raw, Mapping):
            for k, v in dict(bindings_raw).items():
                key = str(k).strip()
                if not key:
                    continue
                bindings[key] = _coerce_int(v, path=f"{path}.bindings.{key}")
        return cls(
            kernel_kind=_as_optional_str(obj.get("kernel_kind"), path=f"{path}.kernel_kind"),
            bindings=bindings,
            arch=_as_optional_str(obj.get("arch"), path=f"{path}.arch"),
            compiler_stack=_as_optional_str(obj.get("compiler_stack"), path=f"{path}.compiler_stack"),
            evidence_refs=_as_str_list(obj.get("evidence_refs") or [], path=f"{path}.evidence_refs"),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kernel_kind": self.kernel_kind,
            "bindings": {str(k): int(v) for k, v in self.bindings.items()},
            "arch": self.arch,
            "compiler_stack": self.compiler_stack,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass
class OrgDoc:
    schema_version: str = ORG_SCHEMA_VERSION_V1
    kernel: str = ""
    source_context: SourceContext | None = None
    goals: list[OrgGoal] = field(default_factory=list)
    mechanisms: list[OrgMechanism] = field(default_factory=list)
    dims: list[OrgDim] = field(default_factory=list)
    tensors: list[OrgTensor] = field(default_factory=list)
    tensor_lifetimes: list[OrgTensorLifetime] = field(default_factory=list)
    dataflow_edges: list[OrgDataflowEdge] = field(default_factory=list)
    mechanism_topology: list[OrgMechanismTopologyEdge] = field(default_factory=list)
    schedule_edges: list[OrgScheduleEdge] = field(default_factory=list)
    source_oracle: SourceOracle | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kernel": self.kernel,
            "source_context": (self.source_context.to_json_dict() if self.source_context is not None else None),
            "goals": [g.to_json_dict() for g in self.goals],
            "mechanisms": [m.to_json_dict() for m in self.mechanisms],
            "dims": [d.to_json_dict() for d in self.dims],
            "tensors": [t.to_json_dict() for t in self.tensors],
            "tensor_lifetimes": [t.to_json_dict() for t in self.tensor_lifetimes],
            "dataflow_edges": [e.to_json_dict() for e in self.dataflow_edges],
            "mechanism_topology": [e.to_json_dict() for e in self.mechanism_topology],
            "schedule_edges": [e.to_json_dict() for e in self.schedule_edges],
            "source_oracle": (self.source_oracle.to_json_dict() if self.source_oracle is not None else None),
            "evidence": [e.to_json_dict() for e in self.evidence],
            "notes": list(self.notes),
        }

    @classmethod
    def from_json_dict(
        cls,
        raw: Any,
        *,
        source_context: Mapping[str, Any] | None = None,
        source_oracle: Mapping[str, Any] | None = None,
    ) -> "OrgDoc":
        obj = _as_dict(raw, path="$")
        schema_version = str(obj.get("schema_version") or ORG_SCHEMA_VERSION_V1).strip()
        if schema_version != ORG_SCHEMA_VERSION_V1:
            raise OrgValidationError(
                f"unsupported schema_version={schema_version!r}; expected {ORG_SCHEMA_VERSION_V1!r}",
                path="schema_version",
            )
        kernel = _as_str(obj.get("kernel"), path="kernel")

        source_context_raw = obj.get("source_context")
        if source_context_raw is None:
            if source_context is None:
                raise OrgValidationError("missing required field", path="source_context")
            source_context_raw = dict(source_context)
        source_context_obj = SourceContext.from_json(source_context_raw, path="source_context")

        goals_raw = obj.get("goals")
        if goals_raw is None:
            raise OrgValidationError("missing required field", path="goals")
        goals = [OrgGoal.from_json(x, path=f"goals[{i}]") for i, x in enumerate(_as_list(goals_raw, path="goals"))]

        mechanisms_raw = obj.get("mechanisms")
        if mechanisms_raw is None:
            raise OrgValidationError("missing required field", path="mechanisms")
        mechanisms = [
            OrgMechanism.from_json(x, path=f"mechanisms[{i}]")
            for i, x in enumerate(_as_list(mechanisms_raw, path="mechanisms"))
        ]

        dims_raw = obj.get("dims")
        if dims_raw is None:
            raise OrgValidationError("missing required field", path="dims")
        dims = [OrgDim.from_json(x, path=f"dims[{i}]") for i, x in enumerate(_as_list(dims_raw, path="dims"))]

        tensors = [
            OrgTensor.from_json(x, path=f"tensors[{i}]")
            for i, x in enumerate(_as_list(obj.get("tensors") or [], path="tensors"))
        ]
        tensor_lifetimes = [
            OrgTensorLifetime.from_json(x, path=f"tensor_lifetimes[{i}]")
            for i, x in enumerate(_as_list(obj.get("tensor_lifetimes") or [], path="tensor_lifetimes"))
        ]
        dataflow_edges = [
            OrgDataflowEdge.from_json(x, path=f"dataflow_edges[{i}]")
            for i, x in enumerate(_as_list(obj.get("dataflow_edges") or [], path="dataflow_edges"))
        ]
        mechanism_topology = [
            OrgMechanismTopologyEdge.from_json(x, path=f"mechanism_topology[{i}]")
            for i, x in enumerate(_as_list(obj.get("mechanism_topology") or [], path="mechanism_topology"))
        ]
        schedule_edges = [
            OrgScheduleEdge.from_json(x, path=f"schedule_edges[{i}]")
            for i, x in enumerate(_as_list(obj.get("schedule_edges") or [], path="schedule_edges"))
        ]

        source_oracle_raw = obj.get("source_oracle")
        if source_oracle_raw is None:
            if source_oracle is None:
                raise OrgValidationError("missing required field", path="source_oracle")
            source_oracle_raw = dict(source_oracle)
        source_oracle_obj = SourceOracle.from_json(source_oracle_raw, path="source_oracle")

        evidence_raw = obj.get("evidence")
        if evidence_raw is None:
            raise OrgValidationError("missing required field", path="evidence")
        evidence = [EvidenceItem.from_json(x, path=f"evidence[{i}]") for i, x in enumerate(_as_list(evidence_raw, path="evidence"))]

        notes = _as_str_list(obj.get("notes") or [], path="notes")

        evidence_ids = [e.id for e in evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise OrgValidationError("duplicate evidence ids", path="evidence")
        evidence_id_set = set(evidence_ids)

        dim_names = [d.name for d in dims]
        if len(set(dim_names)) != len(dim_names):
            raise OrgValidationError("duplicate dim names", path="dims")
        dim_name_set = set(dim_names)

        goal_ids = [g.id for g in goals]
        if len(set(goal_ids)) != len(goal_ids):
            raise OrgValidationError("duplicate goal ids", path="goals")
        goal_id_set = set(goal_ids)

        mechanism_ids = [m.id for m in mechanisms]
        if len(set(mechanism_ids)) != len(mechanism_ids):
            raise OrgValidationError("duplicate mechanism ids", path="mechanisms")
        mechanism_id_set = set(mechanism_ids)

        tensor_ids = [t.id for t in tensors]
        if len(set(tensor_ids)) != len(tensor_ids):
            raise OrgValidationError("duplicate tensor ids", path="tensors")
        tensor_id_set = set(tensor_ids)

        lifetime_ids = [t.id for t in tensor_lifetimes]
        if len(set(lifetime_ids)) != len(lifetime_ids):
            raise OrgValidationError("duplicate tensor_lifetime ids", path="tensor_lifetimes")
        lifetime_id_set = set(lifetime_ids)

        dataflow_ids = [e.id for e in dataflow_edges]
        if len(set(dataflow_ids)) != len(dataflow_ids):
            raise OrgValidationError("duplicate dataflow edge ids", path="dataflow_edges")

        mechanism_topology_ids = [e.id for e in mechanism_topology]
        if len(set(mechanism_topology_ids)) != len(mechanism_topology_ids):
            raise OrgValidationError("duplicate mechanism topology ids", path="mechanism_topology")

        schedule_edge_ids = [e.id for e in schedule_edges]
        if len(set(schedule_edge_ids)) != len(schedule_edge_ids):
            raise OrgValidationError("duplicate schedule edge ids", path="schedule_edges")

        allowed_goal_tags = set(ORG_GOAL_TAGS_BY_KERNEL.get(kernel) or ORG_GOAL_TAGS)
        for idx, goal in enumerate(goals):
            if goal.tag not in allowed_goal_tags:
                raise OrgValidationError(
                    f"unsupported goal tag={goal.tag!r}; allowed={sorted(allowed_goal_tags)}",
                    path=f"goals[{idx}].tag",
                )
            for ref in goal.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"goals[{idx}].evidence_refs")

        for idx, mechanism in enumerate(mechanisms):
            if mechanism.category not in ORG_MECHANISM_CATEGORIES:
                raise OrgValidationError(
                    f"unsupported mechanism category={mechanism.category!r}; allowed={list(ORG_MECHANISM_CATEGORIES)}",
                    path=f"mechanisms[{idx}].category",
                )
            for goal_ref in mechanism.supports_goals:
                if goal_ref not in goal_id_set:
                    raise OrgValidationError("unknown goal ref", path=f"mechanisms[{idx}].supports_goals")
            for dim_ref in mechanism.dims:
                if dim_ref not in dim_name_set:
                    raise OrgValidationError("unknown dim ref", path=f"mechanisms[{idx}].dims")
            for ref in mechanism.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"mechanisms[{idx}].evidence_refs")

        for idx, dim in enumerate(dims):
            for ref in dim.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"dims[{idx}].evidence_refs")

        for idx, tensor in enumerate(tensors):
            if tensor.view_of and tensor.view_of not in tensor_id_set:
                raise OrgValidationError("unknown tensor ref", path=f"tensors[{idx}].view_of")
            for ref in tensor.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"tensors[{idx}].evidence_refs")

        for idx, lifetime in enumerate(tensor_lifetimes):
            if lifetime.tensor not in tensor_id_set:
                raise OrgValidationError("unknown tensor ref", path=f"tensor_lifetimes[{idx}].tensor")
            for mech_ref in lifetime.producer_mechanisms:
                if mech_ref not in mechanism_id_set:
                    raise OrgValidationError("unknown mechanism ref", path=f"tensor_lifetimes[{idx}].producer_mechanisms")
            for mech_ref in lifetime.consumer_mechanisms:
                if mech_ref not in mechanism_id_set:
                    raise OrgValidationError("unknown mechanism ref", path=f"tensor_lifetimes[{idx}].consumer_mechanisms")
            for goal_ref in lifetime.supports_goals:
                if goal_ref not in goal_id_set:
                    raise OrgValidationError("unknown goal ref", path=f"tensor_lifetimes[{idx}].supports_goals")
            for dim_ref in lifetime.dims:
                if dim_ref not in dim_name_set:
                    raise OrgValidationError("unknown dim ref", path=f"tensor_lifetimes[{idx}].dims")
            for ref in lifetime.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"tensor_lifetimes[{idx}].evidence_refs")

        for idx, edge in enumerate(dataflow_edges):
            if edge.src not in lifetime_id_set:
                raise OrgValidationError("unknown tensor lifetime ref", path=f"dataflow_edges[{idx}].src")
            if edge.dst not in lifetime_id_set:
                raise OrgValidationError("unknown tensor lifetime ref", path=f"dataflow_edges[{idx}].dst")
            if edge.tensor not in tensor_id_set:
                raise OrgValidationError("unknown tensor ref", path=f"dataflow_edges[{idx}].tensor")
            for mech_ref in edge.mechanisms:
                if mech_ref not in mechanism_id_set:
                    raise OrgValidationError("unknown mechanism ref", path=f"dataflow_edges[{idx}].mechanisms")
            for ref in edge.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"dataflow_edges[{idx}].evidence_refs")

        for idx, edge in enumerate(mechanism_topology):
            if edge.src not in mechanism_id_set:
                raise OrgValidationError("unknown mechanism ref", path=f"mechanism_topology[{idx}].src")
            if edge.dst not in mechanism_id_set:
                raise OrgValidationError("unknown mechanism ref", path=f"mechanism_topology[{idx}].dst")
            for tensor_ref in edge.tensors:
                if tensor_ref not in tensor_id_set:
                    raise OrgValidationError("unknown tensor ref", path=f"mechanism_topology[{idx}].tensors")
            for lifetime_ref in edge.lifetimes:
                if lifetime_ref not in lifetime_id_set:
                    raise OrgValidationError("unknown tensor lifetime ref", path=f"mechanism_topology[{idx}].lifetimes")
            for ref in edge.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"mechanism_topology[{idx}].evidence_refs")

        schedule_node_ids = set(mechanism_id_set) | set(lifetime_id_set)
        for idx, edge in enumerate(schedule_edges):
            if edge.src not in schedule_node_ids:
                raise OrgValidationError("unknown schedule node ref", path=f"schedule_edges[{idx}].src")
            if edge.dst not in schedule_node_ids:
                raise OrgValidationError("unknown schedule node ref", path=f"schedule_edges[{idx}].dst")
            for lifetime_ref in edge.resources:
                if lifetime_ref not in lifetime_id_set:
                    raise OrgValidationError("unknown tensor lifetime ref", path=f"schedule_edges[{idx}].resources")
            for ref in edge.evidence_refs:
                if ref not in evidence_id_set:
                    raise OrgValidationError("unknown evidence ref", path=f"schedule_edges[{idx}].evidence_refs")

        for ref in source_oracle_obj.evidence_refs:
            if ref not in evidence_id_set:
                raise OrgValidationError("unknown evidence ref", path="source_oracle.evidence_refs")

        return cls(
            schema_version=schema_version,
            kernel=kernel,
            source_context=source_context_obj,
            goals=goals,
            mechanisms=mechanisms,
            dims=dims,
            tensors=tensors,
            tensor_lifetimes=tensor_lifetimes,
            dataflow_edges=dataflow_edges,
            mechanism_topology=mechanism_topology,
            schedule_edges=schedule_edges,
            source_oracle=source_oracle_obj,
            evidence=evidence,
            notes=notes,
        )


def validate_org_doc(
    payload: Any,
    *,
    source_context: Mapping[str, Any] | None = None,
    source_oracle: Mapping[str, Any] | None = None,
) -> OrgDoc:
    return OrgDoc.from_json_dict(payload, source_context=source_context, source_oracle=source_oracle)


__all__ = [
    "EvidenceItem",
    "OrgDataflowEdge",
    "OrgDim",
    "OrgDoc",
    "OrgGoal",
    "OrgMechanism",
    "OrgMechanismTopologyEdge",
    "OrgScheduleEdge",
    "OrgTensor",
    "OrgTensorLifetime",
    "OrgValidationError",
    "SourceContext",
    "SourceOracle",
    "validate_org_doc",
]
