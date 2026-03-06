from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import BACKEND_PLAN_SCHEMA_VERSION_V1


@dataclass(frozen=True)
class BackendCandidate:
    kernel_kind: str
    bindings: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kernel_kind": str(self.kernel_kind),
            "bindings": {str(k): int(v) for k, v in dict(self.bindings or {}).items() if str(k).strip()},
        }
        if str(self.note or "").strip():
            out["note"] = str(self.note)
        return out

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> "BackendCandidate":
        obj = dict(raw or {})
        kind = str(obj.get("kernel_kind") or "").strip()
        if not kind:
            raise ValueError("BackendCandidate.kernel_kind must be non-empty")
        bindings_raw = obj.get("bindings")
        bindings: dict[str, int] = {}
        if isinstance(bindings_raw, Mapping):
            for k, v in dict(bindings_raw).items():
                key = str(k).strip()
                if not key:
                    continue
                try:
                    bindings[key] = int(v)
                except Exception:
                    continue
        note = str(obj.get("note") or "").strip()
        return cls(kernel_kind=kind, bindings=bindings, note=note)


@dataclass(frozen=True)
class BackendModule:
    """
    Minimal, explainable backend building block for ORG -> codegen planning.

    Phase-1 note: modules may correspond to coarse-grained templates, but the
    interface is "module-like" so we can refine to finer-grained composition
    later without changing the ORG contract.
    """

    id: str
    kind: str
    provides: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": str(self.id),
            "kind": str(self.kind),
            "provides": [str(x) for x in list(self.provides or []) if str(x).strip()],
            "requires": [str(x) for x in list(self.requires or []) if str(x).strip()],
            "params": [str(x) for x in list(self.params or []) if str(x).strip()],
            "constraints": [str(x) for x in list(self.constraints or []) if str(x).strip()],
        }
        if self.attrs:
            out["attrs"] = dict(self.attrs)
        return out

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> "BackendModule":
        obj = dict(raw or {})
        module_id = str(obj.get("id") or "").strip()
        kind = str(obj.get("kind") or "").strip()
        if not module_id:
            raise ValueError("BackendModule.id must be non-empty")
        if not kind:
            raise ValueError("BackendModule.kind must be non-empty")
        provides = [str(x) for x in list(obj.get("provides") or []) if str(x).strip()]
        requires = [str(x) for x in list(obj.get("requires") or []) if str(x).strip()]
        params = [str(x) for x in list(obj.get("params") or []) if str(x).strip()]
        constraints = [str(x) for x in list(obj.get("constraints") or []) if str(x).strip()]
        attrs = dict(obj.get("attrs") or {}) if isinstance(obj.get("attrs"), Mapping) else {}
        return cls(
            id=module_id,
            kind=kind,
            provides=provides,
            requires=requires,
            params=params,
            constraints=constraints,
            attrs=attrs,
        )


@dataclass(frozen=True)
class BackendModuleEdge:
    src: str
    dst: str
    edge_type: str = "depends_on"
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"src": str(self.src), "dst": str(self.dst)}
        if str(self.edge_type or "").strip():
            out["edge_type"] = str(self.edge_type)
        if self.attrs:
            out["attrs"] = dict(self.attrs)
        return out

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> "BackendModuleEdge":
        obj = dict(raw or {})
        src = str(obj.get("src") or "").strip()
        dst = str(obj.get("dst") or "").strip()
        if not src:
            raise ValueError("BackendModuleEdge.src must be non-empty")
        if not dst:
            raise ValueError("BackendModuleEdge.dst must be non-empty")
        edge_type = str(obj.get("edge_type") or obj.get("type") or "depends_on").strip() or "depends_on"
        attrs = dict(obj.get("attrs") or {}) if isinstance(obj.get("attrs"), Mapping) else {}
        return cls(src=src, dst=dst, edge_type=edge_type, attrs=attrs)


@dataclass
class BackendPlan:
    schema_version: str = BACKEND_PLAN_SCHEMA_VERSION_V1
    kernel: str = ""
    target: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    modules: list[BackendModule] = field(default_factory=list)
    module_edges: list[BackendModuleEdge] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)
    selected_variants: list[str] = field(default_factory=list)
    param_space: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    candidates: list[BackendCandidate] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": str(self.schema_version),
            "kernel": str(self.kernel),
            "target": str(self.target),
            "hardware": dict(self.hardware or {}),
            "modules": [m.to_json_dict() for m in list(self.modules or [])],
            "module_edges": [e.to_json_dict() for e in list(self.module_edges or [])],
            "passes": [str(x) for x in list(self.passes or []) if str(x).strip()],
            "selected_variants": [str(x) for x in list(self.selected_variants or []) if str(x).strip()],
            "param_space": dict(self.param_space or {}),
            "constraints": [str(x) for x in list(self.constraints or []) if str(x).strip()],
            "trace": dict(self.trace or {}),
            "candidates": [c.to_json_dict() for c in list(self.candidates or [])],
        }
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> "BackendPlan":
        obj = dict(raw or {})
        schema = str(obj.get("schema_version") or "").strip() or BACKEND_PLAN_SCHEMA_VERSION_V1
        kernel = str(obj.get("kernel") or "").strip()
        target = str(obj.get("target") or "").strip()
        hardware = dict(obj.get("hardware") or {}) if isinstance(obj.get("hardware"), Mapping) else {}
        modules_raw = obj.get("modules") or []
        modules: list[BackendModule] = []
        if isinstance(modules_raw, list):
            for x in modules_raw:
                if isinstance(x, Mapping):
                    modules.append(BackendModule.from_json_dict(x))
        module_edges_raw = obj.get("module_edges") or obj.get("module_edge") or []
        module_edges: list[BackendModuleEdge] = []
        if isinstance(module_edges_raw, list):
            for x in module_edges_raw:
                if isinstance(x, Mapping):
                    module_edges.append(BackendModuleEdge.from_json_dict(x))
        passes = [str(x) for x in list(obj.get("passes") or []) if str(x).strip()]
        selected_variants = [str(x) for x in list(obj.get("selected_variants") or []) if str(x).strip()]
        param_space = dict(obj.get("param_space") or {}) if isinstance(obj.get("param_space"), Mapping) else {}
        constraints = [str(x) for x in list(obj.get("constraints") or []) if str(x).strip()]
        trace = dict(obj.get("trace") or {}) if isinstance(obj.get("trace"), Mapping) else {}
        candidates_raw = obj.get("candidates") or []
        candidates: list[BackendCandidate] = []
        if isinstance(candidates_raw, list):
            for x in candidates_raw:
                if isinstance(x, Mapping):
                    candidates.append(BackendCandidate.from_json_dict(x))
        meta = dict(obj.get("meta") or {}) if isinstance(obj.get("meta"), Mapping) else {}
        return cls(
            schema_version=schema,
            kernel=kernel,
            target=target,
            hardware=hardware,
            modules=modules,
            module_edges=module_edges,
            passes=passes,
            selected_variants=selected_variants,
            param_space=param_space,
            constraints=constraints,
            trace=trace,
            candidates=candidates,
            meta=meta,
        )


__all__ = [
    "BackendCandidate",
    "BackendModule",
    "BackendModuleEdge",
    "BackendPlan",
]
