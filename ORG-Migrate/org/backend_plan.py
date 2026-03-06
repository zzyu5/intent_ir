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
        bindings: dict[str, int] = {}
        bindings_raw = obj.get("bindings")
        if isinstance(bindings_raw, Mapping):
            for k, v in dict(bindings_raw).items():
                key = str(k).strip()
                if not key:
                    continue
                try:
                    bindings[key] = int(v)
                except Exception:
                    continue
        return cls(kernel_kind=kind, bindings=bindings, note=str(obj.get("note") or "").strip())


@dataclass(frozen=True)
class BackendModule:
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
        if not module_id or not kind:
            raise ValueError("BackendModule requires non-empty id and kind")
        return cls(
            id=module_id,
            kind=kind,
            provides=[str(x) for x in list(obj.get("provides") or []) if str(x).strip()],
            requires=[str(x) for x in list(obj.get("requires") or []) if str(x).strip()],
            params=[str(x) for x in list(obj.get("params") or []) if str(x).strip()],
            constraints=[str(x) for x in list(obj.get("constraints") or []) if str(x).strip()],
            attrs=(dict(obj.get("attrs") or {}) if isinstance(obj.get("attrs"), Mapping) else {}),
        )


@dataclass(frozen=True)
class BackendModuleEdge:
    src: str
    dst: str
    edge_type: str = "depends_on"
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"src": str(self.src), "dst": str(self.dst), "edge_type": str(self.edge_type or "depends_on")}
        if self.attrs:
            out["attrs"] = dict(self.attrs)
        return out

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> "BackendModuleEdge":
        obj = dict(raw or {})
        src = str(obj.get("src") or "").strip()
        dst = str(obj.get("dst") or "").strip()
        if not src or not dst:
            raise ValueError("BackendModuleEdge requires non-empty src and dst")
        return cls(
            src=src,
            dst=dst,
            edge_type=str(obj.get("edge_type") or obj.get("type") or "depends_on").strip() or "depends_on",
            attrs=(dict(obj.get("attrs") or {}) if isinstance(obj.get("attrs"), Mapping) else {}),
        )


@dataclass
class BackendPlan:
    schema_version: str = BACKEND_PLAN_SCHEMA_VERSION_V1
    kernel: str = ""
    source_oracle: dict[str, Any] = field(default_factory=dict)
    hardware_model: dict[str, Any] = field(default_factory=dict)
    selected_modules: list[BackendModule] = field(default_factory=list)
    module_edges: list[BackendModuleEdge] = field(default_factory=list)
    param_space: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[BackendCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "kernel": str(self.kernel),
            "source_oracle": dict(self.source_oracle or {}),
            "hardware_model": dict(self.hardware_model or {}),
            "selected_modules": [m.to_json_dict() for m in list(self.selected_modules or [])],
            "module_edges": [e.to_json_dict() for e in list(self.module_edges or [])],
            "param_space": dict(self.param_space or {}),
            "constraints": [str(x) for x in list(self.constraints or []) if str(x).strip()],
            "substitutions": [dict(x) for x in list(self.substitutions or []) if isinstance(x, Mapping)],
            "candidates": [c.to_json_dict() for c in list(self.candidates or [])],
            "notes": [str(x) for x in list(self.notes or []) if str(x).strip()],
        }

    @classmethod
    def from_json_dict(cls, raw: Mapping[str, Any]) -> "BackendPlan":
        obj = dict(raw or {})
        selected_modules: list[BackendModule] = []
        for item in list(obj.get("selected_modules") or []):
            if isinstance(item, Mapping):
                selected_modules.append(BackendModule.from_json_dict(item))
        module_edges: list[BackendModuleEdge] = []
        for item in list(obj.get("module_edges") or []):
            if isinstance(item, Mapping):
                module_edges.append(BackendModuleEdge.from_json_dict(item))
        candidates: list[BackendCandidate] = []
        for item in list(obj.get("candidates") or []):
            if isinstance(item, Mapping):
                candidates.append(BackendCandidate.from_json_dict(item))
        substitutions = [dict(x) for x in list(obj.get("substitutions") or []) if isinstance(x, Mapping)]
        return cls(
            schema_version=str(obj.get("schema_version") or BACKEND_PLAN_SCHEMA_VERSION_V1),
            kernel=str(obj.get("kernel") or ""),
            source_oracle=(dict(obj.get("source_oracle") or {}) if isinstance(obj.get("source_oracle"), Mapping) else {}),
            hardware_model=(dict(obj.get("hardware_model") or {}) if isinstance(obj.get("hardware_model"), Mapping) else {}),
            selected_modules=selected_modules,
            module_edges=module_edges,
            param_space=(dict(obj.get("param_space") or {}) if isinstance(obj.get("param_space"), Mapping) else {}),
            constraints=[str(x) for x in list(obj.get("constraints") or []) if str(x).strip()],
            substitutions=substitutions,
            candidates=candidates,
            notes=[str(x) for x in list(obj.get("notes") or []) if str(x).strip()],
        )


__all__ = ["BackendCandidate", "BackendModule", "BackendModuleEdge", "BackendPlan"]
