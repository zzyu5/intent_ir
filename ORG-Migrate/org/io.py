from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schema import OrgDoc, validate_org_doc


@dataclass(frozen=True)
class OrgSeed:
    schema_version: str = "org_seed_v1"
    generated_at: str = ""
    kernel: str = ""
    triton_provider: str = ""
    backend_target: str | None = None
    org: OrgDoc | None = None
    raw_json: dict[str, Any] | None = None
    llm_trace: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "generated_at": str(self.generated_at),
            "kernel": str(self.kernel),
            "triton_provider": str(self.triton_provider),
            "backend_target": (str(self.backend_target) if self.backend_target is not None else None),
            "org": (self.org.to_json_dict() if isinstance(self.org, OrgDoc) else None),
            "raw_json": (dict(self.raw_json) if isinstance(self.raw_json, Mapping) else None),
            "llm_trace": (dict(self.llm_trace) if isinstance(self.llm_trace, Mapping) else {}),
            "quality": (dict(self.quality) if isinstance(self.quality, Mapping) else {}),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "OrgSeed":
        obj = dict(payload or {})
        org_raw = obj.get("org")
        org = validate_org_doc(org_raw) if isinstance(org_raw, Mapping) else None
        raw_json = obj.get("raw_json")
        return cls(
            schema_version=str(obj.get("schema_version") or "org_seed_v1"),
            generated_at=str(obj.get("generated_at") or ""),
            kernel=str(obj.get("kernel") or ""),
            triton_provider=str(obj.get("triton_provider") or ""),
            backend_target=(str(obj.get("backend_target")) if obj.get("backend_target") is not None else None),
            org=org,
            raw_json=(dict(raw_json) if isinstance(raw_json, Mapping) else None),
            llm_trace=(dict(obj.get("llm_trace") or {}) if isinstance(obj.get("llm_trace"), Mapping) else {}),
            quality=(dict(obj.get("quality") or {}) if isinstance(obj.get("quality"), Mapping) else {}),
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blindfold_enabled() -> bool:
    raw = str(os.environ.get("INTENTIR_ORG_BLINDFOLD") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _blindfold_label() -> str:
    raw = str(os.environ.get("INTENTIR_ORG_BLINDFOLD_LABEL") or "").strip()
    return raw or "target_kernel_func"


def _blindfold_seed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not _blindfold_enabled():
        return dict(payload or {})
    label = _blindfold_label()
    obj = dict(payload or {})
    obj["kernel"] = label
    org = obj.get("org")
    if isinstance(org, Mapping):
        org_obj = dict(org)
        org_obj["kernel"] = label
        obj["org"] = org_obj
    raw_json = obj.get("raw_json")
    if isinstance(raw_json, Mapping):
        raw_obj = dict(raw_json)
        raw_obj["kernel"] = label
        obj["raw_json"] = raw_obj
    return obj


def save_org_seed(
    *,
    path: Path,
    kernel: str,
    triton_provider: str,
    backend_target: str | None,
    org: OrgDoc,
    raw_json: Mapping[str, Any] | None,
    llm_trace: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
) -> None:
    seed = OrgSeed(
        generated_at=_utc_now_iso(),
        kernel=str(kernel),
        triton_provider=str(triton_provider),
        backend_target=(str(backend_target) if backend_target is not None else None),
        org=org,
        raw_json=(dict(raw_json) if isinstance(raw_json, Mapping) else None),
        llm_trace=(dict(llm_trace) if isinstance(llm_trace, Mapping) else {}),
        quality=(dict(quality) if isinstance(quality, Mapping) else {}),
    )
    payload = _blindfold_seed_payload(seed.to_json_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_org_seed(path: Path) -> OrgSeed:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("org_seed payload must be an object")
    return OrgSeed.from_json_dict(payload)


def save_org_doc(path: Path, org: OrgDoc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(org.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_org_doc(path: Path) -> OrgDoc:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_org_doc(payload)


def is_seed_trusted_for_auto(seed: OrgSeed) -> tuple[bool, str]:
    """
    Mirror IntentIR intent_seed policy: only trust cached ORG when previous run
    proved the semantic lift (diff+static) for that kernel.
    """
    q = dict(seed.quality or {})
    if not q:
        return False, "untrusted_seed:missing_quality"
    if not bool(q.get("diff_ok")):
        return False, "untrusted_seed:diff_not_ok"
    if not bool(q.get("static_ok")):
        return False, "untrusted_seed:static_not_ok"
    return True, "trusted"


__all__ = [
    "OrgSeed",
    "is_seed_trusted_for_auto",
    "load_org_doc",
    "load_org_seed",
    "save_org_doc",
    "save_org_seed",
]
