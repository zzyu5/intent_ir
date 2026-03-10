"""
LLMOrgHub: unified "KernelDescriptor (+ evidence bundle) -> OrgDoc" entrypoint.

The LLM is responsible for rationale-bearing sections only. Runtime injects:
- source_context
- source_oracle
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from pipeline.interfaces import KernelDescriptor

from intent_ir.llm import DEFAULT_MODEL, LLMClientError, extract_json_object_with_trace
from org.schema import OrgDoc, OrgValidationError, validate_org_doc


def _hash_messages(messages: List[Dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _maybe_truncate_source(source_text: str) -> str:
    text = str(source_text)
    lines = text.splitlines()
    max_lines = 1200
    max_chars = 60000
    head = 400
    tail = 120
    if len(text) <= max_chars and len(lines) <= max_lines:
        return text
    head_lines = lines[: max(0, int(head))]
    tail_lines = lines[-max(0, int(tail)) :] if int(tail) > 0 else []
    banner = f"[IntentIR][ORG] SOURCE TRUNCATED: original_lines={len(lines)} kept_head={len(head_lines)} kept_tail={len(tail_lines)}"
    return "\n".join([banner, *head_lines, "[IntentIR][ORG] ... TRUNCATED ...", *tail_lines])


def _ordered_evidence_blob(
    descriptor: KernelDescriptor,
    *,
    intent_summary: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None,
) -> str:
    extra_dict = dict(extra or {}) if isinstance(extra, Mapping) else {}
    ordered: dict[str, Any] = {
        "kernel": descriptor.name,
        "frontend": descriptor.frontend,
    }
    for key in ("ttgir_facts", "ptx_facts", "source_oracle_facts", "ttir_summary"):
        value = extra_dict.get(key)
        if isinstance(value, Mapping):
            ordered[key] = dict(value)
    if isinstance(intent_summary, Mapping):
        ordered["intent_summary"] = dict(intent_summary)
    ordered["io_spec"] = dict(getattr(descriptor, "io_spec", {}) or {})
    ordered["launch"] = dict(getattr(descriptor, "launch", {}) or {})
    frontend_facts = dict(getattr(descriptor, "frontend_facts", {}) or {})
    if frontend_facts:
        ordered["frontend_facts"] = frontend_facts
    frontend_constraints = dict(getattr(descriptor, "frontend_constraints", {}) or {})
    if frontend_constraints:
        ordered["frontend_constraints"] = frontend_constraints
    runtime_extra = {
        str(k): v
        for k, v in extra_dict.items()
        if str(k) not in {"ttgir_facts", "ptx_facts", "source_oracle_facts", "ttir_summary"} and str(k).strip()
    }
    if runtime_extra:
        ordered["extra"] = runtime_extra
    return json.dumps(ordered, ensure_ascii=False)


def _build_source_context(
    descriptor: KernelDescriptor,
    *,
    extra_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    extra = dict(extra_evidence or {}) if isinstance(extra_evidence, Mapping) else {}
    artifacts: dict[str, str | None] = {}
    art = getattr(descriptor, "artifacts", None)
    for key in ("ttir_path", "ttgir_path", "ptx_text"):
        value = getattr(art, key, None) if art is not None else None
        if value is not None:
            artifacts[key.replace("_text", "")] = str(value)
    meta = dict(getattr(descriptor, "meta", {}) or {})
    for key in ("ttir_original_path", "ttgir_original_path", "ptx_original_path"):
        if meta.get(key) is not None:
            artifacts[key] = str(meta.get(key))
    return {
        "frontend": str(descriptor.frontend),
        "source_arch": str(extra.get("source_arch") or ""),
        "target_arch": str(extra.get("target_arch") or ""),
        "shape_bindings": {str(k): int(v) for k, v in dict(extra.get("shape_bindings") or {}).items() if str(k).strip()},
        "artifacts": artifacts,
    }


def _build_source_oracle(extra_evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    extra = dict(extra_evidence or {}) if isinstance(extra_evidence, Mapping) else {}
    facts = extra.get("source_oracle_facts")
    if isinstance(facts, Mapping):
        oracle = dict((dict(facts).get("oracle") or {}))
        return {
            "kernel_kind": str(oracle.get("kernel_kind") or ""),
            "bindings": {str(k): int(v) for k, v in dict(oracle.get("bindings") or {}).items() if str(k).strip()},
            "arch": str(oracle.get("arch") or ""),
            "compiler_stack": str(oracle.get("compiler_stack") or ""),
            "evidence_refs": [str(x) for x in list(oracle.get("evidence_refs") or []) if str(x).strip()],
        }
    return {
        "kernel_kind": "",
        "bindings": {},
        "arch": str(extra.get("source_arch") or ""),
        "compiler_stack": str(extra.get("source_compiler_stack") or ""),
        "evidence_refs": [],
    }


def _sanitize_raw_org_json(raw_json: Mapping[str, Any] | None) -> dict[str, Any]:
    obj = dict(raw_json or {})
    dims = [dict(x) for x in list(obj.get("dims") or []) if isinstance(x, Mapping)]
    dim_names = {str(item.get("name") or "").strip() for item in dims if str(item.get("name") or "").strip()}
    goal_ids = {str(item.get("id") or "").strip() for item in list(obj.get("goals") or []) if isinstance(item, Mapping)}
    evidence_ids = {
        str(item.get("id") or "").strip() for item in list(obj.get("evidence") or []) if isinstance(item, Mapping)
    }
    mechanisms_out: list[dict[str, Any]] = []
    for raw_mech in list(obj.get("mechanisms") or []):
        if not isinstance(raw_mech, Mapping):
            continue
        mech = dict(raw_mech)
        dims_list = []
        for raw_dim in list(mech.get("dims") or []):
            name = str(raw_dim or "").strip()
            if not name or name not in dim_names:
                continue
            dims_list.append(name)
        mech["dims"] = dims_list
        mechanisms_out.append(mech)
    if mechanisms_out:
        obj["mechanisms"] = mechanisms_out
    mechanism_ids = {
        str(item.get("id") or "").strip() for item in list(obj.get("mechanisms") or []) if isinstance(item, Mapping)
    }
    raw_tensor_ids = {
        str(item.get("id") or "").strip() for item in list(obj.get("tensors") or []) if isinstance(item, Mapping)
    }
    tensors_out: list[dict[str, Any]] = []
    for raw_tensor in list(obj.get("tensors") or []):
        if not isinstance(raw_tensor, Mapping):
            continue
        tensor = dict(raw_tensor)
        view_of = str(tensor.get("view_of") or "").strip()
        if view_of and view_of not in raw_tensor_ids:
            tensor["view_of"] = ""
        tensor["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(tensor.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
        tensors_out.append(tensor)
    if tensors_out or "tensors" in obj:
        obj["tensors"] = tensors_out
    tensor_ids = {
        str(item.get("id") or "").strip() for item in list(obj.get("tensors") or []) if isinstance(item, Mapping)
    }
    lifetimes_out: list[dict[str, Any]] = []
    for raw_lifetime in list(obj.get("tensor_lifetimes") or []):
        if not isinstance(raw_lifetime, Mapping):
            continue
        lifetime = dict(raw_lifetime)
        tensor_id = str(lifetime.get("tensor") or "").strip()
        if tensor_id and tensor_id not in tensor_ids:
            continue
        lifetime["producer_mechanisms"] = [
            ref
            for ref in [str(x).strip() for x in list(lifetime.get("producer_mechanisms") or []) if str(x).strip()]
            if ref in mechanism_ids
        ]
        lifetime["consumer_mechanisms"] = [
            ref
            for ref in [str(x).strip() for x in list(lifetime.get("consumer_mechanisms") or []) if str(x).strip()]
            if ref in mechanism_ids
        ]
        lifetime["supports_goals"] = [
            ref for ref in [str(x).strip() for x in list(lifetime.get("supports_goals") or []) if str(x).strip()] if ref in goal_ids
        ]
        lifetime["dims"] = [
            ref for ref in [str(x).strip() for x in list(lifetime.get("dims") or []) if str(x).strip()] if ref in dim_names
        ]
        lifetime["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(lifetime.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
        lifetimes_out.append(lifetime)
    if lifetimes_out or "tensor_lifetimes" in obj:
        obj["tensor_lifetimes"] = lifetimes_out
    lifetime_ids = {
        str(item.get("id") or "").strip() for item in list(obj.get("tensor_lifetimes") or []) if isinstance(item, Mapping)
    }
    dataflow_out: list[dict[str, Any]] = []
    for raw_edge in list(obj.get("dataflow_edges") or []):
        if not isinstance(raw_edge, Mapping):
            continue
        edge = dict(raw_edge)
        if str(edge.get("src") or "").strip() not in lifetime_ids:
            continue
        if str(edge.get("dst") or "").strip() not in lifetime_ids:
            continue
        if str(edge.get("tensor") or "").strip() not in tensor_ids:
            continue
        edge["mechanisms"] = [
            ref for ref in [str(x).strip() for x in list(edge.get("mechanisms") or []) if str(x).strip()] if ref in mechanism_ids
        ]
        edge["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(edge.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
        dataflow_out.append(edge)
    if dataflow_out or "dataflow_edges" in obj:
        obj["dataflow_edges"] = dataflow_out
    topology_out: list[dict[str, Any]] = []
    for raw_edge in list(obj.get("mechanism_topology") or []):
        if not isinstance(raw_edge, Mapping):
            continue
        edge = dict(raw_edge)
        if str(edge.get("src") or "").strip() not in mechanism_ids:
            continue
        if str(edge.get("dst") or "").strip() not in mechanism_ids:
            continue
        edge["tensors"] = [
            ref for ref in [str(x).strip() for x in list(edge.get("tensors") or []) if str(x).strip()] if ref in tensor_ids
        ]
        edge["lifetimes"] = [
            ref for ref in [str(x).strip() for x in list(edge.get("lifetimes") or []) if str(x).strip()] if ref in lifetime_ids
        ]
        edge["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(edge.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
        topology_out.append(edge)
    if topology_out or "mechanism_topology" in obj:
        obj["mechanism_topology"] = topology_out
    schedule_out: list[dict[str, Any]] = []
    schedule_node_ids = set(mechanism_ids) | set(lifetime_ids)
    for raw_edge in list(obj.get("schedule_edges") or []):
        if not isinstance(raw_edge, Mapping):
            continue
        edge = dict(raw_edge)
        if str(edge.get("src") or "").strip() not in schedule_node_ids:
            continue
        if str(edge.get("dst") or "").strip() not in schedule_node_ids:
            continue
        edge["resources"] = [
            ref for ref in [str(x).strip() for x in list(edge.get("resources") or []) if str(x).strip()] if ref in lifetime_ids
        ]
        edge["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(edge.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
        schedule_out.append(edge)
    if schedule_out or "schedule_edges" in obj:
        obj["schedule_edges"] = schedule_out
    region_graph = obj.get("region_graph")
    if isinstance(region_graph, Mapping):
        region_graph_obj = dict(region_graph)
        raw_regions = [dict(x) for x in list(region_graph_obj.get("regions") or []) if isinstance(x, Mapping)]
        raw_region_ids = {
            str(item.get("id") or "").strip()
            for item in raw_regions
            if str(item.get("id") or "").strip()
        }
        regions_out: list[dict[str, Any]] = []
        for raw_region in raw_regions:
            region = dict(raw_region)
            parent = str(region.get("parent") or "").strip()
            if parent and parent not in raw_region_ids:
                region["parent"] = ""
            region["entry_mechanisms"] = [
                ref
                for ref in [str(x).strip() for x in list(region.get("entry_mechanisms") or []) if str(x).strip()]
                if ref in mechanism_ids
            ]
            region["exit_mechanisms"] = [
                ref
                for ref in [str(x).strip() for x in list(region.get("exit_mechanisms") or []) if str(x).strip()]
                if ref in mechanism_ids
            ]
            region["evidence_refs"] = [
                ref
                for ref in [str(x).strip() for x in list(region.get("evidence_refs") or []) if str(x).strip()]
                if ref in evidence_ids
            ]
            regions_out.append(region)
        region_ids = {
            str(item.get("id") or "").strip()
            for item in regions_out
            if str(item.get("id") or "").strip()
        }
        edges_out: list[dict[str, Any]] = []
        for raw_edge in list(region_graph_obj.get("edges") or []):
            if not isinstance(raw_edge, Mapping):
                continue
            edge = dict(raw_edge)
            if str(edge.get("src") or "").strip() not in region_ids:
                continue
            if str(edge.get("dst") or "").strip() not in region_ids:
                continue
            edge["lifetimes"] = [
                ref
                for ref in [str(x).strip() for x in list(edge.get("lifetimes") or []) if str(x).strip()]
                if ref in lifetime_ids
            ]
            edge["mechanisms"] = [
                ref
                for ref in [str(x).strip() for x in list(edge.get("mechanisms") or []) if str(x).strip()]
                if ref in mechanism_ids
            ]
            edge["evidence_refs"] = [
                ref
                for ref in [str(x).strip() for x in list(edge.get("evidence_refs") or []) if str(x).strip()]
                if ref in evidence_ids
            ]
            edges_out.append(edge)
        obj["region_graph"] = {"regions": regions_out, "edges": edges_out}
    return obj


@dataclass(frozen=True)
class CandidateOrg:
    org: OrgDoc
    raw_json: dict[str, Any]
    llm_trace: dict[str, Any]
    prompt_hash: str = ""


@dataclass
class LLMOrgHub:
    default_model: str = DEFAULT_MODEL
    timeout_s: int = 600
    http_max_retries: int = 4
    http_max_total_wait_s: int = 180
    max_parse_retries: int = 2
    max_schema_retries: int = 1
    extra_chat_kwargs: Dict[str, Any] = field(default_factory=dict)

    def lift(
        self,
        descriptor: KernelDescriptor,
        *,
        intent_summary: Mapping[str, Any] | None = None,
        extra_evidence: Mapping[str, Any] | None = None,
        model: Optional[str] = None,
    ) -> CandidateOrg:
        requested = str(model or self.default_model)
        evidence = _ordered_evidence_blob(descriptor, intent_summary=intent_summary, extra=extra_evidence)
        extra_instruction = "\n".join(
            [
                "Evidence appendix (JSON):",
                evidence,
                "",
                "Hard rule: return ONE ORG JSON object with goals/mechanisms/dims/tensors/tensor_lifetimes/dataflow_edges/mechanism_topology/schedule_edges/region_graph(optional)/evidence only.",
                "Runtime will inject source_context and source_oracle; do not invent backend mappings or target parameter values.",
            ]
        ).strip()

        src = _maybe_truncate_source(descriptor.source_text)
        compact = bool(src.startswith("[IntentIR][ORG] SOURCE TRUNCATED"))
        if descriptor.frontend == "triton":
            from org.frontends.triton.llm_org import build_messages  # noqa: PLC0415

            messages = build_messages(src, kernel_name=descriptor.name, extra_instruction=extra_instruction, compact=compact)
        else:
            raise NotImplementedError(f"LLMOrgHub does not support frontend={descriptor.frontend}")

        chat_kwargs = dict(self.extra_chat_kwargs)
        chat_kwargs.setdefault("max_tokens", 4096)
        chat_kwargs.setdefault("temperature", 0)
        chat_kwargs.setdefault("timeout", int(self.timeout_s))
        chat_kwargs.setdefault("max_retries", int(self.http_max_retries))
        chat_kwargs.setdefault("max_total_wait_s", int(self.http_max_total_wait_s))

        raw_json: dict[str, Any] | None = None
        trace: dict[str, Any] = {}
        cur_messages = list(messages)
        cur_prompt_hash = _hash_messages(messages)
        source_context = _build_source_context(descriptor, extra_evidence=extra_evidence)
        source_oracle = _build_source_oracle(extra_evidence)

        for attempt in range(max(0, int(self.max_schema_retries)) + 1):
            cur_prompt_hash = _hash_messages(cur_messages)
            try:
                raw_json, trace = extract_json_object_with_trace(
                    cur_messages,
                    model=requested,
                    max_parse_retries=int(self.max_parse_retries),
                    **chat_kwargs,
                )
            except LLMClientError as exc:
                raise LLMClientError(f"ORG LLM failed: {exc}") from exc

            try:
                sanitized = _sanitize_raw_org_json(raw_json)
                org = validate_org_doc(sanitized, source_context=source_context, source_oracle=source_oracle)
                return CandidateOrg(org=org, raw_json=dict(sanitized), llm_trace=dict(trace), prompt_hash=str(cur_prompt_hash))
            except OrgValidationError as exc:
                if attempt >= int(self.max_schema_retries):
                    raise OrgValidationError(f"invalid ORG JSON: {exc}", path=getattr(exc, "path", "")) from exc
                repair_user = (
                    "Your previous ORG JSON failed schema validation.\n"
                    f"Error: {exc}\n\n"
                    "Return ONE corrected ORG JSON object only.\n"
                    "Keep top-level keys: schema_version, kernel, goals, mechanisms, dims, tensors, tensor_lifetimes, dataflow_edges, mechanism_topology, schedule_edges, region_graph(optional), evidence, notes(optional).\n"
                    "Do not emit source_context/source_oracle; runtime injects them.\n"
                )
                prev = json.dumps(raw_json, ensure_ascii=False, sort_keys=True) if raw_json is not None else ""
                cur_messages = list(messages)
                if prev:
                    cur_messages.append({"role": "assistant", "content": prev})
                cur_messages.append({"role": "user", "content": repair_user})
                continue
            except Exception as exc:
                raise OrgValidationError(f"invalid ORG JSON: {type(exc).__name__}: {exc}") from exc

        raise OrgValidationError("invalid ORG JSON: exceeded schema retries")


__all__ = ["CandidateOrg", "LLMOrgHub"]
