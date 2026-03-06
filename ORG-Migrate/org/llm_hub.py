"""
LLMOrgHub: unified "KernelDescriptor (+ Intent summary) -> OrgDoc" entrypoint.

This is intentionally separate from IntentIR semantic lifting:
- IntentIR LLM (Task1/2) outputs IntentFunction (semantic IR).
- ORG LLM (this module) outputs optimization rationale (why/how/dims + evidence).

ORG output must NOT perform hardware mapping; mapping is deterministic in backends.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from pipeline.interfaces import KernelDescriptor

from intent_ir.llm import DEFAULT_MODEL, LLMClientError, extract_json_object_with_trace
from org.schema import OrgDoc, OrgValidationError, validate_org_doc


def _dim_naming_hints(kernel_name: str) -> list[str]:
    k = str(kernel_name or "").strip()
    if not k:
        return []
    if k == "flash_attention2d":
        return [
            "- If you mention the KV tile length, name the dim `ATTN_BLOCK_KV` (alias: `BLOCK_KV`).",
            "- If you mention the number of score warps, name the dim `ATTN_SCORE_WARPS`.",
        ]
    if k == "_attn_fwd":
        return [
            "- If you mention the Q tile size, name the dim `ATTN_FWD_BLOCK_M` (alias: `BLOCK_M`).",
            "- If you mention the KV tile size, name the dim `ATTN_FWD_BLOCK_KV` (alias: `BLOCK_KV`).",
        ]
    if k in {"ai_bench_matmul", "matmul_fused_epilogue2d"}:
        return [
            "- If you mention MMA tile sizes, name dims `MMA_BM`, `MMA_BN`, `MMA_BK` (aliases: `BLOCK_M/N/K`).",
            "- If you mention async copy / double-buffering, include dim name `MMA_ASYNC_COPY` (allowed set may include 0/1).",
        ]
    if k == "ai_bench_softmax":
        return [
            "- If you mention the block thread count, name the dim `SOFTMAX_BLOCK_THREADS` (alias: `BLOCK_THREADS`).",
            "- If you mention vec4 vectorization, include dim name `SOFTMAX_VEC4` (allowed set may include 0/1).",
        ]
    return []


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
    try:
        if len(text) <= max_chars and len(lines) <= max_lines:
            return text
    except Exception:
        return text
    head_lines = lines[: max(0, int(head))]
    tail_lines = lines[-max(0, int(tail)) :] if int(tail) > 0 else []
    banner = f"[IntentIR][ORG] SOURCE TRUNCATED: original_lines={len(lines)} kept_head={len(head_lines)} kept_tail={len(tail_lines)}"
    return "\n".join([banner, *head_lines, "[IntentIR][ORG] ... TRUNCATED ...", *tail_lines])


def _evidence_blob(descriptor: KernelDescriptor, *, intent_summary: Mapping[str, Any] | None, extra: Mapping[str, Any] | None) -> str:
    """
    Keep evidence bounded and stable so provider prompts stay small.
    """

    def _summarize_frontend_constraints(fc: Any) -> Any:
        if not isinstance(fc, dict):
            return fc
        out: Dict[str, Any] = {}
        for k in ("needs_mask", "suggested_edge_cases"):
            if k in fc:
                out[k] = fc.get(k)

        meta = fc.get("meta")
        if not isinstance(meta, dict):
            if "meta" in fc:
                out["meta"] = meta
            return out

        meta_out: Dict[str, Any] = {}
        for k in ("symbol_ranges", "tile_hints", "static_ints"):
            if k in meta:
                meta_out[k] = meta.get(k)

        pc = meta.get("predicate_clauses")
        if isinstance(pc, list):
            clipped: List[str] = []
            for x in pc[:64]:
                s = str(x)
                if len(s) > 256:
                    s = s[:256] + "…"
                if s.strip():
                    clipped.append(s)
            meta_out["predicate_clauses"] = clipped

        aw = meta.get("access_witness")
        if isinstance(aw, dict):
            accesses = aw.get("accesses")
            meta_out["access_witness_summary"] = {
                "num_accesses": (len(accesses) if isinstance(accesses, list) else None),
                "tensor_penalty": aw.get("tensor_penalty"),
                "dominant_axis": aw.get("dominant_axis"),
                "dominant_range": aw.get("dominant_range"),
                "dominant_range_len": aw.get("dominant_range_len"),
                "has_contiguous_range": aw.get("has_contiguous_range"),
                "notes": (list(aw.get("notes") or [])[:8] if isinstance(aw.get("notes"), list) else None),
            }

        if meta_out:
            out["meta"] = meta_out
        return out

    ev: dict[str, Any] = {
        "kernel": descriptor.name,
        "frontend": descriptor.frontend,
        "io_spec": descriptor.io_spec,
        "launch": descriptor.launch,
        "frontend_facts": descriptor.frontend_facts,
        "frontend_constraints": _summarize_frontend_constraints(descriptor.frontend_constraints),
        "meta": {
            "versions": {k: descriptor.meta.get(k) for k in ("triton", "torch", "tilelang") if descriptor.meta.get(k) is not None}
        },
    }
    if isinstance(intent_summary, Mapping):
        ev["intent_summary"] = dict(intent_summary)
    if isinstance(extra, Mapping) and dict(extra):
        ev["extra"] = dict(extra)
    return json.dumps(ev, ensure_ascii=False)


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
        evidence = _evidence_blob(descriptor, intent_summary=intent_summary, extra=extra_evidence)
        dim_hints = _dim_naming_hints(descriptor.name)
        extra_lines: list[str] = [
            "Evidence appendix (JSON):",
            evidence,
            "",
            "Hard rules: output must be ONE strict ORG JSON object (intentir_org_v1).",
            "Do NOT map to hardware or output numeric parameter values; only dims + constraints + why/how.",
        ]
        if dim_hints:
            extra_lines.extend(["", "Dim naming hints (for backend mapper):", *dim_hints])
        extra_instruction = "\n".join(extra_lines).strip()

        src = _maybe_truncate_source(descriptor.source_text)
        compact = bool(src.startswith("[IntentIR][ORG] SOURCE TRUNCATED"))

        if descriptor.frontend == "triton":
            from org.frontends.triton.llm_org import build_messages  # noqa: PLC0415

            messages = build_messages(
                src,
                kernel_name=descriptor.name,
                extra_instruction=extra_instruction,
                compact=compact,
            )
        else:
            raise NotImplementedError(f"LLMOrgHub does not support frontend={descriptor.frontend}")

        prompt_hash = _hash_messages(messages)
        chat_kwargs = dict(self.extra_chat_kwargs)
        chat_kwargs.setdefault("max_tokens", 4096)
        chat_kwargs.setdefault("temperature", 0)
        chat_kwargs.setdefault("timeout", int(self.timeout_s))
        chat_kwargs.setdefault("max_retries", int(self.http_max_retries))
        chat_kwargs.setdefault("max_total_wait_s", int(self.http_max_total_wait_s))

        raw_json: dict[str, Any] | None = None
        trace: dict[str, Any] = {}
        cur_messages = list(messages)
        cur_prompt_hash = str(prompt_hash)

        for attempt in range(max(0, int(self.max_schema_retries)) + 1):
            cur_prompt_hash = _hash_messages(cur_messages)
            try:
                raw_json, trace = extract_json_object_with_trace(
                    cur_messages,
                    model=requested,
                    max_parse_retries=int(self.max_parse_retries),
                    **chat_kwargs,
                )
            except LLMClientError as e:
                raise LLMClientError(f"ORG LLM failed: {e}") from e

            try:
                org = validate_org_doc(raw_json)
                return CandidateOrg(org=org, raw_json=dict(raw_json), llm_trace=dict(trace), prompt_hash=str(cur_prompt_hash))
            except OrgValidationError as e:
                # Repair loop: ask the model to fix schema issues without re-mapping to hardware.
                if attempt >= int(self.max_schema_retries):
                    raise OrgValidationError(f"invalid ORG JSON: {e}", path=getattr(e, "path", "")) from e
                repair_user = (
                    "Your previous ORG JSON failed schema validation.\n"
                    f"Error: {e}\n\n"
                    "Return ONE corrected ORG JSON object (intentir_org_v1) only. No prose, no code fences.\n"
                    "Keep the same schema_version/kernel/nodes/edges keys and fix types/fields.\n"
                    "Do NOT map to hardware or output tuned numeric assignments.\n"
                )
                try:
                    prev = json.dumps(raw_json, ensure_ascii=False, sort_keys=True)
                except Exception:
                    prev = ""
                cur_messages = list(messages)
                if prev:
                    cur_messages.append({"role": "assistant", "content": prev})
                cur_messages.append({"role": "user", "content": repair_user})
                continue
            except Exception as e:
                raise OrgValidationError(f"invalid ORG JSON: {type(e).__name__}: {e}") from e

        raise OrgValidationError("invalid ORG JSON: exceeded schema retries")


__all__ = ["CandidateOrg", "LLMOrgHub"]
