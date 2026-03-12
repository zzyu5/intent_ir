"""
LLMOrgHub: unified "KernelDescriptor (+ evidence bundle) -> OrgDoc" entrypoint.

The LLM is responsible for rationale-bearing sections only. Runtime injects:
- source_context
- source_oracle
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pipeline.interfaces import KernelDescriptor

from intent_ir.llm import DEFAULT_MODEL, LLMClientError, extract_json_object_with_trace
from org.schema import OrgDoc, OrgValidationError, validate_org_doc
from org.types import ORG_GOAL_TAGS, ORG_MECHANISM_CATEGORIES


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _blindfold_enabled() -> bool:
    raw = str(os.getenv("INTENTIR_ORG_BLINDFOLD", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _blindfold_label() -> str:
    raw = str(os.getenv("INTENTIR_ORG_BLINDFOLD_LABEL", "") or "").strip()
    return raw or "target_kernel_func"


_BLINDFOLD_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bliger[_a-z0-9]*\b",
        r"\bgroup_norm(?:_kernel|_fwd)?\b",
        r"\blayer_norm(?:_persistent|_fwd)?\b",
        r"\brms_norm(?:_fwd)?\b",
        r"\bsoftmax(?:_inner|_fwd)?\b",
        r"\bmasked_softmax2d\b",
        r"\bai_bench_softmax\b",
        r"\bgeglu\b",
        r"\bswiglu\b",
        r"\bcross_entropy\b",
        r"\brope\b",
        r"\bmrope\b",
        r"\bdyt\b",
    )
]


def _blindfold_text(text: str) -> str:
    out = str(text or "")
    if not _blindfold_enabled() or not out:
        return out
    replacement = _blindfold_label()
    for pattern in _BLINDFOLD_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def _canonical_reuse_window(value: Any, *, storage: Any = "") -> str:
    raw = str(value or "").strip()
    token = _norm_token(raw)
    storage_token = _norm_token(storage)
    if not token:
        return ""
    if token in {"row_processing", "row_normalization", "row_normalization_epilogue", "full_row_program"}:
        return "full_row"
    if storage_token == "register" and token in {"row_epilogue", "affine_epilogue"}:
        return "row_epilogue"
    return raw


def _canonical_goal_tag(value: Any) -> str:
    raw = str(value or "").strip()
    token = _norm_token(raw)
    allowed = set(ORG_GOAL_TAGS)
    if raw in allowed:
        return raw
    alias_map = {
        "streaming_state": "streaming_softmax_state",
        "statistics": "reduction_tree_balance",
        "vectorized_io": "memory_coalescing",
        "vector_io": "memory_coalescing",
        "coalesced_io": "memory_coalescing",
        "coalesced_memory": "memory_coalescing",
        "operand_reuse": "resident_working_set",
        "register_residency": "resident_working_set",
        "warp_reduction": "reduction_tree_balance",
        "row_reduction": "reduction_tree_balance",
        "reduction": "reduction_tree_balance",
        "reduction_balance": "reduction_tree_balance",
        "resident_state": "resident_working_set",
        "resident_row": "resident_working_set",
        "resident_tile": "resident_working_set",
        "row_resident": "resident_working_set",
        "operand_stage": "operand_reuse",
        "operand_tiles": "operand_reuse",
        "mma": "mma_acceleration",
        "tensor_core": "mma_acceleration",
        "no_materialize": "avoid_materialization",
        "avoid_writeback": "avoid_materialization",
        "persistent_cache": "persistent_row_state",
        "persistent_stats": "persistent_row_state",
        "affine_fusion": "affine_epilogue_fusion",
        "affine_epilogue": "affine_epilogue_fusion",
        "fused_epilogue": "fused_epilogue_avoid_writeback",
        "layout_rotation": "memory_coalescing",
        "rope_rotation": "memory_coalescing",
        "mrope_rotation": "memory_coalescing",
        "rope_fusion": "memory_coalescing",
        "branch_fwd_bwd": "latency_hiding",
        "mask_pruning": "mask_causal_pruning",
        "causal_mask": "mask_causal_pruning",
    }
    mapped = alias_map.get(token, raw)
    return mapped if mapped in allowed else raw


def _canonical_mechanism_category(value: Any) -> str:
    raw = str(value or "").strip()
    token = _norm_token(raw)
    if raw in ORG_MECHANISM_CATEGORIES:
        return raw
    prefix = token.split(".", 1)[0]
    alias_map = {
        "tile": "tiling",
        "tiling": "tiling",
        "stage": "staging",
        "staging": "staging",
        "layout": "mapping",
        "dataflow": "mapping",
        "indexing": "mapping",
        "pipeline": "pipeline",
        "parallel": "mapping",
        "parallelism": "mapping",
        "schedule": "pipeline",
        "control": "pipeline",
        "compute": "primitive",
        "computation": "primitive",
        "math": "primitive",
        "reduction": "primitive",
        "statistics": "primitive",
        "map": "mapping",
        "mapping": "mapping",
        "sync": "communication",
        "synchronization": "communication",
        "comm": "communication",
        "communication": "communication",
        "memory": "staging",
        "residency": "staging",
        "reuse": "staging",
        "primitive": "primitive",
        "fusion": "fusion",
    }
    mapped = alias_map.get(prefix, prefix)
    return mapped if mapped in ORG_MECHANISM_CATEGORIES else raw


def _canonical_mechanism_tag(value: Any, *, category: Any = "", goal_tags: set[str] | None = None) -> str:
    raw = str(value or "").strip()
    token = _norm_token(raw)
    category_token = _norm_token(category)
    goals = set(goal_tags or set())
    token_tail = token.split(".")[-1]
    token_compound = token.replace(".", "_")
    normalized_tokens = {token, token_tail, token_compound}
    if token in {
        "row_tile_resident",
        "group_tile_resident",
        "tile_resident",
        "vector_row_path",
        "vector_global_io",
        "tile_load_stage",
        "row_reduction",
        "warp_reduction",
        "warp_reduction_tree",
        "online_safe_math_reduction",
        "online_softmax_reduce",
        "parallel_softmax",
        "multi_output_stats_resident",
        "affine_epilogue",
        "blocked_register_layout",
        "block_synchronization",
        "activation_then_mul_fusion",
    }:
        return token
    if normalized_tokens & {"blocked_tiling", "row_tiling", "row_resident", "resident_tile", "tile_resident"}:
        return "row_tile_resident"
    if token.startswith("warp_reduction_") or token_tail.startswith("warp_reduction_"):
        return "warp_reduction"
    if token.startswith("row_reduction_") or token_tail.startswith("row_reduction_"):
        return "row_reduction"
    if normalized_tokens & {"vectorized_load", "vectorized_store", "vector_load", "vector_store", "coalesced_io", "vector_io", "vec4_load", "vec2_load"}:
        return "vector_row_path" if "reduction_tree_balance" in goals or "streaming_softmax_state" in goals else "vector_global_io"
    if normalized_tokens & {"tile_load", "tile_read"} or token.startswith("tile_load_") or token_tail.startswith("tile_load_"):
        return "tile_load_stage"
    if normalized_tokens & {"warp_shuffle", "shuffle_reduce", "shuffle_tree"}:
        return "warp_reduction_tree"
    if normalized_tokens & {"block_sync", "barrier_sync", "block_barrier"}:
        return "block_synchronization"
    if normalized_tokens & {"max_reduction", "sum_reduction", "online_max_sum", "softmax_reduction"}:
        return "online_safe_math_reduction" if "streaming_softmax_state" in goals else "row_reduction"
    if normalized_tokens & {"mean_reduction", "var_reduction", "variance_reduction", "stats_reduction"}:
        return "warp_reduction"
    if (
        normalized_tokens & {"stats_resident", "mean_rstd_resident", "multi_stats", "multi_output_stats"}
        or token.startswith("stats_resident")
        or token.startswith("multi_output_stats")
        or token_tail.startswith("stats_resident")
        or token_tail.startswith("multi_output_stats")
    ):
        return "multi_output_stats_resident"
    if (
        normalized_tokens & {"affine_fusion", "affine_output", "affine_writeback", "affine_scale_shift", "conditional_beta", "optional_bias", "bias_add"}
        or token.startswith("affine_epilogue")
        or token.startswith("affine_out")
        or token_tail.startswith("affine_epilogue")
        or token_tail.startswith("affine_out")
    ):
        return "affine_epilogue"
    if normalized_tokens & {"blocked_layout", "register_blocked_layout"}:
        return "blocked_register_layout"
    if normalized_tokens & {"activation_fusion", "gelu_mul", "geglu_fusion", "swiglu_fusion", "activation_then_mul"}:
        return "activation_then_mul_fusion"
    if category_token == "communication.reduction":
        return "online_safe_math_reduction" if "streaming_softmax_state" in goals else "row_reduction"
    return raw


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


_TRITON_SLICE_PATTERNS = [
    re.compile(p)
    for p in (
        r"^\s*@triton\.jit",
        r"^\s*def\s+",
        r"^\s*for\s+",
        r"^\s*if\s+",
        r"\btl\.constexpr\b",
        r"\btl\.program_id\b",
        r"\btl\.arange\b",
        r"\btl\.load\b",
        r"\btl\.store\b",
        r"\btl\.sum\b",
        r"\btl\.max\b",
        r"\btl\.exp\b",
        r"\btl\.log\b",
        r"\btl\.where\b",
        r"\btl\.math\b",
        r"\btl\.sqrt\b",
        r"\btl\.rsqrt\b",
        r"\btl\.dot\b",
        r"\btl\.debug_barrier\b",
        r"\btl\.make_block_ptr\b",
        r"\btl\.advance\b",
    )
]

_TTGIR_SLICE_PATTERNS = [
    re.compile(p)
    for p in (
        r"#ttg\.blocked<",
        r"\btt\.get_program_id\b",
        r"\bscf\.for\b",
        r"\btt\.load\b",
        r"\btt\.store\b",
        r"\btt\.reduce\b",
        r"\bttg\.convert_layout\b",
        r"\btt\.dot\b",
        r"\bttg\.async_",
        r"\bshared\b",
        r"\blocal_alloc\b",
        r"\barith\.select\b",
        r"\barith\.cmp",
    )
]

_TTGIR_NOISE_PATTERNS = [
    re.compile(p)
    for p in (
        r"\btt\.(?:splat|broadcast)\b",
        r"\barith\.constant\b",
        r"\barith\.(?:addi|subi|muli|divsi|divui|floordivsi|remsi|remui)\b",
        r"\barith\.(?:extsi|extui|trunci|index_cast|sitofp|fptosi|uitofp|fptoui)\b",
        r"\barith\.(?:and|or|xor)i\b",
        r"\bbuiltin\.unrealized_conversion_cast\b",
        r"\btensor\.(?:extract|insert)\b",
    )
]

_PTX_SLICE_PATTERNS = [
    re.compile(p)
    for p in (
        r"^\s*\.entry\b",
        r"^\s*\.reqntid\b",
        r"^\s*\.(?:extern\s+)?shared\b",
        r"\bcp\.async\b",
        r"\bcp\.async\.commit_group\b",
        r"\bcp\.async\.wait_group\b",
        r"\bmma\.sync\b",
        r"\bwgmma\.",
        r"\bldmatrix\b",
        r"\bshfl\.sync\b",
        r"\bbar\.sync\b",
        r"\bsetp\.",
        r"\b@%p[0-9]+\s+bra\b",
        r"\bbra\b",
        r"\bld\.global\b",
        r"\bst\.global\b",
    )
]

_PTX_NOISE_PATTERNS = [
    re.compile(p)
    for p in (
        r"^\s*mov\.",
        r"^\s*cvt\.",
        r"^\s*add\.(?:s|u|f)",
        r"^\s*mul\.(?:lo|wide|hi|f)",
        r"^\s*mad\.(?:lo|wide|hi)",
        r"^\s*shl\.",
        r"^\s*shr\.",
        r"^\s*and\.",
        r"^\s*or\.",
    )
]


def _read_text_path(raw: Any) -> str:
    path = Path(str(raw or "").strip())
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _slice_lines(text: str, *, patterns: list[re.Pattern[str]], context: int = 1, limit: int = 64) -> list[str]:
    lines = str(text or "").splitlines()
    if not lines:
        return []
    keep: set[int] = set()
    for idx, line in enumerate(lines):
        if any(p.search(line) for p in patterns):
            lo = max(0, int(idx) - int(context))
            hi = min(len(lines), int(idx) + int(context) + 1)
            keep.update(range(lo, hi))
    ordered = [f"{i + 1}: {lines[i]}" for i in sorted(keep)]
    return ordered[: max(0, int(limit))]


def _fold_noisy_slice_lines(lines: list[str], *, kind: str) -> list[str]:
    kind_token = _norm_token(kind)
    if not lines:
        return []
    patterns: list[re.Pattern[str]] = []
    if kind_token == "ttgir":
        patterns = list(_TTGIR_NOISE_PATTERNS)
    elif kind_token == "ptx":
        patterns = list(_PTX_NOISE_PATTERNS)
    if not patterns:
        return list(lines)

    def _payload(line: str) -> str:
        _, sep, rhs = str(line).partition(":")
        return rhs if sep else str(line)

    out: list[str] = []
    folded = 0

    def _flush() -> None:
        nonlocal folded
        if folded > 0:
            out.append(f"... [Folded {int(folded)} data-flow ops] ...")
            folded = 0

    for line in list(lines):
        payload = _payload(str(line))
        if any(p.search(payload) for p in patterns):
            folded += 1
            continue
        _flush()
        out.append(str(line))
    _flush()
    return out


def _distill_source_text(source_text: str) -> str:
    text = str(source_text or "")
    sliced = _slice_lines(text, patterns=_TRITON_SLICE_PATTERNS, context=1, limit=140)
    if not sliced:
        return _maybe_truncate_source(text)
    header = f"[IntentIR][ORG] SOURCE DISTILLED: kept_lines={len(sliced)}"
    return "\n".join([header, *sliced])


def _preferred_source_text(descriptor: KernelDescriptor) -> str:
    meta = dict(getattr(descriptor, "meta", {}) or {})
    remote_source = meta.get("remote_source_oracle")
    if isinstance(remote_source, Mapping):
        run_info = dict(remote_source.get("run_info") or {})
        remote_src = str(run_info.get("source_attr") or "").strip()
        if remote_src:
            return _blindfold_text(remote_src)
    return _blindfold_text(str(getattr(descriptor, "source_text", "") or ""))


def _slice_artifact_text(path_like: Any, *, kind: str) -> list[str]:
    text = _blindfold_text(_read_text_path(path_like))
    if not text:
        return []
    if kind == "ttgir":
        sliced = _slice_lines(text, patterns=_TTGIR_SLICE_PATTERNS, context=1, limit=96)
        return _fold_noisy_slice_lines(sliced, kind="ttgir")[:48]
    if kind == "ptx":
        sliced = _slice_lines(text, patterns=_PTX_SLICE_PATTERNS, context=0, limit=72)
        return _fold_noisy_slice_lines(sliced, kind="ptx")[:48]
    return []


def _compact_io_spec(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    io = dict(raw or {})
    tensors_in = dict(io.get("tensors") or {})
    tensors_out: dict[str, Any] = {}
    for key, value in tensors_in.items():
        if not isinstance(value, Mapping):
            continue
        item = dict(value or {})
        tensors_out[str(key)] = {
            "dtype": str(item.get("dtype") or ""),
            "shape": list(item.get("shape") or []),
            "role": str(item.get("role") or ""),
        }
    out: dict[str, Any] = {
        "arg_names": [str(x) for x in list(io.get("arg_names") or []) if str(x).strip()],
        "outputs": [str(x) for x in list(io.get("outputs") or []) if str(x).strip()],
    }
    if tensors_out:
        out["tensors"] = tensors_out
    scalars_in = dict(io.get("scalars") or {})
    if scalars_in:
        out["scalars"] = {str(k): str(v) for k, v in scalars_in.items() if str(k).strip()}
    return out


def _compact_fact_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    out: dict[str, Any] = {}
    mechanisms = {}
    for key, value in dict(payload.get("mechanisms") or {}).items():
        entry = dict(value or {})
        if not bool(entry.get("present")):
            continue
        mechanisms[str(key)] = {
            "present": True,
            "attrs": dict(entry.get("attrs") or {}),
            "evidence_refs": [str(x) for x in list(entry.get("evidence_refs") or []) if str(x).strip()],
        }
    if mechanisms:
        out["mechanisms"] = mechanisms
    evidence = []
    live_refs = {
        str(ref)
        for entry in mechanisms.values()
        for ref in list(entry.get("evidence_refs") or [])
    }
    for item in list(payload.get("evidence") or []):
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id or item_id not in live_refs:
            continue
        evidence.append(
            {
                "id": item_id,
                "kind": str(item.get("kind") or ""),
                "summary": str(item.get("summary") or ""),
            }
        )
    if evidence:
        out["evidence"] = evidence[:64]
    artifacts = dict(payload.get("artifacts") or {})
    if artifacts:
        out["artifacts"] = {
            "ttgir_available": bool(artifacts.get("ttgir_available")),
            "ptx_available": bool(artifacts.get("ptx_available")),
        }
    return out


def _ordered_evidence_blob(
    descriptor: KernelDescriptor,
    *,
    intent_summary: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None,
) -> str:
    extra_dict = dict(extra or {}) if isinstance(extra, Mapping) else {}
    ordered: dict[str, Any] = {
        "kernel": (_blindfold_label() if _blindfold_enabled() else descriptor.name),
        "frontend": descriptor.frontend,
    }
    for key in ("ttgir_facts", "ptx_facts", "source_oracle_facts", "ttir_summary"):
        value = extra_dict.get(key)
        if isinstance(value, Mapping):
            if key in {"ttgir_facts", "ptx_facts"}:
                ordered[key] = _compact_fact_payload(value)
            else:
                ordered[key] = dict(value)
    if isinstance(intent_summary, Mapping):
        ordered["intent_summary"] = dict(intent_summary)
    ordered["io_spec"] = _compact_io_spec(getattr(descriptor, "io_spec", {}) or {})
    runtime_extra = {
        str(k): v
        for k, v in extra_dict.items()
        if str(k) not in {"ttgir_facts", "ptx_facts", "source_oracle_facts", "ttir_summary"} and str(k).strip()
    }
    remote_source = runtime_extra.get("remote_source")
    if not isinstance(remote_source, Mapping):
        remote_source = dict(getattr(descriptor, "meta", {}) or {}).get("remote_source_oracle")
    remote_artifacts = dict(remote_source.get("artifacts") or {}) if isinstance(remote_source, Mapping) else {}
    meta = dict(getattr(descriptor, "meta", {}) or {})
    local_artifacts = {
        "ttgir": str(meta.get("ttgir_original_path") or getattr(getattr(descriptor, "artifacts", None), "ttgir_path", "") or "").strip(),
        "ptx": str(
            meta.get("ptx_original_path")
            or (getattr(getattr(descriptor, "artifacts", None), "extra", {}) or {}).get("ptx_path")
            or ""
        ).strip(),
    }
    ttgir_path = remote_artifacts.get("ttgir") or local_artifacts.get("ttgir")
    ptx_path = remote_artifacts.get("ptx") or local_artifacts.get("ptx")
    if not _read_text_path(ttgir_path):
        ttgir_path = local_artifacts.get("ttgir")
    if not _read_text_path(ptx_path):
        ptx_path = local_artifacts.get("ptx")
    ttgir_slice = _slice_artifact_text(ttgir_path, kind="ttgir")
    ptx_slice = _slice_artifact_text(ptx_path, kind="ptx")
    if ttgir_slice or ptx_slice:
        ordered["artifact_slices"] = {}
        if ttgir_slice:
            ordered["artifact_slices"]["ttgir_core"] = ttgir_slice
        if ptx_slice:
            ordered["artifact_slices"]["ptx_core"] = ptx_slice
    if isinstance(remote_source, Mapping):
        ordered["remote_source"] = {
            "source_arch": str(remote_source.get("source_arch") or ""),
            "gpu_name": str(remote_source.get("gpu_name") or ""),
            "compiler_stack": str(remote_source.get("compiler_stack") or ""),
        }
    if runtime_extra:
        ordered["extra"] = runtime_extra
    return _blindfold_text(json.dumps(ordered, ensure_ascii=False))


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
            "kernel_kind": ("" if _blindfold_enabled() else str(oracle.get("kernel_kind") or "")),
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
    if _blindfold_enabled():
        obj["kernel"] = _blindfold_label()
    schema_version = str(obj.get("schema_version") or "").strip().lower()
    if schema_version in {"org_v1", "intent_ir_org_v1", "intentir_org", "intentir_org_v1"}:
        obj["schema_version"] = "intentir_org_v1"
    for key in (
        "goals",
        "mechanisms",
        "dims",
        "evidence",
        "tensors",
        "tensor_lifetimes",
        "dataflow_edges",
        "mechanism_topology",
        "schedule_edges",
    ):
        if obj.get(key) is None:
            obj[key] = []
    dims = [dict(x) for x in list(obj.get("dims") or []) if isinstance(x, Mapping)]
    dim_names = {str(item.get("name") or "").strip() for item in dims if str(item.get("name") or "").strip()}
    goal_ids = {str(item.get("id") or "").strip() for item in list(obj.get("goals") or []) if isinstance(item, Mapping)}
    evidence_out: list[dict[str, Any]] = []
    for raw_evidence in list(obj.get("evidence") or []):
        if not isinstance(raw_evidence, Mapping):
            continue
        item = dict(raw_evidence)
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        if not str(item.get("kind") or "").strip():
            item["kind"] = "evidence"
        if not str(item.get("path") or "").strip():
            item["path"] = f"evidence:{item_id}"
        if not str(item.get("summary") or "").strip() and str(item.get("text") or "").strip():
            item["summary"] = str(item.get("text") or "").strip()[:160]
        evidence_out.append(item)
    if evidence_out or "evidence" in obj:
        obj["evidence"] = evidence_out
    evidence_ids = {
        str(item.get("id") or "").strip() for item in list(obj.get("evidence") or []) if isinstance(item, Mapping)
    }
    goals_out: list[dict[str, Any]] = []
    for raw_goal in list(obj.get("goals") or []):
        if not isinstance(raw_goal, Mapping):
            continue
        goal = dict(raw_goal)
        if not str(goal.get("tag") or "").strip() and str(goal.get("kind") or "").strip():
            goal["tag"] = str(goal.get("kind") or "")
        if not str(goal.get("summary") or "").strip() and str(goal.get("description") or "").strip():
            goal["summary"] = str(goal.get("description") or "")
        goal["tag"] = _canonical_goal_tag(goal.get("tag"))
        if not str(goal.get("scope") or "").strip():
            goal["scope"] = "kernel"
        goal["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(goal.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
        goals_out.append(goal)
    if goals_out or "goals" in obj:
        obj["goals"] = goals_out
    goal_tags_norm = {
        str(item.get("tag") or "").strip()
        for item in list(obj.get("goals") or [])
        if isinstance(item, Mapping) and str(item.get("tag") or "").strip()
    }
    dims_out: list[dict[str, Any]] = []
    for raw_dim in list(obj.get("dims") or []):
        if not isinstance(raw_dim, Mapping):
            continue
        dim = dict(raw_dim)
        constraints_out: list[str] = []
        for raw_constraint in list(dim.get("constraints") or []):
            text = str(raw_constraint or "").strip()
            if text:
                constraints_out.append(text)
        if constraints_out or "constraints" in dim:
            dim["constraints"] = constraints_out
        dim["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(dim.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
        dims_out.append(dim)
    if dims_out or "dims" in obj:
        obj["dims"] = dims_out
    mechanisms_out: list[dict[str, Any]] = []
    for raw_mech in list(obj.get("mechanisms") or []):
        if not isinstance(raw_mech, Mapping):
            continue
        mech = dict(raw_mech)
        if not str(mech.get("tag") or "").strip() and str(mech.get("kind") or "").strip():
            mech["tag"] = str(mech.get("kind") or "")
        if not str(mech.get("category") or "").strip() and str(mech.get("kind") or "").strip():
            mech["category"] = str(mech.get("kind") or "")
        mech["category"] = _canonical_mechanism_category(mech.get("category"))
        mech["tag"] = _canonical_mechanism_tag(mech.get("tag"), category=mech.get("category"), goal_tags=goal_tags_norm)
        dims_list = []
        for raw_dim in list(mech.get("dims") or []):
            name = str(raw_dim or "").strip()
            if not name or name not in dim_names:
                continue
            dims_list.append(name)
        mech["dims"] = dims_list
        mech["supports_goals"] = [
            ref for ref in [str(x).strip() for x in list(mech.get("supports_goals") or []) if str(x).strip()] if ref in goal_ids
        ]
        mech["evidence_refs"] = [
            ref for ref in [str(x).strip() for x in list(mech.get("evidence_refs") or []) if str(x).strip()] if ref in evidence_ids
        ]
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
        for field_name in ("region", "storage", "start", "end", "scope", "layout"):
            value = lifetime.get(field_name)
            if value is None:
                continue
            if not isinstance(value, str):
                lifetime[field_name] = str(value)
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
        lifetime["reuse_window"] = _canonical_reuse_window(
            lifetime.get("reuse_window"),
            storage=lifetime.get("storage"),
        )
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
    elif not region_graph:
        branch_mechanisms = [
            dict(item)
            for item in list(obj.get("mechanisms") or [])
            if isinstance(item, Mapping) and _norm_token(item.get("tag")) in {"branch_mask", "ignore_mask"}
        ]
        if branch_mechanisms:
            mechanism_ids = {
                str(item.get("id") or "").strip(): dict(item)
                for item in list(obj.get("mechanisms") or [])
                if isinstance(item, Mapping) and str(item.get("id") or "").strip()
            }
            reduction_mechs = {
                mech_id
                for mech_id, mech in mechanism_ids.items()
                if _norm_token(mech.get("tag")) in {"row_reduction", "warp_reduction", "warp_reduction_tree", "online_softmax_reduce", "online_normalization"}
            }
            gather_mechs = {
                mech_id
                for mech_id, mech in mechanism_ids.items()
                if _norm_token(mech.get("tag")) in {"label_gather", "index_gather"}
            }
            finalize_mechs = {
                mech_id
                for mech_id, mech in mechanism_ids.items()
                if _norm_token(mech.get("tag")) in {"loss_finalize", "gradient_fused_epilogue", "argmax_tracking"}
            }
            lifetimes = [dict(item) for item in list(obj.get("tensor_lifetimes") or []) if isinstance(item, Mapping)]
            active_lifetimes = [
                str(item.get("id") or "").strip()
                for item in lifetimes
                if str(item.get("id") or "").strip()
                and (
                    any(str(x).strip() in reduction_mechs for x in list(item.get("consumer_mechanisms") or []))
                    or any(str(x).strip() in gather_mechs for x in list(item.get("consumer_mechanisms") or []))
                    or any(str(x).strip() in finalize_mechs for x in list(item.get("consumer_mechanisms") or []))
                )
            ]
            skip_lifetimes = [
                str(item.get("id") or "").strip()
                for item in lifetimes
                if str(item.get("id") or "").strip()
                and any(str(x).strip() in {str(m.get("id") or "").strip() for m in branch_mechanisms} for x in list(item.get("consumer_mechanisms") or []))
            ]
            branch_mech_ids = [str(item.get("id") or "").strip() for item in branch_mechanisms if str(item.get("id") or "").strip()]
            predicate = str(((branch_mechanisms[0].get("attrs") or {}).get("predicate")) or "").strip() or "branch_mask"
            obj["region_graph"] = {
                "regions": [
                    {
                        "id": "cfg_entry",
                        "kind": "region",
                        "path_id": "pi_entry",
                        "entry_mechanisms": branch_mech_ids,
                        "exit_mechanisms": branch_mech_ids,
                    },
                    {
                        "id": "cfg_active",
                        "kind": "if_true",
                        "parent": "cfg_entry",
                        "path_id": "pi_active",
                        "predicate": f"not ({predicate})",
                        "entry_mechanisms": branch_mech_ids,
                        "exit_mechanisms": sorted(reduction_mechs | gather_mechs | finalize_mechs),
                    },
                    {
                        "id": "cfg_masked",
                        "kind": "if_false",
                        "parent": "cfg_entry",
                        "path_id": "pi_masked",
                        "predicate": str(predicate),
                        "entry_mechanisms": branch_mech_ids,
                        "exit_mechanisms": branch_mech_ids,
                    },
                ],
                "edges": [
                    {
                        "id": "cfg_edge_active",
                        "src": "cfg_entry",
                        "dst": "cfg_active",
                        "relation": "branch",
                        "path_id": "pi_active",
                        "predicate": f"not ({predicate})",
                        "lifetimes": [x for x in active_lifetimes if x],
                        "mechanisms": sorted(reduction_mechs | gather_mechs | finalize_mechs),
                    },
                    {
                        "id": "cfg_edge_masked",
                        "src": "cfg_entry",
                        "dst": "cfg_masked",
                        "relation": "branch",
                        "path_id": "pi_masked",
                        "predicate": str(predicate),
                        "lifetimes": [x for x in skip_lifetimes if x],
                        "mechanisms": branch_mech_ids,
                    },
                ],
            }
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
                "The appendix already contains TTGIR/PTX core slices pruned to scheduling-relevant lines. Prefer those slices over raw source when recovering residency, synchronization, vectorization, online reductions, and branch topology.",
            ]
        ).strip()

        src = _distill_source_text(_preferred_source_text(descriptor))
        compact = bool(src.startswith("[IntentIR][ORG] SOURCE DISTILLED")) or bool(src.startswith("[IntentIR][ORG] SOURCE TRUNCATED")) or len(evidence) > 12000
        if descriptor.frontend == "triton":
            from org.frontends.triton.llm_org import build_messages  # noqa: PLC0415

            prompt_kernel_name = _blindfold_label() if _blindfold_enabled() else descriptor.name
            if _blindfold_enabled():
                extra_instruction = (
                    str(extra_instruction).strip()
                    + "\n\nBlindfold rule: all operator names in source/evidence were anonymized to `"
                    + _blindfold_label()
                    + "`. Infer topology only from dataflow, residency, reduction, layout, and control evidence."
                ).strip()
            messages = build_messages(src, kernel_name=prompt_kernel_name, extra_instruction=extra_instruction, compact=compact)
        else:
            raise NotImplementedError(f"LLMOrgHub does not support frontend={descriptor.frontend}")

        chat_kwargs = dict(self.extra_chat_kwargs)
        chat_kwargs.setdefault("max_tokens", 1800)
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
