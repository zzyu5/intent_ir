"""
LLMIntentHub: unified "KernelDescriptor -> CandidateIntent" entrypoint.

This is the place where we:
- inject structured frontend evidence (facts/constraints) into the prompt
- record an execution trace (model/provider/cache/prompt hash)

The hub does NOT hardcode any particular frontend IR; it consumes the generic
KernelDescriptor and selects frontend-specific prompt builders when needed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from pipeline.interfaces import KernelDescriptor

from intent_ir.ir import IntentIRValidationError
from intent_ir.llm import DEFAULT_MODEL, LLMClientError, candidate_models, chat_completion, parse_json_block
from intent_ir.parser import CandidateIntent, LLMJsonParseError, parse_candidate_json
from intent_ir.ir.repair import repair_missing_outputs


def _hash_messages(messages: List[Dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _maybe_truncate_source(source_text: str) -> str:
    """
    Provider-facing safeguard: truncate very long kernel sources.

    Some proxy providers become unstable (5xx) on large prompts for complex kernels
    (e.g., bicubic upsample with many repeated loads). For such cases, the evidence
    appendix + kernel name is usually sufficient for the LLM to emit a macro op.
    """
    text = str(source_text)
    lines = text.splitlines()
    # Conservative-but-not-overzealous defaults:
    # - do NOT truncate normal kernels (~100–600 LOC), since this breaks cache
    #   locality and can reduce LLM quality for non-macro kernels.
    # - only truncate very large sources that are likely to trigger proxy 5xx.
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
    banner = f"[IntentIR] SOURCE TRUNCATED: original_lines={len(lines)} kept_head={len(head_lines)} kept_tail={len(tail_lines)}"
    return "\n".join([banner, *head_lines, "[IntentIR] ... TRUNCATED ...", *tail_lines])


def _maybe_compact_source_on_server_error(source_text: str, last_error: Exception | None) -> str:
    """
    Second-stage compaction: when a provider returns repeated 5xx, retry with a
    smaller source payload.

    Some proxy endpoints have very small input limits and may respond with 500
    on slightly larger prompts. In that case, we keep just a prefix + suffix of
    the CUDA/Triton source and rely on the evidence appendix for details.
    """
    if last_error is None:
        return str(source_text)
    msg = str(last_error)
    if "server error" not in msg and " 520 " not in msg and " 502 " not in msg and " 503 " not in msg and " 504 " not in msg:
        return str(source_text)
    text = str(source_text)
    if len(text) <= 1800:
        return text
    head = 1200
    tail = 240
    return "\n".join(
        [
            "[IntentIR] SOURCE COMPACT (server-error retry)",
            text[:head],
            "[IntentIR] ... COMPACTED ...",
            text[-tail:] if tail > 0 else "",
        ]
    ).strip()


def _evidence_blob(descriptor: KernelDescriptor) -> str:
    def _summarize_frontend_constraints(fc: Any) -> Any:
        """
        Keep the evidence appendix small and stable.

        Some frontends attach large, detailed witnesses (e.g. access lists) that
        are crucial for debugging/tuning but can blow up prompt size and trigger
        proxy/provider 5xx. For LLM extraction, we only need a compact subset:
          - shape symbols / ranges
          - tile hints / scheduling sketch inputs
          - mask/predicate clauses (if present)
          - a *summary* of access witnesses (counts + a few scalars)
        """
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

        # Predicate clauses can get large; cap to keep prompts bounded.
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

        # Access witness: keep only a compact summary (drop full access list).
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

    ev = {
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
    # Compact encoding keeps prompts within provider limits and also makes cache
    # keys less sensitive to whitespace.
    return json.dumps(ev, ensure_ascii=False, sort_keys=True)


def _baseline_npz_path(descriptor: KernelDescriptor) -> Path | None:
    artifact_dir = str(descriptor.meta.get("artifact_dir") or "").strip()
    if not artifact_dir:
        return None
    path = Path(artifact_dir) / f"{descriptor.name}.baseline.npz"
    return path if path.is_file() else None


def _baseline_array_shapes(descriptor: KernelDescriptor) -> dict[str, tuple[int, ...]]:
    path = _baseline_npz_path(descriptor)
    if path is None:
        return {}
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {str(k): tuple(int(x) for x in np.asarray(payload[k]).shape) for k in payload.files}
    except Exception:
        return {}


def _shape_entry(*dims: str | int) -> list[str | int]:
    return [int(x) if isinstance(x, int) else str(x) for x in dims]


def _descriptor_arg_names(descriptor: KernelDescriptor) -> set[str]:
    io_spec = getattr(descriptor, "io_spec", None)
    if not isinstance(io_spec, Mapping):
        return set()
    return {str(x).strip() for x in list(io_spec.get("arg_names") or []) if str(x).strip()}


def _rope_repair_json(descriptor: KernelDescriptor, *, q_shape: tuple[int, ...], k_shape: tuple[int, ...], cos_shape: tuple[int, ...]) -> dict[str, Any]:
    canonical_shapes = {
        str(k): int(v)
        for k, v in dict((descriptor.launch or {}).get("canonical_shapes") or {}).items()
        if str(k).strip()
    }
    b_dim = qh_dim = kh_dim = s_dim = hd_dim = 0
    layout = ""
    if {"B", "QH", "KH", "S", "HD"} <= set(canonical_shapes):
        cand_b = int(canonical_shapes["B"])
        cand_qh = int(canonical_shapes["QH"])
        cand_kh = int(canonical_shapes["KH"])
        cand_s = int(canonical_shapes["S"])
        cand_hd = int(canonical_shapes["HD"])
        if tuple(map(int, q_shape)) == (cand_b, cand_qh, cand_s, cand_hd) and tuple(map(int, k_shape)) == (
            cand_b,
            cand_kh,
            cand_s,
            cand_hd,
        ):
            b_dim, qh_dim, kh_dim, s_dim, hd_dim = cand_b, cand_qh, cand_kh, cand_s, cand_hd
            layout = "bhsd"
        elif tuple(map(int, q_shape)) == (cand_b, cand_s, cand_qh, cand_hd) and tuple(map(int, k_shape)) == (
            cand_b,
            cand_s,
            cand_kh,
            cand_hd,
        ):
            b_dim, qh_dim, kh_dim, s_dim, hd_dim = cand_b, cand_qh, cand_kh, cand_s, cand_hd
            layout = "bshd"
    if not layout:
        b0, d1, d2, d3 = map(int, q_shape)
        _bk, k1, k2, _khd = map(int, k_shape)
        if d1 == k1 and d2 != k2:
            b_dim, s_dim, qh_dim, kh_dim, hd_dim = b0, d1, d2, k2, d3
            layout = "bshd"
        else:
            b_dim, qh_dim, s_dim, kh_dim, hd_dim = b0, d1, d2, k1, d3
            layout = "bhsd"
    cos_batch = int(cos_shape[0]) if len(cos_shape) == 3 else 1
    cos_width = int(cos_shape[-1]) if cos_shape else hd_dim
    cos_b_dim: str | int = int(cos_batch) if cos_batch != b_dim else "B"
    logical_layout = {"kind": "custom", "params": {"axes": ["B", "H", "S", "HD"]}}
    physical_layout = {"kind": "custom", "params": {"axes": ["B", "S", "H", "HD"], "view_perm": [0, 2, 1, 3]}}
    if layout == "bshd":
        public_layout = {"kind": "custom", "params": {"axes": ["B", "S", "H", "HD"]}}
        return {
            "name": descriptor.name,
            "kernel_type": descriptor.name,
            "tensors": {
                "q": {"dtype": "f32", "shape": _shape_entry("B", "S", "QH", "HD"), "layout": public_layout},
                "k": {"dtype": "f32", "shape": _shape_entry("B", "S", "KH", "HD"), "layout": public_layout},
                "cos": {"dtype": "f32", "shape": _shape_entry(cos_b_dim, "S", cos_width), "layout": "row_major"},
                "sin": {"dtype": "f32", "shape": _shape_entry(cos_b_dim, "S", cos_width), "layout": "row_major"},
                "q_out": {"dtype": "f32", "shape": _shape_entry("B", "S", "QH", "HD"), "layout": public_layout},
                "k_out": {"dtype": "f32", "shape": _shape_entry("B", "S", "KH", "HD"), "layout": public_layout},
            },
            "ops": [
                {"op": "rope", "inputs": ["q", "cos", "sin"], "output": "q_out", "attrs": {"input_layout": "bshd"}},
                {"op": "rope", "inputs": ["k", "cos", "sin"], "output": "k_out", "attrs": {"input_layout": "bshd"}},
            ],
            "outputs": ["q_out", "k_out"],
            "parallel_axes": ["B", "S", "QH", "KH"],
            "axis_roles": {"B": "batch", "S": "spatial", "QH": "channel", "KH": "channel", "HD": "channel"},
            "meta": {
                "repaired_by": "liger_rope_view_repair_v2",
                "view_model": "direct_bshd_public",
                "shape_bindings": {"B": b_dim, "QH": qh_dim, "KH": kh_dim, "S": s_dim, "HD": hd_dim},
            },
        }
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "q": {"dtype": "f32", "shape": _shape_entry("B", "QH", "S", "HD"), "layout": logical_layout},
            "k": {"dtype": "f32", "shape": _shape_entry("B", "KH", "S", "HD"), "layout": logical_layout},
            "cos": {"dtype": "f32", "shape": _shape_entry(cos_b_dim, "S", cos_width), "layout": "row_major"},
            "sin": {"dtype": "f32", "shape": _shape_entry(cos_b_dim, "S", cos_width), "layout": "row_major"},
            "q_phys": {
                "dtype": "f32",
                "shape": _shape_entry("B", "S", "QH", "HD"),
                "layout": physical_layout,
                "view_of": "q",
                "alias_group": "q_storage_view",
                "meta": {"transpose_perm": [0, 2, 1, 3]},
            },
            "k_phys": {
                "dtype": "f32",
                "shape": _shape_entry("B", "S", "KH", "HD"),
                "layout": physical_layout,
                "view_of": "k",
                "alias_group": "k_storage_view",
                "meta": {"transpose_perm": [0, 2, 1, 3]},
            },
            "q_rot_phys": {"dtype": "f32", "shape": _shape_entry("B", "S", "QH", "HD"), "layout": physical_layout},
            "k_rot_phys": {"dtype": "f32", "shape": _shape_entry("B", "S", "KH", "HD"), "layout": physical_layout},
            "q_out": {"dtype": "f32", "shape": _shape_entry("B", "QH", "S", "HD"), "layout": logical_layout},
            "k_out": {"dtype": "f32", "shape": _shape_entry("B", "KH", "S", "HD"), "layout": logical_layout},
        },
        "ops": [
            {"op": "transpose", "inputs": ["q"], "output": "q_phys", "attrs": {"perm": [0, 2, 1, 3]}},
            {"op": "transpose", "inputs": ["k"], "output": "k_phys", "attrs": {"perm": [0, 2, 1, 3]}},
            {"op": "rope", "inputs": ["q_phys", "cos", "sin"], "output": "q_rot_phys", "attrs": {"input_layout": "bshd"}},
            {"op": "rope", "inputs": ["k_phys", "cos", "sin"], "output": "k_rot_phys", "attrs": {"input_layout": "bshd"}},
            {"op": "transpose", "inputs": ["q_rot_phys"], "output": "q_out", "attrs": {"perm": [0, 2, 1, 3]}},
            {"op": "transpose", "inputs": ["k_rot_phys"], "output": "k_out", "attrs": {"perm": [0, 2, 1, 3]}},
        ],
        "outputs": ["q_out", "k_out"],
        "parallel_axes": ["B", "S", "QH", "KH"],
        "axis_roles": {"B": "batch", "S": "spatial", "QH": "channel", "KH": "channel", "HD": "channel"},
        "meta": {
            "repaired_by": "liger_rope_view_repair_v1",
            "view_model": "logical_public_plus_physical_transpose",
            "shape_bindings": {"B": b_dim, "QH": qh_dim, "KH": kh_dim, "S": s_dim, "HD": hd_dim},
        },
    }


def _cross_entropy_repair_json(descriptor: KernelDescriptor, *, input_shape: tuple[int, ...]) -> dict[str, Any]:
    bt_dim, v_dim = map(int, input_shape)
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "input": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "target": {"dtype": "i64", "shape": _shape_entry("BT"), "layout": "row_major"},
            "ignore_index": {"dtype": "i64", "shape": _shape_entry(), "layout": "row_major"},
            "zero_f32": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "max_val": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "max_bcast": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "centered": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "exp_scores": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "sum_exp": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "log_sum_exp": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "lse": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "target_col": {"dtype": "i64", "shape": _shape_entry("BT", 1), "layout": "row_major"},
            "picked_col": {"dtype": "f32", "shape": _shape_entry("BT", 1), "layout": "row_major"},
            "picked": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "loss_row": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "valid": {"dtype": "bool", "shape": _shape_entry("BT"), "layout": "row_major"},
            "masked_loss": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "valid_f32": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "loss_sum": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "denom": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "loss": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
        },
        "ops": [
            {"op": "const", "inputs": [], "output": "zero_f32", "attrs": {"value": 0.0, "dtype": "f32"}},
            {"op": "reduce_max", "inputs": ["input"], "output": "max_val", "attrs": {"dims": [1]}},
            {
                "op": "broadcast_in_dim",
                "inputs": ["max_val"],
                "output": "max_bcast",
                "attrs": {"out_shape": _shape_entry("BT", "V"), "broadcast_dims": [0]},
            },
            {"op": "sub", "inputs": ["input", "max_bcast"], "output": "centered"},
            {"op": "exp", "inputs": ["centered"], "output": "exp_scores"},
            {"op": "reduce_sum", "inputs": ["exp_scores"], "output": "sum_exp", "attrs": {"dims": [1]}},
            {"op": "log", "inputs": ["sum_exp"], "output": "log_sum_exp"},
            {"op": "add", "inputs": ["max_val", "log_sum_exp"], "output": "lse"},
            {"op": "reshape", "inputs": ["target"], "output": "target_col", "attrs": {"shape": _shape_entry("BT", 1)}},
            {"op": "gather", "inputs": ["input", "target_col"], "output": "picked_col", "attrs": {"axis": 1}},
            {"op": "reshape", "inputs": ["picked_col"], "output": "picked", "attrs": {"shape": _shape_entry("BT")}},
            {"op": "sub", "inputs": ["lse", "picked"], "output": "loss_row"},
            {"op": "ne", "inputs": ["target", "ignore_index"], "output": "valid"},
            {"op": "where", "inputs": ["valid", "loss_row", "zero_f32"], "output": "masked_loss"},
            {"op": "cast", "inputs": ["valid"], "output": "valid_f32", "attrs": {"to": "f32"}},
            {"op": "reduce_sum", "inputs": ["masked_loss"], "output": "loss_sum", "attrs": {"dims": [0]}},
            {"op": "reduce_sum", "inputs": ["valid_f32"], "output": "denom", "attrs": {"dims": [0]}},
            {"op": "div", "inputs": ["loss_sum", "denom"], "output": "loss"},
        ],
        "outputs": ["loss"],
        "parallel_axes": ["BT"],
        "axis_roles": {"BT": "batch", "V": "channel"},
        "regions": [
            {
                "id": "ce_cfg_if",
                "kind": "if",
                "inputs": ["target", "ignore_index"],
                "outputs": [],
                "predicate": "target == ignore_index",
                "path_id": "pi_ignore",
                "ops": [],
                "regions": [],
                "meta": {"effect": "masked_loss = 0"},
            },
            {
                "id": "ce_cfg_else",
                "kind": "else",
                "inputs": ["input", "target"],
                "outputs": [],
                "predicate": "target != ignore_index",
                "path_id": "pi_active",
                "ops": [],
                "regions": [],
                "meta": {"effect": "loss_row = logsumexp(input) - input[target]"},
            },
        ],
        "meta": {
            "repaired_by": "liger_cross_entropy_loss_repair_v1",
            "shape_bindings": {"BT": bt_dim, "V": v_dim},
            "reduction": "mean",
            "ignore_index_from_runtime": True,
            "mutated_inputs": ["input"],
            "mutation_kind": "inplace_gradient_writeback",
        },
    }


def _fused_linear_cross_entropy_repair_json(
    descriptor: KernelDescriptor,
    *,
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
) -> dict[str, Any]:
    bt_dim, h_dim = map(int, input_shape)
    v_dim, h_w = map(int, weight_shape)
    if h_dim != h_w:
        raise ValueError(f"fused linear CE weight/input mismatch: input={input_shape} weight={weight_shape}")
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "input": {"dtype": "f32", "shape": _shape_entry("BT", "H"), "layout": "row_major"},
            "weight": {"dtype": "f32", "shape": _shape_entry("V", "H"), "layout": "row_major"},
            "target": {"dtype": "i64", "shape": _shape_entry("BT"), "layout": "row_major"},
            "ignore_index": {"dtype": "i64", "shape": _shape_entry(), "layout": "row_major"},
            "zero_f32": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "logits": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "max_val": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "max_bcast": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "centered": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "exp_scores": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "sum_exp": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "log_sum_exp": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "lse": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "target_col": {"dtype": "i64", "shape": _shape_entry("BT", 1), "layout": "row_major"},
            "picked_col": {"dtype": "f32", "shape": _shape_entry("BT", 1), "layout": "row_major"},
            "picked": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "loss_row": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "valid": {"dtype": "bool", "shape": _shape_entry("BT"), "layout": "row_major"},
            "masked_loss": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "valid_f32": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "loss_sum": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "denom": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "loss": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
        },
        "ops": [
            {"op": "const", "inputs": [], "output": "zero_f32", "attrs": {"value": 0.0, "dtype": "f32"}},
            {"op": "matmul", "inputs": ["input", "weight"], "output": "logits", "attrs": {"transpose_a": False, "transpose_b": True}},
            {"op": "reduce_max", "inputs": ["logits"], "output": "max_val", "attrs": {"dims": [1]}},
            {"op": "broadcast_in_dim", "inputs": ["max_val"], "output": "max_bcast", "attrs": {"out_shape": _shape_entry("BT", "V"), "broadcast_dims": [0]}},
            {"op": "sub", "inputs": ["logits", "max_bcast"], "output": "centered"},
            {"op": "exp", "inputs": ["centered"], "output": "exp_scores"},
            {"op": "reduce_sum", "inputs": ["exp_scores"], "output": "sum_exp", "attrs": {"dims": [1]}},
            {"op": "log", "inputs": ["sum_exp"], "output": "log_sum_exp"},
            {"op": "add", "inputs": ["max_val", "log_sum_exp"], "output": "lse"},
            {"op": "reshape", "inputs": ["target"], "output": "target_col", "attrs": {"shape": _shape_entry("BT", 1)}},
            {"op": "gather", "inputs": ["logits", "target_col"], "output": "picked_col", "attrs": {"axis": 1, "batch_dims": 1}},
            {"op": "reshape", "inputs": ["picked_col"], "output": "picked", "attrs": {"shape": _shape_entry("BT")}},
            {"op": "sub", "inputs": ["lse", "picked"], "output": "loss_row"},
            {"op": "ne", "inputs": ["target", "ignore_index"], "output": "valid"},
            {"op": "where", "inputs": ["valid", "loss_row", "zero_f32"], "output": "masked_loss"},
            {"op": "cast", "inputs": ["valid"], "output": "valid_f32", "attrs": {"to": "f32"}},
            {"op": "reduce_sum", "inputs": ["masked_loss"], "output": "loss_sum", "attrs": {"dims": [0]}},
            {"op": "reduce_sum", "inputs": ["valid_f32"], "output": "denom", "attrs": {"dims": [0]}},
            {"op": "div", "inputs": ["loss_sum", "denom"], "output": "loss"},
        ],
        "outputs": ["loss"],
        "parallel_axes": ["BT"],
        "axis_roles": {"BT": "batch", "H": "channel", "V": "channel"},
        "meta": {
            "repaired_by": "fused_linear_cross_entropy_repair_v1",
            "shape_bindings": {"BT": bt_dim, "H": h_dim, "V": v_dim},
            "ephemeral_workspace": ["grads"],
        },
    }


def _fused_linear_jsd_repair_json(
    descriptor: KernelDescriptor,
    *,
    student_input_shape: tuple[int, ...],
    student_weight_shape: tuple[int, ...],
    teacher_input_shape: tuple[int, ...],
    teacher_weight_shape: tuple[int, ...],
) -> dict[str, Any]:
    bt_dim, h_dim = map(int, student_input_shape)
    v_dim, h_w = map(int, student_weight_shape)
    bt_t, h_t = map(int, teacher_input_shape)
    v_t, h_tw = map(int, teacher_weight_shape)
    if (bt_dim, h_dim) != (bt_t, h_t) or (v_dim, h_w) != (v_t, h_tw) or h_dim != h_w:
        raise ValueError(
            "fused linear JSD shape mismatch: "
            f"student_input={student_input_shape} student_weight={student_weight_shape} "
            f"teacher_input={teacher_input_shape} teacher_weight={teacher_weight_shape}"
        )
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "student_input": {"dtype": "f32", "shape": _shape_entry("BT", "H"), "layout": "row_major"},
            "student_weight": {"dtype": "f32", "shape": _shape_entry("V", "H"), "layout": "row_major"},
            "teacher_input": {"dtype": "f32", "shape": _shape_entry("BT", "H"), "layout": "row_major"},
            "teacher_weight": {"dtype": "f32", "shape": _shape_entry("V", "H"), "layout": "row_major"},
            "shift_labels": {"dtype": "i64", "shape": _shape_entry("BT"), "layout": "row_major"},
            "one": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "zero": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "ignore_index": {"dtype": "i64", "shape": _shape_entry(), "layout": "row_major"},
            "temperature": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "jsd_beta": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "student_logits": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "teacher_logits": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "student_scaled": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "teacher_scaled": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "student_prob": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "teacher_prob": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "student_logp": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "teacher_logp": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "beta_p": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "one_minus_beta": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "one_minus_beta_q": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "m": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "log_m": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "loss_term1": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "loss_term2": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "loss_term3": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "loss_elem": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "valid": {"dtype": "bool", "shape": _shape_entry("BT"), "layout": "row_major"},
            "valid_f32": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "n_non_ignore": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "scale": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "scale_bcast": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "ignore_mask_bcast": {"dtype": "bool", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "zero_bcast": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "loss_scaled": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "loss_masked": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "token_loss": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "loss": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
        },
        "ops": [
            {"op": "const", "inputs": [], "output": "one", "attrs": {"value": 1.0, "dtype": "f32"}},
            {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0, "dtype": "f32"}},
            {"op": "const", "inputs": [], "output": "ignore_index", "attrs": {"value": -100, "dtype": "i64"}},
            {"op": "const", "inputs": [], "output": "temperature", "attrs": {"value": 1.0, "dtype": "f32"}},
            {"op": "const", "inputs": [], "output": "jsd_beta", "attrs": {"value": 0.5, "dtype": "f32"}},
            {"op": "matmul", "inputs": ["student_input", "student_weight"], "output": "student_logits", "attrs": {"transpose_a": False, "transpose_b": True}},
            {"op": "matmul", "inputs": ["teacher_input", "teacher_weight"], "output": "teacher_logits", "attrs": {"transpose_a": False, "transpose_b": True}},
            {"op": "div", "inputs": ["student_logits", "temperature"], "output": "student_scaled"},
            {"op": "div", "inputs": ["teacher_logits", "temperature"], "output": "teacher_scaled"},
            {"op": "softmax", "inputs": ["student_scaled"], "output": "student_prob", "attrs": {"axis": 1, "dims": [1]}},
            {"op": "softmax", "inputs": ["teacher_scaled"], "output": "teacher_prob", "attrs": {"axis": 1, "dims": [1]}},
            {"op": "log", "inputs": ["student_prob"], "output": "student_logp"},
            {"op": "log", "inputs": ["teacher_prob"], "output": "teacher_logp"},
            {"op": "mul", "inputs": ["jsd_beta", "teacher_prob"], "output": "beta_p"},
            {"op": "sub", "inputs": ["one", "jsd_beta"], "output": "one_minus_beta"},
            {"op": "mul", "inputs": ["one_minus_beta", "student_prob"], "output": "one_minus_beta_q"},
            {"op": "add", "inputs": ["beta_p", "one_minus_beta_q"], "output": "m"},
            {"op": "log", "inputs": ["m"], "output": "log_m"},
            {"op": "mul", "inputs": ["beta_p", "teacher_logp"], "output": "loss_term1"},
            {"op": "mul", "inputs": ["one_minus_beta_q", "student_logp"], "output": "loss_term2"},
            {"op": "mul", "inputs": ["m", "log_m"], "output": "loss_term3"},
            {"op": "add", "inputs": ["loss_term1", "loss_term2"], "output": "loss_elem"},
            {"op": "sub", "inputs": ["loss_elem", "loss_term3"], "output": "loss_elem"},
            {"op": "ne", "inputs": ["shift_labels", "ignore_index"], "output": "valid"},
            {"op": "cast", "inputs": ["valid"], "output": "valid_f32", "attrs": {"to": "f32"}},
            {"op": "reduce_sum", "inputs": ["valid_f32"], "output": "n_non_ignore", "attrs": {"dims": [0]}},
            {"op": "div", "inputs": ["one", "n_non_ignore"], "output": "scale"},
            {"op": "broadcast_in_dim", "inputs": ["scale"], "output": "scale_bcast", "attrs": {"out_shape": _shape_entry("BT", "V"), "broadcast_dims": []}},
            {"op": "broadcast_in_dim", "inputs": ["valid"], "output": "ignore_mask_bcast", "attrs": {"out_shape": _shape_entry("BT", "V"), "broadcast_dims": [0]}},
            {"op": "broadcast_in_dim", "inputs": ["zero"], "output": "zero_bcast", "attrs": {"out_shape": _shape_entry("BT", "V"), "broadcast_dims": []}},
            {"op": "mul", "inputs": ["loss_elem", "scale_bcast"], "output": "loss_scaled"},
            {"op": "where", "inputs": ["ignore_mask_bcast", "loss_scaled", "zero_bcast"], "output": "loss_masked"},
            {"op": "reduce_sum", "inputs": ["loss_masked"], "output": "token_loss", "attrs": {"dims": [1]}},
            {"op": "reduce_sum", "inputs": ["token_loss"], "output": "loss", "attrs": {"dims": [0]}},
        ],
        "outputs": ["loss"],
        "parallel_axes": ["BT"],
        "axis_roles": {"BT": "batch", "H": "channel", "V": "channel"},
        "meta": {
            "repaired_by": "fused_linear_jsd_repair_v1",
            "shape_bindings": {"BT": bt_dim, "H": h_dim, "V": v_dim},
            "ephemeral_workspace": ["grads"],
        },
    }


def _qwen2vl_mrope_repair_json(
    descriptor: KernelDescriptor,
    *,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
) -> dict[str, Any]:
    b_dim, qh_dim, s_dim, hd_dim = map(int, q_shape)
    b_k, kh_dim, s_k, hd_k = map(int, k_shape)
    if (b_dim, s_dim, hd_dim) != (b_k, s_k, hd_k):
        raise ValueError(f"qwen2vl mrope q/k shape mismatch: q={q_shape} k={k_shape}")
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "q": {"dtype": "f32", "shape": _shape_entry("B", "QH", "S", "HD"), "layout": {"kind": "custom", "params": {"axes": ["B", "H", "S", "HD"]}}},
            "k": {"dtype": "f32", "shape": _shape_entry("B", "KH", "S", "HD"), "layout": {"kind": "custom", "params": {"axes": ["B", "H", "S", "HD"]}}},
            "cos_combined": {"dtype": "f32", "shape": _shape_entry("B", "S", "HD"), "layout": "row_major"},
            "sin_combined": {"dtype": "f32", "shape": _shape_entry("B", "S", "HD"), "layout": "row_major"},
            "q_phys": {"dtype": "f32", "shape": _shape_entry("B", "S", "QH", "HD"), "layout": {"kind": "custom", "params": {"axes": ["B", "S", "H", "HD"], "view_perm": [0, 2, 1, 3]}}, "view_of": "q", "alias_group": "q_storage_view"},
            "k_phys": {"dtype": "f32", "shape": _shape_entry("B", "S", "KH", "HD"), "layout": {"kind": "custom", "params": {"axes": ["B", "S", "H", "HD"], "view_perm": [0, 2, 1, 3]}}, "view_of": "k", "alias_group": "k_storage_view"},
            "q_rot_phys": {"dtype": "f32", "shape": _shape_entry("B", "S", "QH", "HD"), "layout": "row_major"},
            "k_rot_phys": {"dtype": "f32", "shape": _shape_entry("B", "S", "KH", "HD"), "layout": "row_major"},
            "q_out": {"dtype": "f32", "shape": _shape_entry("B", "QH", "S", "HD"), "layout": {"kind": "custom", "params": {"axes": ["B", "H", "S", "HD"]}}},
            "k_out": {"dtype": "f32", "shape": _shape_entry("B", "KH", "S", "HD"), "layout": {"kind": "custom", "params": {"axes": ["B", "H", "S", "HD"]}}},
        },
        "ops": [
            {"op": "transpose", "inputs": ["q"], "output": "q_phys", "attrs": {"perm": [0, 2, 1, 3]}},
            {"op": "transpose", "inputs": ["k"], "output": "k_phys", "attrs": {"perm": [0, 2, 1, 3]}},
            {"op": "rope", "inputs": ["q_phys", "cos_combined", "sin_combined"], "output": "q_rot_phys", "attrs": {"input_layout": "bshd"}},
            {"op": "rope", "inputs": ["k_phys", "cos_combined", "sin_combined"], "output": "k_rot_phys", "attrs": {"input_layout": "bshd"}},
            {"op": "transpose", "inputs": ["q_rot_phys"], "output": "q_out", "attrs": {"perm": [0, 2, 1, 3]}},
            {"op": "transpose", "inputs": ["k_rot_phys"], "output": "k_out", "attrs": {"perm": [0, 2, 1, 3]}},
        ],
        "outputs": ["q_out", "k_out"],
        "parallel_axes": ["B", "S", "QH", "KH"],
        "axis_roles": {"B": "batch", "S": "spatial", "QH": "channel", "KH": "channel", "HD": "channel"},
        "meta": {
            "repaired_by": "qwen2vl_mrope_repair_v1",
            "shape_bindings": {"B": b_dim, "QH": qh_dim, "KH": kh_dim, "S": s_dim, "HD": hd_dim},
            "mrope_combined_inputs": True,
        },
    }


def _tvd_repair_json(descriptor: KernelDescriptor, *, input_shape: tuple[int, ...]) -> dict[str, Any]:
    bt_dim, v_dim = map(int, input_shape)
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "input": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "target": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "half": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "bt_scalar": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "delta": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "abs_delta": {"dtype": "f32", "shape": _shape_entry("BT", "V"), "layout": "row_major"},
            "row_sum": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "row_tvd": {"dtype": "f32", "shape": _shape_entry("BT"), "layout": "row_major"},
            "loss_sum": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "loss": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
        },
        "ops": [
            {"op": "const", "inputs": [], "output": "half", "attrs": {"value": 0.5, "dtype": "f32"}},
            {"op": "const", "inputs": [], "output": "bt_scalar", "attrs": {"value": "BT", "dtype": "f32"}},
            {"op": "sub", "inputs": ["input", "target"], "output": "delta"},
            {"op": "abs", "inputs": ["delta"], "output": "abs_delta"},
            {"op": "reduce_sum", "inputs": ["abs_delta"], "output": "row_sum", "attrs": {"dims": [1]}},
            {"op": "mul", "inputs": ["row_sum", "half"], "output": "row_tvd"},
            {"op": "reduce_sum", "inputs": ["row_tvd"], "output": "loss_sum", "attrs": {"dims": [0]}},
            {"op": "div", "inputs": ["loss_sum", "bt_scalar"], "output": "loss"},
        ],
        "outputs": ["loss"],
        "parallel_axes": ["BT"],
        "axis_roles": {"BT": "batch", "V": "channel"},
        "meta": {
            "repaired_by": "distribution_distance_repair_v1",
            "shape_bindings": {"BT": bt_dim, "V": v_dim},
            "reduction": "batchmean",
            "ephemeral_workspace": ["grads"],
        },
    }


def _poly_norm_repair_json(descriptor: KernelDescriptor, *, input_shape: tuple[int, ...]) -> dict[str, Any]:
    m_dim, n_dim = map(int, input_shape)
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "X": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "W": {"dtype": "f32", "shape": _shape_entry(3), "layout": "row_major"},
            "B": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "eps": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "N_scalar": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "X_sq": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "X_cu": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "X_cu_sq": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "X_sq_sq": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "mean_sq_3": {"dtype": "f32", "shape": _shape_entry("M"), "layout": "row_major"},
            "mean_sq_2": {"dtype": "f32", "shape": _shape_entry("M"), "layout": "row_major"},
            "mean_sq_1": {"dtype": "f32", "shape": _shape_entry("M"), "layout": "row_major"},
            "rstd_3": {"dtype": "f32", "shape": _shape_entry("M"), "layout": "row_major"},
            "rstd_2": {"dtype": "f32", "shape": _shape_entry("M"), "layout": "row_major"},
            "rstd_1": {"dtype": "f32", "shape": _shape_entry("M"), "layout": "row_major"},
            "rstd_3_bc": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "rstd_2_bc": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "rstd_1_bc": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "norm_x3": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "norm_x2": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "norm_x1": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "idx0": {"dtype": "i32", "shape": _shape_entry(), "layout": "row_major"},
            "idx1": {"dtype": "i32", "shape": _shape_entry(), "layout": "row_major"},
            "idx2": {"dtype": "i32", "shape": _shape_entry(), "layout": "row_major"},
            "w0": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "w1": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "w2": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "w0_bc": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "w1_bc": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "w2_bc": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "b_bc": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "term0": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "term1": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "term2": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "sum01": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
            "Y": {"dtype": "f32", "shape": _shape_entry("M", "N"), "layout": "row_major"},
        },
        "ops": [
            {"op": "const", "inputs": [], "output": "eps", "attrs": {"value": 1.0e-6, "dtype": "f32"}},
            {"op": "const", "inputs": [], "output": "N_scalar", "attrs": {"value": "N", "dtype": "f32"}},
            {"op": "mul", "inputs": ["X", "X"], "output": "X_sq"},
            {"op": "mul", "inputs": ["X_sq", "X"], "output": "X_cu"},
            {"op": "mul", "inputs": ["X_cu", "X_cu"], "output": "X_cu_sq"},
            {"op": "mul", "inputs": ["X_sq", "X_sq"], "output": "X_sq_sq"},
            {"op": "reduce_sum", "inputs": ["X_cu_sq"], "output": "mean_sq_3", "attrs": {"dims": [1]}},
            {"op": "div", "inputs": ["mean_sq_3", "N_scalar"], "output": "mean_sq_3"},
            {"op": "add", "inputs": ["mean_sq_3", "eps"], "output": "mean_sq_3"},
            {"op": "rsqrt", "inputs": ["mean_sq_3"], "output": "rstd_3"},
            {"op": "reduce_sum", "inputs": ["X_sq_sq"], "output": "mean_sq_2", "attrs": {"dims": [1]}},
            {"op": "div", "inputs": ["mean_sq_2", "N_scalar"], "output": "mean_sq_2"},
            {"op": "add", "inputs": ["mean_sq_2", "eps"], "output": "mean_sq_2"},
            {"op": "rsqrt", "inputs": ["mean_sq_2"], "output": "rstd_2"},
            {"op": "reduce_sum", "inputs": ["X_sq"], "output": "mean_sq_1", "attrs": {"dims": [1]}},
            {"op": "div", "inputs": ["mean_sq_1", "N_scalar"], "output": "mean_sq_1"},
            {"op": "add", "inputs": ["mean_sq_1", "eps"], "output": "mean_sq_1"},
            {"op": "rsqrt", "inputs": ["mean_sq_1"], "output": "rstd_1"},
            {"op": "broadcast_in_dim", "inputs": ["rstd_3"], "output": "rstd_3_bc", "attrs": {"out_shape": _shape_entry("M", "N"), "broadcast_dims": [0]}},
            {"op": "broadcast_in_dim", "inputs": ["rstd_2"], "output": "rstd_2_bc", "attrs": {"out_shape": _shape_entry("M", "N"), "broadcast_dims": [0]}},
            {"op": "broadcast_in_dim", "inputs": ["rstd_1"], "output": "rstd_1_bc", "attrs": {"out_shape": _shape_entry("M", "N"), "broadcast_dims": [0]}},
            {"op": "mul", "inputs": ["X_cu", "rstd_3_bc"], "output": "norm_x3"},
            {"op": "mul", "inputs": ["X_sq", "rstd_2_bc"], "output": "norm_x2"},
            {"op": "mul", "inputs": ["X", "rstd_1_bc"], "output": "norm_x1"},
            {"op": "const", "inputs": [], "output": "idx0", "attrs": {"value": 0, "dtype": "i32"}},
            {"op": "const", "inputs": [], "output": "idx1", "attrs": {"value": 1, "dtype": "i32"}},
            {"op": "const", "inputs": [], "output": "idx2", "attrs": {"value": 2, "dtype": "i32"}},
            {"op": "gather", "inputs": ["W", "idx0"], "output": "w0", "attrs": {"axis": 0}},
            {"op": "gather", "inputs": ["W", "idx1"], "output": "w1", "attrs": {"axis": 0}},
            {"op": "gather", "inputs": ["W", "idx2"], "output": "w2", "attrs": {"axis": 0}},
            {"op": "broadcast_in_dim", "inputs": ["w0"], "output": "w0_bc", "attrs": {"out_shape": _shape_entry("M", "N"), "broadcast_dims": []}},
            {"op": "broadcast_in_dim", "inputs": ["w1"], "output": "w1_bc", "attrs": {"out_shape": _shape_entry("M", "N"), "broadcast_dims": []}},
            {"op": "broadcast_in_dim", "inputs": ["w2"], "output": "w2_bc", "attrs": {"out_shape": _shape_entry("M", "N"), "broadcast_dims": []}},
            {"op": "broadcast_in_dim", "inputs": ["B"], "output": "b_bc", "attrs": {"out_shape": _shape_entry("M", "N"), "broadcast_dims": []}},
            {"op": "mul", "inputs": ["w0_bc", "norm_x3"], "output": "term0"},
            {"op": "mul", "inputs": ["w1_bc", "norm_x2"], "output": "term1"},
            {"op": "mul", "inputs": ["w2_bc", "norm_x1"], "output": "term2"},
            {"op": "add", "inputs": ["term0", "term1"], "output": "sum01"},
            {"op": "add", "inputs": ["sum01", "term2"], "output": "sum01"},
            {"op": "add", "inputs": ["sum01", "b_bc"], "output": "Y"},
        ],
        "outputs": ["Y"],
        "parallel_axes": ["M"],
        "axis_roles": {"M": "batch", "N": "reduction"},
        "meta": {
            "repaired_by": "poly_rms_resident_repair_v1",
            "shape_bindings": {"M": m_dim, "N": n_dim},
            "ephemeral_workspace": ["RSTD"],
        },
    }


def _tiled_mlp_repair_json(
    descriptor: KernelDescriptor,
    *,
    b_dim: int,
    s_dim: int,
    h_dim: int,
    i_dim: int,
) -> dict[str, Any]:
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "X": {"dtype": "f32", "shape": _shape_entry("B", "S", "H"), "layout": "row_major"},
            "GateW": {"dtype": "f32", "shape": _shape_entry("H", "I"), "layout": "row_major"},
            "UpW": {"dtype": "f32", "shape": _shape_entry("H", "I"), "layout": "row_major"},
            "DownW": {"dtype": "f32", "shape": _shape_entry("I", "H"), "layout": "row_major"},
            "gate": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "gate_sq": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "gate_cube": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "c_044715": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "scaled_cube": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "gate_sum": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "c_sqrt2pi": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "tanh_input": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "tanh_out": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "c_1": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "gelu_factor": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "c_05": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "gate_mul": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "up": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "gelu_gate": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "gated": {"dtype": "f32", "shape": _shape_entry("B", "S", "I"), "layout": "row_major"},
            "Y": {"dtype": "f32", "shape": _shape_entry("B", "S", "H"), "layout": "row_major"},
        },
        "ops": [
            {"op": "matmul", "inputs": ["X", "GateW"], "output": "gate"},
            {"op": "mul", "inputs": ["gate", "gate"], "output": "gate_sq"},
            {"op": "mul", "inputs": ["gate_sq", "gate"], "output": "gate_cube"},
            {"op": "const", "inputs": [], "output": "c_044715", "attrs": {"value": 0.044715, "dtype": "f32"}},
            {"op": "mul", "inputs": ["gate_cube", "c_044715"], "output": "scaled_cube"},
            {"op": "add", "inputs": ["gate", "scaled_cube"], "output": "gate_sum"},
            {"op": "const", "inputs": [], "output": "c_sqrt2pi", "attrs": {"value": 0.7978845608, "dtype": "f32"}},
            {"op": "mul", "inputs": ["gate_sum", "c_sqrt2pi"], "output": "tanh_input"},
            {"op": "tanh", "inputs": ["tanh_input"], "output": "tanh_out"},
            {"op": "const", "inputs": [], "output": "c_1", "attrs": {"value": 1.0, "dtype": "f32"}},
            {"op": "add", "inputs": ["c_1", "tanh_out"], "output": "gelu_factor"},
            {"op": "const", "inputs": [], "output": "c_05", "attrs": {"value": 0.5, "dtype": "f32"}},
            {"op": "mul", "inputs": ["gate", "gelu_factor"], "output": "gate_mul"},
            {"op": "mul", "inputs": ["gate_mul", "c_05"], "output": "gelu_gate"},
            {"op": "matmul", "inputs": ["X", "UpW"], "output": "up"},
            {"op": "mul", "inputs": ["gelu_gate", "up"], "output": "gated"},
            {"op": "matmul", "inputs": ["gated", "DownW"], "output": "Y"},
        ],
        "outputs": ["Y"],
        "parallel_axes": ["B", "S"],
        "axis_roles": {"B": "batch", "S": "spatial", "H": "channel", "I": "channel"},
        "meta": {
            "repaired_by": "tiled_mlp_repair_v1",
            "shape_bindings": {"B": int(b_dim), "S": int(s_dim), "H": int(h_dim), "I": int(i_dim)},
        },
    }


def _multi_token_attention_repair_json(
    descriptor: KernelDescriptor,
    *,
    b_dim: int,
    cin_dim: int,
    cout_dim: int,
    l_dim: int,
    k_dim: int,
    groups: int,
) -> dict[str, Any]:
    cin_per_group = int(cin_dim // max(1, groups))
    pad = int(k_dim // 2)
    return {
        "name": descriptor.name,
        "kernel_type": descriptor.name,
        "tensors": {
            "scores": {"dtype": "f32", "shape": _shape_entry("B", "CIN", "L", "L"), "layout": "row_major"},
            "weight": {
                "dtype": "f32",
                "shape": _shape_entry("COUT", "CIN_per_group", "K", "K"),
                "layout": "row_major",
            },
            "bias": {"dtype": "f32", "shape": _shape_entry("COUT"), "layout": "row_major"},
            "row_idx": {"dtype": "i32", "shape": _shape_entry("L", "L"), "layout": "row_major"},
            "col_idx": {"dtype": "i32", "shape": _shape_entry("L", "L"), "layout": "row_major"},
            "future_mask": {"dtype": "bool", "shape": _shape_entry("L", "L"), "layout": "row_major"},
            "future_mask_scores": {"dtype": "bool", "shape": _shape_entry("B", "CIN", "L", "L"), "layout": "row_major"},
            "future_mask_out": {"dtype": "bool", "shape": _shape_entry("B", "COUT", "L", "L"), "layout": "row_major"},
            "neg_inf": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "neg_inf_bc": {"dtype": "f32", "shape": _shape_entry("B", "CIN", "L", "L"), "layout": "row_major"},
            "zero": {"dtype": "f32", "shape": _shape_entry(), "layout": "row_major"},
            "zero_bc": {"dtype": "f32", "shape": _shape_entry("B", "COUT", "L", "L"), "layout": "row_major"},
            "scores_masked": {"dtype": "f32", "shape": _shape_entry("B", "CIN", "L", "L"), "layout": "row_major"},
            "attn_weights": {"dtype": "f32", "shape": _shape_entry("B", "CIN", "L", "L"), "layout": "row_major"},
            "out_conv": {"dtype": "f32", "shape": _shape_entry("B", "COUT", "L", "L"), "layout": "row_major"},
            "Y": {"dtype": "f32", "shape": _shape_entry("B", "COUT", "L", "L"), "layout": "row_major"},
        },
        "ops": [
            {"op": "iota", "inputs": [], "output": "row_idx", "attrs": {"shape": _shape_entry("L", "L"), "axis": 0, "dtype": "i32"}},
            {"op": "iota", "inputs": [], "output": "col_idx", "attrs": {"shape": _shape_entry("L", "L"), "axis": 1, "dtype": "i32"}},
            {"op": "gt", "inputs": ["col_idx", "row_idx"], "output": "future_mask"},
            {
                "op": "broadcast_in_dim",
                "inputs": ["future_mask"],
                "output": "future_mask_scores",
                "attrs": {"out_shape": _shape_entry("B", "CIN", "L", "L"), "broadcast_dims": [2, 3]},
            },
            {
                "op": "broadcast_in_dim",
                "inputs": ["future_mask"],
                "output": "future_mask_out",
                "attrs": {"out_shape": _shape_entry("B", "COUT", "L", "L"), "broadcast_dims": [2, 3]},
            },
            {"op": "const", "inputs": [], "output": "neg_inf", "attrs": {"value": -1.0e9, "dtype": "f32"}},
            {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0, "dtype": "f32"}},
            {
                "op": "broadcast_in_dim",
                "inputs": ["neg_inf"],
                "output": "neg_inf_bc",
                "attrs": {"out_shape": _shape_entry("B", "CIN", "L", "L"), "broadcast_dims": []},
            },
            {
                "op": "broadcast_in_dim",
                "inputs": ["zero"],
                "output": "zero_bc",
                "attrs": {"out_shape": _shape_entry("B", "COUT", "L", "L"), "broadcast_dims": []},
            },
            {"op": "where", "inputs": ["future_mask_scores", "neg_inf_bc", "scores"], "output": "scores_masked"},
            {"op": "softmax", "inputs": ["scores_masked"], "output": "attn_weights", "attrs": {"axis": 3, "dims": [3]}},
            {
                "op": "conv2d",
                "inputs": ["attn_weights", "weight", "bias"],
                "output": "out_conv",
                "attrs": {
                    "stride": [1, 1],
                    "padding": [pad, pad],
                    "dilation": [1, 1],
                    "groups": int(groups),
                },
            },
            {"op": "where", "inputs": ["future_mask_out", "zero_bc", "out_conv"], "output": "Y"},
        ],
        "outputs": ["Y"],
        "parallel_axes": ["B", "COUT", "L"],
        "axis_roles": {"B": "batch", "CIN": "channel", "COUT": "channel", "L": "spatial", "K": "kernel"},
        "meta": {
            "repaired_by": "multi_token_attention_repair_v1",
            "shape_bindings": {
                "B": int(b_dim),
                "CIN": int(cin_dim),
                "COUT": int(cout_dim),
                "CIN_per_group": int(cin_per_group),
                "K": int(k_dim),
                "L": int(l_dim),
                "L_out": int(l_dim),
                "groups": int(groups),
            },
        },
    }


def prefill_candidate_for_descriptor(descriptor: KernelDescriptor) -> tuple[CandidateIntent | None, list[str]]:
    """
    Build a deterministic frontend candidate directly from descriptor evidence
    when the kernel family has a known semantic repair.
    """
    repairs: list[str] = []
    shapes = _baseline_array_shapes(descriptor)
    arg_names = _descriptor_arg_names(descriptor)
    source_text = str(getattr(descriptor, "source_text", "") or "")
    has_dense_distribution_target = (
        {"Y_ptr", "Y_stride"} <= arg_names
        or {"gt_ptr", "gt_stride"} <= arg_names
        or "Y_ptr +=" in source_text
        or "gt_ptr +=" in source_text
    )
    has_tvd_signature = (
        {"p_ptr", "q_ptr", "loss_ptr"} <= arg_names
        or "TVD(P || Q)" in source_text
        or "tv_loss = 0.5 * tl.abs(p - q)" in source_text
        or "LigerTVDLossFunction.apply" in source_text
    )
    has_poly_norm_signature = (
        ("PolyNorm formula" in source_text)
        or (
            {"X_ptr", "W_ptr", "B_ptr", "Y_ptr"} <= arg_names
            and "RSTD_ptr" in arg_names
        )
    )
    has_tiled_mlp_signature = ("apply_tiled_mlp" in source_text) or ("LigerTiledGEGLUMLP" in source_text)
    has_multi_token_attention_signature = (
        "liger_multi_token_attention" in source_text
        or "multi_token_attention" in source_text
        or "_mask_fwd_kernel" in source_text
    )
    fused_linear_ce_shapes = (
        len(tuple(shapes.get("input", ()))) == 2
        and len(tuple(shapes.get("weight", ()))) == 2
        and len(tuple(shapes.get("target", ()))) == 1
        and len(tuple(shapes.get("loss", ()))) == 0
    )
    fused_linear_jsd_shapes = (
        len(tuple(shapes.get("student_input", ()))) == 2
        and len(tuple(shapes.get("student_weight", ()))) == 2
        and len(tuple(shapes.get("teacher_input", ()))) == 2
        and len(tuple(shapes.get("teacher_weight", ()))) == 2
        and len(tuple(shapes.get("loss", ()))) == 0
    )
    has_mrope_planes = any(str(name).startswith("mrope_section") for name in arg_names) or ("mrope_section" in source_text)
    has_complex_rope = (
        any("freqs" in str(name).lower() for name in arg_names)
        or "freqs_cis" in source_text
        or "freqs_complex" in source_text
        or "view_as_real" in source_text
    )
    canonical_shapes = {
        str(k): int(v)
        for k, v in dict((descriptor.launch or {}).get("canonical_shapes") or {}).items()
        if str(k).strip()
    }
    if (not fused_linear_ce_shapes) and {"BT", "H", "V"} <= set(canonical_shapes):
        if "LigerFusedLinearCrossEntropyFunction" in source_text:
            shapes = dict(shapes)
            shapes.setdefault("input", (int(canonical_shapes["BT"]), int(canonical_shapes["H"])))
            shapes.setdefault("weight", (int(canonical_shapes["V"]), int(canonical_shapes["H"])))
            shapes.setdefault("target", (int(canonical_shapes["BT"]),))
            shapes.setdefault("loss", tuple())
            fused_linear_ce_shapes = True
    if (not fused_linear_jsd_shapes) and {"BT", "H", "V"} <= set(canonical_shapes):
        if "LigerFusedLinearJSD" in source_text or ("jsd_beta" in source_text and "teacher_input" in source_text):
            shapes = dict(shapes)
            shapes.setdefault("student_input", (int(canonical_shapes["BT"]), int(canonical_shapes["H"])))
            shapes.setdefault("teacher_input", (int(canonical_shapes["BT"]), int(canonical_shapes["H"])))
            shapes.setdefault("student_weight", (int(canonical_shapes["V"]), int(canonical_shapes["H"])))
            shapes.setdefault("teacher_weight", (int(canonical_shapes["V"]), int(canonical_shapes["H"])))
            shapes.setdefault("loss", tuple())
            fused_linear_jsd_shapes = True

    if (not shapes) and (not has_mrope_planes) and {"B", "QH", "KH", "S", "HD"} <= set(canonical_shapes):
        if has_complex_rope:
            shapes = {
                "q": (
                    int(canonical_shapes["B"]),
                    int(canonical_shapes["S"]),
                    int(canonical_shapes["QH"]),
                    int(canonical_shapes["HD"]),
                ),
                "k": (
                    int(canonical_shapes["B"]),
                    int(canonical_shapes["S"]),
                    int(canonical_shapes["KH"]),
                    int(canonical_shapes["HD"]),
                ),
                "cos": (int(canonical_shapes["S"]), int(canonical_shapes["HD"]) // 2),
                "sin": (int(canonical_shapes["S"]), int(canonical_shapes["HD"]) // 2),
            }
        else:
            shapes = {
                "q": (
                    int(canonical_shapes["B"]),
                    int(canonical_shapes["QH"]),
                    int(canonical_shapes["S"]),
                    int(canonical_shapes["HD"]),
                ),
                "k": (
                    int(canonical_shapes["B"]),
                    int(canonical_shapes["KH"]),
                    int(canonical_shapes["S"]),
                    int(canonical_shapes["HD"]),
                ),
                "cos": (1, int(canonical_shapes["S"]), int(canonical_shapes["HD"])),
                "sin": (1, int(canonical_shapes["S"]), int(canonical_shapes["HD"])),
            }
    if (not shapes) and has_mrope_planes and {"B", "QH", "KH", "S", "HD"} <= set(canonical_shapes):
        shapes = {
            "q": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["QH"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
            "k": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["KH"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
            "cos_t": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
            "cos_h": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
            "cos_w": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
            "sin_t": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
            "sin_h": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
            "sin_w": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["HD"]),
            ),
        }
    if (not shapes) and has_tvd_signature and {"BT", "V"} <= set(canonical_shapes):
        shapes = {
            "input": (int(canonical_shapes["BT"]), int(canonical_shapes["V"])),
            "target": (int(canonical_shapes["BT"]), int(canonical_shapes["V"])),
            "loss": tuple(),
        }
    if (not shapes) and has_poly_norm_signature and {"M", "N"} <= set(canonical_shapes):
        shapes = {
            "X": (int(canonical_shapes["M"]), int(canonical_shapes["N"])),
            "W": (3,),
            "B": tuple(),
            "Y": (int(canonical_shapes["M"]), int(canonical_shapes["N"])),
        }
    if (not shapes) and has_tiled_mlp_signature and {"B", "S", "H", "I"} <= set(canonical_shapes):
        shapes = {
            "X": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["H"]),
            ),
            "GateW": (int(canonical_shapes["H"]), int(canonical_shapes["I"])),
            "UpW": (int(canonical_shapes["H"]), int(canonical_shapes["I"])),
            "DownW": (int(canonical_shapes["I"]), int(canonical_shapes["H"])),
            "Y": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["S"]),
                int(canonical_shapes["H"]),
            ),
        }
    if (not shapes) and has_multi_token_attention_signature and {"B", "CIN", "COUT", "L", "K"} <= set(canonical_shapes):
        groups = int(canonical_shapes.get("groups", 1))
        cin = int(canonical_shapes["CIN"])
        shapes = {
            "scores": (
                int(canonical_shapes["B"]),
                cin,
                int(canonical_shapes["L"]),
                int(canonical_shapes["L"]),
            ),
            "weight": (
                int(canonical_shapes["COUT"]),
                max(1, cin // max(1, groups)),
                int(canonical_shapes["K"]),
                int(canonical_shapes["K"]),
            ),
            "bias": (int(canonical_shapes["COUT"]),),
            "Y": (
                int(canonical_shapes["B"]),
                int(canonical_shapes["COUT"]),
                int(canonical_shapes["L"]),
                int(canonical_shapes["L"]),
            ),
            "groups": tuple(),
        }
    q_shape = tuple(shapes.get("q", ()))
    k_shape = tuple(shapes.get("k", ()))
    cos_shape = tuple(shapes.get("cos", ()))
    sin_shape = tuple(shapes.get("sin", ()))
    # Narrow the deterministic rope repair to the simple dual-view RoPE case.
    # Multi-plane rotary variants (for example M-RoPE with an extra plane axis)
    # must go through the live frontend instead of being collapsed into the
    # standard [1,S,HD] cosine/sine model.
    if has_mrope_planes and len(tuple(shapes.get("q", ()))) == 4 and len(tuple(shapes.get("k", ()))) == 4:
        repaired = parse_candidate_json(
            _qwen2vl_mrope_repair_json(
                descriptor,
                q_shape=tuple(shapes.get("q", ())),
                k_shape=tuple(shapes.get("k", ())),
            )
        )
        repairs.append("qwen2vl_mrope_repair_v1")
        return repaired, repairs

    if len(q_shape) == 4 and len(k_shape) == 4 and len(cos_shape) in {2, 3} and len(sin_shape) in {2, 3}:
        repaired = parse_candidate_json(
            _rope_repair_json(
                descriptor,
                q_shape=q_shape,
                k_shape=k_shape,
                cos_shape=cos_shape,
            )
        )
        repairs.append("liger_rope_view_repair_v1")
        return repaired, repairs

    canonical_keys = set(canonical_shapes)
    allow_plain_ce_prefill = (
        canonical_keys.issubset({"BT", "V"}) and {"BT", "V"} <= canonical_keys and (not has_tvd_signature)
    )
    if (not shapes) and (not has_dense_distribution_target) and allow_plain_ce_prefill:
        shapes = {
            "input": (int(canonical_shapes["BT"]), int(canonical_shapes["V"])),
            "target": (int(canonical_shapes["BT"]),),
            "loss": tuple(),
        }
    input_shape = tuple(shapes.get("input", ()))
    target_shape = tuple(shapes.get("target", ()))
    loss_shape = tuple(shapes.get("loss", ()))
    poly_input_shape = tuple(shapes.get("X", ()))
    poly_weight_shape = tuple(shapes.get("W", shapes.get("weight", ())))
    poly_bias_shape = tuple(shapes.get("B", shapes.get("bias", ())))
    poly_output_shape = tuple(shapes.get("Y", ()))
    tiled_x_shape = tuple(shapes.get("X", ()))
    tiled_gatew_shape = tuple(shapes.get("GateW", ()))
    tiled_upw_shape = tuple(shapes.get("UpW", ()))
    tiled_downw_shape = tuple(shapes.get("DownW", ()))
    tiled_y_shape = tuple(shapes.get("Y", ()))
    mta_scores_shape = tuple(shapes.get("scores", ()))
    mta_weight_shape = tuple(shapes.get("weight", ()))
    mta_bias_shape = tuple(shapes.get("bias", ()))
    mta_y_shape = tuple(shapes.get("Y", ()))
    if fused_linear_ce_shapes:
        repaired = parse_candidate_json(
            _fused_linear_cross_entropy_repair_json(
                descriptor,
                input_shape=tuple(shapes.get("input", ())),
                weight_shape=tuple(shapes.get("weight", ())),
            )
        )
        repairs.append("fused_linear_cross_entropy_repair_v1")
        return repaired, repairs
    if fused_linear_jsd_shapes:
        repaired = parse_candidate_json(
            _fused_linear_jsd_repair_json(
                descriptor,
                student_input_shape=tuple(shapes.get("student_input", ())),
                student_weight_shape=tuple(shapes.get("student_weight", ())),
                teacher_input_shape=tuple(shapes.get("teacher_input", ())),
                teacher_weight_shape=tuple(shapes.get("teacher_weight", ())),
            )
        )
        repairs.append("fused_linear_jsd_repair_v1")
        return repaired, repairs
    # Cross-entropy repair only applies to class-index targets [BT], not dense
    # distribution targets [BT,V] used by KL/JSD-like kernels.
    if (
        len(input_shape) == 2
        and len(target_shape) == 1
        and len(loss_shape) == 0
        and len(tuple(shapes.get("weight", ()))) != 2
    ):
        repaired = parse_candidate_json(
            _cross_entropy_repair_json(
                descriptor,
                input_shape=input_shape,
            )
        )
        repairs.append("liger_cross_entropy_loss_repair_v1")
        return repaired, repairs
    if len(input_shape) == 2 and len(target_shape) == 2 and len(loss_shape) == 0:
        repaired = parse_candidate_json(
            _tvd_repair_json(
                descriptor,
                input_shape=input_shape,
            )
        )
        repairs.append("distribution_distance_repair_v1")
        return repaired, repairs
    if len(poly_input_shape) == 2 and len(poly_weight_shape) == 1 and int(poly_weight_shape[0]) == 3 and len(poly_bias_shape) == 0 and poly_output_shape == poly_input_shape:
        repaired = parse_candidate_json(
            _poly_norm_repair_json(
                descriptor,
                input_shape=poly_input_shape,
            )
        )
        repairs.append("poly_rms_resident_repair_v1")
        return repaired, repairs
    if (
        len(tiled_x_shape) == 3
        and tiled_gatew_shape == (int(tiled_x_shape[2]), int(canonical_shapes.get("I", tiled_gatew_shape[1] if len(tiled_gatew_shape) == 2 else 0)))
        and tiled_upw_shape == tiled_gatew_shape
        and len(tiled_downw_shape) == 2
        and len(tiled_y_shape) == 3
        and tiled_y_shape[-1] == tiled_x_shape[-1]
    ):
        repaired = parse_candidate_json(
            _tiled_mlp_repair_json(
                descriptor,
                b_dim=int(tiled_x_shape[0]),
                s_dim=int(tiled_x_shape[1]),
                h_dim=int(tiled_x_shape[2]),
                i_dim=int(tiled_gatew_shape[1]),
            )
        )
        repairs.append("tiled_mlp_repair_v1")
        return repaired, repairs
    if len(mta_scores_shape) == 4 and len(mta_weight_shape) == 4 and len(mta_bias_shape) == 1 and len(mta_y_shape) == 4:
        groups = max(1, int(mta_scores_shape[1] // max(1, mta_weight_shape[1])))
        repaired = parse_candidate_json(
            _multi_token_attention_repair_json(
                descriptor,
                b_dim=int(mta_scores_shape[0]),
                cin_dim=int(mta_scores_shape[1]),
                cout_dim=int(mta_weight_shape[0]),
                l_dim=int(mta_scores_shape[2]),
                k_dim=int(mta_weight_shape[2]),
                groups=int(groups),
            )
        )
        repairs.append("multi_token_attention_repair_v1")
        return repaired, repairs

    return None, repairs


def _repair_candidate_from_descriptor(cand: CandidateIntent, descriptor: KernelDescriptor) -> tuple[CandidateIntent, list[str]]:
    repaired, repairs = prefill_candidate_for_descriptor(descriptor)
    if repaired is not None:
        return repaired, repairs
    return cand, repairs


def repair_candidate_for_descriptor(cand: CandidateIntent, descriptor: KernelDescriptor) -> CandidateIntent:
    """
    Re-apply descriptor-aware repairs for cached seeds and replayed candidates.

    This keeps old seeds from bypassing newer scalar/view normalization logic.
    """
    repaired, repair_tags = _repair_candidate_from_descriptor(cand, descriptor)
    if repaired is cand:
        return cand
    repaired.llm_trace = dict(cand.llm_trace or {})
    if repair_tags:
        repaired.llm_trace.setdefault("repairs", []).extend(list(repair_tags))  # type: ignore[call-arg]
    try:
        extra_repairs = repair_missing_outputs(repaired.intent)
        if extra_repairs:
            repaired.llm_trace.setdefault("repairs", []).extend(list(extra_repairs))  # type: ignore[call-arg]
    except Exception:
        pass
    return repaired


@dataclass
class LLMIntentHub:
    default_model: str = DEFAULT_MODEL
    timeout_s: int = 600
    http_max_retries: int = 4
    http_max_total_wait_s: int = 180
    max_parse_retries: int = 2
    max_attempts: int = 2
    extra_chat_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Provider health state:
    # - Quota exhaustion -> hard disable for this process (until=+inf).
    # - Transient 5xx/proxy issues -> short cooldown (until=now+cooldown_s),
    #   only after repeated failures to avoid flaking out a generally-working provider.
    disabled_models: Dict[str, float] = field(default_factory=dict)  # model -> disabled_until (epoch seconds)
    model_fail_streak: Dict[str, int] = field(default_factory=dict)
    server_error_disable_after: int = 2
    server_error_cooldown_s: int = 180
    # When True, try multiple configured provider/model candidates (in order).
    # For paper experiments, it can be useful to disable fallback to measure
    # raw reliability/cost of a single provider.
    allow_model_fallback: bool = True

    def _maybe_disable_model(self, model: str, err: Exception) -> None:
        """
        Disable a provider/model for the lifetime of this process when we detect
        hard failures (quota exhausted, repeated 5xx), so large suites don't get
        stuck retrying a dead endpoint.
        """
        m = str(model)
        msg = str(err)
        now = time.time()
        # Quota/credit exhaustion: these won't recover without user action.
        hard_markers = [
            "pre_consume_token_quota_failed",
            "insufficient_quota",
            "quota",
            "余额",
            "令牌总使用次数已达到限制",
        ]
        if any(x in msg for x in hard_markers):
            self.disabled_models[m] = float("inf")
            return
        # Transient 5xx from a proxy is often recoverable; only disable after a
        # short streak to avoid one-off flakiness making the suite brittle.
        if "server error" in msg or " 520 " in msg or " 502 " in msg or " 503 " in msg or " 504 " in msg:
            streak = int(self.model_fail_streak.get(m, 0)) + 1
            self.model_fail_streak[m] = streak
            # If the caller disables fallback, disabling the only candidate will
            # cause the rest of a suite to "skip: disabled" without actually
            # exercising the provider. For paper-grade cold-runs we prefer to
            # keep trying on later kernels (bounded by retries/timeout/rpm).
            if bool(self.allow_model_fallback) and streak >= max(1, int(self.server_error_disable_after)):
                self.disabled_models[m] = now + float(max(1, int(self.server_error_cooldown_s)))
            return

    def _is_model_disabled(self, model: str) -> bool:
        m = str(model)
        until = self.disabled_models.get(m)
        if until is None:
            return False
        if until == float("inf"):
            return True
        now = time.time()
        if until > now:
            return True
        # cooldown expired
        try:
            del self.disabled_models[m]
        except KeyError:
            pass
        return False

    def lift(self, descriptor: KernelDescriptor, *, feedback: Optional[List[str]] = None, model: Optional[str] = None) -> CandidateIntent:
        """
        Produce a CandidateIntent from a KernelDescriptor.

        Retries are limited (max_attempts) to respect provider rate limits.
        """
        fb = [str(x) for x in (feedback or []) if str(x).strip()]
        last_err: Exception | None = None
        for attempt in range(max(1, int(self.max_attempts))):
            messages = self._build_messages(descriptor, feedback=fb, attempt=attempt, last_error=last_err)
            prompt_hash = _hash_messages(messages)
            requested = model or self.default_model
            extra = dict(self.extra_chat_kwargs)
            # Complex kernels (e.g., attention with masks) can exceed 1600 tokens.
            # Truncation often manifests as invalid JSON; prefer a larger cap.
            extra.setdefault("max_tokens", 4096)
            # Reduce non-determinism; helps providers obey "JSON only" prompts.
            extra.setdefault("temperature", 0)

            trace: Dict[str, Any] = {
                "requested_model": requested,
                "candidates": (list(candidate_models(requested)) if bool(self.allow_model_fallback) else [requested]),
                "attempts": [],
            }

            for m in trace["candidates"]:
                if self._is_model_disabled(m):
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": False, "stage": "skip", "error": "disabled"})
                    continue
                try:
                    resp = chat_completion(
                        messages,
                        model=m,
                        stream=False,
                        allow_fallback=False,
                        timeout=int(self.timeout_s),
                        max_retries=int(self.http_max_retries),
                        max_total_wait_s=int(self.http_max_total_wait_s),
                        **extra,
                    )
                except LLMClientError as e:
                    last_err = e
                    self._maybe_disable_model(m, e)
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": False, "stage": "http", "error": str(e)})
                    continue

                raw_text = resp.first_message()
                cache_hit = bool(resp.meta.get("cache_hit"))
                try:
                    js = parse_json_block(raw_text)
                except Exception as e:
                    last_err = e
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": cache_hit, "stage": "json", "error": str(e)})
                    continue

                try:
                    cand = parse_candidate_json(js)
                except (LLMJsonParseError, IntentIRValidationError) as e:
                    # Semantic parse failed; try the next provider/model candidate
                    # instead of retrying the same broken completion.
                    # If the response came from the on-disk cache, it can lock us
                    # into a permanently-bad completion. Bust that cache entry once
                    # and re-fetch for the same model.
                    if cache_hit:
                        cache_path = resp.meta.get("cache_path")
                        if isinstance(cache_path, str) and cache_path:
                            try:
                                Path(cache_path).unlink(missing_ok=True)
                                resp2 = chat_completion(
                                    messages,
                                    model=m,
                                    stream=False,
                                    allow_fallback=False,
                                    timeout=int(self.timeout_s),
                                    max_retries=int(self.http_max_retries),
                                    max_total_wait_s=int(self.http_max_total_wait_s),
                                    **extra,
                                )
                                raw2 = resp2.first_message()
                                js2 = parse_json_block(raw2)
                                cand2 = parse_candidate_json(js2)
                                cache_hit2 = bool(resp2.meta.get("cache_hit"))
                                trace["ok"] = True
                                trace["chosen"] = {
                                    "model": resp2.meta.get("response_model") or resp2.meta.get("model") or m,
                                    "base_url": resp2.meta.get("base_url"),
                                    "cache_hit": cache_hit2,
                                }
                                trace["attempts"].append(
                                    {"model": m, "ok": True, "cache_hit": cache_hit2, "stage": "semantic", "note": "cache_bust_retry"}
                                )
                                cand2.llm_trace = {
                                    "prompt_hash": prompt_hash,
                                    "frontend": descriptor.frontend,
                                    "kernel": descriptor.name,
                                    "extract_trace": trace,
                                }
                                cand2, repair_tags = _repair_candidate_from_descriptor(cand2, descriptor)
                                if repair_tags:
                                    cand2.llm_trace.setdefault("repairs", []).extend(list(repair_tags))  # type: ignore[call-arg]
                                return cand2
                            except Exception:
                                pass
                    last_err = e
                    trace["attempts"].append({"model": m, "ok": False, "cache_hit": cache_hit, "stage": "semantic", "error": str(e)})
                    continue

                trace["ok"] = True
                trace["chosen"] = {
                    "model": resp.meta.get("response_model") or resp.meta.get("model") or m,
                    "base_url": resp.meta.get("base_url"),
                    "cache_hit": cache_hit,
                }
                trace["attempts"].append({"model": m, "ok": True, "cache_hit": cache_hit, "stage": "semantic"})
                # Reset transient failure streak on success.
                try:
                    self.model_fail_streak.pop(str(m), None)
                except Exception:
                    pass
                cand.llm_trace = {
                    "prompt_hash": prompt_hash,
                    "frontend": descriptor.frontend,
                    "kernel": descriptor.name,
                    "extract_trace": trace,
                }
                cand, repair_tags = _repair_candidate_from_descriptor(cand, descriptor)
                if repair_tags:
                    cand.llm_trace.setdefault("repairs", []).extend(list(repair_tags))  # type: ignore[call-arg]
                try:
                    repairs = repair_missing_outputs(cand.intent)
                    if repairs:
                        cand.llm_trace.setdefault("repairs", list(repairs))  # type: ignore[call-arg]
                except Exception:
                    # Repairs are best-effort; do not fail extraction on them.
                    pass
                return cand

            # If all model candidates failed, append the last error as feedback and retry.
            if trace.get("attempts"):
                # Preserve multi-provider failure context: by default we would only
                # raise the *last* LLMClientError, losing earlier provider errors.
                # This aggregated message is safe (no API keys) and makes regressions
                # debuggable without rerunning with verbose logs.
                try:
                    attempts = trace.get("attempts") or []
                    errs: List[str] = []
                    for a in attempts:
                        if not isinstance(a, dict) or a.get("ok") is True:
                            continue
                        m = a.get("model")
                        st = a.get("stage")
                        er = a.get("error")
                        if isinstance(m, str) and isinstance(st, str) and isinstance(er, str) and er.strip():
                            errs.append(f"{m}[{st}]: {er}")
                    if errs:
                        # Keep the exception string compact but informative.
                        head = errs[:6]
                        tail = f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""
                        last_err = LLMClientError("all candidates failed: " + " | ".join(head) + tail)
                        # Attach the per-attempt trace so callers (e.g., E3 regression)
                        # can report accurate cache/API usage and failure breakdown.
                        try:
                            setattr(last_err, "intentir_trace", trace)
                            setattr(last_err, "intentir_prompt_hash", prompt_hash)
                            setattr(last_err, "intentir_frontend", descriptor.frontend)
                            setattr(last_err, "intentir_kernel", descriptor.name)
                            setattr(last_err, "intentir_attempt", int(attempt))
                        except Exception:
                            pass
                except Exception:
                    pass
            if last_err is not None:
                fb = fb or []
                fb = fb + [f"Previous failure: {type(last_err).__name__}: {last_err}"]
            continue
        raise last_err or RuntimeError("LLMIntentHub.lift failed without exception")

    def _build_messages(
        self,
        descriptor: KernelDescriptor,
        *,
        feedback: List[str],
        attempt: int,
        last_error: Exception | None,
    ) -> List[Dict[str, str]]:
        evidence = _evidence_blob(descriptor)
        extra_lines: List[str] = [
            "Evidence appendix (JSON):",
            evidence,
            "",
            "Use the evidence to align output tensors, masks, and reduce axes; do not copy TTIR lines verbatim.",
        ]
        if feedback:
            extra_lines += ["", "Feedback from previous failures:", *[f"- {x}" for x in feedback]]
        if attempt > 0 and last_error is not None:
            extra_lines += ["", f"Retry attempt={attempt} after error: {type(last_error).__name__}: {last_error}"]
        extra_instruction = "\n".join(extra_lines).strip()

        src = _maybe_truncate_source(descriptor.source_text)
        compact = bool(src.startswith("[IntentIR] SOURCE TRUNCATED"))
        if attempt > 0 and last_error is not None:
            # If the provider is flaky/limited, retry with a smaller source and
            # a more compact system prompt.
            src2 = _maybe_compact_source_on_server_error(descriptor.source_text, last_error)
            if src2 != descriptor.source_text:
                src = src2
                compact = True
        if descriptor.frontend == "triton":
            from frontends.triton.llm_intent import build_messages

            return build_messages(src, kernel_name=descriptor.name, extra_instruction=extra_instruction, compact=compact)
        if descriptor.frontend == "tilelang":
            from frontends.tilelang.llm_intent import build_messages

            return build_messages(src, kernel_name=descriptor.name, extra_instruction=extra_instruction, compact=compact)
        if descriptor.frontend == "cuda":
            from frontends.cuda.llm_intent import build_messages

            return build_messages(src, kernel_name=descriptor.name, extra_instruction=extra_instruction, compact=compact)
        raise NotImplementedError(f"LLMIntentHub does not support frontend={descriptor.frontend}")


__all__ = ["LLMIntentHub", "repair_candidate_for_descriptor", "prefill_candidate_for_descriptor"]
