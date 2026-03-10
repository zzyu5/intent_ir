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
from typing import Any, Dict, List, Optional

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


def _rope_repair_json(descriptor: KernelDescriptor, *, q_shape: tuple[int, ...], k_shape: tuple[int, ...], cos_shape: tuple[int, ...]) -> dict[str, Any]:
    b_dim, qh_dim, s_dim, hd_dim = map(int, q_shape)
    _bk, kh_dim, _sk, _hk = map(int, k_shape)
    cos_batch = int(cos_shape[0]) if len(cos_shape) == 3 else 1
    cos_width = int(cos_shape[-1]) if cos_shape else hd_dim
    cos_b_dim: str | int = int(cos_batch) if cos_batch != b_dim else "B"
    logical_layout = {"kind": "custom", "params": {"axes": ["B", "H", "S", "HD"]}}
    physical_layout = {"kind": "custom", "params": {"axes": ["B", "S", "H", "HD"], "view_perm": [0, 2, 1, 3]}}
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
        },
    }


def _repair_candidate_from_descriptor(cand: CandidateIntent, descriptor: KernelDescriptor) -> tuple[CandidateIntent, list[str]]:
    repairs: list[str] = []
    shapes = _baseline_array_shapes(descriptor)
    name = str(descriptor.name or "").strip().lower()
    module = str((descriptor.launch or {}).get("module") or "").strip().lower()

    if "rope" in name and "q" in shapes and "k" in shapes and "cos" in shapes and "sin" in shapes:
        repaired = parse_candidate_json(
            _rope_repair_json(
                descriptor,
                q_shape=tuple(shapes["q"]),
                k_shape=tuple(shapes["k"]),
                cos_shape=tuple(shapes["cos"]),
            )
        )
        repairs.append("liger_rope_view_repair_v1")
        return repaired, repairs

    if ("cross_entropy" in name or "cross_entropy" in module) and "input" in shapes and "target" in shapes and "loss" in shapes:
        repaired = parse_candidate_json(
            _cross_entropy_repair_json(
                descriptor,
                input_shape=tuple(shapes["input"]),
            )
        )
        repairs.append("liger_cross_entropy_loss_repair_v1")
        return repaired, repairs

    return cand, repairs


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


__all__ = ["LLMIntentHub"]
