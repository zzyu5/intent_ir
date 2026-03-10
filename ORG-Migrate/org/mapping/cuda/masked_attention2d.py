from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendPlan
from org.mapping.cuda.module_catalog import masked_attention2d_catalog
from org.mapping.hardware_model import HardwareModel
from org.schema import OrgDoc


def _coerce_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _require_dim(bindings: Mapping[str, Any], key: str) -> int:
    value = _coerce_int(bindings.get(key))
    if value is None:
        raise ValueError(f"missing shape_bindings[{key!r}]")
    return int(value)


def _goal_tags(org: OrgDoc) -> set[str]:
    return {str(goal.tag) for goal in list(getattr(org, "goals", []) or []) if str(getattr(goal, "tag", "")).strip()}


def _mechanism_tags(org: OrgDoc) -> set[str]:
    return {
        str(mechanism.tag)
        for mechanism in list(getattr(org, "mechanisms", []) or [])
        if str(getattr(mechanism, "tag", "")).strip()
    }


def _fact_present(facts: Mapping[str, Any] | None, key: str) -> bool:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    return bool(dict(mechanisms.get(str(key)) or {}).get("present"))


def _sm_number(sm: Any) -> int:
    text = str(sm or "").strip().lower()
    if text.startswith("sm_"):
        text = text[3:]
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _shared_stage_enabled(
    *,
    goal_tags: set[str],
    mechanism_tags: set[str],
    ttgir_facts: Mapping[str, Any] | None,
    ptx_facts: Mapping[str, Any] | None,
    hardware_model: HardwareModel,
    toolchain_model: Mapping[str, Any] | None,
) -> bool:
    if "resident_working_set" not in goal_tags:
        return False
    if "streaming_softmax_state" not in goal_tags:
        return False
    if not (
        "kv_tile_load" in mechanism_tags
        or _fact_present(ttgir_facts, "staging.kv_streamed_tiles")
        or _fact_present(ttgir_facts, "staging.local_or_shared")
    ):
        return False
    if not (
        "mask_causal_apply" in mechanism_tags
        or _fact_present(ttgir_facts, "communication.mask_causal")
    ):
        return False
    if not (
        "online_softmax_reduce" in mechanism_tags
        or _fact_present(ttgir_facts, "communication.streaming_softmax")
        or _fact_present(ttgir_facts, "communication.reduction")
    ):
        return False
    if _sm_number((toolchain_model or {}).get("effective_sm")) < 120:
        return False
    if bool((toolchain_model or {}).get("downleveled")):
        return False
    if int(getattr(hardware_model, "shared_mem_kb", 0) or 0) < 64:
        return False
    if not _fact_present(ptx_facts, "communication.shuffle"):
        return False
    return True


def plan_masked_attention2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    toolchain_model: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    b = max(1, int(budget))
    q_ctx = _require_dim(shape_bindings, "Q_CTX")
    kv_ctx = _require_dim(shape_bindings, "KV_CTX")
    hd = _require_dim(shape_bindings, "HEAD_DIM")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    cluster = str(hardware_model.arch_cluster)
    modules, edges, _passes = masked_attention2d_catalog(hardware_model)
    selected_ids = {
        "masked_attn_q_resident_state",
        "masked_attn_tiny_kv_stage",
        "masked_attn_mask_causal_apply",
        "masked_attn_parallel_softmax",
        "masked_attn_vector_dot_fragment",
        "masked_attn_backend_v18",
        "masked_attn_backend_v14",
        "masked_attn_backend_v10",
    }
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in edges if e.src in selected_ids and e.dst in selected_ids]
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    shared_stage = _shared_stage_enabled(
        goal_tags=goal_tags,
        mechanism_tags=mechanism_tags,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        hardware_model=hardware_model,
        toolchain_model=toolchain_model,
    )

    base: list[BackendCandidate] = []
    if q_ctx == 16 and kv_ctx == 16 and hd == 16:
        base.extend(
            [
                BackendCandidate(
                    kernel_kind="attn2d_causal_softmax_v18",
                    bindings=(
                        {"ATTN_SCORE_WARPS": 2, "MASKED_ATTN_SHARED_STAGE": 1, "MASKED_ATTN_VECTOR_WIDTH": 4}
                        if shared_stage
                        else {"ATTN_SCORE_WARPS": 4}
                    ),
                ),
                BackendCandidate(
                    kernel_kind="attn2d_causal_softmax_v18",
                    bindings=(
                        {"ATTN_SCORE_WARPS": 4, "MASKED_ATTN_SHARED_STAGE": 1, "MASKED_ATTN_VECTOR_WIDTH": 4}
                        if shared_stage
                        else {}
                    ),
                ),
                BackendCandidate(kernel_kind="attn2d_causal_softmax_v10", bindings={}),
                BackendCandidate(kernel_kind="attn2d_causal_softmax_v14", bindings={}),
            ]
        )

    ranked: list[BackendCandidate] = []
    for candidate in base:
        kind = str(candidate.kernel_kind)
        bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
        score = 100.0
        reasons: list[str] = [f"cluster={cluster}", f"shape={q_ctx}x{kv_ctx}x{hd}", f"kind={kind}"]
        if kind == "attn2d_causal_softmax_v18":
            score += 60.0
            reasons.append("parallel_softmax")
            if shared_stage:
                score += 18.0
                reasons.append("shared_stage")
                warps = int(bindings.get("ATTN_SCORE_WARPS") or 4)
                if warps == 2:
                    score -= 8.0
                    reasons.append("tiny_kv:2warp")
                elif warps == 4:
                    score += 10.0
                    reasons.append("tiny_kv:4warp")
        elif kind == "attn2d_causal_softmax_v10":
            score += 40.0
            reasons.append("vector_dot_fragment")
        else:
            score += 24.0
            reasons.append("warp_masked_softmax")
        if "mask_causal_pruning" in goal_tags:
            score += 10.0
            reasons.append("goal:mask_causal")
        if "streaming_softmax_state" in goal_tags:
            score += 8.0
            reasons.append("goal:softmax_state")
        if "avoid_materialization" in goal_tags and kind == "attn2d_causal_softmax_v18":
            score += 8.0
            reasons.append("goal:parallel_reduce")
        if "parallel_softmax" in mechanism_tags and kind == "attn2d_causal_softmax_v18":
            score += 12.0
            reasons.append("mechanism:parallel_softmax")
        if "vector_dot_fragment" in mechanism_tags and kind == "attn2d_causal_softmax_v10":
            score += 8.0
            reasons.append("mechanism:vector_dot")
        if source_kind == kind:
            score += 6.0
            reasons.append("source_exact")
        ranked.append(
            BackendCandidate(
                kernel_kind=kind,
                bindings=dict(candidate.bindings or {}),
                note="masked_attention2d",
                score=float(score),
                score_reason=",".join(reasons),
                cluster=cluster,
                portability_note="portable",
            )
        )
    ranked.sort(key=lambda c: (-float(c.score or 0.0), str(c.kernel_kind)))
    ranked = ranked[:b]
    return BackendPlan(
        kernel="masked_attention2d",
        source_oracle=dict(source_oracle or {}),
        hardware_model=hardware_model.to_json_dict(),
        selected_modules=selected_modules,
        module_edges=selected_edges,
        param_space={
            "kernel_kind": [str(c.kernel_kind) for c in ranked],
            "ATTN_SCORE_WARPS": ([2, 4] if shared_stage else [4]),
            "MASKED_ATTN_SHARED_STAGE": ([1] if shared_stage else []),
            "MASKED_ATTN_VECTOR_WIDTH": ([4] if shared_stage else []),
        },
        constraints=["Q_CTX == 16", "KV_CTX == 16", "HEAD_DIM == 16"],
        substitutions=[],
        candidates=ranked,
        notes=[f"goals={sorted(goal_tags)}", f"cluster={cluster}", f"shared_stage={shared_stage}"],
    )


__all__ = ["plan_masked_attention2d"]
