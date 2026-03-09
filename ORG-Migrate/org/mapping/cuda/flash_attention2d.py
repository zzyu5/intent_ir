from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import flash_attention2d_catalog
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
    return {str(mechanism.tag) for mechanism in list(getattr(org, "mechanisms", []) or []) if str(getattr(mechanism, "tag", "")).strip()}


def _ordered_unique(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        iv = int(value)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def _ordered_param_values(defaults: list[int], preferred: int | None, allowed: list[int]) -> list[int]:
    vals = [int(x) for x in defaults]
    if allowed:
        allowed_set = {int(x) for x in allowed}
        vals = [int(x) for x in vals if int(x) in allowed_set]
    if preferred is not None and int(preferred) not in vals and (not allowed or int(preferred) in set(allowed)):
        vals.append(int(preferred))
    return _ordered_unique(vals)


def _merged_param_values(defaults: list[int], preferred: int | None, allowed: list[int]) -> list[int]:
    vals: list[int] = []
    if preferred is not None:
        vals.append(int(preferred))
    vals.extend(int(x) for x in defaults)
    vals.extend(int(x) for x in list(allowed or []))
    return _ordered_unique(vals)


def _async_copy_guardrails(*, kv_ctx: int, head_dim: int, block_kv: int, score_warps: int) -> tuple[bool, str]:
    if kv_ctx != block_kv:
        return False, "KV_CTX != ATTN_BLOCK_KV"
    if (head_dim % 4) != 0:
        return False, "HEAD_DIM % 4 != 0"
    threads = (2 + int(score_warps)) * 32
    tile_vec4 = (int(block_kv) * int(head_dim)) // 4
    if threads <= 0 or tile_vec4 <= 0:
        return False, "invalid tile/threads"
    if (tile_vec4 % int(threads)) != 0:
        return False, "tile_vec4 % threads != 0"
    return True, ""


def _selected_modules(
    *,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    goal_tags: set[str],
    mechanism_tags: set[str],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
) -> tuple[list[BackendModule], list[BackendModuleEdge], list[dict[str, Any]]]:
    selected_ids = {"q_resident_state", "kv_tile_stage", "output_accumulator"}
    substitutions: list[dict[str, Any]] = []

    if "streaming_softmax_state" in goal_tags or "online_softmax_reduce" in mechanism_tags:
        selected_ids.add("online_softmax_reduce")
    if "latency_hiding" in goal_tags or "prefetch_pipeline" in mechanism_tags:
        if hardware_model.supports_async_copy:
            selected_ids.add("prefetch_pipeline")
        else:
            substitutions.append(
                {
                    "from": "flash.prefetch_pipeline",
                    "to": "flash.sync_prefetch",
                    "reason": "hardware_model.supports_async_copy = false",
                }
            )

    selected_ids.update({"backend_v6", "backend_v7"})
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    if source_kind == "attn2d_causal_softmax_v7":
        substitutions.append(
            {
                "from": "source.variant.preference",
                "to": "cluster_ranked_variants",
                "reason": f"variant ranking follows {hardware_model.arch_cluster}",
            }
        )
    return selected_modules, selected_edges, substitutions


def _candidate_key(candidate: BackendCandidate) -> tuple[str, tuple[tuple[str, int], ...]]:
    return str(candidate.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(candidate.bindings or {}).items()))


def _fact_present(facts: Mapping[str, Any] | None, key: str) -> bool:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    return bool(dict(mechanisms.get(str(key)) or {}).get("present"))


def _fact_attr(facts: Mapping[str, Any] | None, key: str, attr: str, default: Any = None) -> Any:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    attrs = dict(dict(mechanisms.get(str(key)) or {}).get("attrs") or {})
    return attrs.get(str(attr), default)


def _sm_number(sm: str | None) -> int:
    raw = str(sm or "").strip().lower()
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        return int(digits)
    except Exception:
        return 0


def _flash_resident_bytes_hint(*, block_kv: int, head_dim: int, ttgir_facts: Mapping[str, Any] | None) -> int:
    q_bytes = int(_fact_attr(ttgir_facts, "staging.q_resident_state", "resident_bytes_hint", head_dim * 4) or (head_dim * 4))
    kv_bytes = int(_fact_attr(ttgir_facts, "staging.kv_streamed_tiles", "resident_bytes_hint", (block_kv * head_dim * 4 * 2)) or (block_kv * head_dim * 4 * 2))
    output_bytes = int(head_dim * 4)
    return int(q_bytes + kv_bytes + output_bytes)


def _flash_thread_count_hint(*, kind: str, score_warps: int, hardware_model: HardwareModel) -> int:
    warp = max(1, int(hardware_model.warp_size))
    if kind == "attn2d_causal_softmax_v6":
        return int((2 + int(score_warps)) * warp)
    if kind == "attn2d_causal_softmax_v8":
        return int(4 * warp)
    if kind == "attn2d_causal_softmax_v9":
        return int(6 * warp)
    if kind == "attn2d_causal_softmax_v7":
        return int(4 * warp)
    return int(4 * warp)


def _flash_register_pressure_hint(*, kind: str, block_kv: int, score_warps: int) -> int:
    base = {
        "attn2d_causal_softmax_v6": 52,
        "attn2d_causal_softmax_v7": 58,
        "attn2d_causal_softmax_v8": 54,
        "attn2d_causal_softmax_v9": 62,
    }.get(str(kind), 56)
    base += {16: 0, 32: 4, 64: 10}.get(int(block_kv), 12)
    if kind == "attn2d_causal_softmax_v6":
        base += int(score_warps) * 2
    if kind == "attn2d_causal_softmax_v9":
        base += 4
    return int(base)


def _flash_resource_pressure(
    *,
    kind: str,
    block_kv: int,
    score_warps: int,
    resident_bytes: int,
    hardware_model: HardwareModel,
) -> tuple[int, float, float]:
    threads = _flash_thread_count_hint(kind=kind, score_warps=score_warps, hardware_model=hardware_model)
    shared_budget = max(1, int(hardware_model.shared_mem_kb) * 1024)
    resident_ratio = float(resident_bytes) / float(shared_budget)
    reg_hint = _flash_register_pressure_hint(kind=kind, block_kv=block_kv, score_warps=score_warps)
    register_ratio = float(int(threads) * int(reg_hint)) / float(max(1, int(hardware_model.register_budget)))
    return int(threads), float(resident_ratio), float(register_ratio)


def _complete_async_evidence(
    *,
    ttgir_facts: Mapping[str, Any] | None,
    ptx_facts: Mapping[str, Any] | None,
) -> bool:
    return bool(
        _fact_present(ttgir_facts, "pipeline.stage_hint")
        and _fact_present(ttgir_facts, "staging.kv_streamed_tiles")
        and _fact_present(ttgir_facts, "staging.q_resident_state")
        and bool(_fact_attr(ptx_facts, "pipeline.async_copy", "complete_async_pipeline", False))
    )


def _score_flash_candidate(
    *,
    candidate: BackendCandidate,
    goal_tags: set[str],
    cluster: str,
    source_oracle: Mapping[str, Any],
    kv_ctx: int,
    head_dim: int,
    ttgir_facts: Mapping[str, Any] | None,
    async_evidence_ok: bool,
    toolchain_model: Mapping[str, Any] | None,
    hardware_model: HardwareModel,
) -> tuple[float, str, str]:
    kind = str(candidate.kernel_kind)
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    block_kv = int(bindings.get("ATTN_BLOCK_KV", 16))
    score_warps = int(bindings.get("ATTN_SCORE_WARPS", 0))
    is_async = bool(bindings.get("FLASH_ATTN_ASYNC_COPY", 0))
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    resident_bytes = _flash_resident_bytes_hint(block_kv=block_kv, head_dim=head_dim, ttgir_facts=ttgir_facts)
    residency_complete = bool(_fact_present(ttgir_facts, "staging.q_resident_state") and _fact_present(ttgir_facts, "staging.kv_streamed_tiles"))
    threads_hint, resident_ratio, register_ratio = _flash_resource_pressure(
        kind=kind,
        block_kv=block_kv,
        score_warps=score_warps,
        resident_bytes=resident_bytes,
        hardware_model=hardware_model,
    )
    effective_sm = _sm_number((toolchain_model or {}).get("effective_sm"))
    downleveled = bool((toolchain_model or {}).get("downleveled"))
    v7_front_allowed = bool(
        "avoid_materialization" in goal_tags
        and "streaming_softmax_state" in goal_tags
        and async_evidence_ok
    )

    score = 0.0
    reasons: list[str] = [
        f"cluster={cluster}",
        f"resident_bytes={resident_bytes}",
        f"threads_hint={threads_hint}",
        f"resident_ratio={resident_ratio:.3f}",
        f"register_ratio={register_ratio:.3f}",
        f"effective_sm={effective_sm or 0}",
    ]
    portability_note = "portable"

    if cluster == "cuda_tc_mid_smem":
        if kind == "attn2d_causal_softmax_v6":
            score += 140.0
        elif kind == "attn2d_causal_softmax_v8":
            score += (132.0 if effective_sm >= 120 and not downleveled else 116.0)
        elif kind == "attn2d_causal_softmax_v9":
            score += (118.0 if effective_sm >= 120 and not downleveled else 72.0)
        else:
            score += 40.0
    elif cluster == "cuda_tc_large_smem":
        if kind == "attn2d_causal_softmax_v6":
            score += 120.0
        elif kind == "attn2d_causal_softmax_v8":
            score += 104.0
        elif kind == "attn2d_causal_softmax_v9":
            score += 108.0
        else:
            score += (126.0 if v7_front_allowed else 90.0)
    else:
        score += (
            100.0
            if kind == "attn2d_causal_softmax_v6"
            else (88.0 if kind == "attn2d_causal_softmax_v8" else (74.0 if kind == "attn2d_causal_softmax_v9" else 60.0))
        )

    score += {64: 30.0, 32: 20.0, 16: 10.0}.get(int(block_kv), 0.0)
    reasons.append(f"block_kv={block_kv}")
    if kind == "attn2d_causal_softmax_v6":
        score += {6: 15.0, 4: 10.0, 2: 2.0}.get(int(score_warps), 0.0)
        reasons.append(f"score_warps={score_warps}")
        if cluster == "cuda_tc_mid_smem":
            if effective_sm >= 120 and not downleveled:
                if int(block_kv) == 32 and int(score_warps) == 6:
                    score += 34.0
                    reasons.append("sm120_v6_tile32_w6_resource_fit")
                elif int(block_kv) == 32 and int(score_warps) == 4:
                    score += 24.0
                    reasons.append("sm120_v6_tile32_w4")
                elif int(block_kv) == 16 and int(score_warps) == 6:
                    score += 18.0
                    reasons.append("sm120_v6_tile16_w6")
            if residency_complete:
                if int(block_kv) == 64 and int(score_warps) == 4:
                    score += 18.0
                    reasons.append("mid_smem_balanced_resident_tile")
                elif int(block_kv) == 64 and int(score_warps) == 6:
                    score -= 10.0
                    reasons.append("mid_smem_overparallel_resident_tile")
                elif int(block_kv) == 32 and int(score_warps) == 6:
                    score -= 6.0
                    reasons.append("mid_smem_small_tile_overparallel")
            else:
                if int(score_warps) == 6 and int(block_kv) == 32:
                    score += 16.0
                    reasons.append("mid_smem_streaming_fit")
                elif int(score_warps) == 6 and int(block_kv) == 64:
                    score -= 4.0
                    reasons.append("mid_smem_large_tile_pressure")
    if kind == "attn2d_causal_softmax_v8":
        if cluster == "cuda_tc_mid_smem" and int(block_kv) == 32:
            if effective_sm >= 120 and not downleveled:
                if (
                    residency_complete
                    and resident_ratio <= 0.28
                    and register_ratio <= 0.16
                    and hardware_model.supports_async_copy
                    and hardware_model.compute_cluster == "tensor_core"
                    and hardware_model.pipeline_cluster == "async_pipeline"
                ):
                    score += 84.0
                    reasons.append("sm120_v8_tile32_resource_fit")
                else:
                    score += 36.0
                    reasons.append("sm120_v8_tile32_partial_fit")
            else:
                score += 18.0
                reasons.append("mid_smem_v8_tile32")
        elif cluster == "cuda_tc_mid_smem" and int(block_kv) == 64:
            score += 4.0
            reasons.append("mid_smem_v8_tile64")
    if kind == "attn2d_causal_softmax_v9":
        if effective_sm >= 120 and not downleveled:
            if int(block_kv) == 32:
                if resident_ratio <= 0.18 and register_ratio <= 0.24 and hardware_model.supports_async_copy:
                    score += 28.0
                    reasons.append("sm120_v9_tile32_partial_fit")
                else:
                    score += 8.0
                    portability_note = "register_pressure_high"
                    reasons.append("sm120_v9_tile32_register_heavy")
            elif int(block_kv) == 64:
                score += 8.0
                reasons.append("sm120_v9_tile64")
        else:
            score -= 20.0
            portability_note = "toolchain_prefers_v6_v8"
            reasons.append("v9_requires_sm120")
    if kind == "attn2d_causal_softmax_v7" and not v7_front_allowed:
        score -= 60.0
        portability_note = "cluster_prefers_v6"
        reasons.append("v7_front_disallowed")
    if "streaming_softmax_state" in goal_tags and kind == "attn2d_causal_softmax_v6":
        score += 8.0
        reasons.append("preserve:streaming_softmax_state")
    if "avoid_materialization" in goal_tags and kind == "attn2d_causal_softmax_v7" and v7_front_allowed:
        score += 12.0
        reasons.append("preserve:avoid_materialization")
    if resident_bytes > 0 and cluster == "cuda_tc_mid_smem":
        budget_bytes = int(128 * 1024 * 0.8)
        if resident_bytes > budget_bytes:
            score -= 80.0
            portability_note = "resident_bytes_over_budget"
            reasons.append("resident_bytes_over_budget")
        elif register_ratio > 0.26:
            score -= 20.0
            portability_note = "register_pressure_high"
            reasons.append("register_pressure_high")
        if effective_sm >= 120 and not downleveled and kind == "attn2d_causal_softmax_v6" and resident_ratio >= 0.24 and threads_hint >= 192:
            score -= 24.0
            reasons.append("sm120_mid_smem_thread_pressure")
    if source_kind == kind and {str(k): int(v) for k, v in source_bindings.items()} == bindings:
        source_bonus = 4.0
        if cluster == "cuda_tc_mid_smem" and kind == "attn2d_causal_softmax_v6":
            source_bonus = 12.0
        score += source_bonus
        reasons.append("source_exact")
    if effective_sm >= 120 and not downleveled and cluster == "cuda_tc_mid_smem" and kind == "attn2d_causal_softmax_v6":
        if int(block_kv) == 64 and int(score_warps) == 6:
            score -= 12.0
            reasons.append("sm120_v6_not_frontier")
    if is_async:
        score += (6.0 if cluster == "cuda_tc_mid_smem" else 14.0)
        reasons.append("async_pipeline")
    if kv_ctx == block_kv:
        score += 3.0
        reasons.append("full_kv_tile")
    return score, ",".join(reasons), portability_note


def plan_flash_attention2d(
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
    head_dim = _require_dim(shape_bindings, "HEAD_DIM")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)

    modules, module_edges, _passes = flash_attention2d_catalog(hardware_model)
    selected_modules, selected_edges, substitutions = _selected_modules(
        modules=modules,
        module_edges=module_edges,
        goal_tags=goal_tags,
        mechanism_tags=mechanism_tags,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
    )
    preserve_notes: list[str] = []

    if head_dim != 64 or q_ctx <= 0 or kv_ctx <= 0:
        substitutions.append(
            {
                "from": "flash_attention2d",
                "to": "backend.skip",
                "reason": f"unsupported dims: Q_CTX={q_ctx} KV_CTX={kv_ctx} HEAD_DIM={head_dim}",
            }
        )
        return BackendPlan(
            kernel="flash_attention2d",
            source_oracle=dict(source_oracle or {}),
            hardware_model=hardware_model.to_json_dict(),
            selected_modules=selected_modules,
            module_edges=selected_edges,
            param_space={"kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7", "attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"]},
            constraints=["HEAD_DIM == 64"],
            substitutions=substitutions,
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}", f"cluster={hardware_model.arch_cluster}"],
        )

    dim_candidates_norm = collect_dim_candidate_ints_normalized(org)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    if exact_kind := str(source_oracle.get("kernel_kind") or "").strip():
        if exact_kind == "attn2d_causal_softmax_v6":
            source_bindings.setdefault("ATTN_SCORE_WARPS", 6)
    block_candidates = _ordered_param_values(
        defaults=_merged_param_values(
            defaults=[64, 32, 16],
            preferred=_coerce_int(source_bindings.get("ATTN_BLOCK_KV")),
            allowed=union_dim_candidate_ints(dim_candidates_norm, "tile_kv", "ATTN_BLOCK_KV", "BLOCK_KV"),
        ),
        preferred=_coerce_int(source_bindings.get("ATTN_BLOCK_KV")),
        allowed=[],
    )
    score_candidates = _ordered_param_values(
        defaults=_merged_param_values(
            defaults=[6, 4, 2],
            preferred=_coerce_int(source_bindings.get("ATTN_SCORE_WARPS")),
            allowed=union_dim_candidate_ints(dim_candidates_norm, "score_warps", "ATTN_SCORE_WARPS", "SCORE_WARPS"),
        ),
        preferred=_coerce_int(source_bindings.get("ATTN_SCORE_WARPS")),
        allowed=[],
    )
    block_candidates = [int(x) for x in block_candidates if int(x) <= int(kv_ctx)]
    if not block_candidates:
        block_candidates = [16]
        substitutions.append(
            {
                "from": "tile_kv",
                "to": "default_block_kv",
                "reason": "no ORG/source candidate fits KV_CTX",
            }
        )

    cluster = str(hardware_model.arch_cluster)
    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    exact_bindings = dict(source_bindings)
    want_pipeline = "latency_hiding" in goal_tags or "prefetch_pipeline" in mechanism_tags
    if exact_kind:
        preserve_notes.append(f"source_oracle_variant={exact_kind}")
    if "online_softmax_reduce" in mechanism_tags or "streaming_softmax_state" in goal_tags:
        preserve_notes.append("preserve:online_softmax_reduce")
    if want_pipeline:
        preserve_notes.append("preserve:prefetch_pipeline")

    async_evidence_ok = _complete_async_evidence(ttgir_facts=ttgir_facts, ptx_facts=ptx_facts)
    effective_sm = _sm_number((toolchain_model or {}).get("effective_sm"))
    downleveled = bool((toolchain_model or {}).get("downleveled"))
    if want_pipeline and not async_evidence_ok:
        substitutions.append(
            {
                "from": "flash.prefetch_pipeline",
                "to": "flash.sync_prefetch",
                "reason": "incomplete async evidence",
            }
        )
        preserve_notes.append("replace:prefetch_pipeline->sync_prefetch")

    param_space = {
        "kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7", "attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"],
        "ATTN_BLOCK_KV": list(block_candidates),
        "ATTN_SCORE_WARPS": list(score_candidates),
        "FLASH_ATTN_ASYNC_COPY": ([1] if want_pipeline and hardware_model.supports_async_copy and async_evidence_ok else []),
    }
    constraints = [
        "HEAD_DIM == 64",
        "ATTN_BLOCK_KV <= KV_CTX",
        "ATTN_SCORE_WARPS in {2,4,6}",
        "resident_working_set preserved",
        "streaming_softmax_state preserved",
        "async requires complete pipeline evidence",
    ]

    scored: list[BackendCandidate] = []
    if exact_kind in {"attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7", "attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"}:
        score, score_reason, portability_note = _score_flash_candidate(
            candidate=BackendCandidate(kernel_kind=exact_kind, bindings=dict(exact_bindings)),
            goal_tags=goal_tags,
            cluster=cluster,
            source_oracle=source_oracle,
            kv_ctx=kv_ctx,
            head_dim=head_dim,
            ttgir_facts=ttgir_facts,
            async_evidence_ok=async_evidence_ok,
            toolchain_model=toolchain_model,
            hardware_model=hardware_model,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=exact_kind,
                bindings=dict(exact_bindings),
                note="source_exact",
                score=score,
                score_reason=score_reason,
                cluster=cluster,
                portability_note=portability_note,
            )
        )

    for bk in block_candidates:
        for sw in score_candidates:
            candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v6", bindings={"ATTN_BLOCK_KV": int(bk), "ATTN_SCORE_WARPS": int(sw)})
            score, score_reason, portability_note = _score_flash_candidate(
                candidate=candidate,
                goal_tags=goal_tags,
                cluster=cluster,
                source_oracle=source_oracle,
                kv_ctx=kv_ctx,
                head_dim=head_dim,
                ttgir_facts=ttgir_facts,
                async_evidence_ok=async_evidence_ok,
                toolchain_model=toolchain_model,
                hardware_model=hardware_model,
            )
            scored.append(
                BackendCandidate(
                    kernel_kind=candidate.kernel_kind,
                    bindings=dict(candidate.bindings),
                    note="cluster_rank",
                    score=score,
                    score_reason=score_reason,
                    cluster=cluster,
                    portability_note=portability_note,
                )
            )

    for bk in block_candidates:
        candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v7", bindings={"ATTN_BLOCK_KV": int(bk)})
        score, score_reason, portability_note = _score_flash_candidate(
            candidate=candidate,
            goal_tags=goal_tags,
            cluster=cluster,
            source_oracle=source_oracle,
            kv_ctx=kv_ctx,
            head_dim=head_dim,
            ttgir_facts=ttgir_facts,
            async_evidence_ok=async_evidence_ok,
            toolchain_model=toolchain_model,
            hardware_model=hardware_model,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=candidate.kernel_kind,
                bindings=dict(candidate.bindings),
                note="cluster_rank",
                score=score,
                score_reason=score_reason,
                cluster=cluster,
                portability_note=portability_note,
            )
        )

    for bk in block_candidates:
        candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v8", bindings={"ATTN_BLOCK_KV": int(bk)})
        score, score_reason, portability_note = _score_flash_candidate(
            candidate=candidate,
            goal_tags=goal_tags,
            cluster=cluster,
            source_oracle=source_oracle,
            kv_ctx=kv_ctx,
            head_dim=head_dim,
            ttgir_facts=ttgir_facts,
            async_evidence_ok=async_evidence_ok,
            toolchain_model=toolchain_model,
            hardware_model=hardware_model,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=candidate.kernel_kind,
                bindings=dict(candidate.bindings),
                note="cluster_rank",
                score=score,
                score_reason=score_reason,
                cluster=cluster,
                portability_note=portability_note,
            )
        )

    if effective_sm >= 120 and not downleveled:
        for bk in block_candidates:
            candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v9", bindings={"ATTN_BLOCK_KV": int(bk)})
            score, score_reason, portability_note = _score_flash_candidate(
                candidate=candidate,
                goal_tags=goal_tags,
                cluster=cluster,
                source_oracle=source_oracle,
                kv_ctx=kv_ctx,
                head_dim=head_dim,
                ttgir_facts=ttgir_facts,
                async_evidence_ok=async_evidence_ok,
                toolchain_model=toolchain_model,
                hardware_model=hardware_model,
            )
            scored.append(
                BackendCandidate(
                    kernel_kind=candidate.kernel_kind,
                    bindings=dict(candidate.bindings),
                    note="toolchain_frontier",
                    score=score,
                    score_reason=score_reason,
                    cluster=cluster,
                    portability_note=portability_note,
                )
            )

    if want_pipeline and hardware_model.supports_async_copy and async_evidence_ok:
        preferred_score = score_candidates[0]
        for bk in block_candidates:
            ok, reason = _async_copy_guardrails(kv_ctx=kv_ctx, head_dim=head_dim, block_kv=int(bk), score_warps=int(preferred_score))
            if not ok:
                substitutions.append(
                    {
                        "from": "flash.prefetch_pipeline",
                        "to": "flash.sync_prefetch",
                        "reason": reason,
                        "detail": {"ATTN_BLOCK_KV": int(bk), "ATTN_SCORE_WARPS": int(preferred_score)},
                    }
                )
                continue
            candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v7", bindings={"ATTN_BLOCK_KV": int(bk), "FLASH_ATTN_ASYNC_COPY": 1})
            score, score_reason, portability_note = _score_flash_candidate(
                candidate=candidate,
                goal_tags=goal_tags,
                cluster=cluster,
                source_oracle=source_oracle,
                kv_ctx=kv_ctx,
                head_dim=head_dim,
                ttgir_facts=ttgir_facts,
                async_evidence_ok=async_evidence_ok,
                toolchain_model=toolchain_model,
                hardware_model=hardware_model,
            )
            scored.append(
                BackendCandidate(
                    kernel_kind=candidate.kernel_kind,
                    bindings=dict(candidate.bindings),
                    note="latency_hiding_async",
                    score=score,
                    score_reason=score_reason,
                    cluster=cluster,
                    portability_note=portability_note,
                )
            )

    if (source_bindings.get("FLASH_ATTN_ASYNC_COPY") or 0) == 1 and not any(c.bindings.get("FLASH_ATTN_ASYNC_COPY") == 1 for c in scored):
        substitutions.append(
            {
                "from": "source.prefetch_pipeline",
                "to": "flash.sync_prefetch",
                "reason": "source async-copy candidate has no valid target realization",
            }
        )

    final: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    ordered = sorted(
        scored,
        key=lambda c: (
            -float(c.score if c.score is not None else 0.0),
            0 if c.kernel_kind == "attn2d_causal_softmax_v6" else 1,
            -int(c.bindings.get("ATTN_BLOCK_KV", 0)),
            -int(c.bindings.get("ATTN_SCORE_WARPS", 0)),
        ),
    )
    for candidate in ordered:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        final.append(candidate)
        if len(final) >= b:
            break

    if cluster == "cuda_tc_mid_smem" and not any(c.kernel_kind == "attn2d_causal_softmax_v8" for c in final):
        best_v8 = next((c for c in ordered if c.kernel_kind == "attn2d_causal_softmax_v8"), None)
        if best_v8 is not None:
            if len(final) >= b and final:
                final[-1] = best_v8
            else:
                final.append(best_v8)

    return BackendPlan(
        kernel="flash_attention2d",
        source_oracle=dict(source_oracle or {}),
        hardware_model=hardware_model.to_json_dict(),
        selected_modules=selected_modules,
        module_edges=selected_edges,
        param_space=param_space,
        constraints=constraints,
        substitutions=substitutions,
        candidates=final,
        notes=[
            f"goals={sorted(goal_tags)}",
            f"mechanisms={sorted(mechanism_tags)}",
            f"source_kernel_kind={exact_kind or 'none'}",
            f"cluster={cluster}",
            f"toolchain_effective_sm={str((toolchain_model or {}).get('effective_sm') or '')}",
            f"toolchain_downleveled={bool((toolchain_model or {}).get('downleveled'))}",
            f"async_evidence={bool(async_evidence_ok)}",
            *preserve_notes,
        ],
    )


__all__ = ["plan_flash_attention2d"]
