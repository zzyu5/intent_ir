from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import attn_fwd_catalog
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


def _fact_present(facts: Mapping[str, Any] | None, key: str) -> bool:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    return bool(dict(mechanisms.get(str(key)) or {}).get("present"))


def _selected_modules(
    *,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    goal_tags: set[str],
    mechanism_tags: set[str],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
) -> tuple[list[BackendModule], list[BackendModuleEdge], list[dict[str, Any]]]:
    selected_ids = {"mask_causal_apply", "output_accumulator", "backend_attn_fwd_softmax_v1"}
    substitutions: list[dict[str, Any]] = []
    if "streaming_softmax_state" in goal_tags or "online_softmax_reduce" in mechanism_tags:
        selected_ids.update({"online_softmax_reduce", "backend_attn_fwd_softmax_v2"})
    if "resident_working_set" in goal_tags or "qkv_stage" in mechanism_tags or "kv_streamed_tiles" in mechanism_tags:
        selected_ids.update({"qkv_stage", "backend_attn_fwd_tiled_v3"})
    if "latency_hiding" in goal_tags or "prefetch_pipeline" in mechanism_tags:
        if hardware_model.supports_async_copy:
            selected_ids.add("prefetch_pipeline")
        else:
            substitutions.append(
                {
                    "from": "attn_fwd.prefetch_pipeline",
                    "to": "attn_fwd.sync_prefetch",
                    "reason": "hardware_model.supports_async_copy = false",
                }
            )
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    if source_kind and source_kind != "attn_fwd_tiled_v3":
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


def _complete_pipeline_evidence(ttgir_facts: Mapping[str, Any] | None, ptx_facts: Mapping[str, Any] | None) -> bool:
    return bool(
        _fact_present(ttgir_facts, "pipeline.stage_hint")
        and _fact_present(ttgir_facts, "staging.kv_streamed_tiles")
        and _fact_present(ttgir_facts, "communication.streaming_softmax")
        and bool(dict(dict((ptx_facts or {}).get("mechanisms") or {}).get("pipeline.async_copy") or {}).get("attrs", {}).get("complete_async_pipeline"))
    )


def _score_attn_fwd_candidate(
    *,
    candidate: BackendCandidate,
    cluster: str,
    source_oracle: Mapping[str, Any],
    goal_tags: set[str],
    mechanism_tags: set[str],
    pipeline_evidence_ok: bool,
) -> tuple[float, str, str]:
    kind = str(candidate.kernel_kind)
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    block_m = int(bindings.get("ATTN_FWD_BLOCK_M", 0))
    block_kv = int(bindings.get("ATTN_FWD_BLOCK_KV", 0))
    score = 0.0
    reasons: list[str] = [f"cluster={cluster}"]
    portability_note = "portable"

    if cluster == "cuda_tc_mid_smem":
        if kind == "attn_fwd_tiled_v3":
            score += (132.0 if pipeline_evidence_ok else 120.0)
        elif kind == "attn_fwd_softmax_v2":
            score += 100.0
        else:
            score += 82.0
    elif cluster == "cuda_tc_large_smem":
        if kind == "attn_fwd_tiled_v3":
            score += (145.0 if pipeline_evidence_ok else 118.0)
        elif kind == "attn_fwd_softmax_v2":
            score += 104.0
        else:
            score += 84.0
    else:
        score += (90.0 if kind == "attn_fwd_tiled_v3" else (76.0 if kind == "attn_fwd_softmax_v2" else 60.0))

    if kind == "attn_fwd_tiled_v3":
        score += {8: 12.0, 4: 6.0}.get(block_m, 0.0)
        score += {32: 10.0, 16: 4.0}.get(block_kv, 0.0)
        reasons.extend([f"block_m={block_m}", f"block_kv={block_kv}"])
    if "streaming_softmax_state" in goal_tags and kind in {"attn_fwd_tiled_v3", "attn_fwd_softmax_v2"}:
        score += 8.0
        reasons.append("preserve:streaming_softmax_state")
    if ("mask_causal_apply" in mechanism_tags or "mask_causal" in mechanism_tags) and kind in {"attn_fwd_tiled_v3", "attn_fwd_softmax_v2", "attn_fwd_softmax_v1"}:
        score += 6.0
        reasons.append("preserve:mask_causal")
    if pipeline_evidence_ok and kind == "attn_fwd_tiled_v3":
        score += 8.0
        reasons.append("prefetch_pipeline")
    if source_kind == kind and source_bindings == bindings:
        score += 24.0
        reasons.append("source_exact")
    return score, ",".join(reasons), portability_note


def plan_attn_fwd(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    b = max(1, int(budget))
    q_ctx = _require_dim(shape_bindings, "Q_CTX")
    kv_ctx = _require_dim(shape_bindings, "KV_CTX")
    head_dim = _require_dim(shape_bindings, "HEAD_DIM")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)

    modules, module_edges, _passes = attn_fwd_catalog(hardware_model)
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
                "from": "_attn_fwd",
                "to": "backend.skip",
                "reason": f"unsupported dims: Q_CTX={q_ctx} KV_CTX={kv_ctx} HEAD_DIM={head_dim}",
            }
        )
        return BackendPlan(
            kernel="_attn_fwd",
            source_oracle=dict(source_oracle or {}),
            hardware_model=hardware_model.to_json_dict(),
            selected_modules=selected_modules,
            module_edges=selected_edges,
            param_space={"kernel_kind": ["attn_fwd_tiled_v3", "attn_fwd_softmax_v2", "attn_fwd_softmax_v1"]},
            constraints=["HEAD_DIM == 64"],
            substitutions=substitutions,
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}", f"cluster={hardware_model.arch_cluster}"],
        )

    dim_candidates_norm = collect_dim_candidate_ints_normalized(org)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    block_m_values = _ordered_param_values(
        defaults=[8, 4],
        preferred=_coerce_int(source_bindings.get("ATTN_FWD_BLOCK_M")),
        allowed=union_dim_candidate_ints(dim_candidates_norm, "block_m", "tile_m", "ATTN_FWD_BLOCK_M"),
    )
    block_kv_values = _ordered_param_values(
        defaults=[32, 16],
        preferred=_coerce_int(source_bindings.get("ATTN_FWD_BLOCK_KV")),
        allowed=union_dim_candidate_ints(dim_candidates_norm, "block_kv", "tile_kv", "ATTN_FWD_BLOCK_KV"),
    )
    block_m_values = [int(x) for x in block_m_values if int(x) <= int(q_ctx)]
    block_kv_values = [int(x) for x in block_kv_values if int(x) <= int(kv_ctx)]
    if not block_m_values:
        block_m_values = [8]
    if not block_kv_values:
        block_kv_values = [16]

    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    exact_bindings = dict(source_bindings)
    cluster = str(hardware_model.arch_cluster)
    pipeline_evidence_ok = _complete_pipeline_evidence(ttgir_facts=ttgir_facts, ptx_facts=ptx_facts)
    if exact_kind:
        preserve_notes.append(f"source_oracle_variant={exact_kind}")
    if "streaming_softmax_state" in goal_tags:
        preserve_notes.append("preserve:online_softmax_reduce")
    if "latency_hiding" in goal_tags:
        preserve_notes.append("preserve:prefetch_pipeline")

    param_space = {
        "kernel_kind": ["attn_fwd_tiled_v3", "attn_fwd_softmax_v2", "attn_fwd_softmax_v1"],
        "ATTN_FWD_BLOCK_M": list(block_m_values),
        "ATTN_FWD_BLOCK_KV": list(block_kv_values),
    }
    constraints = [
        "HEAD_DIM == 64",
        "ATTN_FWD_BLOCK_M <= Q_CTX",
        "ATTN_FWD_BLOCK_KV <= KV_CTX",
        "streaming_softmax_state preserved",
        "mask_causal handling preserved",
    ]

    scored: list[BackendCandidate] = []
    if exact_kind == "attn_fwd_tiled_v3":
        score, score_reason, portability_note = _score_attn_fwd_candidate(
            candidate=BackendCandidate(kernel_kind=exact_kind, bindings=dict(exact_bindings)),
            cluster=cluster,
            source_oracle=source_oracle,
            goal_tags=goal_tags,
            mechanism_tags=mechanism_tags,
            pipeline_evidence_ok=pipeline_evidence_ok,
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

    for bm in block_m_values:
        for bk in block_kv_values:
            candidate = BackendCandidate(kernel_kind="attn_fwd_tiled_v3", bindings={"ATTN_FWD_BLOCK_M": int(bm), "ATTN_FWD_BLOCK_KV": int(bk)})
            score, score_reason, portability_note = _score_attn_fwd_candidate(
                candidate=candidate,
                cluster=cluster,
                source_oracle=source_oracle,
                goal_tags=goal_tags,
                mechanism_tags=mechanism_tags,
                pipeline_evidence_ok=pipeline_evidence_ok,
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

    for kind in ("attn_fwd_softmax_v2", "attn_fwd_softmax_v1"):
        score, score_reason, portability_note = _score_attn_fwd_candidate(
            candidate=BackendCandidate(kernel_kind=kind, bindings={}),
            cluster=cluster,
            source_oracle=source_oracle,
            goal_tags=goal_tags,
            mechanism_tags=mechanism_tags,
            pipeline_evidence_ok=pipeline_evidence_ok,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=kind,
                bindings={},
                note="cluster_rank",
                score=score,
                score_reason=score_reason,
                cluster=cluster,
                portability_note=portability_note,
            )
        )

    final: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    ordered = sorted(
        scored,
        key=lambda c: (
            -float(c.score if c.score is not None else 0.0),
            0 if c.kernel_kind == "attn_fwd_tiled_v3" else (1 if c.kernel_kind == "attn_fwd_softmax_v2" else 2),
            -int(c.bindings.get("ATTN_FWD_BLOCK_M", 0)),
            -int(c.bindings.get("ATTN_FWD_BLOCK_KV", 0)),
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

    return BackendPlan(
        kernel="_attn_fwd",
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
            f"pipeline_evidence={bool(pipeline_evidence_ok)}",
            *preserve_notes,
        ],
    )


__all__ = ["plan_attn_fwd"]
