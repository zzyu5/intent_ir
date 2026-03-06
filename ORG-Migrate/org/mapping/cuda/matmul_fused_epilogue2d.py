from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import matmul_fused_epilogue2d_catalog
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


def _mma_async_guardrails(*, bm: int, bn: int, bk: int, threads: int) -> tuple[bool, str]:
    if threads <= 0:
        return False, "invalid threads"
    vec_copy = (
        (int(bk) % 4) == 0
        and (int(bn) % 4) == 0
        and ((int(bm) * int(bk)) % 4) == 0
        and ((int(bk) * int(bn)) % 4) == 0
    )
    if not vec_copy:
        return False, "vec4_copy_not_eligible"
    tile_a4 = (int(bm) * int(bk)) // 4
    tile_b4 = (int(bk) * int(bn)) // 4
    if tile_a4 <= 0 or tile_b4 <= 0:
        return False, "invalid_vec4_tile"
    if (tile_a4 % int(threads)) != 0 or (tile_b4 % int(threads)) != 0:
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
    selected_ids = {"ab_tile_stage", "epilogue_fused_writeback", "backend_tile_v2", "backend_tile_v1"}
    substitutions: list[dict[str, Any]] = []
    if "mma_acceleration" in goal_tags or "mma_core" in mechanism_tags or str(source_oracle.get("kernel_kind") or "").startswith("matmul_mma"):
        if hardware_model.supports_mma:
            selected_ids.update({"mma_core", "backend_mma_v1"})
        else:
            substitutions.append(
                {
                    "from": "matmul.mma_core",
                    "to": "matmul.tile_core",
                    "reason": "hardware_model.supports_mma = false",
                }
            )
    if "latency_hiding" in goal_tags or "prefetch_pipeline" in mechanism_tags:
        if hardware_model.supports_async_copy:
            selected_ids.add("prefetch_pipeline")
        else:
            substitutions.append(
                {
                    "from": "matmul.prefetch_pipeline",
                    "to": "matmul.sync_prefetch",
                    "reason": "hardware_model.supports_async_copy = false",
                }
            )
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    if str(source_oracle.get("kernel_kind") or "").startswith("matmul_mma") and "mma_core" not in {m.id for m in selected_modules}:
        substitutions.append(
            {
                "from": "source.mma_core",
                "to": "matmul.tile_core",
                "reason": "source MMA path not preserved by selected modules",
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


def _complete_async_evidence(
    *,
    ttgir_facts: Mapping[str, Any] | None,
    ptx_facts: Mapping[str, Any] | None,
) -> bool:
    return bool(
        _fact_present(ttgir_facts, "staging.operand_tile_stage")
        and _fact_present(ttgir_facts, "pipeline.stage_hint")
        and bool(_fact_attr(ptx_facts, "pipeline.async_copy", "complete_async_pipeline", False))
        and bool(_fact_attr(ptx_facts, "primitive.mma", "complete_matrix_pipeline", False))
    )


def _score_matmul_candidate(
    *,
    candidate: BackendCandidate,
    cluster: str,
    source_oracle: Mapping[str, Any],
    complete_async_evidence: bool,
    goal_tags: set[str],
) -> tuple[float, str, str]:
    kind = str(candidate.kernel_kind)
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    is_async = bool(bindings.get("MMA_ASYNC_COPY", 0))
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    source_async = bool(source_bindings.get("MMA_ASYNC_COPY", 0))

    score = 0.0
    reasons: list[str] = [f"cluster={cluster}"]
    portability_note = "portable"

    if kind == "matmul_mma_tf32_v1":
        score += 120.0
        reasons.append("mma_core")
    elif kind == "matmul_tile_v2":
        score += 60.0
        portability_note = "tile_fallback"
    else:
        score += 45.0
        portability_note = "tile_fallback"

    if kind == "matmul_mma_tf32_v1":
        score += {32: 12.0, 16: 6.0, 64: 4.0}.get(int(bindings.get("MMA_BK", 0)), 0.0)
        score += {32: 10.0, 64: 4.0}.get(int(bindings.get("MMA_BM", 0)), 0.0)
        score += {32: 10.0, 16: 4.0}.get(int(bindings.get("MMA_BN", 0)), 0.0)
    if "fused_epilogue_avoid_writeback" in goal_tags and kind.startswith("matmul_"):
        score += 4.0
        reasons.append("preserve:epilogue")
    if "mma_acceleration" in goal_tags and kind == "matmul_mma_tf32_v1":
        score += 8.0
        reasons.append("preserve:mma")

    if cluster == "cuda_tc_mid_smem" and source_async and not complete_async_evidence:
        if kind == "matmul_tile_v2":
            score += 150.0
            portability_note = "cluster_prefers_tile_v2"
            reasons.append("mid_smem_portable_tile")
        elif kind == "matmul_mma_tf32_v1" and not is_async:
            score -= 18.0
            reasons.append("mid_smem_sync_mma_underperforms_tile")

    if is_async:
        if cluster == "cuda_generic" or not complete_async_evidence:
            score -= 80.0
            portability_note = "async_portability_blocked"
            reasons.append("async_portability_blocked")
        elif cluster == "cuda_tc_mid_smem":
            score -= 10.0
            reasons.append("async_mid_smem")
        else:
            score += 10.0
            reasons.append("async_large_smem")
    if source_kind == kind and source_bindings == bindings:
        score += 4.0
        reasons.append("source_exact")
    return score, ",".join(reasons), portability_note


def plan_matmul_fused_epilogue2d(
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
    m_dim = _require_dim(shape_bindings, "M")
    n_dim = _require_dim(shape_bindings, "N")
    k_dim = _require_dim(shape_bindings, "K")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)

    modules, module_edges, _passes = matmul_fused_epilogue2d_catalog(hardware_model)
    selected_modules, selected_edges, substitutions = _selected_modules(
        modules=modules,
        module_edges=module_edges,
        goal_tags=goal_tags,
        mechanism_tags=mechanism_tags,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
    )
    preserve_notes: list[str] = []

    if m_dim <= 0 or n_dim <= 0 or k_dim <= 0 or (k_dim % 8) != 0:
        substitutions.append(
            {
                "from": "matmul_fused_epilogue2d",
                "to": "backend.skip",
                "reason": f"unsupported dims: M={m_dim} N={n_dim} K={k_dim}",
            }
        )
        return BackendPlan(
            kernel="matmul_fused_epilogue2d",
            source_oracle=dict(source_oracle or {}),
            hardware_model=hardware_model.to_json_dict(),
            selected_modules=selected_modules,
            module_edges=selected_edges,
            param_space={"kernel_kind": ["matmul_mma_tf32_v1", "matmul_tile_v2", "matmul_tile_v1"]},
            constraints=["K % 8 == 0"],
            substitutions=substitutions,
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}", f"cluster={hardware_model.arch_cluster}"],
        )

    dim_candidates_norm = collect_dim_candidate_ints_normalized(org)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    bm_values = _ordered_param_values(
        defaults=[32, 64],
        preferred=_coerce_int(source_bindings.get("MMA_BM")),
        allowed=union_dim_candidate_ints(dim_candidates_norm, "tile_m", "MMA_BM", "BLOCK_M"),
    )
    bn_values = _ordered_param_values(
        defaults=[32, 16],
        preferred=_coerce_int(source_bindings.get("MMA_BN")),
        allowed=union_dim_candidate_ints(dim_candidates_norm, "tile_n", "MMA_BN", "BLOCK_N"),
    )
    bk_values = _ordered_param_values(
        defaults=[32, 16, 64],
        preferred=_coerce_int(source_bindings.get("MMA_BK")),
        allowed=union_dim_candidate_ints(dim_candidates_norm, "tile_k", "MMA_BK", "BLOCK_K"),
    )

    cluster = str(hardware_model.arch_cluster)
    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    exact_bindings = dict(source_bindings)
    want_async = bool(
        any(m.id == "prefetch_pipeline" for m in selected_modules)
        and hardware_model.supports_async_copy
        and "latency_hiding" in goal_tags
    )
    want_mma = any(m.id == "mma_core" for m in selected_modules)
    if exact_kind:
        preserve_notes.append(f"source_oracle_variant={exact_kind}")
    if "mma_core" in mechanism_tags or "mma_acceleration" in goal_tags:
        preserve_notes.append("preserve:mma_core")
    if "epilogue_fused_writeback" in mechanism_tags or "fused_epilogue_avoid_writeback" in goal_tags:
        preserve_notes.append("preserve:epilogue_fused_writeback")

    complete_async_evidence = _complete_async_evidence(ttgir_facts=ttgir_facts, ptx_facts=ptx_facts)
    if want_async and not complete_async_evidence:
        substitutions.append(
            {
                "from": "matmul.prefetch_pipeline",
                "to": "matmul.sync_prefetch",
                "reason": "incomplete async evidence",
            }
        )
        preserve_notes.append("replace:prefetch_pipeline->sync_prefetch")

    param_space = {
        "kernel_kind": ["matmul_mma_tf32_v1", "matmul_tile_v2", "matmul_tile_v1"],
        "MMA_BM": list(bm_values),
        "MMA_BN": list(bn_values),
        "MMA_BK": list(bk_values),
        "MMA_ASYNC_COPY": ([1] if want_async and complete_async_evidence and cluster != "cuda_generic" else []),
    }
    constraints = [
        "MMA_BM%16==0",
        "MMA_BN%16==0",
        "MMA_BK%8==0",
        "fused_epilogue_avoid_writeback preserved",
        "portable sync MMA outranks async on mid_smem",
    ]

    scored: list[BackendCandidate] = []
    if want_mma:
        for bm in bm_values:
            if (m_dim % int(bm)) != 0:
                continue
            for bn in bn_values:
                if (n_dim % int(bn)) != 0:
                    continue
                warps = (int(bm) // 16) * (int(bn) // 16)
                threads = int(warps) * int(hardware_model.warp_size)
                if threads <= 0 or threads > 1024:
                    continue
                for bk in bk_values:
                    if (k_dim % int(bk)) != 0:
                        continue
                    sync_candidate = BackendCandidate(kernel_kind="matmul_mma_tf32_v1", bindings={"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk)})
                    score, score_reason, portability_note = _score_matmul_candidate(
                        candidate=sync_candidate,
                        cluster=cluster,
                        source_oracle=source_oracle,
                        complete_async_evidence=complete_async_evidence,
                        goal_tags=goal_tags,
                    )
                    if exact_kind == "matmul_mma_tf32_v1" and int(source_bindings.get("MMA_ASYNC_COPY", 0)) == 1 and sync_candidate.bindings == {
                        "MMA_BM": int(source_bindings.get("MMA_BM", bm)),
                        "MMA_BN": int(source_bindings.get("MMA_BN", bn)),
                        "MMA_BK": int(source_bindings.get("MMA_BK", bk)),
                    }:
                        score += 30.0
                        portability_note = "drop:MMA_ASYNC_COPY"
                        score_reason = f"{score_reason},portable_mma_repair"
                    scored.append(
                        BackendCandidate(
                            kernel_kind=sync_candidate.kernel_kind,
                            bindings=dict(sync_candidate.bindings),
                            note="portable_mma_neighbor",
                            score=score,
                            score_reason=score_reason,
                            cluster=cluster,
                            portability_note=portability_note,
                        )
                    )
                    if want_async and complete_async_evidence and cluster != "cuda_generic":
                        ok, reason = _mma_async_guardrails(bm=int(bm), bn=int(bn), bk=int(bk), threads=int(threads))
                        if not ok:
                            substitutions.append(
                                {
                                    "from": "matmul.prefetch_pipeline",
                                    "to": "matmul.sync_prefetch",
                                    "reason": reason,
                                    "detail": {"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk)},
                                }
                            )
                            continue
                        async_candidate = BackendCandidate(
                            kernel_kind="matmul_mma_tf32_v1",
                            bindings={"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk), "MMA_ASYNC_COPY": 1},
                        )
                        async_score, async_reason, async_portability = _score_matmul_candidate(
                            candidate=async_candidate,
                            cluster=cluster,
                            source_oracle=source_oracle,
                            complete_async_evidence=complete_async_evidence,
                            goal_tags=goal_tags,
                        )
                        scored.append(
                            BackendCandidate(
                                kernel_kind=async_candidate.kernel_kind,
                                bindings=dict(async_candidate.bindings),
                                note="portable_async_mma",
                                score=async_score,
                                score_reason=async_reason,
                                cluster=cluster,
                                portability_note=async_portability,
                            )
                        )

    if exact_kind == "matmul_mma_tf32_v1" and int(source_bindings.get("MMA_ASYNC_COPY", 0)) == 1 and not any(
        c.note == "portable_mma_neighbor" and c.portability_note == "drop:MMA_ASYNC_COPY" for c in scored
    ):
        substitutions.append(
            {
                "from": "source.prefetch_pipeline",
                "to": "matmul.sync_prefetch",
                "reason": "source async MMA path has no valid portable target realization",
            }
        )

    for kind, note in (("matmul_tile_v2", "tile_baseline"), ("matmul_tile_v1", "tile_fallback")):
        tile_candidate = BackendCandidate(kernel_kind=kind, bindings={})
        tile_score, tile_reason, tile_portability = _score_matmul_candidate(
            candidate=tile_candidate,
            cluster=cluster,
            source_oracle=source_oracle,
            complete_async_evidence=complete_async_evidence,
            goal_tags=goal_tags,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=kind,
                bindings={},
                note=note,
                score=tile_score,
                score_reason=tile_reason,
                cluster=cluster,
                portability_note=tile_portability,
            )
        )

    final: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    ordered = sorted(
        scored,
        key=lambda c: (
            -float(c.score if c.score is not None else 0.0),
            0 if c.kernel_kind == "matmul_mma_tf32_v1" else (1 if c.kernel_kind == "matmul_tile_v2" else 2),
            -int(c.bindings.get("MMA_BK", 0)),
            -int(c.bindings.get("MMA_BM", 0)),
            -int(c.bindings.get("MMA_BN", 0)),
            0 if int(c.bindings.get("MMA_ASYNC_COPY", 0)) == 0 else 1,
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
        kernel="matmul_fused_epilogue2d",
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
            f"async_evidence={bool(complete_async_evidence)}",
            *preserve_notes,
        ],
    )


__all__ = ["plan_matmul_fused_epilogue2d"]
