from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import layer_norm_persistent_catalog
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


def _fact_attrs(facts: Mapping[str, Any] | None, key: str) -> dict[str, Any]:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    return dict(dict(mechanisms.get(str(key)) or {}).get("attrs") or {})


def _blocked_layout_hints(ttgir_facts: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    attrs = _fact_attrs(ttgir_facts, "tiling.blocked_layout")
    layouts = list(attrs.get("layouts") or [])
    if not layouts:
        return None, None
    first = dict(layouts[0] or {})
    size_per_thread = list(first.get("size_per_thread") or [])
    warps_per_cta = list(first.get("warps_per_cta") or [])
    threads_per_warp_layout = list(first.get("threads_per_warp_layout") or [])
    vector_width = int(size_per_thread[0]) if size_per_thread else None
    threads_hint: int | None = None
    if warps_per_cta and threads_per_warp_layout:
        threads_hint = int(warps_per_cta[0]) * int(threads_per_warp_layout[0])
    return threads_hint, vector_width


def _candidate_key(candidate: BackendCandidate) -> tuple[str, tuple[tuple[str, int], ...]]:
    return str(candidate.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(candidate.bindings or {}).items()))


def _selected_modules(
    *,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    mechanism_tags: set[str],
) -> tuple[list[BackendModule], list[BackendModuleEdge]]:
    selected_ids = {
        "layer_norm_row_tile_resident",
        "layer_norm_warp_statistics",
        "layer_norm_affine_epilogue",
        "layer_norm_backend_v1",
    }
    if mechanism_tags & {"register_staging", "warp_parallel_execution", "tile_load_stage"}:
        selected_ids.add("layer_norm_register_stage")
    if mechanism_tags & {"row_tile_resident", "register_staging", "persistent_row_cache", "block_synchronization"}:
        selected_ids.add("layer_norm_persistent_row_cache")
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    return selected_modules, selected_edges


def _score_layer_norm_candidate(
    *,
    candidate: BackendCandidate,
    row_width: int,
    goal_tags: set[str],
    mechanism_tags: set[str],
    cluster: str,
    blocked_threads_hint: int | None,
    blocked_vector_hint: int | None,
    shared_mem_kb: int,
) -> tuple[float, str, str]:
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    threads = int(bindings.get("LAYER_NORM_BLOCK_THREADS", 0))
    vector_width = int(bindings.get("LAYER_NORM_VECTOR_WIDTH", 1))
    persistent_row = int(bindings.get("LAYER_NORM_PERSISTENT_ROW", 0))
    score = 104.0
    reasons: list[str] = [f"cluster={cluster}", f"row_width={row_width}", f"threads={threads}", f"vec={vector_width}"]
    portability = "portable"

    effective_width = max(1, int(threads * max(1, vector_width)))
    waste_ratio = float(max(0, effective_width - row_width)) / float(max(1, row_width))
    score -= waste_ratio * 16.0
    reasons.append(f"waste={waste_ratio:.3f}")
    large_row = int(row_width) >= 16384
    bandwidth_row = int(row_width) >= 1048576
    if large_row:
        if int(vector_width) == 4:
            score += 16.0
            reasons.append("large_row:vec4")
        elif int(vector_width) == 2:
            score += 8.0
            reasons.append("large_row:vec2")
        if int(threads) in {128, 256}:
            score += 10.0
            reasons.append("large_row:wide_cta")
    if bandwidth_row and int(persistent_row) == 0:
        if int(threads) == 256 and int(vector_width) == 2:
            score += 40.0
            reasons.append("bandwidth_row:cta256_vec2")
        elif int(threads) == 128 and int(vector_width) == 2:
            score += 14.0
            reasons.append("bandwidth_row:cta128_vec2")
        elif int(vector_width) == 4:
            score -= 28.0
            reasons.append("bandwidth_row:vec4_pressure")

    if blocked_threads_hint is not None:
        if int(threads) == int(blocked_threads_hint):
            score += 8.0
            reasons.append("preserve:ttgir_threads")
        else:
            score -= abs(int(threads) - int(blocked_threads_hint)) / 48.0
    if blocked_vector_hint is not None:
        if int(vector_width) == int(blocked_vector_hint):
            score += 10.0
            reasons.append("preserve:ttgir_vector")
        else:
            score -= abs(int(vector_width) - int(blocked_vector_hint)) * 2.0

    if mechanism_tags & {"row_tile_resident", "register_staging", "persistent_row_cache"} and int(persistent_row) == 1:
        score += 12.0
        reasons.append("rationale:persistent_row")
    if mechanism_tags & {"warp_reduction", "warp_parallel_execution", "row_parallel_axis"} and int(threads) in {32, 64, 128}:
        score += 10.0
        reasons.append("rationale:warp_stats")
    if mechanism_tags & {"register_staging", "warp_parallel_execution", "tile_load_stage"} and int(vector_width) > 1:
        score += 8.0
        reasons.append("rationale:vector_stage")
    if "persistent_row_state" in goal_tags and int(persistent_row) == 1:
        score += 8.0
        reasons.append("goal:persistent")
    if "memory_coalescing" in goal_tags and int(vector_width) > 1:
        score += 5.0
        reasons.append("goal:coalescing")
    if "affine_epilogue_fusion" in goal_tags:
        score += 4.0
        reasons.append("goal:affine_epilogue")

    resident_ratio = float(row_width * 4) / float(max(1, shared_mem_kb * 1024))
    if int(persistent_row) == 1:
        reasons.append(f"resident_ratio={resident_ratio:.4f}")
        if resident_ratio > 1.0:
            score -= 96.0
            reasons.append("resident_over_budget")
            portability = "requires_streaming_repair"
        else:
            score += 6.0 if resident_ratio <= 0.25 else -12.0
        if large_row and int(vector_width) == 2 and int(threads) in {128, 256}:
            score += 18.0
            reasons.append("large_row:persistent_cache_reuse")
        elif large_row and int(vector_width) == 4:
            score -= 10.0
            reasons.append("large_row:persistent_vec4_pressure")
        if bandwidth_row:
            score -= 18.0
            reasons.append("bandwidth_row:persistent_reuse_loss")
    if int(persistent_row) == 0 and mechanism_tags & {"row_tile_resident", "register_staging", "persistent_row_cache"}:
        score -= 24.0
        portability = "missing_persistent_row"
    if large_row and int(persistent_row) == 0:
        score += 6.0
        reasons.append("large_row:streaming_row")
    if int(threads) not in {32, 64, 128, 256}:
        score -= 100.0
        portability = "requires_thread_repair"
    if int(vector_width) not in {1, 2, 4}:
        score -= 100.0
        portability = "requires_vector_repair"
    return score, ",".join(reasons), portability


def plan_layer_norm_persistent(
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
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    modules, module_edges, _passes = layer_norm_persistent_catalog(hardware_model)
    selected_modules, selected_edges = _selected_modules(modules=modules, module_edges=module_edges, mechanism_tags=mechanism_tags)
    dim_candidates = collect_dim_candidate_ints_normalized(org)
    blocked_threads_hint, blocked_vector_hint = _blocked_layout_hints(ttgir_facts)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()

    thread_values = union_dim_candidate_ints(dim_candidates, "threads_per_block", "num_warps", "LAYER_NORM_BLOCK_THREADS")
    expanded_threads: list[int] = []
    for value in thread_values:
        iv = int(value)
        if iv in {1, 2, 4, 8}:
            expanded_threads.append(iv * int(hardware_model.warp_size))
        else:
            expanded_threads.append(iv)
    expanded_threads.extend([32, 64, 128, 256])
    if blocked_threads_hint is not None:
        expanded_threads.append(int(blocked_threads_hint))
    if "LAYER_NORM_BLOCK_THREADS" in source_bindings:
        expanded_threads.append(int(source_bindings["LAYER_NORM_BLOCK_THREADS"]))
    thread_values = [int(x) for x in dict.fromkeys(expanded_threads) if int(x) in {32, 64, 128, 256}]

    vector_values = union_dim_candidate_ints(dim_candidates, "vector_width", "size_per_thread", "LAYER_NORM_VECTOR_WIDTH")
    vector_values.extend([1, 2, 4])
    if blocked_vector_hint is not None:
        vector_values.append(int(blocked_vector_hint))
    if "LAYER_NORM_VECTOR_WIDTH" in source_bindings:
        vector_values.append(int(source_bindings["LAYER_NORM_VECTOR_WIDTH"]))
    vector_values = [int(x) for x in dict.fromkeys(vector_values) if int(x) in {1, 2, 4}]
    if not vector_values:
        vector_values = [1, 2, 4]

    cluster = str(hardware_model.arch_cluster)
    persistent_rationale = bool(
        mechanism_tags & {"row_tile_resident", "register_staging", "persistent_row_cache", "warp_parallel_execution"}
    ) or bool(goal_tags & {"persistent_row_state", "resident_working_set", "memory_coalescing"})
    persistent_allowed = bool(
        persistent_rationale
        and int(hardware_model.shared_mem_kb or 0) >= 64
        and (int(n_dim) <= 1024 or int(n_dim) >= 16384)
    )
    param_space = {
        "kernel_kind": ["layer_norm_axis1_v1"],
        "LAYER_NORM_BLOCK_THREADS": list(thread_values),
        "LAYER_NORM_VECTOR_WIDTH": list(vector_values),
        "LAYER_NORM_PERSISTENT_ROW": ([0, 1] if persistent_allowed else [0]),
    }
    constraints = [
        "M > 0",
        "N > 0",
        "LAYER_NORM_BLOCK_THREADS in {32,64,128,256}",
        "LAYER_NORM_VECTOR_WIDTH in {1,2,4}",
    ]

    scored: list[BackendCandidate] = []
    if exact_kind == "layer_norm_axis1_v1" and source_bindings:
        score, reason, portability = _score_layer_norm_candidate(
            candidate=BackendCandidate(kernel_kind="layer_norm_axis1_v1", bindings=dict(source_bindings)),
            row_width=n_dim,
            goal_tags=goal_tags,
            mechanism_tags=mechanism_tags,
            cluster=cluster,
            blocked_threads_hint=blocked_threads_hint,
            blocked_vector_hint=blocked_vector_hint,
            shared_mem_kb=int(hardware_model.shared_mem_kb),
        )
        scored.append(
            BackendCandidate(
                kernel_kind="layer_norm_axis1_v1",
                bindings=dict(source_bindings),
                note="source_exact",
                score=score + 14.0,
                score_reason=f"{reason},source_exact",
                cluster=cluster,
                portability_note=portability,
            )
        )

    for threads in thread_values:
        for vector_width in vector_values:
            if int(vector_width) > max(1, n_dim):
                continue
            persistent_values = [0, 1] if persistent_allowed else [0]
            for persistent_row in persistent_values:
                score, reason, portability = _score_layer_norm_candidate(
                    candidate=BackendCandidate(
                        kernel_kind="layer_norm_axis1_v1",
                        bindings={
                            "LAYER_NORM_BLOCK_THREADS": int(threads),
                            "LAYER_NORM_VECTOR_WIDTH": int(vector_width),
                            "LAYER_NORM_PERSISTENT_ROW": int(persistent_row),
                        },
                    ),
                    row_width=n_dim,
                    goal_tags=goal_tags,
                    mechanism_tags=mechanism_tags,
                    cluster=cluster,
                    blocked_threads_hint=blocked_threads_hint,
                    blocked_vector_hint=blocked_vector_hint,
                    shared_mem_kb=int(hardware_model.shared_mem_kb),
                )
                scored.append(
                    BackendCandidate(
                        kernel_kind="layer_norm_axis1_v1",
                        bindings={
                            "LAYER_NORM_BLOCK_THREADS": int(threads),
                            "LAYER_NORM_VECTOR_WIDTH": int(vector_width),
                            "LAYER_NORM_PERSISTENT_ROW": int(persistent_row),
                        },
                        note="cluster_rank",
                        score=score,
                        score_reason=reason,
                        cluster=cluster,
                        portability_note=portability,
                    )
                )

    ordered = sorted(
        scored,
        key=lambda c: (
            -float(c.score if c.score is not None else 0.0),
            0 if int(c.bindings.get("LAYER_NORM_PERSISTENT_ROW", 0)) == 1 else 1,
            -int(c.bindings.get("LAYER_NORM_VECTOR_WIDTH", 1)),
            -int(c.bindings.get("LAYER_NORM_BLOCK_THREADS", 0)),
        ),
    )
    final: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    for candidate in ordered:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        final.append(candidate)
        if len(final) >= b:
            break

    return BackendPlan(
        kernel="layer_norm_persistent",
        source_oracle=dict(source_oracle or {}),
        hardware_model=hardware_model.to_json_dict(),
        selected_modules=selected_modules,
        module_edges=selected_edges,
        param_space=param_space,
        constraints=constraints,
        substitutions=[],
        candidates=final,
        notes=[
            f"goals={sorted(goal_tags)}",
            f"mechanisms={sorted(mechanism_tags)}",
            f"cluster={cluster}",
            f"persistent_allowed={persistent_allowed}",
        ],
    )


__all__ = ["plan_layer_norm_persistent"]
