from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import row_reduction_catalog
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


def _candidate_key(candidate: BackendCandidate) -> tuple[str, tuple[tuple[str, int], ...]]:
    return str(candidate.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(candidate.bindings or {}).items()))


def _selected_modules(
    *,
    reduction_kind: str,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    mechanism_tags: set[str],
) -> tuple[list[BackendModule], list[BackendModuleEdge]]:
    prefix = f"row_{reduction_kind}"
    selected_ids = {f"{prefix}_row_tile_resident", f"{prefix}_warp_reduction_tree", f"{prefix}_writeback", f"{prefix}_backend_v2"}
    if mechanism_tags & {"vector_row_path", "tile_load_stage"}:
        selected_ids.add(f"{prefix}_vector_row_load")
    if mechanism_tags & {"shared_staging", "block_synchronization", "warp_parallel_rows", "row_parallel_axis"}:
        selected_ids.add(f"{prefix}_shared_warp_exchange")
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    return selected_modules, selected_edges


def _score_row_reduce_candidate(
    *,
    candidate: BackendCandidate,
    reduction_kind: str,
    row_width: int,
    goal_tags: set[str],
    mechanism_tags: set[str],
    cluster: str,
    blocked_threads_hint: int | None,
    blocked_vector_hint: int | None,
    reduction_scope: str,
) -> tuple[float, str, str]:
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    threads = int(bindings.get("ROW_REDUCE_BLOCK_THREADS", 0))
    vector_width = int(bindings.get("ROW_REDUCE_VECTOR_WIDTH", 1))
    shared_stage = int(bindings.get("ROW_REDUCE_SHARED_STAGE", 0))
    score = 100.0
    reasons: list[str] = [f"cluster={cluster}", f"row_width={row_width}", f"threads={threads}", f"vec={vector_width}"]
    portability = "portable"

    effective_width = max(1, int(threads * max(1, vector_width)))
    waste_ratio = float(max(0, effective_width - row_width)) / float(max(1, row_width))
    score -= waste_ratio * 18.0
    reasons.append(f"waste={waste_ratio:.3f}")
    large_row = int(row_width) >= 16384
    bandwidth_row = int(row_width) >= 1048576
    if large_row:
        score += min(14.0, float(int(vector_width)) * 3.5)
        reasons.append("large_row:vectorized")
        if int(threads) in {128, 256}:
            score += 8.0
            reasons.append("large_row:wide_cta")
        if int(shared_stage) == 1:
            score += 6.0
            reasons.append("large_row:shared_tree")
    if bandwidth_row and int(shared_stage) == 1:
        if int(threads) == 256 and int(vector_width) == 4:
            score += 20.0
            reasons.append("bandwidth_row:cta256_vec4")
        elif int(threads) == 256 and int(vector_width) == 2:
            score += 12.0
            reasons.append("bandwidth_row:cta256_vec2")
        elif int(threads) == 128 and int(vector_width) == 4:
            score += 8.0
            reasons.append("bandwidth_row:cta128_vec4")

    if blocked_threads_hint is not None:
        if int(threads) == int(blocked_threads_hint):
            score += 12.0
            reasons.append("preserve:ttgir_threads")
        else:
            score -= abs(int(threads) - int(blocked_threads_hint)) / 32.0
    if blocked_vector_hint is not None:
        if int(vector_width) == int(blocked_vector_hint):
            score += 10.0
            reasons.append("preserve:ttgir_vector")
        else:
            score -= abs(int(vector_width) - int(blocked_vector_hint)) * 3.0

    if mechanism_tags & {"vector_row_path", "tile_load_stage"} and int(vector_width) > 1:
        score += 9.0
        reasons.append("rationale:vector_row")
    if mechanism_tags & {"row_reduction", "warp_reduction_tree"}:
        score += 10.0
        reasons.append("rationale:reduction_tree")
    if mechanism_tags & {"shared_staging", "block_synchronization"} and int(shared_stage) == 1:
        score += 6.0
        reasons.append("rationale:shared_stage")
    if "resident_working_set" in goal_tags:
        score += 4.0
        reasons.append("goal:resident")
    if "memory_coalescing" in goal_tags and int(vector_width) > 1:
        score += 5.0
        reasons.append("goal:coalescing")
    if "reduction_tree_balance" in goal_tags and int(threads) in {32, 64, 128}:
        score += 6.0
        reasons.append("goal:balanced_tree")
    if reduction_scope == "warp" and int(threads) in {32, 64}:
        score += 8.0
        reasons.append("fact:warp_scope")

    if int(threads) > 32 and int(shared_stage) != 1:
        score -= 80.0
        portability = "requires_shared_repair"
        reasons.append("missing:shared_exchange")
    if int(vector_width) not in {1, 2, 4}:
        score -= 100.0
        portability = "requires_vector_repair"
    if int(threads) not in {32, 64, 128, 256}:
        score -= 100.0
        portability = "requires_thread_repair"
    if int(vector_width) * 8 > max(1, row_width) and int(row_width) < 32:
        score -= 8.0

    if reduction_kind == "sum" and int(vector_width) == 2:
        score += 2.0
    if reduction_kind == "max" and int(threads) == 64:
        score += 2.0
    if large_row and reduction_kind == "sum" and int(vector_width) == 4:
        score += 4.0
    if large_row and reduction_kind == "max" and int(vector_width) == 4:
        score += 4.0

    return score, ",".join(reasons), portability


def _plan_row_reduction(
    *,
    kernel: str,
    reduction_kind: str,
    org: OrgDoc,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None,
    budget: int,
) -> BackendPlan:
    b = max(1, int(budget))
    m_dim = _require_dim(shape_bindings, "M")
    n_dim = _require_dim(shape_bindings, "N")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    modules, module_edges, _passes = row_reduction_catalog(hardware_model, reduction_kind=reduction_kind)
    selected_modules, selected_edges = _selected_modules(
        reduction_kind=reduction_kind,
        modules=modules,
        module_edges=module_edges,
        mechanism_tags=mechanism_tags,
    )
    dim_candidates = collect_dim_candidate_ints_normalized(org)
    blocked_threads_hint, blocked_vector_hint = _blocked_layout_hints(ttgir_facts)
    reduction_scope = str(_fact_attrs(ttgir_facts, "communication.reduction").get("reduction_scope") or "")
    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    exact_threads = _coerce_int(source_bindings.get("ROW_REDUCE_BLOCK_THREADS"))
    exact_vector = _coerce_int(source_bindings.get("ROW_REDUCE_VECTOR_WIDTH"))

    thread_values = _ordered_unique(
        [
            int(x)
            for x in (
                union_dim_candidate_ints(dim_candidates, "threads_per_block", "ROW_REDUCE_BLOCK_THREADS")
                + union_dim_candidate_ints(dim_candidates, "num_warps")
                + [32, 64, 128]
            )
            if _coerce_int(x) is not None
        ]
    )
    expanded_threads: list[int] = []
    for value in thread_values:
        if int(value) in {1, 2, 4, 8}:
            expanded_threads.append(int(value) * int(hardware_model.warp_size))
        else:
            expanded_threads.append(int(value))
    if blocked_threads_hint is not None:
        expanded_threads.append(int(blocked_threads_hint))
    if exact_threads is not None:
        expanded_threads.append(int(exact_threads))
    thread_values = [int(x) for x in _ordered_unique(expanded_threads) if int(x) in {32, 64, 128, 256}]
    if not thread_values:
        thread_values = [32, 64, 128, 256]

    vector_values = _ordered_unique(
        [
            int(x)
            for x in (
                union_dim_candidate_ints(dim_candidates, "size_per_thread", "vector_width", "ROW_REDUCE_VECTOR_WIDTH")
                + [1, 2, 4]
            )
            if _coerce_int(x) is not None
        ]
    )
    if blocked_vector_hint is not None:
        vector_values.append(int(blocked_vector_hint))
    if exact_vector is not None:
        vector_values.append(int(exact_vector))
    vector_values = [int(x) for x in _ordered_unique(vector_values) if int(x) in {1, 2, 4}]
    if not vector_values:
        vector_values = [1, 2, 4]

    cluster = str(hardware_model.arch_cluster)
    row_tile_present = "row_tile_resident" in mechanism_tags
    vector_present = bool(mechanism_tags & {"vector_row_path", "tile_load_stage"}) or _fact_present(ttgir_facts, "layout.vector_row_path")
    shared_present = bool(mechanism_tags & {"shared_staging", "block_synchronization"}) or (int(hardware_model.shared_mem_kb or 0) >= 64)
    reduction_present = bool(mechanism_tags & {"row_reduction", "warp_reduction_tree"}) or _fact_present(ttgir_facts, "communication.reduction")

    param_space = {
        "kernel_kind": [f"row_{reduction_kind}_axis1_v2"],
        "ROW_REDUCE_BLOCK_THREADS": list(thread_values),
        "ROW_REDUCE_VECTOR_WIDTH": list(vector_values),
        "ROW_REDUCE_SHARED_STAGE": ([0, 1] if shared_present else [0]),
    }
    constraints = [
        "M > 0",
        "N > 0",
        "ROW_REDUCE_BLOCK_THREADS in {32,64,128,256}",
        "ROW_REDUCE_VECTOR_WIDTH in {1,2,4}",
    ]
    if int(n_dim) <= 0 or int(m_dim) <= 0:
        return BackendPlan(
            kernel=kernel,
            source_oracle=dict(source_oracle or {}),
            hardware_model=hardware_model.to_json_dict(),
            selected_modules=selected_modules,
            module_edges=selected_edges,
            param_space=param_space,
            constraints=constraints,
            substitutions=[{"from": kernel, "to": "backend.skip", "reason": f"unsupported dims: M={m_dim} N={n_dim}"}],
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}", f"mechanisms={sorted(mechanism_tags)}", f"cluster={cluster}"],
        )

    scored: list[BackendCandidate] = []
    exact_kind_allowed = exact_kind == f"row_{reduction_kind}_axis1_v2"
    if exact_kind_allowed and exact_threads is not None and exact_vector is not None:
        exact_shared = int(source_bindings.get("ROW_REDUCE_SHARED_STAGE", 1 if exact_threads > 32 else 0))
        score, reason, portability = _score_row_reduce_candidate(
            candidate=BackendCandidate(
                kernel_kind=exact_kind,
                bindings={
                    "ROW_REDUCE_BLOCK_THREADS": int(exact_threads),
                    "ROW_REDUCE_VECTOR_WIDTH": int(exact_vector),
                    "ROW_REDUCE_SHARED_STAGE": int(exact_shared),
                },
            ),
            reduction_kind=reduction_kind,
            row_width=n_dim,
            goal_tags=goal_tags,
            mechanism_tags=mechanism_tags,
            cluster=cluster,
            blocked_threads_hint=blocked_threads_hint,
            blocked_vector_hint=blocked_vector_hint,
            reduction_scope=reduction_scope,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=exact_kind,
                bindings={
                    "ROW_REDUCE_BLOCK_THREADS": int(exact_threads),
                    "ROW_REDUCE_VECTOR_WIDTH": int(exact_vector),
                    "ROW_REDUCE_SHARED_STAGE": int(exact_shared),
                },
                note="source_exact",
                score=score + 18.0,
                score_reason=f"{reason},source_exact",
                cluster=cluster,
                portability_note=portability,
            )
        )

    for threads in thread_values:
        for vector_width in vector_values:
            if int(vector_width) > int(max(1, n_dim)):
                continue
            if int(threads) * int(vector_width) > int(max(32, 2 * n_dim)):
                continue
            shared_options = [1] if int(threads) > int(hardware_model.warp_size) else [0]
            if shared_present and int(threads) > int(hardware_model.warp_size):
                shared_options = [1]
            for shared_stage in shared_options:
                score, reason, portability = _score_row_reduce_candidate(
                    candidate=BackendCandidate(
                        kernel_kind=f"row_{reduction_kind}_axis1_v2",
                        bindings={
                            "ROW_REDUCE_BLOCK_THREADS": int(threads),
                            "ROW_REDUCE_VECTOR_WIDTH": int(vector_width),
                            "ROW_REDUCE_SHARED_STAGE": int(shared_stage),
                        },
                    ),
                    reduction_kind=reduction_kind,
                    row_width=n_dim,
                    goal_tags=goal_tags,
                    mechanism_tags=(mechanism_tags | ({"vector_row_path"} if vector_present else set()) | ({"row_reduction"} if reduction_present else set())),
                    cluster=cluster,
                    blocked_threads_hint=blocked_threads_hint,
                    blocked_vector_hint=blocked_vector_hint,
                    reduction_scope=reduction_scope,
                )
                scored.append(
                    BackendCandidate(
                        kernel_kind=f"row_{reduction_kind}_axis1_v2",
                        bindings={
                            "ROW_REDUCE_BLOCK_THREADS": int(threads),
                            "ROW_REDUCE_VECTOR_WIDTH": int(vector_width),
                            "ROW_REDUCE_SHARED_STAGE": int(shared_stage),
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
            -int(c.bindings.get("ROW_REDUCE_VECTOR_WIDTH", 1)),
            0 if int(c.bindings.get("ROW_REDUCE_SHARED_STAGE", 0)) == 1 else 1,
            -int(c.bindings.get("ROW_REDUCE_BLOCK_THREADS", 0)),
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
        kernel=kernel,
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
            f"row_tile_present={bool(row_tile_present)}",
            f"vector_present={bool(vector_present)}",
            f"reduction_present={bool(reduction_present)}",
            f"shared_present={bool(shared_present)}",
        ],
    )


def plan_row_sum(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return _plan_row_reduction(
        kernel="row_sum",
        reduction_kind="sum",
        org=org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        budget=budget,
    )


def plan_row_max(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return _plan_row_reduction(
        kernel="row_max",
        reduction_kind="max",
        org=org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        budget=budget,
    )


__all__ = ["plan_row_sum", "plan_row_max"]
