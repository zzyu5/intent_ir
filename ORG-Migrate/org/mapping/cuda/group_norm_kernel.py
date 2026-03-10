from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import group_norm_kernel_catalog
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
    vector_width = int(size_per_thread[-1]) if size_per_thread else None
    threads_hint: int | None = None
    if warps_per_cta and threads_per_warp_layout:
        product = 1
        for value in threads_per_warp_layout:
            product *= int(value)
        warps = 1
        for value in warps_per_cta:
            warps *= int(value)
        threads_hint = int(product * warps)
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
        "group_norm_group_tile_resident",
        "group_norm_warp_reduction",
        "group_norm_online_normalization",
        "group_norm_affine_fused_epilogue",
        "group_norm_backend_v1",
    }
    if mechanism_tags & {"blocked_layout", "group_tile_resident", "warp_reduction", "vector_group_io"}:
        selected_ids.add("group_norm_vector_group_io")
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    return selected_modules, selected_edges


def _score_group_norm_candidate(
    *,
    candidate: BackendCandidate,
    n_dim: int,
    c_dim: int,
    hw_dim: int,
    group_size: int,
    goal_tags: set[str],
    mechanism_tags: set[str],
    cluster: str,
    blocked_threads_hint: int | None,
    blocked_vector_hint: int | None,
) -> tuple[float, str, str]:
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    threads = int(bindings.get("GROUP_NORM_BLOCK_THREADS", 0))
    vector_width = int(bindings.get("GROUP_NORM_VECTOR_WIDTH", 1))
    elems = int(group_size * hw_dim)
    score = 102.0
    reasons: list[str] = [f"cluster={cluster}", f"threads={threads}", f"vec={vector_width}", f"elems={elems}"]
    portability = "portable"

    if int(elems) >= 1024:
        if int(vector_width) == 4:
            score += 18.0
            reasons.append("large_group:vec4")
        elif int(vector_width) == 2:
            score += 8.0
            reasons.append("large_group:vec2")
        if int(threads) == 256:
            score += 10.0
            reasons.append("large_group:cta256")
        elif int(threads) == 128:
            score += 4.0
            reasons.append("large_group:cta128")

    if blocked_threads_hint is not None:
        if int(threads) == int(blocked_threads_hint):
            score += 8.0
            reasons.append("preserve:ttgir_threads")
        else:
            score -= abs(int(threads) - int(blocked_threads_hint)) / 64.0
    if blocked_vector_hint is not None:
        if int(vector_width) == int(blocked_vector_hint):
            score += 6.0
            reasons.append("preserve:ttgir_vector")
        else:
            score -= abs(int(vector_width) - int(blocked_vector_hint)) * 2.0

    if mechanism_tags & {"group_tile_resident", "blocked_layout"} and int(vector_width) > 1:
        score += 8.0
        reasons.append("rationale:group_tile")
    if mechanism_tags & {"warp_reduction", "online_normalization"} and int(threads) in {128, 256}:
        score += 10.0
        reasons.append("rationale:warp_reduce")
    if mechanism_tags & {"affine_fused_epilogue"}:
        score += 5.0
        reasons.append("rationale:affine_fused")
    if "memory_coalescing" in goal_tags and int(vector_width) > 1:
        score += 8.0
        reasons.append("goal:coalescing")
    if "reduction_tree_balance" in goal_tags and int(threads) in {128, 256}:
        score += 6.0
        reasons.append("goal:reduction_tree")
    if "fused_epilogue_avoid_writeback" in goal_tags:
        score += 4.0
        reasons.append("goal:fused_epilogue")
    if int(group_size) != 1 and int(vector_width) > 1:
        score -= 48.0
        portability = "requires_group_scalarization"
        reasons.append("shape:group_size_scalar_only")

    if int(threads) not in {64, 128, 256}:
        score -= 100.0
        portability = "requires_thread_repair"
    if int(vector_width) not in {1, 2, 4}:
        score -= 100.0
        portability = "requires_vector_repair"
    if int(hw_dim) % int(max(1, vector_width * 4)) != 0 and int(vector_width) > 1:
        score -= 16.0
        reasons.append("tail_vec_penalty")

    return score, ",".join(reasons), portability


def plan_group_norm_kernel(
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
    n_dim = _require_dim(shape_bindings, "N")
    c_dim = _require_dim(shape_bindings, "C")
    hw_dim = _require_dim(shape_bindings, "HW")
    num_groups = _require_dim(shape_bindings, "num_groups")
    group_size = _require_dim(shape_bindings, "group_size")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    modules, module_edges, _passes = group_norm_kernel_catalog(hardware_model)
    selected_modules, selected_edges = _selected_modules(modules=modules, module_edges=module_edges, mechanism_tags=mechanism_tags)
    dim_candidates = collect_dim_candidate_ints_normalized(org)
    blocked_threads_hint, blocked_vector_hint = _blocked_layout_hints(ttgir_facts)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}

    thread_values = union_dim_candidate_ints(
        dim_candidates,
        "GROUP_NORM_BLOCK_THREADS",
        "threads_per_block",
        "num_warps",
        "block_threads",
    )
    expanded_threads: list[int] = []
    for value in thread_values:
        iv = int(value)
        if iv in {1, 2, 4, 8}:
            expanded_threads.append(iv * int(hardware_model.warp_size))
        else:
            expanded_threads.append(iv)
    expanded_threads.extend([128, 256])
    if blocked_threads_hint is not None:
        expanded_threads.append(int(blocked_threads_hint))
    if "GROUP_NORM_BLOCK_THREADS" in source_bindings:
        expanded_threads.append(int(source_bindings["GROUP_NORM_BLOCK_THREADS"]))
    thread_values = [int(x) for x in dict.fromkeys(expanded_threads) if int(x) in {64, 128, 256}]
    if not thread_values:
        thread_values = [128, 256]

    vector_values = union_dim_candidate_ints(
        dim_candidates,
        "GROUP_NORM_VECTOR_WIDTH",
        "vector_width",
        "size_per_thread",
        "BLOCK_HW_SIZE",
    )
    normalized_vectors: list[int] = []
    for value in vector_values:
        iv = int(value)
        if iv >= 32:
            continue
        normalized_vectors.append(iv)
    normalized_vectors.extend([1, 2, 4])
    if blocked_vector_hint is not None:
        normalized_vectors.append(int(blocked_vector_hint))
    if "GROUP_NORM_VECTOR_WIDTH" in source_bindings:
        normalized_vectors.append(int(source_bindings["GROUP_NORM_VECTOR_WIDTH"]))
    vector_values = [int(x) for x in dict.fromkeys(normalized_vectors) if int(x) in {1, 2, 4}]
    if not vector_values:
        vector_values = [1, 2, 4]

    cluster = str(hardware_model.arch_cluster)
    param_space = {
        "kernel_kind": ["group_norm_v1"],
        "GROUP_NORM_BLOCK_THREADS": list(thread_values),
        "GROUP_NORM_VECTOR_WIDTH": list(vector_values),
    }
    constraints = [
        "N > 0",
        "C > 0",
        "HW > 0",
        "num_groups > 0",
        "C % num_groups == 0",
        "GROUP_NORM_BLOCK_THREADS in {64,128,256}",
        "GROUP_NORM_VECTOR_WIDTH in {1,2,4}",
    ]
    substitutions: list[dict[str, Any]] = []
    candidates: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    for threads in thread_values:
        for vector_width in vector_values:
            base_bindings = {
                "GROUP_NORM_BLOCK_THREADS": int(threads),
                "GROUP_NORM_VECTOR_WIDTH": int(vector_width),
            }
            candidate = BackendCandidate(
                kernel_kind="group_norm_v1",
                bindings=base_bindings,
                note="group_norm_kernel",
                cluster=cluster,
            )
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            score, score_reason, portability = _score_group_norm_candidate(
                candidate=candidate,
                n_dim=n_dim,
                c_dim=c_dim,
                hw_dim=hw_dim,
                group_size=group_size,
                goal_tags=goal_tags,
                mechanism_tags=mechanism_tags,
                cluster=cluster,
                blocked_threads_hint=blocked_threads_hint,
                blocked_vector_hint=blocked_vector_hint,
            )
            candidates.append(
                BackendCandidate(
                    kernel_kind="group_norm_v1",
                    bindings=base_bindings,
                    note="group_norm_kernel",
                    score=float(score),
                    score_reason=str(score_reason),
                    cluster=cluster,
                    portability_note=str(portability),
                )
            )

    candidates.sort(
        key=lambda c: (
            -float(c.score or 0.0),
            -int(dict(c.bindings or {}).get("GROUP_NORM_VECTOR_WIDTH", 1)),
            -int(dict(c.bindings or {}).get("GROUP_NORM_BLOCK_THREADS", 0)),
            str(c.kernel_kind),
        )
    )
    candidates = candidates[:b]

    return BackendPlan(
        kernel="group_norm_kernel",
        source_oracle=dict(source_oracle or {}),
        hardware_model=dict(hardware_model.to_json_dict()),
        selected_modules=list(selected_modules),
        module_edges=list(selected_edges),
        param_space=param_space,
        constraints=constraints,
        substitutions=substitutions,
        candidates=candidates,
        notes=[
            f"cluster={cluster}",
            f"shape=(N={n_dim},C={c_dim},HW={hw_dim},G={num_groups})",
        ],
    )


__all__ = ["plan_group_norm_kernel"]
