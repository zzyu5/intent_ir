from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import elementwise2d_catalog
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
    op_kind: str,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    mechanism_tags: set[str],
) -> tuple[list[BackendModule], list[BackendModuleEdge]]:
    prefix = f"elementwise_{str(op_kind).strip()}"
    primitive_id = f"{prefix}_{'add_primitive' if op_kind == 'add' else 'exp_primitive'}"
    selected_ids = {
        f"{prefix}_tile_resident",
        f"{prefix}_two_axis_grid_mapping",
        primitive_id,
        f"{prefix}_backend_v1",
    }
    if mechanism_tags & {"blocked_register_layout", "tile_load_direct", "vector_global_io", "vector_row_path"}:
        selected_ids.add(f"{prefix}_vector_global_io")
    if mechanism_tags & {"masked_edge_handling", "two_axis_grid_mapping"}:
        selected_ids.add(f"{prefix}_masked_edge_handling")
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    return selected_modules, selected_edges


def _score_elementwise_candidate(
    *,
    candidate: BackendCandidate,
    op_kind: str,
    m_dim: int,
    n_dim: int,
    goal_tags: set[str],
    mechanism_tags: set[str],
    cluster: str,
    blocked_threads_hint: int | None,
    blocked_vector_hint: int | None,
) -> tuple[float, str, str]:
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    threads = int(bindings.get("ELEMENTWISE_BLOCK_THREADS", 0))
    vector_width = int(bindings.get("ELEMENTWISE_VECTOR_WIDTH", 1))
    total = int(m_dim) * int(n_dim)
    score = 96.0
    reasons: list[str] = [f"cluster={cluster}", f"threads={threads}", f"vec={vector_width}", f"total={total}"]
    portability = "portable"

    if int(total) >= 1_000_000:
        if int(vector_width) == 4:
            score += 24.0
            reasons.append("bandwidth:vec4")
        elif int(vector_width) == 2:
            score += 10.0
            reasons.append("bandwidth:vec2")
        if int(threads) == 256:
            score += 10.0
            reasons.append("bandwidth:cta256")
        elif int(threads) == 512:
            score += 8.0
            reasons.append("bandwidth:cta512")

    if blocked_threads_hint is not None:
        if int(threads) == int(blocked_threads_hint):
            score += 6.0
            reasons.append("preserve:ttgir_threads")
        else:
            score -= abs(int(threads) - int(blocked_threads_hint)) / 64.0
    if blocked_vector_hint is not None:
        if int(vector_width) == int(blocked_vector_hint):
            score += 8.0
            reasons.append("preserve:ttgir_vector")
        else:
            score -= abs(int(vector_width) - int(blocked_vector_hint)) * 2.0

    if mechanism_tags & {"blocked_register_layout", "tile_load_direct", "vector_global_io"} and int(vector_width) > 1:
        score += 10.0
        reasons.append("rationale:vector_io")
    if mechanism_tags & {"two_axis_grid_mapping"} and int(threads) in {256, 512}:
        score += 6.0
        reasons.append("rationale:grid_mapping")
    if mechanism_tags & {"masked_edge_handling"} and int(total) % int(max(1, threads * vector_width * 4)) != 0:
        score += 2.0
        reasons.append("rationale:edge_mask")
    if "memory_coalescing" in goal_tags and int(vector_width) > 1:
        score += 8.0
        reasons.append("goal:coalescing")
    if "resident_working_set" in goal_tags and int(threads) in {128, 256, 512}:
        score += 4.0
        reasons.append("goal:resident")
    if "avoid_materialization" in goal_tags:
        score += 3.0
        reasons.append("goal:materialization")
    if "latency_hiding" in goal_tags and int(threads) >= 256:
        score += 3.0
        reasons.append("goal:latency")
    if str(op_kind) == "exp" and int(vector_width) == 4:
        score += 4.0
        reasons.append("primitive:exp_vec4")

    if int(threads) not in {64, 128, 256, 512}:
        score -= 100.0
        portability = "requires_thread_repair"
    if int(vector_width) not in {1, 2, 4}:
        score -= 100.0
        portability = "requires_vector_repair"
    if int(threads) * int(vector_width) > int(total):
        score -= 8.0
        reasons.append("oversized_launch")

    return score, ",".join(reasons), portability


def _plan_elementwise2d(
    *,
    kernel: str,
    op_kind: str,
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
    modules, module_edges, _passes = elementwise2d_catalog(hardware_model, op_kind=op_kind)
    selected_modules, selected_edges = _selected_modules(
        op_kind=op_kind,
        modules=modules,
        module_edges=module_edges,
        mechanism_tags=mechanism_tags,
    )
    dim_candidates = collect_dim_candidate_ints_normalized(org)
    blocked_threads_hint, blocked_vector_hint = _blocked_layout_hints(ttgir_facts)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}

    thread_values = union_dim_candidate_ints(
        dim_candidates,
        "ELEMENTWISE_BLOCK_THREADS",
        "threads_per_block",
        "num_warps",
        "block_threads",
    )
    expanded_threads: list[int] = []
    for value in thread_values:
        iv = int(value)
        if iv in {1, 2, 4, 8, 16}:
            expanded_threads.append(iv * int(hardware_model.warp_size))
        else:
            expanded_threads.append(iv)
    expanded_threads.extend([128, 256, 512])
    if blocked_threads_hint is not None:
        expanded_threads.append(int(blocked_threads_hint))
    if "ELEMENTWISE_BLOCK_THREADS" in source_bindings:
        expanded_threads.append(int(source_bindings["ELEMENTWISE_BLOCK_THREADS"]))
    thread_values = [int(x) for x in dict.fromkeys(expanded_threads) if int(x) in {64, 128, 256, 512}]
    if not thread_values:
        thread_values = [128, 256, 512]

    vector_values = union_dim_candidate_ints(
        dim_candidates,
        "ELEMENTWISE_VECTOR_WIDTH",
        "vector_width",
        "size_per_thread",
        "BLOCK_N",
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
    if "ELEMENTWISE_VECTOR_WIDTH" in source_bindings:
        normalized_vectors.append(int(source_bindings["ELEMENTWISE_VECTOR_WIDTH"]))
    vector_values = [int(x) for x in dict.fromkeys(normalized_vectors) if int(x) in {1, 2, 4}]
    if not vector_values:
        vector_values = [1, 2, 4]

    cluster = str(hardware_model.arch_cluster)
    param_space = {
        "kernel_kind": ["elementwise_v1"],
        "ELEMENTWISE_BLOCK_THREADS": list(thread_values),
        "ELEMENTWISE_VECTOR_WIDTH": list(vector_values),
    }
    constraints = [
        "M > 0",
        "N > 0",
        "ELEMENTWISE_BLOCK_THREADS in {64,128,256,512}",
        "ELEMENTWISE_VECTOR_WIDTH in {1,2,4}",
    ]
    substitutions: list[dict[str, Any]] = []
    candidates: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    for threads in thread_values:
        for vector_width in vector_values:
            base_bindings = {
                "ELEMENTWISE_BLOCK_THREADS": int(threads),
                "ELEMENTWISE_VECTOR_WIDTH": int(vector_width),
            }
            candidate = BackendCandidate(
                kernel_kind="elementwise_v1",
                bindings=base_bindings,
                note=f"{kernel}:{op_kind}",
                cluster=cluster,
            )
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            score, score_reason, portability = _score_elementwise_candidate(
                candidate=candidate,
                op_kind=op_kind,
                m_dim=m_dim,
                n_dim=n_dim,
                goal_tags=goal_tags,
                mechanism_tags=mechanism_tags,
                cluster=cluster,
                blocked_threads_hint=blocked_threads_hint,
                blocked_vector_hint=blocked_vector_hint,
            )
            candidates.append(
                BackendCandidate(
                    kernel_kind="elementwise_v1",
                    bindings=base_bindings,
                    note=f"{kernel}:{op_kind}",
                    score=float(score),
                    score_reason=str(score_reason),
                    cluster=cluster,
                    portability_note=str(portability),
                )
            )

    candidates.sort(
        key=lambda c: (
            -float(c.score or 0.0),
            -int(dict(c.bindings or {}).get("ELEMENTWISE_VECTOR_WIDTH", 1)),
            -int(dict(c.bindings or {}).get("ELEMENTWISE_BLOCK_THREADS", 0)),
            str(c.kernel_kind),
        )
    )
    candidates = candidates[:b]

    return BackendPlan(
        kernel=str(kernel),
        source_oracle=dict(source_oracle or {}),
        hardware_model=dict(hardware_model.to_json_dict()),
        selected_modules=list(selected_modules),
        module_edges=list(selected_edges),
        param_space=param_space,
        constraints=constraints,
        substitutions=substitutions,
        candidates=candidates,
        notes=[f"cluster={cluster}", f"op_kind={op_kind}", f"shape=({m_dim},{n_dim})"],
    )


def plan_add2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return _plan_elementwise2d(
        kernel="add2d",
        op_kind="add",
        org=org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        budget=budget,
    )


def plan_exp2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return _plan_elementwise2d(
        kernel="exp2d",
        op_kind="exp",
        org=org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        budget=budget,
    )


__all__ = ["plan_add2d", "plan_exp2d"]
