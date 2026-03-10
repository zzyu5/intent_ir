from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendPlan
from org.mapping.cuda.module_catalog import ai_bench_softmax_catalog
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


def _selected_modules(org: OrgDoc, hardware_model: HardwareModel):
    modules, edges, _passes = ai_bench_softmax_catalog(hardware_model)
    mechanism_tags = _mechanism_tags(org)
    selected_ids = {"ai_softmax_row_tile_resident", "ai_softmax_row_reduction", "ai_softmax_backend_v1"}
    if mechanism_tags & {"vector_row_path", "power2_padding"}:
        selected_ids.add("ai_softmax_power2_padding")
    if mechanism_tags & {"vector_row_path"}:
        selected_ids.add("ai_softmax_vector_row_path")
        selected_ids.add("ai_softmax_backend_vec4_v2")
    return [m for m in modules if m.id in selected_ids], [e for e in edges if e.src in selected_ids and e.dst in selected_ids]


def _score_candidate(
    *,
    candidate: BackendCandidate,
    row_width: int,
    goal_tags: set[str],
    mechanism_tags: set[str],
    cluster: str,
) -> tuple[float, str]:
    kind = str(candidate.kernel_kind)
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    score = 100.0
    reasons: list[str] = [f"cluster={cluster}", f"row_width={row_width}", f"kind={kind}"]
    if row_width >= 1024:
        if kind == "row_softmax_axis1_v1":
            score += 72.0
            reasons.append("wide_row:generic_shuffle_path")
        if kind == "row_softmax_axis1_triton_v1":
            threads = int(bindings.get("SOFTMAX_BLOCK_THREADS") or 256)
            if threads == 512:
                score += 92.0
                reasons.append("wide_row:half_row_cta")
            elif threads == 1024:
                score += 36.0
                reasons.append("wide_row:full_row_cta")
            else:
                score += 24.0
                reasons.append("wide_row:triton256")
        if kind == "row_softmax_axis1_vec4_v2":
            score -= 24.0
            reasons.append("wide_row:vec4_register_pressure")
    else:
        if kind == "row_softmax_axis1_vec4_v2":
            score += 18.0
            reasons.append("small_row:vec4")
    if "memory_coalescing" in goal_tags and kind == "row_softmax_axis1_vec4_v2":
        score += 8.0
        reasons.append("goal:coalescing")
    if "streaming_softmax_state" in goal_tags:
        score += 6.0
        reasons.append("goal:softmax_state")
    if "row_reduction" in mechanism_tags and kind == "row_softmax_axis1_v1":
        score += 10.0
        reasons.append("mechanism:row_reduction")
    if "vector_row_path" in mechanism_tags and kind == "row_softmax_axis1_vec4_v2":
        score += 10.0
        reasons.append("mechanism:vector_row_path")
    return score, ",".join(reasons)


def plan_ai_bench_softmax(
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
    del ttgir_facts, ptx_facts, toolchain_model
    b = max(1, int(budget))
    r_dim = _require_dim(shape_bindings, "R")
    c_dim = _require_dim(shape_bindings, "C")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    selected_modules, selected_edges = _selected_modules(org, hardware_model)
    cluster = str(hardware_model.arch_cluster)

    candidates = [
        BackendCandidate(kernel_kind="row_softmax_axis1_v1", bindings={}),
        BackendCandidate(kernel_kind="row_softmax_axis1_triton_v1", bindings={"SOFTMAX_BLOCK_THREADS": 1024}),
        BackendCandidate(kernel_kind="row_softmax_axis1_triton_v1", bindings={"SOFTMAX_BLOCK_THREADS": 512}),
        BackendCandidate(kernel_kind="row_softmax_axis1_triton_v1", bindings={"SOFTMAX_BLOCK_THREADS": 256}),
        BackendCandidate(kernel_kind="row_softmax_axis1_vec4_v2", bindings={"SOFTMAX_BLOCK_THREADS": 256, "SOFTMAX_VEC4": 1}),
    ]
    ranked: list[BackendCandidate] = []
    for candidate in candidates:
        score, reason = _score_candidate(
            candidate=candidate,
            row_width=c_dim,
            goal_tags=goal_tags,
            mechanism_tags=mechanism_tags,
            cluster=cluster,
        )
        ranked.append(
            BackendCandidate(
                kernel_kind=str(candidate.kernel_kind),
                bindings=dict(candidate.bindings or {}),
                note="ai_bench_softmax",
                score=float(score),
                score_reason=str(reason),
                cluster=cluster,
                portability_note="portable",
            )
        )
    ranked.sort(key=lambda c: (-float(c.score or 0.0), str(c.kernel_kind)))
    ranked = ranked[:b]
    return BackendPlan(
        kernel="ai_bench_softmax",
        source_oracle=dict(source_oracle or {}),
        hardware_model=hardware_model.to_json_dict(),
        selected_modules=selected_modules,
        module_edges=selected_edges,
        param_space={
            "kernel_kind": ["row_softmax_axis1_v1", "row_softmax_axis1_triton_v1", "row_softmax_axis1_vec4_v2"],
            "SOFTMAX_BLOCK_THREADS": [256, 512, 1024],
            "SOFTMAX_VEC4": [1],
        },
        constraints=["R > 0", "C > 0", "C <= 1024"],
        substitutions=[],
        candidates=ranked,
        notes=[f"rows={r_dim}", f"row_width={c_dim}", f"goals={sorted(goal_tags)}", f"cluster={cluster}"],
    )


__all__ = ["plan_ai_bench_softmax"]
