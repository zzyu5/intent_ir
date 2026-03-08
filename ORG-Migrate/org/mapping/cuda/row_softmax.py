from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import row_softmax_catalog
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


def _triton_candidate_supported(*, row_width: int, block_threads: int) -> bool:
    if int(row_width) <= 0:
        return False
    if int(block_threads) <= 0 or (int(block_threads) % 32) != 0:
        return False
    if int(block_threads) > 256:
        return False
    pad_n = 1 << (int(row_width) - 1).bit_length()
    pad_n = int(min(int(pad_n), 1024))
    return int(pad_n) % int(block_threads) == 0


def _selected_modules(
    *,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    mechanism_tags: set[str],
    masked: bool,
) -> tuple[list[BackendModule], list[BackendModuleEdge]]:
    prefix = "masked_softmax" if masked else "softmax_inner"
    selected_ids = {f"{prefix}_row_reduction", f"{prefix}_backend_v1"}
    if "row_tile_resident" in mechanism_tags or "vector_row_path" in mechanism_tags:
        selected_ids.update({f"{prefix}_row_tile_resident", f"{prefix}_vector_row_path", f"{prefix}_backend_triton_v1"})
    if masked:
        selected_ids.add(f"{prefix}_mask_apply")
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    return selected_modules, selected_edges


def _candidate_key(candidate: BackendCandidate) -> tuple[str, tuple[tuple[str, int], ...]]:
    return str(candidate.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(candidate.bindings or {}).items()))


def _score_softmax_candidate(
    *,
    candidate: BackendCandidate,
    cluster: str,
    goal_tags: set[str],
    mechanism_tags: set[str],
    row_width: int,
    masked: bool,
) -> tuple[float, str, str]:
    kind = str(candidate.kernel_kind)
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    block_threads = int(bindings.get("SOFTMAX_BLOCK_THREADS", 0))
    score = 0.0
    reasons: list[str] = [f"cluster={cluster}", f"row_width={row_width}"]
    portability_note = "portable"

    if kind == "row_softmax_axis1_triton_v1":
        score += 120.0
    elif kind == "row_softmax_axis1_v1":
        score += 86.0
    else:
        score += 78.0

    if kind == "row_softmax_axis1_triton_v1":
        score += {64: 14.0, 128: 18.0, 256: 10.0}.get(block_threads, 0.0)
        reasons.append(f"block_threads={block_threads}")
        if not _triton_candidate_supported(row_width=row_width, block_threads=block_threads):
            score -= 140.0
            portability_note = "requires_fallback"
            reasons.append("incompatible:block_threads")
        if row_width <= 64 and block_threads == 64:
            score += 14.0
            reasons.append("small_row_fit")
        elif row_width <= 256 and block_threads == 128:
            score += 10.0
            reasons.append("mid_row_fit")
    if "streaming_softmax_state" in goal_tags or "row_reduction" in mechanism_tags:
        score += 8.0
        reasons.append("preserve:row_reduction")
    if "avoid_materialization" in goal_tags and kind == "row_softmax_axis1_triton_v1":
        score += 6.0
        reasons.append("preserve:avoid_materialization")
    if masked and "mask_apply" in mechanism_tags:
        score += 6.0
        reasons.append("preserve:mask_apply")
    return score, ",".join(reasons), portability_note


def _plan_row_softmax(
    *,
    kernel: str,
    org: OrgDoc,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None,
    budget: int,
    masked: bool,
) -> BackendPlan:
    b = max(1, int(budget))
    m_dim = _require_dim(shape_bindings, "M")
    n_dim = _require_dim(shape_bindings, "N")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    modules, module_edges, _passes = row_softmax_catalog(hardware_model, masked=masked)
    selected_modules, selected_edges = _selected_modules(modules=modules, module_edges=module_edges, mechanism_tags=mechanism_tags, masked=masked)

    if m_dim <= 0 or n_dim <= 0:
        return BackendPlan(
            kernel=kernel,
            source_oracle=dict(source_oracle or {}),
            hardware_model=hardware_model.to_json_dict(),
            selected_modules=selected_modules,
            module_edges=selected_edges,
            param_space={"kernel_kind": ["row_masked_softmax_axis1_v1"] if masked else ["row_softmax_axis1_triton_v1", "row_softmax_axis1_v1"]},
            constraints=["M > 0", "N > 0"],
            substitutions=[{"from": kernel, "to": "backend.skip", "reason": f"unsupported dims: M={m_dim} N={n_dim}"}],
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}", f"cluster={hardware_model.arch_cluster}"],
        )

    dim_candidates_norm = collect_dim_candidate_ints_normalized(org)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    thread_values = _ordered_param_values(
        defaults=[64, 128],
        preferred=_coerce_int(source_bindings.get("SOFTMAX_BLOCK_THREADS")),
        allowed=union_dim_candidate_ints(dim_candidates_norm, "block_threads", "SOFTMAX_BLOCK_THREADS"),
    )
    cluster = str(hardware_model.arch_cluster)
    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    exact_bindings = dict(source_bindings)
    row_tile_present = _fact_present(ttgir_facts, "staging.row_tile_resident")
    vector_present = _fact_present(ttgir_facts, "layout.vector_row_path")

    param_space = {
        "kernel_kind": (["row_masked_softmax_axis1_v1"] if masked else ["row_softmax_axis1_triton_v1", "row_softmax_axis1_v1"]),
        "SOFTMAX_BLOCK_THREADS": ([] if masked else list(thread_values)),
    }
    constraints = ["M > 0", "N > 0", "streaming_softmax_state preserved"]
    if masked:
        constraints.append("mask_apply preserved")

    scored: list[BackendCandidate] = []
    if masked:
        cand = BackendCandidate(kernel_kind="row_masked_softmax_axis1_v1", bindings={})
        score, reason, portability = _score_softmax_candidate(
            candidate=BackendCandidate(kernel_kind="row_softmax_axis1_v1", bindings={}),
            cluster=cluster,
            goal_tags=goal_tags,
            mechanism_tags=mechanism_tags | {"mask_apply"},
            row_width=n_dim,
            masked=True,
        )
        scored.append(BackendCandidate(kernel_kind=cand.kernel_kind, bindings={}, note="cluster_rank", score=score, score_reason=reason, cluster=cluster, portability_note=portability))
    else:
        if exact_kind == "row_softmax_axis1_triton_v1":
            source_threads = int(exact_bindings.get("SOFTMAX_BLOCK_THREADS", 0) or 128)
            if _triton_candidate_supported(row_width=n_dim, block_threads=source_threads):
                score, reason, portability = _score_softmax_candidate(
                    candidate=BackendCandidate(kernel_kind=exact_kind, bindings=dict(exact_bindings)),
                    cluster=cluster,
                    goal_tags=goal_tags,
                    mechanism_tags=mechanism_tags,
                    row_width=n_dim,
                    masked=False,
                )
                scored.append(
                    BackendCandidate(
                        kernel_kind=exact_kind,
                        bindings=dict(exact_bindings),
                        note="source_exact",
                        score=score + 18.0,
                        score_reason=f"{reason},source_exact",
                        cluster=cluster,
                        portability_note=portability,
                    )
                )
        for threads in thread_values:
            score, reason, portability = _score_softmax_candidate(
                candidate=BackendCandidate(kernel_kind="row_softmax_axis1_triton_v1", bindings={"SOFTMAX_BLOCK_THREADS": int(threads)}),
                cluster=cluster,
                goal_tags=goal_tags,
                mechanism_tags=mechanism_tags | ({"row_tile_resident"} if row_tile_present else set()) | ({"vector_row_path"} if vector_present else set()),
                row_width=n_dim,
                masked=False,
            )
            if _triton_candidate_supported(row_width=n_dim, block_threads=int(threads)):
                scored.append(
                    BackendCandidate(
                        kernel_kind="row_softmax_axis1_triton_v1",
                        bindings={"SOFTMAX_BLOCK_THREADS": int(threads)},
                        note="cluster_rank",
                        score=score,
                        score_reason=reason,
                        cluster=cluster,
                        portability_note=portability,
                    )
                )
        score, reason, portability = _score_softmax_candidate(
            candidate=BackendCandidate(kernel_kind="row_softmax_axis1_v1", bindings={}),
            cluster=cluster,
            goal_tags=goal_tags,
            mechanism_tags=mechanism_tags,
            row_width=n_dim,
            masked=False,
        )
        scored.append(BackendCandidate(kernel_kind="row_softmax_axis1_v1", bindings={}, note="fallback", score=score, score_reason=reason, cluster=cluster, portability_note=portability))

    final: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    ordered = sorted(
        scored,
        key=lambda c: (
            -float(c.score if c.score is not None else 0.0),
            0 if "triton" in c.kernel_kind or "masked" in c.kernel_kind else 1,
            -int(c.bindings.get("SOFTMAX_BLOCK_THREADS", 0)),
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

    notes = [
        f"goals={sorted(goal_tags)}",
        f"mechanisms={sorted(mechanism_tags)}",
        f"source_kernel_kind={exact_kind or 'none'}",
        f"cluster={cluster}",
        f"row_tile_present={bool(row_tile_present)}",
        f"vector_row_present={bool(vector_present)}",
    ]
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
        notes=notes,
    )


def plan_softmax_inner(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return _plan_row_softmax(
        kernel="softmax_inner",
        org=org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        budget=budget,
        masked=False,
    )


def plan_masked_softmax2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return _plan_row_softmax(
        kernel="masked_softmax2d",
        org=org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        budget=budget,
        masked=True,
    )


__all__ = ["plan_softmax_inner", "plan_masked_softmax2d"]
