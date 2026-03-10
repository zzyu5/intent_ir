from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendPlan
from org.mapping.cuda.module_catalog import ai_bench_matmul_catalog
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


def _fact_attr(facts: Mapping[str, Any] | None, key: str, attr: str, default: Any = None) -> Any:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    attrs = dict(dict(mechanisms.get(str(key)) or {}).get("attrs") or {})
    return attrs.get(str(attr), default)


def _complete_async_evidence(*, ttgir_facts: Mapping[str, Any] | None, ptx_facts: Mapping[str, Any] | None) -> bool:
    return bool(
        (_fact_present(ttgir_facts, "staging.operand_tile_stage") or _fact_present(ttgir_facts, "staging.local_or_shared"))
        and bool(_fact_attr(ptx_facts, "pipeline.async_copy", "complete_async_pipeline", False))
        and (
            bool(_fact_attr(ptx_facts, "primitive.mma", "complete_matrix_pipeline", False))
            or bool(_fact_present(ptx_facts, "pipeline.async_copy"))
        )
    )


def plan_ai_bench_matmul(
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
    m_dim = _require_dim(shape_bindings, "M")
    n_dim = _require_dim(shape_bindings, "N")
    k_dim = _require_dim(shape_bindings, "K")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    cluster = str(hardware_model.arch_cluster)
    modules, edges, _passes = ai_bench_matmul_catalog(hardware_model)
    selected_ids = {"ai_matmul_operand_tile_stage", "ai_matmul_tile_fallback", "ai_matmul_backend_tile_v2"}
    if hardware_model.supports_mma:
        selected_ids.add("ai_matmul_mma_core")
        selected_ids.update({"ai_matmul_backend_mma_v1", "ai_matmul_backend_global_v1", "ai_matmul_backend_mma_v2"})
    if hardware_model.supports_async_copy:
        selected_ids.add("ai_matmul_async_prefetch")
    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in edges if e.src in selected_ids and e.dst in selected_ids]

    async_ok = _complete_async_evidence(ttgir_facts=ttgir_facts, ptx_facts=ptx_facts)
    del toolchain_model
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    bm = int(source_bindings.get("MMA_BM") or 64)
    bn = int(source_bindings.get("MMA_BN") or 16)
    bk = int(source_bindings.get("MMA_BK") or 32)

    base = [
        BackendCandidate(kernel_kind="matmul_mma_tf32_v2", bindings={"MMA_BM": bm, "MMA_BN": bn, "MMA_BK": bk, "MMA_ASYNC_COPY": 1}),
        BackendCandidate(kernel_kind="matmul_mma_tf32_v1", bindings={"MMA_BM": bm, "MMA_BN": bn, "MMA_BK": bk}),
        BackendCandidate(kernel_kind="matmul_mma_tf32_global_v1", bindings={"MMA_BM": bm, "MMA_BN": bn, "MMA_BK": bk}),
        BackendCandidate(kernel_kind="matmul_tile_v2", bindings={}),
    ]
    ranked: list[BackendCandidate] = []
    for candidate in base:
        kind = str(candidate.kernel_kind)
        score = 80.0
        reasons: list[str] = [f"cluster={cluster}", f"kind={kind}", f"MNK={m_dim}x{n_dim}x{k_dim}"]
        portability = "portable"
        if kind == "matmul_mma_tf32_v2":
            score += 60.0
            if not async_ok:
                score -= 120.0
                portability = "requires_async_repair"
                reasons.append("async_evidence_missing")
            else:
                reasons.append("async_mma")
        elif kind == "matmul_mma_tf32_v1":
            score += 28.0
            reasons.append("mma_sync")
        elif kind == "matmul_mma_tf32_global_v1":
            score += 12.0
            reasons.append("mma_global")
        else:
            score += 4.0
            portability = "tile_fallback"
            reasons.append("tile_fallback")
        if "mma_acceleration" in goal_tags and kind.startswith("matmul_mma"):
            score += 12.0
            reasons.append("goal:mma")
        if "operand_reuse" in goal_tags and kind in {"matmul_mma_tf32_v2", "matmul_mma_tf32_v1"}:
            score += 10.0
            reasons.append("goal:reuse")
        if "latency_hiding" in goal_tags and kind == "matmul_mma_tf32_v2" and async_ok:
            score += 10.0
            reasons.append("goal:latency")
        if kind == exact_kind and dict(candidate.bindings or {}) == source_bindings:
            score += 8.0
            reasons.append("source_exact")
        ranked.append(
            BackendCandidate(
                kernel_kind=kind,
                bindings=dict(candidate.bindings or {}),
                note="ai_bench_matmul",
                score=float(score),
                score_reason=",".join(reasons),
                cluster=cluster,
                portability_note=portability,
            )
        )
    ranked.sort(key=lambda c: (-float(c.score or 0.0), str(c.kernel_kind)))
    ranked = ranked[:b]
    substitutions: list[dict[str, Any]] = []
    if not async_ok:
        substitutions.append({"from": "ai_matmul.async_prefetch", "to": "ai_matmul.sync_mma", "reason": "incomplete async evidence"})
    return BackendPlan(
        kernel="ai_bench_matmul",
        source_oracle=dict(source_oracle or {}),
        hardware_model=hardware_model.to_json_dict(),
        selected_modules=selected_modules,
        module_edges=selected_edges,
        param_space={
            "kernel_kind": ["matmul_mma_tf32_v2", "matmul_mma_tf32_v1", "matmul_mma_tf32_global_v1", "matmul_tile_v2"],
            "MMA_BM": [bm],
            "MMA_BN": [bn],
            "MMA_BK": [bk],
            "MMA_ASYNC_COPY": ([1] if async_ok else []),
        },
        constraints=["M > 0", "N > 0", "K > 0", "K % 8 == 0"],
        substitutions=substitutions,
        candidates=ranked,
        notes=[f"goals={sorted(goal_tags)}", f"cluster={cluster}", f"async_ok={async_ok}"],
    )


__all__ = ["plan_ai_bench_matmul"]
