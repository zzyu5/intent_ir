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
        if preferred is not None and int(preferred) in allowed_set and int(preferred) not in vals:
            vals.insert(0, int(preferred))
    if preferred is not None and int(preferred) in vals:
        vals = [int(preferred)] + [int(x) for x in vals if int(x) != int(preferred)]
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


def plan_matmul_fused_epilogue2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    budget: int = 32,
) -> BackendPlan:
    b = max(1, int(budget))
    m_dim = _require_dim(shape_bindings, "M")
    n_dim = _require_dim(shape_bindings, "N")
    k_dim = _require_dim(shape_bindings, "K")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)

    modules, module_edges, passes = matmul_fused_epilogue2d_catalog(hardware_model)
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
            notes=[f"goals={sorted(goal_tags)}"],
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

    param_space = {
        "kernel_kind": ["matmul_mma_tf32_v1", "matmul_tile_v2", "matmul_tile_v1"],
        "MMA_BM": list(bm_values),
        "MMA_BN": list(bn_values),
        "MMA_BK": list(bk_values),
        "MMA_ASYNC_COPY": ([1] if hardware_model.supports_async_copy and "prefetch_pipeline" in {m.id for m in selected_modules} else []),
    }
    constraints = [
        "MMA_BM%16==0",
        "MMA_BN%16==0",
        "MMA_BK%8==0",
        "fused_epilogue_avoid_writeback preserved",
    ]

    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    exact_bindings = dict(source_bindings)
    want_async = any(m.id == "prefetch_pipeline" for m in selected_modules)
    want_mma = any(m.id == "mma_core" for m in selected_modules)
    if exact_kind:
        preserve_notes.append(f"source_oracle_variant={exact_kind}")
    if "mma_core" in mechanism_tags or "mma_acceleration" in goal_tags:
        preserve_notes.append("preserve:mma_core")
    if "epilogue_fused_writeback" in mechanism_tags or "fused_epilogue_avoid_writeback" in goal_tags:
        preserve_notes.append("preserve:epilogue_fused_writeback")
    if want_async:
        preserve_notes.append("preserve:prefetch_pipeline")
    ordered: list[BackendCandidate] = []
    if exact_kind in {"matmul_mma_tf32_v1", "matmul_tile_v2", "matmul_tile_v1"}:
        ordered.append(BackendCandidate(kernel_kind=exact_kind, bindings=exact_bindings, note="source_exact"))

    if want_mma:
        async_ok = False
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
                    ordered.append(
                        BackendCandidate(
                            kernel_kind="matmul_mma_tf32_v1",
                            bindings={"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk)},
                            note="mma_neighbor",
                        )
                    )
                    if want_async:
                        ok, reason = _mma_async_guardrails(bm=int(bm), bn=int(bn), bk=int(bk), threads=int(threads))
                        if ok:
                            async_ok = True
                            ordered.insert(
                                0,
                                BackendCandidate(
                                    kernel_kind="matmul_mma_tf32_v1",
                                    bindings={
                                        "MMA_BM": int(bm),
                                        "MMA_BN": int(bn),
                                        "MMA_BK": int(bk),
                                        "MMA_ASYNC_COPY": 1,
                                    },
                                    note="latency_hiding_async",
                                ),
                            )
                        else:
                            substitutions.append(
                                {
                                    "from": "matmul.prefetch_pipeline",
                                    "to": "matmul.sync_prefetch",
                                    "reason": reason,
                                    "detail": {"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk)},
                                }
                            )
        if (source_bindings.get("MMA_ASYNC_COPY") or 0) == 1 and want_async and not async_ok:
            substitutions.append(
                {
                    "from": "source.prefetch_pipeline",
                    "to": "matmul.sync_prefetch",
                    "reason": "source async MMA path has no valid target realization",
                }
            )
            preserve_notes.append("replace:prefetch_pipeline->sync_prefetch")

    ordered.append(BackendCandidate(kernel_kind="matmul_tile_v2", bindings={}, note="tile_baseline"))
    ordered.append(BackendCandidate(kernel_kind="matmul_tile_v1", bindings={}, note="tile_fallback"))

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
            *preserve_notes,
        ],
    )


__all__ = ["plan_matmul_fused_epilogue2d"]
