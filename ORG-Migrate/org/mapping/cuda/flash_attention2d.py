from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import flash_attention2d_catalog
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


def _async_copy_guardrails(*, kv_ctx: int, head_dim: int, block_kv: int, score_warps: int) -> tuple[bool, str]:
    if kv_ctx != block_kv:
        return False, "KV_CTX != ATTN_BLOCK_KV"
    if (head_dim % 4) != 0:
        return False, "HEAD_DIM % 4 != 0"
    threads = (2 + int(score_warps)) * 32
    tile_vec4 = (int(block_kv) * int(head_dim)) // 4
    if threads <= 0 or tile_vec4 <= 0:
        return False, "invalid tile/threads"
    if (tile_vec4 % int(threads)) != 0:
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
    selected_ids = {"q_resident_state", "kv_tile_stage", "output_accumulator"}
    substitutions: list[dict[str, Any]] = []

    if "streaming_softmax_state" in goal_tags or "online_softmax_reduce" in mechanism_tags:
        selected_ids.add("online_softmax_reduce")

    if "latency_hiding" in goal_tags or "prefetch_pipeline" in mechanism_tags:
        if hardware_model.supports_async_copy:
            selected_ids.add("prefetch_pipeline")
        else:
            substitutions.append(
                {
                    "from": "flash.prefetch_pipeline",
                    "to": "flash.sync_prefetch",
                    "reason": "hardware_model.supports_async_copy = false",
                }
            )

    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    if source_kind == "attn2d_causal_softmax_v7":
        selected_ids.add("backend_v7")
        selected_ids.add("backend_v6")
    else:
        selected_ids.add("backend_v6")
        selected_ids.add("backend_v7")

    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    if source_kind == "attn2d_causal_softmax_v7" and "backend_v7" not in selected_ids:
        substitutions.append(
            {
                "from": "source.variant.v7",
                "to": "backend_v6",
                "reason": "source variant v7 not preserved in selected modules",
            }
        )
    return selected_modules, selected_edges, substitutions


def _candidate_key(candidate: BackendCandidate) -> tuple[str, tuple[tuple[str, int], ...]]:
    return str(candidate.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(candidate.bindings or {}).items()))


def plan_flash_attention2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    budget: int = 32,
) -> BackendPlan:
    b = max(1, int(budget))
    q_ctx = _require_dim(shape_bindings, "Q_CTX")
    kv_ctx = _require_dim(shape_bindings, "KV_CTX")
    head_dim = _require_dim(shape_bindings, "HEAD_DIM")
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)

    modules, module_edges, passes = flash_attention2d_catalog(hardware_model)
    selected_modules, selected_edges, substitutions = _selected_modules(
        modules=modules,
        module_edges=module_edges,
        goal_tags=goal_tags,
        mechanism_tags=mechanism_tags,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
    )

    if head_dim != 64 or q_ctx <= 0 or kv_ctx <= 0:
        substitutions.append(
            {
                "from": "flash_attention2d",
                "to": "backend.skip",
                "reason": f"unsupported dims: Q_CTX={q_ctx} KV_CTX={kv_ctx} HEAD_DIM={head_dim}",
            }
        )
        return BackendPlan(
            kernel="flash_attention2d",
            source_oracle=dict(source_oracle or {}),
            hardware_model=hardware_model.to_json_dict(),
            selected_modules=selected_modules,
            module_edges=selected_edges,
            param_space={"kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7"]},
            constraints=["HEAD_DIM == 64"],
            substitutions=substitutions,
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}"],
        )

    dim_candidates_norm = collect_dim_candidate_ints_normalized(org)
    source_bindings = {
        str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()
    }
    preferred_block = _coerce_int(source_bindings.get("ATTN_BLOCK_KV"))
    preferred_score = _coerce_int(source_bindings.get("ATTN_SCORE_WARPS"))
    block_candidates = _ordered_param_values(
        defaults=[32, 64, 16],
        preferred=preferred_block,
        allowed=union_dim_candidate_ints(dim_candidates_norm, "tile_kv", "ATTN_BLOCK_KV", "BLOCK_KV"),
    )
    score_candidates = _ordered_param_values(
        defaults=[6, 4, 2],
        preferred=preferred_score,
        allowed=union_dim_candidate_ints(dim_candidates_norm, "score_warps", "ATTN_SCORE_WARPS", "SCORE_WARPS"),
    )
    block_candidates = [int(x) for x in block_candidates if int(x) <= int(kv_ctx)]
    if not block_candidates:
        block_candidates = [16]
        substitutions.append(
            {
                "from": "tile_kv",
                "to": "default_block_kv",
                "reason": "no ORG/source candidate fits KV_CTX",
            }
        )

    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    exact_bindings = {str(k): int(v) for k, v in source_bindings.items()}
    want_pipeline = "latency_hiding" in goal_tags or "prefetch_pipeline" in mechanism_tags

    param_space = {
        "kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7"],
        "ATTN_BLOCK_KV": list(block_candidates),
        "ATTN_SCORE_WARPS": list(score_candidates),
        "FLASH_ATTN_ASYNC_COPY": ([1] if want_pipeline and hardware_model.supports_async_copy else []),
    }
    constraints = [
        "HEAD_DIM == 64",
        "ATTN_BLOCK_KV <= KV_CTX",
        "ATTN_SCORE_WARPS in {2,4,6}",
        "resident_working_set preserved",
        "streaming_softmax_state preserved",
    ]

    ordered: list[BackendCandidate] = []
    if exact_kind in {"attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7"}:
        ordered.append(BackendCandidate(kernel_kind=exact_kind, bindings=dict(exact_bindings), note="source_exact"))

    for bk in block_candidates:
        ordered.append(
            BackendCandidate(
                kernel_kind="attn2d_causal_softmax_v7",
                bindings={"ATTN_BLOCK_KV": int(bk)},
                note=("goal_mix" if goal_tags else "default"),
            )
        )
    for bk in block_candidates:
        for sw in score_candidates:
            ordered.append(
                BackendCandidate(
                    kernel_kind="attn2d_causal_softmax_v6",
                    bindings={"ATTN_BLOCK_KV": int(bk), "ATTN_SCORE_WARPS": int(sw)},
                    note=("goal_mix" if goal_tags else "default"),
                )
            )

    if want_pipeline and hardware_model.supports_async_copy:
        async_candidates: list[BackendCandidate] = []
        async_ok = False
        for bk in block_candidates:
            sw = preferred_score if preferred_score is not None else score_candidates[0]
            ok, reason = _async_copy_guardrails(kv_ctx=kv_ctx, head_dim=head_dim, block_kv=int(bk), score_warps=int(sw))
            if ok:
                async_ok = True
                async_candidates.append(
                    BackendCandidate(
                        kernel_kind="attn2d_causal_softmax_v7",
                        bindings={"ATTN_BLOCK_KV": int(bk), "FLASH_ATTN_ASYNC_COPY": 1},
                        note="latency_hiding_async",
                    )
                )
            else:
                substitutions.append(
                    {
                        "from": "flash.prefetch_pipeline",
                        "to": "flash.sync_prefetch",
                        "reason": reason,
                        "detail": {"ATTN_BLOCK_KV": int(bk), "ATTN_SCORE_WARPS": int(sw)},
                    }
                )
        ordered = async_candidates + ordered
        if (source_bindings.get("FLASH_ATTN_ASYNC_COPY") or 0) == 1 and not async_ok:
            substitutions.append(
                {
                    "from": "source.prefetch_pipeline",
                    "to": "flash.sync_prefetch",
                    "reason": "source async-copy candidate has no valid target realization",
                }
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
        kernel="flash_attention2d",
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
        ],
    )


__all__ = ["plan_flash_attention2d"]
