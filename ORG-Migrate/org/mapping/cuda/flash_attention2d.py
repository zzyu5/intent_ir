from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_allowed_ints_normalized, union_dim_allowed
from org.schema import OrgDoc


def _norm_tag(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    return "_".join(s.replace("-", "_").split())


def _collect_tags(org: OrgDoc) -> set[str]:
    tags: set[str] = set()
    for n in list(getattr(org, "nodes", []) or []):
        nt = _norm_tag(getattr(n, "node_type", ""))
        if nt:
            tags.add(nt)
        for t in list(getattr(n, "why", []) or []):
            nt = _norm_tag(t)
            if nt:
                tags.add(nt)
        for t in list(getattr(n, "how", []) or []):
            nt = _norm_tag(t)
            if nt:
                tags.add(nt)
    return tags


def _coerce_int(x: Any) -> int | None:
    try:
        v = int(x)
    except Exception:
        return None
    return v


def _require_dim(bindings: Mapping[str, Any], key: str) -> int:
    v = _coerce_int(bindings.get(key))
    if v is None:
        raise ValueError(f"missing shape_bindings[{key!r}]")
    return int(v)


@dataclass(frozen=True)
class _AsyncGuardrails:
    ok: bool
    reason: str = ""


def _async_copy_guardrails(
    *,
    kv_ctx: int,
    head_dim: int,
    block_kv: int,
    score_warps: int,
) -> _AsyncGuardrails:
    # Mirrors C++ plugin guardrails:
    # asyncCopy = (!directKV) && (KV == blockKV) && (HD % 4 == 0) && ((tileVec4 % threads) == 0)
    if kv_ctx != block_kv:
        return _AsyncGuardrails(ok=False, reason="KV_CTX != ATTN_BLOCK_KV (requires single tile, no tail)")
    if (head_dim % 4) != 0:
        return _AsyncGuardrails(ok=False, reason="HEAD_DIM % 4 != 0 (requires vector<4xf32>)")
    out_warps = 2
    threads = (out_warps + int(score_warps)) * 32
    tile_vec4 = (int(block_kv) * int(head_dim)) // 4
    if tile_vec4 <= 0:
        return _AsyncGuardrails(ok=False, reason="invalid tile_vec4")
    if (tile_vec4 % int(threads)) != 0:
        return _AsyncGuardrails(ok=False, reason="tile_vec4 % threads != 0 (vectorized copy imbalance)")
    return _AsyncGuardrails(ok=True)


def plan_flash_attention2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    target: str,
    budget: int = 32,
    enable_cpp_extras: bool = False,
) -> BackendPlan:
    """
    Deterministic ORG -> CUDA backend plan for `flash_attention2d`.

    Phase-1 scope:
      - variants: attn2d_causal_softmax_v6 / v7
      - dims: ATTN_BLOCK_KV, ATTN_SCORE_WARPS
      - (cpp_plugin extras): FLASH_ATTN_ASYNC_COPY, FLASH_ATTN_DIRECT_GMEM
    """

    b = max(1, int(budget))
    tags = _collect_tags(org)
    dim_allowed_norm = collect_dim_allowed_ints_normalized(org)

    q_ctx = _require_dim(shape_bindings, "Q_CTX")
    kv_ctx = _require_dim(shape_bindings, "KV_CTX")
    head_dim = _require_dim(shape_bindings, "HEAD_DIM")

    block_kv_order = [32, 64, 16]
    score_warps_order = [6, 4, 2]
    allowed_block_kv = union_dim_allowed(
        dim_allowed_norm,
        "ATTN_BLOCK_KV",
        # Common source-level names / synonyms.
        "BLOCK_KV",
        "tile_kv",
        "TILE_KV",
        "kv_tile",
        "kv_block",
    )
    allowed_score_warps = union_dim_allowed(
        dim_allowed_norm,
        "ATTN_SCORE_WARPS",
        "SCORE_WARPS",
        "score_warps",
        "ATTN_WARPS",
        "score_warp_count",
    )

    param_space = {
        "ATTN_BLOCK_KV": list(block_kv_order),
        "ATTN_SCORE_WARPS": list(score_warps_order),
        "kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7"],
    }
    constraints: list[str] = [
        "ATTN_BLOCK_KV in {16,32,64}",
        "ATTN_SCORE_WARPS in {2,4,6}",
        "threads = (2 + ATTN_SCORE_WARPS) * 32 <= 1024",
        "HEAD_DIM == 64",
    ]

    trace: dict[str, Any] = {"substitutions": []}
    selected_variants = ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7"]
    passes = [
        "mechanism_to_variant_selection",
        "param_space_enumeration",
        "constraint_filtering",
        "org_priority_sort",
        "dedupe_clip",
    ]
    modules = [
        BackendModule(
            id="template_v6",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v6"],
            params=["ATTN_BLOCK_KV", "ATTN_SCORE_WARPS"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="template_v7",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v7"],
            params=["ATTN_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="mechanism_online_softmax",
            kind="special_primitive",
            provides=["abstract.streaming_softmax_state"],
            params=["ATTN_BLOCK_KV"],
            constraints=[],
        ),
        BackendModule(
            id="mechanism_scratchpad_staging",
            kind="staging",
            provides=["abstract.scratchpad_staging"],
            params=["ATTN_BLOCK_KV"],
            constraints=[],
        ),
    ]
    module_edges: list[BackendModuleEdge] = [
        BackendModuleEdge(src="template_v6", dst="mechanism_online_softmax", edge_type="uses"),
        BackendModuleEdge(src="template_v6", dst="mechanism_scratchpad_staging", edge_type="uses"),
        BackendModuleEdge(src="template_v7", dst="mechanism_online_softmax", edge_type="uses"),
        BackendModuleEdge(src="template_v7", dst="mechanism_scratchpad_staging", edge_type="uses"),
    ]

    if head_dim != 64 or q_ctx <= 0 or kv_ctx <= 0:
        trace["substitutions"].append(
            {
                "from": "org.flash_attention2d",
                "to": "backend.skip",
                "reason": f"unsupported dims: Q_CTX={q_ctx} KV_CTX={kv_ctx} HEAD_DIM={head_dim} (expects HEAD_DIM==64)",
            }
        )
        return BackendPlan(
            kernel="flash_attention2d",
            target=str(target),
            hardware={"q_ctx": q_ctx, "kv_ctx": kv_ctx, "head_dim": head_dim},
            modules=list(modules),
            module_edges=list(module_edges),
            passes=list(passes),
            selected_variants=selected_variants,
            param_space=param_space,
            constraints=constraints,
            trace=trace,
            candidates=[],
            meta={"org_tags": sorted(tags)},
        )

    want_async = bool(
        tags
        & {
            "overlap_pipeline",
            "double_buffering",
            "async_prefetch",
            "pipeline_overlap",
            "hide_memory_latency",
        }
    )
    prefer_v7 = bool(tags & {"avoid_recompute", "scores_cached", "score_cache", "cache_scores", "avoid_dot_recompute"})
    want_resident = bool(tags & {"resident_working_set", "iterate_in_scratchpad", "iterate_in_local_scratchpad"})

    if want_async and not enable_cpp_extras:
        trace["substitutions"].append(
            {
                "from": "abstract.async_copy",
                "to": "sync_load",
                "reason": "cpp_extras_disabled (INTENTIR_COMPILER_STACK is not cpp_plugin)",
            }
        )
        modules.append(
            BackendModule(
                id="mechanism_sync_load",
                kind="overlap_pipeline",
                provides=["abstract.sync_load"],
                params=[],
                constraints=[],
                attrs={"reason": "cpp_extras_disabled"},
            )
        )
        module_edges.append(BackendModuleEdge(src="template_v6", dst="mechanism_sync_load", edge_type="optional"))
        module_edges.append(BackendModuleEdge(src="template_v7", dst="mechanism_sync_load", edge_type="optional"))

    # Block KV preference: prioritize 32/64 when residency is a goal; otherwise keep stable default.
    block_kv_vals = [int(x) for x in block_kv_order if int(x) <= int(kv_ctx)]
    if allowed_block_kv:
        filtered = [int(x) for x in block_kv_vals if int(x) in allowed_block_kv]
        if filtered:
            block_kv_vals = filtered
            param_space["ATTN_BLOCK_KV"] = [int(x) for x in block_kv_order if int(x) in allowed_block_kv]
        else:
            trace["substitutions"].append(
                {
                    "from": "org.dim.ATTN_BLOCK_KV",
                    "to": "backend.default",
                    "reason": "org allowed set has no feasible values (filtered by KV_CTX)",
                    "detail": {"allowed": sorted({int(x) for x in allowed_block_kv}), "KV_CTX": int(kv_ctx)},
                }
            )
    if not block_kv_vals:
        block_kv_vals = [16]

    score_warps_vals = list(score_warps_order)
    if allowed_score_warps:
        filtered = [int(x) for x in score_warps_vals if int(x) in allowed_score_warps]
        if filtered:
            score_warps_vals = filtered
            param_space["ATTN_SCORE_WARPS"] = [int(x) for x in score_warps_order if int(x) in allowed_score_warps]
        else:
            trace["substitutions"].append(
                {
                    "from": "org.dim.ATTN_SCORE_WARPS",
                    "to": "backend.default",
                    "reason": "org allowed set has no supported values",
                    "detail": {"allowed": sorted({int(x) for x in allowed_score_warps})},
                }
            )

    candidates_v7: list[BackendCandidate] = []
    for bk in block_kv_vals:
        candidates_v7.append(
            BackendCandidate(
                kernel_kind="attn2d_causal_softmax_v7",
                bindings={"ATTN_BLOCK_KV": int(bk)},
                note=("prefer_v7" if prefer_v7 else "baseline"),
            )
        )

    candidates_v6: list[BackendCandidate] = []
    for bk in block_kv_vals:
        for sw in score_warps_vals:
            candidates_v6.append(
                BackendCandidate(
                    kernel_kind="attn2d_causal_softmax_v6",
                    bindings={"ATTN_BLOCK_KV": int(bk), "ATTN_SCORE_WARPS": int(sw)},
                    note=("resident_ws" if want_resident else "baseline"),
                )
            )

    candidates_cpp_async: list[BackendCandidate] = []
    async_reasons: dict[tuple[int, int], str] = {}
    if enable_cpp_extras and want_async:
        # v7: default score_warps=6 unless explicitly overridden.
        for bk in block_kv_vals:
            g = _async_copy_guardrails(kv_ctx=kv_ctx, head_dim=head_dim, block_kv=bk, score_warps=6)
            if not g.ok:
                async_reasons[(int(bk), 6)] = g.reason
                continue
            candidates_cpp_async.append(
                BackendCandidate(
                    kernel_kind="attn2d_causal_softmax_v7",
                    bindings={"ATTN_BLOCK_KV": int(bk), "FLASH_ATTN_ASYNC_COPY": 1},
                    note="cpp_plugin_async_copy",
                )
            )

        # v6: keep a minimal async slice (score_warps=6) to avoid ballooning candidates.
        for bk in block_kv_vals:
            sw = 6
            g = _async_copy_guardrails(kv_ctx=kv_ctx, head_dim=head_dim, block_kv=bk, score_warps=sw)
            if not g.ok:
                async_reasons[(int(bk), int(sw))] = g.reason
                continue
            candidates_cpp_async.append(
                BackendCandidate(
                    kernel_kind="attn2d_causal_softmax_v6",
                    bindings={
                        "ATTN_BLOCK_KV": int(bk),
                        "ATTN_SCORE_WARPS": int(sw),
                        "FLASH_ATTN_ASYNC_COPY": 1,
                    },
                    note="cpp_plugin_async_copy",
                )
            )
        if not candidates_cpp_async:
            trace["substitutions"].append(
                {
                    "from": "abstract.async_copy",
                    "to": "sync_load",
                    "reason": "no candidates satisfy async-copy guardrails",
                    "detail": {"examples": list(list(async_reasons.items())[:3])},
                }
            )
            modules.append(
                BackendModule(
                    id="mechanism_sync_load",
                    kind="overlap_pipeline",
                    provides=["abstract.sync_load"],
                    params=[],
                    constraints=[],
                    attrs={"reason": "async_guardrails_reject"},
                )
            )
            module_edges.append(BackendModuleEdge(src="template_v6", dst="mechanism_sync_load", edge_type="optional"))
            module_edges.append(BackendModuleEdge(src="template_v7", dst="mechanism_sync_load", edge_type="optional"))
        else:
            modules.append(
                BackendModule(
                    id="mechanism_async_copy",
                    kind="overlap_pipeline",
                    provides=["abstract.async_copy"],
                    params=["FLASH_ATTN_ASYNC_COPY"],
                    constraints=["KV_CTX == ATTN_BLOCK_KV", "HEAD_DIM % 4 == 0"],
                    attrs={"note": "cpp_plugin_only"},
                )
            )
            module_edges.append(BackendModuleEdge(src="template_v6", dst="mechanism_async_copy", edge_type="optional"))
            module_edges.append(BackendModuleEdge(src="template_v7", dst="mechanism_async_copy", edge_type="optional"))

    candidates_cpp_direct: list[BackendCandidate] = []
    if enable_cpp_extras and want_resident:
        # Fallback exploration: direct global KV loads (no shared staging).
        # Keep this minimal to avoid ballooning the candidate set.
        bk = int(block_kv_vals[0])
        sw = int(score_warps_vals[0])
        direct_kind = "attn2d_causal_softmax_v7" if prefer_v7 else "attn2d_causal_softmax_v6"
        direct_bindings: dict[str, int] = {"ATTN_BLOCK_KV": int(bk), "FLASH_ATTN_DIRECT_GMEM": 1}
        if direct_kind == "attn2d_causal_softmax_v6":
            direct_bindings["ATTN_SCORE_WARPS"] = int(sw)
        candidates_cpp_direct.append(
            BackendCandidate(
                kernel_kind=direct_kind,
                bindings=direct_bindings,
                note="cpp_plugin_direct_gmem",
            )
        )
        modules.append(
            BackendModule(
                id="mechanism_direct_gmem_kv",
                kind="staging",
                provides=["abstract.direct_gmem_kv"],
                params=["FLASH_ATTN_DIRECT_GMEM"],
                constraints=[],
                attrs={"note": "cpp_plugin_only"},
            )
        )
        module_edges.append(BackendModuleEdge(src="template_v6", dst="mechanism_direct_gmem_kv", edge_type="optional"))
        module_edges.append(BackendModuleEdge(src="template_v7", dst="mechanism_direct_gmem_kv", edge_type="optional"))

    # Priority ordering.
    ordered: list[BackendCandidate] = []
    if want_async and candidates_cpp_async:
        if prefer_v7:
            ordered.extend([c for c in candidates_cpp_async if c.kernel_kind == "attn2d_causal_softmax_v7"])
            ordered.extend([c for c in candidates_cpp_async if c.kernel_kind != "attn2d_causal_softmax_v7"])
        else:
            ordered.extend([c for c in candidates_cpp_async if c.kernel_kind == "attn2d_causal_softmax_v6"])
            ordered.extend([c for c in candidates_cpp_async if c.kernel_kind != "attn2d_causal_softmax_v6"])

    if prefer_v7:
        ordered.extend(candidates_v7)
        ordered.extend(candidates_v6)
    else:
        ordered.extend(candidates_v6)
        ordered.extend(candidates_v7)

    ordered.extend(candidates_cpp_direct)

    # Deduplicate and clip to budget.
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    final: list[BackendCandidate] = []
    for c in ordered:
        key = (str(c.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(c.bindings or {}).items())))
        if key in seen:
            continue
        seen.add(key)
        final.append(c)
        if len(final) >= b:
            break

    return BackendPlan(
        kernel="flash_attention2d",
        target=str(target),
        hardware={"q_ctx": int(q_ctx), "kv_ctx": int(kv_ctx), "head_dim": int(head_dim)},
        modules=list(modules),
        module_edges=list(module_edges),
        passes=list(passes),
        selected_variants=selected_variants,
        param_space=param_space,
        constraints=constraints,
        trace=trace,
        candidates=final,
        meta={
            "org_tags": sorted(tags),
            "want_async": bool(want_async),
            "prefer_v7": bool(prefer_v7),
            "want_resident": bool(want_resident),
            "enable_cpp_extras": bool(enable_cpp_extras),
        },
    )


__all__ = ["plan_flash_attention2d"]
