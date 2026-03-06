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
            tt = _norm_tag(t)
            if tt:
                tags.add(tt)
        for t in list(getattr(n, "how", []) or []):
            tt = _norm_tag(t)
            if tt:
                tags.add(tt)
    return tags


def _coerce_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _require_dim(bindings: Mapping[str, Any], key: str) -> int:
    v = _coerce_int(bindings.get(key))
    if v is None:
        raise ValueError(f"missing shape_bindings[{key!r}]")
    return int(v)


@dataclass(frozen=True)
class _AsyncGuardrails:
    ok: bool
    reason: str = ""


def _mma_v1_async_guardrails(*, bm: int, bn: int, bk: int, threads: int) -> _AsyncGuardrails:
    # Mirrors matmul_mma_tf32_v1 `async_copy_enabled` conditions in lowering:
    #   async_copy = requested && vec4_copy && (tileA4 % threads == 0) && (tileB4 % threads == 0)
    if threads <= 0:
        return _AsyncGuardrails(ok=False, reason="invalid threads")
    vec_copy = (
        (int(bk) % 4) == 0
        and (int(bn) % 4) == 0
        and ((int(bm) * int(bk)) % 4) == 0
        and ((int(bk) * int(bn)) % 4) == 0
    )
    if not vec_copy:
        return _AsyncGuardrails(ok=False, reason="vec4_copy_not_eligible")
    tile_a4 = (int(bm) * int(bk)) // 4
    tile_b4 = (int(bk) * int(bn)) // 4
    if tile_a4 <= 0 or tile_b4 <= 0:
        return _AsyncGuardrails(ok=False, reason="invalid_vec4_tile")
    if (tile_a4 % int(threads)) != 0 or (tile_b4 % int(threads)) != 0:
        return _AsyncGuardrails(ok=False, reason="tile_vec4 % threads != 0 (imbalance)")
    return _AsyncGuardrails(ok=True)


def plan_matmul_fused_epilogue2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    target: str,
    budget: int = 32,
) -> BackendPlan:
    """
    Deterministic ORG -> CUDA backend plan for `matmul_fused_epilogue2d`.

    Scope:
      - always include tiled baselines: matmul_tile_v2 (+ matmul_tile_v1 fallback)
      - generate matmul_mma_tf32_v1 candidates (MMA_BM/MMA_BN/MMA_BK; optional MMA_ASYNC_COPY=1)

    Notes:
      - `matmul_mma_tf32_global_v1` is NOT included because it rejects fused epilogues
        (bias/masks/relu) in lowering.
    """

    b = max(1, int(budget))
    tags = _collect_tags(org)
    dim_allowed_norm = collect_dim_allowed_ints_normalized(org)

    m_dim = _require_dim(shape_bindings, "M")
    n_dim = _require_dim(shape_bindings, "N")
    k_dim = _require_dim(shape_bindings, "K")

    param_space: dict[str, Any] = {
        "kernel_kind": ["matmul_mma_tf32_v1", "matmul_tile_v2", "matmul_tile_v1"],
        "MMA_BM": [32, 64],
        "MMA_BN": [32, 16],
        "MMA_BK": [32, 16, 64],
        "MMA_ASYNC_COPY": [1],
    }
    constraints: list[str] = [
        "MMA_BM%16==0, MMA_BN%16==0, MMA_BK%8==0",
        "M%MMA_BM==0, N%MMA_BN==0, K%MMA_BK==0, K%8==0",
        "threads=((MMA_BM/16)*(MMA_BN/16))*32 <= 1024",
        "matmul_mma_tf32_v1 async-copy requires vec4 copy and tile_vec4%threads==0",
    ]

    trace: dict[str, Any] = {"substitutions": []}
    selected_variants = ["matmul_mma_tf32_v1", "matmul_tile_v2", "matmul_tile_v1"]
    passes = [
        "mechanism_to_variant_selection",
        "param_space_enumeration",
        "constraint_filtering",
        "org_priority_sort",
        "dedupe_clip",
    ]
    modules: list[BackendModule] = [
        BackendModule(
            id="mechanism_mma_tf32",
            kind="special_primitive",
            provides=["abstract.matrix_primitive.mma_tf32"],
            params=["MMA_BM", "MMA_BN", "MMA_BK"],
            constraints=["MMA_BM%16==0", "MMA_BN%16==0", "MMA_BK%8==0"],
        ),
        BackendModule(
            id="mechanism_async_copy",
            kind="overlap_pipeline",
            provides=["abstract.async_copy"],
            params=["MMA_ASYNC_COPY"],
            constraints=["vec4_copy_eligible", "tile_vec4%threads==0"],
        ),
        BackendModule(
            id="mechanism_fused_epilogue",
            kind="special_primitive",
            provides=["abstract.fused_epilogue"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="template_mma_v1",
            kind="template",
            provides=["backend.kernel_kind.matmul_mma_tf32_v1"],
            params=["MMA_BM", "MMA_BN", "MMA_BK", "MMA_ASYNC_COPY"],
            constraints=[],
        ),
        BackendModule(
            id="template_tile_v2",
            kind="template",
            provides=["backend.kernel_kind.matmul_tile_v2"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="template_tile_v1",
            kind="template",
            provides=["backend.kernel_kind.matmul_tile_v1"],
            params=[],
            constraints=[],
        ),
    ]
    module_edges: list[BackendModuleEdge] = [
        BackendModuleEdge(src="template_mma_v1", dst="mechanism_mma_tf32", edge_type="uses"),
        BackendModuleEdge(src="template_mma_v1", dst="mechanism_async_copy", edge_type="optional"),
        BackendModuleEdge(src="template_mma_v1", dst="mechanism_fused_epilogue", edge_type="uses"),
        BackendModuleEdge(src="template_tile_v2", dst="mechanism_fused_epilogue", edge_type="uses"),
        BackendModuleEdge(src="template_tile_v1", dst="mechanism_fused_epilogue", edge_type="uses"),
    ]

    if m_dim <= 0 or n_dim <= 0 or k_dim <= 0 or (k_dim % 8) != 0:
        trace["substitutions"].append(
            {
                "from": "org.matmul_fused_epilogue2d",
                "to": "backend.skip",
                "reason": f"unsupported dims: M={m_dim} N={n_dim} K={k_dim} (requires K%8==0 and positive dims)",
            }
        )
        return BackendPlan(
            kernel="matmul_fused_epilogue2d",
            target=str(target),
            hardware={"M": int(m_dim), "N": int(n_dim), "K": int(k_dim)},
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
    want_mma = bool(tags & {"special_primitive", "tensor_core", "mma", "wmma", "dot"})

    bm_vals = [32, 64]
    bn_vals = [32, 16]
    bk_vals = [32, 16, 64]

    allowed_bm = union_dim_allowed(dim_allowed_norm, "MMA_BM", "BLOCK_M", "BM", "tile_m", "TILE_M")
    allowed_bn = union_dim_allowed(dim_allowed_norm, "MMA_BN", "BLOCK_N", "BN", "tile_n", "TILE_N")
    allowed_bk = union_dim_allowed(dim_allowed_norm, "MMA_BK", "BLOCK_K", "BK", "tile_k", "TILE_K")

    if allowed_bm:
        bm_vals = [int(x) for x in bm_vals if int(x) in allowed_bm]
        if not bm_vals:
            trace["substitutions"].append(
                {
                    "from": "org.dim.MMA_BM",
                    "to": "backend.default",
                    "reason": "org allowed set has no supported values",
                    "detail": {"allowed": sorted({int(x) for x in allowed_bm})},
                }
            )
            bm_vals = [32, 64]
    if allowed_bn:
        bn_vals = [int(x) for x in bn_vals if int(x) in allowed_bn]
        if not bn_vals:
            trace["substitutions"].append(
                {
                    "from": "org.dim.MMA_BN",
                    "to": "backend.default",
                    "reason": "org allowed set has no supported values",
                    "detail": {"allowed": sorted({int(x) for x in allowed_bn})},
                }
            )
            bn_vals = [32, 16]
    if allowed_bk:
        bk_vals = [int(x) for x in bk_vals if int(x) in allowed_bk]
        if not bk_vals:
            trace["substitutions"].append(
                {
                    "from": "org.dim.MMA_BK",
                    "to": "backend.default",
                    "reason": "org allowed set has no supported values",
                    "detail": {"allowed": sorted({int(x) for x in allowed_bk})},
                }
            )
            bk_vals = [32, 16, 64]

    param_space["MMA_BM"] = list(bm_vals)
    param_space["MMA_BN"] = list(bn_vals)
    param_space["MMA_BK"] = list(bk_vals)

    mma_async: list[BackendCandidate] = []
    mma_sync: list[BackendCandidate] = []
    async_rejects: list[str] = []
    for bm in bm_vals:
        if (m_dim % int(bm)) != 0 or (int(bm) % 16) != 0:
            continue
        for bn in bn_vals:
            if (n_dim % int(bn)) != 0 or (int(bn) % 16) != 0:
                continue
            warps = (int(bm) // 16) * (int(bn) // 16)
            threads = int(warps) * 32
            if warps <= 0 or warps > 32 or threads <= 0 or threads > 1024:
                continue
            for bk in bk_vals:
                if (k_dim % int(bk)) != 0 or (int(bk) % 8) != 0:
                    continue
                g = _mma_v1_async_guardrails(bm=int(bm), bn=int(bn), bk=int(bk), threads=int(threads))
                if g.ok:
                    mma_async.append(
                        BackendCandidate(
                            kernel_kind="matmul_mma_tf32_v1",
                            bindings={"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk), "MMA_ASYNC_COPY": 1},
                            note=("want_async" if want_async else "mma_async"),
                        )
                    )
                else:
                    async_rejects.append(f"BM={bm} BN={bn} BK={bk}: {g.reason}")
                mma_sync.append(
                    BackendCandidate(
                        kernel_kind="matmul_mma_tf32_v1",
                        bindings={"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk)},
                        note=("mma_sync" if want_mma else "baseline"),
                    )
                )

    if want_async and not mma_async:
        trace["substitutions"].append(
            {
                "from": "abstract.async_copy",
                "to": "sync_copy",
                "reason": "no matmul_mma_tf32_v1 candidates satisfy async-copy guardrails",
                "detail": {"examples": async_rejects[:3]},
            }
        )

    tiled_baseline: list[BackendCandidate] = [
        BackendCandidate(kernel_kind="matmul_tile_v2", bindings={}, note="tile_baseline"),
        BackendCandidate(kernel_kind="matmul_tile_v1", bindings={}, note="tile_fallback"),
    ]

    ordered: list[BackendCandidate] = []
    if want_async or want_mma:
        ordered.extend(mma_async)
        ordered.extend(mma_sync)
        ordered.extend(tiled_baseline)
    else:
        ordered.extend(tiled_baseline[:1])
        ordered.extend(mma_async)
        ordered.extend(mma_sync)
        ordered.extend(tiled_baseline[1:])

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
        kernel="matmul_fused_epilogue2d",
        target=str(target),
        hardware={"M": int(m_dim), "N": int(n_dim), "K": int(k_dim)},
        modules=list(modules),
        module_edges=list(module_edges),
        passes=list(passes),
        selected_variants=selected_variants,
        param_space=param_space,
        constraints=constraints,
        trace=trace,
        candidates=final,
        meta={"org_tags": sorted(tags), "want_async": bool(want_async), "want_mma": bool(want_mma)},
    )


__all__ = ["plan_matmul_fused_epilogue2d"]
