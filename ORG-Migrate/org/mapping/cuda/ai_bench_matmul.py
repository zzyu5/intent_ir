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


def _mma_v2_async_guardrails(
    *,
    bm: int,
    bn: int,
    bk: int,
    threads: int,
) -> _AsyncGuardrails:
    # Mirrors both python and cpp_plugin lowering constraints for matmul_mma_tf32_v2.
    # - static shared <= 48KiB
    # - vectorized (vec4) shared copy eligibility
    static_shared_bytes = 8 * int(bk) * (int(bm) + int(bn))
    if static_shared_bytes > (48 * 1024):
        return _AsyncGuardrails(ok=False, reason="static_shared_bytes > 48KiB")

    vec_copy = (int(bk) % 4) == 0 and (int(bn) % 4) == 0 and ((int(bm) * int(bk)) % 4) == 0 and ((int(bk) * int(bn)) % 4) == 0
    if not vec_copy:
        return _AsyncGuardrails(ok=False, reason="vec4_copy_not_eligible")
    tile_a4 = (int(bm) * int(bk)) // 4
    tile_b4 = (int(bk) * int(bn)) // 4
    if tile_a4 <= 0 or tile_b4 <= 0:
        return _AsyncGuardrails(ok=False, reason="invalid_vec4_tile")
    if (tile_a4 % int(threads)) != 0 or (tile_b4 % int(threads)) != 0:
        return _AsyncGuardrails(ok=False, reason="tile_vec4 % threads != 0 (imbalance)")
    return _AsyncGuardrails(ok=True)


def plan_ai_bench_matmul(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    target: str,
    budget: int = 32,
) -> BackendPlan:
    """
    Deterministic ORG -> CUDA backend plan for `ai_bench_matmul`.

    Scope:
      - kernel kinds: matmul_mma_tf32_v2 (cp.async double-buffer), matmul_mma_tf32_global_v1
      - bindings: MMA_BM, MMA_BN, MMA_BK, MMA_ASYNC_COPY (v2 only)
    """

    b = max(1, int(budget))
    tags = _collect_tags(org)
    dim_allowed_norm = collect_dim_allowed_ints_normalized(org)

    m_dim = _require_dim(shape_bindings, "M")
    n_dim = _require_dim(shape_bindings, "N")
    k_dim = _require_dim(shape_bindings, "K")

    param_space: dict[str, Any] = {
        "kernel_kind": ["matmul_mma_tf32_v2", "matmul_mma_tf32_global_v1"],
        "MMA_BM": [64, 32],
        "MMA_BN": [16, 32],
        "MMA_BK": [32, 64],
        "MMA_ASYNC_COPY": [1],
    }
    constraints: list[str] = [
        "MMA_BM%16==0, MMA_BN%16==0, MMA_BK%8==0",
        "M%MMA_BM==0, N%MMA_BN==0, K%MMA_BK==0, K%8==0",
        "threads=((MMA_BM/16)*(MMA_BN/16))*32 <= 1024",
        "matmul_mma_tf32_v2 requires MMA_ASYNC_COPY=1",
        "matmul_mma_tf32_v2 requires static_shared_bytes=8*MMA_BK*(MMA_BM+MMA_BN) <= 49152",
    ]

    trace: dict[str, Any] = {"substitutions": []}
    selected_variants = ["matmul_mma_tf32_v2", "matmul_mma_tf32_global_v1"]
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
            constraints=["static_shared_bytes <= 49152", "vec4_copy_eligible"],
            attrs={"note": "v2_only"},
        ),
        BackendModule(
            id="template_v2",
            kind="template",
            provides=["backend.kernel_kind.matmul_mma_tf32_v2"],
            params=["MMA_BM", "MMA_BN", "MMA_BK", "MMA_ASYNC_COPY"],
            constraints=["MMA_ASYNC_COPY == 1"],
        ),
        BackendModule(
            id="template_global_v1",
            kind="template",
            provides=["backend.kernel_kind.matmul_mma_tf32_global_v1"],
            params=["MMA_BM", "MMA_BN", "MMA_BK"],
            constraints=[],
        ),
    ]
    module_edges: list[BackendModuleEdge] = [
        BackendModuleEdge(src="template_v2", dst="mechanism_mma_tf32", edge_type="uses"),
        BackendModuleEdge(src="template_v2", dst="mechanism_async_copy", edge_type="uses"),
        BackendModuleEdge(src="template_global_v1", dst="mechanism_mma_tf32", edge_type="uses"),
    ]

    if m_dim <= 0 or n_dim <= 0 or k_dim <= 0 or (k_dim % 8) != 0:
        trace["substitutions"].append(
            {
                "from": "org.ai_bench_matmul",
                "to": "backend.skip",
                "reason": f"unsupported dims: M={m_dim} N={n_dim} K={k_dim} (requires K%8==0 and positive dims)",
            }
        )
        return BackendPlan(
            kernel="ai_bench_matmul",
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

    # Candidate enumeration (small, deterministic).
    bm_vals = [64, 32]
    bn_vals = [16, 32]
    bk_vals = [32, 64]

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
            bm_vals = [64, 32]
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
            bn_vals = [16, 32]
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
            bk_vals = [32, 64]

    param_space["MMA_BM"] = list(bm_vals)
    param_space["MMA_BN"] = list(bn_vals)
    param_space["MMA_BK"] = list(bk_vals)

    candidates_v2: list[BackendCandidate] = []
    v2_rejects: list[str] = []
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
                g = _mma_v2_async_guardrails(bm=int(bm), bn=int(bn), bk=int(bk), threads=int(threads))
                if not g.ok:
                    v2_rejects.append(f"BM={bm} BN={bn} BK={bk}: {g.reason}")
                    continue
                candidates_v2.append(
                    BackendCandidate(
                        kernel_kind="matmul_mma_tf32_v2",
                        bindings={
                            "MMA_BM": int(bm),
                            "MMA_BN": int(bn),
                            "MMA_BK": int(bk),
                            "MMA_ASYNC_COPY": 1,
                        },
                        note=("want_async" if want_async else "baseline"),
                    )
                )

    candidates_global: list[BackendCandidate] = []
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
                candidates_global.append(
                    BackendCandidate(
                        kernel_kind="matmul_mma_tf32_global_v1",
                        bindings={"MMA_BM": int(bm), "MMA_BN": int(bn), "MMA_BK": int(bk)},
                        note="fallback_global_load",
                    )
                )

    if want_async and (not candidates_v2):
        trace["substitutions"].append(
            {
                "from": "abstract.async_copy",
                "to": "global_load",
                "reason": "no matmul_mma_tf32_v2 candidates satisfy guardrails",
                "detail": {"examples": v2_rejects[:3]},
            }
        )

    ordered: list[BackendCandidate] = []
    v2_head = candidates_v2[0] if candidates_v2 else None
    global_head = candidates_global[0] if candidates_global else None
    v2_tail = candidates_v2[1:] if len(candidates_v2) > 1 else []
    global_tail = candidates_global[1:] if len(candidates_global) > 1 else []

    if want_async:
        if v2_head is not None:
            ordered.append(v2_head)
        ordered.extend(v2_tail)
        if global_head is not None:
            ordered.append(global_head)
        ordered.extend(global_tail)
    else:
        if global_head is not None:
            ordered.append(global_head)
        if v2_head is not None:
            ordered.append(v2_head)
        ordered.extend(global_tail)
        ordered.extend(v2_tail)

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
        kernel="ai_bench_matmul",
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
        meta={"org_tags": sorted(tags), "want_async": bool(want_async)},
    )


__all__ = ["plan_ai_bench_matmul"]
