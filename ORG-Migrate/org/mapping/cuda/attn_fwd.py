from __future__ import annotations

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


def _normalize_stack(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in {"cpp", "c++"}:
        return "cpp_plugin"
    return s or "python"


def plan_attn_fwd(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    target: str,
    budget: int = 32,
    compiler_stack: str = "python",
) -> BackendPlan:
    """
    Deterministic ORG -> CUDA backend plan for `_attn_fwd`.

    Stack-aware kernel kinds:
      - python: attn_fwd_tiled_v3 (supports ATTN_FWD_BLOCK_M in {4,8}, ATTN_FWD_BLOCK_KV in {16,32})
      - cpp_plugin: attn_fwd_softmax_v7 (ATTN_FWD_BLOCK_M in {4,8}, ATTN_FWD_BLOCK_KV in {16,32,64})
    """

    b = max(1, int(budget))
    tags = _collect_tags(org)
    dim_allowed_norm = collect_dim_allowed_ints_normalized(org)
    stack = _normalize_stack(compiler_stack)

    q_ctx = _require_dim(shape_bindings, "Q_CTX")
    kv_ctx = _require_dim(shape_bindings, "KV_CTX")
    head_dim = _require_dim(shape_bindings, "HEAD_DIM")

    trace: dict[str, Any] = {"substitutions": []}
    constraints: list[str] = ["HEAD_DIM == 64"]
    passes = [
        "mechanism_to_variant_selection",
        "param_space_enumeration",
        "constraint_filtering",
        "org_priority_sort",
        "dedupe_clip",
    ]
    modules: list[BackendModule] = [
        BackendModule(
            id="mechanism_online_softmax",
            kind="special_primitive",
            provides=["abstract.streaming_softmax_state"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="mechanism_qk_tiling",
            kind="tiling",
            provides=["abstract.tiling"],
            params=["ATTN_FWD_BLOCK_M", "ATTN_FWD_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="mechanism_kv_staging",
            kind="staging",
            provides=["abstract.scratchpad_staging"],
            params=["ATTN_FWD_BLOCK_KV"],
            constraints=[],
        ),
    ]
    module_edges: list[BackendModuleEdge] = []

    if head_dim != 64 or q_ctx <= 0 or kv_ctx <= 0:
        trace["substitutions"].append(
            {
                "from": "org._attn_fwd",
                "to": "backend.skip",
                "reason": f"unsupported dims: Q_CTX={q_ctx} KV_CTX={kv_ctx} HEAD_DIM={head_dim} (expects HEAD_DIM==64)",
            }
        )
        return BackendPlan(
            kernel="_attn_fwd",
            target=str(target),
            hardware={"q_ctx": int(q_ctx), "kv_ctx": int(kv_ctx), "head_dim": int(head_dim)},
            modules=list(modules),
            module_edges=list(module_edges),
            passes=list(passes),
            selected_variants=[],
            param_space={},
            constraints=constraints,
            trace=trace,
            candidates=[],
            meta={"org_tags": sorted(tags), "compiler_stack": str(stack)},
        )

    # Defaults and stack-specific supported sets.
    block_m_order = [8, 4]
    block_kv_order = [32, 16] if stack != "cpp_plugin" else [32, 64, 16]
    supported_block_m = {4, 8}
    supported_block_kv = {16, 32} if stack != "cpp_plugin" else {16, 32, 64}

    # Allow ORG to prune the discrete space via dims.allowed.
    allowed_block_m = union_dim_allowed(
        dim_allowed_norm,
        "ATTN_FWD_BLOCK_M",
        "BLOCK_M",
        "tile_m",
        "TILE_M",
        "BLOCK_Q",
    )
    allowed_block_kv = union_dim_allowed(
        dim_allowed_norm,
        "ATTN_FWD_BLOCK_KV",
        "BLOCK_KV",
        "tile_kv",
        "TILE_KV",
        "BLOCK_N",
    )

    block_m_vals = [int(x) for x in block_m_order if int(x) in supported_block_m and int(x) <= int(q_ctx)]
    if allowed_block_m:
        filtered = [int(x) for x in block_m_vals if int(x) in allowed_block_m]
        if filtered:
            block_m_vals = filtered
        else:
            trace["substitutions"].append(
                {
                    "from": "org.dim.ATTN_FWD_BLOCK_M",
                    "to": "backend.default",
                    "reason": "org allowed set has no supported/feasible values (filtered by Q_CTX)",
                    "detail": {"allowed": sorted({int(x) for x in allowed_block_m}), "Q_CTX": int(q_ctx)},
                }
            )
    if not block_m_vals:
        block_m_vals = [8]

    block_kv_vals = [int(x) for x in block_kv_order if int(x) in supported_block_kv and int(x) <= int(kv_ctx)]
    if allowed_block_kv:
        filtered = [int(x) for x in block_kv_vals if int(x) in allowed_block_kv]
        if filtered:
            block_kv_vals = filtered
        else:
            trace["substitutions"].append(
                {
                    "from": "org.dim.ATTN_FWD_BLOCK_KV",
                    "to": "backend.default",
                    "reason": "org allowed set has no supported/feasible values (filtered by KV_CTX)",
                    "detail": {"allowed": sorted({int(x) for x in allowed_block_kv}), "KV_CTX": int(kv_ctx)},
                }
            )
    if not block_kv_vals:
        block_kv_vals = [16]

    want_parallel = bool(tags & {"parallel_mapping", "warp_reduce", "block_reduce", "parallel_softmax"})
    kernel_kind = "attn_fwd_softmax_v7" if stack == "cpp_plugin" else "attn_fwd_tiled_v3"
    selected_variants = [str(kernel_kind)]
    template_id = f"template_{kernel_kind}"
    modules.append(
        BackendModule(
            id=str(template_id),
            kind="template",
            provides=[f"backend.kernel_kind.{kernel_kind}"],
            params=["ATTN_FWD_BLOCK_M", "ATTN_FWD_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
            attrs={"compiler_stack": str(stack)},
        )
    )
    module_edges.extend(
        [
            BackendModuleEdge(src=str(template_id), dst="mechanism_online_softmax", edge_type="uses"),
            BackendModuleEdge(src=str(template_id), dst="mechanism_qk_tiling", edge_type="uses"),
            BackendModuleEdge(src=str(template_id), dst="mechanism_kv_staging", edge_type="uses"),
        ]
    )

    param_space: dict[str, Any] = {
        "kernel_kind": list(selected_variants),
        "ATTN_FWD_BLOCK_M": list(block_m_vals),
        "ATTN_FWD_BLOCK_KV": list(block_kv_vals),
    }
    constraints.extend(
        [
            "ATTN_FWD_BLOCK_M in {4,8}",
            ("ATTN_FWD_BLOCK_KV in {16,32,64}" if stack == "cpp_plugin" else "ATTN_FWD_BLOCK_KV in {16,32}"),
        ]
    )

    ordered: list[BackendCandidate] = []
    for bm in block_m_vals:
        for bkv in block_kv_vals:
            ordered.append(
                BackendCandidate(
                    kernel_kind=str(kernel_kind),
                    bindings={"ATTN_FWD_BLOCK_M": int(bm), "ATTN_FWD_BLOCK_KV": int(bkv)},
                    note=("prefer_parallel" if want_parallel else "baseline"),
                )
            )

    # Deduplicate and clip.
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
        kernel="_attn_fwd",
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
        meta={"org_tags": sorted(tags), "compiler_stack": str(stack), "want_parallel": bool(want_parallel)},
    )


__all__ = ["plan_attn_fwd"]
