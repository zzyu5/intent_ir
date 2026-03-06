from __future__ import annotations

from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
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


def plan_masked_attention2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    target: str,
    budget: int = 32,
    compiler_stack: str = "python",
) -> BackendPlan:
    """
    Deterministic ORG -> CUDA backend plan for `masked_attention2d`.

    Notes:
      - python stack: prefers `attn2d_causal_softmax_v18` for the canonical tiny case (Q=16,KV=16,HD=16),
        with `attn2d_causal_softmax_v4` as a fallback.
      - cpp_plugin stack: uses `attn2d_causal_softmax_masked_hd16_keys_v1` for the canonical tiny case.
    """

    b = max(1, int(budget))
    tags = _collect_tags(org)
    stack = _normalize_stack(compiler_stack)

    q_ctx = _require_dim(shape_bindings, "Q_CTX")
    kv_ctx = _require_dim(shape_bindings, "KV_CTX")
    head_dim = _require_dim(shape_bindings, "HEAD_DIM")

    trace: dict[str, Any] = {"substitutions": []}
    passes = [
        "mechanism_to_variant_selection",
        "constraint_filtering",
        "org_priority_sort",
        "dedupe_clip",
    ]
    modules: list[BackendModule] = [
        BackendModule(
            id="mechanism_mask",
            kind="special_primitive",
            provides=["abstract.attention_mask"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="mechanism_online_softmax",
            kind="special_primitive",
            provides=["abstract.streaming_softmax_state"],
            params=[],
            constraints=[],
        ),
    ]
    module_edges: list[BackendModuleEdge] = []
    constraints: list[str] = []
    param_space: dict[str, Any] = {"kernel_kind": []}
    selected_variants: list[str] = []
    ordered: list[BackendCandidate] = []

    if stack == "cpp_plugin":
        selected_variants = ["attn2d_causal_softmax_masked_hd16_keys_v1"]
        param_space["kernel_kind"] = list(selected_variants)
        constraints = ["Q_CTX==16", "KV_CTX==16", "HEAD_DIM==16"]
        modules.append(
            BackendModule(
                id="template_masked_hd16_keys_v1",
                kind="template",
                provides=["backend.kernel_kind.attn2d_causal_softmax_masked_hd16_keys_v1"],
                params=[],
                constraints=list(constraints),
                attrs={"compiler_stack": "cpp_plugin"},
            )
        )
        module_edges.extend(
            [
                BackendModuleEdge(src="template_masked_hd16_keys_v1", dst="mechanism_mask", edge_type="uses"),
                BackendModuleEdge(src="template_masked_hd16_keys_v1", dst="mechanism_online_softmax", edge_type="uses"),
            ]
        )
        if q_ctx == 16 and kv_ctx == 16 and head_dim == 16:
            ordered.append(
                BackendCandidate(
                    kernel_kind="attn2d_causal_softmax_masked_hd16_keys_v1",
                    bindings={},
                    note="cpp_plugin_masked_hd16_keys",
                )
            )
        else:
            trace["substitutions"].append(
                {
                    "from": "org.masked_attention2d",
                    "to": "backend.skip",
                    "reason": f"unsupported dims for cpp_plugin: Q_CTX={q_ctx} KV_CTX={kv_ctx} HEAD_DIM={head_dim} (expects 16,16,16)",
                }
            )
    else:
        # python / other stacks: prefer v18 for canonical tiny case; otherwise fall back to v4 baseline.
        selected_variants = ["attn2d_causal_softmax_v18", "attn2d_causal_softmax_v4"]
        param_space["kernel_kind"] = list(selected_variants)
        constraints = [
            "attn2d_causal_softmax_v18 requires Q_CTX==16, KV_CTX==16, HEAD_DIM==16",
        ]
        modules.extend(
            [
                BackendModule(
                    id="template_v18",
                    kind="template",
                    provides=["backend.kernel_kind.attn2d_causal_softmax_v18"],
                    params=[],
                    constraints=["Q_CTX==16", "KV_CTX==16", "HEAD_DIM==16"],
                    attrs={"compiler_stack": str(stack)},
                ),
                BackendModule(
                    id="template_v4",
                    kind="template",
                    provides=["backend.kernel_kind.attn2d_causal_softmax_v4"],
                    params=[],
                    constraints=[],
                    attrs={"compiler_stack": str(stack)},
                ),
            ]
        )
        module_edges.extend(
            [
                BackendModuleEdge(src="template_v18", dst="mechanism_mask", edge_type="uses"),
                BackendModuleEdge(src="template_v18", dst="mechanism_online_softmax", edge_type="uses"),
                BackendModuleEdge(src="template_v4", dst="mechanism_mask", edge_type="uses"),
                BackendModuleEdge(src="template_v4", dst="mechanism_online_softmax", edge_type="uses"),
            ]
        )
        prefer_parallel = bool(tags & {"parallel_mapping", "parallel_softmax", "warp_reduce", "block_reduce"})
        if q_ctx == 16 and kv_ctx == 16 and head_dim == 16:
            if prefer_parallel:
                ordered.append(BackendCandidate(kernel_kind="attn2d_causal_softmax_v18", bindings={}, note="prefer_parallel"))
                ordered.append(BackendCandidate(kernel_kind="attn2d_causal_softmax_v4", bindings={}, note="fallback"))
            else:
                ordered.append(BackendCandidate(kernel_kind="attn2d_causal_softmax_v18", bindings={}, note="baseline"))
                ordered.append(BackendCandidate(kernel_kind="attn2d_causal_softmax_v4", bindings={}, note="fallback"))
        elif q_ctx > 0 and kv_ctx > 0 and head_dim > 0:
            trace["substitutions"].append(
                {
                    "from": "org.masked_attention2d",
                    "to": "attn2d_causal_softmax_v4",
                    "reason": "non-canonical dims; fall back to v4 baseline",
                    "detail": {"Q_CTX": int(q_ctx), "KV_CTX": int(kv_ctx), "HEAD_DIM": int(head_dim)},
                }
            )
            ordered.append(BackendCandidate(kernel_kind="attn2d_causal_softmax_v4", bindings={}, note="fallback"))
        else:
            trace["substitutions"].append(
                {
                    "from": "org.masked_attention2d",
                    "to": "backend.skip",
                    "reason": f"invalid dims: Q_CTX={q_ctx} KV_CTX={kv_ctx} HEAD_DIM={head_dim}",
                }
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
        kernel="masked_attention2d",
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
        meta={"org_tags": sorted(tags), "compiler_stack": str(stack)},
    )


__all__ = ["plan_masked_attention2d"]
