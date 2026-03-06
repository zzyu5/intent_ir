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


def plan_ai_bench_softmax(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    target: str,
    budget: int = 32,
) -> BackendPlan:
    """
    Deterministic ORG -> CUDA backend plan for `ai_bench_softmax`.

    Kernel kinds:
      - row_softmax_axis1_v1 (default real-MLIR softmax; optional SOFTMAX_BLOCK_THREADS, SOFTMAX_VEC4)
      - row_softmax_axis1_triton_v1 (Triton-like mapping; requires C<=1024)

    Bindings:
      - SOFTMAX_BLOCK_THREADS (multiple of 32, <=256)
      - SOFTMAX_VEC4 (1 enables vec4 path for ai_bench_softmax when feasible)
    """

    b = max(1, int(budget))
    tags = _collect_tags(org)
    dim_allowed_norm = collect_dim_allowed_ints_normalized(org)

    r_dim = _require_dim(shape_bindings, "R")
    c_dim = _require_dim(shape_bindings, "C")

    param_space: dict[str, Any] = {
        "kernel_kind": ["row_softmax_axis1_v1", "row_softmax_axis1_triton_v1"],
        "SOFTMAX_BLOCK_THREADS": [256, 128, 64],
        "SOFTMAX_VEC4": [1],
    }
    constraints: list[str] = [
        "SOFTMAX_BLOCK_THREADS is a positive multiple of 32 and <=256",
        "row_softmax_axis1_triton_v1 requires C<=1024",
        "SOFTMAX_VEC4=1 requires SOFTMAX_BLOCK_THREADS=256 and ceil(C/256)==4 and C<=2048",
    ]
    trace: dict[str, Any] = {"substitutions": []}
    selected_variants = ["row_softmax_axis1_v1", "row_softmax_axis1_triton_v1"]
    passes = [
        "mechanism_to_variant_selection",
        "param_space_enumeration",
        "constraint_filtering",
        "org_priority_sort",
        "dedupe_clip",
    ]
    modules: list[BackendModule] = [
        BackendModule(
            id="mechanism_vec4",
            kind="special_primitive",
            provides=["abstract.vec4"],
            params=["SOFTMAX_VEC4"],
            constraints=["ceil(C/256)==4", "C<=2048"],
        ),
        BackendModule(
            id="mechanism_warp_reduce",
            kind="communication",
            provides=["abstract.warp_reduce"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="template_row_softmax_v1",
            kind="template",
            provides=["backend.kernel_kind.row_softmax_axis1_v1"],
            params=["SOFTMAX_BLOCK_THREADS", "SOFTMAX_VEC4"],
            constraints=["SOFTMAX_BLOCK_THREADS%32==0", "SOFTMAX_BLOCK_THREADS<=256"],
        ),
        BackendModule(
            id="template_row_softmax_triton_v1",
            kind="template",
            provides=["backend.kernel_kind.row_softmax_axis1_triton_v1"],
            params=[],
            constraints=["C<=1024"],
        ),
    ]
    module_edges: list[BackendModuleEdge] = [
        BackendModuleEdge(src="template_row_softmax_v1", dst="mechanism_warp_reduce", edge_type="uses"),
        BackendModuleEdge(src="template_row_softmax_v1", dst="mechanism_vec4", edge_type="optional"),
        BackendModuleEdge(src="template_row_softmax_triton_v1", dst="mechanism_warp_reduce", edge_type="uses"),
    ]

    if r_dim <= 0 or c_dim <= 0:
        trace["substitutions"].append(
            {
                "from": "org.ai_bench_softmax",
                "to": "backend.skip",
                "reason": f"invalid dims: R={r_dim} C={c_dim}",
            }
        )
        return BackendPlan(
            kernel="ai_bench_softmax",
            target=str(target),
            hardware={"R": int(r_dim), "C": int(c_dim)},
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

    threads_order = [256, 128, 64]
    allowed_threads = union_dim_allowed(
        dim_allowed_norm,
        "SOFTMAX_BLOCK_THREADS",
        "BLOCK_THREADS",
        "threads",
        "num_threads",
        "BLOCK_SIZE",
    )
    if allowed_threads:
        filtered = [int(x) for x in threads_order if int(x) in allowed_threads]
        if filtered:
            threads_order = filtered
            param_space["SOFTMAX_BLOCK_THREADS"] = list(filtered)
        else:
            trace["substitutions"].append(
                {
                    "from": "org.dim.SOFTMAX_BLOCK_THREADS",
                    "to": "backend.default",
                    "reason": "org allowed set has no supported values",
                    "detail": {"allowed": sorted({int(x) for x in allowed_threads})},
                }
            )

    allowed_vec4 = union_dim_allowed(dim_allowed_norm, "SOFTMAX_VEC4", "VEC4", "vec4", "vectorize_vec4")
    want_vec4 = bool(allowed_vec4 and 1 in allowed_vec4) or bool(tags & {"vectorize", "vec4", "coalesced_vec4"})

    vec4_feasible = False
    if 256 in threads_order:
        ept = int((int(c_dim) + 256 - 1) // 256)
        vec4_feasible = bool(int(c_dim) <= 2048 and int(ept) == 4)

    if want_vec4 and not vec4_feasible:
        trace["substitutions"].append(
            {
                "from": "abstract.vec4",
                "to": "scalar_load",
                "reason": f"vec4 path not feasible for C={c_dim} (requires ceil(C/256)==4 and C<=2048)",
            }
        )

    ordered: list[BackendCandidate] = []
    if want_vec4 and vec4_feasible and (not allowed_vec4 or 1 in allowed_vec4):
        ordered.append(
            BackendCandidate(
                kernel_kind="row_softmax_axis1_v1",
                bindings={"SOFTMAX_BLOCK_THREADS": 256, "SOFTMAX_VEC4": 1},
                note="vec4",
            )
        )

    for t in threads_order:
        ordered.append(
            BackendCandidate(
                kernel_kind="row_softmax_axis1_v1",
                bindings={"SOFTMAX_BLOCK_THREADS": int(t)},
                note="thread_override",
            )
        )

    if int(c_dim) <= 1024:
        ordered.append(
            BackendCandidate(
                kernel_kind="row_softmax_axis1_triton_v1",
                bindings={},
                note="triton_like",
            )
        )
    else:
        trace["substitutions"].append(
            {
                "from": "row_softmax_axis1_triton_v1",
                "to": "row_softmax_axis1_v1",
                "reason": f"C={c_dim} exceeds 1024",
            }
        )

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
        kernel="ai_bench_softmax",
        target=str(target),
        hardware={"R": int(r_dim), "C": int(c_dim)},
        modules=list(modules),
        module_edges=list(module_edges),
        passes=list(passes),
        selected_variants=selected_variants,
        param_space=param_space,
        constraints=constraints,
        trace=trace,
        candidates=final,
        meta={"org_tags": sorted(tags), "want_vec4": bool(want_vec4), "vec4_feasible": bool(vec4_feasible)},
    )


__all__ = ["plan_ai_bench_softmax"]
