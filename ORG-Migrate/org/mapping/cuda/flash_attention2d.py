from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import flash_attention2d_catalog
from org.mapping.hardware_model import HardwareModel
from org.schema import OrgDoc, OrgTensorLifetime
from org.topology import find_lifetimes, find_tensor_ids, has_mechanism_relation, lifetime_mechanism_tags


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


def _merged_param_values(defaults: list[int], preferred: int | None, allowed: list[int]) -> list[int]:
    vals: list[int] = []
    if preferred is not None:
        vals.append(int(preferred))
    vals.extend(int(x) for x in defaults)
    vals.extend(int(x) for x in list(allowed or []))
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


def _candidate_key(candidate: BackendCandidate) -> tuple[str, tuple[tuple[str, int], ...]]:
    return str(candidate.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(candidate.bindings or {}).items()))


def _fact_present(facts: Mapping[str, Any] | None, key: str) -> bool:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    return bool(dict(mechanisms.get(str(key)) or {}).get("present"))


def _fact_attr(facts: Mapping[str, Any] | None, key: str, attr: str, default: Any = None) -> Any:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    attrs = dict(dict(mechanisms.get(str(key)) or {}).get("attrs") or {})
    return attrs.get(str(attr), default)


def _sm_number(sm: str | None) -> int:
    raw = str(sm or "").strip().lower()
    digits = "".join(ch for ch in raw if ch.isdigit())
    try:
        return int(digits)
    except Exception:
        return 0


def _norm_token(x: Any) -> str:
    return str(x or "").strip().lower().replace("-", "_").replace(" ", "_")


def _goal_ids_by_tag(org: OrgDoc) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for goal in list(getattr(org, "goals", []) or []):
        goal_id = str(getattr(goal, "id", "")).strip()
        tag = str(getattr(goal, "tag", "")).strip()
        if not goal_id or not tag:
            continue
        out.setdefault(tag, set()).add(goal_id)
    return out


def _reuse_window_scope(value: Any) -> str:
    token = _norm_token(value)
    if not token:
        return ""
    if token in {"cta_tile", "tile", "kv_tile", "tile_window", "stream_tile", "stream_chunk"}:
        return "tile"
    if token in {"kv_loop", "outer_loop", "loop_carried", "loop", "cta_loop"}:
        return "loop"
    return token


def _lifetime_ids(items: list[OrgTensorLifetime]) -> set[str]:
    return {str(item.id) for item in list(items or []) if str(getattr(item, "id", "")).strip()}


def _dataflow_connects(org: OrgDoc, *, src_ids: set[str], dst_ids: set[str], kinds: set[str]) -> bool:
    wanted_kinds = {_norm_token(kind) for kind in list(kinds or set()) if _norm_token(kind)}
    for edge in list(getattr(org, "dataflow_edges", []) or []):
        src = str(getattr(edge, "src", "")).strip()
        dst = str(getattr(edge, "dst", "")).strip()
        if src_ids and src not in src_ids:
            continue
        if dst_ids and dst not in dst_ids:
            continue
        if wanted_kinds and _norm_token(getattr(edge, "kind", "")) not in wanted_kinds:
            continue
        return True
    return False


def _max_lifetime_bytes(items: list[OrgTensorLifetime], default: int = 0) -> int:
    values = [int(item.bytes_hint) for item in list(items or []) if getattr(item, "bytes_hint", None) is not None]
    return int(max(values) if values else default)


def _sum_lifetime_bytes(items: list[OrgTensorLifetime], default: int = 0) -> int:
    values = [int(item.bytes_hint) for item in list(items or []) if getattr(item, "bytes_hint", None) is not None]
    return int(sum(values) if values else default)


def _first_dim_candidate(org: OrgDoc, *names: str) -> int | None:
    dim_candidates_norm = collect_dim_candidate_ints_normalized(org)
    values = union_dim_candidate_ints(dim_candidates_norm, *names)
    if values:
        return int(values[0])
    return None


@dataclass(frozen=True)
class FlashTopologySummary:
    graph_mode: bool = False
    q_lifetime_ids: tuple[str, ...] = ()
    kv_lifetime_ids: tuple[str, ...] = ()
    kv_shared_lifetime_ids: tuple[str, ...] = ()
    softmax_state_lifetime_ids: tuple[str, ...] = ()
    output_lifetime_ids: tuple[str, ...] = ()
    q_resident_path: bool = False
    kv_stream_path: bool = False
    online_softmax_path: bool = False
    output_path: bool = False
    kv_shared_stage_path: bool = False
    pipeline_path: bool = False
    q_resident_bytes_hint: int = 0
    kv_stage_bytes_hint: int = 0
    softmax_state_bytes_hint: int = 0
    output_bytes_hint: int = 0
    pipeline_depth_hint: int = 1
    base_block_kv: int = 0


@dataclass(frozen=True)
class FlashSharedStageFit:
    allowed: bool
    reason: str
    shared_bytes: int
    shared_ratio: float
    resident_ratio: float
    register_ratio: float


def _topology_summary(
    org: OrgDoc,
    *,
    kv_ctx: int,
    head_dim: int,
    source_oracle: Mapping[str, Any],
    ttgir_facts: Mapping[str, Any] | None,
) -> FlashTopologySummary:
    graph_mode = bool(
        list(getattr(org, "tensors", []) or [])
        or list(getattr(org, "tensor_lifetimes", []) or [])
        or list(getattr(org, "dataflow_edges", []) or [])
        or list(getattr(org, "mechanism_topology", []) or [])
    )
    if not graph_mode:
        return FlashTopologySummary(graph_mode=False)

    goal_ids = _goal_ids_by_tag(org)
    resident_goal_ids = goal_ids.get("resident_working_set", set())
    softmax_goal_ids = goal_ids.get("streaming_softmax_state", set())
    latency_goal_ids = goal_ids.get("latency_hiding", set())

    q_tensor_ids = find_tensor_ids(org, "Q", "query", "query_state", "query_tile")
    k_tensor_ids = find_tensor_ids(org, "K", "key", "key_tile", "k_tile")
    v_tensor_ids = find_tensor_ids(org, "V", "value", "value_tile", "v_tile")
    max_tensor_ids = find_tensor_ids(org, "m_i", "row_max", "max_state", "softmax_max")
    sum_tensor_ids = find_tensor_ids(org, "l_i", "row_sum", "sum_state", "softmax_sum")
    out_tensor_ids = find_tensor_ids(org, "Out", "output", "output_accumulator", "accumulator", "acc", "attn_out")

    q_lifetimes = [
        item
        for item in find_lifetimes(
            org,
            tensor_ids=q_tensor_ids,
            required_mechanism_tags={"q_resident_state"},
            required_goal_ids=(resident_goal_ids or None),
        )
        if _norm_token(getattr(item, "storage", "")) in {"register", "shared", "local"}
    ]
    k_lifetimes = [
        item
        for item in find_lifetimes(
            org,
            tensor_ids=k_tensor_ids,
            required_mechanism_tags={"kv_streamed_tiles"},
        )
        if _norm_token(getattr(item, "storage", "")) in {"global", "shared", "register", "local"}
    ]
    v_lifetimes = [
        item
        for item in find_lifetimes(
            org,
            tensor_ids=v_tensor_ids,
            required_mechanism_tags={"kv_streamed_tiles"},
        )
        if _norm_token(getattr(item, "storage", "")) in {"global", "shared", "register", "local"}
    ]
    softmax_max_lifetimes = find_lifetimes(
        org,
        tensor_ids=max_tensor_ids,
        required_mechanism_tags={"online_softmax_reduce"},
        required_goal_ids=(softmax_goal_ids or None),
    )
    softmax_sum_lifetimes = find_lifetimes(
        org,
        tensor_ids=sum_tensor_ids,
        required_mechanism_tags={"online_softmax_reduce"},
        required_goal_ids=(softmax_goal_ids or None),
    )
    output_lifetimes = [
        item
        for item in find_lifetimes(org, tensor_ids=out_tensor_ids)
        if (
            lifetime_mechanism_tags(org, item) & {"online_softmax_reduce", "output_layout_convert"}
            or _norm_token(getattr(item, "region", "")) in {"kv_loop", "epilogue", "store"}
        )
        and _norm_token(getattr(item, "storage", "")) in {"register", "shared", "local"}
    ]
    kv_lifetimes = list(k_lifetimes) + list(v_lifetimes)
    kv_shared_lifetimes = [item for item in kv_lifetimes if _norm_token(getattr(item, "storage", "")) == "shared"]
    softmax_state_lifetimes = list(softmax_max_lifetimes) + list(softmax_sum_lifetimes)

    q_ids = _lifetime_ids(q_lifetimes)
    kv_ids = _lifetime_ids(kv_lifetimes)
    kv_shared_ids = _lifetime_ids(kv_shared_lifetimes)
    softmax_ids = _lifetime_ids(softmax_state_lifetimes)
    out_ids = _lifetime_ids(output_lifetimes)

    q_resident_path = bool(
        q_ids
        and (
            _dataflow_connects(org, src_ids=q_ids, dst_ids=(softmax_ids | out_ids), kinds={"stage", "reduce", "score", "update"})
            or has_mechanism_relation(
                org,
                src_tags={"q_resident_state"},
                dst_tags={"online_softmax_reduce", "output_layout_convert"},
                relation="feeds",
                lifetime_ids=(q_ids | softmax_ids | out_ids),
            )
        )
    )
    kv_stream_path = bool(
        k_lifetimes
        and v_lifetimes
        and (
            _dataflow_connects(org, src_ids=kv_ids, dst_ids=(softmax_ids | out_ids), kinds={"stage", "stream", "reduce", "update"})
            or has_mechanism_relation(
                org,
                src_tags={"kv_streamed_tiles"},
                dst_tags={"online_softmax_reduce", "output_layout_convert"},
                relation="feeds",
                lifetime_ids=(kv_ids | softmax_ids | out_ids),
            )
        )
    )
    softmax_to_output = bool(
        _dataflow_connects(org, src_ids=softmax_ids, dst_ids=out_ids, kinds={"normalize", "update", "epilogue", "store"})
        or has_mechanism_relation(
            org,
            src_tags={"online_softmax_reduce"},
            dst_tags={"output_layout_convert"},
            relation="feeds",
            lifetime_ids=(softmax_ids | out_ids),
        )
        or any("online_softmax_reduce" in lifetime_mechanism_tags(org, item) for item in output_lifetimes)
    )
    online_softmax_path = bool(softmax_state_lifetimes and softmax_to_output)
    output_path = bool(out_ids and (softmax_to_output or _dataflow_connects(org, src_ids=kv_ids, dst_ids=out_ids, kinds={"update", "store"})))
    pipeline_path = bool(
        latency_goal_ids
        and (
            has_mechanism_relation(
                org,
                src_tags={"kv_streamed_tiles"},
                dst_tags={"prefetch_pipeline"},
                relation="gates",
                lifetime_ids=(kv_ids or None),
            )
            or has_mechanism_relation(
                org,
                src_tags={"prefetch_pipeline"},
                dst_tags={"kv_streamed_tiles"},
                relation="feeds",
                lifetime_ids=(kv_ids or None),
            )
            or any("pipeline_stages" in {_norm_token(dim) for dim in list(getattr(item, "dims", []) or [])} for item in kv_lifetimes)
        )
    )
    kv_shared_stage_path = bool(
        kv_stream_path
        and kv_shared_ids
        and (
            _dataflow_connects(org, src_ids=kv_shared_ids, dst_ids=(softmax_ids | out_ids), kinds={"stage", "update", "reduce"})
            or has_mechanism_relation(
                org,
                src_tags={"kv_streamed_tiles"},
                dst_tags={"online_softmax_reduce", "prefetch_pipeline"},
                relation="feeds",
                lifetime_ids=(kv_shared_ids | softmax_ids | out_ids),
            )
            or has_mechanism_relation(
                org,
                src_tags={"kv_streamed_tiles"},
                dst_tags={"prefetch_pipeline"},
                relation="gates",
                lifetime_ids=kv_shared_ids,
            )
        )
    )

    q_fact_bytes = int(_fact_attr(ttgir_facts, "staging.q_resident_state", "resident_bytes_hint", 0) or 0)
    kv_fact_bytes = int(_fact_attr(ttgir_facts, "staging.kv_streamed_tiles", "resident_bytes_hint", 0) or 0)
    pipeline_fact_depth = _coerce_int(_fact_attr(ttgir_facts, "pipeline.stage_hint", "pipeline_depth_hint", 1)) or 1

    q_resident_bytes_hint = max(_max_lifetime_bytes(q_lifetimes, default=0), q_fact_bytes, int(head_dim * 4))
    kv_stage_bytes_hint = max(
        _sum_lifetime_bytes(kv_shared_lifetimes, default=0),
        _sum_lifetime_bytes(kv_lifetimes, default=0),
        (int(kv_fact_bytes) * 2 if kv_fact_bytes > 0 else 0),
        int(min(kv_ctx, 32) * head_dim * 4 * 2),
    )
    softmax_state_bytes_hint = max(_sum_lifetime_bytes(softmax_state_lifetimes, default=0), 8)
    output_bytes_hint = max(_max_lifetime_bytes(output_lifetimes, default=0), int(head_dim * 4))

    pipeline_depth_hint = max(
        1,
        int(
            _first_dim_candidate(org, "pipeline_stages", "PIPELINE_STAGES")
            or _coerce_int(_fact_attr(ttgir_facts, "prefetch_pipeline", "pipeline_depth", 0))
            or pipeline_fact_depth
            or 1
        ),
    )
    base_block_kv = int(
        _coerce_int(dict(source_oracle or {}).get("bindings", {}).get("ATTN_BLOCK_KV"))
        or _first_dim_candidate(org, "tile_kv", "ATTN_BLOCK_KV", "BLOCK_KV")
        or kv_ctx
        or 1
    )

    return FlashTopologySummary(
        graph_mode=True,
        q_lifetime_ids=tuple(sorted(q_ids)),
        kv_lifetime_ids=tuple(sorted(kv_ids)),
        kv_shared_lifetime_ids=tuple(sorted(kv_shared_ids)),
        softmax_state_lifetime_ids=tuple(sorted(softmax_ids)),
        output_lifetime_ids=tuple(sorted(out_ids)),
        q_resident_path=q_resident_path,
        kv_stream_path=kv_stream_path,
        online_softmax_path=online_softmax_path,
        output_path=output_path,
        kv_shared_stage_path=kv_shared_stage_path,
        pipeline_path=pipeline_path,
        q_resident_bytes_hint=int(q_resident_bytes_hint),
        kv_stage_bytes_hint=int(kv_stage_bytes_hint),
        softmax_state_bytes_hint=int(softmax_state_bytes_hint),
        output_bytes_hint=int(output_bytes_hint),
        pipeline_depth_hint=int(pipeline_depth_hint),
        base_block_kv=int(max(1, base_block_kv)),
    )


def _scaled_kv_stage_bytes(topology: FlashTopologySummary, *, block_kv: int) -> int:
    if topology.kv_stage_bytes_hint <= 0:
        return 0
    base = max(1, int(topology.base_block_kv))
    scaled = int(round(float(topology.kv_stage_bytes_hint) * (float(block_kv) / float(base))))
    return max(0, scaled)


def _flash_resident_bytes_hint(
    *,
    block_kv: int,
    head_dim: int,
    topology: FlashTopologySummary,
    ttgir_facts: Mapping[str, Any] | None,
) -> int:
    q_bytes = int(
        topology.q_resident_bytes_hint
        or _fact_attr(ttgir_facts, "staging.q_resident_state", "resident_bytes_hint", 0)
        or (head_dim * 4)
    )
    kv_bytes = int(
        _scaled_kv_stage_bytes(topology, block_kv=block_kv)
        or ((_fact_attr(ttgir_facts, "staging.kv_streamed_tiles", "resident_bytes_hint", 0) or 0) * 2)
        or (block_kv * head_dim * 4 * 2)
    )
    softmax_bytes = int(topology.softmax_state_bytes_hint or 8)
    output_bytes = int(topology.output_bytes_hint or (head_dim * 4))
    return int(q_bytes + kv_bytes + softmax_bytes + output_bytes)


def _flash_thread_count_hint(*, kind: str, score_warps: int, hardware_model: HardwareModel) -> int:
    warp = max(1, int(hardware_model.warp_size))
    if kind == "attn2d_causal_softmax_v6":
        return int((2 + int(score_warps)) * warp)
    if kind == "attn2d_causal_softmax_v8":
        return int(4 * warp)
    if kind == "attn2d_causal_softmax_v9":
        return int(6 * warp)
    if kind == "attn2d_causal_softmax_v7":
        return int(4 * warp)
    return int(4 * warp)


def _flash_register_pressure_hint(*, kind: str, block_kv: int, score_warps: int) -> int:
    base = {
        "attn2d_causal_softmax_v6": 52,
        "attn2d_causal_softmax_v7": 58,
        "attn2d_causal_softmax_v8": 54,
        "attn2d_causal_softmax_v9": 62,
    }.get(str(kind), 56)
    base += {16: 0, 32: 4, 64: 10}.get(int(block_kv), 12)
    if kind == "attn2d_causal_softmax_v6":
        base += int(score_warps) * 2
    if kind == "attn2d_causal_softmax_v9":
        base += 4
    return int(base)


def _flash_resource_pressure(
    *,
    kind: str,
    block_kv: int,
    score_warps: int,
    resident_bytes: int,
    hardware_model: HardwareModel,
) -> tuple[int, float, float]:
    threads = _flash_thread_count_hint(kind=kind, score_warps=score_warps, hardware_model=hardware_model)
    shared_budget = max(1, int(hardware_model.shared_mem_kb) * 1024)
    resident_ratio = float(resident_bytes) / float(shared_budget)
    reg_hint = _flash_register_pressure_hint(kind=kind, block_kv=block_kv, score_warps=score_warps)
    register_ratio = float(int(threads) * int(reg_hint)) / float(max(1, int(hardware_model.register_budget)))
    return int(threads), float(resident_ratio), float(register_ratio)


def _complete_async_evidence(
    *,
    ttgir_facts: Mapping[str, Any] | None,
    ptx_facts: Mapping[str, Any] | None,
) -> bool:
    return bool(
        _fact_present(ttgir_facts, "pipeline.stage_hint")
        and _fact_present(ttgir_facts, "staging.kv_streamed_tiles")
        and _fact_present(ttgir_facts, "staging.q_resident_state")
        and bool(_fact_attr(ptx_facts, "pipeline.async_copy", "complete_async_pipeline", False))
    )


def _kv_shared_stage_fit(
    *,
    topology: FlashTopologySummary,
    ttgir_facts: Mapping[str, Any] | None,
    hardware_model: HardwareModel,
    toolchain_model: Mapping[str, Any] | None,
    block_kv: int,
    head_dim: int,
) -> FlashSharedStageFit:
    resident_bytes = _flash_resident_bytes_hint(
        block_kv=block_kv,
        head_dim=head_dim,
        topology=topology,
        ttgir_facts=ttgir_facts,
    )
    _, resident_ratio, register_ratio = _flash_resource_pressure(
        kind="attn2d_causal_softmax_v8",
        block_kv=block_kv,
        score_warps=0,
        resident_bytes=resident_bytes,
        hardware_model=hardware_model,
    )
    stage_multiplier = (2 if topology.pipeline_path and int(topology.pipeline_depth_hint) >= 2 else 1)
    shared_bytes = int(_scaled_kv_stage_bytes(topology, block_kv=block_kv) * stage_multiplier)
    shared_budget = max(1, int(hardware_model.shared_mem_kb) * 1024)
    shared_ratio = float(shared_bytes) / float(shared_budget)

    if not topology.graph_mode:
        return FlashSharedStageFit(False, "topology_missing", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if not topology.q_resident_path:
        return FlashSharedStageFit(False, "q_path_missing", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if not topology.kv_stream_path:
        return FlashSharedStageFit(False, "kv_stream_missing", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if not topology.online_softmax_path or not topology.output_path:
        return FlashSharedStageFit(False, "softmax_or_output_missing", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if not topology.kv_shared_stage_path:
        return FlashSharedStageFit(False, "shared_kv_lifetime_missing", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    effective_sm = _sm_number((toolchain_model or {}).get("effective_sm"))
    downleveled = bool((toolchain_model or {}).get("downleveled"))
    if effective_sm < 120 or downleveled:
        return FlashSharedStageFit(False, "toolchain_not_sm120", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if int(hardware_model.shared_mem_kb) < 96:
        return FlashSharedStageFit(False, "shared_budget_small", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if not hardware_model.supports_async_copy:
        return FlashSharedStageFit(False, "async_copy_unsupported", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if hardware_model.compute_cluster != "tensor_core":
        return FlashSharedStageFit(False, "compute_cluster_mismatch", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if shared_bytes <= 0:
        return FlashSharedStageFit(False, "shared_bytes_zero", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if shared_ratio > 0.30:
        return FlashSharedStageFit(False, "shared_bytes_over_budget", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if resident_ratio > 0.32:
        return FlashSharedStageFit(False, "resident_bytes_over_budget", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    if register_ratio > 0.18:
        return FlashSharedStageFit(False, "register_pressure_high", shared_bytes, shared_ratio, resident_ratio, register_ratio)
    return FlashSharedStageFit(True, "ok", shared_bytes, shared_ratio, resident_ratio, register_ratio)


def _selected_modules(
    *,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    topology: FlashTopologySummary,
    hardware_model: HardwareModel,
    any_shared_stage_fit: bool,
) -> tuple[list[BackendModule], list[BackendModuleEdge], list[dict[str, Any]]]:
    selected_ids = {"output_accumulator"}
    substitutions: list[dict[str, Any]] = []

    if topology.q_resident_path:
        selected_ids.add("q_resident_state")
    else:
        substitutions.append(
            {
                "from": "flash.q_resident_state",
                "to": "flash.direct_q_fetch",
                "reason": "topology missing resident Q lifetime path",
            }
        )
    if topology.kv_stream_path:
        selected_ids.add("kv_tile_stage")
    else:
        substitutions.append(
            {
                "from": "flash.kv_tile_stage",
                "to": "flash.direct_kv_fetch",
                "reason": "topology missing streamed K/V lifetime path",
            }
        )
    if topology.online_softmax_path:
        selected_ids.add("online_softmax_reduce")
    else:
        substitutions.append(
            {
                "from": "flash.online_softmax_reduce",
                "to": "flash.materialized_softmax",
                "reason": "topology missing online max/sum state path",
            }
        )
    if topology.pipeline_path and hardware_model.supports_async_copy:
        selected_ids.add("prefetch_pipeline")
    elif topology.pipeline_path:
        substitutions.append(
            {
                "from": "flash.prefetch_pipeline",
                "to": "flash.sync_prefetch",
                "reason": "hardware_model.supports_async_copy = false",
            }
        )
    if topology.kv_shared_stage_path and any_shared_stage_fit:
        selected_ids.add("kv_shared_stage")

    if topology.q_resident_path and topology.kv_stream_path and topology.online_softmax_path:
        selected_ids.add("backend_v6")
    if topology.q_resident_path and topology.kv_stream_path and topology.output_path:
        selected_ids.add("backend_v7")
        selected_ids.add("backend_v9")
    if topology.q_resident_path and topology.kv_stream_path and topology.output_path and topology.kv_shared_stage_path and any_shared_stage_fit:
        selected_ids.add("backend_v8")

    selected_modules = [m for m in modules if m.id in selected_ids]
    selected_edges = [e for e in module_edges if e.src in selected_ids and e.dst in selected_ids]
    return selected_modules, selected_edges, substitutions


def _score_flash_candidate(
    *,
    candidate: BackendCandidate,
    topology: FlashTopologySummary,
    cluster: str,
    source_oracle: Mapping[str, Any],
    kv_ctx: int,
    head_dim: int,
    ttgir_facts: Mapping[str, Any] | None,
    async_evidence_ok: bool,
    toolchain_model: Mapping[str, Any] | None,
    hardware_model: HardwareModel,
) -> tuple[float, str, str]:
    kind = str(candidate.kernel_kind)
    bindings = {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()}
    block_kv = int(bindings.get("ATTN_BLOCK_KV", 16))
    score_warps = int(bindings.get("ATTN_SCORE_WARPS", 0))
    is_async = bool(bindings.get("FLASH_ATTN_ASYNC_COPY", 0))
    wants_shared_stage = bool(bindings.get("FLASH_KV_SHARED_STAGE", 0))
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}

    resident_bytes = _flash_resident_bytes_hint(
        block_kv=block_kv,
        head_dim=head_dim,
        topology=topology,
        ttgir_facts=ttgir_facts,
    )
    threads_hint, resident_ratio, register_ratio = _flash_resource_pressure(
        kind=kind,
        block_kv=block_kv,
        score_warps=score_warps,
        resident_bytes=resident_bytes,
        hardware_model=hardware_model,
    )
    shared_fit = _kv_shared_stage_fit(
        topology=topology,
        ttgir_facts=ttgir_facts,
        hardware_model=hardware_model,
        toolchain_model=toolchain_model,
        block_kv=block_kv,
        head_dim=head_dim,
    )
    effective_sm = _sm_number((toolchain_model or {}).get("effective_sm"))
    downleveled = bool((toolchain_model or {}).get("downleveled"))
    topology_complete = bool(
        topology.q_resident_path
        and topology.kv_stream_path
        and topology.online_softmax_path
        and topology.output_path
    )
    preferred_score_warps = (
        4
        if resident_ratio >= 0.20 or shared_fit.shared_ratio >= 0.16 or int(block_kv) >= 64
        else 6
    )

    score = 0.0
    reasons: list[str] = [
        f"cluster={cluster}",
        "topology=graph" if topology.graph_mode else "topology=missing",
        f"q_path={int(topology.q_resident_path)}",
        f"kv_path={int(topology.kv_stream_path)}",
        f"softmax_path={int(topology.online_softmax_path)}",
        f"output_path={int(topology.output_path)}",
        f"shared_stage_path={int(topology.kv_shared_stage_path)}",
        f"resident_bytes={resident_bytes}",
        f"shared_bytes={shared_fit.shared_bytes}",
        f"threads_hint={threads_hint}",
        f"resident_ratio={resident_ratio:.3f}",
        f"shared_ratio={shared_fit.shared_ratio:.3f}",
        f"register_ratio={register_ratio:.3f}",
        f"pipeline_depth={int(topology.pipeline_depth_hint)}",
        f"effective_sm={effective_sm or 0}",
        f"shared_stage_fit={shared_fit.reason}",
    ]
    portability_note = "portable"

    if cluster == "cuda_tc_mid_smem":
        score += {
            "attn2d_causal_softmax_v8": 132.0,
            "attn2d_causal_softmax_v6": 124.0,
            "attn2d_causal_softmax_v9": 112.0,
            "attn2d_causal_softmax_v7": 92.0,
        }.get(kind, 40.0)
    elif cluster == "cuda_tc_large_smem":
        score += {
            "attn2d_causal_softmax_v7": 130.0 if topology.pipeline_path else 98.0,
            "attn2d_causal_softmax_v6": 118.0,
            "attn2d_causal_softmax_v9": 108.0,
            "attn2d_causal_softmax_v8": 104.0,
        }.get(kind, 50.0)
    else:
        score += {
            "attn2d_causal_softmax_v6": 100.0,
            "attn2d_causal_softmax_v7": 80.0,
            "attn2d_causal_softmax_v8": 72.0,
            "attn2d_causal_softmax_v9": 70.0,
        }.get(kind, 40.0)

    if not topology_complete:
        score -= 140.0
        portability_note = "topology_incomplete"
        reasons.append("topology_incomplete")

    if topology.q_resident_path:
        score += 8.0
    if topology.kv_stream_path:
        score += 8.0
    if topology.online_softmax_path:
        score += 10.0
    if topology.output_path:
        score += 6.0

    if kind == "attn2d_causal_softmax_v6":
        if topology.online_softmax_path:
            score += 18.0
            reasons.append("topology_online_softmax")
        else:
            score -= 80.0
            reasons.append("v6_requires_online_softmax")
        score += {6: 8.0, 4: 12.0, 2: 2.0}.get(int(score_warps), 0.0)
        reasons.append(f"score_warps={score_warps}")
        if int(score_warps) == int(preferred_score_warps):
            score += 14.0
            reasons.append("topology_balanced_softmax_warps")
        elif int(score_warps) == 6 and preferred_score_warps == 4:
            score -= 10.0
            reasons.append("topology_overparallel_softmax")
    elif kind == "attn2d_causal_softmax_v7":
        if topology.pipeline_path:
            score += 16.0
            reasons.append("topology_prefetch_pipeline")
        if is_async:
            if async_evidence_ok and topology.pipeline_path and hardware_model.supports_async_copy:
                score += (40.0 if cluster == "cuda_tc_large_smem" else 24.0)
                reasons.append("async_pipeline")
            else:
                score -= 48.0
                portability_note = "async_pipeline_incomplete"
                reasons.append("async_pipeline_incomplete")
        elif topology.pipeline_path:
            score -= 10.0
            reasons.append("pipeline_missing_async_binding")
    elif kind == "attn2d_causal_softmax_v8":
        if not wants_shared_stage:
            score -= 96.0
            portability_note = "missing_shared_stage_binding"
            reasons.append("missing_shared_stage_binding")
        elif shared_fit.allowed:
            score += 84.0
            reasons.append("topology_shared_stage_fit")
        else:
            score -= 88.0
            portability_note = shared_fit.reason
            reasons.append(f"shared_stage_rejected:{shared_fit.reason}")
    elif kind == "attn2d_causal_softmax_v9":
        if effective_sm >= 120 and not downleveled:
            score += 12.0
            reasons.append("sm120_frontier")
            if int(block_kv) == 32 and resident_ratio <= 0.18 and register_ratio <= 0.24:
                score += 28.0
                reasons.append("sm120_v9_tile32_fit")
            elif int(block_kv) == 64:
                score += 4.0
                reasons.append("sm120_v9_tile64")
        else:
            score -= 28.0
            portability_note = "toolchain_prefers_v6_v7"
            reasons.append("v9_requires_sm120")

    if topology.kv_shared_stage_path and kind in {"attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"}:
        if shared_fit.shared_ratio <= 0.18 and int(block_kv) == 32:
            score += 22.0
            reasons.append("topology_tile32_shared_fit")
        elif shared_fit.shared_ratio <= 0.28 and int(block_kv) == 64:
            score += (12.0 if cluster == "cuda_tc_large_smem" else -2.0)
            reasons.append("topology_tile64_shared_pressure")
        elif shared_fit.shared_ratio > 0.28 and int(block_kv) == 64:
            score -= 24.0
            reasons.append("tile64_shared_over_budget")
    elif kind == "attn2d_causal_softmax_v7" and is_async and cluster == "cuda_tc_large_smem":
        if int(block_kv) == 64:
            score += 28.0
            reasons.append("large_smem_async_full_tile")
        elif int(block_kv) == 32:
            score += 10.0
            reasons.append("large_smem_async_half_tile")
    else:
        score += {64: 18.0, 32: 12.0, 16: 4.0}.get(int(block_kv), 0.0)
        reasons.append(f"block_kv={block_kv}")

    if resident_ratio > 0.30:
        score -= 80.0
        portability_note = "resident_bytes_over_budget"
        reasons.append("resident_bytes_over_budget")
    elif register_ratio > 0.26:
        score -= 20.0
        portability_note = "register_pressure_high"
        reasons.append("register_pressure_high")

    if cluster == "cuda_tc_mid_smem" and kind == "attn2d_causal_softmax_v6" and resident_ratio >= 0.24 and threads_hint >= 192:
        score -= 20.0
        reasons.append("mid_smem_thread_pressure")

    if kv_ctx == block_kv:
        score += 3.0
        reasons.append("full_kv_tile")

    if source_kind == kind and {str(k): int(v) for k, v in source_bindings.items()} == bindings:
        source_bonus = 6.0
        if kind == "attn2d_causal_softmax_v6" and cluster == "cuda_tc_mid_smem":
            source_bonus = 12.0
        score += source_bonus
        reasons.append("source_exact")

    return score, ",".join(reasons), portability_note


def plan_flash_attention2d(
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
    q_ctx = _require_dim(shape_bindings, "Q_CTX")
    kv_ctx = _require_dim(shape_bindings, "KV_CTX")
    head_dim = _require_dim(shape_bindings, "HEAD_DIM")
    goal_tags = _goal_tags(org)

    topology = _topology_summary(
        org,
        kv_ctx=kv_ctx,
        head_dim=head_dim,
        source_oracle=source_oracle,
        ttgir_facts=ttgir_facts,
    )
    modules, module_edges, _passes = flash_attention2d_catalog(hardware_model)
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    exact_kind = str(source_oracle.get("kernel_kind") or "").strip()
    if exact_kind == "attn2d_causal_softmax_v6":
        source_bindings.setdefault("ATTN_SCORE_WARPS", 6)

    substitutions: list[dict[str, Any]] = []
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
            selected_modules=[],
            module_edges=[],
            param_space={"kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7", "attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"]},
            constraints=["HEAD_DIM == 64"],
            substitutions=substitutions,
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}", f"cluster={hardware_model.arch_cluster}", "topology_mode=graph" if topology.graph_mode else "topology_mode=missing"],
        )

    if not topology.graph_mode:
        substitutions.append(
            {
                "from": "flash_attention2d.org",
                "to": "backend.skip",
                "reason": "topology_graph_missing",
            }
        )
        return BackendPlan(
            kernel="flash_attention2d",
            source_oracle=dict(source_oracle or {}),
            hardware_model=hardware_model.to_json_dict(),
            selected_modules=[],
            module_edges=[],
            param_space={"kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7", "attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"]},
            constraints=["HEAD_DIM == 64", "topology graph required"],
            substitutions=substitutions,
            candidates=[],
            notes=[f"goals={sorted(goal_tags)}", f"cluster={hardware_model.arch_cluster}", "topology_mode=missing"],
        )

    dim_candidates_norm = collect_dim_candidate_ints_normalized(org)
    block_candidates = _ordered_param_values(
        defaults=_merged_param_values(
            defaults=[32, 64, 16],
            preferred=_coerce_int(source_bindings.get("ATTN_BLOCK_KV")),
            allowed=union_dim_candidate_ints(dim_candidates_norm, "tile_kv", "ATTN_BLOCK_KV", "BLOCK_KV"),
        ),
        preferred=_coerce_int(source_bindings.get("ATTN_BLOCK_KV")),
        allowed=[],
    )
    score_candidates = _ordered_param_values(
        defaults=_merged_param_values(
            defaults=[4, 6, 2],
            preferred=_coerce_int(source_bindings.get("ATTN_SCORE_WARPS")),
            allowed=union_dim_candidate_ints(dim_candidates_norm, "score_warps", "ATTN_SCORE_WARPS", "SCORE_WARPS"),
        ),
        preferred=_coerce_int(source_bindings.get("ATTN_SCORE_WARPS")),
        allowed=[],
    )
    block_candidates = [int(x) for x in block_candidates if int(x) <= int(kv_ctx)]
    if not block_candidates:
        block_candidates = [16]
        substitutions.append(
            {
                "from": "tile_kv",
                "to": "default_block_kv",
                "reason": "no topology/source candidate fits KV_CTX",
            }
        )

    cluster = str(hardware_model.arch_cluster)
    async_evidence_ok = bool(topology.pipeline_path and _complete_async_evidence(ttgir_facts=ttgir_facts, ptx_facts=ptx_facts))
    shared_fit_map = {
        int(bk): _kv_shared_stage_fit(
            topology=topology,
            ttgir_facts=ttgir_facts,
            hardware_model=hardware_model,
            toolchain_model=toolchain_model,
            block_kv=int(bk),
            head_dim=head_dim,
        )
        for bk in block_candidates
    }
    any_shared_stage_fit = any(item.allowed for item in shared_fit_map.values())
    selected_modules, selected_edges, selected_substitutions = _selected_modules(
        modules=modules,
        module_edges=module_edges,
        topology=topology,
        hardware_model=hardware_model,
        any_shared_stage_fit=any_shared_stage_fit,
    )
    substitutions.extend(selected_substitutions)

    if topology.pipeline_path and not async_evidence_ok:
        substitutions.append(
            {
                "from": "flash.prefetch_pipeline",
                "to": "flash.sync_prefetch",
                "reason": "incomplete async evidence",
            }
        )

    param_space = {
        "kernel_kind": ["attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7", "attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"],
        "ATTN_BLOCK_KV": list(block_candidates),
        "ATTN_SCORE_WARPS": list(score_candidates),
        "FLASH_ATTN_ASYNC_COPY": ([1] if topology.pipeline_path and hardware_model.supports_async_copy and async_evidence_ok else []),
        "FLASH_KV_SHARED_STAGE": ([1] if any_shared_stage_fit else []),
    }
    constraints = [
        "HEAD_DIM == 64",
        "ATTN_BLOCK_KV <= KV_CTX",
        "ATTN_SCORE_WARPS in {2,4,6}",
        "flash topology requires Q resident -> KV stream -> softmax state -> output path",
        "FLASH_ATTN_ASYNC_COPY requires topology pipeline path + complete async evidence",
        "FLASH_KV_SHARED_STAGE requires shared K/V lifetimes + shared_mem fit + sm120-ready toolchain",
    ]

    scored: list[BackendCandidate] = []
    if exact_kind in {"attn2d_causal_softmax_v6", "attn2d_causal_softmax_v7", "attn2d_causal_softmax_v8", "attn2d_causal_softmax_v9"}:
        exact_bindings_scored = dict(source_bindings)
        if exact_kind == "attn2d_causal_softmax_v8" and shared_fit_map.get(int(exact_bindings_scored.get("ATTN_BLOCK_KV", topology.base_block_kv)), FlashSharedStageFit(False, "", 0, 0.0, 0.0, 0.0)).allowed:
            exact_bindings_scored["FLASH_KV_SHARED_STAGE"] = 1
        score, score_reason, portability_note = _score_flash_candidate(
            candidate=BackendCandidate(kernel_kind=exact_kind, bindings=dict(exact_bindings_scored)),
            topology=topology,
            cluster=cluster,
            source_oracle=source_oracle,
            kv_ctx=kv_ctx,
            head_dim=head_dim,
            ttgir_facts=ttgir_facts,
            async_evidence_ok=async_evidence_ok,
            toolchain_model=toolchain_model,
            hardware_model=hardware_model,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=exact_kind,
                bindings=dict(exact_bindings_scored),
                note="source_exact",
                score=score,
                score_reason=score_reason,
                cluster=cluster,
                portability_note=portability_note,
            )
        )

    for bk in block_candidates:
        for sw in score_candidates:
            candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v6", bindings={"ATTN_BLOCK_KV": int(bk), "ATTN_SCORE_WARPS": int(sw)})
            score, score_reason, portability_note = _score_flash_candidate(
                candidate=candidate,
                topology=topology,
                cluster=cluster,
                source_oracle=source_oracle,
                kv_ctx=kv_ctx,
                head_dim=head_dim,
                ttgir_facts=ttgir_facts,
                async_evidence_ok=async_evidence_ok,
                toolchain_model=toolchain_model,
                hardware_model=hardware_model,
            )
            scored.append(
                BackendCandidate(
                    kernel_kind=candidate.kernel_kind,
                    bindings=dict(candidate.bindings),
                    note="topology_rank",
                    score=score,
                    score_reason=score_reason,
                    cluster=cluster,
                    portability_note=portability_note,
                )
            )

    for bk in block_candidates:
        candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v7", bindings={"ATTN_BLOCK_KV": int(bk)})
        score, score_reason, portability_note = _score_flash_candidate(
            candidate=candidate,
            topology=topology,
            cluster=cluster,
            source_oracle=source_oracle,
            kv_ctx=kv_ctx,
            head_dim=head_dim,
            ttgir_facts=ttgir_facts,
            async_evidence_ok=async_evidence_ok,
            toolchain_model=toolchain_model,
            hardware_model=hardware_model,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=candidate.kernel_kind,
                bindings=dict(candidate.bindings),
                note="topology_rank",
                score=score,
                score_reason=score_reason,
                cluster=cluster,
                portability_note=portability_note,
            )
        )

    for bk in block_candidates:
        v8_bindings = {"ATTN_BLOCK_KV": int(bk)}
        if shared_fit_map[int(bk)].allowed:
            v8_bindings["FLASH_KV_SHARED_STAGE"] = 1
        candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v8", bindings=v8_bindings)
        score, score_reason, portability_note = _score_flash_candidate(
            candidate=candidate,
            topology=topology,
            cluster=cluster,
            source_oracle=source_oracle,
            kv_ctx=kv_ctx,
            head_dim=head_dim,
            ttgir_facts=ttgir_facts,
            async_evidence_ok=async_evidence_ok,
            toolchain_model=toolchain_model,
            hardware_model=hardware_model,
        )
        scored.append(
            BackendCandidate(
                kernel_kind=candidate.kernel_kind,
                bindings=dict(candidate.bindings),
                note="topology_rank",
                score=score,
                score_reason=score_reason,
                cluster=cluster,
                portability_note=portability_note,
            )
        )

    effective_sm = _sm_number((toolchain_model or {}).get("effective_sm"))
    downleveled = bool((toolchain_model or {}).get("downleveled"))
    if effective_sm >= 120 and not downleveled:
        for bk in block_candidates:
            candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v9", bindings={"ATTN_BLOCK_KV": int(bk)})
            score, score_reason, portability_note = _score_flash_candidate(
                candidate=candidate,
                topology=topology,
                cluster=cluster,
                source_oracle=source_oracle,
                kv_ctx=kv_ctx,
                head_dim=head_dim,
                ttgir_facts=ttgir_facts,
                async_evidence_ok=async_evidence_ok,
                toolchain_model=toolchain_model,
                hardware_model=hardware_model,
            )
            scored.append(
                BackendCandidate(
                    kernel_kind=candidate.kernel_kind,
                    bindings=dict(candidate.bindings),
                    note="toolchain_frontier",
                    score=score,
                    score_reason=score_reason,
                    cluster=cluster,
                    portability_note=portability_note,
                )
            )

    if topology.pipeline_path and hardware_model.supports_async_copy and async_evidence_ok:
        for bk in block_candidates:
            async_score_warps: int | None = None
            rejected: list[dict[str, int | str]] = []
            for sw in score_candidates:
                ok, reason = _async_copy_guardrails(kv_ctx=kv_ctx, head_dim=head_dim, block_kv=int(bk), score_warps=int(sw))
                if ok:
                    async_score_warps = int(sw)
                    break
                rejected.append({"ATTN_BLOCK_KV": int(bk), "ATTN_SCORE_WARPS": int(sw), "reason": str(reason)})
            if async_score_warps is None:
                substitutions.append(
                    {
                        "from": "flash.prefetch_pipeline",
                        "to": "flash.sync_prefetch",
                        "reason": "no async-copy guardrail fit",
                        "detail": rejected,
                    }
                )
                continue
            candidate = BackendCandidate(kernel_kind="attn2d_causal_softmax_v7", bindings={"ATTN_BLOCK_KV": int(bk), "FLASH_ATTN_ASYNC_COPY": 1})
            score, score_reason, portability_note = _score_flash_candidate(
                candidate=candidate,
                topology=topology,
                cluster=cluster,
                source_oracle=source_oracle,
                kv_ctx=kv_ctx,
                head_dim=head_dim,
                ttgir_facts=ttgir_facts,
                async_evidence_ok=async_evidence_ok,
                toolchain_model=toolchain_model,
                hardware_model=hardware_model,
            )
            scored.append(
                BackendCandidate(
                    kernel_kind=candidate.kernel_kind,
                    bindings=dict(candidate.bindings),
                    note="latency_hiding_async",
                    score=score,
                    score_reason=score_reason,
                    cluster=cluster,
                    portability_note=portability_note,
                )
            )

    if (source_bindings.get("FLASH_ATTN_ASYNC_COPY") or 0) == 1 and not any(c.bindings.get("FLASH_ATTN_ASYNC_COPY") == 1 for c in scored):
        substitutions.append(
            {
                "from": "source.prefetch_pipeline",
                "to": "flash.sync_prefetch",
                "reason": "source async-copy candidate has no valid target realization",
            }
        )

    final: list[BackendCandidate] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    ordered = sorted(
        scored,
        key=lambda c: (
            -float(c.score if c.score is not None else 0.0),
            0 if c.kernel_kind == "attn2d_causal_softmax_v8" else (1 if c.kernel_kind == "attn2d_causal_softmax_v6" else 2),
            -int(c.bindings.get("FLASH_KV_SHARED_STAGE", 0)),
            -int(c.bindings.get("FLASH_ATTN_ASYNC_COPY", 0)),
            -int(c.bindings.get("ATTN_BLOCK_KV", 0)),
            -int(c.bindings.get("ATTN_SCORE_WARPS", 0)),
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

    if cluster == "cuda_tc_mid_smem" and any_shared_stage_fit and not any(c.kernel_kind == "attn2d_causal_softmax_v8" for c in final):
        best_v8 = next((c for c in ordered if c.kernel_kind == "attn2d_causal_softmax_v8"), None)
        if best_v8 is not None:
            if len(final) >= b and final:
                final[-1] = best_v8
            else:
                final.append(best_v8)

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
            f"source_kernel_kind={exact_kind or 'none'}",
            f"cluster={cluster}",
            f"toolchain_effective_sm={str((toolchain_model or {}).get('effective_sm') or '')}",
            f"toolchain_downleveled={bool((toolchain_model or {}).get('downleveled'))}",
            f"async_evidence={bool(async_evidence_ok)}",
            "topology_mode=graph",
            f"topology_q_resident_path={bool(topology.q_resident_path)}",
            f"topology_kv_stream_path={bool(topology.kv_stream_path)}",
            f"topology_softmax_state_path={bool(topology.online_softmax_path)}",
            f"topology_output_path={bool(topology.output_path)}",
            f"topology_shared_stage_path={bool(topology.kv_shared_stage_path)}",
            f"topology_pipeline_path={bool(topology.pipeline_path)}",
            f"topology_q_lifetimes={list(topology.q_lifetime_ids)}",
            f"topology_kv_lifetimes={list(topology.kv_lifetime_ids)}",
            f"topology_softmax_lifetimes={list(topology.softmax_state_lifetime_ids)}",
            f"topology_output_lifetimes={list(topology.output_lifetime_ids)}",
            f"topology_q_bytes={int(topology.q_resident_bytes_hint)}",
            f"topology_kv_stage_bytes={int(topology.kv_stage_bytes_hint)}",
            f"topology_softmax_bytes={int(topology.softmax_state_bytes_hint)}",
            f"topology_output_bytes={int(topology.output_bytes_hint)}",
            f"topology_pipeline_depth={int(topology.pipeline_depth_hint)}",
            f"topology_base_block_kv={int(topology.base_block_kv)}",
            f"topology_any_shared_stage_fit={bool(any_shared_stage_fit)}",
        ],
    )


__all__ = ["plan_flash_attention2d"]
