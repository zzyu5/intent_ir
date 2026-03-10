from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Mapping

from org.backend_plan import BackendCandidate, BackendModule, BackendModuleEdge, BackendPlan
from org.dim_utils import collect_dim_candidate_ints_normalized, union_dim_candidate_ints
from org.mapping.cuda.module_catalog import (
    ai_bench_matmul_catalog,
    ai_bench_softmax_catalog,
    attn_fwd_catalog,
    elementwise2d_catalog,
    flash_attention2d_catalog,
    group_norm_kernel_catalog,
    layer_norm_persistent_catalog,
    masked_attention2d_catalog,
    matmul_fused_epilogue2d_catalog,
    row_reduction_catalog,
    row_softmax_catalog,
)
from org.mapping.hardware_model import HardwareModel
from org.schema import OrgDoc, OrgTensorLifetime
from org.topology import lifetime_mechanism_tags, mechanism_tag_map, tensor_by_id


def _norm_token(x: Any) -> str:
    return str(x or "").strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _lookup_binding(bindings: Mapping[str, Any], key: str) -> Any:
    if str(key) in bindings:
        return bindings[str(key)]
    target = _norm_token(key)
    for binding_key, value in dict(bindings or {}).items():
        if _norm_token(binding_key) == target:
            return value
    return None


def _require_dim(bindings: Mapping[str, Any], key: str) -> int:
    value = _coerce_int(_lookup_binding(bindings, key))
    if value is None:
        raise ValueError(f"missing shape_bindings[{key!r}]")
    return int(value)


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


def _goal_tags(org: OrgDoc) -> set[str]:
    return {
        _norm_token(getattr(goal, "tag", ""))
        for goal in list(getattr(org, "goals", []) or [])
        if _norm_token(getattr(goal, "tag", ""))
    }


def _mechanism_tags(org: OrgDoc) -> set[str]:
    return {
        _norm_token(getattr(mechanism, "tag", ""))
        for mechanism in list(getattr(org, "mechanisms", []) or [])
        if _norm_token(getattr(mechanism, "tag", ""))
    }


def _tensor_role_tokens(org: OrgDoc) -> dict[str, set[str]]:
    tensors = tensor_by_id(org)
    out: dict[str, set[str]] = {}
    for tensor_id, tensor in tensors.items():
        tokens = {
            _norm_token(getattr(tensor, "name", "")),
            _norm_token(getattr(tensor, "role", "")),
            _norm_token(getattr(tensor, "layout", "")),
            _norm_token(getattr(tensor, "alias_group", "")),
        }
        if str(getattr(tensor, "view_of", "")).strip():
            tokens.add("view")
        for alias in list(getattr(tensor, "aliases", []) or []):
            tokens.add(_norm_token(alias))
        out[str(tensor_id)] = {x for x in tokens if x}
    return out


def _fact_present(facts: Mapping[str, Any] | None, key: str) -> bool:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    return bool(dict(mechanisms.get(str(key)) or {}).get("present"))


def _fact_attr(facts: Mapping[str, Any] | None, key: str, attr: str, default: Any = None) -> Any:
    mechanisms = dict((facts or {}).get("mechanisms") or {})
    attrs = dict(dict(mechanisms.get(str(key)) or {}).get("attrs") or {})
    return attrs.get(str(attr), default)


def _blocked_layout_hints(ttgir_facts: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    layouts = list(_fact_attr(ttgir_facts, "tiling.blocked_layout", "layouts", []) or [])
    if not layouts:
        return None, None
    first = dict(layouts[0] or {})
    size_per_thread = list(first.get("size_per_thread") or [])
    warps_per_cta = list(first.get("warps_per_cta") or [])
    threads_per_warp_layout = list(first.get("threads_per_warp_layout") or [])
    vector_width = int(size_per_thread[0]) if size_per_thread else None
    threads_hint: int | None = None
    if warps_per_cta and threads_per_warp_layout:
        threads_hint = int(warps_per_cta[0]) * int(threads_per_warp_layout[0])
    return threads_hint, vector_width


def _normalize_scope(value: Any) -> str:
    token = _norm_token(value)
    if not token:
        return ""
    if token in {"cta_tile", "tile", "row_tile", "tile_window", "stream_tile", "stream_chunk"}:
        return "tile"
    if token in {"row_reduce", "row", "row_scope"}:
        return "row"
    if token in {"row_epilogue", "full_row", "kv_loop", "loop", "loop_carried", "outer_loop"}:
        return "loop"
    if token in {"warp", "warp_reduce"}:
        return "warp"
    return token


@dataclass(frozen=True)
class ResourceGroup:
    name: str
    lifetime_ids: tuple[str, ...] = ()
    bytes_hint: int = 0
    reuse_scope: str = ""
    storage: str = ""
    tensor_count: int = 0


@dataclass(frozen=True)
class GraphProfile:
    goal_tags: frozenset[str]
    mechanism_tags: frozenset[str]
    signals: frozenset[str]
    dim_candidates: dict[str, list[int]]
    blocked_threads_hint: int | None = None
    blocked_vector_hint: int | None = None
    reduction_scope: str = ""
    pipeline_depth: int = 1
    shared_bytes: int = 0
    register_bytes: int = 0
    resource_groups: dict[str, ResourceGroup] = field(default_factory=dict)
    resident_window_scope: str = ""
    total_work: int = 0
    row_width: int = 0
    head_dim: int = 0
    schedule_relations: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()

    def has_signal(self, name: str) -> bool:
        return _norm_token(name) in set(self.signals or frozenset())


@dataclass(frozen=True)
class ParamSpec:
    name: str
    role: str
    dim_aliases: tuple[str, ...] = ()
    defaults: tuple[int, ...] = ()
    allowed_values: tuple[int, ...] = ()
    shape_cap_key: str = ""
    cap_mode: str = ""


@dataclass(frozen=True)
class OptionalModuleSpec:
    module_id: str
    signals: tuple[str, ...] = ()
    gate_param: str = ""


@dataclass(frozen=True)
class TemplateSpec:
    kernel_kind: str
    module_id: str
    param_names: tuple[str, ...] = ()
    enabled_flags: tuple[str, ...] = ()
    required_signals: tuple[str, ...] = ()
    signal_weights: dict[str, float] = field(default_factory=dict)
    cluster_bonus: dict[str, float] = field(default_factory=dict)
    portability_note: str = "portable"


@dataclass(frozen=True)
class FamilySpec:
    kernel: str
    catalog_builder: Callable[[HardwareModel], tuple[list[BackendModule], list[BackendModuleEdge], list[str]]]
    required_shape_keys: tuple[str, ...]
    base_modules: tuple[str, ...]
    optional_modules: tuple[OptionalModuleSpec, ...]
    params: tuple[ParamSpec, ...]
    templates: tuple[TemplateSpec, ...]


@dataclass(frozen=True)
class ResourceFit:
    allowed: bool
    reason: str
    shared_bytes: int
    shared_ratio: float
    register_ratio: float


@dataclass(frozen=True)
class CandidateEval:
    candidate: BackendCandidate
    template: TemplateSpec
    score: float
    score_reason: str
    portability_note: str
    modules: tuple[str, ...]
    resource_fit: ResourceFit
    enabled_params: tuple[str, ...]


def _active_flag_params(spec: FamilySpec, template: TemplateSpec, bindings: Mapping[str, int]) -> tuple[str, ...]:
    out: list[str] = []
    declared = {str(x) for x in list(getattr(template, "enabled_flags", ()) or ()) if str(x).strip()}
    for param in list(spec.params or ()):
        name = str(getattr(param, "name", "")).strip()
        if not name or name not in set(template.param_names or ()):
            continue
        role = _norm_token(getattr(param, "role", ""))
        if role not in {"shared_stage", "persistent_stage", "async_copy"} and name not in declared:
            continue
        if int(bindings.get(name, 0)) == 1:
            out.append(name)
    return tuple(sorted(set(out)))


def _graph_profile(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    ttgir_facts: Mapping[str, Any] | None,
    ptx_facts: Mapping[str, Any] | None,
) -> GraphProfile:
    goal_tags = _goal_tags(org)
    mechanism_tags = _mechanism_tags(org)
    tensor_roles = _tensor_role_tokens(org)
    mech_tags_by_id = mechanism_tag_map(org)
    dim_candidates = collect_dim_candidate_ints_normalized(org)
    blocked_threads_hint, blocked_vector_hint = _blocked_layout_hints(ttgir_facts)
    reduction_scope = str(_fact_attr(ttgir_facts, "communication.reduction", "reduction_scope", "") or "")
    schedule_relations = {
        _norm_token(getattr(edge, "relation", ""))
        for edge in list(getattr(org, "schedule_edges", []) or [])
        if _norm_token(getattr(edge, "relation", ""))
    }
    pipeline_depth = 1
    for mechanism in list(getattr(org, "mechanisms", []) or []):
        attrs = dict(getattr(mechanism, "attrs", {}) or {})
        value = _coerce_int(attrs.get("pipeline_depth"))
        if value is not None:
            pipeline_depth = max(pipeline_depth, int(value))
    for lifetime in list(getattr(org, "tensor_lifetimes", []) or []):
        value = _coerce_int(getattr(lifetime, "pipeline_stage", None))
        if value is not None:
            pipeline_depth = max(pipeline_depth, int(value) + 1)
    for key in ("pipeline_depth", "pipeline_stages", "stage_count"):
        values = union_dim_candidate_ints(dim_candidates, key)
        if values:
            pipeline_depth = max(pipeline_depth, int(values[0]))

    signal_names: set[str] = set()
    resource_groups: dict[str, ResourceGroup] = {}
    shared_bytes = 0
    register_bytes = 0
    resident_window_scope = ""
    stream_shared_ids: list[str] = []
    row_shared_ids: list[str] = []
    operand_ids: list[str] = []
    persistent_ids: list[str] = []
    output_ids: list[str] = []
    state_ids: list[str] = []

    if "resident_working_set" in goal_tags:
        signal_names.add("resident_state")
    if "streaming_softmax_state" in goal_tags:
        signal_names.add("online_state")
    if "mma_acceleration" in goal_tags:
        signal_names.add("mma_path")
    if "fused_epilogue_avoid_writeback" in goal_tags or "affine_epilogue_fusion" in goal_tags:
        signal_names.add("fused_epilogue")
    if "memory_coalescing" in goal_tags:
        signal_names.add("vector_path")
    if "persistent_row_state" in goal_tags:
        signal_names.add("persistent_path")

    if mechanism_tags & {"shared_staging", "block_synchronization"}:
        signal_names.add("sync_path")
    if mechanism_tags & {"vector_row_path", "vector_global_io", "tile_load_stage", "blocked_register_layout", "vector_group_io", "vector_dot_fragment"}:
        signal_names.add("vector_path")
    if mechanism_tags & {"row_reduction", "warp_reduction", "warp_reduction_tree", "warp_statistics", "online_normalization"}:
        signal_names.add("reduction_path")
    if mechanism_tags & {"mask_apply", "mask_causal_apply"}:
        signal_names.add("mask_path")
    if mechanism_tags & {"prefetch_pipeline", "async_prefetch"}:
        signal_names.add("async_pipeline")
    if mechanism_tags & {"output_layout_convert", "affine_epilogue", "affine_fused_epilogue", "bias_fused_epilogue", "epilogue_fused_writeback"}:
        signal_names.add("fused_epilogue")
    if mechanism_tags & {"persistent_row_cache"}:
        signal_names.add("persistent_path")
    if mechanism_tags & {"kv_streamed_tiles", "qkv_stage", "tiny_kv_stage"}:
        signal_names.add("stream_stage")
    if mechanism_tags & {"q_resident_state", "row_tile_resident", "group_tile_resident", "tile_resident"}:
        signal_names.add("resident_state")
    if mechanism_tags & {"mma_core", "dot_op"}:
        signal_names.add("mma_path")

    def _ensure_group(name: str, *, lifetimes: list[OrgTensorLifetime], storage: str = "", reuse_scope: str = "") -> None:
        nonlocal resource_groups
        bytes_hint = int(sum(max(0, int(getattr(item, "bytes_hint", 0) or 0)) for item in list(lifetimes or [])))
        group = ResourceGroup(
            name=str(name),
            lifetime_ids=tuple(str(item.id) for item in list(lifetimes or []) if str(getattr(item, "id", "")).strip()),
            bytes_hint=int(bytes_hint),
            reuse_scope=str(reuse_scope),
            storage=str(storage),
            tensor_count=int(len({str(getattr(item, "tensor", "")) for item in list(lifetimes or []) if str(getattr(item, "tensor", "")).strip()})),
        )
        resource_groups[str(name)] = group

    for lifetime in list(getattr(org, "tensor_lifetimes", []) or []):
        lifetime_id = str(getattr(lifetime, "id", "")).strip()
        tensor_id = str(getattr(lifetime, "tensor", "")).strip()
        role_tokens = set(tensor_roles.get(tensor_id) or set())
        mech_tokens = lifetime_mechanism_tags(org, lifetime)
        all_tokens = set(role_tokens) | set(mech_tokens)
        reuse_scope = _normalize_scope(getattr(lifetime, "reuse_window", "") or getattr(lifetime, "scope", "") or getattr(lifetime, "region", ""))
        storage = _norm_token(getattr(lifetime, "storage", ""))
        bytes_hint = max(0, int(getattr(lifetime, "bytes_hint", 0) or 0))
        if storage == "shared":
            shared_bytes += int(bytes_hint)
        if storage == "register":
            register_bytes += int(bytes_hint)
        if reuse_scope in {"tile", "loop"} and not resident_window_scope:
            resident_window_scope = reuse_scope
        if "resident_working_set" in goal_tags or any(tok in all_tokens for tok in {"row_tile_resident", "group_tile_resident", "tile_resident", "q_resident_state"}):
            signal_names.add("resident_state")
        if any(tok in all_tokens for tok in {"kv_streamed_tiles", "qkv_stage", "tiny_kv_stage", "operand_tile_stage", "ab_tile_stage"}):
            signal_names.add("stream_stage")
        if any(tok in all_tokens for tok in {"online_softmax_reduce", "parallel_softmax"}):
            signal_names.add("online_state")
        if any(tok in all_tokens for tok in {"row_reduction", "warp_reduction", "warp_reduction_tree", "warp_statistics", "online_normalization"}):
            signal_names.add("reduction_path")
        if any(tok in all_tokens for tok in {"vector_row_path", "vector_global_io", "tile_load_stage", "blocked_register_layout", "vector_group_io", "vector_dot_fragment"}):
            signal_names.add("vector_path")
        if any(tok in all_tokens for tok in {"mask_apply", "mask_causal_apply"}):
            signal_names.add("mask_path")
        if any(tok in all_tokens for tok in {"prefetch_pipeline", "async_prefetch"}):
            signal_names.add("async_pipeline")
        if any(tok in all_tokens for tok in {"mma_core", "dot_op"}):
            signal_names.add("mma_path")
        if any(tok in all_tokens for tok in {"output_layout_convert", "affine_epilogue", "affine_fused_epilogue", "bias_fused_epilogue", "epilogue_fused_writeback"}):
            signal_names.add("fused_epilogue")
        if any(tok in all_tokens for tok in {"persistent_row_cache", "row_stats"}) or "persistent_row_state" in goal_tags:
            signal_names.add("persistent_path")
        if any(tok in all_tokens for tok in {"blocked_register_layout", "output_layout_convert"}) or _norm_token(getattr(lifetime, "layout", "")):
            signal_names.add("layout_path")
        if any(tok in all_tokens for tok in {"block_synchronization", "shared_staging"}):
            signal_names.add("sync_path")
        if any(tok in role_tokens for tok in {"output_accumulator", "affine_out", "out", "output"}):
            signal_names.add("output_accumulator")
            output_ids.append(lifetime_id)
        if any(tok in role_tokens for tok in {"softmax_max", "softmax_sum", "row_stats", "mean", "rstd", "state", "max_state", "sum_state"}):
            signal_names.add("stateful_recurrence")
            state_ids.append(lifetime_id)
        if str(getattr(tensor_by_id(org).get(tensor_id, None), "view_of", "")).strip():
            signal_names.add("alias_view")
        if getattr(tensor_by_id(org).get(tensor_id, None), "alias_group", ""):
            signal_names.add("alias_view")
        if storage == "shared" and any(tok in all_tokens for tok in {"kv_streamed_tiles", "qkv_stage", "tiny_kv_stage"} | {"key_tile", "value_tile", "kv_tile"}):
            stream_shared_ids.append(lifetime_id)
        if storage == "shared" and any(tok in all_tokens for tok in {"row_tile_resident", "group_tile_resident", "tile_resident", "input_row"}):
            row_shared_ids.append(lifetime_id)
        if storage == "shared" and any(tok in all_tokens for tok in {"operand_tile_stage", "ab_tile_stage", "mma_core"}):
            operand_ids.append(lifetime_id)
        if any(tok in all_tokens for tok in {"persistent_row_cache", "row_stats"}) or reuse_scope in {"loop", "row"}:
            persistent_ids.append(lifetime_id)
        elif (
            "persistent_row_state" in goal_tags
            and storage == "shared"
            and any(tok in all_tokens for tok in {"input_row", "row_tile_resident", "group_tile_resident", "tile_resident"})
        ):
            persistent_ids.append(lifetime_id)
    if "latency_hiding" in goal_tags and ("async_pipeline" in signal_names or _fact_present(ptx_facts, "pipeline.async_copy")):
        signal_names.add("async_pipeline")
    if _fact_present(ptx_facts, "pipeline.async_copy") and bool(_fact_attr(ptx_facts, "pipeline.async_copy", "complete_async_pipeline", False)):
        signal_names.add("async_evidence")
        signal_names.add("async_pipeline")
    if bool(_fact_attr(ptx_facts, "primitive.mma", "complete_matrix_pipeline", False)):
        signal_names.add("mma_evidence")
    if schedule_relations & {"async_prefetch", "double_buffer", "pipeline_overlap"}:
        signal_names.add("async_pipeline")
    if schedule_relations & {"sync_before", "barrier", "wait_group", "commit_group"}:
        signal_names.add("sync_path")
    if schedule_relations & {"layout_convert", "swizzle", "transpose_view"}:
        signal_names.add("layout_path")
        signal_names.add("alias_view")

    lifetimes_by_id = {
        str(getattr(item, "id", "")).strip(): item
        for item in list(getattr(org, "tensor_lifetimes", []) or [])
        if str(getattr(item, "id", "")).strip()
    }
    _ensure_group("stream_shared", lifetimes=[lifetimes_by_id[x] for x in stream_shared_ids if x in lifetimes_by_id], storage="shared", reuse_scope="tile")
    _ensure_group("row_shared", lifetimes=[lifetimes_by_id[x] for x in row_shared_ids if x in lifetimes_by_id], storage="shared", reuse_scope=resident_window_scope or "tile")
    _ensure_group("operand_stage", lifetimes=[lifetimes_by_id[x] for x in operand_ids if x in lifetimes_by_id], storage="shared", reuse_scope="tile")
    _ensure_group("persistent", lifetimes=[lifetimes_by_id[x] for x in persistent_ids if x in lifetimes_by_id], storage="shared", reuse_scope=resident_window_scope or "loop")
    _ensure_group("output_accumulator", lifetimes=[lifetimes_by_id[x] for x in output_ids if x in lifetimes_by_id], storage="register", reuse_scope="loop")
    _ensure_group("state", lifetimes=[lifetimes_by_id[x] for x in state_ids if x in lifetimes_by_id], storage="register", reuse_scope="loop")
    if stream_shared_ids:
        signal_names.add("stream_shared")
        signal_names.add("shared_stage_path")
    if row_shared_ids:
        signal_names.add("row_shared")
        signal_names.add("shared_stage_path")

    total_work = 1
    row_width = 0
    for key in ("M", "N", "K", "Q_CTX", "KV_CTX", "HEAD_DIM"):
        iv = _coerce_int(shape_bindings.get(key))
        if iv is not None and iv > 0:
            total_work *= int(iv)
    for key in ("N", "KV_CTX", "Q_CTX"):
        iv = _coerce_int(shape_bindings.get(key))
        if iv is not None and iv > 0:
            row_width = int(iv)
            break
    head_dim = int(_coerce_int(shape_bindings.get("HEAD_DIM")) or 0)

    notes = [
        "topology_mode=graph",
        f"topology_pipeline_depth={int(pipeline_depth)}",
        f"topology_pipeline_path={bool('async_pipeline' in signal_names)}",
        f"topology_shared_stage_path={bool(('stream_shared' in resource_groups and resource_groups['stream_shared'].bytes_hint > 0) or ('row_shared' in resource_groups and resource_groups['row_shared'].bytes_hint > 0))}",
    ]
    if resident_window_scope:
        notes.append(f"topology_resident_window_scope={resident_window_scope}")

    return GraphProfile(
        goal_tags=frozenset(goal_tags),
        mechanism_tags=frozenset(mechanism_tags),
        signals=frozenset(signal_names),
        dim_candidates={str(k): [int(x) for x in list(v or [])] for k, v in dict(dim_candidates or {}).items()},
        blocked_threads_hint=blocked_threads_hint,
        blocked_vector_hint=blocked_vector_hint,
        reduction_scope=str(reduction_scope),
        pipeline_depth=max(1, int(pipeline_depth)),
        shared_bytes=int(shared_bytes),
        register_bytes=int(register_bytes),
        resource_groups=resource_groups,
        resident_window_scope=str(resident_window_scope),
        total_work=int(total_work),
        row_width=int(row_width),
        head_dim=int(head_dim),
        schedule_relations=frozenset(schedule_relations),
        notes=tuple(notes),
    )


def _next_power_of_two(x: int) -> int:
    value = max(1, int(x))
    out = 1
    while out < value:
        out <<= 1
    return out


def _resolve_param_values(
    *,
    param: ParamSpec,
    profile: GraphProfile,
    shape_bindings: Mapping[str, Any],
    source_bindings: Mapping[str, int],
    hardware_model: HardwareModel,
) -> list[int]:
    values: list[int] = []
    source_value = _coerce_int(source_bindings.get(param.name))
    if source_value is not None:
        values.append(int(source_value))
    raw_values = union_dim_candidate_ints(profile.dim_candidates, param.name, *param.dim_aliases)
    for raw_value in raw_values:
        iv = int(raw_value)
        if param.role == "threads" and iv in {1, 2, 4, 8, 16}:
            values.append(int(iv * int(hardware_model.warp_size)))
        values.append(iv)
    if param.role == "threads" and profile.blocked_threads_hint is not None:
        values.append(int(profile.blocked_threads_hint))
    if param.role == "vector_width" and profile.blocked_vector_hint is not None:
        values.append(int(profile.blocked_vector_hint))
    values.extend(int(x) for x in list(param.defaults or ()))
    values = _ordered_unique([int(x) for x in values if _coerce_int(x) is not None])
    if param.allowed_values:
        allowed = {int(x) for x in list(param.allowed_values or ())}
        values = [int(x) for x in values if int(x) in allowed]
    if param.shape_cap_key:
        cap = _coerce_int(_lookup_binding(shape_bindings, param.shape_cap_key))
        if cap is not None and int(cap) > 0:
            values = [int(x) for x in values if int(x) <= int(cap)]
    if param.cap_mode == "softmax_threads":
        row_width = int(profile.row_width or _coerce_int(_lookup_binding(shape_bindings, "N")) or 0)
        if row_width > 0:
            softmax_cap = min(128, _next_power_of_two(max(1, row_width)))
            values = [int(x) for x in values if int(x) <= int(softmax_cap)]
            if softmax_cap not in values and softmax_cap in set(int(x) for x in list(param.allowed_values or ()) or [softmax_cap]):
                values.insert(0, int(softmax_cap))
    return values


def _binding_for_role(spec: FamilySpec, bindings: Mapping[str, int], role: str) -> int | None:
    target = _norm_token(role)
    for param in list(spec.params or ()):
        if _norm_token(getattr(param, "role", "")) != target:
            continue
        value = _coerce_int(bindings.get(str(param.name)))
        if value is not None:
            return int(value)
    return None


def _effective_group_bytes(
    *,
    spec: FamilySpec,
    profile: GraphProfile,
    bindings: Mapping[str, int],
    group_name: str,
    hardware_model: HardwareModel,
) -> int:
    group = dict(profile.resource_groups or {}).get(str(group_name))
    if group is None:
        return 0
    bytes_hint = int(getattr(group, "bytes_hint", 0) or 0)
    shared_budget = int(hardware_model.shared_mem_kb) * 1024
    if bytes_hint > 0 and bytes_hint <= int(shared_budget * 2):
        return int(bytes_hint)
    threads = _binding_for_role(spec, bindings, "threads")
    vector_width = _binding_for_role(spec, bindings, "vector_width")
    tile_kv = _binding_for_role(spec, bindings, "tile_kv")
    tile_m = _binding_for_role(spec, bindings, "tile_m")
    tile_n = _binding_for_role(spec, bindings, "tile_n")
    tile_k = _binding_for_role(spec, bindings, "tile_k")
    head_dim = int(profile.head_dim or 0)
    estimate = 0
    if str(getattr(group, "reuse_scope", "")) == "tile":
        if tile_kv and head_dim:
            estimate = max(estimate, int(tile_kv) * int(head_dim) * 4 * max(1, int(group.tensor_count or 1)))
        if threads and vector_width:
            estimate = max(estimate, int(threads) * max(1, int(vector_width)) * 4 * max(1, int(group.tensor_count or 1)))
        if tile_m and tile_n and tile_k:
            estimate = max(estimate, (int(tile_m) * int(tile_k) + int(tile_k) * int(tile_n)) * 4)
    if estimate > 0:
        return int(estimate)
    return int(bytes_hint)


def _resource_fit(
    *,
    spec: FamilySpec,
    template: TemplateSpec,
    profile: GraphProfile,
    bindings: Mapping[str, int],
    hardware_model: HardwareModel,
) -> ResourceFit:
    shared_budget = int(hardware_model.shared_mem_kb) * 1024
    register_budget = int(hardware_model.register_budget or 65536)
    shared_bytes = 0
    enabled_flag_names = set(_active_flag_params(spec, template, bindings))
    if any(_norm_token(x).endswith("shared_stage") for x in enabled_flag_names):
        shared_bytes += _effective_group_bytes(spec=spec, profile=profile, bindings=bindings, group_name="stream_shared", hardware_model=hardware_model)
        shared_bytes *= max(1, int(profile.pipeline_depth))
    if any(_norm_token(x).endswith("persistent_row") for x in enabled_flag_names):
        shared_bytes += _effective_group_bytes(spec=spec, profile=profile, bindings=bindings, group_name="persistent", hardware_model=hardware_model)
    if any(_norm_token(x) == "mma_async_copy" for x in enabled_flag_names):
        shared_bytes += _effective_group_bytes(spec=spec, profile=profile, bindings=bindings, group_name="operand_stage", hardware_model=hardware_model)
        shared_bytes *= max(1, int(profile.pipeline_depth))
    if any(_norm_token(x).endswith("shared_stage") for x in enabled_flag_names) and not profile.has_signal("async_evidence") and profile.has_signal("async_pipeline"):
        shared_bytes = max(shared_bytes, _effective_group_bytes(spec=spec, profile=profile, bindings=bindings, group_name="stream_shared", hardware_model=hardware_model))
    register_bytes = int(profile.register_bytes or 0)
    shared_ratio = float(shared_bytes) / float(max(1, shared_budget))
    register_ratio = float(register_bytes) / float(max(1, register_budget))
    allowed = True
    reason = "ok"
    if shared_bytes > shared_budget:
        allowed = False
        reason = f"shared_budget_exceeded:{shared_bytes}>{shared_budget}"
    elif register_ratio > 1.0:
        allowed = False
        reason = f"register_budget_exceeded:{register_ratio:.3f}"
    return ResourceFit(
        allowed=bool(allowed),
        reason=str(reason),
        shared_bytes=int(shared_bytes),
        shared_ratio=float(shared_ratio),
        register_ratio=float(register_ratio),
    )


def _score_param(
    *,
    spec: FamilySpec,
    param: ParamSpec,
    value: int,
    profile: GraphProfile,
    source_bindings: Mapping[str, int],
    shape_bindings: Mapping[str, Any],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    iv = int(value)
    source_value = _coerce_int(source_bindings.get(param.name))
    if source_value is not None and int(source_value) == iv:
        score += 10.0
        reasons.append(f"source:{param.name}")
    if param.role == "threads":
        if profile.blocked_threads_hint is not None and int(profile.blocked_threads_hint) == iv:
            score += 8.0
            reasons.append("ttgir_threads")
        if profile.total_work >= 1_000_000 and iv in {256, 512}:
            score += 8.0
            reasons.append("large_work_threads")
        if profile.row_width > 0 and "streaming_softmax_state" in set(profile.goal_tags):
            ideal = min(128, _next_power_of_two(max(1, profile.row_width)))
            if iv == ideal:
                score += 20.0
                reasons.append("softmax_threads_ideal")
            elif iv > ideal:
                score -= 40.0
                reasons.append("softmax_threads_oversized")
        if profile.has_signal("reduction_path") and profile.reduction_scope == "warp" and iv in {32, 64, 128}:
            score += 6.0
            reasons.append("warp_reduction_threads")
    elif param.role == "vector_width":
        if profile.blocked_vector_hint is not None and int(profile.blocked_vector_hint) == iv:
            score += 10.0
            reasons.append("ttgir_vector")
        if profile.has_signal("vector_path") and iv > 1:
            score += 12.0
            reasons.append("vector_path")
        if profile.total_work >= 1_000_000:
            score += {4: 16.0, 2: 8.0}.get(iv, 0.0)
            if iv > 1:
                reasons.append("bandwidth_vector")
        group_size = _coerce_int(_lookup_binding(shape_bindings, "GROUP_SIZE"))
        if group_size is not None and profile.has_signal("vector_path"):
            if int(group_size) == 1 and iv > 1:
                score += {4: 18.0, 2: 10.0}.get(iv, 0.0)
                if iv > 1:
                    reasons.append("group_unit_vector")
            elif int(group_size) > 1 and iv > 1:
                score -= 36.0
                reasons.append("group_scalarize")
    elif param.role == "tile_kv":
        if profile.head_dim == 64 and iv == 32:
            score += 16.0
            reasons.append("tile_kv32")
        elif profile.head_dim == 64 and iv == 64:
            score += 6.0
            reasons.append("tile_kv64")
    elif param.role == "score_warps":
        if profile.reduction_scope == "warp" and iv in {4, 6}:
            score += 6.0
            reasons.append("warp_score")
    elif param.role == "shared_stage":
        if iv == 1 and profile.has_signal("stream_shared"):
            score += 36.0
            reasons.append("shared_stage_path")
    elif param.role == "persistent_stage":
        if iv == 1 and profile.has_signal("persistent_path"):
            score += 28.0
            reasons.append("persistent_path")
    elif param.role == "async_copy":
        if iv == 1 and profile.has_signal("async_pipeline"):
            score += 24.0
            reasons.append("async_pipeline")
    elif param.role in {"tile_m", "tile_n", "tile_k"}:
        if profile.has_signal("mma_path"):
            score += 6.0
            reasons.append("mma_tile")
    return score, reasons


def _evaluate_template_candidate(
    *,
    spec: FamilySpec,
    template: TemplateSpec,
    bindings: dict[str, int],
    profile: GraphProfile,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
) -> CandidateEval:
    score = 100.0
    reasons: list[str] = [f"cluster={hardware_model.arch_cluster}", f"kind={template.kernel_kind}"]
    source_kind = str(source_oracle.get("kernel_kind") or "").strip()
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}

    missing_required = [sig for sig in list(template.required_signals or ()) if not profile.has_signal(sig)]
    if missing_required:
        score -= 240.0
        reasons.append(f"missing_required={','.join(sorted(missing_required))}")
    for signal_name, weight in dict(template.signal_weights or {}).items():
        if profile.has_signal(signal_name):
            score += float(weight)
            reasons.append(f"signal:{signal_name}")
    score += float(template.cluster_bonus.get(str(hardware_model.arch_cluster), 0.0))
    for param in list(spec.params or ()):
        if str(param.name) not in set(template.param_names or ()):
            continue
        value = int(bindings.get(str(param.name), 0))
        delta, param_reasons = _score_param(
            spec=spec,
            param=param,
            value=value,
            profile=profile,
            source_bindings=source_bindings,
            shape_bindings=shape_bindings,
        )
        score += float(delta)
        reasons.extend(param_reasons)

    incomplete_async_portability = (
        profile.has_signal("mma_path")
        and profile.has_signal("fused_epilogue")
        and profile.has_signal("async_pipeline")
        and not profile.has_signal("async_evidence")
        and str(hardware_model.arch_cluster) == "cuda_tc_mid_smem"
    )
    template_kind_token = _norm_token(template.kernel_kind)
    portability = str(template.portability_note or "portable")
    if incomplete_async_portability:
        if "mma_path" in set(template.required_signals or ()):
            score -= 104.0
            reasons.append("mid_smem_incomplete_async_penalty")
            portability = "async_evidence_incomplete"
        elif "tile" in template_kind_token:
            score += 84.0
            reasons.append("mid_smem_portable_tile")
            if template_kind_token.endswith("tile_v2"):
                score += 8.0
                reasons.append("mid_smem_tile_v2")
                portability = "cluster_prefers_tile_v2"

    if source_kind == str(template.kernel_kind):
        score += 20.0
        reasons.append("source_kind")
        if bindings == source_bindings:
            score += 48.0
            reasons.append("source_exact")
    fit = _resource_fit(
        spec=spec,
        template=template,
        profile=profile,
        bindings=bindings,
        hardware_model=hardware_model,
    )
    enabled_params = _active_flag_params(spec, template, bindings)
    if enabled_params and not fit.allowed:
        score -= 240.0
        portability = "resource_blocked"
        reasons.append(f"resource_blocked={fit.reason}")
    if any(_norm_token(name).endswith("shared_stage") for name in enabled_params):
        reasons.append(f"topology_shared_stage_fit={fit.allowed}")
        reasons.append(f"shared_bytes={fit.shared_bytes}")
    if any(_norm_token(name).endswith("persistent_row") for name in enabled_params):
        reasons.append(f"topology_persistent_fit={fit.allowed}")
    if any(_norm_token(name) == "mma_async_copy" for name in enabled_params):
        reasons.append(f"topology_async_fit={fit.allowed}")
    candidate = BackendCandidate(
        kernel_kind=str(template.kernel_kind),
        bindings={str(k): int(v) for k, v in dict(bindings or {}).items() if str(k).strip()},
        note=str(spec.kernel),
        score=float(score),
        score_reason=",".join(reasons),
        cluster=str(hardware_model.arch_cluster),
        portability_note=str(portability),
    )
    modules = tuple(sorted(set(spec.base_modules) | {str(template.module_id)}))
    return CandidateEval(
        candidate=candidate,
        template=template,
        score=float(score),
        score_reason=str(candidate.score_reason),
        portability_note=str(portability),
        modules=modules,
        resource_fit=fit,
        enabled_params=enabled_params,
    )


def _candidate_key(candidate: BackendCandidate) -> tuple[str, tuple[tuple[str, int], ...]]:
    return str(candidate.kernel_kind), tuple(sorted((str(k), int(v)) for k, v in dict(candidate.bindings or {}).items()))


def _generate_candidate_evals(
    *,
    spec: FamilySpec,
    profile: GraphProfile,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
) -> list[CandidateEval]:
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    param_map = {str(param.name): param for param in list(spec.params or ())}
    param_values: dict[str, list[int]] = {}
    for param in list(spec.params or ()):
        values = _resolve_param_values(
            param=param,
            profile=profile,
            shape_bindings=shape_bindings,
            source_bindings=source_bindings,
            hardware_model=hardware_model,
        )
        if not values:
            values = [0] if _norm_token(param.role) in {"shared_stage", "persistent_stage", "async_copy"} else [1]
        param_values[str(param.name)] = list(values)

    out: list[CandidateEval] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    for template in list(spec.templates or ()):
        template_param_names = [str(x) for x in list(template.param_names or ()) if str(x).strip()]
        lists: list[list[int]] = []
        for name in template_param_names:
            values = list(param_values.get(str(name)) or [0])
            if str(name) in set(template.enabled_flags or ()):
                values = [int(x) for x in values if int(x) == 1] or [1]
            lists.append(values)
        if not lists:
            combos = [()]
        else:
            combos = product(*lists)
        for combo in combos:
            bindings = {template_param_names[i]: int(combo[i]) for i in range(len(template_param_names))}
            key = _candidate_key(BackendCandidate(kernel_kind=str(template.kernel_kind), bindings=bindings))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                _evaluate_template_candidate(
                    spec=spec,
                    template=template,
                    bindings=bindings,
                    profile=profile,
                    shape_bindings=shape_bindings,
                    source_oracle=source_oracle,
                    hardware_model=hardware_model,
                )
            )
    out.sort(key=lambda item: (-float(item.score), str(item.candidate.kernel_kind), sorted(dict(item.candidate.bindings or {}).items())))
    return out


def _selected_modules(
    *,
    spec: FamilySpec,
    modules: list[BackendModule],
    module_edges: list[BackendModuleEdge],
    profile: GraphProfile,
    ranked: list[CandidateEval],
) -> tuple[list[BackendModule], list[BackendModuleEdge]]:
    selected_ids = set(str(x) for x in list(spec.base_modules or ()))
    enabled_params = {
        str(param)
        for item in list(ranked or [])
        for param in list(getattr(item, "enabled_params", ()) or ())
        if str(param).strip() and bool(item.resource_fit.allowed)
    }
    for optional in list(spec.optional_modules or ()):
        if optional.gate_param:
            if str(optional.gate_param) not in enabled_params:
                continue
        if optional.signals and not all(profile.has_signal(sig) for sig in list(optional.signals or ())):
            continue
        selected_ids.add(str(optional.module_id))
    for item in list(ranked or []):
        selected_ids.update(set(item.modules))
    selected_modules = [module for module in list(modules or []) if str(module.id) in selected_ids]
    selected_edges = [
        edge
        for edge in list(module_edges or [])
        if str(getattr(edge, "src", "")) in selected_ids and str(getattr(edge, "dst", "")) in selected_ids
    ]
    return selected_modules, selected_edges


def _family_specs() -> dict[str, FamilySpec]:
    return {
        "flash_attention2d": FamilySpec(
            kernel="flash_attention2d",
            catalog_builder=flash_attention2d_catalog,
            required_shape_keys=("Q_CTX", "KV_CTX", "HEAD_DIM"),
            base_modules=("q_resident_state", "kv_tile_stage", "online_softmax_reduce", "output_accumulator"),
            optional_modules=(
                OptionalModuleSpec(module_id="prefetch_pipeline", signals=("async_pipeline",)),
                OptionalModuleSpec(module_id="kv_shared_stage", signals=("stream_stage",), gate_param="FLASH_KV_SHARED_STAGE"),
            ),
            params=(
                ParamSpec(name="ATTN_BLOCK_KV", role="tile_kv", dim_aliases=("tile_kv", "BLOCK_KV"), defaults=(32, 64, 16), allowed_values=(16, 32, 64), shape_cap_key="KV_CTX"),
                ParamSpec(name="ATTN_SCORE_WARPS", role="score_warps", dim_aliases=("score_warps", "SCORE_WARPS"), defaults=(6, 4, 2), allowed_values=(2, 4, 6)),
                ParamSpec(name="FLASH_KV_SHARED_STAGE", role="shared_stage", defaults=(1, 0), allowed_values=(0, 1)),
                ParamSpec(name="FLASH_ATTN_ASYNC_COPY", role="async_copy", defaults=(1, 0), allowed_values=(0, 1)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="attn2d_causal_softmax_v8",
                    module_id="backend_v8",
                    param_names=("ATTN_BLOCK_KV", "FLASH_KV_SHARED_STAGE"),
                    enabled_flags=("FLASH_KV_SHARED_STAGE",),
                    required_signals=("resident_state", "stream_stage", "output_accumulator"),
                    signal_weights={"stream_stage": 18.0, "shared_stage_path": 18.0, "online_state": 6.0},
                    cluster_bonus={"cuda_tc_mid_smem": 54.0, "cuda_tc_large_smem": 24.0},
                ),
                TemplateSpec(
                    kernel_kind="attn2d_causal_softmax_v7",
                    module_id="backend_v7",
                    param_names=("ATTN_BLOCK_KV", "FLASH_ATTN_ASYNC_COPY"),
                    enabled_flags=("FLASH_ATTN_ASYNC_COPY",),
                    required_signals=("resident_state", "stream_stage", "async_pipeline"),
                    signal_weights={"async_evidence": 22.0, "online_state": 2.0},
                    cluster_bonus={"cuda_tc_large_smem": 48.0, "cuda_tc_mid_smem": 8.0},
                ),
                TemplateSpec(
                    kernel_kind="attn2d_causal_softmax_v6",
                    module_id="backend_v6",
                    param_names=("ATTN_BLOCK_KV", "ATTN_SCORE_WARPS"),
                    required_signals=("resident_state", "stream_stage", "online_state"),
                    signal_weights={"online_state": 16.0, "async_pipeline": 4.0},
                    cluster_bonus={"cuda_tc_mid_smem": 12.0, "cuda_tc_large_smem": 10.0},
                ),
                TemplateSpec(
                    kernel_kind="attn2d_causal_softmax_v9",
                    module_id="backend_v9",
                    param_names=("ATTN_BLOCK_KV",),
                    required_signals=("resident_state", "stream_stage"),
                    signal_weights={"online_state": 4.0},
                    cluster_bonus={"cuda_tc_mid_smem": 6.0, "cuda_tc_large_smem": 4.0},
                ),
            ),
        ),
        "_attn_fwd": FamilySpec(
            kernel="_attn_fwd",
            catalog_builder=attn_fwd_catalog,
            required_shape_keys=("Q_CTX", "KV_CTX", "HEAD_DIM"),
            base_modules=("qkv_stage", "online_softmax_reduce", "mask_causal_apply", "output_accumulator"),
            optional_modules=(OptionalModuleSpec(module_id="prefetch_pipeline", signals=("async_pipeline",)),),
            params=(
                ParamSpec(name="ATTN_FWD_BLOCK_M", role="tile_m", dim_aliases=("block_m",), defaults=(8, 4), allowed_values=(4, 8)),
                ParamSpec(name="ATTN_FWD_BLOCK_KV", role="tile_kv", dim_aliases=("block_kv",), defaults=(32, 16), allowed_values=(16, 32), shape_cap_key="KV_CTX"),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="attn_fwd_tiled_v3",
                    module_id="backend_attn_fwd_tiled_v3",
                    param_names=("ATTN_FWD_BLOCK_M", "ATTN_FWD_BLOCK_KV"),
                    required_signals=("stream_stage", "online_state", "mask_path"),
                    signal_weights={"async_pipeline": 10.0},
                    cluster_bonus={"cuda_tc_mid_smem": 18.0, "cuda_tc_large_smem": 22.0},
                ),
                TemplateSpec(
                    kernel_kind="attn_fwd_softmax_v2",
                    module_id="backend_attn_fwd_softmax_v2",
                    param_names=(),
                    required_signals=("online_state", "mask_path"),
                    signal_weights={},
                    cluster_bonus={"cuda_tc_mid_smem": 4.0},
                ),
                TemplateSpec(
                    kernel_kind="attn_fwd_softmax_v1",
                    module_id="backend_attn_fwd_softmax_v1",
                    param_names=(),
                    required_signals=("mask_path",),
                    signal_weights={},
                ),
            ),
        ),
        "softmax_inner": FamilySpec(
            kernel="softmax_inner",
            catalog_builder=lambda hw: row_softmax_catalog(hw, masked=False),
            required_shape_keys=("M", "N"),
            base_modules=("softmax_inner_row_tile_resident", "softmax_inner_row_reduction"),
            optional_modules=(OptionalModuleSpec(module_id="softmax_inner_vector_row_path", signals=("vector_path",)),),
            params=(ParamSpec(name="SOFTMAX_BLOCK_THREADS", role="threads", dim_aliases=("block_threads",), defaults=(64, 128), allowed_values=(32, 64, 128), cap_mode="softmax_threads"),),
            templates=(
                TemplateSpec(
                    kernel_kind="row_softmax_axis1_triton_v1",
                    module_id="softmax_inner_backend_triton_v1",
                    param_names=("SOFTMAX_BLOCK_THREADS",),
                    required_signals=("resident_state", "reduction_path"),
                    signal_weights={"vector_path": 10.0, "online_state": 8.0},
                ),
                TemplateSpec(
                    kernel_kind="row_softmax_axis1_v1",
                    module_id="softmax_inner_backend_v1",
                    param_names=(),
                    required_signals=("reduction_path",),
                    signal_weights={"online_state": 6.0},
                ),
            ),
        ),
        "masked_softmax2d": FamilySpec(
            kernel="masked_softmax2d",
            catalog_builder=lambda hw: row_softmax_catalog(hw, masked=True),
            required_shape_keys=("M", "N"),
            base_modules=("masked_softmax_row_tile_resident", "masked_softmax_row_reduction", "masked_softmax_mask_apply"),
            optional_modules=(OptionalModuleSpec(module_id="masked_softmax_vector_row_path", signals=("vector_path",)),),
            params=(ParamSpec(name="SOFTMAX_BLOCK_THREADS", role="threads", dim_aliases=("block_threads",), defaults=(64, 128), allowed_values=(32, 64, 128), cap_mode="softmax_threads"),),
            templates=(
                TemplateSpec(
                    kernel_kind="row_softmax_axis1_triton_v1",
                    module_id="masked_softmax_backend_triton_v1",
                    param_names=("SOFTMAX_BLOCK_THREADS",),
                    required_signals=("resident_state", "reduction_path", "mask_path"),
                    signal_weights={"vector_path": 10.0, "online_state": 8.0},
                ),
                TemplateSpec(
                    kernel_kind="row_masked_softmax_axis1_v1",
                    module_id="masked_softmax_backend_v1",
                    param_names=(),
                    required_signals=("reduction_path", "mask_path"),
                    signal_weights={"online_state": 6.0},
                ),
            ),
        ),
        "ai_bench_softmax": FamilySpec(
            kernel="ai_bench_softmax",
            catalog_builder=ai_bench_softmax_catalog,
            required_shape_keys=("M", "N"),
            base_modules=("ai_softmax_row_tile_resident", "ai_softmax_row_reduction", "ai_softmax_power2_padding"),
            optional_modules=(OptionalModuleSpec(module_id="ai_softmax_vector_row_path", signals=("vector_path",)),),
            params=(
                ParamSpec(name="SOFTMAX_BLOCK_THREADS", role="threads", dim_aliases=("block_threads",), defaults=(256,), allowed_values=(64, 128, 256), cap_mode="softmax_threads"),
                ParamSpec(name="SOFTMAX_VEC4", role="vector_width", dim_aliases=("vector_width", "vec4"), defaults=(1, 0), allowed_values=(0, 1)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="row_softmax_axis1_vec4_v2",
                    module_id="ai_softmax_backend_vec4_v2",
                    param_names=("SOFTMAX_BLOCK_THREADS", "SOFTMAX_VEC4"),
                    enabled_flags=("SOFTMAX_VEC4",),
                    required_signals=("resident_state", "reduction_path", "vector_path"),
                    signal_weights={"vector_path": 18.0, "online_state": 10.0},
                ),
                TemplateSpec(
                    kernel_kind="row_softmax_axis1_v1",
                    module_id="ai_softmax_backend_v1",
                    param_names=(),
                    required_signals=("reduction_path",),
                    signal_weights={"online_state": 8.0},
                ),
            ),
        ),
        "row_sum": FamilySpec(
            kernel="row_sum",
            catalog_builder=lambda hw: row_reduction_catalog(hw, reduction_kind="sum"),
            required_shape_keys=("M", "N"),
            base_modules=("row_sum_row_tile_resident", "row_sum_warp_reduction_tree", "row_sum_writeback"),
            optional_modules=(
                OptionalModuleSpec(module_id="row_sum_vector_row_load", signals=("vector_path",)),
                OptionalModuleSpec(module_id="row_sum_shared_warp_exchange", signals=("sync_path",), gate_param="ROW_REDUCE_SHARED_STAGE"),
            ),
            params=(
                ParamSpec(name="ROW_REDUCE_BLOCK_THREADS", role="threads", dim_aliases=("block_threads", "threads_per_block", "num_warps"), defaults=(64, 128, 256), allowed_values=(32, 64, 128, 256)),
                ParamSpec(name="ROW_REDUCE_VECTOR_WIDTH", role="vector_width", dim_aliases=("vector_width", "size_per_thread"), defaults=(1, 2, 4), allowed_values=(1, 2, 4)),
                ParamSpec(name="ROW_REDUCE_SHARED_STAGE", role="shared_stage", defaults=(1, 0), allowed_values=(0, 1)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="row_sum_axis1_v2",
                    module_id="row_sum_backend_v2",
                    param_names=("ROW_REDUCE_BLOCK_THREADS", "ROW_REDUCE_VECTOR_WIDTH", "ROW_REDUCE_SHARED_STAGE"),
                    required_signals=("resident_state", "reduction_path"),
                    signal_weights={"vector_path": 14.0, "sync_path": 18.0},
                ),
            ),
        ),
        "row_max": FamilySpec(
            kernel="row_max",
            catalog_builder=lambda hw: row_reduction_catalog(hw, reduction_kind="max"),
            required_shape_keys=("M", "N"),
            base_modules=("row_max_row_tile_resident", "row_max_warp_reduction_tree", "row_max_writeback"),
            optional_modules=(
                OptionalModuleSpec(module_id="row_max_vector_row_load", signals=("vector_path",)),
                OptionalModuleSpec(module_id="row_max_shared_warp_exchange", signals=("sync_path",), gate_param="ROW_REDUCE_SHARED_STAGE"),
            ),
            params=(
                ParamSpec(name="ROW_REDUCE_BLOCK_THREADS", role="threads", dim_aliases=("block_threads", "threads_per_block", "num_warps"), defaults=(64, 128, 256), allowed_values=(32, 64, 128, 256)),
                ParamSpec(name="ROW_REDUCE_VECTOR_WIDTH", role="vector_width", dim_aliases=("vector_width", "size_per_thread"), defaults=(1, 2, 4), allowed_values=(1, 2, 4)),
                ParamSpec(name="ROW_REDUCE_SHARED_STAGE", role="shared_stage", defaults=(1, 0), allowed_values=(0, 1)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="row_max_axis1_v2",
                    module_id="row_max_backend_v2",
                    param_names=("ROW_REDUCE_BLOCK_THREADS", "ROW_REDUCE_VECTOR_WIDTH", "ROW_REDUCE_SHARED_STAGE"),
                    required_signals=("resident_state", "reduction_path"),
                    signal_weights={"vector_path": 14.0, "sync_path": 18.0},
                ),
            ),
        ),
        "add2d": FamilySpec(
            kernel="add2d",
            catalog_builder=lambda hw: elementwise2d_catalog(hw, op_kind="add"),
            required_shape_keys=("M", "N"),
            base_modules=("elementwise_add_tile_resident", "elementwise_add_two_axis_grid_mapping", "elementwise_add_add_primitive", "elementwise_add_backend_v1"),
            optional_modules=(
                OptionalModuleSpec(module_id="elementwise_add_vector_global_io", signals=("vector_path",)),
                OptionalModuleSpec(module_id="elementwise_add_masked_edge_handling", signals=("sync_path", "vector_path")),
            ),
            params=(
                ParamSpec(name="ELEMENTWISE_BLOCK_THREADS", role="threads", dim_aliases=("block_threads", "threads_per_block", "num_warps"), defaults=(128, 256, 512), allowed_values=(64, 128, 256, 512)),
                ParamSpec(name="ELEMENTWISE_VECTOR_WIDTH", role="vector_width", dim_aliases=("vector_width", "size_per_thread"), defaults=(1, 2, 4), allowed_values=(1, 2, 4)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="elementwise_v1",
                    module_id="elementwise_add_backend_v1",
                    param_names=("ELEMENTWISE_BLOCK_THREADS", "ELEMENTWISE_VECTOR_WIDTH"),
                    required_signals=("resident_state", "vector_path"),
                    signal_weights={"vector_path": 20.0},
                ),
            ),
        ),
        "exp2d": FamilySpec(
            kernel="exp2d",
            catalog_builder=lambda hw: elementwise2d_catalog(hw, op_kind="exp"),
            required_shape_keys=("M", "N"),
            base_modules=("elementwise_exp_tile_resident", "elementwise_exp_two_axis_grid_mapping", "elementwise_exp_exp_primitive", "elementwise_exp_backend_v1"),
            optional_modules=(
                OptionalModuleSpec(module_id="elementwise_exp_vector_global_io", signals=("vector_path",)),
                OptionalModuleSpec(module_id="elementwise_exp_masked_edge_handling", signals=("sync_path", "vector_path")),
            ),
            params=(
                ParamSpec(name="ELEMENTWISE_BLOCK_THREADS", role="threads", dim_aliases=("block_threads", "threads_per_block", "num_warps"), defaults=(128, 256, 512), allowed_values=(64, 128, 256, 512)),
                ParamSpec(name="ELEMENTWISE_VECTOR_WIDTH", role="vector_width", dim_aliases=("vector_width", "size_per_thread"), defaults=(1, 2, 4), allowed_values=(1, 2, 4)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="elementwise_v1",
                    module_id="elementwise_exp_backend_v1",
                    param_names=("ELEMENTWISE_BLOCK_THREADS", "ELEMENTWISE_VECTOR_WIDTH"),
                    required_signals=("resident_state", "vector_path"),
                    signal_weights={"vector_path": 20.0},
                ),
            ),
        ),
        "layer_norm_persistent": FamilySpec(
            kernel="layer_norm_persistent",
            catalog_builder=layer_norm_persistent_catalog,
            required_shape_keys=("M", "N"),
            base_modules=("layer_norm_row_tile_resident", "layer_norm_warp_statistics", "layer_norm_affine_epilogue", "layer_norm_backend_v1"),
            optional_modules=(
                OptionalModuleSpec(module_id="layer_norm_register_stage", signals=("vector_path",)),
                OptionalModuleSpec(module_id="layer_norm_persistent_row_cache", signals=("persistent_path",), gate_param="LAYER_NORM_PERSISTENT_ROW"),
            ),
            params=(
                ParamSpec(name="LAYER_NORM_BLOCK_THREADS", role="threads", dim_aliases=("block_threads", "threads_per_block", "num_warps"), defaults=(32, 64, 128, 256), allowed_values=(32, 64, 128, 256)),
                ParamSpec(name="LAYER_NORM_VECTOR_WIDTH", role="vector_width", dim_aliases=("vector_width", "size_per_thread"), defaults=(1, 2, 4), allowed_values=(1, 2, 4)),
                ParamSpec(name="LAYER_NORM_PERSISTENT_ROW", role="persistent_stage", dim_aliases=("persistent_row",), defaults=(1, 0), allowed_values=(0, 1)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="layer_norm_axis1_v1",
                    module_id="layer_norm_backend_v1",
                    param_names=("LAYER_NORM_BLOCK_THREADS", "LAYER_NORM_VECTOR_WIDTH", "LAYER_NORM_PERSISTENT_ROW"),
                    required_signals=("resident_state", "reduction_path", "fused_epilogue"),
                    signal_weights={"persistent_path": 24.0, "vector_path": 10.0},
                ),
            ),
        ),
        "group_norm_kernel": FamilySpec(
            kernel="group_norm_kernel",
            catalog_builder=group_norm_kernel_catalog,
            required_shape_keys=("N", "GROUP_SIZE"),
            base_modules=("group_norm_group_tile_resident", "group_norm_warp_reduction", "group_norm_online_normalization", "group_norm_affine_fused_epilogue", "group_norm_backend_v1"),
            optional_modules=(OptionalModuleSpec(module_id="group_norm_vector_group_io", signals=("vector_path",)),),
            params=(
                ParamSpec(name="GROUP_NORM_BLOCK_THREADS", role="threads", dim_aliases=("block_threads",), defaults=(64, 128, 256), allowed_values=(64, 128, 256)),
                ParamSpec(name="GROUP_NORM_VECTOR_WIDTH", role="vector_width", dim_aliases=("vector_width",), defaults=(1, 2, 4), allowed_values=(1, 2, 4)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="group_norm_v1",
                    module_id="group_norm_backend_v1",
                    param_names=("GROUP_NORM_BLOCK_THREADS", "GROUP_NORM_VECTOR_WIDTH"),
                    required_signals=("resident_state", "reduction_path", "fused_epilogue"),
                    signal_weights={"vector_path": 10.0},
                ),
            ),
        ),
        "masked_attention2d": FamilySpec(
            kernel="masked_attention2d",
            catalog_builder=masked_attention2d_catalog,
            required_shape_keys=("Q_CTX", "KV_CTX", "HEAD_DIM"),
            base_modules=("masked_attn_q_resident_state", "masked_attn_tiny_kv_stage", "masked_attn_mask_causal_apply"),
            optional_modules=(
                OptionalModuleSpec(module_id="masked_attn_parallel_softmax", signals=("online_state",)),
                OptionalModuleSpec(module_id="masked_attn_vector_dot_fragment", signals=("vector_path",)),
            ),
            params=(
                ParamSpec(name="ATTN_SCORE_WARPS", role="score_warps", dim_aliases=("score_warps",), defaults=(4, 6), allowed_values=(2, 4, 6)),
                ParamSpec(name="MASKED_ATTN_SHARED_STAGE", role="shared_stage", defaults=(1, 0), allowed_values=(0, 1)),
                ParamSpec(name="MASKED_ATTN_VECTOR_WIDTH", role="vector_width", dim_aliases=("vector_width",), defaults=(1, 2), allowed_values=(1, 2, 4)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="attn2d_causal_softmax_v18",
                    module_id="masked_attn_backend_v18",
                    param_names=("ATTN_SCORE_WARPS", "MASKED_ATTN_SHARED_STAGE", "MASKED_ATTN_VECTOR_WIDTH"),
                    enabled_flags=("MASKED_ATTN_SHARED_STAGE",),
                    required_signals=("resident_state", "stream_stage", "mask_path", "online_state"),
                    signal_weights={"vector_path": 6.0},
                ),
                TemplateSpec(
                    kernel_kind="attn2d_causal_softmax_v14",
                    module_id="masked_attn_backend_v14",
                    param_names=("ATTN_SCORE_WARPS", "MASKED_ATTN_SHARED_STAGE", "MASKED_ATTN_VECTOR_WIDTH"),
                    enabled_flags=("MASKED_ATTN_SHARED_STAGE",),
                    required_signals=("resident_state", "stream_stage", "mask_path", "online_state"),
                    signal_weights={"vector_path": 4.0},
                ),
                TemplateSpec(
                    kernel_kind="attn2d_causal_softmax_v10",
                    module_id="masked_attn_backend_v10",
                    param_names=("ATTN_SCORE_WARPS", "MASKED_ATTN_VECTOR_WIDTH"),
                    required_signals=("resident_state", "stream_stage", "mask_path", "vector_path"),
                    signal_weights={},
                ),
            ),
        ),
        "matmul_fused_epilogue2d": FamilySpec(
            kernel="matmul_fused_epilogue2d",
            catalog_builder=matmul_fused_epilogue2d_catalog,
            required_shape_keys=("M", "N", "K"),
            base_modules=("ab_tile_stage", "epilogue_fused_writeback"),
            optional_modules=(
                OptionalModuleSpec(module_id="mma_core", signals=("mma_path",)),
                OptionalModuleSpec(module_id="prefetch_pipeline", signals=("async_pipeline",), gate_param="MMA_ASYNC_COPY"),
            ),
            params=(
                ParamSpec(name="MMA_BM", role="tile_m", dim_aliases=("tile_m", "MMA_BM"), defaults=(32, 64), allowed_values=(16, 32, 64)),
                ParamSpec(name="MMA_BN", role="tile_n", dim_aliases=("tile_n", "MMA_BN"), defaults=(16, 32), allowed_values=(16, 32, 64)),
                ParamSpec(name="MMA_BK", role="tile_k", dim_aliases=("tile_k", "MMA_BK"), defaults=(32, 16), allowed_values=(8, 16, 32, 64)),
                ParamSpec(name="MMA_ASYNC_COPY", role="async_copy", defaults=(1, 0), allowed_values=(0, 1)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="matmul_mma_tf32_v1",
                    module_id="backend_mma_v1",
                    param_names=("MMA_BM", "MMA_BN", "MMA_BK", "MMA_ASYNC_COPY"),
                    required_signals=("mma_path", "fused_epilogue"),
                    signal_weights={"async_pipeline": 10.0},
                    cluster_bonus={"cuda_tc_mid_smem": 12.0, "cuda_tc_large_smem": 18.0},
                ),
                TemplateSpec(
                    kernel_kind="matmul_tile_v2",
                    module_id="backend_tile_v2",
                    param_names=(),
                    required_signals=("fused_epilogue",),
                    signal_weights={},
                    cluster_bonus={"cuda_tc_mid_smem": 10.0},
                    portability_note="cluster_prefers_tile_v2",
                ),
                TemplateSpec(
                    kernel_kind="matmul_tile_v1",
                    module_id="backend_tile_v1",
                    param_names=(),
                    required_signals=("fused_epilogue",),
                    signal_weights={},
                ),
            ),
        ),
        "ai_bench_matmul": FamilySpec(
            kernel="ai_bench_matmul",
            catalog_builder=ai_bench_matmul_catalog,
            required_shape_keys=("M", "N", "K"),
            base_modules=("ai_matmul_operand_tile_stage", "ai_matmul_tile_fallback"),
            optional_modules=(
                OptionalModuleSpec(module_id="ai_matmul_mma_core", signals=("mma_path",)),
                OptionalModuleSpec(module_id="ai_matmul_async_prefetch", signals=("async_pipeline",), gate_param="MMA_ASYNC_COPY"),
            ),
            params=(
                ParamSpec(name="MMA_BM", role="tile_m", dim_aliases=("tile_m", "MMA_BM"), defaults=(64, 32), allowed_values=(16, 32, 64)),
                ParamSpec(name="MMA_BN", role="tile_n", dim_aliases=("tile_n", "MMA_BN"), defaults=(16, 32), allowed_values=(16, 32, 64)),
                ParamSpec(name="MMA_BK", role="tile_k", dim_aliases=("tile_k", "MMA_BK"), defaults=(32, 16), allowed_values=(8, 16, 32, 64)),
                ParamSpec(name="MMA_ASYNC_COPY", role="async_copy", defaults=(1, 0), allowed_values=(0, 1)),
            ),
            templates=(
                TemplateSpec(
                    kernel_kind="matmul_mma_tf32_v2",
                    module_id="ai_matmul_backend_mma_v2",
                    param_names=("MMA_BM", "MMA_BN", "MMA_BK", "MMA_ASYNC_COPY"),
                    enabled_flags=("MMA_ASYNC_COPY",),
                    required_signals=("mma_path", "async_pipeline"),
                    signal_weights={"async_evidence": 18.0},
                    cluster_bonus={"cuda_tc_large_smem": 18.0, "cuda_tc_mid_smem": 8.0},
                ),
                TemplateSpec(
                    kernel_kind="matmul_mma_tf32_v1",
                    module_id="ai_matmul_backend_mma_v1",
                    param_names=("MMA_BM", "MMA_BN", "MMA_BK"),
                    required_signals=("mma_path",),
                    signal_weights={},
                    cluster_bonus={"cuda_tc_large_smem": 12.0, "cuda_tc_mid_smem": 8.0},
                ),
                TemplateSpec(
                    kernel_kind="matmul_mma_tf32_global_v1",
                    module_id="ai_matmul_backend_global_v1",
                    param_names=("MMA_BM", "MMA_BN", "MMA_BK"),
                    required_signals=("mma_path",),
                    signal_weights={},
                ),
                TemplateSpec(
                    kernel_kind="matmul_tile_v2",
                    module_id="ai_matmul_backend_tile_v2",
                    param_names=(),
                    required_signals=(),
                    signal_weights={},
                ),
            ),
        ),
    }


def plan_cuda_kernel(
    kernel: str,
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
    specs = _family_specs()
    kernel_name = str(kernel).strip()
    spec = specs.get(kernel_name)
    if spec is None:
        raise ValueError(f"unsupported universal cuda planner kernel={kernel_name!r}")
    del toolchain_model
    for key in list(spec.required_shape_keys or ()):
        _require_dim(shape_bindings, key)
    modules, module_edges, _passes = spec.catalog_builder(hardware_model)
    profile = _graph_profile(org, shape_bindings=shape_bindings, ttgir_facts=ttgir_facts, ptx_facts=ptx_facts)
    ranked = _generate_candidate_evals(
        spec=spec,
        profile=profile,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
    )
    limit = max(1, int(budget))
    ranked = list(ranked[:limit])
    selected_modules, selected_edges = _selected_modules(
        spec=spec,
        modules=modules,
        module_edges=module_edges,
        profile=profile,
        ranked=ranked,
    )
    candidates = [item.candidate for item in list(ranked or [])]
    any_shared_fit = any(item.resource_fit.allowed for item in list(ranked or []) if any(_norm_token(x).endswith("shared_stage") for x in item.enabled_params))
    notes = list(profile.notes)
    notes.append(f"engine=universal_graph_v1")
    notes.append(f"graph_signals={','.join(sorted(set(profile.signals)))}")
    notes.append(f"shared_bytes={int(profile.shared_bytes)}")
    notes.append(f"register_bytes={int(profile.register_bytes)}")
    if any(_norm_token(getattr(param, 'role', '')) == "shared_stage" for param in list(spec.params or ())):
        notes.append(f"topology_any_shared_stage_fit={bool(any_shared_fit)}")
    substitutions: list[dict[str, Any]] = []
    incomplete_async_evidence = profile.has_signal("async_pipeline") and not profile.has_signal("async_evidence")
    if incomplete_async_evidence:
        substitutions.append({"from": "pipeline.async_prefetch", "to": "pipeline.sync_prefetch", "reason": "incomplete async evidence"})
    if profile.has_signal("mma_path") and not hardware_model.supports_mma:
        substitutions.append({"from": "primitive.mma", "to": "primitive.tile", "reason": "hardware_model.supports_mma = false"})
    if incomplete_async_evidence:
        source_kind = str(source_oracle.get("kernel_kind") or kernel_name).strip() or kernel_name
        notes.append(f"preserve:async_portability={source_kind}")
    param_space: dict[str, Any] = {"kernel_kind": [str(item.candidate.kernel_kind) for item in list(ranked or [])]}
    source_bindings = {str(k): int(v) for k, v in dict(source_oracle.get("bindings") or {}).items() if str(k).strip()}
    for param in list(spec.params or ()):
        values = _resolve_param_values(
            param=param,
            profile=profile,
            shape_bindings=shape_bindings,
            source_bindings=source_bindings,
            hardware_model=hardware_model,
        )
        param_space[str(param.name)] = list(values)
    constraints = [f"{key} > 0" for key in list(spec.required_shape_keys or ())]
    for param in list(spec.params or ()):
        if param.allowed_values:
            allowed = ",".join(str(int(x)) for x in list(param.allowed_values or ()))
            constraints.append(f"{param.name} in {{{allowed}}}")
    return BackendPlan(
        kernel=str(kernel_name),
        source_oracle=dict(source_oracle or {}),
        hardware_model=hardware_model.to_json_dict(),
        selected_modules=selected_modules,
        module_edges=selected_edges,
        param_space=param_space,
        constraints=constraints,
        substitutions=substitutions,
        candidates=candidates,
        notes=notes,
    )


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
    return plan_cuda_kernel(
        "flash_attention2d",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        toolchain_model=toolchain_model,
        budget=budget,
    )


def plan_attn_fwd(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "_attn_fwd",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_softmax_inner(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "softmax_inner",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_masked_softmax2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "masked_softmax2d",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_ai_bench_softmax(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "ai_bench_softmax",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_row_sum(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "row_sum",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_row_max(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "row_max",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_add2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "add2d",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_exp2d(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "exp2d",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_layer_norm_persistent(
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
    return plan_cuda_kernel(
        "layer_norm_persistent",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        toolchain_model=toolchain_model,
        budget=budget,
    )


def plan_group_norm_kernel(
    org: OrgDoc,
    *,
    shape_bindings: Mapping[str, Any],
    source_oracle: Mapping[str, Any],
    hardware_model: HardwareModel,
    ttgir_facts: Mapping[str, Any] | None = None,
    ptx_facts: Mapping[str, Any] | None = None,
    budget: int = 32,
) -> BackendPlan:
    return plan_cuda_kernel(
        "group_norm_kernel",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        budget=budget,
    )


def plan_masked_attention2d(
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
    return plan_cuda_kernel(
        "masked_attention2d",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        toolchain_model=toolchain_model,
        budget=budget,
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
    return plan_cuda_kernel(
        "ai_bench_matmul",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        toolchain_model=toolchain_model,
        budget=budget,
    )


def plan_matmul_fused_epilogue2d(
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
    return plan_cuda_kernel(
        "matmul_fused_epilogue2d",
        org,
        shape_bindings=shape_bindings,
        source_oracle=source_oracle,
        hardware_model=hardware_model,
        ttgir_facts=ttgir_facts,
        ptx_facts=ptx_facts,
        toolchain_model=toolchain_model,
        budget=budget,
    )


__all__ = [
    "plan_add2d",
    "plan_ai_bench_matmul",
    "plan_ai_bench_softmax",
    "plan_attn_fwd",
    "plan_cuda_kernel",
    "plan_exp2d",
    "plan_flash_attention2d",
    "plan_group_norm_kernel",
    "plan_layer_norm_persistent",
    "plan_masked_attention2d",
    "plan_masked_softmax2d",
    "plan_matmul_fused_epilogue2d",
    "plan_row_max",
    "plan_row_sum",
    "plan_softmax_inner",
]
