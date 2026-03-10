from __future__ import annotations

ORG_SCHEMA_VERSION_V1 = "intentir_org_v1"
BACKEND_PLAN_SCHEMA_VERSION_V1 = "intentir_backend_plan_v1"

ORG_GOAL_TAGS: tuple[str, ...] = (
    "resident_working_set",
    "streaming_softmax_state",
    "avoid_materialization",
    "latency_hiding",
    "operand_reuse",
    "mma_acceleration",
    "fused_epilogue_avoid_writeback",
    "reduction_tree_balance",
    "memory_coalescing",
    "persistent_row_state",
    "affine_epilogue_fusion",
)

ORG_GOAL_TAGS_BY_KERNEL: dict[str, tuple[str, ...]] = {
    "add2d": (
        "resident_working_set",
        "memory_coalescing",
        "avoid_materialization",
        "latency_hiding",
    ),
    "exp2d": (
        "resident_working_set",
        "memory_coalescing",
        "avoid_materialization",
        "latency_hiding",
    ),
    "flash_attention2d": (
        "resident_working_set",
        "streaming_softmax_state",
        "avoid_materialization",
        "latency_hiding",
    ),
    "_attn_fwd": (
        "resident_working_set",
        "streaming_softmax_state",
        "avoid_materialization",
        "latency_hiding",
    ),
    "masked_softmax2d": (
        "resident_working_set",
        "streaming_softmax_state",
        "avoid_materialization",
        "latency_hiding",
    ),
    "softmax_inner": (
        "resident_working_set",
        "streaming_softmax_state",
        "avoid_materialization",
        "latency_hiding",
    ),
    "row_sum": (
        "resident_working_set",
        "reduction_tree_balance",
        "memory_coalescing",
        "latency_hiding",
    ),
    "row_max": (
        "resident_working_set",
        "reduction_tree_balance",
        "memory_coalescing",
        "latency_hiding",
    ),
    "layer_norm_persistent": (
        "resident_working_set",
        "persistent_row_state",
        "memory_coalescing",
        "affine_epilogue_fusion",
        "latency_hiding",
    ),
    "group_norm_kernel": (
        "resident_working_set",
        "reduction_tree_balance",
        "memory_coalescing",
        "fused_epilogue_avoid_writeback",
        "latency_hiding",
    ),
    "matmul_fused_epilogue2d": (
        "operand_reuse",
        "mma_acceleration",
        "fused_epilogue_avoid_writeback",
        "latency_hiding",
    ),
}

ORG_MECHANISM_CATEGORIES: tuple[str, ...] = (
    "tiling",
    "staging",
    "pipeline",
    "mapping",
    "communication",
    "primitive",
    "fusion",
)

__all__ = [
    "BACKEND_PLAN_SCHEMA_VERSION_V1",
    "ORG_GOAL_TAGS",
    "ORG_GOAL_TAGS_BY_KERNEL",
    "ORG_MECHANISM_CATEGORIES",
    "ORG_SCHEMA_VERSION_V1",
]
