from __future__ import annotations

ORG_SCHEMA_VERSION_V1 = "intentir_org_v1"
BACKEND_PLAN_SCHEMA_VERSION_V1 = "intentir_backend_plan_v1"

ORG_NODE_TYPES: tuple[str, ...] = (
    "tiling",
    "staging",
    "overlap_pipeline",
    "parallel_mapping",
    "communication",
    "special_primitive",
)

__all__ = [
    "BACKEND_PLAN_SCHEMA_VERSION_V1",
    "ORG_NODE_TYPES",
    "ORG_SCHEMA_VERSION_V1",
]

