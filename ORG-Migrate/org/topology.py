from __future__ import annotations

from typing import Any

from .schema import OrgDoc, OrgMechanism, OrgMechanismTopologyEdge, OrgTensor, OrgTensorLifetime


def _norm_token(x: Any) -> str:
    return str(x or "").strip().lower().replace("-", "_").replace(" ", "_")


def mechanism_by_id(org: OrgDoc) -> dict[str, OrgMechanism]:
    return {str(item.id): item for item in list(getattr(org, "mechanisms", []) or []) if str(getattr(item, "id", "")).strip()}


def tensor_by_id(org: OrgDoc) -> dict[str, OrgTensor]:
    return {str(item.id): item for item in list(getattr(org, "tensors", []) or []) if str(getattr(item, "id", "")).strip()}


def lifetime_by_id(org: OrgDoc) -> dict[str, OrgTensorLifetime]:
    return {
        str(item.id): item for item in list(getattr(org, "tensor_lifetimes", []) or []) if str(getattr(item, "id", "")).strip()
    }


def mechanism_tag_map(org: OrgDoc) -> dict[str, str]:
    return {
        str(item.id): str(item.tag)
        for item in list(getattr(org, "mechanisms", []) or [])
        if str(getattr(item, "id", "")).strip() and str(getattr(item, "tag", "")).strip()
    }


def find_tensor_ids(org: OrgDoc, *tokens: str) -> set[str]:
    wanted = {_norm_token(token) for token in list(tokens or []) if _norm_token(token)}
    if not wanted:
        return set()
    out: set[str] = set()
    for tensor in list(getattr(org, "tensors", []) or []):
        candidates = {_norm_token(getattr(tensor, "name", "")), _norm_token(getattr(tensor, "role", ""))}
        for alias in list(getattr(tensor, "aliases", []) or []):
            candidates.add(_norm_token(alias))
        if wanted & candidates:
            out.add(str(getattr(tensor, "id", "")).strip())
    return {item for item in out if item}


def lifetime_mechanism_tags(org: OrgDoc, lifetime: OrgTensorLifetime) -> set[str]:
    tags = mechanism_tag_map(org)
    out: set[str] = set()
    for mech_id in list(getattr(lifetime, "producer_mechanisms", []) or []):
        tag = _norm_token(tags.get(str(mech_id)))
        if tag:
            out.add(tag)
    for mech_id in list(getattr(lifetime, "consumer_mechanisms", []) or []):
        tag = _norm_token(tags.get(str(mech_id)))
        if tag:
            out.add(tag)
    return out


def find_lifetimes(
    org: OrgDoc,
    *,
    tensor_ids: set[str] | None = None,
    storage: str | None = None,
    required_mechanism_tags: set[str] | None = None,
    required_goal_ids: set[str] | None = None,
) -> list[OrgTensorLifetime]:
    wanted_storage = _norm_token(storage)
    wanted_mechanisms = {_norm_token(x) for x in list(required_mechanism_tags or set()) if _norm_token(x)}
    wanted_goals = {str(x).strip() for x in list(required_goal_ids or set()) if str(x).strip()}
    out: list[OrgTensorLifetime] = []
    for lifetime in list(getattr(org, "tensor_lifetimes", []) or []):
        if tensor_ids and str(getattr(lifetime, "tensor", "")).strip() not in tensor_ids:
            continue
        if wanted_storage and _norm_token(getattr(lifetime, "storage", "")) != wanted_storage:
            continue
        tags = lifetime_mechanism_tags(org, lifetime)
        if wanted_mechanisms and not wanted_mechanisms.issubset(tags):
            continue
        goal_ids = {str(x).strip() for x in list(getattr(lifetime, "supports_goals", []) or []) if str(x).strip()}
        if wanted_goals and not wanted_goals.issubset(goal_ids):
            continue
        out.append(lifetime)
    return out


def topology_edges(org: OrgDoc) -> list[OrgMechanismTopologyEdge]:
    return [edge for edge in list(getattr(org, "mechanism_topology", []) or []) if isinstance(edge, OrgMechanismTopologyEdge)]


def has_mechanism_relation(
    org: OrgDoc,
    *,
    src_tags: set[str],
    dst_tags: set[str],
    relation: str | None = None,
    lifetime_ids: set[str] | None = None,
) -> bool:
    wanted_src = {_norm_token(x) for x in list(src_tags or set()) if _norm_token(x)}
    wanted_dst = {_norm_token(x) for x in list(dst_tags or set()) if _norm_token(x)}
    wanted_relation = _norm_token(relation)
    tags = mechanism_tag_map(org)
    for edge in topology_edges(org):
        src_tag = _norm_token(tags.get(str(getattr(edge, "src", ""))))
        dst_tag = _norm_token(tags.get(str(getattr(edge, "dst", ""))))
        if wanted_src and src_tag not in wanted_src:
            continue
        if wanted_dst and dst_tag not in wanted_dst:
            continue
        if wanted_relation and _norm_token(getattr(edge, "relation", "")) != wanted_relation:
            continue
        if lifetime_ids:
            edge_lifetimes = {str(x).strip() for x in list(getattr(edge, "lifetimes", []) or []) if str(x).strip()}
            if not edge_lifetimes & set(lifetime_ids):
                continue
        return True
    return False


__all__ = [
    "find_lifetimes",
    "find_tensor_ids",
    "has_mechanism_relation",
    "lifetime_by_id",
    "lifetime_mechanism_tags",
    "mechanism_by_id",
    "mechanism_tag_map",
    "tensor_by_id",
    "topology_edges",
]
