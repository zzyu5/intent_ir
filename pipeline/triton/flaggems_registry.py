"""
FlagGems semantic-op registry and capability/status matrix.

Source of truth for coverage baseline:
- `flag_gems.ops.__all__` (semantic-op granularity)
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backends.capability import check_dual_backend_support
from pipeline.triton.flaggems_semantic_rules import resolve_semantic_mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("flaggems_registry.json")
DEFAULT_FLAGGEMS_OPSET = "deterministic_forward"
STATUS_VALUES = {"dual_pass", "rvv_only", "cuda_only", "blocked_ir", "blocked_backend"}
FAMILY_ORDER = [
    "elementwise_broadcast",
    "reduction",
    "norm_activation",
    "index_scatter_gather",
    "matmul_linear",
    "conv_pool_interp",
    "attention_sequence",
]

_RANDOM_OP_HINTS = {
    "rand",
    "randn",
    "bernoulli",
    "dropout",
    "uniform",
    "normal",
    "multinomial",
    "poisson",
    "cauchy",
    "geometric",
    "log_normal",
    "exponential",
}

_NON_SEMANTIC_OP_HINTS = {
    "scheduler",
    "metadata",
    "meta",
    "capability",
    "registry",
}

_NON_SEMANTIC_OP_EXACT = {
    "get_scheduler_metadata",
}

# Semantic op -> implemented Triton KernelSpec (if any).
_SEMANTIC_TO_E2E_SPEC: dict[str, str] = {
    "any": "any_kernel_dim",
    "add": "add2d",
    "sub": "sub2d",
    "mul": "mul2d",
    "div": "div2d",
    "eq": "eq2d",
    "ne": "ne2d",
    "gt": "gt2d",
    "ge": "ge2d",
    "lt": "lt2d",
    "le": "le2d",
    "neg": "neg2d",
    "ceil": "ceil2d",
    "reciprocal": "reciprocal2d",
    "sqrt": "sqrt2d",
    "exp2": "exp22d",
    "silu": "silu2d",
    "tanh": "tanh2d",
    "sigmoid": "sigmoid2d",
    "mm": "mm2d",
    "bmm": "bmm3d",
    "addmm": "addmm2d",
    "baddbmm": "baddbmm3d",
    "dot": "dot1d",
    "vdot": "vdot1d",
    "mv": "mv2d",
    "addmv": "addmv2d",
    "logical_and": "logical_and2d",
    "logical_or": "logical_or2d",
    "logical_not": "logical_not2d",
    "logical_xor": "logical_xor2d",
    "abs": "abs2d",
    "rsqrt": "rsqrt2d",
    "relu": "relu2d",
    "exp": "exp2d",
    "isclose": "isclose2d",
    "allclose": "allclose2d",
    "isfinite": "isfinite2d",
    "isinf": "isinf2d",
    "isnan": "isnan2d",
    "masked_fill": "masked_fill2d",
    "threshold": "threshold2d",
    "where_self": "where2d",
    "gather": "gather2d",
    "index": "index_select2d",
    "index_select": "index_select2d",
    "sum": "row_sum",
    "mean": "row_mean",
    "all": "row_all",
    "max": "row_max",
    "maximum": "maximum2d",
    "minimum": "minimum2d",
    "full": "full2d",
    "ones": "full2d",
    "zeros": "full2d",
    "copy": "identity2d",
    "contiguous": "identity2d",
    "resolve_conj": "identity2d",
    "resolve_neg": "identity2d",
    "to_copy": "cast2d",
    "clamp": "clamp2d",
    "group_norm": "group_norm_kernel",
    "batch_norm": "batch_norm2d",
    "layer_norm": "layer_norm_persistent",
    "rms_norm": "rms_norm2d",
    "lerp": "lerp2d",
    "softmax": "softmax_inner",
    "upsample_bicubic2d_aa": "upsample_bicubic2d_aa",
}

_SEMANTIC_E2E_ALIASES: dict[str, str] = {
    "sum_dim": "sum",
    "mean_dim": "mean",
    "amax": "max",
    "max_dim": "max",
    "where_scalar_self": "where_self",
    "where_scalar_other": "where_self",
    "true_divide": "div",
    "floor_divide": "div",
    "div_mode": "div",
    "eq_scalar": "eq",
    "equal": "eq",
    "ne_scalar": "ne",
    "gt_scalar": "gt",
    "ge_scalar": "ge",
    "lt_scalar": "lt",
    "le_scalar": "le",
    "fill_scalar": "full",
    "fill_tensor": "full",
    "full_like": "full",
    "ones_like": "ones",
    "zeros_like": "zeros",
    "clamp_min": "maximum",
    "clamp_tensor": "clamp",
    "lerp_scalar": "lerp",
    "lerp_tensor": "lerp",
    "rms_norm_forward": "rms_norm",
    "scaled_softmax_forward": "softmax",
}

_E2E_SPEC_TO_SEMANTIC: dict[str, str] = {v: k for k, v in _SEMANTIC_TO_E2E_SPEC.items()}


def ensure_flaggems_importable(flaggems_src: str | Path | None = None) -> None:
    candidates: list[Path] = []
    if flaggems_src is not None:
        candidates.append(Path(str(flaggems_src)))
    env = os.getenv("FLAGGEMS_SRC")
    if isinstance(env, str) and env.strip():
        candidates.append(Path(env.strip()))
    candidates.append(ROOT / "experiment" / "FlagGems" / "src")

    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_flaggems_all_ops(flaggems_src: str | Path | None = None) -> list[str]:
    ensure_flaggems_importable(flaggems_src)
    from flag_gems import ops as fg_ops  # noqa: PLC0415

    raw = list(getattr(fg_ops, "__all__", []) or [])
    out: list[str] = []
    for x in raw:
        if not isinstance(x, str) or not x:
            continue
        obj = getattr(fg_ops, x, None)
        if not callable(obj):
            continue
        out.append(str(x))
    return out


def normalize_semantic_name(op_name: str) -> str:
    s = str(op_name).strip()
    if not s:
        return s
    if s.startswith("_"):
        s = s[1:]
    if s.endswith("_out"):
        s = s[:-4]
    if s.endswith("_") and s:
        s = s[:-1]
    if s.endswith("_dims") and s[:-5] in {"any", "all"}:
        s = s[:-5]
    elif s.endswith("_dim") and s[:-4] in {"any", "all"}:
        s = s[:-4]
    if s in {"arange_start", "arange_start_step"}:
        return "arange"
    return s


def is_deterministic_forward_op(op_name: str) -> bool:
    name = str(op_name).strip()
    if not name:
        return False
    low = name.lower()
    if low in _NON_SEMANTIC_OP_EXACT:
        return False
    if "backward" in low:
        return False
    if low.endswith("_"):
        return False
    if any(tok in low for tok in _NON_SEMANTIC_OP_HINTS):
        return False
    if any(tok in low for tok in _RANDOM_OP_HINTS):
        return False
    return True


def classify_semantic_family(semantic_op: str) -> str:
    s = str(semantic_op).lower()
    if any(k in s for k in ("attention", "rope", "embedding", "causal")):
        return "attention_sequence"
    if any(k in s for k in ("conv", "pool", "upsample", "interpolate", "pixel_shuffle", "unfold")):
        return "conv_pool_interp"
    if any(k in s for k in ("matmul", "linear", "mm", "bmm", "dot")):
        return "matmul_linear"
    if any(k in s for k in ("gather", "scatter", "index", "select", "slice", "take", "where", "flip", "roll")):
        return "index_scatter_gather"
    if any(k in s for k in ("norm", "relu", "gelu", "silu", "sigmoid", "tanh", "softplus", "softsign")):
        return "norm_activation"
    if any(
        k in s
        for k in (
            "reduce",
            "sum",
            "mean",
            "amax",
            "amin",
            "argmax",
            "argmin",
            "var",
            "std",
            "any",
            "all",
            "prod",
            "softmax",
            "logsumexp",
            "cumsum",
        )
    ):
        return "reduction"
    return "elementwise_broadcast"


def semantic_to_intent_ops(semantic_op: str) -> list[str]:
    return list(resolve_semantic_mapping(semantic_op).intent_ops)


def e2e_spec_for_semantic(semantic_op: str) -> str | None:
    s = str(semantic_op)
    if s in _SEMANTIC_TO_E2E_SPEC:
        return _SEMANTIC_TO_E2E_SPEC[s]
    alias = _SEMANTIC_E2E_ALIASES.get(s)
    if alias is not None:
        return _SEMANTIC_TO_E2E_SPEC.get(alias)
    return None


def semantic_for_e2e_spec(spec_name: str) -> str | None:
    return _E2E_SPEC_TO_SEMANTIC.get(str(spec_name))


def _derive_status(
    *,
    intent_ops: list[str],
    has_e2e_spec: bool,
    backend_support: dict[str, Any],
) -> tuple[str, str]:
    if not intent_ops:
        return "blocked_ir", "no_intentir_mapping"

    rvv_ok = bool(((backend_support.get("rvv") or {}).get("ok")))
    h100_ok = bool(((backend_support.get("cuda_h100") or {}).get("ok")))
    g5090_ok = bool(((backend_support.get("cuda_5090d") or {}).get("ok")))
    cuda_ok = h100_ok and g5090_ok

    if not has_e2e_spec:
        return "blocked_backend", "missing_e2e_spec"
    if rvv_ok and cuda_ok:
        return "dual_pass", "all_targets_supported_with_e2e_spec"
    if rvv_ok and not cuda_ok:
        return "rvv_only", "cuda_target_missing_ops"
    if cuda_ok and not rvv_ok:
        return "cuda_only", "rvv_target_missing_ops"
    return "blocked_backend", "backend_missing_ops"


def build_registry(
    *,
    all_ops: list[str],
    flaggems_commit: str | None = None,
    flaggems_source: str | None = None,
    opset: str = DEFAULT_FLAGGEMS_OPSET,
    generated_at: str | None = None,
) -> dict:
    if str(opset) != DEFAULT_FLAGGEMS_OPSET:
        raise ValueError(f"unsupported flaggems opset: {opset}")
    if not isinstance(flaggems_commit, str) or not flaggems_commit.strip():
        raise ValueError("flaggems_commit must be provided when building registry")

    filtered: list[str] = []
    for name in all_ops:
        if opset == DEFAULT_FLAGGEMS_OPSET and not is_deterministic_forward_op(name):
            continue
        filtered.append(str(name))

    grouped: dict[str, list[str]] = {}
    for src_name in filtered:
        sem = normalize_semantic_name(src_name)
        if not sem:
            continue
        grouped.setdefault(sem, []).append(src_name)

    entries: list[dict[str, Any]] = []
    for sem, source_ops in grouped.items():
        family = classify_semantic_family(sem)
        mapping = resolve_semantic_mapping(sem)
        intent_ops = list(mapping.intent_ops)
        backend_support = check_dual_backend_support(intent_ops)
        e2e_spec = e2e_spec_for_semantic(sem)
        status, reason = _derive_status(
            intent_ops=intent_ops,
            has_e2e_spec=bool(e2e_spec),
            backend_support=backend_support,
        )
        detail = mapping.status_reason_detail
        if reason == "missing_e2e_spec":
            detail = f"{detail}; no e2e spec registered"
        elif reason in {"cuda_target_missing_ops", "rvv_target_missing_ops", "backend_missing_ops"}:
            missing_detail: list[str] = []
            rvv_missing = list(((backend_support.get("rvv") or {}).get("missing_ops") or []))
            h100_missing = list(((backend_support.get("cuda_h100") or {}).get("missing_ops") or []))
            g5090_missing = list(((backend_support.get("cuda_5090d") or {}).get("missing_ops") or []))
            if rvv_missing:
                missing_detail.append(f"rvv_missing={rvv_missing}")
            if h100_missing:
                missing_detail.append(f"cuda_h100_missing={h100_missing}")
            if g5090_missing:
                missing_detail.append(f"cuda_5090d_missing={g5090_missing}")
            if missing_detail:
                detail = f"{detail}; " + "; ".join(missing_detail)
        if status not in STATUS_VALUES:
            raise RuntimeError(f"invalid status generated: {status}")
        entries.append(
            {
                "semantic_op": sem,
                "source_ops": sorted(set(source_ops)),
                "family": family,
                "intent_ops": list(intent_ops),
                "mapping_kind": mapping.mapping_kind,
                "intent_pattern_id": mapping.intent_pattern_id,
                "e2e_spec": e2e_spec,
                "status": status,
                "status_reason": reason,
                "status_reason_detail": detail,
                "backend_support": backend_support,
            }
        )

    family_rank = {k: i for i, k in enumerate(FAMILY_ORDER)}
    entries.sort(key=lambda x: (family_rank.get(str(x.get("family")), 999), str(x.get("semantic_op"))))

    counts_by_status: dict[str, int] = {}
    counts_by_family: dict[str, int] = {}
    for e in entries:
        s = str(e["status"])
        f = str(e["family"])
        counts_by_status[s] = counts_by_status.get(s, 0) + 1
        counts_by_family[f] = counts_by_family.get(f, 0) + 1

    return {
        "schema_version": "flaggems_registry_v2",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "opset": str(opset),
        "flaggems_commit": str(flaggems_commit),
        "flaggems_source": (str(flaggems_source) if flaggems_source else None),
        "family_order": list(FAMILY_ORDER),
        "counts": {
            "source_all_ops": int(len(all_ops)),
            "source_filtered_ops": int(len(filtered)),
            "semantic_ops": int(len(entries)),
            "by_status": dict(sorted(counts_by_status.items(), key=lambda kv: kv[0])),
            "by_family": dict(sorted(counts_by_family.items(), key=lambda kv: kv[0])),
        },
        "entries": entries,
    }


def write_registry(path: str | Path, payload: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return out


def load_registry(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    default_src = ROOT / "experiment" / "FlagGems" / "src"
    all_ops = load_flaggems_all_ops(flaggems_src=default_src)
    commit = infer_flaggems_commit_from_src(default_src)
    if not commit:
        raise RuntimeError(f"unable to infer FlagGems commit from {default_src}")
    return build_registry(all_ops=all_ops, flaggems_commit=commit, flaggems_source=str(default_src))


def load_registry_entry_by_semantic(registry: dict, semantic_op: str) -> dict | None:
    sem = str(semantic_op)
    for e in (registry.get("entries") or []):
        if isinstance(e, dict) and str(e.get("semantic_op")) == sem:
            return e
    return None


def load_registry_entry_by_spec(registry: dict, spec_name: str) -> dict | None:
    sp = str(spec_name)
    for e in (registry.get("entries") or []):
        if isinstance(e, dict) and str(e.get("e2e_spec") or "") == sp:
            return e
    return None


def list_supported_e2e_specs(registry: dict) -> list[str]:
    out: list[str] = []
    for e in (registry.get("entries") or []):
        if not isinstance(e, dict):
            continue
        spec = e.get("e2e_spec")
        if isinstance(spec, str) and spec and spec not in out:
            out.append(spec)
    return out


def infer_flaggems_commit_from_src(flaggems_src: str | Path | None) -> str | None:
    if flaggems_src is None:
        return None
    p = Path(str(flaggems_src)).resolve()
    # Expected layout: <repo>/src
    repo = p.parent if p.name == "src" else p
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{40}", text):
        return text
    if text.startswith("ref:"):
        ref = text.split(":", 1)[1].strip()
        ref_path = repo / ".git" / ref
        if ref_path.exists():
            sha = ref_path.read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[0-9a-f]{40}", sha):
                return sha
    return None


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "DEFAULT_FLAGGEMS_OPSET",
    "STATUS_VALUES",
    "FAMILY_ORDER",
    "ensure_flaggems_importable",
    "load_flaggems_all_ops",
    "normalize_semantic_name",
    "is_deterministic_forward_op",
    "classify_semantic_family",
    "semantic_to_intent_ops",
    "e2e_spec_for_semantic",
    "semantic_for_e2e_spec",
    "build_registry",
    "write_registry",
    "load_registry",
    "load_registry_entry_by_semantic",
    "load_registry_entry_by_spec",
    "list_supported_e2e_specs",
    "infer_flaggems_commit_from_src",
]
