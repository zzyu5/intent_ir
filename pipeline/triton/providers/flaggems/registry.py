"""
FlagGems semantic-op registry and capability/status matrix.

Source of truth for coverage baseline:
- `flag_gems.ops.__all__` (semantic-op granularity)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backends.capability import check_dual_backend_support
from pipeline.triton.providers.flaggems.semantic_rules import resolve_semantic_mapping


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY_PATH = ROOT / "pipeline" / "triton" / "flaggems_registry.json"
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
    "acos": "acos2d",
    "sub": "sub2d",
    "mul": "mul2d",
    "atan": "atan2d",
    "angle": "angle2d",
    "cat": "cat2d",
    "arange": "arange1d",
    "addcmul": "addcmul2d",
    "addcdiv": "addcdiv2d",
    "addr": "addr2d",
    "bitwise_and": "bitwise_and2d",
    "bitwise_or": "bitwise_or2d",
    "bitwise_not": "bitwise_not2d",
    "bitwise_left_shift": "bitwise_left_shift2d",
    "bitwise_right_shift": "bitwise_right_shift2d",
    "avg_pool2d": "avg_pool2d_nchw",
    "argmax": "argmax2d",
    "argmin": "argmin2d",
    "count_nonzero": "count_nonzero2d",
    "diag": "diag2d",
    "diag_embed": "diag_embed2d",
    "trace": "trace2d",
    "triu": "triu2d",
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
    "tan": "tan2d",
    "softplus": "softplus2d",
    "cos": "cos2d",
    "erf": "erf2d",
    "gelu": "gelu2d",
    "cumsum": "cumsum2d",
    "log": "log2d",
    "log_sigmoid": "log_sigmoid2d",
    "log_softmax": "log_softmax2d",
    "min": "min2d",
    "min_dim": "min_dim2d",
    "nonzero": "nonzero2d",
    "normed_cumsum": "normed_cumsum2d",
    "pad": "pad2d",
    "per_token_group_quant_fp8": "per_token_group_quant_fp8_2d",
    "pow_scalar": "pow_scalar2d",
    "pow_tensor_scalar": "pow_tensor_scalar2d",
    "pow_tensor_tensor": "pow_tensor_tensor2d",
    "prod": "prod2d",
    "prod_dim": "prod_dim2d",
    "remainder": "remainder2d",
    "repeat": "repeat2d",
    "repeat_interleave_self_int": "repeat_interleave_self_int1d",
    "repeat_interleave_self_tensor": "repeat_interleave_self_tensor1d",
    "repeat_interleave_tensor": "repeat_interleave_tensor1d",
    "tile": "tile2d",
    "stack": "stack2d",
    "sort": "sort2d",
    "sort_stable": "sort_stable2d",
    "topk": "topk2d",
    "std": "std2d",
    "var_mean": "var_mean2d",
    "vector_norm": "vector_norm2d",
    "sin": "sin2d",
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
    "celu": "celu2d",
    "elu": "elu2d",
    "exp": "exp2d",
    "isclose": "isclose2d",
    "allclose": "allclose2d",
    "isfinite": "isfinite2d",
    "isinf": "isinf2d",
    "isnan": "isnan2d",
    "masked_fill": "masked_fill2d",
    "threshold": "threshold2d",
    "where_self": "where2d",
    "flip": "flip2d",
    "embedding": "embedding2d",
    "isin": "isin1d",
    "kron": "kron2d",
    "linspace": "linspace1d",
    "logspace": "logspace1d",
    "masked_select": "masked_select2d",
    "masked_scatter": "masked_scatter2d",
    "max_pool2d_with_indices": "max_pool2d_with_indices_nchw",
    "conv1d": "conv1d_ncl",
    "conv3d": "conv3d_ncdhw",
    "conv_depthwise2d": "conv_depthwise2d_nchw",
    "scatter": "scatter2d",
    "select_scatter": "select_scatter2d",
    "slice_scatter": "slice_scatter2d",
    "quantile": "quantile2d",
    "polar": "polar2d",
    "unique2": "unique2d",
    "weight_norm_interface": "weight_norm2d",
    "scaled_dot_product_attention": "scaled_dot_product_attention_bhsd",
    "flash_attn_varlen_func": "flash_attn_varlen_func_bhsd",
    "upsample_nearest1d": "upsample_nearest1d_ncl",
    "upsample_nearest2d": "upsample_nearest2d_nchw",
    "constant_pad_nd": "constant_pad_nd2d",
    "hstack": "hstack2d",
    "vstack": "vstack2d",
    "conv2d": "conv2d_nchw",
    "mse_loss": "mse_loss2d",
    "nan_to_num": "nan_to_num2d",
    "nll_loss_forward": "nll_loss_forward",
    "nll_loss2d_forward": "nll_loss2d_forward",
    "one_hot": "one_hot2d",
    "glu": "glu2d",
    "cummax": "cummax1d",
    "cummin": "cummin1d",
    "index_add": "index_add2d",
    "index_put": "index_put2d",
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
    "eye": "eye2d",
    "eye_m": "eye_m2d",
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
    "ScaleDotProductAttention": "scaled_dot_product_attention",
    "scaled_dot_product_attention_forward": "scaled_dot_product_attention",
    "flash_attention_forward": "scaled_dot_product_attention",
    "bitwise_and_scalar": "bitwise_and",
    "bitwise_and_scalar_tensor": "bitwise_and",
    "bitwise_and_tensor": "bitwise_and",
    "bitwise_or_scalar": "bitwise_or",
    "bitwise_or_scalar_tensor": "bitwise_or",
    "bitwise_or_tensor": "bitwise_or",
}

_E2E_SPEC_TO_SEMANTIC: dict[str, str] = {v: k for k, v in _SEMANTIC_TO_E2E_SPEC.items()}


def _is_valid_flaggems_src_dir(path: Path) -> bool:
    p = Path(path)
    return bool((p / "flag_gems" / "__init__.py").is_file())


def _iter_flaggems_src_candidates(flaggems_src: str | Path | None) -> list[Path]:
    raw_candidates: list[str] = []
    if flaggems_src is not None:
        raw_candidates.append(str(flaggems_src))
    env = os.getenv("FLAGGEMS_SRC")
    if isinstance(env, str) and env.strip():
        raw_candidates.append(env.strip())
    try:
        if DEFAULT_REGISTRY_PATH.is_file():
            payload = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
            reg_src = str(payload.get("flaggems_source") or "").strip()
            if reg_src:
                raw_candidates.append(reg_src)
    except Exception:
        pass

    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        s = str(raw).strip()
        if not s or s.startswith("python:"):
            continue
        p = Path(s)
        # Allow callers to provide either ".../src" or ".../src/flag_gems".
        if p.name == "flag_gems" and (p / "__init__.py").is_file():
            p = p.parent
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _drop_broken_flaggems_editable_finders() -> None:
    # Editable installs based on `_flag_gems_editable` can keep stale absolute
    # source paths after local cleanups. If all mapped files are gone, drop the
    # finder so normal source-path import resolution can proceed.
    for finder in list(sys.meta_path):
        if type(finder).__module__ != "_flag_gems_editable":
            continue
        known = getattr(finder, "known_source_files", None)
        if not isinstance(known, dict) or not known:
            continue
        paths = [Path(str(v)) for v in known.values() if isinstance(v, str)]
        if not paths:
            continue
        if not all(not p.exists() for p in paths):
            continue
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            continue


def _force_import_flaggems_from_src(src_dir: Path) -> None:
    pkg_dir = Path(src_dir) / "flag_gems"
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"flag_gems package not found under source dir: {src_dir}")
    for key in list(sys.modules.keys()):
        if key == "flag_gems" or key.startswith("flag_gems."):
            sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(
        "flag_gems",
        str(init_py),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to build import spec for flag_gems from: {init_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flag_gems"] = mod
    spec.loader.exec_module(mod)


def ensure_flaggems_importable(flaggems_src: str | Path | None = None) -> None:
    candidates = _iter_flaggems_src_candidates(flaggems_src)
    valid_candidates: list[Path] = []
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
        if _is_valid_flaggems_src_dir(p):
            valid_candidates.append(p.resolve())

    if not valid_candidates:
        return

    _drop_broken_flaggems_editable_finders()

    try:
        mod = importlib.import_module("flag_gems")
        origin = Path(str(getattr(mod, "__file__", "")))
        if origin.is_file():
            return
    except Exception:
        pass

    _force_import_flaggems_from_src(valid_candidates[0])


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
            "count_nonzero",
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
    # No frozen registry present. Try importing the installed `flag_gems` package
    # (or an explicitly configured source via FLAGGEMS_SRC / explicit src path).
    try:
        all_ops = load_flaggems_all_ops(flaggems_src=None)
    except Exception as e:
        raise RuntimeError(
            "flaggems registry missing and `flag_gems` is not importable; "
            "install the `flag_gems` package or set FLAGGEMS_SRC / pass an explicit source path"
        ) from e
    # Commit/hash is best-effort here; the canonical metadata lives in the frozen registry.
    env = os.getenv("FLAGGEMS_SRC")
    commit = infer_flaggems_commit_from_src(env) if env else None
    return build_registry(
        all_ops=all_ops,
        flaggems_commit=str(commit or "unknown"),
        flaggems_source=(str(env) if env else "python:flag_gems"),
    )


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
