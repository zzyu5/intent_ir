#!/usr/bin/env python3
"""Build a CUDA real-MLIR support report from deterministic intent seeds.

This script is a *static* readiness snapshot:
- reads the current kernel denominator from workflow state (coverage_batches.json)
- loads expanded intent seeds (prefer artifacts/flaggems_seed_cache; fallback to provider canonical)
- classifies which kernels are supported by the current CUDA real-MLIR lowering pass

The report is committed under workflow state to keep wave planning auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.utils.repo_state import repo_state  # noqa: E402
from pipeline.triton.providers.registry import get_provider_plugin  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _rel(path: Path) -> str:
    try:
        rp = path.resolve()
    except Exception:
        rp = path
    try:
        return str(rp.relative_to(ROOT))
    except Exception:
        return str(path)


def _collect_kernels(coverage_batches: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for b in list(coverage_batches.get("batches") or []):
        if not isinstance(b, dict):
            continue
        for k in list(b.get("kernels") or []):
            name = str(k).strip()
            if name:
                out.append(name)
    # preserve stable order but dedupe
    seen: set[str] = set()
    uniq: list[str] = []
    for k in out:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq


def _seed_payload_for_kernel(*, kernel: str, seed_dir: Path, provider: Any) -> dict[str, Any] | None:
    p = seed_dir / f"{kernel}.intent_seed.json"
    if p.is_file():
        try:
            return _load_json(p)
        except Exception:
            return None
    try:
        payload = provider.seed_payload_for_spec(spec_name=str(kernel))
    except Exception:
        payload = None
    return payload


def _expanded_intent_from_seed(seed: dict[str, Any]) -> dict[str, Any] | None:
    intent = seed.get("intent_expanded")
    if isinstance(intent, dict):
        return intent
    intent = seed.get("intent")
    if isinstance(intent, dict):
        return intent
    return None


SUPPORTED_OPS = {
    # Existing elementwise subset.
    "abs",
    "add",
    "cast",
    "ceil",
    "const",
    "cos",
    "div",
    "eq",
    "erf",
    "exp",
    "exp2",
    "floor",
    "ge",
    "gt",
    "identity",
    "le",
    "log",
    "lt",
    "max",
    "min",
    "mul",
    "ne",
    "neg",
    "relu",
    "rsqrt",
    "sin",
    "sqrt",
    "sub",
    "tan",
    "where",
    # New (non-reduction) ops supported by cuda real-mlir lowering.
    "broadcast_in_dim",
    "reshape",
    "and",
    "not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_not",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "remainder",
    "pow",
    "acos",
    "atan",
    "concat",
    "pad",
    "gather",
    "iota",
    "stack",
    "polar",
    "upsample_nearest1d",
    "upsample_nearest2d",
    "glu",
    "softmax",
    "matmul",
    "mse_loss",
    "std",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_any",
    "argmax",
    "argmin",
}


@dataclass(frozen=True)
class SupportResult:
    ok: bool
    reasons: list[str]
    unsupported_ops: list[str]


def _tensor_shape(intent: dict[str, Any], name: str) -> list[Any]:
    t = (intent.get("tensors") or {}).get(name) if isinstance(intent.get("tensors"), dict) else None
    if not isinstance(t, dict):
        return []
    s = t.get("shape")
    return list(s) if isinstance(s, list) else []


def _tensor_dtype(intent: dict[str, Any], name: str) -> str:
    t = (intent.get("tensors") or {}).get(name) if isinstance(intent.get("tensors"), dict) else None
    if not isinstance(t, dict):
        return ""
    return str(t.get("dtype") or "").strip()


def _check_broadcast_in_dim(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["broadcast_in_dim_invalid_inputs"]
    in_shape = _tensor_shape(intent, ins[0])
    if len(in_shape) == 0:
        return []
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    out_shape = attrs.get("out_shape")
    bcast_dims = attrs.get("broadcast_dims")
    if not (isinstance(out_shape, list) and isinstance(bcast_dims, list)):
        return ["broadcast_in_dim_missing_attrs"]
    if len(out_shape) < len(in_shape):
        return ["broadcast_in_dim_out_rank_lt_in_rank"]
    try:
        dims = [int(x) for x in list(bcast_dims)]
    except Exception:
        return ["broadcast_in_dim_invalid_dims"]
    if len(dims) != len(in_shape) or len(set(dims)) != len(dims):
        return ["broadcast_in_dim_invalid_dims"]
    if any((d < 0) or (d >= len(out_shape)) for d in dims):
        return ["broadcast_in_dim_invalid_dims"]
    if dims != sorted(dims):
        return ["broadcast_in_dim_dims_not_sorted"]
    return []


def _check_concat(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 2:
        return ["concat_requires_two_inputs"]
    axis = op.get("attrs", {}).get("axis") if isinstance(op.get("attrs"), dict) else None
    if axis not in (0, 1):
        return ["concat_axis_not_0_or_1"]
    # Require 2D inputs/output.
    out = str(op.get("output") or "").strip()
    if len(_tensor_shape(intent, out)) != 2:
        return ["concat_non_2d_output"]
    if any(len(_tensor_shape(intent, x)) != 2 for x in ins):
        return ["concat_non_2d_inputs"]
    return []


def _check_pad(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["pad_invalid_inputs"]
    out = str(op.get("output") or "").strip()
    if len(_tensor_shape(intent, out)) != 2 or len(_tensor_shape(intent, ins[0])) != 2:
        return ["pad_requires_2d"]
    attrs = op.get("attrs")
    if not isinstance(attrs, dict):
        return ["pad_missing_attrs"]
    mode = str(attrs.get("mode") or "").strip().lower()
    if mode and mode != "constant":
        return ["pad_mode_not_constant"]
    pad_width = attrs.get("pad_width")
    if not isinstance(pad_width, dict):
        return ["pad_missing_pad_width"]
    pairs = pad_width.get("pairs")
    if not (isinstance(pairs, list) and len(pairs) == 2):
        return ["pad_invalid_pad_width_pairs"]
    return []


def _check_gather(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    intent_name = str(intent.get("name") or "").strip()
    if intent_name == "upsample_bicubic2d_aa":
        # Special-case: current CUDA real-MLIR lowering matches this kernel by name
        # and emits a dedicated kernel; the expanded seed contains 4D gathers with
        # explicit (n,c,iy,ix) indices.
        if len(ins) == 5:
            in_shape = _tensor_shape(intent, ins[0])
            if in_shape and len(in_shape) != 4:
                return ["gather_bicubic_requires_4d_input"]
            return []
    if len(ins) != 3:
        return ["gather_invalid_inputs"]
    # Current lowering supports 2D inp + (row_idx,col_idx) -> out (rank 1/2).
    if len(_tensor_shape(intent, ins[0])) != 2:
        return ["gather_requires_2d_input"]
    return []


def _check_stack(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 2:
        return ["stack_requires_two_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    try:
        axis_i = int(axis) if axis is not None else None
    except Exception:
        axis_i = None
    if axis_i != 0:
        return ["stack_axis_not_0"]
    out = str(op.get("output") or "").strip()
    out_shape = _tensor_shape(intent, out)
    if out_shape and len(out_shape) != 3:
        return ["stack_output_rank_not_3"]
    if out_shape:
        try:
            d0 = int(out_shape[0])
        except Exception:
            d0 = None
        if d0 is not None and d0 != 2:
            return ["stack_output_first_dim_not_2"]
    if any((_tensor_dtype(intent, x) not in {"", "f32"}) for x in ins):
        return ["stack_input_dtype_not_f32"]
    if out and _tensor_dtype(intent, out) not in {"", "f32"}:
        return ["stack_output_dtype_not_f32"]
    return []


def _check_polar(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 2:
        return ["polar_requires_two_inputs"]
    out = str(op.get("output") or "").strip()
    out_shape = _tensor_shape(intent, out)
    if out_shape and len(out_shape) != 3:
        return ["polar_output_rank_not_3"]
    if out_shape:
        try:
            d_last = int(out_shape[-1])
        except Exception:
            d_last = None
        if d_last is not None and d_last != 2:
            return ["polar_output_last_dim_not_2"]
    if any((_tensor_dtype(intent, x) not in {"", "f32"}) for x in ins):
        return ["polar_input_dtype_not_f32"]
    if out and _tensor_dtype(intent, out) not in {"", "f32"}:
        return ["polar_output_dtype_not_f32"]
    return []


def _check_upsample_nearest1d(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["upsample_nearest1d_invalid_inputs"]
    out = str(op.get("output") or "").strip()
    if len(_tensor_shape(intent, ins[0])) != 3:
        return ["upsample_nearest1d_requires_rank3_input"]
    if len(_tensor_shape(intent, out)) != 3:
        return ["upsample_nearest1d_requires_rank3_output"]
    if _tensor_dtype(intent, ins[0]) not in {"", "f32"}:
        return ["upsample_nearest1d_input_dtype_not_f32"]
    if out and _tensor_dtype(intent, out) not in {"", "f32"}:
        return ["upsample_nearest1d_output_dtype_not_f32"]
    return []


def _check_upsample_nearest2d(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["upsample_nearest2d_invalid_inputs"]
    out = str(op.get("output") or "").strip()
    if len(_tensor_shape(intent, ins[0])) != 4:
        return ["upsample_nearest2d_requires_rank4_input"]
    if len(_tensor_shape(intent, out)) != 4:
        return ["upsample_nearest2d_requires_rank4_output"]
    if _tensor_dtype(intent, ins[0]) not in {"", "f32"}:
        return ["upsample_nearest2d_input_dtype_not_f32"]
    if out and _tensor_dtype(intent, out) not in {"", "f32"}:
        return ["upsample_nearest2d_output_dtype_not_f32"]
    return []


def _check_glu(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["glu_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    try:
        axis_i = int(axis) if axis is not None else None
    except Exception:
        axis_i = None
    if axis_i != 1:
        return ["glu_axis_not_1"]
    out = str(op.get("output") or "").strip()
    if len(_tensor_shape(intent, ins[0])) != 2:
        return ["glu_requires_rank2_input"]
    if len(_tensor_shape(intent, out)) != 2:
        return ["glu_requires_rank2_output"]
    if _tensor_dtype(intent, ins[0]) not in {"", "f32"}:
        return ["glu_input_dtype_not_f32"]
    if out and _tensor_dtype(intent, out) not in {"", "f32"}:
        return ["glu_output_dtype_not_f32"]
    return []


def _check_softmax(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["softmax_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    try:
        axis_i = int(axis) if axis is not None else None
    except Exception:
        axis_i = None
    if axis_i != 1:
        return ["softmax_axis_not_1"]
    return []


def _check_matmul(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 2:
        return ["matmul_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    ta = attrs.get("transpose_a", False)
    tb = attrs.get("transpose_b", False)
    if bool(ta) or bool(tb):
        return ["matmul_transpose_not_supported"]

    a_shape = _tensor_shape(intent, ins[0])
    b_shape = _tensor_shape(intent, ins[1])
    out = str(op.get("output") or "").strip()
    out_shape = _tensor_shape(intent, out) if out else []

    if a_shape and len(a_shape) != 2:
        return ["matmul_requires_rank2_a"]
    if b_shape and len(b_shape) not in (1, 2):
        return ["matmul_requires_rank1_or_2_b"]

    if a_shape and b_shape:
        if len(b_shape) == 2:
            if a_shape[1] != b_shape[0]:
                return ["matmul_k_mismatch"]
            if out_shape and (len(out_shape) != 2 or out_shape != [a_shape[0], b_shape[1]]):
                return ["matmul_output_shape_mismatch"]
        else:
            if a_shape[1] != b_shape[0]:
                return ["matmul_k_mismatch"]
            if out_shape and (len(out_shape) != 1 or out_shape != [a_shape[0]]):
                return ["matmul_output_shape_mismatch"]

    # Dtypes: current lowering is f32-focused; keep wave plan narrow.
    dt_a = _tensor_dtype(intent, ins[0])
    dt_b = _tensor_dtype(intent, ins[1])
    if (dt_a and dt_a != "f32") or (dt_b and dt_b != "f32"):
        return ["matmul_requires_f32"]
    return []


def _check_mse_loss(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 2:
        return ["mse_loss_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    reduction = attrs.get("reduction", None)
    try:
        red_i = int(reduction) if reduction is not None else None
    except Exception:
        red_i = None
    if red_i != 1:
        return ["mse_loss_reduction_not_mean"]

    s0 = _tensor_shape(intent, ins[0])
    s1 = _tensor_shape(intent, ins[1])
    if s0 and s1 and s0 != s1:
        return ["mse_loss_input_shapes_mismatch"]

    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 0:
            return ["mse_loss_requires_scalar_output"]

    dt0 = _tensor_dtype(intent, ins[0])
    dt1 = _tensor_dtype(intent, ins[1])
    if (dt0 and dt0 != "f32") or (dt1 and dt1 != "f32"):
        return ["mse_loss_requires_f32"]
    return []


def _check_std(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["std_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dims = attrs.get("axes", attrs.get("dims", attrs.get("axis")))
    dims_list = list(dims) if isinstance(dims, list) else []
    if dims_list != [1]:
        return ["std_dims_not_axis1"]
    if bool(attrs.get("keepdims", False)):
        return ["std_keepdims_not_supported"]
    correction = attrs.get("correction", attrs.get("ddof", 0))
    try:
        ddof = int(correction)
    except Exception:
        ddof = 0
    if ddof != 1:
        return ["std_ddof_not_1"]
    x_shape = _tensor_shape(intent, ins[0])
    if x_shape and len(x_shape) != 2:
        return ["std_requires_rank2_input"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 1:
            return ["std_output_rank_not_1"]
    dt = _tensor_dtype(intent, ins[0])
    if dt and dt != "f32":
        return ["std_requires_f32"]
    return []


def _check_reduce_sum(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["reduce_sum_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    intent_name = str(intent.get("name") or "").strip()
    if intent_name == "group_norm_kernel":
        # Special-case: current CUDA real-MLIR lowering matches this kernel by name
        # and emits a dedicated kernel, so per-op reduce_sum shape constraints here
        # are too strict for wave planning.
        return []
    if dims_list == [0]:
        if intent_name not in {"dot1d", "vdot1d"}:
            return ["reduce_sum_dims_0_only_supported_for_dot_or_vdot"]
        out = str(op.get("output") or "").strip()
        if out:
            out_shape = _tensor_shape(intent, out)
            if out_shape and len(out_shape) != 0:
                return ["reduce_sum_dims_0_requires_scalar_output"]
        return []
    if dims_list not in ([1], [0, 1]):
        return ["reduce_sum_dims_not_axis1_or_01"]
    out = str(op.get("output") or "").strip()
    if dims_list == [0, 1]:
        if intent_name not in {"trace2d", "count_nonzero2d"}:
            return ["reduce_sum_dims_01_only_supported_for_trace_or_count_nonzero"]
        if out:
            out_shape = _tensor_shape(intent, out)
            if out_shape and len(out_shape) != 0:
                return ["reduce_sum_dims_01_requires_scalar_output"]
        return []
    if out:
        out_shape = _tensor_shape(intent, out)
        if not out_shape:
            return []
        if len(out_shape) == 2:
            try:
                d1 = int(out_shape[1])
            except Exception:
                return ["reduce_sum_keepdims_requires_literal_last_dim_1"]
            if d1 != 1:
                return ["reduce_sum_keepdims_requires_last_dim_1"]
        elif len(out_shape) != 1:
            return ["reduce_sum_output_rank_not_1_or_2"]
    return []


def _check_reduce_max(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["reduce_max_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    if dims_list != [1]:
        return ["reduce_max_dims_not_axis1"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if not out_shape:
            return []
        if len(out_shape) == 2:
            try:
                d1 = int(out_shape[1])
            except Exception:
                return ["reduce_max_keepdims_requires_literal_last_dim_1"]
            if d1 != 1:
                return ["reduce_max_keepdims_requires_last_dim_1"]
        elif len(out_shape) != 1:
            return ["reduce_max_output_rank_not_1_or_2"]
    return []


def _check_reduce_min(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["reduce_min_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    out = str(op.get("output") or "").strip()
    if dims_list == [1]:
        if out:
            out_shape = _tensor_shape(intent, out)
            if not out_shape:
                return []
            if len(out_shape) == 2:
                try:
                    d1 = int(out_shape[1])
                except Exception:
                    return ["reduce_min_keepdims_requires_literal_last_dim_1"]
                if d1 != 1:
                    return ["reduce_min_keepdims_requires_last_dim_1"]
            elif len(out_shape) != 1:
                return ["reduce_min_output_rank_not_1_or_2"]
        return []
    if dims_list == [0, 1]:
        if out:
            out_shape = _tensor_shape(intent, out)
            if out_shape and len(out_shape) != 0:
                return ["reduce_min_dims_01_requires_scalar_output"]
        return []
    return ["reduce_min_dims_not_axis1_or_01"]


def _check_reduce_prod(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["reduce_prod_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    intent_name = str(intent.get("name") or "").strip()
    if dims_list == [0, 1]:
        if intent_name != "prod2d":
            return ["reduce_prod_dims_01_only_supported_for_prod2d"]
        out = str(op.get("output") or "").strip()
        if out:
            out_shape = _tensor_shape(intent, out)
            if out_shape and len(out_shape) != 0:
                return ["reduce_prod_dims_01_requires_scalar_output"]
        return []
    if dims_list != [1]:
        return ["reduce_prod_dims_not_axis1_or_01"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if not out_shape:
            return []
        if len(out_shape) == 2:
            try:
                d1 = int(out_shape[1])
            except Exception:
                return ["reduce_prod_keepdims_requires_literal_last_dim_1"]
            if d1 != 1:
                return ["reduce_prod_keepdims_requires_last_dim_1"]
        elif len(out_shape) != 1:
            return ["reduce_prod_output_rank_not_1_or_2"]
    return []


def _check_argmax(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["argmax_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    if axis != 1:
        return ["argmax_axis_not_1"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 1:
            return ["argmax_requires_rank1_output"]
        out_dtype = _tensor_dtype(intent, out)
        if out_dtype and out_dtype != "i32":
            return [f"argmax_output_dtype_not_i32:{out_dtype}"]
    return []


def _check_argmin(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["argmin_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    if axis != 1:
        return ["argmin_axis_not_1"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 1:
            return ["argmin_requires_rank1_output"]
        out_dtype = _tensor_dtype(intent, out)
        if out_dtype and out_dtype != "i32":
            return [f"argmin_output_dtype_not_i32:{out_dtype}"]
    return []


def _check_reduce_any(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["reduce_any_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    intent_name = str(intent.get("name") or "").strip()
    if dims_list == [0, 1]:
        if intent_name != "allclose2d":
            return ["reduce_any_dims_01_only_supported_for_allclose2d"]
        out = str(op.get("output") or "").strip()
        if out:
            out_shape = _tensor_shape(intent, out)
            if out_shape and len(out_shape) != 0:
                return ["reduce_any_dims_01_requires_scalar_output"]
        return []
    if dims_list != [1]:
        return ["reduce_any_dims_not_axis1_or_01"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if not out_shape:
            # Many expanded seeds omit intermediate tensor specs; runtime lowering
            # repairs missing tensor metadata before codegen. Treat unknown shape
            # as "not checked" here to avoid false negatives in the wave plan.
            return []
        if len(out_shape) == 2:
            # keepdims case: accept [M,1]
            try:
                d1 = int(out_shape[1])
            except Exception:
                return ["reduce_any_keepdims_requires_literal_last_dim_1"]
            if d1 != 1:
                return ["reduce_any_keepdims_requires_last_dim_1"]
        elif len(out_shape) != 1:
            return ["reduce_any_output_rank_not_1_or_2"]
    return []


def _supported_by_cuda_real_mlir(intent: dict[str, Any]) -> SupportResult:
    outputs = [str(x) for x in list(intent.get("outputs") or []) if str(x).strip()]
    intent_name = str(intent.get("name") or "").strip()
    multi_output_ok = {
        "group_norm_kernel",
        "layer_norm_persistent",
        "layer_norm_residual2d",
        "ai_bench_layernorm",
        "rms_norm2d",
        "rms_norm_residual2d",
        "min_dim2d",
    }
    if not outputs:
        return SupportResult(ok=False, reasons=["missing_outputs"], unsupported_ops=[])
    if len(outputs) != 1 and intent_name not in multi_output_ok:
        return SupportResult(ok=False, reasons=["multi_output"], unsupported_ops=[])
    out_rank = max((len(_tensor_shape(intent, o)) for o in outputs), default=0)
    allowed_high_rank = {
        "stack2d",
        "polar2d",
        "diag_embed2d",
        "upsample_nearest1d_ncl",
        "upsample_nearest2d_nchw",
        "group_norm_kernel",
        "upsample_bicubic2d_aa",
    }
    if out_rank > 2 and intent_name not in allowed_high_rank:
        return SupportResult(ok=False, reasons=[f"output_rank_gt2:{out_rank}"], unsupported_ops=[])

    ops = [o for o in list(intent.get("ops") or []) if isinstance(o, dict)]
    op_names = [str(o.get("op") or "").strip() for o in ops if str(o.get("op") or "").strip()]
    uniq = sorted(set(op_names))
    unsupported = sorted([x for x in uniq if x not in SUPPORTED_OPS])
    if unsupported:
        return SupportResult(ok=False, reasons=["unsupported_ops"], unsupported_ops=unsupported)

    reasons: list[str] = []
    for o in ops:
        name = str(o.get("op") or "").strip()
        if name == "broadcast_in_dim":
            reasons.extend(_check_broadcast_in_dim(intent, o))
        elif name == "concat":
            reasons.extend(_check_concat(intent, o))
        elif name == "pad":
            reasons.extend(_check_pad(intent, o))
        elif name == "gather":
            reasons.extend(_check_gather(intent, o))
        elif name == "stack":
            reasons.extend(_check_stack(intent, o))
        elif name == "polar":
            reasons.extend(_check_polar(intent, o))
        elif name == "upsample_nearest1d":
            reasons.extend(_check_upsample_nearest1d(intent, o))
        elif name == "upsample_nearest2d":
            reasons.extend(_check_upsample_nearest2d(intent, o))
        elif name == "glu":
            reasons.extend(_check_glu(intent, o))
        elif name == "softmax":
            reasons.extend(_check_softmax(intent, o))
        elif name == "matmul":
            reasons.extend(_check_matmul(intent, o))
        elif name == "mse_loss":
            reasons.extend(_check_mse_loss(intent, o))
        elif name == "std":
            reasons.extend(_check_std(intent, o))
        elif name == "reduce_sum":
            reasons.extend(_check_reduce_sum(intent, o))
        elif name == "reduce_max":
            reasons.extend(_check_reduce_max(intent, o))
        elif name == "reduce_min":
            reasons.extend(_check_reduce_min(intent, o))
        elif name == "reduce_prod":
            reasons.extend(_check_reduce_prod(intent, o))
        elif name == "reduce_any":
            reasons.extend(_check_reduce_any(intent, o))
        elif name == "argmax":
            reasons.extend(_check_argmax(intent, o))
        elif name == "argmin":
            reasons.extend(_check_argmin(intent, o))
    if reasons:
        return SupportResult(ok=False, reasons=sorted(set(reasons)), unsupported_ops=[])
    return SupportResult(ok=True, reasons=[], unsupported_ops=[])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coverage-batches",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "coverage_batches.json"),
    )
    ap.add_argument(
        "--seed-dir",
        type=Path,
        default=(ROOT / "artifacts" / "flaggems_seed_cache"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "cuda_real_mlir_support_report.json"),
    )
    ap.add_argument(
        "--emit-wave",
        type=Path,
        default=None,
        help="Optional path to emit a cuda_real_mlir_<wave>_kernels.json file containing supported kernels.",
    )
    ap.add_argument("--wave", type=str, default="wave3")
    args = ap.parse_args()

    cov_path = Path(args.coverage_batches)
    if not cov_path.is_file():
        raise SystemExit(f"missing coverage_batches.json: {cov_path}")
    cov = _load_json(cov_path)
    kernels = _collect_kernels(cov)

    provider = get_provider_plugin("flaggems")
    seed_dir = Path(args.seed_dir)

    supported: list[str] = []
    unsupported_rows: list[dict[str, Any]] = []
    op_freq: dict[str, int] = {}
    for k in kernels:
        seed = _seed_payload_for_kernel(kernel=str(k), seed_dir=seed_dir, provider=provider)
        if seed is None:
            unsupported_rows.append({"kernel": str(k), "reason": "missing_seed"})
            continue
        intent = _expanded_intent_from_seed(seed)
        if intent is None:
            unsupported_rows.append({"kernel": str(k), "reason": "invalid_seed_format"})
            continue
        ops = [o for o in list(intent.get("ops") or []) if isinstance(o, dict)]
        for o in ops:
            name = str(o.get("op") or "").strip()
            if name:
                op_freq[name] = int(op_freq.get(name, 0) + 1)

        res = _supported_by_cuda_real_mlir(intent)
        if res.ok:
            supported.append(str(k))
        else:
            row: dict[str, Any] = {"kernel": str(k), "reasons": list(res.reasons)}
            if res.unsupported_ops:
                row["unsupported_ops"] = list(res.unsupported_ops)
            unsupported_rows.append(row)

    out: dict[str, Any] = {
        "schema_version": "intentir_cuda_real_mlir_support_report_v1",
        "generated_at_utc": _utc_now_iso(),
        "repo": repo_state(root=ROOT),
        "sources": {
            "coverage_batches": str(cov_path.relative_to(ROOT)),
            "seed_dir": str(seed_dir.relative_to(ROOT)) if seed_dir.is_absolute() and seed_dir.exists() else str(seed_dir),
        },
        "supported_ops": sorted(SUPPORTED_OPS),
        "kernels_total": int(len(kernels)),
        "kernels_supported": int(len(supported)),
        "kernels_unsupported": int(len(unsupported_rows)),
        "supported_kernels": list(supported),
        "unsupported_kernels": list(unsupported_rows),
        "op_freq": dict(sorted(op_freq.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    out_path = _dump_json(Path(args.out), out)
    print(_rel(out_path))

    if args.emit_wave is not None:
        wave_name = str(args.wave).strip().lower()
        wave_payload = {
            "schema_version": "intentir_cuda_real_mlir_wave_v1",
            "wave": str(wave_name),
            "kernels": list(supported),
        }
        wave_path = _dump_json(Path(args.emit_wave), wave_payload)
        print(_rel(wave_path))


if __name__ == "__main__":
    main()
