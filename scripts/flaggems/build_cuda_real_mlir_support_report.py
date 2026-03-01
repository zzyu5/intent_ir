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
    "nll_loss_forward",
    "nll_loss2d_forward",
    "std",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_any",
    "argmax",
    "argmin",
    "cumsum",
    "cummax",
    "cummin",
    "kron",
    "sort",
    "quantile",
    "avg_pool2d",
    "max_pool2d_with_indices",
    "index_add",
    "index_put",
    "scatter",
    "select_scatter",
    "slice_scatter",
    "masked_select",
    "masked_scatter",
    "scaled_dot_product_attention",
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
    intent_name = str(intent.get("name") or "").strip()
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

    # Batched matmul: [B,M,K] x [B,K,N] -> [B,M,N].
    if intent_name in {"bmm3d", "baddbmm3d"}:
        if a_shape and len(a_shape) != 3:
            return ["matmul_requires_rank3_a_for_bmm"]
        if b_shape and len(b_shape) != 3:
            return ["matmul_requires_rank3_b_for_bmm"]
        if out_shape and len(out_shape) != 3:
            return ["matmul_requires_rank3_output_for_bmm"]
        if a_shape and b_shape and out_shape:
            if a_shape[0] != b_shape[0] or a_shape[0] != out_shape[0]:
                return ["matmul_batch_mismatch"]
            if a_shape[2] != b_shape[1]:
                return ["matmul_k_mismatch"]
            if out_shape[1] != a_shape[1] or out_shape[2] != b_shape[2]:
                return ["matmul_output_shape_mismatch"]
        dt_a = _tensor_dtype(intent, ins[0])
        dt_b = _tensor_dtype(intent, ins[1])
        if (dt_a and dt_a != "f32") or (dt_b and dt_b != "f32"):
            return ["matmul_requires_f32"]
        return []

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
    if intent_name == "batch_norm2d" and dims_list == [0, 2]:
        # Channel-wise reduction: input[N,C,HW] -> out[C]
        out = str(op.get("output") or "").strip()
        if out:
            out_shape = _tensor_shape(intent, out)
            if out_shape and len(out_shape) != 1:
                return ["reduce_sum_batch_norm2d_dims_02_requires_rank1_output"]
        x_shape = _tensor_shape(intent, ins[0])
        if x_shape and len(x_shape) != 3:
            return ["reduce_sum_batch_norm2d_dims_02_requires_rank3_input"]
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


def _check_scaled_dot_product_attention(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 3:
        return ["scaled_dot_product_attention_invalid_inputs"]
    out = str(op.get("output") or "").strip()
    if not out:
        return ["scaled_dot_product_attention_missing_output"]

    q_shape = _tensor_shape(intent, ins[0])
    k_shape = _tensor_shape(intent, ins[1])
    v_shape = _tensor_shape(intent, ins[2])
    o_shape = _tensor_shape(intent, out)
    if any(len(s) != 4 for s in (q_shape, k_shape, v_shape, o_shape)):
        return ["scaled_dot_product_attention_requires_rank4"]
    if q_shape != o_shape:
        return ["scaled_dot_product_attention_out_shape_mismatch"]
    if k_shape != v_shape:
        return ["scaled_dot_product_attention_kv_shape_mismatch"]
    # Expect (B,H,Q,D) x (B,H,K,D) -> (B,H,Q,D)
    if q_shape[0] != k_shape[0] or q_shape[1] != k_shape[1] or q_shape[3] != k_shape[3]:
        return ["scaled_dot_product_attention_bhd_mismatch"]

    dt_q = _tensor_dtype(intent, ins[0])
    dt_k = _tensor_dtype(intent, ins[1])
    dt_v = _tensor_dtype(intent, ins[2])
    dt_o = _tensor_dtype(intent, out)
    if any(dt and dt != "f32" for dt in (dt_q, dt_k, dt_v, dt_o)):
        return ["scaled_dot_product_attention_requires_f32"]

    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    is_causal = attrs.get("is_causal", False)
    if not isinstance(is_causal, (bool, int)):
        return ["scaled_dot_product_attention_is_causal_not_bool"]
    return []


def _check_cumsum(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["cumsum_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    if axis != 1:
        return ["cumsum_axis_not_1"]
    x_shape = _tensor_shape(intent, ins[0])
    if x_shape and len(x_shape) != 2:
        return ["cumsum_requires_rank2_input"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 2:
            return ["cumsum_requires_rank2_output"]
    dt = _tensor_dtype(intent, ins[0])
    if dt and dt != "f32":
        return ["cumsum_requires_f32"]
    return []


def _check_cummax(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["cummax_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    if axis != 0:
        return ["cummax_axis_not_0"]
    x_shape = _tensor_shape(intent, ins[0])
    if x_shape and len(x_shape) != 1:
        return ["cummax_requires_rank1_input"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 1:
            return ["cummax_requires_rank1_output"]
    dt = _tensor_dtype(intent, ins[0])
    if dt and dt != "f32":
        return ["cummax_requires_f32"]
    return []


def _check_cummin(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["cummin_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    if axis != 0:
        return ["cummin_axis_not_0"]
    x_shape = _tensor_shape(intent, ins[0])
    if x_shape and len(x_shape) != 1:
        return ["cummin_requires_rank1_input"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 1:
            return ["cummin_requires_rank1_output"]
    dt = _tensor_dtype(intent, ins[0])
    if dt and dt != "f32":
        return ["cummin_requires_f32"]
    return []


def _check_kron(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 2:
        return ["kron_invalid_inputs"]
    a_shape = _tensor_shape(intent, ins[0])
    b_shape = _tensor_shape(intent, ins[1])
    if a_shape and len(a_shape) != 2:
        return ["kron_requires_rank2_input_a"]
    if b_shape and len(b_shape) != 2:
        return ["kron_requires_rank2_input_b"]
    out = str(op.get("output") or "").strip()
    if out:
        out_shape = _tensor_shape(intent, out)
        if out_shape and len(out_shape) != 2:
            return ["kron_requires_rank2_output"]
    dt = _tensor_dtype(intent, ins[0])
    if dt and dt != "f32":
        return ["kron_requires_f32"]
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


def _check_sort(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 1:
        return ["sort_invalid_inputs"]
    if not out:
        return ["sort_missing_output"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    try:
        axis_i = int(axis) if axis is not None else None
    except Exception:
        axis_i = None
    if axis_i != 1:
        return ["sort_axis_not_1"]
    for flag in ("descending", "stable"):
        if flag in attrs and not isinstance(attrs.get(flag), (bool, int)):
            return [f"sort_{flag}_not_bool"]
    in_shape = _tensor_shape(intent, ins[0])
    out_shape = _tensor_shape(intent, out)
    if in_shape and len(in_shape) != 2:
        return ["sort_requires_rank2_input"]
    if out_shape and len(out_shape) != 2:
        return ["sort_requires_rank2_output"]
    if in_shape and out_shape and in_shape != out_shape:
        return ["sort_shape_mismatch"]
    return []


def _check_quantile(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 2:
        return ["quantile_invalid_inputs"]
    if not out:
        return ["quantile_missing_output"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dim = attrs.get("dim")
    keepdim = attrs.get("keepdim")
    interp = str(attrs.get("interpolation") or "").strip().lower()
    try:
        dim_i = int(dim) if dim is not None else None
    except Exception:
        dim_i = None
    if dim_i != 1:
        return ["quantile_dim_not_1"]
    if bool(keepdim):
        return ["quantile_keepdim_true"]
    if interp and interp not in {"linear"}:
        return ["quantile_interpolation_not_linear"]
    in_shape = _tensor_shape(intent, ins[0])
    q_shape = _tensor_shape(intent, ins[1])
    out_shape = _tensor_shape(intent, out)
    if in_shape and len(in_shape) != 2:
        return ["quantile_input_rank_not_2"]
    if q_shape and len(q_shape) != 0:
        return ["quantile_q_not_scalar"]
    if out_shape and len(out_shape) != 1:
        return ["quantile_output_rank_not_1"]
    if in_shape and out_shape and out_shape[0] != in_shape[0]:
        return ["quantile_output_mismatch"]
    return []


def _check_avg_pool2d(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 1 or not out:
        return ["avg_pool2d_invalid_io"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    kernel_size = attrs.get("kernel_size")
    stride = attrs.get("stride")
    padding = attrs.get("padding")
    ceil_mode = bool(attrs.get("ceil_mode")) if attrs.get("ceil_mode") is not None else False
    kernel_size_list = list(kernel_size) if isinstance(kernel_size, list) else []
    stride_list = list(stride) if isinstance(stride, list) else []
    padding_list = list(padding) if isinstance(padding, list) else []
    if kernel_size_list != [2, 2] or stride_list != [2, 2] or padding_list != [0, 0] or ceil_mode:
        return ["avg_pool2d_noncanonical_params"]
    in_shape = _tensor_shape(intent, ins[0])
    out_shape = _tensor_shape(intent, out)
    if in_shape and len(in_shape) != 4:
        return ["avg_pool2d_input_rank_not_4"]
    if out_shape and len(out_shape) != 4:
        return ["avg_pool2d_output_rank_not_4"]
    if in_shape and out_shape:
        if out_shape[0] != in_shape[0] or out_shape[1] != in_shape[1]:
            return ["avg_pool2d_nc_mismatch"]
        # If we have concrete shapes, run a cheap bounds sanity check.
        try:
            h = int(in_shape[2])
            w = int(in_shape[3])
            oh = int(out_shape[2])
            ow = int(out_shape[3])
        except Exception:
            h = w = oh = ow = None
        if h is not None and w is not None and oh is not None and ow is not None:
            if 2 * oh > h or 2 * ow > w:
                return ["avg_pool2d_output_out_of_bounds"]
    dt_in = _tensor_dtype(intent, ins[0])
    dt_out = _tensor_dtype(intent, out)
    if dt_in and dt_in != "f32":
        return ["avg_pool2d_input_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["avg_pool2d_output_dtype_not_f32"]
    return []


def _check_max_pool2d_with_indices(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 1 or not out:
        return ["max_pool2d_with_indices_invalid_io"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    kernel_size = attrs.get("kernel_size")
    stride = attrs.get("stride")
    padding = attrs.get("padding")
    dilation = attrs.get("dilation")
    ceil_mode = bool(attrs.get("ceil_mode")) if attrs.get("ceil_mode") is not None else False
    select = str(attrs.get("select") or "").strip().lower()
    kernel_size_list = list(kernel_size) if isinstance(kernel_size, list) else []
    stride_list = list(stride) if isinstance(stride, list) else []
    padding_list = list(padding) if isinstance(padding, list) else []
    dilation_list = list(dilation) if isinstance(dilation, list) else []
    if (
        kernel_size_list != [2, 2]
        or stride_list != [2, 2]
        or padding_list != [0, 0]
        or dilation_list != [1, 1]
        or ceil_mode
        or select not in {"values", "indices"}
    ):
        return ["max_pool2d_with_indices_noncanonical_params"]
    in_shape = _tensor_shape(intent, ins[0])
    out_shape = _tensor_shape(intent, out)
    if in_shape and len(in_shape) != 4:
        return ["max_pool2d_with_indices_input_rank_not_4"]
    if out_shape and len(out_shape) != 4:
        return ["max_pool2d_with_indices_output_rank_not_4"]
    if in_shape and out_shape:
        if out_shape[0] != in_shape[0] or out_shape[1] != in_shape[1]:
            return ["max_pool2d_with_indices_nc_mismatch"]
        # If we have concrete shapes, run a cheap bounds sanity check.
        try:
            h = int(in_shape[2])
            w = int(in_shape[3])
            oh = int(out_shape[2])
            ow = int(out_shape[3])
        except Exception:
            h = w = oh = ow = None
        if h is not None and w is not None and oh is not None and ow is not None:
            if 2 * oh > h or 2 * ow > w:
                return ["max_pool2d_with_indices_output_out_of_bounds"]
    dt_in = _tensor_dtype(intent, ins[0])
    dt_out = _tensor_dtype(intent, out)
    if dt_in and dt_in != "f32":
        return ["max_pool2d_with_indices_input_dtype_not_f32"]
    if select == "values":
        if dt_out and dt_out != "f32":
            return ["max_pool2d_with_indices_values_dtype_not_f32"]
    else:
        if dt_out and dt_out != "i64":
            return ["max_pool2d_with_indices_indices_dtype_not_i64"]
    return []


def _check_index_add(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 3 or not out:
        return ["index_add_invalid_io"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    axis = attrs.get("axis")
    alpha = attrs.get("alpha")
    try:
        axis_i = int(axis) if axis is not None else None
    except Exception:
        axis_i = None
    if axis_i != 0:
        return ["index_add_axis_not_0"]
    if alpha is not None:
        try:
            float(alpha)
        except Exception:
            return ["index_add_alpha_not_number"]
    base_shape = _tensor_shape(intent, ins[0])
    index_shape = _tensor_shape(intent, ins[1])
    src_shape = _tensor_shape(intent, ins[2])
    out_shape = _tensor_shape(intent, out)
    if base_shape and len(base_shape) != 2:
        return ["index_add_base_rank_not_2"]
    if out_shape and len(out_shape) != 2:
        return ["index_add_out_rank_not_2"]
    if index_shape and len(index_shape) != 1:
        return ["index_add_index_rank_not_1"]
    if src_shape and len(src_shape) != 2:
        return ["index_add_src_rank_not_2"]
    if base_shape and out_shape and base_shape != out_shape:
        return ["index_add_out_shape_mismatch"]
    if index_shape and src_shape and index_shape[0] != src_shape[0]:
        return ["index_add_l_mismatch"]
    if base_shape and src_shape and base_shape and src_shape and len(base_shape) == 2 and len(src_shape) == 2:
        if base_shape[1] != src_shape[1]:
            return ["index_add_n_mismatch"]
    dt_base = _tensor_dtype(intent, ins[0])
    dt_idx = _tensor_dtype(intent, ins[1])
    dt_src = _tensor_dtype(intent, ins[2])
    dt_out = _tensor_dtype(intent, out)
    if dt_base and dt_base != "f32":
        return ["index_add_base_dtype_not_f32"]
    if dt_src and dt_src != "f32":
        return ["index_add_src_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["index_add_out_dtype_not_f32"]
    if dt_idx and dt_idx != "i32":
        return ["index_add_index_dtype_not_i32"]
    return []


def _check_index_put(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 4 or not out:
        return ["index_put_invalid_io"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    accumulate = bool(attrs.get("accumulate")) if attrs.get("accumulate") is not None else False
    if accumulate:
        return ["index_put_accumulate_true"]
    base_shape = _tensor_shape(intent, ins[0])
    row_shape = _tensor_shape(intent, ins[1])
    col_shape = _tensor_shape(intent, ins[2])
    val_shape = _tensor_shape(intent, ins[3])
    out_shape = _tensor_shape(intent, out)
    if base_shape and len(base_shape) != 2:
        return ["index_put_base_rank_not_2"]
    if out_shape and len(out_shape) != 2:
        return ["index_put_out_rank_not_2"]
    if row_shape and len(row_shape) != 1:
        return ["index_put_row_rank_not_1"]
    if col_shape and len(col_shape) != 1:
        return ["index_put_col_rank_not_1"]
    if val_shape and len(val_shape) != 1:
        return ["index_put_values_rank_not_1"]
    if base_shape and out_shape and base_shape != out_shape:
        return ["index_put_out_shape_mismatch"]
    if row_shape and col_shape and row_shape != col_shape:
        return ["index_put_row_col_shape_mismatch"]
    if row_shape and val_shape and row_shape != val_shape:
        return ["index_put_row_values_shape_mismatch"]
    dt_base = _tensor_dtype(intent, ins[0])
    dt_row = _tensor_dtype(intent, ins[1])
    dt_col = _tensor_dtype(intent, ins[2])
    dt_val = _tensor_dtype(intent, ins[3])
    dt_out = _tensor_dtype(intent, out)
    if dt_base and dt_base != "f32":
        return ["index_put_base_dtype_not_f32"]
    if dt_val and dt_val != "f32":
        return ["index_put_values_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["index_put_out_dtype_not_f32"]
    if dt_row and dt_row != "i32":
        return ["index_put_row_dtype_not_i32"]
    if dt_col and dt_col != "i32":
        return ["index_put_col_dtype_not_i32"]
    return []


def _check_scatter(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 3 or not out:
        return ["scatter_invalid_io"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dim = attrs.get("dim")
    try:
        dim_i = int(dim) if dim is not None else None
    except Exception:
        dim_i = None
    if dim_i != 1:
        return ["scatter_dim_not_1"]
    inp_shape = _tensor_shape(intent, ins[0])
    index_shape = _tensor_shape(intent, ins[1])
    src_shape = _tensor_shape(intent, ins[2])
    out_shape = _tensor_shape(intent, out)
    if inp_shape and len(inp_shape) != 2:
        return ["scatter_inp_rank_not_2"]
    if out_shape and len(out_shape) != 2:
        return ["scatter_out_rank_not_2"]
    if index_shape and len(index_shape) != 2:
        return ["scatter_index_rank_not_2"]
    if src_shape and len(src_shape) != 2:
        return ["scatter_src_rank_not_2"]
    if inp_shape and out_shape and inp_shape != out_shape:
        return ["scatter_out_shape_mismatch"]
    if inp_shape and index_shape and inp_shape != index_shape:
        return ["scatter_index_shape_mismatch"]
    if inp_shape and src_shape and inp_shape != src_shape:
        return ["scatter_src_shape_mismatch"]
    dt_inp = _tensor_dtype(intent, ins[0])
    dt_idx = _tensor_dtype(intent, ins[1])
    dt_src = _tensor_dtype(intent, ins[2])
    dt_out = _tensor_dtype(intent, out)
    if dt_inp and dt_inp != "f32":
        return ["scatter_inp_dtype_not_f32"]
    if dt_src and dt_src != "f32":
        return ["scatter_src_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["scatter_out_dtype_not_f32"]
    if dt_idx and dt_idx != "i32":
        return ["scatter_index_dtype_not_i32"]
    return []


def _check_select_scatter(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 2 or not out:
        return ["select_scatter_invalid_io"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dim = attrs.get("dim")
    index = attrs.get("index")
    try:
        dim_i = int(dim) if dim is not None else None
    except Exception:
        dim_i = None
    if dim_i != 1:
        return ["select_scatter_dim_not_1"]
    try:
        idx_i = int(index) if index is not None else None
    except Exception:
        idx_i = None
    if idx_i is None:
        return ["select_scatter_missing_index"]
    inp_shape = _tensor_shape(intent, ins[0])
    src_shape = _tensor_shape(intent, ins[1])
    out_shape = _tensor_shape(intent, out)
    if inp_shape and len(inp_shape) != 2:
        return ["select_scatter_inp_rank_not_2"]
    if out_shape and len(out_shape) != 2:
        return ["select_scatter_out_rank_not_2"]
    if src_shape and len(src_shape) != 1:
        return ["select_scatter_src_rank_not_1"]
    if inp_shape and out_shape and inp_shape != out_shape:
        return ["select_scatter_out_shape_mismatch"]
    if inp_shape and src_shape and len(inp_shape) == 2 and len(src_shape) == 1:
        if inp_shape[0] != src_shape[0]:
            return ["select_scatter_m_mismatch"]
        # If N is static, range-check the selected column.
        try:
            n = int(inp_shape[1])
        except Exception:
            n = None
        if n is not None and (idx_i < 0 or idx_i >= n):
            return ["select_scatter_index_oob"]
    dt_inp = _tensor_dtype(intent, ins[0])
    dt_src = _tensor_dtype(intent, ins[1])
    dt_out = _tensor_dtype(intent, out)
    if dt_inp and dt_inp != "f32":
        return ["select_scatter_inp_dtype_not_f32"]
    if dt_src and dt_src != "f32":
        return ["select_scatter_src_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["select_scatter_out_dtype_not_f32"]
    return []


def _check_slice_scatter(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 2 or not out:
        return ["slice_scatter_invalid_io"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dim = attrs.get("dim")
    start = attrs.get("start")
    end = attrs.get("end")
    step = attrs.get("step")
    try:
        dim_i = int(dim) if dim is not None else None
    except Exception:
        dim_i = None
    if dim_i != 1:
        return ["slice_scatter_dim_not_1"]
    try:
        start_i = int(start) if start is not None else None
    except Exception:
        start_i = None
    try:
        end_i = int(end) if end is not None else None
    except Exception:
        end_i = None
    try:
        step_i = int(step) if step is not None else 1
    except Exception:
        step_i = 1
    if start_i is None or end_i is None or step_i <= 0:
        return ["slice_scatter_invalid_slice"]
    inp_shape = _tensor_shape(intent, ins[0])
    src_shape = _tensor_shape(intent, ins[1])
    out_shape = _tensor_shape(intent, out)
    if inp_shape and len(inp_shape) != 2:
        return ["slice_scatter_inp_rank_not_2"]
    if out_shape and len(out_shape) != 2:
        return ["slice_scatter_out_rank_not_2"]
    if src_shape and len(src_shape) != 2:
        return ["slice_scatter_src_rank_not_2"]
    if inp_shape and out_shape and inp_shape != out_shape:
        return ["slice_scatter_out_shape_mismatch"]
    if inp_shape and src_shape and len(inp_shape) == 2 and len(src_shape) == 2:
        if inp_shape[0] != src_shape[0]:
            return ["slice_scatter_m_mismatch"]
        # If slice length is known statically, validate src second dim.
        try:
            n = int(inp_shape[1])
        except Exception:
            n = None
        if n is not None:
            if not (0 <= start_i <= end_i <= n):
                return ["slice_scatter_slice_oob"]
            expected = len(list(range(int(start_i), int(end_i), int(step_i))))
            try:
                l = int(src_shape[1])
            except Exception:
                l = None
            if l is not None and l != expected:
                return ["slice_scatter_l_mismatch"]
    dt_inp = _tensor_dtype(intent, ins[0])
    dt_src = _tensor_dtype(intent, ins[1])
    dt_out = _tensor_dtype(intent, out)
    if dt_inp and dt_inp != "f32":
        return ["slice_scatter_inp_dtype_not_f32"]
    if dt_src and dt_src != "f32":
        return ["slice_scatter_src_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["slice_scatter_out_dtype_not_f32"]
    return []


def _check_masked_select(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 2 or not out:
        return ["masked_select_invalid_io"]
    inp_shape = _tensor_shape(intent, ins[0])
    mask_shape = _tensor_shape(intent, ins[1])
    out_shape = _tensor_shape(intent, out)
    if inp_shape and len(inp_shape) != 2:
        return ["masked_select_inp_rank_not_2"]
    if mask_shape and len(mask_shape) != 2:
        return ["masked_select_mask_rank_not_2"]
    if out_shape and len(out_shape) != 1:
        return ["masked_select_out_rank_not_1"]
    if inp_shape and mask_shape and inp_shape != mask_shape:
        return ["masked_select_mask_shape_mismatch"]
    dt_inp = _tensor_dtype(intent, ins[0])
    dt_mask = _tensor_dtype(intent, ins[1])
    dt_out = _tensor_dtype(intent, out)
    if dt_inp and dt_inp != "f32":
        return ["masked_select_inp_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["masked_select_out_dtype_not_f32"]
    if dt_mask and dt_mask not in {"bool", "i1"}:
        return ["masked_select_mask_dtype_not_bool"]
    return []


def _check_masked_scatter(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 3 or not out:
        return ["masked_scatter_invalid_io"]
    inp_shape = _tensor_shape(intent, ins[0])
    mask_shape = _tensor_shape(intent, ins[1])
    src_shape = _tensor_shape(intent, ins[2])
    out_shape = _tensor_shape(intent, out)
    if inp_shape and len(inp_shape) != 2:
        return ["masked_scatter_inp_rank_not_2"]
    if mask_shape and len(mask_shape) != 2:
        return ["masked_scatter_mask_rank_not_2"]
    if out_shape and len(out_shape) != 2:
        return ["masked_scatter_out_rank_not_2"]
    if src_shape and len(src_shape) != 1:
        return ["masked_scatter_source_rank_not_1"]
    if inp_shape and mask_shape and inp_shape != mask_shape:
        return ["masked_scatter_mask_shape_mismatch"]
    if inp_shape and out_shape and inp_shape != out_shape:
        return ["masked_scatter_out_shape_mismatch"]
    dt_inp = _tensor_dtype(intent, ins[0])
    dt_mask = _tensor_dtype(intent, ins[1])
    dt_src = _tensor_dtype(intent, ins[2])
    dt_out = _tensor_dtype(intent, out)
    if dt_inp and dt_inp != "f32":
        return ["masked_scatter_inp_dtype_not_f32"]
    if dt_src and dt_src != "f32":
        return ["masked_scatter_source_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["masked_scatter_out_dtype_not_f32"]
    if dt_mask and dt_mask not in {"bool", "i1"}:
        return ["masked_scatter_mask_dtype_not_bool"]
    return []


def _check_nll_loss_forward(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 3 or not out:
        return ["nll_loss_forward_invalid_io"]
    logits_shape = _tensor_shape(intent, ins[0])
    tgt_shape = _tensor_shape(intent, ins[1])
    w_shape = _tensor_shape(intent, ins[2])
    out_shape = _tensor_shape(intent, out)
    if logits_shape and len(logits_shape) != 2:
        return ["nll_loss_forward_logits_rank_not_2"]
    if tgt_shape and len(tgt_shape) != 1:
        return ["nll_loss_forward_target_rank_not_1"]
    if w_shape and len(w_shape) != 1:
        return ["nll_loss_forward_weight_rank_not_1"]
    if out_shape and len(out_shape) != 0:
        return ["nll_loss_forward_out_rank_not_0"]
    if logits_shape and tgt_shape:
        if logits_shape[0] != tgt_shape[0]:
            return ["nll_loss_forward_n_mismatch"]
    if logits_shape and w_shape:
        if logits_shape[1] != w_shape[0]:
            return ["nll_loss_forward_c_mismatch"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    reduction = attrs.get("reduction", 1)
    try:
        reduction_i = int(reduction)
    except Exception:
        reduction_i = 1
    if reduction_i != 1:
        return ["nll_loss_forward_reduction_not_mean"]
    dt_logits = _tensor_dtype(intent, ins[0])
    dt_tgt = _tensor_dtype(intent, ins[1])
    dt_w = _tensor_dtype(intent, ins[2])
    dt_out = _tensor_dtype(intent, out)
    if dt_logits and dt_logits != "f32":
        return ["nll_loss_forward_logits_dtype_not_f32"]
    if dt_tgt and dt_tgt != "i64":
        return ["nll_loss_forward_target_dtype_not_i64"]
    if dt_w and dt_w != "f32":
        return ["nll_loss_forward_weight_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["nll_loss_forward_out_dtype_not_f32"]
    return []


def _check_nll_loss2d_forward(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    out = str(op.get("output") or "").strip()
    if len(ins) != 3 or not out:
        return ["nll_loss2d_forward_invalid_io"]
    logits_shape = _tensor_shape(intent, ins[0])
    tgt_shape = _tensor_shape(intent, ins[1])
    w_shape = _tensor_shape(intent, ins[2])
    out_shape = _tensor_shape(intent, out)
    if logits_shape and len(logits_shape) != 4:
        return ["nll_loss2d_forward_logits_rank_not_4"]
    if tgt_shape and len(tgt_shape) != 3:
        return ["nll_loss2d_forward_target_rank_not_3"]
    if w_shape and len(w_shape) != 1:
        return ["nll_loss2d_forward_weight_rank_not_1"]
    if out_shape and len(out_shape) != 0:
        return ["nll_loss2d_forward_out_rank_not_0"]
    if logits_shape and tgt_shape:
        if logits_shape[0] != tgt_shape[0] or logits_shape[2] != tgt_shape[1] or logits_shape[3] != tgt_shape[2]:
            return ["nll_loss2d_forward_nhw_mismatch"]
    if logits_shape and w_shape:
        if logits_shape[1] != w_shape[0]:
            return ["nll_loss2d_forward_c_mismatch"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    reduction = attrs.get("reduction", 1)
    try:
        reduction_i = int(reduction)
    except Exception:
        reduction_i = 1
    if reduction_i != 1:
        return ["nll_loss2d_forward_reduction_not_mean"]
    dt_logits = _tensor_dtype(intent, ins[0])
    dt_tgt = _tensor_dtype(intent, ins[1])
    dt_w = _tensor_dtype(intent, ins[2])
    dt_out = _tensor_dtype(intent, out)
    if dt_logits and dt_logits != "f32":
        return ["nll_loss2d_forward_logits_dtype_not_f32"]
    if dt_tgt and dt_tgt != "i64":
        return ["nll_loss2d_forward_target_dtype_not_i64"]
    if dt_w and dt_w != "f32":
        return ["nll_loss2d_forward_weight_dtype_not_f32"]
    if dt_out and dt_out != "f32":
        return ["nll_loss2d_forward_out_dtype_not_f32"]
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
        "max_pool2d_with_indices_nchw",
        "per_token_group_quant_fp8_2d",
        "batch_norm2d",
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
        "bmm3d",
        "baddbmm3d",
        "group_norm_kernel",
        "batch_norm2d",
        "scaled_dot_product_attention_bhsd",
        "flash_attn_varlen_func_bhsd",
        "upsample_bicubic2d_aa",
        "avg_pool2d_nchw",
        "max_pool2d_with_indices_nchw",
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
        elif name == "nll_loss_forward":
            reasons.extend(_check_nll_loss_forward(intent, o))
        elif name == "nll_loss2d_forward":
            reasons.extend(_check_nll_loss2d_forward(intent, o))
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
        elif name == "cumsum":
            reasons.extend(_check_cumsum(intent, o))
        elif name == "cummax":
            reasons.extend(_check_cummax(intent, o))
        elif name == "cummin":
            reasons.extend(_check_cummin(intent, o))
        elif name == "kron":
            reasons.extend(_check_kron(intent, o))
        elif name == "sort":
            reasons.extend(_check_sort(intent, o))
        elif name == "quantile":
            reasons.extend(_check_quantile(intent, o))
        elif name == "avg_pool2d":
            reasons.extend(_check_avg_pool2d(intent, o))
        elif name == "max_pool2d_with_indices":
            reasons.extend(_check_max_pool2d_with_indices(intent, o))
        elif name == "index_add":
            reasons.extend(_check_index_add(intent, o))
        elif name == "index_put":
            reasons.extend(_check_index_put(intent, o))
        elif name == "scatter":
            reasons.extend(_check_scatter(intent, o))
        elif name == "select_scatter":
            reasons.extend(_check_select_scatter(intent, o))
        elif name == "slice_scatter":
            reasons.extend(_check_slice_scatter(intent, o))
        elif name == "masked_select":
            reasons.extend(_check_masked_select(intent, o))
        elif name == "masked_scatter":
            reasons.extend(_check_masked_scatter(intent, o))
        elif name == "scaled_dot_product_attention":
            reasons.extend(_check_scaled_dot_product_attention(intent, o))
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
