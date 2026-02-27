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
    "reduce_sum",
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
    # CUDA real-MLIR lowering treats broadcast_in_dim as shape-only and relies on
    # IO-spec broadcast classification. Accept common 1D->2D patterns.
    if (
        isinstance(out_shape, list)
        and len(out_shape) == 2
        and len(in_shape) == 1
        and isinstance(bcast_dims, list)
        and len(bcast_dims) == 1
        and int(bcast_dims[0]) in {0, 1}
    ):
        return []
    return ["broadcast_in_dim_unsupported_pattern"]
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
    if len(ins) != 3:
        return ["gather_invalid_inputs"]
    # Current lowering supports 2D inp + (row_idx,col_idx) -> out (rank 1/2).
    if len(_tensor_shape(intent, ins[0])) != 2:
        return ["gather_requires_2d_input"]
    return []


def _check_reduce_sum(intent: dict[str, Any], op: dict[str, Any]) -> list[str]:
    ins = [str(x) for x in list(op.get("inputs") or []) if str(x).strip()]
    if len(ins) != 1:
        return ["reduce_sum_invalid_inputs"]
    attrs = op.get("attrs") if isinstance(op.get("attrs"), dict) else {}
    dims = attrs.get("dims")
    dims_list = list(dims) if isinstance(dims, list) else []
    if dims_list != [1]:
        return ["reduce_sum_dims_not_axis1"]
    outs = [str(x) for x in list(intent.get("outputs") or []) if str(x).strip()]
    if len(outs) != 1:
        return ["reduce_sum_requires_single_output"]
    final_out = outs[0]
    if len(_tensor_shape(intent, final_out)) != 1:
        return ["reduce_sum_requires_rank1_final_output"]
    out = str(op.get("output") or "").strip()
    if out and len(_tensor_shape(intent, out)) != 1:
        return ["reduce_sum_output_not_rank1"]
    return []


def _supported_by_cuda_real_mlir(intent: dict[str, Any]) -> SupportResult:
    outputs = [str(x) for x in list(intent.get("outputs") or []) if str(x).strip()]
    if len(outputs) != 1:
        return SupportResult(ok=False, reasons=["multi_output"], unsupported_ops=[])
    out = outputs[0]
    out_rank = len(_tensor_shape(intent, out))
    if out_rank > 2:
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
        elif name == "reduce_sum":
            reasons.extend(_check_reduce_sum(intent, o))
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
