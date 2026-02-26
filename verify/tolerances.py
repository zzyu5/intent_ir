"""
Numerical tolerances for diff-based verification (P0 gap fix).

The system used to rely on a single fixed tolerance (atol=1e-3, rtol=1e-3)
for all kernels. This module introduces a small heuristic that:
  - picks a kernel-level tolerance based on the op mix
  - adjusts for output dtype (f16/bf16 vs f32)

This is deliberately conservative: it never *loosens* beyond the historical
defaults for the common kernels unless explicitly required by dtype/op class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set

import numpy as np

from intent_ir.ir import IntentFunction


@dataclass(frozen=True)
class Tolerances:
    atol: float
    rtol: float

    def to_dict(self) -> Dict[str, float]:
        return {"atol": float(self.atol), "rtol": float(self.rtol)}


# Per-op baseline tolerances for f32 outputs.
# NOTE: This is a kernel-level heuristic: we take max across ops.
_OP_TOL_F32: Dict[str, Tolerances] = {
    # Numerically sensitive ops (transcendentals / normalization).
    "softmax": Tolerances(1e-4, 1e-4),
    "exp": Tolerances(3e-4, 3e-4),
    "rsqrt": Tolerances(3e-4, 3e-4),
    # Accumulation-heavy ops.
    # GPU vs CPU (or different lowering choices) can produce noticeably
    # different rounding for dot-products. Keep this looser than the legacy
    # default to avoid false negatives on matmul-heavy kernels.
    "matmul": Tolerances(2e-2, 2e-2),
    # Attention primitive is effectively two matmuls plus softmax and is
    # routinely more sensitive across execution paths.
    "scaled_dot_product_attention": Tolerances(2e-2, 2e-2),
    "reduce_sum": Tolerances(1e-3, 1e-3),
    "reduce_max": Tolerances(1e-3, 1e-3),
    "reduce_any": Tolerances(0.0, 0.0),
}

# Historical default used by diff_runner before we introduced this module.
_LEGACY_DEFAULT = Tolerances(1e-3, 1e-3)


def _dtype_kind(dt: np.dtype) -> str:
    d = np.dtype(dt)
    if d == np.float16:
        return "f16"
    if d == np.float32:
        return "f32"
    if d == np.float64:
        return "f64"
    if d == np.bool_:
        return "bool"
    if np.issubdtype(d, np.integer):
        return "int"
    return str(d)


def infer_tolerances(
    intent: IntentFunction,
    *,
    ref_out: Optional[Dict[str, np.ndarray]] = None,
) -> Tolerances:
    """
    Infer a *kernel-level* (atol, rtol).

    - Uses op mix heuristic to select a baseline.
    - If any output dtype is f16, keep legacy 1e-3 tolerances (avoid false positives).
    - If output is f32-only and op mix is "stable", uses tighter defaults (e.g., softmax 1e-4).
    """
    op_names = [str(op.op) for op in intent.ops]
    op_set: Set[str] = set(op_names)
    matmul_count = sum(1 for op_name in op_names if op_name == "matmul")
    tensor_dtypes = {str(name): str(t.dtype) for name, t in intent.tensors.items() if t is not None}
    tensor_names = set(tensor_dtypes.keys())
    output_names = set(str(x) for x in list(intent.outputs or []))

    return _infer_tolerances_core(
        op_set=op_set,
        matmul_count=matmul_count,
        tensor_dtypes=tensor_dtypes,
        tensor_names=tensor_names,
        output_names=output_names,
        ref_out=ref_out,
    )


def _infer_tolerances_core(
    *,
    op_set: Set[str],
    matmul_count: int,
    tensor_dtypes: Dict[str, str],
    tensor_names: Set[str],
    output_names: Set[str],
    ref_out: Optional[Dict[str, np.ndarray]],
) -> Tolerances:

    # 1) Op mix.
    tol = Tolerances(1e-4, 1e-4)
    for op_name in op_set:
        op_tol = _OP_TOL_F32.get(op_name)
        if op_tol is None:
            continue
        tol = Tolerances(atol=max(tol.atol, op_tol.atol), rtol=max(tol.rtol, op_tol.rtol))
    if matmul_count > 1:
        # Multiple matmuls in one graph (e.g., attention / MLP) accumulate more rounding error.
        matmul_base = 2e-2 * (1.0 + 0.5 * max(0, matmul_count - 1))
        tol = Tolerances(atol=max(tol.atol, matmul_base), rtol=max(tol.rtol, matmul_base))

    # 2) Dtype adjustment (outputs AND external inputs).
    # If any external dtype is f16/bf16, keep legacy tolerances (avoid false positives).
    out_dtypes: Set[str] = set()
    if ref_out is not None:
        for name in (set(output_names) | set(tensor_names)):
            if name in ref_out:
                try:
                    out_dtypes.add(_dtype_kind(np.asarray(ref_out[name]).dtype))
                except Exception:
                    pass
    else:
        for name, dtype in tensor_dtypes.items():
            if str(name).strip():
                out_dtypes.add(str(dtype))

    # Conservative: if any output is f16/bf16, do not tighten below legacy default.
    if ("f16" in out_dtypes) or ("bf16" in out_dtypes):
        tol = Tolerances(atol=max(tol.atol, _LEGACY_DEFAULT.atol), rtol=max(tol.rtol, _LEGACY_DEFAULT.rtol))
        # bf16 min/max style kernels often differ slightly across backends due to
        # conversion and tie-breaking behavior near representable boundaries.
        if ("bf16" in out_dtypes) and ({"min", "max"} & op_set):
            tol = Tolerances(atol=max(tol.atol, 1e-2), rtol=max(tol.rtol, 1e-2))

    # If we couldn't infer anything meaningful, keep legacy behavior.
    if tol.atol == 0.0 and tol.rtol == 0.0:
        return _LEGACY_DEFAULT

    # 3) Global cap: keep most kernels at legacy tolerances, but allow explicit loosening
    # for certain op classes (e.g. matmul).
    cap = _LEGACY_DEFAULT
    if matmul_count > 0:
        # A single matmul typically stays within ~2e-2 across backends, but fused graphs
        # (e.g., attention or MLP with multiple matmuls) can accumulate more rounding error,
        # especially when the reference uses TF32-like dot paths.
        matmul_cap = 2e-2 * (1.0 + 0.5 * max(0, matmul_count - 1))
        cap = Tolerances(atol=max(cap.atol, matmul_cap), rtol=max(cap.rtol, matmul_cap))
    if ("bf16" in out_dtypes) and ({"min", "max"} & op_set):
        cap = Tolerances(atol=max(cap.atol, 1e-2), rtol=max(cap.rtol, 1e-2))
    tol = Tolerances(atol=min(max(tol.atol, 0.0), cap.atol), rtol=min(max(tol.rtol, 0.0), cap.rtol))
    return tol


def infer_tolerances_from_intent_json(
    intent_json: Dict[str, object],
    *,
    ref_out: Optional[Dict[str, np.ndarray]] = None,
) -> Tolerances:
    """JSON-first tolerance inference without rebuilding IntentFunction."""
    payload = dict(intent_json or {})
    op_names = [str((op or {}).get("op") or "") for op in list(payload.get("ops") or []) if isinstance(op, dict)]
    op_names = [x for x in op_names if x]
    op_set: Set[str] = set(op_names)
    matmul_count = sum(1 for op_name in op_names if op_name == "matmul")

    tensors_raw = payload.get("tensors")
    tensor_dtypes: Dict[str, str] = {}
    tensor_names: Set[str] = set()
    if isinstance(tensors_raw, dict):
        for name, spec in tensors_raw.items():
            key = str(name).strip()
            if not key:
                continue
            tensor_names.add(key)
            if isinstance(spec, dict):
                tensor_dtypes[key] = str(spec.get("dtype") or "")

    outputs_raw = payload.get("outputs")
    output_names: Set[str] = set()
    if isinstance(outputs_raw, list):
        output_names = {str(x).strip() for x in outputs_raw if str(x).strip()}

    return _infer_tolerances_core(
        op_set=op_set,
        matmul_count=matmul_count,
        tensor_dtypes=tensor_dtypes,
        tensor_names=tensor_names,
        output_names=output_names,
        ref_out=ref_out,
    )


__all__ = ["Tolerances", "infer_tolerances", "infer_tolerances_from_intent_json"]
