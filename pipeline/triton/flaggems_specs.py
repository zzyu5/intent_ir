"""
FlagGems-backed Triton kernel specs for the existing Triton full pipeline.

Important:
- This is NOT a new frontend. We reuse `pipeline.triton.core.run_pipeline_for_spec`.
- Runners execute PyTorch APIs under `flag_gems.use_gems(...)` so ATen dispatch
  goes through FlagGems Triton kernels.
- Spec names intentionally align with existing Triton/TileLang semantic kernels
  (`add2d`, `softmax_inner`) so deterministic fallback remains available when
  LLM providers are unavailable.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from pipeline.triton.core import KernelSpec
from pipeline.triton.flaggems_registry import (
    DEFAULT_FLAGGEMS_OPSET,
    list_supported_e2e_specs,
    load_registry,
    load_registry_entry_by_spec,
)
from verify.gen_cases import TestCase


ROOT = Path(__file__).resolve().parents[2]


def _ensure_flaggems_importable() -> None:
    candidates: List[Path] = []
    env = os.getenv("FLAGGEMS_SRC")
    if isinstance(env, str) and env.strip():
        candidates.append(Path(env.strip()))
    candidates.append(ROOT / "experiment" / "FlagGems" / "src")
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


_ensure_flaggems_importable()
import flag_gems  # noqa: E402
from flag_gems import ops as flag_gems_ops  # noqa: E402


def _module_source_text(mod_name: str) -> str:
    mod = importlib.import_module(mod_name)
    p = Path(getattr(mod, "__file__", ""))
    if not p.is_file():
        return str(mod)
    return p.read_text(encoding="utf-8")


# Use full module source so LLM/evidence stage has enough context.
FLAGGEMS_ANY_SRC = _module_source_text("flag_gems.ops.any")
FLAGGEMS_ADD_SRC = _module_source_text("flag_gems.ops.add")
FLAGGEMS_SUB_SRC = _module_source_text("flag_gems.ops.sub")
FLAGGEMS_MUL_SRC = _module_source_text("flag_gems.ops.mul")
FLAGGEMS_DIV_SRC = _module_source_text("flag_gems.ops.div")
FLAGGEMS_EQ_SRC = _module_source_text("flag_gems.ops.eq")
FLAGGEMS_NE_SRC = _module_source_text("flag_gems.ops.ne")
FLAGGEMS_GT_SRC = _module_source_text("flag_gems.ops.gt")
FLAGGEMS_GE_SRC = _module_source_text("flag_gems.ops.ge")
FLAGGEMS_LT_SRC = _module_source_text("flag_gems.ops.lt")
FLAGGEMS_LE_SRC = _module_source_text("flag_gems.ops.le")
FLAGGEMS_NEG_SRC = _module_source_text("flag_gems.ops.neg")
FLAGGEMS_CEIL_SRC = _module_source_text("flag_gems.ops.ceil")
FLAGGEMS_RECIPROCAL_SRC = _module_source_text("flag_gems.ops.reciprocal")
FLAGGEMS_SQRT_SRC = _module_source_text("flag_gems.ops.sqrt")
FLAGGEMS_EXP2_SRC = _module_source_text("flag_gems.ops.exp2")
FLAGGEMS_SIGMOID_SRC = _module_source_text("flag_gems.ops.sigmoid")
FLAGGEMS_SILU_SRC = _module_source_text("flag_gems.ops.silu")
FLAGGEMS_TANH_SRC = _module_source_text("flag_gems.ops.tanh")
FLAGGEMS_LOGICAL_AND_SRC = _module_source_text("flag_gems.ops.logical_and")
FLAGGEMS_LOGICAL_OR_SRC = _module_source_text("flag_gems.ops.logical_or")
FLAGGEMS_LOGICAL_NOT_SRC = _module_source_text("flag_gems.ops.logical_not")
FLAGGEMS_LOGICAL_XOR_SRC = _module_source_text("flag_gems.ops.logical_xor")
FLAGGEMS_GROUP_NORM_SRC = _module_source_text("flag_gems.ops.groupnorm")
FLAGGEMS_LAYER_NORM_SRC = _module_source_text("flag_gems.ops.layernorm")
FLAGGEMS_BATCH_NORM_SRC = _module_source_text("flag_gems.ops.batch_norm")
FLAGGEMS_RMS_NORM_SRC = _module_source_text("flag_gems.ops.rms_norm")
FLAGGEMS_LERP_SRC = _module_source_text("flag_gems.ops.lerp")
FLAGGEMS_MM_SRC = _module_source_text("flag_gems.ops.mm")
FLAGGEMS_BMM_SRC = _module_source_text("flag_gems.ops.bmm")
FLAGGEMS_ADDMM_SRC = _module_source_text("flag_gems.ops.addmm")
FLAGGEMS_BADDBMM_SRC = _module_source_text("flag_gems.ops.baddbmm")
FLAGGEMS_DOT_SRC = _module_source_text("flag_gems.ops.dot")
FLAGGEMS_VDOT_SRC = _module_source_text("flag_gems.ops.vdot")
FLAGGEMS_MV_SRC = _module_source_text("flag_gems.ops.mv")
FLAGGEMS_ADDMV_SRC = _module_source_text("flag_gems.ops.addmv")
FLAGGEMS_FLIP_SRC = _module_source_text("flag_gems.ops.flip")
FLAGGEMS_INDEX_SELECT_SRC = _module_source_text("flag_gems.ops.index_select")
FLAGGEMS_SOFTMAX_SRC = _module_source_text("flag_gems.ops.softmax")
FLAGGEMS_RELU_SRC = _module_source_text("flag_gems.ops.relu")
FLAGGEMS_EXP_SRC = _module_source_text("flag_gems.ops.exp")
FLAGGEMS_ABS_SRC = _module_source_text("flag_gems.ops.abs")
FLAGGEMS_RSQRT_SRC = _module_source_text("flag_gems.ops.rsqrt")
FLAGGEMS_GATHER_SRC = _module_source_text("flag_gems.ops.gather")
FLAGGEMS_WHERE_SRC = _module_source_text("flag_gems.ops.where")
FLAGGEMS_SUM_SRC = _module_source_text("flag_gems.ops.sum")
FLAGGEMS_MAX_SRC = _module_source_text("flag_gems.ops.max")
FLAGGEMS_MEAN_SRC = _module_source_text("flag_gems.ops.mean")
FLAGGEMS_ALL_SRC = _module_source_text("flag_gems.ops.all")
FLAGGEMS_MAXIMUM_SRC = _module_source_text("flag_gems.ops.maximum")
FLAGGEMS_MINIMUM_SRC = _module_source_text("flag_gems.ops.minimum")
FLAGGEMS_FULL_SRC = _module_source_text("flag_gems.ops.full")
FLAGGEMS_COPY_SRC = _module_source_text("flag_gems.ops.copy")
FLAGGEMS_TO_COPY_SRC = _module_source_text("flag_gems.ops.to")
FLAGGEMS_CLAMP_SRC = _module_source_text("flag_gems.ops.clamp")
FLAGGEMS_UPSAMPLE_BICUBIC2D_AA_SRC = _module_source_text("flag_gems.ops.upsample_bicubic2d_aa")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _as_f32_tensor(x: np.ndarray, *, device: str) -> torch.Tensor:
    t = torch.as_tensor(x, device=device)
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    return t


def _run_flaggems_add2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["add"]):
        c = a + b

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    alpha = np.array(1.0, dtype=np.float32)
    if case.inputs and "alpha" in case.inputs:
        alpha = np.asarray(case.inputs["alpha"], dtype=np.float32)
    return {
        "A": a_np,
        "B": b_np,
        "C": c_np,
        "alpha": alpha,
        # Common aliases for parser/runtime naming variance.
        "a": a_np,
        "b": b_np,
        "x": a_np,
        "y": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_sub2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["sub"]):
        c = a - b

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "C": c_np,
        "a": a_np,
        "b": b_np,
        "x": a_np,
        "y": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_mul2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["mul"]):
        c = a * b

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "C": c_np,
        "a": a_np,
        "b": b_np,
        "x": a_np,
        "y": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_binary2d_reference(
    case: TestCase,
    *,
    include: List[str],
    op_name: str,
) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if op_name == "div":
        # Stabilize division to avoid accidental inf/nan from near-zero denominator.
        b = torch.where(torch.abs(b) < 1e-3, b + 1e-3, b)

    with flag_gems.use_gems(include=include):
        if op_name == "div":
            c = torch.div(a, b)
        elif op_name == "max":
            c = torch.maximum(a, b)
        elif op_name == "min":
            c = torch.minimum(a, b)
        elif op_name == "eq":
            c = torch.eq(a, b)
        elif op_name == "ne":
            c = torch.ne(a, b)
        elif op_name == "gt":
            c = torch.gt(a, b)
        elif op_name == "ge":
            c = torch.ge(a, b)
        elif op_name == "lt":
            c = torch.lt(a, b)
        elif op_name == "le":
            c = torch.le(a, b)
        else:
            raise ValueError(f"unsupported op_name for binary2d runner: {op_name}")

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "C": c_np,
        "a": a_np,
        "b": b_np,
        "x": a_np,
        "y": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_div2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["div", "div_mode", "true_divide", "floor_divide"],
        op_name="div",
    )


def _run_flaggems_eq2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["eq", "equal", "eq_scalar"],
        op_name="eq",
    )


def _run_flaggems_ne2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["ne", "ne_scalar"],
        op_name="ne",
    )


def _run_flaggems_gt2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["gt", "gt_scalar"],
        op_name="gt",
    )


def _run_flaggems_ge2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["ge", "ge_scalar"],
        op_name="ge",
    )


def _run_flaggems_lt2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["lt", "lt_scalar"],
        op_name="lt",
    )


def _run_flaggems_le2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["le", "le_scalar"],
        op_name="le",
    )


def _run_flaggems_neg2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["neg"]):
        out = torch.neg(inp)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "A": inp_np,
        "Out": out_np,
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_unary2d_reference(
    case: TestCase,
    *,
    include: List[str],
    op_name: str,
    positive_input: bool = False,
) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        raw = rg.standard_normal((m, n), dtype=np.float32)
        if positive_input:
            raw = np.abs(raw) + 1e-3
        inp = torch.from_numpy(raw).to(device)

    if positive_input:
        inp = torch.where(inp <= 1e-3, torch.full_like(inp, 1e-3), inp)

    with flag_gems.use_gems(include=include):
        if op_name == "ceil":
            out = torch.ceil(inp)
        elif op_name == "reciprocal":
            out = torch.reciprocal(inp)
        elif op_name == "sqrt":
            out = torch.sqrt(inp)
        elif op_name == "exp2":
            out = torch.exp2(inp)
        elif op_name == "sigmoid":
            out = torch.sigmoid(inp)
        elif op_name == "silu":
            out = torch.nn.functional.silu(inp)
        elif op_name == "tanh":
            out = torch.tanh(inp)
        else:
            raise ValueError(f"unsupported op_name for unary2d runner: {op_name}")

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "A": inp_np,
        "Out": out_np,
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_ceil2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_unary2d_reference(case, include=["ceil"], op_name="ceil")


def _run_flaggems_reciprocal2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_unary2d_reference(
        case,
        include=["reciprocal"],
        op_name="reciprocal",
        positive_input=True,
    )


def _run_flaggems_sqrt2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_unary2d_reference(
        case,
        include=["sqrt"],
        op_name="sqrt",
        positive_input=True,
    )


def _run_flaggems_sigmoid2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_unary2d_reference(
        case,
        include=["sigmoid"],
        op_name="sigmoid",
    )


def _run_flaggems_exp22d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_unary2d_reference(
        case,
        include=["exp2"],
        op_name="exp2",
    )


def _run_flaggems_silu2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_unary2d_reference(
        case,
        include=["silu"],
        op_name="silu",
    )


def _run_flaggems_tanh2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_unary2d_reference(
        case,
        include=["tanh"],
        op_name="tanh",
    )


def _run_flaggems_row_mean_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["mean", "mean_dim"]):
        out = torch.mean(inp, dim=1)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_row_all_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = torch.as_tensor(np.asarray(case.inputs["inp"]), device=device, dtype=torch.bool)
    else:
        inp = torch.from_numpy((rg.random((m, n), dtype=np.float32) > 0.3)).to(device=device, dtype=torch.bool)

    with flag_gems.use_gems(include=["all", "all_dim", "all_dims"]):
        out = torch.all(inp, dim=1)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_logical_binary2d_reference(
    case: TestCase,
    *,
    include: List[str],
    op_name: str,
) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = torch.as_tensor(np.asarray(case.inputs["A"]), device=device, dtype=torch.bool)
    else:
        a = torch.from_numpy((rg.random((m, n), dtype=np.float32) > 0.5)).to(device=device, dtype=torch.bool)

    if case.inputs and "B" in case.inputs:
        b = torch.as_tensor(np.asarray(case.inputs["B"]), device=device, dtype=torch.bool)
    else:
        b = torch.from_numpy((rg.random((m, n), dtype=np.float32) > 0.5)).to(device=device, dtype=torch.bool)

    with flag_gems.use_gems(include=include):
        if op_name == "logical_and":
            c = torch.logical_and(a, b)
        elif op_name == "logical_or":
            c = torch.logical_or(a, b)
        elif op_name == "logical_xor":
            c = torch.logical_xor(a, b)
        else:
            raise ValueError(f"unsupported op_name for logical binary runner: {op_name}")

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "C": c_np,
        "a": a_np,
        "b": b_np,
        "x": a_np,
        "y": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_logical_and2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_logical_binary2d_reference(
        case,
        include=["logical_and"],
        op_name="logical_and",
    )


def _run_flaggems_logical_or2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_logical_binary2d_reference(
        case,
        include=["logical_or"],
        op_name="logical_or",
    )


def _run_flaggems_logical_xor2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_logical_binary2d_reference(
        case,
        include=["logical_xor"],
        op_name="logical_xor",
    )


def _run_flaggems_logical_not2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = torch.as_tensor(np.asarray(case.inputs["inp"]), device=device, dtype=torch.bool)
    else:
        inp = torch.from_numpy((rg.random((m, n), dtype=np.float32) > 0.5)).to(device=device, dtype=torch.bool)

    with flag_gems.use_gems(include=["logical_not"]):
        out = torch.logical_not(inp)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "A": inp_np,
        "Out": out_np,
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_full2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)

    if case.inputs and "value" in case.inputs:
        value = float(np.asarray(case.inputs["value"], dtype=np.float32).reshape(()))
    else:
        value = 0.25

    with flag_gems.use_gems(include=["full", "full_like", "ones", "ones_like", "zeros", "zeros_like", "fill_scalar", "fill_tensor"]):
        out = torch.full((m, n), value, device=device, dtype=torch.float32)

    out_np = _to_np(out)
    value_np = np.array(value, dtype=np.float32)
    return {
        "out": out_np,
        "output": out_np,
        "value": value_np,
        "fill_value": value_np,
    }


def _run_flaggems_maximum2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["maximum", "clamp_min"],
        op_name="max",
    )


def _run_flaggems_minimum2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    return _run_flaggems_binary2d_reference(
        case,
        include=["minimum"],
        op_name="min",
    )


def _run_flaggems_identity2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "src" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["src"]), device=device)
    elif case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        # Make the input non-contiguous so contiguous/copy path is exercised.
        base = torch.from_numpy(rg.standard_normal((n, m), dtype=np.float32)).to(device)
        inp = base.transpose(0, 1)

    with flag_gems.use_gems(include=["copy", "contiguous", "resolve_conj", "resolve_neg"]):
        template = torch.empty_like(inp)
        out = flag_gems_ops.copy(template, inp)
        out = flag_gems_ops.contiguous(out)
        out = flag_gems_ops.resolve_conj(out)
        out = flag_gems_ops.resolve_neg(out)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "src": inp_np,
        "dst": out_np,
        "A": inp_np,
        "Out": out_np,
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_cast2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = torch.as_tensor(np.asarray(case.inputs["inp"]), device=device, dtype=torch.float16)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device=device, dtype=torch.float16)

    with flag_gems.use_gems(include=["to_copy"]):
        out = torch.ops.aten._to_copy(inp, dtype=torch.float32)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "A": inp_np,
        "Out": out_np,
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_mm2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 16))
    k = int(case.shapes.get("K", 32))
    n = int(case.shapes.get("N", 16))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((m, k), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((k, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["mm"]):
        c = torch.mm(a, b)

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "C": c_np,
        "mat1": a_np,
        "mat2": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_bmm3d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    batch = int(case.shapes.get("BATCH", 2))
    m = int(case.shapes.get("M", 8))
    k = int(case.shapes.get("K", 16))
    n = int(case.shapes.get("N", 8))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((batch, m, k), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((batch, k, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["bmm"]):
        c = torch.bmm(a, b)

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "C": c_np,
        "batch1": a_np,
        "batch2": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_addmm2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 16))
    k = int(case.shapes.get("K", 32))
    n = int(case.shapes.get("N", 16))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "input" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["input"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "mat1" in case.inputs:
        mat1 = _as_f32_tensor(np.asarray(case.inputs["mat1"]), device=device)
    else:
        mat1 = torch.from_numpy(rg.standard_normal((m, k), dtype=np.float32)).to(device)

    if case.inputs and "mat2" in case.inputs:
        mat2 = _as_f32_tensor(np.asarray(case.inputs["mat2"]), device=device)
    else:
        mat2 = torch.from_numpy(rg.standard_normal((k, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["addmm", "mm"]):
        out = torch.addmm(inp, mat1, mat2)

    inp_np = _to_np(inp)
    mat1_np = _to_np(mat1)
    mat2_np = _to_np(mat2)
    out_np = _to_np(out)
    return {
        "a": mat1_np,
        "b": mat2_np,
        "i": inp_np,
        "c": out_np,
        "alpha": np.array(1.0, dtype=np.float32),
        "beta": np.array(1.0, dtype=np.float32),
        "input": inp_np,
        "mat1": mat1_np,
        "mat2": mat2_np,
        "out": out_np,
        "output": out_np,
    }


def _run_flaggems_baddbmm3d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    batch = int(case.shapes.get("BATCH", 2))
    m = int(case.shapes.get("M", 8))
    k = int(case.shapes.get("K", 16))
    n = int(case.shapes.get("N", 8))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "input" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["input"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((batch, m, n), dtype=np.float32)).to(device)

    if case.inputs and "batch1" in case.inputs:
        batch1 = _as_f32_tensor(np.asarray(case.inputs["batch1"]), device=device)
    else:
        batch1 = torch.from_numpy(rg.standard_normal((batch, m, k), dtype=np.float32)).to(device)

    if case.inputs and "batch2" in case.inputs:
        batch2 = _as_f32_tensor(np.asarray(case.inputs["batch2"]), device=device)
    else:
        batch2 = torch.from_numpy(rg.standard_normal((batch, k, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["baddbmm", "bmm"]):
        out = torch.baddbmm(inp, batch1, batch2)

    inp_np = _to_np(inp)
    batch1_np = _to_np(batch1)
    batch2_np = _to_np(batch2)
    out_np = _to_np(out)
    return {
        "A": batch1_np,
        "B": batch2_np,
        "O": out_np,
        "bias": inp_np,
        "alpha": np.array(1.0, dtype=np.float32),
        "beta": np.array(1.0, dtype=np.float32),
        "input": inp_np,
        "batch1": batch1_np,
        "batch2": batch2_np,
        "out": out_np,
        "output": out_np,
    }


def _run_flaggems_dot1d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes.get("N", 256))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device).reshape(n)
    else:
        a = torch.from_numpy(rg.standard_normal((n,), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device).reshape(n)
    else:
        b = torch.from_numpy(rg.standard_normal((n,), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["dot"]):
        out = torch.dot(a, b)

    a_np = _to_np(a)
    b_np = _to_np(b)
    out_np = _to_np(out)
    return {
        "A": a_np,
        "B": b_np,
        "x": a_np,
        "y": b_np,
        "out": out_np,
        "output": out_np,
        "result": out_np,
    }


def _run_flaggems_vdot1d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes.get("N", 256))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device).reshape(n)
    else:
        a = torch.from_numpy(rg.standard_normal((n,), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device).reshape(n)
    else:
        b = torch.from_numpy(rg.standard_normal((n,), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["vdot"]):
        out = torch.vdot(a, b)

    a_np = _to_np(a)
    b_np = _to_np(b)
    out_np = _to_np(out)
    return {
        "A": a_np,
        "B": b_np,
        "inp_ptr": a_np,
        "other_ptr": b_np,
        "out_ptr": out_np,
        "out": out_np,
        "output": out_np,
        "result": out_np,
    }


def _run_flaggems_mv2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 16))
    n = int(case.shapes.get("N", 32))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        mat = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    elif case.inputs and "mat" in case.inputs:
        mat = _as_f32_tensor(np.asarray(case.inputs["mat"]), device=device)
    else:
        # Align with inferred intent conventions: A shape [N, M], B shape [M], C shape [N].
        mat = torch.from_numpy(rg.standard_normal((n, m), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        vec = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device).reshape(m)
    elif case.inputs and "vec" in case.inputs:
        vec = _as_f32_tensor(np.asarray(case.inputs["vec"]), device=device).reshape(m)
    else:
        vec = torch.from_numpy(rg.standard_normal((m,), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["mv"]):
        out = torch.mv(mat, vec)

    mat_np = _to_np(mat)
    vec_np = _to_np(vec)
    out_np = _to_np(out)
    return {
        "A": mat_np,
        "B": vec_np,
        "C": out_np,
        "mat": mat_np,
        "vec": vec_np,
        "out": out_np,
        "output": out_np,
    }


def _run_flaggems_addmv2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 16))
    n = int(case.shapes.get("N", 32))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "Inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["Inp"]), device=device).reshape(n)
    elif case.inputs and "input" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["input"]), device=device).reshape(n)
    else:
        inp = torch.from_numpy(rg.standard_normal((n,), dtype=np.float32)).to(device)

    if case.inputs and "A" in case.inputs:
        mat = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    elif case.inputs and "mat" in case.inputs:
        mat = _as_f32_tensor(np.asarray(case.inputs["mat"]), device=device)
    else:
        # Align with inferred intent conventions: A shape [N, M], B shape [M], Inp/Out shape [N].
        mat = torch.from_numpy(rg.standard_normal((n, m), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        vec = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device).reshape(m)
    elif case.inputs and "vec" in case.inputs:
        vec = _as_f32_tensor(np.asarray(case.inputs["vec"]), device=device).reshape(m)
    else:
        vec = torch.from_numpy(rg.standard_normal((m,), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["addmv", "mv"]):
        out = torch.addmv(inp, mat, vec)

    inp_np = _to_np(inp)
    mat_np = _to_np(mat)
    vec_np = _to_np(vec)
    out_np = _to_np(out)
    return {
        "A": mat_np,
        "B": vec_np,
        "Inp": inp_np,
        "Out": out_np,
        "alpha": np.array(1.0, dtype=np.float32),
        "beta": np.array(1.0, dtype=np.float32),
        "input": inp_np,
        "mat": mat_np,
        "vec": vec_np,
        "out": out_np,
        "output": out_np,
    }


def _run_flaggems_flip2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["flip"]):
        out = torch.flip(inp, dims=[1])

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "A": inp_np,
        "Out": out_np,
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_index_select2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 16))
    n = int(case.shapes.get("N", 32))
    l = int(case.shapes.get("L", 8))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "index" in case.inputs:
        index = torch.as_tensor(np.asarray(case.inputs["index"]), device=device, dtype=torch.int64).reshape(-1)
    else:
        # Most extracted intents for index_select choose row axis first.
        index = torch.from_numpy(rg.integers(0, max(1, m), size=(l,), dtype=np.int64)).to(device)

    if index.numel() == 0:
        index = torch.zeros((1,), device=device, dtype=torch.int64)

    with flag_gems.use_gems(include=["index_select"]):
        out = torch.index_select(inp, dim=0, index=index)

    inp_np = _to_np(inp)
    index_np = _to_np(index).astype(np.int32, copy=False)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "index": index_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_any_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 8))
    device = str(flag_gems.device)

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        # deterministic non-degenerate pattern
        inp = torch.zeros((m, n), device=device, dtype=torch.float32)
        for i in range(int(m)):
            if i % 2 == 1:
                inp[i, (i * 3) % max(1, int(n))] = 1.0

    with flag_gems.use_gems(include=["any", "any_dim", "any_dims"]):
        out = torch.any(inp, dim=1)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        # Common aliases for parser/runtime naming variance.
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_group_norm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes.get("N", 2))
    c = int(case.shapes.get("C", 4))
    hw = int(case.shapes.get("HW", 4))
    num_groups = int(case.shapes.get("num_groups", case.shapes.get("group", 2)))
    eps = float(case.shapes.get("eps", 1e-5))
    if c % max(1, num_groups) != 0:
        raise ValueError(f"group_norm requires C divisible by num_groups, got C={c} num_groups={num_groups}")
    group_size = c // max(1, num_groups)
    device = str(flag_gems.device)

    if case.inputs and "X" in case.inputs:
        x = _as_f32_tensor(np.asarray(case.inputs["X"]), device=device)
    else:
        x = torch.randn((n, c, hw), device=device, dtype=torch.float32)

    if case.inputs and "W" in case.inputs:
        w = _as_f32_tensor(np.asarray(case.inputs["W"]), device=device).reshape(c)
    else:
        w = torch.ones((c,), device=device, dtype=torch.float32)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device).reshape(c)
    else:
        b = torch.zeros((c,), device=device, dtype=torch.float32)

    # Use FlagGems op directly; this avoids PyTorch's guard for tiny shapes
    # (e.g., [1,1,1]) while still exercising FlagGems Triton kernels.
    with flag_gems.use_gems(include=["group_norm"]):
        y, mean, rstd = flag_gems_ops.group_norm(x, w, b, n, c, hw, num_groups, eps)

    # Keep defensive fallback for environments where ops.group_norm may return
    # implementation-dependent auxiliary tensors.
    if tuple(mean.shape) != (n, num_groups) or tuple(rstd.shape) != (n, num_groups):
        xv = x.view(n, num_groups, group_size, hw)
        mean = xv.mean(dim=(2, 3))
        var = xv.var(dim=(2, 3), unbiased=False)
        rstd = torch.rsqrt(var + eps)

    x_np = _to_np(x)
    w_np = _to_np(w)
    b_np = _to_np(b)
    y_np = _to_np(y)
    mean_np = _to_np(mean)
    rstd_np = _to_np(rstd)
    return {
        "X": x_np,
        "W": w_np,
        "B": b_np,
        "Y": y_np,
        "Mean": mean_np,
        "Rstd": rstd_np,
        # Common aliases.
        "input": x_np,
        "weight": w_np,
        "bias": b_np,
        "output": y_np,
    }


def _run_flaggems_batch_norm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes.get("N", 2))
    c = int(case.shapes.get("C", 4))
    hw = int(case.shapes.get("HW", 4))
    eps = float(case.shapes.get("eps", 1e-5))
    momentum = float(case.shapes.get("momentum", 0.1))
    training = bool(case.shapes.get("training", 1))
    device = str(flag_gems.device)

    if case.inputs and "X" in case.inputs:
        x = _as_f32_tensor(np.asarray(case.inputs["X"]), device=device)
    else:
        x = torch.randn((n, c, hw), device=device, dtype=torch.float32)

    if case.inputs and "W" in case.inputs:
        w = _as_f32_tensor(np.asarray(case.inputs["W"]), device=device).reshape(c)
    else:
        w = torch.ones((c,), device=device, dtype=torch.float32)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device).reshape(c)
    else:
        b = torch.zeros((c,), device=device, dtype=torch.float32)

    if case.inputs and "RunningMean" in case.inputs:
        running_mean = _as_f32_tensor(np.asarray(case.inputs["RunningMean"]), device=device).reshape(c)
    else:
        running_mean = torch.zeros((c,), device=device, dtype=torch.float32)

    if case.inputs and "RunningVar" in case.inputs:
        running_var = _as_f32_tensor(np.asarray(case.inputs["RunningVar"]), device=device).reshape(c)
    else:
        running_var = torch.ones((c,), device=device, dtype=torch.float32)

    with flag_gems.use_gems(include=["batch_norm"]):
        y, mean, inv_std = flag_gems_ops.batch_norm(
            x,
            w,
            b,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
        )

    x_np = _to_np(x)
    w_np = _to_np(w)
    b_np = _to_np(b)
    running_mean_np = _to_np(running_mean)
    running_var_np = _to_np(running_var)
    y_np = _to_np(y)
    mean_np = _to_np(mean)
    inv_std_np = _to_np(inv_std)
    return {
        "X": x_np,
        "W": w_np,
        "B": b_np,
        "RunningMean": running_mean_np,
        "RunningVar": running_var_np,
        "Y": y_np,
        "Mean": mean_np,
        "InvStd": inv_std_np,
        "running_mean": running_mean_np,
        "running_var": running_var_np,
        "mean": mean_np,
        "inv_std": inv_std_np,
        "output_1": y_np,
        "running_mean_out": running_mean_np,
        "running_var_out": running_var_np,
        "eps": np.array(eps, dtype=np.float32),
        "momentum": np.array(momentum, dtype=np.float32),
        "training": np.array(1 if training else 0, dtype=np.int32),
        # Common aliases.
        "input": x_np,
        "weight": w_np,
        "bias": b_np,
        "output": y_np,
    }


def _run_flaggems_layer_norm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    eps = float(case.shapes.get("eps", 1e-5))
    device = str(flag_gems.device)

    if case.inputs and "in_ptr" in case.inputs:
        x = _as_f32_tensor(np.asarray(case.inputs["in_ptr"]), device=device)
    else:
        x = torch.randn((m, n), device=device, dtype=torch.float32)

    if case.inputs and "weight_ptr" in case.inputs:
        w = _as_f32_tensor(np.asarray(case.inputs["weight_ptr"]), device=device).reshape(n)
    else:
        w = torch.ones((n,), device=device, dtype=torch.float32)

    if case.inputs and "bias_ptr" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["bias_ptr"]), device=device).reshape(n)
    else:
        b = torch.zeros((n,), device=device, dtype=torch.float32)

    with flag_gems.use_gems(include=["layer_norm"]):
        y = torch.nn.functional.layer_norm(x, (n,), weight=w, bias=b, eps=eps)

    mean = x.mean(dim=1)
    var = x.var(dim=1, unbiased=False)
    rstd = torch.rsqrt(var + eps)

    x_np = _to_np(x)
    w_np = _to_np(w)
    b_np = _to_np(b)
    y_np = _to_np(y)
    mean_np = _to_np(mean)
    rstd_np = _to_np(rstd)
    return {
        "in_ptr": x_np,
        "weight_ptr": w_np,
        "bias_ptr": b_np,
        "out_ptr": y_np,
        "out_mean_ptr": mean_np,
        "out_rstd_ptr": rstd_np,
        "eps": np.array(eps, dtype=np.float32),
        # Common aliases.
        "x": x_np,
        "w": w_np,
        "b": b_np,
        "output": y_np,
    }


def _run_flaggems_rms_norm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    eps = float(case.shapes.get("eps", 1e-5))
    device = str(flag_gems.device)

    if case.inputs and "X" in case.inputs:
        x = _as_f32_tensor(np.asarray(case.inputs["X"]), device=device)
    else:
        x = torch.randn((m, n), device=device, dtype=torch.float32)

    if case.inputs and "W" in case.inputs:
        w = _as_f32_tensor(np.asarray(case.inputs["W"]), device=device).reshape(n)
    else:
        w = torch.ones((n,), device=device, dtype=torch.float32)

    with flag_gems.use_gems(include=["rms_norm", "rms_norm_forward"]):
        y, inv_rms = flag_gems_ops.rms_norm_forward(x, (n,), w, eps)

    x_np = _to_np(x)
    w_np = _to_np(w)
    y_np = _to_np(y)
    inv_rms_np = _to_np(inv_rms)
    return {
        "X": x_np,
        "W": w_np,
        "Y": y_np,
        "InvRms": inv_rms_np,
        "eps": np.array(eps, dtype=np.float32),
        # Common aliases.
        "input": x_np,
        "weight": w_np,
        "output": y_np,
    }


def _run_flaggems_softmax_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "input_ptr" in case.inputs:
        x = _as_f32_tensor(np.asarray(case.inputs["input_ptr"]), device=device)
    else:
        x = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["softmax"]):
        y = torch.softmax(x, dim=1)

    x_np = _to_np(x)
    y_np = _to_np(y)
    return {
        "input_ptr": x_np,
        "output_ptr": y_np,
        # Common aliases for parser/runtime naming variance.
        "input": x_np,
        "output": y_np,
        "out": y_np,
    }


def _run_flaggems_relu2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["relu"]):
        out = torch.relu(inp)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_exp2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["exp"]):
        out = torch.exp(inp)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_abs2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["abs"]):
        out = torch.abs(inp)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_rsqrt2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        # keep positive values for rsqrt numerical stability
        inp = torch.from_numpy(np.abs(rg.standard_normal((m, n), dtype=np.float32)) + 1e-3).to(device)

    with flag_gems.use_gems(include=["rsqrt"]):
        out = torch.rsqrt(inp)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_lerp2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    weight_array: np.ndarray | None = None
    if case.inputs:
        for key in ("W", "weight"):
            if key in case.inputs:
                weight_array = np.asarray(case.inputs[key], dtype=np.float32)
                break

    if weight_array is not None:
        if weight_array.ndim == 0 or int(weight_array.size) == 1:
            w_scalar = float(weight_array.reshape(()))
            with flag_gems.use_gems(include=["lerp_scalar", "lerp_tensor"]):
                c = torch.lerp(a, b, w_scalar)
            w_out = np.array(w_scalar, dtype=np.float32)
        else:
            w = _as_f32_tensor(weight_array, device=device)
            with flag_gems.use_gems(include=["lerp_scalar", "lerp_tensor"]):
                c = torch.lerp(a, b, w)
            w_out = _to_np(w)
    else:
        # Keep scalar in [0, 1] to avoid unstable interpolation branches.
        w_scalar = float(rg.random())
        with flag_gems.use_gems(include=["lerp_scalar", "lerp_tensor"]):
            c = torch.lerp(a, b, w_scalar)
        w_out = np.array(w_scalar, dtype=np.float32)

    a_np = _to_np(a)
    b_np = _to_np(b)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "W": w_out,
        "C": c_np,
        # Common aliases.
        "input": a_np,
        "end": b_np,
        "weight": w_out,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_where2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "A" in case.inputs:
        a = _as_f32_tensor(np.asarray(case.inputs["A"]), device=device)
    else:
        a = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)
    if case.inputs and "B" in case.inputs:
        b = _as_f32_tensor(np.asarray(case.inputs["B"]), device=device)
    else:
        b = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["gt", "where_self", "where_scalar_self", "where_scalar_other"]):
        cond = torch.gt(a, b)
        c = torch.where(cond, a, b)

    a_np = _to_np(a)
    b_np = _to_np(b)
    cond_np = _to_np(cond)
    c_np = _to_np(c)
    return {
        "A": a_np,
        "B": b_np,
        "cond": cond_np,
        "C": c_np,
        "a": a_np,
        "b": b_np,
        "out": c_np,
        "output": c_np,
    }


def _run_flaggems_row_sum_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["sum", "sum_dim", "sum_out", "sum_dim_out"]):
        out = torch.sum(inp, dim=1)

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_row_max_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["max", "max_dim"]):
        out = torch.max(inp, dim=1).values

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_gather2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 64))
    n = int(case.shapes.get("N", 64))
    l = int(case.shapes.get("L", 256))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)
    if case.inputs and "row_idx" in case.inputs:
        row_idx = torch.as_tensor(np.asarray(case.inputs["row_idx"]), device=device, dtype=torch.int64)
    else:
        row_idx = torch.from_numpy(rg.integers(0, max(1, m), size=(l,), dtype=np.int64)).to(device)
    if case.inputs and "col_idx" in case.inputs:
        col_idx = torch.as_tensor(np.asarray(case.inputs["col_idx"]), device=device, dtype=torch.int64)
    else:
        col_idx = torch.from_numpy(rg.integers(0, max(1, n), size=(l,), dtype=np.int64)).to(device)

    with flag_gems.use_gems(include=["gather"]):
        out = inp[row_idx, col_idx]

    inp_np = _to_np(inp)
    row_idx_np = _to_np(row_idx).astype(np.int32, copy=False)
    col_idx_np = _to_np(col_idx).astype(np.int32, copy=False)
    out_np = _to_np(out)
    return {
        "inp": inp_np,
        "row_idx": row_idx_np,
        "col_idx": col_idx_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_clamp2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 4))
    n = int(case.shapes.get("N", 64))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "inp" in case.inputs:
        inp = _as_f32_tensor(np.asarray(case.inputs["inp"]), device=device)
    else:
        inp = torch.from_numpy(rg.standard_normal((m, n), dtype=np.float32)).to(device)

    if case.inputs and "lo" in case.inputs:
        lo = float(np.asarray(case.inputs["lo"], dtype=np.float32).reshape(()))
    else:
        lo = -0.5
    if case.inputs and "hi" in case.inputs:
        hi = float(np.asarray(case.inputs["hi"], dtype=np.float32).reshape(()))
    else:
        hi = 0.5
    if hi < lo:
        lo, hi = hi, lo

    with flag_gems.use_gems(include=["clamp", "maximum", "minimum"]):
        out = torch.clamp(inp, min=float(lo), max=float(hi))

    inp_np = _to_np(inp)
    out_np = _to_np(out)
    lo_np = np.array(lo, dtype=np.float32)
    hi_np = np.array(hi, dtype=np.float32)
    return {
        "inp": inp_np,
        "lo": lo_np,
        "hi": hi_np,
        "out": out_np,
        "input": inp_np,
        "output": out_np,
    }


def _run_flaggems_upsample_bicubic2d_aa_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes.get("N", 1))
    c = int(case.shapes.get("C", 1))
    ih = int(case.shapes.get("IH", 4))
    iw = int(case.shapes.get("IW", 4))
    oh = int(case.shapes.get("OH", 4))
    ow = int(case.shapes.get("OW", 4))
    device = str(flag_gems.device)
    rg = _rng(int(case.seed))

    if case.inputs and "I" in case.inputs:
        x = _as_f32_tensor(np.asarray(case.inputs["I"]), device=device)
    else:
        x = torch.from_numpy(rg.standard_normal((n, c, ih, iw), dtype=np.float32)).to(device)

    with flag_gems.use_gems(include=["_upsample_bicubic2d_aa"]):
        y = torch.ops.aten._upsample_bicubic2d_aa(x, [oh, ow], False, None, None)

    x_np = _to_np(x)
    y_np = _to_np(y)
    reciprocal_scale_h = np.array(float(ih) / float(oh), dtype=np.float32)
    reciprocal_scale_w = np.array(float(iw) / float(ow), dtype=np.float32)
    return {
        "I": x_np,
        "O": y_np,
        # Expose scales so fallback/LLM intents that model them as scalar tensors
        # can execute in the interpreter without unbound-symbol failures.
        "reciprocal_scale_h": reciprocal_scale_h,
        "reciprocal_scale_w": reciprocal_scale_w,
        # Common aliases.
        "input": x_np,
        "output": y_np,
        "out": y_np,
    }


def _norm_groupnorm(shapes: Dict[str, int]) -> Dict[str, int]:
    out = dict(shapes)
    if "C" not in out:
        return out
    c = int(out["C"])
    requested_g = int(out.get("num_groups", out.get("group", 1)))
    requested_g = max(1, min(requested_g, max(1, c)))

    divisors: List[int] = []
    d = 1
    while d * d <= max(1, c):
        if c % d == 0:
            divisors.append(d)
            if d != c // d:
                divisors.append(c // d)
        d += 1
    divisors = sorted(set(divisors))
    best_g = min(divisors, key=lambda x: (abs(x - requested_g), -x))
    out["num_groups"] = int(best_g)
    out.pop("group", None)
    out["group_size"] = c // int(best_g)
    return out


_FLAGGEMS_SPEC_BUILDERS = {
    "any_kernel_dim": lambda: KernelSpec(
        name="any_kernel_dim",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_ANY_SRC",
        runner=_run_flaggems_any_reference,
        canonical_shapes={"M": 4, "N": 8},
        vary_axes=["M", "N"],
    ),
    "add2d": lambda: KernelSpec(
        # Keep this semantic name to reuse deterministic fallback if LLM fails.
        name="add2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_ADD_SRC",
        runner=_run_flaggems_add2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "sub2d": lambda: KernelSpec(
        name="sub2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_SUB_SRC",
        runner=_run_flaggems_sub2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "mul2d": lambda: KernelSpec(
        name="mul2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_MUL_SRC",
        runner=_run_flaggems_mul2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "div2d": lambda: KernelSpec(
        name="div2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_DIV_SRC",
        runner=_run_flaggems_div2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "eq2d": lambda: KernelSpec(
        name="eq2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_EQ_SRC",
        runner=_run_flaggems_eq2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "ne2d": lambda: KernelSpec(
        name="ne2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_NE_SRC",
        runner=_run_flaggems_ne2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "gt2d": lambda: KernelSpec(
        name="gt2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_GT_SRC",
        runner=_run_flaggems_gt2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "ge2d": lambda: KernelSpec(
        name="ge2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_GE_SRC",
        runner=_run_flaggems_ge2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "lt2d": lambda: KernelSpec(
        name="lt2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LT_SRC",
        runner=_run_flaggems_lt2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "le2d": lambda: KernelSpec(
        name="le2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LE_SRC",
        runner=_run_flaggems_le2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "neg2d": lambda: KernelSpec(
        name="neg2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_NEG_SRC",
        runner=_run_flaggems_neg2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "ceil2d": lambda: KernelSpec(
        name="ceil2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_CEIL_SRC",
        runner=_run_flaggems_ceil2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "reciprocal2d": lambda: KernelSpec(
        name="reciprocal2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_RECIPROCAL_SRC",
        runner=_run_flaggems_reciprocal2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "sqrt2d": lambda: KernelSpec(
        name="sqrt2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_SQRT_SRC",
        runner=_run_flaggems_sqrt2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "sigmoid2d": lambda: KernelSpec(
        name="sigmoid2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_SIGMOID_SRC",
        runner=_run_flaggems_sigmoid2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "silu2d": lambda: KernelSpec(
        name="silu2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_SILU_SRC",
        runner=_run_flaggems_silu2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "tanh2d": lambda: KernelSpec(
        name="tanh2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_TANH_SRC",
        runner=_run_flaggems_tanh2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "mm2d": lambda: KernelSpec(
        name="mm2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_MM_SRC",
        runner=_run_flaggems_mm2d_reference,
        canonical_shapes={"M": 16, "K": 32, "N": 16},
        vary_axes=["M", "K", "N"],
    ),
    "bmm3d": lambda: KernelSpec(
        name="bmm3d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_BMM_SRC",
        runner=_run_flaggems_bmm3d_reference,
        canonical_shapes={"BATCH": 2, "M": 8, "K": 16, "N": 8},
        vary_axes=["BATCH", "M", "K", "N"],
    ),
    "addmm2d": lambda: KernelSpec(
        name="addmm2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_ADDMM_SRC",
        runner=_run_flaggems_addmm2d_reference,
        canonical_shapes={"M": 16, "K": 32, "N": 16},
        vary_axes=["M", "K", "N"],
    ),
    "baddbmm3d": lambda: KernelSpec(
        name="baddbmm3d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_BADDBMM_SRC",
        runner=_run_flaggems_baddbmm3d_reference,
        canonical_shapes={"BATCH": 2, "M": 8, "K": 16, "N": 8},
        vary_axes=["BATCH", "M", "K", "N"],
    ),
    "dot1d": lambda: KernelSpec(
        name="dot1d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_DOT_SRC",
        runner=_run_flaggems_dot1d_reference,
        canonical_shapes={"N": 256},
        vary_axes=["N"],
    ),
    "vdot1d": lambda: KernelSpec(
        name="vdot1d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_VDOT_SRC",
        runner=_run_flaggems_vdot1d_reference,
        canonical_shapes={"N": 256},
        vary_axes=["N"],
    ),
    "mv2d": lambda: KernelSpec(
        name="mv2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_MV_SRC",
        runner=_run_flaggems_mv2d_reference,
        canonical_shapes={"M": 16, "N": 32},
        vary_axes=["M", "N"],
    ),
    "addmv2d": lambda: KernelSpec(
        name="addmv2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_ADDMV_SRC",
        runner=_run_flaggems_addmv2d_reference,
        canonical_shapes={"M": 16, "N": 32},
        vary_axes=["M", "N"],
    ),
    "group_norm_kernel": lambda: KernelSpec(
        name="group_norm_kernel",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_GROUP_NORM_SRC",
        runner=_run_flaggems_group_norm_reference,
        canonical_shapes={"N": 2, "C": 4, "HW": 4, "num_groups": 2},
        vary_axes=["N", "C", "HW", "num_groups"],
        exclude_axes=["group_size"],
        normalize_shapes=_norm_groupnorm,
    ),
    "batch_norm2d": lambda: KernelSpec(
        name="batch_norm2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_BATCH_NORM_SRC",
        runner=_run_flaggems_batch_norm_reference,
        canonical_shapes={"N": 2, "C": 4, "HW": 4},
        vary_axes=["N", "C", "HW"],
    ),
    "layer_norm_persistent": lambda: KernelSpec(
        name="layer_norm_persistent",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LAYER_NORM_SRC",
        runner=_run_flaggems_layer_norm_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "rms_norm2d": lambda: KernelSpec(
        name="rms_norm2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_RMS_NORM_SRC",
        runner=_run_flaggems_rms_norm_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "softmax_inner": lambda: KernelSpec(
        # Keep this semantic name to reuse deterministic fallback if LLM fails.
        name="softmax_inner",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_SOFTMAX_SRC",
        runner=_run_flaggems_softmax_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "relu2d": lambda: KernelSpec(
        name="relu2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_RELU_SRC",
        runner=_run_flaggems_relu2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "exp2d": lambda: KernelSpec(
        name="exp2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_EXP_SRC",
        runner=_run_flaggems_exp2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "exp22d": lambda: KernelSpec(
        name="exp22d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_EXP2_SRC",
        runner=_run_flaggems_exp22d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "abs2d": lambda: KernelSpec(
        name="abs2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_ABS_SRC",
        runner=_run_flaggems_abs2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "rsqrt2d": lambda: KernelSpec(
        name="rsqrt2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_RSQRT_SRC",
        runner=_run_flaggems_rsqrt2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "where2d": lambda: KernelSpec(
        name="where2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_WHERE_SRC",
        runner=_run_flaggems_where2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        # Keep fixed-shape for now; some FlagGems where variants can be slow on
        # large auto-generated edge cases during Stage7 diff.
        vary_axes=[],
        # where2d bounded exhaustive space is combinatorial (3^18). Cap Stage C.
        stage_c_max_cases=64,
        # Stage-B diff is the gate for first-pass coverage; skip mutation kill.
        enable_mutation_kill=False,
    ),
    "row_sum": lambda: KernelSpec(
        name="row_sum",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_SUM_SRC",
        runner=_run_flaggems_row_sum_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "row_max": lambda: KernelSpec(
        name="row_max",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_MAX_SRC",
        runner=_run_flaggems_row_max_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "row_mean": lambda: KernelSpec(
        name="row_mean",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_MEAN_SRC",
        runner=_run_flaggems_row_mean_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "row_all": lambda: KernelSpec(
        name="row_all",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_ALL_SRC",
        runner=_run_flaggems_row_all_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "logical_and2d": lambda: KernelSpec(
        name="logical_and2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LOGICAL_AND_SRC",
        runner=_run_flaggems_logical_and2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "logical_or2d": lambda: KernelSpec(
        name="logical_or2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LOGICAL_OR_SRC",
        runner=_run_flaggems_logical_or2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "logical_not2d": lambda: KernelSpec(
        name="logical_not2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LOGICAL_NOT_SRC",
        runner=_run_flaggems_logical_not2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "logical_xor2d": lambda: KernelSpec(
        name="logical_xor2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LOGICAL_XOR_SRC",
        runner=_run_flaggems_logical_xor2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "maximum2d": lambda: KernelSpec(
        name="maximum2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_MAXIMUM_SRC",
        runner=_run_flaggems_maximum2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "minimum2d": lambda: KernelSpec(
        name="minimum2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_MINIMUM_SRC",
        runner=_run_flaggems_minimum2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "full2d": lambda: KernelSpec(
        name="full2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_FULL_SRC",
        runner=_run_flaggems_full2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "identity2d": lambda: KernelSpec(
        name="identity2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_COPY_SRC",
        runner=_run_flaggems_identity2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "cast2d": lambda: KernelSpec(
        name="cast2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_TO_COPY_SRC",
        runner=_run_flaggems_cast2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "flip2d": lambda: KernelSpec(
        name="flip2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_FLIP_SRC",
        runner=_run_flaggems_flip2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "index_select2d": lambda: KernelSpec(
        name="index_select2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_INDEX_SELECT_SRC",
        runner=_run_flaggems_index_select2d_reference,
        canonical_shapes={"M": 16, "N": 32, "L": 8},
        vary_axes=["M", "N", "L"],
    ),
    "gather2d": lambda: KernelSpec(
        name="gather2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_GATHER_SRC",
        runner=_run_flaggems_gather2d_reference,
        canonical_shapes={"M": 64, "N": 64, "L": 256},
        vary_axes=["M", "N", "L"],
    ),
    "clamp2d": lambda: KernelSpec(
        name="clamp2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_CLAMP_SRC",
        runner=_run_flaggems_clamp2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "lerp2d": lambda: KernelSpec(
        name="lerp2d",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LERP_SRC",
        runner=_run_flaggems_lerp2d_reference,
        canonical_shapes={"M": 4, "N": 64},
        vary_axes=["M", "N"],
    ),
    "upsample_bicubic2d_aa": lambda: KernelSpec(
        name="upsample_bicubic2d_aa",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_UPSAMPLE_BICUBIC2D_AA_SRC",
        runner=_run_flaggems_upsample_bicubic2d_aa_reference,
        canonical_shapes={"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 4, "OW": 4},
        vary_axes=["IH", "IW", "OH", "OW"],
    ),
}

_FLAGGEMS_SMOKE_ORDER = [
    "any_kernel_dim",
    "add2d",
    "group_norm_kernel",
    "layer_norm_persistent",
    "softmax_inner",
    "upsample_bicubic2d_aa",
]


def _capability_state_for_target(entry: Dict[str, Any], backend_target: str | None) -> str:
    status = str(entry.get("status") or "blocked_ir")
    if backend_target is None:
        return status
    support = entry.get("backend_support") or {}
    if str(backend_target) == "rvv":
        return "supported" if bool(((support.get("rvv") or {}).get("ok"))) else "blocked_backend"
    if str(backend_target) in {"cuda_h100", "cuda_5090d"}:
        return "supported" if bool(((support.get(str(backend_target)) or {}).get("ok"))) else "blocked_backend"
    return status


def _attach_registry_meta(spec: KernelSpec, *, entry: Dict[str, Any] | None, backend_target: str | None) -> KernelSpec:
    setattr(spec, "provider", "flaggems")
    setattr(spec, "backend_target", (str(backend_target) if backend_target else None))
    if entry is None:
        setattr(spec, "source_op", spec.name)
        setattr(spec, "capability_state", "blocked_ir")
        setattr(spec, "capability_status", "blocked_ir")
        setattr(spec, "intent_ops", [])
        return spec

    semantic_op = str(entry.get("semantic_op") or spec.name)
    status = str(entry.get("status") or "blocked_ir")
    setattr(spec, "source_op", semantic_op)
    setattr(spec, "capability_state", _capability_state_for_target(entry, backend_target))
    setattr(spec, "capability_status", status)
    setattr(spec, "intent_ops", list(entry.get("intent_ops") or []))
    setattr(spec, "status_reason", str(entry.get("status_reason") or ""))
    setattr(spec, "family", str(entry.get("family") or ""))
    return spec


def _build_specs(
    spec_names: List[str],
    *,
    registry: Dict[str, Any],
    backend_target: str | None,
) -> List[KernelSpec]:
    out: List[KernelSpec] = []
    for name in spec_names:
        build = _FLAGGEMS_SPEC_BUILDERS.get(str(name))
        if build is None:
            continue
        spec = build()
        entry = load_registry_entry_by_spec(registry, spec.name)
        spec = _attach_registry_meta(spec, entry=entry, backend_target=backend_target)
        out.append(spec)
    return out


def default_flaggems_kernel_specs(
    *,
    flaggems_opset: str = DEFAULT_FLAGGEMS_OPSET,
    backend_target: str | None = None,
) -> List[KernelSpec]:
    # Registry is the source of truth for semantic coverage; the smoke suite
    # remains a stable ordered subset for fast verification.
    registry = load_registry()
    if str(flaggems_opset) != DEFAULT_FLAGGEMS_OPSET:
        raise ValueError(f"unsupported flaggems opset: {flaggems_opset}")
    return _build_specs(list(_FLAGGEMS_SMOKE_ORDER), registry=registry, backend_target=backend_target)


def coverage_flaggems_kernel_specs(
    *,
    flaggems_opset: str = DEFAULT_FLAGGEMS_OPSET,
    backend_target: str | None = None,
) -> List[KernelSpec]:
    registry = load_registry()
    if str(flaggems_opset) != DEFAULT_FLAGGEMS_OPSET:
        raise ValueError(f"unsupported flaggems opset: {flaggems_opset}")
    spec_names = [name for name in list_supported_e2e_specs(registry) if name in _FLAGGEMS_SPEC_BUILDERS]
    if not spec_names:
        spec_names = list(_FLAGGEMS_SMOKE_ORDER)
    return _build_specs(spec_names, registry=registry, backend_target=backend_target)


__all__ = [
    "FLAGGEMS_ANY_SRC",
    "FLAGGEMS_ADD_SRC",
    "FLAGGEMS_GROUP_NORM_SRC",
    "FLAGGEMS_BATCH_NORM_SRC",
    "FLAGGEMS_LAYER_NORM_SRC",
    "FLAGGEMS_RMS_NORM_SRC",
    "FLAGGEMS_LERP_SRC",
    "FLAGGEMS_SOFTMAX_SRC",
    "FLAGGEMS_RELU_SRC",
    "FLAGGEMS_EXP_SRC",
    "FLAGGEMS_WHERE_SRC",
    "FLAGGEMS_SUM_SRC",
    "FLAGGEMS_MAX_SRC",
    "FLAGGEMS_CLAMP_SRC",
    "FLAGGEMS_UPSAMPLE_BICUBIC2D_AA_SRC",
    "default_flaggems_kernel_specs",
    "coverage_flaggems_kernel_specs",
]
