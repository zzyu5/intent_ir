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
FLAGGEMS_GROUP_NORM_SRC = _module_source_text("flag_gems.ops.groupnorm")
FLAGGEMS_LAYER_NORM_SRC = _module_source_text("flag_gems.ops.layernorm")
FLAGGEMS_SOFTMAX_SRC = _module_source_text("flag_gems.ops.softmax")
FLAGGEMS_RELU_SRC = _module_source_text("flag_gems.ops.relu")
FLAGGEMS_EXP_SRC = _module_source_text("flag_gems.ops.exp")
FLAGGEMS_GATHER_SRC = _module_source_text("flag_gems.ops.gather")
FLAGGEMS_WHERE_SRC = _module_source_text("flag_gems.ops.where")
FLAGGEMS_SUM_SRC = _module_source_text("flag_gems.ops.sum")
FLAGGEMS_MAX_SRC = _module_source_text("flag_gems.ops.max")
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
    "layer_norm_persistent": lambda: KernelSpec(
        name="layer_norm_persistent",
        module="pipeline.triton.flaggems_specs",
        attr="FLAGGEMS_LAYER_NORM_SRC",
        runner=_run_flaggems_layer_norm_reference,
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
    "FLAGGEMS_LAYER_NORM_SRC",
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
