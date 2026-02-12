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
from typing import Dict, List

import numpy as np
import torch

from pipeline.triton.core import KernelSpec
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


def default_flaggems_kernel_specs() -> List[KernelSpec]:
    return [
        KernelSpec(
            name="any_kernel_dim",
            module="pipeline.triton.flaggems_specs",
            attr="FLAGGEMS_ANY_SRC",
            runner=_run_flaggems_any_reference,
            canonical_shapes={"M": 4, "N": 8},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            # Keep this semantic name to reuse deterministic fallback if LLM fails.
            name="add2d",
            module="pipeline.triton.flaggems_specs",
            attr="FLAGGEMS_ADD_SRC",
            runner=_run_flaggems_add2d_reference,
            canonical_shapes={"M": 4, "N": 64},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="group_norm_kernel",
            module="pipeline.triton.flaggems_specs",
            attr="FLAGGEMS_GROUP_NORM_SRC",
            runner=_run_flaggems_group_norm_reference,
            canonical_shapes={"N": 2, "C": 4, "HW": 4, "num_groups": 2},
            vary_axes=["N", "C", "HW", "num_groups"],
            exclude_axes=["group_size"],
            normalize_shapes=_norm_groupnorm,
        ),
        KernelSpec(
            name="layer_norm_persistent",
            module="pipeline.triton.flaggems_specs",
            attr="FLAGGEMS_LAYER_NORM_SRC",
            runner=_run_flaggems_layer_norm_reference,
            canonical_shapes={"M": 4, "N": 64},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            # Keep this semantic name to reuse deterministic fallback if LLM fails.
            name="softmax_inner",
            module="pipeline.triton.flaggems_specs",
            attr="FLAGGEMS_SOFTMAX_SRC",
            runner=_run_flaggems_softmax_reference,
            canonical_shapes={"M": 4, "N": 64},
            vary_axes=["M", "N"],
        ),
        KernelSpec(
            name="upsample_bicubic2d_aa",
            module="pipeline.triton.flaggems_specs",
            attr="FLAGGEMS_UPSAMPLE_BICUBIC2D_AA_SRC",
            runner=_run_flaggems_upsample_bicubic2d_aa_reference,
            canonical_shapes={"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 4, "OW": 4},
            vary_axes=["IH", "IW", "OH", "OW"],
        ),
    ]


def coverage_flaggems_kernel_specs() -> List[KernelSpec]:
    # Placeholder for future expansion (layernorm/dropout/attention subsets).
    return list(default_flaggems_kernel_specs())


__all__ = [
    "FLAGGEMS_ANY_SRC",
    "FLAGGEMS_ADD_SRC",
    "FLAGGEMS_GROUP_NORM_SRC",
    "FLAGGEMS_LAYER_NORM_SRC",
    "FLAGGEMS_SOFTMAX_SRC",
    "FLAGGEMS_UPSAMPLE_BICUBIC2D_AA_SRC",
    "default_flaggems_kernel_specs",
    "coverage_flaggems_kernel_specs",
]
