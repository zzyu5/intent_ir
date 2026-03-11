from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pipeline.triton.core import KernelSpec
from verify.gen_cases import TestCase


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
LIGER_SRC_ROOT = WORKSPACE_ROOT / "Liger-Kernel" / "src"
if LIGER_SRC_ROOT.is_dir():
    liger_src = str(LIGER_SRC_ROOT)
    if liger_src not in sys.path:
        sys.path.insert(0, liger_src)

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel, cross_entropy_forward  # noqa: E402
from liger_kernel.ops.geglu import _geglu_tanh_forward_kernel, geglu_forward  # noqa: E402
from liger_kernel.ops.fused_add_rms_norm import (  # noqa: E402
    _fused_add_rms_norm_forward_kernel,
    fused_add_rms_norm_forward,
)
from liger_kernel.ops.layer_norm import _layer_norm_forward_kernel, layer_norm_forward  # noqa: E402
from liger_kernel.ops.rms_norm import _rms_norm_forward_kernel, rms_norm_forward  # noqa: E402
from liger_kernel.ops.rope import _triton_rope, rope_forward  # noqa: E402
from liger_kernel.ops.softmax import _softmax_single_block_forward_kernel, _softmax_forward  # noqa: E402
from liger_kernel.ops.swiglu import _swiglu_forward_kernel, swiglu_forward  # noqa: E402


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _torch_randn(shape: tuple[int, ...], *, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(int(seed))
    return torch.randn(shape, device="cuda", dtype=dtype, generator=g)


def _torch_randint(high: int, shape: tuple[int, ...], *, seed: int, dtype: torch.dtype = torch.int64) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(int(seed))
    return torch.randint(0, int(high), shape, device="cuda", dtype=dtype, generator=g)


def _swiglu_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    a = _torch_randn((m, n), seed=int(case.seed) + 1)
    b = _torch_randn((m, n), seed=int(case.seed) + 2)
    a_in = a.clone()
    b_in = b.clone()
    _a, _b, c = swiglu_forward(a, b)
    torch.cuda.synchronize()
    return {"a": _to_np(a_in), "b": _to_np(b_in), "c": _to_np(c)}


def _rms_norm_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    x = _torch_randn((m, n), seed=int(case.seed) + 3)
    w = _torch_randn((n,), seed=int(case.seed) + 4)
    x_in = x.clone()
    w_in = w.clone()
    y, _x2, rstd, _block_size, _num_warps, _casting_mode = rms_norm_forward(
        x,
        w,
        1.0e-5,
        0.0,
        "none",
        False,
    )
    torch.cuda.synchronize()
    return {
        "X": _to_np(x_in),
        "W": _to_np(w_in),
        "eps": np.array(1.0e-5, dtype=np.float32),
        "offset": np.array(0.0, dtype=np.float32),
        "Y": _to_np(y),
        "RSTD": _to_np(rstd),
    }


def _fused_add_rms_norm_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    x = _torch_randn((m, n), seed=int(case.seed) + 5)
    r = _torch_randn((m, n), seed=int(case.seed) + 6)
    w = _torch_randn((n,), seed=int(case.seed) + 7)
    x_in = x.clone()
    r_in = r.clone()
    w_in = w.clone()
    y, s, rstd, _block_size, _num_warps, _casting_mode = fused_add_rms_norm_forward(
        x,
        r,
        w,
        1.0e-5,
        0.0,
        "none",
    )
    torch.cuda.synchronize()
    return {
        "X": _to_np(x_in),
        "R": _to_np(r_in),
        "W": _to_np(w_in),
        "eps": np.array(1.0e-5, dtype=np.float32),
        "offset": np.array(0.0, dtype=np.float32),
        "Y": _to_np(y),
        "S": _to_np(s),
        "RSTD": _to_np(rstd),
    }


def _rope_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    qh = int(case.shapes["QH"])
    kh = int(case.shapes["KH"])
    s = int(case.shapes["S"])
    hd = int(case.shapes["HD"])
    q = _torch_randn((b, qh, s, hd), seed=int(case.seed) + 8)
    k = _torch_randn((b, kh, s, hd), seed=int(case.seed) + 9)
    cos = _torch_randn((1, s, hd), seed=int(case.seed) + 10)
    sin = _torch_randn((1, s, hd), seed=int(case.seed) + 11)
    q_in = q.clone()
    k_in = k.clone()
    cos_in = cos.clone()
    sin_in = sin.clone()
    q_out, k_out, _cos_out, _sin_out = rope_forward(q, k, cos, sin)
    torch.cuda.synchronize()
    return {
        "q": _to_np(q_in),
        "k": _to_np(k_in),
        "cos": _to_np(cos_in),
        "sin": _to_np(sin_in),
        "q_out": _to_np(q_out),
        "k_out": _to_np(k_out),
    }


def _cross_entropy_runner(case: TestCase) -> Dict[str, np.ndarray]:
    bt = int(case.shapes["BT"])
    v = int(case.shapes["V"])
    x = _torch_randn((bt, v), seed=int(case.seed) + 12)
    target = _torch_randint(v, (bt,), seed=int(case.seed) + 13)
    x_in = x.clone()
    target_in = target.clone()
    loss, _z_loss, _token_accuracy, _predicted_tokens, x_out = cross_entropy_forward(
        x,
        target,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        return_token_accuracy=False,
        return_predicted_tokens=False,
    )
    torch.cuda.synchronize()
    return {
        "input": _to_np(x_in),
        "target": _to_np(target_in),
        "ignore_index": np.array(-100, dtype=np.int64),
        "loss": _to_np(loss),
        "input_after": _to_np(x_out),
    }


def _geglu_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    a = _torch_randn((m, n), seed=int(case.seed) + 14)
    b = _torch_randn((m, n), seed=int(case.seed) + 15)
    a_in = a.clone()
    b_in = b.clone()
    _a, _b, c = geglu_forward(a, b)
    torch.cuda.synchronize()
    return {"a": _to_np(a_in), "b": _to_np(b_in), "c": _to_np(c)}


def _layer_norm_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    x = _torch_randn((m, n), seed=int(case.seed) + 16)
    w = _torch_randn((n,), seed=int(case.seed) + 17)
    b = _torch_randn((n,), seed=int(case.seed) + 18)
    x_in = x.clone()
    w_in = w.clone()
    b_in = b.clone()
    y, _x2, mean, rstd, _block_size, _num_warps = layer_norm_forward(x, w, b, 1.0e-5)
    torch.cuda.synchronize()
    return {
        "X": _to_np(x_in),
        "W": _to_np(w_in),
        "B": _to_np(b_in),
        "eps": np.array(1.0e-5, dtype=np.float32),
        "Y": _to_np(y),
        "Mean": _to_np(mean),
        "RSTD": _to_np(rstd),
    }


def _softmax_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    x = _torch_randn((m, n), seed=int(case.seed) + 19)
    x_in = x.clone()
    y, _block_size, _num_warps, _multi = _softmax_forward(x)
    torch.cuda.synchronize()
    return {
        "X": _to_np(x_in),
        "Y": _to_np(y),
    }


def liger_kernel_specs() -> List[KernelSpec]:
    module = "pipeline.triton.providers.liger.specs"
    return [
        KernelSpec(
            name="liger_swiglu",
            module=module,
            attr="_swiglu_forward_kernel.src",
            runner=_swiglu_runner,
            canonical_shapes={"M": 65536, "N": 256},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_rms_norm",
            module=module,
            attr="_rms_norm_forward_kernel.src",
            runner=_rms_norm_runner,
            canonical_shapes={"M": 2048, "N": 32768},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_fused_add_rms_norm",
            module=module,
            attr="_fused_add_rms_norm_forward_kernel.src",
            runner=_fused_add_rms_norm_runner,
            canonical_shapes={"M": 2048, "N": 32768},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_rope",
            module=module,
            attr="_triton_rope.src",
            runner=_rope_runner,
            canonical_shapes={"B": 2, "QH": 32, "KH": 8, "S": 2048, "HD": 128},
            vary_axes=["B", "QH", "KH", "S", "HD"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_cross_entropy",
            module=module,
            attr="liger_cross_entropy_kernel.src",
            runner=_cross_entropy_runner,
            canonical_shapes={"BT": 2048, "V": 4096},
            vary_axes=["BT", "V"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_geglu",
            module=module,
            attr="_geglu_tanh_forward_kernel.src",
            runner=_geglu_runner,
            canonical_shapes={"M": 65536, "N": 256},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_layer_norm",
            module=module,
            attr="_layer_norm_forward_kernel.src",
            runner=_layer_norm_runner,
            canonical_shapes={"M": 2048, "N": 4096},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_softmax",
            module=module,
            attr="_softmax_single_block_forward_kernel.src",
            runner=_softmax_runner,
            canonical_shapes={"M": 2048, "N": 4096},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
    ]


__all__ = ["liger_kernel_specs"]
