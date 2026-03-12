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
from liger_kernel.ops.dyt import _dyt_fwd_kernel, liger_dyt_fwd  # noqa: E402
from liger_kernel.ops.geglu import _geglu_tanh_forward_kernel, geglu_forward  # noqa: E402
from liger_kernel.ops.fused_add_rms_norm import (  # noqa: E402
    _fused_add_rms_norm_forward_kernel,
    fused_add_rms_norm_forward,
)
from liger_kernel.ops.group_norm import _group_norm_forward_kernel, group_norm_forward  # noqa: E402
from liger_kernel.ops.jsd import _jsd_kernel, jsd_forward  # noqa: E402
from liger_kernel.ops.kl_div import _kldiv_kernel_forward, kldiv_forward_triton  # noqa: E402
from liger_kernel.ops.layer_norm import _layer_norm_forward_kernel, layer_norm_forward  # noqa: E402
from liger_kernel.ops.qwen2vl_mrope import _triton_qwen2vl_mrope, qwen2vl_mrope_forward  # noqa: E402
from liger_kernel.ops.rms_norm import _rms_norm_forward_kernel, rms_norm_forward  # noqa: E402
from liger_kernel.ops.rope import _triton_rope, rope_forward  # noqa: E402
from liger_kernel.ops.softmax import _softmax_single_block_forward_kernel, _softmax_forward  # noqa: E402
from liger_kernel.ops.sparsemax import _sparsemax_forward, _sparsemax_forward_kernel  # noqa: E402
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


def _group_norm_runner(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes["N"])
    c = int(case.shapes["C"])
    hw = int(case.shapes["HW"])
    num_groups = int(case.shapes["num_groups"])
    x = _torch_randn((n, c, hw), seed=int(case.seed) + 20)
    w = _torch_randn((c,), seed=int(case.seed) + 21)
    b = _torch_randn((c,), seed=int(case.seed) + 22)
    x_in = x.clone()
    w_in = w.clone()
    b_in = b.clone()
    y, _x2, mean, rstd, _block_size = group_norm_forward(
        x,
        c,
        num_groups,
        w,
        b,
        1.0e-5,
    )
    torch.cuda.synchronize()
    rstd_np = _to_np(rstd)
    return {
        "X": _to_np(x_in),
        "W": _to_np(w_in),
        "B": _to_np(b_in),
        "eps": np.array(1.0e-5, dtype=np.float32),
        "num_groups": np.array(num_groups, dtype=np.int32),
        "Y": _to_np(y),
        "Mean": _to_np(mean),
        "Rstd": rstd_np,
        "RSTD": rstd_np,
    }


def _dyt_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    x = _torch_randn((m, n), seed=int(case.seed) + 23)
    alpha = _torch_randn((1,), seed=int(case.seed) + 24)
    gamma = _torch_randn((n,), seed=int(case.seed) + 25)
    beta = _torch_randn((n,), seed=int(case.seed) + 26)
    x_in = x.clone()
    alpha_in = alpha.clone()
    gamma_in = gamma.clone()
    beta_in = beta.clone()
    y = liger_dyt_fwd(x, alpha, gamma, beta)
    torch.cuda.synchronize()
    return {
        "X": _to_np(x_in),
        "Alpha": _to_np(alpha_in.reshape(())),
        "Gamma": _to_np(gamma_in),
        "Beta": _to_np(beta_in),
        "Y": _to_np(y),
    }


def _qwen2vl_mrope_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    qh = int(case.shapes["QH"])
    kh = int(case.shapes["KH"])
    s = int(case.shapes["S"])
    hd = int(case.shapes["HD"])
    q = _torch_randn((b, qh, s, hd), seed=int(case.seed) + 27)
    k = _torch_randn((b, kh, s, hd), seed=int(case.seed) + 28)
    cos = _torch_randn((3, b, s, hd), seed=int(case.seed) + 29)
    sin = _torch_randn((3, b, s, hd), seed=int(case.seed) + 30)
    q_in = q.clone()
    k_in = k.clone()
    cos_in = cos.clone()
    sin_in = sin.clone()
    section_t = max(1, hd // 8)
    section_h = max(1, hd // 8)
    q_out, k_out, _cos_out, _sin_out = qwen2vl_mrope_forward(q, k, cos, sin, [section_t, section_h])
    torch.cuda.synchronize()
    q_phys_in = q_in.transpose(1, 2).contiguous()
    k_phys_in = k_in.transpose(1, 2).contiguous()
    q_phys_out = q_out.transpose(1, 2).contiguous()
    k_phys_out = k_out.transpose(1, 2).contiguous()
    return {
        "q": _to_np(q_in),
        "k": _to_np(k_in),
        "Q": _to_np(q_phys_in),
        "K": _to_np(k_phys_in),
        "cos": _to_np(cos_in),
        "sin": _to_np(sin_in),
        "cos_t": _to_np(cos_in[0]),
        "cos_h": _to_np(cos_in[1]),
        "cos_w": _to_np(cos_in[2]),
        "sin_t": _to_np(sin_in[0]),
        "sin_h": _to_np(sin_in[1]),
        "sin_w": _to_np(sin_in[2]),
        "mrope_section_t": np.array(section_t, dtype=np.int32),
        "mrope_section_h": np.array(section_h, dtype=np.int32),
        "q_out": _to_np(q_out),
        "k_out": _to_np(k_out),
        "Q_out": _to_np(q_phys_out),
        "K_out": _to_np(k_phys_out),
        "mrope_section": np.asarray([section_t, section_h], dtype=np.int32),
    }


def _sparsemax_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    x = _torch_randn((m, n), seed=int(case.seed) + 31)
    x_in = x.clone()
    y, _flat = _sparsemax_forward(x, -1)
    torch.cuda.synchronize()
    return {
        "X": _to_np(x_in),
        "Y": _to_np(y),
    }


def _kl_div_runner(case: TestCase) -> Dict[str, np.ndarray]:
    bt = int(case.shapes["BT"])
    v = int(case.shapes["V"])
    logits = _torch_randn((bt, v), seed=int(case.seed) + 32)
    target_logits = _torch_randn((bt, v), seed=int(case.seed) + 33)
    y_pred = torch.log_softmax(logits, dim=-1)
    y_true = torch.softmax(target_logits, dim=-1)
    y_pred_in = y_pred.clone()
    y_true_in = y_true.clone()
    loss = kldiv_forward_triton(y_pred, y_true, False, "batchmean", 1.0e-10)
    torch.cuda.synchronize()
    loss_row = torch.sum(y_true_in * (torch.log(torch.clamp_min(y_true_in, 1.0e-10)) - y_pred_in), dim=-1)
    return {
        "input": _to_np(y_pred_in),
        "target": _to_np(y_true_in),
        "y": _to_np(y_pred_in),
        "gt": _to_np(y_true_in),
        "eps": np.array(1.0e-10, dtype=np.float32),
        "log_target": np.array(0, dtype=np.int32),
        "reduction": np.array(3, dtype=np.int32),
        "loss": _to_np(loss_row),
        "loss_scalar": _to_np(loss),
    }


def _jsd_runner(case: TestCase) -> Dict[str, np.ndarray]:
    bt = int(case.shapes["BT"])
    v = int(case.shapes["V"])
    input_logits = _torch_randn((bt, v), seed=int(case.seed) + 34)
    target_logits = _torch_randn((bt, v), seed=int(case.seed) + 35)
    shift_labels = _torch_randint(v, (bt,), seed=int(case.seed) + 36)
    x = torch.log_softmax(input_logits, dim=-1)
    y = torch.log_softmax(target_logits, dim=-1)
    x_in = x.clone()
    y_in = y.clone()
    labels_in = shift_labels.clone()
    loss, d_x = jsd_forward(x, y, shift_labels, 0.5, -100, True)
    torch.cuda.synchronize()
    n_non_ignore = int((shift_labels != -100).sum().item())
    y_max = torch.max(y_in, dim=-1, keepdim=True).values
    x_max = torch.max(x_in, dim=-1, keepdim=True).values
    max_val = torch.maximum(x_max, y_max)
    q = torch.exp(x_in - max_val) * torch.exp(max_val)
    p = torch.exp(y_in - max_val) * torch.exp(max_val)
    beta = 0.5
    beta_p = beta * p
    one_minus_beta_q = (1.0 - beta) * q
    m = beta_p + one_minus_beta_q
    log_m = torch.log(m)
    loss_row = beta_p * y_in + one_minus_beta_q * x_in - m * log_m
    if n_non_ignore > 0:
        loss_row = loss_row / float(n_non_ignore)
    ignore_mask = (labels_in == -100).unsqueeze(-1)
    loss_row = torch.where(ignore_mask, torch.zeros_like(loss_row), loss_row)
    return {
        "input": _to_np(x_in),
        "target": _to_np(y_in),
        "X": _to_np(x_in),
        "Y": _to_np(y_in),
        "label": _to_np(labels_in),
        "shift_labels": _to_np(labels_in),
        "beta": np.array(0.5, dtype=np.float32),
        "n_non_ignore": np.array(n_non_ignore, dtype=np.int32),
        "ignore_index": np.array(-100, dtype=np.int64),
        "loss": _to_np(loss_row),
        "loss_scalar": _to_np(loss),
        "dX": _to_np(d_x),
    }


def _norm_group_norm_shapes(shapes: Dict[str, int]) -> Dict[str, int]:
    out = {str(k): int(v) for k, v in dict(shapes or {}).items()}
    n = int(out.get("N", 0))
    c = int(out.get("C", 0))
    requested_g = int(out.get("num_groups", out.get("group", 1)))
    hw = int(out.get("HW", 0))
    if n <= 0 or c <= 0 or hw <= 0:
        return out
    divisors = [g for g in range(1, c + 1) if c % g == 0]
    if not divisors:
        return out
    best_g = min(sorted(set(divisors)), key=lambda x: (abs(x - requested_g), -x))
    out["num_groups"] = int(best_g)
    out.pop("group", None)
    out["group_size"] = int(c // int(best_g))
    out["channels_per_group"] = int(out["group_size"])
    out["hidden_size_per_channel"] = int(hw)
    out["hidden_size"] = int(out["group_size"] * hw)
    return out


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
        KernelSpec(
            name="liger_group_norm",
            module=module,
            attr="_group_norm_forward_kernel.src",
            runner=_group_norm_runner,
            canonical_shapes={"N": 32, "C": 512, "HW": 64, "num_groups": 32},
            vary_axes=["N", "C", "HW", "num_groups"],
            exclude_axes=["group_size", "channels_per_group", "hidden_size_per_channel", "hidden_size"],
            normalize_shapes=_norm_group_norm_shapes,
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_dyt",
            module=module,
            attr="_dyt_fwd_kernel.src",
            runner=_dyt_runner,
            canonical_shapes={"M": 2048, "N": 4096},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_qwen2vl_mrope",
            module=module,
            attr="_triton_qwen2vl_mrope.src",
            runner=_qwen2vl_mrope_runner,
            canonical_shapes={"B": 2, "QH": 32, "KH": 8, "S": 2048, "HD": 128},
            vary_axes=["B", "QH", "KH", "S", "HD"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_sparsemax",
            module=module,
            attr="_sparsemax_forward_kernel.src",
            runner=_sparsemax_runner,
            canonical_shapes={"M": 2048, "N": 4096},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_kl_div",
            module=module,
            attr="_kldiv_kernel_forward.src",
            runner=_kl_div_runner,
            canonical_shapes={"BT": 2048, "V": 4096},
            vary_axes=["BT", "V"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_jsd",
            module=module,
            attr="_jsd_kernel.src",
            runner=_jsd_runner,
            canonical_shapes={"BT": 2048, "V": 4096},
            vary_axes=["BT", "V"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
    ]


__all__ = ["liger_kernel_specs"]
