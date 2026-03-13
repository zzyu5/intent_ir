from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

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
from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward  # noqa: E402
from liger_kernel.ops.fused_linear_jsd import fused_linear_jsd_forward  # noqa: E402
from liger_kernel.ops.geglu import _geglu_tanh_forward_kernel, geglu_forward  # noqa: E402
from liger_kernel.ops.fused_add_rms_norm import (  # noqa: E402
    _fused_add_rms_norm_forward_kernel,
    fused_add_rms_norm_forward,
)
from liger_kernel.ops.fused_neighborhood_attention import (  # noqa: E402
    _fused_neighborhood_attention_qk_kernel,
    fused_neighborhood_attention_forward,
)
from liger_kernel.ops.group_norm import _group_norm_forward_kernel, group_norm_forward  # noqa: E402
from liger_kernel.ops.jsd import _jsd_kernel, jsd_forward  # noqa: E402
from liger_kernel.ops.kl_div import _kldiv_kernel_forward, kldiv_forward_triton  # noqa: E402
from liger_kernel.ops.layer_norm import _layer_norm_forward_kernel, layer_norm_forward  # noqa: E402
from liger_kernel.ops.llama4_rope import _llama4_rope_kernel, llama4_rope_forward  # noqa: E402
from liger_kernel.ops.mhc import _mhc_mm_norm_fwd_kernel, mhc_mm_norm_fwd  # noqa: E402
from liger_kernel.ops.poly_norm import _poly_norm_forward_kernel, poly_norm_forward  # noqa: E402
from liger_kernel.ops.qwen2vl_mrope import _triton_qwen2vl_mrope, qwen2vl_mrope_forward  # noqa: E402
from liger_kernel.ops.rms_norm import _rms_norm_forward_kernel, rms_norm_forward  # noqa: E402
from liger_kernel.ops.rope import _triton_rope, rope_forward  # noqa: E402
from liger_kernel.ops.softmax import _softmax_single_block_forward_kernel, _softmax_forward  # noqa: E402
from liger_kernel.ops.sparsemax import _sparsemax_forward, _sparsemax_forward_kernel  # noqa: E402
from liger_kernel.ops.swiglu import _swiglu_forward_kernel, swiglu_forward  # noqa: E402
from liger_kernel.ops.tvd import _tv_distance_kernel, tv_distance_forward_triton  # noqa: E402
from liger_kernel.transformers.functional import (  # noqa: E402
    liger_fused_linear_cross_entropy,
    liger_fused_linear_jsd,
    liger_fused_neighborhood_attention,
    liger_mhc_forward,
    liger_multi_token_attention,
    liger_poly_norm,
    liger_tvd,
)
from liger_kernel.transformers.grpo_loss import triton_grpo_loss  # noqa: E402
from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP  # noqa: E402


class _LazyModuleSource:
    def __init__(self, mod_name: str):
        self._mod_name = str(mod_name)
        self._cached: str | None = None

    def __str__(self) -> str:
        if self._cached is not None:
            return self._cached
        try:
            mod = importlib.import_module(self._mod_name)
            mod_path = Path(getattr(mod, "__file__", ""))
            if mod_path.is_file():
                self._cached = mod_path.read_text(encoding="utf-8")
            else:
                self._cached = str(mod)
        except Exception as exc:
            self._cached = f"# source unavailable: {self._mod_name} ({type(exc).__name__}: {exc})"
        return self._cached


def _module_source_text(mod_name: str) -> _LazyModuleSource:
    return _LazyModuleSource(str(mod_name))


LIGER_FUSED_LINEAR_CE_SRC = _module_source_text("liger_kernel.transformers.fused_linear_cross_entropy")
LIGER_FUSED_LINEAR_JSD_SRC = _module_source_text("liger_kernel.transformers.fused_linear_jsd")
LIGER_FUSED_NEIGHBORHOOD_ATTN_SRC = _module_source_text("liger_kernel.transformers.fused_neighborhood_attention")
LIGER_GRPO_LOSS_SRC = _module_source_text("liger_kernel.transformers.grpo_loss")
LIGER_LLAMA4_ROPE_SRC = _module_source_text("liger_kernel.transformers.llama4_rope")
LIGER_MHC_SRC = _module_source_text("liger_kernel.transformers.mhc")
LIGER_MULTI_TOKEN_ATTN_SRC = _module_source_text("liger_kernel.transformers.multi_token_attention")
LIGER_POLY_NORM_SRC = _module_source_text("liger_kernel.transformers.poly_norm")
LIGER_TILED_MLP_SRC = _module_source_text("liger_kernel.transformers.tiled_mlp")
LIGER_TVD_SRC = _module_source_text("liger_kernel.transformers.tvd")


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
    hd_idx = torch.arange(hd, device=q.device)
    t_mask = (hd_idx < int(section_t)).to(dtype=cos_in.dtype).view(1, 1, hd)
    h_end = int(section_t) + int(section_h)
    h_mask = ((hd_idx >= int(section_t)) & (hd_idx < h_end)).to(dtype=cos_in.dtype).view(1, 1, hd)
    w_mask = (hd_idx >= h_end).to(dtype=cos_in.dtype).view(1, 1, hd)
    cos_combined = (cos_in[0] * t_mask) + (cos_in[1] * h_mask) + (cos_in[2] * w_mask)
    sin_combined = (sin_in[0] * t_mask) + (sin_in[1] * h_mask) + (sin_in[2] * w_mask)
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
        "cos_combined": _to_np(cos_combined),
        "sin_combined": _to_np(sin_combined),
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


def _fused_linear_cross_entropy_runner(case: TestCase) -> Dict[str, np.ndarray]:
    bt = int(case.shapes["BT"])
    h = int(case.shapes["H"])
    v = int(case.shapes["V"])
    x = _torch_randn((bt, h), seed=int(case.seed) + 37)
    w = _torch_randn((v, h), seed=int(case.seed) + 38)
    bias = torch.zeros((v,), device="cuda", dtype=torch.float32)
    ce_weight = torch.ones((v,), device="cuda", dtype=torch.float32)
    target = _torch_randint(v, (bt,), seed=int(case.seed) + 39)
    x_in = x.clone()
    w_in = w.clone()
    target_in = target.clone()
    loss = liger_fused_linear_cross_entropy(
        x,
        w,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss=False,
    )
    torch.cuda.synchronize()
    return {
        "input": _to_np(x_in),
        "weight": _to_np(w_in),
        "bias": _to_np(bias),
        "ce_weight": _to_np(ce_weight),
        "target": _to_np(target_in),
        "ignore_index": np.array(-100, dtype=np.int64),
        "lse_square_scale": np.array(0.0, dtype=np.float32),
        "label_smoothing": np.array(0.0, dtype=np.float32),
        "loss": _to_np(loss),
    }


def _fused_linear_jsd_runner(case: TestCase) -> Dict[str, np.ndarray]:
    bt = int(case.shapes["BT"])
    h = int(case.shapes["H"])
    v = int(case.shapes["V"])
    student_input = _torch_randn((bt, h), seed=int(case.seed) + 40)
    teacher_input = _torch_randn((bt, h), seed=int(case.seed) + 41)
    student_weight = _torch_randn((v, h), seed=int(case.seed) + 42)
    teacher_weight = _torch_randn((v, h), seed=int(case.seed) + 43)
    shift_labels = _torch_randint(v, (bt,), seed=int(case.seed) + 44)
    loss = liger_fused_linear_jsd(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        shift_labels,
        jsd_beta=0.5,
        ignore_index=-100,
        temperature=1.0,
    )
    torch.cuda.synchronize()
    return {
        "student_input": _to_np(student_input),
        "student_weight": _to_np(student_weight),
        "teacher_input": _to_np(teacher_input),
        "teacher_weight": _to_np(teacher_weight),
        "shift_labels": _to_np(shift_labels),
        "ignore_index": np.array(-100, dtype=np.int64),
        "temperature": np.array(1.0, dtype=np.float32),
        "loss": _to_np(loss),
    }


def _fused_neighborhood_attention_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    qh = int(case.shapes["QH"])
    s = int(case.shapes["S"])
    hd = int(case.shapes["HD"])
    kernel_size = int(case.shapes.get("kernel_size", 7))
    dilation = int(case.shapes.get("dilation", 1))
    scale = np.array(1.0 / np.sqrt(float(max(1, hd))), dtype=np.float32)
    query = _torch_randn((b, qh, s, hd), seed=int(case.seed) + 45)
    key = _torch_randn((b, qh, s, hd), seed=int(case.seed) + 46)
    value = _torch_randn((b, qh, s, hd), seed=int(case.seed) + 47)
    y = liger_fused_neighborhood_attention(query, key, value, kernel_size=kernel_size, dilation=dilation, scale=None)
    torch.cuda.synchronize()
    return {
        "query": _to_np(query),
        "key": _to_np(key),
        "value": _to_np(value),
        "Q": _to_np(query),
        "K": _to_np(key),
        "V": _to_np(value),
        "kernel_size": np.array(kernel_size, dtype=np.int32),
        "dilation": np.array(dilation, dtype=np.int32),
        "scale": scale,
        "Y": _to_np(y),
        "O": _to_np(y),
    }


def _grpo_loss_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    t = int(case.shapes["T"])
    v = int(case.shapes["V"])
    logits = _torch_randn((b, t + 1, v), seed=int(case.seed) + 48)
    old_logp = _torch_randn((b, t), seed=int(case.seed) + 49)
    ref_logp = _torch_randn((b, t), seed=int(case.seed) + 50)
    completion_ids = _torch_randint(v, (b, t), seed=int(case.seed) + 51)
    advantages = _torch_randn((b,), seed=int(case.seed) + 52)
    completion_mask = torch.ones((b, t), device="cuda", dtype=torch.float32)
    temperature = np.array(0.9, dtype=np.float32)
    beta = np.array(0.04, dtype=np.float32)
    eps_low = np.array(0.2, dtype=np.float32)
    eps_high = np.array(0.4, dtype=np.float32)
    loss, metrics = triton_grpo_loss(
        logits,
        old_logp,
        ref_logp,
        completion_ids,
        advantages,
        completion_mask,
        temperature=float(temperature),
        beta=float(beta),
        eps_low=float(eps_low),
        eps_high=float(eps_high),
        inplace=True,
        loss_type="dapo",
        importance_sampling_level="token",
        reduce=True,
    )
    torch.cuda.synchronize()
    metrics_arr = np.asarray([float(x.detach().cpu().item()) for x in list(metrics or [])], dtype=np.float32)
    return {
        "logits": _to_np(logits),
        "old_logp": _to_np(old_logp),
        "ref_logp": _to_np(ref_logp),
        "completion_ids": _to_np(completion_ids),
        "advantages": _to_np(advantages),
        "completion_mask": _to_np(completion_mask),
        "temperature": temperature,
        "beta": beta,
        "eps_low": eps_low,
        "eps_high": eps_high,
        "loss": _to_np(loss),
        "metrics": metrics_arr,
    }


def _llama4_rope_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    qh = int(case.shapes["QH"])
    kh = int(case.shapes["KH"])
    s = int(case.shapes["S"])
    hd = int(case.shapes["HD"])
    q = _torch_randn((b, s, qh, hd), seed=int(case.seed) + 53)
    k = _torch_randn((b, s, kh, hd), seed=int(case.seed) + 54)
    real = _torch_randn((s, hd // 2), seed=int(case.seed) + 55)
    imag = _torch_randn((s, hd // 2), seed=int(case.seed) + 56)
    q_in = q.clone()
    k_in = k.clone()
    freqs_cis = torch.complex(real, imag)
    q_out, k_out = llama4_rope_forward(q, k, freqs_cis)
    torch.cuda.synchronize()
    q_sem = torch.cat([q_in[..., 0::2], q_in[..., 1::2]], dim=-1)
    k_sem = torch.cat([k_in[..., 0::2], k_in[..., 1::2]], dim=-1)
    q_out_sem = torch.cat([q_out[..., 0::2], q_out[..., 1::2]], dim=-1)
    k_out_sem = torch.cat([k_out[..., 0::2], k_out[..., 1::2]], dim=-1)
    return {
        "q": _to_np(q_sem),
        "k": _to_np(k_sem),
        "cos": _to_np(real),
        "sin": _to_np(imag),
        "q_out": _to_np(q_out_sem),
        "k_out": _to_np(k_out_sem),
    }


def _mhc_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    t = int(case.shapes["T"])
    hc = int(case.shapes["HC"])
    c = int(case.shapes["C"])
    x = _torch_randn((b, t, hc, c), seed=int(case.seed) + 57)
    phi = _torch_randn((hc * c, hc * hc + 2 * hc), seed=int(case.seed) + 58)
    bias = _torch_randn((hc * hc + 2 * hc,), seed=int(case.seed) + 59)
    alpha_pre = _torch_randn((1,), seed=int(case.seed) + 60)
    alpha_post = _torch_randn((1,), seed=int(case.seed) + 61)
    alpha_res = _torch_randn((1,), seed=int(case.seed) + 62)
    layer = torch.nn.Linear(c, c, bias=False, device="cuda", dtype=torch.float32)
    layer_weight = _torch_randn((c, c), seed=int(case.seed) + 63)
    with torch.no_grad():
        layer.weight.copy_(layer_weight)
    y = liger_mhc_forward(
        x,
        layer,
        phi,
        bias,
        alpha_pre.reshape(()),
        alpha_post.reshape(()),
        alpha_res.reshape(()),
        allow_fp32=True,
        tmax=8,
    )
    torch.cuda.synchronize()
    return {
        "X": _to_np(x),
        "Phi": _to_np(phi),
        "B": _to_np(bias),
        "AlphaPre": _to_np(alpha_pre.reshape(())),
        "AlphaPost": _to_np(alpha_post.reshape(())),
        "AlphaRes": _to_np(alpha_res.reshape(())),
        "LayerW": _to_np(layer_weight),
        "Y": _to_np(y),
    }


def _multi_token_attention_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    c_in = int(case.shapes["CIN"])
    c_out = int(case.shapes["COUT"])
    l = int(case.shapes["L"])
    k = int(case.shapes.get("K", 3))
    groups = int(case.shapes.get("groups", 1))
    scores = _torch_randn((b, c_in, l, l), seed=int(case.seed) + 64)
    weight = _torch_randn((c_out, c_in // groups, k, k), seed=int(case.seed) + 65)
    bias = _torch_randn((c_out,), seed=int(case.seed) + 66)
    y = liger_multi_token_attention(scores, weight, bias, stride=1, padding=k // 2, dilation=1, groups=groups, sparse=False)
    torch.cuda.synchronize()
    return {
        "scores": _to_np(scores),
        "weight": _to_np(weight),
        "bias": _to_np(bias),
        "groups": np.array(groups, dtype=np.int32),
        "kernel_size": np.array(k, dtype=np.int32),
        "Y": _to_np(y),
    }


def _poly_norm_runner(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    x = _torch_randn((m, n), seed=int(case.seed) + 67)
    w = _torch_randn((3,), seed=int(case.seed) + 68)
    b = _torch_randn((1,), seed=int(case.seed) + 69).reshape(())
    y = liger_poly_norm(x, w, b, 1.0e-6, True)
    torch.cuda.synchronize()
    return {
        "X": _to_np(x),
        "W": _to_np(w),
        "B": _to_np(b),
        "eps": np.array(1.0e-6, dtype=np.float32),
        "Y": _to_np(y),
    }


def _tiled_mlp_runner(case: TestCase) -> Dict[str, np.ndarray]:
    b = int(case.shapes["B"])
    s = int(case.shapes["S"])
    h = int(case.shapes["H"])
    i = int(case.shapes["I"])
    num_shards = max(1, min(int(case.shapes.get("num_shards", 4)), s))
    cfg = SimpleNamespace(hidden_size=h, intermediate_size=i, hidden_act="gelu_pytorch_tanh")
    mlp = LigerTiledGEGLUMLP(config=cfg, num_shards=num_shards).to("cuda").to(torch.float32)
    gate_w = _torch_randn((i, h), seed=int(case.seed) + 70)
    up_w = _torch_randn((i, h), seed=int(case.seed) + 71)
    down_w = _torch_randn((h, i), seed=int(case.seed) + 72)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(gate_w)
        mlp.up_proj.weight.copy_(up_w)
        mlp.down_proj.weight.copy_(down_w)
    x = _torch_randn((b, s, h), seed=int(case.seed) + 73)
    y = mlp(x)
    torch.cuda.synchronize()
    return {
        "X": _to_np(x),
        "GateW": _to_np(gate_w.transpose(0, 1).contiguous()),
        "UpW": _to_np(up_w.transpose(0, 1).contiguous()),
        "DownW": _to_np(down_w.transpose(0, 1).contiguous()),
        "num_shards": np.array(num_shards, dtype=np.int32),
        "Y": _to_np(y),
    }


def _tvd_runner(case: TestCase) -> Dict[str, np.ndarray]:
    bt = int(case.shapes["BT"])
    v = int(case.shapes["V"])
    p = torch.softmax(_torch_randn((bt, v), seed=int(case.seed) + 74), dim=-1)
    q = torch.softmax(_torch_randn((bt, v), seed=int(case.seed) + 75), dim=-1)
    loss = liger_tvd(p, q, None, reduction="batchmean", ignore_index=-100)
    torch.cuda.synchronize()
    return {
        "input": _to_np(p),
        "target": _to_np(q),
        "ignore_index": np.array(-100, dtype=np.int64),
        "loss": _to_np(loss),
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
        KernelSpec(
            name="liger_fused_linear_cross_entropy",
            module=module,
            attr="LIGER_FUSED_LINEAR_CE_SRC",
            runner=_fused_linear_cross_entropy_runner,
            canonical_shapes={"BT": 2048, "H": 2048, "V": 4096},
            vary_axes=["BT", "H"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_fused_linear_jsd",
            module=module,
            attr="LIGER_FUSED_LINEAR_JSD_SRC",
            runner=_fused_linear_jsd_runner,
            canonical_shapes={"BT": 2048, "H": 2048, "V": 4096},
            vary_axes=["BT", "H"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_fused_neighborhood_attention",
            module=module,
            attr="LIGER_FUSED_NEIGHBORHOOD_ATTN_SRC",
            runner=_fused_neighborhood_attention_runner,
            canonical_shapes={"B": 1, "QH": 8, "S": 512, "HD": 64, "kernel_size": 7, "dilation": 1},
            vary_axes=["B", "QH", "S", "HD"],
            exclude_axes=["kernel_size", "dilation"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_grpo_loss",
            module=module,
            attr="LIGER_GRPO_LOSS_SRC",
            runner=_grpo_loss_runner,
            canonical_shapes={"B": 4, "T": 512, "V": 4096},
            vary_axes=["B", "T"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_llama4_rope",
            module=module,
            attr="LIGER_LLAMA4_ROPE_SRC",
            runner=_llama4_rope_runner,
            canonical_shapes={"B": 1, "QH": 32, "KH": 8, "S": 2048, "HD": 64},
            vary_axes=["B", "QH", "KH", "S", "HD"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_mhc",
            module=module,
            attr="LIGER_MHC_SRC",
            runner=_mhc_runner,
            canonical_shapes={"B": 2, "T": 512, "HC": 4, "C": 128},
            vary_axes=["B", "T", "HC", "C"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_multi_token_attention",
            module=module,
            attr="LIGER_MULTI_TOKEN_ATTN_SRC",
            runner=_multi_token_attention_runner,
            canonical_shapes={"B": 2, "CIN": 4, "COUT": 4, "L": 128, "K": 3, "groups": 1},
            vary_axes=["B", "CIN", "L"],
            exclude_axes=["COUT", "K", "groups"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_poly_norm",
            module=module,
            attr="LIGER_POLY_NORM_SRC",
            runner=_poly_norm_runner,
            canonical_shapes={"M": 2048, "N": 4096},
            vary_axes=["M", "N"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_tiled_mlp",
            module=module,
            attr="LIGER_TILED_MLP_SRC",
            runner=_tiled_mlp_runner,
            canonical_shapes={"B": 1, "S": 4096, "H": 2048, "I": 5632, "num_shards": 4},
            vary_axes=["B", "S"],
            exclude_axes=["H", "I", "num_shards"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
        KernelSpec(
            name="liger_tvd",
            module=module,
            attr="LIGER_TVD_SRC",
            runner=_tvd_runner,
            canonical_shapes={"BT": 2048, "V": 4096},
            vary_axes=["BT", "V"],
            enable_stage_c=False,
            enable_mutation_kill=False,
        ),
    ]


__all__ = ["liger_kernel_specs"]
