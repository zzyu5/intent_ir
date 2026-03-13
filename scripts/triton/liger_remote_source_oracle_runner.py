from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _find_latest_artifact(dump_dir: Path, *, suffix: str, hint: str) -> Path | None:
    hits = sorted(dump_dir.rglob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in hits:
        if hint and hint in path.name:
            return path
    return hits[0] if hits else None


def _prepare_dump_dirs(out_dir: Path, kernel: str) -> tuple[Path, Path]:
    dump_dir = out_dir / "_triton_dump" / kernel
    cache_dir = out_dir / "_triton_cache" / kernel
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    os.environ["TRITON_DUMP_DIR"] = str(dump_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
    return dump_dir, cache_dir


def _module_source_attr(mod_name: str) -> str:
    mod = importlib.import_module(str(mod_name))
    mod_path = Path(getattr(mod, "__file__", ""))
    if mod_path.is_file():
        return mod_path.read_text(encoding="utf-8")
    return str(mod)


def _run_kernel(kernel: str, bindings: dict[str, int]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available on remote source host")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32

    if kernel == "liger_swiglu":
        from liger_kernel.ops.swiglu import _swiglu_forward_kernel, swiglu_forward

        m = int(bindings.get("M", 65536))
        n = int(bindings.get("N", 256))
        a = torch.randn((m, n), device=device, dtype=dtype)
        b = torch.randn((m, n), device=device, dtype=dtype)
        _a, _b, c = swiglu_forward(a, b)
        torch.cuda.synchronize()
        return {"entry_hint": "_swiglu_forward_kernel", "source_attr": str(_swiglu_forward_kernel.src), "output_shape": list(c.shape)}

    if kernel == "liger_rms_norm":
        from liger_kernel.ops.rms_norm import _rms_norm_forward_kernel, rms_norm_forward

        m = int(bindings.get("M", 2048))
        n = int(bindings.get("N", 32768))
        x = torch.randn((m, n), device=device, dtype=dtype)
        w = torch.randn((n,), device=device, dtype=dtype)
        y, _x2, rstd, block_size, num_warps, casting_mode = rms_norm_forward(
            x,
            w,
            1.0e-5,
            0.0,
            "none",
            False,
        )
        torch.cuda.synchronize()
        return {
            "entry_hint": "_rms_norm_forward_kernel",
            "source_attr": str(_rms_norm_forward_kernel.src),
            "output_shape": list(y.shape),
            "rstd_shape": list(rstd.shape),
            "block_size": int(block_size),
            "num_warps": int(num_warps),
            "casting_mode": int(casting_mode),
        }

    if kernel == "liger_fused_add_rms_norm":
        from liger_kernel.ops.fused_add_rms_norm import _fused_add_rms_norm_forward_kernel, fused_add_rms_norm_forward

        m = int(bindings.get("M", 2048))
        n = int(bindings.get("N", 32768))
        x = torch.randn((m, n), device=device, dtype=dtype)
        r = torch.randn((m, n), device=device, dtype=dtype)
        w = torch.randn((n,), device=device, dtype=dtype)
        y, s, rstd, block_size, num_warps, casting_mode = fused_add_rms_norm_forward(
            x,
            r,
            w,
            1.0e-5,
            0.0,
            "none",
        )
        torch.cuda.synchronize()
        return {
            "entry_hint": "_fused_add_rms_norm_forward_kernel",
            "source_attr": str(_fused_add_rms_norm_forward_kernel.src),
            "output_shape": list(y.shape),
            "residual_shape": list(s.shape),
            "rstd_shape": list(rstd.shape),
            "block_size": int(block_size),
            "num_warps": int(num_warps),
            "casting_mode": int(casting_mode),
        }

    if kernel == "liger_rope":
        from liger_kernel.ops.rope import _triton_rope, rope_forward

        b = int(bindings.get("B", 2))
        qh = int(bindings.get("QH", 32))
        kh = int(bindings.get("KH", 8))
        s = int(bindings.get("S", 2048))
        hd = int(bindings.get("HD", 128))
        q = torch.randn((b, qh, s, hd), device=device, dtype=dtype)
        k = torch.randn((b, kh, s, hd), device=device, dtype=dtype)
        cos = torch.randn((1, s, hd), device=device, dtype=dtype)
        sin = torch.randn((1, s, hd), device=device, dtype=dtype)
        q_out, k_out, cos_out, sin_out = rope_forward(q, k, cos, sin)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_triton_rope",
            "source_attr": str(_triton_rope.src),
            "q_shape": list(q_out.shape),
            "k_shape": list(k_out.shape),
            "cos_shape": list(cos_out.shape),
            "sin_shape": list(sin_out.shape),
        }

    if kernel == "liger_cross_entropy":
        from liger_kernel.ops.cross_entropy import cross_entropy_forward, liger_cross_entropy_kernel

        bt = int(bindings.get("BT", 2048))
        v = int(bindings.get("V", 4096))
        x = torch.randn((bt, v), device=device, dtype=dtype)
        target = torch.randint(0, v, (bt,), device=device, dtype=torch.int64)
        loss, z_loss, token_accuracy, predicted_tokens, x_out = cross_entropy_forward(
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
            "entry_hint": "liger_cross_entropy_kernel",
            "source_attr": str(liger_cross_entropy_kernel.src),
            "loss_shape": list(loss.shape),
            "input_shape": list(x_out.shape),
            "z_loss_is_none": bool(z_loss is None),
            "token_accuracy_is_none": bool(token_accuracy is None),
            "predicted_tokens_is_none": bool(predicted_tokens is None),
        }

    if kernel == "liger_geglu":
        from liger_kernel.ops.geglu import _geglu_tanh_forward_kernel, geglu_forward

        m = int(bindings.get("M", 65536))
        n = int(bindings.get("N", 256))
        a = torch.randn((m, n), device=device, dtype=dtype)
        b = torch.randn((m, n), device=device, dtype=dtype)
        _a, _b, c = geglu_forward(a, b)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_geglu_tanh_forward_kernel",
            "source_attr": str(_geglu_tanh_forward_kernel.src),
            "output_shape": list(c.shape),
        }

    if kernel == "liger_layer_norm":
        from liger_kernel.ops.layer_norm import _layer_norm_forward_kernel, layer_norm_forward

        m = int(bindings.get("M", 2048))
        n = int(bindings.get("N", 4096))
        x = torch.randn((m, n), device=device, dtype=dtype)
        w = torch.randn((n,), device=device, dtype=dtype)
        b = torch.randn((n,), device=device, dtype=dtype)
        y, _x2, mean, rstd, block_size, num_warps = layer_norm_forward(x, w, b, 1.0e-5)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_layer_norm_forward_kernel",
            "source_attr": str(_layer_norm_forward_kernel.src),
            "output_shape": list(y.shape),
            "mean_shape": list(mean.shape),
            "rstd_shape": list(rstd.shape),
            "block_size": int(block_size),
            "num_warps": int(num_warps),
        }

    if kernel == "liger_softmax":
        from liger_kernel.ops.softmax import _softmax_forward, _softmax_single_block_forward_kernel

        m = int(bindings.get("M", 2048))
        n = int(bindings.get("N", 4096))
        x = torch.randn((m, n), device=device, dtype=dtype)
        y, block_size, num_warps, multi_block_launch = _softmax_forward(x)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_softmax_single_block_forward_kernel",
            "source_attr": str(_softmax_single_block_forward_kernel.src),
            "output_shape": list(y.shape),
            "block_size": int(block_size),
            "num_warps": int(num_warps),
            "multi_block_launch": bool(multi_block_launch),
        }

    if kernel == "liger_group_norm":
        from liger_kernel.ops.group_norm import _group_norm_forward_kernel, group_norm_forward

        n = int(bindings.get("N", 32))
        c = int(bindings.get("C", 512))
        hw = int(bindings.get("HW", 64))
        num_groups = int(bindings.get("num_groups", 32))
        x = torch.randn((n, c, hw), device=device, dtype=dtype)
        w = torch.randn((c,), device=device, dtype=dtype)
        b = torch.randn((c,), device=device, dtype=dtype)
        y, _x2, mean, rstd, block_size = group_norm_forward(x, c, num_groups, w, b, 1.0e-5)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_group_norm_forward_kernel",
            "source_attr": str(_group_norm_forward_kernel.src),
            "output_shape": list(y.shape),
            "mean_shape": list(mean.shape),
            "rstd_shape": list(rstd.shape),
            "block_size": int(block_size),
            "num_groups": int(num_groups),
        }

    if kernel == "liger_dyt":
        from liger_kernel.ops.dyt import _dyt_fwd_kernel, liger_dyt_fwd

        m = int(bindings.get("M", 2048))
        n = int(bindings.get("N", 4096))
        x = torch.randn((m, n), device=device, dtype=dtype)
        alpha = torch.randn((1,), device=device, dtype=dtype)
        gamma = torch.randn((n,), device=device, dtype=dtype)
        beta = torch.randn((n,), device=device, dtype=dtype)
        y = liger_dyt_fwd(x, alpha, gamma, beta)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_dyt_fwd_kernel",
            "source_attr": str(_dyt_fwd_kernel.src),
            "output_shape": list(y.shape),
            "alpha_shape": list(alpha.shape),
            "gamma_shape": list(gamma.shape),
            "beta_shape": list(beta.shape),
        }

    if kernel == "liger_qwen2vl_mrope":
        from liger_kernel.ops.qwen2vl_mrope import _triton_qwen2vl_mrope, qwen2vl_mrope_forward

        b = int(bindings.get("B", 2))
        qh = int(bindings.get("QH", 32))
        kh = int(bindings.get("KH", 8))
        s = int(bindings.get("S", 2048))
        hd = int(bindings.get("HD", 128))
        q = torch.randn((b, qh, s, hd), device=device, dtype=dtype)
        k = torch.randn((b, kh, s, hd), device=device, dtype=dtype)
        cos = torch.randn((3, b, s, hd), device=device, dtype=dtype)
        sin = torch.randn((3, b, s, hd), device=device, dtype=dtype)
        section_t = max(1, hd // 8)
        section_h = max(1, hd // 8)
        q_out, k_out, _cos_out, _sin_out = qwen2vl_mrope_forward(q, k, cos, sin, [section_t, section_h])
        torch.cuda.synchronize()
        return {
            "entry_hint": "_triton_qwen2vl_mrope",
            "source_attr": str(_triton_qwen2vl_mrope.src),
            "q_shape": list(q_out.shape),
            "k_shape": list(k_out.shape),
            "cos_shape": list(cos.shape),
            "sin_shape": list(sin.shape),
            "mrope_section": [int(section_t), int(section_h)],
        }

    if kernel == "liger_sparsemax":
        from liger_kernel.ops.sparsemax import _sparsemax_forward, _sparsemax_forward_kernel

        m = int(bindings.get("M", 2048))
        n = int(bindings.get("N", 4096))
        x = torch.randn((m, n), device=device, dtype=dtype)
        y, out_flat = _sparsemax_forward(x, -1)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_sparsemax_forward_kernel",
            "source_attr": str(_sparsemax_forward_kernel.src),
            "output_shape": list(y.shape),
            "flat_shape": list(out_flat.shape),
        }

    if kernel == "liger_kl_div":
        from liger_kernel.ops.kl_div import _kldiv_kernel_forward, kldiv_forward_triton

        bt = int(bindings.get("BT", 2048))
        v = int(bindings.get("V", 4096))
        y_pred = torch.log_softmax(torch.randn((bt, v), device=device, dtype=dtype), dim=-1)
        y_true = torch.softmax(torch.randn((bt, v), device=device, dtype=dtype), dim=-1)
        loss = kldiv_forward_triton(y_pred, y_true, False, "batchmean", 1.0e-10)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_kldiv_kernel_forward",
            "source_attr": str(_kldiv_kernel_forward.src),
            "loss_shape": list(loss.shape),
            "input_shape": list(y_pred.shape),
            "target_shape": list(y_true.shape),
        }

    if kernel == "liger_jsd":
        from liger_kernel.ops.jsd import _jsd_kernel, jsd_forward

        bt = int(bindings.get("BT", 2048))
        v = int(bindings.get("V", 4096))
        x = torch.log_softmax(torch.randn((bt, v), device=device, dtype=dtype), dim=-1)
        y = torch.log_softmax(torch.randn((bt, v), device=device, dtype=dtype), dim=-1)
        labels = torch.randint(0, v, (bt,), device=device, dtype=torch.int64)
        loss, d_x = jsd_forward(x, y, labels, 0.5, -100, True)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_jsd_kernel",
            "source_attr": str(_jsd_kernel.src),
            "loss_shape": list(loss.shape),
            "input_shape": list(x.shape),
            "target_shape": list(y.shape),
            "dx_shape": list(d_x.shape),
        }

    if kernel == "liger_fused_linear_cross_entropy":
        from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
        from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy

        bt = int(bindings.get("BT", 2048))
        h = int(bindings.get("H", 2048))
        v = int(bindings.get("V", 4096))
        x = torch.randn((bt, h), device=device, dtype=dtype)
        w = torch.randn((v, h), device=device, dtype=dtype)
        target = torch.randint(0, v, (bt,), device=device, dtype=torch.int64)
        loss = liger_fused_linear_cross_entropy(x, w, target, None, None, -100, 0.0, 0.0, "mean", None, False)
        torch.cuda.synchronize()
        return {
            "entry_hint": "liger_cross_entropy_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.fused_linear_cross_entropy"),
            "loss_shape": list(loss.shape),
            "input_shape": list(x.shape),
            "weight_shape": list(w.shape),
            "target_shape": list(target.shape),
            "kernel_src_head": str(liger_cross_entropy_kernel.src)[:256],
        }

    if kernel == "liger_fused_linear_jsd":
        from liger_kernel.ops.jsd import _jsd_kernel
        from liger_kernel.transformers.functional import liger_fused_linear_jsd

        bt = int(bindings.get("BT", 2048))
        h = int(bindings.get("H", 2048))
        v = int(bindings.get("V", 4096))
        student_input = torch.randn((bt, h // 2), device=device, dtype=dtype)
        teacher_input = torch.randn((bt, h), device=device, dtype=dtype)
        student_weight = torch.randn((v, h // 2), device=device, dtype=dtype)
        teacher_weight = torch.randn((v, h), device=device, dtype=dtype)
        labels = torch.randint(0, v, (bt,), device=device, dtype=torch.int64)
        loss = liger_fused_linear_jsd(
            student_input,
            student_weight,
            teacher_input,
            teacher_weight,
            labels,
            jsd_beta=0.5,
            ignore_index=-100,
            temperature=1.0,
        )
        torch.cuda.synchronize()
        return {
            "entry_hint": "_jsd_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.fused_linear_jsd"),
            "loss_shape": list(loss.shape),
            "student_input_shape": list(student_input.shape),
            "teacher_input_shape": list(teacher_input.shape),
            "student_weight_shape": list(student_weight.shape),
            "teacher_weight_shape": list(teacher_weight.shape),
            "label_shape": list(labels.shape),
            "kernel_src_head": str(_jsd_kernel.src)[:256],
        }

    if kernel == "liger_fused_neighborhood_attention":
        from liger_kernel.ops.fused_neighborhood_attention import _fused_neighborhood_attention_qk_kernel
        from liger_kernel.transformers.functional import liger_fused_neighborhood_attention

        b = int(bindings.get("B", 1))
        qh = int(bindings.get("QH", 8))
        s = int(bindings.get("S", 512))
        hd = int(bindings.get("HD", 64))
        kernel_size = int(bindings.get("kernel_size", 7))
        dilation = int(bindings.get("dilation", 1))
        query = torch.randn((b, qh, s, hd), device=device, dtype=dtype)
        key = torch.randn((b, qh, s, hd), device=device, dtype=dtype)
        value = torch.randn((b, qh, s, hd), device=device, dtype=dtype)
        y = liger_fused_neighborhood_attention(query, key, value, kernel_size=kernel_size, dilation=dilation, scale=None)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_fused_neighborhood_attention_qk_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.fused_neighborhood_attention"),
            "output_shape": list(y.shape),
            "query_shape": list(query.shape),
            "kernel_size": kernel_size,
            "dilation": dilation,
            "kernel_src_head": str(_fused_neighborhood_attention_qk_kernel.src)[:256],
        }

    if kernel == "liger_grpo_loss":
        from liger_kernel.ops.grpo_loss import _grpo_loss_fwd_kernel
        from liger_kernel.transformers.grpo_loss import triton_grpo_loss

        b = int(bindings.get("B", 4))
        t = int(bindings.get("T", 512))
        v = int(bindings.get("V", 4096))
        logits = torch.randn((b, t + 1, v), device=device, dtype=dtype)
        old_logp = torch.randn((b, t), device=device, dtype=dtype)
        ref_logp = torch.randn((b, t), device=device, dtype=dtype)
        completion_ids = torch.randint(0, v, (b, t), device=device, dtype=torch.int64)
        advantages = torch.randn((b,), device=device, dtype=dtype)
        completion_mask = torch.ones((b, t), device=device, dtype=dtype)
        loss, metrics = triton_grpo_loss(
            logits,
            old_logp,
            ref_logp,
            completion_ids,
            advantages,
            completion_mask,
            reduce=True,
        )
        torch.cuda.synchronize()
        return {
            "entry_hint": "_grpo_loss_fwd_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.grpo_loss"),
            "loss_shape": list(loss.shape),
            "metrics": [float(x.detach().cpu().item()) for x in list(metrics or [])],
            "logits_shape": list(logits.shape),
            "completion_shape": list(completion_ids.shape),
            "kernel_src_head": str(_grpo_loss_fwd_kernel.src)[:256],
        }

    if kernel == "liger_llama4_rope":
        from liger_kernel.ops.llama4_rope import _llama4_rope_kernel, llama4_rope_forward

        b = int(bindings.get("B", 1))
        qh = int(bindings.get("QH", 32))
        kh = int(bindings.get("KH", 8))
        s = int(bindings.get("S", 2048))
        hd = int(bindings.get("HD", 64))
        q = torch.randn((b, s, qh, hd), device=device, dtype=dtype)
        k = torch.randn((b, s, kh, hd), device=device, dtype=dtype)
        freqs_cis = torch.complex(
            torch.randn((s, hd // 2), device=device, dtype=dtype),
            torch.randn((s, hd // 2), device=device, dtype=dtype),
        )
        q_out, k_out = llama4_rope_forward(q, k, freqs_cis)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_llama4_rope_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.llama4_rope"),
            "q_shape": list(q_out.shape),
            "k_shape": list(k_out.shape),
            "freqs_shape": list(freqs_cis.shape),
            "kernel_src_head": str(_llama4_rope_kernel.src)[:256],
        }

    if kernel == "liger_mhc":
        from liger_kernel.ops.mhc import _mhc_mm_norm_fwd_kernel
        from liger_kernel.transformers.functional import liger_mhc_forward

        b = int(bindings.get("B", 2))
        t = int(bindings.get("T", 512))
        hc = int(bindings.get("HC", 4))
        c = int(bindings.get("C", 128))
        x = torch.randn((b, t, hc, c), device=device, dtype=dtype)
        phi = torch.randn((hc * c, hc * hc + 2 * hc), device=device, dtype=dtype)
        bias = torch.randn((hc * hc + 2 * hc,), device=device, dtype=dtype)
        alpha_pre = torch.randn((), device=device, dtype=dtype)
        alpha_post = torch.randn((), device=device, dtype=dtype)
        alpha_res = torch.randn((), device=device, dtype=dtype)
        layer = torch.nn.Linear(c, c, bias=False, device=device, dtype=dtype)
        y = liger_mhc_forward(
            x,
            layer,
            phi,
            bias,
            alpha_pre,
            alpha_post,
            alpha_res,
            allow_fp32=True,
            tmax=8,
        )
        torch.cuda.synchronize()
        return {
            "entry_hint": "_mhc_mm_norm_fwd_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.mhc"),
            "output_shape": list(y.shape),
            "x_shape": list(x.shape),
            "phi_shape": list(phi.shape),
            "kernel_src_head": str(_mhc_mm_norm_fwd_kernel.src)[:256],
        }

    if kernel == "liger_multi_token_attention":
        from liger_kernel.ops.multi_token_attention import _mask_fwd_kernel
        from liger_kernel.transformers.functional import liger_multi_token_attention

        b = int(bindings.get("B", 2))
        c_in = int(bindings.get("CIN", 4))
        c_out = int(bindings.get("COUT", 4))
        l = int(bindings.get("L", 128))
        k = int(bindings.get("K", 3))
        groups = int(bindings.get("groups", 1))
        scores = torch.randn((b, c_in, l, l), device=device, dtype=dtype)
        weight = torch.randn((c_out, c_in // groups, k, k), device=device, dtype=dtype)
        bias = torch.randn((c_out,), device=device, dtype=dtype)
        y = liger_multi_token_attention(scores, weight, bias, stride=1, padding=k // 2, dilation=1, groups=groups, sparse=False)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_mask_fwd_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.multi_token_attention"),
            "output_shape": list(y.shape),
            "scores_shape": list(scores.shape),
            "weight_shape": list(weight.shape),
            "kernel_src_head": str(_mask_fwd_kernel.src)[:256],
        }

    if kernel == "liger_poly_norm":
        from liger_kernel.ops.poly_norm import _poly_norm_forward_kernel
        from liger_kernel.transformers.functional import liger_poly_norm

        m = int(bindings.get("M", 2048))
        n = int(bindings.get("N", 4096))
        x = torch.randn((m, n), device=device, dtype=dtype)
        w = torch.randn((3,), device=device, dtype=dtype)
        b = torch.randn((), device=device, dtype=dtype)
        y = liger_poly_norm(x, w, b, 1.0e-6, True)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_poly_norm_forward_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.poly_norm"),
            "output_shape": list(y.shape),
            "input_shape": list(x.shape),
            "kernel_src_head": str(_poly_norm_forward_kernel.src)[:256],
        }

    if kernel == "liger_tiled_mlp":
        from liger_kernel.ops.geglu import _geglu_tanh_forward_kernel
        from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP

        b = int(bindings.get("B", 1))
        s = int(bindings.get("S", 4096))
        h = int(bindings.get("H", 2048))
        i = int(bindings.get("I", 5632))
        num_shards = int(bindings.get("num_shards", 4))
        cfg = SimpleNamespace(hidden_size=h, intermediate_size=i, hidden_act="gelu_pytorch_tanh")
        mlp = LigerTiledGEGLUMLP(config=cfg, num_shards=num_shards).to(device).to(dtype)
        x = torch.randn((b, s, h), device=device, dtype=dtype)
        y = mlp(x)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_geglu_tanh_forward_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.tiled_mlp"),
            "output_shape": list(y.shape),
            "input_shape": list(x.shape),
            "num_shards": num_shards,
            "kernel_src_head": str(_geglu_tanh_forward_kernel.src)[:256],
        }

    if kernel == "liger_tvd":
        from liger_kernel.ops.tvd import _tv_distance_kernel
        from liger_kernel.transformers.functional import liger_tvd

        bt = int(bindings.get("BT", 2048))
        v = int(bindings.get("V", 4096))
        p = torch.softmax(torch.randn((bt, v), device=device, dtype=dtype), dim=-1)
        q = torch.softmax(torch.randn((bt, v), device=device, dtype=dtype), dim=-1)
        loss = liger_tvd(p, q, None, reduction="batchmean", ignore_index=-100)
        torch.cuda.synchronize()
        return {
            "entry_hint": "_tv_distance_kernel",
            "source_attr": _module_source_attr("liger_kernel.transformers.tvd"),
            "loss_shape": list(loss.shape),
            "input_shape": list(p.shape),
            "target_shape": list(q.shape),
            "kernel_src_head": str(_tv_distance_kernel.src)[:256],
        }

    raise KeyError(f"unsupported kernel: {kernel}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bindings-json", default="{}")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bindings = json.loads(str(args.bindings_json))
    if not isinstance(bindings, dict):
        raise TypeError("--bindings-json must decode to an object")

    dump_dir, cache_dir = _prepare_dump_dirs(out_dir, str(args.kernel))
    run_info = _run_kernel(str(args.kernel), {str(k): int(v) for k, v in dict(bindings).items()})

    hint = str(run_info.get("entry_hint") or "")
    ttir = _find_latest_artifact(dump_dir, suffix=".ttir", hint=hint) or _find_latest_artifact(cache_dir, suffix=".ttir", hint=hint)
    ttgir = _find_latest_artifact(dump_dir, suffix=".ttgir", hint=hint) or _find_latest_artifact(cache_dir, suffix=".ttgir", hint=hint)
    ptx = _find_latest_artifact(dump_dir, suffix=".ptx", hint=hint) or _find_latest_artifact(cache_dir, suffix=".ptx", hint=hint)
    llir = _find_latest_artifact(dump_dir, suffix=".llir", hint=hint) or _find_latest_artifact(cache_dir, suffix=".llir", hint=hint)
    cubin = _find_latest_artifact(dump_dir, suffix=".cubin", hint=hint) or _find_latest_artifact(cache_dir, suffix=".cubin", hint=hint)

    import torch
    import triton

    manifest = {
        "kernel": str(args.kernel),
        "bindings": {str(k): int(v) for k, v in dict(bindings).items()},
        "source_arch": "sm90",
        "gpu_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__),
        "triton_version": str(triton.__version__),
        "dump_dir": str(dump_dir),
        "cache_dir": str(cache_dir),
        "run_info": run_info,
        "artifacts": {
            "ttir": (str(ttir) if ttir is not None else ""),
            "ttgir": (str(ttgir) if ttgir is not None else ""),
            "ptx": (str(ptx) if ptx is not None else ""),
            "llir": (str(llir) if llir is not None else ""),
            "cubin": (str(cubin) if cubin is not None else ""),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "kernel": args.kernel, "source_arch": "sm90", "ptx": manifest["artifacts"]["ptx"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
