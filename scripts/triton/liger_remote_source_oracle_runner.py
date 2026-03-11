from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
