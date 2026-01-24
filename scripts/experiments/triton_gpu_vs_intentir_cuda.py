"""
E5 (CUDA GPU): Triton vs IntentIR->CUDA backend performance comparison.

This experiment answers: given the same high-level kernel (AI-Bench8 suite),
how fast is:
  (1) Triton JIT kernel on NVIDIA GPU, vs
  (2) IntentIR lowered into a CUDA kernel by our CUDA backend.

Notes:
  - This is *not* the schedule decoupling (freeze/retune) experiment.
  - We exclude compilation time by warming up once before timing.
  - Benchmarks use torch.cuda.Event timing (kernel time, not wall clock).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel  # noqa: E402
from frontends.cuda.runtime import CudaLaunch, compile_cuda_extension  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402
from pipeline.triton.core import coverage_kernel_specs, run_pipeline_for_spec  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "full_pipeline_verify"
OUT_DEFAULT = ROOT / "artifacts" / "experiments" / "E5" / "e5_triton_gpu_vs_intentir_cuda.json"

AI_BENCH_KERNELS: list[str] = [
    "ai_bench_matmul",
    "ai_bench_dropout",
    "ai_bench_softmax",
    "ai_bench_layernorm",
    "ai_bench_correlation",
    "ai_bench_resize",
    "ai_bench_rope",
    "ai_bench_warp",
]

# Same shapes as the RVV/Triton-CPU experiments.
AI_BENCH_SHAPES: dict[str, dict[str, Any]] = {
    "ai_bench_matmul": {"M": 256, "N": 512, "K": 256},
    "ai_bench_dropout": {"n_elements": 1048576, "p": 0.5, "seed": 123},
    "ai_bench_softmax": {"R": 1823, "C": 781},
    "ai_bench_layernorm": {"M": 1151, "N": 8192, "eps": 1e-5},
    "ai_bench_correlation": {"out_channel": 5, "in_channel": 58, "height": 112, "width": 88, "out_shift": 0},
    "ai_bench_resize": {"C": 3, "H": 512, "W": 512, "OH": 1024, "OW": 1024},
    "ai_bench_rope": {"SEQ_LEN": 512, "BATCH_NUM": 16, "HEAD_NUM": 8, "HEAD_DIM": 1024},
    "ai_bench_warp": {"C": 3, "H": 1024, "W": 1024},
}


def _log(msg: str) -> None:
    print(str(msg), file=sys.stderr, flush=True)


def _cuda_ok() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda is not available")


def _geom_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0]
    if not vals:
        return 1.0
    return float(math.exp(sum(math.log(x) for x in vals) / len(vals)))


def _bench_cuda(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    # One warm call to trigger JIT/compile.
    fn()
    torch.cuda.synchronize()
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(iters)):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = float(start.elapsed_time(end))
    return (ms * 1e6) / float(iters)  # ns/iter


def _ensure_artifact(kernel_name: str, *, refresh: bool, cases_limit: int) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / f"{kernel_name}.json"
    if report_path.exists() and not refresh:
        return report_path
    spec_map = {s.name: s for s in coverage_kernel_specs()}
    if kernel_name not in spec_map:
        raise KeyError(f"kernel not found in triton coverage specs: {kernel_name}")
    _log(f"[E5-CUDA:{kernel_name}] generating artifacts via pipeline (refresh={refresh})")
    report = run_pipeline_for_spec(spec_map[kernel_name], out_dir=ARTIFACT_DIR, cases_limit=int(cases_limit))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _descriptor_constexpr(report: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer the full descriptor JSON on disk.
    try:
        dp = report.get("descriptor_path")
        if isinstance(dp, str) and dp and Path(dp).is_file():
            d = json.loads(Path(dp).read_text(encoding="utf-8"))
            launch = d.get("launch") if isinstance(d, dict) else None
            cc = (launch or {}).get("constexpr") if isinstance(launch, dict) else None
            return dict(cc) if isinstance(cc, dict) else {}
    except Exception:
        pass
    try:
        desc = report.get("descriptor")
        launch = (desc or {}).get("launch") if isinstance(desc, dict) else None
        cc = (launch or {}).get("constexpr") if isinstance(launch, dict) else None
        return dict(cc) if isinstance(cc, dict) else {}
    except Exception:
        return {}


def _prep_args_for_cuda_ext(
    *,
    kernel: str,
    io_spec: Dict[str, Any],
    bindings: Mapping[str, Any],
    output_names: Iterable[str],
    launch: CudaLaunch,
    device: str,
    seed: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    torch.manual_seed(int(seed))
    out_set = {str(x) for x in output_names}

    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), dict) else {}
    scalars = io_spec.get("scalars") if isinstance(io_spec.get("scalars"), dict) else {}
    arg_names = io_spec.get("arg_names") if isinstance(io_spec.get("arg_names"), list) else []
    arg_names = [str(x) for x in arg_names]

    def _shape_of(name: str) -> Tuple[int, ...]:
        spec = tensors.get(name) if isinstance(tensors.get(name), dict) else {}
        shp = spec.get("shape") if isinstance(spec.get("shape"), list) else []
        out: list[int] = []
        for d in shp:
            if isinstance(d, str):
                out.append(int(bindings[d]))
            else:
                out.append(int(d))
        return tuple(out)

    def _torch_dtype(dt: str):
        return {
            "f16": torch.float16,
            "f32": torch.float32,
            "i8": torch.int8,
            "i16": torch.int16,
            "i32": torch.int32,
            "i64": torch.int64,
            "u8": torch.uint8,
            "bool": torch.bool,
        }[dt]

    def _alloc_input(name: str, dt: str, shape: Tuple[int, ...]) -> Any:
        if kernel == "ai_bench_correlation" and name in {"src0", "src1"} and dt == "i8" and len(shape) == 3:
            ic, h, w = shape
            n = int(ic) * int(h) * int(w)
            base = torch.arange(n, device=device, dtype=torch.int32)
            vals = (base % (16 if name == "src0" else 35)).to(torch.int8)
            return vals.reshape(shape)
        if kernel in {"ai_bench_resize", "ai_bench_warp"} and name == "src" and dt == "i8" and len(shape) == 3:
            c, h, w = shape
            n = int(c) * int(h) * int(w)
            base = torch.arange(n, device=device, dtype=torch.int32)
            return (base % 17).to(torch.int8).reshape(shape)
        if kernel == "ai_bench_warp" and name == "offset" and dt == "i16":
            return torch.zeros(shape, device=device, dtype=torch.int16)
        if dt == "f32":
            return torch.randn(shape, device=device, dtype=torch.float32)
        if dt == "i8":
            return torch.randint(-128, 127, shape, device=device, dtype=torch.int8)
        if dt == "u8":
            return torch.randint(0, 256, shape, device=device, dtype=torch.uint8)
        if dt == "i16":
            return torch.randint(-32768, 32767, shape, device=device, dtype=torch.int16)
        if dt == "i32":
            return torch.randint(-(2**31), 2**31 - 1, shape, device=device, dtype=torch.int32)
        if dt == "i64":
            return torch.randint(-(2**31), 2**31 - 1, shape, device=device, dtype=torch.int64)
        if dt == "bool":
            return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
        raise RuntimeError(f"unsupported dtype for input alloc: {dt}")

    args: list[Any] = []
    outputs: Dict[str, Any] = {}
    for name in arg_names:
        if name in tensors:
            spec = tensors[name] if isinstance(tensors.get(name), dict) else {}
            dt = str(spec.get("dtype") or "f32")
            shape = _shape_of(name)
            if name in out_set:
                t = torch.empty(shape, device=device, dtype=_torch_dtype(dt))
                outputs[name] = t
                args.append(t)
            else:
                if shape == () and name in bindings and dt in {"f32", "i32", "i64"}:
                    # Scalar tensor modeled as [].
                    val = bindings[name]
                    t = torch.tensor(val, device=device, dtype=_torch_dtype(dt))
                else:
                    t = _alloc_input(name, dt, shape)
                args.append(t.contiguous())
        elif name in scalars:
            dt = str(scalars[name])
            if dt == "f32":
                args.append(float(bindings[name]))
            else:
                args.append(int(bindings[name]))
        else:
            args.append(int(bindings[name]))

    gx, gy, gz = (int(x) for x in launch.grid)
    bx, by, bz = (int(x) for x in launch.block)
    args += [gx, gy, gz, bx, by, bz, int(launch.shared_mem)]
    return args, outputs


def _make_triton_runner(kernel: str, shapes: Mapping[str, Any], device: str) -> Callable[[], None]:
    import triton  # noqa: PLC0415

    if kernel == "ai_bench_matmul":
        from kernels.triton.ops.ai_bench_matmul import ai_bench_matmul_kernel  # noqa: PLC0415

        M, N, K = int(shapes["M"]), int(shapes["N"]), int(shapes["K"])
        a = torch.randn((M, K), device=device, dtype=torch.float32)
        b = torch.randn((K, N), device=device, dtype=torch.float32)
        c = torch.empty((M, N), device=device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

        def run() -> None:
            ai_bench_matmul_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M=64,
                BLOCK_N=16,
                BLOCK_K=16,
            )

        return run

    if kernel == "ai_bench_dropout":
        from kernels.triton.ops.ai_bench_dropout import ai_bench_dropout_kernel  # noqa: PLC0415

        n = int(shapes["n_elements"])
        p = float(shapes.get("p", 0.5))
        seed = int(shapes.get("seed", 123))
        x = torch.randn((n,), device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        block = 256
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        def run() -> None:
            ai_bench_dropout_kernel[grid](x, y, n, float(p), int(seed), BLOCK_SIZE=int(block))

        return run

    if kernel == "ai_bench_softmax":
        from kernels.triton.ops.ai_bench_softmax import ai_bench_softmax_kernel  # noqa: PLC0415

        R, C = int(shapes["R"]), int(shapes["C"])
        x = torch.randn((R, C), device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        block = 1 << (int(C) - 1).bit_length()
        if block > 1024:
            block = 1024

        def run() -> None:
            ai_bench_softmax_kernel[(R,)](y, x, R, C, BLOCK_SIZE=int(block))

        return run

    if kernel == "ai_bench_layernorm":
        from kernels.triton.ops.ai_bench_layernorm import ai_bench_layernorm_fwd_kernel  # noqa: PLC0415

        M, N = int(shapes["M"]), int(shapes["N"])
        eps = float(shapes.get("eps", 1e-5))
        x = torch.randn((M, N), device=device, dtype=torch.float32)
        w = torch.randn((N,), device=device, dtype=torch.float32)
        b = torch.randn((N,), device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        mean = torch.empty((M,), device=device, dtype=torch.float32)
        rstd = torch.empty((M,), device=device, dtype=torch.float32)
        block = 16

        def run() -> None:
            ai_bench_layernorm_fwd_kernel[(M,)](x, y, w, b, mean, rstd, M, N, eps, BLOCK_SIZE=int(block))

        return run

    if kernel == "ai_bench_correlation":
        from kernels.triton.ops.ai_bench_correlation import ai_bench_correlation_kernel  # noqa: PLC0415

        out_channel = int(shapes["out_channel"])
        in_channel = int(shapes["in_channel"])
        height = int(shapes["height"])
        width = int(shapes["width"])
        out_shift = int(shapes.get("out_shift", 0))
        in_size = int(in_channel) * int(height) * int(width)
        vals0 = (torch.arange(in_size, device=device) % 16).to(torch.int8)
        vals1 = (torch.arange(in_size, device=device) % 35).to(torch.int8)
        src0 = vals0.reshape((in_channel, height, width))
        src1 = vals1.reshape((in_channel, height, width))
        out = torch.empty((out_channel, height, width), device=device, dtype=torch.int8)
        block_h, block_w, block_ic = 1, 8, 64
        grid = lambda meta: (
            triton.cdiv(width, meta["BLOCK_W"]),
            triton.cdiv(height, meta["BLOCK_H"]),
            out_channel,
        )

        def run() -> None:
            ai_bench_correlation_kernel[grid](
                src0,
                src1,
                out,
                out_channel,
                in_channel,
                height,
                width,
                int(out_shift),
                BLOCK_H=int(block_h),
                BLOCK_W=int(block_w),
                BLOCK_IC=int(block_ic),
            )

        return run

    if kernel == "ai_bench_resize":
        from kernels.triton.ops.ai_bench_resize import ai_bench_resize_kernel  # noqa: PLC0415

        C, H, W = int(shapes["C"]), int(shapes["H"]), int(shapes["W"])
        in_size = int(C) * int(H) * int(W)
        vals = (torch.arange(in_size, device=device) % 17).to(torch.int8)
        src = vals.reshape((C, H, W))
        out = torch.empty((C, 2 * H, 2 * W), device=device, dtype=torch.int8)
        block_w = 128
        grid = lambda meta: (
            2 * H,
            C,
            triton.cdiv(2 * W, meta["BLOCK_W"]),
        )

        def run() -> None:
            ai_bench_resize_kernel[grid](src, out, C, H, W, BLOCK_W=int(block_w))

        return run

    if kernel == "ai_bench_rope":
        from kernels.triton.ops.ai_bench_rope import ai_bench_rope_fwd_kernel  # noqa: PLC0415

        seq_len = int(shapes["SEQ_LEN"])
        batch_num = int(shapes["BATCH_NUM"])
        head_num = int(shapes["HEAD_NUM"])
        head_dim = int(shapes["HEAD_DIM"])
        assert head_dim % 2 == 0
        x = torch.randn((seq_len, batch_num, head_num, head_dim), device=device, dtype=torch.float32)
        out = torch.empty_like(x)
        cos = torch.randn((seq_len, head_dim // 2), device=device, dtype=torch.float32)
        sin = torch.randn((seq_len, head_dim // 2), device=device, dtype=torch.float32)
        grid = (head_num, batch_num, seq_len)
        block = 32

        def run() -> None:
            ai_bench_rope_fwd_kernel[grid](x, out, cos, sin, seq_len, batch_num, head_num, head_dim, BLOCK_SIZE=int(block))

        return run

    if kernel == "ai_bench_warp":
        from kernels.triton.ops.ai_bench_warp import ai_bench_warp_kernel  # noqa: PLC0415

        C, H, W = int(shapes["C"]), int(shapes["H"]), int(shapes["W"])
        in_size = int(C) * int(H) * int(W)
        vals = (torch.arange(in_size, device=device) % 17).to(torch.int8)
        src = vals.reshape((C, H, W))
        offset = torch.zeros((H, W), device=device, dtype=torch.int16)
        out = torch.empty((C, H, W), device=device, dtype=torch.int8)
        block_w = 128
        grid = lambda meta: (
            H,
            C,
            triton.cdiv(W, meta["BLOCK_W"]),
        )

        def run() -> None:
            ai_bench_warp_kernel[grid](src, offset, out, C, H, W, BLOCK_W=int(block_w))

        return run

    raise KeyError(f"unsupported triton benchmark kernel: {kernel}")


def main() -> None:
    _cuda_ok()

    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", default=[], help="Kernel name (repeatable). Default: AI-Bench8 suite.")
    ap.add_argument("--refresh-artifacts", action="store_true", help="Regenerate IntentIR artifacts via pipeline.")
    ap.add_argument("--cases-limit", type=int, default=1, help="Pipeline cases_limit when refreshing artifacts.")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations (excluded from timing).")
    ap.add_argument("--iters", type=int, default=200, help="Benchmark iterations (timed).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for input generation.")
    ap.add_argument("--device", type=str, default="cuda", help="Device string (default: cuda).")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Output JSON path.")
    args = ap.parse_args()

    wanted = list(args.kernel or []) or list(AI_BENCH_KERNELS)

    meta = {
        "experiment": "E5_triton_gpu_vs_intentir_cuda",
        "torch": str(torch.__version__),
        "triton": None,
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "seed": int(args.seed),
        "kernels": wanted,
    }
    try:
        import triton  # noqa: PLC0415

        meta["triton"] = str(triton.__version__)
    except Exception:
        meta["triton"] = None

    results: list[dict[str, Any]] = []
    for k in wanted:
        _log(f"[E5-CUDA] {k}")
        try:
            report_path = _ensure_artifact(k, refresh=bool(args.refresh_artifacts), cases_limit=int(args.cases_limit))
            report = _load_report(report_path)

            # Build bindings: real shapes + descriptor constexpr (e.g., BLOCK_M/BLOCK_N).
            bindings: Dict[str, Any] = dict(AI_BENCH_SHAPES.get(k, {}))
            bindings.update(_descriptor_constexpr(report))

            intent = IntentFunction.from_json_dict(report["intent"])
            lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

            # Compile CUDA extension once (excluded from timing).
            mod = compile_cuda_extension(kernel_name=lowered.kernel_name, cuda_src=lowered.cuda_src, io_spec=lowered.io_spec)

            ours_args, _ = _prep_args_for_cuda_ext(
                kernel=k,
                io_spec=lowered.io_spec,
                bindings=lowered.bindings,
                output_names=lowered.output_names,
                launch=lowered.launch,
                device=str(args.device),
                seed=int(args.seed),
            )

            def ours_run() -> None:
                mod.launch(*ours_args)

            triton_run = _make_triton_runner(k, AI_BENCH_SHAPES.get(k, {}), str(args.device))

            ours_ns = _bench_cuda(ours_run, warmup=int(args.warmup), iters=int(args.iters))
            triton_ns = _bench_cuda(triton_run, warmup=int(args.warmup), iters=int(args.iters))

            speedup = float(triton_ns) / float(ours_ns) if ours_ns > 0 else 0.0
            results.append(
                {
                    "kernel": k,
                    "status": "OK",
                    "ours": {"ns_per_iter": ours_ns, "launch": {"grid": list(lowered.launch.grid), "block": list(lowered.launch.block)}},
                    "triton": {"ns_per_iter": triton_ns},
                    "speedup_ours_over_triton": speedup,
                }
            )
            _log(f"  ours={ours_ns/1e3:.2f}us  triton={triton_ns/1e3:.2f}us  speedup={speedup:.2f}x")
        except Exception as e:
            results.append({"kernel": k, "status": "FAIL", "error": f"{type(e).__name__}: {e}"})
            _log(f"  FAIL: {type(e).__name__}: {e}")

    ok_rates = [r["speedup_ours_over_triton"] for r in results if r.get("status") == "OK"]
    summary = {
        "ok": sum(1 for r in results if r.get("status") == "OK"),
        "total": len(results),
        "geom_speedup_ours_over_triton": _geom_mean(ok_rates),
        "min_speedup": min(ok_rates) if ok_rates else None,
        "max_speedup": max(ok_rates) if ok_rates else None,
    }

    out = {"meta": meta, "summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

