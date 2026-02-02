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
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Union

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel  # noqa: E402
from frontends.cuda.runtime import CudaLaunch, compile_cuda_extension  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402


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


def _bench_cuda_repeated(fn: Callable[[], None], *, warmup: int, iters: int, repeats: int) -> tuple[float, list[float]]:
    rs = max(1, int(repeats))
    times: list[float] = []
    for _ in range(rs):
        times.append(_bench_cuda(fn, warmup=int(warmup), iters=int(iters)))
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    return median, times


def _bench_cuda_graph_repeated(fn: Callable[[], None], *, warmup: int, iters: int, repeats: int) -> tuple[float, list[float]]:
    """
    Benchmark via CUDA Graph replay to reduce Python launch overhead.

    This is important for very small kernels where Python submission latency can
    dominate the event window and inflate/deflate speedups unfairly.
    """
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for _ in range(int(iters)):
                fn()
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(f"cuda graph capture failed: {type(e).__name__}: {e}") from e

    rs = max(1, int(repeats))
    times: list[float] = []
    for _ in range(rs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        ms = float(start.elapsed_time(end))
        times.append((ms * 1e6) / float(iters))
    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    return median, times


def _bench_cuda_graph_multi_repeated(
    fns: dict[str, Callable[[], None]],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, tuple[float, list[float]]]:
    """
    Benchmark multiple callables via CUDA Graph replay, measuring them in a rotated
    order per repeat to reduce systematic order/clock bias in ablation results.

    Returns: name -> (median_ns_per_iter, ns_per_iter_repeats)
    """
    if not fns:
        return {}

    torch.cuda.synchronize()

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, fn in fns.items():
        fn()
        torch.cuda.synchronize()
        for _ in range(int(warmup)):
            fn()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                for _ in range(int(iters)):
                    fn()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(f"cuda graph capture failed for {name}: {type(e).__name__}: {e}") from e
        graphs[name] = g

    names = list(graphs.keys())
    rs = max(1, int(repeats))
    times: dict[str, list[float]] = {n: [] for n in names}

    for rep in range(rs):
        # Deterministic rotation to avoid giving any single variant a consistent
        # advantage from being measured later (higher clocks) or earlier (cooler GPU).
        shift = rep % len(names)
        order = names[shift:] + names[:shift]
        for n in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[n].replay()
            end.record()
            torch.cuda.synchronize()
            ms = float(start.elapsed_time(end))
            times[n].append((ms * 1e6) / float(iters))

    out: dict[str, tuple[float, list[float]]] = {}
    for n in names:
        ts = times[n]
        ts_sorted = sorted(ts)
        median = ts_sorted[len(ts_sorted) // 2]
        out[n] = (median, ts)
    return out


def _ensure_artifact(kernel_name: str, *, refresh: bool, cases_limit: int) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / f"{kernel_name}.json"
    if report_path.exists() and not refresh:
        return report_path
    # Import pipeline only when needed: the perf benchmark path uses pre-generated
    # artifacts and shouldn't require the (heavier) pipeline/LLM dependencies.
    from pipeline.triton.core import coverage_kernel_specs, run_pipeline_for_spec  # noqa: PLC0415

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
            "i1": torch.bool,
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
        if dt in {"bool", "i1"}:
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


def _scalar(v: Union[int, float, torch.Tensor]) -> Union[int, float]:
    if isinstance(v, torch.Tensor):
        if v.numel() != 1:
            raise ValueError("expected scalar tensor")
        return v.detach().item()
    return v


def _make_triton_runner_from_shared_args(
    kernel: str,
    arg_map: Mapping[str, Any],
    triton_outputs: Mapping[str, torch.Tensor],
) -> Callable[[], None]:
    import triton  # noqa: PLC0415

    if kernel == "ai_bench_matmul":
        from kernels.triton.ops.ai_bench_matmul import ai_bench_matmul_kernel  # noqa: PLC0415

        a = arg_map["a"]
        b = arg_map["b"]
        c = triton_outputs["c"]
        M = int(_scalar(arg_map["M"]))
        N = int(_scalar(arg_map["N"]))
        K = int(_scalar(arg_map["K"]))
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        # The original AI-Bench config (64x16x16) is minimal; but picking a single
        # "better" config can be GPU-sensitive. Use a tiny fixed candidate set and
        # pick the best once (graph-mode microbench), so Triton is a strong baseline
        # without turning this experiment into a full autotune study.
        candidates = [
            # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
            (64, 16, 16, 4, 3),
            (64, 32, 16, 4, 3),
        ]
        run_fns: dict[str, Callable[[], None]] = {}
        cand_meta: dict[str, dict[str, int]] = {}
        for bm, bn, bk, warps, stages in candidates:
            name = f"bm{bm}_bn{bn}_bk{bk}_w{warps}_s{stages}"

            def _make_run(bm=bm, bn=bn, bk=bk, warps=warps, stages=stages) -> Callable[[], None]:
                def _run() -> None:
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
                        BLOCK_M=int(bm),
                        BLOCK_N=int(bn),
                        BLOCK_K=int(bk),
                        num_warps=int(warps),
                        num_stages=int(stages),
                    )

                return _run

            run_fns[name] = _make_run()
            cand_meta[name] = {
                "BLOCK_M": int(bm),
                "BLOCK_N": int(bn),
                "BLOCK_K": int(bk),
                "num_warps": int(warps),
                "num_stages": int(stages),
            }

        # NOTE: This is intentionally more stable than the main benchmark's
        # per-kernel repeats: the chosen Triton baseline must not drift between
        # runs, otherwise speedups become meaningless. We benchmark all
        # candidates together (rotated order per repeat) and select the best
        # by median graph-replay time.
        timing = _bench_cuda_graph_multi_repeated(run_fns, warmup=10, iters=100, repeats=5)
        best_name = min(timing.keys(), key=lambda n: float(timing[n][0]))
        best = cand_meta[best_name]
        block_m, block_n, block_k = best["BLOCK_M"], best["BLOCK_N"], best["BLOCK_K"]
        num_warps, num_stages = best["num_warps"], best["num_stages"]

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
                BLOCK_M=int(block_m),
                BLOCK_N=int(block_n),
                BLOCK_K=int(block_k),
                num_warps=int(num_warps),
                num_stages=int(num_stages),
            )

        run._intentir_triton_meta = {
            "selected_config": best,
            "candidate_median_ns_per_iter": {k: float(v[0]) for k, v in timing.items()},
        }
        return run

    if kernel == "ai_bench_dropout":
        from kernels.triton.ops.ai_bench_dropout import ai_bench_dropout_kernel  # noqa: PLC0415

        x = arg_map["X"]
        y = triton_outputs["Out"]
        n = int(_scalar(arg_map.get("n_elements", int(x.numel()))))
        p = float(_scalar(arg_map["p"]))
        seed = int(_scalar(arg_map["seed"]))
        block = 256
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        def run() -> None:
            ai_bench_dropout_kernel[grid](x, y, n, float(p), int(seed), BLOCK_SIZE=int(block))

        return run

    if kernel == "ai_bench_softmax":
        from kernels.triton.ops.ai_bench_softmax import ai_bench_softmax_kernel  # noqa: PLC0415

        x = arg_map["in_ptr"]
        y = triton_outputs["out_ptr"]
        R = int(_scalar(arg_map["R"]))
        C = int(_scalar(arg_map["C"]))
        block = 1 << (int(C) - 1).bit_length()
        if block > 1024:
            block = 1024

        def run() -> None:
            ai_bench_softmax_kernel[(R,)](y, x, R, C, BLOCK_SIZE=int(block))

        return run

    if kernel == "ai_bench_layernorm":
        from kernels.triton.ops.ai_bench_layernorm import ai_bench_layernorm_fwd_kernel  # noqa: PLC0415

        x = arg_map["X"]
        w = arg_map["W"]
        b = arg_map["B"]
        y = triton_outputs["Y"]
        mean = triton_outputs["Mean"]
        rstd = triton_outputs["Rstd"]
        M = int(_scalar(arg_map["M"]))
        N = int(_scalar(arg_map["N"]))
        eps = float(_scalar(arg_map.get("eps", 1e-5)))
        # Stronger baseline than BLOCK_SIZE=16 to avoid a misleadingly weak Triton ref.
        block = 1024
        # Empirically best for this benchmark shape; higher warps increase overhead.
        num_warps = 4

        def run() -> None:
            ai_bench_layernorm_fwd_kernel[(M,)](x, y, w, b, mean, rstd, M, N, eps, BLOCK_SIZE=int(block), num_warps=int(num_warps))

        return run

    if kernel == "ai_bench_correlation":
        from kernels.triton.ops.ai_bench_correlation import ai_bench_correlation_kernel  # noqa: PLC0415

        src0 = arg_map["src0"]
        src1 = arg_map["src1"]
        out = triton_outputs["out"]
        out_channel = int(_scalar(arg_map["out_channel"]))
        in_channel = int(_scalar(arg_map["in_channel"]))
        height = int(_scalar(arg_map["height"]))
        width = int(_scalar(arg_map["width"]))
        out_shift = int(_scalar(arg_map.get("out_shift", 0)))
        # Stronger baseline: bigger spatial tiles (the default 1x8 is extremely underutilized).
        block_h, block_w, block_ic = 4, 32, 64
        num_warps = 4
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
                num_warps=int(num_warps),
            )

        return run

    if kernel == "ai_bench_resize":
        from kernels.triton.ops.ai_bench_resize import ai_bench_resize_kernel  # noqa: PLC0415

        src = arg_map["src"]
        out = triton_outputs["out"]
        C = int(_scalar(arg_map["C"]))
        H = int(_scalar(arg_map["H"]))
        W = int(_scalar(arg_map["W"]))
        block_w = 512
        num_warps = 4
        grid = lambda meta: (
            2 * H,
            C,
            triton.cdiv(2 * W, meta["BLOCK_W"]),
        )

        def run() -> None:
            ai_bench_resize_kernel[grid](src, out, C, H, W, BLOCK_W=int(block_w), num_warps=int(num_warps))

        return run

    if kernel == "ai_bench_rope":
        from kernels.triton.ops.ai_bench_rope import ai_bench_rope_fwd_kernel  # noqa: PLC0415

        x = arg_map["input"]
        cos = arg_map["cos"]
        sin = arg_map["sin"]
        out = triton_outputs["output"]
        seq_len = int(_scalar(arg_map["SEQ_LEN"]))
        batch_num = int(_scalar(arg_map["BATCH_NUM"]))
        head_num = int(_scalar(arg_map["HEAD_NUM"]))
        head_dim = int(_scalar(arg_map["HEAD_DIM"]))
        grid = (head_num, batch_num, seq_len)
        block = 32

        def run() -> None:
            ai_bench_rope_fwd_kernel[grid](x, out, cos, sin, seq_len, batch_num, head_num, head_dim, BLOCK_SIZE=int(block))

        return run

    if kernel == "ai_bench_warp":
        from kernels.triton.ops.ai_bench_warp import ai_bench_warp_kernel  # noqa: PLC0415

        src = arg_map["src"]
        offset = arg_map["offset"]
        out = triton_outputs["out"]
        C = int(_scalar(arg_map.get("C", int(src.shape[0]))))
        H = int(_scalar(arg_map.get("H", int(src.shape[1]))))
        W = int(_scalar(arg_map.get("W", int(src.shape[2]))))
        # Keep BLOCK_W<=128: the benchmark kernel casts indices to int8.
        block_w = 128
        num_warps = 2
        grid = lambda meta: (
            H,
            C,
            triton.cdiv(W, meta["BLOCK_W"]),
        )

        def run() -> None:
            ai_bench_warp_kernel[grid](src, offset, out, C, H, W, BLOCK_W=int(block_w), num_warps=int(num_warps))

        return run

    raise KeyError(f"unsupported triton benchmark kernel: {kernel}")


def _check_outputs_close(
    *,
    kernel: str,
    ours: Mapping[str, torch.Tensor],
    triton: Mapping[str, torch.Tensor],
    allow_tf32: bool,
) -> None:
    for name, a in ours.items():
        if name not in triton:
            raise AssertionError(f"{kernel}: missing triton output buffer: {name}")
        b = triton[name]
        if tuple(a.shape) != tuple(b.shape):
            raise AssertionError(f"{kernel}: output shape mismatch for {name}: ours={tuple(a.shape)} triton={tuple(b.shape)}")
        if a.dtype != b.dtype:
            raise AssertionError(f"{kernel}: output dtype mismatch for {name}: ours={a.dtype} triton={b.dtype}")

        if a.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            if allow_tf32:
                atol, rtol = 1e-1, 1e-2
            elif a.dtype == torch.float16:
                atol, rtol = 3e-3, 3e-3
            else:
                atol, rtol = 1e-3, 1e-3
            if not torch.allclose(a, b, atol=atol, rtol=rtol):
                diff = (a - b).abs()
                max_abs = float(diff.max().item())
                denom = b.abs().max().item()
                max_rel = float(max_abs / (float(denom) + 1e-8))
                raise AssertionError(
                    f"{kernel}: output mismatch for {name} (atol={atol}, rtol={rtol}): max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
                )
        else:
            if not torch.equal(a, b):
                neq = int((a != b).sum().item())
                raise AssertionError(f"{kernel}: output mismatch for {name}: {neq} elements differ")


def main() -> None:
    _cuda_ok()

    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", default=[], help="Kernel name (repeatable). Default: AI-Bench8 suite.")
    ap.add_argument("--refresh-artifacts", action="store_true", help="Regenerate IntentIR artifacts via pipeline.")
    ap.add_argument("--cases-limit", type=int, default=1, help="Pipeline cases_limit when refreshing artifacts.")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations (excluded from timing).")
    ap.add_argument("--iters", type=int, default=200, help="Benchmark iterations (timed).")
    ap.add_argument("--repeats", type=int, default=5, help="Repeat measurements and report median ns/iter.")
    ap.add_argument(
        "--bench-mode",
        type=str,
        default="event",
        choices=["event", "graph"],
        help="Benchmark mode: 'event' (default) or 'graph' (CUDA Graph replay; less Python overhead).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for input generation.")
    ap.add_argument("--device", type=str, default="cuda", help="Device string (default: cuda).")
    ap.add_argument(
        "--fast-math",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile IntentIR CUDA kernels with --use_fast_math (default: enabled).",
    )
    ap.add_argument(
        "--ablation",
        action="store_true",
        help="Run evidence ablation: compare evidence-gated specialization vs selected evidence-off modes.",
    )
    ap.add_argument(
        "--ablation-modes",
        type=str,
        default="evidence_on,contract_off",
        help="Comma-separated modes for --ablation. Supported: evidence_on, dispatch_off, contract_off. Default: evidence_on,contract_off.",
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Output JSON path.")
    ap.add_argument(
        "--bind",
        action="append",
        default=[],
        help="Override shape bindings as KEY=VAL (repeatable). VAL parsed as int/float/str.",
    )
    args = ap.parse_args()

    # This experiment compares *kernel runtime*; enabling fast-math for our kernels
    # makes the comparison fairer because Triton GPU kernels typically use fast math
    # for transcendental ops as well (e.g., exp in softmax).
    os.environ["INTENTIR_CUDA_USE_FAST_MATH"] = "1" if bool(args.fast_math) else "0"
    if not os.getenv("TORCH_CUDA_ARCH_LIST"):
        try:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass

    wanted = list(args.kernel or []) or list(AI_BENCH_KERNELS)

    meta = {
        "experiment": "E5_triton_gpu_vs_intentir_cuda",
        "torch": str(torch.__version__),
        "triton": None,
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "intentir_cuda_fast_math": bool(args.fast_math),
        "torch_cuda_arch_list": os.getenv("TORCH_CUDA_ARCH_LIST"),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "repeats": int(args.repeats),
        "bench_mode": str(args.bench_mode),
        "seed": int(args.seed),
        "kernels": wanted,
    }
    if bool(args.ablation):
        modes = [m.strip() for m in str(args.ablation_modes).split(",") if m.strip()]
        bad = [m for m in modes if m not in {"evidence_on", "dispatch_off", "contract_off"}]
        if bad:
            raise SystemExit(f"--ablation-modes has unknown entries: {bad}")
        if not modes:
            raise SystemExit("--ablation-modes must not be empty when --ablation is set")
        meta["ablation"] = {
            "enabled": True,
            "modes": modes,
            "contract_off_level": "OUT_OF_SCOPE",
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
            # Performance experiment (paper baseline): specialize resolved dims as compile-time constants
            # to reduce overhead and let nvcc aggressively optimize/unroll.
            #
            # NOTE: In ablation mode we still keep this baseline, and override per-variant bindings
            # (e.g., contract_off forces specialize dims off) so comparisons stay apples-to-apples.
            bindings.setdefault("CUDA_SPECIALIZE_DIMS", 1)
            # Optional overrides for quick tuning experiments.
            for item in list(args.bind or []):
                if "=" not in str(item):
                    continue
                key, raw = str(item).split("=", 1)
                key = key.strip()
                raw = raw.strip()
                if not key:
                    continue
                val: Any = raw
                try:
                    val = int(raw, 0)
                except Exception:
                    try:
                        val = float(raw)
                    except Exception:
                        val = raw
                bindings[key] = val
            # For fair comparison, allow TF32 on matmul (Triton tl.dot uses TF32 on modern GPUs).
            if k == "ai_bench_matmul":
                bindings.setdefault("ALLOW_TF32", 1)

            def _intent_contract_level(it: Any) -> str | None:
                try:
                    if not isinstance(it, dict):
                        return None
                    meta_j = it.get("meta")
                    if not isinstance(meta_j, dict):
                        return None
                    cv2 = meta_j.get("contract_v2")
                    if isinstance(cv2, dict) and isinstance(cv2.get("level"), str):
                        return str(cv2.get("level"))
                    return None
                except Exception:
                    return None

            def _intent_with_canonical_shapes(it: dict[str, Any]) -> dict[str, Any]:
                # For ablations we want auto-specialization to engage based on "this
                # benchmark's canonical shapes", even if the artifact didn't record
                # canonical_shapes. This keeps the comparison focused on contract_v2
                # gating rather than missing metadata.
                cs_src = dict(AI_BENCH_SHAPES.get(k, {}))
                cs: dict[str, int] = {}
                for kk, vv in cs_src.items():
                    if isinstance(vv, bool):
                        cs[str(kk)] = int(vv)
                    elif isinstance(vv, int):
                        cs[str(kk)] = int(vv)
                    elif isinstance(vv, float) and float(vv).is_integer():
                        cs[str(kk)] = int(vv)
                it2: dict[str, Any] = dict(it)
                meta_j = it2.get("meta")
                meta2: dict[str, Any] = dict(meta_j) if isinstance(meta_j, dict) else {}
                if cs:
                    meta2["canonical_shapes"] = cs
                it2["meta"] = meta2
                return it2

            def _intent_with_cert_evidence(it: dict[str, Any], report_json: dict[str, Any]) -> dict[str, Any]:
                """
                Ensure `intent.meta` contains the evidence summaries consumed by the backend.

                Some reports may embed a fresh `certificate_v2` but have a stale `intent.meta`
                (e.g., only `contract_v2`/`canonical_shapes`). For the GPU experiments we want
                evidence-guided codegen/dispatch, so we attach `access_witness` and
                `schedule_hints_v2` from CertificateV2 on-the-fly.
                """
                if not isinstance(it, dict):
                    return it
                meta_j = it.get("meta")
                meta2: dict[str, Any] = dict(meta_j) if isinstance(meta_j, dict) else {}

                need_aw = not isinstance(meta2.get("access_witness"), dict)
                need_sh = not isinstance(meta2.get("schedule_hints_v2"), dict)
                if not (need_aw or need_sh):
                    return it

                cv2 = report_json.get("certificate_v2")
                if not isinstance(cv2, dict):
                    return it

                # Shape bindings for resolving simple stride witnesses.
                shape_bindings: dict[str, int] = {}
                cs = meta2.get("canonical_shapes")
                if isinstance(cs, dict):
                    for kk, vv in cs.items():
                        if isinstance(kk, str) and isinstance(vv, (int, float)) and float(vv).is_integer():
                            shape_bindings[str(kk)] = int(vv)
                if not shape_bindings:
                    for kk, vv in (AI_BENCH_SHAPES.get(k, {}) or {}).items():
                        if isinstance(kk, str) and isinstance(vv, (int, float)) and float(vv).is_integer():
                            shape_bindings[str(kk)] = int(vv)

                try:
                    from frontends.common.access_witness import build_stride_summary  # noqa: PLC0415

                    ss = build_stride_summary(cv2, shape_bindings=shape_bindings)
                    j = ss.to_json_dict()
                    tp = j.get("tensor_penalty") if isinstance(j.get("tensor_penalty"), dict) else {}
                    top = sorted(((str(kk), float(vv)) for kk, vv in tp.items()), key=lambda kv: kv[1], reverse=True)[:8]

                    if need_aw:
                        meta2["access_witness"] = {
                            "dominant_axis": j.get("dominant_axis"),
                            "dominant_range": j.get("dominant_range"),
                            "dominant_range_len": j.get("dominant_range_len"),
                            "has_contiguous_range": bool(j.get("has_contiguous_range")),
                            "tensor_penalty_top": top,
                            "notes": list(j.get("notes") or []) if isinstance(j.get("notes"), list) else [],
                        }

                    if need_sh:
                        sh = cv2.get("schedule_hints") if isinstance(cv2.get("schedule_hints"), dict) else {}
                        meta2["schedule_hints_v2"] = {
                            "tile_hints": list(sh.get("tile_hints") or []) if isinstance(sh.get("tile_hints"), list) else [],
                            "symbol_ranges": dict(sh.get("symbol_ranges") or {}) if isinstance(sh.get("symbol_ranges"), dict) else {},
                        }
                except Exception:
                    return it

                it2: dict[str, Any] = dict(it)
                it2["meta"] = meta2
                return it2

            def _intent_with_contract_level(it: dict[str, Any], level: str) -> dict[str, Any]:
                it2: dict[str, Any] = dict(it)
                meta_j = it2.get("meta")
                meta2: dict[str, Any] = dict(meta_j) if isinstance(meta_j, dict) else {}
                cv2 = meta2.get("contract_v2")
                cv2_2: dict[str, Any] = dict(cv2) if isinstance(cv2, dict) else {}
                cv2_2["level"] = str(level)
                meta2["contract_v2"] = cv2_2
                it2["meta"] = meta2
                return it2

            def _run_ours(it_json: dict[str, Any], bnd: dict[str, Any]) -> tuple[dict[str, Any], Callable[[], None], dict[str, Any]]:
                intent = IntentFunction.from_json_dict(it_json)
                lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bnd)
                mod = compile_cuda_extension(kernel_name=lowered.kernel_name, cuda_src=lowered.cuda_src, io_spec=lowered.io_spec)
                ours_args, ours_outputs = _prep_args_for_cuda_ext(
                    kernel=k,
                    io_spec=lowered.io_spec,
                    bindings=lowered.bindings,
                    output_names=lowered.output_names,
                    launch=lowered.launch,
                    device=str(args.device),
                    seed=int(args.seed),
                )

                def run() -> None:
                    mod.launch(*ours_args)

                run()
                torch.cuda.synchronize()

                sel_variant: int | None = None
                sel_tag: str | None = None
                try:
                    if hasattr(mod, "selected_variant"):
                        sel_variant = int(mod.selected_variant())
                    if hasattr(mod, "selected_tag"):
                        sel_tag = str(mod.selected_tag())
                except Exception:
                    sel_variant = None
                    sel_tag = None

                rec = {
                    "ns_per_iter": None,
                    "ns_per_iter_repeats": None,
                    "launch": {"grid": list(lowered.launch.grid), "block": list(lowered.launch.block)},
                    "host_launch": bool(lowered.io_spec.get("host_launch")),
                    "selected_variant": sel_variant,
                    "selected_tag": sel_tag,
                }
                ctx = {"mod": mod, "outputs": ours_outputs, "lowered": lowered, "args": ours_args}
                return rec, run, ctx

            intent_json = report.get("intent")
            if not isinstance(intent_json, dict):
                raise RuntimeError("report missing intent json")

            intent_contract_v2_level = _intent_contract_level(intent_json)

            if bool(args.ablation):
                intent_json = _intent_with_canonical_shapes(intent_json)
            intent_json = _intent_with_cert_evidence(intent_json, report)

            # Evidence-on (default): keep the artifact intent.meta (contract_v2 etc).
            ours_rec, ours_run, ours_ctx = _run_ours(intent_json, dict(bindings))

            # Build Triton runner from the evidence-on shared args.
            lowered0 = ours_ctx["lowered"]
            ours_args0 = ours_ctx["args"]
            ours_outputs0 = ours_ctx["outputs"]
            arg_names0 = lowered0.io_spec.get("arg_names") if isinstance(lowered0.io_spec, dict) else None
            arg_names0 = [str(x) for x in (arg_names0 or [])]
            arg_map0 = {name: ours_args0[i] for i, name in enumerate(arg_names0)}
            triton_outputs = {name: torch.empty_like(t) for name, t in ours_outputs0.items()}
            triton_run = _make_triton_runner_from_shared_args(k, arg_map0, triton_outputs)

            # Sanity correctness check for evidence-on.
            triton_run()
            torch.cuda.synchronize()
            _check_outputs_close(kernel=k, ours=ours_outputs0, triton=triton_outputs, allow_tf32=(k == "ai_bench_matmul"))

            # Evidence-off (ablation): disable host-dispatch *selection* (seed-only) while keeping
            # evidence/specialization on. This keeps the generated kernel variants identical to the
            # "quick" path and isolates the benefit of host-side selection.
            ours_dispatch_off_rec: dict[str, Any] | None = None
            ours_dispatch_off_run: Callable[[], None] | None = None
            if bool(args.ablation) and "dispatch_off" in (meta.get("ablation") or {}).get("modes", []):
                bnd2 = dict(bindings)
                bnd2["CUDA_HOST_DISPATCH"] = 1
                bnd2["CUDA_HOST_DISPATCH_SELECT"] = 0
                ours_dispatch_off_rec, ours_dispatch_off_run, ours_disp_ctx = _run_ours(intent_json, bnd2)
                ours_outputs_disp = ours_disp_ctx["outputs"]
                _check_outputs_close(kernel=k, ours=ours_outputs_disp, triton=triton_outputs, allow_tf32=(k == "ai_bench_matmul"))

            # Evidence-off (ablation): force contract_v2 to OUT_OF_SCOPE and re-lower.
            ours_contract_off_rec: dict[str, Any] | None = None
            ours_contract_off_run: Callable[[], None] | None = None
            if bool(args.ablation) and "contract_off" in (meta.get("ablation") or {}).get("modes", []):
                intent_off = _intent_with_contract_level(intent_json, "OUT_OF_SCOPE")
                bnd3 = dict(bindings)
                bnd3["CUDA_SPECIALIZE_DIMS"] = 0
                bnd3.setdefault("CUDA_AUTO_SPECIALIZE_DIMS", 1)
                ours_contract_off_rec, ours_contract_off_run, ours_off_ctx = _run_ours(intent_off, bnd3)
                ours_outputs_off = ours_off_ctx["outputs"]
                _check_outputs_close(kernel=k, ours=ours_outputs_off, triton=triton_outputs, allow_tf32=(k == "ai_bench_matmul"))

            if str(args.bench_mode) == "graph":
                try:
                    if bool(args.ablation):
                        fns: dict[str, Callable[[], None]] = {"ours": ours_run, "triton": triton_run}
                        if ours_dispatch_off_run is not None:
                            fns["dispatch_off"] = ours_dispatch_off_run
                        if ours_contract_off_run is not None:
                            fns["contract_off"] = ours_contract_off_run
                        multi = _bench_cuda_graph_multi_repeated(
                            fns, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                        )
                        ours_ns, ours_reps = multi["ours"]
                        triton_ns, triton_reps = multi["triton"]
                        ours_dispatch_off_ns, ours_dispatch_off_reps = multi.get("dispatch_off", (None, None))
                        ours_contract_off_ns, ours_contract_off_reps = multi.get("contract_off", (None, None))
                    else:
                        ours_ns, ours_reps = _bench_cuda_graph_repeated(
                            ours_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                        )
                        triton_ns, triton_reps = _bench_cuda_graph_repeated(
                            triton_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                        )
                        ours_dispatch_off_ns = None
                        ours_dispatch_off_reps = None
                        ours_contract_off_ns = None
                        ours_contract_off_reps = None
                except Exception as e:
                    _log(f"  WARN: bench_mode=graph failed ({type(e).__name__}: {e}); falling back to event mode")
                    ours_ns, ours_reps = _bench_cuda_repeated(ours_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats))
                    triton_ns, triton_reps = _bench_cuda_repeated(
                        triton_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                    )
                    ours_dispatch_off_ns = None
                    ours_dispatch_off_reps = None
                    if bool(args.ablation) and ours_dispatch_off_run is not None:
                        ours_dispatch_off_ns, ours_dispatch_off_reps = _bench_cuda_repeated(
                            ours_dispatch_off_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                        )
                    ours_contract_off_ns = None
                    ours_contract_off_reps = None
                    if bool(args.ablation) and ours_contract_off_run is not None:
                        ours_contract_off_ns, ours_contract_off_reps = _bench_cuda_repeated(
                            ours_contract_off_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                        )
            else:
                ours_ns, ours_reps = _bench_cuda_repeated(ours_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats))
                triton_ns, triton_reps = _bench_cuda_repeated(
                    triton_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                )
                ours_dispatch_off_ns = None
                ours_dispatch_off_reps = None
                if bool(args.ablation) and ours_dispatch_off_run is not None:
                    ours_dispatch_off_ns, ours_dispatch_off_reps = _bench_cuda_repeated(
                        ours_dispatch_off_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                    )
                ours_contract_off_ns = None
                ours_contract_off_reps = None
                if bool(args.ablation) and ours_contract_off_run is not None:
                    ours_contract_off_ns, ours_contract_off_reps = _bench_cuda_repeated(
                        ours_contract_off_run, warmup=int(args.warmup), iters=int(args.iters), repeats=int(args.repeats)
                    )

            ours_rec["ns_per_iter"] = ours_ns
            ours_rec["ns_per_iter_repeats"] = ours_reps
            if ours_dispatch_off_rec is not None:
                ours_dispatch_off_rec["ns_per_iter"] = ours_dispatch_off_ns
                ours_dispatch_off_rec["ns_per_iter_repeats"] = ours_dispatch_off_reps
            if ours_contract_off_rec is not None:
                ours_contract_off_rec["ns_per_iter"] = ours_contract_off_ns
                ours_contract_off_rec["ns_per_iter_repeats"] = ours_contract_off_reps

            speedup = float(triton_ns) / float(ours_ns) if ours_ns > 0 else 0.0
            speedup_dispatch_off = None
            slowdown_dispatch_off = None
            if bool(args.ablation) and ours_dispatch_off_ns is not None and ours_dispatch_off_ns > 0:
                speedup_dispatch_off = float(triton_ns) / float(ours_dispatch_off_ns)
                slowdown_dispatch_off = float(ours_dispatch_off_ns) / float(ours_ns) if ours_ns > 0 else None
            speedup_contract_off = None
            speedup_gain = None
            if bool(args.ablation) and ours_contract_off_ns is not None and ours_contract_off_ns > 0:
                speedup_contract_off = float(triton_ns) / float(ours_contract_off_ns)
                speedup_gain = float(ours_contract_off_ns) / float(ours_ns) if ours_ns > 0 else None
            triton_rec: dict[str, Any] = {"ns_per_iter": triton_ns, "ns_per_iter_repeats": triton_reps}
            triton_meta = getattr(triton_run, "_intentir_triton_meta", None)
            if isinstance(triton_meta, dict) and triton_meta:
                triton_rec["meta"] = triton_meta

            results.append(
                {
                    "kernel": k,
                    "status": "OK",
                    "intent": {
                        "contract_v2_level": intent_contract_v2_level,
                    },
                    "ours": ours_rec,
                    "ours_dispatch_off": ours_dispatch_off_rec,
                    "ours_contract_off": ours_contract_off_rec,
                    "triton": triton_rec,
                    "speedup_ours_over_triton": speedup,
                    "speedup_ours_dispatch_off_over_triton": speedup_dispatch_off,
                    "speedup_ours_contract_off_over_triton": speedup_contract_off,
                    "slowdown_dispatch_off_over_ours": slowdown_dispatch_off,
                    "slowdown_contract_off_over_ours": speedup_gain,
                }
            )
            _log(f"  ours={ours_ns/1e3:.2f}us  triton={triton_ns/1e3:.2f}us  speedup={speedup:.2f}x")
        except Exception as e:
            results.append({"kernel": k, "status": "FAIL", "error": f"{type(e).__name__}: {e}"})
            _log(f"  FAIL: {type(e).__name__}: {e}")

    ok_rates = [r["speedup_ours_over_triton"] for r in results if r.get("status") == "OK"]
    ok_rates_dispatch_off = [
        r["speedup_ours_dispatch_off_over_triton"]
        for r in results
        if r.get("status") == "OK" and isinstance(r.get("speedup_ours_dispatch_off_over_triton"), (int, float))
    ]
    ok_rates_contract_off = [
        r["speedup_ours_contract_off_over_triton"]
        for r in results
        if r.get("status") == "OK" and isinstance(r.get("speedup_ours_contract_off_over_triton"), (int, float))
    ]
    summary = {
        "ok": sum(1 for r in results if r.get("status") == "OK"),
        "total": len(results),
        "geom_speedup_ours_over_triton": _geom_mean(ok_rates),
        "geom_speedup_ours_dispatch_off_over_triton": _geom_mean(ok_rates_dispatch_off) if ok_rates_dispatch_off else None,
        "geom_speedup_ours_contract_off_over_triton": _geom_mean(ok_rates_contract_off) if ok_rates_contract_off else None,
        "min_speedup": min(ok_rates) if ok_rates else None,
        "max_speedup": max(ok_rates) if ok_rates else None,
    }

    out = {"meta": meta, "summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
