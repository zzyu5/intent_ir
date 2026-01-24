"""
E5.2 (CUDA): Performance decoupling ablation (freeze vs retune) on NVIDIA GPU.

Goal:
  Reproduce the "freeze schedule vs retune schedule" experiment, but using a
  CUDA backend (IntentIR -> CUDA kernel -> run/bench on GPU), to strengthen the
  portability story beyond RVV/RISC-V.

Scope (MVP):
  - The 8 AI-Bench kernels (ai_bench_*), using the same "real" shapes as the
    external baseline harness.
  - Freeze: use the schedule coming from the recovered IntentIR + descriptor
    constexpr (e.g., BLOCK_M/BLOCK_N/BLOCK_SIZE/BLOCK_W).
  - Retune: try a small set of schedule candidates and pick the fastest by
    measured GPU time.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import coverage_kernel_specs, run_pipeline_for_spec  # noqa: E402
from intent_ir.ir import IntentFunction, ScheduleSketch  # noqa: E402
from backends.cuda.codegen.intentir_to_cuda import CudaLoweringError, lower_intent_to_cuda_kernel  # noqa: E402
from frontends.cuda.runtime import CudaLaunch, compile_cuda_extension  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "full_pipeline_verify"
OUT_DEFAULT = ROOT / "artifacts" / "experiments" / "E5" / "e5_2_portability_vs_perf_cuda.json"

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

# Same "real" shapes as the existing RVV E5.2 script (and AI-Benchmark harness).
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


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ensure_artifact(kernel_name: str, *, cases_limit: int, refresh: bool) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / f"{kernel_name}.json"
    if report_path.exists() and not refresh:
        return report_path
    spec_map = {s.name: s for s in coverage_kernel_specs()}
    if kernel_name not in spec_map:
        raise KeyError(f"kernel not found in triton coverage specs: {kernel_name}")
    spec = spec_map[kernel_name]
    _log(f"[E5.2-CUDA:{kernel_name}] generate artifacts (cases_limit={cases_limit}, refresh={refresh})")
    report = run_pipeline_for_spec(spec, out_dir=ARTIFACT_DIR, cases_limit=int(cases_limit))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _descriptor_constexpr(report: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer the full descriptor JSON on disk (artifacts/full_pipeline_verify/*.descriptor.json)
    # because the in-report `descriptor` is a trimmed summary in newer pipelines.
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


def _geom_mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0]
    if not vals:
        return 1.0
    return float(math.exp(sum(math.log(x) for x in vals) / len(vals)))


def _prep_inputs_torch(
    *,
    kernel: str,
    io_spec: Dict[str, Any],
    bindings: Mapping[str, Any],
    output_names: Iterable[str],
    device: str,
    seed: int,
) -> Tuple[List[Any], Dict[str, Any]]:
    import torch  # noqa: PLC0415

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

    def _alloc_input(name: str, dt: str, shape: Tuple[int, ...]) -> Any:
        if kernel == "ai_bench_correlation" and name in {"src0", "src1"} and dt == "i8" and len(shape) == 3:
            # Match the reference generator (deterministic arange mod).
            ic, h, w = shape
            n = int(ic) * int(h) * int(w)
            base = torch.arange(n, device=device, dtype=torch.int32)
            if name == "src0":
                vals = (base % 16).to(torch.int8)
            else:
                vals = (base % 35).to(torch.int8)
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
                t = torch.empty(shape, device=device, dtype=getattr(torch, {"f32": "float32", "f16": "float16", "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64", "u8": "uint8", "bool": "bool"}[dt]))
                outputs[name] = t
                args.append(t)
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
            # unknown arg treated as int
            args.append(int(bindings[name]))
    return args, outputs


def _bench_launch(mod: Any, args: List[Any], *, warmup: int, iters: int) -> float:
    import torch  # noqa: PLC0415

    for _ in range(int(warmup)):
        mod.launch(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(iters)):
        mod.launch(*args)
    end.record()
    torch.cuda.synchronize()
    ms = float(start.elapsed_time(end))
    return (ms * 1e6) / float(iters)  # ns/iter


def _candidate_schedules(kernel: str, freeze: ScheduleSketch) -> List[ScheduleSketch]:
    """
    Small deterministic search space. We keep this intentionally small for a
    paper-oriented experiment run time.
    """
    cand: list[ScheduleSketch] = []
    if kernel == "ai_bench_matmul":
        xs = [8, 16, 32]
        ys = [8, 16, 32, 64]
        for by in ys:
            for bx in xs:
                if bx * by <= 1024:
                    cand.append(ScheduleSketch(tile_m=by, tile_n=bx))
        return cand
    # 1D kernels: tune block size only.
    blocks = [64, 128, 256, 512, 1024]
    # Softmax/layernorm reductions prefer power-of-two blocks.
    if kernel in {"ai_bench_softmax", "ai_bench_layernorm"}:
        blocks = [128, 256, 512, 1024]
    for b in blocks:
        cand.append(ScheduleSketch(tile_n=b))
    return cand


def _freeze_schedule_from_descriptor(kernel: str, report: Dict[str, Any], intent: IntentFunction, bindings: Dict[str, Any]) -> ScheduleSketch:
    """
    Freeze schedule = schedule recovered by extractor + descriptor constexpr.

    Some kernels (warp/resize) do not currently have tile_n in the recovered
    schedule; for those, we map BLOCK_W -> tile_n.
    """
    sched = intent.schedule or ScheduleSketch()
    cc = _descriptor_constexpr(report)
    # A small heuristic mapping: 1D "W" blocks become tile_n.
    if sched.tile_n is None and "BLOCK_W" in cc and kernel in {"ai_bench_warp", "ai_bench_resize"}:
        return ScheduleSketch(tile_n="BLOCK_W")
    # If the intent already references BLOCK_* names, keep it.
    return ScheduleSketch(tile_m=sched.tile_m, tile_n=sched.tile_n, tile_k=sched.tile_k, vec_width=sched.vec_width, pipeline_depth=sched.pipeline_depth)


def _schedule_to_dict(s: ScheduleSketch) -> Dict[str, Any]:
    return {
        "tile_m": s.tile_m,
        "tile_n": s.tile_n,
        "tile_k": s.tile_k,
        "vec_width": s.vec_width,
        "pipeline_depth": s.pipeline_depth,
        "axis_bindings": dict(getattr(s, "axis_bindings", {}) or {}),
        "vec_axis": getattr(s, "vec_axis", None),
        "parallel_axes": list(getattr(s, "parallel_axes", []) or []),
        "memory_hint": dict(getattr(s, "memory_hint", {}) or {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", default=None, help="repeatable; explicit kernel names to run")
    ap.add_argument("--suite", choices=["ai_bench8"], default="ai_bench8")
    ap.add_argument("--cases-limit", type=int, default=4)
    ap.add_argument("--refresh-artifacts", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=100)
    ap.add_argument("--bench-warmup", type=int, default=10)
    ap.add_argument("--bench-seed", type=int, default=0)
    ap.add_argument("--tune-budget", type=int, default=12, help="max evaluated schedules (excluding freeze)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    if not _cuda_available():
        raise SystemExit("CUDA not available (torch.cuda.is_available() is false)")
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found on PATH (required for torch extension build)")

    wanted = list(args.kernel or []) or list(AI_BENCH_KERNELS)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "experiment": "E5_2_portability_vs_perf_cuda",
        "ts_unix": int(time.time()),
        "kernels": wanted,
        "bench": {"iters": int(args.bench_iters), "warmup": int(args.bench_warmup), "seed": int(args.bench_seed)},
        "results": [],
        "summary": {},
    }

    speedups: list[float] = []
    for k in wanted:
        rp = _ensure_artifact(k, cases_limit=int(args.cases_limit), refresh=bool(args.refresh_artifacts))
        rj = _load_report(rp)
        intent_json = rj.get("intent")
        if not isinstance(intent_json, dict):
            raise RuntimeError(f"missing intent JSON in {rp}")
        intent0 = IntentFunction.from_json_dict(intent_json)

        # Bindings: real shapes + descriptor constexpr.
        bindings: Dict[str, Any] = dict(AI_BENCH_SHAPES.get(k, {}))
        bindings.update(_descriptor_constexpr(rj))

        freeze_sched = _freeze_schedule_from_descriptor(k, rj, intent0, bindings)
        try:
            frozen = lower_intent_to_cuda_kernel(IntentFunction.from_json_dict(intent_json), shape_bindings=bindings, schedule_override=freeze_sched)
        except CudaLoweringError as e:
            report["results"].append({"kernel": k, "status": "UNSUPPORTED", "error": str(e)})
            continue

        # Compile once (code does not depend on block size in this MVP).
        mod = compile_cuda_extension(kernel_name=frozen.kernel_name, cuda_src=frozen.cuda_src, io_spec=frozen.io_spec)

        # Prepare torch inputs once (freeze/retune share them).
        args_torch, outputs = _prep_inputs_torch(
            kernel=k,
            io_spec=frozen.io_spec,
            bindings=frozen.bindings,
            output_names=frozen.output_names,
            device="cuda",
            seed=int(args.bench_seed),
        )

        def _with_launch(a: List[Any], launch: CudaLaunch) -> List[Any]:
            gx, gy, gz = (int(x) for x in launch.grid)
            bx, by, bz = (int(x) for x in launch.block)
            return list(a) + [gx, gy, gz, bx, by, bz, int(launch.shared_mem)]

        freeze_ns = _bench_launch(mod, _with_launch(args_torch, frozen.launch), warmup=int(args.bench_warmup), iters=int(args.bench_iters))

        # Retune: evaluate candidates; keep best.
        candidates = _candidate_schedules(k, freeze_sched)
        # Deterministic truncation to a small budget.
        candidates = candidates[: max(0, int(args.tune_budget))]
        evaluated: list[dict] = []
        best = {"ns_per_iter": freeze_ns, "schedule": freeze_sched, "launch": frozen.launch}
        for s in candidates:
            try:
                cand = lower_intent_to_cuda_kernel(IntentFunction.from_json_dict(intent_json), shape_bindings=bindings, schedule_override=s)
            except Exception:
                continue
            ns = _bench_launch(mod, _with_launch(args_torch, cand.launch), warmup=int(args.bench_warmup), iters=int(args.bench_iters))
            evaluated.append({"schedule": cand.launch.block, "ns_per_iter": ns})
            if ns < float(best["ns_per_iter"]):
                best = {"ns_per_iter": ns, "schedule": s, "launch": cand.launch}

        speedup = float(freeze_ns) / float(best["ns_per_iter"]) if float(best["ns_per_iter"]) > 0 else 1.0
        speedups.append(speedup)

        report["results"].append(
            {
                "kernel": k,
                "status": "OK",
                "freeze": {"schedule": _schedule_to_dict(freeze_sched), "launch": {"grid": list(frozen.launch.grid), "block": list(frozen.launch.block)}, "ns_per_iter": freeze_ns},
                "retune": {"schedule": (_schedule_to_dict(best["schedule"]) if isinstance(best["schedule"], ScheduleSketch) else {}), "launch": {"grid": list(best["launch"].grid), "block": list(best["launch"].block)}, "ns_per_iter": float(best["ns_per_iter"])},
                "speedup": speedup,
                "evaluated": evaluated,
            }
        )

    report["summary"] = {"geom_speedup": _geom_mean(speedups), "n_ok": sum(1 for r in report["results"] if r.get("status") == "OK")}
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
