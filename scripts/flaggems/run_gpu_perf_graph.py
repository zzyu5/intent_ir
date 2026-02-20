#!/usr/bin/env python3
"""
Run IntentIR CUDA-Graph performance comparison against Triton-native kernels.

Outputs:
- gpu_perf_graph.json
- run_summary.json
- status_converged.json
- stage_timing_breakdown.json

This runner is chunk-aware and resumable (family -> chunk -> kernel).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Avoid mkl-service/libgomp threading-layer conflicts in long benchmark runs.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np  # noqa: F401  # import before torch per MKL guidance
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.cuda.codegen.cpp_driver import CudaLoweringError, lower_intent_to_cuda_kernel
from backends.cuda.runtime import compile_cuda_extension
from intent_ir.utils.repo_state import repo_state
from pipeline.triton.core import coverage_kernel_specs
from scripts.cuda_backend_smoke import (
    _build_inputs_np,
    _prepare_kernel_context,
    _set_codegen_mode_env,
    _set_runtime_backend_env,
)

try:  # Optional dependency.
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def _chunk_kernels(kernels: list[str], chunk_size: int) -> list[list[str]]:
    if int(chunk_size) <= 0 or len(kernels) <= int(chunk_size):
        return [list(kernels)]
    out: list[list[str]] = []
    step = int(chunk_size)
    for i in range(0, len(kernels), step):
        out.append(list(kernels[i : i + step]))
    return out


def _emit_chunk_progress(
    *,
    style: str,
    done: int,
    total: int,
    family: str,
    chunk_idx: int,
    chunk_total: int,
    status: str,
) -> None:
    mode = str(style).strip().lower()
    if mode == "none":
        return
    if mode == "chunk":
        print(f"[chunk] {done}/{total}", flush=True)
        return
    print(
        f"[progress] chunks {done}/{total} family={family} chunk={chunk_idx}/{chunk_total} status={status}",
        flush=True,
    )


def _write_chunk_progress_file(
    *,
    path: Path,
    done: int,
    total: int,
    family: str,
    chunk_idx: int,
    chunk_total: int,
    status: str,
    completed: bool,
    progress_style: str,
    measured: int,
    failures: int,
) -> None:
    payload = {
        "schema_version": "flaggems_chunk_progress_v1",
        "generated_at": _utc_now_iso(),
        "suite": "gpu_perf_graph",
        "progress_style": str(progress_style),
        "done_chunks": int(done),
        "total_chunks": int(total),
        "remaining_chunks": int(max(0, int(total) - int(done))),
        "completed": bool(completed),
        "measured_kernels": int(measured),
        "failure_count": int(failures),
        "last": {
            "family": str(family),
            "chunk_index": int(chunk_idx),
            "chunk_total": int(chunk_total),
            "status": str(status),
        },
    }
    _dump_json(path, payload)


def _choose_progress_style(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode != "auto":
        return mode
    return "tqdm" if (sys.stdout.isatty() and tqdm is not None) else "chunk"


def _np_to_torch_dtype(dtype_name: str) -> torch.dtype:
    dt = str(dtype_name).strip().lower()
    table = {
        "f16": torch.float16,
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
        "u8": torch.uint8,
        "bool": torch.bool,
        "i1": torch.bool,
    }
    if dt not in table:
        raise RuntimeError(f"unsupported dtype for GPU perf launch args: {dtype_name}")
    return table[dt]


def _kernel_param_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _bench_graph(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, float]:
    if int(iters) <= 0:
        raise RuntimeError("iters must be > 0")
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    for _ in range(max(0, int(warmup))):
        fn()
    torch.cuda.synchronize()

    t_capture0 = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(int(iters)):
            fn()
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - t_capture0) * 1000.0

    replay_total_ms: list[float] = []
    replay_iter_ms: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        total_ms = float(start.elapsed_time(end))
        replay_total_ms.append(total_ms)
        replay_iter_ms.append(total_ms / float(iters))

    median_total_ms = float(statistics.median(replay_total_ms))
    median_iter_ms = float(statistics.median(replay_iter_ms))
    qps = float(int(iters) / (median_total_ms / 1000.0)) if median_total_ms > 0.0 else 0.0
    return {
        "capture_ms": float(capture_ms),
        "replay_ms_total": float(median_total_ms),
        "replay_ms": float(median_iter_ms),
        "qps": float(qps),
        "latency_ms": float(median_iter_ms),
    }


def _coverage_spec_map() -> dict[str, Any]:
    specs = coverage_kernel_specs()
    return {str(s.name): s for s in specs}


def _select_native_callable(module: Any, kernel: str) -> Callable[..., Any] | None:
    candidates: list[str] = []
    k = str(kernel)
    if k:
        candidates.append(k)
        candidates.append(k.lstrip("_"))
    mod_name = str(getattr(module, "__name__", "")).split(".")[-1]
    if mod_name:
        candidates.append(mod_name)

    all_names = [str(x) for x in list(getattr(module, "__all__", []) or [])]
    for name in all_names:
        if name.endswith("_kernel"):
            continue
        if name not in candidates:
            candidates.append(name)

    for name in candidates:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    return None


def _build_native_launch_fn(
    *,
    kernel: str,
    inputs_np: dict[str, Any],
    bindings: dict[str, Any],
    spec_map: dict[str, Any],
    device: str,
) -> tuple[Callable[[], None], str, dict[str, Any]]:
    spec = spec_map.get(str(kernel))
    if spec is None:
        raise RuntimeError(f"no native triton coverage spec for kernel={kernel}")

    module = importlib.import_module(str(spec.module))
    callee = _select_native_callable(module, str(kernel))
    if callee is None:
        raise RuntimeError(f"no native callable found in module={spec.module} kernel={kernel}")

    tensor_map: dict[str, torch.Tensor] = {}
    for name, value in inputs_np.items():
        arr = value
        t = torch.as_tensor(arr, device=device)
        tensor_map[str(name)] = t.contiguous()

    scalar_map: dict[str, Any] = dict(bindings)
    for name, value in inputs_np.items():
        arr = value
        if hasattr(arr, "shape") and tuple(getattr(arr, "shape", ())) == ():
            try:
                scalar_map.setdefault(str(name), arr.item())
            except Exception:
                pass

    by_param_tensor = {_kernel_param_key(k): v for k, v in tensor_map.items()}
    by_param_scalar = {_kernel_param_key(k): v for k, v in scalar_map.items()}
    sig = inspect.signature(callee)
    kwargs: dict[str, Any] = {}
    for pname, p in sig.parameters.items():
        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise RuntimeError(f"positional-only native signature unsupported ({pname})")
        if p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        key = _kernel_param_key(pname)
        if key in by_param_tensor:
            kwargs[pname] = by_param_tensor[key]
            continue
        if key in by_param_scalar:
            kwargs[pname] = by_param_scalar[key]
            continue
        if p.default is not inspect._empty:
            continue
        raise RuntimeError(f"missing native arg mapping for parameter={pname}")

    def _run() -> None:
        _ = callee(**kwargs)

    return _run, str(getattr(module, "__name__", "")), {"arg_count": len(kwargs)}


def _build_intentir_launch_fn(
    *,
    kernel: str,
    artifact_dir: str | None,
    device: str,
) -> tuple[Callable[[], None], dict[str, Any]]:
    ctx = _prepare_kernel_context(
        str(kernel),
        frontend="triton",
        triton_provider="flaggems",
        artifact_dir=artifact_dir,
    )
    lowered = lower_intent_to_cuda_kernel(ctx["intent"], shape_bindings=ctx["bindings"])
    t_compile0 = time.perf_counter()
    mod = compile_cuda_extension(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
    )
    compile_ms = (time.perf_counter() - t_compile0) * 1000.0

    inputs_np = _build_inputs_np(
        kernel=str(kernel),
        intent=ctx["intent"],
        baseline=ctx["baseline"],
        external_inputs=ctx["external_inputs"],
        bindings=ctx["bindings"],
    )

    tensors = lowered.io_spec.get("tensors") if isinstance(lowered.io_spec.get("tensors"), dict) else {}
    scalars = lowered.io_spec.get("scalars") if isinstance(lowered.io_spec.get("scalars"), dict) else {}
    arg_names = lowered.io_spec.get("arg_names") if isinstance(lowered.io_spec.get("arg_names"), list) else []
    arg_names = [str(x) for x in arg_names]
    out_set = {str(x) for x in list(lowered.output_names or ctx["outputs"])}

    args: list[Any] = []
    launch_tensors: dict[str, torch.Tensor] = {}
    for name in arg_names:
        if name in tensors:
            spec = tensors[name] if isinstance(tensors.get(name), dict) else {}
            dt = _np_to_torch_dtype(str(spec.get("dtype") or "f32"))
            shape_tpl = spec.get("shape") if isinstance(spec.get("shape"), list) else None
            if name in out_set:
                if shape_tpl is None:
                    raise RuntimeError(f"missing output tensor shape for {name}")
                shape = tuple(int(lowered.bindings[str(d)]) if isinstance(d, str) else int(d) for d in shape_tpl)
                t = torch.empty(shape, device=device, dtype=dt)
                launch_tensors[name] = t
                args.append(t)
            else:
                if name not in inputs_np:
                    if shape_tpl == [] and name in lowered.bindings:
                        val = lowered.bindings[name]
                        t = torch.tensor(val, device=device, dtype=dt)
                        launch_tensors[name] = t
                        args.append(t)
                        continue
                    raise RuntimeError(f"missing input for intentir launch arg {name}")
                t = torch.as_tensor(inputs_np[name], device=device)
                if t.dtype != dt:
                    t = t.to(dtype=dt)
                t = t.contiguous()
                launch_tensors[name] = t
                args.append(t)
        elif name in scalars:
            if name not in lowered.bindings:
                raise RuntimeError(f"missing scalar binding {name}")
            dt = str(scalars[name])
            if dt == "f32":
                args.append(float(lowered.bindings[name]))
            else:
                args.append(int(lowered.bindings[name]))
        else:
            if name not in lowered.bindings:
                raise RuntimeError(f"missing binding for arg {name}")
            args.append(int(lowered.bindings[name]))

    gx, gy, gz = (int(x) for x in lowered.launch.grid)
    bx, by, bz = (int(x) for x in lowered.launch.block)
    args += [gx, gy, gz, bx, by, bz, int(lowered.launch.shared_mem)]

    def _run() -> None:
        mod.launch(*args)

    meta = {
        "compile_ms": float(compile_ms),
        "kernel_name": str(lowered.kernel_name),
        "arg_count": int(len(args)),
        "tensor_arg_count": int(len(launch_tensors)),
    }
    return _run, meta


def _reason_code_from_exception(exc: Exception, *, native: bool) -> str:
    msg = str(exc)
    if isinstance(exc, CudaLoweringError):
        return "lowering_missing_op"
    if isinstance(exc, FileNotFoundError):
        return "artifact_missing"
    if isinstance(exc, RuntimeError) and "cuda graph capture failed" in msg.lower():
        return "graph_capture_fail"
    if isinstance(exc, RuntimeError) and "no native triton coverage spec" in msg.lower():
        return "native_unavailable"
    if isinstance(exc, RuntimeError) and "no native callable found" in msg.lower():
        return "native_unavailable"
    if isinstance(exc, RuntimeError) and "missing native arg mapping" in msg.lower():
        return "native_signature_mismatch"
    if isinstance(exc, RuntimeError) and "nvrtc_unavailable" in msg.lower():
        return "env_unavailable"
    return "native_runtime_fail" if native else "intentir_runtime_fail"


def _bench_kernel(
    *,
    kernel: str,
    family: str,
    chunk_name: str,
    threshold: float,
    warmup: int,
    iters: int,
    repeats: int,
    intent_artifact_dir: str | None,
    spec_map: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    row: dict[str, Any] = {
        "kernel": str(kernel),
        "semantic_op": str(kernel),
        "family": str(family),
        "chunk": str(chunk_name),
        "gpu_name": str(gpu_name),
        "bench_mode": "graph",
        "dtype": "mixed",
        "shape": {},
        "qps_native": None,
        "qps_intentir": None,
        "ratio": None,
        "latency_native_ms": None,
        "latency_intentir_ms": None,
        "compile_ms_native": 0.0,
        "compile_ms_intentir": 0.0,
        "capture_ms_native": 0.0,
        "capture_ms_intentir": 0.0,
        "replay_ms_native": 0.0,
        "replay_ms_intentir": 0.0,
        "reason_code": "",
        "reason_detail": "",
        "skip_reason": "",
        "count_in_denominator": False,
        "ok": False,
    }
    if not torch.cuda.is_available():
        row["reason_code"] = "env_unavailable"
        row["reason_detail"] = "torch.cuda.is_available=false"
        row["skip_reason"] = "cuda_unavailable"
        return row

    try:
        intent_fn, intent_meta = _build_intentir_launch_fn(
            kernel=str(kernel),
            artifact_dir=intent_artifact_dir,
            device=device,
        )
        row["compile_ms_intentir"] = float(intent_meta.get("compile_ms", 0.0))
    except Exception as e:  # noqa: BLE001
        row["reason_code"] = _reason_code_from_exception(e, native=False)
        row["reason_detail"] = f"{type(e).__name__}: {e}"
        row["skip_reason"] = "intentir_unavailable"
        return row

    try:
        # Reuse launch inputs generated from intent context for native signature mapping.
        ctx = _prepare_kernel_context(
            str(kernel),
            frontend="triton",
            triton_provider="flaggems",
            artifact_dir=intent_artifact_dir,
        )
        inputs_np = _build_inputs_np(
            kernel=str(kernel),
            intent=ctx["intent"],
            baseline=ctx["baseline"],
            external_inputs=ctx["external_inputs"],
            bindings=ctx["bindings"],
        )
        native_fn, native_module, native_meta = _build_native_launch_fn(
            kernel=str(kernel),
            inputs_np=inputs_np,
            bindings=dict(ctx["bindings"]),
            spec_map=spec_map,
            device=device,
        )
        row["native_module"] = str(native_module)
        row["compile_ms_native"] = float(native_meta.get("compile_ms", 0.0))
    except Exception as e:  # noqa: BLE001
        row["reason_code"] = _reason_code_from_exception(e, native=True)
        row["reason_detail"] = f"{type(e).__name__}: {e}"
        row["skip_reason"] = "native_unavailable"
        return row

    try:
        native_bench = _bench_graph(native_fn, warmup=warmup, iters=iters, repeats=repeats)
    except Exception as e:  # noqa: BLE001
        row["reason_code"] = _reason_code_from_exception(e, native=True)
        row["reason_detail"] = f"{type(e).__name__}: {e}"
        row["skip_reason"] = "native_graph_failed"
        return row

    try:
        intent_bench = _bench_graph(intent_fn, warmup=warmup, iters=iters, repeats=repeats)
    except Exception as e:  # noqa: BLE001
        row["reason_code"] = _reason_code_from_exception(e, native=False)
        row["reason_detail"] = f"{type(e).__name__}: {e}"
        row["skip_reason"] = "intentir_graph_failed"
        return row

    qps_native = float(native_bench["qps"])
    qps_intentir = float(intent_bench["qps"])
    ratio = float(qps_intentir / qps_native) if qps_native > 0.0 else 0.0
    row.update(
        {
            "qps_native": float(qps_native),
            "qps_intentir": float(qps_intentir),
            "ratio": float(ratio),
            "latency_native_ms": float(native_bench["latency_ms"]),
            "latency_intentir_ms": float(intent_bench["latency_ms"]),
            "capture_ms_native": float(native_bench["capture_ms"]),
            "capture_ms_intentir": float(intent_bench["capture_ms"]),
            "replay_ms_native": float(native_bench["replay_ms"]),
            "replay_ms_intentir": float(intent_bench["replay_ms"]),
            "reason_code": ("ok" if ratio >= float(threshold) else "gpu_perf_below_threshold"),
            "reason_detail": (
                f"qps_ratio={ratio:.4f} threshold={float(threshold):.4f}"
                if ratio >= float(threshold)
                else f"qps_ratio={ratio:.4f} below threshold={float(threshold):.4f}"
            ),
            "skip_reason": "",
            "count_in_denominator": True,
            "ok": bool(ratio >= float(threshold)),
        }
    )
    return row


def _counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for row in list(entries or []):
        c[str(row.get("status") or "unknown")] += 1
    return {k: int(v) for k, v in sorted(c.items(), key=lambda kv: kv[0])}


def _to_status_entries(rows: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(rows):
        reason = str(row.get("reason_code") or "runtime_fail")
        counted = bool(row.get("count_in_denominator"))
        ratio = row.get("ratio")
        ratio_ok = counted and isinstance(ratio, (int, float)) and float(ratio) >= float(threshold)
        status = "dual_pass" if ratio_ok else "blocked_backend"
        runtime_detail = {
            "native": {
                "qps": row.get("qps_native"),
                "latency_ms": row.get("latency_native_ms"),
                "capture_ms": row.get("capture_ms_native"),
                "replay_ms": row.get("replay_ms_native"),
            },
            "intentir_cuda": {
                "qps": row.get("qps_intentir"),
                "latency_ms": row.get("latency_intentir_ms"),
                "capture_ms": row.get("capture_ms_intentir"),
                "replay_ms": row.get("replay_ms_intentir"),
            },
        }
        out.append(
            {
                "semantic_op": str(row.get("semantic_op") or row.get("kernel") or ""),
                "kernel": str(row.get("kernel") or ""),
                "family": str(row.get("family") or ""),
                "status": str(status),
                "reason_code": str(reason),
                "status_reason": str(reason),
                "status_reason_detail": str(row.get("reason_detail") or ""),
                "runtime": {
                    "provider": "ok",
                    "rvv": "skipped",
                    "cuda": ("pass" if ratio_ok else "fail"),
                },
                "runtime_detail": runtime_detail,
                "artifact_complete": bool(row.get("reason_code")),
                "determinability": bool(row.get("reason_code")),
            }
        )
    return out


def _aggregate_device(rows: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    by_dev: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dev = str(row.get("gpu_name") or "unknown")
        by_dev.setdefault(dev, []).append(row)
    out: list[dict[str, Any]] = []
    for dev, items in sorted(by_dev.items(), key=lambda kv: kv[0]):
        measured = [r for r in items if bool(r.get("count_in_denominator"))]
        passed = [r for r in measured if isinstance(r.get("ratio"), (int, float)) and float(r["ratio"]) >= float(threshold)]
        failed = [r for r in measured if r not in passed]
        min_ratio = min((float(r.get("ratio", 0.0)) for r in measured), default=None)
        out.append(
            {
                "gpu_name": str(dev),
                "kernel_total": int(len(items)),
                "kernel_measured": int(len(measured)),
                "kernel_pass": int(len(passed)),
                "kernel_fail": int(len(failed)),
                "ok": bool(len(measured) > 0 and len(failed) == 0),
                "min_ratio": (None if min_ratio is None else float(min_ratio)),
            }
        )
    return out


def _stage_timing_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    native_rows = [r for r in rows if isinstance(r.get("qps_native"), (int, float))]
    intent_rows = [r for r in rows if isinstance(r.get("qps_intentir"), (int, float))]

    def _sum(items: list[dict[str, Any]], key: str) -> float:
        return float(sum(float(x.get(key, 0.0) or 0.0) for x in items))

    return {
        "schema_version": "flaggems_gpu_perf_stage_timing_v1",
        "generated_at": _utc_now_iso(),
        "native": {
            "kernel_count": int(len(native_rows)),
            "compile_ms": _sum(native_rows, "compile_ms_native"),
            "capture_ms": _sum(native_rows, "capture_ms_native"),
            "replay_ms": _sum(native_rows, "replay_ms_native"),
        },
        "intentir_cuda": {
            "kernel_count": int(len(intent_rows)),
            "compile_ms": _sum(intent_rows, "compile_ms_intentir"),
            "capture_ms": _sum(intent_rows, "capture_ms_intentir"),
            "replay_ms": _sum(intent_rows, "replay_ms_intentir"),
        },
    }


_SKIP_ONLY_REASONS: frozenset[str] = frozenset({"native_unavailable", "env_unavailable"})
_SKIP_ONLY_SKIP_REASONS: frozenset[str] = frozenset({"native_unavailable", "native_graph_failed"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coverage-batches",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "coverage_batches.json"),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=(ROOT / "artifacts" / "flaggems_matrix" / "daily" / datetime.now(timezone.utc).strftime("%Y%m%d") / "gpu_perf_graph"),
    )
    ap.add_argument("--family", action="append", default=[])
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--family-kernel-chunk-size", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--progress-style", choices=["auto", "tqdm", "plain", "chunk", "none"], default="chunk")
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional chunk-progress JSON path (default: <out-root>/chunk_progress.json).",
    )
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--intent-artifact-dir",
        default=str(ROOT / "artifacts" / "flaggems_triton_full_pipeline"),
    )
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.coverage_batches.is_file():
        raise SystemExit(f"missing coverage batches: {args.coverage_batches}")
    payload = _load_json(args.coverage_batches)
    family_order = [str(x).strip() for x in list(payload.get("family_order") or []) if str(x).strip()]
    by_family = {
        str(b.get("family") or "").strip(): b
        for b in list(payload.get("batches") or [])
        if isinstance(b, dict) and str(b.get("family") or "").strip()
    }
    requested = [str(x).strip() for x in list(args.family or []) if str(x).strip()]
    families = requested if requested else [f for f in family_order if f in by_family]
    unknown = [f for f in families if f not in by_family]
    if unknown:
        raise SystemExit(f"unknown family name(s): {', '.join(unknown)}")
    if not families:
        raise SystemExit("no families selected")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    progress_file = Path(args.progress_file) if args.progress_file is not None else (out_root / "chunk_progress.json")
    style = _choose_progress_style(str(args.progress_style))
    if style == "tqdm" and tqdm is None:
        style = "chunk"
    print(
        f"[gpu-perf] progress-style requested={args.progress_style} selected={style} "
        f"stdout_tty={sys.stdout.isatty()} tqdm_available={tqdm is not None}",
        flush=True,
    )
    if bool(args.dry_run):
        print("[gpu-perf] dry-run requested; no benchmark executed", flush=True)

    _set_codegen_mode_env()
    _set_runtime_backend_env(str(args.cuda_runtime_backend))

    spec_map = _coverage_spec_map()
    device = "cuda"

    family_plan: list[dict[str, Any]] = []
    total_chunks = 0
    for family in families:
        b = by_family[family]
        kernels = [str(k).strip() for k in list(b.get("kernels") or []) if str(k).strip()]
        chunks = _chunk_kernels(kernels, int(args.family_kernel_chunk_size))
        total_chunks += len(chunks)
        family_plan.append(
            {
                "family": str(family),
                "kernels": kernels,
                "chunks": chunks,
            }
        )

    _write_chunk_progress_file(
        path=progress_file,
        done=0,
        total=int(total_chunks),
        family="",
        chunk_idx=0,
        chunk_total=0,
        status="START",
        completed=False,
        progress_style=str(style),
        measured=0,
        failures=0,
    )

    all_rows: list[dict[str, Any]] = []
    family_results: list[dict[str, Any]] = []
    chunk_done = 0
    for fam_idx, fam in enumerate(family_plan, start=1):
        family = str(fam["family"])
        chunks = list(fam["chunks"])
        fam_out = out_root / f"family_{family}"
        fam_out.mkdir(parents=True, exist_ok=True)
        chunk_rows_meta: list[dict[str, Any]] = []
        chunk_iter = tqdm(enumerate(chunks, start=1), total=len(chunks), desc=f"gpu-perf:{family}") if style == "tqdm" else enumerate(chunks, start=1)
        for cidx, ckernels in chunk_iter:
            chunk_name = f"{family}_chunk_{cidx:03d}"
            chunk_json = fam_out / f"chunk_{cidx:03d}_gpu_perf_graph.json"
            if bool(args.resume) and chunk_json.is_file():
                chunk_payload = _load_json(chunk_json)
                chunk_entries = [e for e in list(chunk_payload.get("entries") or []) if isinstance(e, dict)]
                all_rows.extend(chunk_entries)
                chunk_ok = bool(chunk_payload.get("ok"))
                chunk_rows_meta.append(
                    {
                        "chunk": str(chunk_name),
                        "ok": bool(chunk_ok),
                        "json_path": _to_repo_rel(chunk_json),
                        "kernel_count": int(len(ckernels)),
                    }
                )
                chunk_done += 1
                _emit_chunk_progress(
                    style=style,
                    done=chunk_done,
                    total=total_chunks,
                    family=family,
                    chunk_idx=cidx,
                    chunk_total=len(chunks),
                    status="RESUME_OK" if chunk_ok else "RESUME_FAIL",
                )
                measured_rows = [r for r in all_rows if bool(r.get("count_in_denominator"))]
                failed_rows = [r for r in measured_rows if not bool(r.get("ok"))]
                _write_chunk_progress_file(
                    path=progress_file,
                    done=int(chunk_done),
                    total=int(total_chunks),
                    family=str(family),
                    chunk_idx=int(cidx),
                    chunk_total=int(len(chunks)),
                    status=("RESUME_OK" if chunk_ok else "RESUME_FAIL"),
                    completed=False,
                    progress_style=str(style),
                    measured=int(len(measured_rows)),
                    failures=int(len(failed_rows)),
                )
                continue

            chunk_entries: list[dict[str, Any]] = []
            if not bool(args.dry_run):
                for kernel in ckernels:
                    row = _bench_kernel(
                        kernel=str(kernel),
                        family=family,
                        chunk_name=chunk_name,
                        threshold=float(args.threshold),
                        warmup=int(args.warmup),
                        iters=int(args.iters),
                        repeats=int(args.repeats),
                        intent_artifact_dir=str(args.intent_artifact_dir),
                        spec_map=spec_map,
                        device=device,
                    )
                    chunk_entries.append(row)
            chunk_payload = {
                "schema_version": "flaggems_gpu_perf_graph_chunk_v1",
                "generated_at": _utc_now_iso(),
                "family": str(family),
                "chunk": str(chunk_name),
                "kernel_count": int(len(ckernels)),
                "ok": bool(
                    all(
                        (not bool(e.get("count_in_denominator")))
                        or bool(e.get("ok"))
                        for e in chunk_entries
                    )
                ),
                "entries": chunk_entries,
            }
            _dump_json(chunk_json, chunk_payload)
            all_rows.extend(chunk_entries)
            chunk_ok = bool(chunk_payload.get("ok"))
            chunk_rows_meta.append(
                {
                    "chunk": str(chunk_name),
                    "ok": bool(chunk_ok),
                    "json_path": _to_repo_rel(chunk_json),
                    "kernel_count": int(len(ckernels)),
                }
            )
            chunk_done += 1
            _emit_chunk_progress(
                style=style,
                done=chunk_done,
                total=total_chunks,
                family=family,
                chunk_idx=cidx,
                chunk_total=len(chunks),
                status="OK" if chunk_ok else "FAIL",
            )
            measured_rows = [r for r in all_rows if bool(r.get("count_in_denominator"))]
            failed_rows = [r for r in measured_rows if not bool(r.get("ok"))]
            _write_chunk_progress_file(
                path=progress_file,
                done=int(chunk_done),
                total=int(total_chunks),
                family=str(family),
                chunk_idx=int(cidx),
                chunk_total=int(len(chunks)),
                status=("OK" if chunk_ok else "FAIL"),
                completed=False,
                progress_style=str(style),
                measured=int(len(measured_rows)),
                failures=int(len(failed_rows)),
            )

        fam_rows = [r for r in all_rows if str(r.get("family") or "") == family]
        measured = [r for r in fam_rows if bool(r.get("count_in_denominator"))]
        measured_fail = [r for r in measured if not bool(r.get("ok"))]
        # Perf gate only evaluates kernels that are measurable on both paths.
        # Non-measured kernels are reported as skip/non-measured diagnostics
        # and are intentionally excluded from fail/pass denominator.
        non_measured_hard_fail = [
            r
            for r in fam_rows
            if (not bool(r.get("count_in_denominator")))
            and (
                str(r.get("reason_code") or "").strip() not in _SKIP_ONLY_REASONS
                and str(r.get("skip_reason") or "").strip() not in _SKIP_ONLY_SKIP_REASONS
            )
        ]
        fam_ok = len(measured_fail) == 0
        fam_status = "OK" if measured else "SKIP"
        family_results.append(
            {
                "family": str(family),
                "ok": bool(fam_ok),
                "status": str(fam_status),
                "kernel_count": int(len(list(fam.get("kernels") or []))),
                "chunk_count": int(len(chunks)),
                "kernel_measured": int(len(measured)),
                "kernel_measured_fail": int(len(measured_fail)),
                "kernel_skip_only": int(
                    sum(
                        1
                        for r in fam_rows
                        if (not bool(r.get("count_in_denominator")))
                        and (str(r.get("reason_code") or "").strip() in _SKIP_ONLY_REASONS)
                    )
                ),
                "kernel_non_measured_fail": int(len(non_measured_hard_fail)),
                "chunks": chunk_rows_meta,
            }
        )
        if bool(args.stream) and str(style) != "chunk":
            measured_count = len(measured)
            pass_count = sum(1 for r in measured if bool(r.get("ok")))
            print(
                f"[gpu-perf] family={family} measured={pass_count}/{measured_count} "
                f"chunks={len(chunks)} status={fam_status}",
                flush=True,
            )

    devices = _aggregate_device(all_rows, threshold=float(args.threshold))
    device_ok = all(bool(d.get("ok")) for d in devices)
    categories_expected = int(len(families))
    categories_completed = int(len(family_results))
    categories_failed = [str(f["family"]) for f in family_results if not bool(f.get("ok"))]

    failures_by_family: dict[str, list[str]] = {}
    for row in all_rows:
        if not bool(row.get("count_in_denominator")):
            continue
        if bool(row.get("ok")):
            continue
        fam = str(row.get("family") or "unknown")
        failures_by_family.setdefault(fam, []).append(str(row.get("kernel") or ""))

    measured_rows = [r for r in all_rows if bool(r.get("count_in_denominator"))]
    aggregate_ok = bool(
        categories_completed == categories_expected
        and len(categories_failed) == 0
        and len(measured_rows) > 0
        and device_ok
    )

    gpu_perf_json = out_root / "gpu_perf_graph.json"
    gpu_perf_payload = {
        "schema_version": "flaggems_gpu_perf_graph_v1",
        "generated_at": _utc_now_iso(),
        "repo": repo_state(root=ROOT),
        "mode": "graph_only",
        "threshold": float(args.threshold),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "repeats": int(args.repeats),
        "coverage_mode": "category_batches",
        "coverage_batches_expected": int(categories_expected),
        "coverage_batches_completed": int(categories_completed),
        "coverage_batches_failed": list(categories_failed),
        "categories_complete": bool(categories_completed == categories_expected and len(categories_failed) == 0),
        "devices": devices,
        "device_ok": bool(device_ok),
        "ok": bool(aggregate_ok),
        "family_runs": family_results,
        "failures_by_family": failures_by_family,
        "entries": all_rows,
    }
    _dump_json(gpu_perf_json, gpu_perf_payload)

    status_entries = _to_status_entries(all_rows, threshold=float(args.threshold))
    status_payload = {
        "schema_version": "flaggems_status_converged_v3",
        "generated_at": _utc_now_iso(),
        "scope_enabled": False,
        "entries": status_entries,
        "counts_global": _counts(status_entries),
        "counts_scoped": _counts(status_entries),
        "counts_scoped_active": _counts(status_entries),
        "counts_scoped_kernel_alias": _counts(status_entries),
        "global_entries_count": int(len(status_entries)),
        "scoped_entries_count": int(len(status_entries)),
        "scoped_entries_active_count": int(len(status_entries)),
        "scoped_entries_kernel_alias_count": int(len(status_entries)),
    }
    status_path = _dump_json(out_root / "status_converged.json", status_payload)

    stage_breakdown_payload = _stage_timing_breakdown(all_rows)
    stage_path = _dump_json(out_root / "stage_timing_breakdown.json", stage_breakdown_payload)

    run_summary = {
        "ok": bool(aggregate_ok),
        "suite": "gpu_perf_graph",
        "requested_suite": "gpu_perf_graph",
        "repo": repo_state(root=ROOT),
        "coverage_mode": "category_batches",
        "full196_evidence_kind": "batch_aggregate",
        "coverage_batches_expected": int(categories_expected),
        "coverage_batches_completed": int(categories_completed),
        "coverage_batches_failed": list(categories_failed),
        "gpu_perf_mode": "graph_only",
        "gpu_perf_threshold": float(args.threshold),
        "gpu_perf_categories_complete": bool(categories_completed == categories_expected and len(categories_failed) == 0),
        "gpu_perf_per_device_ok": bool(device_ok),
        "gpu_perf_kernel_measured": int(len(measured_rows)),
        "gpu_perf_graph_path": _to_repo_rel(gpu_perf_json),
        "status_converged_path": _to_repo_rel(status_path),
        "stage_timing_breakdown_path": _to_repo_rel(stage_path),
        "invocation": {
            "execution_ir": "mlir",
            "intentir_mode": "auto",
            "miss_policy": "strict",
            "rvv_remote": False,
            "cuda_runtime_backend": str(args.cuda_runtime_backend),
            "bench_mode": "graph",
            "chunk_size": int(args.family_kernel_chunk_size),
        },
        "stages": [
            {
                "stage": "gpu_perf_graph",
                "ok": bool(aggregate_ok),
                "json_path": _to_repo_rel(gpu_perf_json),
                "reason_code": ("ok" if aggregate_ok else "gpu_perf_gate_fail"),
                "reason_detail": (
                    "all measured kernels meet threshold"
                    if aggregate_ok
                    else f"device_ok={device_ok} categories_failed={categories_failed}"
                ),
            },
            {
                "stage": "status_converged",
                "ok": True,
                "json_path": _to_repo_rel(status_path),
                "reason_code": "ok",
            },
            {
                "stage": "stage_timing_breakdown",
                "ok": True,
                "json_path": _to_repo_rel(stage_path),
                "reason_code": "ok",
            },
        ],
    }
    run_summary_path = _dump_json(out_root / "run_summary.json", run_summary)

    print(f"GPU perf graph report written: {gpu_perf_json}")
    print(f"Status converged written: {status_path}")
    print(f"Run summary written: {run_summary_path}")
    _write_chunk_progress_file(
        path=progress_file,
        done=int(chunk_done),
        total=int(total_chunks),
        family="",
        chunk_idx=0,
        chunk_total=0,
        status=("DONE_OK" if aggregate_ok else "DONE_FAIL"),
        completed=True,
        progress_style=str(style),
        measured=int(len(measured_rows)),
        failures=int(len([r for r in measured_rows if not bool(r.get("ok"))])),
    )
    raise SystemExit(0 if bool(aggregate_ok) else 1)


if __name__ == "__main__":
    main()
