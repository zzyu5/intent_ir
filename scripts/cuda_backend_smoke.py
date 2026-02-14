"""
CUDA backend smoke from existing frontend artifacts (no LLM, no remote).

For each kernel artifact under `artifacts/<frontend>_full_pipeline/`, this script:
  - loads expanded IntentIR (or expands macros)
  - loads baseline inputs/outputs from `<kernel>.baseline.npz`
  - lowers IntentIR to CUDA
  - runs the CUDA kernel via runtime wrapper
  - compares outputs against baseline reference
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from queue import Empty
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.cuda.codegen.intentir_to_cuda import CudaLoweringError, lower_intent_to_cuda_kernel  # noqa: E402
from backends.cuda.runtime import CudaRuntimeError, compile_cuda_extension, run_cuda_kernel  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402
from intent_ir.macros import expand_macros  # noqa: E402
from verify.diff_runner import _with_io_aliases as _with_io_aliases_for_diff  # noqa: E402


DEFAULT_KERNELS = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]


def _default_kernels_for(
    *,
    frontend: str,
    triton_provider: str,
    flaggems_opset: str,
    backend_target: str,
) -> list[str]:
    if str(frontend) == "triton" and str(triton_provider) == "flaggems":
        from pipeline.triton.providers.flaggems.specs import default_flaggems_kernel_specs  # noqa: PLC0415

        return [
            str(s.name)
            for s in default_flaggems_kernel_specs(
                flaggems_opset=str(flaggems_opset),
                backend_target=str(backend_target),
            )
        ]
    return list(DEFAULT_KERNELS)


def _artifact_dir_for_frontend(frontend: str, *, triton_provider: str = "native") -> str:
    if frontend == "triton":
        p = str(triton_provider)
        if p == "flaggems":
            return "flaggems_triton_full_pipeline"
        if p == "native":
            return "full_pipeline_verify"
        raise ValueError(f"unsupported triton provider: {triton_provider}")
    if frontend == "tilelang":
        return "tilelang_full_pipeline"
    raise ValueError(f"unsupported frontend: {frontend}")


def _load_intent(report: dict) -> IntentFunction:
    intent_macro = IntentFunction.from_json_dict(report["intent"])
    intent_expanded_json = report.get("intent_expanded")
    if isinstance(intent_expanded_json, dict):
        return IntentFunction.from_json_dict(intent_expanded_json)
    return expand_macros(intent_macro)


def _external_inputs(intent: IntentFunction) -> tuple[list[str], list[str]]:
    produced = {op.output for op in intent.ops if op.output}
    used: set[str] = set()
    for op in intent.ops:
        used.update(op.inputs)
    external_inputs = sorted([n for n in used if n in intent.tensors and n not in produced])
    return external_inputs, list(intent.outputs)


def _np_dtype(dt: str) -> Any:
    m = {
        "f16": np.float16,
        "bf16": np.float32,
        "f32": np.float32,
        "f64": np.float64,
        "i8": np.int8,
        "u8": np.uint8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "i1": np.bool_,
        "bool": np.bool_,
    }
    return m.get(str(dt), np.float32)


def _coerce_bindings(bindings: dict) -> dict:
    out = dict(bindings)
    if "batch" in out and "Z" not in out:
        out["Z"] = out["batch"]
    if "Z" in out and "batch" not in out:
        out["batch"] = out["Z"]
    if "group" in out and "num_groups" not in out:
        out["num_groups"] = out["group"]
    if "num_groups" in out and "C" in out and "group_size" not in out:
        try:
            g = int(out["num_groups"])
            c = int(out["C"])
            if g > 0:
                out["group_size"] = c // g if (c % g == 0) else (c + g - 1) // g
        except Exception:
            pass
    if "group_size" in out and "HW" in out and "num_elements" not in out:
        try:
            out["num_elements"] = int(out["group_size"]) * int(out["HW"])
        except Exception:
            pass
    if "N" in out and "group_size" in out and "G" not in out:
        try:
            n = int(out["N"])
            gs = int(out["group_size"])
            if gs > 0 and n % gs == 0:
                out["G"] = n // gs
        except Exception:
            pass
    if "HEAD_DIM" in out:
        try:
            hd = int(out["HEAD_DIM"])
            if hd > 0:
                out.setdefault("HEAD_DIM_DIV2", hd // 2)
        except Exception:
            pass
    return out


def _derive_scalar_input(name: str, *, dtype: str, bindings: dict) -> np.ndarray | None:
    value = None
    if name == "sm_scale":
        hd = bindings.get("HEAD_DIM")
        if hd is not None and int(hd) > 0:
            value = 1.0 / math.sqrt(float(hd))
    elif name in bindings:
        value = bindings.get(name)
    if value is None:
        return None
    return np.array(value, dtype=_np_dtype(dtype))


def _compare_output(name: str, got: np.ndarray, ref: np.ndarray, *, atol: float, rtol: float) -> tuple[bool, str]:
    g = np.asarray(got)
    r = np.asarray(ref)
    if g.shape != r.shape:
        return False, f"shape mismatch {name}: {g.shape} vs {r.shape}"
    if g.dtype == np.bool_ or r.dtype == np.bool_:
        ok = bool(np.array_equal(g.astype(np.bool_), r.astype(np.bool_)))
        return ok, ("ok" if ok else f"bool mismatch in {name}")
    ok = bool(np.allclose(g, r, atol=float(atol), rtol=float(rtol), equal_nan=True))
    if ok:
        return True, "ok"
    abs_err = np.max(np.abs(g.astype(np.float64) - r.astype(np.float64))) if g.size else 0.0
    return False, f"mismatch in {name} (max_abs={float(abs_err):.6g}, atol={atol}, rtol={rtol})"


def _prepare_kernel_context(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
) -> dict[str, Any]:
    artifact_rel = _artifact_dir_for_frontend(frontend, triton_provider=str(triton_provider))
    artifact_root = (Path(artifact_dir) if artifact_dir else (ROOT / "artifacts" / artifact_rel)).resolve()
    report_path = artifact_root / f"{kernel}.json"
    baseline_npz_path = artifact_root / f"{kernel}.baseline.npz"
    if not report_path.exists():
        raise FileNotFoundError(f"missing artifact report: {report_path}")
    if not baseline_npz_path.exists():
        raise FileNotFoundError(f"missing baseline npz: {baseline_npz_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    intent = _load_intent(report)
    baseline = dict(np.load(baseline_npz_path, allow_pickle=False))
    baseline = _with_io_aliases_for_diff(intent, baseline)
    external_inputs, outputs = _external_inputs(intent)
    raw_bindings = ((report.get("baseline") or {}).get("shapes") or {}) if isinstance(report.get("baseline"), dict) else {}
    bindings = _coerce_bindings(raw_bindings)
    tol = (report.get("tolerances") or {}) if isinstance(report.get("tolerances"), dict) else {}
    return {
        "kernel": str(kernel),
        "intent": intent,
        "baseline": baseline,
        "external_inputs": external_inputs,
        "outputs": outputs,
        "bindings": bindings,
        "atol": float(tol.get("atol", 1e-3)),
        "rtol": float(tol.get("rtol", 1e-3)),
    }


def _build_inputs_np(
    *,
    kernel: str,
    intent: IntentFunction,
    baseline: dict[str, np.ndarray],
    external_inputs: list[str],
    bindings: dict[str, Any],
) -> dict[str, np.ndarray]:
    inputs_np: dict[str, np.ndarray] = {}
    for name in external_inputs:
        if name in baseline:
            inputs_np[name] = np.asarray(baseline[name])
            continue
        tt = intent.tensors.get(name)
        if tt is None or tt.shape:
            raise RuntimeError(f"baseline missing input {name} for {kernel}")
        derived = _derive_scalar_input(name, dtype=str(tt.dtype), bindings=bindings)
        if derived is None:
            raise RuntimeError(f"baseline missing input {name} for {kernel}")
        inputs_np[name] = derived
    return inputs_np


def _set_codegen_mode_env(codegen_mode: str) -> None:
    mode = str(codegen_mode).strip().lower()
    if mode in {"py", "cpp"}:
        os.environ["INTENTIR_CUDA_CODEGEN"] = mode
    else:
        os.environ.pop("INTENTIR_CUDA_CODEGEN", None)


def _set_runtime_backend_env(runtime_backend: str) -> None:
    mode = str(runtime_backend).strip().lower()
    if mode == "nvrtc":
        os.environ["INTENTIR_CUDA_FORCE_NVRTC"] = "1"
        os.environ.setdefault("INTENTIR_CUDA_NVRTC_FALLBACK", "1")
        try:
            from cuda import nvrtc as _nvrtc  # noqa: PLC0415
        except Exception as e:
            raise RuntimeError(f"nvrtc_unavailable: {type(e).__name__}: {e}") from e
    else:
        os.environ.pop("INTENTIR_CUDA_FORCE_NVRTC", None)


def _run_compile_stage(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    ctx = _prepare_kernel_context(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)
    t_lower = time.perf_counter()
    lowered = lower_intent_to_cuda_kernel(ctx["intent"], shape_bindings=ctx["bindings"])
    lower_ms = (time.perf_counter() - t_lower) * 1000.0

    t_compile = time.perf_counter()
    compile_cuda_extension(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
    )
    compile_ms = (time.perf_counter() - t_compile) * 1000.0

    return {
        "kernel": str(kernel),
        "ok": True,
        "reason_code": "ok",
        "lower_ms": float(lower_ms),
        "compile_ms": float(compile_ms),
        "launch_ms": 0.0,
        "total_ms": float((time.perf_counter() - t_all) * 1000.0),
        "bindings": dict(ctx["bindings"]),
    }


def _run_launch_stage(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    ctx = _prepare_kernel_context(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)

    t_lower = time.perf_counter()
    lowered = lower_intent_to_cuda_kernel(ctx["intent"], shape_bindings=ctx["bindings"])
    lower_ms = (time.perf_counter() - t_lower) * 1000.0

    t_compile = time.perf_counter()
    compile_cuda_extension(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
    )
    compile_ms = (time.perf_counter() - t_compile) * 1000.0

    inputs_np = _build_inputs_np(
        kernel=str(kernel),
        intent=ctx["intent"],
        baseline=ctx["baseline"],
        external_inputs=ctx["external_inputs"],
        bindings=ctx["bindings"],
    )

    t_launch = time.perf_counter()
    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np=inputs_np,
        output_names=lowered.output_names or ctx["outputs"],
    )
    out = _with_io_aliases_for_diff(ctx["intent"], out)
    launch_ms = (time.perf_counter() - t_launch) * 1000.0

    checks: list[dict[str, Any]] = []
    ok_all = True
    for name in ctx["outputs"]:
        if name not in ctx["baseline"]:
            checks.append({"name": str(name), "ok": False, "summary": f"baseline missing output {name}"})
            ok_all = False
            continue
        if name not in out:
            checks.append({"name": str(name), "ok": False, "summary": f"cuda output missing {name}"})
            ok_all = False
            continue
        ok, summary = _compare_output(name, out[name], ctx["baseline"][name], atol=ctx["atol"], rtol=ctx["rtol"])
        checks.append({"name": str(name), "ok": bool(ok), "summary": str(summary)})
        ok_all = ok_all and bool(ok)

    return {
        "kernel": str(kernel),
        "ok": bool(ok_all),
        "reason_code": ("ok" if ok_all else "diff_fail"),
        "atol": float(ctx["atol"]),
        "rtol": float(ctx["rtol"]),
        "checks": checks,
        "bindings": dict(ctx["bindings"]),
        "lower_ms": float(lower_ms),
        "compile_ms": float(compile_ms),
        "launch_ms": float(launch_ms),
        "total_ms": float((time.perf_counter() - t_all) * 1000.0),
    }


def run_one(
    kernel: str,
    *,
    frontend: str = "triton",
    triton_provider: str = "native",
    artifact_dir: str | None = None,
) -> dict:
    # Compatibility helper for ad-hoc invocations.
    _ = _run_compile_stage(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)
    return _run_launch_stage(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)


def _reason_code_for_exception(e: Exception) -> str:
    if isinstance(e, CudaLoweringError):
        return "lowering_missing_op"
    if isinstance(e, CudaRuntimeError):
        msg = str(e).lower()
        if "timeout" in msg:
            return "launch_timeout"
        return "runtime_fail"
    if isinstance(e, FileNotFoundError):
        return "artifact_missing"
    if isinstance(e, RuntimeError):
        if "nvrtc_unavailable" in str(e).lower():
            return "env_unavailable"
        return "runtime_fail"
    if isinstance(e, ValueError):
        return "runtime_fail"
    return "runtime_fail"


def _cuda_compile_worker(
    queue: mp.Queue,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    codegen_mode: str,
    runtime_backend: str,
) -> None:
    try:
        _set_codegen_mode_env(codegen_mode)
        _set_runtime_backend_env(runtime_backend)
        res = _run_compile_stage(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)
        queue.put({"ok": True, "result": res})
    except Exception as ex:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": {
                    "type": type(ex).__name__,
                    "message": str(ex),
                    "reason_code": _reason_code_for_exception(ex),
                },
            }
        )


def _cuda_launch_worker(
    queue: mp.Queue,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    codegen_mode: str,
    runtime_backend: str,
) -> None:
    try:
        _set_codegen_mode_env(codegen_mode)
        _set_runtime_backend_env(runtime_backend)
        res = _run_launch_stage(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)
        queue.put({"ok": True, "result": res})
    except Exception as ex:  # noqa: BLE001
        queue.put(
            {
                "ok": False,
                "error": {
                    "type": type(ex).__name__,
                    "message": str(ex),
                    "reason_code": _reason_code_for_exception(ex),
                },
            }
        )


def _run_worker_with_timeout(
    worker: Any,
    *,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    codegen_mode: str,
    runtime_backend: str,
    timeout_sec: int,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=worker,
        args=(q, str(kernel), str(frontend), str(triton_provider), artifact_dir, str(codegen_mode), str(runtime_backend)),
    )
    proc.start()
    timeout = None if int(timeout_sec) <= 0 else float(timeout_sec)
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        return {"ok": False, "timed_out": True}
    try:
        payload = q.get_nowait()
    except Empty:
        payload = None
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "timed_out": False,
            "error": {
                "type": "RuntimeError",
                "message": "no result returned from worker process",
                "reason_code": "runtime_fail",
            },
        }
    payload["timed_out"] = False
    return payload


def _run_one_with_stage_timeouts(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    compile_timeout_sec: int,
    launch_timeout_sec: int,
    codegen_mode: str,
    runtime_backend: str,
) -> dict[str, Any]:
    compile_res = _run_worker_with_timeout(
        _cuda_compile_worker,
        kernel=kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        codegen_mode=codegen_mode,
        runtime_backend=runtime_backend,
        timeout_sec=int(compile_timeout_sec),
    )
    if bool(compile_res.get("timed_out")):
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": "compile_timeout",
            "error": {"type": "TimeoutError", "message": f"compile stage exceeded timeout_sec={int(compile_timeout_sec)}"},
        }
    if not bool(compile_res.get("ok")):
        err = compile_res.get("error") if isinstance(compile_res.get("error"), dict) else {}
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": str(err.get("reason_code") or "runtime_fail"),
            "error": {"type": str(err.get("type") or "RuntimeError"), "message": str(err.get("message") or "compile stage failed")},
        }
    compile_payload = dict(compile_res.get("result") or {})

    launch_res = _run_worker_with_timeout(
        _cuda_launch_worker,
        kernel=kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        codegen_mode=codegen_mode,
        runtime_backend=runtime_backend,
        timeout_sec=int(launch_timeout_sec),
    )
    if bool(launch_res.get("timed_out")):
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": "launch_timeout",
            "error": {"type": "TimeoutError", "message": f"launch stage exceeded timeout_sec={int(launch_timeout_sec)}"},
            "lower_ms": float(compile_payload.get("lower_ms", 0.0)),
            "compile_ms": float(compile_payload.get("compile_ms", 0.0)),
            "launch_ms": 0.0,
            "total_ms": float(compile_payload.get("total_ms", 0.0)),
        }
    if not bool(launch_res.get("ok")):
        err = launch_res.get("error") if isinstance(launch_res.get("error"), dict) else {}
        return {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": str(err.get("reason_code") or "runtime_fail"),
            "error": {"type": str(err.get("type") or "RuntimeError"), "message": str(err.get("message") or "launch stage failed")},
            "lower_ms": float(compile_payload.get("lower_ms", 0.0)),
            "compile_ms": float(compile_payload.get("compile_ms", 0.0)),
            "launch_ms": 0.0,
            "total_ms": float(compile_payload.get("total_ms", 0.0)),
        }

    launch_payload = dict(launch_res.get("result") or {})
    launch_ms = float(launch_payload.get("launch_ms", 0.0))
    lower_ms = float(compile_payload.get("lower_ms", launch_payload.get("lower_ms", 0.0)))
    compile_ms = float(compile_payload.get("compile_ms", launch_payload.get("compile_ms", 0.0)))
    launch_payload["lower_ms"] = lower_ms
    launch_payload["compile_ms"] = compile_ms
    launch_payload["launch_ms"] = launch_ms
    launch_payload["total_ms"] = float(lower_ms + compile_ms + launch_ms)
    return launch_payload


def _probe_timeout_runtime_detail(
    *,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    compile_timeout_sec: int,
    launch_timeout_sec: int,
    runtime_backend: str,
) -> dict[str, Any]:
    probe = _run_one_with_stage_timeouts(
        kernel=kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        compile_timeout_sec=max(1, min(int(compile_timeout_sec), 30)),
        launch_timeout_sec=max(1, min(int(launch_timeout_sec), 30)),
        codegen_mode="py",
        runtime_backend=runtime_backend,
    )
    detail = {
        "ok": bool(probe.get("ok")),
        "reason_code": str(probe.get("reason_code") or "runtime_fail"),
    }
    if isinstance(probe.get("error"), dict):
        detail["error"] = dict(probe.get("error") or {})
    return detail


def _cuda_env_ready() -> tuple[bool, str]:
    if shutil.which("nvcc") is None:
        return False, "nvcc_not_found"
    try:
        import torch  # noqa: PLC0415

        if not bool(torch.cuda.is_available()):
            return False, "torch_cuda_unavailable"
    except Exception as e:
        return False, f"torch_import_error:{type(e).__name__}"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument(
        "--triton-provider",
        choices=["native", "flaggems"],
        default="native",
        help="Triton artifact provider (default: native)",
    )
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs kernel suite")
    ap.add_argument(
        "--flaggems-opset",
        choices=["deterministic_forward"],
        default="deterministic_forward",
        help="FlagGems semantic-op set used to resolve default kernels.",
    )
    ap.add_argument(
        "--backend-target",
        choices=["rvv", "cuda_h100", "cuda_5090d"],
        default="cuda_h100",
        help="Capability target passed to FlagGems spec registry when selecting defaults.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Compatibility default timeout in seconds used when stage-specific timeouts are not set.",
    )
    ap.add_argument(
        "--compile-timeout-sec",
        type=int,
        default=None,
        help="Compile-stage timeout in seconds (defaults to --timeout-sec).",
    )
    ap.add_argument(
        "--launch-timeout-sec",
        type=int,
        default=None,
        help="Launch-stage timeout in seconds (defaults to --timeout-sec).",
    )
    ap.add_argument(
        "--codegen-mode",
        choices=["auto", "cpp", "py"],
        default="auto",
        help="CUDA codegen mode for smoke run (default: auto).",
    )
    ap.add_argument(
        "--runtime-backend",
        choices=["auto", "nvcc", "nvrtc"],
        default="auto",
        help="CUDA runtime compile backend selector. nvrtc forces NVRTC path, nvcc forces extension path.",
    )
    ap.add_argument(
        "--refine-timeout-reason",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When timeout happens under auto mode, run a short py-codegen probe and attach runtime_detail.",
    )
    ap.add_argument("--artifact-dir", default=None, help="Override artifact report directory.")
    ap.add_argument("--allow-skip", action="store_true", help="exit 0 with ok=false when CUDA environment is unavailable")
    ap.add_argument("--json", action="store_true", help="print machine-readable summary JSON")
    ap.add_argument("--out", default=None, help="write summary JSON to this path")
    args = ap.parse_args()

    compile_timeout_sec = int(args.compile_timeout_sec) if args.compile_timeout_sec is not None else int(args.timeout_sec)
    launch_timeout_sec = int(args.launch_timeout_sec) if args.launch_timeout_sec is not None else int(args.timeout_sec)
    runtime_backend = str(args.runtime_backend)
    if runtime_backend == "auto":
        runtime_backend = "nvcc"

    if args.kernel:
        kernels = list(args.kernel)
    else:
        kernels = _default_kernels_for(
            frontend=str(args.frontend),
            triton_provider=str(args.triton_provider),
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        )

    env_ok, env_reason = _cuda_env_ready()
    if not env_ok:
        summary = {
            "frontend": str(args.frontend),
            "triton_provider": (str(args.triton_provider) if str(args.frontend) == "triton" else None),
            "flaggems_opset": str(args.flaggems_opset),
            "backend_target": str(args.backend_target),
            "artifact_dir": (str(args.artifact_dir) if args.artifact_dir else None),
            "runtime_backend": str(runtime_backend),
            "timeout_sec": int(args.timeout_sec),
            "compile_timeout_sec": int(compile_timeout_sec),
            "launch_timeout_sec": int(launch_timeout_sec),
            "kernels": list(kernels),
            "results": [],
            "ok": False,
            "skipped": True,
            "skip_reason": str(env_reason),
        }
        if args.out:
            Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.json:
            print(json.dumps(summary, ensure_ascii=False))
        if args.allow_skip:
            raise SystemExit(0)
        raise SystemExit(1)

    results: list[dict[str, Any]] = []
    ok_all = True
    for k in kernels:
        try:
            r = _run_one_with_stage_timeouts(
                str(k),
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                compile_timeout_sec=int(compile_timeout_sec),
                launch_timeout_sec=int(launch_timeout_sec),
                codegen_mode=str(args.codegen_mode),
                runtime_backend=runtime_backend,
            )
        except (CudaLoweringError, CudaRuntimeError, FileNotFoundError, RuntimeError, ValueError) as e:
            r = {
                "kernel": str(k),
                "ok": False,
                "reason_code": _reason_code_for_exception(e),
                "error": {"type": type(e).__name__, "message": str(e)},
            }
        if str(r.get("reason_code") or "") not in {"ok", "compile_timeout", "launch_timeout"}:
            if "lower_ms" not in r:
                r["lower_ms"] = 0.0
            if "compile_ms" not in r:
                r["compile_ms"] = 0.0
            if "launch_ms" not in r:
                r["launch_ms"] = 0.0
            if "total_ms" not in r:
                r["total_ms"] = float(r["lower_ms"]) + float(r["compile_ms"]) + float(r["launch_ms"])

        if (
            (not bool(r.get("ok")))
            and str(r.get("reason_code") or "") in {"compile_timeout", "launch_timeout"}
            and str(args.codegen_mode) == "auto"
            and bool(args.refine_timeout_reason)
        ):
            detail = _probe_timeout_runtime_detail(
                kernel=str(k),
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                compile_timeout_sec=int(compile_timeout_sec),
                launch_timeout_sec=int(launch_timeout_sec),
                runtime_backend=runtime_backend,
            )
            runtime_detail = dict(r.get("runtime_detail") or {})
            runtime_detail["timeout_probe"] = detail
            r["runtime_detail"] = runtime_detail
        if "reason_code" not in r:
            r["reason_code"] = ("ok" if bool(r.get("ok")) else "runtime_fail")
        if "lower_ms" not in r:
            r["lower_ms"] = 0.0
        if "compile_ms" not in r:
            r["compile_ms"] = 0.0
        if "launch_ms" not in r:
            r["launch_ms"] = 0.0
        if "total_ms" not in r:
            r["total_ms"] = float(r["lower_ms"]) + float(r["compile_ms"]) + float(r["launch_ms"])
        results.append(r)
        ok_all = ok_all and bool(r.get("ok"))
        if not args.json:
            print(f"[{k}] {'OK' if r.get('ok') else 'FAIL'}")
            if r.get("error"):
                print(f"  {r['error']['type']}: {r['error']['message']}")

    summary = {
        "frontend": str(args.frontend),
        "triton_provider": (str(args.triton_provider) if str(args.frontend) == "triton" else None),
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "artifact_dir": (str(args.artifact_dir) if args.artifact_dir else None),
        "runtime_backend": str(runtime_backend),
        "timeout_sec": int(args.timeout_sec),
        "compile_timeout_sec": int(compile_timeout_sec),
        "launch_timeout_sec": int(launch_timeout_sec),
        "kernels": list(kernels),
        "results": results,
        "ok": bool(ok_all),
        "skipped": False,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
