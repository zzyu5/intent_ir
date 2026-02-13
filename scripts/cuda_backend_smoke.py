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
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.cuda.codegen.intentir_to_cuda import CudaLoweringError, lower_intent_to_cuda_kernel  # noqa: E402
from backends.cuda.runtime import CudaRuntimeError, run_cuda_kernel  # noqa: E402
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


def run_one(
    kernel: str,
    *,
    frontend: str = "triton",
    triton_provider: str = "native",
    artifact_dir: str | None = None,
) -> dict:
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
    atol = float(tol.get("atol", 1e-3))
    rtol = float(tol.get("rtol", 1e-3))

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

    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)
    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np=inputs_np,
        output_names=lowered.output_names or outputs,
    )
    out = _with_io_aliases_for_diff(intent, out)

    checks: list[dict[str, Any]] = []
    ok_all = True
    for name in outputs:
        if name not in baseline:
            checks.append({"name": str(name), "ok": False, "summary": f"baseline missing output {name}"})
            ok_all = False
            continue
        if name not in out:
            checks.append({"name": str(name), "ok": False, "summary": f"cuda output missing {name}"})
            ok_all = False
            continue
        ok, summary = _compare_output(name, out[name], baseline[name], atol=atol, rtol=rtol)
        checks.append({"name": str(name), "ok": bool(ok), "summary": str(summary)})
        ok_all = ok_all and bool(ok)

    return {
        "kernel": str(kernel),
        "ok": bool(ok_all),
        "reason_code": ("ok" if ok_all else "diff_fail"),
        "atol": float(atol),
        "rtol": float(rtol),
        "checks": checks,
        "bindings": dict(bindings),
    }


def _reason_code_for_exception(e: Exception) -> str:
    if isinstance(e, CudaLoweringError):
        return "lowering_missing_op"
    if isinstance(e, CudaRuntimeError):
        msg = str(e).lower()
        if "timeout" in msg:
            return "runtime_timeout"
        return "runtime_fail"
    if isinstance(e, FileNotFoundError):
        return "artifact_missing"
    if isinstance(e, RuntimeError):
        return "runtime_fail"
    if isinstance(e, ValueError):
        return "runtime_fail"
    return "runtime_fail"


def _run_one_with_timeout(
    kernel: str,
    *,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    timeout_sec: int,
    codegen_mode: str,
) -> dict[str, Any]:
    if int(timeout_sec) <= 0:
        return run_one(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()

    proc = ctx.Process(
        target=_cuda_worker,
        args=(q, str(kernel), str(frontend), str(triton_provider), artifact_dir, str(codegen_mode)),
    )
    proc.start()
    proc.join(timeout=float(timeout_sec))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        return {
            "kernel": str(kernel),
            "ok": False,
            "error": {
                "type": "TimeoutError",
                "message": f"kernel execution exceeded timeout_sec={int(timeout_sec)}",
            },
            "reason_code": "runtime_timeout",
        }
    if q.empty():
        return {
            "kernel": str(kernel),
            "ok": False,
            "error": {"type": "RuntimeError", "message": "no result returned from worker process"},
            "reason_code": "runtime_fail",
        }
    payload = q.get()
    if bool(payload.get("ok")):
        return dict(payload.get("result") or {})
    err = payload.get("error") or {}
    return {
        "kernel": str(kernel),
        "ok": False,
        "error": {"type": str(err.get("type")), "message": str(err.get("message"))},
        "reason_code": str(err.get("reason_code") or "runtime_fail"),
    }


def _cuda_worker(
    queue: mp.Queue,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    codegen_mode: str,
) -> None:
    mode = str(codegen_mode).strip().lower()
    if mode in {"py", "cpp"}:
        os.environ["INTENTIR_CUDA_CODEGEN"] = mode
    else:
        os.environ.pop("INTENTIR_CUDA_CODEGEN", None)
    try:
        res = run_one(kernel, frontend=frontend, triton_provider=triton_provider, artifact_dir=artifact_dir)
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


def _refine_timeout_reason(
    *,
    kernel: str,
    frontend: str,
    triton_provider: str,
    artifact_dir: str | None,
    timeout_sec: int,
) -> dict[str, Any] | None:
    """
    Best-effort second pass for timeout cases.

    The default (C++ codegen) path can hang for unsupported ops in some envs.
    Retry once with Python codegen to produce an actionable reason code.
    """
    probe = _run_one_with_timeout(
        kernel=kernel,
        frontend=frontend,
        triton_provider=triton_provider,
        artifact_dir=artifact_dir,
        timeout_sec=max(1, min(int(timeout_sec), 30)),
        codegen_mode="py",
    )
    if bool(probe.get("ok")):
        return {
            "reason_code": "runtime_timeout",
            "error": {
                "type": "TimeoutError",
                "message": "cpp codegen path timed out; py codegen path passed",
            },
            "refined_by": "py_codegen_probe",
            "refined_result": {
                "ok": True,
            },
        }
    probe_reason = str(probe.get("reason_code") or "")
    if probe_reason == "runtime_timeout":
        return None
    probe_error = probe.get("error") if isinstance(probe.get("error"), dict) else {}
    return {
        "reason_code": probe_reason or "runtime_timeout",
        "error": {
            "type": str(probe_error.get("type") or "RuntimeError"),
            "message": str(probe_error.get("message") or "timeout refined by py_codegen_probe"),
        },
        "refined_by": "py_codegen_probe",
        "refined_result": {
            "ok": False,
            "reason_code": probe_reason or "runtime_fail",
        },
    }


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
        help="Per-kernel timeout in seconds (<=0 disables timeout wrapper).",
    )
    ap.add_argument(
        "--codegen-mode",
        choices=["auto", "cpp", "py"],
        default="auto",
        help="CUDA codegen mode for smoke run (default: auto).",
    )
    ap.add_argument(
        "--refine-timeout-reason",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When timeout happens under auto mode, retry once with py codegen to classify reason.",
    )
    ap.add_argument("--artifact-dir", default=None, help="Override artifact report directory.")
    ap.add_argument("--allow-skip", action="store_true", help="exit 0 with ok=false when CUDA environment is unavailable")
    ap.add_argument("--json", action="store_true", help="print machine-readable summary JSON")
    ap.add_argument("--out", default=None, help="write summary JSON to this path")
    args = ap.parse_args()

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
            r = _run_one_with_timeout(
                str(k),
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                timeout_sec=int(args.timeout_sec),
                codegen_mode=str(args.codegen_mode),
            )
        except (CudaLoweringError, CudaRuntimeError, FileNotFoundError, RuntimeError, ValueError) as e:
            r = {
                "kernel": str(k),
                "ok": False,
                "reason_code": _reason_code_for_exception(e),
                "error": {"type": type(e).__name__, "message": str(e)},
            }
        if (
            (not bool(r.get("ok")))
            and str(r.get("reason_code") or "") == "runtime_timeout"
            and str(args.codegen_mode) == "auto"
            and bool(args.refine_timeout_reason)
        ):
            refined = _refine_timeout_reason(
                kernel=str(k),
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                timeout_sec=int(args.timeout_sec),
            )
            if refined is not None:
                r["reason_code"] = str(refined.get("reason_code") or r.get("reason_code"))
                r["error"] = dict(refined.get("error") or r.get("error") or {})
                r["refined_by"] = str(refined.get("refined_by") or "py_codegen_probe")
                r["refined_result"] = dict(refined.get("refined_result") or {})
        if "reason_code" not in r:
            r["reason_code"] = ("ok" if bool(r.get("ok")) else "runtime_fail")
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
        "timeout_sec": int(args.timeout_sec),
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
