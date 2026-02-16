"""
Backend codegen smoke test (no LLM, no remote).

For each kernel artifact under `artifacts/<frontend>_full_pipeline/`, this script:
  - loads expanded IntentIR (or expands macros)
  - loads baseline inputs/outputs from `<kernel>.baseline.npz`
  - invokes Task6 backend codegen (C++ tool) to generate a standalone C program
  - compiles and runs the C program locally to compare against baseline outputs

This validates "IntentIR ops -> C" backend generation end-to-end, without
requiring an RVV host.
"""

from __future__ import annotations

import argparse
import re
import json
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.codegen.cpp_driver import lower_intent_to_c_with_files  # noqa: E402
from backends.spmd_rvv.analysis.device_query import load_profile  # noqa: E402
from backends.spmd_rvv.analysis.tuning import TuningRequest, parse_constraints, parse_locks, select_schedule  # noqa: E402
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


def _resolve_tensor_shape(tensor: Any, bindings: dict) -> tuple[int, ...] | None:
    shape: list[int] = []
    for d in list(getattr(tensor, "shape", []) or []):
        if hasattr(d, "kind") and getattr(d, "kind") == "sym":
            key = str(getattr(d, "value"))
            if key not in bindings:
                return None
            try:
                shape.append(int(bindings[key]))
            except Exception:
                return None
            continue
        if hasattr(d, "kind") and getattr(d, "kind") == "const":
            try:
                shape.append(int(getattr(d, "value")))
            except Exception:
                return None
            continue
        if isinstance(d, int):
            shape.append(int(d))
            continue
        key = str(d)
        if key in bindings:
            try:
                shape.append(int(bindings[key]))
                continue
            except Exception:
                return None
        try:
            shape.append(int(key))
        except Exception:
            return None
    return tuple(shape)


def _augment_bindings_from_arrays(*, intent: IntentFunction, bindings: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    out = dict(bindings)
    for name, arr in arrays.items():
        tensor = intent.tensors.get(str(name))
        if tensor is None:
            continue
        arr_np = np.asarray(arr)
        spec_shape = list(getattr(tensor, "shape", []) or [])
        if len(spec_shape) == 0 and arr_np.size == 1:
            # Scalar tensors often carry symbolic shape params (e.g., group_size).
            # Bind by tensor name so codegen can resolve reshape expressions.
            key = str(name)
            if key and key not in out:
                try:
                    out[key] = int(arr_np.reshape(()).item())
                except Exception:
                    pass
        arr_shape = tuple(int(v) for v in np.asarray(arr).shape)
        if len(spec_shape) != len(arr_shape):
            continue
        for dim_spec, dim_val in zip(spec_shape, arr_shape):
            key: str | None = None
            if hasattr(dim_spec, "kind") and getattr(dim_spec, "kind") == "sym":
                key = str(getattr(dim_spec, "value"))
            elif isinstance(dim_spec, str):
                try:
                    int(dim_spec)
                except Exception:
                    key = str(dim_spec)
            if key and key not in out:
                out[key] = int(dim_val)
    return out


def _derive_optional_input_array(name: str, *, tensor: Any, bindings: dict) -> np.ndarray | None:
    if str(name) == "sm_scale":
        hd = bindings.get("HEAD_DIM")
        try:
            if hd is not None and int(hd) > 0:
                return np.array(1.0 / np.sqrt(float(hd)), dtype=_np_dtype(str(getattr(tensor, "dtype", "f32"))))
        except Exception:
            pass
    if str(name) == "attn_mask":
        shape = _resolve_tensor_shape(tensor, bindings)
        if shape is not None:
            return np.zeros(shape, dtype=_np_dtype(str(getattr(tensor, "dtype", "f32"))))
    return None


def _write_bin(path: Path, arr: np.ndarray, dtype: str) -> None:
    arr_np = np.asarray(arr)
    dtype = str(dtype)
    # Respect declared tensor dtype first. Baseline arrays may carry bool views
    # for tensors that are declared as f32/i32 in intent tensors.
    if dtype in {"bool", "i1"}:
        raw = np.asarray(arr_np, dtype=np.uint8).tobytes(order="C")
    elif dtype == "i8":
        raw = np.asarray(arr_np, dtype=np.int8).tobytes(order="C")
    elif dtype == "u8":
        raw = np.asarray(arr_np, dtype=np.uint8).tobytes(order="C")
    elif dtype == "i16":
        raw = np.asarray(arr_np, dtype=np.int16).tobytes(order="C")
    elif dtype == "i32":
        raw = np.asarray(arr_np, dtype=np.int32).tobytes(order="C")
    elif dtype == "i64":
        raw = np.asarray(arr_np, dtype=np.int64).tobytes(order="C")
    elif dtype == "f16":
        raw = np.asarray(arr_np, dtype=np.float16).tobytes(order="C")
    elif dtype == "f64":
        raw = np.asarray(arr_np, dtype=np.float64).tobytes(order="C")
    else:
        raw = np.asarray(arr_np, dtype=np.float32).tobytes(order="C")
    path.write_bytes(raw)


_BUF_DESC_RE = re.compile(r'\{\s*"(?P<name>[^"]+)"\s*,.*?(?P<dtype>INTENTIR_DTYPE_[A-Z0-9_]+)')


def _dtype_from_buffer_token(tok: str) -> str | None:
    m = {
        "INTENTIR_DTYPE_F16": "f16",
        "INTENTIR_DTYPE_BF16": "bf16",
        "INTENTIR_DTYPE_F32": "f32",
        "INTENTIR_DTYPE_F64": "f64",
        "INTENTIR_DTYPE_I8": "i8",
        "INTENTIR_DTYPE_U8": "u8",
        "INTENTIR_DTYPE_I16": "i16",
        "INTENTIR_DTYPE_I32": "i32",
        "INTENTIR_DTYPE_I64": "i64",
        "INTENTIR_DTYPE_BOOL": "bool",
        "INTENTIR_DTYPE_I1": "i1",
    }
    return m.get(str(tok))


def _extract_buffer_declared_dtypes(c_src: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(c_src).splitlines():
        s = line.strip()
        if not s.startswith('{"') or "INTENTIR_DTYPE_" not in s:
            continue
        m = _BUF_DESC_RE.search(s)
        if not m:
            continue
        name = str(m.group("name"))
        dt = _dtype_from_buffer_token(str(m.group("dtype")))
        if dt:
            out[name] = dt
    return out


def run_one(
    kernel: str,
    *,
    frontend: str = "triton",
    triton_provider: str = "native",
    artifact_dir: str | None = None,
    keep_tmp: bool = False,
    tune_request: TuningRequest | None = None,
    tune_profile: str | None = None,
) -> dict:
    t_total = perf_counter()
    lower_ms = 0.0
    compile_ms = 0.0
    launch_ms = 0.0
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
    cert_v2 = report.get("certificate_v2") or {}
    tile_hints: list[int] = []
    try:
        sh = cert_v2.get("schedule_hints") or {}
        th = sh.get("tile_hints")
        if isinstance(th, list):
            tile_hints = [int(x) for x in th if isinstance(x, (int, float, str)) and int(x) > 0]
    except Exception:
        tile_hints = []

    baseline = dict(np.load(baseline_npz_path, allow_pickle=False))
    baseline = _with_io_aliases_for_diff(intent, baseline)
    external_inputs, outputs = _external_inputs(intent)
    produced = {op.output for op in intent.ops if op.output}
    if baseline:
        # Verify only outputs available in the baseline bundle. Some kernels
        # expose auxiliary outputs (e.g., indices) that are intentionally not
        # included in baseline artifacts.
        outputs = [name for name in outputs if name in baseline]
    else:
        outputs = [name for name in outputs if name in produced]
    if not outputs:
        outputs = [name for name in intent.outputs if (name in baseline)] if baseline else list(intent.outputs)
    if not outputs:
        # Keep declared outputs to avoid constructing an invalid empty-output intent.
        outputs = list(intent.outputs)
    intent_codegen = intent
    if outputs != list(intent.outputs):
        intent_j = intent.to_json_dict()
        intent_j["outputs"] = list(outputs)
        intent_codegen = IntentFunction.from_json_dict(intent_j)

    bindings = ((report.get("baseline") or {}).get("shapes") or {}) if isinstance(report.get("baseline"), dict) else {}
    # Common axis aliases (match pipeline/runner conventions).
    if "batch" in bindings and "Z" not in bindings:
        bindings["Z"] = bindings["batch"]
    if "Z" in bindings and "batch" not in bindings:
        bindings["batch"] = bindings["Z"]
    if "group" in bindings and "num_groups" not in bindings:
        bindings["num_groups"] = bindings["group"]
    if "num_groups" in bindings and "C" in bindings and "group_size" not in bindings:
        try:
            g = int(bindings["num_groups"])
            c = int(bindings["C"])
            if g > 0:
                bindings["group_size"] = c // g if (c % g == 0) else (c + g - 1) // g
        except Exception:
            pass
    if "group_size" in bindings and "HW" in bindings and "num_elements" not in bindings:
        try:
            bindings["num_elements"] = int(bindings["group_size"]) * int(bindings["HW"])
        except Exception:
            pass
    if "N" in bindings and "group_size" in bindings and "G" not in bindings:
        try:
            n = int(bindings["N"])
            gs = int(bindings["group_size"])
            if gs > 0 and n % gs == 0:
                bindings["G"] = n // gs
        except Exception:
            pass
    bindings = _augment_bindings_from_arrays(intent=intent, bindings=bindings, arrays=baseline)
    if tune_request is not None:
        prof = load_profile(tune_profile or "generic_rvv_256")
        tuned = select_schedule(intent, shape_bindings=bindings, profile=prof, request=tune_request, tile_hints=tile_hints, evidence=cert_v2)
        intent.schedule = tuned.schedule
        intent_codegen.schedule = tuned.schedule
    tol = {
        "any_kernel_dim": (0.0, 0.0),
        "group_norm_kernel": (1e-3, 1e-3),
        "_attn_fwd": (1e-2, 1e-2),
        "softmax_inner": (1e-3, 1e-3),
        "layer_norm_persistent": (1e-3, 1e-3),
        "upsample_bicubic2d_aa": (1e-3, 1e-3),
    }
    atol, rtol = tol.get(kernel, (1e-3, 1e-3))

    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"intentir_codegen_smoke_{kernel}_")
    td = Path(tmp_ctx.name)
    try:
        t_lower = perf_counter()
        c_src = lower_intent_to_c_with_files(intent_codegen, shape_bindings=bindings, atol=float(atol), rtol=float(rtol))
        lower_ms = (perf_counter() - t_lower) * 1000.0
        declared_dtypes = _extract_buffer_declared_dtypes(c_src)

        # Write inputs / reference outputs after lowering, so file dtypes follow
        # emitted buffer descriptors instead of potentially stale intent tensor dtypes.
        for name in external_inputs:
            if name not in baseline:
                tt = intent.tensors.get(name)
                if tt is not None:
                    derived = _derive_optional_input_array(name, tensor=tt, bindings=bindings)
                    if derived is not None:
                        baseline[name] = derived
            if name not in baseline:
                raise RuntimeError(f"baseline missing input {name} for {kernel}")
            dtype = declared_dtypes.get(name, str(intent.tensors[name].dtype))
            _write_bin(td / f"{name}.bin", np.asarray(baseline[name]), dtype)
        for name in outputs:
            if name not in baseline:
                raise RuntimeError(f"baseline missing output {name} for {kernel}")
            dtype = declared_dtypes.get(name, str(intent.tensors[name].dtype))
            _write_bin(td / f"{name}_ref.bin", np.asarray(baseline[name]), dtype)

        (td / "main.c").write_text(c_src, encoding="utf-8")

        runtime_dir = ROOT / "backends" / "spmd_rvv" / "runtime"
        for fn in [
            "intentir_runtime.h",
            "intentir_runtime.c",
            "intentir_driver.h",
            "intentir_driver.c",
            "intentir_ops.h",
            "intentir_ops.c",
        ]:
            src_p = runtime_dir / fn
            if not src_p.exists():
                raise FileNotFoundError(f"missing RVV runtime file: {src_p}")
            shutil.copy(src_p, td / fn)

        compile_cmd = [
            "gcc",
            "-O2",
            "-std=c11",
            "-D_POSIX_C_SOURCE=200809L",
            "-I.",
            "-o",
            str(td / "run"),
            str(td / "main.c"),
            str(td / "intentir_runtime.c"),
            str(td / "intentir_driver.c"),
            str(td / "intentir_ops.c"),
            "-lm",
            "-lrt",
        ]
        t_compile = perf_counter()
        cp = subprocess.run(compile_cmd, cwd=td, capture_output=True, text=True)
        compile_ms = (perf_counter() - t_compile) * 1000.0
        if cp.returncode != 0:
            return {
                "kernel": kernel,
                "ok": False,
                "rc": int(cp.returncode),
                "reason_code": "compile_fail",
                "stdout": (cp.stdout or "").strip(),
                "stderr": (cp.stderr or "").strip(),
                "tmpdir": str(td) if keep_tmp else None,
                "lower_ms": float(lower_ms),
                "compile_ms": float(compile_ms),
                "launch_ms": float(launch_ms),
                "total_ms": float((perf_counter() - t_total) * 1000.0),
            }

        t_launch = perf_counter()
        rp = subprocess.run([str(td / "run")], cwd=td, capture_output=True, text=True)
        launch_ms = (perf_counter() - t_launch) * 1000.0
        return {
            "kernel": kernel,
            "ok": rp.returncode == 0,
            "rc": rp.returncode,
            "reason_code": ("ok" if rp.returncode == 0 else "runtime_fail"),
            "stdout": (rp.stdout or "").strip(),
            "stderr": (rp.stderr or "").strip(),
            "tmpdir": str(td) if keep_tmp else None,
            "lower_ms": float(lower_ms),
            "compile_ms": float(compile_ms),
            "launch_ms": float(launch_ms),
            "total_ms": float((perf_counter() - t_total) * 1000.0),
        }
    finally:
        if keep_tmp:
            # Keep temporary artifacts for postmortem debugging.
            # TemporaryDirectory schedules deletion via an internal finalizer,
            # so overriding cleanup() is not sufficient.
            try:
                tmp_ctx._finalizer.detach()  # type: ignore[attr-defined]
            except Exception:
                tmp_ctx.cleanup = lambda: None  # type: ignore[attr-defined]
        else:
            tmp_ctx.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument(
        "--triton-provider",
        choices=["native", "flaggems"],
        default="native",
        help="Triton artifact provider (default: native)",
    )
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument(
        "--flaggems-opset",
        choices=["deterministic_forward"],
        default="deterministic_forward",
        help="FlagGems semantic-op set used to resolve default kernels.",
    )
    ap.add_argument(
        "--backend-target",
        choices=["rvv", "cuda_h100", "cuda_5090d"],
        default="rvv",
        help="Capability target passed to FlagGems spec registry when selecting defaults.",
    )
    ap.add_argument("--artifact-dir", default=None, help="Override artifact report directory.")
    ap.add_argument("--keep-tmp", action="store_true", help="keep generated C + binaries in a temp dir")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default=None)
    ap.add_argument("--lock", action="append", default=[], help="repeatable; e.g. --lock tile_n=128")
    ap.add_argument("--constraint", action="append", default=[], help="repeatable; e.g. --constraint 'tile_n in (64,128)'")
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: generic_rvv_256 when tuning)")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-kernel START/DONE progress lines even in --json mode.",
    )
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
    tune_req = None
    if args.tune_mode:
        tune_req = TuningRequest(
            mode=str(args.tune_mode),
            budget=0,
            locks=parse_locks(args.lock or []),
            constraints=parse_constraints(args.constraint or []),
        )
    results: list[dict[str, Any]] = []
    ok_all = True
    total_kernels = len(kernels)
    for idx, k in enumerate(kernels, start=1):
        if bool(args.progress):
            print(f"[rvv][{idx}/{total_kernels}] START kernel={k}", flush=True)
        try:
            r = run_one(
                k,
                frontend=str(args.frontend),
                triton_provider=str(args.triton_provider),
                artifact_dir=(str(args.artifact_dir) if args.artifact_dir else None),
                keep_tmp=bool(args.keep_tmp),
                tune_request=tune_req,
                tune_profile=str(args.profile) if args.profile else None,
            )
        except Exception as e:
            r = {
                "kernel": str(k),
                "ok": False,
                "rc": 1,
                "reason_code": "runtime_fail",
                "stdout": "",
                "stderr": f"{type(e).__name__}: {e}",
                "error": {"type": type(e).__name__, "message": str(e)},
                "lower_ms": 0.0,
                "compile_ms": 0.0,
                "launch_ms": 0.0,
                "total_ms": 0.0,
            }
        results.append(r)
        ok_all = ok_all and bool(r["ok"])
        if bool(args.progress):
            print(
                f"[rvv][{idx}/{total_kernels}] DONE kernel={k} ok={bool(r.get('ok'))} "
                f"reason={str(r.get('reason_code') or '')} "
                f"lower_ms={float(r.get('lower_ms', 0.0)):.3f} "
                f"compile_ms={float(r.get('compile_ms', 0.0)):.3f} "
                f"launch_ms={float(r.get('launch_ms', 0.0)):.3f} "
                f"total_ms={float(r.get('total_ms', 0.0)):.3f}",
                flush=True,
            )
        if not bool(args.json):
            status = "OK" if r["ok"] else "FAIL"
            print(f"[{k}] {status} rc={r.get('rc', 1)}")
            if r.get("stdout"):
                print(r["stdout"])
            if r.get("stderr"):
                print(r["stderr"])
            if args.keep_tmp and r.get("tmpdir"):
                print(f"  tmpdir={r['tmpdir']}")

    summary = {
        "frontend": str(args.frontend),
        "triton_provider": (str(args.triton_provider) if str(args.frontend) == "triton" else None),
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "artifact_dir": (str(args.artifact_dir) if args.artifact_dir else None),
        "kernels": list(kernels),
        "results": results,
        "ok": bool(ok_all),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if bool(args.json):
        print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
