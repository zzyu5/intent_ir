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
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.codegen.intentir_to_c import lower_intent_to_c_with_files  # noqa: E402
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


def _write_bin(path: Path, arr: np.ndarray, dtype: str) -> None:
    if dtype in {"bool", "i1"}:
        raw = np.asarray(arr, dtype=np.uint8).tobytes(order="C")
    else:
        raw = np.asarray(arr, dtype=np.float32).tobytes(order="C")
    path.write_bytes(raw)


def run_one(
    kernel: str,
    *,
    frontend: str = "triton",
    keep_tmp: bool = False,
    tune_request: TuningRequest | None = None,
    tune_profile: str | None = None,
) -> dict:
    artifact_dir = "full_pipeline_verify" if frontend == "triton" else "tilelang_full_pipeline"
    report_path = ROOT / "artifacts" / artifact_dir / f"{kernel}.json"
    baseline_npz_path = ROOT / "artifacts" / artifact_dir / f"{kernel}.baseline.npz"
    if not report_path.exists():
        raise FileNotFoundError(f"missing artifact report: {report_path}")
    if not baseline_npz_path.exists():
        raise FileNotFoundError(f"missing baseline npz: {baseline_npz_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    intent = _load_intent(report)
    tile_hints: list[int] = []
    try:
        cert_v2 = report.get("certificate_v2") or {}
        sh = cert_v2.get("schedule_hints") or {}
        th = sh.get("tile_hints")
        if isinstance(th, list):
            tile_hints = [int(x) for x in th if isinstance(x, (int, float, str)) and int(x) > 0]
    except Exception:
        tile_hints = []

    baseline = dict(np.load(baseline_npz_path, allow_pickle=False))
    baseline = _with_io_aliases_for_diff(intent, baseline)
    external_inputs, outputs = _external_inputs(intent)

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
    if tune_request is not None:
        prof = load_profile(tune_profile or "generic_rvv_256")
        tuned = select_schedule(intent, shape_bindings=bindings, profile=prof, request=tune_request, tile_hints=tile_hints)
        intent.schedule = tuned.schedule
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
        # Write inputs / reference outputs.
        for name in external_inputs:
            if name not in baseline:
                raise RuntimeError(f"baseline missing input {name} for {kernel}")
            _write_bin(td / f"{name}.bin", np.asarray(baseline[name]), intent.tensors[name].dtype)
        for name in outputs:
            if name not in baseline:
                raise RuntimeError(f"baseline missing output {name} for {kernel}")
            _write_bin(td / f"{name}_ref.bin", np.asarray(baseline[name]), intent.tensors[name].dtype)

        c_src = lower_intent_to_c_with_files(intent, shape_bindings=bindings, atol=float(atol), rtol=float(rtol))
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
        cp = subprocess.run(compile_cmd, cwd=td, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"compile failed:\n{cp.stderr or cp.stdout}")

        rp = subprocess.run([str(td / "run")], cwd=td, capture_output=True, text=True)
        return {
            "kernel": kernel,
            "ok": rp.returncode == 0,
            "rc": rp.returncode,
            "stdout": (rp.stdout or "").strip(),
            "stderr": (rp.stderr or "").strip(),
            "tmpdir": str(td) if keep_tmp else None,
        }
    finally:
        if keep_tmp:
            tmp_ctx.cleanup = lambda: None  # type: ignore[attr-defined]
        else:
            tmp_ctx.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument("--keep-tmp", action="store_true", help="keep generated C + binaries in a temp dir")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default=None)
    ap.add_argument("--lock", action="append", default=[], help="repeatable; e.g. --lock tile_n=128")
    ap.add_argument("--constraint", action="append", default=[], help="repeatable; e.g. --constraint 'tile_n in (64,128)'")
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: generic_rvv_256 when tuning)")
    args = ap.parse_args()

    kernels = args.kernel or DEFAULT_KERNELS
    tune_req = None
    if args.tune_mode:
        tune_req = TuningRequest(
            mode=str(args.tune_mode),
            budget=0,
            locks=parse_locks(args.lock or []),
            constraints=parse_constraints(args.constraint or []),
        )
    results = []
    ok_all = True
    for k in kernels:
        r = run_one(
            k,
            frontend=str(args.frontend),
            keep_tmp=bool(args.keep_tmp),
            tune_request=tune_req,
            tune_profile=str(args.profile) if args.profile else None,
        )
        results.append(r)
        ok_all = ok_all and bool(r["ok"])
        status = "OK" if r["ok"] else "FAIL"
        print(f"[{k}] {status} rc={r['rc']}")
        if r["stdout"]:
            print(r["stdout"])
        if r["stderr"]:
            print(r["stderr"])
        if args.keep_tmp and r.get("tmpdir"):
            print(f"  tmpdir={r['tmpdir']}")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
