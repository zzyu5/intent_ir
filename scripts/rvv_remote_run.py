"""
Prototype: end-to-end RVV remote run for supported kernels.

Current support:
- generic IntentIR ops lowering to a standalone C program (default: C++ codegen).

Usage:
  INTENTIR_SSH_PASSWORD=... python scripts/rvv_remote_run.py --kernel any_kernel_dim --host <host> --user <user>
  # or omit INTENTIR_SSH_PASSWORD and type it when prompted
Requires: `artifacts/<frontend>_full_pipeline/<kernel>.json` produced beforehand
(Triton uses `artifacts/full_pipeline_verify/` for historical reasons).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Callable

import paramiko
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.codegen.intentir_to_c import lower_intent_to_c_with_files
from backends.spmd_rvv.analysis.device_query import load_profile, query_remote_device
from backends.spmd_rvv.analysis.tuning import TuningRequest, parse_constraints, parse_locks, propose_schedule_candidates, select_schedule
from intent_ir.ir import IntentFunction
from intent_ir.macros import expand_macros
from verify.gen_cases import TestCase
from verify.tolerances import infer_tolerances


def _normalize_io_name(name: str) -> str:
    s = str(name).strip().strip("_").lower()
    if s.startswith("ptr_"):
        s = s[4:]
    if s.endswith("_ptr"):
        s = s[:-4]
    if s.endswith("ptr") and len(s) > 3:
        s = s[:-3]
    # Single-letter IO names from some kernels/LLM outputs.
    if s == "i":
        s = "input"
    if s == "o":
        s = "output"
    if s == "x":
        s = "input"
    if s == "y":
        s = "output"
    if s == "w":
        s = "weight"
    if s == "b":
        s = "bias"
    if s == "in":
        s = "input"
    if s == "out":
        s = "output"
    return s


def _with_io_aliases(intent: IntentFunction, io: dict) -> dict:
    out = dict(io)
    norm_to_keys: dict[str, list[str]] = {}
    for k in io.keys():
        norm_to_keys.setdefault(_normalize_io_name(k), []).append(k)
    wanted = set(intent.tensors.keys()) | set(intent.outputs)
    for name in wanted:
        if name in out:
            continue
        keys = norm_to_keys.get(_normalize_io_name(name)) or []
        if keys:
            if len(keys) == 1:
                out[name] = io[keys[0]]
                continue
            # If collisions exist (e.g., both "Input" and "input"), pick a stable best match.
            lower_name = str(name).lower()
            norm = _normalize_io_name(name)
            preferred = None
            for k in keys:
                if str(k).lower() == lower_name:
                    preferred = k
                    break
            if preferred is None and norm in io:
                preferred = norm
            if preferred is None:
                preferred = keys[0]
            out[name] = io[preferred]
            continue
        norm = _normalize_io_name(name)
        # Avoid overly-short names (e.g., "N") accidentally matching "input".
        if len(norm) >= 3:
            candidates = [k for k in io.keys() if norm and (norm in _normalize_io_name(k))]
            if len(candidates) == 1:
                out[name] = io[candidates[0]]
    return out


def _sftp_mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = [p for p in path.split("/") if p]
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def _sftp_write_bytes(sftp: paramiko.SFTPClient, path: str, data: bytes) -> None:
    with sftp.file(path, "wb") as f:
        f.write(data)


def run_remote(
    kernel: str,
    frontend: str,
    host: str,
    user: str,
    password: str | None,
    port: int = 22,
    case_index: int = 0,
    shape_overrides: dict | None = None,
    baseline_npz: str | None = None,
    prefer_live_baseline: bool = False,
    tune_request: TuningRequest | None = None,
    tune_profile: str | None = None,
    bench_iters: int = 0,
    bench_warmup: int = 1,
    profile_ops: bool = False,
    log: Callable[[str], None] | None = None,
):
    def _log(msg: str) -> None:
        if log is None:
            return
        try:
            log(str(msg))
        except Exception:
            pass

    artifact_dir = "full_pipeline_verify" if frontend == "triton" else "tilelang_full_pipeline"
    report_path = ROOT / "artifacts" / artifact_dir / f"{kernel}.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"artifact not found: {report_path}, please run scripts/{frontend}/full_pipeline_verify.py first"
        )
    _log(f"[{frontend}:{kernel}] load artifact: {report_path}")
    report = json.loads(report_path.read_text())
    intent_macro = IntentFunction.from_json_dict(report["intent"])
    intent_expanded_json = report.get("intent_expanded")
    if isinstance(intent_expanded_json, dict):
        intent = IntentFunction.from_json_dict(intent_expanded_json)
    else:
        intent = expand_macros(intent_macro)

    # Frontend-derived "tile-ish" constants (schedule hints). Triton CertificateV2
    # extracts these from TTIR; TileLang may leave empty (OK).
    cert_v2 = report.get("certificate_v2") or {}
    tile_hints: list[int] = []
    try:
        sh = cert_v2.get("schedule_hints") or {}
        th = sh.get("tile_hints")
        if isinstance(th, list):
            for x in th:
                try:
                    v = int(x)
                except Exception:
                    continue
                if v > 0:
                    tile_hints.append(v)
    except Exception:
        tile_hints = []
    if not tile_hints:
        try:
            # Legacy v1 certificate format (fallback).
            cert1 = report.get("certificate") or {}
            th = cert1.get("tile_hints")
            if isinstance(th, list):
                tile_hints = [int(x) for x in th if isinstance(x, (int, float, str)) and int(x) > 0]
        except Exception:
            tile_hints = []
    tile_hints = sorted(set(tile_hints))
    # Select shapes from cases (for binding).
    cases_raw = report.get("cases") or []
    cases: list[dict] = []
    if isinstance(cases_raw, dict):
        # v1.2 format: {"in_contract":[...], "out_of_contract":[...]}
        in_contract = cases_raw.get("in_contract")
        if isinstance(in_contract, list):
            cases = [c for c in in_contract if isinstance(c, dict)]
    elif isinstance(cases_raw, list):
        # legacy: list[dict]
        cases = [c for c in cases_raw if isinstance(c, dict)]
    case_idx = min(max(int(case_index), 0), len(cases) - 1) if cases else 0
    bindings = dict(cases[case_idx]) if cases else {}
    if shape_overrides:
        bindings.update(shape_overrides)
    # Common axis aliases (align kernel-signature symbols with user-friendly names).
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

    baseline = None
    npz_path = baseline_npz or (report.get("baseline") or {}).get("npz_path")
    if (not prefer_live_baseline) and npz_path:
        _log(f"[{frontend}:{kernel}] load baseline npz: {npz_path}")
        npz_path = str(npz_path)
        baseline_npz_path = Path(npz_path)
        if not baseline_npz_path.is_absolute():
            baseline_npz_path = (ROOT / baseline_npz_path).resolve()
        if not baseline_npz_path.exists():
            raise FileNotFoundError(f"baseline npz not found: {baseline_npz_path}")
        baseline = dict(np.load(baseline_npz_path, allow_pickle=False))

    # Fallback: if baseline is missing (or user forces live), try to re-launch Triton
    # to get baseline IO. This keeps Task6 usable on machines with CUDA.
    if baseline is None:
        _log(f"[{frontend}:{kernel}] baseline npz missing; try live baseline launch")
        try:
            if frontend == "triton":
                from pipeline.triton.core import default_kernel_specs

                spec_map = {s.name: s for s in default_kernel_specs()}
                if kernel not in spec_map:
                    raise RuntimeError(f"unknown kernel {kernel}")
                spec = spec_map[kernel]
                if not bindings:
                    bindings = dict(spec.canonical_shapes)
                baseline = spec.runner(TestCase(shapes=bindings, dtypes={}, seed=0))
            else:
                from pipeline.tilelang.core import default_kernel_specs, mvp_kernel_specs

                spec_map = {s.name: s for s in (mvp_kernel_specs() + default_kernel_specs())}
                if kernel not in spec_map:
                    raise RuntimeError(f"unknown kernel {kernel}")
                spec = spec_map[kernel]
                if not bindings:
                    bindings = dict(spec.canonical_shapes)
                baseline = spec.runner(TestCase(shapes=bindings, dtypes={}, seed=0))
        except Exception as e:
            raise RuntimeError(
                "baseline not available: no cached baseline .npz in artifacts and live Triton launch failed. "
                "Run `python scripts/triton/full_pipeline_verify.py` on a CUDA machine to produce "
                "`artifacts/full_pipeline_verify/<kernel>.baseline.npz`, or pass --baseline-npz.\n"
                f"live launch error: {type(e).__name__}: {e}"
            ) from e

    # Prefer baseline shapes (the embedded inputs correspond to that launch).
    baseline_shapes = ((report.get("baseline") or {}).get("shapes") or {}) if isinstance(report.get("baseline"), dict) else {}
    if baseline_shapes:
        bindings = dict(baseline_shapes)
    if shape_overrides:
        bindings.update(dict(shape_overrides))

    # Common axis aliases (align kernel-signature symbols with user-friendly names).
    if "batch" in bindings and "Z" not in bindings:
        bindings["Z"] = bindings["batch"]
    if "Z" in bindings and "batch" not in bindings:
        bindings["batch"] = bindings["Z"]

    # Add naming aliases to baseline to match IntentIR tensor names.
    # Add naming aliases to baseline to match IntentIR tensor names.
    # Use the macro intent here to avoid iterating over a huge expanded tensor set.
    baseline = _with_io_aliases(intent_macro, baseline)

    # Derive a few common implicit symbols.
    if "group" in bindings and "num_groups" not in bindings:
        bindings["num_groups"] = bindings["group"]
    if "num_groups" in bindings and "C" in bindings and "group_size" not in bindings:
        g = int(bindings["num_groups"])
        c = int(bindings["C"])
        if g > 0 and c % g == 0:
            bindings["group_size"] = c // g
    if "group_size" in bindings and "HW" in bindings and "num_elements" not in bindings:
        try:
            bindings["num_elements"] = int(bindings["group_size"]) * int(bindings["HW"])
        except Exception:
            pass

    tuning_info: dict | None = None
    tune_candidates = None
    if tune_request is not None:
        _log(f"[{frontend}:{kernel}] schedule selection: mode={tune_request.mode} budget={getattr(tune_request,'budget',0)}")
        if tune_profile:
            prof = load_profile(str(tune_profile))
            prof_src = str(tune_profile)
        else:
            _log(f"[{frontend}:{kernel}] query remote RVV profile (probe)")
            prof = query_remote_device(host, user=user, password=password, port=port, timeout=20)
            prof_src = "remote"

        shape_bindings_int: dict[str, int] = {}
        for k, v in dict(bindings).items():
            try:
                shape_bindings_int[str(k)] = int(v)
            except Exception:
                continue

        budget = int(getattr(tune_request, "budget", 0) or 0)
        if budget > 1:
            if int(bench_iters) <= 0:
                raise ValueError("tune-budget > 1 requires --bench-iters > 0 (measured autotune)")
            tune_candidates = propose_schedule_candidates(
                intent,
                shape_bindings=shape_bindings_int,
                profile=prof,
                request=tune_request,
                tile_hints=tile_hints,
                limit=budget,
                evidence=cert_v2,
            )
            if not tune_candidates:
                tune_candidates = propose_schedule_candidates(
                    intent,
                    shape_bindings=shape_bindings_int,
                    profile=prof,
                    request=tune_request,
                    tile_hints=tile_hints,
                    limit=1,
                    evidence=cert_v2,
                )
            # Keep a deterministic default schedule in case benchmarking fails.
            if tune_candidates:
                intent.schedule = tune_candidates[0].schedule
            tuning_info = {
                "profile_source": prof_src,
                "profile": prof.__dict__,
                "mode": str(tune_request.mode),
                "budget": int(budget),
                "tile_hints": list(tile_hints),
                "candidates_pred": [
                    {
                        "score": float(c.score),
                        "tile_mnk": (list(c.tile_mnk) if c.tile_mnk is not None else None),
                        "notes": list(c.notes),
                        "schedule": {
                            "tile_m": c.schedule.tile_m,
                            "tile_n": c.schedule.tile_n,
                            "tile_k": c.schedule.tile_k,
                            "vec_width": c.schedule.vec_width,
                            "pipeline_depth": c.schedule.pipeline_depth,
                            "axis_bindings": dict(c.schedule.axis_bindings or {}),
                            "vec_axis": c.schedule.vec_axis,
                            "parallel_axes": list(c.schedule.parallel_axes or []),
                            "memory_hint": dict(c.schedule.memory_hint or {}),
                        },
                    }
                    for c in tune_candidates
                ],
            }
        else:
            tuned = select_schedule(
                intent,
                shape_bindings=shape_bindings_int,
                profile=prof,
                request=tune_request,
                tile_hints=tile_hints,
                evidence=cert_v2,
            )
            intent.schedule = tuned.schedule
            tuning_info = {
                "profile_source": prof_src,
                "profile": prof.__dict__,
                "mode": str(tune_request.mode),
                "budget": int(budget),
                "tile_hints": list(tile_hints),
                "notes": list(tuned.notes),
                "schedule": (intent.to_json_dict().get("schedule") or {}),
            }
            if getattr(tuned, "debug", None) is not None:
                tuning_info["debug"] = tuned.debug

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    _log(f"[{frontend}:{kernel}] ssh connect: {user}@{host}:{port}")
    client.connect(hostname=host, port=port, username=user, password=password, timeout=20)
    sftp = client.open_sftp()
    remote_dir = f"/tmp/intentir_{kernel}_rvv"
    _sftp_mkdir_p(sftp, remote_dir)
    _log(f"[{frontend}:{kernel}] remote dir: {remote_dir}")

    # Prepare code + data according to kernel kind.
    remote_c = f"{remote_dir}/main.c"
    remote_bin = f"{remote_dir}/run"

    # Upload the target-side runtime (shared helpers) once per kernel.
    runtime_dir = ROOT / "backends" / "spmd_rvv" / "runtime"
    runtime_h = runtime_dir / "intentir_runtime.h"
    runtime_c_local = runtime_dir / "intentir_runtime.c"
    driver_h = runtime_dir / "intentir_driver.h"
    driver_c_local = runtime_dir / "intentir_driver.c"
    ops_h = runtime_dir / "intentir_ops.h"
    ops_c_local = runtime_dir / "intentir_ops.c"
    if (
        not runtime_h.exists()
        or not runtime_c_local.exists()
        or not driver_h.exists()
        or not driver_c_local.exists()
        or not ops_h.exists()
        or not ops_c_local.exists()
    ):
        raise FileNotFoundError(
            f"missing RVV runtime: {runtime_h} / {runtime_c_local} / {driver_h} / {driver_c_local} / {ops_h} / {ops_c_local}"
        )
    _log(f"[{frontend}:{kernel}] upload runtime")
    with sftp.file(f"{remote_dir}/intentir_runtime.h", "w") as f:
        f.write(runtime_h.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_runtime.c", "w") as f:
        f.write(runtime_c_local.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_driver.h", "w") as f:
        f.write(driver_h.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_driver.c", "w") as f:
        f.write(driver_c_local.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_ops.h", "w") as f:
        f.write(ops_h.read_text(encoding="utf-8"))
    with sftp.file(f"{remote_dir}/intentir_ops.c", "w") as f:
        f.write(ops_c_local.read_text(encoding="utf-8"))

    # Generic lowering path: upload all external inputs and reference outputs, then lower ops list.
    produced = {op.output for op in intent.ops if op.output}
    used = set()
    for op in intent.ops:
        for n in op.inputs:
            used.add(n)
    external_inputs = sorted([n for n in used if n in intent.tensors and n not in produced])
    outputs = list(intent.outputs)

    # Upload inputs
    _log(f"[{frontend}:{kernel}] upload inputs/refs")
    for name in external_inputs:
        if name not in baseline:
            raise RuntimeError(f"baseline missing input tensor {name} for {kernel}")
        dt = intent.tensors[name].dtype
        arr = np.asarray(baseline[name])
        if dt in {"bool", "i1"}:
            raw = np.asarray(arr, dtype=np.uint8).tobytes(order="C")
        else:
            raw = np.asarray(arr, dtype=np.float32).tobytes(order="C")
        _sftp_write_bytes(sftp, f"{remote_dir}/{name}.bin", raw)

    # Upload reference outputs
    for name in outputs:
        if name not in baseline:
            raise RuntimeError(f"baseline missing output tensor {name} for {kernel}")
        dt = intent.tensors[name].dtype
        arr = np.asarray(baseline[name])
        if dt in {"bool", "i1"}:
            raw = np.asarray(arr, dtype=np.uint8).tobytes(order="C")
        else:
            raw = np.asarray(arr, dtype=np.float32).tobytes(order="C")
        _sftp_write_bytes(sftp, f"{remote_dir}/{name}_ref.bin", raw)

    # Lower the IntentIR op list to C and run on RVV target.
    #
    # Remote runs compare against a GPU-produced baseline (Triton/TileLang CUDA),
    # so keep tolerances at least legacy-default to avoid false negatives.
    auto_tol = infer_tolerances(intent, ref_out=baseline).to_dict()
    atol_use = max(float(auto_tol.get("atol", 1e-3)), 1e-3)
    rtol_use = max(float(auto_tol.get("rtol", 1e-3)), 1e-3)
    backend_used = "cpp"

    def _compile_and_run(schedule) -> dict:
        # Generate + upload code for this schedule.
        intent.schedule = schedule
        src = lower_intent_to_c_with_files(intent, shape_bindings=bindings, atol=float(atol_use), rtol=float(rtol_use))
        with sftp.file(remote_c, "w") as f:
            f.write(src)

        _log(f"[{frontend}:{kernel}] remote compile")
        compile_cmd = (
            f"gcc -O2 -std=c11 -march=rv64gcv -I{remote_dir} -o {remote_bin} {remote_c} "
            f"{remote_dir}/intentir_runtime.c {remote_dir}/intentir_driver.c {remote_dir}/intentir_ops.c -lm -lrt"
        )
        stdin, stdout, stderr = client.exec_command(compile_cmd, timeout=60)
        comp_out = stdout.read().decode()
        comp_err = stderr.read().decode()
        compile_rc = stdout.channel.recv_exit_status()
        if compile_rc != 0:
            return {
                "compile_rc": compile_rc,
                "compile_stdout": comp_out,
                "compile_stderr": comp_err,
                "run_rc": None,
                "stdout": "",
                "stderr": "",
                "bench": None,
                "profile_ops": None,
            }

        _log(f"[{frontend}:{kernel}] remote run")
        env_prefix = ""
        if bool(profile_ops):
            env_prefix += "INTENTIR_PROFILE_OPS=1 "
        run_cmd = f"cd {remote_dir} && {env_prefix}{remote_bin}"
        if int(bench_iters) > 0:
            bi = int(bench_iters)
            bw = int(bench_warmup)
            if bw < 0:
                bw = 0
            run_cmd = f"cd {remote_dir} && {env_prefix}INTENTIR_BENCH_ITERS={bi} INTENTIR_BENCH_WARMUP={bw} {remote_bin}"
        stdin, stdout, stderr = client.exec_command(run_cmd, timeout=60)
        run_out = stdout.read().decode()
        run_err = stderr.read().decode()
        run_rc = stdout.channel.recv_exit_status()

        bench = None
        try:
            for ln in str(run_out).splitlines():
                if ln.startswith("INTENTIR_BENCH "):
                    bench = json.loads(ln[len("INTENTIR_BENCH ") :].strip())
                    break
        except Exception:
            bench = None

        prof = None
        try:
            for ln in str(run_out).splitlines():
                if ln.startswith("INTENTIR_PROFILE "):
                    prof = json.loads(ln[len("INTENTIR_PROFILE ") :].strip())
                    break
        except Exception:
            prof = None

        return {
            "compile_rc": compile_rc,
            "compile_stdout": comp_out,
            "compile_stderr": comp_err,
            "run_rc": run_rc,
            "stdout": run_out,
            "stderr": run_err,
            "bench": bench,
            "profile_ops": prof,
        }

    chosen_schedule = intent.schedule
    chosen = None
    if tune_candidates is not None and len(tune_candidates) > 0:
        # Measured autotune: benchmark top-K candidates and pick the best passing one.
        cand_runs: list[dict] = []
        best_idx = None
        best_ns = None
        for i, c in enumerate(tune_candidates):
            r = _compile_and_run(c.schedule)
            cand_runs.append(
                {
                    "idx": int(i),
                    "pred_score": float(c.score),
                    "tile_mnk": (list(c.tile_mnk) if c.tile_mnk is not None else None),
                    "notes": list(c.notes),
                    "schedule": {
                        "tile_m": c.schedule.tile_m,
                        "tile_n": c.schedule.tile_n,
                        "tile_k": c.schedule.tile_k,
                        "vec_width": c.schedule.vec_width,
                        "pipeline_depth": c.schedule.pipeline_depth,
                    },
                    "compile_rc": r.get("compile_rc"),
                    "run_rc": r.get("run_rc"),
                    "bench": r.get("bench"),
                }
            )
            if r.get("compile_rc") != 0 or r.get("run_rc") != 0:
                continue
            b = r.get("bench") or {}
            ns = b.get("ns_per_iter")
            if isinstance(ns, (int, float)) and ns > 0:
                if best_ns is None or float(ns) < float(best_ns):
                    best_ns = float(ns)
                    best_idx = int(i)
                    chosen = r
                    chosen_schedule = c.schedule
            elif best_idx is None:
                # No bench info (shouldn't happen if bench_iters>0), but keep the first passing run.
                best_idx = int(i)
                chosen = r
                chosen_schedule = c.schedule

        if tuning_info is None:
            tuning_info = {}
        tuning_info["measured_autotune"] = {
            "evaluated": cand_runs,
            "best_index": best_idx,
            "best_ns_per_iter": best_ns,
        }
        if chosen is None:
            # Fallback: run once with the current schedule (already set to the first candidate above).
            chosen = _compile_and_run(intent.schedule)
    else:
        chosen = _compile_and_run(intent.schedule)

    sftp.close()
    client.close()

    rc = int(chosen.get("compile_rc") or 0)
    run_rc = int(chosen.get("run_rc") or 0)
    run_out = str(chosen.get("stdout") or "")
    run_err = str(chosen.get("stderr") or "")
    bench = chosen.get("bench")
    prof = chosen.get("profile_ops")
    # Include a compact baseline summary for quick inspection (avoid huge blobs).
    baseline_summary = {}
    try:
        if kernel == "any_kernel_dim" and "out" in baseline:
            out_ref = np.asarray(baseline["out"]).reshape(-1).astype(np.uint8)
            baseline_summary = {
                "out_len": int(out_ref.size),
                "out_sum": int(out_ref.sum()),
                "out_first": [int(x) for x in out_ref[: min(32, out_ref.size)]],
            }
        elif kernel == "group_norm_kernel" and "Y" in baseline:
            y = np.asarray(baseline["Y"], dtype=np.float32).reshape(-1)
            baseline_summary = {"Y_len": int(y.size), "Y_mean": float(y.mean()), "Y_std": float(y.std())}
        elif kernel == "_attn_fwd" and "Out" in baseline:
            o = np.asarray(baseline["Out"], dtype=np.float32).reshape(-1)
            baseline_summary = {"Out_len": int(o.size), "Out_mean": float(o.mean()), "Out_std": float(o.std())}
        elif kernel == "softmax_inner" and "output" in baseline:
            o = np.asarray(baseline["output"], dtype=np.float32).reshape(-1)
            baseline_summary = {"output_len": int(o.size), "output_mean": float(o.mean()), "output_std": float(o.std())}
        elif kernel == "layer_norm_persistent" and "out_ptr" in baseline:
            o = np.asarray(baseline["out_ptr"], dtype=np.float32).reshape(-1)
            baseline_summary = {"out_ptr_len": int(o.size), "out_ptr_mean": float(o.mean()), "out_ptr_std": float(o.std())}
    except Exception:
        baseline_summary = {}
    return {
        "backend": backend_used,
        "compile_rc": rc,
        "run_rc": run_rc,
        "stdout": run_out,
        "stderr": run_err,
        "baseline_summary": baseline_summary,
        "tuning": tuning_info,
        "bench": bench,
        "profile_ops": prof,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="any_kernel_dim")
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--password", default=None, help="SSH password (prefer env INTENTIR_SSH_PASSWORD or prompt)")
    ap.add_argument("--use-key", action="store_true", help="use SSH key auth (no password prompt)")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--case-index", type=int, default=0, help="pick case from artifacts report (default 0)")
    ap.add_argument("--baseline-npz", default=None, help="override baseline npz path (default: from artifact report)")
    ap.add_argument("--prefer-live-baseline", action="store_true", help="re-launch Triton for baseline even if npz exists")
    ap.add_argument("--no-tune", action="store_true", help="disable backend schedule selection (use IntentIR schedule as-is)")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default="auto")
    ap.add_argument("--tune-budget", type=int, default=1, help="if >1, benchmark top-K predicted schedules (requires --bench-iters>0)")
    ap.add_argument("--tune-debug", action="store_true", help="include structured tuning/cost-model debug in JSON output")
    ap.add_argument("--lock", action="append", default=[], help="repeatable; e.g. --lock tile_n=128")
    ap.add_argument("--constraint", action="append", default=[], help="repeatable; e.g. --constraint 'tile_n in (64,128)'")
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: query remote host)")
    ap.add_argument("--bench-iters", type=int, default=0, help="if >0, run microbenchmark loop and print INTENTIR_BENCH JSON line")
    ap.add_argument("--bench-warmup", type=int, default=1, help="warmup iterations for benchmark loop")
    ap.add_argument("--profile-ops", action="store_true", help="emit per-op timing JSON line (INTENTIR_PROFILE) from the RVV program")
    ap.add_argument("--json", action="store_true", help="print result as JSON (stable for tooling)")
    ap.add_argument("--quiet", action="store_true", help="disable progress logs")
    args = ap.parse_args()
    password: str | None = None
    if not bool(args.use_key):
        password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
        if password is None:
            password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")
    tune_req = None
    if not bool(args.no_tune):
        tune_req = TuningRequest(
            mode=str(args.tune_mode),
            budget=int(args.tune_budget),
            locks=parse_locks(args.lock or []),
            constraints=parse_constraints(args.constraint or []),
            debug=bool(args.tune_debug),
        )
    def _log(msg: str) -> None:
        if bool(args.quiet):
            return
        print(str(msg), file=sys.stderr, flush=True)

    res = run_remote(
        args.kernel,
        args.frontend,
        args.host,
        args.user,
        password,
        port=args.port,
        case_index=args.case_index,
        baseline_npz=args.baseline_npz,
        prefer_live_baseline=bool(args.prefer_live_baseline),
        tune_request=tune_req,
        tune_profile=str(args.profile) if args.profile else None,
        bench_iters=int(args.bench_iters),
        bench_warmup=int(args.bench_warmup),
        profile_ops=bool(args.profile_ops),
        log=_log,
    )
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(res)


if __name__ == "__main__":
    main()
