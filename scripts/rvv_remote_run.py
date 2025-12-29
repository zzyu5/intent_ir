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

import paramiko
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.codegen.intentir_to_c import lower_intent_to_c_with_files
from intent_ir.ir import IntentFunction
from intent_ir.macros import expand_macros
from verify.gen_cases import TestCase


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
    password: str,
    port: int = 22,
    case_index: int = 0,
    shape_overrides: dict | None = None,
    baseline_npz: str | None = None,
    prefer_live_baseline: bool = False,
):
    artifact_dir = "full_pipeline_verify" if frontend == "triton" else "tilelang_full_pipeline"
    report_path = ROOT / "artifacts" / artifact_dir / f"{kernel}.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"artifact not found: {report_path}, please run scripts/{frontend}/full_pipeline_verify.py first"
        )
    report = json.loads(report_path.read_text())
    intent_macro = IntentFunction.from_json_dict(report["intent"])
    intent_expanded_json = report.get("intent_expanded")
    if isinstance(intent_expanded_json, dict):
        intent = IntentFunction.from_json_dict(intent_expanded_json)
    else:
        intent = expand_macros(intent_macro)
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

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, password=password, timeout=20)
    sftp = client.open_sftp()
    remote_dir = f"/tmp/intentir_{kernel}_rvv"
    _sftp_mkdir_p(sftp, remote_dir)

    # Prepare code + data according to kernel kind.
    remote_c = f"{remote_dir}/main.c"
    remote_bin = f"{remote_dir}/run"

    # Generic lowering path: upload all external inputs and reference outputs, then lower ops list.
    produced = {op.output for op in intent.ops if op.output}
    used = set()
    for op in intent.ops:
        for n in op.inputs:
            used.add(n)
    external_inputs = sorted([n for n in used if n in intent.tensors and n not in produced])
    outputs = list(intent.outputs)

    # Upload inputs
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
    tol = {
        "any_kernel_dim": (0.0, 0.0),
        "group_norm_kernel": (1e-3, 1e-3),
        "_attn_fwd": (1e-2, 1e-2),
        "softmax_inner": (1e-3, 1e-3),
        "layer_norm_persistent": (1e-3, 1e-3),
        # upsample is currently OUT_OF_SCOPE; keep a loose tol for experiments.
        "upsample_bicubic2d_aa": (1e-3, 1e-3),
    }
    atol_use, rtol_use = tol.get(kernel, (1e-3, 1e-3))
    src = lower_intent_to_c_with_files(intent, shape_bindings=bindings, atol=float(atol_use), rtol=float(rtol_use))
    backend_used = "cpp"
    with sftp.file(remote_c, "w") as f:
        f.write(src)

    sftp.close()

    compile_cmd = f"gcc -O2 -std=c11 -march=rv64gcv -o {remote_bin} {remote_c} -lm"
    stdin, stdout, stderr = client.exec_command(compile_cmd, timeout=60)
    comp_out = stdout.read().decode()
    comp_err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        client.close()
        raise RuntimeError(f"remote compile failed rc={rc} stderr={comp_err}")

    stdin, stdout, stderr = client.exec_command(f"cd {remote_dir} && {remote_bin}", timeout=60)
    run_out = stdout.read().decode()
    run_err = stderr.read().decode()
    run_rc = stdout.channel.recv_exit_status()
    client.close()
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="any_kernel_dim")
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--password", default=None, help="SSH password (prefer env INTENTIR_SSH_PASSWORD or prompt)")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--case-index", type=int, default=0, help="pick case from artifacts report (default 0)")
    ap.add_argument("--baseline-npz", default=None, help="override baseline npz path (default: from artifact report)")
    ap.add_argument("--prefer-live-baseline", action="store_true", help="re-launch Triton for baseline even if npz exists")
    args = ap.parse_args()
    password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
    if password is None:
        password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")
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
    )
    print(res)


if __name__ == "__main__":
    main()
