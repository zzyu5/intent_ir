#!/usr/bin/env python3
"""
Unified IntentIR user-facing CLI.

This provides one stable entrypoint for:
- suite-level runs (full/category/smoke)
- single-kernel runs
- environment inspection
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: Sequence[str], *, stream: bool, dry_run: bool) -> int:
    pretty = " ".join(str(x) for x in cmd)
    print(f"[intentir] $ {pretty}", flush=True)
    if dry_run:
        return 0
    if stream:
        p = subprocess.run(list(cmd), cwd=str(ROOT))
    else:
        p = subprocess.run(list(cmd), cwd=str(ROOT), capture_output=True, text=True)
        if p.stdout:
            print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, file=sys.stderr, end="")
    return int(p.returncode)


def _python_cmd(script: str, *extra: str) -> list[str]:
    return [sys.executable, script, *extra]


def _cmd_suite(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root) if args.out_root else (ROOT / "artifacts" / "intentir_suite")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.suite == "flaggems-full196":
        rc = _run(
            _python_cmd("scripts/flaggems/build_coverage_batches.py"),
            stream=bool(args.stream),
            dry_run=bool(args.dry_run),
        )
        if rc != 0:
            return rc
        cmd = _python_cmd(
            "scripts/flaggems/run_coverage_batches.py",
            "--out-root",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--resume" if args.resume else "--no-resume",
        )
        for fam in list(args.family or []):
            cmd.extend(["--family", str(fam)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))

    if args.suite == "triton-smoke":
        cmd = _python_cmd(
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            "smoke",
            "--out-dir",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--stream-subprocess-output" if args.stream else "--no-stream-subprocess-output",
        )
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))

    if args.suite == "tilelang-smoke":
        cmd = _python_cmd(
            "scripts/tilelang/full_pipeline_verify.py",
            "--out-dir",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))

    if args.suite == "cuda-smoke":
        cmd = _python_cmd(
            "scripts/cuda/full_pipeline_verify.py",
            "--out-dir",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))

    print(f"unsupported suite: {args.suite}", file=sys.stderr)
    return 2


def _cmd_kernel(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "intentir_kernel")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.frontend == "triton":
        cmd = _python_cmd(
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            "smoke",
            "--kernel",
            str(args.kernel),
            "--out-dir",
            str(out_dir),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--stream-subprocess-output" if args.stream else "--no-stream-subprocess-output",
        )
        if str(args.provider) == "native":
            cmd.extend(["--flaggems-path", "original"])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))

    if args.frontend == "tilelang":
        cmd = _python_cmd(
            "scripts/tilelang/full_pipeline_verify.py",
            "--kernel",
            str(args.kernel),
            "--out-dir",
            str(out_dir),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))

    if args.frontend == "cuda":
        cmd = _python_cmd(
            "scripts/cuda/full_pipeline_verify.py",
            "--kernel",
            str(args.kernel),
            "--out-dir",
            str(out_dir),
            "--cases-limit",
            str(int(args.cases_limit)),
        )
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))

    print(f"unsupported frontend: {args.frontend}", file=sys.stderr)
    return 2


def _import_version(name: str) -> tuple[str, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return ("missing", str(e))
    version = str(getattr(mod, "__version__", "unknown"))
    location = str(getattr(mod, "__file__", "built-in"))
    return (version, location)


def _cmd_env(_: argparse.Namespace) -> int:
    print(f"repo_root: {ROOT}")
    print(f"python: {sys.version.splitlines()[0]}")
    for name in ("torch", "triton", "tilelang", "flag_gems"):
        ver, loc = _import_version(name)
        print(f"{name}: version={ver} location={loc}")
    try:
        import torch  # type: ignore

        print(f"torch.cuda.is_available: {bool(torch.cuda.is_available())}")
        if bool(torch.cuda.is_available()):
            print(f"torch.cuda.device_count: {int(torch.cuda.device_count())}")
            print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"cuda_probe_error: {e}")
    print(f"rvv_default_host: {os.getenv('INTENTIR_RVV_HOST', '192.168.8.72')}")
    print(f"rvv_default_user: {os.getenv('INTENTIR_RVV_USER', 'ubuntu')}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IntentIR unified CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    suite = sub.add_parser("suite", help="Run suite-level verification")
    suite.add_argument("--suite", choices=["flaggems-full196", "triton-smoke", "tilelang-smoke", "cuda-smoke"], required=True)
    suite.add_argument("--out-root", default=None)
    suite.add_argument("--family", action="append", default=[])
    suite.add_argument("--kernel", action="append", default=[])
    suite.add_argument("--cases-limit", type=int, default=8)
    suite.add_argument("--family-kernel-chunk-size", type=int, default=12)
    suite.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--flaggems-path", choices=["intentir", "original"], default="intentir")
    suite.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="force_compile")
    suite.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    suite.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--rvv-host", default="192.168.8.72")
    suite.add_argument("--rvv-user", default="ubuntu")
    suite.add_argument("--rvv-port", type=int, default=22)
    suite.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    suite.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    suite.set_defaults(func=_cmd_suite)

    kernel = sub.add_parser("kernel", help="Run one kernel")
    kernel.add_argument("--frontend", choices=["triton", "tilelang", "cuda"], required=True)
    kernel.add_argument("--provider", choices=["native", "flaggems"], default="flaggems")
    kernel.add_argument("--kernel", required=True)
    kernel.add_argument("--out-dir", default=None)
    kernel.add_argument("--cases-limit", type=int, default=8)
    kernel.add_argument("--flaggems-path", choices=["intentir", "original"], default="intentir")
    kernel.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="force_compile")
    kernel.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    kernel.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--rvv-host", default="192.168.8.72")
    kernel.add_argument("--rvv-user", default="ubuntu")
    kernel.add_argument("--rvv-port", type=int, default=22)
    kernel.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    kernel.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    kernel.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    kernel.set_defaults(func=_cmd_kernel)

    env = sub.add_parser("env", help="Show environment and dependency status")
    env.set_defaults(func=_cmd_env)

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    rc = int(args.func(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
