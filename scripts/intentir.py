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
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: Sequence[str], *, stream: bool, dry_run: bool, env_overrides: dict[str, str] | None = None) -> int:
    pretty = " ".join(str(x) for x in cmd)
    print(f"[intentir] $ {pretty}", flush=True)
    if dry_run:
        return 0
    env = None
    if env_overrides:
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in env_overrides.items()})
    if stream:
        p = subprocess.run(list(cmd), cwd=str(ROOT), env=env)
    else:
        p = subprocess.run(list(cmd), cwd=str(ROOT), capture_output=True, text=True, env=env)
        if p.stdout:
            print(p.stdout, end="")
        if p.stderr:
            print(p.stderr, file=sys.stderr, end="")
    return int(p.returncode)


def _python_cmd(script: str, *extra: str) -> list[str]:
    return [sys.executable, script, *extra]


def _execution_ir_default() -> str:
    return "mlir"


def _cmd_suite(args: argparse.Namespace) -> int:
    out_root = Path(args.out_root) if args.out_root else (ROOT / "artifacts" / "intentir_suite")
    out_root.mkdir(parents=True, exist_ok=True)
    env_overrides = {"INTENTIR_EXECUTION_IR": str(args.execution_ir)}

    if args.suite == "flaggems-full196":
        rc = _run(
            _python_cmd("scripts/flaggems/build_coverage_batches.py"),
            stream=bool(args.stream),
            dry_run=bool(args.dry_run),
            env_overrides=env_overrides,
        )
        if rc != 0:
            return rc
        cmd = _python_cmd(
            "scripts/flaggems/run_coverage_batches.py",
            "--out-root",
            str(out_root),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--execution-ir",
            str(args.execution_ir),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            str(args.intentir_miss_policy),
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
            "--progress-style",
            str(args.progress_style),
            "--stream-subprocess-output"
            if bool(args.stream_subprocess_detail)
            else "--no-stream-subprocess-output",
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--skip-rvv-local" if args.skip_rvv_local else "--no-skip-rvv-local",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--resume" if args.resume else "--no-resume",
        )
        if bool(args.write_registry):
            cmd.append("--write-registry")
        for fam in list(args.family or []):
            cmd.extend(["--family", str(fam)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "gpu-perf-graph":
        rc = _run(
            _python_cmd("scripts/flaggems/build_coverage_batches.py"),
            stream=bool(args.stream),
            dry_run=bool(args.dry_run),
            env_overrides=env_overrides,
        )
        if rc != 0:
            return rc
        cmd = _python_cmd(
            "scripts/flaggems/run_gpu_perf_graph.py",
            "--out-root",
            str(out_root),
            "--family-kernel-chunk-size",
            str(int(args.family_kernel_chunk_size)),
            "--threshold",
            str(float(args.gpu_perf_threshold)),
            "--warmup",
            str(int(args.perf_warmup)),
            "--iters",
            str(int(args.perf_iters)),
            "--repeats",
            str(int(args.perf_repeats)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--progress-style",
            str(args.progress_style),
            "--resume" if args.resume else "--no-resume",
            "--stream" if args.stream else "--no-stream",
        )
        for fam in list(args.family or []):
            cmd.extend(["--family", str(fam)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

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
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--skip-rvv-local" if args.skip_rvv_local else "--no-skip-rvv-local",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--stream-subprocess-output" if args.stream else "--no-stream-subprocess-output",
        )
        if bool(args.write_registry):
            cmd.append("--write-registry")
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    if args.suite == "flaggems-coverage-single":
        cmd = _python_cmd(
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            "coverage",
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
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--allow-cuda-skip" if args.allow_cuda_skip else "--no-allow-cuda-skip",
            "--run-rvv-remote" if args.run_rvv_remote else "--no-run-rvv-remote",
            "--skip-rvv-local" if args.skip_rvv_local else "--no-skip-rvv-local",
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--rvv-use-key" if args.rvv_use_key else "--no-rvv-use-key",
            "--stream-subprocess-output" if args.stream else "--no-stream-subprocess-output",
        )
        if bool(args.write_registry):
            cmd.append("--write-registry")
        for kernel in list(args.kernel or []):
            cmd.extend(["--kernel", str(kernel)])
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

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
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

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
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

    print(f"unsupported suite: {args.suite}", file=sys.stderr)
    return 2


def _cmd_kernel(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "intentir_kernel")
    out_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = {"INTENTIR_EXECUTION_IR": str(args.execution_ir)}

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
            "--pipeline-timeout-sec",
            str(int(args.pipeline_timeout_sec)),
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
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

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
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

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
        return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run), env_overrides=env_overrides)

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
    print(f"intentir.execution_ir: {os.getenv('INTENTIR_EXECUTION_IR', 'mlir')}")
    return 0


def _cmd_mlir_check(args: argparse.Namespace) -> int:
    from intent_ir.mlir import detect_mlir_toolchain, to_mlir, to_intent  # noqa: PLC0415
    from intent_ir.ir import IntentFunction  # noqa: PLC0415

    toolchain = detect_mlir_toolchain()
    missing = [str(x) for x in list(toolchain.get("missing_required_tools") or []) if str(x).strip()]
    report: dict[str, object] = {"toolchain": toolchain}
    report["toolchain_required_ok"] = bool(toolchain.get("ok"))
    report["toolchain_required_missing"] = missing
    if missing:
        report["toolchain_install_hint"] = (
            "Install MLIR/LLVM tools so these binaries are on PATH: "
            + ", ".join(missing)
        )
    if args.intent_json is not None:
        payload = json.loads(Path(args.intent_json).read_text(encoding="utf-8"))
        if "intent" in payload and isinstance(payload.get("intent"), dict):
            payload = payload["intent"]
        intent = IntentFunction.from_json_dict(payload)
        module = to_mlir(intent)
        _ = to_intent(module)
        report["roundtrip_ok"] = True
        report["intent_name"] = str(intent.name)
        report["symbol_count"] = len(list(module.symbols or []))
    out = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[intentir] wrote {out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _cmd_mlir_pass(args: argparse.Namespace) -> int:
    from intent_ir.mlir import run_pipeline, to_mlir  # noqa: PLC0415
    from intent_ir.ir import IntentFunction  # noqa: PLC0415

    payload = json.loads(Path(args.intent_json).read_text(encoding="utf-8"))
    if "intent" in payload and isinstance(payload.get("intent"), dict):
        payload = payload["intent"]
    intent = IntentFunction.from_json_dict(payload)
    module = to_mlir(intent)
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "intentir_mlir_pass")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_module, trace = run_pipeline(
        module,
        str(args.pipeline),
        backend=(str(args.backend) if args.backend else None),
        out_dir=out_dir,
        fail_on_error=bool(args.fail_on_error),
    )
    out_mlir = out_dir / "module.mlir"
    out_mlir.write_text(out_module.module_text, encoding="utf-8")
    out_trace = out_dir / "pass_trace.json"
    if "pass_trace_path" not in trace:
        out_trace.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[intentir] mlir module: {out_mlir}")
    print(f"[intentir] pass trace: {out_trace}")
    return 0 if bool(trace.get("ok")) else 1


def _cmd_mlir_provision_toolchain(args: argparse.Namespace) -> int:
    cmd = _python_cmd(
        "scripts/intentir/provision_mlir_toolchain.py",
        "--version",
        str(int(args.version)),
    )
    if args.prefix is not None:
        cmd.extend(["--prefix", str(args.prefix)])
    if args.toolchain_root is not None:
        cmd.extend(["--toolchain-root", str(args.toolchain_root)])
    if args.current_link is not None:
        cmd.extend(["--current-link", str(args.current_link)])
    if args.env_file is not None:
        cmd.extend(["--env-file", str(args.env_file)])
    cmd.append("--force" if bool(args.force) else "--no-force")
    cmd.append("--use-current-link" if bool(args.use_current_link) else "--no-use-current-link")
    if args.out is not None:
        cmd.extend(["--out", str(args.out)])
    return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))


def _cmd_tilelang_export_cuda_snapshots(args: argparse.Namespace) -> int:
    cmd = _python_cmd(
        "scripts/tilelang/export_cuda_snapshots.py",
        "--out-dir",
        str(args.out_dir),
    )
    if bool(args.refresh):
        cmd.append("--refresh")
    for kernel in list(args.kernel or []):
        cmd.extend(["--kernel", str(kernel)])
    return _run(cmd, stream=bool(args.stream), dry_run=bool(args.dry_run))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IntentIR unified CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    suite = sub.add_parser("suite", help="Run suite-level verification")
    suite.add_argument(
        "--suite",
        choices=[
            "flaggems-full196",
            "gpu-perf-graph",
            "flaggems-coverage-single",
            "triton-smoke",
            "tilelang-smoke",
            "cuda-smoke",
        ],
        required=True,
    )
    suite.add_argument("--out-root", default=None)
    suite.add_argument("--family", action="append", default=[])
    suite.add_argument("--kernel", action="append", default=[])
    suite.add_argument("--cases-limit", type=int, default=8)
    suite.add_argument("--execution-ir", choices=["mlir"], default=_execution_ir_default())
    suite.add_argument("--family-kernel-chunk-size", type=int, default=12)
    suite.add_argument("--gpu-perf-threshold", type=float, default=0.80)
    suite.add_argument("--perf-warmup", type=int, default=20)
    suite.add_argument("--perf-iters", type=int, default=200)
    suite.add_argument("--perf-repeats", type=int, default=5)
    suite.add_argument(
        "--pipeline-timeout-sec",
        type=int,
        default=0,
        help="Pipeline stage timeout (seconds) for matrix runners; 0 disables.",
    )
    suite.add_argument("--progress-style", choices=["auto", "tqdm", "plain", "chunk", "none"], default="auto")
    suite.add_argument(
        "--stream-subprocess-detail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward verbose per-kernel backend logs from internal matrix runners (default: false).",
    )
    suite.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--write-registry", action=argparse.BooleanOptionalAction, default=False)
    suite.add_argument("--flaggems-path", choices=["intentir", "original"], default="intentir")
    suite.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="auto")
    suite.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="strict")
    suite.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--skip-rvv-local", action=argparse.BooleanOptionalAction, default=True)
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
    kernel.add_argument(
        "--pipeline-timeout-sec",
        type=int,
        default=0,
        help="Pipeline stage timeout (seconds) for matrix runners; 0 disables.",
    )
    kernel.add_argument("--execution-ir", choices=["mlir"], default=_execution_ir_default())
    kernel.add_argument("--flaggems-path", choices=["intentir", "original"], default="intentir")
    kernel.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="auto")
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

    tilelang = sub.add_parser("tilelang", help="TileLang helper commands")
    tilelang_sub = tilelang.add_subparsers(dest="tilelang_cmd", required=True)
    tilelang_export = tilelang_sub.add_parser("export-cuda-snapshots", help="Export TileLang kernels to CUDA snapshots")
    tilelang_export.add_argument("--out-dir", default=str(ROOT / "kernels" / "cuda" / "ops" / "snapshots"))
    tilelang_export.add_argument("--kernel", action="append", default=[])
    tilelang_export.add_argument("--refresh", action=argparse.BooleanOptionalAction, default=False)
    tilelang_export.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    tilelang_export.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    tilelang_export.set_defaults(func=_cmd_tilelang_export_cuda_snapshots)

    mlir = sub.add_parser("mlir", help="MLIR migration helpers")
    mlir_sub = mlir.add_subparsers(dest="mlir_cmd", required=True)

    mlir_check = mlir_sub.add_parser("check", help="Probe MLIR toolchain and optional Intent roundtrip")
    mlir_check.add_argument("--intent-json", default=None, help="Path to intent JSON (or report with `intent` field)")
    mlir_check.add_argument("--out", default=None, help="Optional JSON output path")
    mlir_check.set_defaults(func=_cmd_mlir_check)

    mlir_pass = mlir_sub.add_parser("pass", help="Run MLIR pipeline on one intent payload")
    mlir_pass.add_argument("--intent-json", required=True, help="Path to intent JSON (or report with `intent` field)")
    mlir_pass.add_argument(
        "--pipeline",
        required=True,
        choices=["upstream", "midend", "downstream_cuda", "downstream_rvv", "downstream_cuda_llvm"],
    )
    mlir_pass.add_argument("--backend", default=None, help="Optional backend hint (cuda/rvv)")
    mlir_pass.add_argument("--out-dir", default=None, help="Output directory for module + pass trace")
    mlir_pass.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=False)
    mlir_pass.set_defaults(func=_cmd_mlir_pass)

    mlir_provision = mlir_sub.add_parser(
        "provision-toolchain",
        help="Provision repository-local MLIR toolchain (apt download + extract, no sudo required)",
    )
    mlir_provision.add_argument("--version", type=int, default=14)
    mlir_provision.add_argument("--prefix", default=None)
    mlir_provision.add_argument("--toolchain-root", default=None)
    mlir_provision.add_argument("--current-link", default=None)
    mlir_provision.add_argument("--env-file", default=None)
    mlir_provision.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    mlir_provision.add_argument("--use-current-link", action=argparse.BooleanOptionalAction, default=True)
    mlir_provision.add_argument("--out", default=None)
    mlir_provision.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    mlir_provision.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    mlir_provision.set_defaults(func=_cmd_mlir_provision_toolchain)

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    rc = int(args.func(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
