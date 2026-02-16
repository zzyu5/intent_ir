"""
Run FlagGems end-to-end matrix and converge registry status.

Stages:
1) Triton provider pipeline (flaggems)
2) RVV local backend smoke
3) CUDA local backend smoke
4) Status convergence report
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")


def _with_env_prefix(cmd: list[str], env_map: dict[str, str] | None) -> list[str]:
    if not env_map:
        return list(cmd)
    pairs = [f"{k}={v}" for k, v in sorted(env_map.items()) if str(k).strip() and str(v).strip()]
    if not pairs:
        return list(cmd)
    return ["env", *pairs, *list(cmd)]


def _load_active_semantic_ops(active_batch_path: Path) -> list[str]:
    if not active_batch_path.is_file():
        return []
    try:
        payload = json.loads(active_batch_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        op = str(it.get("semantic_op") or "").strip()
        if op and op not in out:
            out.append(op)
    return out


def _suite_kernel_names(*, suite: str, flaggems_opset: str, backend_target: str) -> list[str]:
    if str(suite) == "smoke":
        from pipeline.triton.providers.flaggems.specs import default_flaggems_kernel_specs  # noqa: PLC0415

        specs = default_flaggems_kernel_specs(
            flaggems_opset=str(flaggems_opset),
            backend_target=str(backend_target),
        )
    else:
        from pipeline.triton.providers.flaggems.specs import coverage_flaggems_kernel_specs  # noqa: PLC0415

        specs = coverage_flaggems_kernel_specs(
            flaggems_opset=str(flaggems_opset),
            backend_target=str(backend_target),
        )
    return [str(s.name) for s in specs]


def _collect_missing_provider_reports(provider_report_dir: Path, kernels: list[str]) -> list[str]:
    missing: list[str] = []
    for kernel in kernels:
        report_path = provider_report_dir / f"{kernel}.json"
        if not report_path.is_file():
            missing.append(str(kernel))
    return missing


def _resolve_suite_and_kernel_filter(
    *,
    requested_suite: str,
    requested_kernels: list[str],
    flaggems_opset: str,
    backend_target: str,
) -> tuple[str, list[str]]:
    suite = str(requested_suite)
    kernels_raw = [str(k) for k in (requested_kernels or []) if str(k).strip()]
    kernels: list[str] = []
    for k in kernels_raw:
        if k not in kernels:
            kernels.append(k)

    if not kernels:
        if suite in {"coverage", "all"}:
            return "coverage", _suite_kernel_names(
                suite="coverage",
                flaggems_opset=str(flaggems_opset),
                backend_target=str(backend_target),
            )
        return suite, []

    if suite == "smoke":
        smoke = set(
            _suite_kernel_names(
                suite="smoke",
                flaggems_opset=str(flaggems_opset),
                backend_target=str(backend_target),
            )
        )
        missing_from_smoke = [k for k in kernels if k not in smoke]
        if not missing_from_smoke:
            return "smoke", kernels

        coverage = set(
            _suite_kernel_names(
                suite="coverage",
                flaggems_opset=str(flaggems_opset),
                backend_target=str(backend_target),
            )
        )
        unknown = [k for k in missing_from_smoke if k not in coverage]
        if unknown:
            raise SystemExit(
                f"unknown kernel(s) for deterministic_forward opset: {', '.join(sorted(set(unknown)))}"
            )
        return "coverage", kernels

    coverage = set(
        _suite_kernel_names(
            suite="coverage",
            flaggems_opset=str(flaggems_opset),
            backend_target=str(backend_target),
        )
    )
    unknown = [k for k in kernels if k not in coverage]
    if unknown:
        raise SystemExit(f"unknown kernel(s) for deterministic_forward opset: {', '.join(sorted(set(unknown)))}")
    return "coverage", kernels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; optional kernel filter")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument(
        "--flaggems-path",
        choices=["original", "intentir"],
        default="intentir",
        help="Execution path for FlagGems pipeline stage.",
    )
    ap.add_argument(
        "--intentir-mode",
        choices=["auto", "force_compile", "force_cache"],
        default="auto",
        help="IntentIR mode (only valid when --flaggems-path=intentir).",
    )
    ap.add_argument(
        "--intentir-miss-policy",
        choices=["deterministic", "strict"],
        default="deterministic",
        help="IntentIR miss policy passed to FlagGems full pipeline script.",
    )
    ap.add_argument(
        "--fallback-policy",
        choices=["deterministic", "strict"],
        default=None,
        help="Deprecated alias for --intentir-miss-policy.",
    )
    ap.add_argument("--seed-cache-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_seed_cache"))
    ap.add_argument("--pipeline-out-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_triton_full_pipeline"))
    ap.add_argument("--active-batch", type=Path, default=None)
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    ap.add_argument(
        "--lane",
        choices=["coverage", "ir_arch", "backend_compiler"],
        default="coverage",
        help="Workflow lane used to resolve active semantic scope (default: coverage).",
    )
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-rvv", action="store_true")
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument(
        "--run-rvv-remote",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run RVV remote suite via SSH after local RVV stage.",
    )
    ap.add_argument("--rvv-host", default=os.getenv("INTENTIR_RVV_HOST", "192.168.8.72"))
    ap.add_argument("--rvv-user", default=os.getenv("INTENTIR_RVV_USER", "ubuntu"))
    ap.add_argument("--rvv-port", type=int, default=22)
    ap.add_argument(
        "--rvv-use-key",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SSH key auth for rvv_remote_suite (default true).",
    )
    ap.add_argument(
        "--allow-cuda-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow CUDA stage to exit 0 with skipped status when CUDA env is unavailable.",
    )
    ap.add_argument(
        "--cuda-timeout-sec",
        type=int,
        default=120,
        help="Compatibility timeout passed to cuda_backend_smoke.py.",
    )
    ap.add_argument(
        "--cuda-compile-timeout-sec",
        type=int,
        default=None,
        help="CUDA compile-stage timeout passed to cuda_backend_smoke.py.",
    )
    ap.add_argument(
        "--cuda-launch-timeout-sec",
        type=int,
        default=None,
        help="CUDA launch-stage timeout passed to cuda_backend_smoke.py.",
    )
    ap.add_argument(
        "--cuda-runtime-backend",
        choices=["auto", "nvcc", "nvrtc"],
        default="auto",
        help="Runtime backend selector passed to cuda_backend_smoke.py (default: auto).",
    )
    ap.add_argument(
        "--schedule-profile-tag",
        default="",
        help="Optional profile tag suffix for backend schedule selection.",
    )
    ap.add_argument("--cuda-tile-m", type=int, default=None)
    ap.add_argument("--cuda-tile-n", type=int, default=None)
    ap.add_argument("--cuda-tile-k", type=int, default=None)
    ap.add_argument("--rvv-tile-m", type=int, default=None)
    ap.add_argument("--rvv-tile-n", type=int, default=None)
    ap.add_argument("--rvv-tile-k", type=int, default=None)
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    ap.add_argument("--out-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "daily" / date_tag))
    ap.add_argument("--write-registry", action="store_true")
    args = ap.parse_args()
    miss_policy = str(args.intentir_miss_policy)
    if args.fallback_policy is not None:
        miss_policy = str(args.fallback_policy)
    if str(args.flaggems_path) == "original" and str(args.intentir_mode) != "auto":
        raise SystemExit("--intentir-mode is only valid when --flaggems-path=intentir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_out_dir = Path(args.pipeline_out_dir)
    pipeline_out_dir.mkdir(parents=True, exist_ok=True)
    seed_cache_dir = Path(args.seed_cache_dir)
    seed_cache_dir.mkdir(parents=True, exist_ok=True)

    stage_results: list[dict[str, Any]] = []
    explicit_kernel_filter = [str(k).strip() for k in (args.kernel or []) if str(k).strip()]
    effective_suite, kernel_filter = _resolve_suite_and_kernel_filter(
        requested_suite=str(args.suite),
        requested_kernels=list(explicit_kernel_filter),
        flaggems_opset=str(args.flaggems_opset),
        backend_target=str(args.backend_target),
    )
    scoped_kernels = list(kernel_filter)
    if not scoped_kernels:
        scoped_kernels = _suite_kernel_names(
            suite=str(effective_suite),
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        )
    default_active = ROOT / "workflow" / "flaggems" / "state" / f"active_batch_{args.lane}.json"
    if str(args.lane) == "coverage" and not default_active.is_file():
        default_active = ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"
    active_batch_path = Path(args.active_batch) if args.active_batch is not None else default_active
    scoped_semantic_ops = _load_active_semantic_ops(active_batch_path)
    profile_tag = str(args.schedule_profile_tag or "").strip()
    rvv_env: dict[str, str] = {}
    cuda_env: dict[str, str] = {}
    if profile_tag:
        rvv_env["INTENTIR_RVV_SCHEDULE_PROFILE_TAG"] = profile_tag
        cuda_env["INTENTIR_CUDA_SCHEDULE_PROFILE_TAG"] = profile_tag
        rvv_env["INTENTIR_SCHEDULE_PROFILE_TAG"] = profile_tag
        cuda_env["INTENTIR_SCHEDULE_PROFILE_TAG"] = profile_tag
    if args.rvv_tile_m is not None:
        rvv_env["INTENTIR_RVV_TILE_M"] = str(int(args.rvv_tile_m))
    if args.rvv_tile_n is not None:
        rvv_env["INTENTIR_RVV_TILE_N"] = str(int(args.rvv_tile_n))
    if args.rvv_tile_k is not None:
        rvv_env["INTENTIR_RVV_TILE_K"] = str(int(args.rvv_tile_k))
    if args.cuda_tile_m is not None:
        cuda_env["INTENTIR_CUDA_TILE_M"] = str(int(args.cuda_tile_m))
    if args.cuda_tile_n is not None:
        cuda_env["INTENTIR_CUDA_TILE_N"] = str(int(args.cuda_tile_n))
    if args.cuda_tile_k is not None:
        cuda_env["INTENTIR_CUDA_TILE_K"] = str(int(args.cuda_tile_k))

    def _record(stage: str, rc: int, stdout: str, stderr: str, extra: dict | None = None) -> None:
        row = {
            "stage": str(stage),
            "rc": int(rc),
            "ok": int(rc) == 0,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }
        if extra:
            row.update(extra)
        stage_results.append(row)

    if not bool(args.skip_pipeline):
        cmd = [
            sys.executable,
            "scripts/triton/flaggems_full_pipeline_verify.py",
            "--suite",
            str(effective_suite),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            str(args.backend_target),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            miss_policy,
            "--strict-kernel-failure",
            "--seed-cache-dir",
            str(seed_cache_dir),
            "--out-dir",
            str(pipeline_out_dir),
        ]
        for k in kernel_filter:
            cmd += ["--kernel", str(k)]
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("pipeline", rc, out, err, extra={"cmd": cmd})

    missing_provider_reports = _collect_missing_provider_reports(pipeline_out_dir, scoped_kernels)
    if missing_provider_reports:
        _record(
            "provider_report_precheck",
            1,
            "",
            "missing provider report(s) for requested kernels",
            extra={
                "reason_code": "pipeline_missing_report",
                "missing_provider_reports": list(missing_provider_reports),
                "provider_report_dir": str(pipeline_out_dir),
            },
        )
    else:
        _record(
            "provider_report_precheck",
            0,
            f"provider reports complete for {len(scoped_kernels)} kernel(s)",
            "",
            extra={"provider_report_dir": str(pipeline_out_dir)},
        )

    rvv_json = out_dir / "rvv_local.json"
    if not bool(args.skip_rvv) and not missing_provider_reports:
        cmd = [
            sys.executable,
            "scripts/backend_codegen_smoke.py",
            "--frontend",
            "triton",
            "--triton-provider",
            "flaggems",
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            "rvv",
            "--artifact-dir",
            str(pipeline_out_dir),
            "--json",
            "--out",
            str(rvv_json),
        ]
        for k in kernel_filter:
            cmd += ["--kernel", str(k)]
        cmd_run = _with_env_prefix(cmd, rvv_env)
        rc, out, err = _run(cmd_run, cwd=ROOT)
        _record(
            "rvv_local",
            rc,
            out,
            err,
            extra={"cmd": cmd_run, "json_path": str(rvv_json), "env_overrides": dict(rvv_env)},
        )

    rvv_remote_json = out_dir / "rvv_remote.json"
    if bool(args.run_rvv_remote) and not bool(args.skip_rvv) and not missing_provider_reports:
        cmd = [
            sys.executable,
            "scripts/rvv_remote_suite.py",
            "--frontend",
            "triton",
            "--suite",
            str(effective_suite),
            "--triton-provider",
            "flaggems",
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            "rvv",
            "--artifact-dir",
            str(pipeline_out_dir),
            "--host",
            str(args.rvv_host),
            "--user",
            str(args.rvv_user),
            "--port",
            str(int(args.rvv_port)),
            "--out",
            str(rvv_remote_json),
        ]
        if bool(args.rvv_use_key):
            cmd.append("--use-key")
        for k in kernel_filter:
            cmd += ["--kernel", str(k)]
        cmd_run = _with_env_prefix(cmd, rvv_env)
        rc, out, err = _run(cmd_run, cwd=ROOT)
        _record(
            "rvv_remote",
            rc,
            out,
            err,
            extra={"cmd": cmd_run, "json_path": str(rvv_remote_json), "env_overrides": dict(rvv_env)},
        )

    cuda_json = out_dir / "cuda_local.json"
    if not bool(args.skip_cuda) and not missing_provider_reports:
        cuda_compile_timeout_sec = (
            int(args.cuda_compile_timeout_sec)
            if args.cuda_compile_timeout_sec is not None
            else int(args.cuda_timeout_sec)
        )
        cuda_launch_timeout_sec = (
            int(args.cuda_launch_timeout_sec)
            if args.cuda_launch_timeout_sec is not None
            else int(args.cuda_timeout_sec)
        )
        cmd = [
            sys.executable,
            "scripts/cuda_backend_smoke.py",
            "--frontend",
            "triton",
            "--triton-provider",
            "flaggems",
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            "cuda_h100",
            "--artifact-dir",
            str(pipeline_out_dir),
            "--timeout-sec",
            str(int(args.cuda_timeout_sec)),
            "--compile-timeout-sec",
            str(int(cuda_compile_timeout_sec)),
            "--launch-timeout-sec",
            str(int(cuda_launch_timeout_sec)),
            "--runtime-backend",
            str(args.cuda_runtime_backend),
            "--json",
            "--out",
            str(cuda_json),
        ]
        if bool(args.allow_cuda_skip):
            cmd.append("--allow-skip")
        for k in kernel_filter:
            cmd += ["--kernel", str(k)]
        cmd_run = _with_env_prefix(cmd, cuda_env)
        rc, out, err = _run(cmd_run, cwd=ROOT)
        _record(
            "cuda_local",
            rc,
            out,
            err,
            extra={"cmd": cmd_run, "json_path": str(cuda_json), "env_overrides": dict(cuda_env)},
        )

    stage_timing_breakdown = out_dir / "stage_timing_breakdown.json"
    if rvv_json.is_file() and cuda_json.is_file():
        cmd = [
            sys.executable,
            "scripts/flaggems/compute_stage_timing_breakdown.py",
            "--rvv-json",
            str(rvv_json),
            "--cuda-json",
            str(cuda_json),
            "--out",
            str(stage_timing_breakdown),
        ]
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("stage_timing_breakdown", rc, out, err, extra={"cmd": cmd, "json_path": str(stage_timing_breakdown)})
    else:
        _record(
            "stage_timing_breakdown",
            0,
            "stage timing breakdown skipped (rvv/cuda json not both present)",
            "",
            extra={
                "reason_code": "skipped_missing_backend_json",
                "json_path": str(stage_timing_breakdown),
            },
        )

    converged = out_dir / "status_converged.json"
    cmd = [
        sys.executable,
        "scripts/flaggems/converge_status.py",
        "--provider-report-dir",
        str(pipeline_out_dir),
        "--out",
        str(converged),
    ]
    if rvv_remote_json.is_file():
        cmd += ["--rvv-json", str(rvv_remote_json)]
    elif rvv_json.is_file():
        cmd += ["--rvv-json", str(rvv_json)]
    if cuda_json.is_file():
        cmd += ["--cuda-json", str(cuda_json)]
    for k in scoped_kernels:
        cmd += ["--scope-kernels", str(k)]
    for sop in scoped_semantic_ops:
        cmd += ["--scope-semantic-ops", str(sop)]
    cmd += ["--scope-mode", "active_only"]
    if bool(args.write_registry):
        cmd.append("--write-registry")
    rc, out, err = _run(cmd, cwd=ROOT)
    _record("converge", rc, out, err, extra={"cmd": cmd, "json_path": str(converged)})

    coverage_integrity = out_dir / "coverage_integrity.json"
    full_coverage_run = (str(effective_suite) == "coverage") and (len(explicit_kernel_filter) == 0)
    if converged.is_file() and full_coverage_run:
        cmd = [
            sys.executable,
            "scripts/flaggems/recompute_coverage_integrity.py",
            "--registry",
            str(ROOT / "pipeline" / "triton" / "flaggems_registry.json"),
            "--run-summary",
            str(out_dir / "run_summary.json"),
            "--status-converged",
            str(converged),
            "--out",
            str(coverage_integrity),
        ]
        # run_summary.json is written after this stage block; emit a temporary run summary
        # with stages seen so far to satisfy recompute input contract.
        tmp_summary = {
            "ok": all(bool(r.get("ok")) for r in stage_results),
            "stages": stage_results,
        }
        (out_dir / "run_summary.json").write_text(json.dumps(tmp_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("coverage_integrity", rc, out, err, extra={"cmd": cmd, "json_path": str(coverage_integrity)})
    else:
        _record(
            "coverage_integrity",
            0,
            "coverage integrity skipped for partial scope run",
            "",
            extra={
                "reason_code": "skipped_partial_scope",
                "full_coverage_run": bool(full_coverage_run),
                "json_path": str(coverage_integrity),
            },
        )

    ok = all(bool(r.get("ok")) for r in stage_results)
    summary = {
        "ok": bool(ok),
        "lane": str(args.lane),
        "requested_suite": str(args.suite),
        "suite": str(effective_suite),
        "kernel_filter": list(kernel_filter),
        "scope_kernels": list(scoped_kernels),
        "missing_provider_reports": list(missing_provider_reports),
        "flaggems_path": str(args.flaggems_path),
        "intentir_mode": str(args.intentir_mode),
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "cuda_timeout_sec": int(args.cuda_timeout_sec),
        "cuda_compile_timeout_sec": (
            int(args.cuda_compile_timeout_sec)
            if args.cuda_compile_timeout_sec is not None
            else int(args.cuda_timeout_sec)
        ),
        "cuda_launch_timeout_sec": (
            int(args.cuda_launch_timeout_sec)
            if args.cuda_launch_timeout_sec is not None
            else int(args.cuda_timeout_sec)
        ),
        "cuda_runtime_backend": str(args.cuda_runtime_backend),
        "intentir_miss_policy": miss_policy,
        "fallback_policy": miss_policy,
        "schedule_profile_tag": profile_tag,
        "rvv_schedule_overrides": dict(rvv_env),
        "cuda_schedule_overrides": dict(cuda_env),
        "seed_cache_dir": str(seed_cache_dir),
        "pipeline_out_dir": str(pipeline_out_dir),
        "active_batch_path": str(active_batch_path),
        "stages": stage_results,
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Matrix summary written: {summary_path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
