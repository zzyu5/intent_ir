"""
Nightly maintenance runner for converged FlagGems integration.

This script orchestrates:
1) run_multibackend_matrix.py
2) ci_gate.py

It is intended for post-coverage drift detection after reaching full dual-pass.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_date_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> tuple[int, str, str]:
    if dry_run:
        return 0, "(dry-run)", ""
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="coverage")
    ap.add_argument(
        "--coverage-mode",
        choices=["single_run", "category_batches"],
        default="category_batches",
        help="Coverage orchestrator mode when --suite=coverage (default: category_batches).",
    )
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--flaggems-path", choices=["original", "intentir"], default="intentir")
    ap.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="auto")
    ap.add_argument("--intentir-miss-policy", choices=["deterministic", "strict"], default="deterministic")
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    ap.add_argument(
        "--lane",
        choices=["coverage", "ir_arch", "backend_compiler"],
        default="coverage",
        help="Workflow lane for matrix scope (default: coverage).",
    )
    ap.add_argument(
        "--ci-profiles",
        action="append",
        default=[],
        help="CI gate profiles (repeatable/comma-separated). Default: coverage",
    )
    ap.add_argument(
        "--run-rvv-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run RVV remote stage (default: true).",
    )
    ap.add_argument("--rvv-host", default="192.168.8.72")
    ap.add_argument("--rvv-user", default="ubuntu")
    ap.add_argument("--rvv-port", type=int, default=22)
    ap.add_argument(
        "--rvv-use-key",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SSH key for RVV remote stage (default: true).",
    )
    ap.add_argument(
        "--allow-cuda-skip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow CUDA stage to skip when CUDA env is unavailable (default: true).",
    )
    ap.add_argument("--cuda-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-compile-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-launch-timeout-sec", type=int, default=120)
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    ap.add_argument(
        "--max-total-regression-pct",
        type=float,
        default=8.0,
        help="Backend compiler perf gate threshold passed to ci_gate/check_batch_gate.",
    )
    ap.add_argument(
        "--min-regression-delta-ms",
        type=float,
        default=50.0,
        help="Backend compiler minimum delta-ms threshold passed to ci_gate/check_batch_gate.",
    )
    ap.add_argument(
        "--max-regression-ratio",
        type=float,
        default=0.5,
        help="Backend compiler regression ratio threshold passed to ci_gate/check_batch_gate.",
    )
    ap.add_argument(
        "--write-registry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --write-registry to matrix run (default: false).",
    )
    ap.add_argument("--out-root", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "daily"))
    ap.add_argument("--date-tag", default=_utc_date_tag())
    ap.add_argument("--run-name", default="nightly_maintenance")
    ap.add_argument(
        "--recompute-coverage-integrity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run coverage integrity recompute stage after matrix output is available.",
    )
    ap.add_argument(
        "--update-workflow-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update current_status/session_context snapshots after nightly run.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    miss_policy = str(args.intentir_miss_policy)

    if str(args.flaggems_path) == "original" and str(args.intentir_mode) != "auto":
        raise SystemExit("--intentir-mode is only valid when --flaggems-path=intentir")

    out_dir = Path(args.out_root) / str(args.date_tag) / str(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_report = out_dir / "catalog_validation.json"
    catalog_cmd = [
        sys.executable,
        "scripts/validate_catalog.py",
        "--catalog",
        str(ROOT / "scripts" / "CATALOG.json"),
        "--out",
        str(catalog_report),
    ]
    catalog_rc, catalog_out, catalog_err = _run(catalog_cmd, cwd=ROOT, dry_run=bool(args.dry_run))

    use_category_batches = bool(
        str(args.suite) == "coverage"
        and str(args.lane) == "coverage"
        and str(args.coverage_mode) == "category_batches"
    )

    if use_category_batches:
        build_batches_cmd: list[str] = [
            sys.executable,
            "scripts/flaggems/build_coverage_batches.py",
        ]
        if catalog_rc == 0:
            build_batches_rc, build_batches_out, build_batches_err = _run(
                build_batches_cmd, cwd=ROOT, dry_run=bool(args.dry_run)
            )
        else:
            build_batches_rc, build_batches_out, build_batches_err = (1, "", "catalog validation failed")

        matrix_cmd: list[str] = [
            sys.executable,
            "scripts/flaggems/run_coverage_batches.py",
            "--out-root",
            str(out_dir),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            miss_policy,
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            str(args.backend_target),
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--cuda-timeout-sec",
            str(int(args.cuda_timeout_sec)),
            "--cuda-compile-timeout-sec",
            str(int(args.cuda_compile_timeout_sec)),
            "--cuda-launch-timeout-sec",
            str(int(args.cuda_launch_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
        ]
        matrix_cmd.append("--run-rvv-remote" if bool(args.run_rvv_remote) else "--no-run-rvv-remote")
        matrix_cmd.append("--rvv-use-key" if bool(args.rvv_use_key) else "--no-rvv-use-key")
        matrix_cmd.append("--allow-cuda-skip" if bool(args.allow_cuda_skip) else "--no-allow-cuda-skip")
        if bool(args.write_registry):
            matrix_cmd.append("--write-registry")
        matrix_cmd.append("--aggregate")
        matrix_cmd.append("--stream-subprocess-output")
    else:
        build_batches_cmd = []
        build_batches_rc, build_batches_out, build_batches_err = (0, "", "")
        matrix_cmd = [
            sys.executable,
            "scripts/flaggems/run_multibackend_matrix.py",
            "--suite",
            str(args.suite),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-path",
            str(args.flaggems_path),
            "--intentir-mode",
            str(args.intentir_mode),
            "--intentir-miss-policy",
            miss_policy,
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            str(args.backend_target),
            "--lane",
            str(args.lane),
            "--rvv-host",
            str(args.rvv_host),
            "--rvv-user",
            str(args.rvv_user),
            "--rvv-port",
            str(int(args.rvv_port)),
            "--cuda-timeout-sec",
            str(int(args.cuda_timeout_sec)),
            "--cuda-compile-timeout-sec",
            str(int(args.cuda_compile_timeout_sec)),
            "--cuda-launch-timeout-sec",
            str(int(args.cuda_launch_timeout_sec)),
            "--cuda-runtime-backend",
            str(args.cuda_runtime_backend),
            "--out-dir",
            str(out_dir),
        ]
        matrix_cmd.append("--run-rvv-remote" if bool(args.run_rvv_remote) else "--no-run-rvv-remote")
        matrix_cmd.append("--rvv-use-key" if bool(args.rvv_use_key) else "--no-rvv-use-key")
        matrix_cmd.append("--allow-cuda-skip" if bool(args.allow_cuda_skip) else "--no-allow-cuda-skip")
        if bool(args.write_registry):
            matrix_cmd.append("--write-registry")

    if catalog_rc == 0 and build_batches_rc == 0:
        matrix_rc, matrix_out, matrix_err = _run(matrix_cmd, cwd=ROOT, dry_run=bool(args.dry_run))
    else:
        fail_reason = "catalog validation failed"
        if build_batches_rc != 0:
            fail_reason = "coverage batch build failed"
        matrix_rc, matrix_out, matrix_err = (1, "", fail_reason)
    run_summary_path = out_dir / "run_summary.json"
    status_converged_path = out_dir / "status_converged.json"
    ci_gate_path = out_dir / "ci_gate.json"
    ci_cmd = [
        sys.executable,
        "scripts/flaggems/ci_gate.py",
        "--run-summary",
        str(run_summary_path),
        "--status-converged",
        str(status_converged_path),
        "--max-total-regression-pct",
        str(float(args.max_total_regression_pct)),
        "--min-regression-delta-ms",
        str(float(args.min_regression_delta_ms)),
        "--max-regression-ratio",
        str(float(args.max_regression_ratio)),
        "--out",
        str(ci_gate_path),
    ]
    profiles_raw = list(args.ci_profiles or [])
    if not profiles_raw:
        profiles_raw = ["coverage"]
    for raw in profiles_raw:
        ci_cmd += ["--profiles", str(raw)]

    ci_rc = 0
    ci_out = ""
    ci_err = ""
    ci_skipped = False
    recompute_rc = 0
    recompute_out = ""
    recompute_err = ""
    recompute_skipped = False
    if matrix_rc == 0:
        ci_rc, ci_out, ci_err = _run(ci_cmd, cwd=ROOT, dry_run=bool(args.dry_run))
        if bool(args.recompute_coverage_integrity):
            recompute_cmd = [
                sys.executable,
                "scripts/flaggems/recompute_coverage_integrity.py",
                "--registry",
                str(ROOT / "pipeline" / "triton" / "flaggems_registry.json"),
                "--run-summary",
                str(run_summary_path),
                "--status-converged",
                str(status_converged_path),
                "--out",
                str(out_dir / "coverage_integrity.json"),
            ]
            recompute_rc, recompute_out, recompute_err = _run(recompute_cmd, cwd=ROOT, dry_run=bool(args.dry_run))
            payload_cmds_recompute = recompute_cmd
        else:
            recompute_skipped = True
            payload_cmds_recompute = []
    else:
        ci_skipped = True
        recompute_skipped = True
        payload_cmds_recompute = []

    payload: dict[str, Any] = {
        "ok": bool(matrix_rc == 0 and (ci_rc == 0 or ci_skipped) and (recompute_rc == 0 or recompute_skipped)),
        "mode": "dry-run" if bool(args.dry_run) else "execute",
        "out_dir": str(out_dir),
        "artifacts": {
            "catalog_validation": str(catalog_report),
            "run_summary": str(run_summary_path),
            "status_converged": str(status_converged_path),
            "ci_gate": str(ci_gate_path),
            "coverage_integrity": str(out_dir / "coverage_integrity.json"),
        },
        "commands": {
            "catalog_validate": catalog_cmd,
            "matrix": matrix_cmd,
            "ci_gate": ci_cmd,
            "coverage_integrity": payload_cmds_recompute,
            "build_coverage_batches": build_batches_cmd,
        },
        "results": {
            "catalog_validate": {
                "rc": int(catalog_rc),
                "stdout": str(catalog_out).strip(),
                "stderr": str(catalog_err).strip(),
            },
            "build_coverage_batches": {
                "rc": int(build_batches_rc),
                "stdout": str(build_batches_out).strip(),
                "stderr": str(build_batches_err).strip(),
            },
            "matrix": {
                "rc": int(matrix_rc),
                "stdout": str(matrix_out).strip(),
                "stderr": str(matrix_err).strip(),
                "mode": ("category_batches" if use_category_batches else "single_run"),
            },
            "ci_gate": {
                "skipped": bool(ci_skipped),
                "rc": int(ci_rc),
                "stdout": str(ci_out).strip(),
                "stderr": str(ci_err).strip(),
            },
            "coverage_integrity": {
                "skipped": bool(recompute_skipped),
                "rc": int(recompute_rc),
                "stdout": str(recompute_out).strip(),
                "stderr": str(recompute_err).strip(),
            },
        },
    }
    summary_path = out_dir / "nightly_maintenance_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    workflow_state_cmd = [
        sys.executable,
        "scripts/flaggems/build_workflow_state.py",
    ]
    workflow_state_rc = 0
    workflow_state_out = ""
    workflow_state_err = ""
    if bool(args.update_workflow_state):
        workflow_state_rc, workflow_state_out, workflow_state_err = _run(workflow_state_cmd, cwd=ROOT, dry_run=bool(args.dry_run))
        payload["results"]["workflow_state"] = {
            "rc": int(workflow_state_rc),
            "stdout": str(workflow_state_out).strip(),
            "stderr": str(workflow_state_err).strip(),
        }
        payload["commands"]["workflow_state"] = workflow_state_cmd
        payload["ok"] = bool(payload.get("ok")) and bool(workflow_state_rc == 0)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Nightly maintenance summary written: {summary_path}")
    raise SystemExit(0 if bool(payload.get("ok")) else 1)


if __name__ == "__main__":
    main()
