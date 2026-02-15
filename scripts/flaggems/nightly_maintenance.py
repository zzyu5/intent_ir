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
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--flaggems-path", choices=["original", "intentir"], default="intentir")
    ap.add_argument("--intentir-mode", choices=["auto", "force_compile", "force_cache"], default="auto")
    ap.add_argument("--fallback-policy", choices=["deterministic", "strict"], default="deterministic")
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
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
    ap.add_argument("--cuda-codegen-mode", choices=["auto", "cpp", "py"], default="py")
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    ap.add_argument(
        "--write-registry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --write-registry to matrix run (default: false).",
    )
    ap.add_argument("--out-root", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "daily"))
    ap.add_argument("--date-tag", default=_utc_date_tag())
    ap.add_argument("--run-name", default="nightly_maintenance")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if str(args.flaggems_path) == "original" and str(args.intentir_mode) != "auto":
        raise SystemExit("--intentir-mode is only valid when --flaggems-path=intentir")

    out_dir = Path(args.out_root) / str(args.date_tag) / str(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_cmd: list[str] = [
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
        "--fallback-policy",
        str(args.fallback_policy),
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
        "--cuda-codegen-mode",
        str(args.cuda_codegen_mode),
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

    matrix_rc, matrix_out, matrix_err = _run(matrix_cmd, cwd=ROOT, dry_run=bool(args.dry_run))
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
        "--out",
        str(ci_gate_path),
    ]

    ci_rc = 0
    ci_out = ""
    ci_err = ""
    ci_skipped = False
    if matrix_rc == 0:
        ci_rc, ci_out, ci_err = _run(ci_cmd, cwd=ROOT, dry_run=bool(args.dry_run))
    else:
        ci_skipped = True

    payload: dict[str, Any] = {
        "ok": bool(matrix_rc == 0 and (ci_rc == 0 or ci_skipped)),
        "mode": "dry-run" if bool(args.dry_run) else "execute",
        "out_dir": str(out_dir),
        "artifacts": {
            "run_summary": str(run_summary_path),
            "status_converged": str(status_converged_path),
            "ci_gate": str(ci_gate_path),
        },
        "commands": {
            "matrix": matrix_cmd,
            "ci_gate": ci_cmd,
        },
        "results": {
            "matrix": {
                "rc": int(matrix_rc),
                "stdout": str(matrix_out).strip(),
                "stderr": str(matrix_err).strip(),
            },
            "ci_gate": {
                "skipped": bool(ci_skipped),
                "rc": int(ci_rc),
                "stdout": str(ci_out).strip(),
                "stderr": str(ci_err).strip(),
            },
        },
    }
    summary_path = out_dir / "nightly_maintenance_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Nightly maintenance summary written: {summary_path}")
    raise SystemExit(0 if bool(payload.get("ok")) else 1)


if __name__ == "__main__":
    main()
