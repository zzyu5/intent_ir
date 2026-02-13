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
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")


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
    ap.add_argument("--seed-cache-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_seed_cache"))
    ap.add_argument("--pipeline-out-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_triton_full_pipeline"))
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
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
        help="Per-kernel timeout passed to cuda_backend_smoke.py.",
    )
    ap.add_argument("--out-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix"))
    ap.add_argument("--write-registry", action="store_true")
    args = ap.parse_args()
    if str(args.flaggems_path) == "original" and str(args.intentir_mode) != "auto":
        raise SystemExit("--intentir-mode is only valid when --flaggems-path=intentir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_out_dir = Path(args.pipeline_out_dir)
    pipeline_out_dir.mkdir(parents=True, exist_ok=True)
    seed_cache_dir = Path(args.seed_cache_dir)
    seed_cache_dir.mkdir(parents=True, exist_ok=True)

    stage_results: list[dict[str, Any]] = []
    kernel_filter = [str(k) for k in (args.kernel or []) if str(k).strip()]

    # For coverage/all, backend smoke should use the same e2e kernel set as
    # provider pipeline, not the small default smoke list.
    if not kernel_filter and str(args.suite) in {"coverage", "all"}:
        from pipeline.triton.flaggems_specs import coverage_flaggems_kernel_specs  # noqa: PLC0415

        kernel_filter = [
            str(s.name)
            for s in coverage_flaggems_kernel_specs(
                flaggems_opset=str(args.flaggems_opset),
                backend_target=str(args.backend_target),
            )
        ]

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
            str(args.suite),
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
            "--seed-cache-dir",
            str(seed_cache_dir),
            "--out-dir",
            str(pipeline_out_dir),
        ]
        for k in kernel_filter:
            cmd += ["--kernel", str(k)]
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("pipeline", rc, out, err, extra={"cmd": cmd})

    rvv_json = out_dir / "rvv_local.json"
    if not bool(args.skip_rvv):
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
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("rvv_local", rc, out, err, extra={"cmd": cmd, "json_path": str(rvv_json)})

    rvv_remote_json = out_dir / "rvv_remote.json"
    if bool(args.run_rvv_remote) and not bool(args.skip_rvv):
        cmd = [
            sys.executable,
            "scripts/rvv_remote_suite.py",
            "--frontend",
            "triton",
            "--suite",
            str(args.suite),
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
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("rvv_remote", rc, out, err, extra={"cmd": cmd, "json_path": str(rvv_remote_json)})

    cuda_json = out_dir / "cuda_local.json"
    if not bool(args.skip_cuda):
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
            "--json",
            "--out",
            str(cuda_json),
        ]
        if bool(args.allow_cuda_skip):
            cmd.append("--allow-skip")
        for k in kernel_filter:
            cmd += ["--kernel", str(k)]
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("cuda_local", rc, out, err, extra={"cmd": cmd, "json_path": str(cuda_json)})

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
    if bool(args.write_registry):
        cmd.append("--write-registry")
    rc, out, err = _run(cmd, cwd=ROOT)
    _record("converge", rc, out, err, extra={"cmd": cmd, "json_path": str(converged)})

    ok = all(bool(r.get("ok")) for r in stage_results)
    summary = {
        "ok": bool(ok),
        "suite": str(args.suite),
        "kernel_filter": list(kernel_filter),
        "flaggems_path": str(args.flaggems_path),
        "intentir_mode": str(args.intentir_mode),
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "seed_cache_dir": str(seed_cache_dir),
        "pipeline_out_dir": str(pipeline_out_dir),
        "stages": stage_results,
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Matrix summary written: {summary_path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
