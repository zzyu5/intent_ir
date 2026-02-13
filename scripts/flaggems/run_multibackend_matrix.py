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
        "--use-intent-ir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable IntentIR pipeline for Triton stage (default: on).",
    )
    ap.add_argument(
        "--intentir-seed-policy",
        choices=["auto", "force_llm", "force_cache"],
        default="auto",
        help="IntentIR seed policy for Triton stage.",
    )
    legacy_llm = ap.add_mutually_exclusive_group()
    legacy_llm.add_argument(
        "--use-llm",
        dest="legacy_llm_switch",
        action="store_const",
        const="force_llm",
        help="Legacy alias: equivalent to --use-intent-ir --intentir-seed-policy force_llm.",
    )
    legacy_llm.add_argument(
        "--no-use-llm",
        dest="legacy_llm_switch",
        action="store_const",
        const="traditional",
        help="Legacy alias: equivalent to --no-use-intent-ir.",
    )
    ap.set_defaults(legacy_llm_switch=None)
    ap.add_argument(
        "--allow-deterministic-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When intentir-seed-policy is force_cache and seed cache is missing, allow deterministic fallback intents.",
    )
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
    use_intent_ir = bool(args.use_intent_ir)
    seed_policy = str(args.intentir_seed_policy)
    if args.legacy_llm_switch == "force_llm":
        use_intent_ir = True
        seed_policy = "force_llm"
    elif args.legacy_llm_switch == "traditional":
        use_intent_ir = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            "scripts/triton/full_pipeline_verify.py",
            "--provider",
            "flaggems",
            "--suite",
            str(args.suite),
            "--cases-limit",
            str(int(args.cases_limit)),
            "--flaggems-opset",
            str(args.flaggems_opset),
            "--backend-target",
            str(args.backend_target),
        ]
        if bool(use_intent_ir):
            cmd.append("--use-intent-ir")
        else:
            cmd.append("--no-use-intent-ir")
        cmd += ["--intentir-seed-policy", str(seed_policy)]
        if bool(args.allow_deterministic_fallback):
            cmd.append("--allow-deterministic-fallback")
        else:
            cmd.append("--no-allow-deterministic-fallback")
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
        str(ROOT / "artifacts" / "flaggems_triton_full_pipeline"),
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
        "flaggems_opset": str(args.flaggems_opset),
        "backend_target": str(args.backend_target),
        "stages": stage_results,
        "out_dir": str(out_dir),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Matrix summary written: {summary_path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
