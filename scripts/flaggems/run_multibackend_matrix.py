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
    ap.add_argument("--use-llm", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--flaggems-opset", choices=["deterministic_forward"], default="deterministic_forward")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="rvv")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-rvv", action="store_true")
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--allow-cuda-skip", action="store_true", default=True)
    ap.add_argument("--out-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix"))
    ap.add_argument("--write-registry", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_results: list[dict[str, Any]] = []

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
        if bool(args.use_llm):
            cmd.append("--use-llm")
        else:
            cmd.append("--no-use-llm")
        for k in (args.kernel or []):
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
        for k in (args.kernel or []):
            cmd += ["--kernel", str(k)]
        rc, out, err = _run(cmd, cwd=ROOT)
        _record("rvv_local", rc, out, err, extra={"cmd": cmd, "json_path": str(rvv_json)})

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
            "--json",
            "--out",
            str(cuda_json),
        ]
        if bool(args.allow_cuda_skip):
            cmd.append("--allow-skip")
        for k in (args.kernel or []):
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
    if rvv_json.is_file():
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
        "kernel_filter": list(args.kernel or []),
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
