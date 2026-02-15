"""
Run backend_compiler lane batch using matrix smoke path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], *, dry_run: bool) -> tuple[int, str, str]:
    if dry_run:
        return 0, "(dry-run)", ""
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")


def _default_out_dir() -> Path:
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_name = f"backend_compiler_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    return ROOT / "artifacts" / "flaggems_matrix" / "daily" / date_tag / run_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke")
    ap.add_argument("--kernel", action="append", default=["add2d", "mul2d"])
    ap.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rvv-host", default="192.168.8.72")
    ap.add_argument("--rvv-user", default="ubuntu")
    ap.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="nvrtc")
    ap.add_argument("--cuda-codegen-mode", choices=["auto", "cpp", "py"], default="py")
    ap.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cmd = [
        sys.executable,
        "scripts/flaggems/run_multibackend_matrix.py",
        "--suite",
        str(args.suite),
        "--lane",
        "coverage",
        "--flaggems-path",
        "intentir",
        "--intentir-mode",
        "auto",
        "--out-dir",
        str(args.out_dir),
        "--cuda-runtime-backend",
        str(args.cuda_runtime_backend),
        "--cuda-codegen-mode",
        str(args.cuda_codegen_mode),
        "--rvv-host",
        str(args.rvv_host),
        "--rvv-user",
        str(args.rvv_user),
    ]
    if bool(args.run_rvv_remote):
        cmd.append("--run-rvv-remote")
    else:
        cmd.append("--no-run-rvv-remote")
    if bool(args.rvv_use_key):
        cmd.append("--rvv-use-key")
    else:
        cmd.append("--no-rvv-use-key")
    if bool(args.allow_cuda_skip):
        cmd.append("--allow-cuda-skip")
    else:
        cmd.append("--no-allow-cuda-skip")
    for k in list(args.kernel or []):
        cmd += ["--kernel", str(k)]

    rc, stdout, stderr = _run(cmd, dry_run=bool(args.dry_run))
    summary = {
        "ok": bool(rc == 0),
        "lane": "backend_compiler",
        "cmd": cmd,
        "out_dir": str(args.out_dir),
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch_summary = out / "backend_compiler_batch_summary.json"
    batch_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"backend_compiler batch summary written: {batch_summary}")
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
