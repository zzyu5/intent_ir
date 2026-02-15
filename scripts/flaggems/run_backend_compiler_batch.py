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


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _guess_baseline_dir(out_dir: Path) -> Path | None:
    current_status = ROOT / "workflow" / "flaggems" / "state" / "current_status.json"
    if not current_status.is_file():
        return None
    payload = _load_json(current_status)
    run_summary = str((payload.get("latest_artifacts") or {}).get("run_summary") or "").strip()
    if not run_summary:
        return None
    p = Path(run_summary)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        return None
    baseline_dir = p.parent
    if baseline_dir.resolve() == out_dir.resolve():
        return None
    return baseline_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke")
    ap.add_argument("--kernel", action="append", default=["add2d", "mul2d"])
    ap.add_argument("--run-rvv-remote", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rvv-host", default="192.168.8.72")
    ap.add_argument("--rvv-user", default="ubuntu")
    ap.add_argument("--rvv-use-key", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cuda-runtime-backend", choices=["auto", "nvcc", "nvrtc"], default="auto")
    ap.add_argument("--allow-cuda-skip", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--emit-timing-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit stage timing delta report against previous backend run.",
    )
    ap.add_argument("--timing-baseline-dir", type=Path, default=None)
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
    timing_delta_path = ""
    timing_delta_ok = True
    timing_delta_stderr = ""
    schedule_profiles_path = ""
    schedule_profiles_ok = True
    schedule_profiles_stderr = ""
    baseline_dir = Path(args.timing_baseline_dir) if args.timing_baseline_dir is not None else _guess_baseline_dir(Path(args.out_dir))
    if bool(args.emit_timing_delta) and not bool(args.dry_run) and int(rc) == 0:
        current_rvv = Path(args.out_dir) / "rvv_local.json"
        current_cuda = Path(args.out_dir) / "cuda_local.json"
        baseline_rvv = (baseline_dir / "rvv_local.json") if baseline_dir is not None else None
        baseline_cuda = (baseline_dir / "cuda_local.json") if baseline_dir is not None else None
        timing_delta = Path(args.out_dir) / "timing_delta.json"
        delta_cmd = [
            sys.executable,
            "scripts/flaggems/compute_stage_timing_delta.py",
            "--current-rvv",
            str(current_rvv),
            "--current-cuda",
            str(current_cuda),
            "--out",
            str(timing_delta),
        ]
        if baseline_rvv is not None:
            delta_cmd += ["--baseline-rvv", str(baseline_rvv)]
        if baseline_cuda is not None:
            delta_cmd += ["--baseline-cuda", str(baseline_cuda)]
        rc_delta, _out_delta, err_delta = _run(delta_cmd, dry_run=False)
        timing_delta_ok = int(rc_delta) == 0
        timing_delta_stderr = str(err_delta).strip()
        timing_delta_path = str(timing_delta)
        run_summary_path = Path(args.out_dir) / "run_summary.json"
        if run_summary_path.is_file():
            run_summary = _load_json(run_summary_path)
            stages = [s for s in list(run_summary.get("stages") or []) if isinstance(s, dict)]
            stages = [s for s in stages if str(s.get("stage") or "") != "timing_delta"]
            stages.append(
                {
                    "stage": "timing_delta",
                    "rc": 0 if bool(timing_delta_ok) else 1,
                    "ok": bool(timing_delta_ok),
                    "stdout": f"timing delta generated at {timing_delta_path}" if timing_delta_path else "",
                    "stderr": str(timing_delta_stderr),
                    "cmd": delta_cmd,
                    "json_path": timing_delta_path,
                    "baseline_dir": str(baseline_dir) if baseline_dir is not None else "",
                }
            )
            run_summary["stages"] = stages
            run_summary["ok"] = bool(run_summary.get("ok")) and bool(timing_delta_ok)
            run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if not bool(args.dry_run) and int(rc) == 0:
        schedule_profiles = Path(args.out_dir) / "schedule_profiles.json"
        profiles_cmd = [
            sys.executable,
            "scripts/flaggems/export_schedule_profiles.py",
            "--out",
            str(schedule_profiles),
        ]
        rc_profiles, _out_profiles, err_profiles = _run(profiles_cmd, dry_run=False)
        schedule_profiles_ok = int(rc_profiles) == 0
        schedule_profiles_stderr = str(err_profiles).strip()
        schedule_profiles_path = str(schedule_profiles)
        run_summary_path = Path(args.out_dir) / "run_summary.json"
        if run_summary_path.is_file():
            run_summary = _load_json(run_summary_path)
            stages = [s for s in list(run_summary.get("stages") or []) if isinstance(s, dict)]
            stages = [s for s in stages if str(s.get("stage") or "") != "schedule_profiles"]
            stages.append(
                {
                    "stage": "schedule_profiles",
                    "rc": 0 if bool(schedule_profiles_ok) else 1,
                    "ok": bool(schedule_profiles_ok),
                    "stdout": f"schedule profiles generated at {schedule_profiles_path}" if schedule_profiles_path else "",
                    "stderr": str(schedule_profiles_stderr),
                    "cmd": profiles_cmd,
                    "json_path": schedule_profiles_path,
                }
            )
            run_summary["stages"] = stages
            run_summary["ok"] = bool(run_summary.get("ok")) and bool(schedule_profiles_ok)
            run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "ok": bool(rc == 0 and timing_delta_ok and schedule_profiles_ok),
        "lane": "backend_compiler",
        "cmd": cmd,
        "out_dir": str(args.out_dir),
        "cuda_runtime_backend": str(args.cuda_runtime_backend),
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "timing_delta": {
            "ok": bool(timing_delta_ok),
            "path": timing_delta_path,
            "baseline_dir": str(baseline_dir) if baseline_dir is not None else "",
            "stderr": timing_delta_stderr,
        },
        "schedule_profiles": {
            "ok": bool(schedule_profiles_ok),
            "path": schedule_profiles_path,
            "stderr": schedule_profiles_stderr,
        },
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch_summary = out / "backend_compiler_batch_summary.json"
    batch_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"backend_compiler batch summary written: {batch_summary}")
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
