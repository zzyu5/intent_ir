#!/usr/bin/env python3
"""
Run a multi-frontend validation matrix and write a single summary JSON.

Frontends covered:
- Triton native coverage (pipeline.triton.core.coverage_kernel_specs)
- Triton FlagGems coverage (pipeline.triton.providers.flaggems.specs.coverage_flaggems_kernel_specs)
- TileLang coverage
- CUDA frontend coverage

This runner is intentionally orchestration-only: it shells out to the existing
per-frontend runners so each frontend keeps ownership of its flags and report
schema.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _utc_date_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, stream: bool) -> tuple[int, float]:
    t0 = time.perf_counter()
    if stream:
        rc = subprocess.run(cmd, cwd=str(cwd)).returncode
    else:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        if p.stdout:
            print(p.stdout, end="", flush=True)
        if p.stderr:
            print(p.stderr, end="", file=sys.stderr, flush=True)
        rc = p.returncode
    dt = float(time.perf_counter() - t0)
    return int(rc), dt


def _next_round_index(base_dir: Path) -> int:
    if not base_dir.is_dir():
        return 1
    pat = re.compile(r"^round(\d+)_")
    seen: list[int] = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        try:
            seen.append(int(m.group(1)))
        except Exception:
            continue
    return (max(seen) + 1) if seen else 1


def _summarize_kernel_reports(out_dir: Path) -> dict[str, Any]:
    # Kernel reports are written as `<kernel>.json` (no extra dot segments).
    reports = sorted([p for p in out_dir.glob("*.json") if p.is_file() and p.name.count(".") == 1])
    total = int(len(reports))
    passed = 0
    failed = 0
    missing_diff = 0
    for p in reports:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            failed += 1
            continue
        diff = payload.get("diff")
        if not isinstance(diff, dict):
            missing_diff += 1
            continue
        ok = bool(diff.get("ok"))
        if ok:
            passed += 1
        else:
            failed += 1
    return {
        "report_dir": str(out_dir),
        "kernel_reports": total,
        "diff_pass": int(passed),
        "diff_fail": int(failed),
        "diff_missing": int(missing_diff),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-tag", default=_utc_date_tag(), help="UTC date tag, e.g. 20260227")
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default="cuda_5090d")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--with-perf", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--perf-out-dir", default="", help="Optional perf out dir (default: <round-dir>/perf)")
    args = ap.parse_args()

    date_tag = str(args.date_tag).strip()
    backend_target = str(args.backend_target).strip()
    base_dir = ROOT / "artifacts" / "validation_rounds" / date_tag
    base_dir.mkdir(parents=True, exist_ok=True)

    round_idx = _next_round_index(base_dir)
    round_dir = base_dir / f"round{round_idx:02d}_frontend_matrix"
    round_dir.mkdir(parents=True, exist_ok=True)

    suite_dirs = {
        "triton_native_coverage_full": round_dir / "triton_native_coverage_full",
        "triton_flaggems_coverage_full": round_dir / "triton_flaggems_coverage_full",
        "tilelang_coverage_full": round_dir / "tilelang_coverage_full",
        "cuda_coverage_full": round_dir / "cuda_coverage_full",
    }
    for d in suite_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    plan = [
        {
            "name": "triton_native_coverage_full",
            "cmd": [
                sys.executable,
                "scripts/triton/full_pipeline_verify.py",
                "--suite",
                "coverage",
                "--cases-limit",
                str(int(args.cases_limit)),
                "--backend-target",
                backend_target,
                "--out-dir",
                str(suite_dirs["triton_native_coverage_full"]),
            ],
        },
        {
            "name": "triton_flaggems_coverage_full",
            "cmd": [
                sys.executable,
                "scripts/triton/flaggems_full_pipeline_verify.py",
                "--suite",
                "coverage",
                "--cases-limit",
                str(int(args.cases_limit)),
                "--flaggems-path",
                "intentir",
                "--intentir-mode",
                "auto",
                "--intentir-miss-policy",
                "strict",
                "--backend-target",
                backend_target,
                "--out-dir",
                str(suite_dirs["triton_flaggems_coverage_full"]),
            ],
        },
        {
            "name": "tilelang_coverage_full",
            "cmd": [
                sys.executable,
                "scripts/tilelang/full_pipeline_verify.py",
                "--suite",
                "coverage",
                "--cases-limit",
                str(int(args.cases_limit)),
                "--backend-target",
                backend_target,
                "--out-dir",
                str(suite_dirs["tilelang_coverage_full"]),
            ],
        },
        {
            "name": "cuda_coverage_full",
            "cmd": [
                sys.executable,
                "scripts/cuda/full_pipeline_verify.py",
                "--suite",
                "coverage",
                "--cases-limit",
                str(int(args.cases_limit)),
                "--backend-target",
                backend_target,
                "--out-dir",
                str(suite_dirs["cuda_coverage_full"]),
            ],
        },
    ]

    stages: list[dict[str, Any]] = []
    for row in plan:
        name = str(row["name"])
        cmd = [str(x) for x in list(row["cmd"])]
        print(f"\n[matrix] START {name}: {' '.join(cmd)}", flush=True)
        rc, sec = _run(cmd, cwd=ROOT, stream=bool(args.stream))
        summary = _summarize_kernel_reports(Path(suite_dirs[name]))
        stages.append(
            {
                "stage": name,
                "rc": int(rc),
                "sec": float(sec),
                "summary": dict(summary),
            }
        )
        print(f"[matrix] DONE {name}: rc={rc} sec={sec:.2f}", flush=True)

    perf_rows: list[dict[str, Any]] = []
    if bool(args.with_perf):
        perf_dir = Path(args.perf_out_dir) if str(args.perf_out_dir).strip() else (round_dir / "perf")
        perf_dir.mkdir(parents=True, exist_ok=True)

        for suite_name in ("gpu-perf-triton-native", "gpu-perf-graph"):
            out_root = perf_dir / suite_name
            cmd = [
                sys.executable,
                "scripts/intentir.py",
                "suite",
                "--suite",
                suite_name,
                "--backend-target",
                backend_target,
                "--out-root",
                str(out_root),
            ]
            print(f"\n[matrix] START perf:{suite_name}: {' '.join(cmd)}", flush=True)
            rc, sec = _run(cmd, cwd=ROOT, stream=bool(args.stream))
            perf_rows.append(
                {
                    "suite": str(suite_name),
                    "rc": int(rc),
                    "sec": float(sec),
                    "out_root": str(out_root),
                }
            )
            print(f"[matrix] DONE perf:{suite_name}: rc={rc} sec={sec:.2f}", flush=True)

    out_summary = {
        "schema_version": "intentir_frontend_matrix_summary_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "date_tag": str(date_tag),
        "backend_target": str(backend_target),
        "cases_limit": int(args.cases_limit),
        "round_dir": str(round_dir),
        "stages": stages,
        "perf": perf_rows,
    }
    summary_path = round_dir / "matrix_summary.json"
    _dump_json(summary_path, out_summary)
    print(f"\n[matrix] summary: {summary_path}", flush=True)
    # Non-zero if any stage fails (perf suites might be monitor-only).
    stage_fail = any(int(s.get("rc") or 0) != 0 for s in stages)
    raise SystemExit(1 if stage_fail else 0)


if __name__ == "__main__":
    main()

