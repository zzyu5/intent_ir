"""
Performance benchmark suite for the RVV backend (P0 gap fix).

This is a thin wrapper around `scripts/rvv_remote_suite.py` that:
  - runs the 6-kernel suite with INTENTIR_BENCH_ITERS enabled
  - extracts ns_per_iter (and other bench fields) into a compact JSON

Typical usage:
  PYTHONPATH=. python scripts/benchmark_suite.py --host 192.168.8.149 --user ubuntu --use-key \\
    --frontend both --bench-iters 50 --bench-warmup 5 --out artifacts/perf_latest.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.analysis.perf_analysis import extract_remote_suite_rows, summarize_rows  # noqa: E402


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        return str(out)
    except Exception:
        return None


def _run_remote_suite(raw_out: Path, argv: List[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "rvv_remote_suite.py")] + argv + ["--out", str(raw_out)]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"rvv_remote_suite failed (rc={rc})")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "both"], default="both")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--password", default=None)
    ap.add_argument("--use-key", action="store_true")
    ap.add_argument("--case-index", type=int, default=0)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default="auto")
    ap.add_argument("--tune-budget", type=int, default=1)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--profile-ops", action="store_true", help="also collect per-op timing (INTENTIR_PROFILE)")
    ap.add_argument("--tune-debug", action="store_true", help="include structured tuning debug in raw output (enables pred-vs-measured analysis)")
    ap.add_argument("--out", required=True, help="write compact perf JSON to this path")
    ap.add_argument("--raw-out", default=None, help="optional: save raw rvv_remote_suite JSON too")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out = Path(args.raw_out) if args.raw_out else out_path.with_suffix(".raw.json")
    raw_out.parent.mkdir(parents=True, exist_ok=True)

    argv: List[str] = [
        "--frontend",
        str(args.frontend),
        "--host",
        str(args.host),
        "--user",
        str(args.user),
        "--port",
        str(int(args.port)),
        "--case-index",
        str(int(args.case_index)),
        "--tune-mode",
        str(args.tune_mode),
        "--tune-budget",
        str(int(args.tune_budget)),
        "--bench-iters",
        str(int(args.bench_iters)),
        "--bench-warmup",
        str(int(args.bench_warmup)),
    ]
    for k in args.kernel or []:
        argv += ["--kernel", str(k)]
    if args.use_key:
        argv.append("--use-key")
    if args.no_tune:
        argv.append("--no-tune")
    if args.profile:
        argv += ["--profile", str(args.profile)]
    if (not args.use_key) and args.password:
        argv += ["--password", str(args.password)]
    if args.profile_ops:
        argv.append("--profile-ops")
    if args.tune_debug:
        argv.append("--tune-debug")

    t0 = time.time()
    _run_remote_suite(raw_out, argv)
    raw = json.loads(raw_out.read_text(encoding="utf-8"))
    rows, errors = extract_remote_suite_rows(raw)
    dt = time.time() - t0

    out: Dict[str, Any] = {
        "kind": "rvv_benchmark_suite",
        "timestamp": int(time.time()),
        "git_commit": _git_head(),
        "runtime_s": float(dt),
        "config": {
            "frontend": str(args.frontend),
            "bench_iters": int(args.bench_iters),
            "bench_warmup": int(args.bench_warmup),
            "tune_mode": str(args.tune_mode),
            "tune_budget": int(args.tune_budget),
            "no_tune": bool(args.no_tune),
            "profile_ops": bool(args.profile_ops),
            "tune_debug": bool(args.tune_debug),
            "case_index": int(args.case_index),
        },
        "remote": {
            "host": str(args.host),
            "user": str(args.user),
            "port": int(args.port),
            "profile": raw.get("profile") or raw.get("tuning", {}).get("profile"),
        },
        "results": rows,
        "summary": summarize_rows(rows),
        "ok": bool(not errors),
        "errors": errors,
        "raw_path": str(raw_out),
    }

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
