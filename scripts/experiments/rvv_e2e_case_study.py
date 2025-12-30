"""
Paper utility: end-to-end RVV remote runs across the default 6 kernels.

This script is a thin orchestrator that calls `scripts/rvv_remote_run.py` so we:
  - reuse the same artifact format (IntentIR + baseline.npz)
  - reuse the same tuning / profile / bench flags
  - avoid importing "scripts as modules" (keep scripts user-facing)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_KERNELS = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]


def _run_one(
    *,
    kernel: str,
    frontend: str,
    host: str,
    user: str,
    port: int,
    password: str | None,
    case_index: int,
    tune_mode: str,
    tune_budget: int,
    profile: str | None,
    bench_iters: int,
    bench_warmup: int,
) -> Dict[str, Any]:
    cmd: List[str] = [
        sys.executable,
        str(ROOT / "scripts" / "rvv_remote_run.py"),
        "--kernel",
        str(kernel),
        "--frontend",
        str(frontend),
        "--host",
        str(host),
        "--user",
        str(user),
        "--port",
        str(int(port)),
        "--case-index",
        str(int(case_index)),
        "--tune-mode",
        str(tune_mode),
        "--tune-budget",
        str(int(tune_budget)),
        "--bench-iters",
        str(int(bench_iters)),
        "--bench-warmup",
        str(int(bench_warmup)),
        "--json",
    ]
    if profile:
        cmd += ["--profile", str(profile)]

    env = dict(os.environ)
    if password is not None:
        env["INTENTIR_SSH_PASSWORD"] = str(password)

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return {"kernel": kernel, "frontend": frontend, "ok": False, "error": "subprocess failed", "stderr": proc.stderr.strip(), "stdout": proc.stdout.strip()}
    try:
        data = json.loads(proc.stdout)
    except Exception as e:
        return {"kernel": kernel, "frontend": frontend, "ok": False, "error": f"parse json failed: {type(e).__name__}: {e}", "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    data["kernel"] = kernel
    data["frontend"] = frontend
    data["ok"] = bool(data.get("compile_rc") == 0 and data.get("run_rc") == 0)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "both"], default="both")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--password", default=None, help="SSH password (prefer env INTENTIR_SSH_PASSWORD)")
    ap.add_argument("--case-index", type=int, default=0)
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default="auto")
    ap.add_argument("--tune-budget", type=int, default=1)
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: query remote host)")
    ap.add_argument("--bench-iters", type=int, default=0)
    ap.add_argument("--bench-warmup", type=int, default=1)
    ap.add_argument("--out", default=None, help="write JSON report to this path (default: stdout)")
    args = ap.parse_args()

    kernels = args.kernel or list(DEFAULT_KERNELS)
    frontends = ["triton", "tilelang"] if args.frontend == "both" else [str(args.frontend)]

    password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
    results: List[Dict[str, Any]] = []
    for fe in frontends:
        for k in kernels:
            results.append(
                _run_one(
                    kernel=k,
                    frontend=fe,
                    host=str(args.host),
                    user=str(args.user),
                    port=int(args.port),
                    password=password,
                    case_index=int(args.case_index),
                    tune_mode=str(args.tune_mode),
                    tune_budget=int(args.tune_budget),
                    profile=str(args.profile) if args.profile else None,
                    bench_iters=int(args.bench_iters),
                    bench_warmup=int(args.bench_warmup),
                )
            )

    out: Dict[str, Any] = {
        "host": str(args.host),
        "user": str(args.user),
        "port": int(args.port),
        "frontends": frontends,
        "kernels": kernels,
        "tuning": {"mode": str(args.tune_mode), "budget": int(args.tune_budget), "bench_iters": int(args.bench_iters), "bench_warmup": int(args.bench_warmup), "profile": (str(args.profile) if args.profile else None)},
        "results": results,
    }

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

