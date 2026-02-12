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
DEFAULT_RVV_HOST = os.getenv("INTENTIR_RVV_HOST", "192.168.8.72")
DEFAULT_RVV_USER = os.getenv("INTENTIR_RVV_USER", "ubuntu")

DEFAULT_KERNELS = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]

FLAGGEMS_KERNELS = [
    "any_kernel_dim",
    "add2d",
    "group_norm_kernel",
    "layer_norm_persistent",
    "softmax_inner",
    "upsample_bicubic2d_aa",
]


def _run_one(
    *,
    kernel: str,
    frontend: str,
    triton_provider: str,
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
    if frontend == "triton":
        cmd += ["--triton-provider", str(triton_provider)]
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
    ap.add_argument(
        "--triton-provider",
        choices=["native", "flaggems"],
        default="native",
        help="Triton artifact provider (default: native)",
    )
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument(
        "--host",
        default=DEFAULT_RVV_HOST,
        help=f"RVV host (default: {DEFAULT_RVV_HOST}; env: INTENTIR_RVV_HOST)",
    )
    ap.add_argument(
        "--user",
        default=DEFAULT_RVV_USER,
        help=f"SSH user (default: {DEFAULT_RVV_USER}; env: INTENTIR_RVV_USER)",
    )
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

    frontends = ["triton", "tilelang"] if args.frontend == "both" else [str(args.frontend)]
    explicit_kernels = list(args.kernel or [])

    password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
    results: List[Dict[str, Any]] = []
    kernels_by_frontend: Dict[str, List[str]] = {}
    for fe in frontends:
        kernels = (
            list(explicit_kernels)
            if explicit_kernels
            else (list(FLAGGEMS_KERNELS) if fe == "triton" and str(args.triton_provider) == "flaggems" else list(DEFAULT_KERNELS))
        )
        kernels_by_frontend[str(fe)] = list(kernels)
        for k in kernels:
            results.append(
                _run_one(
                    kernel=k,
                    frontend=fe,
                    triton_provider=str(args.triton_provider),
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

    kernels_union: List[str] = []
    for ks in kernels_by_frontend.values():
        for k in ks:
            if k not in kernels_union:
                kernels_union.append(str(k))

    out: Dict[str, Any] = {
        "host": str(args.host),
        "user": str(args.user),
        "port": int(args.port),
        "frontends": frontends,
        "triton_provider": (str(args.triton_provider) if "triton" in frontends else None),
        "kernels": kernels_union,
        "kernels_by_frontend": kernels_by_frontend,
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
