"""
Run RVV remote tests across kernel suites (smoke/coverage/all).

This is the "remote equivalent" of `scripts/backend_codegen_smoke.py`:
- backend_codegen_smoke: local compile+run (no SSH)
- rvv_remote_suite: remote compile+run on a real RVV host (SSH)

It reuses `scripts/rvv_remote_run.py` for the per-kernel logic.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_KERNELS = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]

P3_COVERAGE_KERNELS = [
    "add2d",
    "transpose2d",
]

KERNEL_SUITES = {
    "smoke": list(DEFAULT_KERNELS),
    "coverage": list(DEFAULT_KERNELS) + list(P3_COVERAGE_KERNELS),
    "all": list(DEFAULT_KERNELS) + list(P3_COVERAGE_KERNELS),
}


def _run_one(cmd: List[str], *, env: Dict[str, str]) -> Dict[str, Any]:
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stderr_buf: List[str] = []

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            stderr_buf.append(line)
            # Stream progress logs to the user (remote_run logs go to stderr).
            sys.stderr.write(line)
            sys.stderr.flush()

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()
    stdout = ""
    if proc.stdout is not None:
        stdout = proc.stdout.read()
    rc = proc.wait()
    try:
        t.join(timeout=2.0)
    except Exception:
        pass
    stderr = "".join(stderr_buf).strip()

    if rc != 0:
        return {
            "ok": False,
            "error": "subprocess failed",
            "rc": int(rc),
            "stdout": (stdout or "").strip(),
            "stderr": stderr,
        }
    try:
        return json.loads(stdout)
    except Exception as e:
        return {
            "ok": False,
            "error": f"parse json failed: {type(e).__name__}: {e}",
            "stdout": (stdout or "").strip(),
            "stderr": stderr,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "both"], default="both")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke", help="Kernel suite (default: smoke)")
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--password", default=None, help="SSH password (prefer env INTENTIR_SSH_PASSWORD or prompt)")
    ap.add_argument("--use-key", action="store_true", help="use SSH key auth (no password prompt)")
    ap.add_argument("--case-index", type=int, default=0)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default="auto")
    ap.add_argument("--tune-budget", type=int, default=1)
    ap.add_argument("--profile", default=None, help="RVV profile name/JSON path; default probes once and reuses")
    ap.add_argument("--bench-iters", type=int, default=0)
    ap.add_argument("--bench-warmup", type=int, default=1)
    ap.add_argument("--profile-ops", action="store_true", help="enable per-op timing JSON from the RVV program")
    ap.add_argument("--tune-debug", action="store_true", help="include structured tuning/cost-model debug in JSON output")
    ap.add_argument("--out", default=None, help="write JSON report to this path (default: stdout)")
    args = ap.parse_args()

    kernels = args.kernel or list(KERNEL_SUITES[str(args.suite)])
    frontends = ["triton", "tilelang"] if args.frontend == "both" else [str(args.frontend)]

    base_env = dict(os.environ)
    if not bool(args.use_key):
        password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
        if password is None:
            password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")
        base_env["INTENTIR_SSH_PASSWORD"] = str(password)
    else:
        base_env.pop("INTENTIR_SSH_PASSWORD", None)

    # Probe hardware profile once and reuse across all kernels (avoids repeated remote probe per subprocess).
    profile_path: Optional[str] = str(args.profile) if args.profile else None
    probed_profile: Optional[dict] = None
    if (not bool(args.no_tune)) and profile_path is None:
        try:
            from backends.spmd_rvv.analysis.device_query import query_remote_device

            t0 = time.time()
            prof = query_remote_device(
                host=str(args.host),
                user=str(args.user),
                password=(None if bool(args.use_key) else base_env.get("INTENTIR_SSH_PASSWORD")),
                port=int(args.port),
                timeout=60,
            )
            probed_profile = dict(prof.__dict__)
            cache_dir = ROOT / "artifacts" / "rvv_profiles"
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_host = str(args.host).replace("/", "_").replace(":", "_")
            profile_file = cache_dir / f"{safe_host}_{int(args.port)}.json"
            profile_file.write_text(json.dumps(probed_profile, indent=2, ensure_ascii=False), encoding="utf-8")
            profile_path = str(profile_file)
            dt = time.time() - t0
            print(f"[probe] cached RVV profile to {profile_file} ({dt:.2f}s)", file=sys.stderr, flush=True)
        except Exception as e:
            print(
                f"[probe] WARNING: remote profile probe failed, will fall back to per-kernel probe: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            profile_path = None

    results: List[Dict[str, Any]] = []
    ok_all = True
    for fe in frontends:
        for k in kernels:
            cmd: List[str] = [
                sys.executable,
                str(ROOT / "scripts" / "rvv_remote_run.py"),
                "--kernel",
                str(k),
                "--frontend",
                str(fe),
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
                "--json",
            ]
            if args.profile_ops:
                cmd.append("--profile-ops")
            if args.tune_debug:
                cmd.append("--tune-debug")
            if args.use_key:
                cmd.append("--use-key")
            if args.no_tune:
                cmd.append("--no-tune")
            if profile_path:
                cmd += ["--profile", str(profile_path)]

            r = _run_one(cmd, env=base_env)
            r["kernel"] = str(k)
            r["frontend"] = str(fe)
            r["ok"] = bool(r.get("compile_rc") == 0 and r.get("run_rc") == 0) if "compile_rc" in r else bool(r.get("ok"))
            ok_all = ok_all and bool(r["ok"])
            results.append(r)

            status = "OK" if r["ok"] else "FAIL"
            print(f"[{fe}:{k}] {status}")

    out: Dict[str, Any] = {
        "host": str(args.host),
        "user": str(args.user),
        "port": int(args.port),
        "frontends": frontends,
        "kernels": kernels,
        "tuning": {
            "enabled": (not bool(args.no_tune)),
            "mode": str(args.tune_mode),
            "budget": int(args.tune_budget),
            "profile": (str(profile_path) if profile_path else None),
            "bench_iters": int(args.bench_iters),
            "bench_warmup": int(args.bench_warmup),
        },
        "profile": (dict(probed_profile) if probed_profile else None),
        "results": results,
        "ok": bool(ok_all),
    }

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)

    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
