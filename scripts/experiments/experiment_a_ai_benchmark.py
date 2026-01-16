"""
Experiment A (E5 baseline): compare Triton-CPU (external baseline) vs IntentIR->RVV backend.

Design notes:
  - Baseline numbers can be taken from the local AI-Benchmark report (fast), or measured
    by running the remote *.elf binaries (slow).
  - Our RVV backend runs on the remote RISC-V host via scripts/rvv_remote_run.py and uses
    --bench-only to avoid uploading huge IO blobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import paramiko

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import coverage_kernel_specs, run_pipeline_for_spec
from scripts.rvv_remote_run import run_remote
from backends.spmd_rvv.analysis.tuning import TuningRequest


ARTIFACT_DIR = ROOT / "artifacts" / "full_pipeline_verify"
AI_BENCH_REPORT = ROOT / "experiment" / "AI-Benchmark" / "COMPLETE_PERFORMANCE_SUMMARY.md"


@dataclass(frozen=True)
class BaselineKernel:
    name: str
    binary: str
    arg: str
    run_count: int
    # Total time in seconds for the full RUN_COUNT loop on the RISC-V host.
    t1_total_s: float
    t16_total_s: float

    def per_iter_s(self, threads: int) -> float:
        total = self.t1_total_s if threads == 1 else self.t16_total_s
        return float(total) / float(self.run_count)


BASELINE_KERNELS: dict[str, BaselineKernel] = {
    "matmul": BaselineKernel(
        name="matmul",
        binary="matmul.elf",
        arg="256x512x256x10",
        run_count=10,
        t1_total_s=2.149,
        t16_total_s=0.086,
    ),
    "dropout": BaselineKernel(
        name="dropout",
        binary="dropout.elf",
        arg="1048576x100",
        run_count=100,
        t1_total_s=1.392,
        t16_total_s=0.117,
    ),
    "softmax": BaselineKernel(
        name="softmax",
        binary="softmax_kernel.elf",
        arg="1823x781x100",
        run_count=100,
        t1_total_s=6.765,
        t16_total_s=0.565,
    ),
    "layernorm": BaselineKernel(
        name="layernorm",
        binary="layernorm.elf",
        arg="1151x8192x100",
        run_count=100,
        t1_total_s=8.659,
        t16_total_s=1.701,
    ),
    "correlation": BaselineKernel(
        name="correlation",
        binary="correlation.elf",
        arg="5x58x112x88x100",
        run_count=100,
        t1_total_s=0.369,
        t16_total_s=0.038,
    ),
    "resize": BaselineKernel(
        name="resize",
        binary="resize.elf",
        arg="512x512x3x100",
        run_count=100,
        t1_total_s=5.251,
        t16_total_s=0.332,
    ),
    "rope": BaselineKernel(
        name="rope",
        binary="rope.elf",
        arg="512x16x8x1024x100",
        run_count=100,
        t1_total_s=65.512,
        t16_total_s=4.475,
    ),
    "warp": BaselineKernel(
        name="warp",
        binary="warp.elf",
        arg="1024x1024x3x100",
        run_count=100,
        t1_total_s=2.808,
        t16_total_s=0.169,
    ),
}


OURS_EQUIV: dict[str, dict[str, Any]] = {
    # baseline_name -> our pipeline kernel name + shape bindings
    "matmul": {"kernel": "ai_bench_matmul", "shapes": {"M": 256, "N": 512, "K": 256}},
    "softmax": {"kernel": "ai_bench_softmax", "shapes": {"R": 1823, "C": 781}},
    # NOTE: AI-Benchmark's layernorm baseline measures fwd+bwd; our current kernel is fwd-only.
    "layernorm": {"kernel": "ai_bench_layernorm", "shapes": {"M": 1151, "N": 8192}},
    "rope": {"kernel": "ai_bench_rope", "shapes": {"SEQ_LEN": 512, "BATCH_NUM": 16, "HEAD_NUM": 8, "HEAD_DIM": 1024}},
}


def _log(msg: str) -> None:
    print(str(msg), file=sys.stderr, flush=True)


def _parse_baseline_time_from_output(text: str) -> Optional[float]:
    # Prints: "Running Triton Kernel Time: <value> s"
    m = re.search(r"Running\\s+Triton\\s+Kernel\\s+Time:\\s*([0-9]*\\.?[0-9]+)\\s*s", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _remote_run_baseline(
    *,
    host: str,
    user: str,
    password: str | None,
    use_key: bool,
    port: int,
    remote_root: str,
    kernel: BaselineKernel,
    threads: int,
    timeout_s: int,
) -> float:
    cmd = (
        f"cd {remote_root}/bin/triton && "
        f"TRITON_CPU_MAX_THREADS={int(threads)} "
        f"./{kernel.binary} {kernel.arg}"
    )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, password=None if use_key else password, timeout=20)
    try:
        _log(f"[baseline:{kernel.name}] {cmd}")
        _, stdout, stderr = client.exec_command(cmd, timeout=int(timeout_s))
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        t = _parse_baseline_time_from_output(err + "\n" + out)
        if t is None:
            raise RuntimeError(f"failed to parse baseline time for {kernel.name}. stdout={out!r} stderr={err!r}")
        return float(t)
    finally:
        client.close()


def _ensure_ours_artifact(kernel_name: str, *, cases_limit: int) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / f"{kernel_name}.json"
    if report_path.exists():
        return
    spec_map = {s.name: s for s in coverage_kernel_specs()}
    if kernel_name not in spec_map:
        raise KeyError(f"kernel not found in triton coverage specs: {kernel_name}")
    _log(f"[ours:{kernel_name}] generating artifacts via pipeline (cases_limit={cases_limit})")
    report = run_pipeline_for_spec(spec_map[kernel_name], out_dir=ARTIFACT_DIR, cases_limit=int(cases_limit))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_ours_remote_bench(
    *,
    kernel_name: str,
    host: str,
    user: str,
    password: str | None,
    use_key: bool,
    port: int,
    shapes: dict[str, int],
    bench_iters: int,
    bench_warmup: int,
    omp_threads: int,
    tune: bool,
    timeout_s: int,
) -> dict:
    # scripts/rvv_remote_run.py currently has a fixed paramiko timeout of 60s;
    # use a conservative bench_iters for large shapes, and keep this script-side
    # timeout as best-effort (rvv_remote_run itself may still time out).
    res = run_remote(
        kernel=kernel_name,
        frontend="triton",
        host=host,
        user=user,
        password=None if use_key else password,
        port=int(port),
        case_index=0,
        shape_overrides=dict(shapes),
        tune_request=(TuningRequest(mode="auto", budget=1) if bool(tune) else None),
        bench_iters=int(bench_iters),
        bench_warmup=int(bench_warmup),
        bench_only=True,
        omp_threads=int(omp_threads),
        log=_log,
    )
    bench = res.get("bench")
    if not isinstance(bench, dict):
        raise RuntimeError(f"missing bench result for {kernel_name}: {res}")
    ns_per_iter = bench.get("ns_per_iter")
    if not isinstance(ns_per_iter, (int, float)):
        raise RuntimeError(f"missing ns_per_iter for {kernel_name}: {bench}")
    res["bench_seconds_per_iter"] = float(ns_per_iter) / 1e9
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--password", default=None)
    ap.add_argument("--use-key", action="store_true")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--baseline-root", default="/home/ubuntu/triton_benchmark")
    ap.add_argument("--baseline-mode", choices=["report", "remote"], default="report")
    ap.add_argument("--baseline-threads", type=int, default=1, choices=[1, 16])
    ap.add_argument("--no-tune", action="store_true", help="disable schedule selection (use artifact schedule as-is)")
    ap.add_argument("--cases-limit", type=int, default=4)
    ap.add_argument("--bench-iters", type=int, default=3)
    ap.add_argument("--bench-warmup", type=int, default=1)
    ap.add_argument("--ours-threads", type=int, default=1, help="OpenMP threads for our RVV backend (default: 1)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "experiment_a_ai_benchmark.json"))
    args = ap.parse_args()

    password: str | None = None
    if not bool(args.use_key):
        password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
        if password is None:
            raise SystemExit("missing SSH password (use --password, INTENTIR_SSH_PASSWORD, or --use-key)")

    baseline_threads = int(args.baseline_threads)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "baseline": {
            "mode": str(args.baseline_mode),
            "threads": baseline_threads,
            "source": str(AI_BENCH_REPORT) if args.baseline_mode == "report" else "remote",
        },
        "ours": {
            "bench_only": True,
            "bench_iters": int(args.bench_iters),
            "bench_warmup": int(args.bench_warmup),
            "omp_threads": int(args.ours_threads),
            "tune": (not bool(args.no_tune)),
        },
        "kernels": [],
    }

    for base_name, base in BASELINE_KERNELS.items():
        entry: dict[str, Any] = {
            "baseline_name": base_name,
            "baseline": {
                "binary": base.binary,
                "arg": base.arg,
                "run_count": base.run_count,
            },
            "ours": None,
        }
        if args.baseline_mode == "remote":
            t_total = _remote_run_baseline(
                host=str(args.host),
                user=str(args.user),
                password=password,
                use_key=bool(args.use_key),
                port=int(args.port),
                remote_root=str(args.baseline_root),
                kernel=base,
                threads=baseline_threads,
                timeout_s=int(args.timeout),
            )
            entry["baseline"]["measured_total_s"] = float(t_total)
            entry["baseline"]["seconds_per_iter"] = float(t_total) / float(base.run_count)
        else:
            entry["baseline"]["measured_total_s"] = float(base.t1_total_s if baseline_threads == 1 else base.t16_total_s)
            entry["baseline"]["seconds_per_iter"] = float(base.per_iter_s(baseline_threads))

        equiv = OURS_EQUIV.get(base_name)
        if equiv is None:
            entry["ours"] = {"status": "UNSUPPORTED", "reason": "no IntentIR/RVV equivalent yet"}
            results["kernels"].append(entry)
            continue

        kernel_name = str(equiv["kernel"])
        shapes = dict(equiv["shapes"])

        try:
            _ensure_ours_artifact(kernel_name, cases_limit=int(args.cases_limit))
        except Exception as e:
            entry["ours"] = {"status": "PIPELINE_FAIL", "kernel": kernel_name, "error": f"{type(e).__name__}: {e}"}
            results["kernels"].append(entry)
            continue

        try:
            ours = _run_ours_remote_bench(
                kernel_name=kernel_name,
                host=str(args.host),
                user=str(args.user),
                password=password,
                use_key=bool(args.use_key),
                port=int(args.port),
                shapes=shapes,
                bench_iters=int(args.bench_iters),
                bench_warmup=int(args.bench_warmup),
                omp_threads=int(args.ours_threads),
                tune=(not bool(args.no_tune)),
                timeout_s=int(args.timeout),
            )
            entry["ours"] = {"status": "OK", "kernel": kernel_name, "shapes": shapes, "remote": ours}
            ours_s = float(ours.get("bench_seconds_per_iter") or 0.0)
            base_s = float(entry["baseline"]["seconds_per_iter"])
            if ours_s > 0 and base_s > 0:
                entry["speedup_baseline_over_ours"] = float(ours_s / base_s)  # <1 means ours faster
                entry["speedup_ours_over_baseline"] = float(base_s / ours_s)
        except Exception as e:
            entry["ours"] = {"status": "REMOTE_FAIL", "kernel": kernel_name, "error": f"{type(e).__name__}: {e}"}

        results["kernels"].append(entry)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(str(out_path))


if __name__ == "__main__":
    main()
