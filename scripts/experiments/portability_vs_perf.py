"""
E5: Portability vs Performance (performance decoupling) experiment runner.

This script focuses on the "freeze tile vs retune" comparison:
  - Freeze: reuse frontend tile/constexpr parameters on the RVV target (no tuning)
  - Retune: select schedule on the RVV target via our tuning interface

The experiment is designed to be paper-friendly:
  - Uses the same artifact pipeline as the Triton full pipeline (LLM -> IntentIR -> contract).
  - Runs on a real remote RVV host via scripts/rvv_remote_run.py.
  - Produces a single JSON report suitable for tables/plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import coverage_kernel_specs, run_pipeline_for_spec  # noqa: E402
from scripts.rvv_remote_run import run_remote  # noqa: E402
from backends.spmd_rvv.analysis.tuning import TuningRequest  # noqa: E402
from backends.spmd_rvv.baseline_freeze_tile import freeze_tile_schedule  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "full_pipeline_verify"


def _log(msg: str) -> None:
    print(str(msg), file=sys.stderr, flush=True)


def _ensure_artifact(kernel_name: str, *, cases_limit: int, refresh: bool) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / f"{kernel_name}.json"
    if report_path.exists() and not refresh:
        return report_path

    spec_map = {s.name: s for s in coverage_kernel_specs()}
    if kernel_name not in spec_map:
        raise KeyError(f"kernel not found in triton coverage specs: {kernel_name}")
    spec = spec_map[kernel_name]
    _log(f"[E5:{kernel_name}] generate artifacts (cases_limit={cases_limit}, refresh={refresh})")
    report = run_pipeline_for_spec(spec, out_dir=ARTIFACT_DIR, cases_limit=int(cases_limit))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def _seconds_per_iter(res: dict) -> Optional[float]:
    bench = res.get("bench")
    if not isinstance(bench, dict):
        return None
    ns = bench.get("ns_per_iter")
    if not isinstance(ns, (int, float)) or ns <= 0:
        return None
    return float(ns) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--password", default=None)
    ap.add_argument("--use-key", action="store_true")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--kernel", action="append", default=None, help="repeatable; default: ai_bench_matmul only")
    ap.add_argument("--cases-limit", type=int, default=4)
    ap.add_argument("--refresh-artifacts", action="store_true", help="force regenerate pipeline artifacts")
    ap.add_argument("--bench-iters", type=int, default=5)
    ap.add_argument("--bench-warmup", type=int, default=1)
    ap.add_argument("--omp-threads", type=int, default=16)
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default="auto")
    ap.add_argument("--tune-budget", type=int, default=1, help=">1 enables measured autotune (requires bench-iters>0)")
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: probe remote host)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "portability_vs_perf.json"))
    args = ap.parse_args()

    wanted = list(args.kernel or [])
    if not wanted:
        wanted = ["ai_bench_matmul"]

    if int(args.tune_budget) > 1 and int(args.bench_iters) <= 0:
        raise ValueError("--tune-budget>1 requires --bench-iters>0 (measured autotune)")

    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_map = {s.name: s for s in coverage_kernel_specs()}

    results: list[dict[str, Any]] = []
    for k in wanted:
        if k not in spec_map:
            _log(f"[E5] skip unknown kernel: {k}")
            continue
        spec = spec_map[k]
        report_path = _ensure_artifact(k, cases_limit=int(args.cases_limit), refresh=bool(args.refresh_artifacts))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        intent = IntentFunction.from_json_dict(report["intent"])

        desc = report.get("descriptor") or {}
        # Backward-compat: older artifacts may not record launch constexpr values yet.
        # For E5 freeze-tile, use the frontend launcher config from KernelSpec.
        if isinstance(desc, dict):
            launch = desc.get("launch")
            if not isinstance(launch, dict):
                launch = {}
            if "constexpr" not in launch and getattr(spec, "constexpr", None):
                launch["constexpr"] = dict(getattr(spec, "constexpr") or {})
            desc["launch"] = launch

        frozen = freeze_tile_schedule(intent, desc=desc)
        freeze_sched = frozen.schedule

        shape_bindings = dict(getattr(spec, "canonical_shapes", {}) or {})
        _log(f"[E5:{k}] freeze vs retune (omp={int(args.omp_threads)})")

        freeze_res = run_remote(
            kernel=k,
            frontend="triton",
            host=str(args.host),
            user=str(args.user),
            password=None if bool(args.use_key) else (args.password or os.getenv("INTENTIR_SSH_PASSWORD")),
            port=int(args.port),
            case_index=0,
            shape_overrides=shape_bindings,
            tune_request=None,
            schedule_override=freeze_sched,
            tune_profile=(str(args.profile) if args.profile else None),
            bench_iters=int(args.bench_iters),
            bench_warmup=int(args.bench_warmup),
            bench_only=True,
            omp_threads=int(args.omp_threads),
            log=_log,
        )

        retune_res = run_remote(
            kernel=k,
            frontend="triton",
            host=str(args.host),
            user=str(args.user),
            password=None if bool(args.use_key) else (args.password or os.getenv("INTENTIR_SSH_PASSWORD")),
            port=int(args.port),
            case_index=0,
            shape_overrides=shape_bindings,
            tune_request=TuningRequest(mode=str(args.tune_mode), budget=int(args.tune_budget)),
            schedule_override=None,
            tune_profile=(str(args.profile) if args.profile else None),
            bench_iters=int(args.bench_iters),
            bench_warmup=int(args.bench_warmup),
            bench_only=True,
            omp_threads=int(args.omp_threads),
            log=_log,
        )

        freeze_s = _seconds_per_iter(freeze_res)
        retune_s = _seconds_per_iter(retune_res)
        speedup = None
        if isinstance(freeze_s, float) and isinstance(retune_s, float) and retune_s > 0:
            speedup = float(freeze_s) / float(retune_s)

        results.append(
            {
                "kernel": k,
                "shapes": shape_bindings,
                "freeze": {
                    "notes": list(frozen.notes),
                    "schedule": freeze_res.get("schedule"),
                    "seconds_per_iter": freeze_s,
                    "remote": freeze_res,
                },
                "retune": {
                    "tune_mode": str(args.tune_mode),
                    "tune_budget": int(args.tune_budget),
                    "schedule": retune_res.get("schedule"),
                    "seconds_per_iter": retune_s,
                    "remote": retune_res,
                },
                "speedup_retune_vs_freeze": speedup,
            }
        )

    out = {
        "experiment": "E5_portability_vs_perf_v1",
        "host": str(args.host),
        "omp_threads": int(args.omp_threads),
        "bench_iters": int(args.bench_iters),
        "bench_warmup": int(args.bench_warmup),
        "kernels": results,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
