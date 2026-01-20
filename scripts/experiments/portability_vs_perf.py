"""
E5.2: Performance decoupling ablation (freeze-tile vs retune).

This script focuses on the "freeze tile vs retune" comparison:
  - Freeze: reuse frontend tile/constexpr parameters on the RVV target (no tuning)
  - Retune: select schedule on the RVV target via our tuning interface

The experiment is designed to be paper-friendly:
  - Uses the same artifact pipeline as the Triton full pipeline (LLM -> IntentIR -> contract).
  - Runs on a real remote RVV host via scripts/rvv_remote_run.py.
  - Produces a single JSON report suitable for tables/plots.

E5.1 (external baseline comparison) is handled separately by:
  - scripts/experiments/experiment_a_ai_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import math

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import coverage_kernel_specs, run_pipeline_for_spec  # noqa: E402
from scripts.rvv_remote_run import run_remote  # noqa: E402
from backends.spmd_rvv.analysis.tuning import TuningRequest  # noqa: E402
from backends.spmd_rvv.analysis.device_query import query_remote_device  # noqa: E402
from backends.spmd_rvv.baseline_freeze_tile import freeze_tile_schedule  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402


ARTIFACT_DIR = ROOT / "artifacts" / "full_pipeline_verify"

AI_BENCH_KERNELS: list[str] = [
    "ai_bench_matmul",
    "ai_bench_dropout",
    "ai_bench_softmax",
    "ai_bench_layernorm",
    "ai_bench_correlation",
    "ai_bench_resize",
    "ai_bench_rope",
    "ai_bench_warp",
]

# Use the same "real" shapes as the external AI-Benchmark baseline harness.
# (Some KernelSpec.canonical_shapes are smaller to keep the GPU artifact pipeline
# snappy; E5.2 is a performance experiment so we prefer realistic workloads.)
AI_BENCH_SHAPES: dict[str, dict[str, int]] = {
    "ai_bench_matmul": {"M": 256, "N": 512, "K": 256},
    "ai_bench_dropout": {"n_elements": 1048576},
    "ai_bench_softmax": {"R": 1823, "C": 781},
    "ai_bench_layernorm": {"M": 1151, "N": 8192},
    "ai_bench_correlation": {"out_channel": 5, "in_channel": 58, "height": 112, "width": 88, "out_shift": 0},
    "ai_bench_resize": {"C": 3, "H": 512, "W": 512, "OH": 1024, "OW": 1024},
    "ai_bench_rope": {"SEQ_LEN": 512, "BATCH_NUM": 16, "HEAD_NUM": 8, "HEAD_DIM": 1024},
    "ai_bench_warp": {"C": 3, "H": 1024, "W": 1024},
}

DEFAULT6_KERNELS: list[str] = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]

TRITON_COVERAGE_SUITE_NAME = "triton_coverage"


def _log(msg: str) -> None:
    print(str(msg), file=sys.stderr, flush=True)


def _anchor_tier_from_contract(contract: dict) -> str:
    try:
        signals = contract.get("signals") if isinstance(contract, dict) else None
        anchors = (signals or {}).get("anchors") if isinstance(signals, dict) else None
        if isinstance(anchors, dict):
            if bool(anchors.get("has_dot")):
                return "A_dot"
            if bool(anchors.get("has_reduce")):
                return "B_reduce"
            if bool(anchors.get("has_copy")):
                return "C_copy"
    except Exception:
        pass
    return "D_none"


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


def _paired_speedup_from_measured_autotune(retune_remote: dict) -> Dict[str, Any] | None:
    """
    Compute a *paired* freeze-vs-retune speedup from a single measured_autotune run.

    This avoids cross-run drift/noise (freeze and retune are otherwise two separate
    remote benchmark invocations).
    """
    if not isinstance(retune_remote, dict):
        return None
    tuning = retune_remote.get("tuning")
    if not isinstance(tuning, dict):
        return None
    ma = tuning.get("measured_autotune")
    if not isinstance(ma, dict):
        return None
    evaluated = ma.get("evaluated")
    if not isinstance(evaluated, list) or not evaluated:
        return None
    best_index = ma.get("best_index")
    if not isinstance(best_index, int) or not (0 <= best_index < len(evaluated)):
        return None

    def _bench_ns(x: Any) -> float | None:
        if not isinstance(x, dict):
            return None
        b = x.get("bench")
        if not isinstance(b, dict):
            return None
        ns = b.get("ns_per_iter")
        if not isinstance(ns, (int, float)) or ns <= 0:
            return None
        return float(ns)

    baseline = None
    for x in evaluated:
        if not isinstance(x, dict):
            continue
        notes = x.get("notes")
        if not isinstance(notes, list):
            continue
        if any("freeze_baseline" in str(n) for n in notes):
            baseline = x
            break
    if baseline is None:
        return None

    best = evaluated[best_index]
    b_ns = _bench_ns(baseline)
    best_ns = _bench_ns(best)
    if b_ns is None or best_ns is None or best_ns <= 0:
        return None

    return {
        "baseline_index": int(baseline.get("idx") if isinstance(baseline.get("idx"), int) else evaluated.index(baseline)),
        "baseline_ns_per_iter": b_ns,
        "baseline_schedule": (baseline.get("schedule") if isinstance(baseline.get("schedule"), dict) else None),
        "baseline_notes": (list(baseline.get("notes") or []) if isinstance(baseline.get("notes"), list) else []),
        "best_index": int(best.get("idx") if isinstance(best.get("idx"), int) else best_index),
        "best_ns_per_iter": best_ns,
        "best_schedule": (best.get("schedule") if isinstance(best.get("schedule"), dict) else None),
        "best_notes": (list(best.get("notes") or []) if isinstance(best.get("notes"), list) else []),
        "speedup": float(b_ns) / float(best_ns),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--password", default=None)
    ap.add_argument("--use-key", action="store_true")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--kernel", action="append", default=None, help="repeatable; explicit kernel names to run")
    ap.add_argument(
        "--suite",
        action="append",
        default=[],
        choices=["ai_bench8", "default6", "all", TRITON_COVERAGE_SUITE_NAME],
        help="kernel suite to run (repeatable)",
    )
    ap.add_argument("--cases-limit", type=int, default=4)
    ap.add_argument("--refresh-artifacts", action="store_true", help="force regenerate pipeline artifacts")
    ap.add_argument("--bench-iters", type=int, default=50, help="bench iterations per schedule candidate (paper default: 50)")
    ap.add_argument("--bench-warmup", type=int, default=5, help="warmup iterations before timing (paper default: 5)")
    ap.add_argument("--bench-seed", type=int, default=0, help="deterministic bench input seed (freeze/retune share this)")
    ap.add_argument("--omp-threads", type=int, default=16)
    ap.add_argument("--omp-proc-bind", default="spread", help="OpenMP proc bind policy for E5.2 (fixed to isolate schedule)")
    ap.add_argument("--tune-mode", choices=["auto", "guided", "locked"], default="auto")
    ap.add_argument(
        "--tune-budget",
        type=int,
        default=8,
        help=">1 enables measured autotune (requires bench-iters>0). Default=8 to reduce retune regressions.",
    )
    ap.add_argument("--profile", default=None, help="RVV profile name or JSON path (default: probe remote host)")
    ap.add_argument("--include-out-of-scope", action="store_true", help="include OUT_OF_SCOPE kernels (default: skip)")
    ap.add_argument("--retry-errors", action="store_true", help="retry kernels that previously failed in the output JSON")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "E5" / "portability_vs_perf.json"))
    args = ap.parse_args()

    wanted: list[str] = []
    wanted.extend(list(args.kernel or []))
    suites = list(args.suite or [])
    if "all" in suites:
        suites = ["ai_bench8", "default6"]
    if "ai_bench8" in suites:
        wanted.extend(AI_BENCH_KERNELS)
    if "default6" in suites:
        wanted.extend(DEFAULT6_KERNELS)
    # Triton full coverage suite (currently 38 kernels).
    if TRITON_COVERAGE_SUITE_NAME in suites:
        wanted.extend([s.name for s in coverage_kernel_specs()])
    if not wanted:
        wanted = list(AI_BENCH_KERNELS)
    # De-dup while preserving order.
    dedup: list[str] = []
    seen: set[str] = set()
    for k in wanted:
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    wanted = dedup

    if int(args.tune_budget) > 1 and int(args.bench_iters) <= 0:
        raise ValueError("--tune-budget>1 requires --bench-iters>0 (measured autotune)")

    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_map = {s.name: s for s in coverage_kernel_specs()}

    out_path = Path(args.out)
    existing_by_kernel: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            prev_list = prev.get("kernels")
            if isinstance(prev_list, list):
                for item in prev_list:
                    if isinstance(item, dict) and isinstance(item.get("kernel"), str):
                        existing_by_kernel[str(item["kernel"])] = item
        except Exception:
            existing_by_kernel = {}

    # Cache remote profile to avoid probing per-kernel.
    profile_name_or_path = str(args.profile) if args.profile else None
    if profile_name_or_path is None:
        cache_path = out_dir / f"rvv_profile_{str(args.host).replace(':', '_')}.json"
        if cache_path.exists():
            profile_name_or_path = str(cache_path)
        else:
            _log(f"[E5] probe remote RVV profile once: {args.user}@{args.host}")
            probe_password = None
            if not bool(args.use_key):
                probe_password = args.password or os.getenv("INTENTIR_SSH_PASSWORD")
            prof = query_remote_device(
                str(args.host),
                user=str(args.user),
                password=probe_password,
                port=int(args.port),
                timeout=60,
            )
            cache_path.write_text(json.dumps(prof.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
            profile_name_or_path = str(cache_path)

    results: list[dict[str, Any]] = []
    for k in wanted:
        prev = existing_by_kernel.get(k)
        if prev is not None:
            prev_status = prev.get("status")
            if prev_status == "ok" or prev_status == "skip_out_of_scope":
                results.append(prev)
                continue
            if not bool(args.retry_errors):
                results.append(prev)
                continue
        if k not in spec_map:
            _log(f"[E5] skip unknown kernel: {k}")
            continue
        spec = spec_map[k]
        try:
            report_path = _ensure_artifact(k, cases_limit=int(args.cases_limit), refresh=bool(args.refresh_artifacts))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            intent = IntentFunction.from_json_dict(report["intent"])
        except Exception as e:
            results.append({"kernel": k, "status": "artifact_error", "error": str(e)})
            continue

        contract = report.get("contract") or {}
        tier = _anchor_tier_from_contract(contract) if isinstance(contract, dict) else "D_none"
        if isinstance(contract, dict) and (contract.get("level") == "OUT_OF_SCOPE") and (not bool(args.include_out_of_scope)):
            item = {
                "kernel": k,
                "status": "skip_out_of_scope",
                "anchor_tier": tier,
                "contract": contract,
            }
            results.append(item)
            # Write progress so long runs can be resumed.
            out_path.write_text(
                json.dumps(
                    {
                        "experiment": "E5_portability_vs_perf_v1",
                        "host": str(args.host),
                        "omp_threads": int(args.omp_threads),
                        "bench_iters": int(args.bench_iters),
                        "bench_warmup": int(args.bench_warmup),
                        "bench_seed": int(args.bench_seed),
                        "tune_mode": str(args.tune_mode),
                        "tune_budget": int(args.tune_budget),
                        "profile": profile_name_or_path,
                        "kernels_requested": list(wanted),
                        "kernels": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            continue

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

        try:
            frozen = freeze_tile_schedule(intent, desc=desc)
            freeze_sched = frozen.schedule
            freeze_notes = list(frozen.notes)
        except Exception as e:
            freeze_sched = intent.schedule
            freeze_notes = [f"freeze_tile_error: {e}"]

        shape_bindings = dict(getattr(spec, "canonical_shapes", {}) or {})
        if k in AI_BENCH_SHAPES:
            shape_bindings = dict(AI_BENCH_SHAPES[k])
        _log(f"[E5:{k}] freeze vs retune (omp={int(args.omp_threads)})")

        try:
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
                tune_profile=profile_name_or_path,
                bench_iters=int(args.bench_iters),
                bench_warmup=int(args.bench_warmup),
                bench_seed=int(args.bench_seed),
                bench_only=True,
                omp_threads=int(args.omp_threads),
                omp_proc_bind=str(args.omp_proc_bind),
                log=_log,
            )
        except Exception as e:
            results.append({"kernel": k, "status": "freeze_error", "error": str(e)})
            out_path.write_text(
                json.dumps(
                    {
                        "experiment": "E5_portability_vs_perf_v1",
                        "host": str(args.host),
                        "omp_threads": int(args.omp_threads),
                        "bench_iters": int(args.bench_iters),
                        "bench_warmup": int(args.bench_warmup),
                        "tune_mode": str(args.tune_mode),
                        "tune_budget": int(args.tune_budget),
                        "profile": profile_name_or_path,
                        "kernels_requested": list(wanted),
                        "kernels": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            continue

        try:
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
                tune_profile=profile_name_or_path,
                bench_iters=int(args.bench_iters),
                bench_warmup=int(args.bench_warmup),
                bench_seed=int(args.bench_seed),
                bench_only=True,
                omp_threads=int(args.omp_threads),
                omp_proc_bind=str(args.omp_proc_bind),
                log=_log,
            )
        except Exception as e:
            results.append({"kernel": k, "status": "retune_error", "error": str(e)})
            out_path.write_text(
                json.dumps(
                    {
                        "experiment": "E5_portability_vs_perf_v1",
                        "host": str(args.host),
                        "omp_threads": int(args.omp_threads),
                        "bench_iters": int(args.bench_iters),
                        "bench_warmup": int(args.bench_warmup),
                        "bench_seed": int(args.bench_seed),
                        "tune_mode": str(args.tune_mode),
                        "tune_budget": int(args.tune_budget),
                        "profile": profile_name_or_path,
                        "kernels_requested": list(wanted),
                        "kernels": results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            continue

        freeze_s = _seconds_per_iter(freeze_res)
        retune_s = _seconds_per_iter(retune_res)
        speedup_unpaired = None
        if isinstance(freeze_s, float) and isinstance(retune_s, float) and retune_s > 0:
            speedup_unpaired = float(freeze_s) / float(retune_s)

        paired = _paired_speedup_from_measured_autotune(retune_res)
        speedup_paired = None
        if isinstance(paired, dict) and isinstance(paired.get("speedup"), (int, float)) and float(paired["speedup"]) > 0:
            speedup_paired = float(paired["speedup"])

        # Prefer paired numbers for paper plots (avoid cross-run drift).
        speedup = speedup_paired if isinstance(speedup_paired, float) else speedup_unpaired

        results.append(
            {
                "kernel": k,
                "status": "ok",
                "anchor_tier": tier,
                "shapes": shape_bindings,
                "contract": contract,
                "freeze": {
                    "notes": freeze_notes,
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
                "speedup_retune_vs_freeze_unpaired": speedup_unpaired,
                "speedup_retune_vs_freeze_paired": speedup_paired,
                "paired_autotune": paired,
            }
        )

        # Incremental checkpoint for long runs (resume-friendly).
        out_path.write_text(
            json.dumps(
                {
                    "experiment": "E5_portability_vs_perf_v1",
                    "host": str(args.host),
                    "omp_threads": int(args.omp_threads),
                    "bench_iters": int(args.bench_iters),
                    "bench_warmup": int(args.bench_warmup),
                    "bench_seed": int(args.bench_seed),
                    "tune_mode": str(args.tune_mode),
                    "tune_budget": int(args.tune_budget),
                    "profile": profile_name_or_path,
                    "kernels_requested": list(wanted),
                    "kernels": results,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    out = {
        "experiment": "E5_portability_vs_perf_v1",
        "host": str(args.host),
        "omp_threads": int(args.omp_threads),
        "omp_proc_bind": str(args.omp_proc_bind),
        "bench_iters": int(args.bench_iters),
        "bench_warmup": int(args.bench_warmup),
        "bench_seed": int(args.bench_seed),
        "tune_mode": str(args.tune_mode),
        "tune_budget": int(args.tune_budget),
        "profile": profile_name_or_path,
        "kernels_requested": list(wanted),
        "kernels": results,
    }
    # Paper-friendly aggregate metrics:
    # - report paired + unpaired to make noise sources explicit
    # - prefer paired for the "main" headline numbers
    speedups = []
    speedups_unpaired: list[float] = []
    freeze_total = 0.0
    retune_total = 0.0
    speedups_paired: list[float] = []
    freeze_total_paired = 0.0
    retune_total_paired = 0.0
    for it in results:
        if it.get("status") != "ok":
            continue
        fs = (it.get("freeze") or {}).get("seconds_per_iter")
        rs = (it.get("retune") or {}).get("seconds_per_iter")
        sp = it.get("speedup_retune_vs_freeze")
        sp_p = it.get("speedup_retune_vs_freeze_paired")
        if isinstance(fs, (int, float)) and isinstance(rs, (int, float)) and fs > 0 and rs > 0:
            freeze_total += float(fs)
            retune_total += float(rs)
        if isinstance(sp, (int, float)) and float(sp) > 0:
            speedups.append(float(sp))
        sp_u = it.get("speedup_retune_vs_freeze_unpaired")
        if isinstance(sp_u, (int, float)) and float(sp_u) > 0:
            speedups_unpaired.append(float(sp_u))

        # Paired totals (same measured_autotune run).
        paired = it.get("paired_autotune")
        if isinstance(paired, dict):
            b_ns = paired.get("baseline_ns_per_iter")
            best_ns = paired.get("best_ns_per_iter")
            if isinstance(b_ns, (int, float)) and isinstance(best_ns, (int, float)) and b_ns > 0 and best_ns > 0:
                freeze_total_paired += float(b_ns) / 1e9
                retune_total_paired += float(best_ns) / 1e9
        if isinstance(sp_p, (int, float)) and float(sp_p) > 0:
            speedups_paired.append(float(sp_p))
    geom = None
    if speedups:
        geom = math.exp(sum(math.log(x) for x in speedups) / float(len(speedups)))
    geom_unpaired = None
    if speedups_unpaired:
        geom_unpaired = math.exp(sum(math.log(x) for x in speedups_unpaired) / float(len(speedups_unpaired)))
    geom_paired = None
    if speedups_paired:
        geom_paired = math.exp(sum(math.log(x) for x in speedups_paired) / float(len(speedups_paired)))
    total_speedup = None
    if retune_total > 0:
        total_speedup = float(freeze_total) / float(retune_total)
    total_speedup_paired = None
    if retune_total_paired > 0:
        total_speedup_paired = float(freeze_total_paired) / float(retune_total_paired)
    out["summary"] = {
        "kernels_total": int(len(results)),
        "kernels_ok": int(sum(1 for it in results if it.get("status") == "ok")),
        "kernels_skipped_out_of_scope": int(sum(1 for it in results if it.get("status") == "skip_out_of_scope")),
        "kernels_errors": int(sum(1 for it in results if it.get("status") not in {"ok", "skip_out_of_scope"})),
        "geom_mean_speedup_retune_vs_freeze": geom,
        "geom_mean_speedup_retune_vs_freeze_unpaired": geom_unpaired,
        "geom_mean_speedup_retune_vs_freeze_paired": geom_paired,
        "total_seconds_per_iter_freeze": freeze_total,
        "total_seconds_per_iter_retune": retune_total,
        "total_speedup_retune_vs_freeze": total_speedup,
        "total_seconds_per_iter_freeze_paired": freeze_total_paired,
        "total_seconds_per_iter_retune_paired": retune_total_paired,
        "total_speedup_retune_vs_freeze_paired": total_speedup_paired,
    }

    # Regression distribution for paper plots (retune slower than freeze).
    regressions: list[dict[str, object]] = []
    regressions_unpaired: list[dict[str, object]] = []
    regressions_paired: list[dict[str, object]] = []
    for it in results:
        if it.get("status") != "ok":
            continue
        sp_u = it.get("speedup_retune_vs_freeze_unpaired")
        if isinstance(sp_u, (int, float)) and float(sp_u) > 0 and float(sp_u) < 1.0:
            regressions_unpaired.append({"kernel": it.get("kernel"), "anchor_tier": it.get("anchor_tier"), "speedup": float(sp_u)})
        sp_p = it.get("speedup_retune_vs_freeze_paired")
        if isinstance(sp_p, (int, float)) and float(sp_p) > 0 and float(sp_p) < 1.0:
            regressions_paired.append({"kernel": it.get("kernel"), "anchor_tier": it.get("anchor_tier"), "speedup": float(sp_p)})
        # Prefer paired (stable) if available.
        sp = sp_p if isinstance(sp_p, (int, float)) else it.get("speedup_retune_vs_freeze")
        if isinstance(sp, (int, float)) and float(sp) > 0 and float(sp) < 1.0:
            regressions.append({"kernel": it.get("kernel"), "anchor_tier": it.get("anchor_tier"), "speedup": float(sp)})
    out["summary"]["regressions_n"] = int(len(regressions))
    out["summary"]["regressions"] = regressions
    out["summary"]["regressions_unpaired_n"] = int(len(regressions_unpaired))
    out["summary"]["regressions_unpaired"] = regressions_unpaired
    out["summary"]["regressions_paired_n"] = int(len(regressions_paired))
    out["summary"]["regressions_paired"] = regressions_paired

    # Speedup histogram (stable bins; easy to plot).
    bins = [0.0, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0, float("inf")]
    hist = {f"[{bins[i]},{bins[i+1]})": 0 for i in range(len(bins) - 1)}
    for sp in (speedups_paired or speedups):
        for i in range(len(bins) - 1):
            if bins[i] <= sp < bins[i + 1]:
                hist[f"[{bins[i]},{bins[i+1]})"] += 1
                break
    out["summary"]["speedup_histogram"] = hist
    hist_u = {f"[{bins[i]},{bins[i+1]})": 0 for i in range(len(bins) - 1)}
    for sp in (speedups_unpaired or []):
        for i in range(len(bins) - 1):
            if bins[i] <= sp < bins[i + 1]:
                hist_u[f"[{bins[i]},{bins[i+1]})"] += 1
                break
    out["summary"]["speedup_histogram_unpaired"] = hist_u
    hist_p = {f"[{bins[i]},{bins[i+1]})": 0 for i in range(len(bins) - 1)}
    for sp in (speedups_paired or []):
        for i in range(len(bins) - 1):
            if bins[i] <= sp < bins[i + 1]:
                hist_p[f"[{bins[i]},{bins[i+1]})"] += 1
                break
    out["summary"]["speedup_histogram_paired"] = hist_p

    # Tier breakdown.
    by_tier: dict[str, dict[str, int]] = {}
    for it in results:
        tier = str(it.get("anchor_tier") or "D_none")
        t = by_tier.setdefault(tier, {"n": 0, "ok": 0, "regressions": 0})
        t["n"] += 1
        if it.get("status") == "ok":
            t["ok"] += 1
            sp = it.get("speedup_retune_vs_freeze_paired")
            if not isinstance(sp, (int, float)):
                sp = it.get("speedup_retune_vs_freeze")
            if isinstance(sp, (int, float)) and float(sp) > 0 and float(sp) < 1.0:
                t["regressions"] += 1
    out["summary"]["by_tier"] = by_tier
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
