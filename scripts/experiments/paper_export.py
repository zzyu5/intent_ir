"""
Paper JSON exporter.

This script does **not** rerun expensive pipelines. It:
  - reads the latest E1–E5 experiment artifacts under `artifacts/experiments/`
  - post-processes them into *plot-friendly* JSONs (stable keys, small payload)
  - writes an index JSON that points to per-figure files

Artifacts are gitignored; this is intended for local paper iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        return out if out else None
    except Exception:
        return None


def _latest_file(dir_path: Path, pattern: str) -> Path | None:
    cand = [p for p in dir_path.glob(pattern) if p.is_file()]
    if not cand:
        return None
    cand.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return cand[-1]


def _geom_mean(xs: list[float]) -> float | None:
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / float(len(xs)))


def _paired_speedup_from_measured_autotune(retune_remote: dict[str, Any] | None) -> float | None:
    """
    Extract a paired (stable) freeze-vs-retune speedup from a single measured_autotune run.

    We avoid importing `scripts/experiments/portability_vs_perf.py` as a module,
    because `scripts/` is not a Python package.
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
    return float(b_ns) / float(best_ns)


@dataclass(frozen=True)
class PaperPaths:
    out_dir: Path
    e1: Path
    e1e3: Path
    e2: Path
    e4: Path
    e5_1: Path
    e5_2: Path
    index: Path


def _default_paths(out_dir: Path) -> PaperPaths:
    return PaperPaths(
        out_dir=out_dir,
        e1=out_dir / "e1_rule_only.paper.json",
        e1e3=out_dir / "e1e3_llm_regression.paper.json",
        e2=out_dir / "e2_trust_ablation.paper.json",
        e4=out_dir / "e4_cross_frontend_consistency.paper.json",
        e5_1=out_dir / "e5_1_external_baseline.paper.json",
        e5_2=out_dir / "e5_2_portability_vs_perf.paper.json",
        index=out_dir / "paper_index.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "experiments" / "paper"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    paths = _default_paths(out_dir)

    exp_dir = ROOT / "artifacts" / "experiments"
    e1_dir = exp_dir / "E1"
    e1e3_dir = exp_dir / "E1E3"
    e2_dir = exp_dir / "E2"
    e4_dir = exp_dir / "E4"
    e5_dir = exp_dir / "E5"

    head = _git_head()

    # E1: rule-only coverage.
    e1_src = _latest_file(e1_dir, "e1_rule_only_*.json") or (e1_dir / "e1_rule_only_coverage.json")
    e1_obj = _load_json(e1_src) if e1_src.exists() else {}
    _write_json(
        paths.e1,
        {
            "experiment": "E1_rule_only",
            "git_head": head,
            "source": str(e1_src) if e1_src.exists() else None,
            "summary": e1_obj.get("summary"),
        },
    )

    # E1E3: LLM regression suite (one-shot vs feedback).
    e1e3_src = _latest_file(e1e3_dir, "e1e3_llm_regression_all_coverage_cold*.json") or _latest_file(
        e1e3_dir, "e1e3_llm_regression_all_coverage*.json"
    )
    e1e3_obj = _load_json(e1e3_src) if (e1e3_src and e1e3_src.exists()) else {}
    _write_json(
        paths.e1e3,
        {
            "experiment": "E1E3_llm_regression",
            "git_head": head,
            "source": str(e1e3_src) if e1e3_src else None,
            "suite": e1e3_obj.get("suite"),
            "frontends": e1e3_obj.get("frontends"),
            "delta": e1e3_obj.get("delta"),
            "label_eval": e1e3_obj.get("label_eval"),
            "summary": {
                v: (e1e3_obj.get("variants", {}).get(v, {}).get("summary") if isinstance(e1e3_obj.get("variants"), dict) else None)
                for v in ["one_shot", "feedback"]
            },
        },
    )

    # E2: trust ablation (mutation-kill harness).
    e2_srcs = {
        "triton": _latest_file(e2_dir, "e2_trust_ablation_triton_coverage.json"),
        "tilelang": _latest_file(e2_dir, "e2_trust_ablation_tilelang_coverage.json"),
        "cuda": _latest_file(e2_dir, "e2_trust_ablation_cuda_coverage.json"),
    }
    e2_by_frontend: dict[str, Any] = {}
    for fe, p in e2_srcs.items():
        if p is None or not p.exists():
            continue
        o = _load_json(p)
        e2_by_frontend[fe] = {
            "source": str(p),
            "summary": o.get("summary"),
        }
    _write_json(
        paths.e2,
        {
            "experiment": "E2_trust_ablation",
            "git_head": head,
            "by_frontend": e2_by_frontend,
        },
    )

    # E4: cross-frontend consistency.
    e4_src = _latest_file(e4_dir, "e4_triton_tilelang_artifact_intersection_axisroles_v*.json") or _latest_file(
        e4_dir, "e4_*artifact_intersection*.json"
    )
    e4_obj = _load_json(e4_src) if (e4_src and e4_src.exists()) else {}
    _write_json(
        paths.e4,
        {
            "experiment": "E4_cross_frontend_consistency",
            "git_head": head,
            "source": str(e4_src) if e4_src else None,
            "frontends": e4_obj.get("frontends"),
            "suite": e4_obj.get("suite"),
            "kernels": e4_obj.get("kernels"),
            "summary": e4_obj.get("summary"),
        },
    )

    # E5.1: external baseline comparison.
    e5_1_src = _latest_file(e5_dir, "e5_1_external_ai_benchmark_baseline16_ours16*.json")
    e5_1_obj = _load_json(e5_1_src) if (e5_1_src and e5_1_src.exists()) else {}
    _write_json(
        paths.e5_1,
        {
            "experiment": "E5_1_external_baseline",
            "git_head": head,
            "source": str(e5_1_src) if e5_1_src else None,
            "summary": e5_1_obj.get("summary"),
            "kernels": e5_1_obj.get("kernels"),
        },
    )

    # E5.2: portability vs performance (freeze vs retune).
    e5_2_src = _latest_file(e5_dir, "e5_2_triton_coverage_freeze_vs_retune*.json") or _latest_file(
        e5_dir, "e5_2_default6_freeze_vs_retune*.json"
    )
    e5_2_obj = _load_json(e5_2_src) if (e5_2_src and e5_2_src.exists()) else {}

    # Post-process paired/unpaired from stored remote tuning.measured_autotune.
    paired_speedups: list[float] = []
    unpaired_speedups: list[float] = []
    per_kernel: list[dict[str, Any]] = []
    for it in list(e5_2_obj.get("kernels") or []):
        if not isinstance(it, dict) or it.get("status") != "ok":
            continue
        k = str(it.get("kernel"))
        tier = str(it.get("anchor_tier") or "D_none")
        fs = (it.get("freeze") or {}).get("seconds_per_iter")
        rs = (it.get("retune") or {}).get("seconds_per_iter")
        sp_u = None
        if isinstance(fs, (int, float)) and isinstance(rs, (int, float)) and float(fs) > 0 and float(rs) > 0:
            sp_u = float(fs) / float(rs)
            unpaired_speedups.append(float(sp_u))

        sp_p = _paired_speedup_from_measured_autotune(((it.get("retune") or {}).get("remote")))
        if isinstance(sp_p, float) and sp_p > 0:
            paired_speedups.append(float(sp_p))

        per_kernel.append(
            {
                "kernel": k,
                "anchor_tier": tier,
                "speedup_unpaired": sp_u,
                "speedup_paired": sp_p,
            }
        )

    _write_json(
        paths.e5_2,
        {
            "experiment": "E5_2_portability_vs_perf",
            "git_head": head,
            "source": str(e5_2_src) if e5_2_src else None,
            "bench": {
                "bench_iters": e5_2_obj.get("bench_iters"),
                "bench_warmup": e5_2_obj.get("bench_warmup"),
                "bench_seed": e5_2_obj.get("bench_seed"),
                "omp_threads": e5_2_obj.get("omp_threads"),
                "omp_proc_bind": e5_2_obj.get("omp_proc_bind"),
                "tune_budget": e5_2_obj.get("tune_budget"),
                "tune_mode": e5_2_obj.get("tune_mode"),
            },
            "summary": {
                "n": int(len(per_kernel)),
                "geom_speedup_unpaired": _geom_mean(unpaired_speedups),
                "geom_speedup_paired": _geom_mean(paired_speedups),
                "regressions_unpaired_n": int(sum(1 for x in unpaired_speedups if x < 1.0)),
                "regressions_paired_n": int(sum(1 for x in paired_speedups if x < 1.0)),
            },
            "per_kernel": per_kernel,
        },
    )

    _write_json(
        paths.index,
        {
            "git_head": head,
            "out_dir": str(out_dir),
            "figures": {
                "E1": str(paths.e1),
                "E1E3": str(paths.e1e3),
                "E2": str(paths.e2),
                "E4": str(paths.e4),
                "E5_1": str(paths.e5_1),
                "E5_2": str(paths.e5_2),
            },
        },
    )

    print(str(paths.index))


if __name__ == "__main__":
    main()
