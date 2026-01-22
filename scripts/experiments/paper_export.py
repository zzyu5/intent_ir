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


def _latest_e6_2_coverage(e6_dir: Path) -> Path | None:
    """
    Prefer a *full* E6.2 coverage run for paper exports.

    E6 directories often contain many debug/smoke/subset runs; picking by mtime
    alone is brittle and can accidentally export a partial run as the paper
    result.
    """
    cand = [p for p in e6_dir.glob("e6_2_contract_calibration*.json") if p.is_file()]
    if not cand:
        return None
    cand.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    best: Path | None = None
    best_n = -1
    for p in cand:
        try:
            obj = _load_json(p)
        except Exception:
            continue
        if str(obj.get("experiment") or "") != "E6_2_contract_calibration":
            continue
        if str(obj.get("suite") or "") != "coverage":
            continue
        n = len(list(obj.get("results") or []))
        # Full CUDA coverage with (full,no_mask,no_anchors) and (intentir,linalg)
        # is typically 32 * 3 * 2 = 192 samples. Use a loose threshold to avoid
        # exporting tiny subset/debug runs.
        if n < 150:
            continue
        # Prefer the largest run; break ties by recency (cand is sorted by mtime).
        if n > best_n or (n == best_n):
            best = p
            best_n = n
    return best or cand[-1]


def _geom_mean(xs: list[float]) -> float | None:
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / float(len(xs)))


def _e6cc_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize a list of E6.2 contract-calibration samples.

    Keeps the payload small and plot-friendly (counts + rates + level dists).
    """
    n = int(len(rows))
    ok = int(sum(1 for r in rows if bool(r.get("ok"))))
    over = int(sum(1 for r in rows if bool(r.get("overclaim"))))
    under = int(sum(1 for r in rows if bool(r.get("underclaim"))))
    bind_ok = int(sum(1 for r in rows if bool(r.get("binding_ok"))))
    full = int(sum(1 for r in rows if str(r.get("contract_level")) == "FULL"))
    partial = int(sum(1 for r in rows if str(r.get("contract_level")) == "PARTIAL"))
    oos = int(sum(1 for r in rows if str(r.get("contract_level")) == "OUT_OF_SCOPE"))
    full_false = int(sum(1 for r in rows if str(r.get("contract_level")) == "FULL" and not bool(r.get("ok"))))

    oracle_levels: dict[str, int] = {}
    contract_levels: dict[str, int] = {}
    for r in rows:
        ol = r.get("oracle_level")
        cl = r.get("contract_level")
        if isinstance(ol, str) and ol:
            oracle_levels[ol] = int(oracle_levels.get(ol, 0)) + 1
        if isinstance(cl, str) and cl:
            contract_levels[cl] = int(contract_levels.get(cl, 0)) + 1

    def _rate(x: int, denom: int) -> float | None:
        return (float(x) / float(denom)) if denom > 0 else None

    return {
        "n": n,
        "ok": ok,
        "ok_rate": _rate(ok, n),
        "overclaim": over,
        "overclaim_rate": _rate(over, n),
        "underclaim": under,
        "underclaim_rate": _rate(under, n),
        "binding_ok": bind_ok,
        "binding_ok_rate": _rate(bind_ok, n),
        "full_claims": full,
        "partial_claims": partial,
        "oos_claims": oos,
        "full_false_accept": full_false,
        "full_false_accept_rate": _rate(full_false, full),
        "oracle_levels": oracle_levels,
        "contract_levels": contract_levels,
    }


def _summarize_e6_2_contract_calibration(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Paper-facing summary for E6.2.

    Key idea: ablation effects are easiest to interpret when conditioned on the
    oracle level (e.g., no_mask => oracle=PARTIAL for in-scope kernels, but some
    kernels may still be oracle=OUT_OF_SCOPE due to barrier/atomic anchors).
    """
    results = [r for r in list(obj.get("results") or []) if isinstance(r, dict)]
    reps = sorted({str(r.get("rep")) for r in results if isinstance(r.get("rep"), str)})
    ablations = sorted({str(r.get("ablation")) for r in results if isinstance(r.get("ablation"), str)})
    oracle_levels = sorted({str(r.get("oracle_level")) for r in results if isinstance(r.get("oracle_level"), str)})

    by_rep: dict[str, Any] = {}
    by_rep_ablation: dict[str, Any] = {}
    by_rep_ablation_oracle: dict[str, Any] = {}

    for rep in reps:
        rows = [r for r in results if r.get("rep") == rep]
        by_rep[rep] = _e6cc_metrics(rows)
        by_rep_ablation[rep] = {}
        by_rep_ablation_oracle[rep] = {}
        for ab in ablations:
            rows_ab = [r for r in rows if r.get("ablation") == ab]
            by_rep_ablation[rep][ab] = _e6cc_metrics(rows_ab)
            by_rep_ablation_oracle[rep].setdefault(ab, {})
            for ol in oracle_levels:
                rows_ol = [r for r in rows_ab if r.get("oracle_level") == ol]
                if rows_ol:
                    by_rep_ablation_oracle[rep][ab][ol] = _e6cc_metrics(rows_ol)

    # Convenience: punchline slice used in the paper narrative (no_mask + oracle=PARTIAL).
    punchline: dict[str, Any] = {"ablation": "no_mask", "oracle_level": "PARTIAL", "by_rep": {}}
    for rep in reps:
        rows = [r for r in results if r.get("rep") == rep and r.get("ablation") == "no_mask" and r.get("oracle_level") == "PARTIAL"]
        if rows:
            punchline["by_rep"][rep] = _e6cc_metrics(rows)

    # Identify kernels that are oracle OUT_OF_SCOPE (anchor says barrier/atomic).
    # Use rep=linalg to avoid IntentIR parse noise affecting this metadata.
    oos_kernels: list[str] = []
    for r in results:
        if r.get("rep") != "linalg":
            continue
        if r.get("oracle_level") == "OUT_OF_SCOPE":
            k = r.get("kernel")
            if isinstance(k, str) and k not in oos_kernels:
                oos_kernels.append(k)
    oos_kernels.sort()

    return {
        "objective": "E6.2 evidence ablation + contract calibration (fair IR+contract for both reps)",
        "reps": reps,
        "ablations": ablations,
        "oracle_levels": oracle_levels,
        "by_rep": by_rep,
        "by_rep_ablation": by_rep_ablation,
        "by_rep_ablation_oracle": by_rep_ablation_oracle,
        "punchline": punchline,
        "oracle_out_of_scope_kernels": oos_kernels,
        "note": "For interpretation: focus on punchline (no_mask & oracle=PARTIAL). OUT_OF_SCOPE kernels are scope gaps (barrier/atomic) and can be reported separately.",
    }


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


def _get_e1e3_variant_summary(obj: dict[str, Any], variant: str) -> dict[str, Any] | None:
    variants = obj.get("variants")
    if not isinstance(variants, dict):
        return None
    v = variants.get(str(variant))
    if not isinstance(v, dict):
        return None
    s = v.get("summary")
    return s if isinstance(s, dict) else None


def _delta_by_frontend(one_shot: dict[str, Any] | None, feedback: dict[str, Any] | None) -> dict[str, Any]:
    """
    Compute OK-rate gains (feedback - one_shot) per-frontend from summary blobs.
    """
    out: dict[str, Any] = {"by_frontend": {}}
    os_bf = (one_shot or {}).get("by_frontend")
    fb_bf = (feedback or {}).get("by_frontend")
    if not isinstance(os_bf, dict) or not isinstance(fb_bf, dict):
        return out

    for fe in sorted(set(os_bf.keys()) | set(fb_bf.keys())):
        os_s = os_bf.get(fe)
        fb_s = fb_bf.get(fe)
        os_ok = os_s.get("ok") if isinstance(os_s, dict) else None
        os_n = os_s.get("n") if isinstance(os_s, dict) else None
        fb_ok = fb_s.get("ok") if isinstance(fb_s, dict) else None
        fb_n = fb_s.get("n") if isinstance(fb_s, dict) else None
        os_rate = os_s.get("ok_rate") if isinstance(os_s, dict) else None
        fb_rate = fb_s.get("ok_rate") if isinstance(fb_s, dict) else None

        gain = None
        if isinstance(os_rate, (int, float)) and isinstance(fb_rate, (int, float)):
            gain = float(fb_rate) - float(os_rate)

        out["by_frontend"][str(fe)] = {
            "one_shot": {"ok": os_ok, "n": os_n, "ok_rate": os_rate},
            "feedback": {"ok": fb_ok, "n": fb_n, "ok_rate": fb_rate},
            "ok_rate_gain": gain,
        }
    return out


@dataclass(frozen=True)
class PaperPaths:
    out_dir: Path
    e1: Path
    e1e3: Path
    e2: Path
    e4: Path
    e5_1: Path
    e5_2: Path
    e6: Path
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
        e6=out_dir / "e6_ir_usability.paper.json",
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
    e6_dir = exp_dir / "E6"

    head = _git_head()

    # E1: rule-only coverage.
    e1_src = _latest_file(e1_dir, "e1_rule_only_*.json") or (e1_dir / "e1_rule_only_coverage.json")
    e1_obj = _load_json(e1_src) if e1_src.exists() else {}
    labels_path = ROOT / "data" / "human_labels.json"
    labels = _load_json(labels_path) if labels_path.exists() else {}
    e1_sem_acc: dict[str, dict[str, Any]] = {}
    for r in list(e1_obj.get("results") or []):
        if not isinstance(r, dict):
            continue
        k = r.get("kernel")
        if not isinstance(k, str) or ":" not in k:
            continue
        fe, name = k.split(":", 1)
        fe = str(fe)
        name = str(name)
        lab = labels.get(name) if isinstance(labels, dict) else None
        if not isinstance(lab, dict):
            continue
        true_cls = lab.get("semantic_class")
        pred_cls = r.get("kernel_kind")
        if not isinstance(true_cls, str) or not isinstance(pred_cls, str):
            continue
        st = e1_sem_acc.setdefault(fe, {"n": 0, "ok": 0, "confusion": {}})
        st["n"] += 1
        if str(true_cls) == str(pred_cls):
            st["ok"] += 1
        st["confusion"].setdefault(str(true_cls), {})
        st["confusion"][str(true_cls)][str(pred_cls)] = int(st["confusion"][str(true_cls)].get(str(pred_cls), 0)) + 1
    for fe, st in e1_sem_acc.items():
        n = int(st.get("n") or 0)
        ok = int(st.get("ok") or 0)
        st["acc"] = (float(ok) / float(n)) if n > 0 else None
    _write_json(
        paths.e1,
        {
            "experiment": "E1_rule_only",
            "git_head": head,
            "source": str(e1_src) if e1_src.exists() else None,
            "summary": e1_obj.get("summary"),
            "label_eval": {
                "labels": str(labels_path) if labels_path.exists() else None,
                "semantic_class_acc_by_frontend": e1_sem_acc,
            },
        },
    )

    # E1E3: LLM regression suite (one-shot vs feedback).
    # Base: newest all-frontends cold run (has label_eval + both variants).
    e1e3_base = _latest_file(e1e3_dir, "e1e3_llm_regression_all_coverage_cold*.json") or _latest_file(
        e1e3_dir, "e1e3_llm_regression_all_coverage*.json"
    )
    e1e3_base_obj = _load_json(e1e3_base) if (e1e3_base and e1e3_base.exists()) else {}

    # Newer per-frontend cold follow-ups (e.g., CUDA barrier/O6 fixes) may exist.
    # We override only the affected frontend+variant summaries, but we keep the
    # all-frontends cold run as the base for label_eval and metadata.
    e1e3_cuda_cold = _latest_file(e1e3_dir, "e1e3_llm_regression_cuda_coverage_cold*.json")
    e1e3_cuda_obj = _load_json(e1e3_cuda_cold) if (e1e3_cuda_cold and e1e3_cuda_cold.exists()) else {}

    e1e3_one_shot = _get_e1e3_variant_summary(e1e3_base_obj, "one_shot")
    e1e3_feedback = _get_e1e3_variant_summary(e1e3_base_obj, "feedback")

    sources: dict[str, Any] = {"base": (str(e1e3_base) if e1e3_base else None), "overrides": {}}
    cuda_fb = _get_e1e3_variant_summary(e1e3_cuda_obj, "feedback")
    if isinstance(cuda_fb, dict) and isinstance(cuda_fb.get("by_frontend"), dict) and "cuda" in cuda_fb.get("by_frontend", {}):
        if not isinstance(e1e3_feedback, dict):
            e1e3_feedback = {"by_frontend": {}}
        e1e3_feedback.setdefault("by_frontend", {})
        if isinstance(e1e3_feedback.get("by_frontend"), dict):
            e1e3_feedback["by_frontend"]["cuda"] = cuda_fb["by_frontend"]["cuda"]
            sources["overrides"].setdefault("feedback", {})
            sources["overrides"]["feedback"]["cuda"] = str(e1e3_cuda_cold)

    _write_json(
        paths.e1e3,
        {
            "experiment": "E1E3_llm_regression",
            "git_head": head,
            "source": sources,
            "suite": e1e3_base_obj.get("suite"),
            "frontends": e1e3_base_obj.get("frontends"),
            "delta": _delta_by_frontend(e1e3_one_shot, e1e3_feedback),
            "label_eval": e1e3_base_obj.get("label_eval"),
            "summary": {
                "one_shot": e1e3_one_shot,
                "feedback": e1e3_feedback,
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
    e5_1_speedups: list[float] = []
    e5_1_min = None
    e5_1_max = None
    e5_1_slower = 0
    for it in list(e5_1_obj.get("kernels") or []):
        if not isinstance(it, dict):
            continue
        sp = it.get("speedup_ours_over_baseline")
        if not isinstance(sp, (int, float)) or float(sp) <= 0:
            continue
        sp = float(sp)
        e5_1_speedups.append(sp)
        if e5_1_min is None or sp < e5_1_min[1]:
            e5_1_min = (str(it.get("baseline_name") or it.get("kernel") or ""), sp)
        if e5_1_max is None or sp > e5_1_max[1]:
            e5_1_max = (str(it.get("baseline_name") or it.get("kernel") or ""), sp)
        if sp < 1.0:
            e5_1_slower += 1
    _write_json(
        paths.e5_1,
        {
            "experiment": "E5_1_external_baseline",
            "git_head": head,
            "source": str(e5_1_src) if e5_1_src else None,
            "baseline": e5_1_obj.get("baseline"),
            "ours": e5_1_obj.get("ours"),
            "summary": {
                "n": int(len(e5_1_speedups)),
                "geom_speedup_ours_over_baseline": _geom_mean(e5_1_speedups),
                "slower_count": int(e5_1_slower),
                "min": e5_1_min,
                "max": e5_1_max,
            },
            "per_kernel": [
                {
                    "name": str(it.get("baseline_name") or it.get("kernel") or ""),
                    "speedup_ours_over_baseline": it.get("speedup_ours_over_baseline"),
                }
                for it in list(e5_1_obj.get("kernels") or [])
                if isinstance(it, dict)
            ],
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

    # E6: IR suitability for LLM lifting under uncertainty.
    # Prefer the newer E6.2 contract-calibration experiment (fairer than E6.1).
    e6_src = _latest_e6_2_coverage(e6_dir) or _latest_file(e6_dir, "e6_ir_usability*.json")
    e6_obj = _load_json(e6_src) if (e6_src and e6_src.exists()) else {}
    e6_summary = e6_obj.get("summary")
    if str(e6_obj.get("experiment") or "").startswith("E6_2_contract_calibration"):
        # Make E6.2 plot-friendly by adding ablation/oracle slices and a punchline summary.
        e6_summary = {
            "source_summary": e6_obj.get("summary"),
            "paper_summary": _summarize_e6_2_contract_calibration(e6_obj),
        }
    _write_json(
        paths.e6,
        {
            "experiment": str(e6_obj.get("experiment") or "E6"),
            "git_head": head,
            "source": str(e6_src) if e6_src else None,
            "suite": e6_obj.get("suite"),
            "frontends": e6_obj.get("frontends"),
            "reps": e6_obj.get("reps"),
            "ablations": e6_obj.get("ablations"),
            "model": e6_obj.get("model"),
            "cache": e6_obj.get("cache"),
            "repair_rounds": e6_obj.get("repair_rounds"),
            "summary": e6_summary,
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
                "E6": str(paths.e6),
            },
        },
    )

    print(str(paths.index))


if __name__ == "__main__":
    main()
