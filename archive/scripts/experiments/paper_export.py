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
import re
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


def _best_e4_consistency(e4_dir: Path) -> Path | None:
    """
    Prefer the *most complete* E4 artifact for paper exports.

    The E4 directory can contain many debug/subset re-runs; picking the newest
    file by mtime is brittle and can accidentally export a partial run where
    some pairs are missing Intent/cert.
    """
    patterns = [
        "e4_triton_tilelang_artifact_intersection_axisroles_v*.json",
        "e4_*artifact_intersection*.json",
        "e4_cross_frontend_consistency*.json",
    ]
    cand: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for p in e4_dir.glob(pat):
            if not p.is_file():
                continue
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            cand.append(p)
    if not cand:
        return None

    best: Path | None = None
    best_score: tuple[int, int, int, float] | None = None
    for p in cand:
        try:
            obj = _load_json(p)
        except Exception:
            continue
        if str(obj.get("experiment") or "") != "E4_cross_frontend_consistency":
            continue
        s = obj.get("summary")
        if not isinstance(s, dict):
            continue
        n = s.get("n")
        if not isinstance(n, int) or n <= 0:
            n = len(list(obj.get("results") or []))

        struct_ok = s.get("intent_structural_ok")
        if not isinstance(struct_ok, int):
            rate = s.get("intent_structural_ok_rate")
            if isinstance(rate, (int, float)) and n > 0:
                struct_ok = int(round(float(rate) * float(n)))
            else:
                struct_ok = 0

        portable_ok = s.get("intent_ok")
        if not isinstance(portable_ok, int):
            rate = s.get("intent_ok_rate")
            if isinstance(rate, (int, float)) and n > 0:
                portable_ok = int(round(float(rate) * float(n)))
            else:
                portable_ok = 0

        score = (int(struct_ok), int(portable_ok), int(n), float(p.stat().st_mtime))
        if best is None or best_score is None or score > best_score:
            best = p
            best_score = score

    return best or _latest_file(e4_dir, patterns[0]) or _latest_file(e4_dir, patterns[1])


def _augment_e4_summary(e4_obj: dict[str, Any]) -> dict[str, Any]:
    """
    Add a paper-friendly "normalization ladder" view to E4.

    We compute exact-match rates under progressively more semantic normalizations:
      raw -> drop schedule hints -> portable canonical form -> structural hash
    and attribute drift to the earliest step that resolves it.

    This is derived from the per-pair `results[*].per_frontend` payload that is
    already present in E4 artifacts.
    """

    s0 = e4_obj.get("summary")
    s: dict[str, Any] = dict(s0) if isinstance(s0, dict) else {}

    frontends = e4_obj.get("frontends")
    fes = [str(x) for x in (frontends or []) if str(x).strip()] if isinstance(frontends, list) else []
    if len(fes) < 2:
        return s

    base_fe = str(fes[0])
    other_fes = [str(x) for x in fes[1:]]

    def _drop_parallel_axes(sig: dict[str, Any]) -> dict[str, Any]:
        out = dict(sig)
        out.pop("parallel_axes", None)
        return out

    def _as_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float)):
            return int(v) != 0
        return False

    def _drift_bucket(*, contract_oos: bool, ok_structural: bool, ok_raw: bool, ok_no_schedule: bool, ok_portable: bool) -> str:
        if contract_oos:
            return "facts_missing_or_out_of_scope"
        if not ok_structural:
            return "semantic_or_interface_mismatch"
        if ok_raw:
            return "raw_exact"
        if ok_no_schedule:
            return "schedule_hint_drift"
        if ok_portable:
            return "specialization_view_drift"
        return "representation_drift"

    results = e4_obj.get("results")
    rows = [r for r in list(results or []) if isinstance(r, dict)]
    if not rows:
        return s

    raw_ok = 0
    nosched_ok = 0
    portable_ok = 0
    structural_ok = 0
    drift_counts: dict[str, int] = {}
    by_tier_aug: dict[str, dict[str, Any]] = {}
    by_kind_aug: dict[str, dict[str, Any]] = {}

    for r in rows:
        per = r.get("per_frontend")
        if not isinstance(per, dict):
            continue

        tier = str(r.get("anchor_tier") or "D_none")
        kind = str(r.get("kernel_kind") or "unknown")

        ok = r.get("ok")
        ok_struct = _as_bool(ok.get("intent_structural") if isinstance(ok, dict) else None)

        base = per.get(base_fe) if isinstance(per.get(base_fe), dict) else {}
        base_sig = base.get("sig_intent")
        base_strict = base.get("sig_intent_strict")

        ok_raw = isinstance(base_sig, dict)
        ok_no_schedule = isinstance(base_sig, dict)
        ok_portable = isinstance(base_strict, dict)

        for fe in other_fes:
            other = per.get(fe) if isinstance(per.get(fe), dict) else {}
            sig = other.get("sig_intent")
            sig_strict = other.get("sig_intent_strict")
            if not isinstance(sig, dict) or not isinstance(base_sig, dict):
                ok_raw = False
                ok_no_schedule = False
            else:
                if sig != base_sig:
                    ok_raw = False
                if _drop_parallel_axes(sig) != _drop_parallel_axes(base_sig):
                    ok_no_schedule = False
            if not (isinstance(sig_strict, dict) and isinstance(base_strict, dict) and sig_strict == base_strict):
                ok_portable = False

        contract_oos = any(
            str((per.get(fe) or {}).get("contract")) == "OUT_OF_SCOPE"
            for fe in fes
            if isinstance(per.get(fe), dict)
        )
        bucket = _drift_bucket(
            contract_oos=bool(contract_oos),
            ok_structural=bool(ok_struct),
            ok_raw=bool(ok_raw),
            ok_no_schedule=bool(ok_no_schedule),
            ok_portable=bool(ok_portable),
        )

        raw_ok += int(ok_raw)
        nosched_ok += int(ok_no_schedule)
        portable_ok += int(ok_portable)
        structural_ok += int(ok_struct)
        drift_counts[bucket] = drift_counts.get(bucket, 0) + 1

        bt = by_tier_aug.setdefault(
            tier,
            {
                "n": 0,
                "ok_intent_raw": 0,
                "ok_intent_no_schedule": 0,
                "ok_intent": 0,
                "ok_intent_structural": 0,
                "drift_breakdown": {},
            },
        )
        bt["n"] = int(bt.get("n", 0)) + 1
        bt["ok_intent_raw"] = int(bt.get("ok_intent_raw", 0)) + int(ok_raw)
        bt["ok_intent_no_schedule"] = int(bt.get("ok_intent_no_schedule", 0)) + int(ok_no_schedule)
        bt["ok_intent"] = int(bt.get("ok_intent", 0)) + int(ok_portable)
        bt["ok_intent_structural"] = int(bt.get("ok_intent_structural", 0)) + int(ok_struct)
        dbt = bt.setdefault("drift_breakdown", {})
        if isinstance(dbt, dict):
            dbt[bucket] = int(dbt.get(bucket, 0)) + 1

        bk = by_kind_aug.setdefault(
            kind,
            {
                "n": 0,
                "ok_intent_raw": 0,
                "ok_intent_no_schedule": 0,
                "ok_intent": 0,
                "ok_intent_structural": 0,
                "drift_breakdown": {},
            },
        )
        bk["n"] = int(bk.get("n", 0)) + 1
        bk["ok_intent_raw"] = int(bk.get("ok_intent_raw", 0)) + int(ok_raw)
        bk["ok_intent_no_schedule"] = int(bk.get("ok_intent_no_schedule", 0)) + int(ok_no_schedule)
        bk["ok_intent"] = int(bk.get("ok_intent", 0)) + int(ok_portable)
        bk["ok_intent_structural"] = int(bk.get("ok_intent_structural", 0)) + int(ok_struct)
        dbk = bk.setdefault("drift_breakdown", {})
        if isinstance(dbk, dict):
            dbk[bucket] = int(dbk.get(bucket, 0)) + 1

    n = len(rows)
    s["n"] = int(s.get("n")) if isinstance(s.get("n"), int) else int(n)
    s["intent_raw_ok"] = int(raw_ok)
    s["intent_raw_ok_rate"] = (float(raw_ok) / float(n)) if n > 0 else 0.0
    s["intent_no_schedule_ok"] = int(nosched_ok)
    s["intent_no_schedule_ok_rate"] = (float(nosched_ok) / float(n)) if n > 0 else 0.0
    s["intent_ok"] = int(portable_ok)
    s["intent_ok_rate"] = (float(portable_ok) / float(n)) if n > 0 else 0.0
    s["intent_structural_ok"] = int(structural_ok)
    s["intent_structural_ok_rate"] = (float(structural_ok) / float(n)) if n > 0 else 0.0
    s["drift_breakdown"] = {k: int(v) for k, v in sorted(drift_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))}

    # Merge augmented per-tier/per-kind stats into the original summary if present.
    by_tier_out: dict[str, Any] = dict(s.get("by_tier") or {}) if isinstance(s.get("by_tier"), dict) else {}
    for tier, aug in by_tier_aug.items():
        base = by_tier_out.get(tier)
        merged = dict(base) if isinstance(base, dict) else {}
        merged.update(aug)
        by_tier_out[str(tier)] = merged
    s["by_tier"] = by_tier_out

    by_kind_out: dict[str, Any] = dict(s.get("by_kind") or {}) if isinstance(s.get("by_kind"), dict) else {}
    for kind, aug in by_kind_aug.items():
        base = by_kind_out.get(kind)
        merged = dict(base) if isinstance(base, dict) else {}
        merged.update(aug)
        by_kind_out[str(kind)] = merged
    s["by_kind"] = by_kind_out

    return s


def _geom_mean(xs: list[float]) -> float | None:
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and float(x) > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / float(len(xs)))


def _parse_ai_bench_report_t1_t16(path: Path) -> dict[str, dict[str, float]]:
    """
    Parse AI-Benchmark's markdown summary and extract per-kernel total times for 1T/16T.

    We intentionally parse the "性能排行榜（按 16 线程加速比）" table, which contains one
    row per kernel with columns "1线程时间" and "16线程时间".
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        # Example:
        # | 🥇 1 | **matmul** | **24.97x** | 2.149s | 0.086s | 最佳扩展性 |
        m = re.match(
            r"^\|\s*[^|]*\|\s*\*\*(?P<name>[A-Za-z0-9_]+)\*\*\s*\|\s*\*\*?[0-9]*\.?[0-9]+x\*\*?\s*\|\s*(?P<t1>[0-9]*\.?[0-9]+)s\s*\|\s*(?P<t16>[0-9]*\.?[0-9]+)s\s*\|",
            line.strip(),
        )
        if not m:
            continue
        name = str(m.group("name"))
        try:
            t1 = float(m.group("t1"))
            t16 = float(m.group("t16"))
        except Exception:
            continue
        out[name] = {"t1_total_s": t1, "t16_total_s": t16}
    return out


def _extract_intentir_bench_seconds_per_iter(remote: Any) -> float | None:
    """
    Extract seconds-per-iter from an rvv_remote_run bench-only stdout blob:
      INTENTIR_BENCH {"ns_per_iter": ...}
    """
    if not isinstance(remote, dict):
        return None
    stdout = remote.get("stdout")
    if not isinstance(stdout, str) or not stdout:
        return None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("INTENTIR_BENCH "):
            continue
        payload = line[len("INTENTIR_BENCH ") :].strip()
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        ns = obj.get("ns_per_iter")
        if isinstance(ns, (int, float)) and float(ns) > 0:
            return float(ns) / 1e9
    return None


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
    e4_src = _best_e4_consistency(e4_dir) or _latest_file(e4_dir, "e4_triton_tilelang_artifact_intersection_axisroles_v*.json") or _latest_file(
        e4_dir, "e4_*artifact_intersection*.json"
    )
    e4_obj = _load_json(e4_src) if (e4_src and e4_src.exists()) else {}
    e4_summary = _augment_e4_summary(e4_obj) if isinstance(e4_obj, dict) else {}
    _write_json(
        paths.e4,
        {
            "experiment": "E4_cross_frontend_consistency",
            "git_head": head,
            "source": str(e4_src) if e4_src else None,
            "frontends": e4_obj.get("frontends"),
            "suite": e4_obj.get("suite"),
            "kernels": e4_obj.get("kernels"),
            "summary": e4_summary,
        },
    )

    # E5.1: external baseline comparison (AI-Benchmark, 8 kernels).
    #
    # Paper needs both single-thread and multi-thread views. We therefore prefer
    # the "experiment_a_*" artifacts, which contain our remote bench-only runs
    # for omp_threads=1 and omp_threads=16.
    e5_1_mt_src = _latest_file(e5_dir, "experiment_a_baseline16_ours16_v*.json") or _latest_file(e5_dir, "experiment_a_baseline16_ours16.json")
    e5_1_st_src = _latest_file(e5_dir, "experiment_a_baseline16_ours1_v*.json") or _latest_file(e5_dir, "experiment_a_baseline16_ours1.json")
    # Fallback for older runs (only 16T).
    e5_1_old_src = _latest_file(e5_dir, "e5_1_external_ai_benchmark_baseline16_ours16*.json")

    e5_1_mt_obj = _load_json(e5_1_mt_src) if (e5_1_mt_src and e5_1_mt_src.exists()) else {}
    e5_1_st_obj = _load_json(e5_1_st_src) if (e5_1_st_src and e5_1_st_src.exists()) else {}
    e5_1_old_obj = _load_json(e5_1_old_src) if (e5_1_old_src and e5_1_old_src.exists()) else {}

    # Resolve baseline report (to extract both 1T and 16T totals).
    baseline_source = None
    for obj in [e5_1_mt_obj, e5_1_st_obj, e5_1_old_obj]:
        b = obj.get("baseline")
        if isinstance(b, dict) and isinstance(b.get("source"), str) and b.get("source"):
            baseline_source = Path(str(b.get("source")))
            break
    if baseline_source is None:
        baseline_source = ROOT / "experiment" / "AI-Benchmark" / "COMPLETE_PERFORMANCE_SUMMARY.md"
    ai_bench_times = _parse_ai_bench_report_t1_t16(baseline_source) if baseline_source.exists() else {}

    def _ours_s_per_iter_by_name(obj: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for it in list(obj.get("kernels") or []):
            if not isinstance(it, dict):
                continue
            name = str(it.get("baseline_name") or it.get("kernel") or "")
            ours = it.get("ours")
            remote = (ours or {}).get("remote") if isinstance(ours, dict) else None
            s = _extract_intentir_bench_seconds_per_iter(remote)
            if isinstance(s, float) and s > 0:
                out[name] = float(s)
        return out

    def _baseline_run_count_by_name(obj: dict[str, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in list(obj.get("kernels") or []):
            if not isinstance(it, dict):
                continue
            name = str(it.get("baseline_name") or it.get("kernel") or "")
            b = it.get("baseline")
            rc = (b or {}).get("run_count") if isinstance(b, dict) else None
            if isinstance(rc, int) and rc > 0:
                out[name] = int(rc)
        return out

    ours_t1 = _ours_s_per_iter_by_name(e5_1_st_obj)
    ours_t16 = _ours_s_per_iter_by_name(e5_1_mt_obj or e5_1_old_obj)
    run_counts = _baseline_run_count_by_name(e5_1_mt_obj or e5_1_old_obj or e5_1_st_obj)

    per_kernel: list[dict[str, Any]] = []
    speedups_t1: list[float] = []
    speedups_t16: list[float] = []
    slower_t1 = 0
    slower_t16 = 0

    names = sorted(set(run_counts.keys()) | set(ours_t1.keys()) | set(ours_t16.keys()))
    for name in names:
        rc = run_counts.get(name)
        bt = ai_bench_times.get(name, {}) if isinstance(ai_bench_times, dict) else {}
        b1_total = bt.get("t1_total_s")
        b16_total = bt.get("t16_total_s")

        b1 = (float(b1_total) / float(rc)) if isinstance(b1_total, (int, float)) and isinstance(rc, int) and rc > 0 else None
        b16 = (float(b16_total) / float(rc)) if isinstance(b16_total, (int, float)) and isinstance(rc, int) and rc > 0 else None

        o1 = ours_t1.get(name)
        o16 = ours_t16.get(name)

        sp1 = (float(b1) / float(o1)) if isinstance(b1, (int, float)) and isinstance(o1, (int, float)) and o1 > 0 else None
        sp16 = (float(b16) / float(o16)) if isinstance(b16, (int, float)) and isinstance(o16, (int, float)) and o16 > 0 else None

        if isinstance(sp1, float) and sp1 > 0:
            speedups_t1.append(float(sp1))
            if sp1 < 1.0:
                slower_t1 += 1
        if isinstance(sp16, float) and sp16 > 0:
            speedups_t16.append(float(sp16))
            if sp16 < 1.0:
                slower_t16 += 1

        per_kernel.append(
            {
                "name": name,
                "run_count": rc,
                "baseline_seconds_per_iter_t1": b1,
                "baseline_seconds_per_iter_t16": b16,
                "ours_seconds_per_iter_t1": float(o1) if isinstance(o1, (int, float)) else None,
                "ours_seconds_per_iter_t16": float(o16) if isinstance(o16, (int, float)) else None,
                "speedup_ours_over_baseline_t1": sp1,
                "speedup_ours_over_baseline_t16": sp16,
            }
        )

    _write_json(
        paths.e5_1,
        {
            "experiment": "E5_1_external_baseline",
            "git_head": head,
            "sources": {
                "ours_t1": str(e5_1_st_src) if e5_1_st_src else None,
                "ours_t16": str(e5_1_mt_src) if e5_1_mt_src else (str(e5_1_old_src) if e5_1_old_src else None),
                "baseline_report": str(baseline_source) if baseline_source else None,
            },
            "summary": {
                "n": int(len(per_kernel)),
                "geom_speedup_ours_over_baseline_t1": _geom_mean(speedups_t1),
                "geom_speedup_ours_over_baseline_t16": _geom_mean(speedups_t16),
                "slower_count_t1": int(slower_t1),
                "slower_count_t16": int(slower_t16),
            },
            "per_kernel": per_kernel,
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
