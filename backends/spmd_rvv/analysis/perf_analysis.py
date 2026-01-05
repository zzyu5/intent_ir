"""
Performance analysis helpers for the RVV backend.

This module is intentionally dependency-light (no numpy/pandas). It consumes the
JSON produced by:
  - scripts/rvv_remote_suite.py  (raw remote compile+run)
  - scripts/benchmark_suite.py   (compact perf JSON)

Primary goal (PROJECT_CRITICAL_GAPS_ANALYSIS_2025.md):
  - connect *predicted* (cost model / tuning debug) with *measured* (remote bench)
  - produce stable, explainable summaries usable in papers and regressions
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _rankdata(xs: Sequence[float]) -> List[float]:
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i + 1
        while j < len(idx) and xs[idx[j]] == xs[idx[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[idx[k]] = avg
        i = j
    return ranks


def spearman_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    denx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    deny = math.sqrt(sum((r - my) ** 2 for r in ry))
    if denx == 0.0 or deny == 0.0:
        return float("nan")
    return num / (denx * deny)


def predicted_gflops_from_tuning_debug(tuning_debug: Dict[str, Any] | None) -> Optional[float]:
    if not isinstance(tuning_debug, dict):
        return None
    kind = str(tuning_debug.get("kind") or "")
    cm = tuning_debug.get("cost_model")
    if not isinstance(cm, dict):
        return None
    if kind == "gemm":
        v = cm.get("achievable_gflops")
        return float(v) if isinstance(v, (int, float)) and float(v) > 0 else None
    if kind == "program":
        tot = cm.get("total")
        if isinstance(tot, dict):
            v = tot.get("gflops")
            return float(v) if isinstance(v, (int, float)) and float(v) > 0 else None
    return None


def predicted_ms_from_tuning_debug(tuning_debug: Dict[str, Any] | None) -> Optional[float]:
    if not isinstance(tuning_debug, dict):
        return None
    kind = str(tuning_debug.get("kind") or "")
    cm = tuning_debug.get("cost_model")
    if not isinstance(cm, dict):
        return None
    if kind == "program":
        tot = cm.get("total")
        if isinstance(tot, dict):
            v = tot.get("ms")
            return float(v) if isinstance(v, (int, float)) and float(v) >= 0 else None
        return None
    if kind == "gemm":
        gflops = cm.get("achievable_gflops")
        mnk = tuning_debug.get("model_mnk")
        if not (isinstance(gflops, (int, float)) and float(gflops) > 0):
            return None
        if not (isinstance(mnk, (list, tuple)) and len(mnk) == 3):
            return None
        try:
            M = int(mnk[0])
            N = int(mnk[1])
            K = int(mnk[2])
        except Exception:
            return None
        if M <= 0 or N <= 0 or K <= 0:
            return None
        flops = 2.0 * float(M) * float(N) * float(K)
        # ms = flops / (GFLOPs * 1e9) * 1e3 = flops / (GFLOPs * 1e6)
        return float(flops / (float(gflops) * 1e6))
    return None


def _profile_top(profile_ops: Dict[str, Any] | None, *, topk: int = 6) -> List[Dict[str, Any]] | None:
    if not isinstance(profile_ops, dict):
        return None
    ops = profile_ops.get("ops")
    if not isinstance(ops, list):
        return None
    rows: List[Dict[str, Any]] = []
    for o in ops:
        if not isinstance(o, dict):
            continue
        name = o.get("name")
        ns = o.get("ns")
        if isinstance(name, str) and isinstance(ns, (int, float)):
            rows.append({"name": name, "ns": float(ns)})
    rows.sort(key=lambda x: x["ns"], reverse=True)
    return rows[: max(0, int(topk))]


def extract_remote_suite_rows(raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Normalize `scripts/rvv_remote_suite.py --out <raw.json>` into row dicts.
    """
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for r in raw.get("results") or []:
        fe = str(r.get("frontend") or "")
        k = str(r.get("kernel") or "")
        ok = bool(r.get("ok"))
        bench = r.get("bench")
        tuning = r.get("tuning") if isinstance(r.get("tuning"), dict) else None
        tdebug = tuning.get("debug") if isinstance(tuning, dict) else None
        prof = r.get("profile_ops")

        if not ok:
            errors.append(f"{fe}:{k} failed (compile/run)")
            rows.append({"frontend": fe, "kernel": k, "ok": False})
            continue
        if not isinstance(bench, dict):
            errors.append(f"{fe}:{k} missing bench (run with --bench-iters > 0)")
            rows.append({"frontend": fe, "kernel": k, "ok": False, "bench": bench})
            continue

        ns = bench.get("ns_per_iter")
        ns_f = float(ns) if isinstance(ns, (int, float)) and float(ns) > 0 else None
        measured_ms = (ns_f / 1e6) if ns_f is not None else None

        pred_ms = predicted_ms_from_tuning_debug(tdebug) if isinstance(tdebug, dict) else None
        pred_gflops = predicted_gflops_from_tuning_debug(tdebug) if isinstance(tdebug, dict) else None
        ratio = (measured_ms / pred_ms) if (measured_ms is not None and pred_ms is not None and pred_ms > 0) else None

        profile_total_ns = None
        if isinstance(prof, dict) and isinstance(prof.get("total_ns"), (int, float)):
            profile_total_ns = float(prof.get("total_ns"))

        rows.append(
            {
                "frontend": fe,
                "kernel": k,
                "ok": True,
                "ns_per_iter": ns_f,
                "measured_ms": measured_ms,
                "matmul_gflops": (float(bench.get("matmul_gflops")) if isinstance(bench.get("matmul_gflops"), (int, float)) else None),
                "pred_ms": pred_ms,
                "pred_gflops": pred_gflops,
                "pred_ratio": ratio,
                "profile_total_ns": profile_total_ns,
                "profile_top_ops": _profile_top(prof, topk=6) if isinstance(prof, dict) else None,
                "tuning": tuning,
                "bench": bench,
            }
        )
    return rows, errors


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize predicted-vs-measured accuracy and provide debug-friendly stats.
    """
    by_fe: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_fe.setdefault(str(r.get("frontend") or ""), []).append(r)

    out: Dict[str, Any] = {"per_frontend": {}, "overall": {}}
    all_pairs: List[Tuple[float, float]] = []

    def _summ(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        pairs = [(float(r["pred_ms"]), float(r["measured_ms"])) for r in rs if isinstance(r.get("pred_ms"), (int, float)) and isinstance(r.get("measured_ms"), (int, float)) and float(r["pred_ms"]) > 0 and float(r["measured_ms"]) > 0]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        for p in pairs:
            all_pairs.append(p)

        ratios = [float(r.get("pred_ratio")) for r in rs if isinstance(r.get("pred_ratio"), (int, float)) and float(r.get("pred_ratio")) > 0]
        ratios.sort()
        worst = sorted(
            [
                {
                    "kernel": str(r.get("kernel") or ""),
                    "pred_ms": float(r["pred_ms"]),
                    "measured_ms": float(r["measured_ms"]),
                    "ratio": float(r["pred_ratio"]),
                }
                for r in rs
                if isinstance(r.get("pred_ratio"), (int, float)) and isinstance(r.get("pred_ms"), (int, float)) and isinstance(r.get("measured_ms"), (int, float))
            ],
            key=lambda x: x["ratio"],
            reverse=True,
        )[:5]

        return {
            "count": int(len(rs)),
            "count_with_pred": int(len(pairs)),
            "spearman_pred_ms_vs_measured_ms": float(spearman_r(xs, ys)) if len(pairs) >= 2 else None,
            "median_ratio": (ratios[len(ratios) // 2] if ratios else None),
            "worst": worst,
        }

    for fe, rs in by_fe.items():
        out["per_frontend"][fe] = _summ(rs)

    # Overall summary across frontends.
    if len(all_pairs) >= 2:
        xs = [p[0] for p in all_pairs]
        ys = [p[1] for p in all_pairs]
        out["overall"]["spearman_pred_ms_vs_measured_ms"] = float(spearman_r(xs, ys))
    else:
        out["overall"]["spearman_pred_ms_vs_measured_ms"] = None

    return out


__all__ = [
    "spearman_r",
    "predicted_gflops_from_tuning_debug",
    "predicted_ms_from_tuning_debug",
    "extract_remote_suite_rows",
    "summarize_rows",
]

