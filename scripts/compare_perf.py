"""
Compare two RVV benchmark suite JSONs and fail on regressions.

Example:
  python scripts/compare_perf.py --baseline artifacts/rvv_baseline.json --current artifacts/perf_latest.json --threshold 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _index(rows) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows or []:
        fe = str(r.get("frontend") or "")
        k = str(r.get("kernel") or "")
        out[(fe, k)] = dict(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--threshold", type=float, default=0.05, help="allowed slowdown ratio (e.g., 0.05 = +5%)")
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    b = _index(baseline.get("results"))
    c = _index(current.get("results"))

    thr = float(args.threshold)
    regressions = []
    missing = []
    for key, cur in c.items():
        if key not in b:
            missing.append(key)
            continue
        base = b[key]
        ns0 = base.get("ns_per_iter")
        ns1 = cur.get("ns_per_iter")
        if not isinstance(ns0, (int, float)) or not isinstance(ns1, (int, float)) or ns0 <= 0 or ns1 <= 0:
            continue
        ratio = float(ns1) / float(ns0)
        if ratio > 1.0 + thr:
            regressions.append({"frontend": key[0], "kernel": key[1], "baseline_ns": ns0, "current_ns": ns1, "ratio": ratio})

    if missing:
        print(f"WARNING: missing baseline entries for {len(missing)} keys: {missing}")
    if regressions:
        print(f"FAIL: {len(regressions)} regressions (threshold={thr:.3f})")
        for r in regressions:
            fe = r["frontend"]
            k = r["kernel"]
            ratio = r["ratio"]
            print(f"  {fe}:{k}  {r['baseline_ns']:.1f}ns -> {r['current_ns']:.1f}ns  x{ratio:.3f}")
        raise SystemExit(1)

    print("OK: no regressions")


if __name__ == "__main__":
    main()

