"""
Analyze RVV benchmark outputs (predicted vs measured).

This consumes the JSON produced by `scripts/benchmark_suite.py` and emits a
compact summary JSON to stdout (or --out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.spmd_rvv.analysis.perf_analysis import summarize_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", required=True, help="benchmark_suite JSON path")
    ap.add_argument("--out", default=None, help="write summary JSON to this path (default: stdout)")
    args = ap.parse_args()

    perf = json.loads(Path(args.perf).read_text(encoding="utf-8"))
    rows = perf.get("results") or []
    summary = summarize_rows(rows)
    out: Dict[str, Any] = {
        "kind": "rvv_perf_analysis",
        "input": str(args.perf),
        "config": perf.get("config"),
        "remote": perf.get("remote"),
        "summary": summary,
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
