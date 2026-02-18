"""
TileLang frontend full pipeline runner (Tasks 1–5).

This is intentionally thin; the orchestration lives in `pipeline/tilelang/core.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tilelang.core import default_kernel_specs, run_pipeline_for_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", default=None, help="Run a single kernel by name (repeatable)")
    ap.add_argument("--list", action="store_true", help="List available kernels and exit")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "tilelang_full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        for s in default_kernel_specs():
            print(s.name)
        return
    wanted = set(args.kernel or [])

    for spec in default_kernel_specs():
        if wanted and spec.name not in wanted:
            continue
        print(f"\n=== {spec.name} ===")
        try:
            report = run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=int(args.cases_limit))
        except Exception as e:
            print("Pipeline failed:", e)
            continue
        try:
            out_path = out_dir / f"{spec.name}.json"
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Report:", out_path)
        except Exception as e:
            print("Failed to write report:", e)
        diff_ok = bool((report.get("diff") or {}).get("ok"))
        contract_level = (report.get("contract") or {}).get("level")
        print(f"contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
