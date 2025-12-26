"""
TileLang frontend full pipeline runner (PR#9 MVP).

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

from pipeline.tilelang.core import default_kernel_specs, mvp_kernel_specs, run_pipeline_for_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["default", "mvp", "all"], default="default")
    ap.add_argument("--kernel", type=str, default=None, help="Run a single kernel by name")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--no-stage-c", action="store_true")
    ap.add_argument("--no-mutation-kill", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "artifacts" / "tilelang_full_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    suite: list = []
    if args.suite in {"mvp", "all"}:
        suite.extend(mvp_kernel_specs())
    if args.suite in {"default", "all"}:
        suite.extend(default_kernel_specs())
    if args.kernel:
        suite = [s for s in suite if s.name == args.kernel]
        if not suite:
            raise SystemExit(f"unknown kernel: {args.kernel}")

    for spec in suite:
        print(f"\n=== {spec.name} ===")
        report = run_pipeline_for_spec(
            spec,
            out_dir=out_dir,
            cases_limit=int(args.cases_limit),
            stage_c=not bool(args.no_stage_c),
            mutation_kill=not bool(args.no_mutation_kill),
        )
        out_path = out_dir / f"{spec.name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Report:", out_path)
        diff_ok = bool((report.get("diff") or {}).get("ok"))
        contract_level = (report.get("contract") or {}).get("level")
        print(f"contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
