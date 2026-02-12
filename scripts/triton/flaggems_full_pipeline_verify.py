"""
FlagGems-backed Triton full pipeline runner (reuses Triton frontend pipeline).

This script does not introduce a new frontend. It only swaps the kernel spec
source/runner to FlagGems implementations while keeping the existing Triton
Task4/Task5 flow unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import run_pipeline_for_spec
from pipeline.triton.flaggems_specs import (
    coverage_flaggems_kernel_specs,
    default_flaggems_kernel_specs,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", default=None, help="Run a single kernel by name (repeatable)")
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke")
    ap.add_argument("--list", action="store_true", help="List available kernels and exit")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM extraction (default: on). Use --no-use-llm to force deterministic fallback intents.",
    )
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "flaggems_triton_full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    suites = {
        "smoke": default_flaggems_kernel_specs,
        "coverage": coverage_flaggems_kernel_specs,
        "all": coverage_flaggems_kernel_specs,
    }
    specs = list(suites[str(args.suite)]())

    if args.list:
        for s in specs:
            print(s.name)
        return

    wanted = set(args.kernel or [])
    for spec in specs:
        if wanted and spec.name not in wanted:
            continue
        print(f"\n=== flaggems:{spec.name} ===")
        try:
            report = run_pipeline_for_spec(
                spec,
                out_dir=out_dir,
                cases_limit=int(args.cases_limit),
                use_llm=bool(args.use_llm),
            )
        except Exception as e:
            print("Pipeline failed:", e)
            continue

        contract_level = (report.get("contract") or {}).get("level")
        diff = report.get("diff") or {}
        print(f"TTIR: {report.get('ttir_path', 'N/A')} | contract={contract_level}")
        print(f"Diff: {'OK' if diff.get('ok') else 'FAIL'}")

        out_path = out_dir / f"{spec.name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Report:", out_path)


if __name__ == "__main__":
    main()
