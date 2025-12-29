"""
Unified full-pipeline runner for multiple frontends.

Examples:
  PYTHONPATH=. python scripts/full_pipeline_verify.py --frontend triton --kernel softmax_inner
  PYTHONPATH=. python scripts/full_pipeline_verify.py --frontend tilelang --kernel upsample_bicubic2d_aa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang"], default="triton")
    ap.add_argument("--kernel", action="append", default=None, help="Run a single kernel by name (repeatable)")
    ap.add_argument("--list", action="store_true", help="List available kernels and exit")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    wanted = set(args.kernel or [])

    if args.frontend == "triton":
        from pipeline.triton.core import default_kernel_specs, run_pipeline_for_spec

        out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "full_pipeline_verify")
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.list:
            for s in default_kernel_specs():
                print(s.name)
            return
        for spec in default_kernel_specs():
            if wanted and spec.name not in wanted:
                continue
            print(f"\n=== {spec.name} ===")
            report = run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=int(args.cases_limit))
            out_path = out_dir / f"{spec.name}.json"
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            diff_ok = bool((report.get("diff") or {}).get("ok"))
            contract_level = (report.get("contract") or {}).get("level")
            print(f"Report: {out_path} | contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")
        return

    # tilelang
    from pipeline.tilelang.core import default_kernel_specs, run_pipeline_for_spec

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "tilelang_full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = list(default_kernel_specs())
    if args.list:
        for s in specs:
            print(s.name)
        return
    for spec in specs:
        if wanted and spec.name not in wanted:
            continue
        print(f"\n=== {spec.name} ===")
        report = run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=int(args.cases_limit))
        out_path = out_dir / f"{spec.name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        diff_ok = bool((report.get("diff") or {}).get("ok"))
        contract_level = (report.get("contract") or {}).get("level")
        print(f"Report: {out_path} | contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
