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
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke", help="Kernel suite (default: smoke)")
    ap.add_argument("--list", action="store_true", help="List available kernels and exit")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    wanted = set(args.kernel or [])
    from pipeline.run import process_batch

    if args.frontend == "triton":
        from pipeline.triton.core import coverage_kernel_specs, default_kernel_specs, run_pipeline_for_spec

        out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "full_pipeline_verify")
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.list:
            suites = {"smoke": default_kernel_specs, "coverage": coverage_kernel_specs, "all": coverage_kernel_specs}
            for s in suites[str(args.suite)]():
                print(s.name)
            return
        suites = {
            "smoke": default_kernel_specs,
            # Coverage currently includes smoke + extra kernels (see pipeline/*/core.py).
            "coverage": coverage_kernel_specs,
            "all": coverage_kernel_specs,
        }
        specs = [s for s in suites[str(args.suite)]() if (not wanted or s.name in wanted)]

        def _write(name: str, payload: object) -> Path:
            out_path = out_dir / f"{name}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return out_path

        def run_one(spec) -> dict:
            print(f"\n=== {spec.name} ===")
            report = run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=int(args.cases_limit))
            out_path = _write(spec.name, report)
            diff_ok = bool((report.get("diff") or {}).get("ok"))
            contract_level = (report.get("contract") or {}).get("level")
            print(f"Report: {out_path} | contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")
            return report

        def on_error(spec, err: dict) -> None:
            out_path = _write(spec.name, err)
            et = ((err.get("error") or {}).get("type") or "Exception")
            em = ((err.get("error") or {}).get("message") or "")
            print(f"Pipeline failed: {et}: {em}")
            print(f"Report: {out_path} | contract=N/A diff=FAIL")

        process_batch(specs, run_one, name_fn=lambda s: s.name, on_error=on_error)
        return

    # tilelang
    from pipeline.tilelang.core import coverage_kernel_specs, default_kernel_specs, run_pipeline_for_spec

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "tilelang_full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = {
        "smoke": default_kernel_specs,
        "coverage": coverage_kernel_specs,
        "all": coverage_kernel_specs,
    }
    specs = list(suites[str(args.suite)]())
    if args.list:
        for s in suites[str(args.suite)]():
            print(s.name)
        return
    specs = [s for s in specs if (not wanted or s.name in wanted)]

    def _write(name: str, payload: object) -> Path:
        out_path = out_dir / f"{name}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def run_one(spec) -> dict:
        print(f"\n=== {spec.name} ===")
        report = run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=int(args.cases_limit))
        out_path = _write(spec.name, report)
        diff_ok = bool((report.get("diff") or {}).get("ok"))
        contract_level = (report.get("contract") or {}).get("level")
        print(f"Report: {out_path} | contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")
        return report

    def on_error(spec, err: dict) -> None:
        out_path = _write(spec.name, err)
        et = ((err.get("error") or {}).get("type") or "Exception")
        em = ((err.get("error") or {}).get("message") or "")
        print(f"Pipeline failed: {et}: {em}")
        print(f"Report: {out_path} | contract=N/A diff=FAIL")

    process_batch(specs, run_one, name_fn=lambda s: s.name, on_error=on_error)


if __name__ == "__main__":
    main()
