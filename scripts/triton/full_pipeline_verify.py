"""
Triton frontend full pipeline runner (Tasks 1–5).

This file lives under `scripts/triton/` to keep Triton-specific entrypoints
separate from generic scripts (backend smoke, remote RVV run, etc).

Examples:
  PYTHONPATH=. python scripts/triton/full_pipeline_verify.py --suite smoke
  PYTHONPATH=. python scripts/triton/full_pipeline_verify.py --provider flaggems --suite all --no-use-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--provider",
        choices=["native", "flaggems"],
        default="native",
        help="Kernel source/runner provider (default: native)",
    )
    ap.add_argument("--kernel", action="append", default=None, help="Run a single kernel by name (repeatable)")
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke")
    ap.add_argument("--list", action="store_true", help="List available kernels and exit")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument(
        "--use-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM extraction (default: on). Use --no-use-llm to replay cached intent seeds.",
    )
    ap.add_argument(
        "--allow-deterministic-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --no-use-llm and cache is missing, allow legacy deterministic fallback intents.",
    )
    ap.add_argument(
        "--flaggems-opset",
        choices=["deterministic_forward"],
        default="deterministic_forward",
        help="FlagGems semantic-op set to use (default: deterministic_forward).",
    )
    ap.add_argument(
        "--backend-target",
        choices=["rvv", "cuda_h100", "cuda_5090d"],
        default="rvv",
        help="Backend preflight target for IntentIR capability checks (default: rvv).",
    )
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    from pipeline.triton.core import run_pipeline_for_spec

    provider = str(args.provider)
    if provider == "flaggems":
        from pipeline.triton.flaggems_specs import coverage_flaggems_kernel_specs, default_flaggems_kernel_specs

        def coverage_kernel_specs():
            return coverage_flaggems_kernel_specs(
                flaggems_opset=str(args.flaggems_opset),
                backend_target=str(args.backend_target),
            )

        def default_kernel_specs():
            return default_flaggems_kernel_specs(
                flaggems_opset=str(args.flaggems_opset),
                backend_target=str(args.backend_target),
            )

        default_out_dir = ROOT / "artifacts" / "flaggems_triton_full_pipeline"
    else:
        from pipeline.triton.core import coverage_kernel_specs, default_kernel_specs

        default_out_dir = ROOT / "artifacts" / "full_pipeline_verify"

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suites = {
        "smoke": default_kernel_specs,
        "coverage": coverage_kernel_specs,
        "all": coverage_kernel_specs,
    }
    if args.list:
        for s in suites[str(args.suite)]():
            print(s.name)
        return
    wanted = set(args.kernel or [])

    if wanted:
        specs = [s for s in coverage_kernel_specs() if s.name in wanted]
    else:
        specs = list(suites[str(args.suite)]())

    for spec in specs:
        if wanted and spec.name not in wanted:
            continue
        prefix = "flaggems:" if provider == "flaggems" else ""
        print(f"\n=== {prefix}{spec.name} ===")
        try:
            report = run_pipeline_for_spec(
                spec,
                out_dir=out_dir,
                cases_limit=int(args.cases_limit),
                use_llm=bool(args.use_llm),
                allow_deterministic_fallback=bool(args.allow_deterministic_fallback),
                triton_provider=str(provider),
                backend_target=str(args.backend_target),
            )
        except Exception as e:
            print("Pipeline failed:", e)
            continue

        # Compact terminal summary
        contract_level = report.get("contract", {}).get("level")
        diff = report.get("diff", {})
        stage_c = report.get("stage_c", {})
        print(f"TTIR: {report.get('ttir_path', 'N/A')} | contract={contract_level}")
        if "certificate" in report and report["certificate"] and report["certificate"].get("obligations"):
            ob = report["certificate"]["obligations"]
            short = [f"{o['id']}:{o['status']}" for o in ob][:6]
            print("Certificate obligations:", ", ".join(short))
        print(f"Diff: {'OK' if diff.get('ok') else 'FAIL'}")
        if stage_c:
            meta_ok = stage_c.get("metamorphic", {}).get("ok", True)
            bounded = stage_c.get("bounded_exhaustive", {})
            bounded_state = "SKIP" if bounded.get("total", 0) == 0 else ("OK" if bounded.get("ok") else "FAIL")
            print(f"StageC metamorphic={meta_ok} bounded={bounded_state}")

        # Persist report JSON
        try:
            out_path = out_dir / f"{spec.name}.json"
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Report:", out_path)
        except Exception as e:
            print("Failed to write report:", e)


if __name__ == "__main__":
    main()
