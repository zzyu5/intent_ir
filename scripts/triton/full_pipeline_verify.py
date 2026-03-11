"""
Triton frontend full pipeline runner (Tasks 1–5).

This is intentionally thin; the orchestration lives in `pipeline/triton/core.py`.

Note: This runner is for the native Triton coverage suite (currently 38 kernels).
FlagGems provider runs are handled by `scripts/triton/flaggems_full_pipeline_verify.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import coverage_kernel_specs, default_kernel_specs, run_pipeline_for_spec
from pipeline.triton.execution_policy import make_execution_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", default=None, help="Run a single kernel by name (repeatable)")
    ap.add_argument("--suite", choices=["smoke", "coverage"], default="smoke")
    ap.add_argument("--list", action="store_true", help="List available kernels and exit")
    ap.add_argument("--cases-limit", type=int, default=8)
    ap.add_argument("--backend-target", choices=["rvv", "cuda_h100", "cuda_5090d"], default=None)
    ap.add_argument(
        "--stage-c",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Stage C verification (metamorphic/bounded/numerical stability).",
    )
    ap.add_argument(
        "--stage-c-max-cases",
        type=int,
        default=None,
        help="Max cases for bounded exhaustive Stage C (None uses kernel spec default).",
    )
    ap.add_argument(
        "--mutation-kill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mutation-kill verification (very expensive).",
    )
    ap.add_argument(
        "--mutation-bounded-max-cases",
        type=int,
        default=None,
        help="Max bounded cases inside mutation-kill (None uses stage-c-max-cases).",
    )
    ap.add_argument(
        "--intentir-mode",
        choices=["auto", "force_compile", "force_cache"],
        default=str(os.getenv("INTENTIR_MODE", "auto")).strip().lower(),
        help="IntentIR seed policy: auto(cache->LLM), force_compile(always LLM), force_cache(never LLM).",
    )
    ap.add_argument(
        "--intentir-miss-policy",
        choices=["deterministic", "strict"],
        default=str(os.getenv("INTENTIR_FALLBACK_POLICY", "deterministic")).strip().lower(),
        help="When seed cache is missing: deterministic allows fallback; strict fails fast.",
    )
    ap.add_argument(
        "--seed-cache-dir",
        type=str,
        default=str(os.getenv("INTENTIR_TRITON_SEED_CACHE_DIR", "artifacts/triton_seed_cache")).strip(),
        help="Optional shared seed cache directory. Set to 'none' to disable.",
    )
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "triton_full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    suites = {
        "smoke": default_kernel_specs,
        "coverage": coverage_kernel_specs,
    }
    specs = list(suites[str(args.suite)]())

    if args.list:
        for s in specs:
            print(s.name)
        return

    seed_cache_dir = str(args.seed_cache_dir or "").strip()
    if seed_cache_dir.lower() in {"", "none", "off", "0"}:
        seed_cache_dir_path = None
    else:
        seed_cache_dir_path = Path(seed_cache_dir)

    policy = make_execution_policy(
        path="intentir",
        intentir_mode=str(args.intentir_mode),
        seed_cache_dir=seed_cache_dir_path,
        fallback_policy=str(args.intentir_miss_policy),
    )

    wanted_list = [str(x) for x in list(args.kernel or []) if str(x).strip()]
    if wanted_list:
        # User explicitly requested kernels; allow selecting from the full coverage
        # universe even when --suite=smoke (smoke is a small subset and does not
        # include many kernels like flash_attention2d).
        all_specs = list(coverage_kernel_specs())
        by_name = {str(s.name): s for s in all_specs}
        missing = [k for k in wanted_list if k not in by_name]
        if missing:
            print(f"Unknown kernel(s): {missing}", file=sys.stderr, flush=True)
            raise SystemExit(2)
        run_specs = [by_name[k] for k in wanted_list]
    else:
        run_specs = list(specs)

    for spec in run_specs:
        print(f"\n=== {spec.name} ===")
        try:
            report = run_pipeline_for_spec(
                spec,
                out_dir=out_dir,
                cases_limit=int(args.cases_limit),
                backend_target=(str(args.backend_target) if args.backend_target else None),
                enable_stage_c=bool(args.stage_c),
                stage_c_max_cases=args.stage_c_max_cases,
                enable_mutation_kill=bool(args.mutation_kill),
                mutation_bounded_max_cases=args.mutation_bounded_max_cases,
                execution_policy=policy,
            )
        except Exception as e:
            print("Pipeline failed:", e)
            continue
        out_path = out_dir / f"{spec.name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        diff_ok = bool((report.get("diff") or {}).get("ok"))
        contract_level = (report.get("contract") or {}).get("level")
        print(f"Report: {out_path} | contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
