"""
FlagGems-backed Triton full pipeline runner (reuses Triton frontend pipeline).

This script is FlagGems-specific and intentionally owns FlagGems execution
controls:
- --flaggems-path original|intentir
- --intentir-mode auto|force_compile|force_cache

`original` means "run native FlagGems path without IntentIR".
`intentir` means "run through IntentIR path".
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
from pipeline.triton.flaggems_execution import (
    sync_seed_back_to_cache,
    sync_seed_into_run_dir,
    resolve_flaggems_execution,
)
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
        "--flaggems-path",
        choices=["original", "intentir"],
        default="intentir",
        help="Execution path: original FlagGems or IntentIR-integrated path (default: intentir).",
    )
    ap.add_argument(
        "--intentir-mode",
        choices=["auto", "force_compile", "force_cache"],
        default="auto",
        help="IntentIR mode (only valid for --flaggems-path=intentir): auto, force_compile, force_cache.",
    )
    ap.add_argument(
        "--seed-cache-dir",
        type=Path,
        default=(ROOT / "artifacts" / "flaggems_seed_cache"),
        help="Shared seed cache directory for intentir mode.",
    )
    ap.add_argument(
        "--fallback-policy",
        choices=["deterministic", "strict"],
        default="deterministic",
        help="IntentIR fallback policy when cache/LLM paths fail (default: deterministic).",
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

    seed_cache_dir = Path(args.seed_cache_dir)
    seed_cache_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_flaggems_execution(
        flaggems_path=str(args.flaggems_path),
        intentir_mode=str(args.intentir_mode),
        seed_cache_dir=seed_cache_dir,
        fallback_policy=str(args.fallback_policy),
    )

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "flaggems_triton_full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    suites = {
        "smoke": lambda: default_flaggems_kernel_specs(
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        ),
        "coverage": lambda: coverage_flaggems_kernel_specs(
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        ),
        "all": lambda: coverage_flaggems_kernel_specs(
            flaggems_opset=str(args.flaggems_opset),
            backend_target=str(args.backend_target),
        ),
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
        if config.use_intent_ir:
            cache_event = sync_seed_into_run_dir(
                spec_name=str(spec.name),
                seed_cache_dir=seed_cache_dir,
                run_out_dir=out_dir,
                intentir_mode=config.intentir_mode,
            )
            if cache_event == "hit":
                print(f"[{spec.name}] intent seed cache hit: {seed_cache_dir / f'{spec.name}.intent_seed.json'}")
            elif cache_event == "miss":
                print(f"[{spec.name}] intent seed cache miss: {seed_cache_dir / f'{spec.name}.intent_seed.json'}")
        else:
            # Ensure original path does not accidentally consume stale per-run seed.
            run_seed = out_dir / f"{spec.name}.intent_seed.json"
            if run_seed.is_file():
                try:
                    run_seed.unlink()
                except OSError:
                    pass
        try:
            report = run_pipeline_for_spec(
                spec,
                out_dir=out_dir,
                cases_limit=int(args.cases_limit),
                execution_policy=config.execution_policy,
                triton_provider="flaggems",
                backend_target=str(args.backend_target),
            )
        except Exception as e:
            print("Pipeline failed:", e)
            continue
        if config.use_intent_ir:
            wrote = sync_seed_back_to_cache(
                spec_name=str(spec.name),
                seed_cache_dir=seed_cache_dir,
                run_out_dir=out_dir,
                intentir_mode=config.intentir_mode,
            )
            if wrote:
                print(f"[{spec.name}] intent seed cached: {seed_cache_dir / f'{spec.name}.intent_seed.json'}")

        contract_level = (report.get("contract") or {}).get("level")
        diff = report.get("diff") or {}
        print(f"TTIR: {report.get('ttir_path', 'N/A')} | contract={contract_level}")
        print(f"Diff: {'OK' if diff.get('ok') else 'FAIL'}")

        out_path = out_dir / f"{spec.name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Report:", out_path)
    print(f"Generated pipeline artifacts: {out_dir}")
    if config.use_intent_ir:
        print(f"Intent seed cache dir: {seed_cache_dir}")


if __name__ == "__main__":
    main()
