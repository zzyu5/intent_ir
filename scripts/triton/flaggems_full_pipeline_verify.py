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
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import run_pipeline_for_spec
from pipeline.triton.providers.flaggems.execution import (
    sync_seed_back_to_cache,
    sync_seed_into_run_dir,
    resolve_flaggems_execution,
)


def _append_progress_row(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def default_flaggems_kernel_specs(*, flaggems_opset: str, backend_target: str):
    from pipeline.triton.providers.flaggems.specs import default_flaggems_kernel_specs as _impl  # noqa: PLC0415

    return _impl(flaggems_opset=str(flaggems_opset), backend_target=str(backend_target))


def coverage_flaggems_kernel_specs(*, flaggems_opset: str, backend_target: str):
    from pipeline.triton.providers.flaggems.specs import coverage_flaggems_kernel_specs as _impl  # noqa: PLC0415

    return _impl(flaggems_opset=str(flaggems_opset), backend_target=str(backend_target))


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
        "--intentir-miss-policy",
        choices=["deterministic", "strict"],
        default="deterministic",
        help="IntentIR miss policy when cache/LLM paths fail (default: deterministic).",
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
        "--strict-kernel-failure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when any kernel raises pipeline exception after writing failure report.",
    )
    ap.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Optional per-kernel JSONL progress log path (default: <out-dir>/kernel_progress.jsonl).",
    )
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()
    miss_policy = str(args.intentir_miss_policy)

    seed_cache_dir = Path(args.seed_cache_dir)
    seed_cache_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_flaggems_execution(
        flaggems_path=str(args.flaggems_path),
        intentir_mode=str(args.intentir_mode),
        seed_cache_dir=seed_cache_dir,
        fallback_policy=miss_policy,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / "flaggems_triton_full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_log = Path(args.progress_log) if args.progress_log is not None else (out_dir / "kernel_progress.jsonl")
    if progress_log.is_file():
        try:
            progress_log.unlink()
        except OSError:
            pass

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
    selected_specs = [spec for spec in specs if (not wanted or spec.name in wanted)]
    kernel_failures: list[str] = []
    total = len(selected_specs)
    _append_progress_row(
        progress_log,
        {
            "event": "session_start",
            "suite": str(args.suite),
            "total_kernels": int(total),
            "flaggems_path": str(args.flaggems_path),
            "intentir_mode": str(args.intentir_mode),
            "intentir_miss_policy": miss_policy,
            "out_dir": str(out_dir),
        },
    )
    for idx, spec in enumerate(selected_specs, start=1):
        start_ts = time.time()
        print(f"\n[{idx}/{total}] START flaggems:{spec.name}", flush=True)
        _append_progress_row(
            progress_log,
            {
                "event": "kernel_start",
                "index": int(idx),
                "total": int(total),
                "kernel": str(spec.name),
            },
        )
        if config.use_intent_ir:
            cache_event = sync_seed_into_run_dir(
                spec_name=str(spec.name),
                seed_cache_dir=seed_cache_dir,
                run_out_dir=out_dir,
                intentir_mode=config.intentir_mode,
            )
            if cache_event == "hit":
                print(f"[{spec.name}] intent seed cache hit: {seed_cache_dir / f'{spec.name}.intent_seed.json'}", flush=True)
            elif cache_event == "miss":
                print(f"[{spec.name}] intent seed cache miss: {seed_cache_dir / f'{spec.name}.intent_seed.json'}", flush=True)
            elif cache_event == "synthesized":
                print(f"[{spec.name}] intent seed synthesized from provider canonical template: {seed_cache_dir / f'{spec.name}.intent_seed.json'}", flush=True)
        else:
            # Ensure original path does not accidentally consume stale per-run seed.
            run_seed = out_dir / f"{spec.name}.intent_seed.json"
            if run_seed.is_file():
                try:
                    run_seed.unlink()
                except OSError:
                    pass
        out_path = out_dir / f"{spec.name}.json"
        report: dict[str, Any]
        try:
            report = run_pipeline_for_spec(
                spec,
                out_dir=out_dir,
                cases_limit=int(args.cases_limit),
                execution_policy=config.execution_policy,
                triton_provider="flaggems",
                backend_target=str(args.backend_target),
                enable_stage_c=bool(args.stage_c),
                stage_c_max_cases=args.stage_c_max_cases,
                enable_mutation_kill=bool(args.mutation_kill),
                mutation_bounded_max_cases=args.mutation_bounded_max_cases,
            )
        except Exception as e:
            elapsed = time.time() - start_ts
            print("Pipeline failed:", e, flush=True)
            kernel_failures.append(str(spec.name))
            report = {
                "kernel": str(spec.name),
                "triton_provider": "flaggems",
                "backend_target": str(args.backend_target),
                "execution": {
                    "flaggems_path": str(args.flaggems_path),
                    "intentir_mode": str(args.intentir_mode),
                    "intentir_miss_policy": miss_policy,
                },
                "diff": {
                    "ok": False,
                    "reason_code": "pipeline_exception",
                    "reason_detail": f"{type(e).__name__}: {e}",
                },
                "contract": {
                    "ok": False,
                    "level": "pipeline_failed",
                },
                "reason_code": "pipeline_exception",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _append_progress_row(
                progress_log,
                {
                    "event": "kernel_end",
                    "index": int(idx),
                    "total": int(total),
                    "kernel": str(spec.name),
                    "ok": False,
                    "status": "pipeline_exception",
                    "reason_code": "pipeline_exception",
                    "elapsed_sec": float(round(elapsed, 6)),
                    "report_path": str(out_path),
                },
            )
            print(
                f"[{idx}/{total}] DONE flaggems:{spec.name} status=PIPELINE_EXCEPTION "
                f"elapsed={elapsed:.2f}s report={out_path}",
                flush=True,
            )
            continue

        if config.use_intent_ir:
            wrote = sync_seed_back_to_cache(
                spec_name=str(spec.name),
                seed_cache_dir=seed_cache_dir,
                run_out_dir=out_dir,
                intentir_mode=config.intentir_mode,
            )
            if wrote:
                print(f"[{spec.name}] intent seed cached: {seed_cache_dir / f'{spec.name}.intent_seed.json'}", flush=True)

        contract_level = (report.get("contract") or {}).get("level")
        diff = report.get("diff") or {}
        kernel_ok = bool(diff.get("ok"))
        elapsed = time.time() - start_ts
        reason_code = str(diff.get("reason_code") or report.get("reason_code") or "")
        if not kernel_ok:
            kernel_failures.append(str(spec.name))
        print(f"TTIR: {report.get('ttir_path', 'N/A')} | contract={contract_level}", flush=True)
        print(f"Diff: {'OK' if kernel_ok else 'FAIL'}", flush=True)

        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _append_progress_row(
            progress_log,
            {
                "event": "kernel_end",
                "index": int(idx),
                "total": int(total),
                "kernel": str(spec.name),
                "ok": bool(kernel_ok),
                "status": ("ok" if kernel_ok else "diff_fail"),
                "reason_code": reason_code,
                "elapsed_sec": float(round(elapsed, 6)),
                "report_path": str(out_path),
            },
        )
        print(
            f"[{idx}/{total}] DONE flaggems:{spec.name} status={'OK' if kernel_ok else 'DIFF_FAIL'} "
            f"elapsed={elapsed:.2f}s report={out_path}",
            flush=True,
        )
    _append_progress_row(
        progress_log,
        {
            "event": "session_end",
            "total_kernels": int(total),
            "failed_kernels": list(kernel_failures),
            "failed_count": int(len(kernel_failures)),
            "ok": bool(len(kernel_failures) == 0),
        },
    )
    print(f"Generated pipeline artifacts: {out_dir}", flush=True)
    print(f"Kernel progress log: {progress_log}", flush=True)
    if config.use_intent_ir:
        print(f"Intent seed cache dir: {seed_cache_dir}", flush=True)
    if kernel_failures:
        print(f"Kernel pipeline failures: {', '.join(kernel_failures)}", flush=True)
        if bool(args.strict_kernel_failure):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
