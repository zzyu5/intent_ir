"""
Aggregate CI-style gates for FlagGems long-running workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.providers.flaggems.workflow import load_json, validate_feature_list_sync  # noqa: E402


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": str(name), "ok": bool(ok), "detail": str(detail)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--scripts-catalog", type=Path, default=(ROOT / "scripts" / "CATALOG.json"))
    ap.add_argument("--feature-list", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument(
        "--active-batch-coverage",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_coverage.json"),
    )
    ap.add_argument(
        "--active-batch",
        type=Path,
        default=None,
        help="Compatibility alias for --active-batch-coverage.",
    )
    ap.add_argument(
        "--active-batch-ir-arch",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_ir_arch.json"),
    )
    ap.add_argument(
        "--active-batch-backend-compiler",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_backend_compiler.json"),
    )
    ap.add_argument(
        "--profiles",
        action="append",
        default=[],
        help="Profiles to evaluate (repeatable or comma-separated). Default: coverage,ir_arch,backend_compiler",
    )
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--status-converged", type=Path, required=True)
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--handoff", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "handoff.md"))
    ap.add_argument(
        "--max-total-regression-pct",
        type=float,
        default=8.0,
        help="Pass-through backend_compiler timing budget threshold.",
    )
    ap.add_argument(
        "--min-regression-delta-ms",
        type=float,
        default=50.0,
        help="Pass-through backend_compiler minimum delta-ms regression threshold.",
    )
    ap.add_argument(
        "--max-regression-ratio",
        type=float,
        default=0.5,
        help="Pass-through backend_compiler regression ratio threshold.",
    )
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "ci_gate.json"))
    args = ap.parse_args()
    active_legacy = ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"

    def _parse_profiles(raw: list[str]) -> list[str]:
        out: list[str] = []
        src = raw or ["coverage"]
        for item in src:
            for token in str(item).split(","):
                p = str(token).strip()
                if p and p not in out:
                    out.append(p)
        return out

    profiles = _parse_profiles(list(args.profiles))
    coverage_active = Path(args.active_batch) if args.active_batch is not None else args.active_batch_coverage
    active_by_profile: dict[str, Path] = {
        "coverage": coverage_active if coverage_active.is_file() else active_legacy,
        "ir_arch": args.active_batch_ir_arch,
        "backend_compiler": args.active_batch_backend_compiler,
    }

    checks: list[dict[str, Any]] = []
    for p in (args.scripts_catalog,):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))
    for p in (args.registry, args.feature_list, args.run_summary, args.status_converged):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))
    for profile in profiles:
        p = active_by_profile.get(profile)
        if p is None:
            checks.append(_check(f"active_batch::{profile}", False, "unsupported profile"))
            continue
        if profile == "coverage":
            checks.append(_check(f"exists::active_batch::{profile}", p.is_file(), "file exists" if p.is_file() else "missing file"))
        else:
            checks.append(
                _check(
                    f"exists::active_batch::{profile}",
                    True,
                    "file exists" if p.is_file() else "missing optional lane batch (treated as empty)",
                )
            )

    if args.registry.is_file() and args.feature_list.is_file():
        registry_payload = load_json(args.registry)
        feature_payload = load_json(args.feature_list)
        sync_ok, sync_errors = validate_feature_list_sync(
            feature_payload=feature_payload,
            registry_payload=registry_payload,
            expected_source_registry_path=_to_repo_rel(args.registry),
        )
        checks.append(
            _check(
                "feature_list.sync_with_registry",
                sync_ok,
                "feature list is synced with registry" if sync_ok else "; ".join(sync_errors),
            )
        )

    if args.scripts_catalog.is_file():
        catalog_report = args.out.with_name("catalog_validation_ci.json")
        cmd = [
            sys.executable,
            "scripts/validate_catalog.py",
            "--catalog",
            str(args.scripts_catalog),
            "--out",
            str(catalog_report),
        ]
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        checks.append(
            _check(
                "scripts.catalog_valid",
                p.returncode == 0,
                "scripts catalog validation passed"
                if p.returncode == 0
                else f"scripts catalog validation failed: {(p.stderr or p.stdout).strip()}",
            )
        )

    for profile in profiles:
        active_path = active_by_profile.get(profile)
        if active_path is None:
            continue
        run_gate = True
        if active_path.is_file():
            payload = load_json(active_path)
            items = [e for e in list(payload.get("items") or []) if isinstance(e, dict)]
            if profile != "coverage" and not items:
                run_gate = False
        if not active_path.is_file() and profile != "coverage":
            run_gate = False
        if not run_gate:
            checks.append(_check(f"batch_gate::{profile}", True, "skipped (no active items for profile)"))
            continue
        batch_gate_path = args.out.with_name(f"batch_gate_ci_{profile}.json")
        cmd = [
            sys.executable,
            "scripts/flaggems/check_batch_gate.py",
            "--profile",
            str(profile),
            "--active-batch",
            str(active_path),
            "--run-summary",
            str(args.run_summary),
            "--status-converged",
            str(args.status_converged),
            "--progress-log",
            str(args.progress_log),
            "--handoff",
            str(args.handoff),
            "--out",
            str(batch_gate_path),
        ]
        if profile != "coverage":
            cmd.append("--no-require-active-dual-pass")
        if profile == "backend_compiler":
            cmd += [
                "--max-total-regression-pct",
                str(float(args.max_total_regression_pct)),
                "--min-regression-delta-ms",
                str(float(args.min_regression_delta_ms)),
                "--max-regression-ratio",
                str(float(args.max_regression_ratio)),
            ]
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        checks.append(
            _check(
                f"batch_gate::{profile}",
                p.returncode == 0,
                "check_batch_gate passed" if p.returncode == 0 else f"check_batch_gate failed: {(p.stderr or p.stdout).strip()}",
            )
        )

    if args.status_converged.is_file():
        converged = load_json(args.status_converged)
        entries = [e for e in (converged.get("entries") or []) if isinstance(e, dict)]
        has_unknown_reason = any(str(e.get("reason_code") or "") in {"", "unknown"} for e in entries)
        checks.append(
            _check(
                "status_converged.no_unknown_reason_code",
                not has_unknown_reason,
                "no unknown reason_code" if not has_unknown_reason else "unknown reason_code present",
            )
        )

    ok = all(bool(c.get("ok")) for c in checks)
    payload = {
        "ok": bool(ok),
        "checks": checks,
        "artifacts": {
            "scripts_catalog": _to_repo_rel(args.scripts_catalog),
            "registry": _to_repo_rel(args.registry),
            "feature_list": _to_repo_rel(args.feature_list),
            "active_batch_coverage": _to_repo_rel(active_by_profile["coverage"]),
            "active_batch_ir_arch": _to_repo_rel(active_by_profile["ir_arch"]),
            "active_batch_backend_compiler": _to_repo_rel(active_by_profile["backend_compiler"]),
            "run_summary": _to_repo_rel(args.run_summary),
            "status_converged": _to_repo_rel(args.status_converged),
            "profiles": profiles,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CI gate report written: {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
