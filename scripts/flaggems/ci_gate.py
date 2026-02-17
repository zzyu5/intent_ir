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


def _git_head_commit() -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return ""
    return str(p.stdout or "").strip()


def _validate_coverage_fresh_on_head(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    payload = load_json(current_status_path)
    validated_commit = str(payload.get("full196_validated_commit") or "").strip()
    full196_last_ok = bool(payload.get("full196_last_ok"))
    commits_since = payload.get("full196_commits_since_validated")
    if not validated_commit:
        return False, "current_status.full196_validated_commit is empty"
    if not full196_last_ok:
        return False, "current_status.full196_last_ok is not true"
    head = _git_head_commit()
    if not head:
        return False, "failed to resolve git HEAD for freshness check"
    if validated_commit != head:
        return False, (
            f"full196 evidence stale: validated_commit={validated_commit} head={head}"
            + (f" commits_since={commits_since}" if commits_since is not None else "")
        )
    if commits_since is not None:
        try:
            if int(commits_since) != 0:
                return False, f"full196_commits_since_validated={commits_since} (expected 0)"
        except Exception:
            return False, f"invalid full196_commits_since_validated={commits_since}"
    return True, "full196 evidence is fresh on HEAD"


def _validate_coverage_categories_complete(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    payload = load_json(current_status_path)
    mode = str(payload.get("coverage_mode") or "").strip()
    evidence = str(payload.get("full196_evidence_kind") or "").strip()
    expected = payload.get("coverage_batches_expected")
    completed = payload.get("coverage_batches_completed")
    failed = list(payload.get("coverage_batches_failed") or [])
    try:
        expected_i = int(expected)
        completed_i = int(completed)
    except Exception:
        return False, "coverage batch counters missing/invalid in current_status"
    if mode != "category_batches":
        return False, f"coverage_mode is {mode!r} (expected 'category_batches')"
    if evidence != "batch_aggregate":
        return False, f"full196_evidence_kind is {evidence!r} (expected 'batch_aggregate')"
    if expected_i <= 0:
        return False, f"coverage_batches_expected={expected_i} (expected >0)"
    if completed_i != expected_i:
        return False, f"coverage batches incomplete: {completed_i}/{expected_i}"
    if failed:
        return False, f"coverage batches failed: {failed}"
    return True, f"coverage categories complete: {completed_i}/{expected_i}"


def _stage_map(run_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [r for r in list(run_summary.get("stages") or []) if isinstance(r, dict)]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("stage") or "").strip()
        if name:
            out[name] = row
    return out


def _is_full_coverage_run(run_summary: dict[str, Any]) -> bool:
    suite = str(run_summary.get("suite") or "").strip()
    if suite != "coverage":
        return False
    coverage_stage = _stage_map(run_summary).get("coverage_integrity") or {}
    if not coverage_stage:
        return False
    if str(coverage_stage.get("reason_code") or "").strip() == "skipped_partial_scope":
        return False
    if "full_coverage_run" in coverage_stage and not bool(coverage_stage.get("full_coverage_run")):
        return False
    return True


def _validate_stage_timing_breakdown(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing stage_timing_breakdown json: {path}"
    payload = load_json(path)
    if str(payload.get("schema_version") or "") != "flaggems_stage_timing_breakdown_v1":
        return False, "unexpected stage_timing_breakdown schema_version"
    backends = payload.get("backends")
    if not isinstance(backends, dict):
        return False, "stage_timing_breakdown missing backends section"
    failures: list[str] = []
    for backend in ("rvv", "cuda"):
        section = backends.get(backend)
        if not isinstance(section, dict):
            failures.append(f"{backend}:missing_section")
            continue
        if not bool(section.get("available")):
            failures.append(f"{backend}:not_available")
            continue
        totals = section.get("totals_ms")
        if not isinstance(totals, dict):
            failures.append(f"{backend}:totals_not_object")
            continue
        try:
            total_ms = float(totals.get("total_ms", 0.0))
        except Exception:
            total_ms = 0.0
            failures.append(f"{backend}:total_ms_invalid")
        if total_ms <= 0.0:
            failures.append(f"{backend}:total_ms_nonpositive")
    if failures:
        return False, "; ".join(failures)
    return True, "stage_timing_breakdown present with positive totals for rvv/cuda"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--scripts-catalog", type=Path, default=(ROOT / "scripts" / "CATALOG.json"))
    ap.add_argument(
        "--current-status",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "current_status.json"),
        help="Workflow current_status snapshot used for full196 freshness checks.",
    )
    ap.add_argument("--feature-list", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument(
        "--active-batch-coverage",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_coverage.json"),
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
        "--require-coverage-fresh-on-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require full196 evidence validated on current HEAD commit.",
    )
    ap.add_argument(
        "--require-coverage-categories-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require full196 evidence to come from completed category aggregate coverage.",
    )
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
    coverage_active = args.active_batch_coverage
    active_by_profile: dict[str, Path] = {
        "coverage": coverage_active,
        "ir_arch": args.active_batch_ir_arch,
        "backend_compiler": args.active_batch_backend_compiler,
    }

    checks: list[dict[str, Any]] = []
    for p in (args.scripts_catalog,):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))
    for p in (args.registry, args.feature_list, args.run_summary, args.status_converged):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))
    if bool(args.require_coverage_fresh_on_head) or bool(args.require_coverage_categories_complete):
        checks.append(
            _check(
                f"exists::{_to_repo_rel(args.current_status)}",
                args.current_status.is_file(),
                "file exists" if args.current_status.is_file() else "missing file",
            )
        )
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

    if bool(args.require_coverage_fresh_on_head):
        freshness_ok, freshness_detail = _validate_coverage_fresh_on_head(args.current_status)
        checks.append(_check("coverage_fresh_on_head", freshness_ok, freshness_detail))
    if bool(args.require_coverage_categories_complete):
        categories_ok, categories_detail = _validate_coverage_categories_complete(args.current_status)
        checks.append(_check("coverage_categories_complete", categories_ok, categories_detail))

    run_summary_payload: dict[str, Any] = {}
    if args.run_summary.is_file():
        run_summary_payload = load_json(args.run_summary)
        if _is_full_coverage_run(run_summary_payload):
            stage_row = _stage_map(run_summary_payload).get("stage_timing_breakdown") or {}
            stage_path_raw = str(stage_row.get("json_path") or "").strip()
            if not stage_path_raw:
                checks.append(
                    _check(
                        "run_summary.stage_timing_breakdown_full196",
                        False,
                        "full coverage run missing stage_timing_breakdown json_path",
                    )
                )
            else:
                stage_path = Path(stage_path_raw)
                if not stage_path.is_absolute():
                    stage_path = ROOT / stage_path
                stage_ok, stage_detail = _validate_stage_timing_breakdown(stage_path)
                checks.append(
                    _check(
                        "run_summary.stage_timing_breakdown_full196",
                        stage_ok,
                        stage_detail,
                    )
                )
        else:
            checks.append(
                _check(
                    "run_summary.stage_timing_breakdown_full196",
                    True,
                    "skipped: run_summary is not a full coverage run",
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
            "--current-status",
            str(args.current_status),
            "--out",
            str(batch_gate_path),
        ]
        if profile != "coverage":
            cmd.append("--no-require-active-dual-pass")
        if bool(args.require_coverage_fresh_on_head):
            cmd.append("--require-coverage-fresh-on-head")
        else:
            cmd.append("--no-require-coverage-fresh-on-head")
        if profile == "coverage" and bool(args.require_coverage_categories_complete):
            is_batch_aggregate = str(run_summary_payload.get("full196_evidence_kind") or "") == "batch_aggregate"
            if is_batch_aggregate and _is_full_coverage_run(run_summary_payload):
                cmd.append("--require-all-categories-complete")
            else:
                cmd.append("--no-require-all-categories-complete")
        else:
            cmd.append("--no-require-all-categories-complete")
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
