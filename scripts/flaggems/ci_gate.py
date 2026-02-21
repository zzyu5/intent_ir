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


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


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
    phase = str(payload.get("coverage_integrity_phase") or "").strip()
    if phase == "stale_or_unverifiable":
        return False, "full196 evidence unverifiable (missing repo provenance in artifacts); rerun full196 on HEAD"
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


def _validate_gpu_perf_fresh_on_head(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    payload = load_json(current_status_path)
    validated_commit = str(payload.get("gpu_perf_validated_commit") or "").strip()
    commits_since = payload.get("gpu_perf_commits_since_validated")
    phase = str(payload.get("gpu_perf_phase") or "").strip()
    if phase in {"stale_or_invalid", "recompute_stale"}:
        return False, f"gpu perf phase stale: {phase}"
    if not validated_commit:
        return False, "current_status.gpu_perf_validated_commit is empty"
    head = _git_head_commit()
    if not head:
        return False, "failed to resolve git HEAD for gpu perf freshness check"
    if validated_commit != head:
        return False, f"gpu perf evidence stale: validated_commit={validated_commit} head={head}"
    if commits_since is not None:
        try:
            if int(commits_since) != 0:
                return False, f"gpu_perf_commits_since_validated={commits_since} (expected 0)"
        except Exception:
            return False, f"invalid gpu_perf_commits_since_validated={commits_since}"
    return True, "gpu perf evidence is fresh on HEAD"


def _validate_gpu_perf_categories_complete(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    payload = load_json(current_status_path)
    expected = payload.get("gpu_perf_categories_expected")
    completed = payload.get("gpu_perf_categories_completed")
    failed = [str(x) for x in list(payload.get("gpu_perf_categories_failed") or []) if str(x).strip()]
    try:
        expected_i = int(expected)
        completed_i = int(completed)
    except Exception:
        return False, "gpu perf category counters missing/invalid in current_status"
    if expected_i <= 0:
        return False, f"gpu_perf_categories_expected={expected_i} (expected >0)"
    if completed_i != expected_i:
        return False, f"gpu perf categories incomplete: {completed_i}/{expected_i}"
    if failed:
        return False, f"gpu perf categories failed: {failed}"
    return True, f"gpu perf categories complete: {completed_i}/{expected_i}"


def _validate_gpu_perf_per_device_ok(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    payload = load_json(current_status_path)
    devices = [d for d in list(payload.get("gpu_perf_devices") or []) if isinstance(d, dict)]
    if not devices:
        return False, "current_status.gpu_perf_devices is empty"
    failing = [str(d.get("gpu_name") or "unknown") for d in devices if not bool(d.get("ok"))]
    if failing:
        return False, f"gpu perf per-device failures: {failing}"
    return True, "gpu perf per-device status is healthy"


def _validate_mlir_fresh_on_head(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    payload = load_json(current_status_path)
    validated_commit = str(payload.get("mlir_full196_validated_commit") or "").strip()
    full196_last_ok = bool(payload.get("full196_last_ok"))
    execution_ir = str(payload.get("full196_validated_execution_ir") or "").strip().lower()
    if not validated_commit:
        return False, "current_status.mlir_full196_validated_commit is empty"
    if not full196_last_ok:
        return False, "current_status.full196_last_ok is not true"
    if execution_ir and execution_ir != "mlir":
        return False, f"full196_validated_execution_ir={execution_ir!r} (expected 'mlir')"
    head = _git_head_commit()
    if not head:
        return False, "failed to resolve git HEAD for mlir freshness check"
    if validated_commit != head:
        return False, f"mlir full196 evidence stale: validated_commit={validated_commit} head={head}"
    return True, "mlir full196 evidence is fresh on HEAD"


def _validate_mlir_toolchain_required(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    payload = load_json(current_status_path)
    toolchain_ok = payload.get("mlir_toolchain_ok")
    if toolchain_ok is not True:
        return False, "current_status.mlir_toolchain_ok is not true"
    cutover = str(payload.get("mlir_cutover_level") or "").strip()
    if cutover == "blocked_toolchain":
        return False, "mlir_cutover_level=blocked_toolchain"
    return True, "MLIR toolchain requirement satisfied"


def _validate_mlir_llvm_artifact_complete(run_summary: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(run_summary, dict) or not run_summary:
        return False, "run_summary missing for MLIR LLVM artifact check"
    execution_ir = str(
        run_summary.get("execution_ir")
        or (run_summary.get("invocation") or {}).get("execution_ir")
        or ""
    ).strip().lower()
    if execution_ir and execution_ir != "mlir":
        return True, f"skipped: execution_ir={execution_ir!r}"
    if bool(run_summary.get("mlir_llvm_artifact_complete")):
        return True, "run_summary.mlir_llvm_artifact_complete=true"
    llvm_ir_path = str(run_summary.get("llvm_ir_path") or "").strip()
    if llvm_ir_path:
        llvm_path = Path(llvm_ir_path)
        if not llvm_path.is_absolute():
            llvm_path = ROOT / llvm_path
        if llvm_path.is_file():
            return True, f"llvm_ir_path exists: {llvm_path}"
    stage_map = _stage_map(run_summary)
    stage = stage_map.get("mlir_llvm_artifacts") or {}
    if isinstance(stage, dict) and stage:
        artifact_complete = stage.get("artifact_complete")
        if artifact_complete is not None and not bool(artifact_complete):
            return False, "mlir_llvm_artifacts stage reports artifact_complete=false"
        stage_path_raw = str(stage.get("json_path") or "").strip()
        if stage_path_raw:
            stage_path = Path(stage_path_raw)
            if not stage_path.is_absolute():
                stage_path = ROOT / stage_path
            if stage_path.is_file():
                payload = load_json(stage_path)
                if bool(payload.get("artifact_complete")):
                    return True, f"mlir_llvm_artifacts complete: {stage_path}"
                return False, f"mlir_llvm_artifacts incomplete: {stage_path}"
        if bool(artifact_complete):
            return True, "mlir_llvm_artifacts stage marked complete"
    llvm_emit_stage = stage_map.get("llvm_emit") or {}
    if isinstance(llvm_emit_stage, dict) and bool(llvm_emit_stage.get("ok")):
        stage_path_raw = str(llvm_emit_stage.get("json_path") or "").strip()
        if not stage_path_raw:
            return True, "llvm_emit stage ok"
        stage_path = Path(stage_path_raw)
        if not stage_path.is_absolute():
            stage_path = ROOT / stage_path
        if stage_path.is_file():
            return True, f"llvm_emit stage artifact exists: {stage_path}"
    # Category-batch aggregate coverage runs often carry MLIR/LLVM evidence via
    # stage_timing_breakdown aggregation.
    timing_stage = stage_map.get("stage_timing_breakdown") or {}
    if isinstance(timing_stage, dict):
        timing_json_raw = str(timing_stage.get("json_path") or "").strip()
        if timing_json_raw:
            timing_path = Path(timing_json_raw)
            if not timing_path.is_absolute():
                timing_path = ROOT / timing_path
            if timing_path.is_file():
                payload = load_json(timing_path)
                mlir = payload.get("mlir")
                if isinstance(mlir, dict) and bool(mlir.get("available")):
                    totals = mlir.get("totals_ms")
                    if isinstance(totals, dict):
                        try:
                            total_ms = float(totals.get("mlir_total_ms", 0.0))
                        except Exception:
                            total_ms = 0.0
                        if total_ms > 0.0:
                            return True, f"stage_timing_breakdown mlir totals present: {timing_path}"
    return False, "missing LLVM artifact evidence (llvm_ir_path/mlir_llvm_artifacts/llvm_emit)"


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


def _validate_stage_timing_breakdown(path: Path, *, require_mlir: bool = False) -> tuple[bool, str]:
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
    if bool(require_mlir):
        mlir = payload.get("mlir")
        if not isinstance(mlir, dict):
            return False, "mlir section missing in stage_timing_breakdown"
        if not bool(mlir.get("available")):
            return False, "mlir section not available in stage_timing_breakdown"
        totals = mlir.get("totals_ms")
        if not isinstance(totals, dict):
            return False, "mlir.totals_ms missing in stage_timing_breakdown"
        total_ms = _safe_float(totals.get("mlir_total_ms"))
        if total_ms <= 0.0:
            return False, "mlir_total_ms nonpositive in stage_timing_breakdown"
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
        "--active-batch-gpu-perf",
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
        "--active-batch-workflow",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_workflow.json"),
    )
    ap.add_argument(
        "--active-batch-mlir-migration",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_mlir_migration.json"),
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
        "--require-gpu-perf-fresh-on-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require gpu perf evidence to be validated on current HEAD when gpu_perf profile is evaluated.",
    )
    ap.add_argument(
        "--require-gpu-perf-categories-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require gpu perf category aggregate completion in current_status when gpu_perf profile is evaluated.",
    )
    ap.add_argument(
        "--require-gpu-perf-per-device-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require gpu perf per-device status to be all-ok when gpu_perf profile is evaluated.",
    )
    ap.add_argument(
        "--gpu-perf-threshold",
        type=float,
        default=0.80,
        help="Pass-through gpu perf ratio threshold for check_batch_gate (default: 0.80).",
    )
    ap.add_argument(
        "--require-mlir-fresh-on-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require mlir full196 evidence to be fresh on HEAD when mlir_migration profile is evaluated.",
    )
    ap.add_argument(
        "--require-mlir-toolchain-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require current_status.mlir_toolchain_ok=true when mlir_migration profile is evaluated.",
    )
    ap.add_argument(
        "--require-mlir-llvm-artifact-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require LLVM artifact evidence for MLIR migration runs.",
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
        "gpu_perf": args.active_batch_gpu_perf,
        "ir_arch": args.active_batch_ir_arch,
        "backend_compiler": args.active_batch_backend_compiler,
        "workflow": args.active_batch_workflow,
        "mlir_migration": args.active_batch_mlir_migration,
    }

    checks: list[dict[str, Any]] = []
    for p in (args.scripts_catalog,):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))
    for p in (args.registry, args.feature_list, args.run_summary, args.status_converged):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))
    if (
        bool(args.require_coverage_fresh_on_head)
        or bool(args.require_coverage_categories_complete)
        or (bool(args.require_gpu_perf_fresh_on_head) and ("gpu_perf" in profiles))
        or (bool(args.require_gpu_perf_categories_complete) and ("gpu_perf" in profiles))
        or (bool(args.require_gpu_perf_per_device_ok) and ("gpu_perf" in profiles))
        or (bool(args.require_mlir_fresh_on_head) and ("mlir_migration" in profiles))
        or (bool(args.require_mlir_toolchain_required) and ("mlir_migration" in profiles))
    ):
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
        if profile in {"coverage", "gpu_perf"}:
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
    if bool(args.require_gpu_perf_fresh_on_head) and ("gpu_perf" in profiles):
        gpu_fresh_ok, gpu_fresh_detail = _validate_gpu_perf_fresh_on_head(args.current_status)
        checks.append(_check("gpu_perf_fresh_on_head", gpu_fresh_ok, gpu_fresh_detail))
    if bool(args.require_gpu_perf_categories_complete) and ("gpu_perf" in profiles):
        gpu_cat_ok, gpu_cat_detail = _validate_gpu_perf_categories_complete(args.current_status)
        checks.append(_check("gpu_perf_categories_complete", gpu_cat_ok, gpu_cat_detail))
    if bool(args.require_gpu_perf_per_device_ok) and ("gpu_perf" in profiles):
        gpu_dev_ok, gpu_dev_detail = _validate_gpu_perf_per_device_ok(args.current_status)
        checks.append(_check("gpu_perf_per_device_ok", gpu_dev_ok, gpu_dev_detail))
    if bool(args.require_mlir_fresh_on_head) and ("mlir_migration" in profiles):
        mlir_ok, mlir_detail = _validate_mlir_fresh_on_head(args.current_status)
        checks.append(_check("mlir_fresh_on_head", mlir_ok, mlir_detail))
    if bool(args.require_mlir_toolchain_required) and ("mlir_migration" in profiles):
        tc_ok, tc_detail = _validate_mlir_toolchain_required(args.current_status)
        checks.append(_check("mlir_toolchain_required", tc_ok, tc_detail))

    run_summary_payload: dict[str, Any] = {}
    if args.run_summary.is_file():
        run_summary_payload = load_json(args.run_summary)
        if bool(args.require_mlir_llvm_artifact_complete) and ("mlir_migration" in profiles):
            llvm_ok, llvm_detail = _validate_mlir_llvm_artifact_complete(run_summary_payload)
            checks.append(_check("mlir_llvm_artifact_complete", llvm_ok, llvm_detail))
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
                execution_ir = str(run_summary_payload.get("execution_ir") or "").strip().lower()
                stage_ok, stage_detail = _validate_stage_timing_breakdown(
                    stage_path,
                    require_mlir=(execution_ir == "mlir"),
                )
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
            if profile not in {"coverage", "gpu_perf"} and not items:
                run_gate = False
        if not active_path.is_file() and profile not in {"coverage", "gpu_perf"}:
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
        if bool(args.require_gpu_perf_fresh_on_head) and profile == "gpu_perf":
            cmd.append("--require-gpu-perf-fresh-on-head")
        else:
            cmd.append("--no-require-gpu-perf-fresh-on-head")
        if bool(args.require_gpu_perf_categories_complete) and profile == "gpu_perf":
            cmd.append("--require-gpu-perf-categories-complete")
        else:
            cmd.append("--no-require-gpu-perf-categories-complete")
        if bool(args.require_mlir_fresh_on_head) and profile == "mlir_migration":
            cmd.append("--require-mlir-fresh-on-head")
        else:
            cmd.append("--no-require-mlir-fresh-on-head")
        if bool(args.require_mlir_toolchain_required) and profile == "mlir_migration":
            cmd.append("--require-mlir-toolchain-required")
        else:
            cmd.append("--no-require-mlir-toolchain-required")
        if bool(args.require_mlir_llvm_artifact_complete) and profile == "mlir_migration":
            cmd.append("--require-mlir-llvm-artifact-complete")
        else:
            cmd.append("--no-require-mlir-llvm-artifact-complete")
        if profile == "coverage" and bool(args.require_coverage_categories_complete):
            is_batch_aggregate = str(run_summary_payload.get("full196_evidence_kind") or "") == "batch_aggregate"
            if is_batch_aggregate and _is_full_coverage_run(run_summary_payload):
                cmd.append("--require-all-categories-complete")
            else:
                cmd.append("--no-require-all-categories-complete")
        else:
            cmd.append("--no-require-all-categories-complete")
        if profile == "gpu_perf":
            cmd += [
                "--gpu-perf-threshold",
                str(float(args.gpu_perf_threshold)),
                "--no-require-progress-tail-match",
            ]
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
            "active_batch_gpu_perf": _to_repo_rel(active_by_profile["gpu_perf"]),
            "active_batch_ir_arch": _to_repo_rel(active_by_profile["ir_arch"]),
            "active_batch_backend_compiler": _to_repo_rel(active_by_profile["backend_compiler"]),
            "active_batch_workflow": _to_repo_rel(active_by_profile["workflow"]),
            "active_batch_mlir_migration": _to_repo_rel(active_by_profile["mlir_migration"]),
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
