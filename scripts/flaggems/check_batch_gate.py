"""
Validate hard gate artifacts for one FlagGems batch.
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    status = _load_json(current_status_path)
    validated_commit = str(status.get("full196_validated_commit") or "").strip()
    full196_last_ok = bool(status.get("full196_last_ok"))
    commits_since = status.get("full196_commits_since_validated")
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


def _validate_mlir_fresh_on_head(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    status = _load_json(current_status_path)
    validated_commit = str(status.get("mlir_full196_validated_commit") or "").strip()
    full196_last_ok = bool(status.get("full196_last_ok"))
    execution_ir = str(status.get("full196_validated_execution_ir") or "").strip().lower()
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
    status = _load_json(current_status_path)
    toolchain_ok = status.get("mlir_toolchain_ok")
    if toolchain_ok is not True:
        return False, "current_status.mlir_toolchain_ok is not true"
    cutover = str(status.get("mlir_cutover_level") or "").strip()
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
        p = Path(llvm_ir_path)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            return True, f"llvm_ir_path exists: {p}"
    stage_map = _stage_map(run_summary)
    stage = stage_map.get("mlir_llvm_artifacts") or {}
    if isinstance(stage, dict) and stage:
        artifact_complete = stage.get("artifact_complete")
        if artifact_complete is not None and not bool(artifact_complete):
            return False, "mlir_llvm_artifacts stage reports artifact_complete=false"
        stage_json = str(stage.get("json_path") or "").strip()
        if stage_json:
            p = Path(stage_json)
            if not p.is_absolute():
                p = ROOT / p
            if p.is_file():
                payload = _load_json(p)
                if bool(payload.get("artifact_complete")):
                    return True, f"mlir_llvm_artifacts complete: {p}"
                return False, f"mlir_llvm_artifacts incomplete: {p}"
        if bool(artifact_complete):
            return True, "mlir_llvm_artifacts stage marked complete"
    llvm_emit_stage = stage_map.get("llvm_emit") or {}
    if isinstance(llvm_emit_stage, dict) and bool(llvm_emit_stage.get("ok")):
        stage_json = str(llvm_emit_stage.get("json_path") or "").strip()
        if not stage_json:
            return True, "llvm_emit stage ok"
        p = Path(stage_json)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            return True, f"llvm_emit stage artifact exists: {p}"
    return False, "missing LLVM artifact evidence (llvm_ir_path/mlir_llvm_artifacts/llvm_emit)"


def _validate_gpu_perf_fresh_on_head(current_status_path: Path) -> tuple[bool, str]:
    if not current_status_path.is_file():
        return False, f"missing current_status: {current_status_path}"
    status = _load_json(current_status_path)
    validated_commit = str(status.get("gpu_perf_validated_commit") or "").strip()
    commits_since = status.get("gpu_perf_commits_since_validated")
    phase = str(status.get("gpu_perf_phase") or "").strip()
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
    status = _load_json(current_status_path)
    expected = status.get("gpu_perf_categories_expected")
    completed = status.get("gpu_perf_categories_completed")
    failed = [str(x) for x in list(status.get("gpu_perf_categories_failed") or []) if str(x).strip()]
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


def _validate_gpu_perf_json(
    gpu_perf_path: Path,
    *,
    threshold: float,
    require_categories_complete: bool,
) -> tuple[bool, str]:
    if not gpu_perf_path.is_file():
        return False, f"missing gpu_perf_graph json: {gpu_perf_path}"
    payload = _load_json(gpu_perf_path)
    schema = str(payload.get("schema_version") or "")
    if schema != "flaggems_gpu_perf_graph_v1":
        return False, f"unexpected gpu_perf schema: {schema}"
    if str(payload.get("mode") or "") != "graph_only":
        return False, f"unexpected gpu perf mode: {payload.get('mode')!r}"
    if require_categories_complete:
        expected = payload.get("coverage_batches_expected")
        completed = payload.get("coverage_batches_completed")
        failed = [str(x) for x in list(payload.get("coverage_batches_failed") or []) if str(x).strip()]
        try:
            expected_i = int(expected)
            completed_i = int(completed)
        except Exception:
            return False, "gpu perf category counters missing/invalid in payload"
        if expected_i <= 0:
            return False, f"gpu perf coverage_batches_expected={expected_i} (expected >0)"
        if completed_i != expected_i:
            return False, f"gpu perf categories incomplete: {completed_i}/{expected_i}"
        if failed:
            return False, f"gpu perf categories failed: {failed}"
    devices = [d for d in list(payload.get("devices") or []) if isinstance(d, dict)]
    if not devices:
        return False, "gpu perf payload has no devices[] entries"
    failing_devices = [str(d.get("gpu_name") or "unknown") for d in devices if not bool(d.get("ok"))]
    if failing_devices:
        return False, f"gpu perf per-device gate failed: {failing_devices}"
    entries = [e for e in list(payload.get("entries") or []) if isinstance(e, dict)]
    measured = [e for e in entries if bool(e.get("count_in_denominator"))]
    if not measured:
        return False, "gpu perf payload has zero measured kernels"
    bad = [
        str(e.get("kernel") or "unknown")
        for e in measured
        if (not isinstance(e.get("ratio"), (int, float))) or float(e.get("ratio")) < float(threshold)
    ]
    if bad:
        return False, f"gpu perf ratio below threshold {float(threshold):.2f}: {bad[:8]}"
    return True, "gpu perf payload passes threshold/per-device checks"


def _default_active_batch(profile: str) -> Path:
    if profile == "coverage":
        return ROOT / "workflow" / "flaggems" / "state" / "active_batch_coverage.json"
    if profile == "gpu_perf":
        return ROOT / "workflow" / "flaggems" / "state" / "active_batch_coverage.json"
    if profile == "mlir_migration":
        return ROOT / "workflow" / "flaggems" / "state" / "active_batch_mlir_migration.json"
    return ROOT / "workflow" / "flaggems" / "state" / f"active_batch_{profile}.json"


def _stage_map(run_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [r for r in list(run_summary.get("stages") or []) if isinstance(r, dict)]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("stage") or "").strip()
        if not name:
            continue
        out[name] = row
    return out


def _resolve_backend_stage(
    stage_map: dict[str, dict[str, Any]], stage_name: str
) -> tuple[str, dict[str, Any]]:
    aliases: dict[str, tuple[str, ...]] = {
        "rvv_local": ("rvv_local", "rvv_remote"),
        "rvv_remote": ("rvv_remote", "rvv_local"),
        "cuda_local": ("cuda_local", "cuda"),
        "cuda": ("cuda", "cuda_local"),
    }
    for candidate in aliases.get(stage_name, (stage_name,)):
        row = stage_map.get(candidate)
        if isinstance(row, dict):
            return candidate, row
    return stage_name, {}


def _validate_backend_timing(summary_path: Path) -> tuple[bool, str]:
    if not summary_path.is_file():
        return False, f"backend stage json missing: {summary_path}"
    payload = _load_json(summary_path)
    results = payload.get("results")
    if not isinstance(results, list):
        return False, "backend stage json missing results[]"
    required = {"lower_ms", "compile_ms", "launch_ms", "total_ms"}
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            return False, f"backend stage result[{idx}] is not object"
        missing = [k for k in required if k not in row]
        if missing:
            return False, f"backend stage result[{idx}] missing timing fields: {missing}"
    return True, "backend stage timing fields present"


def _validate_backend_stage_realized(summary_path: Path) -> tuple[bool, str]:
    if not summary_path.is_file():
        return False, f"backend stage json missing: {summary_path}"
    payload = _load_json(summary_path)
    results = payload.get("results")
    if not isinstance(results, list):
        return False, "backend stage json missing results[]"
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            return False, f"backend stage result[{idx}] is not object"
        reason = str(row.get("reason_code") or "").strip()
        if reason in {"", "unknown"}:
            return False, f"backend stage result[{idx}] has empty/unknown reason_code"
        for field in ("compile_ms", "launch_ms"):
            if field not in row:
                return False, f"backend stage result[{idx}] missing {field}"
            try:
                ms = float(row.get(field, 0.0))
            except Exception:
                return False, f"backend stage result[{idx}] invalid {field}"
            if ms < 0.0:
                return False, f"backend stage result[{idx}] negative {field}"
        for marker in ("pipeline_mode", "compile_stage_mode", "launch_stage_mode", "mode"):
            val = str(row.get(marker) or "").strip().lower()
            if val == "deferred":
                return False, f"backend stage result[{idx}] still deferred ({marker}=deferred)"
    return True, "backend stages realized (no deferred marker, reason_code complete)"


def _validate_codegen_purity(stage_map: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    forbidden_flags = {
        "--codegen-mode",
        "--codegen-strict",
        "--cpp-engine",
        "--cpp-engine-strict",
        "--cuda-codegen-mode",
        "--cuda-codegen-strict",
        "--cuda-cpp-engine",
        "--cuda-cpp-engine-strict",
    }
    forbidden_env_prefixes = (
        "INTENTIR_CUDA_CODEGEN",
        "INTENTIR_CUDA_CPP_CODEGEN_ENGINE",
    )
    violations: list[str] = []
    for stage_name, row in stage_map.items():
        cmd = row.get("cmd")
        tokens: list[str] = []
        if isinstance(cmd, list):
            tokens = [str(x) for x in cmd]
        elif isinstance(cmd, str):
            tokens = str(cmd).split()
        for flag in forbidden_flags:
            if flag in tokens:
                violations.append(f"{stage_name}:forbidden_flag={flag}")
        env_overrides = row.get("env_overrides")
        if isinstance(env_overrides, dict):
            for key in env_overrides.keys():
                key_s = str(key)
                if any(key_s.startswith(prefix) for prefix in forbidden_env_prefixes):
                    violations.append(f"{stage_name}:forbidden_env={key_s}")
    if violations:
        return False, "; ".join(violations)
    return True, "no deprecated codegen fallback flags/env in stage commands"


def _validate_stage_timing_breakdown(stage_path: Path) -> tuple[bool, str]:
    if not stage_path.is_file():
        return False, f"stage timing breakdown json missing: {stage_path}"
    payload = _load_json(stage_path)
    schema = str(payload.get("schema_version") or "")
    if schema != "flaggems_stage_timing_breakdown_v1":
        return False, f"unexpected stage timing schema: {schema}"
    backends = payload.get("backends")
    if not isinstance(backends, dict):
        return False, "stage timing breakdown missing backends object"
    failures: list[str] = []
    for backend in ("rvv", "cuda"):
        section = backends.get(backend)
        if not isinstance(section, dict):
            failures.append(f"{backend}:missing_section")
            continue
        if not bool(section.get("available")):
            failures.append(f"{backend}:not_available")
            continue
        for field in ("kernel_count", "totals_ms", "avg_ms", "stage_share_pct"):
            if field not in section:
                failures.append(f"{backend}:missing_{field}")
        totals = section.get("totals_ms")
        if isinstance(totals, dict):
            for key in ("lower_ms", "compile_ms", "launch_ms", "total_ms"):
                if key not in totals:
                    failures.append(f"{backend}:totals_missing_{key}")
            try:
                total_ms = float(totals.get("total_ms", 0.0))
            except Exception:
                failures.append(f"{backend}:totals_invalid_total_ms")
                total_ms = 0.0
            if total_ms <= 0.0:
                failures.append(f"{backend}:totals_nonpositive_total_ms")
        else:
            failures.append(f"{backend}:totals_not_object")
    combined = payload.get("combined")
    if isinstance(combined, dict):
        totals = combined.get("totals_ms")
        if isinstance(totals, dict):
            try:
                combined_total_ms = float(totals.get("total_ms", 0.0))
            except Exception:
                combined_total_ms = 0.0
                failures.append("combined:invalid_total_ms")
            if combined_total_ms <= 0.0:
                failures.append("combined:total_ms_nonpositive")
    if failures:
        return False, "; ".join(failures)
    return True, "stage timing breakdown complete for rvv/cuda"


def _breakdown_backend_has_evidence(stage_path: Path, backend: str) -> bool:
    if not stage_path.is_file():
        return False
    payload = _load_json(stage_path)
    backends = payload.get("backends")
    if not isinstance(backends, dict):
        return False
    section = backends.get(backend)
    if not isinstance(section, dict):
        return False
    if not bool(section.get("available")):
        return False
    totals = section.get("totals_ms")
    if not isinstance(totals, dict):
        return False
    try:
        return float(totals.get("total_ms", 0.0)) > 0.0
    except Exception:
        return False


def _validate_ir_arch_mapping_quality(
    mapping_report_path: Path,
    *,
    max_complex_single_intent_ratio: float,
    max_global_unique_single_primitive_ratio: float,
) -> tuple[bool, str]:
    if not mapping_report_path.is_file():
        return False, f"mapping complexity report missing: {mapping_report_path}"
    payload = _load_json(mapping_report_path)
    try:
        complex_ratio = float(
            payload.get("composition_required_single_intent_ratio", payload.get("complex_family_single_semantic_ratio", 0.0))
        )
        global_unique_ratio = float(payload.get("global_unique_single_primitive_ratio", 0.0))
    except Exception:
        return False, "invalid mapping complexity ratios in report"
    if complex_ratio > float(max_complex_single_intent_ratio):
        return (
            False,
            f"complex ratio {complex_ratio:.4f} > {float(max_complex_single_intent_ratio):.4f}",
        )
    if global_unique_ratio > float(max_global_unique_single_primitive_ratio):
        return (
            False,
            f"global unique single-primitive ratio {global_unique_ratio:.4f} > "
            f"{float(max_global_unique_single_primitive_ratio):.4f}",
        )
    return (
        True,
        "mapping quality within thresholds "
        f"(complex={complex_ratio:.4f}, global_unique={global_unique_ratio:.4f})",
    )


def _validate_timing_delta_budget(
    timing_delta_path: Path,
    *,
    max_total_regression_pct: float,
    min_regression_delta_ms: float,
    max_regression_ratio: float,
) -> tuple[bool, str]:
    if not timing_delta_path.is_file():
        return False, f"timing_delta json missing: {timing_delta_path}"
    payload = _load_json(timing_delta_path)
    regressions: list[str] = []
    compare_enabled_any = False
    ratio_failures: list[str] = []
    for backend in ("rvv", "cuda"):
        section = payload.get(backend)
        if not isinstance(section, dict):
            continue
        if not bool(section.get("compare_enabled")):
            continue
        compare_enabled_any = True
        rows = [r for r in list(section.get("rows") or []) if isinstance(r, dict)]
        backend_total = 0
        backend_regressions = 0
        for row in rows:
            kernel = str(row.get("kernel") or "")
            total = row.get("total_ms")
            if not isinstance(total, dict):
                continue
            try:
                delta_pct = float(total.get("delta_pct", 0.0))
                delta_ms = float(total.get("delta_ms", 0.0))
            except Exception:
                continue
            backend_total += 1
            if delta_pct > float(max_total_regression_pct) and delta_ms > float(min_regression_delta_ms):
                backend_regressions += 1
                regressions.append(f"{backend}:{kernel}:{delta_pct:.2f}% ({delta_ms:.2f}ms)")
        if backend_total > 0:
            ratio = float(backend_regressions) / float(backend_total)
            if ratio > float(max_regression_ratio):
                ratio_failures.append(
                    f"{backend}:ratio={ratio:.3f} ({backend_regressions}/{backend_total})"
                )
    if not compare_enabled_any:
        return True, "timing_delta compare disabled (no baseline), budget check skipped"
    if ratio_failures:
        detail = (
            f"regression ratio > {max_regression_ratio:.3f} "
            f"(pct>{max_total_regression_pct:.2f} and delta_ms>{min_regression_delta_ms:.2f})"
        )
        if regressions:
            detail += f": {', '.join(regressions)} | {'; '.join(ratio_failures)}"
        else:
            detail += f": {'; '.join(ratio_failures)}"
        return False, detail
    return True, (
        f"timing_delta regression ratio <= {max_regression_ratio:.3f} "
        f"(pct>{max_total_regression_pct:.2f} and delta_ms>{min_regression_delta_ms:.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        choices=["coverage", "gpu_perf", "ir_arch", "backend_compiler", "workflow", "mlir_migration"],
        default="coverage",
        help="Gate profile to evaluate.",
    )
    ap.add_argument("--active-batch", type=Path, default=None)
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--status-converged", type=Path, required=True)
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--handoff", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "handoff.md"))
    ap.add_argument(
        "--current-status",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "current_status.json"),
        help="Workflow current_status snapshot used for coverage freshness checks.",
    )
    ap.add_argument(
        "--require-coverage-fresh-on-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require full196 evidence to be validated on current HEAD.",
    )
    ap.add_argument(
        "--require-gpu-perf-fresh-on-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require gpu perf evidence to be validated on current HEAD.",
    )
    ap.add_argument(
        "--require-gpu-perf-categories-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require gpu perf categories to be completed in current_status and payload.",
    )
    ap.add_argument(
        "--require-progress-tail-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require progress_log tail to reference current run/status artifacts (default: true).",
    )
    ap.add_argument(
        "--gpu-perf-threshold",
        type=float,
        default=0.80,
        help="Minimum qps ratio threshold for gpu_perf profile (default: 0.80).",
    )
    ap.add_argument(
        "--gpu-perf-json",
        type=Path,
        default=None,
        help="Optional explicit gpu_perf_graph.json path (otherwise inferred from run_summary stage).",
    )
    ap.add_argument(
        "--require-mlir-fresh-on-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require MLIR full196 evidence to be validated on current HEAD.",
    )
    ap.add_argument(
        "--require-mlir-toolchain-required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require current_status.mlir_toolchain_ok=true (and cutover not blocked_toolchain).",
    )
    ap.add_argument(
        "--require-mlir-llvm-artifact-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require LLVM artifact evidence for MLIR migration runs.",
    )
    ap.add_argument(
        "--require-stage",
        action="append",
        default=[],
        help="Required stage name(s) in run_summary.stages for ir_arch/backend_compiler profiles.",
    )
    ap.add_argument(
        "--ir-max-complex-single-intent-ratio",
        type=float,
        default=0.10,
        help="ir_arch gate threshold for complex-family single-intent ratio (default: 0.10).",
    )
    ap.add_argument(
        "--ir-max-global-unique-single-primitive-ratio",
        type=float,
        default=0.40,
        help="ir_arch gate threshold for global unique single-primitive ratio (default: 0.40).",
    )
    ap.add_argument(
        "--require-active-dual-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every active semantic op to be dual_pass (default: true).",
    )
    ap.add_argument(
        "--require-all-categories-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For coverage profile, require category aggregate completion metadata in run_summary.",
    )
    ap.add_argument(
        "--max-total-regression-pct",
        type=float,
        default=8.0,
        help="Max allowed total_ms regression percentage for backend_compiler timing_delta checks.",
    )
    ap.add_argument(
        "--min-regression-delta-ms",
        type=float,
        default=50.0,
        help="Minimum total_ms delta to count as a performance regression sample.",
    )
    ap.add_argument(
        "--max-regression-ratio",
        type=float,
        default=0.5,
        help="Max allowed ratio of regression samples (per backend) in timing_delta.",
    )
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "batch_gate.json"))
    args = ap.parse_args()

    profile = str(args.profile)
    active_batch = Path(args.active_batch) if args.active_batch is not None else _default_active_batch(profile)
    checks: list[dict[str, Any]] = []
    for p in (active_batch, args.run_summary, args.status_converged, args.progress_log, args.handoff):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))
    require_mlir_fresh = bool(args.require_mlir_fresh_on_head) or profile == "mlir_migration"
    require_mlir_toolchain = bool(args.require_mlir_toolchain_required) or profile == "mlir_migration"
    require_mlir_llvm_artifact = bool(args.require_mlir_llvm_artifact_complete) or profile == "mlir_migration"
    require_gpu_perf_fresh = bool(args.require_gpu_perf_fresh_on_head)
    require_gpu_perf_categories = bool(args.require_gpu_perf_categories_complete)
    if (
        bool(args.require_coverage_fresh_on_head)
        or require_mlir_fresh
        or require_mlir_toolchain
        or require_gpu_perf_fresh
        or require_gpu_perf_categories
    ):
        checks.append(
            _check(
                f"exists::{_to_repo_rel(args.current_status)}",
                args.current_status.is_file(),
                "file exists" if args.current_status.is_file() else "missing file",
            )
        )

    active = _load_json(active_batch) if active_batch.is_file() else {}
    run_summary = _load_json(args.run_summary) if args.run_summary.is_file() else {}
    status_converged = _load_json(args.status_converged) if args.status_converged.is_file() else {}

    run_ok = bool(run_summary.get("ok"))
    checks.append(_check("run_summary.ok", run_ok, "run summary reports ok=true" if run_ok else "run summary reports failure"))

    entries = [e for e in (status_converged.get("entries") or []) if isinstance(e, dict)]
    scope_enabled = bool(status_converged.get("scope_enabled"))
    scoped_entries_active = [e for e in (status_converged.get("scoped_entries_active") or []) if isinstance(e, dict)]
    scoped_entries = [e for e in (status_converged.get("scoped_entries") or []) if isinstance(e, dict)]
    if scope_enabled and scoped_entries_active:
        gate_entries = scoped_entries_active
    elif not scoped_entries and scope_enabled:
        scoped_entries = [e for e in entries if bool(e.get("in_scope"))]
        gate_entries = scoped_entries
    else:
        gate_entries = scoped_entries if scope_enabled else entries

    active_items = [e for e in (active.get("items") or []) if isinstance(e, dict)]
    active_ops = [str(e.get("semantic_op") or "") for e in active_items if str(e.get("semantic_op") or "")]

    if profile in {"gpu_perf", "ir_arch", "backend_compiler", "workflow", "mlir_migration"}:
        reason_complete = True
    else:
        if not gate_entries and not active_ops:
            reason_complete = True
        else:
            reason_complete = all(
                isinstance(e.get("reason_code"), str) and str(e.get("reason_code")).strip() for e in gate_entries
            )
    checks.append(
        _check(
            "status_converged.reason_code_complete",
            reason_complete,
            "all gate entries have non-empty reason_code" if reason_complete else "missing reason_code detected in gate scope",
        )
    )

    if profile == "coverage":
        if bool(args.require_all_categories_complete):
            expected = run_summary.get("coverage_batches_expected")
            completed = run_summary.get("coverage_batches_completed")
            failed = list(run_summary.get("coverage_batches_failed") or [])
            evidence_kind = str(run_summary.get("full196_evidence_kind") or "")
            coverage_mode = str(run_summary.get("coverage_mode") or "")
            try:
                expected_i = int(expected)
                completed_i = int(completed)
            except Exception:
                expected_i = -1
                completed_i = -1
            all_complete = (
                evidence_kind == "batch_aggregate"
                and coverage_mode == "category_batches"
                and expected_i > 0
                and completed_i == expected_i
                and len(failed) == 0
            )
            checks.append(
                _check(
                    "coverage_categories_complete",
                    all_complete,
                    (
                        f"categories complete: {completed_i}/{expected_i} (batch aggregate)"
                        if all_complete
                        else (
                            "coverage category aggregate incomplete "
                            f"(mode={coverage_mode}, evidence={evidence_kind}, completed={completed_i}, "
                            f"expected={expected_i}, failed={failed})"
                        )
                    ),
                )
            )
        status_map = {str(e.get("semantic_op")): e for e in gate_entries if isinstance(e.get("semantic_op"), str)}
        covered = all(op in status_map for op in active_ops)
        checks.append(
            _check(
                "active_batch.covered_by_status",
                covered,
                "all active ops present in scoped status report"
                if (covered and scope_enabled)
                else ("all active ops present in status report" if covered else "active op missing in gate status scope"),
            )
        )
        left_blocked_ir = [op for op in active_ops if str((status_map.get(op) or {}).get("status")) == "blocked_ir"]
        checks.append(
            _check(
                "active_batch.leave_blocked_ir",
                not left_blocked_ir,
                "active batch no longer blocked_ir" if not left_blocked_ir else f"still blocked_ir: {left_blocked_ir}",
            )
        )
        if bool(args.require_active_dual_pass):
            non_dual = [op for op in active_ops if str((status_map.get(op) or {}).get("status")) != "dual_pass"]
            checks.append(
                _check(
                    "active_batch.all_dual_pass",
                    not non_dual,
                    "all active ops are dual_pass" if not non_dual else f"non dual_pass active ops: {non_dual}",
                )
            )
    elif profile == "gpu_perf":
        stage_map = _stage_map(run_summary)
        perf_stage = stage_map.get("gpu_perf_graph") or {}
        stage_path_raw = str(perf_stage.get("json_path") or "").strip()
        perf_path = args.gpu_perf_json
        if perf_path is None:
            perf_path = Path(stage_path_raw) if stage_path_raw else Path()
        if perf_path and (not perf_path.is_absolute()):
            perf_path = ROOT / perf_path
        checks.append(
            _check(
                "gpu_perf.stage_present",
                bool(perf_stage),
                "gpu_perf_graph stage found in run_summary"
                if bool(perf_stage)
                else "missing gpu_perf_graph stage in run_summary",
            )
        )
        if perf_path:
            perf_ok, perf_detail = _validate_gpu_perf_json(
                Path(perf_path),
                threshold=float(args.gpu_perf_threshold),
                require_categories_complete=bool(require_gpu_perf_categories),
            )
        else:
            perf_ok, perf_detail = (False, "gpu perf json path is not provided and cannot be inferred")
        checks.append(_check("gpu_perf.payload_ok", perf_ok, perf_detail))
    elif profile == "ir_arch":
        stage_map = _stage_map(run_summary)
        required = list(
            args.require_stage
            or ["primitive_reuse", "macro_composition", "mapping_complexity", "mapping_tests", "intentir_semantics"]
        )
        missing_or_fail = [s for s in required if not bool((stage_map.get(s) or {}).get("ok"))]
        checks.append(
            _check(
                "ir_arch.required_stage_ok",
                not missing_or_fail,
                "required IR architecture stages passed"
                if not missing_or_fail
                else f"missing or failed ir_arch stage(s): {missing_or_fail}",
            )
        )
        mapping_stage = stage_map.get("mapping_complexity") or {}
        mapping_json_path = str(mapping_stage.get("json_path") or "").strip()
        if not mapping_json_path:
            checks.append(
                _check(
                    "ir_arch.mapping_quality_thresholds",
                    False,
                    "mapping_complexity stage missing json_path",
                )
            )
        else:
            ok_map, detail_map = _validate_ir_arch_mapping_quality(
                Path(mapping_json_path),
                max_complex_single_intent_ratio=float(args.ir_max_complex_single_intent_ratio),
                max_global_unique_single_primitive_ratio=float(args.ir_max_global_unique_single_primitive_ratio),
            )
            checks.append(_check("ir_arch.mapping_quality_thresholds", ok_map, detail_map))
    elif profile == "backend_compiler":
        stage_map = _stage_map(run_summary)
        stage_timing_row = stage_map.get("stage_timing_breakdown") or {}
        stage_timing_json = str(stage_timing_row.get("json_path") or "").strip()
        stage_timing_path = Path(stage_timing_json) if stage_timing_json else Path("")

        def _backend_key_for_stage(stage_name: str) -> str:
            if stage_name.startswith("rvv"):
                return "rvv"
            if stage_name.startswith("cuda"):
                return "cuda"
            return ""

        def _can_use_breakdown(stage_name: str) -> bool:
            backend_key = _backend_key_for_stage(stage_name)
            if not backend_key or not stage_timing_json:
                return False
            return _breakdown_backend_has_evidence(stage_timing_path, backend_key)

        required = list(args.require_stage or ["rvv_remote", "cuda_local"])
        missing_or_fail: list[str] = []
        resolved_required: dict[str, tuple[str, dict[str, Any]]] = {}
        for stage_name in required:
            resolved_name, resolved_row = _resolve_backend_stage(stage_map, stage_name)
            resolved_required[stage_name] = (resolved_name, resolved_row)
            if not bool(resolved_row.get("ok")) and not _can_use_breakdown(resolved_name):
                missing_or_fail.append(f"{stage_name}->{resolved_name}")
        checks.append(
            _check(
                "backend_compiler.required_stage_ok",
                not missing_or_fail,
                "required backend stages passed"
                if not missing_or_fail
                else f"missing or failed backend stage(s): {missing_or_fail}",
            )
        )
        timing_failures: list[str] = []
        timing_required_stages = {"rvv_local", "rvv_remote", "cuda_local", "cuda"}
        for stage_name in required:
            resolved_name, stage = resolved_required.get(stage_name, (stage_name, {}))
            if resolved_name not in timing_required_stages:
                continue
            json_path = str(stage.get("json_path") or "").strip()
            if not json_path:
                if _can_use_breakdown(resolved_name):
                    continue
                timing_failures.append(f"{stage_name}->{resolved_name}:missing_json_path")
                continue
            ok_timing, detail_timing = _validate_backend_timing(Path(json_path))
            if not ok_timing:
                timing_failures.append(f"{stage_name}->{resolved_name}:{detail_timing}")
        checks.append(
            _check(
                "backend_compiler.stage_timing_complete",
                not timing_failures,
                "backend timing fields complete" if not timing_failures else "; ".join(timing_failures),
            )
        )
        if not stage_timing_json:
            checks.append(
                _check(
                    "backend_compiler.stage_timing_breakdown_complete",
                    False,
                    "missing stage_timing_breakdown stage/json_path",
                )
            )
        else:
            breakdown_ok, breakdown_detail = _validate_stage_timing_breakdown(Path(stage_timing_json))
            checks.append(
                _check(
                    "backend_compiler.stage_timing_breakdown_complete",
                    breakdown_ok,
                    breakdown_detail,
                )
            )
        realized_failures: list[str] = []
        for stage_name in required:
            resolved_name, stage = resolved_required.get(stage_name, (stage_name, {}))
            if resolved_name not in timing_required_stages:
                continue
            json_path = str(stage.get("json_path") or "").strip()
            if not json_path:
                if _can_use_breakdown(resolved_name):
                    continue
                realized_failures.append(f"{stage_name}->{resolved_name}:missing_json_path")
                continue
            ok_realized, detail_realized = _validate_backend_stage_realized(Path(json_path))
            if not ok_realized:
                realized_failures.append(f"{stage_name}->{resolved_name}:{detail_realized}")
        checks.append(
            _check(
                "backend_compiler.pipeline_stage_realized",
                not realized_failures,
                "backend compile/launch stages are realized"
                if not realized_failures
                else "; ".join(realized_failures),
            )
        )
        purity_ok, purity_detail = _validate_codegen_purity(stage_map)
        checks.append(
            _check(
                "backend_compiler.compiler_purity.no_fallback_path",
                purity_ok,
                purity_detail,
            )
        )
        timing_delta_stage = stage_map.get("timing_delta") or {}
        timing_delta_json = str(timing_delta_stage.get("json_path") or "").strip()
        if timing_delta_json:
            budget_ok, budget_detail = _validate_timing_delta_budget(
                Path(timing_delta_json),
                max_total_regression_pct=float(args.max_total_regression_pct),
                min_regression_delta_ms=float(args.min_regression_delta_ms),
                max_regression_ratio=float(args.max_regression_ratio),
            )
        else:
            budget_ok, budget_detail = (True, "timing_delta stage missing; regression budget skipped")
        checks.append(
            _check(
                "backend_compiler.performance_budget",
                budget_ok,
                budget_detail,
            )
        )
    elif profile in {"workflow", "mlir_migration"}:
        stage_map = _stage_map(run_summary)
        required = list(args.require_stage or [])
        missing_or_fail = [s for s in required if not bool((stage_map.get(s) or {}).get("ok"))]
        checks.append(
            _check(
                f"{profile}.required_stage_ok",
                not missing_or_fail,
                f"required {profile} stages passed"
                if not missing_or_fail
                else f"missing or failed {profile} stage(s): {missing_or_fail}",
            )
        )

    handoff_text = args.handoff.read_text(encoding="utf-8") if args.handoff.is_file() else ""
    has_next_focus = ("Next Focus:" in handoff_text)
    checks.append(_check("handoff.has_next_focus", has_next_focus, "handoff contains Next Focus section" if has_next_focus else "missing Next Focus in handoff"))

    progress_ok = False
    progress_detail = "progress log missing"
    if args.progress_log.is_file():
        raw_lines = [ln for ln in args.progress_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if raw_lines:
            try:
                last = json.loads(raw_lines[-1])
                run_path_ok = str(last.get("run_summary_path") or "") == _to_repo_rel(args.run_summary)
                status_path_ok = str(last.get("status_converged_path") or "") == _to_repo_rel(args.status_converged)
                progress_ok = bool(run_path_ok and status_path_ok)
                progress_detail = (
                    "progress log tail points to current run artifacts"
                    if progress_ok
                    else "progress tail does not reference current run/status artifacts"
                )
            except Exception:
                progress_ok = False
                progress_detail = "progress log tail is not valid JSON"
    if bool(args.require_progress_tail_match):
        checks.append(_check("progress_log.tail_matches_artifacts", progress_ok, progress_detail))
    else:
        checks.append(
            _check(
                "progress_log.tail_matches_artifacts",
                True,
                "skipped by --no-require-progress-tail-match",
            )
        )

    if bool(args.require_coverage_fresh_on_head):
        fresh_ok, fresh_detail = _validate_coverage_fresh_on_head(args.current_status)
        checks.append(_check("coverage_fresh_on_head", fresh_ok, fresh_detail))
    if require_gpu_perf_fresh:
        gpu_fresh_ok, gpu_fresh_detail = _validate_gpu_perf_fresh_on_head(args.current_status)
        checks.append(_check("gpu_perf_fresh_on_head", gpu_fresh_ok, gpu_fresh_detail))
    if require_gpu_perf_categories:
        gpu_cat_ok, gpu_cat_detail = _validate_gpu_perf_categories_complete(args.current_status)
        checks.append(_check("gpu_perf_categories_complete", gpu_cat_ok, gpu_cat_detail))
    if require_mlir_fresh:
        mlir_ok, mlir_detail = _validate_mlir_fresh_on_head(args.current_status)
        checks.append(_check("mlir_fresh_on_head", mlir_ok, mlir_detail))
    if require_mlir_toolchain:
        tc_ok, tc_detail = _validate_mlir_toolchain_required(args.current_status)
        checks.append(_check("mlir_toolchain_required", tc_ok, tc_detail))
    if require_mlir_llvm_artifact:
        llvm_ok, llvm_detail = _validate_mlir_llvm_artifact_complete(run_summary)
        checks.append(_check("mlir_llvm_artifact_complete", llvm_ok, llvm_detail))

    ok = all(bool(c.get("ok")) for c in checks)
    payload = {
        "ok": bool(ok),
        "profile": profile,
        "active_batch_path": _to_repo_rel(active_batch),
        "run_summary_path": _to_repo_rel(args.run_summary),
        "status_converged_path": _to_repo_rel(args.status_converged),
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Batch gate report written: {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
