"""
Workflow helpers for long-running FlagGems integration sessions.

This module is script-friendly and test-friendly: scripts under `scripts/flaggems/`
call these helpers to keep JSON contracts stable and reduce duplicate logic.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def summarize_registry(registry_payload: dict[str, Any]) -> dict[str, Any]:
    entries = list(registry_payload.get("entries") or [])
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for e in entries:
        s = str(e.get("status") or "unknown")
        f = str(e.get("family") or "unknown")
        by_status[s] = int(by_status.get(s, 0)) + 1
        by_family[f] = int(by_family.get(f, 0)) + 1
    return {
        "semantic_ops": len(entries),
        "by_status": by_status,
        "by_family": by_family,
    }


LANES: tuple[str, ...] = ("coverage", "ir_arch", "backend_compiler", "workflow", "mlir_migration")


def normalize_lane(raw: str) -> str:
    lane = str(raw or "").strip()
    if lane not in LANES:
        raise ValueError(f"unsupported lane: {lane}")
    return lane


def _normalized_task(task: Mapping[str, Any]) -> dict[str, Any]:
    lane = normalize_lane(str(task.get("track") or "coverage"))
    status = str(task.get("status") or "pending")
    passes = bool(task.get("passes", False))
    if status == "done":
        passes = True
    gate_profile = str(task.get("gate_profile") or lane)
    if gate_profile not in LANES:
        gate_profile = lane
    return {
        "id": str(task.get("id") or ""),
        "semantic_op": str(task.get("semantic_op") or ""),
        "family": str(task.get("family") or "workflow"),
        "status": status,
        "passes": bool(passes),
        "reason_code": str(task.get("reason_code") or ""),
        "next_action": str(task.get("next_action") or "none"),
        "e2e_spec": task.get("e2e_spec"),
        "intent_ops": list(task.get("intent_ops") or []),
        "track": lane,
        "task_type": str(task.get("task_type") or "task"),
        "priority": int(task.get("priority") or 100),
        "acceptance": list(task.get("acceptance") or []),
        "gate_profile": gate_profile,
        "depends_on": list(task.get("depends_on") or []),
        "evidence_paths": list(task.get("evidence_paths") or []),
        "description": str(task.get("description") or ""),
    }


def build_feature_list_payload(
    *,
    registry_payload: dict[str, Any],
    source_registry_path: str,
    manual_tasks: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    from pipeline.triton.providers.flaggems.semantic_rules import resolve_semantic_mapping

    entries = list(registry_payload.get("entries") or [])
    features: list[dict[str, Any]] = []
    for e in entries:
        semantic_op = str(e.get("semantic_op") or "")
        mapping = resolve_semantic_mapping(semantic_op) if semantic_op else None
        intent_ops = (
            list(mapping.intent_ops)
            if mapping is not None
            else [str(x) for x in list(e.get("intent_ops") or []) if str(x).strip()]
        )
        mapping_kind = (
            str(mapping.mapping_kind)
            if mapping is not None
            else str(e.get("mapping_kind") or "")
        )
        intent_pattern_id = (
            str(mapping.intent_pattern_id)
            if mapping is not None
            else str(e.get("intent_pattern_id") or "")
        )
        status = str(e.get("status") or "unknown")
        reason = str(e.get("status_reason") or "")
        if status == "blocked_ir":
            next_action = "semantic_mapping"
        elif status == "blocked_backend" and reason == "missing_e2e_spec":
            next_action = "add_e2e_spec"
        elif status == "blocked_backend":
            next_action = "backend_lowering"
        else:
            next_action = "none"
        features.append(
            {
                "id": f"flaggems::{semantic_op or 'unknown'}",
                "semantic_op": semantic_op,
                "family": str(e.get("family") or ""),
                "status": status,
                "passes": bool(status == "dual_pass"),
                "reason_code": reason,
                "next_action": next_action,
                "e2e_spec": e.get("e2e_spec"),
                "intent_ops": list(intent_ops),
                "mapping_kind": mapping_kind,
                "intent_pattern_id": intent_pattern_id,
                "track": "coverage",
                "task_type": "semantic_op",
                "priority": 0,
                "acceptance": [
                    "intentir_mapping_present",
                    "pipeline_diff_ok",
                    "rvv_local_or_remote_pass",
                    "cuda_pass",
                ],
                "gate_profile": "coverage",
                "depends_on": [],
                "evidence_paths": [],
                "description": "",
            }
        )

    for task in list(manual_tasks or []):
        normalized = _normalized_task(task)
        if not normalized["id"]:
            continue
        features.append(normalized)

    by_track: dict[str, int] = {}
    for f in features:
        t = str(f.get("track") or "coverage")
        by_track[t] = int(by_track.get(t, 0)) + 1

    return {
        "schema_version": "flaggems_feature_list_v2",
        "generated_at": utc_now_iso(),
        "source_registry_path": str(source_registry_path),
        "summary": {
            **summarize_registry(registry_payload),
            "tasks_total": len(features),
            "by_track": by_track,
        },
        "features": features,
    }


def validate_feature_list_sync(
    *,
    feature_payload: dict[str, Any],
    registry_payload: dict[str, Any],
    expected_source_registry_path: str | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    feat_summary = dict(feature_payload.get("summary") or {})
    reg_summary = summarize_registry(registry_payload)
    if int(feat_summary.get("semantic_ops", -1)) != int(reg_summary.get("semantic_ops", -2)):
        errors.append("feature_list.summary.semantic_ops mismatches registry")
    if dict(feat_summary.get("by_status") or {}) != dict(reg_summary.get("by_status") or {}):
        errors.append("feature_list.summary.by_status mismatches registry")
    if dict(feat_summary.get("by_family") or {}) != dict(reg_summary.get("by_family") or {}):
        errors.append("feature_list.summary.by_family mismatches registry")
    coverage_features = [f for f in list(feature_payload.get("features") or []) if str(f.get("track") or "coverage") == "coverage"]
    if len(coverage_features) != int(reg_summary.get("semantic_ops", 0)):
        errors.append("feature_list.coverage_features count mismatches registry semantic ops")
    if expected_source_registry_path is not None:
        src = str(feature_payload.get("source_registry_path") or "")
        if src != str(expected_source_registry_path):
            errors.append(
                "feature_list.source_registry_path mismatches expected source "
                f"(expected={expected_source_registry_path}, got={src})"
            )
    return (len(errors) == 0), errors


def build_active_batch_payload(
    *,
    batch: list[dict[str, Any]],
    branch: str,
    batch_size: int,
    lane: str = "coverage",
    feature_list_path: str = "",
    progress_log_path: str = "",
    git_log: str = "",
    progress_tail: list[str] | None = None,
    status_snapshot_path: str = "",
    session_context_path: str = "",
) -> dict[str, Any]:
    lane_norm = normalize_lane(lane)
    return {
        "schema_version": "flaggems_active_batch_v2",
        "generated_at": utc_now_iso(),
        "branch": str(branch),
        "lane": lane_norm,
        "batch_size": int(batch_size),
        "selection_policy": (
            ["blocked_ir", "missing_e2e_spec", "backend_missing_ops"]
            if lane_norm == "coverage"
            else ["status:blocking", "priority", "dependency"]
        ),
        "items": list(batch),
        "context": {
            "feature_list_path": str(feature_list_path),
            "progress_log_path": str(progress_log_path),
            "git_log": str(git_log),
            "progress_tail": list(progress_tail or []),
            "status_snapshot_path": str(status_snapshot_path),
            "session_context_path": str(session_context_path),
        },
    }


def _is_task_pending(f: Mapping[str, Any]) -> bool:
    if bool(f.get("passes")):
        return False
    status = str(f.get("status") or "").strip()
    return status not in {"dual_pass", "done"}


def _is_task_done(f: Mapping[str, Any]) -> bool:
    if bool(f.get("passes")):
        return True
    status = str(f.get("status") or "").strip()
    return status in {"dual_pass", "done"}


def _mapping_quality_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    coverage_rows = [f for f in features if str(f.get("track") or "coverage") == "coverage"]
    by_family: dict[str, dict[str, int]] = {}
    total = 0
    single = 0
    multi = 0
    zero = 0
    complex_families = {
        "index_scatter_gather",
        "conv_pool_interp",
        "matmul_linear",
        "attention_sequence",
        "reduction",
        "norm_activation",
    }
    complex_total = 0
    complex_single = 0
    unique_single_primitives: set[str] = set()
    for row in coverage_rows:
        fam = str(row.get("family") or "unknown")
        ops = [str(x) for x in list(row.get("intent_ops") or []) if str(x).strip()]
        n = len(ops)
        fam_bucket = by_family.setdefault(
            fam,
            {
                "total": 0,
                "single_intent_ops": 0,
                "multi_intent_ops": 0,
                "zero_intent_ops": 0,
            },
        )
        fam_bucket["total"] += 1
        total += 1
        if n == 0:
            fam_bucket["zero_intent_ops"] += 1
            zero += 1
        elif n == 1:
            fam_bucket["single_intent_ops"] += 1
            single += 1
            unique_single_primitives.add(str(ops[0]))
        else:
            fam_bucket["multi_intent_ops"] += 1
            multi += 1
        if fam in complex_families:
            complex_total += 1
            if n == 1:
                complex_single += 1
    single_ratio = (float(single) / float(total)) if total > 0 else 0.0
    complex_single_ratio = (float(complex_single) / float(complex_total)) if complex_total > 0 else 0.0
    global_unique_single_primitive_ratio = (float(len(unique_single_primitives)) / float(total)) if total > 0 else 0.0
    return {
        "coverage_semantic_ops": int(total),
        "single_intent_ops": int(single),
        "multi_intent_ops": int(multi),
        "zero_intent_ops": int(zero),
        # Raw semantic-level ratio (alias kept explicit for gating/reporting clarity).
        "raw_single_semantic_ratio": float(single_ratio),
        "single_intent_ratio": float(single_ratio),
        "global_unique_single_primitive_count": int(len(unique_single_primitives)),
        "global_unique_single_primitive_ratio": float(global_unique_single_primitive_ratio),
        "complex_family_single_semantic_ratio": float(complex_single_ratio),
        "complex_family_single_intent_ratio": float(complex_single_ratio),
        "complex_family_total": int(complex_total),
        "complex_family_single_intent_ops": int(complex_single),
        "by_family": {k: v for k, v in sorted(by_family.items(), key=lambda kv: kv[0])},
    }


def select_next_batch(*, feature_payload: dict[str, Any], batch_size: int, lane: str = "coverage") -> list[dict[str, Any]]:
    feats = list(feature_payload.get("features") or [])
    n = max(1, int(batch_size))
    lane_norm = normalize_lane(lane)

    def _priority(f: dict[str, Any]) -> tuple[int, str]:
        status = str(f.get("status") or "")
        reason = str(f.get("reason_code") or "")
        name = str(f.get("semantic_op") or "")
        if status == "blocked_ir":
            return (0, name)
        if status == "blocked_backend" and reason == "missing_e2e_spec":
            return (1, name)
        if status == "blocked_backend":
            return (2, name)
        return (3, name)

    if lane_norm == "coverage":
        pending = [f for f in feats if str(f.get("track") or "coverage") == "coverage" and _is_task_pending(f)]
        pending.sort(key=_priority)
        return pending[:n]

    blocking_rank = {"blocked": 0, "in_progress": 1, "pending": 2}

    def _task_priority(f: dict[str, Any]) -> tuple[int, int, str]:
        status = str(f.get("status") or "pending")
        rank = int(blocking_rank.get(status, 3))
        prio = int(f.get("priority") or 100)
        name = str(f.get("id") or f.get("semantic_op") or "")
        return (rank, prio, name)

    lane_rows = [
        f
        for f in feats
        if str(f.get("track") or "coverage") == lane_norm
    ]
    by_id = {
        str(f.get("id") or ""): f
        for f in lane_rows
        if str(f.get("id") or "").strip()
    }
    pending = [f for f in lane_rows if _is_task_pending(f)]

    def _deps_ready(f: Mapping[str, Any]) -> bool:
        deps = [str(x).strip() for x in list(f.get("depends_on") or []) if str(x).strip()]
        for dep in deps:
            dep_row = by_id.get(dep)
            if dep_row is None:
                return False
            if not _is_task_done(dep_row):
                return False
        return True

    eligible = [f for f in pending if _deps_ready(f)]
    if eligible:
        pending = eligible
    pending.sort(key=_task_priority)
    return pending[:n]


@dataclass(frozen=True)
class BaselineSnapshot:
    semantic_ops: int
    dual_pass: int
    blocked_ir: int
    blocked_backend: int
    coverage_specs: int


def _extract_baseline_metrics(registry_payload: dict[str, Any], coverage_specs: Iterable[str]) -> BaselineSnapshot:
    summary = summarize_registry(registry_payload)
    by_status = dict(summary.get("by_status") or {})
    return BaselineSnapshot(
        semantic_ops=int(summary.get("semantic_ops", 0)),
        dual_pass=int(by_status.get("dual_pass", 0)),
        blocked_ir=int(by_status.get("blocked_ir", 0)),
        blocked_backend=int(by_status.get("blocked_backend", 0)),
        coverage_specs=len(list(coverage_specs)),
    )


def freeze_baseline_snapshot(
    *,
    baselines_dir: Path,
    registry_payload: dict[str, Any],
    coverage_specs: Iterable[str],
    status_converged_path: Path | None = None,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics = _extract_baseline_metrics(registry_payload, coverage_specs)
    payload: dict[str, Any] = {
        "schema_version": "flaggems_baseline_v1",
        "captured_at": utc_now_iso(),
        "metrics": {
            "semantic_ops": metrics.semantic_ops,
            "dual_pass": metrics.dual_pass,
            "blocked_ir": metrics.blocked_ir,
            "blocked_backend": metrics.blocked_backend,
            "coverage_specs": metrics.coverage_specs,
        },
        "status_converged_path": str(status_converged_path) if status_converged_path is not None else None,
    }
    out_dir = Path(baselines_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = out_dir / f"registry_baseline_{ts}.json"
    latest = out_dir / "registry_baseline_latest.json"
    dump_json(stamped, payload)
    dump_json(latest, payload)
    return stamped


def read_git_log(*, cwd: Path, lines: int = 20) -> str:
    cmd = ["git", "log", "--oneline", f"-n{max(1, int(lines))}"]
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        return ""
    return str(p.stdout or "").strip()


def append_progress_log(*, progress_log_path: Path, entry: dict[str, Any]) -> None:
    p = Path(progress_log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_metrics_history(*, metrics_history_path: Path, entry: dict[str, Any]) -> None:
    p = Path(metrics_history_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_handoff(*, handoff_path: Path, content: str) -> Path:
    p = Path(handoff_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def load_progress_tail(*, progress_log_path: Path, lines: int = 5) -> list[dict[str, Any]]:
    p = Path(progress_log_path)
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    raw_lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in raw_lines[-max(1, int(lines)) :]:
        try:
            val = json.loads(line)
        except Exception:
            continue
        if isinstance(val, dict):
            out.append(val)
    return out


def build_current_status_payload(
    *,
    branch: str,
    head_commit: str,
    feature_payload: Mapping[str, Any],
    latest_run_summary_path: str = "",
    latest_status_converged_path: str = "",
    full196_run_summary_path: str = "",
    lane_batch_paths: Mapping[str, str] | None = None,
    coverage_integrity_phase: str = "recompute_pending",
    full196_last_ok: bool | None = None,
    full196_validated_commit: str = "",
    full196_validated_commit_source: str = "",
    full196_commits_since_validated: int | None = None,
    full196_commits_since_validated_total: int | None = None,
    full196_lifted_to_head: bool = False,
    full196_validated_mode: str = "",
    full196_validated_execution_ir: str = "",
    full196_validated_scope: str = "",
    full196_validated_with_rvv_remote: bool | None = None,
    coverage_mode: str = "single_run",
    coverage_batches_expected: int | None = None,
    coverage_batches_completed: int | None = None,
    coverage_batches_failed: list[str] | None = None,
    full196_evidence_kind: str = "single_run",
    full196_last_run_repo_head_commit: str = "",
    full196_last_run_repo_branch: str = "",
    full196_last_run_dirty: bool | None = None,
    full196_artifact_repo_stamp_ok: bool | None = None,
    mlir_migration_phase: str = "",
    mlir_cutover_level: str = "",
    mlir_default_enabled: bool | None = None,
    mlir_toolchain_ok: bool | None = None,
    mlir_backend_contract_ready: bool | None = None,
    mlir_llvm_chain_ok: bool | None = None,
    mlir_full196_validated_commit: str = "",
    gpu_perf_phase: str = "",
    gpu_perf_last_run: str = "",
    gpu_perf_last_ok: bool | None = None,
    gpu_perf_validated_commit: str = "",
    gpu_perf_commits_since_validated: int | None = None,
    gpu_perf_mode: str = "",
    gpu_perf_threshold: float | None = None,
    gpu_perf_devices: list[dict[str, Any]] | None = None,
    gpu_perf_failures_by_family: Mapping[str, Any] | None = None,
    gpu_perf_categories_expected: int | None = None,
    gpu_perf_categories_completed: int | None = None,
    gpu_perf_categories_failed: list[str] | None = None,
    catalog_path: str = "scripts/CATALOG.json",
    catalog_validated: bool = False,
    active_lanes: list[str] | None = None,
    next_focus_by_lane: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    summary = dict(feature_payload.get("summary") or {})
    by_status = dict(summary.get("by_status") or {})
    semantic_ops = int(summary.get("semantic_ops") or 0)
    dual_pass = int(by_status.get("dual_pass") or 0)
    blocked_ir = int(by_status.get("blocked_ir") or 0)
    blocked_backend = int(by_status.get("blocked_backend") or 0)
    features = [f for f in list(feature_payload.get("features") or []) if isinstance(f, dict)]

    lanes: dict[str, dict[str, int]] = {}
    pending_any = False
    for lane in LANES:
        rows = [f for f in features if str(f.get("track") or "coverage") == lane]
        pending = sum(1 for f in rows if _is_task_pending(f))
        done = sum(1 for f in rows if not _is_task_pending(f))
        lanes[lane] = {"pending": pending, "done": done, "total": len(rows)}
        pending_any = pending_any or pending > 0

    mode = "mixed_development" if pending_any else "maintenance"
    mapping_quality = _mapping_quality_summary(features)
    return {
        "schema_version": "flaggems_current_status_v1",
        "updated_at": utc_now_iso(),
        "branch": str(branch),
        "head_commit": str(head_commit),
        "mode": mode,
        "coverage_integrity_phase": str(coverage_integrity_phase),
        "full196_last_run": str(full196_run_summary_path),
        "full196_last_ok": (None if full196_last_ok is None else bool(full196_last_ok)),
        "full196_validated_commit": str(full196_validated_commit or ""),
        "full196_validated_commit_source": str(full196_validated_commit_source or ""),
        "full196_commits_since_validated": (
            None if full196_commits_since_validated is None else int(full196_commits_since_validated)
        ),
        "full196_commits_since_validated_total": (
            None
            if full196_commits_since_validated_total is None
            else int(full196_commits_since_validated_total)
        ),
        "full196_lifted_to_head": bool(full196_lifted_to_head),
        "full196_validated_mode": str(full196_validated_mode or ""),
        "full196_validated_execution_ir": str(full196_validated_execution_ir or ""),
        "full196_validated_scope": str(full196_validated_scope or ""),
        "full196_validated_with_rvv_remote": (
            None
            if full196_validated_with_rvv_remote is None
            else bool(full196_validated_with_rvv_remote)
        ),
        "coverage_mode": str(coverage_mode or "single_run"),
        "coverage_batches_expected": (
            None if coverage_batches_expected is None else int(coverage_batches_expected)
        ),
        "coverage_batches_completed": (
            None if coverage_batches_completed is None else int(coverage_batches_completed)
        ),
        "coverage_batches_failed": [str(x) for x in list(coverage_batches_failed or []) if str(x).strip()],
        "full196_evidence_kind": str(full196_evidence_kind or "single_run"),
        "full196_last_run_repo_head_commit": str(full196_last_run_repo_head_commit or ""),
        "full196_last_run_repo_branch": str(full196_last_run_repo_branch or ""),
        "full196_last_run_dirty": (None if full196_last_run_dirty is None else bool(full196_last_run_dirty)),
        "full196_artifact_repo_stamp_ok": (
            None if full196_artifact_repo_stamp_ok is None else bool(full196_artifact_repo_stamp_ok)
        ),
        "mlir_migration_phase": str(mlir_migration_phase or ""),
        "mlir_cutover_level": str(mlir_cutover_level or ""),
        "mlir_default_enabled": (None if mlir_default_enabled is None else bool(mlir_default_enabled)),
        "mlir_toolchain_ok": (None if mlir_toolchain_ok is None else bool(mlir_toolchain_ok)),
        "mlir_backend_contract_ready": (
            None if mlir_backend_contract_ready is None else bool(mlir_backend_contract_ready)
        ),
        "mlir_llvm_chain_ok": (None if mlir_llvm_chain_ok is None else bool(mlir_llvm_chain_ok)),
        "mlir_full196_validated_commit": str(mlir_full196_validated_commit or ""),
        "gpu_perf_phase": str(gpu_perf_phase or ""),
        "gpu_perf_last_run": str(gpu_perf_last_run or ""),
        "gpu_perf_last_ok": (None if gpu_perf_last_ok is None else bool(gpu_perf_last_ok)),
        "gpu_perf_validated_commit": str(gpu_perf_validated_commit or ""),
        "gpu_perf_commits_since_validated": (
            None if gpu_perf_commits_since_validated is None else int(gpu_perf_commits_since_validated)
        ),
        "gpu_perf_mode": str(gpu_perf_mode or ""),
        "gpu_perf_threshold": (None if gpu_perf_threshold is None else float(gpu_perf_threshold)),
        "gpu_perf_devices": list(gpu_perf_devices or []),
        "gpu_perf_failures_by_family": {
            str(k): [str(x) for x in list(v or []) if str(x).strip()]
            for k, v in dict(gpu_perf_failures_by_family or {}).items()
        },
        "gpu_perf_categories_expected": (
            None if gpu_perf_categories_expected is None else int(gpu_perf_categories_expected)
        ),
        "gpu_perf_categories_completed": (
            None if gpu_perf_categories_completed is None else int(gpu_perf_categories_completed)
        ),
        "gpu_perf_categories_failed": [
            str(x) for x in list(gpu_perf_categories_failed or []) if str(x).strip()
        ],
        "coverage": {
            "semantic_ops": semantic_ops,
            "dual_pass": dual_pass,
            "blocked_ir": blocked_ir,
            "blocked_backend": blocked_backend,
        },
        "mapping_quality": mapping_quality,
        "script_governance": {
            "catalog_path": str(catalog_path),
            "catalog_validated": bool(catalog_validated),
        },
        "active_lanes": [str(x) for x in list(active_lanes or [])],
        "next_focus_by_lane": {str(k): str(v) for k, v in dict(next_focus_by_lane or {}).items()},
        "latest_artifacts": {
            "run_summary": str(latest_run_summary_path),
            "status_converged": str(latest_status_converged_path),
        },
        "lanes": {
            lane: {
                **metrics,
                "active_batch_path": str((lane_batch_paths or {}).get(lane, "")),
            }
            for lane, metrics in lanes.items()
        },
    }


def build_session_context_payload(
    *,
    git_log_short: str,
    progress_tail: list[dict[str, Any]],
    next_focus: str,
    known_risks: list[str],
    must_read_scripts_catalog: str = "scripts/CATALOG.json",
    active_lanes: list[str] | None = None,
    next_focus_by_lane: Mapping[str, str] | None = None,
    full196_coverage_rule: str = (
        "full196 is valid only after all configured coverage categories complete and aggregate coverage integrity passes"
    ),
) -> dict[str, Any]:
    return {
        "schema_version": "flaggems_session_context_v1",
        "updated_at": utc_now_iso(),
        "read_order": [
            "workflow/flaggems/state/current_status.json",
            "workflow/flaggems/state/session_context.json",
            str(must_read_scripts_catalog),
            "workflow/flaggems/state/active_batch_coverage.json",
            "workflow/flaggems/state/active_batch_ir_arch.json",
            "workflow/flaggems/state/active_batch_backend_compiler.json",
            "workflow/flaggems/state/active_batch_workflow.json",
            "workflow/flaggems/state/active_batch_mlir_migration.json",
            "workflow/flaggems/state/handoff.md",
        ],
        "must_read_scripts_catalog": str(must_read_scripts_catalog),
        "git_log_short": str(git_log_short),
        "progress_tail": list(progress_tail),
        "next_focus": str(next_focus),
        "full196_coverage_rule": str(full196_coverage_rule),
        "active_lanes": [str(x) for x in list(active_lanes or [])],
        "next_focus_by_lane": {str(k): str(v) for k, v in dict(next_focus_by_lane or {}).items()},
        "known_risks": list(known_risks),
    }
