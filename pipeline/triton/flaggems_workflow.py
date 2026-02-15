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


LANES: tuple[str, ...] = ("coverage", "ir_arch", "backend_compiler")


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
    entries = list(registry_payload.get("entries") or [])
    features: list[dict[str, Any]] = []
    for e in entries:
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
                "id": f"flaggems::{str(e.get('semantic_op') or 'unknown')}",
                "semantic_op": str(e.get("semantic_op") or ""),
                "family": str(e.get("family") or ""),
                "status": status,
                "passes": bool(status == "dual_pass"),
                "reason_code": reason,
                "next_action": next_action,
                "e2e_spec": e.get("e2e_spec"),
                "intent_ops": list(e.get("intent_ops") or []),
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
    lane_batch_paths: Mapping[str, str] | None = None,
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
    return {
        "schema_version": "flaggems_current_status_v1",
        "updated_at": utc_now_iso(),
        "branch": str(branch),
        "head_commit": str(head_commit),
        "mode": mode,
        "coverage": {
            "semantic_ops": semantic_ops,
            "dual_pass": dual_pass,
            "blocked_ir": blocked_ir,
            "blocked_backend": blocked_backend,
        },
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
) -> dict[str, Any]:
    return {
        "schema_version": "flaggems_session_context_v1",
        "updated_at": utc_now_iso(),
        "read_order": [
            "workflow/flaggems/state/current_status.json",
            "workflow/flaggems/state/session_context.json",
            "workflow/flaggems/state/active_batch_coverage.json",
            "workflow/flaggems/state/active_batch_ir_arch.json",
            "workflow/flaggems/state/active_batch_backend_compiler.json",
            "workflow/flaggems/state/handoff.md",
        ],
        "git_log_short": str(git_log_short),
        "progress_tail": list(progress_tail),
        "next_focus": str(next_focus),
        "known_risks": list(known_risks),
    }
