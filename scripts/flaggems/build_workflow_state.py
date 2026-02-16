"""
Build current workflow state snapshots for long-running sessions.
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

from pipeline.triton.providers.flaggems.workflow import (
    build_current_status_payload,
    build_session_context_payload,
    dump_json,
    load_json,
    read_git_log,
)


FULL196_VALIDATED_SCOPE = "coverage_158_kernels_to_196_semantics"


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _git(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        return ""
    return str(p.stdout or "").strip()


def _parse_next_focus(handoff_path: Path) -> str:
    if not handoff_path.is_file():
        return ""
    for line in handoff_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("- Next Focus:"):
            return line.split(":", 1)[1].strip()
    return ""


def _known_risks(progress_tail: list[dict[str, Any]]) -> list[str]:
    risks: list[str] = []
    for row in progress_tail:
        if not isinstance(row, dict):
            continue
        if bool(row.get("run_ok")):
            continue
        summary = str(row.get("summary") or "").strip()
        if summary:
            risks.append(summary)
    dedup: list[str] = []
    for risk in risks:
        if risk not in dedup:
            dedup.append(risk)
    return dedup[:5]


def _next_focus_by_lane(progress_tail: list[dict[str, Any]]) -> dict[str, str]:
    by_lane: dict[str, str] = {}
    for row in progress_tail:
        if not isinstance(row, dict):
            continue
        lane = str(row.get("lane") or "coverage").strip()
        if not lane:
            lane = "coverage"
        focus = str(row.get("next_focus") or "").strip()
        if focus:
            by_lane[lane] = focus
    return by_lane


def _active_lanes(feature_payload: dict[str, Any]) -> list[str]:
    features = [f for f in list(feature_payload.get("features") or []) if isinstance(f, dict)]
    lane_pending: dict[str, int] = {}
    for row in features:
        lane = str(row.get("track") or "coverage").strip() or "coverage"
        status = str(row.get("status") or "").strip()
        passes = bool(row.get("passes"))
        pending = (not passes) and status not in {"dual_pass", "done"}
        if pending:
            lane_pending[lane] = int(lane_pending.get(lane, 0)) + 1
    return sorted([lane for lane, cnt in lane_pending.items() if cnt > 0])


def _load_progress_rows(progress_log_path: Path) -> list[dict[str, Any]]:
    if not progress_log_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in progress_log_path.read_text(encoding="utf-8").splitlines():
        line_s = str(line).strip()
        if not line_s:
            continue
        try:
            row = json.loads(line_s)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _resolve_artifact(path_raw: str) -> Path | None:
    path_s = str(path_raw or "").strip()
    if not path_s:
        return None
    p = Path(path_s)
    if not p.is_absolute():
        p = ROOT / p
    return p


def _load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or (not path.is_file()):
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _classify_full196_run(run_summary_path: Path | None) -> tuple[bool, bool | None, dict[str, Any]]:
    run_summary = _load_json_if_exists(run_summary_path)
    if not run_summary:
        return False, None, {}
    suite = str(run_summary.get("suite") or "").strip()
    scope_kernels = list(run_summary.get("scope_kernels") or [])
    if suite != "coverage" or (len(scope_kernels) == 0):
        return False, None, {}
    stages = [s for s in list(run_summary.get("stages") or []) if isinstance(s, dict)]
    coverage_stage = next((s for s in stages if str(s.get("stage") or "") == "coverage_integrity"), None)
    if coverage_stage is None:
        return False, None, {}
    if str(coverage_stage.get("reason_code") or "").strip() == "skipped_partial_scope":
        return False, None, {}
    if "full_coverage_run" in coverage_stage and not bool(coverage_stage.get("full_coverage_run")):
        return False, None, {}
    coverage_json = _resolve_artifact(str(coverage_stage.get("json_path") or ""))
    coverage_payload = _load_json_if_exists(coverage_json)
    coverage_ok = bool(coverage_payload.get("coverage_integrity_ok"))
    stage_map = {str(s.get("stage") or ""): s for s in stages}
    # Use coverage_integrity as the source of truth for full196 health.
    # run_summary.ok can be false for non-functional governance mismatches.
    metadata = {
        "validated_mode": str(run_summary.get("intentir_mode") or ""),
        "validated_scope": FULL196_VALIDATED_SCOPE,
        "validated_with_rvv_remote": bool((stage_map.get("rvv_remote") or {}).get("ok")),
    }
    return True, bool(coverage_ok), metadata


def _latest_full196_from_progress(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        run_summary_path = _resolve_artifact(str(row.get("run_summary_path") or ""))
        is_full, is_ok, metadata = _classify_full196_run(run_summary_path)
        if not is_full or run_summary_path is None:
            continue
        return {
            "run_summary_path": _to_repo_rel(run_summary_path),
            "last_ok": is_ok,
            "validated_commit": str(row.get("commit") or ""),
            "validated_mode": str(metadata.get("validated_mode") or ""),
            "validated_scope": str(metadata.get("validated_scope") or ""),
            "validated_with_rvv_remote": bool(metadata.get("validated_with_rvv_remote")),
        }
    return {
        "run_summary_path": "",
        "last_ok": None,
        "validated_commit": "",
        "validated_mode": "",
        "validated_scope": "",
        "validated_with_rvv_remote": None,
    }


def _commits_since_validated(validated_commit: str, head_commit: str) -> int | None:
    validated = str(validated_commit or "").strip()
    head = str(head_commit or "").strip()
    if not validated or not head:
        return None
    p = subprocess.run(
        ["git", "rev-list", "--count", f"{validated}..{head}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return None
    try:
        return int(str(p.stdout or "").strip())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-list", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--handoff", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "handoff.md"))
    ap.add_argument("--active-batch-coverage", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_coverage.json"))
    ap.add_argument("--active-batch-ir-arch", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_ir_arch.json"))
    ap.add_argument(
        "--active-batch-backend-compiler",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_backend_compiler.json"),
    )
    ap.add_argument("--current-status-out", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "current_status.json"))
    ap.add_argument("--session-context-out", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "session_context.json"))
    ap.add_argument("--scripts-catalog", type=Path, default=(ROOT / "scripts" / "CATALOG.json"))
    ap.add_argument("--git-log-lines", type=int, default=20)
    args = ap.parse_args()

    feature_payload = load_json(args.feature_list)
    progress_rows = _load_progress_rows(args.progress_log)
    progress_tail = list(progress_rows[-8:])
    latest = progress_tail[-1] if progress_tail else {}
    latest_run_summary = str(latest.get("run_summary_path") or "")
    latest_status_converged = str(latest.get("status_converged_path") or "")
    full196_info = _latest_full196_from_progress(progress_rows)
    full196_run_summary = str(full196_info.get("run_summary_path") or "")
    full196_last_ok = full196_info.get("last_ok")
    full196_validated_commit = str(full196_info.get("validated_commit") or "")
    full196_validated_mode = str(full196_info.get("validated_mode") or "")
    full196_validated_scope = str(full196_info.get("validated_scope") or "")
    full196_validated_with_rvv_remote = full196_info.get("validated_with_rvv_remote")
    branch = _git(["git", "branch", "--show-current"]) or "unknown"
    head_commit = _git(["git", "rev-parse", "HEAD"]) or "unknown"
    full196_commits_since_validated = _commits_since_validated(full196_validated_commit, head_commit)
    git_log_short = read_git_log(cwd=ROOT, lines=int(args.git_log_lines))
    next_focus = _parse_next_focus(args.handoff) or str(latest.get("next_focus") or "")
    active_lanes = _active_lanes(feature_payload)
    next_focus_by_lane = _next_focus_by_lane(progress_tail)
    catalog_exists = bool(args.scripts_catalog.is_file())
    if not full196_run_summary:
        coverage_integrity_phase = "recompute_pending"
    elif full196_commits_since_validated and int(full196_commits_since_validated) > 0:
        coverage_integrity_phase = "recompute_stale"
    else:
        coverage_integrity_phase = "recomputed_ok" if bool(full196_last_ok) else "recomputed_failed"
    if (full196_commits_since_validated is not None) and int(full196_commits_since_validated) > 0:
        active_lanes = sorted(set(list(active_lanes) + ["coverage"]))
        next_focus_by_lane.setdefault("coverage", "Run full196 force_compile matrix to refresh coverage evidence on HEAD.")

    current_status = build_current_status_payload(
        branch=branch,
        head_commit=head_commit,
        feature_payload=feature_payload,
        latest_run_summary_path=latest_run_summary,
        latest_status_converged_path=latest_status_converged,
        full196_run_summary_path=full196_run_summary,
        coverage_integrity_phase=coverage_integrity_phase,
        full196_last_ok=full196_last_ok,
        full196_validated_commit=full196_validated_commit,
        full196_commits_since_validated=full196_commits_since_validated,
        full196_validated_mode=full196_validated_mode,
        full196_validated_scope=full196_validated_scope,
        full196_validated_with_rvv_remote=full196_validated_with_rvv_remote,
        catalog_path=_to_repo_rel(args.scripts_catalog),
        catalog_validated=catalog_exists,
        active_lanes=active_lanes,
        next_focus_by_lane=next_focus_by_lane,
        lane_batch_paths={
            "coverage": _to_repo_rel(args.active_batch_coverage),
            "ir_arch": _to_repo_rel(args.active_batch_ir_arch),
            "backend_compiler": _to_repo_rel(args.active_batch_backend_compiler),
        },
    )
    session_context = build_session_context_payload(
        git_log_short=git_log_short,
        progress_tail=progress_tail,
        next_focus=next_focus,
        known_risks=_known_risks(progress_tail),
        must_read_scripts_catalog=_to_repo_rel(args.scripts_catalog),
        active_lanes=active_lanes,
        next_focus_by_lane=next_focus_by_lane,
    )

    out_status = dump_json(args.current_status_out, current_status)
    out_context = dump_json(args.session_context_out, session_context)
    print(f"Current status updated: {out_status}")
    print(f"Session context updated: {out_context}")


if __name__ == "__main__":
    main()
