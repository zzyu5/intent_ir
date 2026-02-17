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
IR_ARCH_COMPLEX_SINGLE_RATIO_THRESHOLD = 0.10
IR_ARCH_GLOBAL_UNIQUE_SINGLE_RATIO_THRESHOLD = 0.40
FULL196_COVERAGE_RULE = (
    "full196 is valid only when all coverage categories complete and aggregate coverage integrity passes on current HEAD"
)


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


def _mapping_quality_needs_ir_arch(feature_payload: dict[str, Any]) -> tuple[bool, str]:
    features = [f for f in list(feature_payload.get("features") or []) if isinstance(f, dict)]
    coverage_rows = [f for f in features if str(f.get("track") or "coverage") == "coverage"]
    total = 0
    complex_total = 0
    complex_single = 0
    unique_single_primitives: set[str] = set()
    complex_families = {
        "index_scatter_gather",
        "conv_pool_interp",
        "matmul_linear",
        "attention_sequence",
        "reduction",
        "norm_activation",
    }
    for row in coverage_rows:
        total += 1
        fam = str(row.get("family") or "unknown")
        ops = [str(x) for x in list(row.get("intent_ops") or []) if str(x).strip()]
        if len(ops) == 1:
            unique_single_primitives.add(str(ops[0]))
        if fam in complex_families:
            complex_total += 1
            if len(ops) == 1:
                complex_single += 1
    complex_ratio = (float(complex_single) / float(complex_total)) if complex_total > 0 else 0.0
    global_unique_ratio = (float(len(unique_single_primitives)) / float(total)) if total > 0 else 0.0
    if complex_ratio > float(IR_ARCH_COMPLEX_SINGLE_RATIO_THRESHOLD):
        return True, (
            f"mapping quality breach: complex_family_single_semantic_ratio={complex_ratio:.4f} "
            f"> {IR_ARCH_COMPLEX_SINGLE_RATIO_THRESHOLD:.4f}"
        )
    if global_unique_ratio > float(IR_ARCH_GLOBAL_UNIQUE_SINGLE_RATIO_THRESHOLD):
        return True, (
            f"mapping quality breach: global_unique_single_primitive_ratio={global_unique_ratio:.4f} "
            f"> {IR_ARCH_GLOBAL_UNIQUE_SINGLE_RATIO_THRESHOLD:.4f}"
        )
    return False, ""


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


def _infer_category_batch_rvv_remote_ok(run_summary: dict[str, Any]) -> bool:
    """Infer rvv_remote execution status for category-batch aggregate runs.

    Aggregate run summaries intentionally store per-family run paths instead of
    forwarding backend stages. Use those family run summaries to recover a
    trustworthy rvv_remote signal for workflow freshness display.
    """
    families = [x for x in list(run_summary.get("families") or []) if isinstance(x, dict)]
    if not families:
        families = [x for x in list(run_summary.get("family_runs") or []) if isinstance(x, dict)]
    if not families:
        return False
    for family in families:
        family_run_path = _resolve_artifact(str(family.get("run_summary_path") or ""))
        family_run_payload = _load_json_if_exists(family_run_path)
        family_stages = [s for s in list(family_run_payload.get("stages") or []) if isinstance(s, dict)]
        family_stage_map = {str(s.get("stage") or ""): s for s in family_stages}
        if bool((family_stage_map.get("rvv_remote") or {}).get("ok")):
            continue
        # Chunked family summaries may not expose backend stages at family root.
        # In that case, require every chunk run summary to pass rvv_remote.
        chunk_runs = [x for x in list(family_run_payload.get("chunk_runs") or []) if isinstance(x, dict)]
        if not chunk_runs:
            return False
        for chunk in chunk_runs:
            chunk_run_path = _resolve_artifact(str(chunk.get("run_summary_path") or ""))
            chunk_run_payload = _load_json_if_exists(chunk_run_path)
            chunk_stages = [s for s in list(chunk_run_payload.get("stages") or []) if isinstance(s, dict)]
            chunk_stage_map = {str(s.get("stage") or ""): s for s in chunk_stages}
            if not bool((chunk_stage_map.get("rvv_remote") or {}).get("ok")):
                return False
    return True


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
    rvv_remote_ok = bool((stage_map.get("rvv_remote") or {}).get("ok"))
    if not rvv_remote_ok and str(run_summary.get("coverage_mode") or "") == "category_batches":
        rvv_remote_ok = _infer_category_batch_rvv_remote_ok(run_summary)
    # Use coverage_integrity as the source of truth for full196 health.
    # run_summary.ok can be false for non-functional governance mismatches.
    metadata = {
        "validated_mode": str(run_summary.get("intentir_mode") or ""),
        "validated_scope": FULL196_VALIDATED_SCOPE,
        "validated_with_rvv_remote": bool(rvv_remote_ok),
        "coverage_mode": str(run_summary.get("coverage_mode") or "single_run"),
        "full196_evidence_kind": str(run_summary.get("full196_evidence_kind") or "single_run"),
        "coverage_batches_expected": run_summary.get("coverage_batches_expected"),
        "coverage_batches_completed": run_summary.get("coverage_batches_completed"),
        "coverage_batches_failed": list(run_summary.get("coverage_batches_failed") or []),
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
            "coverage_mode": str(metadata.get("coverage_mode") or "single_run"),
            "full196_evidence_kind": str(metadata.get("full196_evidence_kind") or "single_run"),
            "coverage_batches_expected": metadata.get("coverage_batches_expected"),
            "coverage_batches_completed": metadata.get("coverage_batches_completed"),
            "coverage_batches_failed": list(metadata.get("coverage_batches_failed") or []),
        }
    return {
        "run_summary_path": "",
        "last_ok": None,
        "validated_commit": "",
        "validated_mode": "",
        "validated_scope": "",
        "validated_with_rvv_remote": None,
        "coverage_mode": "single_run",
        "full196_evidence_kind": "single_run",
        "coverage_batches_expected": None,
        "coverage_batches_completed": None,
        "coverage_batches_failed": [],
    }


def _commit_exists(commit: str) -> bool:
    c = str(commit or "").strip()
    if not c:
        return False
    p = subprocess.run(
        ["git", "cat-file", "-e", f"{c}^{{commit}}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return p.returncode == 0


def _is_ancestor(base_commit: str, head_commit: str) -> bool:
    b = str(base_commit or "").strip()
    h = str(head_commit or "").strip()
    if not b or not h:
        return False
    p = subprocess.run(
        ["git", "merge-base", "--is-ancestor", b, h],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return p.returncode == 0


def _as_int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


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
    ap.add_argument(
        "--coverage-batches",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "coverage_batches.json"),
    )
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
    coverage_mode = str(full196_info.get("coverage_mode") or "single_run")
    full196_evidence_kind = str(full196_info.get("full196_evidence_kind") or "single_run")
    coverage_batches_expected = _as_int_or_none(full196_info.get("coverage_batches_expected"))
    coverage_batches_completed = _as_int_or_none(full196_info.get("coverage_batches_completed"))
    coverage_batches_failed = [str(x) for x in list(full196_info.get("coverage_batches_failed") or []) if str(x).strip()]
    branch = _git(["git", "branch", "--show-current"]) or "unknown"
    head_commit = _git(["git", "rev-parse", "HEAD"]) or "unknown"
    coverage_batches_payload = load_json(args.coverage_batches) if args.coverage_batches.is_file() else {}
    if coverage_batches_expected is None:
        try:
            coverage_batches_expected = int(len(list(coverage_batches_payload.get("family_order") or [])))
        except Exception:
            coverage_batches_expected = None

    validated_commit_state = "missing"
    if full196_validated_commit:
        if not _commit_exists(full196_validated_commit):
            validated_commit_state = "invalid"
        elif not _is_ancestor(full196_validated_commit, head_commit):
            validated_commit_state = "invalid"
        else:
            validated_commit_state = "reachable"
    full196_commits_since_validated = (
        _commits_since_validated(full196_validated_commit, head_commit)
        if validated_commit_state == "reachable"
        else None
    )
    if validated_commit_state == "reachable" and full196_commits_since_validated is not None and int(full196_commits_since_validated) > 0:
        validated_commit_state = "stale"
    elif validated_commit_state == "reachable":
        validated_commit_state = "fresh"
    git_log_short = read_git_log(cwd=ROOT, lines=int(args.git_log_lines))
    next_focus = _parse_next_focus(args.handoff) or str(latest.get("next_focus") or "")
    active_lanes = _active_lanes(feature_payload)
    next_focus_by_lane = _next_focus_by_lane(progress_tail)
    need_ir_arch_lane, ir_arch_reason = _mapping_quality_needs_ir_arch(feature_payload)
    catalog_exists = bool(args.scripts_catalog.is_file())
    if not full196_run_summary:
        coverage_integrity_phase = "recompute_pending"
    elif validated_commit_state == "invalid":
        coverage_integrity_phase = "stale_or_invalid"
    elif validated_commit_state == "stale":
        coverage_integrity_phase = "recompute_stale"
    else:
        coverage_integrity_phase = "recomputed_ok" if bool(full196_last_ok) else "recomputed_failed"
    if validated_commit_state in {"invalid", "stale"}:
        active_lanes = sorted(set(list(active_lanes) + ["coverage"]))
        next_focus_by_lane.setdefault(
            "coverage",
            "Run coverage categories (7/7) with force_compile and aggregate full196 evidence on HEAD.",
        )
    if need_ir_arch_lane:
        active_lanes = sorted(set(list(active_lanes) + ["ir_arch"]))
        next_focus_by_lane.setdefault(
            "ir_arch",
            "Reduce IR single-op mapping complexity to satisfy complex<=10% and global_unique<=40% thresholds.",
        )
        if not next_focus:
            next_focus = ir_arch_reason

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
        coverage_mode=coverage_mode,
        coverage_batches_expected=coverage_batches_expected,
        coverage_batches_completed=coverage_batches_completed,
        coverage_batches_failed=coverage_batches_failed,
        full196_evidence_kind=full196_evidence_kind,
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
        full196_coverage_rule=FULL196_COVERAGE_RULE,
    )

    out_status = dump_json(args.current_status_out, current_status)
    out_context = dump_json(args.session_context_out, session_context)
    print(f"Current status updated: {out_status}")
    print(f"Session context updated: {out_context}")


if __name__ == "__main__":
    main()
