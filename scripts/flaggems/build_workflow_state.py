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
    load_progress_tail,
    read_git_log,
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
    ap.add_argument("--git-log-lines", type=int, default=20)
    args = ap.parse_args()

    feature_payload = load_json(args.feature_list)
    progress_tail = load_progress_tail(progress_log_path=args.progress_log, lines=8)
    latest = progress_tail[-1] if progress_tail else {}
    latest_run_summary = str(latest.get("run_summary_path") or "")
    latest_status_converged = str(latest.get("status_converged_path") or "")
    branch = _git(["git", "branch", "--show-current"]) or "unknown"
    head_commit = _git(["git", "rev-parse", "HEAD"]) or "unknown"
    git_log_short = read_git_log(cwd=ROOT, lines=int(args.git_log_lines))
    next_focus = _parse_next_focus(args.handoff) or str(latest.get("next_focus") or "")

    current_status = build_current_status_payload(
        branch=branch,
        head_commit=head_commit,
        feature_payload=feature_payload,
        latest_run_summary_path=latest_run_summary,
        latest_status_converged_path=latest_status_converged,
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
    )

    out_status = dump_json(args.current_status_out, current_status)
    out_context = dump_json(args.session_context_out, session_context)
    print(f"Current status updated: {out_status}")
    print(f"Session context updated: {out_context}")


if __name__ == "__main__":
    main()
