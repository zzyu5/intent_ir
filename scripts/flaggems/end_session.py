"""
Finalize one FlagGems coding session and write handoff artifacts.
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
    append_progress_log,
    load_json,
    normalize_lane,
    utc_now_iso,
    write_handoff,
)


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _head_commit(cwd: Path) -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        return "unknown"
    return str(p.stdout or "").strip() or "unknown"


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lane",
        choices=["coverage", "ir_arch", "backend_compiler"],
        default="coverage",
        help="Session lane (default: coverage).",
    )
    ap.add_argument("--active-batch", type=Path, default=None)
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--handoff", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "handoff.md"))
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--status-converged", type=Path, required=True)
    ap.add_argument("--summary", required=True, help="One-line summary for this session.")
    ap.add_argument("--next-focus", default="", help="Optional explicit next focus.")
    ap.add_argument(
        "--evidence",
        action="append",
        default=[],
        help="Additional evidence paths (repeatable) for this lane.",
    )
    ap.add_argument("--commit", default=None, help="Override commit SHA (default: HEAD).")
    args = ap.parse_args()
    lane = normalize_lane(str(args.lane))
    default_active = ROOT / "workflow" / "flaggems" / "state" / f"active_batch_{lane}.json"
    if lane == "coverage" and not default_active.is_file():
        default_active = ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"
    active_batch = Path(args.active_batch) if args.active_batch is not None else default_active

    if not args.run_summary.is_file():
        raise FileNotFoundError(f"run summary not found: {args.run_summary}")
    if not args.status_converged.is_file():
        raise FileNotFoundError(f"status converged not found: {args.status_converged}")

    active = load_json(active_batch) if active_batch.is_file() else {"items": []}
    run_summary = _load_optional_json(args.run_summary)
    converged = _load_optional_json(args.status_converged)
    commit = str(args.commit) if args.commit else _head_commit(ROOT)

    items = list(active.get("items") or [])
    item_names = [str(x.get("semantic_op")) for x in items]

    entry: dict[str, Any] = {
        "ts": utc_now_iso(),
        "commit": commit,
        "lane": lane,
        "summary": str(args.summary),
        "batch_ops": item_names,
        "active_batch_path": _to_repo_rel(active_batch),
        "run_summary_path": _to_repo_rel(args.run_summary),
        "status_converged_path": _to_repo_rel(args.status_converged),
        "next_focus": str(args.next_focus or ""),
        "evidence_paths": sorted(
            set(
                [
                    _to_repo_rel(args.run_summary),
                    _to_repo_rel(args.status_converged),
                    *[str(x).strip() for x in list(args.evidence or []) if str(x).strip()],
                ]
            )
        ),
    }
    if isinstance(run_summary, dict):
        entry["run_ok"] = bool(run_summary.get("ok"))
        entry["run_out_dir"] = run_summary.get("out_dir")
    if isinstance(converged, dict):
        entry["counts_by_status"] = converged.get("counts_by_status")

    append_progress_log(progress_log_path=args.progress_log, entry=entry)

    handoff = [
        "# FlagGems Session Handoff",
        "",
        f"- Timestamp: {entry['ts']}",
        f"- Commit: `{commit}`",
        f"- Lane: `{lane}`",
        f"- Summary: {args.summary}",
        f"- Batch Ops ({len(item_names)}): {', '.join(item_names) if item_names else '(none)'}",
        f"- Run Summary: `{args.run_summary}`",
        f"- Status Converged: `{args.status_converged}`",
        f"- Evidence Paths: {', '.join(entry['evidence_paths']) if entry['evidence_paths'] else '(none)'}",
    ]
    if str(args.next_focus or "").strip():
        handoff.append(f"- Next Focus: {args.next_focus}")
    write_handoff(handoff_path=args.handoff, content="\n".join(handoff) + "\n")

    print(f"Progress log appended: {args.progress_log}")
    print(f"Handoff updated: {args.handoff}")


if __name__ == "__main__":
    main()
