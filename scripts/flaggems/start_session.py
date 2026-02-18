"""
Start a FlagGems coding session by validating and summarizing active batch state.

Batch planning is intentionally handled by `scripts/flaggems/plan_next_batch.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.providers.flaggems.workflow import load_json, normalize_lane  # noqa: E402


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lane",
        choices=["coverage", "ir_arch", "backend_compiler", "workflow", "mlir_migration"],
        default="coverage",
        help="Session lane (default: coverage).",
    )
    ap.add_argument("--active-batch", type=Path, default=None)
    ap.add_argument(
        "--current-status",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "current_status.json"),
        help="Current status snapshot required for quick session onboarding.",
    )
    ap.add_argument(
        "--session-context",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "session_context.json"),
        help="Session context snapshot with read-order and handoff focus.",
    )
    ap.add_argument(
        "--require-read-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require current_status + session_context to exist and be valid.",
    )
    ap.add_argument("--require-non-empty", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--print-json", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()
    lane = normalize_lane(str(args.lane))
    default_active = ROOT / "workflow" / "flaggems" / "state" / f"active_batch_{lane}.json"
    active_batch = Path(args.active_batch) if args.active_batch is not None else default_active

    if not active_batch.is_file():
        raise FileNotFoundError(
            f"active batch not found: {active_batch}. "
            "Run `python scripts/flaggems/plan_next_batch.py --lane <lane>` first."
        )
    active = load_json(active_batch)
    items = list(active.get("items") or [])
    if bool(args.require_non_empty) and not items:
        raise RuntimeError(
            "active batch is empty; no pending items in this lane. "
            "Regenerate feature list and re-plan batch if this is unexpected."
        )
    current_status: dict[str, Any] = {}
    session_context: dict[str, Any] = {}
    if bool(args.require_read_context):
        if not args.current_status.is_file():
            raise FileNotFoundError(f"missing current status snapshot: {args.current_status}")
        if not args.session_context.is_file():
            raise FileNotFoundError(f"missing session context snapshot: {args.session_context}")
        current_status = load_json(args.current_status)
        session_context = load_json(args.session_context)
        if str(current_status.get("schema_version") or "") != "flaggems_current_status_v1":
            raise RuntimeError("invalid current_status schema_version")
        if str(session_context.get("schema_version") or "") != "flaggems_session_context_v1":
            raise RuntimeError("invalid session_context schema_version")
        if not isinstance(session_context.get("read_order"), list):
            raise RuntimeError("session_context.read_order must be a list")

    summary = {
        "schema_version": str(active.get("schema_version") or ""),
        "generated_at": str(active.get("generated_at") or ""),
        "branch": str(active.get("branch") or ""),
        "lane": lane,
        "batch_size": int(active.get("batch_size") or 0),
        "selected_items": len(items),
        "active_batch_path": _to_repo_rel(active_batch),
        "mode": str(current_status.get("mode") or ""),
        "next_focus": str(session_context.get("next_focus") or ""),
        "current_status_path": _to_repo_rel(args.current_status),
        "session_context_path": _to_repo_rel(args.session_context),
    }
    if bool(args.print_json):
        print(
            json.dumps(
                {
                    "summary": summary,
                    "items": items,
                    "read_order": list(session_context.get("read_order") or []),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    print(f"Session context ready: {active_batch}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if items:
        names = [str(x.get("semantic_op") or x.get("id") or "") for x in items]
        print("Items:", ", ".join(n for n in names if n))


if __name__ == "__main__":
    main()
