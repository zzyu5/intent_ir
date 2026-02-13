"""
Start a FlagGems coding session by validating and summarizing active batch state.

Batch planning is intentionally handled by `scripts/flaggems/plan_next_batch.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.flaggems_workflow import load_json  # noqa: E402


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-batch", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"))
    ap.add_argument("--require-non-empty", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--print-json", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    if not args.active_batch.is_file():
        raise FileNotFoundError(
            f"active batch not found: {args.active_batch}. "
            "Run `python scripts/flaggems/plan_next_batch.py` first."
        )
    active = load_json(args.active_batch)
    items = list(active.get("items") or [])
    if bool(args.require_non_empty) and not items:
        raise RuntimeError(
            "active batch is empty; no pending blocked_ir/blocked_backend items. "
            "Regenerate feature list and re-plan batch if this is unexpected."
        )

    summary = {
        "schema_version": str(active.get("schema_version") or ""),
        "generated_at": str(active.get("generated_at") or ""),
        "branch": str(active.get("branch") or ""),
        "batch_size": int(active.get("batch_size") or 0),
        "selected_items": len(items),
        "active_batch_path": _to_repo_rel(args.active_batch),
    }
    if bool(args.print_json):
        print(json.dumps({"summary": summary, "items": items}, indent=2, ensure_ascii=False))
        return

    print(f"Session context ready: {args.active_batch}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if items:
        print("Ops:", ", ".join(str(x.get("semantic_op")) for x in items))


if __name__ == "__main__":
    main()
