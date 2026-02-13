"""
Create an active FlagGems batch for the next coding session.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.flaggems_workflow import load_json, read_git_log, select_next_batch, utc_now_iso, dump_json


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _git_branch(cwd: Path) -> str:
    p = subprocess.run(["git", "branch", "--show-current"], cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        return "unknown"
    return str(p.stdout or "").strip() or "unknown"


def _tail_progress(progress_log: Path, lines: int = 5) -> list[str]:
    if not progress_log.is_file():
        return []
    raw = progress_log.read_text(encoding="utf-8").strip().splitlines()
    return raw[-max(1, int(lines)) :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-list", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--active-batch", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"))
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--git-log-lines", type=int, default=20)
    args = ap.parse_args()

    payload = load_json(args.feature_list)
    batch = select_next_batch(feature_payload=payload, batch_size=int(args.batch_size))
    branch = _git_branch(ROOT)
    git_log = read_git_log(cwd=ROOT, lines=int(args.git_log_lines))
    progress_tail = _tail_progress(args.progress_log, lines=5)

    active = {
        "schema_version": "flaggems_active_batch_v1",
        "generated_at": utc_now_iso(),
        "branch": branch,
        "batch_size": int(args.batch_size),
        "selection_policy": ["blocked_ir", "missing_e2e_spec", "backend_missing_ops"],
        "items": batch,
        "context": {
            "feature_list_path": _to_repo_rel(args.feature_list),
            "progress_log_path": _to_repo_rel(args.progress_log),
            "git_log": git_log,
            "progress_tail": progress_tail,
        },
    }
    out = dump_json(args.active_batch, active)
    print(f"Active batch written: {out}")
    print(f"Selected {len(batch)} ops on branch {branch}")
    if batch:
        print("Ops:", ", ".join(str(x.get("semantic_op")) for x in batch))


if __name__ == "__main__":
    main()
