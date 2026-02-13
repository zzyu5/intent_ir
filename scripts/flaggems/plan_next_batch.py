"""
Select the next FlagGems active batch from registry-driven feature truth.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.providers.flaggems.workflow import (  # noqa: E402
    build_active_batch_payload,
    dump_json,
    load_json,
    read_git_log,
    select_next_batch,
    validate_feature_list_sync,
)


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
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--feature-list", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--active-batch", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"))
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--git-log-lines", type=int, default=20)
    ap.add_argument("--strict-sync", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    registry_payload = load_json(args.registry)
    feature_payload = load_json(args.feature_list)
    source_registry_path = _to_repo_rel(args.registry)
    sync_ok, sync_errors = validate_feature_list_sync(
        feature_payload=feature_payload,
        registry_payload=registry_payload,
        expected_source_registry_path=source_registry_path,
    )
    if not sync_ok and bool(args.strict_sync):
        raise SystemExit("feature_list is out-of-sync with registry: " + "; ".join(sync_errors))

    batch = select_next_batch(feature_payload=feature_payload, batch_size=int(args.batch_size))
    branch = _git_branch(ROOT)
    git_log = read_git_log(cwd=ROOT, lines=int(args.git_log_lines))
    progress_tail = _tail_progress(args.progress_log, lines=5)

    active = build_active_batch_payload(
        batch=batch,
        branch=branch,
        batch_size=int(args.batch_size),
        feature_list_path=_to_repo_rel(args.feature_list),
        progress_log_path=_to_repo_rel(args.progress_log),
        git_log=git_log,
        progress_tail=progress_tail,
    )
    out = dump_json(args.active_batch, active)
    print(f"Active batch planned: {out}")
    print(f"Selected {len(batch)} ops on branch {branch}")
    if batch:
        print("Ops:", ", ".join(str(x.get("semantic_op")) for x in batch))
    if sync_errors:
        print("Sync warnings:", "; ".join(sync_errors))


if __name__ == "__main__":
    main()
