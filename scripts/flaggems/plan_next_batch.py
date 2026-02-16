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
    normalize_lane,
    dump_json,
    load_json,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--feature-list", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument(
        "--lane",
        choices=["coverage", "ir_arch", "backend_compiler"],
        default="coverage",
        help="Batch lane selector (default: coverage).",
    )
    ap.add_argument("--active-batch", type=Path, default=None, help="Optional explicit active batch output path.")
    ap.add_argument(
        "--status-snapshot",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "current_status.json"),
        help="Current status snapshot path recorded into active batch context.",
    )
    ap.add_argument(
        "--session-context",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "session_context.json"),
        help="Session context path recorded into active batch context.",
    )
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument(
        "--progress-log",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"),
        help="Compatibility arg retained; progress history now lives in progress_log.jsonl only.",
    )
    ap.add_argument("--strict-sync", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    lane = normalize_lane(str(args.lane))
    default_active = ROOT / "workflow" / "flaggems" / "state" / f"active_batch_{lane}.json"
    active_batch_path = Path(args.active_batch) if args.active_batch is not None else default_active

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

    batch = select_next_batch(feature_payload=feature_payload, batch_size=int(args.batch_size), lane=lane)
    branch = _git_branch(ROOT)
    active = build_active_batch_payload(
        batch=batch,
        branch=branch,
        batch_size=int(args.batch_size),
        lane=lane,
        feature_list_path=_to_repo_rel(args.feature_list),
        status_snapshot_path=_to_repo_rel(args.status_snapshot),
        session_context_path=_to_repo_rel(args.session_context),
    )
    out = dump_json(active_batch_path, active)
    print(f"Active batch planned: {out}")
    print(f"Selected {len(batch)} items on branch {branch} for lane={lane}")
    if batch:
        names = [str(x.get("semantic_op") or x.get("id") or "") for x in batch]
        print("Items:", ", ".join(n for n in names if n))
    if sync_errors:
        print("Sync warnings:", "; ".join(sync_errors))


if __name__ == "__main__":
    main()
