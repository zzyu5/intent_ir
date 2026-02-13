"""
Aggregate CI-style gates for FlagGems long-running workflow.
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

from pipeline.triton.flaggems_workflow import load_json, validate_feature_list_sync  # noqa: E402


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": str(name), "ok": bool(ok), "detail": str(detail)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--feature-list", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "feature_list.json"))
    ap.add_argument("--active-batch", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"))
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--status-converged", type=Path, required=True)
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--handoff", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "handoff.md"))
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "ci_gate.json"))
    args = ap.parse_args()

    checks: list[dict[str, Any]] = []
    for p in (args.registry, args.feature_list, args.active_batch, args.run_summary, args.status_converged):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))

    if args.registry.is_file() and args.feature_list.is_file():
        registry_payload = load_json(args.registry)
        feature_payload = load_json(args.feature_list)
        sync_ok, sync_errors = validate_feature_list_sync(
            feature_payload=feature_payload,
            registry_payload=registry_payload,
            expected_source_registry_path=_to_repo_rel(args.registry),
        )
        checks.append(
            _check(
                "feature_list.sync_with_registry",
                sync_ok,
                "feature list is synced with registry" if sync_ok else "; ".join(sync_errors),
            )
        )

    batch_gate_path = args.out.with_name("batch_gate_ci.json")
    cmd = [
        sys.executable,
        "scripts/flaggems/check_batch_gate.py",
        "--active-batch",
        str(args.active_batch),
        "--run-summary",
        str(args.run_summary),
        "--status-converged",
        str(args.status_converged),
        "--progress-log",
        str(args.progress_log),
        "--handoff",
        str(args.handoff),
        "--out",
        str(batch_gate_path),
    ]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    checks.append(
        _check(
            "batch_gate",
            p.returncode == 0,
            "check_batch_gate passed" if p.returncode == 0 else f"check_batch_gate failed: {(p.stderr or p.stdout).strip()}",
        )
    )

    if args.status_converged.is_file():
        converged = load_json(args.status_converged)
        entries = [e for e in (converged.get("entries") or []) if isinstance(e, dict)]
        has_unknown_reason = any(str(e.get("reason_code") or "") in {"", "unknown"} for e in entries)
        checks.append(
            _check(
                "status_converged.no_unknown_reason_code",
                not has_unknown_reason,
                "no unknown reason_code" if not has_unknown_reason else "unknown reason_code present",
            )
        )

    ok = all(bool(c.get("ok")) for c in checks)
    payload = {
        "ok": bool(ok),
        "checks": checks,
        "artifacts": {
            "registry": _to_repo_rel(args.registry),
            "feature_list": _to_repo_rel(args.feature_list),
            "active_batch": _to_repo_rel(args.active_batch),
            "run_summary": _to_repo_rel(args.run_summary),
            "status_converged": _to_repo_rel(args.status_converged),
            "batch_gate": _to_repo_rel(batch_gate_path),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CI gate report written: {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
