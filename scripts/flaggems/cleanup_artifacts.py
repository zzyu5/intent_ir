"""
Artifact retention cleanup for FlagGems workflow runs.

Policy goal:
- Keep only validated baseline evidence trees.
- Remove transient caches/tmp trees aggressively.
- Emit auditable cleanup reports before/after execution.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_date_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dump_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _parse_progress_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line).strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _resolve_path(path_raw: str) -> Path | None:
    s = str(path_raw or "").strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _is_descendant_or_same(path: Path, root: Path) -> bool:
    return _is_relative_to(path, root)


def _has_preserve_descendant(path: Path, preserve_roots: list[Path]) -> bool:
    p = path.resolve()
    for keep in preserve_roots:
        try:
            keep.resolve().relative_to(p)
            return True
        except Exception:
            continue
    return False


def _under_preserve_root(path: Path, preserve_roots: list[Path]) -> bool:
    p = path.resolve()
    for keep in preserve_roots:
        if _is_descendant_or_same(p, keep):
            return True
    return False


def _matches_purge_pattern(*, artifacts_root: Path, path: Path, patterns: list[str]) -> bool:
    rel = str(path.resolve().relative_to(artifacts_root.resolve())).replace(os.sep, "/")
    base = path.name
    for pat in patterns:
        p = str(pat or "").strip()
        if not p:
            continue
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(base, p):
            return True
    return False


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        p = subprocess.run(
            ["du", "-sb", str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            first = str(p.stdout or "").strip().split()[0]
            return int(first)
    except Exception:
        pass
    total = 0
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += int(fp.stat().st_size)
            except Exception:
                continue
    return int(total)


def _latest_mlir_wave_run_summary_from_progress(progress_rows: list[dict[str, Any]]) -> str:
    for row in reversed(progress_rows):
        if str(row.get("lane") or "").strip() != "mlir_migration":
            continue
        run_summary_path = str(row.get("run_summary_path") or "").strip()
        if run_summary_path:
            return run_summary_path
    return ""


def _looks_like_mlir_wave_run_dir(run_dir: Path) -> bool:
    parts = [str(x).strip().lower() for x in run_dir.parts]
    for token in parts:
        if "mlir_contract_wave" in token:
            return True
    return False


def _resolve_preserve_roots(
    *,
    artifacts_root: Path,
    policy: dict[str, Any],
    current_status: dict[str, Any],
    progress_rows: list[dict[str, Any]],
    purge_toolchains: bool,
) -> dict[str, Any]:
    keep_run_tokens = [str(x) for x in list(policy.get("keep_runs") or []) if str(x).strip()]
    keep_dirs_raw = [str(x) for x in list(policy.get("keep_dirs") or []) if str(x).strip()]
    require_distinct_mlir_wave = bool(policy.get("require_distinct_latest_mlir_wave"))
    if purge_toolchains:
        keep_dirs_raw = [x for x in keep_dirs_raw if not str(x).startswith("artifacts/toolchains")]

    run_summary_by_token: dict[str, str] = {
        "full196_validated": str(current_status.get("full196_last_run") or ""),
        "gpu_perf_validated": str(current_status.get("gpu_perf_last_run") or ""),
        "latest_mlir_wave": str(
            ((current_status.get("latest_artifacts") or {}).get("run_summary") or "")
            or _latest_mlir_wave_run_summary_from_progress(progress_rows)
        ),
    }

    preserve_roots: list[Path] = []
    preserve_meta: list[dict[str, Any]] = []
    strict_errors: list[str] = []

    for token in keep_run_tokens:
        run_summary_raw = str(run_summary_by_token.get(token) or "").strip()
        if not run_summary_raw:
            preserve_meta.append(
                {
                    "kind": "keep_run",
                    "token": str(token),
                    "ok": False,
                    "reason": "missing_run_summary_path",
                }
            )
            continue
        run_summary_path = _resolve_path(run_summary_raw)
        if run_summary_path is None:
            preserve_meta.append(
                {
                    "kind": "keep_run",
                    "token": str(token),
                    "ok": False,
                    "reason": "invalid_run_summary_path",
                    "run_summary": str(run_summary_raw),
                }
            )
            continue
        run_dir = run_summary_path.parent.resolve()
        if not _is_relative_to(run_dir, artifacts_root):
            preserve_meta.append(
                {
                    "kind": "keep_run",
                    "token": str(token),
                    "ok": False,
                    "reason": "run_dir_outside_artifacts_root",
                    "run_summary": _to_repo_rel(run_summary_path),
                }
            )
            continue
        if token == "latest_mlir_wave" and require_distinct_mlir_wave and (not _looks_like_mlir_wave_run_dir(run_dir)):
            preserve_meta.append(
                {
                    "kind": "keep_run",
                    "token": str(token),
                    "ok": False,
                    "reason": "latest_mlir_wave_not_distinct_mlir_contract_wave",
                    "run_summary": _to_repo_rel(run_summary_path),
                    "run_dir": _to_repo_rel(run_dir),
                }
            )
            strict_errors.append(
                "policy requires distinct latest_mlir_wave, but resolved path is not under mlir_contract_wave*: "
                f"{_to_repo_rel(run_dir)}"
            )
            continue
        preserve_roots.append(run_dir)
        preserve_meta.append(
            {
                "kind": "keep_run",
                "token": str(token),
                "ok": True,
                "run_summary": _to_repo_rel(run_summary_path),
                "run_dir": _to_repo_rel(run_dir),
                "run_summary_exists": bool(run_summary_path.is_file()),
            }
        )

    for keep_raw in keep_dirs_raw:
        keep_path = _resolve_path(keep_raw)
        if keep_path is None:
            continue
        if not _is_relative_to(keep_path, artifacts_root):
            continue
        preserve_roots.append(keep_path)
        preserve_meta.append(
            {
                "kind": "keep_dir",
                "token": str(keep_raw),
                "ok": True,
                "run_dir": _to_repo_rel(keep_path),
                "exists": bool(keep_path.exists()),
            }
        )

    dedup: dict[str, Path] = {}
    for p in preserve_roots:
        dedup[str(p.resolve())] = p.resolve()
    keep = sorted(dedup.values(), key=lambda x: str(x))
    return {
        "preserve_roots": keep,
        "preserve_meta": preserve_meta,
        "strict_errors": strict_errors,
    }


def _collect_cleanup_operations(
    *,
    artifacts_root: Path,
    preserve_roots: list[Path],
    purge_patterns: list[str],
) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _push(path: Path, *, reason: str) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        ops.append(
            {
                "path": str(path.resolve()),
                "relpath": _to_repo_rel(path.resolve()),
                "is_dir": bool(path.is_dir()),
                "reason": str(reason),
            }
        )

    def _purge_inside_preserved(root: Path) -> None:
        if not root.exists() or not root.is_dir():
            return
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if _matches_purge_pattern(artifacts_root=artifacts_root, path=child, patterns=purge_patterns):
                _push(child, reason="purge_pattern")
                continue
            if child.is_dir():
                _purge_inside_preserved(child)

    def _walk_container(container: Path) -> None:
        if not container.exists() or not container.is_dir():
            return
        for child in sorted(container.iterdir(), key=lambda p: p.name):
            child_r = child.resolve()
            if _under_preserve_root(child_r, preserve_roots):
                # Child is inside a preserve root; only purge configured patterns.
                if child.is_dir():
                    _purge_inside_preserved(child_r)
                else:
                    if _matches_purge_pattern(artifacts_root=artifacts_root, path=child_r, patterns=purge_patterns):
                        _push(child_r, reason="purge_pattern")
                continue
            if _has_preserve_descendant(child_r, preserve_roots):
                # Ancestor container toward a preserve root.
                if child.is_dir():
                    _walk_container(child_r)
                continue
            reason = "purge_pattern" if _matches_purge_pattern(
                artifacts_root=artifacts_root, path=child_r, patterns=purge_patterns
            ) else "prune_not_kept"
            _push(child_r, reason=reason)

    _walk_container(artifacts_root.resolve())
    return ops


def _delete_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def build_cleanup_plan(
    *,
    artifacts_root: Path,
    policy: dict[str, Any],
    current_status: dict[str, Any],
    progress_rows: list[dict[str, Any]],
    purge_toolchains: bool,
) -> dict[str, Any]:
    artifacts_root = artifacts_root.resolve()
    purge_patterns = [str(x) for x in list(policy.get("purge_patterns") or []) if str(x).strip()]
    keep_info = _resolve_preserve_roots(
        artifacts_root=artifacts_root,
        policy=policy,
        current_status=current_status,
        progress_rows=progress_rows,
        purge_toolchains=bool(purge_toolchains),
    )
    preserve_roots = list(keep_info.get("preserve_roots") or [])
    ops = _collect_cleanup_operations(
        artifacts_root=artifacts_root,
        preserve_roots=preserve_roots,
        purge_patterns=purge_patterns,
    )
    return {
        "schema_version": "flaggems_cleanup_plan_v1",
        "generated_at": _utc_iso(),
        "artifacts_root": _to_repo_rel(artifacts_root),
        "policy": {
            "schema_version": str(policy.get("schema_version") or ""),
            "mode": str(policy.get("mode") or ""),
            "keep_runs": [str(x) for x in list(policy.get("keep_runs") or [])],
            "keep_dirs": [str(x) for x in list(policy.get("keep_dirs") or [])],
            "purge_patterns": list(purge_patterns),
            "require_distinct_latest_mlir_wave": bool(policy.get("require_distinct_latest_mlir_wave")),
        },
        "preserve_roots": [_to_repo_rel(Path(p)) for p in preserve_roots],
        "preserve_meta": list(keep_info.get("preserve_meta") or []),
        "strict_errors": list(keep_info.get("strict_errors") or []),
        "delete_candidates_count": int(len(ops)),
        "delete_candidates": list(ops),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "artifact_retention_policy.json"),
    )
    ap.add_argument(
        "--current-status",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "current_status.json"),
    )
    ap.add_argument(
        "--progress-log",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"),
    )
    ap.add_argument("--artifacts-root", type=Path, default=(ROOT / "artifacts"))
    ap.add_argument(
        "--reports-root",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "cleanup_reports"),
    )
    ap.add_argument("--execute", action="store_true", help="Apply deletions. Default is dry-run.")
    ap.add_argument("--dry-run", action="store_true", help="Force dry-run mode.")
    ap.add_argument(
        "--purge-toolchains",
        action="store_true",
        help="Allow cleanup to remove artifacts/toolchains too.",
    )
    args = ap.parse_args()

    if bool(args.execute) and bool(args.dry_run):
        raise SystemExit("--execute and --dry-run are mutually exclusive")
    execute = bool(args.execute) and (not bool(args.dry_run))

    policy = _load_json(args.policy)
    if str(policy.get("schema_version") or "") != "flaggems_artifact_retention_policy_v1":
        raise SystemExit(f"unexpected policy schema_version: {policy.get('schema_version')!r}")
    current_status = _load_json(args.current_status)
    progress_rows = _parse_progress_log(args.progress_log)
    artifacts_root = args.artifacts_root.resolve()

    date_dir = args.reports_root.resolve() / _utc_date_tag()
    date_dir.mkdir(parents=True, exist_ok=True)

    bytes_before = _size_bytes(artifacts_root)
    plan = build_cleanup_plan(
        artifacts_root=artifacts_root,
        policy=policy,
        current_status=current_status,
        progress_rows=progress_rows,
        purge_toolchains=bool(args.purge_toolchains),
    )
    strict_errors = [str(x) for x in list(plan.get("strict_errors") or []) if str(x).strip()]
    if strict_errors:
        detail = "; ".join(strict_errors)
        raise SystemExit(f"cleanup policy strict validation failed: {detail}")
    plan["mode"] = "execute" if execute else "dry_run"
    plan["bytes_before"] = int(bytes_before)

    plan_path = _dump_json(date_dir / "plan.json", plan)

    deleted_path = date_dir / "deleted.jsonl"
    deleted_path.parent.mkdir(parents=True, exist_ok=True)
    with deleted_path.open("w", encoding="utf-8") as f:
        for row in list(plan.get("delete_candidates") or []):
            entry = {
                "ts": _utc_iso(),
                "mode": ("execute" if execute else "dry_run"),
                "path": str(row.get("path") or ""),
                "relpath": str(row.get("relpath") or ""),
                "reason": str(row.get("reason") or ""),
                "is_dir": bool(row.get("is_dir")),
                "deleted": False,
                "error": "",
            }
            if execute:
                try:
                    _delete_path(Path(entry["path"]))
                    entry["deleted"] = True
                except Exception as e:
                    entry["error"] = f"{type(e).__name__}: {e}"
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    bytes_after = _size_bytes(artifacts_root)
    summary = {
        "schema_version": "flaggems_cleanup_summary_v1",
        "generated_at": _utc_iso(),
        "mode": ("execute" if execute else "dry_run"),
        "artifacts_root": _to_repo_rel(artifacts_root),
        "policy_path": _to_repo_rel(args.policy.resolve()),
        "current_status_path": _to_repo_rel(args.current_status.resolve()),
        "progress_log_path": _to_repo_rel(args.progress_log.resolve()),
        "plan_path": _to_repo_rel(plan_path.resolve()),
        "deleted_log_path": _to_repo_rel(deleted_path.resolve()),
        "bytes_before": int(bytes_before),
        "bytes_after": int(bytes_after),
        "bytes_delta": int(bytes_after - bytes_before),
        "deleted_paths_count": int(len(list(plan.get("delete_candidates") or []))),
        "purge_toolchains": bool(args.purge_toolchains),
    }
    summary_path = _dump_json(date_dir / "summary.json", summary)

    print(f"Cleanup plan written: {plan_path}")
    print(f"Cleanup log written: {deleted_path}")
    print(f"Cleanup summary written: {summary_path}")


if __name__ == "__main__":
    main()
