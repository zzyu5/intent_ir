"""
Workflow helpers for long-running FlagGems integration sessions.

This module is script-friendly and test-friendly: scripts under `scripts/flaggems/`
call these helpers to keep JSON contracts stable and reduce duplicate logic.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def summarize_registry(registry_payload: dict[str, Any]) -> dict[str, Any]:
    entries = list(registry_payload.get("entries") or [])
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for e in entries:
        s = str(e.get("status") or "unknown")
        f = str(e.get("family") or "unknown")
        by_status[s] = int(by_status.get(s, 0)) + 1
        by_family[f] = int(by_family.get(f, 0)) + 1
    return {
        "semantic_ops": len(entries),
        "by_status": by_status,
        "by_family": by_family,
    }


def build_feature_list_payload(*, registry_payload: dict[str, Any], source_registry_path: str) -> dict[str, Any]:
    entries = list(registry_payload.get("entries") or [])
    features: list[dict[str, Any]] = []
    for e in entries:
        status = str(e.get("status") or "unknown")
        reason = str(e.get("status_reason") or "")
        if status == "blocked_ir":
            next_action = "semantic_mapping"
        elif status == "blocked_backend" and reason == "missing_e2e_spec":
            next_action = "add_e2e_spec"
        elif status == "blocked_backend":
            next_action = "backend_lowering"
        else:
            next_action = "none"
        features.append(
            {
                "id": f"flaggems::{str(e.get('semantic_op') or 'unknown')}",
                "semantic_op": str(e.get("semantic_op") or ""),
                "family": str(e.get("family") or ""),
                "status": status,
                "passes": bool(status == "dual_pass"),
                "reason_code": reason,
                "next_action": next_action,
                "e2e_spec": e.get("e2e_spec"),
                "intent_ops": list(e.get("intent_ops") or []),
            }
        )

    return {
        "schema_version": "flaggems_feature_list_v1",
        "generated_at": utc_now_iso(),
        "source_registry_path": str(source_registry_path),
        "summary": summarize_registry(registry_payload),
        "features": features,
    }


def select_next_batch(*, feature_payload: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    feats = list(feature_payload.get("features") or [])
    n = max(1, int(batch_size))

    def _priority(f: dict[str, Any]) -> tuple[int, str]:
        status = str(f.get("status") or "")
        reason = str(f.get("reason_code") or "")
        name = str(f.get("semantic_op") or "")
        if status == "blocked_ir":
            return (0, name)
        if status == "blocked_backend" and reason == "missing_e2e_spec":
            return (1, name)
        if status == "blocked_backend":
            return (2, name)
        return (3, name)

    pending = [f for f in feats if str(f.get("status")) in {"blocked_ir", "blocked_backend"}]
    pending.sort(key=_priority)
    return pending[:n]


@dataclass(frozen=True)
class BaselineSnapshot:
    semantic_ops: int
    dual_pass: int
    blocked_ir: int
    blocked_backend: int
    coverage_specs: int


def _extract_baseline_metrics(registry_payload: dict[str, Any], coverage_specs: Iterable[str]) -> BaselineSnapshot:
    summary = summarize_registry(registry_payload)
    by_status = dict(summary.get("by_status") or {})
    return BaselineSnapshot(
        semantic_ops=int(summary.get("semantic_ops", 0)),
        dual_pass=int(by_status.get("dual_pass", 0)),
        blocked_ir=int(by_status.get("blocked_ir", 0)),
        blocked_backend=int(by_status.get("blocked_backend", 0)),
        coverage_specs=len(list(coverage_specs)),
    )


def freeze_baseline_snapshot(
    *,
    baselines_dir: Path,
    registry_payload: dict[str, Any],
    coverage_specs: Iterable[str],
    status_converged_path: Path | None = None,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics = _extract_baseline_metrics(registry_payload, coverage_specs)
    payload: dict[str, Any] = {
        "schema_version": "flaggems_baseline_v1",
        "captured_at": utc_now_iso(),
        "metrics": {
            "semantic_ops": metrics.semantic_ops,
            "dual_pass": metrics.dual_pass,
            "blocked_ir": metrics.blocked_ir,
            "blocked_backend": metrics.blocked_backend,
            "coverage_specs": metrics.coverage_specs,
        },
        "status_converged_path": str(status_converged_path) if status_converged_path is not None else None,
    }
    out_dir = Path(baselines_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = out_dir / f"registry_baseline_{ts}.json"
    latest = out_dir / "registry_baseline_latest.json"
    dump_json(stamped, payload)
    dump_json(latest, payload)
    return stamped


def read_git_log(*, cwd: Path, lines: int = 20) -> str:
    cmd = ["git", "log", "--oneline", f"-n{max(1, int(lines))}"]
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        return ""
    return str(p.stdout or "").strip()


def append_progress_log(*, progress_log_path: Path, entry: dict[str, Any]) -> None:
    p = Path(progress_log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_handoff(*, handoff_path: Path, content: str) -> Path:
    p = Path(handoff_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p

