"""
Validate hard gate artifacts for one FlagGems batch.
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": str(name), "ok": bool(ok), "detail": str(detail)}


def _default_active_batch(profile: str) -> Path:
    if profile == "coverage":
        cov = ROOT / "workflow" / "flaggems" / "state" / "active_batch_coverage.json"
        if cov.is_file():
            return cov
        return ROOT / "workflow" / "flaggems" / "state" / "active_batch.json"
    return ROOT / "workflow" / "flaggems" / "state" / f"active_batch_{profile}.json"


def _stage_map(run_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [r for r in list(run_summary.get("stages") or []) if isinstance(r, dict)]
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("stage") or "").strip()
        if not name:
            continue
        out[name] = row
    return out


def _validate_backend_timing(summary_path: Path) -> tuple[bool, str]:
    if not summary_path.is_file():
        return False, f"backend stage json missing: {summary_path}"
    payload = _load_json(summary_path)
    results = payload.get("results")
    if not isinstance(results, list):
        return False, "backend stage json missing results[]"
    required = {"lower_ms", "compile_ms", "launch_ms", "total_ms"}
    for idx, row in enumerate(results):
        if not isinstance(row, dict):
            return False, f"backend stage result[{idx}] is not object"
        missing = [k for k in required if k not in row]
        if missing:
            return False, f"backend stage result[{idx}] missing timing fields: {missing}"
    return True, "backend stage timing fields present"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        choices=["coverage", "ir_arch", "backend_compiler"],
        default="coverage",
        help="Gate profile to evaluate.",
    )
    ap.add_argument("--active-batch", type=Path, default=None)
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--status-converged", type=Path, required=True)
    ap.add_argument("--progress-log", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "progress_log.jsonl"))
    ap.add_argument("--handoff", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "handoff.md"))
    ap.add_argument(
        "--require-stage",
        action="append",
        default=[],
        help="Required stage name(s) in run_summary.stages for ir_arch/backend_compiler profiles.",
    )
    ap.add_argument(
        "--require-active-dual-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every active semantic op to be dual_pass (default: true).",
    )
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "batch_gate.json"))
    args = ap.parse_args()

    profile = str(args.profile)
    active_batch = Path(args.active_batch) if args.active_batch is not None else _default_active_batch(profile)
    checks: list[dict[str, Any]] = []
    for p in (active_batch, args.run_summary, args.status_converged, args.progress_log, args.handoff):
        checks.append(_check(f"exists::{_to_repo_rel(p)}", p.is_file(), "file exists" if p.is_file() else "missing file"))

    active = _load_json(active_batch) if active_batch.is_file() else {}
    run_summary = _load_json(args.run_summary) if args.run_summary.is_file() else {}
    status_converged = _load_json(args.status_converged) if args.status_converged.is_file() else {}

    run_ok = bool(run_summary.get("ok"))
    checks.append(_check("run_summary.ok", run_ok, "run summary reports ok=true" if run_ok else "run summary reports failure"))

    entries = [e for e in (status_converged.get("entries") or []) if isinstance(e, dict)]
    scope_enabled = bool(status_converged.get("scope_enabled"))
    scoped_entries_active = [e for e in (status_converged.get("scoped_entries_active") or []) if isinstance(e, dict)]
    scoped_entries = [e for e in (status_converged.get("scoped_entries") or []) if isinstance(e, dict)]
    if scope_enabled and scoped_entries_active:
        gate_entries = scoped_entries_active
    elif not scoped_entries and scope_enabled:
        scoped_entries = [e for e in entries if bool(e.get("in_scope"))]
        gate_entries = scoped_entries
    else:
        gate_entries = scoped_entries if scope_enabled else entries

    active_items = [e for e in (active.get("items") or []) if isinstance(e, dict)]
    active_ops = [str(e.get("semantic_op") or "") for e in active_items if str(e.get("semantic_op") or "")]

    if profile in {"ir_arch", "backend_compiler"}:
        reason_complete = True
    else:
        if not gate_entries and not active_ops:
            reason_complete = True
        else:
            reason_complete = all(
                isinstance(e.get("reason_code"), str) and str(e.get("reason_code")).strip() for e in gate_entries
            )
    checks.append(
        _check(
            "status_converged.reason_code_complete",
            reason_complete,
            "all gate entries have non-empty reason_code" if reason_complete else "missing reason_code detected in gate scope",
        )
    )

    if profile == "coverage":
        status_map = {str(e.get("semantic_op")): e for e in gate_entries if isinstance(e.get("semantic_op"), str)}
        covered = all(op in status_map for op in active_ops)
        checks.append(
            _check(
                "active_batch.covered_by_status",
                covered,
                "all active ops present in scoped status report"
                if (covered and scope_enabled)
                else ("all active ops present in status report" if covered else "active op missing in gate status scope"),
            )
        )
        left_blocked_ir = [op for op in active_ops if str((status_map.get(op) or {}).get("status")) == "blocked_ir"]
        checks.append(
            _check(
                "active_batch.leave_blocked_ir",
                not left_blocked_ir,
                "active batch no longer blocked_ir" if not left_blocked_ir else f"still blocked_ir: {left_blocked_ir}",
            )
        )
        if bool(args.require_active_dual_pass):
            non_dual = [op for op in active_ops if str((status_map.get(op) or {}).get("status")) != "dual_pass"]
            checks.append(
                _check(
                    "active_batch.all_dual_pass",
                    not non_dual,
                    "all active ops are dual_pass" if not non_dual else f"non dual_pass active ops: {non_dual}",
                )
            )
    elif profile == "ir_arch":
        stage_map = _stage_map(run_summary)
        required = list(args.require_stage or ["primitive_reuse", "macro_composition", "mapping_tests", "intentir_semantics"])
        missing_or_fail = [s for s in required if not bool((stage_map.get(s) or {}).get("ok"))]
        checks.append(
            _check(
                "ir_arch.required_stage_ok",
                not missing_or_fail,
                "required IR architecture stages passed"
                if not missing_or_fail
                else f"missing or failed ir_arch stage(s): {missing_or_fail}",
            )
        )
    elif profile == "backend_compiler":
        stage_map = _stage_map(run_summary)
        required = list(args.require_stage or ["rvv_local", "cuda_local"])
        missing_or_fail = [s for s in required if not bool((stage_map.get(s) or {}).get("ok"))]
        checks.append(
            _check(
                "backend_compiler.required_stage_ok",
                not missing_or_fail,
                "required backend stages passed"
                if not missing_or_fail
                else f"missing or failed backend stage(s): {missing_or_fail}",
            )
        )
        timing_failures: list[str] = []
        timing_required_stages = {"rvv_local", "cuda_local"}
        for stage_name in required:
            if stage_name not in timing_required_stages:
                continue
            stage = stage_map.get(stage_name) or {}
            json_path = str(stage.get("json_path") or "").strip()
            if not json_path:
                timing_failures.append(f"{stage_name}:missing_json_path")
                continue
            ok_timing, detail_timing = _validate_backend_timing(Path(json_path))
            if not ok_timing:
                timing_failures.append(f"{stage_name}:{detail_timing}")
        checks.append(
            _check(
                "backend_compiler.stage_timing_complete",
                not timing_failures,
                "backend timing fields complete" if not timing_failures else "; ".join(timing_failures),
            )
        )

    handoff_text = args.handoff.read_text(encoding="utf-8") if args.handoff.is_file() else ""
    has_next_focus = ("Next Focus:" in handoff_text)
    checks.append(_check("handoff.has_next_focus", has_next_focus, "handoff contains Next Focus section" if has_next_focus else "missing Next Focus in handoff"))

    progress_ok = False
    progress_detail = "progress log missing"
    if args.progress_log.is_file():
        raw_lines = [ln for ln in args.progress_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if raw_lines:
            try:
                last = json.loads(raw_lines[-1])
                run_path_ok = str(last.get("run_summary_path") or "") == _to_repo_rel(args.run_summary)
                status_path_ok = str(last.get("status_converged_path") or "") == _to_repo_rel(args.status_converged)
                progress_ok = bool(run_path_ok and status_path_ok)
                progress_detail = (
                    "progress log tail points to current run artifacts"
                    if progress_ok
                    else "progress tail does not reference current run/status artifacts"
                )
            except Exception:
                progress_ok = False
                progress_detail = "progress log tail is not valid JSON"
    checks.append(_check("progress_log.tail_matches_artifacts", progress_ok, progress_detail))

    ok = all(bool(c.get("ok")) for c in checks)
    payload = {
        "ok": bool(ok),
        "profile": profile,
        "active_batch_path": _to_repo_rel(active_batch),
        "run_summary_path": _to_repo_rel(args.run_summary),
        "status_converged_path": _to_repo_rel(args.status_converged),
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Batch gate report written: {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
