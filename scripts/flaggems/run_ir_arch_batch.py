"""
Run ir_arch lane checks and emit workflow-compatible artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.ops.composition_policy import single_intent_ratio_target


def _run(cmd: list[str], *, dry_run: bool) -> tuple[int, str, str]:
    if dry_run:
        return 0, "(dry-run)", ""
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")


def _now_date_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_cmd(raw: str) -> list[str]:
    tokens = shlex.split(str(raw))
    if not tokens:
        raise ValueError("empty stage command")
    if tokens[0] == "python":
        tokens[0] = sys.executable
    return tokens


def _default_out_dir() -> Path:
    date_tag = _now_date_tag()
    run_name = f"ir_arch_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    return ROOT / "artifacts" / "flaggems_matrix" / "daily" / date_tag / run_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active-batch", type=Path, default=(ROOT / "workflow" / "flaggems" / "state" / "active_batch_ir_arch.json"))
    ap.add_argument("--out-dir", type=Path, default=_default_out_dir())
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument(
        "--primitive-reuse-allow-macro",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow IntentIR macro ops in primitive reuse check (default: true).",
    )
    ap.add_argument(
        "--mapping-tests-cmd",
        default=f"{sys.executable} -m pytest -q tests/frontends/triton/test_flaggems_semantic_rules.py tests/frontends/triton/test_provider_hooks.py",
    )
    ap.add_argument(
        "--intentir-semantics-cmd",
        default=f"{sys.executable} -m pytest -q tests/test_ir_types.py tests/test_opset_matrix.py",
    )
    ap.add_argument(
        "--macro-composition-cmd",
        default=f"{sys.executable} scripts/intentir/check_macro_composition.py",
    )
    ap.add_argument(
        "--mapping-complexity-threshold",
        type=float,
        default=float(single_intent_ratio_target("m2")),
        help="Max allowed complex-family single-intent ratio for ir_arch gate.",
    )
    ap.add_argument(
        "--mapping-global-unique-threshold",
        type=float,
        default=0.40,
        help="Max allowed global unique single-primitive ratio for ir_arch gate.",
    )
    ap.add_argument(
        "--mapping-policy-stage",
        choices=["m1", "m2"],
        default="m2",
        help="Composition policy stage label for complexity report (default: m2).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = out_dir / "run_summary.json"
    status_converged_path = out_dir / "status_converged.json"
    primitive_report = out_dir / "primitive_reuse_report.json"
    macro_report = out_dir / "macro_composition_report.json"
    mapping_complexity_report = out_dir / "mapping_complexity_report.json"

    primitive_cmd = [
        sys.executable,
        "scripts/intentir/check_primitive_reuse.py",
        "--registry",
        str(args.registry),
        "--out",
        str(primitive_report),
    ]
    if bool(args.primitive_reuse_allow_macro):
        primitive_cmd.append("--allow-macro")
    stage_defs = [
        ("primitive_reuse", primitive_cmd),
        ("macro_composition", _parse_cmd(str(args.macro_composition_cmd)) + ["--registry", str(args.registry), "--out", str(macro_report)]),
        (
            "mapping_complexity",
            [
                sys.executable,
                "scripts/intentir/report_mapping_complexity.py",
                "--registry",
                str(args.registry),
                "--out",
                str(mapping_complexity_report),
                "--refresh-mappings-from-rules",
                "--policy-stage",
                str(args.mapping_policy_stage),
                "--max-complex-single-intent-ratio",
                str(float(args.mapping_complexity_threshold)),
                "--fail-on-threshold-breach",
                "--max-global-unique-single-primitive-ratio",
                str(float(args.mapping_global_unique_threshold)),
                "--fail-on-global-threshold-breach",
            ],
        ),
        ("mapping_tests", _parse_cmd(str(args.mapping_tests_cmd))),
        ("intentir_semantics", _parse_cmd(str(args.intentir_semantics_cmd))),
    ]
    stage_rows: list[dict[str, Any]] = []
    overall_ok = True
    for stage_name, cmd in stage_defs:
        rc, stdout, stderr = _run(cmd, dry_run=bool(args.dry_run))
        row: dict[str, Any] = {
            "stage": stage_name,
            "cmd": cmd,
            "rc": int(rc),
            "ok": int(rc) == 0,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }
        if stage_name == "primitive_reuse":
            row["json_path"] = str(primitive_report)
        if stage_name == "macro_composition":
            row["json_path"] = str(macro_report)
        if stage_name == "mapping_complexity":
            row["json_path"] = str(mapping_complexity_report)
        stage_rows.append(row)
        overall_ok = overall_ok and bool(row["ok"])

    run_summary = {
        "ok": bool(overall_ok),
        "lane": "ir_arch",
        "active_batch_path": str(args.active_batch),
        "registry": str(args.registry),
        "stages": stage_rows,
        "out_dir": str(out_dir),
    }
    run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ir_arch profile does not depend on semantic scoped convergence entries.
    status_converged = {
        "schema_version": "flaggems_registry_converged_v2",
        "lane": "ir_arch",
        "entries": [],
        "counts_by_status": {},
        "reason_code": "ir_arch_stage_checks",
    }
    status_converged_path.write_text(json.dumps(status_converged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ir_arch run summary written: {run_summary_path}")
    print(f"ir_arch status converged written: {status_converged_path}")
    raise SystemExit(0 if bool(overall_ok) else 1)


if __name__ == "__main__":
    main()
