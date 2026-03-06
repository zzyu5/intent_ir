from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_comparison(path: Path) -> dict[str, Any]:
    obj = _load_json(path)
    comp = dict(obj.get("comparisons") or {})
    guided_first = dict(comp.get("guided_first_candidate") or {})
    source_first = dict(comp.get("source_replay_first_candidate") or {})
    target_first = dict(comp.get("target_oracle_first_candidate") or {})
    guided_fail = dict(comp.get("guided_failure") or {})
    source_fail = dict(comp.get("source_replay_failure") or {})
    target_fail = dict(comp.get("target_oracle_failure") or {})
    guided_outcome = dict(comp.get("guided_outcome") or {})
    source_outcome = dict(comp.get("source_replay_outcome") or {})
    target_outcome = dict(comp.get("target_oracle_outcome") or {})
    source_analysis = dict(comp.get("source_replay_analysis") or {})
    target_analysis = dict(comp.get("target_oracle_analysis") or {})
    source_portable = dict(comp.get("source_replay_portable_outcome") or {})
    target_portable = dict(comp.get("target_oracle_portable_outcome") or {})
    return {
        "comparison_path": str(path),
        "kernel": str(obj.get("kernel") or ""),
        "backend_target": str(obj.get("backend_target") or ""),
        "source_arch": str(obj.get("source_arch") or ""),
        "target_arch": str(obj.get("target_arch") or ""),
        "compiler_stack": str(obj.get("compiler_stack") or ""),
        "compiler_cpp_wave": str(obj.get("compiler_cpp_wave") or ""),
        "guided_compiler_stack": str(obj.get("guided_compiler_stack") or ""),
        "source_compiler_stack": str(obj.get("source_compiler_stack") or ""),
        "target_compiler_stack": str(obj.get("target_compiler_stack") or ""),
        "shape_bindings": json.dumps(obj.get("shape_bindings") or {}, ensure_ascii=False, sort_keys=True),
        "evidence_primary": str(dict(obj.get("evidence_source") or {}).get("primary") or ""),
        "hardware_cluster": str(dict(obj.get("hardware_model") or {}).get("arch_cluster") or ""),
        "guided_best_ratio": comp.get("guided_best_ratio"),
        "guided_best_qps_intentir": comp.get("guided_best_qps_intentir"),
        "guided_best_qps_native": comp.get("guided_best_qps_native"),
        "source_replay_raw_ratio": comp.get("source_replay_raw_ratio"),
        "source_replay_raw_qps_intentir": comp.get("source_replay_raw_qps_intentir"),
        "source_replay_raw_qps_native": comp.get("source_replay_raw_qps_native"),
        "source_replay_portable_ratio": comp.get("source_replay_portable_ratio"),
        "source_replay_portable_qps_intentir": comp.get("source_replay_portable_qps_intentir"),
        "source_replay_portable_qps_native": comp.get("source_replay_portable_qps_native"),
        "target_oracle_raw_ratio": comp.get("target_oracle_raw_ratio"),
        "target_oracle_raw_qps_intentir": comp.get("target_oracle_raw_qps_intentir"),
        "target_oracle_raw_qps_native": comp.get("target_oracle_raw_qps_native"),
        "target_oracle_portable_ratio": comp.get("target_oracle_portable_ratio"),
        "target_oracle_portable_qps_intentir": comp.get("target_oracle_portable_qps_intentir"),
        "target_oracle_portable_qps_native": comp.get("target_oracle_portable_qps_native"),
        "shared_native_qps": comp.get("shared_native_qps"),
        "native_qps_spread_ratio": comp.get("native_qps_spread_ratio"),
        "guided_shared_native_ratio": comp.get("guided_shared_native_ratio"),
        "source_replay_portable_shared_native_ratio": comp.get("source_replay_portable_shared_native_ratio"),
        "target_oracle_portable_shared_native_ratio": comp.get("target_oracle_portable_shared_native_ratio"),
        "source_replay_best_ratio": comp.get("source_replay_best_ratio"),
        "target_oracle_best_ratio": comp.get("target_oracle_best_ratio"),
        "guided_vs_source_replay_raw": comp.get("guided_vs_source_replay_raw"),
        "guided_vs_source_replay_portable": comp.get("guided_vs_source_replay_portable"),
        "guided_vs_target_oracle_raw": comp.get("guided_vs_target_oracle_raw"),
        "guided_vs_portable_target_oracle": comp.get("guided_vs_portable_target_oracle"),
        "guided_vs_source_replay": comp.get("guided_vs_source_replay"),
        "guided_vs_target_oracle": comp.get("guided_vs_target_oracle"),
        "guided_kernel_kind": str(guided_first.get("kernel_kind") or ""),
        "guided_bindings": json.dumps(guided_first.get("bindings") or {}, ensure_ascii=False, sort_keys=True),
        "guided_outcome": str(guided_outcome.get("status") or ""),
        "guided_returncode": guided_outcome.get("returncode"),
        "guided_failure_code": str(guided_fail.get("reason_code") or ""),
        "guided_failure_detail": str(guided_fail.get("reason_detail") or ""),
        "source_candidate": str(obj.get("source_candidate") or ""),
        "source_candidate_origin": str(obj.get("source_candidate_origin") or ""),
        "source_kernel_kind": str(source_first.get("kernel_kind") or ""),
        "source_bindings": json.dumps(source_first.get("bindings") or {}, ensure_ascii=False, sort_keys=True),
        "target_candidate": str(obj.get("target_candidate") or ""),
        "target_candidate_origin": str(obj.get("target_candidate_origin") or ""),
        "target_kernel_kind": str(target_first.get("kernel_kind") or ""),
        "target_bindings": json.dumps(target_first.get("bindings") or {}, ensure_ascii=False, sort_keys=True),
        "source_outcome": str(source_outcome.get("status") or ""),
        "source_returncode": source_outcome.get("returncode"),
        "target_outcome": str(target_outcome.get("status") or ""),
        "target_returncode": target_outcome.get("returncode"),
        "source_portable_outcome": str(source_portable.get("status") or ""),
        "source_portable_candidate": str(source_portable.get("candidate") or ""),
        "source_portable_reason": str(source_portable.get("reason") or ""),
        "target_portable_outcome": str(target_portable.get("status") or ""),
        "target_portable_candidate": str(target_portable.get("candidate") or ""),
        "target_portable_reason": str(target_portable.get("reason") or ""),
        "source_analysis": str(source_analysis.get("status") or ""),
        "source_repair_candidate": str(dict(source_analysis.get("repair") or {}).get("repair_candidate") or ""),
        "source_repair_reason": str(dict(source_analysis.get("repair") or {}).get("reason") or ""),
        "target_analysis": str(target_analysis.get("status") or ""),
        "target_repair_candidate": str(dict(target_analysis.get("repair") or {}).get("repair_candidate") or ""),
        "target_repair_reason": str(dict(target_analysis.get("repair") or {}).get("reason") or ""),
        "source_failure_code": str(source_fail.get("reason_code") or ""),
        "source_failure_detail": str(source_fail.get("reason_detail") or ""),
        "target_failure_code": str(target_fail.get("reason_code") or ""),
        "target_failure_detail": str(target_fail.get("reason_detail") or ""),
        "source_ok": source_fail.get("ok"),
        "target_ok": target_fail.get("ok"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize multiple ORG comparison.json files into paper-friendly tables.")
    parser.add_argument("--root", required=True, help="Root directory to scan for comparison.json files.")
    parser.add_argument("--out-dir", required=True, help="Directory to write comparison_table.jsonl/csv.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_flatten_comparison(path) for path in sorted(root.rglob("comparison.json"))]
    jsonl_path = out_dir / "comparison_table.jsonl"
    csv_path = out_dir / "comparison_table.csv"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "comparison_path",
        "kernel",
        "backend_target",
        "source_arch",
        "target_arch",
        "compiler_stack",
        "compiler_cpp_wave",
        "guided_compiler_stack",
        "source_compiler_stack",
        "target_compiler_stack",
        "shape_bindings",
        "evidence_primary",
        "hardware_cluster",
        "guided_best_ratio",
        "guided_best_qps_intentir",
        "guided_best_qps_native",
        "source_replay_raw_ratio",
        "source_replay_raw_qps_intentir",
        "source_replay_raw_qps_native",
        "source_replay_portable_ratio",
        "source_replay_portable_qps_intentir",
        "source_replay_portable_qps_native",
        "target_oracle_raw_ratio",
        "target_oracle_raw_qps_intentir",
        "target_oracle_raw_qps_native",
        "target_oracle_portable_ratio",
        "target_oracle_portable_qps_intentir",
        "target_oracle_portable_qps_native",
        "shared_native_qps",
        "native_qps_spread_ratio",
        "guided_shared_native_ratio",
        "source_replay_portable_shared_native_ratio",
        "target_oracle_portable_shared_native_ratio",
        "source_replay_best_ratio",
        "target_oracle_best_ratio",
        "guided_vs_source_replay_raw",
        "guided_vs_source_replay_portable",
        "guided_vs_target_oracle_raw",
        "guided_vs_portable_target_oracle",
        "guided_vs_source_replay",
        "guided_vs_target_oracle",
        "guided_kernel_kind",
        "guided_bindings",
        "guided_outcome",
        "guided_returncode",
        "guided_failure_code",
        "guided_failure_detail",
        "source_candidate",
        "source_candidate_origin",
        "source_kernel_kind",
        "source_bindings",
        "target_candidate",
        "target_candidate_origin",
        "target_kernel_kind",
        "target_bindings",
        "source_outcome",
        "source_returncode",
        "target_outcome",
        "target_returncode",
        "source_portable_outcome",
        "source_portable_candidate",
        "source_portable_reason",
        "target_portable_outcome",
        "target_portable_candidate",
        "target_portable_reason",
        "source_analysis",
        "source_repair_candidate",
        "source_repair_reason",
        "target_analysis",
        "target_repair_candidate",
        "target_repair_reason",
        "source_failure_code",
        "source_failure_detail",
        "target_failure_code",
        "target_failure_detail",
        "source_ok",
        "target_ok",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote: {jsonl_path}")
    print(f"wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
