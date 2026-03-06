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
    source_fail = dict(comp.get("source_replay_failure") or {})
    target_fail = dict(comp.get("target_oracle_failure") or {})
    return {
        "comparison_path": str(path),
        "kernel": str(obj.get("kernel") or ""),
        "backend_target": str(obj.get("backend_target") or ""),
        "source_arch": str(obj.get("source_arch") or ""),
        "target_arch": str(obj.get("target_arch") or ""),
        "guided_best_ratio": comp.get("guided_best_ratio"),
        "source_replay_best_ratio": comp.get("source_replay_best_ratio"),
        "target_oracle_best_ratio": comp.get("target_oracle_best_ratio"),
        "guided_vs_source_replay": comp.get("guided_vs_source_replay"),
        "guided_vs_target_oracle": comp.get("guided_vs_target_oracle"),
        "guided_kernel_kind": str(guided_first.get("kernel_kind") or ""),
        "guided_bindings": json.dumps(guided_first.get("bindings") or {}, ensure_ascii=False, sort_keys=True),
        "source_candidate": str(obj.get("source_candidate") or ""),
        "target_candidate": str(obj.get("target_candidate") or ""),
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
        "guided_best_ratio",
        "source_replay_best_ratio",
        "target_oracle_best_ratio",
        "guided_vs_source_replay",
        "guided_vs_target_oracle",
        "guided_kernel_kind",
        "guided_bindings",
        "source_candidate",
        "target_candidate",
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
