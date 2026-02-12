"""
Generate a machine-readable FlagGems coverage report JSON.

The report is derived from the frozen registry and is suitable for CI gates.
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

from pipeline.triton.flaggems_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_registry,
)


def _summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    blockers: list[dict[str, str]] = []

    for e in entries:
        status = str(e.get("status") or "unknown")
        family = str(e.get("family") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_family[family] = by_family.get(family, 0) + 1
        if status in {"blocked_ir", "blocked_backend"}:
            blockers.append(
                {
                    "semantic_op": str(e.get("semantic_op")),
                    "status": status,
                    "reason": str(e.get("status_reason") or ""),
                }
            )

    return {
        "total_semantic_ops": int(len(entries)),
        "by_status": dict(sorted(by_status.items(), key=lambda kv: kv[0])),
        "by_family": dict(sorted(by_family.items(), key=lambda kv: kv[0])),
        "blockers": blockers,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "flaggems_coverage" / "coverage_report.json"))
    args = ap.parse_args()

    registry = load_registry(args.registry)
    entries = [e for e in (registry.get("entries") or []) if isinstance(e, dict)]
    report = {
        "schema_version": "flaggems_coverage_report_v1",
        "registry_path": str(Path(args.registry)),
        "registry_schema_version": registry.get("schema_version"),
        "registry_generated_at": registry.get("generated_at"),
        "summary": _summarize(entries),
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Coverage report written: {args.out}")


if __name__ == "__main__":
    main()
