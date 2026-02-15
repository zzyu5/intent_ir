"""
Check that registry intent mappings reuse allowed IntentIR primitives.
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

from intent_ir.ops.primitive_catalog import catalog_summary, is_allowed_primitive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--allow-macro", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "intentir" / "primitive_reuse_report.json"))
    args = ap.parse_args()

    if not args.registry.is_file():
        raise FileNotFoundError(f"registry not found: {args.registry}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    entries = [e for e in list(registry.get("entries") or []) if isinstance(e, dict)]

    violations: list[dict[str, Any]] = []
    reused: dict[str, int] = {}
    for entry in entries:
        semantic_op = str(entry.get("semantic_op") or "")
        intent_ops = [str(x) for x in list(entry.get("intent_ops") or []) if isinstance(x, str) and str(x)]
        for op in intent_ops:
            reused[op] = int(reused.get(op, 0)) + 1
            if not is_allowed_primitive(op, include_macro=bool(args.allow_macro)):
                violations.append(
                    {
                        "semantic_op": semantic_op,
                        "intent_op": op,
                        "reason": "unsupported_or_provider_specific_primitive",
                    }
                )
            if op.startswith("flaggems_") or op.startswith("triton_"):
                violations.append(
                    {
                        "semantic_op": semantic_op,
                        "intent_op": op,
                        "reason": "provider_specific_primitive_name",
                    }
                )

    payload = {
        "ok": len(violations) == 0,
        "registry": str(args.registry),
        "allow_macro": bool(args.allow_macro),
        "catalog_summary": dict(catalog_summary()),
        "reused_primitives": dict(sorted(reused.items(), key=lambda kv: kv[0])),
        "violations": violations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Primitive reuse report written: {args.out}")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
