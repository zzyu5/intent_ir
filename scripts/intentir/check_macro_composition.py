"""
Check that macro-level intent ops stay in shared IntentIR namespace.
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

from intent_ir.ops import MACRO_OPS
from intent_ir.ops.specs import op_spec_for
from source_loader import load_entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument(
        "--mlir-manifest",
        type=Path,
        default=None,
        help="Use MLIR manifest as source instead of registry.",
    )
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "intentir" / "macro_composition_report.json"))
    args = ap.parse_args()

    entries, source = load_entries(
        registry_path=(None if args.mlir_manifest is not None else args.registry),
        mlir_manifest_path=args.mlir_manifest,
    )

    macro_hits: dict[str, int] = {}
    violations: list[dict[str, Any]] = []

    for entry in entries:
        semantic_op = str(entry.get("semantic_op") or "")
        intent_ops = [str(x) for x in list(entry.get("intent_ops") or []) if isinstance(x, str) and str(x)]
        for op in intent_ops:
            if op.startswith("flaggems_") or op.startswith("triton_"):
                violations.append(
                    {
                        "semantic_op": semantic_op,
                        "intent_op": op,
                        "reason": "provider_specific_intent_op_name",
                    }
                )
            spec = op_spec_for(op)
            if spec is None:
                violations.append(
                    {
                        "semantic_op": semantic_op,
                        "intent_op": op,
                        "reason": "op_not_declared_in_intentir_specs",
                    }
                )
                continue
            if op in MACRO_OPS:
                macro_hits[op] = int(macro_hits.get(op, 0)) + 1
                if spec.tier != "macro":
                    violations.append(
                        {
                            "semantic_op": semantic_op,
                            "intent_op": op,
                            "reason": "macro_op_not_tagged_macro_tier",
                        }
                    )

    report = {
        "ok": len(violations) == 0,
        "source": str(source),
        "macro_ops_in_catalog": sorted(list(MACRO_OPS)),
        "macro_hits": dict(sorted(macro_hits.items(), key=lambda kv: kv[0])),
        "violations": violations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Macro composition report written: {args.out}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
