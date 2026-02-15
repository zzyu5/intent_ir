"""
Report IntentIR mapping complexity from FlagGems registry.

This script is used by ir_arch lane to track whether mappings are drifting
toward one-op-to-one-semantic patterns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(numer: int, denom: int) -> float:
    return (float(numer) / float(denom)) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "intentir" / "mapping_complexity_report.json"))
    ap.add_argument(
        "--complex-families",
        action="append",
        default=[
            "index_scatter_gather",
            "conv_pool_interp",
            "matmul_linear",
            "attention_sequence",
            "reduction",
            "norm_activation",
        ],
        help="Families considered composition-critical for anti 1:1 guard.",
    )
    args = ap.parse_args()

    if not args.registry.is_file():
        raise FileNotFoundError(f"registry not found: {args.registry}")
    payload = _load_json(args.registry)
    entries = [e for e in list(payload.get("entries") or []) if isinstance(e, dict)]

    complex_families = set(str(x).strip() for x in list(args.complex_families or []) if str(x).strip())
    by_family: dict[str, dict[str, int]] = {}
    total = 0
    single = 0
    multi = 0
    zero = 0
    complex_total = 0
    complex_single = 0
    for row in entries:
        fam = str(row.get("family") or "unknown")
        ops = [str(x) for x in list(row.get("intent_ops") or []) if str(x).strip()]
        n_ops = len(ops)
        bucket = by_family.setdefault(
            fam,
            {
                "total": 0,
                "single_intent_ops": 0,
                "multi_intent_ops": 0,
                "zero_intent_ops": 0,
            },
        )
        bucket["total"] += 1
        total += 1
        if n_ops == 0:
            bucket["zero_intent_ops"] += 1
            zero += 1
        elif n_ops == 1:
            bucket["single_intent_ops"] += 1
            single += 1
        else:
            bucket["multi_intent_ops"] += 1
            multi += 1
        if fam in complex_families:
            complex_total += 1
            if n_ops == 1:
                complex_single += 1

    out = {
        "schema_version": "intentir_mapping_complexity_v1",
        "registry": str(args.registry),
        "total": int(total),
        "single_intent_ops": int(single),
        "multi_intent_ops": int(multi),
        "zero_intent_ops": int(zero),
        "single_intent_ratio": _ratio(single, total),
        "complex_families": sorted(list(complex_families)),
        "complex_total": int(complex_total),
        "complex_single_intent_ops": int(complex_single),
        "complex_single_intent_ratio": _ratio(complex_single, complex_total),
        "by_family": {k: v for k, v in sorted(by_family.items(), key=lambda kv: kv[0])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mapping complexity report written: {args.out}")


if __name__ == "__main__":
    main()
