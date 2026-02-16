"""
Report IntentIR mapping complexity from FlagGems registry.

This script is used by ir_arch lane to track whether mappings are drifting
toward one-op-to-one-semantic patterns.
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

from intent_ir.ops.composition_policy import (
    COMPLEX_FAMILIES,
    composition_required,
    evaluate_complex_family_ratio,
    single_intent_ratio_target,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(numer: int, denom: int) -> float:
    return (float(numer) / float(denom)) if denom > 0 else 0.0


def _split_complex_families(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        for token in str(item).split(","):
            value = str(token).strip()
            if value and value not in out:
                out.append(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "intentir" / "mapping_complexity_report.json"))
    ap.add_argument(
        "--complex-families",
        action="append",
        default=list(COMPLEX_FAMILIES),
        help="Families considered composition-critical for anti 1:1 guard.",
    )
    ap.add_argument(
        "--policy-stage",
        choices=["m1", "m2"],
        default="m1",
        help="Composition policy stage to evaluate ratio target (default: m1).",
    )
    ap.add_argument(
        "--max-complex-single-intent-ratio",
        type=float,
        default=None,
        help="Optional explicit max complex-family single-intent ratio threshold.",
    )
    ap.add_argument(
        "--fail-on-threshold-breach",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when complex-family ratio exceeds threshold.",
    )
    args = ap.parse_args()

    if not args.registry.is_file():
        raise FileNotFoundError(f"registry not found: {args.registry}")
    payload = _load_json(args.registry)
    entries = [e for e in list(payload.get("entries") or []) if isinstance(e, dict)]

    complex_families = set(_split_complex_families(list(args.complex_families or [])))
    by_family: dict[str, dict[str, int]] = {}
    total = 0
    single = 0
    multi = 0
    zero = 0
    complex_total = 0
    complex_single = 0
    required_total = 0
    required_single = 0
    for row in entries:
        fam = str(row.get("family") or "unknown")
        semantic_op = str(row.get("semantic_op") or "")
        mapping_kind = str(row.get("mapping_kind") or "")
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
            if composition_required(
                semantic_op=semantic_op,
                family=fam,
                mapping_kind=mapping_kind,
            ):
                required_total += 1
                if n_ops == 1:
                    required_single += 1

    threshold = (
        float(args.max_complex_single_intent_ratio)
        if args.max_complex_single_intent_ratio is not None
        else float(single_intent_ratio_target(str(args.policy_stage)))
    )
    gate = evaluate_complex_family_ratio(
        ratio=_ratio(required_single, required_total),
        stage=str(args.policy_stage),
    )
    gate["threshold"] = float(threshold)
    gate["ok"] = bool(float(gate["ratio"]) <= float(threshold))
    gate["detail"] = (
        f"complex_family_single_intent_ratio {float(gate['ratio']):.4f} <= {float(threshold):.4f}"
        if bool(gate["ok"])
        else f"complex_family_single_intent_ratio {float(gate['ratio']):.4f} > {float(threshold):.4f}"
    )
    out = {
        "schema_version": "intentir_mapping_complexity_v1",
        "registry": str(args.registry),
        "policy_stage": str(args.policy_stage),
        "total": int(total),
        "single_intent_ops": int(single),
        "multi_intent_ops": int(multi),
        "zero_intent_ops": int(zero),
        "single_intent_ratio": _ratio(single, total),
        "complex_families": sorted(list(complex_families)),
        "complex_total": int(complex_total),
        "complex_single_intent_ops": int(complex_single),
        "complex_single_intent_ratio": _ratio(complex_single, complex_total),
        "composition_required_total": int(required_total),
        "composition_required_single_intent_ops": int(required_single),
        "composition_required_single_intent_ratio": _ratio(required_single, required_total),
        "gate": gate,
        "by_family": {k: v for k, v in sorted(by_family.items(), key=lambda kv: kv[0])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mapping complexity report written: {args.out}")
    if bool(args.fail_on_threshold_breach) and (not bool(gate["ok"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
