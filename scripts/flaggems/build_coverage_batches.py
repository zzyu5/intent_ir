"""
Build fixed family coverage batches from FlagGems registry.

The output is the canonical input for category-based full coverage runs:
`workflow/flaggems/state/coverage_batches.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FAMILY_ORDER = [
    "elementwise_broadcast",
    "reduction",
    "norm_activation",
    "index_scatter_gather",
    "matmul_linear",
    "conv_pool_interp",
    "attention_sequence",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _append_unique(dst: list[str], value: str) -> None:
    val = str(value).strip()
    if not val:
        return
    if val not in dst:
        dst.append(val)


def _build_batches(registry_payload: dict[str, Any]) -> dict[str, Any]:
    entries = [e for e in list(registry_payload.get("entries") or []) if isinstance(e, dict)]
    family_order_raw = [str(x).strip() for x in list(registry_payload.get("family_order") or []) if str(x).strip()]
    family_order = family_order_raw or list(DEFAULT_FAMILY_ORDER)

    by_family: dict[str, dict[str, list[str]]] = {
        fam: {"semantic_ops": [], "kernels": []}
        for fam in family_order
    }
    unknown_family_bucket: dict[str, list[str]] = {"semantic_ops": [], "kernels": []}
    all_semantics: list[str] = []
    all_kernels: list[str] = []

    for entry in entries:
        family = str(entry.get("family") or "").strip()
        semantic_op = str(entry.get("semantic_op") or "").strip()
        kernel = str(entry.get("e2e_spec") or "").strip()
        if not semantic_op:
            continue
        _append_unique(all_semantics, semantic_op)
        if kernel:
            _append_unique(all_kernels, kernel)

        if family in by_family:
            _append_unique(by_family[family]["semantic_ops"], semantic_op)
            if kernel:
                _append_unique(by_family[family]["kernels"], kernel)
        else:
            # Keep unknown-family rows visible; do not silently drop.
            _append_unique(unknown_family_bucket["semantic_ops"], semantic_op)
            if kernel:
                _append_unique(unknown_family_bucket["kernels"], kernel)

    batches: list[dict[str, Any]] = []
    for family in family_order:
        row = by_family.get(family) or {"semantic_ops": [], "kernels": []}
        semantics = list(row.get("semantic_ops") or [])
        kernels = list(row.get("kernels") or [])
        batches.append(
            {
                "family": str(family),
                "semantic_ops": semantics,
                "kernels": kernels,
                "semantic_count": int(len(semantics)),
                "kernel_count": int(len(kernels)),
            }
        )

    if unknown_family_bucket["semantic_ops"]:
        batches.append(
            {
                "family": "unknown",
                "semantic_ops": list(unknown_family_bucket["semantic_ops"]),
                "kernels": list(unknown_family_bucket["kernels"]),
                "semantic_count": int(len(unknown_family_bucket["semantic_ops"])),
                "kernel_count": int(len(unknown_family_bucket["kernels"])),
            }
        )

    family_semantic_counts = {
        str(row["family"]): int(row["semantic_count"])
        for row in batches
    }
    family_kernel_counts = {
        str(row["family"]): int(row["kernel_count"])
        for row in batches
    }

    return {
        "schema_version": "flaggems_coverage_batches_v1",
        "generated_at": _utc_now_iso(),
        "family_order": list(family_order),
        "batches": batches,
        "summary": {
            "semantic_ops_total": int(len(all_semantics)),
            "kernels_total": int(len(all_kernels)),
            "families_total": int(len(batches)),
            "family_semantic_counts": family_semantic_counts,
            "family_kernel_counts": family_kernel_counts,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--registry",
        type=Path,
        default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "state" / "coverage_batches.json"),
    )
    args = ap.parse_args()

    registry_payload = _load_json(args.registry)
    payload = _build_batches(registry_payload)
    payload["source_registry_path"] = str(args.registry)

    out_path = _dump_json(args.out, payload)
    summary = dict(payload.get("summary") or {})
    print(
        "Coverage batches written: "
        f"{out_path} (families={summary.get('families_total')}, "
        f"semantic_ops={summary.get('semantic_ops_total')}, kernels={summary.get('kernels_total')})"
    )


if __name__ == "__main__":
    main()
