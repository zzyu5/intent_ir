#!/usr/bin/env python3
"""
Build FlagGems denominator gap report:
source_all_ops -> source_filtered_ops -> semantic_ops.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.utils.repo_state import repo_state  # noqa: E402
from pipeline.triton.providers.flaggems.registry import (  # noqa: E402
    DEFAULT_FLAGGEMS_OPSET,
    DEFAULT_REGISTRY_PATH,
    is_deterministic_forward_op,
    load_flaggems_all_ops,
    normalize_semantic_name,
)

_RANDOM_OP_HINTS = {
    "rand",
    "randn",
    "bernoulli",
    "dropout",
    "uniform",
    "normal",
    "multinomial",
    "poisson",
    "cauchy",
    "geometric",
    "log_normal",
    "exponential",
}

_NON_SEMANTIC_OP_HINTS = {
    "scheduler",
    "metadata",
    "meta",
    "capability",
    "registry",
}

_NON_SEMANTIC_OP_EXACT = {
    "get_scheduler_metadata",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_output_path() -> Path:
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    return ROOT / "artifacts" / "flaggems_matrix" / "daily" / date_tag / "registry_gap_report_v1.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _deterministic_filter_reason(op_name: str) -> str:
    low = str(op_name).strip().lower()
    if not low:
        return "empty_name"
    if low in _NON_SEMANTIC_OP_EXACT:
        return "non_semantic_exact"
    if "backward" in low:
        return "backward"
    if low.endswith("_"):
        return "trailing_underscore"
    if any(tok in low for tok in _NON_SEMANTIC_OP_HINTS):
        return "non_semantic_hint"
    if any(tok in low for tok in _RANDOM_OP_HINTS):
        return "randomized"
    return "other_nondeterministic"


def _sort_counter(counter: Counter[str]) -> dict[str, int]:
    return {k: int(v) for k, v in sorted(counter.items(), key=lambda kv: kv[0])}


def build_gap_report(
    *,
    registry_payload: dict[str, Any],
    all_ops: list[str],
    max_examples_per_reason: int = 20,
) -> dict[str, Any]:
    unique_all_ops = [str(x) for x in list(all_ops or []) if isinstance(x, str) and str(x).strip()]
    filtered_ops = [op for op in unique_all_ops if is_deterministic_forward_op(op)]
    excluded_ops = [op for op in unique_all_ops if not is_deterministic_forward_op(op)]

    excluded_reason_counts: Counter[str] = Counter()
    excluded_examples_by_reason: dict[str, list[str]] = defaultdict(list)
    for op in excluded_ops:
        reason = _deterministic_filter_reason(op)
        excluded_reason_counts[reason] += 1
        if len(excluded_examples_by_reason[reason]) < int(max_examples_per_reason):
            excluded_examples_by_reason[reason].append(str(op))

    grouped_filtered: dict[str, list[str]] = defaultdict(list)
    for src in filtered_ops:
        semantic = normalize_semantic_name(src)
        if not str(semantic).strip():
            continue
        grouped_filtered[str(semantic)].append(str(src))
    filtered_semantics = sorted(grouped_filtered.keys())

    collapsed_semantics: list[dict[str, Any]] = []
    collapsed_source_ops_count = 0
    for semantic in sorted(grouped_filtered.keys()):
        source_ops = sorted(set(grouped_filtered[semantic]))
        if len(source_ops) <= 1:
            continue
        collapsed_source_ops_count += int(len(source_ops) - 1)
        collapsed_semantics.append(
            {
                "semantic_op": str(semantic),
                "source_ops": source_ops,
                "source_ops_count": int(len(source_ops)),
            }
        )
    collapsed_semantics.sort(key=lambda x: (-int(x.get("source_ops_count") or 0), str(x.get("semantic_op") or "")))

    entries = [e for e in list(registry_payload.get("entries") or []) if isinstance(e, dict)]
    registry_semantics = sorted({str(e.get("semantic_op") or "").strip() for e in entries if str(e.get("semantic_op") or "").strip()})
    registry_semantics_set = set(registry_semantics)
    filtered_semantics_set = set(filtered_semantics)

    missing_in_registry = sorted(filtered_semantics_set - registry_semantics_set)
    extra_in_registry = sorted(registry_semantics_set - filtered_semantics_set)

    status_counts: Counter[str] = Counter(str(e.get("status") or "") for e in entries if str(e.get("status") or "").strip())
    reason_counts: Counter[str] = Counter(str(e.get("status_reason") or "") for e in entries if str(e.get("status_reason") or "").strip())

    no_mapping_semantics = sorted(
        {
            str(e.get("semantic_op") or "").strip()
            for e in entries
            if (
                (not list(e.get("intent_ops") or []))
                or (str(e.get("status_reason") or "") == "no_intentir_mapping")
                or (str(e.get("status") or "") == "blocked_ir")
            )
            and str(e.get("semantic_op") or "").strip()
        }
    )
    missing_e2e_semantics = sorted(
        {
            str(e.get("semantic_op") or "").strip()
            for e in entries
            if (
                (not str(e.get("e2e_spec") or "").strip())
                or (str(e.get("status_reason") or "") == "missing_e2e_spec")
            )
            and str(e.get("semantic_op") or "").strip()
        }
    )

    counts_payload = dict(registry_payload.get("counts") or {})
    source_all_ops_registry = counts_payload.get("source_all_ops")
    source_filtered_ops_registry = counts_payload.get("source_filtered_ops")
    semantic_ops_registry = counts_payload.get("semantic_ops")
    counts_consistent = {
        "source_all_ops_match": bool(source_all_ops_registry == len(unique_all_ops)),
        "source_filtered_ops_match": bool(source_filtered_ops_registry == len(filtered_ops)),
        "semantic_ops_match": bool(semantic_ops_registry == len(registry_semantics)),
    }

    report = {
        "schema_version": "flaggems_registry_gap_report_v1",
        "generated_at": _utc_now_iso(),
        "counts": {
            "source_all_ops": int(len(unique_all_ops)),
            "source_filtered_ops": int(len(filtered_ops)),
            "semantic_ops": int(len(registry_semantics)),
        },
        "counts_registry": {
            "source_all_ops": source_all_ops_registry,
            "source_filtered_ops": source_filtered_ops_registry,
            "semantic_ops": semantic_ops_registry,
            "consistent_with_recomputed": counts_consistent,
        },
        "gaps": {
            "source_all_to_filtered": {
                "excluded_ops_count": int(len(excluded_ops)),
                "excluded_reason_counts": _sort_counter(excluded_reason_counts),
                "excluded_examples_by_reason": {k: list(v) for k, v in sorted(excluded_examples_by_reason.items(), key=lambda kv: kv[0])},
            },
            "filtered_to_semantic": {
                "source_filtered_ops_count": int(len(filtered_ops)),
                "semantic_unique_count": int(len(filtered_semantics)),
                "collapsed_source_ops_count": int(collapsed_source_ops_count),
                "collapsed_semantic_count": int(len(collapsed_semantics)),
                "collapsed_semantics": collapsed_semantics,
            },
            "semantic_registry_alignment": {
                "filtered_semantic_count": int(len(filtered_semantics)),
                "registry_semantic_count": int(len(registry_semantics)),
                "missing_in_registry_count": int(len(missing_in_registry)),
                "extra_in_registry_count": int(len(extra_in_registry)),
                "missing_in_registry": missing_in_registry,
                "extra_in_registry": extra_in_registry,
            },
        },
        "semantic_quality": {
            "status_counts": _sort_counter(status_counts),
            "status_reason_counts": _sort_counter(reason_counts),
            "no_mapping_count": int(len(no_mapping_semantics)),
            "no_mapping_semantic_ops": no_mapping_semantics,
            "missing_e2e_spec_count": int(len(missing_e2e_semantics)),
            "missing_e2e_spec_semantic_ops": missing_e2e_semantics,
        },
        "wave_hints": {
            "focus_source_ops_not_in_196": sorted(excluded_ops),
            "focus_semantics_requiring_mapping": no_mapping_semantics,
            "focus_semantics_requiring_e2e_spec": missing_e2e_semantics,
        },
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    ap.add_argument("--flaggems-src", type=Path, default=None, help="Optional FlagGems src/ path for source-all-op scan.")
    ap.add_argument("--opset", type=str, default=DEFAULT_FLAGGEMS_OPSET, choices=[DEFAULT_FLAGGEMS_OPSET])
    ap.add_argument("--max-examples-per-reason", type=int, default=20)
    ap.add_argument("--output", type=Path, default=_default_output_path())
    args = ap.parse_args()

    registry_path = Path(args.registry).resolve()
    if not registry_path.is_file():
        raise SystemExit(f"missing registry file: {registry_path}")
    registry_payload = _load_json(registry_path)

    all_ops = load_flaggems_all_ops(flaggems_src=(str(args.flaggems_src) if args.flaggems_src else None))
    report = build_gap_report(
        registry_payload=registry_payload,
        all_ops=all_ops,
        max_examples_per_reason=int(args.max_examples_per_reason),
    )
    report["opset"] = str(args.opset)
    report["repo"] = repo_state(root=ROOT)
    report["artifacts"] = {
        "registry_path": str(registry_path),
        "source_op_count": int(len(all_ops)),
    }

    out = _dump_json(Path(args.output).resolve(), report)
    print(f"Registry gap report written: {out}")
    print(
        json.dumps(
            {
                "source_all_ops": report["counts"]["source_all_ops"],
                "source_filtered_ops": report["counts"]["source_filtered_ops"],
                "semantic_ops": report["counts"]["semantic_ops"],
                "excluded_ops": report["gaps"]["source_all_to_filtered"]["excluded_ops_count"],
                "collapsed_source_ops": report["gaps"]["filtered_to_semantic"]["collapsed_source_ops_count"],
                "no_mapping_count": report["semantic_quality"]["no_mapping_count"],
                "missing_e2e_spec_count": report["semantic_quality"]["missing_e2e_spec_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
