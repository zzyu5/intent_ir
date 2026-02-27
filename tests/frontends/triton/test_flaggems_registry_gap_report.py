from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "flaggems" / "build_registry_gap_report.py"
    spec = importlib.util.spec_from_file_location("build_registry_gap_report", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_gap_report_classifies_all_to_filtered_and_collapse() -> None:
    mod = _load_module()
    registry_payload = {
        "counts": {"source_all_ops": 8, "source_filtered_ops": 4, "semantic_ops": 3},
        "entries": [
            {
                "semantic_op": "abs",
                "intent_ops": ["abs"],
                "e2e_spec": "abs2d",
                "status": "dual_pass",
                "status_reason": "runtime_dual_backend_pass",
            },
            {
                "semantic_op": "mm",
                "intent_ops": ["matmul"],
                "e2e_spec": "mm2d",
                "status": "dual_pass",
                "status_reason": "runtime_dual_backend_pass",
            },
            {
                "semantic_op": "where_self",
                "intent_ops": ["where"],
                "e2e_spec": "where2d",
                "status": "dual_pass",
                "status_reason": "runtime_dual_backend_pass",
            },
        ],
    }
    all_ops = [
        "abs",
        "abs_out",  # collapse into semantic abs
        "mm",
        "where_scalar_self",
        "foo_backward",  # backward
        "randn",  # randomized
        "get_scheduler_metadata",  # non-semantic
        "mm_",  # trailing underscore
    ]

    report = mod.build_gap_report(registry_payload=registry_payload, all_ops=all_ops, max_examples_per_reason=10)

    reason_counts = dict(report["gaps"]["source_all_to_filtered"]["excluded_reason_counts"])
    assert int(report["counts"]["source_all_ops"]) == 8
    assert int(report["counts"]["source_filtered_ops"]) == 4
    assert int(reason_counts.get("backward") or 0) == 1
    assert int(reason_counts.get("randomized") or 0) == 1
    assert int(reason_counts.get("non_semantic_exact") or 0) == 1
    assert int(reason_counts.get("trailing_underscore") or 0) == 1

    filtered_to_sem = dict(report["gaps"]["filtered_to_semantic"])
    assert int(filtered_to_sem["semantic_unique_count"]) == 3
    assert int(filtered_to_sem["collapsed_source_ops_count"]) == 1
    collapsed = list(filtered_to_sem["collapsed_semantics"])
    assert collapsed[0]["semantic_op"] == "abs"
    assert sorted(collapsed[0]["source_ops"]) == ["abs", "abs_out"]


def test_build_gap_report_exposes_mapping_and_e2e_gaps() -> None:
    mod = _load_module()
    registry_payload = {
        "counts": {"source_all_ops": 3, "source_filtered_ops": 3, "semantic_ops": 3},
        "entries": [
            {
                "semantic_op": "mapped_ok",
                "intent_ops": ["add"],
                "e2e_spec": "add2d",
                "status": "dual_pass",
                "status_reason": "runtime_dual_backend_pass",
            },
            {
                "semantic_op": "missing_map",
                "intent_ops": [],
                "e2e_spec": "missing_map2d",
                "status": "blocked_ir",
                "status_reason": "no_intentir_mapping",
            },
            {
                "semantic_op": "missing_e2e",
                "intent_ops": ["mul"],
                "e2e_spec": "",
                "status": "blocked_backend",
                "status_reason": "missing_e2e_spec",
            },
        ],
    }
    report = mod.build_gap_report(registry_payload=registry_payload, all_ops=["mapped_ok", "missing_map", "missing_e2e"])

    quality = dict(report["semantic_quality"])
    assert int(quality["no_mapping_count"]) == 1
    assert list(quality["no_mapping_semantic_ops"]) == ["missing_map"]
    assert int(quality["missing_e2e_spec_count"]) == 1
    assert list(quality["missing_e2e_spec_semantic_ops"]) == ["missing_e2e"]
