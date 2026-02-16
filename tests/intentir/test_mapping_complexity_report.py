from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mapping_complexity_report_counts_family_ratios(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    out = tmp_path / "mapping_complexity.json"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {"semantic_op": "a", "family": "elementwise_broadcast", "intent_ops": ["add"]},
                    {"semantic_op": "b", "family": "index_scatter_gather", "intent_ops": ["gather"]},
                    {"semantic_op": "c", "family": "index_scatter_gather", "intent_ops": ["reshape", "gather"]},
                    {"semantic_op": "d", "family": "attention_sequence", "intent_ops": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir/report_mapping_complexity.py",
            "--registry",
            str(registry),
            "--out",
            str(out),
            "--complex-families",
            "index_scatter_gather,attention_sequence",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "intentir_mapping_complexity_v1"
    assert payload["total"] == 4
    assert payload["single_intent_ops"] == 2
    assert payload["multi_intent_ops"] == 1
    assert payload["zero_intent_ops"] == 1
    assert payload["raw_single_semantic_ratio"] == payload["single_intent_ratio"]
    assert payload["global_unique_single_primitive_count"] == 2
    assert payload["global_unique_single_primitive_ratio"] == 0.5
    assert payload["complex_family_single_semantic_ratio"] == payload["complex_single_intent_ratio"]
    assert payload["by_family"]["index_scatter_gather"]["total"] == 2
    assert "gate" in payload


def test_mapping_complexity_report_can_fail_on_threshold_breach(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    out = tmp_path / "mapping_complexity.json"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "scaled_dot_product_attention",
                        "family": "attention_sequence",
                        "mapping_kind": "macro_template",
                        "intent_ops": ["scaled_dot_product_attention"],
                    },
                    {
                        "semantic_op": "where_scalar_self",
                        "family": "index_scatter_gather",
                        "mapping_kind": "index_template",
                        "intent_ops": ["where"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir/report_mapping_complexity.py",
            "--registry",
            str(registry),
            "--out",
            str(out),
            "--complex-families",
            "attention_sequence,index_scatter_gather",
            "--max-complex-single-intent-ratio",
            "0.2",
            "--fail-on-threshold-breach",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate"]["ok"] is False


def test_mapping_complexity_report_refreshes_from_semantic_rules(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    out = tmp_path / "mapping_complexity.json"
    # Intentionally stale intent_ops for complex families.
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "semantic_op": "conv2d",
                        "family": "conv_pool_interp",
                        "mapping_kind": "direct_supported_op",
                        "intent_ops": ["conv2d"],
                    },
                    {
                        "semantic_op": "index_add",
                        "family": "index_scatter_gather",
                        "mapping_kind": "direct_supported_op",
                        "intent_ops": ["index_add"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir/report_mapping_complexity.py",
            "--registry",
            str(registry),
            "--out",
            str(out),
            "--complex-families",
            "conv_pool_interp,index_scatter_gather",
            "--refresh-mappings-from-rules",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    # semantic_rules currently map both to multi-intent compositions.
    assert payload["single_intent_ops"] == 0
    assert payload["complex_family_single_semantic_ratio"] == 0.0
