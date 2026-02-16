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
    assert payload["by_family"]["index_scatter_gather"]["total"] == 2
    assert "gate" in payload


def test_mapping_complexity_report_can_fail_on_threshold_breach(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    out = tmp_path / "mapping_complexity.json"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {"semantic_op": "a", "family": "index_scatter_gather", "intent_ops": ["gather"]},
                    {"semantic_op": "b", "family": "index_scatter_gather", "intent_ops": ["scatter"]},
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
            "index_scatter_gather",
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
