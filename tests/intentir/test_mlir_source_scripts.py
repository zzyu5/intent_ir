from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir

ROOT = Path(__file__).resolve().parents[2]


def _intent_add() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}},
            ],
            "outputs": ["z"],
        }
    )


def _intent_add_relu() -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": "add_relu2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "tmp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["x", "y"], "output": "tmp", "attrs": {}},
                {"op": "relu", "inputs": ["tmp"], "output": "z", "attrs": {}},
            ],
            "outputs": ["z"],
        }
    )


def _write_mlir_manifest(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    a = tmp_path / "a.mlir"
    b = tmp_path / "b.mlir"
    c = tmp_path / "c.mlir"
    a.write_text(to_mlir(_intent_add()).module_text, encoding="utf-8")
    b.write_text(to_mlir(_intent_add_relu()).module_text, encoding="utf-8")
    c.write_text(to_mlir(_intent_add()).module_text, encoding="utf-8")
    entries = [
        {"semantic_op": "op_a", "family": "elementwise_broadcast", "module_path": str(a)},
        {"semantic_op": "op_b", "family": "conv_pool_interp", "module_path": str(b)},
        {"semantic_op": "op_c", "family": "attention_sequence", "module_path": str(c)},
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
    return manifest, entries


def test_check_primitive_reuse_from_mlir_manifest(tmp_path: Path) -> None:
    manifest, _ = _write_mlir_manifest(tmp_path)
    out = tmp_path / "reuse.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir/check_primitive_reuse.py",
            "--mlir-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr or p.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert str(payload["source"]).startswith("mlir_manifest:")
    assert payload["reused_primitives"]["add"] >= 1


def test_check_macro_composition_from_mlir_manifest(tmp_path: Path) -> None:
    manifest, _ = _write_mlir_manifest(tmp_path)
    out = tmp_path / "macro.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir/check_macro_composition.py",
            "--mlir-manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr or p.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert str(payload["source"]).startswith("mlir_manifest:")


def test_mapping_complexity_from_mlir_manifest(tmp_path: Path) -> None:
    manifest, _ = _write_mlir_manifest(tmp_path)
    out = tmp_path / "complexity.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir/report_mapping_complexity.py",
            "--mlir-manifest",
            str(manifest),
            "--out",
            str(out),
            "--complex-families",
            "conv_pool_interp,attention_sequence",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr or p.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["total"] == 3
    assert payload["single_intent_ops"] == 2
    assert payload["multi_intent_ops"] == 1
    assert payload["complex_total"] == 2
    assert payload["complex_single_intent_ops"] == 1
    assert payload["complex_family_single_semantic_ratio"] == 0.5
    assert str(payload["source"]).startswith("mlir_manifest:")

