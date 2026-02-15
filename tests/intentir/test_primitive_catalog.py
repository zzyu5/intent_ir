from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from intent_ir.ops.primitive_catalog import catalog_summary, is_allowed_primitive, primitive_names


ROOT = Path(__file__).resolve().parents[2]


def test_primitive_catalog_has_core_ops() -> None:
    names = primitive_names()
    assert "add" in names
    assert "matmul" in names
    assert "upsample_bicubic2d_aa" not in names
    assert is_allowed_primitive("upsample_bicubic2d_aa") is False
    assert is_allowed_primitive("upsample_bicubic2d_aa", include_macro=True) is True
    summary = dict(catalog_summary())
    assert int(summary["total"]) >= len(names)


def test_check_primitive_reuse_script_flags_unknown_ops(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    out = tmp_path / "reuse.json"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {"semantic_op": "a", "intent_ops": ["add"]},
                    {"semantic_op": "b", "intent_ops": ["flaggems_private_op"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/intentir/check_primitive_reuse.py",
            "--registry",
            str(registry),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["violations"]
