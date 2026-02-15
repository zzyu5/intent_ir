from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(registry_payload: dict, out_dir: Path) -> subprocess.CompletedProcess[str]:
    reg = out_dir / "registry.json"
    out = out_dir / "macro_report.json"
    reg.write_text(json.dumps(registry_payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "scripts/intentir/check_macro_composition.py",
            "--registry",
            str(reg),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )


def test_macro_composition_passes_for_shared_intent_ops(tmp_path: Path) -> None:
    p = _run(
        {
            "entries": [
                {"semantic_op": "add", "intent_ops": ["add"]},
                {"semantic_op": "upsample_bicubic2d_aa", "intent_ops": ["upsample_bicubic2d_aa"]},
            ]
        },
        tmp_path,
    )
    assert p.returncode == 0, p.stderr


def test_macro_composition_fails_for_provider_specific_names(tmp_path: Path) -> None:
    p = _run(
        {
            "entries": [
                {"semantic_op": "bad_semantic", "intent_ops": ["flaggems_magic_op"]},
            ]
        },
        tmp_path,
    )
    assert p.returncode != 0

