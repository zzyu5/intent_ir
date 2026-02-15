from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_validate_catalog_passes_on_repo_catalog(tmp_path: Path) -> None:
    out = tmp_path / "catalog_validation.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/validate_catalog.py",
            "--catalog",
            "scripts/CATALOG.json",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert int(payload["active_count"]) > 0
