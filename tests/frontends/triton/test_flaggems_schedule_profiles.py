from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_export_schedule_profiles_script(tmp_path: Path) -> None:
    out = tmp_path / "schedule_profiles.json"
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/export_schedule_profiles.py",
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
    assert payload["schema_version"] == "flaggems_schedule_profiles_v1"
    assert "matmul_conv" in payload["profiles"]["cuda"]
    assert "elementwise_reduction" in payload["profiles"]["cuda"]
    assert "matmul_conv" in payload["profiles"]["rvv"]
    assert "elementwise_reduction" in payload["profiles"]["rvv"]

