from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_install_systemd_nightly_dry_run() -> None:
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/install_systemd_nightly.py",
            "--dry-run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert "would write" in p.stdout
