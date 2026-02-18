from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _run_git(root: Path, args: list[str]) -> str:
    p = subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return ""
    return str(p.stdout or "").strip()


def repo_state(*, root: Path) -> dict[str, Any]:
    """
    Best-effort repo snapshot for artifact provenance.

    This is intentionally resilient: it returns partial info rather than raising.
    """
    head = _run_git(root, ["rev-parse", "HEAD"])
    branch = _run_git(root, ["branch", "--show-current"])
    describe = _run_git(root, ["describe", "--always", "--dirty", "--tags"])
    status = _run_git(root, ["status", "--porcelain"])
    dirty = bool(status.strip())
    return {
        "head_commit": head,
        "branch": branch,
        "dirty": dirty,
        "git_describe": describe,
    }

