"""
Generate and freeze FlagGems semantic registry from `flag_gems.ops.__all__`.

Example:
  # Default: import the installed `flag_gems` package
  PYTHONPATH=. python scripts/flaggems/generate_registry.py

  # If you want an exact upstream git commit recorded in the registry metadata
  PYTHONPATH=. python scripts/flaggems/generate_registry.py --flag-gems-repo-path archive/experiment/FlagGems
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.providers.flaggems.registry import (  # noqa: E402
    DEFAULT_FLAGGEMS_OPSET,
    DEFAULT_REGISTRY_PATH,
    build_registry,
    infer_flaggems_commit_from_src,
    load_flaggems_all_ops,
    write_registry,
)

_ARCHIVE_FLAGGEMS_REPO_DEFAULT = ROOT / "archive" / "experiment" / "FlagGems"


def _infer_pip_tag() -> str | None:
    try:
        import importlib.metadata as md  # py>=3.8  # noqa: PLC0415

        for name in ("flag-gems", "flag_gems"):
            try:
                v = md.version(name)
                if v:
                    return f"pip:{name}=={v}"
            except md.PackageNotFoundError:
                pass
    except Exception:
        pass
    try:
        flag_gems = importlib.import_module("flag_gems")
        v = getattr(flag_gems, "__version__", None)
        if isinstance(v, str) and v.strip():
            return f"pip:flag_gems=={v.strip()}"
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_REGISTRY_PATH, help="Output registry JSON path.")
    ap.add_argument(
        "--flag-gems-repo-path",
        type=Path,
        default=None,
        help=(
            "Optional FlagGems repo root (used to import from <repo>/src and infer a git commit). "
            "If omitted, we import the installed `flag_gems` Python package."
        ),
    )
    ap.add_argument(
        "--flaggems-src",
        type=Path,
        default=None,
        help="(deprecated) Explicit FlagGems src/ directory. Prefer --flag-gems-repo-path.",
    )
    ap.add_argument("--flaggems-opset", choices=[DEFAULT_FLAGGEMS_OPSET], default=DEFAULT_FLAGGEMS_OPSET)
    ap.add_argument("--flaggems-commit", type=str, default=None, help="Override FlagGems commit hash in metadata.")
    args = ap.parse_args()

    commit = str(args.flaggems_commit).strip() if args.flaggems_commit else ""
    source_label = "python:flag_gems"

    if args.flag_gems_repo_path is not None or args.flaggems_src is not None:
        repo_or_src = args.flag_gems_repo_path if args.flag_gems_repo_path is not None else args.flaggems_src
        assert repo_or_src is not None
        p = Path(repo_or_src).resolve()
        src = p if p.name == "src" else (p / "src")
        source_label = str(src)
        all_ops = load_flaggems_all_ops(flaggems_src=src)
        if not commit:
            commit = str(infer_flaggems_commit_from_src(src) or "")
        if not commit:
            raise RuntimeError(f"unable to resolve FlagGems commit from {src}; pass --flaggems-commit explicitly")
    else:
        # Best default for new users: use the installed python package.
        try:
            all_ops = load_flaggems_all_ops(flaggems_src=None)
        except Exception as e:
            raise RuntimeError(
                "unable to import `flag_gems`; install the package (pip/uv/conda) or pass "
                f\"--flag-gems-repo-path {_ARCHIVE_FLAGGEMS_REPO_DEFAULT} (or another FlagGems checkout)\"  # noqa: ISC003
            ) from e
        if not commit:
            commit = _infer_pip_tag() or "unknown"
    payload = build_registry(
        all_ops=all_ops,
        flaggems_commit=commit,
        flaggems_source=source_label,
        opset=str(args.flaggems_opset),
    )
    out = write_registry(args.output, payload)
    counts = payload.get("counts") or {}
    print(f"Registry written: {out}")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
