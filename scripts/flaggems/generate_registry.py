"""
Generate and freeze FlagGems semantic registry from `flag_gems.ops.__all__`.

Example:
  PYTHONPATH=. python scripts/flaggems/generate_registry.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.flaggems_registry import (  # noqa: E402
    DEFAULT_FLAGGEMS_OPSET,
    DEFAULT_REGISTRY_PATH,
    build_registry,
    infer_flaggems_commit_from_src,
    load_flaggems_all_ops,
    write_registry,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_REGISTRY_PATH, help="Output registry JSON path.")
    ap.add_argument("--flaggems-src", type=Path, default=(ROOT / "experiment" / "FlagGems" / "src"))
    ap.add_argument("--flaggems-opset", choices=[DEFAULT_FLAGGEMS_OPSET], default=DEFAULT_FLAGGEMS_OPSET)
    ap.add_argument("--flaggems-commit", type=str, default=None, help="Override FlagGems commit hash in metadata.")
    args = ap.parse_args()

    all_ops = load_flaggems_all_ops(flaggems_src=args.flaggems_src)
    commit = str(args.flaggems_commit) if args.flaggems_commit else infer_flaggems_commit_from_src(args.flaggems_src)
    payload = build_registry(
        all_ops=all_ops,
        flaggems_commit=commit,
        flaggems_source=str(args.flaggems_src),
        opset=str(args.flaggems_opset),
    )
    out = write_registry(args.output, payload)
    counts = payload.get("counts") or {}
    print(f"Registry written: {out}")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
