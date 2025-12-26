"""
TileLang frontend full pipeline runner (PR#9 MVP).

This is intentionally thin; the orchestration lives in `pipeline/tilelang/core.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.tilelang.core import default_kernel_specs, run_pipeline_for_spec


def main() -> None:
    out_dir = ROOT / "artifacts" / "tilelang_full_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in default_kernel_specs():
        print(f"\n=== {spec.name} ===")
        report = run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=8)
        out_path = out_dir / f"{spec.name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Report:", out_path)
        diff_ok = bool((report.get("diff") or {}).get("ok"))
        contract_level = (report.get("contract") or {}).get("level")
        print(f"contract={contract_level} diff={'OK' if diff_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
