"""
Triton frontend full pipeline runner (Tasks 1–5).

This file lives under `scripts/triton/` to keep Triton-specific entrypoints
separate from generic scripts (backend smoke, remote RVV run, etc).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.core import default_kernel_specs, run_pipeline_for_spec


def main() -> None:
    out_dir = ROOT / "artifacts" / "full_pipeline_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Optional: run a subset via `--kernel name` (can pass multiple).
    wanted = set()
    if "--kernel" in sys.argv:
        idxs = [i for i, a in enumerate(sys.argv) if a == "--kernel"]
        for i in idxs:
            if i + 1 < len(sys.argv):
                wanted.add(sys.argv[i + 1])
    if "--list" in sys.argv:
        for s in default_kernel_specs():
            print(s.name)
        return

    for spec in default_kernel_specs():
        if wanted and spec.name not in wanted:
            continue
        print(f"\n=== {spec.name} ===")
        try:
            report = run_pipeline_for_spec(spec, out_dir=out_dir, cases_limit=8)
        except Exception as e:
            print("Pipeline failed:", e)
            continue

        # Compact terminal summary
        contract_level = report.get("contract", {}).get("level")
        diff = report.get("diff", {})
        stage_c = report.get("stage_c", {})
        print(f"TTIR: {report.get('ttir_path', 'N/A')} | contract={contract_level}")
        if "certificate" in report and report["certificate"] and report["certificate"].get("obligations"):
            ob = report["certificate"]["obligations"]
            short = [f"{o['id']}:{o['status']}" for o in ob][:6]
            print("Certificate obligations:", ", ".join(short))
        print(f"Diff: {'OK' if diff.get('ok') else 'FAIL'}")
        if stage_c:
            meta_ok = stage_c.get("metamorphic", {}).get("ok", True)
            bounded = stage_c.get("bounded_exhaustive", {})
            bounded_state = "SKIP" if bounded.get("total", 0) == 0 else ("OK" if bounded.get("ok") else "FAIL")
            print(f"StageC metamorphic={meta_ok} bounded={bounded_state}")

        # Persist report JSON
        try:
            out_path = out_dir / f"{spec.name}.json"
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Report:", out_path)
        except Exception as e:
            print("Failed to write report:", e)


if __name__ == "__main__":
    main()
