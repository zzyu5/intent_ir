"""
Paper utility: summarize mutation-kill outcomes across the default 6 kernels.

This does NOT re-run mutation generation (which may be expensive). It only reads
existing pipeline artifacts under:
  - artifacts/full_pipeline_verify/*.json (Triton)
  - artifacts/tilelang_full_pipeline/*.json (TileLang)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_KERNELS = [
    "any_kernel_dim",
    "group_norm_kernel",
    "_attn_fwd",
    "softmax_inner",
    "layer_norm_persistent",
    "upsample_bicubic2d_aa",
]

STAGE_ORDER = ["invalid", "A_static", "B_diff", "C_metamorphic", "C_bounded"]


def _artifact_dir(frontend: str) -> str:
    return "full_pipeline_verify" if frontend == "triton" else "tilelang_full_pipeline"


def _load_report(frontend: str, kernel: str) -> Optional[Dict[str, Any]]:
    p = ROOT / "artifacts" / _artifact_dir(frontend) / f"{kernel}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _summarize_one(report: Dict[str, Any]) -> Dict[str, Any]:
    mk = report.get("mutation_kill") or {}
    if isinstance(mk, dict) and mk.get("skipped"):
        return {"skipped": True, "reason": mk.get("reason")}
    if not isinstance(mk, dict):
        return {"skipped": True, "reason": "missing mutation_kill"}

    total = int(mk.get("total") or 0)
    killed = int(mk.get("killed") or 0)
    survived = int(mk.get("survived") or 0)
    killed_by_stage = mk.get("killed_by_stage") or {}
    if not isinstance(killed_by_stage, dict):
        killed_by_stage = {}
    stage_counts = {str(k): int(v) for k, v in killed_by_stage.items() if isinstance(v, (int, float))}

    cumulative = []
    acc = 0
    for st in STAGE_ORDER:
        acc += int(stage_counts.get(st, 0))
        cumulative.append(
            {
                "stage": st,
                "killed": int(acc),
                "kill_rate": (float(acc) / float(total) if total > 0 else 0.0),
            }
        )

    return {
        "skipped": False,
        "total": total,
        "killed": killed,
        "survived": survived,
        "kill_rate": (float(killed) / float(total) if total > 0 else 0.0),
        "killed_by_stage": {k: int(stage_counts.get(k, 0)) for k in STAGE_ORDER if stage_counts.get(k)},
        "cumulative": cumulative,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "both"], default="both")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; default runs 6 kernels")
    ap.add_argument("--out", default=None, help="write JSON summary to this path (default: stdout)")
    args = ap.parse_args()

    kernels = args.kernel or list(DEFAULT_KERNELS)
    frontends = ["triton", "tilelang"] if args.frontend == "both" else [str(args.frontend)]

    out: Dict[str, Any] = {"kernels": kernels, "frontends": frontends, "results": {}}
    for fe in frontends:
        per_kernel = []
        agg_total = agg_killed = 0
        agg_stage: Dict[str, int] = {k: 0 for k in STAGE_ORDER}
        for k in kernels:
            report = _load_report(fe, k)
            if report is None:
                per_kernel.append({"kernel": k, "missing": True})
                continue
            summ = _summarize_one(report)
            per_kernel.append({"kernel": k, **summ})
            if not summ.get("skipped") and not summ.get("missing"):
                agg_total += int(summ.get("total") or 0)
                agg_killed += int(summ.get("killed") or 0)
                for st in STAGE_ORDER:
                    agg_stage[st] += int((summ.get("killed_by_stage") or {}).get(st) or 0)
        out["results"][fe] = {
            "per_kernel": per_kernel,
            "aggregate": {
                "total": int(agg_total),
                "killed": int(agg_killed),
                "kill_rate": (float(agg_killed) / float(agg_total) if agg_total > 0 else 0.0),
                "killed_by_stage": {k: int(v) for k, v in agg_stage.items() if v},
            },
        }

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
