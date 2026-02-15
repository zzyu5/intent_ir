"""
Export backend schedule profiles by op family from staged pipeline drivers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.cuda.pipeline.driver import run_cuda_pipeline
from backends.spmd_rvv.pipeline.driver import run_rvv_pipeline
from intent_ir.ir import IntentFunction


def _build_intent(family: str, *, backend_tag: str) -> IntentFunction:
    if family == "matmul_conv":
        return IntentFunction.from_json_dict(
            {
                "name": f"{backend_tag}_{family}",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                    "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                    "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
                "outputs": ["C"],
                "parallel_axes": ["M", "N"],
                "schedule": {"tile_m": 32, "tile_n": 64, "tile_k": 16, "parallel_axes": ["M", "N"]},
            }
        )
    if family == "elementwise_reduction":
        return IntentFunction.from_json_dict(
            {
                "name": f"{backend_tag}_{family}",
                "tensors": {
                    "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "O": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "add", "inputs": ["X", "Y"], "output": "O", "attrs": {}}],
                "outputs": ["O"],
                "parallel_axes": ["M", "N"],
                "schedule": {"tile_n": 256, "parallel_axes": ["M", "N"]},
            }
        )
    raise ValueError(f"unsupported family: {family}")


def _schedule_stage(result: Any) -> dict[str, Any]:
    stages = list(getattr(result, "stages", []) or [])
    for st in stages:
        if str(getattr(st, "name", "")) == "schedule":
            artifacts = getattr(st, "artifacts", {})
            return dict(artifacts) if isinstance(artifacts, dict) else {}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    families = ["matmul_conv", "elementwise_reduction"]
    backends = {
        "cuda": run_cuda_pipeline,
        "rvv": run_rvv_pipeline,
    }
    payload: dict[str, Any] = {
        "ok": True,
        "schema_version": "flaggems_schedule_profiles_v1",
        "profiles": {},
        "missing": [],
    }
    for backend_name, runner in backends.items():
        backend_profiles: dict[str, Any] = {}
        for family in families:
            intent = _build_intent(family, backend_tag=backend_name)
            result = runner(intent)
            stage = _schedule_stage(result)
            if not bool(getattr(result, "ok", False)) or not stage:
                payload["missing"].append({"backend": backend_name, "family": family, "reason": "missing_schedule_stage"})
                continue
            backend_profiles[family] = {
                "schedule_profile": stage.get("schedule_profile"),
                "op_family": stage.get("op_family"),
                "schedule_hints": stage.get("schedule_hints"),
                "rewrite_aware": bool(stage.get("rewrite_aware")),
            }
        payload["profiles"][backend_name] = backend_profiles

    required_pairs = [(b, f) for b in backends.keys() for f in families]
    missing_pairs = [
        (b, f)
        for b, f in required_pairs
        if not isinstance((payload.get("profiles", {}).get(b, {})).get(f), dict)
    ]
    if missing_pairs:
        payload["ok"] = False
        for b, f in missing_pairs:
            payload["missing"].append({"backend": b, "family": f, "reason": "profile_not_exported"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Schedule profile export written: {args.out}")
    raise SystemExit(0 if bool(payload["ok"]) else 1)


if __name__ == "__main__":
    main()
