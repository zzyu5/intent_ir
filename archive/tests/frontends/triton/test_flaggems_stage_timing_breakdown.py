from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_compute_stage_timing_breakdown_emits_expected_schema(tmp_path: Path) -> None:
    rvv = tmp_path / "rvv_local.json"
    cuda = tmp_path / "cuda_local.json"
    out = tmp_path / "stage_timing_breakdown.json"
    rvv.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "kernel": "add2d",
                        "reason_code": "ok",
                        "lower_ms": 1.0,
                        "compile_ms": 2.0,
                        "launch_ms": 3.0,
                        "total_ms": 6.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cuda.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "kernel": "add2d",
                        "reason_code": "ok",
                        "lower_ms": 1.5,
                        "compile_ms": 2.5,
                        "launch_ms": 3.5,
                        "total_ms": 7.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/compute_stage_timing_breakdown.py",
            "--rvv-json",
            str(rvv),
            "--cuda-json",
            str(cuda),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "flaggems_stage_timing_breakdown_v1"
    assert payload["ok"] is True
    assert payload["backends"]["rvv"]["kernel_count"] == 1
    assert payload["backends"]["cuda"]["kernel_count"] == 1
    assert payload["combined"]["kernel_count_total"] == 2

