from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _write_backend_json(path: Path, *, kernel: str, lower: float, compile_ms: float, launch: float, total: float) -> None:
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "kernel": kernel,
                        "lower_ms": lower,
                        "compile_ms": compile_ms,
                        "launch_ms": launch,
                        "total_ms": total,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_compute_stage_timing_delta_script(tmp_path: Path) -> None:
    cur_rvv = tmp_path / "cur_rvv.json"
    cur_cuda = tmp_path / "cur_cuda.json"
    base_rvv = tmp_path / "base_rvv.json"
    base_cuda = tmp_path / "base_cuda.json"
    out = tmp_path / "timing_delta.json"

    _write_backend_json(cur_rvv, kernel="add2d", lower=10.0, compile_ms=20.0, launch=30.0, total=60.0)
    _write_backend_json(cur_cuda, kernel="add2d", lower=11.0, compile_ms=22.0, launch=33.0, total=66.0)
    _write_backend_json(base_rvv, kernel="add2d", lower=8.0, compile_ms=16.0, launch=24.0, total=48.0)
    _write_backend_json(base_cuda, kernel="add2d", lower=10.0, compile_ms=20.0, launch=30.0, total=60.0)

    p = subprocess.run(
        [
            sys.executable,
            "scripts/flaggems/compute_stage_timing_delta.py",
            "--current-rvv",
            str(cur_rvv),
            "--current-cuda",
            str(cur_cuda),
            "--baseline-rvv",
            str(base_rvv),
            "--baseline-cuda",
            str(base_cuda),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rvv"]["matched_kernels"] == 1
    assert payload["cuda"]["matched_kernels"] == 1
    rvv_row = payload["rvv"]["rows"][0]
    assert rvv_row["total_ms"]["delta_ms"] == 12.0

