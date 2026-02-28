from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / "scripts" / "flaggems" / "run_gpu_perf_graph.py"


def _load_perf_runner_module():
    spec = importlib.util.spec_from_file_location("run_gpu_perf_graph", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gpu_perf_native_baseline_mv2d_binds_matrix_not_inp_vector() -> None:
    mod = _load_perf_runner_module()
    spec_map = mod._coverage_spec_map()
    spec_entry = spec_map.get("mv2d")
    if not isinstance(spec_entry, dict) or str(spec_entry.get("source") or "") != "flaggems_native":
        pytest.skip("flaggems_native mv2d coverage spec unavailable")

    m = 16
    n = 32
    inputs_np = {
        "A": np.random.randn(n, m).astype(np.float32),
        "B": np.random.randn(m).astype(np.float32),
        "Inp": np.zeros((n,), dtype=np.float32),
        "alpha": np.array(1.0, dtype=np.float32),
        "beta": np.array(0.0, dtype=np.float32),
        "C": np.empty((n,), dtype=np.float32),
    }
    fn, _module_name, meta = mod._build_native_launch_fn(
        kernel="mv2d",
        inputs_np=inputs_np,
        bindings={"M": m, "N": n},
        spec_map=spec_map,
        device="cpu",
    )
    assert str(meta.get("launch_source") or "") == "torch.mv:out"
    fn()

