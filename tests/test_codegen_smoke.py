from pathlib import Path

import pytest

from backends.spmd_rvv import SPMDProfile, choose_tiles  # noqa: E402
from backends.spmd_rvv.analysis.hardware_profile import RVVHardwareProfile  # noqa: E402
from backends.spmd_rvv.analysis.cost_model import GEMMCostModel  # noqa: E402
from backends.spmd_rvv.experiments.matmul_c import generate_c  # noqa: E402
from intent_ir.ir import IntentFunction  # noqa: E402


def _make_matmul_intent():
    data = {
        "name": "mm",
        "tensors": {
            "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
            "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
            "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
        },
        "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
        "outputs": ["C"],
        "parallel_axes": ["M", "N"],
        "schedule": {"tile_m": 16, "tile_n": 16, "tile_k": 16, "parallel_axes": ["M", "N"]},
    }
    return IntentFunction.from_json_dict(data)


def test_choose_tiles_returns_positive_ints():
    intent = _make_matmul_intent()
    profile = SPMDProfile(num_cores=8)
    tile = choose_tiles(intent, profile, constraints=None)
    assert tile.tile_m > 0 and tile.tile_n > 0 and tile.tile_k > 0
    # If rvv is enabled with vlen, vec_width should be set.
    profile_rvv = SPMDProfile(num_cores=8, rvv_enabled=True, rvv_vlen_bits=256)
    tile2 = choose_tiles(intent, profile_rvv, constraints=None)
    assert tile2.vec_width is not None and tile2.vec_width > 0
    # RVV profile + cost model path
    rvv_prof = RVVHardwareProfile(num_cores=4, rvv_vlen_bits=256, frequency_ghz=1.5, mem_bandwidth_gbps=12.0)
    model = GEMMCostModel(rvv_prof, M=128, N=128, K=128)
    est = model.evaluate_tile(16, 16, 16)
    assert est.gflops > 0


def test_codegen_emits_c_loops():
    intent = _make_matmul_intent()
    profile = SPMDProfile(num_cores=4)
    tile = choose_tiles(intent, profile, constraints=None)
    src = generate_c(intent, tile, profile)
    assert "void matmul_mm" in src
    for kw in ["#pragma omp", "for (int m0", "for (int n0", "for (int k0", "acc +="]:
        assert kw in src
    # Ensure signature arguments are present
    assert "const float* A" in src and "const float* B" in src and "float* C" in src
