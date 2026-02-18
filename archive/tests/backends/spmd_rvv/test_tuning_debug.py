import math

from backends.spmd_rvv.analysis.cost_model import GEMMCostModel
from backends.spmd_rvv.analysis.hardware_profile import RVVHardwareProfile
from backends.spmd_rvv.analysis.tuning import TuningRequest, select_schedule
from intent_ir.ir import IntentFunction


def test_gemm_cost_model_verbose_has_basic_breakdown():
    prof = RVVHardwareProfile(num_cores=4, rvv_vlen_bits=256, frequency_ghz=2.0, mem_bandwidth_gbps=12.0)
    model = GEMMCostModel(prof, M=128, N=128, K=128, dtype_size=4)
    est, dbg = model.evaluate_tile_verbose(16, 32, 16)
    assert dbg["model"] == "gemm_roofline_v0"
    assert dbg["tile"]["tile_m"] == 16
    assert dbg["tile"]["tile_n"] == 32
    assert dbg["tile"]["tile_k"] == 16
    assert dbg["cache_level"] in {"L1", "L2", "L3", "DRAM"}
    assert 0.0 < float(dbg["achievable_gflops"])
    assert math.isclose(float(est.gflops), float(dbg["achievable_gflops"]), rel_tol=1e-9, abs_tol=0.0)


def test_select_schedule_can_emit_cost_model_debug():
    intent = IntentFunction.from_json_dict(
        {
            "name": "toy_gemm",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "C"}],
            "outputs": ["C"],
        }
    )
    prof = RVVHardwareProfile(num_cores=4, rvv_vlen_bits=256, frequency_ghz=2.0, mem_bandwidth_gbps=12.0)
    res = select_schedule(
        intent,
        shape_bindings={"M": 128, "N": 128, "K": 128},
        profile=prof,
        request=TuningRequest(debug=True),
        evidence=None,
    )
    assert res.debug is not None
    assert res.debug["kind"] in {"gemm", "program"}
    assert isinstance(res.debug.get("cost_model"), dict)

