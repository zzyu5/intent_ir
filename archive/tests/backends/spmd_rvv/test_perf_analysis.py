from backends.spmd_rvv.analysis.perf_analysis import (
    extract_remote_suite_rows,
    predicted_gflops_from_tuning_debug,
    predicted_ms_from_tuning_debug,
    summarize_rows,
)


def test_predicted_ms_program():
    dbg = {"kind": "program", "cost_model": {"total": {"ms": 1.25, "gflops": 0.5}}}
    assert predicted_ms_from_tuning_debug(dbg) == 1.25
    assert predicted_gflops_from_tuning_debug(dbg) == 0.5


def test_predicted_ms_gemm():
    dbg = {"kind": "gemm", "model_mnk": [128, 128, 128], "cost_model": {"achievable_gflops": 16.0}}
    ms = predicted_ms_from_tuning_debug(dbg)
    assert ms is not None
    # 2*M*N*K / (GFLOPs*1e6) = 2*128^3 / (16e6) = 0.262144 ms
    assert abs(ms - 0.262144) < 1e-6
    assert predicted_gflops_from_tuning_debug(dbg) == 16.0


def test_extract_remote_suite_rows_and_summary():
    raw = {
        "results": [
            {
                "frontend": "triton",
                "kernel": "k0",
                "ok": True,
                "bench": {"ns_per_iter": 1000000.0, "matmul_gflops": 12.0},
                "profile_ops": {"total_ns": 900000.0, "ops": [{"name": "op[0] add -> y", "ns": 900000.0}]},
                "tuning": {"debug": {"kind": "program", "cost_model": {"total": {"ms": 1.1, "gflops": 0.3}}}},
            },
            {
                "frontend": "triton",
                "kernel": "k1",
                "ok": True,
                "bench": {"ns_per_iter": 2000000.0, "matmul_gflops": 6.0},
                "tuning": {"debug": {"kind": "program", "cost_model": {"total": {"ms": 2.0, "gflops": 0.2}}}},
            },
        ]
    }
    rows, errors = extract_remote_suite_rows(raw)
    assert not errors
    assert rows[0]["pred_ms"] == 1.1
    assert rows[0]["measured_ms"] == 1.0
    assert rows[0]["profile_total_ns"] == 900000.0
    s = summarize_rows(rows)
    assert "overall" in s

