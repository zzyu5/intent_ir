# FlagGems Session Handoff

- Timestamp: 2026-03-05T20:15:24+00:00
- Commit: `8782a660b27f5f4cc2b0af328d3822537898f509`
- Lane: `workflow`
- Summary: sm120 autotune matmul_fused_epilogue2d: global_v1 BM16 BN16 BK32 ratio~1.20 (cpp_plugin, strict,llc)
- Batch Ops (0): (none)
- Run Summary: `artifacts/remote_import/20260306/sm120/tuning_matmul_fused_epilogue2d_sm120_v1/07_matmul_fused_epilogue_mma_tf32_global_v1_f1b00c720b/perf/run_summary.json`
- Status Converged: `artifacts/remote_import/20260306/sm120/tuning_matmul_fused_epilogue2d_sm120_v1/07_matmul_fused_epilogue_mma_tf32_global_v1_f1b00c720b/perf/status_converged.json`
- Evidence Paths: artifacts/remote_import/20260306/sm120/tuning_matmul_fused_epilogue2d_sm120_v1/07_matmul_fused_epilogue_mma_tf32_global_v1_f1b00c720b/perf/gpu_perf_graph.json, artifacts/remote_import/20260306/sm120/tuning_matmul_fused_epilogue2d_sm120_v1/07_matmul_fused_epilogue_mma_tf32_global_v1_f1b00c720b/perf/run_summary.json, artifacts/remote_import/20260306/sm120/tuning_matmul_fused_epilogue2d_sm120_v1/07_matmul_fused_epilogue_mma_tf32_global_v1_f1b00c720b/perf/status_converged.json, artifacts/remote_import/20260306/sm120/tuning_matmul_fused_epilogue2d_sm120_v1/recommended.jsonl
- Next Focus: H100(sm90) matmul_fused_epilogue2d >=1.05: extend candidate sweep (BM/BN=16) + consider global_v2 warp epilogue
