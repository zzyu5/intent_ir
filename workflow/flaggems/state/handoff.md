# FlagGems Session Handoff

- Timestamp: 2026-02-28T05:16:29+00:00
- Commit: `35b4b5f974049ac29dafe51f9478c29a4da117da`
- Lane: `workflow`
- Summary: triton-native real-MLIR perf smoke (38 kernels): min=0.2866 p50=0.998 (slow: _attn_fwd, flash_attention2d, ai_bench_matmul)
- Batch Ops (0): (none)
- Run Summary: `artifacts/flaggems_matrix/daily/20260228/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32/run_summary.json`
- Status Converged: `artifacts/flaggems_matrix/daily/20260228/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32/status_converged.json`
- Evidence Paths: artifacts/flaggems_matrix/daily/20260228/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32/gpu_perf_graph.json, artifacts/flaggems_matrix/daily/20260228/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32/run_summary.json, artifacts/flaggems_matrix/daily/20260228/gpu_perf_triton_native_wave9_realmlir_smoke_v7_attn_warp32/status_converged.json, artifacts/validation_rounds/20260228/triton_native_coverage_full_wave9_realmlir_v4_attn_warp32, intent_ir/mlir/pipelines/downstream_cuda_std_llvm.yaml, workflow/flaggems/state/cuda_real_mlir_wave9_kernels.json
- Next Focus: Perf: optimize attention/matmul real-MLIR kernels to >=0.8; add shape telemetry; then expand wave beyond triton-native 38.
