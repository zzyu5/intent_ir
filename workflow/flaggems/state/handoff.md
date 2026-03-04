# FlagGems Session Handoff

- Timestamp: 2026-03-04T10:29:40+00:00
- Commit: `05bbbe0acd5a95b4d3274c5baefee701ad110c24`
- Lane: `workflow`
- Summary: cuda cpp_plugin flash_attention2d v6: sm89 ratio~0.94 (strict,llc); masked warp~0.90
- Batch Ops (0): (none)
- Run Summary: `artifacts/flaggems_matrix/daily/20260304/gpu_perf_triton_native_cuda_cpp_attn2d_sm89_v4_v6ok/run_summary.json`
- Status Converged: `artifacts/flaggems_matrix/daily/20260304/gpu_perf_triton_native_cuda_cpp_attn2d_sm89_v4_v6ok/status_converged.json`
- Evidence Paths: artifacts/flaggems_matrix/daily/20260304/gpu_perf_triton_native_cuda_cpp_attn2d_sm89_v4_v6ok/gpu_perf_graph.json, artifacts/flaggems_matrix/daily/20260304/gpu_perf_triton_native_cuda_cpp_attn2d_sm89_v4_v6ok/run_summary.json, artifacts/flaggems_matrix/daily/20260304/gpu_perf_triton_native_cuda_cpp_attn2d_sm89_v4_v6ok/status_converged.json, artifacts/validation_rounds/20260304/triton_native_cuda_cpp_attn2d_cov_sm89_v5_v6ok/flash_attention2d.intentir.intentdialect.downstream_cuda_std_cpp_llvm.contract.json, artifacts/validation_rounds/20260304/triton_native_cuda_cpp_attn2d_cov_sm89_v5_v6ok/flash_attention2d.intentir.intentdialect.downstream_cuda_std_cpp_llvm.kernel.ptx, artifacts/validation_rounds/20260304/triton_native_cuda_cpp_attn2d_cov_sm89_v5_v6ok/flash_attention2d.json, artifacts/validation_rounds/20260304/triton_native_cuda_cpp_attn2d_cov_sm89_v5_v6ok/masked_attention2d.intentir.intentdialect.downstream_cuda_std_cpp_llvm.contract.json, artifacts/validation_rounds/20260304/triton_native_cuda_cpp_attn2d_cov_sm89_v5_v6ok/masked_attention2d.intentir.intentdialect.downstream_cuda_std_cpp_llvm.kernel.ptx, artifacts/validation_rounds/20260304/triton_native_cuda_cpp_attn2d_cov_sm89_v5_v6ok/masked_attention2d.json
- Next Focus: run attn2d (flash/masked) on sm90/sm120 + add _attn_fwd cpp_plugin
