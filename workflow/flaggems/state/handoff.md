# FlagGems Session Handoff

- Timestamp: 2026-02-14T07:31:35+00:00
- Commit: `c043118-dirty`
- Summary: Rebased active attention/angle/bitwise batch to intentir auto mode, added SDPA tolerance primitive, and reran pipeline+RVV(local/remote)+CUDA scoped convergence.
- Batch Ops (10): ScaleDotProductAttention, angle, argmax, argmin, avg_pool2d, bitwise_and_scalar, bitwise_and_scalar_tensor, bitwise_and_tensor, bitwise_left_shift, bitwise_not
- Run Summary: `artifacts/flaggems_matrix/daily/20260214/batch_active10_attention_angle_bitwise_v6/run_summary.json`
- Status Converged: `artifacts/flaggems_matrix/daily/20260214/batch_active10_attention_angle_bitwise_v6/status_converged.json`
- Next Focus: 1) Resolve CUDA runtime_timeout for scaled_dot_product_attention/angle/bitwise_left_shift. 2) Keep provider-path plugin boundaries while adding non-timeout CUDA classifications. 3) Continue next backend_missing_ops batch after gate clean pass.
