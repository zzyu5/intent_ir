// Extracted from intentir_cuda_codegen.cpp (wave3 modular split)
json lower_intent_to_cuda(const Intent& intent, const json& bindings_json) {
  if (!bindings_json.is_object()) fail("bindings must be object");

  #include "dispatch_direct_kernels.inc"
  #include "dispatch_pattern_kernels.inc"
  fail(
      "unsupported intent for cuda cpp codegen (supported: addmv, matmul, dropout, softmax, layernorm, correlation, resize, rope, warp, transpose, "
      "reduce_sum, reduce_max, reduce_min, reduce_mean_pattern, argmax, argmin, any_dim, all_dim, gather, diag, diag_embed, nonzero, count_nonzero, topk, trace, "
      "conv1d, conv2d(+bias pattern), conv3d, conv_depthwise2d, avg_pool2d, max_pool2d_with_indices, upsample_nearest1d, upsample_nearest2d, upsample_bicubic2d_aa, "
      "concat, pad, mse_loss, bmm, baddbmm, addmm, cumsum, normed_cumsum, cummax, cummin, dot, allclose, batch_norm(training), group_norm, "
      "rms_norm, var_mean, vector_norm, weight_norm, layer_norm_persistent, "
      "index_add, index_put, scatter, select_scatter, slice_scatter, stack, masked_scatter, masked_select, nll_loss_forward, nll_loss2d_forward, reduce_prod, quantile, polar, "
      "flash_attn_varlen_func, glu, scaled_dot_product_attention, fused_elementwise)");
}
