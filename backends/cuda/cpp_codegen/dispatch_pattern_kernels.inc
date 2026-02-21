  // Pattern-based kernels.
  if (intent.ops.size() == 3 && intent.ops[0].op == "const" && intent.ops[1].op == "ne" && intent.ops[2].op == "reduce_any") {
    return emit_any_dim_f32_to_i1(intent, bindings_json);
  }
  if (intent.ops.size() == 5 && intent.ops[0].op == "broadcast_in_dim" && intent.ops[1].op == "broadcast_in_dim" && intent.ops[2].op == "ne" &&
      intent.ops[3].op == "not" && intent.ops[4].op == "reduce_any") {
    return emit_isin1d_i32_bool(intent, bindings_json);
  }
  if (intent.ops.size() == 4 && intent.ops[0].op == "const" && intent.ops[1].op == "eq" && intent.ops[2].op == "reduce_any" &&
      intent.ops[3].op == "not") {
    return emit_all_dim_f32_to_i1(intent, bindings_json);
  }
  if (intent.ops.size() == 6 && intent.ops[0].op == "transpose" && intent.ops[1].op == "matmul" && intent.ops[2].op == "mul" &&
      intent.ops[3].op == "add" && intent.ops[4].op == "softmax" && intent.ops[5].op == "matmul") {
    return emit_flash_attn_varlen_func_bhsd_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 4 && intent.ops[0].op == "cumsum" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "add" &&
      intent.ops[3].op == "div") {
    return emit_normed_cumsum2d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 2 && intent.ops[0].op == "std" && intent.ops[1].op == "mul") {
    return emit_var_mean2d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 9 && intent.ops[0].op == "const" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "broadcast_in_dim" &&
      intent.ops[3].op == "sub" && intent.ops[4].op == "mul" && intent.ops[5].op == "reduce_sum" && intent.ops[6].op == "sub" &&
      intent.ops[7].op == "div" && intent.ops[8].op == "identity") {
    return emit_var_mean2d_f32(intent, bindings_json);
  }
  {
    bool vector_norm_like = false;
    if (intent.ops.size() == 3) {
      vector_norm_like = (intent.ops[0].op == "mul" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "sqrt");
    } else if (intent.ops.size() == 6) {
      vector_norm_like = (intent.ops[0].op == "abs" && intent.ops[1].op == "mul" && intent.ops[2].op == "reduce_sum" &&
                          intent.ops[3].op == "rsqrt" && intent.ops[4].op == "const" && intent.ops[5].op == "div");
    }
    if (vector_norm_like) return emit_vector_norm2d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 6 && intent.ops[0].op == "mul" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "sqrt" &&
      intent.ops[3].op == "div" && intent.ops[4].op == "broadcast_in_dim" && intent.ops[5].op == "mul") {
    return emit_weight_norm2d_f32(intent, bindings_json);
  }
  {
    bool allclose_like = false;
    if (intent.ops.size() == 9) {
      allclose_like = (intent.ops[0].op == "sub" && intent.ops[1].op == "abs" && intent.ops[2].op == "abs" && intent.ops[3].op == "mul" &&
                       intent.ops[4].op == "add" && (intent.ops[5].op == "le" || intent.ops[5].op == "ge") && intent.ops[6].op == "not" &&
                       intent.ops[7].op == "reduce_any" && intent.ops[8].op == "not");
    }
    if (allclose_like) return emit_allclose2d_f32_bool(intent, bindings_json);
  }
  if (intent.ops.size() == 4 && intent.ops[0].op == "cast" && intent.ops[1].op == "cast" && intent.ops[2].op == "mul" &&
      intent.ops[3].op == "reduce_sum") {
    return emit_dot1d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 25) {
    bool bn_like = true;
    static const std::vector<std::string> kBatchNormOps = {
        "const", "reduce_sum", "div", "broadcast_in_dim", "sub", "mul", "reduce_sum", "div", "add", "rsqrt", "broadcast_in_dim", "mul", "broadcast_in_dim",
        "mul", "broadcast_in_dim", "add", "sub", "mul", "mul", "add", "div", "mul", "mul", "mul", "add"};
    for (size_t i = 0; i < kBatchNormOps.size(); ++i) {
      if (intent.ops[i].op != kBatchNormOps[i]) {
        bn_like = false;
        break;
      }
    }
    if (bn_like) return emit_batch_norm2d_training_f32(intent, bindings_json);
  }
  if (intent.ops.size() >= 25 && intent.ops.size() <= 26 && intent.outputs.size() == 3 && intent.outputs[0] == "Y" && intent.outputs[1] == "Mean" &&
      intent.outputs[2] == "Rstd") {
    return emit_group_norm_kernel_f32(intent, bindings_json);
  }
  if (intent.outputs.size() == 2 && intent.outputs[0] == "out" && intent.outputs[1] == "INV_RMS") {
    bool has_reduce_sum = false;
    bool has_rsqrt = false;
    for (const auto& op : intent.ops) {
      if (op.op == "reduce_sum") has_reduce_sum = true;
      if (op.op == "rsqrt") has_rsqrt = true;
    }
    if (has_reduce_sum && has_rsqrt) return emit_rms_norm2d_f32(intent, bindings_json);
  }
  if (intent.ops.size() >= 17 && intent.ops.size() <= 21 && intent.outputs.size() == 3 && intent.outputs[0] == "out_ptr" && intent.outputs[1] == "out_mean_ptr" &&
      intent.outputs[2] == "out_rstd_ptr") {
    return emit_layer_norm_persistent_f32(intent, bindings_json);
  }
  {
    bool hasY = false, hasMean = false, hasRstd = false;
    for (const auto& o : intent.outputs) {
      if (o == "Y") hasY = true;
      if (o == "Mean") hasMean = true;
      if (o == "Rstd") hasRstd = true;
    }
    bool rank2_x = false;
    auto x_it = intent.tensors.find("X");
    if (x_it != intent.tensors.end()) {
      rank2_x = (x_it->second.shape.size() == 2);
    }
    if (hasY && hasMean && hasRstd && rank2_x) return emit_layernorm_2d_f32(intent, bindings_json);
  }
  {
    bool diag_like = false;
    if (intent.ops.size() == 2) {
      diag_like = (intent.ops[0].op == "iota" && intent.ops[1].op == "gather" && intent.ops[1].inputs.size() == 3 &&
                   intent.ops[1].inputs[1] == intent.ops[0].output && intent.ops[1].inputs[2] == intent.ops[0].output);
    }
    if (diag_like) return emit_diag2d_extract_f32(intent, bindings_json);
  }
  {
    bool diag_embed_like = false;
    if (intent.ops.size() == 7) {
      diag_embed_like = (intent.ops[0].op == "const" && intent.ops[1].op == "broadcast_in_dim" && intent.ops[2].op == "iota" &&
                         intent.ops[3].op == "iota" && intent.ops[4].op == "eq" && intent.ops[5].op == "broadcast_in_dim" &&
                         intent.ops[6].op == "where");
    }
    if (diag_embed_like) return emit_diag_embed2d_f32(intent, bindings_json);
  }
  {
    bool nonzero_like = false;
    if (intent.ops.size() == 1) nonzero_like = (intent.ops[0].op == "nonzero");
    if (nonzero_like) return emit_nonzero2d_f32(intent, bindings_json);
  }
  {
    bool count_nonzero_like = false;
    if (intent.ops.size() == 4) {
      count_nonzero_like = (intent.ops[0].op == "const" && intent.ops[1].op == "ne" && intent.ops[2].op == "cast" &&
                            intent.ops[3].op == "reduce_sum");
    }
    if (count_nonzero_like) return emit_count_nonzero2d_f32_i64(intent, bindings_json);
  }
  {
    bool topk_like = false;
    if (intent.ops.size() == 4) {
      topk_like = (intent.ops[0].op == "sort" && intent.ops[1].op == "iota" && intent.ops[2].op == "iota" &&
                   intent.ops[3].op == "gather");
    }
    if (topk_like) return emit_topk2d_rowwise_f32(intent, bindings_json);
  }
  {
    bool trace_like = false;
    if (intent.ops.size() == 6) {
      trace_like = (intent.ops[0].op == "iota" && intent.ops[1].op == "iota" && intent.ops[2].op == "eq" &&
                    intent.ops[3].op == "const" && intent.ops[4].op == "where" && intent.ops[5].op == "reduce_sum");
    }
    if (trace_like) return emit_trace2d_sum_f32(intent, bindings_json);
  }
  {
    bool concat_like = (intent.ops.size() == 1 && intent.ops[0].op == "concat");
    if (concat_like) return emit_concat2d_f32(intent, bindings_json);
  }
  {
    bool pad_like = (intent.ops.size() == 1 && intent.ops[0].op == "pad");
    if (pad_like) return emit_pad2d_const_f32(intent, bindings_json);
  }
  {
    bool mse_loss_like = (intent.ops.size() == 1 && intent.ops[0].op == "mse_loss");
    if (mse_loss_like) return emit_mse_loss2d_f32(intent, bindings_json);
  }
  {
    bool per_token_group_quant_fp8_like = (intent.ops.size() == 11 && intent.ops[0].op == "reshape" && intent.ops[1].op == "abs" &&
                                           intent.ops[2].op == "reduce_max" && intent.ops[3].op == "max" && intent.ops[4].op == "div" &&
                                           intent.ops[5].op == "reshape" && intent.ops[6].op == "broadcast_in_dim" && intent.ops[7].op == "div" &&
                                           intent.ops[8].op == "max" && intent.ops[9].op == "min" && intent.ops[10].op == "reshape");
    if (per_token_group_quant_fp8_like) return emit_per_token_group_quant_fp8_2d_f32(intent, bindings_json);
  }
  {
    const std::string nm = ascii_lower(intent.name);
    bool is_softmax = (nm.find("softmax") != std::string::npos);
    if (!is_softmax) {
      bool has_reduce_max = false, has_reduce_sum = false, has_exp = false, has_div = false;
      for (const auto& op : intent.ops) {
        if (op.op == "reduce_max") has_reduce_max = true;
        if (op.op == "reduce_sum") has_reduce_sum = true;
        if (op.op == "exp") has_exp = true;
        if (op.op == "div") has_div = true;
      }
      is_softmax = has_reduce_max && has_reduce_sum && has_exp && has_div;
    }
    if (is_softmax) return emit_softmax_2d_last_f32(intent, bindings_json);
  }
  {
    auto is_elem_op = [](const std::string& op) -> bool {
      static const std::unordered_map<std::string, bool> k = {
          {"const", true}, {"iota", true},     {"identity", true},         {"broadcast_in_dim", true}, {"cast", true},
          {"add", true},   {"sub", true},      {"mul", true},              {"div", true},              {"max", true},
          {"min", true},   {"remainder", true}, {"pow", true},              {"relu", true},             {"abs", true},              {"sin", true},              {"cos", true},
          {"tan", true},   {"erf", true},      {"exp", true},              {"acos", true},             {"atan", true},             {"log", true},
          {"ceil", true},  {"sqrt", true},     {"floor", true},            {"rsqrt", true},            {"eq", true},               {"ne", true},
          {"lt", true},    {"le", true},       {"gt", true},               {"ge", true},               {"and", true},
          {"or", true},    {"not", true},      {"bitwise_and", true},      {"bitwise_or", true},
          {"bitwise_not", true},               {"bitwise_left_shift", true}, {"bitwise_right_shift", true}, {"where", true},
      };
      return k.find(op) != k.end();
    };
    bool all_elem = !intent.ops.empty();
    for (const auto& op : intent.ops) {
      if (!is_elem_op(op.op)) {
        all_elem = false;
        break;
      }
    }
    if (all_elem) return emit_fused_elementwise(intent, bindings_json);
  }
