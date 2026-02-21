  if (intent.ops.size() == 4 && intent.ops[0].op == "matmul" && intent.ops[1].op == "mul" && intent.ops[2].op == "mul" && intent.ops[3].op == "add") {
    auto a_it = intent.tensors.find(intent.ops[0].inputs[0]);
    auto b_it = intent.tensors.find(intent.ops[0].inputs[1]);
    if (b_it != intent.tensors.end() && b_it->second.shape.size() == 1) return emit_addmv_f32(intent, bindings_json);
    if (a_it != intent.tensors.end() && b_it != intent.tensors.end()) {
      if (a_it->second.shape.size() == 2 && b_it->second.shape.size() == 2) return emit_addmm2d_f32(intent, bindings_json);
      if (a_it->second.shape.size() == 3 && b_it->second.shape.size() == 3) return emit_baddbmm3d_f32(intent, bindings_json);
    }
    return emit_baddbmm3d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 10 && intent.ops[0].op == "cast" && intent.ops[1].op == "cast" && intent.ops[2].op == "broadcast_in_dim" &&
      intent.ops[3].op == "mul" && intent.ops[4].op == "reduce_sum" && intent.ops[5].op == "mul" && intent.ops[6].op == "cast" &&
      intent.ops[7].op == "mul" && intent.ops[8].op == "add" && intent.ops[9].op == "cast") {
    return emit_addmv_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 9 && (intent.ops[0].op == "cast" || intent.ops[0].op == "identity") &&
      (intent.ops[1].op == "cast" || intent.ops[1].op == "identity") && intent.ops[2].op == "mul" && intent.ops[3].op == "reduce_sum" &&
      intent.ops[4].op == "mul" && (intent.ops[5].op == "cast" || intent.ops[5].op == "identity") && intent.ops[6].op == "mul" &&
      intent.ops[7].op == "add" && (intent.ops[8].op == "cast" || intent.ops[8].op == "identity")) {
    return emit_addmv_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 5 && intent.ops[0].op == "matmul" && intent.ops[1].op == "mul" && intent.ops[2].op == "mul" && intent.ops[3].op == "add" &&
      (intent.ops[4].op == "cast" || intent.ops[4].op == "identity")) {
    return emit_addmm2d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "matmul") {
    const Op& mm = intent.ops[0];
    if (mm.inputs.size() == 2) {
      auto a_it = intent.tensors.find(mm.inputs[0]);
      auto b_it = intent.tensors.find(mm.inputs[1]);
      if (a_it != intent.tensors.end() && b_it != intent.tensors.end() && a_it->second.shape.size() == 3 && b_it->second.shape.size() == 3) {
        return emit_bmm3d_f32(intent, bindings_json);
      }
    }
    return emit_matmul_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 3 && intent.ops[0].op == "cast" && intent.ops[1].op == "cast" && intent.ops[2].op == "matmul") {
    return emit_bmm3d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 2 && intent.ops[0].op == "matmul" && intent.ops[1].op == "cast") {
    const Op& mm = intent.ops[0];
    const Op& cs = intent.ops[1];
    if (cs.inputs.size() == 1 && cs.inputs[0] == mm.output) {
      const std::string to_dtype = cs.attrs.value("to", std::string("f32"));
      if (to_dtype == "f32") {
        Intent tmp = intent;
        tmp.ops = {mm};
        tmp.ops[0].output = cs.output;
        return emit_matmul_f32(tmp, bindings_json);
      }
    }
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "transpose") return emit_transpose_2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "reduce_sum") return emit_reduce_sum_2d_axis1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "reduce_max") return emit_reduce_max_2d_axis1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "reduce_prod") {
    if (_reduce_axis_is_all_2d(intent.ops[0].attrs)) return emit_reduce_prod_2d_all_f32(intent, bindings_json);
    if (_reduce_axis_is_1(intent.ops[0].attrs)) return emit_reduce_prod_2d_axis1_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "std") return emit_std_2d_axis1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "reduce_min") {
    if (_reduce_axis_is_all_2d(intent.ops[0].attrs) && !_attrs_return_indices(intent.ops[0].attrs)) {
      return emit_reduce_min_2d_all_f32(intent, bindings_json);
    }
  }
  if (intent.ops.size() == 2 && intent.ops[0].op == "reduce_min" && intent.ops[1].op == "argmin") {
    const Op& red = intent.ops[0];
    const Op& arg = intent.ops[1];
    const bool same_input = (red.inputs.size() == 1 && arg.inputs.size() == 1 && red.inputs[0] == arg.inputs[0]);
    bool arg_axis_is_1 = false;
    const auto axis_it = arg.attrs.find("axis");
    if (axis_it != arg.attrs.end() && axis_it->is_number_integer()) {
      arg_axis_is_1 = (axis_it->get<int64_t>() == 1);
    }
    if (same_input && _reduce_axis_is_1(red.attrs) && arg_axis_is_1) {
      Intent tmp = intent;
      tmp.ops = {red};
      tmp.ops[0].attrs["return_indices"] = true;
      tmp.ops[0].attrs["index_output"] = arg.output;
      return emit_reduce_min_2d_axis1_with_indices_f32_i64(tmp, bindings_json);
    }
  }
  if (!intent.ops.empty() && intent.ops.back().op == "reduce_min" && _reduce_axis_is_1(intent.ops.back().attrs) &&
      _attrs_return_indices(intent.ops.back().attrs)) {
    return emit_reduce_min_2d_axis1_with_indices_f32_i64(intent, bindings_json);
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "argmax") return emit_arg_extrema_2d_axis1_i64(intent, bindings_json, true);
  if (intent.ops.size() == 1 && intent.ops[0].op == "argmin") return emit_arg_extrema_2d_axis1_i64(intent, bindings_json, false);
  if (intent.ops.size() == 3 && intent.ops[0].op == "const" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "div") {
    return emit_reduce_mean_2d_axis1_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 4 && intent.ops[0].op == "const" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "cast" &&
      intent.ops[3].op == "div") {
    const Op& cst = intent.ops[0];
    const Op& red = intent.ops[1];
    const Op& cast = intent.ops[2];
    const Op& div = intent.ops[3];
    const bool cast_wires_const = (cast.inputs.size() == 1 && cast.inputs[0] == cst.output);
    const bool div_wires_cast = (div.inputs.size() == 2 && div.inputs[0] == red.output && div.inputs[1] == cast.output);
    const bool cast_to_f32 = (cast.attrs.value("to", std::string("f32")) == "f32");
    if (cast_wires_const && div_wires_cast && cast_to_f32) {
      Intent tmp = intent;
      Op mean_div = div;
      mean_div.inputs = {red.output, cst.output};
      tmp.ops = {cst, red, mean_div};
      return emit_reduce_mean_2d_axis1_f32(tmp, bindings_json);
    }
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "gather") return emit_gather2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "conv1d") return emit_conv1d_ncl_f32(intent, bindings_json);
  if ((intent.ops.size() == 1 && intent.ops[0].op == "conv2d") ||
      (intent.ops.size() == 3 && intent.ops[0].op == "conv2d" && intent.ops[1].op == "broadcast_in_dim" && intent.ops[2].op == "add")) {
    return emit_conv2d_nchw_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "conv3d") return emit_conv3d_ncdhw_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "conv_depthwise2d") return emit_conv_depthwise2d_nchw_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "avg_pool2d") return emit_avg_pool2d_nchw_f32(intent, bindings_json);
  if ((intent.ops.size() == 1 && intent.ops[0].op == "max_pool2d_with_indices") ||
      (intent.ops.size() == 2 && intent.ops[0].op == "max_pool2d_with_indices" && intent.ops[1].op == "max_pool2d_with_indices")) {
    return emit_max_pool2d_with_indices_nchw_f32_i64(intent, bindings_json);
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "sort") return emit_sort2d_axis1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "quantile") return emit_quantile2d_dim1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "stack") return emit_stack_axis0_2x2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "unique") return emit_unique1d_i32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "polar") return emit_polar2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "kron") return emit_kron2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "glu") return emit_glu2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "index_add") return emit_index_add2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "index_put") return emit_index_put2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "scatter") return emit_scatter2d_dim1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "select_scatter") return emit_select_scatter2d_dim1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "slice_scatter") return emit_slice_scatter2d_dim1_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "masked_scatter") return emit_masked_scatter2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "masked_select") return emit_masked_select2d_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "nll_loss2d_forward") return emit_nll_loss2d_forward_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "nll_loss_forward") return emit_nll_loss_forward_f32(intent, bindings_json);
  if ((intent.ops.size() == 1 && intent.ops[0].op == "upsample_bicubic2d_aa") || intent.name == "upsample_bicubic2d_aa") {
    return emit_upsample_bicubic2d_aa_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "upsample_nearest1d") return emit_upsample_nearest1d_ncl_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "upsample_nearest2d") return emit_upsample_nearest2d_nchw_f32(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "cumsum") return emit_cumsum2d_axis1_f32(intent, bindings_json);
  if (intent.ops.size() == 2 && intent.ops[0].op == "mul" && intent.ops[1].op == "reduce_sum") return emit_dot1d_f32(intent, bindings_json);
  if (intent.ops.size() == 4 && (intent.ops[0].op == "cast" || intent.ops[0].op == "identity") &&
      (intent.ops[1].op == "cast" || intent.ops[1].op == "identity") && intent.ops[2].op == "mul" && intent.ops[3].op == "reduce_sum") {
    return emit_dot1d_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 4 && intent.ops[0].op == "iota" && intent.ops[1].op == "identity" && intent.ops[2].op == "cumsum" &&
      intent.ops[3].op == "identity") {
    const Op& id_in = intent.ops[1];
    const Op& cumsum = intent.ops[2];
    const Op& id_out = intent.ops[3];
    const bool cumsum_wires_id = (cumsum.inputs.size() == 1 && cumsum.inputs[0] == id_in.output);
    const bool out_wires_cumsum = (id_out.inputs.size() == 1 && id_out.inputs[0] == cumsum.output);
    if (cumsum_wires_id && out_wires_cumsum && id_in.inputs.size() == 1) {
      Intent tmp = intent;
      Op cumsum_op = cumsum;
      cumsum_op.inputs = {id_in.inputs[0]};
      cumsum_op.output = id_out.output;
      tmp.ops = {cumsum_op};
      return emit_cumsum2d_axis1_f32(tmp, bindings_json);
    }
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "cummax") return emit_cumextrema1d_f32(intent, bindings_json, true);
  if (intent.ops.size() == 1 && intent.ops[0].op == "cummin") return emit_cumextrema1d_f32(intent, bindings_json, false);
  if (intent.ops.size() == 1 && intent.ops[0].op == "scaled_dot_product_attention") {
    return emit_scaled_dot_product_attention_bhsd_f32(intent, bindings_json);
  }
  if (intent.ops.size() == 1 && intent.ops[0].op == "dropout") return emit_dropout(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "warp") return emit_warp(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "correlation") return emit_correlation(intent, bindings_json);
  if (intent.ops.size() == 1 && intent.ops[0].op == "resize") return emit_resize_bilinear2x_i8(intent, bindings_json);
  if (intent.ops.size() >= 1 && intent.ops.back().op == "rope") {
    bool ok = true;
    for (size_t i = 0; i + 1 < intent.ops.size(); ++i) {
      if (intent.ops[i].op != "const") {
        ok = false;
        break;
      }
    }
    if (ok) return emit_rope_f32(intent, bindings_json);
  }
