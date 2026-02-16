// Extracted from intentir_codegen.cpp (wave3 modular split)
  void emit_compute_fn() {
    w.line("static void intentir_compute(void) {");
    w.indent();
    std::vector<size_t> emit_ops;
    emit_ops.reserve(intent.ops.size());
    for (size_t i = 0; i < intent.ops.size(); ++i) {
      const auto& op = intent.ops[i];
      if (op.op == "const") continue;
      if (op.op == "reshape" || op.op == "identity" || op.op == "layout_cast") continue;
      emit_ops.push_back(i);
    }

    // Backend fusion: detect canonical LayerNorm forward sequences and emit a fused runtime call.
    // Keep profiling indices stable by emitting empty slots for fused-away ops.
    enum FuseTag : uint8_t { FUSE_NONE = 0, FUSE_LN_HEAD = 1, FUSE_LN_TAIL = 2 };
    std::vector<uint8_t> fuse_tags(emit_ops.size(), (uint8_t)FUSE_NONE);
    const size_t LN_LEN = 15;  // number of non-alias, non-const ops in the canonical pattern

    auto get_reduce_dims = [&](const Op& op) -> std::vector<int> {
      std::vector<int> dims;
      if (op.attrs.contains("axes")) {
        for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
      } else if (op.attrs.contains("dims")) {
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
      } else if (op.attrs.contains("axis")) {
        if (op.attrs["axis"].is_array()) {
          for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
        } else {
          dims.push_back(op.attrs["axis"].get<int>());
        }
      }
      return dims;
    };

    for (size_t pos = 0; pos + LN_LEN <= emit_ops.size(); ++pos) {
      if (fuse_tags[pos] != (uint8_t)FUSE_NONE) continue;
      const Op& o0 = intent.ops[emit_ops[pos + 0]];
      const Op& o1 = intent.ops[emit_ops[pos + 1]];
      const Op& o2 = intent.ops[emit_ops[pos + 2]];
      const Op& o3 = intent.ops[emit_ops[pos + 3]];
      const Op& o4 = intent.ops[emit_ops[pos + 4]];
      const Op& o5 = intent.ops[emit_ops[pos + 5]];
      const Op& o6 = intent.ops[emit_ops[pos + 6]];
      const Op& o7 = intent.ops[emit_ops[pos + 7]];
      const Op& o8 = intent.ops[emit_ops[pos + 8]];
      const Op& o9 = intent.ops[emit_ops[pos + 9]];
      const Op& o10 = intent.ops[emit_ops[pos + 10]];
      const Op& o11 = intent.ops[emit_ops[pos + 11]];
      const Op& o12 = intent.ops[emit_ops[pos + 12]];
      const Op& o13 = intent.ops[emit_ops[pos + 13]];
      const Op& o14 = intent.ops[emit_ops[pos + 14]];

      if (!(o0.op == "reduce_sum" && o1.op == "div" && o2.op == "broadcast_in_dim" && o3.op == "sub" && o4.op == "mul" &&
            o5.op == "reduce_sum" && o6.op == "div" && o7.op == "add" && o8.op == "rsqrt" && o9.op == "broadcast_in_dim" &&
            o10.op == "broadcast_in_dim" && o11.op == "broadcast_in_dim" && o12.op == "mul" && o13.op == "mul" && o14.op == "add")) {
        continue;
      }

      if (o0.inputs.size() != 1 || o0.inputs[0].empty()) continue;
      const std::string& X = o0.inputs[0];
      if (!shape_env.count(X) || !dtype_env.count(X)) continue;
      const auto& x_shape = shape_env.at(X);
      if (x_shape.size() != 2) continue;
      if (dtype_env.at(X) != "f32") continue;

      auto d0 = get_reduce_dims(o0);
      auto d5 = get_reduce_dims(o5);
      if (!(d0.size() == 1 && d0[0] == 1 && d5.size() == 1 && d5[0] == 1)) continue;

      const std::string& var_name = o6.output;
      if (o7.inputs.size() != 2) continue;
      std::string eps_name;
      if (o7.inputs[0] == var_name) eps_name = o7.inputs[1];
      else if (o7.inputs[1] == var_name) eps_name = o7.inputs[0];
      else continue;
      if (!dtype_env.count(eps_name) || dtype_env.at(eps_name) != "f32") continue;

      if (o10.inputs.size() != 1 || o11.inputs.size() != 1) continue;
      const std::string& W = o10.inputs[0];
      const std::string& B = o11.inputs[0];
      if (!shape_env.count(W) || !shape_env.count(B)) continue;
      if (shape_env.at(W).size() != 1 || shape_env.at(B).size() != 1) continue;
      if (shape_env.at(W)[0] != x_shape[1] || shape_env.at(B)[0] != x_shape[1]) continue;
      if (!dtype_env.count(W) || !dtype_env.count(B)) continue;
      if (dtype_env.at(W) != "f32" || dtype_env.at(B) != "f32") continue;

      const std::string& mean_name = o1.output;
      const std::string& rstd_name = o8.output;
      const std::string& y_name = o14.output;
      if (!shape_env.count(mean_name) || !shape_env.count(rstd_name) || !shape_env.count(y_name)) continue;
      if (shape_env.at(mean_name).size() != 1 || shape_env.at(rstd_name).size() != 1) continue;
      if (shape_env.at(mean_name)[0] != x_shape[0] || shape_env.at(rstd_name)[0] != x_shape[0]) continue;
      if (shape_env.at(y_name) != x_shape) continue;

      fuse_tags[pos] = (uint8_t)FUSE_LN_HEAD;
      for (size_t j = 1; j < LN_LEN; ++j) fuse_tags[pos + j] = (uint8_t)FUSE_LN_TAIL;
      pos += LN_LEN - 1;
    }

    size_t prof_i = 0;
    for (size_t pos = 0; pos < emit_ops.size(); ++pos) {
      const Op& op = intent.ops[emit_ops[pos]];
      const std::string& out = op.output;
	      const std::string& out_var = v(out);
	      const std::vector<int64_t>& out_shape = shape_env.at(out);
      w.line("{");
      w.indent();
      w.line("uint64_t __intentir_t0 = 0;");
      w.line("if (intentir_profile_enabled) __intentir_t0 = intentir_now_ns();");
      w.line("// op " + op.op + " -> " + out);

      if (fuse_tags[pos] == (uint8_t)FUSE_LN_HEAD) {
        const Op& o0 = intent.ops[emit_ops[pos + 0]];
        const Op& o1 = intent.ops[emit_ops[pos + 1]];
        const Op& o6 = intent.ops[emit_ops[pos + 6]];
        const Op& o7 = intent.ops[emit_ops[pos + 7]];
        const Op& o8 = intent.ops[emit_ops[pos + 8]];
        const Op& o10 = intent.ops[emit_ops[pos + 10]];
        const Op& o11 = intent.ops[emit_ops[pos + 11]];
        const Op& o14 = intent.ops[emit_ops[pos + 14]];

        const std::string& X = o0.inputs[0];
        const std::string& W = o10.inputs[0];
        const std::string& B = o11.inputs[0];
        const std::string& mean_name = o1.output;
        const std::string& rstd_name = o8.output;
        const std::string& y_name = o14.output;
        const auto& x_shape = shape_env.at(X);
        const int64_t M = x_shape[0];
        const int64_t N = x_shape[1];

        const std::string& var_name = o6.output;
        std::string eps_name;
        if (o7.inputs[0] == var_name) eps_name = o7.inputs[1];
        else eps_name = o7.inputs[0];

        w.line("// fused: layernorm forward (2D f32)");
        w.line("intentir_layernorm_2d_f32(" + v(X) + ", " + v(y_name) + ", " + v(W) + ", " + v(B) + ", " + v(mean_name) + ", " + v(rstd_name) +
               ", " + std::to_string(M) + ", " + std::to_string(N) + ", " + v(eps_name) + "[0]);");
      } else if (fuse_tags[pos] == (uint8_t)FUSE_LN_TAIL) {
        w.line("// fused into previous layernorm");
      #include "emit_op_dispatch.inc"
      w.line("if (intentir_profile_enabled) intentir_profile_ns[" + std::to_string(prof_i) +
             "] += intentir_now_ns() - __intentir_t0;");
      w.dedent();
      w.line("}");
      w.blank();
      prof_i++;
    }
    w.dedent();
    w.line("}");
    w.blank();
  }
