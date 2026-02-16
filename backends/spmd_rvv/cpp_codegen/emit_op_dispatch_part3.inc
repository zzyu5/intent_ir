	        if (IH <= 0 || IW <= 0 || OH <= 0 || OW <= 0) fail("upsample_nearest2d expects positive H/W");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int c = 0; c < " + std::to_string(C) + "; ++c) {");
	        w.indent();
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("int ih = (int)(((int64_t)oh * " + std::to_string(IH) + ") / " + std::to_string(OH) + ");");
	        w.line("if (ih >= " + std::to_string(IH) + ") ih = " + std::to_string(IH - 1) + ";");
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        w.line("int iw = (int)(((int64_t)ow * " + std::to_string(IW) + ") / " + std::to_string(OW) + ");");
	        w.line("if (iw >= " + std::to_string(IW) + ") iw = " + std::to_string(IW - 1) + ";");
	        w.line(out_var + "[(((n * " + std::to_string(C) + " + c) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = " +
	               v(op.inputs[0]) + "[(((n * " + std::to_string(C) + " + c) * " + std::to_string(IH) + " + ih) * " + std::to_string(IW) + " + iw)];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "warp") {
	        if (op.inputs.size() != 2) fail("warp requires 2 inputs (src, offset)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(op.inputs[1]) != "i16" || dtype_env.at(out) != "i8") fail("warp expects src/out i8 and offset i16");
	        const auto& s = shape_env.at(op.inputs[0]);
	        const auto& off = shape_env.at(op.inputs[1]);
	        if (s.size() != 3 || off.size() != 2 || out_shape.size() != 3) fail("warp expects src/out rank-3 and offset rank-2");
	        if (out_shape != s) fail("warp output shape must match src shape");
	        if (off[0] != s[1] || off[1] != s[2]) fail("warp offset shape must match [H,W]");
	        w.line("intentir_warp_q8_8_i8_i16(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + out_var + ", " + std::to_string(s[0]) + ", " +
	               std::to_string(s[1]) + ", " + std::to_string(s[2]) + ");");
	      } else if (op.op == "rope") {
	        if (op.inputs.size() != 3) fail("rope requires 3 inputs (input, cos, sin)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32")
	          fail("rope supports only f32 tensors");
	        const auto& x = shape_env.at(op.inputs[0]);
	        const auto& c = shape_env.at(op.inputs[1]);
	        const auto& s = shape_env.at(op.inputs[2]);
	        if (x.size() != 4 || out_shape.size() != 4) fail("rope expects rank-4 input/output");
	        if (c.size() != 2 || s.size() != 2) fail("rope expects rank-2 cos/sin");
	        if (out_shape != x) fail("rope output shape must match input shape");
	        const int64_t SEQ = x[0], B = x[1], H = x[2], D = x[3];
	        if ((D & 1) != 0) fail("rope expects even HEAD_DIM");
	        const int64_t half = D / 2;
	        if (c[0] != SEQ || c[1] != half) fail("rope cos shape must be [SEQ_LEN, HEAD_DIM/2]");
	        if (s[0] != SEQ || s[1] != half) fail("rope sin shape must be [SEQ_LEN, HEAD_DIM/2]");
	        w.line("intentir_rope_f32(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + v(op.inputs[2]) + ", " + out_var + ", " + std::to_string(SEQ) +
	               ", " + std::to_string(B) + ", " + std::to_string(H) + ", " + std::to_string(D) + ");");
	      } else if (op.op == "iota") {
	        int axis = op.attrs.value("axis", 0);
	        emit_iota(w, out_var, out_shape, axis, dtype_env.at(out));
	      } else if (op.op == "gather") {
	        std::vector<std::string> idxs;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idxs.push_back(v(op.inputs[i]));
	        std::vector<std::vector<int64_t>> idx_shapes;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idx_shapes.push_back(shape_env.at(op.inputs[i]));
	        std::vector<std::string> idx_dtypes;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idx_dtypes.push_back(dtype_env.at(op.inputs[i]));
	        emit_gather(w, out_var, v(op.inputs[0]), idxs, shape_env.at(op.inputs[0]), idx_shapes, out_shape, dtype_env.at(op.inputs[0]), idx_dtypes, dtype_env.at(out));
	      } else if (op.op == "index_add") {
	        if (op.inputs.size() != 3) fail("index_add expects inputs [base, index, src]");
	        int axis = op.attrs.value("axis", 0);
	        double alpha = 1.0;
	        if (op.attrs.contains("alpha")) alpha = resolve_const_value(op.attrs["alpha"], bindings);
	        emit_index_add_2d(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            out_shape,
	            dtype_env.at(op.inputs[1]),
	            axis,
	            alpha);
	      } else if (op.op == "index_put") {
	        if (op.inputs.size() != 4) fail("index_put expects inputs [base, row_idx, col_idx, values]");
	        bool accumulate = op.attrs.value("accumulate", false);
	        emit_index_put_2d(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            v(op.inputs[3]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            shape_env.at(op.inputs[3]),
	            out_shape,
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(op.inputs[2]),
	            accumulate);
	      } else if (op.op == "masked_select") {
	        if (op.inputs.size() != 2) fail("masked_select expects inputs [inp, mask]");
	        emit_masked_select_flat(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            out_shape,
	            dtype_env.at(op.inputs[0]),
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(out));
	      } else if (op.op == "masked_scatter") {
	        if (op.inputs.size() != 3) fail("masked_scatter expects inputs [inp, mask, source]");
	        emit_masked_scatter_flat(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            out_shape,
	            dtype_env.at(op.inputs[0]),
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(op.inputs[2]),
	            dtype_env.at(out));
	      } else if (op.op == "kron") {
	        if (op.inputs.size() != 2) fail("kron expects inputs [A, B]");
	        emit_kron_2d(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape);
	      } else if (op.op == "reduce_sum") {
        // Accept both "axes" and "dims" (parser/LLM may emit either). Prefer "axes"
        // when both are present to match interpreter semantics.
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
        } else {
          fail("reduce_sum missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        std::optional<double> scale;
	        if (op.attrs.contains("scale")) scale = resolve_const_value(op.attrs["scale"], bindings);
	        emit_reduce_sum(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims, scale, dtype_env.at(out));
	      } else if (op.op == "reduce_prod") {
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
        } else {
          fail("reduce_prod missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_prod(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_max") {
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
        } else {
          fail("reduce_max missing dims/axes");
	        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_max(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_min") {
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
        } else {
          fail("reduce_min missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_min(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_any") {
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
        } else {
          fail("reduce_any missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_any(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "quantile") {
	        if (op.inputs.size() != 2) fail("quantile expects inputs [inp, q]");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        const auto& q_shape = shape_env.at(op.inputs[1]);
	        if (in_shape.size() != 2) fail("quantile currently supports rank-2 input");
	        if (!q_shape.empty()) fail("quantile q must be a scalar tensor (rank-0)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("quantile currently supports only f32 tensors");
	        }
	        int axis = op.attrs.value("dim", op.attrs.value("axis", 1));
	        if (axis < 0) axis += 2;
	        if (axis != 1) fail("quantile currently supports dim=1 only");
	        bool keepdim = op.attrs.value("keepdim", false);
	        std::string interpolation = op.attrs.value("interpolation", std::string("linear"));
	        if (interpolation != "linear") fail("quantile currently supports interpolation=linear only");
	        const int64_t M = in_shape[0];
	        const int64_t N = in_shape[1];
	        if (N <= 0) fail("quantile expects N > 0");
	        if (out_shape.size() == 1) {
	          if (out_shape[0] != M) fail("quantile output shape mismatch: expect [M]");
	        } else if (out_shape.size() == 2) {
	          if (!keepdim) fail("quantile rank-2 output requires keepdim=true");
	          if (out_shape[0] != M || out_shape[1] != 1) fail("quantile output shape mismatch: expect [M,1]");
	        } else {
	          fail("quantile currently supports output rank 1 or 2");
	        }
	        w.line("float __q = " + v(op.inputs[1]) + "[0];");
	        w.line("if (__q < 0.0f) __q = 0.0f;");
	        w.line("if (__q > 1.0f) __q = 1.0f;");
	        w.line("float* __quantile_row = (float*)malloc(sizeof(float) * (size_t)" + std::to_string(N) + ");");
	        w.line("if (!__quantile_row) { fprintf(stderr, \"alloc failed: quantile_row\\n\"); return; }");
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("const float* __row = " + v(op.inputs[0]) + " + (size_t)i * (size_t)" + std::to_string(N) + ";");
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) __quantile_row[(size_t)j] = __row[(size_t)j];");
	        w.line("for (int64_t __ii = 1; __ii < " + std::to_string(N) + "; ++__ii) {");
	        w.indent();
	        w.line("float __v = __quantile_row[(size_t)__ii];");
	        w.line("int64_t __jj = __ii - 1;");
	        w.line("while (__jj >= 0 && __quantile_row[(size_t)__jj] > __v) {");
	        w.indent();
	        w.line("__quantile_row[(size_t)(__jj + 1)] = __quantile_row[(size_t)__jj];");
	        w.line("--__jj;");
	        w.dedent();
	        w.line("}");
	        w.line("__quantile_row[(size_t)(__jj + 1)] = __v;");
	        w.dedent();
	        w.line("}");
	        if (N == 1) {
	          w.line("const float __qv = __quantile_row[(size_t)0];");
	        } else {
	          w.line("const float __pos = __q * (float)" + std::to_string(N - 1) + ";");
	          w.line("int64_t __lo = (int64_t)__pos;");
	          w.line("if (__lo < 0) __lo = 0;");
	          w.line("if (__lo > " + std::to_string(N - 1) + ") __lo = " + std::to_string(N - 1) + ";");
	          w.line("int64_t __hi = (__lo + 1 <= " + std::to_string(N - 1) + ") ? (__lo + 1) : __lo;");
	          w.line("const float __t = __pos - (float)__lo;");
	          w.line("const float __qv = __quantile_row[(size_t)__lo] + (__quantile_row[(size_t)__hi] - __quantile_row[(size_t)__lo]) * __t;");
	        }
	        if (out_shape.size() == 1) {
	          w.line(out_var + "[(size_t)i] = __qv;");
	        } else {
	          w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(out_shape[1]) + "] = __qv;");
	        }
	        w.dedent();
	        w.line("}");
	        w.line("free(__quantile_row);");
	      } else if (op.op == "argmax" || op.op == "argmin") {
        int axis = op.attrs.value("axis", -1);
        emit_arg_reduce_axis1(w, op.op, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis);
	      } else if (op.op == "avg_pool2d") {
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
          if (value.is_array()) {
            if (value.size() != 2) fail("avg_pool2d " + key + " must have length 2");
            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
          }
          int v = static_cast<int>(resolve_const_value(value, bindings));
          return {v, v};
        };
        std::pair<int, int> ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int, int>{2, 2};
        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
        bool count_include_pad = op.attrs.value("count_include_pad", true);
        emit_avg_pool2d(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, ksz.first, ksz.second, stride.first, stride.second,
                        padding.first, padding.second, count_include_pad);
	      } else if (op.op == "max_pool2d_with_indices") {
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
          if (value.is_array()) {
            if (value.size() != 2) fail("max_pool2d_with_indices " + key + " must have length 2");
            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
          }
          int v = static_cast<int>(resolve_const_value(value, bindings));
          return {v, v};
        };
        std::pair<int, int> ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int, int>{2, 2};
        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
        bool ceil_mode = op.attrs.value("ceil_mode", false);
        if (ceil_mode) fail("max_pool2d_with_indices ceil_mode=true is not supported");
        if (dilation.first != 1 || dilation.second != 1) fail("max_pool2d_with_indices dilation != 1 is not supported");
        std::string select = op.attrs.value("select", "values");
        bool emit_indices = (select == "indices") || (dtype_env.at(out) == "i64");
        emit_max_pool2d_with_indices(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, ksz.first, ksz.second, stride.first, stride.second,
                                     padding.first, padding.second, emit_indices);
	      } else if (op.op == "mse_loss") {
        if (op.inputs.size() != 2) fail("mse_loss expects [input, target]");
        const auto& in_shape = shape_env.at(op.inputs[0]);
        const auto& tgt_shape = shape_env.at(op.inputs[1]);
        if (in_shape != tgt_shape) fail("mse_loss shape mismatch between input and target");
        std::string reduction;
        if (op.attrs.contains("reduction")) {
          if (op.attrs["reduction"].is_number_integer()) {
            int mode = op.attrs["reduction"].get<int>();
            if (mode == 0) reduction = "none";
            else if (mode == 2) reduction = "sum";
            else reduction = "mean";
          } else {
            reduction = op.attrs["reduction"].get<std::string>();
          }
        } else {
          reduction = "mean";
        }
        emit_mse_loss(w, out_var, v(op.inputs[0]), v(op.inputs[1]), in_shape, out_shape, reduction);
	      } else if (op.op == "scaled_dot_product_attention") {
        if (op.inputs.size() < 3) fail("scaled_dot_product_attention requires query/key/value");
        bool is_causal = op.attrs.value("is_causal", false);
        emit_scaled_dot_product_attention(w, out_var, v(op.inputs[0]), v(op.inputs[1]), v(op.inputs[2]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]),
                                          shape_env.at(op.inputs[2]), out_shape, is_causal);
	      } else if (op.op == "glu") {
          int axis = op.attrs.value("axis", -1);
          emit_glu(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis);
	      } else if (op.op == "exp") {
	        int64_t n = numel(out_shape);
          std::optional<double> base;
          if (op.attrs.contains("base")) {
            base = resolve_const_value(op.attrs["base"], bindings);
          }
          if (base.has_value()) {
            const double b = *base;
            if (std::fabs(b - 2.0) < 1e-6) {
              w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out_var + "[i] = exp2f(" + v(op.inputs[0]) + "[i]);");
            } else {
              w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out_var + "[i] = powf((float)" + std::to_string(b) + ", " +
                     v(op.inputs[0]) + "[i]);");
            }
          } else {
            w.line("intentir_exp_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ");");
          }
	      } else if (op.op == "log") {
	        emit_log(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "relu") {
	        int64_t n = numel(out_shape);
	        w.line("intentir_relu_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ");");
	      } else if (op.op == "softmax") {
	        int axis = op.attrs.value("axis", -1);
	        emit_softmax(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), axis);
	      } else if (op.op == "std") {
	        int axis = op.attrs.value("axis", 1);
	        bool keepdims = op.attrs.value("keepdims", false);
	        int correction = op.attrs.value("correction", op.attrs.value("ddof", 0));
	        emit_std_axis1(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis, keepdims, correction);
	      } else if (op.op == "sort") {
	        int axis = op.attrs.value("axis", -1);
	        bool descending = op.attrs.value("descending", false);
	        emit_sort_axis1(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis, descending);
	      } else if (op.op == "unique") {
	        if (op.inputs.size() != 1) fail("unique expects one input");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        if (in_shape.size() != 1) fail("unique lowering currently supports rank-1 input");
	        if (out_shape.size() != 1) fail("unique lowering currently supports rank-1 output");
	        const std::string in_dt = dtype_env.at(op.inputs[0]);
	        const std::string out_dt = dtype_env.at(out);
	        if (in_dt != out_dt) fail("unique dtype mismatch between input and output");
	        if (in_dt != "i32") fail("unique lowering currently supports i32 tensors");
	        const int64_t N = in_shape[0];
	        const int64_t U = out_shape[0];
	        w.line("int64_t __unique_count = 0;");
	        w.line("for (int64_t i = 0; i < " + std::to_string(N) + "; ++i) {");
	        w.indent();
	        w.line("const int32_t __v = " + v(op.inputs[0]) + "[(size_t)i];");
	        w.line("int __seen = 0;");
	        w.line("for (int64_t j = 0; j < __unique_count; ++j) {");
	        w.indent();
	        w.line("if (" + out_var + "[(size_t)j] == __v) { __seen = 1; break; }");
	        w.dedent();
	        w.line("}");
	        w.line("if (__seen == 0) {");
	        w.indent();
	        w.line("if (__unique_count < " + std::to_string(U) + ") " + out_var + "[(size_t)__unique_count] = __v;");
	        w.line("++__unique_count;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line("for (int64_t i = __unique_count; i < " + std::to_string(U) + "; ++i) " + out_var + "[(size_t)i] = 0;");
	        w.line("for (int64_t i = 1; i < " + std::to_string(U) + "; ++i) {");
	        w.indent();
	        w.line("int32_t __v = " + out_var + "[(size_t)i];");
	        w.line("int64_t __j = i - 1;");
	        w.line("while (__j >= 0 && " + out_var + "[(size_t)__j] > __v) {");
	        w.indent();
	        w.line(out_var + "[(size_t)(__j + 1)] = " + out_var + "[(size_t)__j];");
	        w.line("--__j;");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(size_t)(__j + 1)] = __v;");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "stack") {
	        if (op.inputs.size() != 2) fail("stack currently supports exactly 2 inputs");
	        int axis = op.attrs.value("axis", 0);
	        if (axis != 0) fail("stack lowering currently supports axis=0 only");
	        emit_stack_axis0_2x2d(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape);
	      } else if (op.op == "matmul") {
	        bool ta = op.attrs.value("transpose_a", false);
	        bool tb = op.attrs.value("transpose_b", false);
	        emit_matmul(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, ta, tb,
	                    sched_tile_m, sched_tile_n, sched_tile_k);
	      } else {
	        fail("unsupported op lowering: " + op.op);
	      }
