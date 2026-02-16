	        }
	        const int64_t N = logits_shape[0];
	        const int64_t C = logits_shape[1];
	        const int64_t H = logits_shape[2];
	        const int64_t W = logits_shape[3];
	        if (target_shape[0] != N || target_shape[1] != H || target_shape[2] != W) fail("nll_loss2d_forward target shape mismatch");
	        if (weight_shape[0] != C) fail("nll_loss2d_forward weight length mismatch");
	        const std::string target_dt = dtype_env.at(op.inputs[1]);
	        if (!(target_dt == "i64" || target_dt == "i32")) fail("nll_loss2d_forward target dtype must be i64/i32");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("nll_loss2d_forward currently supports f32 logits/weight/output");
	        }
	        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
	        const int ignore_index = op.attrs.contains("ignore_index") ? (int)resolve_const_value(op.attrs["ignore_index"], bindings) : -100;
	        if (reduction == 0) fail("nll_loss2d_forward reduction=none is not supported in RVV lowering yet");
	        w.line("double loss_sum = 0.0;");
	        w.line("double weight_sum = 0.0;");
	        w.line("for (int64_t n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int64_t h = 0; h < " + std::to_string(H) + "; ++h) {");
	        w.indent();
	        w.line("for (int64_t ww = 0; ww < " + std::to_string(W) + "; ++ww) {");
	        w.indent();
	        w.line("const int64_t t = (int64_t)" + v(op.inputs[1]) + "[((size_t)n * " + std::to_string(H) + " + (size_t)h) * " + std::to_string(W) + " + (size_t)ww];");
	        w.line("if (t == (int64_t)" + std::to_string(ignore_index) + ") continue;");
	        w.line("if (t < 0 || t >= " + std::to_string(C) + ") continue;");
	        w.line("const float wv = " + v(op.inputs[2]) + "[(size_t)t];");
	        w.line("const size_t idx = (((size_t)n * " + std::to_string(C) + " + (size_t)t) * " + std::to_string(H) + " + (size_t)h) * " + std::to_string(W) + " + (size_t)ww;");
	        w.line("const float lv = -" + v(op.inputs[0]) + "[idx] * wv;");
	        w.line("loss_sum += (double)lv;");
	        w.line("weight_sum += (double)wv;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        if (reduction == 1) {
	          w.line(out_var + "[0] = (weight_sum > 0.0) ? (float)(loss_sum / weight_sum) : 0.0f;");
	        } else {
	          w.line(out_var + "[0] = (float)loss_sum;");
	        }
	      } else if (op.op == "conv1d") {
	        if (op.inputs.size() != 3) fail("conv1d expects inputs [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const auto& b_shape = shape_env.at(op.inputs[2]);
	        if (x_shape.size() != 3 || w_shape.size() != 3 || b_shape.size() != 1 || out_shape.size() != 3)
	          fail("conv1d expects input[N,C,L], weight[CO,C_PER_G,K], bias[CO], out[N,CO,OL]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv1d currently supports f32 only");
	        const int64_t N = x_shape[0];
	        const int64_t C_IN = x_shape[1];
	        const int64_t L = x_shape[2];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t K = w_shape[2];
	        if (b_shape[0] != C_OUT) fail("conv1d bias shape mismatch");
	        const int64_t OL = out_shape[2];
	        const int stride = op.attrs.contains("stride") ? (int)resolve_const_value(op.attrs["stride"], bindings) : 1;
	        const int padding = op.attrs.contains("padding") ? (int)resolve_const_value(op.attrs["padding"], bindings) : 0;
	        const int dilation = op.attrs.contains("dilation") ? (int)resolve_const_value(op.attrs["dilation"], bindings) : 1;
	        const int groups = op.attrs.contains("groups") ? (int)resolve_const_value(op.attrs["groups"], bindings) : 1;
	        if (groups <= 0 || stride <= 0 || dilation <= 0) fail("conv1d attrs stride/dilation/groups must be positive");
	        if (C_IN != C_PER_G * groups) fail("conv1d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv1d C_OUT must be divisible by groups");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int ol = 0; ol < " + std::to_string(OL) + "; ++ol) {");
	        w.indent();
	        w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int k = 0; k < " + std::to_string(K) + "; ++k) {");
	        w.indent();
	        w.line("const int li = ol * " + std::to_string(stride) + " - " + std::to_string(padding) + " + k * " + std::to_string(dilation) + ";");
	        w.line("if (li >= 0 && li < " + std::to_string(L) + ") {");
	        w.indent();
	        w.line("const int x_idx = ((n * " + std::to_string(C_IN) + " + c) * " + std::to_string(L) + " + li);");
	        w.line("const int w_idx = ((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(K) + " + k);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OL) + " + ol)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv2d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv2d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4)
	          fail("conv2d expects input[N,C,H,W], weight[CO,C_PER_G,KH,KW], out[N,CO,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv2d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv2d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv2d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN_TOTAL = x_shape[1];
	        const int64_t H = x_shape[2];
	        const int64_t W = x_shape[3];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t KH = w_shape[2];
	        const int64_t KW = w_shape[3];
	        if (has_bias && b_shape[0] != C_OUT) fail("conv2d bias shape mismatch");
	        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
	          if (value.is_array()) {
	            if (value.size() != 2) fail("conv2d " + key + " must have length 2");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v};
	        };
	        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int, int>{1, 1};
	        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
	        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
	        const int groups = op.attrs.contains("groups") ? static_cast<int>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
	        if (groups <= 0 || stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
	          fail("conv2d attrs stride/dilation/groups must be positive");
	        if (C_IN_TOTAL != C_PER_G * groups) fail("conv2d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv2d C_OUT must be divisible by groups");
	        const int64_t OH = out_shape[2];
	        const int64_t OW = out_shape[3];
	        const int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
	        if (expected_oh != OH || expected_ow != OW) fail("conv2d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride.first) + " - " + std::to_string(padding.first) + " + kh * " + std::to_string(dilation.first) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride.second) + " - " + std::to_string(padding.second) + " + kw * " + std::to_string(dilation.second) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = (((n * " + std::to_string(C_IN_TOTAL) + " + c) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = (((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv3d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv3d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 5 || w_shape.size() != 5 || out_shape.size() != 5)
	          fail("conv3d expects input[N,C,D,H,W], weight[CO,C_PER_G,KD,KH,KW], out[N,CO,OD,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv3d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv3d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv3d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN_TOTAL = x_shape[1];
	        const int64_t D = x_shape[2];
	        const int64_t H = x_shape[3];
	        const int64_t W = x_shape[4];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t KD = w_shape[2];
	        const int64_t KH = w_shape[3];
	        const int64_t KW = w_shape[4];
	        if (has_bias && b_shape[0] != C_OUT) fail("conv3d bias shape mismatch");
	        auto parse_triple = [&](const json& value, const std::string& key) -> std::array<int, 3> {
	          if (value.is_array()) {
	            if (value.size() != 3) fail("conv3d " + key + " must have length 3");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)),
	                    static_cast<int>(resolve_const_value(value[1], bindings)),
	                    static_cast<int>(resolve_const_value(value[2], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v, v};
	        };
	        std::array<int, 3> stride = op.attrs.contains("stride") ? parse_triple(op.attrs["stride"], "stride") : std::array<int, 3>{1, 1, 1};
	        std::array<int, 3> padding = op.attrs.contains("padding") ? parse_triple(op.attrs["padding"], "padding") : std::array<int, 3>{0, 0, 0};
	        std::array<int, 3> dilation = op.attrs.contains("dilation") ? parse_triple(op.attrs["dilation"], "dilation") : std::array<int, 3>{1, 1, 1};
	        const int groups = op.attrs.contains("groups") ? static_cast<int>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
	        if (groups <= 0 || stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0 || dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0)
	          fail("conv3d attrs stride/dilation/groups must be positive");
	        if (C_IN_TOTAL != C_PER_G * groups) fail("conv3d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv3d C_OUT must be divisible by groups");
	        const int64_t OD = out_shape[2];
	        const int64_t OH = out_shape[3];
	        const int64_t OW = out_shape[4];
	        const int64_t expected_od = ((D + 2LL * padding[0] - (int64_t)dilation[0] * (KD - 1) - 1) / stride[0]) + 1;
	        const int64_t expected_oh = ((H + 2LL * padding[1] - (int64_t)dilation[1] * (KH - 1) - 1) / stride[1]) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding[2] - (int64_t)dilation[2] * (KW - 1) - 1) / stride[2]) + 1;
	        if (expected_od != OD || expected_oh != OH || expected_ow != OW) fail("conv3d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int od = 0; od < " + std::to_string(OD) + "; ++od) {");
	        w.indent();
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int kd = 0; kd < " + std::to_string(KD) + "; ++kd) {");
	        w.indent();
	        w.line("const int id = od * " + std::to_string(stride[0]) + " - " + std::to_string(padding[0]) + " + kd * " + std::to_string(dilation[0]) + ";");
	        w.line("if ((unsigned)id >= (unsigned)" + std::to_string(D) + ") continue;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride[1]) + " - " + std::to_string(padding[1]) + " + kh * " + std::to_string(dilation[1]) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride[2]) + " - " + std::to_string(padding[2]) + " + kw * " + std::to_string(dilation[2]) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = ((((n * " + std::to_string(C_IN_TOTAL) + " + c) * " + std::to_string(D) + " + id) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = ((((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(KD) + " + kd) * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[((((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OD) + " + od) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv_depthwise2d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv_depthwise2d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4)
	          fail("conv_depthwise2d expects input[N,C,H,W], weight[C_OUT,1,KH,KW], out[N,C_OUT,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv_depthwise2d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv_depthwise2d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv_depthwise2d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN = x_shape[1];
	        const int64_t H = x_shape[2];
	        const int64_t W = x_shape[3];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t KH = w_shape[2];
	        const int64_t KW = w_shape[3];
	        if (w_shape[1] != 1) fail("conv_depthwise2d expects weight second dim == 1");
	        if (C_IN <= 0 || C_OUT <= 0 || (C_OUT % C_IN) != 0) fail("conv_depthwise2d channel multiplier mismatch");
	        if (has_bias && b_shape[0] != C_OUT) fail("conv_depthwise2d bias shape mismatch");
	        const int64_t channel_multiplier = C_OUT / C_IN;
	        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
	          if (value.is_array()) {
	            if (value.size() != 2) fail("conv_depthwise2d " + key + " must have length 2");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v};
	        };
	        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int, int>{1, 1};
	        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
	        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
	        if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
	          fail("conv_depthwise2d attrs stride/dilation must be positive");
	        const int64_t OH = out_shape[2];
	        const int64_t OW = out_shape[3];
	        const int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
	        if (expected_oh != OH || expected_ow != OW) fail("conv_depthwise2d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int ci = co / " + std::to_string(channel_multiplier) + ";");
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride.first) + " - " + std::to_string(padding.first) + " + kh * " + std::to_string(dilation.first) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride.second) + " - " + std::to_string(padding.second) + " + kw * " + std::to_string(dilation.second) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = (((n * " + std::to_string(C_IN) + " + ci) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = ((co * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "dropout") {
	        if (op.inputs.size() != 3) fail("dropout requires 3 inputs (X, p, seed)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("dropout supports only f32 tensors");
	        if (!shape_env.at(op.inputs[1]).empty()) fail("dropout p must be a scalar tensor (rank-0)");
	        if (!shape_env.at(op.inputs[2]).empty()) fail("dropout seed must be a scalar tensor (rank-0)");
	        const std::string seed_dt = dtype_env.at(op.inputs[2]);
	        std::string seed_expr;
	        if (seed_dt == "i32") seed_expr = "(uint64_t)(uint32_t)" + v(op.inputs[2]) + "[0]";
	        else if (seed_dt == "i64") seed_expr = "(uint64_t)" + v(op.inputs[2]) + "[0]";
	        else fail("dropout seed dtype must be i32 or i64");
	        int rounds = 10;
	        if (op.attrs.contains("n_rounds")) rounds = (int)resolve_const_value(op.attrs["n_rounds"], bindings);
	        int64_t n = numel(out_shape);
	        w.line("intentir_dropout_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ", " + v(op.inputs[1]) +
	               "[0], " + seed_expr + ", " + std::to_string(rounds) + ");");
	      } else if (op.op == "correlation") {
	        if (op.inputs.size() != 3) fail("correlation requires 3 inputs (src0, src1, out_shift)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(op.inputs[1]) != "i8" || dtype_env.at(out) != "i8") fail("correlation supports only i8 tensors");
	        const auto& s0 = shape_env.at(op.inputs[0]);
	        const auto& s1 = shape_env.at(op.inputs[1]);
	        if (s0.size() != 3 || s1.size() != 3 || out_shape.size() != 3) fail("correlation expects rank-3 tensors");
	        if (s0 != s1) fail("correlation expects src0/src1 shapes to match");
	        if (s0[1] != out_shape[1] || s0[2] != out_shape[2]) fail("correlation spatial shape mismatch");
	        if (!shape_env.at(op.inputs[2]).empty()) fail("correlation out_shift must be a scalar tensor (rank-0)");
	        const std::string sh_dt = dtype_env.at(op.inputs[2]);
	        if (!(sh_dt == "i32" || sh_dt == "i64")) fail("correlation out_shift dtype must be i32 or i64");
	        const int64_t OC = out_shape[0];
	        const int64_t IC = s0[0];
	        const int64_t H = out_shape[1];
	        const int64_t W = out_shape[2];
	        w.line("intentir_correlation_i8(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + out_var + ", " + std::to_string(OC) +
	               ", " + std::to_string(IC) + ", " + std::to_string(H) + ", " + std::to_string(W) + ", (int32_t)" + v(op.inputs[2]) + "[0]);");
	      } else if (op.op == "resize") {
	        if (op.inputs.size() != 1) fail("resize requires 1 input (src)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(out) != "i8") fail("resize supports only i8 tensors");
	        const auto& s = shape_env.at(op.inputs[0]);
	        if (s.size() != 3 || out_shape.size() != 3) fail("resize expects rank-3 tensors [C,H,W]");
	        if (out_shape[0] != s[0]) fail("resize channel mismatch");
	        if (out_shape[1] != 2 * s[1] || out_shape[2] != 2 * s[2]) fail("resize currently supports only 2x upsample");
	        int hw_fl = 7;
	        if (op.attrs.contains("hw_fl")) hw_fl = (int)resolve_const_value(op.attrs["hw_fl"], bindings);
	        w.line("intentir_resize_bilinear2x_i8(" + v(op.inputs[0]) + ", " + out_var + ", " + std::to_string(s[0]) + ", " + std::to_string(s[1]) + ", " +
	               std::to_string(s[2]) + ", " + std::to_string(hw_fl) + ");");
	      } else if (op.op == "upsample_nearest1d") {
	        if (op.inputs.size() != 1) fail("upsample_nearest1d requires 1 input");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("upsample_nearest1d supports only f32 tensors");
	        const auto& s = shape_env.at(op.inputs[0]);
	        if (s.size() != 3 || out_shape.size() != 3) fail("upsample_nearest1d expects input/output rank-3 [N,C,L]");
	        if (s[0] != out_shape[0] || s[1] != out_shape[1]) fail("upsample_nearest1d N/C mismatch");
	        const int64_t N = s[0], C = s[1], IL = s[2], OL = out_shape[2];
	        if (IL <= 0 || OL <= 0) fail("upsample_nearest1d expects positive IL/OL");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int c = 0; c < " + std::to_string(C) + "; ++c) {");
	        w.indent();
	        w.line("for (int ol = 0; ol < " + std::to_string(OL) + "; ++ol) {");
	        w.indent();
	        w.line("int il = (int)(((int64_t)ol * " + std::to_string(IL) + ") / " + std::to_string(OL) + ");");
	        w.line("if (il >= " + std::to_string(IL) + ") il = " + std::to_string(IL - 1) + ";");
	        w.line(out_var + "[((n * " + std::to_string(C) + " + c) * " + std::to_string(OL) + " + ol)] = " + v(op.inputs[0]) +
	               "[((n * " + std::to_string(C) + " + c) * " + std::to_string(IL) + " + il)];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "upsample_nearest2d") {
	        if (op.inputs.size() != 1) fail("upsample_nearest2d requires 1 input");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("upsample_nearest2d supports only f32 tensors");
	        const auto& s = shape_env.at(op.inputs[0]);
	        if (s.size() != 4 || out_shape.size() != 4) fail("upsample_nearest2d expects input/output rank-4 [N,C,H,W]");
	        if (s[0] != out_shape[0] || s[1] != out_shape[1]) fail("upsample_nearest2d N/C mismatch");
	        const int64_t N = s[0], C = s[1], IH = s[2], IW = s[3], OH = out_shape[2], OW = out_shape[3];
