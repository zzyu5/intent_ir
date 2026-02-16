      const std::string& kind = op.op;
      const std::string& out = op.output;
      if (kind == "const") {
        shape_env[out] = {};
        dtype_env[out] = op.attrs.value("dtype", dtype_env.count(out) ? dtype_env[out] : std::string("f32"));
        continue;
      }
      if (shape_env.find(out) != shape_env.end()) {
        if (kind == "eq" || kind == "ne" || kind == "lt" || kind == "le" || kind == "gt" || kind == "ge" || kind == "reduce_any") {
          dtype_env[out] = "bool";
        }
        continue;
      }

      auto get_shape = [&](const std::string& n) -> const std::vector<int64_t>& {
        auto it = shape_env.find(n);
        if (it == shape_env.end()) fail("missing shape for " + n);
        return it->second;
      };
      auto get_dtype = [&](const std::string& n) -> std::string {
        auto it = dtype_env.find(n);
        return it == dtype_env.end() ? std::string("f32") : it->second;
      };

      if (kind == "reshape") {
        std::vector<int64_t> shp;
        if (!op.attrs.contains("shape") || !op.attrs["shape"].is_array()) fail("reshape requires attrs.shape");
        for (const auto& d : op.attrs["shape"]) {
          if (d.is_number_integer()) shp.push_back(d.get<int64_t>());
          else if (d.is_string()) {
            std::string s = d.get<std::string>();
            auto it = bindings.find(s);
            if (it == bindings.end()) fail("unbound shape symbol in reshape.shape: " + s);
            shp.push_back(it->second);
          } else {
            fail("invalid reshape dim type");
          }
        }
        shape_env[out] = shp;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "identity" || kind == "layout_cast") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "transpose") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (!op.attrs.contains("perm") || !op.attrs["perm"].is_array()) fail("transpose requires perm");
        std::vector<int64_t> shp;
        for (const auto& p : op.attrs["perm"]) shp.push_back(in_shape[p.get<int>()]);
        shape_env[out] = shp;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "broadcast_in_dim") {
        if (!op.attrs.contains("out_shape") || !op.attrs["out_shape"].is_array()) fail("broadcast_in_dim requires out_shape");
        std::vector<int64_t> shp;
        for (const auto& d : op.attrs["out_shape"]) {
          if (d.is_number_integer()) shp.push_back(d.get<int64_t>());
          else if (d.is_string()) {
            std::string s = d.get<std::string>();
            auto it = bindings.find(s);
            if (it == bindings.end()) fail("unbound symbol in broadcast_in_dim.out_shape: " + s);
            shp.push_back(it->second);
          } else fail("invalid out_shape dim");
        }
        shape_env[out] = shp;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "concat") {
        if (op.inputs.empty()) fail("concat requires at least one input");
        const auto& first = get_shape(op.inputs[0]);
        if (first.empty()) fail("concat expects rank >= 1");
        int axis = op.attrs.value("axis", 0);
        if (axis < 0) axis += static_cast<int>(first.size());
        if (axis < 0 || axis >= static_cast<int>(first.size())) fail("concat axis out of range");
        std::vector<int64_t> shp = first;
        shp[axis] = 0;
        for (const auto& in_name : op.inputs) {
          const auto& s = get_shape(in_name);
          if (s.size() != first.size()) fail("concat requires matching ranks");
          for (int i = 0; i < (int)s.size(); ++i) {
            if (i == axis) shp[i] += s[i];
            else if (s[i] != first[i]) fail("concat requires matching non-axis dimensions");
          }
        }
        shape_env[out] = shp;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "pad") {
        if (op.inputs.size() != 1) fail("pad expects one input tensor");
        const auto& in_shape = get_shape(op.inputs[0]);
        json padw = op.attrs.value("pad_width", json{});
        if (padw.is_object() && padw.contains("pairs")) padw = padw["pairs"];
        if (!padw.is_array() || padw.size() != in_shape.size()) fail("pad_width rank mismatch");
        std::vector<int64_t> shp = in_shape;
        for (int i = 0; i < (int)in_shape.size(); ++i) {
          const auto& pair = padw[i];
          if (!pair.is_array() || pair.size() != 2) fail("pad_width entries must be [left,right]");
          int64_t left = static_cast<int64_t>(resolve_const_value(pair[0], bindings));
          int64_t right = static_cast<int64_t>(resolve_const_value(pair[1], bindings));
          shp[i] = in_shape[i] + left + right;
          if (shp[i] < 0) fail("pad produced negative output dimension");
        }
        shape_env[out] = shp;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "add" || kind == "sub" || kind == "mul" || kind == "div" || kind == "max" || kind == "min" || kind == "pow" || kind == "remainder") {
        if (op.inputs.size() == 2) {
          const auto& sa = get_shape(op.inputs[0]);
          const auto& sb = get_shape(op.inputs[1]);
          shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
          dtype_env[out] = get_dtype(op.inputs[0]);
        } else if (op.inputs.size() == 1) {
          // Scalar-attr form: output shape matches the single tensor input.
          shape_env[out] = get_shape(op.inputs[0]);
          dtype_env[out] = get_dtype(op.inputs[0]);
        } else {
          fail("elemwise op requires 1 or 2 inputs");
        }
        continue;
      }
      if (kind == "eq" || kind == "ne" || kind == "lt" || kind == "le" || kind == "gt" || kind == "ge") {
        const auto& sa = get_shape(op.inputs[0]);
        const auto& sb = get_shape(op.inputs[1]);
        shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
        dtype_env[out] = "bool";
        continue;
      }
      if (kind == "and" || kind == "or" || kind == "bitwise_and" || kind == "bitwise_or" || kind == "bitwise_left_shift" || kind == "bitwise_right_shift") {
        const auto& sa = get_shape(op.inputs[0]);
        const auto& sb = get_shape(op.inputs[1]);
        shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
        const std::string in_dt = get_dtype(op.inputs[0]);
        if (kind == "and" || kind == "or") {
          if (in_dt == "bool" || in_dt == "i1" || in_dt == "u8") dtype_env[out] = "bool";
          else dtype_env[out] = in_dt;
        } else {
          dtype_env[out] = in_dt;
        }
        continue;
      }
      if (kind == "not" || kind == "bitwise_not") {
        shape_env[out] = get_shape(op.inputs[0]);
        const std::string in_dt = get_dtype(op.inputs[0]);
        if (kind == "not" && (in_dt == "bool" || in_dt == "i1" || in_dt == "u8")) {
          dtype_env[out] = "bool";
        } else {
          dtype_env[out] = in_dt;
        }
        continue;
      }
      if (kind == "argmax" || kind == "argmin") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail(kind + " infer currently supports rank-2 input");
        int axis = op.attrs.value("axis", -1);
        if (axis < 0) axis += static_cast<int>(in_shape.size());
        if (axis != 1) fail(kind + " infer currently supports axis=1");
        shape_env[out] = {in_shape[0]};
        dtype_env[out] = "i32";
        continue;
      }
      if (kind == "conv1d") {
        if (op.inputs.size() != 3) fail("conv1d infer expects inputs [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const auto& b_shape = get_shape(op.inputs[2]);
        if (in_shape.size() != 3 || w_shape.size() != 3 || b_shape.size() != 1) fail("conv1d infer expects input[N,C,L], weight[CO,C_PER_G,K], bias[CO]");
        const int64_t N = in_shape[0];
        const int64_t C_IN = in_shape[1];
        const int64_t L = in_shape[2];
        const int64_t C_OUT = w_shape[0];
        const int64_t C_PER_G = w_shape[1];
        const int64_t K = w_shape[2];
        if (b_shape[0] != C_OUT) fail("conv1d infer bias shape mismatch");
        const int64_t stride = op.attrs.contains("stride") ? static_cast<int64_t>(resolve_const_value(op.attrs["stride"], bindings)) : 1;
        const int64_t padding = op.attrs.contains("padding") ? static_cast<int64_t>(resolve_const_value(op.attrs["padding"], bindings)) : 0;
        const int64_t dilation = op.attrs.contains("dilation") ? static_cast<int64_t>(resolve_const_value(op.attrs["dilation"], bindings)) : 1;
        const int64_t groups = op.attrs.contains("groups") ? static_cast<int64_t>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
        if (stride <= 0 || dilation <= 0 || groups <= 0) fail("conv1d infer stride/dilation/groups must be positive");
        if (C_IN != C_PER_G * groups) fail("conv1d infer channel/group mismatch");
        if ((C_OUT % groups) != 0) fail("conv1d infer C_OUT must be divisible by groups");
        const int64_t OL = ((L + 2 * padding - dilation * (K - 1) - 1) / stride) + 1;
        if (OL <= 0) fail("conv1d infer produced non-positive output length");
        shape_env[out] = {N, C_OUT, OL};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "conv2d") {
        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv2d infer expects inputs [input, weight] or [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const bool has_bias = op.inputs.size() == 3;
        const auto& b_shape = has_bias ? get_shape(op.inputs[2]) : std::vector<int64_t>{};
        if (in_shape.size() != 4 || w_shape.size() != 4) fail("conv2d infer expects input[N,C,H,W], weight[CO,C_PER_G,KH,KW]");
        const int64_t N = in_shape[0];
        const int64_t C_IN_TOTAL = in_shape[1];
        const int64_t H = in_shape[2];
        const int64_t W = in_shape[3];
        const int64_t C_OUT = w_shape[0];
        const int64_t C_PER_G = w_shape[1];
        const int64_t KH = w_shape[2];
        const int64_t KW = w_shape[3];
        if (has_bias) {
          if (b_shape.size() != 1) fail("conv2d infer bias must be rank-1");
          if (b_shape[0] != C_OUT) fail("conv2d infer bias shape mismatch");
        }
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("conv2d infer " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int64_t, int64_t>{1, 1};
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int64_t, int64_t>{1, 1};
        const int64_t groups = op.attrs.contains("groups") ? static_cast<int64_t>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
        if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0 || groups <= 0)
          fail("conv2d infer stride/dilation/groups must be positive");
        if (C_IN_TOTAL != C_PER_G * groups) fail("conv2d infer channel/group mismatch");
        if ((C_OUT % groups) != 0) fail("conv2d infer C_OUT must be divisible by groups");
        const int64_t OH = ((H + 2 * padding.first - dilation.first * (KH - 1) - 1) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - dilation.second * (KW - 1) - 1) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("conv2d infer produced non-positive output shape");
        shape_env[out] = {N, C_OUT, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "conv3d") {
        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv3d infer expects inputs [input, weight] or [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const bool has_bias = op.inputs.size() == 3;
        const auto& b_shape = has_bias ? get_shape(op.inputs[2]) : std::vector<int64_t>{};
        if (in_shape.size() != 5 || w_shape.size() != 5) fail("conv3d infer expects input[N,C,D,H,W], weight[CO,C_PER_G,KD,KH,KW]");
        const int64_t N = in_shape[0];
        const int64_t C_IN_TOTAL = in_shape[1];
        const int64_t D = in_shape[2];
        const int64_t H = in_shape[3];
        const int64_t W = in_shape[4];
        const int64_t C_OUT = w_shape[0];
        const int64_t C_PER_G = w_shape[1];
        const int64_t KD = w_shape[2];
        const int64_t KH = w_shape[3];
        const int64_t KW = w_shape[4];
        if (has_bias) {
          if (b_shape.size() != 1) fail("conv3d infer bias must be rank-1");
          if (b_shape[0] != C_OUT) fail("conv3d infer bias shape mismatch");
        }
        auto parse_triple = [&](const json& value, const std::string& key) -> std::array<int64_t, 3> {
          if (value.is_array()) {
            if (value.size() != 3) fail("conv3d infer " + key + " must have length 3");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)),
                    static_cast<int64_t>(resolve_const_value(value[1], bindings)),
                    static_cast<int64_t>(resolve_const_value(value[2], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v, v};
        };
        auto stride = op.attrs.contains("stride") ? parse_triple(op.attrs["stride"], "stride") : std::array<int64_t, 3>{1, 1, 1};
        auto padding = op.attrs.contains("padding") ? parse_triple(op.attrs["padding"], "padding") : std::array<int64_t, 3>{0, 0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_triple(op.attrs["dilation"], "dilation") : std::array<int64_t, 3>{1, 1, 1};
        const int64_t groups = op.attrs.contains("groups") ? static_cast<int64_t>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
        if (stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0 || dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0 || groups <= 0)
          fail("conv3d infer stride/dilation/groups must be positive");
        if (C_IN_TOTAL != C_PER_G * groups) fail("conv3d infer channel/group mismatch");
        if ((C_OUT % groups) != 0) fail("conv3d infer C_OUT must be divisible by groups");
        const int64_t OD = ((D + 2 * padding[0] - dilation[0] * (KD - 1) - 1) / stride[0]) + 1;
        const int64_t OH = ((H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) / stride[1]) + 1;
        const int64_t OW = ((W + 2 * padding[2] - dilation[2] * (KW - 1) - 1) / stride[2]) + 1;
        if (OD <= 0 || OH <= 0 || OW <= 0) fail("conv3d infer produced non-positive output shape");
        shape_env[out] = {N, C_OUT, OD, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "conv_depthwise2d") {
        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv_depthwise2d infer expects inputs [input, weight] or [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const bool has_bias = op.inputs.size() == 3;
        const auto& b_shape = has_bias ? get_shape(op.inputs[2]) : std::vector<int64_t>{};
        if (in_shape.size() != 4 || w_shape.size() != 4) fail("conv_depthwise2d infer expects input[N,C,H,W], weight[C_OUT,1,KH,KW]");
        const int64_t N = in_shape[0];
        const int64_t C_IN = in_shape[1];
        const int64_t H = in_shape[2];
        const int64_t W = in_shape[3];
        const int64_t C_OUT = w_shape[0];
        const int64_t KH = w_shape[2];
        const int64_t KW = w_shape[3];
        if (w_shape[1] != 1) fail("conv_depthwise2d infer expects weight second dim == 1");
        if (C_IN <= 0 || C_OUT <= 0 || (C_OUT % C_IN) != 0) fail("conv_depthwise2d infer channel multiplier mismatch");
        if (has_bias) {
          if (b_shape.size() != 1) fail("conv_depthwise2d infer bias must be rank-1");
          if (b_shape[0] != C_OUT) fail("conv_depthwise2d infer bias shape mismatch");
        }
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("conv_depthwise2d infer " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int64_t, int64_t>{1, 1};
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int64_t, int64_t>{1, 1};
        if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
          fail("conv_depthwise2d infer stride/dilation must be positive");
        const int64_t OH = ((H + 2 * padding.first - dilation.first * (KH - 1) - 1) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - dilation.second * (KW - 1) - 1) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("conv_depthwise2d infer produced non-positive output shape");
        shape_env[out] = {N, C_OUT, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "avg_pool2d") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 4) fail("avg_pool2d infer expects rank-4 NCHW input");
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("avg_pool2d " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int64_t, int64_t>{2, 2};
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        const int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
        const int64_t OH = ((H + 2 * padding.first - ksz.first) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - ksz.second) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("avg_pool2d infer produced non-positive output shape");
        shape_env[out] = {N, C, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "max_pool2d_with_indices") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 4) fail("max_pool2d_with_indices infer expects rank-4 NCHW input");
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("max_pool2d_with_indices " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int64_t, int64_t>{2, 2};
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int64_t, int64_t>{1, 1};
        bool ceil_mode = op.attrs.value("ceil_mode", false);
        if (ceil_mode) fail("max_pool2d_with_indices infer ceil_mode=true is not supported");
        if (dilation.first != 1 || dilation.second != 1) fail("max_pool2d_with_indices infer dilation != 1 is not supported");
        const int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
        const int64_t OH = ((H + 2 * padding.first - ksz.first) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - ksz.second) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("max_pool2d_with_indices infer produced non-positive output shape");
        shape_env[out] = {N, C, OH, OW};
        std::string select = op.attrs.value("select", "values");
        dtype_env[out] = (select == "indices") ? "i64" : get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "upsample_nearest1d") {
        if (op.inputs.size() != 1) fail("upsample_nearest1d infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 3) fail("upsample_nearest1d infer expects rank-3 input [N,C,IL]");
        if (!intent.tensors.count(out)) fail("upsample_nearest1d output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) oshape.push_back(d.get<int64_t>());
          else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in upsample_nearest1d output shape: " + sym);
            oshape.push_back(it->second);
          } else fail("invalid upsample_nearest1d output dim");
        }
        if (oshape.size() != 3) fail("upsample_nearest1d infer expects rank-3 output [N,C,OL]");
        if (oshape[0] != in_shape[0] || oshape[1] != in_shape[1]) fail("upsample_nearest1d N/C mismatch");
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "upsample_nearest2d") {
        if (op.inputs.size() != 1) fail("upsample_nearest2d infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 4) fail("upsample_nearest2d infer expects rank-4 input [N,C,IH,IW]");
        if (!intent.tensors.count(out)) fail("upsample_nearest2d output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) oshape.push_back(d.get<int64_t>());
          else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in upsample_nearest2d output shape: " + sym);
            oshape.push_back(it->second);
          } else fail("invalid upsample_nearest2d output dim");
        }
        if (oshape.size() != 4) fail("upsample_nearest2d infer expects rank-4 output [N,C,OH,OW]");
        if (oshape[0] != in_shape[0] || oshape[1] != in_shape[1]) fail("upsample_nearest2d N/C mismatch");
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "scaled_dot_product_attention") {
        if (op.inputs.size() < 3) fail("scaled_dot_product_attention infer requires query/key/value");
        const auto& q = get_shape(op.inputs[0]);
        const auto& k = get_shape(op.inputs[1]);
        const auto& v = get_shape(op.inputs[2]);
        if (q.size() != 4 || k.size() != 4 || v.size() != 4) fail("scaled_dot_product_attention infer expects rank-4 tensors");
        if (q[0] != k[0] || q[1] != k[1] || k[2] != v[2]) fail("scaled_dot_product_attention infer shape mismatch");
        shape_env[out] = {q[0], q[1], q[2], v[3]};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "mse_loss") {
        if (op.inputs.size() != 2) fail("mse_loss infer expects [input, target]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& tgt_shape = get_shape(op.inputs[1]);
        if (in_shape != tgt_shape) fail("mse_loss infer shape mismatch between input and target");
        std::string reduction = "mean";
        if (op.attrs.contains("reduction")) {
          if (op.attrs["reduction"].is_number_integer()) {
            int mode = op.attrs["reduction"].get<int>();
            if (mode == 0) reduction = "none";
            else if (mode == 2) reduction = "sum";
            else reduction = "mean";
          } else {
            reduction = op.attrs["reduction"].get<std::string>();
          }
        }
        if (reduction == "none") {
          shape_env[out] = in_shape;
        } else {
          shape_env[out] = {};
        }
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "abs" || kind == "floor" || kind == "ceil" || kind == "acos" || kind == "atan" || kind == "tan" || kind == "cos" || kind == "erf" ||
          kind == "log" || kind == "sqrt") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "cast") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = op.attrs.value("to", "f32");
        continue;
      }
      if (kind == "iota") {
        if (!op.attrs.contains("shape") || !op.attrs["shape"].is_array()) fail("iota requires shape");
        std::vector<int64_t> shp;
        for (const auto& d : op.attrs["shape"]) {
          if (d.is_number_integer()) shp.push_back(d.get<int64_t>());
          else if (d.is_string()) {
            std::string s = d.get<std::string>();
            auto it = bindings.find(s);
            if (it == bindings.end()) fail("unbound symbol in iota.shape: " + s);
            shp.push_back(it->second);
          } else fail("invalid iota dim");
        }
        shape_env[out] = shp;
        dtype_env[out] = op.attrs.value("dtype", "i32");
        continue;
      }
      if (kind == "gather") {
        if (op.inputs.size() < 2) fail("gather requires data + indices");
        std::vector<int64_t> oshape = get_shape(op.inputs[1]);
        for (size_t i = 2; i < op.inputs.size(); ++i) oshape = broadcast_shape(oshape, get_shape(op.inputs[i]));
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "index_add") {
        if (op.inputs.size() != 3) fail("index_add expects inputs [base, index, src]");
        const auto& base_shape = get_shape(op.inputs[0]);
        const auto& idx_shape = get_shape(op.inputs[1]);
        const auto& src_shape = get_shape(op.inputs[2]);
        if (base_shape.size() != 2 || idx_shape.size() != 1 || src_shape.size() != 2) fail("index_add infer expects base[2], index[1], src[2]");
        int axis = op.attrs.value("axis", 0);
        int axis_norm = axis;
        if (axis_norm < 0) axis_norm += 2;
        if (axis_norm != 0) fail("index_add infer currently supports axis=0 only");
        if (src_shape[0] != idx_shape[0] || src_shape[1] != base_shape[1]) fail("index_add infer shape mismatch");
        shape_env[out] = base_shape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "index_put") {
        if (op.inputs.size() != 4) fail("index_put expects inputs [base, row_idx, col_idx, values]");
        const auto& base_shape = get_shape(op.inputs[0]);
        const auto& row_shape = get_shape(op.inputs[1]);
        const auto& col_shape = get_shape(op.inputs[2]);
        const auto& val_shape = get_shape(op.inputs[3]);
        if (base_shape.size() != 2 || row_shape.size() != 1 || col_shape.size() != 1 || val_shape.size() != 1) {
          fail("index_put infer expects base[2], row_idx[1], col_idx[1], values[1]");
        }
        if (row_shape[0] != col_shape[0] || row_shape[0] != val_shape[0]) fail("index_put infer length mismatch");
        shape_env[out] = base_shape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "kron") {
        if (op.inputs.size() != 2) fail("kron expects inputs [A, B]");
        const auto& a_shape = get_shape(op.inputs[0]);
        const auto& b_shape = get_shape(op.inputs[1]);
        if (a_shape.size() != 2 || b_shape.size() != 2) fail("kron infer currently supports rank-2 tensors");
        shape_env[out] = {a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "where") {
        std::vector<int64_t> oshape = broadcast_shape(get_shape(op.inputs[0]), get_shape(op.inputs[1]));
        oshape = broadcast_shape(oshape, get_shape(op.inputs[2]));
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[1]);
        continue;
      }
      if (kind == "std") {
        if (op.inputs.size() != 1) fail("std infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail("std infer currently supports rank-2 input");
        int axis = 1;
        if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            if (op.attrs["axis"].size() != 1) fail("std infer currently supports one reduction axis");
            axis = op.attrs["axis"][0].get<int>();
          } else {
            axis = op.attrs["axis"].get<int>();
          }
        } else if (op.attrs.contains("dims") && op.attrs["dims"].is_array()) {
          if (op.attrs["dims"].size() != 1) fail("std infer currently supports one reduction axis");
          axis = op.attrs["dims"][0].get<int>();
        } else if (op.attrs.contains("axes") && op.attrs["axes"].is_array()) {
          if (op.attrs["axes"].size() != 1) fail("std infer currently supports one reduction axis");
          axis = op.attrs["axes"][0].get<int>();
        }
        int axis_norm = axis;
        if (axis_norm < 0) axis_norm += static_cast<int>(in_shape.size());
        if (axis_norm != 1) fail("std infer currently supports axis=1");
        const bool keepdims = op.attrs.value("keepdims", false);
        if (keepdims) shape_env[out] = {in_shape[0], 1};
        else shape_env[out] = {in_shape[0]};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "sort") {
        if (op.inputs.size() != 1) fail("sort infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail("sort infer currently supports rank-2 input");
        int axis = op.attrs.value("axis", -1);
        if (axis < 0) axis += static_cast<int>(in_shape.size());
        if (axis != 1) fail("sort infer currently supports axis=1");
        shape_env[out] = in_shape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "unique") {
        if (op.inputs.size() != 1) fail("unique infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 1) fail("unique infer currently supports rank-1 input");
        if (!intent.tensors.count(out)) fail("unique output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) {
            oshape.push_back(d.get<int64_t>());
          } else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in unique output shape: " + sym);
            oshape.push_back(it->second);
          } else {
            fail("invalid unique output dim");
          }
        }
        if (oshape.size() != 1) fail("unique infer expects rank-1 output");
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "stack") {
        if (op.inputs.size() != 2) fail("stack infer currently supports exactly 2 inputs");
        const auto& a_shape = get_shape(op.inputs[0]);
        const auto& b_shape = get_shape(op.inputs[1]);
        if (a_shape.size() != 2 || b_shape.size() != 2) fail("stack infer currently supports rank-2 inputs");
        if (a_shape != b_shape) fail("stack infer input shape mismatch");
        int axis = op.attrs.value("axis", 0);
        if (axis < 0) axis += 3;
        if (axis != 0) fail("stack infer currently supports axis=0");
        shape_env[out] = {2, a_shape[0], a_shape[1]};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "cumsum" || kind == "cummax" || kind == "cummin") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "nonzero") {
        if (op.inputs.size() != 1) fail("nonzero expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail("nonzero infer currently supports rank-2 input");
        if (!intent.tensors.count(out)) fail("nonzero output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) {
            oshape.push_back(d.get<int64_t>());
          } else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in nonzero output shape: " + sym);
            oshape.push_back(it->second);
          } else {
            fail("invalid nonzero output dim");
          }
        }
        if (oshape.size() != 2 || oshape[1] != 2) fail("nonzero infer expects output shape [K,2]");
        shape_env[out] = oshape;
        dtype_env[out] = out_t.dtype;
        continue;
      }
      if (kind == "nll_loss_forward") {
        if (op.inputs.size() != 3) fail("nll_loss_forward expects inputs [self,target,weight]");
        const auto& logits_shape = get_shape(op.inputs[0]);
        const auto& target_shape = get_shape(op.inputs[1]);
        const auto& weight_shape = get_shape(op.inputs[2]);
        if (logits_shape.size() != 2 || target_shape.size() != 1 || weight_shape.size() != 1) {
          fail("nll_loss_forward infer expects self[N,C], target[N], weight[C]");
        }
        const int64_t N = logits_shape[0];
        const int64_t C = logits_shape[1];
        if (target_shape[0] != N) fail("nll_loss_forward infer target length mismatch");
        if (weight_shape[0] != C) fail("nll_loss_forward infer weight length mismatch");
        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
        if (reduction == 0) shape_env[out] = {N};
        else shape_env[out] = {};
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "nll_loss2d_forward") {
        if (op.inputs.size() != 3) fail("nll_loss2d_forward expects inputs [self,target,weight]");
        const auto& logits_shape = get_shape(op.inputs[0]);
        const auto& target_shape = get_shape(op.inputs[1]);
        const auto& weight_shape = get_shape(op.inputs[2]);
        if (logits_shape.size() != 4 || target_shape.size() != 3 || weight_shape.size() != 1) {
          fail("nll_loss2d_forward infer expects self[N,C,H,W], target[N,H,W], weight[C]");
        }
        const int64_t N = logits_shape[0];
        const int64_t C = logits_shape[1];
        const int64_t H = logits_shape[2];
        const int64_t W = logits_shape[3];
        if (target_shape[0] != N || target_shape[1] != H || target_shape[2] != W) {
          fail("nll_loss2d_forward infer target shape mismatch");
        }
        if (weight_shape[0] != C) fail("nll_loss2d_forward infer weight length mismatch");
        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
        if (reduction == 0) shape_env[out] = {N, H, W};
        else shape_env[out] = {};
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "glu") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.empty()) fail("glu expects rank >= 1 input");
        int axis = op.attrs.value("axis", -1);
        int axis_norm = axis;
        if (axis_norm < 0) axis_norm += static_cast<int>(in_shape.size());
        if (axis_norm < 0 || axis_norm >= static_cast<int>(in_shape.size())) fail("glu axis out of range");
        if ((in_shape[axis_norm] % 2) != 0) fail("glu requires even extent along selected axis");
        std::vector<int64_t> oshape = in_shape;
        oshape[axis_norm] = in_shape[axis_norm] / 2;
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "rsqrt" || kind == "exp" || kind == "log" || kind == "relu" || kind == "softmax") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "polar") {
        if (op.inputs.size() != 2) fail("polar infer expects inputs [abs, angle]");
        const auto& abs_shape = get_shape(op.inputs[0]);
        const auto& angle_shape = get_shape(op.inputs[1]);
        if (abs_shape != angle_shape) fail("polar infer expects abs/angle shape match");
        if (abs_shape.size() != 2) fail("polar infer currently supports rank-2 inputs");
        shape_env[out] = {abs_shape[0], abs_shape[1], 2};
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "reduce_sum" || kind == "reduce_prod" || kind == "reduce_max" || kind == "reduce_min" || kind == "reduce_any") {
        const auto& in_shape = get_shape(op.inputs[0]);
        std::unordered_map<int,int> dims_set;
        if (op.attrs.contains("dims") && op.attrs["dims"].is_array()) {
          for (const auto& d : op.attrs["dims"]) dims_set.emplace(d.get<int>(), 1);
        } else if (op.attrs.contains("axes") && op.attrs["axes"].is_array()) {
          for (const auto& d : op.attrs["axes"]) dims_set.emplace(d.get<int>(), 1);
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims_set.emplace(d.get<int>(), 1);
          } else {
            dims_set.emplace(op.attrs["axis"].get<int>(), 1);
          }
        } else {
          fail(kind + " requires dims/axes list[int]");
        }
        bool keepdims = op.attrs.value("keepdims", false);
        std::vector<int64_t> oshape;
        for (int i = 0; i < (int)in_shape.size(); ++i) {
          if (dims_set.count(i)) {
            if (keepdims) oshape.push_back(1);
          } else {
            oshape.push_back(in_shape[i]);
          }
        }
        shape_env[out] = oshape;
        dtype_env[out] = (kind == "reduce_any") ? "bool" : "f32";
        continue;
      }
      if (kind == "matmul") {
        const auto& sa = get_shape(op.inputs[0]);
        const auto& sb = get_shape(op.inputs[1]);
        bool ta = op.attrs.value("transpose_a", false);
        bool tb = op.attrs.value("transpose_b", false);
        if (sa.size() == 2 && sb.size() == 2) {
          int64_t M = ta ? sa[1] : sa[0];
          int64_t K = ta ? sa[0] : sa[1];
          int64_t K2 = tb ? sb[1] : sb[0];
          int64_t N = tb ? sb[0] : sb[1];
          if (K2 != K) fail("matmul infer shape mismatch (2D)");
          shape_env[out] = {M, N};
          dtype_env[out] = "f32";
          continue;
        }
        if (sa.size() == 2 && sb.size() == 1) {
          if (tb) fail("matmul infer rank-2 x rank-1 does not support transpose_b");
          int64_t M = ta ? sa[1] : sa[0];
          int64_t K = ta ? sa[0] : sa[1];
          int64_t K2 = sb[0];
          if (K2 != K) fail("matmul infer shape mismatch (2D x 1D)");
          shape_env[out] = {M};
          dtype_env[out] = "f32";
          continue;
        }
        if (sa.size() == 3 && sb.size() == 3) {
          int64_t B = sa[0];
          int64_t M = ta ? sa[2] : sa[1];
          int64_t K = ta ? sa[1] : sa[2];
          int64_t B2 = sb[0];
          int64_t K2 = tb ? sb[2] : sb[1];
          int64_t N = tb ? sb[1] : sb[2];
          if (B2 != B || K2 != K) fail("matmul infer shape mismatch (3D)");
          shape_env[out] = {B, M, N};
          dtype_env[out] = "f32";
          continue;
        }
        if (sa.size() == 4 && sb.size() == 4) {
          if (sa[0] != sb[0] || sa[1] != sb[1]) fail("matmul infer shape mismatch (4D batch/head)");
          int64_t B = sa[0], H = sa[1];
          int64_t M = ta ? sa[3] : sa[2];
          int64_t K = ta ? sa[2] : sa[3];
          int64_t K2 = tb ? sb[3] : sb[2];
          int64_t N = tb ? sb[2] : sb[3];
          if (K2 != K) fail("matmul infer shape mismatch (4D)");
          shape_env[out] = {B, H, M, N};
          dtype_env[out] = "f32";
          continue;
        }
        fail("matmul infer supports rank-2/3/4 and rank-2x1");
      }
      fail("cannot infer output shape for op: " + kind);
