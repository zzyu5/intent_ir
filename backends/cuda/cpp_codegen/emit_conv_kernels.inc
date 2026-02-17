json emit_conv1d_ncl_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "conv1d")) fail("conv1d lowering expects a single conv1d op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv1d expects [input, weight] or [input, weight, bias]");
  const std::string x_name = op.inputs[0];
  const std::string w_name = op.inputs[1];
  const bool has_bias = op.inputs.size() == 3;
  const std::string b_name = has_bias ? op.inputs[2] : std::string();
  const std::string out_name = op.output;

  auto x_it = intent.tensors.find(x_name);
  auto w_it = intent.tensors.find(w_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || w_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("conv1d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "conv1d.input.shape");
  auto w_shape = resolve_shape_required(w_it->second, bindings, "conv1d.weight.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "conv1d.output.shape");
  if (x_shape.size() != 3 || w_shape.size() != 3 || o_shape.size() != 3) fail("conv1d expects rank-3 tensors");
  if (x_it->second.dtype != "f32" || w_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("conv1d supports f32 only");

  int64_t N = x_shape[0], C_IN_TOTAL = x_shape[1], L = x_shape[2];
  int64_t C_OUT = w_shape[0], C_PER_G = w_shape[1], K = w_shape[2];
  int64_t OL = o_shape[2];
  int stride = parse_attr_int(op.attrs, "stride", 1, bindings, "conv1d");
  int padding = parse_attr_int(op.attrs, "padding", 0, bindings, "conv1d");
  int dilation = parse_attr_int(op.attrs, "dilation", 1, bindings, "conv1d");
  int groups = parse_attr_int(op.attrs, "groups", 1, bindings, "conv1d");
  if (groups <= 0 || stride <= 0 || dilation <= 0) fail("conv1d requires positive stride/dilation/groups");
  if (C_IN_TOTAL != C_PER_G * groups) fail("conv1d channel/group mismatch");
  if ((C_OUT % groups) != 0) fail("conv1d C_OUT must be divisible by groups");
  int64_t expected_ol = ((L + 2LL * padding - (int64_t)dilation * (K - 1) - 1) / stride) + 1;
  if (expected_ol != OL) fail("conv1d output shape mismatch");

  if (has_bias) {
    auto b_it = intent.tensors.find(b_name);
    if (b_it == intent.tensors.end()) fail("conv1d bias tensor missing");
    auto b_shape = resolve_shape_required(b_it->second, bindings, "conv1d.bias.shape");
    if (b_it->second.dtype != "f32" || b_shape.size() != 1 || b_shape[0] != C_OUT) fail("conv1d bias must be f32[C_OUT]");
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C_OUT * OL;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("const float* __restrict__ " + w_name + ",");
  if (has_bias) w.line("const float* __restrict__ " + b_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ol = t % " + std::to_string(OL) + "LL; t /= " + std::to_string(OL) + "LL;");
  w.line("const int64_t co = t % " + std::to_string(C_OUT) + "LL; t /= " + std::to_string(C_OUT) + "LL;");
  w.line("const int64_t n = t;");
  w.line("const int64_t co_per_g = " + std::to_string(C_OUT / groups) + "LL;");
  w.line("const int64_t g = co / co_per_g;");
  w.line("const int64_t c_start = g * " + std::to_string(C_PER_G) + "LL;");
  if (has_bias)
    w.line("float acc = " + b_name + "[co];");
  else
    w.line("float acc = 0.0f;");
  w.line("for (int64_t ci = 0; ci < " + std::to_string(C_PER_G) + "LL; ++ci) {");
  w.indent();
  w.line("const int64_t c = c_start + ci;");
  w.line("for (int64_t k = 0; k < " + std::to_string(K) + "LL; ++k) {");
  w.indent();
  w.line("const int64_t li = ol * " + std::to_string(stride) + "LL - " + std::to_string(padding) + "LL + k * " + std::to_string(dilation) + "LL;");
  w.line("if ((unsigned long long)li >= (unsigned long long)" + std::to_string(L) + "LL) continue;");
  w.line("const int64_t x_idx = ((n * " + std::to_string(C_IN_TOTAL) + "LL + c) * " + std::to_string(L) + "LL + li);");
  w.line("const int64_t w_idx = ((co * " + std::to_string(C_PER_G) + "LL + ci) * " + std::to_string(K) + "LL + k);");
  w.line("acc += " + x_name + "[x_idx] * " + w_name + "[w_idx];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = acc;");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {x_name, w_name};
  if (has_bias) tensor_args.push_back(b_name);
  tensor_args.push_back(out_name);

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, tensor_args);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_conv2d_nchw_f32(const Intent& intent, const json& bindings) {
  const Op* conv_op = nullptr;
  std::string x_name;
  std::string w_name;
  std::string b_name;
  std::string out_name;
  bool has_bias = false;

  if (intent.ops.size() == 1 && intent.ops[0].op == "conv2d") {
    const Op& op = intent.ops[0];
    if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv2d expects [input, weight] or [input, weight, bias]");
    conv_op = &op;
    x_name = op.inputs[0];
    w_name = op.inputs[1];
    if (op.inputs.size() == 3) {
      has_bias = true;
      b_name = op.inputs[2];
    }
    out_name = op.output;
  } else if (intent.ops.size() == 3 && intent.ops[0].op == "conv2d" && intent.ops[1].op == "broadcast_in_dim" && intent.ops[2].op == "add") {
    const Op& op = intent.ops[0];
    const Op& bcast = intent.ops[1];
    const Op& add = intent.ops[2];
    if (op.inputs.size() != 2) fail("conv2d+bias pattern expects conv2d with [input, weight]");
    if (bcast.inputs.size() != 1) fail("conv2d+bias pattern expects broadcast_in_dim(bias)");
    if (add.inputs.size() != 2) fail("conv2d+bias pattern expects add(conv_out, bias_bcast)");
    if (add.inputs[0] != op.output && add.inputs[1] != op.output) fail("conv2d+bias pattern add must consume conv output");
    if (add.inputs[0] != bcast.output && add.inputs[1] != bcast.output) fail("conv2d+bias pattern add must consume broadcast bias");
    conv_op = &op;
    x_name = op.inputs[0];
    w_name = op.inputs[1];
    has_bias = true;
    b_name = bcast.inputs[0];
    out_name = add.output;
  } else {
    fail("conv2d lowering expects conv2d or conv2d+broadcast_in_dim+add pattern");
  }

  auto x_it = intent.tensors.find(x_name);
  auto w_it = intent.tensors.find(w_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || w_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("conv2d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "conv2d.input.shape");
  auto w_shape = resolve_shape_required(w_it->second, bindings, "conv2d.weight.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "conv2d.output.shape");
  if (x_shape.size() != 4 || w_shape.size() != 4 || o_shape.size() != 4) fail("conv2d expects rank-4 NCHW tensors");
  if (x_it->second.dtype != "f32" || w_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("conv2d supports f32 only");

  int64_t N = x_shape[0], C_IN_TOTAL = x_shape[1], H = x_shape[2], W = x_shape[3];
  int64_t C_OUT = w_shape[0], C_PER_G = w_shape[1], KH = w_shape[2], KW = w_shape[3];
  int64_t OH = o_shape[2], OW = o_shape[3];
  auto stride = parse_attr_pair(conv_op->attrs, "stride", 1, bindings, "conv2d");
  auto padding = parse_attr_pair(conv_op->attrs, "padding", 0, bindings, "conv2d");
  auto dilation = parse_attr_pair(conv_op->attrs, "dilation", 1, bindings, "conv2d");
  int groups = parse_attr_int(conv_op->attrs, "groups", 1, bindings, "conv2d");
  if (groups <= 0 || stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
    fail("conv2d requires positive stride/dilation/groups");
  if (C_IN_TOTAL != C_PER_G * groups) fail("conv2d channel/group mismatch");
  if ((C_OUT % groups) != 0) fail("conv2d C_OUT must be divisible by groups");
  int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
  int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
  if (expected_oh != OH || expected_ow != OW) fail("conv2d output shape mismatch");

  if (has_bias) {
    auto b_it = intent.tensors.find(b_name);
    if (b_it == intent.tensors.end()) fail("conv2d bias tensor missing");
    auto b_shape = resolve_shape_required(b_it->second, bindings, "conv2d.bias.shape");
    if (b_it->second.dtype != "f32" || b_shape.size() != 1 || b_shape[0] != C_OUT) fail("conv2d bias must be f32[C_OUT]");
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C_OUT * OH * OW;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("const float* __restrict__ " + w_name + ",");
  if (has_bias) w.line("const float* __restrict__ " + b_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t co = t % " + std::to_string(C_OUT) + "LL; t /= " + std::to_string(C_OUT) + "LL;");
  w.line("const int64_t n = t;");
  w.line("const int64_t co_per_g = " + std::to_string(C_OUT / groups) + "LL;");
  w.line("const int64_t g = co / co_per_g;");
  w.line("const int64_t c_start = g * " + std::to_string(C_PER_G) + "LL;");
  if (has_bias)
    w.line("float acc = " + b_name + "[co];");
  else
    w.line("float acc = 0.0f;");
  w.line("for (int64_t ci = 0; ci < " + std::to_string(C_PER_G) + "LL; ++ci) {");
  w.indent();
  w.line("const int64_t c = c_start + ci;");
  w.line("for (int64_t kh = 0; kh < " + std::to_string(KH) + "LL; ++kh) {");
  w.indent();
  w.line("const int64_t ih = oh * " + std::to_string(stride.first) + "LL - " + std::to_string(padding.first) + "LL + kh * " + std::to_string(dilation.first) +
         "LL;");
  w.line("if ((unsigned long long)ih >= (unsigned long long)" + std::to_string(H) + "LL) continue;");
  w.line("for (int64_t kw = 0; kw < " + std::to_string(KW) + "LL; ++kw) {");
  w.indent();
  w.line("const int64_t iw = ow * " + std::to_string(stride.second) + "LL - " + std::to_string(padding.second) + "LL + kw * " +
         std::to_string(dilation.second) + "LL;");
  w.line("if ((unsigned long long)iw >= (unsigned long long)" + std::to_string(W) + "LL) continue;");
  w.line("const int64_t x_idx = (((n * " + std::to_string(C_IN_TOTAL) + "LL + c) * " + std::to_string(H) + "LL + ih) * " + std::to_string(W) + "LL + iw);");
  w.line("const int64_t w_idx = (((co * " + std::to_string(C_PER_G) + "LL + ci) * " + std::to_string(KH) + "LL + kh) * " + std::to_string(KW) + "LL + kw);");
  w.line("acc += " + x_name + "[x_idx] * " + w_name + "[w_idx];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = acc;");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {x_name, w_name};
  if (has_bias) tensor_args.push_back(b_name);
  tensor_args.push_back(out_name);

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, tensor_args);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_conv3d_ncdhw_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "conv3d")) fail("conv3d lowering expects a single conv3d op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv3d expects [input, weight] or [input, weight, bias]");
  const std::string x_name = op.inputs[0];
  const std::string w_name = op.inputs[1];
  const bool has_bias = op.inputs.size() == 3;
  const std::string b_name = has_bias ? op.inputs[2] : std::string();
  const std::string out_name = op.output;

  auto x_it = intent.tensors.find(x_name);
  auto w_it = intent.tensors.find(w_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || w_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("conv3d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "conv3d.input.shape");
  auto w_shape = resolve_shape_required(w_it->second, bindings, "conv3d.weight.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "conv3d.output.shape");
  if (x_shape.size() != 5 || w_shape.size() != 5 || o_shape.size() != 5) fail("conv3d expects rank-5 NCDHW tensors");
  if (x_it->second.dtype != "f32" || w_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("conv3d supports f32 only");

  int64_t N = x_shape[0], C_IN_TOTAL = x_shape[1], D = x_shape[2], H = x_shape[3], W = x_shape[4];
  int64_t C_OUT = w_shape[0], C_PER_G = w_shape[1], KD = w_shape[2], KH = w_shape[3], KW = w_shape[4];
  int64_t OD = o_shape[2], OH = o_shape[3], OW = o_shape[4];
  auto stride = parse_attr_triple(op.attrs, "stride", 1, bindings, "conv3d");
  auto padding = parse_attr_triple(op.attrs, "padding", 0, bindings, "conv3d");
  auto dilation = parse_attr_triple(op.attrs, "dilation", 1, bindings, "conv3d");
  int groups = parse_attr_int(op.attrs, "groups", 1, bindings, "conv3d");
  if (groups <= 0 || stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0 || dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0)
    fail("conv3d requires positive stride/dilation/groups");
  if (C_IN_TOTAL != C_PER_G * groups) fail("conv3d channel/group mismatch");
  if ((C_OUT % groups) != 0) fail("conv3d C_OUT must be divisible by groups");
  int64_t expected_od = ((D + 2LL * padding[0] - (int64_t)dilation[0] * (KD - 1) - 1) / stride[0]) + 1;
  int64_t expected_oh = ((H + 2LL * padding[1] - (int64_t)dilation[1] * (KH - 1) - 1) / stride[1]) + 1;
  int64_t expected_ow = ((W + 2LL * padding[2] - (int64_t)dilation[2] * (KW - 1) - 1) / stride[2]) + 1;
  if (expected_od != OD || expected_oh != OH || expected_ow != OW) fail("conv3d output shape mismatch");

  if (has_bias) {
    auto b_it = intent.tensors.find(b_name);
    if (b_it == intent.tensors.end()) fail("conv3d bias tensor missing");
    auto b_shape = resolve_shape_required(b_it->second, bindings, "conv3d.bias.shape");
    if (b_it->second.dtype != "f32" || b_shape.size() != 1 || b_shape[0] != C_OUT) fail("conv3d bias must be f32[C_OUT]");
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C_OUT * OD * OH * OW;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("const float* __restrict__ " + w_name + ",");
  if (has_bias) w.line("const float* __restrict__ " + b_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t od = t % " + std::to_string(OD) + "LL; t /= " + std::to_string(OD) + "LL;");
  w.line("const int64_t co = t % " + std::to_string(C_OUT) + "LL; t /= " + std::to_string(C_OUT) + "LL;");
  w.line("const int64_t n = t;");
  w.line("const int64_t co_per_g = " + std::to_string(C_OUT / groups) + "LL;");
  w.line("const int64_t g = co / co_per_g;");
  w.line("const int64_t c_start = g * " + std::to_string(C_PER_G) + "LL;");
  if (has_bias)
    w.line("float acc = " + b_name + "[co];");
  else
    w.line("float acc = 0.0f;");
  w.line("for (int64_t ci = 0; ci < " + std::to_string(C_PER_G) + "LL; ++ci) {");
  w.indent();
  w.line("const int64_t c = c_start + ci;");
  w.line("for (int64_t kd = 0; kd < " + std::to_string(KD) + "LL; ++kd) {");
  w.indent();
  w.line("const int64_t id = od * " + std::to_string(stride[0]) + "LL - " + std::to_string(padding[0]) + "LL + kd * " + std::to_string(dilation[0]) +
         "LL;");
  w.line("if ((unsigned long long)id >= (unsigned long long)" + std::to_string(D) + "LL) continue;");
  w.line("for (int64_t kh = 0; kh < " + std::to_string(KH) + "LL; ++kh) {");
  w.indent();
  w.line("const int64_t ih = oh * " + std::to_string(stride[1]) + "LL - " + std::to_string(padding[1]) + "LL + kh * " + std::to_string(dilation[1]) +
         "LL;");
  w.line("if ((unsigned long long)ih >= (unsigned long long)" + std::to_string(H) + "LL) continue;");
  w.line("for (int64_t kw = 0; kw < " + std::to_string(KW) + "LL; ++kw) {");
  w.indent();
  w.line("const int64_t iw = ow * " + std::to_string(stride[2]) + "LL - " + std::to_string(padding[2]) + "LL + kw * " + std::to_string(dilation[2]) +
         "LL;");
  w.line("if ((unsigned long long)iw >= (unsigned long long)" + std::to_string(W) + "LL) continue;");
  w.line("const int64_t x_idx = ((((n * " + std::to_string(C_IN_TOTAL) + "LL + c) * " + std::to_string(D) + "LL + id) * " + std::to_string(H) +
         "LL + ih) * " + std::to_string(W) + "LL + iw);");
  w.line("const int64_t w_idx = ((((co * " + std::to_string(C_PER_G) + "LL + ci) * " + std::to_string(KD) + "LL + kd) * " + std::to_string(KH) +
         "LL + kh) * " + std::to_string(KW) + "LL + kw);");
  w.line("acc += " + x_name + "[x_idx] * " + w_name + "[w_idx];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = acc;");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {x_name, w_name};
  if (has_bias) tensor_args.push_back(b_name);
  tensor_args.push_back(out_name);

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, tensor_args);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_conv_depthwise2d_nchw_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "conv_depthwise2d")) fail("conv_depthwise2d lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv_depthwise2d expects [input, weight] or [input, weight, bias]");
  const std::string x_name = op.inputs[0];
  const std::string w_name = op.inputs[1];
  const bool has_bias = op.inputs.size() == 3;
  const std::string b_name = has_bias ? op.inputs[2] : std::string();
  const std::string out_name = op.output;

  auto x_it = intent.tensors.find(x_name);
  auto w_it = intent.tensors.find(w_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || w_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("conv_depthwise2d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "conv_depthwise2d.input.shape");
  auto w_shape = resolve_shape_required(w_it->second, bindings, "conv_depthwise2d.weight.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "conv_depthwise2d.output.shape");
  if (x_shape.size() != 4 || w_shape.size() != 4 || o_shape.size() != 4) fail("conv_depthwise2d expects rank-4 NCHW tensors");
  if (x_it->second.dtype != "f32" || w_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("conv_depthwise2d supports f32 only");
  if (w_shape[1] != 1) fail("conv_depthwise2d expects weight shape [C_OUT,1,KH,KW]");

  int64_t N = x_shape[0], C_IN = x_shape[1], H = x_shape[2], W = x_shape[3];
  int64_t C_OUT = w_shape[0], KH = w_shape[2], KW = w_shape[3];
  int64_t OH = o_shape[2], OW = o_shape[3];
  if (C_IN <= 0 || C_OUT <= 0 || (C_OUT % C_IN) != 0) fail("conv_depthwise2d channel multiplier mismatch");
  int64_t channel_multiplier = C_OUT / C_IN;
  auto stride = parse_attr_pair(op.attrs, "stride", 1, bindings, "conv_depthwise2d");
  auto padding = parse_attr_pair(op.attrs, "padding", 0, bindings, "conv_depthwise2d");
  auto dilation = parse_attr_pair(op.attrs, "dilation", 1, bindings, "conv_depthwise2d");
  if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
    fail("conv_depthwise2d requires positive stride/dilation");
  int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
  int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
  if (expected_oh != OH || expected_ow != OW) fail("conv_depthwise2d output shape mismatch");

  if (has_bias) {
    auto b_it = intent.tensors.find(b_name);
    if (b_it == intent.tensors.end()) fail("conv_depthwise2d bias tensor missing");
    auto b_shape = resolve_shape_required(b_it->second, bindings, "conv_depthwise2d.bias.shape");
    if (b_it->second.dtype != "f32" || b_shape.size() != 1 || b_shape[0] != C_OUT) fail("conv_depthwise2d bias must be f32[C_OUT]");
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C_OUT * OH * OW;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("const float* __restrict__ " + w_name + ",");
  if (has_bias) w.line("const float* __restrict__ " + b_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t co = t % " + std::to_string(C_OUT) + "LL; t /= " + std::to_string(C_OUT) + "LL;");
  w.line("const int64_t n = t;");
  w.line("const int64_t ci = co / " + std::to_string(channel_multiplier) + "LL;");
  if (has_bias)
    w.line("float acc = " + b_name + "[co];");
  else
    w.line("float acc = 0.0f;");
  w.line("for (int64_t kh = 0; kh < " + std::to_string(KH) + "LL; ++kh) {");
  w.indent();
  w.line("const int64_t ih = oh * " + std::to_string(stride.first) + "LL - " + std::to_string(padding.first) + "LL + kh * " + std::to_string(dilation.first) +
         "LL;");
  w.line("if ((unsigned long long)ih >= (unsigned long long)" + std::to_string(H) + "LL) continue;");
  w.line("for (int64_t kw = 0; kw < " + std::to_string(KW) + "LL; ++kw) {");
  w.indent();
  w.line("const int64_t iw = ow * " + std::to_string(stride.second) + "LL - " + std::to_string(padding.second) + "LL + kw * " +
         std::to_string(dilation.second) + "LL;");
  w.line("if ((unsigned long long)iw >= (unsigned long long)" + std::to_string(W) + "LL) continue;");
  w.line("const int64_t x_idx = (((n * " + std::to_string(C_IN) + "LL + ci) * " + std::to_string(H) + "LL + ih) * " + std::to_string(W) + "LL + iw);");
  w.line("const int64_t w_idx = ((co * " + std::to_string(KH) + "LL + kh) * " + std::to_string(KW) + "LL + kw);");
  w.line("acc += " + x_name + "[x_idx] * " + w_name + "[w_idx];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = acc;");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {x_name, w_name};
  if (has_bias) tensor_args.push_back(b_name);
  tensor_args.push_back(out_name);

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, tensor_args);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

