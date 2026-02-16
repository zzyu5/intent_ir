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

json emit_avg_pool2d_nchw_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "avg_pool2d")) fail("avg_pool2d lowering expects single avg_pool2d op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("avg_pool2d expects one input");
  const std::string x_name = op.inputs[0];
  const std::string out_name = op.output;
  auto x_it = intent.tensors.find(x_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("avg_pool2d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "avg_pool2d.input.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "avg_pool2d.output.shape");
  if (x_shape.size() != 4 || o_shape.size() != 4) fail("avg_pool2d expects rank-4 NCHW tensors");
  if (x_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("avg_pool2d supports f32 only");

  int64_t N = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
  int64_t OH = o_shape[2], OW = o_shape[3];
  auto ksz = parse_attr_pair(op.attrs, "kernel_size", 2, bindings, "avg_pool2d");
  auto stride = parse_attr_pair(op.attrs, "stride", ksz.first, bindings, "avg_pool2d");
  auto padding = parse_attr_pair(op.attrs, "padding", 0, bindings, "avg_pool2d");
  bool count_include_pad = op.attrs.value("count_include_pad", true);
  if (ksz.first <= 0 || ksz.second <= 0 || stride.first <= 0 || stride.second <= 0) fail("avg_pool2d requires positive kernel/stride");
  int64_t expected_oh = ((H + 2LL * padding.first - ksz.first) / stride.first) + 1;
  int64_t expected_ow = ((W + 2LL * padding.second - ksz.second) / stride.second) + 1;
  if (expected_oh != OH || expected_ow != OW) fail("avg_pool2d output shape mismatch");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C * OH * OW;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t c = t % " + std::to_string(C) + "LL; t /= " + std::to_string(C) + "LL;");
  w.line("const int64_t n = t;");
  w.line("float acc = 0.0f;");
  w.line("int valid = 0;");
  w.line("for (int kh = 0; kh < " + std::to_string(ksz.first) + "; ++kh) {");
  w.indent();
  w.line("const int64_t ih = oh * " + std::to_string(stride.first) + "LL - " + std::to_string(padding.first) + "LL + (int64_t)kh;");
  w.line("for (int kw = 0; kw < " + std::to_string(ksz.second) + "; ++kw) {");
  w.indent();
  w.line("const int64_t iw = ow * " + std::to_string(stride.second) + "LL - " + std::to_string(padding.second) + "LL + (int64_t)kw;");
  w.line("if ((unsigned long long)ih < (unsigned long long)" + std::to_string(H) + "LL && (unsigned long long)iw < (unsigned long long)" + std::to_string(W) +
         "LL) {");
  w.indent();
  w.line("const int64_t idx = (((n * " + std::to_string(C) + "LL + c) * " + std::to_string(H) + "LL + ih) * " + std::to_string(W) + "LL + iw);");
  w.line("acc += " + x_name + "[idx];");
  w.line("++valid;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  if (count_include_pad)
    w.line("const float denom = (float)(" + std::to_string(ksz.first * ksz.second) + ");");
  else
    w.line("const float denom = (valid > 0) ? (float)valid : 1.0f;");
  w.line(out_name + "[tid] = acc / denom;");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {x_name, out_name}, {}, {x_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_max_pool2d_with_indices_nchw_f32_i64(const Intent& intent, const json& bindings) {
  if (intent.ops.empty()) fail("max_pool2d_with_indices lowering expects non-empty ops");
  if (!(intent.ops.size() == 1 || intent.ops.size() == 2)) fail("max_pool2d_with_indices lowering expects 1 or 2 ops");
  if (intent.ops[0].op != "max_pool2d_with_indices") fail("max_pool2d_with_indices lowering expects max_pool2d_with_indices ops");
  if (intent.ops.size() == 2 && intent.ops[1].op != "max_pool2d_with_indices") {
    fail("max_pool2d_with_indices lowering expects max_pool2d_with_indices ops");
  }

  const Op& ref = intent.ops[0];
  if (ref.inputs.size() != 1) fail("max_pool2d_with_indices expects one input tensor");
  const std::string x_name = ref.inputs[0];
  std::string out_val_name;
  std::string out_idx_name;

  auto pick_select = [](const Op& op) -> std::string {
    if (op.attrs.is_object() && op.attrs.contains("select") && op.attrs["select"].is_string()) {
      return op.attrs["select"].get<std::string>();
    }
    return "values";
  };
  auto register_out = [&](const Op& op) {
    const std::string sel = pick_select(op);
    if (sel == "indices")
      out_idx_name = op.output;
    else
      out_val_name = op.output;
  };
  register_out(intent.ops[0]);
  if (intent.ops.size() == 2) register_out(intent.ops[1]);
  if (out_val_name.empty() && out_idx_name.empty()) fail("max_pool2d_with_indices has no outputs");

  auto x_it = intent.tensors.find(x_name);
  if (x_it == intent.tensors.end()) fail("max_pool2d_with_indices input tensor missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "max_pool2d_with_indices.input.shape");
  if (x_shape.size() != 4) fail("max_pool2d_with_indices expects rank-4 NCHW input");
  if (x_it->second.dtype != "f32") fail("max_pool2d_with_indices supports f32 input only");
  int64_t N = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];

  std::vector<int64_t> out_shape;
  if (!out_val_name.empty()) {
    auto it = intent.tensors.find(out_val_name);
    if (it == intent.tensors.end()) fail("max_pool2d_with_indices values tensor missing");
    out_shape = resolve_shape_required(it->second, bindings, "max_pool2d_with_indices.values.shape");
    if (it->second.dtype != "f32") fail("max_pool2d_with_indices values output must be f32");
  }
  if (!out_idx_name.empty()) {
    auto it = intent.tensors.find(out_idx_name);
    if (it == intent.tensors.end()) fail("max_pool2d_with_indices indices tensor missing");
    auto idx_shape = resolve_shape_required(it->second, bindings, "max_pool2d_with_indices.indices.shape");
    if (it->second.dtype != "i64") fail("max_pool2d_with_indices indices output must be i64");
    if (out_shape.empty()) out_shape = idx_shape;
    if (idx_shape != out_shape) fail("max_pool2d_with_indices values/indices output shape mismatch");
  }
  if (out_shape.size() != 4) fail("max_pool2d_with_indices expects rank-4 output");
  int64_t OH = out_shape[2], OW = out_shape[3];

  auto ksz = parse_attr_pair(ref.attrs, "kernel_size", 2, bindings, "max_pool2d_with_indices");
  auto stride = parse_attr_pair(ref.attrs, "stride", ksz.first, bindings, "max_pool2d_with_indices");
  auto padding = parse_attr_pair(ref.attrs, "padding", 0, bindings, "max_pool2d_with_indices");
  auto dilation = parse_attr_pair(ref.attrs, "dilation", 1, bindings, "max_pool2d_with_indices");
  bool ceil_mode = ref.attrs.value("ceil_mode", false);
  if (ceil_mode) fail("max_pool2d_with_indices ceil_mode=true not supported");
  if (dilation.first != 1 || dilation.second != 1) fail("max_pool2d_with_indices dilation != 1 not supported");
  if (ksz.first <= 0 || ksz.second <= 0 || stride.first <= 0 || stride.second <= 0) fail("max_pool2d_with_indices invalid kernel/stride");
  int64_t expected_oh = ((H + 2LL * padding.first - ksz.first) / stride.first) + 1;
  int64_t expected_ow = ((W + 2LL * padding.second - ksz.second) / stride.second) + 1;
  if (expected_oh != OH || expected_ow != OW) fail("max_pool2d_with_indices output shape mismatch");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C * OH * OW;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  if (!out_val_name.empty()) w.line("float* __restrict__ " + out_val_name + ",");
  if (!out_idx_name.empty()) w.line("long long* __restrict__ " + out_idx_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t c = t % " + std::to_string(C) + "LL; t /= " + std::to_string(C) + "LL;");
  w.line("const int64_t n = t;");
  w.line("float maxv = -3.402823466e+38F;");
  w.line("int64_t maxidx = 0;");
  w.line("for (int kh = 0; kh < " + std::to_string(ksz.first) + "; ++kh) {");
  w.indent();
  w.line("const int64_t ih = oh * " + std::to_string(stride.first) + "LL - " + std::to_string(padding.first) + "LL + (int64_t)kh;");
  w.line("for (int kw = 0; kw < " + std::to_string(ksz.second) + "; ++kw) {");
  w.indent();
  w.line("const int64_t iw = ow * " + std::to_string(stride.second) + "LL - " + std::to_string(padding.second) + "LL + (int64_t)kw;");
  w.line("if ((unsigned long long)ih >= (unsigned long long)" + std::to_string(H) + "LL || (unsigned long long)iw >= (unsigned long long)" +
         std::to_string(W) + "LL) continue;");
  w.line("const int64_t idx = (((n * " + std::to_string(C) + "LL + c) * " + std::to_string(H) + "LL + ih) * " + std::to_string(W) + "LL + iw);");
  w.line("const float v = " + x_name + "[idx];");
  w.line("if (v > maxv) { maxv = v; maxidx = ih * " + std::to_string(W) + "LL + iw; }");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  if (!out_val_name.empty()) w.line(out_val_name + "[tid] = maxv;");
  if (!out_idx_name.empty()) w.line(out_idx_name + "[tid] = (long long)maxidx;");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {x_name};
  if (!out_val_name.empty()) tensor_args.push_back(out_val_name);
  if (!out_idx_name.empty()) tensor_args.push_back(out_idx_name);

  std::vector<std::string> output_names;
  if (!out_val_name.empty()) output_names.push_back(out_val_name);
  if (!out_idx_name.empty()) output_names.push_back(out_idx_name);

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, tensor_args);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = output_names;
  out["bindings"] = bindings;
  return out;
}

json emit_sort2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "sort")) fail("sort lowering expects single sort op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("sort expects one input");
  const std::string in_name = op.inputs[0];
  const std::string out_name = op.output;
  auto in_it = intent.tensors.find(in_name);
  auto out_it = intent.tensors.find(out_name);
  if (in_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("sort tensors missing");
  auto in_shape = resolve_shape_required(in_it->second, bindings, "sort.input.shape");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "sort.output.shape");
  if (in_it->second.dtype != "f32" || out_it->second.dtype != "f32") fail("sort lowering supports f32 only");
  if (in_shape.size() != 2 || out_shape.size() != 2 || in_shape != out_shape) fail("sort lowering expects rank-2 equal input/output shapes");
  int axis = -1;
  if (op.attrs.is_object() && op.attrs.contains("axis")) {
    axis = static_cast<int>(resolve_dim_token_required(op.attrs["axis"], bindings, "sort.axis"));
  }
  if (axis < 0) axis += 2;
  if (axis != 1) fail("sort lowering currently supports axis=1 only");
  bool descending = op.attrs.value("descending", false);
  int64_t M = in_shape[0];
  int64_t N = in_shape[1];
  if (M <= 0 || N <= 0) fail("sort lowering requires positive shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + in_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (i >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = i * " + std::to_string(N) + "LL;");
  w.line("for (int64_t j = 0; j < " + std::to_string(N) + "LL; ++j) " + out_name + "[base + j] = " + in_name + "[base + j];");
  w.line("for (int64_t ii = 1; ii < " + std::to_string(N) + "LL; ++ii) {");
  w.indent();
  w.line("float v = " + out_name + "[base + ii];");
  w.line("int64_t jj = ii - 1;");
  w.line("while (jj >= 0 && " + out_name + "[base + jj] > v) {");
  w.indent();
  w.line(out_name + "[base + jj + 1] = " + out_name + "[base + jj];");
  w.line("--jj;");
  w.dedent();
  w.line("}");
  w.line(out_name + "[base + jj + 1] = v;");
  w.dedent();
  w.line("}");
  if (descending) {
    w.line("for (int64_t j = 0; j < " + std::to_string(N / 2) + "LL; ++j) {");
    w.indent();
    w.line("float tmp = " + out_name + "[base + j];");
    w.line(out_name + "[base + j] = " + out_name + "[base + (" + std::to_string(N) + "LL - 1 - j)];");
    w.line(out_name + "[base + (" + std::to_string(N) + "LL - 1 - j)] = tmp;");
    w.dedent();
    w.line("}");
  }
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_unique1d_i32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "unique")) fail("unique lowering expects single unique op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("unique expects one input");
  const std::string in_name = op.inputs[0];
  const std::string out_name = op.output;
  auto in_it = intent.tensors.find(in_name);
  auto out_it = intent.tensors.find(out_name);
  if (in_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("unique tensors missing");
  auto in_shape = resolve_shape_required(in_it->second, bindings, "unique.input.shape");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "unique.output.shape");
  if (in_it->second.dtype != "i32" || out_it->second.dtype != "i32") fail("unique lowering supports i32 only");
  if (in_shape.size() != 1 || out_shape.size() != 1) fail("unique lowering expects rank-1 tensors");
  int64_t N = in_shape[0];
  int64_t U = out_shape[0];
  if (N <= 0 || U <= 0) fail("unique lowering requires positive shape");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(const int* __restrict__ " + in_name + ", int* __restrict__ " + out_name + ") {");
  w.indent();
  w.line("if (blockIdx.x != 0 || threadIdx.x != 0) return;");
  w.line("int64_t unique_count = 0;");
  w.line("for (int64_t i = 0; i < " + std::to_string(N) + "LL; ++i) {");
  w.indent();
  w.line("const int v = " + in_name + "[i];");
  w.line("bool seen = false;");
  w.line("for (int64_t j = 0; j < unique_count; ++j) {");
  w.indent();
  w.line("if (" + out_name + "[j] == v) { seen = true; break; }");
  w.dedent();
  w.line("}");
  w.line("if (!seen) {");
  w.indent();
  w.line("if (unique_count < " + std::to_string(U) + "LL) " + out_name + "[unique_count] = v;");
  w.line("++unique_count;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("for (int64_t i = unique_count; i < " + std::to_string(U) + "LL; ++i) " + out_name + "[i] = 0;");
  w.line("for (int64_t i = 1; i < " + std::to_string(U) + "LL; ++i) {");
  w.indent();
  w.line("int v = " + out_name + "[i];");
  w.line("int64_t j = i - 1;");
  w.line("while (j >= 0 && " + out_name + "[j] > v) {");
  w.indent();
  w.line(out_name + "[j + 1] = " + out_name + "[j];");
  w.line("--j;");
  w.dedent();
  w.line("}");
  w.line(out_name + "[j + 1] = v;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_kron2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "kron")) fail("kron lowering expects single kron op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("kron expects two inputs");
  const std::string a_name = op.inputs[0];
  const std::string b_name = op.inputs[1];
  const std::string out_name = op.output;
  auto a_it = intent.tensors.find(a_name);
  auto b_it = intent.tensors.find(b_name);
  auto out_it = intent.tensors.find(out_name);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("kron tensors missing");
  auto a_shape = resolve_shape_required(a_it->second, bindings, "kron.A.shape");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "kron.B.shape");
  auto o_shape = resolve_shape_required(out_it->second, bindings, "kron.out.shape");
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || out_it->second.dtype != "f32") fail("kron lowering supports f32 only");
  if (a_shape.size() != 2 || b_shape.size() != 2 || o_shape.size() != 2) fail("kron lowering expects rank-2 tensors");
  int64_t M = a_shape[0], N = a_shape[1], P = b_shape[0], Q = b_shape[1];
  if (o_shape[0] != M * P || o_shape[1] != N * Q) fail("kron output shape mismatch");
  int64_t MP = o_shape[0], NQ = o_shape[1];
  int64_t total = MP * NQ;
  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + a_name + ",");
  w.line("const float* __restrict__ " + b_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("const int64_t oi = tid / " + std::to_string(NQ) + "LL;");
  w.line("const int64_t oj = tid - oi * " + std::to_string(NQ) + "LL;");
  w.line("const int64_t i = oi / " + std::to_string(P) + "LL;");
  w.line("const int64_t p = oi - i * " + std::to_string(P) + "LL;");
  w.line("const int64_t j = oj / " + std::to_string(Q) + "LL;");
  w.line("const int64_t q = oj - j * " + std::to_string(Q) + "LL;");
  w.line("const float av = " + a_name + "[i * " + std::to_string(N) + "LL + j];");
  w.line("const float bv = " + b_name + "[p * " + std::to_string(Q) + "LL + q];");
  w.line(out_name + "[tid] = av * bv;");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {a_name, b_name, out_name}, {}, {a_name, b_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_isin1d_i32_bool(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 5) fail("isin lowering expects 5-op pattern");
  if (!(intent.ops[0].op == "broadcast_in_dim" && intent.ops[1].op == "broadcast_in_dim" && intent.ops[2].op == "ne" && intent.ops[3].op == "not" &&
        intent.ops[4].op == "reduce_any")) {
    fail("isin lowering expects broadcast+broadcast+ne+not+reduce_any");
  }
  const Op& b0 = intent.ops[0];
  const Op& b1 = intent.ops[1];
  const Op& red = intent.ops[4];
  if (b0.inputs.size() != 1 || b1.inputs.size() != 1 || red.inputs.size() != 1) fail("isin lowering invalid inputs");
  const std::string in0_name = b0.inputs[0];
  const std::string in1_name = b1.inputs[0];
  const std::string out_name = red.output;
  auto a_it = intent.tensors.find(in0_name);
  auto b_it = intent.tensors.find(in1_name);
  auto o_it = intent.tensors.find(out_name);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("isin tensors missing");
  auto a_shape = resolve_shape_required(a_it->second, bindings, "isin.in0.shape");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "isin.in1.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "isin.out.shape");
  if (a_it->second.dtype != "i32" || b_it->second.dtype != "i32") fail("isin lowering supports i32 inputs only");
  if (!(o_it->second.dtype == "bool" || o_it->second.dtype == "i1")) fail("isin lowering expects bool output");
  if (a_shape.size() != 1 || b_shape.size() != 1 || o_shape.size() != 1) fail("isin lowering expects rank-1 tensors");
  int64_t M = a_shape[0];
  int64_t K = b_shape[0];
  if (o_shape[0] != M) fail("isin output shape mismatch");
  if (M <= 0 || K <= 0) fail("isin lowering requires positive shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const int* __restrict__ " + in0_name + ",");
  w.line("const int* __restrict__ " + in1_name + ",");
  w.line("bool* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (i >= " + std::to_string(M) + "LL) return;");
  w.line("const int v = " + in0_name + "[i];");
  w.line("bool any = false;");
  w.line("for (int64_t k = 0; k < " + std::to_string(K) + "LL; ++k) { if (v == " + in1_name + "[k]) { any = true; break; } }");
  w.line(out_name + "[i] = any;");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {in0_name, in1_name, out_name}, {}, {in0_name, in1_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_upsample_nearest1d_ncl_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "upsample_nearest1d")) fail("upsample_nearest1d lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("upsample_nearest1d expects one input");
  const std::string x_name = op.inputs[0];
  const std::string out_name = op.output;
  auto x_it = intent.tensors.find(x_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("upsample_nearest1d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "upsample_nearest1d.input.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "upsample_nearest1d.output.shape");
  if (x_shape.size() != 3 || o_shape.size() != 3) fail("upsample_nearest1d expects rank-3 NCL tensors");
  if (x_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("upsample_nearest1d supports f32 only");
  int64_t N = x_shape[0], C = x_shape[1], IL = x_shape[2], OL = o_shape[2];
  if (IL <= 0 || OL <= 0) fail("upsample_nearest1d invalid shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C * OL;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ol = t % " + std::to_string(OL) + "LL; t /= " + std::to_string(OL) + "LL;");
  w.line("const int64_t c = t % " + std::to_string(C) + "LL; t /= " + std::to_string(C) + "LL;");
  w.line("const int64_t n = t;");
  w.line("int64_t il = (ol * " + std::to_string(IL) + "LL) / " + std::to_string(OL) + "LL;");
  w.line("if (il < 0) il = 0;");
  w.line("if (il >= " + std::to_string(IL) + "LL) il = " + std::to_string(IL - 1) + "LL;");
  w.line("const int64_t in_idx = ((n * " + std::to_string(C) + "LL + c) * " + std::to_string(IL) + "LL + il);");
  w.line(out_name + "[tid] = " + x_name + "[in_idx];");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {x_name, out_name}, {}, {x_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_upsample_nearest2d_nchw_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "upsample_nearest2d")) fail("upsample_nearest2d lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("upsample_nearest2d expects one input");
  const std::string x_name = op.inputs[0];
  const std::string out_name = op.output;
  auto x_it = intent.tensors.find(x_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("upsample_nearest2d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "upsample_nearest2d.input.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "upsample_nearest2d.output.shape");
  if (x_shape.size() != 4 || o_shape.size() != 4) fail("upsample_nearest2d expects rank-4 NCHW tensors");
  if (x_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("upsample_nearest2d supports f32 only");
  int64_t N = x_shape[0], C = x_shape[1], IH = x_shape[2], IW = x_shape[3];
  int64_t OH = o_shape[2], OW = o_shape[3];
  if (IH <= 0 || IW <= 0 || OH <= 0 || OW <= 0) fail("upsample_nearest2d invalid shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C * OH * OW;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t c = t % " + std::to_string(C) + "LL; t /= " + std::to_string(C) + "LL;");
  w.line("const int64_t n = t;");
  w.line("int64_t ih = (oh * " + std::to_string(IH) + "LL) / " + std::to_string(OH) + "LL;");
  w.line("int64_t iw = (ow * " + std::to_string(IW) + "LL) / " + std::to_string(OW) + "LL;");
  w.line("if (ih < 0) ih = 0;");
  w.line("if (iw < 0) iw = 0;");
  w.line("if (ih >= " + std::to_string(IH) + "LL) ih = " + std::to_string(IH - 1) + "LL;");
  w.line("if (iw >= " + std::to_string(IW) + "LL) iw = " + std::to_string(IW - 1) + "LL;");
  w.line("const int64_t in_idx = (((n * " + std::to_string(C) + "LL + c) * " + std::to_string(IH) + "LL + ih) * " + std::to_string(IW) + "LL + iw);");
  w.line(out_name + "[tid] = " + x_name + "[in_idx];");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {x_name, out_name}, {}, {x_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

