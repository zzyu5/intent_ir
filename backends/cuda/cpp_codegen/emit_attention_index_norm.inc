json emit_scaled_dot_product_attention_bhsd_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "scaled_dot_product_attention")) {
    fail("sdpa lowering expects single scaled_dot_product_attention op");
  }
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("scaled_dot_product_attention expects 3 inputs");
  const std::string query = op.inputs[0];
  const std::string key = op.inputs[1];
  const std::string value = op.inputs[2];
  const std::string out_name = op.output;
  bool is_causal = false;
  if (op.attrs.is_object() && op.attrs.contains("is_causal")) {
    const auto& v = op.attrs["is_causal"];
    if (v.is_boolean())
      is_causal = v.get<bool>();
    else if (v.is_number_integer())
      is_causal = (v.get<int64_t>() != 0);
    else if (v.is_number())
      is_causal = (v.get<double>() != 0.0);
  }

  auto q_it = intent.tensors.find(query);
  auto k_it = intent.tensors.find(key);
  auto v_it = intent.tensors.find(value);
  auto o_it = intent.tensors.find(out_name);
  if (q_it == intent.tensors.end() || k_it == intent.tensors.end() || v_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("sdpa lowering missing tensors");
  }
  if (q_it->second.dtype != "f32" || k_it->second.dtype != "f32" || v_it->second.dtype != "f32" || o_it->second.dtype != "f32") {
    fail("sdpa lowering supports f32 only");
  }
  auto q_shape = resolve_shape_required(q_it->second, bindings, "sdpa.query");
  auto k_shape = resolve_shape_required(k_it->second, bindings, "sdpa.key");
  auto v_shape = resolve_shape_required(v_it->second, bindings, "sdpa.value");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "sdpa.out");
  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || o_shape.size() != 4) fail("sdpa expects rank-4 [B,H,Q/K,D]");

  const int64_t B = q_shape[0];
  const int64_t H = q_shape[1];
  const int64_t Q = q_shape[2];
  const int64_t D = q_shape[3];
  const int64_t K = k_shape[2];
  if (k_shape[0] != B || k_shape[1] != H || k_shape[3] != D) fail("sdpa key shape mismatch");
  if (v_shape[0] != B || v_shape[1] != H || v_shape[2] != K || v_shape[3] != D) fail("sdpa value shape mismatch");
  if (o_shape[0] != B || o_shape[1] != H || o_shape[2] != Q || o_shape[3] != D) fail("sdpa output shape mismatch");
  if (B <= 0 || H <= 0 || Q <= 0 || K <= 0 || D <= 0) fail("sdpa invalid shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total = B * H * Q * D;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + query + ",");
  w.line("const float* __restrict__ " + key + ",");
  w.line("const float* __restrict__ " + value + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t d = t % " + std::to_string(D) + "LL; t /= " + std::to_string(D) + "LL;");
  w.line("const int64_t q = t % " + std::to_string(Q) + "LL; t /= " + std::to_string(Q) + "LL;");
  w.line("const int64_t h = t % " + std::to_string(H) + "LL; t /= " + std::to_string(H) + "LL;");
  w.line("const int64_t b = t;");
  w.line("const float scale = rsqrtf((float)" + std::to_string(D) + ");");
  w.line("const int64_t q_base = (((b * " + std::to_string(H) + "LL + h) * " + std::to_string(Q) + "LL + q) * " + std::to_string(D) + "LL);");
  w.line("float max_score = -3.402823466e+38f;");
  w.line("for (int64_t k_idx = 0; k_idx < " + std::to_string(K) + "LL; ++k_idx) {");
  w.indent();
  if (is_causal) w.line("if (k_idx > q) continue;");
  w.line("const int64_t k_base = (((b * " + std::to_string(H) + "LL + h) * " + std::to_string(K) + "LL + k_idx) * " + std::to_string(D) + "LL);");
  w.line("float score = 0.0f;");
  w.line("#pragma unroll");
  w.line("for (int64_t dd = 0; dd < " + std::to_string(D) + "LL; ++dd) score = fmaf(" + query + "[q_base + dd], " + key + "[k_base + dd], score);");
  w.line("score *= scale;");
  w.line("if (score > max_score) max_score = score;");
  w.dedent();
  w.line("}");
  w.line("float denom = 0.0f;");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t k_idx = 0; k_idx < " + std::to_string(K) + "LL; ++k_idx) {");
  w.indent();
  if (is_causal) w.line("if (k_idx > q) continue;");
  w.line("const int64_t k_base = (((b * " + std::to_string(H) + "LL + h) * " + std::to_string(K) + "LL + k_idx) * " + std::to_string(D) + "LL);");
  w.line("float score = 0.0f;");
  w.line("#pragma unroll");
  w.line("for (int64_t dd = 0; dd < " + std::to_string(D) + "LL; ++dd) score = fmaf(" + query + "[q_base + dd], " + key + "[k_base + dd], score);");
  w.line("score = score * scale - max_score;");
  w.line("const float wv = __expf(score);");
  w.line("denom += wv;");
  w.line("acc = fmaf(wv, " + value + "[k_base + d], acc);");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = (denom > 0.0f) ? (acc / denom) : 0.0f;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["B"] = B;
  out_bindings["H"] = H;
  out_bindings["Q"] = Q;
  out_bindings["K"] = K;
  out_bindings["D"] = D;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {query, key, value, out_name}, {}, {query, key, value, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_flash_attn_varlen_func_bhsd_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 6) fail("flash_attn_varlen_func lowering expects 6-op pattern");
  static const std::vector<std::string> kOps = {"transpose", "matmul", "mul", "add", "softmax", "matmul"};
  for (size_t i = 0; i < kOps.size(); ++i) {
    if (intent.ops[i].op != kOps[i]) fail("flash_attn_varlen_func pattern mismatch");
  }
  const Op& tr = intent.ops[0];
  const Op& mm_qk = intent.ops[1];
  const Op& mul = intent.ops[2];
  const Op& add = intent.ops[3];
  const Op& mm_out = intent.ops[5];
  if (tr.inputs.size() != 1 || mm_qk.inputs.size() != 2 || mul.inputs.size() != 2 || add.inputs.size() != 2 || mm_out.inputs.size() != 2) {
    fail("flash_attn_varlen_func arity mismatch");
  }
  const std::string Q = mm_qk.inputs[0];
  const std::string K = tr.inputs[0];
  const std::string V = mm_out.inputs[1];
  const std::string attn_mask = (add.inputs[0] == mul.output) ? add.inputs[1] : add.inputs[0];
  const std::string sm_scale = (mul.inputs[0] == mm_qk.output) ? mul.inputs[1] : mul.inputs[0];
  const std::string out_name = mm_out.output;

  auto q_it = intent.tensors.find(Q);
  auto k_it = intent.tensors.find(K);
  auto v_it = intent.tensors.find(V);
  auto m_it = intent.tensors.find(attn_mask);
  auto s_it = intent.tensors.find(sm_scale);
  auto o_it = intent.tensors.find(out_name);
  if (q_it == intent.tensors.end() || k_it == intent.tensors.end() || v_it == intent.tensors.end() || m_it == intent.tensors.end() ||
      s_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("flash_attn_varlen_func missing tensors");
  }
  if (q_it->second.dtype != "f32" || k_it->second.dtype != "f32" || v_it->second.dtype != "f32" || m_it->second.dtype != "f32" ||
      s_it->second.dtype != "f32" || o_it->second.dtype != "f32") {
    fail("flash_attn_varlen_func supports f32 only");
  }

  auto q_shape = resolve_shape_required(q_it->second, bindings, "flash_attn.Q");
  auto k_shape = resolve_shape_required(k_it->second, bindings, "flash_attn.K");
  auto v_shape = resolve_shape_required(v_it->second, bindings, "flash_attn.V");
  auto m_shape = resolve_shape_required(m_it->second, bindings, "flash_attn.mask");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "flash_attn.out");
  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || m_shape.size() != 4 || o_shape.size() != 4) {
    fail("flash_attn_varlen_func expects rank-4 tensors");
  }
  const int64_t B = q_shape[0];
  const int64_t H = q_shape[1];
  const int64_t QL = q_shape[2];
  const int64_t D = q_shape[3];
  const int64_t KL = k_shape[2];
  if (k_shape[0] != B || k_shape[1] != H || k_shape[3] != D) fail("flash_attn K shape mismatch");
  if (v_shape[0] != B || v_shape[1] != H || v_shape[2] != KL || v_shape[3] != D) fail("flash_attn V shape mismatch");
  if (m_shape[0] != B || m_shape[1] != H || m_shape[2] != QL || m_shape[3] != KL) fail("flash_attn mask shape mismatch");
  if (o_shape[0] != B || o_shape[1] != H || o_shape[2] != QL || o_shape[3] != D) fail("flash_attn out shape mismatch");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total = B * H * QL * D;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + Q + ",");
  w.line("const float* __restrict__ " + K + ",");
  w.line("const float* __restrict__ " + V + ",");
  w.line("const float* __restrict__ " + attn_mask + ",");
  w.line("const float* __restrict__ " + sm_scale + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t d = t % " + std::to_string(D) + "LL; t /= " + std::to_string(D) + "LL;");
  w.line("const int64_t q = t % " + std::to_string(QL) + "LL; t /= " + std::to_string(QL) + "LL;");
  w.line("const int64_t h = t % " + std::to_string(H) + "LL; t /= " + std::to_string(H) + "LL;");
  w.line("const int64_t b = t;");
  w.line("const float sm = " + sm_scale + " ? " + sm_scale + "[0] : 1.0f;");
  w.line("const int64_t q_base = (((b * " + std::to_string(H) + "LL + h) * " + std::to_string(QL) + "LL + q) * " + std::to_string(D) + "LL);");
  w.line("const int64_t mask_base = (((b * " + std::to_string(H) + "LL + h) * " + std::to_string(QL) + "LL + q) * " + std::to_string(KL) + "LL);");
  w.line("float max_score = -3.402823466e+38f;");
  w.line("for (int64_t k_idx = 0; k_idx < " + std::to_string(KL) + "LL; ++k_idx) {");
  w.indent();
  w.line("const int64_t k_base = (((b * " + std::to_string(H) + "LL + h) * " + std::to_string(KL) + "LL + k_idx) * " + std::to_string(D) + "LL);");
  w.line("float score = 0.0f;");
  w.line("#pragma unroll");
  w.line("for (int64_t dd = 0; dd < " + std::to_string(D) + "LL; ++dd) score = fmaf(" + Q + "[q_base + dd], " + K + "[k_base + dd], score);");
  w.line("score = score * sm + " + attn_mask + "[mask_base + k_idx];");
  w.line("if (score > max_score) max_score = score;");
  w.dedent();
  w.line("}");
  w.line("float denom = 0.0f;");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t k_idx = 0; k_idx < " + std::to_string(KL) + "LL; ++k_idx) {");
  w.indent();
  w.line("const int64_t k_base = (((b * " + std::to_string(H) + "LL + h) * " + std::to_string(KL) + "LL + k_idx) * " + std::to_string(D) + "LL);");
  w.line("float score = 0.0f;");
  w.line("#pragma unroll");
  w.line("for (int64_t dd = 0; dd < " + std::to_string(D) + "LL; ++dd) score = fmaf(" + Q + "[q_base + dd], " + K + "[k_base + dd], score);");
  w.line("score = score * sm + " + attn_mask + "[mask_base + k_idx] - max_score;");
  w.line("const float wv = __expf(score);");
  w.line("denom += wv;");
  w.line("acc = fmaf(wv, " + V + "[k_base + d], acc);");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = (denom > 0.0f) ? (acc / denom) : 0.0f;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["B"] = B;
  out_bindings["H"] = H;
  out_bindings["Q"] = QL;
  out_bindings["K"] = KL;
  out_bindings["D"] = D;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {Q, K, V, attn_mask, sm_scale, out_name}, {}, {Q, K, V, attn_mask, sm_scale, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_glu2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "glu")) fail("glu lowering expects single glu op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("glu expects one input");
  int axis = 1;
  if (op.attrs.is_object() && op.attrs.contains("axis")) axis = static_cast<int>(resolve_dim_token_required(op.attrs["axis"], bindings, "glu.axis"));
  if (axis != 1) fail("glu lowering currently supports axis=1 only");

  const std::string X = op.inputs[0];
  const std::string O = op.output;
  auto x_it = intent.tensors.find(X);
  auto o_it = intent.tensors.find(O);
  if (x_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("glu missing tensors");
  if (x_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("glu supports f32 only");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "glu.x");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "glu.out");
  if (x_shape.size() != 2 || o_shape.size() != 2) fail("glu expects rank-2");
  const int64_t M = x_shape[0];
  const int64_t N = x_shape[1];
  const int64_t N_HALF = o_shape[1];
  if (o_shape[0] != M || N != N_HALF * 2) fail("glu shape mismatch");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total = M * N_HALF;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + X + ",");
  w.line("float* __restrict__ " + O);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("const int64_t m = tid / " + std::to_string(N_HALF) + "LL;");
  w.line("const int64_t n = tid - m * " + std::to_string(N_HALF) + "LL;");
  w.line("const int64_t base = m * " + std::to_string(N) + "LL;");
  w.line("const float a = " + X + "[base + n];");
  w.line("const float b = " + X + "[base + n + " + std::to_string(N_HALF) + "LL];");
  w.line(O + "[tid] = a / (1.0f + expf(-b));");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["N_HALF"] = N_HALF;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {X, O}, {}, {X, O});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {O};
  out["bindings"] = out_bindings;
  return out;
}

json emit_group_norm_kernel_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() < 25 || intent.ops.size() > 26) fail("group_norm lowering expects 25/26-op expanded pattern");
  const std::string X = "X";
  const std::string W = "W";
  const std::string B = "B";
  const std::string Y = "Y";
  const std::string Mean = "Mean";
  const std::string Rstd = "Rstd";
  const std::string eps = "eps";
  auto x_it = intent.tensors.find(X);
  auto w_it = intent.tensors.find(W);
  auto b_it = intent.tensors.find(B);
  auto y_it = intent.tensors.find(Y);
  auto mean_it = intent.tensors.find(Mean);
  auto rstd_it = intent.tensors.find(Rstd);
  if (x_it == intent.tensors.end() || w_it == intent.tensors.end() || b_it == intent.tensors.end() || y_it == intent.tensors.end() ||
      mean_it == intent.tensors.end() || rstd_it == intent.tensors.end()) {
    fail("group_norm missing tensors");
  }
  for (const auto* t : {&x_it->second, &w_it->second, &b_it->second, &y_it->second, &mean_it->second, &rstd_it->second}) {
    if (t->dtype != "f32") fail("group_norm supports f32 only");
  }
  auto x_shape = resolve_shape_required(x_it->second, bindings, "group_norm.X");
  auto mean_shape = resolve_shape_required(mean_it->second, bindings, "group_norm.Mean");
  auto y_shape = resolve_shape_required(y_it->second, bindings, "group_norm.Y");
  if (x_shape.size() != 3 || y_shape.size() != 3 || mean_shape.size() != 2) fail("group_norm expects X/Y rank-3 and Mean rank-2");
  const int64_t N = x_shape[0];
  const int64_t C = x_shape[1];
  const int64_t HW = x_shape[2];
  const int64_t G = mean_shape[1];
  if (N <= 0 || C <= 0 || HW <= 0 || G <= 0 || (C % G) != 0) fail("group_norm invalid shape");
  const int64_t GROUP_SIZE = C / G;

  float eps_val = 1e-5f;
  for (const auto& node : intent.ops) {
    if (node.op != "const" || node.output != eps || !node.attrs.is_object() || !node.attrs.contains("value")) continue;
    const auto& v = node.attrs["value"];
    if (v.is_number()) eps_val = static_cast<float>(v.get<double>());
    else if (v.is_number_integer()) eps_val = static_cast<float>(v.get<int64_t>());
    break;
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total = N * G;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + X + ",");
  w.line("const float* __restrict__ " + W + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("float* __restrict__ " + Y + ",");
  w.line("float* __restrict__ " + Mean + ",");
  w.line("float* __restrict__ " + Rstd);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("const int64_t n = tid / " + std::to_string(G) + "LL;");
  w.line("const int64_t g = tid - n * " + std::to_string(G) + "LL;");
  w.line("const int64_t c0 = g * " + std::to_string(GROUP_SIZE) + "LL;");
  w.line("const float eps_v = " + std::to_string(eps_val) + "f;");
  w.line("const float denom = (float)(" + std::to_string(GROUP_SIZE) + " * " + std::to_string(HW) + ");");
  w.line("float sum = 0.0f;");
  w.line("for (int64_t gc = 0; gc < " + std::to_string(GROUP_SIZE) + "LL; ++gc) {");
  w.indent();
  w.line("const int64_t c = c0 + gc;");
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) sum += " + X + "[base + hw];");
  w.dedent();
  w.line("}");
  w.line("const float mean = sum / denom;");
  w.line(Mean + "[tid] = mean;");
  w.line("float var_sum = 0.0f;");
  w.line("for (int64_t gc = 0; gc < " + std::to_string(GROUP_SIZE) + "LL; ++gc) {");
  w.indent();
  w.line("const int64_t c = c0 + gc;");
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) {");
  w.indent();
  w.line("const float d = " + X + "[base + hw] - mean;");
  w.line("var_sum = fmaf(d, d, var_sum);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("const float rstd = rsqrtf(var_sum / denom + eps_v);");
  w.line(Rstd + "[tid] = rstd;");
  w.line("for (int64_t gc = 0; gc < " + std::to_string(GROUP_SIZE) + "LL; ++gc) {");
  w.indent();
  w.line("const int64_t c = c0 + gc;");
  w.line("const float wv = " + W + "[c];");
  w.line("const float bv = " + B + "[c];");
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) {");
  w.indent();
  w.line("const float centered = " + X + "[base + hw] - mean;");
  w.line(Y + "[base + hw] = centered * rstd * wv + bv;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  out_bindings["C"] = C;
  out_bindings["HW"] = HW;
  out_bindings["num_groups"] = G;
  out_bindings["group_size"] = GROUP_SIZE;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {X, W, B, Y, Mean, Rstd}, {}, {X, W, B, Y, Mean, Rstd});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {Y, Mean, Rstd};
  out["bindings"] = out_bindings;
  return out;
}

json emit_addcmul2d_f32(const Intent& intent, const json& bindings) {
  // Canonical pattern:
  //   mul(tensor1, tensor2) -> t1_t2
  //   mul(value, t1_t2) -> scaled
  //   add(inp, scaled) -> out
  if (intent.ops.size() != 3 || intent.ops[0].op != "mul" || intent.ops[1].op != "mul" || intent.ops[2].op != "add") {
    fail("addcmul lowering expects mul->mul->add pattern");
  }
  const Op& mul_t1_t2 = intent.ops[0];
  const Op& mul_scale = intent.ops[1];
  const Op& add_out = intent.ops[2];
  if (mul_t1_t2.inputs.size() != 2 || mul_scale.inputs.size() != 2 || add_out.inputs.size() != 2) {
    fail("addcmul lowering invalid op arity");
  }
  if (mul_scale.inputs[0] != mul_t1_t2.output && mul_scale.inputs[1] != mul_t1_t2.output) {
    fail("addcmul lowering expects second mul to consume first mul output");
  }
  const std::string t1_t2 = mul_t1_t2.output;
  const std::string value = (mul_scale.inputs[0] == t1_t2) ? mul_scale.inputs[1] : mul_scale.inputs[0];
  if (add_out.inputs[0] != mul_scale.output && add_out.inputs[1] != mul_scale.output) {
    fail("addcmul lowering expects add to consume scaled value");
  }
  const std::string inp = (add_out.inputs[0] == mul_scale.output) ? add_out.inputs[1] : add_out.inputs[0];
  const std::string out = add_out.output;

  const std::string tensor1 = mul_t1_t2.inputs[0];
  const std::string tensor2 = mul_t1_t2.inputs[1];

  auto t1_it = intent.tensors.find(tensor1);
  auto t2_it = intent.tensors.find(tensor2);
  auto val_it = intent.tensors.find(value);
  auto inp_it = intent.tensors.find(inp);
  auto out_it = intent.tensors.find(out);
  if (t1_it == intent.tensors.end() || t2_it == intent.tensors.end() || val_it == intent.tensors.end() || inp_it == intent.tensors.end() ||
      out_it == intent.tensors.end()) {
    fail("addcmul lowering missing tensors");
  }
  if (t1_it->second.dtype != "f32" || t2_it->second.dtype != "f32" || val_it->second.dtype != "f32" || inp_it->second.dtype != "f32" ||
      out_it->second.dtype != "f32") {
    fail("addcmul lowering supports f32 only");
  }

  auto t1_shape = resolve_shape_required(t1_it->second, bindings, "addcmul.tensor1");
  auto t2_shape = resolve_shape_required(t2_it->second, bindings, "addcmul.tensor2");
  auto inp_shape = resolve_shape_required(inp_it->second, bindings, "addcmul.inp");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "addcmul.out");
  auto val_shape = resolve_shape_required(val_it->second, bindings, "addcmul.value");
  if (t1_shape.size() != 2 || t2_shape.size() != 2 || inp_shape.size() != 2 || out_shape.size() != 2) {
    fail("addcmul lowering expects rank-2 tensors");
  }
  if (t1_shape != t2_shape || t1_shape != inp_shape || t1_shape != out_shape) {
    fail("addcmul lowering shape mismatch");
  }
  if (!val_shape.empty()) fail("addcmul lowering expects scalar value tensor");

  const int64_t M = t1_shape[0];
  const int64_t N = t1_shape[1];
  if (M <= 0 || N <= 0) fail("addcmul lowering requires positive shape");
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + tensor1 + ",");
  w.line("const float* __restrict__ " + tensor2 + ",");
  w.line("const float* __restrict__ " + value + ",");
  w.line("const float* __restrict__ " + inp + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("const float vv = " + value + " ? " + value + "[0] : 1.0f;");
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float t12 = " + tensor1 + "[(size_t)i] * " + tensor2 + "[(size_t)i];");
  w.line(out + "[(size_t)i] = fmaf(vv, t12, " + inp + "[(size_t)i]);");
  w.dedent();
  w.line("}");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float t12 = " + tensor1 + "[(size_t)(i + 1)] * " + tensor2 + "[(size_t)(i + 1)];");
  w.line(out + "[(size_t)(i + 1)] = fmaf(vv, t12, " + inp + "[(size_t)(i + 1)]);");
  w.dedent();
  w.line("}");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float t12 = " + tensor1 + "[(size_t)(i + 2)] * " + tensor2 + "[(size_t)(i + 2)];");
  w.line(out + "[(size_t)(i + 2)] = fmaf(vv, t12, " + inp + "[(size_t)(i + 2)]);");
  w.dedent();
  w.line("}");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float t12 = " + tensor1 + "[(size_t)(i + 3)] * " + tensor2 + "[(size_t)(i + 3)];");
  w.line(out + "[(size_t)(i + 3)] = fmaf(vv, t12, " + inp + "[(size_t)(i + 3)]);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {tensor1, tensor2, value, inp, out}, {}, {tensor1, tensor2, value, inp, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_addcdiv2d_f32(const Intent& intent, const json& bindings) {
  // Canonical pattern:
  //   div(t1, t2) -> div_t1_t2
  //   mul(value, div_t1_t2) -> scaled
  //   add(x, scaled) -> out
  if (intent.ops.size() != 3 || intent.ops[0].op != "div" || intent.ops[1].op != "mul" || intent.ops[2].op != "add") {
    fail("addcdiv lowering expects div->mul->add pattern");
  }
  const Op& div_t1_t2 = intent.ops[0];
  const Op& mul_scale = intent.ops[1];
  const Op& add_out = intent.ops[2];
  if (div_t1_t2.inputs.size() != 2 || mul_scale.inputs.size() != 2 || add_out.inputs.size() != 2) {
    fail("addcdiv lowering invalid op arity");
  }
  if (mul_scale.inputs[0] != div_t1_t2.output && mul_scale.inputs[1] != div_t1_t2.output) {
    fail("addcdiv lowering expects mul to consume div output");
  }
  const std::string div_out = div_t1_t2.output;
  const std::string value = (mul_scale.inputs[0] == div_out) ? mul_scale.inputs[1] : mul_scale.inputs[0];
  if (add_out.inputs[0] != mul_scale.output && add_out.inputs[1] != mul_scale.output) {
    fail("addcdiv lowering expects add to consume scaled value");
  }
  const std::string x = (add_out.inputs[0] == mul_scale.output) ? add_out.inputs[1] : add_out.inputs[0];
  const std::string out = add_out.output;

  const std::string t1 = div_t1_t2.inputs[0];
  const std::string t2 = div_t1_t2.inputs[1];

  auto t1_it = intent.tensors.find(t1);
  auto t2_it = intent.tensors.find(t2);
  auto val_it = intent.tensors.find(value);
  auto x_it = intent.tensors.find(x);
  auto out_it = intent.tensors.find(out);
  if (t1_it == intent.tensors.end() || t2_it == intent.tensors.end() || val_it == intent.tensors.end() || x_it == intent.tensors.end() ||
      out_it == intent.tensors.end()) {
    fail("addcdiv lowering missing tensors");
  }
  if (t1_it->second.dtype != "f32" || t2_it->second.dtype != "f32" || val_it->second.dtype != "f32" || x_it->second.dtype != "f32" ||
      out_it->second.dtype != "f32") {
    fail("addcdiv lowering supports f32 only");
  }

  auto t1_shape = resolve_shape_required(t1_it->second, bindings, "addcdiv.t1");
  auto t2_shape = resolve_shape_required(t2_it->second, bindings, "addcdiv.t2");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "addcdiv.x");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "addcdiv.out");
  auto val_shape = resolve_shape_required(val_it->second, bindings, "addcdiv.value");
  if (t1_shape.size() != 2 || t2_shape.size() != 2 || x_shape.size() != 2 || out_shape.size() != 2) {
    fail("addcdiv lowering expects rank-2 tensors");
  }
  if (t1_shape != t2_shape || t1_shape != x_shape || t1_shape != out_shape) {
    fail("addcdiv lowering shape mismatch");
  }
  if (!val_shape.empty()) fail("addcdiv lowering expects scalar value tensor");

  const int64_t M = t1_shape[0];
  const int64_t N = t1_shape[1];
  if (M <= 0 || N <= 0) fail("addcdiv lowering requires positive shape");
  const int64_t total = M * N;

  constexpr int64_t kElemsPerThread = 4;
  int64_t block_x = choose_1d_block_threads(intent, bindings, total, /*default_threads=*/128, kElemsPerThread);
  const int64_t grid_x = (total + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + t1 + ",");
  w.line("const float* __restrict__ " + t2 + ",");
  w.line("const float* __restrict__ " + value + ",");
  w.line("const float* __restrict__ " + x + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("const float vv = " + value + " ? " + value + "[0] : 1.0f;");
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float q = __fdividef(" + t1 + "[(size_t)i], " + t2 + "[(size_t)i]);");
  w.line(out + "[(size_t)i] = fmaf(vv, q, " + x + "[(size_t)i]);");
  w.dedent();
  w.line("}");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float q = __fdividef(" + t1 + "[(size_t)(i + 1)], " + t2 + "[(size_t)(i + 1)]);");
  w.line(out + "[(size_t)(i + 1)] = fmaf(vv, q, " + x + "[(size_t)(i + 1)]);");
  w.dedent();
  w.line("}");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float q = __fdividef(" + t1 + "[(size_t)(i + 2)], " + t2 + "[(size_t)(i + 2)]);");
  w.line(out + "[(size_t)(i + 2)] = fmaf(vv, q, " + x + "[(size_t)(i + 2)]);");
  w.dedent();
  w.line("}");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float q = __fdividef(" + t1 + "[(size_t)(i + 3)], " + t2 + "[(size_t)(i + 3)]);");
  w.line(out + "[(size_t)(i + 3)] = fmaf(vv, q, " + x + "[(size_t)(i + 3)]);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {t1, t2, value, x, out}, {}, {t1, t2, value, x, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_abs2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "abs")) fail("abs lowering expects single abs op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("abs lowering expects one input");
  const std::string inp = op.inputs[0];
  const std::string out = op.output;

  auto inp_it = intent.tensors.find(inp);
  auto out_it = intent.tensors.find(out);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("abs lowering missing tensors");
  if (inp_it->second.dtype != "f32" || out_it->second.dtype != "f32") fail("abs lowering supports f32 only");

  auto inp_shape = resolve_shape_required(inp_it->second, bindings, "abs.inp");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "abs.out");
  if (inp_shape.size() != 2 || out_shape.size() != 2 || inp_shape != out_shape) fail("abs lowering expects equal rank-2 input/output");

  const int64_t M = inp_shape[0];
  const int64_t N = inp_shape[1];
  if (M <= 0 || N <= 0) fail("abs lowering requires positive shape");
  const int64_t total = M * N;

  constexpr int64_t kElemsPerThread = 4;
  int64_t block_x = choose_1d_block_threads(intent, bindings, total, /*default_threads=*/128, kElemsPerThread);
  const int64_t grid_x = (total + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) " + out + "[(size_t)i] = fabsf(" + inp + "[(size_t)i]);");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 1)] = fabsf(" + inp + "[(size_t)(i + 1)]);");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 2)] = fabsf(" + inp + "[(size_t)(i + 2)]);");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 3)] = fabsf(" + inp + "[(size_t)(i + 3)]);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {inp, out}, {}, {inp, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_acos2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "acos")) fail("acos lowering expects single acos op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("acos lowering expects one input");
  const std::string inp = op.inputs[0];
  const std::string out = op.output;

  auto inp_it = intent.tensors.find(inp);
  auto out_it = intent.tensors.find(out);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("acos lowering missing tensors");
  if (inp_it->second.dtype != "f32" || out_it->second.dtype != "f32") fail("acos lowering supports f32 only");

  auto inp_shape = resolve_shape_required(inp_it->second, bindings, "acos.inp");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "acos.out");
  if (inp_shape.size() != 2 || out_shape.size() != 2 || inp_shape != out_shape) fail("acos lowering expects equal rank-2 input/output");

  const int64_t M = inp_shape[0];
  const int64_t N = inp_shape[1];
  if (M <= 0 || N <= 0) fail("acos lowering requires positive shape");
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  constexpr int64_t kElemsPerThread = 4;
  const int64_t grid_x = (total + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) " + out + "[(size_t)i] = acosf(" + inp + "[(size_t)i]);");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 1)] = acosf(" + inp + "[(size_t)(i + 1)]);");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 2)] = acosf(" + inp + "[(size_t)(i + 2)]);");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 3)] = acosf(" + inp + "[(size_t)(i + 3)]);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {inp, out}, {}, {inp, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_atan2d_f32(const Intent& intent, const json& bindings) {
  // Canonical pattern:
  //   div(y, x) -> q
  //   atan(q) -> out
  // Lower directly to atan2f(y, x) to avoid explicit divide.
  if (!(intent.ops.size() == 2 && intent.ops[0].op == "div" && intent.ops[1].op == "atan")) {
    fail("atan2 lowering expects div->atan pattern");
  }
  const Op& div_op = intent.ops[0];
  const Op& atan_op = intent.ops[1];
  if (div_op.inputs.size() != 2 || atan_op.inputs.size() != 1 || atan_op.inputs[0] != div_op.output) {
    fail("atan2 lowering invalid wiring");
  }
  const std::string y = div_op.inputs[0];
  const std::string x = div_op.inputs[1];
  const std::string out = atan_op.output;

  auto y_it = intent.tensors.find(y);
  auto x_it = intent.tensors.find(x);
  auto out_it = intent.tensors.find(out);
  if (y_it == intent.tensors.end() || x_it == intent.tensors.end() || out_it == intent.tensors.end()) {
    fail("atan2 lowering missing tensors");
  }
  if (y_it->second.dtype != "f32" || x_it->second.dtype != "f32" || out_it->second.dtype != "f32") {
    fail("atan2 lowering supports f32 only");
  }

  auto y_shape = resolve_shape_required(y_it->second, bindings, "atan2.y");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "atan2.x");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "atan2.out");
  if (y_shape.size() != 2 || x_shape.size() != 2 || out_shape.size() != 2) {
    fail("atan2 lowering expects rank-2 tensors");
  }
  if (y_shape != x_shape || y_shape != out_shape) fail("atan2 lowering shape mismatch");
  const int64_t M = y_shape[0];
  const int64_t N = y_shape[1];
  if (M <= 0 || N <= 0) fail("atan2 lowering requires positive shape");
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  constexpr int64_t kElemsPerThread = 4;
  const int64_t grid_x = (total + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + y + ",");
  w.line("const float* __restrict__ " + x + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) " + out + "[(size_t)i] = atan2f(" + y + "[(size_t)i], " + x + "[(size_t)i]);");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 1)] = atan2f(" + y + "[(size_t)(i + 1)], " + x + "[(size_t)(i + 1)]);");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 2)] = atan2f(" + y + "[(size_t)(i + 2)], " + x + "[(size_t)(i + 2)]);");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 3)] = atan2f(" + y + "[(size_t)(i + 3)], " + x + "[(size_t)(i + 3)]);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {y, x, out}, {}, {y, x, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_pow_scalar2d_f32(const Intent& intent, const json& bindings) {
  // Canonical pattern:
  //   broadcast_in_dim(exponent_scalar) -> exp_bcast
  //   pow(inp, exp_bcast) -> out
  if (!(intent.ops.size() == 2 && intent.ops[0].op == "broadcast_in_dim" && intent.ops[1].op == "pow")) {
    fail("pow_scalar lowering expects broadcast_in_dim->pow pattern");
  }
  const Op& bcast = intent.ops[0];
  const Op& powop = intent.ops[1];
  if (bcast.inputs.size() != 1 || powop.inputs.size() != 2) fail("pow_scalar lowering invalid arity");
  const std::string exp_scalar = bcast.inputs[0];
  const std::string exp_bcast = bcast.output;
  if (powop.inputs[0] != exp_bcast && powop.inputs[1] != exp_bcast) {
    fail("pow_scalar lowering expects pow to consume broadcasted exponent");
  }
  const std::string inp = (powop.inputs[0] == exp_bcast) ? powop.inputs[1] : powop.inputs[0];
  const std::string out = powop.output;

  auto inp_it = intent.tensors.find(inp);
  auto exp_it = intent.tensors.find(exp_scalar);
  auto out_it = intent.tensors.find(out);
  if (inp_it == intent.tensors.end() || exp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("pow_scalar lowering missing tensors");
  if (inp_it->second.dtype != "f32" || exp_it->second.dtype != "f32" || out_it->second.dtype != "f32") {
    fail("pow_scalar lowering supports f32 only");
  }
  auto inp_shape = resolve_shape_required(inp_it->second, bindings, "pow_scalar.inp");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "pow_scalar.out");
  auto exp_shape = resolve_shape_required(exp_it->second, bindings, "pow_scalar.exp");
  if (!exp_shape.empty()) fail("pow_scalar lowering expects scalar exponent tensor");
  if (inp_shape.size() != 2 || out_shape.size() != 2 || inp_shape != out_shape) fail("pow_scalar lowering expects equal rank-2 input/output");

  const int64_t M = inp_shape[0];
  const int64_t N = inp_shape[1];
  if (M <= 0 || N <= 0) fail("pow_scalar lowering requires positive shape");
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  constexpr int64_t kElemsPerThread = 4;
  const int64_t grid_x = (total + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + exp_scalar + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("const float e = " + exp_scalar + " ? " + exp_scalar + "[0] : 1.0f;");
  w.line("if (fabsf(e - 2.0f) <= 1e-7f) {");
  w.indent();
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) { const float x = " + inp + "[(size_t)i]; " + out + "[(size_t)i] = x * x; }");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) { const float x = " + inp + "[(size_t)(i + 1)]; " + out + "[(size_t)(i + 1)] = x * x; }");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) { const float x = " + inp + "[(size_t)(i + 2)]; " + out + "[(size_t)(i + 2)] = x * x; }");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) { const float x = " + inp + "[(size_t)(i + 3)]; " + out + "[(size_t)(i + 3)] = x * x; }");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("} else if (fabsf(e - 1.0f) <= 1e-7f) {");
  w.indent();
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) " + out + "[(size_t)i] = " + inp + "[(size_t)i];");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 1)] = " + inp + "[(size_t)(i + 1)];");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 2)] = " + inp + "[(size_t)(i + 2)];");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 3)] = " + inp + "[(size_t)(i + 3)];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("} else if (fabsf(e - 0.5f) <= 1e-7f) {");
  w.indent();
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) " + out + "[(size_t)i] = sqrtf(" + inp + "[(size_t)i]);");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 1)] = sqrtf(" + inp + "[(size_t)(i + 1)]);");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 2)] = sqrtf(" + inp + "[(size_t)(i + 2)]);");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 3)] = sqrtf(" + inp + "[(size_t)(i + 3)]);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("} else if (fabsf(e) <= 1e-7f) {");
  w.indent();
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) " + out + "[(size_t)i] = 1.0f;");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 1)] = 1.0f;");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 2)] = 1.0f;");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 3)] = 1.0f;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("} else {");
  w.indent();
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) " + out + "[(size_t)i] = powf(" + inp + "[(size_t)i], e);");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 1)] = powf(" + inp + "[(size_t)(i + 1)], e);");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 2)] = powf(" + inp + "[(size_t)(i + 2)], e);");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) " + out + "[(size_t)(i + 3)] = powf(" + inp + "[(size_t)(i + 3)], e);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {inp, exp_scalar, out}, {}, {inp, exp_scalar, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_linspace1d_f32(const Intent& intent, const json& bindings) {
  // Canonical pattern:
  // iota -> cast -> sub(end,start) -> div(delta,denom) -> mul(idx_f,step) -> add(start,scaled)
  if (!(intent.ops.size() == 6 && intent.ops[0].op == "iota" && intent.ops[1].op == "cast" && intent.ops[2].op == "sub" &&
        intent.ops[3].op == "div" && intent.ops[4].op == "mul" && intent.ops[5].op == "add")) {
    fail("linspace lowering expects iota->cast->sub->div->mul->add pattern");
  }
  const Op& iota = intent.ops[0];
  const Op& cast = intent.ops[1];
  const Op& sub = intent.ops[2];
  const Op& div = intent.ops[3];
  const Op& mul = intent.ops[4];
  const Op& add = intent.ops[5];

  if (!iota.inputs.empty()) fail("linspace lowering expects iota with no inputs");
  if (!(cast.inputs.size() == 1 && cast.inputs[0] == iota.output)) fail("linspace lowering cast wiring mismatch");
  if (sub.inputs.size() != 2) fail("linspace lowering sub arity mismatch");
  if (!(div.inputs.size() == 2 && div.inputs[0] == sub.output)) fail("linspace lowering div wiring mismatch");
  if (mul.inputs.size() != 2) fail("linspace lowering mul arity mismatch");
  if (add.inputs.size() != 2) fail("linspace lowering add arity mismatch");

  const std::string idx_f = cast.output;
  const bool mul_uses_cast_lhs = (mul.inputs[0] == idx_f);
  const bool mul_uses_cast_rhs = (mul.inputs[1] == idx_f);
  if (!(mul_uses_cast_lhs || mul_uses_cast_rhs)) fail("linspace lowering mul expects cast(iota)");
  const std::string step = div.output;
  if (!((mul_uses_cast_lhs && mul.inputs[1] == step) || (mul_uses_cast_rhs && mul.inputs[0] == step))) {
    fail("linspace lowering mul expects div output");
  }
  const std::string scaled = mul.output;

  const std::string start = sub.inputs[1];
  const std::string end = sub.inputs[0];
  const std::string denom = div.inputs[1];
  if (!((add.inputs[0] == start && add.inputs[1] == scaled) || (add.inputs[1] == start && add.inputs[0] == scaled))) {
    fail("linspace lowering add expects start + scaled");
  }
  const std::string out = add.output;

  auto start_it = intent.tensors.find(start);
  auto end_it = intent.tensors.find(end);
  auto denom_it = intent.tensors.find(denom);
  auto out_it = intent.tensors.find(out);
  if (start_it == intent.tensors.end() || end_it == intent.tensors.end() || denom_it == intent.tensors.end() || out_it == intent.tensors.end()) {
    fail("linspace lowering missing tensors");
  }
  if (start_it->second.dtype != "f32" || end_it->second.dtype != "f32" || denom_it->second.dtype != "f32" || out_it->second.dtype != "f32") {
    fail("linspace lowering supports f32 only");
  }
  auto start_shape = resolve_shape_required(start_it->second, bindings, "linspace.start");
  auto end_shape = resolve_shape_required(end_it->second, bindings, "linspace.end");
  auto denom_shape = resolve_shape_required(denom_it->second, bindings, "linspace.denom");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "linspace.out");
  if (!start_shape.empty() || !end_shape.empty() || !denom_shape.empty()) fail("linspace lowering expects scalar start/end/denom");
  if (out_shape.size() != 1) fail("linspace lowering expects rank-1 output");

  const int64_t N = out_shape[0];
  if (N <= 0) fail("linspace lowering requires positive output length");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  constexpr int64_t kElemsPerThread = 4;
  const int64_t grid_x = (N + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + start + ",");
  w.line("const float* __restrict__ " + end + ",");
  w.line("const float* __restrict__ " + denom + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const float s = " + start + " ? " + start + "[0] : 0.0f;");
  w.line("const float e = " + end + " ? " + end + "[0] : 0.0f;");
  w.line("const float d = " + denom + " ? " + denom + "[0] : 1.0f;");
  w.line("const float stepv = (e - s) / d;");
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("for (int64_t i = base; i < " + std::to_string(N) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(N) + "LL) " + out + "[(size_t)i] = fmaf((float)i, stepv, s);");
  w.line("if (i + 1 < " + std::to_string(N) + "LL) " + out + "[(size_t)(i + 1)] = fmaf((float)(i + 1), stepv, s);");
  w.line("if (i + 2 < " + std::to_string(N) + "LL) " + out + "[(size_t)(i + 2)] = fmaf((float)(i + 2), stepv, s);");
  w.line("if (i + 3 < " + std::to_string(N) + "LL) " + out + "[(size_t)(i + 3)] = fmaf((float)(i + 3), stepv, s);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {start, end, denom, out}, {}, {start, end, denom, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_cmp2d_f32_bool(const Intent& intent, const json& bindings) {
  auto is_cmp = [](const std::string& op) {
    return op == "gt" || op == "ge" || op == "lt" || op == "le" || op == "eq" || op == "ne";
  };

  const Op* cmp = nullptr;
  std::string lhs;
  std::string rhs;
  std::string out;

  if (intent.ops.size() == 1 && is_cmp(intent.ops[0].op)) {
    cmp = &intent.ops[0];
    if (cmp->inputs.size() != 2) fail("cmp lowering expects binary compare");
    lhs = cmp->inputs[0];
    rhs = cmp->inputs[1];
    out = cmp->output;
  } else if (intent.ops.size() == 3 && intent.ops[0].op == "cast" && intent.ops[1].op == "cast" && is_cmp(intent.ops[2].op)) {
    const Op& c0 = intent.ops[0];
    const Op& c1 = intent.ops[1];
    cmp = &intent.ops[2];
    if (!(c0.inputs.size() == 1 && c1.inputs.size() == 1 && cmp->inputs.size() == 2)) fail("cmp cast lowering arity mismatch");
    if (!(cmp->inputs[0] == c0.output && cmp->inputs[1] == c1.output)) fail("cmp cast lowering wiring mismatch");
    lhs = c0.inputs[0];
    rhs = c1.inputs[0];
    out = cmp->output;
  } else {
    fail("cmp lowering expects compare-only or cast-cast-compare pattern");
  }

  auto lhs_it = intent.tensors.find(lhs);
  auto rhs_it = intent.tensors.find(rhs);
  auto out_it = intent.tensors.find(out);
  if (lhs_it == intent.tensors.end() || rhs_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("cmp lowering missing tensors");
  const std::string out_dtype = out_it->second.dtype;
  if (lhs_it->second.dtype != "f32" || rhs_it->second.dtype != "f32" || !(out_dtype == "bool" || out_dtype == "i32")) {
    fail("cmp lowering supports f32,f32->(bool|i32) only");
  }
  auto lhs_shape = resolve_shape_required(lhs_it->second, bindings, "cmp.lhs");
  auto rhs_shape = resolve_shape_required(rhs_it->second, bindings, "cmp.rhs");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "cmp.out");
  if (lhs_shape.size() != 2 || rhs_shape.size() != 2 || out_shape.size() != 2) fail("cmp lowering expects rank-2 tensors");
  if (lhs_shape != rhs_shape || lhs_shape != out_shape) fail("cmp lowering expects equal rank-2 shapes");

  const int64_t M = out_shape[0];
  const int64_t N = out_shape[1];
  if (M <= 0 || N <= 0) fail("cmp lowering requires positive shape");
  const int64_t total = M * N;
  const bool same_input = (lhs == rhs);
  const std::string rhs_ref = same_input ? lhs : rhs;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  constexpr int64_t kElemsPerThread = 4;
  const int64_t grid_x = (total + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::string cmp_expr;
  if (cmp->op == "gt") cmp_expr = "(a > b)";
  else if (cmp->op == "ge") cmp_expr = "(a >= b)";
  else if (cmp->op == "lt") cmp_expr = "(a < b)";
  else if (cmp->op == "le") cmp_expr = "(a <= b)";
  else if (cmp->op == "eq") cmp_expr = "(a == b)";
  else if (cmp->op == "ne") cmp_expr = "(a != b)";
  else fail("cmp lowering unsupported compare op");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + lhs + ",");
  if (!same_input) w.line("const float* __restrict__ " + rhs + ",");
  w.line(c_type_for_dtype(out_dtype) + "* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  const std::string cast_prefix = (out_dtype == "i32") ? "(int)" : "";
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) { const float a = " + lhs + "[(size_t)i]; const float b = " + rhs_ref + "[(size_t)i]; " + out +
         "[(size_t)i] = " + cast_prefix + cmp_expr + "; }");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) { const float a = " + lhs + "[(size_t)(i + 1)]; const float b = " + rhs_ref +
         "[(size_t)(i + 1)]; " + out + "[(size_t)(i + 1)] = " + cast_prefix + cmp_expr + "; }");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) { const float a = " + lhs + "[(size_t)(i + 2)]; const float b = " + rhs_ref +
         "[(size_t)(i + 2)]; " + out + "[(size_t)(i + 2)] = " + cast_prefix + cmp_expr + "; }");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) { const float a = " + lhs + "[(size_t)(i + 3)]; const float b = " + rhs_ref +
         "[(size_t)(i + 3)]; " + out + "[(size_t)(i + 3)] = " + cast_prefix + cmp_expr + "; }");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  std::vector<std::string> tensor_args = {lhs};
  std::vector<std::string> arg_names = {lhs};
  if (!same_input) {
    tensor_args.push_back(rhs);
    arg_names.push_back(rhs);
  }
  tensor_args.push_back(out);
  arg_names.push_back(out);
  out_json["io_spec"] = io_spec_from_args(intent, tensor_args, {}, arg_names);
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_isclose2d_f32_bool(const Intent& intent, const json& bindings) {
  // Canonical pattern:
  // sub(A,B) -> abs -> abs(B) -> mul(rtol,abs_b) -> add(atol,rtol_term) -> le(abs_diff,tol)
  if (!(intent.ops.size() == 6 && intent.ops[0].op == "sub" && intent.ops[1].op == "abs" && intent.ops[2].op == "abs" &&
        intent.ops[3].op == "mul" && intent.ops[4].op == "add" && intent.ops[5].op == "le")) {
    fail("isclose lowering expects sub->abs->abs->mul->add->le pattern");
  }
  const Op& sub = intent.ops[0];
  const Op& abs_diff = intent.ops[1];
  const Op& abs_b = intent.ops[2];
  const Op& mul = intent.ops[3];
  const Op& add = intent.ops[4];
  const Op& le = intent.ops[5];

  if (sub.inputs.size() != 2 || abs_diff.inputs.size() != 1 || abs_b.inputs.size() != 1 || mul.inputs.size() != 2 || add.inputs.size() != 2 || le.inputs.size() != 2) {
    fail("isclose lowering arity mismatch");
  }
  if (abs_diff.inputs[0] != sub.output) fail("isclose lowering abs(diff) wiring mismatch");

  const std::string A = sub.inputs[0];
  const std::string B = sub.inputs[1];
  if (abs_b.inputs[0] != B) fail("isclose lowering abs(B) wiring mismatch");

  const std::string rtol = (mul.inputs[0] == abs_b.output) ? mul.inputs[1] : ((mul.inputs[1] == abs_b.output) ? mul.inputs[0] : "");
  if (rtol.empty()) fail("isclose lowering mul expects abs(B) * rtol");
  if (!((add.inputs[0] == mul.output) || (add.inputs[1] == mul.output))) fail("isclose lowering add expects mul output");
  const std::string atol = (add.inputs[0] == mul.output) ? add.inputs[1] : add.inputs[0];
  if (!(le.inputs[0] == abs_diff.output && le.inputs[1] == add.output)) fail("isclose lowering final le wiring mismatch");
  const std::string out = le.output;

  auto a_it = intent.tensors.find(A);
  auto b_it = intent.tensors.find(B);
  auto rtol_it = intent.tensors.find(rtol);
  auto atol_it = intent.tensors.find(atol);
  auto out_it = intent.tensors.find(out);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || rtol_it == intent.tensors.end() || atol_it == intent.tensors.end() ||
      out_it == intent.tensors.end()) {
    fail("isclose lowering missing tensors");
  }
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || rtol_it->second.dtype != "f32" || atol_it->second.dtype != "f32" ||
      out_it->second.dtype != "bool") {
    fail("isclose lowering supports f32,f32,f32,f32->bool");
  }
  auto a_shape = resolve_shape_required(a_it->second, bindings, "isclose.A");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "isclose.B");
  auto rtol_shape = resolve_shape_required(rtol_it->second, bindings, "isclose.rtol");
  auto atol_shape = resolve_shape_required(atol_it->second, bindings, "isclose.atol");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "isclose.out");
  if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 2 || a_shape != b_shape || a_shape != out_shape) {
    fail("isclose lowering expects equal rank-2 A/B/out shapes");
  }
  if (!rtol_shape.empty() || !atol_shape.empty()) fail("isclose lowering expects scalar rtol/atol");

  const int64_t M = a_shape[0];
  const int64_t N = a_shape[1];
  if (M <= 0 || N <= 0) fail("isclose lowering requires positive shape");
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  constexpr int64_t kElemsPerThread = 4;
  const int64_t grid_x = (total + block_x * kElemsPerThread - 1) / (block_x * kElemsPerThread);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + A + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("const float* __restrict__ " + rtol + ",");
  w.line("const float* __restrict__ " + atol + ",");
  w.line("bool* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const float r = " + rtol + " ? " + rtol + "[0] : 0.0f;");
  w.line("const float a = " + atol + " ? " + atol + "[0] : 0.0f;");
  w.line("const int64_t lane = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t base = lane * 4LL;");
  w.line("const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x * 4LL;");
  w.line("for (int64_t i = base; i < " + std::to_string(total) + "LL; i += stride) {");
  w.indent();
  w.line("if (i < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float bv = " + B + "[(size_t)i];");
  w.line("const float tol = fmaf(r, fabsf(bv), a);");
  w.line(out + "[(size_t)i] = (fabsf(" + A + "[(size_t)i] - bv) <= tol);");
  w.dedent();
  w.line("}");
  w.line("if (i + 1 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float bv = " + B + "[(size_t)(i + 1)];");
  w.line("const float tol = fmaf(r, fabsf(bv), a);");
  w.line(out + "[(size_t)(i + 1)] = (fabsf(" + A + "[(size_t)(i + 1)] - bv) <= tol);");
  w.dedent();
  w.line("}");
  w.line("if (i + 2 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float bv = " + B + "[(size_t)(i + 2)];");
  w.line("const float tol = fmaf(r, fabsf(bv), a);");
  w.line(out + "[(size_t)(i + 2)] = (fabsf(" + A + "[(size_t)(i + 2)] - bv) <= tol);");
  w.dedent();
  w.line("}");
  w.line("if (i + 3 < " + std::to_string(total) + "LL) {");
  w.indent();
  w.line("const float bv = " + B + "[(size_t)(i + 3)];");
  w.line("const float tol = fmaf(r, fabsf(bv), a);");
  w.line(out + "[(size_t)(i + 3)] = (fabsf(" + A + "[(size_t)(i + 3)] - bv) <= tol);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {A, B, rtol, atol, out}, {}, {A, B, rtol, atol, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_index_add2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "index_add")) fail("index_add lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("index_add expects base,index,src");
  int axis = static_cast<int>(op.attrs.value("axis", 0));
  if (axis != 0) fail("index_add lowering currently supports axis=0");
  float alpha = 1.0f;
  if (op.attrs.is_object() && op.attrs.contains("alpha")) {
    const auto& a = op.attrs["alpha"];
    if (a.is_number()) alpha = static_cast<float>(a.get<double>());
    else if (a.is_number_integer()) alpha = static_cast<float>(a.get<int64_t>());
  }
  const std::string base = op.inputs[0];
  const std::string index = op.inputs[1];
  const std::string src = op.inputs[2];
  const std::string out = op.output;
  auto b_it = intent.tensors.find(base);
  auto i_it = intent.tensors.find(index);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out);
  if (b_it == intent.tensors.end() || i_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("index_add missing tensors");
  }
  if (b_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("index_add supports f32 base/src/out");
  if (i_it->second.dtype != "i32") fail("index_add supports i32 index");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "index_add.base");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "index_add.index");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "index_add.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "index_add.out");
  if (b_shape.size() != 2 || s_shape.size() != 2 || o_shape.size() != 2 || i_shape.size() != 1) fail("index_add shape rank mismatch");
  const int64_t M = b_shape[0];
  const int64_t N = b_shape[1];
  const int64_t L = i_shape[0];
  if (s_shape[0] != L || s_shape[1] != N || o_shape[0] != M || o_shape[1] != N) fail("index_add shape mismatch");

  int64_t total = M * N;
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
  w.line("const float* __restrict__ " + base + ",");
  w.line("const int* __restrict__ " + index + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("const int64_t row = tid / " + std::to_string(N) + "LL;");
  w.line("const int64_t col = tid - row * " + std::to_string(N) + "LL;");
  w.line("float acc = " + base + "[tid];");
  w.line("for (int64_t l = 0; l < " + std::to_string(L) + "LL; ++l) {");
  w.indent();
  w.line("const int64_t idx = (int64_t)" + index + "[l];");
  w.line("if (idx == row) acc += " + std::to_string(alpha) + "f * " + src + "[l * " + std::to_string(N) + "LL + col];");
  w.dedent();
  w.line("}");
  w.line(out + "[tid] = acc;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["L"] = L;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {base, index, src, out}, {}, {base, index, src, out});
  out_json["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_index_put2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "index_put")) fail("index_put lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 4) fail("index_put expects base,row_idx,col_idx,values");
  const bool accumulate = op.attrs.value("accumulate", false);
  const std::string base = op.inputs[0];
  const std::string row_idx = op.inputs[1];
  const std::string col_idx = op.inputs[2];
  const std::string values = op.inputs[3];
  const std::string out = op.output;
  auto b_it = intent.tensors.find(base);
  auto r_it = intent.tensors.find(row_idx);
  auto c_it = intent.tensors.find(col_idx);
  auto v_it = intent.tensors.find(values);
  auto o_it = intent.tensors.find(out);
  if (b_it == intent.tensors.end() || r_it == intent.tensors.end() || c_it == intent.tensors.end() || v_it == intent.tensors.end() ||
      o_it == intent.tensors.end()) {
    fail("index_put missing tensors");
  }
  if (b_it->second.dtype != "f32" || v_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("index_put supports f32 base/values/out");
  if (r_it->second.dtype != "i32" || c_it->second.dtype != "i32") fail("index_put supports i32 row/col indices");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "index_put.base");
  auto r_shape = resolve_shape_required(r_it->second, bindings, "index_put.row");
  auto c_shape = resolve_shape_required(c_it->second, bindings, "index_put.col");
  auto v_shape = resolve_shape_required(v_it->second, bindings, "index_put.values");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "index_put.out");
  if (b_shape.size() != 2 || o_shape.size() != 2 || r_shape.size() != 1 || c_shape.size() != 1 || v_shape.size() != 1) {
    fail("index_put shape rank mismatch");
  }
  const int64_t M = b_shape[0];
  const int64_t N = b_shape[1];
  const int64_t L = r_shape[0];
  if (c_shape[0] != L || v_shape[0] != L || o_shape[0] != M || o_shape[1] != N) fail("index_put shape mismatch");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total_elems = M * N;
  int64_t max_useful = 32;
  while ((max_useful << 1) <= total_elems && (max_useful << 1) <= 1024) max_useful <<= 1;
  if (block_x > max_useful) block_x = max_useful;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + base + ",");
  w.line("const int* __restrict__ " + row_idx + ",");
  w.line("const int* __restrict__ " + col_idx + ",");
  w.line("const float* __restrict__ " + values + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0) return;");
  w.line("const int tid = (int)threadIdx.x;");
  w.line("for (int64_t i = (int64_t)tid; i < " + std::to_string(total_elems) + "LL; i += (int64_t)blockDim.x) " + out + "[i] = " + base + "[i];");
  w.line("__syncthreads();");
  w.line("if (tid != 0) return;");
  w.line("for (int64_t l = 0; l < " + std::to_string(L) + "LL; ++l) {");
  w.indent();
  w.line("const int64_t r = (int64_t)" + row_idx + "[l];");
  w.line("const int64_t c = (int64_t)" + col_idx + "[l];");
  w.line("if (r < 0 || r >= " + std::to_string(M) + "LL || c < 0 || c >= " + std::to_string(N) + "LL) continue;");
  if (accumulate) {
    w.line(out + "[r * " + std::to_string(N) + "LL + c] += " + values + "[l];");
  } else {
    w.line(out + "[r * " + std::to_string(N) + "LL + c] = " + values + "[l];");
  }
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["L"] = L;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {base, row_idx, col_idx, values, out}, {}, {base, row_idx, col_idx, values, out});
  out_json["launch"] = {{"grid", {1, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_masked_scatter2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "masked_scatter")) fail("masked_scatter lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("masked_scatter expects inp,mask,source");
  const std::string inp = op.inputs[0];
  const std::string mask = op.inputs[1];
  const std::string src = op.inputs[2];
  const std::string out = op.output;
  auto i_it = intent.tensors.find(inp);
  auto m_it = intent.tensors.find(mask);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out);
  if (i_it == intent.tensors.end() || m_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("masked_scatter missing tensors");
  }
  if (i_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("masked_scatter supports f32 inp/src/out");
  if (m_it->second.dtype != "bool") fail("masked_scatter expects bool mask");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "masked_scatter.inp");
  auto m_shape = resolve_shape_required(m_it->second, bindings, "masked_scatter.mask");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "masked_scatter.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "masked_scatter.out");
  if (i_shape.size() != 2 || m_shape.size() != 2 || o_shape.size() != 2 || s_shape.size() != 1) fail("masked_scatter shape rank mismatch");
  if (i_shape != m_shape || i_shape != o_shape) fail("masked_scatter shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const int64_t L = s_shape[0];
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t max_useful = 32;
  while ((max_useful << 1) <= total && (max_useful << 1) <= 1024) max_useful <<= 1;
  if (block_x > max_useful) block_x = max_useful;
  const int64_t shared_mem = block_x * static_cast<int64_t>(sizeof(int));

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const bool* __restrict__ " + mask + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0) return;");
  w.line("const int tid = (int)threadIdx.x;");
  w.line("for (int64_t i = (int64_t)tid; i < " + std::to_string(total) + "LL; i += (int64_t)blockDim.x) " + out + "[i] = " + inp + "[i];");
  w.line("__syncthreads();");
  w.line("extern __shared__ int s_scan[];");
  w.line("__shared__ int64_t s_total;");
  w.line("if (tid == 0) s_total = 0;");
  w.line("__syncthreads();");
  w.line("for (int64_t base = 0; base < " + std::to_string(total) + "LL; base += (int64_t)blockDim.x) {");
  w.indent();
  w.line("const int64_t idx = base + (int64_t)tid;");
  w.line("int flag = 0;");
  w.line("if (idx < " + std::to_string(total) + "LL) flag = " + mask + "[idx] ? 1 : 0;");
  w.line("s_scan[tid] = flag;");
  w.line("__syncthreads();");
  w.line("for (int off = 1; off < (int)blockDim.x; off <<= 1) {");
  w.indent();
  w.line("int add = 0;");
  w.line("if (tid >= off) add = s_scan[tid - off];");
  w.line("__syncthreads();");
  w.line("s_scan[tid] += add;");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("const int chunk_count = s_scan[(int)blockDim.x - 1];");
  w.line("if (flag) {");
  w.indent();
  w.line("const int64_t src_pos = s_total + (int64_t)s_scan[tid] - 1LL;");
  w.line("if (src_pos < " + std::to_string(L) + "LL) " + out + "[idx] = " + src + "[src_pos];");
  w.dedent();
  w.line("}");
  w.line("__syncthreads();");
  w.line("if (tid == 0) s_total += (int64_t)chunk_count;");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["L"] = L;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {inp, mask, src, out}, {}, {inp, mask, src, out});
  out_json["launch"] = {{"grid", {1, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", shared_mem}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_masked_select2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "masked_select")) fail("masked_select lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("masked_select expects inp,mask");
  const std::string inp = op.inputs[0];
  const std::string mask = op.inputs[1];
  const std::string out = op.output;
  auto i_it = intent.tensors.find(inp);
  auto m_it = intent.tensors.find(mask);
  auto o_it = intent.tensors.find(out);
  if (i_it == intent.tensors.end() || m_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("masked_select missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("masked_select supports f32 inp/out");
  if (m_it->second.dtype != "bool") fail("masked_select expects bool mask");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "masked_select.inp");
  auto m_shape = resolve_shape_required(m_it->second, bindings, "masked_select.mask");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "masked_select.out");
  if (i_shape.size() != 2 || m_shape.size() != 2 || o_shape.size() != 1) fail("masked_select shape rank mismatch");
  if (i_shape != m_shape) fail("masked_select inp/mask shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const int64_t L = o_shape[0];
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t max_useful = 32;
  while ((max_useful << 1) <= total && (max_useful << 1) <= 1024) max_useful <<= 1;
  if (block_x > max_useful) block_x = max_useful;
  const int64_t shared_mem = block_x * static_cast<int64_t>(sizeof(int));

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const bool* __restrict__ " + mask + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0) return;");
  w.line("const int tid = (int)threadIdx.x;");
  w.line("extern __shared__ int s_scan[];");
  w.line("__shared__ int64_t s_total;");
  w.line("if (tid == 0) s_total = 0;");
  w.line("__syncthreads();");
  w.line("for (int64_t base = 0; base < " + std::to_string(total) + "LL; base += (int64_t)blockDim.x) {");
  w.indent();
  w.line("const int64_t idx = base + (int64_t)tid;");
  w.line("int flag = 0;");
  w.line("if (idx < " + std::to_string(total) + "LL) flag = " + mask + "[idx] ? 1 : 0;");
  w.line("s_scan[tid] = flag;");
  w.line("__syncthreads();");
  w.line("for (int off = 1; off < (int)blockDim.x; off <<= 1) {");
  w.indent();
  w.line("int add = 0;");
  w.line("if (tid >= off) add = s_scan[tid - off];");
  w.line("__syncthreads();");
  w.line("s_scan[tid] += add;");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("const int chunk_count = s_scan[(int)blockDim.x - 1];");
  w.line("if (flag) {");
  w.indent();
  w.line("const int64_t out_pos = s_total + (int64_t)s_scan[tid] - 1LL;");
  w.line("if (out_pos < " + std::to_string(L) + "LL) " + out + "[out_pos] = " + inp + "[idx];");
  w.dedent();
  w.line("}");
  w.line("__syncthreads();");
  w.line("if (tid == 0) s_total += (int64_t)chunk_count;");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if (tid == 0) {");
  w.indent();
  w.line("for (int64_t i = s_total; i < " + std::to_string(L) + "LL; ++i) " + out + "[i] = 0.0f;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["L"] = L;
  json out_json;
  out_json["kernel_name"] = intent.name;
  out_json["cuda_src"] = cuda_ss.str();
  out_json["io_spec"] = io_spec_from_args(intent, {inp, mask, out}, {}, {inp, mask, out});
  out_json["launch"] = {{"grid", {1, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", shared_mem}};
  out_json["output_names"] = {out};
  out_json["bindings"] = out_bindings;
  return out_json;
}

json emit_layer_norm_persistent_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() < 17 || intent.ops.size() > 21) fail("layer_norm_persistent lowering expects 17~21-op pattern");
  const std::string X = "in_ptr";
  const std::string W = "weight_ptr";
  const std::string B = "bias_ptr";
  const std::string Y = "out_ptr";
  const std::string Mean = "out_mean_ptr";
  const std::string Rstd = "out_rstd_ptr";
  const std::string eps = "eps";
  auto x_it = intent.tensors.find(X);
  auto w_it = intent.tensors.find(W);
  auto b_it = intent.tensors.find(B);
  auto y_it = intent.tensors.find(Y);
  auto m_it = intent.tensors.find(Mean);
  auto r_it = intent.tensors.find(Rstd);
  auto e_it = intent.tensors.find(eps);
  if (x_it == intent.tensors.end() || w_it == intent.tensors.end() || b_it == intent.tensors.end() || y_it == intent.tensors.end() ||
      m_it == intent.tensors.end() || r_it == intent.tensors.end() || e_it == intent.tensors.end()) {
    fail("layer_norm_persistent missing tensors");
  }
  for (const auto* t : {&x_it->second, &w_it->second, &b_it->second, &y_it->second, &m_it->second, &r_it->second, &e_it->second}) {
    if (t->dtype != "f32") fail("layer_norm_persistent supports f32 only");
  }
  auto x_shape = resolve_shape_required(x_it->second, bindings, "layer_norm_persistent.X");
  auto y_shape = resolve_shape_required(y_it->second, bindings, "layer_norm_persistent.Y");
  if (x_shape.size() != 2 || y_shape.size() != 2 || x_shape != y_shape) fail("layer_norm_persistent expects rank-2 X/Y");
  const int64_t M = x_shape[0];
  const int64_t N = x_shape[1];

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 64) block_x = 64;
  if (block_x > 1024) block_x = 1024;
  int64_t block_pow2 = 1;
  while ((block_pow2 << 1) <= block_x) block_pow2 <<= 1;
  block_x = block_pow2;
  const int64_t grid_x = M;
  const int64_t shared_mem = block_x * static_cast<int64_t>(sizeof(float));

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + X + ",");
  w.line("const float* __restrict__ " + W + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("const float* __restrict__ " + eps + ",");
  w.line("float* __restrict__ " + Y + ",");
  w.line("float* __restrict__ " + Mean + ",");
  w.line("float* __restrict__ " + Rstd);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const float eps_v = " + eps + " ? " + eps + "[0] : 1e-5f;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) sum += " + X + "[base + n];");
  w.line("const float mean = sum / (float)" + std::to_string(N) + ";");
  w.line(Mean + "[row] = mean;");
  w.line("float var_sum = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float d = " + X + "[base + n] - mean;");
  w.line("var_sum = fmaf(d, d, var_sum);");
  w.dedent();
  w.line("}");
  w.line("const float rstd = rsqrtf(var_sum / (float)" + std::to_string(N) + " + eps_v);");
  w.line(Rstd + "[row] = rstd;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float norm = (" + X + "[base + n] - mean) * rstd;");
  w.line(Y + "[base + n] = norm * " + W + "[n] + " + B + "[n];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {X, W, B, eps, Y, Mean, Rstd}, {}, {X, W, B, eps, Y, Mean, Rstd});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {Y, Mean, Rstd};
  out["bindings"] = out_bindings;
  return out;
}

json emit_nll_loss2d_forward_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "nll_loss2d_forward")) fail("nll_loss2d_forward lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("nll_loss2d_forward expects self,target,weight");
  const int reduction = op.attrs.value("reduction", 1);
  const int64_t ignore_index = op.attrs.value("ignore_index", static_cast<int64_t>(-100));
  const std::string self = op.inputs[0];
  const std::string target = op.inputs[1];
  const std::string weight = op.inputs[2];
  const std::string output = op.output;
  auto s_it = intent.tensors.find(self);
  auto t_it = intent.tensors.find(target);
  auto w_it = intent.tensors.find(weight);
  auto o_it = intent.tensors.find(output);
  if (s_it == intent.tensors.end() || t_it == intent.tensors.end() || w_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("nll_loss2d_forward missing tensors");
  }
  if (s_it->second.dtype != "f32" || w_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("nll_loss2d_forward expects f32 self/weight/out");
  if (t_it->second.dtype != "i64") fail("nll_loss2d_forward expects i64 target");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "nll_loss2d_forward.self");
  auto t_shape = resolve_shape_required(t_it->second, bindings, "nll_loss2d_forward.target");
  auto w_shape = resolve_shape_required(w_it->second, bindings, "nll_loss2d_forward.weight");
  if (s_shape.size() != 4 || t_shape.size() != 3 || w_shape.size() != 1) fail("nll_loss2d_forward shape rank mismatch");
  const int64_t N = s_shape[0];
  const int64_t C = s_shape[1];
  const int64_t H = s_shape[2];
  const int64_t Wd = s_shape[3];
  if (t_shape[0] != N || t_shape[1] != H || t_shape[2] != Wd || w_shape[0] != C) fail("nll_loss2d_forward shape mismatch");

  const int64_t total_sites = N * H * Wd;
  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t max_useful = 32;
  while ((max_useful << 1) <= total_sites && (max_useful << 1) <= 1024) max_useful <<= 1;
  if (block_x > max_useful) block_x = max_useful;
  const bool tiny_serial = (total_sites <= 16);
  const bool use_warp_reduce = (!tiny_serial && block_x == 32);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  if (!tiny_serial && !use_warp_reduce) {
    w.line("#include \"kernels/reduce.cuh\"");
  }
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(tiny_serial ? 1 : block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + self + ",");
  w.line("const int64_t* __restrict__ " + target + ",");
  w.line("const float* __restrict__ " + weight + ",");
  w.line("float* __restrict__ " + output);
  w.dedent();
  w.line(") {");
  w.indent();
  if (tiny_serial) {
    w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
    w.line("float loss = 0.0f;");
    w.line("float total_weight = 0.0f;");
    w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
    w.indent();
    w.line("for (int64_t h = 0; h < " + std::to_string(H) + "LL; ++h) {");
    w.indent();
    w.line("for (int64_t w = 0; w < " + std::to_string(Wd) + "LL; ++w) {");
    w.indent();
    w.line("const int64_t t = " + target + "[(n * " + std::to_string(H) + "LL + h) * " + std::to_string(Wd) + "LL + w];");
    w.line("if (t == " + std::to_string(ignore_index) + "LL) continue;");
    w.line("if (t < 0 || t >= " + std::to_string(C) + "LL) continue;");
    w.line("const float wv = " + weight + "[t];");
    w.line("const int64_t idx = (((n * " + std::to_string(C) + "LL + t) * " + std::to_string(H) + "LL + h) * " + std::to_string(Wd) + "LL + w);");
    w.line("loss += -" + self + "[idx] * wv;");
    w.line("total_weight += wv;");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    if (reduction == 2) {
      w.line(output + "[0] = loss;");
    } else {
      w.line(output + "[0] = (total_weight > 0.0f) ? (loss / total_weight) : 0.0f;");
    }
  } else {
    w.line("float loss = 0.0f;");
    w.line("float total_weight = 0.0f;");
    w.line("const int64_t hw = " + std::to_string(H) + "LL * " + std::to_string(Wd) + "LL;");
    w.line("for (int64_t linear = (int64_t)threadIdx.x; linear < " + std::to_string(total_sites) + "LL; linear += (int64_t)blockDim.x) {");
    w.indent();
    w.line("const int64_t n = linear / hw;");
    w.line("const int64_t rem = linear - n * hw;");
    w.line("const int64_t h = rem / " + std::to_string(Wd) + "LL;");
    w.line("const int64_t w = rem - h * " + std::to_string(Wd) + "LL;");
    w.line("const int64_t t = " + target + "[linear];");
    w.line("if (t == " + std::to_string(ignore_index) + "LL || t < 0 || t >= " + std::to_string(C) + "LL) continue;");
    w.line("const float wv = " + weight + "[t];");
    w.line("const int64_t idx = (((n * " + std::to_string(C) + "LL + t) * " + std::to_string(H) + "LL + h) * " + std::to_string(Wd) + "LL + w);");
    w.line("loss += -" + self + "[idx] * wv;");
    w.line("total_weight += wv;");
    w.dedent();
    w.line("}");
    if (use_warp_reduce) {
      w.line("const unsigned mask = 0xffffffffu;");
      w.line("for (int offset = 16; offset > 0; offset >>= 1) {");
      w.indent();
      w.line("loss += __shfl_down_sync(mask, loss, offset);");
      w.line("total_weight += __shfl_down_sync(mask, total_weight, offset);");
      w.dedent();
      w.line("}");
      if (reduction == 2) {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = loss;");
      } else {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = (total_weight > 0.0f) ? (loss / total_weight) : 0.0f;");
      }
    } else {
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
      w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red_loss;");
      w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red_w;");
      w.line("const float loss_sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(loss, &red_loss);");
      w.line("const float w_sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(total_weight, &red_w);");
      if (reduction == 2) {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = loss_sum;");
      } else {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = (w_sum > 0.0f) ? (loss_sum / w_sum) : 0.0f;");
      }
    }
  }
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  out_bindings["C"] = C;
  out_bindings["H"] = H;
  out_bindings["W"] = Wd;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {self, target, weight, output}, {}, {self, target, weight, output});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {tiny_serial ? 1 : block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {output};
  out["bindings"] = out_bindings;
  return out;
}

json emit_nll_loss_forward_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "nll_loss_forward")) fail("nll_loss_forward lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("nll_loss_forward expects self,target[,weight]");
  const int reduction = op.attrs.value("reduction", 1);
  const int64_t ignore_index = op.attrs.value("ignore_index", static_cast<int64_t>(-100));
  const std::string self = op.inputs[0];
  const std::string target = op.inputs[1];
  const std::string weight = (op.inputs.size() >= 3) ? op.inputs[2] : std::string();
  const std::string output = op.output;

  auto s_it = intent.tensors.find(self);
  auto t_it = intent.tensors.find(target);
  auto o_it = intent.tensors.find(output);
  if (s_it == intent.tensors.end() || t_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("nll_loss_forward missing tensors");
  if (s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("nll_loss_forward expects f32 self/out");
  if (t_it->second.dtype != "i64") fail("nll_loss_forward expects i64 target");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "nll_loss_forward.self");
  auto t_shape = resolve_shape_required(t_it->second, bindings, "nll_loss_forward.target");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "nll_loss_forward.output");
  if (s_shape.size() != 2 || t_shape.size() != 1) fail("nll_loss_forward expects self:[N,C], target:[N]");
  const int64_t N = s_shape[0];
  const int64_t C = s_shape[1];
  if (t_shape[0] != N) fail("nll_loss_forward target shape mismatch");
  const bool has_weight = !weight.empty();
  if (has_weight) {
    auto w_it = intent.tensors.find(weight);
    if (w_it == intent.tensors.end()) fail("nll_loss_forward missing weight tensor");
    if (w_it->second.dtype != "f32") fail("nll_loss_forward expects f32 weight");
    auto w_shape = resolve_shape_required(w_it->second, bindings, "nll_loss_forward.weight");
    if (w_shape.size() != 1 || w_shape[0] != C) fail("nll_loss_forward weight shape mismatch");
  }
  if (reduction == 0) {
    if (!(o_shape.size() == 1 && o_shape[0] == N)) fail("nll_loss_forward reduction=0 expects output shape [N]");
  } else {
    if (!o_shape.empty()) fail("nll_loss_forward reduction!=0 expects scalar output");
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t max_useful = 32;
  while ((max_useful << 1) <= N && (max_useful << 1) <= 1024) max_useful <<= 1;
  if (block_x > max_useful) block_x = max_useful;
  int64_t grid_x = 1;
  if (reduction == 0) grid_x = (N + block_x - 1) / block_x;
  const bool use_warp_reduce = (reduction != 0 && block_x == 32);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  if (reduction != 0 && !use_warp_reduce) {
    w.line("#include \"kernels/reduce.cuh\"");
  }
  w.blank();

  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + self + ",");
  w.line("const int64_t* __restrict__ " + target + ",");
  if (has_weight) w.line("const float* __restrict__ " + weight + ",");
  w.line("float* __restrict__ " + output);
  w.dedent();
  w.line(") {");
  w.indent();
  if (reduction == 0) {
    w.line("const int64_t n = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
    w.line("if (n >= " + std::to_string(N) + "LL) return;");
    w.line("const int64_t t = " + target + "[n];");
    w.line("if (t == " + std::to_string(ignore_index) + "LL || t < 0 || t >= " + std::to_string(C) + "LL) { " + output + "[n] = 0.0f; return; }");
    if (has_weight) {
      w.line("const float wv = " + weight + "[t];");
    } else {
      w.line("const float wv = 1.0f;");
    }
    w.line("const int64_t idx = n * " + std::to_string(C) + "LL + t;");
    w.line(output + "[n] = -" + self + "[idx] * wv;");
  } else {
    w.line("float loss = 0.0f;");
    w.line("float total_weight = 0.0f;");
    w.line("for (int64_t n = (int64_t)threadIdx.x; n < " + std::to_string(N) + "LL; n += (int64_t)blockDim.x) {");
    w.indent();
    w.line("const int64_t t = " + target + "[n];");
    w.line("if (t == " + std::to_string(ignore_index) + "LL || t < 0 || t >= " + std::to_string(C) + "LL) continue;");
    if (has_weight) {
      w.line("const float wv = " + weight + "[t];");
    } else {
      w.line("const float wv = 1.0f;");
    }
    w.line("const int64_t idx = n * " + std::to_string(C) + "LL + t;");
    w.line("loss += -" + self + "[idx] * wv;");
    w.line("total_weight += wv;");
    w.dedent();
    w.line("}");
    if (use_warp_reduce) {
      w.line("const unsigned mask = 0xffffffffu;");
      w.line("for (int offset = 16; offset > 0; offset >>= 1) {");
      w.indent();
      w.line("loss += __shfl_down_sync(mask, loss, offset);");
      w.line("total_weight += __shfl_down_sync(mask, total_weight, offset);");
      w.dedent();
      w.line("}");
      if (reduction == 2) {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = loss;");
      } else {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = (total_weight > 0.0f) ? (loss / total_weight) : 0.0f;");
      }
    } else {
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
      w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red_loss;");
      w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red_w;");
      w.line("const float loss_sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(loss, &red_loss);");
      w.line("const float w_sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(total_weight, &red_w);");
      if (reduction == 2) {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = loss_sum;");
      } else {
        w.line("if ((int)threadIdx.x == 0) " + output + "[0] = (w_sum > 0.0f) ? (loss_sum / w_sum) : 0.0f;");
      }
    }
  }
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  out_bindings["C"] = C;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  if (has_weight) {
    out["io_spec"] = io_spec_from_args(intent, {self, target, weight, output}, {}, {self, target, weight, output});
  } else {
    out["io_spec"] = io_spec_from_args(intent, {self, target, output}, {}, {self, target, output});
  }
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {output};
  out["bindings"] = out_bindings;
  return out;
}

json emit_normed_cumsum2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 4) fail("normed_cumsum lowering expects 4-op pattern");
  const Op& cumsum = intent.ops[0];
  const Op& reduce = intent.ops[1];
  const Op& add = intent.ops[2];
  const Op& div = intent.ops[3];
  if (cumsum.op != "cumsum" || reduce.op != "reduce_sum" || add.op != "add" || div.op != "div") {
    fail("normed_cumsum lowering pattern mismatch");
  }
  if (cumsum.inputs.size() != 1 || reduce.inputs.size() != 1 || add.inputs.size() != 2 || div.inputs.size() != 2) {
    fail("normed_cumsum arity mismatch");
  }
  int axis = 1;
  if (cumsum.attrs.is_object() && cumsum.attrs.contains("axis")) {
    axis = static_cast<int>(resolve_dim_token_required(cumsum.attrs["axis"], bindings, "normed_cumsum.axis"));
  }
  if (axis != 1 || !_reduce_axis_is_1(reduce.attrs)) fail("normed_cumsum expects cumsum/reduce_sum axis=1");
  const std::string inp = cumsum.inputs[0];
  if (reduce.inputs[0] != inp) fail("normed_cumsum expects reduce_sum(inp)");
  if (div.inputs[0] != cumsum.output || div.inputs[1] != add.output) fail("normed_cumsum expects div(cumsum_out, sum_eps)");
  const std::string eps = (add.inputs[0] == reduce.output) ? add.inputs[1] : ((add.inputs[1] == reduce.output) ? add.inputs[0] : std::string());
  if (eps.empty()) fail("normed_cumsum expects add(reduce_sum_out, eps)");
  const std::string out_name = div.output;
  auto i_it = intent.tensors.find(inp);
  auto e_it = intent.tensors.find(eps);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || e_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("normed_cumsum missing tensors");
  if (i_it->second.dtype != "f32" || e_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("normed_cumsum supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "normed_cumsum.inp");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "normed_cumsum.out");
  if (i_shape.size() != 2 || o_shape.size() != 2 || i_shape != o_shape) fail("normed_cumsum expects rank-2 inp/out with same shape");
  auto e_shape = resolve_shape_required(e_it->second, bindings, "normed_cumsum.eps");
  if (!e_shape.empty()) fail("normed_cumsum expects scalar eps");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];

  // Correctness-first fallback: use one thread per row for stable prefix+normalize.
  // This avoids scan-race edge cases that can produce INF in CUDA smoke/full196.
  const int64_t block_x = 1;
  const int64_t grid_x = M;
  const int64_t shared_mem = 0;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + eps + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("if ((int)threadIdx.x != 0) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("const float eps_v = (" + eps + " ? " + eps + "[0] : 1e-12f);");
  w.line("float denom = eps_v;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) denom += " + inp + "[base + n];");
  w.line("float run = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("run += " + inp + "[base + n];");
  w.line(out_name + "[base + n] = run / denom;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, eps, out_name}, {}, {inp, eps, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", shared_mem}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_polar2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "polar")) fail("polar lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("polar expects abs,angle inputs");
  const std::string abs_name = op.inputs[0];
  const std::string angle_name = op.inputs[1];
  const std::string out_name = op.output;
  auto a_it = intent.tensors.find(abs_name);
  auto g_it = intent.tensors.find(angle_name);
  auto o_it = intent.tensors.find(out_name);
  if (a_it == intent.tensors.end() || g_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("polar missing tensors");
  if (a_it->second.dtype != "f32" || g_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("polar supports f32 only");
  auto a_shape = resolve_shape_required(a_it->second, bindings, "polar.abs");
  auto g_shape = resolve_shape_required(g_it->second, bindings, "polar.angle");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "polar.out");
  if (a_shape.size() != 2 || g_shape.size() != 2 || a_shape != g_shape) fail("polar expects rank-2 abs/angle with same shape");
  if (o_shape.size() != 3 || o_shape[0] != a_shape[0] || o_shape[1] != a_shape[1] || o_shape[2] != 2) {
    fail("polar expects output shape [M,N,2]");
  }
  const int64_t M = a_shape[0];
  const int64_t N = a_shape[1];
  const int64_t total = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + abs_name + ",");
  w.line("const float* __restrict__ " + angle_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("const float r = " + abs_name + "[tid];");
  w.line("const float a = " + angle_name + "[tid];");
  w.line("const int64_t out_base = tid * 2LL;");
  w.line(out_name + "[out_base + 0] = r * cosf(a);");
  w.line(out_name + "[out_base + 1] = r * sinf(a);");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {abs_name, angle_name, out_name}, {}, {abs_name, angle_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}
