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
  w.line("for (int64_t dd = 0; dd < " + std::to_string(D) + "LL; ++dd) score = fmaf(" + query + "[q_base + dd], " + key + "[k_base + dd], score);");
  w.line("score = score * scale - max_score;");
  w.line("const float wv = expf(score);");
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
  w.line("for (int64_t dd = 0; dd < " + std::to_string(D) + "LL; ++dd) score = fmaf(" + Q + "[q_base + dd], " + K + "[k_base + dd], score);");
  w.line("score = score * sm + " + attn_mask + "[mask_base + k_idx] - max_score;");
  w.line("const float wv = expf(score);");
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

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + base + ",");
  w.line("const int* __restrict__ " + index + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out + "[i] = " + base + "[i];");
  w.line("for (int64_t l = 0; l < " + std::to_string(L) + "LL; ++l) {");
  w.indent();
  w.line("const int64_t row = (int64_t)" + index + "[l];");
  w.line("if (row < 0 || row >= " + std::to_string(M) + "LL) continue;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line(out + "[row * " + std::to_string(N) + "LL + n] += " + std::to_string(alpha) + "f * " + src + "[l * " + std::to_string(N) + "LL + n];");
  w.dedent();
  w.line("}");
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
  out_json["io_spec"] = io_spec_from_args(intent, {base, index, src, out}, {}, {base, index, src, out});
  out_json["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
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

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + base + ",");
  w.line("const int* __restrict__ " + row_idx + ",");
  w.line("const int* __restrict__ " + col_idx + ",");
  w.line("const float* __restrict__ " + values + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out + "[i] = " + base + "[i];");
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
  out_json["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
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

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const bool* __restrict__ " + mask + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("const int64_t total = " + std::to_string(M * N) + "LL;");
  w.line("for (int64_t i = 0; i < total; ++i) " + out + "[i] = " + inp + "[i];");
  w.line("int64_t src_pos = 0;");
  w.line("for (int64_t i = 0; i < total; ++i) {");
  w.indent();
  w.line("if (" + mask + "[i]) {");
  w.indent();
  w.line("if (src_pos < " + std::to_string(L) + "LL) " + out + "[i] = " + src + "[src_pos];");
  w.line("++src_pos;");
  w.dedent();
  w.line("}");
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
  out_json["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
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

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const bool* __restrict__ " + mask + ",");
  w.line("float* __restrict__ " + out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("const int64_t total = " + std::to_string(M * N) + "LL;");
  w.line("int64_t out_pos = 0;");
  w.line("for (int64_t i = 0; i < total; ++i) {");
  w.indent();
  w.line("if (" + mask + "[i]) {");
  w.indent();
  w.line("if (out_pos < " + std::to_string(L) + "LL) " + out + "[out_pos] = " + inp + "[i];");
  w.line("++out_pos;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("for (int64_t i = out_pos; i < " + std::to_string(L) + "LL; ++i) " + out + "[i] = 0.0f;");
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
  out_json["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
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

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

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

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + self + ",");
  w.line("const int64_t* __restrict__ " + target + ",");
  w.line("const float* __restrict__ " + weight + ",");
  w.line("float* __restrict__ " + output);
  w.dedent();
  w.line(") {");
  w.indent();
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
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
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

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + self + ",");
  w.line("const int64_t* __restrict__ " + target + ",");
  if (has_weight) w.line("const float* __restrict__ " + weight + ",");
  w.line("float* __restrict__ " + output);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  if (reduction == 0) {
    w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
    w.indent();
    w.line("const int64_t t = " + target + "[n];");
    w.line("if (t == " + std::to_string(ignore_index) + "LL || t < 0 || t >= " + std::to_string(C) + "LL) { " + output + "[n] = 0.0f; continue; }");
    if (has_weight) {
      w.line("const float wv = " + weight + "[t];");
    } else {
      w.line("const float wv = 1.0f;");
    }
    w.line("const int64_t idx = n * " + std::to_string(C) + "LL + t;");
    w.line(output + "[n] = -" + self + "[idx] * wv;");
    w.dedent();
    w.line("}");
  } else {
    w.line("float loss = 0.0f;");
    w.line("float total_weight = 0.0f;");
    w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
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
    if (reduction == 2) {
      w.line(output + "[0] = loss;");
    } else {
      w.line(output + "[0] = (total_weight > 0.0f) ? (loss / total_weight) : 0.0f;");
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
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
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

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

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
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) sum += " + inp + "[base + n];");
  w.line("const float denom = sum + (" + eps + " ? " + eps + "[0] : 1e-12f);");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("acc += " + inp + "[base + n];");
  w.line(out_name + "[base + n] = acc / denom;");
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
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
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
