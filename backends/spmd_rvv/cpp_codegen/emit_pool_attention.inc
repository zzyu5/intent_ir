void emit_avg_pool2d(CodeWriter& w, const std::string& out, const std::string& inp,
                     const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                     int kh, int kw, int sh, int sw, int ph, int pw, bool count_include_pad) {
  if (in_shape.size() != 4 || out_shape.size() != 4) fail("avg_pool2d expects rank-4 NCHW tensors");
  const int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
  const int64_t ON = out_shape[0], OC = out_shape[1], OH = out_shape[2], OW = out_shape[3];
  if (N != ON || C != OC) fail("avg_pool2d output N/C mismatch");
  w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
  w.indent();
  w.line("for (int c = 0; c < " + std::to_string(C) + "; ++c) {");
  w.indent();
  w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
  w.indent();
  w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
  w.indent();
  w.line("const int h_start = oh * " + std::to_string(sh) + " - " + std::to_string(ph) + ";");
  w.line("const int w_start = ow * " + std::to_string(sw) + " - " + std::to_string(pw) + ";");
  w.line("const int h_end = h_start + " + std::to_string(kh) + ";");
  w.line("const int w_end = w_start + " + std::to_string(kw) + ";");
  w.line("float sum = 0.0f;");
  w.line("int cnt = 0;");
  w.line("for (int ih = h_start; ih < h_end; ++ih) {");
  w.indent();
  w.line("if (ih < 0 || ih >= " + std::to_string(H) + ") continue;");
  w.line("for (int iw = w_start; iw < w_end; ++iw) {");
  w.indent();
  w.line("if (iw < 0 || iw >= " + std::to_string(W) + ") continue;");
  w.line("sum += " + inp + "[idx4(n,c,ih,iw," + std::to_string(C) + "," + std::to_string(H) + "," + std::to_string(W) + ")];");
  w.line("cnt += 1;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  if (count_include_pad) {
    w.line("const float denom = (float)" + std::to_string(kh * kw) + ";");
  } else {
    w.line("const float denom = (float)(cnt > 0 ? cnt : 1);");
  }
  w.line(out + "[idx4(n,c,oh,ow," + std::to_string(C) + "," + std::to_string(OH) + "," + std::to_string(OW) + ")] = sum / denom;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

void emit_max_pool2d_with_indices(CodeWriter& w, const std::string& out, const std::string& inp,
                                  const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                                  int kh, int kw, int sh, int sw, int ph, int pw, bool emit_indices) {
  if (in_shape.size() != 4 || out_shape.size() != 4) fail("max_pool2d_with_indices expects rank-4 NCHW tensors");
  const int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
  const int64_t ON = out_shape[0], OC = out_shape[1], OH = out_shape[2], OW = out_shape[3];
  if (N != ON || C != OC) fail("max_pool2d_with_indices output N/C mismatch");
  w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
  w.indent();
  w.line("for (int c = 0; c < " + std::to_string(C) + "; ++c) {");
  w.indent();
  w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
  w.indent();
  w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
  w.indent();
  w.line("const int h_start = oh * " + std::to_string(sh) + " - " + std::to_string(ph) + ";");
  w.line("const int w_start = ow * " + std::to_string(sw) + " - " + std::to_string(pw) + ";");
  w.line("const int h_end = h_start + " + std::to_string(kh) + ";");
  w.line("const int w_end = w_start + " + std::to_string(kw) + ";");
  w.line("float best = -INFINITY;");
  w.line("int32_t best_idx = 0;");
  w.line("for (int ih = h_start; ih < h_end; ++ih) {");
  w.indent();
  w.line("if (ih < 0 || ih >= " + std::to_string(H) + ") continue;");
  w.line("for (int iw = w_start; iw < w_end; ++iw) {");
  w.indent();
  w.line("if (iw < 0 || iw >= " + std::to_string(W) + ") continue;");
  w.line("const float v = " + inp + "[idx4(n,c,ih,iw," + std::to_string(C) + "," + std::to_string(H) + "," + std::to_string(W) + ")];");
  w.line("if (v > best) { best = v; best_idx = ih * " + std::to_string(W) + " + iw; }");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  if (emit_indices) {
    w.line(out + "[idx4(n,c,oh,ow," + std::to_string(C) + "," + std::to_string(OH) + "," + std::to_string(OW) + ")] = (int64_t)best_idx;");
  } else {
    w.line(out + "[idx4(n,c,oh,ow," + std::to_string(C) + "," + std::to_string(OH) + "," + std::to_string(OW) + ")] = best;");
  }
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

void emit_mse_loss(CodeWriter& w, const std::string& out, const std::string& a, const std::string& b,
                   const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape, const std::string& reduction) {
  if (in_shape.empty()) fail("mse_loss expects non-scalar input");
  const int64_t n = numel(in_shape);
  if (n <= 0) fail("mse_loss expects positive numel");
  if (reduction == "none") fail("mse_loss reduction=none lowering not implemented");
  if (!out_shape.empty()) fail("mse_loss reduced output must be scalar");
  w.line("double acc = 0.0;");
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  w.indent();
  w.line("const float d = " + a + "[i] - " + b + "[i];");
  w.line("acc += (double)(d * d);");
  w.dedent();
  w.line("}");
  if (reduction == "sum") {
    w.line(out + "[0] = (float)acc;");
  } else {
    w.line(out + "[0] = (float)(acc / (double)" + std::to_string(n) + ");");
  }
}

void emit_scaled_dot_product_attention(CodeWriter& w, const std::string& out,
                                       const std::string& query, const std::string& key, const std::string& value,
                                       const std::vector<int64_t>& q_shape, const std::vector<int64_t>& k_shape,
                                       const std::vector<int64_t>& v_shape, const std::vector<int64_t>& out_shape,
                                       bool is_causal) {
  if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || out_shape.size() != 4) {
    fail("scaled_dot_product_attention expects rank-4 BHQD/BHKD tensors");
  }
  const int64_t B = q_shape[0], H = q_shape[1], Q = q_shape[2], D = q_shape[3];
  const int64_t BK = k_shape[0], HK = k_shape[1], K = k_shape[2], DK = k_shape[3];
  const int64_t BV = v_shape[0], HV = v_shape[1], KV = v_shape[2], DV = v_shape[3];
  if (BK != B || HK != H || DK != D) fail("scaled_dot_product_attention shape mismatch between query/key");
  if (BV != B || HV != H || KV != K) fail("scaled_dot_product_attention shape mismatch between key/value");
  if (out_shape[0] != B || out_shape[1] != H || out_shape[2] != Q || out_shape[3] != DV) {
    fail("scaled_dot_product_attention output shape mismatch");
  }
  w.line("const float inv_sqrt_d = 1.0f / sqrtf((float)" + std::to_string(D) + ");");
  w.line("for (int b = 0; b < " + std::to_string(B) + "; ++b) {");
  w.indent();
  w.line("for (int h = 0; h < " + std::to_string(H) + "; ++h) {");
  w.indent();
  w.line("for (int q = 0; q < " + std::to_string(Q) + "; ++q) {");
  w.indent();
  w.line("for (int d = 0; d < " + std::to_string(DV) + "; ++d) {");
  w.indent();
  w.line("float max_score = -INFINITY;");
  w.line("for (int k = 0; k < " + std::to_string(K) + "; ++k) {");
  w.indent();
  if (is_causal) {
    w.line("if (k > q) continue;");
  }
  w.line("float dot = 0.0f;");
  w.line("for (int t = 0; t < " + std::to_string(D) + "; ++t) {");
  w.indent();
  w.line("dot += " + query + "[idx4(b,h,q,t," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(D) + ")] * " +
         key + "[idx4(b,h,k,t," + std::to_string(H) + "," + std::to_string(K) + "," + std::to_string(D) + ")];");
  w.dedent();
  w.line("}");
  w.line("float score = dot * inv_sqrt_d;");
  w.line("if (score > max_score) max_score = score;");
  w.dedent();
  w.line("}");
  w.line("float denom = 0.0f;");
  w.line("float numer = 0.0f;");
  w.line("for (int k = 0; k < " + std::to_string(K) + "; ++k) {");
  w.indent();
  if (is_causal) {
    w.line("if (k > q) continue;");
  }
  w.line("float dot = 0.0f;");
  w.line("for (int t = 0; t < " + std::to_string(D) + "; ++t) {");
  w.indent();
  w.line("dot += " + query + "[idx4(b,h,q,t," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(D) + ")] * " +
         key + "[idx4(b,h,k,t," + std::to_string(H) + "," + std::to_string(K) + "," + std::to_string(D) + ")];");
  w.dedent();
  w.line("}");
  w.line("float wgt = expf(dot * inv_sqrt_d - max_score);");
  w.line("denom += wgt;");
  w.line("numer += wgt * " + value + "[idx4(b,h,k,d," + std::to_string(H) + "," + std::to_string(K) + "," + std::to_string(DV) + ")];");
  w.dedent();
  w.line("}");
  w.line(out + "[idx4(b,h,q,d," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(DV) + ")] = (denom > 0.0f) ? (numer / denom) : 0.0f;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}
