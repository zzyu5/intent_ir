void emit_reduce_sum(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                     const std::vector<int>& dims, bool keepdims, std::optional<double> scale, const std::string& out_dtype) {
  const int r = static_cast<int>(in_shape.size());
  if (r > 4) fail("reduce_sum supports rank<=4");
  const std::string out_ct = ctype_for_dtype(out_dtype);
  std::vector<int> dims_set = dims;
  std::vector<int64_t> D = in_shape;
  std::vector<int64_t> OD = out_shape;
  const int out_rank = static_cast<int>(out_shape.size());
  if (keepdims) {
    if (out_rank != r) fail("reduce_sum keepdims expects out rank == in rank");
  } else {
    if (out_rank != r - (int)dims_set.size()) fail("reduce_sum out rank mismatch");
  }

  // Fast path: 2D row-reduction over the last axis.
  if (out_ct == "float" && r == 2 && dims_set.size() == 1 && dims_set[0] == 1) {
    int64_t M = D[0], K = D[1];
    std::string scale_val = scale.has_value() ? c_float(scale.value()) : c_float(1.0);
    w.line("intentir_reduce_sum_2d_axis1_f32(" + a + ", " + out + ", " + std::to_string(M) + ", " + std::to_string(K) + ", " + scale_val +
           ", " + std::string(scale.has_value() ? "1" : "0") + ");");
    return;
  }

  // Fast path: 4D reduce over the innermost 2 dims (e.g., [N,G,group_size,HW] -> [N,G,1,1]).
  if (out_ct == "float" && r == 4 && keepdims && dims_set.size() == 2) {
    std::vector<int> ds = dims_set;
    std::sort(ds.begin(), ds.end());
    if (ds[0] == 2 && ds[1] == 3 && OD.size() == 4 && OD[0] == D[0] && OD[1] == D[1] && OD[2] == 1 && OD[3] == 1) {
      int64_t N = D[0], G = D[1], GS = D[2], HW = D[3];
      std::string scale_val = scale.has_value() ? c_float(scale.value()) : c_float(1.0);
      w.line("intentir_reduce_sum_4d_axis23_f32(" + a + ", " + out + ", " + std::to_string(N) + ", " + std::to_string(G) + ", " +
             std::to_string(GS) + ", " + std::to_string(HW) + ", " + scale_val + ", " + std::string(scale.has_value() ? "1" : "0") + ");");
      return;
    }
  }

  std::vector<std::string> out_vars;
  for (int i = 0; i < out_rank; ++i) out_vars.push_back("o" + std::to_string(i));
  for (int i = 0; i < out_rank; ++i) {
    w.line("for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  w.line("double acc = 0.0;");

  // build in_vars mapping
  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) {
    w.line("for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
    w.indent();
  }
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  w.line("acc += (double)" + a + "[" + in_idx + "];");
  for (size_t i = 0; i < dims_set.size(); ++i) {
    w.dedent();
    w.line("}");
  }
  if (scale.has_value()) w.line("acc *= (double)" + c_float(scale.value()) + ";");
  w.line(out + "[" + out_idx + "] = (" + out_ct + ")acc;");
  for (int i = 0; i < out_rank; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_reduce_prod(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                      const std::vector<int>& dims, bool keepdims) {
  const int r = static_cast<int>(in_shape.size());
  if (r > 4) fail("reduce_prod supports rank<=4");
  std::vector<int> dims_set = dims;
  std::vector<int64_t> D = in_shape;
  std::vector<int64_t> OD = out_shape;
  const int out_rank = static_cast<int>(out_shape.size());
  if (keepdims) {
    if (out_rank != r) fail("reduce_prod keepdims expects out rank == in rank");
  } else {
    if (out_rank != r - (int)dims_set.size()) fail("reduce_prod out rank mismatch");
  }

  std::vector<std::string> out_vars;
  for (int i = 0; i < out_rank; ++i) out_vars.push_back("o" + std::to_string(i));
  for (int i = 0; i < out_rank; ++i) {
    w.line("for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  w.line("double acc = 1.0;");

  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) {
    w.line("for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
    w.indent();
  }
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  w.line("acc *= (double)" + a + "[" + in_idx + "];");
  for (size_t i = 0; i < dims_set.size(); ++i) {
    w.dedent();
    w.line("}");
  }
  w.line(out + "[" + out_idx + "] = (float)acc;");
  for (int i = 0; i < out_rank; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_reduce_any(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                     const std::vector<int>& dims, bool keepdims, const std::string& in_dtype, const std::string& out_dtype) {
  const int r = static_cast<int>(in_shape.size());
  if (r > 4) fail("reduce_any supports rank<=4");
  std::vector<int> dims_set = dims;
  std::vector<int64_t> D = in_shape;
  std::vector<int64_t> OD = out_shape;
  const int out_rank = static_cast<int>(out_shape.size());
  if (keepdims) {
    if (out_rank != r) fail("reduce_any keepdims expects out rank == in rank");
  } else {
    if (out_rank != r - (int)dims_set.size()) fail("reduce_any out rank mismatch");
  }

  // Fast path: 2D row-reduction over the last axis (u8) without keepdims.
  if (!keepdims && r == 2 && dims_set.size() == 1 && dims_set[0] == 1 && out_shape.size() == 1 && out_shape[0] == in_shape[0]) {
    const std::string in_ct = ctype_for_dtype(in_dtype);
    const std::string out_ct = ctype_for_dtype(out_dtype);
    if (in_ct == "uint8_t" && out_ct == "uint8_t") {
      int64_t M = in_shape[0], K = in_shape[1];
      w.line("intentir_reduce_any_2d_axis1_u8(" + a + ", " + out + ", " + std::to_string(M) + ", " + std::to_string(K) + ");");
      return;
    }
  }

  std::vector<std::string> out_vars;
  for (int i = 0; i < out_rank; ++i) out_vars.push_back("o" + std::to_string(i));
  for (int i = 0; i < out_rank; ++i) {
    w.line("for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  w.line("uint8_t acc = 0;");
  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) {
    w.line("for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
    w.indent();
  }
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  w.line("acc = acc || (" + a + "[" + in_idx + "] != 0);");
  for (size_t i = 0; i < dims_set.size(); ++i) {
    w.dedent();
    w.line("}");
  }
  w.line(out + "[" + out_idx + "] = acc;");
  for (int i = 0; i < out_rank; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_reduce_max(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                     const std::vector<int>& dims, bool keepdims) {
  const int r = static_cast<int>(in_shape.size());
  if (r > 4) fail("reduce_max supports rank<=4");
  std::vector<int> dims_set = dims;
  std::vector<int64_t> D = in_shape;
  std::vector<int64_t> OD = out_shape;
  const int out_rank = static_cast<int>(out_shape.size());
  if (keepdims) {
    if (out_rank != r) fail("reduce_max keepdims expects out rank == in rank");
  } else {
    if (out_rank != r - (int)dims_set.size()) fail("reduce_max out rank mismatch");
  }

  // Fast path: 2D row-reduction over the last axis.
  if (r == 2 && dims_set.size() == 1 && dims_set[0] == 1) {
    int64_t M = D[0], K = D[1];
    w.line("intentir_reduce_max_2d_axis1_f32(" + a + ", " + out + ", " + std::to_string(M) + ", " + std::to_string(K) + ");");
    return;
  }

  std::vector<std::string> out_vars;
  for (int i = 0; i < out_rank; ++i) out_vars.push_back("o" + std::to_string(i));
  for (int i = 0; i < out_rank; ++i) {
    w.line("for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  w.line("float m = -INFINITY;");
  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) {
    w.line("for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
    w.indent();
  }
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  w.line("float v = " + a + "[" + in_idx + "];");
  w.line("m = fmaxf(m, v);");
  for (size_t i = 0; i < dims_set.size(); ++i) {
    w.dedent();
    w.line("}");
  }
  w.line(out + "[" + out_idx + "] = m;");
  for (int i = 0; i < out_rank; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_reduce_min(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                     const std::vector<int>& dims, bool keepdims) {
  const int r = static_cast<int>(in_shape.size());
  if (r > 4) fail("reduce_min supports rank<=4");
  std::vector<int> dims_set = dims;
  std::vector<int64_t> D = in_shape;
  std::vector<int64_t> OD = out_shape;
  const int out_rank = static_cast<int>(out_shape.size());
  if (keepdims) {
    if (out_rank != r) fail("reduce_min keepdims expects out rank == in rank");
  } else {
    if (out_rank != r - (int)dims_set.size()) fail("reduce_min out rank mismatch");
  }

  std::vector<std::string> out_vars;
  for (int i = 0; i < out_rank; ++i) out_vars.push_back("o" + std::to_string(i));
  for (int i = 0; i < out_rank; ++i) {
    w.line("for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  w.line("float m = INFINITY;");
  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) {
    w.line("for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
    w.indent();
  }
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  w.line("float v = " + a + "[" + in_idx + "];");
  w.line("m = fminf(m, v);");
  for (size_t i = 0; i < dims_set.size(); ++i) {
    w.dedent();
    w.line("}");
  }
  w.line(out + "[" + out_idx + "] = m;");
  for (int i = 0; i < out_rank; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_softmax(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, int axis) {
  const int r = static_cast<int>(in_shape.size());
  if (r == 0) fail("softmax on scalar unsupported");
  int ax = axis;
  if (ax < 0) ax += r;
  if (ax != r - 1) fail("softmax supports axis == last only");
  if (r == 1) {
    int64_t K = in_shape[0];
    w.line("intentir_softmax_1d_last_f32(" + a + ", " + out + ", " + std::to_string(K) + ");");
    return;
  }
  if (r == 2) {
    int64_t M = in_shape[0], K = in_shape[1];
    w.line("intentir_softmax_2d_last_f32(" + a + ", " + out + ", " + std::to_string(M) + ", " + std::to_string(K) + ");");
    return;
  }
  if (r == 3) {
    int64_t A0 = in_shape[0], A1 = in_shape[1], K = in_shape[2];
    w.line("intentir_softmax_3d_last_f32(" + a + ", " + out + ", " + std::to_string(A0) + ", " + std::to_string(A1) + ", " + std::to_string(K) +
           ");");
    return;
  }
  if (r == 4) {
    int64_t B = in_shape[0], H = in_shape[1], Q = in_shape[2], K = in_shape[3];
    w.line("intentir_softmax_4d_last_f32(" + a + ", " + out + ", " + std::to_string(B) + ", " + std::to_string(H) + ", " + std::to_string(Q) + ", " +
           std::to_string(K) + ");");
    return;
  }
  fail("softmax supports rank<=4");
}

void emit_bitwise_left_shift(CodeWriter& w, const std::string& out, const std::string& a_var, const std::string& b_var,
                             const std::string& a_name, const std::string& b_name,
                             const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
                             const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                             const std::string& a_dtype, const std::string& b_dtype, const std::string& out_dtype) {
  if (ctype_for_dtype(a_dtype) != "int32_t" || ctype_for_dtype(b_dtype) != "int32_t" || ctype_for_dtype(out_dtype) != "int32_t") {
    fail("bitwise_left_shift currently supports only i32");
  }
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("bitwise_left_shift supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a_shape, b_name, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b_shape, a_name, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("bitwise_left_shift broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("bitwise_left_shift broadcast mismatch (b)");
  }
  std::vector<std::string> idx = {"i0", "i1", "i2", "i3"};
  idx.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(idx, out_shape);
  auto idx_expr = [&](const std::vector<int64_t>& padded) -> std::string {
    std::vector<std::string> in_vars;
    in_vars.reserve(r);
    for (int i = 0; i < r; ++i) in_vars.push_back(padded[i] == 1 ? "0" : idx[i]);
    return flat_idx_expr(in_vars, padded);
  };
  std::string a_idx = idx_expr(pa);
  std::string b_idx = idx_expr(pb);
  w.line(out + "[" + out_idx + "] = (int32_t)(" + a_var + "[" + a_idx + "] << (" + b_var + "[" + b_idx + "] & 31));");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_bitwise_right_shift(CodeWriter& w, const std::string& out, const std::string& a_var, const std::string& b_var,
                              const std::string& a_name, const std::string& b_name,
                              const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
                              const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                              const std::string& a_dtype, const std::string& b_dtype, const std::string& out_dtype) {
  if (ctype_for_dtype(a_dtype) != "int32_t" || ctype_for_dtype(b_dtype) != "int32_t" || ctype_for_dtype(out_dtype) != "int32_t") {
    fail("bitwise_right_shift currently supports only i32");
  }
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("bitwise_right_shift supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a_shape, b_name, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b_shape, a_name, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("bitwise_right_shift broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("bitwise_right_shift broadcast mismatch (b)");
  }
  std::vector<std::string> idx = {"i0", "i1", "i2", "i3"};
  idx.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(idx, out_shape);
  auto idx_expr = [&](const std::vector<int64_t>& padded) -> std::string {
    std::vector<std::string> in_vars;
    in_vars.reserve(r);
    for (int i = 0; i < r; ++i) in_vars.push_back(padded[i] == 1 ? "0" : idx[i]);
    return flat_idx_expr(in_vars, padded);
  };
  std::string a_idx = idx_expr(pa);
  std::string b_idx = idx_expr(pb);
  w.line(out + "[" + out_idx + "] = (int32_t)(" + a_var + "[" + a_idx + "] >> (" + b_var + "[" + b_idx + "] & 31));");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_arg_reduce_axis1(CodeWriter& w, const std::string& op, const std::string& out, const std::string& a,
                           const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape, int axis) {
  if (in_shape.size() != 2 || out_shape.size() != 1) fail(op + " currently supports rank-2 -> rank-1");
  if (axis < 0) axis += 2;
  if (axis != 1) fail(op + " currently supports axis=1 only");
  if (out_shape[0] != in_shape[0]) fail(op + " output shape mismatch");
  const int64_t M = in_shape[0];
  const int64_t N = in_shape[1];
  w.line("for (int i = 0; i < " + std::to_string(M) + "; ++i) {");
  w.indent();
  w.line("int32_t best_idx = 0;");
  w.line("float best_val = " + a + "[idx2(i,0," + std::to_string(N) + ")];");
  w.line("for (int j = 1; j < " + std::to_string(N) + "; ++j) {");
  w.indent();
  w.line("float v = " + a + "[idx2(i,j," + std::to_string(N) + ")];");
  if (op == "argmax") {
    w.line("if (v > best_val) { best_val = v; best_idx = j; }");
  } else {
    w.line("if (v < best_val) { best_val = v; best_idx = j; }");
  }
  w.dedent();
  w.line("}");
  w.line(out + "[i] = best_idx;");
  w.dedent();
  w.line("}");
}

