void emit_transpose_4d_0132(CodeWriter& w, const std::string& out, const std::string& inp, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape) {
  if (in_shape.size() != 4 || out_shape.size() != 4) fail("transpose expects rank-4");
  int64_t B = in_shape[0], H = in_shape[1], K = in_shape[2], D = in_shape[3];
  w.line("for (int b = 0; b < " + std::to_string(B) + "; ++b) {");
  w.indent();
  w.line("for (int h = 0; h < " + std::to_string(H) + "; ++h) {");
  w.indent();
  w.line("for (int k = 0; k < " + std::to_string(K) + "; ++k) {");
  w.indent();
  w.line("for (int d = 0; d < " + std::to_string(D) + "; ++d) {");
  w.indent();
  w.line(out + "[idx4(b,h,d,k," + std::to_string(H) + "," + std::to_string(out_shape[2]) + "," + std::to_string(out_shape[3]) + ")] = " + inp + "[idx4(b,h,k,d," + std::to_string(H) + "," + std::to_string(K) + "," + std::to_string(D) + ")];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

void emit_transpose_generic(CodeWriter& w, const std::string& out, const std::string& inp,
                            const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                            const std::vector<int>& perm) {
  const int r = static_cast<int>(in_shape.size());
  if (static_cast<int>(out_shape.size()) != r) fail("transpose expects same-rank shapes");
  if (static_cast<int>(perm.size()) != r) fail("transpose perm rank mismatch");
  if (r > 4) fail("transpose supports rank<=4");
  std::vector<int> seen(r, 0);
  for (int p : perm) {
    if (p < 0 || p >= r) fail("transpose perm out of range");
    seen[p] += 1;
  }
  for (int i = 0; i < r; ++i) if (seen[i] != 1) fail("transpose perm must be a permutation");

  std::vector<std::string> iv = {"i0", "i1", "i2", "i3"};
  iv.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + iv[i] + " = 0; " + iv[i] + " < " + std::to_string(out_shape[i]) + "; ++" + iv[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(iv, out_shape);
  std::vector<std::string> in_iv(r, "0");
  for (int od = 0; od < r; ++od) {
    in_iv[perm[od]] = iv[od];
  }
  std::string in_idx = flat_idx_expr(in_iv, in_shape);
  w.line(out + "[" + out_idx + "] = " + inp + "[" + in_idx + "];");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_matmul(CodeWriter& w, const std::string& out, const std::string& a, const std::string& b,
                 const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
                 bool transpose_a, bool transpose_b,
                 std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k) {
  // Matmul is implemented as a target-side runtime primitive (RVV/scalar), to keep
  // the host codegen "compiler-like" (lowering + calls), not a giant template.
  if (a_shape.size() == 2 && b_shape.size() == 2) {
    int64_t M = transpose_a ? a_shape[1] : a_shape[0];
    int64_t K = transpose_a ? a_shape[0] : a_shape[1];
    int64_t K2 = transpose_b ? b_shape[1] : b_shape[0];
    int64_t N = transpose_b ? b_shape[0] : b_shape[1];
    if (K2 != K) fail("matmul shape mismatch (2D)");
    if (out_shape.size() != 2 || out_shape[0] != M || out_shape[1] != N) fail("matmul output shape mismatch (2D)");
    const int64_t tm = (tile_m && *tile_m > 0) ? *tile_m : 0;
    const int64_t tn = (tile_n && *tile_n > 0) ? *tile_n : 0;
    const int64_t tk = (tile_k && *tile_k > 0) ? *tile_k : 0;
    w.line("intentir_matmul_2d_f32(" + a + ", " + b + ", " + out + ", " + std::to_string(M) + ", " + std::to_string(N) +
           ", " + std::to_string(K) + ", " + (transpose_a ? "1" : "0") + ", " + (transpose_b ? "1" : "0") + ", " +
           std::to_string(tm) + ", " + std::to_string(tn) + ", " + std::to_string(tk) + ");");
    return;
  }
  if (a_shape.size() == 2 && b_shape.size() == 1) {
    if (transpose_b) fail("matmul rank-2 x rank-1 does not support transpose_b");
    int64_t M = transpose_a ? a_shape[1] : a_shape[0];
    int64_t K = transpose_a ? a_shape[0] : a_shape[1];
    int64_t K2 = b_shape[0];
    if (K2 != K) fail("matmul shape mismatch (2D x 1D)");
    if (out_shape.size() != 1 || out_shape[0] != M) fail("matmul output shape mismatch (2D x 1D)");
    for (int64_t i = 0; i < M; ++i) {
      w.line("{");
      w.indent();
      w.line("double acc = 0.0;");
      for (int64_t k = 0; k < K; ++k) {
        const int64_t a_idx = transpose_a ? (k * M + i) : (i * K + k);
        w.line("acc += (double)" + a + "[" + std::to_string(a_idx) + "] * (double)" + b + "[" + std::to_string(k) + "];");
      }
      w.line(out + "[" + std::to_string(i) + "] = (float)acc;");
      w.dedent();
      w.line("}");
    }
    return;
  }
  if (a_shape.size() == 3 && b_shape.size() == 3 && out_shape.size() == 3) {
    int64_t B0 = a_shape[0];
    int64_t M = transpose_a ? a_shape[2] : a_shape[1];
    int64_t K = transpose_a ? a_shape[1] : a_shape[2];
    int64_t B2 = b_shape[0];
    int64_t K2 = transpose_b ? b_shape[2] : b_shape[1];
    int64_t N = transpose_b ? b_shape[1] : b_shape[2];
    if (B2 != B0 || K2 != K) fail("matmul shape mismatch (3D)");
    if (out_shape[0] != B0 || out_shape[1] != M || out_shape[2] != N) fail("matmul output shape mismatch (3D)");
    const int64_t tm = (tile_m && *tile_m > 0) ? *tile_m : 0;
    const int64_t tn = (tile_n && *tile_n > 0) ? *tile_n : 0;
    const int64_t tk = (tile_k && *tile_k > 0) ? *tile_k : 0;
    w.line("intentir_matmul_3d_f32(" + a + ", " + b + ", " + out + ", " + std::to_string(B0) + ", " + std::to_string(M) + ", " +
           std::to_string(N) + ", " + std::to_string(K) + ", " + (transpose_a ? "1" : "0") + ", " + (transpose_b ? "1" : "0") + ", " +
           std::to_string(tm) + ", " + std::to_string(tn) + ", " + std::to_string(tk) + ");");
    return;
  }
  if (a_shape.size() == 4 && b_shape.size() == 4 && out_shape.size() == 4) {
    int64_t B0 = a_shape[0], H0 = a_shape[1];
    int64_t M = transpose_a ? a_shape[3] : a_shape[2];
    int64_t K = transpose_a ? a_shape[2] : a_shape[3];
    int64_t B2 = b_shape[0], H2 = b_shape[1];
    int64_t K2 = transpose_b ? b_shape[3] : b_shape[2];
    int64_t N = transpose_b ? b_shape[2] : b_shape[3];
    if (B2 != B0 || H2 != H0 || K2 != K) fail("matmul shape mismatch (4D)");
    if (out_shape[0] != B0 || out_shape[1] != H0 || out_shape[2] != M || out_shape[3] != N) fail("matmul output shape mismatch (4D)");
    const int64_t tm = (tile_m && *tile_m > 0) ? *tile_m : 0;
    const int64_t tn = (tile_n && *tile_n > 0) ? *tile_n : 0;
    const int64_t tk = (tile_k && *tile_k > 0) ? *tile_k : 0;
    w.line("intentir_matmul_4d_f32(" + a + ", " + b + ", " + out + ", " + std::to_string(B0) + ", " + std::to_string(H0) +
           ", " + std::to_string(M) + ", " + std::to_string(N) + ", " + std::to_string(K) + ", " + (transpose_a ? "1" : "0") +
           ", " + (transpose_b ? "1" : "0") + ", " + std::to_string(tm) + ", " + std::to_string(tn) + ", " + std::to_string(tk) +
           ");");
    return;
  }
  fail("matmul supports rank-2/3/4 and rank-2 x rank-1");
}

