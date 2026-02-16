// ---- emit helpers ----

void emit_elemwise_bin(CodeWriter& w, const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                       const std::string& op, const std::string& out, const std::string& a_var, const std::string& b_var,
                       const std::string& a_name, const std::string& b_name,
                       const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
                       const std::string& a_dtype, const std::string& b_dtype, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("elemwise broadcast supports rank<=4");
  // IMPORTANT: use IntentIR tensor names for broadcast alignment heuristics.
  // C variable names (t_*) are not present in intent.tensors, so they can't be
  // used for symbol-aware alignment (e.g., [M] -> [M,N] prefix broadcast).
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a_shape, b_name, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b_shape, a_name, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("elemwise broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("elemwise broadcast mismatch (b)");
  }
  const std::string out_ct = ctype_for_dtype(out_dtype);
  const std::string a_ct = ctype_for_dtype(a_dtype);
  const std::string b_ct = ctype_for_dtype(b_dtype);
  const bool runtime_supported_op = (op == "add" || op == "sub" || op == "mul" || op == "div" || op == "max" || op == "min");
  const bool can_runtime = (out_ct == "float") && (a_ct == "float") && (b_ct == "float") && (r >= 1) && runtime_supported_op;

  if (can_runtime) {
    std::string op_code;
    if (op == "add") op_code = "INTENTIR_F32_BIN_ADD";
    else if (op == "sub") op_code = "INTENTIR_F32_BIN_SUB";
    else if (op == "mul") op_code = "INTENTIR_F32_BIN_MUL";
    else if (op == "div") op_code = "INTENTIR_F32_BIN_DIV";
    else if (op == "max") op_code = "INTENTIR_F32_BIN_MAX";
    else if (op == "min") op_code = "INTENTIR_F32_BIN_MIN";
    else fail("unsupported runtime elemwise op: " + op);

    auto arr = [&](const std::vector<int64_t>& v) -> std::string {
      std::string s = "(int64_t[]){";
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
      }
      s += "}";
      return s;
    };

    w.line("intentir_f32_bin_broadcast(" + a_var + ", " + b_var + ", " + out + ", " + arr(out_shape) + ", " + arr(pa) + ", " + arr(pb) +
           ", " + std::to_string(r) + ", " + op_code + ");");
    return;
  }

  auto emit_scalar = [&]() {
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
    std::string expr;
    if (op == "max" || op == "min") {
      if (out_ct == "float") {
        expr = (op == "max" ? "fmaxf" : "fminf");
        expr += "(" + a_var + "[" + a_idx + "], " + b_var + "[" + b_idx + "])";
      } else if (out_ct == "double") {
        expr = (op == "max" ? "fmax" : "fmin");
        expr += "(" + a_var + "[" + a_idx + "], " + b_var + "[" + b_idx + "])";
      } else {
        expr = "(" + a_var + "[" + a_idx + "] " + (op == "max" ? ">" : "<") + " " + b_var + "[" + b_idx + "]) ? " + a_var + "[" + a_idx + "] : " + b_var + "[" + b_idx + "]";
      }
    } else {
      if (op == "pow") {
        if (out_ct == "float") expr = "powf(" + a_var + "[" + a_idx + "], " + b_var + "[" + b_idx + "])";
        else expr = "pow(" + a_var + "[" + a_idx + "], " + b_var + "[" + b_idx + "])";
      } else if (op == "remainder") {
        const std::string a_expr = a_var + "[" + a_idx + "]";
        const std::string b_expr = b_var + "[" + b_idx + "]";
        expr = "((" + b_expr + ") == 0.0f ? NAN : (" + a_expr + " - floorf((" + a_expr + ")/(" + b_expr + ")) * (" + b_expr + ")))";
      } else {
        const std::string c_op = (op == "add"   ? "+"
                                  : op == "sub" ? "-"
                                  : op == "mul" ? "*"
                                  : op == "div" ? "/"
                                                : "");
        if (c_op.empty()) fail("unsupported elemwise op: " + op);
        expr = "(" + a_var + "[" + a_idx + "] " + c_op + " " + b_var + "[" + b_idx + "])";
      }
    }
    w.line(out + "[" + out_idx + "] = (" + out_ct + ")" + expr + ";");
    for (int i = 0; i < r; ++i) {
      w.dedent();
      w.line("}");
    }
  };
  emit_scalar();
}

void emit_elemwise_scalar_f32(CodeWriter& w, const std::string& op, const std::string& out, const std::string& a,
                              const std::vector<int64_t>& out_shape, double scalar) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("elemwise scalar supports rank<=4");
  const int64_t n = numel(out_shape);
  const std::string s = c_float(scalar);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  w.indent();
  if (op == "add") {
    w.line(out + "[i] = " + a + "[i] + " + s + ";");
  } else if (op == "sub") {
    w.line(out + "[i] = " + a + "[i] - " + s + ";");
  } else if (op == "mul") {
    w.line(out + "[i] = " + a + "[i] * " + s + ";");
  } else if (op == "div") {
    w.line(out + "[i] = " + a + "[i] / " + s + ";");
  } else if (op == "max") {
    w.line(out + "[i] = fmaxf(" + a + "[i], " + s + ");");
  } else if (op == "min") {
    w.line(out + "[i] = fminf(" + a + "[i], " + s + ");");
  } else {
    fail("unsupported elemwise scalar op: " + op);
  }
  w.dedent();
  w.line("}");
}

void emit_cmp(CodeWriter& w, const std::string& cmp_op, const std::string& out, const std::string& a_var, const std::string& b_var,
              const std::string& a_name, const std::string& b_name,
              const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
              const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
              const std::string& a_dtype, const std::string& b_dtype, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("cmp broadcast supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a_shape, b_name, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b_shape, a_name, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("cmp broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("cmp broadcast mismatch (b)");
  }

  const std::string a_ct = ctype_for_dtype(a_dtype);
  const std::string b_ct = ctype_for_dtype(b_dtype);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  const bool can_runtime = (r >= 1) && (out_ct == "uint8_t") && (a_ct == b_ct) && (a_ct == "float" || a_ct == "int32_t") &&
                           (cmp_op == "lt" || cmp_op == "le" || cmp_op == "gt" || cmp_op == "ge");
  if (can_runtime) {
    std::string op_code;
    if (cmp_op == "lt") op_code = "INTENTIR_CMP_LT";
    else if (cmp_op == "le") op_code = "INTENTIR_CMP_LE";
    else if (cmp_op == "gt") op_code = "INTENTIR_CMP_GT";
    else if (cmp_op == "ge") op_code = "INTENTIR_CMP_GE";
    else fail("unsupported cmp op");

    auto arr = [&](const std::vector<int64_t>& v) -> std::string {
      std::string s = "(int64_t[]){";
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
      }
      s += "}";
      return s;
    };

    if (a_ct == "float") {
      w.line("intentir_cmp_f32_broadcast_u8(" + a_var + ", " + b_var + ", " + out + ", " + arr(out_shape) + ", " + arr(pa) + ", " + arr(pb) + ", " +
             std::to_string(r) + ", " + op_code + ");");
      return;
    }
    if (a_ct == "int32_t") {
      w.line("intentir_cmp_i32_broadcast_u8(" + a_var + ", " + b_var + ", " + out + ", " + arr(out_shape) + ", " + arr(pa) + ", " + arr(pb) + ", " +
             std::to_string(r) + ", " + op_code + ");");
      return;
    }
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
  std::string cop;
  if (cmp_op == "eq") cop = "==";
  else if (cmp_op == "lt") cop = "<";
  else if (cmp_op == "le") cop = "<=";
  else if (cmp_op == "gt") cop = ">";
  else if (cmp_op == "ge") cop = ">=";
  else fail("unsupported cmp op");
  w.line(out + "[" + out_idx + "] = (" + a_var + "[" + a_idx + "] " + cop + " " + b_var + "[" + b_idx + "]) ? 1 : 0;");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_ne(CodeWriter& w, const std::string& out, const std::string& a_var, const std::string& b_var,
             const std::string& a_name, const std::string& b_name,
             const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
             const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
             const std::string& a_dtype, const std::string& b_dtype, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("ne broadcast supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a_shape, b_name, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b_shape, a_name, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("ne broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("ne broadcast mismatch (b)");
  }

  const std::string a_ct = ctype_for_dtype(a_dtype);
  const std::string b_ct = ctype_for_dtype(b_dtype);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  const bool can_runtime = (r >= 1) && (out_ct == "uint8_t") && (a_ct == b_ct) && (a_ct == "float" || a_ct == "int32_t");
  if (can_runtime) {
    auto arr = [&](const std::vector<int64_t>& v) -> std::string {
      std::string s = "(int64_t[]){";
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
      }
      s += "}";
      return s;
    };

    if (a_ct == "float") {
      w.line("intentir_cmp_f32_broadcast_u8(" + a_var + ", " + b_var + ", " + out + ", " + arr(out_shape) + ", " + arr(pa) + ", " + arr(pb) + ", " +
             std::to_string(r) + ", INTENTIR_CMP_NE);");
      return;
    }
    if (a_ct == "int32_t") {
      w.line("intentir_cmp_i32_broadcast_u8(" + a_var + ", " + b_var + ", " + out + ", " + arr(out_shape) + ", " + arr(pa) + ", " + arr(pb) + ", " +
             std::to_string(r) + ", INTENTIR_CMP_NE);");
      return;
    }
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
  w.line(out + "[" + out_idx + "] = (" + a_var + "[" + a_idx + "] != " + b_var + "[" + b_idx + "]) ? 1 : 0;");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_bool_bin(CodeWriter& w, const std::string& op, const std::string& out, const std::string& a_var, const std::string& b_var,
                   const std::string& a_name, const std::string& b_name,
                   const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
                   const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                   const std::string& a_dtype, const std::string& b_dtype, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("bool broadcast supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a_shape, b_name, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b_shape, a_name, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("bool broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("bool broadcast mismatch (b)");
  }

  const std::string a_ct = ctype_for_dtype(a_dtype);
  const std::string b_ct = ctype_for_dtype(b_dtype);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  const bool can_runtime = (r >= 1) && (a_ct == "uint8_t") && (b_ct == "uint8_t") && (out_ct == "uint8_t");
  if (can_runtime) {
    std::string op_code = (op == "and") ? "INTENTIR_BOOL_BIN_AND" : (op == "or") ? "INTENTIR_BOOL_BIN_OR" : "";
    if (op_code.empty()) fail("unsupported bool op");

    auto arr = [&](const std::vector<int64_t>& v) -> std::string {
      std::string s = "(int64_t[]){";
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
      }
      s += "}";
      return s;
    };

    w.line("intentir_bool_bin_broadcast_u8(" + a_var + ", " + b_var + ", " + out + ", " + arr(out_shape) + ", " + arr(pa) + ", " + arr(pb) + ", " +
           std::to_string(r) + ", " + op_code + ");");
    return;
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
  if (a_ct == "int32_t" && b_ct == "int32_t" && out_ct == "int32_t") {
    std::string cop = (op == "and") ? "&" : (op == "or") ? "|" : "";
    if (cop.empty()) fail("unsupported bitwise op");
    w.line(out + "[" + out_idx + "] = " + a_var + "[" + a_idx + "] " + cop + " " + b_var + "[" + b_idx + "];");
  } else {
    std::string cop = (op == "and") ? "&&" : (op == "or") ? "||" : "";
    if (cop.empty()) fail("unsupported bool op");
    w.line(out + "[" + out_idx + "] = ((" + a_var + "[" + a_idx + "] != 0) " + cop + " (" + b_var + "[" + b_idx + "] != 0)) ? 1 : 0;");
  }
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_bool_not(
    CodeWriter& w,
    const std::string& out,
    const std::string& a,
    const std::vector<int64_t>& out_shape,
    const std::string& in_dtype,
    const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("not supports rank<=4");
  const std::string in_ct = ctype_for_dtype(in_dtype);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  std::vector<std::string> idx = {"i0", "i1", "i2", "i3"};
  idx.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(idx, out_shape);
  if (in_ct == "int32_t" && out_ct == "int32_t") {
    w.line(out + "[" + out_idx + "] = ~" + a + "[" + out_idx + "];");
  } else {
    w.line(out + "[" + out_idx + "] = (" + a + "[" + out_idx + "] == 0) ? 1 : 0;");
  }
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_unary_abs(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape, const std::string& in_dt, const std::string& out_dt) {
  const int64_t n = numel(out_shape);
  const std::string in_ct = ctype_for_dtype(in_dt);
  const std::string out_ct = ctype_for_dtype(out_dt);
  if (in_ct == "float" && out_ct == "float") {
    w.line("intentir_abs_f32(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ");");
    return;
  }
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  w.indent();
  if (in_ct == "double" && out_ct == "double") {
    w.line(out + "[i] = fabs(" + a + "[i]);");
  } else {
    w.line(in_ct + " x = " + a + "[i];");
    w.line(out + "[i] = (x < 0) ? (" + out_ct + ")(-x) : (" + out_ct + ")x;");
  }
  w.dedent();
  w.line("}");
}

void emit_floor(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("intentir_floor_f32(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ");");
}

void emit_ceil(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = ceilf(" + a + "[i]);");
}

void emit_rsqrt(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("intentir_rsqrt_f32(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ");");
}

void emit_sqrt(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  w.indent();
  w.line("const float __v = " + a + "[i];");
  w.line(out + "[i] = sqrtf(__v);");
  w.dedent();
  w.line("}");
}

void emit_log(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = logf(" + a + "[i]);");
}

void emit_cos(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("intentir_cos_f32(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ");");
}

void emit_sin(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = sinf(" + a + "[i]);");
}

void emit_acos(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = acosf(" + a + "[i]);");
}

void emit_atan(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = atanf(" + a + "[i]);");
}

void emit_tan(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = tanf(" + a + "[i]);");
}

void emit_erf(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("intentir_erf_f32(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ");");
}

void emit_std_axis1(CodeWriter& w, const std::string& out, const std::string& inp, const std::vector<int64_t>& in_shape,
                    const std::vector<int64_t>& out_shape, int axis, bool keepdims, int correction) {
  if (in_shape.size() != 2) fail("std lowering currently supports rank-2 input");
  int axis_norm = axis;
  if (axis_norm < 0) axis_norm += 2;
  if (axis_norm != 1) fail("std lowering currently supports axis=1");
  const int64_t M = in_shape[0];
  const int64_t N = in_shape[1];
  if (out_shape.size() == 1) {
    if (out_shape[0] != M) fail("std output shape mismatch: expect [M]");
  } else if (out_shape.size() == 2) {
    if (!keepdims) fail("std rank-2 output requires keepdims=true");
    if (out_shape[0] != M || out_shape[1] != 1) fail("std output shape mismatch: expect [M,1]");
  } else {
    fail("std output rank must be 1 or 2");
  }
  w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
  w.indent();
  w.line("const float* __row = " + inp + " + (size_t)i * (size_t)" + std::to_string(N) + ";");
  w.line("double __sum = 0.0;");
  w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) __sum += (double)__row[(size_t)j];");
  w.line("const double __mean = __sum / (double)" + std::to_string(N) + ";");
  w.line("double __sq = 0.0;");
  w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
  w.indent();
  w.line("const double __d = (double)__row[(size_t)j] - __mean;");
  w.line("__sq += __d * __d;");
  w.dedent();
  w.line("}");
  w.line("double __den = (double)" + std::to_string(N) + " - (double)" + std::to_string(correction) + ";");
  w.line("if (__den <= 0.0) {");
  w.indent();
  if (out_shape.size() == 1) {
    w.line(out + "[(size_t)i] = nanf(\"\");");
  } else {
    w.line(out + "[(size_t)i * (size_t)" + std::to_string(out_shape[1]) + "] = nanf(\"\");");
  }
  w.line("continue;");
  w.dedent();
  w.line("}");
  w.line("double __var = __sq / __den;");
  w.line("if (__var < 0.0) __var = 0.0;");
  w.line("const float __std = (float)sqrt(__var);");
  if (out_shape.size() == 1) {
    w.line(out + "[(size_t)i] = __std;");
  } else {
    w.line(out + "[(size_t)i * (size_t)" + std::to_string(out_shape[1]) + "] = __std;");
  }
  w.dedent();
  w.line("}");
}

void emit_sort_axis1(CodeWriter& w, const std::string& out, const std::string& inp, const std::vector<int64_t>& in_shape,
                     const std::vector<int64_t>& out_shape, int axis, bool descending) {
  if (in_shape.size() != 2 || out_shape.size() != 2) fail("sort lowering currently supports rank-2 tensors");
  if (in_shape != out_shape) fail("sort output shape mismatch");
  int axis_norm = axis;
  if (axis_norm < 0) axis_norm += 2;
  if (axis_norm != 1) fail("sort lowering currently supports axis=1");
  const int64_t M = in_shape[0];
  const int64_t N = in_shape[1];
  w.line("float* __sort_row = (float*)malloc(sizeof(float) * (size_t)" + std::to_string(N) + ");");
  w.line("if (!__sort_row) { fprintf(stderr, \"alloc failed: sort_row\\n\"); return; }");
  w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
  w.indent();
  w.line("const float* __row = " + inp + " + (size_t)i * (size_t)" + std::to_string(N) + ";");
  w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) __sort_row[(size_t)j] = __row[(size_t)j];");
  w.line("for (int64_t __ii = 1; __ii < " + std::to_string(N) + "; ++__ii) {");
  w.indent();
  w.line("float __v = __sort_row[(size_t)__ii];");
  w.line("int64_t __jj = __ii - 1;");
  w.line("while (__jj >= 0 && __sort_row[(size_t)__jj] > __v) {");
  w.indent();
  w.line("__sort_row[(size_t)(__jj + 1)] = __sort_row[(size_t)__jj];");
  w.line("--__jj;");
  w.dedent();
  w.line("}");
  w.line("__sort_row[(size_t)(__jj + 1)] = __v;");
  w.dedent();
  w.line("}");
  if (descending) {
    w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
    w.indent();
    w.line(out + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)j] = __sort_row[(size_t)(" + std::to_string(N - 1) + " - j)];");
    w.dedent();
    w.line("}");
  } else {
    w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
    w.indent();
    w.line(out + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)j] = __sort_row[(size_t)j];");
    w.dedent();
    w.line("}");
  }
  w.dedent();
  w.line("}");
  w.line("free(__sort_row);");
}

void emit_stack_axis0_2x2d(CodeWriter& w, const std::string& out, const std::string& a, const std::string& b,
                           const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape,
                           const std::vector<int64_t>& out_shape) {
  if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 3) {
    fail("stack lowering currently supports rank-2 inputs -> rank-3 output");
  }
  if (a_shape != b_shape) fail("stack inputs shape mismatch");
  const int64_t M = a_shape[0];
  const int64_t N = a_shape[1];
  if (out_shape[0] != 2 || out_shape[1] != M || out_shape[2] != N) {
    fail("stack output shape mismatch: expect [2,M,N]");
  }
  w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
  w.indent();
  w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
  w.indent();
  w.line("const size_t in_idx = (size_t)i * (size_t)" + std::to_string(N) + " + (size_t)j;");
  w.line(out + "[(size_t)0 * (size_t)" + std::to_string(M * N) + " + in_idx] = " + a + "[in_idx];");
  w.line(out + "[(size_t)1 * (size_t)" + std::to_string(M * N) + " + in_idx] = " + b + "[in_idx];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

void emit_glu(CodeWriter& w, const std::string& out, const std::string& x, const std::vector<int64_t>& in_shape,
              const std::vector<int64_t>& out_shape, int axis) {
  if (in_shape.size() != 2 || out_shape.size() != 2) fail("glu lowering currently supports rank-2 tensors");
  int axis_norm = axis;
  if (axis_norm < 0) axis_norm += 2;
  if (axis_norm != 1) fail("glu lowering currently supports axis=1");
  const int64_t M = in_shape[0];
  const int64_t N = in_shape[1];
  const int64_t N2 = out_shape[1];
  if (out_shape[0] != M) fail("glu output shape mismatch");
  if (N != (2 * N2)) fail("glu expects input axis extent == 2 * output extent");
  w.line("for (int i = 0; i < " + std::to_string(M) + "; ++i) {");
  w.indent();
  w.line("for (int j = 0; j < " + std::to_string(N2) + "; ++j) {");
  w.indent();
  w.line("const int in_base = i * " + std::to_string(N) + ";");
  w.line("const float a = " + x + "[in_base + j];");
  w.line("const float b = " + x + "[in_base + j + " + std::to_string(N2) + "];");
  w.line(out + "[i * " + std::to_string(N2) + " + j] = a * (1.0f / (1.0f + expf(-b)));");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

void emit_cast(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape, const std::string& from_dt, const std::string& to_dt) {
  const int64_t n = numel(out_shape);
  w.line("intentir_cast_1d(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ", " + typecode_for_dtype(from_dt) + ", " + typecode_for_dtype(to_dt) +
         ");");
}

void emit_where(CodeWriter& w, const std::string& out, const std::string& cond, const std::string& x, const std::string& y,
                const std::vector<int64_t>& cond_shape, const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape,
                const std::vector<int64_t>& out_shape, const std::string& cond_dtype, const std::string& x_dtype, const std::string& y_dtype,
                const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("where supports rank<=4");

  // Degenerate form: where(cond, x, x) == broadcast(x). This pattern appears in
  // placeholder index outputs where cond shape may not match output shape.
  if (x == y) {
    if ((int)x_shape.size() > r) fail("where broadcast mismatch (x==y): x rank > out rank");
    std::vector<int64_t> px(r, 1);
    for (int i = 0; i < (int)x_shape.size(); ++i) px[r - (int)x_shape.size() + i] = x_shape[i];
    for (int i = 0; i < r; ++i) {
      if (px[i] != 1 && px[i] != out_shape[i]) fail("where broadcast mismatch (x==y)");
    }
    const std::string out_ct = ctype_for_dtype(out_dtype);
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
    std::string x_idx = idx_expr(px);
    w.line(out + "[" + out_idx + "] = (" + out_ct + ")" + x + "[" + x_idx + "];");
    for (int i = 0; i < r; ++i) {
      w.dedent();
      w.line("}");
    }
    return;
  }

  if ((int)cond_shape.size() > r) fail("where broadcast mismatch: cond rank > out rank");
  if ((int)x_shape.size() > r) fail("where broadcast mismatch: x rank > out rank");
  if ((int)y_shape.size() > r) fail("where broadcast mismatch: y rank > out rank");

  std::vector<int64_t> pc(r, 1), px(r, 1), py(r, 1);
  for (int i = 0; i < (int)cond_shape.size(); ++i) pc[r - (int)cond_shape.size() + i] = cond_shape[i];
  for (int i = 0; i < (int)x_shape.size(); ++i) px[r - (int)x_shape.size() + i] = x_shape[i];
  for (int i = 0; i < (int)y_shape.size(); ++i) py[r - (int)y_shape.size() + i] = y_shape[i];
  for (int i = 0; i < r; ++i) {
    if (pc[i] != 1 && pc[i] != out_shape[i]) fail("where broadcast mismatch (cond)");
    if (px[i] != 1 && px[i] != out_shape[i]) fail("where broadcast mismatch (x)");
    if (py[i] != 1 && py[i] != out_shape[i]) fail("where broadcast mismatch (y)");
  }
  const std::string cond_ct = ctype_for_dtype(cond_dtype);
  const std::string x_ct = ctype_for_dtype(x_dtype);
  const std::string y_ct = ctype_for_dtype(y_dtype);
  std::vector<std::string> idx = {"i0", "i1", "i2", "i3"};
  idx.resize(r);
  const std::string out_ct = ctype_for_dtype(out_dtype);

  const bool can_runtime = (r >= 1) && (cond_ct == "uint8_t") && (x_ct == "float") && (y_ct == "float") && (out_ct == "float");
  if (can_runtime) {
    auto arr = [&](const std::vector<int64_t>& v) -> std::string {
      std::string s = "(int64_t[]){";
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
      }
      s += "}";
      return s;
    };
    w.line("intentir_where_broadcast_f32(" + cond + ", " + x + ", " + y + ", " + out + ", " + arr(out_shape) + ", " + arr(pc) + ", " + arr(px) +
           ", " + arr(py) + ", " + std::to_string(r) + ");");
    return;
  }

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
  std::string c_idx = idx_expr(pc);
  std::string x_idx = idx_expr(px);
  std::string y_idx = idx_expr(py);
  w.line(out + "[" + out_idx + "] = (" + cond + "[" + c_idx + "] != 0) ? (" + out_ct + ")" + x + "[" + x_idx + "] : (" + out_ct + ")" + y + "[" + y_idx + "];");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_cumsum(CodeWriter& w, const std::string& out, const std::string& x, const std::vector<int64_t>& out_shape, int axis, const std::string& dt) {
  if (dt != "f32") fail("cumsum currently supports f32 only");
  const int r = static_cast<int>(out_shape.size());
  if (r == 1) {
    if (axis < 0) axis += 1;
    if (axis != 0) fail("cumsum rank-1 supports axis=0 only");
    w.line("float acc = 0.0f;");
    w.line("for (int i = 0; i < " + std::to_string(out_shape[0]) + "; ++i) {");
    w.indent();
    w.line("acc += " + x + "[i];");
    w.line(out + "[i] = acc;");
    w.dedent();
    w.line("}");
    return;
  }
  if (r == 2) {
    if (axis < 0) axis += 2;
    const int64_t M = out_shape[0];
    const int64_t N = out_shape[1];
    if (axis == 0) {
      w.line("for (int j = 0; j < " + std::to_string(N) + "; ++j) {");
      w.indent();
      w.line("float acc = 0.0f;");
      w.line("for (int i = 0; i < " + std::to_string(M) + "; ++i) {");
      w.indent();
      w.line("const int idx = i * " + std::to_string(N) + " + j;");
      w.line("acc += " + x + "[idx];");
      w.line(out + "[idx] = acc;");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      return;
    }
    if (axis == 1) {
      w.line("for (int i = 0; i < " + std::to_string(M) + "; ++i) {");
      w.indent();
      w.line("float acc = 0.0f;");
      w.line("for (int j = 0; j < " + std::to_string(N) + "; ++j) {");
      w.indent();
      w.line("const int idx = i * " + std::to_string(N) + " + j;");
      w.line("acc += " + x + "[idx];");
      w.line(out + "[idx] = acc;");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      return;
    }
    fail("cumsum rank-2 supports axis=0/1 only");
  }
  fail("cumsum currently supports rank-1/2 only");
}

void emit_cumext_1d(CodeWriter& w, const std::string& out, const std::string& x, const std::vector<int64_t>& out_shape, int axis,
                    const std::string& dt, bool is_max) {
  if (dt != "f32") fail(std::string(is_max ? "cummax" : "cummin") + " currently supports f32 only");
  if (out_shape.size() != 1) fail(std::string(is_max ? "cummax" : "cummin") + " currently supports rank-1 only");
  if (axis < 0) axis += 1;
  if (axis != 0) fail(std::string(is_max ? "cummax" : "cummin") + " rank-1 supports axis=0 only");
  const std::string fun = is_max ? "fmaxf" : "fminf";
  w.line("if (" + std::to_string(out_shape[0]) + " <= 0) return;");
  w.line("float best = " + x + "[0];");
  w.line(out + "[0] = best;");
  w.line("for (int i = 1; i < " + std::to_string(out_shape[0]) + "; ++i) {");
  w.indent();
  w.line("best = " + fun + "(best, " + x + "[i]);");
  w.line(out + "[i] = best;");
  w.dedent();
  w.line("}");
}

void emit_iota(CodeWriter& w, const std::string& out, const std::vector<int64_t>& out_shape, int axis, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r == 0) fail("iota requires rank>=1");
  if (r > 4) fail("iota supports rank<=4");
  if (axis < 0 || axis >= r) fail("iota axis out of range");
  const std::string out_ct = ctype_for_dtype(out_dtype);
  if (out_ct == "int32_t") {
    std::string s = "(int64_t[]){";
    for (size_t i = 0; i < out_shape.size(); ++i) {
      if (i) s += ",";
      s += std::to_string(out_shape[i]);
    }
    s += "}";
    w.line("intentir_iota_i32(" + out + ", " + s + ", " + std::to_string(r) + ", " + std::to_string(axis) + ");");
    return;
  }
  std::vector<std::string> idx = {"i0", "i1", "i2", "i3"};
  idx.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(idx, out_shape);
  w.line(out + "[" + out_idx + "] = (" + out_ct + ")" + idx[axis] + ";");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_gather(CodeWriter& w, const std::string& out, const std::string& data, const std::vector<std::string>& idxs,
                 const std::vector<int64_t>& data_shape, const std::vector<std::vector<int64_t>>& idx_shapes,
                 const std::vector<int64_t>& out_shape, const std::string& data_dtype, const std::vector<std::string>& idx_dtypes,
                 const std::string& out_dtype) {
  const int r_data = static_cast<int>(data_shape.size());
  if (r_data < 1 || r_data > 4) fail("gather supports data rank 1..4");
  if ((int)idxs.size() != r_data) fail("gather expects indices == data rank");
  if ((int)idx_dtypes.size() != r_data) fail("gather expects idx dtypes == data rank");
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("gather out rank<=4");
  // pad idx shapes to out rank
  std::vector<std::vector<int64_t>> padded;
  padded.reserve(idxs.size());
  for (const auto& s : idx_shapes) {
    std::vector<int64_t> p(r, 1);
    for (int i = 0; i < (int)s.size(); ++i) p[r - (int)s.size() + i] = s[i];
    for (int i = 0; i < r; ++i) if (p[i] != 1 && p[i] != out_shape[i]) fail("gather idx broadcast mismatch");
    padded.push_back(std::move(p));
  }

  const std::string data_ct = ctype_for_dtype(data_dtype);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  bool idx_i32 = true;
  for (const auto& dt : idx_dtypes) idx_i32 = idx_i32 && (ctype_for_dtype(dt) == "int32_t");
  const bool can_runtime = (r >= 1) && (data_ct == "float") && (out_ct == "float") && idx_i32;
  if (can_runtime) {
    auto arr64 = [&](const std::vector<int64_t>& v) -> std::string {
      std::string s = "(int64_t[]){";
      for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
      }
      s += "}";
      return s;
    };
    std::string idx_ptrs = "(const int32_t* const[]){";
    for (size_t i = 0; i < idxs.size(); ++i) {
      if (i) idx_ptrs += ",";
      idx_ptrs += idxs[i];
    }
    idx_ptrs += "}";

    std::string flat = "(int64_t[]){";
    bool first = true;
    for (const auto& ps : padded) {
      for (auto d : ps) {
        if (!first) flat += ",";
        first = false;
        flat += std::to_string(d);
      }
    }
    flat += "}";

    w.line("intentir_gather_f32_i32(" + data + ", " + out + ", " + idx_ptrs + ", " + std::to_string(r_data) + ", " + arr64(data_shape) + ", " +
           std::to_string(r_data) + ", " + arr64(out_shape) + ", " + std::to_string(r) + ", " + flat + ");");
    return;
  }

  std::vector<std::string> iv = {"i0", "i1", "i2", "i3"};
  iv.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + iv[i] + " = 0; " + iv[i] + " < " + std::to_string(out_shape[i]) + "; ++" + iv[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(iv, out_shape);
  auto idx_expr = [&](const std::vector<int64_t>& ps) -> std::string {
    std::vector<std::string> vars;
    vars.reserve(r);
    for (int i = 0; i < r; ++i) vars.push_back(ps[i] == 1 ? "0" : iv[i]);
    return flat_idx_expr(vars, ps);
  };
  for (int ax = 0; ax < r_data; ++ax) {
    w.line("int d" + std::to_string(ax) + " = (int)" + idxs[ax] + "[" + idx_expr(padded[ax]) + "];");
  }
  if (r_data == 1) w.line("size_t di = (size_t)d0;");
  else if (r_data == 2) w.line("size_t di = idx2(d0,d1," + std::to_string(data_shape[1]) + ");");
  else if (r_data == 3) w.line("size_t di = idx3(d0,d1,d2," + std::to_string(data_shape[1]) + "," + std::to_string(data_shape[2]) + ");");
  else w.line("size_t di = idx4(d0,d1,d2,d3," + std::to_string(data_shape[1]) + "," + std::to_string(data_shape[2]) + "," + std::to_string(data_shape[3]) + ");");
  w.line(out + "[" + out_idx + "] = (" + out_ct + ")" + data + "[di];");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_index_add_2d(CodeWriter& w, const std::string& out, const std::string& base, const std::string& index, const std::string& src,
                       const std::vector<int64_t>& base_shape, const std::vector<int64_t>& index_shape, const std::vector<int64_t>& src_shape,
                       const std::vector<int64_t>& out_shape, const std::string& idx_dtype, int axis, double alpha) {
  if (base_shape.size() != 2 || out_shape.size() != 2) fail("index_add expects rank-2 base/output");
  if (index_shape.size() != 1) fail("index_add expects rank-1 index");
  if (src_shape.size() != 2) fail("index_add expects rank-2 src");
  if (out_shape != base_shape) fail("index_add output shape mismatch");
  int axis_norm = axis;
  if (axis_norm < 0) axis_norm += 2;
  if (axis_norm != 0) fail("index_add currently supports axis=0 only");
  const std::string idx_ct = ctype_for_dtype(idx_dtype);
  if (!(idx_ct == "int32_t" || idx_ct == "int64_t")) fail("index_add index dtype must be i32/i64");
  const int64_t M = base_shape[0];
  const int64_t N = base_shape[1];
  const int64_t L = index_shape[0];
  if (src_shape[0] != L || src_shape[1] != N) fail("index_add src shape mismatch");

  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(numel(out_shape)) + "; ++i) " + out + "[i] = " + base + "[i];");
  w.line("for (int l = 0; l < " + std::to_string(L) + "; ++l) {");
  w.indent();
  w.line("const int r = (int)" + index + "[l];");
  w.line("if ((unsigned)r >= (unsigned)" + std::to_string(M) + ") continue;");
  w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
  w.indent();
  w.line(out + "[r * " + std::to_string(N) + " + n] += (" + c_float(alpha) + ") * " + src + "[l * " + std::to_string(N) + " + n];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

void emit_index_put_2d(CodeWriter& w, const std::string& out, const std::string& base, const std::string& row_idx, const std::string& col_idx,
                       const std::string& values, const std::vector<int64_t>& base_shape, const std::vector<int64_t>& row_shape,
                       const std::vector<int64_t>& col_shape, const std::vector<int64_t>& values_shape, const std::vector<int64_t>& out_shape,
                       const std::string& row_dtype, const std::string& col_dtype, bool accumulate) {
  if (base_shape.size() != 2 || out_shape.size() != 2) fail("index_put expects rank-2 base/output");
  if (row_shape.size() != 1 || col_shape.size() != 1 || values_shape.size() != 1) fail("index_put expects rank-1 indices/values");
  if (out_shape != base_shape) fail("index_put output shape mismatch");
  const std::string row_ct = ctype_for_dtype(row_dtype);
  const std::string col_ct = ctype_for_dtype(col_dtype);
  if (!((row_ct == "int32_t" || row_ct == "int64_t") && (col_ct == "int32_t" || col_ct == "int64_t"))) {
    fail("index_put row/col dtype must be i32/i64");
  }
  const int64_t M = base_shape[0];
  const int64_t N = base_shape[1];
  const int64_t L = row_shape[0];
  if (col_shape[0] != L || values_shape[0] != L) fail("index_put index/value length mismatch");

  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(numel(out_shape)) + "; ++i) " + out + "[i] = " + base + "[i];");
  w.line("for (int l = 0; l < " + std::to_string(L) + "; ++l) {");
  w.indent();
  w.line("const int r = (int)" + row_idx + "[l];");
  w.line("const int c = (int)" + col_idx + "[l];");
  w.line("if ((unsigned)r >= (unsigned)" + std::to_string(M) + " || (unsigned)c >= (unsigned)" + std::to_string(N) + ") continue;");
  w.line("const size_t pos = (size_t)r * (size_t)" + std::to_string(N) + " + (size_t)c;");
  if (accumulate) {
    w.line(out + "[pos] += " + values + "[l];");
  } else {
    w.line(out + "[pos] = " + values + "[l];");
  }
  w.dedent();
  w.line("}");
}

void emit_masked_select_flat(CodeWriter& w, const std::string& out, const std::string& inp, const std::string& mask,
                             const std::vector<int64_t>& inp_shape, const std::vector<int64_t>& mask_shape,
                             const std::vector<int64_t>& out_shape, const std::string& inp_dtype,
                             const std::string& mask_dtype, const std::string& out_dtype) {
  if (inp_shape != mask_shape) fail("masked_select expects input/mask shapes to match");
  if (out_shape.size() != 1) fail("masked_select currently expects rank-1 output");
  const std::string inp_ct = ctype_for_dtype(inp_dtype);
  const std::string mask_ct = ctype_for_dtype(mask_dtype);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  if (inp_ct != out_ct) fail("masked_select currently expects input/output dtype to match");
  if (mask_ct != "uint8_t") fail("masked_select expects bool/u8 mask");
  const int64_t total = numel(inp_shape);
  const int64_t L = out_shape[0];
  w.line("int64_t out_pos = 0;");
  w.line("for (int64_t i = 0; i < " + std::to_string(total) + "; ++i) {");
  w.indent();
  w.line("if (" + mask + "[i] != 0) {");
  w.indent();
  w.line("if (out_pos < " + std::to_string(L) + ") " + out + "[out_pos] = " + inp + "[i];");
  w.line("out_pos += 1;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("for (int64_t i = out_pos; i < " + std::to_string(L) + "; ++i) " + out + "[i] = (" + out_ct + ")0;");
}

void emit_masked_scatter_flat(CodeWriter& w, const std::string& out, const std::string& inp, const std::string& mask,
                              const std::string& src, const std::vector<int64_t>& inp_shape,
                              const std::vector<int64_t>& mask_shape, const std::vector<int64_t>& src_shape,
                              const std::vector<int64_t>& out_shape, const std::string& inp_dtype,
                              const std::string& mask_dtype, const std::string& src_dtype, const std::string& out_dtype) {
  if (inp_shape != mask_shape) fail("masked_scatter expects input/mask shapes to match");
  if (out_shape != inp_shape) fail("masked_scatter expects output shape to match input shape");
  if (src_shape.size() != 1) fail("masked_scatter currently expects rank-1 source tensor");
  const std::string inp_ct = ctype_for_dtype(inp_dtype);
  const std::string mask_ct = ctype_for_dtype(mask_dtype);
  const std::string src_ct = ctype_for_dtype(src_dtype);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  if (inp_ct != out_ct || src_ct != out_ct) fail("masked_scatter expects input/source/output dtype to match");
  if (mask_ct != "uint8_t") fail("masked_scatter expects bool/u8 mask");
  const int64_t total = numel(inp_shape);
  const int64_t L = src_shape[0];
  w.line("for (int64_t i = 0; i < " + std::to_string(total) + "; ++i) " + out + "[i] = " + inp + "[i];");
  w.line("int64_t src_pos = 0;");
  w.line("for (int64_t i = 0; i < " + std::to_string(total) + "; ++i) {");
  w.indent();
  w.line("if (" + mask + "[i] != 0) {");
  w.indent();
  w.line("if (src_pos < " + std::to_string(L) + ") " + out + "[i] = " + src + "[src_pos];");
  w.line("src_pos += 1;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

void emit_kron_2d(CodeWriter& w, const std::string& out, const std::string& a, const std::string& b, const std::vector<int64_t>& a_shape,
                  const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape) {
  if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 2) fail("kron currently supports rank-2 tensors");
  const int64_t M = a_shape[0], N = a_shape[1];
  const int64_t P = b_shape[0], Q = b_shape[1];
  if (out_shape[0] != M * P || out_shape[1] != N * Q) fail("kron output shape mismatch");
  const int64_t NQ = out_shape[1];
  w.line("for (int i = 0; i < " + std::to_string(M) + "; ++i) {");
  w.indent();
  w.line("for (int j = 0; j < " + std::to_string(N) + "; ++j) {");
  w.indent();
  w.line("const float av = " + a + "[i * " + std::to_string(N) + " + j];");
  w.line("for (int p = 0; p < " + std::to_string(P) + "; ++p) {");
  w.indent();
  w.line("for (int q = 0; q < " + std::to_string(Q) + "; ++q) {");
  w.indent();
  w.line("const size_t oi = (size_t)(i * " + std::to_string(P) + " + p);");
  w.line("const size_t oj = (size_t)(j * " + std::to_string(Q) + " + q);");
  w.line(out + "[oi * (size_t)" + std::to_string(NQ) + " + oj] = av * " + b + "[p * " + std::to_string(Q) + " + q];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
}

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

