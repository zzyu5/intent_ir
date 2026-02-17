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

