#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "code_writer.h"

using json = nlohmann::json;

namespace {

struct Tensor {
  std::string dtype;
  std::vector<json> shape;  // dims are int or string (symbol)
};

struct Op {
  std::string op;
  std::vector<std::string> inputs;
  std::string output;
  json attrs;
};

struct Intent {
  std::string name;
  std::unordered_map<std::string, Tensor> tensors;
  std::vector<Op> ops;
  std::vector<std::string> outputs;
  json schedule;  // optional ScheduleSketch (tile/vec hints), may be null/object
};

struct ConstVal {
  std::string dtype;
  double value = 0.0;
};

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error(msg);
}

std::string c_float(double x) {
  if (std::isnan(x)) return "NAN";
  if (std::isinf(x)) return x > 0 ? "INFINITY" : "(-INFINITY)";
  std::ostringstream oss;
  oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
  oss.precision(10);
  oss << x;
  std::string s = oss.str();
  bool has_dot = (s.find('.') != std::string::npos) || (s.find('e') != std::string::npos) || (s.find('E') != std::string::npos);
  if (!has_dot) s += ".0";
  return s + "f";
}

std::string ctype_for_dtype(const std::string& dt) {
  if (dt == "bool" || dt == "i1" || dt == "u8") return "uint8_t";
  if (dt == "i8") return "int8_t";
  if (dt == "i32") return "int32_t";
  if (dt == "i64") return "int64_t";
  if (dt == "f64") return "double";
  return "float";
}

std::string flat_idx_expr(const std::vector<std::string>& vars, const std::vector<int64_t>& shape) {
  const int r = static_cast<int>(shape.size());
  if (r == 0) return "0";
  if (r == 1) return "(size_t)" + vars[0];
  if (r == 2) return "idx2(" + vars[0] + "," + vars[1] + "," + std::to_string(shape[1]) + ")";
  if (r == 3) return "idx3(" + vars[0] + "," + vars[1] + "," + vars[2] + "," + std::to_string(shape[1]) + "," + std::to_string(shape[2]) + ")";
  if (r == 4) return "idx4(" + vars[0] + "," + vars[1] + "," + vars[2] + "," + vars[3] + "," + std::to_string(shape[1]) + "," + std::to_string(shape[2]) + "," + std::to_string(shape[3]) + ")";
  fail("indexing supports rank<=4");
}

int64_t numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (auto d : shape) n *= static_cast<int64_t>(d);
  return n;
}

std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const int ra = static_cast<int>(a.size());
  const int rb = static_cast<int>(b.size());
  const int r = std::max(ra, rb);
  std::vector<int64_t> aa(r, 1), bb(r, 1);
  for (int i = 0; i < ra; ++i) aa[r - ra + i] = a[i];
  for (int i = 0; i < rb; ++i) bb[r - rb + i] = b[i];
  std::vector<int64_t> out;
  out.reserve(r);
  for (int i = 0; i < r; ++i) {
    const int64_t da = aa[i], db = bb[i];
    if (da == db) out.push_back(da);
    else if (da == 1) out.push_back(db);
    else if (db == 1) out.push_back(da);
    else fail("broadcast shape mismatch");
  }
  return out;
}

bool is_digits(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::optional<int64_t> resolve_dim_token(const json& tok, const std::unordered_map<std::string, int64_t>& bindings) {
  if (tok.is_number_integer()) return tok.get<int64_t>();
  if (tok.is_number()) return static_cast<int64_t>(tok.get<double>());
  if (tok.is_string()) {
    std::string s = tok.get<std::string>();
    if (is_digits(s)) return std::stoll(s);
    auto it = bindings.find(s);
    if (it != bindings.end()) return it->second;
  }
  return std::nullopt;
}

std::optional<int> match_axis_1d_to_tensor(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                           const std::string& vec_name, const std::string& tensor_name, int tensor_rank) {
  auto vit = intent.tensors.find(vec_name);
  auto tit = intent.tensors.find(tensor_name);
  if (vit == intent.tensors.end() || tit == intent.tensors.end()) return std::nullopt;
  const auto& vshape = vit->second.shape;
  const auto& tshape = tit->second.shape;
  if (vshape.size() != 1 || static_cast<int>(tshape.size()) != tensor_rank) return std::nullopt;

  const json& v0 = vshape[0];
  // 1) Exact symbol match (strongest).
  if (v0.is_string()) {
    std::string sym = v0.get<std::string>();
    if (!is_digits(sym)) {
      std::vector<int> hits;
      for (int i = 0; i < tensor_rank; ++i) {
        if (tshape[i].is_string() && tshape[i].get<std::string>() == sym) hits.push_back(i);
      }
      if (hits.size() == 1) return hits[0];
    }
  }
  // 2) Resolved numeric match (weaker; can be ambiguous when symbols bind equal).
  auto vnum = resolve_dim_token(v0, bindings);
  if (!vnum) return std::nullopt;
  std::vector<int> hits;
  for (int i = 0; i < tensor_rank; ++i) {
    auto tnum = resolve_dim_token(tshape[i], bindings);
    if (tnum && *tnum == *vnum) hits.push_back(i);
  }
  if (hits.size() == 1) return hits[0];
  return std::nullopt;
}

std::vector<int64_t> pad_for_broadcast(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                       const std::string& name, const std::vector<int64_t>& shape,
                                       const std::string& other_name, const std::vector<int64_t>& other_shape, int out_rank) {
  std::vector<int64_t> padded(out_rank, 1);
  for (int i = 0; i < (int)shape.size(); ++i) padded[out_rank - (int)shape.size() + i] = shape[i];

  if ((int)shape.size() == 1 && out_rank >= 2 && (int)other_shape.size() == out_rank) {
    auto ax = match_axis_1d_to_tensor(intent, bindings, name, other_name, out_rank);
    if (ax && *ax >= 0 && *ax < out_rank) {
      if (shape[0] == other_shape[*ax]) {
        std::vector<int64_t> named(out_rank, 1);
        named[*ax] = shape[0];
        return named;
      }
    }
  }
  return padded;
}

std::vector<int64_t> broadcast_shape_named(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                           const std::string& a_name, const std::string& b_name,
                                           const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const int ra = static_cast<int>(a.size());
  const int rb = static_cast<int>(b.size());
  const int r = std::max(ra, rb);
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a, b_name, b, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b, a_name, a, r);
  std::vector<int64_t> out;
  out.reserve(r);
  for (int i = 0; i < r; ++i) {
    const int64_t da = pa[i], db = pb[i];
    if (da == db) out.push_back(da);
    else if (da == 1) out.push_back(db);
    else if (db == 1) out.push_back(da);
    else fail("broadcast shape mismatch");
  }
  return out;
}

std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) fail("cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

json read_json_file(const std::string& path) {
  return json::parse(read_text_file(path));
}

Intent parse_intent(const json& j) {
  Intent out;
  out.name = j.value("name", "intent_fn");
  if (!j.contains("tensors") || !j["tensors"].is_object()) fail("intent.tensors must be object");
  for (auto it = j["tensors"].begin(); it != j["tensors"].end(); ++it) {
    const std::string name = it.key();
    const json& tj = it.value();
    Tensor t;
    t.dtype = tj.value("dtype", "f32");
    if (!tj.contains("shape") || !tj["shape"].is_array()) fail("tensor.shape must be list");
    for (const auto& d : tj["shape"]) t.shape.push_back(d);
    out.tensors.emplace(name, std::move(t));
  }
  if (!j.contains("ops") || !j["ops"].is_array()) fail("intent.ops must be list");
  for (const auto& oj : j["ops"]) {
    Op op;
    op.op = oj.value("op", "");
    if (!oj.contains("inputs") || !oj["inputs"].is_array()) fail("op.inputs must be list");
    for (const auto& x : oj["inputs"]) op.inputs.push_back(x.get<std::string>());
    op.output = oj.value("output", "");
    op.attrs = oj.value("attrs", json::object());
    out.ops.push_back(std::move(op));
  }
  if (!j.contains("outputs") || !j["outputs"].is_array()) fail("intent.outputs must be list");
  for (const auto& x : j["outputs"]) out.outputs.push_back(x.get<std::string>());
  if (j.contains("schedule")) out.schedule = j["schedule"];
  return out;
}

// ---- small expression evaluator for const values (supports +,-,*,/,(), symbols, numbers) ----

enum class TokKind { End, Number, Ident, Plus, Minus, Star, Slash, LParen, RParen };

struct Tok {
  TokKind kind;
  double num = 0.0;
  std::string ident;
};

struct Lexer {
  explicit Lexer(std::string s) : src(std::move(s)) {}
  std::string src;
  size_t i = 0;

  void skip_ws() {
    while (i < src.size() && (src[i] == ' ' || src[i] == '\t' || src[i] == '\n' || src[i] == '\r')) ++i;
  }

  Tok next() {
    skip_ws();
    if (i >= src.size()) return {TokKind::End};
    char c = src[i];
    if (c == '+') { ++i; return {TokKind::Plus}; }
    if (c == '-') { ++i; return {TokKind::Minus}; }
    if (c == '*') { ++i; return {TokKind::Star}; }
    if (c == '/') { ++i; return {TokKind::Slash}; }
    if (c == '(') { ++i; return {TokKind::LParen}; }
    if (c == ')') { ++i; return {TokKind::RParen}; }
    if ((c >= '0' && c <= '9') || c == '.') {
      size_t start = i;
      while (i < src.size()) {
        char d = src[i];
        if ((d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') {
          // stop '+'/'-' unless part of exponent; crude but ok for our use
          if ((d == '+' || d == '-') && i > start && (src[i-1] != 'e' && src[i-1] != 'E')) break;
          ++i;
          continue;
        }
        break;
      }
      double v = std::stod(src.substr(start, i - start));
      Tok t{TokKind::Number};
      t.num = v;
      return t;
    }
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') {
      size_t start = i;
      ++i;
      while (i < src.size()) {
        char d = src[i];
        if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') || (d >= '0' && d <= '9') || d == '_') ++i;
        else break;
      }
      Tok t{TokKind::Ident};
      t.ident = src.substr(start, i - start);
      return t;
    }
    fail(std::string("unexpected char in expr: ") + c);
  }
};

struct Parser {
  explicit Parser(std::string s, const std::unordered_map<std::string, double>& vars) : lex(std::move(s)), vars(vars) {
    cur = lex.next();
  }
  Lexer lex;
  Tok cur;
  const std::unordered_map<std::string, double>& vars;

  void eat(TokKind k) {
    if (cur.kind != k) fail("expr parse error");
    cur = lex.next();
  }

  double parse_expr() {
    double v = parse_term();
    while (cur.kind == TokKind::Plus || cur.kind == TokKind::Minus) {
      TokKind k = cur.kind;
      eat(k);
      double rhs = parse_term();
      v = (k == TokKind::Plus) ? (v + rhs) : (v - rhs);
    }
    return v;
  }

  double parse_term() {
    double v = parse_factor();
    while (cur.kind == TokKind::Star || cur.kind == TokKind::Slash) {
      TokKind k = cur.kind;
      eat(k);
      double rhs = parse_factor();
      v = (k == TokKind::Star) ? (v * rhs) : (v / rhs);
    }
    return v;
  }

  double parse_factor() {
    if (cur.kind == TokKind::Plus) { eat(TokKind::Plus); return parse_factor(); }
    if (cur.kind == TokKind::Minus) { eat(TokKind::Minus); return -parse_factor(); }
    if (cur.kind == TokKind::Number) {
      double v = cur.num;
      eat(TokKind::Number);
      return v;
    }
    if (cur.kind == TokKind::Ident) {
      std::string name = cur.ident;
      eat(TokKind::Ident);
      auto it = vars.find(name);
      if (it != vars.end()) return it->second;
      if (name == "eps") return 1e-5;
      fail("unresolved symbol in const expr: " + name);
    }
    if (cur.kind == TokKind::LParen) {
      eat(TokKind::LParen);
      double v = parse_expr();
      eat(TokKind::RParen);
      return v;
    }
    fail("invalid expr");
  }
};

double eval_expr(const std::string& expr, const std::unordered_map<std::string, double>& vars) {
  Parser p(expr, vars);
  double v = p.parse_expr();
  if (p.cur.kind != TokKind::End) fail("trailing tokens in expr");
  return v;
}

double resolve_const_value(const json& v, const std::unordered_map<std::string, int64_t>& bindings) {
  if (v.is_number()) return v.get<double>();
  if (v.is_string()) {
    std::string s = v.get<std::string>();
    if (s == "eps") return 1e-5;
    // numeric literal?
    try {
      size_t pos = 0;
      double x = std::stod(s, &pos);
      if (pos == s.size()) return x;
    } catch (...) {
    }
    // symbol binding?
    auto it = bindings.find(s);
    if (it != bindings.end()) return static_cast<double>(it->second);
    // expression
    std::unordered_map<std::string, double> vars;
    for (auto& kv : bindings) vars.emplace(kv.first, static_cast<double>(kv.second));
    return eval_expr(s, vars);
  }
  fail("unsupported const value type");
}

// ---- emit helpers ----

void emit_elemwise_bin(CodeWriter& w, const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                       const std::string& op, const std::string& out, const std::string& a, const std::string& b,
                       const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
                       const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("elemwise broadcast supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a, a_shape, b, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b, b_shape, a, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("elemwise broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("elemwise broadcast mismatch (b)");
  }
  const std::string out_ct = ctype_for_dtype(out_dtype);
  const int64_t n_total = numel(out_shape);
  const bool is_contig_elemwise = (pa == out_shape) && (pb == out_shape);

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
        expr += "(" + a + "[" + a_idx + "], " + b + "[" + b_idx + "])";
      } else if (out_ct == "double") {
        expr = (op == "max" ? "fmax" : "fmin");
        expr += "(" + a + "[" + a_idx + "], " + b + "[" + b_idx + "])";
      } else {
        expr = "(" + a + "[" + a_idx + "] " + (op == "max" ? ">" : "<") + " " + b + "[" + b_idx + "]) ? " + a + "[" + a_idx + "] : " + b + "[" + b_idx + "]";
      }
    } else {
      const std::string c_op = (op == "add"   ? "+"
                                : op == "sub" ? "-"
                                : op == "mul" ? "*"
                                : op == "div" ? "/"
                                              : "");
      if (c_op.empty()) fail("unsupported elemwise op: " + op);
      expr = "(" + a + "[" + a_idx + "] " + c_op + " " + b + "[" + b_idx + "])";
    }
    w.line(out + "[" + out_idx + "] = (" + out_ct + ")" + expr + ";");
    for (int i = 0; i < r; ++i) {
      w.dedent();
      w.line("}");
    }
  };

  const bool can_vector = (out_ct == "float") && is_contig_elemwise && (n_total > 0) &&
                          (op == "add" || op == "sub" || op == "mul" || op == "div" || op == "max" || op == "min");
  if (!can_vector) {
    emit_scalar();
    return;
  }

  w.pp_line("#if defined(__riscv_vector) || defined(__riscv_v)");
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n_total) + "; ) {");
  w.indent();
  w.line("size_t vl = intentir_vsetvl_e32m1((size_t)" + std::to_string(n_total) + " - i);");
  w.line("vfloat32m1_t va = __riscv_vle32_v_f32m1(&" + a + "[i], vl);");
  w.line("vfloat32m1_t vb = __riscv_vle32_v_f32m1(&" + b + "[i], vl);");
  if (op == "add") w.line("vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);");
  else if (op == "sub") w.line("vfloat32m1_t vc = __riscv_vfsub_vv_f32m1(va, vb, vl);");
  else if (op == "mul") w.line("vfloat32m1_t vc = __riscv_vfmul_vv_f32m1(va, vb, vl);");
  else if (op == "div") w.line("vfloat32m1_t vc = __riscv_vfdiv_vv_f32m1(va, vb, vl);");
  else if (op == "max") w.line("vfloat32m1_t vc = __riscv_vfmax_vv_f32m1(va, vb, vl);");
  else if (op == "min") w.line("vfloat32m1_t vc = __riscv_vfmin_vv_f32m1(va, vb, vl);");
  w.line("__riscv_vse32_v_f32m1(&" + out + "[i], vc, vl);");
  w.line("i += vl;");
  w.dedent();
  w.line("}");
  w.pp_line("#else");
  emit_scalar();
  w.pp_line("#endif");
}

void emit_cmp(CodeWriter& w, const std::string& cmp_op, const std::string& out, const std::string& a, const std::string& b,
              const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
              const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("cmp broadcast supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a, a_shape, b, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b, b_shape, a, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("cmp broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("cmp broadcast mismatch (b)");
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
  if (cmp_op == "lt") cop = "<";
  else if (cmp_op == "le") cop = "<=";
  else if (cmp_op == "gt") cop = ">";
  else if (cmp_op == "ge") cop = ">=";
  else fail("unsupported cmp op");
  w.line(out + "[" + out_idx + "] = (" + a + "[" + a_idx + "] " + cop + " " + b + "[" + b_idx + "]) ? 1 : 0;");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_ne(CodeWriter& w, const std::string& out, const std::string& a, const std::string& b,
             const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
             const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("ne broadcast supports rank<=4");
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a, a_shape, b, b_shape, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b, b_shape, a, a_shape, r);
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("ne broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("ne broadcast mismatch (b)");
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
  w.line(out + "[" + out_idx + "] = (" + a + "[" + a_idx + "] != " + b + "[" + b_idx + "]) ? 1 : 0;");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_bool_bin(CodeWriter& w, const std::string& op, const std::string& out, const std::string& a, const std::string& b,
                   const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("bool broadcast supports rank<=4");
  std::vector<int64_t> pa(r, 1), pb(r, 1);
  for (int i = 0; i < (int)a_shape.size(); ++i) pa[r - (int)a_shape.size() + i] = a_shape[i];
  for (int i = 0; i < (int)b_shape.size(); ++i) pb[r - (int)b_shape.size() + i] = b_shape[i];
  for (int i = 0; i < r; ++i) {
    if (pa[i] != 1 && pa[i] != out_shape[i]) fail("bool broadcast mismatch (a)");
    if (pb[i] != 1 && pb[i] != out_shape[i]) fail("bool broadcast mismatch (b)");
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
  std::string cop = (op == "and") ? "&&" : (op == "or") ? "||" : "";
  if (cop.empty()) fail("unsupported bool op");
  w.line(out + "[" + out_idx + "] = ((" + a + "[" + a_idx + "] != 0) " + cop + " (" + b + "[" + b_idx + "] != 0)) ? 1 : 0;");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_bool_not(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("not supports rank<=4");
  std::vector<std::string> idx = {"i0", "i1", "i2", "i3"};
  idx.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(idx, out_shape);
  w.line(out + "[" + out_idx + "] = (" + a + "[" + out_idx + "] == 0) ? 1 : 0;");
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
    w.pp_line("#if defined(__riscv_vector) || defined(__riscv_v)");
    w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ) {");
    w.indent();
    w.line("size_t vl = intentir_vsetvl_e32m1((size_t)" + std::to_string(n) + " - i);");
    w.line("vfloat32m1_t vx = __riscv_vle32_v_f32m1(&" + a + "[i], vl);");
    w.line("vfloat32m1_t vy = __riscv_vfabs_v_f32m1(vx, vl);");
    w.line("__riscv_vse32_v_f32m1(&" + out + "[i], vy, vl);");
    w.line("i += vl;");
    w.dedent();
    w.line("}");
    w.pp_line("#else");
    w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) { " + out + "[i] = fabsf(" + a + "[i]); }");
    w.pp_line("#endif");
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
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  w.indent();
  w.line(out + "[i] = floorf(" + a + "[i]);");
  w.dedent();
  w.line("}");
}

void emit_rsqrt(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.pp_line("#if defined(__riscv_vector) || defined(__riscv_v)");
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ) {");
  w.indent();
  w.line("size_t vl = intentir_vsetvl_e32m1((size_t)" + std::to_string(n) + " - i);");
  w.line("vfloat32m1_t vx = __riscv_vle32_v_f32m1(&" + a + "[i], vl);");
  w.line("vfloat32m1_t vs = __riscv_vfsqrt_v_f32m1(vx, vl);");
  w.line("vfloat32m1_t vy = __riscv_vfrdiv_vf_f32m1(vs, 1.0f, vl);");
  w.line("__riscv_vse32_v_f32m1(&" + out + "[i], vy, vl);");
  w.line("i += vl;");
  w.dedent();
  w.line("}");
  w.pp_line("#else");
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) { " + out + "[i] = 1.0f / sqrtf(" + a + "[i]); }");
  w.pp_line("#endif");
}

void emit_cast(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape, const std::string& from_dt, const std::string& to_dt) {
  const int64_t n = numel(out_shape);
  const std::string from_ct = ctype_for_dtype(from_dt);
  const std::string to_ct = ctype_for_dtype(to_dt);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  w.indent();
  if (to_ct == "uint8_t") {
    w.line(out + "[i] = (" + a + "[i] != 0) ? 1 : 0;");
  } else if (from_ct == "uint8_t" && (to_ct == "float" || to_ct == "double")) {
    const std::string one = (to_ct == "float") ? "1.0f" : "1.0";
    const std::string zero = (to_ct == "float") ? "0.0f" : "0.0";
    w.line(out + "[i] = (" + a + "[i] != 0) ? " + one + " : " + zero + ";");
  } else {
    w.line(out + "[i] = (" + to_ct + ")(" + a + "[i]);");
  }
  w.dedent();
  w.line("}");
}

void emit_where(CodeWriter& w, const std::string& out, const std::string& cond, const std::string& x, const std::string& y,
                const std::vector<int64_t>& cond_shape, const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape,
                const std::vector<int64_t>& out_shape, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("where supports rank<=4");
  std::vector<int64_t> pc(r, 1), px(r, 1), py(r, 1);
  for (int i = 0; i < (int)cond_shape.size(); ++i) pc[r - (int)cond_shape.size() + i] = cond_shape[i];
  for (int i = 0; i < (int)x_shape.size(); ++i) px[r - (int)x_shape.size() + i] = x_shape[i];
  for (int i = 0; i < (int)y_shape.size(); ++i) py[r - (int)y_shape.size() + i] = y_shape[i];
  for (int i = 0; i < r; ++i) {
    if (pc[i] != 1 && pc[i] != out_shape[i]) fail("where broadcast mismatch (cond)");
    if (px[i] != 1 && px[i] != out_shape[i]) fail("where broadcast mismatch (x)");
    if (py[i] != 1 && py[i] != out_shape[i]) fail("where broadcast mismatch (y)");
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
  std::string c_idx = idx_expr(pc);
  std::string x_idx = idx_expr(px);
  std::string y_idx = idx_expr(py);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  w.line(out + "[" + out_idx + "] = (" + cond + "[" + c_idx + "] != 0) ? (" + out_ct + ")" + x + "[" + x_idx + "] : (" + out_ct + ")" + y + "[" + y_idx + "];");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_iota(CodeWriter& w, const std::string& out, const std::vector<int64_t>& out_shape, int axis, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r == 0) fail("iota requires rank>=1");
  if (r > 4) fail("iota supports rank<=4");
  if (axis < 0 || axis >= r) fail("iota axis out of range");
  std::vector<std::string> idx = {"i0", "i1", "i2", "i3"};
  idx.resize(r);
  for (int i = 0; i < r; ++i) {
    w.line("for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
    w.indent();
  }
  std::string out_idx = flat_idx_expr(idx, out_shape);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  w.line(out + "[" + out_idx + "] = (" + out_ct + ")" + idx[axis] + ";");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_gather(CodeWriter& w, const std::string& out, const std::string& data, const std::vector<std::string>& idxs,
                 const std::vector<int64_t>& data_shape, const std::vector<std::vector<int64_t>>& idx_shapes,
                 const std::vector<int64_t>& out_shape, const std::string& out_dtype) {
  const int r_data = static_cast<int>(data_shape.size());
  if (r_data < 1 || r_data > 4) fail("gather supports data rank 1..4");
  if ((int)idxs.size() != r_data) fail("gather expects indices == data rank");
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
  const std::string out_ct = ctype_for_dtype(out_dtype);
  w.line(out + "[" + out_idx + "] = (" + out_ct + ")" + data + "[di];");
  for (int i = 0; i < r; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_reduce_sum(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                     const std::vector<int>& dims, bool keepdims, std::optional<double> scale) {
  const int r = static_cast<int>(in_shape.size());
  if (r > 4) fail("reduce_sum supports rank<=4");
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
  if (r == 2 && dims_set.size() == 1 && dims_set[0] == 1) {
    int64_t M = D[0], K = D[1];
    std::string scale_val = scale.has_value() ? c_float(scale.value()) : c_float(1.0);
    w.line("intentir_reduce_sum_2d_axis1_f32(" + a + ", " + out + ", " + std::to_string(M) + ", " + std::to_string(K) + ", " + scale_val +
           ", " + std::string(scale.has_value() ? "1" : "0") + ");");
    return;
  }

  // Fast path: 4D reduce over the innermost 2 dims (e.g., [N,G,group_size,HW] -> [N,G,1,1]).
  if (r == 4 && keepdims && dims_set.size() == 2) {
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
  w.line(out + "[" + out_idx + "] = (float)acc;");
  for (int i = 0; i < out_rank; ++i) {
    w.dedent();
    w.line("}");
  }
}

void emit_reduce_any(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
                     const std::vector<int>& dims, bool keepdims) {
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
                 std::optional<int64_t> tile_m, std::optional<int64_t> tile_n, std::optional<int64_t> tile_k,
                 std::optional<int64_t> vec_width) {
  // Scalar matmul (rank-2 or rank-4).
  if (a_shape.size() == 2 && b_shape.size() == 2) {
    int64_t M = transpose_a ? a_shape[1] : a_shape[0];
    int64_t K = transpose_a ? a_shape[0] : a_shape[1];
    int64_t K2 = transpose_b ? b_shape[1] : b_shape[0];
    int64_t N = transpose_b ? b_shape[0] : b_shape[1];
    if (K2 != K) fail("matmul shape mismatch (2D)");
    if (out_shape.size() != 2 || out_shape[0] != M || out_shape[1] != N) fail("matmul output shape mismatch (2D)");

    const int64_t tm = (tile_m && *tile_m > 0) ? std::min<int64_t>(*tile_m, M) : M;
    const int64_t tn = (tile_n && *tile_n > 0) ? std::min<int64_t>(*tile_n, N) : N;
    const int64_t tk = (tile_k && *tile_k > 0) ? std::min<int64_t>(*tile_k, K) : K;
    if (!transpose_a) {
      w.pp_line("#if defined(__riscv_vector) || defined(__riscv_v)");
      w.line("for (int m_base = 0; m_base < " + std::to_string(M) + "; m_base += " + std::to_string(tm) + ") {");
      w.indent();
      w.line("int m_end = m_base + " + std::to_string(tm) + "; if (m_end > " + std::to_string(M) + ") m_end = " + std::to_string(M) + ";");
      w.line("for (int n_base = 0; n_base < " + std::to_string(N) + "; n_base += " + std::to_string(tn) + ") {");
      w.indent();
      w.line("int n_end = n_base + " + std::to_string(tn) + "; if (n_end > " + std::to_string(N) + ") n_end = " + std::to_string(N) + ";");
      w.line("for (int m = m_base; m < m_end; ++m) {");
      w.indent();
      w.line("for (int n0 = n_base; n0 < n_end; ) {");
      w.indent();
      w.line("size_t rem = (size_t)(n_end - n0);");
      w.line("size_t vl = intentir_vsetvl_e32m1(rem);");
      w.line("vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vl);");
      w.line("for (int k_base = 0; k_base < " + std::to_string(K) + "; k_base += " + std::to_string(tk) + ") {");
      w.indent();
      w.line("int k_end = k_base + " + std::to_string(tk) + "; if (k_end > " + std::to_string(K) + ") k_end = " + std::to_string(K) + ";");
      w.line("for (int k = k_base; k < k_end; ++k) {");
      w.indent();
      w.line("float av = " + a + "[idx2(m,k," + std::to_string(K) + ")];");
      if (!transpose_b) {
        w.line("vfloat32m1_t vb = __riscv_vle32_v_f32m1(&" + b + "[idx2(k,n0," + std::to_string(N) + ")], vl);");
      } else {
        w.line("vfloat32m1_t vb = __riscv_vlse32_v_f32m1(&" + b + "[idx2(n0,k," + std::to_string(K) + ")], (ptrdiff_t)((size_t)" + std::to_string(K) + " * sizeof(float)), vl);");
      }
      w.line("vacc = __riscv_vfmacc_vf_f32m1(vacc, av, vb, vl);");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.line("__riscv_vse32_v_f32m1(&" + out + "[idx2(m,n0," + std::to_string(N) + ")], vacc, vl);");
      w.line("n0 += (int)vl;");
      w.dedent();
      w.line("}");
	      w.dedent();
	      w.line("}");
	      w.dedent();
	      w.line("}");
	      w.dedent();
	      w.line("}");
	      w.pp_line("#else");
	    }
    w.line("for (int m_base = 0; m_base < " + std::to_string(M) + "; m_base += " + std::to_string(tm) + ") {");
    w.indent();
    w.line("int m_end = m_base + " + std::to_string(tm) + "; if (m_end > " + std::to_string(M) + ") m_end = " + std::to_string(M) + ";");
    w.line("for (int n_base = 0; n_base < " + std::to_string(N) + "; n_base += " + std::to_string(tn) + ") {");
    w.indent();
    w.line("int n_end = n_base + " + std::to_string(tn) + "; if (n_end > " + std::to_string(N) + ") n_end = " + std::to_string(N) + ";");
    w.line("for (int m = m_base; m < m_end; ++m) {");
    w.indent();
    w.line("for (int n = n_base; n < n_end; ++n) {");
    w.indent();
    w.line("double acc = 0.0;");
    w.line("for (int k_base = 0; k_base < " + std::to_string(K) + "; k_base += " + std::to_string(tk) + ") {");
    w.indent();
    w.line("int k_end = k_base + " + std::to_string(tk) + "; if (k_end > " + std::to_string(K) + ") k_end = " + std::to_string(K) + ";");
    w.line("for (int k = k_base; k < k_end; ++k) {");
    w.indent();
    std::string a_idx = transpose_a ? ("idx2(k,m," + std::to_string(M) + ")") : ("idx2(m,k," + std::to_string(K) + ")");
    std::string b_idx = transpose_b ? ("idx2(n,k," + std::to_string(K) + ")") : ("idx2(k,n," + std::to_string(N) + ")");
    w.line("acc += (double)" + a + "[" + a_idx + "] * (double)" + b + "[" + b_idx + "];");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.line(out + "[idx2(m,n," + std::to_string(N) + ")] = (float)acc;");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
	    w.dedent();
	    w.line("}");
	    w.dedent();
	    w.line("}");
	    if (!transpose_a) w.pp_line("#endif");
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

    const int64_t tm = (tile_m && *tile_m > 0) ? std::min<int64_t>(*tile_m, M) : M;
    const int64_t tn = (tile_n && *tile_n > 0) ? std::min<int64_t>(*tile_n, N) : N;
    const int64_t tk = (tile_k && *tile_k > 0) ? std::min<int64_t>(*tile_k, K) : K;
    if (!transpose_a) {
      w.pp_line("#if defined(__riscv_vector) || defined(__riscv_v)");
      w.line("for (int b0 = 0; b0 < " + std::to_string(B0) + "; ++b0) {");
      w.indent();
      w.line("for (int h0 = 0; h0 < " + std::to_string(H0) + "; ++h0) {");
      w.indent();
      w.line("for (int m_base = 0; m_base < " + std::to_string(M) + "; m_base += " + std::to_string(tm) + ") {");
      w.indent();
      w.line("int m_end = m_base + " + std::to_string(tm) + "; if (m_end > " + std::to_string(M) + ") m_end = " + std::to_string(M) + ";");
      w.line("for (int n_base = 0; n_base < " + std::to_string(N) + "; n_base += " + std::to_string(tn) + ") {");
      w.indent();
      w.line("int n_end = n_base + " + std::to_string(tn) + "; if (n_end > " + std::to_string(N) + ") n_end = " + std::to_string(N) + ";");
      w.line("for (int m0 = m_base; m0 < m_end; ++m0) {");
      w.indent();
      w.line("for (int n0 = n_base; n0 < n_end; ) {");
      w.indent();
      w.line("size_t rem = (size_t)(n_end - n0);");
      w.line("size_t vl = intentir_vsetvl_e32m1(rem);");
      w.line("vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vl);");
      w.line("for (int k_base = 0; k_base < " + std::to_string(K) + "; k_base += " + std::to_string(tk) + ") {");
      w.indent();
      w.line("int k_end = k_base + " + std::to_string(tk) + "; if (k_end > " + std::to_string(K) + ") k_end = " + std::to_string(K) + ";");
      w.line("for (int k0 = k_base; k0 < k_end; ++k0) {");
      w.indent();
      w.line("float av = " + a + "[idx4(b0,h0,m0,k0," + std::to_string(H0) + "," + std::to_string(M) + "," + std::to_string(K) + ")];");
      if (!transpose_b) {
        w.line("vfloat32m1_t vb = __riscv_vle32_v_f32m1(&" + b + "[idx4(b0,h0,k0,n0," + std::to_string(H0) + "," + std::to_string(K) + "," + std::to_string(N) + ")], vl);");
      } else {
        w.line("vfloat32m1_t vb = __riscv_vlse32_v_f32m1(&" + b + "[idx4(b0,h0,n0,k0," + std::to_string(H0) + "," + std::to_string(N) + "," + std::to_string(K) + ")], (ptrdiff_t)((size_t)" + std::to_string(K) + " * sizeof(float)), vl);");
      }
      w.line("vacc = __riscv_vfmacc_vf_f32m1(vacc, av, vb, vl);");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.line("__riscv_vse32_v_f32m1(&" + out + "[idx4(b0,h0,m0,n0," + std::to_string(H0) + "," + std::to_string(M) + "," + std::to_string(N) + ")], vacc, vl);");
      w.line("n0 += (int)vl;");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.pp_line("#else");
    }
    w.line("for (int b0 = 0; b0 < " + std::to_string(B0) + "; ++b0) {");
    w.indent();
    w.line("for (int h0 = 0; h0 < " + std::to_string(H0) + "; ++h0) {");
    w.indent();
    w.line("for (int m_base = 0; m_base < " + std::to_string(M) + "; m_base += " + std::to_string(tm) + ") {");
    w.indent();
    w.line("int m_end = m_base + " + std::to_string(tm) + "; if (m_end > " + std::to_string(M) + ") m_end = " + std::to_string(M) + ";");
    w.line("for (int n_base = 0; n_base < " + std::to_string(N) + "; n_base += " + std::to_string(tn) + ") {");
    w.indent();
    w.line("int n_end = n_base + " + std::to_string(tn) + "; if (n_end > " + std::to_string(N) + ") n_end = " + std::to_string(N) + ";");
    w.line("for (int m0 = m_base; m0 < m_end; ++m0) {");
    w.indent();
    w.line("for (int n0 = n_base; n0 < n_end; ++n0) {");
    w.indent();
    w.line("double acc = 0.0;");
    w.line("for (int k_base = 0; k_base < " + std::to_string(K) + "; k_base += " + std::to_string(tk) + ") {");
    w.indent();
    w.line("int k_end = k_base + " + std::to_string(tk) + "; if (k_end > " + std::to_string(K) + ") k_end = " + std::to_string(K) + ";");
    w.line("for (int k0 = k_base; k0 < k_end; ++k0) {");
    w.indent();
    std::string a_idx = transpose_a
                            ? ("idx4(b0,h0,k0,m0," + std::to_string(H0) + "," + std::to_string(K) + "," + std::to_string(M) + ")")
                            : ("idx4(b0,h0,m0,k0," + std::to_string(H0) + "," + std::to_string(M) + "," + std::to_string(K) + ")");
    std::string b_idx = transpose_b
                            ? ("idx4(b0,h0,n0,k0," + std::to_string(H0) + "," + std::to_string(N) + "," + std::to_string(K) + ")")
                            : ("idx4(b0,h0,k0,n0," + std::to_string(H0) + "," + std::to_string(K) + "," + std::to_string(N) + ")");
    w.line("acc += (double)" + a + "[" + a_idx + "] * (double)" + b + "[" + b_idx + "];");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.line(out + "[idx4(b0,h0,m0,n0," + std::to_string(H0) + "," + std::to_string(M) + "," + std::to_string(N) + ")] = (float)acc;");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    if (!transpose_a) w.pp_line("#endif");
    return;
  }
  fail("matmul supports rank-2 or rank-4");
}

struct CProgramEmitter {
  CodeWriter& w;
  const Intent& intent;
  const std::unordered_map<std::string, int64_t>& bindings;
  const std::vector<std::string>& external_inputs;
  const std::unordered_map<std::string, std::vector<int64_t>>& shape_env;
  const std::unordered_map<std::string, std::string>& dtype_env;
  const std::unordered_map<std::string, ConstVal>& const_vals;
  std::optional<int64_t> sched_tile_m;
  std::optional<int64_t> sched_tile_n;
  std::optional<int64_t> sched_tile_k;
  std::optional<int64_t> sched_vec_width;
  double atol = 1e-3;
  double rtol = 1e-3;
  double matmul_flops_total = 0.0;

	  void emit_program() {
	    emit_prelude();
	    emit_globals();
	    w.line("static void intentir_compute(void);");
	    w.blank();
	    emit_main();
	    w.blank();
	    emit_compute_fn();
	  }

	  void emit_prelude() {
	    int64_t vw = 0;
	    if (sched_vec_width && *sched_vec_width > 0) vw = *sched_vec_width;
	    w.pp_line("#ifndef INTENTIR_VEC_WIDTH");
	    w.pp_line("#define INTENTIR_VEC_WIDTH " + std::to_string(vw));
	    w.pp_line("#endif");
	    w.pp_line("#include \"intentir_driver.h\"");
	    w.pp_line("#include \"intentir_ops.h\"");
	    w.blank();
	  }

  void emit_globals() {
    std::vector<std::string> names;
    names.reserve(shape_env.size());
    for (const auto& kv : shape_env) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    w.line("// intent tensors (globals)");
    for (const auto& name : names) {
      std::string ct = ctype_for_dtype(dtype_env.at(name));
      w.line("static " + ct + "* " + name + " = NULL;");
    }
    w.blank();
  }

	  void emit_alloc_tensor(const std::string& name) {
	    const auto& shp = shape_env.at(name);
	    int64_t n = numel(shp);
	    std::string ct = ctype_for_dtype(dtype_env.at(name));
	    w.line(name + " = (" + ct + "*)malloc(sizeof(" + ct + ") * (size_t)" + std::to_string(n) + ");");
    w.line("if (!" + name + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
  }

	  void emit_main() {
	    w.line("int main() {");
	    w.indent();

	    w.line("setvbuf(stdout, NULL, _IONBF, 0);");
	    w.line("setvbuf(stderr, NULL, _IONBF, 0);");
	    w.line("const float ATOL = " + c_float(atol) + ";");
	    w.line("const float RTOL = " + c_float(rtol) + ";");
	    {
	      std::ostringstream oss;
	      oss.setf(std::ios::fixed);
	      oss.precision(1);
	      oss << matmul_flops_total;
	      w.line("const double MATMUL_FLOPS = (double)" + oss.str() + ";");
	    }
	    w.blank();

	    // inputs
	    if (!external_inputs.empty()) {
	      w.line("IntentirBufferDesc inputs[] = {");
	      w.indent();
	      for (const auto& name : external_inputs) {
	        const auto& shp = shape_env.at(name);
	        int64_t n = numel(shp);
	        std::string ct = ctype_for_dtype(dtype_env.at(name));
	        std::string dt = dtype_env.at(name);
	        std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
	        if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
	        w.line("{\"" + name + "\", (void**)&" + name + ", (size_t)(" + bytes_expr + "), " +
	               ((dt == "bool" || dt == "i1") ? "INTENTIR_DTYPE_U8" : "INTENTIR_DTYPE_F32") + "},");
	      }
	      w.dedent();
	      w.line("};");
	      w.line("if (!intentir_alloc_and_load_inputs(inputs, (size_t)(sizeof(inputs) / sizeof(inputs[0])))) return 2;");
	      w.blank();
	    }

	    // consts as 1-element arrays (deterministic order for reproducible C).
	    {
	      std::vector<std::string> cnames;
      cnames.reserve(const_vals.size());
      for (const auto& kv : const_vals) cnames.push_back(kv.first);
      std::sort(cnames.begin(), cnames.end());
      for (const auto& name : cnames) {
        const ConstVal& cv = const_vals.at(name);
        const std::string ct = ctype_for_dtype(cv.dtype);
        w.line(name + " = (" + ct + "*)malloc(sizeof(" + ct + "));");
        w.line("if (!" + name + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
        if (ct == "float") {
          w.line(name + "[0] = " + c_float(cv.value) + ";");
        } else if (ct == "double") {
          std::ostringstream oss;
          oss.precision(17);
          oss << cv.value;
          w.line(name + "[0] = (double)" + oss.str() + ";");
        } else {
          w.line(name + "[0] = (" + ct + ")" + std::to_string((int64_t)cv.value) + ";");
        }
        w.blank();
      }
	    }

	    // allocate outputs / set alias views
	    for (const auto& op : intent.ops) {
	      if (op.op == "const") continue;
      const std::string& out = op.output;
      if (op.op == "reshape" || op.op == "identity" || op.op == "layout_cast") {
        std::string ct = ctype_for_dtype(dtype_env.at(out));
        w.line("// " + op.op + " alias");
        w.line(out + " = (" + ct + "*)" + op.inputs[0] + ";");
        w.blank();
        continue;
      }
      w.line("// op " + op.op + " -> " + out);
      emit_alloc_tensor(out);
      w.blank();
	    }

	    // compute once
	    w.line("intentir_compute();");
	    w.line("intentir_maybe_bench(intentir_compute, MATMUL_FLOPS);");
	    w.blank();

	    // compare outputs
	    if (intent.outputs.empty()) {
	      w.line("int ok = 1;");
	    } else {
	      w.line("IntentirBufferDesc outputs[] = {");
	      w.indent();
	      for (const auto& name : intent.outputs) {
	        const auto& shp = shape_env.at(name);
	        int64_t n = numel(shp);
	        std::string ct = ctype_for_dtype(dtype_env.at(name));
	        std::string dt = dtype_env.at(name);
	        std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
	        if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
	        w.line("{\"" + name + "\", (void**)&" + name + ", (size_t)(" + bytes_expr + "), " +
	               ((dt == "bool" || dt == "i1") ? "INTENTIR_DTYPE_U8" : "INTENTIR_DTYPE_F32") + "},");
	      }
	      w.dedent();
	      w.line("};");
	      w.line("int ok = intentir_compare_outputs_with_refs(outputs, (size_t)(sizeof(outputs) / sizeof(outputs[0])), ATOL, RTOL);");
	    }
	    w.line("printf(ok ? \"PASS lowered\\n\" : \"FAIL lowered\\n\");");
	    w.line("return ok ? 0 : 1;");

	    w.dedent();
	    w.line("}");
	  }

  void emit_broadcast_in_dim(const Op& op, const std::vector<int64_t>& out_shape) {
    const std::string& out = op.output;
    const std::string& inp = op.inputs[0];
    const auto& in_shape = shape_env.at(inp);
    std::vector<int> bcast_dims;
    for (const auto& d : op.attrs["broadcast_dims"]) bcast_dims.push_back(d.get<int>());

    std::vector<int64_t> strides(in_shape.size(), 1);
    int64_t s = 1;
    for (int i = (int)in_shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= in_shape[i];
    }
    const int r_out = (int)out_shape.size();
    if (r_out > 4) fail("broadcast_in_dim supports rank<=4");
    std::vector<std::string> iv = {"i0", "i1", "i2", "i3"};
    iv.resize(r_out);
    for (int i = 0; i < r_out; ++i) {
      w.line("for (int " + iv[i] + " = 0; " + iv[i] + " < " + std::to_string(out_shape[i]) + "; ++" + iv[i] + ") {");
      w.indent();
    }
    std::string out_idx = flat_idx_expr(iv, out_shape);
    std::vector<std::string> terms;
    for (int in_dim = 0; in_dim < (int)in_shape.size(); ++in_dim) {
      int od = bcast_dims[in_dim];
      if (in_shape[in_dim] == 1) terms.push_back("0");
      else terms.push_back("((size_t)" + iv[od] + " * (size_t)" + std::to_string(strides[in_dim]) + ")");
    }
    std::string in_idx = terms.empty() ? "0" : terms[0];
    for (size_t t = 1; t < terms.size(); ++t) in_idx += " + " + terms[t];
    w.line(out + "[" + out_idx + "] = " + inp + "[" + in_idx + "];");
    for (int i = 0; i < r_out; ++i) {
      w.dedent();
      w.line("}");
    }
  }

  void emit_compute_fn() {
    w.line("static void intentir_compute(void) {");
    w.indent();
    for (const auto& op : intent.ops) {
      if (op.op == "const") continue;
      if (op.op == "reshape" || op.op == "identity" || op.op == "layout_cast") continue;
      const std::string& out = op.output;
      const std::vector<int64_t>& out_shape = shape_env.at(out);
      w.line("// op " + op.op + " -> " + out);

      if (op.op == "transpose") {
        const auto& in_shape = shape_env.at(op.inputs[0]);
        std::vector<int> perm;
        for (const auto& p : op.attrs["perm"]) perm.push_back(p.get<int>());
        if (perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2) {
          emit_transpose_4d_0132(w, out, op.inputs[0], in_shape, out_shape);
        } else {
          emit_transpose_generic(w, out, op.inputs[0], in_shape, out_shape, perm);
        }
      } else if (op.op == "broadcast_in_dim") {
        emit_broadcast_in_dim(op, out_shape);
      } else if (op.op == "add" || op.op == "sub" || op.op == "mul" || op.op == "div" || op.op == "max" || op.op == "min") {
        emit_elemwise_bin(w, intent, bindings, op.op, out, op.inputs[0], op.inputs[1], shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, dtype_env.at(out));
      } else if (op.op == "ne") {
        emit_ne(w, out, op.inputs[0], op.inputs[1], shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings);
      } else if (op.op == "lt" || op.op == "le" || op.op == "gt" || op.op == "ge") {
        emit_cmp(w, op.op, out, op.inputs[0], op.inputs[1], shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings);
      } else if (op.op == "and" || op.op == "or") {
        emit_bool_bin(w, op.op, out, op.inputs[0], op.inputs[1], shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape);
      } else if (op.op == "not") {
        emit_bool_not(w, out, op.inputs[0], out_shape);
      } else if (op.op == "rsqrt") {
        emit_rsqrt(w, out, op.inputs[0], out_shape);
      } else if (op.op == "abs") {
        emit_unary_abs(w, out, op.inputs[0], out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
      } else if (op.op == "floor") {
        emit_floor(w, out, op.inputs[0], out_shape);
      } else if (op.op == "cast") {
        emit_cast(w, out, op.inputs[0], out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
      } else if (op.op == "where") {
        emit_where(w, out, op.inputs[0], op.inputs[1], op.inputs[2], shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), shape_env.at(op.inputs[2]), out_shape, dtype_env.at(out));
      } else if (op.op == "iota") {
        int axis = op.attrs.value("axis", 0);
        emit_iota(w, out, out_shape, axis, dtype_env.at(out));
      } else if (op.op == "gather") {
        std::vector<std::string> idxs(op.inputs.begin() + 1, op.inputs.end());
        std::vector<std::vector<int64_t>> idx_shapes;
        for (const auto& nm : idxs) idx_shapes.push_back(shape_env.at(nm));
        emit_gather(w, out, op.inputs[0], idxs, shape_env.at(op.inputs[0]), idx_shapes, out_shape, dtype_env.at(out));
      } else if (op.op == "reduce_sum") {
        std::vector<int> dims;
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        bool keepdims = op.attrs.value("keepdims", false);
        std::optional<double> scale;
        if (op.attrs.contains("scale")) scale = resolve_const_value(op.attrs["scale"], bindings);
        emit_reduce_sum(w, out, op.inputs[0], shape_env.at(op.inputs[0]), out_shape, dims, keepdims, scale);
      } else if (op.op == "reduce_max") {
        std::vector<int> dims;
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        bool keepdims = op.attrs.value("keepdims", false);
        emit_reduce_max(w, out, op.inputs[0], shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
      } else if (op.op == "reduce_any") {
        std::vector<int> dims;
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        bool keepdims = op.attrs.value("keepdims", false);
        emit_reduce_any(w, out, op.inputs[0], shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
      } else if (op.op == "exp") {
        int64_t n = numel(out_shape);
        w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) { " + out + "[i] = expf(" + op.inputs[0] + "[i]); }");
      } else if (op.op == "relu") {
        int64_t n = numel(out_shape);
        w.pp_line("#if defined(__riscv_vector) || defined(__riscv_v)");
        w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ) {");
        w.indent();
        w.line("size_t vl = intentir_vsetvl_e32m1((size_t)" + std::to_string(n) + " - i);");
        w.line("vfloat32m1_t vx = __riscv_vle32_v_f32m1(&" + op.inputs[0] + "[i], vl);");
        w.line("vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);");
        w.line("vfloat32m1_t vy = __riscv_vfmax_vv_f32m1(vx, v0, vl);");
        w.line("__riscv_vse32_v_f32m1(&" + out + "[i], vy, vl);");
        w.line("i += vl;");
        w.dedent();
        w.line("}");
        w.pp_line("#else");
        w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) { float v = " + op.inputs[0] + "[i]; " + out + "[i] = v > 0.0f ? v : 0.0f; }");
        w.pp_line("#endif");
      } else if (op.op == "softmax") {
        int axis = op.attrs.value("axis", -1);
        emit_softmax(w, out, op.inputs[0], shape_env.at(op.inputs[0]), axis);
      } else if (op.op == "matmul") {
        bool ta = op.attrs.value("transpose_a", false);
        bool tb = op.attrs.value("transpose_b", false);
        emit_matmul(w, out, op.inputs[0], op.inputs[1], shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, ta, tb,
                    sched_tile_m, sched_tile_n, sched_tile_k, sched_vec_width);
      } else {
        fail("unsupported op lowering: " + op.op);
      }
      w.blank();
    }
    w.dedent();
    w.line("}");
    w.blank();
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string intent_path;
    std::string shapes_path;
    double atol = 1e-3;
    double rtol = 1e-3;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--intent" && i + 1 < argc) intent_path = argv[++i];
      else if (a == "--shapes" && i + 1 < argc) shapes_path = argv[++i];
      else if (a == "--atol" && i + 1 < argc) atol = std::stod(argv[++i]);
      else if (a == "--rtol" && i + 1 < argc) rtol = std::stod(argv[++i]);
      else if (a == "-h" || a == "--help") {
        std::cout << "usage: intentir_codegen --intent <intent.json> --shapes <shapes.json> [--atol x] [--rtol x]\\n";
        return 0;
      } else {
        fail("unknown arg: " + a);
      }
    }
    if (intent_path.empty() || shapes_path.empty()) {
      std::cerr << "missing --intent/--shapes (use --help)\\n";
      return 2;
    }

    Intent intent = parse_intent(read_json_file(intent_path));
    json shapes_json = read_json_file(shapes_path);
    if (!shapes_json.is_object()) fail("shapes must be object");
    std::unordered_map<std::string, int64_t> bindings;
    for (auto it = shapes_json.begin(); it != shapes_json.end(); ++it) {
      if (it.value().is_number_integer()) bindings.emplace(it.key(), it.value().get<int64_t>());
      else if (it.value().is_number()) bindings.emplace(it.key(), static_cast<int64_t>(it.value().get<double>()));
    }

    // Backend schedule knobs (C): optional tunables/hints.
    auto resolve_sched_int = [&](const char* key) -> std::optional<int64_t> {
      if (!intent.schedule.is_object()) return std::nullopt;
      auto it = intent.schedule.find(key);
      if (it == intent.schedule.end()) return std::nullopt;
      return resolve_dim_token(*it, bindings);
    };
    std::optional<int64_t> sched_tile_m = resolve_sched_int("tile_m");
    std::optional<int64_t> sched_tile_n = resolve_sched_int("tile_n");
    std::optional<int64_t> sched_tile_k = resolve_sched_int("tile_k");
    std::optional<int64_t> sched_vec_width = resolve_sched_int("vec_width");

    // Determine external inputs.
    std::unordered_map<std::string, bool> produced;
    std::unordered_map<std::string, bool> used;
    for (const auto& op : intent.ops) {
      produced[op.output] = true;
      for (const auto& x : op.inputs) used[x] = true;
    }
    std::vector<std::string> external_inputs;
    for (const auto& kv : used) {
      const std::string& name = kv.first;
      if (intent.tensors.find(name) != intent.tensors.end() && produced.find(name) == produced.end()) external_inputs.push_back(name);
    }
    std::sort(external_inputs.begin(), external_inputs.end());

    // Resolve declared tensor shapes.
    std::unordered_map<std::string, std::vector<int64_t>> shape_env;
    std::unordered_map<std::string, std::string> dtype_env;
    for (const auto& kv : intent.tensors) {
      const std::string& name = kv.first;
      const Tensor& t = kv.second;
      std::vector<int64_t> shp;
      for (const auto& d : t.shape) {
        if (d.is_number_integer()) shp.push_back(d.get<int64_t>());
        else if (d.is_string()) {
          std::string s = d.get<std::string>();
          if (!s.empty() && std::all_of(s.begin(), s.end(), ::isdigit)) {
            shp.push_back(std::stoll(s));
          } else {
            auto it = bindings.find(s);
            if (it == bindings.end()) fail("unbound symbol in shape: " + s);
            shp.push_back(it->second);
          }
        } else {
          fail("invalid dim type in tensor shape");
        }
      }
      shape_env[name] = shp;
      dtype_env[name] = t.dtype;
    }

    // Infer missing intermediates.
    for (const auto& op : intent.ops) {
      const std::string& kind = op.op;
      const std::string& out = op.output;
      if (kind == "const") {
        shape_env[out] = {};
        dtype_env[out] = op.attrs.value("dtype", dtype_env.count(out) ? dtype_env[out] : std::string("f32"));
        continue;
      }
      if (shape_env.find(out) != shape_env.end()) {
        if (kind == "ne" || kind == "lt" || kind == "le" || kind == "gt" || kind == "ge" || kind == "and" || kind == "or" || kind == "not" || kind == "reduce_any") {
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
      if (kind == "add" || kind == "sub" || kind == "mul" || kind == "div" || kind == "max" || kind == "min") {
        const auto& sa = get_shape(op.inputs[0]);
        const auto& sb = get_shape(op.inputs[1]);
        shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "ne" || kind == "lt" || kind == "le" || kind == "gt" || kind == "ge" || kind == "and" || kind == "or") {
        const auto& sa = get_shape(op.inputs[0]);
        const auto& sb = get_shape(op.inputs[1]);
        shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
        dtype_env[out] = "bool";
        continue;
      }
      if (kind == "not") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = "bool";
        continue;
      }
      if (kind == "abs" || kind == "floor") {
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
      if (kind == "where") {
        std::vector<int64_t> oshape = broadcast_shape(get_shape(op.inputs[0]), get_shape(op.inputs[1]));
        oshape = broadcast_shape(oshape, get_shape(op.inputs[2]));
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[1]);
        continue;
      }
      if (kind == "rsqrt" || kind == "exp" || kind == "relu" || kind == "softmax") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "reduce_sum" || kind == "reduce_max" || kind == "reduce_any") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (!op.attrs.contains("dims") || !op.attrs["dims"].is_array()) fail(kind + " requires dims list[int]");
        std::unordered_map<int,int> dims_set;
        for (const auto& d : op.attrs["dims"]) dims_set.emplace(d.get<int>(), 1);
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
        fail("matmul infer supports only rank-2 or rank-4");
      }
      fail("cannot infer output shape for op: " + kind);
    }

    // Collect consts.
    std::unordered_map<std::string, ConstVal> const_vals;
    for (const auto& op : intent.ops) {
      if (op.op == "const") {
        const json v = op.attrs.value("value", json());
        const std::string dt = op.attrs.value("dtype", dtype_env.count(op.output) ? dtype_env[op.output] : std::string("f32"));
        const_vals[op.output] = {dt, resolve_const_value(v, bindings)};
      }
    }

    // Best-effort FLOPs accounting (for benchmarking / cost-model experiments).
    // We only count matmul FLOPs for now (GEMM model validation).
    double matmul_flops_total = 0.0;
    for (const auto& op : intent.ops) {
      if (op.op != "matmul") continue;
      if (op.inputs.size() < 2) continue;
      const auto& sa = shape_env[op.inputs[0]];
      const auto& sb = shape_env[op.inputs[1]];
      bool ta = op.attrs.value("transpose_a", false);
      bool tb = op.attrs.value("transpose_b", false);
      if (sa.size() == 2 && sb.size() == 2) {
        int64_t M = ta ? sa[1] : sa[0];
        int64_t K = ta ? sa[0] : sa[1];
        int64_t K2 = tb ? sb[1] : sb[0];
        int64_t N = tb ? sb[0] : sb[1];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)M * (double)N * (double)K;
        continue;
      }
      if (sa.size() == 4 && sb.size() == 4) {
        int64_t B = sa[0];
        int64_t H = sa[1];
        int64_t M = ta ? sa[3] : sa[2];
        int64_t K = ta ? sa[2] : sa[3];
        int64_t K2 = tb ? sb[3] : sb[2];
        int64_t N = tb ? sb[2] : sb[3];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)B * (double)H * (double)M * (double)N * (double)K;
        continue;
      }
    }

    // ---- emit C program ----
    std::ostringstream c_src;
    CodeWriter cw(c_src);
    CProgramEmitter emitter{cw, intent, bindings, external_inputs, shape_env, dtype_env, const_vals,
                            sched_tile_m, sched_tile_n, sched_tile_k, sched_vec_width,
                            atol, rtol, matmul_flops_total};
    emitter.emit_program();
    std::cout << c_src.str();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "intentir_codegen error: " << e.what() << "\n";
    return 1;
  }
}
