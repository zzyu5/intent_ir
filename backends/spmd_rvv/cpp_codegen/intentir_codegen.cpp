#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <array>
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
  if (dt == "i16") return "int16_t";
  if (dt == "i32") return "int32_t";
  if (dt == "i64") return "int64_t";
  if (dt == "f64") return "double";
  return "float";
}

std::string typecode_for_dtype(const std::string& dt) {
  if (dt == "bool" || dt == "i1" || dt == "u8") return "INTENTIR_TYPE_U8";
  if (dt == "i8") return "INTENTIR_TYPE_I8";
  if (dt == "i16") return "INTENTIR_TYPE_I16";
  if (dt == "i32") return "INTENTIR_TYPE_I32";
  if (dt == "i64") return "INTENTIR_TYPE_I64";
  if (dt == "f64") return "INTENTIR_TYPE_F64";
  return "INTENTIR_TYPE_F32";
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

  // Heuristic: if a 1D tensor broadcasts into a higher-rank tensor and its single
  // dimension matches exactly one axis of the other shape, align it to that axis.
  // This avoids relying on symbol-name matching (which may diverge from emitted C identifiers).
  if ((int)shape.size() == 1 && out_rank >= 2 && (int)other_shape.size() == out_rank) {
    // Prefer IntentIR symbol-level matching when available: this disambiguates
    // cases like [M] broadcasting into [M,N] when M==N numerically for some
    // testcases (common in square matrices).
    if (auto ax = match_axis_1d_to_tensor(intent, bindings, name, other_name, out_rank)) {
      if (*ax >= 0 && *ax < out_rank) {
        std::vector<int64_t> named(out_rank, 1);
        named[*ax] = shape[0];
        return named;
      }
    }
    std::vector<int> hits;
    for (int ax = 0; ax < out_rank; ++ax) {
      if (other_shape[ax] == shape[0]) hits.push_back(ax);
    }
    if (hits.size() == 1) {
      std::vector<int64_t> named(out_rank, 1);
      named[hits[0]] = shape[0];
      return named;
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

enum class TokKind { End, Number, Ident, Plus, Minus, Star, Slash, Pow, LParen, RParen };

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
    if (c == '*') {
      // exponentiation: '**'
      if (i + 1 < src.size() && src[i + 1] == '*') { i += 2; return {TokKind::Pow}; }
      ++i;
      return {TokKind::Star};
    }
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
    return parse_pow();
  }

  double parse_pow() {
    double v = parse_atom();
    if (cur.kind == TokKind::Pow) {
      eat(TokKind::Pow);
      double rhs = parse_factor();  // right-associative; exponent can be unary
      v = std::pow(v, rhs);
    }
    return v;
  }

  double parse_atom() {
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
    // Some LLM/providers may emit Python-style integer division ("//").
    // Support it by rewriting to "/" and flooring the final result.
    if (s.find("//") != std::string::npos) {
      std::string t = s;
      size_t pos = 0;
      while ((pos = t.find("//", pos)) != std::string::npos) {
        t.replace(pos, 2, "/");
        pos += 1;
      }
      return std::floor(eval_expr(t, vars));
    }
    return eval_expr(s, vars);
  }
  fail("unsupported const value type");
}

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

void emit_log(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = logf(" + a + "[i]);");
}

void emit_cos(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("intentir_cos_f32(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ");");
}

void emit_acos(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = acosf(" + a + "[i]);");
}

void emit_atan(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out + "[i] = atanf(" + a + "[i]);");
}

void emit_erf(CodeWriter& w, const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  w.line("intentir_erf_f32(" + a + ", " + out + ", (size_t)" + std::to_string(n) + ");");
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

struct CProgramEmitter {
  CodeWriter& w;
  const Intent& intent;
  const std::unordered_map<std::string, int64_t>& bindings;
  const std::vector<std::string>& external_inputs;
  const std::unordered_map<std::string, std::vector<int64_t>>& shape_env;
  const std::unordered_map<std::string, std::string>& dtype_env;
  const std::unordered_map<std::string, ConstVal>& const_vals;
  std::unordered_map<std::string, std::string> cvars;
  std::optional<int64_t> sched_tile_m;
  std::optional<int64_t> sched_tile_n;
  std::optional<int64_t> sched_tile_k;
  std::optional<int64_t> sched_vec_width;
  double atol = 1e-3;
  double rtol = 1e-3;
  double matmul_flops_total = 0.0;
  bool bench_only = false;

  static std::string sanitize_ident(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
      unsigned char uc = static_cast<unsigned char>(c);
      if ((uc >= 'a' && uc <= 'z') || (uc >= 'A' && uc <= 'Z') || (uc >= '0' && uc <= '9') || c == '_') out.push_back(c);
      else out.push_back('_');
    }
    if (out.empty()) out = "_";
    unsigned char c0 = static_cast<unsigned char>(out[0]);
    if (!((c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || out[0] == '_')) out = "_" + out;
    return out;
  }

  void build_cvars(const std::vector<std::string>& names) {
    std::unordered_map<std::string, int> used;
    for (const auto& n : names) {
      std::string base = "t_" + sanitize_ident(n);
      int k = ++used[base];
      std::string v = (k == 1) ? base : (base + "_" + std::to_string(k));
      cvars.emplace(n, v);
    }
  }

  const std::string& v(const std::string& name) const {
    auto it = cvars.find(name);
    if (it == cvars.end()) fail("internal error: missing cvar for " + name);
    return it->second;
  }

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
    build_cvars(names);
    w.line("// intent tensors (globals)");
    for (const auto& name : names) {
      std::string ct = ctype_for_dtype(dtype_env.at(name));
      w.line("static " + ct + "* " + v(name) + " = NULL;");
    }
    w.blank();

    // Optional per-op profiling (Task6 perf tooling). Enabled by env:
    //   INTENTIR_PROFILE_OPS=1
    // Output:
    //   INTENTIR_PROFILE {"total_ns":...,"ops":[{"name":"...","ns":...},...]}
    std::vector<std::string> prof_names;
    prof_names.reserve(intent.ops.size());
    for (size_t i = 0; i < intent.ops.size(); ++i) {
      const auto& op = intent.ops[i];
      if (op.op == "const") continue;
      if (op.op == "reshape" || op.op == "identity" || op.op == "layout_cast") continue;
      prof_names.push_back("op[" + std::to_string(i) + "] " + op.op + " -> " + op.output);
    }
    w.line("// profiler state (optional)");
    w.line("static int intentir_profile_enabled = 0;");
    if (!prof_names.empty()) {
      w.line("static const char* intentir_profile_names[] = {");
      w.indent();
      for (const auto& n : prof_names) {
        w.line("\"" + n + "\",");
      }
      w.dedent();
      w.line("};");
      w.line("static uint64_t intentir_profile_ns[sizeof(intentir_profile_names) / sizeof(intentir_profile_names[0])];");
    }
    w.line("static void intentir_print_profile(void) {");
    w.indent();
    w.line("if (!intentir_profile_enabled) return;");
    if (prof_names.empty()) {
      w.line("printf(\"INTENTIR_PROFILE {\\\"total_ns\\\":0,\\\"ops\\\":[]}\\n\");");
      w.line("return;");
    } else {
      w.line("const size_t n = (size_t)(sizeof(intentir_profile_names) / sizeof(intentir_profile_names[0]));");
      w.line("uint64_t total = 0;");
      w.line("for (size_t i = 0; i < n; ++i) total += intentir_profile_ns[i];");
      w.line("printf(\"INTENTIR_PROFILE {\\\"total_ns\\\":%llu,\\\"ops\\\":[\", (unsigned long long)total);");
      w.line("for (size_t i = 0; i < n; ++i) {");
      w.indent();
      w.line("if (i) printf(\",\");");
      w.line("printf(\"{\\\"name\\\":\\\"%s\\\",\\\"ns\\\":%llu}\", intentir_profile_names[i], (unsigned long long)intentir_profile_ns[i]);");
      w.dedent();
      w.line("}");
      w.line("printf(\"]}\\n\");");
      w.line("return;");
    }
    w.dedent();
    w.line("}");
    w.blank();
  }

		  void emit_alloc_tensor(const std::string& name) {
		    const auto& shp = shape_env.at(name);
		    int64_t n = numel(shp);
		    std::string ct = ctype_for_dtype(dtype_env.at(name));
		    const std::string& var = v(name);
		    w.line(var + " = (" + ct + "*)malloc(sizeof(" + ct + ") * (size_t)" + std::to_string(n) + ");");
	    w.line("if (!" + var + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
	  }

	  void emit_main() {
	    w.line("int main() {");
	    w.indent();

	    w.line("setvbuf(stdout, NULL, _IONBF, 0);");
	    w.line("setvbuf(stderr, NULL, _IONBF, 0);");
	    w.line("intentir_runtime_init();");
	    w.line("const char* __intentir_prof = getenv(\"INTENTIR_PROFILE_OPS\");");
	    w.line("if (__intentir_prof && atoi(__intentir_prof) > 0) intentir_profile_enabled = 1;");
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
		        const bool byte_buf = (dt == "bool" || dt == "i1" || dt == "u8" || dt == "i8" || dt == "i16" || dt == "i32" || dt == "i64");
		        const std::string& var = v(name);
		        std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
		        if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
		        w.line("{\"" + name + "\", (void**)&" + var + ", (size_t)(" + bytes_expr + "), " +
		               (byte_buf ? "INTENTIR_DTYPE_U8" : "INTENTIR_DTYPE_F32") + "},");
		      }
	      w.dedent();
	      w.line("};");
		      if (!bench_only) {
		        w.line("if (!intentir_alloc_and_load_inputs(inputs, (size_t)(sizeof(inputs) / sizeof(inputs[0])))) return 2;");
		      } else {
		        w.line("if (!intentir_alloc(inputs, (size_t)(sizeof(inputs) / sizeof(inputs[0])))) return 2;");
		        w.line("uint64_t intentir_bench_seed = 0;");
		        w.line("const char* __intentir_bs = getenv(\"INTENTIR_BENCH_SEED\");");
		        w.line("if (__intentir_bs && *__intentir_bs) intentir_bench_seed = (uint64_t)strtoull(__intentir_bs, NULL, 10);");
		        w.line("for (size_t bi = 0; bi < (size_t)(sizeof(inputs) / sizeof(inputs[0])); ++bi) {");
		        w.indent();
	        w.line("IntentirBufferDesc* b = &inputs[bi];");
	        w.line("if (!b->ptr || !*b->ptr) return 2;");
	        w.line("if (b->dtype == INTENTIR_DTYPE_U8) {");
	        w.indent();
	        w.line("uint8_t* p = (uint8_t*)(*b->ptr);");
	        w.line("for (size_t i = 0; i < b->bytes; ++i) p[i] = (uint8_t)(((uint64_t)i + intentir_bench_seed) & 1u);");
	        w.dedent();
	        w.line("} else {");
	        w.indent();
	        w.line("float* p = (float*)(*b->ptr);");
	        w.line("size_t n = b->bytes / sizeof(float);");
	        w.line("for (size_t i = 0; i < n; ++i) {");
	        w.indent();
	        w.line("size_t j = (size_t)((uint64_t)i + intentir_bench_seed);");
	        w.line("int t = (int)(j % 127u) - 63;");
	        w.line("p[i] = (float)t * 0.01f;");
	        w.dedent();
	        w.line("}");
		        w.dedent();
		        w.line("}");
		        w.dedent();
		        w.line("}");
		      }
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
	        const std::string& var = v(name);
	        w.line(var + " = (" + ct + "*)malloc(sizeof(" + ct + "));");
	        w.line("if (!" + var + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
	        if (ct == "float") {
	          w.line(var + "[0] = " + c_float(cv.value) + ";");
	        } else if (ct == "double") {
	          std::ostringstream oss;
	          oss.precision(17);
	          oss << cv.value;
	          w.line(var + "[0] = (double)" + oss.str() + ";");
	        } else {
	          w.line(var + "[0] = (" + ct + ")" + std::to_string((int64_t)cv.value) + ";");
	        }
	        w.blank();
	      }
	    }

	    // allocate outputs / set alias views
		    for (const auto& op : intent.ops) {
		      if (op.op == "const") continue;
	      const std::string& out = op.output;
	      const std::string& out_var = v(out);
	      if (op.op == "reshape" || op.op == "identity" || op.op == "layout_cast") {
	        std::string ct = ctype_for_dtype(dtype_env.at(out));
	        const std::string& in0_var = v(op.inputs[0]);
	        w.line("// " + op.op + " alias");
	        w.line(out_var + " = (" + ct + "*)" + in0_var + ";");
	        w.blank();
	        continue;
	      }
	      w.line("// op " + op.op + " -> " + out);
	      emit_alloc_tensor(out);
	      w.blank();
		    }

	    // compute once
	    w.line("intentir_compute();");
	    w.line("intentir_print_profile();");
	    w.line("if (intentir_profile_enabled) intentir_profile_enabled = 0;");
	    w.line("intentir_maybe_bench(intentir_compute, MATMUL_FLOPS);");
	    w.blank();

	    // compare outputs
	    if (bench_only) {
	      w.line("printf(\"PASS bench_only\\n\");");
	      w.line("return 0;");
	    } else {
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
		          const bool byte_buf = (dt == "bool" || dt == "i1" || dt == "u8" || dt == "i8" || dt == "i16" || dt == "i32" || dt == "i64");
		          const std::string& var = v(name);
		          std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
		          if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
		          w.line("{\"" + name + "\", (void**)&" + var + ", (size_t)(" + bytes_expr + "), " +
		                 (byte_buf ? "INTENTIR_DTYPE_U8" : "INTENTIR_DTYPE_F32") + "},");
		        }
	        w.dedent();
	        w.line("};");
	        w.line("int ok = intentir_compare_outputs_with_refs(outputs, (size_t)(sizeof(outputs) / sizeof(outputs[0])), ATOL, RTOL);");
	      }
	      w.line("printf(ok ? \"PASS lowered\\n\" : \"FAIL lowered\\n\");");
	      w.line("return ok ? 0 : 1;");
	    }

	    w.dedent();
	    w.line("}");
	  }

		  void emit_broadcast_in_dim(const Op& op, const std::vector<int64_t>& out_shape) {
		    const std::string& out = op.output;
		    const std::string& inp = op.inputs[0];
		    const std::string& out_var = v(out);
		    const std::string& inp_var = v(inp);
		    const auto& in_shape = shape_env.at(inp);
		    std::vector<int> bcast_dims;
		    for (const auto& d : op.attrs["broadcast_dims"]) bcast_dims.push_back(d.get<int>());

	    const std::string in_ct = ctype_for_dtype(dtype_env.at(inp));
	    const std::string out_ct = ctype_for_dtype(dtype_env.at(out));
	    const int in_rank = (int)in_shape.size();
	    const int out_rank = (int)out_shape.size();
	    if (in_ct == "float" && out_ct == "float" && in_rank <= 4 && out_rank <= 4) {
	      auto arr64 = [&](const std::vector<int64_t>& v) -> std::string {
	        std::string s = "(int64_t[]){";
	        for (size_t i = 0; i < v.size(); ++i) {
	          if (i) s += ",";
	          s += std::to_string(v[i]);
	        }
	        s += "}";
	        return s;
	      };
	      auto arri = [&](const std::vector<int>& v) -> std::string {
	        std::string s = "(int[]){";
	        for (size_t i = 0; i < v.size(); ++i) {
	          if (i) s += ",";
	          s += std::to_string(v[i]);
	        }
	        s += "}";
	        return s;
	      };
		      w.line("intentir_broadcast_in_dim_f32(" + inp_var + ", " + out_var + ", " + arr64(in_shape) + ", " + std::to_string(in_rank) + ", " + arr64(out_shape) + ", " +
		             std::to_string(out_rank) + ", " + arri(bcast_dims) + ");");
		      return;
		    }

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
	    w.line(out_var + "[" + out_idx + "] = " + inp_var + "[" + in_idx + "];");
    for (int i = 0; i < r_out; ++i) {
      w.dedent();
      w.line("}");
    }
  }

  void emit_compute_fn() {
    w.line("static void intentir_compute(void) {");
    w.indent();
    std::vector<size_t> emit_ops;
    emit_ops.reserve(intent.ops.size());
    for (size_t i = 0; i < intent.ops.size(); ++i) {
      const auto& op = intent.ops[i];
      if (op.op == "const") continue;
      if (op.op == "reshape" || op.op == "identity" || op.op == "layout_cast") continue;
      emit_ops.push_back(i);
    }

    // Backend fusion: detect canonical LayerNorm forward sequences and emit a fused runtime call.
    // Keep profiling indices stable by emitting empty slots for fused-away ops.
    enum FuseTag : uint8_t { FUSE_NONE = 0, FUSE_LN_HEAD = 1, FUSE_LN_TAIL = 2 };
    std::vector<uint8_t> fuse_tags(emit_ops.size(), (uint8_t)FUSE_NONE);
    const size_t LN_LEN = 15;  // number of non-alias, non-const ops in the canonical pattern

    auto get_reduce_dims = [&](const Op& op) -> std::vector<int> {
      std::vector<int> dims;
      if (op.attrs.contains("axes")) {
        for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
      } else if (op.attrs.contains("dims")) {
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
      } else if (op.attrs.contains("axis")) {
        if (op.attrs["axis"].is_array()) {
          for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
        } else {
          dims.push_back(op.attrs["axis"].get<int>());
        }
      }
      return dims;
    };

    for (size_t pos = 0; pos + LN_LEN <= emit_ops.size(); ++pos) {
      if (fuse_tags[pos] != (uint8_t)FUSE_NONE) continue;
      const Op& o0 = intent.ops[emit_ops[pos + 0]];
      const Op& o1 = intent.ops[emit_ops[pos + 1]];
      const Op& o2 = intent.ops[emit_ops[pos + 2]];
      const Op& o3 = intent.ops[emit_ops[pos + 3]];
      const Op& o4 = intent.ops[emit_ops[pos + 4]];
      const Op& o5 = intent.ops[emit_ops[pos + 5]];
      const Op& o6 = intent.ops[emit_ops[pos + 6]];
      const Op& o7 = intent.ops[emit_ops[pos + 7]];
      const Op& o8 = intent.ops[emit_ops[pos + 8]];
      const Op& o9 = intent.ops[emit_ops[pos + 9]];
      const Op& o10 = intent.ops[emit_ops[pos + 10]];
      const Op& o11 = intent.ops[emit_ops[pos + 11]];
      const Op& o12 = intent.ops[emit_ops[pos + 12]];
      const Op& o13 = intent.ops[emit_ops[pos + 13]];
      const Op& o14 = intent.ops[emit_ops[pos + 14]];

      if (!(o0.op == "reduce_sum" && o1.op == "div" && o2.op == "broadcast_in_dim" && o3.op == "sub" && o4.op == "mul" &&
            o5.op == "reduce_sum" && o6.op == "div" && o7.op == "add" && o8.op == "rsqrt" && o9.op == "broadcast_in_dim" &&
            o10.op == "broadcast_in_dim" && o11.op == "broadcast_in_dim" && o12.op == "mul" && o13.op == "mul" && o14.op == "add")) {
        continue;
      }

      if (o0.inputs.size() != 1 || o0.inputs[0].empty()) continue;
      const std::string& X = o0.inputs[0];
      if (!shape_env.count(X) || !dtype_env.count(X)) continue;
      const auto& x_shape = shape_env.at(X);
      if (x_shape.size() != 2) continue;
      if (dtype_env.at(X) != "f32") continue;

      auto d0 = get_reduce_dims(o0);
      auto d5 = get_reduce_dims(o5);
      if (!(d0.size() == 1 && d0[0] == 1 && d5.size() == 1 && d5[0] == 1)) continue;

      const std::string& var_name = o6.output;
      if (o7.inputs.size() != 2) continue;
      std::string eps_name;
      if (o7.inputs[0] == var_name) eps_name = o7.inputs[1];
      else if (o7.inputs[1] == var_name) eps_name = o7.inputs[0];
      else continue;
      if (!dtype_env.count(eps_name) || dtype_env.at(eps_name) != "f32") continue;

      if (o10.inputs.size() != 1 || o11.inputs.size() != 1) continue;
      const std::string& W = o10.inputs[0];
      const std::string& B = o11.inputs[0];
      if (!shape_env.count(W) || !shape_env.count(B)) continue;
      if (shape_env.at(W).size() != 1 || shape_env.at(B).size() != 1) continue;
      if (shape_env.at(W)[0] != x_shape[1] || shape_env.at(B)[0] != x_shape[1]) continue;
      if (!dtype_env.count(W) || !dtype_env.count(B)) continue;
      if (dtype_env.at(W) != "f32" || dtype_env.at(B) != "f32") continue;

      const std::string& mean_name = o1.output;
      const std::string& rstd_name = o8.output;
      const std::string& y_name = o14.output;
      if (!shape_env.count(mean_name) || !shape_env.count(rstd_name) || !shape_env.count(y_name)) continue;
      if (shape_env.at(mean_name).size() != 1 || shape_env.at(rstd_name).size() != 1) continue;
      if (shape_env.at(mean_name)[0] != x_shape[0] || shape_env.at(rstd_name)[0] != x_shape[0]) continue;
      if (shape_env.at(y_name) != x_shape) continue;

      fuse_tags[pos] = (uint8_t)FUSE_LN_HEAD;
      for (size_t j = 1; j < LN_LEN; ++j) fuse_tags[pos + j] = (uint8_t)FUSE_LN_TAIL;
      pos += LN_LEN - 1;
    }

    size_t prof_i = 0;
    for (size_t pos = 0; pos < emit_ops.size(); ++pos) {
      const Op& op = intent.ops[emit_ops[pos]];
      const std::string& out = op.output;
	      const std::string& out_var = v(out);
	      const std::vector<int64_t>& out_shape = shape_env.at(out);
      w.line("{");
      w.indent();
      w.line("uint64_t __intentir_t0 = 0;");
      w.line("if (intentir_profile_enabled) __intentir_t0 = intentir_now_ns();");
      w.line("// op " + op.op + " -> " + out);

      if (fuse_tags[pos] == (uint8_t)FUSE_LN_HEAD) {
        const Op& o0 = intent.ops[emit_ops[pos + 0]];
        const Op& o1 = intent.ops[emit_ops[pos + 1]];
        const Op& o6 = intent.ops[emit_ops[pos + 6]];
        const Op& o7 = intent.ops[emit_ops[pos + 7]];
        const Op& o8 = intent.ops[emit_ops[pos + 8]];
        const Op& o10 = intent.ops[emit_ops[pos + 10]];
        const Op& o11 = intent.ops[emit_ops[pos + 11]];
        const Op& o14 = intent.ops[emit_ops[pos + 14]];

        const std::string& X = o0.inputs[0];
        const std::string& W = o10.inputs[0];
        const std::string& B = o11.inputs[0];
        const std::string& mean_name = o1.output;
        const std::string& rstd_name = o8.output;
        const std::string& y_name = o14.output;
        const auto& x_shape = shape_env.at(X);
        const int64_t M = x_shape[0];
        const int64_t N = x_shape[1];

        const std::string& var_name = o6.output;
        std::string eps_name;
        if (o7.inputs[0] == var_name) eps_name = o7.inputs[1];
        else eps_name = o7.inputs[0];

        w.line("// fused: layernorm forward (2D f32)");
        w.line("intentir_layernorm_2d_f32(" + v(X) + ", " + v(y_name) + ", " + v(W) + ", " + v(B) + ", " + v(mean_name) + ", " + v(rstd_name) +
               ", " + std::to_string(M) + ", " + std::to_string(N) + ", " + v(eps_name) + "[0]);");
      } else if (fuse_tags[pos] == (uint8_t)FUSE_LN_TAIL) {
        w.line("// fused into previous layernorm");
      } else if (op.op == "transpose") {
		        const auto& in_shape = shape_env.at(op.inputs[0]);
		        std::vector<int> perm;
		        for (const auto& p : op.attrs["perm"]) perm.push_back(p.get<int>());
		        const std::string in_ct = ctype_for_dtype(dtype_env.at(op.inputs[0]));
		        const std::string out_ct = ctype_for_dtype(dtype_env.at(out));
		        if (in_ct == "float" && out_ct == "float") {
	          auto arr64 = [&](const std::vector<int64_t>& v) -> std::string {
	            std::string s = "(int64_t[]){";
	            for (size_t i = 0; i < v.size(); ++i) {
	              if (i) s += ",";
	              s += std::to_string(v[i]);
	            }
	            s += "}";
	            return s;
	          };
	          auto arri = [&](const std::vector<int>& v) -> std::string {
	            std::string s = "(int[]){";
	            for (size_t i = 0; i < v.size(); ++i) {
	              if (i) s += ",";
	              s += std::to_string(v[i]);
	            }
	            s += "}";
	            return s;
		          };
		          if (perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2 && in_shape.size() == 4) {
		            w.line("intentir_transpose_4d_0132_f32(" + v(op.inputs[0]) + ", " + out_var + ", " + std::to_string(in_shape[0]) + ", " +
		                   std::to_string(in_shape[1]) + ", " + std::to_string(in_shape[2]) + ", " + std::to_string(in_shape[3]) + ");");
		          } else {
		            w.line("intentir_transpose_f32(" + v(op.inputs[0]) + ", " + out_var + ", " + arr64(in_shape) + ", " + arr64(out_shape) + ", " + arri(perm) +
		                   ", " + std::to_string((int)in_shape.size()) + ");");
		          }
		        } else {
		          if (perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2) {
		            emit_transpose_4d_0132(w, out_var, v(op.inputs[0]), in_shape, out_shape);
		          } else {
		            emit_transpose_generic(w, out_var, v(op.inputs[0]), in_shape, out_shape, perm);
		          }
		        }
		      } else if (op.op == "broadcast_in_dim") {
			        emit_broadcast_in_dim(op, out_shape);
	      } else if (op.op == "concat") {
	        if (op.inputs.size() != 2) fail("concat currently supports exactly 2 inputs");
	        const auto& a_shape = shape_env.at(op.inputs[0]);
	        const auto& b_shape = shape_env.at(op.inputs[1]);
	        if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 2) fail("concat currently supports rank-2 tensors");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("concat currently supports only f32 tensors");
	        int axis = 0;
	        if (op.attrs.contains("axis")) axis = (int)resolve_const_value(op.attrs["axis"], bindings);
	        if (axis < 0) axis += 2;
	        if (axis < 0 || axis >= 2) fail("concat axis out of range for rank-2 tensor");
	        if (axis == 1) {
	          if (a_shape[0] != b_shape[0]) fail("concat axis=1 requires matching row count");
	          if (out_shape[0] != a_shape[0] || out_shape[1] != (a_shape[1] + b_shape[1])) fail("concat output shape mismatch");
	          w.line("for (int i = 0; i < " + std::to_string(out_shape[0]) + "; ++i) {");
	          w.indent();
	          w.line("for (int j = 0; j < " + std::to_string(out_shape[1]) + "; ++j) {");
	          w.indent();
	          w.line("if (j < " + std::to_string(a_shape[1]) + ") " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[0]) +
	                 "[i * " + std::to_string(a_shape[1]) + " + j];");
	          w.line("else " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[1]) + "[i * " + std::to_string(b_shape[1]) +
	                 " + (j - " + std::to_string(a_shape[1]) + ")];");
	          w.dedent();
	          w.line("}");
	          w.dedent();
	          w.line("}");
	        } else {
	          if (a_shape[1] != b_shape[1]) fail("concat axis=0 requires matching column count");
	          if (out_shape[1] != a_shape[1] || out_shape[0] != (a_shape[0] + b_shape[0])) fail("concat output shape mismatch");
	          w.line("for (int i = 0; i < " + std::to_string(out_shape[0]) + "; ++i) {");
	          w.indent();
	          w.line("for (int j = 0; j < " + std::to_string(out_shape[1]) + "; ++j) {");
	          w.indent();
	          w.line("if (i < " + std::to_string(a_shape[0]) + ") " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[0]) +
	                 "[i * " + std::to_string(a_shape[1]) + " + j];");
	          w.line("else " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[1]) + "[(i - " + std::to_string(a_shape[0]) +
	                 ") * " + std::to_string(b_shape[1]) + " + j];");
	          w.dedent();
	          w.line("}");
	          w.dedent();
	          w.line("}");
	        }
	      } else if (op.op == "pad") {
	        if (op.inputs.size() != 1) fail("pad expects exactly 1 input tensor");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        if (in_shape.size() != 2 || out_shape.size() != 2) fail("pad currently supports rank-2 tensors");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("pad currently supports only f32 tensors");
	        std::string mode = op.attrs.value("mode", "constant");
	        if (mode != "constant") fail("pad currently supports mode=constant");
	        json padw = op.attrs.value("pad_width", json{});
	        if (padw.is_object() && padw.contains("pairs")) padw = padw["pairs"];
	        if (!padw.is_array() || padw.size() != 2) fail("pad_width must be rank-2 pairs");
	        const auto& p0 = padw[0];
	        const auto& p1 = padw[1];
	        if (!p0.is_array() || p0.size() != 2 || !p1.is_array() || p1.size() != 2) fail("pad_width pairs must each contain 2 elements");
	        const int pad_top = (int)resolve_const_value(p0[0], bindings);
	        const int pad_bottom = (int)resolve_const_value(p0[1], bindings);
	        const int pad_left = (int)resolve_const_value(p1[0], bindings);
	        const int pad_right = (int)resolve_const_value(p1[1], bindings);
	        (void)pad_bottom;
	        (void)pad_right;
	        const double pad_val = op.attrs.contains("value") ? resolve_const_value(op.attrs["value"], bindings) : 0.0;
	        w.line("for (int i = 0; i < " + std::to_string(out_shape[0]) + "; ++i) {");
	        w.indent();
	        w.line("for (int j = 0; j < " + std::to_string(out_shape[1]) + "; ++j) {");
	        w.indent();
	        w.line("const int src_i = i - " + std::to_string(pad_top) + ";");
	        w.line("const int src_j = j - " + std::to_string(pad_left) + ";");
	        w.line("if (src_i >= 0 && src_i < " + std::to_string(in_shape[0]) + " && src_j >= 0 && src_j < " + std::to_string(in_shape[1]) + ") {");
	        w.indent();
	        w.line(out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[0]) + "[src_i * " + std::to_string(in_shape[1]) + " + src_j];");
	        w.dedent();
	        w.line("} else {");
	        w.indent();
	        w.line(out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = (float)(" + std::to_string(pad_val) + ");");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "add" || op.op == "sub" || op.op == "mul" || op.op == "div" || op.op == "max" || op.op == "min" || op.op == "pow") {
		        if (op.inputs.size() == 2) {
		          emit_elemwise_bin(w, intent, bindings, op.op, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                            shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape,
		                            dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
		        } else if (op.inputs.size() == 1) {
	          // Legacy scalar form (1 input + scalar attr), used by some LLM outputs and
	          // by deterministic intent builders for reductions (mean/var normalization).
          std::string key;
          if (op.op == "div" && op.attrs.contains("divisor")) key = "divisor";
          else if (op.op == "add" && op.attrs.contains("addend")) key = "addend";
          else if (op.op == "sub" && op.attrs.contains("subtract")) key = "subtract";
          else if (op.op == "mul" && op.attrs.contains("mul_factor")) key = "mul_factor";
          else if ((op.op == "max" || op.op == "min") && op.attrs.contains("other")) key = "other";
          else fail("elemwise op requires 2 inputs (or 1 + scalar attr)");
	          if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("elemwise scalar path supports only f32");
	          if (shape_env.at(op.inputs[0]) != out_shape) fail("elemwise scalar expects input shape == output shape");
	          double scalar = resolve_const_value(op.attrs[key], bindings);
	          emit_elemwise_scalar_f32(w, op.op, out_var, v(op.inputs[0]), out_shape, scalar);
		        } else {
		          fail("elemwise op requires 2 inputs (or 1 + scalar attr)");
		        }
		      } else if (op.op == "ne") {
		        emit_ne(w, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
		                dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
	      } else if (op.op == "eq" || op.op == "lt" || op.op == "le" || op.op == "gt" || op.op == "ge") {
		        emit_cmp(w, op.op, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                 shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
		                 dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
      } else if (op.op == "and" || op.op == "or" || op.op == "bitwise_and" || op.op == "bitwise_or") {
        const std::string logical_op = (op.op == "bitwise_and") ? "and" : ((op.op == "bitwise_or") ? "or" : op.op);
        emit_bool_bin(w, logical_op, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                      shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
		                      dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
		      } else if (op.op == "not" || op.op == "bitwise_not") {
		        emit_bool_not(w, out_var, v(op.inputs[0]), out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "bitwise_left_shift") {
	        emit_bitwise_left_shift(w, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
	                                shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
	                                dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
	      } else if (op.op == "bitwise_right_shift") {
	        emit_bitwise_right_shift(w, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
	                                 shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
	                                 dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
	      } else if (op.op == "rsqrt") {
	        emit_rsqrt(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "abs") {
	        emit_unary_abs(w, out_var, v(op.inputs[0]), out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "floor") {
	        emit_floor(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "ceil") {
	        emit_ceil(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "acos") {
	        emit_acos(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "atan") {
	        emit_atan(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "cos") {
	        emit_cos(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "erf") {
	        emit_erf(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "cast") {
	        emit_cast(w, out_var, v(op.inputs[0]), out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "where") {
	        emit_where(w, out_var, v(op.inputs[0]), v(op.inputs[1]), v(op.inputs[2]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), shape_env.at(op.inputs[2]), out_shape,
	                   dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(op.inputs[2]), dtype_env.at(out));
	      } else if (op.op == "polar") {
	        if (op.inputs.size() != 2) fail("polar expects 2 inputs [abs, angle]");
	        const auto& abs_shape = shape_env.at(op.inputs[0]);
	        const auto& angle_shape = shape_env.at(op.inputs[1]);
	        if (abs_shape != angle_shape) fail("polar expects abs/angle shape match");
	        if (abs_shape.size() != 2 || out_shape.size() != 3) fail("polar currently supports [M,N] -> [M,N,2]");
	        if (out_shape[0] != abs_shape[0] || out_shape[1] != abs_shape[1] || out_shape[2] != 2) {
	          fail("polar output shape mismatch");
	        }
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("polar currently supports only f32");
	        }
	        const int64_t M = abs_shape[0];
	        const int64_t N = abs_shape[1];
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        w.line("const size_t idx = (size_t)i * " + std::to_string(N) + " + (size_t)j;");
	        w.line("const float r = " + v(op.inputs[0]) + "[idx];");
	        w.line("const float a0 = " + v(op.inputs[1]) + "[idx];");
	        w.line(out_var + "[idx * 2 + 0] = r * cosf(a0);");
	        w.line(out_var + "[idx * 2 + 1] = r * sinf(a0);");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "cumsum") {
	        int axis = op.attrs.value("axis", 0);
	        emit_cumsum(w, out_var, v(op.inputs[0]), out_shape, axis, dtype_env.at(out));
	      } else if (op.op == "cummax") {
	        int axis = op.attrs.value("axis", 0);
	        emit_cumext_1d(w, out_var, v(op.inputs[0]), out_shape, axis, dtype_env.at(out), /*is_max=*/true);
	      } else if (op.op == "cummin") {
	        int axis = op.attrs.value("axis", 0);
	        emit_cumext_1d(w, out_var, v(op.inputs[0]), out_shape, axis, dtype_env.at(out), /*is_max=*/false);
	      } else if (op.op == "nonzero") {
	        if (op.inputs.size() != 1) fail("nonzero expects 1 input");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        if (in_shape.size() != 2 || out_shape.size() != 2) fail("nonzero currently supports in[2] -> out[2]");
	        if (out_shape[1] != 2) fail("nonzero expects output shape [K, 2]");
	        if (dtype_env.at(out) != "i64") fail("nonzero currently supports i64 output");
	        const std::string in_dt = dtype_env.at(op.inputs[0]);
	        const std::string in_ct = ctype_for_dtype(in_dt);
	        const int64_t M = in_shape[0];
	        const int64_t N = in_shape[1];
	        const int64_t K = out_shape[0];
	        w.line("int64_t write_pos = 0;");
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        if (in_dt == "bool" || in_dt == "i1" || in_dt == "u8") {
	          w.line("const int nz = (" + v(op.inputs[0]) + "[(size_t)(i * " + std::to_string(N) + " + j)] != (uint8_t)0);");
	        } else {
	          w.line("const int nz = (" + v(op.inputs[0]) + "[(size_t)(i * " + std::to_string(N) + " + j)] != (" + in_ct + ")0);");
	        }
	        w.line("if (nz) {");
	        w.indent();
	        w.line("if (write_pos < " + std::to_string(K) + ") {");
	        w.indent();
	        w.line(out_var + "[(size_t)write_pos * 2 + 0] = (int64_t)i;");
	        w.line(out_var + "[(size_t)write_pos * 2 + 1] = (int64_t)j;");
	        w.dedent();
	        w.line("}");
	        w.line("write_pos += 1;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line("for (int64_t t = write_pos; t < " + std::to_string(K) + "; ++t) {");
	        w.indent();
	        w.line(out_var + "[(size_t)t * 2 + 0] = 0;");
	        w.line(out_var + "[(size_t)t * 2 + 1] = 0;");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "nll_loss_forward") {
	        if (op.inputs.size() != 3) fail("nll_loss_forward expects inputs [self, target, weight]");
	        const auto& logits_shape = shape_env.at(op.inputs[0]);
	        const auto& target_shape = shape_env.at(op.inputs[1]);
	        const auto& weight_shape = shape_env.at(op.inputs[2]);
	        if (logits_shape.size() != 2 || target_shape.size() != 1 || weight_shape.size() != 1) {
	          fail("nll_loss_forward expects self[N,C], target[N], weight[C]");
	        }
	        const int64_t N = logits_shape[0];
	        const int64_t C = logits_shape[1];
	        if (target_shape[0] != N) fail("nll_loss_forward target length mismatch");
	        if (weight_shape[0] != C) fail("nll_loss_forward weight length mismatch");
	        const std::string target_dt = dtype_env.at(op.inputs[1]);
	        if (!(target_dt == "i64" || target_dt == "i32")) fail("nll_loss_forward target dtype must be i64/i32");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("nll_loss_forward currently supports f32 logits/weight/output");
	        }
	        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
	        const int ignore_index = op.attrs.contains("ignore_index") ? (int)resolve_const_value(op.attrs["ignore_index"], bindings) : -100;
	        if (reduction == 0) fail("nll_loss_forward reduction=none is not supported in RVV lowering yet");
	        w.line("double loss_sum = 0.0;");
	        w.line("double weight_sum = 0.0;");
	        w.line("for (int64_t n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("const int64_t t = (int64_t)" + v(op.inputs[1]) + "[(size_t)n];");
	        w.line("if (t == (int64_t)" + std::to_string(ignore_index) + ") continue;");
	        w.line("if (t < 0 || t >= " + std::to_string(C) + ") continue;");
	        w.line("const float wv = " + v(op.inputs[2]) + "[(size_t)t];");
	        w.line("const float lv = -" + v(op.inputs[0]) + "[(size_t)n * " + std::to_string(C) + " + (size_t)t] * wv;");
	        w.line("loss_sum += (double)lv;");
	        w.line("weight_sum += (double)wv;");
	        w.dedent();
	        w.line("}");
	        if (reduction == 1) {
	          w.line(out_var + "[0] = (weight_sum > 0.0) ? (float)(loss_sum / weight_sum) : 0.0f;");
	        } else {
	          w.line(out_var + "[0] = (float)loss_sum;");
	        }
	      } else if (op.op == "nll_loss2d_forward") {
	        if (op.inputs.size() != 3) fail("nll_loss2d_forward expects inputs [self, target, weight]");
	        const auto& logits_shape = shape_env.at(op.inputs[0]);
	        const auto& target_shape = shape_env.at(op.inputs[1]);
	        const auto& weight_shape = shape_env.at(op.inputs[2]);
	        if (logits_shape.size() != 4 || target_shape.size() != 3 || weight_shape.size() != 1) {
	          fail("nll_loss2d_forward expects self[N,C,H,W], target[N,H,W], weight[C]");
	        }
	        const int64_t N = logits_shape[0];
	        const int64_t C = logits_shape[1];
	        const int64_t H = logits_shape[2];
	        const int64_t W = logits_shape[3];
	        if (target_shape[0] != N || target_shape[1] != H || target_shape[2] != W) fail("nll_loss2d_forward target shape mismatch");
	        if (weight_shape[0] != C) fail("nll_loss2d_forward weight length mismatch");
	        const std::string target_dt = dtype_env.at(op.inputs[1]);
	        if (!(target_dt == "i64" || target_dt == "i32")) fail("nll_loss2d_forward target dtype must be i64/i32");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("nll_loss2d_forward currently supports f32 logits/weight/output");
	        }
	        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
	        const int ignore_index = op.attrs.contains("ignore_index") ? (int)resolve_const_value(op.attrs["ignore_index"], bindings) : -100;
	        if (reduction == 0) fail("nll_loss2d_forward reduction=none is not supported in RVV lowering yet");
	        w.line("double loss_sum = 0.0;");
	        w.line("double weight_sum = 0.0;");
	        w.line("for (int64_t n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int64_t h = 0; h < " + std::to_string(H) + "; ++h) {");
	        w.indent();
	        w.line("for (int64_t ww = 0; ww < " + std::to_string(W) + "; ++ww) {");
	        w.indent();
	        w.line("const int64_t t = (int64_t)" + v(op.inputs[1]) + "[((size_t)n * " + std::to_string(H) + " + (size_t)h) * " + std::to_string(W) + " + (size_t)ww];");
	        w.line("if (t == (int64_t)" + std::to_string(ignore_index) + ") continue;");
	        w.line("if (t < 0 || t >= " + std::to_string(C) + ") continue;");
	        w.line("const float wv = " + v(op.inputs[2]) + "[(size_t)t];");
	        w.line("const size_t idx = (((size_t)n * " + std::to_string(C) + " + (size_t)t) * " + std::to_string(H) + " + (size_t)h) * " + std::to_string(W) + " + (size_t)ww;");
	        w.line("const float lv = -" + v(op.inputs[0]) + "[idx] * wv;");
	        w.line("loss_sum += (double)lv;");
	        w.line("weight_sum += (double)wv;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        if (reduction == 1) {
	          w.line(out_var + "[0] = (weight_sum > 0.0) ? (float)(loss_sum / weight_sum) : 0.0f;");
	        } else {
	          w.line(out_var + "[0] = (float)loss_sum;");
	        }
	      } else if (op.op == "conv1d") {
	        if (op.inputs.size() != 3) fail("conv1d expects inputs [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const auto& b_shape = shape_env.at(op.inputs[2]);
	        if (x_shape.size() != 3 || w_shape.size() != 3 || b_shape.size() != 1 || out_shape.size() != 3)
	          fail("conv1d expects input[N,C,L], weight[CO,C_PER_G,K], bias[CO], out[N,CO,OL]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv1d currently supports f32 only");
	        const int64_t N = x_shape[0];
	        const int64_t C_IN = x_shape[1];
	        const int64_t L = x_shape[2];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t K = w_shape[2];
	        if (b_shape[0] != C_OUT) fail("conv1d bias shape mismatch");
	        const int64_t OL = out_shape[2];
	        const int stride = op.attrs.contains("stride") ? (int)resolve_const_value(op.attrs["stride"], bindings) : 1;
	        const int padding = op.attrs.contains("padding") ? (int)resolve_const_value(op.attrs["padding"], bindings) : 0;
	        const int dilation = op.attrs.contains("dilation") ? (int)resolve_const_value(op.attrs["dilation"], bindings) : 1;
	        const int groups = op.attrs.contains("groups") ? (int)resolve_const_value(op.attrs["groups"], bindings) : 1;
	        if (groups <= 0 || stride <= 0 || dilation <= 0) fail("conv1d attrs stride/dilation/groups must be positive");
	        if (C_IN != C_PER_G * groups) fail("conv1d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv1d C_OUT must be divisible by groups");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int ol = 0; ol < " + std::to_string(OL) + "; ++ol) {");
	        w.indent();
	        w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int k = 0; k < " + std::to_string(K) + "; ++k) {");
	        w.indent();
	        w.line("const int li = ol * " + std::to_string(stride) + " - " + std::to_string(padding) + " + k * " + std::to_string(dilation) + ";");
	        w.line("if (li >= 0 && li < " + std::to_string(L) + ") {");
	        w.indent();
	        w.line("const int x_idx = ((n * " + std::to_string(C_IN) + " + c) * " + std::to_string(L) + " + li);");
	        w.line("const int w_idx = ((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(K) + " + k);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OL) + " + ol)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv2d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv2d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4)
	          fail("conv2d expects input[N,C,H,W], weight[CO,C_PER_G,KH,KW], out[N,CO,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv2d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv2d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv2d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN_TOTAL = x_shape[1];
	        const int64_t H = x_shape[2];
	        const int64_t W = x_shape[3];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t KH = w_shape[2];
	        const int64_t KW = w_shape[3];
	        if (has_bias && b_shape[0] != C_OUT) fail("conv2d bias shape mismatch");
	        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
	          if (value.is_array()) {
	            if (value.size() != 2) fail("conv2d " + key + " must have length 2");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v};
	        };
	        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int, int>{1, 1};
	        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
	        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
	        const int groups = op.attrs.contains("groups") ? static_cast<int>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
	        if (groups <= 0 || stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
	          fail("conv2d attrs stride/dilation/groups must be positive");
	        if (C_IN_TOTAL != C_PER_G * groups) fail("conv2d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv2d C_OUT must be divisible by groups");
	        const int64_t OH = out_shape[2];
	        const int64_t OW = out_shape[3];
	        const int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
	        if (expected_oh != OH || expected_ow != OW) fail("conv2d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride.first) + " - " + std::to_string(padding.first) + " + kh * " + std::to_string(dilation.first) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride.second) + " - " + std::to_string(padding.second) + " + kw * " + std::to_string(dilation.second) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = (((n * " + std::to_string(C_IN_TOTAL) + " + c) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = (((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv3d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv3d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 5 || w_shape.size() != 5 || out_shape.size() != 5)
	          fail("conv3d expects input[N,C,D,H,W], weight[CO,C_PER_G,KD,KH,KW], out[N,CO,OD,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv3d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv3d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv3d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN_TOTAL = x_shape[1];
	        const int64_t D = x_shape[2];
	        const int64_t H = x_shape[3];
	        const int64_t W = x_shape[4];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t KD = w_shape[2];
	        const int64_t KH = w_shape[3];
	        const int64_t KW = w_shape[4];
	        if (has_bias && b_shape[0] != C_OUT) fail("conv3d bias shape mismatch");
	        auto parse_triple = [&](const json& value, const std::string& key) -> std::array<int, 3> {
	          if (value.is_array()) {
	            if (value.size() != 3) fail("conv3d " + key + " must have length 3");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)),
	                    static_cast<int>(resolve_const_value(value[1], bindings)),
	                    static_cast<int>(resolve_const_value(value[2], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v, v};
	        };
	        std::array<int, 3> stride = op.attrs.contains("stride") ? parse_triple(op.attrs["stride"], "stride") : std::array<int, 3>{1, 1, 1};
	        std::array<int, 3> padding = op.attrs.contains("padding") ? parse_triple(op.attrs["padding"], "padding") : std::array<int, 3>{0, 0, 0};
	        std::array<int, 3> dilation = op.attrs.contains("dilation") ? parse_triple(op.attrs["dilation"], "dilation") : std::array<int, 3>{1, 1, 1};
	        const int groups = op.attrs.contains("groups") ? static_cast<int>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
	        if (groups <= 0 || stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0 || dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0)
	          fail("conv3d attrs stride/dilation/groups must be positive");
	        if (C_IN_TOTAL != C_PER_G * groups) fail("conv3d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv3d C_OUT must be divisible by groups");
	        const int64_t OD = out_shape[2];
	        const int64_t OH = out_shape[3];
	        const int64_t OW = out_shape[4];
	        const int64_t expected_od = ((D + 2LL * padding[0] - (int64_t)dilation[0] * (KD - 1) - 1) / stride[0]) + 1;
	        const int64_t expected_oh = ((H + 2LL * padding[1] - (int64_t)dilation[1] * (KH - 1) - 1) / stride[1]) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding[2] - (int64_t)dilation[2] * (KW - 1) - 1) / stride[2]) + 1;
	        if (expected_od != OD || expected_oh != OH || expected_ow != OW) fail("conv3d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int od = 0; od < " + std::to_string(OD) + "; ++od) {");
	        w.indent();
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int kd = 0; kd < " + std::to_string(KD) + "; ++kd) {");
	        w.indent();
	        w.line("const int id = od * " + std::to_string(stride[0]) + " - " + std::to_string(padding[0]) + " + kd * " + std::to_string(dilation[0]) + ";");
	        w.line("if ((unsigned)id >= (unsigned)" + std::to_string(D) + ") continue;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride[1]) + " - " + std::to_string(padding[1]) + " + kh * " + std::to_string(dilation[1]) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride[2]) + " - " + std::to_string(padding[2]) + " + kw * " + std::to_string(dilation[2]) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = ((((n * " + std::to_string(C_IN_TOTAL) + " + c) * " + std::to_string(D) + " + id) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = ((((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(KD) + " + kd) * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[((((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OD) + " + od) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
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
	      } else if (op.op == "conv_depthwise2d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv_depthwise2d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4)
	          fail("conv_depthwise2d expects input[N,C,H,W], weight[C_OUT,1,KH,KW], out[N,C_OUT,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv_depthwise2d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv_depthwise2d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv_depthwise2d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN = x_shape[1];
	        const int64_t H = x_shape[2];
	        const int64_t W = x_shape[3];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t KH = w_shape[2];
	        const int64_t KW = w_shape[3];
	        if (w_shape[1] != 1) fail("conv_depthwise2d expects weight second dim == 1");
	        if (C_IN <= 0 || C_OUT <= 0 || (C_OUT % C_IN) != 0) fail("conv_depthwise2d channel multiplier mismatch");
	        if (has_bias && b_shape[0] != C_OUT) fail("conv_depthwise2d bias shape mismatch");
	        const int64_t channel_multiplier = C_OUT / C_IN;
	        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
	          if (value.is_array()) {
	            if (value.size() != 2) fail("conv_depthwise2d " + key + " must have length 2");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v};
	        };
	        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int, int>{1, 1};
	        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
	        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
	        if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
	          fail("conv_depthwise2d attrs stride/dilation must be positive");
	        const int64_t OH = out_shape[2];
	        const int64_t OW = out_shape[3];
	        const int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
	        if (expected_oh != OH || expected_ow != OW) fail("conv_depthwise2d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int ci = co / " + std::to_string(channel_multiplier) + ";");
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride.first) + " - " + std::to_string(padding.first) + " + kh * " + std::to_string(dilation.first) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride.second) + " - " + std::to_string(padding.second) + " + kw * " + std::to_string(dilation.second) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = (((n * " + std::to_string(C_IN) + " + ci) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = ((co * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "dropout") {
	        if (op.inputs.size() != 3) fail("dropout requires 3 inputs (X, p, seed)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("dropout supports only f32 tensors");
	        if (!shape_env.at(op.inputs[1]).empty()) fail("dropout p must be a scalar tensor (rank-0)");
	        if (!shape_env.at(op.inputs[2]).empty()) fail("dropout seed must be a scalar tensor (rank-0)");
	        const std::string seed_dt = dtype_env.at(op.inputs[2]);
	        std::string seed_expr;
	        if (seed_dt == "i32") seed_expr = "(uint64_t)(uint32_t)" + v(op.inputs[2]) + "[0]";
	        else if (seed_dt == "i64") seed_expr = "(uint64_t)" + v(op.inputs[2]) + "[0]";
	        else fail("dropout seed dtype must be i32 or i64");
	        int rounds = 10;
	        if (op.attrs.contains("n_rounds")) rounds = (int)resolve_const_value(op.attrs["n_rounds"], bindings);
	        int64_t n = numel(out_shape);
	        w.line("intentir_dropout_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ", " + v(op.inputs[1]) +
	               "[0], " + seed_expr + ", " + std::to_string(rounds) + ");");
	      } else if (op.op == "correlation") {
	        if (op.inputs.size() != 3) fail("correlation requires 3 inputs (src0, src1, out_shift)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(op.inputs[1]) != "i8" || dtype_env.at(out) != "i8") fail("correlation supports only i8 tensors");
	        const auto& s0 = shape_env.at(op.inputs[0]);
	        const auto& s1 = shape_env.at(op.inputs[1]);
	        if (s0.size() != 3 || s1.size() != 3 || out_shape.size() != 3) fail("correlation expects rank-3 tensors");
	        if (s0 != s1) fail("correlation expects src0/src1 shapes to match");
	        if (s0[1] != out_shape[1] || s0[2] != out_shape[2]) fail("correlation spatial shape mismatch");
	        if (!shape_env.at(op.inputs[2]).empty()) fail("correlation out_shift must be a scalar tensor (rank-0)");
	        const std::string sh_dt = dtype_env.at(op.inputs[2]);
	        if (!(sh_dt == "i32" || sh_dt == "i64")) fail("correlation out_shift dtype must be i32 or i64");
	        const int64_t OC = out_shape[0];
	        const int64_t IC = s0[0];
	        const int64_t H = out_shape[1];
	        const int64_t W = out_shape[2];
	        w.line("intentir_correlation_i8(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + out_var + ", " + std::to_string(OC) +
	               ", " + std::to_string(IC) + ", " + std::to_string(H) + ", " + std::to_string(W) + ", (int32_t)" + v(op.inputs[2]) + "[0]);");
	      } else if (op.op == "resize") {
	        if (op.inputs.size() != 1) fail("resize requires 1 input (src)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(out) != "i8") fail("resize supports only i8 tensors");
	        const auto& s = shape_env.at(op.inputs[0]);
	        if (s.size() != 3 || out_shape.size() != 3) fail("resize expects rank-3 tensors [C,H,W]");
	        if (out_shape[0] != s[0]) fail("resize channel mismatch");
	        if (out_shape[1] != 2 * s[1] || out_shape[2] != 2 * s[2]) fail("resize currently supports only 2x upsample");
	        int hw_fl = 7;
	        if (op.attrs.contains("hw_fl")) hw_fl = (int)resolve_const_value(op.attrs["hw_fl"], bindings);
	        w.line("intentir_resize_bilinear2x_i8(" + v(op.inputs[0]) + ", " + out_var + ", " + std::to_string(s[0]) + ", " + std::to_string(s[1]) + ", " +
	               std::to_string(s[2]) + ", " + std::to_string(hw_fl) + ");");
	      } else if (op.op == "warp") {
	        if (op.inputs.size() != 2) fail("warp requires 2 inputs (src, offset)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(op.inputs[1]) != "i16" || dtype_env.at(out) != "i8") fail("warp expects src/out i8 and offset i16");
	        const auto& s = shape_env.at(op.inputs[0]);
	        const auto& off = shape_env.at(op.inputs[1]);
	        if (s.size() != 3 || off.size() != 2 || out_shape.size() != 3) fail("warp expects src/out rank-3 and offset rank-2");
	        if (out_shape != s) fail("warp output shape must match src shape");
	        if (off[0] != s[1] || off[1] != s[2]) fail("warp offset shape must match [H,W]");
	        w.line("intentir_warp_q8_8_i8_i16(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + out_var + ", " + std::to_string(s[0]) + ", " +
	               std::to_string(s[1]) + ", " + std::to_string(s[2]) + ");");
	      } else if (op.op == "rope") {
	        if (op.inputs.size() != 3) fail("rope requires 3 inputs (input, cos, sin)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32")
	          fail("rope supports only f32 tensors");
	        const auto& x = shape_env.at(op.inputs[0]);
	        const auto& c = shape_env.at(op.inputs[1]);
	        const auto& s = shape_env.at(op.inputs[2]);
	        if (x.size() != 4 || out_shape.size() != 4) fail("rope expects rank-4 input/output");
	        if (c.size() != 2 || s.size() != 2) fail("rope expects rank-2 cos/sin");
	        if (out_shape != x) fail("rope output shape must match input shape");
	        const int64_t SEQ = x[0], B = x[1], H = x[2], D = x[3];
	        if ((D & 1) != 0) fail("rope expects even HEAD_DIM");
	        const int64_t half = D / 2;
	        if (c[0] != SEQ || c[1] != half) fail("rope cos shape must be [SEQ_LEN, HEAD_DIM/2]");
	        if (s[0] != SEQ || s[1] != half) fail("rope sin shape must be [SEQ_LEN, HEAD_DIM/2]");
	        w.line("intentir_rope_f32(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + v(op.inputs[2]) + ", " + out_var + ", " + std::to_string(SEQ) +
	               ", " + std::to_string(B) + ", " + std::to_string(H) + ", " + std::to_string(D) + ");");
	      } else if (op.op == "iota") {
	        int axis = op.attrs.value("axis", 0);
	        emit_iota(w, out_var, out_shape, axis, dtype_env.at(out));
	      } else if (op.op == "gather") {
	        std::vector<std::string> idxs;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idxs.push_back(v(op.inputs[i]));
	        std::vector<std::vector<int64_t>> idx_shapes;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idx_shapes.push_back(shape_env.at(op.inputs[i]));
	        std::vector<std::string> idx_dtypes;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idx_dtypes.push_back(dtype_env.at(op.inputs[i]));
	        emit_gather(w, out_var, v(op.inputs[0]), idxs, shape_env.at(op.inputs[0]), idx_shapes, out_shape, dtype_env.at(op.inputs[0]), idx_dtypes, dtype_env.at(out));
	      } else if (op.op == "index_add") {
	        if (op.inputs.size() != 3) fail("index_add expects inputs [base, index, src]");
	        int axis = op.attrs.value("axis", 0);
	        double alpha = 1.0;
	        if (op.attrs.contains("alpha")) alpha = resolve_const_value(op.attrs["alpha"], bindings);
	        emit_index_add_2d(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            out_shape,
	            dtype_env.at(op.inputs[1]),
	            axis,
	            alpha);
	      } else if (op.op == "index_put") {
	        if (op.inputs.size() != 4) fail("index_put expects inputs [base, row_idx, col_idx, values]");
	        bool accumulate = op.attrs.value("accumulate", false);
	        emit_index_put_2d(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            v(op.inputs[3]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            shape_env.at(op.inputs[3]),
	            out_shape,
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(op.inputs[2]),
	            accumulate);
	      } else if (op.op == "masked_select") {
	        if (op.inputs.size() != 2) fail("masked_select expects inputs [inp, mask]");
	        emit_masked_select_flat(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            out_shape,
	            dtype_env.at(op.inputs[0]),
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(out));
	      } else if (op.op == "masked_scatter") {
	        if (op.inputs.size() != 3) fail("masked_scatter expects inputs [inp, mask, source]");
	        emit_masked_scatter_flat(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            out_shape,
	            dtype_env.at(op.inputs[0]),
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(op.inputs[2]),
	            dtype_env.at(out));
	      } else if (op.op == "kron") {
	        if (op.inputs.size() != 2) fail("kron expects inputs [A, B]");
	        emit_kron_2d(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape);
	      } else if (op.op == "reduce_sum") {
        // Accept both "axes" and "dims" (parser/LLM may emit either). Prefer "axes"
        // when both are present to match interpreter semantics.
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_sum missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        std::optional<double> scale;
	        if (op.attrs.contains("scale")) scale = resolve_const_value(op.attrs["scale"], bindings);
	        emit_reduce_sum(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims, scale);
	      } else if (op.op == "reduce_prod") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_prod missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_prod(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_max") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_max missing dims/axes");
	        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_max(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_min") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_min missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_min(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_any") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_any missing dims/axes");
	        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_any(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "argmax" || op.op == "argmin") {
        int axis = op.attrs.value("axis", -1);
        emit_arg_reduce_axis1(w, op.op, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis);
	      } else if (op.op == "avg_pool2d") {
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
          if (value.is_array()) {
            if (value.size() != 2) fail("avg_pool2d " + key + " must have length 2");
            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
          }
          int v = static_cast<int>(resolve_const_value(value, bindings));
          return {v, v};
        };
        std::pair<int, int> ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int, int>{2, 2};
        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
        bool count_include_pad = op.attrs.value("count_include_pad", true);
        emit_avg_pool2d(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, ksz.first, ksz.second, stride.first, stride.second,
                        padding.first, padding.second, count_include_pad);
	      } else if (op.op == "max_pool2d_with_indices") {
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
          if (value.is_array()) {
            if (value.size() != 2) fail("max_pool2d_with_indices " + key + " must have length 2");
            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
          }
          int v = static_cast<int>(resolve_const_value(value, bindings));
          return {v, v};
        };
        std::pair<int, int> ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int, int>{2, 2};
        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
        bool ceil_mode = op.attrs.value("ceil_mode", false);
        if (ceil_mode) fail("max_pool2d_with_indices ceil_mode=true is not supported");
        if (dilation.first != 1 || dilation.second != 1) fail("max_pool2d_with_indices dilation != 1 is not supported");
        std::string select = op.attrs.value("select", "values");
        bool emit_indices = (select == "indices") || (dtype_env.at(out) == "i64");
        emit_max_pool2d_with_indices(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, ksz.first, ksz.second, stride.first, stride.second,
                                     padding.first, padding.second, emit_indices);
	      } else if (op.op == "mse_loss") {
        if (op.inputs.size() != 2) fail("mse_loss expects [input, target]");
        const auto& in_shape = shape_env.at(op.inputs[0]);
        const auto& tgt_shape = shape_env.at(op.inputs[1]);
        if (in_shape != tgt_shape) fail("mse_loss shape mismatch between input and target");
        std::string reduction;
        if (op.attrs.contains("reduction")) {
          if (op.attrs["reduction"].is_number_integer()) {
            int mode = op.attrs["reduction"].get<int>();
            if (mode == 0) reduction = "none";
            else if (mode == 2) reduction = "sum";
            else reduction = "mean";
          } else {
            reduction = op.attrs["reduction"].get<std::string>();
          }
        } else {
          reduction = "mean";
        }
        emit_mse_loss(w, out_var, v(op.inputs[0]), v(op.inputs[1]), in_shape, out_shape, reduction);
	      } else if (op.op == "scaled_dot_product_attention") {
        if (op.inputs.size() < 3) fail("scaled_dot_product_attention requires query/key/value");
        bool is_causal = op.attrs.value("is_causal", false);
        emit_scaled_dot_product_attention(w, out_var, v(op.inputs[0]), v(op.inputs[1]), v(op.inputs[2]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]),
                                          shape_env.at(op.inputs[2]), out_shape, is_causal);
	      } else if (op.op == "glu") {
          int axis = op.attrs.value("axis", -1);
          emit_glu(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis);
	      } else if (op.op == "exp") {
	        int64_t n = numel(out_shape);
          std::optional<double> base;
          if (op.attrs.contains("base")) {
            base = resolve_const_value(op.attrs["base"], bindings);
          }
          if (base.has_value()) {
            const double b = *base;
            if (std::fabs(b - 2.0) < 1e-6) {
              w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out_var + "[i] = exp2f(" + v(op.inputs[0]) + "[i]);");
            } else {
              w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out_var + "[i] = powf((float)" + std::to_string(b) + ", " +
                     v(op.inputs[0]) + "[i]);");
            }
          } else {
            w.line("intentir_exp_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ");");
          }
	      } else if (op.op == "log") {
	        emit_log(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "relu") {
	        int64_t n = numel(out_shape);
	        w.line("intentir_relu_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ");");
	      } else if (op.op == "softmax") {
	        int axis = op.attrs.value("axis", -1);
	        emit_softmax(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), axis);
	      } else if (op.op == "matmul") {
	        bool ta = op.attrs.value("transpose_a", false);
	        bool tb = op.attrs.value("transpose_b", false);
	        emit_matmul(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, ta, tb,
	                    sched_tile_m, sched_tile_n, sched_tile_k);
	      } else {
	        fail("unsupported op lowering: " + op.op);
	      }
      w.line("if (intentir_profile_enabled) intentir_profile_ns[" + std::to_string(prof_i) +
             "] += intentir_now_ns() - __intentir_t0;");
      w.dedent();
      w.line("}");
      w.blank();
      prof_i++;
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
    std::string mode = "verify";
    double atol = 1e-3;
    double rtol = 1e-3;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--intent" && i + 1 < argc) intent_path = argv[++i];
      else if (a == "--shapes" && i + 1 < argc) shapes_path = argv[++i];
      else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
      else if (a == "--atol" && i + 1 < argc) atol = std::stod(argv[++i]);
      else if (a == "--rtol" && i + 1 < argc) rtol = std::stod(argv[++i]);
      else if (a == "-h" || a == "--help") {
        std::cout << "usage: intentir_codegen --intent <intent.json> --shapes <shapes.json> [--mode verify|bench] [--atol x] [--rtol x]\\n";
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
    auto binding_get = [&](const std::string& key) -> std::optional<int64_t> {
      auto it = bindings.find(key);
      if (it == bindings.end()) return std::nullopt;
      return it->second;
    };
    auto derive_binding = [&](const std::string& key) -> std::optional<int64_t> {
      auto c_in = binding_get("C_IN");
      auto groups = binding_get("GROUPS");
      auto c_per_g = binding_get("C_PER_G");
      auto c_out = binding_get("C_OUT");
      auto mult = binding_get("MULT");
      if (key == "C_IN_TOTAL") {
        if (c_in && groups) return (*c_in) * (*groups);
        if (c_in) return *c_in;
      }
      if (key == "C_PER_G") {
        if (c_per_g) return *c_per_g;
        if (c_in && groups && *groups > 0 && ((*c_in) % (*groups) == 0)) return (*c_in) / (*groups);
        if (c_in) return *c_in;
      }
      if (key == "C_OUT") {
        if (c_out) return *c_out;
        if (c_in && mult) return (*c_in) * (*mult);
        if (c_per_g && groups) return (*c_per_g) * (*groups);
      }
      if (key == "OH") {
        auto H = binding_get("H");
        auto PH = binding_get("PH");
        auto KH = binding_get("KH");
        auto SH = binding_get("SH");
        auto DH = binding_get("DH");
        if (H && PH && KH && SH && DH && *SH > 0) return ((*H) + 2 * (*PH) - (*DH) * ((*KH) - 1) - 1) / (*SH) + 1;
      }
      if (key == "OW") {
        auto W = binding_get("W");
        auto PW = binding_get("PW");
        auto KW = binding_get("KW");
        auto SW = binding_get("SW");
        auto DW = binding_get("DW");
        if (W && PW && KW && SW && DW && *SW > 0) return ((*W) + 2 * (*PW) - (*DW) * ((*KW) - 1) - 1) / (*SW) + 1;
      }
      if (key == "OD") {
        auto D = binding_get("D");
        auto PD = binding_get("PD");
        auto KD = binding_get("KD");
        auto SD = binding_get("SD");
        auto DD = binding_get("DD");
        if (D && PD && KD && SD && DD && *SD > 0) return ((*D) + 2 * (*PD) - (*DD) * ((*KD) - 1) - 1) / (*SD) + 1;
      }
      return std::nullopt;
    };

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
            if (it == bindings.end()) {
              auto dv = derive_binding(s);
              if (!dv.has_value()) fail("unbound symbol in shape: " + s);
              bindings.emplace(s, *dv);
              shp.push_back(*dv);
            } else {
              shp.push_back(it->second);
            }
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
        if (kind == "eq" || kind == "ne" || kind == "lt" || kind == "le" || kind == "gt" || kind == "ge" || kind == "reduce_any") {
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
      if (kind == "concat") {
        if (op.inputs.empty()) fail("concat requires at least one input");
        const auto& first = get_shape(op.inputs[0]);
        if (first.empty()) fail("concat expects rank >= 1");
        int axis = op.attrs.value("axis", 0);
        if (axis < 0) axis += static_cast<int>(first.size());
        if (axis < 0 || axis >= static_cast<int>(first.size())) fail("concat axis out of range");
        std::vector<int64_t> shp = first;
        shp[axis] = 0;
        for (const auto& in_name : op.inputs) {
          const auto& s = get_shape(in_name);
          if (s.size() != first.size()) fail("concat requires matching ranks");
          for (int i = 0; i < (int)s.size(); ++i) {
            if (i == axis) shp[i] += s[i];
            else if (s[i] != first[i]) fail("concat requires matching non-axis dimensions");
          }
        }
        shape_env[out] = shp;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "pad") {
        if (op.inputs.size() != 1) fail("pad expects one input tensor");
        const auto& in_shape = get_shape(op.inputs[0]);
        json padw = op.attrs.value("pad_width", json{});
        if (padw.is_object() && padw.contains("pairs")) padw = padw["pairs"];
        if (!padw.is_array() || padw.size() != in_shape.size()) fail("pad_width rank mismatch");
        std::vector<int64_t> shp = in_shape;
        for (int i = 0; i < (int)in_shape.size(); ++i) {
          const auto& pair = padw[i];
          if (!pair.is_array() || pair.size() != 2) fail("pad_width entries must be [left,right]");
          int64_t left = static_cast<int64_t>(resolve_const_value(pair[0], bindings));
          int64_t right = static_cast<int64_t>(resolve_const_value(pair[1], bindings));
          shp[i] = in_shape[i] + left + right;
          if (shp[i] < 0) fail("pad produced negative output dimension");
        }
        shape_env[out] = shp;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "add" || kind == "sub" || kind == "mul" || kind == "div" || kind == "max" || kind == "min" || kind == "pow") {
        if (op.inputs.size() == 2) {
          const auto& sa = get_shape(op.inputs[0]);
          const auto& sb = get_shape(op.inputs[1]);
          shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
          dtype_env[out] = get_dtype(op.inputs[0]);
        } else if (op.inputs.size() == 1) {
          // Scalar-attr form: output shape matches the single tensor input.
          shape_env[out] = get_shape(op.inputs[0]);
          dtype_env[out] = get_dtype(op.inputs[0]);
        } else {
          fail("elemwise op requires 1 or 2 inputs");
        }
        continue;
      }
      if (kind == "eq" || kind == "ne" || kind == "lt" || kind == "le" || kind == "gt" || kind == "ge") {
        const auto& sa = get_shape(op.inputs[0]);
        const auto& sb = get_shape(op.inputs[1]);
        shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
        dtype_env[out] = "bool";
        continue;
      }
      if (kind == "and" || kind == "or" || kind == "bitwise_and" || kind == "bitwise_or" || kind == "bitwise_left_shift" || kind == "bitwise_right_shift") {
        const auto& sa = get_shape(op.inputs[0]);
        const auto& sb = get_shape(op.inputs[1]);
        shape_env[out] = broadcast_shape_named(intent, bindings, op.inputs[0], op.inputs[1], sa, sb);
        const std::string in_dt = get_dtype(op.inputs[0]);
        if (kind == "and" || kind == "or") {
          if (in_dt == "bool" || in_dt == "i1" || in_dt == "u8") dtype_env[out] = "bool";
          else dtype_env[out] = in_dt;
        } else {
          dtype_env[out] = in_dt;
        }
        continue;
      }
      if (kind == "not" || kind == "bitwise_not") {
        shape_env[out] = get_shape(op.inputs[0]);
        const std::string in_dt = get_dtype(op.inputs[0]);
        if (kind == "not" && (in_dt == "bool" || in_dt == "i1" || in_dt == "u8")) {
          dtype_env[out] = "bool";
        } else {
          dtype_env[out] = in_dt;
        }
        continue;
      }
      if (kind == "argmax" || kind == "argmin") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail(kind + " infer currently supports rank-2 input");
        int axis = op.attrs.value("axis", -1);
        if (axis < 0) axis += static_cast<int>(in_shape.size());
        if (axis != 1) fail(kind + " infer currently supports axis=1");
        shape_env[out] = {in_shape[0]};
        dtype_env[out] = "i32";
        continue;
      }
      if (kind == "conv1d") {
        if (op.inputs.size() != 3) fail("conv1d infer expects inputs [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const auto& b_shape = get_shape(op.inputs[2]);
        if (in_shape.size() != 3 || w_shape.size() != 3 || b_shape.size() != 1) fail("conv1d infer expects input[N,C,L], weight[CO,C_PER_G,K], bias[CO]");
        const int64_t N = in_shape[0];
        const int64_t C_IN = in_shape[1];
        const int64_t L = in_shape[2];
        const int64_t C_OUT = w_shape[0];
        const int64_t C_PER_G = w_shape[1];
        const int64_t K = w_shape[2];
        if (b_shape[0] != C_OUT) fail("conv1d infer bias shape mismatch");
        const int64_t stride = op.attrs.contains("stride") ? static_cast<int64_t>(resolve_const_value(op.attrs["stride"], bindings)) : 1;
        const int64_t padding = op.attrs.contains("padding") ? static_cast<int64_t>(resolve_const_value(op.attrs["padding"], bindings)) : 0;
        const int64_t dilation = op.attrs.contains("dilation") ? static_cast<int64_t>(resolve_const_value(op.attrs["dilation"], bindings)) : 1;
        const int64_t groups = op.attrs.contains("groups") ? static_cast<int64_t>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
        if (stride <= 0 || dilation <= 0 || groups <= 0) fail("conv1d infer stride/dilation/groups must be positive");
        if (C_IN != C_PER_G * groups) fail("conv1d infer channel/group mismatch");
        if ((C_OUT % groups) != 0) fail("conv1d infer C_OUT must be divisible by groups");
        const int64_t OL = ((L + 2 * padding - dilation * (K - 1) - 1) / stride) + 1;
        if (OL <= 0) fail("conv1d infer produced non-positive output length");
        shape_env[out] = {N, C_OUT, OL};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "conv2d") {
        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv2d infer expects inputs [input, weight] or [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const bool has_bias = op.inputs.size() == 3;
        const auto& b_shape = has_bias ? get_shape(op.inputs[2]) : std::vector<int64_t>{};
        if (in_shape.size() != 4 || w_shape.size() != 4) fail("conv2d infer expects input[N,C,H,W], weight[CO,C_PER_G,KH,KW]");
        const int64_t N = in_shape[0];
        const int64_t C_IN_TOTAL = in_shape[1];
        const int64_t H = in_shape[2];
        const int64_t W = in_shape[3];
        const int64_t C_OUT = w_shape[0];
        const int64_t C_PER_G = w_shape[1];
        const int64_t KH = w_shape[2];
        const int64_t KW = w_shape[3];
        if (has_bias) {
          if (b_shape.size() != 1) fail("conv2d infer bias must be rank-1");
          if (b_shape[0] != C_OUT) fail("conv2d infer bias shape mismatch");
        }
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("conv2d infer " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int64_t, int64_t>{1, 1};
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int64_t, int64_t>{1, 1};
        const int64_t groups = op.attrs.contains("groups") ? static_cast<int64_t>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
        if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0 || groups <= 0)
          fail("conv2d infer stride/dilation/groups must be positive");
        if (C_IN_TOTAL != C_PER_G * groups) fail("conv2d infer channel/group mismatch");
        if ((C_OUT % groups) != 0) fail("conv2d infer C_OUT must be divisible by groups");
        const int64_t OH = ((H + 2 * padding.first - dilation.first * (KH - 1) - 1) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - dilation.second * (KW - 1) - 1) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("conv2d infer produced non-positive output shape");
        shape_env[out] = {N, C_OUT, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "conv3d") {
        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv3d infer expects inputs [input, weight] or [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const bool has_bias = op.inputs.size() == 3;
        const auto& b_shape = has_bias ? get_shape(op.inputs[2]) : std::vector<int64_t>{};
        if (in_shape.size() != 5 || w_shape.size() != 5) fail("conv3d infer expects input[N,C,D,H,W], weight[CO,C_PER_G,KD,KH,KW]");
        const int64_t N = in_shape[0];
        const int64_t C_IN_TOTAL = in_shape[1];
        const int64_t D = in_shape[2];
        const int64_t H = in_shape[3];
        const int64_t W = in_shape[4];
        const int64_t C_OUT = w_shape[0];
        const int64_t C_PER_G = w_shape[1];
        const int64_t KD = w_shape[2];
        const int64_t KH = w_shape[3];
        const int64_t KW = w_shape[4];
        if (has_bias) {
          if (b_shape.size() != 1) fail("conv3d infer bias must be rank-1");
          if (b_shape[0] != C_OUT) fail("conv3d infer bias shape mismatch");
        }
        auto parse_triple = [&](const json& value, const std::string& key) -> std::array<int64_t, 3> {
          if (value.is_array()) {
            if (value.size() != 3) fail("conv3d infer " + key + " must have length 3");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)),
                    static_cast<int64_t>(resolve_const_value(value[1], bindings)),
                    static_cast<int64_t>(resolve_const_value(value[2], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v, v};
        };
        auto stride = op.attrs.contains("stride") ? parse_triple(op.attrs["stride"], "stride") : std::array<int64_t, 3>{1, 1, 1};
        auto padding = op.attrs.contains("padding") ? parse_triple(op.attrs["padding"], "padding") : std::array<int64_t, 3>{0, 0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_triple(op.attrs["dilation"], "dilation") : std::array<int64_t, 3>{1, 1, 1};
        const int64_t groups = op.attrs.contains("groups") ? static_cast<int64_t>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
        if (stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0 || dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0 || groups <= 0)
          fail("conv3d infer stride/dilation/groups must be positive");
        if (C_IN_TOTAL != C_PER_G * groups) fail("conv3d infer channel/group mismatch");
        if ((C_OUT % groups) != 0) fail("conv3d infer C_OUT must be divisible by groups");
        const int64_t OD = ((D + 2 * padding[0] - dilation[0] * (KD - 1) - 1) / stride[0]) + 1;
        const int64_t OH = ((H + 2 * padding[1] - dilation[1] * (KH - 1) - 1) / stride[1]) + 1;
        const int64_t OW = ((W + 2 * padding[2] - dilation[2] * (KW - 1) - 1) / stride[2]) + 1;
        if (OD <= 0 || OH <= 0 || OW <= 0) fail("conv3d infer produced non-positive output shape");
        shape_env[out] = {N, C_OUT, OD, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "conv_depthwise2d") {
        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv_depthwise2d infer expects inputs [input, weight] or [input, weight, bias]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& w_shape = get_shape(op.inputs[1]);
        const bool has_bias = op.inputs.size() == 3;
        const auto& b_shape = has_bias ? get_shape(op.inputs[2]) : std::vector<int64_t>{};
        if (in_shape.size() != 4 || w_shape.size() != 4) fail("conv_depthwise2d infer expects input[N,C,H,W], weight[C_OUT,1,KH,KW]");
        const int64_t N = in_shape[0];
        const int64_t C_IN = in_shape[1];
        const int64_t H = in_shape[2];
        const int64_t W = in_shape[3];
        const int64_t C_OUT = w_shape[0];
        const int64_t KH = w_shape[2];
        const int64_t KW = w_shape[3];
        if (w_shape[1] != 1) fail("conv_depthwise2d infer expects weight second dim == 1");
        if (C_IN <= 0 || C_OUT <= 0 || (C_OUT % C_IN) != 0) fail("conv_depthwise2d infer channel multiplier mismatch");
        if (has_bias) {
          if (b_shape.size() != 1) fail("conv_depthwise2d infer bias must be rank-1");
          if (b_shape[0] != C_OUT) fail("conv_depthwise2d infer bias shape mismatch");
        }
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("conv_depthwise2d infer " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int64_t, int64_t>{1, 1};
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int64_t, int64_t>{1, 1};
        if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
          fail("conv_depthwise2d infer stride/dilation must be positive");
        const int64_t OH = ((H + 2 * padding.first - dilation.first * (KH - 1) - 1) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - dilation.second * (KW - 1) - 1) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("conv_depthwise2d infer produced non-positive output shape");
        shape_env[out] = {N, C_OUT, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "avg_pool2d") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 4) fail("avg_pool2d infer expects rank-4 NCHW input");
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("avg_pool2d " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int64_t, int64_t>{2, 2};
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        const int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
        const int64_t OH = ((H + 2 * padding.first - ksz.first) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - ksz.second) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("avg_pool2d infer produced non-positive output shape");
        shape_env[out] = {N, C, OH, OW};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "max_pool2d_with_indices") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 4) fail("max_pool2d_with_indices infer expects rank-4 NCHW input");
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int64_t, int64_t> {
          if (value.is_array()) {
            if (value.size() != 2) fail("max_pool2d_with_indices " + key + " must have length 2");
            return {static_cast<int64_t>(resolve_const_value(value[0], bindings)), static_cast<int64_t>(resolve_const_value(value[1], bindings))};
          }
          int64_t v = static_cast<int64_t>(resolve_const_value(value, bindings));
          return {v, v};
        };
        auto ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int64_t, int64_t>{2, 2};
        auto stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        auto padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int64_t, int64_t>{0, 0};
        auto dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int64_t, int64_t>{1, 1};
        bool ceil_mode = op.attrs.value("ceil_mode", false);
        if (ceil_mode) fail("max_pool2d_with_indices infer ceil_mode=true is not supported");
        if (dilation.first != 1 || dilation.second != 1) fail("max_pool2d_with_indices infer dilation != 1 is not supported");
        const int64_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
        const int64_t OH = ((H + 2 * padding.first - ksz.first) / stride.first) + 1;
        const int64_t OW = ((W + 2 * padding.second - ksz.second) / stride.second) + 1;
        if (OH <= 0 || OW <= 0) fail("max_pool2d_with_indices infer produced non-positive output shape");
        shape_env[out] = {N, C, OH, OW};
        std::string select = op.attrs.value("select", "values");
        dtype_env[out] = (select == "indices") ? "i64" : get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "scaled_dot_product_attention") {
        if (op.inputs.size() < 3) fail("scaled_dot_product_attention infer requires query/key/value");
        const auto& q = get_shape(op.inputs[0]);
        const auto& k = get_shape(op.inputs[1]);
        const auto& v = get_shape(op.inputs[2]);
        if (q.size() != 4 || k.size() != 4 || v.size() != 4) fail("scaled_dot_product_attention infer expects rank-4 tensors");
        if (q[0] != k[0] || q[1] != k[1] || k[2] != v[2]) fail("scaled_dot_product_attention infer shape mismatch");
        shape_env[out] = {q[0], q[1], q[2], v[3]};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "mse_loss") {
        if (op.inputs.size() != 2) fail("mse_loss infer expects [input, target]");
        const auto& in_shape = get_shape(op.inputs[0]);
        const auto& tgt_shape = get_shape(op.inputs[1]);
        if (in_shape != tgt_shape) fail("mse_loss infer shape mismatch between input and target");
        std::string reduction = "mean";
        if (op.attrs.contains("reduction")) {
          if (op.attrs["reduction"].is_number_integer()) {
            int mode = op.attrs["reduction"].get<int>();
            if (mode == 0) reduction = "none";
            else if (mode == 2) reduction = "sum";
            else reduction = "mean";
          } else {
            reduction = op.attrs["reduction"].get<std::string>();
          }
        }
        if (reduction == "none") {
          shape_env[out] = in_shape;
        } else {
          shape_env[out] = {};
        }
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "abs" || kind == "floor" || kind == "ceil" || kind == "acos" || kind == "atan" || kind == "cos" || kind == "erf" || kind == "log") {
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
      if (kind == "index_add") {
        if (op.inputs.size() != 3) fail("index_add expects inputs [base, index, src]");
        const auto& base_shape = get_shape(op.inputs[0]);
        const auto& idx_shape = get_shape(op.inputs[1]);
        const auto& src_shape = get_shape(op.inputs[2]);
        if (base_shape.size() != 2 || idx_shape.size() != 1 || src_shape.size() != 2) fail("index_add infer expects base[2], index[1], src[2]");
        int axis = op.attrs.value("axis", 0);
        int axis_norm = axis;
        if (axis_norm < 0) axis_norm += 2;
        if (axis_norm != 0) fail("index_add infer currently supports axis=0 only");
        if (src_shape[0] != idx_shape[0] || src_shape[1] != base_shape[1]) fail("index_add infer shape mismatch");
        shape_env[out] = base_shape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "index_put") {
        if (op.inputs.size() != 4) fail("index_put expects inputs [base, row_idx, col_idx, values]");
        const auto& base_shape = get_shape(op.inputs[0]);
        const auto& row_shape = get_shape(op.inputs[1]);
        const auto& col_shape = get_shape(op.inputs[2]);
        const auto& val_shape = get_shape(op.inputs[3]);
        if (base_shape.size() != 2 || row_shape.size() != 1 || col_shape.size() != 1 || val_shape.size() != 1) {
          fail("index_put infer expects base[2], row_idx[1], col_idx[1], values[1]");
        }
        if (row_shape[0] != col_shape[0] || row_shape[0] != val_shape[0]) fail("index_put infer length mismatch");
        shape_env[out] = base_shape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "kron") {
        if (op.inputs.size() != 2) fail("kron expects inputs [A, B]");
        const auto& a_shape = get_shape(op.inputs[0]);
        const auto& b_shape = get_shape(op.inputs[1]);
        if (a_shape.size() != 2 || b_shape.size() != 2) fail("kron infer currently supports rank-2 tensors");
        shape_env[out] = {a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]};
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
      if (kind == "cumsum" || kind == "cummax" || kind == "cummin") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "nonzero") {
        if (op.inputs.size() != 1) fail("nonzero expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail("nonzero infer currently supports rank-2 input");
        if (!intent.tensors.count(out)) fail("nonzero output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) {
            oshape.push_back(d.get<int64_t>());
          } else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in nonzero output shape: " + sym);
            oshape.push_back(it->second);
          } else {
            fail("invalid nonzero output dim");
          }
        }
        if (oshape.size() != 2 || oshape[1] != 2) fail("nonzero infer expects output shape [K,2]");
        shape_env[out] = oshape;
        dtype_env[out] = out_t.dtype;
        continue;
      }
      if (kind == "nll_loss_forward") {
        if (op.inputs.size() != 3) fail("nll_loss_forward expects inputs [self,target,weight]");
        const auto& logits_shape = get_shape(op.inputs[0]);
        const auto& target_shape = get_shape(op.inputs[1]);
        const auto& weight_shape = get_shape(op.inputs[2]);
        if (logits_shape.size() != 2 || target_shape.size() != 1 || weight_shape.size() != 1) {
          fail("nll_loss_forward infer expects self[N,C], target[N], weight[C]");
        }
        const int64_t N = logits_shape[0];
        const int64_t C = logits_shape[1];
        if (target_shape[0] != N) fail("nll_loss_forward infer target length mismatch");
        if (weight_shape[0] != C) fail("nll_loss_forward infer weight length mismatch");
        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
        if (reduction == 0) shape_env[out] = {N};
        else shape_env[out] = {};
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "nll_loss2d_forward") {
        if (op.inputs.size() != 3) fail("nll_loss2d_forward expects inputs [self,target,weight]");
        const auto& logits_shape = get_shape(op.inputs[0]);
        const auto& target_shape = get_shape(op.inputs[1]);
        const auto& weight_shape = get_shape(op.inputs[2]);
        if (logits_shape.size() != 4 || target_shape.size() != 3 || weight_shape.size() != 1) {
          fail("nll_loss2d_forward infer expects self[N,C,H,W], target[N,H,W], weight[C]");
        }
        const int64_t N = logits_shape[0];
        const int64_t C = logits_shape[1];
        const int64_t H = logits_shape[2];
        const int64_t W = logits_shape[3];
        if (target_shape[0] != N || target_shape[1] != H || target_shape[2] != W) {
          fail("nll_loss2d_forward infer target shape mismatch");
        }
        if (weight_shape[0] != C) fail("nll_loss2d_forward infer weight length mismatch");
        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
        if (reduction == 0) shape_env[out] = {N, H, W};
        else shape_env[out] = {};
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "glu") {
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.empty()) fail("glu expects rank >= 1 input");
        int axis = op.attrs.value("axis", -1);
        int axis_norm = axis;
        if (axis_norm < 0) axis_norm += static_cast<int>(in_shape.size());
        if (axis_norm < 0 || axis_norm >= static_cast<int>(in_shape.size())) fail("glu axis out of range");
        if ((in_shape[axis_norm] % 2) != 0) fail("glu requires even extent along selected axis");
        std::vector<int64_t> oshape = in_shape;
        oshape[axis_norm] = in_shape[axis_norm] / 2;
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "rsqrt" || kind == "exp" || kind == "log" || kind == "relu" || kind == "softmax") {
        shape_env[out] = get_shape(op.inputs[0]);
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "polar") {
        if (op.inputs.size() != 2) fail("polar infer expects inputs [abs, angle]");
        const auto& abs_shape = get_shape(op.inputs[0]);
        const auto& angle_shape = get_shape(op.inputs[1]);
        if (abs_shape != angle_shape) fail("polar infer expects abs/angle shape match");
        if (abs_shape.size() != 2) fail("polar infer currently supports rank-2 inputs");
        shape_env[out] = {abs_shape[0], abs_shape[1], 2};
        dtype_env[out] = "f32";
        continue;
      }
      if (kind == "reduce_sum" || kind == "reduce_prod" || kind == "reduce_max" || kind == "reduce_min" || kind == "reduce_any") {
        const auto& in_shape = get_shape(op.inputs[0]);
        std::unordered_map<int,int> dims_set;
        if (op.attrs.contains("dims") && op.attrs["dims"].is_array()) {
          for (const auto& d : op.attrs["dims"]) dims_set.emplace(d.get<int>(), 1);
        } else if (op.attrs.contains("axes") && op.attrs["axes"].is_array()) {
          for (const auto& d : op.attrs["axes"]) dims_set.emplace(d.get<int>(), 1);
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims_set.emplace(d.get<int>(), 1);
          } else {
            dims_set.emplace(op.attrs["axis"].get<int>(), 1);
          }
        } else {
          fail(kind + " requires dims/axes list[int]");
        }
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
        if (sa.size() == 2 && sb.size() == 1) {
          if (tb) fail("matmul infer rank-2 x rank-1 does not support transpose_b");
          int64_t M = ta ? sa[1] : sa[0];
          int64_t K = ta ? sa[0] : sa[1];
          int64_t K2 = sb[0];
          if (K2 != K) fail("matmul infer shape mismatch (2D x 1D)");
          shape_env[out] = {M};
          dtype_env[out] = "f32";
          continue;
        }
        if (sa.size() == 3 && sb.size() == 3) {
          int64_t B = sa[0];
          int64_t M = ta ? sa[2] : sa[1];
          int64_t K = ta ? sa[1] : sa[2];
          int64_t B2 = sb[0];
          int64_t K2 = tb ? sb[2] : sb[1];
          int64_t N = tb ? sb[1] : sb[2];
          if (B2 != B || K2 != K) fail("matmul infer shape mismatch (3D)");
          shape_env[out] = {B, M, N};
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
        fail("matmul infer supports rank-2/3/4 and rank-2x1");
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
      if (sa.size() == 2 && sb.size() == 1) {
        if (tb) continue;
        int64_t M = ta ? sa[1] : sa[0];
        int64_t K = ta ? sa[0] : sa[1];
        int64_t K2 = sb[0];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)M * (double)K;
        continue;
      }
      if (sa.size() == 3 && sb.size() == 3) {
        int64_t B = sa[0];
        int64_t M = ta ? sa[2] : sa[1];
        int64_t K = ta ? sa[1] : sa[2];
        int64_t K2 = tb ? sb[2] : sb[1];
        int64_t N = tb ? sb[1] : sb[2];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)B * (double)M * (double)N * (double)K;
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
    CProgramEmitter emitter{cw, intent, bindings, external_inputs, shape_env, dtype_env, const_vals, {},
                            sched_tile_m, sched_tile_n, sched_tile_k, sched_vec_width,
                            atol, rtol, matmul_flops_total, (mode == "bench")};
    emitter.emit_program();
    std::cout << c_src.str();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "intentir_codegen error: " << e.what() << "\n";
    return 1;
  }
}
