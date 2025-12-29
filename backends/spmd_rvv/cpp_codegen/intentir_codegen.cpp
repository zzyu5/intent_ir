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

std::vector<std::string> emit_elemwise_bin(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
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
  std::vector<std::string> idx = {"i0","i1","i2","i3"};
  idx.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
  std::string out_idx = flat_idx_expr(idx, out_shape);
  auto idx_expr = [&](const std::vector<int64_t>& padded) -> std::string {
    std::vector<std::string> in_vars;
    in_vars.reserve(r);
    for (int i = 0; i < r; ++i) in_vars.push_back(padded[i] == 1 ? "0" : idx[i]);
    return flat_idx_expr(in_vars, padded);
  };
  std::string a_idx = idx_expr(pa);
  std::string b_idx = idx_expr(pb);
  const std::string out_ct = ctype_for_dtype(out_dtype);
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
    const std::string c_op = (op == "add" ? "+" : op == "sub" ? "-" : op == "mul" ? "*" : op == "div" ? "/" : "");
    if (c_op.empty()) fail("unsupported elemwise op: " + op);
    expr = "(" + a + "[" + a_idx + "] " + c_op + " " + b + "[" + b_idx + "])";
  }
  lines.push_back("    " + out + "[" + out_idx + "] = (" + out_ct + ")" + expr + ";");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_cmp(const std::string& cmp_op, const std::string& out, const std::string& a, const std::string& b,
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
  std::vector<std::string> idx = {"i0","i1","i2","i3"};
  idx.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
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
  lines.push_back("    " + out + "[" + out_idx + "] = (" + a + "[" + a_idx + "] " + cop + " " + b + "[" + b_idx + "]) ? 1 : 0;");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_ne(const std::string& out, const std::string& a, const std::string& b,
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
  std::vector<std::string> idx = {"i0","i1","i2","i3"};
  idx.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
  std::string out_idx = flat_idx_expr(idx, out_shape);
  auto idx_expr = [&](const std::vector<int64_t>& padded) -> std::string {
    std::vector<std::string> in_vars;
    in_vars.reserve(r);
    for (int i = 0; i < r; ++i) in_vars.push_back(padded[i] == 1 ? "0" : idx[i]);
    return flat_idx_expr(in_vars, padded);
  };
  std::string a_idx = idx_expr(pa);
  std::string b_idx = idx_expr(pb);
  lines.push_back("    " + out + "[" + out_idx + "] = (" + a + "[" + a_idx + "] != " + b + "[" + b_idx + "]) ? 1 : 0;");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_bool_bin(const std::string& op, const std::string& out, const std::string& a, const std::string& b,
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
  std::vector<std::string> idx = {"i0","i1","i2","i3"};
  idx.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
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
  lines.push_back("    " + out + "[" + out_idx + "] = ((" + a + "[" + a_idx + "] != 0) " + cop + " (" + b + "[" + b_idx + "] != 0)) ? 1 : 0;");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_bool_not(const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int r = static_cast<int>(out_shape.size());
  if (r > 4) fail("not supports rank<=4");
  std::vector<std::string> idx = {"i0","i1","i2","i3"};
  idx.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
  std::string out_idx = flat_idx_expr(idx, out_shape);
  lines.push_back("    " + out + "[" + out_idx + "] = (" + a + "[" + out_idx + "] == 0) ? 1 : 0;");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_unary_abs(const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape, const std::string& in_dt, const std::string& out_dt) {
  const int64_t n = numel(out_shape);
  const std::string in_ct = ctype_for_dtype(in_dt);
  const std::string out_ct = ctype_for_dtype(out_dt);
  std::vector<std::string> lines;
  lines.push_back("  for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  if (in_ct == "float" && out_ct == "float") {
    lines.push_back("    " + out + "[i] = fabsf(" + a + "[i]);");
  } else if (in_ct == "double" && out_ct == "double") {
    lines.push_back("    " + out + "[i] = fabs(" + a + "[i]);");
  } else {
    lines.push_back("    " + in_ct + " x = " + a + "[i];");
    lines.push_back("    " + out + "[i] = (x < 0) ? (" + out_ct + ")(-x) : (" + out_ct + ")x;");
  }
  lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_floor(const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  return {
      "  for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {",
      "    " + out + "[i] = floorf(" + a + "[i]);",
      "  }",
  };
}

std::vector<std::string> emit_rsqrt(const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape) {
  const int64_t n = numel(out_shape);
  return {
      "  for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {",
      "    " + out + "[i] = 1.0f / sqrtf(" + a + "[i]);",
      "  }",
  };
}

std::vector<std::string> emit_cast(const std::string& out, const std::string& a, const std::vector<int64_t>& out_shape, const std::string& from_dt, const std::string& to_dt) {
  const int64_t n = numel(out_shape);
  const std::string from_ct = ctype_for_dtype(from_dt);
  const std::string to_ct = ctype_for_dtype(to_dt);
  std::vector<std::string> lines;
  lines.push_back("  for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) {");
  if (to_ct == "uint8_t") {
    lines.push_back("    " + out + "[i] = (" + a + "[i] != 0) ? 1 : 0;");
  } else if (from_ct == "uint8_t" && (to_ct == "float" || to_ct == "double")) {
    const std::string one = (to_ct == "float") ? "1.0f" : "1.0";
    const std::string zero = (to_ct == "float") ? "0.0f" : "0.0";
    lines.push_back("    " + out + "[i] = (" + a + "[i] != 0) ? " + one + " : " + zero + ";");
  } else {
    lines.push_back("    " + out + "[i] = (" + to_ct + ")(" + a + "[i]);");
  }
  lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_where(const std::string& out, const std::string& cond, const std::string& x, const std::string& y,
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
  std::vector<std::string> idx = {"i0","i1","i2","i3"};
  idx.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
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
  lines.push_back("    " + out + "[" + out_idx + "] = (" + cond + "[" + c_idx + "] != 0) ? (" + out_ct + ")" + x + "[" + x_idx + "] : (" + out_ct + ")" + y + "[" + y_idx + "];");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_iota(const std::string& out, const std::vector<int64_t>& out_shape, int axis, const std::string& out_dtype) {
  const int r = static_cast<int>(out_shape.size());
  if (r == 0) fail("iota requires rank>=1");
  if (r > 4) fail("iota supports rank<=4");
  if (axis < 0 || axis >= r) fail("iota axis out of range");
  std::vector<std::string> idx = {"i0","i1","i2","i3"};
  idx.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + idx[i] + " = 0; " + idx[i] + " < " + std::to_string(out_shape[i]) + "; ++" + idx[i] + ") {");
  std::string out_idx = flat_idx_expr(idx, out_shape);
  const std::string out_ct = ctype_for_dtype(out_dtype);
  lines.push_back("    " + out + "[" + out_idx + "] = (" + out_ct + ")" + idx[axis] + ";");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_gather(const std::string& out, const std::string& data, const std::vector<std::string>& idxs,
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

  std::vector<std::string> iv = {"i0","i1","i2","i3"};
  iv.resize(r);
  std::vector<std::string> lines;
  for (int i = 0; i < r; ++i) lines.push_back("  for (int " + iv[i] + " = 0; " + iv[i] + " < " + std::to_string(out_shape[i]) + "; ++" + iv[i] + ") {");
  std::string out_idx = flat_idx_expr(iv, out_shape);
  auto idx_expr = [&](const std::vector<int64_t>& ps) -> std::string {
    std::vector<std::string> vars;
    vars.reserve(r);
    for (int i = 0; i < r; ++i) vars.push_back(ps[i] == 1 ? "0" : iv[i]);
    return flat_idx_expr(vars, ps);
  };
  for (int ax = 0; ax < r_data; ++ax) {
    lines.push_back("    int d" + std::to_string(ax) + " = (int)" + idxs[ax] + "[" + idx_expr(padded[ax]) + "];");
  }
  if (r_data == 1) lines.push_back("    size_t di = (size_t)d0;");
  else if (r_data == 2) lines.push_back("    size_t di = idx2(d0,d1," + std::to_string(data_shape[1]) + ");");
  else if (r_data == 3) lines.push_back("    size_t di = idx3(d0,d1,d2," + std::to_string(data_shape[1]) + "," + std::to_string(data_shape[2]) + ");");
  else lines.push_back("    size_t di = idx4(d0,d1,d2,d3," + std::to_string(data_shape[1]) + "," + std::to_string(data_shape[2]) + "," + std::to_string(data_shape[3]) + ");");
  const std::string out_ct = ctype_for_dtype(out_dtype);
  lines.push_back("    " + out + "[" + out_idx + "] = (" + out_ct + ")" + data + "[di];");
  for (int i = 0; i < r; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_reduce_sum(const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
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

  std::vector<std::string> out_vars;
  for (int i = 0; i < out_rank; ++i) out_vars.push_back("o" + std::to_string(i));
  std::vector<std::string> lines;
  for (int i = 0; i < out_rank; ++i) lines.push_back("  for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  lines.push_back("    double acc = 0.0;");

  // build in_vars mapping
  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) lines.push_back("    for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  lines.push_back("      acc += (double)" + a + "[" + in_idx + "];");
  for (size_t i = 0; i < dims_set.size(); ++i) lines.push_back("    }");
  if (scale.has_value()) lines.push_back("    acc *= (double)" + c_float(scale.value()) + ";");
  lines.push_back("    " + out + "[" + out_idx + "] = (float)acc;");
  for (int i = 0; i < out_rank; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_reduce_any(const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
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
  std::vector<std::string> lines;
  for (int i = 0; i < out_rank; ++i) lines.push_back("  for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  lines.push_back("    uint8_t acc = 0;");
  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) lines.push_back("    for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  lines.push_back("      acc = acc || (" + a + "[" + in_idx + "] != 0);");
  for (size_t i = 0; i < dims_set.size(); ++i) lines.push_back("    }");
  lines.push_back("    " + out + "[" + out_idx + "] = acc;");
  for (int i = 0; i < out_rank; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_reduce_max(const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
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
  std::vector<std::string> out_vars;
  for (int i = 0; i < out_rank; ++i) out_vars.push_back("o" + std::to_string(i));
  std::vector<std::string> lines;
  for (int i = 0; i < out_rank; ++i) lines.push_back("  for (int " + out_vars[i] + " = 0; " + out_vars[i] + " < " + std::to_string(OD[i]) + "; ++" + out_vars[i] + ") {");
  std::string out_idx = flat_idx_expr(out_vars, out_shape);
  lines.push_back("    float m = -INFINITY;");
  std::vector<std::string> in_vars(r);
  int out_it = 0;
  for (int di = 0; di < r; ++di) {
    bool reduced = false;
    for (int d : dims_set) if (d == di) reduced = true;
    if (reduced) in_vars[di] = "r" + std::to_string(di);
    else in_vars[di] = out_vars[out_it++];
  }
  for (int d : dims_set) lines.push_back("    for (int r" + std::to_string(d) + " = 0; r" + std::to_string(d) + " < " + std::to_string(D[d]) + "; ++r" + std::to_string(d) + ") {");
  std::string in_idx = flat_idx_expr(in_vars, in_shape);
  lines.push_back("      float v = " + a + "[" + in_idx + "];");
  lines.push_back("      m = fmaxf(m, v);");
  for (size_t i = 0; i < dims_set.size(); ++i) lines.push_back("    }");
  lines.push_back("    " + out + "[" + out_idx + "] = m;");
  for (int i = 0; i < out_rank; ++i) lines.push_back("  }");
  return lines;
}

std::vector<std::string> emit_softmax(const std::string& out, const std::string& a, const std::vector<int64_t>& in_shape, int axis) {
  const int r = static_cast<int>(in_shape.size());
  if (r == 0) fail("softmax on scalar unsupported");
  int ax = axis;
  if (ax < 0) ax += r;
  if (ax != r - 1) fail("softmax supports axis == last only");
  std::vector<std::string> lines;
  if (r == 1) {
    int64_t K = in_shape[0];
    lines = {
        "  {",
        "    double mx = -1e30;",
        "    for (int k = 0; k < " + std::to_string(K) + "; ++k) { double v = (double)" + a + "[(size_t)k]; if (v > mx) mx = v; }",
        "    double sum = 0.0;",
        "    for (int k = 0; k < " + std::to_string(K) + "; ++k) { double e = exp((double)" + a + "[(size_t)k] - mx); " + out + "[(size_t)k] = (float)e; sum += e; }",
        "    double inv = 1.0 / sum;",
        "    for (int k = 0; k < " + std::to_string(K) + "; ++k) { " + out + "[(size_t)k] = (float)((double)" + out + "[(size_t)k] * inv); }",
        "  }",
    };
    return lines;
  }
  if (r == 2) {
    int64_t M = in_shape[0], K = in_shape[1];
    lines.push_back("  for (int m = 0; m < " + std::to_string(M) + "; ++m) {");
    lines.push_back("    double mx = -1e30;");
    lines.push_back("    for (int k = 0; k < " + std::to_string(K) + "; ++k) { double v = (double)" + a + "[idx2(m,k," + std::to_string(K) + ")]; if (v > mx) mx = v; }");
    lines.push_back("    double sum = 0.0;");
    lines.push_back("    for (int k = 0; k < " + std::to_string(K) + "; ++k) { double e = exp((double)" + a + "[idx2(m,k," + std::to_string(K) + ")] - mx); " + out + "[idx2(m,k," + std::to_string(K) + ")] = (float)e; sum += e; }");
    lines.push_back("    double inv = 1.0 / sum;");
    lines.push_back("    for (int k = 0; k < " + std::to_string(K) + "; ++k) { " + out + "[idx2(m,k," + std::to_string(K) + ")] = (float)((double)" + out + "[idx2(m,k," + std::to_string(K) + ")] * inv); }");
    lines.push_back("  }");
    return lines;
  }
  if (r == 3) {
    int64_t A0 = in_shape[0], A1 = in_shape[1], K = in_shape[2];
    lines.push_back("  for (int i0 = 0; i0 < " + std::to_string(A0) + "; ++i0) {");
    lines.push_back("    for (int i1 = 0; i1 < " + std::to_string(A1) + "; ++i1) {");
    lines.push_back("      double mx = -1e30;");
    lines.push_back("      for (int k = 0; k < " + std::to_string(K) + "; ++k) { double v = (double)" + a + "[idx3(i0,i1,k," + std::to_string(A1) + "," + std::to_string(K) + ")]; if (v > mx) mx = v; }");
    lines.push_back("      double sum = 0.0;");
    lines.push_back("      for (int k = 0; k < " + std::to_string(K) + "; ++k) { double e = exp((double)" + a + "[idx3(i0,i1,k," + std::to_string(A1) + "," + std::to_string(K) + ")] - mx); " + out + "[idx3(i0,i1,k," + std::to_string(A1) + "," + std::to_string(K) + ")] = (float)e; sum += e; }");
    lines.push_back("      double inv = 1.0 / sum;");
    lines.push_back("      for (int k = 0; k < " + std::to_string(K) + "; ++k) { " + out + "[idx3(i0,i1,k," + std::to_string(A1) + "," + std::to_string(K) + ")] = (float)((double)" + out + "[idx3(i0,i1,k," + std::to_string(A1) + "," + std::to_string(K) + ")] * inv); }");
    lines.push_back("    }");
    lines.push_back("  }");
    return lines;
  }
  if (r == 4) {
    int64_t B = in_shape[0], H = in_shape[1], Q = in_shape[2], K = in_shape[3];
    lines.push_back("  for (int b = 0; b < " + std::to_string(B) + "; ++b) {");
    lines.push_back("    for (int h = 0; h < " + std::to_string(H) + "; ++h) {");
    lines.push_back("      for (int q = 0; q < " + std::to_string(Q) + "; ++q) {");
    lines.push_back("        double mx = -1e30;");
    lines.push_back("        for (int k = 0; k < " + std::to_string(K) + "; ++k) { double v = (double)" + a + "[idx4(b,h,q,k," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(K) + ")]; if (v > mx) mx = v; }");
    lines.push_back("        double sum = 0.0;");
    lines.push_back("        for (int k = 0; k < " + std::to_string(K) + "; ++k) { double e = exp((double)" + a + "[idx4(b,h,q,k," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(K) + ")] - mx); " + out + "[idx4(b,h,q,k," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(K) + ")] = (float)e; sum += e; }");
    lines.push_back("        double inv = 1.0 / sum;");
    lines.push_back("        for (int k = 0; k < " + std::to_string(K) + "; ++k) { " + out + "[idx4(b,h,q,k," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(K) + ")] = (float)((double)" + out + "[idx4(b,h,q,k," + std::to_string(H) + "," + std::to_string(Q) + "," + std::to_string(K) + ")] * inv); }");
    lines.push_back("      }");
    lines.push_back("    }");
    lines.push_back("  }");
    return lines;
  }
  fail("softmax supports rank<=4");
}

std::vector<std::string> emit_transpose_4d_0132(const std::string& out, const std::string& inp, const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape) {
  if (in_shape.size() != 4 || out_shape.size() != 4) fail("transpose expects rank-4");
  int64_t B = in_shape[0], H = in_shape[1], K = in_shape[2], D = in_shape[3];
  return {
      "  for (int b = 0; b < " + std::to_string(B) + "; ++b) {",
      "    for (int h = 0; h < " + std::to_string(H) + "; ++h) {",
      "      for (int k = 0; k < " + std::to_string(K) + "; ++k) {",
      "        for (int d = 0; d < " + std::to_string(D) + "; ++d) {",
      "          " + out + "[idx4(b,h,d,k," + std::to_string(H) + "," + std::to_string(out_shape[2]) + "," + std::to_string(out_shape[3]) + ")] = " + inp + "[idx4(b,h,k,d," + std::to_string(H) + "," + std::to_string(K) + "," + std::to_string(D) + ")];",
      "        }",
      "      }",
      "    }",
      "  }",
  };
}

std::vector<std::string> emit_matmul(const std::string& out, const std::string& a, const std::string& b,
                                     const std::vector<int64_t>& a_shape, const std::vector<int64_t>& b_shape, const std::vector<int64_t>& out_shape,
                                     bool transpose_a, bool transpose_b) {
  // Scalar matmul (rank-2 or rank-4).
  std::vector<std::string> lines;
  if (a_shape.size() == 2 && b_shape.size() == 2) {
    int64_t M = transpose_a ? a_shape[1] : a_shape[0];
    int64_t K = transpose_a ? a_shape[0] : a_shape[1];
    int64_t K2 = transpose_b ? b_shape[1] : b_shape[0];
    int64_t N = transpose_b ? b_shape[0] : b_shape[1];
    if (K2 != K) fail("matmul shape mismatch (2D)");
    if (out_shape.size() != 2 || out_shape[0] != M || out_shape[1] != N) fail("matmul output shape mismatch (2D)");
    lines.push_back("  for (int m = 0; m < " + std::to_string(M) + "; ++m) {");
    lines.push_back("    for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
    lines.push_back("      double acc = 0.0;");
    lines.push_back("      for (int k = 0; k < " + std::to_string(K) + "; ++k) {");
    std::string a_idx = transpose_a ? ("idx2(k,m," + std::to_string(M) + ")") : ("idx2(m,k," + std::to_string(K) + ")");
    std::string b_idx = transpose_b ? ("idx2(n,k," + std::to_string(K) + ")") : ("idx2(k,n," + std::to_string(N) + ")");
    lines.push_back("        acc += (double)" + a + "[" + a_idx + "] * (double)" + b + "[" + b_idx + "];");
    lines.push_back("      }");
    lines.push_back("      " + out + "[idx2(m,n," + std::to_string(N) + ")] = (float)acc;");
    lines.push_back("    }");
    lines.push_back("  }");
    return lines;
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
    lines.push_back("  for (int b0 = 0; b0 < " + std::to_string(B0) + "; ++b0) {");
    lines.push_back("    for (int h0 = 0; h0 < " + std::to_string(H0) + "; ++h0) {");
    lines.push_back("      for (int m0 = 0; m0 < " + std::to_string(M) + "; ++m0) {");
    lines.push_back("        for (int n0 = 0; n0 < " + std::to_string(N) + "; ++n0) {");
    lines.push_back("          double acc = 0.0;");
    lines.push_back("          for (int k0 = 0; k0 < " + std::to_string(K) + "; ++k0) {");
    std::string a_idx = transpose_a
                            ? ("idx4(b0,h0,k0,m0," + std::to_string(H0) + "," + std::to_string(K) + "," + std::to_string(M) + ")")
                            : ("idx4(b0,h0,m0,k0," + std::to_string(H0) + "," + std::to_string(M) + "," + std::to_string(K) + ")");
    std::string b_idx = transpose_b
                            ? ("idx4(b0,h0,n0,k0," + std::to_string(H0) + "," + std::to_string(N) + "," + std::to_string(K) + ")")
                            : ("idx4(b0,h0,k0,n0," + std::to_string(H0) + "," + std::to_string(K) + "," + std::to_string(N) + ")");
    lines.push_back("            acc += (double)" + a + "[" + a_idx + "] * (double)" + b + "[" + b_idx + "];");
    lines.push_back("          }");
    lines.push_back("          " + out + "[idx4(b0,h0,m0,n0," + std::to_string(H0) + "," + std::to_string(M) + "," + std::to_string(N) + ")] = (float)acc;");
    lines.push_back("        }");
    lines.push_back("      }");
    lines.push_back("    }");
    lines.push_back("  }");
    return lines;
  }
  fail("matmul supports rank-2 or rank-4");
}

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
    struct ConstVal { std::string dtype; double value; };
    std::unordered_map<std::string, ConstVal> const_vals;
    for (const auto& op : intent.ops) {
      if (op.op == "const") {
        const json v = op.attrs.value("value", json());
        const std::string dt = op.attrs.value("dtype", dtype_env.count(op.output) ? dtype_env[op.output] : std::string("f32"));
        const_vals[op.output] = {dt, resolve_const_value(v, bindings)};
      }
    }

    // ---- emit C program ----
    std::vector<std::string> lines;
    lines.push_back("#include <math.h>");
    lines.push_back("#include <stdint.h>");
    lines.push_back("#include <stddef.h>");
    lines.push_back("#include <stdio.h>");
    lines.push_back("#include <stdlib.h>");
    lines.push_back("#if defined(__riscv_vector) || defined(__riscv_v)\n#include <riscv_vector.h>\n#endif");
    lines.push_back("");
    lines.push_back("static int read_bytes(const char* path, void* dst, size_t bytes) {");
    lines.push_back("  FILE* f = fopen(path, \"rb\");");
    lines.push_back("  if (!f) { perror(path); return 0; }");
    lines.push_back("  size_t got = fread(dst, 1, bytes, f);");
    lines.push_back("  fclose(f);");
    lines.push_back("  return got == bytes;");
    lines.push_back("}");
    lines.push_back("");
    lines.push_back("static int compare_f32(const char* name, const float* got, const float* ref, size_t n, float atol, float rtol) {");
    lines.push_back("  double max_abs = 0.0, max_rel = 0.0; size_t worst = 0;");
    lines.push_back("  for (size_t i = 0; i < n; ++i) {");
    lines.push_back("    double a = (double)got[i];");
    lines.push_back("    double b = (double)ref[i];");
    lines.push_back("    double abs_e = fabs(a - b);");
    lines.push_back("    double rel_e = abs_e / (fabs(b) + 1e-8);");
    lines.push_back("    if (abs_e > max_abs) { max_abs = abs_e; max_rel = rel_e; worst = i; }");
    lines.push_back("  }");
    lines.push_back("  int ok = (max_abs <= (double)atol) || (max_rel <= (double)rtol);");
    lines.push_back("  printf(\"%s: ok=%d max_abs=%g max_rel=%g worst_i=%zu got=%g ref=%g\\n\",");
    lines.push_back("         name, ok, max_abs, max_rel, worst, (double)got[worst], (double)ref[worst]);");
    lines.push_back("  return ok;");
    lines.push_back("}");
    lines.push_back("");
    lines.push_back("static int compare_u8(const char* name, const uint8_t* got, const uint8_t* ref, size_t n) {");
    lines.push_back("  for (size_t i = 0; i < n; ++i) {");
    lines.push_back("    if (got[i] != ref[i]) {");
    lines.push_back("      fprintf(stderr, \"%s mismatch at %zu: got=%u ref=%u\\n\", name, i, (unsigned)got[i], (unsigned)ref[i]);");
    lines.push_back("      return 0;");
    lines.push_back("    }");
    lines.push_back("  }");
    lines.push_back("  printf(\"%s: ok=1 (exact)\\n\", name);");
    lines.push_back("  return 1;");
    lines.push_back("}");
    lines.push_back("");
    lines.push_back("static inline size_t idx2(int i, int j, int D1) { return (size_t)i * (size_t)D1 + (size_t)j; }");
    lines.push_back("static inline size_t idx3(int i, int j, int k, int D1, int D2) { return ((size_t)i * (size_t)D1 + (size_t)j) * (size_t)D2 + (size_t)k; }");
    lines.push_back("static inline size_t idx4(int i, int j, int k, int l, int D1, int D2, int D3) { return (((size_t)i * (size_t)D1 + (size_t)j) * (size_t)D2 + (size_t)k) * (size_t)D3 + (size_t)l; }");
    lines.push_back("");
    lines.push_back("int main() {");
    lines.push_back("  setvbuf(stdout, NULL, _IONBF, 0);");
    lines.push_back("  setvbuf(stderr, NULL, _IONBF, 0);");
    lines.push_back("  const float ATOL = " + c_float(atol) + ";");
    lines.push_back("  const float RTOL = " + c_float(rtol) + ";");
    lines.push_back("");

    // inputs
    for (const auto& name : external_inputs) {
      const auto& shp = shape_env[name];
      int64_t n = numel(shp);
      std::string ct = ctype_for_dtype(dtype_env[name]);
      lines.push_back("  // input " + name + ": shape=" + std::to_string((int)shp.size()) + "D");
      lines.push_back("  " + ct + "* " + name + " = (" + ct + "*)malloc(sizeof(" + ct + ") * (size_t)" + std::to_string(n) + ");");
      lines.push_back("  if (!" + name + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
      lines.push_back("  if (!read_bytes(\"" + name + ".bin\", " + name + ", sizeof(" + ct + ") * (size_t)" + std::to_string(n) + ")) return 2;");
      lines.push_back("");
    }

    // consts as 1-element arrays
    for (const auto& kv : const_vals) {
      const std::string& name = kv.first;
      const ConstVal& cv = kv.second;
      const std::string ct = ctype_for_dtype(cv.dtype);
      lines.push_back("  " + ct + "* " + name + " = (" + ct + "*)malloc(sizeof(" + ct + "));");
      lines.push_back("  if (!" + name + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
      if (ct == "float") lines.push_back("  " + name + "[0] = " + c_float(cv.value) + ";");
      else if (ct == "double") {
        std::ostringstream oss; oss.precision(17); oss << cv.value;
        lines.push_back("  " + name + "[0] = (double)" + oss.str() + ";");
      } else lines.push_back("  " + name + "[0] = (" + ct + ")" + std::to_string((int64_t)cv.value) + ";");
      lines.push_back("");
    }

    auto alloc_out = [&](const std::string& name) {
      const auto& shp = shape_env[name];
      int64_t n = numel(shp);
      std::string ct = ctype_for_dtype(dtype_env[name]);
      lines.push_back("  " + ct + "* " + name + " = (" + ct + "*)malloc(sizeof(" + ct + ") * (size_t)" + std::to_string(n) + ");");
      lines.push_back("  if (!" + name + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
    };

    // ops
    for (const auto& op : intent.ops) {
      if (op.op == "const") continue;
      const std::string& out = op.output;
      if (op.op == "reshape" || op.op == "identity" || op.op == "layout_cast") {
        std::string ct = ctype_for_dtype(dtype_env[out]);
        lines.push_back("  // " + op.op + " alias");
        lines.push_back("  " + ct + "* " + out + " = (" + ct + "*)" + op.inputs[0] + ";");
        lines.push_back("");
        continue;
      }
      // allocate output
      lines.push_back("  // op " + op.op + " -> " + out);
      alloc_out(out);

      const std::vector<int64_t>& out_shape = shape_env[out];
      if (op.op == "transpose") {
        const auto& in_shape = shape_env[op.inputs[0]];
        std::vector<int> perm;
        for (const auto& p : op.attrs["perm"]) perm.push_back(p.get<int>());
        if (perm.size() != 4 || perm[0] != 0 || perm[1] != 1 || perm[2] != 3 || perm[3] != 2) fail("transpose supports only perm [0,1,3,2]");
        auto bl = emit_transpose_4d_0132(out, op.inputs[0], in_shape, out_shape);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "broadcast_in_dim") {
        // Use the scalar broadcast emitter logic by lowering it via per-index mapping (rank<=4).
        // For simplicity in this tool, we reuse the elementwise broadcast helper by emitting explicit loops here.
        // (This keeps behavior aligned with Python backend.)
        const std::string& inp = op.inputs[0];
        const auto& in_shape = shape_env[inp];
        std::vector<int> bcast_dims;
        for (const auto& d : op.attrs["broadcast_dims"]) bcast_dims.push_back(d.get<int>());
        // Build strides
        std::vector<int64_t> strides(in_shape.size(), 1);
        int64_t s = 1;
        for (int i = (int)in_shape.size() - 1; i >= 0; --i) { strides[i] = s; s *= in_shape[i]; }
        const int r_out = (int)out_shape.size();
        if (r_out > 4) fail("broadcast_in_dim supports rank<=4");
        std::vector<std::string> iv = {"i0","i1","i2","i3"};
        iv.resize(r_out);
        for (int i = 0; i < r_out; ++i) lines.push_back("  for (int " + iv[i] + " = 0; " + iv[i] + " < " + std::to_string(out_shape[i]) + "; ++" + iv[i] + ") {");
        std::string out_idx = flat_idx_expr(iv, out_shape);
        // input idx expression
        std::vector<std::string> terms;
        for (int in_dim = 0; in_dim < (int)in_shape.size(); ++in_dim) {
          int od = bcast_dims[in_dim];
          if (in_shape[in_dim] == 1) terms.push_back("0");
          else terms.push_back("((size_t)" + iv[od] + " * (size_t)" + std::to_string(strides[in_dim]) + ")");
        }
        std::string in_idx = terms.empty() ? "0" : terms[0];
        for (size_t t = 1; t < terms.size(); ++t) in_idx += " + " + terms[t];
        lines.push_back("    " + out + "[" + out_idx + "] = " + inp + "[" + in_idx + "];");
        for (int i = 0; i < r_out; ++i) lines.push_back("  }");
      } else if (op.op == "add" || op.op == "sub" || op.op == "mul" || op.op == "div" || op.op == "max" || op.op == "min") {
        auto bl = emit_elemwise_bin(intent, bindings, op.op, out, op.inputs[0], op.inputs[1], shape_env[op.inputs[0]], shape_env[op.inputs[1]], out_shape, dtype_env[out]);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "ne") {
        auto bl = emit_ne(out, op.inputs[0], op.inputs[1], shape_env[op.inputs[0]], shape_env[op.inputs[1]], out_shape, intent, bindings);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "lt" || op.op == "le" || op.op == "gt" || op.op == "ge") {
        auto bl = emit_cmp(op.op, out, op.inputs[0], op.inputs[1], shape_env[op.inputs[0]], shape_env[op.inputs[1]], out_shape, intent, bindings);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "and" || op.op == "or") {
        auto bl = emit_bool_bin(op.op, out, op.inputs[0], op.inputs[1], shape_env[op.inputs[0]], shape_env[op.inputs[1]], out_shape);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "not") {
        auto bl = emit_bool_not(out, op.inputs[0], out_shape);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "rsqrt") {
        auto bl = emit_rsqrt(out, op.inputs[0], out_shape);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "abs") {
        auto bl = emit_unary_abs(out, op.inputs[0], out_shape, dtype_env[op.inputs[0]], dtype_env[out]);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "floor") {
        auto bl = emit_floor(out, op.inputs[0], out_shape);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "cast") {
        auto bl = emit_cast(out, op.inputs[0], out_shape, dtype_env[op.inputs[0]], dtype_env[out]);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "where") {
        auto bl = emit_where(out, op.inputs[0], op.inputs[1], op.inputs[2], shape_env[op.inputs[0]], shape_env[op.inputs[1]], shape_env[op.inputs[2]], out_shape, dtype_env[out]);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "iota") {
        int axis = op.attrs.value("axis", 0);
        auto bl = emit_iota(out, out_shape, axis, dtype_env[out]);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "gather") {
        std::vector<std::string> idxs(op.inputs.begin() + 1, op.inputs.end());
        std::vector<std::vector<int64_t>> idx_shapes;
        for (const auto& nm : idxs) idx_shapes.push_back(shape_env[nm]);
        auto bl = emit_gather(out, op.inputs[0], idxs, shape_env[op.inputs[0]], idx_shapes, out_shape, dtype_env[out]);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "reduce_sum") {
        std::vector<int> dims;
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        bool keepdims = op.attrs.value("keepdims", false);
        std::optional<double> scale;
        if (op.attrs.contains("scale")) scale = resolve_const_value(op.attrs["scale"], bindings);
        auto bl = emit_reduce_sum(out, op.inputs[0], shape_env[op.inputs[0]], out_shape, dims, keepdims, scale);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "reduce_max") {
        std::vector<int> dims;
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        bool keepdims = op.attrs.value("keepdims", false);
        auto bl = emit_reduce_max(out, op.inputs[0], shape_env[op.inputs[0]], out_shape, dims, keepdims);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "reduce_any") {
        std::vector<int> dims;
        for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        bool keepdims = op.attrs.value("keepdims", false);
        auto bl = emit_reduce_any(out, op.inputs[0], shape_env[op.inputs[0]], out_shape, dims, keepdims);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "exp") {
        int64_t n = numel(out_shape);
        lines.push_back("  for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) { " + out + "[i] = expf(" + op.inputs[0] + "[i]); }");
      } else if (op.op == "relu") {
        int64_t n = numel(out_shape);
        lines.push_back("  for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) { float v = " + op.inputs[0] + "[i]; " + out + "[i] = v > 0.0f ? v : 0.0f; }");
      } else if (op.op == "softmax") {
        int axis = op.attrs.value("axis", -1);
        auto bl = emit_softmax(out, op.inputs[0], shape_env[op.inputs[0]], axis);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else if (op.op == "matmul") {
        bool ta = op.attrs.value("transpose_a", false);
        bool tb = op.attrs.value("transpose_b", false);
        auto bl = emit_matmul(out, op.inputs[0], op.inputs[1], shape_env[op.inputs[0]], shape_env[op.inputs[1]], out_shape, ta, tb);
        lines.insert(lines.end(), bl.begin(), bl.end());
      } else {
        fail("unsupported op lowering: " + op.op);
      }
      lines.push_back("");
    }

    // compare outputs
    std::vector<std::string> ok_exprs;
    for (const auto& name : intent.outputs) {
      const auto& shp = shape_env[name];
      int64_t n = numel(shp);
      std::string dt = dtype_env[name];
      if (dt == "bool" || dt == "i1") {
        lines.push_back("  uint8_t* " + name + "_ref = (uint8_t*)malloc(sizeof(uint8_t) * (size_t)" + std::to_string(n) + ");");
        lines.push_back("  if (!" + name + "_ref) { fprintf(stderr, \"alloc failed: " + name + "_ref\\n\"); return 2; }");
        lines.push_back("  if (!read_bytes(\"" + name + "_ref.bin\", " + name + "_ref, sizeof(uint8_t) * (size_t)" + std::to_string(n) + ")) return 2;");
        lines.push_back("  int ok_" + name + " = compare_u8(\"" + name + "\", (const uint8_t*)" + name + ", (const uint8_t*)" + name + "_ref, (size_t)" + std::to_string(n) + ");");
      } else {
        lines.push_back("  float* " + name + "_ref = (float*)malloc(sizeof(float) * (size_t)" + std::to_string(n) + ");");
        lines.push_back("  if (!" + name + "_ref) { fprintf(stderr, \"alloc failed: " + name + "_ref\\n\"); return 2; }");
        lines.push_back("  if (!read_bytes(\"" + name + "_ref.bin\", " + name + "_ref, sizeof(float) * (size_t)" + std::to_string(n) + ")) return 2;");
        lines.push_back("  int ok_" + name + " = compare_f32(\"" + name + "\", (const float*)" + name + ", (const float*)" + name + "_ref, (size_t)" + std::to_string(n) + ", ATOL, RTOL);");
      }
      ok_exprs.push_back("ok_" + name);
      lines.push_back("");
    }
    if (ok_exprs.empty()) lines.push_back("  int ok = 1;");
    else {
      std::string expr = ok_exprs[0];
      for (size_t i = 1; i < ok_exprs.size(); ++i) expr += " && " + ok_exprs[i];
      lines.push_back("  int ok = " + expr + ";");
    }
    lines.push_back("  printf(ok ? \"PASS lowered\\n\" : \"FAIL lowered\\n\");");
    lines.push_back("  return ok ? 0 : 1;");
    lines.push_back("}");

    for (const auto& l : lines) std::cout << l << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "intentir_codegen error: " << e.what() << "\n";
    return 1;
  }
}
