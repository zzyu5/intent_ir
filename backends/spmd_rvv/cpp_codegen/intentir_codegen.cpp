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
#include "common_utils.h"
#include "ir_model.h"
#include "shape_eval.h"

using json = nlohmann::json;

namespace {
using namespace intentir_rvv_codegen;

struct ConstVal {
  std::string dtype;
  double value = 0.0;
};

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

#include "emit_elemwise_reduce_helpers.inc"
#include "emit_pool_attention.inc"

#include "emit_layout_matmul_helpers.inc"
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

  #include "emit_program_stages.inc"
  #include "emit_compute_fn.inc"
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
#include "infer_shapes_dispatch.inc"
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
