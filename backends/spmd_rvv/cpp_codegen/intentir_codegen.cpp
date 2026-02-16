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
		        const std::string& var = v(name);
		        std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
		        if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
		        w.line("{\"" + name + "\", (void**)&" + var + ", (size_t)(" + bytes_expr + "), " +
		               buffer_desc_dtype_for_dtype(dt) + "},");
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
		          const std::string& var = v(name);
		          std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
		          if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
		          w.line("{\"" + name + "\", (void**)&" + var + ", (size_t)(" + bytes_expr + "), " +
		                 buffer_desc_dtype_for_dtype(dt) + "},");
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
      if (kind == "add" || kind == "sub" || kind == "mul" || kind == "div" || kind == "max" || kind == "min" || kind == "pow" || kind == "remainder") {
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
      if (kind == "upsample_nearest1d") {
        if (op.inputs.size() != 1) fail("upsample_nearest1d infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 3) fail("upsample_nearest1d infer expects rank-3 input [N,C,IL]");
        if (!intent.tensors.count(out)) fail("upsample_nearest1d output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) oshape.push_back(d.get<int64_t>());
          else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in upsample_nearest1d output shape: " + sym);
            oshape.push_back(it->second);
          } else fail("invalid upsample_nearest1d output dim");
        }
        if (oshape.size() != 3) fail("upsample_nearest1d infer expects rank-3 output [N,C,OL]");
        if (oshape[0] != in_shape[0] || oshape[1] != in_shape[1]) fail("upsample_nearest1d N/C mismatch");
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "upsample_nearest2d") {
        if (op.inputs.size() != 1) fail("upsample_nearest2d infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 4) fail("upsample_nearest2d infer expects rank-4 input [N,C,IH,IW]");
        if (!intent.tensors.count(out)) fail("upsample_nearest2d output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) oshape.push_back(d.get<int64_t>());
          else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in upsample_nearest2d output shape: " + sym);
            oshape.push_back(it->second);
          } else fail("invalid upsample_nearest2d output dim");
        }
        if (oshape.size() != 4) fail("upsample_nearest2d infer expects rank-4 output [N,C,OH,OW]");
        if (oshape[0] != in_shape[0] || oshape[1] != in_shape[1]) fail("upsample_nearest2d N/C mismatch");
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
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
      if (kind == "abs" || kind == "floor" || kind == "ceil" || kind == "acos" || kind == "atan" || kind == "tan" || kind == "cos" || kind == "erf" ||
          kind == "log" || kind == "sqrt") {
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
      if (kind == "std") {
        if (op.inputs.size() != 1) fail("std infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail("std infer currently supports rank-2 input");
        int axis = 1;
        if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            if (op.attrs["axis"].size() != 1) fail("std infer currently supports one reduction axis");
            axis = op.attrs["axis"][0].get<int>();
          } else {
            axis = op.attrs["axis"].get<int>();
          }
        } else if (op.attrs.contains("dims") && op.attrs["dims"].is_array()) {
          if (op.attrs["dims"].size() != 1) fail("std infer currently supports one reduction axis");
          axis = op.attrs["dims"][0].get<int>();
        } else if (op.attrs.contains("axes") && op.attrs["axes"].is_array()) {
          if (op.attrs["axes"].size() != 1) fail("std infer currently supports one reduction axis");
          axis = op.attrs["axes"][0].get<int>();
        }
        int axis_norm = axis;
        if (axis_norm < 0) axis_norm += static_cast<int>(in_shape.size());
        if (axis_norm != 1) fail("std infer currently supports axis=1");
        const bool keepdims = op.attrs.value("keepdims", false);
        if (keepdims) shape_env[out] = {in_shape[0], 1};
        else shape_env[out] = {in_shape[0]};
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "sort") {
        if (op.inputs.size() != 1) fail("sort infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 2) fail("sort infer currently supports rank-2 input");
        int axis = op.attrs.value("axis", -1);
        if (axis < 0) axis += static_cast<int>(in_shape.size());
        if (axis != 1) fail("sort infer currently supports axis=1");
        shape_env[out] = in_shape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "unique") {
        if (op.inputs.size() != 1) fail("unique infer expects 1 input");
        const auto& in_shape = get_shape(op.inputs[0]);
        if (in_shape.size() != 1) fail("unique infer currently supports rank-1 input");
        if (!intent.tensors.count(out)) fail("unique output tensor must exist in intent.tensors");
        const auto& out_t = intent.tensors.at(out);
        std::vector<int64_t> oshape;
        for (const auto& d : out_t.shape) {
          if (d.is_number_integer()) {
            oshape.push_back(d.get<int64_t>());
          } else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it = bindings.find(sym);
            if (it == bindings.end()) fail("unbound symbol in unique output shape: " + sym);
            oshape.push_back(it->second);
          } else {
            fail("invalid unique output dim");
          }
        }
        if (oshape.size() != 1) fail("unique infer expects rank-1 output");
        shape_env[out] = oshape;
        dtype_env[out] = get_dtype(op.inputs[0]);
        continue;
      }
      if (kind == "stack") {
        if (op.inputs.size() != 2) fail("stack infer currently supports exactly 2 inputs");
        const auto& a_shape = get_shape(op.inputs[0]);
        const auto& b_shape = get_shape(op.inputs[1]);
        if (a_shape.size() != 2 || b_shape.size() != 2) fail("stack infer currently supports rank-2 inputs");
        if (a_shape != b_shape) fail("stack infer input shape mismatch");
        int axis = op.attrs.value("axis", 0);
        if (axis < 0) axis += 3;
        if (axis != 0) fail("stack infer currently supports axis=0");
        shape_env[out] = {2, a_shape[0], a_shape[1]};
        dtype_env[out] = get_dtype(op.inputs[0]);
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
