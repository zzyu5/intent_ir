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

#include "const_expr_eval.inc"
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

#include "driver_main.inc"
