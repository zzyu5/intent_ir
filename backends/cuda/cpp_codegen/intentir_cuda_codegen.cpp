#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
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

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

std::string c_float(double x) {
  if (std::isnan(x)) return "NAN";
  if (std::isinf(x)) return x > 0 ? "INFINITY" : "(-INFINITY)";
  std::ostringstream oss;
  oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
  oss.precision(10);
  oss << x;
  std::string s = oss.str();
  const bool has_dot = (s.find('.') != std::string::npos) || (s.find('e') != std::string::npos) || (s.find('E') != std::string::npos);
  if (!has_dot) s += ".0";
  return s + "f";
}

std::string ascii_lower(std::string s) {
  for (auto& ch : s) {
    if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
  }
  return s;
}

json read_json_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) fail("failed to open: " + path);
  json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    fail("failed to parse json: " + path + " (" + e.what() + ")");
  }
  return j;
}

Intent parse_intent(const json& j) {
  if (!j.is_object()) fail("intent must be object");
  Intent out;
  out.name = j.value("name", "intent_fn");

  json tensors_json = j.value("tensors", json::object());
  if (!tensors_json.is_object()) fail("intent.tensors must be object");
  for (auto it = tensors_json.begin(); it != tensors_json.end(); ++it) {
    const std::string name = it.key();
    const json& tj = it.value();
    if (!tj.is_object()) fail("tensor must be object: " + name);
    Tensor t;
    t.dtype = tj.value("dtype", "f32");
    json shape_json = tj.value("shape", json::array());
    if (!shape_json.is_array()) fail("tensor.shape must be list: " + name);
    for (const auto& d : shape_json) t.shape.push_back(d);
    out.tensors.emplace(name, std::move(t));
  }

  json ops_json = j.value("ops", json::array());
  if (!ops_json.is_array()) fail("intent.ops must be list");
  for (const auto& oj : ops_json) {
    if (!oj.is_object()) fail("op must be object");
    Op o;
    o.op = oj.value("op", "");
    o.output = oj.value("output", "");
    o.attrs = oj.value("attrs", json::object());
    json inputs_json = oj.value("inputs", json::array());
    if (!inputs_json.is_array()) fail("op.inputs must be list");
    for (const auto& x : inputs_json) o.inputs.push_back(x.get<std::string>());
    out.ops.push_back(std::move(o));
  }

  json outs_json = j.value("outputs", json::array());
  if (!outs_json.is_array()) fail("intent.outputs must be list");
  for (const auto& x : outs_json) out.outputs.push_back(x.get<std::string>());

  out.schedule = j.value("schedule", json());
  return out;
}

bool is_digits(const std::string& s) {
  if (s.empty()) return false;
  for (unsigned char c : s) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

std::optional<double> binding_double(const json& bindings, const std::string& key) {
  if (!bindings.is_object()) return std::nullopt;
  auto it = bindings.find(key);
  if (it == bindings.end()) return std::nullopt;
  if (it->is_number()) return it->get<double>();
  if (it->is_number_integer()) return static_cast<double>(it->get<int64_t>());
  if (it->is_string()) {
    try {
      return std::stod(it->get<std::string>());
    } catch (...) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::string dim_str(const json& tok) {
  if (tok.is_string()) return tok.get<std::string>();
  if (tok.is_number_integer()) return std::to_string(tok.get<int64_t>());
  if (tok.is_number()) return std::to_string(static_cast<int64_t>(tok.get<double>()));
  return tok.dump();
}

std::string c_ident(std::string_view name) {
  std::string out;
  out.reserve(name.size() + 1);
  for (char ch : name) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || (ch == '_');
    out.push_back(ok ? ch : '_');
  }
  if (out.empty()) return "v";
  if ((out[0] >= '0' && out[0] <= '9') || out[0] == '_') out.insert(out.begin(), 'v');
  return out;
}

std::string c_type_for_dtype(const std::string& dt) {
  if (dt == "f32") return "float";
  if (dt == "i32") return "int";
  if (dt == "i64") return "int64_t";
  if (dt == "u8") return "uint8_t";
  if (dt == "i8") return "int8_t";
  if (dt == "i16") return "int16_t";
  if (dt == "bool" || dt == "i1") return "bool";
  fail("unsupported dtype for CUDA elementwise: " + dt);
}

std::string c_scalar_literal(const std::string& dt, const json& value) {
  if (dt == "bool" || dt == "i1") {
    int64_t v = 0;
    if (value.is_boolean()) v = value.get<bool>() ? 1 : 0;
    else if (value.is_number_integer()) v = value.get<int64_t>();
    else if (value.is_number()) v = static_cast<int64_t>(value.get<double>());
    else if (value.is_string()) {
      try {
        v = static_cast<int64_t>(std::stod(value.get<std::string>()));
      } catch (...) {
        v = 0;
      }
    }
    return (v != 0) ? "true" : "false";
  }
  if (dt == "i32") {
    int64_t v = 0;
    if (value.is_number_integer()) v = value.get<int64_t>();
    else if (value.is_number()) v = static_cast<int64_t>(value.get<double>());
    else if (value.is_string()) {
      try {
        v = static_cast<int64_t>(std::stod(value.get<std::string>()));
      } catch (...) {
        v = 0;
      }
    }
    return std::to_string(static_cast<int>(v));
  }
  if (dt == "i64") {
    int64_t v = 0;
    if (value.is_number_integer()) v = value.get<int64_t>();
    else if (value.is_number()) v = static_cast<int64_t>(value.get<double>());
    else if (value.is_string()) {
      try {
        v = static_cast<int64_t>(std::stod(value.get<std::string>()));
      } catch (...) {
        v = 0;
      }
    }
    return std::to_string(static_cast<long long>(v)) + "LL";
  }
  if (dt == "f32") {
    double v = 0.0;
    if (value.is_number()) v = value.get<double>();
    else if (value.is_number_integer()) v = static_cast<double>(value.get<int64_t>());
    else if (value.is_string()) {
      try {
        v = std::stod(value.get<std::string>());
      } catch (...) {
        v = 0.0;
      }
    }
    return c_float(v);
  }
  if (dt == "f16") {
    // Minimal support; avoid f16 elementwise for now.
    double v = 0.0;
    if (value.is_number()) v = value.get<double>();
    else if (value.is_number_integer()) v = static_cast<double>(value.get<int64_t>());
    else if (value.is_string()) {
      try {
        v = std::stod(value.get<std::string>());
      } catch (...) {
        v = 0.0;
      }
    }
    return "__float2half(" + c_float(v) + ")";
  }
  fail("unsupported const dtype for CUDA elementwise: " + dt);
}

std::optional<int64_t> binding_int(const json& bindings, const std::string& key) {
  if (!bindings.is_object()) return std::nullopt;
  auto it = bindings.find(key);
  if (it == bindings.end()) return std::nullopt;
  if (it->is_number_integer()) return it->get<int64_t>();
  if (it->is_number()) return static_cast<int64_t>(it->get<double>());
  if (it->is_string()) {
    std::string s = it->get<std::string>();
    if (is_digits(s)) return std::stoll(s);
  }
  return std::nullopt;
}

std::optional<int64_t> resolve_dim_token(const json& tok, const json& bindings) {
  if (tok.is_number_integer()) return tok.get<int64_t>();
  if (tok.is_number()) return static_cast<int64_t>(tok.get<double>());
  if (tok.is_string()) {
    std::string s = tok.get<std::string>();
    if (is_digits(s)) return std::stoll(s);
    if (auto v = binding_int(bindings, s)) return v;
  }
  return std::nullopt;
}

int64_t resolve_schedule_int(const Intent& intent, const json& bindings, const char* key, int64_t default_val) {
  if (!intent.schedule.is_object()) return default_val;
  auto it = intent.schedule.find(key);
  if (it == intent.schedule.end()) return default_val;
  auto v = resolve_dim_token(*it, bindings);
  if (!v.has_value()) return default_val;
  return *v;
}

bool is_scalar_tensor(const Intent& intent, const std::string& name, const std::string& dtype) {
  auto it = intent.tensors.find(name);
  if (it == intent.tensors.end()) return false;
  if (!it->second.shape.empty()) return false;
  if (!dtype.empty() && it->second.dtype != dtype) return false;
  return true;
}

json tensor_io_spec(const Intent& intent, const std::string& name) {
  auto it = intent.tensors.find(name);
  if (it == intent.tensors.end()) fail("io_spec tensor missing from intent.tensors: " + name);
  json t;
  t["dtype"] = it->second.dtype;
  json shape = json::array();
  for (const auto& d : it->second.shape) shape.push_back(d);
  t["shape"] = shape;
  return t;
}

json io_spec_from_args(const Intent& intent, const std::vector<std::string>& tensor_args,
                       const std::unordered_map<std::string, std::string>& scalar_args,
                       const std::vector<std::string>& arg_names) {
  json tensors = json::object();
  for (const auto& n : tensor_args) tensors[n] = tensor_io_spec(intent, n);
  json scalars = json::object();
  for (const auto& kv : scalar_args) scalars[kv.first] = kv.second;
  json io_spec;
  io_spec["arg_names"] = arg_names;
  io_spec["tensors"] = tensors;
  io_spec["scalars"] = scalars;
  return io_spec;
}

json emit_dropout(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "dropout")) fail("dropout lowering expects a single dropout op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("dropout expects 3 inputs (X,p,seed)");
  const std::string X = op.inputs[0];
  const std::string p_name = op.inputs[1];
  const std::string seed_name = op.inputs[2];
  const std::string Y = op.output;

  auto n_opt = binding_int(bindings, "n_elements");
  if (!n_opt.has_value()) fail("missing binding: n_elements");
  const int64_t n = *n_opt;

  int rounds = 10;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("n_rounds");
    if (it != op.attrs.end()) {
      if (it->is_number_integer()) rounds = it->get<int>();
      else if (it->is_number()) rounds = static_cast<int>(it->get<double>());
    }
  }
  if (rounds <= 0) rounds = 10;
  if (rounds > 10) rounds = 10;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x > 1024) block_x = 1024;

  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;
  const auto tuned_block = binding_int(bindings, "DROPOUT_THREADS");
  if (tuned_block.has_value() && *tuned_block > 0 && *tuned_block <= 1024) {
    block_x = *tuned_block;
  } else if (!respect_schedule) {
    block_x = (n >= (1LL << 20)) ? 128 : 256;
  }
  // Keep block size warp-aligned.
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  if (block_x > 1024) block_x = 1024;

  int ept = 8;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("elements_per_thread");
    if (it != op.attrs.end()) {
      if (it->is_number_integer()) ept = it->get<int>();
      else if (it->is_number()) ept = static_cast<int>(it->get<double>());
    }
  }
  if (ept == 8) {
    if (auto v = binding_int(bindings, "DROPOUT_EPT")) ept = static_cast<int>(*v);
  }
  if (ept <= 0) ept = 1;
  if (ept > 8) ept = 8;

  const int64_t denom = block_x * static_cast<int64_t>(ept);
  const int64_t grid_x = (n + denom - 1) / denom;

  const bool p_is_scalar = is_scalar_tensor(intent, p_name, "f32");
  const bool seed_is_scalar = is_scalar_tensor(intent, seed_name, "i32");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/dropout.cuh\"");
  w.blank();
  if (p_is_scalar && seed_is_scalar) {
    w.line("extern \"C\" __global__ void " + intent.name +
           "(const float* X, float p, int seed, float* Y, int64_t n_elements) {");
    w.indent();
    w.line("constexpr int EPT = " + std::to_string(ept) + ";");
    w.line("constexpr int N_ROUNDS = " + std::to_string(rounds) + ";");
    w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS>(X, p, (uint32_t)seed, Y, n_elements);");
    w.dedent();
    w.line("}");
  } else {
    w.line("extern \"C\" __global__ void " + intent.name +
           "(const float* X, const float* p_ptr, const int* seed_ptr, float* Y, int64_t n_elements) {");
    w.indent();
    w.line("constexpr int EPT = " + std::to_string(ept) + ";");
    w.line("constexpr int N_ROUNDS = " + std::to_string(rounds) + ";");
    w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS>(X, p_ptr, seed_ptr, Y, n_elements);");
    w.dedent();
    w.line("}");
  }

  json io_spec;
  if (p_is_scalar && seed_is_scalar) {
    io_spec = io_spec_from_args(
        intent,
        /*tensor_args=*/{X, Y},
        /*scalar_args=*/{{p_name, "f32"}, {seed_name, "i32"}, {"n_elements", "i64"}},
        /*arg_names=*/{X, p_name, seed_name, Y, "n_elements"});
  } else {
    io_spec = io_spec_from_args(
        intent,
        /*tensor_args=*/{X, p_name, seed_name, Y},
        /*scalar_args=*/{{"n_elements", "i64"}},
        /*arg_names=*/{X, p_name, seed_name, Y, "n_elements"});
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec;
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {Y};
  json out_bindings = bindings;
  if (!out_bindings.contains("n_elements")) out_bindings["n_elements"] = n;
  out["bindings"] = out_bindings;
  return out;
}

json emit_matmul_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "matmul")) fail("matmul lowering expects a single matmul op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("matmul expects 2 inputs");
  const std::string a = op.inputs[0];
  const std::string b = op.inputs[1];
  const std::string c = op.output;

  auto a_it = intent.tensors.find(a);
  auto b_it = intent.tensors.find(b);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end()) fail("matmul missing A/B tensors in intent.tensors");
  if (a_it->second.shape.size() != 2 || b_it->second.shape.size() != 2) fail("matmul expects rank-2 inputs");

  const json M_dim = a_it->second.shape[0];
  const json K_dim = a_it->second.shape[1];
  const json K2_dim = b_it->second.shape[0];
  const json N_dim = b_it->second.shape[1];
  if (dim_str(K_dim) != dim_str(K2_dim)) fail("matmul K mismatch between A and B");

  auto M_opt = resolve_dim_token(M_dim, bindings);
  auto N_opt = resolve_dim_token(N_dim, bindings);
  auto K_opt = resolve_dim_token(K_dim, bindings);
  if (!M_opt.has_value() || !N_opt.has_value() || !K_opt.has_value()) fail("matmul missing bindings for M/N/K");
  const int64_t M = *M_opt;
  const int64_t N = *N_opt;
  const int64_t K = *K_opt;

  int64_t block_y = resolve_schedule_int(intent, bindings, "tile_m", 16);
  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 16);
  int64_t block_k = resolve_schedule_int(intent, bindings, "tile_k", 16);
  if (block_x <= 0) block_x = 16;
  if (block_y <= 0) block_y = 16;
  if (block_k <= 0) block_k = 16;
  if (block_x > 64) block_x = 64;

  bool allow_tf32 = binding_int(bindings, "ALLOW_TF32").value_or(0) != 0;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("allow_tf32");
    if (it != op.attrs.end()) {
      if (it->is_boolean()) allow_tf32 = allow_tf32 || it->get<bool>();
      else if (it->is_number_integer()) allow_tf32 = allow_tf32 || (it->get<int64_t>() != 0);
      else if (it->is_number()) allow_tf32 = allow_tf32 || (it->get<double>() != 0.0);
    }
  }
  const bool use_wmma = allow_tf32 && ((M % 16) == 0) && ((N % 16) == 0) && ((K % 8) == 0);

  int64_t thread_m = std::min<int64_t>(16, block_y);
  if (block_x * thread_m > 1024) thread_m = std::max<int64_t>(1, 1024 / std::max<int64_t>(1, block_x));
  block_k = std::max<int64_t>(1, std::min<int64_t>(block_k, std::min<int64_t>(block_x, thread_m)));
  const int64_t rows_per_thread = std::max<int64_t>(1, (block_y + thread_m - 1) / thread_m);

  const int64_t grid_x = (N + block_x - 1) / block_x;
  const int64_t grid_y = (M + block_y - 1) / block_y;

  const std::string M_name = dim_str(M_dim);
  const std::string N_name = dim_str(N_dim);
  const std::string K_name = dim_str(K_dim);
  const bool m_is_tensor = is_scalar_tensor(intent, M_name, "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, N_name, "i32");
  const bool k_is_tensor = is_scalar_tensor(intent, K_name, "i32");

  const bool specialize_dims = binding_int(bindings, "CUDA_SPECIALIZE_DIMS").value_or(0) != 0;

  const std::string m_param = m_is_tensor ? ("const int* " + M_name + "_ptr") : "int M_in";
  const std::string n_param = n_is_tensor ? ("const int* " + N_name + "_ptr") : "int N_in";
  const std::string k_param = k_is_tensor ? ("const int* " + K_name + "_ptr") : "int K_in";

  const std::string m_init = m_is_tensor ? ("(" + M_name + "_ptr ? " + M_name + "_ptr[0] : 0)") : "M_in";
  const std::string n_init = n_is_tensor ? ("(" + N_name + "_ptr ? " + N_name + "_ptr[0] : 0)") : "N_in";
  const std::string k_init = k_is_tensor ? ("(" + K_name + "_ptr ? " + K_name + "_ptr[0] : 0)") : "K_in";

  std::string mnk_load;
  if (specialize_dims) {
    const std::string m_unused = m_is_tensor ? (M_name + "_ptr") : "M_in";
    const std::string n_unused = n_is_tensor ? (N_name + "_ptr") : "N_in";
    const std::string k_unused = k_is_tensor ? (K_name + "_ptr") : "K_in";
    std::ostringstream ss;
    CodeWriter w(ss);
    w.line("(void)" + m_unused + ";");
    w.line("(void)" + n_unused + ";");
    w.line("(void)" + k_unused + ";");
    w.line("constexpr int M = " + std::to_string(M) + ";");
    w.line("constexpr int N = " + std::to_string(N) + ";");
    w.line("constexpr int K = " + std::to_string(K) + ";");
    mnk_load = ss.str();
  } else {
    if (m_is_tensor || n_is_tensor || k_is_tensor) {
      std::ostringstream ss;
      CodeWriter w(ss);
      w.line("__shared__ int intentir_M;");
      w.line("__shared__ int intentir_N;");
      w.line("__shared__ int intentir_K;");
      w.line("if ((int)threadIdx.x == 0 && (int)threadIdx.y == 0 && (int)threadIdx.z == 0) {");
      w.indent();
      w.line("intentir_M = " + m_init + ";");
      w.line("intentir_N = " + n_init + ";");
      w.line("intentir_K = " + k_init + ";");
      w.dedent();
      w.line("}");
      w.line("__syncthreads();");
      w.line("const int M = intentir_M;");
      w.line("const int N = intentir_N;");
      w.line("const int K = intentir_K;");
      mnk_load = ss.str();
    } else {
      std::ostringstream ss;
      CodeWriter w(ss);
      w.line("const int M = M_in;");
      w.line("const int N = N_in;");
      w.line("const int K = K_in;");
      mnk_load = ss.str();
    }
  }

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  json launch;
  int64_t shared_mem = 0;

  if (use_wmma) {
    int64_t wmma_warps_m = binding_int(bindings, "WMMA_WARPS_M").value_or(0);
    int64_t wmma_warps_n = binding_int(bindings, "WMMA_WARPS_N").value_or(0);
    const bool warps_m_override = wmma_warps_m > 0;
    const bool warps_n_override = wmma_warps_n > 0;
    (void)warps_n_override;
    int64_t wmma_frag_n = binding_int(bindings, "WMMA_FRAG_N").value_or(1);

    if (wmma_warps_m <= 0 || wmma_warps_n <= 0) {
      wmma_warps_m = std::max<int64_t>(1, std::min<int64_t>(4, block_y / 16));
      wmma_warps_n = std::max<int64_t>(1, std::min<int64_t>(8, block_x / 16));
      if (wmma_warps_n <= 1) {
        if (N >= 256)
          wmma_warps_n = 4;
        else if (N >= 64)
          wmma_warps_n = 2;
      }
      if ((block_y % 16) != 0) wmma_warps_m = (M >= 64) ? 4 : 2;
      if ((block_x % 16) != 0) wmma_warps_n = (N >= 32) ? 2 : 1;
    }
    wmma_warps_m = std::max<int64_t>(1, std::min<int64_t>(4, wmma_warps_m));
    wmma_warps_n = std::max<int64_t>(1, std::min<int64_t>(8, wmma_warps_n));
    if (!warps_m_override && M <= 256) wmma_warps_m = std::max<int64_t>(wmma_warps_m, 2);

    if (wmma_frag_n <= 0) wmma_frag_n = 1;
    if (!(wmma_frag_n == 1 || wmma_frag_n == 2)) wmma_frag_n = 1;
    if ((wmma_warps_n % wmma_frag_n) != 0) wmma_frag_n = 1;
    wmma_warps_n = std::max<int64_t>(1, wmma_warps_n / wmma_frag_n);

    while ((wmma_warps_m * wmma_warps_n) > 16) {
      if (wmma_warps_n > 1)
        wmma_warps_n -= 1;
      else if (wmma_warps_m > 1)
        wmma_warps_m -= 1;
      else
        break;
    }
    if (!warps_m_override && M <= 256 && wmma_warps_m > 2) wmma_warps_m = 2;

    int64_t wmma_stage_k = binding_int(bindings, "WMMA_STAGE_K").value_or(0);
    if (wmma_stage_k <= 0) wmma_stage_k = (K >= 256) ? 64 : 32;
    if (!(wmma_stage_k == 8 || wmma_stage_k == 16 || wmma_stage_k == 32 || wmma_stage_k == 64 || wmma_stage_k == 128)) wmma_stage_k = 32;
    if ((wmma_stage_k % 8) != 0 || (wmma_stage_k % 4) != 0) wmma_stage_k = 32;
    if ((K % wmma_stage_k) != 0) wmma_stage_k = 16;
    if ((K % wmma_stage_k) != 0) wmma_stage_k = 8;

    const int64_t wmma_tile_m = 16 * wmma_warps_m;
    const int64_t wmma_tile_n = 16 * wmma_warps_n * wmma_frag_n;
    const int64_t wmma_grid_x = (N + wmma_tile_n - 1) / wmma_tile_n;
    const int64_t wmma_grid_y = (M + wmma_tile_m - 1) / wmma_tile_m;

    auto policy_norm = [&](const char* key) -> std::string {
      if (!bindings.is_object()) return "";
      auto it = bindings.find(key);
      if (it == bindings.end()) return "";
      if (!it->is_string()) return "";
      std::string s = ascii_lower(it->get<std::string>());
      if (s == "ca" || s == "cg") return s;
      return "";
    };
    std::string wmma_cp_a_policy = policy_norm("WMMA_CP_A_POLICY");
    std::string wmma_cp_b_policy = policy_norm("WMMA_CP_B_POLICY");
    if (wmma_cp_a_policy.empty()) wmma_cp_a_policy = (wmma_warps_n >= wmma_warps_m) ? "ca" : "cg";
    if (wmma_cp_b_policy.empty()) wmma_cp_b_policy = (wmma_warps_m > wmma_warps_n) ? "ca" : "cg";

    int64_t use_cp_async_raw = -1;
    if (auto v = binding_int(bindings, "WMMA_USE_CP_ASYNC")) use_cp_async_raw = *v;
    const bool wmma_use_cp_async = (use_cp_async_raw < 0) ? true : (use_cp_async_raw != 0);

    int64_t wmma_pipe_stages = binding_int(bindings, "WMMA_PIPE_STAGES").value_or(0);
    if (!wmma_use_cp_async) {
      wmma_pipe_stages = 1;
    } else {
      if (wmma_pipe_stages <= 0) wmma_pipe_stages = 3;
      if (!(wmma_pipe_stages == 2 || wmma_pipe_stages == 3)) wmma_pipe_stages = 3;
    }

    int64_t wmma_as_pad = binding_int(bindings, "WMMA_AS_PAD").value_or(8);
    int64_t wmma_bs_pad = binding_int(bindings, "WMMA_BS_PAD").value_or(8);
    if (wmma_as_pad < 0) wmma_as_pad = 0;
    if (wmma_bs_pad < 0) wmma_bs_pad = 0;
    if (wmma_as_pad > 32) wmma_as_pad = 32;
    if (wmma_bs_pad > 32) wmma_bs_pad = 32;
    if ((wmma_as_pad % 4) != 0) wmma_as_pad = (wmma_as_pad / 4) * 4;
    if ((wmma_bs_pad % 4) != 0) wmma_bs_pad = (wmma_bs_pad / 4) * 4;

    int64_t max_smem_optin = binding_int(bindings, "CUDA_MAX_SMEM_OPTIN").value_or(0);
    if (max_smem_optin <= 0) max_smem_optin = 96 * 1024;

    auto wmma_smem_bytes = [&](int64_t stage_k, int64_t pipe_stages) -> int64_t {
      const int64_t as_ld = stage_k + wmma_as_pad;
      const int64_t bs_ld = wmma_tile_n + wmma_bs_pad;
      return 4LL * (pipe_stages * wmma_tile_m * as_ld + pipe_stages * stage_k * bs_ld);
    };

    int64_t shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
    if (wmma_pipe_stages > 2 && shared_bytes > max_smem_optin) {
      wmma_pipe_stages = 2;
      shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
    }
    if (shared_bytes > max_smem_optin) {
      for (int64_t cand : {int64_t(64), int64_t(32), int64_t(16), int64_t(8)}) {
        if (cand >= wmma_stage_k) continue;
        if ((K % cand) != 0) continue;
        if (wmma_smem_bytes(cand, wmma_pipe_stages) <= max_smem_optin) {
          wmma_stage_k = cand;
          shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
          break;
        }
      }
    }

    bool wmma_force_sync = false;
    if (wmma_pipe_stages == 2) {
      const int64_t bytes3 = wmma_smem_bytes(wmma_stage_k, 3);
      if (bytes3 <= max_smem_optin) {
        wmma_pipe_stages = 3;
        shared_bytes = bytes3;
      } else {
        wmma_force_sync = true;
        wmma_pipe_stages = 1;
        shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
      }
    }

    shared_bytes = ((shared_bytes + 15) / 16) * 16;
    const bool wmma_disable_fastpath = wmma_force_sync || (binding_int(bindings, "WMMA_DISABLE_FASTPATH").value_or(0) != 0);
    const bool specialize_full_tile = specialize_dims && ((M % wmma_tile_m) == 0) && ((N % wmma_tile_n) == 0) && ((K % wmma_stage_k) == 0) &&
                                      ((K & 3) == 0) && ((N & 3) == 0) && (!wmma_disable_fastpath);

    const std::string wmma_cp_a_enum = (wmma_cp_a_policy == "ca") ? "intentir_cuda::CpAsyncPolicy::CA" : "intentir_cuda::CpAsyncPolicy::CG";
    const std::string wmma_cp_b_enum = (wmma_cp_b_policy == "ca") ? "intentir_cuda::CpAsyncPolicy::CA" : "intentir_cuda::CpAsyncPolicy::CG";
    const std::string use_cp_async_const = wmma_use_cp_async ? "true" : "false";
    const std::string enable_fastpath_const = wmma_disable_fastpath ? "false" : "true";
    const std::string specialize_full_tile_const = specialize_full_tile ? "true" : "false";

    w.line("#include \"kernels/wmma_matmul.cuh\"");
    w.blank();
    w.line("extern \"C\" __global__ void " + intent.name + "(");
    w.indent();
    w.line("const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,");
    w.line(m_param + ", " + n_param + ", " + k_param + ") {");
    w.dedent();
    w.indent();
    {
      std::istringstream mnk_ss(mnk_load);
      for (std::string line; std::getline(mnk_ss, line);) {
        if (!line.empty()) w.line(line);
      }
    }
    w.blank();
    w.line("constexpr int WARPS_M = " + std::to_string(wmma_warps_m) + ";");
    w.line("constexpr int WARPS_N = " + std::to_string(wmma_warps_n) + ";");
    w.line("constexpr int FRAG_N = " + std::to_string(wmma_frag_n) + ";");
    w.line("constexpr int STAGE_K = " + std::to_string(wmma_stage_k) + ";");
    w.line("constexpr int AS_PAD = " + std::to_string(wmma_as_pad) + ";");
    w.line("constexpr int BS_PAD = " + std::to_string(wmma_bs_pad) + ";");
    w.line("constexpr int PIPE_STAGES = " + std::to_string(wmma_pipe_stages) + ";");
    w.blank();
    w.line("intentir_cuda::wmma_matmul_f32_tf32<");
    w.indent();
    w.line("WARPS_M,");
    w.line("WARPS_N,");
    w.line("FRAG_N,");
    w.line("STAGE_K,");
    w.line("AS_PAD,");
    w.line("BS_PAD,");
    w.line("PIPE_STAGES,");
    w.line(use_cp_async_const + ",");
    w.line(wmma_cp_a_enum + ",");
    w.line(wmma_cp_b_enum + ",");
    w.line(enable_fastpath_const + ",");
    w.line(specialize_full_tile_const + ">(A, B, C, M, N, K);");
    w.dedent();
    w.dedent();
    w.line("}");

    launch = {{"grid", {wmma_grid_x, wmma_grid_y, 1}},
              {"block", {32 * wmma_warps_m * wmma_warps_n, 1, 1}},
              {"shared_mem", shared_bytes}};
    shared_mem = shared_bytes;
  } else {
    w.line("#include \"kernels/matmul_fallback.cuh\"");
    w.blank();
    w.line("extern \"C\" __global__ void " + intent.name + "(");
    w.indent();
    w.line("const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,");
    w.line(m_param + ", " + n_param + ", " + k_param + ") {");
    w.dedent();
    w.indent();
    {
      std::istringstream mnk_ss(mnk_load);
      for (std::string line; std::getline(mnk_ss, line);) {
        if (!line.empty()) w.line(line);
      }
    }
    w.line("constexpr int BLOCK_M = " + std::to_string(block_y) + ";");
    w.line("constexpr int BLOCK_N = " + std::to_string(block_x) + ";");
    w.line("constexpr int BLOCK_K = " + std::to_string(block_k) + ";");
    w.line("constexpr int THREAD_M = " + std::to_string(thread_m) + ";");
    w.line("constexpr int ROWS_PER_THREAD = " + std::to_string(rows_per_thread) + ";");
    w.line("__shared__ float As[BLOCK_M * BLOCK_K];");
    w.line("__shared__ float Bs[BLOCK_K * BLOCK_N];");
    w.line("intentir_cuda::matmul_f32_fallback<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, ROWS_PER_THREAD>(A, B, C, M, N, K, As, Bs);");
    w.dedent();
    w.line("}");

    launch = {{"grid", {grid_x, grid_y, 1}}, {"block", {block_x, thread_m, 1}}, {"shared_mem", 0}};
    shared_mem = 0;
  }

  std::vector<std::string> tensor_args = {a, b, c};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {a, b, c};
  if (m_is_tensor) {
    tensor_args.push_back(M_name);
    arg_names.push_back(M_name);
  } else {
    scalar_args.emplace(M_name, "i32");
    arg_names.push_back(M_name);
  }
  if (n_is_tensor) {
    tensor_args.push_back(N_name);
    arg_names.push_back(N_name);
  } else {
    scalar_args.emplace(N_name, "i32");
    arg_names.push_back(N_name);
  }
  if (k_is_tensor) {
    tensor_args.push_back(K_name);
    arg_names.push_back(K_name);
  } else {
    scalar_args.emplace(K_name, "i32");
    arg_names.push_back(K_name);
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = launch;
  out["output_names"] = {c};
  out["bindings"] = bindings;
  (void)shared_mem;
  return out;
}

json emit_warp(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "warp")) fail("warp lowering expects a single warp op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("warp expects 2 inputs (src, offset)");
  const std::string src_name = op.inputs[0];
  const std::string offset_name = op.inputs[1];
  const std::string out_name = op.output;

  const int64_t C = binding_int(bindings, "C").value_or(-1);
  const int64_t H = binding_int(bindings, "H").value_or(-1);
  const int64_t W = binding_int(bindings, "W").value_or(-1);
  if (C <= 0 || H <= 0 || W <= 0) fail("warp missing/invalid bindings: C/H/W");

  int64_t block_w = binding_int(bindings, "BLOCK_W").value_or(128);
  const int64_t hinted = resolve_schedule_int(intent, bindings, "tile_n", block_w);
  if (0 < hinted && hinted <= 1024) block_w = hinted;
  if (block_w <= 0) block_w = 128;
  if (block_w > 1024) block_w = 1024;
  const int64_t grid_w = (W + block_w - 1) / block_w;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/warp.cuh\"");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name +
         "(const int8_t* __restrict__ " + src_name + ", const int16_t* __restrict__ " + offset_name + ", int8_t* __restrict__ " + out_name +
         ", int C, int H, int W) {");
  w.indent();
  w.line("constexpr int BLOCK_W = " + std::to_string(block_w) + ";");
  w.line("intentir_cuda::warp_q8_8_i8_i16<BLOCK_W>(" + src_name + ", " + offset_name + ", " + out_name + ", C, H, W);");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{src_name, offset_name, out_name},
      /*scalar_args=*/{{"C", "i32"}, {"H", "i32"}, {"W", "i32"}},
      /*arg_names=*/{src_name, offset_name, out_name, "C", "H", "W"});
  out["launch"] = {{"grid", {grid_w, H, C}}, {"block", {block_w, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_correlation(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "correlation")) fail("correlation lowering expects a single correlation op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("correlation expects 3 inputs (src0, src1, out_shift)");
  const std::string src0_name = op.inputs[0];
  const std::string src1_name = op.inputs[1];
  const std::string out_name = op.output;

  const int64_t OC = binding_int(bindings, "out_channel").value_or(-1);
  const int64_t IC = binding_int(bindings, "in_channel").value_or(-1);
  const int64_t H = binding_int(bindings, "height").value_or(-1);
  const int64_t W = binding_int(bindings, "width").value_or(-1);
  if (OC <= 0 || IC <= 0 || H <= 0 || W <= 0) fail("correlation missing/invalid bindings: out_channel/in_channel/height/width");
  const int64_t total = OC * H * W;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  const bool oc_is_tensor = is_scalar_tensor(intent, "out_channel", "i32");
  const bool ic_is_tensor = is_scalar_tensor(intent, "in_channel", "i32");
  const bool h_is_tensor = is_scalar_tensor(intent, "height", "i32");
  const bool w_is_tensor = is_scalar_tensor(intent, "width", "i32");
  const bool sh_is_tensor = is_scalar_tensor(intent, "out_shift", "i32");

  const std::string oc_param = oc_is_tensor ? "const int* out_channel_ptr" : "int out_channel";
  const std::string ic_param = ic_is_tensor ? "const int* in_channel_ptr" : "int in_channel";
  const std::string h_param = h_is_tensor ? "const int* height_ptr" : "int height";
  const std::string w_param = w_is_tensor ? "const int* width_ptr" : "int width";
  const std::string sh_param = sh_is_tensor ? "const int* out_shift_ptr" : "int out_shift";

  const std::string oc_load = oc_is_tensor ? "const int out_channel = out_channel_ptr ? out_channel_ptr[0] : 0;" : "";
  const std::string ic_load = ic_is_tensor ? "const int in_channel = in_channel_ptr ? in_channel_ptr[0] : 0;" : "";
  const std::string h_load = h_is_tensor ? "const int height = height_ptr ? height_ptr[0] : 0;" : "";
  const std::string w_load = w_is_tensor ? "const int width = width_ptr ? width_ptr[0] : 0;" : "";
  const std::string sh_load = sh_is_tensor ? "const int out_shift = out_shift_ptr ? out_shift_ptr[0] : 0;" : "";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/correlation.cuh\"");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const int8_t* __restrict__ " + src0_name + ",");
  w.line("const int8_t* __restrict__ " + src1_name + ",");
  w.line("int8_t* __restrict__ " + out_name + ",");
  w.line(oc_param + ", " + ic_param + ", " + h_param + ", " + w_param + ", " + sh_param + ") {");
  w.dedent();
  w.indent();
  if (!oc_load.empty()) w.line(oc_load);
  if (!ic_load.empty()) w.line(ic_load);
  if (!h_load.empty()) w.line(h_load);
  if (!w_load.empty()) w.line(w_load);
  if (!sh_load.empty()) w.line(sh_load);
  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
  w.line("intentir_cuda::correlation_i8<BLOCK_THREADS>(" + src0_name + ", " + src1_name + ", " + out_name +
         ", out_channel, in_channel, height, width, out_shift);");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {src0_name, src1_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {src0_name, src1_name, out_name};
  for (const auto& dim_name : {"out_channel", "in_channel", "height", "width", "out_shift"}) {
    if (is_scalar_tensor(intent, dim_name, "i32")) {
      tensor_args.push_back(dim_name);
      arg_names.push_back(dim_name);
    } else {
      scalar_args.emplace(dim_name, "i32");
      arg_names.push_back(dim_name);
    }
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_resize_bilinear2x_i8(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "resize")) fail("resize lowering expects a single resize op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("resize expects 1 input");
  const std::string src_name = op.inputs[0];
  const std::string out_name = op.output;

  const int64_t C = binding_int(bindings, "C").value_or(-1);
  const int64_t H = binding_int(bindings, "H").value_or(-1);
  const int64_t W = binding_int(bindings, "W").value_or(-1);
  if (C <= 0 || H <= 0 || W <= 0) fail("resize missing/invalid bindings: C/H/W");
  const int64_t OH = binding_int(bindings, "OH").value_or(2 * H);
  const int64_t OW = binding_int(bindings, "OW").value_or(2 * W);
  if (OH != 2 * H || OW != 2 * W) fail("resize MVP supports only 2x upsample");

  int hw_fl = 7;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("hw_fl");
    if (it != op.attrs.end()) {
      if (it->is_number_integer()) hw_fl = it->get<int>();
      else if (it->is_number()) hw_fl = static_cast<int>(it->get<double>());
    }
  }
  if (hw_fl != 7) fail("resize MVP expects hw_fl=7");

  int64_t block_w = binding_int(bindings, "BLOCK_W").value_or(128);
  const int64_t hinted = resolve_schedule_int(intent, bindings, "tile_n", block_w);
  if (0 < hinted && hinted <= 1024) block_w = hinted;
  if (block_w <= 0) block_w = 128;
  if (block_w > 1024) block_w = 1024;
  const int64_t grid_w = (W + block_w - 1) / block_w;

  const bool c_is_tensor = is_scalar_tensor(intent, "C", "i32");
  const bool h_is_tensor = is_scalar_tensor(intent, "H", "i32");
  const bool w_is_tensor = is_scalar_tensor(intent, "W", "i32");

  const std::string c_param = c_is_tensor ? "const int* C_ptr" : "int C";
  const std::string h_param = h_is_tensor ? "const int* H_ptr" : "int H";
  const std::string w_param = w_is_tensor ? "const int* W_ptr" : "int W";
  const std::string c_load = c_is_tensor ? "const int C = C_ptr ? C_ptr[0] : 0;" : "";
  const std::string h_load = h_is_tensor ? "const int H = H_ptr ? H_ptr[0] : 0;" : "";
  const std::string w_load = w_is_tensor ? "const int W = W_ptr ? W_ptr[0] : 0;" : "";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/resize.cuh\"");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(const int8_t* __restrict__ " + src_name + ", int8_t* __restrict__ " + out_name +
         ", " + c_param + ", " + h_param + ", " + w_param + ") {");
  w.indent();
  if (!c_load.empty()) w.line(c_load);
  if (!h_load.empty()) w.line(h_load);
  if (!w_load.empty()) w.line(w_load);
  w.line("constexpr int BLOCK_W = " + std::to_string(block_w) + ";");
  w.line("intentir_cuda::resize_bilinear2x_i8<BLOCK_W>(" + src_name + ", " + out_name + ", C, H, W);");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {src_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {src_name, out_name};
  for (const auto& dim_name : {"C", "H", "W"}) {
    if (is_scalar_tensor(intent, dim_name, "i32")) {
      tensor_args.push_back(dim_name);
      arg_names.push_back(dim_name);
    } else {
      scalar_args.emplace(dim_name, "i32");
      arg_names.push_back(dim_name);
    }
  }

  json out_bindings = bindings;
  if (!out_bindings.contains("OH")) out_bindings["OH"] = OH;
  if (!out_bindings.contains("OW")) out_bindings["OW"] = OW;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_w, OH, C}}, {"block", {block_w, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_rope_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "rope")) fail("rope lowering expects a single rope op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("rope expects 3 inputs (input, cos, sin)");
  const std::string in_name = op.inputs[0];
  const std::string cos_name = op.inputs[1];
  const std::string sin_name = op.inputs[2];
  const std::string out_name = op.output;

  const int64_t SEQ = binding_int(bindings, "SEQ_LEN").value_or(-1);
  const int64_t B = binding_int(bindings, "BATCH_NUM").value_or(-1);
  const int64_t H = binding_int(bindings, "HEAD_NUM").value_or(-1);
  const int64_t D = binding_int(bindings, "HEAD_DIM").value_or(-1);
  if (SEQ <= 0 || B <= 0 || H <= 0 || D <= 0) fail("rope missing/invalid bindings: SEQ_LEN/BATCH_NUM/HEAD_NUM/HEAD_DIM");
  if ((D & 1) != 0) fail("rope expects even HEAD_DIM");
  const int64_t half = D / 2;

  int64_t heads_per_block = binding_int(bindings, "ROPE_HEADS_PER_BLOCK").value_or(1);
  if (heads_per_block <= 0) heads_per_block = 1;
  if (heads_per_block > 16) heads_per_block = 16;
  if (heads_per_block > H) heads_per_block = H;

  int64_t rope_vec = binding_int(bindings, "ROPE_VEC").value_or(4);
  if (!(rope_vec == 1 || rope_vec == 2 || rope_vec == 4)) rope_vec = 4;
  if (rope_vec == 4 && (half & 3) != 0) rope_vec = ((half & 1) == 0) ? 2 : 1;
  if (rope_vec == 2 && (half & 1) != 0) rope_vec = 1;

  int64_t packs = half;
  if (rope_vec == 4) packs = half / 4;
  else if (rope_vec == 2) packs = half / 2;

  auto next_pow2 = [](int64_t x) -> int64_t {
    if (x <= 1) return 1;
    int64_t p = 1;
    while (p < x) p <<= 1;
    return p;
  };

  int64_t block_x = binding_int(bindings, "ROPE_THREADS").value_or(0);
  if (block_x <= 0) {
    const int64_t target = std::max<int64_t>(1, packs / 4);
    block_x = next_pow2(target);
    if (block_x < 32) block_x = 32;
    if (block_x > 256) block_x = 256;
  }
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  if (block_x > 1024) block_x = 1024;

  const int64_t total_elems = SEQ * B * H * D;
  const bool use_i32 = total_elems <= ((1LL << 31) - 1);
  const std::string idx_t = use_i32 ? "int" : "size_t";

  const bool seq_is_tensor = is_scalar_tensor(intent, "SEQ_LEN", "i32");
  const bool b_is_tensor = is_scalar_tensor(intent, "BATCH_NUM", "i32");
  const bool h_is_tensor = is_scalar_tensor(intent, "HEAD_NUM", "i32");
  const bool d_is_tensor = is_scalar_tensor(intent, "HEAD_DIM", "i32");

  auto dim_param = [&](const char* name, bool is_tensor) -> std::string {
    return is_tensor ? std::string("const int* ") + name + "_ptr" : std::string("int ") + name;
  };
  auto dim_load = [&](const char* name, bool is_tensor) -> std::string {
    if (!is_tensor) return "";
    return std::string("const int ") + name + " = " + name + "_ptr ? " + name + "_ptr[0] : 0;";
  };

  const int64_t iters = std::max<int64_t>(1, (packs + block_x - 1) / block_x);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/rope.cuh\"");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + in_name + ", const float* __restrict__ " + cos_name + ", const float* __restrict__ " + sin_name +
         ", float* __restrict__ " + out_name + ",");
  w.line(dim_param("SEQ_LEN", seq_is_tensor) + ", " + dim_param("BATCH_NUM", b_is_tensor) + ", " + dim_param("HEAD_NUM", h_is_tensor) + ", " +
         dim_param("HEAD_DIM", d_is_tensor) + ") {");
  w.dedent();
  w.indent();
  const std::string seq_load = dim_load("SEQ_LEN", seq_is_tensor);
  const std::string b_load = dim_load("BATCH_NUM", b_is_tensor);
  const std::string h_load = dim_load("HEAD_NUM", h_is_tensor);
  const std::string d_load = dim_load("HEAD_DIM", d_is_tensor);
  if (!seq_load.empty()) w.line(seq_load);
  if (!b_load.empty()) w.line(b_load);
  if (!h_load.empty()) w.line(h_load);
  if (!d_load.empty()) w.line(d_load);
  w.line("constexpr int HEADS_PER_BLOCK = " + std::to_string(heads_per_block) + ";");
  w.line("constexpr int ROPE_VEC = " + std::to_string(rope_vec) + ";");
  w.line("constexpr int BLOCK_X = " + std::to_string(block_x) + ";");
  w.line("constexpr int ITERS = " + std::to_string(iters) + ";");
  w.line("using idx_t = " + idx_t + ";");
  w.line("intentir_cuda::rope_f32<HEADS_PER_BLOCK, ROPE_VEC, BLOCK_X, ITERS, idx_t>(" + in_name + ", " + cos_name + ", " + sin_name + ", " +
         out_name + ", SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM);");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {in_name, cos_name, sin_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {in_name, cos_name, sin_name, out_name};
  for (const auto& dim_name : {"SEQ_LEN", "BATCH_NUM", "HEAD_NUM", "HEAD_DIM"}) {
    if (is_scalar_tensor(intent, dim_name, "i32")) {
      tensor_args.push_back(dim_name);
      arg_names.push_back(dim_name);
    } else {
      scalar_args.emplace(dim_name, "i32");
      arg_names.push_back(dim_name);
    }
  }

  json out_bindings = bindings;
  try {
    auto it = intent.tensors.find(cos_name);
    if (it != intent.tensors.end() && it->second.shape.size() >= 2 && it->second.shape[1].is_string()) {
      const std::string sym = it->second.shape[1].get<std::string>();
      if (!sym.empty()) out_bindings.emplace(sym, half);
    } else {
      out_bindings.emplace("HEAD_DIM_DIV_2", half);
    }
  } catch (...) {
    out_bindings.emplace("HEAD_DIM_DIV_2", half);
  }

  const int64_t grid_x = (H + heads_per_block - 1) / heads_per_block;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_x, B, SEQ}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_transpose_2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "transpose")) fail("transpose lowering expects a single transpose op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("transpose expects 1 input");
  const std::string inp_name = op.inputs[0];
  const std::string out_name = op.output;

  json perm = json();
  if (op.attrs.is_object()) perm = op.attrs.value("perm", json());
  bool ok_perm = false;
  if (perm.is_array() && perm.size() == 2) {
    try {
      ok_perm = (perm[0].get<int>() == 1 && perm[1].get<int>() == 0);
    } catch (...) {
      ok_perm = false;
    }
  }
  if (!ok_perm) fail("transpose MVP supports only perm=[1,0]");

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("transpose missing/invalid bindings: M/N");

  const int64_t tile_n = resolve_schedule_int(intent, bindings, "tile_n", 16);
  const int64_t tile_m = resolve_schedule_int(intent, bindings, "tile_m", tile_n);
  int64_t tile = std::min<int64_t>(tile_m, tile_n);
  if (tile < 8) tile = 8;
  if (tile > 32) tile = 32;

  const int64_t grid_x = (N + tile - 1) / tile;
  const int64_t grid_y = (M + tile - 1) / tile;

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + inp_name + ", float* __restrict__ " + out_name + ", " +
         m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("__shared__ float tile[" + std::to_string(tile) + "][" + std::to_string(tile) + " + 1];");
  w.line("const int x = (int)blockIdx.x * " + std::to_string(tile) + " + (int)threadIdx.x;");
  w.line("const int y = (int)blockIdx.y * " + std::to_string(tile) + " + (int)threadIdx.y;");
  w.line("if (x < N && y < M) {");
  w.indent();
  w.line("tile[(int)threadIdx.y][(int)threadIdx.x] = " + inp_name + "[(size_t)y * (size_t)N + (size_t)x];");
  w.dedent();
  w.line("}");
  w.line("__syncthreads();");
  w.line("const int ox = (int)blockIdx.y * " + std::to_string(tile) + " + (int)threadIdx.x;");
  w.line("const int oy = (int)blockIdx.x * " + std::to_string(tile) + " + (int)threadIdx.y;");
  w.line("if (ox < M && oy < N) {");
  w.indent();
  w.line(out_name + "[(size_t)oy * (size_t)M + (size_t)ox] = tile[(int)threadIdx.x][(int)threadIdx.y];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  for (const auto& dim_name : {"M", "N"}) {
    if (is_scalar_tensor(intent, dim_name, "i32")) {
      tensor_args.push_back(dim_name);
      arg_names.push_back(dim_name);
    } else {
      scalar_args.emplace(dim_name, "i32");
      arg_names.push_back(dim_name);
    }
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_x, grid_y, 1}}, {"block", {tile, tile, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

bool _reduce_axis_is_1(const json& attrs) {
  if (!attrs.is_object()) return false;
  auto dims = attrs.value("dims", json());
  if (dims.is_array() && dims.size() == 1) {
    try {
      if (dims[0].get<int>() == 1) return true;
    } catch (...) {
    }
  }
  auto axis = attrs.value("axis", json());
  if (axis.is_number_integer()) return axis.get<int64_t>() == 1;
  if (axis.is_number()) return static_cast<int64_t>(axis.get<double>()) == 1;
  if (axis.is_string()) return axis.get<std::string>() == "1";
  if (axis.is_array() && axis.size() == 1) {
    try {
      if (axis[0].get<int>() == 1) return true;
    } catch (...) {
    }
  }
  return false;
}

int64_t _pow2_block(int64_t x) {
  if (x <= 0) return 256;
  if (x > 1024) x = 1024;
  auto is_pow2 = [](int64_t v) -> bool { return v > 0 && ((v & (v - 1)) == 0); };
  auto next_pow2 = [](int64_t v) -> int64_t {
    if (v <= 1) return 1;
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
  };
  if (!is_pow2(x)) x = next_pow2(x);
  if (x < 32) x = 32;
  if (x > 1024) x = 1024;
  return x;
}

json emit_reduce_sum_2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "reduce_sum")) fail("reduce_sum lowering expects a single reduce_sum op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("reduce_sum expects 1 input");
  if (!_reduce_axis_is_1(op.attrs)) fail("reduce_sum MVP supports only axis=1 for 2D tensors");
  const std::string inp_name = op.inputs[0];
  const std::string out_name = op.output;

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("reduce_sum missing/invalid bindings: M/N");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  block_x = _pow2_block(block_x);

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + inp_name + ", float* __restrict__ " + out_name + ", " +
         m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("__shared__ float smem[" + std::to_string(block_x) + "];");
  w.line("float acc = 0.0f;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc += row[n];");
  w.line("smem[(int)threadIdx.x] = acc;");
  w.line("__syncthreads();");
  w.line("for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {");
  w.indent();
  w.line("if ((int)threadIdx.x < off) smem[(int)threadIdx.x] += smem[(int)threadIdx.x + off];");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[m] = smem[0];");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  for (const auto& dim_name : {"M", "N"}) {
    if (is_scalar_tensor(intent, dim_name, "i32")) {
      tensor_args.push_back(dim_name);
      arg_names.push_back(dim_name);
    } else {
      scalar_args.emplace(dim_name, "i32");
      arg_names.push_back(dim_name);
    }
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {M, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_reduce_max_2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "reduce_max")) fail("reduce_max lowering expects a single reduce_max op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("reduce_max expects 1 input");
  if (!_reduce_axis_is_1(op.attrs)) fail("reduce_max MVP supports only axis=1 for 2D tensors");
  const std::string inp_name = op.inputs[0];
  const std::string out_name = op.output;

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("reduce_max missing/invalid bindings: M/N");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  block_x = _pow2_block(block_x);

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + inp_name + ", float* __restrict__ " + out_name + ", " +
         m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("__shared__ float smem[" + std::to_string(block_x) + "];");
  w.line("float acc = -INFINITY;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc = fmaxf(acc, row[n]);");
  w.line("smem[(int)threadIdx.x] = acc;");
  w.line("__syncthreads();");
  w.line("for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {");
  w.indent();
  w.line("if ((int)threadIdx.x < off) smem[(int)threadIdx.x] = fmaxf(smem[(int)threadIdx.x], smem[(int)threadIdx.x + off]);");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[m] = smem[0];");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  for (const auto& dim_name : {"M", "N"}) {
    if (is_scalar_tensor(intent, dim_name, "i32")) {
      tensor_args.push_back(dim_name);
      arg_names.push_back(dim_name);
    } else {
      scalar_args.emplace(dim_name, "i32");
      arg_names.push_back(dim_name);
    }
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {M, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_any_dim_f32_to_i1(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 3) fail("any-dim lowering expects 3 ops (const, ne, reduce_any)");
  const Op& c0 = intent.ops[0];
  const Op& ne0 = intent.ops[1];
  const Op& r0 = intent.ops[2];
  if (c0.op != "const" || ne0.op != "ne" || r0.op != "reduce_any") fail("any-dim lowering expects ops const->ne->reduce_any");
  if (ne0.inputs.size() != 2 || r0.inputs.size() != 1) fail("any-dim lowering invalid op arity");
  const std::string const_out = c0.output;
  const std::string a0 = ne0.inputs[0];
  const std::string b0 = ne0.inputs[1];
  std::string inp_name;
  if (a0 == const_out && b0 != const_out)
    inp_name = b0;
  else if (b0 == const_out && a0 != const_out)
    inp_name = a0;
  else
    fail("any-dim lowering expects ne(inp, const)");
  const std::string out_name = r0.output;

  double z = 0.0;
  if (c0.attrs.is_object()) {
    auto it = c0.attrs.find("value");
    if (it != c0.attrs.end() && it->is_number()) z = it->get<double>();
    else if (it != c0.attrs.end() && it->is_string()) {
      try {
        z = std::stod(it->get<std::string>());
      } catch (...) {
        z = 0.0;
      }
    }
  }
  if (!_reduce_axis_is_1(r0.attrs)) fail("any-dim MVP supports only axis=1 for 2D tensors");

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("any-dim missing/invalid bindings: M/N");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  block_x = _pow2_block(block_x);

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";

  const std::string z_lit = c_float(z);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + inp_name + ", bool* __restrict__ " + out_name + ", " +
         m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("__shared__ int smem[" + std::to_string(block_x) + "];");
  w.line("int anyv = 0;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) anyv |= (row[n] != " + z_lit + ");");
  w.line("smem[(int)threadIdx.x] = anyv;");
  w.line("__syncthreads();");
  w.line("for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {");
  w.indent();
  w.line("if ((int)threadIdx.x < off) smem[(int)threadIdx.x] |= smem[(int)threadIdx.x + off];");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[m] = (smem[0] != 0);");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  for (const auto& dim_name : {"M", "N"}) {
    if (is_scalar_tensor(intent, dim_name, "i32")) {
      tensor_args.push_back(dim_name);
      arg_names.push_back(dim_name);
    } else {
      scalar_args.emplace(dim_name, "i32");
      arg_names.push_back(dim_name);
    }
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {M, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_gather2d_f32(const Intent& intent, const json& bindings) {
  // gather2d lowering expects exactly one gather op (single-op fast path).
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "gather")) fail("gather2d lowering expects a single gather op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("gather expects inputs (inp, row_idx, col_idx)");
  const std::string inp_name = op.inputs[0];
  const std::string row_name = op.inputs[1];
  const std::string col_name = op.inputs[2];
  const std::string out_name = op.output;

  double other = 0.0;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("other_value");
    if (it != op.attrs.end() && it->is_number()) other = it->get<double>();
    else if (it != op.attrs.end() && it->is_string()) {
      try {
        other = std::stod(it->get<std::string>());
      } catch (...) {
        other = 0.0;
      }
    }
  }

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  const int64_t L = binding_int(bindings, "L").value_or(-1);
  if (M <= 0 || N <= 0 || L <= 0) fail("gather missing/invalid bindings: M/N/L");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x > 1024) block_x = 1024;
  const int64_t grid_x = (L + block_x - 1) / block_x;

  const std::string other_lit = c_float(other);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("const int* __restrict__ " + row_name + ",");
  w.line("const int* __restrict__ " + col_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line("int M, int N, int L) {");
  w.dedent();
  w.indent();
  w.line("const int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;");
  w.line("if (tid >= L) return;");
  w.line("const int r = " + row_name + "[tid];");
  w.line("const int c = " + col_name + "[tid];");
  w.line("float v = " + other_lit + ";");
  w.line("if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)N) {");
  w.indent();
  w.line("v = " + inp_name + "[(size_t)r * (size_t)N + (size_t)c];");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = v;");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{inp_name, row_name, col_name, out_name},
      /*scalar_args=*/{{"M", "i32"}, {"N", "i32"}, {"L", "i32"}},
      /*arg_names=*/{inp_name, row_name, col_name, out_name, "M", "N", "L"});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

std::string emit_broadcast_index_expr(int out_rank, const std::vector<json>& in_shape, const std::vector<std::string>& out_idxs,
                                      const std::unordered_map<std::string, std::string>& dim_expr) {
  const int in_rank = static_cast<int>(in_shape.size());
  if (in_rank == 0) return "0";
  if (in_rank > out_rank) fail("broadcast: in_rank > out_rank");
  const int shift = out_rank - in_rank;

  std::vector<std::string> in_sizes;
  in_sizes.reserve(in_rank);
  for (const auto& d : in_shape) {
    if (d.is_number_integer()) {
      in_sizes.push_back(std::to_string(d.get<int64_t>()));
    } else if (d.is_number()) {
      in_sizes.push_back(std::to_string(static_cast<int64_t>(d.get<double>())));
    } else if (d.is_string()) {
      const std::string sym = d.get<std::string>();
      auto it = dim_expr.find(sym);
      if (it == dim_expr.end()) fail("broadcast missing dim binding: " + sym);
      in_sizes.push_back(it->second);
    } else {
      fail("broadcast invalid dim token");
    }
  }

  std::vector<std::string> strides;
  strides.reserve(in_rank);
  for (int j = 0; j < in_rank; ++j) {
    if (j == in_rank - 1) {
      strides.push_back("1");
    } else {
      std::string expr;
      for (int k = j + 1; k < in_rank; ++k) {
        if (!expr.empty()) expr += " * ";
        expr += "(int64_t)" + in_sizes[k];
      }
      strides.push_back(expr.empty() ? "1" : expr);
    }
  }

  std::vector<std::string> terms;
  for (int j = 0; j < in_rank; ++j) {
    const int out_k = j + shift;
    const std::string& idx = out_idxs[out_k];
    const std::string& size_expr = in_sizes[j];
    if (size_expr == "1") continue;
    terms.push_back("((int64_t)" + idx + ") * (" + strides[j] + ")");
  }
  if (terms.empty()) return "0";
  std::string out;
  for (size_t i = 0; i < terms.size(); ++i) {
    if (i) out += " + ";
    out += terms[i];
  }
  return out;
}

json emit_fused_elementwise(const Intent& intent, const json& bindings) {
  if (intent.outputs.size() != 1) fail("elementwise lowering requires a single output");
  const std::string out_name = intent.outputs[0];
  auto out_it = intent.tensors.find(out_name);
  if (out_it == intent.tensors.end()) fail("elementwise lowering: missing output tensor in intent.tensors");

  const Tensor& out_t = out_it->second;
  const int out_rank = static_cast<int>(out_t.shape.size());
  if (out_rank > 4) fail("elementwise lowering supports rank<=4");

  std::unordered_map<std::string, bool> produced;
  for (const auto& op : intent.ops) produced[op.output] = true;

  std::unordered_map<std::string, bool> outs;
  outs[out_name] = true;

  std::unordered_map<std::string, bool> used_tensors;
  used_tensors[out_name] = true;
  for (const auto& op : intent.ops) {
    for (const auto& inp : op.inputs) {
      if (intent.tensors.find(inp) != intent.tensors.end()) used_tensors[inp] = true;
    }
    if (intent.tensors.find(op.output) != intent.tensors.end()) used_tensors[op.output] = true;
  }

  std::unordered_map<std::string, bool> dim_syms_map;
  for (const auto& kv : used_tensors) {
    auto it = intent.tensors.find(kv.first);
    if (it == intent.tensors.end()) continue;
    for (const auto& d : it->second.shape) {
      if (d.is_string()) dim_syms_map[d.get<std::string>()] = true;
    }
  }
  std::vector<std::string> dim_syms;
  dim_syms.reserve(dim_syms_map.size());
  for (const auto& kv : dim_syms_map) dim_syms.push_back(kv.first);
  std::sort(dim_syms.begin(), dim_syms.end());

  std::vector<std::string> tensor_dim_args;
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> dim_param;
  std::vector<std::string> dim_load;
  std::unordered_map<std::string, std::string> dim_expr;
  for (const auto& sym : dim_syms) {
    if (is_scalar_tensor(intent, sym, "i32")) {
      tensor_dim_args.push_back(sym);
      dim_param.push_back("const int* " + sym + "_ptr");
      dim_load.push_back("const int " + sym + " = " + sym + "_ptr ? " + sym + "_ptr[0] : 0;");
      dim_expr[sym] = sym;
    } else {
      scalar_args[sym] = "i32";
      dim_param.push_back("int " + sym);
      dim_expr[sym] = sym;
    }
  }

  std::vector<std::string> external_inputs;
  for (const auto& op : intent.ops) {
    for (const auto& inp : op.inputs) {
      if (intent.tensors.find(inp) == intent.tensors.end()) continue;
      if (produced.find(inp) != produced.end()) continue;
      if (outs.find(inp) != outs.end()) continue;
      if (is_scalar_tensor(intent, inp, "i32") && dim_syms_map.find(inp) != dim_syms_map.end()) continue;
      bool seen = false;
      for (const auto& x : external_inputs)
        if (x == inp) seen = true;
      if (!seen) external_inputs.push_back(inp);
    }
  }

  std::vector<std::string> tensor_args;
  tensor_args.reserve(external_inputs.size() + 1 + tensor_dim_args.size());
  for (const auto& x : external_inputs) tensor_args.push_back(x);
  tensor_args.push_back(out_name);
  for (const auto& x : tensor_dim_args) tensor_args.push_back(x);

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;

  std::vector<std::string> out_dims_expr;
  out_dims_expr.reserve(out_rank);
  for (const auto& d : out_t.shape) {
    if (d.is_number_integer()) out_dims_expr.push_back(std::to_string(d.get<int64_t>()));
    else if (d.is_number()) out_dims_expr.push_back(std::to_string(static_cast<int64_t>(d.get<double>())));
    else if (d.is_string()) out_dims_expr.push_back(d.get<std::string>());
    else fail("elementwise: invalid out shape dim");
  }

  std::vector<std::string> idx_vars;
  std::vector<std::string> idx_code;
  if (out_rank == 0) {
  } else if (out_rank == 1) {
    idx_vars = {"i0"};
    idx_code.push_back("const int64_t i0 = tid;");
  } else if (out_rank == 2) {
    idx_vars = {"i0", "i1"};
    const std::string d1 = out_dims_expr[1].size() && std::isdigit((unsigned char)out_dims_expr[1][0]) ? out_dims_expr[1] : dim_expr.at(out_dims_expr[1]);
    idx_code.push_back("const int64_t i0 = tid / (int64_t)(" + d1 + ");");
    idx_code.push_back("const int64_t i1 = tid - i0 * (int64_t)(" + d1 + ");");
  } else if (out_rank == 3) {
    idx_vars = {"i0", "i1", "i2"};
    const std::string d1 = out_dims_expr[1].size() && std::isdigit((unsigned char)out_dims_expr[1][0]) ? out_dims_expr[1] : dim_expr.at(out_dims_expr[1]);
    const std::string d2 = out_dims_expr[2].size() && std::isdigit((unsigned char)out_dims_expr[2][0]) ? out_dims_expr[2] : dim_expr.at(out_dims_expr[2]);
    idx_code.push_back("const int64_t i0 = tid / ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + "));");
    idx_code.push_back("const int64_t rem = tid - i0 * ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + "));");
    idx_code.push_back("const int64_t i1 = rem / (int64_t)(" + d2 + ");");
    idx_code.push_back("const int64_t i2 = rem - i1 * (int64_t)(" + d2 + ");");
  } else {
    idx_vars = {"i0", "i1", "i2", "i3"};
    const std::string d1 = out_dims_expr[1].size() && std::isdigit((unsigned char)out_dims_expr[1][0]) ? out_dims_expr[1] : dim_expr.at(out_dims_expr[1]);
    const std::string d2 = out_dims_expr[2].size() && std::isdigit((unsigned char)out_dims_expr[2][0]) ? out_dims_expr[2] : dim_expr.at(out_dims_expr[2]);
    const std::string d3 = out_dims_expr[3].size() && std::isdigit((unsigned char)out_dims_expr[3][0]) ? out_dims_expr[3] : dim_expr.at(out_dims_expr[3]);
    idx_code.push_back("const int64_t i0 = tid / ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t rem0 = tid - i0 * ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t i1 = rem0 / ((int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t rem1 = rem0 - i1 * ((int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t i2 = rem1 / (int64_t)(" + d3 + ");");
    idx_code.push_back("const int64_t i3 = rem1 - i2 * (int64_t)(" + d3 + ");");
  }

  std::unordered_map<std::string, std::string> value_expr;
  std::unordered_map<std::string, std::string> value_type;

  auto load_tensor = [&](const std::string& name) -> std::string {
    auto it = intent.tensors.find(name);
    if (it == intent.tensors.end()) fail("elementwise: unknown tensor " + name);
    const Tensor& t = it->second;
    const std::string idx_expr = emit_broadcast_index_expr(out_rank, t.shape, idx_vars, dim_expr);
    return name + "[(size_t)(" + idx_expr + ")]";
  };
  auto val = [&](const std::string& name) -> std::string {
    auto it = value_expr.find(name);
    if (it != value_expr.end()) return it->second;
    if (dim_syms_map.find(name) != dim_syms_map.end() && is_scalar_tensor(intent, name, "i32")) return name;
    if (intent.tensors.find(name) == intent.tensors.end()) fail("elementwise: unknown value " + name);
    return load_tensor(name);
  };

  std::vector<std::string> code_lines;
  for (const auto& op : intent.ops) {
    const std::string opname = op.op;
    const std::string outn = op.output;
    auto outt_it = intent.tensors.find(outn);
    if (outt_it == intent.tensors.end()) fail("elementwise: op output missing from tensors: " + outn);
    const std::string out_dt = outt_it->second.dtype;
    const std::string cty = c_type_for_dtype(out_dt);

    auto emit_assign = [&](const std::string& expr) {
      const std::string vname = "v_" + c_ident(outn);
      code_lines.push_back(cty + " " + vname + " = " + expr + ";");
      value_expr[outn] = vname;
      value_type[outn] = out_dt;
    };

    if (opname == "const") {
      json v = op.attrs.is_object() ? op.attrs.value("value", json(0)) : json(0);
      emit_assign(c_scalar_literal(out_dt, v));
    } else if (opname == "identity") {
      if (op.inputs.size() != 1) fail("identity expects 1 input");
      emit_assign(val(op.inputs[0]));
    } else if (opname == "broadcast_in_dim") {
      if (op.inputs.size() != 1) fail("broadcast_in_dim expects 1 input");
      emit_assign(val(op.inputs[0]));
    } else if (opname == "cast") {
      if (op.inputs.size() != 1) fail("cast expects 1 input");
      emit_assign("(" + cty + ")(" + val(op.inputs[0]) + ")");
    } else if (opname == "where") {
      if (op.inputs.size() != 3) fail("where expects 3 inputs (cond, x, y)");
      emit_assign("(" + val(op.inputs[0]) + " ? " + val(op.inputs[1]) + " : " + val(op.inputs[2]) + ")");
    } else if (opname == "not") {
      if (op.inputs.size() != 1) fail("not expects 1 input");
      emit_assign("(!" + val(op.inputs[0]) + ")");
    } else if (opname == "and" || opname == "or") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      const std::string op2 = (opname == "and") ? "&&" : "||";
      emit_assign("(" + val(op.inputs[0]) + " " + op2 + " " + val(op.inputs[1]) + ")");
    } else if (opname == "add" || opname == "sub" || opname == "mul" || opname == "div" || opname == "max" || opname == "min") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      const std::string a = val(op.inputs[0]);
      const std::string b = val(op.inputs[1]);
      if (opname == "add")
        emit_assign("(" + a + " + " + b + ")");
      else if (opname == "sub")
        emit_assign("(" + a + " - " + b + ")");
      else if (opname == "mul")
        emit_assign("(" + a + " * " + b + ")");
      else if (opname == "div")
        emit_assign("(" + a + " / " + b + ")");
      else if (opname == "max")
        emit_assign("fmaxf(" + a + ", " + b + ")");
      else
        emit_assign("fminf(" + a + ", " + b + ")");
    } else if (opname == "relu") {
      if (op.inputs.size() != 1) fail("relu expects 1 input");
      emit_assign("fmaxf(" + val(op.inputs[0]) + ", 0.0f)");
    } else if (opname == "abs") {
      if (op.inputs.size() != 1) fail("abs expects 1 input");
      emit_assign("fabsf(" + val(op.inputs[0]) + ")");
    } else if (opname == "exp") {
      if (op.inputs.size() != 1) fail("exp expects 1 input");
      emit_assign("__expf(" + val(op.inputs[0]) + ")");
    } else if (opname == "floor") {
      if (op.inputs.size() != 1) fail("floor expects 1 input");
      emit_assign("floorf(" + val(op.inputs[0]) + ")");
    } else if (opname == "rsqrt") {
      if (op.inputs.size() != 1) fail("rsqrt expects 1 input");
      emit_assign("rsqrtf(" + val(op.inputs[0]) + ")");
    } else if (opname == "ne" || opname == "lt" || opname == "le" || opname == "gt" || opname == "ge") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      std::string cmp;
      if (opname == "ne") cmp = "!=";
      else if (opname == "lt")
        cmp = "<";
      else if (opname == "le")
        cmp = "<=";
      else if (opname == "gt")
        cmp = ">";
      else
        cmp = ">=";
      emit_assign("(" + val(op.inputs[0]) + " " + cmp + " " + val(op.inputs[1]) + ")");
    } else {
      fail("elementwise lowering unsupported op: " + opname);
    }
  }

  if (value_expr.find(out_name) == value_expr.end()) fail("elementwise lowering did not produce the output value");
  const std::string out_var = value_expr.at(out_name);
  const std::string out_cty = c_type_for_dtype(out_t.dtype);

  std::string total_expr;
  if (out_rank == 0) {
    total_expr = "1";
  } else {
    for (int i = 0; i < out_rank; ++i) {
      const auto& d = out_t.shape[i];
      std::string part;
      if (d.is_number_integer())
        part = "(int64_t)" + std::to_string(d.get<int64_t>());
      else if (d.is_number())
        part = "(int64_t)" + std::to_string(static_cast<int64_t>(d.get<double>()));
      else if (d.is_string())
        part = "(int64_t)" + dim_expr.at(d.get<std::string>());
      else
        fail("elementwise: invalid out dim");
      if (!total_expr.empty()) total_expr += " * ";
      total_expr += part;
    }
    if (total_expr.empty()) total_expr = "1";
  }

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  std::vector<std::string> params;
  for (const auto& n : external_inputs) {
    const auto it = intent.tensors.find(n);
    if (it == intent.tensors.end()) fail("elementwise: missing tensor for param " + n);
    params.push_back("const " + c_type_for_dtype(it->second.dtype) + "* __restrict__ " + n);
  }
  params.push_back(c_type_for_dtype(out_t.dtype) + "* __restrict__ " + out_name);
  for (const auto& p : dim_param) params.push_back(p);
  for (size_t i = 0; i < params.size(); ++i) {
    const std::string comma = (i + 1 < params.size()) ? "," : "";
    w.line(params[i] + comma);
  }
  w.dedent();
  w.line(") {");
  w.indent();
  for (const auto& dl : dim_load) w.line(dl);
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t total = " + total_expr + ";");
  w.line("if (tid >= total) return;");
  for (const auto& l : idx_code) w.line(l);
  for (const auto& l : code_lines) w.line(l);
  w.line(out_name + "[(size_t)tid] = (" + out_cty + ")" + out_var + ";");
  w.dedent();
  w.line("}");

  std::vector<std::string> arg_names;
  for (const auto& x : external_inputs) arg_names.push_back(x);
  arg_names.push_back(out_name);
  for (const auto& sym : dim_syms) arg_names.push_back(sym);

  int64_t total = 1;
  for (const auto& d : out_t.shape) {
    auto v = resolve_dim_token(d, bindings);
    if (!v.has_value()) fail("elementwise missing binding for out dim");
    total *= *v;
  }
  const int64_t grid_x = (total + block_x - 1) / block_x;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_softmax_2d_last_f32(const Intent& intent, const json& bindings) {
  const std::string out_name = !intent.outputs.empty() ? intent.outputs[0] : std::string("out");

  std::string in_name;
  for (const auto& op : intent.ops) {
    if (op.op == "reduce_max" && !op.inputs.empty()) {
      in_name = op.inputs[0];
      break;
    }
  }
  if (in_name.empty()) {
    std::unordered_map<std::string, bool> produced;
    for (const auto& op : intent.ops) produced[op.output] = true;
    for (const auto& kv : intent.tensors) {
      const std::string& tn = kv.first;
      const Tensor& tt = kv.second;
      if (tn == out_name) continue;
      if (produced.find(tn) != produced.end()) continue;
      if (tt.dtype == "f32" && tt.shape.size() == 2) {
        in_name = tn;
        break;
      }
    }
  }
  if (in_name.empty()) fail("softmax lowering failed to identify input tensor");

  auto it = intent.tensors.find(in_name);
  if (it == intent.tensors.end()) fail("softmax missing input tensor in intent.tensors: " + in_name);
  if (it->second.shape.size() != 2) fail("softmax expects rank-2 input");
  const json R_dim = it->second.shape[0];
  const json C_dim = it->second.shape[1];
  auto R_opt = resolve_dim_token(R_dim, bindings);
  auto C_opt = resolve_dim_token(C_dim, bindings);
  if (!R_opt.has_value() || !C_opt.has_value()) fail("softmax missing bindings for R/C");
  const int64_t R = *R_opt;
  const int64_t C = *C_opt;
  if (C > 1024) fail("softmax MVP supports only C<=1024");

  int64_t block_threads = 0;
  const int64_t hinted_threads = resolve_schedule_int(intent, bindings, "tile_m", 0);
  if (hinted_threads && 0 < hinted_threads && hinted_threads <= 1024) block_threads = hinted_threads;
  if (block_threads <= 0) block_threads = (C >= 128) ? 128 : 64;
  if (block_threads < 32) block_threads = 32;
  if ((block_threads % 32) != 0) block_threads = ((block_threads + 31) / 32) * 32;
  if (block_threads > 1024) block_threads = 1024;

  if (auto tuned = binding_int(bindings, "SOFTMAX_THREADS")) {
    if (0 < *tuned && *tuned <= 1024) {
      block_threads = *tuned;
      if (block_threads < 32) block_threads = 32;
      if ((block_threads % 32) != 0) block_threads = ((block_threads + 31) / 32) * 32;
      if (block_threads > 1024) block_threads = 1024;
    }
  }

  int64_t max_ept = binding_int(bindings, "SOFTMAX_MAX_EPT").value_or(16);
  if (max_ept < 4) max_ept = 4;
  if (max_ept > 16) max_ept = 16;
  int64_t min_threads = std::max<int64_t>(32, (C + max_ept - 1) / max_ept);
  if ((min_threads % 32) != 0) min_threads = ((min_threads + 31) / 32) * 32;
  if (block_threads < min_threads) block_threads = min_threads;
  const int64_t ept = std::max<int64_t>(1, (C + block_threads - 1) / block_threads);

  const bool specialize_dims = binding_int(bindings, "CUDA_SPECIALIZE_DIMS").value_or(0) != 0;
  const std::string R_name = dim_str(R_dim);
  const std::string C_name = dim_str(C_dim);
  bool r_is_tensor = is_scalar_tensor(intent, R_name, "i32");
  bool c_is_tensor = is_scalar_tensor(intent, C_name, "i32");
  if (specialize_dims && r_is_tensor && bindings.contains(R_name)) r_is_tensor = false;
  if (specialize_dims && c_is_tensor && bindings.contains(C_name)) c_is_tensor = false;

  const std::string r_param = r_is_tensor ? ("const int* " + R_name + "_ptr") : "int R";
  const std::string c_param = c_is_tensor ? ("const int* " + C_name + "_ptr") : "int C";
  const std::string r_load = r_is_tensor ? ("const int R = " + R_name + "_ptr ? " + R_name + "_ptr[0] : 0;") : "";
  const std::string c_load = c_is_tensor ? ("const int C = " + C_name + "_ptr ? " + C_name + "_ptr[0] : 0;") : "";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/softmax.cuh\"");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + in_name + ", float* __restrict__ " + out_name + ", " +
         r_param + ", " + c_param + ") {");
  w.indent();
  if (!r_load.empty()) w.line(r_load);
  if (!c_load.empty()) w.line(c_load);
  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_threads) + ";");
  w.line("constexpr int EPT = " + std::to_string(ept) + ";");
  w.line("intentir_cuda::softmax_2d_last_f32<BLOCK_THREADS, EPT>(" + in_name + ", " + out_name + ", R, C);");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {in_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {in_name, out_name};
  if (r_is_tensor) {
    tensor_args.push_back(R_name);
    arg_names.push_back(R_name);
  } else {
    scalar_args.emplace(R_name, "i32");
    arg_names.push_back(R_name);
  }
  if (c_is_tensor) {
    tensor_args.push_back(C_name);
    arg_names.push_back(C_name);
  } else {
    scalar_args.emplace(C_name, "i32");
    arg_names.push_back(C_name);
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {R, 1, 1}}, {"block", {block_threads, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_layernorm_2d_f32(const Intent& intent, const json& bindings) {
  std::unordered_map<std::string, bool> produced;
  for (const auto& op : intent.ops) produced[op.output] = true;
  std::unordered_map<std::string, bool> outs;
  for (const auto& n : intent.outputs) outs[n] = true;

  std::vector<std::string> inputs;
  for (const auto& kv : intent.tensors) {
    const std::string& name = kv.first;
    if (produced.find(name) != produced.end()) continue;
    if (outs.find(name) != outs.end()) continue;
    inputs.push_back(name);
  }

  std::string X_name;
  for (const auto& n : inputs) {
    if (n == "X") {
      X_name = n;
      break;
    }
  }
  if (X_name.empty()) {
    for (const auto& n : inputs) {
      const Tensor& t = intent.tensors.at(n);
      if (t.dtype == "f32" && t.shape.size() == 2) {
        X_name = n;
        break;
      }
    }
  }
  if (X_name.empty()) fail("layernorm lowering cannot find input X");

  std::string W_name;
  std::string B_name;
  for (const auto& n : inputs) {
    if (n == "W") W_name = n;
    if (n == "B") B_name = n;
  }
  if (W_name.empty() || B_name.empty()) {
    std::vector<std::string> rank1;
    for (const auto& n : inputs) {
      const Tensor& t = intent.tensors.at(n);
      if (t.dtype == "f32" && t.shape.size() == 1) rank1.push_back(n);
    }
    if (W_name.empty() && !rank1.empty()) W_name = rank1[0];
    if (B_name.empty() && rank1.size() > 1) B_name = rank1[1];
  }
  if (W_name.empty() || B_name.empty()) fail("layernorm lowering cannot find W/B inputs");

  if (intent.outputs.size() != 3) fail("layernorm lowering expects 3 outputs (Y, Mean, Rstd)");
  const std::string Y_name = intent.outputs[0];
  const std::string Mean_name = intent.outputs[1];
  const std::string Rstd_name = intent.outputs[2];

  auto x_it = intent.tensors.find(X_name);
  if (x_it == intent.tensors.end()) fail("layernorm missing X tensor in intent.tensors");
  if (x_it->second.shape.size() != 2) fail("layernorm expects rank-2 X");
  auto M_opt = resolve_dim_token(x_it->second.shape[0], bindings);
  auto N_opt = resolve_dim_token(x_it->second.shape[1], bindings);
  if (!M_opt.has_value() || !N_opt.has_value()) fail("layernorm missing bindings for M/N");
  const int64_t M = *M_opt;
  const int64_t N = *N_opt;
  (void)N;

  std::optional<double> eps;
  for (const auto& op : intent.ops) {
    if (op.op == "const" && op.output == "eps" && op.attrs.is_object()) {
      auto it = op.attrs.find("value");
      if (it != op.attrs.end() && it->is_number()) eps = it->get<double>();
      else if (it != op.attrs.end() && it->is_string()) {
        try {
          eps = std::stod(it->get<std::string>());
        } catch (...) {
          eps = std::nullopt;
        }
      }
      break;
    }
  }
  if (!eps.has_value()) eps = binding_double(bindings, "eps").value_or(1e-5);

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x > 1024) block_x = 1024;
  auto is_pow2 = [](int64_t x) -> bool { return x > 0 && ((x & (x - 1)) == 0); };
  auto floor_pow2 = [](int64_t x) -> int64_t {
    if (x <= 1) return 1;
    int64_t p = 1;
    while ((p << 1) <= x) p <<= 1;
    return p;
  };
  if (!is_pow2(block_x)) {
    block_x = floor_pow2(block_x);
    if (block_x < 1) block_x = 1;
    if (block_x > 1024) block_x = 1024;
  }

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/layernorm.cuh\"");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + X_name + ",");
  w.line("float* __restrict__ " + Y_name + ",");
  w.line("const float* __restrict__ " + W_name + ",");
  w.line("const float* __restrict__ " + B_name + ",");
  w.line("float* __restrict__ " + Mean_name + ",");
  w.line("float* __restrict__ " + Rstd_name + ",");
  w.line("int M,");
  w.line("int N,");
  w.line("float eps) {");
  w.dedent();
  w.indent();
  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
  w.line("intentir_cuda::layernorm_2d_f32<BLOCK_THREADS>(" + X_name + ", " + Y_name + ", " + W_name + ", " + B_name + ", " + Mean_name + ", " +
         Rstd_name + ", M, N, eps);");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  if (!out_bindings.contains("eps")) out_bindings["eps"] = *eps;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{X_name, Y_name, W_name, B_name, Mean_name, Rstd_name},
      /*scalar_args=*/{{"M", "i32"}, {"N", "i32"}, {"eps", "f32"}},
      /*arg_names=*/{X_name, Y_name, W_name, B_name, Mean_name, Rstd_name, "M", "N", "eps"});
  out["launch"] = {{"grid", {M, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {Y_name, Mean_name, Rstd_name};
  out["bindings"] = out_bindings;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string intent_path;
    std::string bindings_path;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--intent" && i + 1 < argc) intent_path = argv[++i];
      else if ((a == "--bindings" || a == "--shapes") && i + 1 < argc) bindings_path = argv[++i];
      else if (a == "-h" || a == "--help") {
        std::cout << "usage: intentir_cuda_codegen --intent <intent.json> --bindings <bindings.json>\\n";
        return 0;
      } else {
        fail("unknown arg: " + a);
      }
    }
    if (intent_path.empty() || bindings_path.empty()) {
      std::cerr << "missing --intent/--bindings (use --help)\\n";
      return 2;
    }

    Intent intent = parse_intent(read_json_file(intent_path));
    json bindings_json = read_json_file(bindings_path);
    if (!bindings_json.is_object()) fail("bindings must be object");

    if (intent.ops.size() == 1 && intent.ops[0].op == "matmul") {
      json out = emit_matmul_f32(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "transpose") {
      json out = emit_transpose_2d_f32(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "reduce_sum") {
      json out = emit_reduce_sum_2d_axis1_f32(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "reduce_max") {
      json out = emit_reduce_max_2d_axis1_f32(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "gather") {
      json out = emit_gather2d_f32(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "dropout") {
      json out = emit_dropout(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "warp") {
      json out = emit_warp(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "correlation") {
      json out = emit_correlation(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "resize") {
      json out = emit_resize_bilinear2x_i8(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    if (intent.ops.size() == 1 && intent.ops[0].op == "rope") {
      json out = emit_rope_f32(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }

    // Pattern-based kernels.
    if (intent.ops.size() == 3 && intent.ops[0].op == "const" && intent.ops[1].op == "ne" && intent.ops[2].op == "reduce_any") {
      json out = emit_any_dim_f32_to_i1(intent, bindings_json);
      std::cout << out.dump() << "\n";
      return 0;
    }
    {
      bool hasY = false, hasMean = false, hasRstd = false;
      for (const auto& o : intent.outputs) {
        if (o == "Y") hasY = true;
        if (o == "Mean") hasMean = true;
        if (o == "Rstd") hasRstd = true;
      }
      if (hasY && hasMean && hasRstd) {
        json out = emit_layernorm_2d_f32(intent, bindings_json);
        std::cout << out.dump() << "\n";
        return 0;
      }
    }
    {
      const std::string nm = ascii_lower(intent.name);
      bool is_softmax = (nm.find("softmax") != std::string::npos);
      if (!is_softmax) {
        bool has_reduce_max = false, has_reduce_sum = false, has_exp = false, has_div = false;
        for (const auto& op : intent.ops) {
          if (op.op == "reduce_max") has_reduce_max = true;
          if (op.op == "reduce_sum") has_reduce_sum = true;
          if (op.op == "exp") has_exp = true;
          if (op.op == "div") has_div = true;
        }
        is_softmax = has_reduce_max && has_reduce_sum && has_exp && has_div;
      }
      if (is_softmax) {
        json out = emit_softmax_2d_last_f32(intent, bindings_json);
        std::cout << out.dump() << "\n";
        return 0;
      }
    }
    {
      auto is_elem_op = [](const std::string& op) -> bool {
        static const std::unordered_map<std::string, bool> k = {
            {"const", true},
            {"identity", true},
            {"broadcast_in_dim", true},
            {"cast", true},
            {"add", true},
            {"sub", true},
            {"mul", true},
            {"div", true},
            {"max", true},
            {"min", true},
            {"relu", true},
            {"abs", true},
            {"exp", true},
            {"floor", true},
            {"rsqrt", true},
            {"ne", true},
            {"lt", true},
            {"le", true},
            {"gt", true},
            {"ge", true},
            {"and", true},
            {"or", true},
            {"not", true},
            {"where", true},
        };
        return k.find(op) != k.end();
      };
      bool all_elem = !intent.ops.empty();
      for (const auto& op : intent.ops) {
        if (!is_elem_op(op.op)) {
          all_elem = false;
          break;
        }
      }
      if (all_elem) {
        json out = emit_fused_elementwise(intent, bindings_json);
        std::cout << out.dump() << "\n";
        return 0;
      }
    }

    fail(
        "unsupported intent for cuda cpp codegen (supported: matmul, dropout, softmax, layernorm, correlation, resize, rope, warp, transpose, reduce_sum, reduce_max, any_dim, gather, fused_elementwise)");
  } catch (const std::exception& e) {
    std::cerr << "intentir_cuda_codegen error: " << e.what() << "\\n";
    return 3;
  }
}
