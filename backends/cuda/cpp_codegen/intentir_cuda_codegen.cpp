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

    fail("unsupported intent for cuda cpp codegen (supported: dropout, warp, correlation, resize, rope, softmax, layernorm)");
  } catch (const std::exception& e) {
    std::cerr << "intentir_cuda_codegen error: " << e.what() << "\\n";
    return 3;
  }
}
