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

    fail("unsupported intent for cuda cpp codegen (supported: dropout, warp, correlation, resize)");
  } catch (const std::exception& e) {
    std::cerr << "intentir_cuda_codegen error: " << e.what() << "\\n";
    return 3;
  }
}
