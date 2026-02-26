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

bool _reduce_axis_is_all_2d(const json& attrs) {
  if (!attrs.is_object()) return false;
  auto dims = attrs.value("dims", json());
  if (!dims.is_array() || dims.size() != 2) return false;
  bool has0 = false, has1 = false;
  for (const auto& d : dims) {
    try {
      const int v = d.get<int>();
      if (v == 0) has0 = true;
      if (v == 1) has1 = true;
    } catch (...) {
      return false;
    }
  }
  return has0 && has1;
}

bool _attrs_return_indices(const json& attrs) {
  if (!attrs.is_object()) return false;
  auto it = attrs.find("return_indices");
  if (it == attrs.end()) return false;
  if (it->is_boolean()) return it->get<bool>();
  if (it->is_number_integer()) return it->get<int64_t>() != 0;
  if (it->is_number()) return it->get<double>() != 0.0;
  if (it->is_string()) {
    const std::string s = ascii_lower(it->get<std::string>());
    return s == "1" || s == "true" || s == "yes";
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

int64_t _threads_from_contiguous_range(int64_t range_len, int64_t fallback) {
  if (range_len <= 0) return fallback;
  int64_t t = fallback;
  if (range_len >= 1024)
    t = 256;
  else if (range_len >= 512)
    t = 128;
  else if (range_len >= 128)
    t = 64;
  t = std::max<int64_t>(32, std::min<int64_t>(1024, t));
  if ((t % 32) != 0) t = ((t + 31) / 32) * 32;
  return t;
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
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("#include \"kernels/reduce.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name +
         "(const float* __restrict__ " + inp_name + ", float* __restrict__ " + out_name + ", " +
         m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("float acc = 0.0f;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc += intentir_ldg_f32(row + (size_t)n);");
  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
  w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red;");
  w.line("const float sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(acc, &red);");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[(size_t)m] = sum;");
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
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("#include \"kernels/reduce.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name +
         "(const float* __restrict__ " + inp_name + ", float* __restrict__ " + out_name + ", " +
         m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("float acc = -INFINITY;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc = fmaxf(acc, intentir_ldg_f32(row + (size_t)n));");
  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
  w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red;");
  w.line("const float mx = intentir_cuda::block_allreduce_max<BLOCK_THREADS>(acc, &red);");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[(size_t)m] = mx;");
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

json emit_reduce_mean_2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 3 && intent.ops[0].op == "const" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "div")) {
    fail("reduce_mean lowering expects const->reduce_sum->div");
  }
  const Op& const_op = intent.ops[0];
  const Op& sum_op = intent.ops[1];
  const Op& div_op = intent.ops[2];
  if (sum_op.inputs.size() != 1) fail("reduce_mean reduce_sum expects 1 input");
  if (!_reduce_axis_is_1(sum_op.attrs)) fail("reduce_mean MVP supports only axis=1 for 2D tensors");
  if (div_op.inputs.size() != 2) fail("reduce_mean div expects 2 inputs");
  if (div_op.inputs[0] != sum_op.output || div_op.inputs[1] != const_op.output) {
    fail("reduce_mean expects div(sum_out, const_out)");
  }
  const std::string inp_name = sum_op.inputs[0];
  const std::string out_name = div_op.output;

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("reduce_mean missing/invalid bindings: M/N");

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
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("#include \"kernels/reduce.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name +
         "(const float* __restrict__ " + inp_name + ", float* __restrict__ " + out_name + ", " +
         m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("float acc = 0.0f;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) acc += intentir_ldg_f32(row + (size_t)n);");
  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
  w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red;");
  w.line("const float sum = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(acc, &red);");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[(size_t)m] = (N > 0 ? (sum / (float)N) : 0.0f);");
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

json emit_std_2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "std")) fail("std lowering expects a single std op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("std expects one input");
  if (!_reduce_axis_is_1(op.attrs)) fail("std MVP supports only axis=1 for 2D tensors");
  const bool keepdims = op.attrs.value("keepdims", false);
  if (keepdims) fail("std MVP expects keepdims=false");
  const int correction = op.attrs.value("correction", 1);

  const std::string inp_name = op.inputs[0];
  const std::string out_name = op.output;
  auto x_it = intent.tensors.find(inp_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("std tensors missing");
  if (x_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("std MVP supports f32 only");

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("std missing/invalid bindings: M/N");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

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
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line(m_param + ",");
  w.line(n_param);
  w.dedent();
  w.line(") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= (int64_t)M) return;");
  w.line("const float* xrow = " + inp_name + " + row * (int64_t)N;");
  w.line("float sum = 0.0f;");
  w.line("float sumsq = 0.0f;");
  w.line("int64_t j = 0;");
  w.line("for (; j + 3 < (int64_t)N; j += 4) {");
  w.indent();
  w.line("const float x0 = xrow[j + 0];");
  w.line("const float x1 = xrow[j + 1];");
  w.line("const float x2 = xrow[j + 2];");
  w.line("const float x3 = xrow[j + 3];");
  w.line("sum += x0 + x1 + x2 + x3;");
  w.line("sumsq = fmaf(x0, x0, sumsq);");
  w.line("sumsq = fmaf(x1, x1, sumsq);");
  w.line("sumsq = fmaf(x2, x2, sumsq);");
  w.line("sumsq = fmaf(x3, x3, sumsq);");
  w.dedent();
  w.line("}");
  w.line("for (; j < (int64_t)N; ++j) {");
  w.indent();
  w.line("const float x = xrow[j];");
  w.line("sum += x;");
  w.line("sumsq = fmaf(x, x, sumsq);");
  w.dedent();
  w.line("}");
  w.line("const float mean = sum / (float)N;");
  w.line("const float acc = fmaxf(fmaf(-mean, sum, sumsq), 0.0f);");
  w.line("const int64_t denom_i = (int64_t)N - (int64_t)" + std::to_string(correction) + ";");
  w.line("if (denom_i <= 0) { " + out_name + "[row] = NAN; return; }");
  w.line("const float var = acc / (float)denom_i;");
  w.line(out_name + "[row] = sqrtf(fmaxf(var, 0.0f));");
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
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_reduce_min_2d_all_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "reduce_min")) fail("reduce_min(all) lowering expects a single reduce_min op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("reduce_min(all) expects 1 input");
  if (!_reduce_axis_is_all_2d(op.attrs)) fail("reduce_min(all) expects dims=[0,1]");
  const std::string inp_name = op.inputs[0];
  const std::string out_name = op.output;

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("reduce_min(all) missing/invalid bindings: M/N");

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
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name +
         "(const float* __restrict__ " + inp_name + ", float* __restrict__ " + out_name + ", " + m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int total = M * N;");
  w.line("float acc = INFINITY;");
  w.line("for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x) acc = fminf(acc, intentir_ldg_f32(" + inp_name + " + (size_t)i));");
  w.line("__shared__ float smin[" + std::to_string(block_x) + "];");
  w.line("smin[(int)threadIdx.x] = acc;");
  w.line("__syncthreads();");
  w.line("for (int step = (int)blockDim.x >> 1; step > 0; step >>= 1) {");
  w.indent();
  w.line("if ((int)threadIdx.x < step) smin[(int)threadIdx.x] = fminf(smin[(int)threadIdx.x], smin[(int)threadIdx.x + step]);");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[0] = smin[0];");
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
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_reduce_min_2d_axis1_with_indices_f32_i64(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() >= 1 && intent.ops.back().op == "reduce_min")) fail("reduce_min(axis=1,indices) expects trailing reduce_min");
  const Op& op = intent.ops.back();
  if (op.inputs.size() != 1) fail("reduce_min(axis=1,indices) expects 1 input");
  if (!_reduce_axis_is_1(op.attrs)) fail("reduce_min(axis=1,indices) expects axis=1");
  if (!_attrs_return_indices(op.attrs)) fail("reduce_min(axis=1,indices) expects return_indices=true");
  const auto idx_it = op.attrs.find("index_output");
  if (idx_it == op.attrs.end() || !idx_it->is_string()) fail("reduce_min(axis=1,indices) missing index_output");

  const std::string inp_name = op.inputs[0];
  const std::string out_name = op.output;
  const std::string idx_name = idx_it->get<std::string>();
  auto idx_tensor_it = intent.tensors.find(idx_name);
  if (idx_tensor_it == intent.tensors.end()) fail("reduce_min(axis=1,indices) index_output tensor not found");
  const std::string idx_dtype = idx_tensor_it->second.dtype;
  const std::string idx_ctype = (idx_dtype == "i64") ? "int64_t" : ((idx_dtype == "i32") ? "int" : "");
  if (idx_ctype.empty()) fail("reduce_min(axis=1,indices) index_output dtype must be i32/i64");

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("reduce_min(axis=1,indices) missing/invalid bindings: M/N");

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
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line(idx_ctype + "* __restrict__ " + idx_name + ",");
  w.line(m_param + ",");
  w.line(n_param + ") {");
  w.dedent();
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("float best = INFINITY;");
  w.line("int best_idx = 0;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {");
  w.indent();
  w.line("const float v = intentir_ldg_f32(row + (size_t)n);");
  w.line("if (v < best || (v == best && n < best_idx)) { best = v; best_idx = n; }");
  w.dedent();
  w.line("}");
  w.line("__shared__ float s_val[" + std::to_string(block_x) + "];");
  w.line("__shared__ int s_idx[" + std::to_string(block_x) + "];");
  w.line("s_val[(int)threadIdx.x] = best;");
  w.line("s_idx[(int)threadIdx.x] = best_idx;");
  w.line("__syncthreads();");
  w.line("for (int step = (int)blockDim.x >> 1; step > 0; step >>= 1) {");
  w.indent();
  w.line("if ((int)threadIdx.x < step) {");
  w.indent();
  w.line("const float ov = s_val[(int)threadIdx.x + step];");
  w.line("const int oi = s_idx[(int)threadIdx.x + step];");
  w.line("if (ov < s_val[(int)threadIdx.x] || (ov == s_val[(int)threadIdx.x] && oi < s_idx[(int)threadIdx.x])) {");
  w.indent();
  w.line("s_val[(int)threadIdx.x] = ov;");
  w.line("s_idx[(int)threadIdx.x] = oi;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if ((int)threadIdx.x == 0) {");
  w.indent();
  w.line(out_name + "[(size_t)m] = s_val[0];");
  w.line(idx_name + "[(size_t)m] = (" + idx_ctype + ")s_idx[0];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name, idx_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name, idx_name};
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
  out["output_names"] = {out_name, idx_name};
  out["bindings"] = bindings;
  return out;
}

json emit_arg_extrema_2d_axis1_i64(const Intent& intent, const json& bindings, bool is_max) {
  if (!(intent.ops.size() == 1 && (intent.ops[0].op == "argmax" || intent.ops[0].op == "argmin"))) {
    fail("arg extrema lowering expects a single argmax/argmin op");
  }
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("arg extrema expects 1 input");
  if (!_reduce_axis_is_1(op.attrs)) fail("arg extrema MVP supports only axis=1");
  const std::string inp_name = op.inputs[0];
  const std::string out_name = op.output;
  auto out_tensor_it = intent.tensors.find(out_name);
  if (out_tensor_it == intent.tensors.end()) fail("arg extrema output tensor not found");
  const std::string out_dtype = out_tensor_it->second.dtype;
  const std::string out_ctype = (out_dtype == "i64") ? "int64_t" : ((out_dtype == "i32") ? "int" : "");
  if (out_ctype.empty()) fail("arg extrema output dtype must be i32/i64");

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("arg extrema missing/invalid bindings: M/N");

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
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line(out_ctype + "* __restrict__ " + out_name + ",");
  w.line(m_param + ",");
  w.line(n_param + ") {");
  w.dedent();
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  if (is_max) {
    w.line("float best = -INFINITY;");
  } else {
    w.line("float best = INFINITY;");
  }
  w.line("int best_idx = 0;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {");
  w.indent();
  w.line("const float v = intentir_ldg_f32(row + (size_t)n);");
  if (is_max) {
    w.line("if (v > best || (v == best && n < best_idx)) { best = v; best_idx = n; }");
  } else {
    w.line("if (v < best || (v == best && n < best_idx)) { best = v; best_idx = n; }");
  }
  w.dedent();
  w.line("}");
  w.line("__shared__ float s_val[" + std::to_string(block_x) + "];");
  w.line("__shared__ int s_idx[" + std::to_string(block_x) + "];");
  w.line("s_val[(int)threadIdx.x] = best;");
  w.line("s_idx[(int)threadIdx.x] = best_idx;");
  w.line("__syncthreads();");
  w.line("for (int step = (int)blockDim.x >> 1; step > 0; step >>= 1) {");
  w.indent();
  w.line("if ((int)threadIdx.x < step) {");
  w.indent();
  w.line("const float ov = s_val[(int)threadIdx.x + step];");
  w.line("const int oi = s_idx[(int)threadIdx.x + step];");
  if (is_max) {
    w.line("if (ov > s_val[(int)threadIdx.x] || (ov == s_val[(int)threadIdx.x] && oi < s_idx[(int)threadIdx.x])) {");
  } else {
    w.line("if (ov < s_val[(int)threadIdx.x] || (ov == s_val[(int)threadIdx.x] && oi < s_idx[(int)threadIdx.x])) {");
  }
  w.indent();
  w.line("s_val[(int)threadIdx.x] = ov;");
  w.line("s_idx[(int)threadIdx.x] = oi;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[(size_t)m] = (" + out_ctype + ")s_idx[0];");
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
	  w.line("#include <stddef.h>");
	  w.line("#include <stdint.h>");
	  w.line("#include \"intentir_cuda_ops.cuh\"");
	  w.line("#include \"kernels/reduce.cuh\"");
	  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name +
	         "(const float* __restrict__ " + inp_name + ", bool* __restrict__ " + out_name + ", " + m_param + ", " + n_param + ") {");
	  w.indent();
	  if (!m_load.empty()) w.line(m_load);
	  if (!n_load.empty()) w.line(n_load);
	  w.line("const int m = (int)blockIdx.x;");
	  w.line("if (m >= M) return;");
	  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
	  w.line("int anyv = 0;");
	  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
	  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) anyv |= (intentir_ldg_f32(row + (size_t)n) != " + z_lit +
	         ");");
	  w.line("__shared__ intentir_cuda::BlockAllreduceI32<BLOCK_THREADS> red;");
	  w.line("const int any_out = intentir_cuda::block_allreduce_max<BLOCK_THREADS>(anyv, &red);");
	  w.line("if ((int)threadIdx.x == 0) " + out_name + "[(size_t)m] = (any_out != 0);");
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

json emit_all_dim_f32_to_i1(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 4) fail("all-dim lowering expects 4 ops (const, eq, reduce_any, not)");
  const Op& c0 = intent.ops[0];
  const Op& eq0 = intent.ops[1];
  const Op& r0 = intent.ops[2];
  const Op& n0 = intent.ops[3];
  if (c0.op != "const" || eq0.op != "eq" || r0.op != "reduce_any" || n0.op != "not") {
    fail("all-dim lowering expects ops const->eq->reduce_any->not");
  }
  if (eq0.inputs.size() != 2 || r0.inputs.size() != 1 || n0.inputs.size() != 1) fail("all-dim lowering invalid op arity");
  if (r0.inputs[0] != eq0.output) fail("all-dim lowering expects reduce_any(eq_out)");
  if (n0.inputs[0] != r0.output) fail("all-dim lowering expects not(reduce_any_out)");
  const std::string const_out = c0.output;
  const std::string a0 = eq0.inputs[0];
  const std::string b0 = eq0.inputs[1];
  std::string inp_name;
  if (a0 == const_out && b0 != const_out)
    inp_name = b0;
  else if (b0 == const_out && a0 != const_out)
    inp_name = a0;
  else
    fail("all-dim lowering expects eq(inp, const)");
  const std::string out_name = n0.output;

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
  if (!_reduce_axis_is_1(r0.attrs)) fail("all-dim MVP supports only axis=1 for 2D tensors");

  const int64_t M = binding_int(bindings, "M").value_or(-1);
  const int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) fail("all-dim missing/invalid bindings: M/N");

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
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("#include \"kernels/reduce.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name +
         "(const float* __restrict__ " + inp_name + ", bool* __restrict__ " + out_name + ", " + m_param + ", " + n_param + ") {");
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int m = (int)blockIdx.x;");
  w.line("if (m >= M) return;");
  w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
  w.line("int any_zero = 0;");
  w.line("const float* row = " + inp_name + " + (size_t)m * (size_t)N;");
  w.line("for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) any_zero |= (intentir_ldg_f32(row + (size_t)n) == " + z_lit + ");");
  w.line("__shared__ intentir_cuda::BlockAllreduceI32<BLOCK_THREADS> red;");
  w.line("const int any = intentir_cuda::block_allreduce_max<BLOCK_THREADS>(any_zero, &red);");
  w.line("if ((int)threadIdx.x == 0) " + out_name + "[(size_t)m] = ((any != 0) ? false : true);");
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
  int64_t total_elems = -1;
  auto out_it = intent.tensors.find(out_name);
  if (out_it != intent.tensors.end()) {
    const auto resolve_dim = [&](const json& d) -> int64_t {
      if (d.is_number_integer()) return d.get<int64_t>();
      if (d.is_number()) return static_cast<int64_t>(d.get<double>());
      if (d.is_string()) {
        auto b = binding_int(bindings, d.get<std::string>());
        if (!b.has_value()) fail("gather missing binding for output dim: " + d.get<std::string>());
        return *b;
      }
      fail("gather invalid output dim token");
      return -1;
    };
    const auto& out_shape = out_it->second.shape;
    if (out_shape.size() == 1) {
      total_elems = resolve_dim(out_shape[0]);
    } else if (out_shape.size() == 2) {
      total_elems = resolve_dim(out_shape[0]) * resolve_dim(out_shape[1]);
    } else if (!out_shape.empty()) {
      fail("gather2d MVP supports output rank<=2");
    }
  }
  if (total_elems <= 0) total_elems = binding_int(bindings, "L").value_or(-1);
  if (M <= 0 || N <= 0 || total_elems <= 0) fail("gather missing/invalid bindings: M/N/L");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x > 1024) block_x = 1024;
  const int64_t grid_x = (total_elems + block_x - 1) / block_x;

  const std::string other_lit = c_float(other);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("const int* __restrict__ " + row_name + ",");
  w.line("const int* __restrict__ " + col_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line("int M, int N, int T) {");
  w.dedent();
  w.indent();
  w.line("const int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;");
  w.line("if (tid >= T) return;");
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
      /*scalar_args=*/{{"M", "i32"}, {"N", "i32"}, {"T", "i32"}},
      /*arg_names=*/{inp_name, row_name, col_name, out_name, "M", "N", "T"});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["T"] = total_elems;
  out["bindings"] = out_bindings;
  return out;
}

json emit_topk2d_rowwise_f32(const Intent& intent, const json& bindings) {
  // Expected canonical pattern:
  //   sort(inp) -> sorted_vals
  //   iota(...) -> row_idx
  //   iota(...) -> col_idx
  //   gather(sorted_vals, row_idx, col_idx) -> out
  if (intent.ops.size() != 4) fail("topk lowering expects 4 ops (sort, iota, iota, gather)");
  const Op& sort_op = intent.ops[0];
  const Op& iota_row = intent.ops[1];
  const Op& iota_col = intent.ops[2];
  const Op& gather_op = intent.ops[3];
  if (sort_op.op != "sort" || iota_row.op != "iota" || iota_col.op != "iota" || gather_op.op != "gather") {
    fail("topk lowering expects sort->iota->iota->gather");
  }
  if (sort_op.inputs.size() != 1) fail("topk sort expects 1 input");
  if (gather_op.inputs.size() != 3) fail("topk gather expects 3 inputs");
  if (!intent.outputs.size()) fail("topk lowering expects one output");

  const std::string inp_name = sort_op.inputs[0];
  const std::string sorted_name = sort_op.output;
  const std::string out_name = intent.outputs[0];
  if (gather_op.inputs[0] != sorted_name) fail("topk gather first input must be sort output");
  if (gather_op.output != out_name) fail("topk gather output must match intent output");

  // This MVP supports row-wise topk (axis=1, descending=true).
  int axis = 1;
  if (sort_op.attrs.is_object() && sort_op.attrs.contains("axis")) {
    if (sort_op.attrs["axis"].is_number_integer())
      axis = static_cast<int>(sort_op.attrs["axis"].get<int64_t>());
    else if (sort_op.attrs["axis"].is_number())
      axis = static_cast<int>(sort_op.attrs["axis"].get<double>());
  }
  if (axis != 1) fail("topk MVP supports axis=1 only");
  bool descending = true;
  if (sort_op.attrs.is_object() && sort_op.attrs.contains("descending")) {
    const json& dv = sort_op.attrs["descending"];
    if (dv.is_boolean())
      descending = dv.get<bool>();
    else if (dv.is_number_integer())
      descending = (dv.get<int64_t>() != 0);
    else if (dv.is_number())
      descending = (dv.get<double>() != 0.0);
  }
  if (!descending) fail("topk MVP supports descending=true only");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("topk missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("topk invalid dim token");
    return -1;
  };

  int64_t M = binding_int(bindings, "M").value_or(-1);
  int64_t N = binding_int(bindings, "N").value_or(-1);
  int64_t K = binding_int(bindings, "K").value_or(-1);
  auto inp_it = intent.tensors.find(inp_name);
  auto out_it = intent.tensors.find(out_name);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("topk tensors missing in intent");
  if (M <= 0 || N <= 0) {
    const auto& in_shape = inp_it->second.shape;
    if (in_shape.size() != 2) fail("topk MVP expects input rank=2");
    if (M <= 0) M = resolve_dim(in_shape[0]);
    if (N <= 0) N = resolve_dim(in_shape[1]);
  }
  if (K <= 0) {
    const auto& out_shape = out_it->second.shape;
    if (out_shape.size() != 2) fail("topk MVP expects output rank=2");
    K = resolve_dim(out_shape[1]);
  }
  if (M <= 0 || N <= 0 || K <= 0) fail("topk missing/invalid bindings: M/N/K");
  if (K > N) fail("topk invalid shape: K must be <= N");
  if (K > 256) fail("topk MVP supports K<=256");

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const bool k_is_tensor = is_scalar_tensor(intent, "K", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string k_param = k_is_tensor ? "const int* K_ptr" : "int K";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";
  const std::string k_load = k_is_tensor ? "const int K = K_ptr ? K_ptr[0] : 0;" : "";
  auto next_pow2 = [](int64_t v) -> int64_t {
    int64_t p = 1;
    while (p < v && p < 256) p <<= 1;
    return p;
  };
  const bool use_parallel = (N <= 256);
  int64_t block_x = use_parallel ? next_pow2(N) : 1;
  if (use_parallel && block_x < 32) block_x = 32;
  if (use_parallel && block_x > 256) block_x = 256;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line(m_param + ", " + n_param + ", " + k_param + ") {");
  w.dedent();
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  if (!k_load.empty()) w.line(k_load);
  if (use_parallel) {
    const int64_t p2 = block_x;
    w.line("const int row = (int)blockIdx.x;");
    w.line("const int tid = (int)threadIdx.x;");
    w.line("if (row >= M || K <= 0) return;");
    w.line("__shared__ float vals[" + std::to_string(p2) + "];");
    w.line("const float* in_row = " + inp_name + " + (size_t)row * (size_t)N;");
    w.line("if (tid < N) {");
    w.indent();
    w.line("const float v = in_row[(size_t)tid];");
    w.line("vals[tid] = -v;");
    w.dedent();
    w.line("} else {");
    w.indent();
    w.line("vals[tid] = INFINITY;");
    w.dedent();
    w.line("}");
    w.line("__syncthreads();");
    w.line("for (int k = 2; k <= " + std::to_string(p2) + "; k <<= 1) {");
    w.indent();
    w.line("for (int j = (k >> 1); j > 0; j >>= 1) {");
    w.indent();
    w.line("const int ixj = tid ^ j;");
    w.line("if (ixj > tid) {");
    w.indent();
    w.line("const float a = vals[tid];");
    w.line("const float b = vals[ixj];");
    w.line("const bool up = ((tid & k) == 0);");
    w.line("const bool gt = (a > b);");
    w.line("const bool lt = (a < b);");
    w.line("if (up ? gt : lt) { vals[tid] = b; vals[ixj] = a; }");
    w.dedent();
    w.line("}");
    w.line("__syncthreads();");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.line("if (tid < K) {");
    w.indent();
    w.line("float* out_row = " + out_name + " + (size_t)row * (size_t)K;");
    w.line("out_row[(size_t)tid] = -vals[tid];");
    w.dedent();
    w.line("}");
  } else {
    w.line("const int row = (int)blockIdx.x;");
    w.line("if (row >= M || K <= 0) return;");
    w.line("float top_vals[256];");
    w.line("for (int j = 0; j < K; ++j) top_vals[j] = -INFINITY;");
    w.line("const float* in_row = " + inp_name + " + (size_t)row * (size_t)N;");
    w.line("for (int n = 0; n < N; ++n) {");
    w.indent();
    w.line("float v = in_row[(size_t)n];");
    w.line("for (int j = 0; j < K; ++j) {");
    w.indent();
    w.line("if (v > top_vals[j]) {");
    w.indent();
    w.line("const float tmp = top_vals[j];");
    w.line("top_vals[j] = v;");
    w.line("v = tmp;");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.line("float* out_row = " + out_name + " + (size_t)row * (size_t)K;");
    w.line("for (int j = 0; j < K; ++j) out_row[(size_t)j] = top_vals[j];");
  }
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  for (const auto& dim_name : {"M", "N", "K"}) {
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
  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["K"] = K;
  out["bindings"] = out_bindings;
  return out;
}

json emit_trace2d_sum_f32(const Intent& intent, const json& bindings) {
  // Expected canonical pattern:
  // iota(row), iota(col), eq, const(0), where(mask,input,0), reduce_sum(all)
  if (intent.ops.size() != 6) fail("trace lowering expects 6 ops");
  const Op& o0 = intent.ops[0];
  const Op& o1 = intent.ops[1];
  const Op& o2 = intent.ops[2];
  const Op& o3 = intent.ops[3];
  const Op& o4 = intent.ops[4];
  const Op& o5 = intent.ops[5];
  if (o0.op != "iota" || o1.op != "iota" || o2.op != "eq" || o3.op != "const" || o4.op != "where" || o5.op != "reduce_sum") {
    fail("trace lowering expects iota->iota->eq->const->where->reduce_sum");
  }
  if (o4.inputs.size() != 3) fail("trace where expects 3 inputs");
  if (o5.inputs.size() != 1) fail("trace reduce_sum expects 1 input");
  if (o5.inputs[0] != o4.output) fail("trace reduce_sum input must be where output");
  if (intent.outputs.empty()) fail("trace lowering expects one output");
  const std::string out_name = intent.outputs[0];
  if (o5.output != out_name) fail("trace reduce_sum output must match intent output");

  const std::string inp_name = o4.inputs[1];
  auto inp_it = intent.tensors.find(inp_name);
  auto out_it = intent.tensors.find(out_name);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("trace tensors missing in intent");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("trace missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("trace invalid dim token");
    return -1;
  };

  int64_t M = binding_int(bindings, "M").value_or(-1);
  int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) {
    const auto& in_shape = inp_it->second.shape;
    if (in_shape.size() != 2) fail("trace MVP expects input rank=2");
    if (M <= 0) M = resolve_dim(in_shape[0]);
    if (N <= 0) N = resolve_dim(in_shape[1]);
  }
  if (M <= 0 || N <= 0) fail("trace missing/invalid bindings: M/N");

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";

  const int64_t block_x = 256;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line(m_param + ", " + n_param + ") {");
  w.dedent();
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("const int L = (M < N) ? M : N;");
  w.line("float acc = 0.0f;");
  w.line("for (int i = 0; i < L; ++i) {");
  w.indent();
  w.line("acc += " + inp_name + "[(size_t)i * (size_t)N + (size_t)i];");
  w.dedent();
  w.line("}");
  w.line(out_name + "[0] = acc;");
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
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out["bindings"] = out_bindings;
  return out;
}

json emit_diag2d_extract_f32(const Intent& intent, const json& bindings) {
  // Expected canonical pattern:
  // iota(diag_idx) + gather(data, diag_idx, diag_idx) -> output
  if (intent.ops.size() != 2) fail("diag2d lowering expects 2 ops");
  const Op& o0 = intent.ops[0];
  const Op& o1 = intent.ops[1];
  if (o0.op != "iota" || o1.op != "gather") fail("diag2d lowering expects iota->gather");
  if (o1.inputs.size() != 3) fail("diag2d gather expects 3 inputs");
  if (o1.inputs[1] != o0.output || o1.inputs[2] != o0.output) fail("diag2d gather indices must use the same iota output");
  if (intent.outputs.empty()) fail("diag2d lowering expects one output");

  const std::string inp_name = o1.inputs[0];
  const std::string out_name = intent.outputs[0];
  if (o1.output != out_name) fail("diag2d gather output must match intent output");
  auto inp_it = intent.tensors.find(inp_name);
  auto out_it = intent.tensors.find(out_name);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("diag2d tensors missing in intent");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("diag2d missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("diag2d invalid dim token");
    return -1;
  };

  int64_t M = binding_int(bindings, "M").value_or(-1);
  if (M <= 0) {
    const auto& in_shape = inp_it->second.shape;
    if (in_shape.size() != 2) fail("diag2d expects rank-2 input");
    M = resolve_dim(in_shape[0]);
  }
  if (M <= 0) fail("diag2d missing/invalid binding: M");

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";

  const int64_t block_x = 256;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line(m_param + ") {");
  w.dedent();
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  w.line("const int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;");
  w.line("if (row < 0 || row >= M) return;");
  w.line(out_name + "[(size_t)row] = " + inp_name + "[(size_t)row * (size_t)M + (size_t)row];");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  if (m_is_tensor) {
    tensor_args.push_back("M");
    arg_names.push_back("M");
  } else {
    scalar_args.emplace("M", "i32");
    arg_names.push_back("M");
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["M"] = M;
  out["bindings"] = out_bindings;
  return out;
}

json emit_diag_embed2d_f32(const Intent& intent, const json& bindings) {
  // Canonicalized intent writes y[B,N,N] from x[B,N] with row==col mask.
  if (intent.outputs.empty()) fail("diag_embed lowering expects one output");
  const std::string out_name = intent.outputs[0];

  std::string inp_name;
  if (intent.tensors.find("x") != intent.tensors.end()) inp_name = "x";
  if (inp_name.empty() && !intent.ops.empty()) {
    for (const auto& op : intent.ops) {
      for (const auto& in : op.inputs) {
        auto it = intent.tensors.find(in);
        if (it != intent.tensors.end() && it->second.shape.size() == 2 && it->second.dtype == "f32") {
          inp_name = in;
          break;
        }
      }
      if (!inp_name.empty()) break;
    }
  }
  if (inp_name.empty()) fail("diag_embed lowering could not resolve input tensor");

  auto inp_it = intent.tensors.find(inp_name);
  auto out_it = intent.tensors.find(out_name);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("diag_embed tensors missing in intent");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("diag_embed missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("diag_embed invalid dim token");
    return -1;
  };

  int64_t B = binding_int(bindings, "B").value_or(-1);
  int64_t N = binding_int(bindings, "N").value_or(-1);
  if (B <= 0 || N <= 0) {
    const auto& in_shape = inp_it->second.shape;
    if (in_shape.size() != 2) fail("diag_embed expects rank-2 input");
    if (B <= 0) B = resolve_dim(in_shape[0]);
    if (N <= 0) N = resolve_dim(in_shape[1]);
  }
  if (B <= 0 || N <= 0) fail("diag_embed missing/invalid bindings: B/N");

  const bool b_is_tensor = is_scalar_tensor(intent, "B", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const std::string b_param = b_is_tensor ? "const int* B_ptr" : "int B";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string b_load = b_is_tensor ? "const int B = B_ptr ? B_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";

  const int64_t total = B * N * N;
  const int64_t block_x = 256;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line(b_param + ", " + n_param + ") {");
  w.dedent();
  w.indent();
  if (!b_load.empty()) w.line(b_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t total = (int64_t)B * (int64_t)N * (int64_t)N;");
  w.line("if (tid < 0 || tid >= total) return;");
  w.line("const int64_t nn = (int64_t)N * (int64_t)N;");
  w.line("const int b = (int)(tid / nn);");
  w.line("const int64_t rem = tid - (int64_t)b * nn;");
  w.line("const int row = (int)(rem / (int64_t)N);");
  w.line("const int col = (int)(rem - (int64_t)row * (int64_t)N);");
  w.line("const float v = (row == col) ? " + inp_name + "[(size_t)b * (size_t)N + (size_t)col] : 0.0f;");
  w.line(out_name + "[(size_t)tid] = v;");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  for (const auto& dim_name : {"B", "N"}) {
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
  json out_bindings = bindings;
  out_bindings["B"] = B;
  out_bindings["N"] = N;
  out["bindings"] = out_bindings;
  return out;
}

json emit_nonzero2d_f32(const Intent& intent, const json& bindings) {
  // Expected canonical pattern: nonzero(inp) -> out[num_nonzeros, 2]
  if (intent.ops.size() != 1 || intent.ops[0].op != "nonzero") fail("nonzero lowering expects a single nonzero op");
  if (intent.ops[0].inputs.size() != 1) fail("nonzero op expects one input");
  if (intent.outputs.empty()) fail("nonzero lowering expects one output");

  const std::string inp_name = intent.ops[0].inputs[0];
  const std::string out_name = intent.outputs[0];
  if (intent.ops[0].output != out_name) fail("nonzero output must match intent outputs[0]");
  auto inp_it = intent.tensors.find(inp_name);
  auto out_it = intent.tensors.find(out_name);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("nonzero tensors missing in intent");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("nonzero missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("nonzero invalid dim token");
    return -1;
  };

  int64_t M = binding_int(bindings, "M").value_or(-1);
  int64_t N = binding_int(bindings, "N").value_or(-1);
  int64_t K = binding_int(bindings, "num_nonzeros").value_or(-1);
  if (M <= 0 || N <= 0) {
    const auto& in_shape = inp_it->second.shape;
    if (in_shape.size() != 2) fail("nonzero expects rank-2 input");
    if (M <= 0) M = resolve_dim(in_shape[0]);
    if (N <= 0) N = resolve_dim(in_shape[1]);
  }
  if (K <= 0) {
    const auto& out_shape = out_it->second.shape;
    if (out_shape.size() >= 1) K = resolve_dim(out_shape[0]);
  }
  if (M <= 0 || N <= 0 || K <= 0) fail("nonzero missing/invalid bindings: M/N/num_nonzeros");

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const bool k_is_tensor = is_scalar_tensor(intent, "num_nonzeros", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string k_param = k_is_tensor ? "const int* K_ptr" : "int K";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";
  const std::string k_load = k_is_tensor ? "const int K = K_ptr ? K_ptr[0] : 0;" : "";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("int64_t* __restrict__ " + out_name + ",");
  w.line(m_param + ", " + n_param + ", " + k_param + ") {");
  w.dedent();
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  if (!k_load.empty()) w.line(k_load);
  w.line("if ((int)blockIdx.x != 0) return;");
  w.line("constexpr int BLOCK = 256;");
  w.line("__shared__ int s_scan[BLOCK];");
  w.line("__shared__ int s_total;");
  w.line("if ((int)threadIdx.x == 0) s_total = 0;");
  w.line("__syncthreads();");
  w.line("const int tid = (int)threadIdx.x;");
  w.line("const int total = M * N;");
  w.line("for (int base = 0; base < total; base += BLOCK) {");
  w.indent();
  w.line("const int idx = base + tid;");
  w.line("int flag = 0;");
  w.line("if (idx < total) {");
  w.indent();
  w.line("const float v = " + inp_name + "[(size_t)idx];");
  w.line("flag = (v != 0.0f) ? 1 : 0;");
  w.dedent();
  w.line("}");
  w.line("s_scan[tid] = flag;");
  w.line("__syncthreads();");
  w.line("for (int off = 1; off < BLOCK; off <<= 1) {");
  w.indent();
  w.line("int add = 0;");
  w.line("if (tid >= off) add = s_scan[tid - off];");
  w.line("__syncthreads();");
  w.line("s_scan[tid] += add;");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("const int chunk_count = s_scan[BLOCK - 1];");
  w.line("if (flag) {");
  w.indent();
  w.line("const int pos = s_total + s_scan[tid] - 1;");
  w.line("if (pos < K) {");
  w.indent();
  w.line("const int i = idx / N;");
  w.line("const int j = idx - i * N;");
  w.line(out_name + "[(size_t)pos * 2] = (int64_t)i;");
  w.line(out_name + "[(size_t)pos * 2 + 1] = (int64_t)j;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("__syncthreads();");
  w.line("if (tid == 0) s_total += chunk_count;");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if (tid == 0) {");
  w.indent();
  w.line("for (int p = s_total; p < K; ++p) {");
  w.indent();
  w.line(out_name + "[(size_t)p * 2] = 0;");
  w.line(out_name + "[(size_t)p * 2 + 1] = 0;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {inp_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {inp_name, out_name};
  for (const auto& dim_name : {"M", "N", "num_nonzeros"}) {
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
  // Single-CTA parallel scan/writeback (BLOCK=256) for nonzero compaction.
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {256, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["num_nonzeros"] = K;
  out["bindings"] = out_bindings;
  return out;
}

json emit_count_nonzero2d_f32_i64(const Intent& intent, const json& bindings) {
  // Expected canonical pattern:
  // const(0) -> ne(x,0) -> cast(i64) -> reduce_sum(dims=[0,1]) -> out
  if (intent.ops.size() != 4) fail("count_nonzero lowering expects 4 ops");
  const Op& o0 = intent.ops[0];
  const Op& o1 = intent.ops[1];
  const Op& o2 = intent.ops[2];
  const Op& o3 = intent.ops[3];
  if (o0.op != "const" || o1.op != "ne" || o2.op != "cast" || o3.op != "reduce_sum") {
    fail("count_nonzero lowering expects const->ne->cast->reduce_sum");
  }
  if (o1.inputs.size() != 2) fail("count_nonzero ne expects 2 inputs");
  if (o1.inputs[1] != o0.output) fail("count_nonzero ne second input must be const output");
  if (o2.inputs.size() != 1 || o2.inputs[0] != o1.output) fail("count_nonzero cast input mismatch");
  if (o3.inputs.size() != 1 || o3.inputs[0] != o2.output) fail("count_nonzero reduce_sum input mismatch");
  if (intent.outputs.empty()) fail("count_nonzero lowering expects one output");
  const std::string inp_name = o1.inputs[0];
  const std::string out_name = intent.outputs[0];
  if (o3.output != out_name) fail("count_nonzero reduce_sum output must match intent output");
  auto inp_it = intent.tensors.find(inp_name);
  auto out_it = intent.tensors.find(out_name);
  if (inp_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("count_nonzero tensors missing in intent");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("count_nonzero missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("count_nonzero invalid dim token");
    return -1;
  };

  int64_t M = binding_int(bindings, "M").value_or(-1);
  int64_t N = binding_int(bindings, "N").value_or(-1);
  if (M <= 0 || N <= 0) {
    const auto& in_shape = inp_it->second.shape;
    if (in_shape.size() != 2) fail("count_nonzero expects rank-2 input");
    if (M <= 0) M = resolve_dim(in_shape[0]);
    if (N <= 0) N = resolve_dim(in_shape[1]);
  }
  if (M <= 0 || N <= 0) fail("count_nonzero missing/invalid bindings: M/N");

  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M";
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N";
  const std::string m_load = m_is_tensor ? "const int M = M_ptr ? M_ptr[0] : 0;" : "";
  const std::string n_load = n_is_tensor ? "const int N = N_ptr ? N_ptr[0] : 0;" : "";
  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 1024);
  if (block_x <= 0) block_x = 1024;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 64) block_x = 64;
  block_x = _pow2_block(block_x);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("#include \"intentir_cuda_ops.cuh\"");
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("int64_t* __restrict__ " + out_name + ",");
  w.line(m_param + ", " + n_param + ") {");
  w.dedent();
  w.indent();
  if (!m_load.empty()) w.line(m_load);
  if (!n_load.empty()) w.line(n_load);
  w.line("if ((int)blockIdx.x != 0) return;");
  w.line("const int tid = (int)threadIdx.x;");
  w.line("__shared__ int64_t warp_sums[" + std::to_string((block_x + 31) / 32) + "];");
  w.line("int64_t acc = 0;");
  w.line("const int64_t total = (int64_t)M * (int64_t)N;");
  w.line("const int64_t stride = (int64_t)blockDim.x * 4LL;");
  w.line("for (int64_t i = (int64_t)tid * 4LL; i < total; i += stride) {");
  w.indent();
  w.line("if (intentir_ldg_f32(" + inp_name + " + (size_t)i) != 0.0f) ++acc;");
  w.line("if (i + 1 < total && intentir_ldg_f32(" + inp_name + " + (size_t)(i + 1)) != 0.0f) ++acc;");
  w.line("if (i + 2 < total && intentir_ldg_f32(" + inp_name + " + (size_t)(i + 2)) != 0.0f) ++acc;");
  w.line("if (i + 3 < total && intentir_ldg_f32(" + inp_name + " + (size_t)(i + 3)) != 0.0f) ++acc;");
  w.dedent();
  w.line("}");
  w.line("const unsigned mask = 0xffffffffu;");
  w.line("for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(mask, acc, offset);");
  w.line("const int lane = tid & 31;");
  w.line("const int warp = tid >> 5;");
  w.line("if (lane == 0) warp_sums[warp] = acc;");
  w.line("__syncthreads();");
  w.line("if (warp == 0) {");
  w.indent();
  w.line("int64_t block_sum = (lane < " + std::to_string((block_x + 31) / 32) + ") ? warp_sums[lane] : 0;");
  w.line("for (int offset = 16; offset > 0; offset >>= 1) block_sum += __shfl_down_sync(mask, block_sum, offset);");
  w.line("if (lane == 0) " + out_name + "[0] = block_sum;");
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
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out["bindings"] = out_bindings;
  return out;
}

json emit_concat2d_f32(const Intent& intent, const json& bindings) {
  // MVP: one concat op, rank-2 tensors, axis in {0,1}, two inputs.
  if (intent.ops.size() != 1 || intent.ops[0].op != "concat") fail("concat lowering expects a single concat op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("concat lowering expects two inputs");
  if (intent.outputs.empty()) fail("concat lowering expects one output");
  const std::string a_name = op.inputs[0];
  const std::string b_name = op.inputs[1];
  const std::string out_name = intent.outputs[0];
  if (op.output != out_name) fail("concat output must match intent outputs[0]");

  auto a_it = intent.tensors.find(a_name);
  auto b_it = intent.tensors.find(b_name);
  auto out_it = intent.tensors.find(out_name);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || out_it == intent.tensors.end()) {
    fail("concat tensors missing in intent");
  }
  if (a_it->second.shape.size() != 2 || b_it->second.shape.size() != 2 || out_it->second.shape.size() != 2) {
    fail("concat MVP expects rank-2 tensors");
  }

  int axis = 1;
  if (op.attrs.is_object() && op.attrs.contains("axis")) {
    const auto& ax = op.attrs["axis"];
    if (ax.is_number_integer())
      axis = static_cast<int>(ax.get<int64_t>());
    else if (ax.is_number())
      axis = static_cast<int>(ax.get<double>());
    else
      fail("concat axis must be numeric");
  }
  if (axis < 0) axis += 2;
  if (axis != 0 && axis != 1) fail("concat MVP supports axis 0/1 only");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("concat missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("concat invalid dim token");
    return -1;
  };

  const int64_t M0 = resolve_dim(a_it->second.shape[0]);
  const int64_t N0 = resolve_dim(a_it->second.shape[1]);
  const int64_t M1 = resolve_dim(b_it->second.shape[0]);
  const int64_t N1 = resolve_dim(b_it->second.shape[1]);
  const int64_t M_OUT = resolve_dim(out_it->second.shape[0]);
  const int64_t N_OUT = resolve_dim(out_it->second.shape[1]);

  if (M0 <= 0 || N0 <= 0 || M1 <= 0 || N1 <= 0 || M_OUT <= 0 || N_OUT <= 0) fail("concat invalid shape bindings");
  if (axis == 1) {
    if (!(M0 == M1 && M1 == M_OUT && N_OUT == (N0 + N1))) fail("concat axis=1 shape mismatch");
  } else {
    if (!(N0 == N1 && N1 == N_OUT && M_OUT == (M0 + M1))) fail("concat axis=0 shape mismatch");
  }

  const int64_t total = M_OUT * N_OUT;
  const int64_t block_x = 256;
  const int64_t grid_x = (total + block_x - 1) / block_x;
  const std::string kernel_sym = std::string("intentir_concat2d_f32_axis") + std::to_string(axis);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + kernel_sym + "(");
  w.indent();
  w.line("const float* __restrict__ " + a_name + ",");
  w.line("const float* __restrict__ " + b_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line("int M0, int N0, int M1, int N1, int M_OUT, int N_OUT) {");
  w.dedent();
  w.indent();
  w.line("(void)M1;");
  w.line("(void)N1;");
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t total = (int64_t)M_OUT * (int64_t)N_OUT;");
  w.line("if (tid < 0 || tid >= total) return;");
  w.line("const int row = (int)(tid / (int64_t)N_OUT);");
  w.line("const int col = (int)(tid - (int64_t)row * (int64_t)N_OUT);");
  if (axis == 1) {
    w.line("if (col < N0) {");
    w.indent();
    w.line(out_name + "[(size_t)tid] = " + a_name + "[(size_t)row * (size_t)N0 + (size_t)col];");
    w.dedent();
    w.line("} else {");
    w.indent();
    w.line("const int c1 = col - N0;");
    w.line(out_name + "[(size_t)tid] = " + b_name + "[(size_t)row * (size_t)N1 + (size_t)c1];");
    w.dedent();
    w.line("}");
  } else {
    w.line("if (row < M0) {");
    w.indent();
    w.line(out_name + "[(size_t)tid] = " + a_name + "[(size_t)row * (size_t)N0 + (size_t)col];");
    w.dedent();
    w.line("} else {");
    w.indent();
    w.line("const int r1 = row - M0;");
    w.line(out_name + "[(size_t)tid] = " + b_name + "[(size_t)r1 * (size_t)N1 + (size_t)col];");
    w.dedent();
    w.line("}");
  }
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = kernel_sym;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{a_name, b_name, out_name},
      /*scalar_args=*/{{"M0", "i32"}, {"N0", "i32"}, {"M1", "i32"}, {"N1", "i32"}, {"M_OUT", "i32"}, {"N_OUT", "i32"}},
      /*arg_names=*/{a_name, b_name, out_name, "M0", "N0", "M1", "N1", "M_OUT", "N_OUT"});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["M0"] = M0;
  out_bindings["N0"] = N0;
  out_bindings["M1"] = M1;
  out_bindings["N1"] = N1;
  out_bindings["M_OUT"] = M_OUT;
  out_bindings["N_OUT"] = N_OUT;
  out["bindings"] = out_bindings;
  return out;
}

json emit_pad2d_const_f32(const Intent& intent, const json& bindings) {
  // MVP: one pad op, rank-2, mode=constant.
  if (intent.ops.size() != 1 || intent.ops[0].op != "pad") fail("pad lowering expects a single pad op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("pad expects one input");
  if (intent.outputs.empty()) fail("pad lowering expects one output");
  const std::string inp_name = op.inputs[0];
  const std::string out_name = intent.outputs[0];
  if (op.output != out_name) fail("pad output must match intent outputs[0]");

  auto in_it = intent.tensors.find(inp_name);
  auto out_it = intent.tensors.find(out_name);
  if (in_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("pad tensors missing in intent");
  if (in_it->second.shape.size() != 2 || out_it->second.shape.size() != 2) fail("pad MVP expects rank-2 tensors");

  if (!op.attrs.is_object()) fail("pad expects attrs object");
  std::string mode = "constant";
  if (op.attrs.contains("mode")) {
    if (!op.attrs["mode"].is_string()) fail("pad mode must be string");
    mode = ascii_lower(op.attrs["mode"].get<std::string>());
  }
  if (mode != "constant") fail("pad MVP supports mode=constant only");
  double pad_value = 0.0;
  if (op.attrs.contains("value")) {
    const auto& v = op.attrs["value"];
    if (v.is_number())
      pad_value = v.get<double>();
    else if (v.is_string()) {
      try {
        pad_value = std::stod(v.get<std::string>());
      } catch (...) {
        fail("pad constant value parse failed");
      }
    } else {
      fail("pad value must be numeric");
    }
  }

  int64_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
  if (op.attrs.contains("pad_width")) {
    const auto& pw = op.attrs["pad_width"];
    if (pw.is_object() && pw.contains("pairs") && pw["pairs"].is_array() && pw["pairs"].size() == 2) {
      const auto parse_pair = [&](const json& pair, int64_t& lo, int64_t& hi) {
        if (!pair.is_array() || pair.size() != 2) fail("pad_width pair must be [before, after]");
        if (!pair[0].is_number() || !pair[1].is_number()) fail("pad_width values must be numeric");
        lo = static_cast<int64_t>(pair[0].get<double>());
        hi = static_cast<int64_t>(pair[1].get<double>());
      };
      parse_pair(pw["pairs"][0], pad_top, pad_bottom);
      parse_pair(pw["pairs"][1], pad_left, pad_right);
    } else {
      fail("pad_width expects object {pairs:[[top,bottom],[left,right]]}");
    }
  }

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("pad missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("pad invalid dim token");
    return -1;
  };

  const int64_t M = resolve_dim(in_it->second.shape[0]);
  const int64_t N = resolve_dim(in_it->second.shape[1]);
  const int64_t M_OUT = resolve_dim(out_it->second.shape[0]);
  const int64_t N_OUT = resolve_dim(out_it->second.shape[1]);
  if (M <= 0 || N <= 0 || M_OUT <= 0 || N_OUT <= 0) fail("pad invalid shape bindings");
  if (M_OUT != (M + pad_top + pad_bottom) || N_OUT != (N + pad_left + pad_right)) fail("pad output shape mismatch");

  const int64_t total = M_OUT * N_OUT;
  const int64_t block_x = 256;
  const int64_t grid_x = (total + block_x - 1) / block_x;
  const std::string kernel_sym = "intentir_pad2d_const_f32";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  w.line("extern \"C\" __global__ void " + kernel_sym + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line("int M, int N, int M_OUT, int N_OUT) {");
  w.dedent();
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("const int64_t total = (int64_t)M_OUT * (int64_t)N_OUT;");
  w.line("if (tid < 0 || tid >= total) return;");
  w.line("const int row = (int)(tid / (int64_t)N_OUT);");
  w.line("const int col = (int)(tid - (int64_t)row * (int64_t)N_OUT);");
  w.line("const int in_row = row - " + std::to_string(pad_top) + ";");
  w.line("const int in_col = col - " + std::to_string(pad_left) + ";");
  w.line("float v = " + c_float(pad_value) + ";");
  w.line("if ((unsigned)in_row < (unsigned)M && (unsigned)in_col < (unsigned)N) {");
  w.indent();
  w.line("v = " + inp_name + "[(size_t)in_row * (size_t)N + (size_t)in_col];");
  w.dedent();
  w.line("}");
  w.line(out_name + "[(size_t)tid] = v;");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = kernel_sym;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{inp_name, out_name},
      /*scalar_args=*/{{"M", "i32"}, {"N", "i32"}, {"M_OUT", "i32"}, {"N_OUT", "i32"}},
      /*arg_names=*/{inp_name, out_name, "M", "N", "M_OUT", "N_OUT"});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["M_OUT"] = M_OUT;
  out_bindings["N_OUT"] = N_OUT;
  out["bindings"] = out_bindings;
  return out;
}

json emit_mse_loss2d_f32(const Intent& intent, const json& bindings) {
  // MVP: one mse_loss op, rank-2 input/target, scalar output.
  if (intent.ops.size() != 1 || intent.ops[0].op != "mse_loss") fail("mse_loss lowering expects a single mse_loss op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("mse_loss expects 2 inputs");
  if (intent.outputs.empty()) fail("mse_loss lowering expects one output");
  const std::string inp_name = op.inputs[0];
  const std::string tgt_name = op.inputs[1];
  const std::string out_name = intent.outputs[0];
  if (op.output != out_name) fail("mse_loss output must match intent outputs[0]");

  auto in_it = intent.tensors.find(inp_name);
  auto tgt_it = intent.tensors.find(tgt_name);
  auto out_it = intent.tensors.find(out_name);
  if (in_it == intent.tensors.end() || tgt_it == intent.tensors.end() || out_it == intent.tensors.end()) {
    fail("mse_loss tensors missing in intent");
  }
  if (in_it->second.shape.size() != 2 || tgt_it->second.shape.size() != 2) fail("mse_loss MVP expects rank-2 inputs");
  if (!out_it->second.shape.empty()) fail("mse_loss MVP expects scalar output");

  int reduction = 1;  // 1=mean, 2=sum (Torch-style enum used by provider mapping).
  if (op.attrs.is_object() && op.attrs.contains("reduction")) {
    const auto& rv = op.attrs["reduction"];
    if (rv.is_number_integer())
      reduction = static_cast<int>(rv.get<int64_t>());
    else if (rv.is_number())
      reduction = static_cast<int>(rv.get<double>());
    else if (rv.is_string()) {
      const std::string r = ascii_lower(rv.get<std::string>());
      if (r == "mean")
        reduction = 1;
      else if (r == "sum")
        reduction = 2;
      else
        fail("mse_loss MVP supports reduction mean/sum only");
    } else {
      fail("mse_loss reduction must be numeric or string");
    }
  }
  if (reduction != 1 && reduction != 2) fail("mse_loss MVP supports reduction=1(mean) or 2(sum)");

  auto resolve_dim = [&](const json& d) -> int64_t {
    if (d.is_number_integer()) return d.get<int64_t>();
    if (d.is_number()) return static_cast<int64_t>(d.get<double>());
    if (d.is_string()) {
      auto b = binding_int(bindings, d.get<std::string>());
      if (!b.has_value()) fail("mse_loss missing binding for dim: " + d.get<std::string>());
      return *b;
    }
    fail("mse_loss invalid dim token");
    return -1;
  };

  const int64_t M = resolve_dim(in_it->second.shape[0]);
  const int64_t N = resolve_dim(in_it->second.shape[1]);
  const int64_t M_t = resolve_dim(tgt_it->second.shape[0]);
  const int64_t N_t = resolve_dim(tgt_it->second.shape[1]);
  if (M <= 0 || N <= 0 || M_t <= 0 || N_t <= 0) fail("mse_loss invalid shape bindings");
  if (M != M_t || N != N_t) fail("mse_loss input/target shape mismatch");
  const int64_t total_elems = M * N;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t max_useful = 32;
  while ((max_useful << 1) <= total_elems && (max_useful << 1) <= 1024) max_useful <<= 1;
  if (block_x > max_useful) block_x = max_useful;
  const bool tiny_serial = (total_elems <= 16);
  const bool use_warp_reduce = (!tiny_serial && block_x == 32);

  const std::string kernel_sym = "intentir_mse_loss2d_f32";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stddef.h>");
  w.line("#include <stdint.h>");
  if (!tiny_serial) {
    w.line("#include <math.h>");
    w.line("#include \"intentir_cuda_ops.cuh\"");
    if (!use_warp_reduce) {
      w.line("#include \"kernels/reduce.cuh\"");
    }
    w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + kernel_sym + "(");
  } else {
    w.line("extern \"C\" __global__ void " + kernel_sym + "(");
  }
  w.indent();
  w.line("const float* __restrict__ " + inp_name + ",");
  w.line("const float* __restrict__ " + tgt_name + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line("int M, int N) {");
  w.dedent();
  w.indent();
  if (tiny_serial) {
    w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
    w.line("const int64_t total = (int64_t)M * (int64_t)N;");
    w.line("float acc = 0.0f;");
    w.line("for (int64_t i = 0; i < total; ++i) {");
    w.indent();
    w.line("const float d = " + inp_name + "[(size_t)i] - " + tgt_name + "[(size_t)i];");
    w.line("acc += d * d;");
    w.dedent();
    w.line("}");
    if (reduction == 1) {
      w.line(out_name + "[0] = (total > 0) ? (acc / (float)total) : 0.0f;");
    } else {
      w.line(out_name + "[0] = acc;");
    }
  } else {
    w.line("const int64_t total = (int64_t)M * (int64_t)N;");
    w.line("float acc = 0.0f;");
    w.line("for (int64_t i = (int64_t)threadIdx.x; i < total; i += (int64_t)blockDim.x) {");
    w.indent();
    w.line("const float x = intentir_ldg_f32(" + inp_name + " + (size_t)i);");
    w.line("const float y = intentir_ldg_f32(" + tgt_name + " + (size_t)i);");
    w.line("const float d = x - y;");
    w.line("acc = fmaf(d, d, acc);");
    w.dedent();
    w.line("}");
    if (use_warp_reduce) {
      w.line("const unsigned mask = 0xffffffffu;");
      w.line("for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down_sync(mask, acc, offset);");
      if (reduction == 1) {
        w.line("if ((int)threadIdx.x == 0) " + out_name + "[0] = (total > 0) ? (acc / (float)total) : 0.0f;");
      } else {
        w.line("if ((int)threadIdx.x == 0) " + out_name + "[0] = acc;");
      }
    } else {
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
      w.line("__shared__ intentir_cuda::BlockAllreduceF32<BLOCK_THREADS> red;");
      w.line("const float sum_sq = intentir_cuda::block_allreduce_sum<BLOCK_THREADS>(acc, &red);");
      if (reduction == 1) {
        w.line("if ((int)threadIdx.x == 0) " + out_name + "[0] = (total > 0) ? (sum_sq / (float)total) : 0.0f;");
      } else {
        w.line("if ((int)threadIdx.x == 0) " + out_name + "[0] = sum_sq;");
      }
    }
  }
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = kernel_sym;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{inp_name, tgt_name, out_name},
      /*scalar_args=*/{{"M", "i32"}, {"N", "i32"}},
      /*arg_names=*/{inp_name, tgt_name, out_name, "M", "N"});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {tiny_serial ? 1 : block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out["bindings"] = out_bindings;
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
