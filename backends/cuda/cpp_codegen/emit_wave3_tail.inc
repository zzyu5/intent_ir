json emit_allclose2d_f32_bool(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 9) fail("allclose lowering expects 9-op pattern");
  static const std::vector<std::string> kOps = {"sub", "abs", "abs", "mul", "add", "", "not", "reduce_any", "not"};
  for (size_t i = 0; i < kOps.size(); ++i) {
    if (i == 5) {
      if (intent.ops[i].op != "le" && intent.ops[i].op != "ge") fail("allclose lowering pattern mismatch");
      continue;
    }
    if (intent.ops[i].op != kOps[i]) fail("allclose lowering pattern mismatch");
  }

  const std::string A = intent.ops[0].inputs[0];
  const std::string B = intent.ops[0].inputs[1];
  const std::string rtol = intent.ops[3].inputs[0] == "abs_b" ? intent.ops[3].inputs[1] : intent.ops[3].inputs[0];
  const std::string atol = intent.ops[4].inputs[0] == intent.ops[3].output ? intent.ops[4].inputs[1] : intent.ops[4].inputs[0];
  const std::string OUT = intent.ops.back().output;

  auto a_it = intent.tensors.find(A);
  auto b_it = intent.tensors.find(B);
  auto r_it = intent.tensors.find(rtol);
  auto t_it = intent.tensors.find(atol);
  auto o_it = intent.tensors.find(OUT);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || r_it == intent.tensors.end() || t_it == intent.tensors.end() ||
      o_it == intent.tensors.end()) {
    fail("allclose lowering missing tensors");
  }
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || r_it->second.dtype != "f32" || t_it->second.dtype != "f32") {
    fail("allclose lowering supports f32 A/B/rtol/atol");
  }
  if (o_it->second.dtype != "bool") fail("allclose output must be bool");
  auto a_shape = resolve_shape_required(a_it->second, bindings, "allclose.A");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "allclose.B");
  if (a_shape.size() != 2 || b_shape.size() != 2) fail("allclose expects rank-2 A/B");
  if (a_shape != b_shape) fail("allclose shape mismatch");
  const int64_t M = a_shape[0];
  const int64_t N = a_shape[1];
  if (M <= 0 || N <= 0) fail("allclose invalid shape");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + A + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("const float* __restrict__ " + rtol + ",");
  w.line("const float* __restrict__ " + atol + ",");
  w.line("bool* __restrict__ " + OUT);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("const float rtol_v = " + rtol + " ? " + rtol + "[0] : 1e-5f;");
  w.line("const float atol_v = " + atol + " ? " + atol + "[0] : 1e-8f;");
  w.line("bool all_close = true;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) {");
  w.indent();
  w.line("const float a = " + A + "[i];");
  w.line("const float b = " + B + "[i];");
  w.line("const float tol = atol_v + rtol_v * fabsf(b);");
  w.line("if (fabsf(a - b) > tol) { all_close = false; break; }");
  w.dedent();
  w.line("}");
  w.line(OUT + "[0] = all_close;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {A, B, rtol, atol, OUT}, {}, {A, B, rtol, atol, OUT});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {OUT};
  out["bindings"] = out_bindings;
  return out;
}

json emit_batch_norm2d_training_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 25) fail("batch_norm lowering expects 25-op expanded pattern");
  const std::string input = "input";
  const std::string weight = "weight";
  const std::string bias = "bias";
  const std::string running_mean = "running_mean";
  const std::string running_var = "running_var";
  const std::string eps = "eps";
  const std::string momentum = "momentum";
  const std::string n_elements = "n_elements";
  const std::string n_minus_1 = "n_minus_1";
  const std::string output = "output_1";
  const std::string mean = "mean";
  const std::string inv_std = "inv_std";
  const std::string running_mean_out = "running_mean_out";
  const std::string running_var_out = "running_var_out";

  auto in_it = intent.tensors.find(input);
  auto w_it = intent.tensors.find(weight);
  auto b_it = intent.tensors.find(bias);
  auto rm_it = intent.tensors.find(running_mean);
  auto rv_it = intent.tensors.find(running_var);
  auto out_it = intent.tensors.find(output);
  auto mean_it = intent.tensors.find(mean);
  auto inv_it = intent.tensors.find(inv_std);
  auto rmo_it = intent.tensors.find(running_mean_out);
  auto rvo_it = intent.tensors.find(running_var_out);
  if (in_it == intent.tensors.end() || w_it == intent.tensors.end() || b_it == intent.tensors.end() || rm_it == intent.tensors.end() ||
      rv_it == intent.tensors.end() || out_it == intent.tensors.end() || mean_it == intent.tensors.end() || inv_it == intent.tensors.end() ||
      rmo_it == intent.tensors.end() || rvo_it == intent.tensors.end()) {
    fail("batch_norm lowering missing tensors");
  }
  for (const auto* t : {&in_it->second, &w_it->second, &b_it->second, &rm_it->second, &rv_it->second, &out_it->second, &mean_it->second, &inv_it->second,
                        &rmo_it->second, &rvo_it->second}) {
    if (t->dtype != "f32") fail("batch_norm lowering supports f32 only");
  }

  auto in_shape = resolve_shape_required(in_it->second, bindings, "batch_norm.input");
  if (in_shape.size() != 3) fail("batch_norm expects input rank-3 [N,C,HW]");
  const int64_t N = in_shape[0];
  const int64_t C = in_shape[1];
  const int64_t HW = in_shape[2];
  if (N <= 0 || C <= 0 || HW <= 0) fail("batch_norm invalid shape");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + input + ",");
  w.line("const float* __restrict__ " + weight + ",");
  w.line("const float* __restrict__ " + bias + ",");
  w.line("const float* __restrict__ " + running_mean + ",");
  w.line("const float* __restrict__ " + running_var + ",");
  w.line("const float* __restrict__ " + eps + ",");
  w.line("const float* __restrict__ " + momentum + ",");
  w.line("const float* __restrict__ " + n_elements + ",");
  w.line("const float* __restrict__ " + n_minus_1 + ",");
  w.line("float* __restrict__ " + output + ",");
  w.line("float* __restrict__ " + mean + ",");
  w.line("float* __restrict__ " + inv_std + ",");
  w.line("float* __restrict__ " + running_mean_out + ",");
  w.line("float* __restrict__ " + running_var_out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("const float eps_v = " + eps + " ? " + eps + "[0] : 1e-5f;");
  w.line("const float momentum_v = " + momentum + " ? " + momentum + "[0] : 0.1f;");
  w.line("float n_elems = " + n_elements + " ? " + n_elements + "[0] : (float)(" + std::to_string(N) + " * " + std::to_string(HW) + ");");
  w.line("if (n_elems <= 0.0f) n_elems = (float)(" + std::to_string(N) + " * " + std::to_string(HW) + ");");
  w.line("float n_minus = " + n_minus_1 + " ? " + n_minus_1 + "[0] : (n_elems - 1.0f);");
  w.line("if (n_minus <= 0.0f) n_minus = 1.0f;");
  w.line("const float one_minus_momentum = 1.0f - momentum_v;");
  w.line("for (int64_t c = 0; c < " + std::to_string(C) + "LL; ++c) {");
  w.indent();
  w.line("float sum = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) sum += " + input + "[base + hw];");
  w.dedent();
  w.line("}");
  w.line("const float mean_c = sum / n_elems;");
  w.line(mean + "[c] = mean_c;");
  w.line("float var_sum = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) {");
  w.indent();
  w.line("const float diff = " + input + "[base + hw] - mean_c;");
  w.line("var_sum = fmaf(diff, diff, var_sum);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("const float var_c = var_sum / n_elems;");
  w.line("const float inv_c = rsqrtf(var_c + eps_v);");
  w.line(inv_std + "[c] = inv_c;");
  w.line("const float wv = " + weight + "[c];");
  w.line("const float bv = " + bias + "[c];");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) {");
  w.indent();
  w.line("const float centered = " + input + "[base + hw] - mean_c;");
  w.line(output + "[base + hw] = centered * inv_c * wv + bv;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(running_mean_out + "[c] = one_minus_momentum * " + running_mean + "[c] + momentum_v * mean_c;");
  w.line("const float unbiased_var = var_c * (n_elems / n_minus);");
  w.line(running_var_out + "[c] = one_minus_momentum * " + running_var + "[c] + momentum_v * unbiased_var;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  out_bindings["C"] = C;
  out_bindings["HW"] = HW;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent,
                                     {input, weight, bias, running_mean, running_var, eps, momentum, n_elements, n_minus_1, output, mean, inv_std,
                                      running_mean_out, running_var_out},
                                     {},
                                     {input, weight, bias, running_mean, running_var, eps, momentum, n_elements, n_minus_1, output, mean, inv_std,
                                      running_mean_out, running_var_out});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {output, mean, inv_std, running_mean_out, running_var_out};
  out["bindings"] = out_bindings;
  return out;
}

#include "emit_attention_index_norm.inc"
json emit_reduce_prod_2d_all_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "reduce_prod")) fail("reduce_prod(all) lowering expects single reduce_prod");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("reduce_prod(all) expects 1 input");
  if (!_reduce_axis_is_all_2d(op.attrs)) fail("reduce_prod(all) expects dims=[0,1]");
  const std::string in_name = op.inputs[0];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(in_name);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("reduce_prod(all) missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("reduce_prod(all) supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "reduce_prod.all.in");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "reduce_prod.all.out");
  if (i_shape.size() != 2 || !o_shape.empty()) fail("reduce_prod(all) expects rank-2 input and scalar output");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + in_name + ", float* __restrict__ " + out_name + ") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("float p = 1.0f;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) p *= " + in_name + "[i];");
  w.line(out_name + "[0] = p;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_reduce_prod_2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "reduce_prod")) fail("reduce_prod(axis1) lowering expects single reduce_prod");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("reduce_prod(axis1) expects 1 input");
  if (!_reduce_axis_is_1(op.attrs)) fail("reduce_prod(axis1) expects axis=1");
  const std::string in_name = op.inputs[0];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(in_name);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("reduce_prod(axis1) missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("reduce_prod(axis1) supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "reduce_prod.axis1.in");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "reduce_prod.axis1.out");
  if (i_shape.size() != 2) fail("reduce_prod(axis1) expects rank-2 input");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  if (!((o_shape.size() == 1 && o_shape[0] == M) || (o_shape.size() == 2 && o_shape[0] == M && o_shape[1] == 1))) {
    fail("reduce_prod(axis1) expects output shape [M] or [M,1]");
  }
  const bool keepdim = (o_shape.size() == 2);

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + in_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float p = 1.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) p *= " + in_name + "[base + n];");
  if (keepdim) {
    w.line(out_name + "[row] = p;");
  } else {
    w.line(out_name + "[row] = p;");
  }
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_quantile2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "quantile")) fail("quantile lowering expects single quantile op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("quantile expects inp,q");
  int dim = op.attrs.value("dim", 1);
  if (dim == -1) dim = 1;
  if (dim != 1) fail("quantile lowering currently supports dim=1 only");
  const std::string interpolation = op.attrs.value("interpolation", std::string("linear"));
  if (!(interpolation == "linear" || interpolation == "lower" || interpolation == "higher" || interpolation == "nearest" ||
        interpolation == "midpoint")) {
    fail("quantile interpolation unsupported: " + interpolation);
  }
  const std::string inp = op.inputs[0];
  const std::string q = op.inputs[1];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto q_it = intent.tensors.find(q);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || q_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("quantile missing tensors");
  if (i_it->second.dtype != "f32" || q_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("quantile supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "quantile.inp");
  auto q_shape = resolve_shape_required(q_it->second, bindings, "quantile.q");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "quantile.out");
  if (i_shape.size() != 2 || !q_shape.empty()) fail("quantile expects inp:[M,N], q:[]");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const bool keepdim = op.attrs.value("keepdim", false);
  if (keepdim) {
    if (!(o_shape.size() == 2 && o_shape[0] == M && o_shape[1] == 1)) fail("quantile keepdim expects out shape [M,1]");
  } else {
    if (!(o_shape.size() == 1 && o_shape[0] == M)) fail("quantile expects out shape [M]");
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + q + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("float qv = " + q + " ? " + q + "[0] : 0.5f;");
  w.line("if (qv < 0.0f) qv = 0.0f;");
  w.line("if (qv > 1.0f) qv = 1.0f;");
  w.line("const float h = qv * (float)(" + std::to_string(N - 1) + ");");
  w.line("int64_t k0 = (int64_t)floorf(h);");
  w.line("int64_t k1 = (int64_t)ceilf(h);");
  w.line("if (k0 < 0) k0 = 0;");
  w.line("if (k1 < 0) k1 = 0;");
  w.line("if (k0 >= " + std::to_string(N) + "LL) k0 = " + std::to_string(N - 1) + "LL;");
  w.line("if (k1 >= " + std::to_string(N) + "LL) k1 = " + std::to_string(N - 1) + "LL;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float v0 = 0.0f;");
  w.line("float v1 = 0.0f;");
  w.line("bool f0 = false;");
  w.line("bool f1 = false;");
  w.line("for (int64_t i = 0; i < " + std::to_string(N) + "LL; ++i) {");
  w.indent();
  w.line("const float vi = " + inp + "[base + i];");
  w.line("int64_t rank = 0;");
  w.line("for (int64_t j = 0; j < " + std::to_string(N) + "LL; ++j) {");
  w.indent();
  w.line("const float vj = " + inp + "[base + j];");
  w.line("if (vj < vi || (vj == vi && j < i)) ++rank;");
  w.dedent();
  w.line("}");
  w.line("if (!f0 && rank == k0) { v0 = vi; f0 = true; }");
  w.line("if (!f1 && rank == k1) { v1 = vi; f1 = true; }");
  w.line("if (f0 && f1) break;");
  w.dedent();
  w.line("}");
  w.line("if (!f0) v0 = " + inp + "[base];");
  w.line("if (!f1) v1 = " + inp + "[base];");
  w.line("const float frac = h - floorf(h);");
  std::string q_expr;
  if (interpolation == "lower") {
    q_expr = "v0";
  } else if (interpolation == "higher") {
    q_expr = "v1";
  } else if (interpolation == "nearest") {
    q_expr = "(frac < 0.5f ? v0 : v1)";
  } else if (interpolation == "midpoint") {
    q_expr = "(0.5f * (v0 + v1))";
  } else {
    q_expr = "(v0 + (v1 - v0) * frac)";
  }
  w.line("const float q_out = " + q_expr + ";");
  w.line(out_name + "[row] = q_out;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, q, out_name}, {}, {inp, q, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_rms_norm2d_f32(const Intent& intent, const json& bindings) {
  if (intent.outputs.size() != 2) fail("rms_norm lowering expects two outputs [out, INV_RMS]");
  const std::string out_name = intent.outputs[0];
  const std::string inv_name = intent.outputs[1];
  const std::string input = "input";
  const std::string weight = "weight";
  auto in_it = intent.tensors.find(input);
  auto w_it = intent.tensors.find(weight);
  auto o_it = intent.tensors.find(out_name);
  auto r_it = intent.tensors.find(inv_name);
  if (in_it == intent.tensors.end() || w_it == intent.tensors.end() || o_it == intent.tensors.end() || r_it == intent.tensors.end()) {
    fail("rms_norm lowering missing input/weight/out/inv tensors");
  }
  if (in_it->second.dtype != "f32" || w_it->second.dtype != "f32" || o_it->second.dtype != "f32" || r_it->second.dtype != "f32") {
    fail("rms_norm lowering supports f32 only");
  }
  auto in_shape = resolve_shape_required(in_it->second, bindings, "rms_norm.input");
  auto w_shape = resolve_shape_required(w_it->second, bindings, "rms_norm.weight");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "rms_norm.out");
  auto r_shape = resolve_shape_required(r_it->second, bindings, "rms_norm.inv");
  if (in_shape.size() != 2 || w_shape.size() != 1 || o_shape.size() != 2 || r_shape.size() != 1) {
    fail("rms_norm expects input:[M,N], weight:[N], out:[M,N], INV_RMS:[M]");
  }
  const int64_t M = in_shape[0];
  const int64_t N = in_shape[1];
  if (w_shape[0] != N || o_shape[0] != M || o_shape[1] != N || r_shape[0] != M) fail("rms_norm shape mismatch");
  float eps_v = 1e-5f;
  for (const auto& op : intent.ops) {
    if (op.op == "const" && op.output == "eps" && op.attrs.is_object() && op.attrs.contains("value")) {
      const auto& v = op.attrs["value"];
      if (v.is_number()) eps_v = static_cast<float>(v.get<double>());
      break;
    }
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + input + ",");
  w.line("const float* __restrict__ " + weight + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line("float* __restrict__ " + inv_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum_sq = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float x = " + input + "[base + n];");
  w.line("sum_sq = fmaf(x, x, sum_sq);");
  w.dedent();
  w.line("}");
  w.line("const float rrms = rsqrtf(sum_sq / (float)" + std::to_string(N) + " + " + std::to_string(eps_v) + "f);");
  w.line(inv_name + "[row] = rrms;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line(out_name + "[base + n] = " + input + "[base + n] * rrms * " + weight + "[n];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {input, weight, out_name, inv_name}, {}, {input, weight, out_name, inv_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name, inv_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_stack_axis0_2x2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "stack")) fail("stack lowering expects single stack op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("stack currently supports exactly 2 inputs");
  int axis = op.attrs.value("axis", 0);
  if (axis < 0) axis += 3;
  if (axis != 0) fail("stack lowering currently supports axis=0 only");
  const std::string a = op.inputs[0];
  const std::string b = op.inputs[1];
  const std::string out_name = op.output;
  auto a_it = intent.tensors.find(a);
  auto b_it = intent.tensors.find(b);
  auto o_it = intent.tensors.find(out_name);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("stack missing tensors");
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("stack supports f32 only");
  auto a_shape = resolve_shape_required(a_it->second, bindings, "stack.a");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "stack.b");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "stack.out");
  if (a_shape.size() != 2 || b_shape.size() != 2 || a_shape != b_shape) fail("stack expects matching rank-2 inputs");
  if (o_shape.size() != 3 || o_shape[0] != 2 || o_shape[1] != a_shape[0] || o_shape[2] != a_shape[1]) {
    fail("stack output shape mismatch: expect [2,M,N]");
  }
  const int64_t M = a_shape[0];
  const int64_t N = a_shape[1];
  const int64_t MN = M * N;
  const int64_t total = 2 * MN;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + a + ",");
  w.line("const float* __restrict__ " + b + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("if (tid < " + std::to_string(MN) + "LL) {");
  w.indent();
  w.line(out_name + "[tid] = " + a + "[tid];");
  w.dedent();
  w.line("} else {");
  w.indent();
  w.line("const int64_t i = tid - " + std::to_string(MN) + "LL;");
  w.line(out_name + "[tid] = " + b + "[i];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {a, b, out_name}, {}, {a, b, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_scatter2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "scatter")) fail("scatter lowering expects single scatter op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("scatter expects inp,index,src");
  int dim = op.attrs.value("dim", 1);
  if (dim < 0) dim += 2;
  if (dim != 1) fail("scatter currently supports dim=1 only");
  const std::string inp = op.inputs[0];
  const std::string index = op.inputs[1];
  const std::string src = op.inputs[2];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto x_it = intent.tensors.find(index);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || x_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("scatter missing tensors");
  }
  if (i_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("scatter supports f32 inp/src/out");
  const std::string idx_dt = x_it->second.dtype;
  if (!(idx_dt == "i32" || idx_dt == "i64")) fail("scatter index dtype must be i32/i64");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "scatter.inp");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "scatter.index");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "scatter.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "scatter.out");
  if (i_shape.size() != 2 || x_shape.size() != 2 || s_shape.size() != 2 || o_shape.size() != 2) fail("scatter expects rank-2 tensors");
  if (i_shape != x_shape || i_shape != s_shape || i_shape != o_shape) fail("scatter shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const std::string idx_ctype = (idx_dt == "i32") ? "int" : "int64_t";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const " + idx_ctype + "* __restrict__ " + index + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out_name + "[i] = " + inp + "[i];");
  w.line("for (int64_t m = 0; m < " + std::to_string(M) + "LL; ++m) {");
  w.indent();
  w.line("const int64_t row = m * " + std::to_string(N) + "LL;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t col = (int64_t)" + index + "[row + n];");
  w.line("if (col < 0 || col >= " + std::to_string(N) + "LL) continue;");
  w.line(out_name + "[row + col] = " + src + "[row + n];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, index, src, out_name}, {}, {inp, index, src, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_select_scatter2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "select_scatter")) fail("select_scatter lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("select_scatter expects inp,src");
  int dim = op.attrs.value("dim", 1);
  if (dim < 0) dim += 2;
  if (dim != 1) fail("select_scatter currently supports dim=1 only");
  int64_t index = op.attrs.value("index", static_cast<int64_t>(0));
  const std::string inp = op.inputs[0];
  const std::string src = op.inputs[1];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("select_scatter missing tensors");
  if (i_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("select_scatter supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "select_scatter.inp");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "select_scatter.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "select_scatter.out");
  if (i_shape.size() != 2 || s_shape.size() != 1 || o_shape.size() != 2) fail("select_scatter expects inp/out rank-2 and src rank-1");
  if (i_shape != o_shape || s_shape[0] != i_shape[0]) fail("select_scatter shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  if (index < 0) index += N;
  if (index < 0 || index >= N) fail("select_scatter index out of range");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out_name + "[i] = " + inp + "[i];");
  w.line("for (int64_t m = 0; m < " + std::to_string(M) + "LL; ++m) {");
  w.indent();
  w.line(out_name + "[m * " + std::to_string(N) + "LL + " + std::to_string(index) + "LL] = " + src + "[m];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["index"] = index;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, src, out_name}, {}, {inp, src, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_slice_scatter2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "slice_scatter")) fail("slice_scatter lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("slice_scatter expects inp,src");
  int dim = op.attrs.value("dim", 1);
  if (dim < 0) dim += 2;
  if (dim != 1) fail("slice_scatter currently supports dim=1 only");
  int64_t start = op.attrs.value("start", static_cast<int64_t>(0));
  int64_t end = op.attrs.value("end", static_cast<int64_t>(0));
  int64_t step = op.attrs.value("step", static_cast<int64_t>(1));
  if (step <= 0) fail("slice_scatter requires step > 0");
  const std::string inp = op.inputs[0];
  const std::string src = op.inputs[1];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("slice_scatter missing tensors");
  if (i_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("slice_scatter supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "slice_scatter.inp");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "slice_scatter.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "slice_scatter.out");
  if (i_shape.size() != 2 || s_shape.size() != 2 || o_shape.size() != 2) fail("slice_scatter expects rank-2 tensors");
  if (i_shape != o_shape || s_shape[0] != i_shape[0]) fail("slice_scatter shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  if (start < 0) start += N;
  if (end < 0) end += N;
  if (start < 0) start = 0;
  if (end > N) end = N;
  if (end < start) end = start;
  const int64_t expected_L = (end - start + step - 1) / step;
  if (s_shape[1] != expected_L) fail("slice_scatter src second dim mismatch with (start,end,step)");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out_name + "[i] = " + inp + "[i];");
  w.line("for (int64_t m = 0; m < " + std::to_string(M) + "LL; ++m) {");
  w.indent();
  w.line("int64_t l = 0;");
  w.line("for (int64_t n = " + std::to_string(start) + "LL; n < " + std::to_string(end) + "LL; n += " + std::to_string(step) + "LL) {");
  w.indent();
  w.line(out_name + "[m * " + std::to_string(N) + "LL + n] = " + src + "[m * " + std::to_string(expected_L) + "LL + l];");
  w.line("++l;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["start"] = start;
  out_bindings["end"] = end;
  out_bindings["step"] = step;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, src, out_name}, {}, {inp, src, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_var_mean2d_f32(const Intent& intent, const json& bindings) {
  const bool std_mul_pattern = (intent.ops.size() == 2 && intent.ops[0].op == "std" && intent.ops[1].op == "mul");
  const bool expanded_pattern =
      (intent.ops.size() == 9 && intent.ops[0].op == "const" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "broadcast_in_dim" &&
       intent.ops[3].op == "sub" && intent.ops[4].op == "mul" && intent.ops[5].op == "reduce_sum" && intent.ops[6].op == "sub" &&
       intent.ops[7].op == "div" && intent.ops[8].op == "identity");
  if (!std_mul_pattern && !expanded_pattern) {
    fail("var_mean lowering expects std->mul or expanded reduce_sum->...->div->identity pattern");
  }

  std::string inp;
  std::string out_var;
  std::string out_mean;
  bool has_mean_output = false;
  int64_t correction = 1;
  std::string correction_tensor;
  std::string correction_dtype;

  if (std_mul_pattern) {
    const Op& std_op = intent.ops[0];
    const Op& mul_op = intent.ops[1];
    if (std_op.inputs.size() != 1 || mul_op.inputs.size() != 2) fail("var_mean arity mismatch");
    if (!((mul_op.inputs[0] == std_op.output && mul_op.inputs[1] == std_op.output))) fail("var_mean expects mul(std_out,std_out)");
    if (!_reduce_axis_is_1(std_op.attrs)) fail("var_mean currently supports axis=1 only");
    inp = std_op.inputs[0];
    out_var = mul_op.output;
    correction = std_op.attrs.value("correction", static_cast<int64_t>(1));
  } else {
    const Op& red_mean = intent.ops[1];
    const Op& bcast_mean = intent.ops[2];
    const Op& sub_center = intent.ops[3];
    const Op& mul_sq = intent.ops[4];
    const Op& red_var = intent.ops[5];
    const Op& sub_denom = intent.ops[6];
    const Op& div_var = intent.ops[7];
    const Op& id_mean = intent.ops[8];
    if (red_mean.inputs.size() != 1) fail("var_mean expanded: reduce_sum(mean) expects 1 input");
    if (!_reduce_axis_is_1(red_mean.attrs)) fail("var_mean expanded: reduce_sum(mean) expects axis=1");
    inp = red_mean.inputs[0];
    if (bcast_mean.inputs.size() != 1 || bcast_mean.inputs[0] != red_mean.output) fail("var_mean expanded: broadcast wiring mismatch");
    if (sub_center.inputs.size() != 2 || sub_center.inputs[0] != inp || sub_center.inputs[1] != bcast_mean.output) {
      fail("var_mean expanded: center-sub wiring mismatch");
    }
    if (mul_sq.inputs.size() != 2 || mul_sq.inputs[0] != sub_center.output || mul_sq.inputs[1] != sub_center.output) {
      fail("var_mean expanded: squared-diff wiring mismatch");
    }
    if (red_var.inputs.size() != 1 || red_var.inputs[0] != mul_sq.output || !_reduce_axis_is_1(red_var.attrs)) {
      fail("var_mean expanded: reduce_sum(var) wiring/axis mismatch");
    }
    if (sub_denom.inputs.size() != 2 || sub_denom.inputs[0] != intent.ops[0].output) {
      fail("var_mean expanded: denominator sub wiring mismatch");
    }
    if (div_var.inputs.size() != 2 || div_var.inputs[0] != red_var.output || div_var.inputs[1] != sub_denom.output) {
      fail("var_mean expanded: div wiring mismatch");
    }
    if (id_mean.inputs.size() != 1 || id_mean.inputs[0] != red_mean.output) fail("var_mean expanded: mean identity wiring mismatch");

    out_var = div_var.output;
    out_mean = id_mean.output;
    has_mean_output = true;

    correction_tensor = sub_denom.inputs[1];
    auto c_it = intent.tensors.find(correction_tensor);
    if (c_it != intent.tensors.end()) {
      correction_dtype = c_it->second.dtype;
    }
  }

  auto i_it = intent.tensors.find(inp);
  auto v_it = intent.tensors.find(out_var);
  if (i_it == intent.tensors.end() || v_it == intent.tensors.end()) fail("var_mean missing tensors");
  if (i_it->second.dtype != "f32" || v_it->second.dtype != "f32") fail("var_mean supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "var_mean.inp");
  auto v_shape = resolve_shape_required(v_it->second, bindings, "var_mean.var_out");
  if (i_shape.size() != 2 || v_shape.size() != 1 || v_shape[0] != i_shape[0]) fail("var_mean shape mismatch for variance output");
  if (has_mean_output) {
    auto m_it = intent.tensors.find(out_mean);
    if (m_it == intent.tensors.end()) fail("var_mean mean output tensor missing");
    if (m_it->second.dtype != "f32") fail("var_mean mean output supports f32 only");
    auto m_shape = resolve_shape_required(m_it->second, bindings, "var_mean.mean_out");
    if (m_shape.size() != 1 || m_shape[0] != i_shape[0]) fail("var_mean mean output shape mismatch");
  }
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const float static_denom = (N - correction > 0) ? static_cast<float>(N - correction) : 1.0f;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  if (has_mean_output) {
    w.line("float* __restrict__ " + out_var + ",");
    const bool has_corr_arg = (!correction_tensor.empty() && (correction_dtype == "i32" || correction_dtype == "f32"));
    if (has_corr_arg) {
      w.line("float* __restrict__ " + out_mean + ",");
      if (correction_dtype == "i32") {
        w.line("const int* __restrict__ " + correction_tensor);
      } else {
        w.line("const float* __restrict__ " + correction_tensor);
      }
    } else {
      w.line("float* __restrict__ " + out_mean);
    }
  } else {
    // Keep signature comma-safe for std->mul variant that only writes variance.
    w.line("float* __restrict__ " + out_var);
  }
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float mean = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) mean += " + inp + "[base + n];");
  w.line("mean /= (float)" + std::to_string(N) + ";");
  w.line("float ss = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float d = " + inp + "[base + n] - mean;");
  w.line("ss = fmaf(d, d, ss);");
  w.dedent();
  w.line("}");
  if (has_mean_output) {
    if (!correction_tensor.empty() && correction_dtype == "i32") {
      w.line("const float corr = " + correction_tensor + " ? (float)" + correction_tensor + "[0] : 1.0f;");
      w.line("const float denom = fmaxf((float)" + std::to_string(N) + " - corr, 1.0f);");
    } else if (!correction_tensor.empty() && correction_dtype == "f32") {
      w.line("const float corr = " + correction_tensor + " ? " + correction_tensor + "[0] : 1.0f;");
      w.line("const float denom = fmaxf((float)" + std::to_string(N) + " - corr, 1.0f);");
    } else {
      w.line("const float denom = " + std::to_string(static_denom) + "f;");
    }
    w.line(out_var + "[row] = ss / denom;");
    w.line(out_mean + "[row] = mean;");
  } else {
    w.line(out_var + "[row] = ss / " + std::to_string(static_denom) + "f;");
  }
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["correction"] = correction;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  if (has_mean_output) {
    std::vector<std::string> tensor_args = {inp, out_var, out_mean};
    std::vector<std::string> arg_names = {inp, out_var, out_mean};
    if (!correction_tensor.empty() && (correction_dtype == "i32" || correction_dtype == "f32")) {
      tensor_args.push_back(correction_tensor);
      arg_names.push_back(correction_tensor);
    }
    out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, arg_names);
    out["output_names"] = {out_var, out_mean};
  } else {
    out["io_spec"] = io_spec_from_args(intent, {inp, out_var}, {}, {inp, out_var});
    out["output_names"] = {out_var};
  }
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["bindings"] = out_bindings;
  return out;
}

json emit_vector_norm2d_f32(const Intent& intent, const json& bindings) {
  bool classic = (intent.ops.size() == 3 && intent.ops[0].op == "mul" && intent.ops[1].op == "reduce_sum" && intent.ops[2].op == "sqrt");
  bool expanded = (intent.ops.size() == 6 && intent.ops[0].op == "abs" && intent.ops[1].op == "mul" && intent.ops[2].op == "reduce_sum" &&
                   intent.ops[3].op == "rsqrt" && intent.ops[4].op == "const" && intent.ops[5].op == "div");
  if (!classic && !expanded) fail("vector_norm lowering expects classic or expanded pattern");
  const Op* mul = nullptr;
  const Op* red = nullptr;
  std::string inp;
  std::string out_name;
  if (classic) {
    const Op& mul_op = intent.ops[0];
    const Op& red_op = intent.ops[1];
    const Op& sqr = intent.ops[2];
    if (mul_op.inputs.size() != 2 || red_op.inputs.size() != 1 || sqr.inputs.size() != 1) fail("vector_norm arity mismatch");
    if (!(mul_op.inputs[0] == mul_op.inputs[1])) fail("vector_norm expects mul(inp,inp)");
    if (red_op.inputs[0] != mul_op.output || sqr.inputs[0] != red_op.output) fail("vector_norm wiring mismatch");
    mul = &mul_op;
    red = &red_op;
    inp = mul_op.inputs[0];
    out_name = sqr.output;
  } else {
    const Op& abs_op = intent.ops[0];
    const Op& mul_op = intent.ops[1];
    const Op& red_op = intent.ops[2];
    const Op& rsqrt_op = intent.ops[3];
    const Op& one_op = intent.ops[4];
    const Op& div_op = intent.ops[5];
    if (abs_op.inputs.size() != 1 || mul_op.inputs.size() != 2 || red_op.inputs.size() != 1 || rsqrt_op.inputs.size() != 1 || div_op.inputs.size() != 2) {
      fail("vector_norm expanded arity mismatch");
    }
    if (!(mul_op.inputs[0] == abs_op.output && mul_op.inputs[1] == abs_op.output)) fail("vector_norm expanded expects mul(abs(inp),abs(inp))");
    if (red_op.inputs[0] != mul_op.output || rsqrt_op.inputs[0] != red_op.output) fail("vector_norm expanded wiring mismatch");
    bool div_ok = ((div_op.inputs[0] == one_op.output && div_op.inputs[1] == rsqrt_op.output) ||
                   (div_op.inputs[1] == one_op.output && div_op.inputs[0] == rsqrt_op.output));
    if (!div_ok) fail("vector_norm expanded expects div(1, rsqrt(sum))");
    mul = &mul_op;
    red = &red_op;
    inp = abs_op.inputs[0];
    out_name = div_op.output;
  }
  if (mul == nullptr || red == nullptr) fail("vector_norm internal lowering state invalid");
  if (!_reduce_axis_is_1(red->attrs)) fail("vector_norm currently supports axis=1 only");
  auto i_it = intent.tensors.find(inp);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("vector_norm missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("vector_norm supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "vector_norm.inp");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "vector_norm.out");
  bool out_ok = false;
  if (i_shape.size() == 2 && o_shape.size() == 1 && o_shape[0] == i_shape[0]) out_ok = true;
  if (i_shape.size() == 2 && o_shape.size() == 2 && o_shape[0] == i_shape[0] && o_shape[1] == 1) out_ok = true;
  if (!out_ok) fail("vector_norm shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum_sq = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float x = " + inp + "[base + n];");
  w.line("sum_sq = fmaf(x, x, sum_sq);");
  w.dedent();
  w.line("}");
  w.line(out_name + "[row] = sqrtf(sum_sq);");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, out_name}, {}, {inp, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_weight_norm2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 6 || intent.ops[0].op != "mul" || intent.ops[1].op != "reduce_sum" || intent.ops[2].op != "sqrt" ||
      intent.ops[3].op != "div" || intent.ops[4].op != "broadcast_in_dim" || intent.ops[5].op != "mul") {
    fail("weight_norm lowering expects mul->reduce_sum->sqrt->div->broadcast->mul");
  }
  const Op& vv = intent.ops[0];
  const Op& red = intent.ops[1];
  const Op& sqr = intent.ops[2];
  const Op& div = intent.ops[3];
  const Op& bcast = intent.ops[4];
  const Op& mul = intent.ops[5];
  if (vv.inputs.size() != 2 || red.inputs.size() != 1 || sqr.inputs.size() != 1 || div.inputs.size() != 2 || bcast.inputs.size() != 1 ||
      mul.inputs.size() != 2) {
    fail("weight_norm arity mismatch");
  }
  if (!(vv.inputs[0] == vv.inputs[1])) fail("weight_norm expects vv = mul(v,v)");
  if (red.inputs[0] != vv.output || sqr.inputs[0] != red.output) fail("weight_norm reduce/sqrt wiring mismatch");
  if (!_reduce_axis_is_1(red.attrs)) fail("weight_norm currently supports axis=1 only");
  const std::string v = vv.inputs[0];
  const std::string g = (div.inputs[0] == sqr.output) ? div.inputs[1] : ((div.inputs[1] == sqr.output) ? div.inputs[0] : std::string());
  if (g.empty()) fail("weight_norm expects div(g,norm)");
  if (bcast.inputs[0] != div.output) fail("weight_norm expects broadcast(scale)");
  if (!((mul.inputs[0] == v && mul.inputs[1] == bcast.output) || (mul.inputs[1] == v && mul.inputs[0] == bcast.output))) {
    fail("weight_norm expects mul(v, scale_bc)");
  }
  const std::string out_name = mul.output;
  auto v_it = intent.tensors.find(v);
  auto g_it = intent.tensors.find(g);
  auto o_it = intent.tensors.find(out_name);
  if (v_it == intent.tensors.end() || g_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("weight_norm missing tensors");
  if (v_it->second.dtype != "f32" || g_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("weight_norm supports f32 only");
  auto v_shape = resolve_shape_required(v_it->second, bindings, "weight_norm.v");
  auto g_shape = resolve_shape_required(g_it->second, bindings, "weight_norm.g");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "weight_norm.out");
  if (v_shape.size() != 2 || g_shape.size() != 1 || o_shape.size() != 2) fail("weight_norm shape rank mismatch");
  if (o_shape != v_shape || g_shape[0] != v_shape[0]) fail("weight_norm shape mismatch");
  const int64_t M = v_shape[0];
  const int64_t N = v_shape[1];

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + v + ",");
  w.line("const float* __restrict__ " + g + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum_sq = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float x = " + v + "[base + n];");
  w.line("sum_sq = fmaf(x, x, sum_sq);");
  w.dedent();
  w.line("}");
  w.line("const float norm = sqrtf(sum_sq);");
  w.line("const float scale = (norm > 0.0f) ? (" + g + "[row] / norm) : 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line(out_name + "[base + n] = " + v + "[base + n] * scale;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {v, g, out_name}, {}, {v, g, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_upsample_bicubic2d_aa_f32(const Intent& intent, const json& bindings) {
  bool direct_semantic = false;
  if (intent.ops.size() == 1 && intent.ops[0].op == "upsample_bicubic2d_aa") direct_semantic = true;
  const Op* semantic_op = direct_semantic ? &intent.ops[0] : nullptr;

  std::string in_name;
  std::string out_name;
  bool has_scale_tensors = false;
  std::string rsh_name;
  std::string rsw_name;
  if (semantic_op != nullptr) {
    if (!(semantic_op->inputs.size() == 1 || semantic_op->inputs.size() == 3)) {
      fail("upsample_bicubic2d_aa expects ptr_i or ptr_i+reciprocal scales");
    }
    in_name = semantic_op->inputs[0];
    out_name = semantic_op->output;
    has_scale_tensors = (semantic_op->inputs.size() == 3);
    rsh_name = has_scale_tensors ? semantic_op->inputs[1] : std::string();
    rsw_name = has_scale_tensors ? semantic_op->inputs[2] : std::string();
  } else {
    // Fallback for expanded intent payloads used by certain harnesses.
    if (intent.tensors.find("ptr_i") == intent.tensors.end() || intent.tensors.find("ptr_o") == intent.tensors.end()) {
      fail("upsample_bicubic2d_aa expanded lowering expects tensors ptr_i/ptr_o");
    }
    in_name = "ptr_i";
    out_name = "ptr_o";
    has_scale_tensors = (intent.tensors.find("reciprocal_scale_h") != intent.tensors.end() && intent.tensors.find("reciprocal_scale_w") != intent.tensors.end());
    rsh_name = has_scale_tensors ? "reciprocal_scale_h" : std::string();
    rsw_name = has_scale_tensors ? "reciprocal_scale_w" : std::string();
  }
  auto i_it = intent.tensors.find(in_name);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("upsample_bicubic2d_aa missing tensors");
  }
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") {
    fail("upsample_bicubic2d_aa supports f32 only");
  }
  float reciprocal_scale_h_const = 1.0f;
  float reciprocal_scale_w_const = 1.0f;
  if (semantic_op != nullptr) {
    reciprocal_scale_h_const = static_cast<float>(semantic_op->attrs.value("invscale", 1.0));
    reciprocal_scale_w_const = static_cast<float>(semantic_op->attrs.value("invscale", 1.0));
    if (semantic_op->attrs.is_object()) {
      if (semantic_op->attrs.contains("reciprocal_scale_h")) reciprocal_scale_h_const = static_cast<float>(semantic_op->attrs["reciprocal_scale_h"].get<double>());
      if (semantic_op->attrs.contains("reciprocal_scale_w")) reciprocal_scale_w_const = static_cast<float>(semantic_op->attrs["reciprocal_scale_w"].get<double>());
    }
  }
  if (has_scale_tensors) {
    auto h_it = intent.tensors.find(rsh_name);
    auto w_it = intent.tensors.find(rsw_name);
    if (h_it == intent.tensors.end() || w_it == intent.tensors.end()) fail("upsample_bicubic2d_aa missing reciprocal scale tensors");
    if (h_it->second.dtype != "f32" || w_it->second.dtype != "f32") fail("upsample_bicubic2d_aa reciprocal scales must be f32");
  }
  auto i_shape = resolve_shape_required(i_it->second, bindings, "upsample_bicubic2d_aa.in");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "upsample_bicubic2d_aa.out");
  if (i_shape.size() != 4 || o_shape.size() != 4) fail("upsample_bicubic2d_aa expects rank-4 [N,C,H,W]");
  const int64_t N = i_shape[0];
  const int64_t C = i_shape[1];
  const int64_t IH = i_shape[2];
  const int64_t IW = i_shape[3];
  const int64_t ON = o_shape[0];
  const int64_t OC = o_shape[1];
  const int64_t OH = o_shape[2];
  const int64_t OW = o_shape[3];
  if (ON != N || OC != C) fail("upsample_bicubic2d_aa N/C mismatch");
  const int64_t total = N * C * OH * OW;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + in_name + ",");
  if (has_scale_tensors) {
    w.line("const float* __restrict__ " + rsh_name + ",");
    w.line("const float* __restrict__ " + rsw_name + ",");
  }
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t c = t % " + std::to_string(C) + "LL; t /= " + std::to_string(C) + "LL;");
  w.line("const int64_t n = t;");
  if (has_scale_tensors) {
    w.line("const float reciprocal_scale_h_val = " + rsh_name + " ? " + rsh_name + "[0] : ((float)" + std::to_string(IH) + " / (float)" +
           std::to_string(OH) + ");");
    w.line("const float reciprocal_scale_w_val = " + rsw_name + " ? " + rsw_name + "[0] : ((float)" + std::to_string(IW) + " / (float)" +
           std::to_string(OW) + ");");
  } else {
    w.line("const float reciprocal_scale_h_val = " + std::to_string(reciprocal_scale_h_const) + "f;");
    w.line("const float reciprocal_scale_w_val = " + std::to_string(reciprocal_scale_w_const) + "f;");
  }
  w.line("const float support_h = (reciprocal_scale_h_val >= 1.0f) ? (2.0f * reciprocal_scale_h_val) : 2.0f;");
  w.line("const float support_w = (reciprocal_scale_w_val >= 1.0f) ? (2.0f * reciprocal_scale_w_val) : 2.0f;");
  w.line("const int interp_h = ((int)(support_h + 0.5f)) * 2 + 1;");
  w.line("const int interp_w = ((int)(support_w + 0.5f)) * 2 + 1;");
  w.line("const float center_h = ((float)oh + 0.5f) * reciprocal_scale_h_val;");
  w.line("const float center_w = ((float)ow + 0.5f) * reciprocal_scale_w_val;");
  w.line("const int span_start_h = (int)fmaxf(center_h - support_h + 0.5f, 0.0f);");
  w.line("const int span_start_w = (int)fmaxf(center_w - support_w + 0.5f, 0.0f);");
  w.line("const int span_size_h = (int)(fminf(center_h + support_h + 0.5f, (float)" + std::to_string(IH) + ") - (float)span_start_h);");
  w.line("const int span_size_w = (int)(fminf(center_w + support_w + 0.5f, (float)" + std::to_string(IW) + ") - (float)span_start_w);");
  w.line("const float invscale_h = (reciprocal_scale_h_val >= 1.0f) ? (1.0f / reciprocal_scale_h_val) : 1.0f;");
  w.line("const float invscale_w = (reciprocal_scale_w_val >= 1.0f) ? (1.0f / reciprocal_scale_w_val) : 1.0f;");
  w.line("const float start_minus_center_h = (float)span_start_h - center_h;");
  w.line("const float start_minus_center_w = (float)span_start_w - center_w;");
  w.line("const float a = -0.5f;");
  w.line("float wy[17];");
  w.line("float wx[17];");
  w.line("float sum_wy = 0.0f;");
  w.line("float sum_wx = 0.0f;");
  w.line("for (int y = 0; y < interp_h; ++y) {");
  w.indent();
  w.line("float wv = 0.0f;");
  w.line("if (y < span_size_h) {");
  w.indent();
  w.line("const float d = fabsf(((float)y + start_minus_center_h + 0.5f) * invscale_h);");
  w.line("if (d < 1.0f) wv = ((a + 2.0f) * d - (a + 3.0f)) * d * d + 1.0f;");
  w.line("else if (d < 2.0f) wv = (((d - 5.0f) * d + 8.0f) * d - 4.0f) * a;");
  w.dedent();
  w.line("}");
  w.line("wy[y] = wv;");
  w.line("sum_wy += wv;");
  w.dedent();
  w.line("}");
  w.line("for (int x = 0; x < interp_w; ++x) {");
  w.indent();
  w.line("float wv = 0.0f;");
  w.line("if (x < span_size_w) {");
  w.indent();
  w.line("const float d = fabsf(((float)x + start_minus_center_w + 0.5f) * invscale_w);");
  w.line("if (d < 1.0f) wv = ((a + 2.0f) * d - (a + 3.0f)) * d * d + 1.0f;");
  w.line("else if (d < 2.0f) wv = (((d - 5.0f) * d + 8.0f) * d - 4.0f) * a;");
  w.dedent();
  w.line("}");
  w.line("wx[x] = wv;");
  w.line("sum_wx += wv;");
  w.dedent();
  w.line("}");
  w.line("if (sum_wy == 0.0f) sum_wy = 1.0f;");
  w.line("if (sum_wx == 0.0f) sum_wx = 1.0f;");
  w.line("for (int y = 0; y < interp_h; ++y) wy[y] /= sum_wy;");
  w.line("for (int x = 0; x < interp_w; ++x) wx[x] /= sum_wx;");
  w.line("float acc = 0.0f;");
  w.line("for (int y = 0; y < interp_h; ++y) {");
  w.indent();
  w.line("const int iy = span_start_h + y;");
  w.line("if (iy < 0 || iy >= " + std::to_string(IH) + ") continue;");
  w.line("for (int x = 0; x < interp_w; ++x) {");
  w.indent();
  w.line("const int ix = span_start_w + x;");
  w.line("if (ix < 0 || ix >= " + std::to_string(IW) + ") continue;");
  w.line("const int64_t in_idx = (((n * " + std::to_string(C) + "LL + c) * " + std::to_string(IH) + "LL + (int64_t)iy) * " +
         std::to_string(IW) + "LL + (int64_t)ix);");
  w.line("acc += " + in_name + "[in_idx] * wy[y] * wx[x];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = acc;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  out_bindings["C"] = C;
  out_bindings["IH"] = IH;
  out_bindings["IW"] = IW;
  out_bindings["OH"] = OH;
  out_bindings["OW"] = OW;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  if (has_scale_tensors) {
    out["io_spec"] = io_spec_from_args(intent, {in_name, rsh_name, rsw_name, out_name}, {}, {in_name, rsh_name, rsw_name, out_name});
  } else {
    out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  }
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_per_token_group_quant_fp8_2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 11) fail("per_token_group_quant_fp8 lowering expects 11-op pattern");
  static const std::vector<std::string> kOps = {
      "reshape", "abs", "reduce_max", "max", "div", "reshape", "broadcast_in_dim", "div", "max", "min", "reshape"};
  for (size_t i = 0; i < kOps.size(); ++i) {
    if (intent.ops[i].op != kOps[i]) fail("per_token_group_quant_fp8 lowering pattern mismatch");
  }
  const Op& reshape0 = intent.ops[0];
  const Op& clamp_eps = intent.ops[3];
  const Op& scale_div = intent.ops[4];
  const Op& scale_reshape = intent.ops[5];
  const Op& clamp_min = intent.ops[8];
  const Op& clamp_max = intent.ops[9];
  const Op& out_reshape = intent.ops[10];

  if (reshape0.inputs.size() != 1) fail("per_token_group_quant_fp8 reshape input mismatch");
  if (clamp_eps.inputs.size() != 2 || scale_div.inputs.size() != 2 || clamp_min.inputs.size() != 2 || clamp_max.inputs.size() != 2) {
    fail("per_token_group_quant_fp8 scalar inputs mismatch");
  }

  const std::string y_name = reshape0.inputs[0];
  const std::string y_q_name = out_reshape.output;
  const std::string y_s_name = scale_reshape.output;
  const std::string eps_name = clamp_eps.inputs[1];
  const std::string fp8_max_name = scale_div.inputs[1];
  const std::string fp8_min_name = clamp_min.inputs[1];

  auto y_it = intent.tensors.find(y_name);
  auto yq_it = intent.tensors.find(y_q_name);
  auto ys_it = intent.tensors.find(y_s_name);
  auto eps_it = intent.tensors.find(eps_name);
  auto min_it = intent.tensors.find(fp8_min_name);
  auto max_it = intent.tensors.find(fp8_max_name);
  if (y_it == intent.tensors.end() || yq_it == intent.tensors.end() || ys_it == intent.tensors.end() || eps_it == intent.tensors.end() ||
      min_it == intent.tensors.end() || max_it == intent.tensors.end()) {
    fail("per_token_group_quant_fp8 tensors missing");
  }
  if (y_it->second.dtype != "f32" || yq_it->second.dtype != "f32" || ys_it->second.dtype != "f32" || eps_it->second.dtype != "f32" ||
      min_it->second.dtype != "f32" || max_it->second.dtype != "f32") {
    fail("per_token_group_quant_fp8 supports f32 tensors/scalars only");
  }

  auto y_shape = resolve_shape_required(y_it->second, bindings, "per_token_group_quant_fp8.y.shape");
  auto ys_shape = resolve_shape_required(ys_it->second, bindings, "per_token_group_quant_fp8.y_s.shape");
  if (y_shape.size() != 2 || ys_shape.size() != 2) fail("per_token_group_quant_fp8 expects rank-2 tensors");
  const int64_t M = y_shape[0];
  const int64_t N = y_shape[1];
  const int64_t G = ys_shape[1];
  if (M <= 0 || N <= 0 || G <= 0) fail("per_token_group_quant_fp8 invalid shape");
  if (ys_shape[0] != M) fail("per_token_group_quant_fp8 y_s shape mismatch");
  if ((N % G) != 0) fail("per_token_group_quant_fp8 requires N % G == 0");
  const int64_t GROUP_SIZE = N / G;
  const int64_t MG = M * G;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (MG + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + y_name + ",");
  w.line("const float* __restrict__ " + eps_name + ",");
  w.line("const float* __restrict__ " + fp8_min_name + ",");
  w.line("const float* __restrict__ " + fp8_max_name + ",");
  w.line("float* __restrict__ " + y_q_name + ",");
  w.line("float* __restrict__ " + y_s_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t gid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (gid >= " + std::to_string(MG) + "LL) return;");
  w.line("const int64_t m = gid / " + std::to_string(G) + "LL;");
  w.line("const int64_t g = gid - m * " + std::to_string(G) + "LL;");
  w.line("const int64_t base = m * " + std::to_string(N) + "LL + g * " + std::to_string(GROUP_SIZE) + "LL;");
  w.line("float absmax = 0.0f;");
  w.line("for (int64_t k = 0; k < " + std::to_string(GROUP_SIZE) + "LL; ++k) {");
  w.indent();
  w.line("absmax = fmaxf(absmax, fabsf(" + y_name + "[base + k]));");
  w.dedent();
  w.line("}");
  w.line("float scale = fmaxf(absmax, " + eps_name + "[0]) / " + fp8_max_name + "[0];");
  w.line("if (!isfinite(scale) || scale == 0.0f) scale = 1.0f;");
  w.line(y_s_name + "[m * " + std::to_string(G) + "LL + g] = scale;");
  w.line("const float inv_scale = 1.0f / scale;");
  w.line("const float clamp_lo = " + fp8_min_name + "[0];");
  w.line("const float clamp_hi = " + fp8_max_name + "[0];");
  w.line("for (int64_t k = 0; k < " + std::to_string(GROUP_SIZE) + "LL; ++k) {");
  w.indent();
  w.line("float q = " + y_name + "[base + k] * inv_scale;");
  w.line("q = fmaxf(q, clamp_lo);");
  w.line("q = fminf(q, clamp_hi);");
  w.line(y_q_name + "[base + k] = q;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {y_name, eps_name, fp8_min_name, fp8_max_name, y_q_name, y_s_name};
  std::vector<std::string> arg_names = {y_name, eps_name, fp8_min_name, fp8_max_name, y_q_name, y_s_name};
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, arg_names);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {y_q_name, y_s_name};
  out["bindings"] = bindings;
  return out;
}
