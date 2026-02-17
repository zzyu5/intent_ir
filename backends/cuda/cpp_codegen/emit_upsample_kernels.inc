json emit_upsample_nearest1d_ncl_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "upsample_nearest1d")) fail("upsample_nearest1d lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("upsample_nearest1d expects one input");
  const std::string x_name = op.inputs[0];
  const std::string out_name = op.output;
  auto x_it = intent.tensors.find(x_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("upsample_nearest1d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "upsample_nearest1d.input.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "upsample_nearest1d.output.shape");
  if (x_shape.size() != 3 || o_shape.size() != 3) fail("upsample_nearest1d expects rank-3 NCL tensors");
  if (x_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("upsample_nearest1d supports f32 only");
  int64_t N = x_shape[0], C = x_shape[1], IL = x_shape[2], OL = o_shape[2];
  if (IL <= 0 || OL <= 0) fail("upsample_nearest1d invalid shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C * OL;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ol = t % " + std::to_string(OL) + "LL; t /= " + std::to_string(OL) + "LL;");
  w.line("const int64_t c = t % " + std::to_string(C) + "LL; t /= " + std::to_string(C) + "LL;");
  w.line("const int64_t n = t;");
  w.line("int64_t il = (ol * " + std::to_string(IL) + "LL) / " + std::to_string(OL) + "LL;");
  w.line("if (il < 0) il = 0;");
  w.line("if (il >= " + std::to_string(IL) + "LL) il = " + std::to_string(IL - 1) + "LL;");
  w.line("const int64_t in_idx = ((n * " + std::to_string(C) + "LL + c) * " + std::to_string(IL) + "LL + il);");
  w.line(out_name + "[tid] = " + x_name + "[in_idx];");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {x_name, out_name}, {}, {x_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_upsample_nearest2d_nchw_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "upsample_nearest2d")) fail("upsample_nearest2d lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("upsample_nearest2d expects one input");
  const std::string x_name = op.inputs[0];
  const std::string out_name = op.output;
  auto x_it = intent.tensors.find(x_name);
  auto o_it = intent.tensors.find(out_name);
  if (x_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("upsample_nearest2d tensors missing");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "upsample_nearest2d.input.shape");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "upsample_nearest2d.output.shape");
  if (x_shape.size() != 4 || o_shape.size() != 4) fail("upsample_nearest2d expects rank-4 NCHW tensors");
  if (x_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("upsample_nearest2d supports f32 only");
  int64_t N = x_shape[0], C = x_shape[1], IH = x_shape[2], IW = x_shape[3];
  int64_t OH = o_shape[2], OW = o_shape[3];
  if (IH <= 0 || IW <= 0 || OH <= 0 || OW <= 0) fail("upsample_nearest2d invalid shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  int64_t total = N * C * OH * OW;
  int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + x_name + ",");
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
  w.line("int64_t ih = (oh * " + std::to_string(IH) + "LL) / " + std::to_string(OH) + "LL;");
  w.line("int64_t iw = (ow * " + std::to_string(IW) + "LL) / " + std::to_string(OW) + "LL;");
  w.line("if (ih < 0) ih = 0;");
  w.line("if (iw < 0) iw = 0;");
  w.line("if (ih >= " + std::to_string(IH) + "LL) ih = " + std::to_string(IH - 1) + "LL;");
  w.line("if (iw >= " + std::to_string(IW) + "LL) iw = " + std::to_string(IW - 1) + "LL;");
  w.line("const int64_t in_idx = (((n * " + std::to_string(C) + "LL + c) * " + std::to_string(IH) + "LL + ih) * " + std::to_string(IW) + "LL + iw);");
  w.line(out_name + "[tid] = " + x_name + "[in_idx];");
  w.dedent();
  w.line("}");

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {x_name, out_name}, {}, {x_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

