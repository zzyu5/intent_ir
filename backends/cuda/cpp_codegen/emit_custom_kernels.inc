json emit_addmv_f32(const Intent& intent, const json& bindings) {
  std::string A;
  std::string B;
  std::string Inp;
  std::string Out;
  std::string alpha_name;
  std::string beta_name;

  auto has_binary_inputs = [](const Op& op, const std::string& x, const std::string& y) -> bool {
    if (op.inputs.size() != 2) return false;
    return (op.inputs[0] == x && op.inputs[1] == y) || (op.inputs[0] == y && op.inputs[1] == x);
  };

  if (intent.ops.size() == 4) {
    const Op& mm = intent.ops[0];
    const Op& mul_mv = intent.ops[1];
    const Op& mul_inp = intent.ops[2];
    const Op& add = intent.ops[3];
    if (mm.op != "matmul" || mul_mv.op != "mul" || mul_inp.op != "mul" || add.op != "add") {
      fail("addmv lowering pattern mismatch");
    }
    if (mm.inputs.size() != 2 || mul_mv.inputs.size() != 2 || mul_inp.inputs.size() != 2 || add.inputs.size() != 2) {
      fail("addmv lowering: invalid arity in pattern ops");
    }

    A = mm.inputs[0];
    B = mm.inputs[1];
    const std::string mv_out = mm.output;

    if (mul_mv.inputs[0] == mv_out)
      alpha_name = mul_mv.inputs[1];
    else if (mul_mv.inputs[1] == mv_out)
      alpha_name = mul_mv.inputs[0];
    else
      fail("addmv lowering expects one mul input from matmul output");

    auto is_rank1_f32 = [&](const std::string& n) -> bool {
      auto it = intent.tensors.find(n);
      return it != intent.tensors.end() && it->second.dtype == "f32" && it->second.shape.size() == 1;
    };
    auto is_scalar_f32 = [&](const std::string& n) -> bool {
      auto it = intent.tensors.find(n);
      return it != intent.tensors.end() && it->second.dtype == "f32" && it->second.shape.empty();
    };
    if (is_rank1_f32(mul_inp.inputs[0]) && is_scalar_f32(mul_inp.inputs[1])) {
      Inp = mul_inp.inputs[0];
      beta_name = mul_inp.inputs[1];
    } else if (is_rank1_f32(mul_inp.inputs[1]) && is_scalar_f32(mul_inp.inputs[0])) {
      Inp = mul_inp.inputs[1];
      beta_name = mul_inp.inputs[0];
    } else {
      fail("addmv lowering expects mul(input,beta_scalar) pattern");
    }

    const std::string scaled_mv = mul_mv.output;
    const std::string scaled_inp = mul_inp.output;
    const bool add_inputs_ok =
        ((add.inputs[0] == scaled_mv && add.inputs[1] == scaled_inp) || (add.inputs[0] == scaled_inp && add.inputs[1] == scaled_mv));
    if (!add_inputs_ok) fail("addmv lowering expects add(scaled_mv, scaled_inp)");
    Out = add.output;
  } else if (intent.ops.size() == 10) {
    const Op& cast_A = intent.ops[0];
    const Op& cast_B = intent.ops[1];
    const Op& bcast_B = intent.ops[2];
    const Op& mul_AB = intent.ops[3];
    const Op& reduce_mv = intent.ops[4];
    const Op& mul_alpha = intent.ops[5];
    const Op& cast_Inp = intent.ops[6];
    const Op& mul_beta = intent.ops[7];
    const Op& add_out = intent.ops[8];
    const Op& cast_Out = intent.ops[9];

    if (cast_A.op != "cast" || cast_B.op != "cast" || bcast_B.op != "broadcast_in_dim" || mul_AB.op != "mul" ||
        reduce_mv.op != "reduce_sum" || mul_alpha.op != "mul" || cast_Inp.op != "cast" || mul_beta.op != "mul" ||
        add_out.op != "add" || cast_Out.op != "cast") {
      fail("addmv lowering pattern mismatch for cast+broadcast+reduce form");
    }
    if (cast_A.inputs.size() != 1 || cast_B.inputs.size() != 1 || bcast_B.inputs.size() != 1 || mul_AB.inputs.size() != 2 ||
        reduce_mv.inputs.size() != 1 || mul_alpha.inputs.size() != 2 || cast_Inp.inputs.size() != 1 || mul_beta.inputs.size() != 2 ||
        add_out.inputs.size() != 2 || cast_Out.inputs.size() != 1) {
      fail("addmv lowering: invalid arity in cast+broadcast+reduce form");
    }

    A = cast_A.inputs[0];
    B = cast_B.inputs[0];
    Inp = cast_Inp.inputs[0];

    if (bcast_B.inputs[0] != cast_B.output) fail("addmv lowering expects broadcast(cast(B))");
    if (!has_binary_inputs(mul_AB, cast_A.output, bcast_B.output)) fail("addmv lowering expects mul(cast(A), broadcast(cast(B)))");
    if (reduce_mv.inputs[0] != mul_AB.output) fail("addmv lowering expects reduce_sum(mul_AB)");
    if (mul_alpha.inputs[0] == reduce_mv.output)
      alpha_name = mul_alpha.inputs[1];
    else if (mul_alpha.inputs[1] == reduce_mv.output)
      alpha_name = mul_alpha.inputs[0];
    else
      fail("addmv lowering expects mul(reduce_mv, alpha)");
    if (mul_beta.inputs[0] == cast_Inp.output)
      beta_name = mul_beta.inputs[1];
    else if (mul_beta.inputs[1] == cast_Inp.output)
      beta_name = mul_beta.inputs[0];
    else
      fail("addmv lowering expects mul(cast(inp), beta)");
    if (!has_binary_inputs(add_out, mul_alpha.output, mul_beta.output)) fail("addmv lowering expects add(mul_alpha, mul_beta)");
    if (cast_Out.inputs[0] != add_out.output) fail("addmv lowering expects cast(add_out) as final output");
    Out = cast_Out.output;
  } else {
    fail("addmv lowering expects either 4-op or 10-op canonical pattern");
  }

  auto A_it = intent.tensors.find(A);
  auto B_it = intent.tensors.find(B);
  if (A_it == intent.tensors.end() || B_it == intent.tensors.end()) fail("addmv lowering missing A/B tensors");
  if (A_it->second.shape.size() != 2 || B_it->second.shape.size() != 1) fail("addmv lowering expects A[2D], B[1D]");
  if (A_it->second.dtype != "f32" || B_it->second.dtype != "f32") fail("addmv lowering currently supports f32 tensors only");

  auto Out_it = intent.tensors.find(Out);
  if (Out_it == intent.tensors.end()) fail("addmv lowering missing output tensor");
  if (Out_it->second.dtype != "f32" || Out_it->second.shape.size() != 1) fail("addmv lowering expects Out as f32 rank-1");

  auto alpha_it = intent.tensors.find(alpha_name);
  if (alpha_it == intent.tensors.end()) fail("addmv lowering missing alpha tensor");
  if (alpha_it->second.dtype != "f32" || !alpha_it->second.shape.empty()) fail("addmv lowering expects alpha scalar f32 tensor");

  auto Inp_it = intent.tensors.find(Inp);
  if (Inp_it == intent.tensors.end()) fail("addmv lowering missing input vector tensor");
  if (Inp_it->second.dtype != "f32" || Inp_it->second.shape.size() != 1) fail("addmv lowering expects input vector f32 rank-1");

  const json N_dim = A_it->second.shape[0];
  const json M_dim = A_it->second.shape[1];
  const json M2_dim = B_it->second.shape[0];
  const json N2_dim = Inp_it->second.shape[0];
  if (dim_str(M_dim) != dim_str(M2_dim)) fail("addmv lowering: A second dim must match B first dim");
  if (dim_str(N_dim) != dim_str(N2_dim)) fail("addmv lowering: A first dim must match input vector dim");

  auto N_opt = resolve_dim_token(N_dim, bindings);
  auto M_opt = resolve_dim_token(M_dim, bindings);
  if (!N_opt.has_value() || !M_opt.has_value()) fail("addmv lowering missing bindings for N/M");
  const int64_t N = *N_opt;
  const int64_t M = *M_opt;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  const int64_t grid_x = (N + block_x - 1) / block_x;

  const bool n_is_tensor = is_scalar_tensor(intent, "N", "i32");
  const bool m_is_tensor = is_scalar_tensor(intent, "M", "i32");
  const std::string n_param = n_is_tensor ? "const int* N_ptr" : "int N_in";
  const std::string m_param = m_is_tensor ? "const int* M_ptr" : "int M_in";
  const std::string n_load = n_is_tensor ? "(N_ptr ? N_ptr[0] : 0)" : "N_in";
  const std::string m_load = m_is_tensor ? "(M_ptr ? M_ptr[0] : 0)" : "M_in";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.blank();
  emit_selected_api(w);
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + A + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("const float* __restrict__ " + Inp + ",");
  w.line("float* __restrict__ " + Out + ",");
  w.line("const float* __restrict__ " + alpha_name + ",");
  w.line("const float* __restrict__ " + beta_name + ",");
  w.line(n_param + ",");
  w.line(m_param);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int N = " + n_load + ";");
  w.line("const int M = " + m_load + ";");
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= (int64_t)N) return;");
  w.line("const float alpha_v = " + alpha_name + " ? " + alpha_name + "[0] : 1.0f;");
  w.line("const float beta_v = " + beta_name + " ? " + beta_name + "[0] : 1.0f;");
  w.line("float acc = 0.0f;");
  w.line("for (int k = 0; k < M; ++k) {");
  w.indent();
  w.line("acc = fmaf(" + A + "[(size_t)tid * (size_t)M + (size_t)k], " + B + "[(size_t)k], acc);");
  w.dedent();
  w.line("}");
  w.line(Out + "[(size_t)tid] = acc * alpha_v + " + Inp + "[(size_t)tid] * beta_v;");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {A, B, Inp, Out, alpha_name, beta_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {A, B, Inp, Out, alpha_name, beta_name};
  if (n_is_tensor) {
    tensor_args.push_back("N");
    arg_names.push_back("N");
  } else {
    scalar_args.emplace("N", "i32");
    arg_names.push_back("N");
  }
  if (m_is_tensor) {
    tensor_args.push_back("M");
    arg_names.push_back("M");
  } else {
    scalar_args.emplace("M", "i32");
    arg_names.push_back("M");
  }

  json out_bindings = bindings;
  if (!out_bindings.contains("N")) out_bindings["N"] = N;
  if (!out_bindings.contains("M")) out_bindings["M"] = M;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {Out};
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

  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;
  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");
  const bool enable_host_dispatch = want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims;
  const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
  bool host_launch = false;

  int64_t block_w = binding_int(bindings, "BLOCK_W").value_or(0);
  if (block_w <= 0) {
    const int64_t hinted = resolve_schedule_int(intent, bindings, "tile_n", 0);
    if (0 < hinted && hinted <= 1024) block_w = hinted;
  }
  if (block_w <= 0) {
    // Shape-driven seed (paper-friendly): pick a block width proportional to W.
    // For the AI-Bench warp workload (W=1024), this avoids a weak seed (128).
    if (W >= 512)
      block_w = 512;
    else if (W >= 256)
      block_w = 256;
    else if (W >= 128)
      block_w = 128;
    else
      block_w = 64;
  }
  if (block_w < 32) block_w = 32;
  if (block_w > 1024) block_w = 1024;
  if ((block_w % 32) != 0) block_w = ((block_w + 31) / 32) * 32;
  const int64_t grid_w = (W + block_w - 1) / block_w;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/warp.cuh\"");
  w.blank();
  emit_selected_api(w);
  if (!enable_host_dispatch) {
    emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
    w.line("extern \"C\" __global__ void " + intent.name +
           "(const int8_t* __restrict__ " + src_name + ", const int16_t* __restrict__ " + offset_name + ", int8_t* __restrict__ " +
           out_name + ", int C_in, int H_in, int W_in) {");
    w.indent();
    w.line("constexpr int BLOCK_W = " + std::to_string(block_w) + ";");
    if (specialize_dims) {
      w.line("(void)C_in;");
      w.line("(void)H_in;");
      w.line("(void)W_in;");
      w.line("constexpr int C = " + std::to_string(C) + ";");
      w.line("constexpr int H = " + std::to_string(H) + ";");
      w.line("constexpr int W = " + std::to_string(W) + ";");
    } else {
      w.line("const int C = C_in;");
      w.line("const int H = H_in;");
      w.line("const int W = W_in;");
    }
    const bool full_w_fast = contract_full && specialize_dims && ((W % block_w) == 0);
    w.line("intentir_cuda::warp_q8_8_i8_i16<BLOCK_W, " + std::string(full_w_fast ? "true" : "false") + ">(" + src_name + ", " + offset_name +
           ", " + out_name + ", C, H, W);");
    w.dedent();
    w.line("}");
  } else {
    host_launch = true;
    struct WarpVariant {
      int64_t block_w;
      int64_t grid_w;
      bool full_w;
      std::string suffix;
    };
    std::vector<WarpVariant> variants;
    auto norm_bw = [](int64_t b) -> int64_t {
      if (b < 32) b = 32;
      if (b > 1024) b = 1024;
      if ((b % 32) != 0) b = ((b + 31) / 32) * 32;
      if (b > 1024) b = 1024;
      return b;
	    };
		    auto add_variant = [&](int64_t bw, const std::string& tag) {
		      bw = norm_bw(bw);
		      const int64_t gw = (W + bw - 1) / bw;
		      const bool full = contract_full && ((W % bw) == 0);
	      for (const auto& v : variants) {
	        if (v.block_w == bw) return;
	      }
		      variants.push_back(WarpVariant{bw, gw, full, tag});
	    };

	    add_variant(block_w, "seed_bw" + std::to_string(block_w));
	    add_variant(block_w / 2, "bw_half");
	    add_variant(block_w * 2, "bw_double");
	    if (!has_evidence) {
	      // Evidence-off: widen the candidate set so selection can recover without
	      // certificate priors (kept small to avoid codegen blowup).
	      add_variant(64, "bw64");
	      add_variant(128, "bw128");
	      add_variant(256, "bw256");
	      add_variant(512, "bw512");
	      add_variant(1024, "bw1024");
	    }
	    if (variants.empty()) add_variant(block_w, "fallback");

	    emit_const_introspection_api(w, (int)variants.size(), has_evidence, contract_level, specialize_dims);

    for (const auto& v : variants) {
      const std::string kname = intent.name + "__" + v.suffix;
      w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.block_w) + ") void " + kname +
             "(const int8_t* __restrict__ " + src_name + ", const int16_t* __restrict__ " + offset_name + ", int8_t* __restrict__ " +
             out_name + ", int C_in, int H_in, int W_in) {");
      w.indent();
      w.line("constexpr int BLOCK_W = " + std::to_string(v.block_w) + ";");
      w.line("(void)C_in; (void)H_in; (void)W_in;");
      w.line("constexpr int C = " + std::to_string(C) + ";");
      w.line("constexpr int H = " + std::to_string(H) + ";");
      w.line("constexpr int W = " + std::to_string(W) + ";");
      if (v.full_w) {
        w.line("intentir_cuda::warp_q8_8_i8_i16<BLOCK_W, true>(" + src_name + ", " + offset_name + ", " + out_name + ", C, H, W);");
      } else {
        w.line("intentir_cuda::warp_q8_8_i8_i16<BLOCK_W, false>(" + src_name + ", " + offset_name + ", " + out_name + ", C, H, W);");
      }
      w.dedent();
      w.line("}");
      w.blank();
    }

    w.line("extern \"C\" void " + intent.name + "_host_launch(");
    w.indent();
    w.line("int8_t* " + src_name + ", int16_t* " + offset_name + ", int8_t* " + out_name + ", int C_in, int H_in, int W_in,");
    w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
    w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
    w.line("int64_t shared_mem, cudaStream_t stream) {");
    w.dedent();
    w.indent();
    w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
    w.line("(void)block_x; (void)block_y; (void)block_z;");
    w.line("(void)shared_mem;");
    w.line("(void)C_in; (void)H_in; (void)W_in;");
    w.line("constexpr int C = " + std::to_string(C) + ";");
    w.line("constexpr int H = " + std::to_string(H) + ";");
    w.line("constexpr int W = " + std::to_string(W) + ";");
    w.line("constexpr bool INTENTIR_HOST_DISPATCH_SELECT = " + std::string(enable_host_dispatch_select ? "true" : "false") + ";");
    w.line("intentir_cuda_dispatch_total_ms = 0.0f;");
    w.line("intentir_cuda_dispatch_evals = 0;");
    w.line("intentir_cuda_fastpath_enabled = 0;");
    w.line("static int intentir_selected = -1;");
    w.line("if (intentir_selected < 0) {");
    w.indent();
    w.line("if (!INTENTIR_HOST_DISPATCH_SELECT) {");
    w.indent();
    w.line("intentir_selected = 0;");
    w.dedent();
    w.line("} else {");
    w.indent();
    w.line("cudaEvent_t start = nullptr;");
    w.line("cudaEvent_t end = nullptr;");
    w.line("TORCH_CHECK(cudaEventCreate(&start) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaEventCreate(&end) == cudaSuccess);");
    w.line("cudaStream_t sel_stream = nullptr;");
    w.line("TORCH_CHECK(cudaStreamCreateWithFlags(&sel_stream, cudaStreamNonBlocking) == cudaSuccess);");
    // Match the benchmark harness: capture a CUDA graph with `iters` launches
    // and time a single replay. This avoids picking variants that only win
    // under Python submission overhead (small kernels).
    w.line("constexpr int warm = 3;");
    w.line("constexpr int iters = 50;");
    w.line("cudaGraph_t graphs[" + std::to_string(variants.size()) + "] = {};");
    w.line("cudaGraphExec_t execs[" + std::to_string(variants.size()) + "] = {};");
    w.line("float ms_acc[" + std::to_string(variants.size()) + "] = {0};");

    // Capture & instantiate graphs for each variant.
    for (size_t i = 0; i < variants.size(); ++i) {
      const auto& v = variants[i];
      const std::string kname = intent.name + "__" + v.suffix;
      w.line("{");
      w.indent();
      w.line("dim3 b((unsigned)" + std::to_string(v.block_w) + ", 1u, 1u);");
      w.line("dim3 g((unsigned)" + std::to_string(v.grid_w) + ", (unsigned)H, (unsigned)C);");
      w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + src_name + ", " + offset_name + ", " + out_name +
             ", C, H, W);");
      w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
      w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + src_name + ", " + offset_name + ", " + out_name +
             ", C, H, W);");
      w.line("TORCH_CHECK(cudaStreamEndCapture(sel_stream, &graphs[" + std::to_string(i) + "]) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphInstantiate(&execs[" + std::to_string(i) + "], graphs[" + std::to_string(i) +
             "], nullptr, nullptr, 0) == cudaSuccess);");
      w.dedent();
      w.line("}");
    }

    // Forward pass then reverse pass to reduce clock/thermal order bias.
    for (size_t i = 0; i < variants.size(); ++i) {
      w.line("{");
      w.indent();
      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
      w.line("float ms = 0.0f;");
      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
      w.line("ms_acc[" + std::to_string(i) + "] += ms;");
      w.dedent();
      w.line("}");
    }
    for (size_t ri = variants.size(); ri-- > 0;) {
      w.line("{");
      w.indent();
      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
      w.line("float ms = 0.0f;");
      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
      w.line("ms_acc[" + std::to_string(ri) + "] += ms;");
      w.dedent();
      w.line("}");
    }

    w.line("float best_ms = 1e30f;");
    w.line("int best_i = 0;");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
    w.indent();
    w.line("const float ms = ms_acc[i];");
    w.line("if (ms < best_ms) { best_ms = ms; best_i = i; }");
    w.dedent();
    w.line("}");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
    w.indent();
    w.line("if (execs[i]) TORCH_CHECK(cudaGraphExecDestroy(execs[i]) == cudaSuccess);");
    w.line("if (graphs[i]) TORCH_CHECK(cudaGraphDestroy(graphs[i]) == cudaSuccess);");
    w.dedent();
    w.line("}");
    w.line("TORCH_CHECK(cudaEventDestroy(start) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaEventDestroy(end) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaStreamDestroy(sel_stream) == cudaSuccess);");
    w.line("intentir_selected = best_i;");
    w.line("float total_ms = 0.0f;");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) total_ms += ms_acc[i];");
    w.line("intentir_cuda_dispatch_total_ms = total_ms;");
    w.line("intentir_cuda_dispatch_evals = " + std::to_string(2 * variants.size()) + ";");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.line("switch (intentir_selected) {");
	    for (size_t i = 0; i < variants.size(); ++i) {
	      const auto& v = variants[i];
	      const std::string kname = intent.name + "__" + v.suffix;
	      w.line("case " + std::to_string(i) + ": {");
	      w.indent();
	      w.line("intentir_cuda_selected_variant_idx = " + std::to_string(i) + ";");
	      w.line("intentir_cuda_selected_variant_tag = \"" + v.suffix + "\";");
	      w.line("intentir_cuda_fastpath_enabled = " + std::string(v.full_w ? "1" : "0") + ";");
	      w.line("dim3 b((unsigned)" + std::to_string(v.block_w) + ", 1u, 1u);");
	      w.line("dim3 g((unsigned)" + std::to_string(v.grid_w) + ", (unsigned)H, (unsigned)C);");
	      w.line(kname + "<<<g, b, 0, stream>>>(" + src_name + ", " + offset_name + ", " + out_name + ", C, H, W);");
	      w.line("break;");
      w.dedent();
      w.line("}");
    }
    w.line("default: {");
    w.indent();
	    const auto& v0 = variants.front();
	    const std::string k0 = intent.name + "__" + v0.suffix;
	    w.line("intentir_cuda_selected_variant_idx = 0;");
	    w.line("intentir_cuda_selected_variant_tag = \"" + v0.suffix + "\";");
	    w.line("intentir_cuda_fastpath_enabled = " + std::string(v0.full_w ? "1" : "0") + ";");
	    w.line("dim3 b((unsigned)" + std::to_string(v0.block_w) + ", 1u, 1u);");
	    w.line("dim3 g((unsigned)" + std::to_string(v0.grid_w) + ", (unsigned)H, (unsigned)C);");
	    w.line(k0 + "<<<g, b, 0, stream>>>(" + src_name + ", " + offset_name + ", " + out_name + ", C, H, W);");
	    w.line("break;");
    w.dedent();
    w.line("}");
    w.line("}");
    w.dedent();
    w.line("}");
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{src_name, offset_name, out_name},
      /*scalar_args=*/{{"C", "i32"}, {"H", "i32"}, {"W", "i32"}},
      /*arg_names=*/{src_name, offset_name, out_name, "C", "H", "W"});
  if (host_launch) out["io_spec"]["host_launch"] = true;
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

  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;
  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");
  const bool enable_host_dispatch = want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims;
  const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
  bool host_launch = false;

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
  emit_selected_api(w);
  auto dim_unused = [](const char* scalar_name, bool is_tensor) -> std::string {
    return std::string("(void)") + (is_tensor ? (std::string(scalar_name) + "_ptr") : std::string(scalar_name)) + ";";
  };
  const std::string oc_unused = dim_unused("out_channel", oc_is_tensor);
  const std::string ic_unused = dim_unused("in_channel", ic_is_tensor);
  const std::string h_unused = dim_unused("height", h_is_tensor);
  const std::string w_unused = dim_unused("width", w_is_tensor);
  const std::string sh_unused = dim_unused("out_shift", sh_is_tensor);

  if (!enable_host_dispatch) {
    emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
    w.line("extern \"C\" __global__ void " + intent.name + "(");
    w.indent();
    w.line("const int8_t* __restrict__ " + src0_name + ",");
    w.line("const int8_t* __restrict__ " + src1_name + ",");
    w.line("int8_t* __restrict__ " + out_name + ",");
    w.line(oc_param + ", " + ic_param + ", " + h_param + ", " + w_param + ", " + sh_param + ") {");
    w.dedent();
    w.indent();
    if (specialize_dims) {
      w.line(oc_unused);
      w.line(ic_unused);
      w.line(h_unused);
      w.line(w_unused);
      w.line(sh_unused);
      w.line("constexpr int out_channel_v = " + std::to_string(OC) + ";");
      w.line("constexpr int in_channel_v = " + std::to_string(IC) + ";");
      w.line("constexpr int height_v = " + std::to_string(H) + ";");
      w.line("constexpr int width_v = " + std::to_string(W) + ";");
      w.line("constexpr int out_shift_v = " + std::to_string(binding_int(bindings, "out_shift").value_or(0)) + ";");
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
      const bool full_tile_fast = contract_full && ((total % block_x) == 0);
      w.line("intentir_cuda::correlation_i8<BLOCK_THREADS, " + std::string(full_tile_fast ? "true" : "false") + ">(" + src0_name + ", " + src1_name +
             ", " + out_name + ", out_channel_v, in_channel_v, height_v, width_v, out_shift_v);");
    } else {
      if (!oc_load.empty()) w.line(oc_load);
      if (!ic_load.empty()) w.line(ic_load);
      if (!h_load.empty()) w.line(h_load);
      if (!w_load.empty()) w.line(w_load);
      if (!sh_load.empty()) w.line(sh_load);
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
      w.line("intentir_cuda::correlation_i8<BLOCK_THREADS, false>(" + src0_name + ", " + src1_name + ", " + out_name +
             ", out_channel, in_channel, height, width, out_shift);");
    }
    w.dedent();
    w.line("}");
  } else {
    host_launch = true;
    struct CorrVariant {
      int64_t threads;
      int64_t grid_x;
      bool full_tile;
      std::string suffix;
    };
    std::vector<CorrVariant> variants;
    auto norm_threads = [](int64_t t) -> int64_t {
      if (t < 32) t = 32;
      if (t > 1024) t = 1024;
      if ((t % 32) != 0) t = ((t + 31) / 32) * 32;
      if (t > 1024) t = 1024;
      return t;
    };
		    auto add_variant = [&](int64_t threads, const std::string& tag) {
		      threads = norm_threads(threads);
		      const int64_t gx = (total + threads - 1) / threads;
		      const bool full = contract_full && ((total % threads) == 0);
		      for (const auto& v : variants) {
		        if (v.threads == threads) return;
		      }
		      variants.push_back(CorrVariant{threads, gx, full, tag});
		    };
		    add_variant(block_x, "seed");
		    add_variant(block_x / 2, "t_half");
		    add_variant(block_x * 2, "t_double");
		    if (has_evidence && (total >= 32768)) {
		      // Evidence-on: keep a tight neighborhood but include a high-threads option
		      // for large outputs (can significantly improve throughput on wide kernels).
		      add_variant(512, "t512");
		    }
		    if (!has_evidence) {
		      // Evidence-off: widen the neighborhood.
		      add_variant(block_x - 32, "t_m32");
		      add_variant(block_x + 32, "t_p32");
		      add_variant(64, "t64");
		      add_variant(128, "t128");
		      add_variant(256, "t256");
		      add_variant(512, "t512");
		      add_variant(1024, "t1024");
		    }
		    if (variants.empty()) add_variant(block_x, "fallback");

		    emit_const_introspection_api(w, (int)variants.size(), has_evidence, contract_level, specialize_dims);

    const int64_t out_shift_val = binding_int(bindings, "out_shift").value_or(0);
    const std::string oc_arg = oc_is_tensor ? "out_channel_ptr" : "out_channel";
    const std::string ic_arg = ic_is_tensor ? "in_channel_ptr" : "in_channel";
    const std::string h_arg = h_is_tensor ? "height_ptr" : "height";
    const std::string w_arg = w_is_tensor ? "width_ptr" : "width";
    const std::string sh_arg = sh_is_tensor ? "out_shift_ptr" : "out_shift";

	    for (const auto& v : variants) {
	      const std::string kname = intent.name + "__" + v.suffix;
	      w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.threads) + ") void " + kname + "(");
      w.indent();
      w.line("const int8_t* __restrict__ " + src0_name + ",");
      w.line("const int8_t* __restrict__ " + src1_name + ",");
      w.line("int8_t* __restrict__ " + out_name + ",");
      w.line(oc_param + ", " + ic_param + ", " + h_param + ", " + w_param + ", " + sh_param + ") {");
      w.dedent();
      w.indent();
      w.line(oc_unused);
      w.line(ic_unused);
      w.line(h_unused);
      w.line(w_unused);
      w.line(sh_unused);
      w.line("constexpr int out_channel_v = " + std::to_string(OC) + ";");
      w.line("constexpr int in_channel_v = " + std::to_string(IC) + ";");
	      w.line("constexpr int height_v = " + std::to_string(H) + ";");
	      w.line("constexpr int width_v = " + std::to_string(W) + ";");
	      w.line("constexpr int out_shift_v = " + std::to_string(out_shift_val) + ";");
	      w.line("constexpr int BLOCK_THREADS = " + std::to_string(v.threads) + ";");
	      w.line("intentir_cuda::correlation_i8<BLOCK_THREADS, " + std::string(v.full_tile ? "true" : "false") + ">(" + src0_name + ", " + src1_name +
	             ", " + out_name + ", out_channel_v, in_channel_v, height_v, width_v, out_shift_v);");
	      w.dedent();
	      w.line("}");
	      w.blank();
	    }

    w.line("extern \"C\" void " + intent.name + "_host_launch(");
    w.indent();
    w.line("int8_t* " + src0_name + ", int8_t* " + src1_name + ", int8_t* " + out_name + ", " + oc_param + ", " + ic_param + ", " + h_param +
           ", " + w_param + ", " + sh_param + ",");
    w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
    w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
    w.line("int64_t shared_mem, cudaStream_t stream) {");
    w.dedent();
    w.indent();
    w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
    w.line("(void)block_x; (void)block_y; (void)block_z;");
    w.line("(void)shared_mem;");
    w.line(oc_unused);
    w.line(ic_unused);
    w.line(h_unused);
    w.line(w_unused);
    w.line(sh_unused);
    w.line("constexpr int out_channel_v = " + std::to_string(OC) + ";");
    w.line("constexpr int in_channel_v = " + std::to_string(IC) + ";");
    w.line("constexpr int height_v = " + std::to_string(H) + ";");
    w.line("constexpr int width_v = " + std::to_string(W) + ";");
    w.line("constexpr int out_shift_v = " + std::to_string(out_shift_val) + ";");
    w.line("constexpr bool INTENTIR_HOST_DISPATCH_SELECT = " + std::string(enable_host_dispatch_select ? "true" : "false") + ";");
    w.line("intentir_cuda_dispatch_total_ms = 0.0f;");
    w.line("intentir_cuda_dispatch_evals = 0;");
    w.line("intentir_cuda_fastpath_enabled = 0;");
    w.line("static int intentir_selected = -1;");
    w.line("if (intentir_selected < 0) {");
    w.indent();
    w.line("if (!INTENTIR_HOST_DISPATCH_SELECT) {");
    w.indent();
    w.line("intentir_selected = 0;");
    w.dedent();
    w.line("} else {");
	    w.indent();
	    w.line("cudaEvent_t start = nullptr;");
	    w.line("cudaEvent_t end = nullptr;");
	    w.line("TORCH_CHECK(cudaEventCreate(&start) == cudaSuccess);");
	    w.line("TORCH_CHECK(cudaEventCreate(&end) == cudaSuccess);");
	    w.line("cudaStream_t sel_stream = nullptr;");
	    w.line("TORCH_CHECK(cudaStreamCreateWithFlags(&sel_stream, cudaStreamNonBlocking) == cudaSuccess);");
	    // Match the benchmark harness: capture a CUDA graph with `iters` launches
	    // and time a single replay. This avoids picking variants that only win
	    // under Python submission overhead (small kernels).
	    w.line("constexpr int warm = 3;");
	    w.line("constexpr int iters = 50;");
	    w.line("cudaGraph_t graphs[" + std::to_string(variants.size()) + "] = {};");
	    w.line("cudaGraphExec_t execs[" + std::to_string(variants.size()) + "] = {};");
	    w.line("float ms_acc[" + std::to_string(variants.size()) + "] = {0};");

	    // Capture & instantiate graphs for each variant.
	    for (size_t i = 0; i < variants.size(); ++i) {
	      const auto& v = variants[i];
	      const std::string kname = intent.name + "__" + v.suffix;
	      w.line("{");
	      w.indent();
	      w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
	      w.line("dim3 g((unsigned)" + std::to_string(v.grid_x) + ", 1u, 1u);");
	      w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + src0_name + ", " + src1_name + ", " + out_name +
	             ", " + oc_arg + ", " + ic_arg + ", " + h_arg + ", " + w_arg + ", " + sh_arg + ");");
	      w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
	      w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + src0_name + ", " + src1_name + ", " + out_name +
	             ", " + oc_arg + ", " + ic_arg + ", " + h_arg + ", " + w_arg + ", " + sh_arg + ");");
	      w.line("TORCH_CHECK(cudaStreamEndCapture(sel_stream, &graphs[" + std::to_string(i) + "]) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaGraphInstantiate(&execs[" + std::to_string(i) + "], graphs[" + std::to_string(i) +
	             "], nullptr, nullptr, 0) == cudaSuccess);");
	      w.dedent();
	      w.line("}");
	    }

	    // Forward pass then reverse pass to reduce clock/thermal order bias.
	    for (size_t i = 0; i < variants.size(); ++i) {
	      w.line("{");
	      w.indent();
	      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
	      w.line("float ms = 0.0f;");
	      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
	      w.line("ms_acc[" + std::to_string(i) + "] += ms;");
	      w.dedent();
	      w.line("}");
	    }
	    for (size_t ri = variants.size(); ri-- > 0;) {
	      w.line("{");
	      w.indent();
	      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
	      w.line("float ms = 0.0f;");
	      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
	      w.line("ms_acc[" + std::to_string(ri) + "] += ms;");
	      w.dedent();
	      w.line("}");
	    }
	    w.line("float best_ms = 1e30f;");
	    w.line("int best_i = 0;");
	    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
	    w.indent();
	    w.line("const float ms = ms_acc[i];");
	    w.line("if (ms < best_ms) { best_ms = ms; best_i = i; }");
	    w.dedent();
	    w.line("}");
	    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
	    w.indent();
	    w.line("if (execs[i]) TORCH_CHECK(cudaGraphExecDestroy(execs[i]) == cudaSuccess);");
	    w.line("if (graphs[i]) TORCH_CHECK(cudaGraphDestroy(graphs[i]) == cudaSuccess);");
	    w.dedent();
	    w.line("}");
	    w.line("TORCH_CHECK(cudaEventDestroy(start) == cudaSuccess);");
	    w.line("TORCH_CHECK(cudaEventDestroy(end) == cudaSuccess);");
	    w.line("TORCH_CHECK(cudaStreamDestroy(sel_stream) == cudaSuccess);");
	    w.line("intentir_selected = best_i;");
	    w.line("float total_ms = 0.0f;");
	    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) total_ms += ms_acc[i];");
	    w.line("intentir_cuda_dispatch_total_ms = total_ms;");
	    w.line("intentir_cuda_dispatch_evals = " + std::to_string(2 * variants.size()) + ";");
	    w.dedent();
	    w.line("}");
    w.dedent();
    w.line("}");
    w.line("switch (intentir_selected) {");
	    for (size_t i = 0; i < variants.size(); ++i) {
	      const auto& v = variants[i];
	      const std::string kname = intent.name + "__" + v.suffix;
	      w.line("case " + std::to_string(i) + ": {");
	      w.indent();
	      w.line("intentir_cuda_selected_variant_idx = " + std::to_string(i) + ";");
	      w.line("intentir_cuda_selected_variant_tag = \"" + v.suffix + "\";");
	      w.line("intentir_cuda_fastpath_enabled = " + std::string(v.full_tile ? "1" : "0") + ";");
	      w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
	      w.line("dim3 g((unsigned)" + std::to_string(v.grid_x) + ", 1u, 1u);");
	      w.line(kname + "<<<g, b, 0, stream>>>(" + src0_name + ", " + src1_name + ", " + out_name +
             ", " + oc_arg + ", " + ic_arg + ", " + h_arg + ", " + w_arg + ", " + sh_arg + ");");
      w.line("break;");
      w.dedent();
      w.line("}");
    }
    w.line("default: {");
    w.indent();
	    const auto& v0 = variants.front();
	    const std::string k0 = intent.name + "__" + v0.suffix;
	    w.line("intentir_cuda_selected_variant_idx = 0;");
	    w.line("intentir_cuda_selected_variant_tag = \"" + v0.suffix + "\";");
	    w.line("intentir_cuda_fastpath_enabled = " + std::string(v0.full_tile ? "1" : "0") + ";");
	    w.line("dim3 b((unsigned)" + std::to_string(v0.threads) + ", 1u, 1u);");
	    w.line("dim3 g((unsigned)" + std::to_string(v0.grid_x) + ", 1u, 1u);");
	    w.line(k0 + "<<<g, b, 0, stream>>>(" + src0_name + ", " + src1_name + ", " + out_name +
           ", " + oc_arg + ", " + ic_arg + ", " + h_arg + ", " + w_arg + ", " + sh_arg + ");");
    w.line("break;");
    w.dedent();
    w.line("}");
    w.line("}");
    w.dedent();
    w.line("}");
  }

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
  if (host_launch) out["io_spec"]["host_launch"] = true;
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

  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;
  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");
  const bool enable_host_dispatch = want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims;
  const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
  bool host_launch = false;

  int hw_fl = 7;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("hw_fl");
    if (it != op.attrs.end()) {
      if (it->is_number_integer()) hw_fl = it->get<int>();
      else if (it->is_number()) hw_fl = static_cast<int>(it->get<double>());
    }
  }
  if (hw_fl != 7) fail("resize MVP expects hw_fl=7");

  int64_t block_w = binding_int(bindings, "BLOCK_W").value_or(0);
  if (block_w <= 0) {
    const int64_t hinted = resolve_schedule_int(intent, bindings, "tile_n", 0);
    if (0 < hinted && hinted <= 1024) block_w = hinted;
  }
  if (block_w <= 0) {
    // Shape-driven seed (paper-friendly): pick a block width proportional to W.
    // For the AI-Bench resize workload (W=512), this avoids a weak seed (128).
    if (W >= 512)
      block_w = 512;
    else if (W >= 256)
      block_w = 256;
    else if (W >= 128)
      block_w = 128;
    else
      block_w = 64;
  }
  if (block_w < 32) block_w = 32;
  if (block_w > 1024) block_w = 1024;
  if ((block_w % 32) != 0) block_w = ((block_w + 31) / 32) * 32;
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
  const std::string c_unused = std::string("(void)") + (c_is_tensor ? "C_ptr" : "C") + ";";
  const std::string h_unused = std::string("(void)") + (h_is_tensor ? "H_ptr" : "H") + ";";
  const std::string w_unused = std::string("(void)") + (w_is_tensor ? "W_ptr" : "W") + ";";
  const std::string c_arg = c_is_tensor ? "C_ptr" : "C";
  const std::string h_arg = h_is_tensor ? "H_ptr" : "H";
  const std::string w_arg = w_is_tensor ? "W_ptr" : "W";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/resize.cuh\"");
  w.blank();
  emit_selected_api(w);
  if (!enable_host_dispatch) {
    emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
    w.line("extern \"C\" __global__ void " + intent.name + "(const int8_t* __restrict__ " + src_name + ", int8_t* __restrict__ " + out_name +
           ", " + c_param + ", " + h_param + ", " + w_param + ") {");
    w.indent();
    w.line("constexpr int BLOCK_W = " + std::to_string(block_w) + ";");
    if (specialize_dims) {
      w.line(c_unused);
      w.line(h_unused);
      w.line(w_unused);
      w.line("constexpr int C0 = " + std::to_string(C) + ";");
      w.line("constexpr int H0 = " + std::to_string(H) + ";");
      w.line("constexpr int W0 = " + std::to_string(W) + ";");
      const bool full_w_fast = contract_full && ((W % block_w) == 0);
      w.line("intentir_cuda::resize_bilinear2x_i8<BLOCK_W, " + std::string(full_w_fast ? "true" : "false") + ">(" + src_name + ", " + out_name +
             ", C0, H0, W0);");
    } else {
      if (!c_load.empty()) w.line(c_load);
      if (!h_load.empty()) w.line(h_load);
      if (!w_load.empty()) w.line(w_load);
      w.line("intentir_cuda::resize_bilinear2x_i8<BLOCK_W, false>(" + src_name + ", " + out_name + ", C, H, W);");
    }
    w.dedent();
    w.line("}");
  } else {
    host_launch = true;
    struct ResizeVariant {
      int64_t block_w;
      int64_t grid_w;
      bool full_w;
      std::string suffix;
    };
    std::vector<ResizeVariant> variants;
    auto norm_bw = [](int64_t b) -> int64_t {
      if (b < 32) b = 32;
      if (b > 1024) b = 1024;
      if ((b % 32) != 0) b = ((b + 31) / 32) * 32;
      if (b > 1024) b = 1024;
      return b;
	    };
		    auto add_variant = [&](int64_t bw, const std::string& tag) {
		      bw = norm_bw(bw);
		      const int64_t gw = (W + bw - 1) / bw;
		      const bool full = contract_full && ((W % bw) == 0);
	      for (const auto& v : variants) {
	        if (v.block_w == bw) return;
	      }
		      variants.push_back(ResizeVariant{bw, gw, full, tag});
		    };

	    add_variant(block_w, "seed_bw" + std::to_string(block_w));
	    add_variant(block_w / 2, "bw_half");
	    add_variant(block_w * 2, "bw_double");
	    if (!has_evidence) {
	      // Evidence-off: widen the candidate set so selection can recover without priors.
	      add_variant(64, "bw64");
	      add_variant(128, "bw128");
	      add_variant(256, "bw256");
	      add_variant(512, "bw512");
	      add_variant(1024, "bw1024");
	    }
	    if (variants.empty()) add_variant(block_w, "fallback");

	    emit_const_introspection_api(w, (int)variants.size(), has_evidence, contract_level, specialize_dims);

	    for (const auto& v : variants) {
	      const std::string kname = intent.name + "__" + v.suffix;
	      w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.block_w) + ") void " + kname + "(const int8_t* __restrict__ " +
	             src_name + ", int8_t* __restrict__ " + out_name + ", " + c_param + ", " + h_param + ", " + w_param + ") {");
      w.indent();
      w.line("constexpr int BLOCK_W = " + std::to_string(v.block_w) + ";");
      w.line(c_unused);
      w.line(h_unused);
      w.line(w_unused);
	      w.line("constexpr int C0 = " + std::to_string(C) + ";");
	      w.line("constexpr int H0 = " + std::to_string(H) + ";");
	      w.line("constexpr int W0 = " + std::to_string(W) + ";");
	      w.line("intentir_cuda::resize_bilinear2x_i8<BLOCK_W, " + std::string(v.full_w ? "true" : "false") + ">(" + src_name + ", " + out_name +
	             ", C0, H0, W0);");
	      w.dedent();
	      w.line("}");
	      w.blank();
	    }

    w.line("extern \"C\" void " + intent.name + "_host_launch(");
    w.indent();
    w.line("int8_t* " + src_name + ", int8_t* " + out_name + ", " + c_param + ", " + h_param + ", " + w_param + ",");
    w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
    w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
    w.line("int64_t shared_mem, cudaStream_t stream) {");
    w.dedent();
    w.indent();
    w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
    w.line("(void)block_x; (void)block_y; (void)block_z;");
    w.line("(void)shared_mem;");
    w.line(c_unused);
    w.line(h_unused);
    w.line(w_unused);
    w.line("constexpr int C0 = " + std::to_string(C) + ";");
    w.line("constexpr int H0 = " + std::to_string(H) + ";");
    w.line("constexpr int W0 = " + std::to_string(W) + ";");
    w.line("constexpr int OH0 = " + std::to_string(OH) + ";");
    w.line("constexpr bool INTENTIR_HOST_DISPATCH_SELECT = " + std::string(enable_host_dispatch_select ? "true" : "false") + ";");
    w.line("intentir_cuda_dispatch_total_ms = 0.0f;");
    w.line("intentir_cuda_dispatch_evals = 0;");
    w.line("intentir_cuda_fastpath_enabled = 0;");
    w.line("static int intentir_selected = -1;");
    w.line("if (intentir_selected < 0) {");
    w.indent();
    w.line("if (!INTENTIR_HOST_DISPATCH_SELECT) {");
    w.indent();
    w.line("intentir_selected = 0;");
    w.dedent();
    w.line("} else {");
    w.indent();
    w.line("cudaEvent_t start = nullptr;");
    w.line("cudaEvent_t end = nullptr;");
    w.line("TORCH_CHECK(cudaEventCreate(&start) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaEventCreate(&end) == cudaSuccess);");
    w.line("cudaStream_t sel_stream = nullptr;");
    w.line("TORCH_CHECK(cudaStreamCreateWithFlags(&sel_stream, cudaStreamNonBlocking) == cudaSuccess);");
    // Match the benchmark harness: capture a CUDA graph with `iters` launches
    // and time a single replay. This avoids picking variants that only win
    // under Python submission overhead (small kernels).
    w.line("constexpr int warm = 3;");
    w.line("constexpr int iters = 50;");
    w.line("cudaGraph_t graphs[" + std::to_string(variants.size()) + "] = {};");
    w.line("cudaGraphExec_t execs[" + std::to_string(variants.size()) + "] = {};");
    w.line("float ms_acc[" + std::to_string(variants.size()) + "] = {0};");

    // Capture & instantiate graphs for each variant.
    for (size_t i = 0; i < variants.size(); ++i) {
      const auto& v = variants[i];
      const std::string kname = intent.name + "__" + v.suffix;
      w.line("{");
      w.indent();
      w.line("dim3 b((unsigned)" + std::to_string(v.block_w) + ", 1u, 1u);");
      w.line("dim3 g((unsigned)" + std::to_string(v.grid_w) + ", (unsigned)OH0, (unsigned)C0);");
      w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + src_name + ", " + out_name + ", " + c_arg + ", " + h_arg +
             ", " + w_arg + ");");
      w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
      w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + src_name + ", " + out_name + ", " + c_arg + ", " + h_arg +
             ", " + w_arg + ");");
      w.line("TORCH_CHECK(cudaStreamEndCapture(sel_stream, &graphs[" + std::to_string(i) + "]) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphInstantiate(&execs[" + std::to_string(i) + "], graphs[" + std::to_string(i) +
             "], nullptr, nullptr, 0) == cudaSuccess);");
      w.dedent();
      w.line("}");
    }

    // Forward pass then reverse pass to reduce clock/thermal order bias.
    for (size_t i = 0; i < variants.size(); ++i) {
      w.line("{");
      w.indent();
      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
      w.line("float ms = 0.0f;");
      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
      w.line("ms_acc[" + std::to_string(i) + "] += ms;");
      w.dedent();
      w.line("}");
    }
    for (size_t ri = variants.size(); ri-- > 0;) {
      w.line("{");
      w.indent();
      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
      w.line("float ms = 0.0f;");
      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
      w.line("ms_acc[" + std::to_string(ri) + "] += ms;");
      w.dedent();
      w.line("}");
    }

    w.line("float best_ms = 1e30f;");
    w.line("int best_i = 0;");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
    w.indent();
    w.line("const float ms = ms_acc[i];");
    w.line("if (ms < best_ms) { best_ms = ms; best_i = i; }");
    w.dedent();
    w.line("}");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
    w.indent();
    w.line("if (execs[i]) TORCH_CHECK(cudaGraphExecDestroy(execs[i]) == cudaSuccess);");
    w.line("if (graphs[i]) TORCH_CHECK(cudaGraphDestroy(graphs[i]) == cudaSuccess);");
    w.dedent();
    w.line("}");
    w.line("TORCH_CHECK(cudaEventDestroy(start) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaEventDestroy(end) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaStreamDestroy(sel_stream) == cudaSuccess);");
    w.line("intentir_selected = best_i;");
    w.line("float total_ms = 0.0f;");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) total_ms += ms_acc[i];");
    w.line("intentir_cuda_dispatch_total_ms = total_ms;");
    w.line("intentir_cuda_dispatch_evals = " + std::to_string(2 * variants.size()) + ";");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.line("switch (intentir_selected) {");
	    for (size_t i = 0; i < variants.size(); ++i) {
	      const auto& v = variants[i];
	      const std::string kname = intent.name + "__" + v.suffix;
	      w.line("case " + std::to_string(i) + ": {");
	      w.indent();
	      w.line("intentir_cuda_selected_variant_idx = " + std::to_string(i) + ";");
	      w.line("intentir_cuda_selected_variant_tag = \"" + v.suffix + "\";");
	      w.line("intentir_cuda_fastpath_enabled = " + std::string(v.full_w ? "1" : "0") + ";");
	      w.line("dim3 b((unsigned)" + std::to_string(v.block_w) + ", 1u, 1u);");
	      w.line("dim3 g((unsigned)" + std::to_string(v.grid_w) + ", (unsigned)OH0, (unsigned)C0);");
	      w.line(kname + "<<<g, b, 0, stream>>>(" + src_name + ", " + out_name + ", " + c_arg + ", " + h_arg + ", " + w_arg + ");");
      w.line("break;");
      w.dedent();
      w.line("}");
    }
    w.line("default: {");
    w.indent();
	    const auto& v0 = variants.front();
	    const std::string k0 = intent.name + "__" + v0.suffix;
	    w.line("intentir_cuda_selected_variant_idx = 0;");
	    w.line("intentir_cuda_selected_variant_tag = \"" + v0.suffix + "\";");
	    w.line("intentir_cuda_fastpath_enabled = " + std::string(v0.full_w ? "1" : "0") + ";");
	    w.line("dim3 b((unsigned)" + std::to_string(v0.block_w) + ", 1u, 1u);");
	    w.line("dim3 g((unsigned)" + std::to_string(v0.grid_w) + ", (unsigned)OH0, (unsigned)C0);");
	    w.line(k0 + "<<<g, b, 0, stream>>>(" + src_name + ", " + out_name + ", " + c_arg + ", " + h_arg + ", " + w_arg + ");");
    w.line("break;");
    w.dedent();
    w.line("}");
    w.line("}");
    w.dedent();
    w.line("}");
  }

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
  if (host_launch) out["io_spec"]["host_launch"] = true;
  out["launch"] = {{"grid", {grid_w, OH, C}}, {"block", {block_w, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_rope_f32(const Intent& intent, const json& bindings) {
  // Newer pipeline versions may prepend derived scalars as unused `const` ops
  // (e.g. HEAD_DIM_DIV2 = HEAD_DIM // 2). Treat `const* + rope` as equivalent.
  if (intent.ops.empty()) fail("rope lowering expects a rope op");
  if (intent.ops.back().op != "rope") fail("rope lowering expects `const* + rope` (rope last)");
  for (size_t i = 0; i + 1 < intent.ops.size(); ++i) {
    if (intent.ops[i].op != "const") fail("rope lowering expects `const* + rope` (only consts before rope)");
  }
  const Op& op = intent.ops.back();
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

  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;
  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");
  const bool enable_host_dispatch = want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims;
  const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
  bool host_launch = false;

  int64_t heads_per_block = binding_int(bindings, "ROPE_HEADS_PER_BLOCK").value_or(1);
  if (heads_per_block <= 0) heads_per_block = 1;
  if (heads_per_block > 16) heads_per_block = 16;
  if (heads_per_block > H) heads_per_block = H;

  int64_t rope_vec = binding_int(bindings, "ROPE_VEC").value_or(4);
  if (!(rope_vec == 1 || rope_vec == 2 || rope_vec == 4)) rope_vec = 4;
  if (rope_vec == 4 && (half & 3) != 0) rope_vec = ((half & 1) == 0) ? 2 : 1;
  if (rope_vec == 2 && (half & 1) != 0) rope_vec = 1;
  if (!contract_full) rope_vec = 1;

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
    return is_tensor ? (std::string("const int* ") + name + "_ptr") : (std::string("int ") + name + "_in");
  };
  auto dim_load = [&](const char* name, bool is_tensor) -> std::string {
    if (is_tensor) {
      return std::string("const int ") + name + " = " + name + "_ptr ? " + name + "_ptr[0] : 0;";
    }
    return std::string("const int ") + name + " = " + name + "_in;";
  };
  auto dim_unused = [&](const char* name, bool is_tensor) -> std::string {
    return std::string("(void)") + (is_tensor ? (std::string(name) + "_ptr") : (std::string(name) + "_in")) + ";";
  };

  const int64_t iters = std::max<int64_t>(1, (packs + block_x - 1) / block_x);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/rope.cuh\"");
  w.line("#include <math.h>");
  w.blank();
  emit_selected_api(w);
  const std::string seq_arg = seq_is_tensor ? "SEQ_LEN_ptr" : "SEQ_LEN_in";
  const std::string b_arg = b_is_tensor ? "BATCH_NUM_ptr" : "BATCH_NUM_in";
  const std::string h_arg = h_is_tensor ? "HEAD_NUM_ptr" : "HEAD_NUM_in";
  const std::string d_arg = d_is_tensor ? "HEAD_DIM_ptr" : "HEAD_DIM_in";

  if (!enable_host_dispatch) {
    emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
    w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
    w.indent();
    w.line("const float* __restrict__ " + in_name + ", const float* __restrict__ " + cos_name + ", const float* __restrict__ " + sin_name +
           ", float* __restrict__ " + out_name + ",");
    w.line(dim_param("SEQ_LEN", seq_is_tensor) + ", " + dim_param("BATCH_NUM", b_is_tensor) + ", " + dim_param("HEAD_NUM", h_is_tensor) + ", " +
           dim_param("HEAD_DIM", d_is_tensor) + ") {");
    w.dedent();
    w.indent();
    if (specialize_dims) {
      w.line(dim_unused("SEQ_LEN", seq_is_tensor));
      w.line(dim_unused("BATCH_NUM", b_is_tensor));
      w.line(dim_unused("HEAD_NUM", h_is_tensor));
      w.line(dim_unused("HEAD_DIM", d_is_tensor));
      w.line("constexpr int SEQ_LEN = " + std::to_string(SEQ) + ";");
      w.line("constexpr int BATCH_NUM = " + std::to_string(B) + ";");
      w.line("constexpr int HEAD_NUM = " + std::to_string(H) + ";");
      w.line("constexpr int HEAD_DIM = " + std::to_string(D) + ";");
    } else {
      w.line(dim_load("SEQ_LEN", seq_is_tensor));
      w.line(dim_load("BATCH_NUM", b_is_tensor));
      w.line(dim_load("HEAD_NUM", h_is_tensor));
      w.line(dim_load("HEAD_DIM", d_is_tensor));
    }
    w.line("constexpr int HEADS_PER_BLOCK = " + std::to_string(heads_per_block) + ";");
    w.line("constexpr int ROPE_VEC = " + std::to_string(rope_vec) + ";");
    w.line("constexpr int BLOCK_X = " + std::to_string(block_x) + ";");
    w.line("constexpr int ITERS = " + std::to_string(iters) + ";");
    w.line("using idx_t = " + idx_t + ";");
    const bool full_heads_fast = contract_full && (heads_per_block != 1) && ((H % heads_per_block) == 0);
    const bool full_tile_fast = contract_full && (heads_per_block == 1) && ((packs % block_x) == 0);
    w.line("intentir_cuda::rope_f32<HEADS_PER_BLOCK, ROPE_VEC, BLOCK_X, ITERS, " + std::string(full_heads_fast ? "true" : "false") + ", " +
           std::string(full_tile_fast ? "true" : "false") + ", idx_t>(" + in_name + ", " + cos_name + ", " + sin_name + ", " + out_name +
           ", SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM);");
    w.dedent();
    w.line("}");
  } else {
    host_launch = true;
    struct RopeVariant {
      int heads_per_block;
      int rope_vec;
      int64_t block_x;
      int64_t iters;
      int64_t grid_x;
      bool full_heads;
      bool full_tile;
      std::string suffix;
    };
    std::vector<RopeVariant> variants;

    auto legal_rope_vec = [&](int v) -> bool {
      if (!(v == 1 || v == 2 || v == 4)) return false;
      if (v == 4) return (half % 4) == 0;
      if (v == 2) return (half % 2) == 0;
      return true;
    };
    auto norm_block_x = [](int64_t bx) -> int64_t {
      if (bx < 32) bx = 32;
      if (bx > 256) bx = 256;
      if ((bx % 32) != 0) bx = ((bx + 31) / 32) * 32;
      if (bx > 256) bx = 256;
      return bx;
    };
    auto packs_for = [&](int v) -> int64_t {
      if (v == 4) return half / 4;
      if (v == 2) return half / 2;
      return half;
    };
    auto add_variant = [&](int hp, int rv, int64_t bx, const char* prefix = nullptr) {
      if (hp <= 0) hp = 1;
      if (hp > 16) hp = 16;
      if (hp > H) hp = static_cast<int>(H);
      if (!legal_rope_vec(rv)) return;
      bx = norm_block_x(bx);
      const int64_t vpacks = packs_for(rv);
      const int64_t viters = std::max<int64_t>(1, (vpacks + bx - 1) / bx);
      if (viters > 1024) return;
      const int64_t gx = (H + hp - 1) / hp;
      const bool full_heads = contract_full && (hp != 1) && ((H % hp) == 0);
      const bool full_tile = contract_full && (hp == 1) && ((vpacks % bx) == 0);
      for (const auto& v : variants) {
        if (v.heads_per_block == hp && v.rope_vec == rv && v.block_x == bx) return;
      }
      std::ostringstream ss;
      if (prefix && prefix[0]) ss << prefix << "_";
      ss << "hp" << hp << "_v" << rv << "_bx" << bx;
      variants.push_back(RopeVariant{hp, rv, bx, viters, gx, full_heads, full_tile, ss.str()});
    };

    // Evidence-guided tiny search space (avoid huge codegen/compile times):
    // - small set of head groupings (often 1 is best; 2 sometimes helps)
    // - best legal vector width + one fallback
    // - a few warp-aligned thread counts
	    std::vector<int> hp_cands;
	    auto add_hp = [&](int hp) {
	      if (hp <= 0) hp = 1;
      if (hp > 16) hp = 16;
      if (hp > H) hp = static_cast<int>(H);
      for (int x : hp_cands)
        if (x == hp) return;
	      hp_cands.push_back(hp);
	    };
	    add_hp((int)heads_per_block);
	    add_hp(1);
	    add_hp(2);
	    if (!has_evidence) {
	      // Evidence-off: widen head-group candidates.
	      add_hp(4);
	      add_hp(8);
	    }

	    std::vector<int> rv_cands;
	    auto add_rv = [&](int rv) {
	      if (!legal_rope_vec(rv)) return;
      for (int x : rv_cands)
        if (x == rv) return;
	      rv_cands.push_back(rv);
	    };
	    add_rv((int)rope_vec);
	    // Keep a single fallback vec-width across both modes: widening this dimension
	    // blows up codegen/dispatch cost (RoPE already has many variants).
	    if (rope_vec == 4) add_rv(2);
	    else if (rope_vec == 2)
	      add_rv(1);

	    std::vector<int64_t> bx_cands;
	    auto add_bx = [&](int64_t bx) {
	      bx = norm_block_x(bx);
      for (int64_t x : bx_cands)
        if (x == bx) return;
	      bx_cands.push_back(bx);
	    };
	    add_bx(block_x);
	    if (has_evidence) {
	      // Evidence-on: avoid bloating codegen/dispatch; keep a small, stable set.
	      add_bx(128);
	      add_bx(256);
	    } else {
	      // Evidence-off: widen the neighborhood.
	      add_bx(32);
	      add_bx(64);
	      add_bx(128);
	      add_bx(256);
	    }
	    const int64_t sched_threads = resolve_schedule_int(intent, bindings, "tile_n", 0);
	    if (sched_threads > 0) add_bx(sched_threads);

    // Make the ablation semantics crisp: variant[0] is always the seed config
    // derived from (evidence/schedule) defaults; dispatch_off uses this.
    add_variant(static_cast<int>(heads_per_block), static_cast<int>(rope_vec), block_x, "seed");

    for (int hp : hp_cands) {
      for (int rv : rv_cands) {
        for (int64_t bx : bx_cands) {
          add_variant(hp, rv, bx);
        }
      }
    }
    if (variants.empty()) add_variant(static_cast<int>(heads_per_block), static_cast<int>(rope_vec), block_x);

    emit_const_introspection_api(w, (int)variants.size(), has_evidence, contract_level, specialize_dims);

    for (const auto& v : variants) {
      const std::string kname = intent.name + "__" + v.suffix;
      w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.block_x) + ") void " + kname + "(");
      w.indent();
      w.line("const float* __restrict__ " + in_name + ", const float* __restrict__ " + cos_name + ", const float* __restrict__ " + sin_name +
             ", float* __restrict__ " + out_name + ",");
      w.line(dim_param("SEQ_LEN", seq_is_tensor) + ", " + dim_param("BATCH_NUM", b_is_tensor) + ", " + dim_param("HEAD_NUM", h_is_tensor) + ", " +
             dim_param("HEAD_DIM", d_is_tensor) + ") {");
      w.dedent();
      w.indent();
      w.line(dim_unused("SEQ_LEN", seq_is_tensor));
      w.line(dim_unused("BATCH_NUM", b_is_tensor));
      w.line(dim_unused("HEAD_NUM", h_is_tensor));
      w.line(dim_unused("HEAD_DIM", d_is_tensor));
      w.line("constexpr int SEQ_LEN = " + std::to_string(SEQ) + ";");
      w.line("constexpr int BATCH_NUM = " + std::to_string(B) + ";");
      w.line("constexpr int HEAD_NUM = " + std::to_string(H) + ";");
      w.line("constexpr int HEAD_DIM = " + std::to_string(D) + ";");
      w.line("constexpr int HEADS_PER_BLOCK = " + std::to_string(v.heads_per_block) + ";");
      w.line("constexpr int ROPE_VEC = " + std::to_string(v.rope_vec) + ";");
      w.line("constexpr int BLOCK_X = " + std::to_string(v.block_x) + ";");
      w.line("constexpr int ITERS = " + std::to_string(v.iters) + ";");
      w.line("using idx_t = " + idx_t + ";");
      w.line("intentir_cuda::rope_f32<HEADS_PER_BLOCK, ROPE_VEC, BLOCK_X, ITERS, " + std::string(v.full_heads ? "true" : "false") + ", " +
             std::string(v.full_tile ? "true" : "false") + ", idx_t>(" + in_name + ", " + cos_name + ", " + sin_name + ", " + out_name +
             ", SEQ_LEN, BATCH_NUM, HEAD_NUM, HEAD_DIM);");
      w.dedent();
      w.line("}");
      w.blank();
    }

    w.line("extern \"C\" void " + intent.name + "_host_launch(");
    w.indent();
    w.line("float* " + in_name + ", float* " + cos_name + ", float* " + sin_name + ", float* " + out_name + ",");
    w.line(dim_param("SEQ_LEN", seq_is_tensor) + ", " + dim_param("BATCH_NUM", b_is_tensor) + ", " + dim_param("HEAD_NUM", h_is_tensor) + ", " +
           dim_param("HEAD_DIM", d_is_tensor) + ",");
    w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
    w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
    w.line("int64_t shared_mem, cudaStream_t stream) {");
    w.dedent();
    w.indent();
    w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
    w.line("(void)block_x; (void)block_y; (void)block_z;");
    w.line("(void)shared_mem;");
    w.line(dim_unused("SEQ_LEN", seq_is_tensor));
    w.line(dim_unused("BATCH_NUM", b_is_tensor));
    w.line(dim_unused("HEAD_NUM", h_is_tensor));
    w.line(dim_unused("HEAD_DIM", d_is_tensor));
    w.line("constexpr int SEQ_LEN = " + std::to_string(SEQ) + ";");
    w.line("constexpr int BATCH_NUM = " + std::to_string(B) + ";");
    w.line("constexpr int HEAD_NUM = " + std::to_string(H) + ";");
    w.line("constexpr int HEAD_DIM = " + std::to_string(D) + ";");
    w.line("constexpr bool INTENTIR_HOST_DISPATCH_SELECT = " + std::string(enable_host_dispatch_select ? "true" : "false") + ";");
    w.line("intentir_cuda_dispatch_total_ms = 0.0f;");
    w.line("intentir_cuda_dispatch_evals = 0;");
    w.line("intentir_cuda_fastpath_enabled = 0;");
    w.line("static int intentir_selected = -1;");
    w.line("if (intentir_selected < 0) {");
    w.indent();
    w.line("if (!INTENTIR_HOST_DISPATCH_SELECT) {");
    w.indent();
    w.line("intentir_selected = 0;");
    w.dedent();
    w.line("} else {");
    w.indent();
    w.line("cudaEvent_t start = nullptr;");
    w.line("cudaEvent_t end = nullptr;");
    w.line("TORCH_CHECK(cudaEventCreate(&start) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaEventCreate(&end) == cudaSuccess);");
    w.line("cudaStream_t sel_stream = nullptr;");
    w.line("TORCH_CHECK(cudaStreamCreateWithFlags(&sel_stream, cudaStreamNonBlocking) == cudaSuccess);");
    // Match the benchmark harness: capture a CUDA graph with `iters` launches
    // and time a single replay. This avoids picking variants that only win
    // under Python submission overhead (small kernels).
    w.line("constexpr int warm = 3;");
    w.line("constexpr int iters = 50;");
    w.line("cudaGraph_t graphs[" + std::to_string(variants.size()) + "] = {};");
    w.line("cudaGraphExec_t execs[" + std::to_string(variants.size()) + "] = {};");
    w.line("float ms_acc[" + std::to_string(variants.size()) + "] = {0};");

    // Capture & instantiate graphs for each variant.
    for (size_t i = 0; i < variants.size(); ++i) {
      const auto& v = variants[i];
      const std::string kname = intent.name + "__" + v.suffix;
      w.line("{");
      w.indent();
      w.line("dim3 b((unsigned)" + std::to_string(v.block_x) + ", 1u, 1u);");
      w.line("dim3 g((unsigned)" + std::to_string(v.grid_x) + ", (unsigned)BATCH_NUM, (unsigned)SEQ_LEN);");
      w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + in_name + ", " + cos_name + ", " + sin_name + ", " + out_name +
             ", " + seq_arg + ", " + b_arg + ", " + h_arg + ", " + d_arg + ");");
      w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
      w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + in_name + ", " + cos_name + ", " + sin_name + ", " + out_name +
             ", " + seq_arg + ", " + b_arg + ", " + h_arg + ", " + d_arg + ");");
      w.line("TORCH_CHECK(cudaStreamEndCapture(sel_stream, &graphs[" + std::to_string(i) + "]) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphInstantiate(&execs[" + std::to_string(i) + "], graphs[" + std::to_string(i) +
             "], nullptr, nullptr, 0) == cudaSuccess);");
      w.dedent();
      w.line("}");
    }

    // Forward pass then reverse pass to reduce clock/thermal order bias.
    for (size_t i = 0; i < variants.size(); ++i) {
      w.line("{");
      w.indent();
      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
      w.line("float ms = 0.0f;");
      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
      w.line("ms_acc[" + std::to_string(i) + "] += ms;");
      w.dedent();
      w.line("}");
    }
    for (size_t ri = variants.size(); ri-- > 0;) {
      w.line("{");
      w.indent();
      w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) + "], sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
      w.line("float ms = 0.0f;");
      w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
      w.line("ms_acc[" + std::to_string(ri) + "] += ms;");
      w.dedent();
      w.line("}");
    }

    w.line("float best_ms = 1e30f;");
    w.line("int best_i = 0;");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
    w.indent();
    w.line("const float ms = ms_acc[i];");
    w.line("if (ms < best_ms) { best_ms = ms; best_i = i; }");
    w.dedent();
    w.line("}");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
    w.indent();
    w.line("if (execs[i]) TORCH_CHECK(cudaGraphExecDestroy(execs[i]) == cudaSuccess);");
    w.line("if (graphs[i]) TORCH_CHECK(cudaGraphDestroy(graphs[i]) == cudaSuccess);");
    w.dedent();
    w.line("}");
    w.line("TORCH_CHECK(cudaEventDestroy(start) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaEventDestroy(end) == cudaSuccess);");
    w.line("TORCH_CHECK(cudaStreamDestroy(sel_stream) == cudaSuccess);");
    w.line("intentir_selected = best_i;");
    w.line("float total_ms = 0.0f;");
    w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) total_ms += ms_acc[i];");
    w.line("intentir_cuda_dispatch_total_ms = total_ms;");
    w.line("intentir_cuda_dispatch_evals = " + std::to_string(2 * variants.size()) + ";");
    w.dedent();
    w.line("}");
    w.dedent();
    w.line("}");
    w.line("switch (intentir_selected) {");
    for (size_t i = 0; i < variants.size(); ++i) {
      const auto& v = variants[i];
      const std::string kname = intent.name + "__" + v.suffix;
	      w.line("case " + std::to_string(i) + ": {");
	      w.indent();
	      w.line("intentir_cuda_selected_variant_idx = " + std::to_string(i) + ";");
	      w.line("intentir_cuda_selected_variant_tag = \"" + v.suffix + "\";");
	      w.line("intentir_cuda_fastpath_enabled = " + std::string((v.full_heads || v.full_tile) ? "1" : "0") + ";");
	      w.line("dim3 b((unsigned)" + std::to_string(v.block_x) + ", 1u, 1u);");
	      w.line("dim3 g((unsigned)" + std::to_string(v.grid_x) + ", (unsigned)BATCH_NUM, (unsigned)SEQ_LEN);");
	      w.line(kname + "<<<g, b, 0, stream>>>(" + in_name + ", " + cos_name + ", " + sin_name + ", " + out_name + ", " + seq_arg + ", " + b_arg +
             ", " + h_arg + ", " + d_arg + ");");
      w.line("break;");
      w.dedent();
      w.line("}");
    }
    w.line("default: {");
    w.indent();
	    const auto& v0 = variants.front();
	    const std::string k0 = intent.name + "__" + v0.suffix;
	    w.line("intentir_cuda_selected_variant_idx = 0;");
	    w.line("intentir_cuda_selected_variant_tag = \"" + v0.suffix + "\";");
	    w.line("intentir_cuda_fastpath_enabled = " + std::string((v0.full_heads || v0.full_tile) ? "1" : "0") + ";");
	    w.line("dim3 b((unsigned)" + std::to_string(v0.block_x) + ", 1u, 1u);");
	    w.line("dim3 g((unsigned)" + std::to_string(v0.grid_x) + ", (unsigned)BATCH_NUM, (unsigned)SEQ_LEN);");
	    w.line(k0 + "<<<g, b, 0, stream>>>(" + in_name + ", " + cos_name + ", " + sin_name + ", " + out_name + ", " + seq_arg + ", " + b_arg + ", " +
           h_arg + ", " + d_arg + ");");
    w.line("break;");
    w.dedent();
    w.line("}");
    w.line("}");
    w.dedent();
    w.line("}");
  }

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
  if (host_launch) out["io_spec"]["host_launch"] = true;
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
