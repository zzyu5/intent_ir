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

