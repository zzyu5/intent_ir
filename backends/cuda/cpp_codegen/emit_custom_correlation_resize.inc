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

