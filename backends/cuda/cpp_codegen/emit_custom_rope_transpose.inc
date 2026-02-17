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
