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
    // Throughput-oriented default: prefer more threads for large vectors.
    block_x = (n >= (1LL << 20)) ? 256 : 128;
  }
  // Keep block size warp-aligned.
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  if (block_x > 1024) block_x = 1024;

  int ept = 8;
  bool ept_user = false;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("elements_per_thread");
    if (it != op.attrs.end()) {
      if (it->is_number_integer()) ept = it->get<int>();
      else if (it->is_number()) ept = static_cast<int>(it->get<double>());
      ept_user = true;
    }
  }
  if (!ept_user) {
    if (auto v = binding_int(bindings, "DROPOUT_EPT")) {
      ept = static_cast<int>(*v);
      ept_user = true;
    }
  }
  if (ept <= 0) ept = 1;
  if (ept > 8) ept = 8;

  // Shape-driven seed stabilization: for very large vectors, avoid a too-large
  // (threads*EPT) tile that produces too few blocks and underutilizes the GPU.
  //
  // We keep this conservative (only adjust when EPT wasn't explicitly set) so
  // the user's schedule/tune overrides still win.
  if (!ept_user && !respect_schedule) {
    constexpr int64_t kMinBlocks = 1024;
    int best = ept;
    if (n >= (1LL << 20)) {
      for (int cand : {4, 2, 1}) {
        const int64_t tile = block_x * static_cast<int64_t>(cand);
        if (tile <= 0) continue;
        const int64_t gx = (n + tile - 1) / tile;
        if (gx >= kMinBlocks) {
          best = cand;
          break;
        }
      }
    }
    ept = best;
  }

  const int64_t denom = block_x * static_cast<int64_t>(ept);
  const int64_t grid_x = (n + denom - 1) / denom;

  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");

  const bool p_is_scalar = is_scalar_tensor(intent, p_name, "f32");
  const bool seed_is_scalar = is_scalar_tensor(intent, seed_name, "i32");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <cuda_runtime.h>");
  w.line("#include \"kernels/dropout.cuh\"");
  w.blank();
  emit_selected_api(w);
  const bool enable_host_dispatch = want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims;
  const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
  bool host_launch = false;

  if (enable_host_dispatch) {
    host_launch = true;
    // Default to enabling vec4 variants: they are guarded by runtime alignment
    // checks in the kernel and reduce instruction count for large vectors.
    // Can be disabled via CUDA_DROPOUT_VEC4=0 for debugging.
    const bool enable_vec4 = binding_int(bindings, "CUDA_DROPOUT_VEC4").value_or(1) != 0;
    struct DropoutVariant {
      int64_t threads;
      int ept;
      int64_t grid_x;
      bool full_tile;
      bool vec4;
      std::string suffix;
    };
    std::vector<DropoutVariant> variants;
    auto norm_threads = [](int64_t t) -> int64_t {
      if (t < 32) t = 32;
      if (t > 1024) t = 1024;
      if ((t % 32) != 0) t = ((t + 31) / 32) * 32;
      if (t > 1024) t = 1024;
      return t;
    };
    auto add_variant = [&](int64_t threads, int vept, bool vec4, const std::string& tag) {
      threads = norm_threads(threads);
      if (vept <= 0) vept = 1;
      if (vept > 8) vept = 8;
      if (vec4 && ((vept % 4) != 0)) return;
      const int64_t tile = threads * (int64_t)vept;
      if (tile <= 0) return;
      const int64_t gx = (n + tile - 1) / tile;
      const bool full = ((n % tile) == 0);
      for (const auto& e : variants) {
        if (e.threads == threads && e.ept == vept && e.vec4 == vec4) return;
      }
      variants.push_back(DropoutVariant{threads, vept, gx, full, vec4, tag});
    };

    auto add_variant_pair = [&](int64_t threads, int vept, const std::string& tag) {
      add_variant(threads, vept, /*vec4=*/false, tag);
      if (enable_vec4 && ((vept % 4) == 0)) add_variant(threads, vept, /*vec4=*/true, tag + "_v4");
    };

    // Evidence-guided candidate set:
    // - Variant[0] is always the "seed" (dispatch_off uses this).
    // - Then, derive a small cross-product of (threads × EPT) candidates from:
    //   (a) the seed schedule/defaults, and (b) certificate stride evidence.
    //
    // This is the research-friendly story: evidence doesn't "find the best tile"
    // by itself, but it *shrinks and shapes* the tuning space so host-dispatch
    // selection is cheaper and more portable.
    add_variant_pair(block_x, ept, "seed");

    int64_t contig = 0;
    if (auto aw = access_witness_meta(intent)) {
      auto it = aw->axis_contig_len.find("n_elements");
      if (it != aw->axis_contig_len.end()) contig = it->second;
      if (contig <= 0 && aw->dominant_axis == "n_elements") contig = aw->dominant_range_len;
    }

    auto add_ept = [&](std::vector<int>& xs, int v) {
      if (v <= 0) return;
      if (v > 8) v = 8;
      for (int x : xs) {
        if (x == v) return;
      }
      xs.push_back(v);
    };
    int max_ept_from_evidence = 1;
    if (contig >= 32)
      max_ept_from_evidence = 8;
    else if (contig >= 16)
      max_ept_from_evidence = 4;
    else if (contig >= 8)
      max_ept_from_evidence = 2;
    else
      max_ept_from_evidence = 1;

    std::vector<int> ept_cands;
    add_ept(ept_cands, ept);
    if (has_evidence) {
      // Evidence-on: keep a tight EPT neighborhood derived from the witness.
      add_ept(ept_cands, max_ept_from_evidence);
      if (max_ept_from_evidence > 1) add_ept(ept_cands, max_ept_from_evidence / 2);
    } else {
      // Evidence-off: widen the candidate set to hedge against unknown contiguity.
      for (int cand : {1, 2, 4, 8}) add_ept(ept_cands, cand);
    }
    // Large vectors are typically bandwidth/RNG dominated; keep 4 as a stable ILP anchor.
    if (n >= (1LL << 20)) add_ept(ept_cands, 4);

    auto add_thread = [&](std::vector<int64_t>& xs, int64_t t) {
      t = norm_threads(t);
      for (int64_t x : xs) {
        if (x == t) return;
      }
      xs.push_back(t);
    };
    std::vector<int64_t> thread_cands;
    add_thread(thread_cands, block_x);
    if (has_evidence) {
      // Evidence-on: restrict threads to a small, stable set near the seed.
      add_thread(thread_cands, block_x / 2);
      add_thread(thread_cands, 128);
      add_thread(thread_cands, 256);
    } else {
      // Evidence-off: explore a wider neighborhood so host dispatch can recover
      // a good configuration without certificate priors.
      add_thread(thread_cands, block_x / 2);
      add_thread(thread_cands, block_x * 2);
      add_thread(thread_cands, 64);
      add_thread(thread_cands, 128);
      add_thread(thread_cands, 256);
      add_thread(thread_cands, 512);
      add_thread(thread_cands, 1024);
    }

    for (int64_t t : thread_cands) {
      for (int vept : ept_cands) {
        if (t == block_x && vept == ept) continue;  // already added as seed
        add_variant_pair(t, vept, "t" + std::to_string(t) + "_e" + std::to_string(vept));
      }
    }
    if (variants.empty()) add_variant_pair(block_x, ept, "fallback");

    emit_const_introspection_api(w, (int)variants.size(), has_evidence, contract_level, specialize_dims);

    for (const auto& v : variants) {
      const std::string kname = intent.name + "__" + v.suffix;
      if (p_is_scalar && seed_is_scalar) {
        w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.threads) + ") void " + kname +
               "(const float* X, float p, int seed, float* Y, int64_t n_elements_in) {");
        w.indent();
        w.line("(void)n_elements_in;");
        w.line("constexpr int64_t n_elements = " + std::to_string(n) + "LL;");
        w.line("constexpr int EPT = " + std::to_string(v.ept) + ";");
        w.line("constexpr int N_ROUNDS = " + std::to_string(rounds) + ";");
        if (contract_full && v.full_tile) {
          if (v.vec4) {
            w.line("intentir_cuda::dropout_f32_vec4<EPT, N_ROUNDS, true>(X, p, (uint32_t)seed, Y, n_elements);");
          } else {
            w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, true>(X, p, (uint32_t)seed, Y, n_elements);");
          }
        } else {
          if (v.vec4) {
            w.line("intentir_cuda::dropout_f32_vec4<EPT, N_ROUNDS, false>(X, p, (uint32_t)seed, Y, n_elements);");
          } else {
            w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, false>(X, p, (uint32_t)seed, Y, n_elements);");
          }
        }
        w.dedent();
        w.line("}");
      } else {
        w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.threads) + ") void " + kname +
               "(const float* X, const float* p_ptr, const int* seed_ptr, float* Y, int64_t n_elements_in) {");
        w.indent();
        w.line("(void)n_elements_in;");
        w.line("constexpr int64_t n_elements = " + std::to_string(n) + "LL;");
        w.line("constexpr int EPT = " + std::to_string(v.ept) + ";");
        w.line("constexpr int N_ROUNDS = " + std::to_string(rounds) + ";");
        if (contract_full && v.full_tile) {
          if (v.vec4) {
            w.line("intentir_cuda::dropout_f32_vec4<EPT, N_ROUNDS, true>(X, p_ptr, seed_ptr, Y, n_elements);");
          } else {
            w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, true>(X, p_ptr, seed_ptr, Y, n_elements);");
          }
        } else {
          if (v.vec4) {
            w.line("intentir_cuda::dropout_f32_vec4<EPT, N_ROUNDS, false>(X, p_ptr, seed_ptr, Y, n_elements);");
          } else {
            w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, false>(X, p_ptr, seed_ptr, Y, n_elements);");
          }
        }
        w.dedent();
        w.line("}");
      }
      w.blank();
    }

    // Host dispatcher: pick the best variant once (evidence-guided, small search space).
    if (p_is_scalar && seed_is_scalar) {
      w.line("extern \"C\" void " + intent.name + "_host_launch(");
      w.indent();
      w.line("float* X, float p, int seed, float* Y, int64_t n_elements_in,");
      w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
      w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
      w.line("int64_t shared_mem, cudaStream_t stream) {");
      w.dedent();
      w.indent();
      w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
      w.line("(void)block_x; (void)block_y; (void)block_z;");
      w.line("(void)shared_mem;");
      w.line("(void)n_elements_in;");
      w.line("constexpr int64_t n_elements = " + std::to_string(n) + "LL;");
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
        w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(X, p, seed, Y, n_elements);");
        w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
        w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(X, p, seed, Y, n_elements);");
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
        w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) +
               "], sel_stream) == cudaSuccess);");
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
        w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) +
               "], sel_stream) == cudaSuccess);");
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
        w.line("intentir_cuda_fastpath_enabled = " + std::string((contract_full && v.full_tile) ? "1" : "0") + ";");
        w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
        w.line("dim3 g((unsigned)" + std::to_string(v.grid_x) + ", 1u, 1u);");
        w.line(kname + "<<<g, b, 0, stream>>>(X, p, seed, Y, n_elements);");
        w.line("break;");
        w.dedent();
        w.line("}");
      }
      w.line("default: {");
      w.indent();
      const auto& v0 = variants.empty() ? DropoutVariant{block_x, ept, grid_x, (n % denom) == 0, /*vec4=*/false, "fallback"} : variants[0];
      const std::string k0 = intent.name + "__" + v0.suffix;
      w.line("intentir_cuda_selected_variant_idx = 0;");
      w.line("intentir_cuda_selected_variant_tag = \"" + v0.suffix + "\";");
      w.line("intentir_cuda_fastpath_enabled = " + std::string((contract_full && v0.full_tile) ? "1" : "0") + ";");
      w.line("dim3 b((unsigned)" + std::to_string(v0.threads) + ", 1u, 1u);");
      w.line("dim3 g((unsigned)" + std::to_string(v0.grid_x) + ", 1u, 1u);");
      w.line(k0 + "<<<g, b, 0, stream>>>(X, p, seed, Y, n_elements);");
      w.line("break;");
      w.dedent();
      w.line("}");
      w.line("}");
      w.dedent();
      w.line("}");
    } else {
      w.line("extern \"C\" void " + intent.name + "_host_launch(");
      w.indent();
      w.line("float* X, float* p_ptr, int* seed_ptr, float* Y, int64_t n_elements_in,");
      w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
      w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
      w.line("int64_t shared_mem, cudaStream_t stream) {");
      w.dedent();
      w.indent();
      w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
      w.line("(void)block_x; (void)block_y; (void)block_z;");
      w.line("(void)shared_mem;");
      w.line("(void)n_elements_in;");
      w.line("constexpr int64_t n_elements = " + std::to_string(n) + "LL;");
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
        w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(X, p_ptr, seed_ptr, Y, n_elements);");
        w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
        w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(X, p_ptr, seed_ptr, Y, n_elements);");
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
        w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(i) +
               "], sel_stream) == cudaSuccess);");
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
        w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(ri) +
               "], sel_stream) == cudaSuccess);");
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
        w.line("intentir_cuda_fastpath_enabled = " + std::string((contract_full && v.full_tile) ? "1" : "0") + ";");
        w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
        w.line("dim3 g((unsigned)" + std::to_string(v.grid_x) + ", 1u, 1u);");
        w.line(kname + "<<<g, b, 0, stream>>>(X, p_ptr, seed_ptr, Y, n_elements);");
        w.line("break;");
        w.dedent();
        w.line("}");
      }
      w.line("default: {");
      w.indent();
      const auto& v0 = variants.empty() ? DropoutVariant{block_x, ept, grid_x, (n % denom) == 0, /*vec4=*/false, "fallback"} : variants[0];
      const std::string k0 = intent.name + "__" + v0.suffix;
      w.line("intentir_cuda_selected_variant_idx = 0;");
      w.line("intentir_cuda_selected_variant_tag = \"" + v0.suffix + "\";");
      w.line("intentir_cuda_fastpath_enabled = " + std::string((contract_full && v0.full_tile) ? "1" : "0") + ";");
      w.line("dim3 b((unsigned)" + std::to_string(v0.threads) + ", 1u, 1u);");
      w.line("dim3 g((unsigned)" + std::to_string(v0.grid_x) + ", 1u, 1u);");
      w.line(k0 + "<<<g, b, 0, stream>>>(X, p_ptr, seed_ptr, Y, n_elements);");
      w.line("break;");
      w.dedent();
      w.line("}");
      w.line("}");
      w.dedent();
      w.line("}");
    }
  } else {
    // Single-kernel fallback path (no host dispatcher).
    emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
    if (p_is_scalar && seed_is_scalar) {
      w.line("extern \"C\" __global__ void " + intent.name +
             "(const float* X, float p, int seed, float* Y, int64_t n_elements_in) {");
      w.indent();
      if (specialize_dims) {
        w.line("(void)n_elements_in;");
        w.line("constexpr int64_t n_elements = " + std::to_string(n) + "LL;");
      } else {
        w.line("const int64_t n_elements = n_elements_in;");
      }
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
      w.line("constexpr int EPT = " + std::to_string(ept) + ";");
      w.line("constexpr int N_ROUNDS = " + std::to_string(rounds) + ";");
      w.line("const int64_t tile = (int64_t)BLOCK_THREADS * (int64_t)EPT;");
      w.line("const bool full_tile = (tile > 0) ? ((n_elements % tile) == 0) : false;");
      w.line(std::string("if (") + (contract_full ? "true" : "false") + " && full_tile) {");
      w.indent();
      w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, true>(X, p, (uint32_t)seed, Y, n_elements);");
      w.dedent();
      w.line("} else {");
      w.indent();
      w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, false>(X, p, (uint32_t)seed, Y, n_elements);");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
    } else {
      w.line("extern \"C\" __global__ void " + intent.name +
             "(const float* X, const float* p_ptr, const int* seed_ptr, float* Y, int64_t n_elements_in) {");
      w.indent();
      if (specialize_dims) {
        w.line("(void)n_elements_in;");
        w.line("constexpr int64_t n_elements = " + std::to_string(n) + "LL;");
      } else {
        w.line("const int64_t n_elements = n_elements_in;");
      }
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
      w.line("constexpr int EPT = " + std::to_string(ept) + ";");
      w.line("constexpr int N_ROUNDS = " + std::to_string(rounds) + ";");
      w.line("const int64_t tile = (int64_t)BLOCK_THREADS * (int64_t)EPT;");
      w.line("const bool full_tile = (tile > 0) ? ((n_elements % tile) == 0) : false;");
      w.line(std::string("if (") + (contract_full ? "true" : "false") + " && full_tile) {");
      w.indent();
      w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, true>(X, p_ptr, seed_ptr, Y, n_elements);");
      w.dedent();
      w.line("} else {");
      w.indent();
      w.line("intentir_cuda::dropout_f32<EPT, N_ROUNDS, false>(X, p_ptr, seed_ptr, Y, n_elements);");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
    }
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
  if (host_launch) out["io_spec"]["host_launch"] = true;
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {Y};
  json out_bindings = bindings;
  if (!out_bindings.contains("n_elements")) out_bindings["n_elements"] = n;
  out["bindings"] = out_bindings;
  return out;
}

