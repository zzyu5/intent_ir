json emit_matmul_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "matmul")) fail("matmul lowering expects a single matmul op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("matmul expects 2 inputs");
  const std::string a = op.inputs[0];
  const std::string b = op.inputs[1];
  const std::string c = op.output;

  auto a_it = intent.tensors.find(a);
  auto b_it = intent.tensors.find(b);
  auto c_it = intent.tensors.find(c);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || c_it == intent.tensors.end())
    fail("matmul missing A/B/C tensors in intent.tensors");
  if (a_it->second.shape.size() != 2 || b_it->second.shape.size() != 2) fail("matmul expects rank-2 inputs");

  const std::string a_dtype = a_it->second.dtype;
  const std::string b_dtype = b_it->second.dtype;
  const std::string c_dtype = c_it->second.dtype;
  const bool a_supported = (a_dtype == "f32" || a_dtype == "f16");
  const bool b_supported = (b_dtype == "f32" || b_dtype == "f16");
  if (!a_supported || !b_supported) {
    fail("matmul input dtype unsupported for CUDA matmul fallback: A=" + a_dtype + " B=" + b_dtype);
  }
  if (c_dtype != "f32") {
    fail("matmul output dtype unsupported for CUDA matmul fallback: C=" + c_dtype + " (expected f32)");
  }

  const json M_dim = a_it->second.shape[0];
  const json K_dim = a_it->second.shape[1];
  const json K2_dim = b_it->second.shape[0];
  const json N_dim = b_it->second.shape[1];
  if (dim_str(K_dim) != dim_str(K2_dim)) fail("matmul K mismatch between A and B");

  auto M_opt = resolve_dim_token(M_dim, bindings);
  auto N_opt = resolve_dim_token(N_dim, bindings);
  auto K_opt = resolve_dim_token(K_dim, bindings);
  if (!M_opt.has_value() || !N_opt.has_value() || !K_opt.has_value()) fail("matmul missing bindings for M/N/K");
  const int64_t M = *M_opt;
  const int64_t N = *N_opt;
  const int64_t K = *K_opt;

  int64_t block_y = resolve_schedule_int(intent, bindings, "tile_m", 16);
  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 16);
  int64_t block_k = resolve_schedule_int(intent, bindings, "tile_k", 16);
  if (block_x <= 0) block_x = 16;
  if (block_y <= 0) block_y = 16;
  if (block_k <= 0) block_k = 16;
  if (block_x > 64) block_x = 64;

  bool allow_tf32 = binding_int(bindings, "ALLOW_TF32").value_or(0) != 0;
  if (op.attrs.is_object()) {
    auto it = op.attrs.find("allow_tf32");
    if (it != op.attrs.end()) {
      if (it->is_boolean()) allow_tf32 = allow_tf32 || it->get<bool>();
      else if (it->is_number_integer()) allow_tf32 = allow_tf32 || (it->get<int64_t>() != 0);
      else if (it->is_number()) allow_tf32 = allow_tf32 || (it->get<double>() != 0.0);
    }
  }
  const bool use_wmma = (a_dtype == "f32") && (b_dtype == "f32") && (c_dtype == "f32") && allow_tf32 && ((M % 16) == 0) && ((N % 16) == 0) &&
                        ((K % 8) == 0);
  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;

  int64_t thread_m = std::min<int64_t>(16, block_y);
  if (block_x * thread_m > 1024) thread_m = std::max<int64_t>(1, 1024 / std::max<int64_t>(1, block_x));
  block_k = std::max<int64_t>(1, std::min<int64_t>(block_k, std::min<int64_t>(block_x, thread_m)));
  const int64_t rows_per_thread = std::max<int64_t>(1, (block_y + thread_m - 1) / thread_m);

  const int64_t grid_x = (N + block_x - 1) / block_x;
  const int64_t grid_y = (M + block_y - 1) / block_y;

  const std::string M_name = dim_str(M_dim);
  const std::string N_name = dim_str(N_dim);
  const std::string K_name = dim_str(K_dim);
  const bool m_is_tensor = is_scalar_tensor(intent, M_name, "i32");
  const bool n_is_tensor = is_scalar_tensor(intent, N_name, "i32");
  const bool k_is_tensor = is_scalar_tensor(intent, K_name, "i32");

  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");

  const std::string m_param = m_is_tensor ? ("const int* " + M_name + "_ptr") : "int M_in";
  const std::string n_param = n_is_tensor ? ("const int* " + N_name + "_ptr") : "int N_in";
  const std::string k_param = k_is_tensor ? ("const int* " + K_name + "_ptr") : "int K_in";

  const std::string m_init = m_is_tensor ? ("(" + M_name + "_ptr ? " + M_name + "_ptr[0] : 0)") : "M_in";
  const std::string n_init = n_is_tensor ? ("(" + N_name + "_ptr ? " + N_name + "_ptr[0] : 0)") : "N_in";
  const std::string k_init = k_is_tensor ? ("(" + K_name + "_ptr ? " + K_name + "_ptr[0] : 0)") : "K_in";

  std::string mnk_load;
  if (specialize_dims) {
    const std::string m_unused = m_is_tensor ? (M_name + "_ptr") : "M_in";
    const std::string n_unused = n_is_tensor ? (N_name + "_ptr") : "N_in";
    const std::string k_unused = k_is_tensor ? (K_name + "_ptr") : "K_in";
    std::ostringstream ss;
    CodeWriter w(ss);
    w.line("(void)" + m_unused + ";");
    w.line("(void)" + n_unused + ";");
    w.line("(void)" + k_unused + ";");
    w.line("constexpr int M = " + std::to_string(M) + ";");
    w.line("constexpr int N = " + std::to_string(N) + ";");
    w.line("constexpr int K = " + std::to_string(K) + ";");
    mnk_load = ss.str();
  } else {
    if (m_is_tensor || n_is_tensor || k_is_tensor) {
      std::ostringstream ss;
      CodeWriter w(ss);
      w.line("__shared__ int intentir_M;");
      w.line("__shared__ int intentir_N;");
      w.line("__shared__ int intentir_K;");
      w.line("if ((int)threadIdx.x == 0 && (int)threadIdx.y == 0 && (int)threadIdx.z == 0) {");
      w.indent();
      w.line("intentir_M = " + m_init + ";");
      w.line("intentir_N = " + n_init + ";");
      w.line("intentir_K = " + k_init + ";");
      w.dedent();
      w.line("}");
      w.line("__syncthreads();");
      w.line("const int M = intentir_M;");
      w.line("const int N = intentir_N;");
      w.line("const int K = intentir_K;");
      mnk_load = ss.str();
    } else {
      std::ostringstream ss;
      CodeWriter w(ss);
      w.line("const int M = M_in;");
      w.line("const int N = N_in;");
      w.line("const int K = K_in;");
      mnk_load = ss.str();
    }
  }

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  json launch;
  int64_t shared_mem = 0;
  bool host_launch = false;

  if (use_wmma) {
    int64_t wmma_warps_m = binding_int(bindings, "WMMA_WARPS_M").value_or(0);
    int64_t wmma_warps_n = binding_int(bindings, "WMMA_WARPS_N").value_or(0);
    const bool warps_m_override = wmma_warps_m > 0;
    const bool warps_n_override = wmma_warps_n > 0;
    int64_t wmma_frag_m = binding_int(bindings, "WMMA_FRAG_M").value_or(1);
    int64_t wmma_frag_n = binding_int(bindings, "WMMA_FRAG_N").value_or(1);

    if (wmma_warps_m <= 0) wmma_warps_m = std::max<int64_t>(1, std::min<int64_t>(4, block_y / 16));
    if (wmma_warps_n <= 0) wmma_warps_n = std::max<int64_t>(1, std::min<int64_t>(8, block_x / 16));
    if (!warps_n_override && wmma_warps_n <= 1) {
      // Evidence-guided seed: if the access witness gives a labeled contiguous
      // range for N, prefer a smaller WARPS_N to avoid over-provisioning the
      // tile width (helps dispatch_off be representative, not artificially bad).
      int64_t n_contig = 0;
      if (auto aw = access_witness_meta(intent)) {
        auto it_n = aw->axis_contig_len.find(N_name);
        if (it_n != aw->axis_contig_len.end()) n_contig = it_n->second;
        if (n_contig <= 0 && aw->dominant_axis == N_name) n_contig = aw->dominant_range_len;
      }
      if (n_contig > 0) {
        int64_t wn = std::max<int64_t>(1, std::min<int64_t>(4, n_contig / 16));
        if (N >= 64) wn = std::max<int64_t>(2, wn);
        wmma_warps_n = wn;
      } else {
        if (N >= 256)
          wmma_warps_n = 4;
        else if (N >= 64)
          wmma_warps_n = 2;
      }
    }
    if (!warps_m_override && (block_y % 16) != 0) wmma_warps_m = (M >= 64) ? 4 : 2;
    if (!warps_n_override && (block_x % 16) != 0) wmma_warps_n = (N >= 32) ? 2 : 1;
    wmma_warps_m = std::max<int64_t>(1, std::min<int64_t>(4, wmma_warps_m));
    wmma_warps_n = std::max<int64_t>(1, std::min<int64_t>(8, wmma_warps_n));
    if (!warps_m_override && M <= 256) wmma_warps_m = std::max<int64_t>(wmma_warps_m, 2);

    if (wmma_frag_m <= 0) wmma_frag_m = 1;
    if (!(wmma_frag_m == 1 || wmma_frag_m == 2)) wmma_frag_m = 1;
    if ((wmma_warps_m % wmma_frag_m) != 0) wmma_frag_m = 1;
    wmma_warps_m = std::max<int64_t>(1, wmma_warps_m / wmma_frag_m);

    if (wmma_frag_n <= 0) wmma_frag_n = 1;
    if (!(wmma_frag_n == 1 || wmma_frag_n == 2)) wmma_frag_n = 1;
    if ((wmma_warps_n % wmma_frag_n) != 0) wmma_frag_n = 1;
    wmma_warps_n = std::max<int64_t>(1, wmma_warps_n / wmma_frag_n);

    while ((wmma_warps_m * wmma_warps_n) > 16) {
      if (wmma_warps_n > 1)
        wmma_warps_n -= 1;
      else if (wmma_warps_m > 1)
        wmma_warps_m -= 1;
      else
        break;
    }
    if (!warps_m_override && M <= 256 && wmma_warps_m > 2) wmma_warps_m = 2;

    int64_t wmma_stage_k = binding_int(bindings, "WMMA_STAGE_K").value_or(0);
    if (wmma_stage_k <= 0) wmma_stage_k = (K >= 256) ? 64 : 32;
    if (!(wmma_stage_k == 8 || wmma_stage_k == 16 || wmma_stage_k == 32 || wmma_stage_k == 64 || wmma_stage_k == 128)) wmma_stage_k = 32;
    if ((wmma_stage_k % 8) != 0 || (wmma_stage_k % 4) != 0) wmma_stage_k = 32;
    if ((K % wmma_stage_k) != 0) wmma_stage_k = 16;
    if ((K % wmma_stage_k) != 0) wmma_stage_k = 8;

    const int64_t wmma_tile_m = 16 * wmma_warps_m * wmma_frag_m;
    const int64_t wmma_tile_n = 16 * wmma_warps_n * wmma_frag_n;
    const int64_t wmma_grid_x = (N + wmma_tile_n - 1) / wmma_tile_n;
    const int64_t wmma_grid_y = (M + wmma_tile_m - 1) / wmma_tile_m;

    auto policy_norm = [&](const char* key) -> std::string {
      if (!bindings.is_object()) return "";
      auto it = bindings.find(key);
      if (it == bindings.end()) return "";
      if (!it->is_string()) return "";
      std::string s = ascii_lower(it->get<std::string>());
      if (s == "ca" || s == "cg") return s;
      return "";
    };
    std::string wmma_cp_a_policy = policy_norm("WMMA_CP_A_POLICY");
    std::string wmma_cp_b_policy = policy_norm("WMMA_CP_B_POLICY");
    if (wmma_cp_a_policy.empty()) wmma_cp_a_policy = (wmma_warps_n >= wmma_warps_m) ? "ca" : "cg";
    if (wmma_cp_b_policy.empty()) wmma_cp_b_policy = (wmma_warps_m > wmma_warps_n) ? "ca" : "cg";

    int64_t use_cp_async_raw = -1;
    if (auto v = binding_int(bindings, "WMMA_USE_CP_ASYNC")) use_cp_async_raw = *v;
    bool wmma_use_cp_async = (use_cp_async_raw < 0) ? true : (use_cp_async_raw != 0);

	    auto wmma_pipe_opt = binding_int(bindings, "WMMA_PIPE_STAGES");
	    const bool pipe_stages_override = wmma_pipe_opt.has_value() && (*wmma_pipe_opt > 0);
	    int64_t wmma_pipe_stages = wmma_pipe_opt.value_or(0);
	    if (!wmma_use_cp_async) {
	      wmma_pipe_stages = 1;
	    } else {
	      const int64_t sched_pipe = (!pipe_stages_override && respect_schedule) ? resolve_schedule_int(intent, bindings, "pipeline_depth", 0) : 0;
	      if (!pipe_stages_override && sched_pipe > 0) wmma_pipe_stages = sched_pipe;
	      if (wmma_pipe_stages <= 0) wmma_pipe_stages = 3;
	      if (!(wmma_pipe_stages == 2 || wmma_pipe_stages == 3 || wmma_pipe_stages == 4)) wmma_pipe_stages = 3;
	    }

    int64_t wmma_as_pad = binding_int(bindings, "WMMA_AS_PAD").value_or(8);
    int64_t wmma_bs_pad = binding_int(bindings, "WMMA_BS_PAD").value_or(8);
    if (wmma_as_pad < 0) wmma_as_pad = 0;
    if (wmma_bs_pad < 0) wmma_bs_pad = 0;
    if (wmma_as_pad > 32) wmma_as_pad = 32;
    if (wmma_bs_pad > 32) wmma_bs_pad = 32;
    if ((wmma_as_pad % 4) != 0) wmma_as_pad = (wmma_as_pad / 4) * 4;
    if ((wmma_bs_pad % 4) != 0) wmma_bs_pad = (wmma_bs_pad / 4) * 4;

    const bool allow_large_smem_variants = (!respect_schedule) && specialize_dims && (!m_is_tensor) && (!n_is_tensor) && (!k_is_tensor);
    int64_t max_smem_optin = binding_int(bindings, "CUDA_MAX_SMEM_OPTIN").value_or(0);
    if (max_smem_optin <= 0) max_smem_optin = allow_large_smem_variants ? (256 * 1024) : (96 * 1024);

    auto wmma_smem_bytes = [&](int64_t stage_k, int64_t pipe_stages) -> int64_t {
      const int64_t as_ld = stage_k + wmma_as_pad;
      const int64_t bs_ld = wmma_tile_n + wmma_bs_pad;
      return 4LL * (pipe_stages * wmma_tile_m * as_ld + pipe_stages * stage_k * bs_ld);
    };

	    int64_t shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
	    if (shared_bytes > max_smem_optin) {
	      for (int64_t cand : {int64_t(64), int64_t(32), int64_t(16), int64_t(8)}) {
	        if (cand >= wmma_stage_k) continue;
	        if ((K % cand) != 0) continue;
	        if (wmma_smem_bytes(cand, wmma_pipe_stages) <= max_smem_optin) {
	          wmma_stage_k = cand;
	          shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
	          break;
	        }
	      }
	    }
	    // If the selected pipeline depth is too large even after stage_k adjustment,
	    // gracefully shrink it (4->3->2). PIPE_STAGES==1 is treated as sync fallback.
	    if (wmma_pipe_stages == 4 && shared_bytes > max_smem_optin) {
	      wmma_pipe_stages = 3;
	      shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
	    }
	    if (wmma_pipe_stages == 3 && shared_bytes > max_smem_optin) {
	      wmma_pipe_stages = 2;
	      shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
	    }

		    if (!pipe_stages_override && wmma_pipe_stages == 2) {
		      const int64_t bytes3 = wmma_smem_bytes(wmma_stage_k, 3);
		      if (bytes3 <= max_smem_optin) {
		        wmma_pipe_stages = 3;
		        shared_bytes = bytes3;
		      }
		    }
    if (shared_bytes > max_smem_optin) {
      wmma_use_cp_async = false;
      wmma_pipe_stages = 1;
      shared_bytes = wmma_smem_bytes(wmma_stage_k, wmma_pipe_stages);
    }

    shared_bytes = ((shared_bytes + 15) / 16) * 16;
    const bool wmma_disable_fastpath = (binding_int(bindings, "WMMA_DISABLE_FASTPATH").value_or(0) != 0);
    const bool specialize_full_tile = contract_full && specialize_dims && ((M % wmma_tile_m) == 0) && ((N % wmma_tile_n) == 0) &&
                                      ((K % wmma_stage_k) == 0) &&
                                      ((K & 3) == 0) && ((N & 3) == 0) && (!wmma_disable_fastpath);

	    const std::string wmma_cp_a_enum = (wmma_cp_a_policy == "ca") ? "intentir_cuda::CpAsyncPolicy::CA" : "intentir_cuda::CpAsyncPolicy::CG";
	    const std::string wmma_cp_b_enum = (wmma_cp_b_policy == "ca") ? "intentir_cuda::CpAsyncPolicy::CA" : "intentir_cuda::CpAsyncPolicy::CG";
	    const int wmma_cp_a_int = (wmma_cp_a_policy == "ca") ? 0 : 1;
	    const int wmma_cp_b_int = (wmma_cp_b_policy == "ca") ? 0 : 1;
	    const std::string use_cp_async_const = wmma_use_cp_async ? "true" : "false";
	    const std::string enable_fastpath_const = (contract_full && (!wmma_disable_fastpath)) ? "true" : "false";
	    const std::string specialize_full_tile_const = specialize_full_tile ? "true" : "false";

    w.line("#include <cuda_runtime.h>");
    w.line("#include <math.h>");
    w.line("#include <cstdlib>");
    w.line("#include <cstdio>");
    w.line("#include \"kernels/wmma_matmul.cuh\"");
    w.blank();
    emit_selected_api(w);
	    struct WmmaVariant {
	      int64_t warps_m;
	      int64_t warps_n;
	      int64_t frag_m;
	      int64_t frag_n;
	      int64_t tile_m;
	      int64_t tile_n;
	      int64_t stage_k;
	      int64_t pipe_stages;
	      bool use_cp_async;
	      bool specialize_full_tile;
	      int64_t shared_bytes;
	      int64_t threads;
	      int64_t as_pad;
	      int64_t bs_pad;
	      int cp_a_policy;
	      int cp_b_policy;
	      std::string suffix;
	    };

    const bool enable_host_dispatch =
        want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims && (!m_is_tensor) && (!n_is_tensor) && (!k_is_tensor);
    const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
    const int64_t base_threads = 32 * wmma_warps_m * wmma_warps_n;
    const int64_t tile_warps_m = wmma_warps_m * wmma_frag_m;
    const int64_t tile_warps_n = wmma_warps_n * wmma_frag_n;

	    auto make_variant = [&](int64_t warps_m,
	                            int64_t warps_n,
	                            int64_t frag_m,
	                            int64_t frag_n,
	                            int64_t stage_k,
	                            int64_t pipe_stages,
	                            bool use_cp_async,
	                            int64_t as_pad,
	                            int64_t bs_pad,
	                            int cp_a_policy,
	                            int cp_b_policy,
	                            const std::string& suffix) -> std::optional<WmmaVariant> {
	      if (warps_m <= 0 || warps_n <= 0) return std::nullopt;
	      if (frag_m <= 0 || frag_n <= 0) return std::nullopt;
	      if (!(frag_m == 1 || frag_m == 2)) return std::nullopt;
	      if (!(frag_n == 1 || frag_n == 2)) return std::nullopt;
      if (warps_m > 4 || warps_n > 8) return std::nullopt;
      if ((warps_m * warps_n) > 16) return std::nullopt;
      const int64_t threads = 32 * warps_m * warps_n;
      if (threads <= 0 || threads > 1024) return std::nullopt;
	      const int64_t tile_m = 16 * warps_m * frag_m;
	      const int64_t tile_n = 16 * warps_n * frag_n;
	      if (stage_k <= 0) return std::nullopt;
	      if ((stage_k % 8) != 0) return std::nullopt;
	      if ((K % stage_k) != 0) return std::nullopt;
	      if (as_pad < 0 || bs_pad < 0) return std::nullopt;
	      if (as_pad > 32 || bs_pad > 32) return std::nullopt;
	      if ((as_pad % 4) != 0) return std::nullopt;
	      if ((bs_pad % 4) != 0) return std::nullopt;
	      if (!(cp_a_policy == 0 || cp_a_policy == 1)) return std::nullopt;
	      if (!(cp_b_policy == 0 || cp_b_policy == 1)) return std::nullopt;
	      if (use_cp_async) {
	        if (!(pipe_stages == 2 || pipe_stages == 3 || pipe_stages == 4)) return std::nullopt;
	      } else {
	        pipe_stages = 1;
	      }
	      const int64_t as_ld = stage_k + as_pad;
	      const int64_t bs_ld = tile_n + bs_pad;
	      const int64_t smem = 4LL * (pipe_stages * tile_m * as_ld + pipe_stages * stage_k * bs_ld);
	      if (smem > max_smem_optin) return std::nullopt;
		      const bool specialize_full_tile_v = contract_full && specialize_dims && ((M % tile_m) == 0) && ((N % tile_n) == 0) &&
		                                          ((K % stage_k) == 0) && ((K & 3) == 0) &&
		                                          ((N & 3) == 0) && (!wmma_disable_fastpath);
	      WmmaVariant v{warps_m, warps_n, frag_m, frag_n, tile_m, tile_n, stage_k, pipe_stages, use_cp_async, specialize_full_tile_v, smem, threads,
	                    as_pad, bs_pad, cp_a_policy, cp_b_policy, suffix};
	      return v;
	    };

    if (!enable_host_dispatch) {
      // Single-kernel codegen path.
      emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
      w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(base_threads) + ") void " + intent.name + "(");
      w.indent();
      w.line("const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,");
      w.line(m_param + ", " + n_param + ", " + k_param + ") {");
      w.dedent();
      w.indent();
      {
        std::istringstream mnk_ss(mnk_load);
        for (std::string line; std::getline(mnk_ss, line);) {
          if (!line.empty()) w.line(line);
        }
      }
      w.blank();
      w.line("constexpr int WARPS_M = " + std::to_string(wmma_warps_m) + ";");
      w.line("constexpr int WARPS_N = " + std::to_string(wmma_warps_n) + ";");
      w.line("constexpr int FRAG_M = " + std::to_string(wmma_frag_m) + ";");
      w.line("constexpr int FRAG_N = " + std::to_string(wmma_frag_n) + ";");
      w.line("constexpr int STAGE_K = " + std::to_string(wmma_stage_k) + ";");
      w.line("constexpr int AS_PAD = " + std::to_string(wmma_as_pad) + ";");
      w.line("constexpr int BS_PAD = " + std::to_string(wmma_bs_pad) + ";");
      w.line("constexpr int PIPE_STAGES = " + std::to_string(wmma_pipe_stages) + ";");
      w.blank();
      w.line("intentir_cuda::wmma_matmul_f32_tf32<");
      w.indent();
      w.line("WARPS_M,");
      w.line("WARPS_N,");
      w.line("FRAG_M,");
      w.line("FRAG_N,");
      w.line("STAGE_K,");
      w.line("AS_PAD,");
      w.line("BS_PAD,");
      w.line("PIPE_STAGES,");
      w.line(use_cp_async_const + ",");
      w.line(wmma_cp_a_enum + ",");
      w.line(wmma_cp_b_enum + ",");
      w.line(enable_fastpath_const + ",");
      w.line(specialize_full_tile_const + ">(A, B, C, M, N, K);");
      w.dedent();
      w.dedent();
      w.line("}");
      w.blank();
	    } else {
	      host_launch = true;
	      // Multi-version codegen + host dispatcher (paper-friendly: evidence-guided small search).
	      std::vector<WmmaVariant> variants;
	      auto add_variant_ex = [&](int64_t warps_m,
	                                int64_t warps_n,
	                                int64_t frag_m,
	                                int64_t frag_n,
	                                int64_t stage_k,
	                                int64_t pipe_stages,
	                                bool use_cp_async,
	                                int64_t as_pad,
	                                int64_t bs_pad,
	                                int cp_a_policy,
	                                int cp_b_policy,
	                                const std::string& suffix) {
	        auto v =
	            make_variant(warps_m, warps_n, frag_m, frag_n, stage_k, pipe_stages, use_cp_async, as_pad, bs_pad, cp_a_policy, cp_b_policy, suffix);
	        if (!v.has_value()) return;
	        for (const auto& existing : variants) {
	          if (existing.warps_m == v->warps_m && existing.warps_n == v->warps_n && existing.frag_m == v->frag_m && existing.frag_n == v->frag_n &&
	              existing.stage_k == v->stage_k && existing.pipe_stages == v->pipe_stages && existing.use_cp_async == v->use_cp_async &&
	              existing.as_pad == v->as_pad && existing.bs_pad == v->bs_pad && existing.cp_a_policy == v->cp_a_policy && existing.cp_b_policy == v->cp_b_policy)
	            return;
	        }
	        variants.push_back(*v);
	      };
	      auto add_variant = [&](int64_t warps_m,
	                             int64_t warps_n,
	                             int64_t frag_m,
	                             int64_t frag_n,
	                             int64_t stage_k,
	                             int64_t pipe_stages,
	                             bool use_cp_async,
	                             const std::string& suffix) {
	        add_variant_ex(warps_m, warps_n, frag_m, frag_n, stage_k, pipe_stages, use_cp_async, wmma_as_pad, wmma_bs_pad, wmma_cp_a_int, wmma_cp_b_int,
	                       suffix);
	      };

      struct WmmaGeom {
        int64_t warps_m;
        int64_t warps_n;
        int64_t frag_m;
        int64_t frag_n;
        bool is_base_tile;
        std::string tag;
      };
      std::vector<WmmaGeom> geoms;
      auto add_geom = [&](int64_t tile_warps_m,
                          int64_t tile_warps_n,
                          int64_t frag_m,
                          int64_t frag_n,
                          bool is_base_tile,
                          const std::string& tag) {
        if (frag_m <= 0 || frag_n <= 0) return;
        if ((tile_warps_m % frag_m) != 0) return;
        if ((tile_warps_n % frag_n) != 0) return;
        const int64_t warps_m = tile_warps_m / frag_m;
        const int64_t warps_n = tile_warps_n / frag_n;
        if (warps_m <= 0 || warps_n <= 0) return;
        if (warps_m > 4 || warps_n > 8) return;
        if ((warps_m * warps_n) > 16) return;
        for (const auto& g : geoms) {
          if (g.warps_m == warps_m && g.warps_n == warps_n && g.frag_m == frag_m && g.frag_n == frag_n) return;
        }
        geoms.push_back(WmmaGeom{warps_m, warps_n, frag_m, frag_n, is_base_tile, tag});
      };

      const int64_t base_tw_m = tile_warps_m;
      const int64_t base_tw_n = tile_warps_n;
      auto tile_tag = [&](int64_t tw_m, int64_t tw_n) -> std::string {
        std::ostringstream ss;
        ss << "tm" << (16 * tw_m) << "_tn" << (16 * tw_n);
        return ss.str();
      };

      const std::string base_tile_tag = tile_tag(base_tw_m, base_tw_n);
      add_geom(base_tw_m, base_tw_n, 1, 1, /*is_base_tile=*/true, base_tile_tag + "_fm1_fn1");
      add_geom(base_tw_m, base_tw_n, 2, 1, /*is_base_tile=*/true, base_tile_tag + "_fm2_fn1");
      add_geom(base_tw_m, base_tw_n, 1, 2, /*is_base_tile=*/true, base_tile_tag + "_fm1_fn2");
      add_geom(base_tw_m, base_tw_n, 2, 2, /*is_base_tile=*/true, base_tile_tag + "_fm2_fn2");

      auto add_tile_family = [&](int64_t tw_m, int64_t tw_n, bool is_base_tile, bool rich) {
        if (tw_m <= 0 || tw_n <= 0) return;
        if (!rich) {
          add_geom(tw_m, tw_n, 1, 1, /*is_base_tile=*/is_base_tile, tile_tag(tw_m, tw_n) + "_fm1_fn1");
          return;
        }
        const std::string tag = tile_tag(tw_m, tw_n);
        add_geom(tw_m, tw_n, 1, 1, /*is_base_tile=*/is_base_tile, tag + "_fm1_fn1");
        add_geom(tw_m, tw_n, 2, 1, /*is_base_tile=*/is_base_tile, tag + "_fm2_fn1");
        add_geom(tw_m, tw_n, 1, 2, /*is_base_tile=*/is_base_tile, tag + "_fm1_fn2");
        add_geom(tw_m, tw_n, 2, 2, /*is_base_tile=*/is_base_tile, tag + "_fm2_fn2");
      };

      auto add_neighbor_tile = [&](int64_t tw_m, int64_t tw_n) {
        if (tw_m <= 0 || tw_n <= 0) return;
        if (tw_m == base_tw_m && tw_n == base_tw_n) return;
        // For downscaled tiles, keep only the simplest frag config; for upscaled tiles, include a small
        // family so the dispatcher can trade WARPS vs FRAG shape.
        const bool rich = (!has_evidence) ? true : ((tw_m > base_tw_m) || (tw_n > base_tw_n));
        add_tile_family(tw_m, tw_n, /*is_base_tile=*/false, rich);
      };

      // Neighborhood around the schedule-derived base tile:
      //  - downscale (smaller tiles) helps small problems / occupancy
      //  - upscale (larger tiles) helps newer GPUs where a bigger CTA can be profitable
      if (base_tw_m > 1) add_neighbor_tile(base_tw_m / 2, base_tw_n);
      if (base_tw_n > 1) add_neighbor_tile(base_tw_m, base_tw_n / 2);
      if (base_tw_m > 1 && base_tw_n > 1) add_neighbor_tile(base_tw_m / 2, base_tw_n / 2);

      if (base_tw_m * 2 <= 8) add_neighbor_tile(base_tw_m * 2, base_tw_n);
      if (base_tw_n * 2 <= 8) add_neighbor_tile(base_tw_m, base_tw_n * 2);
      if (base_tw_m * 2 <= 8 && base_tw_n * 2 <= 8) add_neighbor_tile(base_tw_m * 2, base_tw_n * 2);

      // Aspect swap candidate (often helps when schedule tile_m/tile_n is imbalanced).
      if (base_tw_m != base_tw_n) add_neighbor_tile(base_tw_n, base_tw_m);
      if (geoms.empty()) geoms.push_back(WmmaGeom{wmma_warps_m, wmma_warps_n, wmma_frag_m, wmma_frag_n, true, "base"});

      const std::string focus_base = base_tile_tag + "_fm1_fn1";
      std::string focus_half;
      if (base_tw_m > 1 && base_tw_n > 1) {
        focus_half = tile_tag(base_tw_m / 2, base_tw_n / 2) + "_fm1_fn1";
      }

      auto cp_tag = [](int p) -> std::string { return (p == 0) ? "ca" : "cg"; };
      const int cp_pol[2] = {0, 1};
      // Keep the micro-search small: either use the default padding or disable it.
      const int64_t pad_pair_as[2] = {wmma_as_pad, 0};
      const int64_t pad_pair_bs[2] = {wmma_bs_pad, 0};

      std::vector<int64_t> focus_stage_cands;
      auto add_focus_stage = [&](int64_t sk) {
        if (sk <= 0) return;
        for (int64_t x : focus_stage_cands) {
          if (x == sk) return;
        }
        focus_stage_cands.push_back(sk);
      };
      add_focus_stage(wmma_stage_k);
      // Evidence-guided priors: reuse TTIR/Certificate schedule hints or witness ranges when present.
      if (auto aw = access_witness_meta(intent)) {
        if (aw->has_contiguous_range && !aw->dominant_axis.empty() && aw->dominant_axis == K_name) {
          if (aw->dominant_range_len > 0) add_focus_stage(aw->dominant_range_len);
        }
      }
      for (int64_t th : schedule_hints_tile_hints(intent)) add_focus_stage(th);
      // Conservative fallbacks for common stage sizes.
      if ((K % 32) == 0) add_focus_stage(32);
      if ((K % 128) == 0) add_focus_stage(128);
      while (focus_stage_cands.size() > 3) focus_stage_cands.pop_back();

      auto add_focus_microsearch = [&](const WmmaGeom& g) {
        for (int64_t sk : focus_stage_cands) {
          if (sk <= 0) continue;
          if (K < sk || (K % sk) != 0) continue;
          for (int pipe : {3, 4}) {
            const std::string base_suffix =
                (sk == wmma_stage_k) ? (g.tag + "_p" + std::to_string(pipe))
                                     : (g.tag + "_k" + std::to_string(sk) + "_p" + std::to_string(pipe));
            for (int pi = 0; pi < 2; ++pi) {
              const int64_t as_pad = pad_pair_as[pi];
              const int64_t bs_pad = pad_pair_bs[pi];
              for (int cp_a : cp_pol) {
                for (int cp_b : cp_pol) {
                  std::string suf = base_suffix;
                  suf += "_ap" + std::to_string(as_pad) + "_bp" + std::to_string(bs_pad);
                  suf += "_a" + cp_tag(cp_a) + "_b" + cp_tag(cp_b);
                  add_variant_ex(g.warps_m,
                                 g.warps_n,
                                 g.frag_m,
                                 g.frag_n,
                                 sk,
                                 pipe,
                                 /*use_cp_async=*/true,
                                 as_pad,
                                 bs_pad,
                                 cp_a,
                                 cp_b,
                                 suf);
                }
              }
            }
          }
        }
      };

      const bool wmma_tiny_search = binding_int(bindings, "WMMA_TINY_SEARCH").value_or(1) != 0;
      if (wmma_tiny_search) {
        // Evidence-guided *tiny* search space (paper-friendly and fast to compile):
        // - Keep only a handful of tiles around the schedule-derived base tile.
        // - Keep only a small set of (stage_k, pipe_stages) candidates derived from
        //   the access witness / schedule hints.
        //
        // This makes the "search space reduction" claim concrete: the IR/certificate
        // provides strong priors, so we don't need a large brute-force autotune set.
        struct TileCand {
          int64_t tw_m;
          int64_t tw_n;
          bool is_base;
        };
        std::vector<TileCand> tiles;
        auto add_tile = [&](int64_t tw_m, int64_t tw_n) {
          if (tw_m <= 0 || tw_n <= 0) return;
          for (const auto& t : tiles) {
            if (t.tw_m == tw_m && t.tw_n == tw_n) return;
          }
          tiles.push_back(TileCand{tw_m, tw_n, (tw_m == base_tw_m && tw_n == base_tw_n)});
        };
        add_tile(base_tw_m, base_tw_n);
        if (base_tw_m > 1) add_tile(base_tw_m / 2, base_tw_n);
        if (base_tw_n > 1) add_tile(base_tw_m, base_tw_n / 2);
        if (base_tw_m > 1 && base_tw_n > 1) add_tile(base_tw_m / 2, base_tw_n / 2);
        // A couple of conservative upscales (still within warp limits) often help newer GPUs.
        if (base_tw_m * 2 <= 4) add_tile(base_tw_m * 2, base_tw_n);
        if (base_tw_n * 2 <= 8) add_tile(base_tw_m, base_tw_n * 2);
        if (base_tw_m != base_tw_n) add_tile(base_tw_n, base_tw_m);
        // If the access witness provides labeled contiguous ranges for M/N, add that as a tile hint.
        // This is stronger than unlabeled `tile_hints` and directly ties the search space to evidence.
        if (auto aw = access_witness_meta(intent)) {
          auto it_m = aw->axis_contig_len.find(M_name);
          auto it_n = aw->axis_contig_len.find(N_name);
          if (it_m != aw->axis_contig_len.end() && it_n != aw->axis_contig_len.end()) {
            const int64_t tm = it_m->second;
            const int64_t tn = it_n->second;
            if ((tm % 16) == 0 && (tn % 16) == 0) {
              add_tile(tm / 16, tn / 16);
            }
          }
        }
        // Seed additional tiles from schedule-hint tile sizes (typically the frontend's BLOCK_M/BLOCK_N).
        // The hints are unlabeled, so we add a few symmetric/one-axis substitutions and let the dispatcher
        // pick the winner.
        std::vector<int64_t> hint_tws;
        for (int64_t h : schedule_hints_tile_hints(intent)) {
          if ((h % 16) != 0) continue;
          const int64_t tw = h / 16;
          if (tw <= 0 || tw > 8) continue;
          bool seen = false;
          for (int64_t x : hint_tws) {
            if (x == tw) {
              seen = true;
              break;
            }
          }
          if (!seen) hint_tws.push_back(tw);
        }
        if (!hint_tws.empty()) {
          std::sort(hint_tws.begin(), hint_tws.end());
          const int64_t lo = hint_tws.front();
          const int64_t hi = hint_tws.back();
          if (lo == hi) {
            add_tile(lo, lo);
          } else {
            // Add both orientations since hints are unlabeled.
            add_tile(lo, hi);
            add_tile(hi, lo);
          }
        }
        if (tiles.empty()) add_tile(base_tw_m, base_tw_n);

        // Pick a very small stage_k candidate set: the current heuristic + the smallest
        // evidence hint that divides K (+ an optional middle ground like 32).
        auto norm_stage = [&](int64_t sk) -> int64_t {
          if (sk <= 0) return 0;
          if ((sk % 8) != 0) return 0;
          if ((K % sk) != 0) return 0;
          if (sk > 128) return 0;
          return sk;
        };
        int64_t hint_stage = 0;
        for (int64_t sk : focus_stage_cands) {
          sk = norm_stage(sk);
          if (sk <= 0) continue;
          if (hint_stage <= 0 || sk < hint_stage) hint_stage = sk;
        }
        std::vector<int64_t> stage_cands;
        auto add_stage = [&](int64_t sk) {
          sk = norm_stage(sk);
          if (sk <= 0) return;
          for (int64_t x : stage_cands) {
            if (x == sk) return;
          }
          stage_cands.push_back(sk);
        };
        add_stage(wmma_stage_k);
        add_stage(hint_stage);
        if (auto bk = binding_int(bindings, "BLOCK_K")) add_stage(*bk);
        add_stage(32);
        if (stage_cands.empty()) add_stage(wmma_stage_k);
        while (stage_cands.size() > 3) stage_cands.pop_back();

        // Pipe depth candidates:
        // - Base tile explores a small neighborhood.
        // - Other tiles keep only the heuristic default to cap compile time.
        std::vector<int> pipe_cands_base;
        std::vector<int> pipe_cands_other;
        auto add_pipe = [&](std::vector<int>& dst, int p) {
          if (!(p == 2 || p == 3 || p == 4)) return;
          for (int x : dst) {
            if (x == p) return;
          }
          dst.push_back(p);
        };
        if (wmma_use_cp_async) {
          const int p0 = (int)wmma_pipe_stages;
          add_pipe(pipe_cands_base, p0);
          if (p0 != 2) add_pipe(pipe_cands_base, 2);
          if (p0 != 3) add_pipe(pipe_cands_base, 3);
          if (p0 != 4) add_pipe(pipe_cands_base, 4);
          while (pipe_cands_base.size() > 3) pipe_cands_base.pop_back();
          add_pipe(pipe_cands_other, p0);
        }

        std::vector<int64_t> stage_cands_base = stage_cands;
        std::vector<int64_t> stage_cands_other;
        if (!stage_cands.empty())
          stage_cands_other.push_back(stage_cands[0]);
        else
          stage_cands_other.push_back(wmma_stage_k);

        for (const auto& t : tiles) {
          // Use only the simplest frag geometry (fm1_fn1) to keep compile/search small.
          const int64_t warps_m = t.tw_m;
          const int64_t warps_n = t.tw_n;
          if (warps_m <= 0 || warps_n <= 0) continue;
          if (warps_m > 4 || warps_n > 8) continue;
          if ((warps_m * warps_n) > 16) continue;
          const std::string base_tag = tile_tag(t.tw_m, t.tw_n) + "_fm1_fn1";

          if (wmma_use_cp_async) {
            const auto& sks = t.is_base ? stage_cands_base : stage_cands_other;
            const auto& ps = t.is_base ? pipe_cands_base : pipe_cands_other;
            if (!ps.empty()) {
              for (int64_t sk : sks) {
                for (int p : ps) {
                std::string suf = base_tag;
                if (sk != wmma_stage_k) suf += "_k" + std::to_string(sk);
                suf += "_p" + std::to_string(p);
                add_variant((int64_t)warps_m, (int64_t)warps_n, /*frag_m=*/1, /*frag_n=*/1, sk, p, /*use_cp_async=*/true, suf);
                }
              }
            }
            // Optional microsearch: for evidence-off runs (no certificate priors), widen the
            // base-tile neighborhood by varying cp.async cache policy and padding on/off. When
            // evidence is present, we keep the candidate set tight so the "evidence shapes the
            // search space" claim is measurable via variant_count/dispatch_evals.
            if (t.is_base && (!has_evidence)) {
              WmmaGeom g{warps_m, warps_n, 1, 1, /*is_base_tile=*/true, base_tag};
              add_focus_microsearch(g);
            }
          }

          // Always include a sync fallback (pipe_stages=1).
          const int64_t sk_sync = stage_cands_other.empty() ? wmma_stage_k : stage_cands_other[0];
          std::string suf_sync = base_tag;
          if (sk_sync != wmma_stage_k) suf_sync += "_k" + std::to_string(sk_sync);
          suf_sync += "_sync";
          add_variant((int64_t)warps_m, (int64_t)warps_n, /*frag_m=*/1, /*frag_n=*/1, sk_sync, 1, /*use_cp_async=*/false, suf_sync);
        }
      } else {
        // Full search (debug): larger candidate set.
        for (const auto& g : geoms) {
          const bool is_focus = (g.tag == focus_base) || (!focus_half.empty() && g.tag == focus_half);
          if (wmma_use_cp_async) {
            add_variant(g.warps_m,
                        g.warps_n,
                        g.frag_m,
                        g.frag_n,
                        wmma_stage_k,
                        wmma_pipe_stages,
                        /*use_cp_async=*/true,
                        g.tag + "_p" + std::to_string(wmma_pipe_stages));

            if (g.is_base_tile) {
              add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, wmma_stage_k, 4, /*use_cp_async=*/true, g.tag + "_p4");
              add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, wmma_stage_k, 3, /*use_cp_async=*/true, g.tag + "_p3");
              add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, wmma_stage_k, 2, /*use_cp_async=*/true, g.tag + "_p2");
              if (K >= 128 && (K % 128) == 0) {
                add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, 128, 4, /*use_cp_async=*/true, g.tag + "_k128_p4");
                add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, 128, 3, /*use_cp_async=*/true, g.tag + "_k128_p3");
                add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, 128, 2, /*use_cp_async=*/true, g.tag + "_k128_p2");
              }
            } else {
              if (wmma_pipe_stages == 3) {
                add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, wmma_stage_k, 2, /*use_cp_async=*/true, g.tag + "_p2");
              }
            }

            // For the most likely winners (base fm1_fn1 and base/2 fm1_fn1), try a small cross-product of
            // cp.async cache policy (CA/CG), padding on/off, stage_k, and PIPE=3/4.
            if (is_focus) {
              // Also include a couple of simpler stage_k alternatives for PIPE=2/3.
              for (int64_t sk : focus_stage_cands) {
                if (sk <= 0) continue;
                if (sk == wmma_stage_k) continue;
                if (K >= sk && (K % sk) == 0) {
                  add_variant(g.warps_m,
                              g.warps_n,
                              g.frag_m,
                              g.frag_n,
                              sk,
                              3,
                              /*use_cp_async=*/true,
                              g.tag + "_k" + std::to_string(sk) + "_p3");
                  add_variant(g.warps_m,
                              g.warps_n,
                              g.frag_m,
                              g.frag_n,
                              sk,
                              2,
                              /*use_cp_async=*/true,
                              g.tag + "_k" + std::to_string(sk) + "_p2");
                }
              }
              add_focus_microsearch(g);
            }
          }
          add_variant(g.warps_m, g.warps_n, g.frag_m, g.frag_n, wmma_stage_k, 1, /*use_cp_async=*/false, g.tag + "_sync");
        }
      }
      if (variants.empty()) add_variant(wmma_warps_m, wmma_warps_n, wmma_frag_m, wmma_frag_n, wmma_stage_k, 1, /*use_cp_async=*/false, "fallback");

      emit_const_introspection_api(w, (int)variants.size(), has_evidence, contract_level, specialize_dims);

      w.line("static const char* intentir_matmul_variant_tags[] = {");
      w.indent();
      for (const auto& v : variants) w.line("\"" + v.suffix + "\",");
      w.dedent();
      w.line("};");
      w.blank();

      for (const auto& v : variants) {
        const std::string kname = intent.name + "__" + v.suffix;
        w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.threads) + ") void " + kname + "(");
        w.indent();
        w.line("const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,");
        w.line(m_param + ", " + n_param + ", " + k_param + ") {");
        w.dedent();
        w.indent();
        {
          std::istringstream mnk_ss(mnk_load);
          for (std::string line; std::getline(mnk_ss, line);) {
            if (!line.empty()) w.line(line);
          }
        }
        w.blank();
	        w.line("constexpr int WARPS_M = " + std::to_string(v.warps_m) + ";");
	        w.line("constexpr int WARPS_N = " + std::to_string(v.warps_n) + ";");
	        w.line("constexpr int FRAG_M = " + std::to_string(v.frag_m) + ";");
	        w.line("constexpr int FRAG_N = " + std::to_string(v.frag_n) + ";");
	        w.line("constexpr int STAGE_K = " + std::to_string(v.stage_k) + ";");
	        w.line("constexpr int AS_PAD = " + std::to_string(v.as_pad) + ";");
	        w.line("constexpr int BS_PAD = " + std::to_string(v.bs_pad) + ";");
	        w.line("constexpr int PIPE_STAGES = " + std::to_string(v.pipe_stages) + ";");
	        w.blank();
	        const std::string use_cp_async_v = v.use_cp_async ? "true" : "false";
	        const std::string specialize_full_tile_v = v.specialize_full_tile ? "true" : "false";
	        const std::string cp_a_enum_v = (v.cp_a_policy == 0) ? "intentir_cuda::CpAsyncPolicy::CA" : "intentir_cuda::CpAsyncPolicy::CG";
	        const std::string cp_b_enum_v = (v.cp_b_policy == 0) ? "intentir_cuda::CpAsyncPolicy::CA" : "intentir_cuda::CpAsyncPolicy::CG";
	        w.line("intentir_cuda::wmma_matmul_f32_tf32<");
	        w.indent();
	        w.line("WARPS_M,");
        w.line("WARPS_N,");
        w.line("FRAG_M,");
        w.line("FRAG_N,");
        w.line("STAGE_K,");
        w.line("AS_PAD,");
	        w.line("BS_PAD,");
	        w.line("PIPE_STAGES,");
	        w.line(use_cp_async_v + ",");
	        w.line(cp_a_enum_v + ",");
	        w.line(cp_b_enum_v + ",");
	        w.line(enable_fastpath_const + ",");
	        w.line(specialize_full_tile_v + ">(A, B, C, M, N, K);");
        w.dedent();
        w.dedent();
        w.line("}");
        w.blank();
      }

      w.line("extern \"C\" void " + intent.name + "_host_launch(");
      w.indent();
      w.line("float* A, float* B, float* C,");
      w.line("int M_in, int N_in, int K_in,");
      w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
      w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
      w.line("int64_t shared_mem, cudaStream_t stream) {");
      w.dedent();
      w.indent();
      w.line("(void)grid_x;");
      w.line("(void)grid_y;");
      w.line("(void)grid_z;");
      w.line("(void)block_x;");
      w.line("(void)block_y;");
      w.line("(void)block_z;");
      w.line("(void)shared_mem;");

      w.line("const int M = M_in;");
      w.line("const int N = N_in;");
      w.line("const int K = K_in;");

      int fallback_i = 0;
      if (!variants.empty()) {
        int64_t best_smem = variants[0].shared_bytes;
        for (size_t i = 1; i < variants.size(); ++i) {
          if (variants[i].shared_bytes < best_smem) {
            best_smem = variants[i].shared_bytes;
            fallback_i = static_cast<int>(i);
          }
        }
      }

      w.line("static int intentir_max_smem_optin = -1;");
      w.line("if (intentir_max_smem_optin < 0) {");
      w.indent();
      w.line("int dev = 0;");
      w.line("cudaError_t e0 = cudaGetDevice(&dev);");
      w.line("if (e0 != cudaSuccess) dev = 0;");
      w.line("int v = 0;");
      w.line("cudaError_t e1 = cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);");
      w.line("if (e1 != cudaSuccess || v <= 0) {");
      w.indent();
      w.line("(void)cudaGetLastError();");
      w.line("v = 0;");
      w.line("cudaError_t e2 = cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlock, dev);");
      w.line("if (e2 != cudaSuccess || v <= 0) v = 49152;");
      w.dedent();
      w.line("}");
      w.line("intentir_max_smem_optin = v;");
      w.dedent();
      w.line("}");

      w.line("constexpr bool INTENTIR_HOST_DISPATCH_SELECT = " + std::string(enable_host_dispatch_select ? "true" : "false") + ";");
      w.line("intentir_cuda_dispatch_total_ms = 0.0f;");
      w.line("intentir_cuda_dispatch_evals = 0;");
      w.line("intentir_cuda_fastpath_enabled = 0;");
      w.line("static int intentir_selected = -1;");
      w.line("static int intentir_smem_cached = -1;");
      w.line("static const void* intentir_kernel_cached = nullptr;");
      w.line("if (intentir_selected < 0) {");
      w.indent();
      w.line("if (!INTENTIR_HOST_DISPATCH_SELECT) {");
      w.indent();
      w.line("int seed_i = -1;");
      for (size_t vi = 0; vi < variants.size(); ++vi) {
        const auto& v = variants[vi];
        const int64_t smem_v = ((v.shared_bytes + 15) / 16) * 16;
        if (vi == 0) {
          w.line("if ((int)" + std::to_string(smem_v) + " <= intentir_max_smem_optin) seed_i = 0;");
        } else {
          w.line("else if ((int)" + std::to_string(smem_v) + " <= intentir_max_smem_optin) seed_i = " + std::to_string(vi) + ";");
        }
      }
      w.line("intentir_selected = (seed_i >= 0) ? seed_i : " + std::to_string(fallback_i) + ";");
      w.dedent();
      w.line("} else {");
      w.indent();
      w.line("cudaEvent_t start = nullptr;");
      w.line("cudaEvent_t end = nullptr;");
      w.line("TORCH_CHECK(cudaEventCreate(&start) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventCreate(&end) == cudaSuccess);");
      w.line("cudaStream_t sel_stream = nullptr;");
      w.line("TORCH_CHECK(cudaStreamCreateWithFlags(&sel_stream, cudaStreamNonBlocking) == cudaSuccess);");
      w.line("constexpr int warm = 3;");
      w.line("constexpr int iters = 50;");
      w.line("float ms_acc[" + std::to_string(variants.size()) + "] = {0};");
      w.line("bool ok[" + std::to_string(variants.size()) + "] = {false};");
      w.line("cudaGraph_t graphs[" + std::to_string(variants.size()) + "] = {};");
      w.line("cudaGraphExec_t execs[" + std::to_string(variants.size()) + "] = {};");

      for (size_t vi = 0; vi < variants.size(); ++vi) {
        const auto& v = variants[vi];
        const std::string kname = intent.name + "__" + v.suffix;
        const int64_t smem_v = ((v.shared_bytes + 15) / 16) * 16;
        w.line("{");
        w.indent();
        w.line("const int smem = (int)" + std::to_string(smem_v) + ";");
        w.line("if (smem <= intentir_max_smem_optin) {");
        w.indent();
        w.line("ok[" + std::to_string(vi) + "] = true;");
        w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
        w.line("dim3 g((unsigned)((N + " + std::to_string(v.tile_n) + " - 1) / " + std::to_string(v.tile_n) + "), "
               "(unsigned)((M + " + std::to_string(v.tile_m) + " - 1) / " + std::to_string(v.tile_m) + "), 1u);");
        w.line("const void* kptr = (const void*)" + kname + ";");
        w.line("if (smem >= 49152 && (intentir_kernel_cached != kptr || intentir_smem_cached != smem)) {");
        w.indent();
        w.line("TORCH_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);");
        w.line("intentir_kernel_cached = kptr;");
        w.line("intentir_smem_cached = smem;");
        w.dedent();
        w.line("}");
        w.line("for (int i = 0; i < warm; ++i) {");
        w.indent();
        w.line(kname + "<<<g, b, (size_t)smem, sel_stream>>>((const float*)A, (const float*)B, (float*)C, M_in, N_in, K_in);");
        w.dedent();
        w.line("}");
        w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
        w.line("for (int i = 0; i < iters; ++i) {");
        w.indent();
        w.line(kname + "<<<g, b, (size_t)smem, sel_stream>>>((const float*)A, (const float*)B, (float*)C, M_in, N_in, K_in);");
        w.dedent();
        w.line("}");
        w.line("TORCH_CHECK(cudaStreamEndCapture(sel_stream, &graphs[" + std::to_string(vi) + "]) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaGraphInstantiate(&execs[" + std::to_string(vi) + "], graphs[" + std::to_string(vi) +
               "], nullptr, nullptr, 0) == cudaSuccess);");
        w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(vi) +
               "], sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(vi) + "], sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
        w.line("float ms = 0.0f;");
        w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
        w.line("ms = ms / (float)iters;");
        w.line("ms_acc[" + std::to_string(vi) + "] += ms;");
        w.dedent();
        w.line("}");
        w.dedent();
        w.line("}");
      }

      for (size_t rvi = variants.size(); rvi-- > 0;) {
        const auto& v = variants[rvi];
        const std::string kname = intent.name + "__" + v.suffix;
        const int64_t smem_v = ((v.shared_bytes + 15) / 16) * 16;
        w.line("{");
        w.indent();
        w.line("const int smem = (int)" + std::to_string(smem_v) + ";");
        w.line("if (smem <= intentir_max_smem_optin) {");
        w.indent();
        w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
        w.line("dim3 g((unsigned)((N + " + std::to_string(v.tile_n) + " - 1) / " + std::to_string(v.tile_n) + "), "
               "(unsigned)((M + " + std::to_string(v.tile_m) + " - 1) / " + std::to_string(v.tile_m) + "), 1u);");
        w.line("const void* kptr = (const void*)" + kname + ";");
        w.line("if (smem >= 49152 && (intentir_kernel_cached != kptr || intentir_smem_cached != smem)) {");
        w.indent();
        w.line("TORCH_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);");
        w.line("intentir_kernel_cached = kptr;");
        w.line("intentir_smem_cached = smem;");
        w.dedent();
        w.line("}");
        w.line("for (int i = 0; i < warm; ++i) TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(rvi) +
               "], sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaEventRecord(start, sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaGraphLaunch(execs[" + std::to_string(rvi) + "], sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaEventRecord(end, sel_stream) == cudaSuccess);");
        w.line("TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess);");
        w.line("float ms = 0.0f;");
        w.line("TORCH_CHECK(cudaEventElapsedTime(&ms, start, end) == cudaSuccess);");
        w.line("ms = ms / (float)iters;");
        w.line("ms_acc[" + std::to_string(rvi) + "] += ms;");
        w.dedent();
        w.line("}");
        w.dedent();
        w.line("}");
      }

      w.line("float best_ms = 1e30f;");
      w.line("int best_i = " + std::to_string(fallback_i) + ";");
      w.line("bool intentir_found = false;");
      w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) {");
      w.indent();
      w.line("if (!ok[i]) continue;");
      w.line("intentir_found = true;");
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
      w.line("TORCH_CHECK(cudaStreamDestroy(sel_stream) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventDestroy(start) == cudaSuccess);");
      w.line("TORCH_CHECK(cudaEventDestroy(end) == cudaSuccess);");
      w.line("intentir_selected = intentir_found ? best_i : " + std::to_string(fallback_i) + ";");
      w.line("float total_ms = 0.0f;");
      w.line("int evals = 0;");
      w.line("for (int i = 0; i < " + std::to_string(variants.size()) + "; ++i) { if (ok[i]) { total_ms += ms_acc[i]; evals += 2; } }");
      w.line("total_ms = total_ms * (float)iters;");
      w.line("intentir_cuda_dispatch_total_ms = total_ms;");
      w.line("intentir_cuda_dispatch_evals = evals;");
      w.line("if (const char* dbg = std::getenv(\"INTENTIR_CUDA_MATMUL_DISPATCH_DEBUG\")) {");
      w.indent();
      w.line("if (dbg[0]) std::fprintf(stderr, \"[intentir][matmul] selected=%d tag=%s best_ms_per_iter=%f (warm=%d iters=%d passes=2)\\n\",");
      w.indent();
      w.line("intentir_selected, intentir_matmul_variant_tags[intentir_selected], (double)(best_ms * 0.5), warm, iters);");
      w.dedent();
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");
      w.dedent();
      w.line("}");

      w.line("switch (intentir_selected) {");
      for (size_t vi = 0; vi < variants.size(); ++vi) {
        const auto& v = variants[vi];
        const std::string kname = intent.name + "__" + v.suffix;
        const int64_t smem_v = ((v.shared_bytes + 15) / 16) * 16;
        w.line("case " + std::to_string(vi) + ": {");
        w.indent();
        w.line("intentir_cuda_selected_variant_idx = " + std::to_string(vi) + ";");
        w.line("intentir_cuda_selected_variant_tag = \"" + v.suffix + "\";");
        w.line("intentir_cuda_fastpath_enabled = " + std::string(v.specialize_full_tile ? "1" : "0") + ";");
        w.line("const int smem = (int)" + std::to_string(smem_v) + ";");
        w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
        w.line("dim3 g((unsigned)((N + " + std::to_string(v.tile_n) + " - 1) / " + std::to_string(v.tile_n) + "), "
               "(unsigned)((M + " + std::to_string(v.tile_m) + " - 1) / " + std::to_string(v.tile_m) + "), 1u);");
        w.line("const void* kptr = (const void*)" + kname + ";");
        w.line("if (smem >= 49152 && (intentir_kernel_cached != kptr || intentir_smem_cached != smem)) {");
        w.indent();
        w.line("TORCH_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);");
        w.line("intentir_kernel_cached = kptr;");
        w.line("intentir_smem_cached = smem;");
        w.dedent();
        w.line("}");
        w.line(kname + "<<<g, b, (size_t)smem, stream>>>((const float*)A, (const float*)B, (float*)C, M_in, N_in, K_in);");
        w.line("break;");
        w.dedent();
        w.line("}");
      }
      w.line("default: {");
      w.indent();
      const std::string kname0 = intent.name + "__" + variants[0].suffix;
      const int64_t smem0 = ((variants[0].shared_bytes + 15) / 16) * 16;
      w.line("intentir_cuda_selected_variant_idx = 0;");
      w.line("intentir_cuda_selected_variant_tag = \"" + variants[0].suffix + "\";");
      w.line("intentir_cuda_fastpath_enabled = " + std::string(variants[0].specialize_full_tile ? "1" : "0") + ";");
      w.line("const int smem = (int)" + std::to_string(smem0) + ";");
      w.line("dim3 b((unsigned)" + std::to_string(variants[0].threads) + ", 1u, 1u);");
      w.line("dim3 g((unsigned)((N + " + std::to_string(variants[0].tile_n) + " - 1) / " + std::to_string(variants[0].tile_n) + "), "
             "(unsigned)((M + " + std::to_string(variants[0].tile_m) + " - 1) / " + std::to_string(variants[0].tile_m) + "), 1u);");
      w.line("const void* kptr = (const void*)" + kname0 + ";");
      w.line("if (smem >= 49152 && (intentir_kernel_cached != kptr || intentir_smem_cached != smem)) {");
      w.indent();
      w.line("TORCH_CHECK(cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess);");
      w.line("intentir_kernel_cached = kptr;");
      w.line("intentir_smem_cached = smem;");
      w.dedent();
      w.line("}");
      w.line(kname0 + "<<<g, b, (size_t)smem, stream>>>((const float*)A, (const float*)B, (float*)C, M_in, N_in, K_in);");
      w.line("break;");
      w.dedent();
      w.line("}");
      w.line("}");
      w.dedent();
      w.line("}");
    }

    launch = {{"grid", {wmma_grid_x, wmma_grid_y, 1}},
              {"block", {32 * wmma_warps_m * wmma_warps_n, 1, 1}},
              {"shared_mem", shared_bytes}};
    shared_mem = shared_bytes;
  } else {
    w.line("#include \"kernels/matmul_fallback.cuh\"");
    w.blank();
    emit_selected_api(w);
    const std::string a_ptr_cty = c_type_for_dtype(a_dtype);
    const std::string b_ptr_cty = c_type_for_dtype(b_dtype);
    w.line("extern \"C\" __global__ void " + intent.name + "(");
    w.indent();
    w.line("const " + a_ptr_cty + "* __restrict__ A, const " + b_ptr_cty + "* __restrict__ B, float* __restrict__ C,");
    w.line(m_param + ", " + n_param + ", " + k_param + ") {");
    w.dedent();
    w.indent();
    {
      std::istringstream mnk_ss(mnk_load);
      for (std::string line; std::getline(mnk_ss, line);) {
        if (!line.empty()) w.line(line);
      }
    }
    w.line("constexpr int BLOCK_M = " + std::to_string(block_y) + ";");
    w.line("constexpr int BLOCK_N = " + std::to_string(block_x) + ";");
    w.line("constexpr int BLOCK_K = " + std::to_string(block_k) + ";");
    w.line("constexpr int THREAD_M = " + std::to_string(thread_m) + ";");
    w.line("constexpr int ROWS_PER_THREAD = " + std::to_string(rows_per_thread) + ";");
    w.line("__shared__ float As[BLOCK_M * BLOCK_K];");
    w.line("__shared__ float Bs[BLOCK_K * BLOCK_N];");
    w.line("intentir_cuda::matmul_f32_accum_fallback<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, ROWS_PER_THREAD, " + a_ptr_cty + ", " + b_ptr_cty +
           ">(A, B, C, M, N, K, As, Bs);");
    w.dedent();
    w.line("}");

    launch = {{"grid", {grid_x, grid_y, 1}}, {"block", {block_x, thread_m, 1}}, {"shared_mem", 0}};
    shared_mem = 0;
  }

  std::vector<std::string> tensor_args = {a, b, c};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {a, b, c};
  if (m_is_tensor) {
    tensor_args.push_back(M_name);
    arg_names.push_back(M_name);
  } else {
    scalar_args.emplace(M_name, "i32");
    arg_names.push_back(M_name);
  }
  if (n_is_tensor) {
    tensor_args.push_back(N_name);
    arg_names.push_back(N_name);
  } else {
    scalar_args.emplace(N_name, "i32");
    arg_names.push_back(N_name);
  }
  if (k_is_tensor) {
    tensor_args.push_back(K_name);
    arg_names.push_back(K_name);
  } else {
    scalar_args.emplace(K_name, "i32");
    arg_names.push_back(K_name);
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  json io_spec = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  if (host_launch) io_spec["host_launch"] = true;
  out["io_spec"] = io_spec;
  out["launch"] = launch;
  out["output_names"] = {c};
  out["bindings"] = bindings;
  (void)shared_mem;
  return out;
}

#include "emit_custom_kernels.inc"
#include "emit_reduction_index_misc.inc"
json emit_fused_elementwise(const Intent& intent, const json& bindings) {
  if (intent.outputs.size() != 1) fail("elementwise lowering requires a single output");
  const std::string out_name = intent.outputs[0];
  auto out_it = intent.tensors.find(out_name);
  if (out_it == intent.tensors.end()) fail("elementwise lowering: missing output tensor in intent.tensors");

  const Tensor& out_t = out_it->second;
  const int out_rank = static_cast<int>(out_t.shape.size());
  if (out_rank > 4) fail("elementwise lowering supports rank<=4");

  const bool specialize_dims = want_specialize_dims(intent, bindings);

  std::unordered_map<std::string, bool> produced;
  for (const auto& op : intent.ops) produced[op.output] = true;

  std::unordered_map<std::string, bool> outs;
  outs[out_name] = true;

  std::unordered_map<std::string, bool> used_tensors;
  used_tensors[out_name] = true;
  for (const auto& op : intent.ops) {
    for (const auto& inp : op.inputs) {
      if (intent.tensors.find(inp) != intent.tensors.end()) used_tensors[inp] = true;
    }
    if (intent.tensors.find(op.output) != intent.tensors.end()) used_tensors[op.output] = true;
  }

  std::unordered_map<std::string, bool> dim_syms_map;
  for (const auto& kv : used_tensors) {
    auto it = intent.tensors.find(kv.first);
    if (it == intent.tensors.end()) continue;
    for (const auto& d : it->second.shape) {
      if (d.is_string()) dim_syms_map[d.get<std::string>()] = true;
    }
  }
  std::vector<std::string> dim_syms;
  dim_syms.reserve(dim_syms_map.size());
  for (const auto& kv : dim_syms_map) dim_syms.push_back(kv.first);
  std::sort(dim_syms.begin(), dim_syms.end());

  std::vector<std::string> tensor_dim_args;
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> dim_param;
  std::vector<std::string> dim_load;
  std::unordered_map<std::string, std::string> dim_expr;
  for (const auto& sym : dim_syms) {
    const auto bound = binding_int(bindings, sym);
    if (is_scalar_tensor(intent, sym, "i32")) {
      tensor_dim_args.push_back(sym);
      dim_param.push_back("const int* " + sym + "_ptr");
      if (specialize_dims && bound.has_value()) {
        dim_load.push_back("(void)" + sym + "_ptr;");
        dim_load.push_back("constexpr int " + sym + " = " + std::to_string(*bound) + ";");
      } else {
        dim_load.push_back("const int " + sym + " = " + sym + "_ptr ? " + sym + "_ptr[0] : 0;");
      }
      dim_expr[sym] = sym;
    } else {
      scalar_args[sym] = "i32";
      dim_param.push_back("int " + sym + "_in");
      if (specialize_dims && bound.has_value()) {
        dim_load.push_back("(void)" + sym + "_in;");
        dim_load.push_back("constexpr int " + sym + " = " + std::to_string(*bound) + ";");
      } else {
        dim_load.push_back("const int " + sym + " = " + sym + "_in;");
      }
      dim_expr[sym] = sym;
    }
  }

  std::vector<std::string> external_inputs;
  for (const auto& op : intent.ops) {
    for (const auto& inp : op.inputs) {
      if (intent.tensors.find(inp) == intent.tensors.end()) continue;
      if (produced.find(inp) != produced.end()) continue;
      if (outs.find(inp) != outs.end()) continue;
      if (is_scalar_tensor(intent, inp, "i32") && dim_syms_map.find(inp) != dim_syms_map.end()) continue;
      bool seen = false;
      for (const auto& x : external_inputs)
        if (x == inp) seen = true;
      if (!seen) external_inputs.push_back(inp);
    }
  }

  std::vector<std::string> tensor_args;
  tensor_args.reserve(external_inputs.size() + 1 + tensor_dim_args.size());
  for (const auto& x : external_inputs) tensor_args.push_back(x);
  tensor_args.push_back(out_name);
  for (const auto& x : tensor_dim_args) tensor_args.push_back(x);

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;

  std::vector<std::string> out_dims_expr;
  out_dims_expr.reserve(out_rank);
  for (const auto& d : out_t.shape) {
    if (d.is_number_integer()) out_dims_expr.push_back(std::to_string(d.get<int64_t>()));
    else if (d.is_number()) out_dims_expr.push_back(std::to_string(static_cast<int64_t>(d.get<double>())));
    else if (d.is_string()) out_dims_expr.push_back(d.get<std::string>());
    else fail("elementwise: invalid out shape dim");
  }

  std::vector<std::string> idx_vars;
  std::vector<std::string> idx_code;
  if (out_rank == 0) {
  } else if (out_rank == 1) {
    idx_vars = {"i0"};
    idx_code.push_back("const int64_t i0 = tid;");
  } else if (out_rank == 2) {
    idx_vars = {"i0", "i1"};
    const std::string d1 = out_dims_expr[1].size() && std::isdigit((unsigned char)out_dims_expr[1][0]) ? out_dims_expr[1] : dim_expr.at(out_dims_expr[1]);
    idx_code.push_back("const int64_t i0 = tid / (int64_t)(" + d1 + ");");
    idx_code.push_back("const int64_t i1 = tid - i0 * (int64_t)(" + d1 + ");");
  } else if (out_rank == 3) {
    idx_vars = {"i0", "i1", "i2"};
    const std::string d1 = out_dims_expr[1].size() && std::isdigit((unsigned char)out_dims_expr[1][0]) ? out_dims_expr[1] : dim_expr.at(out_dims_expr[1]);
    const std::string d2 = out_dims_expr[2].size() && std::isdigit((unsigned char)out_dims_expr[2][0]) ? out_dims_expr[2] : dim_expr.at(out_dims_expr[2]);
    idx_code.push_back("const int64_t i0 = tid / ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + "));");
    idx_code.push_back("const int64_t rem = tid - i0 * ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + "));");
    idx_code.push_back("const int64_t i1 = rem / (int64_t)(" + d2 + ");");
    idx_code.push_back("const int64_t i2 = rem - i1 * (int64_t)(" + d2 + ");");
  } else {
    idx_vars = {"i0", "i1", "i2", "i3"};
    const std::string d1 = out_dims_expr[1].size() && std::isdigit((unsigned char)out_dims_expr[1][0]) ? out_dims_expr[1] : dim_expr.at(out_dims_expr[1]);
    const std::string d2 = out_dims_expr[2].size() && std::isdigit((unsigned char)out_dims_expr[2][0]) ? out_dims_expr[2] : dim_expr.at(out_dims_expr[2]);
    const std::string d3 = out_dims_expr[3].size() && std::isdigit((unsigned char)out_dims_expr[3][0]) ? out_dims_expr[3] : dim_expr.at(out_dims_expr[3]);
    idx_code.push_back("const int64_t i0 = tid / ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t rem0 = tid - i0 * ((int64_t)(" + d1 + ") * (int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t i1 = rem0 / ((int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t rem1 = rem0 - i1 * ((int64_t)(" + d2 + ") * (int64_t)(" + d3 + "));");
    idx_code.push_back("const int64_t i2 = rem1 / (int64_t)(" + d3 + ");");
    idx_code.push_back("const int64_t i3 = rem1 - i2 * (int64_t)(" + d3 + ");");
  }

  std::unordered_map<std::string, std::string> value_expr;
  std::unordered_map<std::string, std::string> value_type;
  auto dtype_of = [&](const std::string& name) -> std::string {
    auto it_local = value_type.find(name);
    if (it_local != value_type.end()) return it_local->second;
    auto it_tensor = intent.tensors.find(name);
    if (it_tensor != intent.tensors.end()) return it_tensor->second.dtype;
    fail("elementwise: unknown dtype source for " + name);
    return "f32";
  };

  auto load_tensor = [&](const std::string& name) -> std::string {
    auto it = intent.tensors.find(name);
    if (it == intent.tensors.end()) fail("elementwise: unknown tensor " + name);
    const Tensor& t = it->second;
    const std::string idx_expr = emit_broadcast_index_expr(out_rank, t.shape, idx_vars, dim_expr);
    return name + "[(size_t)(" + idx_expr + ")]";
  };
  auto val = [&](const std::string& name) -> std::string {
    auto it = value_expr.find(name);
    if (it != value_expr.end()) return it->second;
    if (dim_syms_map.find(name) != dim_syms_map.end() && is_scalar_tensor(intent, name, "i32")) return name;
    if (intent.tensors.find(name) == intent.tensors.end()) fail("elementwise: unknown value " + name);
    return load_tensor(name);
  };

  std::vector<std::string> code_lines;
  for (const auto& op : intent.ops) {
    const std::string opname = op.op;
    const std::string outn = op.output;
    auto outt_it = intent.tensors.find(outn);
    if (outt_it == intent.tensors.end()) fail("elementwise: op output missing from tensors: " + outn);
    const std::string out_dt = outt_it->second.dtype;
    const std::string cty = c_type_for_dtype(out_dt);

    auto emit_assign = [&](const std::string& expr) {
      const std::string vname = "v_" + c_ident(outn);
      code_lines.push_back(cty + " " + vname + " = " + expr + ";");
      value_expr[outn] = vname;
      value_type[outn] = out_dt;
    };

    if (opname == "const") {
      json v = op.attrs.is_object() ? op.attrs.value("value", json(0)) : json(0);
      emit_assign(c_scalar_literal(out_dt, v));
    } else if (opname == "iota") {
      if (!op.inputs.empty()) fail("iota expects 0 inputs");
      int axis = 0;
      if (op.attrs.is_object() && op.attrs.contains("axis")) axis = op.attrs["axis"].get<int>();
      const int op_rank = static_cast<int>(outt_it->second.shape.size());
      if (op_rank <= 0) fail("iota expects rank>=1 output");
      if (axis < 0) axis += op_rank;
      if (axis < 0 || axis >= op_rank) fail("iota axis out of range");
      const int shift = out_rank - op_rank;
      if (shift < 0) fail("iota output rank cannot exceed final output rank");
      emit_assign("(" + cty + ")(" + idx_vars.at(static_cast<size_t>(shift + axis)) + ")");
    } else if (opname == "identity") {
      if (op.inputs.size() != 1) fail("identity expects 1 input");
      emit_assign(val(op.inputs[0]));
    } else if (opname == "broadcast_in_dim") {
      if (op.inputs.size() != 1) fail("broadcast_in_dim expects 1 input");
      const std::string& in_name = op.inputs[0];
      const auto in_it = intent.tensors.find(in_name);
      if (in_it == intent.tensors.end()) fail("broadcast_in_dim input tensor missing: " + in_name);
      const Tensor& in_t = in_it->second;
      const bool has_bdims = op.attrs.is_object() && op.attrs.contains("broadcast_dims") && op.attrs["broadcast_dims"].is_array();
      if (!has_bdims) {
        emit_assign(val(in_name));
      } else {
        const auto& bdims = op.attrs["broadcast_dims"];
        const int in_rank = static_cast<int>(in_t.shape.size());
        if (in_rank == 0) {
          // Scalar broadcast: reuse the computed scalar expression instead of emitting tensor indexing.
          emit_assign(val(in_name));
          continue;
        }
        if (static_cast<int>(bdims.size()) != in_rank) fail("broadcast_in_dim expects len(broadcast_dims) == input rank");
        std::vector<std::string> in_dim_expr;
        in_dim_expr.reserve(static_cast<size_t>(in_rank));
        for (const auto& d : in_t.shape) {
          if (d.is_number_integer()) {
            in_dim_expr.push_back(std::to_string(d.get<int64_t>()));
          } else if (d.is_number()) {
            in_dim_expr.push_back(std::to_string(static_cast<int64_t>(d.get<double>())));
          } else if (d.is_string()) {
            const std::string sym = d.get<std::string>();
            auto it_sym = dim_expr.find(sym);
            if (it_sym == dim_expr.end()) fail("broadcast_in_dim missing dim binding: " + sym);
            in_dim_expr.push_back(it_sym->second);
          } else {
            fail("broadcast_in_dim unsupported shape token");
          }
        }
        std::vector<std::string> strides(static_cast<size_t>(in_rank), "1");
        for (int i = in_rank - 2; i >= 0; --i) {
          strides[static_cast<size_t>(i)] = "(" + in_dim_expr[static_cast<size_t>(i + 1)] + " * " + strides[static_cast<size_t>(i + 1)] + ")";
        }
        std::string idx = "0";
        for (int i = 0; i < in_rank; ++i) {
          int axis = 0;
          const auto& tok = bdims[static_cast<size_t>(i)];
          if (tok.is_number_integer())
            axis = static_cast<int>(tok.get<int64_t>());
          else if (tok.is_number())
            axis = static_cast<int>(tok.get<double>());
          else
            fail("broadcast_in_dim dims must be integer");
          if (axis < 0) axis += out_rank;
          if (axis < 0 || axis >= out_rank) fail("broadcast_in_dim axis out of range");
          idx += " + ((int64_t)(" + idx_vars[static_cast<size_t>(axis)] + ") * (" + strides[static_cast<size_t>(i)] + "))";
        }
        emit_assign(in_name + "[(size_t)(" + idx + ")]");
      }
    } else if (opname == "cast") {
      if (op.inputs.size() != 1) fail("cast expects 1 input");
      const std::string in_name = op.inputs[0];
      const std::string from_dt = dtype_of(in_name);
      if (from_dt == "bf16" && out_dt == "f32") {
        emit_assign("__bfloat162float(" + val(in_name) + ")");
      } else if (from_dt == "f16" && out_dt == "f32") {
        emit_assign("__half2float(" + val(in_name) + ")");
      } else if (from_dt == "f32" && out_dt == "f16") {
        emit_assign("__float2half(" + val(in_name) + ")");
      } else if (from_dt == "f32" && out_dt == "bf16") {
        emit_assign("__float2bfloat16(" + val(in_name) + ")");
      } else {
        emit_assign("(" + cty + ")(" + val(in_name) + ")");
      }
    } else if (opname == "where") {
      if (op.inputs.size() != 3) fail("where expects 3 inputs (cond, x, y)");
      emit_assign("(" + val(op.inputs[0]) + " ? " + val(op.inputs[1]) + " : " + val(op.inputs[2]) + ")");
    } else if (opname == "not") {
      if (op.inputs.size() != 1) fail("not expects 1 input");
      emit_assign("(!" + val(op.inputs[0]) + ")");
    } else if (opname == "and" || opname == "or") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      const std::string op2 = (opname == "and") ? "&&" : "||";
      emit_assign("(" + val(op.inputs[0]) + " " + op2 + " " + val(op.inputs[1]) + ")");
    } else if (opname == "add" || opname == "sub" || opname == "mul" || opname == "div" || opname == "max" || opname == "min" ||
               opname == "remainder" || opname == "pow") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      const std::string a = val(op.inputs[0]);
      const std::string b = val(op.inputs[1]);
      if (opname == "add")
        emit_assign("(" + a + " + " + b + ")");
      else if (opname == "sub")
        emit_assign("(" + a + " - " + b + ")");
      else if (opname == "mul")
        emit_assign("(" + a + " * " + b + ")");
      else if (opname == "div")
        emit_assign("(" + a + " / " + b + ")");
      else if (opname == "pow") {
        if (out_dt == "i32" || out_dt == "i64") {
          emit_assign("(" + cty + ")pow((double)(" + a + "), (double)(" + b + "))");
        } else {
          emit_assign("powf(" + a + ", " + b + ")");
        }
      }
      else if (opname == "remainder") {
        if (out_dt == "i32" || out_dt == "i64") {
          emit_assign("((" + b + ") == 0 ? 0 : (" + a + " % " + b + "))");
        } else {
          // Match PyTorch-style remainder semantics for floating point:
          // r = a - floor(a / b) * b, and b==0 -> NaN.
          emit_assign("((" + b + ") == 0.0f ? NAN : (((" + a + ") - floorf((" + a + ") / (" + b + ")) * (" + b + "))))");
        }
      }
      else if (opname == "max")
        emit_assign("fmaxf(" + a + ", " + b + ")");
      else
        emit_assign("fminf(" + a + ", " + b + ")");
    } else if (opname == "relu") {
      if (op.inputs.size() != 1) fail("relu expects 1 input");
      emit_assign("fmaxf(" + val(op.inputs[0]) + ", 0.0f)");
    } else if (opname == "abs") {
      if (op.inputs.size() != 1) fail("abs expects 1 input");
      emit_assign("fabsf(" + val(op.inputs[0]) + ")");
    } else if (opname == "sin") {
      if (op.inputs.size() != 1) fail("sin expects 1 input");
      emit_assign("sinf(" + val(op.inputs[0]) + ")");
    } else if (opname == "cos") {
      if (op.inputs.size() != 1) fail("cos expects 1 input");
      emit_assign("cosf(" + val(op.inputs[0]) + ")");
    } else if (opname == "tan") {
      if (op.inputs.size() != 1) fail("tan expects 1 input");
      emit_assign("tanf(" + val(op.inputs[0]) + ")");
    } else if (opname == "erf") {
      if (op.inputs.size() != 1) fail("erf expects 1 input");
      emit_assign("erff(" + val(op.inputs[0]) + ")");
    } else if (opname == "exp") {
      if (op.inputs.size() != 1) fail("exp expects 1 input");
      bool use_exp2 = false;
      if (op.attrs.is_object() && op.attrs.contains("base")) {
        const auto& base = op.attrs["base"];
        if (base.is_number()) {
          const double v = base.get<double>();
          use_exp2 = std::abs(v - 2.0) < 1e-9;
        } else if (base.is_string()) {
          use_exp2 = (base.get<std::string>() == "2" || base.get<std::string>() == "2.0");
        }
      }
      emit_assign(use_exp2 ? ("exp2f(" + val(op.inputs[0]) + ")") : ("__expf(" + val(op.inputs[0]) + ")"));
    } else if (opname == "acos") {
      if (op.inputs.size() != 1) fail("acos expects 1 input");
      emit_assign("acosf(" + val(op.inputs[0]) + ")");
    } else if (opname == "atan") {
      if (op.inputs.size() != 1) fail("atan expects 1 input");
      emit_assign("atanf(" + val(op.inputs[0]) + ")");
    } else if (opname == "log") {
      if (op.inputs.size() != 1) fail("log expects 1 input");
      emit_assign("logf(" + val(op.inputs[0]) + ")");
    } else if (opname == "ceil") {
      if (op.inputs.size() != 1) fail("ceil expects 1 input");
      emit_assign("ceilf(" + val(op.inputs[0]) + ")");
    } else if (opname == "sqrt") {
      if (op.inputs.size() != 1) fail("sqrt expects 1 input");
      emit_assign("sqrtf(" + val(op.inputs[0]) + ")");
    } else if (opname == "floor") {
      if (op.inputs.size() != 1) fail("floor expects 1 input");
      emit_assign("floorf(" + val(op.inputs[0]) + ")");
    } else if (opname == "rsqrt") {
      if (op.inputs.size() != 1) fail("rsqrt expects 1 input");
      emit_assign("rsqrtf(" + val(op.inputs[0]) + ")");
    } else if (opname == "eq" || opname == "ne" || opname == "lt" || opname == "le" || opname == "gt" || opname == "ge") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      std::string cmp;
      if (opname == "eq")
        cmp = "==";
      else if (opname == "ne")
        cmp = "!=";
      else if (opname == "lt")
        cmp = "<";
      else if (opname == "le")
        cmp = "<=";
      else if (opname == "gt")
        cmp = ">";
      else
        cmp = ">=";
      emit_assign("(" + val(op.inputs[0]) + " " + cmp + " " + val(op.inputs[1]) + ")");
    } else if (opname == "bitwise_and" || opname == "bitwise_or") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      const std::string bop = (opname == "bitwise_and") ? "&" : "|";
      emit_assign("(" + val(op.inputs[0]) + " " + bop + " " + val(op.inputs[1]) + ")");
    } else if (opname == "bitwise_not") {
      if (op.inputs.size() != 1) fail("bitwise_not expects 1 input");
      emit_assign("(~" + val(op.inputs[0]) + ")");
    } else if (opname == "bitwise_left_shift" || opname == "bitwise_right_shift") {
      if (op.inputs.size() != 2) fail(opname + " expects 2 inputs");
      const std::string bop = (opname == "bitwise_left_shift") ? "<<" : ">>";
      emit_assign("(" + val(op.inputs[0]) + " " + bop + " " + val(op.inputs[1]) + ")");
    } else {
      fail("elementwise lowering unsupported op: " + opname);
    }
  }

  if (value_expr.find(out_name) == value_expr.end()) fail("elementwise lowering did not produce the output value");
  const std::string out_var = value_expr.at(out_name);
  const std::string out_cty = c_type_for_dtype(out_t.dtype);

  std::string total_expr;
  if (out_rank == 0) {
    total_expr = "1";
  } else {
    for (int i = 0; i < out_rank; ++i) {
      const auto& d = out_t.shape[i];
      std::string part;
      if (d.is_number_integer())
        part = "(int64_t)" + std::to_string(d.get<int64_t>());
      else if (d.is_number())
        part = "(int64_t)" + std::to_string(static_cast<int64_t>(d.get<double>()));
      else if (d.is_string())
        part = "(int64_t)" + dim_expr.at(d.get<std::string>());
      else
        fail("elementwise: invalid out dim");
      if (!total_expr.empty()) total_expr += " * ";
      total_expr += part;
    }
    if (total_expr.empty()) total_expr = "1";
  }

  int64_t total = 1;
  for (const auto& d : out_t.shape) {
    auto v = resolve_dim_token(d, bindings);
    if (!v.has_value()) fail("elementwise missing binding for out dim");
    total *= *v;
  }
  const bool full_tile = (block_x > 0) ? ((total % block_x) == 0) : false;
  const int64_t grid_x = full_tile ? (total / block_x) : ((total + block_x - 1) / block_x);

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.line("#include <cuda_fp16.h>");
  w.line("#include <cuda_bf16.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  std::vector<std::string> params;
  for (const auto& n : external_inputs) {
    const auto it = intent.tensors.find(n);
    if (it == intent.tensors.end()) fail("elementwise: missing tensor for param " + n);
    params.push_back("const " + c_type_for_dtype(it->second.dtype) + "* __restrict__ " + n);
  }
  params.push_back(c_type_for_dtype(out_t.dtype) + "* __restrict__ " + out_name);
  for (const auto& p : dim_param) params.push_back(p);
  for (size_t i = 0; i < params.size(); ++i) {
    const std::string comma = (i + 1 < params.size()) ? "," : "";
    w.line(params[i] + comma);
  }
  w.dedent();
  w.line(") {");
  w.indent();
  for (const auto& dl : dim_load) w.line(dl);
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  if (!full_tile) {
    w.line("const int64_t total = " + total_expr + ";");
    w.line("if (tid >= total) return;");
  }
  for (const auto& l : idx_code) w.line(l);
  for (const auto& l : code_lines) w.line(l);
  w.line(out_name + "[(size_t)tid] = (" + out_cty + ")" + out_var + ";");
  w.dedent();
  w.line("}");

  std::vector<std::string> arg_names;
  for (const auto& x : external_inputs) arg_names.push_back(x);
  arg_names.push_back(out_name);
  for (const auto& sym : dim_syms) arg_names.push_back(sym);

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

