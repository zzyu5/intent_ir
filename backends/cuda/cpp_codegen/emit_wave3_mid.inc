json emit_softmax_2d_last_f32(const Intent& intent, const json& bindings) {
  const std::string out_name = !intent.outputs.empty() ? intent.outputs[0] : std::string("out");
  bool emit_log_output = false;
  if (intent.ops.size() == 2 && intent.ops[0].op == "softmax" && intent.ops[1].op == "log") {
    const auto& softmax_op = intent.ops[0];
    const auto& log_op = intent.ops[1];
    if (!softmax_op.inputs.empty() && log_op.inputs.size() == 1 && log_op.inputs[0] == softmax_op.output) {
      emit_log_output = true;
    }
  }

  std::string in_name;
  for (const auto& op : intent.ops) {
    if (op.op == "reduce_max" && !op.inputs.empty()) {
      in_name = op.inputs[0];
      break;
    }
  }
  if (in_name.empty()) {
    std::unordered_map<std::string, bool> produced;
    for (const auto& op : intent.ops) produced[op.output] = true;
    for (const auto& kv : intent.tensors) {
      const std::string& tn = kv.first;
      const Tensor& tt = kv.second;
      if (tn == out_name) continue;
      if (produced.find(tn) != produced.end()) continue;
      if (tt.dtype == "f32" && tt.shape.size() == 2) {
        in_name = tn;
        break;
      }
    }
  }
  if (in_name.empty()) fail("softmax lowering failed to identify input tensor");

  auto it = intent.tensors.find(in_name);
  if (it == intent.tensors.end()) fail("softmax missing input tensor in intent.tensors: " + in_name);
  if (it->second.shape.size() != 2) fail("softmax expects rank-2 input");
  const json R_dim = it->second.shape[0];
  const json C_dim = it->second.shape[1];
  auto R_opt = resolve_dim_token(R_dim, bindings);
  auto C_opt = resolve_dim_token(C_dim, bindings);
  if (!R_opt.has_value() || !C_opt.has_value()) fail("softmax missing bindings for R/C");
  const int64_t R = *R_opt;
  const int64_t C = *C_opt;
  if (C > 1024) fail("softmax MVP supports only C<=1024");

	  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;

		  int64_t max_ept = binding_int(bindings, "SOFTMAX_MAX_EPT").value_or(16);
		  if (max_ept < 4) max_ept = 4;
		  if (max_ept > 16) max_ept = 16;

	  int64_t block_threads = 0;
	  int64_t sched_threads = 0;
	  int64_t ept_override = 0;
	  if (respect_schedule) {
	    // NOTE: For softmax, schedule.tile_n usually refers to the reduction width (e.g., Triton BLOCK_SIZE),
	    // not the CUDA block thread count. Map it into a (threads, EPT) pair instead of blindly using it.
	    int64_t sched_elems = resolve_schedule_int(intent, bindings, "tile_n", 0);
	    if (sched_elems > 1024) sched_elems = 1024;
	    if (sched_elems >= C && sched_elems > 0) {
	      for (int64_t t : {128, 64, 256, 512}) {
	        if (t <= 0 || t > 1024) continue;
	        if ((sched_elems % t) != 0) continue;
	        const int64_t e = sched_elems / t;
	        if (e <= 0 || e > max_ept) continue;
	        block_threads = t;
	        sched_threads = t;
	        ept_override = e;
	        break;
	      }
	    }
	  }
	  if (block_threads <= 0) block_threads = (C >= 128) ? 128 : 64;
  // Evidence-guided hint: if the access witness identifies a contiguous dominant axis that
  // matches the reduction axis, prefer a thread count sized to that contiguous range.
  if (!respect_schedule && !binding_int(bindings, "SOFTMAX_THREADS")) {
    if (auto aw = access_witness_meta(intent)) {
      if (aw->has_contiguous_range && !aw->dominant_axis.empty() && aw->dominant_axis == dim_str(C_dim)) {
        const int64_t eff_len = (aw->dominant_range_len > 0) ? std::min<int64_t>(C, aw->dominant_range_len) : C;
        block_threads = _threads_from_contiguous_range(eff_len, block_threads);
      }
    }
  }
  if (block_threads < 32) block_threads = 32;
  if ((block_threads % 32) != 0) block_threads = ((block_threads + 31) / 32) * 32;
  if (block_threads > 1024) block_threads = 1024;

	  if (auto tuned = binding_int(bindings, "SOFTMAX_THREADS")) {
	    if (0 < *tuned && *tuned <= 1024) {
	      block_threads = *tuned;
      if (block_threads < 32) block_threads = 32;
      if ((block_threads % 32) != 0) block_threads = ((block_threads + 31) / 32) * 32;
      if (block_threads > 1024) block_threads = 1024;
	    }
	  }

	  int64_t min_threads = std::max<int64_t>(32, (C + max_ept - 1) / max_ept);
	  if ((min_threads % 32) != 0) min_threads = ((min_threads + 31) / 32) * 32;
	  if (block_threads < min_threads) block_threads = min_threads;
	  int64_t ept = std::max<int64_t>(1, (C + block_threads - 1) / block_threads);
	  if (ept_override > 0 && block_threads == sched_threads) ept = ept_override;

  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");
  const std::string R_name = dim_str(R_dim);
  const std::string C_name = dim_str(C_dim);
  bool r_is_tensor = is_scalar_tensor(intent, R_name, "i32");
  bool c_is_tensor = is_scalar_tensor(intent, C_name, "i32");
  if (specialize_dims && r_is_tensor && bindings.contains(R_name)) r_is_tensor = false;
  if (specialize_dims && c_is_tensor && bindings.contains(C_name)) c_is_tensor = false;

  const std::string r_param = r_is_tensor ? ("const int* " + R_name + "_ptr") : ("int " + R_name + "_in");
  const std::string c_param = c_is_tensor ? ("const int* " + C_name + "_ptr") : ("int " + C_name + "_in");
  const std::string r_load = r_is_tensor ? ("const int R = " + R_name + "_ptr ? " + R_name + "_ptr[0] : 0;")
                                         : ("const int R = " + R_name + "_in;");
  const std::string c_load = c_is_tensor ? ("const int C = " + C_name + "_ptr ? " + C_name + "_ptr[0] : 0;")
                                         : ("const int C = " + C_name + "_in;");
  const std::string r_unused = std::string("(void)") + (r_is_tensor ? (R_name + "_ptr") : (R_name + "_in")) + ";";
  const std::string c_unused = std::string("(void)") + (c_is_tensor ? (C_name + "_ptr") : (C_name + "_in")) + ";";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <cstdlib>");
  w.line("#include <cstdio>");
	  const auto exp2_opt = binding_int(bindings, "SOFTMAX_USE_EXP2");
	  const bool softmax_use_exp2 = exp2_opt.has_value() ? ((*exp2_opt) != 0) : false;
	  w.line("#include \"kernels/softmax.cuh\"");
	  w.blank();
	  emit_selected_api(w);
	  const bool enable_host_dispatch = (!emit_log_output) && want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims;
	  const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
	  bool host_launch = false;

	  if (!enable_host_dispatch) {
	    emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
	    w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_threads) + ") void " + intent.name +
	           "(const float* __restrict__ " + in_name + ", float* __restrict__ " + out_name + ", " + r_param + ", " + c_param + ") {");
	    w.indent();
	    if (specialize_dims) {
      w.line(r_unused);
      w.line(c_unused);
      w.line("constexpr int R = " + std::to_string(R) + ";");
      w.line("constexpr int C = " + std::to_string(C) + ";");
    } else {
      w.line(r_load);
      w.line(c_load);
	    }
	    w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_threads) + ";");
	    w.line("constexpr int EPT = " + std::to_string(ept) + ";");
      const bool full_tile = contract_full && specialize_dims && (C == (int64_t)block_threads * (int64_t)ept);
      w.line(
          std::string("intentir_cuda::softmax_2d_last_f32<BLOCK_THREADS, EPT, ") + (softmax_use_exp2 ? "true" : "false") +
          ", " + std::string(full_tile ? "true" : "false") + ", " + std::string(emit_log_output ? "true" : "false") + ">(" +
          in_name + ", " + out_name + ", R, C);");
	    w.dedent();
	    w.line("}");
		  } else {
			    host_launch = true;
			    struct SoftmaxVariant {
			      int64_t threads;
			      int64_t ept;
			      int64_t tiles;
			      int64_t rows_per_block;
			      bool vec4;
			      bool ragged;
			      bool warp_ragged;
			      bool warp4;
			      bool warp_expbuf;
			      bool use_exp2;
			      std::string suffix;
			    };
			    std::vector<SoftmaxVariant> variants;
	    auto norm_threads = [](int64_t t) -> int64_t {
	      if (t < 32) t = 32;
	      if (t > 1024) t = 1024;
      if ((t % 32) != 0) t = ((t + 31) / 32) * 32;
      if (t > 1024) t = 1024;
	      return t;
	    };
		    std::vector<bool> exp2_cands;
		    if (exp2_opt.has_value())
		      exp2_cands.push_back((*exp2_opt) != 0);
		    else {
		      exp2_cands.push_back(false);
		      exp2_cands.push_back(true);
		    }
		    auto tag_exp = [](const std::string& tag, bool use_exp2) -> std::string { return use_exp2 ? (tag + "_exp2") : tag; };

			    auto add_strided_variant_with_ept = [&](int64_t threads, int64_t vept, bool use_exp2, const std::string& tag) {
			      threads = norm_threads(threads);
			      if (threads <= 0) return;
			      if (vept <= 0 || vept > 32) return;
			      if (vept > max_ept) return;
			      for (const auto& v : variants) {
			        if (!v.warp4 && !v.warp_expbuf && !v.warp_ragged && !v.vec4 && !v.ragged && v.threads == threads && v.ept == vept &&
			            v.use_exp2 == use_exp2)
			          return;
			      }
			      variants.push_back(SoftmaxVariant{threads, vept, 0, 1, false, false, false, false, false, use_exp2, tag});
			    };

		    auto add_strided_variant = [&](int64_t threads, const std::string& tag) {
		      const int64_t vept = std::max<int64_t>(1, (C + threads - 1) / threads);
		      for (bool use_exp2 : exp2_cands) add_strided_variant_with_ept(threads, vept, use_exp2, tag_exp(tag, use_exp2));
		    };

			    auto add_vec4_variant = [&](int64_t threads, bool use_exp2, const std::string& tag) {
			      threads = norm_threads(threads);
			      if (threads <= 0) return;
			      const int64_t tiles = (C + (threads * 4) - 1) / (threads * 4);
			      if (tiles <= 0 || tiles > 8) return;
			      for (const auto& v : variants) {
			        if (v.vec4 && v.threads == threads && v.use_exp2 == use_exp2) return;
			      }
			      variants.push_back(SoftmaxVariant{threads, 0, tiles, 1, true, false, false, false, false, use_exp2, tag});
			    };

			    auto add_ragged_variant = [&](int64_t threads, bool use_exp2, const std::string& tag) {
			      if (!(contract_full && specialize_dims)) return;
			      threads = norm_threads(threads);
			      if (threads <= 0) return;
			      for (const auto& v : variants) {
			        if (v.ragged && v.threads == threads && v.use_exp2 == use_exp2) return;
			      }
			      variants.push_back(SoftmaxVariant{threads, 0, 0, 1, false, true, false, false, false, use_exp2, tag});
			    };

			    auto add_block_pair = [&](int64_t threads, const std::string& tag) {
			      add_strided_variant(threads, tag);
			      for (bool use_exp2 : exp2_cands) add_vec4_variant(threads, use_exp2, tag_exp("vec4_" + tag, use_exp2));
			    };

	    int64_t block_elems = resolve_schedule_int(intent, bindings, "tile_n", 0);
	    if (!(block_elems > 0 && block_elems <= 1024)) {
	      block_elems = 1;
	      while (block_elems < C) block_elems <<= 1;
	      if (block_elems > 1024) block_elems = 1024;
	    }
		    auto add_strided_pow2_variant = [&](int64_t threads, const std::string& tag) {
		      threads = norm_threads(threads);
		      if (threads <= 0) return;
		      if ((block_elems % threads) != 0) return;
		      const int64_t vept = block_elems / threads;
		      for (bool use_exp2 : exp2_cands) add_strided_variant_with_ept(threads, vept, use_exp2, tag_exp(tag, use_exp2));
		    };

			    auto add_warp4_variant = [&](int64_t warps_per_block, bool use_exp2, const std::string& tag) {
			      if (warps_per_block <= 0) return;
			      if (warps_per_block > 8) return;
			      const int64_t threads = norm_threads(warps_per_block * 32);
			      if (threads != warps_per_block * 32) return;
			      for (const auto& v : variants) {
			        if (v.warp4 && v.rows_per_block == warps_per_block && v.use_exp2 == use_exp2) return;
			      }
			      variants.push_back(SoftmaxVariant{threads, 0, 0, warps_per_block, false, false, false, true, false, use_exp2, tag});
			    };

			    auto add_warp_ragged_variant = [&](int64_t warps_per_block, bool use_exp2, const std::string& tag) {
			      if (!(contract_full && specialize_dims)) return;
			      if (warps_per_block <= 0) return;
			      if (warps_per_block > 8) return;
			      const int64_t threads = norm_threads(warps_per_block * 32);
			      if (threads != warps_per_block * 32) return;
			      for (const auto& v : variants) {
			        if (v.warp_ragged && v.rows_per_block == warps_per_block && v.use_exp2 == use_exp2) return;
			      }
			      variants.push_back(SoftmaxVariant{threads, 0, 0, warps_per_block, false, false, true, false, false, use_exp2, tag});
			    };

			    auto add_warp_expbuf_variant = [&](int64_t warps_per_block, bool use_exp2, const std::string& tag) {
		      if (warps_per_block <= 0) return;
		      if (warps_per_block > 8) return;
		      const int64_t threads = norm_threads(warps_per_block * 32);
		      if (threads != warps_per_block * 32) return;
			      for (const auto& v : variants) {
			        if (v.warp_expbuf && v.rows_per_block == warps_per_block && v.use_exp2 == use_exp2) return;
			      }
			      variants.push_back(SoftmaxVariant{threads, 0, 0, warps_per_block, false, false, false, false, true, use_exp2, tag});
			    };

		    // Evidence-guided small search space:
		    // - Variant[0] is always the strided "seed" (dispatch_off uses this).
		    // - Then, a couple of neighbors (half/double) for occupancy vs ILP tradeoffs.
		    // - One pow2-aligned reduction candidate.
		    // - 1-2 warp-specialized candidates (often good on newer GPUs).
		    //
		    // This makes the ablation semantics crisp:
		    //   evidence_on  = evidence + host selection,
		    //   dispatch_off = evidence + seed-only (no selection),
		    //   contract_off = no contract/specialize fastpaths.
			    const bool intentir_evidence_on = has_evidence;
			    for (bool use_exp2 : exp2_cands) add_strided_variant_with_ept(block_threads, (int)ept, use_exp2, tag_exp("seed", use_exp2));
			    for (bool use_exp2 : exp2_cands) add_vec4_variant(block_threads, use_exp2, tag_exp("vec4_seed", use_exp2));

		    std::vector<int64_t> thread_cands;
		    auto add_thread = [&](int64_t t) {
		      t = norm_threads(t);
		      if (t == norm_threads(block_threads)) return;
		      for (int64_t x : thread_cands) {
		        if (x == t) return;
		      }
		      thread_cands.push_back(t);
		    };
		    if (intentir_evidence_on) {
		      // Evidence-on: keep the candidate set small; trust the witness/schedule-derived seed.
		      add_thread(block_threads / 2);
		      add_thread(block_threads * 2);
		      add_thread(min_threads);
		    } else {
		      // Evidence-off: widen the neighborhood so selection can recover without priors.
		      add_thread(block_threads / 2);
		      add_thread(block_threads * 2);
		      add_thread(block_threads - 32);
		      add_thread(block_threads + 32);
		      add_thread(64);
		      add_thread(128);
		      add_thread(256);
		      // Ensure we cover the minimum threads required by EPT constraints.
		      add_thread(min_threads);
		    }

				    for (int64_t t : thread_cands) add_block_pair(t, "t" + std::to_string(t));
				    add_strided_pow2_variant(block_threads, "pow2_seed");
				    if (!intentir_evidence_on) {
				      for (int64_t t : thread_cands) add_strided_pow2_variant(t, "pow2_t" + std::to_string(t));
				    }

					    const int64_t warps_seed = std::max<int64_t>(1, std::min<int64_t>(8, block_threads / 32));
					    for (bool use_exp2 : exp2_cands) {
					      add_warp_ragged_variant(warps_seed, use_exp2, tag_exp("warprag_seed", use_exp2));
					      add_warp4_variant(warps_seed, use_exp2, tag_exp("warp4_seed", use_exp2));
					      add_warp_expbuf_variant(warps_seed, use_exp2, tag_exp("warpexp_seed", use_exp2));
					    }

						    // Always include a tiny warp neighborhood around the seed; without evidence we widen further.
						    for (int64_t dw : {-1, +1}) {
						      const int64_t w = warps_seed + dw;
						      if (w <= 0 || w > 8 || w == warps_seed) continue;
						      for (bool use_exp2 : exp2_cands) {
						        add_warp_ragged_variant(w, use_exp2, tag_exp("warprag_w" + std::to_string(w), use_exp2));
						        add_warp4_variant(w, use_exp2, tag_exp("warp4_w" + std::to_string(w), use_exp2));
						        add_warp_expbuf_variant(w, use_exp2, tag_exp("warpexp_w" + std::to_string(w), use_exp2));
						      }
						    }
						    if (intentir_evidence_on) {
						      // Evidence-on still needs a "warp=1" escape hatch: on some GPUs the best
						      // contract-fastpath is a single-warp ragged kernel (e.g., when C is small).
						      for (bool use_exp2 : exp2_cands) {
						        add_warp_ragged_variant(1, use_exp2, tag_exp("warprag_1", use_exp2));
						      }
						    }

					    if (!intentir_evidence_on) {
					      // Only widen the warp neighborhood without evidence; keep evidence-on minimal.
						      if (warps_seed > 1) {
						        for (bool use_exp2 : exp2_cands) {
					          add_warp_ragged_variant(warps_seed / 2, use_exp2, tag_exp("warprag_half", use_exp2));
					          add_warp4_variant(warps_seed / 2, use_exp2, tag_exp("warp4_half", use_exp2));
					          add_warp_expbuf_variant(warps_seed / 2, use_exp2, tag_exp("warpexp_half", use_exp2));
					        }
					      }
					      for (bool use_exp2 : exp2_cands) {
					        add_warp_ragged_variant(1, use_exp2, tag_exp("warprag_1", use_exp2));
					        add_warp4_variant(1, use_exp2, tag_exp("warp4_1", use_exp2));
					        add_warp_expbuf_variant(1, use_exp2, tag_exp("warpexp_1", use_exp2));
					      }
					      if (warps_seed < 8) {
					        for (bool use_exp2 : exp2_cands) {
					          add_warp_ragged_variant(std::min<int64_t>(8, warps_seed * 2), use_exp2, tag_exp("warprag_double", use_exp2));
					          add_warp4_variant(std::min<int64_t>(8, warps_seed * 2), use_exp2, tag_exp("warp4_double", use_exp2));
					          add_warp_expbuf_variant(std::min<int64_t>(8, warps_seed * 2), use_exp2, tag_exp("warpexp_double", use_exp2));
					        }
					      }
					    }

				    if (variants.empty()) add_block_pair(block_threads, "fallback");

	    emit_const_introspection_api(w, (int)variants.size(), has_evidence, contract_level, specialize_dims);

	    w.line("static const char* intentir_softmax_variant_tags[] = {");
	    w.indent();
	    for (const auto& v : variants) w.line("\"" + v.suffix + "\",");
	    w.dedent();
	    w.line("};");
	    w.blank();

			    for (const auto& v : variants) {
			      const std::string kname = intent.name + "__" + v.suffix;
			      w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.threads) + ") void " + kname +
			             "(const float* __restrict__ " + in_name + ", float* __restrict__ " + out_name + ", " + r_param + ", " + c_param + ") {");
	      w.indent();
	      w.line(r_unused);
	      w.line(c_unused);
			      w.line("constexpr int R = " + std::to_string(R) + ";");
			      w.line("constexpr int C = " + std::to_string(C) + ";");
			      const std::string exp2 = v.use_exp2 ? "true" : "false";
			      if (v.warp_ragged) {
			        w.line("constexpr int WARPS_PER_BLOCK = " + std::to_string(v.rows_per_block) + ";");
			        w.line("intentir_cuda::softmax_2d_last_f32_warp_ragged<WARPS_PER_BLOCK, " + std::to_string(C) + ", " + exp2 +
			               ">(" + in_name + ", " + out_name + ", R);");
			      } else if (v.ragged) {
			        w.line("constexpr int BLOCK_THREADS = " + std::to_string(v.threads) + ";");
			        w.line("intentir_cuda::softmax_2d_last_f32_ragged<BLOCK_THREADS, " + std::to_string(C) + ", " + exp2 + ">(" +
			               in_name + ", " + out_name + ", R);");
			      } else if (v.warp4) {
			        w.line("constexpr int WARPS_PER_BLOCK = " + std::to_string(v.rows_per_block) + ";");
			        w.line("intentir_cuda::softmax_2d_last_f32_warp4<WARPS_PER_BLOCK, " + exp2 + ">(" + in_name + ", " + out_name + ", R, C);");
			      } else if (v.warp_expbuf) {
			        w.line("constexpr int WARPS_PER_BLOCK = " + std::to_string(v.rows_per_block) + ";");
			        w.line("intentir_cuda::softmax_2d_last_f32_warp_expbuf<WARPS_PER_BLOCK, " + exp2 + ">(" + in_name + ", " + out_name + ", R, C);");
			      } else if (v.vec4) {
	            const bool full_tile = contract_full && specialize_dims && (C == (int64_t)v.threads * 4LL * (int64_t)v.tiles);
			        w.line("constexpr int BLOCK_THREADS = " + std::to_string(v.threads) + ";");
			        w.line("constexpr int TILES = " + std::to_string(v.tiles) + ";");
			        w.line("intentir_cuda::softmax_2d_last_f32_vec4<BLOCK_THREADS, TILES, " + exp2 + ", " +
			               std::string(full_tile ? "true" : "false") + ">(" + in_name + ", " + out_name + ", R, C);");
		      } else {
            const bool full_tile = contract_full && specialize_dims && (C == (int64_t)v.threads * (int64_t)v.ept);
		        w.line("constexpr int BLOCK_THREADS = " + std::to_string(v.threads) + ";");
		        w.line("constexpr int EPT = " + std::to_string(v.ept) + ";");
		        w.line("intentir_cuda::softmax_2d_last_f32<BLOCK_THREADS, EPT, " + exp2 + ", " +
			               std::string(full_tile ? "true" : "false") + ">(" + in_name + ", " + out_name + ", R, C);");
		      }
	      w.dedent();
	      w.line("}");
	      w.blank();
	    }

    // Host dispatcher: pick the best variant once (evidence-guided, small search space).
    w.line("extern \"C\" void " + intent.name + "_host_launch(");
    w.indent();
    w.line("float* " + in_name + ", float* " + out_name + ", " + r_param + ", " + c_param + ",");
    w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
    w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
    w.line("int64_t shared_mem, cudaStream_t stream) {");
    w.dedent();
    w.indent();
    w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
    w.line("(void)block_x; (void)block_y; (void)block_z;");
    w.line("(void)shared_mem;");
    w.line(r_unused);
    w.line(c_unused);
    w.line("constexpr int R = " + std::to_string(R) + ";");
    w.line("constexpr int C = " + std::to_string(C) + ";");
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
	      w.line("dim3 g((unsigned)((R + " + std::to_string(v.rows_per_block) + " - 1) / " + std::to_string(v.rows_per_block) +
	             "), 1u, 1u);");
	      w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
	      w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + in_name + ", " + out_name + ", " +
	             (r_is_tensor ? (R_name + "_ptr") : (R_name + "_in")) + ", " + (c_is_tensor ? (C_name + "_ptr") : (C_name + "_in")) + ");");
	      w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
	      w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + in_name + ", " + out_name + ", " +
	             (r_is_tensor ? (R_name + "_ptr") : (R_name + "_in")) + ", " + (c_is_tensor ? (C_name + "_ptr") : (C_name + "_in")) + ");");
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
	    w.line("if (const char* dbg = std::getenv(\"INTENTIR_CUDA_SOFTMAX_DISPATCH_DEBUG\")) {");
	    w.indent();
	    w.line("if (dbg[0]) std::fprintf(stderr, \"[intentir][softmax] selected=%d tag=%s best_ms=%f (warm=%d iters=%d passes=2)\\n\", intentir_selected, intentir_softmax_variant_tags[intentir_selected], (double)best_ms, warm, iters);");
	    w.dedent();
	    w.line("}");
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
	          if (v.warp_ragged) {
	            w.line("intentir_cuda_fastpath_enabled = " + std::string((contract_full && specialize_dims) ? "1" : "0") + ";");
	          } else if (!v.warp4 && !v.warp_expbuf) {
	            const bool full_tile = contract_full && specialize_dims &&
	                                   (v.vec4 ? (C == (int64_t)v.threads * 4LL * (int64_t)v.tiles)
	                                           : (C == (int64_t)v.threads * (int64_t)v.ept));
	            const bool fastpath = v.ragged ? (contract_full && specialize_dims) : full_tile;
	            w.line("intentir_cuda_fastpath_enabled = " + std::string(fastpath ? "1" : "0") + ";");
	          } else {
	            w.line("intentir_cuda_fastpath_enabled = 0;");
	          }
		      w.line("dim3 g((unsigned)((R + " + std::to_string(v.rows_per_block) + " - 1) / " + std::to_string(v.rows_per_block) +
		             "), 1u, 1u);");
		      w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
	      w.line(kname + "<<<g, b, 0, stream>>>(" + in_name + ", " + out_name + ", " + (r_is_tensor ? (R_name + "_ptr") : (R_name + "_in")) +
	             ", " + (c_is_tensor ? (C_name + "_ptr") : (C_name + "_in")) + ");");
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
	        if (v0.warp_ragged) {
	          w.line("intentir_cuda_fastpath_enabled = " + std::string((contract_full && specialize_dims) ? "1" : "0") + ";");
	        } else if (!v0.warp4 && !v0.warp_expbuf) {
	          const bool full_tile = contract_full && specialize_dims &&
	                                 (v0.vec4 ? (C == (int64_t)v0.threads * 4LL * (int64_t)v0.tiles)
	                                          : (C == (int64_t)v0.threads * (int64_t)v0.ept));
	          const bool fastpath = v0.ragged ? (contract_full && specialize_dims) : full_tile;
	          w.line("intentir_cuda_fastpath_enabled = " + std::string(fastpath ? "1" : "0") + ";");
	        } else {
	          w.line("intentir_cuda_fastpath_enabled = 0;");
	        }
		    w.line("dim3 g((unsigned)((R + " + std::to_string(v0.rows_per_block) + " - 1) / " + std::to_string(v0.rows_per_block) +
		           "), 1u, 1u);");
		    w.line("dim3 b((unsigned)" + std::to_string(v0.threads) + ", 1u, 1u);");
	    w.line(k0 + "<<<g, b, 0, stream>>>(" + in_name + ", " + out_name + ", " + (r_is_tensor ? (R_name + "_ptr") : (R_name + "_in")) +
	           ", " + (c_is_tensor ? (C_name + "_ptr") : (C_name + "_in")) + ");");
	    w.line("break;");
    w.dedent();
    w.line("}");
    w.line("}");
    w.dedent();
    w.line("}");
  }

  std::vector<std::string> tensor_args = {in_name, out_name};
  std::unordered_map<std::string, std::string> scalar_args;
  std::vector<std::string> arg_names = {in_name, out_name};
  if (r_is_tensor) {
    tensor_args.push_back(R_name);
    arg_names.push_back(R_name);
  } else {
    scalar_args.emplace(R_name, "i32");
    arg_names.push_back(R_name);
  }
  if (c_is_tensor) {
    tensor_args.push_back(C_name);
    arg_names.push_back(C_name);
  } else {
    scalar_args.emplace(C_name, "i32");
    arg_names.push_back(C_name);
  }

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, scalar_args, arg_names);
  if (host_launch) out["io_spec"]["host_launch"] = true;
  out["launch"] = {{"grid", {R, 1, 1}}, {"block", {block_threads, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = bindings;
  return out;
}

json emit_layernorm_2d_f32(const Intent& intent, const json& bindings) {
  std::unordered_map<std::string, bool> produced;
  for (const auto& op : intent.ops) produced[op.output] = true;
  std::unordered_map<std::string, bool> outs;
  for (const auto& n : intent.outputs) outs[n] = true;

  std::vector<std::string> inputs;
  for (const auto& kv : intent.tensors) {
    const std::string& name = kv.first;
    if (produced.find(name) != produced.end()) continue;
    if (outs.find(name) != outs.end()) continue;
    inputs.push_back(name);
  }

  std::string X_name;
  for (const auto& n : inputs) {
    if (n == "X") {
      X_name = n;
      break;
    }
  }
  if (X_name.empty()) {
    for (const auto& n : inputs) {
      const Tensor& t = intent.tensors.at(n);
      if (t.dtype == "f32" && t.shape.size() == 2) {
        X_name = n;
        break;
      }
    }
  }
  if (X_name.empty()) fail("layernorm lowering cannot find input X");

  std::string W_name;
  std::string B_name;
  for (const auto& n : inputs) {
    if (n == "W") W_name = n;
    if (n == "B") B_name = n;
  }
  if (W_name.empty() || B_name.empty()) {
    std::vector<std::string> rank1;
    for (const auto& n : inputs) {
      const Tensor& t = intent.tensors.at(n);
      if (t.dtype == "f32" && t.shape.size() == 1) rank1.push_back(n);
    }
    if (W_name.empty() && !rank1.empty()) W_name = rank1[0];
    if (B_name.empty() && rank1.size() > 1) B_name = rank1[1];
  }
  if (W_name.empty() || B_name.empty()) fail("layernorm lowering cannot find W/B inputs");

  if (intent.outputs.size() != 3) fail("layernorm lowering expects 3 outputs (Y, Mean, Rstd)");
  const std::string Y_name = intent.outputs[0];
  const std::string Mean_name = intent.outputs[1];
  const std::string Rstd_name = intent.outputs[2];

  auto x_it = intent.tensors.find(X_name);
  if (x_it == intent.tensors.end()) fail("layernorm missing X tensor in intent.tensors");
  if (x_it->second.shape.size() != 2) fail("layernorm expects rank-2 X");
  auto M_opt = resolve_dim_token(x_it->second.shape[0], bindings);
  auto N_opt = resolve_dim_token(x_it->second.shape[1], bindings);
  if (!M_opt.has_value() || !N_opt.has_value()) fail("layernorm missing bindings for M/N");
  const int64_t M = *M_opt;
  const int64_t N = *N_opt;
  (void)N;

  std::optional<double> eps;
  for (const auto& op : intent.ops) {
    if (op.op == "const" && op.output == "eps" && op.attrs.is_object()) {
      auto it = op.attrs.find("value");
      if (it != op.attrs.end() && it->is_number()) eps = it->get<double>();
      else if (it != op.attrs.end() && it->is_string()) {
        try {
          eps = std::stod(it->get<std::string>());
        } catch (...) {
          eps = std::nullopt;
        }
      }
      break;
    }
  }
  if (!eps.has_value()) eps = binding_double(bindings, "eps").value_or(1e-5);

  const bool respect_schedule = binding_int(bindings, "CUDA_RESPECT_SCHEDULE").value_or(0) != 0;

  int64_t block_x = 0;
  if (auto tuned = binding_int(bindings, "LAYERNORM_THREADS")) {
    if (0 < *tuned && *tuned <= 1024) block_x = *tuned;
  }
  if (block_x <= 0 && respect_schedule) {
    block_x = resolve_schedule_int(intent, bindings, "tile_n", 0);
  }
  if (block_x <= 0) {
    // Heuristic default: favor more threads for wide reductions, but avoid very
    // large blocks that can hurt occupancy.
    block_x = (N >= 2048) ? 128 : 64;
    // Evidence-guided hint: if the access witness identifies a contiguous dominant axis
    // that matches the reduction axis, size threads to cover that range efficiently.
    if (!respect_schedule) {
      if (auto aw = access_witness_meta(intent)) {
        if (aw->has_contiguous_range && !aw->dominant_axis.empty() && aw->dominant_axis == dim_str(x_it->second.shape[1])) {
          block_x = _threads_from_contiguous_range(aw->dominant_range_len, block_x);
        }
      }
    }
  }
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  auto is_pow2 = [](int64_t x) -> bool { return x > 0 && ((x & (x - 1)) == 0); };
  auto floor_pow2 = [](int64_t x) -> int64_t {
    if (x <= 1) return 1;
    int64_t p = 1;
    while ((p << 1) <= x) p <<= 1;
    return p;
  };
  if (!is_pow2(block_x)) {
    block_x = floor_pow2(block_x);
    if (block_x < 1) block_x = 1;
    if (block_x > 1024) block_x = 1024;
  }

  const bool specialize_dims = want_specialize_dims(intent, bindings);
  const bool has_evidence = intent_has_evidence(intent);
  const int contract_level = contract_level_code(intent);
  const bool contract_full = (contract_level_v2(intent).value_or("PARTIAL") == "FULL");
  const bool enable_host_dispatch = want_host_dispatch(bindings) && (!respect_schedule) && specialize_dims;
  const bool enable_host_dispatch_select = want_host_dispatch_select(bindings);
  bool host_launch = false;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include \"kernels/layernorm.cuh\"");
  w.blank();
  emit_selected_api(w);
  if (!enable_host_dispatch) {
    emit_const_introspection_api(w, /*variant_count=*/1, has_evidence, contract_level, specialize_dims);
    w.line("extern \"C\" __global__ void " + intent.name + "(");
    w.indent();
    w.line("const float* __restrict__ " + X_name + ",");
    w.line("float* __restrict__ " + Y_name + ",");
    w.line("const float* __restrict__ " + W_name + ",");
    w.line("const float* __restrict__ " + B_name + ",");
    w.line("float* __restrict__ " + Mean_name + ",");
    w.line("float* __restrict__ " + Rstd_name + ",");
    w.line("int M,");
    w.line("int N,");
    w.line("float eps) {");
    w.dedent();
    w.indent();
    w.line("constexpr int BLOCK_THREADS = " + std::to_string(block_x) + ";");
    w.line("intentir_cuda::layernorm_2d_f32<BLOCK_THREADS, " + std::string(contract_full ? "true" : "false") + ">(" + X_name + ", " + Y_name + ", " +
           W_name + ", " + B_name + ", " + Mean_name + ", " + Rstd_name + ", M, N, eps);");
    w.dedent();
    w.line("}");
  } else {
    host_launch = true;
    struct LnVariant {
      int64_t threads;
      std::string suffix;
    };
    std::vector<LnVariant> variants;
    auto add_variant = [&](int64_t threads, const std::string& tag) {
      if (threads < 32) threads = 32;
      if (threads > 1024) threads = 1024;
      // Keep power-of-two threads for reduction.
      auto is_pow2 = [](int64_t x) -> bool { return x > 0 && ((x & (x - 1)) == 0); };
      auto floor_pow2 = [](int64_t x) -> int64_t {
        if (x <= 1) return 1;
        int64_t p = 1;
        while ((p << 1) <= x) p <<= 1;
        return p;
      };
      if (!is_pow2(threads)) threads = floor_pow2(threads);
      if (threads < 32) threads = 32;
      if (threads > 1024) threads = 1024;
      for (const auto& v : variants) {
        if (v.threads == threads) return;
      }
      variants.push_back(LnVariant{threads, tag});
	    };

	    add_variant(block_x, "seed");
	    add_variant(block_x / 2, "t_half");
	    add_variant(block_x * 2, "t_double");
	    if (has_evidence && (N >= 4096)) {
	      // Evidence-on: include a high-threads option for wide reductions.
	      add_variant(512, "t512");
	    }
	    if (!has_evidence) {
	      // Evidence-off: widen the candidate set.
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

    for (const auto& v : variants) {
      const std::string kname = intent.name + "__" + v.suffix;
      w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(v.threads) + ") void " + kname + "(");
      w.indent();
      w.line("const float* __restrict__ " + X_name + ",");
      w.line("float* __restrict__ " + Y_name + ",");
      w.line("const float* __restrict__ " + W_name + ",");
      w.line("const float* __restrict__ " + B_name + ",");
      w.line("float* __restrict__ " + Mean_name + ",");
      w.line("float* __restrict__ " + Rstd_name + ",");
      w.line("int M,");
      w.line("int N,");
      w.line("float eps) {");
      w.dedent();
      w.indent();
      w.line("constexpr int BLOCK_THREADS = " + std::to_string(v.threads) + ";");
      w.line("intentir_cuda::layernorm_2d_f32<BLOCK_THREADS, " + std::string(contract_full ? "true" : "false") + ">(" + X_name + ", " + Y_name + ", " +
             W_name + ", " + B_name + ", " + Mean_name + ", " + Rstd_name + ", M, N, eps);");
      w.dedent();
      w.line("}");
      w.blank();
    }

    w.line("extern \"C\" void " + intent.name + "_host_launch(");
    w.indent();
    w.line("float* " + X_name + ", float* " + Y_name + ", float* " + W_name + ", float* " + B_name + ", float* " + Mean_name + ", float* " +
           Rstd_name + ", int M, int N, float eps,");
    w.line("int64_t grid_x, int64_t grid_y, int64_t grid_z,");
    w.line("int64_t block_x, int64_t block_y, int64_t block_z,");
    w.line("int64_t shared_mem, cudaStream_t stream) {");
    w.dedent();
    w.indent();
    w.line("(void)grid_x; (void)grid_y; (void)grid_z;");
    w.line("(void)block_x; (void)block_y; (void)block_z;");
    w.line("(void)shared_mem;");
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
	      w.line("dim3 g((unsigned)M, 1u, 1u);");
	      w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
	      w.line("for (int i = 0; i < warm; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + X_name + ", " + Y_name + ", " + W_name + ", " + B_name +
	             ", " + Mean_name + ", " + Rstd_name + ", M, N, eps);");
	      w.line("TORCH_CHECK(cudaStreamSynchronize(sel_stream) == cudaSuccess);");
	      w.line("TORCH_CHECK(cudaStreamBeginCapture(sel_stream, cudaStreamCaptureModeGlobal) == cudaSuccess);");
	      w.line("for (int i = 0; i < iters; ++i) " + kname + "<<<g, b, 0, sel_stream>>>(" + X_name + ", " + Y_name + ", " + W_name + ", " + B_name +
	             ", " + Mean_name + ", " + Rstd_name + ", M, N, eps);");
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
    w.line("dim3 g((unsigned)M, 1u, 1u);");
    w.line("switch (intentir_selected) {");
    for (size_t i = 0; i < variants.size(); ++i) {
      const auto& v = variants[i];
      const std::string kname = intent.name + "__" + v.suffix;
      w.line("case " + std::to_string(i) + ": {");
      w.indent();
      w.line("intentir_cuda_selected_variant_idx = " + std::to_string(i) + ";");
      w.line("intentir_cuda_selected_variant_tag = \"" + v.suffix + "\";");
      w.line("intentir_cuda_fastpath_enabled = " + std::string(contract_full ? "1" : "0") + ";");
      w.line("dim3 b((unsigned)" + std::to_string(v.threads) + ", 1u, 1u);");
      w.line(kname + "<<<g, b, 0, stream>>>(" + X_name + ", " + Y_name + ", " + W_name + ", " + B_name + ", " + Mean_name + ", " + Rstd_name +
             ", M, N, eps);");
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
    w.line("intentir_cuda_fastpath_enabled = " + std::string(contract_full ? "1" : "0") + ";");
    w.line("dim3 b((unsigned)" + std::to_string(v0.threads) + ", 1u, 1u);");
    w.line(k0 + "<<<g, b, 0, stream>>>(" + X_name + ", " + Y_name + ", " + W_name + ", " + B_name + ", " + Mean_name + ", " + Rstd_name +
           ", M, N, eps);");
    w.line("break;");
    w.dedent();
    w.line("}");
    w.line("}");
    w.dedent();
    w.line("}");
  }

  json out_bindings = bindings;
  if (!out_bindings.contains("eps")) out_bindings["eps"] = *eps;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(
      intent,
      /*tensor_args=*/{X_name, Y_name, W_name, B_name, Mean_name, Rstd_name},
      /*scalar_args=*/{{"M", "i32"}, {"N", "i32"}, {"eps", "f32"}},
      /*arg_names=*/{X_name, Y_name, W_name, B_name, Mean_name, Rstd_name, "M", "N", "eps"});
  if (host_launch) out["io_spec"]["host_launch"] = true;
  out["launch"] = {{"grid", {M, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {Y_name, Mean_name, Rstd_name};
  out["bindings"] = out_bindings;
  return out;
}

#include "emit_conv_pool_upsample.inc"
json emit_bmm3d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.empty()) fail("bmm lowering expects non-empty ops");

  const Op* mm = nullptr;
  if (intent.ops.size() == 1 && intent.ops[0].op == "matmul") {
    mm = &intent.ops[0];
  } else if (intent.ops.size() == 3 && intent.ops[0].op == "cast" && intent.ops[1].op == "cast" && intent.ops[2].op == "matmul") {
    mm = &intent.ops[2];
  } else {
    fail("bmm lowering expects matmul or cast+cast+matmul pattern");
  }
  if (mm->inputs.size() != 2) fail("bmm matmul expects 2 inputs");
  const std::string A = mm->inputs[0];
  const std::string B = mm->inputs[1];
  const std::string O = mm->output;

  auto a_it = intent.tensors.find(A);
  auto b_it = intent.tensors.find(B);
  auto o_it = intent.tensors.find(O);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("bmm lowering missing tensors");
  }
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || o_it->second.dtype != "f32") {
    fail("bmm lowering supports f32 only");
  }

  auto a_shape = resolve_shape_required(a_it->second, bindings, "bmm.A");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "bmm.B");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "bmm.O");
  if (a_shape.size() != 3 || b_shape.size() != 3 || o_shape.size() != 3) fail("bmm expects rank-3 tensors");

  const int64_t BATCH = a_shape[0];
  const int64_t M = a_shape[1];
  const int64_t K = a_shape[2];
  const int64_t BK = b_shape[1];
  const int64_t N = b_shape[2];
  if (b_shape[0] != BATCH || BK != K) fail("bmm shape mismatch");
  if (o_shape[0] != BATCH || o_shape[1] != M || o_shape[2] != N) fail("bmm output shape mismatch");
  if (BATCH <= 0 || M <= 0 || K <= 0 || N <= 0) fail("bmm invalid shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total = BATCH * M * N;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + A + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("float* __restrict__ " + O);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t n = t % " + std::to_string(N) + "LL; t /= " + std::to_string(N) + "LL;");
  w.line("const int64_t m = t % " + std::to_string(M) + "LL; t /= " + std::to_string(M) + "LL;");
  w.line("const int64_t b = t;");
  w.line("const int64_t a_base = (b * " + std::to_string(M) + "LL + m) * " + std::to_string(K) + "LL;");
  w.line("const int64_t b_base = b * " + std::to_string(K) + "LL * " + std::to_string(N) + "LL + n;");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t k = 0; k < " + std::to_string(K) + "LL; ++k) {");
  w.indent();
  w.line("acc = fmaf(" + A + "[a_base + k], " + B + "[b_base + k * " + std::to_string(N) + "LL], acc);");
  w.dedent();
  w.line("}");
  w.line(O + "[tid] = acc;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["BATCH"] = BATCH;
  out_bindings["M"] = M;
  out_bindings["K"] = K;
  out_bindings["N"] = N;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {A, B, O}, {}, {A, B, O});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {O};
  out["bindings"] = out_bindings;
  return out;
}

json emit_addmm2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 5) fail("addmm lowering expects 5 ops (matmul,mul,mul,add,cast)");
  const Op& mm = intent.ops[0];
  const Op& mul_mm = intent.ops[1];
  const Op& mul_bias = intent.ops[2];
  const Op& add = intent.ops[3];
  const Op& cast = intent.ops[4];
  if (mm.op != "matmul" || mul_mm.op != "mul" || mul_bias.op != "mul" || add.op != "add" || cast.op != "cast") {
    fail("addmm lowering pattern mismatch");
  }
  if (mm.inputs.size() != 2 || mul_mm.inputs.size() != 2 || mul_bias.inputs.size() != 2 || add.inputs.size() != 2 || cast.inputs.size() != 1) {
    fail("addmm lowering: invalid arity");
  }
  if (cast.attrs.value("to", std::string("f32")) != "f32") fail("addmm lowering expects cast(to=f32)");

  const std::string A = mm.inputs[0];
  const std::string B = mm.inputs[1];
  const std::string MM = mm.output;

  auto is_scalar_f32 = [&](const std::string& n) -> bool {
    auto it = intent.tensors.find(n);
    return it != intent.tensors.end() && it->second.dtype == "f32" && it->second.shape.empty();
  };
  auto is_rank2_f32 = [&](const std::string& n) -> bool {
    auto it = intent.tensors.find(n);
    return it != intent.tensors.end() && it->second.dtype == "f32" && it->second.shape.size() == 2;
  };

  std::string alpha_name;
  if (mul_mm.inputs[0] == MM && is_scalar_f32(mul_mm.inputs[1]))
    alpha_name = mul_mm.inputs[1];
  else if (mul_mm.inputs[1] == MM && is_scalar_f32(mul_mm.inputs[0]))
    alpha_name = mul_mm.inputs[0];
  else
    fail("addmm lowering expects mul(matmul_out, alpha)");

  std::string bias_name;
  std::string beta_name;
  if (is_rank2_f32(mul_bias.inputs[0]) && is_scalar_f32(mul_bias.inputs[1])) {
    bias_name = mul_bias.inputs[0];
    beta_name = mul_bias.inputs[1];
  } else if (is_rank2_f32(mul_bias.inputs[1]) && is_scalar_f32(mul_bias.inputs[0])) {
    bias_name = mul_bias.inputs[1];
    beta_name = mul_bias.inputs[0];
  } else {
    fail("addmm lowering expects mul(bias, beta)");
  }

  const std::string scaled_mm = mul_mm.output;
  const std::string scaled_bias = mul_bias.output;
  if (!((add.inputs[0] == scaled_mm && add.inputs[1] == scaled_bias) || (add.inputs[0] == scaled_bias && add.inputs[1] == scaled_mm))) {
    fail("addmm lowering expects add(scaled_matmul, scaled_bias)");
  }
  if (cast.inputs[0] != add.output) fail("addmm lowering expects cast(add_out)");

  auto a_it = intent.tensors.find(A);
  auto b_it = intent.tensors.find(B);
  auto bias_it = intent.tensors.find(bias_name);
  auto out_it = intent.tensors.find(cast.output);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || bias_it == intent.tensors.end() || out_it == intent.tensors.end()) {
    fail("addmm lowering missing tensors");
  }
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || bias_it->second.dtype != "f32" || out_it->second.dtype != "f32") {
    fail("addmm lowering supports f32 only");
  }
  auto a_shape = resolve_shape_required(a_it->second, bindings, "addmm.A");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "addmm.B");
  auto i_shape = resolve_shape_required(bias_it->second, bindings, "addmm.bias");
  auto o_shape = resolve_shape_required(out_it->second, bindings, "addmm.out");
  if (a_shape.size() != 2 || b_shape.size() != 2 || i_shape.size() != 2 || o_shape.size() != 2) fail("addmm expects rank-2 tensors");

  const int64_t M = a_shape[0];
  const int64_t K = a_shape[1];
  const int64_t N = b_shape[1];
  if (b_shape[0] != K) fail("addmm shape mismatch K");
  if (i_shape[0] != M || i_shape[1] != N || o_shape[0] != M || o_shape[1] != N) fail("addmm shape mismatch output/bias");
  if (M <= 0 || N <= 0 || K <= 0) fail("addmm invalid shape");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total = M * N;
  const int64_t grid_x = (total + block_x - 1) / block_x;
  const std::string O = cast.output;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + A + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("const float* __restrict__ " + bias_name + ",");
  w.line("const float* __restrict__ " + alpha_name + ",");
  w.line("const float* __restrict__ " + beta_name + ",");
  w.line("float* __restrict__ " + O);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("const int64_t m = tid / " + std::to_string(N) + "LL;");
  w.line("const int64_t n = tid - m * " + std::to_string(N) + "LL;");
  w.line("const float alpha_v = " + alpha_name + " ? " + alpha_name + "[0] : 1.0f;");
  w.line("const float beta_v = " + beta_name + " ? " + beta_name + "[0] : 1.0f;");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t k = 0; k < " + std::to_string(K) + "LL; ++k) {");
  w.indent();
  w.line("acc = fmaf(" + A + "[m * " + std::to_string(K) + "LL + k], " + B + "[k * " + std::to_string(N) + "LL + n], acc);");
  w.dedent();
  w.line("}");
  w.line(O + "[tid] = alpha_v * acc + beta_v * " + bias_name + "[tid];");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["K"] = K;
  out_bindings["N"] = N;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {A, B, bias_name, alpha_name, beta_name, O}, {}, {A, B, bias_name, alpha_name, beta_name, O});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {O};
  out["bindings"] = out_bindings;
  return out;
}

json emit_baddbmm3d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 4) fail("baddbmm lowering expects 4 ops");
  const Op& mm = intent.ops[0];
  const Op& mul_mm = intent.ops[1];
  const Op& mul_bias = intent.ops[2];
  const Op& add = intent.ops[3];
  if (mm.op != "matmul" || mul_mm.op != "mul" || mul_bias.op != "mul" || add.op != "add") {
    fail("baddbmm lowering pattern mismatch");
  }
  if (mm.inputs.size() != 2 || mul_mm.inputs.size() != 2 || mul_bias.inputs.size() != 2 || add.inputs.size() != 2) {
    fail("baddbmm lowering: invalid arity");
  }

  const std::string A = mm.inputs[0];
  const std::string B = mm.inputs[1];
  const std::string MM = mm.output;
  auto is_scalar_f32 = [&](const std::string& n) -> bool {
    auto it = intent.tensors.find(n);
    return it != intent.tensors.end() && it->second.dtype == "f32" && it->second.shape.empty();
  };
  auto is_rank3_f32 = [&](const std::string& n) -> bool {
    auto it = intent.tensors.find(n);
    return it != intent.tensors.end() && it->second.dtype == "f32" && it->second.shape.size() == 3;
  };

  std::string alpha_name;
  if (mul_mm.inputs[0] == MM && is_scalar_f32(mul_mm.inputs[1]))
    alpha_name = mul_mm.inputs[1];
  else if (mul_mm.inputs[1] == MM && is_scalar_f32(mul_mm.inputs[0]))
    alpha_name = mul_mm.inputs[0];
  else
    fail("baddbmm lowering expects mul(matmul_out,alpha)");

  std::string bias_name;
  std::string beta_name;
  if (is_rank3_f32(mul_bias.inputs[0]) && is_scalar_f32(mul_bias.inputs[1])) {
    bias_name = mul_bias.inputs[0];
    beta_name = mul_bias.inputs[1];
  } else if (is_rank3_f32(mul_bias.inputs[1]) && is_scalar_f32(mul_bias.inputs[0])) {
    bias_name = mul_bias.inputs[1];
    beta_name = mul_bias.inputs[0];
  } else {
    fail("baddbmm lowering expects mul(bias,beta)");
  }

  const std::string scaled_mm = mul_mm.output;
  const std::string scaled_bias = mul_bias.output;
  if (!((add.inputs[0] == scaled_mm && add.inputs[1] == scaled_bias) || (add.inputs[0] == scaled_bias && add.inputs[1] == scaled_mm))) {
    fail("baddbmm lowering expects add(scaled_matmul,scaled_bias)");
  }
  const std::string O = add.output;

  auto a_it = intent.tensors.find(A);
  auto b_it = intent.tensors.find(B);
  auto bias_it = intent.tensors.find(bias_name);
  auto o_it = intent.tensors.find(O);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || bias_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("baddbmm missing tensors");
  }
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || bias_it->second.dtype != "f32" || o_it->second.dtype != "f32") {
    fail("baddbmm supports f32 only");
  }
  auto a_shape = resolve_shape_required(a_it->second, bindings, "baddbmm.A");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "baddbmm.B");
  auto i_shape = resolve_shape_required(bias_it->second, bindings, "baddbmm.bias");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "baddbmm.out");
  if (a_shape.size() != 3 || b_shape.size() != 3 || i_shape.size() != 3 || o_shape.size() != 3) fail("baddbmm expects rank-3 tensors");

  const int64_t BATCH = a_shape[0];
  const int64_t M = a_shape[1];
  const int64_t K = a_shape[2];
  const int64_t BK = b_shape[1];
  const int64_t N = b_shape[2];
  if (b_shape[0] != BATCH || BK != K) fail("baddbmm shape mismatch matmul");
  if (i_shape[0] != BATCH || i_shape[1] != M || i_shape[2] != N) fail("baddbmm bias shape mismatch");
  if (o_shape[0] != BATCH || o_shape[1] != M || o_shape[2] != N) fail("baddbmm output shape mismatch");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t total = BATCH * M * N;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + A + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("const float* __restrict__ " + bias_name + ",");
  w.line("const float* __restrict__ " + alpha_name + ",");
  w.line("const float* __restrict__ " + beta_name + ",");
  w.line("float* __restrict__ " + O);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t n = t % " + std::to_string(N) + "LL; t /= " + std::to_string(N) + "LL;");
  w.line("const int64_t m = t % " + std::to_string(M) + "LL; t /= " + std::to_string(M) + "LL;");
  w.line("const int64_t b = t;");
  w.line("const float alpha_v = " + alpha_name + " ? " + alpha_name + "[0] : 1.0f;");
  w.line("const float beta_v = " + beta_name + " ? " + beta_name + "[0] : 1.0f;");
  w.line("const int64_t a_base = (b * " + std::to_string(M) + "LL + m) * " + std::to_string(K) + "LL;");
  w.line("const int64_t b_base = b * " + std::to_string(K) + "LL * " + std::to_string(N) + "LL + n;");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t k = 0; k < " + std::to_string(K) + "LL; ++k) {");
  w.indent();
  w.line("acc = fmaf(" + A + "[a_base + k], " + B + "[b_base + k * " + std::to_string(N) + "LL], acc);");
  w.dedent();
  w.line("}");
  w.line(O + "[tid] = alpha_v * acc + beta_v * " + bias_name + "[tid];");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["BATCH"] = BATCH;
  out_bindings["M"] = M;
  out_bindings["K"] = K;
  out_bindings["N"] = N;

  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {A, B, bias_name, alpha_name, beta_name, O}, {}, {A, B, bias_name, alpha_name, beta_name, O});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {O};
  out["bindings"] = out_bindings;
  return out;
}

json emit_dot1d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 4) fail("dot lowering expects cast,cast,mul,reduce_sum pattern");
  if (intent.ops[0].op != "cast" || intent.ops[1].op != "cast" || intent.ops[2].op != "mul" || intent.ops[3].op != "reduce_sum") {
    fail("dot lowering pattern mismatch");
  }
  const Op& cast_x = intent.ops[0];
  const Op& cast_y = intent.ops[1];
  const Op& mul = intent.ops[2];
  const Op& red = intent.ops[3];
  if (cast_x.inputs.size() != 1 || cast_y.inputs.size() != 1 || mul.inputs.size() != 2 || red.inputs.size() != 1) {
    fail("dot lowering arity mismatch");
  }
  if (cast_x.attrs.value("to", std::string("f32")) != "f32" || cast_y.attrs.value("to", std::string("f32")) != "f32") {
    fail("dot lowering expects cast(to=f32)");
  }
  if (!((mul.inputs[0] == cast_x.output && mul.inputs[1] == cast_y.output) || (mul.inputs[1] == cast_x.output && mul.inputs[0] == cast_y.output))) {
    fail("dot lowering expects mul(cast_x, cast_y)");
  }
  if (red.inputs[0] != mul.output) fail("dot lowering expects reduce_sum(mul)");

  const std::string X = cast_x.inputs[0];
  const std::string Y = cast_y.inputs[0];
  const std::string OUT = red.output;
  auto x_it = intent.tensors.find(X);
  auto y_it = intent.tensors.find(Y);
  auto o_it = intent.tensors.find(OUT);
  if (x_it == intent.tensors.end() || y_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("dot lowering missing tensors");
  if (x_it->second.dtype != "f32" || y_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("dot lowering supports f32 only");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "dot.x");
  auto y_shape = resolve_shape_required(y_it->second, bindings, "dot.y");
  if (x_shape.size() != 1 || y_shape.size() != 1) fail("dot expects rank-1 inputs");
  if (x_shape[0] != y_shape[0]) fail("dot shape mismatch");
  const int64_t N = x_shape[0];
  if (N <= 0) fail("dot invalid N");

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + X + ",");
  w.line("const float* __restrict__ " + Y + ",");
  w.line("float* __restrict__ " + OUT);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("__shared__ float red[" + std::to_string(block_x) + "];");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t i = (int64_t)threadIdx.x; i < " + std::to_string(N) + "LL; i += (int64_t)blockDim.x) {");
  w.indent();
  w.line("acc = fmaf(" + X + "[i], " + Y + "[i], acc);");
  w.dedent();
  w.line("}");
  w.line("red[(int)threadIdx.x] = acc;");
  w.line("__syncthreads();");
  w.line("for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {");
  w.indent();
  w.line("if ((int)threadIdx.x < stride) red[(int)threadIdx.x] += red[(int)threadIdx.x + stride];");
  w.line("__syncthreads();");
  w.dedent();
  w.line("}");
  w.line("if ((int)threadIdx.x == 0) " + OUT + "[0] = red[0];");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {X, Y, OUT}, {}, {X, Y, OUT});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {OUT};
  out["bindings"] = out_bindings;
  return out;
}

json emit_cumsum2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "cumsum")) fail("cumsum lowering expects one cumsum op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("cumsum expects 1 input");
  int axis = 1;
  if (op.attrs.is_object() && op.attrs.contains("axis")) {
    axis = static_cast<int>(resolve_dim_token_required(op.attrs["axis"], bindings, "cumsum.axis"));
  }
  if (axis != 1) fail("cumsum lowering currently supports axis=1 only");

  const std::string IN = op.inputs[0];
  const std::string OUT = op.output;
  auto in_it = intent.tensors.find(IN);
  auto out_it = intent.tensors.find(OUT);
  if (in_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("cumsum missing tensors");
  if (in_it->second.dtype != "f32" || out_it->second.dtype != "f32") fail("cumsum supports f32 only");
  auto in_shape = resolve_shape_required(in_it->second, bindings, "cumsum.in");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "cumsum.out");
  if (in_shape.size() != 2 || out_shape.size() != 2) fail("cumsum expects rank-2");
  if (in_shape != out_shape) fail("cumsum input/output shape mismatch");
  const int64_t M = in_shape[0];
  const int64_t N = in_shape[1];
  if (M <= 0 || N <= 0) fail("cumsum invalid shape");

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
  w.line("const float* __restrict__ " + IN + ",");
  w.line("float* __restrict__ " + OUT);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float acc = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("acc += " + IN + "[base + n];");
  w.line(OUT + "[base + n] = acc;");
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
  out["io_spec"] = io_spec_from_args(intent, {IN, OUT}, {}, {IN, OUT});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {OUT};
  out["bindings"] = out_bindings;
  return out;
}

json emit_cumextrema1d_f32(const Intent& intent, const json& bindings, bool is_max) {
  if (!(intent.ops.size() == 1 && (intent.ops[0].op == "cummax" || intent.ops[0].op == "cummin"))) {
    fail("cumextrema lowering expects one cummax/cummin op");
  }
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("cumextrema expects 1 input");
  int axis = 0;
  if (op.attrs.is_object() && op.attrs.contains("axis")) {
    axis = static_cast<int>(resolve_dim_token_required(op.attrs["axis"], bindings, "cumextrema.axis"));
  }
  if (axis != 0) fail("cumextrema currently supports axis=0 only");
  const std::string IN = op.inputs[0];
  const std::string OUT = op.output;
  auto in_it = intent.tensors.find(IN);
  auto out_it = intent.tensors.find(OUT);
  if (in_it == intent.tensors.end() || out_it == intent.tensors.end()) fail("cumextrema missing tensors");
  if (in_it->second.dtype != "f32" || out_it->second.dtype != "f32") fail("cumextrema supports f32 only");
  auto in_shape = resolve_shape_required(in_it->second, bindings, "cumextrema.in");
  auto out_shape = resolve_shape_required(out_it->second, bindings, "cumextrema.out");
  if (in_shape.size() != 1 || out_shape.size() != 1) fail("cumextrema expects rank-1");
  if (in_shape[0] != out_shape[0]) fail("cumextrema shape mismatch");
  const int64_t N = in_shape[0];
  if (N <= 0) fail("cumextrema invalid N");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + IN + ", float* __restrict__ " + OUT + ") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("float acc = " + IN + "[0];");
  w.line(OUT + "[0] = acc;");
  w.line("for (int64_t i = 1; i < " + std::to_string(N) + "LL; ++i) {");
  w.indent();
  w.line(std::string("acc = ") + (is_max ? "fmaxf(acc, " : "fminf(acc, ") + IN + "[i]);");
  w.line(OUT + "[i] = acc;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {IN, OUT}, {}, {IN, OUT});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {OUT};
  out["bindings"] = out_bindings;
  return out;
}

