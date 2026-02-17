	  void emit_prelude() {
	    int64_t vw = 0;
	    if (sched_vec_width && *sched_vec_width > 0) vw = *sched_vec_width;
	    w.pp_line("#ifndef INTENTIR_VEC_WIDTH");
	    w.pp_line("#define INTENTIR_VEC_WIDTH " + std::to_string(vw));
	    w.pp_line("#endif");
	    w.pp_line("#include \"intentir_driver.h\"");
	    w.pp_line("#include \"intentir_ops.h\"");
	    w.blank();
	  }

  void emit_globals() {
    std::vector<std::string> names;
    names.reserve(shape_env.size());
    for (const auto& kv : shape_env) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    build_cvars(names);
    w.line("// intent tensors (globals)");
    for (const auto& name : names) {
      std::string ct = ctype_for_dtype(dtype_env.at(name));
      w.line("static " + ct + "* " + v(name) + " = NULL;");
    }
    w.blank();

    // Optional per-op profiling (Task6 perf tooling). Enabled by env:
    //   INTENTIR_PROFILE_OPS=1
    // Output:
    //   INTENTIR_PROFILE {"total_ns":...,"ops":[{"name":"...","ns":...},...]}
    std::vector<std::string> prof_names;
    prof_names.reserve(intent.ops.size());
    for (size_t i = 0; i < intent.ops.size(); ++i) {
      const auto& op = intent.ops[i];
      if (op.op == "const") continue;
      if (op.op == "reshape" || op.op == "layout_cast") continue;
      prof_names.push_back("op[" + std::to_string(i) + "] " + op.op + " -> " + op.output);
    }
    w.line("// profiler state (optional)");
    w.line("static int intentir_profile_enabled = 0;");
    if (!prof_names.empty()) {
      w.line("static const char* intentir_profile_names[] = {");
      w.indent();
      for (const auto& n : prof_names) {
        w.line("\"" + n + "\",");
      }
      w.dedent();
      w.line("};");
      w.line("static uint64_t intentir_profile_ns[sizeof(intentir_profile_names) / sizeof(intentir_profile_names[0])];");
    }
    w.line("static void intentir_print_profile(void) {");
    w.indent();
    w.line("if (!intentir_profile_enabled) return;");
    if (prof_names.empty()) {
      w.line("printf(\"INTENTIR_PROFILE {\\\"total_ns\\\":0,\\\"ops\\\":[]}\\n\");");
      w.line("return;");
    } else {
      w.line("const size_t n = (size_t)(sizeof(intentir_profile_names) / sizeof(intentir_profile_names[0]));");
      w.line("uint64_t total = 0;");
      w.line("for (size_t i = 0; i < n; ++i) total += intentir_profile_ns[i];");
      w.line("printf(\"INTENTIR_PROFILE {\\\"total_ns\\\":%llu,\\\"ops\\\":[\", (unsigned long long)total);");
      w.line("for (size_t i = 0; i < n; ++i) {");
      w.indent();
      w.line("if (i) printf(\",\");");
      w.line("printf(\"{\\\"name\\\":\\\"%s\\\",\\\"ns\\\":%llu}\", intentir_profile_names[i], (unsigned long long)intentir_profile_ns[i]);");
      w.dedent();
      w.line("}");
      w.line("printf(\"]}\\n\");");
      w.line("return;");
    }
    w.dedent();
    w.line("}");
    w.blank();
  }

		  void emit_alloc_tensor(const std::string& name) {
		    const auto& shp = shape_env.at(name);
		    int64_t n = numel(shp);
		    std::string ct = ctype_for_dtype(dtype_env.at(name));
		    const std::string& var = v(name);
		    w.line(var + " = (" + ct + "*)malloc(sizeof(" + ct + ") * (size_t)" + std::to_string(n) + ");");
	    w.line("if (!" + var + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
	  }

	  void emit_main() {
	    w.line("int main() {");
	    w.indent();

	    w.line("setvbuf(stdout, NULL, _IONBF, 0);");
	    w.line("setvbuf(stderr, NULL, _IONBF, 0);");
	    w.line("intentir_runtime_init();");
	    w.line("const char* __intentir_prof = getenv(\"INTENTIR_PROFILE_OPS\");");
	    w.line("if (__intentir_prof && atoi(__intentir_prof) > 0) intentir_profile_enabled = 1;");
	    w.line("const float ATOL = " + c_float(atol) + ";");
	    w.line("const float RTOL = " + c_float(rtol) + ";");
	    {
	      std::ostringstream oss;
	      oss.setf(std::ios::fixed);
	      oss.precision(1);
	      oss << matmul_flops_total;
	      w.line("const double MATMUL_FLOPS = (double)" + oss.str() + ";");
	    }
	    w.blank();

	    // inputs
	    if (!external_inputs.empty()) {
	      w.line("IntentirBufferDesc inputs[] = {");
	      w.indent();
		      for (const auto& name : external_inputs) {
		        const auto& shp = shape_env.at(name);
		        int64_t n = numel(shp);
		        std::string ct = ctype_for_dtype(dtype_env.at(name));
		        std::string dt = dtype_env.at(name);
		        const std::string& var = v(name);
		        std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
		        if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
		        w.line("{\"" + name + "\", (void**)&" + var + ", (size_t)(" + bytes_expr + "), " +
		               buffer_desc_dtype_for_dtype(dt) + "},");
		      }
	      w.dedent();
	      w.line("};");
		      if (!bench_only) {
		        w.line("if (!intentir_alloc_and_load_inputs(inputs, (size_t)(sizeof(inputs) / sizeof(inputs[0])))) return 2;");
		      } else {
		        w.line("if (!intentir_alloc(inputs, (size_t)(sizeof(inputs) / sizeof(inputs[0])))) return 2;");
		        w.line("uint64_t intentir_bench_seed = 0;");
		        w.line("const char* __intentir_bs = getenv(\"INTENTIR_BENCH_SEED\");");
		        w.line("if (__intentir_bs && *__intentir_bs) intentir_bench_seed = (uint64_t)strtoull(__intentir_bs, NULL, 10);");
		        w.line("for (size_t bi = 0; bi < (size_t)(sizeof(inputs) / sizeof(inputs[0])); ++bi) {");
		        w.indent();
	        w.line("IntentirBufferDesc* b = &inputs[bi];");
	        w.line("if (!b->ptr || !*b->ptr) return 2;");
	        w.line("if (b->dtype == INTENTIR_DTYPE_U8) {");
	        w.indent();
	        w.line("uint8_t* p = (uint8_t*)(*b->ptr);");
	        w.line("for (size_t i = 0; i < b->bytes; ++i) p[i] = (uint8_t)(((uint64_t)i + intentir_bench_seed) & 1u);");
	        w.dedent();
	        w.line("} else {");
	        w.indent();
	        w.line("float* p = (float*)(*b->ptr);");
	        w.line("size_t n = b->bytes / sizeof(float);");
	        w.line("for (size_t i = 0; i < n; ++i) {");
	        w.indent();
	        w.line("size_t j = (size_t)((uint64_t)i + intentir_bench_seed);");
	        w.line("int t = (int)(j % 127u) - 63;");
	        w.line("p[i] = (float)t * 0.01f;");
	        w.dedent();
	        w.line("}");
		        w.dedent();
		        w.line("}");
		        w.dedent();
		        w.line("}");
		      }
		      w.blank();
		    }

	    // consts as 1-element arrays (deterministic order for reproducible C).
	    {
	      std::vector<std::string> cnames;
      cnames.reserve(const_vals.size());
      for (const auto& kv : const_vals) cnames.push_back(kv.first);
      std::sort(cnames.begin(), cnames.end());
	      for (const auto& name : cnames) {
	        const ConstVal& cv = const_vals.at(name);
	        const std::string ct = ctype_for_dtype(cv.dtype);
	        const std::string& var = v(name);
	        w.line(var + " = (" + ct + "*)malloc(sizeof(" + ct + "));");
	        w.line("if (!" + var + ") { fprintf(stderr, \"alloc failed: " + name + "\\n\"); return 2; }");
	        if (ct == "float") {
	          w.line(var + "[0] = " + c_float(cv.value) + ";");
	        } else if (ct == "double") {
	          std::ostringstream oss;
	          oss.precision(17);
	          oss << cv.value;
	          w.line(var + "[0] = (double)" + oss.str() + ";");
	        } else {
	          w.line(var + "[0] = (" + ct + ")" + std::to_string((int64_t)cv.value) + ";");
	        }
	        w.blank();
	      }
	    }

	    // allocate outputs / set alias views
		    for (const auto& op : intent.ops) {
		      if (op.op == "const") continue;
	      const std::string& out = op.output;
	      const std::string& out_var = v(out);
	      const bool identity_alias =
	          (op.op == "identity" && !op.inputs.empty() && dtype_env.at(out) == dtype_env.at(op.inputs[0]));
	      if (op.op == "reshape" || op.op == "layout_cast" || identity_alias) {
	        std::string ct = ctype_for_dtype(dtype_env.at(out));
	        const std::string& in0_var = v(op.inputs[0]);
	        w.line("// " + op.op + " alias");
	        w.line(out_var + " = (" + ct + "*)" + in0_var + ";");
	        w.blank();
	        continue;
	      }
	      w.line("// op " + op.op + " -> " + out);
	      emit_alloc_tensor(out);
	      w.blank();
		    }

	    // compute once
	    w.line("intentir_compute();");
	    w.line("intentir_print_profile();");
	    w.line("if (intentir_profile_enabled) intentir_profile_enabled = 0;");
	    w.line("intentir_maybe_bench(intentir_compute, MATMUL_FLOPS);");
	    w.blank();

	    // compare outputs
	    if (bench_only) {
	      w.line("printf(\"PASS bench_only\\n\");");
	      w.line("return 0;");
	    } else {
	      if (intent.outputs.empty()) {
	        w.line("int ok = 1;");
	      } else {
	        w.line("IntentirBufferDesc outputs[] = {");
	        w.indent();
		        for (const auto& name : intent.outputs) {
		          const auto& shp = shape_env.at(name);
		          int64_t n = numel(shp);
		          std::string ct = ctype_for_dtype(dtype_env.at(name));
		          std::string dt = dtype_env.at(name);
		          const std::string& var = v(name);
		          std::string bytes_expr = "sizeof(" + ct + ") * (size_t)" + std::to_string(n);
		          if (dt == "bool" || dt == "i1") bytes_expr = "sizeof(uint8_t) * (size_t)" + std::to_string(n);
		          w.line("{\"" + name + "\", (void**)&" + var + ", (size_t)(" + bytes_expr + "), " +
		                 buffer_desc_dtype_for_dtype(dt) + "},");
		        }
	        w.dedent();
	        w.line("};");
	        w.line("int ok = intentir_compare_outputs_with_refs(outputs, (size_t)(sizeof(outputs) / sizeof(outputs[0])), ATOL, RTOL);");
	      }
	      w.line("printf(ok ? \"PASS lowered\\n\" : \"FAIL lowered\\n\");");
	      w.line("return ok ? 0 : 1;");
	    }

	    w.dedent();
	    w.line("}");
	  }

		  void emit_broadcast_in_dim(const Op& op, const std::vector<int64_t>& out_shape) {
		    const std::string& out = op.output;
		    const std::string& inp = op.inputs[0];
		    const std::string& out_var = v(out);
		    const std::string& inp_var = v(inp);
		    const auto& in_shape = shape_env.at(inp);
		    std::vector<int> bcast_dims;
		    for (const auto& d : op.attrs["broadcast_dims"]) bcast_dims.push_back(d.get<int>());

	    const std::string in_ct = ctype_for_dtype(dtype_env.at(inp));
	    const std::string out_ct = ctype_for_dtype(dtype_env.at(out));
	    const int in_rank = (int)in_shape.size();
	    const int out_rank = (int)out_shape.size();
	    if (in_ct == "float" && out_ct == "float" && in_rank <= 4 && out_rank <= 4) {
	      auto arr64 = [&](const std::vector<int64_t>& v) -> std::string {
	        std::string s = "(int64_t[]){";
	        for (size_t i = 0; i < v.size(); ++i) {
	          if (i) s += ",";
	          s += std::to_string(v[i]);
	        }
	        s += "}";
	        return s;
	      };
	      auto arri = [&](const std::vector<int>& v) -> std::string {
	        std::string s = "(int[]){";
	        for (size_t i = 0; i < v.size(); ++i) {
	          if (i) s += ",";
	          s += std::to_string(v[i]);
	        }
	        s += "}";
	        return s;
	      };
		      w.line("intentir_broadcast_in_dim_f32(" + inp_var + ", " + out_var + ", " + arr64(in_shape) + ", " + std::to_string(in_rank) + ", " + arr64(out_shape) + ", " +
		             std::to_string(out_rank) + ", " + arri(bcast_dims) + ");");
		      return;
		    }

	    std::vector<int64_t> strides(in_shape.size(), 1);
	    int64_t s = 1;
	    for (int i = (int)in_shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= in_shape[i];
    }
    const int r_out = (int)out_shape.size();
    if (r_out > 4) fail("broadcast_in_dim supports rank<=4");
    std::vector<std::string> iv = {"i0", "i1", "i2", "i3"};
    iv.resize(r_out);
    for (int i = 0; i < r_out; ++i) {
      w.line("for (int " + iv[i] + " = 0; " + iv[i] + " < " + std::to_string(out_shape[i]) + "; ++" + iv[i] + ") {");
      w.indent();
    }
    std::string out_idx = flat_idx_expr(iv, out_shape);
    std::vector<std::string> terms;
    for (int in_dim = 0; in_dim < (int)in_shape.size(); ++in_dim) {
      int od = bcast_dims[in_dim];
      if (in_shape[in_dim] == 1) terms.push_back("0");
      else terms.push_back("((size_t)" + iv[od] + " * (size_t)" + std::to_string(strides[in_dim]) + ")");
    }
    std::string in_idx = terms.empty() ? "0" : terms[0];
    for (size_t t = 1; t < terms.size(); ++t) in_idx += " + " + terms[t];
	    w.line(out_var + "[" + out_idx + "] = " + inp_var + "[" + in_idx + "];");
    for (int i = 0; i < r_out; ++i) {
      w.dedent();
      w.line("}");
    }
  }
