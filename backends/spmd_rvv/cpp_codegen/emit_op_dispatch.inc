      } else if (op.op == "transpose") {
		        const auto& in_shape = shape_env.at(op.inputs[0]);
		        std::vector<int> perm;
		        for (const auto& p : op.attrs["perm"]) perm.push_back(p.get<int>());
		        const std::string in_ct = ctype_for_dtype(dtype_env.at(op.inputs[0]));
		        const std::string out_ct = ctype_for_dtype(dtype_env.at(out));
		        if (in_ct == "float" && out_ct == "float") {
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
		          if (perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2 && in_shape.size() == 4) {
		            w.line("intentir_transpose_4d_0132_f32(" + v(op.inputs[0]) + ", " + out_var + ", " + std::to_string(in_shape[0]) + ", " +
		                   std::to_string(in_shape[1]) + ", " + std::to_string(in_shape[2]) + ", " + std::to_string(in_shape[3]) + ");");
		          } else {
		            w.line("intentir_transpose_f32(" + v(op.inputs[0]) + ", " + out_var + ", " + arr64(in_shape) + ", " + arr64(out_shape) + ", " + arri(perm) +
		                   ", " + std::to_string((int)in_shape.size()) + ");");
		          }
		        } else {
		          if (perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2) {
		            emit_transpose_4d_0132(w, out_var, v(op.inputs[0]), in_shape, out_shape);
		          } else {
		            emit_transpose_generic(w, out_var, v(op.inputs[0]), in_shape, out_shape, perm);
		          }
		        }
		      } else if (op.op == "broadcast_in_dim") {
			        emit_broadcast_in_dim(op, out_shape);
	      } else if (op.op == "concat") {
	        if (op.inputs.size() != 2) fail("concat currently supports exactly 2 inputs");
	        const auto& a_shape = shape_env.at(op.inputs[0]);
	        const auto& b_shape = shape_env.at(op.inputs[1]);
	        if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 2) fail("concat currently supports rank-2 tensors");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("concat currently supports only f32 tensors");
	        int axis = 0;
	        if (op.attrs.contains("axis")) axis = (int)resolve_const_value(op.attrs["axis"], bindings);
	        if (axis < 0) axis += 2;
	        if (axis < 0 || axis >= 2) fail("concat axis out of range for rank-2 tensor");
	        if (axis == 1) {
	          if (a_shape[0] != b_shape[0]) fail("concat axis=1 requires matching row count");
	          if (out_shape[0] != a_shape[0] || out_shape[1] != (a_shape[1] + b_shape[1])) fail("concat output shape mismatch");
	          w.line("for (int i = 0; i < " + std::to_string(out_shape[0]) + "; ++i) {");
	          w.indent();
	          w.line("for (int j = 0; j < " + std::to_string(out_shape[1]) + "; ++j) {");
	          w.indent();
	          w.line("if (j < " + std::to_string(a_shape[1]) + ") " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[0]) +
	                 "[i * " + std::to_string(a_shape[1]) + " + j];");
	          w.line("else " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[1]) + "[i * " + std::to_string(b_shape[1]) +
	                 " + (j - " + std::to_string(a_shape[1]) + ")];");
	          w.dedent();
	          w.line("}");
	          w.dedent();
	          w.line("}");
	        } else {
	          if (a_shape[1] != b_shape[1]) fail("concat axis=0 requires matching column count");
	          if (out_shape[1] != a_shape[1] || out_shape[0] != (a_shape[0] + b_shape[0])) fail("concat output shape mismatch");
	          w.line("for (int i = 0; i < " + std::to_string(out_shape[0]) + "; ++i) {");
	          w.indent();
	          w.line("for (int j = 0; j < " + std::to_string(out_shape[1]) + "; ++j) {");
	          w.indent();
	          w.line("if (i < " + std::to_string(a_shape[0]) + ") " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[0]) +
	                 "[i * " + std::to_string(a_shape[1]) + " + j];");
	          w.line("else " + out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[1]) + "[(i - " + std::to_string(a_shape[0]) +
	                 ") * " + std::to_string(b_shape[1]) + " + j];");
	          w.dedent();
	          w.line("}");
	          w.dedent();
	          w.line("}");
	        }
	      } else if (op.op == "pad") {
	        if (op.inputs.size() != 1) fail("pad expects exactly 1 input tensor");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        if (in_shape.size() != 2 || out_shape.size() != 2) fail("pad currently supports rank-2 tensors");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("pad currently supports only f32 tensors");
	        std::string mode = op.attrs.value("mode", "constant");
	        if (mode != "constant") fail("pad currently supports mode=constant");
	        json padw = op.attrs.value("pad_width", json{});
	        if (padw.is_object() && padw.contains("pairs")) padw = padw["pairs"];
	        if (!padw.is_array() || padw.size() != 2) fail("pad_width must be rank-2 pairs");
	        const auto& p0 = padw[0];
	        const auto& p1 = padw[1];
	        if (!p0.is_array() || p0.size() != 2 || !p1.is_array() || p1.size() != 2) fail("pad_width pairs must each contain 2 elements");
	        const int pad_top = (int)resolve_const_value(p0[0], bindings);
	        const int pad_bottom = (int)resolve_const_value(p0[1], bindings);
	        const int pad_left = (int)resolve_const_value(p1[0], bindings);
	        const int pad_right = (int)resolve_const_value(p1[1], bindings);
	        (void)pad_bottom;
	        (void)pad_right;
	        const double pad_val = op.attrs.contains("value") ? resolve_const_value(op.attrs["value"], bindings) : 0.0;
	        w.line("for (int i = 0; i < " + std::to_string(out_shape[0]) + "; ++i) {");
	        w.indent();
	        w.line("for (int j = 0; j < " + std::to_string(out_shape[1]) + "; ++j) {");
	        w.indent();
	        w.line("const int src_i = i - " + std::to_string(pad_top) + ";");
	        w.line("const int src_j = j - " + std::to_string(pad_left) + ";");
	        w.line("if (src_i >= 0 && src_i < " + std::to_string(in_shape[0]) + " && src_j >= 0 && src_j < " + std::to_string(in_shape[1]) + ") {");
	        w.indent();
	        w.line(out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = " + v(op.inputs[0]) + "[src_i * " + std::to_string(in_shape[1]) + " + src_j];");
	        w.dedent();
	        w.line("} else {");
	        w.indent();
	        w.line(out_var + "[i * " + std::to_string(out_shape[1]) + " + j] = (float)(" + std::to_string(pad_val) + ");");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "add" || op.op == "sub" || op.op == "mul" || op.op == "div" || op.op == "max" || op.op == "min" || op.op == "pow" || op.op == "remainder") {
		        if (op.inputs.size() == 2) {
		          emit_elemwise_bin(w, intent, bindings, op.op, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                            shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape,
		                            dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
		        } else if (op.inputs.size() == 1) {
	          // Legacy scalar form (1 input + scalar attr), used by some LLM outputs and
	          // by deterministic intent builders for reductions (mean/var normalization).
          std::string key;
          if (op.op == "div" && op.attrs.contains("divisor")) key = "divisor";
          else if (op.op == "add" && op.attrs.contains("addend")) key = "addend";
          else if (op.op == "sub" && op.attrs.contains("subtract")) key = "subtract";
          else if (op.op == "mul" && op.attrs.contains("mul_factor")) key = "mul_factor";
          else if ((op.op == "max" || op.op == "min") && op.attrs.contains("other")) key = "other";
          else fail("elemwise op requires 2 inputs (or 1 + scalar attr)");
	          if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("elemwise scalar path supports only f32");
	          if (shape_env.at(op.inputs[0]) != out_shape) fail("elemwise scalar expects input shape == output shape");
	          double scalar = resolve_const_value(op.attrs[key], bindings);
	          emit_elemwise_scalar_f32(w, op.op, out_var, v(op.inputs[0]), out_shape, scalar);
		        } else {
		          fail("elemwise op requires 2 inputs (or 1 + scalar attr)");
		        }
		      } else if (op.op == "ne") {
		        emit_ne(w, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
		                dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
	      } else if (op.op == "eq" || op.op == "lt" || op.op == "le" || op.op == "gt" || op.op == "ge") {
		        emit_cmp(w, op.op, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                 shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
		                 dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
      } else if (op.op == "and" || op.op == "or" || op.op == "bitwise_and" || op.op == "bitwise_or") {
        const std::string logical_op = (op.op == "bitwise_and") ? "and" : ((op.op == "bitwise_or") ? "or" : op.op);
        emit_bool_bin(w, logical_op, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
		                      shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
		                      dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
		      } else if (op.op == "not" || op.op == "bitwise_not") {
		        emit_bool_not(w, out_var, v(op.inputs[0]), out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "bitwise_left_shift") {
	        emit_bitwise_left_shift(w, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
	                                shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
	                                dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
	      } else if (op.op == "bitwise_right_shift") {
	        emit_bitwise_right_shift(w, out_var, v(op.inputs[0]), v(op.inputs[1]), op.inputs[0], op.inputs[1],
	                                 shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, intent, bindings,
	                                 dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(out));
	      } else if (op.op == "rsqrt") {
	        emit_rsqrt(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "sqrt") {
	        emit_sqrt(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "abs") {
	        emit_unary_abs(w, out_var, v(op.inputs[0]), out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "floor") {
	        emit_floor(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "ceil") {
	        emit_ceil(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "acos") {
	        emit_acos(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "atan") {
	        emit_atan(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "tan") {
	        emit_tan(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "sin") {
	        emit_sin(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "cos") {
	        emit_cos(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "erf") {
	        emit_erf(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "cast") {
	        emit_cast(w, out_var, v(op.inputs[0]), out_shape, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "where") {
	        emit_where(w, out_var, v(op.inputs[0]), v(op.inputs[1]), v(op.inputs[2]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), shape_env.at(op.inputs[2]), out_shape,
	                   dtype_env.at(op.inputs[0]), dtype_env.at(op.inputs[1]), dtype_env.at(op.inputs[2]), dtype_env.at(out));
	      } else if (op.op == "polar") {
	        if (op.inputs.size() != 2) fail("polar expects 2 inputs [abs, angle]");
	        const auto& abs_shape = shape_env.at(op.inputs[0]);
	        const auto& angle_shape = shape_env.at(op.inputs[1]);
	        if (abs_shape != angle_shape) fail("polar expects abs/angle shape match");
	        if (abs_shape.size() != 2 || out_shape.size() != 3) fail("polar currently supports [M,N] -> [M,N,2]");
	        if (out_shape[0] != abs_shape[0] || out_shape[1] != abs_shape[1] || out_shape[2] != 2) {
	          fail("polar output shape mismatch");
	        }
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("polar currently supports only f32");
	        }
	        const int64_t M = abs_shape[0];
	        const int64_t N = abs_shape[1];
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        w.line("const size_t idx = (size_t)i * " + std::to_string(N) + " + (size_t)j;");
	        w.line("const float r = " + v(op.inputs[0]) + "[idx];");
	        w.line("const float a0 = " + v(op.inputs[1]) + "[idx];");
	        w.line(out_var + "[idx * 2 + 0] = r * cosf(a0);");
	        w.line(out_var + "[idx * 2 + 1] = r * sinf(a0);");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "cumsum") {
	        int axis = op.attrs.value("axis", 0);
	        emit_cumsum(w, out_var, v(op.inputs[0]), out_shape, axis, dtype_env.at(out));
	      } else if (op.op == "cummax") {
	        int axis = op.attrs.value("axis", 0);
	        emit_cumext_1d(w, out_var, v(op.inputs[0]), out_shape, axis, dtype_env.at(out), /*is_max=*/true);
	      } else if (op.op == "cummin") {
	        int axis = op.attrs.value("axis", 0);
	        emit_cumext_1d(w, out_var, v(op.inputs[0]), out_shape, axis, dtype_env.at(out), /*is_max=*/false);
	      } else if (op.op == "scatter") {
	        if (op.inputs.size() != 3) fail("scatter expects inputs [inp, index, src]");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        const auto& idx_shape = shape_env.at(op.inputs[1]);
	        const auto& src_shape = shape_env.at(op.inputs[2]);
	        if (in_shape.size() != 2 || idx_shape.size() != 2 || src_shape.size() != 2 || out_shape.size() != 2) {
	          fail("scatter currently supports rank-2 tensors only");
	        }
	        if (idx_shape != in_shape || src_shape != in_shape || out_shape != in_shape) {
	          fail("scatter shape mismatch: expected inp/index/src/out all [M,N]");
	        }
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("scatter currently supports f32 inp/src/out");
	        }
	        const std::string idx_dt = dtype_env.at(op.inputs[1]);
	        if (!(idx_dt == "i32" || idx_dt == "i64")) fail("scatter index dtype must be i32/i64");
	        int axis = op.attrs.value("dim", op.attrs.value("axis", 0));
	        if (axis < 0) axis += 2;
	        if (axis != 1) fail("scatter currently supports dim=1 only");
	        const int64_t M = in_shape[0];
	        const int64_t N = in_shape[1];
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)j] = " + v(op.inputs[0]) + "[(size_t)i * (size_t)" + std::to_string(N) +
	               " + (size_t)j];");
	        w.dedent();
	        w.line("}");
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        w.line("const int64_t dst = (int64_t)" + v(op.inputs[1]) + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)j];");
	        w.line("if (dst >= 0 && dst < " + std::to_string(N) + ") {");
	        w.indent();
	        w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)dst] = " + v(op.inputs[2]) + "[(size_t)i * (size_t)" + std::to_string(N) +
	               " + (size_t)j];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "select_scatter") {
	        if (op.inputs.size() != 2) fail("select_scatter expects inputs [inp, src]");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        const auto& src_shape = shape_env.at(op.inputs[1]);
	        if (in_shape.size() != 2 || out_shape.size() != 2 || src_shape.size() != 1) {
	          fail("select_scatter currently supports inp/out rank-2 and src rank-1");
	        }
	        if (in_shape != out_shape) fail("select_scatter output shape must equal input shape");
	        if (src_shape[0] != in_shape[0]) fail("select_scatter src shape mismatch: expected [M]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("select_scatter currently supports f32 tensors");
	        }
	        int axis = op.attrs.value("dim", op.attrs.value("axis", 0));
	        if (axis < 0) axis += 2;
	        if (axis != 1) fail("select_scatter currently supports dim=1 only");
	        int64_t idx = 0;
	        if (op.attrs.contains("index")) idx = static_cast<int64_t>(resolve_const_value(op.attrs["index"], bindings));
	        const int64_t M = in_shape[0];
	        const int64_t N = in_shape[1];
	        if (idx < 0 || idx >= N) fail("select_scatter index out of range");
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)j] = " + v(op.inputs[0]) + "[(size_t)i * (size_t)" + std::to_string(N) +
	               " + (size_t)j];");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)" + std::to_string(idx) + "] = " + v(op.inputs[1]) + "[(size_t)i];");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "slice_scatter") {
	        if (op.inputs.size() != 2) fail("slice_scatter expects inputs [inp, src]");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        const auto& src_shape = shape_env.at(op.inputs[1]);
	        if (in_shape.size() != 2 || out_shape.size() != 2 || src_shape.size() != 2) {
	          fail("slice_scatter currently supports rank-2 tensors");
	        }
	        if (in_shape != out_shape) fail("slice_scatter output shape must equal input shape");
	        if (src_shape[0] != in_shape[0]) fail("slice_scatter src first dim must match M");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("slice_scatter currently supports f32 tensors");
	        }
	        int axis = op.attrs.value("dim", op.attrs.value("axis", 0));
	        if (axis < 0) axis += 2;
	        if (axis != 1) fail("slice_scatter currently supports dim=1 only");
	        int64_t start = op.attrs.contains("start") ? static_cast<int64_t>(resolve_const_value(op.attrs["start"], bindings)) : 0;
	        int64_t end = op.attrs.contains("end") ? static_cast<int64_t>(resolve_const_value(op.attrs["end"], bindings)) : in_shape[1];
	        int64_t step = op.attrs.contains("step") ? static_cast<int64_t>(resolve_const_value(op.attrs["step"], bindings)) : 1;
	        if (step <= 0) fail("slice_scatter requires step > 0");
	        const int64_t M = in_shape[0];
	        const int64_t N = in_shape[1];
	        if (end > N) end = N;
	        if (start < 0) start += N;
	        if (end < 0) end += N;
	        if (start < 0) start = 0;
	        if (end < 0) end = 0;
	        const int64_t span = (end > start) ? (end - start) : 0;
	        const int64_t expected_L = (span + step - 1) / step;
	        if (src_shape[1] != expected_L) fail("slice_scatter src second dim mismatch with (start,end,step)");
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)j] = " + v(op.inputs[0]) + "[(size_t)i * (size_t)" + std::to_string(N) +
	               " + (size_t)j];");
	        w.dedent();
	        w.line("}");
	        w.line("for (int64_t k = 0; k < " + std::to_string(expected_L) + "; ++k) {");
	        w.indent();
	        w.line("const int64_t dst = " + std::to_string(start) + " + k * " + std::to_string(step) + ";");
	        w.line("if (dst >= 0 && dst < " + std::to_string(N) + ") {");
	        w.indent();
	        w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(N) + " + (size_t)dst] = " + v(op.inputs[1]) + "[(size_t)i * (size_t)" +
	               std::to_string(expected_L) + " + (size_t)k];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "nonzero") {
	        if (op.inputs.size() != 1) fail("nonzero expects 1 input");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        if (in_shape.size() != 2 || out_shape.size() != 2) fail("nonzero currently supports in[2] -> out[2]");
	        if (out_shape[1] != 2) fail("nonzero expects output shape [K, 2]");
	        if (dtype_env.at(out) != "i64") fail("nonzero currently supports i64 output");
	        const std::string in_dt = dtype_env.at(op.inputs[0]);
	        const std::string in_ct = ctype_for_dtype(in_dt);
	        const int64_t M = in_shape[0];
	        const int64_t N = in_shape[1];
	        const int64_t K = out_shape[0];
	        w.line("int64_t write_pos = 0;");
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) {");
	        w.indent();
	        if (in_dt == "bool" || in_dt == "i1" || in_dt == "u8") {
	          w.line("const int nz = (" + v(op.inputs[0]) + "[(size_t)(i * " + std::to_string(N) + " + j)] != (uint8_t)0);");
	        } else {
	          w.line("const int nz = (" + v(op.inputs[0]) + "[(size_t)(i * " + std::to_string(N) + " + j)] != (" + in_ct + ")0);");
	        }
	        w.line("if (nz) {");
	        w.indent();
	        w.line("if (write_pos < " + std::to_string(K) + ") {");
	        w.indent();
	        w.line(out_var + "[(size_t)write_pos * 2 + 0] = (int64_t)i;");
	        w.line(out_var + "[(size_t)write_pos * 2 + 1] = (int64_t)j;");
	        w.dedent();
	        w.line("}");
	        w.line("write_pos += 1;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line("for (int64_t t = write_pos; t < " + std::to_string(K) + "; ++t) {");
	        w.indent();
	        w.line(out_var + "[(size_t)t * 2 + 0] = 0;");
	        w.line(out_var + "[(size_t)t * 2 + 1] = 0;");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "nll_loss_forward") {
	        if (op.inputs.size() != 3) fail("nll_loss_forward expects inputs [self, target, weight]");
	        const auto& logits_shape = shape_env.at(op.inputs[0]);
	        const auto& target_shape = shape_env.at(op.inputs[1]);
	        const auto& weight_shape = shape_env.at(op.inputs[2]);
	        if (logits_shape.size() != 2 || target_shape.size() != 1 || weight_shape.size() != 1) {
	          fail("nll_loss_forward expects self[N,C], target[N], weight[C]");
	        }
	        const int64_t N = logits_shape[0];
	        const int64_t C = logits_shape[1];
	        if (target_shape[0] != N) fail("nll_loss_forward target length mismatch");
	        if (weight_shape[0] != C) fail("nll_loss_forward weight length mismatch");
	        const std::string target_dt = dtype_env.at(op.inputs[1]);
	        if (!(target_dt == "i64" || target_dt == "i32")) fail("nll_loss_forward target dtype must be i64/i32");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("nll_loss_forward currently supports f32 logits/weight/output");
	        }
	        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
	        const int ignore_index = op.attrs.contains("ignore_index") ? (int)resolve_const_value(op.attrs["ignore_index"], bindings) : -100;
	        if (reduction == 0) fail("nll_loss_forward reduction=none is not supported in RVV lowering yet");
	        w.line("double loss_sum = 0.0;");
	        w.line("double weight_sum = 0.0;");
	        w.line("for (int64_t n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("const int64_t t = (int64_t)" + v(op.inputs[1]) + "[(size_t)n];");
	        w.line("if (t == (int64_t)" + std::to_string(ignore_index) + ") continue;");
	        w.line("if (t < 0 || t >= " + std::to_string(C) + ") continue;");
	        w.line("const float wv = " + v(op.inputs[2]) + "[(size_t)t];");
	        w.line("const float lv = -" + v(op.inputs[0]) + "[(size_t)n * " + std::to_string(C) + " + (size_t)t] * wv;");
	        w.line("loss_sum += (double)lv;");
	        w.line("weight_sum += (double)wv;");
	        w.dedent();
	        w.line("}");
	        if (reduction == 1) {
	          w.line(out_var + "[0] = (weight_sum > 0.0) ? (float)(loss_sum / weight_sum) : 0.0f;");
	        } else {
	          w.line(out_var + "[0] = (float)loss_sum;");
	        }
	      } else if (op.op == "nll_loss2d_forward") {
	        if (op.inputs.size() != 3) fail("nll_loss2d_forward expects inputs [self, target, weight]");
	        const auto& logits_shape = shape_env.at(op.inputs[0]);
	        const auto& target_shape = shape_env.at(op.inputs[1]);
	        const auto& weight_shape = shape_env.at(op.inputs[2]);
	        if (logits_shape.size() != 4 || target_shape.size() != 3 || weight_shape.size() != 1) {
	          fail("nll_loss2d_forward expects self[N,C,H,W], target[N,H,W], weight[C]");
	        }
	        const int64_t N = logits_shape[0];
	        const int64_t C = logits_shape[1];
	        const int64_t H = logits_shape[2];
	        const int64_t W = logits_shape[3];
	        if (target_shape[0] != N || target_shape[1] != H || target_shape[2] != W) fail("nll_loss2d_forward target shape mismatch");
	        if (weight_shape[0] != C) fail("nll_loss2d_forward weight length mismatch");
	        const std::string target_dt = dtype_env.at(op.inputs[1]);
	        if (!(target_dt == "i64" || target_dt == "i32")) fail("nll_loss2d_forward target dtype must be i64/i32");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("nll_loss2d_forward currently supports f32 logits/weight/output");
	        }
	        const int reduction = op.attrs.contains("reduction") ? (int)resolve_const_value(op.attrs["reduction"], bindings) : 1;
	        const int ignore_index = op.attrs.contains("ignore_index") ? (int)resolve_const_value(op.attrs["ignore_index"], bindings) : -100;
	        if (reduction == 0) fail("nll_loss2d_forward reduction=none is not supported in RVV lowering yet");
	        w.line("double loss_sum = 0.0;");
	        w.line("double weight_sum = 0.0;");
	        w.line("for (int64_t n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int64_t h = 0; h < " + std::to_string(H) + "; ++h) {");
	        w.indent();
	        w.line("for (int64_t ww = 0; ww < " + std::to_string(W) + "; ++ww) {");
	        w.indent();
	        w.line("const int64_t t = (int64_t)" + v(op.inputs[1]) + "[((size_t)n * " + std::to_string(H) + " + (size_t)h) * " + std::to_string(W) + " + (size_t)ww];");
	        w.line("if (t == (int64_t)" + std::to_string(ignore_index) + ") continue;");
	        w.line("if (t < 0 || t >= " + std::to_string(C) + ") continue;");
	        w.line("const float wv = " + v(op.inputs[2]) + "[(size_t)t];");
	        w.line("const size_t idx = (((size_t)n * " + std::to_string(C) + " + (size_t)t) * " + std::to_string(H) + " + (size_t)h) * " + std::to_string(W) + " + (size_t)ww;");
	        w.line("const float lv = -" + v(op.inputs[0]) + "[idx] * wv;");
	        w.line("loss_sum += (double)lv;");
	        w.line("weight_sum += (double)wv;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        if (reduction == 1) {
	          w.line(out_var + "[0] = (weight_sum > 0.0) ? (float)(loss_sum / weight_sum) : 0.0f;");
	        } else {
	          w.line(out_var + "[0] = (float)loss_sum;");
	        }
	      } else if (op.op == "conv1d") {
	        if (op.inputs.size() != 3) fail("conv1d expects inputs [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const auto& b_shape = shape_env.at(op.inputs[2]);
	        if (x_shape.size() != 3 || w_shape.size() != 3 || b_shape.size() != 1 || out_shape.size() != 3)
	          fail("conv1d expects input[N,C,L], weight[CO,C_PER_G,K], bias[CO], out[N,CO,OL]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv1d currently supports f32 only");
	        const int64_t N = x_shape[0];
	        const int64_t C_IN = x_shape[1];
	        const int64_t L = x_shape[2];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t K = w_shape[2];
	        if (b_shape[0] != C_OUT) fail("conv1d bias shape mismatch");
	        const int64_t OL = out_shape[2];
	        const int stride = op.attrs.contains("stride") ? (int)resolve_const_value(op.attrs["stride"], bindings) : 1;
	        const int padding = op.attrs.contains("padding") ? (int)resolve_const_value(op.attrs["padding"], bindings) : 0;
	        const int dilation = op.attrs.contains("dilation") ? (int)resolve_const_value(op.attrs["dilation"], bindings) : 1;
	        const int groups = op.attrs.contains("groups") ? (int)resolve_const_value(op.attrs["groups"], bindings) : 1;
	        if (groups <= 0 || stride <= 0 || dilation <= 0) fail("conv1d attrs stride/dilation/groups must be positive");
	        if (C_IN != C_PER_G * groups) fail("conv1d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv1d C_OUT must be divisible by groups");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int ol = 0; ol < " + std::to_string(OL) + "; ++ol) {");
	        w.indent();
	        w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int k = 0; k < " + std::to_string(K) + "; ++k) {");
	        w.indent();
	        w.line("const int li = ol * " + std::to_string(stride) + " - " + std::to_string(padding) + " + k * " + std::to_string(dilation) + ";");
	        w.line("if (li >= 0 && li < " + std::to_string(L) + ") {");
	        w.indent();
	        w.line("const int x_idx = ((n * " + std::to_string(C_IN) + " + c) * " + std::to_string(L) + " + li);");
	        w.line("const int w_idx = ((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(K) + " + k);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OL) + " + ol)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv2d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv2d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4)
	          fail("conv2d expects input[N,C,H,W], weight[CO,C_PER_G,KH,KW], out[N,CO,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv2d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv2d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv2d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN_TOTAL = x_shape[1];
	        const int64_t H = x_shape[2];
	        const int64_t W = x_shape[3];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t KH = w_shape[2];
	        const int64_t KW = w_shape[3];
	        if (has_bias && b_shape[0] != C_OUT) fail("conv2d bias shape mismatch");
	        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
	          if (value.is_array()) {
	            if (value.size() != 2) fail("conv2d " + key + " must have length 2");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v};
	        };
	        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int, int>{1, 1};
	        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
	        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
	        const int groups = op.attrs.contains("groups") ? static_cast<int>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
	        if (groups <= 0 || stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
	          fail("conv2d attrs stride/dilation/groups must be positive");
	        if (C_IN_TOTAL != C_PER_G * groups) fail("conv2d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv2d C_OUT must be divisible by groups");
	        const int64_t OH = out_shape[2];
	        const int64_t OW = out_shape[3];
	        const int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
	        if (expected_oh != OH || expected_ow != OW) fail("conv2d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride.first) + " - " + std::to_string(padding.first) + " + kh * " + std::to_string(dilation.first) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride.second) + " - " + std::to_string(padding.second) + " + kw * " + std::to_string(dilation.second) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = (((n * " + std::to_string(C_IN_TOTAL) + " + c) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = (((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv3d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv3d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 5 || w_shape.size() != 5 || out_shape.size() != 5)
	          fail("conv3d expects input[N,C,D,H,W], weight[CO,C_PER_G,KD,KH,KW], out[N,CO,OD,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv3d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv3d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv3d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN_TOTAL = x_shape[1];
	        const int64_t D = x_shape[2];
	        const int64_t H = x_shape[3];
	        const int64_t W = x_shape[4];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t C_PER_G = w_shape[1];
	        const int64_t KD = w_shape[2];
	        const int64_t KH = w_shape[3];
	        const int64_t KW = w_shape[4];
	        if (has_bias && b_shape[0] != C_OUT) fail("conv3d bias shape mismatch");
	        auto parse_triple = [&](const json& value, const std::string& key) -> std::array<int, 3> {
	          if (value.is_array()) {
	            if (value.size() != 3) fail("conv3d " + key + " must have length 3");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)),
	                    static_cast<int>(resolve_const_value(value[1], bindings)),
	                    static_cast<int>(resolve_const_value(value[2], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v, v};
	        };
	        std::array<int, 3> stride = op.attrs.contains("stride") ? parse_triple(op.attrs["stride"], "stride") : std::array<int, 3>{1, 1, 1};
	        std::array<int, 3> padding = op.attrs.contains("padding") ? parse_triple(op.attrs["padding"], "padding") : std::array<int, 3>{0, 0, 0};
	        std::array<int, 3> dilation = op.attrs.contains("dilation") ? parse_triple(op.attrs["dilation"], "dilation") : std::array<int, 3>{1, 1, 1};
	        const int groups = op.attrs.contains("groups") ? static_cast<int>(resolve_const_value(op.attrs["groups"], bindings)) : 1;
	        if (groups <= 0 || stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0 || dilation[0] <= 0 || dilation[1] <= 0 || dilation[2] <= 0)
	          fail("conv3d attrs stride/dilation/groups must be positive");
	        if (C_IN_TOTAL != C_PER_G * groups) fail("conv3d channel/group mismatch");
	        if ((C_OUT % groups) != 0) fail("conv3d C_OUT must be divisible by groups");
	        const int64_t OD = out_shape[2];
	        const int64_t OH = out_shape[3];
	        const int64_t OW = out_shape[4];
	        const int64_t expected_od = ((D + 2LL * padding[0] - (int64_t)dilation[0] * (KD - 1) - 1) / stride[0]) + 1;
	        const int64_t expected_oh = ((H + 2LL * padding[1] - (int64_t)dilation[1] * (KH - 1) - 1) / stride[1]) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding[2] - (int64_t)dilation[2] * (KW - 1) - 1) / stride[2]) + 1;
	        if (expected_od != OD || expected_oh != OH || expected_ow != OW) fail("conv3d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int co_per_g = " + std::to_string(C_OUT / groups) + ";");
	        w.line("const int g = co / co_per_g;");
	        w.line("const int c_start = g * " + std::to_string(C_PER_G) + ";");
	        w.line("for (int od = 0; od < " + std::to_string(OD) + "; ++od) {");
	        w.indent();
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int ci = 0; ci < " + std::to_string(C_PER_G) + "; ++ci) {");
	        w.indent();
	        w.line("const int c = c_start + ci;");
	        w.line("for (int kd = 0; kd < " + std::to_string(KD) + "; ++kd) {");
	        w.indent();
	        w.line("const int id = od * " + std::to_string(stride[0]) + " - " + std::to_string(padding[0]) + " + kd * " + std::to_string(dilation[0]) + ";");
	        w.line("if ((unsigned)id >= (unsigned)" + std::to_string(D) + ") continue;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride[1]) + " - " + std::to_string(padding[1]) + " + kh * " + std::to_string(dilation[1]) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride[2]) + " - " + std::to_string(padding[2]) + " + kw * " + std::to_string(dilation[2]) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = ((((n * " + std::to_string(C_IN_TOTAL) + " + c) * " + std::to_string(D) + " + id) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = ((((co * " + std::to_string(C_PER_G) + " + ci) * " + std::to_string(KD) + " + kd) * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[((((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OD) + " + od) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "conv_depthwise2d") {
	        if (op.inputs.size() < 2 || op.inputs.size() > 3) fail("conv_depthwise2d expects inputs [input, weight] or [input, weight, bias]");
	        const auto& x_shape = shape_env.at(op.inputs[0]);
	        const auto& w_shape = shape_env.at(op.inputs[1]);
	        const bool has_bias = op.inputs.size() == 3;
	        const auto& b_shape = has_bias ? shape_env.at(op.inputs[2]) : std::vector<int64_t>{};
	        if (x_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4)
	          fail("conv_depthwise2d expects input[N,C,H,W], weight[C_OUT,1,KH,KW], out[N,C_OUT,OH,OW]");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32")
	          fail("conv_depthwise2d currently supports f32 only");
	        if (has_bias) {
	          if (b_shape.size() != 1) fail("conv_depthwise2d bias must be rank-1");
	          if (dtype_env.at(op.inputs[2]) != "f32") fail("conv_depthwise2d bias must be f32");
	        }
	        const int64_t N = x_shape[0];
	        const int64_t C_IN = x_shape[1];
	        const int64_t H = x_shape[2];
	        const int64_t W = x_shape[3];
	        const int64_t C_OUT = w_shape[0];
	        const int64_t KH = w_shape[2];
	        const int64_t KW = w_shape[3];
	        if (w_shape[1] != 1) fail("conv_depthwise2d expects weight second dim == 1");
	        if (C_IN <= 0 || C_OUT <= 0 || (C_OUT % C_IN) != 0) fail("conv_depthwise2d channel multiplier mismatch");
	        if (has_bias && b_shape[0] != C_OUT) fail("conv_depthwise2d bias shape mismatch");
	        const int64_t channel_multiplier = C_OUT / C_IN;
	        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
	          if (value.is_array()) {
	            if (value.size() != 2) fail("conv_depthwise2d " + key + " must have length 2");
	            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
	          }
	          int v = static_cast<int>(resolve_const_value(value, bindings));
	          return {v, v};
	        };
	        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : std::pair<int, int>{1, 1};
	        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
	        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
	        if (stride.first <= 0 || stride.second <= 0 || dilation.first <= 0 || dilation.second <= 0)
	          fail("conv_depthwise2d attrs stride/dilation must be positive");
	        const int64_t OH = out_shape[2];
	        const int64_t OW = out_shape[3];
	        const int64_t expected_oh = ((H + 2LL * padding.first - (int64_t)dilation.first * (KH - 1) - 1) / stride.first) + 1;
	        const int64_t expected_ow = ((W + 2LL * padding.second - (int64_t)dilation.second * (KW - 1) - 1) / stride.second) + 1;
	        if (expected_oh != OH || expected_ow != OW) fail("conv_depthwise2d output shape mismatch");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int co = 0; co < " + std::to_string(C_OUT) + "; ++co) {");
	        w.indent();
	        w.line("const int ci = co / " + std::to_string(channel_multiplier) + ";");
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        if (has_bias) w.line("float acc = " + v(op.inputs[2]) + "[co];");
	        else w.line("float acc = 0.0f;");
	        w.line("for (int kh = 0; kh < " + std::to_string(KH) + "; ++kh) {");
	        w.indent();
	        w.line("const int ih = oh * " + std::to_string(stride.first) + " - " + std::to_string(padding.first) + " + kh * " + std::to_string(dilation.first) + ";");
	        w.line("if ((unsigned)ih >= (unsigned)" + std::to_string(H) + ") continue;");
	        w.line("for (int kw = 0; kw < " + std::to_string(KW) + "; ++kw) {");
	        w.indent();
	        w.line("const int iw = ow * " + std::to_string(stride.second) + " - " + std::to_string(padding.second) + " + kw * " + std::to_string(dilation.second) + ";");
	        w.line("if ((unsigned)iw >= (unsigned)" + std::to_string(W) + ") continue;");
	        w.line("const int x_idx = (((n * " + std::to_string(C_IN) + " + ci) * " + std::to_string(H) + " + ih) * " + std::to_string(W) + " + iw);");
	        w.line("const int w_idx = ((co * " + std::to_string(KH) + " + kh) * " + std::to_string(KW) + " + kw);");
	        w.line("acc += " + v(op.inputs[0]) + "[x_idx] * " + v(op.inputs[1]) + "[w_idx];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(((n * " + std::to_string(C_OUT) + " + co) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = acc;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "dropout") {
	        if (op.inputs.size() != 3) fail("dropout requires 3 inputs (X, p, seed)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("dropout supports only f32 tensors");
	        if (!shape_env.at(op.inputs[1]).empty()) fail("dropout p must be a scalar tensor (rank-0)");
	        if (!shape_env.at(op.inputs[2]).empty()) fail("dropout seed must be a scalar tensor (rank-0)");
	        const std::string seed_dt = dtype_env.at(op.inputs[2]);
	        std::string seed_expr;
	        if (seed_dt == "i32") seed_expr = "(uint64_t)(uint32_t)" + v(op.inputs[2]) + "[0]";
	        else if (seed_dt == "i64") seed_expr = "(uint64_t)" + v(op.inputs[2]) + "[0]";
	        else fail("dropout seed dtype must be i32 or i64");
	        int rounds = 10;
	        if (op.attrs.contains("n_rounds")) rounds = (int)resolve_const_value(op.attrs["n_rounds"], bindings);
	        int64_t n = numel(out_shape);
	        w.line("intentir_dropout_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ", " + v(op.inputs[1]) +
	               "[0], " + seed_expr + ", " + std::to_string(rounds) + ");");
	      } else if (op.op == "correlation") {
	        if (op.inputs.size() != 3) fail("correlation requires 3 inputs (src0, src1, out_shift)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(op.inputs[1]) != "i8" || dtype_env.at(out) != "i8") fail("correlation supports only i8 tensors");
	        const auto& s0 = shape_env.at(op.inputs[0]);
	        const auto& s1 = shape_env.at(op.inputs[1]);
	        if (s0.size() != 3 || s1.size() != 3 || out_shape.size() != 3) fail("correlation expects rank-3 tensors");
	        if (s0 != s1) fail("correlation expects src0/src1 shapes to match");
	        if (s0[1] != out_shape[1] || s0[2] != out_shape[2]) fail("correlation spatial shape mismatch");
	        if (!shape_env.at(op.inputs[2]).empty()) fail("correlation out_shift must be a scalar tensor (rank-0)");
	        const std::string sh_dt = dtype_env.at(op.inputs[2]);
	        if (!(sh_dt == "i32" || sh_dt == "i64")) fail("correlation out_shift dtype must be i32 or i64");
	        const int64_t OC = out_shape[0];
	        const int64_t IC = s0[0];
	        const int64_t H = out_shape[1];
	        const int64_t W = out_shape[2];
	        w.line("intentir_correlation_i8(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + out_var + ", " + std::to_string(OC) +
	               ", " + std::to_string(IC) + ", " + std::to_string(H) + ", " + std::to_string(W) + ", (int32_t)" + v(op.inputs[2]) + "[0]);");
	      } else if (op.op == "resize") {
	        if (op.inputs.size() != 1) fail("resize requires 1 input (src)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(out) != "i8") fail("resize supports only i8 tensors");
	        const auto& s = shape_env.at(op.inputs[0]);
	        if (s.size() != 3 || out_shape.size() != 3) fail("resize expects rank-3 tensors [C,H,W]");
	        if (out_shape[0] != s[0]) fail("resize channel mismatch");
	        if (out_shape[1] != 2 * s[1] || out_shape[2] != 2 * s[2]) fail("resize currently supports only 2x upsample");
	        int hw_fl = 7;
	        if (op.attrs.contains("hw_fl")) hw_fl = (int)resolve_const_value(op.attrs["hw_fl"], bindings);
	        w.line("intentir_resize_bilinear2x_i8(" + v(op.inputs[0]) + ", " + out_var + ", " + std::to_string(s[0]) + ", " + std::to_string(s[1]) + ", " +
	               std::to_string(s[2]) + ", " + std::to_string(hw_fl) + ");");
	      } else if (op.op == "upsample_nearest1d") {
	        if (op.inputs.size() != 1) fail("upsample_nearest1d requires 1 input");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("upsample_nearest1d supports only f32 tensors");
	        const auto& s = shape_env.at(op.inputs[0]);
	        if (s.size() != 3 || out_shape.size() != 3) fail("upsample_nearest1d expects input/output rank-3 [N,C,L]");
	        if (s[0] != out_shape[0] || s[1] != out_shape[1]) fail("upsample_nearest1d N/C mismatch");
	        const int64_t N = s[0], C = s[1], IL = s[2], OL = out_shape[2];
	        if (IL <= 0 || OL <= 0) fail("upsample_nearest1d expects positive IL/OL");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int c = 0; c < " + std::to_string(C) + "; ++c) {");
	        w.indent();
	        w.line("for (int ol = 0; ol < " + std::to_string(OL) + "; ++ol) {");
	        w.indent();
	        w.line("int il = (int)(((int64_t)ol * " + std::to_string(IL) + ") / " + std::to_string(OL) + ");");
	        w.line("if (il >= " + std::to_string(IL) + ") il = " + std::to_string(IL - 1) + ";");
	        w.line(out_var + "[((n * " + std::to_string(C) + " + c) * " + std::to_string(OL) + " + ol)] = " + v(op.inputs[0]) +
	               "[((n * " + std::to_string(C) + " + c) * " + std::to_string(IL) + " + il)];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "upsample_nearest2d") {
	        if (op.inputs.size() != 1) fail("upsample_nearest2d requires 1 input");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(out) != "f32") fail("upsample_nearest2d supports only f32 tensors");
	        const auto& s = shape_env.at(op.inputs[0]);
	        if (s.size() != 4 || out_shape.size() != 4) fail("upsample_nearest2d expects input/output rank-4 [N,C,H,W]");
	        if (s[0] != out_shape[0] || s[1] != out_shape[1]) fail("upsample_nearest2d N/C mismatch");
	        const int64_t N = s[0], C = s[1], IH = s[2], IW = s[3], OH = out_shape[2], OW = out_shape[3];
	        if (IH <= 0 || IW <= 0 || OH <= 0 || OW <= 0) fail("upsample_nearest2d expects positive H/W");
	        w.line("for (int n = 0; n < " + std::to_string(N) + "; ++n) {");
	        w.indent();
	        w.line("for (int c = 0; c < " + std::to_string(C) + "; ++c) {");
	        w.indent();
	        w.line("for (int oh = 0; oh < " + std::to_string(OH) + "; ++oh) {");
	        w.indent();
	        w.line("int ih = (int)(((int64_t)oh * " + std::to_string(IH) + ") / " + std::to_string(OH) + ");");
	        w.line("if (ih >= " + std::to_string(IH) + ") ih = " + std::to_string(IH - 1) + ";");
	        w.line("for (int ow = 0; ow < " + std::to_string(OW) + "; ++ow) {");
	        w.indent();
	        w.line("int iw = (int)(((int64_t)ow * " + std::to_string(IW) + ") / " + std::to_string(OW) + ");");
	        w.line("if (iw >= " + std::to_string(IW) + ") iw = " + std::to_string(IW - 1) + ";");
	        w.line(out_var + "[(((n * " + std::to_string(C) + " + c) * " + std::to_string(OH) + " + oh) * " + std::to_string(OW) + " + ow)] = " +
	               v(op.inputs[0]) + "[(((n * " + std::to_string(C) + " + c) * " + std::to_string(IH) + " + ih) * " + std::to_string(IW) + " + iw)];");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "warp") {
	        if (op.inputs.size() != 2) fail("warp requires 2 inputs (src, offset)");
	        if (dtype_env.at(op.inputs[0]) != "i8" || dtype_env.at(op.inputs[1]) != "i16" || dtype_env.at(out) != "i8") fail("warp expects src/out i8 and offset i16");
	        const auto& s = shape_env.at(op.inputs[0]);
	        const auto& off = shape_env.at(op.inputs[1]);
	        if (s.size() != 3 || off.size() != 2 || out_shape.size() != 3) fail("warp expects src/out rank-3 and offset rank-2");
	        if (out_shape != s) fail("warp output shape must match src shape");
	        if (off[0] != s[1] || off[1] != s[2]) fail("warp offset shape must match [H,W]");
	        w.line("intentir_warp_q8_8_i8_i16(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + out_var + ", " + std::to_string(s[0]) + ", " +
	               std::to_string(s[1]) + ", " + std::to_string(s[2]) + ");");
	      } else if (op.op == "rope") {
	        if (op.inputs.size() != 3) fail("rope requires 3 inputs (input, cos, sin)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(op.inputs[2]) != "f32" || dtype_env.at(out) != "f32")
	          fail("rope supports only f32 tensors");
	        const auto& x = shape_env.at(op.inputs[0]);
	        const auto& c = shape_env.at(op.inputs[1]);
	        const auto& s = shape_env.at(op.inputs[2]);
	        if (x.size() != 4 || out_shape.size() != 4) fail("rope expects rank-4 input/output");
	        if (c.size() != 2 || s.size() != 2) fail("rope expects rank-2 cos/sin");
	        if (out_shape != x) fail("rope output shape must match input shape");
	        const int64_t SEQ = x[0], B = x[1], H = x[2], D = x[3];
	        if ((D & 1) != 0) fail("rope expects even HEAD_DIM");
	        const int64_t half = D / 2;
	        if (c[0] != SEQ || c[1] != half) fail("rope cos shape must be [SEQ_LEN, HEAD_DIM/2]");
	        if (s[0] != SEQ || s[1] != half) fail("rope sin shape must be [SEQ_LEN, HEAD_DIM/2]");
	        w.line("intentir_rope_f32(" + v(op.inputs[0]) + ", " + v(op.inputs[1]) + ", " + v(op.inputs[2]) + ", " + out_var + ", " + std::to_string(SEQ) +
	               ", " + std::to_string(B) + ", " + std::to_string(H) + ", " + std::to_string(D) + ");");
	      } else if (op.op == "iota") {
	        int axis = op.attrs.value("axis", 0);
	        emit_iota(w, out_var, out_shape, axis, dtype_env.at(out));
	      } else if (op.op == "gather") {
	        std::vector<std::string> idxs;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idxs.push_back(v(op.inputs[i]));
	        std::vector<std::vector<int64_t>> idx_shapes;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idx_shapes.push_back(shape_env.at(op.inputs[i]));
	        std::vector<std::string> idx_dtypes;
	        for (size_t i = 1; i < op.inputs.size(); ++i) idx_dtypes.push_back(dtype_env.at(op.inputs[i]));
	        emit_gather(w, out_var, v(op.inputs[0]), idxs, shape_env.at(op.inputs[0]), idx_shapes, out_shape, dtype_env.at(op.inputs[0]), idx_dtypes, dtype_env.at(out));
	      } else if (op.op == "index_add") {
	        if (op.inputs.size() != 3) fail("index_add expects inputs [base, index, src]");
	        int axis = op.attrs.value("axis", 0);
	        double alpha = 1.0;
	        if (op.attrs.contains("alpha")) alpha = resolve_const_value(op.attrs["alpha"], bindings);
	        emit_index_add_2d(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            out_shape,
	            dtype_env.at(op.inputs[1]),
	            axis,
	            alpha);
	      } else if (op.op == "index_put") {
	        if (op.inputs.size() != 4) fail("index_put expects inputs [base, row_idx, col_idx, values]");
	        bool accumulate = op.attrs.value("accumulate", false);
	        emit_index_put_2d(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            v(op.inputs[3]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            shape_env.at(op.inputs[3]),
	            out_shape,
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(op.inputs[2]),
	            accumulate);
	      } else if (op.op == "masked_select") {
	        if (op.inputs.size() != 2) fail("masked_select expects inputs [inp, mask]");
	        emit_masked_select_flat(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            out_shape,
	            dtype_env.at(op.inputs[0]),
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(out));
	      } else if (op.op == "masked_scatter") {
	        if (op.inputs.size() != 3) fail("masked_scatter expects inputs [inp, mask, source]");
	        emit_masked_scatter_flat(
	            w,
	            out_var,
	            v(op.inputs[0]),
	            v(op.inputs[1]),
	            v(op.inputs[2]),
	            shape_env.at(op.inputs[0]),
	            shape_env.at(op.inputs[1]),
	            shape_env.at(op.inputs[2]),
	            out_shape,
	            dtype_env.at(op.inputs[0]),
	            dtype_env.at(op.inputs[1]),
	            dtype_env.at(op.inputs[2]),
	            dtype_env.at(out));
	      } else if (op.op == "kron") {
	        if (op.inputs.size() != 2) fail("kron expects inputs [A, B]");
	        emit_kron_2d(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape);
	      } else if (op.op == "reduce_sum") {
        // Accept both "axes" and "dims" (parser/LLM may emit either). Prefer "axes"
        // when both are present to match interpreter semantics.
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_sum missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        std::optional<double> scale;
	        if (op.attrs.contains("scale")) scale = resolve_const_value(op.attrs["scale"], bindings);
	        emit_reduce_sum(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims, scale, dtype_env.at(out));
	      } else if (op.op == "reduce_prod") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_prod missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_prod(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_max") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_max missing dims/axes");
	        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_max(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_min") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_min missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_min(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims);
	      } else if (op.op == "reduce_any") {
        std::vector<int> dims;
        if (op.attrs.contains("axes")) {
          for (const auto& d : op.attrs["axes"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("dims")) {
          for (const auto& d : op.attrs["dims"]) dims.push_back(d.get<int>());
        } else if (op.attrs.contains("axis")) {
          if (op.attrs["axis"].is_array()) {
            for (const auto& d : op.attrs["axis"]) dims.push_back(d.get<int>());
          } else {
            dims.push_back(op.attrs["axis"].get<int>());
          }
        } else {
          fail("reduce_any missing dims/axes");
        }
	        bool keepdims = op.attrs.value("keepdims", false);
	        emit_reduce_any(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, dims, keepdims, dtype_env.at(op.inputs[0]), dtype_env.at(out));
	      } else if (op.op == "quantile") {
	        if (op.inputs.size() != 2) fail("quantile expects inputs [inp, q]");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        const auto& q_shape = shape_env.at(op.inputs[1]);
	        if (in_shape.size() != 2) fail("quantile currently supports rank-2 input");
	        if (!q_shape.empty()) fail("quantile q must be a scalar tensor (rank-0)");
	        if (dtype_env.at(op.inputs[0]) != "f32" || dtype_env.at(op.inputs[1]) != "f32" || dtype_env.at(out) != "f32") {
	          fail("quantile currently supports only f32 tensors");
	        }
	        int axis = op.attrs.value("dim", op.attrs.value("axis", 1));
	        if (axis < 0) axis += 2;
	        if (axis != 1) fail("quantile currently supports dim=1 only");
	        bool keepdim = op.attrs.value("keepdim", false);
	        std::string interpolation = op.attrs.value("interpolation", std::string("linear"));
	        if (interpolation != "linear") fail("quantile currently supports interpolation=linear only");
	        const int64_t M = in_shape[0];
	        const int64_t N = in_shape[1];
	        if (N <= 0) fail("quantile expects N > 0");
	        if (out_shape.size() == 1) {
	          if (out_shape[0] != M) fail("quantile output shape mismatch: expect [M]");
	        } else if (out_shape.size() == 2) {
	          if (!keepdim) fail("quantile rank-2 output requires keepdim=true");
	          if (out_shape[0] != M || out_shape[1] != 1) fail("quantile output shape mismatch: expect [M,1]");
	        } else {
	          fail("quantile currently supports output rank 1 or 2");
	        }
	        w.line("float __q = " + v(op.inputs[1]) + "[0];");
	        w.line("if (__q < 0.0f) __q = 0.0f;");
	        w.line("if (__q > 1.0f) __q = 1.0f;");
	        w.line("float* __quantile_row = (float*)malloc(sizeof(float) * (size_t)" + std::to_string(N) + ");");
	        w.line("if (!__quantile_row) { fprintf(stderr, \"alloc failed: quantile_row\\n\"); return; }");
	        w.line("for (int64_t i = 0; i < " + std::to_string(M) + "; ++i) {");
	        w.indent();
	        w.line("const float* __row = " + v(op.inputs[0]) + " + (size_t)i * (size_t)" + std::to_string(N) + ";");
	        w.line("for (int64_t j = 0; j < " + std::to_string(N) + "; ++j) __quantile_row[(size_t)j] = __row[(size_t)j];");
	        w.line("for (int64_t __ii = 1; __ii < " + std::to_string(N) + "; ++__ii) {");
	        w.indent();
	        w.line("float __v = __quantile_row[(size_t)__ii];");
	        w.line("int64_t __jj = __ii - 1;");
	        w.line("while (__jj >= 0 && __quantile_row[(size_t)__jj] > __v) {");
	        w.indent();
	        w.line("__quantile_row[(size_t)(__jj + 1)] = __quantile_row[(size_t)__jj];");
	        w.line("--__jj;");
	        w.dedent();
	        w.line("}");
	        w.line("__quantile_row[(size_t)(__jj + 1)] = __v;");
	        w.dedent();
	        w.line("}");
	        if (N == 1) {
	          w.line("const float __qv = __quantile_row[(size_t)0];");
	        } else {
	          w.line("const float __pos = __q * (float)" + std::to_string(N - 1) + ";");
	          w.line("int64_t __lo = (int64_t)__pos;");
	          w.line("if (__lo < 0) __lo = 0;");
	          w.line("if (__lo > " + std::to_string(N - 1) + ") __lo = " + std::to_string(N - 1) + ";");
	          w.line("int64_t __hi = (__lo + 1 <= " + std::to_string(N - 1) + ") ? (__lo + 1) : __lo;");
	          w.line("const float __t = __pos - (float)__lo;");
	          w.line("const float __qv = __quantile_row[(size_t)__lo] + (__quantile_row[(size_t)__hi] - __quantile_row[(size_t)__lo]) * __t;");
	        }
	        if (out_shape.size() == 1) {
	          w.line(out_var + "[(size_t)i] = __qv;");
	        } else {
	          w.line(out_var + "[(size_t)i * (size_t)" + std::to_string(out_shape[1]) + "] = __qv;");
	        }
	        w.dedent();
	        w.line("}");
	        w.line("free(__quantile_row);");
	      } else if (op.op == "argmax" || op.op == "argmin") {
        int axis = op.attrs.value("axis", -1);
        emit_arg_reduce_axis1(w, op.op, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis);
	      } else if (op.op == "avg_pool2d") {
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
          if (value.is_array()) {
            if (value.size() != 2) fail("avg_pool2d " + key + " must have length 2");
            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
          }
          int v = static_cast<int>(resolve_const_value(value, bindings));
          return {v, v};
        };
        std::pair<int, int> ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int, int>{2, 2};
        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
        bool count_include_pad = op.attrs.value("count_include_pad", true);
        emit_avg_pool2d(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, ksz.first, ksz.second, stride.first, stride.second,
                        padding.first, padding.second, count_include_pad);
	      } else if (op.op == "max_pool2d_with_indices") {
        auto parse_pair = [&](const json& value, const std::string& key) -> std::pair<int, int> {
          if (value.is_array()) {
            if (value.size() != 2) fail("max_pool2d_with_indices " + key + " must have length 2");
            return {static_cast<int>(resolve_const_value(value[0], bindings)), static_cast<int>(resolve_const_value(value[1], bindings))};
          }
          int v = static_cast<int>(resolve_const_value(value, bindings));
          return {v, v};
        };
        std::pair<int, int> ksz = op.attrs.contains("kernel_size") ? parse_pair(op.attrs["kernel_size"], "kernel_size") : std::pair<int, int>{2, 2};
        std::pair<int, int> stride = op.attrs.contains("stride") ? parse_pair(op.attrs["stride"], "stride") : ksz;
        std::pair<int, int> padding = op.attrs.contains("padding") ? parse_pair(op.attrs["padding"], "padding") : std::pair<int, int>{0, 0};
        std::pair<int, int> dilation = op.attrs.contains("dilation") ? parse_pair(op.attrs["dilation"], "dilation") : std::pair<int, int>{1, 1};
        bool ceil_mode = op.attrs.value("ceil_mode", false);
        if (ceil_mode) fail("max_pool2d_with_indices ceil_mode=true is not supported");
        if (dilation.first != 1 || dilation.second != 1) fail("max_pool2d_with_indices dilation != 1 is not supported");
        std::string select = op.attrs.value("select", "values");
        bool emit_indices = (select == "indices") || (dtype_env.at(out) == "i64");
        emit_max_pool2d_with_indices(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, ksz.first, ksz.second, stride.first, stride.second,
                                     padding.first, padding.second, emit_indices);
	      } else if (op.op == "mse_loss") {
        if (op.inputs.size() != 2) fail("mse_loss expects [input, target]");
        const auto& in_shape = shape_env.at(op.inputs[0]);
        const auto& tgt_shape = shape_env.at(op.inputs[1]);
        if (in_shape != tgt_shape) fail("mse_loss shape mismatch between input and target");
        std::string reduction;
        if (op.attrs.contains("reduction")) {
          if (op.attrs["reduction"].is_number_integer()) {
            int mode = op.attrs["reduction"].get<int>();
            if (mode == 0) reduction = "none";
            else if (mode == 2) reduction = "sum";
            else reduction = "mean";
          } else {
            reduction = op.attrs["reduction"].get<std::string>();
          }
        } else {
          reduction = "mean";
        }
        emit_mse_loss(w, out_var, v(op.inputs[0]), v(op.inputs[1]), in_shape, out_shape, reduction);
	      } else if (op.op == "scaled_dot_product_attention") {
        if (op.inputs.size() < 3) fail("scaled_dot_product_attention requires query/key/value");
        bool is_causal = op.attrs.value("is_causal", false);
        emit_scaled_dot_product_attention(w, out_var, v(op.inputs[0]), v(op.inputs[1]), v(op.inputs[2]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]),
                                          shape_env.at(op.inputs[2]), out_shape, is_causal);
	      } else if (op.op == "glu") {
          int axis = op.attrs.value("axis", -1);
          emit_glu(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis);
	      } else if (op.op == "exp") {
	        int64_t n = numel(out_shape);
          std::optional<double> base;
          if (op.attrs.contains("base")) {
            base = resolve_const_value(op.attrs["base"], bindings);
          }
          if (base.has_value()) {
            const double b = *base;
            if (std::fabs(b - 2.0) < 1e-6) {
              w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out_var + "[i] = exp2f(" + v(op.inputs[0]) + "[i]);");
            } else {
              w.line("for (size_t i = 0; i < (size_t)" + std::to_string(n) + "; ++i) " + out_var + "[i] = powf((float)" + std::to_string(b) + ", " +
                     v(op.inputs[0]) + "[i]);");
            }
          } else {
            w.line("intentir_exp_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ");");
          }
	      } else if (op.op == "log") {
	        emit_log(w, out_var, v(op.inputs[0]), out_shape);
	      } else if (op.op == "relu") {
	        int64_t n = numel(out_shape);
	        w.line("intentir_relu_f32(" + v(op.inputs[0]) + ", " + out_var + ", (size_t)" + std::to_string(n) + ");");
	      } else if (op.op == "softmax") {
	        int axis = op.attrs.value("axis", -1);
	        emit_softmax(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), axis);
	      } else if (op.op == "std") {
	        int axis = op.attrs.value("axis", 1);
	        bool keepdims = op.attrs.value("keepdims", false);
	        int correction = op.attrs.value("correction", op.attrs.value("ddof", 0));
	        emit_std_axis1(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis, keepdims, correction);
	      } else if (op.op == "sort") {
	        int axis = op.attrs.value("axis", -1);
	        bool descending = op.attrs.value("descending", false);
	        emit_sort_axis1(w, out_var, v(op.inputs[0]), shape_env.at(op.inputs[0]), out_shape, axis, descending);
	      } else if (op.op == "unique") {
	        if (op.inputs.size() != 1) fail("unique expects one input");
	        const auto& in_shape = shape_env.at(op.inputs[0]);
	        if (in_shape.size() != 1) fail("unique lowering currently supports rank-1 input");
	        if (out_shape.size() != 1) fail("unique lowering currently supports rank-1 output");
	        const std::string in_dt = dtype_env.at(op.inputs[0]);
	        const std::string out_dt = dtype_env.at(out);
	        if (in_dt != out_dt) fail("unique dtype mismatch between input and output");
	        if (in_dt != "i32") fail("unique lowering currently supports i32 tensors");
	        const int64_t N = in_shape[0];
	        const int64_t U = out_shape[0];
	        w.line("int64_t __unique_count = 0;");
	        w.line("for (int64_t i = 0; i < " + std::to_string(N) + "; ++i) {");
	        w.indent();
	        w.line("const int32_t __v = " + v(op.inputs[0]) + "[(size_t)i];");
	        w.line("int __seen = 0;");
	        w.line("for (int64_t j = 0; j < __unique_count; ++j) {");
	        w.indent();
	        w.line("if (" + out_var + "[(size_t)j] == __v) { __seen = 1; break; }");
	        w.dedent();
	        w.line("}");
	        w.line("if (__seen == 0) {");
	        w.indent();
	        w.line("if (__unique_count < " + std::to_string(U) + ") " + out_var + "[(size_t)__unique_count] = __v;");
	        w.line("++__unique_count;");
	        w.dedent();
	        w.line("}");
	        w.dedent();
	        w.line("}");
	        w.line("for (int64_t i = __unique_count; i < " + std::to_string(U) + "; ++i) " + out_var + "[(size_t)i] = 0;");
	        w.line("for (int64_t i = 1; i < " + std::to_string(U) + "; ++i) {");
	        w.indent();
	        w.line("int32_t __v = " + out_var + "[(size_t)i];");
	        w.line("int64_t __j = i - 1;");
	        w.line("while (__j >= 0 && " + out_var + "[(size_t)__j] > __v) {");
	        w.indent();
	        w.line(out_var + "[(size_t)(__j + 1)] = " + out_var + "[(size_t)__j];");
	        w.line("--__j;");
	        w.dedent();
	        w.line("}");
	        w.line(out_var + "[(size_t)(__j + 1)] = __v;");
	        w.dedent();
	        w.line("}");
	      } else if (op.op == "stack") {
	        if (op.inputs.size() != 2) fail("stack currently supports exactly 2 inputs");
	        int axis = op.attrs.value("axis", 0);
	        if (axis != 0) fail("stack lowering currently supports axis=0 only");
	        emit_stack_axis0_2x2d(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape);
	      } else if (op.op == "matmul") {
	        bool ta = op.attrs.value("transpose_a", false);
	        bool tb = op.attrs.value("transpose_b", false);
	        emit_matmul(w, out_var, v(op.inputs[0]), v(op.inputs[1]), shape_env.at(op.inputs[0]), shape_env.at(op.inputs[1]), out_shape, ta, tb,
	                    sched_tile_m, sched_tile_n, sched_tile_k);
	      } else {
	        fail("unsupported op lowering: " + op.op);
	      }
