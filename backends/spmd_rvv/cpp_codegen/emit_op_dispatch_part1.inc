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
