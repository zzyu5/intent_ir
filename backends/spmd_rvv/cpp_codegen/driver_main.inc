using namespace intentir_rvv_codegen;
int main(int argc, char** argv) {
  try {
    std::string intent_path;
    std::string shapes_path;
    std::string mode = "verify";
    double atol = 1e-3;
    double rtol = 1e-3;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--intent" && i + 1 < argc) intent_path = argv[++i];
      else if (a == "--shapes" && i + 1 < argc) shapes_path = argv[++i];
      else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
      else if (a == "--atol" && i + 1 < argc) atol = std::stod(argv[++i]);
      else if (a == "--rtol" && i + 1 < argc) rtol = std::stod(argv[++i]);
      else if (a == "-h" || a == "--help") {
        std::cout << "usage: intentir_codegen --intent <intent.json> --shapes <shapes.json> [--mode verify|bench] [--atol x] [--rtol x]\\n";
        return 0;
      } else {
        fail("unknown arg: " + a);
      }
    }
    if (intent_path.empty() || shapes_path.empty()) {
      std::cerr << "missing --intent/--shapes (use --help)\\n";
      return 2;
    }

    Intent intent = parse_intent(read_json_file(intent_path));
    json shapes_json = read_json_file(shapes_path);
    if (!shapes_json.is_object()) fail("shapes must be object");
    std::unordered_map<std::string, int64_t> bindings;
    for (auto it = shapes_json.begin(); it != shapes_json.end(); ++it) {
      if (it.value().is_number_integer()) bindings.emplace(it.key(), it.value().get<int64_t>());
      else if (it.value().is_number()) bindings.emplace(it.key(), static_cast<int64_t>(it.value().get<double>()));
    }
    auto binding_get = [&](const std::string& key) -> std::optional<int64_t> {
      auto it = bindings.find(key);
      if (it == bindings.end()) return std::nullopt;
      return it->second;
    };
    auto derive_binding = [&](const std::string& key) -> std::optional<int64_t> {
      auto c_in = binding_get("C_IN");
      auto groups = binding_get("GROUPS");
      auto c_per_g = binding_get("C_PER_G");
      auto c_out = binding_get("C_OUT");
      auto mult = binding_get("MULT");
      if (key == "C_IN_TOTAL") {
        if (c_in && groups) return (*c_in) * (*groups);
        if (c_in) return *c_in;
      }
      if (key == "C_PER_G") {
        if (c_per_g) return *c_per_g;
        if (c_in && groups && *groups > 0 && ((*c_in) % (*groups) == 0)) return (*c_in) / (*groups);
        if (c_in) return *c_in;
      }
      if (key == "C_OUT") {
        if (c_out) return *c_out;
        if (c_in && mult) return (*c_in) * (*mult);
        if (c_per_g && groups) return (*c_per_g) * (*groups);
      }
      if (key == "OH") {
        auto H = binding_get("H");
        auto PH = binding_get("PH");
        auto KH = binding_get("KH");
        auto SH = binding_get("SH");
        auto DH = binding_get("DH");
        if (H && PH && KH && SH && DH && *SH > 0) return ((*H) + 2 * (*PH) - (*DH) * ((*KH) - 1) - 1) / (*SH) + 1;
      }
      if (key == "OW") {
        auto W = binding_get("W");
        auto PW = binding_get("PW");
        auto KW = binding_get("KW");
        auto SW = binding_get("SW");
        auto DW = binding_get("DW");
        if (W && PW && KW && SW && DW && *SW > 0) return ((*W) + 2 * (*PW) - (*DW) * ((*KW) - 1) - 1) / (*SW) + 1;
      }
      if (key == "OD") {
        auto D = binding_get("D");
        auto PD = binding_get("PD");
        auto KD = binding_get("KD");
        auto SD = binding_get("SD");
        auto DD = binding_get("DD");
        if (D && PD && KD && SD && DD && *SD > 0) return ((*D) + 2 * (*PD) - (*DD) * ((*KD) - 1) - 1) / (*SD) + 1;
      }
      return std::nullopt;
    };

    // Backend schedule knobs (C): optional tunables/hints.
    auto resolve_sched_int = [&](const char* key) -> std::optional<int64_t> {
      if (!intent.schedule.is_object()) return std::nullopt;
      auto it = intent.schedule.find(key);
      if (it == intent.schedule.end()) return std::nullopt;
      return resolve_dim_token(*it, bindings);
    };
    std::optional<int64_t> sched_tile_m = resolve_sched_int("tile_m");
    std::optional<int64_t> sched_tile_n = resolve_sched_int("tile_n");
    std::optional<int64_t> sched_tile_k = resolve_sched_int("tile_k");
    std::optional<int64_t> sched_vec_width = resolve_sched_int("vec_width");

    // Determine external inputs.
    std::unordered_map<std::string, bool> produced;
    std::unordered_map<std::string, bool> used;
    for (const auto& op : intent.ops) {
      produced[op.output] = true;
      for (const auto& x : op.inputs) used[x] = true;
    }
    std::vector<std::string> external_inputs;
    for (const auto& kv : used) {
      const std::string& name = kv.first;
      if (intent.tensors.find(name) != intent.tensors.end() && produced.find(name) == produced.end()) external_inputs.push_back(name);
    }
    std::sort(external_inputs.begin(), external_inputs.end());

    // Resolve declared tensor shapes.
    std::unordered_map<std::string, std::vector<int64_t>> shape_env;
    std::unordered_map<std::string, std::string> dtype_env;
    for (const auto& kv : intent.tensors) {
      const std::string& name = kv.first;
      const Tensor& t = kv.second;
      std::vector<int64_t> shp;
      for (const auto& d : t.shape) {
        if (d.is_number_integer()) shp.push_back(d.get<int64_t>());
        else if (d.is_string()) {
          std::string s = d.get<std::string>();
          if (!s.empty() && std::all_of(s.begin(), s.end(), ::isdigit)) {
            shp.push_back(std::stoll(s));
          } else {
            auto it = bindings.find(s);
            if (it == bindings.end()) {
              auto dv = derive_binding(s);
              if (!dv.has_value()) fail("unbound symbol in shape: " + s);
              bindings.emplace(s, *dv);
              shp.push_back(*dv);
            } else {
              shp.push_back(it->second);
            }
          }
        } else {
          fail("invalid dim type in tensor shape");
        }
      }
      shape_env[name] = shp;
      dtype_env[name] = t.dtype;
    }

    // Infer missing intermediates.
    for (const auto& op : intent.ops) {
#include "infer_shapes_dispatch.inc"
    }

    // Collect consts.
    std::unordered_map<std::string, ConstVal> const_vals;
    for (const auto& op : intent.ops) {
      if (op.op == "const") {
        const json v = op.attrs.value("value", json());
        const std::string dt = op.attrs.value("dtype", dtype_env.count(op.output) ? dtype_env[op.output] : std::string("f32"));
        const_vals[op.output] = {dt, resolve_const_value(v, bindings)};
      }
    }

    // Best-effort FLOPs accounting (for benchmarking / cost-model experiments).
    // We only count matmul FLOPs for now (GEMM model validation).
    double matmul_flops_total = 0.0;
    for (const auto& op : intent.ops) {
      if (op.op != "matmul") continue;
      if (op.inputs.size() < 2) continue;
      const auto& sa = shape_env[op.inputs[0]];
      const auto& sb = shape_env[op.inputs[1]];
      bool ta = op.attrs.value("transpose_a", false);
      bool tb = op.attrs.value("transpose_b", false);
      if (sa.size() == 2 && sb.size() == 2) {
        int64_t M = ta ? sa[1] : sa[0];
        int64_t K = ta ? sa[0] : sa[1];
        int64_t K2 = tb ? sb[1] : sb[0];
        int64_t N = tb ? sb[0] : sb[1];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)M * (double)N * (double)K;
        continue;
      }
      if (sa.size() == 2 && sb.size() == 1) {
        if (tb) continue;
        int64_t M = ta ? sa[1] : sa[0];
        int64_t K = ta ? sa[0] : sa[1];
        int64_t K2 = sb[0];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)M * (double)K;
        continue;
      }
      if (sa.size() == 3 && sb.size() == 3) {
        int64_t B = sa[0];
        int64_t M = ta ? sa[2] : sa[1];
        int64_t K = ta ? sa[1] : sa[2];
        int64_t K2 = tb ? sb[2] : sb[1];
        int64_t N = tb ? sb[1] : sb[2];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)B * (double)M * (double)N * (double)K;
        continue;
      }
      if (sa.size() == 4 && sb.size() == 4) {
        int64_t B = sa[0];
        int64_t H = sa[1];
        int64_t M = ta ? sa[3] : sa[2];
        int64_t K = ta ? sa[2] : sa[3];
        int64_t K2 = tb ? sb[3] : sb[2];
        int64_t N = tb ? sb[2] : sb[3];
        if (K2 != K) continue;
        matmul_flops_total += 2.0 * (double)B * (double)H * (double)M * (double)N * (double)K;
        continue;
      }
    }

    // ---- emit C program ----
    std::ostringstream c_src;
    CodeWriter cw(c_src);
    CProgramEmitter emitter{cw, intent, bindings, external_inputs, shape_env, dtype_env, const_vals, {},
                            sched_tile_m, sched_tile_n, sched_tile_k, sched_vec_width,
                            atol, rtol, matmul_flops_total, (mode == "bench")};
    emitter.emit_program();
    std::cout << c_src.str();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "intentir_codegen error: " << e.what() << "\n";
    return 1;
  }
}
