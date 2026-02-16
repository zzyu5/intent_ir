#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "code_writer.h"
#include "common_utils.h"
#include "ir_model.h"
#include "shape_eval.h"

#ifdef INTENTIR_CUDA_CODEGEN_PYBIND
#include <pybind11/pybind11.h>
#endif

using json = nlohmann::json;

namespace {
using namespace intentir_cuda_codegen;

void emit_selected_api(CodeWriter& w) {
  // Selection introspection (for paper evidence/debug): the generated module can
  // expose `selected_variant()` / `selected_tag()` to Python.
  //
  // Host-dispatch kernels update these; single-kernel paths keep the default.
  w.line("static int intentir_cuda_selected_variant_idx = -1;");
  w.line("static const char* intentir_cuda_selected_variant_tag = \"unset\";");
  w.line("static float intentir_cuda_dispatch_total_ms = 0.0f;");
  w.line("static int intentir_cuda_dispatch_evals = 0;");
  w.line("static int intentir_cuda_fastpath_enabled = 0;");
  w.line("extern \"C\" int intentir_cuda_selected_variant() { return intentir_cuda_selected_variant_idx; }");
  w.line("extern \"C\" const char* intentir_cuda_selected_tag() { return intentir_cuda_selected_variant_tag; }");
  w.line("extern \"C\" float intentir_cuda_get_dispatch_total_ms() { return intentir_cuda_dispatch_total_ms; }");
  w.line("extern \"C\" int intentir_cuda_get_dispatch_evals() { return intentir_cuda_dispatch_evals; }");
  w.line("extern \"C\" int intentir_cuda_get_fastpath_enabled() { return intentir_cuda_fastpath_enabled; }");
  w.blank();
}

void emit_const_introspection_api(CodeWriter& w, int variant_count, bool has_evidence, int contract_level, bool specialize_dims) {
  if (variant_count < 1) variant_count = 1;
  w.line("extern \"C\" int intentir_cuda_variant_count() { return " + std::to_string(variant_count) + "; }");
  w.line("extern \"C\" int intentir_cuda_has_evidence() { return " + std::string(has_evidence ? "1" : "0") + "; }");
  w.line("extern \"C\" int intentir_cuda_contract_level() { return " + std::to_string(contract_level) + "; }");
  w.line("extern \"C\" int intentir_cuda_specialize_dims() { return " + std::string(specialize_dims ? "1" : "0") + "; }");
  w.blank();
}

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

json emit_addmv_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 4) fail("addmv lowering expects 4 ops (matmul,mul,mul,add)");
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

  const std::string A = mm.inputs[0];
  const std::string B = mm.inputs[1];
  const std::string mv_out = mm.output;

  auto A_it = intent.tensors.find(A);
  auto B_it = intent.tensors.find(B);
  if (A_it == intent.tensors.end() || B_it == intent.tensors.end()) fail("addmv lowering missing A/B tensors");
  if (A_it->second.shape.size() != 2 || B_it->second.shape.size() != 1) fail("addmv lowering expects A[2D], B[1D]");
  if (A_it->second.dtype != "f32" || B_it->second.dtype != "f32") fail("addmv lowering currently supports f32 tensors only");

  const std::string Out = add.output;
  auto Out_it = intent.tensors.find(Out);
  if (Out_it == intent.tensors.end()) fail("addmv lowering missing output tensor");
  if (Out_it->second.dtype != "f32" || Out_it->second.shape.size() != 1) fail("addmv lowering expects Out as f32 rank-1");

  // mul(mv_out, alpha) -> scaled_mv
  std::string alpha_name;
  if (mul_mv.inputs[0] == mv_out)
    alpha_name = mul_mv.inputs[1];
  else if (mul_mv.inputs[1] == mv_out)
    alpha_name = mul_mv.inputs[0];
  else
    fail("addmv lowering expects one mul input from matmul output");
  auto alpha_it = intent.tensors.find(alpha_name);
  if (alpha_it == intent.tensors.end()) fail("addmv lowering missing alpha tensor");
  if (alpha_it->second.dtype != "f32" || !alpha_it->second.shape.empty()) fail("addmv lowering expects alpha scalar f32 tensor");

  // mul(inp, beta) -> scaled_inp
  std::string Inp;
  std::string beta_name;
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
  auto Inp_it = intent.tensors.find(Inp);
  if (Inp_it == intent.tensors.end()) fail("addmv lowering missing input vector tensor");
  if (Inp_it->second.dtype != "f32" || Inp_it->second.shape.size() != 1) fail("addmv lowering expects input vector f32 rank-1");

  const std::string scaled_mv = mul_mv.output;
  const std::string scaled_inp = mul_inp.output;
  const bool add_inputs_ok =
      ((add.inputs[0] == scaled_mv && add.inputs[1] == scaled_inp) || (add.inputs[0] == scaled_inp && add.inputs[1] == scaled_mv));
  if (!add_inputs_ok) fail("addmv lowering expects add(scaled_mv, scaled_inp)");

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

json emit_allclose2d_f32_bool(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 9) fail("allclose lowering expects 9-op pattern");
  static const std::vector<std::string> kOps = {"sub", "abs", "abs", "mul", "add", "le", "not", "reduce_any", "not"};
  for (size_t i = 0; i < kOps.size(); ++i) {
    if (intent.ops[i].op != kOps[i]) fail("allclose lowering pattern mismatch");
  }

  const std::string A = intent.ops[0].inputs[0];
  const std::string B = intent.ops[0].inputs[1];
  const std::string rtol = intent.ops[3].inputs[0] == "abs_b" ? intent.ops[3].inputs[1] : intent.ops[3].inputs[0];
  const std::string atol = intent.ops[4].inputs[0] == intent.ops[3].output ? intent.ops[4].inputs[1] : intent.ops[4].inputs[0];
  const std::string OUT = intent.ops.back().output;

  auto a_it = intent.tensors.find(A);
  auto b_it = intent.tensors.find(B);
  auto r_it = intent.tensors.find(rtol);
  auto t_it = intent.tensors.find(atol);
  auto o_it = intent.tensors.find(OUT);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || r_it == intent.tensors.end() || t_it == intent.tensors.end() ||
      o_it == intent.tensors.end()) {
    fail("allclose lowering missing tensors");
  }
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || r_it->second.dtype != "f32" || t_it->second.dtype != "f32") {
    fail("allclose lowering supports f32 A/B/rtol/atol");
  }
  if (o_it->second.dtype != "bool") fail("allclose output must be bool");
  auto a_shape = resolve_shape_required(a_it->second, bindings, "allclose.A");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "allclose.B");
  if (a_shape.size() != 2 || b_shape.size() != 2) fail("allclose expects rank-2 A/B");
  if (a_shape != b_shape) fail("allclose shape mismatch");
  const int64_t M = a_shape[0];
  const int64_t N = a_shape[1];
  if (M <= 0 || N <= 0) fail("allclose invalid shape");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + A + ",");
  w.line("const float* __restrict__ " + B + ",");
  w.line("const float* __restrict__ " + rtol + ",");
  w.line("const float* __restrict__ " + atol + ",");
  w.line("bool* __restrict__ " + OUT);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("const float rtol_v = " + rtol + " ? " + rtol + "[0] : 1e-5f;");
  w.line("const float atol_v = " + atol + " ? " + atol + "[0] : 1e-8f;");
  w.line("bool all_close = true;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) {");
  w.indent();
  w.line("const float a = " + A + "[i];");
  w.line("const float b = " + B + "[i];");
  w.line("const float tol = atol_v + rtol_v * fabsf(b);");
  w.line("if (fabsf(a - b) > tol) { all_close = false; break; }");
  w.dedent();
  w.line("}");
  w.line(OUT + "[0] = all_close;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {A, B, rtol, atol, OUT}, {}, {A, B, rtol, atol, OUT});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {OUT};
  out["bindings"] = out_bindings;
  return out;
}

json emit_batch_norm2d_training_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 25) fail("batch_norm lowering expects 25-op expanded pattern");
  const std::string input = "input";
  const std::string weight = "weight";
  const std::string bias = "bias";
  const std::string running_mean = "running_mean";
  const std::string running_var = "running_var";
  const std::string eps = "eps";
  const std::string momentum = "momentum";
  const std::string n_elements = "n_elements";
  const std::string n_minus_1 = "n_minus_1";
  const std::string output = "output_1";
  const std::string mean = "mean";
  const std::string inv_std = "inv_std";
  const std::string running_mean_out = "running_mean_out";
  const std::string running_var_out = "running_var_out";

  auto in_it = intent.tensors.find(input);
  auto w_it = intent.tensors.find(weight);
  auto b_it = intent.tensors.find(bias);
  auto rm_it = intent.tensors.find(running_mean);
  auto rv_it = intent.tensors.find(running_var);
  auto out_it = intent.tensors.find(output);
  auto mean_it = intent.tensors.find(mean);
  auto inv_it = intent.tensors.find(inv_std);
  auto rmo_it = intent.tensors.find(running_mean_out);
  auto rvo_it = intent.tensors.find(running_var_out);
  if (in_it == intent.tensors.end() || w_it == intent.tensors.end() || b_it == intent.tensors.end() || rm_it == intent.tensors.end() ||
      rv_it == intent.tensors.end() || out_it == intent.tensors.end() || mean_it == intent.tensors.end() || inv_it == intent.tensors.end() ||
      rmo_it == intent.tensors.end() || rvo_it == intent.tensors.end()) {
    fail("batch_norm lowering missing tensors");
  }
  for (const auto* t : {&in_it->second, &w_it->second, &b_it->second, &rm_it->second, &rv_it->second, &out_it->second, &mean_it->second, &inv_it->second,
                        &rmo_it->second, &rvo_it->second}) {
    if (t->dtype != "f32") fail("batch_norm lowering supports f32 only");
  }

  auto in_shape = resolve_shape_required(in_it->second, bindings, "batch_norm.input");
  if (in_shape.size() != 3) fail("batch_norm expects input rank-3 [N,C,HW]");
  const int64_t N = in_shape[0];
  const int64_t C = in_shape[1];
  const int64_t HW = in_shape[2];
  if (N <= 0 || C <= 0 || HW <= 0) fail("batch_norm invalid shape");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + input + ",");
  w.line("const float* __restrict__ " + weight + ",");
  w.line("const float* __restrict__ " + bias + ",");
  w.line("const float* __restrict__ " + running_mean + ",");
  w.line("const float* __restrict__ " + running_var + ",");
  w.line("const float* __restrict__ " + eps + ",");
  w.line("const float* __restrict__ " + momentum + ",");
  w.line("const float* __restrict__ " + n_elements + ",");
  w.line("const float* __restrict__ " + n_minus_1 + ",");
  w.line("float* __restrict__ " + output + ",");
  w.line("float* __restrict__ " + mean + ",");
  w.line("float* __restrict__ " + inv_std + ",");
  w.line("float* __restrict__ " + running_mean_out + ",");
  w.line("float* __restrict__ " + running_var_out);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("const float eps_v = " + eps + " ? " + eps + "[0] : 1e-5f;");
  w.line("const float momentum_v = " + momentum + " ? " + momentum + "[0] : 0.1f;");
  w.line("float n_elems = " + n_elements + " ? " + n_elements + "[0] : (float)(" + std::to_string(N) + " * " + std::to_string(HW) + ");");
  w.line("if (n_elems <= 0.0f) n_elems = (float)(" + std::to_string(N) + " * " + std::to_string(HW) + ");");
  w.line("float n_minus = " + n_minus_1 + " ? " + n_minus_1 + "[0] : (n_elems - 1.0f);");
  w.line("if (n_minus <= 0.0f) n_minus = 1.0f;");
  w.line("const float one_minus_momentum = 1.0f - momentum_v;");
  w.line("for (int64_t c = 0; c < " + std::to_string(C) + "LL; ++c) {");
  w.indent();
  w.line("float sum = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) sum += " + input + "[base + hw];");
  w.dedent();
  w.line("}");
  w.line("const float mean_c = sum / n_elems;");
  w.line(mean + "[c] = mean_c;");
  w.line("float var_sum = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) {");
  w.indent();
  w.line("const float diff = " + input + "[base + hw] - mean_c;");
  w.line("var_sum = fmaf(diff, diff, var_sum);");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line("const float var_c = var_sum / n_elems;");
  w.line("const float inv_c = rsqrtf(var_c + eps_v);");
  w.line(inv_std + "[c] = inv_c;");
  w.line("const float wv = " + weight + "[c];");
  w.line("const float bv = " + bias + "[c];");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t base = (n * " + std::to_string(C) + "LL + c) * " + std::to_string(HW) + "LL;");
  w.line("for (int64_t hw = 0; hw < " + std::to_string(HW) + "LL; ++hw) {");
  w.indent();
  w.line("const float centered = " + input + "[base + hw] - mean_c;");
  w.line(output + "[base + hw] = centered * inv_c * wv + bv;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(running_mean_out + "[c] = one_minus_momentum * " + running_mean + "[c] + momentum_v * mean_c;");
  w.line("const float unbiased_var = var_c * (n_elems / n_minus);");
  w.line(running_var_out + "[c] = one_minus_momentum * " + running_var + "[c] + momentum_v * unbiased_var;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  out_bindings["C"] = C;
  out_bindings["HW"] = HW;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent,
                                     {input, weight, bias, running_mean, running_var, eps, momentum, n_elements, n_minus_1, output, mean, inv_std,
                                      running_mean_out, running_var_out},
                                     {},
                                     {input, weight, bias, running_mean, running_var, eps, momentum, n_elements, n_minus_1, output, mean, inv_std,
                                      running_mean_out, running_var_out});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {output, mean, inv_std, running_mean_out, running_var_out};
  out["bindings"] = out_bindings;
  return out;
}

#include "emit_attention_index_norm.inc"
json emit_reduce_prod_2d_all_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "reduce_prod")) fail("reduce_prod(all) lowering expects single reduce_prod");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("reduce_prod(all) expects 1 input");
  if (!_reduce_axis_is_all_2d(op.attrs)) fail("reduce_prod(all) expects dims=[0,1]");
  const std::string in_name = op.inputs[0];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(in_name);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("reduce_prod(all) missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("reduce_prod(all) supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "reduce_prod.all.in");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "reduce_prod.all.out");
  if (i_shape.size() != 2 || !o_shape.empty()) fail("reduce_prod(all) expects rank-2 input and scalar output");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(const float* __restrict__ " + in_name + ", float* __restrict__ " + out_name + ") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("float p = 1.0f;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) p *= " + in_name + "[i];");
  w.line(out_name + "[0] = p;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_reduce_prod_2d_axis1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "reduce_prod")) fail("reduce_prod(axis1) lowering expects single reduce_prod");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 1) fail("reduce_prod(axis1) expects 1 input");
  if (!_reduce_axis_is_1(op.attrs)) fail("reduce_prod(axis1) expects axis=1");
  const std::string in_name = op.inputs[0];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(in_name);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("reduce_prod(axis1) missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("reduce_prod(axis1) supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "reduce_prod.axis1.in");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "reduce_prod.axis1.out");
  if (i_shape.size() != 2) fail("reduce_prod(axis1) expects rank-2 input");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  if (!((o_shape.size() == 1 && o_shape[0] == M) || (o_shape.size() == 2 && o_shape[0] == M && o_shape[1] == 1))) {
    fail("reduce_prod(axis1) expects output shape [M] or [M,1]");
  }
  const bool keepdim = (o_shape.size() == 2);

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
  w.line("const float* __restrict__ " + in_name + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float p = 1.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) p *= " + in_name + "[base + n];");
  if (keepdim) {
    w.line(out_name + "[row] = p;");
  } else {
    w.line(out_name + "[row] = p;");
  }
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_quantile2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "quantile")) fail("quantile lowering expects single quantile op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("quantile expects inp,q");
  int dim = op.attrs.value("dim", 1);
  if (dim == -1) dim = 1;
  if (dim != 1) fail("quantile lowering currently supports dim=1 only");
  const std::string interpolation = op.attrs.value("interpolation", std::string("linear"));
  if (!(interpolation == "linear" || interpolation == "lower" || interpolation == "higher" || interpolation == "nearest" ||
        interpolation == "midpoint")) {
    fail("quantile interpolation unsupported: " + interpolation);
  }
  const std::string inp = op.inputs[0];
  const std::string q = op.inputs[1];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto q_it = intent.tensors.find(q);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || q_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("quantile missing tensors");
  if (i_it->second.dtype != "f32" || q_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("quantile supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "quantile.inp");
  auto q_shape = resolve_shape_required(q_it->second, bindings, "quantile.q");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "quantile.out");
  if (i_shape.size() != 2 || !q_shape.empty()) fail("quantile expects inp:[M,N], q:[]");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const bool keepdim = op.attrs.value("keepdim", false);
  if (keepdim) {
    if (!(o_shape.size() == 2 && o_shape[0] == M && o_shape[1] == 1)) fail("quantile keepdim expects out shape [M,1]");
  } else {
    if (!(o_shape.size() == 1 && o_shape[0] == M)) fail("quantile expects out shape [M]");
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + q + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("float qv = " + q + " ? " + q + "[0] : 0.5f;");
  w.line("if (qv < 0.0f) qv = 0.0f;");
  w.line("if (qv > 1.0f) qv = 1.0f;");
  w.line("const float h = qv * (float)(" + std::to_string(N - 1) + ");");
  w.line("int64_t k0 = (int64_t)floorf(h);");
  w.line("int64_t k1 = (int64_t)ceilf(h);");
  w.line("if (k0 < 0) k0 = 0;");
  w.line("if (k1 < 0) k1 = 0;");
  w.line("if (k0 >= " + std::to_string(N) + "LL) k0 = " + std::to_string(N - 1) + "LL;");
  w.line("if (k1 >= " + std::to_string(N) + "LL) k1 = " + std::to_string(N - 1) + "LL;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float v0 = 0.0f;");
  w.line("float v1 = 0.0f;");
  w.line("bool f0 = false;");
  w.line("bool f1 = false;");
  w.line("for (int64_t i = 0; i < " + std::to_string(N) + "LL; ++i) {");
  w.indent();
  w.line("const float vi = " + inp + "[base + i];");
  w.line("int64_t rank = 0;");
  w.line("for (int64_t j = 0; j < " + std::to_string(N) + "LL; ++j) {");
  w.indent();
  w.line("const float vj = " + inp + "[base + j];");
  w.line("if (vj < vi || (vj == vi && j < i)) ++rank;");
  w.dedent();
  w.line("}");
  w.line("if (!f0 && rank == k0) { v0 = vi; f0 = true; }");
  w.line("if (!f1 && rank == k1) { v1 = vi; f1 = true; }");
  w.line("if (f0 && f1) break;");
  w.dedent();
  w.line("}");
  w.line("if (!f0) v0 = " + inp + "[base];");
  w.line("if (!f1) v1 = " + inp + "[base];");
  w.line("const float frac = h - floorf(h);");
  std::string q_expr;
  if (interpolation == "lower") {
    q_expr = "v0";
  } else if (interpolation == "higher") {
    q_expr = "v1";
  } else if (interpolation == "nearest") {
    q_expr = "(frac < 0.5f ? v0 : v1)";
  } else if (interpolation == "midpoint") {
    q_expr = "(0.5f * (v0 + v1))";
  } else {
    q_expr = "(v0 + (v1 - v0) * frac)";
  }
  w.line("const float q_out = " + q_expr + ";");
  w.line(out_name + "[row] = q_out;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, q, out_name}, {}, {inp, q, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_rms_norm2d_f32(const Intent& intent, const json& bindings) {
  if (intent.outputs.size() != 2) fail("rms_norm lowering expects two outputs [out, INV_RMS]");
  const std::string out_name = intent.outputs[0];
  const std::string inv_name = intent.outputs[1];
  const std::string input = "input";
  const std::string weight = "weight";
  auto in_it = intent.tensors.find(input);
  auto w_it = intent.tensors.find(weight);
  auto o_it = intent.tensors.find(out_name);
  auto r_it = intent.tensors.find(inv_name);
  if (in_it == intent.tensors.end() || w_it == intent.tensors.end() || o_it == intent.tensors.end() || r_it == intent.tensors.end()) {
    fail("rms_norm lowering missing input/weight/out/inv tensors");
  }
  if (in_it->second.dtype != "f32" || w_it->second.dtype != "f32" || o_it->second.dtype != "f32" || r_it->second.dtype != "f32") {
    fail("rms_norm lowering supports f32 only");
  }
  auto in_shape = resolve_shape_required(in_it->second, bindings, "rms_norm.input");
  auto w_shape = resolve_shape_required(w_it->second, bindings, "rms_norm.weight");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "rms_norm.out");
  auto r_shape = resolve_shape_required(r_it->second, bindings, "rms_norm.inv");
  if (in_shape.size() != 2 || w_shape.size() != 1 || o_shape.size() != 2 || r_shape.size() != 1) {
    fail("rms_norm expects input:[M,N], weight:[N], out:[M,N], INV_RMS:[M]");
  }
  const int64_t M = in_shape[0];
  const int64_t N = in_shape[1];
  if (w_shape[0] != N || o_shape[0] != M || o_shape[1] != N || r_shape[0] != M) fail("rms_norm shape mismatch");
  float eps_v = 1e-5f;
  for (const auto& op : intent.ops) {
    if (op.op == "const" && op.output == "eps" && op.attrs.is_object() && op.attrs.contains("value")) {
      const auto& v = op.attrs["value"];
      if (v.is_number()) eps_v = static_cast<float>(v.get<double>());
      break;
    }
  }

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + input + ",");
  w.line("const float* __restrict__ " + weight + ",");
  w.line("float* __restrict__ " + out_name + ",");
  w.line("float* __restrict__ " + inv_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum_sq = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float x = " + input + "[base + n];");
  w.line("sum_sq = fmaf(x, x, sum_sq);");
  w.dedent();
  w.line("}");
  w.line("const float rrms = rsqrtf(sum_sq / (float)" + std::to_string(N) + " + " + std::to_string(eps_v) + "f);");
  w.line(inv_name + "[row] = rrms;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line(out_name + "[base + n] = " + input + "[base + n] * rrms * " + weight + "[n];");
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
  out["io_spec"] = io_spec_from_args(intent, {input, weight, out_name, inv_name}, {}, {input, weight, out_name, inv_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name, inv_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_stack_axis0_2x2d_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "stack")) fail("stack lowering expects single stack op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("stack currently supports exactly 2 inputs");
  int axis = op.attrs.value("axis", 0);
  if (axis < 0) axis += 3;
  if (axis != 0) fail("stack lowering currently supports axis=0 only");
  const std::string a = op.inputs[0];
  const std::string b = op.inputs[1];
  const std::string out_name = op.output;
  auto a_it = intent.tensors.find(a);
  auto b_it = intent.tensors.find(b);
  auto o_it = intent.tensors.find(out_name);
  if (a_it == intent.tensors.end() || b_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("stack missing tensors");
  if (a_it->second.dtype != "f32" || b_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("stack supports f32 only");
  auto a_shape = resolve_shape_required(a_it->second, bindings, "stack.a");
  auto b_shape = resolve_shape_required(b_it->second, bindings, "stack.b");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "stack.out");
  if (a_shape.size() != 2 || b_shape.size() != 2 || a_shape != b_shape) fail("stack expects matching rank-2 inputs");
  if (o_shape.size() != 3 || o_shape[0] != 2 || o_shape[1] != a_shape[0] || o_shape[2] != a_shape[1]) {
    fail("stack output shape mismatch: expect [2,M,N]");
  }
  const int64_t M = a_shape[0];
  const int64_t N = a_shape[1];
  const int64_t MN = M * N;
  const int64_t total = 2 * MN;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + a + ",");
  w.line("const float* __restrict__ " + b + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("if (tid < " + std::to_string(MN) + "LL) {");
  w.indent();
  w.line(out_name + "[tid] = " + a + "[tid];");
  w.dedent();
  w.line("} else {");
  w.indent();
  w.line("const int64_t i = tid - " + std::to_string(MN) + "LL;");
  w.line(out_name + "[tid] = " + b + "[i];");
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
  out["io_spec"] = io_spec_from_args(intent, {a, b, out_name}, {}, {a, b, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_scatter2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "scatter")) fail("scatter lowering expects single scatter op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 3) fail("scatter expects inp,index,src");
  int dim = op.attrs.value("dim", 1);
  if (dim < 0) dim += 2;
  if (dim != 1) fail("scatter currently supports dim=1 only");
  const std::string inp = op.inputs[0];
  const std::string index = op.inputs[1];
  const std::string src = op.inputs[2];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto x_it = intent.tensors.find(index);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || x_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("scatter missing tensors");
  }
  if (i_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("scatter supports f32 inp/src/out");
  const std::string idx_dt = x_it->second.dtype;
  if (!(idx_dt == "i32" || idx_dt == "i64")) fail("scatter index dtype must be i32/i64");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "scatter.inp");
  auto x_shape = resolve_shape_required(x_it->second, bindings, "scatter.index");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "scatter.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "scatter.out");
  if (i_shape.size() != 2 || x_shape.size() != 2 || s_shape.size() != 2 || o_shape.size() != 2) fail("scatter expects rank-2 tensors");
  if (i_shape != x_shape || i_shape != s_shape || i_shape != o_shape) fail("scatter shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const std::string idx_ctype = (idx_dt == "i32") ? "int" : "int64_t";

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const " + idx_ctype + "* __restrict__ " + index + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out_name + "[i] = " + inp + "[i];");
  w.line("for (int64_t m = 0; m < " + std::to_string(M) + "LL; ++m) {");
  w.indent();
  w.line("const int64_t row = m * " + std::to_string(N) + "LL;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const int64_t col = (int64_t)" + index + "[row + n];");
  w.line("if (col < 0 || col >= " + std::to_string(N) + "LL) continue;");
  w.line(out_name + "[row + col] = " + src + "[row + n];");
  w.dedent();
  w.line("}");
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
  out["io_spec"] = io_spec_from_args(intent, {inp, index, src, out_name}, {}, {inp, index, src, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_select_scatter2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "select_scatter")) fail("select_scatter lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("select_scatter expects inp,src");
  int dim = op.attrs.value("dim", 1);
  if (dim < 0) dim += 2;
  if (dim != 1) fail("select_scatter currently supports dim=1 only");
  int64_t index = op.attrs.value("index", static_cast<int64_t>(0));
  const std::string inp = op.inputs[0];
  const std::string src = op.inputs[1];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("select_scatter missing tensors");
  if (i_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("select_scatter supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "select_scatter.inp");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "select_scatter.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "select_scatter.out");
  if (i_shape.size() != 2 || s_shape.size() != 1 || o_shape.size() != 2) fail("select_scatter expects inp/out rank-2 and src rank-1");
  if (i_shape != o_shape || s_shape[0] != i_shape[0]) fail("select_scatter shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  if (index < 0) index += N;
  if (index < 0 || index >= N) fail("select_scatter index out of range");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out_name + "[i] = " + inp + "[i];");
  w.line("for (int64_t m = 0; m < " + std::to_string(M) + "LL; ++m) {");
  w.indent();
  w.line(out_name + "[m * " + std::to_string(N) + "LL + " + std::to_string(index) + "LL] = " + src + "[m];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["index"] = index;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, src, out_name}, {}, {inp, src, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_slice_scatter2d_dim1_f32(const Intent& intent, const json& bindings) {
  if (!(intent.ops.size() == 1 && intent.ops[0].op == "slice_scatter")) fail("slice_scatter lowering expects single op");
  const Op& op = intent.ops[0];
  if (op.inputs.size() != 2) fail("slice_scatter expects inp,src");
  int dim = op.attrs.value("dim", 1);
  if (dim < 0) dim += 2;
  if (dim != 1) fail("slice_scatter currently supports dim=1 only");
  int64_t start = op.attrs.value("start", static_cast<int64_t>(0));
  int64_t end = op.attrs.value("end", static_cast<int64_t>(0));
  int64_t step = op.attrs.value("step", static_cast<int64_t>(1));
  if (step <= 0) fail("slice_scatter requires step > 0");
  const std::string inp = op.inputs[0];
  const std::string src = op.inputs[1];
  const std::string out_name = op.output;
  auto i_it = intent.tensors.find(inp);
  auto s_it = intent.tensors.find(src);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || s_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("slice_scatter missing tensors");
  if (i_it->second.dtype != "f32" || s_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("slice_scatter supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "slice_scatter.inp");
  auto s_shape = resolve_shape_required(s_it->second, bindings, "slice_scatter.src");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "slice_scatter.out");
  if (i_shape.size() != 2 || s_shape.size() != 2 || o_shape.size() != 2) fail("slice_scatter expects rank-2 tensors");
  if (i_shape != o_shape || s_shape[0] != i_shape[0]) fail("slice_scatter shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  if (start < 0) start += N;
  if (end < 0) end += N;
  if (start < 0) start = 0;
  if (end > N) end = N;
  if (end < start) end = start;
  const int64_t expected_L = (end - start + step - 1) / step;
  if (s_shape[1] != expected_L) fail("slice_scatter src second dim mismatch with (start,end,step)");

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("const float* __restrict__ " + src + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("if ((int)blockIdx.x != 0 || (int)threadIdx.x != 0) return;");
  w.line("for (int64_t i = 0; i < " + std::to_string(M * N) + "LL; ++i) " + out_name + "[i] = " + inp + "[i];");
  w.line("for (int64_t m = 0; m < " + std::to_string(M) + "LL; ++m) {");
  w.indent();
  w.line("int64_t l = 0;");
  w.line("for (int64_t n = " + std::to_string(start) + "LL; n < " + std::to_string(end) + "LL; n += " + std::to_string(step) + "LL) {");
  w.indent();
  w.line(out_name + "[m * " + std::to_string(N) + "LL + n] = " + src + "[m * " + std::to_string(expected_L) + "LL + l];");
  w.line("++l;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["start"] = start;
  out_bindings["end"] = end;
  out_bindings["step"] = step;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, src, out_name}, {}, {inp, src, out_name});
  out["launch"] = {{"grid", {1, 1, 1}}, {"block", {1, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_var_mean2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 2 || intent.ops[0].op != "std" || intent.ops[1].op != "mul") {
    fail("var_mean lowering expects std->mul pattern");
  }
  const Op& std_op = intent.ops[0];
  const Op& mul_op = intent.ops[1];
  if (std_op.inputs.size() != 1 || mul_op.inputs.size() != 2) fail("var_mean arity mismatch");
  if (!((mul_op.inputs[0] == std_op.output && mul_op.inputs[1] == std_op.output))) fail("var_mean expects mul(std_out,std_out)");
  if (!_reduce_axis_is_1(std_op.attrs)) fail("var_mean currently supports axis=1 only");
  const int64_t correction = std_op.attrs.value("correction", static_cast<int64_t>(1));
  const std::string inp = std_op.inputs[0];
  const std::string out_name = mul_op.output;
  auto i_it = intent.tensors.find(inp);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("var_mean missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("var_mean supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "var_mean.inp");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "var_mean.out");
  if (i_shape.size() != 2 || o_shape.size() != 1 || o_shape[0] != i_shape[0]) fail("var_mean shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];
  const float denom = (N - correction > 0) ? static_cast<float>(N - correction) : 1.0f;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float mean = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) mean += " + inp + "[base + n];");
  w.line("mean /= (float)" + std::to_string(N) + ";");
  w.line("float ss = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float d = " + inp + "[base + n] - mean;");
  w.line("ss = fmaf(d, d, ss);");
  w.dedent();
  w.line("}");
  w.line(out_name + "[row] = ss / " + std::to_string(denom) + "f;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  out_bindings["correction"] = correction;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, out_name}, {}, {inp, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_vector_norm2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 3 || intent.ops[0].op != "mul" || intent.ops[1].op != "reduce_sum" || intent.ops[2].op != "sqrt") {
    fail("vector_norm lowering expects mul->reduce_sum->sqrt");
  }
  const Op& mul = intent.ops[0];
  const Op& red = intent.ops[1];
  const Op& sqr = intent.ops[2];
  if (mul.inputs.size() != 2 || red.inputs.size() != 1 || sqr.inputs.size() != 1) fail("vector_norm arity mismatch");
  if (!(mul.inputs[0] == mul.inputs[1])) fail("vector_norm expects mul(inp,inp)");
  if (red.inputs[0] != mul.output || sqr.inputs[0] != red.output) fail("vector_norm wiring mismatch");
  if (!_reduce_axis_is_1(red.attrs)) fail("vector_norm currently supports axis=1 only");
  const std::string inp = mul.inputs[0];
  const std::string out_name = sqr.output;
  auto i_it = intent.tensors.find(inp);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("vector_norm missing tensors");
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("vector_norm supports f32 only");
  auto i_shape = resolve_shape_required(i_it->second, bindings, "vector_norm.inp");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "vector_norm.out");
  if (i_shape.size() != 2 || o_shape.size() != 1 || o_shape[0] != i_shape[0]) fail("vector_norm shape mismatch");
  const int64_t M = i_shape[0];
  const int64_t N = i_shape[1];

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + inp + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum_sq = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float x = " + inp + "[base + n];");
  w.line("sum_sq = fmaf(x, x, sum_sq);");
  w.dedent();
  w.line("}");
  w.line(out_name + "[row] = sqrtf(sum_sq);");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["M"] = M;
  out_bindings["N"] = N;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, {inp, out_name}, {}, {inp, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_weight_norm2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 6 || intent.ops[0].op != "mul" || intent.ops[1].op != "reduce_sum" || intent.ops[2].op != "sqrt" ||
      intent.ops[3].op != "div" || intent.ops[4].op != "broadcast_in_dim" || intent.ops[5].op != "mul") {
    fail("weight_norm lowering expects mul->reduce_sum->sqrt->div->broadcast->mul");
  }
  const Op& vv = intent.ops[0];
  const Op& red = intent.ops[1];
  const Op& sqr = intent.ops[2];
  const Op& div = intent.ops[3];
  const Op& bcast = intent.ops[4];
  const Op& mul = intent.ops[5];
  if (vv.inputs.size() != 2 || red.inputs.size() != 1 || sqr.inputs.size() != 1 || div.inputs.size() != 2 || bcast.inputs.size() != 1 ||
      mul.inputs.size() != 2) {
    fail("weight_norm arity mismatch");
  }
  if (!(vv.inputs[0] == vv.inputs[1])) fail("weight_norm expects vv = mul(v,v)");
  if (red.inputs[0] != vv.output || sqr.inputs[0] != red.output) fail("weight_norm reduce/sqrt wiring mismatch");
  if (!_reduce_axis_is_1(red.attrs)) fail("weight_norm currently supports axis=1 only");
  const std::string v = vv.inputs[0];
  const std::string g = (div.inputs[0] == sqr.output) ? div.inputs[1] : ((div.inputs[1] == sqr.output) ? div.inputs[0] : std::string());
  if (g.empty()) fail("weight_norm expects div(g,norm)");
  if (bcast.inputs[0] != div.output) fail("weight_norm expects broadcast(scale)");
  if (!((mul.inputs[0] == v && mul.inputs[1] == bcast.output) || (mul.inputs[1] == v && mul.inputs[0] == bcast.output))) {
    fail("weight_norm expects mul(v, scale_bc)");
  }
  const std::string out_name = mul.output;
  auto v_it = intent.tensors.find(v);
  auto g_it = intent.tensors.find(g);
  auto o_it = intent.tensors.find(out_name);
  if (v_it == intent.tensors.end() || g_it == intent.tensors.end() || o_it == intent.tensors.end()) fail("weight_norm missing tensors");
  if (v_it->second.dtype != "f32" || g_it->second.dtype != "f32" || o_it->second.dtype != "f32") fail("weight_norm supports f32 only");
  auto v_shape = resolve_shape_required(v_it->second, bindings, "weight_norm.v");
  auto g_shape = resolve_shape_required(g_it->second, bindings, "weight_norm.g");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "weight_norm.out");
  if (v_shape.size() != 2 || g_shape.size() != 1 || o_shape.size() != 2) fail("weight_norm shape rank mismatch");
  if (o_shape != v_shape || g_shape[0] != v_shape[0]) fail("weight_norm shape mismatch");
  const int64_t M = v_shape[0];
  const int64_t N = v_shape[1];

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_m", 256);
  if (block_x <= 0) block_x = 256;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (M + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + v + ",");
  w.line("const float* __restrict__ " + g + ",");
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t row = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (row >= " + std::to_string(M) + "LL) return;");
  w.line("const int64_t base = row * " + std::to_string(N) + "LL;");
  w.line("float sum_sq = 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line("const float x = " + v + "[base + n];");
  w.line("sum_sq = fmaf(x, x, sum_sq);");
  w.dedent();
  w.line("}");
  w.line("const float norm = sqrtf(sum_sq);");
  w.line("const float scale = (norm > 0.0f) ? (" + g + "[row] / norm) : 0.0f;");
  w.line("for (int64_t n = 0; n < " + std::to_string(N) + "LL; ++n) {");
  w.indent();
  w.line(out_name + "[base + n] = " + v + "[base + n] * scale;");
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
  out["io_spec"] = io_spec_from_args(intent, {v, g, out_name}, {}, {v, g, out_name});
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_upsample_bicubic2d_aa_f32(const Intent& intent, const json& bindings) {
  bool direct_semantic = false;
  if (intent.ops.size() == 1 && intent.ops[0].op == "upsample_bicubic2d_aa") direct_semantic = true;
  const Op* semantic_op = direct_semantic ? &intent.ops[0] : nullptr;

  std::string in_name;
  std::string out_name;
  bool has_scale_tensors = false;
  std::string rsh_name;
  std::string rsw_name;
  if (semantic_op != nullptr) {
    if (!(semantic_op->inputs.size() == 1 || semantic_op->inputs.size() == 3)) {
      fail("upsample_bicubic2d_aa expects ptr_i or ptr_i+reciprocal scales");
    }
    in_name = semantic_op->inputs[0];
    out_name = semantic_op->output;
    has_scale_tensors = (semantic_op->inputs.size() == 3);
    rsh_name = has_scale_tensors ? semantic_op->inputs[1] : std::string();
    rsw_name = has_scale_tensors ? semantic_op->inputs[2] : std::string();
  } else {
    // Fallback for expanded intent payloads used by certain harnesses.
    if (intent.tensors.find("ptr_i") == intent.tensors.end() || intent.tensors.find("ptr_o") == intent.tensors.end()) {
      fail("upsample_bicubic2d_aa expanded lowering expects tensors ptr_i/ptr_o");
    }
    in_name = "ptr_i";
    out_name = "ptr_o";
    has_scale_tensors = (intent.tensors.find("reciprocal_scale_h") != intent.tensors.end() && intent.tensors.find("reciprocal_scale_w") != intent.tensors.end());
    rsh_name = has_scale_tensors ? "reciprocal_scale_h" : std::string();
    rsw_name = has_scale_tensors ? "reciprocal_scale_w" : std::string();
  }
  auto i_it = intent.tensors.find(in_name);
  auto o_it = intent.tensors.find(out_name);
  if (i_it == intent.tensors.end() || o_it == intent.tensors.end()) {
    fail("upsample_bicubic2d_aa missing tensors");
  }
  if (i_it->second.dtype != "f32" || o_it->second.dtype != "f32") {
    fail("upsample_bicubic2d_aa supports f32 only");
  }
  float reciprocal_scale_h_const = 1.0f;
  float reciprocal_scale_w_const = 1.0f;
  if (semantic_op != nullptr) {
    reciprocal_scale_h_const = static_cast<float>(semantic_op->attrs.value("invscale", 1.0));
    reciprocal_scale_w_const = static_cast<float>(semantic_op->attrs.value("invscale", 1.0));
    if (semantic_op->attrs.is_object()) {
      if (semantic_op->attrs.contains("reciprocal_scale_h")) reciprocal_scale_h_const = static_cast<float>(semantic_op->attrs["reciprocal_scale_h"].get<double>());
      if (semantic_op->attrs.contains("reciprocal_scale_w")) reciprocal_scale_w_const = static_cast<float>(semantic_op->attrs["reciprocal_scale_w"].get<double>());
    }
  }
  if (has_scale_tensors) {
    auto h_it = intent.tensors.find(rsh_name);
    auto w_it = intent.tensors.find(rsw_name);
    if (h_it == intent.tensors.end() || w_it == intent.tensors.end()) fail("upsample_bicubic2d_aa missing reciprocal scale tensors");
    if (h_it->second.dtype != "f32" || w_it->second.dtype != "f32") fail("upsample_bicubic2d_aa reciprocal scales must be f32");
  }
  auto i_shape = resolve_shape_required(i_it->second, bindings, "upsample_bicubic2d_aa.in");
  auto o_shape = resolve_shape_required(o_it->second, bindings, "upsample_bicubic2d_aa.out");
  if (i_shape.size() != 4 || o_shape.size() != 4) fail("upsample_bicubic2d_aa expects rank-4 [N,C,H,W]");
  const int64_t N = i_shape[0];
  const int64_t C = i_shape[1];
  const int64_t IH = i_shape[2];
  const int64_t IW = i_shape[3];
  const int64_t ON = o_shape[0];
  const int64_t OC = o_shape[1];
  const int64_t OH = o_shape[2];
  const int64_t OW = o_shape[3];
  if (ON != N || OC != C) fail("upsample_bicubic2d_aa N/C mismatch");
  const int64_t total = N * C * OH * OW;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x < 32) block_x = 32;
  if (block_x > 1024) block_x = 1024;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (total + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + in_name + ",");
  if (has_scale_tensors) {
    w.line("const float* __restrict__ " + rsh_name + ",");
    w.line("const float* __restrict__ " + rsw_name + ",");
  }
  w.line("float* __restrict__ " + out_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (tid >= " + std::to_string(total) + "LL) return;");
  w.line("int64_t t = tid;");
  w.line("const int64_t ow = t % " + std::to_string(OW) + "LL; t /= " + std::to_string(OW) + "LL;");
  w.line("const int64_t oh = t % " + std::to_string(OH) + "LL; t /= " + std::to_string(OH) + "LL;");
  w.line("const int64_t c = t % " + std::to_string(C) + "LL; t /= " + std::to_string(C) + "LL;");
  w.line("const int64_t n = t;");
  if (has_scale_tensors) {
    w.line("const float reciprocal_scale_h_val = " + rsh_name + " ? " + rsh_name + "[0] : ((float)" + std::to_string(IH) + " / (float)" +
           std::to_string(OH) + ");");
    w.line("const float reciprocal_scale_w_val = " + rsw_name + " ? " + rsw_name + "[0] : ((float)" + std::to_string(IW) + " / (float)" +
           std::to_string(OW) + ");");
  } else {
    w.line("const float reciprocal_scale_h_val = " + std::to_string(reciprocal_scale_h_const) + "f;");
    w.line("const float reciprocal_scale_w_val = " + std::to_string(reciprocal_scale_w_const) + "f;");
  }
  w.line("const float support_h = (reciprocal_scale_h_val >= 1.0f) ? (2.0f * reciprocal_scale_h_val) : 2.0f;");
  w.line("const float support_w = (reciprocal_scale_w_val >= 1.0f) ? (2.0f * reciprocal_scale_w_val) : 2.0f;");
  w.line("const int interp_h = ((int)(support_h + 0.5f)) * 2 + 1;");
  w.line("const int interp_w = ((int)(support_w + 0.5f)) * 2 + 1;");
  w.line("const float center_h = ((float)oh + 0.5f) * reciprocal_scale_h_val;");
  w.line("const float center_w = ((float)ow + 0.5f) * reciprocal_scale_w_val;");
  w.line("const int span_start_h = (int)fmaxf(center_h - support_h + 0.5f, 0.0f);");
  w.line("const int span_start_w = (int)fmaxf(center_w - support_w + 0.5f, 0.0f);");
  w.line("const int span_size_h = (int)(fminf(center_h + support_h + 0.5f, (float)" + std::to_string(IH) + ") - (float)span_start_h);");
  w.line("const int span_size_w = (int)(fminf(center_w + support_w + 0.5f, (float)" + std::to_string(IW) + ") - (float)span_start_w);");
  w.line("const float invscale_h = (reciprocal_scale_h_val >= 1.0f) ? (1.0f / reciprocal_scale_h_val) : 1.0f;");
  w.line("const float invscale_w = (reciprocal_scale_w_val >= 1.0f) ? (1.0f / reciprocal_scale_w_val) : 1.0f;");
  w.line("const float start_minus_center_h = (float)span_start_h - center_h;");
  w.line("const float start_minus_center_w = (float)span_start_w - center_w;");
  w.line("const float a = -0.5f;");
  w.line("float wy[17];");
  w.line("float wx[17];");
  w.line("float sum_wy = 0.0f;");
  w.line("float sum_wx = 0.0f;");
  w.line("for (int y = 0; y < interp_h; ++y) {");
  w.indent();
  w.line("float wv = 0.0f;");
  w.line("if (y < span_size_h) {");
  w.indent();
  w.line("const float d = fabsf(((float)y + start_minus_center_h + 0.5f) * invscale_h);");
  w.line("if (d < 1.0f) wv = ((a + 2.0f) * d - (a + 3.0f)) * d * d + 1.0f;");
  w.line("else if (d < 2.0f) wv = (((d - 5.0f) * d + 8.0f) * d - 4.0f) * a;");
  w.dedent();
  w.line("}");
  w.line("wy[y] = wv;");
  w.line("sum_wy += wv;");
  w.dedent();
  w.line("}");
  w.line("for (int x = 0; x < interp_w; ++x) {");
  w.indent();
  w.line("float wv = 0.0f;");
  w.line("if (x < span_size_w) {");
  w.indent();
  w.line("const float d = fabsf(((float)x + start_minus_center_w + 0.5f) * invscale_w);");
  w.line("if (d < 1.0f) wv = ((a + 2.0f) * d - (a + 3.0f)) * d * d + 1.0f;");
  w.line("else if (d < 2.0f) wv = (((d - 5.0f) * d + 8.0f) * d - 4.0f) * a;");
  w.dedent();
  w.line("}");
  w.line("wx[x] = wv;");
  w.line("sum_wx += wv;");
  w.dedent();
  w.line("}");
  w.line("if (sum_wy == 0.0f) sum_wy = 1.0f;");
  w.line("if (sum_wx == 0.0f) sum_wx = 1.0f;");
  w.line("for (int y = 0; y < interp_h; ++y) wy[y] /= sum_wy;");
  w.line("for (int x = 0; x < interp_w; ++x) wx[x] /= sum_wx;");
  w.line("float acc = 0.0f;");
  w.line("for (int y = 0; y < interp_h; ++y) {");
  w.indent();
  w.line("const int iy = span_start_h + y;");
  w.line("if (iy < 0 || iy >= " + std::to_string(IH) + ") continue;");
  w.line("for (int x = 0; x < interp_w; ++x) {");
  w.indent();
  w.line("const int ix = span_start_w + x;");
  w.line("if (ix < 0 || ix >= " + std::to_string(IW) + ") continue;");
  w.line("const int64_t in_idx = (((n * " + std::to_string(C) + "LL + c) * " + std::to_string(IH) + "LL + (int64_t)iy) * " +
         std::to_string(IW) + "LL + (int64_t)ix);");
  w.line("acc += " + in_name + "[in_idx] * wy[y] * wx[x];");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");
  w.line(out_name + "[tid] = acc;");
  w.dedent();
  w.line("}");

  json out_bindings = bindings;
  out_bindings["N"] = N;
  out_bindings["C"] = C;
  out_bindings["IH"] = IH;
  out_bindings["IW"] = IW;
  out_bindings["OH"] = OH;
  out_bindings["OW"] = OW;
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  if (has_scale_tensors) {
    out["io_spec"] = io_spec_from_args(intent, {in_name, rsh_name, rsw_name, out_name}, {}, {in_name, rsh_name, rsw_name, out_name});
  } else {
    out["io_spec"] = io_spec_from_args(intent, {in_name, out_name}, {}, {in_name, out_name});
  }
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {out_name};
  out["bindings"] = out_bindings;
  return out;
}

json emit_per_token_group_quant_fp8_2d_f32(const Intent& intent, const json& bindings) {
  if (intent.ops.size() != 11) fail("per_token_group_quant_fp8 lowering expects 11-op pattern");
  static const std::vector<std::string> kOps = {
      "reshape", "abs", "reduce_max", "max", "div", "reshape", "broadcast_in_dim", "div", "max", "min", "reshape"};
  for (size_t i = 0; i < kOps.size(); ++i) {
    if (intent.ops[i].op != kOps[i]) fail("per_token_group_quant_fp8 lowering pattern mismatch");
  }
  const Op& reshape0 = intent.ops[0];
  const Op& clamp_eps = intent.ops[3];
  const Op& scale_div = intent.ops[4];
  const Op& scale_reshape = intent.ops[5];
  const Op& clamp_min = intent.ops[8];
  const Op& clamp_max = intent.ops[9];
  const Op& out_reshape = intent.ops[10];

  if (reshape0.inputs.size() != 1) fail("per_token_group_quant_fp8 reshape input mismatch");
  if (clamp_eps.inputs.size() != 2 || scale_div.inputs.size() != 2 || clamp_min.inputs.size() != 2 || clamp_max.inputs.size() != 2) {
    fail("per_token_group_quant_fp8 scalar inputs mismatch");
  }

  const std::string y_name = reshape0.inputs[0];
  const std::string y_q_name = out_reshape.output;
  const std::string y_s_name = scale_reshape.output;
  const std::string eps_name = clamp_eps.inputs[1];
  const std::string fp8_max_name = scale_div.inputs[1];
  const std::string fp8_min_name = clamp_min.inputs[1];

  auto y_it = intent.tensors.find(y_name);
  auto yq_it = intent.tensors.find(y_q_name);
  auto ys_it = intent.tensors.find(y_s_name);
  auto eps_it = intent.tensors.find(eps_name);
  auto min_it = intent.tensors.find(fp8_min_name);
  auto max_it = intent.tensors.find(fp8_max_name);
  if (y_it == intent.tensors.end() || yq_it == intent.tensors.end() || ys_it == intent.tensors.end() || eps_it == intent.tensors.end() ||
      min_it == intent.tensors.end() || max_it == intent.tensors.end()) {
    fail("per_token_group_quant_fp8 tensors missing");
  }
  if (y_it->second.dtype != "f32" || yq_it->second.dtype != "f32" || ys_it->second.dtype != "f32" || eps_it->second.dtype != "f32" ||
      min_it->second.dtype != "f32" || max_it->second.dtype != "f32") {
    fail("per_token_group_quant_fp8 supports f32 tensors/scalars only");
  }

  auto y_shape = resolve_shape_required(y_it->second, bindings, "per_token_group_quant_fp8.y.shape");
  auto ys_shape = resolve_shape_required(ys_it->second, bindings, "per_token_group_quant_fp8.y_s.shape");
  if (y_shape.size() != 2 || ys_shape.size() != 2) fail("per_token_group_quant_fp8 expects rank-2 tensors");
  const int64_t M = y_shape[0];
  const int64_t N = y_shape[1];
  const int64_t G = ys_shape[1];
  if (M <= 0 || N <= 0 || G <= 0) fail("per_token_group_quant_fp8 invalid shape");
  if (ys_shape[0] != M) fail("per_token_group_quant_fp8 y_s shape mismatch");
  if ((N % G) != 0) fail("per_token_group_quant_fp8 requires N % G == 0");
  const int64_t GROUP_SIZE = N / G;
  const int64_t MG = M * G;

  int64_t block_x = resolve_schedule_int(intent, bindings, "tile_n", 128);
  if (block_x <= 0) block_x = 128;
  if (block_x > 1024) block_x = 1024;
  if (block_x < 32) block_x = 32;
  if ((block_x % 32) != 0) block_x = ((block_x + 31) / 32) * 32;
  const int64_t grid_x = (MG + block_x - 1) / block_x;

  std::ostringstream cuda_ss;
  CodeWriter w(cuda_ss);
  w.line("#include <math.h>");
  w.line("#include <stdint.h>");
  w.blank();
  w.line("extern \"C\" __global__ __launch_bounds__(" + std::to_string(block_x) + ") void " + intent.name + "(");
  w.indent();
  w.line("const float* __restrict__ " + y_name + ",");
  w.line("const float* __restrict__ " + eps_name + ",");
  w.line("const float* __restrict__ " + fp8_min_name + ",");
  w.line("const float* __restrict__ " + fp8_max_name + ",");
  w.line("float* __restrict__ " + y_q_name + ",");
  w.line("float* __restrict__ " + y_s_name);
  w.dedent();
  w.line(") {");
  w.indent();
  w.line("const int64_t gid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;");
  w.line("if (gid >= " + std::to_string(MG) + "LL) return;");
  w.line("const int64_t m = gid / " + std::to_string(G) + "LL;");
  w.line("const int64_t g = gid - m * " + std::to_string(G) + "LL;");
  w.line("const int64_t base = m * " + std::to_string(N) + "LL + g * " + std::to_string(GROUP_SIZE) + "LL;");
  w.line("float absmax = 0.0f;");
  w.line("for (int64_t k = 0; k < " + std::to_string(GROUP_SIZE) + "LL; ++k) {");
  w.indent();
  w.line("absmax = fmaxf(absmax, fabsf(" + y_name + "[base + k]));");
  w.dedent();
  w.line("}");
  w.line("float scale = fmaxf(absmax, " + eps_name + "[0]) / " + fp8_max_name + "[0];");
  w.line("if (!isfinite(scale) || scale == 0.0f) scale = 1.0f;");
  w.line(y_s_name + "[m * " + std::to_string(G) + "LL + g] = scale;");
  w.line("const float inv_scale = 1.0f / scale;");
  w.line("const float clamp_lo = " + fp8_min_name + "[0];");
  w.line("const float clamp_hi = " + fp8_max_name + "[0];");
  w.line("for (int64_t k = 0; k < " + std::to_string(GROUP_SIZE) + "LL; ++k) {");
  w.indent();
  w.line("float q = " + y_name + "[base + k] * inv_scale;");
  w.line("q = fmaxf(q, clamp_lo);");
  w.line("q = fminf(q, clamp_hi);");
  w.line(y_q_name + "[base + k] = q;");
  w.dedent();
  w.line("}");
  w.dedent();
  w.line("}");

  std::vector<std::string> tensor_args = {y_name, eps_name, fp8_min_name, fp8_max_name, y_q_name, y_s_name};
  std::vector<std::string> arg_names = {y_name, eps_name, fp8_min_name, fp8_max_name, y_q_name, y_s_name};
  json out;
  out["kernel_name"] = intent.name;
  out["cuda_src"] = cuda_ss.str();
  out["io_spec"] = io_spec_from_args(intent, tensor_args, {}, arg_names);
  out["launch"] = {{"grid", {grid_x, 1, 1}}, {"block", {block_x, 1, 1}}, {"shared_mem", 0}};
  out["output_names"] = {y_q_name, y_s_name};
  out["bindings"] = bindings;
  return out;
}

#include "dispatch_lowering.inc"

}  // namespace

#ifdef INTENTIR_CUDA_CODEGEN_PYBIND
namespace py = pybind11;

static std::string lower_from_json_str(const std::string& intent_json, const std::string& bindings_json) {
  Intent intent = parse_intent(json::parse(intent_json));
  json bindings = json::parse(bindings_json);
  json out = lower_intent_to_cuda(intent, bindings);
  return out.dump();
}

#ifndef INTENTIR_CUDA_CODEGEN_PYBIND_MODULE_NAME
#define INTENTIR_CUDA_CODEGEN_PYBIND_MODULE_NAME intentir_cuda_codegen_ext
#endif

PYBIND11_MODULE(INTENTIR_CUDA_CODEGEN_PYBIND_MODULE_NAME, m) {
  m.doc() = "IntentIR CUDA C++ codegen (pybind11 wrapper)";
  m.def("lower_from_json_str", &lower_from_json_str, py::arg("intent_json"), py::arg("bindings_json"));
}
#endif

#ifndef INTENTIR_CUDA_CODEGEN_NO_MAIN
int main(int argc, char** argv) {
  try {
    std::string intent_path;
    std::string bindings_path;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--intent" && i + 1 < argc) intent_path = argv[++i];
      else if ((a == "--bindings" || a == "--shapes") && i + 1 < argc) bindings_path = argv[++i];
      else if (a == "-h" || a == "--help") {
        std::cout << "usage: intentir_cuda_codegen --intent <intent.json> --bindings <bindings.json>\\n";
        return 0;
      } else {
        fail("unknown arg: " + a);
      }
    }
    if (intent_path.empty() || bindings_path.empty()) {
      std::cerr << "missing --intent/--bindings (use --help)\\n";
      return 2;
    }

    Intent intent = parse_intent(read_json_file(intent_path));
    json bindings_json = read_json_file(bindings_path);
    json out = lower_intent_to_cuda(intent, bindings_json);
    std::cout << out.dump() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "intentir_cuda_codegen error: " << e.what() << "\\n";
    return 3;
  }
}
#endif
