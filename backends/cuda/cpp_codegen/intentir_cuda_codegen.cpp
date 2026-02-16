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

#include "emit_dropout.inc"
#include "emit_remaining_kernels_wave3.inc"
#include "dispatch_lowering.inc"

}  // namespace

#include "entry_pybind.inc"
#include "entry_main.inc"
