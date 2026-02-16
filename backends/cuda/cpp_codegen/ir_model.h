#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace intentir_cuda_codegen {

using json = nlohmann::json;

struct Tensor {
  std::string dtype;
  std::vector<json> shape;  // dims are int or string (symbol)
};

struct Op {
  std::string op;
  std::vector<std::string> inputs;
  std::string output;
  json attrs;
};

struct Intent {
  std::string name;
  std::unordered_map<std::string, Tensor> tensors;
  std::vector<Op> ops;
  std::vector<std::string> outputs;
  json schedule;  // optional ScheduleSketch (tile/vec hints), may be null/object
  json meta;      // optional metadata (evidence / frontend hints), may be null/object
};

struct AccessWitnessMeta {
  std::string dominant_axis;
  int64_t dominant_range_len = 0;
  bool has_contiguous_range = false;
  std::unordered_map<std::string, int64_t> axis_contig_len;  // e.g., {"M":64,"N":16}
};

std::optional<AccessWitnessMeta> access_witness_meta(const Intent& intent);
Intent parse_intent(const json& j);
std::vector<int64_t> schedule_hints_tile_hints(const Intent& intent);
std::optional<std::string> contract_level_v2(const Intent& intent);
bool bindings_match_canonical_shapes(const Intent& intent, const json& bindings);
bool want_specialize_dims(const Intent& intent, const json& bindings);
bool want_host_dispatch(const json& bindings);
bool want_host_dispatch_select(const json& bindings);
int contract_level_code(const Intent& intent);
bool intent_has_evidence(const Intent& intent);

}  // namespace intentir_cuda_codegen

