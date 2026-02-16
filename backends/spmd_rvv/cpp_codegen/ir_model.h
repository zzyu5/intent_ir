#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace intentir_rvv_codegen {

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
};

Intent parse_intent(const json& j);

}  // namespace intentir_rvv_codegen
