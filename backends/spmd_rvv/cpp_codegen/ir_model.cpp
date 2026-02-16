#include "ir_model.h"

#include <string>
#include <utility>

#include "common_utils.h"

namespace intentir_rvv_codegen {

Intent parse_intent(const json& j) {
  Intent out;
  out.name = j.value("name", "intent_fn");
  if (!j.contains("tensors") || !j["tensors"].is_object()) fail("intent.tensors must be object");
  for (auto it = j["tensors"].begin(); it != j["tensors"].end(); ++it) {
    const std::string name = it.key();
    const json& tj = it.value();
    Tensor t;
    t.dtype = tj.value("dtype", "f32");
    if (!tj.contains("shape") || !tj["shape"].is_array()) fail("tensor.shape must be list");
    for (const auto& d : tj["shape"]) t.shape.push_back(d);
    out.tensors.emplace(name, std::move(t));
  }
  if (!j.contains("ops") || !j["ops"].is_array()) fail("intent.ops must be list");
  for (const auto& oj : j["ops"]) {
    Op op;
    op.op = oj.value("op", "");
    if (!oj.contains("inputs") || !oj["inputs"].is_array()) fail("op.inputs must be list");
    for (const auto& x : oj["inputs"]) op.inputs.push_back(x.get<std::string>());
    op.output = oj.value("output", "");
    op.attrs = oj.value("attrs", json::object());
    out.ops.push_back(std::move(op));
  }
  if (!j.contains("outputs") || !j["outputs"].is_array()) fail("intent.outputs must be list");
  for (const auto& x : j["outputs"]) out.outputs.push_back(x.get<std::string>());
  if (j.contains("schedule")) out.schedule = j["schedule"];
  return out;
}

}  // namespace intentir_rvv_codegen
