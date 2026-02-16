#include "ir_model.h"

#include <algorithm>
#include <string>
#include <utility>

#include "common_utils.h"

namespace intentir_cuda_codegen {

std::optional<AccessWitnessMeta> access_witness_meta(const Intent& intent) {
  if (!intent.meta.is_object()) return std::nullopt;
  auto it = intent.meta.find("access_witness");
  if (it == intent.meta.end() || !it->is_object()) return std::nullopt;
  const json& aw = *it;

  AccessWitnessMeta out;
  auto ax = aw.find("dominant_axis");
  if (ax != aw.end() && ax->is_string()) out.dominant_axis = ax->get<std::string>();
  auto ln = aw.find("dominant_range_len");
  if (ln != aw.end()) {
    if (ln->is_number_integer())
      out.dominant_range_len = ln->get<int64_t>();
    else if (ln->is_number())
      out.dominant_range_len = static_cast<int64_t>(ln->get<double>());
  }
  auto cr = aw.find("has_contiguous_range");
  if (cr != aw.end()) {
    if (cr->is_boolean()) out.has_contiguous_range = cr->get<bool>();
    else if (cr->is_number_integer())
      out.has_contiguous_range = (cr->get<int64_t>() != 0);
    else if (cr->is_number())
      out.has_contiguous_range = (cr->get<double>() != 0.0);
  }

  auto acl = aw.find("axis_contig_len");
  if (acl != aw.end() && acl->is_object()) {
    for (auto kv = acl->begin(); kv != acl->end(); ++kv) {
      const std::string axis = kv.key();
      const json& v = kv.value();
      std::optional<int64_t> len;
      if (v.is_number_integer())
        len = v.get<int64_t>();
      else if (v.is_number())
        len = static_cast<int64_t>(v.get<double>());
      else if (v.is_string()) {
        try {
          len = std::stoll(v.get<std::string>());
        } catch (...) {
          len = std::nullopt;
        }
      }
      if (!len.has_value()) continue;
      if (*len <= 0) continue;
      out.axis_contig_len[axis] = *len;
    }
  }

  if (out.dominant_axis.empty() && out.dominant_range_len <= 0 && !out.has_contiguous_range && out.axis_contig_len.empty())
    return std::nullopt;
  return out;
}

Intent parse_intent(const json& j) {
  if (!j.is_object()) fail("intent must be object");
  Intent out;
  out.name = j.value("name", "intent_fn");

  json tensors_json = j.value("tensors", json::object());
  if (!tensors_json.is_object()) fail("intent.tensors must be object");
  for (auto it = tensors_json.begin(); it != tensors_json.end(); ++it) {
    const std::string name = it.key();
    const json& tj = it.value();
    if (!tj.is_object()) fail("tensor must be object: " + name);
    Tensor t;
    t.dtype = tj.value("dtype", "f32");
    json shape_json = tj.value("shape", json::array());
    if (!shape_json.is_array()) fail("tensor.shape must be list: " + name);
    for (const auto& d : shape_json) t.shape.push_back(d);
    out.tensors.emplace(name, std::move(t));
  }

  json ops_json = j.value("ops", json::array());
  if (!ops_json.is_array()) fail("intent.ops must be list");
  for (const auto& oj : ops_json) {
    if (!oj.is_object()) fail("op must be object");
    Op o;
    o.op = oj.value("op", "");
    o.output = oj.value("output", "");
    o.attrs = oj.value("attrs", json::object());
    json inputs_json = oj.value("inputs", json::array());
    if (!inputs_json.is_array()) fail("op.inputs must be list");
    for (const auto& x : inputs_json) o.inputs.push_back(x.get<std::string>());
    out.ops.push_back(std::move(o));
  }

  json outs_json = j.value("outputs", json::array());
  if (!outs_json.is_array()) fail("intent.outputs must be list");
  for (const auto& x : outs_json) out.outputs.push_back(x.get<std::string>());

  out.schedule = j.value("schedule", json());
  out.meta = j.value("meta", json::object());
  return out;
}

std::vector<int64_t> schedule_hints_tile_hints(const Intent& intent) {
  std::vector<int64_t> out;
  if (!intent.meta.is_object()) return out;
  auto it = intent.meta.find("schedule_hints_v2");
  if (it == intent.meta.end() || !it->is_object()) return out;
  const json& sh = *it;
  auto th = sh.find("tile_hints");
  if (th == sh.end() || !th->is_array()) return out;
  for (const auto& x : *th) {
    auto v = json_int64(x);
    if (!v.has_value()) continue;
    if (*v <= 0) continue;
    out.push_back(*v);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

std::optional<std::string> contract_level_v2(const Intent& intent) {
  if (!intent.meta.is_object()) return std::nullopt;
  auto it = intent.meta.find("contract_v2");
  if (it == intent.meta.end() || !it->is_object()) return std::nullopt;
  const json& c = *it;
  auto lv = c.find("level");
  if (lv != c.end() && lv->is_string()) return lv->get<std::string>();
  return std::nullopt;
}

bool bindings_match_canonical_shapes(const Intent& intent, const json& bindings) {
  if (!intent.meta.is_object() || !bindings.is_object()) return false;
  auto it = intent.meta.find("canonical_shapes");
  if (it == intent.meta.end() || !it->is_object()) return false;
  const json& cs = *it;
  bool has_any = false;
  for (auto kv = cs.begin(); kv != cs.end(); ++kv) {
    has_any = true;
    const std::string key = kv.key();
    auto canon = json_int64(kv.value());
    if (!canon.has_value()) return false;
    auto actual = binding_int(bindings, key);
    if (!actual.has_value() || *actual != *canon) return false;
  }
  return has_any;
}

bool want_specialize_dims(const Intent& intent, const json& bindings) {
  if (binding_int(bindings, "CUDA_SPECIALIZE_DIMS").value_or(0) != 0) return true;
  if (binding_int(bindings, "CUDA_AUTO_SPECIALIZE_DIMS").value_or(1) == 0) return false;
  if (!bindings_match_canonical_shapes(intent, bindings)) return false;
  if (contract_level_v2(intent).value_or("PARTIAL") == "OUT_OF_SCOPE") return false;
  return true;
}

bool want_host_dispatch(const json& bindings) {
  if (binding_int(bindings, "CUDA_DISABLE_HOST_DISPATCH").value_or(0) != 0) return false;
  if (binding_int(bindings, "CUDA_HOST_DISPATCH").value_or(1) == 0) return false;
  return true;
}

bool want_host_dispatch_select(const json& bindings) {
  if (binding_int(bindings, "CUDA_HOST_DISPATCH_SELECT").value_or(1) == 0) return false;
  return true;
}

int contract_level_code(const Intent& intent) {
  const std::string lv = contract_level_v2(intent).value_or("PARTIAL");
  if (lv == "OUT_OF_SCOPE") return 0;
  if (lv == "FULL") return 2;
  return 1;  // PARTIAL/unknown
}

bool intent_has_evidence(const Intent& intent) {
  if (access_witness_meta(intent).has_value()) return true;
  if (!schedule_hints_tile_hints(intent).empty()) return true;
  return false;
}

}  // namespace intentir_cuda_codegen

