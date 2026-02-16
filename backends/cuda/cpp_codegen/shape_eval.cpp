#include "shape_eval.h"

#include "common_utils.h"

namespace intentir_cuda_codegen {

std::optional<int64_t> resolve_dim_token(const json& tok, const json& bindings) {
  if (tok.is_number_integer()) return tok.get<int64_t>();
  if (tok.is_number()) return static_cast<int64_t>(tok.get<double>());
  if (tok.is_string()) {
    std::string s = tok.get<std::string>();
    if (is_digits(s)) return std::stoll(s);
    if (auto v = binding_int(bindings, s)) return v;
  }
  return std::nullopt;
}

int64_t resolve_dim_token_required(const json& tok, const json& bindings, const std::string& where) {
  auto v = resolve_dim_token(tok, bindings);
  if (!v.has_value()) fail(where + ": missing dim binding");
  return *v;
}

std::vector<int64_t> resolve_shape_required(const Tensor& t, const json& bindings, const std::string& where) {
  std::vector<int64_t> out;
  out.reserve(t.shape.size());
  for (const auto& d : t.shape) out.push_back(resolve_dim_token_required(d, bindings, where));
  return out;
}

int parse_attr_int(const json& attrs, const char* key, int default_val, const json& bindings, const std::string& where) {
  if (!attrs.is_object() || !attrs.contains(key)) return default_val;
  int64_t v = resolve_dim_token_required(attrs[key], bindings, where + "." + key);
  return static_cast<int>(v);
}

std::pair<int, int> parse_attr_pair(const json& attrs, const char* key, int default_val, const json& bindings, const std::string& where) {
  if (!attrs.is_object() || !attrs.contains(key)) return {default_val, default_val};
  const auto& value = attrs[key];
  if (value.is_array()) {
    if (value.size() != 2) fail(where + "." + key + " must have length 2");
    int a = static_cast<int>(resolve_dim_token_required(value[0], bindings, where + "." + key + "[0]"));
    int b = static_cast<int>(resolve_dim_token_required(value[1], bindings, where + "." + key + "[1]"));
    return {a, b};
  }
  int v = static_cast<int>(resolve_dim_token_required(value, bindings, where + "." + key));
  return {v, v};
}

std::vector<int> parse_attr_triple(const json& attrs, const char* key, int default_val, const json& bindings, const std::string& where) {
  if (!attrs.is_object() || !attrs.contains(key)) return {default_val, default_val, default_val};
  const auto& value = attrs[key];
  if (value.is_array()) {
    if (value.size() != 3) fail(where + "." + key + " must have length 3");
    return {
        static_cast<int>(resolve_dim_token_required(value[0], bindings, where + "." + key + "[0]")),
        static_cast<int>(resolve_dim_token_required(value[1], bindings, where + "." + key + "[1]")),
        static_cast<int>(resolve_dim_token_required(value[2], bindings, where + "." + key + "[2]")),
    };
  }
  int v = static_cast<int>(resolve_dim_token_required(value, bindings, where + "." + key));
  return {v, v, v};
}

int64_t resolve_schedule_int(const Intent& intent, const json& bindings, const char* key, int64_t default_val) {
  if (!intent.schedule.is_object()) return default_val;
  auto it = intent.schedule.find(key);
  if (it == intent.schedule.end()) return default_val;
  auto v = resolve_dim_token(*it, bindings);
  if (!v.has_value()) return default_val;
  return *v;
}

bool is_scalar_tensor(const Intent& intent, const std::string& name, const std::string& dtype) {
  auto it = intent.tensors.find(name);
  if (it == intent.tensors.end()) return false;
  if (!it->second.shape.empty()) return false;
  if (!dtype.empty() && it->second.dtype != dtype) return false;
  return true;
}

json tensor_io_spec(const Intent& intent, const std::string& name) {
  auto it = intent.tensors.find(name);
  if (it == intent.tensors.end()) fail("io_spec tensor missing from intent.tensors: " + name);
  json t;
  t["dtype"] = it->second.dtype;
  json shape = json::array();
  for (const auto& d : it->second.shape) shape.push_back(d);
  t["shape"] = shape;
  return t;
}

json io_spec_from_args(const Intent& intent, const std::vector<std::string>& tensor_args,
                       const std::unordered_map<std::string, std::string>& scalar_args,
                       const std::vector<std::string>& arg_names) {
  json tensors = json::object();
  for (const auto& n : tensor_args) tensors[n] = tensor_io_spec(intent, n);
  json scalars = json::object();
  for (const auto& kv : scalar_args) scalars[kv.first] = kv.second;
  json io_spec;
  io_spec["arg_names"] = arg_names;
  io_spec["tensors"] = tensors;
  io_spec["scalars"] = scalars;
  return io_spec;
}

}  // namespace intentir_cuda_codegen

