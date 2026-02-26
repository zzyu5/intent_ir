#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ir_model.h"

namespace intentir_cuda_codegen {

using json = nlohmann::json;

std::optional<int64_t> resolve_dim_token(const json& tok, const json& bindings);
int64_t resolve_dim_token_required(const json& tok, const json& bindings, const std::string& where);
std::vector<int64_t> resolve_shape_required(const Tensor& t, const json& bindings, const std::string& where);
int parse_attr_int(const json& attrs, const char* key, int default_val, const json& bindings, const std::string& where);
std::pair<int, int> parse_attr_pair(const json& attrs, const char* key, int default_val, const json& bindings, const std::string& where);
std::vector<int> parse_attr_triple(const json& attrs, const char* key, int default_val, const json& bindings, const std::string& where);
int64_t resolve_schedule_int(const Intent& intent, const json& bindings, const char* key, int64_t default_val);
int64_t choose_1d_block_threads(const Intent& intent, const json& bindings, int64_t total_elems, int64_t default_threads,
                                int64_t elems_per_thread = 1, bool promote_tiny_hint = true);
bool is_scalar_tensor(const Intent& intent, const std::string& name, const std::string& dtype);
json tensor_io_spec(const Intent& intent, const std::string& name);
json io_spec_from_args(const Intent& intent, const std::vector<std::string>& tensor_args,
                       const std::unordered_map<std::string, std::string>& scalar_args,
                       const std::vector<std::string>& arg_names);

}  // namespace intentir_cuda_codegen
