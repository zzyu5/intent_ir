#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "ir_model.h"

namespace intentir_rvv_codegen {

using json = nlohmann::json;

std::optional<int64_t> resolve_dim_token(const json& tok, const std::unordered_map<std::string, int64_t>& bindings);
std::optional<int> match_axis_1d_to_tensor(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                           const std::string& vec_name, const std::string& tensor_name, int tensor_rank);
std::vector<int64_t> pad_for_broadcast(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                       const std::string& name, const std::vector<int64_t>& shape,
                                       const std::string& other_name, const std::vector<int64_t>& other_shape, int out_rank);
std::vector<int64_t> broadcast_shape_named(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                           const std::string& a_name, const std::string& b_name,
                                           const std::vector<int64_t>& a, const std::vector<int64_t>& b);

}  // namespace intentir_rvv_codegen
