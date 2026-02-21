#include "shape_eval.h"

#include <algorithm>

#include "common_utils.h"

namespace intentir_rvv_codegen {

std::optional<int64_t> resolve_dim_token(const json& tok, const std::unordered_map<std::string, int64_t>& bindings) {
  if (tok.is_number_integer()) return tok.get<int64_t>();
  if (tok.is_number()) return static_cast<int64_t>(tok.get<double>());
  if (tok.is_string()) {
    std::string s = tok.get<std::string>();
    if (is_digits(s)) return std::stoll(s);
    auto it = bindings.find(s);
    if (it != bindings.end()) return it->second;
  }
  return std::nullopt;
}

std::optional<int> match_axis_1d_to_tensor(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                           const std::string& vec_name, const std::string& tensor_name, int tensor_rank) {
  auto vit = intent.tensors.find(vec_name);
  auto tit = intent.tensors.find(tensor_name);
  if (vit == intent.tensors.end() || tit == intent.tensors.end()) return std::nullopt;
  const auto& vshape = vit->second.shape;
  const auto& tshape = tit->second.shape;
  if (vshape.size() != 1 || static_cast<int>(tshape.size()) != tensor_rank) return std::nullopt;

  const json& v0 = vshape[0];
  if (v0.is_string()) {
    std::string sym = v0.get<std::string>();
    if (!is_digits(sym)) {
      std::vector<int> hits;
      for (int i = 0; i < tensor_rank; ++i) {
        if (tshape[i].is_string() && tshape[i].get<std::string>() == sym) hits.push_back(i);
      }
      if (hits.size() == 1) return hits[0];
    }
  }

  auto vnum = resolve_dim_token(v0, bindings);
  if (!vnum) return std::nullopt;
  std::vector<int> hits;
  for (int i = 0; i < tensor_rank; ++i) {
    auto tnum = resolve_dim_token(tshape[i], bindings);
    if (tnum && *tnum == *vnum) hits.push_back(i);
  }
  if (hits.size() == 1) return hits[0];
  return std::nullopt;
}

std::optional<std::vector<int>> match_axes_to_tensor(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                                     const std::string& name, const std::string& tensor_name, int tensor_rank) {
  auto sit = intent.tensors.find(name);
  auto tit = intent.tensors.find(tensor_name);
  if (sit == intent.tensors.end() || tit == intent.tensors.end()) return std::nullopt;
  const auto& sshape = sit->second.shape;
  const auto& tshape = tit->second.shape;
  const int sr = static_cast<int>(sshape.size());
  if (sr <= 0 || sr >= tensor_rank || static_cast<int>(tshape.size()) != tensor_rank) return std::nullopt;

  std::vector<int> axes;
  axes.reserve(static_cast<size_t>(sr));
  std::vector<int> used(static_cast<size_t>(tensor_rank), 0);
  bool matched_by_symbol = true;

  for (int i = 0; i < sr; ++i) {
    const json& stok = sshape[static_cast<size_t>(i)];
    if (!(stok.is_string() && !is_digits(stok.get<std::string>()))) {
      matched_by_symbol = false;
      break;
    }
    const std::string sym = stok.get<std::string>();
    std::vector<int> hits;
    for (int ax = 0; ax < tensor_rank; ++ax) {
      if (used[static_cast<size_t>(ax)] == 1) continue;
      if (tshape[static_cast<size_t>(ax)].is_string() && tshape[static_cast<size_t>(ax)].get<std::string>() == sym) {
        hits.push_back(ax);
      }
    }
    if (hits.size() != 1) {
      matched_by_symbol = false;
      break;
    }
    axes.push_back(hits[0]);
    used[static_cast<size_t>(hits[0])] = 1;
  }

  auto strictly_increasing = [](const std::vector<int>& v) -> bool {
    for (size_t i = 1; i < v.size(); ++i) {
      if (v[i] <= v[i - 1]) return false;
    }
    return true;
  };

  if (matched_by_symbol && static_cast<int>(axes.size()) == sr && strictly_increasing(axes)) return axes;

  axes.clear();
  std::fill(used.begin(), used.end(), 0);
  for (int i = 0; i < sr; ++i) {
    auto sval = resolve_dim_token(sshape[static_cast<size_t>(i)], bindings);
    if (!sval) return std::nullopt;
    std::vector<int> hits;
    for (int ax = 0; ax < tensor_rank; ++ax) {
      if (used[static_cast<size_t>(ax)] == 1) continue;
      auto tval = resolve_dim_token(tshape[static_cast<size_t>(ax)], bindings);
      if (tval && *tval == *sval) hits.push_back(ax);
    }
    if (hits.size() != 1) return std::nullopt;
    axes.push_back(hits[0]);
    used[static_cast<size_t>(hits[0])] = 1;
  }
  if (strictly_increasing(axes)) return axes;
  return std::nullopt;
}

std::vector<int64_t> pad_for_broadcast(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                       const std::string& name, const std::vector<int64_t>& shape,
                                       const std::string& other_name, const std::vector<int64_t>& other_shape, int out_rank) {
  std::vector<int64_t> padded(out_rank, 1);
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) padded[out_rank - static_cast<int>(shape.size()) + i] = shape[i];

  if (static_cast<int>(shape.size()) >= 2 && static_cast<int>(shape.size()) < out_rank && static_cast<int>(other_shape.size()) == out_rank) {
    if (auto axes = match_axes_to_tensor(intent, bindings, name, other_name, out_rank)) {
      std::vector<int64_t> named(out_rank, 1);
      for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
        const int ax = (*axes)[static_cast<size_t>(i)];
        if (ax >= 0 && ax < out_rank) named[ax] = shape[static_cast<size_t>(i)];
      }
      return named;
    }
  }

  if (static_cast<int>(shape.size()) == 1 && out_rank >= 2 && static_cast<int>(other_shape.size()) == out_rank) {
    if (auto ax = match_axis_1d_to_tensor(intent, bindings, name, other_name, out_rank)) {
      if (*ax >= 0 && *ax < out_rank) {
        std::vector<int64_t> named(out_rank, 1);
        named[*ax] = shape[0];
        return named;
      }
    }
    std::vector<int> hits;
    for (int ax = 0; ax < out_rank; ++ax) {
      if (other_shape[ax] == shape[0]) hits.push_back(ax);
    }
    if (hits.size() == 1) {
      std::vector<int64_t> named(out_rank, 1);
      named[hits[0]] = shape[0];
      return named;
    }
  }
  return padded;
}

std::vector<int64_t> broadcast_shape_named(const Intent& intent, const std::unordered_map<std::string, int64_t>& bindings,
                                           const std::string& a_name, const std::string& b_name,
                                           const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const int ra = static_cast<int>(a.size());
  const int rb = static_cast<int>(b.size());
  const int r = std::max(ra, rb);
  std::vector<int64_t> pa = pad_for_broadcast(intent, bindings, a_name, a, b_name, b, r);
  std::vector<int64_t> pb = pad_for_broadcast(intent, bindings, b_name, b, a_name, a, r);
  std::vector<int64_t> out;
  out.reserve(r);
  for (int i = 0; i < r; ++i) {
    const int64_t da = pa[i], db = pb[i];
    if (da == db) out.push_back(da);
    else if (da == 1) out.push_back(db);
    else if (db == 1) out.push_back(da);
    else fail("broadcast shape mismatch");
  }
  return out;
}

}  // namespace intentir_rvv_codegen
