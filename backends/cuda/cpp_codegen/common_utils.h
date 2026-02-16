#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

[[noreturn]] inline void fail(const std::string& msg) { throw std::runtime_error(msg); }

inline std::string c_float(double x) {
  if (std::isnan(x)) return "NAN";
  if (std::isinf(x)) return x > 0 ? "INFINITY" : "(-INFINITY)";
  std::ostringstream oss;
  oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
  oss.precision(10);
  oss << x;
  std::string s = oss.str();
  const bool has_dot =
      (s.find('.') != std::string::npos) || (s.find('e') != std::string::npos) || (s.find('E') != std::string::npos);
  if (!has_dot) s += ".0";
  return s + "f";
}

inline std::string ascii_lower(std::string s) {
  for (auto& ch : s) {
    if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
  }
  return s;
}

inline nlohmann::json read_json_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) fail("failed to open: " + path);
  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    fail("failed to parse json: " + path + " (" + e.what() + ")");
  }
  return j;
}

inline bool is_digits(const std::string& s) {
  if (s.empty()) return false;
  for (unsigned char c : s) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

inline std::optional<double> binding_double(const nlohmann::json& bindings, const std::string& key) {
  if (!bindings.is_object()) return std::nullopt;
  auto it = bindings.find(key);
  if (it == bindings.end()) return std::nullopt;
  if (it->is_number()) return it->get<double>();
  if (it->is_number_integer()) return static_cast<double>(it->get<int64_t>());
  if (it->is_string()) {
    try {
      return std::stod(it->get<std::string>());
    } catch (...) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

inline std::string dim_str(const nlohmann::json& tok) {
  if (tok.is_string()) return tok.get<std::string>();
  if (tok.is_number_integer()) return std::to_string(tok.get<int64_t>());
  if (tok.is_number()) return std::to_string(static_cast<int64_t>(tok.get<double>()));
  return tok.dump();
}

inline std::string c_ident(std::string_view name) {
  std::string out;
  out.reserve(name.size() + 1);
  for (char ch : name) {
    const bool ok =
        (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || (ch == '_');
    out.push_back(ok ? ch : '_');
  }
  if (out.empty()) return "v";
  if ((out[0] >= '0' && out[0] <= '9') || out[0] == '_') out.insert(out.begin(), 'v');
  return out;
}

inline std::string c_type_for_dtype(const std::string& dt) {
  if (dt == "f32") return "float";
  if (dt == "f16") return "__half";
  if (dt == "bf16") return "__nv_bfloat16";
  if (dt == "i32") return "int";
  if (dt == "i64") return "int64_t";
  if (dt == "u8") return "uint8_t";
  if (dt == "i8") return "int8_t";
  if (dt == "i16") return "int16_t";
  if (dt == "bool" || dt == "i1") return "bool";
  fail("unsupported dtype for CUDA elementwise: " + dt);
}

inline std::string c_scalar_literal(const std::string& dt, const nlohmann::json& value) {
  if (dt == "bool" || dt == "i1") {
    int64_t v = 0;
    if (value.is_boolean())
      v = value.get<bool>() ? 1 : 0;
    else if (value.is_number_integer())
      v = value.get<int64_t>();
    else if (value.is_number())
      v = static_cast<int64_t>(value.get<double>());
    else if (value.is_string()) {
      try {
        v = static_cast<int64_t>(std::stod(value.get<std::string>()));
      } catch (...) {
        v = 0;
      }
    }
    return (v != 0) ? "true" : "false";
  }
  if (dt == "i32") {
    int64_t v = 0;
    if (value.is_number_integer())
      v = value.get<int64_t>();
    else if (value.is_number())
      v = static_cast<int64_t>(value.get<double>());
    else if (value.is_string()) {
      try {
        v = static_cast<int64_t>(std::stod(value.get<std::string>()));
      } catch (...) {
        v = 0;
      }
    }
    return std::to_string(static_cast<int>(v));
  }
  if (dt == "i64") {
    int64_t v = 0;
    if (value.is_number_integer())
      v = value.get<int64_t>();
    else if (value.is_number())
      v = static_cast<int64_t>(value.get<double>());
    else if (value.is_string()) {
      try {
        v = static_cast<int64_t>(std::stod(value.get<std::string>()));
      } catch (...) {
        v = 0;
      }
    }
    return std::to_string(static_cast<long long>(v)) + "LL";
  }
  if (dt == "f32") {
    double v = 0.0;
    if (value.is_number())
      v = value.get<double>();
    else if (value.is_number_integer())
      v = static_cast<double>(value.get<int64_t>());
    else if (value.is_string()) {
      try {
        v = std::stod(value.get<std::string>());
      } catch (...) {
        v = 0.0;
      }
    }
    return c_float(v);
  }
  if (dt == "f16") {
    double v = 0.0;
    if (value.is_number())
      v = value.get<double>();
    else if (value.is_number_integer())
      v = static_cast<double>(value.get<int64_t>());
    else if (value.is_string()) {
      try {
        v = std::stod(value.get<std::string>());
      } catch (...) {
        v = 0.0;
      }
    }
    return "__float2half(" + c_float(v) + ")";
  }
  if (dt == "bf16") {
    double v = 0.0;
    if (value.is_number())
      v = value.get<double>();
    else if (value.is_number_integer())
      v = static_cast<double>(value.get<int64_t>());
    else if (value.is_string()) {
      try {
        v = std::stod(value.get<std::string>());
      } catch (...) {
        v = 0.0;
      }
    }
    return "__float2bfloat16(" + c_float(v) + ")";
  }
  fail("unsupported const dtype for CUDA elementwise: " + dt);
}

inline std::optional<int64_t> binding_int(const nlohmann::json& bindings, const std::string& key) {
  if (!bindings.is_object()) return std::nullopt;
  auto it = bindings.find(key);
  if (it == bindings.end()) return std::nullopt;
  if (it->is_number_integer()) return it->get<int64_t>();
  if (it->is_number()) return static_cast<int64_t>(it->get<double>());
  if (it->is_string()) {
    std::string s = it->get<std::string>();
    if (is_digits(s)) return std::stoll(s);
  }
  return std::nullopt;
}

inline std::optional<int64_t> json_int64(const nlohmann::json& v) {
  if (v.is_number_integer()) return v.get<int64_t>();
  if (v.is_number()) return static_cast<int64_t>(v.get<double>());
  if (v.is_string()) {
    const std::string s = v.get<std::string>();
    if (is_digits(s)) return std::stoll(s);
  }
  return std::nullopt;
}
