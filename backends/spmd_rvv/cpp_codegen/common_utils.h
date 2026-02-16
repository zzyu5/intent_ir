#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
  bool has_dot = (s.find('.') != std::string::npos) || (s.find('e') != std::string::npos) ||
                 (s.find('E') != std::string::npos);
  if (!has_dot) s += ".0";
  return s + "f";
}

inline std::string ctype_for_dtype(const std::string& dt) {
  if (dt == "bool" || dt == "i1" || dt == "u8") return "uint8_t";
  if (dt == "i8") return "int8_t";
  if (dt == "i16") return "int16_t";
  if (dt == "i32") return "int32_t";
  if (dt == "i64") return "int64_t";
  if (dt == "f64") return "double";
  return "float";
}

inline std::string typecode_for_dtype(const std::string& dt) {
  if (dt == "bool" || dt == "i1" || dt == "u8") return "INTENTIR_TYPE_U8";
  if (dt == "i8") return "INTENTIR_TYPE_I8";
  if (dt == "i16") return "INTENTIR_TYPE_I16";
  if (dt == "i32") return "INTENTIR_TYPE_I32";
  if (dt == "i64") return "INTENTIR_TYPE_I64";
  if (dt == "f64") return "INTENTIR_TYPE_F64";
  return "INTENTIR_TYPE_F32";
}

inline std::string flat_idx_expr(const std::vector<std::string>& vars, const std::vector<int64_t>& shape) {
  const int r = static_cast<int>(shape.size());
  if (r == 0) return "0";
  if (r == 1) return "(size_t)" + vars[0];
  if (r == 2) return "idx2(" + vars[0] + "," + vars[1] + "," + std::to_string(shape[1]) + ")";
  if (r == 3)
    return "idx3(" + vars[0] + "," + vars[1] + "," + vars[2] + "," + std::to_string(shape[1]) + "," +
           std::to_string(shape[2]) + ")";
  if (r == 4)
    return "idx4(" + vars[0] + "," + vars[1] + "," + vars[2] + "," + vars[3] + "," +
           std::to_string(shape[1]) + "," + std::to_string(shape[2]) + "," + std::to_string(shape[3]) + ")";
  fail("indexing supports rank<=4");
}

inline int64_t numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (auto d : shape) n *= static_cast<int64_t>(d);
  return n;
}

inline std::vector<int64_t> broadcast_shape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const int ra = static_cast<int>(a.size());
  const int rb = static_cast<int>(b.size());
  const int r = std::max(ra, rb);
  std::vector<int64_t> aa(r, 1), bb(r, 1);
  for (int i = 0; i < ra; ++i) aa[r - ra + i] = a[i];
  for (int i = 0; i < rb; ++i) bb[r - rb + i] = b[i];
  std::vector<int64_t> out;
  out.reserve(r);
  for (int i = 0; i < r; ++i) {
    const int64_t da = aa[i], db = bb[i];
    if (da == db)
      out.push_back(da);
    else if (da == 1)
      out.push_back(db);
    else if (db == 1)
      out.push_back(da);
    else
      fail("broadcast shape mismatch");
  }
  return out;
}

inline bool is_digits(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

inline std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) fail("cannot open file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

inline nlohmann::json read_json_file(const std::string& path) { return nlohmann::json::parse(read_text_file(path)); }
