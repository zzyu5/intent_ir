#pragma once

#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

class CodeWriter {
 public:
  explicit CodeWriter(std::ostream& os, int indent_width = 2) : os_(os), indent_width_(indent_width) {}

  void indent() { indent_ += 1; }
  void dedent() {
    if (indent_ <= 0) throw std::runtime_error("CodeWriter: dedent underflow");
    indent_ -= 1;
  }

  void blank() { os_ << '\n'; }

  void pp_line(std::string_view s) { os_ << s << '\n'; }

  void line(std::string_view s) {
    // Keep preprocessor directives at column 0 even inside blocks.
    for (char c : s) {
      if (c == ' ' || c == '\t') continue;
      if (c == '#') {
        pp_line(s);
        return;
      }
      break;
    }
    for (int i = 0; i < indent_ * indent_width_; ++i) os_ << ' ';
    os_ << s << '\n';
  }

 private:
  std::ostream& os_;
  int indent_width_ = 2;
  int indent_ = 0;
};

