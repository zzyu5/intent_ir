// ---- small expression evaluator for const values (supports +,-,*,/,(), symbols, numbers) ----

enum class TokKind { End, Number, Ident, Plus, Minus, Star, Slash, Pow, LParen, RParen };

struct Tok {
  TokKind kind;
  double num = 0.0;
  std::string ident;
};

struct Lexer {
  explicit Lexer(std::string s) : src(std::move(s)) {}
  std::string src;
  size_t i = 0;

  void skip_ws() {
    while (i < src.size() && (src[i] == ' ' || src[i] == '\t' || src[i] == '\n' || src[i] == '\r')) ++i;
  }

  Tok next() {
    skip_ws();
    if (i >= src.size()) return {TokKind::End};
    char c = src[i];
    if (c == '+') { ++i; return {TokKind::Plus}; }
    if (c == '-') { ++i; return {TokKind::Minus}; }
    if (c == '*') {
      // exponentiation: '**'
      if (i + 1 < src.size() && src[i + 1] == '*') { i += 2; return {TokKind::Pow}; }
      ++i;
      return {TokKind::Star};
    }
    if (c == '/') { ++i; return {TokKind::Slash}; }
    if (c == '(') { ++i; return {TokKind::LParen}; }
    if (c == ')') { ++i; return {TokKind::RParen}; }
    if ((c >= '0' && c <= '9') || c == '.') {
      size_t start = i;
      while (i < src.size()) {
        char d = src[i];
        if ((d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') {
          // stop '+'/'-' unless part of exponent; crude but ok for our use
          if ((d == '+' || d == '-') && i > start && (src[i-1] != 'e' && src[i-1] != 'E')) break;
          ++i;
          continue;
        }
        break;
      }
      double v = std::stod(src.substr(start, i - start));
      Tok t{TokKind::Number};
      t.num = v;
      return t;
    }
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_') {
      size_t start = i;
      ++i;
      while (i < src.size()) {
        char d = src[i];
        if ((d >= 'A' && d <= 'Z') || (d >= 'a' && d <= 'z') || (d >= '0' && d <= '9') || d == '_') ++i;
        else break;
      }
      Tok t{TokKind::Ident};
      t.ident = src.substr(start, i - start);
      return t;
    }
    fail(std::string("unexpected char in expr: ") + c);
  }
};

struct Parser {
  explicit Parser(std::string s, const std::unordered_map<std::string, double>& vars) : lex(std::move(s)), vars(vars) {
    cur = lex.next();
  }
  Lexer lex;
  Tok cur;
  const std::unordered_map<std::string, double>& vars;

  void eat(TokKind k) {
    if (cur.kind != k) fail("expr parse error");
    cur = lex.next();
  }

  double parse_expr() {
    double v = parse_term();
    while (cur.kind == TokKind::Plus || cur.kind == TokKind::Minus) {
      TokKind k = cur.kind;
      eat(k);
      double rhs = parse_term();
      v = (k == TokKind::Plus) ? (v + rhs) : (v - rhs);
    }
    return v;
  }

  double parse_term() {
    double v = parse_factor();
    while (cur.kind == TokKind::Star || cur.kind == TokKind::Slash) {
      TokKind k = cur.kind;
      eat(k);
      double rhs = parse_factor();
      v = (k == TokKind::Star) ? (v * rhs) : (v / rhs);
    }
    return v;
  }

  double parse_factor() {
    if (cur.kind == TokKind::Plus) { eat(TokKind::Plus); return parse_factor(); }
    if (cur.kind == TokKind::Minus) { eat(TokKind::Minus); return -parse_factor(); }
    return parse_pow();
  }

  double parse_pow() {
    double v = parse_atom();
    if (cur.kind == TokKind::Pow) {
      eat(TokKind::Pow);
      double rhs = parse_factor();  // right-associative; exponent can be unary
      v = std::pow(v, rhs);
    }
    return v;
  }

  double parse_atom() {
    if (cur.kind == TokKind::Number) {
      double v = cur.num;
      eat(TokKind::Number);
      return v;
    }
    if (cur.kind == TokKind::Ident) {
      std::string name = cur.ident;
      eat(TokKind::Ident);
      auto it = vars.find(name);
      if (it != vars.end()) return it->second;
      if (name == "eps") return 1e-5;
      fail("unresolved symbol in const expr: " + name);
    }
    if (cur.kind == TokKind::LParen) {
      eat(TokKind::LParen);
      double v = parse_expr();
      eat(TokKind::RParen);
      return v;
    }
    fail("invalid expr");
  }
};

double eval_expr(const std::string& expr, const std::unordered_map<std::string, double>& vars) {
  Parser p(expr, vars);
  double v = p.parse_expr();
  if (p.cur.kind != TokKind::End) fail("trailing tokens in expr");
  return v;
}

double resolve_const_value(const json& v, const std::unordered_map<std::string, int64_t>& bindings) {
  if (v.is_number()) return v.get<double>();
  if (v.is_string()) {
    std::string s = v.get<std::string>();
    if (s == "eps") return 1e-5;
    // numeric literal?
    try {
      size_t pos = 0;
      double x = std::stod(s, &pos);
      if (pos == s.size()) return x;
    } catch (...) {
    }
    // symbol binding?
    auto it = bindings.find(s);
    if (it != bindings.end()) return static_cast<double>(it->second);
    // expression
    std::unordered_map<std::string, double> vars;
    for (auto& kv : bindings) vars.emplace(kv.first, static_cast<double>(kv.second));
    // Some LLM/providers may emit Python-style integer division ("//").
    // Support it by rewriting to "/" and flooring the final result.
    if (s.find("//") != std::string::npos) {
      std::string t = s;
      size_t pos = 0;
      while ((pos = t.find("//", pos)) != std::string::npos) {
        t.replace(pos, 2, "/");
        pos += 1;
      }
      return std::floor(eval_expr(t, vars));
    }
    return eval_expr(s, vars);
  }
  fail("unsupported const value type");
}

