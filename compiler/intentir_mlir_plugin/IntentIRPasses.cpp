#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

static std::string sanitizeSymbolName(llvm::StringRef raw) {
  std::string out;
  out.reserve(raw.size());
  for (char c : raw) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
      out.push_back(c);
      continue;
    }
    out.push_back('_');
  }
  if (!out.empty() && std::isdigit(static_cast<unsigned char>(out[0]))) {
    out.insert(out.begin(), '_');
  }
  if (out.empty())
    out = "intent_fn";
  return out;
}

static mlir::FailureOr<std::string> getRequiredStringAttr(mlir::ModuleOp module,
                                                          llvm::StringRef key) {
  auto attr = module->getAttrOfType<mlir::StringAttr>(key);
  if (!attr) {
    return mlir::failure();
  }
  return attr.str();
}

static mlir::FailureOr<std::string> decodeB64(llvm::StringRef b64) {
  std::vector<char> decoded;
  decoded.reserve(b64.size());
  if (llvm::Error err = llvm::decodeBase64(b64, decoded)) {
    llvm::consumeError(std::move(err));
    return mlir::failure();
  }
  return std::string(decoded.begin(), decoded.end());
}

static mlir::FailureOr<llvm::json::Value> parseJson(llvm::StringRef text) {
  auto parsed = llvm::json::parse(text);
  if (!parsed) {
    return mlir::failure();
  }
  return std::move(*parsed);
}

static mlir::FailureOr<std::map<std::string, int64_t>>
parseShapeBindings(const llvm::json::Value &val) {
  const auto *obj = val.getAsObject();
  if (!obj) {
    return mlir::failure();
  }
  std::map<std::string, int64_t> out;
  for (const auto &kv : *obj) {
    auto key = kv.first.str();
    const auto &vv = kv.second;
    auto i = vv.getAsInteger();
    if (!i) {
      // Only accept ints for bindings; others are ignored.
      continue;
    }
    out.emplace(std::move(key), static_cast<int64_t>(*i));
  }
  return out;
}

static std::string normalizeCudaArchForTuning(llvm::StringRef raw) {
  std::string s = raw.trim().lower();
  if (s.empty())
    return "";

  bool allDigits = true;
  for (char c : s) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      allDigits = false;
      break;
    }
  }
  if (allDigits) {
    return std::string("sm") + s;
  }

  llvm::StringRef sr(s);
  if (sr.starts_with("sm_")) {
    s = s.substr(3);
  } else if (sr.starts_with("sm")) {
    s = s.substr(2);
  } else {
    return "";
  }
  std::string digits;
  for (char c : s) {
    if (std::isdigit(static_cast<unsigned char>(c)))
      digits.push_back(c);
  }
  if (digits.empty())
    return "";
  return std::string("sm") + digits;
}

static std::string detectCudaArchForTuning() {
  const char *raw = std::getenv("INTENTIR_CUDA_SM");
  if (!raw || !*raw)
    return "";
  return normalizeCudaArchForTuning(raw);
}

static bool stackMatchesForCppPlugin(const llvm::json::Object &row) {
  const llvm::json::Value *v = row.get("compiler_stack");
  if (!v)
    v = row.get("stack");
  if (!v) {
    // Unspecified => apply to all (backward compatible).
    return true;
  }

  auto normalize = [](llvm::StringRef s) -> std::string {
    std::string out = s.trim().lower();
    if (out == "cpp" || out == "c++")
      out = "cpp_plugin";
    return out;
  };

  std::set<std::string> allowed;
  if (auto s = v->getAsString()) {
    auto n = normalize(*s);
    if (!n.empty())
      allowed.insert(std::move(n));
  } else if (auto arr = v->getAsArray()) {
    for (const auto &it : *arr) {
      auto s = it.getAsString();
      if (!s)
        continue;
      auto n = normalize(*s);
      if (!n.empty())
        allowed.insert(std::move(n));
    }
  } else {
    // Unknown type => do not match.
    return false;
  }
  if (allowed.empty())
    return true;
  return allowed.count("cpp_plugin") != 0;
}

static bool matchWhen(const llvm::json::Object &when,
                      const std::map<std::string, int64_t> &shapeBindings) {
  for (const auto &kv : when) {
    const std::string key = kv.first.str();
    auto it = shapeBindings.find(key);
    if (it == shapeBindings.end())
      return false;
    const int64_t v = it->second;

    const llvm::json::Value &cond = kv.second;
    if (auto i = cond.getAsInteger()) {
      if (v != static_cast<int64_t>(*i))
        return false;
      continue;
    }
    if (auto arr = cond.getAsArray()) {
      bool ok = false;
      for (const auto &it2 : *arr) {
        auto ii = it2.getAsInteger();
        if (!ii)
          continue;
        if (v == static_cast<int64_t>(*ii)) {
          ok = true;
          break;
        }
      }
      if (!ok)
        return false;
      continue;
    }
    auto obj = cond.getAsObject();
    if (!obj)
      return false;

    auto getI = [&](llvm::StringRef name) -> std::optional<int64_t> {
      auto ii = obj->getInteger(name);
      if (!ii)
        return std::nullopt;
      return static_cast<int64_t>(*ii);
    };

    if (auto eq = getI("eq"); eq && v != *eq)
      return false;
    if (auto ne = getI("ne"); ne && v == *ne)
      return false;
    if (auto lt = getI("lt"); lt && v >= *lt)
      return false;
    if (auto le = getI("le"); le && v > *le)
      return false;
    if (auto gt = getI("gt"); gt && v <= *gt)
      return false;
    if (auto ge = getI("ge"); ge && v < *ge)
      return false;

    if (auto inArr = obj->getArray("in")) {
      bool ok = false;
      for (const auto &it2 : *inArr) {
        auto ii = it2.getAsInteger();
        if (!ii)
          continue;
        if (v == static_cast<int64_t>(*ii)) {
          ok = true;
          break;
        }
      }
      if (!ok)
        return false;
    }

    if (auto notInArr = obj->getArray("not_in")) {
      for (const auto &it2 : *notInArr) {
        auto ii = it2.getAsInteger();
        if (!ii)
          continue;
        if (v == static_cast<int64_t>(*ii))
          return false;
      }
    }

    if (auto d = getI("divisible_by")) {
      if (*d <= 0)
        return false;
      if ((v % *d) != 0)
        return false;
    }

    if (auto m = getI("mod")) {
      if (*m <= 0)
        return false;
      int64_t eq = 0;
      if (auto me = getI("mod_eq")) {
        eq = *me;
      } else if (auto eq2 = getI("eq")) {
        eq = *eq2;
      }
      if ((v % *m) != eq)
        return false;
    }
  }
  return true;
}

static std::string encodeShapeBindingsJson(const std::map<std::string, int64_t> &shapeBindings) {
  // Keys are stable and simple (A-Z0-9_). Encode compact JSON with sorted keys.
  std::string out;
  out.push_back('{');
  bool first = true;
  for (const auto &kv : shapeBindings) {
    if (!first)
      out.push_back(',');
    first = false;
    out.push_back('"');
    out.append(kv.first);
    out.append("\":");
    out.append(std::to_string(static_cast<long long>(kv.second)));
  }
  out.push_back('}');
  return out;
}

static llvm::json::Object loadIntentirMetaJson(mlir::ModuleOp module) {
  llvm::json::Object out;
  auto attr = module->getAttrOfType<mlir::StringAttr>("intentir.meta_json_b64");
  if (!attr) {
    return out;
  }
  auto decodedOr = decodeB64(attr.str());
  if (mlir::failed(decodedOr)) {
    return out;
  }
  auto parsedOr = parseJson(*decodedOr);
  if (mlir::failed(parsedOr)) {
    return out;
  }
  const auto *obj = (*parsedOr).getAsObject();
  if (!obj) {
    return out;
  }
  for (const auto &kv : *obj) {
    out[kv.first] = kv.second;
  }
  return out;
}

static void storeIntentirMetaJson(mlir::ModuleOp module, const llvm::json::Object &obj) {
  auto *mlirCtx = module.getContext();
  std::string jsonText;
  llvm::raw_string_ostream os(jsonText);
  llvm::json::Object tmp;
  for (const auto &kv : obj) {
    tmp[kv.first] = kv.second;
  }
  os << llvm::json::Value(std::move(tmp));
  os.flush();
  const std::string b64 = llvm::encodeBase64(llvm::StringRef(jsonText));
  module->setAttr("intentir.meta_json_b64", mlir::StringAttr::get(mlirCtx, b64));
}

static void mergeIntentirMetaJson(mlir::ModuleOp module,
                                  llvm::function_ref<void(llvm::json::Object &)> update) {
  llvm::json::Object meta = loadIntentirMetaJson(module);
  update(meta);
  storeIntentirMetaJson(module, meta);
}

static llvm::json::Object makeCudaLaunchOverride(int64_t bx, int64_t by, int64_t bz, int64_t gx,
                                                 int64_t gy, int64_t gz,
                                                 int64_t sharedMem = 0) {
  llvm::json::Object out;
  llvm::json::Array block;
  block.push_back(static_cast<int64_t>(bx));
  block.push_back(static_cast<int64_t>(by));
  block.push_back(static_cast<int64_t>(bz));
  llvm::json::Array grid;
  grid.push_back(static_cast<int64_t>(gx));
  grid.push_back(static_cast<int64_t>(gy));
  grid.push_back(static_cast<int64_t>(gz));
  out["block"] = std::move(block);
  out["grid"] = std::move(grid);
  out["shared_mem"] = static_cast<int64_t>(sharedMem);
  return out;
}

static llvm::json::Object makeJsonIntObject(const std::map<std::string, int64_t> &vals) {
  llvm::json::Object out;
  for (const auto &kv : vals) {
    out[kv.first] = static_cast<int64_t>(kv.second);
  }
  return out;
}

static std::optional<int64_t> resolveDimToken(llvm::json::Value tok,
                                              const std::map<std::string, int64_t> &bindings) {
  if (auto i = tok.getAsInteger()) {
    return static_cast<int64_t>(*i);
  }
  auto sOpt = tok.getAsString();
  if (!sOpt) {
    return std::nullopt;
  }
  std::string s = sOpt->str();
  // Trim spaces.
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
  if (s.empty())
    return std::nullopt;
  auto it = bindings.find(s);
  if (it != bindings.end())
    return it->second;
  // Support a conservative "BASE+INT" form.
  auto plusPos = s.find('+');
  if (plusPos != std::string::npos && plusPos > 0 && plusPos + 1 < s.size()) {
    std::string base = s.substr(0, plusPos);
    std::string deltaStr = s.substr(plusPos + 1);
    char *end = nullptr;
    long delta = std::strtol(deltaStr.c_str(), &end, 10);
    if (end && *end == '\0') {
      auto it2 = bindings.find(base);
      if (it2 != bindings.end()) {
        return it2->second + static_cast<int64_t>(delta);
      }
    }
  }
  // Try parse as int.
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end && *end == '\0') {
    return static_cast<int64_t>(v);
  }
  return std::nullopt;
}

struct TensorSpec {
  std::string dtype;
  std::vector<llvm::json::Value> shapeTokens;
};

static mlir::FailureOr<TensorSpec> getTensorSpec(const llvm::json::Object &intent,
                                                 llvm::StringRef name) {
  const auto *tensors = intent.getObject("tensors");
  if (!tensors)
    return mlir::failure();
  const auto *spec = tensors->getObject(name);
  if (!spec)
    return mlir::failure();
  auto dtype = spec->getString("dtype");
  const auto *shape = spec->getArray("shape");
  if (!dtype || !shape)
    return mlir::failure();
  TensorSpec out;
  out.dtype = dtype->str();
  out.shapeTokens.reserve(shape->size());
  for (const auto &tok : *shape) {
    out.shapeTokens.push_back(tok);
  }
  return out;
}

static mlir::FailureOr<std::vector<int64_t>>
resolveShape(const TensorSpec &spec, const std::map<std::string, int64_t> &bindings) {
  std::vector<int64_t> out;
  out.reserve(spec.shapeTokens.size());
  for (auto tok : spec.shapeTokens) {
    auto v = resolveDimToken(tok, bindings);
    if (!v)
      return mlir::failure();
    out.push_back(*v);
  }
  return out;
}

static mlir::FailureOr<int64_t> shapeNumel(const std::vector<int64_t> &shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d <= 0)
      return mlir::failure();
    if (numel > (std::numeric_limits<int64_t>::max() / d))
      return mlir::failure();
    numel *= d;
  }
  return numel;
}

static mlir::Type dtypeToElemType(mlir::MLIRContext *ctx, llvm::StringRef dtype) {
  auto d = dtype.trim().lower();
  if (d == "f32")
    return mlir::Float32Type::get(ctx);
  if (d == "f16")
    return mlir::Float16Type::get(ctx);
  if (d == "bf16")
    return mlir::BFloat16Type::get(ctx);
  if (d == "i32")
    return mlir::IntegerType::get(ctx, 32);
  if (d == "i64")
    return mlir::IntegerType::get(ctx, 64);
  // Baseline RVV ABI uses i8 for bool; keep conservative mapping.
  if (d == "bool" || d == "i1")
    return mlir::IntegerType::get(ctx, 8);
  return {};
}

struct OpSpec {
  std::string op;
  std::vector<std::string> inputs;
  std::string output;
  llvm::json::Object attrs;
};

static mlir::FailureOr<std::vector<OpSpec>> parseOps(const llvm::json::Object &intent) {
  const auto *ops = intent.getArray("ops");
  if (!ops)
    return mlir::failure();
  std::vector<OpSpec> out;
  out.reserve(ops->size());
  for (const auto &vv : *ops) {
    const auto *obj = vv.getAsObject();
    if (!obj)
      return mlir::failure();
    auto opName = obj->getString("op");
    auto outName = obj->getString("output");
    const auto *ins = obj->getArray("inputs");
    if (!opName || !outName || !ins)
      return mlir::failure();
    OpSpec s;
    s.op = opName->str();
    s.output = outName->str();
    for (const auto &iv : *ins) {
      auto is = iv.getAsString();
      if (!is)
        return mlir::failure();
      s.inputs.push_back(is->str());
    }
    const auto *attrs = obj->getObject("attrs");
    if (attrs) {
      s.attrs = *attrs;
    }
    out.push_back(std::move(s));
  }
  return out;
}

static mlir::FailureOr<std::vector<std::string>>
parseOutputs(const llvm::json::Object &intent) {
  const auto *outs = intent.getArray("outputs");
  if (!outs)
    return mlir::failure();
  std::vector<std::string> out;
  out.reserve(outs->size());
  for (const auto &vv : *outs) {
    auto s = vv.getAsString();
    if (!s)
      return mlir::failure();
    auto str = s->str();
    if (!str.empty())
      out.push_back(str);
  }
  return out;
}

static std::vector<std::string> computeIOArgOrder(
    const std::map<std::string, TensorSpec> &tensors,
    const std::vector<OpSpec> &ops,
    const std::vector<std::string> &outputs,
    const std::map<std::string, int64_t> &shapeBindings) {
  std::set<std::string> produced;
  std::set<std::string> used;
  for (const auto &op : ops) {
    if (!op.output.empty())
      produced.insert(op.output);
    for (const auto &in : op.inputs)
      used.insert(in);
  }

  std::vector<std::string> externalInputs;
  externalInputs.reserve(used.size());
  for (const auto &name : used) {
    if (tensors.count(name) == 0)
      continue;
    if (produced.count(name))
      continue;
    externalInputs.push_back(name);
  }

  // Some kernels require runtime scalar inputs that are not explicitly listed
  // in op.inputs (e.g. reciprocal scales). Treat non-produced tensors that are
  // not pure shape symbols as external inputs as well.
  std::set<std::string> outSet(outputs.begin(), outputs.end());
  for (const auto &kv : tensors) {
    const auto &name = kv.first;
    if (produced.count(name))
      continue;
    if (outSet.count(name))
      continue;
    if (used.count(name))
      continue;
    if (shapeBindings.count(name))
      continue;
    externalInputs.push_back(name);
  }
  std::sort(externalInputs.begin(), externalInputs.end());
  externalInputs.erase(std::unique(externalInputs.begin(), externalInputs.end()), externalInputs.end());

  std::set<std::string> extSet(externalInputs.begin(), externalInputs.end());
  std::vector<std::string> argOrder = externalInputs;
  for (const auto &out : outputs) {
    if (tensors.count(out) == 0)
      continue;
    if (extSet.count(out))
      continue;
    argOrder.push_back(out);
  }
  return argOrder;
}

struct LoweringContext {
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
  std::map<std::string, int64_t> shapeBindings;
  llvm::json::Object intentObj;
  std::map<std::string, TensorSpec> tensors;
  std::vector<OpSpec> ops;
  std::vector<std::string> outputs;
  std::vector<std::string> argOrder;
  std::string kernelName;
  std::string kernelKindOverride;
};

static mlir::Value makeIndexConst(mlir::OpBuilder &b, mlir::Location loc, int64_t v) {
  return b.create<mlir::arith::ConstantIndexOp>(loc, v);
}

static mlir::Value makeI32Const(mlir::OpBuilder &b, mlir::Location loc, int32_t v) {
  return b.create<mlir::arith::ConstantIntOp>(loc, v, 32);
}

static mlir::Value makeI64Const(mlir::OpBuilder &b, mlir::Location loc, int64_t v) {
  return b.create<mlir::arith::ConstantIntOp>(loc, v, 64);
}

static mlir::Value makeI1Const(mlir::OpBuilder &b, mlir::Location loc, bool v) {
  return b.create<mlir::arith::ConstantIntOp>(loc, v ? 1 : 0, 1);
}

static mlir::Value makeF32Const(mlir::OpBuilder &b, mlir::Location loc, float v) {
  return b.create<mlir::arith::ConstantFloatOp>(loc, b.getF32Type(), llvm::APFloat(v));
}

static mlir::Value warpAllReduceSumF32(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v) {
  auto c32 = makeI32Const(b, loc, 32);
  mlir::Value cur = v;
  for (int32_t offset : {16, 8, 4, 2, 1}) {
    auto off = makeI32Const(b, loc, offset);
    auto sh = b.create<mlir::gpu::ShuffleOp>(loc, cur, off, c32, mlir::gpu::ShuffleMode::XOR);
    auto val = sh.getResult(0);
    cur = b.create<mlir::arith::AddFOp>(loc, cur, val).getResult();
  }
  return cur;
}

static mlir::Value warpAllReduceMaxF32(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v) {
  auto c32 = makeI32Const(b, loc, 32);
  mlir::Value cur = v;
  for (int32_t offset : {16, 8, 4, 2, 1}) {
    auto off = makeI32Const(b, loc, offset);
    auto sh = b.create<mlir::gpu::ShuffleOp>(loc, cur, off, c32, mlir::gpu::ShuffleMode::XOR);
    auto val = sh.getResult(0);
    cur = b.create<mlir::arith::MaximumFOp>(loc, cur, val).getResult();
  }
  return cur;
}

static mlir::FailureOr<LoweringContext> parseLoweringContext(mlir::ModuleOp module) {
  auto jsonB64Or = getRequiredStringAttr(module, "intentir.intent_json_b64");
  if (mlir::failed(jsonB64Or)) {
    module.emitError("missing required module attribute: intentir.intent_json_b64");
    return mlir::failure();
  }
  auto bindingsB64Or = getRequiredStringAttr(module, "intentir.shape_bindings_b64");
  if (mlir::failed(bindingsB64Or)) {
    module.emitError("missing required module attribute: intentir.shape_bindings_b64");
    return mlir::failure();
  }
  auto jsonTextOr = decodeB64(*jsonB64Or);
  auto bindingsTextOr = decodeB64(*bindingsB64Or);
  if (mlir::failed(jsonTextOr) || mlir::failed(bindingsTextOr)) {
    module.emitError("failed to decode base64 module payload");
    return mlir::failure();
  }
  auto jsonValOr = parseJson(*jsonTextOr);
  auto bindingsValOr = parseJson(*bindingsTextOr);
  if (mlir::failed(jsonValOr) || mlir::failed(bindingsValOr)) {
    module.emitError("failed to parse JSON payload from module attributes");
    return mlir::failure();
  }
  auto *intentObj = (*jsonValOr).getAsObject();
  if (!intentObj) {
    module.emitError("intent JSON payload is not an object");
    return mlir::failure();
  }
  auto shapeBindingsOr = parseShapeBindings(*bindingsValOr);
  if (mlir::failed(shapeBindingsOr)) {
    module.emitError("shape_bindings JSON payload is not an object");
    return mlir::failure();
  }

  auto opsOr = parseOps(*intentObj);
  auto outsOr = parseOutputs(*intentObj);
  if (mlir::failed(opsOr) || mlir::failed(outsOr)) {
    module.emitError("failed to parse ops/outputs from intent JSON payload");
    return mlir::failure();
  }

  std::map<std::string, TensorSpec> tensors;
  const auto *tensorsObj = intentObj->getObject("tensors");
  if (!tensorsObj) {
    module.emitError("intent JSON missing tensors object");
    return mlir::failure();
  }
  for (const auto &kv : *tensorsObj) {
    auto name = kv.first.str();
    auto specOr = getTensorSpec(*intentObj, name);
    if (mlir::failed(specOr)) {
      module.emitError() << "failed to parse tensor spec for name=" << name;
      return mlir::failure();
    }
    tensors.emplace(std::move(name), *specOr);
  }

  std::string kernelName;
  if (auto attr = module->getAttrOfType<mlir::StringAttr>("intentir.intent_name")) {
    kernelName = attr.str();
  } else if (auto nm = intentObj->getString("name")) {
    kernelName = nm->str();
  }
  if (kernelName.empty())
    kernelName = "intent";

  std::string kernelKindOverride;
  if (auto attr = module->getAttrOfType<mlir::StringAttr>("intentir.kernel_kind_override")) {
    kernelKindOverride = attr.str();
  }

  LoweringContext ctx{
      module,
      mlir::OpBuilder(module.getContext()),
      *shapeBindingsOr,
      *intentObj,
      std::move(tensors),
      *opsOr,
      *outsOr,
      {},
      kernelName,
      kernelKindOverride,
  };
  ctx.argOrder = computeIOArgOrder(ctx.tensors, ctx.ops, ctx.outputs, ctx.shapeBindings);
  return ctx;
}

static void clearModuleBody(mlir::ModuleOp module) {
  auto &block = module.getBodyRegion().front();
  while (!block.empty()) {
    block.front().erase();
  }
}

static mlir::FailureOr<mlir::func::FuncOp>
createFuncWithFlattenedABI(LoweringContext &ctx, llvm::StringRef funcName) {
  auto loc = ctx.module.getLoc();
  auto *mlirCtx = ctx.module.getContext();
  std::vector<mlir::Type> argTypes;
  argTypes.reserve(ctx.argOrder.size());

  for (const auto &name : ctx.argOrder) {
    auto it = ctx.tensors.find(name);
    if (it == ctx.tensors.end()) {
      ctx.module.emitError() << "missing tensor spec for IO name=" << name;
      return mlir::failure();
    }
    const TensorSpec &spec = it->second;
    auto elemTy = dtypeToElemType(mlirCtx, spec.dtype);
    if (!elemTy) {
      ctx.module.emitError() << "unsupported dtype for tensor " << name << ": " << spec.dtype;
      return mlir::failure();
    }
    auto shapeOr = resolveShape(spec, ctx.shapeBindings);
    if (mlir::failed(shapeOr)) {
      ctx.module.emitError() << "failed to resolve shape for tensor " << name;
      return mlir::failure();
    }
    auto numelOr = shapeNumel(*shapeOr);
    if (mlir::failed(numelOr)) {
      ctx.module.emitError() << "invalid resolved shape for tensor " << name;
      return mlir::failure();
    }
    auto memrefTy = mlir::MemRefType::get({*numelOr}, elemTy);
    argTypes.push_back(memrefTy);
  }

  auto fnType = mlir::FunctionType::get(mlirCtx, argTypes, {});
  auto fn = mlir::func::FuncOp::create(loc, funcName, fnType);
  fn.setPrivate();
  ctx.module.push_back(fn);
  auto *entry = fn.addEntryBlock();
  ctx.builder.setInsertionPointToStart(entry);
  return fn;
}

static mlir::Value getArgByName(LoweringContext &ctx, mlir::func::FuncOp fn,
                                llvm::StringRef tensorName) {
  for (size_t i = 0; i < ctx.argOrder.size(); ++i) {
    if (ctx.argOrder[i] == tensorName.str()) {
      return fn.getArgument(static_cast<unsigned>(i));
    }
  }
  return {};
}

static mlir::LogicalResult lowerElementwiseF32(LoweringContext &ctx) {
  if (ctx.outputs.size() != 1) {
    ctx.module.emitError("elementwise: expected single output");
    return mlir::failure();
  }
  std::string outName = ctx.outputs[0];
  auto outIt = ctx.tensors.find(outName);
  if (outIt == ctx.tensors.end()) {
    ctx.module.emitError("elementwise: missing output tensor spec");
    return mlir::failure();
  }
  if (llvm::StringRef(outIt->second.dtype).trim().lower() != "f32") {
    ctx.module.emitError("elementwise: only f32 output supported");
    return mlir::failure();
  }
  auto outShapeOr = resolveShape(outIt->second, ctx.shapeBindings);
  if (mlir::failed(outShapeOr)) {
    ctx.module.emitError("elementwise: failed to resolve output shape");
    return mlir::failure();
  }
  auto outNumelOr = shapeNumel(*outShapeOr);
  if (mlir::failed(outNumelOr)) {
    ctx.module.emitError("elementwise: invalid output shape");
    return mlir::failure();
  }
  int64_t outNumel = *outNumelOr;

  // Validate that all non-scalar external inputs match output numel.
  for (const auto &argName : ctx.argOrder) {
    if (argName == outName)
      continue;
    auto it = ctx.tensors.find(argName);
    if (it == ctx.tensors.end())
      continue;
    auto shOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shOr))
      continue;
    auto nOr = shapeNumel(*shOr);
    if (mlir::failed(nOr))
      continue;
    bool isScalar = (shOr->empty());
    if (!isScalar && *nOr != outNumel) {
      ctx.module.emitError() << "elementwise: input " << argName << " numel=" << *nOr
                             << " does not match output numel=" << outNumel;
      return mlir::failure();
    }
  }

  clearModuleBody(ctx.module);
  auto fnOr = createFuncWithFlattenedABI(ctx, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto outArg = getArgByName(ctx, fn, outName);
  if (!outArg) {
    ctx.module.emitError("elementwise: failed to map output argument");
    return mlir::failure();
  }

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto cT = makeIndexConst(b, loc, outNumel);

  auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cT, c1);
  b.setInsertionPointToStart(forOp.getBody());
  auto i = forOp.getInductionVar();

  std::map<std::string, mlir::Value> env;
  // Load all external inputs for this element.
  for (const auto &name : ctx.argOrder) {
    if (name == outName)
      continue;
    auto arg = getArgByName(ctx, fn, name);
    if (!arg)
      continue;
    auto it = ctx.tensors.find(name);
    if (it == ctx.tensors.end())
      continue;
    auto shOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shOr))
      continue;
    bool isScalar = shOr->empty();
    auto idx = isScalar ? c0 : i;
    auto v = b.create<mlir::memref::LoadOp>(loc, arg, mlir::ValueRange{idx}).getResult();
    env[name] = v;
  }

  // Evaluate ops in order (Intent op list is expected to be topologically sorted).
  for (const auto &op : ctx.ops) {
    if (op.inputs.empty()) {
      ctx.module.emitError() << "elementwise: op has no inputs: " << op.op;
      return mlir::failure();
    }
    std::vector<mlir::Value> ins;
    ins.reserve(op.inputs.size());
    for (const auto &inName : op.inputs) {
      auto it = env.find(inName);
      if (it == env.end()) {
        ctx.module.emitError() << "elementwise: missing SSA value for input=" << inName
                               << " (op=" << op.op << ")";
        return mlir::failure();
      }
      ins.push_back(it->second);
    }

    mlir::Value outV;
    if (op.op == "add" && ins.size() == 2) {
      outV = b.create<mlir::arith::AddFOp>(loc, ins[0], ins[1]).getResult();
    } else if (op.op == "mul" && ins.size() == 2) {
      outV = b.create<mlir::arith::MulFOp>(loc, ins[0], ins[1]).getResult();
    } else if (op.op == "sub" && ins.size() == 2) {
      outV = b.create<mlir::arith::SubFOp>(loc, ins[0], ins[1]).getResult();
    } else if (op.op == "div" && ins.size() == 2) {
      outV = b.create<mlir::arith::DivFOp>(loc, ins[0], ins[1]).getResult();
    } else {
      ctx.module.emitError() << "elementwise: unsupported op=" << op.op << " inputs=" << ins.size();
      return mlir::failure();
    }
    env[op.output] = outV;
  }

  auto itOut = env.find(outName);
  if (itOut == env.end()) {
    ctx.module.emitError("elementwise: output SSA value not produced");
    return mlir::failure();
  }
  b.create<mlir::memref::StoreOp>(loc, itOut->second, outArg, mlir::ValueRange{i});
  b.setInsertionPointAfter(forOp);
  b.create<mlir::func::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(ctx.module.getContext(), "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(ctx.module.getContext(), "rvv_cpu_loops_v1"));
  return mlir::success();
}

static mlir::LogicalResult lowerRowSum(LoweringContext &ctx) {
  if (ctx.outputs.size() != 1) {
    ctx.module.emitError("row_sum: expected single output");
    return mlir::failure();
  }
  if (ctx.ops.size() != 1) {
    ctx.module.emitError("row_sum: expected exactly one op");
    return mlir::failure();
  }
  const auto &op = ctx.ops[0];
  if (op.op != "reduce_sum" || op.inputs.size() != 1) {
    ctx.module.emitError("row_sum: expected reduce_sum with 1 input");
    return mlir::failure();
  }
  auto dimsVal = op.attrs.get("dims");
  const auto *dimsArr = dimsVal ? dimsVal->getAsArray() : nullptr;
  if (!dimsArr || dimsArr->size() != 1 || !(*dimsArr)[0].getAsInteger() ||
      *(*dimsArr)[0].getAsInteger() != 1) {
    ctx.module.emitError("row_sum: expected dims=[1]");
    return mlir::failure();
  }

  std::string inName = op.inputs[0];
  std::string outName = ctx.outputs[0];

  auto inIt = ctx.tensors.find(inName);
  auto outIt = ctx.tensors.find(outName);
  if (inIt == ctx.tensors.end() || outIt == ctx.tensors.end())
    return mlir::failure();
  auto inShapeOr = resolveShape(inIt->second, ctx.shapeBindings);
  auto outShapeOr = resolveShape(outIt->second, ctx.shapeBindings);
  if (mlir::failed(inShapeOr) || mlir::failed(outShapeOr))
    return mlir::failure();
  if (inShapeOr->size() != 2 || outShapeOr->size() != 1 ||
      (*inShapeOr)[0] != (*outShapeOr)[0]) {
    ctx.module.emitError("row_sum: expected input [M,N] and output [M]");
    return mlir::failure();
  }
  int64_t M = (*outShapeOr)[0];
  int64_t N = (*inShapeOr)[1];
  if (M <= 0 || N <= 0)
    return mlir::failure();

  clearModuleBody(ctx.module);
  auto fnOr = createFuncWithFlattenedABI(ctx, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto inArg = getArgByName(ctx, fn, inName);
  auto outArg = getArgByName(ctx, fn, outName);
  if (!inArg || !outArg) {
    ctx.module.emitError("row_sum: failed to map function arguments");
    return mlir::failure();
  }

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto cM = makeIndexConst(b, loc, M);
  auto cN = makeIndexConst(b, loc, N);
  auto init = makeF32Const(b, loc, 0.0f);

  auto outer = b.create<mlir::scf::ForOp>(loc, c0, cM, c1);
  b.setInsertionPointToStart(outer.getBody());
  auto m = outer.getInductionVar();
  auto base = b.create<mlir::arith::MulIOp>(loc, m, cN);
  auto inner = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{init});
  b.setInsertionPointToStart(inner.getBody());
  auto n = inner.getInductionVar();
  auto acc = inner.getRegionIterArgs()[0];
  auto idx = b.create<mlir::arith::AddIOp>(loc, base, n);
  auto v = b.create<mlir::memref::LoadOp>(loc, inArg, mlir::ValueRange{idx}).getResult();
  auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, v).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
  b.setInsertionPointAfter(inner);
  b.create<mlir::memref::StoreOp>(loc, inner.getResult(0), outArg, mlir::ValueRange{m});
  b.setInsertionPointAfter(outer);
  b.create<mlir::func::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(ctx.module.getContext(), "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(ctx.module.getContext(), "rvv_cpu_loops_v1"));
  return mlir::success();
}

static mlir::LogicalResult lowerGather2dLike(LoweringContext &ctx) {
  if (ctx.outputs.size() != 1) {
    ctx.module.emitError("gather2d: expected single output");
    return mlir::failure();
  }
  if (ctx.ops.size() != 1) {
    ctx.module.emitError("gather2d: expected exactly one op");
    return mlir::failure();
  }
  const auto &op = ctx.ops[0];
  if (op.op != "gather" || op.inputs.size() != 3) {
    ctx.module.emitError("gather2d: expected op=gather with 3 inputs");
    return mlir::failure();
  }
  std::string dataName = op.inputs[0];
  std::string rowName = op.inputs[1];
  std::string colName = op.inputs[2];
  std::string outName = ctx.outputs[0];

  auto dataIt = ctx.tensors.find(dataName);
  auto rowIt = ctx.tensors.find(rowName);
  auto colIt = ctx.tensors.find(colName);
  auto outIt = ctx.tensors.find(outName);
  if (dataIt == ctx.tensors.end() || rowIt == ctx.tensors.end() || colIt == ctx.tensors.end() ||
      outIt == ctx.tensors.end()) {
    return mlir::failure();
  }

  auto dataShapeOr = resolveShape(dataIt->second, ctx.shapeBindings);
  auto rowShapeOr = resolveShape(rowIt->second, ctx.shapeBindings);
  auto colShapeOr = resolveShape(colIt->second, ctx.shapeBindings);
  auto outShapeOr = resolveShape(outIt->second, ctx.shapeBindings);
  if (mlir::failed(dataShapeOr) || mlir::failed(rowShapeOr) || mlir::failed(colShapeOr) ||
      mlir::failed(outShapeOr)) {
    return mlir::failure();
  }
  if (dataShapeOr->size() != 2) {
    ctx.module.emitError("gather2d: expected data input rank 2");
    return mlir::failure();
  }
  if (*rowShapeOr != *outShapeOr || *colShapeOr != *outShapeOr) {
    ctx.module.emitError("gather2d: expected row/col idx shapes to match output shape");
    return mlir::failure();
  }
  int64_t N = (*dataShapeOr)[1];
  auto outNumelOr = shapeNumel(*outShapeOr);
  if (mlir::failed(outNumelOr) || N <= 0) {
    return mlir::failure();
  }

  clearModuleBody(ctx.module);
  auto fnOr = createFuncWithFlattenedABI(ctx, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto dataArg = getArgByName(ctx, fn, dataName);
  auto rowArg = getArgByName(ctx, fn, rowName);
  auto colArg = getArgByName(ctx, fn, colName);
  auto outArg = getArgByName(ctx, fn, outName);
  if (!dataArg || !rowArg || !colArg || !outArg) {
    ctx.module.emitError("gather2d: failed to map function arguments");
    return mlir::failure();
  }

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto cT = makeIndexConst(b, loc, *outNumelOr);
  auto cN = makeIndexConst(b, loc, N);

  auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cT, c1);
  b.setInsertionPointToStart(forOp.getBody());
  auto i = forOp.getInductionVar();
  auto r32 = b.create<mlir::memref::LoadOp>(loc, rowArg, mlir::ValueRange{i});
  auto c32 = b.create<mlir::memref::LoadOp>(loc, colArg, mlir::ValueRange{i});
  auto r = b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), r32);
  auto c = b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), c32);
  auto mul = b.create<mlir::arith::MulIOp>(loc, r, cN);
  auto idx = b.create<mlir::arith::AddIOp>(loc, mul, c);
  auto x = b.create<mlir::memref::LoadOp>(loc, dataArg, mlir::ValueRange{idx});
  b.create<mlir::memref::StoreOp>(loc, x, outArg, mlir::ValueRange{i});
  b.setInsertionPointAfter(forOp);
  b.create<mlir::func::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(ctx.module.getContext(), "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(ctx.module.getContext(), "rvv_cpu_loops_v1"));
  return mlir::success();
}

static mlir::LogicalResult lowerConcat2d(LoweringContext &ctx) {
  if (ctx.outputs.size() != 1) {
    ctx.module.emitError("cat2d: expected single output");
    return mlir::failure();
  }
  if (ctx.ops.size() != 1) {
    ctx.module.emitError("cat2d: expected exactly one op");
    return mlir::failure();
  }
  const auto &op = ctx.ops[0];
  if (op.op != "concat" || op.inputs.size() != 2) {
    ctx.module.emitError("cat2d: expected op=concat with 2 inputs");
    return mlir::failure();
  }
  auto axisVal = op.attrs.get("axis");
  auto axisInt = axisVal ? axisVal->getAsInteger() : std::optional<int64_t>{};
  if (!axisInt || (*axisInt != 0 && *axisInt != 1)) {
    ctx.module.emitError("cat2d: expected axis 0 or 1");
    return mlir::failure();
  }
  int64_t axis = *axisInt;

  std::string aName = op.inputs[0];
  std::string bName = op.inputs[1];
  std::string outName = ctx.outputs[0];

  auto aIt = ctx.tensors.find(aName);
  auto bIt = ctx.tensors.find(bName);
  auto outIt = ctx.tensors.find(outName);
  if (aIt == ctx.tensors.end() || bIt == ctx.tensors.end() || outIt == ctx.tensors.end())
    return mlir::failure();
  auto aShapeOr = resolveShape(aIt->second, ctx.shapeBindings);
  auto bShapeOr = resolveShape(bIt->second, ctx.shapeBindings);
  auto outShapeOr = resolveShape(outIt->second, ctx.shapeBindings);
  if (mlir::failed(aShapeOr) || mlir::failed(bShapeOr) || mlir::failed(outShapeOr))
    return mlir::failure();
  if (aShapeOr->size() != 2 || bShapeOr->size() != 2 || outShapeOr->size() != 2) {
    ctx.module.emitError("cat2d: expected rank-2 inputs/outputs");
    return mlir::failure();
  }
  int64_t am = (*aShapeOr)[0], an = (*aShapeOr)[1];
  int64_t bm = (*bShapeOr)[0], bn = (*bShapeOr)[1];
  int64_t om = (*outShapeOr)[0], on = (*outShapeOr)[1];

  if (axis == 0) {
    if (an != bn || on != an || om != (am + bm)) {
      ctx.module.emitError("cat2d: axis=0 shape mismatch");
      return mlir::failure();
    }
  } else {
    if (am != bm || om != am || on != (an + bn)) {
      ctx.module.emitError("cat2d: axis=1 shape mismatch");
      return mlir::failure();
    }
  }

  clearModuleBody(ctx.module);
  auto fnOr = createFuncWithFlattenedABI(ctx, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto aArg = getArgByName(ctx, fn, aName);
  auto bArg = getArgByName(ctx, fn, bName);
  auto outArg = getArgByName(ctx, fn, outName);
  if (!aArg || !bArg || !outArg) {
    ctx.module.emitError("cat2d: failed to map function arguments");
    return mlir::failure();
  }

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto cM = makeIndexConst(b, loc, om);
  auto cN = makeIndexConst(b, loc, on);
  auto cAn = makeIndexConst(b, loc, an);
  auto cBn = makeIndexConst(b, loc, bn);

  auto outer = b.create<mlir::scf::ForOp>(loc, c0, cM, c1);
  b.setInsertionPointToStart(outer.getBody());
  auto m = outer.getInductionVar();
  auto rowOut = b.create<mlir::arith::MulIOp>(loc, m, cN);

  auto inner = b.create<mlir::scf::ForOp>(loc, c0, cN, c1);
  b.setInsertionPointToStart(inner.getBody());
  auto n = inner.getInductionVar();
  auto outIdx = b.create<mlir::arith::AddIOp>(loc, rowOut, n);

  mlir::Value v;
  if (axis == 0) {
    auto takeA = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, m,
                                               makeIndexConst(b, loc, am));
    auto ifOp = b.create<mlir::scf::IfOp>(loc, b.getF32Type(), takeA, true);
    // Then.
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto rowA = b.create<mlir::arith::MulIOp>(loc, m, cAn);
    auto idxA = b.create<mlir::arith::AddIOp>(loc, rowA, n);
    auto xA = b.create<mlir::memref::LoadOp>(loc, aArg, mlir::ValueRange{idxA}).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{xA});
    // Else.
    b.setInsertionPointToStart(&ifOp.getElseRegion().front());
    auto m2 = b.create<mlir::arith::SubIOp>(loc, m, makeIndexConst(b, loc, am));
    auto rowB = b.create<mlir::arith::MulIOp>(loc, m2, cBn);
    auto idxB = b.create<mlir::arith::AddIOp>(loc, rowB, n);
    auto xB = b.create<mlir::memref::LoadOp>(loc, bArg, mlir::ValueRange{idxB}).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{xB});
    b.setInsertionPointAfter(ifOp);
    v = ifOp.getResult(0);
  } else {
    auto takeA = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, n, cAn);
    auto ifOp = b.create<mlir::scf::IfOp>(loc, b.getF32Type(), takeA, true);
    // Then.
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto rowA = b.create<mlir::arith::MulIOp>(loc, m, cAn);
    auto idxA = b.create<mlir::arith::AddIOp>(loc, rowA, n);
    auto xA = b.create<mlir::memref::LoadOp>(loc, aArg, mlir::ValueRange{idxA}).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{xA});
    // Else.
    b.setInsertionPointToStart(&ifOp.getElseRegion().front());
    auto n2 = b.create<mlir::arith::SubIOp>(loc, n, cAn);
    auto rowB = b.create<mlir::arith::MulIOp>(loc, m, cBn);
    auto idxB = b.create<mlir::arith::AddIOp>(loc, rowB, n2);
    auto xB = b.create<mlir::memref::LoadOp>(loc, bArg, mlir::ValueRange{idxB}).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{xB});
    b.setInsertionPointAfter(ifOp);
    v = ifOp.getResult(0);
  }

  b.create<mlir::memref::StoreOp>(loc, v, outArg, mlir::ValueRange{outIdx});
  b.setInsertionPointAfter(inner);
  b.setInsertionPointAfter(outer);
  b.create<mlir::func::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(ctx.module.getContext(), "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(ctx.module.getContext(), "rvv_cpu_loops_v1"));
  return mlir::success();
}

static mlir::LogicalResult lowerDiag2d(LoweringContext &ctx) {
  if (ctx.outputs.size() != 1) {
    ctx.module.emitError("diag2d: expected single output");
    return mlir::failure();
  }
  std::string outName = ctx.outputs[0];
  auto outIt = ctx.tensors.find(outName);
  if (outIt == ctx.tensors.end())
    return mlir::failure();
  auto outShapeOr = resolveShape(outIt->second, ctx.shapeBindings);
  if (mlir::failed(outShapeOr) || outShapeOr->size() != 1) {
    ctx.module.emitError("diag2d: expected rank-1 output");
    return mlir::failure();
  }
  int64_t L = (*outShapeOr)[0];
  if (L <= 0)
    return mlir::failure();

  // Infer a rank-2 f32 external input.
  std::string inName;
  int64_t N = 0;
  for (const auto &nm : ctx.argOrder) {
    if (nm == outName)
      continue;
    auto it = ctx.tensors.find(nm);
    if (it == ctx.tensors.end())
      continue;
    auto elemTy = it->second.dtype;
    if (llvm::StringRef(elemTy).trim().lower() != "f32")
      continue;
    auto shapeOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shapeOr))
      continue;
    if (shapeOr->size() == 2 && (*shapeOr)[0] > 0 && (*shapeOr)[1] > 0) {
      inName = nm;
      N = (*shapeOr)[1];
      break;
    }
  }
  if (inName.empty() || N <= 0) {
    ctx.module.emitError("diag2d: failed to infer input [M,N]");
    return mlir::failure();
  }

  clearModuleBody(ctx.module);
  auto fnOr = createFuncWithFlattenedABI(ctx, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto inArg = getArgByName(ctx, fn, inName);
  auto outArg = getArgByName(ctx, fn, outName);
  if (!inArg || !outArg) {
    ctx.module.emitError("diag2d: failed to map function arguments");
    return mlir::failure();
  }

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto cL = makeIndexConst(b, loc, L);
  auto cN = makeIndexConst(b, loc, N);

  auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cL, c1);
  b.setInsertionPointToStart(forOp.getBody());
  auto i = forOp.getInductionVar();
  auto mul = b.create<mlir::arith::MulIOp>(loc, i, cN);
  auto idx = b.create<mlir::arith::AddIOp>(loc, mul, i);
  auto x = b.create<mlir::memref::LoadOp>(loc, inArg, mlir::ValueRange{idx});
  b.create<mlir::memref::StoreOp>(loc, x, outArg, mlir::ValueRange{i});
  b.setInsertionPointAfter(forOp);
  b.create<mlir::func::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(ctx.module.getContext(), "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(ctx.module.getContext(), "rvv_cpu_loops_v1"));
  return mlir::success();
}

static mlir::FailureOr<mlir::gpu::GPUFuncOp>
createCudaKernelWithFlattenedABI(LoweringContext &ctx, mlir::gpu::GPUModuleOp gpuModule,
                                 llvm::StringRef funcName) {
  auto loc = ctx.module.getLoc();
  auto *mlirCtx = ctx.module.getContext();
  std::vector<mlir::Type> argTypes;
  argTypes.reserve(ctx.argOrder.size());

  auto memSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);
  for (const auto &name : ctx.argOrder) {
    auto it = ctx.tensors.find(name);
    if (it == ctx.tensors.end()) {
      ctx.module.emitError() << "missing tensor spec for IO name=" << name;
      return mlir::failure();
    }
    const TensorSpec &spec = it->second;
    auto elemTy = dtypeToElemType(mlirCtx, spec.dtype);
    if (!elemTy) {
      ctx.module.emitError() << "unsupported dtype for tensor " << name << ": "
                             << spec.dtype;
      return mlir::failure();
    }
    auto shapeOr = resolveShape(spec, ctx.shapeBindings);
    if (mlir::failed(shapeOr)) {
      ctx.module.emitError() << "failed to resolve shape for tensor " << name;
      return mlir::failure();
    }
    auto numelOr = shapeNumel(*shapeOr);
    if (mlir::failed(numelOr)) {
      ctx.module.emitError() << "invalid resolved shape for tensor " << name;
      return mlir::failure();
    }
    auto memrefTy = mlir::MemRefType::get({*numelOr}, elemTy,
                                          mlir::MemRefLayoutAttrInterface{}, memSpace);
    argTypes.push_back(memrefTy);
  }

  auto fnType = mlir::FunctionType::get(mlirCtx, argTypes, {});
  ctx.builder.setInsertionPointToEnd(&gpuModule.getBodyRegion().front());
  auto fn = mlir::gpu::GPUFuncOp::create(ctx.builder, loc, funcName, fnType);
  fn.setPrivate();
  fn->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
              mlir::UnitAttr::get(mlirCtx));
  mlir::Block *entry = nullptr;
  if (fn.getBody().empty()) {
    entry = fn.addEntryBlock();
  } else {
    entry = &fn.getBody().front();
  }
  ctx.builder.setInsertionPointToStart(entry);
  return fn;
}

static mlir::Value getArgByName(LoweringContext &ctx, mlir::gpu::GPUFuncOp fn,
                                llvm::StringRef tensorName) {
  for (size_t i = 0; i < ctx.argOrder.size(); ++i) {
    if (ctx.argOrder[i] == tensorName.str()) {
      return fn.getArgument(static_cast<unsigned>(i));
    }
  }
  return {};
}

static mlir::LogicalResult lowerCudaAiBenchMatmulMmaTF32V1(LoweringContext &ctx) {
  // Match the single-op matmul intent: C = A @ B.
  std::string aName, bName, outName;
  for (const auto &op : ctx.ops) {
    if (op.op != "matmul")
      continue;
    if (op.inputs.size() != 2)
      continue;
    aName = op.inputs[0];
    bName = op.inputs[1];
    outName = op.output;
    break;
  }
  if (aName.empty() || bName.empty() || outName.empty()) {
    ctx.module.emitError("ai_bench_matmul: expected single matmul op");
    return mlir::failure();
  }
  if (ctx.tensors.find(aName) == ctx.tensors.end() ||
      ctx.tensors.find(bName) == ctx.tensors.end() ||
      ctx.tensors.find(outName) == ctx.tensors.end()) {
    ctx.module.emitError("ai_bench_matmul: missing tensor specs for A/B/C");
    return mlir::failure();
  }

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto shapeAOr = resolveShape(ctx.tensors[aName], ctx.shapeBindings);
  auto shapeBOr = resolveShape(ctx.tensors[bName], ctx.shapeBindings);
  auto shapeCOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  if (mlir::failed(shapeAOr) || mlir::failed(shapeBOr) || mlir::failed(shapeCOr)) {
    ctx.module.emitError("ai_bench_matmul: failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeAOr->size() != 2 || shapeBOr->size() != 2 || shapeCOr->size() != 2) {
    ctx.module.emitError("ai_bench_matmul: expected rank-2 tensors");
    return mlir::failure();
  }
  int64_t M = (*shapeAOr)[0];
  int64_t K = (*shapeAOr)[1];
  int64_t K2 = (*shapeBOr)[0];
  int64_t N = (*shapeBOr)[1];
  if (K != K2) {
    ctx.module.emitError("ai_bench_matmul: A.K != B.K");
    return mlir::failure();
  }
  if ((*shapeCOr)[0] != M || (*shapeCOr)[1] != N) {
    ctx.module.emitError("ai_bench_matmul: C shape mismatch");
    return mlir::failure();
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    ctx.module.emitError("ai_bench_matmul: invalid dims");
    return mlir::failure();
  }

  // Tunable tile params (shared-staged WMMA TF32 baseline).
  auto getBind = [&](llvm::StringRef key, int64_t defv) -> int64_t {
    auto it = ctx.shapeBindings.find(key.str());
    if (it == ctx.shapeBindings.end())
      return defv;
    return it->second;
  };
  int64_t bm = getBind("MMA_BM", 64);
  int64_t bn = getBind("MMA_BN", 16);
  int64_t bk = getBind("MMA_BK", 32);
  bool bTranspose = getBind("MMA_B_TRANSPOSE", 0) != 0;
  bool asyncCopyRequested = getBind("MMA_ASYNC_COPY", 0) != 0;

  if (bm <= 0 || bn <= 0 || bk <= 0) {
    ctx.module.emitError("ai_bench_matmul: invalid MMA_BM/MMA_BN/MMA_BK");
    return mlir::failure();
  }
  if ((bm % 16) != 0 || (bn % 16) != 0 || (bk % 8) != 0) {
    ctx.module.emitError("ai_bench_matmul: requires BM%16==0 BN%16==0 BK%8==0");
    return mlir::failure();
  }
  if ((M % bm) != 0 || (N % bn) != 0 || (K % bk) != 0 || (K % 8) != 0) {
    ctx.module.emitError("ai_bench_matmul: requires divisibility by MMA tiles");
    return mlir::failure();
  }
  int64_t warpsM = bm / 16;
  int64_t warpsN = bn / 16;
  int64_t warps = warpsM * warpsN;
  int64_t threads = warps * 32;
  if (warps <= 0 || warps > 32 || threads <= 0 || threads > 1024) {
    ctx.module.emitError("ai_bench_matmul: invalid warps/threads");
    return mlir::failure();
  }

  clearModuleBody(ctx.module);

  // Ensure the module is treated as a GPU container module and has a target triple.
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  // GPU module + kernel.
  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  // Types.
  auto f32 = b.getF32Type();
  auto globalMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);

  // Optional static shared tiles for async-copy MMA path (v2).
  mlir::MemRefType shATy;
  mlir::MemRefType shBTy;
  std::string shA0Name, shA1Name, shB0Name, shB1Name;
  int64_t tileA4 = 0;
  int64_t tileB4 = 0;
  if (asyncCopyRequested) {
    // Double-buffered static shared memory footprint:
    // bytes = 2*(BM*BK*4) + 2*(BK*BN*4) = 8*BK*(BM+BN).
    int64_t staticSharedBytes = 8 * bk * (bm + bn);
    if (staticSharedBytes > (48 * 1024)) {
      ctx.module.emitError("ai_bench_matmul: matmul_mma_tf32_v2 requires static_shared_bytes<=49152 (48KiB)");
      return mlir::failure();
    }
    bool vecCopy = (bk % 4) == 0 && (bn % 4) == 0 && ((bm * bk) % 4) == 0 && ((bk * bn) % 4) == 0;
    tileA4 = vecCopy ? ((bm * bk) / 4) : 0;
    tileB4 = vecCopy ? ((bk * bn) / 4) : 0;
    if (!vecCopy || tileA4 <= 0 || tileB4 <= 0 || (tileA4 % threads) != 0 || (tileB4 % threads) != 0) {
      ctx.module.emitError("ai_bench_matmul: matmul_mma_tf32_v2 requires vectorized async copy eligibility");
      return mlir::failure();
    }

    auto sharedMemSpace =
        mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);
    shATy = mlir::MemRefType::get({bm, bk}, f32,
                                  mlir::MemRefLayoutAttrInterface{},
                                  sharedMemSpace);
    shBTy = mlir::MemRefType::get({bk, bn}, f32,
                                  mlir::MemRefLayoutAttrInterface{},
                                  sharedMemSpace);
    shA0Name = "__intentir_sh_a0_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
    shA1Name = "__intentir_sh_a1_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
    shB0Name = "__intentir_sh_b0_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
    shB1Name = "__intentir_sh_b1_" + sanitizeSymbolName(ctx.kernelName) + "_f32";

    auto align16 = b.getI64IntegerAttr(16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shA0Name, b.getStringAttr("private"), shATy,
        /*initial_value=*/{}, /*constant=*/false, align16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shA1Name, b.getStringAttr("private"), shATy,
        /*initial_value=*/{}, /*constant=*/false, align16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shB0Name, b.getStringAttr("private"), shBTy,
        /*initial_value=*/{}, /*constant=*/false, align16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shB1Name, b.getStringAttr("private"), shBTy,
        /*initial_value=*/{}, /*constant=*/false, align16);
  }

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule,
                                               sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  // Map args.
  auto A = getArgByName(ctx, fn, aName);
  auto Bv = getArgByName(ctx, fn, bName);
  auto C = getArgByName(ctx, fn, outName);
  if (!A || !Bv || !C) {
    ctx.module.emitError("ai_bench_matmul: failed to map kernel args");
    return mlir::failure();
  }

  auto a2Ty = mlir::MemRefType::get({M, K}, f32,
                                    mlir::MemRefLayoutAttrInterface{},
                                    globalMemSpace);
  auto b2Ty = mlir::MemRefType::get({K, N}, f32,
                                    mlir::MemRefLayoutAttrInterface{},
                                    globalMemSpace);
  auto c2Ty = mlir::MemRefType::get({M, N}, f32,
                                    mlir::MemRefLayoutAttrInterface{},
                                    globalMemSpace);

  // Reinterpret 1D memrefs as 2D matrices.
  auto A2 = mlir::memref::ReinterpretCastOp::create(b, loc, a2Ty, A, 0, {M, K},
                                                    {K, 1})
                .getResult();
  auto B2 = mlir::memref::ReinterpretCastOp::create(b, loc, b2Ty, Bv, 0, {K, N},
                                                    {N, 1})
                .getResult();
  auto C2 = mlir::memref::ReinterpretCastOp::create(b, loc, c2Ty, C, 0, {M, N},
                                                    {N, 1})
                .getResult();

  // Thread and block ids.
  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto bidX = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
  auto bidY = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::y);

  auto c16 = makeIndexConst(b, loc, 16);
  auto c32 = makeIndexConst(b, loc, 32);
  auto cBM = makeIndexConst(b, loc, bm);
  auto cBN = makeIndexConst(b, loc, bn);
  auto cBK = makeIndexConst(b, loc, bk);
  auto cThreads = makeIndexConst(b, loc, threads);
  auto cWarpsN = makeIndexConst(b, loc, warpsN);
  auto c0f = makeF32Const(b, loc, 0.0f);

  // Compute warp tile coordinates.
  auto row0 = b.create<mlir::arith::MulIOp>(loc, bidY, cBM);
  auto col0 = b.create<mlir::arith::MulIOp>(loc, bidX, cBN);
  auto warp = b.create<mlir::arith::DivUIOp>(loc, tid, c32);
  auto warpM = b.create<mlir::arith::DivUIOp>(loc, warp, cWarpsN);
  auto warpN = b.create<mlir::arith::RemUIOp>(loc, warp, cWarpsN);
  auto rowW = b.create<mlir::arith::MulIOp>(loc, warpM, c16);
  auto colW = b.create<mlir::arith::MulIOp>(loc, warpN, c16);
  auto gm = b.create<mlir::arith::AddIOp>(loc, row0, rowW);
  auto gn = b.create<mlir::arith::AddIOp>(loc, col0, colW);

  // MMA types.
  auto aFragTy = mlir::gpu::MMAMatrixType::get({16, 8}, f32, "AOp");
  auto bFragTy = mlir::gpu::MMAMatrixType::get({8, 16}, f32, "BOp");
  auto cFragTy = mlir::gpu::MMAMatrixType::get({16, 16}, f32, "COp");

  // Accumulator init.
  auto acc = mlir::gpu::SubgroupMmaConstantMatrixOp::create(b, loc, cFragTy,
                                                           c0f)
                 .getResult();

  auto ldK = b.getIndexAttr(K);
  auto ldN = b.getIndexAttr(N);
  mlir::UnitAttr transposeAttr = bTranspose ? mlir::UnitAttr::get(mlirCtx)
                                            : mlir::UnitAttr();

  std::string kernelKind = "matmul_mma_tf32_global_v1";
  if (asyncCopyRequested) {
    kernelKind = "matmul_mma_tf32_v2";

    // Double-buffered static shared tiles.
    auto As0 =
        mlir::memref::GetGlobalOp::create(b, loc, shATy, shA0Name).getResult();
    auto As1 =
        mlir::memref::GetGlobalOp::create(b, loc, shATy, shA1Name).getResult();
    auto Bs0 =
        mlir::memref::GetGlobalOp::create(b, loc, shBTy, shB0Name).getResult();
    auto Bs1 =
        mlir::memref::GetGlobalOp::create(b, loc, shBTy, shB1Name).getResult();

    auto c4 = makeIndexConst(b, loc, 4);
    int64_t aIters = tileA4 / threads;
    int64_t bIters = tileB4 / threads;
    auto dstElements4 = b.getIndexAttr(4);

    auto emitTile = [&](int64_t kbBase, mlir::Value As, mlir::Value Bs) -> mlir::Value {
      auto kbC = makeIndexConst(b, loc, kbBase);
      llvm::SmallVector<mlir::Value, 16> cpTokens;
      cpTokens.reserve(static_cast<size_t>(aIters + bIters));

      // Copy A tile: each async copies vector<4xf32> (dstElements=4).
      for (int64_t it = 0; it < aIters; ++it) {
        mlir::Value idx = tid;
        if (it != 0) {
          auto off = makeIndexConst(b, loc, it * threads);
          idx = b.create<mlir::arith::AddIOp>(loc, tid, off);
        }
        auto idx4 = b.create<mlir::arith::MulIOp>(loc, idx, c4);
        auto r = b.create<mlir::arith::DivUIOp>(loc, idx4, cBK);
        auto c = b.create<mlir::arith::RemUIOp>(loc, idx4, cBK);
        auto gr = b.create<mlir::arith::AddIOp>(loc, row0, r);
        auto gk = b.create<mlir::arith::AddIOp>(loc, kbC, c);
        auto cp = b.create<mlir::nvgpu::DeviceAsyncCopyOp>(
            loc,
            /*dst=*/As,
            /*dstIndices=*/mlir::ValueRange{r, c},
            /*src=*/A2,
            /*srcIndices=*/mlir::ValueRange{gr, gk},
            /*dstElements=*/dstElements4,
            /*srcElements=*/mlir::Value(),
            /*bypassL1=*/mlir::UnitAttr());
        cpTokens.push_back(cp.getAsyncToken());
      }

      // Copy B tile: each async copies vector<4xf32> (dstElements=4).
      for (int64_t it = 0; it < bIters; ++it) {
        mlir::Value idx = tid;
        if (it != 0) {
          auto off = makeIndexConst(b, loc, it * threads);
          idx = b.create<mlir::arith::AddIOp>(loc, tid, off);
        }
        auto idx4 = b.create<mlir::arith::MulIOp>(loc, idx, c4);
        auto r = b.create<mlir::arith::DivUIOp>(loc, idx4, cBN);
        auto c = b.create<mlir::arith::RemUIOp>(loc, idx4, cBN);
        auto gk = b.create<mlir::arith::AddIOp>(loc, kbC, r);
        auto gn4 = b.create<mlir::arith::AddIOp>(loc, col0, c);
        auto cp = b.create<mlir::nvgpu::DeviceAsyncCopyOp>(
            loc,
            /*dst=*/Bs,
            /*dstIndices=*/mlir::ValueRange{r, c},
            /*src=*/B2,
            /*srcIndices=*/mlir::ValueRange{gk, gn4},
            /*dstElements=*/dstElements4,
            /*srcElements=*/mlir::Value(),
            /*bypassL1=*/mlir::UnitAttr());
        cpTokens.push_back(cp.getAsyncToken());
      }

      return b.create<mlir::nvgpu::DeviceAsyncCreateGroupOp>(loc, cpTokens)
          .getAsyncToken();
    };

    // Preload first tile into buffer0.
    auto group0 = emitTile(/*kbBase=*/0, As0, Bs0);
    b.create<mlir::nvgpu::DeviceAsyncWaitOp>(loc, group0, mlir::IntegerAttr());
    b.create<mlir::gpu::BarrierOp>(loc);

    auto ldBK = b.getIndexAttr(bk);
    auto ldBN = b.getIndexAttr(bn);

    // Main pipelined loop (unrolled).
    int64_t idx = 0;
    for (int64_t kb = 0; kb < K; kb += bk, ++idx) {
      mlir::Value curAs = (idx % 2) == 0 ? As0 : As1;
      mlir::Value curBs = (idx % 2) == 0 ? Bs0 : Bs1;

      bool hasNext = (kb + bk) < K;
      mlir::Value nextGroup;
      if (hasNext) {
        mlir::Value nextAs = (idx % 2) == 0 ? As1 : As0;
        mlir::Value nextBs = (idx % 2) == 0 ? Bs1 : Bs0;
        nextGroup = emitTile(kb + bk, nextAs, nextBs);
      }

      for (int64_t kk = 0; kk < bk; kk += 8) {
        auto kkC = makeIndexConst(b, loc, kk);
        auto aFrag = mlir::gpu::SubgroupMmaLoadMatrixOp::create(
                         b, loc, aFragTy, curAs,
                         mlir::ValueRange{rowW, kkC}, ldBK,
                         /*transpose=*/{})
                         .getResult();
        auto bFrag = mlir::gpu::SubgroupMmaLoadMatrixOp::create(
                         b, loc, bFragTy, curBs,
                         mlir::ValueRange{kkC, colW}, ldBN,
                         transposeAttr)
                         .getResult();
        acc = mlir::gpu::SubgroupMmaComputeOp::create(
                  b, loc, cFragTy, aFrag, bFrag, acc,
                  /*a_transpose=*/{}, /*b_transpose=*/transposeAttr)
                  .getResult();
      }

      if (hasNext) {
        b.create<mlir::nvgpu::DeviceAsyncWaitOp>(loc, nextGroup, mlir::IntegerAttr());
        b.create<mlir::gpu::BarrierOp>(loc);
      }
    }
  } else {
    // Unrolled KB/KK loops (global-load WMMA path).
    for (int64_t kb = 0; kb < K; kb += bk) {
      auto kbC = makeIndexConst(b, loc, kb);

      for (int64_t kk = 0; kk < bk; kk += 8) {
        auto kkC = makeIndexConst(b, loc, kk);
        auto kIdx = b.create<mlir::arith::AddIOp>(loc, kbC, kkC);
        auto aFrag =
            mlir::gpu::SubgroupMmaLoadMatrixOp::create(b, loc, aFragTy, A2,
                                                       mlir::ValueRange{gm, kIdx},
                                                       ldK, /*transpose=*/{})
                .getResult();
        auto bFrag =
            mlir::gpu::SubgroupMmaLoadMatrixOp::create(b, loc, bFragTy, B2,
                                                       mlir::ValueRange{kIdx, gn},
                                                       ldN, transposeAttr)
                .getResult();
        acc = mlir::gpu::SubgroupMmaComputeOp::create(
                  b, loc, cFragTy, aFrag, bFrag, acc,
                  /*a_transpose=*/{}, /*b_transpose=*/transposeAttr)
                  .getResult();
      }
    }
  }

  mlir::gpu::SubgroupMmaStoreMatrixOp::create(b, loc, acc, C2,
                                             mlir::ValueRange{gm, gn}, ldN,
                                             /*transpose=*/{});
  b.create<mlir::gpu::ReturnOp>(loc);

  // Annotate for audit (also mirrored into python meta by the driver).
  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, kernelKind));

  const int64_t gridX = N / bn;
  const int64_t gridY = M / bm;
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = kernelKind;
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/gridX, /*gy=*/gridY, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(M * N);
    llvm::json::Object cfg;
    cfg["BM"] = static_cast<int64_t>(bm);
    cfg["BN"] = static_cast<int64_t>(bn);
    cfg["BK"] = static_cast<int64_t>(bk);
    cfg["mma"] = "tf32";
    cfg["pipeline"] = asyncCopyRequested ? "cp_async_double_buffer" : "global_load";
    cfg["b_transpose"] = static_cast<bool>(bTranspose);
    meta["cuda_real_mlir_matmul_cfg"] = std::move(cfg);
  });

  return mlir::success();
}

static mlir::LogicalResult lowerCudaMatmulFusedEpilogue2dMmaTF32V1(LoweringContext &ctx) {
  if (ctx.outputs.size() != 1) {
    ctx.module.emitError("matmul_fused_epilogue2d: expected single output");
    return mlir::failure();
  }
  std::string outName = ctx.outputs[0];

  // Find the matmul inputs (ignore intermediate names; we lower to final output).
  std::string aName, bName;
  for (const auto &op : ctx.ops) {
    if (op.op != "matmul")
      continue;
    if (op.inputs.size() != 2)
      continue;
    aName = op.inputs[0];
    bName = op.inputs[1];
    break;
  }
  if (aName.empty() || bName.empty()) {
    ctx.module.emitError("matmul_fused_epilogue2d: expected matmul(A,B,...) op");
    return mlir::failure();
  }

  if (ctx.tensors.find(aName) == ctx.tensors.end() ||
      ctx.tensors.find(bName) == ctx.tensors.end() ||
      ctx.tensors.find(outName) == ctx.tensors.end()) {
    ctx.module.emitError("matmul_fused_epilogue2d: missing tensor specs for A/B/out");
    return mlir::failure();
  }

  auto shapeAOr = resolveShape(ctx.tensors[aName], ctx.shapeBindings);
  auto shapeBOr = resolveShape(ctx.tensors[bName], ctx.shapeBindings);
  auto shapeOutOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  if (mlir::failed(shapeAOr) || mlir::failed(shapeBOr) || mlir::failed(shapeOutOr)) {
    ctx.module.emitError("matmul_fused_epilogue2d: failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeAOr->size() != 2 || shapeBOr->size() != 2 || shapeOutOr->size() != 2) {
    ctx.module.emitError("matmul_fused_epilogue2d: expected rank-2 A/B/out tensors");
    return mlir::failure();
  }
  int64_t M = (*shapeAOr)[0];
  int64_t K = (*shapeAOr)[1];
  int64_t K2 = (*shapeBOr)[0];
  int64_t N = (*shapeBOr)[1];
  if (K != K2) {
    ctx.module.emitError("matmul_fused_epilogue2d: A.K != B.K");
    return mlir::failure();
  }
  if ((*shapeOutOr)[0] != M || (*shapeOutOr)[1] != N) {
    ctx.module.emitError("matmul_fused_epilogue2d: out shape mismatch");
    return mlir::failure();
  }
  if (M <= 0 || N <= 0 || K <= 0) {
    ctx.module.emitError("matmul_fused_epilogue2d: invalid dims");
    return mlir::failure();
  }

  // Infer bias (f32 [N]) and masks (bool/i1 [M], [N]) from external inputs.
  std::string biasName;
  std::string rowMaskName;
  std::string colMaskName;
  for (const auto &nm : ctx.argOrder) {
    if (nm == outName || nm == aName || nm == bName)
      continue;
    auto it = ctx.tensors.find(nm);
    if (it == ctx.tensors.end())
      continue;
    auto shpOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shpOr))
      continue;
    llvm::StringRef dt = llvm::StringRef(it->second.dtype).trim().lower();
    if (dt == "f32" && shpOr->size() == 1 && (*shpOr)[0] == N && biasName.empty()) {
      biasName = nm;
      continue;
    }
    if ((dt == "bool" || dt == "i1" || dt == "i8") && shpOr->size() == 1) {
      if ((*shpOr)[0] == M && rowMaskName.empty()) {
        rowMaskName = nm;
        continue;
      }
      if ((*shpOr)[0] == N && colMaskName.empty()) {
        colMaskName = nm;
        continue;
      }
    }
  }
  if (biasName.empty() || rowMaskName.empty() || colMaskName.empty()) {
    ctx.module.emitError("matmul_fused_epilogue2d: failed to infer Bias/RowMask/ColMask inputs");
    return mlir::failure();
  }

  // Tile params (TF32 MMA baseline).
  auto getBind = [&](llvm::StringRef key, int64_t defv) -> int64_t {
    auto it = ctx.shapeBindings.find(key.str());
    if (it == ctx.shapeBindings.end())
      return defv;
    return it->second;
  };
  int64_t bm = getBind("MMA_BM", 32);
  int64_t bn = getBind("MMA_BN", 32);
  int64_t bk = getBind("MMA_BK", 32);
  bool asyncCopyRequested = getBind("MMA_ASYNC_COPY", 0) != 0;

  if (bm <= 0 || bn <= 0 || bk <= 0) {
    ctx.module.emitError("matmul_fused_epilogue2d: invalid MMA_BM/MMA_BN/MMA_BK");
    return mlir::failure();
  }
  if ((bm % 16) != 0 || (bn % 16) != 0 || (bk % 8) != 0) {
    ctx.module.emitError("matmul_fused_epilogue2d: requires BM%16==0 BN%16==0 BK%8==0");
    return mlir::failure();
  }
  if ((M % bm) != 0 || (N % bn) != 0 || (K % bk) != 0 || (K % 8) != 0) {
    ctx.module.emitError("matmul_fused_epilogue2d: requires divisibility by MMA tiles");
    return mlir::failure();
  }
  int64_t warpsM = bm / 16;
  int64_t warpsN = bn / 16;
  int64_t warps = warpsM * warpsN;
  int64_t threads = warps * 32;
  if (warps <= 0 || warps > 32 || threads <= 0 || threads > 1024) {
    ctx.module.emitError("matmul_fused_epilogue2d: invalid warps/threads");
    return mlir::failure();
  }

  // dtypes
  if (llvm::StringRef(ctx.tensors[aName].dtype).trim().lower() != "f32" ||
      llvm::StringRef(ctx.tensors[bName].dtype).trim().lower() != "f32" ||
      llvm::StringRef(ctx.tensors[outName].dtype).trim().lower() != "f32" ||
      llvm::StringRef(ctx.tensors[biasName].dtype).trim().lower() != "f32") {
    ctx.module.emitError("matmul_fused_epilogue2d: expected f32 A/B/out/Bias tensors");
    return mlir::failure();
  }

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  clearModuleBody(ctx.module);

  // Ensure the module is treated as a GPU container module and has a target triple.
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  // GPU module + kernel.
  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto globalMemSpace =
      mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);

  // Optional static shared tiles for async-copy MMA path (v2).
  mlir::MemRefType shATy;
  mlir::MemRefType shBTy;
  std::string shA0Name, shA1Name, shB0Name, shB1Name;
  int64_t tileA4 = 0;
  int64_t tileB4 = 0;
  if (asyncCopyRequested) {
    int64_t staticSharedBytes = 8 * bk * (bm + bn);
    if (staticSharedBytes > (48 * 1024)) {
      ctx.module.emitError(
          "matmul_fused_epilogue2d: v2 requires static_shared_bytes<=49152 (48KiB)");
      return mlir::failure();
    }
    bool vecCopy = (bk % 4) == 0 && (bn % 4) == 0 && ((bm * bk) % 4) == 0 &&
                   ((bk * bn) % 4) == 0;
    tileA4 = vecCopy ? ((bm * bk) / 4) : 0;
    tileB4 = vecCopy ? ((bk * bn) / 4) : 0;
    if (!vecCopy || tileA4 <= 0 || tileB4 <= 0 || (tileA4 % threads) != 0 ||
        (tileB4 % threads) != 0) {
      ctx.module.emitError(
          "matmul_fused_epilogue2d: v2 requires vectorized async copy eligibility");
      return mlir::failure();
    }

    auto sharedMemSpace =
        mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);
    shATy = mlir::MemRefType::get({bm, bk}, f32,
                                  mlir::MemRefLayoutAttrInterface{},
                                  sharedMemSpace);
    shBTy = mlir::MemRefType::get({bk, bn}, f32,
                                  mlir::MemRefLayoutAttrInterface{},
                                  sharedMemSpace);
    shA0Name =
        "__intentir_sh_a0_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
    shA1Name =
        "__intentir_sh_a1_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
    shB0Name =
        "__intentir_sh_b0_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
    shB1Name =
        "__intentir_sh_b1_" + sanitizeSymbolName(ctx.kernelName) + "_f32";

    auto align16 = b.getI64IntegerAttr(16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shA0Name, b.getStringAttr("private"), shATy,
        /*initial_value=*/{}, /*constant=*/false, align16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shA1Name, b.getStringAttr("private"), shATy,
        /*initial_value=*/{}, /*constant=*/false, align16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shB0Name, b.getStringAttr("private"), shBTy,
        /*initial_value=*/{}, /*constant=*/false, align16);
    (void)mlir::memref::GlobalOp::create(
        b, loc, shB1Name, b.getStringAttr("private"), shBTy,
        /*initial_value=*/{}, /*constant=*/false, align16);
  }

  // Shared accumulator tile for fused epilogue: Cs[BM,BN].
  auto sharedMemSpace =
      mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);
  auto shCTy = mlir::MemRefType::get({bm, bn}, f32,
                                     mlir::MemRefLayoutAttrInterface{},
                                     sharedMemSpace);
  std::string shCName =
      "__intentir_sh_c_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
  (void)mlir::memref::GlobalOp::create(
      b, loc, shCName, b.getStringAttr("private"), shCTy,
      /*initial_value=*/{}, /*constant=*/false, b.getI64IntegerAttr(16));

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule,
                                               sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  // Map args.
  auto A = getArgByName(ctx, fn, aName);
  auto Bv = getArgByName(ctx, fn, bName);
  auto Bias = getArgByName(ctx, fn, biasName);
  auto RowMask = getArgByName(ctx, fn, rowMaskName);
  auto ColMask = getArgByName(ctx, fn, colMaskName);
  auto Out = getArgByName(ctx, fn, outName);
  if (!A || !Bv || !Bias || !RowMask || !ColMask || !Out) {
    ctx.module.emitError("matmul_fused_epilogue2d: failed to map kernel args");
    return mlir::failure();
  }

  auto a2Ty = mlir::MemRefType::get({M, K}, f32,
                                    mlir::MemRefLayoutAttrInterface{},
                                    globalMemSpace);
  auto b2Ty = mlir::MemRefType::get({K, N}, f32,
                                    mlir::MemRefLayoutAttrInterface{},
                                    globalMemSpace);
  auto out2Ty = mlir::MemRefType::get({M, N}, f32,
                                      mlir::MemRefLayoutAttrInterface{},
                                      globalMemSpace);

  auto A2 = mlir::memref::ReinterpretCastOp::create(b, loc, a2Ty, A, 0, {M, K},
                                                    {K, 1})
                .getResult();
  auto B2 = mlir::memref::ReinterpretCastOp::create(b, loc, b2Ty, Bv, 0, {K, N},
                                                    {N, 1})
                .getResult();
  auto Out2 = mlir::memref::ReinterpretCastOp::create(b, loc, out2Ty, Out, 0,
                                                      {M, N}, {N, 1})
                  .getResult();

  // Thread and block ids.
  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto bidX = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);
  auto bidY = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::y);

  auto c4 = makeIndexConst(b, loc, 4);
  auto c16 = makeIndexConst(b, loc, 16);
  auto c32 = makeIndexConst(b, loc, 32);
  auto cBM = makeIndexConst(b, loc, bm);
  auto cBN = makeIndexConst(b, loc, bn);
  auto cBK = makeIndexConst(b, loc, bk);
  auto cThreads = makeIndexConst(b, loc, threads);
  auto cWarpsN = makeIndexConst(b, loc, warpsN);
  auto c0f = makeF32Const(b, loc, 0.0f);

  // Compute warp tile coordinates.
  auto row0 = b.create<mlir::arith::MulIOp>(loc, bidY, cBM);
  auto col0 = b.create<mlir::arith::MulIOp>(loc, bidX, cBN);
  auto warp = b.create<mlir::arith::DivUIOp>(loc, tid, c32);
  auto warpM = b.create<mlir::arith::DivUIOp>(loc, warp, cWarpsN);
  auto warpN = b.create<mlir::arith::RemUIOp>(loc, warp, cWarpsN);
  auto rowW = b.create<mlir::arith::MulIOp>(loc, warpM, c16);
  auto colW = b.create<mlir::arith::MulIOp>(loc, warpN, c16);
  auto gm = b.create<mlir::arith::AddIOp>(loc, row0, rowW);
  auto gn = b.create<mlir::arith::AddIOp>(loc, col0, colW);

  // MMA types.
  auto aFragTy = mlir::gpu::MMAMatrixType::get({16, 8}, f32, "AOp");
  auto bFragTy = mlir::gpu::MMAMatrixType::get({8, 16}, f32, "BOp");
  auto cFragTy = mlir::gpu::MMAMatrixType::get({16, 16}, f32, "COp");

  auto acc = mlir::gpu::SubgroupMmaConstantMatrixOp::create(b, loc, cFragTy, c0f)
                 .getResult();

  std::string kernelKind = "matmul_fused_epilogue_mma_tf32_global_v1";
  if (asyncCopyRequested) {
    kernelKind = "matmul_fused_epilogue_mma_tf32_v2";

    auto As0 =
        mlir::memref::GetGlobalOp::create(b, loc, shATy, shA0Name).getResult();
    auto As1 =
        mlir::memref::GetGlobalOp::create(b, loc, shATy, shA1Name).getResult();
    auto Bs0 =
        mlir::memref::GetGlobalOp::create(b, loc, shBTy, shB0Name).getResult();
    auto Bs1 =
        mlir::memref::GetGlobalOp::create(b, loc, shBTy, shB1Name).getResult();

    int64_t aIters = tileA4 / threads;
    int64_t bIters = tileB4 / threads;
    auto dstElements4 = b.getIndexAttr(4);

    auto emitTile = [&](int64_t kbBase, mlir::Value As,
                        mlir::Value Bs) -> mlir::Value {
      auto kbC = makeIndexConst(b, loc, kbBase);
      llvm::SmallVector<mlir::Value, 16> cpTokens;
      cpTokens.reserve(static_cast<size_t>(aIters + bIters));

      for (int64_t it = 0; it < aIters; ++it) {
        mlir::Value idx = tid;
        if (it != 0) {
          auto off = makeIndexConst(b, loc, it * threads);
          idx = b.create<mlir::arith::AddIOp>(loc, tid, off);
        }
        auto idx4 = b.create<mlir::arith::MulIOp>(loc, idx, c4);
        auto r = b.create<mlir::arith::DivUIOp>(loc, idx4, cBK);
        auto c = b.create<mlir::arith::RemUIOp>(loc, idx4, cBK);
        auto gr = b.create<mlir::arith::AddIOp>(loc, row0, r);
        auto gk = b.create<mlir::arith::AddIOp>(loc, kbC, c);
        auto cp = b.create<mlir::nvgpu::DeviceAsyncCopyOp>(
            loc,
            /*dst=*/As,
            /*dstIndices=*/mlir::ValueRange{r, c},
            /*src=*/A2,
            /*srcIndices=*/mlir::ValueRange{gr, gk},
            /*dstElements=*/dstElements4,
            /*srcElements=*/mlir::Value(),
            /*bypassL1=*/mlir::UnitAttr());
        cpTokens.push_back(cp.getAsyncToken());
      }

      for (int64_t it = 0; it < bIters; ++it) {
        mlir::Value idx = tid;
        if (it != 0) {
          auto off = makeIndexConst(b, loc, it * threads);
          idx = b.create<mlir::arith::AddIOp>(loc, tid, off);
        }
        auto idx4 = b.create<mlir::arith::MulIOp>(loc, idx, c4);
        auto r = b.create<mlir::arith::DivUIOp>(loc, idx4, cBN);
        auto c = b.create<mlir::arith::RemUIOp>(loc, idx4, cBN);
        auto gk = b.create<mlir::arith::AddIOp>(loc, kbC, r);
        auto gn4 = b.create<mlir::arith::AddIOp>(loc, col0, c);
        auto cp = b.create<mlir::nvgpu::DeviceAsyncCopyOp>(
            loc,
            /*dst=*/Bs,
            /*dstIndices=*/mlir::ValueRange{r, c},
            /*src=*/B2,
            /*srcIndices=*/mlir::ValueRange{gk, gn4},
            /*dstElements=*/dstElements4,
            /*srcElements=*/mlir::Value(),
            /*bypassL1=*/mlir::UnitAttr());
        cpTokens.push_back(cp.getAsyncToken());
      }

      return b.create<mlir::nvgpu::DeviceAsyncCreateGroupOp>(loc, cpTokens)
          .getAsyncToken();
    };

    auto group0 = emitTile(/*kbBase=*/0, As0, Bs0);
    b.create<mlir::nvgpu::DeviceAsyncWaitOp>(loc, group0, mlir::IntegerAttr());
    b.create<mlir::gpu::BarrierOp>(loc);

    auto ldBK = b.getIndexAttr(bk);
    auto ldBN = b.getIndexAttr(bn);

    int64_t idx = 0;
    for (int64_t kb = 0; kb < K; kb += bk, ++idx) {
      mlir::Value curAs = (idx % 2) == 0 ? As0 : As1;
      mlir::Value curBs = (idx % 2) == 0 ? Bs0 : Bs1;

      bool hasNext = (kb + bk) < K;
      mlir::Value nextGroup;
      if (hasNext) {
        mlir::Value nextAs = (idx % 2) == 0 ? As1 : As0;
        mlir::Value nextBs = (idx % 2) == 0 ? Bs1 : Bs0;
        nextGroup = emitTile(kb + bk, nextAs, nextBs);
      }

      for (int64_t kk = 0; kk < bk; kk += 8) {
        auto kkC = makeIndexConst(b, loc, kk);
        auto aFrag =
            mlir::gpu::SubgroupMmaLoadMatrixOp::create(b, loc, aFragTy, curAs,
                                                      mlir::ValueRange{rowW, kkC}, ldBK,
                                                      /*transpose=*/{})
                .getResult();
        auto bFrag =
            mlir::gpu::SubgroupMmaLoadMatrixOp::create(b, loc, bFragTy, curBs,
                                                      mlir::ValueRange{kkC, colW}, ldBN,
                                                      /*transpose=*/{})
                .getResult();
        acc = mlir::gpu::SubgroupMmaComputeOp::create(b, loc, cFragTy, aFrag,
                                                     bFrag, acc,
                                                     /*a_transpose=*/{},
                                                     /*b_transpose=*/{})
                  .getResult();
      }

      if (hasNext) {
        b.create<mlir::nvgpu::DeviceAsyncWaitOp>(loc, nextGroup,
                                                mlir::IntegerAttr());
        b.create<mlir::gpu::BarrierOp>(loc);
      }
    }
  } else {
    auto ldK = b.getIndexAttr(K);
    auto ldN = b.getIndexAttr(N);
    for (int64_t kb = 0; kb < K; kb += bk) {
      auto kbC = makeIndexConst(b, loc, kb);
      for (int64_t kk = 0; kk < bk; kk += 8) {
        auto kkC = makeIndexConst(b, loc, kk);
        auto kIdx = b.create<mlir::arith::AddIOp>(loc, kbC, kkC);
        auto aFrag =
            mlir::gpu::SubgroupMmaLoadMatrixOp::create(b, loc, aFragTy, A2,
                                                      mlir::ValueRange{gm, kIdx}, ldK,
                                                      /*transpose=*/{})
                .getResult();
        auto bFrag =
            mlir::gpu::SubgroupMmaLoadMatrixOp::create(b, loc, bFragTy, B2,
                                                      mlir::ValueRange{kIdx, gn}, ldN,
                                                      /*transpose=*/{})
                .getResult();
        acc = mlir::gpu::SubgroupMmaComputeOp::create(b, loc, cFragTy, aFrag,
                                                     bFrag, acc,
                                                     /*a_transpose=*/{},
                                                     /*b_transpose=*/{})
                  .getResult();
      }
    }
  }

  // Fused epilogue: acc -> shared Cs -> apply bias + row/col masks -> store Out2.
  auto Cs = mlir::memref::GetGlobalOp::create(b, loc, shCTy, shCName).getResult();
  mlir::gpu::SubgroupMmaStoreMatrixOp::create(b, loc, acc, Cs,
                                             mlir::ValueRange{rowW, colW},
                                             b.getIndexAttr(bn),
                                             /*transpose=*/{});
  b.create<mlir::gpu::BarrierOp>(loc);

  int64_t tileC = bm * bn;
  auto cTileC = makeIndexConst(b, loc, tileC);
  auto forOp = b.create<mlir::scf::ForOp>(loc, tid, cTileC, cThreads);
  b.setInsertionPointToStart(forOp.getBody());
  auto t = forOp.getInductionVar();
  auto tR = b.create<mlir::arith::DivUIOp>(loc, t, cBN);
  auto tC = b.create<mlir::arith::RemUIOp>(loc, t, cBN);
  auto gmE = b.create<mlir::arith::AddIOp>(loc, row0, tR);
  auto gnE = b.create<mlir::arith::AddIOp>(loc, col0, tC);

  auto val0 = b.create<mlir::memref::LoadOp>(loc, Cs, mlir::ValueRange{tR, tC}).getResult();
  auto bias = b.create<mlir::memref::LoadOp>(loc, Bias, mlir::ValueRange{gnE}).getResult();
  auto val1 = b.create<mlir::arith::AddFOp>(loc, val0, bias).getResult();

  auto rm = b.create<mlir::memref::LoadOp>(loc, RowMask, mlir::ValueRange{gmE}).getResult();
  auto cm = b.create<mlir::memref::LoadOp>(loc, ColMask, mlir::ValueRange{gnE}).getResult();
  auto c0i8 = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8);
  auto rmOk =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, rm, c0i8)
          .getResult();
  auto cmOk =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, cm, c0i8)
          .getResult();
  auto cond = b.create<mlir::arith::AndIOp>(loc, rmOk, cmOk).getResult();
  auto val2 = b.create<mlir::arith::SelectOp>(loc, cond, val1, c0f).getResult();
  b.create<mlir::memref::StoreOp>(loc, val2, Out2, mlir::ValueRange{gmE, gnE});

  b.setInsertionPointAfter(forOp);
  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, kernelKind));

  const int64_t gridX = N / bn;
  const int64_t gridY = M / bm;
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = kernelKind;
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/gridX, /*gy=*/gridY, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(M * N);
    llvm::json::Object cfg;
    cfg["BM"] = static_cast<int64_t>(bm);
    cfg["BN"] = static_cast<int64_t>(bn);
    cfg["BK"] = static_cast<int64_t>(bk);
    cfg["mma"] = "tf32";
    cfg["pipeline"] = asyncCopyRequested ? "cp_async_double_buffer" : "global_load";
    cfg["epilogue"] = "bias_rowmask_colmask";
    meta["cuda_real_mlir_matmul_cfg"] = std::move(cfg);
  });
  return mlir::success();
}

static mlir::LogicalResult lowerCudaAttn2dCausalSoftmaxWarpV1(LoweringContext &ctx,
                                                              llvm::StringRef kernelKind) {
  // Specialized causal attention for triton-native 2D kernels:
  // Q:[Q_CTX,HD], K/V:[KV_CTX,HD], Out:[Q_CTX,HD], sm_scale:[]
  //
  // One CTA per query row (grid_x = Q_CTX), one warp (block_x=32).
  // Each lane owns 1 or 2 output columns (d=lane and d=lane+32) and uses
  // warp shuffle XOR for dot all-reduce and one-pass online softmax.

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  const std::string qName = "Q";
  const std::string kName = "K";
  const std::string vName = "V";
  const std::string scaleName = "sm_scale";
  const std::string outName = "Out";
  if (ctx.tensors.find(qName) == ctx.tensors.end() || ctx.tensors.find(kName) == ctx.tensors.end() ||
      ctx.tensors.find(vName) == ctx.tensors.end() || ctx.tensors.find(outName) == ctx.tensors.end()) {
    ctx.module.emitError("attn2d: missing tensor specs for Q/K/V/Out");
    return mlir::failure();
  }

  auto shapeQOr = resolveShape(ctx.tensors[qName], ctx.shapeBindings);
  auto shapeKOr = resolveShape(ctx.tensors[kName], ctx.shapeBindings);
  auto shapeVOr = resolveShape(ctx.tensors[vName], ctx.shapeBindings);
  auto shapeOOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  if (mlir::failed(shapeQOr) || mlir::failed(shapeKOr) || mlir::failed(shapeVOr) ||
      mlir::failed(shapeOOr)) {
    ctx.module.emitError("attn2d: failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeQOr->size() != 2 || shapeKOr->size() != 2 || shapeVOr->size() != 2 ||
      shapeOOr->size() != 2) {
    ctx.module.emitError("attn2d: expected rank-2 tensors");
    return mlir::failure();
  }
  const int64_t Q = (*shapeQOr)[0];
  const int64_t HD = (*shapeQOr)[1];
  const int64_t KV = (*shapeKOr)[0];
  const int64_t HD2 = (*shapeKOr)[1];
  if (KV != (*shapeVOr)[0] || HD2 != (*shapeVOr)[1]) {
    ctx.module.emitError("attn2d: K/V shape mismatch");
    return mlir::failure();
  }
  if ((*shapeOOr)[0] != Q || (*shapeOOr)[1] != HD) {
    ctx.module.emitError("attn2d: Out shape mismatch");
    return mlir::failure();
  }
  if (Q <= 0 || KV <= 0 || HD <= 0) {
    ctx.module.emitError("attn2d: invalid dims");
    return mlir::failure();
  }
  if (HD > 64) {
    ctx.module.emitError("attn2d: HEAD_DIM>64 not supported by warp kernel");
    return mlir::failure();
  }

  clearModuleBody(ctx.module);

  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto globalMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);
  const int64_t threads = 32;

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto QArg = getArgByName(ctx, fn, qName);
  auto KArg = getArgByName(ctx, fn, kName);
  auto VArg = getArgByName(ctx, fn, vName);
  auto SArg = getArgByName(ctx, fn, scaleName);
  auto OutArg = getArgByName(ctx, fn, outName);
  if (!QArg || !KArg || !VArg || !SArg || !OutArg) {
    ctx.module.emitError("attn2d: failed to map kernel args");
    return mlir::failure();
  }

  auto qTy =
      mlir::MemRefType::get({Q, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto kvTy =
      mlir::MemRefType::get({KV, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto outTy =
      mlir::MemRefType::get({Q, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto Q2 = mlir::memref::ReinterpretCastOp::create(b, loc, qTy, QArg, 0, {Q, HD}, {HD, 1})
                .getResult();
  auto K2 = mlir::memref::ReinterpretCastOp::create(b, loc, kvTy, KArg, 0, {KV, HD}, {HD, 1})
                .getResult();
  auto V2 = mlir::memref::ReinterpretCastOp::create(b, loc, kvTy, VArg, 0, {KV, HD}, {HD, 1})
                .getResult();
  auto Out2 =
      mlir::memref::ReinterpretCastOp::create(b, loc, outTy, OutArg, 0, {Q, HD}, {HD, 1})
          .getResult();

  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto qRow = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto cKV = makeIndexConst(b, loc, KV);
  auto c32 = makeIndexConst(b, loc, 32);
  auto cHD = makeIndexConst(b, loc, HD);
  auto c0f = makeF32Const(b, loc, 0.0f);
  auto c1f = makeF32Const(b, loc, 1.0f);
  auto negInf = makeF32Const(b, loc, -std::numeric_limits<float>::infinity());

  auto d0 = tid;
  auto d1 = b.create<mlir::arith::AddIOp>(loc, tid, c32).getResult();
  auto d0In =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, d0, cHD).getResult();
  auto d1In =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, d1, cHD).getResult();

  auto scale = b.create<mlir::memref::LoadOp>(loc, SArg, mlir::ValueRange{c0}).getResult();

  auto kvFor = b.create<mlir::scf::ForOp>(loc, c0, cKV, c1,
                                         mlir::ValueRange{negInf, c0f, c0f, c0f});
  b.setInsertionPointToStart(kvFor.getBody());
  auto kv = kvFor.getInductionVar();
  auto m = kvFor.getRegionIterArgs()[0];
  auto l = kvFor.getRegionIterArgs()[1];
  auto out0 = kvFor.getRegionIterArgs()[2];
  auto out1 = kvFor.getRegionIterArgs()[3];

  // Dot partial (two columns per lane).
  mlir::Value partial = c0f;
  auto if0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d0In, /*withElse=*/true);
  b.setInsertionPointToStart(&if0.getThenRegion().front());
  auto q0 = b.create<mlir::memref::LoadOp>(loc, Q2, mlir::ValueRange{qRow, d0}).getResult();
  auto k0 = b.create<mlir::memref::LoadOp>(loc, K2, mlir::ValueRange{kv, d0}).getResult();
  b.create<mlir::scf::YieldOp>(
      loc, mlir::ValueRange{b.create<mlir::arith::MulFOp>(loc, q0, k0).getResult()});
  b.setInsertionPointToStart(&if0.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(if0);
  partial = b.create<mlir::arith::AddFOp>(loc, partial, if0.getResult(0)).getResult();

  auto if1 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d1In, /*withElse=*/true);
  b.setInsertionPointToStart(&if1.getThenRegion().front());
  auto q1 = b.create<mlir::memref::LoadOp>(loc, Q2, mlir::ValueRange{qRow, d1}).getResult();
  auto k1 = b.create<mlir::memref::LoadOp>(loc, K2, mlir::ValueRange{kv, d1}).getResult();
  b.create<mlir::scf::YieldOp>(
      loc, mlir::ValueRange{b.create<mlir::arith::MulFOp>(loc, q1, k1).getResult()});
  b.setInsertionPointToStart(&if1.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(if1);
  partial = b.create<mlir::arith::AddFOp>(loc, partial, if1.getResult(0)).getResult();

  auto dot = warpAllReduceSumF32(b, loc, partial);
  auto score = b.create<mlir::arith::MulFOp>(loc, dot, scale).getResult();

  // Causal mask: kv > qRow -> -inf.
  auto masked =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, kv, qRow).getResult();
  auto scoreMasked = b.create<mlir::arith::SelectOp>(loc, masked, negInf, score).getResult();

  // Compute the online softmax scalars once per warp (lane 0), then broadcast.
  auto mNewLocal = b.create<mlir::arith::MaximumFOp>(loc, m, scoreMasked).getResult();
  auto isLane0 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tid, c0).getResult();
  auto alphaPIf =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32, f32, f32, f32}, isLane0,
                               /*withElse=*/true);
  b.setInsertionPointToStart(&alphaPIf.getThenRegion().front());
  auto alpha0 = b.create<mlir::math::ExpOp>(
                    loc, b.create<mlir::arith::SubFOp>(loc, m, mNewLocal).getResult())
                    .getResult();
  auto p0 = b.create<mlir::math::ExpOp>(
                loc, b.create<mlir::arith::SubFOp>(loc, scoreMasked, mNewLocal).getResult())
                .getResult();
  auto lNew0 =
      b.create<mlir::arith::AddFOp>(loc, b.create<mlir::arith::MulFOp>(loc, l, alpha0).getResult(),
                                   p0)
          .getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{mNewLocal, alpha0, p0, lNew0});
  b.setInsertionPointToStart(&alphaPIf.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{mNewLocal, c0f, c0f, l});
  b.setInsertionPointAfter(alphaPIf);

  auto lane0 = makeI32Const(b, loc, 0);
  auto width32 = makeI32Const(b, loc, 32);
  auto mNew = b.create<mlir::gpu::ShuffleOp>(loc, alphaPIf.getResult(0), lane0, width32,
                                            mlir::gpu::ShuffleMode::IDX)
                  .getResult(0);
  auto alpha = b.create<mlir::gpu::ShuffleOp>(loc, alphaPIf.getResult(1), lane0, width32,
                                             mlir::gpu::ShuffleMode::IDX)
                   .getResult(0);
  auto p = b.create<mlir::gpu::ShuffleOp>(loc, alphaPIf.getResult(2), lane0, width32,
                                         mlir::gpu::ShuffleMode::IDX)
               .getResult(0);
  auto lNew = b.create<mlir::gpu::ShuffleOp>(loc, alphaPIf.getResult(3), lane0, width32,
                                            mlir::gpu::ShuffleMode::IDX)
                  .getResult(0);

  auto out0Scaled = b.create<mlir::arith::MulFOp>(loc, out0, alpha).getResult();
  auto out1Scaled = b.create<mlir::arith::MulFOp>(loc, out1, alpha).getResult();

  auto ifV0 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d0In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifV0.getThenRegion().front());
  auto v0 = b.create<mlir::memref::LoadOp>(loc, V2, mlir::ValueRange{kv, d0}).getResult();
  b.create<mlir::scf::YieldOp>(
      loc,
      mlir::ValueRange{b.create<mlir::arith::AddFOp>(
                            loc, out0Scaled, b.create<mlir::arith::MulFOp>(loc, p, v0).getResult())
                            .getResult()});
  b.setInsertionPointToStart(&ifV0.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{out0Scaled});
  b.setInsertionPointAfter(ifV0);
  auto out0New = ifV0.getResult(0);

  auto ifV1 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d1In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifV1.getThenRegion().front());
  auto v1 = b.create<mlir::memref::LoadOp>(loc, V2, mlir::ValueRange{kv, d1}).getResult();
  b.create<mlir::scf::YieldOp>(
      loc,
      mlir::ValueRange{b.create<mlir::arith::AddFOp>(
                            loc, out1Scaled, b.create<mlir::arith::MulFOp>(loc, p, v1).getResult())
                            .getResult()});
  b.setInsertionPointToStart(&ifV1.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{out1Scaled});
  b.setInsertionPointAfter(ifV1);
  auto out1New = ifV1.getResult(0);

  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{mNew, lNew, out0New, out1New});
  b.setInsertionPointAfter(kvFor);
  auto lFinal = kvFor.getResult(1);
  auto out0Final = kvFor.getResult(2);
  auto out1Final = kvFor.getResult(3);

  auto invL = b.create<mlir::arith::DivFOp>(loc, c1f, lFinal).getResult();
  auto y0 = b.create<mlir::arith::MulFOp>(loc, out0Final, invL).getResult();
  auto y1 = b.create<mlir::arith::MulFOp>(loc, out1Final, invL).getResult();

  auto store0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, d0In, /*withElse=*/false);
  b.setInsertionPointToStart(&store0.getThenRegion().front());
  b.create<mlir::memref::StoreOp>(loc, y0, Out2, mlir::ValueRange{qRow, d0});
  b.setInsertionPointAfter(store0);

  auto store1 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, d1In, /*withElse=*/false);
  b.setInsertionPointToStart(&store1.getThenRegion().front());
  b.create<mlir::memref::StoreOp>(loc, y1, Out2, mlir::ValueRange{qRow, d1});
  b.setInsertionPointAfter(store1);

  // Note: launch_override must enforce block_x=32 and grid_x=Q.
  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, kernelKind));
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = kernelKind.str();
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/Q, /*gy=*/1, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(Q * HD);
    llvm::json::Object cfg;
    cfg["block_x"] = static_cast<int64_t>(threads);
    cfg["q_ctx"] = static_cast<int64_t>(Q);
    cfg["kv_ctx"] = static_cast<int64_t>(KV);
    cfg["head_dim"] = static_cast<int64_t>(HD);
    cfg["softmax"] = "online_v1_warp";
    meta["cuda_real_mlir_attention_cfg"] = std::move(cfg);
  });
  return mlir::success();
}

static mlir::LogicalResult lowerCudaAttn2dCausalSoftmaxWarpV2(LoweringContext &ctx,
                                                              llvm::StringRef kernelKind) {
  // Two-pass causal attention for triton-native 2D kernels (stable softmax).
  //
  // Compared to warp_v1 (online softmax), warp_v2 does:
  //   pass1: compute m = max(scores)
  //   pass2: compute weights = exp(scores - m), l = sum(weights), acc = sum(weights * V)
  //
  // This matches the triton-native masked_attention2d structure more closely and
  // reduces per-kv rescaling overhead for small KV_CTX.

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  const std::string qName = "Q";
  const std::string kName = "K";
  const std::string vName = "V";
  const std::string scaleName = "sm_scale";
  const std::string outName = "Out";
  if (ctx.tensors.find(qName) == ctx.tensors.end() || ctx.tensors.find(kName) == ctx.tensors.end() ||
      ctx.tensors.find(vName) == ctx.tensors.end() || ctx.tensors.find(outName) == ctx.tensors.end()) {
    ctx.module.emitError("attn2d(warp_v2): missing tensor specs for Q/K/V/Out");
    return mlir::failure();
  }

  auto shapeQOr = resolveShape(ctx.tensors[qName], ctx.shapeBindings);
  auto shapeKOr = resolveShape(ctx.tensors[kName], ctx.shapeBindings);
  auto shapeVOr = resolveShape(ctx.tensors[vName], ctx.shapeBindings);
  auto shapeOOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  if (mlir::failed(shapeQOr) || mlir::failed(shapeKOr) || mlir::failed(shapeVOr) ||
      mlir::failed(shapeOOr)) {
    ctx.module.emitError("attn2d(warp_v2): failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeQOr->size() != 2 || shapeKOr->size() != 2 || shapeVOr->size() != 2 ||
      shapeOOr->size() != 2) {
    ctx.module.emitError("attn2d(warp_v2): expected rank-2 tensors");
    return mlir::failure();
  }
  const int64_t Q = (*shapeQOr)[0];
  const int64_t HD = (*shapeQOr)[1];
  const int64_t KV = (*shapeKOr)[0];
  const int64_t HD2 = (*shapeKOr)[1];
  if (KV != (*shapeVOr)[0] || HD2 != (*shapeVOr)[1]) {
    ctx.module.emitError("attn2d(warp_v2): K/V shape mismatch");
    return mlir::failure();
  }
  if ((*shapeOOr)[0] != Q || (*shapeOOr)[1] != HD) {
    ctx.module.emitError("attn2d(warp_v2): Out shape mismatch");
    return mlir::failure();
  }
  if (Q <= 0 || KV <= 0 || HD <= 0) {
    ctx.module.emitError("attn2d(warp_v2): invalid dims");
    return mlir::failure();
  }
  if (KV > 64) {
    ctx.module.emitError("attn2d(warp_v2): KV_CTX>64 not supported");
    return mlir::failure();
  }
  if (HD > 64) {
    ctx.module.emitError("attn2d(warp_v2): HEAD_DIM>64 not supported by warp kernel");
    return mlir::failure();
  }

  clearModuleBody(ctx.module);

  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto globalMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);
  const int64_t threads = 32;

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto QArg = getArgByName(ctx, fn, qName);
  auto KArg = getArgByName(ctx, fn, kName);
  auto VArg = getArgByName(ctx, fn, vName);
  auto SArg = getArgByName(ctx, fn, scaleName);
  auto OutArg = getArgByName(ctx, fn, outName);
  if (!QArg || !KArg || !VArg || !SArg || !OutArg) {
    ctx.module.emitError("attn2d(warp_v2): failed to map kernel args");
    return mlir::failure();
  }

  auto qTy =
      mlir::MemRefType::get({Q, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto kvTy =
      mlir::MemRefType::get({KV, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto outTy =
      mlir::MemRefType::get({Q, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto Q2 = mlir::memref::ReinterpretCastOp::create(b, loc, qTy, QArg, 0, {Q, HD}, {HD, 1})
                .getResult();
  auto K2 = mlir::memref::ReinterpretCastOp::create(b, loc, kvTy, KArg, 0, {KV, HD}, {HD, 1})
                .getResult();
  auto V2 = mlir::memref::ReinterpretCastOp::create(b, loc, kvTy, VArg, 0, {KV, HD}, {HD, 1})
                .getResult();
  auto Out2 =
      mlir::memref::ReinterpretCastOp::create(b, loc, outTy, OutArg, 0, {Q, HD}, {HD, 1})
          .getResult();

  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto qRow = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto cKV = makeIndexConst(b, loc, KV);
  auto c32 = makeIndexConst(b, loc, 32);
  auto cHD = makeIndexConst(b, loc, HD);
  auto c0f = makeF32Const(b, loc, 0.0f);
  auto c1f = makeF32Const(b, loc, 1.0f);
  auto negInf = makeF32Const(b, loc, -std::numeric_limits<float>::infinity());
  auto cLOG2E = makeF32Const(b, loc, 1.44269504f);

  auto d0 = tid;
  auto d1 = b.create<mlir::arith::AddIOp>(loc, tid, c32).getResult();
  auto d0In =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, d0, cHD).getResult();
  auto d1In =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, d1, cHD).getResult();

  auto scale = b.create<mlir::memref::LoadOp>(loc, SArg, mlir::ValueRange{c0}).getResult();

  // Load Q once (two columns per lane).
  auto ifQ0 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d0In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifQ0.getThenRegion().front());
  auto q0 = b.create<mlir::memref::LoadOp>(loc, Q2, mlir::ValueRange{qRow, d0}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{q0});
  b.setInsertionPointToStart(&ifQ0.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifQ0);
  auto q0v = ifQ0.getResult(0);

  auto ifQ1 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d1In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifQ1.getThenRegion().front());
  auto q1 = b.create<mlir::memref::LoadOp>(loc, Q2, mlir::ValueRange{qRow, d1}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{q1});
  b.setInsertionPointToStart(&ifQ1.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifQ1);
  auto q1v = ifQ1.getResult(0);

  // Pass 1: m = max(scores).
  auto kvForMax = b.create<mlir::scf::ForOp>(loc, c0, cKV, c1, mlir::ValueRange{negInf});
  b.setInsertionPointToStart(kvForMax.getBody());
  auto kv = kvForMax.getInductionVar();
  auto m = kvForMax.getRegionIterArgs()[0];

  auto ifK0 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d0In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifK0.getThenRegion().front());
  auto k0 = b.create<mlir::memref::LoadOp>(loc, K2, mlir::ValueRange{kv, d0}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{k0});
  b.setInsertionPointToStart(&ifK0.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifK0);

  auto ifK1 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d1In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifK1.getThenRegion().front());
  auto k1 = b.create<mlir::memref::LoadOp>(loc, K2, mlir::ValueRange{kv, d1}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{k1});
  b.setInsertionPointToStart(&ifK1.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifK1);

  auto p0 = b.create<mlir::arith::MulFOp>(loc, q0v, ifK0.getResult(0)).getResult();
  auto p1 = b.create<mlir::arith::MulFOp>(loc, q1v, ifK1.getResult(0)).getResult();
  auto partial = b.create<mlir::arith::AddFOp>(loc, p0, p1).getResult();

  auto dot = warpAllReduceSumF32(b, loc, partial);
  auto score = b.create<mlir::arith::MulFOp>(loc, dot, scale).getResult();

  auto masked =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, kv, qRow).getResult();
  auto scoreMasked = b.create<mlir::arith::SelectOp>(loc, masked, negInf, score).getResult();
  auto mNew = b.create<mlir::arith::MaximumFOp>(loc, m, scoreMasked).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{mNew});
  b.setInsertionPointAfter(kvForMax);
  auto mFinal = kvForMax.getResult(0);

  // Pass 2: weights/sum + acc.
  auto kvFor = b.create<mlir::scf::ForOp>(loc, c0, cKV, c1, mlir::ValueRange{c0f, c0f, c0f});
  b.setInsertionPointToStart(kvFor.getBody());
  auto kv2 = kvFor.getInductionVar();
  auto l = kvFor.getRegionIterArgs()[0];
  auto out0 = kvFor.getRegionIterArgs()[1];
  auto out1 = kvFor.getRegionIterArgs()[2];

  auto ifK20 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d0In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifK20.getThenRegion().front());
  auto kk0 = b.create<mlir::memref::LoadOp>(loc, K2, mlir::ValueRange{kv2, d0}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kk0});
  b.setInsertionPointToStart(&ifK20.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifK20);

  auto ifK21 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d1In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifK21.getThenRegion().front());
  auto kk1 = b.create<mlir::memref::LoadOp>(loc, K2, mlir::ValueRange{kv2, d1}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kk1});
  b.setInsertionPointToStart(&ifK21.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifK21);

  auto pp0 = b.create<mlir::arith::MulFOp>(loc, q0v, ifK20.getResult(0)).getResult();
  auto pp1 = b.create<mlir::arith::MulFOp>(loc, q1v, ifK21.getResult(0)).getResult();
  auto partial2 = b.create<mlir::arith::AddFOp>(loc, pp0, pp1).getResult();
  auto dot2 = warpAllReduceSumF32(b, loc, partial2);
  auto score2 = b.create<mlir::arith::MulFOp>(loc, dot2, scale).getResult();
  auto masked2 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, kv2, qRow).getResult();
  auto scoreMasked2 = b.create<mlir::arith::SelectOp>(loc, masked2, negInf, score2).getResult();
  auto w = b.create<mlir::math::Exp2Op>(
               loc,
               b.create<mlir::arith::MulFOp>(
                   loc, b.create<mlir::arith::SubFOp>(loc, scoreMasked2, mFinal).getResult(), cLOG2E)
                   .getResult())
               .getResult();
  auto lNew2 = b.create<mlir::arith::AddFOp>(loc, l, w).getResult();

  auto ifV0 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d0In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifV0.getThenRegion().front());
  auto v0 = b.create<mlir::memref::LoadOp>(loc, V2, mlir::ValueRange{kv2, d0}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{v0});
  b.setInsertionPointToStart(&ifV0.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifV0);
  auto out0New2 =
      b.create<mlir::arith::AddFOp>(loc, out0, b.create<mlir::arith::MulFOp>(loc, w, ifV0.getResult(0)).getResult())
          .getResult();

  auto ifV1 =
      b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, d1In, /*withElse=*/true);
  b.setInsertionPointToStart(&ifV1.getThenRegion().front());
  auto v1 = b.create<mlir::memref::LoadOp>(loc, V2, mlir::ValueRange{kv2, d1}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{v1});
  b.setInsertionPointToStart(&ifV1.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifV1);
  auto out1New2 =
      b.create<mlir::arith::AddFOp>(loc, out1, b.create<mlir::arith::MulFOp>(loc, w, ifV1.getResult(0)).getResult())
          .getResult();

  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{lNew2, out0New2, out1New2});
  b.setInsertionPointAfter(kvFor);
  auto lFinal = kvFor.getResult(0);
  auto out0Final = kvFor.getResult(1);
  auto out1Final = kvFor.getResult(2);

  auto nz =
      b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, lFinal, c0f).getResult();
  auto lSafe = b.create<mlir::arith::SelectOp>(loc, nz, lFinal, c1f).getResult();
  auto invL = b.create<mlir::arith::DivFOp>(loc, c1f, lSafe).getResult();
  auto y0 = b.create<mlir::arith::MulFOp>(loc, out0Final, invL).getResult();
  auto y1 = b.create<mlir::arith::MulFOp>(loc, out1Final, invL).getResult();

  auto store0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, d0In, /*withElse=*/false);
  b.setInsertionPointToStart(&store0.getThenRegion().front());
  b.create<mlir::memref::StoreOp>(loc, y0, Out2, mlir::ValueRange{qRow, d0});
  b.setInsertionPointAfter(store0);

  auto store1 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, d1In, /*withElse=*/false);
  b.setInsertionPointToStart(&store1.getThenRegion().front());
  b.create<mlir::memref::StoreOp>(loc, y1, Out2, mlir::ValueRange{qRow, d1});
  b.setInsertionPointAfter(store1);

  // Note: launch_override must enforce block_x=32 and grid_x=Q.
  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, kernelKind));
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = kernelKind.str();
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/Q, /*gy=*/1, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(Q * HD);
    llvm::json::Object cfg;
    cfg["block_x"] = static_cast<int64_t>(threads);
    cfg["q_ctx"] = static_cast<int64_t>(Q);
    cfg["kv_ctx"] = static_cast<int64_t>(KV);
    cfg["head_dim"] = static_cast<int64_t>(HD);
    cfg["softmax"] = "two_pass_warp";
    meta["cuda_real_mlir_attention_cfg"] = std::move(cfg);
  });
  return mlir::success();
}

static mlir::LogicalResult lowerCudaMaskedAttention2dHd16KeysV1(LoweringContext &ctx,
                                                                llvm::StringRef kernelKind) {
  // masked_attention2d specialization for canonical tiny shapes:
  // - HEAD_DIM==16
  // - KV_CTX<=32 (one warp handles keys)
  //
  // Thread mapping:
  // - tid in [0..KV): key lane computes score[k] (full dot in-thread), participates in warp softmax.
  // - tid in [0..16): output lane computes Out[d] = sum_k prob[k] * V[k,d]

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  const std::string qName = "Q";
  const std::string kName = "K";
  const std::string vName = "V";
  const std::string scaleName = "sm_scale";
  const std::string outName = "Out";
  if (ctx.tensors.find(qName) == ctx.tensors.end() || ctx.tensors.find(kName) == ctx.tensors.end() ||
      ctx.tensors.find(vName) == ctx.tensors.end() || ctx.tensors.find(outName) == ctx.tensors.end()) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): missing tensor specs for Q/K/V/Out");
    return mlir::failure();
  }

  auto shapeQOr = resolveShape(ctx.tensors[qName], ctx.shapeBindings);
  auto shapeKOr = resolveShape(ctx.tensors[kName], ctx.shapeBindings);
  auto shapeVOr = resolveShape(ctx.tensors[vName], ctx.shapeBindings);
  auto shapeOOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  if (mlir::failed(shapeQOr) || mlir::failed(shapeKOr) || mlir::failed(shapeVOr) ||
      mlir::failed(shapeOOr)) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeQOr->size() != 2 || shapeKOr->size() != 2 || shapeVOr->size() != 2 ||
      shapeOOr->size() != 2) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): expected rank-2 tensors");
    return mlir::failure();
  }
  const int64_t Q = (*shapeQOr)[0];
  const int64_t HD = (*shapeQOr)[1];
  const int64_t KV = (*shapeKOr)[0];
  const int64_t HD2 = (*shapeKOr)[1];
  if (KV != (*shapeVOr)[0] || HD2 != (*shapeVOr)[1]) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): K/V shape mismatch");
    return mlir::failure();
  }
  if ((*shapeOOr)[0] != Q || (*shapeOOr)[1] != HD) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): Out shape mismatch");
    return mlir::failure();
  }
  if (Q <= 0 || KV <= 0 || HD <= 0) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): invalid dims");
    return mlir::failure();
  }
  if (HD != 16) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): requires HEAD_DIM==16");
    return mlir::failure();
  }
  if (KV > 32) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): requires KV_CTX<=32");
    return mlir::failure();
  }

  clearModuleBody(ctx.module);
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto globalMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);
  auto sharedMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);

  // Shared layout: q[16] + probs[32].
  const int64_t qElems = 16;
  const int64_t probBase = qElems;
  const int64_t probElems = 32;
  const int64_t shElems = probBase + probElems;
  auto shTy = mlir::MemRefType::get({shElems}, f32, mlir::MemRefLayoutAttrInterface{}, sharedMemSpace);
  std::string shName = "__intentir_sh_" + sanitizeSymbolName(ctx.kernelName) + "_hd16";
  auto align16 = b.getI64IntegerAttr(16);
  (void)mlir::memref::GlobalOp::create(b, loc, shName, b.getStringAttr("private"), shTy,
                                      /*initial_value=*/{}, /*constant=*/false, align16);

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto QArg = getArgByName(ctx, fn, qName);
  auto KArg = getArgByName(ctx, fn, kName);
  auto VArg = getArgByName(ctx, fn, vName);
  auto SArg = getArgByName(ctx, fn, scaleName);
  auto OutArg = getArgByName(ctx, fn, outName);
  if (!QArg || !KArg || !VArg || !SArg || !OutArg) {
    ctx.module.emitError("masked_attention2d(hd16_keys_v1): failed to map kernel args");
    return mlir::failure();
  }

  auto qTy =
      mlir::MemRefType::get({Q, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto kvTy =
      mlir::MemRefType::get({KV, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto outTy =
      mlir::MemRefType::get({Q, HD}, f32, mlir::MemRefLayoutAttrInterface{}, globalMemSpace);
  auto Q2 = mlir::memref::ReinterpretCastOp::create(b, loc, qTy, QArg, 0, {Q, HD}, {HD, 1})
                .getResult();
  auto K2 = mlir::memref::ReinterpretCastOp::create(b, loc, kvTy, KArg, 0, {KV, HD}, {HD, 1})
                .getResult();
  auto V2 = mlir::memref::ReinterpretCastOp::create(b, loc, kvTy, VArg, 0, {KV, HD}, {HD, 1})
                .getResult();
  auto Out2 =
      mlir::memref::ReinterpretCastOp::create(b, loc, outTy, OutArg, 0, {Q, HD}, {HD, 1})
          .getResult();

  auto Sh = mlir::memref::GetGlobalOp::create(b, loc, shTy, shName).getResult();

  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto qRow = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto c16 = makeIndexConst(b, loc, 16);
  auto cKV = makeIndexConst(b, loc, KV);
  auto cProbBase = makeIndexConst(b, loc, probBase);
  auto c0f = makeF32Const(b, loc, 0.0f);
  auto c1f = makeF32Const(b, loc, 1.0f);
  auto negInf = makeF32Const(b, loc, -3.402823466e+38f);
  auto cLOG2E = makeF32Const(b, loc, 1.44269504f);

  auto scale = b.create<mlir::memref::LoadOp>(loc, SArg, mlir::ValueRange{c0}).getResult();

  // Load q[0..16) into shared (tid<16).
  auto predQ =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, c16).getResult();
  auto ifQ = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predQ, /*withElse=*/false);
  b.setInsertionPointToStart(&ifQ.getThenRegion().front());
  auto qv = b.create<mlir::memref::LoadOp>(loc, Q2, mlir::ValueRange{qRow, tid}).getResult();
  b.create<mlir::memref::StoreOp>(loc, qv, Sh, mlir::ValueRange{tid});
  b.setInsertionPointAfter(ifQ);
  b.create<mlir::gpu::BarrierOp>(loc);

  // tid in [0..KV) computes score for key=tid, else contributes -inf/0.
  auto predKey =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cKV).getResult();
  auto ifScore = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, predKey, /*withElse=*/true);
  b.setInsertionPointToStart(&ifScore.getThenRegion().front());
  auto dotFor = b.create<mlir::scf::ForOp>(loc, c0, c16, c1, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(dotFor.getBody());
  auto d = dotFor.getInductionVar();
  auto acc = dotFor.getRegionIterArgs()[0];
  auto qd = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{d}).getResult();
  auto kd = b.create<mlir::memref::LoadOp>(loc, K2, mlir::ValueRange{tid, d}).getResult();
  auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, b.create<mlir::arith::MulFOp>(loc, qd, kd).getResult())
                  .getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
  b.setInsertionPointAfter(dotFor);
  auto dot = dotFor.getResult(0);
  auto score = b.create<mlir::arith::MulFOp>(loc, dot, scale).getResult();
  auto masked =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, tid, qRow).getResult();
  auto scoreMasked = b.create<mlir::arith::SelectOp>(loc, masked, negInf, score).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{scoreMasked});
  b.setInsertionPointToStart(&ifScore.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{negInf});
  b.setInsertionPointAfter(ifScore);
  auto scoreVal = ifScore.getResult(0);

  auto mx = warpAllReduceMaxF32(b, loc, scoreVal);
  auto w = b.create<mlir::math::Exp2Op>(
               loc,
               b.create<mlir::arith::MulFOp>(
                   loc, b.create<mlir::arith::SubFOp>(loc, scoreVal, mx).getResult(), cLOG2E)
                   .getResult())
               .getResult();
  // Sum weights across warp.
  auto sumW = warpAllReduceSumF32(b, loc, w);
  auto nz =
      b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, sumW, c0f).getResult();
  auto sumSafe = b.create<mlir::arith::SelectOp>(loc, nz, sumW, c1f).getResult();
  auto prob = b.create<mlir::arith::DivFOp>(loc, w, sumSafe).getResult();

  // Store prob to shared at probBase+tid (or 0 for tid>=KV).
  auto ifStoreProb = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predKey, /*withElse=*/false);
  b.setInsertionPointToStart(&ifStoreProb.getThenRegion().front());
  auto pIdx = b.create<mlir::arith::AddIOp>(loc, cProbBase, tid).getResult();
  b.create<mlir::memref::StoreOp>(loc, prob, Sh, mlir::ValueRange{pIdx});
  b.setInsertionPointAfter(ifStoreProb);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Output lanes tid<16 compute Out[qRow, d=tid].
  auto ifOut = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, predQ, /*withElse=*/true);
  b.setInsertionPointToStart(&ifOut.getThenRegion().front());
  auto accFor = b.create<mlir::scf::ForOp>(loc, c0, cKV, c1, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(accFor.getBody());
  auto kv = accFor.getInductionVar();
  auto accO = accFor.getRegionIterArgs()[0];
  auto pIdx2 = b.create<mlir::arith::AddIOp>(loc, cProbBase, kv).getResult();
  auto pv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{pIdx2}).getResult();
  auto vv = b.create<mlir::memref::LoadOp>(loc, V2, mlir::ValueRange{kv, tid}).getResult();
  auto accO2 =
      b.create<mlir::arith::AddFOp>(loc, accO, b.create<mlir::arith::MulFOp>(loc, pv, vv).getResult())
          .getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accO2});
  b.setInsertionPointAfter(accFor);
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accFor.getResult(0)});
  b.setInsertionPointToStart(&ifOut.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(ifOut);

  auto ifStore = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predQ, /*withElse=*/false);
  b.setInsertionPointToStart(&ifStore.getThenRegion().front());
  b.create<mlir::memref::StoreOp>(loc, ifOut.getResult(0), Out2, mlir::ValueRange{qRow, tid});
  b.setInsertionPointAfter(ifStore);

  // Note: launch_override must enforce block_x=32 and grid_x=Q.
  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, kernelKind));
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = kernelKind.str();
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/32, /*by=*/1, /*bz=*/1, /*gx=*/Q, /*gy=*/1, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(Q * HD);
    llvm::json::Object cfg;
    cfg["block_x"] = static_cast<int64_t>(32);
    cfg["q_ctx"] = static_cast<int64_t>(Q);
    cfg["kv_ctx"] = static_cast<int64_t>(KV);
    cfg["head_dim"] = static_cast<int64_t>(HD);
    cfg["softmax"] = "hd16_keys_warp";
    meta["cuda_real_mlir_attention_cfg"] = std::move(cfg);
  });
  return mlir::success();
}

static mlir::LogicalResult lowerCudaFlashAttention2dCausalSoftmaxV6(LoweringContext &ctx,
                                                                    llvm::StringRef kernelKind) {
  // Port of the python real-MLIR "attn2d_causal_softmax_v6" strategy:
  // - one query per CTA (grid_x = Q_CTX)
  // - multi-warp CTA (out_warps=2, score_warps configurable via ATTN_SCORE_WARPS)
  // - shared K/V tiles + online softmax scalars in shared
  //
  // This is intentionally restricted to HEAD_DIM==64 for perf-first parity.

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  const std::string qName = "Q";
  const std::string kName = "K";
  const std::string vName = "V";
  const std::string scaleName = "sm_scale";
  const std::string outName = "Out";
  if (ctx.tensors.find(qName) == ctx.tensors.end() || ctx.tensors.find(kName) == ctx.tensors.end() ||
      ctx.tensors.find(vName) == ctx.tensors.end() || ctx.tensors.find(outName) == ctx.tensors.end()) {
    ctx.module.emitError("flash_attention2d: missing tensor specs for Q/K/V/Out");
    return mlir::failure();
  }

  auto shapeQOr = resolveShape(ctx.tensors[qName], ctx.shapeBindings);
  auto shapeKOr = resolveShape(ctx.tensors[kName], ctx.shapeBindings);
  auto shapeVOr = resolveShape(ctx.tensors[vName], ctx.shapeBindings);
  auto shapeOOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  if (mlir::failed(shapeQOr) || mlir::failed(shapeKOr) || mlir::failed(shapeVOr) ||
      mlir::failed(shapeOOr)) {
    ctx.module.emitError("flash_attention2d: failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeQOr->size() != 2 || shapeKOr->size() != 2 || shapeVOr->size() != 2 ||
      shapeOOr->size() != 2) {
    ctx.module.emitError("flash_attention2d: expected rank-2 tensors");
    return mlir::failure();
  }
  const int64_t Q = (*shapeQOr)[0];
  const int64_t HD = (*shapeQOr)[1];
  const int64_t KV = (*shapeKOr)[0];
  const int64_t HD2 = (*shapeKOr)[1];
  if (KV != (*shapeVOr)[0] || HD2 != (*shapeVOr)[1]) {
    ctx.module.emitError("flash_attention2d: K/V shape mismatch");
    return mlir::failure();
  }
  if ((*shapeOOr)[0] != Q || (*shapeOOr)[1] != HD) {
    ctx.module.emitError("flash_attention2d: Out shape mismatch");
    return mlir::failure();
  }
  if (Q <= 0 || KV <= 0 || HD <= 0) {
    ctx.module.emitError("flash_attention2d: invalid dims");
    return mlir::failure();
  }
  if (HD != 64) {
    ctx.module.emitError("flash_attention2d: attn2d_causal_softmax_v6 requires HEAD_DIM==64");
    return mlir::failure();
  }

  // Tuning hooks (via tuning_db -> shape_bindings).
  int64_t blockKV = 32;
  if (auto it = ctx.shapeBindings.find("ATTN_BLOCK_KV"); it != ctx.shapeBindings.end()) {
    blockKV = static_cast<int64_t>(it->second);
  }
  if (blockKV != 16 && blockKV != 32 && blockKV != 64) {
    ctx.module.emitError("flash_attention2d: ATTN_BLOCK_KV must be 16/32/64");
    return mlir::failure();
  }
  int64_t scoreWarps = 6;
  if (auto it = ctx.shapeBindings.find("ATTN_SCORE_WARPS"); it != ctx.shapeBindings.end()) {
    scoreWarps = static_cast<int64_t>(it->second);
  }
  if (scoreWarps != 2 && scoreWarps != 4 && scoreWarps != 6) {
    scoreWarps = 6;
  }
  const int64_t outWarps = 2;
  const int64_t blockWarps = outWarps + scoreWarps;
  const int64_t threads = blockWarps * 32;
  if (threads <= 0 || threads > 1024) {
    ctx.module.emitError("flash_attention2d: invalid block warps/threads");
    return mlir::failure();
  }

  const int64_t qElems = HD;
  const int64_t tileElems = blockKV * HD;

  bool directKV = false;
  if (auto it = ctx.shapeBindings.find("FLASH_ATTN_DIRECT_GMEM"); it != ctx.shapeBindings.end()) {
    directKV = (it->second != 0);
  }
  bool asyncCopy = false;
  if (auto it = ctx.shapeBindings.find("FLASH_ATTN_ASYNC_COPY"); it != ctx.shapeBindings.end()) {
    asyncCopy = (it->second != 0);
  }
  // Guardrails: async-copy uses vector<4xf32> and assumes a single KV tile (no tail).
  if (asyncCopy) {
    const int64_t tileVec4 = tileElems / 4;
    asyncCopy = (!directKV) && (KV == blockKV) && ((HD % 4) == 0) && ((tileVec4 % threads) == 0);
  }

  // Shared layout:
  // - default: [Q(HD), K_tile(blockKV*HD), V_tile(blockKV*HD), scores(blockKV), weights(blockKV), scalars, scratch]
  // - directKV: [Q(HD), scores(blockKV), weights(blockKV), scalars, scratch]  (K/V read from global on demand)
  int64_t offK = 0;
  int64_t offV = 0;
  int64_t offScores = 0;
  if (!directKV) {
    offK = qElems;
    offV = offK + tileElems;
    offScores = offV + tileElems;
  } else {
    offScores = qElems;
  }
  const int64_t offWeights = offScores + blockKV;
  const int64_t offScalars = offWeights + blockKV;
  const int64_t offM = offScalars;
  const int64_t offL = offScalars + 1;
  const int64_t offAlpha = offScalars + 2;
  // Scratch for per-warp reductions (used by v7 softmax update).
  const int64_t offWarpSumScratch = offScalars + 3;
  const int64_t shElems = offWarpSumScratch + blockWarps;

  clearModuleBody(ctx.module);
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto globalMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);
  auto sharedMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);
  auto shTy = mlir::MemRefType::get({shElems}, f32, mlir::MemRefLayoutAttrInterface{}, sharedMemSpace);
  std::string shName = "__intentir_sh_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
  auto align16 = b.getI64IntegerAttr(16);
  (void)mlir::memref::GlobalOp::create(b, loc, shName, b.getStringAttr("private"), shTy,
                                      /*initial_value=*/{}, /*constant=*/false, align16);

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto QArg = getArgByName(ctx, fn, qName);
  auto KArg = getArgByName(ctx, fn, kName);
  auto VArg = getArgByName(ctx, fn, vName);
  auto SArg = getArgByName(ctx, fn, scaleName);
  auto OutArg = getArgByName(ctx, fn, outName);
  if (!QArg || !KArg || !VArg || !SArg || !OutArg) {
    ctx.module.emitError("flash_attention2d: failed to map kernel args");
    return mlir::failure();
  }

  auto Sh = mlir::memref::GetGlobalOp::create(b, loc, shTy, shName).getResult();

  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto bid = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto c2 = makeIndexConst(b, loc, 2);
  auto c32 = makeIndexConst(b, loc, 32);
  auto c4 = makeIndexConst(b, loc, 4);
  auto cHD = makeIndexConst(b, loc, HD);
  auto cKV = makeIndexConst(b, loc, KV);
  auto cBlockKV = makeIndexConst(b, loc, blockKV);
  auto cThreads = makeIndexConst(b, loc, threads);
  auto cOffK = makeIndexConst(b, loc, offK);
  auto cOffV = makeIndexConst(b, loc, offV);
  auto cScores = makeIndexConst(b, loc, offScores);
  auto cWeights = makeIndexConst(b, loc, offWeights);
  auto cMOff = makeIndexConst(b, loc, offM);
  auto cLOff = makeIndexConst(b, loc, offL);
  auto cAlphaOff = makeIndexConst(b, loc, offAlpha);
  const int64_t numWarpsKV = (blockKV + 31) / 32;
  auto cNumWarpsKV = makeIndexConst(b, loc, numWarpsKV);
  auto cWarpSumScratch = makeIndexConst(b, loc, offWarpSumScratch);

  auto c0f = makeF32Const(b, loc, 0.0f);
  auto c1f = makeF32Const(b, loc, 1.0f);
  auto negInf = makeF32Const(b, loc, -3.402823466e+38f);
  auto cLOG2E = makeF32Const(b, loc, 1.44269504f);

  // lane = tid % 32, warp = tid / 32.
  auto lane = b.create<mlir::arith::RemUIOp>(loc, tid, c32).getResult();
  auto warp = b.create<mlir::arith::DivUIOp>(loc, tid, c32).getResult();
  auto lane2 = b.create<mlir::arith::AddIOp>(loc, lane, c32).getResult();
  auto isLane0 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lane, c0).getResult();

  // Output mapping (2 warps cover dim 0..63).
  auto dim = b.create<mlir::arith::AddIOp>(
                 loc, b.create<mlir::arith::MulIOp>(loc, warp, c32).getResult(), lane)
                 .getResult();
  auto predOutWarp =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, warp, c2).getResult();
  auto predDim =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, dim, cHD).getResult();
  auto predOut = b.create<mlir::arith::AndIOp>(loc, predOutWarp, predDim).getResult();

  // Load sm_scale.
  auto sm = b.create<mlir::memref::LoadOp>(loc, SArg, mlir::ValueRange{c0}).getResult();

  // base_q = bid * HD.
  auto baseQ = b.create<mlir::arith::MulIOp>(loc, bid, cHD).getResult();

  // Cooperative Q load: tid < HD.
  auto predQ = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cHD).getResult();
  auto ifQ = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predQ, /*withElse=*/false);
  b.setInsertionPointToStart(&ifQ.getThenRegion().front());
  auto qIdx = b.create<mlir::arith::AddIOp>(loc, baseQ, tid).getResult();
  auto qv = b.create<mlir::memref::LoadOp>(loc, QArg, mlir::ValueRange{qIdx}).getResult();
  b.create<mlir::memref::StoreOp>(loc, qv, Sh, mlir::ValueRange{tid});
  b.setInsertionPointAfter(ifQ);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Init scalars in shared (thread 0).
  auto isTid0 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tid, c0).getResult();
  auto ifInit = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
  b.setInsertionPointToStart(&ifInit.getThenRegion().front());
  b.create<mlir::memref::StoreOp>(loc, negInf, Sh, mlir::ValueRange{cMOff});
  b.create<mlir::memref::StoreOp>(loc, c0f, Sh, mlir::ValueRange{cLOff});
  b.create<mlir::memref::StoreOp>(loc, c0f, Sh, mlir::ValueRange{cAlphaOff});
  b.setInsertionPointAfter(ifInit);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Outer KV tiling loop, all threads participate (barriers inside).
  auto tileFor = b.create<mlir::scf::ForOp>(loc, c0, cKV, cBlockKV, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(tileFor.getBody());
  auto tile0 = tileFor.getInductionVar();
  auto accIn = tileFor.getRegionIterArgs()[0];

  if (!directKV) {
    // Load K/V tile into shared: i in [0, tileElems).
    if (asyncCopy) {
      // Async-copy vector<4xf32> into shared for the single-tile case.
      const int64_t tileVec4 = tileElems / 4;
    auto c4 = makeIndexConst(b, loc, 4);
    auto dstElements4 = b.getIndexAttr(4);
      const int64_t iters = tileVec4 / threads;
      llvm::SmallVector<mlir::Value, 32> cpTokens;
      cpTokens.reserve(static_cast<size_t>(iters * 2));

      for (int64_t it = 0; it < iters; ++it) {
        mlir::Value idx = tid;
        if (it != 0) {
          auto off = makeIndexConst(b, loc, it * threads);
          idx = b.create<mlir::arith::AddIOp>(loc, tid, off).getResult();
        }
        auto idx4 = b.create<mlir::arith::MulIOp>(loc, idx, c4).getResult();
        auto kvOff = b.create<mlir::arith::DivUIOp>(loc, idx4, cHD).getResult();
        auto d = b.create<mlir::arith::RemUIOp>(loc, idx4, cHD).getResult();
        auto kv = b.create<mlir::arith::AddIOp>(loc, tile0, kvOff).getResult();
        auto base = b.create<mlir::arith::MulIOp>(loc, kv, cHD).getResult();
        auto src = b.create<mlir::arith::AddIOp>(loc, base, d).getResult();
        auto dstK = b.create<mlir::arith::AddIOp>(loc, cOffK, idx4).getResult();
        auto dstV = b.create<mlir::arith::AddIOp>(loc, cOffV, idx4).getResult();

        auto cpK = b.create<mlir::nvgpu::DeviceAsyncCopyOp>(
            loc,
            /*dst=*/Sh,
            /*dstIndices=*/mlir::ValueRange{dstK},
            /*src=*/KArg,
            /*srcIndices=*/mlir::ValueRange{src},
            /*dstElements=*/dstElements4,
            /*srcElements=*/mlir::Value(),
            /*bypassL1=*/mlir::UnitAttr());
        auto cpV = b.create<mlir::nvgpu::DeviceAsyncCopyOp>(
            loc,
            /*dst=*/Sh,
            /*dstIndices=*/mlir::ValueRange{dstV},
            /*src=*/VArg,
            /*srcIndices=*/mlir::ValueRange{src},
            /*dstElements=*/dstElements4,
            /*srcElements=*/mlir::Value(),
            /*bypassL1=*/mlir::UnitAttr());
        cpTokens.push_back(cpK.getAsyncToken());
        cpTokens.push_back(cpV.getAsyncToken());
      }

      auto group = b.create<mlir::nvgpu::DeviceAsyncCreateGroupOp>(loc, cpTokens).getAsyncToken();
      b.create<mlir::nvgpu::DeviceAsyncWaitOp>(loc, group, mlir::IntegerAttr());
      b.create<mlir::gpu::BarrierOp>(loc);
    } else {
      auto cTileElems = makeIndexConst(b, loc, tileElems);
      auto tileLoad = b.create<mlir::scf::ForOp>(loc, tid, cTileElems, cThreads);
      b.setInsertionPointToStart(tileLoad.getBody());
      auto i = tileLoad.getInductionVar();
      auto kvOff = b.create<mlir::arith::DivUIOp>(loc, i, cHD).getResult();
      auto d = b.create<mlir::arith::RemUIOp>(loc, i, cHD).getResult();
      auto kv = b.create<mlir::arith::AddIOp>(loc, tile0, kvOff).getResult();
      auto predKV =
          b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, kv, cKV).getResult();
      auto ifKV = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32, f32}, predKV, /*withElse=*/true);
      b.setInsertionPointToStart(&ifKV.getThenRegion().front());
      auto mulKV = b.create<mlir::arith::MulIOp>(loc, kv, cHD).getResult();
      auto idxKV = b.create<mlir::arith::AddIOp>(loc, mulKV, d).getResult();
      auto kVal = b.create<mlir::memref::LoadOp>(loc, KArg, mlir::ValueRange{idxKV}).getResult();
      auto vVal = b.create<mlir::memref::LoadOp>(loc, VArg, mlir::ValueRange{idxKV}).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kVal, vVal});
      b.setInsertionPointToStart(&ifKV.getElseRegion().front());
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f, c0f});
      b.setInsertionPointAfter(ifKV);
      auto shK = b.create<mlir::arith::AddIOp>(loc, cOffK, i).getResult();
      auto shV = b.create<mlir::arith::AddIOp>(loc, cOffV, i).getResult();
      b.create<mlir::memref::StoreOp>(loc, ifKV.getResult(0), Sh, mlir::ValueRange{shK});
      b.create<mlir::memref::StoreOp>(loc, ifKV.getResult(1), Sh, mlir::ValueRange{shV});
      b.setInsertionPointAfter(tileLoad);
      b.create<mlir::gpu::BarrierOp>(loc);
    }
  }

  // Score warps: warps 2.. compute scores[t2] for this tile.
  auto predScoreWarp =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, warp, c2).getResult();
  auto ifScoreWarp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predScoreWarp,
                                              /*withElse=*/false);
  b.setInsertionPointToStart(&ifScoreWarp.getThenRegion().front());
  auto warpS = b.create<mlir::arith::SubIOp>(loc, warp, c2).getResult();
  auto cScoreWarps = makeIndexConst(b, loc, scoreWarps);
  auto scoreFor = b.create<mlir::scf::ForOp>(loc, warpS, cBlockKV, cScoreWarps);
  b.setInsertionPointToStart(scoreFor.getBody());
  auto t2 = scoreFor.getInductionVar();
  auto kv2 = b.create<mlir::arith::AddIOp>(loc, tile0, t2).getResult();
  auto predKV2 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, kv2, cKV).getResult();
  auto predCausal =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, kv2, bid).getResult();
  auto predAttend = b.create<mlir::arith::AndIOp>(loc, predKV2, predCausal).getResult();
  auto ifAttend = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, predAttend,
                                           /*withElse=*/true);
  b.setInsertionPointToStart(&ifAttend.getThenRegion().front());
  mlir::Value k0, k1;
  if (!directKV) {
    auto base = b.create<mlir::arith::MulIOp>(loc, t2, cHD).getResult();
    auto baseK = b.create<mlir::arith::AddIOp>(loc, cOffK, base).getResult();
    auto idxK0 = b.create<mlir::arith::AddIOp>(loc, baseK, lane).getResult();
    auto idxK1 = b.create<mlir::arith::AddIOp>(loc, baseK, lane2).getResult();
    k0 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxK0}).getResult();
    k1 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxK1}).getResult();
  } else {
    auto base = b.create<mlir::arith::MulIOp>(loc, kv2, cHD).getResult();
    auto idxK0 = b.create<mlir::arith::AddIOp>(loc, base, lane).getResult();
    auto idxK1 = b.create<mlir::arith::AddIOp>(loc, base, lane2).getResult();
    k0 = b.create<mlir::memref::LoadOp>(loc, KArg, mlir::ValueRange{idxK0}).getResult();
    k1 = b.create<mlir::memref::LoadOp>(loc, KArg, mlir::ValueRange{idxK1}).getResult();
  }
  auto q0 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{lane}).getResult();
  auto q1 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{lane2}).getResult();
  auto p0 = b.create<mlir::arith::MulFOp>(loc, q0, k0).getResult();
  auto p1 = b.create<mlir::arith::MulFOp>(loc, q1, k1).getResult();
  auto partial = b.create<mlir::arith::AddFOp>(loc, p0, p1).getResult();
  auto dot = warpAllReduceSumF32(b, loc, partial);
  auto scaled = b.create<mlir::arith::MulFOp>(loc, dot, sm).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{scaled});
  b.setInsertionPointToStart(&ifAttend.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{negInf});
  b.setInsertionPointAfter(ifAttend);
  auto ifLane0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isLane0,
                                          /*withElse=*/false);
  b.setInsertionPointToStart(&ifLane0.getThenRegion().front());
  auto sIdx = b.create<mlir::arith::AddIOp>(loc, cScores, t2).getResult();
  b.create<mlir::memref::StoreOp>(loc, ifAttend.getResult(0), Sh, mlir::ValueRange{sIdx});
  b.setInsertionPointAfter(ifLane0);
  b.setInsertionPointAfter(scoreFor);
  b.setInsertionPointAfter(ifScoreWarp);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Softmax update:
  // - v6: thread0 serial max/sum (simpler, ok on small GPUs)
  // - v7: parallel reductions across threads tid<blockKV (better on large GPUs)
  if (kernelKind == "attn2d_causal_softmax_v7") {
    // Parallel softmax scalar update without full-block reductions: blockKV <= 64
    // so max/sum can be reduced with warp shuffles + a small cross-warp step.
    //
    // This cuts down on gpu.barrier usage (important for small KV tiles on large GPUs).

    auto tidInKV =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cBlockKV).getResult();

    auto warpInKV =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, warp, cNumWarpsKV).getResult();

    // scoreOrNegInf = (tid < blockKV) ? scores[tid] : -inf
    auto ifScore = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, tidInKV, /*withElse=*/true);
    b.setInsertionPointToStart(&ifScore.getThenRegion().front());
    auto sIdx = b.create<mlir::arith::AddIOp>(loc, cScores, tid).getResult();
    auto sv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx}).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sv});
    b.setInsertionPointToStart(&ifScore.getElseRegion().front());
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{negInf});
    b.setInsertionPointAfter(ifScore);
    auto scoreOrNegInf = ifScore.getResult(0);

    // Per-warp max across lanes.
    auto warpMax = warpAllReduceMaxF32(b, loc, scoreOrNegInf);
    auto predStoreWarpMax = b.create<mlir::arith::AndIOp>(loc, isLane0, warpInKV).getResult();
    auto ifStoreWarpMax =
        b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predStoreWarpMax, /*withElse=*/false);
    b.setInsertionPointToStart(&ifStoreWarpMax.getThenRegion().front());
    auto idxWarp = b.create<mlir::arith::AddIOp>(loc, cWeights, warp).getResult();
    b.create<mlir::memref::StoreOp>(loc, warpMax, Sh, mlir::ValueRange{idxWarp});
    b.setInsertionPointAfter(ifStoreWarpMax);
    b.create<mlir::gpu::BarrierOp>(loc);

    // Warp0 reduces across per-warp max values and stores weights[0] = max(scores).
    auto isWarp0 =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, warp, c0).getResult();
    auto laneInWarps =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, lane, cNumWarpsKV).getResult();
    auto ifWarp0Val = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, isWarp0, /*withElse=*/true);
    b.setInsertionPointToStart(&ifWarp0Val.getThenRegion().front());
    auto ifLaneVal =
        b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, laneInWarps, /*withElse=*/true);
    b.setInsertionPointToStart(&ifLaneVal.getThenRegion().front());
    auto idxMax = b.create<mlir::arith::AddIOp>(loc, cWeights, lane).getResult();
    auto mxv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxMax}).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{mxv});
    b.setInsertionPointToStart(&ifLaneVal.getElseRegion().front());
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{negInf});
    b.setInsertionPointAfter(ifLaneVal);
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifLaneVal.getResult(0)});
    b.setInsertionPointToStart(&ifWarp0Val.getElseRegion().front());
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{negInf});
    b.setInsertionPointAfter(ifWarp0Val);
    auto maxScratch = ifWarp0Val.getResult(0);
    auto maxTile = warpAllReduceMaxF32(b, loc, maxScratch);
    auto predStoreMax = b.create<mlir::arith::AndIOp>(loc, isWarp0, isLane0).getResult();
    auto ifStoreMax =
        b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predStoreMax, /*withElse=*/false);
    b.setInsertionPointToStart(&ifStoreMax.getThenRegion().front());
    b.create<mlir::memref::StoreOp>(loc, maxTile, Sh, mlir::ValueRange{cWeights});
    b.setInsertionPointAfter(ifStoreMax);

    // Thread0 computes mNew/alpha and stores scalars.
    auto ifScalar = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
    b.setInsertionPointToStart(&ifScalar.getThenRegion().front());
    auto mPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cMOff}).getResult();
    auto lPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();
    auto mTile = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cWeights}).getResult();
    auto mNew = b.create<mlir::arith::MaximumFOp>(loc, mPrev, mTile).getResult();
    auto alpha = b.create<mlir::math::Exp2Op>(
                     loc,
                     b.create<mlir::arith::MulFOp>(
                         loc, b.create<mlir::arith::SubFOp>(loc, mPrev, mNew).getResult(), cLOG2E)
                         .getResult())
                     .getResult();
    auto lScaled = b.create<mlir::arith::MulFOp>(loc, lPrev, alpha).getResult();
    b.create<mlir::memref::StoreOp>(loc, mNew, Sh, mlir::ValueRange{cMOff});
    b.create<mlir::memref::StoreOp>(loc, lScaled, Sh, mlir::ValueRange{cLOff}); // temp
    b.create<mlir::memref::StoreOp>(loc, alpha, Sh, mlir::ValueRange{cAlphaOff});
    b.setInsertionPointAfter(ifScalar);
    b.create<mlir::gpu::BarrierOp>(loc);

    // Compute weights (tid<blockKV) and reduce sum(weights) via warp shuffles.
    auto ifWeightVal = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, tidInKV, /*withElse=*/true);
    b.setInsertionPointToStart(&ifWeightVal.getThenRegion().front());
    auto mNew2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cMOff}).getResult();
    auto sIdx2 = b.create<mlir::arith::AddIOp>(loc, cScores, tid).getResult();
    auto sv2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx2}).getResult();
    auto w = b.create<mlir::math::Exp2Op>(
                 loc,
                 b.create<mlir::arith::MulFOp>(
                     loc, b.create<mlir::arith::SubFOp>(loc, sv2, mNew2).getResult(), cLOG2E)
                     .getResult())
                 .getResult();
    auto wIdx2 = b.create<mlir::arith::AddIOp>(loc, cWeights, tid).getResult();
    b.create<mlir::memref::StoreOp>(loc, w, Sh, mlir::ValueRange{wIdx2});
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{w});
    b.setInsertionPointToStart(&ifWeightVal.getElseRegion().front());
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
    b.setInsertionPointAfter(ifWeightVal);
    auto wForSum = ifWeightVal.getResult(0);

    auto warpSum = warpAllReduceSumF32(b, loc, wForSum);
    auto predStoreWarpSum = b.create<mlir::arith::AndIOp>(loc, isLane0, warpInKV).getResult();
    auto ifStoreWarpSum =
        b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predStoreWarpSum, /*withElse=*/false);
    b.setInsertionPointToStart(&ifStoreWarpSum.getThenRegion().front());
    auto idxSum = b.create<mlir::arith::AddIOp>(loc, cWarpSumScratch, warp).getResult();
    b.create<mlir::memref::StoreOp>(loc, warpSum, Sh, mlir::ValueRange{idxSum});
    b.setInsertionPointAfter(ifStoreWarpSum);
    // Synchronize so all weights[0..blockKV) are visible before output warps read them.
    b.create<mlir::gpu::BarrierOp>(loc);
  } else {
    // Thread 0: update online softmax scalars and write weights[0..blockKV).
    auto ifSoftmax = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
    b.setInsertionPointToStart(&ifSoftmax.getThenRegion().front());
    auto mPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cMOff}).getResult();
    auto lPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();
    auto maxFor = b.create<mlir::scf::ForOp>(loc, c0, cBlockKV, c1, mlir::ValueRange{negInf});
    b.setInsertionPointToStart(maxFor.getBody());
    auto t = maxFor.getInductionVar();
    auto curMax = maxFor.getRegionIterArgs()[0];
    auto sIdx2 = b.create<mlir::arith::AddIOp>(loc, cScores, t).getResult();
    auto sv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx2}).getResult();
    auto mx = b.create<mlir::arith::MaximumFOp>(loc, curMax, sv).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{mx});
    b.setInsertionPointAfter(maxFor);
    auto mTile = maxFor.getResult(0);
    auto mNew = b.create<mlir::arith::MaximumFOp>(loc, mPrev, mTile).getResult();
    auto alpha = b.create<mlir::math::Exp2Op>(
                     loc,
                     b.create<mlir::arith::MulFOp>(
                         loc, b.create<mlir::arith::SubFOp>(loc, mPrev, mNew).getResult(), cLOG2E)
                         .getResult())
                     .getResult();
    auto lScaled = b.create<mlir::arith::MulFOp>(loc, lPrev, alpha).getResult();
    auto sumFor = b.create<mlir::scf::ForOp>(loc, c0, cBlockKV, c1, mlir::ValueRange{c0f});
    b.setInsertionPointToStart(sumFor.getBody());
    auto tt = sumFor.getInductionVar();
    auto curSum = sumFor.getRegionIterArgs()[0];
    auto sIdx3 = b.create<mlir::arith::AddIOp>(loc, cScores, tt).getResult();
    auto sv2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx3}).getResult();
    auto w = b.create<mlir::math::Exp2Op>(
                 loc,
                 b.create<mlir::arith::MulFOp>(
                     loc, b.create<mlir::arith::SubFOp>(loc, sv2, mNew).getResult(), cLOG2E)
                     .getResult())
                 .getResult();
    auto wIdx = b.create<mlir::arith::AddIOp>(loc, cWeights, tt).getResult();
    b.create<mlir::memref::StoreOp>(loc, w, Sh, mlir::ValueRange{wIdx});
    auto sum2 = b.create<mlir::arith::AddFOp>(loc, curSum, w).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum2});
    b.setInsertionPointAfter(sumFor);
    auto sumP = sumFor.getResult(0);
    auto lNew = b.create<mlir::arith::AddFOp>(loc, lScaled, sumP).getResult();
    b.create<mlir::memref::StoreOp>(loc, mNew, Sh, mlir::ValueRange{cMOff});
    b.create<mlir::memref::StoreOp>(loc, lNew, Sh, mlir::ValueRange{cLOff});
    b.create<mlir::memref::StoreOp>(loc, alpha, Sh, mlir::ValueRange{cAlphaOff});
    b.setInsertionPointAfter(ifSoftmax);
    b.create<mlir::gpu::BarrierOp>(loc);
  }

  // Output warps: accumulate acc = acc*alpha + sum(weights * V_tile).
  auto ifOut = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, predOut, /*withElse=*/true);
  b.setInsertionPointToStart(&ifOut.getThenRegion().front());
  auto alpha2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cAlphaOff}).getResult();
  auto accTileFor = b.create<mlir::scf::ForOp>(loc, c0, cBlockKV, c1, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(accTileFor.getBody());
  auto ttt = accTileFor.getInductionVar();
  auto accTile = accTileFor.getRegionIterArgs()[0];
  auto wIdx2 = b.create<mlir::arith::AddIOp>(loc, cWeights, ttt).getResult();
  auto wv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{wIdx2}).getResult();
  mlir::Value vv;
  if (!directKV) {
    auto baseV = b.create<mlir::arith::MulIOp>(loc, ttt, cHD).getResult();
    auto idxV = b.create<mlir::arith::AddIOp>(
                     loc, b.create<mlir::arith::AddIOp>(loc, cOffV, baseV).getResult(), dim)
                    .getResult();
    vv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxV}).getResult();
  } else {
    auto kv = b.create<mlir::arith::AddIOp>(loc, tile0, ttt).getResult();
    auto base = b.create<mlir::arith::MulIOp>(loc, kv, cHD).getResult();
    auto idxV = b.create<mlir::arith::AddIOp>(loc, base, dim).getResult();
    vv = b.create<mlir::memref::LoadOp>(loc, VArg, mlir::ValueRange{idxV}).getResult();
  }
  auto prod = b.create<mlir::arith::MulFOp>(loc, wv, vv).getResult();
  auto accTile2 = b.create<mlir::arith::AddFOp>(loc, accTile, prod).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accTile2});
  b.setInsertionPointAfter(accTileFor);
  auto tileAcc = accTileFor.getResult(0);
  auto accNext =
      b.create<mlir::arith::AddFOp>(loc, b.create<mlir::arith::MulFOp>(loc, accIn, alpha2).getResult(),
                                   tileAcc)
          .getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accNext});
  b.setInsertionPointToStart(&ifOut.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accIn});
  b.setInsertionPointAfter(ifOut);

  b.create<mlir::gpu::BarrierOp>(loc);

  // v7 only: finalize lNew = lScaled + sum(weights) using the per-warp sums we stored in scratch.
  // This happens after the output stage barrier; the next tile iteration will naturally
  // synchronize at the next tile-load barrier.
  if (kernelKind == "attn2d_causal_softmax_v7") {
    auto isWarp0 =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, warp, c0).getResult();
    auto laneInWarps =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, lane, cNumWarpsKV).getResult();
    auto predRead = b.create<mlir::arith::AndIOp>(loc, isWarp0, laneInWarps).getResult();

    auto ifScratch = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, predRead, /*withElse=*/true);
    b.setInsertionPointToStart(&ifScratch.getThenRegion().front());
    auto idx = b.create<mlir::arith::AddIOp>(loc, cWarpSumScratch, lane).getResult();
    auto v = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idx}).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{v});
    b.setInsertionPointToStart(&ifScratch.getElseRegion().front());
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
    b.setInsertionPointAfter(ifScratch);

    auto sumTile = warpAllReduceSumF32(b, loc, ifScratch.getResult(0));
    auto ifFinal = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
    b.setInsertionPointToStart(&ifFinal.getThenRegion().front());
    auto lScaled = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();
    auto lNew = b.create<mlir::arith::AddFOp>(loc, lScaled, sumTile).getResult();
    b.create<mlir::memref::StoreOp>(loc, lNew, Sh, mlir::ValueRange{cLOff});
    b.setInsertionPointAfter(ifFinal);
  }

  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOut.getResult(0)});
  b.setInsertionPointAfter(tileFor);
  auto accOut = tileFor.getResult(0);

  auto lOut = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();
  auto ifStore = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predOut, /*withElse=*/false);
  b.setInsertionPointToStart(&ifStore.getThenRegion().front());
  auto nz =
      b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, lOut, c0f).getResult();
  auto lSafe = b.create<mlir::arith::SelectOp>(loc, nz, lOut, c1f).getResult();
  auto outv = b.create<mlir::arith::DivFOp>(loc, accOut, lSafe).getResult();
  auto oIdx = b.create<mlir::arith::AddIOp>(loc, baseQ, dim).getResult();
  b.create<mlir::memref::StoreOp>(loc, outv, OutArg, mlir::ValueRange{oIdx});
  b.setInsertionPointAfter(ifStore);

  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, kernelKind));
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = kernelKind.str();
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/Q, /*gy=*/1, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(Q * HD);
    llvm::json::Object cfg;
    cfg["block_x"] = static_cast<int64_t>(threads);
    cfg["block_kv"] = static_cast<int64_t>(blockKV);
    cfg["out_warps"] = static_cast<int64_t>(outWarps);
    cfg["score_warps"] = static_cast<int64_t>(scoreWarps);
    cfg["head_dim"] = static_cast<int64_t>(HD);
    cfg["q_ctx"] = static_cast<int64_t>(Q);
    cfg["kv_ctx"] = static_cast<int64_t>(KV);
    cfg["direct_kv"] = static_cast<bool>(directKV);
    cfg["async_copy"] = static_cast<bool>(asyncCopy);
    cfg["softmax"] =
        (kernelKind == "attn2d_causal_softmax_v7") ? "online_v1_parallel_reduce" : "online_v1_serial_t0";
    meta["cuda_real_mlir_attention_cfg"] = std::move(cfg);
  });
  return mlir::success();
}

static mlir::LogicalResult lowerCudaAttnFwdSoftmaxV6(LoweringContext &ctx, llvm::StringRef kernelKind) {
  // _attn_fwd (triton-native) specialized fast path:
  // Q:[Z,q_numhead,Q_CTX,HD], K/V:[Z,kv_numhead,KV_CTX,HD], Out same as Q.
  //
  // One (z, head, q_row) per CTA (grid_x = Z*q_numhead*Q_CTX) with multi-warp CTA
  // (out_warps=2, score_warps configurable).
  //
  // NOTE: attn_mask is currently a no-op in the intent graph; we ignore it here.

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  const std::string qName = "Q";
  const std::string kName = "K";
  const std::string vName = "V";
  const std::string scaleName = "sm_scale";
  const std::string outName = "Out";

  if (ctx.tensors.find(qName) == ctx.tensors.end() || ctx.tensors.find(kName) == ctx.tensors.end() ||
      ctx.tensors.find(vName) == ctx.tensors.end() || ctx.tensors.find(outName) == ctx.tensors.end()) {
    ctx.module.emitError("_attn_fwd: missing tensor specs for Q/K/V/Out");
    return mlir::failure();
  }

  auto shapeQOr = resolveShape(ctx.tensors[qName], ctx.shapeBindings);
  auto shapeKOr = resolveShape(ctx.tensors[kName], ctx.shapeBindings);
  auto shapeVOr = resolveShape(ctx.tensors[vName], ctx.shapeBindings);
  auto shapeOOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  if (mlir::failed(shapeQOr) || mlir::failed(shapeKOr) || mlir::failed(shapeVOr) ||
      mlir::failed(shapeOOr)) {
    ctx.module.emitError("_attn_fwd: failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeQOr->size() != 4 || shapeKOr->size() != 4 || shapeVOr->size() != 4 ||
      shapeOOr->size() != 4) {
    ctx.module.emitError("_attn_fwd: expected rank-4 tensors");
    return mlir::failure();
  }
  const int64_t Z = (*shapeQOr)[0];
  const int64_t QH = (*shapeQOr)[1];
  const int64_t QCTX = (*shapeQOr)[2];
  const int64_t HD = (*shapeQOr)[3];
  const int64_t Z2 = (*shapeKOr)[0];
  const int64_t KH = (*shapeKOr)[1];
  const int64_t KVCTX = (*shapeKOr)[2];
  const int64_t HD2 = (*shapeKOr)[3];
  if (Z != Z2 || HD != HD2) {
    ctx.module.emitError("_attn_fwd: Q/K shape mismatch (Z/HD)");
    return mlir::failure();
  }
  if (KVCTX != (*shapeVOr)[2] || KH != (*shapeVOr)[1] || Z != (*shapeVOr)[0] || HD != (*shapeVOr)[3]) {
    ctx.module.emitError("_attn_fwd: K/V shape mismatch");
    return mlir::failure();
  }
  if ((*shapeOOr)[0] != Z || (*shapeOOr)[1] != QH || (*shapeOOr)[2] != QCTX || (*shapeOOr)[3] != HD) {
    ctx.module.emitError("_attn_fwd: Out shape mismatch");
    return mlir::failure();
  }
  if (Z <= 0 || QH <= 0 || KH <= 0 || QCTX <= 0 || KVCTX <= 0 || HD <= 0) {
    ctx.module.emitError("_attn_fwd: invalid dims");
    return mlir::failure();
  }
  if (QH != KH) {
    ctx.module.emitError("_attn_fwd: q_numhead != kv_numhead not supported");
    return mlir::failure();
  }
  if (HD != 64) {
    ctx.module.emitError("_attn_fwd: attn_fwd_softmax_v6 requires HEAD_DIM==64");
    return mlir::failure();
  }

  // Tuning hooks (via tuning_db -> shape_bindings).
  int64_t blockKV = 32;
  if (auto it = ctx.shapeBindings.find("ATTN_FWD_BLOCK_KV"); it != ctx.shapeBindings.end()) {
    blockKV = static_cast<int64_t>(it->second);
  } else if (auto it2 = ctx.shapeBindings.find("ATTN_BLOCK_KV"); it2 != ctx.shapeBindings.end()) {
    blockKV = static_cast<int64_t>(it2->second);
  }
  if (blockKV != 16 && blockKV != 32 && blockKV != 64) {
    ctx.module.emitError("_attn_fwd: ATTN_FWD_BLOCK_KV must be 16/32/64");
    return mlir::failure();
  }
  int64_t scoreWarps = 6;
  if (auto it = ctx.shapeBindings.find("ATTN_FWD_SCORE_WARPS"); it != ctx.shapeBindings.end()) {
    scoreWarps = static_cast<int64_t>(it->second);
  } else if (auto it2 = ctx.shapeBindings.find("ATTN_SCORE_WARPS"); it2 != ctx.shapeBindings.end()) {
    scoreWarps = static_cast<int64_t>(it2->second);
  }
  if (scoreWarps != 2 && scoreWarps != 4 && scoreWarps != 6) {
    scoreWarps = 6;
  }
  const int64_t outWarps = 2;
  const int64_t blockWarps = outWarps + scoreWarps;
  const int64_t threads = blockWarps * 32;
  if (threads <= 0 || threads > 1024) {
    ctx.module.emitError("_attn_fwd: invalid block warps/threads");
    return mlir::failure();
  }

  // Shared layout: [Q(HD), K_tile(blockKV*HD), V_tile(blockKV*HD),
  // scores(blockKV), weights(blockKV), scalars(m,l,alpha)].
  const int64_t qElems = HD;
  const int64_t tileElems = blockKV * HD;
  const int64_t offK = qElems;
  const int64_t offV = offK + tileElems;
  const int64_t offScores = offV + tileElems;
  const int64_t offWeights = offScores + blockKV;
  const int64_t offScalars = offWeights + blockKV;
  const int64_t offM = offScalars;
  const int64_t offL = offScalars + 1;
  const int64_t offAlpha = offScalars + 2;
  const int64_t shElems = offScalars + 3;

  clearModuleBody(ctx.module);
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto sharedMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);
  auto shTy = mlir::MemRefType::get({shElems}, f32, mlir::MemRefLayoutAttrInterface{}, sharedMemSpace);
  std::string shName = "__intentir_sh_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
  auto align16 = b.getI64IntegerAttr(16);
  (void)mlir::memref::GlobalOp::create(b, loc, shName, b.getStringAttr("private"), shTy,
                                      /*initial_value=*/{}, /*constant=*/false, align16);

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto QArg = getArgByName(ctx, fn, qName);
  auto KArg = getArgByName(ctx, fn, kName);
  auto VArg = getArgByName(ctx, fn, vName);
  auto SArg = getArgByName(ctx, fn, scaleName);
  auto OutArg = getArgByName(ctx, fn, outName);
  if (!QArg || !KArg || !VArg || !SArg || !OutArg) {
    ctx.module.emitError("_attn_fwd: failed to map kernel args");
    return mlir::failure();
  }

  auto Sh = mlir::memref::GetGlobalOp::create(b, loc, shTy, shName).getResult();

  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto bid = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);

  auto c0 = makeIndexConst(b, loc, 0);
  auto c1 = makeIndexConst(b, loc, 1);
  auto c2 = makeIndexConst(b, loc, 2);
  auto c32 = makeIndexConst(b, loc, 32);
  auto cHD = makeIndexConst(b, loc, HD);
  auto cQCTX = makeIndexConst(b, loc, QCTX);
  auto cKVCTX = makeIndexConst(b, loc, KVCTX);
  auto cBlockKV = makeIndexConst(b, loc, blockKV);
  auto cThreads = makeIndexConst(b, loc, threads);
  auto cOffK = makeIndexConst(b, loc, offK);
  auto cOffV = makeIndexConst(b, loc, offV);
  auto cScores = makeIndexConst(b, loc, offScores);
  auto cWeights = makeIndexConst(b, loc, offWeights);
  auto cMOff = makeIndexConst(b, loc, offM);
  auto cLOff = makeIndexConst(b, loc, offL);
  auto cAlphaOff = makeIndexConst(b, loc, offAlpha);

  auto c0f = makeF32Const(b, loc, 0.0f);
  auto c1f = makeF32Const(b, loc, 1.0f);
  auto negInf = makeF32Const(b, loc, -3.402823466e+38f);
  auto cLOG2E = makeF32Const(b, loc, 1.44269504f);

  // lane = tid % 32, warp = tid / 32.
  auto lane = b.create<mlir::arith::RemUIOp>(loc, tid, c32).getResult();
  auto warp = b.create<mlir::arith::DivUIOp>(loc, tid, c32).getResult();
  auto lane2 = b.create<mlir::arith::AddIOp>(loc, lane, c32).getResult();

  // Output mapping (2 warps cover dim 0..63).
  auto dim = b.create<mlir::arith::AddIOp>(
                 loc, b.create<mlir::arith::MulIOp>(loc, warp, c32).getResult(), lane)
                 .getResult();
  auto predOutWarp =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, warp, c2).getResult();
  auto predDim =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, dim, cHD).getResult();
  auto predOut = b.create<mlir::arith::AndIOp>(loc, predOutWarp, predDim).getResult();

  // Load sm_scale.
  auto sm = b.create<mlir::memref::LoadOp>(loc, SArg, mlir::ValueRange{c0}).getResult();

  // Decode bid -> (z, head, q_row).
  auto qRow = b.create<mlir::arith::RemUIOp>(loc, bid, cQCTX).getResult();
  auto tmp = b.create<mlir::arith::DivUIOp>(loc, bid, cQCTX).getResult();
  auto cQH = makeIndexConst(b, loc, QH);
  auto head = b.create<mlir::arith::RemUIOp>(loc, tmp, cQH).getResult();
  auto z = b.create<mlir::arith::DivUIOp>(loc, tmp, cQH).getResult();

  // baseQ = (((z*QH + head)*QCTX + qRow) * HD).
  auto zh = b.create<mlir::arith::AddIOp>(loc, b.create<mlir::arith::MulIOp>(loc, z, cQH).getResult(), head)
                .getResult();
  auto qBaseRow =
      b.create<mlir::arith::AddIOp>(loc, b.create<mlir::arith::MulIOp>(loc, zh, cQCTX).getResult(), qRow)
          .getResult();
  auto baseQ = b.create<mlir::arith::MulIOp>(loc, qBaseRow, cHD).getResult();

  // baseKV0 = ((z*KH + head) * KVCTX * HD).
  auto kvBaseRow = b.create<mlir::arith::MulIOp>(loc, zh, cKVCTX).getResult();
  auto baseKV0 = b.create<mlir::arith::MulIOp>(loc, kvBaseRow, cHD).getResult();

  // Cooperative Q load: tid < HD.
  auto predQ =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cHD).getResult();
  auto ifQ = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predQ, /*withElse=*/false);
  b.setInsertionPointToStart(&ifQ.getThenRegion().front());
  auto qIdx = b.create<mlir::arith::AddIOp>(loc, baseQ, tid).getResult();
  auto qv = b.create<mlir::memref::LoadOp>(loc, QArg, mlir::ValueRange{qIdx}).getResult();
  b.create<mlir::memref::StoreOp>(loc, qv, Sh, mlir::ValueRange{tid});
  b.setInsertionPointAfter(ifQ);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Init scalars in shared (thread 0).
  auto isTid0 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tid, c0).getResult();
  auto ifInit = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
  b.setInsertionPointToStart(&ifInit.getThenRegion().front());
  b.create<mlir::memref::StoreOp>(loc, negInf, Sh, mlir::ValueRange{cMOff});
  b.create<mlir::memref::StoreOp>(loc, c0f, Sh, mlir::ValueRange{cLOff});
  b.create<mlir::memref::StoreOp>(loc, c0f, Sh, mlir::ValueRange{cAlphaOff});
  b.setInsertionPointAfter(ifInit);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Outer KV tiling loop.
  auto tileFor = b.create<mlir::scf::ForOp>(loc, c0, cKVCTX, cBlockKV, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(tileFor.getBody());
  auto tile0 = tileFor.getInductionVar();
  auto accIn = tileFor.getRegionIterArgs()[0];

  // Load K/V tile into shared: i in [0, tileElems).
  auto cTileElems = makeIndexConst(b, loc, tileElems);
  auto tileLoad = b.create<mlir::scf::ForOp>(loc, tid, cTileElems, cThreads);
  b.setInsertionPointToStart(tileLoad.getBody());
  auto i = tileLoad.getInductionVar();
  auto kvOff = b.create<mlir::arith::DivUIOp>(loc, i, cHD).getResult();
  auto d = b.create<mlir::arith::RemUIOp>(loc, i, cHD).getResult();
  auto kv = b.create<mlir::arith::AddIOp>(loc, tile0, kvOff).getResult();
  auto predKV =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, kv, cKVCTX).getResult();
  auto ifKV = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32, f32}, predKV, /*withElse=*/true);
  b.setInsertionPointToStart(&ifKV.getThenRegion().front());
  auto idxKV = b.create<mlir::arith::AddIOp>(
                   loc, baseKV0,
                   b.create<mlir::arith::AddIOp>(loc, b.create<mlir::arith::MulIOp>(loc, kv, cHD).getResult(), d)
                       .getResult())
                   .getResult();
  auto kVal = b.create<mlir::memref::LoadOp>(loc, KArg, mlir::ValueRange{idxKV}).getResult();
  auto vVal = b.create<mlir::memref::LoadOp>(loc, VArg, mlir::ValueRange{idxKV}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kVal, vVal});
  b.setInsertionPointToStart(&ifKV.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f, c0f});
  b.setInsertionPointAfter(ifKV);
  auto shK = b.create<mlir::arith::AddIOp>(loc, cOffK, i).getResult();
  auto shV = b.create<mlir::arith::AddIOp>(loc, cOffV, i).getResult();
  b.create<mlir::memref::StoreOp>(loc, ifKV.getResult(0), Sh, mlir::ValueRange{shK});
  b.create<mlir::memref::StoreOp>(loc, ifKV.getResult(1), Sh, mlir::ValueRange{shV});
  b.setInsertionPointAfter(tileLoad);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Score warps: warps 2.. compute scores[t2] for this tile.
  auto predScoreWarp =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, warp, c2).getResult();
  auto ifScoreWarp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predScoreWarp,
                                              /*withElse=*/false);
  b.setInsertionPointToStart(&ifScoreWarp.getThenRegion().front());
  auto warpS = b.create<mlir::arith::SubIOp>(loc, warp, c2).getResult();
  auto cScoreWarps = makeIndexConst(b, loc, scoreWarps);
  auto scoreFor = b.create<mlir::scf::ForOp>(loc, warpS, cBlockKV, cScoreWarps);
  b.setInsertionPointToStart(scoreFor.getBody());
  auto t2 = scoreFor.getInductionVar();
  auto kv2 = b.create<mlir::arith::AddIOp>(loc, tile0, t2).getResult();
  auto predKV2 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, kv2, cKVCTX).getResult();
  auto ifAttend = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, predKV2,
                                           /*withElse=*/true);
  b.setInsertionPointToStart(&ifAttend.getThenRegion().front());
  auto base = b.create<mlir::arith::MulIOp>(loc, t2, cHD).getResult();
  auto baseK = b.create<mlir::arith::AddIOp>(loc, cOffK, base).getResult();
  auto idxK0 = b.create<mlir::arith::AddIOp>(loc, baseK, lane).getResult();
  auto idxK1 = b.create<mlir::arith::AddIOp>(loc, baseK, lane2).getResult();
  auto k0 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxK0}).getResult();
  auto k1 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxK1}).getResult();
  auto q0 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{lane}).getResult();
  auto q1 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{lane2}).getResult();
  auto p0 = b.create<mlir::arith::MulFOp>(loc, q0, k0).getResult();
  auto p1 = b.create<mlir::arith::MulFOp>(loc, q1, k1).getResult();
  auto partial = b.create<mlir::arith::AddFOp>(loc, p0, p1).getResult();
  auto dot = warpAllReduceSumF32(b, loc, partial);
  auto scaled = b.create<mlir::arith::MulFOp>(loc, dot, sm).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{scaled});
  b.setInsertionPointToStart(&ifAttend.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{negInf});
  b.setInsertionPointAfter(ifAttend);
  auto isLane0 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lane, c0).getResult();
  auto ifLane0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isLane0,
                                          /*withElse=*/false);
  b.setInsertionPointToStart(&ifLane0.getThenRegion().front());
  auto sIdx = b.create<mlir::arith::AddIOp>(loc, cScores, t2).getResult();
  b.create<mlir::memref::StoreOp>(loc, ifAttend.getResult(0), Sh, mlir::ValueRange{sIdx});
  b.setInsertionPointAfter(ifLane0);
  b.setInsertionPointAfter(scoreFor);
  b.setInsertionPointAfter(ifScoreWarp);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Softmax update:
  // - v6: thread0 serial max/sum
  // - v7: parallel reductions across threads tid<blockKV (better on large GPUs)
  if (kernelKind == "attn_fwd_softmax_v7") {
    auto mPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cMOff}).getResult();
    auto lPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();

    auto tidInKV =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cBlockKV).getResult();
    auto ifInitMax = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, tidInKV, /*withElse=*/false);
    b.setInsertionPointToStart(&ifInitMax.getThenRegion().front());
    auto sIdx = b.create<mlir::arith::AddIOp>(loc, cScores, tid).getResult();
    auto sv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx}).getResult();
    auto wIdx = b.create<mlir::arith::AddIOp>(loc, cWeights, tid).getResult();
    b.create<mlir::memref::StoreOp>(loc, sv, Sh, mlir::ValueRange{wIdx});
    b.setInsertionPointAfter(ifInitMax);
    b.create<mlir::gpu::BarrierOp>(loc);

    for (int64_t stride = blockKV / 2; stride >= 1; stride /= 2) {
      auto cStride = makeIndexConst(b, loc, stride);
      auto pred = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cStride).getResult();
      auto ifRed = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, pred, /*withElse=*/false);
      b.setInsertionPointToStart(&ifRed.getThenRegion().front());
      auto idxA = b.create<mlir::arith::AddIOp>(loc, cWeights, tid).getResult();
      auto idxB = b.create<mlir::arith::AddIOp>(loc, idxA, cStride).getResult();
      auto a = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxA}).getResult();
      auto bval = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxB}).getResult();
      auto mx = b.create<mlir::arith::MaximumFOp>(loc, a, bval).getResult();
      b.create<mlir::memref::StoreOp>(loc, mx, Sh, mlir::ValueRange{idxA});
      b.setInsertionPointAfter(ifRed);
      b.create<mlir::gpu::BarrierOp>(loc);
    }

    auto ifScalar = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
    b.setInsertionPointToStart(&ifScalar.getThenRegion().front());
    auto mTile = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cWeights}).getResult();
    auto mNew = b.create<mlir::arith::MaximumFOp>(loc, mPrev, mTile).getResult();
    auto alpha = b.create<mlir::math::Exp2Op>(
                     loc,
                     b.create<mlir::arith::MulFOp>(
                         loc, b.create<mlir::arith::SubFOp>(loc, mPrev, mNew).getResult(), cLOG2E)
                         .getResult())
                     .getResult();
    auto lScaled = b.create<mlir::arith::MulFOp>(loc, lPrev, alpha).getResult();
    b.create<mlir::memref::StoreOp>(loc, mNew, Sh, mlir::ValueRange{cMOff});
    b.create<mlir::memref::StoreOp>(loc, lScaled, Sh, mlir::ValueRange{cLOff}); // temp
    b.create<mlir::memref::StoreOp>(loc, alpha, Sh, mlir::ValueRange{cAlphaOff});
    b.setInsertionPointAfter(ifScalar);
    b.create<mlir::gpu::BarrierOp>(loc);

    auto ifWeights = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, tidInKV, /*withElse=*/false);
    b.setInsertionPointToStart(&ifWeights.getThenRegion().front());
    auto mNew2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cMOff}).getResult();
    auto sIdx2 = b.create<mlir::arith::AddIOp>(loc, cScores, tid).getResult();
    auto sv2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx2}).getResult();
    auto w = b.create<mlir::math::Exp2Op>(
                 loc,
                 b.create<mlir::arith::MulFOp>(
                     loc, b.create<mlir::arith::SubFOp>(loc, sv2, mNew2).getResult(), cLOG2E)
                     .getResult())
                 .getResult();
    auto wIdx2 = b.create<mlir::arith::AddIOp>(loc, cWeights, tid).getResult();
    b.create<mlir::memref::StoreOp>(loc, w, Sh, mlir::ValueRange{wIdx2});
    b.create<mlir::memref::StoreOp>(loc, w, Sh, mlir::ValueRange{sIdx2}); // sum scratch
    b.setInsertionPointAfter(ifWeights);
    b.create<mlir::gpu::BarrierOp>(loc);

    for (int64_t stride = blockKV / 2; stride >= 1; stride /= 2) {
      auto cStride = makeIndexConst(b, loc, stride);
      auto pred = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cStride).getResult();
      auto ifRed = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, pred, /*withElse=*/false);
      b.setInsertionPointToStart(&ifRed.getThenRegion().front());
      auto idxA = b.create<mlir::arith::AddIOp>(loc, cScores, tid).getResult();
      auto idxB = b.create<mlir::arith::AddIOp>(loc, idxA, cStride).getResult();
      auto a = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxA}).getResult();
      auto bval = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxB}).getResult();
      auto sum = b.create<mlir::arith::AddFOp>(loc, a, bval).getResult();
      b.create<mlir::memref::StoreOp>(loc, sum, Sh, mlir::ValueRange{idxA});
      b.setInsertionPointAfter(ifRed);
      b.create<mlir::gpu::BarrierOp>(loc);
    }

    auto ifFinal = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
    b.setInsertionPointToStart(&ifFinal.getThenRegion().front());
    auto lScaled2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();
    auto sumP = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cScores}).getResult();
    auto lNew = b.create<mlir::arith::AddFOp>(loc, lScaled2, sumP).getResult();
    b.create<mlir::memref::StoreOp>(loc, lNew, Sh, mlir::ValueRange{cLOff});
    b.setInsertionPointAfter(ifFinal);
    b.create<mlir::gpu::BarrierOp>(loc);
  } else {
    // Thread 0: update online softmax scalars and write weights[0..blockKV).
    auto ifSoftmax = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isTid0, /*withElse=*/false);
    b.setInsertionPointToStart(&ifSoftmax.getThenRegion().front());
    auto mPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cMOff}).getResult();
    auto lPrev = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();
    auto maxFor = b.create<mlir::scf::ForOp>(loc, c0, cBlockKV, c1, mlir::ValueRange{negInf});
    b.setInsertionPointToStart(maxFor.getBody());
    auto t = maxFor.getInductionVar();
    auto curMax = maxFor.getRegionIterArgs()[0];
    auto sIdx2 = b.create<mlir::arith::AddIOp>(loc, cScores, t).getResult();
    auto sv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx2}).getResult();
    auto mx = b.create<mlir::arith::MaximumFOp>(loc, curMax, sv).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{mx});
    b.setInsertionPointAfter(maxFor);
    auto mTile = maxFor.getResult(0);
    auto mNew = b.create<mlir::arith::MaximumFOp>(loc, mPrev, mTile).getResult();
    auto alpha = b.create<mlir::math::Exp2Op>(
                     loc,
                     b.create<mlir::arith::MulFOp>(
                         loc, b.create<mlir::arith::SubFOp>(loc, mPrev, mNew).getResult(), cLOG2E)
                         .getResult())
                     .getResult();
    auto lScaled = b.create<mlir::arith::MulFOp>(loc, lPrev, alpha).getResult();
    auto sumFor = b.create<mlir::scf::ForOp>(loc, c0, cBlockKV, c1, mlir::ValueRange{c0f});
    b.setInsertionPointToStart(sumFor.getBody());
    auto tt = sumFor.getInductionVar();
    auto curSum = sumFor.getRegionIterArgs()[0];
    auto sIdx3 = b.create<mlir::arith::AddIOp>(loc, cScores, tt).getResult();
    auto sv2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{sIdx3}).getResult();
    auto w = b.create<mlir::math::Exp2Op>(
                 loc,
                 b.create<mlir::arith::MulFOp>(
                     loc, b.create<mlir::arith::SubFOp>(loc, sv2, mNew).getResult(), cLOG2E)
                     .getResult())
                 .getResult();
    auto wIdx = b.create<mlir::arith::AddIOp>(loc, cWeights, tt).getResult();
    b.create<mlir::memref::StoreOp>(loc, w, Sh, mlir::ValueRange{wIdx});
    auto sum2 = b.create<mlir::arith::AddFOp>(loc, curSum, w).getResult();
    b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum2});
    b.setInsertionPointAfter(sumFor);
    auto sumP = sumFor.getResult(0);
    auto lNew = b.create<mlir::arith::AddFOp>(loc, lScaled, sumP).getResult();
    b.create<mlir::memref::StoreOp>(loc, mNew, Sh, mlir::ValueRange{cMOff});
    b.create<mlir::memref::StoreOp>(loc, lNew, Sh, mlir::ValueRange{cLOff});
    b.create<mlir::memref::StoreOp>(loc, alpha, Sh, mlir::ValueRange{cAlphaOff});
    b.setInsertionPointAfter(ifSoftmax);
    b.create<mlir::gpu::BarrierOp>(loc);
  }

  // Output warps: accumulate acc = acc*alpha + sum(weights * V_tile).
  auto ifOut = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, predOut, /*withElse=*/true);
  b.setInsertionPointToStart(&ifOut.getThenRegion().front());
  auto alpha2 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cAlphaOff}).getResult();
  auto accTileFor = b.create<mlir::scf::ForOp>(loc, c0, cBlockKV, c1, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(accTileFor.getBody());
  auto ttt = accTileFor.getInductionVar();
  auto accTile = accTileFor.getRegionIterArgs()[0];
  auto wIdx2 = b.create<mlir::arith::AddIOp>(loc, cWeights, ttt).getResult();
  auto wv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{wIdx2}).getResult();
  auto baseV = b.create<mlir::arith::MulIOp>(loc, ttt, cHD).getResult();
  auto idxV = b.create<mlir::arith::AddIOp>(
                  loc, b.create<mlir::arith::AddIOp>(loc, cOffV, baseV).getResult(), dim)
                  .getResult();
  auto vv = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{idxV}).getResult();
  auto prod = b.create<mlir::arith::MulFOp>(loc, wv, vv).getResult();
  auto accTile2 = b.create<mlir::arith::AddFOp>(loc, accTile, prod).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accTile2});
  b.setInsertionPointAfter(accTileFor);
  auto tileAcc = accTileFor.getResult(0);
  auto accNext =
      b.create<mlir::arith::AddFOp>(loc, b.create<mlir::arith::MulFOp>(loc, accIn, alpha2).getResult(),
                                   tileAcc)
          .getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accNext});
  b.setInsertionPointToStart(&ifOut.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accIn});
  b.setInsertionPointAfter(ifOut);

  b.create<mlir::gpu::BarrierOp>(loc);
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOut.getResult(0)});
  b.setInsertionPointAfter(tileFor);
  auto accOut = tileFor.getResult(0);

  auto lOut = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{cLOff}).getResult();
  auto ifStore = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, predOut, /*withElse=*/false);
  b.setInsertionPointToStart(&ifStore.getThenRegion().front());
  auto nz =
      b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, lOut, c0f).getResult();
  auto lSafe = b.create<mlir::arith::SelectOp>(loc, nz, lOut, c1f).getResult();
  auto outv = b.create<mlir::arith::DivFOp>(loc, accOut, lSafe).getResult();
  auto oIdx = b.create<mlir::arith::AddIOp>(loc, baseQ, dim).getResult();
  b.create<mlir::memref::StoreOp>(loc, outv, OutArg, mlir::ValueRange{oIdx});
  b.setInsertionPointAfter(ifStore);

  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, kernelKind));
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = kernelKind.str();
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/(Z * QH * QCTX), /*gy=*/1, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(Z * QH * QCTX * HD);
    llvm::json::Object cfg;
    cfg["block_x"] = static_cast<int64_t>(threads);
    cfg["block_kv"] = static_cast<int64_t>(blockKV);
    cfg["out_warps"] = static_cast<int64_t>(outWarps);
    cfg["score_warps"] = static_cast<int64_t>(scoreWarps);
    cfg["head_dim"] = static_cast<int64_t>(HD);
    cfg["q_ctx"] = static_cast<int64_t>(QCTX);
    cfg["kv_ctx"] = static_cast<int64_t>(KVCTX);
    cfg["z"] = static_cast<int64_t>(Z);
    cfg["q_numhead"] = static_cast<int64_t>(QH);
    cfg["kv_numhead"] = static_cast<int64_t>(KH);
    cfg["softmax"] =
        (kernelKind == "attn_fwd_softmax_v7") ? "online_v1_parallel_reduce" : "online_v1_serial_t0";
    cfg["attn_mask"] = "ignored";
    meta["cuda_real_mlir_attention_cfg"] = std::move(cfg);
  });
  return mlir::success();
}

static mlir::LogicalResult lowerCudaRmsNorm2dRowwiseV1(LoweringContext &ctx) {
  // Expected intent-expanded graph:
  //   x_sq = input * input
  //   sum_sq = reduce_sum(x_sq, dim=1)  -> [M]
  //   mean_sq = sum_sq / N_scalar
  //   INV_RMS = rsqrt(mean_sq + eps)    -> [M]
  //   out = input * INV_RMS[:,None] * weight[None,:]  -> [M,N]
  if (ctx.outputs.size() != 2) {
    ctx.module.emitError("rms_norm2d: expected 2 outputs (out, INV_RMS)");
    return mlir::failure();
  }
  if (ctx.tensors.find("eps") == ctx.tensors.end() ||
      ctx.tensors.find("N_scalar") == ctx.tensors.end()) {
    ctx.module.emitError("rms_norm2d: missing required scalar inputs (eps/N_scalar)");
    return mlir::failure();
  }

  std::string outName;
  std::string invName;
  for (const auto &nm : ctx.outputs) {
    auto it = ctx.tensors.find(nm);
    if (it == ctx.tensors.end())
      continue;
    auto shOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shOr))
      continue;
    if (shOr->size() == 2) {
      outName = nm;
      continue;
    }
    if (shOr->size() == 1) {
      invName = nm;
      continue;
    }
  }
  if (outName.empty() || invName.empty()) {
    ctx.module.emitError("rms_norm2d: failed to identify rank-2 out and rank-1 INV_RMS outputs");
    return mlir::failure();
  }

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto shapeOutOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  auto shapeInvOr = resolveShape(ctx.tensors[invName], ctx.shapeBindings);
  if (mlir::failed(shapeOutOr) || mlir::failed(shapeInvOr)) {
    ctx.module.emitError("rms_norm2d: failed to resolve output shapes");
    return mlir::failure();
  }
  int64_t M = (*shapeOutOr)[0];
  int64_t N = (*shapeOutOr)[1];
  if (shapeInvOr->size() != 1 || (*shapeInvOr)[0] != M) {
    ctx.module.emitError("rms_norm2d: INV_RMS must be shape [M]");
    return mlir::failure();
  }

  // Infer input matrix and weight vector names from external inputs.
  std::set<std::string> outSet(ctx.outputs.begin(), ctx.outputs.end());
  std::string inputName;
  std::string weightName;
  for (const auto &nm : ctx.argOrder) {
    if (outSet.count(nm))
      continue;
    auto it = ctx.tensors.find(nm);
    if (it == ctx.tensors.end())
      continue;
    if (llvm::StringRef(it->second.dtype).trim().lower() != "f32")
      continue;
    auto shOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shOr))
      continue;
    if (shOr->size() == 2 && (*shOr)[0] == M && (*shOr)[1] == N && inputName.empty()) {
      inputName = nm;
      continue;
    }
    if (shOr->size() == 1 && (*shOr)[0] == N && weightName.empty()) {
      weightName = nm;
      continue;
    }
  }
  if (inputName.empty() || weightName.empty()) {
    ctx.module.emitError("rms_norm2d: failed to infer input/weight external tensors");
    return mlir::failure();
  }

  // Shapes.
  auto shapeInOr = resolveShape(ctx.tensors[inputName], ctx.shapeBindings);
  auto shapeWOr = resolveShape(ctx.tensors[weightName], ctx.shapeBindings);
  if (mlir::failed(shapeInOr) || mlir::failed(shapeWOr)) {
    ctx.module.emitError("rms_norm2d: failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeInOr->size() != 2 || shapeWOr->size() != 1) {
    ctx.module.emitError("rms_norm2d: expected input rank-2 and weight rank-1");
    return mlir::failure();
  }
  if ((*shapeInOr)[0] != M || (*shapeInOr)[1] != N) {
    ctx.module.emitError("rms_norm2d: input shape mismatch");
    return mlir::failure();
  }
  if ((*shapeWOr)[0] != N) {
    ctx.module.emitError("rms_norm2d: weight shape mismatch");
    return mlir::failure();
  }

  // Dtypes.
  for (const auto &name : {inputName, weightName, std::string("eps"), std::string("N_scalar")}) {
    if (llvm::StringRef(ctx.tensors[name].dtype).trim().lower() != "f32") {
      ctx.module.emitError() << "rms_norm2d: expected f32 for tensor " << name;
      return mlir::failure();
    }
  }
  if (llvm::StringRef(ctx.tensors[outName].dtype).trim().lower() != "f32" ||
      llvm::StringRef(ctx.tensors[invName].dtype).trim().lower() != "f32") {
    ctx.module.emitError("rms_norm2d: expected f32 outputs");
    return mlir::failure();
  }

  // Kernel config: 1 CTA per row, 256 threads.
  int64_t threads = 256;
  if (threads <= 0 || threads > 1024 || (threads % 32) != 0) {
    ctx.module.emitError("rms_norm2d: invalid threads");
    return mlir::failure();
  }

  clearModuleBody(ctx.module);

  // Ensure the module is treated as a GPU container module and has a target triple.
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  // GPU module + shared scratch.
  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto globalMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);
  auto sharedMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);

  // Shared buffer for reduction + scalar broadcast.
  auto shTy = mlir::MemRefType::get({threads}, f32,
                                    mlir::MemRefLayoutAttrInterface{},
                                    sharedMemSpace);
  auto shName = "__intentir_sh_rmsnorm_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
  auto align16 = b.getI64IntegerAttr(16);
  (void)mlir::memref::GlobalOp::create(b, loc, shName, b.getStringAttr("private"), shTy,
                                      /*initial_value=*/{}, /*constant=*/false, align16);

  // Kernel.
  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto In = getArgByName(ctx, fn, inputName);
  auto W = getArgByName(ctx, fn, weightName);
  auto EpsArg = getArgByName(ctx, fn, "eps");
  auto NScalarArg = getArgByName(ctx, fn, "N_scalar");
  auto Out = getArgByName(ctx, fn, outName);
  auto Inv = getArgByName(ctx, fn, invName);
  if (!In || !W || !Out || !Inv) {
    ctx.module.emitError("rms_norm2d: failed to map kernel args");
    return mlir::failure();
  }

  // Reinterpret flattened buffers.
  auto in2Ty = mlir::MemRefType::get({M, N}, f32,
                                     mlir::MemRefLayoutAttrInterface{},
                                     globalMemSpace);
  auto out2Ty = mlir::MemRefType::get({M, N}, f32,
                                      mlir::MemRefLayoutAttrInterface{},
                                      globalMemSpace);
  auto In2 = mlir::memref::ReinterpretCastOp::create(b, loc, in2Ty, In, 0, {M, N}, {N, 1})
                 .getResult();
  auto Out2 = mlir::memref::ReinterpretCastOp::create(b, loc, out2Ty, Out, 0, {M, N}, {N, 1})
                  .getResult();

  // Thread/block ids.
  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x);
  auto row = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x);

  auto c0 = makeIndexConst(b, loc, 0);
  auto cN = makeIndexConst(b, loc, N);
  auto cThreads = makeIndexConst(b, loc, threads);
  auto c0f = makeF32Const(b, loc, 0.0f);

  // Scalars: allow either external scalar inputs or const ops.
  auto constEps = [&]() -> std::optional<float> {
    for (const auto &op : ctx.ops) {
      if (op.op != "const")
        continue;
      if (op.output != "eps")
        continue;
      auto dtype = op.attrs.getString("dtype");
      if (dtype && llvm::StringRef(*dtype).trim().lower() != "f32")
        continue;
      if (auto num = op.attrs.getNumber("value")) {
        return static_cast<float>(*num);
      }
    }
    return std::nullopt;
  };

  mlir::Value epsVal;
  if (EpsArg) {
    epsVal = b.create<mlir::memref::LoadOp>(loc, EpsArg, mlir::ValueRange{c0}).getResult();
  } else if (auto epsC = constEps()) {
    epsVal = makeF32Const(b, loc, *epsC);
  } else {
    ctx.module.emitError("rms_norm2d: missing eps scalar (neither arg nor const)");
    return mlir::failure();
  }

  mlir::Value nVal;
  if (NScalarArg) {
    nVal = b.create<mlir::memref::LoadOp>(loc, NScalarArg, mlir::ValueRange{c0}).getResult();
  } else {
    // N_scalar is often a const("N") in intent_expanded; use the resolved N.
    nVal = makeF32Const(b, loc, static_cast<float>(N));
  }

  // Shared buffer handle.
  auto Sh = mlir::memref::GetGlobalOp::create(b, loc, shTy, shName).getResult();

  // Partial sum of squares for this thread.
  auto sumFor = b.create<mlir::scf::ForOp>(loc, tid, cN, cThreads, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(sumFor.getBody());
  auto j = sumFor.getInductionVar();
  auto acc = sumFor.getRegionIterArgs()[0];
  auto x = b.create<mlir::memref::LoadOp>(loc, In2, mlir::ValueRange{row, j}).getResult();
  auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x).getResult();
  auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, x2).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
  b.setInsertionPointAfter(sumFor);
  auto partial = sumFor.getResult(0);

  b.create<mlir::memref::StoreOp>(loc, partial, Sh, mlir::ValueRange{tid});
  b.create<mlir::gpu::BarrierOp>(loc);

  // Block reduction in shared memory.
  for (int64_t stride = threads / 2; stride >= 1; stride /= 2) {
    auto cStride = makeIndexConst(b, loc, stride);
    auto cond =
        b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, tid, cStride)
            .getResult();
    auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, cond, /*withElse=*/false);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto a = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{tid}).getResult();
    auto tid2 = b.create<mlir::arith::AddIOp>(loc, tid, cStride).getResult();
    auto bval = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{tid2}).getResult();
    auto s = b.create<mlir::arith::AddFOp>(loc, a, bval).getResult();
    b.create<mlir::memref::StoreOp>(loc, s, Sh, mlir::ValueRange{tid});
    b.setInsertionPointAfter(ifOp);
    b.create<mlir::gpu::BarrierOp>(loc);
  }

  // Thread 0 computes INV_RMS and broadcasts via Sh[0].
  auto is0 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, tid, c0).getResult();
  auto if0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, is0, /*withElse=*/false);
  b.setInsertionPointToStart(&if0.getThenRegion().front());
  auto sum0 = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{c0}).getResult();
  auto mean = b.create<mlir::arith::DivFOp>(loc, sum0, nVal).getResult();
  auto var = b.create<mlir::arith::AddFOp>(loc, mean, epsVal).getResult();
  auto inv = b.create<mlir::math::RsqrtOp>(loc, var).getResult();
  b.create<mlir::memref::StoreOp>(loc, inv, Inv, mlir::ValueRange{row});
  b.create<mlir::memref::StoreOp>(loc, inv, Sh, mlir::ValueRange{c0});
  b.setInsertionPointAfter(if0);
  b.create<mlir::gpu::BarrierOp>(loc);

  auto invAll = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{c0}).getResult();

  // Write normalized output.
  auto outFor = b.create<mlir::scf::ForOp>(loc, tid, cN, cThreads);
  b.setInsertionPointToStart(outFor.getBody());
  auto jj = outFor.getInductionVar();
  auto xv = b.create<mlir::memref::LoadOp>(loc, In2, mlir::ValueRange{row, jj}).getResult();
  auto wv = b.create<mlir::memref::LoadOp>(loc, W, mlir::ValueRange{jj}).getResult();
  auto y0 = b.create<mlir::arith::MulFOp>(loc, xv, invAll).getResult();
  auto y = b.create<mlir::arith::MulFOp>(loc, y0, wv).getResult();
  b.create<mlir::memref::StoreOp>(loc, y, Out2, mlir::ValueRange{row, jj});
  b.setInsertionPointAfter(outFor);

  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, "rms_norm2d_rowwise_v1"));
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = "rms_norm2d_rowwise_v1";
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/M, /*gy=*/1, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(M * N);
  });
  return mlir::success();
}

static mlir::LogicalResult lowerCudaRmsNorm2dRowwiseV2(LoweringContext &ctx) {
  // Same semantics as v1 but faster reduction:
  // - per-warp sumsq via shuffle XOR
  // - cross-warp reduction via shared (warp0 only)
  // - 2x barriers total
  if (ctx.outputs.size() != 2) {
    ctx.module.emitError("rms_norm2d: expected 2 outputs (out, INV_RMS)");
    return mlir::failure();
  }
  if (ctx.tensors.find("eps") == ctx.tensors.end() ||
      ctx.tensors.find("N_scalar") == ctx.tensors.end()) {
    ctx.module.emitError("rms_norm2d: missing required scalar inputs (eps/N_scalar)");
    return mlir::failure();
  }

  std::string outName;
  std::string invName;
  for (const auto &nm : ctx.outputs) {
    auto it = ctx.tensors.find(nm);
    if (it == ctx.tensors.end())
      continue;
    auto shOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shOr))
      continue;
    if (shOr->size() == 2) {
      outName = nm;
      continue;
    }
    if (shOr->size() == 1) {
      invName = nm;
      continue;
    }
  }
  if (outName.empty() || invName.empty()) {
    ctx.module.emitError("rms_norm2d: failed to identify rank-2 out and rank-1 INV_RMS outputs");
    return mlir::failure();
  }

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  auto shapeOutOr = resolveShape(ctx.tensors[outName], ctx.shapeBindings);
  auto shapeInvOr = resolveShape(ctx.tensors[invName], ctx.shapeBindings);
  if (mlir::failed(shapeOutOr) || mlir::failed(shapeInvOr)) {
    ctx.module.emitError("rms_norm2d: failed to resolve output shapes");
    return mlir::failure();
  }
  int64_t M = (*shapeOutOr)[0];
  int64_t N = (*shapeOutOr)[1];
  if (shapeInvOr->size() != 1 || (*shapeInvOr)[0] != M) {
    ctx.module.emitError("rms_norm2d: INV_RMS must be shape [M]");
    return mlir::failure();
  }

  // Infer input matrix and weight vector names from external inputs.
  std::set<std::string> outSet(ctx.outputs.begin(), ctx.outputs.end());
  std::string inputName;
  std::string weightName;
  for (const auto &nm : ctx.argOrder) {
    if (outSet.count(nm))
      continue;
    auto it = ctx.tensors.find(nm);
    if (it == ctx.tensors.end())
      continue;
    if (llvm::StringRef(it->second.dtype).trim().lower() != "f32")
      continue;
    auto shOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shOr))
      continue;
    if (shOr->size() == 2 && (*shOr)[0] == M && (*shOr)[1] == N && inputName.empty()) {
      inputName = nm;
      continue;
    }
    if (shOr->size() == 1 && (*shOr)[0] == N && weightName.empty()) {
      weightName = nm;
      continue;
    }
  }
  if (inputName.empty() || weightName.empty()) {
    ctx.module.emitError("rms_norm2d: failed to infer input/weight external tensors");
    return mlir::failure();
  }

  // Shapes.
  auto shapeInOr = resolveShape(ctx.tensors[inputName], ctx.shapeBindings);
  auto shapeWOr = resolveShape(ctx.tensors[weightName], ctx.shapeBindings);
  if (mlir::failed(shapeInOr) || mlir::failed(shapeWOr)) {
    ctx.module.emitError("rms_norm2d: failed to resolve shapes");
    return mlir::failure();
  }
  if (shapeInOr->size() != 2 || shapeWOr->size() != 1) {
    ctx.module.emitError("rms_norm2d: expected input rank-2 and weight rank-1");
    return mlir::failure();
  }
  if ((*shapeInOr)[0] != M || (*shapeInOr)[1] != N) {
    ctx.module.emitError("rms_norm2d: input shape mismatch");
    return mlir::failure();
  }
  if ((*shapeWOr)[0] != N) {
    ctx.module.emitError("rms_norm2d: weight shape mismatch");
    return mlir::failure();
  }

  // Dtypes.
  for (const auto &name : {inputName, weightName, std::string("eps"), std::string("N_scalar")}) {
    if (llvm::StringRef(ctx.tensors[name].dtype).trim().lower() != "f32") {
      ctx.module.emitError() << "rms_norm2d: expected f32 for tensor " << name;
      return mlir::failure();
    }
  }
  if (llvm::StringRef(ctx.tensors[outName].dtype).trim().lower() != "f32" ||
      llvm::StringRef(ctx.tensors[invName].dtype).trim().lower() != "f32") {
    ctx.module.emitError("rms_norm2d: expected f32 outputs");
    return mlir::failure();
  }

  // Kernel config: 1 CTA per row, 256 threads.
  const int64_t threads = 256;
  const int64_t warps = threads / 32;
  if (threads <= 0 || threads > 1024 || (threads % 32) != 0 || warps <= 0 || warps > 32) {
    ctx.module.emitError("rms_norm2d: invalid threads/warps");
    return mlir::failure();
  }

  clearModuleBody(ctx.module);

  // Ensure the module is treated as a GPU container module and has a target triple.
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple",
                        mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  // GPU module + shared scratch.
  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32 = b.getF32Type();
  auto globalMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);
  auto sharedMemSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 3);

  // Shared buffer: warp sums (0..warps-1) and INV broadcast at [0].
  auto shTy = mlir::MemRefType::get({threads}, f32,
                                    mlir::MemRefLayoutAttrInterface{},
                                    sharedMemSpace);
  auto shName = "__intentir_sh_rmsnorm_" + sanitizeSymbolName(ctx.kernelName) + "_f32";
  auto align16 = b.getI64IntegerAttr(16);
  (void)mlir::memref::GlobalOp::create(b, loc, shName, b.getStringAttr("private"), shTy,
                                      /*initial_value=*/{}, /*constant=*/false, align16);

  // Kernel.
  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto In = getArgByName(ctx, fn, inputName);
  auto W = getArgByName(ctx, fn, weightName);
  auto EpsArg = getArgByName(ctx, fn, "eps");
  auto NScalarArg = getArgByName(ctx, fn, "N_scalar");
  auto Out = getArgByName(ctx, fn, outName);
  auto Inv = getArgByName(ctx, fn, invName);
  if (!In || !W || !Out || !Inv) {
    ctx.module.emitError("rms_norm2d: failed to map kernel args");
    return mlir::failure();
  }

  // Reinterpret flattened buffers.
  auto in2Ty = mlir::MemRefType::get({M, N}, f32,
                                     mlir::MemRefLayoutAttrInterface{},
                                     globalMemSpace);
  auto out2Ty = mlir::MemRefType::get({M, N}, f32,
                                      mlir::MemRefLayoutAttrInterface{},
                                      globalMemSpace);
  auto In2 = mlir::memref::ReinterpretCastOp::create(b, loc, in2Ty, In, 0, {M, N}, {N, 1})
                 .getResult();
  auto Out2 = mlir::memref::ReinterpretCastOp::create(b, loc, out2Ty, Out, 0, {M, N}, {N, 1})
                  .getResult();

  // Thread/block ids.
  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x).getResult();
  auto row = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x).getResult();

  auto c0 = makeIndexConst(b, loc, 0);
  auto cN = makeIndexConst(b, loc, N);
  auto cThreads = makeIndexConst(b, loc, threads);
  auto c0f = makeF32Const(b, loc, 0.0f);

  // Scalars: allow either external scalar inputs or const ops.
  auto constEps = [&]() -> std::optional<float> {
    for (const auto &op : ctx.ops) {
      if (op.op != "const")
        continue;
      if (op.output != "eps")
        continue;
      auto dtype = op.attrs.getString("dtype");
      if (dtype && llvm::StringRef(*dtype).trim().lower() != "f32")
        continue;
      if (auto num = op.attrs.getNumber("value")) {
        return static_cast<float>(*num);
      }
    }
    return std::nullopt;
  };

  mlir::Value epsVal;
  if (EpsArg) {
    epsVal = b.create<mlir::memref::LoadOp>(loc, EpsArg, mlir::ValueRange{c0}).getResult();
  } else if (auto epsC = constEps()) {
    epsVal = makeF32Const(b, loc, *epsC);
  } else {
    ctx.module.emitError("rms_norm2d: missing eps scalar (neither arg nor const)");
    return mlir::failure();
  }

  mlir::Value nVal;
  if (NScalarArg) {
    nVal = b.create<mlir::memref::LoadOp>(loc, NScalarArg, mlir::ValueRange{c0}).getResult();
  } else {
    nVal = makeF32Const(b, loc, static_cast<float>(N));
  }

  // Shared buffer handle.
  auto Sh = mlir::memref::GetGlobalOp::create(b, loc, shTy, shName).getResult();

  // Partial sum of squares for this thread.
  auto sumFor = b.create<mlir::scf::ForOp>(loc, tid, cN, cThreads, mlir::ValueRange{c0f});
  b.setInsertionPointToStart(sumFor.getBody());
  auto j = sumFor.getInductionVar();
  auto acc = sumFor.getRegionIterArgs()[0];
  auto x = b.create<mlir::memref::LoadOp>(loc, In2, mlir::ValueRange{row, j}).getResult();
  auto x2 = b.create<mlir::arith::MulFOp>(loc, x, x).getResult();
  auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, x2).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
  b.setInsertionPointAfter(sumFor);
  auto partial = sumFor.getResult(0);

  // Warp reduce.
  auto warpSum = warpAllReduceSumF32(b, loc, partial);

  // lane/warp ids.
  auto tidI32 = b.create<mlir::arith::IndexCastOp>(loc, b.getI32Type(), tid).getResult();
  auto c32i = makeI32Const(b, loc, 32);
  auto laneI32 = b.create<mlir::arith::RemUIOp>(loc, tidI32, c32i).getResult();
  auto warpI32 = b.create<mlir::arith::DivUIOp>(loc, tidI32, c32i).getResult();
  auto laneIs0 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, laneI32, makeI32Const(b, loc, 0))
          .getResult();

  // lane0 stores per-warp sum to shared[warp].
  auto ifLane0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, laneIs0, /*withElse=*/false);
  b.setInsertionPointToStart(&ifLane0.getThenRegion().front());
  auto warpIdx = b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), warpI32).getResult();
  b.create<mlir::memref::StoreOp>(loc, warpSum, Sh, mlir::ValueRange{warpIdx});
  b.setInsertionPointAfter(ifLane0);
  b.create<mlir::gpu::BarrierOp>(loc);

  // Warp0 reduces warp sums and writes INV to shared[0] and INV_RMS output.
  auto isWarp0 =
      b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, warpI32, makeI32Const(b, loc, 0))
          .getResult();
  auto ifWarp0 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, isWarp0, /*withElse=*/false);
  b.setInsertionPointToStart(&ifWarp0.getThenRegion().front());
  auto cWarpsI32 = makeI32Const(b, loc, static_cast<int32_t>(warps));
  auto laneIn = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, laneI32, cWarpsI32).getResult();
  auto loadIf = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{f32}, laneIn, /*withElse=*/true);
  b.setInsertionPointToStart(&loadIf.getThenRegion().front());
  auto laneIdx = b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), laneI32).getResult();
  auto wsum = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{laneIdx}).getResult();
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{wsum});
  b.setInsertionPointToStart(&loadIf.getElseRegion().front());
  b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{c0f});
  b.setInsertionPointAfter(loadIf);
  auto sum0 = warpAllReduceSumF32(b, loc, loadIf.getResult(0));

  auto ifLane0b = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{}, laneIs0, /*withElse=*/false);
  b.setInsertionPointToStart(&ifLane0b.getThenRegion().front());
  auto mean = b.create<mlir::arith::DivFOp>(loc, sum0, nVal).getResult();
  auto var = b.create<mlir::arith::AddFOp>(loc, mean, epsVal).getResult();
  auto inv = b.create<mlir::math::RsqrtOp>(loc, var).getResult();
  b.create<mlir::memref::StoreOp>(loc, inv, Inv, mlir::ValueRange{row});
  b.create<mlir::memref::StoreOp>(loc, inv, Sh, mlir::ValueRange{c0});
  b.setInsertionPointAfter(ifLane0b);
  b.setInsertionPointAfter(ifWarp0);
  b.create<mlir::gpu::BarrierOp>(loc);

  auto invAll = b.create<mlir::memref::LoadOp>(loc, Sh, mlir::ValueRange{c0}).getResult();

  // Write normalized output.
  auto outFor = b.create<mlir::scf::ForOp>(loc, tid, cN, cThreads);
  b.setInsertionPointToStart(outFor.getBody());
  auto jj = outFor.getInductionVar();
  auto xv = b.create<mlir::memref::LoadOp>(loc, In2, mlir::ValueRange{row, jj}).getResult();
  auto wv = b.create<mlir::memref::LoadOp>(loc, W, mlir::ValueRange{jj}).getResult();
  auto y0 = b.create<mlir::arith::MulFOp>(loc, xv, invAll).getResult();
  auto y = b.create<mlir::arith::MulFOp>(loc, y0, wv).getResult();
  b.create<mlir::memref::StoreOp>(loc, y, Out2, mlir::ValueRange{row, jj});
  b.setInsertionPointAfter(outFor);

  b.create<mlir::gpu::ReturnOp>(loc);

  ctx.module->setAttr("intentir.compiler_stack",
                      mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_focus_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, "rms_norm2d_rowwise_v2"));
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_focus_v1";
    meta["cuda_real_mlir_kernel_kind"] = "rms_norm2d_rowwise_v2";
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/M, /*gy=*/1, /*gz=*/1,
                               /*sharedMem=*/0);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(M * N);
  });
  return mlir::success();
}

// === CUDA Full196 correctness-first lowering (cpp_plugin) ===

static bool isBoolDtype(llvm::StringRef dtype) {
  auto d = dtype.trim().lower();
  return d == "bool" || d == "i1";
}

static bool isFloatDtype(llvm::StringRef dtype) {
  auto d = dtype.trim().lower();
  return d == "f32" || d == "f16" || d == "bf16";
}

static mlir::Type dtypeToScalarType(mlir::MLIRContext *ctx, llvm::StringRef dtype) {
  auto d = dtype.trim().lower();
  if (d == "f32")
    return mlir::Float32Type::get(ctx);
  if (d == "f16")
    return mlir::Float16Type::get(ctx);
  if (d == "bf16")
    return mlir::BFloat16Type::get(ctx);
  if (d == "i32")
    return mlir::IntegerType::get(ctx, 32);
  if (d == "i64")
    return mlir::IntegerType::get(ctx, 64);
  if (d == "bool" || d == "i1")
    return mlir::IntegerType::get(ctx, 1);
  return {};
}

static mlir::Value mapScalarBoolToMemI8(mlir::OpBuilder &b, mlir::Location loc, mlir::Value vI1) {
  auto i8 = b.getI8Type();
  auto one = b.create<mlir::arith::ConstantIntOp>(loc, 1, 8).getResult();
  auto zero = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult();
  return b.create<mlir::arith::SelectOp>(loc, vI1, one, zero).getResult();
}

static mlir::Value mapMemI8ToScalarBoolI1(mlir::OpBuilder &b, mlir::Location loc, mlir::Value vI8) {
  auto zero = b.create<mlir::arith::ConstantIntOp>(loc, 0, 8).getResult();
  return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, vI8, zero).getResult();
}

static mlir::Value linearizeIndices(mlir::OpBuilder &b, mlir::Location loc,
                                    llvm::ArrayRef<mlir::Value> indices,
                                    llvm::ArrayRef<int64_t> shape) {
  if (shape.empty())
    return makeIndexConst(b, loc, 0);
  mlir::Value cur = indices.front();
  for (size_t axis = 1; axis < shape.size(); ++axis) {
    auto dim = makeIndexConst(b, loc, shape[axis]);
    cur = b.create<mlir::arith::AddIOp>(
               loc, b.create<mlir::arith::MulIOp>(loc, cur, dim).getResult(), indices[axis])
              .getResult();
  }
  return cur;
}

static llvm::SmallVector<mlir::Value>
delinearizeIndex(mlir::OpBuilder &b, mlir::Location loc, mlir::Value linear,
                 llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<mlir::Value> idx(shape.size());
  mlir::Value cur = linear;
  for (int64_t axis = static_cast<int64_t>(shape.size()) - 1; axis >= 0; --axis) {
    auto dim = makeIndexConst(b, loc, shape[axis]);
    auto rem = b.create<mlir::arith::RemUIOp>(loc, cur, dim).getResult();
    auto div = b.create<mlir::arith::DivUIOp>(loc, cur, dim).getResult();
    idx[static_cast<size_t>(axis)] = rem;
    cur = div;
  }
  return idx;
}

static llvm::SmallVector<mlir::Value>
mapIndicesNumpyBroadcast(mlir::OpBuilder &b, mlir::Location loc, llvm::ArrayRef<mlir::Value> outIdx,
                         llvm::ArrayRef<int64_t> outShape, llvm::ArrayRef<int64_t> inShape) {
  llvm::SmallVector<mlir::Value> inIdx;
  if (inShape.empty())
    return inIdx;
  const int64_t outRank = static_cast<int64_t>(outShape.size());
  const int64_t inRank = static_cast<int64_t>(inShape.size());
  if (inRank > outRank)
    return inIdx;
  const int64_t lead = outRank - inRank;
  inIdx.resize(static_cast<size_t>(inRank));
  auto c0 = makeIndexConst(b, loc, 0);
  for (int64_t a = 0; a < inRank; ++a) {
    int64_t dim = inShape[static_cast<size_t>(a)];
    if (dim == 1) {
      inIdx[static_cast<size_t>(a)] = c0;
    } else {
      inIdx[static_cast<size_t>(a)] = outIdx[static_cast<size_t>(lead + a)];
    }
  }
  return inIdx;
}

static mlir::Value castIndexToInt(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v, int bits) {
  auto ty = mlir::IntegerType::get(b.getContext(), bits);
  return b.create<mlir::arith::IndexCastOp>(loc, ty, v).getResult();
}

static mlir::Value castIntToIndex(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v) {
  return b.create<mlir::arith::IndexCastOp>(loc, b.getIndexType(), v).getResult();
}

static mlir::Value castScalar(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v, mlir::Type toTy) {
  auto fromTy = v.getType();
  if (fromTy == toTy)
    return v;
  if (auto fromF = mlir::dyn_cast<mlir::FloatType>(fromTy)) {
    if (auto toF = mlir::dyn_cast<mlir::FloatType>(toTy)) {
      if (fromF.getWidth() == toF.getWidth())
        return v;
      if (fromF.getWidth() < toF.getWidth())
        return b.create<mlir::arith::ExtFOp>(loc, toTy, v).getResult();
      return b.create<mlir::arith::TruncFOp>(loc, toTy, v).getResult();
    }
    if (auto toI = mlir::dyn_cast<mlir::IntegerType>(toTy)) {
      if (toI.getWidth() == 1) {
        // Treat NaN as true (torch bool casting semantics).
        auto zero = makeF32Const(b, loc, 0.0f);
        return b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, v, zero).getResult();
      }
      return b.create<mlir::arith::FPToSIOp>(loc, toTy, v).getResult();
    }
  }
  if (auto fromI = mlir::dyn_cast<mlir::IntegerType>(fromTy)) {
    if (auto toI = mlir::dyn_cast<mlir::IntegerType>(toTy)) {
      if (fromI.getWidth() == toI.getWidth())
        return v;
      if (fromI.getWidth() < toI.getWidth()) {
        if (fromI.getWidth() == 1)
          return b.create<mlir::arith::ExtUIOp>(loc, toTy, v).getResult();
        return b.create<mlir::arith::ExtSIOp>(loc, toTy, v).getResult();
      }
      return b.create<mlir::arith::TruncIOp>(loc, toTy, v).getResult();
    }
    if (auto toF = mlir::dyn_cast<mlir::FloatType>(toTy)) {
      auto promoted = v;
      if (fromI.getWidth() == 1) {
        promoted = b.create<mlir::arith::ExtUIOp>(loc, b.getI32Type(), v).getResult();
      }
      return b.create<mlir::arith::SIToFPOp>(loc, toTy, promoted).getResult();
    }
  }
  // Fallback: best-effort bitcast is not supported; keep v and let verifier fail.
  return v;
}

class CudaFull196Emitter {
public:
  CudaFull196Emitter(LoweringContext &ctx, mlir::gpu::GPUFuncOp fn)
      : ctx(ctx), fn(fn), b(ctx.builder), loc(ctx.module.getLoc()),
        mlirCtx(ctx.module.getContext()) {
    for (auto &op : ctx.ops) {
      producers[op.output] = &op;
    }
    for (const auto &kv : ctx.tensors) {
      dtypes[kv.first] = kv.second.dtype;
      auto shOr = resolveShape(kv.second, ctx.shapeBindings);
      if (!mlir::failed(shOr)) {
        shapes[kv.first] = *shOr;
      }
    }
    for (const auto &name : ctx.argOrder) {
      auto v = getArgByName(ctx, fn, name);
      if (v)
        args[name] = v;
    }
  }

  mlir::FailureOr<mlir::Value> emitScalar(llvm::StringRef name, llvm::ArrayRef<mlir::Value> idx) {
    auto itShape = shapes.find(name.str());
    if (itShape == shapes.end()) {
      ctx.module.emitError() << "full196: missing resolved shape for tensor " << name;
      return mlir::failure();
    }
    if (idx.size() != itShape->second.size()) {
      ctx.module.emitError() << "full196: index rank mismatch for tensor " << name << " expected_rank="
                             << itShape->second.size() << " got_rank=" << idx.size();
      return mlir::failure();
    }

    // External argument tensor (input or output buffer reused as input).
    if (auto itArg = args.find(name.str()); itArg != args.end()) {
      if (producers.count(name.str()) == 0) {
        return loadArgElement(name, idx);
      }
    }

    auto itP = producers.find(name.str());
    if (itP == producers.end()) {
      // Treat as external input arg by default.
      if (auto itArg = args.find(name.str()); itArg != args.end()) {
        return loadArgElement(name, idx);
      }
      ctx.module.emitError() << "full196: no producer and not an ABI arg: " << name;
      return mlir::failure();
    }
    return emitFromOp(*itP->second, idx);
  }

  llvm::ArrayRef<int64_t> shapeOf(llvm::StringRef name) const {
    auto it = shapes.find(name.str());
    if (it == shapes.end())
      return {};
    return it->second;
  }

  llvm::StringRef dtypeOf(llvm::StringRef name) const {
    auto it = dtypes.find(name.str());
    if (it == dtypes.end())
      return llvm::StringRef();
    return llvm::StringRef(it->second);
  }

private:
  LoweringContext &ctx;
  mlir::gpu::GPUFuncOp fn;
  mlir::OpBuilder &b;
  mlir::Location loc;
  mlir::MLIRContext *mlirCtx;

  std::map<std::string, const OpSpec *> producers;
  std::map<std::string, std::string> dtypes;
  std::map<std::string, std::vector<int64_t>> shapes;
  std::map<std::string, mlir::Value> args;

  mlir::Type scalarTypeFor(llvm::StringRef name) const {
    auto dtype = dtypeOf(name);
    return dtypeToScalarType(mlirCtx, dtype);
  }

  mlir::FailureOr<mlir::Value> loadArgElement(llvm::StringRef name, llvm::ArrayRef<mlir::Value> idx) {
    auto itArg = args.find(name.str());
    if (itArg == args.end()) {
      ctx.module.emitError() << "full196: missing ABI argument for tensor " << name;
      return mlir::failure();
    }
    auto memref = itArg->second;
    auto shape = shapeOf(name);
    auto lin = linearizeIndices(b, loc, idx, shape);
    auto raw = b.create<mlir::memref::LoadOp>(loc, memref, mlir::ValueRange{lin}).getResult();
    if (isBoolDtype(dtypeOf(name))) {
      return mapMemI8ToScalarBoolI1(b, loc, raw);
    }
    return raw;
  }

  llvm::SmallVector<mlir::Value> mapElemwiseInputIndices(llvm::StringRef inName,
                                                         llvm::ArrayRef<mlir::Value> outIdx,
                                                         llvm::ArrayRef<int64_t> outShape) {
    auto inShape = shapeOf(inName);
    return mapIndicesNumpyBroadcast(b, loc, outIdx, outShape, inShape);
  }

  mlir::FailureOr<mlir::Value> emitBroadcasted(llvm::StringRef inName,
                                               llvm::ArrayRef<mlir::Value> outIdx,
                                               llvm::ArrayRef<int64_t> outShape) {
    auto inIdx = mapElemwiseInputIndices(inName, outIdx, outShape);
    return emitScalar(inName, inIdx);
  }

  mlir::FailureOr<int64_t> resolveIntParam(const llvm::json::Value &v, llvm::StringRef what) {
    if (auto ii = v.getAsInteger())
      return static_cast<int64_t>(*ii);
    if (auto num = v.getAsNumber())
      return static_cast<int64_t>(*num);
    if (auto s = v.getAsString()) {
      std::string key = s->trim().str();
      if (auto it = ctx.shapeBindings.find(key); it != ctx.shapeBindings.end()) {
        return it->second;
      }
      ctx.module.emitError() << "full196: missing shape binding for " << what << ": " << key;
      return mlir::failure();
    }
    ctx.module.emitError() << "full196: expected int-like attr for " << what;
    return mlir::failure();
  }

  mlir::FailureOr<llvm::SmallVector<int64_t>>
  resolveIntListParam(const OpSpec &op, llvm::StringRef key, size_t count, int64_t defaultVal) {
    llvm::SmallVector<int64_t> out;
    out.resize(count, defaultVal);

    if (const auto *arr = op.attrs.getArray(key)) {
      if (arr->size() != count) {
        ctx.module.emitError() << "full196: expected " << count << " entries for attrs." << key;
        return mlir::failure();
      }
      for (size_t i = 0; i < count; ++i) {
        auto vOr = resolveIntParam((*arr)[i], llvm::Twine("attrs.").concat(key).str());
        if (mlir::failed(vOr))
          return mlir::failure();
        out[i] = *vOr;
      }
      return out;
    }

    if (const llvm::json::Value *vv = op.attrs.get(key)) {
      auto vOr = resolveIntParam(*vv, llvm::Twine("attrs.").concat(key).str());
      if (mlir::failed(vOr))
        return mlir::failure();
      for (size_t i = 0; i < count; ++i) {
        out[i] = *vOr;
      }
    }
    return out;
  }

  mlir::FailureOr<mlir::Value> emitConstLike(const OpSpec &op) {
    auto outTy = scalarTypeFor(op.output);
    if (!outTy) {
      ctx.module.emitError() << "full196: unsupported const dtype for " << op.output << ": "
                             << dtypeOf(op.output);
      return mlir::failure();
    }
    const llvm::json::Value *vv = op.attrs.get("value");
    if (!vv) {
      ctx.module.emitError("full196: const missing attrs.value");
      return mlir::failure();
    }

    // If value is a string, allow symbolic lookup in shape bindings.
    if (auto s = vv->getAsString()) {
      std::string key = s->trim().str();
      if (auto it = ctx.shapeBindings.find(key); it != ctx.shapeBindings.end()) {
        int64_t v = it->second;
        if (mlir::isa<mlir::FloatType>(outTy)) {
          return makeF32Const(b, loc, static_cast<float>(v));
        }
        if (auto it = mlir::dyn_cast<mlir::IntegerType>(outTy)) {
          return b.create<mlir::arith::ConstantIntOp>(loc, v, it.getWidth()).getResult();
        }
      }
      // Try parse as number.
      char *end = nullptr;
      double dv = std::strtod(key.c_str(), &end);
      if (end && *end == '\0') {
        if (mlir::isa<mlir::FloatType>(outTy)) {
          return makeF32Const(b, loc, static_cast<float>(dv));
        }
        if (auto it = mlir::dyn_cast<mlir::IntegerType>(outTy)) {
          return b.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(dv), it.getWidth())
              .getResult();
        }
      }
      ctx.module.emitError() << "full196: const string value not resolvable: " << key;
      return mlir::failure();
    }

    if (auto num = vv->getAsNumber()) {
      double dv = *num;
      if (mlir::isa<mlir::FloatType>(outTy)) {
        return makeF32Const(b, loc, static_cast<float>(dv));
      }
      if (auto it = mlir::dyn_cast<mlir::IntegerType>(outTy)) {
        return b.create<mlir::arith::ConstantIntOp>(loc, static_cast<int64_t>(dv), it.getWidth()).getResult();
      }
    }
    ctx.module.emitError("full196: const value type unsupported");
    return mlir::failure();
  }

  mlir::FailureOr<mlir::Value> emitFromOp(const OpSpec &op, llvm::ArrayRef<mlir::Value> outIdx) {
    auto outShape = shapeOf(op.output);
    auto outDtype = dtypeOf(op.output).trim().lower();

    auto emitBinNumeric = [&](auto makeOpF, auto makeOpI) -> mlir::FailureOr<mlir::Value> {
      if (op.inputs.size() != 2) {
        ctx.module.emitError() << "full196: " << op.op << " expects 2 inputs";
        return mlir::failure();
      }
      auto aOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto bOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto a = *aOr;
      auto bb = *bOr;

      // Promote to a common compute type.
      mlir::Type computeTy;
      if (mlir::isa<mlir::FloatType>(a.getType()) || mlir::isa<mlir::FloatType>(bb.getType())) {
        computeTy = b.getF32Type();
      } else if (a.getType() == b.getI64Type() || bb.getType() == b.getI64Type()) {
        computeTy = b.getI64Type();
      } else if (mlir::isa<mlir::IntegerType>(a.getType()) || mlir::isa<mlir::IntegerType>(bb.getType())) {
        computeTy = b.getI32Type();
      } else {
        computeTy = a.getType();
      }
      a = castScalar(b, loc, a, computeTy);
      bb = castScalar(b, loc, bb, computeTy);

      mlir::Value res;
      if (mlir::isa<mlir::FloatType>(computeTy)) {
        res = makeOpF(a, bb);
      } else {
        res = makeOpI(a, bb);
      }

      auto outTy = scalarTypeFor(op.output);
      if (!outTy) {
        ctx.module.emitError() << "full196: unsupported output dtype for " << op.op << ": " << outDtype;
        return mlir::failure();
      }
      res = castScalar(b, loc, res, outTy);
      if (isBoolDtype(outDtype)) {
        // Ensure bool is i1.
        res = castScalar(b, loc, res, b.getI1Type());
      }
      return res;
    };

    if (op.op == "identity") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: identity expects 1 input");
        return mlir::failure();
      }
      return emitBroadcasted(op.inputs[0], outIdx, outShape);
    }
    if (op.op == "const") {
      return emitConstLike(op);
    }
    if (op.op == "cast") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: cast expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inOr = emitBroadcasted(inName, outIdx, outShape);
      if (mlir::failed(inOr))
        return mlir::failure();
      auto v = *inOr;
      auto to = op.attrs.getString("to");
      if (!to) {
        ctx.module.emitError("full196: cast missing attrs.to");
        return mlir::failure();
      }
      auto toTy = dtypeToScalarType(mlirCtx, *to);
      if (!toTy) {
        ctx.module.emitError() << "full196: cast unsupported to dtype: " << *to;
        return mlir::failure();
      }
      if (mlir::isa<mlir::IntegerType>(toTy) && mlir::cast<mlir::IntegerType>(toTy).getWidth() == 1) {
        // Cast to bool.
        if (mlir::isa<mlir::FloatType>(v.getType())) {
          auto zero = makeF32Const(b, loc, 0.0f);
          return b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, v, zero).getResult();
        }
        if (auto it = mlir::dyn_cast<mlir::IntegerType>(v.getType())) {
          auto zero = b.create<mlir::arith::ConstantIntOp>(loc, 0, it.getWidth()).getResult();
          return b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, v, zero).getResult();
        }
      }
      return castScalar(b, loc, v, toTy);
    }
    if (op.op == "reshape") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: reshape expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      auto lin = linearizeIndices(b, loc, outIdx, outShape);
      auto inIdx = delinearizeIndex(b, loc, lin, inShape);
      return emitScalar(inName, inIdx);
    }
    if (op.op == "broadcast_in_dim") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: broadcast_in_dim expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      const auto *bd = op.attrs.getArray("broadcast_dims");
      if (!bd) {
        ctx.module.emitError("full196: broadcast_in_dim missing broadcast_dims");
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> bcastDims;
      for (const auto &vv : *bd) {
        auto ii = vv.getAsInteger();
        if (!ii) {
          ctx.module.emitError("full196: broadcast_dims must be ints");
          return mlir::failure();
        }
        bcastDims.push_back(static_cast<int64_t>(*ii));
      }
      if (static_cast<int64_t>(bcastDims.size()) != static_cast<int64_t>(inShape.size())) {
        ctx.module.emitError("full196: broadcast_dims size mismatch with input rank");
        return mlir::failure();
      }
      llvm::SmallVector<mlir::Value> inIdx;
      inIdx.resize(inShape.size(), makeIndexConst(b, loc, 0));
      for (size_t a = 0; a < inShape.size(); ++a) {
        int64_t outAxis = bcastDims[a];
        if (outAxis < 0 || outAxis >= static_cast<int64_t>(outIdx.size())) {
          ctx.module.emitError("full196: broadcast_dim out of range");
          return mlir::failure();
        }
        inIdx[a] = outIdx[static_cast<size_t>(outAxis)];
      }
      return emitScalar(inName, inIdx);
    }
    if (op.op == "iota") {
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      if (axis < 0 || axis >= static_cast<int64_t>(outIdx.size())) {
        ctx.module.emitError("full196: iota axis out of range");
        return mlir::failure();
      }
      mlir::Type outTy = scalarTypeFor(op.output);
      if (!outTy) {
        ctx.module.emitError("full196: iota unsupported dtype");
        return mlir::failure();
      }
      auto idxV = outIdx[static_cast<size_t>(axis)];
      if (mlir::isa<mlir::IntegerType>(outTy))
        return b.create<mlir::arith::IndexCastOp>(loc, outTy, idxV).getResult();
      if (mlir::isa<mlir::FloatType>(outTy))
        return castScalar(b, loc, b.create<mlir::arith::IndexCastOp>(loc, b.getI32Type(), idxV).getResult(), outTy);
      return mlir::failure();
    }
    if (op.op == "where") {
      if (op.inputs.size() != 3) {
        ctx.module.emitError("full196: where expects 3 inputs");
        return mlir::failure();
      }
      auto condOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto xOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      auto yOr = emitBroadcasted(op.inputs[2], outIdx, outShape);
      if (mlir::failed(condOr) || mlir::failed(xOr) || mlir::failed(yOr))
        return mlir::failure();
      auto cond = *condOr;
      cond = castScalar(b, loc, cond, b.getI1Type());
      auto xv = *xOr;
      auto yv = *yOr;
      // Promote branches to common type.
      mlir::Type computeTy = xv.getType();
      if (computeTy != yv.getType()) {
        if (mlir::isa<mlir::FloatType>(xv.getType()) || mlir::isa<mlir::FloatType>(yv.getType())) {
          computeTy = b.getF32Type();
        } else if (xv.getType() == b.getI64Type() || yv.getType() == b.getI64Type()) {
          computeTy = b.getI64Type();
        } else {
          computeTy = b.getI32Type();
        }
      }
      xv = castScalar(b, loc, xv, computeTy);
      yv = castScalar(b, loc, yv, computeTy);
      auto sel = b.create<mlir::arith::SelectOp>(loc, cond, xv, yv).getResult();
      auto outTy = scalarTypeFor(op.output);
      if (!outTy) {
        ctx.module.emitError("full196: where unsupported output dtype");
        return mlir::failure();
      }
      return castScalar(b, loc, sel, outTy);
    }
    if (op.op == "add") {
      return emitBinNumeric(
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::AddFOp>(loc, a, bb).getResult();
          },
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::AddIOp>(loc, a, bb).getResult();
          });
    }
    if (op.op == "sub") {
      return emitBinNumeric(
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::SubFOp>(loc, a, bb).getResult();
          },
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::SubIOp>(loc, a, bb).getResult();
          });
    }
    if (op.op == "mul") {
      return emitBinNumeric(
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::MulFOp>(loc, a, bb).getResult();
          },
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::MulIOp>(loc, a, bb).getResult();
          });
    }
    if (op.op == "div") {
      return emitBinNumeric(
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::DivFOp>(loc, a, bb).getResult();
          },
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::DivSIOp>(loc, a, bb).getResult();
          });
    }
    if (op.op == "max") {
      return emitBinNumeric(
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::MaximumFOp>(loc, a, bb).getResult();
          },
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::MaxSIOp>(loc, a, bb).getResult();
          });
    }
    if (op.op == "min") {
      return emitBinNumeric(
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::MinimumFOp>(loc, a, bb).getResult();
          },
          [&](mlir::Value a, mlir::Value bb) {
            return b.create<mlir::arith::MinSIOp>(loc, a, bb).getResult();
          });
    }
    if (op.op == "abs") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: abs expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto res = b.create<mlir::math::AbsFOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "exp") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: exp expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto res = b.create<mlir::math::ExpOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "log") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: log expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto res = b.create<mlir::math::LogOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "sqrt") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: sqrt expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto res = b.create<mlir::math::SqrtOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "rsqrt") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: rsqrt expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto res = b.create<mlir::math::RsqrtOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "erf") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: erf expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto res = b.create<mlir::math::ErfOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "sin" || op.op == "cos" || op.op == "tan" || op.op == "acos" || op.op == "atan") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError() << "full196: " << op.op << " expects 1 input";
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      mlir::Value res;
      if (op.op == "sin")
        res = b.create<mlir::math::SinOp>(loc, x).getResult();
      else if (op.op == "cos")
        res = b.create<mlir::math::CosOp>(loc, x).getResult();
      else if (op.op == "tan")
        res = b.create<mlir::math::TanOp>(loc, x).getResult();
      else if (op.op == "acos")
        res = b.create<mlir::math::AcosOp>(loc, x).getResult();
      else
        res = b.create<mlir::math::AtanOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "ceil") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: ceil expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto res = b.create<mlir::math::CeilOp>(loc, x).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "pow") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: pow expects 2 inputs");
        return mlir::failure();
      }
      auto aOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto bOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto a = castScalar(b, loc, *aOr, b.getF32Type());
      auto bb = castScalar(b, loc, *bOr, b.getF32Type());
      auto res = b.create<mlir::math::PowFOp>(loc, a, bb).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "remainder") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: remainder expects 2 inputs");
        return mlir::failure();
      }
      auto aOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto bOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *aOr, b.getF32Type());
      auto y = castScalar(b, loc, *bOr, b.getF32Type());
      // torch.remainder semantics for f32: x - floor(x/y) * y
      auto div = b.create<mlir::arith::DivFOp>(loc, x, y).getResult();
      auto flo = b.create<mlir::math::FloorOp>(loc, div).getResult();
      auto prod = b.create<mlir::arith::MulFOp>(loc, flo, y).getResult();
      auto res = b.create<mlir::arith::SubFOp>(loc, x, prod).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "not") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: not expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getI1Type());
      auto one = makeI1Const(b, loc, true);
      return b.create<mlir::arith::XOrIOp>(loc, x, one).getResult();
    }
    if (op.op == "and") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: and expects 2 inputs");
        return mlir::failure();
      }
      auto aOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto bOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto a = castScalar(b, loc, *aOr, b.getI1Type());
      auto bb = castScalar(b, loc, *bOr, b.getI1Type());
      return b.create<mlir::arith::AndIOp>(loc, a, bb).getResult();
    }
    if (op.op == "bitwise_not") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: bitwise_not expects 1 input");
        return mlir::failure();
      }
      auto xOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getI64Type());
      auto all1 = makeI64Const(b, loc, -1);
      auto res = b.create<mlir::arith::XOrIOp>(loc, x, all1).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "bitwise_and" || op.op == "bitwise_or") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError() << "full196: " << op.op << " expects 2 inputs";
        return mlir::failure();
      }
      auto aOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto bOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto a = castScalar(b, loc, *aOr, b.getI64Type());
      auto bb = castScalar(b, loc, *bOr, b.getI64Type());
      mlir::Value res = (op.op == "bitwise_and")
                            ? b.create<mlir::arith::AndIOp>(loc, a, bb).getResult()
                            : b.create<mlir::arith::OrIOp>(loc, a, bb).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "bitwise_left_shift" || op.op == "bitwise_right_shift") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError() << "full196: " << op.op << " expects 2 inputs";
        return mlir::failure();
      }
      auto aOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto bOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto a = castScalar(b, loc, *aOr, b.getI64Type());
      auto sh = castScalar(b, loc, *bOr, b.getI64Type());
      mlir::Value res = (op.op == "bitwise_left_shift")
                            ? b.create<mlir::arith::ShLIOp>(loc, a, sh).getResult()
                            : b.create<mlir::arith::ShRUIOp>(loc, a, sh).getResult();
      return castScalar(b, loc, res, scalarTypeFor(op.output));
    }
    if (op.op == "eq" || op.op == "ne" || op.op == "lt" || op.op == "le" || op.op == "gt" ||
        op.op == "ge") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError() << "full196: cmp op expects 2 inputs: " << op.op;
        return mlir::failure();
      }
      auto aOr = emitBroadcasted(op.inputs[0], outIdx, outShape);
      auto bOr = emitBroadcasted(op.inputs[1], outIdx, outShape);
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto a = *aOr;
      auto bb = *bOr;
      mlir::Value pred;
      if (mlir::isa<mlir::FloatType>(a.getType()) || mlir::isa<mlir::FloatType>(bb.getType())) {
        a = castScalar(b, loc, a, b.getF32Type());
        bb = castScalar(b, loc, bb, b.getF32Type());
        mlir::arith::CmpFPredicate p = mlir::arith::CmpFPredicate::OEQ;
        if (op.op == "eq")
          p = mlir::arith::CmpFPredicate::OEQ;
        else if (op.op == "ne")
          p = mlir::arith::CmpFPredicate::UNE;
        else if (op.op == "lt")
          p = mlir::arith::CmpFPredicate::OLT;
        else if (op.op == "le")
          p = mlir::arith::CmpFPredicate::OLE;
        else if (op.op == "gt")
          p = mlir::arith::CmpFPredicate::OGT;
        else if (op.op == "ge")
          p = mlir::arith::CmpFPredicate::OGE;
        pred = b.create<mlir::arith::CmpFOp>(loc, p, a, bb).getResult();
      } else {
        a = castScalar(b, loc, a, b.getI64Type());
        bb = castScalar(b, loc, bb, b.getI64Type());
        mlir::arith::CmpIPredicate p = mlir::arith::CmpIPredicate::eq;
        if (op.op == "eq")
          p = mlir::arith::CmpIPredicate::eq;
        else if (op.op == "ne")
          p = mlir::arith::CmpIPredicate::ne;
        else if (op.op == "lt")
          p = mlir::arith::CmpIPredicate::slt;
        else if (op.op == "le")
          p = mlir::arith::CmpIPredicate::sle;
        else if (op.op == "gt")
          p = mlir::arith::CmpIPredicate::sgt;
        else if (op.op == "ge")
          p = mlir::arith::CmpIPredicate::sge;
        pred = b.create<mlir::arith::CmpIOp>(loc, p, a, bb).getResult();
      }
      // Output is bool.
      return pred;
    }
    if (op.op == "reduce_sum" || op.op == "reduce_max" || op.op == "reduce_min" || op.op == "reduce_prod" ||
        op.op == "reduce_any") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError() << "full196: " << op.op << " expects 1 input";
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      const auto *dimsArr = op.attrs.getArray("dims");
      if (!dimsArr) {
        ctx.module.emitError("full196: reduce_* missing attrs.dims");
        return mlir::failure();
      }
      std::set<int64_t> red;
      for (const auto &vv : *dimsArr) {
        auto ii = vv.getAsInteger();
        if (!ii) {
          ctx.module.emitError("full196: reduce dims must be ints");
          return mlir::failure();
        }
        red.insert(static_cast<int64_t>(*ii));
      }
      bool keepdims = false;
      if (auto kd = op.attrs.getBoolean("keepdims"))
        keepdims = *kd;
      if (auto kd = op.attrs.getBoolean("keepdim"))
        keepdims = *kd;

      // Map output indices to base input indices for non-reduced axes.
      llvm::SmallVector<mlir::Value> baseIdx;
      baseIdx.resize(inShape.size(), makeIndexConst(b, loc, 0));
      if (keepdims) {
        if (outShape.size() != inShape.size()) {
          ctx.module.emitError("full196: keepdims reduction rank mismatch");
          return mlir::failure();
        }
        for (size_t a = 0; a < inShape.size(); ++a) {
          if (red.count(static_cast<int64_t>(a)))
            continue;
          baseIdx[a] = outIdx[a];
        }
      } else {
        size_t srcAxis = 0;
        for (size_t a = 0; a < inShape.size(); ++a) {
          if (red.count(static_cast<int64_t>(a)))
            continue;
          if (srcAxis >= outIdx.size()) {
            ctx.module.emitError("full196: reduction index mapping mismatch");
            return mlir::failure();
          }
          baseIdx[a] = outIdx[srcAxis++];
        }
      }

      // Initial accumulator.
      mlir::Type outTy = scalarTypeFor(op.output);
      if (!outTy) {
        ctx.module.emitError("full196: reduce_* unsupported output dtype");
        return mlir::failure();
      }
      mlir::Value init;
      if (op.op == "reduce_any") {
        init = makeI1Const(b, loc, false);
      } else if (mlir::isa<mlir::FloatType>(outTy)) {
        if (op.op == "reduce_sum")
          init = makeF32Const(b, loc, 0.0f);
        else if (op.op == "reduce_prod")
          init = makeF32Const(b, loc, 1.0f);
        else if (op.op == "reduce_max")
          init = makeF32Const(b, loc, -3.402823466e+38f);
        else
          init = makeF32Const(b, loc, 3.402823466e+38f);
      } else {
        auto it = mlir::cast<mlir::IntegerType>(outTy);
        if (op.op == "reduce_sum")
          init = b.create<mlir::arith::ConstantIntOp>(loc, 0, it.getWidth()).getResult();
        else if (op.op == "reduce_prod")
          init = b.create<mlir::arith::ConstantIntOp>(loc, 1, it.getWidth()).getResult();
        else if (op.op == "reduce_max")
          init = b.create<mlir::arith::ConstantIntOp>(loc, std::numeric_limits<int64_t>::min(), it.getWidth())
                     .getResult();
        else
          init = b.create<mlir::arith::ConstantIntOp>(loc, std::numeric_limits<int64_t>::max(), it.getWidth())
                     .getResult();
      }

      // Build nested loops over reduced axes.
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);

      llvm::SmallVector<int64_t> redAxesSorted(red.begin(), red.end());
      std::sort(redAxesSorted.begin(), redAxesSorted.end());

      std::function<mlir::Value(size_t, mlir::Value)> buildLoop = [&](size_t i, mlir::Value acc) -> mlir::Value {
        if (i >= redAxesSorted.size()) {
          // Leaf: load input element and accumulate.
          auto vOr = emitScalar(inName, baseIdx);
          if (mlir::failed(vOr))
            return acc;
          auto vv = castScalar(b, loc, *vOr, outTy);
          if (op.op == "reduce_sum") {
            if (mlir::isa<mlir::FloatType>(outTy))
              return b.create<mlir::arith::AddFOp>(loc, acc, vv).getResult();
            return b.create<mlir::arith::AddIOp>(loc, acc, vv).getResult();
          }
          if (op.op == "reduce_prod") {
            if (mlir::isa<mlir::FloatType>(outTy))
              return b.create<mlir::arith::MulFOp>(loc, acc, vv).getResult();
            return b.create<mlir::arith::MulIOp>(loc, acc, vv).getResult();
          }
          if (op.op == "reduce_max") {
            if (mlir::isa<mlir::FloatType>(outTy))
              return b.create<mlir::arith::MaximumFOp>(loc, acc, vv).getResult();
            return b.create<mlir::arith::MaxSIOp>(loc, acc, vv).getResult();
          }
          if (op.op == "reduce_min") {
            if (mlir::isa<mlir::FloatType>(outTy))
              return b.create<mlir::arith::MinimumFOp>(loc, acc, vv).getResult();
            return b.create<mlir::arith::MinSIOp>(loc, acc, vv).getResult();
          }
          // reduce_any
          auto vb = castScalar(b, loc, *vOr, b.getI1Type());
          return b.create<mlir::arith::OrIOp>(loc, acc, vb).getResult();
        }

        int64_t axis = redAxesSorted[i];
        if (axis < 0 || axis >= static_cast<int64_t>(inShape.size())) {
          return acc;
        }
        auto ub = makeIndexConst(b, loc, inShape[static_cast<size_t>(axis)]);
        auto forOp = b.create<mlir::scf::ForOp>(loc, c0, ub, c1, mlir::ValueRange{acc});
        b.setInsertionPointToStart(forOp.getBody());
        auto iv = forOp.getInductionVar();
        baseIdx[static_cast<size_t>(axis)] = iv;
        auto nextAcc = buildLoop(i + 1, forOp.getRegionIterArgs()[0]);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{nextAcc});
        b.setInsertionPointAfter(forOp);
        return forOp.getResult(0);
      };

      auto reduced = buildLoop(0, init);
      return reduced;
    }

    if (op.op == "gather") {
      if (op.inputs.size() < 2) {
        ctx.module.emitError("full196: gather expects data + at least 1 index");
        return mlir::failure();
      }
      auto dataName = op.inputs[0];
      auto dataShape = shapeOf(dataName);
      llvm::SmallVector<mlir::Value> dataIdx;
      dataIdx.reserve(op.inputs.size() - 1);
      for (size_t a = 1; a < op.inputs.size(); ++a) {
        auto idxName = op.inputs[a];
        auto idxOr = emitBroadcasted(idxName, outIdx, outShape);
        if (mlir::failed(idxOr))
          return mlir::failure();
        auto idxVal = *idxOr;
        if (isBoolDtype(dtypeOf(idxName))) {
          idxVal = castScalar(b, loc, idxVal, b.getI32Type());
        } else if (mlir::isa<mlir::FloatType>(idxVal.getType())) {
          idxVal = castScalar(b, loc, idxVal, b.getI32Type());
        } else if (mlir::isa<mlir::IntegerType>(idxVal.getType()) &&
                   mlir::cast<mlir::IntegerType>(idxVal.getType()).getWidth() != 32) {
          idxVal = castScalar(b, loc, idxVal, b.getI32Type());
        }
        dataIdx.push_back(castIntToIndex(b, loc, idxVal));
      }
      if (dataIdx.size() != dataShape.size()) {
        ctx.module.emitError("full196: gather index count must match data rank");
        return mlir::failure();
      }
      return emitScalar(dataName, dataIdx);
    }

    if (op.op == "concat" || op.op == "stack") {
      if (op.inputs.size() < 2) {
        ctx.module.emitError() << "full196: " << op.op << " expects >=2 inputs";
        return mlir::failure();
      }
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      if (axis < 0 || axis >= static_cast<int64_t>(outShape.size())) {
        ctx.module.emitError("full196: concat/stack axis out of range");
        return mlir::failure();
      }

      if (op.op == "stack") {
        // New dimension at `axis` selects which input.
        auto which = outIdx[static_cast<size_t>(axis)];
        auto c0 = makeIndexConst(b, loc, 0);
        mlir::Value outV;
        // Build base indices for input by removing axis.
        llvm::SmallVector<mlir::Value> inIdx;
        inIdx.reserve(outShape.size() - 1);
        for (size_t a = 0; a < outShape.size(); ++a) {
          if (static_cast<int64_t>(a) == axis)
            continue;
          inIdx.push_back(outIdx[a]);
        }
        for (size_t i = 0; i < op.inputs.size(); ++i) {
          auto ci = makeIndexConst(b, loc, static_cast<int64_t>(i));
          auto isI = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, which, ci).getResult();
          if (i == 0) {
            auto vOr = emitScalar(op.inputs[i], inIdx);
            if (mlir::failed(vOr))
              return mlir::failure();
            outV = *vOr;
          } else {
            auto vOr = emitScalar(op.inputs[i], inIdx);
            if (mlir::failed(vOr))
              return mlir::failure();
            outV = b.create<mlir::arith::SelectOp>(loc, isI, *vOr, outV).getResult();
          }
        }
        return outV;
      }

      // concat
      llvm::SmallVector<int64_t> offsets;
      offsets.reserve(op.inputs.size() + 1);
      offsets.push_back(0);
      int64_t running = 0;
      for (const auto &inName : op.inputs) {
        auto sh = shapeOf(inName);
        if (sh.empty() || axis >= static_cast<int64_t>(sh.size())) {
          ctx.module.emitError("full196: concat input rank mismatch");
          return mlir::failure();
        }
        running += sh[static_cast<size_t>(axis)];
        offsets.push_back(running);
      }
      auto ax = outIdx[static_cast<size_t>(axis)];
      // Default to last input.
      size_t selected = op.inputs.size() - 1;
      mlir::Value axLocal = ax;
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        int64_t lo = offsets[i];
        int64_t hi = offsets[i + 1];
        auto clo = makeIndexConst(b, loc, lo);
        auto chi = makeIndexConst(b, loc, hi);
        auto geLo = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, ax, clo).getResult();
        auto ltHi = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, ax, chi).getResult();
        auto inRange = b.create<mlir::arith::AndIOp>(loc, geLo, ltHi).getResult();
        // Compute candidate local axis index.
        auto local = b.create<mlir::arith::SubIOp>(loc, ax, clo).getResult();
        if (i == 0) {
          selected = 0;
          axLocal = local;
        } else {
          axLocal = b.create<mlir::arith::SelectOp>(loc, inRange, local, axLocal).getResult();
          selected = selected; // keep for value select below.
        }
      }
      // Select value.
      mlir::Value outV;
      for (size_t i = 0; i < op.inputs.size(); ++i) {
        int64_t lo = offsets[i];
        int64_t hi = offsets[i + 1];
        auto clo = makeIndexConst(b, loc, lo);
        auto chi = makeIndexConst(b, loc, hi);
        auto geLo = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, ax, clo).getResult();
        auto ltHi = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, ax, chi).getResult();
        auto inRange = b.create<mlir::arith::AndIOp>(loc, geLo, ltHi).getResult();
        llvm::SmallVector<mlir::Value> inIdx(outIdx.begin(), outIdx.end());
        inIdx[static_cast<size_t>(axis)] = b.create<mlir::arith::SubIOp>(loc, ax, clo).getResult();
        auto vOr = emitScalar(op.inputs[i], mapIndicesNumpyBroadcast(b, loc, inIdx, outShape, shapeOf(op.inputs[i])));
        if (mlir::failed(vOr))
          return mlir::failure();
        if (i == 0) {
          outV = *vOr;
        } else {
          outV = b.create<mlir::arith::SelectOp>(loc, inRange, *vOr, outV).getResult();
        }
      }
      return outV;
    }

    if (op.op == "pad") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: pad expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      const auto *padWidth = op.attrs.getObject("pad_width");
      if (!padWidth) {
        ctx.module.emitError("full196: pad missing attrs.pad_width");
        return mlir::failure();
      }
      const auto *pairs = padWidth->getArray("pairs");
      if (!pairs || pairs->size() != inShape.size()) {
        ctx.module.emitError("full196: pad expects pad_width.pairs per input dim");
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> before;
      before.reserve(inShape.size());
      for (const auto &pv : *pairs) {
        auto arr = pv.getAsArray();
        if (!arr || arr->size() != 2) {
          ctx.module.emitError("full196: pad_width pairs must be [before,after]");
          return mlir::failure();
        }
        auto b0 = (*arr)[0].getAsInteger();
        if (!b0) {
          ctx.module.emitError("full196: pad before must be int");
          return mlir::failure();
        }
        before.push_back(static_cast<int64_t>(*b0));
      }
      float padValF = 0.0f;
      if (auto num = op.attrs.getNumber("value"))
        padValF = static_cast<float>(*num);
      auto padVal = makeF32Const(b, loc, padValF);

      // Compute in-bounds predicate and input indices.
      auto c0 = makeIndexConst(b, loc, 0);
      mlir::Value pred = makeI1Const(b, loc, true);
      llvm::SmallVector<mlir::Value> inIdx;
      inIdx.resize(inShape.size(), c0);
      for (size_t a = 0; a < inShape.size(); ++a) {
        auto off = makeIndexConst(b, loc, before[a]);
        auto idxIn = b.create<mlir::arith::SubIOp>(loc, outIdx[a], off).getResult();
        inIdx[a] = idxIn;
        auto ge0 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, idxIn, c0).getResult();
        auto ltN =
            b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, idxIn,
                                          makeIndexConst(b, loc, inShape[a]))
                .getResult();
        pred = b.create<mlir::arith::AndIOp>(loc, pred, b.create<mlir::arith::AndIOp>(loc, ge0, ltN).getResult())
                   .getResult();
      }
      auto vOr = emitScalar(inName, inIdx);
      if (mlir::failed(vOr))
        return mlir::failure();
      auto v = castScalar(b, loc, *vOr, b.getF32Type());
      auto sel = b.create<mlir::arith::SelectOp>(loc, pred, v, padVal).getResult();
      return castScalar(b, loc, sel, scalarTypeFor(op.output));
    }

    if (op.op == "polar") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: polar expects 2 inputs");
        return mlir::failure();
      }
      // out[...,0]=abs*cos(angle); out[...,1]=abs*sin(angle)
      if (outIdx.empty()) {
        ctx.module.emitError("full196: polar expects rank>=1 output");
        return mlir::failure();
      }
      auto last = outIdx.back();
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto is0 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, last, c0).getResult();
      llvm::SmallVector<mlir::Value> baseIdx(outIdx.begin(), outIdx.end() - 1);
      auto absOr = emitScalar(op.inputs[0], baseIdx);
      auto angOr = emitScalar(op.inputs[1], baseIdx);
      if (mlir::failed(absOr) || mlir::failed(angOr))
        return mlir::failure();
      auto absV = castScalar(b, loc, *absOr, b.getF32Type());
      auto angV = castScalar(b, loc, *angOr, b.getF32Type());
      auto cosV = b.create<mlir::math::CosOp>(loc, angV).getResult();
      auto sinV = b.create<mlir::math::SinOp>(loc, angV).getResult();
      auto a0 = b.create<mlir::arith::MulFOp>(loc, absV, cosV).getResult();
      auto a1 = b.create<mlir::arith::MulFOp>(loc, absV, sinV).getResult();
      auto outF = b.create<mlir::arith::SelectOp>(loc, is0, a0, a1).getResult();
      return castScalar(b, loc, outF, scalarTypeFor(op.output));
    }

    if (op.op == "kron") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: kron expects 2 inputs");
        return mlir::failure();
      }
      auto A = op.inputs[0];
      auto B = op.inputs[1];
      auto shA = shapeOf(A);
      auto shB = shapeOf(B);
      if (shA.size() != 2 || shB.size() != 2 || outShape.size() != 2) {
        ctx.module.emitError("full196: kron supports rank2 only");
        return mlir::failure();
      }
      auto P = makeIndexConst(b, loc, shB[0]);
      auto Q = makeIndexConst(b, loc, shB[1]);
      auto i = outIdx[0];
      auto j = outIdx[1];
      auto ia = b.create<mlir::arith::DivUIOp>(loc, i, P).getResult();
      auto ib = b.create<mlir::arith::RemUIOp>(loc, i, P).getResult();
      auto ja = b.create<mlir::arith::DivUIOp>(loc, j, Q).getResult();
      auto jb = b.create<mlir::arith::RemUIOp>(loc, j, Q).getResult();
      auto aOr = emitScalar(A, llvm::ArrayRef<mlir::Value>{ia, ja});
      auto bOr = emitScalar(B, llvm::ArrayRef<mlir::Value>{ib, jb});
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto av = castScalar(b, loc, *aOr, b.getF32Type());
      auto bv = castScalar(b, loc, *bOr, b.getF32Type());
      auto prod = b.create<mlir::arith::MulFOp>(loc, av, bv).getResult();
      return castScalar(b, loc, prod, scalarTypeFor(op.output));
    }

    if (op.op == "matmul") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: matmul expects 2 inputs");
        return mlir::failure();
      }
      auto A = op.inputs[0];
      auto Bn = op.inputs[1];
      auto shA = shapeOf(A);
      auto shB = shapeOf(Bn);
      if (shA.empty() || shB.empty()) {
        ctx.module.emitError("full196: matmul missing shapes");
        return mlir::failure();
      }
      // Support rank2@rank2, rank3@rank3, rank2@rank1 (mv).
      if (shA.size() == 2 && shB.size() == 1 && outShape.size() == 1) {
        int64_t M = shA[0];
        int64_t K = shA[1];
        if (shB[0] != K) {
          ctx.module.emitError("full196: mv matmul shape mismatch");
          return mlir::failure();
        }
        auto i = outIdx[0];
        auto c0 = makeIndexConst(b, loc, 0);
        auto c1 = makeIndexConst(b, loc, 1);
        auto cK = makeIndexConst(b, loc, K);
        auto acc0 = makeF32Const(b, loc, 0.0f);
        auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{acc0});
        b.setInsertionPointToStart(forOp.getBody());
        auto kk = forOp.getInductionVar();
        auto acc = forOp.getRegionIterArgs()[0];
        auto aOr = emitScalar(A, llvm::ArrayRef<mlir::Value>{i, kk});
        auto bOr = emitScalar(Bn, llvm::ArrayRef<mlir::Value>{kk});
        auto av = castScalar(b, loc, mlir::failed(aOr) ? acc0 : *aOr, b.getF32Type());
        auto bv = castScalar(b, loc, mlir::failed(bOr) ? acc0 : *bOr, b.getF32Type());
        auto prod = b.create<mlir::arith::MulFOp>(loc, av, bv).getResult();
        auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, prod).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
        b.setInsertionPointAfter(forOp);
        return forOp.getResult(0);
      }
      if (shA.size() == 2 && shB.size() == 2 && outShape.size() == 2) {
        bool ta = false;
        bool tb = false;
        if (auto v = op.attrs.getBoolean("transpose_a"))
          ta = *v;
        if (auto v = op.attrs.getBoolean("transpose_b"))
          tb = *v;
        int64_t aM = ta ? shA[1] : shA[0];
        int64_t aK = ta ? shA[0] : shA[1];
        int64_t bK = tb ? shB[1] : shB[0];
        int64_t bN = tb ? shB[0] : shB[1];
        if (aK != bK) {
          ctx.module.emitError("full196: matmul K mismatch");
          return mlir::failure();
        }
        auto i = outIdx[0];
        auto j = outIdx[1];
        auto c0 = makeIndexConst(b, loc, 0);
        auto c1 = makeIndexConst(b, loc, 1);
        auto cK = makeIndexConst(b, loc, aK);
        auto acc0 = makeF32Const(b, loc, 0.0f);
        auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{acc0});
        b.setInsertionPointToStart(forOp.getBody());
        auto kk = forOp.getInductionVar();
        auto acc = forOp.getRegionIterArgs()[0];
        llvm::SmallVector<mlir::Value> aIdx = ta ? llvm::SmallVector<mlir::Value>{kk, i}
                                                 : llvm::SmallVector<mlir::Value>{i, kk};
        llvm::SmallVector<mlir::Value> bIdx = tb ? llvm::SmallVector<mlir::Value>{j, kk}
                                                 : llvm::SmallVector<mlir::Value>{kk, j};
        auto aOr = emitScalar(A, aIdx);
        auto bOr = emitScalar(Bn, bIdx);
        auto av = castScalar(b, loc, mlir::failed(aOr) ? acc0 : *aOr, b.getF32Type());
        auto bv = castScalar(b, loc, mlir::failed(bOr) ? acc0 : *bOr, b.getF32Type());
        auto prod = b.create<mlir::arith::MulFOp>(loc, av, bv).getResult();
        auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, prod).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
        b.setInsertionPointAfter(forOp);
        return forOp.getResult(0);
      }
      if (shA.size() == 3 && shB.size() == 3 && outShape.size() == 3) {
        int64_t B = shA[0];
        int64_t M = shA[1];
        int64_t K = shA[2];
        if (shB[0] != B || shB[1] != K) {
          ctx.module.emitError("full196: bmm shape mismatch");
          return mlir::failure();
        }
        int64_t N = shB[2];
        auto b0 = outIdx[0];
        auto i = outIdx[1];
        auto j = outIdx[2];
        auto c0 = makeIndexConst(b, loc, 0);
        auto c1 = makeIndexConst(b, loc, 1);
        auto cK = makeIndexConst(b, loc, K);
        auto acc0 = makeF32Const(b, loc, 0.0f);
        auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{acc0});
        b.setInsertionPointToStart(forOp.getBody());
        auto kk = forOp.getInductionVar();
        auto acc = forOp.getRegionIterArgs()[0];
        auto aOr = emitScalar(A, llvm::ArrayRef<mlir::Value>{b0, i, kk});
        auto bOr = emitScalar(Bn, llvm::ArrayRef<mlir::Value>{b0, kk, j});
        auto av = castScalar(b, loc, mlir::failed(aOr) ? acc0 : *aOr, b.getF32Type());
        auto bv = castScalar(b, loc, mlir::failed(bOr) ? acc0 : *bOr, b.getF32Type());
        auto prod = b.create<mlir::arith::MulFOp>(loc, av, bv).getResult();
        auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, prod).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
        b.setInsertionPointAfter(forOp);
        return forOp.getResult(0);
      }
      ctx.module.emitError("full196: matmul unsupported ranks");
      return mlir::failure();
    }

    if (op.op == "cumsum") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: cumsum expects 1 input");
        return mlir::failure();
      }
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      if (axis < 0 || axis >= static_cast<int64_t>(inShape.size())) {
        ctx.module.emitError("full196: cumsum axis out of range");
        return mlir::failure();
      }
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto ub = b.create<mlir::arith::AddIOp>(loc, outIdx[static_cast<size_t>(axis)], c1).getResult();
      auto acc0 = makeF32Const(b, loc, 0.0f);
      auto baseIdx = llvm::SmallVector<mlir::Value>(outIdx.begin(), outIdx.end());
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, ub, c1, mlir::ValueRange{acc0});
      b.setInsertionPointToStart(forOp.getBody());
      auto iv = forOp.getInductionVar();
      auto acc = forOp.getRegionIterArgs()[0];
      baseIdx[static_cast<size_t>(axis)] = iv;
      auto vOr = emitScalar(inName, baseIdx);
      auto vv = castScalar(b, loc, mlir::failed(vOr) ? acc0 : *vOr, b.getF32Type());
      auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, vv).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
      b.setInsertionPointAfter(forOp);
      return castScalar(b, loc, forOp.getResult(0), scalarTypeFor(op.output));
    }

    if (op.op == "cummax" || op.op == "cummin") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError() << "full196: " << op.op << " expects 1 input";
        return mlir::failure();
      }
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      if (axis < 0 || axis >= static_cast<int64_t>(inShape.size())) {
        ctx.module.emitError("full196: cummax/min axis out of range");
        return mlir::failure();
      }
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto ub = b.create<mlir::arith::AddIOp>(loc, outIdx[static_cast<size_t>(axis)], c1).getResult();
      auto init = (op.op == "cummax") ? makeF32Const(b, loc, -3.402823466e+38f)
                                      : makeF32Const(b, loc, 3.402823466e+38f);
      auto baseIdx = llvm::SmallVector<mlir::Value>(outIdx.begin(), outIdx.end());
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, ub, c1, mlir::ValueRange{init});
      b.setInsertionPointToStart(forOp.getBody());
      auto iv = forOp.getInductionVar();
      auto acc = forOp.getRegionIterArgs()[0];
      baseIdx[static_cast<size_t>(axis)] = iv;
      auto vOr = emitScalar(inName, baseIdx);
      auto vv = castScalar(b, loc, mlir::failed(vOr) ? init : *vOr, b.getF32Type());
      auto acc2 = (op.op == "cummax") ? b.create<mlir::arith::MaximumFOp>(loc, acc, vv).getResult()
                                      : b.create<mlir::arith::MinimumFOp>(loc, acc, vv).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
      b.setInsertionPointAfter(forOp);
      return castScalar(b, loc, forOp.getResult(0), scalarTypeFor(op.output));
    }

    if (op.op == "std") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: std expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      if (axis < 0 || axis >= static_cast<int64_t>(inShape.size())) {
        ctx.module.emitError("full196: std axis out of range");
        return mlir::failure();
      }
      int64_t correction = 0;
      if (auto ii = op.attrs.getInteger("correction"))
        correction = static_cast<int64_t>(*ii);
      // Only support std along last axis for our coverage graphs.
      int64_t N = inShape[static_cast<size_t>(axis)];
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cN = makeIndexConst(b, loc, N);
      auto sum0 = makeF32Const(b, loc, 0.0f);
      auto baseIdx = llvm::SmallVector<mlir::Value>(inShape.size(), c0);
      // Map output indices into baseIdx for non-reduced axes.
      if (outShape.size() + 1 == inShape.size()) {
        size_t src = 0;
        for (size_t a = 0; a < inShape.size(); ++a) {
          if (static_cast<int64_t>(a) == axis)
            continue;
          baseIdx[a] = outIdx[src++];
        }
      }

      auto sumFor = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{sum0});
      b.setInsertionPointToStart(sumFor.getBody());
      auto iv = sumFor.getInductionVar();
      auto acc = sumFor.getRegionIterArgs()[0];
      baseIdx[static_cast<size_t>(axis)] = iv;
      auto vOr = emitScalar(inName, baseIdx);
      auto vv = castScalar(b, loc, mlir::failed(vOr) ? sum0 : *vOr, b.getF32Type());
      auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, vv).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
      b.setInsertionPointAfter(sumFor);
      auto sum = sumFor.getResult(0);
      auto denom = makeF32Const(b, loc, static_cast<float>(N));
      auto mean = b.create<mlir::arith::DivFOp>(loc, sum, denom).getResult();

      auto var0 = makeF32Const(b, loc, 0.0f);
      auto varFor = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{var0});
      b.setInsertionPointToStart(varFor.getBody());
      auto iv2 = varFor.getInductionVar();
      auto vacc = varFor.getRegionIterArgs()[0];
      baseIdx[static_cast<size_t>(axis)] = iv2;
      auto v2Or = emitScalar(inName, baseIdx);
      auto v2 = castScalar(b, loc, mlir::failed(v2Or) ? var0 : *v2Or, b.getF32Type());
      auto diff = b.create<mlir::arith::SubFOp>(loc, v2, mean).getResult();
      auto sq = b.create<mlir::arith::MulFOp>(loc, diff, diff).getResult();
      auto vacc2 = b.create<mlir::arith::AddFOp>(loc, vacc, sq).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{vacc2});
      b.setInsertionPointAfter(varFor);
      auto varSum = varFor.getResult(0);

      int64_t denomN = std::max<int64_t>(1, N - correction);
      auto denomVar = makeF32Const(b, loc, static_cast<float>(denomN));
      auto var = b.create<mlir::arith::DivFOp>(loc, varSum, denomVar).getResult();
      auto stdv = b.create<mlir::math::SqrtOp>(loc, var).getResult();
      return castScalar(b, loc, stdv, scalarTypeFor(op.output));
    }

    if (op.op == "softmax") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: softmax expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      if (axis < 0 || axis >= static_cast<int64_t>(inShape.size())) {
        ctx.module.emitError("full196: softmax axis out of range");
        return mlir::failure();
      }
      int64_t N = inShape[static_cast<size_t>(axis)];
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cN = makeIndexConst(b, loc, N);
      auto negInf = makeF32Const(b, loc, -3.402823466e+38f);
      auto baseIdx = llvm::SmallVector<mlir::Value>(inShape.size(), c0);
      // Map out indices to base indices for non-axis dims.
      for (size_t a = 0; a < inShape.size(); ++a) {
        if (static_cast<int64_t>(a) == axis)
          continue;
        baseIdx[a] = outIdx[a];
      }

      // max
      auto maxFor = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{negInf});
      b.setInsertionPointToStart(maxFor.getBody());
      auto iv = maxFor.getInductionVar();
      auto acc = maxFor.getRegionIterArgs()[0];
      baseIdx[static_cast<size_t>(axis)] = iv;
      auto vOr = emitScalar(inName, baseIdx);
      auto vv = castScalar(b, loc, mlir::failed(vOr) ? negInf : *vOr, b.getF32Type());
      auto acc2 = b.create<mlir::arith::MaximumFOp>(loc, acc, vv).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
      b.setInsertionPointAfter(maxFor);
      auto maxv = maxFor.getResult(0);

      // sum exp
      auto sum0 = makeF32Const(b, loc, 0.0f);
      auto sumFor = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{sum0});
      b.setInsertionPointToStart(sumFor.getBody());
      auto iv2 = sumFor.getInductionVar();
      auto accS = sumFor.getRegionIterArgs()[0];
      baseIdx[static_cast<size_t>(axis)] = iv2;
      auto v2Or = emitScalar(inName, baseIdx);
      auto v2 = castScalar(b, loc, mlir::failed(v2Or) ? sum0 : *v2Or, b.getF32Type());
      auto centered = b.create<mlir::arith::SubFOp>(loc, v2, maxv).getResult();
      auto e = b.create<mlir::math::ExpOp>(loc, centered).getResult();
      auto accS2 = b.create<mlir::arith::AddFOp>(loc, accS, e).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accS2});
      b.setInsertionPointAfter(sumFor);
      auto denom = sumFor.getResult(0);

      // exp(x-max)/denom for this element.
      auto xOr = emitScalar(inName, outIdx);
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto centered2 = b.create<mlir::arith::SubFOp>(loc, x, maxv).getResult();
      auto num = b.create<mlir::math::ExpOp>(loc, centered2).getResult();
      auto outV = b.create<mlir::arith::DivFOp>(loc, num, denom).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "argmax" || op.op == "argmin") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError() << "full196: " << op.op << " expects 1 input";
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto inShape = shapeOf(inName);
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      if (inShape.size() != 2 || axis != 1 || outShape.size() != 1) {
        ctx.module.emitError("full196: argmax/min supports rank2 axis=1 only");
        return mlir::failure();
      }
      int64_t N = inShape[1];
      auto row = outIdx[0];
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cN = makeIndexConst(b, loc, N);
      auto initVal = (op.op == "argmax") ? makeF32Const(b, loc, -3.402823466e+38f)
                                         : makeF32Const(b, loc, 3.402823466e+38f);
      auto initIdx = makeIndexConst(b, loc, 0);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{initVal, initIdx});
      b.setInsertionPointToStart(forOp.getBody());
      auto j = forOp.getInductionVar();
      auto bestV = forOp.getRegionIterArgs()[0];
      auto bestI = forOp.getRegionIterArgs()[1];
      auto vOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{row, j});
      auto v = castScalar(b, loc, mlir::failed(vOr) ? initVal : *vOr, b.getF32Type());
      mlir::Value better;
      if (op.op == "argmax") {
        auto gt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, v, bestV).getResult();
        auto eq = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, v, bestV).getResult();
        auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, j, bestI).getResult();
        better = b.create<mlir::arith::OrIOp>(loc, gt, b.create<mlir::arith::AndIOp>(loc, eq, lt).getResult())
                     .getResult();
      } else {
        auto lt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, v, bestV).getResult();
        auto eq = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, v, bestV).getResult();
        auto lt2 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, j, bestI).getResult();
        better = b.create<mlir::arith::OrIOp>(loc, lt, b.create<mlir::arith::AndIOp>(loc, eq, lt2).getResult())
                     .getResult();
      }
      auto bestV2 = b.create<mlir::arith::SelectOp>(loc, better, v, bestV).getResult();
      auto bestI2 = b.create<mlir::arith::SelectOp>(loc, better, j, bestI).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{bestV2, bestI2});
      b.setInsertionPointAfter(forOp);
      auto idxOut = castIndexToInt(b, loc, forOp.getResult(1), 32);
      return castScalar(b, loc, idxOut, scalarTypeFor(op.output));
    }

    if (op.op == "conv1d" || op.op == "conv2d" || op.op == "conv3d" || op.op == "conv_depthwise2d") {
      // correctness-first naive convs (N[C]... layouts used by seeds).
      if (op.inputs.size() != 3) {
        ctx.module.emitError() << "full196: " << op.op << " expects inputs (input, weight, bias)";
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto wName = op.inputs[1];
      auto bName = op.inputs[2];
      auto shIn = shapeOf(inName);
      auto shW = shapeOf(wName);
      auto shOut = outShape;
      if (op.op == "conv1d") {
        // [N,C_IN,L] * [C_OUT,C_PER_G,K] + bias[C_OUT] -> [N,C_OUT,OL]
        if (shIn.size() != 3 || shW.size() != 3 || shOut.size() != 3) {
          ctx.module.emitError("full196: conv1d rank mismatch");
          return mlir::failure();
        }
        int64_t stride = static_cast<int64_t>(op.attrs.getInteger("stride").value_or(1));
        int64_t padding = static_cast<int64_t>(op.attrs.getInteger("padding").value_or(0));
        int64_t dilation = static_cast<int64_t>(op.attrs.getInteger("dilation").value_or(1));
        int64_t groups = static_cast<int64_t>(op.attrs.getInteger("groups").value_or(1));
        int64_t N = shIn[0], C_IN = shIn[1], L = shIn[2];
        int64_t C_OUT = shW[0], C_PER_G = shW[1], K = shW[2];
        auto n = outIdx[0];
        auto oc = outIdx[1];
        auto ol = outIdx[2];
        auto biasOr = emitScalar(bName, llvm::ArrayRef<mlir::Value>{oc});
        auto acc0 = castScalar(b, loc, mlir::failed(biasOr) ? makeF32Const(b, loc, 0.0f) : *biasOr, b.getF32Type());
        auto c0 = makeIndexConst(b, loc, 0);
        auto c1 = makeIndexConst(b, loc, 1);
        auto cCPG = makeIndexConst(b, loc, C_PER_G);
        auto accFor = b.create<mlir::scf::ForOp>(loc, c0, cCPG, c1, mlir::ValueRange{acc0});
        b.setInsertionPointToStart(accFor.getBody());
        auto icg = accFor.getInductionVar();
        auto acc = accFor.getRegionIterArgs()[0];
        auto cK = makeIndexConst(b, loc, K);
        auto kFor = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{acc});
        b.setInsertionPointToStart(kFor.getBody());
        auto kk = kFor.getInductionVar();
        auto acc2 = kFor.getRegionIterArgs()[0];
        // in_c = (oc / (C_OUT/groups))*C_PER_G + icg
        int64_t mult = (groups > 0) ? (C_OUT / groups) : C_OUT;
        auto cMult = makeIndexConst(b, loc, std::max<int64_t>(1, mult));
        auto g = b.create<mlir::arith::DivUIOp>(loc, oc, cMult).getResult();
        auto baseC = b.create<mlir::arith::MulIOp>(loc, g, makeIndexConst(b, loc, C_PER_G)).getResult();
        auto ic = b.create<mlir::arith::AddIOp>(loc, baseC, icg).getResult();
        auto il0 = b.create<mlir::arith::MulIOp>(loc, ol, makeIndexConst(b, loc, stride)).getResult();
        auto il1 = b.create<mlir::arith::AddIOp>(loc, il0,
                                                 b.create<mlir::arith::MulIOp>(loc, kk, makeIndexConst(b, loc, dilation))
                                                     .getResult())
                       .getResult();
        auto il = b.create<mlir::arith::SubIOp>(loc, il1, makeIndexConst(b, loc, padding)).getResult();
        auto ge0 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, il, c0).getResult();
        auto ltL =
            b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, il, makeIndexConst(b, loc, L))
                .getResult();
        auto ok = b.create<mlir::arith::AndIOp>(loc, ge0, ltL).getResult();
        auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type()}, ok, /*withElse=*/true);
        b.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto xOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, ic, il});
        auto wOr = emitScalar(wName, llvm::ArrayRef<mlir::Value>{oc, icg, kk});
        auto xv = castScalar(b, loc, mlir::failed(xOr) ? makeF32Const(b, loc, 0.0f) : *xOr, b.getF32Type());
        auto wv = castScalar(b, loc, mlir::failed(wOr) ? makeF32Const(b, loc, 0.0f) : *wOr, b.getF32Type());
        auto prod = b.create<mlir::arith::MulFOp>(loc, xv, wv).getResult();
        auto sum = b.create<mlir::arith::AddFOp>(loc, acc2, prod).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum});
        b.setInsertionPointToStart(&ifOp.getElseRegion().front());
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
        b.setInsertionPointAfter(ifOp);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0)});
        b.setInsertionPointAfter(kFor);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kFor.getResult(0)});
        b.setInsertionPointAfter(accFor);
        return castScalar(b, loc, accFor.getResult(0), scalarTypeFor(op.output));
      }

      if (op.op == "conv2d" || op.op == "conv_depthwise2d") {
        // [N,C_IN,H,W] * [C_OUT,C_PER_G,KH,KW] + bias[C_OUT] -> [N,C_OUT,OH,OW]
        if (shIn.size() != 4 || shW.size() != 4 || shOut.size() != 4) {
          ctx.module.emitError() << "full196: " << op.op << " rank mismatch";
          return mlir::failure();
        }

        auto strideOr = resolveIntListParam(op, "stride", 2, 1);
        auto paddingOr = resolveIntListParam(op, "padding", 2, 0);
        auto dilationOr = resolveIntListParam(op, "dilation", 2, 1);
        if (mlir::failed(strideOr) || mlir::failed(paddingOr) || mlir::failed(dilationOr))
          return mlir::failure();
        int64_t SH = (*strideOr)[0], SW = (*strideOr)[1];
        int64_t PH = (*paddingOr)[0], PW = (*paddingOr)[1];
        int64_t DH = (*dilationOr)[0], DW = (*dilationOr)[1];
        int64_t groups = 1;
        if (const llvm::json::Value *vv = op.attrs.get("groups")) {
          auto gOr = resolveIntParam(*vv, "attrs.groups");
          if (mlir::failed(gOr))
            return mlir::failure();
          groups = *gOr;
        } else if (op.op == "conv_depthwise2d") {
          // Depthwise conv seeds omit groups; default to groups=C_IN.
          groups = shIn[1];
        }

        int64_t N = shIn[0], C_IN = shIn[1], H = shIn[2], W = shIn[3];
        int64_t C_OUT = shW[0], C_PER_G = shW[1], KH = shW[2], KW = shW[3];
        (void)N;
        (void)C_IN;
        auto n = outIdx[0];
        auto oc = outIdx[1];
        auto oh = outIdx[2];
        auto ow = outIdx[3];

        auto biasOr = emitScalar(bName, llvm::ArrayRef<mlir::Value>{oc});
        auto acc0 =
            castScalar(b, loc, mlir::failed(biasOr) ? makeF32Const(b, loc, 0.0f) : *biasOr, b.getF32Type());

        auto c0 = makeIndexConst(b, loc, 0);
        auto c1 = makeIndexConst(b, loc, 1);
        auto cCPG = makeIndexConst(b, loc, C_PER_G);
        auto accFor = b.create<mlir::scf::ForOp>(loc, c0, cCPG, c1, mlir::ValueRange{acc0});
        b.setInsertionPointToStart(accFor.getBody());
        auto icg = accFor.getInductionVar();
        auto acc = accFor.getRegionIterArgs()[0];

        auto cKH = makeIndexConst(b, loc, KH);
        auto khFor = b.create<mlir::scf::ForOp>(loc, c0, cKH, c1, mlir::ValueRange{acc});
        b.setInsertionPointToStart(khFor.getBody());
        auto kh = khFor.getInductionVar();
        auto acc2 = khFor.getRegionIterArgs()[0];

        auto cKW = makeIndexConst(b, loc, KW);
        auto kwFor = b.create<mlir::scf::ForOp>(loc, c0, cKW, c1, mlir::ValueRange{acc2});
        b.setInsertionPointToStart(kwFor.getBody());
        auto kw = kwFor.getInductionVar();
        auto acc3 = kwFor.getRegionIterArgs()[0];

        int64_t mult = (groups > 0) ? (C_OUT / groups) : C_OUT;
        mult = std::max<int64_t>(1, mult);
        auto cMult = makeIndexConst(b, loc, mult);
        auto g = b.create<mlir::arith::DivUIOp>(loc, oc, cMult).getResult();
        auto baseC = b.create<mlir::arith::MulIOp>(loc, g, makeIndexConst(b, loc, C_PER_G)).getResult();
        auto ic = b.create<mlir::arith::AddIOp>(loc, baseC, icg).getResult();

        auto ih0 = b.create<mlir::arith::MulIOp>(loc, oh, makeIndexConst(b, loc, SH)).getResult();
        auto ih1 =
            b.create<mlir::arith::AddIOp>(
                 loc, ih0,
                 b.create<mlir::arith::MulIOp>(loc, kh, makeIndexConst(b, loc, DH)).getResult())
                .getResult();
        auto ih = b.create<mlir::arith::SubIOp>(loc, ih1, makeIndexConst(b, loc, PH)).getResult();

        auto iw0 = b.create<mlir::arith::MulIOp>(loc, ow, makeIndexConst(b, loc, SW)).getResult();
        auto iw1 =
            b.create<mlir::arith::AddIOp>(
                 loc, iw0,
                 b.create<mlir::arith::MulIOp>(loc, kw, makeIndexConst(b, loc, DW)).getResult())
                .getResult();
        auto iw = b.create<mlir::arith::SubIOp>(loc, iw1, makeIndexConst(b, loc, PW)).getResult();

        auto geH = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, c0).getResult();
        auto ltH = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, makeIndexConst(b, loc, H))
                       .getResult();
        auto geW = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, c0).getResult();
        auto ltW = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, makeIndexConst(b, loc, W))
                       .getResult();
        auto okH = b.create<mlir::arith::AndIOp>(loc, geH, ltH).getResult();
        auto okW = b.create<mlir::arith::AndIOp>(loc, geW, ltW).getResult();
        auto ok = b.create<mlir::arith::AndIOp>(loc, okH, okW).getResult();

        auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type()}, ok, /*withElse=*/true);
        b.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto xOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, ic, ih, iw});
        auto wOr = emitScalar(wName, llvm::ArrayRef<mlir::Value>{oc, icg, kh, kw});
        auto xv = castScalar(b, loc, mlir::failed(xOr) ? makeF32Const(b, loc, 0.0f) : *xOr, b.getF32Type());
        auto wv = castScalar(b, loc, mlir::failed(wOr) ? makeF32Const(b, loc, 0.0f) : *wOr, b.getF32Type());
        auto prod = b.create<mlir::arith::MulFOp>(loc, xv, wv).getResult();
        auto sum = b.create<mlir::arith::AddFOp>(loc, acc3, prod).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum});
        b.setInsertionPointToStart(&ifOp.getElseRegion().front());
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc3});
        b.setInsertionPointAfter(ifOp);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0)});
        b.setInsertionPointAfter(kwFor);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kwFor.getResult(0)});
        b.setInsertionPointAfter(khFor);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{khFor.getResult(0)});
        b.setInsertionPointAfter(accFor);
        return castScalar(b, loc, accFor.getResult(0), scalarTypeFor(op.output));
      }

      if (op.op == "conv3d") {
        // [N,C_IN,D,H,W] * [C_OUT,C_PER_G,KD,KH,KW] + bias[C_OUT] -> [N,C_OUT,OD,OH,OW]
        if (shIn.size() != 5 || shW.size() != 5 || shOut.size() != 5) {
          ctx.module.emitError("full196: conv3d rank mismatch");
          return mlir::failure();
        }

        auto strideOr = resolveIntListParam(op, "stride", 3, 1);
        auto paddingOr = resolveIntListParam(op, "padding", 3, 0);
        auto dilationOr = resolveIntListParam(op, "dilation", 3, 1);
        if (mlir::failed(strideOr) || mlir::failed(paddingOr) || mlir::failed(dilationOr))
          return mlir::failure();
        int64_t SD = (*strideOr)[0], SH = (*strideOr)[1], SW = (*strideOr)[2];
        int64_t PD = (*paddingOr)[0], PH = (*paddingOr)[1], PW = (*paddingOr)[2];
        int64_t DD = (*dilationOr)[0], DH = (*dilationOr)[1], DW = (*dilationOr)[2];
        int64_t groups = 1;
        if (const llvm::json::Value *vv = op.attrs.get("groups")) {
          auto gOr = resolveIntParam(*vv, "attrs.groups");
          if (mlir::failed(gOr))
            return mlir::failure();
          groups = *gOr;
        }

        int64_t N = shIn[0], C_IN = shIn[1], D = shIn[2], H = shIn[3], W = shIn[4];
        int64_t C_OUT = shW[0], C_PER_G = shW[1], KD = shW[2], KH = shW[3], KW = shW[4];
        (void)N;
        (void)C_IN;
        auto n = outIdx[0];
        auto oc = outIdx[1];
        auto od = outIdx[2];
        auto oh = outIdx[3];
        auto ow = outIdx[4];

        auto biasOr = emitScalar(bName, llvm::ArrayRef<mlir::Value>{oc});
        auto acc0 =
            castScalar(b, loc, mlir::failed(biasOr) ? makeF32Const(b, loc, 0.0f) : *biasOr, b.getF32Type());

        auto c0 = makeIndexConst(b, loc, 0);
        auto c1 = makeIndexConst(b, loc, 1);

        auto cCPG = makeIndexConst(b, loc, C_PER_G);
        auto icFor = b.create<mlir::scf::ForOp>(loc, c0, cCPG, c1, mlir::ValueRange{acc0});
        b.setInsertionPointToStart(icFor.getBody());
        auto icg = icFor.getInductionVar();
        auto acc = icFor.getRegionIterArgs()[0];

        auto cKD = makeIndexConst(b, loc, KD);
        auto kdFor = b.create<mlir::scf::ForOp>(loc, c0, cKD, c1, mlir::ValueRange{acc});
        b.setInsertionPointToStart(kdFor.getBody());
        auto kd = kdFor.getInductionVar();
        auto acc2 = kdFor.getRegionIterArgs()[0];

        auto cKH = makeIndexConst(b, loc, KH);
        auto khFor = b.create<mlir::scf::ForOp>(loc, c0, cKH, c1, mlir::ValueRange{acc2});
        b.setInsertionPointToStart(khFor.getBody());
        auto kh = khFor.getInductionVar();
        auto acc3 = khFor.getRegionIterArgs()[0];

        auto cKW = makeIndexConst(b, loc, KW);
        auto kwFor = b.create<mlir::scf::ForOp>(loc, c0, cKW, c1, mlir::ValueRange{acc3});
        b.setInsertionPointToStart(kwFor.getBody());
        auto kw = kwFor.getInductionVar();
        auto acc4 = kwFor.getRegionIterArgs()[0];

        int64_t mult = (groups > 0) ? (C_OUT / groups) : C_OUT;
        mult = std::max<int64_t>(1, mult);
        auto cMult = makeIndexConst(b, loc, mult);
        auto g = b.create<mlir::arith::DivUIOp>(loc, oc, cMult).getResult();
        auto baseC = b.create<mlir::arith::MulIOp>(loc, g, makeIndexConst(b, loc, C_PER_G)).getResult();
        auto ic = b.create<mlir::arith::AddIOp>(loc, baseC, icg).getResult();

        auto id0 = b.create<mlir::arith::MulIOp>(loc, od, makeIndexConst(b, loc, SD)).getResult();
        auto id1 =
            b.create<mlir::arith::AddIOp>(
                 loc, id0,
                 b.create<mlir::arith::MulIOp>(loc, kd, makeIndexConst(b, loc, DD)).getResult())
                .getResult();
        auto id = b.create<mlir::arith::SubIOp>(loc, id1, makeIndexConst(b, loc, PD)).getResult();

        auto ih0 = b.create<mlir::arith::MulIOp>(loc, oh, makeIndexConst(b, loc, SH)).getResult();
        auto ih1 =
            b.create<mlir::arith::AddIOp>(
                 loc, ih0,
                 b.create<mlir::arith::MulIOp>(loc, kh, makeIndexConst(b, loc, DH)).getResult())
                .getResult();
        auto ih = b.create<mlir::arith::SubIOp>(loc, ih1, makeIndexConst(b, loc, PH)).getResult();

        auto iw0 = b.create<mlir::arith::MulIOp>(loc, ow, makeIndexConst(b, loc, SW)).getResult();
        auto iw1 =
            b.create<mlir::arith::AddIOp>(
                 loc, iw0,
                 b.create<mlir::arith::MulIOp>(loc, kw, makeIndexConst(b, loc, DW)).getResult())
                .getResult();
        auto iw = b.create<mlir::arith::SubIOp>(loc, iw1, makeIndexConst(b, loc, PW)).getResult();

        auto geD = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, id, c0).getResult();
        auto ltD =
            b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, id, makeIndexConst(b, loc, D))
                .getResult();
        auto geH = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, c0).getResult();
        auto ltH =
            b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, makeIndexConst(b, loc, H))
                .getResult();
        auto geW = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, c0).getResult();
        auto ltW =
            b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, makeIndexConst(b, loc, W))
                .getResult();
        auto okD = b.create<mlir::arith::AndIOp>(loc, geD, ltD).getResult();
        auto okH = b.create<mlir::arith::AndIOp>(loc, geH, ltH).getResult();
        auto okW = b.create<mlir::arith::AndIOp>(loc, geW, ltW).getResult();
        auto okDH = b.create<mlir::arith::AndIOp>(loc, okD, okH).getResult();
        auto ok = b.create<mlir::arith::AndIOp>(loc, okDH, okW).getResult();

        auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type()}, ok, /*withElse=*/true);
        b.setInsertionPointToStart(&ifOp.getThenRegion().front());
        auto xOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, ic, id, ih, iw});
        auto wOr = emitScalar(wName, llvm::ArrayRef<mlir::Value>{oc, icg, kd, kh, kw});
        auto xv = castScalar(b, loc, mlir::failed(xOr) ? makeF32Const(b, loc, 0.0f) : *xOr, b.getF32Type());
        auto wv = castScalar(b, loc, mlir::failed(wOr) ? makeF32Const(b, loc, 0.0f) : *wOr, b.getF32Type());
        auto prod = b.create<mlir::arith::MulFOp>(loc, xv, wv).getResult();
        auto sum = b.create<mlir::arith::AddFOp>(loc, acc4, prod).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum});
        b.setInsertionPointToStart(&ifOp.getElseRegion().front());
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc4});
        b.setInsertionPointAfter(ifOp);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0)});

        b.setInsertionPointAfter(kwFor);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kwFor.getResult(0)});
        b.setInsertionPointAfter(khFor);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{khFor.getResult(0)});
        b.setInsertionPointAfter(kdFor);
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kdFor.getResult(0)});
        b.setInsertionPointAfter(icFor);
        return castScalar(b, loc, icFor.getResult(0), scalarTypeFor(op.output));
      }

      ctx.module.emitError() << "full196: conv op not yet implemented in cpp_plugin: " << op.op;
      return mlir::failure();
    }

    if (op.op == "avg_pool2d") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: avg_pool2d expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 4 || outShape.size() != 4) {
        ctx.module.emitError("full196: avg_pool2d expects rank4 NCHW");
        return mlir::failure();
      }
      auto kernelOr = resolveIntListParam(op, "kernel_size", 2, 1);
      auto strideOr = resolveIntListParam(op, "stride", 2, 1);
      auto paddingOr = resolveIntListParam(op, "padding", 2, 0);
      if (mlir::failed(kernelOr) || mlir::failed(strideOr) || mlir::failed(paddingOr))
        return mlir::failure();
      int64_t KH = (*kernelOr)[0], KW = (*kernelOr)[1];
      int64_t SH = (*strideOr)[0], SW = (*strideOr)[1];
      int64_t PH = (*paddingOr)[0], PW = (*paddingOr)[1];
      const bool countIncludePad = op.attrs.getBoolean("count_include_pad").value_or(true);

      int64_t H = shIn[2];
      int64_t W = shIn[3];
      auto n = outIdx[0];
      auto c = outIdx[1];
      auto oh = outIdx[2];
      auto ow = outIdx[3];

      auto ihStart = b.create<mlir::arith::SubIOp>(loc,
                                                   b.create<mlir::arith::MulIOp>(loc, oh, makeIndexConst(b, loc, SH))
                                                       .getResult(),
                                                   makeIndexConst(b, loc, PH))
                         .getResult();
      auto iwStart = b.create<mlir::arith::SubIOp>(loc,
                                                   b.create<mlir::arith::MulIOp>(loc, ow, makeIndexConst(b, loc, SW))
                                                       .getResult(),
                                                   makeIndexConst(b, loc, PW))
                         .getResult();

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      int64_t K = KH * KW;
      auto cK = makeIndexConst(b, loc, K);
      auto sum0 = makeF32Const(b, loc, 0.0f);
      auto cnt0 = countIncludePad ? makeIndexConst(b, loc, K) : c0;
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{sum0, cnt0});
      b.setInsertionPointToStart(forOp.getBody());
      auto t = forOp.getInductionVar();
      auto sum = forOp.getRegionIterArgs()[0];
      auto cnt = forOp.getRegionIterArgs()[1];

      auto kw = b.create<mlir::arith::RemUIOp>(loc, t, makeIndexConst(b, loc, KW)).getResult();
      auto kh = b.create<mlir::arith::DivUIOp>(loc, t, makeIndexConst(b, loc, KW)).getResult();
      auto ih = b.create<mlir::arith::AddIOp>(loc, ihStart, kh).getResult();
      auto iw = b.create<mlir::arith::AddIOp>(loc, iwStart, kw).getResult();

      auto geH = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, c0).getResult();
      auto ltH = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, makeIndexConst(b, loc, H))
                     .getResult();
      auto geW = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, c0).getResult();
      auto ltW = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, makeIndexConst(b, loc, W))
                     .getResult();
      auto ok = b.create<mlir::arith::AndIOp>(loc, b.create<mlir::arith::AndIOp>(loc, geH, ltH).getResult(),
                                              b.create<mlir::arith::AndIOp>(loc, geW, ltW).getResult())
                    .getResult();

      auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type(), b.getIndexType()}, ok,
                                            /*withElse=*/true);
      b.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto vOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, c, ih, iw});
      auto vv = castScalar(b, loc, mlir::failed(vOr) ? sum0 : *vOr, b.getF32Type());
      auto sum2 = b.create<mlir::arith::AddFOp>(loc, sum, vv).getResult();
      auto cnt2 = countIncludePad ? cnt : b.create<mlir::arith::AddIOp>(loc, cnt, c1).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum2, cnt2});
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{sum, cnt});
      b.setInsertionPointAfter(ifOp);
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0), ifOp.getResult(1)});
      b.setInsertionPointAfter(forOp);

      auto sumOut = forOp.getResult(0);
      auto cntOut = forOp.getResult(1);
      mlir::Value denomF;
      if (countIncludePad) {
        denomF = makeF32Const(b, loc, static_cast<float>(K));
      } else {
        denomF = b.create<mlir::arith::SIToFPOp>(loc, b.getF32Type(), castIndexToInt(b, loc, cntOut, 32)).getResult();
      }
      auto nonzero = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, cntOut, c0).getResult();
      auto div = b.create<mlir::arith::DivFOp>(loc, sumOut, denomF).getResult();
      auto outV = b.create<mlir::arith::SelectOp>(loc, nonzero, div, makeF32Const(b, loc, 0.0f)).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "max_pool2d_with_indices") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: max_pool2d_with_indices expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 4 || outShape.size() != 4) {
        ctx.module.emitError("full196: max_pool2d_with_indices expects rank4 NCHW");
        return mlir::failure();
      }
      auto kernelOr = resolveIntListParam(op, "kernel_size", 2, 1);
      auto strideOr = resolveIntListParam(op, "stride", 2, 1);
      auto paddingOr = resolveIntListParam(op, "padding", 2, 0);
      auto dilationOr = resolveIntListParam(op, "dilation", 2, 1);
      if (mlir::failed(kernelOr) || mlir::failed(strideOr) || mlir::failed(paddingOr) || mlir::failed(dilationOr))
        return mlir::failure();
      int64_t KH = (*kernelOr)[0], KW = (*kernelOr)[1];
      int64_t SH = (*strideOr)[0], SW = (*strideOr)[1];
      int64_t PH = (*paddingOr)[0], PW = (*paddingOr)[1];
      int64_t DH = (*dilationOr)[0], DW = (*dilationOr)[1];

      int64_t H = shIn[2];
      int64_t W = shIn[3];
      auto n = outIdx[0];
      auto c = outIdx[1];
      auto oh = outIdx[2];
      auto ow = outIdx[3];

      llvm::StringRef select = "values";
      if (auto s = op.attrs.getString("select")) {
        select = s->trim();
      }

      auto ihStart = b.create<mlir::arith::SubIOp>(loc,
                                                   b.create<mlir::arith::MulIOp>(loc, oh, makeIndexConst(b, loc, SH))
                                                       .getResult(),
                                                   makeIndexConst(b, loc, PH))
                         .getResult();
      auto iwStart = b.create<mlir::arith::SubIOp>(loc,
                                                   b.create<mlir::arith::MulIOp>(loc, ow, makeIndexConst(b, loc, SW))
                                                       .getResult(),
                                                   makeIndexConst(b, loc, PW))
                         .getResult();

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      int64_t K = KH * KW;
      auto cK = makeIndexConst(b, loc, K);
      auto best0 = makeF32Const(b, loc, -3.402823466e+38f);
      auto bestIdx0 = c0;
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{best0, bestIdx0});
      b.setInsertionPointToStart(forOp.getBody());
      auto t = forOp.getInductionVar();
      auto bestV = forOp.getRegionIterArgs()[0];
      auto bestI = forOp.getRegionIterArgs()[1];

      auto kw = b.create<mlir::arith::RemUIOp>(loc, t, makeIndexConst(b, loc, KW)).getResult();
      auto kh = b.create<mlir::arith::DivUIOp>(loc, t, makeIndexConst(b, loc, KW)).getResult();
      auto ih = b.create<mlir::arith::AddIOp>(
                    loc, ihStart,
                    b.create<mlir::arith::MulIOp>(loc, kh, makeIndexConst(b, loc, DH)).getResult())
                    .getResult();
      auto iw = b.create<mlir::arith::AddIOp>(
                    loc, iwStart,
                    b.create<mlir::arith::MulIOp>(loc, kw, makeIndexConst(b, loc, DW)).getResult())
                    .getResult();

      auto geH = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, ih, c0).getResult();
      auto ltH = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, ih, makeIndexConst(b, loc, H))
                     .getResult();
      auto geW = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, iw, c0).getResult();
      auto ltW = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, iw, makeIndexConst(b, loc, W))
                     .getResult();
      auto ok = b.create<mlir::arith::AndIOp>(loc, b.create<mlir::arith::AndIOp>(loc, geH, ltH).getResult(),
                                              b.create<mlir::arith::AndIOp>(loc, geW, ltW).getResult())
                    .getResult();

      auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type(), b.getIndexType()}, ok,
                                            /*withElse=*/true);
      b.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto vOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, c, ih, iw});
      auto vv = castScalar(b, loc, mlir::failed(vOr) ? best0 : *vOr, b.getF32Type());
      auto candI =
          b.create<mlir::arith::AddIOp>(loc, b.create<mlir::arith::MulIOp>(loc, ih, makeIndexConst(b, loc, W)).getResult(),
                                        iw)
              .getResult();
      auto gt = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, vv, bestV).getResult();
      auto eq = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, vv, bestV).getResult();
      auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, candI, bestI).getResult();
      auto better = b.create<mlir::arith::OrIOp>(loc, gt, b.create<mlir::arith::AndIOp>(loc, eq, lt).getResult())
                        .getResult();
      auto bestV2 = b.create<mlir::arith::SelectOp>(loc, better, vv, bestV).getResult();
      auto bestI2 = b.create<mlir::arith::SelectOp>(loc, better, candI, bestI).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{bestV2, bestI2});
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{bestV, bestI});
      b.setInsertionPointAfter(ifOp);
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0), ifOp.getResult(1)});
      b.setInsertionPointAfter(forOp);

      auto outVal = forOp.getResult(0);
      auto outIdxV = forOp.getResult(1);
      if (select == "indices") {
        auto i64 = castIndexToInt(b, loc, outIdxV, 64);
        return castScalar(b, loc, i64, scalarTypeFor(op.output));
      }
      return castScalar(b, loc, outVal, scalarTypeFor(op.output));
    }

    if (op.op == "upsample_nearest1d") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: upsample_nearest1d expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 3 || outShape.size() != 3) {
        ctx.module.emitError("full196: upsample_nearest1d expects rank3 NCL");
        return mlir::failure();
      }
      int64_t IL = shIn[2];
      int64_t OL = outShape[2];
      if (IL <= 0 || OL <= 0) {
        ctx.module.emitError("full196: upsample_nearest1d invalid shapes");
        return mlir::failure();
      }
      auto n = outIdx[0];
      auto c = outIdx[1];
      auto ol = outIdx[2];
      auto il =
          b.create<mlir::arith::DivUIOp>(loc,
                                         b.create<mlir::arith::MulIOp>(loc, ol, makeIndexConst(b, loc, IL)).getResult(),
                                         makeIndexConst(b, loc, OL))
              .getResult();
      return emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, c, il});
    }

    if (op.op == "upsample_nearest2d") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: upsample_nearest2d expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 4 || outShape.size() != 4) {
        ctx.module.emitError("full196: upsample_nearest2d expects rank4 NCHW");
        return mlir::failure();
      }
      int64_t IH = shIn[2];
      int64_t IW = shIn[3];
      int64_t OH = outShape[2];
      int64_t OW = outShape[3];
      if (IH <= 0 || IW <= 0 || OH <= 0 || OW <= 0) {
        ctx.module.emitError("full196: upsample_nearest2d invalid shapes");
        return mlir::failure();
      }
      auto n = outIdx[0];
      auto c = outIdx[1];
      auto oh = outIdx[2];
      auto ow = outIdx[3];
      auto ih =
          b.create<mlir::arith::DivUIOp>(loc,
                                         b.create<mlir::arith::MulIOp>(loc, oh, makeIndexConst(b, loc, IH)).getResult(),
                                         makeIndexConst(b, loc, OH))
              .getResult();
      auto iw =
          b.create<mlir::arith::DivUIOp>(loc,
                                         b.create<mlir::arith::MulIOp>(loc, ow, makeIndexConst(b, loc, IW)).getResult(),
                                         makeIndexConst(b, loc, OW))
              .getResult();
      return emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, c, ih, iw});
    }

    if (op.op == "upsample_bicubic2d_aa") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: upsample_bicubic2d_aa expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 4 || outShape.size() != 4) {
        ctx.module.emitError("full196: upsample_bicubic2d_aa expects rank4 NCHW");
        return mlir::failure();
      }
      int64_t IH = shIn[2];
      int64_t IW = shIn[3];
      int64_t OH = outShape[2];
      int64_t OW = outShape[3];
      if (IH <= 0 || IW <= 0 || OH <= 0 || OW <= 0) {
        ctx.module.emitError("full196: upsample_bicubic2d_aa invalid shapes");
        return mlir::failure();
      }

      auto loadOptionalScalarF32 = [&](llvm::StringRef name, float defaultVal) -> mlir::FailureOr<mlir::Value> {
        if (shapes.find(name.str()) == shapes.end()) {
          return makeF32Const(b, loc, defaultVal);
        }
        auto vOr = emitScalar(name, llvm::ArrayRef<mlir::Value>{});
        if (mlir::failed(vOr)) {
          return mlir::failure();
        }
        return castScalar(b, loc, *vOr, b.getF32Type());
      };

      auto recipHOr = loadOptionalScalarF32("reciprocal_scale_h", 1.0f);
      auto recipWOr = loadOptionalScalarF32("reciprocal_scale_w", 1.0f);
      if (mlir::failed(recipHOr) || mlir::failed(recipWOr))
        return mlir::failure();
      auto recipH = *recipHOr;
      auto recipW = *recipWOr;

      float supportV = static_cast<float>(op.attrs.getNumber("support").value_or(2.0));
      float invscaleV = static_cast<float>(op.attrs.getNumber("invscale").value_or(1.0));
      int64_t taps = std::max<int64_t>(1, static_cast<int64_t>(2.0f * supportV + 1.0f));

      auto computeAxis = [&](mlir::Value outPos, int64_t inSize, mlir::Value recipScale,
                             llvm::SmallVector<mlir::Value> &tapIdx,
                             llvm::SmallVector<mlir::Value> &tapW) -> mlir::LogicalResult {
        tapIdx.clear();
        tapW.clear();

        auto outI32 = castIndexToInt(b, loc, outPos, 32);
        auto outF = castScalar(b, loc, outI32, b.getF32Type());

        auto c05 = makeF32Const(b, loc, 0.5f);
        auto support = makeF32Const(b, loc, supportV);
        auto invscale = makeF32Const(b, loc, invscaleV);
        auto zeroF = makeF32Const(b, loc, 0.0f);
        auto oneF = makeF32Const(b, loc, 1.0f);
        auto twoF = makeF32Const(b, loc, 2.0f);

        // center = (out + 0.5) * reciprocal_scale
        auto center = b.create<mlir::arith::MulFOp>(loc,
                                                    b.create<mlir::arith::AddFOp>(loc, outF, c05).getResult(),
                                                    recipScale)
                          .getResult();

        // start = max(center - support + 0.5, 0)
        auto startRaw = b.create<mlir::arith::AddFOp>(
                             loc, b.create<mlir::arith::SubFOp>(loc, center, support).getResult(), c05)
                            .getResult();
        auto start = b.create<mlir::arith::MaximumFOp>(loc, startRaw, zeroF).getResult();

        // span_end = min(center + support + 0.5, in_size)
        auto spanEndRaw = b.create<mlir::arith::AddFOp>(
                               loc, b.create<mlir::arith::AddFOp>(loc, center, support).getResult(), c05)
                              .getResult();
        auto inSizeF = makeF32Const(b, loc, static_cast<float>(inSize));
        auto spanEnd = b.create<mlir::arith::MinimumFOp>(loc, spanEndRaw, inSizeF).getResult();
        auto spanSize = b.create<mlir::arith::SubFOp>(loc, spanEnd, start).getResult();
        auto startMinusCenter = b.create<mlir::arith::SubFOp>(loc, start, center).getResult();

        auto startFloor = b.create<mlir::math::FloorOp>(loc, start).getResult();
        auto startI32 = b.create<mlir::arith::FPToSIOp>(loc, b.getI32Type(), startFloor).getResult();
        auto startIdx = castIntToIndex(b, loc, startI32);
        auto hi = makeIndexConst(b, loc, std::max<int64_t>(0, inSize - 1));

        mlir::Value sumW = zeroF;
        for (int64_t k = 0; k < taps; ++k) {
          auto kF = makeF32Const(b, loc, static_cast<float>(k));

          // abs_arg = abs((k + (start - center) + 0.5) * invscale)
          auto dist = b.create<mlir::arith::MulFOp>(
                           loc,
                           b.create<mlir::arith::AddFOp>(
                               loc, b.create<mlir::arith::AddFOp>(loc, kF, startMinusCenter).getResult(), c05)
                               .getResult(),
                           invscale)
                          .getResult();
          auto t = b.create<mlir::math::AbsFOp>(loc, dist).getResult();
          auto t2 = b.create<mlir::arith::MulFOp>(loc, t, t).getResult();
          auto t3 = b.create<mlir::arith::MulFOp>(loc, t2, t).getResult();

          // Keys cubic kernel (a=-0.5):
          // t < 1: 1 - 2.5*t^2 + 1.5*t^3
          // t < 2: -2*t + 4*t^2 - 2*t^3
          auto w0 = b.create<mlir::arith::AddFOp>(
                         loc, oneF,
                         b.create<mlir::arith::AddFOp>(
                             loc, b.create<mlir::arith::MulFOp>(loc, makeF32Const(b, loc, -2.5f), t2).getResult(),
                             b.create<mlir::arith::MulFOp>(loc, makeF32Const(b, loc, 1.5f), t3).getResult())
                             .getResult())
                        .getResult();
          auto w1 = b.create<mlir::arith::AddFOp>(
                         loc, b.create<mlir::arith::MulFOp>(loc, makeF32Const(b, loc, -2.0f), t).getResult(),
                         b.create<mlir::arith::AddFOp>(
                             loc, b.create<mlir::arith::MulFOp>(loc, makeF32Const(b, loc, 4.0f), t2).getResult(),
                             b.create<mlir::arith::MulFOp>(loc, makeF32Const(b, loc, -2.0f), t3).getResult())
                             .getResult())
                        .getResult();
          auto lt1 = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, t, oneF).getResult();
          auto lt2 = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, t, twoF).getResult();
          auto w12 = b.create<mlir::arith::SelectOp>(loc, lt2, w1, zeroF).getResult();
          auto w = b.create<mlir::arith::SelectOp>(loc, lt1, w0, w12).getResult();

          // mask: k < span_size
          auto mask = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, kF, spanSize).getResult();
          auto wMasked = b.create<mlir::arith::SelectOp>(loc, mask, w, zeroF).getResult();
          sumW = b.create<mlir::arith::AddFOp>(loc, sumW, wMasked).getResult();

          auto kIdx = makeIndexConst(b, loc, k);
          auto idx = b.create<mlir::arith::AddIOp>(loc, startIdx, kIdx).getResult();
          auto idxC = b.create<mlir::arith::MinUIOp>(loc, idx, hi).getResult();
          tapIdx.push_back(idxC);
          tapW.push_back(wMasked);
        }

        // normalize_weights: w /= sum(w)
        auto sumNz = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, sumW, zeroF).getResult();
        auto invSum = b.create<mlir::arith::SelectOp>(
                           loc, sumNz, b.create<mlir::arith::DivFOp>(loc, oneF, sumW).getResult(), zeroF)
                          .getResult();
        for (auto &w : tapW) {
          w = b.create<mlir::arith::MulFOp>(loc, w, invSum).getResult();
        }
        return mlir::success();
      };

      llvm::SmallVector<mlir::Value> xIdx, xW, yIdx, yW;
      auto n = outIdx[0];
      auto c = outIdx[1];
      auto oh = outIdx[2];
      auto ow = outIdx[3];
      if (mlir::failed(computeAxis(ow, IW, recipW, xIdx, xW)) || mlir::failed(computeAxis(oh, IH, recipH, yIdx, yW)))
        return mlir::failure();

      mlir::Value acc = makeF32Const(b, loc, 0.0f);
      for (int64_t dy = 0; dy < taps; ++dy) {
        for (int64_t dx = 0; dx < taps; ++dx) {
          auto vOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{n, c, yIdx[static_cast<size_t>(dy)],
                                                                   xIdx[static_cast<size_t>(dx)]});
          if (mlir::failed(vOr))
            return mlir::failure();
          auto vv = castScalar(b, loc, *vOr, b.getF32Type());
          auto wxy = b.create<mlir::arith::MulFOp>(loc, yW[static_cast<size_t>(dy)], xW[static_cast<size_t>(dx)])
                         .getResult();
          acc = b.create<mlir::arith::AddFOp>(loc, acc, b.create<mlir::arith::MulFOp>(loc, vv, wxy).getResult())
                    .getResult();
        }
      }
      return castScalar(b, loc, acc, scalarTypeFor(op.output));
    }

    if (op.op == "glu") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: glu expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != outShape.size() || shIn.size() != 2) {
        ctx.module.emitError("full196: glu expects rank2 input/output");
        return mlir::failure();
      }
      int64_t axis = 0;
      if (auto ii = op.attrs.getInteger("axis"))
        axis = static_cast<int64_t>(*ii);
      if (axis != 1) {
        ctx.module.emitError("full196: glu supports axis=1 only");
        return mlir::failure();
      }
      int64_t N = shIn[1];
      int64_t half = N / 2;
      auto m = outIdx[0];
      auto n = outIdx[1];
      auto gateIdx = b.create<mlir::arith::AddIOp>(loc, n, makeIndexConst(b, loc, half)).getResult();
      auto aOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{m, n});
      auto bOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{m, gateIdx});
      if (mlir::failed(aOr) || mlir::failed(bOr))
        return mlir::failure();
      auto a = castScalar(b, loc, *aOr, b.getF32Type());
      auto bb = castScalar(b, loc, *bOr, b.getF32Type());
      auto neg = b.create<mlir::arith::NegFOp>(loc, bb).getResult();
      auto expv = b.create<mlir::math::ExpOp>(loc, neg).getResult();
      auto one = makeF32Const(b, loc, 1.0f);
      auto denom = b.create<mlir::arith::AddFOp>(loc, one, expv).getResult();
      auto sig = b.create<mlir::arith::DivFOp>(loc, one, denom).getResult();
      auto outV = b.create<mlir::arith::MulFOp>(loc, a, sig).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "select_scatter") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: select_scatter expects inputs (inp, src)");
        return mlir::failure();
      }
      auto inp = op.inputs[0];
      auto src = op.inputs[1];
      auto shIn = shapeOf(inp);
      if (shIn.size() != 2 || outShape.size() != 2) {
        ctx.module.emitError("full196: select_scatter supports rank2 only");
        return mlir::failure();
      }
      int64_t dim = static_cast<int64_t>(op.attrs.getInteger("dim").value_or(0));
      int64_t index = static_cast<int64_t>(op.attrs.getInteger("index").value_or(0));
      auto m = outIdx[0];
      auto n = outIdx[1];
      if (dim == 0) {
        auto isRow = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, m,
                                                   makeIndexConst(b, loc, index))
                         .getResult();
        auto srcOr = emitScalar(src, llvm::ArrayRef<mlir::Value>{n});
        auto inpOr = emitScalar(inp, outIdx);
        if (mlir::failed(srcOr) || mlir::failed(inpOr))
          return mlir::failure();
        auto sv = *srcOr;
        auto iv = *inpOr;
        auto sel = b.create<mlir::arith::SelectOp>(loc, isRow, sv, iv).getResult();
        return castScalar(b, loc, sel, scalarTypeFor(op.output));
      }
      if (dim == 1) {
        auto isCol = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, n,
                                                   makeIndexConst(b, loc, index))
                         .getResult();
        auto srcOr = emitScalar(src, llvm::ArrayRef<mlir::Value>{m});
        auto inpOr = emitScalar(inp, outIdx);
        if (mlir::failed(srcOr) || mlir::failed(inpOr))
          return mlir::failure();
        auto sv = *srcOr;
        auto iv = *inpOr;
        auto sel = b.create<mlir::arith::SelectOp>(loc, isCol, sv, iv).getResult();
        return castScalar(b, loc, sel, scalarTypeFor(op.output));
      }
      ctx.module.emitError("full196: select_scatter supports dim in {0,1}");
      return mlir::failure();
    }

    if (op.op == "slice_scatter") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: slice_scatter expects inputs (inp, src)");
        return mlir::failure();
      }
      auto inp = op.inputs[0];
      auto src = op.inputs[1];
      auto shIn = shapeOf(inp);
      if (shIn.size() != 2 || outShape.size() != 2) {
        ctx.module.emitError("full196: slice_scatter supports rank2 only");
        return mlir::failure();
      }
      int64_t dim = static_cast<int64_t>(op.attrs.getInteger("dim").value_or(0));
      int64_t start = static_cast<int64_t>(op.attrs.getInteger("start").value_or(0));
      int64_t end = static_cast<int64_t>(op.attrs.getInteger("end").value_or(0));
      int64_t step = static_cast<int64_t>(op.attrs.getInteger("step").value_or(1));
      if (step <= 0) {
        ctx.module.emitError("full196: slice_scatter requires step>0");
        return mlir::failure();
      }
      auto m = outIdx[0];
      auto n = outIdx[1];

      auto chooseSrc = [&](mlir::Value srcRow, mlir::Value srcCol, mlir::Value inV,
                           mlir::Value cond) -> mlir::FailureOr<mlir::Value> {
        auto srcOr = emitScalar(src, llvm::ArrayRef<mlir::Value>{srcRow, srcCol});
        if (mlir::failed(srcOr))
          return mlir::failure();
        auto sel = b.create<mlir::arith::SelectOp>(loc, cond, *srcOr, inV).getResult();
        return sel;
      };

      auto inpOr = emitScalar(inp, outIdx);
      if (mlir::failed(inpOr))
        return mlir::failure();
      auto inV = *inpOr;

      if (dim == 1) {
        // Scatter a slice along columns: out[:, start:end:step] = src
        auto ge = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, n,
                                                makeIndexConst(b, loc, start))
                      .getResult();
        auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, n,
                                                makeIndexConst(b, loc, end))
                      .getResult();
        auto inRange = b.create<mlir::arith::AndIOp>(loc, ge, lt).getResult();
        auto off = b.create<mlir::arith::SubIOp>(loc, n, makeIndexConst(b, loc, start)).getResult();
        auto rem = b.create<mlir::arith::RemUIOp>(loc, off, makeIndexConst(b, loc, step)).getResult();
        auto isStep = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, rem, makeIndexConst(b, loc, 0))
                          .getResult();
        auto cond = b.create<mlir::arith::AndIOp>(loc, inRange, isStep).getResult();
        auto srcCol = b.create<mlir::arith::DivUIOp>(loc, off, makeIndexConst(b, loc, step)).getResult();
        auto vOr = chooseSrc(m, srcCol, inV, cond);
        if (mlir::failed(vOr))
          return mlir::failure();
        return castScalar(b, loc, *vOr, scalarTypeFor(op.output));
      }
      if (dim == 0) {
        // Scatter a slice along rows: out[start:end:step, :] = src
        auto ge = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, m,
                                                makeIndexConst(b, loc, start))
                      .getResult();
        auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, m,
                                                makeIndexConst(b, loc, end))
                      .getResult();
        auto inRange = b.create<mlir::arith::AndIOp>(loc, ge, lt).getResult();
        auto off = b.create<mlir::arith::SubIOp>(loc, m, makeIndexConst(b, loc, start)).getResult();
        auto rem = b.create<mlir::arith::RemUIOp>(loc, off, makeIndexConst(b, loc, step)).getResult();
        auto isStep = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, rem, makeIndexConst(b, loc, 0))
                          .getResult();
        auto cond = b.create<mlir::arith::AndIOp>(loc, inRange, isStep).getResult();
        auto srcRow = b.create<mlir::arith::DivUIOp>(loc, off, makeIndexConst(b, loc, step)).getResult();
        auto vOr = chooseSrc(srcRow, n, inV, cond);
        if (mlir::failed(vOr))
          return mlir::failure();
        return castScalar(b, loc, *vOr, scalarTypeFor(op.output));
      }
      ctx.module.emitError("full196: slice_scatter supports dim in {0,1}");
      return mlir::failure();
    }

    if (op.op == "scatter") {
      if (op.inputs.size() != 3) {
        ctx.module.emitError("full196: scatter expects inputs (inp, index, src)");
        return mlir::failure();
      }
      auto inp = op.inputs[0];
      auto index = op.inputs[1];
      auto src = op.inputs[2];
      if (outShape.size() != 2) {
        ctx.module.emitError("full196: scatter supports rank2 only");
        return mlir::failure();
      }
      int64_t dim = static_cast<int64_t>(op.attrs.getInteger("dim").value_or(0));
      int64_t M = outShape[0];
      int64_t N = outShape[1];
      auto m = outIdx[0];
      auto n = outIdx[1];

      auto baseOr = emitScalar(inp, outIdx);
      if (mlir::failed(baseOr))
        return mlir::failure();
      auto cur0 = castScalar(b, loc, *baseOr, scalarTypeFor(op.output));

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);

      if (dim == 1) {
        auto cN = makeIndexConst(b, loc, N);
        auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{cur0});
        b.setInsertionPointToStart(forOp.getBody());
        auto j = forOp.getInductionVar();
        auto cur = forOp.getRegionIterArgs()[0];
        auto idxOr = emitScalar(index, llvm::ArrayRef<mlir::Value>{m, j});
        auto srcOr = emitScalar(src, llvm::ArrayRef<mlir::Value>{m, j});
        if (mlir::failed(idxOr) || mlir::failed(srcOr))
          return mlir::failure();
        auto idxI32 = castScalar(b, loc, *idxOr, b.getI32Type());
        auto idx = castIntToIndex(b, loc, idxI32);
        auto match = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, idx, n).getResult();
        auto sv = castScalar(b, loc, *srcOr, cur.getType());
        auto cur2 = b.create<mlir::arith::SelectOp>(loc, match, sv, cur).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{cur2});
        b.setInsertionPointAfter(forOp);
        return castScalar(b, loc, forOp.getResult(0), scalarTypeFor(op.output));
      }
      if (dim == 0) {
        auto cM = makeIndexConst(b, loc, M);
        auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cM, c1, mlir::ValueRange{cur0});
        b.setInsertionPointToStart(forOp.getBody());
        auto i = forOp.getInductionVar();
        auto cur = forOp.getRegionIterArgs()[0];
        auto idxOr = emitScalar(index, llvm::ArrayRef<mlir::Value>{i, n});
        auto srcOr = emitScalar(src, llvm::ArrayRef<mlir::Value>{i, n});
        if (mlir::failed(idxOr) || mlir::failed(srcOr))
          return mlir::failure();
        auto idxI32 = castScalar(b, loc, *idxOr, b.getI32Type());
        auto idx = castIntToIndex(b, loc, idxI32);
        auto match = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, idx, m).getResult();
        auto sv = castScalar(b, loc, *srcOr, cur.getType());
        auto cur2 = b.create<mlir::arith::SelectOp>(loc, match, sv, cur).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{cur2});
        b.setInsertionPointAfter(forOp);
        return castScalar(b, loc, forOp.getResult(0), scalarTypeFor(op.output));
      }
      ctx.module.emitError("full196: scatter supports dim in {0,1}");
      return mlir::failure();
    }

    if (op.op == "index_add") {
      if (op.inputs.size() != 3) {
        ctx.module.emitError("full196: index_add expects inputs (base, index, src)");
        return mlir::failure();
      }
      auto baseName = op.inputs[0];
      auto indexName = op.inputs[1];
      auto srcName = op.inputs[2];
      auto shBase = shapeOf(baseName);
      auto shIdx = shapeOf(indexName);
      auto shSrc = shapeOf(srcName);
      if (shBase.size() != 2 || outShape.size() != 2 || shIdx.size() != 1 || shSrc.size() != 2) {
        ctx.module.emitError("full196: index_add supports base[M,N], index[L], src[*,*]");
        return mlir::failure();
      }
      int64_t axis = static_cast<int64_t>(op.attrs.getInteger("axis").value_or(0));
      float alpha = 1.0f;
      if (auto num = op.attrs.getNumber("alpha")) {
        alpha = static_cast<float>(*num);
      } else if (auto ii = op.attrs.getInteger("alpha")) {
        alpha = static_cast<float>(*ii);
      }
      int64_t L = shIdx[0];
      auto m = outIdx[0];
      auto n = outIdx[1];
      auto baseOr = emitScalar(baseName, outIdx);
      if (mlir::failed(baseOr))
        return mlir::failure();
      auto baseV = castScalar(b, loc, *baseOr, b.getF32Type());

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cL = makeIndexConst(b, loc, L);
      auto sum0 = makeF32Const(b, loc, 0.0f);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cL, c1, mlir::ValueRange{sum0});
      b.setInsertionPointToStart(forOp.getBody());
      auto i = forOp.getInductionVar();
      auto acc = forOp.getRegionIterArgs()[0];
      auto idxOr = emitScalar(indexName, llvm::ArrayRef<mlir::Value>{i});
      if (mlir::failed(idxOr))
        return mlir::failure();
      auto idxI = castIntToIndex(b, loc, castScalar(b, loc, *idxOr, b.getI32Type()));
      mlir::Value match;
      mlir::Value srcV;
      if (axis == 0) {
        match = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, idxI, m).getResult();
        auto vOr = emitScalar(srcName, llvm::ArrayRef<mlir::Value>{i, n});
        if (mlir::failed(vOr))
          return mlir::failure();
        srcV = castScalar(b, loc, *vOr, b.getF32Type());
      } else if (axis == 1) {
        match = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, idxI, n).getResult();
        auto vOr = emitScalar(srcName, llvm::ArrayRef<mlir::Value>{m, i});
        if (mlir::failed(vOr))
          return mlir::failure();
        srcV = castScalar(b, loc, *vOr, b.getF32Type());
      } else {
        ctx.module.emitError("full196: index_add supports axis in {0,1}");
        return mlir::failure();
      }
      auto add = b.create<mlir::arith::AddFOp>(loc, acc, srcV).getResult();
      auto acc2 = b.create<mlir::arith::SelectOp>(loc, match, add, acc).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
      b.setInsertionPointAfter(forOp);
      auto sum = forOp.getResult(0);
      auto scaled = b.create<mlir::arith::MulFOp>(loc, sum, makeF32Const(b, loc, alpha)).getResult();
      auto outV = b.create<mlir::arith::AddFOp>(loc, baseV, scaled).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "index_put") {
      if (op.inputs.size() != 4) {
        ctx.module.emitError("full196: index_put expects inputs (base, row_idx, col_idx, values)");
        return mlir::failure();
      }
      auto baseName = op.inputs[0];
      auto rowName = op.inputs[1];
      auto colName = op.inputs[2];
      auto valName = op.inputs[3];
      auto shBase = shapeOf(baseName);
      auto shRow = shapeOf(rowName);
      auto shCol = shapeOf(colName);
      auto shVal = shapeOf(valName);
      if (shBase.size() != 2 || outShape.size() != 2 || shRow.size() != 1 || shCol.size() != 1 || shVal.size() != 1 ||
          shRow[0] != shCol[0] || shRow[0] != shVal[0]) {
        ctx.module.emitError("full196: index_put expects base[M,N], row_idx[L], col_idx[L], values[L]");
        return mlir::failure();
      }
      bool accumulate = op.attrs.getBoolean("accumulate").value_or(false);
      int64_t L = shRow[0];
      auto m = outIdx[0];
      auto n = outIdx[1];
      auto baseOr = emitScalar(baseName, outIdx);
      if (mlir::failed(baseOr))
        return mlir::failure();
      auto cur0 = castScalar(b, loc, *baseOr, b.getF32Type());

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cL = makeIndexConst(b, loc, L);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cL, c1, mlir::ValueRange{cur0});
      b.setInsertionPointToStart(forOp.getBody());
      auto i = forOp.getInductionVar();
      auto cur = forOp.getRegionIterArgs()[0];
      auto rOr = emitScalar(rowName, llvm::ArrayRef<mlir::Value>{i});
      auto cOr = emitScalar(colName, llvm::ArrayRef<mlir::Value>{i});
      auto vOr = emitScalar(valName, llvm::ArrayRef<mlir::Value>{i});
      if (mlir::failed(rOr) || mlir::failed(cOr) || mlir::failed(vOr))
        return mlir::failure();
      auto rI = castIntToIndex(b, loc, castScalar(b, loc, *rOr, b.getI32Type()));
      auto cI = castIntToIndex(b, loc, castScalar(b, loc, *cOr, b.getI32Type()));
      auto rm = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, rI, m).getResult();
      auto cn = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, cI, n).getResult();
      auto match = b.create<mlir::arith::AndIOp>(loc, rm, cn).getResult();
      auto vv = castScalar(b, loc, *vOr, b.getF32Type());
      mlir::Value next;
      if (accumulate) {
        auto add = b.create<mlir::arith::AddFOp>(loc, cur, vv).getResult();
        next = b.create<mlir::arith::SelectOp>(loc, match, add, cur).getResult();
      } else {
        next = b.create<mlir::arith::SelectOp>(loc, match, vv, cur).getResult();
      }
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{next});
      b.setInsertionPointAfter(forOp);
      return castScalar(b, loc, forOp.getResult(0), scalarTypeFor(op.output));
    }

    if (op.op == "masked_select") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: masked_select expects inputs (inp, mask)");
        return mlir::failure();
      }
      auto inp = op.inputs[0];
      auto mask = op.inputs[1];
      auto shIn = shapeOf(inp);
      if (shIn.size() != 2 || outShape.size() != 1) {
        ctx.module.emitError("full196: masked_select expects inp[M,N] -> out[L]");
        return mlir::failure();
      }
      int64_t M = shIn[0];
      int64_t N = shIn[1];
      auto target = outIdx[0];
      auto total = makeIndexConst(b, loc, M * N);
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto found0 = makeF32Const(b, loc, 0.0f);
      auto cnt0 = c0;
      auto done0 = makeI1Const(b, loc, false);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, total, c1, mlir::ValueRange{found0, cnt0, done0});
      b.setInsertionPointToStart(forOp.getBody());
      auto lin = forOp.getInductionVar();
      auto found = forOp.getRegionIterArgs()[0];
      auto cnt = forOp.getRegionIterArgs()[1];
      auto done = forOp.getRegionIterArgs()[2];
      auto idx = delinearizeIndex(b, loc, lin, llvm::ArrayRef<int64_t>{M, N});
      auto mvOr = emitScalar(mask, idx);
      auto xvOr = emitScalar(inp, idx);
      if (mlir::failed(mvOr) || mlir::failed(xvOr))
        return mlir::failure();
      auto mv = castScalar(b, loc, *mvOr, b.getI1Type());
      auto nz = b.create<mlir::arith::AndIOp>(loc, mv, b.create<mlir::arith::XOrIOp>(loc, done, makeI1Const(b, loc, true)).getResult())
                    .getResult();
      auto isTarget = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, cnt, target).getResult();
      auto take = b.create<mlir::arith::AndIOp>(loc, mv, isTarget).getResult();
      auto xv = castScalar(b, loc, *xvOr, b.getF32Type());
      auto found2 = b.create<mlir::arith::SelectOp>(loc, take, xv, found).getResult();
      auto done2 = b.create<mlir::arith::OrIOp>(loc, done, take).getResult();
      auto cnt2 = b.create<mlir::arith::SelectOp>(loc, mv, b.create<mlir::arith::AddIOp>(loc, cnt, c1).getResult(), cnt)
                      .getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{found2, cnt2, done2});
      b.setInsertionPointAfter(forOp);
      return castScalar(b, loc, forOp.getResult(0), scalarTypeFor(op.output));
    }

    if (op.op == "masked_scatter") {
      if (op.inputs.size() != 3) {
        ctx.module.emitError("full196: masked_scatter expects inputs (inp, mask, source)");
        return mlir::failure();
      }
      auto inp = op.inputs[0];
      auto mask = op.inputs[1];
      auto source = op.inputs[2];
      auto shIn = shapeOf(inp);
      if (shIn.size() != 2 || outShape.size() != 2) {
        ctx.module.emitError("full196: masked_scatter expects inp[M,N] -> out[M,N]");
        return mlir::failure();
      }
      int64_t M = shIn[0];
      int64_t N = shIn[1];
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto linCur = linearizeIndices(b, loc, outIdx, outShape);
      auto cnt0 = c0;
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, linCur, c1, mlir::ValueRange{cnt0});
      b.setInsertionPointToStart(forOp.getBody());
      auto lin = forOp.getInductionVar();
      auto cnt = forOp.getRegionIterArgs()[0];
      auto idx = delinearizeIndex(b, loc, lin, llvm::ArrayRef<int64_t>{M, N});
      auto mvOr = emitScalar(mask, idx);
      if (mlir::failed(mvOr))
        return mlir::failure();
      auto mv = castScalar(b, loc, *mvOr, b.getI1Type());
      auto cnt2 = b.create<mlir::arith::SelectOp>(loc, mv, b.create<mlir::arith::AddIOp>(loc, cnt, c1).getResult(), cnt)
                      .getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{cnt2});
      b.setInsertionPointAfter(forOp);
      auto rank = forOp.getResult(0);

      auto mvCurOr = emitScalar(mask, outIdx);
      auto inOr = emitScalar(inp, outIdx);
      if (mlir::failed(mvCurOr) || mlir::failed(inOr))
        return mlir::failure();
      auto mvCur = castScalar(b, loc, *mvCurOr, b.getI1Type());
      auto srcOr = emitScalar(source, llvm::ArrayRef<mlir::Value>{rank});
      if (mlir::failed(srcOr))
        return mlir::failure();
      auto srcV = castScalar(b, loc, *srcOr, b.getF32Type());
      auto inV = castScalar(b, loc, *inOr, b.getF32Type());
      auto outV = b.create<mlir::arith::SelectOp>(loc, mvCur, srcV, inV).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "sort") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: sort expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 2 || outShape.size() != 2) {
        ctx.module.emitError("full196: sort supports rank2 only");
        return mlir::failure();
      }
      int64_t axis = static_cast<int64_t>(op.attrs.getInteger("axis").value_or(1));
      if (axis != 1) {
        ctx.module.emitError("full196: sort supports axis=1 only");
        return mlir::failure();
      }
      const bool descending = op.attrs.getBoolean("descending").value_or(false);
      const bool stable = op.attrs.getBoolean("stable").value_or(false);
      int64_t N = shIn[1];
      auto row = outIdx[0];
      auto pos = outIdx[1];

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cN = makeIndexConst(b, loc, N);
      auto found0 = makeF32Const(b, loc, 0.0f);
      auto done0 = makeI1Const(b, loc, false);
      auto outer = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{found0, done0});
      b.setInsertionPointToStart(outer.getBody());
      auto j = outer.getInductionVar();
      auto found = outer.getRegionIterArgs()[0];
      auto done = outer.getRegionIterArgs()[1];

      auto vOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{row, j});
      if (mlir::failed(vOr))
        return mlir::failure();
      auto v = castScalar(b, loc, *vOr, b.getF32Type());

      auto rank0 = c0;
      auto inner = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{rank0});
      b.setInsertionPointToStart(inner.getBody());
      auto k = inner.getInductionVar();
      auto rank = inner.getRegionIterArgs()[0];
      auto wOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{row, k});
      if (mlir::failed(wOr))
        return mlir::failure();
      auto w = castScalar(b, loc, *wOr, b.getF32Type());
      mlir::Value less;
      if (descending) {
        less = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, w, v).getResult();
      } else {
        less = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, w, v).getResult();
      }
      mlir::Value eqBefore = makeI1Const(b, loc, false);
      if (stable) {
        auto eq = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, w, v).getResult();
        auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, k, j).getResult();
        eqBefore = b.create<mlir::arith::AndIOp>(loc, eq, lt).getResult();
      }
      auto cond = b.create<mlir::arith::OrIOp>(loc, less, eqBefore).getResult();
      auto rank2 =
          b.create<mlir::arith::SelectOp>(loc, cond, b.create<mlir::arith::AddIOp>(loc, rank, c1).getResult(), rank)
              .getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{rank2});
      b.setInsertionPointAfter(inner);
      auto rankOut = inner.getResult(0);

      auto isPos = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, rankOut, pos).getResult();
      auto take = b.create<mlir::arith::AndIOp>(loc, isPos, b.create<mlir::arith::XOrIOp>(loc, done, makeI1Const(b, loc, true)).getResult())
                      .getResult();
      auto found2 = b.create<mlir::arith::SelectOp>(loc, take, v, found).getResult();
      auto done2 = b.create<mlir::arith::OrIOp>(loc, done, take).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{found2, done2});
      b.setInsertionPointAfter(outer);
      return castScalar(b, loc, outer.getResult(0), scalarTypeFor(op.output));
    }

    if (op.op == "unique") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: unique expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 1 || outShape.size() != 1) {
        ctx.module.emitError("full196: unique expects inp[N] -> out[U]");
        return mlir::failure();
      }
      const bool sorted = op.attrs.getBoolean("sorted").value_or(true);
      if (!sorted) {
        ctx.module.emitError("full196: unique supports sorted=true only");
        return mlir::failure();
      }
      int64_t N = shIn[0];
      auto u = outIdx[0];

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cN = makeIndexConst(b, loc, N);
      auto found0 = makeI32Const(b, loc, 0);
      auto done0 = makeI1Const(b, loc, false);
      auto outer = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{found0, done0});
      b.setInsertionPointToStart(outer.getBody());
      auto j = outer.getInductionVar();
      auto found = outer.getRegionIterArgs()[0];
      auto done = outer.getRegionIterArgs()[1];

      auto vOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{j});
      if (mlir::failed(vOr))
        return mlir::failure();
      auto v = castScalar(b, loc, *vOr, b.getI32Type());

      // isFirst(v): no earlier equal.
      auto first0 = makeI1Const(b, loc, true);
      auto firstLoop = b.create<mlir::scf::ForOp>(loc, c0, j, c1, mlir::ValueRange{first0});
      b.setInsertionPointToStart(firstLoop.getBody());
      auto t = firstLoop.getInductionVar();
      auto isFirst = firstLoop.getRegionIterArgs()[0];
      auto wOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{t});
      if (mlir::failed(wOr))
        return mlir::failure();
      auto w = castScalar(b, loc, *wOr, b.getI32Type());
      auto eq = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, w, v).getResult();
      auto notEq = b.create<mlir::arith::XOrIOp>(loc, eq, makeI1Const(b, loc, true)).getResult();
      auto isFirst2 = b.create<mlir::arith::AndIOp>(loc, isFirst, notEq).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{isFirst2});
      b.setInsertionPointAfter(firstLoop);
      auto first = firstLoop.getResult(0);

      // distinctLess(v): count distinct values < v.
      auto distinct0 = c0;
      auto kLoop = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{distinct0});
      b.setInsertionPointToStart(kLoop.getBody());
      auto k = kLoop.getInductionVar();
      auto distinct = kLoop.getRegionIterArgs()[0];
      auto xOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{k});
      if (mlir::failed(xOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getI32Type());
      auto less = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, x, v).getResult();

      // first occurrence of x?
      auto firstX0 = makeI1Const(b, loc, true);
      auto tLoop = b.create<mlir::scf::ForOp>(loc, c0, k, c1, mlir::ValueRange{firstX0});
      b.setInsertionPointToStart(tLoop.getBody());
      auto tt = tLoop.getInductionVar();
      auto firstX = tLoop.getRegionIterArgs()[0];
      auto yOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{tt});
      if (mlir::failed(yOr))
        return mlir::failure();
      auto y = castScalar(b, loc, *yOr, b.getI32Type());
      auto eq2 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, y, x).getResult();
      auto notEq2 = b.create<mlir::arith::XOrIOp>(loc, eq2, makeI1Const(b, loc, true)).getResult();
      auto firstX2 = b.create<mlir::arith::AndIOp>(loc, firstX, notEq2).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{firstX2});
      b.setInsertionPointAfter(tLoop);
      auto firstOcc = tLoop.getResult(0);

      auto inc = b.create<mlir::arith::AndIOp>(loc, less, firstOcc).getResult();
      auto distinct2 =
          b.create<mlir::arith::SelectOp>(loc, inc, b.create<mlir::arith::AddIOp>(loc, distinct, c1).getResult(), distinct)
              .getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{distinct2});
      b.setInsertionPointAfter(kLoop);
      auto distinctLess = kLoop.getResult(0);

      auto matchU = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, distinctLess, u).getResult();
      auto match = b.create<mlir::arith::AndIOp>(loc, first, matchU).getResult();
      auto take = b.create<mlir::arith::AndIOp>(loc, match, b.create<mlir::arith::XOrIOp>(loc, done, makeI1Const(b, loc, true)).getResult())
                      .getResult();
      auto found2 = b.create<mlir::arith::SelectOp>(loc, take, v, found).getResult();
      auto done2 = b.create<mlir::arith::OrIOp>(loc, done, take).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{found2, done2});
      b.setInsertionPointAfter(outer);
      return castScalar(b, loc, outer.getResult(0), scalarTypeFor(op.output));
    }

    if (op.op == "nonzero") {
      if (op.inputs.size() != 1) {
        ctx.module.emitError("full196: nonzero expects 1 input");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 2 || outShape.size() != 2 || outShape[1] != 2) {
        ctx.module.emitError("full196: nonzero expects inp[M,N] -> out[L,2]");
        return mlir::failure();
      }
      int64_t M = shIn[0];
      int64_t N = shIn[1];
      auto target = outIdx[0];
      auto axis = outIdx[1];

      auto total = makeIndexConst(b, loc, M * N);
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cnt0 = c0;
      auto fr0 = c0;
      auto fc0 = c0;
      auto done0 = makeI1Const(b, loc, false);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, total, c1, mlir::ValueRange{cnt0, fr0, fc0, done0});
      b.setInsertionPointToStart(forOp.getBody());
      auto lin = forOp.getInductionVar();
      auto cnt = forOp.getRegionIterArgs()[0];
      auto fr = forOp.getRegionIterArgs()[1];
      auto fc = forOp.getRegionIterArgs()[2];
      auto done = forOp.getRegionIterArgs()[3];
      auto idx = delinearizeIndex(b, loc, lin, llvm::ArrayRef<int64_t>{M, N});
      auto vOr = emitScalar(inName, idx);
      if (mlir::failed(vOr))
        return mlir::failure();
      auto v = castScalar(b, loc, *vOr, b.getF32Type());
      auto zero = makeF32Const(b, loc, 0.0f);
      auto nz = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, v, zero).getResult();
      auto isTarget = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, cnt, target).getResult();
      auto take = b.create<mlir::arith::AndIOp>(loc, nz, isTarget).getResult();
      auto done2 = b.create<mlir::arith::OrIOp>(loc, done, take).getResult();
      auto cnt2 = b.create<mlir::arith::SelectOp>(loc, nz, b.create<mlir::arith::AddIOp>(loc, cnt, c1).getResult(), cnt)
                      .getResult();
      auto fr2 = b.create<mlir::arith::SelectOp>(loc, take, idx[0], fr).getResult();
      auto fc2 = b.create<mlir::arith::SelectOp>(loc, take, idx[1], fc).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{cnt2, fr2, fc2, done2});
      b.setInsertionPointAfter(forOp);
      auto rowI = forOp.getResult(1);
      auto colI = forOp.getResult(2);
      auto wantRow = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, axis, c0).getResult();
      auto idxOut = b.create<mlir::arith::SelectOp>(loc, wantRow, rowI, colI).getResult();
      auto i64 = castIndexToInt(b, loc, idxOut, 64);
      return castScalar(b, loc, i64, scalarTypeFor(op.output));
    }

    if (op.op == "quantile") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: quantile expects inputs (inp, q)");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto qName = op.inputs[1];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 2 || outShape.size() != 1) {
        ctx.module.emitError("full196: quantile expects inp[M,N] -> out[M]");
        return mlir::failure();
      }
      int64_t dim = static_cast<int64_t>(op.attrs.getInteger("dim").value_or(1));
      if (dim != 1) {
        ctx.module.emitError("full196: quantile supports dim=1 only");
        return mlir::failure();
      }
      auto interp = op.attrs.getString("interpolation");
      if (interp && interp->trim() != "linear") {
        ctx.module.emitError("full196: quantile supports interpolation=linear only");
        return mlir::failure();
      }
      int64_t N = shIn[1];
      auto row = outIdx[0];
      auto qOr = emitScalar(qName, llvm::ArrayRef<mlir::Value>{});
      if (mlir::failed(qOr))
        return mlir::failure();
      auto qv = castScalar(b, loc, *qOr, b.getF32Type());
      auto nMinus1 = makeF32Const(b, loc, static_cast<float>(std::max<int64_t>(1, N - 1)));
      auto kf = b.create<mlir::arith::MulFOp>(loc, qv, nMinus1).getResult();
      auto kFloorF = b.create<mlir::math::FloorOp>(loc, kf).getResult();
      auto k0i32 = b.create<mlir::arith::FPToSIOp>(loc, b.getI32Type(), kFloorF).getResult();
      auto k0 = castIntToIndex(b, loc, k0i32);
      auto frac = b.create<mlir::arith::SubFOp>(loc, kf, kFloorF).getResult();
      auto k1 = b.create<mlir::arith::AddIOp>(loc, k0, makeIndexConst(b, loc, 1)).getResult();
      auto k1c = b.create<mlir::arith::MinUIOp>(loc, k1, makeIndexConst(b, loc, std::max<int64_t>(0, N - 1))).getResult();

      auto kthValue = [&](mlir::Value kk) -> mlir::FailureOr<mlir::Value> {
        auto c0 = makeIndexConst(b, loc, 0);
        auto c1 = makeIndexConst(b, loc, 1);
        auto cN = makeIndexConst(b, loc, N);
        auto found0 = makeF32Const(b, loc, 0.0f);
        auto done0 = makeI1Const(b, loc, false);
        auto outer = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{found0, done0});
        b.setInsertionPointToStart(outer.getBody());
        auto j = outer.getInductionVar();
        auto found = outer.getRegionIterArgs()[0];
        auto done = outer.getRegionIterArgs()[1];
        auto vOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{row, j});
        if (mlir::failed(vOr))
          return mlir::failure();
        auto v = castScalar(b, loc, *vOr, b.getF32Type());
        auto rank0 = c0;
        auto inner = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{rank0});
        b.setInsertionPointToStart(inner.getBody());
        auto k = inner.getInductionVar();
        auto rank = inner.getRegionIterArgs()[0];
        auto wOr = emitScalar(inName, llvm::ArrayRef<mlir::Value>{row, k});
        if (mlir::failed(wOr))
          return mlir::failure();
        auto w = castScalar(b, loc, *wOr, b.getF32Type());
        auto less = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, w, v).getResult();
        auto eq = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, w, v).getResult();
        auto lt = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, k, j).getResult();
        auto eqBefore = b.create<mlir::arith::AndIOp>(loc, eq, lt).getResult();
        auto cond = b.create<mlir::arith::OrIOp>(loc, less, eqBefore).getResult();
        auto rank2 =
            b.create<mlir::arith::SelectOp>(loc, cond, b.create<mlir::arith::AddIOp>(loc, rank, c1).getResult(), rank)
                .getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{rank2});
        b.setInsertionPointAfter(inner);
        auto r = inner.getResult(0);
        auto isK = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, r, kk).getResult();
        auto take = b.create<mlir::arith::AndIOp>(loc, isK, b.create<mlir::arith::XOrIOp>(loc, done, makeI1Const(b, loc, true)).getResult())
                        .getResult();
        auto found2 = b.create<mlir::arith::SelectOp>(loc, take, v, found).getResult();
        auto done2 = b.create<mlir::arith::OrIOp>(loc, done, take).getResult();
        b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{found2, done2});
        b.setInsertionPointAfter(outer);
        return outer.getResult(0);
      };

      auto v0Or = kthValue(k0);
      if (mlir::failed(v0Or))
        return mlir::failure();
      auto v1Or = kthValue(k1c);
      if (mlir::failed(v1Or))
        return mlir::failure();
      auto v0 = *v0Or;
      auto v1 = *v1Or;
      auto diff = b.create<mlir::arith::SubFOp>(loc, v1, v0).getResult();
      auto scaled = b.create<mlir::arith::MulFOp>(loc, frac, diff).getResult();
      auto outV = b.create<mlir::arith::AddFOp>(loc, v0, scaled).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "mse_loss") {
      if (op.inputs.size() != 2) {
        ctx.module.emitError("full196: mse_loss expects inputs (inp, target)");
        return mlir::failure();
      }
      auto inName = op.inputs[0];
      auto tgtName = op.inputs[1];
      auto shIn = shapeOf(inName);
      if (shIn.size() != 2 || outShape.size() != 0) {
        ctx.module.emitError("full196: mse_loss expects inp[M,N] -> out[]");
        return mlir::failure();
      }
      int64_t reduction = static_cast<int64_t>(op.attrs.getInteger("reduction").value_or(1));
      int64_t M = shIn[0];
      int64_t N = shIn[1];
      auto total = makeIndexConst(b, loc, M * N);
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto sum0 = makeF32Const(b, loc, 0.0f);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, total, c1, mlir::ValueRange{sum0});
      b.setInsertionPointToStart(forOp.getBody());
      auto lin = forOp.getInductionVar();
      auto acc = forOp.getRegionIterArgs()[0];
      auto idx = delinearizeIndex(b, loc, lin, llvm::ArrayRef<int64_t>{M, N});
      auto xOr = emitScalar(inName, idx);
      auto yOr = emitScalar(tgtName, idx);
      if (mlir::failed(xOr) || mlir::failed(yOr))
        return mlir::failure();
      auto x = castScalar(b, loc, *xOr, b.getF32Type());
      auto y = castScalar(b, loc, *yOr, b.getF32Type());
      auto d = b.create<mlir::arith::SubFOp>(loc, x, y).getResult();
      auto sq = b.create<mlir::arith::MulFOp>(loc, d, d).getResult();
      auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, sq).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
      b.setInsertionPointAfter(forOp);
      auto sum = forOp.getResult(0);
      if (reduction == 2) { // sum
        return castScalar(b, loc, sum, scalarTypeFor(op.output));
      }
      // mean (default)
      auto denom = makeF32Const(b, loc, static_cast<float>(std::max<int64_t>(1, M * N)));
      auto outV = b.create<mlir::arith::DivFOp>(loc, sum, denom).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "nll_loss_forward") {
      if (op.inputs.size() != 3) {
        ctx.module.emitError("full196: nll_loss_forward expects inputs (self, target, weight)");
        return mlir::failure();
      }
      auto selfName = op.inputs[0];
      auto tgtName = op.inputs[1];
      auto wName = op.inputs[2];
      auto shSelf = shapeOf(selfName);
      auto shT = shapeOf(tgtName);
      auto shW = shapeOf(wName);
      if (shSelf.size() != 2 || shT.size() != 1 || shW.size() != 1 || outShape.size() != 0) {
        ctx.module.emitError("full196: nll_loss_forward expects self[N,C], target[N], weight[C] -> output[]");
        return mlir::failure();
      }
      int64_t reduction = static_cast<int64_t>(op.attrs.getInteger("reduction").value_or(1));
      int64_t ignoreIndex = static_cast<int64_t>(op.attrs.getInteger("ignore_index").value_or(-100));
      int64_t N = shSelf[0];

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cN = makeIndexConst(b, loc, N);
      auto loss0 = makeF32Const(b, loc, 0.0f);
      auto wsum0 = makeF32Const(b, loc, 0.0f);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, cN, c1, mlir::ValueRange{loss0, wsum0});
      b.setInsertionPointToStart(forOp.getBody());
      auto n = forOp.getInductionVar();
      auto loss = forOp.getRegionIterArgs()[0];
      auto wsum = forOp.getRegionIterArgs()[1];

      auto tOr = emitScalar(tgtName, llvm::ArrayRef<mlir::Value>{n});
      if (mlir::failed(tOr))
        return mlir::failure();
      auto t = castScalar(b, loc, *tOr, b.getI64Type());
      auto ign = makeI64Const(b, loc, ignoreIndex);
      auto isIgn = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, t, ign).getResult();
      auto keep = b.create<mlir::arith::XOrIOp>(loc, isIgn, makeI1Const(b, loc, true)).getResult();

      auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type(), b.getF32Type()}, keep,
                                            /*withElse=*/true);
      b.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto cls = castIntToIndex(b, loc, t);
      auto wOr = emitScalar(wName, llvm::ArrayRef<mlir::Value>{cls});
      auto xOr = emitScalar(selfName, llvm::ArrayRef<mlir::Value>{n, cls});
      if (mlir::failed(wOr) || mlir::failed(xOr))
        return mlir::failure();
      auto wv = castScalar(b, loc, *wOr, b.getF32Type());
      auto xv = castScalar(b, loc, *xOr, b.getF32Type());
      auto neg = b.create<mlir::arith::NegFOp>(loc, xv).getResult();
      auto term = b.create<mlir::arith::MulFOp>(loc, neg, wv).getResult();
      auto loss2 = b.create<mlir::arith::AddFOp>(loc, loss, term).getResult();
      auto wsum2 = b.create<mlir::arith::AddFOp>(loc, wsum, wv).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{loss2, wsum2});
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{loss, wsum});
      b.setInsertionPointAfter(ifOp);
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0), ifOp.getResult(1)});
      b.setInsertionPointAfter(forOp);

      auto lossOut = forOp.getResult(0);
      auto wsumOut = forOp.getResult(1);
      if (reduction == 2) { // sum
        return castScalar(b, loc, lossOut, scalarTypeFor(op.output));
      }
      auto zero = makeF32Const(b, loc, 0.0f);
      auto nz = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, wsumOut, zero).getResult();
      auto div = b.create<mlir::arith::DivFOp>(loc, lossOut, wsumOut).getResult();
      auto outV = b.create<mlir::arith::SelectOp>(loc, nz, div, zero).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "nll_loss2d_forward") {
      if (op.inputs.size() != 3) {
        ctx.module.emitError("full196: nll_loss2d_forward expects inputs (self, target, weight)");
        return mlir::failure();
      }
      auto selfName = op.inputs[0];
      auto tgtName = op.inputs[1];
      auto wName = op.inputs[2];
      auto shSelf = shapeOf(selfName);
      auto shT = shapeOf(tgtName);
      auto shW = shapeOf(wName);
      if (shSelf.size() != 4 || shT.size() != 3 || shW.size() != 1 || outShape.size() != 0) {
        ctx.module.emitError("full196: nll_loss2d_forward expects self[N,C,H,W], target[N,H,W], weight[C] -> output[]");
        return mlir::failure();
      }
      int64_t reduction = static_cast<int64_t>(op.attrs.getInteger("reduction").value_or(1));
      int64_t ignoreIndex = static_cast<int64_t>(op.attrs.getInteger("ignore_index").value_or(-100));
      int64_t N = shSelf[0];
      int64_t H = shSelf[2];
      int64_t W = shSelf[3];
      auto total = makeIndexConst(b, loc, N * H * W);
      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto loss0 = makeF32Const(b, loc, 0.0f);
      auto wsum0 = makeF32Const(b, loc, 0.0f);
      auto forOp = b.create<mlir::scf::ForOp>(loc, c0, total, c1, mlir::ValueRange{loss0, wsum0});
      b.setInsertionPointToStart(forOp.getBody());
      auto lin = forOp.getInductionVar();
      auto loss = forOp.getRegionIterArgs()[0];
      auto wsum = forOp.getRegionIterArgs()[1];
      // map lin -> (n,h,w)
      auto wIdx = b.create<mlir::arith::RemUIOp>(loc, lin, makeIndexConst(b, loc, W)).getResult();
      auto tmp = b.create<mlir::arith::DivUIOp>(loc, lin, makeIndexConst(b, loc, W)).getResult();
      auto hIdx = b.create<mlir::arith::RemUIOp>(loc, tmp, makeIndexConst(b, loc, H)).getResult();
      auto nIdx = b.create<mlir::arith::DivUIOp>(loc, tmp, makeIndexConst(b, loc, H)).getResult();

      auto tOr = emitScalar(tgtName, llvm::ArrayRef<mlir::Value>{nIdx, hIdx, wIdx});
      if (mlir::failed(tOr))
        return mlir::failure();
      auto t = castScalar(b, loc, *tOr, b.getI64Type());
      auto ign = makeI64Const(b, loc, ignoreIndex);
      auto isIgn = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, t, ign).getResult();
      auto keep = b.create<mlir::arith::XOrIOp>(loc, isIgn, makeI1Const(b, loc, true)).getResult();

      auto ifOp = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type(), b.getF32Type()}, keep,
                                            /*withElse=*/true);
      b.setInsertionPointToStart(&ifOp.getThenRegion().front());
      auto cls = castIntToIndex(b, loc, t);
      auto wOr = emitScalar(wName, llvm::ArrayRef<mlir::Value>{cls});
      auto xOr = emitScalar(selfName, llvm::ArrayRef<mlir::Value>{nIdx, cls, hIdx, wIdx});
      if (mlir::failed(wOr) || mlir::failed(xOr))
        return mlir::failure();
      auto wv = castScalar(b, loc, *wOr, b.getF32Type());
      auto xv = castScalar(b, loc, *xOr, b.getF32Type());
      auto neg = b.create<mlir::arith::NegFOp>(loc, xv).getResult();
      auto term = b.create<mlir::arith::MulFOp>(loc, neg, wv).getResult();
      auto loss2 = b.create<mlir::arith::AddFOp>(loc, loss, term).getResult();
      auto wsum2 = b.create<mlir::arith::AddFOp>(loc, wsum, wv).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{loss2, wsum2});
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{loss, wsum});
      b.setInsertionPointAfter(ifOp);
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifOp.getResult(0), ifOp.getResult(1)});
      b.setInsertionPointAfter(forOp);

      auto lossOut = forOp.getResult(0);
      auto wsumOut = forOp.getResult(1);
      if (reduction == 2) { // sum
        return castScalar(b, loc, lossOut, scalarTypeFor(op.output));
      }
      auto zero = makeF32Const(b, loc, 0.0f);
      auto nz = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, wsumOut, zero).getResult();
      auto div = b.create<mlir::arith::DivFOp>(loc, lossOut, wsumOut).getResult();
      auto outV = b.create<mlir::arith::SelectOp>(loc, nz, div, zero).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    if (op.op == "scaled_dot_product_attention") {
      if (op.inputs.size() != 3) {
        ctx.module.emitError("full196: scaled_dot_product_attention expects inputs (query, key, value)");
        return mlir::failure();
      }
      auto qName = op.inputs[0];
      auto kName = op.inputs[1];
      auto vName = op.inputs[2];
      auto shQ = shapeOf(qName);
      auto shK = shapeOf(kName);
      auto shV = shapeOf(vName);
      if (shQ.size() != 4 || shK.size() != 4 || shV.size() != 4 || outShape.size() != 4) {
        ctx.module.emitError("full196: scaled_dot_product_attention expects rank4 B,H,S,D");
        return mlir::failure();
      }
      int64_t D = shQ[3];
      int64_t K = shK[2];
      auto b0 = outIdx[0];
      auto h0 = outIdx[1];
      auto qpos = outIdx[2];
      auto dpos = outIdx[3];
      bool isCausal = op.attrs.getBoolean("is_causal").value_or(false);
      float scale = 1.0f / std::sqrt(static_cast<float>(std::max<int64_t>(1, D)));

      auto c0 = makeIndexConst(b, loc, 0);
      auto c1 = makeIndexConst(b, loc, 1);
      auto cK = makeIndexConst(b, loc, K);
      auto cD = makeIndexConst(b, loc, D);

      // Pass1: max score.
      auto max0 = makeF32Const(b, loc, -3.402823466e+38f);
      auto kFor = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{max0});
      b.setInsertionPointToStart(kFor.getBody());
      auto k = kFor.getInductionVar();
      auto maxv = kFor.getRegionIterArgs()[0];
      mlir::Value allow = makeI1Const(b, loc, true);
      if (isCausal) {
        allow = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, k, qpos).getResult();
      }
      auto ifMax = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type()}, allow, /*withElse=*/true);
      b.setInsertionPointToStart(&ifMax.getThenRegion().front());
      auto dot0 = makeF32Const(b, loc, 0.0f);
      auto dFor = b.create<mlir::scf::ForOp>(loc, c0, cD, c1, mlir::ValueRange{dot0});
      b.setInsertionPointToStart(dFor.getBody());
      auto dd = dFor.getInductionVar();
      auto acc = dFor.getRegionIterArgs()[0];
      auto qOr = emitScalar(qName, llvm::ArrayRef<mlir::Value>{b0, h0, qpos, dd});
      auto kkOr = emitScalar(kName, llvm::ArrayRef<mlir::Value>{b0, h0, k, dd});
      if (mlir::failed(qOr) || mlir::failed(kkOr))
        return mlir::failure();
      auto qv = castScalar(b, loc, *qOr, b.getF32Type());
      auto kv = castScalar(b, loc, *kkOr, b.getF32Type());
      auto prod = b.create<mlir::arith::MulFOp>(loc, qv, kv).getResult();
      auto acc2 = b.create<mlir::arith::AddFOp>(loc, acc, prod).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{acc2});
      b.setInsertionPointAfter(dFor);
      auto dot = dFor.getResult(0);
      auto score = b.create<mlir::arith::MulFOp>(loc, dot, makeF32Const(b, loc, scale)).getResult();
      auto max2 = b.create<mlir::arith::MaximumFOp>(loc, maxv, score).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{max2});
      b.setInsertionPointToStart(&ifMax.getElseRegion().front());
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{maxv});
      b.setInsertionPointAfter(ifMax);
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ifMax.getResult(0)});
      b.setInsertionPointAfter(kFor);
      auto maxScore = kFor.getResult(0);

      // Pass2: denom + numerator for this dpos.
      auto denom0 = makeF32Const(b, loc, 0.0f);
      auto num0 = makeF32Const(b, loc, 0.0f);
      auto kFor2 = b.create<mlir::scf::ForOp>(loc, c0, cK, c1, mlir::ValueRange{denom0, num0});
      b.setInsertionPointToStart(kFor2.getBody());
      auto kk = kFor2.getInductionVar();
      auto denom = kFor2.getRegionIterArgs()[0];
      auto num = kFor2.getRegionIterArgs()[1];
      mlir::Value allow2 = makeI1Const(b, loc, true);
      if (isCausal) {
        allow2 = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, kk, qpos).getResult();
      }
      auto if2 = b.create<mlir::scf::IfOp>(loc, mlir::TypeRange{b.getF32Type(), b.getF32Type()}, allow2, /*withElse=*/true);
      b.setInsertionPointToStart(&if2.getThenRegion().front());
      auto dot02 = makeF32Const(b, loc, 0.0f);
      auto dFor2 = b.create<mlir::scf::ForOp>(loc, c0, cD, c1, mlir::ValueRange{dot02});
      b.setInsertionPointToStart(dFor2.getBody());
      auto dd2 = dFor2.getInductionVar();
      auto accd = dFor2.getRegionIterArgs()[0];
      auto qOr2 = emitScalar(qName, llvm::ArrayRef<mlir::Value>{b0, h0, qpos, dd2});
      auto kOr2 = emitScalar(kName, llvm::ArrayRef<mlir::Value>{b0, h0, kk, dd2});
      if (mlir::failed(qOr2) || mlir::failed(kOr2))
        return mlir::failure();
      auto qv2 = castScalar(b, loc, *qOr2, b.getF32Type());
      auto kv2 = castScalar(b, loc, *kOr2, b.getF32Type());
      auto prod2 = b.create<mlir::arith::MulFOp>(loc, qv2, kv2).getResult();
      auto accd2 = b.create<mlir::arith::AddFOp>(loc, accd, prod2).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{accd2});
      b.setInsertionPointAfter(dFor2);
      auto dot2 = dFor2.getResult(0);
      auto score2 = b.create<mlir::arith::MulFOp>(loc, dot2, makeF32Const(b, loc, scale)).getResult();
      auto centered = b.create<mlir::arith::SubFOp>(loc, score2, maxScore).getResult();
      auto p = b.create<mlir::math::ExpOp>(loc, centered).getResult();
      auto denom2 = b.create<mlir::arith::AddFOp>(loc, denom, p).getResult();
      auto vOr = emitScalar(vName, llvm::ArrayRef<mlir::Value>{b0, h0, kk, dpos});
      if (mlir::failed(vOr))
        return mlir::failure();
      auto vv = castScalar(b, loc, *vOr, b.getF32Type());
      auto term = b.create<mlir::arith::MulFOp>(loc, p, vv).getResult();
      auto num2 = b.create<mlir::arith::AddFOp>(loc, num, term).getResult();
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{denom2, num2});
      b.setInsertionPointToStart(&if2.getElseRegion().front());
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{denom, num});
      b.setInsertionPointAfter(if2);
      b.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{if2.getResult(0), if2.getResult(1)});
      b.setInsertionPointAfter(kFor2);
      auto denomOut = kFor2.getResult(0);
      auto numOut = kFor2.getResult(1);
      auto zero = makeF32Const(b, loc, 0.0f);
      auto nz = b.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, denomOut, zero).getResult();
      auto div = b.create<mlir::arith::DivFOp>(loc, numOut, denomOut).getResult();
      auto outV = b.create<mlir::arith::SelectOp>(loc, nz, div, zero).getResult();
      return castScalar(b, loc, outV, scalarTypeFor(op.output));
    }

    // TODO: Remaining full196 macro ops are implemented incrementally.
    ctx.module.emitError() << "full196: unsupported op in cpp_plugin: " << op.op << " (output=" << op.output << ")";
    return mlir::failure();
  }
};

static mlir::LogicalResult lowerCudaFull196GraphV1(LoweringContext &ctx) {
  if (ctx.outputs.empty()) {
    ctx.module.emitError("full196: intent.outputs is empty");
    return mlir::failure();
  }

  auto *mlirCtx = ctx.module.getContext();
  auto loc = ctx.module.getLoc();
  auto &b = ctx.builder;

  // Resolve output totals for launch planning.
  int64_t maxOutTotal = 1;
  for (const auto &out : ctx.outputs) {
    auto it = ctx.tensors.find(out);
    if (it == ctx.tensors.end()) {
      ctx.module.emitError() << "full196: missing output tensor spec: " << out;
      return mlir::failure();
    }
    auto shOr = resolveShape(it->second, ctx.shapeBindings);
    if (mlir::failed(shOr)) {
      ctx.module.emitError() << "full196: failed to resolve output shape: " << out;
      return mlir::failure();
    }
    auto nOr = shapeNumel(*shOr);
    if (mlir::failed(nOr)) {
      ctx.module.emitError() << "full196: invalid output shape numel: " << out;
      return mlir::failure();
    }
    maxOutTotal = std::max<int64_t>(maxOutTotal, *nOr);
  }

  clearModuleBody(ctx.module);
  ctx.module->setAttr("gpu.container_module", mlir::UnitAttr::get(mlirCtx));
  if (!ctx.module->hasAttr("llvm.target_triple")) {
    ctx.module->setAttr("llvm.target_triple", mlir::StringAttr::get(mlirCtx, "nvptx64-nvidia-cuda"));
  }

  // GPU module + kernel.
  b.setInsertionPointToStart(&ctx.module.getBodyRegion().front());
  auto gpuModule = mlir::gpu::GPUModuleOp::create(b, loc, "kernels");
  b.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto fnOr = createCudaKernelWithFlattenedABI(ctx, gpuModule, sanitizeSymbolName(ctx.kernelName));
  if (mlir::failed(fnOr))
    return mlir::failure();
  auto fn = *fnOr;

  auto tid = b.create<mlir::gpu::ThreadIdOp>(loc, mlir::gpu::Dimension::x).getResult();
  auto bid = b.create<mlir::gpu::BlockIdOp>(loc, mlir::gpu::Dimension::x).getResult();
  auto bdim = b.create<mlir::gpu::BlockDimOp>(loc, mlir::gpu::Dimension::x).getResult();
  auto gdim = b.create<mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::x).getResult();
  auto lin0 =
      b.create<mlir::arith::AddIOp>(loc, b.create<mlir::arith::MulIOp>(loc, bid, bdim).getResult(), tid).getResult();
  auto stride = b.create<mlir::arith::MulIOp>(loc, bdim, gdim).getResult();

  CudaFull196Emitter emitter(ctx, fn);

  // Emit per-output parallel loops.
  for (const auto &outName : ctx.outputs) {
    auto outArg = getArgByName(ctx, fn, outName);
    if (!outArg) {
      ctx.module.emitError() << "full196: failed to map output arg: " << outName;
      return mlir::failure();
    }
    auto sh = emitter.shapeOf(outName);
    auto nOr = shapeNumel(std::vector<int64_t>(sh.begin(), sh.end()));
    if (mlir::failed(nOr))
      return mlir::failure();
    auto total = makeIndexConst(b, loc, *nOr);
    auto forOp = b.create<mlir::scf::ForOp>(loc, lin0, total, stride);
    b.setInsertionPointToStart(forOp.getBody());
    auto lin = forOp.getInductionVar();
    auto idx = delinearizeIndex(b, loc, lin, sh);
    auto vOr = emitter.emitScalar(outName, idx);
    if (mlir::failed(vOr)) {
      return mlir::failure();
    }
    mlir::Value v = *vOr;
    // Store with ABI bool mapping (i1 -> i8).
    if (isBoolDtype(emitter.dtypeOf(outName))) {
      v = castScalar(b, loc, v, b.getI1Type());
      v = mapScalarBoolToMemI8(b, loc, v);
    } else {
      auto outTy = dtypeToElemType(mlirCtx, emitter.dtypeOf(outName));
      if (!outTy) {
        ctx.module.emitError() << "full196: unsupported output memref dtype: " << emitter.dtypeOf(outName);
        return mlir::failure();
      }
      v = castScalar(b, loc, v, outTy);
    }
    b.create<mlir::memref::StoreOp>(loc, v, outArg, mlir::ValueRange{lin});
    b.setInsertionPointAfter(forOp);
  }

  b.create<mlir::gpu::ReturnOp>(loc);

  // Audit metadata.
  ctx.module->setAttr("intentir.compiler_stack", mlir::StringAttr::get(mlirCtx, "cpp_plugin"));
  ctx.module->setAttr("intentir.lowering_kind", mlir::StringAttr::get(mlirCtx, "cuda_full196_v1"));
  ctx.module->setAttr("intentir.cuda_real_mlir_kernel_kind",
                      mlir::StringAttr::get(mlirCtx, "cuda_full196_graph_v1"));
  int64_t threads = 256;
  int64_t gridX = std::max<int64_t>(1, (maxOutTotal + threads - 1) / threads);
  mergeIntentirMetaJson(ctx.module, [&](llvm::json::Object &meta) {
    meta["schema_version"] = "intentir_meta_v1";
    meta["compiler_stack"] = "cpp_plugin";
    meta["lowering_kind"] = "cuda_full196_v1";
    meta["cuda_real_mlir_kernel_emitted"] = true;
    meta["cuda_real_mlir_kernel_kind"] = "cuda_full196_graph_v1";
    meta["cuda_real_mlir_launch_override"] =
        makeCudaLaunchOverride(/*bx=*/threads, /*by=*/1, /*bz=*/1, /*gx=*/gridX, /*gy=*/1, /*gz=*/1);
    meta["cuda_real_mlir_output_total"] = static_cast<int64_t>(maxOutTotal);
    meta["cuda_real_mlir_elems_per_thread"] = static_cast<int64_t>(1);
  });
  return mlir::success();
}

class IntentIRExtractGPUModuleLLVMV1Pass
    : public mlir::PassWrapper<IntentIRExtractGPUModuleLLVMV1Pass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntentIRExtractGPUModuleLLVMV1Pass)

  llvm::StringRef getArgument() const final {
    return "intentir-extract-gpu-module-llvm-v1";
  }
  llvm::StringRef getDescription() const final {
    return "Move LLVM/NVVM IR out of gpu.module so mlir-translate can emit LLVM IR";
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::gpu::GPUModuleOp gpuModule;
    for (auto m : module.getOps<mlir::gpu::GPUModuleOp>()) {
      gpuModule = m;
      break;
    }
    if (!gpuModule)
      return;

    auto &topBlock = module.getBodyRegion().front();
    auto &gpuBlock = gpuModule.getBodyRegion().front();
    for (auto &op : llvm::make_early_inc_range(gpuBlock)) {
      op.moveBefore(&topBlock, topBlock.end());
    }
    gpuModule.erase();
  }
};

class IntentIRLowerRVVCpuLoopsV1Pass
    : public mlir::PassWrapper<IntentIRLowerRVVCpuLoopsV1Pass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntentIRLowerRVVCpuLoopsV1Pass)

  llvm::StringRef getArgument() const final { return "intentir-lower-rvv-cpu-loops-v1"; }
  llvm::StringRef getDescription() const final {
    return "IntentIR RVV lowering (cpu loops v1) from carrier intent_json_b64";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::math::MathDialect>();
    // Ensure downstream LLVM IR has a RISC-V target triple so rvv_remote_run can
    // compile via llc (-mtriple=...).
    if (!module->hasAttr("llvm.target_triple")) {
      module->setAttr("llvm.target_triple",
                      mlir::StringAttr::get(ctx, "riscv64-unknown-linux-gnu"));
    }

    auto ctxOr = parseLoweringContext(module);
    if (mlir::failed(ctxOr)) {
      signalPassFailure();
      return;
    }
    LoweringContext &lc = *ctxOr;

    const std::string k = lc.kernelName;
    mlir::LogicalResult ok = mlir::failure();
    if (k == "add2d") {
      ok = lowerElementwiseF32(lc);
    } else if (k == "row_sum") {
      ok = lowerRowSum(lc);
    } else if (k == "gather2d" || k == "flip2d") {
      ok = lowerGather2dLike(lc);
    } else if (k == "cat2d") {
      ok = lowerConcat2d(lc);
    } else if (k == "diag2d") {
      ok = lowerDiag2d(lc);
    } else {
      module.emitError() << "unsupported kernel for cpp rvv cpu-loops v1: " << k;
      ok = mlir::failure();
    }

    if (mlir::failed(ok)) {
      signalPassFailure();
      return;
    }
  }
};

class IntentIRApplyTuningDbCudaV1Pass
    : public mlir::PassWrapper<IntentIRApplyTuningDbCudaV1Pass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntentIRApplyTuningDbCudaV1Pass)

  llvm::StringRef getArgument() const final { return "intentir-apply-tuning-db-cuda-v1"; }
  llvm::StringRef getDescription() const final {
    return "Apply tuning_db (bindings + optional kernel_kind override) into carrier module attrs";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *mlirCtx = module.getContext();

    std::string kernel;
    if (auto attr = module->getAttrOfType<mlir::StringAttr>("intentir.intent_name")) {
      kernel = attr.str();
    }
    if (kernel.empty()) {
      return;
    }

    auto sbAttr = module->getAttrOfType<mlir::StringAttr>("intentir.shape_bindings_b64");
    if (!sbAttr) {
      // Without concrete shape bindings we cannot match tuning_db conditions.
      return;
    }
    auto decodedOr = decodeB64(sbAttr.str());
    if (mlir::failed(decodedOr)) {
      module.emitError() << "invalid intentir.shape_bindings_b64 (base64 decode failed)";
      signalPassFailure();
      return;
    }
    auto parsedOr = parseJson(*decodedOr);
    if (mlir::failed(parsedOr)) {
      module.emitError() << "invalid intentir.shape_bindings_b64 (JSON parse failed)";
      signalPassFailure();
      return;
    }
    auto baseBindingsOr = parseShapeBindings(*parsedOr);
    if (mlir::failed(baseBindingsOr)) {
      module.emitError() << "invalid intentir.shape_bindings_b64 (expected JSON object of ints)";
      signalPassFailure();
      return;
    }
    const std::map<std::string, int64_t> baseBindings = *baseBindingsOr;

    const std::string arch = detectCudaArchForTuning();
    if (arch.empty()) {
      // Arch-specific tuning requires INTENTIR_CUDA_SM to be set (or skip).
      return;
    }

    const char *envPath = std::getenv("INTENTIR_CUDA_TUNING_DB");
    if (!envPath || !*envPath)
      envPath = std::getenv("INTENTIR_TUNING_DB");
    const bool explicitPath = (envPath && *envPath);

    std::string path = explicitPath ? std::string(envPath)
                                    : "workflow/flaggems/state/tuning_db/cuda.jsonl";
    if (!llvm::sys::fs::exists(path)) {
      if (explicitPath) {
        module.emitError() << "INTENTIR_CUDA_TUNING_DB points to missing file: " << path;
        signalPassFailure();
      }
      return;
    }

    auto bufOr = llvm::MemoryBuffer::getFile(path);
    if (!bufOr) {
      module.emitError() << "failed to read tuning_db: " << path;
      signalPassFailure();
      return;
    }
    llvm::StringRef rest = (*bufOr)->getBuffer();

    std::map<std::string, int64_t> applied;
    std::string kernelKind;
    int lineNo = 0;
    while (true) {
      ++lineNo;
      auto parts = rest.split('\n');
      llvm::StringRef line = parts.first.trim();
      rest = parts.second;

      if (!line.empty() && !line.starts_with("#")) {
        auto parsedLine = llvm::json::parse(line);
        if (!parsedLine) {
          module.emitError() << "tuning_db parse error at " << path << ":" << lineNo;
          signalPassFailure();
          return;
        }
        const auto *obj = parsedLine->getAsObject();
        if (!obj) {
          module.emitError() << "tuning_db invalid row (expected JSON object) at " << path
                             << ":" << lineNo;
          signalPassFailure();
          return;
        }

        auto backend = obj->getString("backend");
        if (backend && backend->trim().lower() != "cuda") {
          goto next_line;
        }
        auto k = obj->getString("kernel");
        auto a = obj->getString("arch");
        if (!k || !a) {
          goto next_line;
        }
        if (k->trim() != kernel) {
          goto next_line;
        }
        if (a->trim().lower() != arch) {
          goto next_line;
        }
        if (!stackMatchesForCppPlugin(*obj)) {
          goto next_line;
        }

        const llvm::json::Value *whenVal = obj->get("when");
        if (!whenVal)
          whenVal = obj->get("shape");
        if (whenVal) {
          auto whenObj = whenVal->getAsObject();
          if (!whenObj)
            goto next_line;
          if (!matchWhen(*whenObj, baseBindings))
            goto next_line;
        }

        if (auto bindingsObj = obj->getObject("bindings")) {
          for (const auto &kv : *bindingsObj) {
            auto ii = kv.second.getAsInteger();
            if (!ii)
              continue;
            applied[kv.first.str()] = static_cast<int64_t>(*ii);
          }
        }

        auto kk = obj->getString("kernel_kind");
        if (!kk)
          kk = obj->getString("variant");
        if (kk) {
          std::string s = kk->trim().str();
          if (!s.empty())
            kernelKind = std::move(s);
        }
      }

    next_line:
    if (rest.empty())
        break;
    }

    const bool anyTuning = (!applied.empty() || !kernelKind.empty());
    if (anyTuning) {
      std::map<std::string, int64_t> merged = baseBindings;
      for (const auto &kv : applied) {
        merged[kv.first] = kv.second;
      }

      const std::string jsonText = encodeShapeBindingsJson(merged);
      const std::string b64 = llvm::encodeBase64(llvm::StringRef(jsonText));
      module->setAttr("intentir.shape_bindings_b64", mlir::StringAttr::get(mlirCtx, b64));

      if (!kernelKind.empty() && !module->hasAttr("intentir.kernel_kind_override")) {
        module->setAttr("intentir.kernel_kind_override",
                        mlir::StringAttr::get(mlirCtx, kernelKind));
      }
    }

    std::string finalKindOverride;
    if (auto kkAttr = module->getAttrOfType<mlir::StringAttr>("intentir.kernel_kind_override")) {
      finalKindOverride = kkAttr.str();
    }
    mergeIntentirMetaJson(module, [&](llvm::json::Object &meta) {
      meta["schema_version"] = "intentir_meta_v1";
      meta["compiler_stack"] = "cpp_plugin";
      meta["intentir_tuning_source"] = anyTuning ? "tuning_db" : "none";
      meta["intentir_tuning_arch"] = arch;
      meta["intentir_tuning_db"] = path;
      meta["intentir_tuning_applied"] = makeJsonIntObject(applied);
      if (!finalKindOverride.empty()) {
        meta["intentir_kernel_kind_override"] = finalKindOverride;
      }
    });
  }
};

class IntentIRLowerCudaFocusV1Pass
    : public mlir::PassWrapper<IntentIRLowerCudaFocusV1Pass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntentIRLowerCudaFocusV1Pass)

  llvm::StringRef getArgument() const final { return "intentir-lower-cuda-focus-v1"; }
  llvm::StringRef getDescription() const final {
    return "IntentIR CUDA lowering (focus kernels v1) from carrier intent_json_b64";
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
    ctx->getOrLoadDialect<mlir::nvgpu::NVGPUDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::math::MathDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();

    auto ctxOr = parseLoweringContext(module);
    if (mlir::failed(ctxOr)) {
      signalPassFailure();
      return;
    }
    LoweringContext &lc = *ctxOr;

    const std::string k = lc.kernelName;
    mlir::LogicalResult ok = mlir::failure();

    llvm::StringRef kindOverride = llvm::StringRef(lc.kernelKindOverride).trim();

    if (k == "ai_bench_matmul") {
      if (!kindOverride.empty()) {
        if (kindOverride == "matmul_mma_tf32_global_v1") {
          lc.shapeBindings["MMA_ASYNC_COPY"] = 0;
        } else if (kindOverride == "matmul_mma_tf32_v2") {
          lc.shapeBindings["MMA_ASYNC_COPY"] = 1;
        } else {
          module.emitError() << "invalid intentir.kernel_kind_override for ai_bench_matmul: " << kindOverride
                             << "; allowed=[matmul_mma_tf32_global_v1, matmul_mma_tf32_v2]";
          signalPassFailure();
          return;
        }
      }
      ok = lowerCudaAiBenchMatmulMmaTF32V1(lc);
    } else if (k == "matmul_fused_epilogue2d") {
      if (!kindOverride.empty()) {
        if (kindOverride == "matmul_fused_epilogue_mma_tf32_global_v1") {
          lc.shapeBindings["MMA_ASYNC_COPY"] = 0;
        } else if (kindOverride == "matmul_fused_epilogue_mma_tf32_v2") {
          lc.shapeBindings["MMA_ASYNC_COPY"] = 1;
        } else {
          module.emitError() << "invalid intentir.kernel_kind_override for matmul_fused_epilogue2d: " << kindOverride
                             << "; allowed=[matmul_fused_epilogue_mma_tf32_global_v1, matmul_fused_epilogue_mma_tf32_v2]";
          signalPassFailure();
          return;
        }
      }
      ok = lowerCudaMatmulFusedEpilogue2dMmaTF32V1(lc);
    } else if (k == "rms_norm2d") {
      llvm::StringRef kind = "rms_norm2d_rowwise_v2";
      bool valid = true;
      if (!kindOverride.empty()) {
        if (kindOverride == "rms_norm2d_rowwise_v1" || kindOverride == "rms_norm2d_rowwise_v2") {
          kind = kindOverride;
        } else {
          module.emitError() << "invalid intentir.kernel_kind_override for rms_norm2d: " << kindOverride
                             << "; allowed=[rms_norm2d_rowwise_v1, rms_norm2d_rowwise_v2]";
          valid = false;
        }
      }
      ok = !valid ? mlir::failure()
                  : (kind == "rms_norm2d_rowwise_v1" ? lowerCudaRmsNorm2dRowwiseV1(lc)
                                                    : lowerCudaRmsNorm2dRowwiseV2(lc));
    } else if (k == "flash_attention2d") {
      llvm::StringRef kind = "attn2d_causal_softmax_v6";
      bool valid = true;
      if (!kindOverride.empty()) {
        if (kindOverride == "attn2d_causal_softmax_v6" || kindOverride == "attn2d_causal_softmax_v7") {
          kind = kindOverride;
        } else {
          module.emitError() << "invalid intentir.kernel_kind_override for flash_attention2d: "
                             << kindOverride << "; allowed=[attn2d_causal_softmax_v6, attn2d_causal_softmax_v7]";
          valid = false;
        }
      } else {
        bool parallel = false;
        if (auto it = lc.shapeBindings.find("ATTN_PARALLEL_SOFTMAX"); it != lc.shapeBindings.end()) {
          parallel = (it->second != 0);
        }
        kind = parallel ? "attn2d_causal_softmax_v7" : "attn2d_causal_softmax_v6";
      }
      ok = valid ? lowerCudaFlashAttention2dCausalSoftmaxV6(lc, kind) : mlir::failure();
    } else if (k == "masked_attention2d") {
      if (!kindOverride.empty()) {
        if (kindOverride == "attn2d_causal_softmax_masked_hd16_keys_v1") {
          ok = lowerCudaMaskedAttention2dHd16KeysV1(lc, kindOverride);
        } else if (kindOverride == "attn2d_causal_softmax_warp_v2") {
          ok = lowerCudaAttn2dCausalSoftmaxWarpV2(lc, kindOverride);
        } else if (kindOverride == "attn2d_causal_softmax_warp_v1") {
          ok = lowerCudaAttn2dCausalSoftmaxWarpV1(lc, kindOverride);
        } else {
          module.emitError() << "invalid intentir.kernel_kind_override for masked_attention2d: "
                             << kindOverride
                             << "; allowed=[attn2d_causal_softmax_masked_hd16_keys_v1, attn2d_causal_softmax_warp_v2, "
                                "attn2d_causal_softmax_warp_v1]";
          ok = mlir::failure();
        }
      } else {
        bool hd16_keys = false;
        if (auto it = lc.shapeBindings.find("ATTN_MASKED_HD16_KEYS_V1"); it != lc.shapeBindings.end()) {
          hd16_keys = (it->second != 0);
        }
        bool v2 = false;
        if (auto it = lc.shapeBindings.find("ATTN_MASKED_SOFTMAX_V2"); it != lc.shapeBindings.end()) {
          v2 = (it->second != 0);
        }
        ok = hd16_keys ? lowerCudaMaskedAttention2dHd16KeysV1(lc, "attn2d_causal_softmax_masked_hd16_keys_v1")
                       : (v2 ? lowerCudaAttn2dCausalSoftmaxWarpV2(lc, "attn2d_causal_softmax_warp_v2")
                             : lowerCudaAttn2dCausalSoftmaxWarpV1(lc, "attn2d_causal_softmax_warp_v1"));
      }
    } else if (k == "_attn_fwd") {
      llvm::StringRef kind = "attn_fwd_softmax_v6";
      bool valid = true;
      if (!kindOverride.empty()) {
        if (kindOverride == "attn_fwd_softmax_v6" || kindOverride == "attn_fwd_softmax_v7") {
          kind = kindOverride;
        } else {
          module.emitError() << "invalid intentir.kernel_kind_override for _attn_fwd: " << kindOverride
                             << "; allowed=[attn_fwd_softmax_v6, attn_fwd_softmax_v7]";
          valid = false;
        }
      } else {
        bool parallel = false;
        if (auto it = lc.shapeBindings.find("ATTN_FWD_PARALLEL_SOFTMAX"); it != lc.shapeBindings.end()) {
          parallel = (it->second != 0);
        }
        kind = parallel ? "attn_fwd_softmax_v7" : "attn_fwd_softmax_v6";
      }
      ok = valid ? lowerCudaAttnFwdSoftmaxV6(lc, kind) : mlir::failure();
    } else {
      // Not a focus kernel. Leave the carrier module untouched so downstream
      // passes (e.g. full196 correctness-first lowering) can handle it.
      return;
    }
    if (mlir::failed(ok)) {
      signalPassFailure();
      return;
    }
  }
};

class IntentIRLowerCudaFull196V1Pass
    : public mlir::PassWrapper<IntentIRLowerCudaFull196V1Pass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntentIRLowerCudaFull196V1Pass)

  llvm::StringRef getArgument() const final { return "intentir-lower-cuda-full196-v1"; }
  llvm::StringRef getDescription() const final {
    return "IntentIR CUDA lowering (full196 correctness-first v1) from carrier intent_json_b64";
  }

  void runOnOperation() override {
    auto module = getOperation();

    // If focus lowering already produced a GPU kernel, do nothing.
    for (auto m : module.getOps<mlir::gpu::GPUModuleOp>()) {
      (void)m;
      return;
    }

    auto *ctx = module.getContext();
    ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::math::MathDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();

    auto ctxOr = parseLoweringContext(module);
    if (mlir::failed(ctxOr)) {
      signalPassFailure();
      return;
    }
    LoweringContext &lc = *ctxOr;
    if (mlir::failed(lowerCudaFull196GraphV1(lc))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

static void registerIntentIRPasses() {
  mlir::PassRegistration<IntentIRLowerRVVCpuLoopsV1Pass>();
  mlir::PassRegistration<IntentIRApplyTuningDbCudaV1Pass>();
  mlir::PassRegistration<IntentIRLowerCudaFocusV1Pass>();
  mlir::PassRegistration<IntentIRLowerCudaFull196V1Pass>();
  mlir::PassRegistration<IntentIRExtractGPUModuleLLVMV1Pass>();
}

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "IntentIRPasses",
      "v0.1",
      []() { registerIntentIRPasses(); },
  };
}
