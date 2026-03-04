#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "llvm/Support/JSON.h"
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
  if (d == "i32")
    return mlir::IntegerType::get(ctx, 32);
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
    const std::vector<std::string> &outputs) {
  std::set<std::string> produced;
  std::set<std::string> used;
  for (const auto &op : ops) {
    if (!op.output.empty())
      produced.insert(op.output);
    for (const auto &in : op.inputs)
      used.insert(in);
  }

  std::vector<std::string> externalInputs;
  for (const auto &name : used) {
    if (tensors.count(name) == 0)
      continue;
    if (produced.count(name))
      continue;
    externalInputs.push_back(name);
  }
  std::sort(externalInputs.begin(), externalInputs.end());

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
};

static mlir::Value makeIndexConst(mlir::OpBuilder &b, mlir::Location loc, int64_t v) {
  return b.create<mlir::arith::ConstantIndexOp>(loc, v);
}

static mlir::Value makeF32Const(mlir::OpBuilder &b, mlir::Location loc, float v) {
  return b.create<mlir::arith::ConstantFloatOp>(loc, b.getF32Type(), llvm::APFloat(v));
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
  };
  ctx.argOrder = computeIOArgOrder(ctx.tensors, ctx.ops, ctx.outputs);
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

static mlir::LogicalResult lowerCudaAiBenchMatmulMmaTF32GlobalV1(LoweringContext &ctx) {
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

  // Tunable tile params (kept consistent with python matmul_mma_tf32_global_v1).
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

  auto f32 = b.getF32Type();
  auto memSpace = mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), 1);

  auto a2Ty = mlir::MemRefType::get({M, K}, f32, mlir::MemRefLayoutAttrInterface{}, memSpace);
  auto b2Ty = mlir::MemRefType::get({K, N}, f32, mlir::MemRefLayoutAttrInterface{}, memSpace);
  auto c2Ty = mlir::MemRefType::get({M, N}, f32, mlir::MemRefLayoutAttrInterface{}, memSpace);

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

  // Unrolled KB/KK loops (global-load path).
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
      auto next =
          mlir::gpu::SubgroupMmaComputeOp::create(b, loc, cFragTy, aFrag, bFrag,
                                                  acc, /*a_transpose=*/{},
                                                  /*b_transpose=*/transposeAttr)
              .getResult();
      acc = next;
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
                      mlir::StringAttr::get(mlirCtx, "matmul_mma_tf32_global_v1"));

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
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();

    auto ctxOr = parseLoweringContext(module);
    if (mlir::failed(ctxOr)) {
      signalPassFailure();
      return;
    }
    LoweringContext &lc = *ctxOr;

    const std::string k = lc.kernelName;
    mlir::LogicalResult ok = mlir::failure();
    if (k == "ai_bench_matmul") {
      ok = lowerCudaAiBenchMatmulMmaTF32GlobalV1(lc);
    } else {
      module.emitError() << "unsupported kernel for cpp cuda focus v1: " << k;
      ok = mlir::failure();
    }
    if (mlir::failed(ok)) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

static void registerIntentIRPasses() {
  mlir::PassRegistration<IntentIRLowerRVVCpuLoopsV1Pass>();
  mlir::PassRegistration<IntentIRLowerCudaFocusV1Pass>();
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
