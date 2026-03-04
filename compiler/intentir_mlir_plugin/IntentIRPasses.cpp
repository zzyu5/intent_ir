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
  std::string kernelKindOverride;
};

static mlir::Value makeIndexConst(mlir::OpBuilder &b, mlir::Location loc, int64_t v) {
  return b.create<mlir::arith::ConstantIndexOp>(loc, v);
}

static mlir::Value makeI32Const(mlir::OpBuilder &b, mlir::Location loc, int32_t v) {
  return b.create<mlir::arith::ConstantIntOp>(loc, v, 32);
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
      ok = lowerCudaAiBenchMatmulMmaTF32V1(lc);
    } else if (k == "matmul_fused_epilogue2d") {
      ok = lowerCudaMatmulFusedEpilogue2dMmaTF32V1(lc);
    } else if (k == "rms_norm2d") {
      ok = lowerCudaRmsNorm2dRowwiseV1(lc);
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
