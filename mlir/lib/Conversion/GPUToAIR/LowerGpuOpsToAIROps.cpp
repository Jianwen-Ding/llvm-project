//===- LowerGpuOpsToAIROps.cpp - MLIR GPU to AIR lowering passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that lowers GPU modules to the LLVM IR the AIR
// target compiles, in AIR's address spaces (thread 0, device 1, constant 2,
// threadgroup 3), with calls to AIR's functions:
//
//   gpu.thread_id         air.thread_position_in_threadgroup
//   gpu.block_id          air.threadgroup_position_in_grid
//   gpu.block_dim         air.threads_per_threadgroup
//   gpu.grid_dim          air.threadgroups_per_grid
//   gpu.global_id         air.thread_position_in_grid
//   gpu.lane_id           air.thread_index_in_simdgroup
//   gpu.subgroup_id       air.simdgroup_index_in_threadgroup
//   gpu.num_subgroups     air.simdgroups_per_threadgroup
//   gpu.subgroup_size     32
//   gpu.barrier           air.wg.barrier
//   gpu.shuffle           air.simd_shuffle
//   gpu.ballot            air.simd_ballot
//   gpu.subgroup_reduce   air.simd_<op>
//   air.simdgroup_matrix_* air.simdgroup_matrix_8x8_*
//   gpu.func              functions marked "air-kernel"
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToAIR/GPUToAIRPass.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/IndexIntrinsicsOpLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOAIROPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

const char *const mlir::kAIRDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-"
    "f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-"
    "v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-"
    "n8:16:32";

namespace {
// AIR's address spaces.
enum : unsigned {
  kThreadAddressSpace = 0,
  kDeviceAddressSpace = 1,
  kConstantAddressSpace = 2,
  kThreadgroupAddressSpace = 3,
};

/// The discardable attribute that marks kernels during the conversion; they
/// end up with the "air-kernel" function attribute.
constexpr StringLiteral kKernelMarker = "air.kernel";
} // namespace

namespace {
/// What a call of an AIR function does besides computing its result, which
/// its declaration says so that LLVM can CSE, hoist and sink the calls (and
/// keeps from moving the SIMD-group ones across control flow).
enum class AIREffects {
  /// A value of the thread or the launch: no memory, no side effects.
  None,
  /// A SIMD-group function: the threads of the SIMD-group exchange values.
  Convergent,
  /// A SIMD-group matrix load (reads its pointer operand) or store (writes
  /// it).
  ConvergentRead,
  ConvergentWrite,
  /// A barrier: synchronizes the threads, and orders memory.
  Barrier,
};
} // namespace

/// A call of the AIR function `name`, declared in the GPU module as needed
/// with the attributes of `effects`.
static Value callAIR(ConversionPatternRewriter &rewriter, Operation *op,
                     StringRef name, Type result, ValueRange args,
                     AIREffects effects = AIREffects::None,
                     LLVM::ConstantRangeAttr range = nullptr) {
  Operation *module = op->getParentWithTrait<OpTrait::SymbolTable>();
  MLIRContext *ctx = op->getContext();
  SmallVector<Type> params(args.getTypes());
  Type resultType = result ? result : LLVM::LLVMVoidType::get(ctx);
  bool existed = SymbolTable::lookupSymbolIn(module, name) != nullptr;
  FailureOr<LLVM::LLVMFuncOp> fn =
      LLVM::lookupOrCreateFn(rewriter, module, name, params, resultType);
  if (failed(fn))
    return Value();
  if (!existed) {
    using LLVM::ModRefInfo;
    auto memory = [&](ModRefInfo arg, ModRefInfo other) {
      constexpr ModRefInfo none = ModRefInfo::NoModRef;
      return rewriter.getAttr<LLVM::MemoryEffectsAttr>(
          /*other=*/other, /*argMem=*/arg, /*inaccessibleMem=*/none,
          /*errnoMem=*/none, /*targetMem0=*/none, /*targetMem1=*/none);
    };
    fn->setNoUnwind(true);
    fn->setWillReturn(true);
    switch (effects) {
    case AIREffects::None:
    case AIREffects::Convergent:
      fn->setMemoryEffectsAttr(
          memory(ModRefInfo::NoModRef, ModRefInfo::NoModRef));
      break;
    case AIREffects::ConvergentRead:
      fn->setMemoryEffectsAttr(memory(ModRefInfo::Ref, ModRefInfo::NoModRef));
      break;
    case AIREffects::ConvergentWrite:
      fn->setMemoryEffectsAttr(memory(ModRefInfo::Mod, ModRefInfo::NoModRef));
      break;
    case AIREffects::Barrier:
      break;
    }
    fn->setConvergent(effects != AIREffects::None);
  }
  auto call = LLVM::CallOp::create(rewriter, op->getLoc(), *fn, args);
  if (range && result)
    call.setResAttrsAttr(rewriter.getArrayAttr({rewriter.getDictionaryAttr(
        rewriter.getNamedAttr(LLVM::LLVMDialect::getRangeAttrName(), range))}));
  return result ? call.getResult() : Value();
}

/// The value of an integer constant, from the converted operand or the
/// original one.
static std::optional<int64_t> constantInt(Value converted, Value original) {
  if (std::optional<int64_t> c = getConstantIntValue(converted))
    return c;
  return getConstantIntValue(original);
}

/// Whether \p v (an original, unconverted value) is the same in every thread
/// of a threadgroup: a constant, a kernel argument, a threadgroup-level id or
/// size, the induction variable of a loop with uniform bounds, or pure arith
/// of those (L15-E286).
static bool isThreadgroupUniform(Value v, unsigned depth,
                                 SmallPtrSetImpl<Value> &assumed) {
  if (depth > 12)
    return false;
  if (getConstantIntValue(v))
    return true;
  auto uniform = [&](Value o) {
    return isThreadgroupUniform(o, depth + 1, assumed);
  };
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    // The induction variable of a loop whose bounds and step are uniform
    // (every thread runs the same iterations) (L15-E297), and a value it
    // carries from a uniform start through uniform updates, the carried
    // value assumed uniform in its own update (L15-E299).
    if (auto loop =
            dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp())) {
      if (!uniform(loop.getLowerBound()) || !uniform(loop.getUpperBound()) ||
          !uniform(loop.getStep()))
        return false;
      if (arg == loop.getInductionVar())
        return true;
      if (!assumed.insert(v).second)
        return true;
      unsigned n = arg.getArgNumber() - loop.getNumInductionVars();
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      return uniform(loop.getInitArgs()[n]) && uniform(yield.getOperand(n));
    }
    auto func = dyn_cast<gpu::GPUFuncOp>(arg.getOwner()->getParentOp());
    return func && arg.getOwner()->isEntryBlock() &&
           arg.getArgNumber() < func.getNumArguments() &&
           isa<IntegerType, IndexType>(arg.getType());
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp, gpu::NumSubgroupsOp,
          gpu::SubgroupSizeOp>(def))
    return true;
  if (def->getDialect()->getNamespace() != "arith" || def->getNumRegions() ||
      !isMemoryEffectFree(def))
    return false;
  return llvm::all_of(def->getOperands(), uniform);
}

static bool isThreadgroupUniform(Value v) {
  SmallPtrSet<Value, 8> assumed;
  return isThreadgroupUniform(v, 0, assumed);
}

/// Set before the conversion on a gpu.shuffle whose offset
/// isThreadgroupUniform (whose original values the conversion replaces).
static constexpr llvm::StringLiteral kUniformOffsetAttr = "air.uniform_offset";

/// Extend or truncate an i32 to the index type.
static Value toIndexBitwidth(ConversionPatternRewriter &rewriter, Location loc,
                             Value value, unsigned indexBitwidth) {
  Type indexType = rewriter.getIntegerType(indexBitwidth);
  if (indexBitwidth > 32)
    return LLVM::ZExtOp::create(rewriter, loc, indexType, value);
  if (indexBitwidth < 32)
    return LLVM::TruncOp::create(rewriter, loc, indexType, value);
  return value;
}

/// AIR's name suffix for a float or 32-bit integer type ("f32", "s.i32").
static std::string typeSuffix(Type type, bool isUnsigned = false) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  return (isUnsigned ? "u.i" : "s.i") + std::to_string(type.getIntOrFloatBitWidth());
}

namespace {

using gpu::index_lowering::IndexKind;
using gpu::index_lowering::IntrType;

/// The values a component of a system value takes, from the launch bounds
/// the kernel knows (known_block_size, known_grid_size) and, for the op's own
/// dimension, its upper_bound.
template <typename SourceOp>
static LLVM::ConstantRangeAttr systemValueRange(SourceOp op, gpu::Dimension dim,
                                                IndexKind kind,
                                                IntrType type) {
  std::optional<uint32_t> bound;
  if (std::optional<APInt> b = op.getUpperBound(); b && dim == op.getDimension())
    bound = static_cast<uint32_t>(b->getZExtValue());
  return gpu::index_lowering::getIndexOpRange(op, dim, bound, kind, type,
                                              /*bitWidth=*/32);
}

/// A three-dimensional system value: a component (x, y or z) of the call of
/// the AIR function, or a constant when the launch bounds fix it. The call
/// carries the range of all three components when they are all bounded.
template <typename SourceOp>
struct GPUSystemValueLowering : ConvertOpToLLVMPattern<SourceOp> {
  GPUSystemValueLowering(const LLVMTypeConverter &converter, StringRef function,
                         IndexKind kind, IntrType type)
      : ConvertOpToLLVMPattern<SourceOp>(converter), function(function),
        kind(kind), type(type) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    unsigned indexBitwidth = this->getTypeConverter()->getIndexTypeBitwidth();
    LLVM::ConstantRangeAttr own =
        systemValueRange(op, op.getDimension(), kind, type);
    // Known: a constant.
    if (own && own.getLower() + 1 == own.getUpper()) {
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, rewriter.getIntegerType(indexBitwidth),
          own.getLower().getZExtValue());
      return success();
    }
    // The range of the vector: the union of its components'.
    LLVM::ConstantRangeAttr range;
    std::optional<APInt> lower, upper;
    for (gpu::Dimension d :
         {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
      LLVM::ConstantRangeAttr r = d == op.getDimension()
                                      ? own
                                      : systemValueRange(op, d, kind, type);
      if (!r) {
        lower.reset();
        break;
      }
      lower = lower ? llvm::APIntOps::umin(*lower, r.getLower()) : r.getLower();
      upper = upper ? llvm::APIntOps::umax(*upper, r.getUpper()) : r.getUpper();
    }
    if (lower && upper->ugt(*lower))
      range = LLVM::ConstantRangeAttr::get(rewriter.getContext(), *lower,
                                           *upper);
    Value xyz = callAIR(rewriter, op, (function + ".v3i32").str(),
                        VectorType::get({3}, i32), {}, AIREffects::None, range);
    Value dim = LLVM::ConstantOp::create(
        rewriter, loc, i32, static_cast<int32_t>(op.getDimension()));
    Value value = LLVM::ExtractElementOp::create(rewriter, loc, xyz, dim);
    rewriter.replaceOp(op, toIndexBitwidth(rewriter, loc, value, indexBitwidth));
    return success();
  }

  StringRef function;
  IndexKind kind;
  IntrType type;
};

/// The threads of the threadgroup, when the kernel knows its size.
static std::optional<uint64_t> knownThreadgroupThreads(Operation *op) {
  uint64_t threads = 1;
  for (gpu::Dimension d :
       {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    std::optional<uint32_t> n =
        gpu::getKnownDimensionSizeAround(op, gpu::DimensionKind::Block, d);
    if (!n)
      return std::nullopt;
    threads *= *n;
  }
  return threads;
}

/// A scalar system value of the SIMD-groups: lane (in [0, 32)), SIMD-group
/// index and SIMD-group count (a constant when the threadgroup size is
/// known).
template <typename SourceOp>
struct GPUSimdgroupValueLowering : ConvertOpToLLVMPattern<SourceOp> {
  GPUSimdgroupValueLowering(const LLVMTypeConverter &converter,
                            StringRef function)
      : ConvertOpToLLVMPattern<SourceOp>(converter), function(function) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type i32 = rewriter.getI32Type();
    unsigned indexBitwidth = this->getTypeConverter()->getIndexTypeBitwidth();
    std::optional<uint64_t> threads = knownThreadgroupThreads(op);
    // SIMD-groups of the threadgroup: ceil(threads / 32).
    std::optional<uint64_t> groups;
    if (threads)
      groups = (*threads + 31) / 32;
    LLVM::ConstantRangeAttr range;
    if constexpr (std::is_same_v<SourceOp, gpu::LaneIdOp>) {
      range = LLVM::ConstantRangeAttr::get(ctx, 32, 0, 32);
    } else if constexpr (std::is_same_v<SourceOp, gpu::NumSubgroupsOp>) {
      if (groups) {
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
            op, rewriter.getIntegerType(indexBitwidth), *groups);
        return success();
      }
      range = LLVM::ConstantRangeAttr::get(ctx, 32, 1, 33);
    } else {
      if (groups == 1u) {
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
            op, rewriter.getIntegerType(indexBitwidth), 0);
        return success();
      }
      range = LLVM::ConstantRangeAttr::get(ctx, 32, 0, groups ? *groups : 32);
    }
    // upper_bound: fewer lanes (SIMD-groups) than that, or (the count) at
    // most that many.
    if (std::optional<APInt> b = op.getUpperBound()) {
      uint64_t upper = b->getZExtValue() +
                       std::is_same_v<SourceOp, gpu::NumSubgroupsOp>;
      if (upper < range.getUpper().getZExtValue() &&
          upper > range.getLower().getZExtValue())
        range = LLVM::ConstantRangeAttr::get(ctx, range.getLower(),
                                             APInt(32, upper));
    }
    Value value =
        callAIR(rewriter, op, function, i32, {}, AIREffects::None, range);
    rewriter.replaceOp(op, toIndexBitwidth(rewriter, loc, value, indexBitwidth));
    return success();
  }

  StringRef function;
};

struct GPUSubgroupSizeOpLowering
    : ConvertOpToLLVMPattern<gpu::SubgroupSizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupSizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type indexType = rewriter.getIntegerType(
        getTypeConverter()->getIndexTypeBitwidth());
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, indexType, 32);
    return success();
  }
};

/// gpu.barrier orders device and threadgroup memory across the threadgroup.
struct GPUBarrierOpLowering : ConvertOpToLLVMPattern<gpu::BarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    if (adaptor.getNamedBarrier())
      return rewriter.notifyMatchFailure(op, "named barrier");
    // mem_flags: device (1) | threadgroup (2), or the memory of `memfence`
    // (none for `memfence []`, which only synchronizes).
    int64_t memFlags = 3;
    if (std::optional<ArrayAttr> spaces = op.getAddressSpaces()) {
      memFlags = 0;
      for (Attribute a : *spaces)
        switch (cast<gpu::AddressSpaceAttr>(a).getValue()) {
        case gpu::AddressSpace::Global:
          memFlags |= 1;
          break;
        case gpu::AddressSpace::Workgroup:
          memFlags |= 2;
          break;
        case gpu::AddressSpace::Private:
          break;
        }
    }
    Value flags = LLVM::ConstantOp::create(rewriter, loc, i32, memFlags);
    gpu::BarrierScope barrierScope = op.getScope();
    if (barrierScope != gpu::BarrierScope::Workgroup &&
        barrierScope != gpu::BarrierScope::Subgroup)
      return rewriter.notifyMatchFailure(op, "barrier scope beyond workgroup");
    // thread_scope_threadgroup (1).
    Value scope = LLVM::ConstantOp::create(rewriter, loc, i32, 1);
    callAIR(rewriter, op,
            barrierScope == gpu::BarrierScope::Subgroup
                ? "air.simdgroup.barrier"
                : "air.wg.barrier",
            nullptr, {flags, scope}, AIREffects::Barrier);
    rewriter.eraseOp(op);
    return success();
  }
};

/// gpu.shuffle: the source lane (lane - offset, lane + offset, lane ^ offset
/// or offset), valid within the thread's segment of `width` lanes and the
/// thread's own lane outside of it, and a shuffle of each 32-bit part.
///
/// With a constant offset within the segment and the whole SIMD-group as the
/// segment (or, for xor, any power of two), the shuffle is AIR's own
/// shuffle_xor, shuffle_up or shuffle_down, whose source lanes are those, and
/// the lanes they leave the thread keep its value: no lane arithmetic, and
/// for xor (and up and down validity) nothing at all.
struct GPUShuffleOpLowering : ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i1 = rewriter.getI1Type();
    Type i16 = rewriter.getI16Type();
    Type i32 = rewriter.getI32Type();
    auto laneRange = LLVM::ConstantRangeAttr::get(rewriter.getContext(), 32, 0,
                                                  32);
    std::optional<int64_t> width =
        constantInt(adaptor.getWidth(), op.getWidth());
    std::optional<int64_t> offset =
        constantInt(adaptor.getOffset(), op.getOffset());

    // AIR's shuffle_xor / up / down.
    StringRef native;
    if (width && offset && *offset >= 0 && *offset < *width && *width <= 32) {
      switch (op.getMode()) {
      case gpu::ShuffleMode::XOR:
        if (llvm::isPowerOf2_64(*width))
          native = "air.simd_shuffle_xor.u.i32";
        break;
      case gpu::ShuffleMode::UP:
        native = "air.simd_shuffle_up.u.i32";
        break;
      case gpu::ShuffleMode::DOWN:
        native = "air.simd_shuffle_down.u.i32";
        break;
      case gpu::ShuffleMode::IDX:
        break;
      }
    }
    // A narrower power-of-two segment (up, down) or a constant lane (idx):
    // the same AIR shuffle with a uniform offset or lane, and the thread's own
    // value where the source lane is outside its segment, as below (no
    // per-thread lane, which would take a shuffle per quad position)
    // (L15-E285).
    bool ownOutside = false;
    // A uniform lane (idx) or mask (xor) that is not a constant: AIR's shuffle
    // or shuffle_xor of it, the source lane being the lane or lane ^ mask
    // (past the SIMD-group: outside the segment), and the own value outside
    // the segment, as for a constant (L15-E286).
    // Up and down by a uniform offset of either sign: AIR's shift by it, or
    // the opposite shift by its negation where it is negative (the source
    // lane lane -+ offset either way), selected by its sign (L15-E296).
    bool uniformOffset = false;
    StringRef opposite;
    if (!offset && width && *width <= 32 && llvm::isPowerOf2_64(*width) &&
        op->hasAttr(kUniformOffsetAttr)) {
      switch (op.getMode()) {
      case gpu::ShuffleMode::IDX:
        native = "air.simd_shuffle.u.i32";
        break;
      case gpu::ShuffleMode::XOR:
        native = "air.simd_shuffle_xor.u.i32";
        break;
      case gpu::ShuffleMode::UP:
        native = "air.simd_shuffle_up.u.i32";
        opposite = "air.simd_shuffle_down.u.i32";
        break;
      case gpu::ShuffleMode::DOWN:
        native = "air.simd_shuffle_down.u.i32";
        opposite = "air.simd_shuffle_up.u.i32";
        break;
      }
      ownOutside = uniformOffset = true;
    }
    if (width && offset && *offset >= 0 && *width <= 32 &&
        llvm::isPowerOf2_64(*width) &&
        ((native.empty() && op.getMode() == gpu::ShuffleMode::IDX &&
          *offset < 32) ||
         (*width < 32 && *offset < *width &&
          (op.getMode() == gpu::ShuffleMode::UP ||
           op.getMode() == gpu::ShuffleMode::DOWN)))) {
      if (op.getMode() == gpu::ShuffleMode::IDX)
        native = "air.simd_shuffle.u.i32";
      ownOutside = true;
    }

    Value valid, lane16;
    if (ownOutside) {
      Value off = uniformOffset ? adaptor.getOffset()
                                : LLVM::ConstantOp::create(rewriter, loc, i32,
                                                           *offset)
                                      .getResult();
      lane16 = uniformOffset ? LLVM::TruncOp::create(rewriter, loc, i16, off)
                                   .getResult()
                             : LLVM::ConstantOp::create(rewriter, loc, i16,
                                                        *offset)
                                   .getResult();
      Value lane = callAIR(rewriter, op, "air.thread_index_in_simdgroup", i32,
                           {}, AIREffects::None, laneRange);
      auto c = [&](int64_t v) {
        return LLVM::ConstantOp::create(rewriter, loc, i32, v);
      };
      // A constant offset's test is the lane's position in its segment (or,
      // for idx, the segment): up by k, lane % w >= k; down, lane % w < w - k;
      // idx of lane c, lane & -w == c & -w (L15-E306).
      if (!uniformOffset) {
        Value pos = LLVM::AndOp::create(rewriter, loc, i32, lane,
                                        c(op.getMode() == gpu::ShuffleMode::IDX
                                              ? -*width
                                              : *width - 1));
        switch (op.getMode()) {
        case gpu::ShuffleMode::UP:
          valid = LLVM::ICmpOp::create(rewriter, loc,
                                       LLVM::ICmpPredicate::uge, pos,
                                       c(*offset));
          break;
        case gpu::ShuffleMode::DOWN:
          valid = LLVM::ICmpOp::create(rewriter, loc,
                                       LLVM::ICmpPredicate::ult, pos,
                                       c(*width - *offset));
          break;
        default:
          valid = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                       pos, c(*offset & -*width));
          break;
        }
      } else {
        Value start =
            LLVM::AndOp::create(rewriter, loc, i32, lane, c(-*width));
        Value srcLane;
        switch (op.getMode()) {
        case gpu::ShuffleMode::UP:
          srcLane = LLVM::SubOp::create(rewriter, loc, i32, lane, off);
          break;
        case gpu::ShuffleMode::DOWN:
          srcLane = LLVM::AddOp::create(rewriter, loc, i32, lane, off);
          break;
        case gpu::ShuffleMode::XOR:
          srcLane = LLVM::XOrOp::create(rewriter, loc, i32, lane, off);
          break;
        default:
          srcLane = off;
          break;
        }
        // (the source lane in the segment: a uniform offset may be negative)
        Value aboveStart = LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::sge, srcLane, start);
        Value belowEnd = LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::slt, srcLane,
            LLVM::AddOp::create(rewriter, loc, i32, start, c(*width)));
        valid = LLVM::AndOp::create(rewriter, loc, aboveStart, belowEnd);
      }
    } else if (!native.empty()) {
      lane16 = LLVM::ConstantOp::create(rewriter, loc, i16, *offset);
      if (op.getMode() == gpu::ShuffleMode::XOR || *offset == 0) {
        valid = LLVM::ConstantOp::create(rewriter, loc, i1, 1);
      } else {
        // up: lane >= offset; down: lane + offset < 32.
        Value lane = callAIR(rewriter, op, "air.thread_index_in_simdgroup", i32,
                             {}, AIREffects::None, laneRange);
        bool up = op.getMode() == gpu::ShuffleMode::UP;
        Value bound =
            LLVM::ConstantOp::create(rewriter, loc, i32, up ? *offset
                                                            : 32 - *offset);
        valid = LLVM::ICmpOp::create(
            rewriter, loc, up ? LLVM::ICmpPredicate::uge
                              : LLVM::ICmpPredicate::ult,
            lane, bound);
      }
    } else {
      Value lane = callAIR(rewriter, op, "air.thread_index_in_simdgroup", i32,
                           {}, AIREffects::None, laneRange);
      Value width = adaptor.getWidth();
      Value zero = LLVM::ConstantOp::create(rewriter, loc, i32, 0);
      Value negWidth = LLVM::SubOp::create(rewriter, loc, i32, zero, width);
      Value add = LLVM::AddOp::create(rewriter, loc, i32, lane, width);
      Value segmentEnd = LLVM::AndOp::create(rewriter, loc, i32, add, negWidth);
      Value srcLane;
      switch (op.getMode()) {
      case gpu::ShuffleMode::UP:
        srcLane = LLVM::SubOp::create(rewriter, loc, i32, lane,
                                      adaptor.getOffset());
        break;
      case gpu::ShuffleMode::DOWN:
        srcLane = LLVM::AddOp::create(rewriter, loc, i32, lane,
                                      adaptor.getOffset());
        break;
      case gpu::ShuffleMode::XOR:
        srcLane = LLVM::XOrOp::create(rewriter, loc, i32, lane,
                                      adaptor.getOffset());
        break;
      case gpu::ShuffleMode::IDX:
        srcLane = adaptor.getOffset();
        break;
      }
      // The source lane must be in the thread's segment of `width` lanes.
      Value segmentStart =
          LLVM::AndOp::create(rewriter, loc, i32, lane, negWidth);
      valid = LLVM::AndOp::create(
          rewriter, loc,
          LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::sge,
                               srcLane, segmentStart),
          LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt,
                               srcLane, segmentEnd));
      Value selected =
          LLVM::SelectOp::create(rewriter, loc, valid, srcLane, lane);
      lane16 = LLVM::TruncOp::create(rewriter, loc, i16, selected);
      native = "air.simd_shuffle.u.i32";
    }

    SmallVector<Value> parts;
    if (failed(LLVM::decomposeValue(rewriter, loc, adaptor.getValue(), i32,
                                    parts)))
      return rewriter.notifyMatchFailure(op, "cannot split value into i32");
    SmallVector<Value> shuffled;
    Value nonNegative, negLane16;
    if (!opposite.empty()) {
      Value off = adaptor.getOffset();
      Value zero32 = LLVM::ConstantOp::create(rewriter, loc, i32, 0);
      nonNegative = LLVM::ICmpOp::create(rewriter, loc,
                                         LLVM::ICmpPredicate::sge, off, zero32);
      negLane16 = LLVM::TruncOp::create(
          rewriter, loc, i16, LLVM::SubOp::create(rewriter, loc, zero32, off));
    }
    for (Value part : parts) {
      Value v = callAIR(rewriter, op, native, i32, {part, lane16},
                        AIREffects::Convergent);
      if (!opposite.empty())
        v = LLVM::SelectOp::create(
            rewriter, loc, nonNegative, v,
            callAIR(rewriter, op, opposite, i32, {part, negLane16},
                    AIREffects::Convergent));
      shuffled.push_back(v);
    }
    Value result = LLVM::composeValue(rewriter, loc, shuffled,
                                      adaptor.getValue().getType());
    if (ownOutside)
      result = LLVM::SelectOp::create(rewriter, loc, valid, result,
                                      adaptor.getValue());
    rewriter.replaceOp(op, {result, valid});
    return success();
  }
};

/// gpu.subgroup_broadcast: AIR's shuffle of the (uniform) lane, or its
/// broadcast of the first active lane (the first set bit of the ballot of
/// the active lanes), per 32-bit part (L15-E298).
struct GPUSubgroupBroadcastOpLowering
    : ConvertOpToLLVMPattern<gpu::SubgroupBroadcastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    bool first =
        op.getBroadcastType() == gpu::BroadcastType::first_active_lane;
    Value lane16;
    if (!first) {
      if (!adaptor.getLane())
        return rewriter.notifyMatchFailure(op, "no lane");
      lane16 = LLVM::TruncOp::create(rewriter, loc, rewriter.getI16Type(),
                                     adaptor.getLane());
    }
    SmallVector<Value> parts;
    if (failed(LLVM::decomposeValue(rewriter, loc, adaptor.getSrc(), i32,
                                    parts)))
      return rewriter.notifyMatchFailure(op, "cannot split value into i32");
    SmallVector<Value> results;
    for (Value part : parts)
      results.push_back(
          first ? callAIR(rewriter, op, "air.simd_broadcast_first.u.i32", i32,
                          {part}, AIREffects::Convergent)
                : callAIR(rewriter, op, "air.simd_shuffle.u.i32", i32,
                          {part, lane16}, AIREffects::Convergent));
    rewriter.replaceOp(op, LLVM::composeValue(rewriter, loc, results,
                                              adaptor.getSrc().getType()));
    return success();
  }
};

struct GPUBallotOpLowering : ConvertOpToLLVMPattern<gpu::BallotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::BallotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto intType = dyn_cast<IntegerType>(op.getType());
    if (!intType || intType.getWidth() < 32)
      return rewriter.notifyMatchFailure(op, "result must be at least 32 bits");
    Type i64 = rewriter.getI64Type();
    Value mask = callAIR(rewriter, op, "air.simd_ballot.i64", i64,
                         {adaptor.getPredicate()}, AIREffects::Convergent);
    if (intType.getWidth() < 64)
      mask = LLVM::TruncOp::create(rewriter, loc, intType, mask);
    rewriter.replaceOp(op, mask);
    return success();
  }
};

/// The key b ^ ((b >>s 31) & 0x7fffffff) of float bits b, whose signed order
/// is the floats' with -0.0 below +0.0 (NaNs aside); it is its own inverse.
static Value orderedFloatKey(ConversionPatternRewriter &rewriter, Location loc,
                             Value bits) {
  Type i32 = rewriter.getI32Type();
  Value sign = LLVM::AShrOp::create(
      rewriter, loc, bits, LLVM::ConstantOp::create(rewriter, loc, i32, 31));
  Value mask = LLVM::AndOp::create(
      rewriter, loc, sign,
      LLVM::ConstantOp::create(rewriter, loc, i32, 0x7fffffff));
  return LLVM::XOrOp::create(rewriter, loc, bits, mask);
}

/// Whether another gpu.subgroup_reduce maximumf / minimumf reduces the value
/// of \p op: they share its key and "is NaN" (and, over the same lanes, the
/// or of it).
static bool sharesNaNOr(gpu::SubgroupReduceOp op) {
  for (Operation *user : op.getValue().getUsers()) {
    auto other = dyn_cast<gpu::SubgroupReduceOp>(user);
    if (other && other != op &&
        (other.getOp() == gpu::AllReduceOperation::MAXIMUMF ||
         other.getOp() == gpu::AllReduceOperation::MINIMUMF))
      return true;
  }
  return false;
}

/// gpu.subgroup_reduce maximumf / minimumf of f32: AIR's float SIMD-group
/// maximum and minimum are maxnum and minnum, which drop NaNs and may order
/// the zeros either way, so it is the signed i32 maximum (minimum) of the
/// floats' keys, decoded. A NaN's key is replaced by the largest (smallest)
/// float's, the quiet NaN 0x7fc00000's (0xffffffff's), so that a NaN anywhere
/// gives that key, which decodes to the quiet NaN (to a NaN, which is replaced
/// by the quiet NaN), without a second reduction, an or of "is NaN"
/// (L15-E276; alive2: tests/alive-tv/gpu/subgroup-maximum-nan-key,
/// subgroup-minimum-nan-key). Where another such reduction of the value
/// shares the key, the keys are the values' own and a NaN is selected where
/// the or of "is NaN" is set (alive2: subgroup-maximum-key,
/// subgroup-minimum-key), which keeps the selects off the key's path to the
/// reductions (agx_cost throughput: a maximum and minimum of one value).
static Value reduceOrderedFloat(ConversionPatternRewriter &rewriter,
                                gpu::SubgroupReduceOp op, Value value,
                                StringRef group) {
  Location loc = op.getLoc();
  Type i32 = rewriter.getI32Type();
  Type f32 = value.getType();
  bool isMax = op.getOp() == gpu::AllReduceOperation::MAXIMUMF;
  auto i32Cst = [&](int64_t v) {
    return LLVM::ConstantOp::create(rewriter, loc, i32,
                                    rewriter.getI32IntegerAttr(v));
  };
  Value nan = LLVM::ConstantOp::create(
      rewriter, loc, f32,
      rewriter.getFloatAttr(f32, APFloat::getQNaN(APFloat::IEEEsingle())));
  // Used twice: one value even if it is undefined.
  value = LLVM::FreezeOp::create(rewriter, loc, value);
  Value isNaN = LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::uno,
                                     value, value);
  Value key = orderedFloatKey(
      rewriter, loc, LLVM::BitcastOp::create(rewriter, loc, i32, value));
  bool shared = sharesNaNOr(op);
  // (key(0x7fc00000) = 0x7fc00000, key(0xffffffff) = INT32_MIN)
  if (!shared)
    key = LLVM::SelectOp::create(rewriter, loc, isNaN,
                                 i32Cst(isMax ? 0x7fc00000 : INT32_MIN), key);
  Value reduced =
      callAIR(rewriter, op,
              ("air." + group + (isMax ? "_max" : "_min") + ".s.i32").str(),
              i32, {key}, AIREffects::Convergent);
  Value result = LLVM::BitcastOp::create(
      rewriter, loc, f32, orderedFloatKey(rewriter, loc, reduced));
  Value anyNaN;
  if (shared) {
    Value any =
        callAIR(rewriter, op, ("air." + group + "_or.s.i32").str(), i32,
                {LLVM::ZExtOp::create(rewriter, loc, i32, isNaN)},
                AIREffects::Convergent);
    anyNaN = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ne, any,
                                  i32Cst(0));
  } else if (!isMax) {
    // The minimum's key is INT32_MIN, 0xffffffff's, exactly where there is a
    // NaN: the decoded result is a NaN exactly there.
    anyNaN = LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::uno,
                                  result, result);
  } else {
    return result;
  }
  return LLVM::SelectOp::create(rewriter, loc, anyNaN, nan, result);
}

/// math.expm1 and math.log1p accurate for small arguments, where MathToLLVM's
/// exp(x) - 1 and log(1 + x) cancel to 0 (expm1(6e-8) was 0). For |x| < 1/2,
/// Kahan's and Goldberg's forms, in f32 for f16 and bf16 (then rounded once):
///   expm1(x) = x where u = exp(x) is 1, else (u - 1) x / log(u);
///   log1p(x) = x where u = 1 + x is 1, else log(u) x / (u - 1);
/// from 1/2, u - 1 and log(u), which do not cancel there (1 + x is exact
/// below -1/2), and whose products would overflow (expm1(88), log1p(3e38)).
/// A NaN propagates; -0 gives -0; infinities as exp and log give them.
/// expm1(x) from u = exp(x) by rewriteAccurateM1's form: x where u is 1 and
/// (u - 1) x / log(u) for |x| < 1/2, else u - 1 (the caller's u may be the
/// exp of another argument where it does not read the result: L15-E309).
static Value accurateExpM1(OpBuilder &b, Location loc, Value x, Value u) {
  Type t = x.getType();
  Value one = arith::ConstantOp::create(b, loc, t, b.getFloatAttr(t, 1.0));
  Value small = arith::CmpFOp::create(
      b, loc, arith::CmpFPredicate::OLT, math::AbsFOp::create(b, loc, x),
      arith::ConstantOp::create(b, loc, t, b.getFloatAttr(t, 0.5)));
  Value um1 = arith::SubFOp::create(b, loc, u, one);
  Value q = arith::DivFOp::create(b, loc,
                                  arith::MulFOp::create(b, loc, um1, x),
                                  math::LogOp::create(b, loc, u));
  Value isOne =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ, u, one);
  return arith::SelectOp::create(
      b, loc, small, arith::SelectOp::create(b, loc, isOne, x, q), um1);
}

/// Whether convert-gpu-to-air's f64-polynomial-fma is on (set by the pass for
/// its patterns).
static thread_local bool f64PolynomialFma = false;

/// u v + w in the f64 expansions' polynomials and fdlibm's steps (exp, log,
/// expm1): fdlibm's own multiply and add, or an fma with f64-polynomial-fma.
/// The emulated fma (its exact product, a 106-bit sum) costs more than a
/// multiply and an add (newk tput 83.0 against 27.2 + 35.6), and the results
/// differ in a few in 10^4 lanes, within the same maximum error (L15-E815).
static Value f64Fma(ImplicitLocOpBuilder &b, Value u, Value v, Value w) {
  if (f64PolynomialFma)
    return math::FmaOp::create(b, u, v, w);
  return arith::AddFOp::create(b, arith::MulFOp::create(b, u, v), w);
}

/// expm1 of an f64 x: fdlibm's s_expm1.c in f64 and i64 operations, its
/// branches selects (within 1 ulp; L15-E814). x = k ln 2 + r (k =
/// roundeven(x / ln 2), r = hi - lo with its rounding error c, as exp's
/// reduction), expm1(r) from fdlibm's rational form with one division, then
/// its reconstruction, whose six cases are one form here (below); 2^k
/// applied on the exponent field (each such value normal). x is clamped to
/// -40 (below which the result rounds to -1) and a NaN or an x above 709.78
/// (overflow) reduced as 0, the result selected; |x| below 2^-54 gives x
/// (and -0 -0). Not exp's u - 1 and Kahan's (u - 1) x / log(u) near 0: no
/// log and two divisions fewer.
static Value expm1F64(ImplicitLocOpBuilder &b, Value x) {
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto ci = [&](int64_t v) { return arith::ConstantIntOp::create(b, i64, v); };
  auto add = [&](Value u, Value v) { return arith::AddFOp::create(b, u, v); };
  auto sub = [&](Value u, Value v) { return arith::SubFOp::create(b, u, v); };
  auto mul = [&](Value u, Value v) { return arith::MulFOp::create(b, u, v); };
  auto neg = [&](Value u) { return arith::NegFOp::create(b, u); };
  auto fma = [&](Value u, Value v, Value w) -> Value {
    return f64Fma(b, u, v, w);
  };
  auto sel = [&](Value cnd, Value u, Value v) {
    return arith::SelectOp::create(b, cnd, u, v);
  };
  auto fcmp = [&](arith::CmpFPredicate p, Value u, Value v) {
    return arith::CmpFOp::create(b, p, u, v);
  };
  auto icmp = [&](arith::CmpIPredicate p, Value u, Value v) {
    return arith::CmpIOp::create(b, p, u, v);
  };
  auto bits = [&](Value v) { return arith::BitcastOp::create(b, i64, v); };
  auto dbl = [&](Value v) { return arith::BitcastOp::create(b, f64, v); };
  Value over = fcmp(arith::CmpFPredicate::UGT, x, c(7.09782712893383973096e+02));
  Value xr = sel(fcmp(arith::CmpFPredicate::OLT, x, c(-40.0)), c(-40.0), x);
  xr = sel(over, c(0.0), xr);
  Value kf = math::RoundEvenOp::create(b, mul(xr, c(1.44269504088896338700e+00)));
  Value hi = sub(xr, mul(kf, c(6.93147180369123816490e-01)));
  Value lo = mul(kf, c(1.90821492927058770002e-10));
  Value r = sub(hi, lo);
  Value cr = sub(sub(hi, r), lo);
  Value hfx = mul(c(0.5), r);
  Value hxs = mul(r, hfx);
  Value p = c(-2.01099218183624371326e-07);
  for (double q : {4.00821782732936239552e-06, -7.93650757867487942473e-05,
                   1.58730158725481460165e-03, -3.33333333333331316428e-02})
    p = fma(hxs, p, c(q));
  Value r1 = fma(hxs, p, c(1.0));
  Value t = fma(neg(r1), hfx, c(3.0));
  Value e = mul(hxs, arith::DivFOp::create(b, sub(r1, t),
                                           fma(neg(r), t, c(6.0))));
  Value ki = arith::FPToSIOp::create(b, i64, kf);
  // r 2^k: k added to the exponent field
  auto scale = [&](Value y) {
    return dbl(arith::AddIOp::create(b, bits(y),
                                     arith::ShLIOp::create(b, ki, ci(52))));
  };
  // fdlibm's reconstructions are 2^k ((1 - 2^-k) - (e2 - r)), e2 = (r (e -
  // c) - c) - hxs (for k 0, -1, 1 and below 20: with 64-bit fields 1 - 2^-k
  // is exact up to k = 52, and the same for k = 0 and from -58, whose
  // rounding there is the result's), 2^k (1 - ((e2 + 2^-k) - r)) (k from
  // 53; fdlibm's (r - (e2 + 2^-k)) + 1, the same) and 2 ((r + 1/2) - e2)
  // (k = 1 and r below -1/4): one form, (r + f) + g, its operands selected
  // (2^-k of k up to 1000: a 2^-k of a k above 56 does not change the sum)
  Value e2 = sub(fma(r, sub(e, cr), neg(cr)), hxs);
  Value big = icmp(arith::CmpIPredicate::sge, ki, ci(53));
  Value pkBits = arith::ShLIOp::create(
      b,
      arith::SubIOp::create(b, ci(1023),
                            arith::MinSIOp::create(b, ki, ci(1000))),
      ci(52));
  Value zeroBits = ci(0);
  Value e2s = add(e2, dbl(sel(big, pkBits, zeroBits)));
  Value tk = sub(c(1.0), dbl(sel(big, zeroBits, pkBits)));
  Value oneLow = arith::AndIOp::create(
      b, icmp(arith::CmpIPredicate::eq, ki, ci(1)),
      fcmp(arith::CmpFPredicate::OLT, r, c(-0.25)));
  Value f = sel(oneLow, c(0.5), neg(e2s));
  Value g = sel(oneLow, neg(e2), tk);
  Value res = scale(add(add(r, f), g));
  res = sel(fcmp(arith::CmpFPredicate::OLT, math::AbsFOp::create(b, x),
                 c(0x1p-54)),
            x, res);
  res = sel(over, c(std::numeric_limits<double>::infinity()), res);
  return sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, res);
}

/// log1p of an f64 x: fdlibm's s_log1p.c in f64, i64 and f32 operations
/// (L15-E818). Outside fdlibm's k = 0 range (x from -0.2929 to 0.4142, where
/// f = x itself) u = 1 + x is reduced as log's argument, 2^k (1 + f), and
/// its rounding error c ((1 + x) - u, exactly: 1 - (u - x) from u = 2, else
/// x - (u - 1); none from 2^53) corrects the result by c / u, at most 2^-53
/// of it: that quotient in f32 with the hardware's reciprocal, to 2^-21, is
/// enough (fdlibm divides in double). Then log's s = f / (2 + f), R its
/// polynomial, k ln2hi - ((f^2/2 - (s (f^2/2 + R) + (k ln2lo + c/u))) - f).
/// -1 gives -inf, below -1 and NaN the default NaN (as log of 1 + x did),
/// +inf itself, |x| below 2^-54 x (and -0 -0). Not log(1 + x) with Kahan's
/// log(u) x / (u - 1) below 1/2: one division of doubles fewer.
static Value log1pF64(ImplicitLocOpBuilder &b, Value x) {
  Type f64 = b.getF64Type(), i64 = b.getI64Type(), f32 = b.getF32Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto ci = [&](int64_t v) { return arith::ConstantIntOp::create(b, i64, v); };
  auto add = [&](Value u, Value v) { return arith::AddFOp::create(b, u, v); };
  auto sub = [&](Value u, Value v) { return arith::SubFOp::create(b, u, v); };
  auto mul = [&](Value u, Value v) { return arith::MulFOp::create(b, u, v); };
  auto fma = [&](Value u, Value v, Value w) -> Value {
    return f64Fma(b, u, v, w);
  };
  auto sel = [&](Value cnd, Value u, Value v) {
    return arith::SelectOp::create(b, cnd, u, v);
  };
  auto fcmp = [&](arith::CmpFPredicate p, Value u, Value v) {
    return arith::CmpFOp::create(b, p, u, v);
  };
  auto icmp = [&](arith::CmpIPredicate p, Value u, Value v) {
    return arith::CmpIOp::create(b, p, u, v);
  };
  Value u = add(c(1.0), x);
  Value bu = arith::BitcastOp::create(b, i64, u);
  Value hu = arith::ShRUIOp::create(b, bu, ci(32));
  Value k = arith::SubIOp::create(b, arith::ShRUIOp::create(b, hu, ci(20)),
                                  ci(1023));
  Value cr = sel(icmp(arith::CmpIPredicate::sgt, k, ci(0)),
                 sub(c(1.0), sub(u, x)), sub(x, sub(u, c(1.0))));
  cr = sel(icmp(arith::CmpIPredicate::sge, k, ci(53)), c(0.0), cr);
  auto q = arith::DivFOp::create(b, arith::TruncFOp::create(b, f32, cr),
                                 arith::TruncFOp::create(b, f32, u));
  q.setFastmath(arith::FastMathFlags::afn);
  Value cu = arith::ExtFOp::create(b, f64, q.getResult());
  hu = arith::AndIOp::create(b, hu, ci(0xfffff));
  Value up = icmp(arith::CmpIPredicate::uge, hu, ci(0x6a09e));
  Value newHi = arith::OrIOp::create(
      b, hu, sel(up, ci(0x3fe00000), ci(0x3ff00000)));
  k = arith::AddIOp::create(b, k, sel(up, ci(1), ci(0)));
  Value f = sub(arith::BitcastOp::create(
                    b, f64,
                    arith::OrIOp::create(
                        b, arith::ShLIOp::create(b, newHi, ci(32)),
                        arith::AndIOp::create(b, bu, ci(0xffffffff)))),
                c(1.0));
  // fdlibm's k = 0 range: f = x (high words below 0x3fda827a and, negative,
  // at most 0x3fd2bec3)
  Value near = arith::AndIOp::create(
      b, fcmp(arith::CmpFPredicate::OLT, x, c(0x1.a827ap-2)),
      fcmp(arith::CmpFPredicate::OGT, x, c(-0x1.2bec4p-2)));
  f = sel(near, x, f);
  k = sel(near, ci(0), k);
  cu = sel(near, c(0.0), cu);
  Value sv = arith::DivFOp::create(b, f, add(c(2.0), f));
  Value z = mul(sv, sv), w = mul(z, z);
  Value t1 = mul(w, fma(w, fma(w, c(1.531383769920937332e-01),
                               c(2.222219843214978396e-01)),
                        c(3.999999999940941908e-01)));
  Value t2 = mul(z, fma(w, fma(w, fma(w, c(1.479819860511658591e-01),
                                      c(1.818357216161805012e-01)),
                               c(2.857142874366239149e-01)),
                        c(6.666666666666735130e-01)));
  Value r = add(t2, t1);
  Value dk = arith::SIToFPOp::create(b, f64, k);
  Value hfsq = mul(mul(c(0.5), f), f);
  Value res = sub(mul(dk, c(6.93147180369123816490e-01)),
                  sub(sub(hfsq, add(mul(sv, add(hfsq, r)),
                                    add(mul(dk, c(1.90821492927058770002e-10)),
                                        cu))),
                      f));
  Value inf = c(std::numeric_limits<double>::infinity());
  res = sel(fcmp(arith::CmpFPredicate::OLT, math::AbsFOp::create(b, x),
                 c(0x1p-54)),
            x, res);
  res = sel(fcmp(arith::CmpFPredicate::OEQ, x, inf), inf, res);
  res = sel(fcmp(arith::CmpFPredicate::OEQ, x, c(-1.0)),
            c(-std::numeric_limits<double>::infinity()), res);
  return sel(fcmp(arith::CmpFPredicate::ULT, x, c(-1.0)),
             c(std::numeric_limits<double>::quiet_NaN()), res);
}

/// The f32 operations of a half function's expansion that may take the
/// hardware's approximations (fastmath afn: exp, log, square root, division
/// by the reciprocal).
enum HalfApprox : unsigned {
  HalfApproxNone = 0,
  HalfApproxExpLog = 1,
  HalfApproxSqrt = 2,
  HalfApproxDiv = 4,
};

/// Whether convert-gpu-to-air's approximate-half-functions is on (set by the
/// pass for its patterns).
static thread_local bool approximateHalfFunctions = true;
/// Whether convert-gpu-to-air's expm1-f64-direct is on (set as above).
static thread_local bool expm1F64Direct = true;
/// Whether convert-gpu-to-air's log1p-f64-direct is on (set as above).
static thread_local bool log1pF64Direct = true;

/// For a half result computed in f32 from \p x: fastmath afn on the f32
/// operations of the kinds in \p kinds computing \p r, the kinds for which
/// every half input's result was checked to be the one the precise
/// operations give, on the functional model (L15-E330).
static void approximateForHalf(Value r, Value x, unsigned kinds) {
  if (!approximateHalfFunctions)
    return;
  SmallVector<Value> work{r};
  DenseSet<Operation *> seen;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    Operation *op = v.getDefiningOp();
    if (v == x || !op || !seen.insert(op).second)
      continue;
    auto fm = dyn_cast<arith::ArithFastMathInterface>(op);
    unsigned kind =
        isa<math::LogOp, math::ExpOp, math::Log2Op, math::Exp2Op>(op)
            ? HalfApproxExpLog
        : isa<math::SqrtOp>(op)  ? HalfApproxSqrt
        : isa<arith::DivFOp>(op) ? HalfApproxDiv
                                 : HalfApproxNone;
    if (fm && (kind & kinds))
      op->setAttr(fm.getFastMathAttrName(),
                  arith::FastMathFlagsAttr::get(
                      op->getContext(), fm.getFastMathFlagsAttr().getValue() |
                                            arith::FastMathFlags::afn));
    for (Value o : op->getOperands())
      work.push_back(o);
  }
}

template <typename OpTy>
static LogicalResult rewriteAccurateM1(OpTy op, PatternRewriter &rewriter) {
  auto type = dyn_cast<FloatType>(op.getType());
  if (!type || !(type.isF16() || type.isBF16() || type.isF32() || type.isF64()))
    return failure();
  Location loc = op.getLoc();
  // f16 and bf16 in f32; f64 in f64 (its exp and log: rewriteExpF64 /
  // rewriteLogF64)
  Type f32 = type.isF64() ? Type(type) : Type(rewriter.getF32Type());
  Value x = op.getOperand();
  if (!type.isF32() && !type.isF64())
    x = arith::ExtFOp::create(rewriter, loc, f32, x);
  auto c = [&](float v) {
    return arith::ConstantOp::create(rewriter, loc, f32,
                                     rewriter.getFloatAttr(f32, v));
  };
  auto eq = [&](Value a, Value b) {
    return arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OEQ, a, b);
  };
  auto sel = [&](Value cnd, Value a, Value b) {
    return arith::SelectOp::create(rewriter, loc, cnd, a, b);
  };
  Value one = c(1.0f);
  Value small = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLT,
      math::AbsFOp::create(rewriter, loc, x), c(0.5f));
  Value r;
  if constexpr (std::is_same_v<OpTy, math::ExpM1Op>) {
    (void)small;
    if (type.isF64() && expm1F64Direct) {
      ImplicitLocOpBuilder ib(loc, rewriter);
      r = expm1F64(ib, x);
    } else {
      r = accurateExpM1(rewriter, loc, x,
                        math::ExpOp::create(rewriter, loc, x));
    }
  } else if (type.isF64() && log1pF64Direct) {
    (void)small;
    ImplicitLocOpBuilder ib(loc, rewriter);
    r = log1pF64(ib, x);
  } else {
    Value u = arith::AddFOp::create(rewriter, loc, one, x);
    Value lu = math::LogOp::create(rewriter, loc, u);
    Value q = arith::DivFOp::create(
        rewriter, loc, arith::MulFOp::create(rewriter, loc, lu, x),
        arith::SubFOp::create(rewriter, loc, u, one));
    r = sel(small, sel(eq(u, one), x, q), lu);
  }
  // an f16 or bf16 result: the division by the reciprocal (exp and log
  // precise: with the hardware's, expm1 and log1p were up to 1.12 ulp off)
  if (type.isF16() || type.isBF16())
    approximateForHalf(r, x, HalfApproxDiv);
  if (!type.isF32() && !type.isF64())
    r = arith::TruncFOp::create(rewriter, loc, type, r);
  rewriter.replaceOp(op, r);
  return success();
}

/// log1p of an f32 x, accurate (rewriteAccurateM1's form). With \p far and
/// \p near, the one log is of 1 + x where near holds and of far elsewhere,
/// and is returned in \p farLog: a caller that selects this log1p where
/// near holds and something of log(far) elsewhere takes one log, not two,
/// each lane's selected value the same (L15-E308).
static Value accurateLog1pF32(PatternRewriter &rewriter, Location loc, Value x,
                              Value far = {}, Value near = {},
                              Value *farLog = nullptr) {
  Type f32 = rewriter.getF32Type();
  auto c = [&](float v) {
    return arith::ConstantOp::create(rewriter, loc, f32,
                                     rewriter.getF32FloatAttr(v));
  };
  Value one = c(1.0f);
  Value u = arith::AddFOp::create(rewriter, loc, one, x);
  Value lu = math::LogOp::create(
      rewriter, loc,
      far ? arith::SelectOp::create(rewriter, loc, near, u, far).getResult()
          : u);
  if (farLog)
    *farLog = lu;
  Value q = arith::DivFOp::create(
      rewriter, loc, arith::MulFOp::create(rewriter, loc, lu, x),
      arith::SubFOp::create(rewriter, loc, u, one));
  Value small = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLT,
      math::AbsFOp::create(rewriter, loc, x), c(0.5f));
  Value isOne =
      arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OEQ, u, one);
  return arith::SelectOp::create(
      rewriter, loc, small, arith::SelectOp::create(rewriter, loc, isOne, x, q),
      lu);
}

/// asinh, acosh, atanh and cbrt, which MathToLLVM does not lower (and whose
/// upstream expansions, log(x + sqrt(x^2 +- 1)) and log((1 + x) / (1 - x)) / 2,
/// cancel for small x and overflow for large), in f32 (f16 and bf16 rounded
/// once from it), with a = |x| and the sign of x copied back:
///   asinh = log1p(a + a^2 / (1 + sqrt(1 + a^2))), from 2^12 log(a) + ln 2;
///   acosh = log1p(t + sqrt(2 t + t^2)), t = x - 1, from 2^12 log(x) + ln 2
///           (NaN below 1);
///   (each one log, of the argument the range picks);
///   atanh = log1p(2 a / (1 - a)) / 2 (inf at 1, NaN beyond);
///   cbrt  = exp2(log2(a) / 3) and a Newton step y - (y^3 - a) / (3 y^2)
///           (inf and NaN themselves, a signed zero below 2^-126: a
///           denormal flushed).
template <typename OpTy>
static LogicalResult rewriteInverseHyperbolic(OpTy op,
                                              PatternRewriter &rewriter) {
  auto type = dyn_cast<FloatType>(op.getType());
  if (!type || !(type.isF16() || type.isBF16() || type.isF32()))
    return failure();
  Location loc = op.getLoc();
  Type f32 = rewriter.getF32Type();
  Value x = op.getOperand();
  if (!type.isF32())
    x = arith::ExtFOp::create(rewriter, loc, f32, x);
  auto c = [&](float v) {
    return arith::ConstantOp::create(rewriter, loc, f32,
                                     rewriter.getF32FloatAttr(v));
  };
  auto cmp = [&](arith::CmpFPredicate p, Value a, Value b) {
    return arith::CmpFOp::create(rewriter, loc, p, a, b);
  };
  auto sel = [&](Value cnd, Value a, Value b) {
    return arith::SelectOp::create(rewriter, loc, cnd, a, b);
  };
  auto add = [&](Value a, Value b) { return arith::AddFOp::create(rewriter, loc, a, b); };
  auto sub = [&](Value a, Value b) { return arith::SubFOp::create(rewriter, loc, a, b); };
  auto mul = [&](Value a, Value b) { return arith::MulFOp::create(rewriter, loc, a, b); };
  auto div = [&](Value a, Value b) { return arith::DivFOp::create(rewriter, loc, a, b); };
  Value a = math::AbsFOp::create(rewriter, loc, x);
  Value one = c(1.0f), ln2 = c(0.693147180559945309f);
  Value r;
  if constexpr (std::is_same_v<OpTy, math::AsinhOp>) {
    Value a2 = mul(a, a);
    Value s = math::SqrtOp::create(rewriter, loc, add(one, a2));
    Value near = cmp(arith::CmpFPredicate::OLT, a, c(4096.0f));
    Value logA;
    Value nearV = accurateLog1pF32(rewriter, loc, add(a, div(a2, add(one, s))),
                                   a, near, &logA);
    r = sel(near, nearV, add(logA, ln2));
    r = math::CopySignOp::create(rewriter, loc, r, x);
  } else if constexpr (std::is_same_v<OpTy, math::AcoshOp>) {
    Value t = sub(x, one);
    Value s = math::SqrtOp::create(rewriter, loc, add(add(t, t), mul(t, t)));
    Value near = cmp(arith::CmpFPredicate::OLT, x, c(4096.0f));
    Value logX;
    Value nearV =
        accurateLog1pF32(rewriter, loc, add(t, s), x, near, &logX);
    r = sel(near, nearV, add(logX, ln2));
    // below 1 a NaN (t + sqrt(2 t + t^2) is -1 for large negative x)
    r = sel(cmp(arith::CmpFPredicate::OLT, x, one), c(NAN), r);
  } else if constexpr (std::is_same_v<OpTy, math::AtanhOp>) {
    Value q = div(add(a, a), sub(one, a));
    r = mul(c(0.5f), accurateLog1pF32(rewriter, loc, q));
    r = math::CopySignOp::create(rewriter, loc, r, x);
  } else {
    Value y = math::Exp2Op::create(
        rewriter, loc,
        mul(math::Log2Op::create(rewriter, loc, a), c(1.0f / 3.0f)));
    Value y2 = mul(y, y);
    y = sub(y, div(sub(mul(y2, y), a), mul(c(3.0f), y2)));
    // inf and NaN themselves; zero (and a flushed denormal) a signed zero
    r = sel(cmp(arith::CmpFPredicate::UEQ, a, c(INFINITY)), x,
            math::CopySignOp::create(rewriter, loc, y, x));
    r = sel(cmp(arith::CmpFPredicate::OLT, a, c(0x1p-126f)),
            math::CopySignOp::create(rewriter, loc, c(0.0f), x), r);
  }
  // a half result: cbrt all approximate, acosh and atanh the divisions and
  // square roots, asinh the square root (each half input's result as before;
  // asinh's division by the reciprocal changed 4, 2 of them further off, and
  // the hardware's log changed asinh's and atanh's small results)
  if (type.isF16() || type.isBF16()) {
    constexpr bool isCbrt = std::is_same_v<OpTy, math::CbrtOp>;
    constexpr bool isAsinh = std::is_same_v<OpTy, math::AsinhOp>;
    constexpr bool isAcosh = std::is_same_v<OpTy, math::AcoshOp>;
    unsigned all = HalfApproxExpLog | HalfApproxSqrt | HalfApproxDiv;
    unsigned kinds =
        type.isF16()
            ? (isCbrt    ? all
               : isAsinh ? unsigned(HalfApproxSqrt)
                         : HalfApproxSqrt | HalfApproxDiv)
            // bf16 (L15-E332): acosh all; atanh none (its division by the
            // reciprocal gave 0 for x beyond 2^126, not a NaN); the others
            // the divisions and square roots (cbrt's hardware log and exp
            // rounded 2 inputs the other way)
            : (isAcosh                                ? all
               : std::is_same_v<OpTy, math::AtanhOp> ? HalfApproxNone
                                                     : HalfApproxSqrt |
                                                           HalfApproxDiv);
    approximateForHalf(r, x, kinds);
  }
  if (!type.isF32())
    r = arith::TruncFOp::create(rewriter, loc, type, r);
  rewriter.replaceOp(op, r);
  return success();
}

/// atan of an f64 a >= 0 (or NaN): fdlibm's s_atan.c, branch-free. Below
/// 0.4375 a itself is the polynomial's argument; above, (2a - 1) / (2 + a),
/// (a - 1) / (a + 1), (a - 1.5) / (1 + 1.5a) or -1 / a by the range (one
/// division, its operands selected), with atan's value at 0.5, 1, 1.5 or
/// inf in two parts; the reconstruction hi - ((r S - lo) - r) is fdlibm's
/// r - r S where hi and lo are 0 (L15-E821).
static Value atanF64(ImplicitLocOpBuilder &b, Value a) {
  Type f64 = b.getF64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto add = [&](Value u, Value v) { return arith::AddFOp::create(b, u, v); };
  auto sub = [&](Value u, Value v) { return arith::SubFOp::create(b, u, v); };
  auto mul = [&](Value u, Value v) { return arith::MulFOp::create(b, u, v); };
  auto fma = [&](Value u, Value v, Value w) -> Value {
    return f64Fma(b, u, v, w);
  };
  auto sel = [&](Value cnd, Value u, Value v) {
    return arith::SelectOp::create(b, cnd, u, v);
  };
  auto lt = [&](Value u, double v) {
    return arith::CmpFOp::create(b, arith::CmpFPredicate::OLT, u, c(v));
  };
  Value r0 = lt(a, 0.4375), r1 = lt(a, 0.6875), r2 = lt(a, 1.1875),
        r3 = lt(a, 2.4375);
  auto pick = [&](double v0, double v1, double v2, double v3) {
    return sel(r1, c(v0), sel(r2, c(v1), sel(r3, c(v2), c(v3))));
  };
  Value num = sel(r1, sub(add(a, a), c(1.0)),
                  sel(r2, sub(a, c(1.0)),
                      sel(r3, sub(a, c(1.5)), c(-1.0))));
  Value den = sel(r1, add(c(2.0), a),
                  sel(r2, add(a, c(1.0)),
                      sel(r3, add(c(1.0), mul(c(1.5), a)), a)));
  Value x = sel(r0, a, arith::DivFOp::create(b, num, den));
  Value hi = sel(r0, c(0.0),
                 pick(4.63647609000806093515e-01, 7.85398163397448278999e-01,
                      9.82793723247329054082e-01, 1.57079632679489655800e+00));
  Value lo = sel(r0, c(0.0),
                 pick(2.26987774529616870924e-17, 3.06161699786838301793e-17,
                      1.39033110312309984516e-17, 6.12323399573676603587e-17));
  Value z = mul(x, x), w = mul(z, z);
  Value s1 = c(1.62858201153657823623e-02);
  for (double q : {4.97687799461593236017e-02, 6.66107313738753120669e-02,
                   9.09088713343650656196e-02, 1.42857142725034663711e-01,
                   3.33333333333329318027e-01})
    s1 = fma(w, s1, c(q));
  s1 = mul(z, s1);
  Value s2 = c(-3.65315727442169155270e-02);
  for (double q : {-5.83357013379057348645e-02, -7.69187620504482999495e-02,
                   -1.11111104054623557880e-01, -1.99999999998764832476e-01})
    s2 = fma(w, s2, c(q));
  s2 = mul(w, s2);
  return sub(hi, sub(sub(mul(x, add(s1, s2)), lo), x));
}

/// asin's and acos's rational R(t) = p(t) / q(t) of fdlibm (e_asin.c's pS
/// and qS): its dividend and divisor.
static std::pair<Value, Value> asinRationalF64(ImplicitLocOpBuilder &b,
                                               Value t) {
  Type f64 = b.getF64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto fma = [&](Value u, Value v, Value w) -> Value {
    return f64Fma(b, u, v, w);
  };
  Value p = c(3.47933107596021167570e-05);
  for (double v : {7.91534994289814532176e-04, -4.00555345006794114027e-02,
                   2.01212532134862925881e-01, -3.25565818622400915405e-01,
                   1.66666666666666657415e-01})
    p = fma(t, p, c(v));
  p = arith::MulFOp::create(b, t, p);
  Value q = c(7.70381505559019352791e-02);
  for (double v : {-6.88283971605453293030e-01, 2.02094576023350569471e+00,
                   -2.40339491173441421878e+00, 1.0})
    q = fma(t, q, c(v));
  return {p, q};
}

/// atan, atan2, asin and acos of f64 (L15-E821; before, llvm.atan.f64 and
/// the others had no emulation and the kernel did not compile): fdlibm's
/// s_atan.c, e_atan2.c, e_asin.c and e_acos.c, branch-free:
///   atan  = atanF64(|x|) with x's sign; x for a NaN;
///   atan2 = atan(|y / x|) and its quadrant (pi - (z - pi_lo), (z - pi_lo)
///           - pi), y's sign for x > 0; the zeros', infinities' and the
///           exponents' 60 apart cases as e_atan2.c; x + y for a NaN;
///   asin  = x + x R(x^2) below 1/2; from 0.975 pi/2 - (2 (s + s R) -
///           pi/2_lo), s = sqrt((1 - |x|) / 2); between, fdlibm's form with
///           s's high word and its correction c (a second division); x's
///           sign; +-1 +-pi/2, NaN past 1;
///   acos  = pi/2 - (x - (pi/2_lo - x R(x^2))) below 1/2; below -1/2
///           pi - 2 (s + (R s - pi/2_lo)); above 1/2 2 (df + (R s + c)), df
///           s's high word; 1 0, -1 pi, NaN past 1.
/// One R division each (its argument selected), and sqrt of the far one.
template <typename OpTy>
static LogicalResult rewriteInverseTrigF64(OpTy op, PatternRewriter &rewriter) {
  if (!op.getType().isF64())
    return failure();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto add = [&](Value u, Value v) { return arith::AddFOp::create(b, u, v); };
  auto sub = [&](Value u, Value v) { return arith::SubFOp::create(b, u, v); };
  auto mul = [&](Value u, Value v) { return arith::MulFOp::create(b, u, v); };
  auto div = [&](Value u, Value v) { return arith::DivFOp::create(b, u, v); };
  auto sel = [&](Value cnd, Value u, Value v) {
    return arith::SelectOp::create(b, cnd, u, v);
  };
  auto fcmp = [&](arith::CmpFPredicate p, Value u, Value v) {
    return arith::CmpFOp::create(b, p, u, v);
  };
  auto abs = [&](Value v) { return math::AbsFOp::create(b, v); };
  auto copysign = [&](Value m, Value s) {
    return math::CopySignOp::create(b, m, s);
  };
  // s with its low word cleared (fdlibm's SET_LOW_WORD(df, 0))
  auto highWord = [&](Value v) {
    return arith::BitcastOp::create(
        b, f64,
        arith::AndIOp::create(
            b, arith::BitcastOp::create(b, i64, v),
            arith::ConstantIntOp::create(b, i64,
                                         int64_t(0xffffffff00000000ULL))));
  };
  const double pio2Hi = 1.57079632679489655800e+00,
               pio2Lo = 6.12323399573676603587e-17,
               pio4Hi = 7.85398163397448278999e-01,
               pi = 3.1415926535897931160e+00, piLo = 1.2246467991473531772e-16;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  Value r;
  if constexpr (std::is_same_v<OpTy, math::AtanOp>) {
    Value x = op.getOperand();
    r = copysign(atanF64(b, abs(x)), x);
    r = sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, r);
  } else if constexpr (std::is_same_v<OpTy, math::Atan2Op>) {
    Value y = op.getLhs(), x = op.getRhs();
    Value ay = abs(y), ax = abs(x);
    Value yNeg = arith::CmpIOp::create(
        b, arith::CmpIPredicate::slt, arith::BitcastOp::create(b, i64, y),
        arith::ConstantIntOp::create(b, i64, 0));
    Value xNeg = arith::CmpIOp::create(
        b, arith::CmpIPredicate::slt, arith::BitcastOp::create(b, i64, x),
        arith::ConstantIntOp::create(b, i64, 0));
    // the exponents' difference (fdlibm's k = (iy - ix) >> 20)
    auto field = [&](Value v) {
      return arith::ShRUIOp::create(b, arith::BitcastOp::create(b, i64, v),
                                    arith::ConstantIntOp::create(b, i64, 52));
    };
    Value k = arith::SubIOp::create(b, field(ay), field(ax));
    Value far = arith::CmpIOp::create(b, arith::CmpIPredicate::sgt, k,
                                      arith::ConstantIntOp::create(b, i64, 60));
    Value tinyNeg = arith::AndIOp::create(
        b, xNeg,
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, k,
                              arith::ConstantIntOp::create(b, i64, -60)));
    Value z = atanF64(b, abs(div(y, x)));
    z = sel(far, c(pio2Hi + 0.5 * piLo), sel(tinyNeg, c(0.0), z));
    // quadrants (far: as for x > 0)
    Value left = arith::AndIOp::create(
        b, xNeg, arith::XOrIOp::create(b, far, arith::ConstantIntOp::create(
                                                   b, b.getI1Type(), 1)));
    Value q = sel(left, sub(c(pi), sub(z, c(piLo))), z);
    r = sel(yNeg, arith::NegFOp::create(b, q), q);
    // y zero: +-0 for x > 0 (and +0), +-pi for x < 0 (and -0)
    Value yZero = fcmp(arith::CmpFPredicate::OEQ, y, c(0.0));
    r = sel(yZero, sel(xNeg, copysign(c(pi), y), y), r);
    // x zero (y not): +-pi/2
    Value xZero = fcmp(arith::CmpFPredicate::OEQ, x, c(0.0));
    r = sel(arith::AndIOp::create(
                b, xZero,
                arith::XOrIOp::create(
                    b, yZero, arith::ConstantIntOp::create(b, b.getI1Type(), 1))),
            copysign(c(pio2Hi), y), r);
    // infinities
    Value xInf = fcmp(arith::CmpFPredicate::OEQ, ax, c(inf));
    Value yInf = fcmp(arith::CmpFPredicate::OEQ, ay, c(inf));
    Value bothInf = arith::AndIOp::create(b, xInf, yInf);
    r = sel(yInf, copysign(sel(xInf, sel(xNeg, c(3.0 * pio4Hi), c(pio4Hi)),
                               c(pio2Hi)),
                           y),
            r);
    r = sel(arith::AndIOp::create(
                b, xInf,
                arith::XOrIOp::create(
                    b, bothInf,
                    arith::ConstantIntOp::create(b, b.getI1Type(), 1))),
            copysign(sel(xNeg, c(pi), c(0.0)), y), r);
    // (fdlibm's x = 1 case, atan(y), is the general path's atan(|y / 1|)
    // with y's sign: not a second atan)
    r = sel(arith::OrIOp::create(b, fcmp(arith::CmpFPredicate::UNO, x, x),
                                 fcmp(arith::CmpFPredicate::UNO, y, y)),
            add(x, y), r);
  } else {
    Value x = op.getOperand();
    Value a = abs(x);
    Value small = fcmp(arith::CmpFPredicate::OLT, a, c(0.5));
    // the rational's argument: x^2 below 1/2, (1 - |x|) / 2 from it
    Value t = sel(small, mul(x, x), mul(c(0.5), sub(c(1.0), a)));
    auto [p, q] = asinRationalF64(b, t);
    Value rr = div(p, q);
    Value s = math::SqrtOp::create(b, t);
    Value df = highWord(s);
    // (t - df^2) / (s + df): what s lacks below its high word
    Value cc = div(sub(t, mul(df, df)), add(s, df));
    if constexpr (std::is_same_v<OpTy, math::AsinOp>) {
      Value nearOne = fcmp(arith::CmpFPredicate::OGE, a, c(0x1.f3333p-1));
      Value fromOne =
          sub(c(pio2Hi), sub(mul(c(2.0), add(s, mul(s, rr))), c(pio2Lo)));
      Value pp = sub(mul(mul(c(2.0), s), rr), sub(c(pio2Lo), mul(c(2.0), cc)));
      Value qq = sub(c(pio4Hi), mul(c(2.0), df));
      Value between = sub(c(pio4Hi), sub(pp, qq));
      r = sel(nearOne, fromOne, between);
      r = sel(small, add(x, mul(x, rr)), copysign(r, x));
      r = sel(fcmp(arith::CmpFPredicate::OEQ, a, c(1.0)),
              add(mul(x, c(pio2Hi)), mul(x, c(pio2Lo))), r);
      r = sel(fcmp(arith::CmpFPredicate::OLT, a, c(0x1p-27)), x, r);
      r = sel(fcmp(arith::CmpFPredicate::UGT, a, c(1.0)), c(nan), r);
      r = sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, r);
    } else {
      static_assert(std::is_same_v<OpTy, math::AcosOp>);
      Value nearV = sub(c(pio2Hi), sub(x, sub(c(pio2Lo), mul(x, rr))));
      Value negV =
          sub(c(pi), mul(c(2.0), add(s, sub(mul(rr, s), c(pio2Lo)))));
      Value posV = mul(c(2.0), add(df, add(mul(rr, s), cc)));
      r = sel(small, nearV,
              sel(fcmp(arith::CmpFPredicate::OLT, x, c(0.0)), negV, posV));
      r = sel(fcmp(arith::CmpFPredicate::OEQ, x, c(1.0)), c(0.0), r);
      r = sel(fcmp(arith::CmpFPredicate::OEQ, x, c(-1.0)),
              c(2.0 * pio2Hi + 2.0 * pio2Lo), r);
      r = sel(fcmp(arith::CmpFPredicate::UGT, a, c(1.0)), c(nan), r);
      r = sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, r);
    }
  }
  rewriter.replaceOp(op, r);
  return success();
}

/// asinh, acosh, atanh and cbrt of f64 (L15-E820; before, none lowered):
/// fdlibm's forms, from math.log1p and math.sqrt of f64 (each rewritten or
/// emulated in turn), a = |x| and x's sign copied back, x itself for a NaN:
///   asinh = log1p(a + a^2 / (1 + sqrt(1 + a^2))), from 2^28 log1p(a - 1) +
///           ln 2 (log(a) + ln 2: a - 1 exact to 2^53, past it its rounding
///           far below an ulp); x below 2^-28;
///   acosh = log1p(t + sqrt(2 t + t^2)), t = x - 1, from 2^28 log1p(t) + ln 2;
///           NaN below 1;
///   (each one log1p, of the argument the range picks);
///   atanh = log1p(2a + 2a a / (1 - a)) / 2 below 1/2, log1p(2a / (1 - a)) / 2
///           from it (one division, its dividend selected); inf at 1, NaN
///           beyond; x below 2^-28;
///   cbrt  = s_cbrt.c: a quotient of the high word by 3 (denormals scaled by
///           2^54 first) for 5 bits, a polynomial to 23 bits, rounded away
///           from zero to 23 bits, and a Newton step to 53 (within 0.667
///           ulp); zeros, infinities and NaNs themselves.
template <typename OpTy>
static LogicalResult rewriteInverseHyperbolicF64(OpTy op,
                                                 PatternRewriter &rewriter) {
  if (!op.getType().isF64())
    return failure();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto ci = [&](int64_t v) { return arith::ConstantIntOp::create(b, i64, v); };
  auto add = [&](Value u, Value v) { return arith::AddFOp::create(b, u, v); };
  auto sub = [&](Value u, Value v) { return arith::SubFOp::create(b, u, v); };
  auto mul = [&](Value u, Value v) { return arith::MulFOp::create(b, u, v); };
  auto div = [&](Value u, Value v) { return arith::DivFOp::create(b, u, v); };
  auto sel = [&](Value cnd, Value u, Value v) {
    return arith::SelectOp::create(b, cnd, u, v);
  };
  auto fcmp = [&](arith::CmpFPredicate p, Value u, Value v) {
    return arith::CmpFOp::create(b, p, u, v);
  };
  auto log1p = [&](Value v) { return math::Log1pOp::create(b, v); };
  auto sqrt = [&](Value v) { return math::SqrtOp::create(b, v); };
  Value x = op.getOperand();
  Value a = math::AbsFOp::create(b, x);
  const double ln2 = 6.93147180559945286227e-01;
  Value r;
  if constexpr (std::is_same_v<OpTy, math::AsinhOp>) {
    Value near = fcmp(arith::CmpFPredicate::OLT, a, c(0x1p28));
    Value a2 = mul(a, a);
    Value arg = sel(near, add(a, div(a2, add(c(1.0), sqrt(add(c(1.0), a2))))),
                    sub(a, c(1.0)));
    r = add(log1p(arg), sel(near, c(0.0), c(ln2)));
    r = math::CopySignOp::create(b, r, x);
    r = sel(fcmp(arith::CmpFPredicate::OLT, a, c(0x1p-28)), x, r);
  } else if constexpr (std::is_same_v<OpTy, math::AcoshOp>) {
    Value t = sub(x, c(1.0));
    Value near = fcmp(arith::CmpFPredicate::OLT, x, c(0x1p28));
    Value arg = sel(near, add(t, sqrt(add(add(t, t), mul(t, t)))), t);
    r = add(log1p(arg), sel(near, c(0.0), c(ln2)));
    r = sel(fcmp(arith::CmpFPredicate::OLT, x, c(1.0)),
            c(std::numeric_limits<double>::quiet_NaN()), r);
  } else if constexpr (std::is_same_v<OpTy, math::AtanhOp>) {
    Value t = add(a, a);
    Value small = fcmp(arith::CmpFPredicate::OLT, a, c(0.5));
    Value d = div(sel(small, mul(t, a), t), sub(c(1.0), a));
    r = mul(c(0.5), log1p(sel(small, add(t, d), d)));
    r = math::CopySignOp::create(b, r, x);
    r = sel(fcmp(arith::CmpFPredicate::OLT, a, c(0x1p-28)), x, r);
  } else {
    static_assert(std::is_same_v<OpTy, math::CbrtOp>);
    auto bits = [&](Value v) { return arith::BitcastOp::create(b, i64, v); };
    auto dbl = [&](Value v) { return arith::BitcastOp::create(b, f64, v); };
    Value bx = bits(x);
    Value sign = arith::AndIOp::create(b, bx, ci(INT64_MIN));
    // (a denormal scaled by 2^54, its high word's third less 18 << 20)
    Value tiny = fcmp(arith::CmpFPredicate::OLT, a, c(0x1p-1022));
    Value as = sel(tiny, mul(a, c(0x1p54)), a);
    Value hx = arith::ShRUIOp::create(b, bits(as), ci(32));
    Value hi = arith::AddIOp::create(
        b, arith::DivUIOp::create(b, hx, ci(3)),
        sel(tiny, ci(696219795), ci(715094163)));
    Value t = dbl(arith::OrIOp::create(b, sign,
                                       arith::ShLIOp::create(b, hi, ci(32))));
    Value rr = mul(mul(t, t), div(t, x));
    Value p1 = add(c(1.87595182427177009643),
                   mul(rr, add(c(-1.88497979543377169875),
                               mul(rr, c(1.621429720105354466140)))));
    Value p2 = mul(mul(mul(rr, rr), rr),
                   add(c(-0.758397934778766047437),
                       mul(rr, c(0.145996192886612446982))));
    t = mul(t, add(p1, p2));
    // rounded away from zero to 23 bits
    t = dbl(arith::AndIOp::create(
        b, arith::AddIOp::create(b, bits(t), ci(0x80000000LL)),
        ci(int64_t(0xffffffffc0000000ULL))));
    Value s2 = mul(t, t);
    Value q = div(x, s2);
    Value w = add(t, t);
    q = div(sub(q, t), add(w, q));
    r = add(t, mul(t, q));
    // zeros, infinities and NaNs themselves
    r = sel(fcmp(arith::CmpFPredicate::UEQ, a, c(0.0)), x, r);
    r = sel(fcmp(arith::CmpFPredicate::UEQ, a,
                 c(std::numeric_limits<double>::infinity())),
            x, r);
    rewriter.replaceOp(op, r);
    return success();
  }
  r = sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, r);
  rewriter.replaceOp(op, r);
  return success();
}

/// math.erfc of f32: upstream's approximation (PolynomialApproximation.cpp,
/// ErfcPolynomialApproximation: Shepherd and Laframboise's Chebyshev
/// approximation of (1 + 2x) exp(x^2) erfc(x), 2.65 ulps) but for its last
/// step: erfc = r exp(-a^2) with exp(-a^2) = e (1 + t), e = exp(-s), s = a a
/// rounded, t = s - a^2 exactly, was r e + (r e) t, whose correction r e t
/// is below 2^-126 from a ~ 8.7 and flushed (AGX's float arithmetic flushes
/// denormals): up to 64 ulps lost. Here e (r + r t), each intermediate
/// normal while the result is.
static LogicalResult rewriteErfc(math::ErfcOp op, PatternRewriter &rewriter) {
  if (!op.getType().isF32())
    return failure();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Type f32 = b.getF32Type();
  Value x = op.getOperand();
  auto c = [&](float v) {
    return arith::ConstantOp::create(b, f32, b.getF32FloatAttr(v));
  };
  auto fma = [&](Value p, Value q, Value r) {
    return math::FmaOp::create(b, p, q, r);
  };
  Value one = c(1.0f), two = c(2.0f);
  Value a = math::AbsFOp::create(b, x);
  Value r = arith::DivFOp::create(b, one, arith::AddFOp::create(b, a, two));
  Value q = fma(c(-4.0f), r, one);
  Value t = fma(arith::AddFOp::create(b, q, one), c(-2.0f), a);
  Value e = fma(arith::NegFOp::create(b, a), q, t);
  q = fma(r, e, q);
  Value p = c(-0x1.a4a000p-12f);
  for (float k : {-0x1.42a260p-10f, 0x1.585714p-10f, 0x1.1adcc4p-07f,
                  -0x1.081b82p-07f, -0x1.bc0b6ap-05f, 0x1.4ffc46p-03f,
                  -0x1.540840p-03f, -0x1.7bf616p-04f, 0x1.1ba03ap-02f})
    p = fma(p, q, c(k));
  Value d = fma(two, a, one);
  r = arith::DivFOp::create(b, one, d);
  q = fma(p, r, r);
  Value negA = arith::NegFOp::create(b, a);
  e = fma(fma(q, negA, c(0.5f)), two, arith::SubFOp::create(b, p, q));
  r = fma(e, r, q);
  Value s = arith::MulFOp::create(b, a, a);
  e = math::ExpOp::create(b, arith::NegFOp::create(b, s));
  t = fma(negA, a, s);
  r = arith::MulFOp::create(b, e, fma(r, t, r));
  auto cmp = [&](arith::CmpFPredicate pr, Value u, Value v) {
    return arith::CmpFOp::create(b, pr, u, v);
  };
  // NaN (x + x) for a NaN or infinite x; 0 beyond 10.0546875; 2 - r below 0
  // (from -inf: 2).
  r = arith::SelectOp::create(b, cmp(arith::CmpFPredicate::UGE, a, c(INFINITY)),
                              arith::AddFOp::create(b, x, x), r);
  r = arith::SelectOp::create(b, cmp(arith::CmpFPredicate::OGT, a,
                                     c(10.0546875f)),
                              c(0.0f), r);
  r = arith::SelectOp::create(b, cmp(arith::CmpFPredicate::OLT, x, c(0.0f)),
                              arith::SubFOp::create(b, two, r), r);
  // An f16 or bf16 erfc widened to this one (its argument extended, its
  // result only rounded back): the two divisions by the reciprocal, every
  // f16 and bf16 input's result the same (the exp approximate was not:
  // L15-E337).
  Type narrow;
  if (auto ext = x.getDefiningOp<arith::ExtFOp>())
    narrow = ext.getIn().getType();
  if (narrow && (narrow.isF16() || narrow.isBF16()) && !op->use_empty() &&
      llvm::all_of(op->getUsers(), [&](Operation *u) {
        auto t = dyn_cast<arith::TruncFOp>(u);
        return t && t.getType() == narrow;
      }))
    approximateForHalf(r, x, HalfApproxDiv);
  rewriter.replaceOp(op, r);
  return success();
}

/// math.exp of f64 (no double precision exp on the GPU; the AIR emulation
/// of doubles has arithmetic only): fdlibm's e_exp.c in double operations,
/// which the emulation computes. k = roundeven(x / ln 2), r = x - k ln2hi -
/// k ln2lo (k ln2hi exact: ln2hi has 11 trailing zero bits), c = r - r^2
/// P(r^2), exp(r) = 1 - ((lo - r c / (2 - c)) - hi), within 1 ulp; then
/// times 2^k as two powers of two (each normal; the product rounds once
/// where it is denormal). x is clamped to [-746, 710] first (by compares
/// that keep a NaN), beyond which the result is 0 or infinite anyway.
static LogicalResult rewriteExpF64(math::ExpOp op, PatternRewriter &rewriter) {
  if (!op.getType().isF64())
    return failure();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto ci = [&](int64_t v) {
    return arith::ConstantIntOp::create(b, i64, v);
  };
  auto add = [&](Value x, Value y) { return arith::AddFOp::create(b, x, y); };
  auto sub = [&](Value x, Value y) { return arith::SubFOp::create(b, x, y); };
  auto mul = [&](Value x, Value y) { return arith::MulFOp::create(b, x, y); };
  auto fma = [&](Value x, Value y, Value z) -> Value {
    return f64Fma(b, x, y, z);
  };
  auto sel = [&](Value cnd, Value x, Value y) {
    return arith::SelectOp::create(b, cnd, x, y);
  };
  Value x = op.getOperand();
  x = sel(arith::CmpFOp::create(b, arith::CmpFPredicate::OGT, x, c(710.0)),
          c(710.0), x);
  x = sel(arith::CmpFOp::create(b, arith::CmpFPredicate::OLT, x, c(-746.0)),
          c(-746.0), x);
  Value k = math::RoundEvenOp::create(b, mul(x, c(1.44269504088896338700e+00)));
  Value hi = sub(x, mul(k, c(6.93147180369123816490e-01)));
  Value lo = mul(k, c(1.90821492927058770002e-10));
  Value r = sub(hi, lo);
  Value t = mul(r, r);
  Value p = c(4.13813679705723846039e-08);
  for (double coef : {-1.65339022054652515390e-06, 6.61375632143793436117e-05,
                      -2.77777777770155933842e-03, 1.66666666666666019037e-01})
    p = fma(t, p, c(coef));
  Value cc = sub(r, mul(t, p));
  Value y = sub(c(1.0), sub(sub(lo, arith::DivFOp::create(
                                       b, mul(r, cc), sub(c(2.0), cc))),
                            hi));
  // 2^k = 2^k1 2^k2, k1 = k >> 1 (k within [-1077, 1025])
  Value ki = arith::FPToSIOp::create(b, i64, k);
  Value k1 = arith::ShRSIOp::create(b, ki, ci(1));
  Value k2 = arith::SubIOp::create(b, ki, k1);
  auto pow2 = [&](Value e) {
    return arith::BitcastOp::create(
        b, f64,
        arith::ShLIOp::create(b, arith::AddIOp::create(b, e, ci(1023)), ci(52)));
  };
  // y 2^k1 is exact and normal (y within [0.7, 1.42], k1 within [-538,
  // 512]): k1 added to y's exponent field, one integer addition for a double
  // multiplication; a NaN x keeps y (L15-E326)
  Value scaled = arith::BitcastOp::create(
      b, f64,
      arith::AddIOp::create(b, arith::BitcastOp::create(b, i64, y),
                            arith::ShLIOp::create(b, k1, ci(52))));
  scaled = sel(arith::CmpFOp::create(b, arith::CmpFPredicate::UNO, y, y), y,
               scaled);
  rewriter.replaceOp(op, mul(scaled, pow2(k2)));
  return success();
}

/// math.log of f64: fdlibm's e_log.c in double and integer operations
/// (within 1 ulp): x (a denormal scaled by 2^54) is 2^k (1 + f) with 1 + f
/// in [sqrt(2)/2, sqrt(2)), s = f / (2 + f), R a polynomial in s^2, and
/// log x = k ln2hi - ((f^2 / 2 - (s (f^2 / 2 + R) + k ln2lo)) - f), or the
/// form without f^2 / 2 that fdlibm takes near the interval's ends (both
/// computed, one selected); +0 / -0 give -inf, negative values NaN, +inf
/// itself, NaN NaN.
static LogicalResult rewriteLogF64(math::LogOp op, PatternRewriter &rewriter) {
  if (!op.getType().isF64())
    return failure();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto ci = [&](int64_t v) { return arith::ConstantIntOp::create(b, i64, v); };
  auto add = [&](Value x, Value y) { return arith::AddFOp::create(b, x, y); };
  auto sub = [&](Value x, Value y) { return arith::SubFOp::create(b, x, y); };
  auto mul = [&](Value x, Value y) { return arith::MulFOp::create(b, x, y); };
  auto fma = [&](Value x, Value y, Value z) -> Value {
    return f64Fma(b, x, y, z);
  };
  auto sel = [&](Value cnd, Value x, Value y) {
    return arith::SelectOp::create(b, cnd, x, y);
  };
  auto icmp = [&](arith::CmpIPredicate p, Value x, Value y) {
    return arith::CmpIOp::create(b, p, x, y);
  };
  auto fcmp = [&](arith::CmpFPredicate p, Value x, Value y) {
    return arith::CmpFOp::create(b, p, x, y);
  };
  auto bits = [&](Value v) { return arith::BitcastOp::create(b, i64, v); };
  Value x = op.getOperand();
  // denormals scaled by 2^54
  Value tiny = fcmp(arith::CmpFPredicate::OLT, x, c(0x1p-1022));
  Value x2 = sel(tiny, mul(x, c(0x1p54)), x);
  Value k0 = sel(tiny, ci(-54), ci(0));
  Value bx = bits(x2);
  Value hx = arith::ShRUIOp::create(b, bx, ci(32));
  Value k = arith::AddIOp::create(
      b, k0, arith::SubIOp::create(b, arith::ShRUIOp::create(b, hx, ci(20)),
                                   ci(1023)));
  hx = arith::AndIOp::create(b, hx, ci(0xfffff));
  Value i = arith::AndIOp::create(
      b, arith::AddIOp::create(b, hx, ci(0x95f64)), ci(0x100000));
  Value newHi = arith::OrIOp::create(
      b, hx, arith::XOrIOp::create(b, i, ci(0x3ff00000)));
  Value x3 = arith::BitcastOp::create(
      b, f64,
      arith::OrIOp::create(b, arith::ShLIOp::create(b, newHi, ci(32)),
                           arith::AndIOp::create(b, bx, ci(0xffffffff))));
  k = arith::AddIOp::create(b, k, arith::ShRUIOp::create(b, i, ci(20)));
  Value f = sub(x3, c(1.0));
  Value sv = arith::DivFOp::create(b, f, add(c(2.0), f));
  Value z = mul(sv, sv), w = mul(z, z);
  Value t1 = mul(w, fma(w, fma(w, c(1.531383769920937332e-01),
                               c(2.222219843214978396e-01)),
                        c(3.999999999940941908e-01)));
  Value t2 = mul(z, fma(w, fma(w, fma(w, c(1.479819860511658591e-01),
                                      c(1.818357216161805012e-01)),
                               c(2.857142874366239149e-01)),
                        c(6.666666666666735130e-01)));
  Value r = add(t2, t1);
  Value dk = arith::SIToFPOp::create(b, f64, k);
  Value hiPart = mul(dk, c(6.93147180369123816490e-01));
  Value loPart = mul(dk, c(1.90821492927058770002e-10));
  Value hfsq = mul(mul(c(0.5), f), f);
  Value a = sub(hiPart, sub(sub(hfsq, add(mul(sv, add(hfsq, r)), loPart)), f));
  Value bb = sub(hiPart, sub(sub(mul(sv, sub(f, r)), loPart), f));
  Value ij = arith::OrIOp::create(
      b, arith::SubIOp::create(b, hx, ci(0x6147a)),
      arith::SubIOp::create(b, ci(0x6b851), hx));
  Value res = sel(icmp(arith::CmpIPredicate::sgt, ij, ci(0)), a, bb);
  // specials
  Value inf = c(std::numeric_limits<double>::infinity());
  res = sel(fcmp(arith::CmpFPredicate::OEQ, x, inf), inf, res);
  res = sel(fcmp(arith::CmpFPredicate::OEQ, x, c(0.0)),
            c(-std::numeric_limits<double>::infinity()), res);
  res = sel(fcmp(arith::CmpFPredicate::ULT, x, c(0.0)),
            c(std::numeric_limits<double>::quiet_NaN()), res);
  rewriter.replaceOp(op, res);
  return success();
}

/// exp2, log2, log10, cosh, sinh and tanh of f64 from the f64 exp, log and
/// expm1 above (each rewritten in turn), in fdlibm's forms:
///   exp2 x = exp(r ln 2) 2^k, k = roundeven(x), r = x - k (exact), r ln 2
///     with its rounding error recovered by an fma (k applied as two powers
///     of two; x clamped to [-1080, 1030] keeping a NaN);
///   log2 x = k + log(m) log2(e), log10 x = k log10(2) + log(m) log10(e),
///     x = 2^k m with m in [sqrt(2)/2, sqrt(2)) (exact for powers of two);
///   cosh x = (e + 1/e) / 2, e = exp(|x|) (exp(|x| / 2)^2 / 2 from 709: one
///     exp, of the argument the range picks);
///   sinh x = h (2t - t^2 / (t + 1)) (|x| < 1) or h (t + t / (t + 1))
///     (|x| < 22), t = expm1(|x|), h = +-1/2; then h exp(|x|) (exp(|x| /
///     2)^2 h 2 from 709);
///   tanh x = 1 - 2 / (t + 2) from |x| = 1, t / (t + 2) below, t =
///     expm1(2|x|); a - a^3/3 + 2a^5/15 below 2^-9, 1 from 22, with x's sign.
template <typename OpTy>
static LogicalResult rewriteF64Function(OpTy op, PatternRewriter &rewriter) {
  if (!op.getType().isF64())
    return failure();
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  Type f64 = b.getF64Type(), i64 = b.getI64Type();
  auto c = [&](double v) {
    return arith::ConstantOp::create(b, f64, b.getF64FloatAttr(v));
  };
  auto ci = [&](int64_t v) { return arith::ConstantIntOp::create(b, i64, v); };
  auto add = [&](Value x, Value y) { return arith::AddFOp::create(b, x, y); };
  auto sub = [&](Value x, Value y) { return arith::SubFOp::create(b, x, y); };
  auto mul = [&](Value x, Value y) { return arith::MulFOp::create(b, x, y); };
  auto div = [&](Value x, Value y) { return arith::DivFOp::create(b, x, y); };
  auto fma = [&](Value x, Value y, Value z) {
    return math::FmaOp::create(b, x, y, z);
  };
  auto sel = [&](Value cnd, Value x, Value y) {
    return arith::SelectOp::create(b, cnd, x, y);
  };
  auto fcmp = [&](arith::CmpFPredicate p, Value x, Value y) {
    return arith::CmpFOp::create(b, p, x, y);
  };
  auto pow2 = [&](Value e) {
    return arith::BitcastOp::create(
        b, f64,
        arith::ShLIOp::create(b, arith::AddIOp::create(b, e, ci(1023)), ci(52)));
  };
  Value x = op.getOperand();
  Value r;
  if constexpr (std::is_same_v<OpTy, math::Exp2Op>) {
    x = sel(fcmp(arith::CmpFPredicate::OGT, x, c(1030.0)), c(1030.0), x);
    x = sel(fcmp(arith::CmpFPredicate::OLT, x, c(-1080.0)), c(-1080.0), x);
    Value k = math::RoundEvenOp::create(b, x);
    Value rr = sub(x, k);
    Value ln2 = c(0x1.62e42fefa39efp-1);
    Value hi = mul(rr, ln2);
    // r ln 2 - hi: fma's exact remainder plus r times ln 2's own error
    Value lo = add(fma(rr, ln2, arith::NegFOp::create(b, hi)),
                   mul(rr, c(2.319046813846299558e-17)));
    Value e = math::ExpOp::create(b, hi);
    e = fma(e, lo, e);
    Value ki = arith::FPToSIOp::create(b, i64, k);
    Value k1 = arith::ShRSIOp::create(b, ki, ci(1));
    // e 2^k1 exact and normal (e within [0.7, 1.42], k1 within [-540,
    // 515]): on the exponent field, as in exp (L15-E326)
    Value scaled = arith::BitcastOp::create(
        b, f64,
        arith::AddIOp::create(b, arith::BitcastOp::create(b, i64, e),
                              arith::ShLIOp::create(b, k1, ci(52))));
    scaled = sel(fcmp(arith::CmpFPredicate::UNO, e, e), e, scaled);
    r = mul(scaled, pow2(arith::SubIOp::create(b, ki, k1)));
  } else if constexpr (std::is_same_v<OpTy, math::Log2Op> ||
                       std::is_same_v<OpTy, math::Log10Op>) {
    Value tiny = fcmp(arith::CmpFPredicate::OLT, x, c(0x1p-1022));
    Value x2 = sel(tiny, mul(x, c(0x1p54)), x);
    Value bx = arith::BitcastOp::create(b, i64, x2);
    Value k = arith::SubIOp::create(
        b, arith::ShRUIOp::create(b, bx, ci(52)), ci(1023));
    k = arith::AddIOp::create(b, k, sel(tiny, ci(-54), ci(0)));
    Value m = arith::BitcastOp::create(
        b, f64,
        arith::OrIOp::create(b, arith::AndIOp::create(b, bx, ci(0xfffffffffffffLL)),
                             ci(0x3ff0000000000000LL)));
    Value big = fcmp(arith::CmpFPredicate::OGT, m, c(1.4142135623730951));
    m = sel(big, mul(m, c(0.5)), m);
    k = arith::AddIOp::create(b, k, sel(big, ci(1), ci(0)));
    Value dk = arith::SIToFPOp::create(b, f64, k);
    Value lm = math::LogOp::create(b, m);
    if constexpr (std::is_same_v<OpTy, math::Log2Op>)
      r = fma(lm, c(1.4426950408889634), dk);
    else
      r = add(mul(dk, c(0.30102999566398120)),
              fma(lm, c(0.43429448190325182),
                  mul(dk, c(-2.8037281277851704e-18))));
    Value inf = c(std::numeric_limits<double>::infinity());
    r = sel(fcmp(arith::CmpFPredicate::OEQ, x, inf), inf, r);
    r = sel(fcmp(arith::CmpFPredicate::OEQ, x, c(0.0)),
            c(-std::numeric_limits<double>::infinity()), r);
    r = sel(fcmp(arith::CmpFPredicate::ULT, x, c(0.0)),
            c(std::numeric_limits<double>::quiet_NaN()), r);
  } else if constexpr (std::is_same_v<OpTy, math::CoshOp>) {
    Value a = math::AbsFOp::create(b, x);
    // from 709 (where e^|x| overflows): e^(|x|/2) squared, halved; |x| - ln 2
    // would round |x|'s last bits away (up to 500 ulp). One exp, of |x| or
    // |x| / 2 as the range picks (L15-E309).
    Value below = fcmp(arith::CmpFPredicate::OLT, a, c(709.0));
    Value e = math::ExpOp::create(b, sel(below, a, mul(c(0.5), a)));
    Value near = add(mul(c(0.5), e), div(c(0.5), e));
    Value far = mul(mul(c(0.5), e), e);
    r = sel(below, near, far);
    r = sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, r);
  } else if constexpr (std::is_same_v<OpTy, math::SinhOp>) {
    Value a = math::AbsFOp::create(b, x);
    Value h = math::CopySignOp::create(b, c(0.5), x);
    // One exp, of |x| below 709 and |x| / 2 from it: expm1(|x|) (read
    // below 22) from it as math.expm1's rewrite would from exp(|x|), the
    // large range's e^|x| and the huge range's e^(|x| / 2) (L15-E309).
    Value below = fcmp(arith::CmpFPredicate::OLT, a, c(709.0));
    Value arg = sel(below, a, mul(c(0.5), a));
    Value w, t;
    if (expm1F64Direct) {
      // fdlibm's expm1 of the argument (L15-E814), and e^arg = t + 1 where
      // it is read (|x| from 22: t from 2^31, an integer multiple of its
      // ulp, so t + 1 is exact)
      t = expm1F64(b, arg);
      w = add(t, c(1.0));
    } else {
      w = math::ExpOp::create(b, arg);
      t = accurateExpM1(b, b.getLoc(), a, w);
    }
    Value tp1 = add(t, c(1.0));
    // (one division by t + 1, of t^2 below 1 and of t from it, each lane's
    // the same: L15-E323)
    Value below1 = fcmp(arith::CmpFPredicate::OLT, a, c(1.0));
    Value q = div(sel(below1, mul(t, t), t), tp1);
    Value small = mul(h, sub(mul(c(2.0), t), q));
    Value mid = mul(h, add(t, q));
    Value large = mul(h, w);
    Value huge = mul(mul(mul(h, c(2.0)), w), mul(c(0.5), w));
    r = sel(below1, small,
            sel(fcmp(arith::CmpFPredicate::OLT, a, c(22.0)), mid,
                sel(below, large, huge)));
    // tiny x: x itself (also -0)
    r = sel(fcmp(arith::CmpFPredicate::OLT, a, c(0x1p-28)), x, r);
    r = sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, r);
  } else {
    static_assert(std::is_same_v<OpTy, math::TanhOp>);
    Value a = math::AbsFOp::create(b, x);
    Value t1 = math::ExpM1Op::create(b, mul(c(2.0), a));
    // below 1: t / (t + 2) with t = e^2a - 1 > 0, whose relative error
    // passes on scaled by 2 / (t + 2) <= 1 (fdlibm's e^-2a - 1 form scales
    // it by up to 1.8); from 1, 1 - 2 / (t + 2). One division, of 2 or t
    // as the range picks (L15-E323).
    Value from1 = fcmp(arith::CmpFPredicate::OGE, a, c(1.0));
    Value q = div(sel(from1, c(2.0), t1), add(t1, c(2.0)));
    Value z = sel(from1, sub(c(1.0), q), q);
    // below 2^-9: a - a^3/3 + 2a^5/15 (next term 2^-58 relative), as expm1's
    // error would otherwise reach 3 ulp there
    Value a2 = mul(a, a);
    // (multiplies and adds: E815's f64Fma)
    Value series = f64Fma(b, mul(a, a2),
                          f64Fma(b, a2, c(2.0 / 15.0), c(-1.0 / 3.0)), a);
    z = sel(fcmp(arith::CmpFPredicate::OLT, a, c(0x1p-9)), series, z);
    z = sel(fcmp(arith::CmpFPredicate::OGE, a, c(22.0)), c(1.0), z);
    r = math::CopySignOp::create(b, z, x);
    r = sel(fcmp(arith::CmpFPredicate::OLT, a, c(0x1p-55)), x, r);
    r = sel(fcmp(arith::CmpFPredicate::UNO, x, x), x, r);
  }
  rewriter.replaceOp(op, r);
  return success();
}

/// math.isinf (no MathToLLVM pattern): |x| == +inf, ordered (false for NaN).
static LogicalResult rewriteIsInf(math::IsInfOp op, PatternRewriter &rewriter) {
  Type type = op.getOperand().getType();
  auto et = cast<FloatType>(getElementTypeOrSelf(type));
  Attribute inf = rewriter.getFloatAttr(
      et, APFloat::getInf(et.getFloatSemantics()));
  if (auto shaped = dyn_cast<ShapedType>(type))
    inf = SplatElementsAttr::get(shaped, inf);
  Location loc = op.getLoc();
  Value a = math::AbsFOp::create(rewriter, loc, op.getOperand());
  Value c = arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(inf));
  rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, arith::CmpFPredicate::OEQ, a,
                                             c);
  return success();
}

/// math.ipowi of a scalar (no lowering to LLVM: MathToFuncs makes it a
/// function, which a gpu.module cannot call into from outside), with
/// MathToFuncs's semantics: for p >= 0, b^p by squaring (wrapping); for
/// p < 0, 1 for b = 1, +-1 for b = -1 (p odd: -1), else 0.
/// arith.floordivsi by 2^k (k >= 1, a positive constant) is the arithmetic
/// shift by k, for every value (INT_MIN included); ceildivsi by 2^k that
/// shift plus 1 where a low bit is set (no x + 2^k - 1 to overflow). The
/// expansion's division and sign tests go (L15-E290).
template <typename OpTy>
static LogicalResult rewritePow2FloorCeilDiv(OpTy op, PatternRewriter &rewriter) {
  auto type = dyn_cast<IntegerType>(op.getType());
  std::optional<int64_t> d = getConstantIntValue(op.getRhs());
  if (!type || !d || *d < 2)
    return failure();
  Location loc = op.getLoc();
  auto c = [&](int64_t v) {
    return arith::ConstantOp::create(rewriter, loc, type,
                                     rewriter.getIntegerAttr(type, v));
  };
  Value x = op.getLhs();
  if (!llvm::isPowerOf2_64(*d)) {
    // floordivsi by another positive constant: s ^ ((x ^ s) / d), s the
    // sign splat (~x for negative x, never negative, whose truncating
    // quotient is floor((-x - 1) / d), complemented: floor(x / d), INT_MIN
    // included); the backend divides the non-negative value unsigned
    // (L15-E293).
    if constexpr (!std::is_same_v<OpTy, arith::FloorDivSIOp>)
      return failure();
    if (*d >= (int64_t(1) << (type.getWidth() - 1)))
      return failure();
    Value sgn = arith::ShRSIOp::create(rewriter, loc, x,
                                       c(type.getWidth() - 1));
    Value q = arith::DivSIOp::create(
        rewriter, loc, arith::XOrIOp::create(rewriter, loc, x, sgn), c(*d));
    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, q, sgn);
    return success();
  }
  if (llvm::Log2_64(*d) >= type.getWidth() - 1)
    return failure();
  Value q = arith::ShRSIOp::create(rewriter, loc, x, c(llvm::Log2_64(*d)));
  if constexpr (std::is_same_v<OpTy, arith::CeilDivSIOp>) {
    Value low = arith::AndIOp::create(rewriter, loc, x, c(*d - 1));
    Value up = arith::ExtUIOp::create(
        rewriter, loc, type,
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, low,
                              c(0)));
    q = arith::AddIOp::create(rewriter, loc, q, up);
  }
  rewriter.replaceOp(op, q);
  return success();
}

/// gpu.rotate (AIR has no rotate): lane j of a segment of `width` takes lane
/// j + offset where that is in the segment, a shuffle down by offset, else
/// lane j + offset - width, a shuffle up by width - offset; both uniform AIR
/// shifts over the SIMD-group, and a select by the lane's position in its
/// segment; valid where the lane is below width (L15-E298).
static LogicalResult rewriteRotate(gpu::RotateOp op, PatternRewriter &rewriter) {
  Location loc = op.getLoc();
  uint32_t off = op.getOffset(), width = op.getWidth();
  Type i32 = rewriter.getI32Type();
  auto c = [&](int64_t v) {
    return arith::ConstantOp::create(rewriter, loc, i32,
                                     rewriter.getI32IntegerAttr(v));
  };
  Value lane = arith::IndexCastOp::create(
      rewriter, loc, i32,
      gpu::LaneIdOp::create(rewriter, loc, rewriter.getIndexAttr(32)));
  Value valid =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult, lane,
                            c(width));
  Value value = op.getValue();
  if (off == 0) {
    rewriter.replaceOp(op, {value, valid});
    return success();
  }
  Value down = gpu::ShuffleOp::create(rewriter, loc, value, c(off), c(32),
                                      gpu::ShuffleMode::DOWN)
                   .getShuffleResult();
  Value up = gpu::ShuffleOp::create(rewriter, loc, value, c(width - off),
                                    c(32), gpu::ShuffleMode::UP)
                 .getShuffleResult();
  Value inSegment = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult,
      arith::AddIOp::create(rewriter, loc,
                            arith::AndIOp::create(rewriter, loc, lane,
                                                  c(width - 1)),
                            c(off)),
      c(width));
  rewriter.replaceOp(
      op, {arith::SelectOp::create(rewriter, loc, inSegment, down, up), valid});
  return success();
}

static LogicalResult rewriteIPowI(math::IPowIOp op, PatternRewriter &rewriter) {
  auto type = dyn_cast<IntegerType>(op.getType());
  if (!type)
    return failure();
  Location loc = op.getLoc();
  Value b = op.getLhs(), p = op.getRhs();
  auto c = [&](int64_t v) {
    return arith::ConstantIntOp::create(rewriter, loc, type, v);
  };
  auto cmp = [&](arith::CmpIPredicate pr, Value x, Value y) {
    return arith::CmpIOp::create(rewriter, loc, pr, x, y);
  };
  Value zero = c(0), one = c(1);
  Value negative = cmp(arith::CmpIPredicate::slt, p, zero);
  // the loop runs for p >= 0 only
  Value start = arith::SelectOp::create(rewriter, loc, negative, zero, p);
  auto loop = scf::WhileOp::create(
      rewriter, loc, TypeRange{type, type, type}, ValueRange{one, b, start},
      [&](OpBuilder &bb, Location l, ValueRange args) {
        Value go = arith::CmpIOp::create(bb, l, arith::CmpIPredicate::ne,
                                         args[2], zero);
        scf::ConditionOp::create(bb, l, go, args);
      },
      [&](OpBuilder &bb, Location l, ValueRange args) {
        Value odd = arith::CmpIOp::create(
            bb, l, arith::CmpIPredicate::ne,
            arith::AndIOp::create(bb, l, args[2], one), zero);
        Value r = arith::SelectOp::create(
            bb, l, odd, arith::MulIOp::create(bb, l, args[0], args[1]),
            args[0]);
        Value base = arith::MulIOp::create(bb, l, args[1], args[1]);
        Value next = arith::ShRUIOp::create(bb, l, args[2], one);
        scf::YieldOp::create(bb, l, ValueRange{r, base, next});
      });
  Value minusOne = c(-1);
  Value oddP = cmp(arith::CmpIPredicate::ne,
                   arith::AndIOp::create(rewriter, loc, p, one), zero);
  Value negResult = arith::SelectOp::create(
      rewriter, loc, cmp(arith::CmpIPredicate::eq, b, one), one,
      arith::SelectOp::create(
          rewriter, loc, cmp(arith::CmpIPredicate::eq, b, minusOne),
          arith::SelectOp::create(rewriter, loc, oddP, minusOne, one), zero));
  rewriter.replaceOpWithNewOp<arith::SelectOp>(op, negative, negResult,
                                               loop.getResult(0));
  return success();
}

/// Whether GPUSubgroupReduceOpLowering takes \p op to an AIR SIMD-group or
/// quad-group reduction: of i32 or a float (maximumf / minimumf of f32), over
/// the whole SIMD-group or clusters of 4 or 32 consecutive lanes.
static bool airSubgroupReduce(gpu::SubgroupReduceOp op) {
  if (std::optional<uint32_t> cluster = op.getClusterSize())
    if (op.getClusterStride() != 1 || (*cluster != 4 && *cluster != 32))
      return false;
  Type type = op.getValue().getType();
  if (!isa<FloatType>(type) && !type.isInteger(32))
    return false;
  if (op.getOp() == gpu::AllReduceOperation::MAXIMUMF ||
      op.getOp() == gpu::AllReduceOperation::MINIMUMF)
    return type.isF32();
  return true;
}

/// gpu.subgroup_reduce over the whole SIMD-group.
struct GPUSubgroupReduceOpLowering
    : ConvertOpToLLVMPattern<gpu::SubgroupReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Clusters of 4 consecutive lanes are AIR's quad-groups; one of 32 is the
    // SIMD-group.
    StringRef group = "simd";
    if (std::optional<uint32_t> cluster = op.getClusterSize()) {
      if (op.getClusterStride() != 1)
        return rewriter.notifyMatchFailure(op, "strided clusters");
      if (*cluster == 4)
        group = "quad";
      else if (*cluster != 32)
        return rewriter.notifyMatchFailure(op, "clusters of other sizes");
    }
    Type type = adaptor.getValue().getType();
    bool isFloat = isa<FloatType>(type);
    if (!isFloat && !type.isInteger(32))
      return rewriter.notifyMatchFailure(op, "only i32 and floats");
    if (op.getOp() == gpu::AllReduceOperation::MAXIMUMF ||
        op.getOp() == gpu::AllReduceOperation::MINIMUMF) {
      if (!type.isF32())
        return rewriter.notifyMatchFailure(op, "maximumf / minimumf of f32");
      rewriter.replaceOp(op, reduceOrderedFloat(rewriter, op,
                                                adaptor.getValue(), group));
      return success();
    }
    StringRef fn;
    bool isUnsigned = false;
    switch (op.getOp()) {
    case gpu::AllReduceOperation::ADD:
      fn = "sum";
      break;
    case gpu::AllReduceOperation::MUL:
      fn = "product";
      break;
    case gpu::AllReduceOperation::MINSI:
    case gpu::AllReduceOperation::MINNUMF:
      fn = "min";
      break;
    case gpu::AllReduceOperation::MAXSI:
    case gpu::AllReduceOperation::MAXNUMF:
      fn = "max";
      break;
    case gpu::AllReduceOperation::MINUI:
      fn = "min", isUnsigned = true;
      break;
    case gpu::AllReduceOperation::MAXUI:
      fn = "max", isUnsigned = true;
      break;
    case gpu::AllReduceOperation::AND:
      fn = "and";
      break;
    case gpu::AllReduceOperation::OR:
      fn = "or";
      break;
    case gpu::AllReduceOperation::XOR:
      fn = "xor";
      break;
    default:
      return rewriter.notifyMatchFailure(op, "unsupported reduction");
    }
    if (isFloat && (fn == "and" || fn == "or" || fn == "xor"))
      return rewriter.notifyMatchFailure(op, "bitwise reduction of floats");
    std::string name = ("air." + group + "_" + fn + "." +
                        typeSuffix(type, isUnsigned))
                           .str();
    rewriter.replaceOp(op, callAIR(rewriter, op, name, type,
                                   {adaptor.getValue()},
                                   AIREffects::Convergent));
    return success();
  }
};

/// The dimensions (8 x 8), strides and origin (0, 0) of AIR's matrix load
/// and store: the matrix element (r, c) is at c * strides.x + r * strides.y
/// from the pointer.
static SmallVector<Value, 3>
matrixShape(ConversionPatternRewriter &rewriter, Location loc,
            Value descriptor, bool transpose) {
  Type i64 = rewriter.getI64Type();
  auto v2i64 = VectorType::get({2}, i64);
  MemRefDescriptor memref(descriptor);
  Value rowStride = memref.stride(rewriter, loc, 0);
  Value colStride = memref.stride(rewriter, loc, 1);
  auto toI64 = [&](Value v) -> Value {
    if (v.getType() == i64)
      return v;
    return LLVM::ZExtOp::create(rewriter, loc, i64, v);
  };
  Value x = toI64(transpose ? rowStride : colStride);
  Value y = toI64(transpose ? colStride : rowStride);
  Value strides = LLVM::PoisonOp::create(rewriter, loc, v2i64);
  strides = LLVM::InsertElementOp::create(
      rewriter, loc, strides, x, LLVM::ConstantOp::create(rewriter, loc, i64, 0));
  strides = LLVM::InsertElementOp::create(
      rewriter, loc, strides, y, LLVM::ConstantOp::create(rewriter, loc, i64, 1));
  Value dims = LLVM::ConstantOp::create(
      rewriter, loc, v2i64,
      DenseElementsAttr::get(v2i64, ArrayRef<int64_t>{8, 8}));
  Value origin = LLVM::ConstantOp::create(
      rewriter, loc, v2i64,
      DenseElementsAttr::get(v2i64, ArrayRef<int64_t>{0, 0}));
  return {dims, strides, origin};
}

struct MatrixLoadOpLowering : ConvertOpToLLVMPattern<air::MatrixLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MemRefType type = op.getBase().getType();
    Value ptr = getStridedElementPtr(rewriter, loc, type, adaptor.getBase(),
                                     adaptor.getIndices());
    SmallVector<Value, 3> shape =
        matrixShape(rewriter, loc, adaptor.getBase(), op.getTranspose());
    std::string elt = typeSuffix(type.getElementType());
    unsigned as = cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace();
    std::string name = "air.simdgroup_matrix_8x8_load.v64" + elt + ".p" +
                       std::to_string(as) + elt;
    rewriter.replaceOp(op, callAIR(rewriter, op, name, op.getType(),
                                   {ptr, shape[0], shape[1], shape[2]},
                                   AIREffects::ConvergentRead));
    return success();
  }
};

struct MatrixStoreOpLowering : ConvertOpToLLVMPattern<air::MatrixStoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MemRefType type = op.getBase().getType();
    Value ptr = getStridedElementPtr(rewriter, loc, type, adaptor.getBase(),
                                     adaptor.getIndices());
    SmallVector<Value, 3> shape =
        matrixShape(rewriter, loc, adaptor.getBase(), op.getTranspose());
    std::string elt = typeSuffix(type.getElementType());
    unsigned as = cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace();
    std::string name = "air.simdgroup_matrix_8x8_store.v64" + elt + ".p" +
                       std::to_string(as) + elt;
    callAIR(rewriter, op, name, nullptr,
            {adaptor.getValue(), ptr, shape[0], shape[1], shape[2]},
            AIREffects::ConvergentWrite);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MatrixMultiplyAccumulateOpLowering
    : ConvertOpToLLVMPattern<air::MatrixMultiplyAccumulateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixMultiplyAccumulateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string ab = "v64" + typeSuffix(op.getA().getType().getElementType());
    std::string cd = "v64" + typeSuffix(op.getType().getElementType());
    std::string name = "air.simdgroup_matrix_8x8_multiply_accumulate." + cd +
                       "." + ab + "." + ab + "." + cd;
    rewriter.replaceOp(op, callAIR(rewriter, op, name, op.getType(),
                                   {adaptor.getA(), adaptor.getB(),
                                    adaptor.getC()},
                                   AIREffects::Convergent));
    return success();
  }
};

struct MatrixFillOpLowering : ConvertOpToLLVMPattern<air::MatrixFillOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(air::MatrixFillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string elt = typeSuffix(op.getValue().getType());
    std::string name =
        "air.simdgroup_matrix_8x8_init_filled.v64" + elt + "." + elt;
    rewriter.replaceOp(op, callAIR(rewriter, op, name, op.getType(),
                                   {adaptor.getValue()}));
    return success();
  }
};

/// AIR has no generic address space: its address space 0 is the thread's.
/// A memref without a memory space is device memory (the GPU dialect's
/// buffers), except a memref.alloca, which is the thread's.
static void assignDefaultMemorySpaces(gpu::GPUModuleOp m) {
  MLIRContext *ctx = m.getContext();
  auto space = [&](gpu::AddressSpace s) {
    return gpu::AddressSpaceAttr::get(ctx, s);
  };
  m.walk([&](memref::AllocaOp op) {
    MemRefType type = op.getType();
    if (!type.getMemorySpace())
      op.getResult().setType(
          MemRefType::get(type.getShape(), type.getElementType(),
                          type.getLayout(), space(gpu::AddressSpace::Private)));
  });
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](BaseMemRefType type) -> std::optional<Type> {
    if (type.getMemorySpace())
      return std::nullopt;
    Attribute device = space(gpu::AddressSpace::Global);
    if (auto ranked = dyn_cast<MemRefType>(type))
      return MemRefType::get(ranked.getShape(), ranked.getElementType(),
                             ranked.getLayout(), device);
    return UnrankedMemRefType::get(type.getElementType(), device);
  });
  replacer.recursivelyReplaceElementsIn(m, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

/// The 32-bit value of index \p v mod 2^32: additions, subtractions and
/// multiplications commute with the reduction, so their trees are rebuilt in
/// i32 (after the original, so that every use of the original can use it);
/// another index is truncated, and an index widened from i32 is the i32.
static Value narrowIndex(OpBuilder &b, Value v, DenseMap<Value, Value> &done,
                         unsigned depth = 0) {
  if (Value n = done.lookup(v))
    return n;
  Type i32 = b.getI32Type();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfterValue(v);
  Location loc = v.getLoc();
  Operation *def = v.getDefiningOp();
  Value n;
  APInt c;
  if (def && matchPattern(v, m_ConstantInt(&c))) {
    n = arith::ConstantIntOp::create(b, loc, i32, c.trunc(32).getSExtValue());
  } else if (def && isa<arith::IndexCastOp, arith::IndexCastUIOp>(def) &&
             def->getOperand(0).getType() == i32) {
    n = def->getOperand(0);
  } else if (def && depth < 16 &&
             isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(def)) {
    Value l = narrowIndex(b, def->getOperand(0), done, depth + 1);
    Value r = narrowIndex(b, def->getOperand(1), done, depth + 1);
    b.setInsertionPointAfterValue(v);
    if (isa<arith::AddIOp>(def))
      n = arith::AddIOp::create(b, loc, l, r);
    else if (isa<arith::SubIOp>(def))
      n = arith::SubIOp::create(b, loc, l, r);
    else
      n = arith::MulIOp::create(b, loc, l, r);
  } else {
    n = arith::IndexCastOp::create(b, loc, i32, v);
  }
  done[v] = n;
  return n;
}

/// An upper bound below 2^64 of index \p v, if it is a sum of products of
/// values that are not negative (GPU ids and sizes, constants, unsigned casts
/// of i32), each product of one constant of at least 1: then no partial sum
/// or product wraps (each is below 2^64), and each is at most \p v's value.
static std::optional<uint64_t> nonNegativeTree(Value v, unsigned depth = 0) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c))) {
    if (c.isNegative() || c.getActiveBits() > 32)
      return std::nullopt;
    return c.getZExtValue();
  }
  Operation *def = v.getDefiningOp();
  if (!def || depth >= 16)
    return std::nullopt;
  if (isa<gpu::ThreadIdOp, gpu::BlockIdOp, gpu::GlobalIdOp, gpu::BlockDimOp,
          gpu::GridDimOp, gpu::LaneIdOp, gpu::SubgroupIdOp,
          gpu::NumSubgroupsOp, gpu::SubgroupSizeOp>(def))
    return uint64_t(UINT32_MAX); // AIR's are i32.
  if (auto cast = dyn_cast<arith::IndexCastUIOp>(def)) {
    if (!cast.getIn().getType().isInteger(32))
      return std::nullopt;
    return uint64_t(UINT32_MAX);
  }
  bool overflow = false;
  if (isa<arith::AddIOp>(def)) {
    std::optional<uint64_t> x = nonNegativeTree(def->getOperand(0), depth + 1);
    std::optional<uint64_t> y = nonNegativeTree(def->getOperand(1), depth + 1);
    if (!x || !y)
      return std::nullopt;
    uint64_t r = llvm::SaturatingAdd(*x, *y, &overflow);
    return overflow ? std::nullopt : std::optional<uint64_t>(r);
  }
  if (isa<arith::MulIOp>(def)) {
    APInt k;
    for (unsigned i = 0; i < 2; ++i)
      if (matchPattern(def->getOperand(i), m_ConstantInt(&k)) &&
          k.isStrictlyPositive() && k.getActiveBits() <= 32) {
        std::optional<uint64_t> x =
            nonNegativeTree(def->getOperand(1 - i), depth + 1);
        if (!x)
          return std::nullopt;
        uint64_t r = llvm::SaturatingMultiply(*x, k.getZExtValue(), &overflow);
        return overflow ? std::nullopt : std::optional<uint64_t>(r);
      }
  }
  return std::nullopt;
}

/// The i32 additions narrowIndex built for the tree of \p v do not wrap
/// unsigned where the tree is one of values that are not negative
/// (nonNegativeTree) and its value is an in-bounds index of at most 2^32
/// elements: each is at most the index, below 2^32. (An access out of bounds
/// is undefined; a shared subtree is below the index of every access that
/// runs.) With nuw, the LLVM optimizer takes zext(i + c) as zext(i) + c, and
/// the backend the constant as an offset (x[i], x[i + 1] as one vector load).
/// Not the multiplications: a power of two's becomes a shl nuw, through which
/// LLVM moves the zext, and the address arithmetic becomes 64-bit (L15-E177).
static void flagNarrowTree(Value v, DenseMap<Value, Value> &done,
                           unsigned depth = 0) {
  Operation *def = v.getDefiningOp();
  if (!def || depth >= 16 || !isa<arith::AddIOp, arith::MulIOp>(def))
    return;
  Operation *n = done.lookup(v) ? done.lookup(v).getDefiningOp() : nullptr;
  // Not i + k of the counter k of a loop of a trip count not constant (one
  // that is not unrolled): with nuw, LLVM widens i + k to 64 bits in it.
  auto isCounter = [](Value x) {
    auto c = x.getDefiningOp<arith::IndexCastUIOp>();
    auto a = c ? dyn_cast<BlockArgument>(c.getIn()) : BlockArgument();
    auto loop = a ? dyn_cast<scf::ForOp>(a.getOwner()->getParentOp())
                  : scf::ForOp();
    return loop && (!getConstantIntValue(loop.getLowerBound()) ||
                    !getConstantIntValue(loop.getUpperBound()));
  };
  if (isa<arith::AddIOp>(def) &&
      (isCounter(def->getOperand(0)) || isCounter(def->getOperand(1))))
    n = nullptr;
  if (auto a = dyn_cast_or_null<arith::AddIOp>(n))
    a.setOverflowFlags(a.getOverflowFlags() | arith::IntegerOverflowFlags::nuw);
  flagNarrowTree(def->getOperand(0), done, depth + 1);
  flagNarrowTree(def->getOperand(1), done, depth + 1);
}

/// A memref.load's or memref.store's index is below its dimension's size
/// (an access out of bounds is undefined), so where the size is static and at
/// most 2^32 the index is its value mod 2^32 zero-extended, and the
/// arithmetic that computes it can be 32-bit (AGX's integer ALU is; its
/// 64-bit additions and multiplications take several instructions).
///
/// With `linearize`, an access of a memref of two or more dimensions of
/// static shape, non-negative static strides and offset, whose elements are
/// within 2^32 of its base, is one of index offset + sum(index_k * stride_k)
/// computed in i32 (each index below its size, the sum is at most the last
/// element's offset) into the memref reinterpreted as one dimension from its
/// base, rather than the 64-bit linearization of the memref lowering.
static void narrowMemRefIndices(gpu::GPUModuleOp m, bool linearize,
                                bool columnNoWrap) {
  OpBuilder b(m.getContext());
  DenseMap<Value, Value> done;
  auto narrow = [&](Operation *op, MemRefType type, unsigned first) {
    for (auto [k, size] : llvm::enumerate(type.getShape())) {
      if (ShapedType::isDynamic(size) || uint64_t(size) > (uint64_t(1) << 32))
        continue;
      OpOperand &index = op->getOpOperand(first + k);
      Operation *def = index.get().getDefiningOp();
      // Only arithmetic gains: a cast of i32 already is.
      if (!def || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(def))
        continue;
      Value n = narrowIndex(b, index.get(), done);
      if (nonNegativeTree(index.get()).has_value())
        flagNarrowTree(index.get(), done);
      b.setInsertionPoint(op);
      index.set(arith::IndexCastUIOp::create(b, op->getLoc(),
                                             b.getIndexType(), n));
    }
  };
  // The offset of the last element, if the memref qualifies.
  auto lastElement = [](MemRefType type) -> std::optional<uint64_t> {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (type.getRank() < 2 || !type.hasStaticShape() ||
        failed(type.getStridesAndOffset(strides, offset)) ||
        ShapedType::isDynamic(offset) || offset < 0)
      return std::nullopt;
    uint64_t last = offset;
    for (auto [size, stride] : llvm::zip(type.getShape(), strides)) {
      if (size == 0 || ShapedType::isDynamic(stride) || stride < 0 ||
          uint64_t(stride) > (uint64_t(1) << 32))
        return std::nullopt;
      last += uint64_t(size - 1) * uint64_t(stride);
      if (last >= (uint64_t(1) << 32))
        return std::nullopt;
    }
    return last;
  };
  auto flatten = [&](Operation *op, Value memref, MemRefType type,
                     unsigned first) {
    std::optional<uint64_t> last = lastElement(type);
    if (!last)
      return false;
    SmallVector<int64_t> strides;
    int64_t offset;
    (void)type.getStridesAndOffset(strides, offset);
    Location loc = op->getLoc();
    Type i32 = b.getI32Type();
    Value linear;
    b.setInsertionPoint(op);
    if (offset)
      linear = arith::ConstantIntOp::create(b, loc, i32, offset);
    // A column's access: the innermost loop around it steps an outer
    // dimension, the last index the same every iteration. Its sum without
    // wrapping lets LLVM step one pointer and read the unrolled iterations
    // at immediate offsets from it (L15-E641; a row's word indices, E594,
    // lost where the flags reached them: rmsnorm +21%).
    // Not in a loop that stores: its bounds would version it on its
    // ranges, and the copy's loads, hoisted above its stores, took the
    // registers that kept groups resident (E627: col_axpy_dyn +16.9%).
    auto loop = op->getParentOfType<scf::ForOp>();
    bool column = columnNoWrap && loop &&
                  loop.isDefinedOutsideOfLoop(
                      op->getOperand(first + strides.size() - 1)) &&
                  !loop->walk([](memref::StoreOp) {
                         return WalkResult::interrupt();
                       }).wasInterrupted();
    for (auto [k, stride] : llvm::enumerate(strides)) {
      Value index = op->getOperand(first + k);
      if (matchPattern(index, m_Zero()))
        continue;
      Value term = narrowIndex(b, index, done);
      b.setInsertionPoint(op);
      // (an in-bounds access: each index below its size, each product and
      // partial sum at most the last element's offset, below 2^32)
      auto nuw = column ? arith::IntegerOverflowFlags::nuw
                        : arith::IntegerOverflowFlags::none;
      if (stride != 1)
        term = arith::MulIOp::create(
            b, loc, term, arith::ConstantIntOp::create(b, loc, i32, stride),
            nuw);
      linear =
          linear ? arith::AddIOp::create(b, loc, linear, term, nuw) : term;
    }
    b.setInsertionPoint(op);
    if (!linear)
      linear = arith::ConstantIntOp::create(b, loc, i32, 0);
    auto flatType =
        MemRefType::get({int64_t(*last) + 1}, type.getElementType(),
                        MemRefLayoutAttrInterface(), type.getMemorySpace());
    Value flat = memref::ReinterpretCastOp::create(
        b, loc, flatType, memref, b.getIndexAttr(0),
        {b.getIndexAttr(*last + 1)}, {b.getIndexAttr(1)});
    SmallVector<Value> operands(op->getOperands().take_front(first - 1));
    operands.push_back(flat);
    operands.push_back(
        arith::IndexCastUIOp::create(b, loc, b.getIndexType(), linear));
    op->setOperands(operands);
    return true;
  };
  m.walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (!linearize ||
          !flatten(op, load.getMemRef(), load.getMemRefType(), 1))
        narrow(op, load.getMemRefType(), 1);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (!linearize ||
          !flatten(op, store.getMemRef(), store.getMemRefType(), 2))
        narrow(op, store.getMemRefType(), 2);
    }
  });
}

/// The coefficient of the induction variable of `loop` in index `v`, a tree
/// of additions and subtractions of it and of values defined outside the
/// loop.
static std::optional<int64_t> ivCoefficient(scf::ForOp loop, Value v,
                                            unsigned depth = 0) {
  if (v == loop.getInductionVar())
    return 1;
  if (loop.isDefinedOutsideOfLoop(v) || matchPattern(v, m_Constant()))
    return 0;
  Operation *def = v.getDefiningOp();
  if (!def || depth >= 16 || !isa<arith::AddIOp, arith::SubIOp>(def))
    return std::nullopt;
  std::optional<int64_t> l = ivCoefficient(loop, def->getOperand(0), depth + 1);
  std::optional<int64_t> r = ivCoefficient(loop, def->getOperand(1), depth + 1);
  if (!l || !r)
    return std::nullopt;
  return isa<arith::AddIOp>(def) ? *l + *r : *l - *r;
}

/// A bound B <= 2^31 on the offsets o = iv - lb of the iterations that
/// `loop` executes (without undefined behaviour): its constant extent
/// ub - lb, or the size of a static dimension that a memref.load,
/// memref.store or AIR matrix load or store each iteration executes indexes
/// by x + iv or x - iv, x loop invariant. (The accessed indices x + lb + o, o = 0, step, 2 step, ...,
/// are below the size S, which keeps a step of S or more to one iteration
/// and the offsets of more below S.)
static std::optional<uint64_t> loopOffsetBound(scf::ForOp loop) {
  constexpr uint64_t kMax = uint64_t(1) << 31;
  std::optional<uint64_t> bound;
  auto tighten = [&](uint64_t b) {
    if (b >= 1 && b <= kMax)
      bound = bound ? std::min(*bound, b) : b;
  };
  std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
  if (lb && ub && *ub > *lb && uint64_t(*ub) - uint64_t(*lb) <= kMax)
    tighten(uint64_t(*ub) - uint64_t(*lb));
  // The accesses every iteration reaches: those of the body before any op
  // that has regions or calls (which might not return).
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (op.getNumRegions() || isa<CallOpInterface>(op))
      break;
    MemRefType type;
    ValueRange indices;
    if (auto load = dyn_cast<memref::LoadOp>(op))
      type = load.getMemRefType(), indices = load.getIndices();
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      type = store.getMemRefType(), indices = store.getIndices();
    // An 8x8 matrix at [i, j] accesses base[i + r, j + c] (or [i + c, j + r])
    // for r, c in [0, 8): i and j themselves are in bounds.
    else if (auto load = dyn_cast<air::MatrixLoadOp>(op))
      type = cast<MemRefType>(load.getBase().getType()),
      indices = load.getIndices();
    else if (auto store = dyn_cast<air::MatrixStoreOp>(op))
      type = cast<MemRefType>(store.getBase().getType()),
      indices = store.getIndices();
    else
      continue;
    for (auto [size, index] : llvm::zip(type.getShape(), indices)) {
      if (ShapedType::isDynamic(size))
        continue;
      std::optional<int64_t> c = ivCoefficient(loop, index);
      if (c && (*c == 1 || *c == -1))
        tighten(size);
    }
  }
  return bound;
}

/// A binary search's scf.while over index (lo, hi), lo from 0, continuing
/// while lo < hi (unsigned: a signed test of a hi0 negative as an index would
/// end the original loop at once with hi0, not its truncation), whose body loads a static 1-D memref of S <= 2^30 elements
/// at mid = (lo + hi) >> 1 and yields lo, hi, mid or mid + 1 (or selects of
/// them), in i32. Each yielded value is at most max(lo, hi, S); in the first
/// iteration mid = hi0 >> 1 is below S (a load out of bounds is undefined),
/// so hi0 < 2 S and every value stays below 2^31 and every lo + hi below
/// 2^32: the i32 values are the index ones. A hi0 of 2^32 or more makes the
/// first load out of bounds (or, truncated to 0, ends the loop at once, which
/// is then as good), and with hi0 = 0 neither loop runs.
static bool narrowBinarySearch(scf::WhileOp w, IRRewriter &rewriter) {
  if (w.getNumOperands() != 2 || w.getNumResults() != 2)
    return false;
  for (Type t : w.getOperandTypes())
    if (!t.isIndex())
      return false;
  if (!matchPattern(w.getInits()[0], m_Zero()))
    return false;
  Block &before = w.getBefore().front();
  Block &after = w.getAfter().front();
  auto cond = dyn_cast<scf::ConditionOp>(before.getTerminator());
  if (!cond || before.getOperations().size() != 2 ||
      cond.getArgs().size() != 2 || cond.getArgs()[0] != before.getArgument(0) ||
      cond.getArgs()[1] != before.getArgument(1))
    return false;
  auto cmp = cond.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getLhs() != before.getArgument(0) ||
      cmp.getRhs() != before.getArgument(1) ||
      cmp.getPredicate() != arith::CmpIPredicate::ult)
    return false;
  Value lo = after.getArgument(0), hi = after.getArgument(1);
  // mid = (lo + hi) >> 1, loaded at before anything that could leave the
  // body (only ops without regions or calls before the load)
  Value mid, midInc;
  arith::AddIOp sum;
  arith::ShRUIOp half;
  bool loaded = false;
  for (Operation &op : after.without_terminator()) {
    if (op.getNumRegions() || isa<CallOpInterface>(op))
      break;
    if (auto a = dyn_cast<arith::AddIOp>(op);
        a && !sum &&
        ((a.getLhs() == lo && a.getRhs() == hi) ||
         (a.getLhs() == hi && a.getRhs() == lo)))
      sum = a;
    else if (auto sh = dyn_cast<arith::ShRUIOp>(op);
             sh && sum && sh.getLhs() == sum.getResult() &&
             matchPattern(sh.getRhs(), m_One()))
      half = sh, mid = sh.getResult();
    else if (auto load = dyn_cast<memref::LoadOp>(op);
             load && mid && load.getIndices().size() == 1 &&
             load.getIndices()[0] == mid) {
      MemRefType t = load.getMemRefType();
      if (t.getRank() == 1 && !t.isDynamicDim(0) &&
          t.getDimSize(0) <= (int64_t(1) << 30))
        loaded = true;
    } else if (auto a = dyn_cast<arith::AddIOp>(op);
               a && mid && a.getLhs() == mid &&
               matchPattern(a.getRhs(), m_One()))
      midInc = a.getResult();
  }
  if (!loaded)
    return false;
  auto yield = cast<scf::YieldOp>(after.getTerminator());
  // the yielded values: lo, hi, mid, mid + 1, or selects of them
  std::function<bool(Value, unsigned)> bounded = [&](Value v, unsigned d) {
    if (v == lo || v == hi || v == mid || (midInc && v == midInc))
      return true;
    auto sel = v.getDefiningOp<arith::SelectOp>();
    return sel && d < 4 && sel->getBlock() == &after &&
           bounded(sel.getTrueValue(), d + 1) &&
           bounded(sel.getFalseValue(), d + 1);
  };
  for (Value v : yield.getResults())
    if (!bounded(v, 0))
      return false;

  Location loc = w.getLoc();
  Type i32 = rewriter.getI32Type();
  rewriter.setInsertionPoint(w);
  Value init0 = arith::ConstantIntOp::create(rewriter, loc, i32, 0);
  Value init1 = arith::IndexCastOp::create(rewriter, loc, i32, w.getInits()[1]);
  auto beforeBuilder = [&](OpBuilder &b, Location l, ValueRange args) {
    Value go = arith::CmpIOp::create(b, l, cmp.getPredicate(), args[0], args[1]);
    scf::ConditionOp::create(b, l, go, args);
  };
  auto afterBuilder = [&](OpBuilder &b, Location l, ValueRange args) {
    Value lo32 = args[0], hi32 = args[1];
    Type idx = b.getIndexType();
    Value sum32 = arith::AddIOp::create(b, l, lo32, hi32);
    Value one = arith::ConstantIntOp::create(b, l, i32, 1);
    Value mid32 = arith::ShRUIOp::create(b, l, sum32, one);
    Value midInc32 = arith::AddIOp::create(b, l, mid32, one);
    IRMapping map;
    map.map(lo, arith::IndexCastUIOp::create(b, l, idx, lo32));
    map.map(hi, arith::IndexCastUIOp::create(b, l, idx, hi32));
    map.map(mid, arith::IndexCastUIOp::create(b, l, idx, mid32));
    if (midInc)
      map.map(midInc, arith::IndexCastUIOp::create(b, l, idx, midInc32));
    for (Operation &op : after.without_terminator()) {
      if (&op == sum.getOperation() || &op == half.getOperation() ||
          (midInc && op.getNumResults() == 1 && op.getResult(0) == midInc))
        continue;
      b.clone(op, map);
    }
    std::function<Value(Value)> narrowed = [&](Value v) -> Value {
      if (v == lo)
        return lo32;
      if (v == hi)
        return hi32;
      if (v == mid)
        return mid32;
      if (midInc && v == midInc)
        return midInc32;
      auto sel = v.getDefiningOp<arith::SelectOp>();
      return arith::SelectOp::create(b, l, map.lookup(sel.getCondition()),
                                     narrowed(sel.getTrueValue()),
                                     narrowed(sel.getFalseValue()));
    };
    SmallVector<Value> ys;
    for (Value v : yield.getResults())
      ys.push_back(narrowed(v));
    scf::YieldOp::create(b, l, ys);
  };
  auto nw = scf::WhileOp::create(rewriter, loc, TypeRange{i32, i32},
                                 ValueRange{init0, init1}, beforeBuilder,
                                 afterBuilder);
  rewriter.setInsertionPointAfter(nw);
  SmallVector<Value> results;
  for (Value r : nw.getResults())
    results.push_back(arith::IndexCastUIOp::create(rewriter, loc,
                                                   rewriter.getIndexType(), r));
  rewriter.replaceOp(w, results);
  return true;
}

/// The i32 of index \p v if it is an i32 cast to index (by a signed cast if
/// \p isSigned, else an unsigned one) or a constant that one gives.
static Value i32Bound(OpBuilder &b, Location loc, Value v, bool isSigned) {
  Type i32 = b.getI32Type();
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c))) {
    if (isSigned ? !c.isSignedIntN(32) : !c.isIntN(32))
      return {};
    return arith::ConstantIntOp::create(b, loc, i32, c.trunc(32).getSExtValue());
  }
  Operation *def = v.getDefiningOp();
  if (!def || def->getOperand(0).getType() != i32 ||
      !(isSigned ? isa<arith::IndexCastOp>(def)
                 : isa<arith::IndexCastUIOp>(def)))
    return {};
  return def->getOperand(0);
}

/// Whether every iteration of `loop` (that has no undefined behaviour)
/// accesses a memref at an index that is its iv: before any op of its body
/// that has regions or calls. The iv is then not negative.
static bool indexesEveryIteration(scf::ForOp loop) {
  Value iv = loop.getInductionVar();
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (op.getNumRegions() || isa<CallOpInterface>(op))
      return false;
    ValueRange indices;
    if (auto load = dyn_cast<memref::LoadOp>(op))
      indices = load.getIndices();
    else if (auto store = dyn_cast<memref::StoreOp>(op))
      indices = store.getIndices();
    if (llvm::is_contained(indices, iv))
      return true;
  }
  return false;
}

/// An scf.for over index by 1 from lb to ub, both i32 values cast to index by
/// signed casts (or both by unsigned ones), at least one not a constant, is
/// the loop over i32 from lb to ub (compared signed, or unsigned), its iv the
/// i32 iv cast to index the same way: the iterations are the same, and the
/// i32 iv + 1 does not wrap (it is at most ub). (A signed iv from a constant
/// lb >= 0, or that every iteration uses as a memref index, is not negative,
/// and is cast unsigned,
/// which keeps its addresses' extensions zero extensions.) No access bounds are needed
/// (unlike the offsets loop below) and no trip count is computed.
static bool narrowI32BoundsLoop(scf::ForOp loop, IRRewriter &rewriter) {
  if (!matchPattern(loop.getStep(), m_One()))
    return false;
  Value lb = loop.getLowerBound(), ub = loop.getUpperBound();
  if (matchPattern(lb, m_Constant()) && matchPattern(ub, m_Constant()))
    return false;
  Location loc = loop.getLoc();
  for (bool isSigned : {true, false}) {
    rewriter.setInsertionPoint(loop);
    Value l = i32Bound(rewriter, loc, lb, isSigned);
    Value u = l ? i32Bound(rewriter, loc, ub, isSigned) : Value();
    if (!l || !u) {
      for (Value v : {l, u})
        if (v && v.getDefiningOp<arith::ConstantOp>() && v.use_empty())
          rewriter.eraseOp(v.getDefiningOp());
      continue;
    }
    Value one = arith::ConstantIntOp::create(rewriter, loc,
                                             rewriter.getI32Type(), 1);
    auto newLoop = scf::ForOp::create(rewriter, loc, l, u, one,
                                      loop.getInitArgs(),
                                      /*bodyBuilder=*/nullptr,
                                      /*unsignedCmp=*/!isSigned);
    Block *body = newLoop.getBody();
    if (!body->empty())
      rewriter.eraseOp(body->getTerminator());
    rewriter.setInsertionPointToStart(body);
    Value iv = newLoop.getInductionVar();
    Type index = rewriter.getIndexType();
    APInt lbValue;
    std::optional<APInt> lbConst;
    if (matchPattern(lb, m_ConstantInt(&lbValue)))
      lbConst = lbValue;
    Value wide =
        isSigned && !(lbConst && !lbConst->isNegative()) &&
                !indexesEveryIteration(loop)
            ? arith::IndexCastOp::create(rewriter, loc, index, iv)
                       .getResult()
                 : arith::IndexCastUIOp::create(rewriter, loc, index, iv)
                       .getResult();
    SmallVector<Value> args{wide};
    llvm::append_range(args, newLoop.getRegionIterArgs());
    rewriter.mergeBlocks(loop.getBody(), body, args);
    rewriter.replaceOp(loop, newLoop.getResults());
    return true;
  }
  return false;
}

/// The values [lo, hi] an index takes, where it is built from constants,
/// lane and thread positions (their upper_bound, the subgroup size 32 or the
/// known threadgroup size) by additions and multiplications by constants, all
/// non-negative.
static std::optional<std::pair<int64_t, int64_t>> indexRange(Value v,
                                                             int depth = 0) {
  if (depth > 8)
    return std::nullopt;
  if (std::optional<int64_t> c = getConstantIntValue(v))
    return *c >= 0 ? std::optional(std::make_pair(*c, *c)) : std::nullopt;
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto lane = dyn_cast<gpu::LaneIdOp>(op)) {
    int64_t n = 32;
    if (std::optional<APInt> b = lane.getUpperBound())
      n = std::min<int64_t>(n, b->getZExtValue());
    return std::make_pair(int64_t(0), n - 1);
  }
  if (auto tid = dyn_cast<gpu::ThreadIdOp>(op)) {
    std::optional<int64_t> n;
    if (std::optional<APInt> b = tid.getUpperBound())
      n = b->getZExtValue();
    if (std::optional<uint32_t> bs = gpu::getKnownDimensionSizeAround(
            tid, gpu::DimensionKind::Block, tid.getDimension()))
      n = n ? std::min<int64_t>(*n, *bs) : int64_t(*bs);
    if (!n || *n < 1)
      return std::nullopt;
    return std::make_pair(int64_t(0), *n - 1);
  }
  if (isa<arith::AddIOp, arith::MulIOp>(op)) {
    auto a = indexRange(op->getOperand(0), depth + 1);
    auto b = a ? indexRange(op->getOperand(1), depth + 1) : std::nullopt;
    if (!b || a->second > (int64_t(1) << 31) || b->second > (int64_t(1) << 31))
      return std::nullopt;
    if (isa<arith::AddIOp>(op))
      return std::make_pair(a->first + b->first, a->second + b->second);
    return std::make_pair(a->first * b->first, a->second * b->second);
  }
  return std::nullopt;
}

/// An scf.if with results whose regions only compute without side effects
/// (speculatable) and load from memrefs of static, non-empty shapes runs
/// both regions before it, each load at its indices clamped into the
/// memref (umin(i, size - 1)), and selects their results on the condition.
/// Where the region's load runs in the original its indices are in bounds
/// (an access out of bounds is undefined), so the clamp keeps them; where
/// it does not, the clamped access is one of the memref's elements, whose
/// value the select drops. An index known below its size (a constant, a
/// position of that upper bound) is not clamped, and an if needing more
/// than one clamp is left. A scan's guarded load,
/// in = j < n; v = in ? x[r, j] : 0 (find_first: the if, its exec mask
/// and the copies it took, L15-E736).
static void speculateGuardedLoads(gpu::GPUModuleOp m) {
  constexpr unsigned kMaxOps = 16;
  SmallVector<scf::IfOp> ifs;
  m.walk([&](scf::IfOp op) {
    if (op.getNumResults() && !op.getElseRegion().empty())
      ifs.push_back(op);
  });
  auto knownBelow = [](Value v, int64_t size) {
    if (std::optional<int64_t> c = getConstantIntValue(v))
      return *c >= 0 && *c < size;
    std::optional<APInt> ub;
    if (auto o = v.getDefiningOp<gpu::ThreadIdOp>())
      ub = o.getUpperBound();
    else if (auto o = v.getDefiningOp<gpu::BlockIdOp>())
      ub = o.getUpperBound();
    else if (auto o = v.getDefiningOp<gpu::GlobalIdOp>())
      ub = o.getUpperBound();
    else if (auto o = v.getDefiningOp<gpu::LaneIdOp>())
      ub = o.getUpperBound();
    return ub && ub->getActiveBits() <= 63 &&
           int64_t(ub->getZExtValue()) <= size;
  };
  for (scf::IfOp op : ifs) {
    auto loadable = [](memref::LoadOp load) {
      MemRefType type = load.getMemRefType();
      return type.hasStaticShape() &&
             llvm::all_of(type.getShape(), [](int64_t d) { return d > 0; });
    };
    auto qualifies = [&](Region &r) {
      if (!r.hasOneBlock())
        return false;
      unsigned n = 0;
      for (Operation &o : r.front().without_terminator()) {
        if (++n > kMaxOps || o.getNumRegions())
          return false;
        if (auto load = dyn_cast<memref::LoadOp>(o)) {
          if (!loadable(load))
            return false;
          continue;
        }
        if (!isPure(&o) || !isSpeculatable(&o))
          return false;
      }
      return true;
    };
    // (one load at least: otherwise the if is the canonicalizer's to fold;
    // at most one index to clamp: a padded convolution's taps, two each,
    // lost their cheap branches on in-bounds positions, conv3x3_pad +54%)
    bool anyLoad = false;
    unsigned clamps = 0;
    op->walk([&](memref::LoadOp load) {
      anyLoad = true;
      for (auto [k, size] : llvm::enumerate(load.getMemRefType().getShape()))
        clamps += !knownBelow(load.getIndices()[k], size);
    });
    if (!anyLoad || clamps > 1 || !qualifies(op.getThenRegion()) ||
        !qualifies(op.getElseRegion()))
      continue;
    OpBuilder b(op);
    auto hoist = [&](Region &r) {
      Block &blk = r.front();
      for (Operation &o : llvm::make_early_inc_range(blk.without_terminator())) {
        o.moveBefore(op);
        auto load = dyn_cast<memref::LoadOp>(o);
        if (!load)
          continue;
        b.setInsertionPoint(load);
        for (auto [k, size] : llvm::enumerate(load.getMemRefType().getShape())) {
          OpOperand &index = load->getOpOperand(1 + k);
          if (knownBelow(index.get(), size))
            continue;
          Value last = arith::ConstantIndexOp::create(b, load.getLoc(),
                                                      size - 1);
          index.set(arith::MinUIOp::create(b, load.getLoc(), index.get(),
                                           last));
        }
      }
      return cast<scf::YieldOp>(blk.getTerminator()).getOperands();
    };
    SmallVector<Value> thenValues(hoist(op.getThenRegion()));
    SmallVector<Value> elseValues(hoist(op.getElseRegion()));
    b.setInsertionPoint(op);
    SmallVector<Value> results;
    Value cond = op.getCondition();
    for (auto [t, e] : llvm::zip(thenValues, elseValues)) {
      auto sel = arith::SelectOp::create(b, op.getLoc(), cond, t, e);
      results.push_back(sel);
    }
    op.replaceAllUsesWith(results);
    op.erase();
    // A compare of a result used only in an and with the condition reads
    // the then value, the and a select on the condition: where it is false
    // the result is false whatever the compare, also of a poison then value
    // (a plain and would be poison; alive2's speculated-compare test) (the
    // scan's go = in && v <= t).
    for (auto [res, t] : llvm::zip(results, thenValues)) {
      for (OpOperand &use : llvm::make_early_inc_range(res.getUses())) {
        Operation *cmp = use.getOwner();
        if (!isa<arith::CmpFOp, arith::CmpIOp>(cmp) ||
            cmp->getBlock() != res.getDefiningOp()->getBlock() ||
            !llvm::all_of(cmp->getResult(0).getUsers(), [&](Operation *u) {
              auto a = dyn_cast<arith::AndIOp>(u);
              return a && (a.getLhs() == cond || a.getRhs() == cond);
            }))
          continue;
        use.set(t);
        for (Operation *u : llvm::make_early_inc_range(
                 cmp->getResult(0).getUsers())) {
          OpBuilder ab(u);
          Value f = arith::ConstantIntOp::create(ab, u->getLoc(),
                                                 ab.getI1Type(), 0);
          Value sel = arith::SelectOp::create(ab, u->getLoc(), cond,
                                              cmp->getResult(0), f);
          u->getResult(0).replaceAllUsesWith(sel);
          u->erase();
        }
      }
      if (res.use_empty())
        res.getDefiningOp()->erase();
    }
  }
}

/// An scf.for over index from a lb of values [lo, hi] (indexRange) to a
/// constant ub by a constant step, where every lb gives the same trip count
/// T = ceil((ub - lb) / step) > 0: a loop over k from 0 to T by 1, iv = lb +
/// k * step (the same iterations in the same order). Its trip count is then
/// a constant LLVM sees (a lane-strided row loop, `scf.for %j = %lane to
/// %c1024 step %c32`, runs 32 times for every lane), which the AGX unroller
/// needs (L15-E511).
static void constantTripLoops(gpu::GPUModuleOp m) {
  SmallVector<scf::ForOp> loops;
  m.walk([&](scf::ForOp loop) {
    if (loop.getInductionVar().getType().isIndex() && !loop.getUnsignedCmp())
      loops.push_back(loop);
  });
  IRRewriter rewriter(m.getContext());
  for (scf::ForOp loop : loops) {
    std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(loop.getStep());
    if (!ub || !step || *step <= 0 || *ub <= 0 || *ub > (int64_t(1) << 31) ||
        getConstantIntValue(loop.getLowerBound()))
      continue;
    // (an iv compared, a window's j <= row, keeps its loop: LLVM bounds the
    // iterations by it)
    if (any_of(loop.getInductionVar().getUsers(),
               [](Operation *u) { return isa<arith::CmpIOp>(u); }))
      continue;
    std::optional<std::pair<int64_t, int64_t>> r =
        indexRange(loop.getLowerBound());
    if (!r || r->second >= *ub)
      continue;
    auto trips = [&](int64_t lb) { return (*ub - lb + *step - 1) / *step; };
    int64_t t = trips(r->first);
    if (t != trips(r->second) || t < 2)
      continue;
    Location loc = loop.getLoc();
    rewriter.setInsertionPoint(loop);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value count = arith::ConstantIndexOp::create(rewriter, loc, t);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto newLoop = scf::ForOp::create(rewriter, loc, zero, count, one,
                                      loop.getInitArgs());
    Block *body = newLoop.getBody();
    if (!body->empty())
      rewriter.eraseOp(body->getTerminator());
    rewriter.setInsertionPointToStart(body);
    // (lb + k step <= hi + (T - 1) step < ub: no wrap)
    auto flags = arith::IntegerOverflowFlags::nuw |
                 arith::IntegerOverflowFlags::nsw;
    Value offset = arith::MulIOp::create(rewriter, loc,
                                         newLoop.getInductionVar(),
                                         loop.getStep(), flags);
    Value iv = arith::AddIOp::create(rewriter, loc, loop.getLowerBound(),
                                     offset, flags);
    SmallVector<Value> args{iv};
    llvm::append_range(args, newLoop.getRegionIterArgs());
    rewriter.mergeBlocks(loop.getBody(), body, args);
    rewriter.replaceOp(loop, newLoop.getResults());
  }
}

/// An scf.for over index whose offsets o = iv - lb are below a bound B <=
/// 2^31 (loopOffsetBound) is a loop over o in i32, unsigned: from 0 to
/// umin(ub - lb, B) (0 for ub <= lb) by umin(step, B), with iv = lb + o.
/// The trip count is the original's: with two iterations or more, step < B,
/// so the offsets are the same and the original's that reach B would be out
/// of bounds; with one, the second offset min(step, B) is past the extent
/// too. The offsets, below B + B <= 2^32, do not wrap. Index arithmetic of
/// the iv that narrow-indices computes in i32 becomes lb + o in i32.
static void narrowLoops(gpu::GPUModuleOp m) {
  {
    SmallVector<scf::WhileOp> whiles;
    m.walk([&](scf::WhileOp w) { whiles.push_back(w); });
    IRRewriter rewriter(m.getContext());
    for (scf::WhileOp w : whiles)
      narrowBinarySearch(w, rewriter);
  }
  SmallVector<scf::ForOp> loops;
  m.walk([&](scf::ForOp loop) {
    if (loop.getInductionVar().getType().isIndex() && !loop.getUnsignedCmp())
      loops.push_back(loop);
  });
  IRRewriter rewriter(m.getContext());
  for (scf::ForOp loop : loops) {
    if (narrowI32BoundsLoop(loop, rewriter))
      continue;
    std::optional<uint64_t> bound = loopOffsetBound(loop);
    if (!bound)
      continue;
    Location loc = loop.getLoc();
    rewriter.setInsertionPoint(loop);
    Type i32 = rewriter.getI32Type();
    Value lb = loop.getLowerBound(), ub = loop.getUpperBound();
    Value b = arith::ConstantIndexOp::create(rewriter, loc, *bound);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value extent = rewriter.createOrFold<arith::SelectOp>(
        loc,
        rewriter.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                             ub, lb),
        rewriter.createOrFold<arith::MinUIOp>(
            loc, rewriter.createOrFold<arith::SubIOp>(loc, ub, lb), b),
        zero);
    Value step =
        rewriter.createOrFold<arith::MinUIOp>(loc, loop.getStep(), b);
    auto narrow = [&](Value v) {
      return rewriter.createOrFold<arith::IndexCastUIOp>(loc, i32, v);
    };
    auto newLoop = scf::ForOp::create(
        rewriter, loc, arith::ConstantIntOp::create(rewriter, loc, i32, 0),
        narrow(extent), narrow(step), loop.getInitArgs(),
        /*bodyBuilder=*/nullptr, /*unsignedCmp=*/true);
    Block *body = newLoop.getBody();
    if (!body->empty())
      rewriter.eraseOp(body->getTerminator());
    rewriter.setInsertionPointToStart(body);
    Value iv = arith::IndexCastUIOp::create(rewriter, loc,
                                            rewriter.getIndexType(),
                                            newLoop.getInductionVar());
    if (!matchPattern(lb, m_Zero()))
      iv = arith::AddIOp::create(rewriter, loc, lb, iv);
    SmallVector<Value> args{iv};
    llvm::append_range(args, newLoop.getRegionIterArgs());
    rewriter.mergeBlocks(loop.getBody(), body, args);
    rewriter.replaceOp(loop, newLoop.getResults());
  }
}

/// Whether `op` may run again before its kernel returns: it is in a loop, or
/// in a region of more than one block (whose branches may loop).
static bool mayRepeat(Operation *op) {
  for (Region *r = op->getParentRegion(); r;
       r = r->getParentOp()->getParentRegion()) {
    if (!r->hasOneBlock())
      return true;
    Operation *parent = r->getParentOp();
    if (isa<LoopLikeOpInterface>(parent))
      return true;
    if (isa<gpu::GPUFuncOp>(parent))
      return false;
  }
  return true;
}

/// A uniform gpu.all_reduce of an f16, f32 or i32 by an operation AIR's
/// SIMD-group reductions have, in a kernel whose known threadgroup size is a
/// multiple of 32 (whole SIMD-groups), is the SIMD-group reduction, the
/// partial of each SIMD-group in a threadgroup buffer, a barrier, and each
/// SIMD-group's reduction of the partials (lane l reads partial l; the lanes
/// past the SIMD-group count read partial 0, or the identity of an addition,
/// multiplication or xor): the result in every thread, without the generic
/// lowering's partial-SIMD-group shuffles and its second round through
/// threadgroup memory. Where the reduction may run again, a barrier after it
/// keeps the next round's partials from overwriting ones not yet read.
namespace {
/// A gpu.all_reduce lowerAllReducesToSimdgroups takes, with the identity its
/// out-of-range lanes read (none for min, max, and, or).
struct SimdgroupReduce {
  gpu::AllReduceOp op;
  gpu::AllReduceOperation kind;
  TypedAttr identity;
};
} // namespace

static std::optional<SimdgroupReduce> simdgroupReduce(gpu::AllReduceOp op) {
  if (!op.getUniform() || !op.getOp() || !op.getBody().empty())
    return std::nullopt;
  auto func = op->getParentOfType<gpu::GPUFuncOp>();
  std::optional<uint64_t> threads = knownThreadgroupThreads(op);
  if (!func || !threads || *threads == 0 || *threads % 32 || *threads > 1024)
    return std::nullopt;
  Type type = op.getType();
  // (Integers of 8, 16 and 64 bits too, which reduce to the same value in
  // any order: subgroup_reduce's reductions of them, L15-E278.)
  if (!type.isF32() && !type.isF16() &&
      !(type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
        type.isInteger(64)))
    return std::nullopt;
  gpu::AllReduceOperation kind = *op.getOp();
  TypedAttr identity;
  Builder b(op.getContext());
  switch (kind) {
  case gpu::AllReduceOperation::ADD:
    identity = isa<FloatType>(type)
                   ? TypedAttr(b.getFloatAttr(
                         type, APFloat::getZero(
                                   cast<FloatType>(type).getFloatSemantics(),
                                   /*Negative=*/true)))
                   : TypedAttr(b.getIntegerAttr(type, 0));
    break;
  case gpu::AllReduceOperation::MUL:
    identity = isa<FloatType>(type) ? TypedAttr(b.getFloatAttr(type, 1.0))
                                    : TypedAttr(b.getIntegerAttr(type, 1));
    break;
  case gpu::AllReduceOperation::XOR:
    identity = b.getIntegerAttr(type, 0);
    break;
  case gpu::AllReduceOperation::MINSI:
  case gpu::AllReduceOperation::MINUI:
  case gpu::AllReduceOperation::MAXSI:
  case gpu::AllReduceOperation::MAXUI:
  case gpu::AllReduceOperation::MINNUMF:
  case gpu::AllReduceOperation::MAXNUMF:
  case gpu::AllReduceOperation::AND:
  case gpu::AllReduceOperation::OR:
    break;
  // f32: subgroup_reduce's ordered-key maximum (minimum) and NaN or
  // (reduceOrderedFloat), idempotent like maxnum for the lanes past the
  // SIMD-group count (L15-E275).
  case gpu::AllReduceOperation::MAXIMUMF:
  case gpu::AllReduceOperation::MINIMUMF:
    // (f16: its subgroup_reduce is the f32 one, L15-E280.)
    if (!type.isF32() && !type.isF16())
      return std::nullopt;
    break;
  default:
    return std::nullopt;
  }
  return SimdgroupReduce{op, kind, identity};
}

/// Whether `next` can join the round of `group` (whose last is before it in
/// the same block): what lies between has no side effects, uses none of the
/// group's results, and every use of them comes after `next` (where the
/// joint round puts the results).
static bool joinsRound(SmallVectorImpl<SimdgroupReduce> &group,
                       gpu::AllReduceOp next) {
  gpu::AllReduceOp last = group.back().op;
  if (last->getBlock() != next->getBlock() ||
      knownThreadgroupThreads(last) != knownThreadgroupThreads(next))
    return false;
  for (Operation *o = last->getNextNode(); o && o != next.getOperation();
       o = o->getNextNode())
    if (!isMemoryEffectFree(o) || o->getNumRegions())
      return false;
  for (SimdgroupReduce &r : group)
    for (Operation *user : r.op->getUsers()) {
      Operation *a = next->getBlock()->findAncestorOpInBlock(*user);
      if (!a || !next->isBeforeInBlock(a))
        return false;
    }
  return true;
}

/// A uniform gpu.all_reduce of an f16, f32 or i32 by an operation AIR's
/// SIMD-group reductions have, in a kernel whose known threadgroup size is a
/// multiple of 32 (whole SIMD-groups), is the SIMD-group reduction, the
/// partial of each SIMD-group in a threadgroup buffer, a barrier, and each
/// SIMD-group's reduction of the partials (lane l reads partial l; the lanes
/// past the SIMD-group count read partial 0, or the identity of an addition,
/// multiplication or xor): the result in every thread, without the generic
/// lowering's partial-SIMD-group shuffles and its second round through
/// threadgroup memory. Where the reduction may run again, a barrier after it
/// keeps the next round's partials from overwriting ones not yet read.
/// Consecutive independent reductions (a layer norm's sum and sum of
/// squares) share one round: their partials stored together, one barrier
/// (L15-E273).
static void lowerAllReducesToSimdgroups(gpu::GPUModuleOp m) {
  SmallVector<gpu::AllReduceOp> ops;
  m.walk([&](gpu::AllReduceOp op) { ops.push_back(op); });
  SmallVector<SmallVector<SimdgroupReduce, 2>> rounds;
  for (gpu::AllReduceOp op : ops) {
    std::optional<SimdgroupReduce> r = simdgroupReduce(op);
    if (!r)
      continue;
    if (!rounds.empty() && joinsRound(rounds.back(), op))
      rounds.back().push_back(*r);
    else
      rounds.push_back({*r});
  }
  for (SmallVector<SimdgroupReduce, 2> &round : rounds) {
    gpu::AllReduceOp last = round.back().op;
    auto func = last->getParentOfType<gpu::GPUFuncOp>();
    uint64_t groups = *knownThreadgroupThreads(last) / 32;
    OpBuilder b(last);
    Location loc = last.getLoc();
    SmallVector<Value> partials;
    for (SimdgroupReduce &r : round)
      partials.push_back(gpu::SubgroupReduceOp::create(
          b, r.op.getLoc(), r.op.getValue(), r.kind, /*uniform=*/true));
    if (groups == 1) {
      for (auto [r, p] : llvm::zip(round, partials)) {
        r.op.replaceAllUsesWith(p);
        r.op.erase();
      }
      continue;
    }
    SmallVector<Value> buffers;
    for (SimdgroupReduce &r : round)
      buffers.push_back(func.addWorkgroupAttribution(
          MemRefType::get({int64_t(groups)}, r.op.getType(),
                          MemRefLayoutAttrInterface(),
                          gpu::AddressSpaceAttr::get(
                              b.getContext(), gpu::AddressSpace::Workgroup)),
          loc));
    Value lane = gpu::LaneIdOp::create(b, loc, /*upper_bound=*/IntegerAttr());
    Value group =
        gpu::SubgroupIdOp::create(b, loc, /*upper_bound=*/IntegerAttr());
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    Value first =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, zero);
    scf::IfOp::create(b, loc, first, [&](OpBuilder &b, Location loc) {
      for (auto [p, buf] : llvm::zip(partials, buffers))
        memref::StoreOp::create(b, loc, p, buf, group);
      scf::YieldOp::create(b, loc);
    });
    gpu::BarrierOp::create(b, loc);
    Value inRange = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, lane,
        arith::ConstantIndexOp::create(b, loc, groups));
    Value index = arith::SelectOp::create(b, loc, inRange, lane, zero);
    SmallVector<Value> results;
    for (auto [r, buf] : llvm::zip(round, buffers)) {
      Value value = memref::LoadOp::create(b, loc, buf, index);
      if (r.identity)
        value = arith::SelectOp::create(
            b, loc, inRange, value,
            arith::ConstantOp::create(b, loc, r.identity));
      results.push_back(gpu::SubgroupReduceOp::create(b, loc, value, r.kind,
                                                      /*uniform=*/true));
    }
    if (mayRepeat(last))
      gpu::BarrierOp::create(b, loc);
    for (auto [r, v] : llvm::zip(round, results)) {
      r.op.replaceAllUsesWith(v);
      r.op.erase();
    }
  }
}

struct LowerGpuOpsToAIROpsPass final
    : public impl::ConvertGpuOpsToAIROpsBase<LowerGpuOpsToAIROpsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    auto llvmDataLayout = m->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getDataLayoutAttrName());
    if (!llvmDataLayout) {
      llvmDataLayout = StringAttr::get(ctx, kAIRDataLayout);
      m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(), llvmDataLayout);
    }

    LowerToLLVMOptions options(
        ctx, DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.dataLayout = llvm::DataLayout(llvmDataLayout.getValue());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.useBarePtrCallConv = useBarePtrCallConv;

    assignDefaultMemorySpaces(m);

    // gpu.global_id is AIR's thread_position_in_grid, one 32-bit read
    // (which an address takes as a 32-bit index), where the dimension has
    // fewer than 2^32 threads: its upper_bound, or the known threadgroup and
    // grid sizes, say so. Elsewhere the 32-bit register could wrap, and it is
    // block_id * block_dim + thread_id, in index.
    m.walk([&](gpu::GlobalIdOp op) {
      gpu::Dimension d = op.getDimension();
      std::optional<uint64_t> threads;
      if (std::optional<APInt> b = op.getUpperBound())
        threads = b->getZExtValue();
      std::optional<uint32_t> bs = gpu::getKnownDimensionSizeAround(
          op, gpu::DimensionKind::Block, d);
      std::optional<uint32_t> gs = gpu::getKnownDimensionSizeAround(
          op, gpu::DimensionKind::Grid, d);
      if (bs && gs) {
        uint64_t n = uint64_t(*bs) * *gs;
        threads = threads ? std::min(*threads, n) : n;
      }
      if (threads && *threads <= (uint64_t(1) << 32))
        return;
      OpBuilder b(op);
      Location loc = op.getLoc();
      Value bid = gpu::BlockIdOp::create(b, loc, d);
      Value bdim = gpu::BlockDimOp::create(b, loc, d);
      Value tid = gpu::ThreadIdOp::create(b, loc, d);
      Value v = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, bid, bdim), tid);
      op.replaceAllUsesWith(v);
      op.erase();
    });
    // The upper_bounds of the thread and threadgroup positions that hold
    // for the whole kernel: those of reads in its entry block that every
    // execution reaches (no op with regions or a call before them). AIR's
    // position vectors carry a range only where all three components are
    // bounded; the AGX backend bounds each component's register by these
    // (L15-E391).
    llvm::StringMap<std::array<std::array<uint64_t, 3>, 3>> kernelBounds;
    m.walk([&](gpu::GPUFuncOp f) {
      if (!f.isKernel() || f.getBody().empty())
        return;
      std::array<std::array<uint64_t, 3>, 3> bounds{};
      bool any = false;
      auto note = [&](unsigned kind, gpu::Dimension d,
                      std::optional<APInt> ub) {
        if (!ub || ub->getActiveBits() > 32)
          return;
        uint64_t v = ub->getZExtValue();
        uint64_t &b = bounds[kind][static_cast<unsigned>(d)];
        if (v && v < (uint64_t(1) << 32) && (!b || v < b)) {
          b = v;
          any = true;
        }
      };
      for (Operation &op : f.getBody().front()) {
        if (op.getNumRegions() || isa<CallOpInterface>(op))
          break;
        if (auto o = dyn_cast<gpu::ThreadIdOp>(op))
          note(0, o.getDimension(), o.getUpperBound());
        else if (auto o = dyn_cast<gpu::BlockIdOp>(op))
          note(1, o.getDimension(), o.getUpperBound());
        else if (auto o = dyn_cast<gpu::GlobalIdOp>(op))
          note(2, o.getDimension(), o.getUpperBound());
      }
      if (any)
        kernelBounds[f.getName()] = bounds;
    });
    ::approximateHalfFunctions = approximateHalfFunctions;
    ::expm1F64Direct = expm1F64Direct;
    ::log1pF64Direct = log1pF64Direct;
    ::f64PolynomialFma = f64PolynomialFma;
    if (speculateGuardedLoads)
      ::speculateGuardedLoads(m);
    if (constantTripLoops)
      ::constantTripLoops(m);
    if (narrowLoops)
      ::narrowLoops(m);
    if (narrowIndices)
      narrowMemRefIndices(m, linearizeIndices, columnNoWrap);
    if (simdgroupAllReduce)
      lowerAllReducesToSimdgroups(m);
    m.walk([](gpu::ShuffleOp op) {
      if (!getConstantIntValue(op.getOffset()) &&
          isThreadgroupUniform(op.getOffset()))
        op->setAttr(kUniformOffsetAttr, UnitAttr::get(op.getContext()));
    });
    // f32 erf's rational approximation P / Q: Q's denominators stay within
    // a few binades of 1 where the result is used, so the hardware's
    // reciprocal y0, y = y0 + y0 (1 - Q y0), q0 = P y, q = q0 + y (P - Q q0)
    // gives the correctly rounded quotient that the general division does
    // with its exponent scaling and special cases (every f32 input's erf
    // checked the same, flushing: L15-E499). erf expanded here by upstream's
    // patterns, the division they create noted.
    if (erfReciprocalDivision) {
      SmallVector<Operation *> erfs;
      m.walk([&](math::ErfOp op) { erfs.push_back(op); });
      struct Divisions : RewriterBase::Listener {
        SmallVector<arith::DivFOp> divs;
        void notifyOperationInserted(Operation *op,
                                     OpBuilder::InsertPoint) override {
          if (auto d = dyn_cast<arith::DivFOp>(op))
            divs.push_back(d);
        }
      } listener;
      if (!erfs.empty()) {
        RewritePatternSet patterns(ctx);
        populatePolynomialApproximateErfPattern(patterns);
        populateMathF32ExpansionPatterns(
            patterns, [](StringRef name) { return name == "math.erf"; },
            /*benefit=*/2);
        GreedyRewriteConfig config;
        config.setListener(&listener);
        config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
        (void)applyOpPatternsGreedily(erfs, std::move(patterns), config);
      }
      for (arith::DivFOp op : listener.divs) {
        if (!getElementTypeOrSelf(op.getType()).isF32())
          continue;
        OpBuilder b(op);
        Location loc = op.getLoc();
        Type type = op.getType();
        TypedAttr oneAttr = b.getF32FloatAttr(1.0f);
        if (auto vt = dyn_cast<VectorType>(type))
          oneAttr = DenseElementsAttr::get(vt, b.getF32FloatAttr(1.0f));
        Value one = arith::ConstantOp::create(b, loc, type, oneAttr);
        Value p = op.getLhs(), q = op.getRhs();
        Value y0 = arith::DivFOp::create(b, loc, one, q,
                                         arith::FastMathFlags::afn);
        Value nq = arith::NegFOp::create(b, loc, q);
        Value e = math::FmaOp::create(b, loc, nq, y0, one);
        Value y = math::FmaOp::create(b, loc, e, y0, y0);
        Value q0 = arith::MulFOp::create(b, loc, p, y);
        Value r = math::FmaOp::create(b, loc, nq, q0, p);
        Value res = math::FmaOp::create(b, loc, r, y, q0);
        op.replaceAllUsesWith(res);
        op.erase();
      }
    }
    // In-dialect lowering of gpu.all_reduce and friends first.
    {
      RewritePatternSet patterns(ctx);
      populateGpuAllReducePatterns(patterns);
      populateGpuShufflePatterns(patterns);
      patterns.add(rewriteAccurateM1<math::ExpM1Op>);
      patterns.add(rewriteAccurateM1<math::Log1pOp>);
      patterns.add(rewriteInverseHyperbolic<math::AsinhOp>);
      patterns.add(rewriteInverseHyperbolicF64<math::AsinhOp>);
      patterns.add(rewriteInverseHyperbolicF64<math::AcoshOp>);
      patterns.add(rewriteInverseHyperbolicF64<math::AtanhOp>);
      patterns.add(rewriteInverseHyperbolicF64<math::CbrtOp>);
      patterns.add(rewriteInverseHyperbolic<math::AcoshOp>);
      patterns.add(rewriteInverseHyperbolic<math::AtanhOp>);
      patterns.add(rewriteInverseHyperbolic<math::CbrtOp>);
      // erf: upstream's f32 approximation (f16 and bf16 widened to it).
      populatePolynomialApproximateErfPattern(patterns);
      // erfc: upstream's f32 approximation likewise, its last step without
      // a flushed correction (not 1 - erf, which cancels for large x).
      patterns.add(rewriteErfc);
      populateMathF32ExpansionPatterns(
          patterns,
          [](StringRef name) {
            return name == "math.erf" || name == "math.erfc";
          },
          /*benefit=*/2);
      patterns.add(rewriteIsInf);
      patterns.add(rewriteExpF64);
      patterns.add(rewriteLogF64);
      patterns.add(rewriteF64Function<math::Exp2Op>);
      patterns.add(rewriteF64Function<math::Log2Op>);
      patterns.add(rewriteF64Function<math::Log10Op>);
      patterns.add(rewriteF64Function<math::CoshOp>);
      patterns.add(rewriteF64Function<math::SinhOp>);
      patterns.add(rewriteF64Function<math::TanhOp>);
      patterns.add(rewriteInverseTrigF64<math::AtanOp>);
      patterns.add(rewriteInverseTrigF64<math::Atan2Op>);
      patterns.add(rewriteInverseTrigF64<math::AsinOp>);
      patterns.add(rewriteInverseTrigF64<math::AcosOp>);
      patterns.add(rewriteIPowI);
      patterns.add(rewritePow2FloorCeilDiv<arith::FloorDivSIOp>);
      patterns.add(rewritePow2FloorCeilDiv<arith::CeilDivSIOp>);
      patterns.add(rewriteRotate);
      (void)applyPatternsGreedily(m, std::move(patterns));
    }
    // A reduction AIR has no op for (over clusters of 2, 8 or 16 lanes: AIR
    // has quad-group and SIMD-group ones; of i64, i8 and i16 and f16's
    // maximumf being widened first): xor shuffles at 1, 2, 4, ... within each
    // cluster (the order AIR's reductions add in), which the shuffle lowering
    // takes to AIR.
    {
      // i8 and i16: the i32 reduction of the values extended (sign-extended
      // for a signed minimum or maximum), truncated, the same value (the
      // others are exact modulo 2^n), and AIR's 32-bit reduction instead of
      // shuffles (L15-E279).
      // f16 and bf16 maximumf / minimumf likewise in f32 (exact both ways:
      // the result is one of the values, or a NaN), its ordered-key
      // reduction instead of shuffles (L15-E280).
      SmallVector<gpu::SubgroupReduceOp> narrow;
      m.walk([&](gpu::SubgroupReduceOp op) {
        Type type = op.getType();
        if (type.isInteger(8) || type.isInteger(16) ||
            ((type.isF16() || type.isBF16()) &&
             (op.getOp() == gpu::AllReduceOperation::MAXIMUMF ||
              op.getOp() == gpu::AllReduceOperation::MINIMUMF)))
          narrow.push_back(op);
      });
      IRRewriter rewriter(ctx);
      for (gpu::SubgroupReduceOp op : narrow) {
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();
        if (isa<FloatType>(op.getType())) {
          Value wide = arith::ExtFOp::create(rewriter, loc,
                                             rewriter.getF32Type(),
                                             op.getValue());
          auto reduced = gpu::SubgroupReduceOp::create(
              rewriter, loc, wide, op.getOp(), op.getUniform(),
              op.getClusterSize(), op.getClusterStride());
          rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, op.getType(),
                                                       reduced.getResult());
          continue;
        }
        Type i32 = rewriter.getI32Type();
        bool isSigned = op.getOp() == gpu::AllReduceOperation::MINSI ||
                        op.getOp() == gpu::AllReduceOperation::MAXSI;
        Value wide =
            isSigned
                ? Value(arith::ExtSIOp::create(rewriter, loc, i32, op.getValue()))
                : Value(arith::ExtUIOp::create(rewriter, loc, i32, op.getValue()));
        auto reduced = gpu::SubgroupReduceOp::create(
            rewriter, loc, wide, op.getOp(), op.getUniform(),
            op.getClusterSize(), op.getClusterStride());
        rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, op.getType(),
                                                     reduced.getResult());
      }
      // i64 minimum / maximum (signed or not) as two AIR reductions of i32:
      // the high words' (in the operation's signedness), then the low
      // words' (unsigned) of the lanes whose high word is that one, the
      // others giving the identity (0 for a maximum, ~0 for a minimum):
      // instead of a shuffle butterfly of 64-bit compares (L15-E442).
      // i64 and / or / xor and sums likewise (L15-E443).
      SmallVector<gpu::SubgroupReduceOp> wide;
      m.walk([&](gpu::SubgroupReduceOp op) {
        std::optional<uint32_t> cluster = op.getClusterSize();
        if (op.getType().isInteger(64) &&
            (!cluster || (op.getClusterStride() == 1 &&
                          (*cluster == 4 || *cluster == 32))) &&
            (op.getOp() == gpu::AllReduceOperation::MAXUI ||
             op.getOp() == gpu::AllReduceOperation::MINUI ||
             op.getOp() == gpu::AllReduceOperation::MAXSI ||
             op.getOp() == gpu::AllReduceOperation::MINSI ||
             op.getOp() == gpu::AllReduceOperation::AND ||
             op.getOp() == gpu::AllReduceOperation::OR ||
             op.getOp() == gpu::AllReduceOperation::XOR ||
             op.getOp() == gpu::AllReduceOperation::ADD))
          wide.push_back(op);
      });
      for (gpu::SubgroupReduceOp op : wide) {
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();
        Type i32 = rewriter.getI32Type(), i64 = rewriter.getI64Type();
        auto reduce32 = [&](Value x, gpu::AllReduceOperation kind) -> Value {
          return gpu::SubgroupReduceOp::create(rewriter, loc, x, kind,
                                               op.getUniform(),
                                               op.getClusterSize(),
                                               op.getClusterStride());
        };
        auto c64 = [&](int64_t c) -> Value {
          return arith::ConstantIntOp::create(rewriter, loc, i64, c);
        };
        auto words = [&](Value x) -> std::pair<Value, Value> {
          return {arith::TruncIOp::create(
                      rewriter, loc, i32,
                      arith::ShRUIOp::create(rewriter, loc, x, c64(32))),
                  arith::TruncIOp::create(rewriter, loc, i32, x)};
        };
        auto join = [&](Value h, Value l) -> Value {
          Value h64 = arith::ShLIOp::create(
              rewriter, loc, arith::ExtUIOp::create(rewriter, loc, i64, h),
              c64(32));
          return arith::OrIOp::create(
              rewriter, loc, h64, arith::ExtUIOp::create(rewriter, loc, i64, l));
        };
        // and / or / xor: each word on its own. add: the high words modulo
        // 2^32 and the low words as two 16-bit halves, whose sums over 32
        // lanes stay below 2^21, recombined in 64 bits.
        if (op.getOp() == gpu::AllReduceOperation::AND ||
            op.getOp() == gpu::AllReduceOperation::OR ||
            op.getOp() == gpu::AllReduceOperation::XOR) {
          auto [h, l] = words(op.getValue());
          rewriter.replaceOp(op, join(reduce32(h, op.getOp()),
                                      reduce32(l, op.getOp())));
          continue;
        }
        if (op.getOp() == gpu::AllReduceOperation::ADD) {
          auto [h, l] = words(op.getValue());
          Value c16 = arith::ConstantIntOp::create(rewriter, loc, i32, 16);
          Value m16 = arith::ConstantIntOp::create(rewriter, loc, i32, 0xffff);
          Value sh = reduce32(h, gpu::AllReduceOperation::ADD);
          Value sm = reduce32(arith::ShRUIOp::create(rewriter, loc, l, c16),
                              gpu::AllReduceOperation::ADD);
          Value sl = reduce32(arith::AndIOp::create(rewriter, loc, l, m16),
                              gpu::AllReduceOperation::ADD);
          Value r = arith::ShLIOp::create(
              rewriter, loc, arith::ExtUIOp::create(rewriter, loc, i64, sh),
              c64(32));
          r = arith::AddIOp::create(
              rewriter, loc, r,
              arith::ShLIOp::create(
                  rewriter, loc, arith::ExtUIOp::create(rewriter, loc, i64, sm),
                  c64(16)));
          r = arith::AddIOp::create(rewriter, loc, r,
                                    arith::ExtUIOp::create(rewriter, loc, i64, sl));
          rewriter.replaceOp(op, r);
          continue;
        }
        bool isMax = op.getOp() == gpu::AllReduceOperation::MAXUI ||
                     op.getOp() == gpu::AllReduceOperation::MAXSI;
        Value v = op.getValue();
        Value c32 = arith::ConstantIntOp::create(rewriter, loc, i64, 32);
        Value hi = arith::TruncIOp::create(
            rewriter, loc, i32, arith::ShRUIOp::create(rewriter, loc, v, c32));
        Value lo = arith::TruncIOp::create(rewriter, loc, i32, v);
        Value mh = gpu::SubgroupReduceOp::create(
            rewriter, loc, hi, op.getOp(), op.getUniform(), op.getClusterSize(),
            op.getClusterStride());
        Value eq = arith::CmpIOp::create(rewriter, loc,
                                         arith::CmpIPredicate::eq, hi, mh);
        Value id = arith::ConstantIntOp::create(rewriter, loc, i32,
                                                isMax ? 0 : -1);
        Value lo2 = arith::SelectOp::create(rewriter, loc, eq, lo, id);
        Value ml = gpu::SubgroupReduceOp::create(
            rewriter, loc, lo2,
            isMax ? gpu::AllReduceOperation::MAXUI
                  : gpu::AllReduceOperation::MINUI,
            op.getUniform(), op.getClusterSize(), op.getClusterStride());
        Value h64 = arith::ShLIOp::create(
            rewriter, loc, arith::ExtUIOp::create(rewriter, loc, i64, mh), c32);
        Value l64 = arith::ExtUIOp::create(rewriter, loc, i64, ml);
        rewriter.replaceOpWithNewOp<arith::OrIOp>(op, h64, l64);
      }
      SmallVector<Operation *> clustered;
      m.walk([&](gpu::SubgroupReduceOp op) {
        if (!airSubgroupReduce(op))
          clustered.push_back(op);
      });
      RewritePatternSet patterns(ctx);
      // Clusters of 8 or 16 of a type AIR's quad-group reduction takes: it
      // for lanes 1 and 2, then shuffles at 4 (and 8): the same order, fewer
      // instructions.
      patterns.add(+[](gpu::SubgroupReduceOp op, PatternRewriter &rewriter) {
        std::optional<uint32_t> size = op.getClusterSize();
        Type type = op.getType();
        if (!size || (*size != 8 && *size != 16) || op.getClusterStride() != 1 ||
            !(type.isInteger(32) || isa<FloatType>(type)) ||
            ((op.getOp() == gpu::AllReduceOperation::MAXIMUMF ||
              op.getOp() == gpu::AllReduceOperation::MINIMUMF) &&
             !type.isF32()))
          return failure();
        Location loc = op.getLoc();
        Value v = gpu::SubgroupReduceOp::create(rewriter, loc, op.getValue(),
                                                op.getOp(), op.getUniform(),
                                                /*cluster_size=*/4);
        for (uint32_t m = 4; m < *size; m <<= 1) {
          Value s = gpu::ShuffleOp::create(rewriter, loc, v, m, 32,
                                           gpu::ShuffleMode::XOR)
                        .getShuffleResult();
          v = vector::makeArithReduction(rewriter, loc,
                                         gpu::convertReductionKind(op.getOp()),
                                         v, s);
        }
        rewriter.replaceOp(op, v);
        return success();
      });
      if (!clustered.empty())
        (void)applyOpPatternsGreedily(clustered, std::move(patterns));
      clustered.clear();
      m.walk([&](gpu::SubgroupReduceOp op) {
        if (!airSubgroupReduce(op))
          clustered.push_back(op);
      });
      RewritePatternSet generic(ctx);
      populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
          generic, /*subgroupSize=*/32);
      populateGpuLowerSubgroupReduceToShufflePatterns(generic,
                                                      /*subgroupSize=*/32);
      // 64-bit values shuffled whole (the shuffle lowering splits them).
      populateGpuLowerSubgroupReduceToShufflePatterns(
          generic, /*subgroupSize=*/32, /*shuffleBitwidth=*/64,
          /*benefit=*/0);
      populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
          generic, /*subgroupSize=*/32, /*shuffleBitwidth=*/64,
          /*benefit=*/0);
      if (!clustered.empty())
        (void)applyOpPatternsGreedily(clustered, std::move(generic));
      RewritePatternSet shuffles(ctx);
      populateGpuShufflePatterns(shuffles);
      (void)applyPatternsGreedily(m, std::move(shuffles));
    }

    LLVMTypeConverter converter(ctx, options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case gpu::AddressSpace::Global:
            return kDeviceAddressSpace;
          case gpu::AddressSpace::Workgroup:
            return kThreadgroupAddressSpace;
          case gpu::AddressSpace::Private:
            return kThreadAddressSpace;
          case gpu::AddressSpace::Constant:
            return kConstantAddressSpace;
          }
          llvm_unreachable("unknown address space enum value");
        });

    RewritePatternSet llvmPatterns(ctx);
    LLVMConversionTarget target(getContext());
    for (Dialect *dialect : ctx->getLoadedDialects())
      if (auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect))
        iface->populateConvertToLLVMConversionPatterns(target, converter,
                                                       llvmPatterns);
    // Structured control flow (loops in kernels) becomes branches.
    populateSCFToControlFlowConversionPatterns(llvmPatterns);
    target.addIllegalDialect<scf::SCFDialect>();
    populateGpuToAIRConversionPatterns(converter, llvmPatterns);
    configureGpuToAIRConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      return signalPassFailure();

    // Kernels carry the "air-kernel" function attribute, and the threadgroup
    // size when it is known as an AGX hint, which AIR ignores (the AGX
    // backend budgets registers and barriers by it).
    auto *gpuDialect = ctx->getLoadedDialect<gpu::GPUDialect>();
    m.walk([&](LLVM::LLVMFuncOp f) {
      if (!f->hasAttr(kKernelMarker))
        return;
      f->removeAttr(kKernelMarker);
      SmallVector<Attribute> passthrough;
      if (ArrayAttr old = f.getPassthroughAttr())
        passthrough.append(old.begin(), old.end());
      passthrough.push_back(StringAttr::get(ctx, "air-kernel"));
      auto knownBlockSize = gpuDialect->getKnownBlockSizeAttrHelper();
      if (DenseI32ArrayAttr size = knownBlockSize.getAttr(f)) {
        int64_t threads = 1;
        for (int32_t d : size.asArrayRef())
          threads *= d;
        passthrough.push_back(ArrayAttr::get(
            ctx, {StringAttr::get(ctx, "agx-max-threads-per-threadgroup"),
                  StringAttr::get(ctx, std::to_string(threads))}));
        // and each extent (the AGX backend bounds the threadgroup's
        // positions and sizes by it)
        std::string dims;
        for (int32_t d : size.asArrayRef())
          dims += (dims.empty() ? "" : ",") + std::to_string(d);
        passthrough.push_back(ArrayAttr::get(
            ctx, {StringAttr::get(ctx, "agx-threads-per-threadgroup"),
                  StringAttr::get(ctx, dims)}));
      }
      // The grid's extents when known (the launch promises them): the AGX
      // backend bounds the position special registers by them just before
      // instruction selection.
      auto knownGridSize = gpuDialect->getKnownGridSizeAttrHelper();
      if (DenseI32ArrayAttr grid = knownGridSize.getAttr(f)) {
        auto join = [](ArrayRef<int64_t> v) {
          std::string out;
          for (int64_t x : v)
            out += (out.empty() ? "" : ",") + std::to_string(x);
          return out;
        };
        SmallVector<int64_t> groups(grid.asArrayRef());
        passthrough.push_back(ArrayAttr::get(
            ctx, {StringAttr::get(ctx, "agx-threadgroups-per-grid"),
                  StringAttr::get(ctx, join(groups))}));
        if (DenseI32ArrayAttr size = knownBlockSize.getAttr(f)) {
          SmallVector<int64_t> threads;
          for (auto [g, b] : llvm::zip(grid.asArrayRef(), size.asArrayRef()))
            threads.push_back(int64_t(g) * b);
          passthrough.push_back(ArrayAttr::get(
              ctx, {StringAttr::get(ctx, "agx-threads-per-grid"),
                    StringAttr::get(ctx, join(threads))}));
        }
      }
      if (auto it = kernelBounds.find(f.getName()); it != kernelBounds.end()) {
        static const char *const roles[] = {
            "air.thread_position_in_threadgroup",
            "air.threadgroup_position_in_grid", "air.thread_position_in_grid"};
        for (auto [kind, b] : llvm::enumerate(it->second)) {
          if (!b[0] && !b[1] && !b[2])
            continue;
          passthrough.push_back(ArrayAttr::get(
              ctx, {StringAttr::get(ctx, std::string("agx-upper-bounds:") +
                                             roles[kind]),
                    StringAttr::get(ctx, std::to_string(b[0]) + "," +
                                             std::to_string(b[1]) + "," +
                                             std::to_string(b[2]))}));
        }
      }
      f.setPassthroughAttr(ArrayAttr::get(ctx, passthrough));
    });
  }
};

} // namespace

void mlir::configureGpuToAIRConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalDialect<air::AIRDialect>();
  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp>();
}

void mlir::populateGpuToAIRConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GPUSystemValueLowering<gpu::ThreadIdOp>>(
      converter, "air.thread_position_in_threadgroup", IndexKind::Block,
      IntrType::Id);
  patterns.add<GPUSystemValueLowering<gpu::BlockIdOp>>(
      converter, "air.threadgroup_position_in_grid", IndexKind::Grid,
      IntrType::Id);
  patterns.add<GPUSystemValueLowering<gpu::BlockDimOp>>(
      converter, "air.threads_per_threadgroup", IndexKind::Block,
      IntrType::Dim);
  patterns.add<GPUSystemValueLowering<gpu::GridDimOp>>(
      converter, "air.threadgroups_per_grid", IndexKind::Grid, IntrType::Dim);
  patterns.add<GPUSystemValueLowering<gpu::GlobalIdOp>>(
      converter, "air.thread_position_in_grid", IndexKind::Other,
      IntrType::Id);
  patterns.add<GPUSimdgroupValueLowering<gpu::LaneIdOp>>(
      converter, "air.thread_index_in_simdgroup");
  patterns.add<GPUSimdgroupValueLowering<gpu::SubgroupIdOp>>(
      converter, "air.simdgroup_index_in_threadgroup");
  patterns.add<GPUSimdgroupValueLowering<gpu::NumSubgroupsOp>>(
      converter, "air.simdgroups_per_threadgroup");
  patterns.add<GPUSubgroupSizeOpLowering, GPUBarrierOpLowering,
               GPUShuffleOpLowering, GPUBallotOpLowering,
               GPUSubgroupBroadcastOpLowering,
               GPUSubgroupReduceOpLowering, MatrixLoadOpLowering,
               MatrixStoreOpLowering, MatrixMultiplyAccumulateOpLowering,
               MatrixFillOpLowering>(converter);
  patterns.add<GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      GPUFuncOpLoweringOptions{
          /*allocaAddrSpace=*/kThreadAddressSpace,
          /*workgroupAddrSpace=*/kThreadgroupAddressSpace,
          StringAttr::get(&converter.getContext(), kKernelMarker),
          /*kernelBlockSizeAttributeName=*/{},
          /*kernelClusterSizeAttributeName=*/{}});
}
