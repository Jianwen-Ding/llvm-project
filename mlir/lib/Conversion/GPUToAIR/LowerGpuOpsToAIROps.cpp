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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
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
        if (*width == 32)
          native = "air.simd_shuffle_up.u.i32";
        break;
      case gpu::ShuffleMode::DOWN:
        if (*width == 32)
          native = "air.simd_shuffle_down.u.i32";
        break;
      case gpu::ShuffleMode::IDX:
        break;
      }
    }

    Value valid, lane16;
    if (!native.empty()) {
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
    for (Value part : parts)
      shuffled.push_back(callAIR(rewriter, op, native, i32, {part, lane16},
                                 AIREffects::Convergent));
    Value result = LLVM::composeValue(rewriter, loc, shuffled,
                                      adaptor.getValue().getType());
    rewriter.replaceOp(op, {result, valid});
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

/// gpu.subgroup_reduce maximumf / minimumf of f32: AIR's float SIMD-group
/// maximum and minimum are maxnum and minnum, which drop NaNs and may order
/// the zeros either way, so it is the signed i32 maximum (minimum) of the
/// floats' keys, decoded, or a NaN where the group's or of "is NaN" is set
/// (alive2: tests/alive-tv/gpu/subgroup-maximum-key, subgroup-minimum-key).
static Value reduceOrderedFloat(ConversionPatternRewriter &rewriter,
                                gpu::SubgroupReduceOp op, Value value,
                                StringRef group) {
  Location loc = op.getLoc();
  Type i32 = rewriter.getI32Type();
  Type f32 = value.getType();
  bool isMax = op.getOp() == gpu::AllReduceOperation::MAXIMUMF;
  // Used twice: one value even if it is undefined.
  value = LLVM::FreezeOp::create(rewriter, loc, value);
  Value key = orderedFloatKey(
      rewriter, loc, LLVM::BitcastOp::create(rewriter, loc, i32, value));
  Value reduced =
      callAIR(rewriter, op,
              ("air." + group + (isMax ? "_max" : "_min") + ".s.i32").str(),
              i32, {key}, AIREffects::Convergent);
  Value result = LLVM::BitcastOp::create(
      rewriter, loc, f32, orderedFloatKey(rewriter, loc, reduced));
  Value isNaN = LLVM::ZExtOp::create(
      rewriter, loc, i32,
      LLVM::FCmpOp::create(rewriter, loc, LLVM::FCmpPredicate::uno, value,
                           value));
  Value anyNaN = callAIR(rewriter, op, ("air." + group + "_or.s.i32").str(),
                         i32, {isNaN}, AIREffects::Convergent);
  Value nan = LLVM::ConstantOp::create(
      rewriter, loc, f32,
      rewriter.getFloatAttr(f32, APFloat::getQNaN(APFloat::IEEEsingle())));
  return LLVM::SelectOp::create(
      rewriter, loc,
      LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ne, anyNaN,
                           LLVM::ConstantOp::create(rewriter, loc, i32, 0)),
      nan, result);
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
static void narrowMemRefIndices(gpu::GPUModuleOp m, bool linearize) {
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
    for (auto [k, stride] : llvm::enumerate(strides)) {
      Value index = op->getOperand(first + k);
      if (matchPattern(index, m_Zero()))
        continue;
      Value term = narrowIndex(b, index, done);
      b.setInsertionPoint(op);
      if (stride != 1)
        term = arith::MulIOp::create(
            b, loc, term, arith::ConstantIntOp::create(b, loc, i32, stride));
      linear = linear ? arith::AddIOp::create(b, loc, linear, term) : term;
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
/// ub - lb, or the size of a static dimension that a memref.load or
/// memref.store each iteration executes indexes by x + iv or x - iv, x loop
/// invariant. (The accessed indices x + lb + o, o = 0, step, 2 step, ...,
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

/// An scf.for over index whose offsets o = iv - lb are below a bound B <=
/// 2^31 (loopOffsetBound) is a loop over o in i32, unsigned: from 0 to
/// umin(ub - lb, B) (0 for ub <= lb) by umin(step, B), with iv = lb + o.
/// The trip count is the original's: with two iterations or more, step < B,
/// so the offsets are the same and the original's that reach B would be out
/// of bounds; with one, the second offset min(step, B) is past the extent
/// too. The offsets, below B + B <= 2^32, do not wrap. Index arithmetic of
/// the iv that narrow-indices computes in i32 becomes lb + o in i32.
static void narrowLoops(gpu::GPUModuleOp m) {
  SmallVector<scf::ForOp> loops;
  m.walk([&](scf::ForOp loop) {
    if (loop.getInductionVar().getType().isIndex() && !loop.getUnsignedCmp())
      loops.push_back(loop);
  });
  IRRewriter rewriter(m.getContext());
  for (scf::ForOp loop : loops) {
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
static void lowerAllReducesToSimdgroups(gpu::GPUModuleOp m) {
  SmallVector<gpu::AllReduceOp> ops;
  m.walk([&](gpu::AllReduceOp op) { ops.push_back(op); });
  for (gpu::AllReduceOp op : ops) {
    if (!op.getUniform() || !op.getOp() || !op.getBody().empty())
      continue;
    auto func = op->getParentOfType<gpu::GPUFuncOp>();
    std::optional<uint64_t> threads = knownThreadgroupThreads(op);
    if (!func || !threads || *threads == 0 || *threads % 32 ||
        *threads > 1024)
      continue;
    Type type = op.getType();
    if (!type.isF32() && !type.isF16() && !type.isInteger(32))
      continue;
    gpu::AllReduceOperation kind = *op.getOp();
    TypedAttr identity;
    OpBuilder b(op);
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
    default:
      continue;
    }
    Location loc = op.getLoc();
    uint64_t groups = *threads / 32;
    Value partial = gpu::SubgroupReduceOp::create(b, loc, op.getValue(), kind,
                                                  /*uniform=*/true);
    if (groups == 1) {
      op.replaceAllUsesWith(partial);
      op.erase();
      continue;
    }
    auto bufferType = MemRefType::get(
        {int64_t(groups)}, type, MemRefLayoutAttrInterface(),
        gpu::AddressSpaceAttr::get(b.getContext(),
                                   gpu::AddressSpace::Workgroup));
    Value buffer = func.addWorkgroupAttribution(bufferType, loc);
    Value lane = gpu::LaneIdOp::create(b, loc, /*upper_bound=*/IntegerAttr());
    Value group =
        gpu::SubgroupIdOp::create(b, loc, /*upper_bound=*/IntegerAttr());
    Value zero = arith::ConstantIndexOp::create(b, loc, 0);
    Value first =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, zero);
    scf::IfOp::create(b, loc, first, [&](OpBuilder &b, Location loc) {
      memref::StoreOp::create(b, loc, partial, buffer, group);
      scf::YieldOp::create(b, loc);
    });
    gpu::BarrierOp::create(b, loc);
    Value inRange = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, lane,
        arith::ConstantIndexOp::create(b, loc, groups));
    Value index = arith::SelectOp::create(b, loc, inRange, lane, zero);
    Value value = memref::LoadOp::create(b, loc, buffer, index);
    if (identity)
      value = arith::SelectOp::create(
          b, loc, inRange, value,
          arith::ConstantOp::create(b, loc, identity));
    Value result =
        gpu::SubgroupReduceOp::create(b, loc, value, kind, /*uniform=*/true);
    if (mayRepeat(op))
      gpu::BarrierOp::create(b, loc);
    op.replaceAllUsesWith(result);
    op.erase();
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
    if (narrowLoops)
      ::narrowLoops(m);
    if (narrowIndices)
      narrowMemRefIndices(m, linearizeIndices);
    if (simdgroupAllReduce)
      lowerAllReducesToSimdgroups(m);
    // In-dialect lowering of gpu.all_reduce and friends first.
    {
      RewritePatternSet patterns(ctx);
      populateGpuAllReducePatterns(patterns);
      populateGpuShufflePatterns(patterns);
      (void)applyPatternsGreedily(m, std::move(patterns));
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
