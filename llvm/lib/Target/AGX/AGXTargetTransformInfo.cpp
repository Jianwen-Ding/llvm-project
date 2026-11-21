//===-- AGXTargetTransformInfo.cpp - AGX specific TTI ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXTargetTransformInfo.h"
#include "AGX.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"

using namespace llvm;

#define DEBUG_TYPE "agxtti"

static cl::opt<unsigned> MaxStoreVectorBytes(
    "agx-max-store-vector-bytes", cl::init(16), cl::Hidden,
    cl::desc("The widest store the load/store vectorizer makes (bytes)"));

static cl::opt<int> StorePressureLimit(
    "agx-store-vector-pressure", cl::init(-1), cl::Hidden,
    cl::desc("Merge no stores in functions whose estimated register "
             "pressure (32-bit registers) is at least this (0: no limit; "
             "-1, the default: the general purpose registers of the "
             "subtarget)"));

/// The most 32-bit registers that IR values hold at once in F: SSA liveness,
/// each value by its size (at least one register).
unsigned llvm::agxEstimatePressure(const Function &F) {
  const DataLayout &DL = F.getDataLayout();
  auto Size = [&](const Value *V) -> unsigned {
    Type *T = V->getType();
    if (T->isVoidTy() || T->isLabelTy() || T->isTokenTy() || T->isMetadataTy())
      return 0;
    return std::max<uint64_t>(1, divideCeil(DL.getTypeSizeInBits(T), 32));
  };
  auto Tracked = [](const Value *V) {
    return isa<Instruction>(V) || isa<Argument>(V);
  };
  // Live-in sets by backward dataflow over the blocks.
  DenseMap<const BasicBlock *, SmallPtrSet<const Value *, 32>> LiveIn;
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (const BasicBlock &BB : reverse(F)) {
      SmallPtrSet<const Value *, 32> Live;
      for (const BasicBlock *Succ : successors(&BB)) {
        for (const Value *V : LiveIn[Succ])
          Live.insert(V);
        for (const PHINode &P : Succ->phis()) {
          Live.erase(&P);
          if (const Value *In = P.getIncomingValueForBlock(&BB); Tracked(In))
            Live.insert(In);
        }
      }
      for (const Instruction &I : reverse(BB)) {
        Live.erase(&I);
        if (!isa<PHINode>(I))
          for (const Value *Op : I.operands())
            if (Tracked(Op))
              Live.insert(Op);
      }
      auto &Old = LiveIn[&BB];
      if (Old.size() != Live.size()) {
        Old = Live;
        Changed = true;
      }
    }
  }
  unsigned Max = 0;
  for (const BasicBlock &BB : F) {
    SmallPtrSet<const Value *, 32> Live;
    for (const BasicBlock *Succ : successors(&BB)) {
      for (const Value *V : LiveIn[Succ])
        Live.insert(V);
      for (const PHINode &P : Succ->phis()) {
        Live.erase(&P);
        if (const Value *In = P.getIncomingValueForBlock(&BB); Tracked(In))
          Live.insert(In);
      }
    }
    unsigned Cur = 0;
    for (const Value *V : Live)
      Cur += Size(V);
    Max = std::max(Max, Cur);
    for (const Instruction &I : reverse(BB)) {
      if (Live.erase(&I))
        Cur -= Size(&I);
      if (!isa<PHINode>(I))
        for (const Value *Op : I.operands())
          if (Tracked(Op) && Live.insert(Op).second)
            Cur += Size(Op);
      Max = std::max(Max, Cur);
    }
  }
  return Max;
}

unsigned AGXTTIImpl::maxStoreVectorBytes() const {
  // A kernel whose values nearly fill the register file: the vector
  // stores' consecutive registers cost copies and spills (16 stores merged
  // into 4: 122 -> 128 GPRs and spills, +300%; sgemm_reg4x4 -28% tput
  // without them), more than the issue slots they save.
  unsigned Limit = StorePressureLimit < 0 ? ST->getMaxGPRs()
                                          : unsigned(StorePressureLimit);
  if (Limit) {
    if (!Pressure)
      Pressure = agxEstimatePressure(*Fn);
    if (*Pressure >= Limit)
      return 4;
  }
  return MaxStoreVectorBytes;
}

static cl::opt<unsigned> AGXThreadgroupUnroll(
    "agx-threadgroup-unroll", cl::init(0), cl::Hidden,
    cl::desc("Most iterations a loop that loads threadgroup memory is "
             "unrolled by, so that its loads can be merged (0: as far as "
             "contiguous loads become 16-byte ones, 1: no unrolling)"));

static cl::opt<unsigned> AGXDeviceUnrollNarrow(
    "agx-device-unroll-narrow", cl::init(32), cl::Hidden,
    cl::desc("The factor of agx-device-unroll for loops whose device loads "
             "are all of one or two bytes (0: the same)"));

static cl::opt<unsigned> AGXDeviceUnrollStore(
    "agx-device-unroll-store", cl::init(16), cl::Hidden,
    cl::desc("The factor of agx-device-unroll for loops that also store to "
             "device memory, where it divides the trip count (0: the same)"));

static cl::opt<unsigned> AGXDeviceUnrollStrided(
    "agx-device-unroll-strided", cl::init(16), cl::Hidden,
    cl::desc("The factor of agx-device-unroll for loops whose device loads "
             "all step more than 64 bytes an iteration, where it divides the "
             "trip count (0: the same)"));

static cl::opt<bool> AGXDeviceUnrollCarriedSelect(
    "agx-device-unroll-carried-select", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-argmax also for a carried value kept by "
             "selects on one compare"));

static cl::opt<unsigned> AGXDeviceUnrollArgMax(
    "agx-device-unroll-argmax", cl::init(16), cl::Hidden,
    cl::desc("Iterations a device-load arg-max / arg-min loop (a carried "
             "value selected on a compare against it, and an index) is "
             "unrolled by (0: no such rule)"));

static cl::opt<unsigned> AGXDeviceUnrollFullMaxSize(
    "agx-device-unroll-full-max-size", cl::init(1024), cl::Hidden,
    cl::desc("Trips times the loop's instructions above which "
             "-agx-device-unroll-full-trips does not unroll it whole"));

static cl::opt<unsigned> AGXDeviceUnrollFullTrips(
    "agx-device-unroll-full-trips", cl::init(64), cl::Hidden,
    cl::desc("Device-load loops of at most this many trips (not running "
             "maxima) unrolled whole (0: not)"));

static cl::opt<unsigned> AGXDeviceUnrollTranscendental(
    "agx-device-unroll-transcendental", cl::init(8), cl::Hidden,
    cl::desc("Unroll factor of device-load loops with exponentials that no "
             "other factor takes (0: the default)"));

static cl::opt<unsigned> AGXDeviceUnrollRuntime(
    "agx-device-unroll-runtime", cl::init(4), cl::Hidden,
    cl::desc("Unroll a light device-load loop of a runtime trip count this "
             "many times, with a remainder loop (0: not)"));

static cl::opt<unsigned> AGXDeviceUnrollRuntimeStrided(
    "agx-device-unroll-runtime-strided", cl::init(8), cl::Hidden,
    cl::desc("agx-device-unroll-runtime's factor where its far loads step by "
             "a runtime stride (not a gather)"));

static cl::opt<bool> AGXDeviceUnrollSomeStrided(
    "agx-device-unroll-some-strided", cl::init(true), cl::Hidden,
    cl::desc("agx-device-unroll-strided's factor also where only some of a "
             "loop's device loads are strided"));

static cl::opt<unsigned> AGXDeviceUnrollRuntimeContiguous(
    "agx-device-unroll-runtime-contiguous", cl::init(4), cl::Hidden,
    cl::desc("Iterations a runtime-trip loop over a thread's row (contiguous "
             "loads, no stores, light arithmetic) is unrolled by, twice that "
             "for one load an iteration (0: none)"));

static cl::opt<bool> AGXDeviceUnrollRuntimeStoresNarrow(
    "agx-device-unroll-runtime-stores-narrow", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores also for 1- and 2-byte "
             "accesses (and roundeven)"));

static cl::opt<bool> AGXDeviceUnrollRuntimeExp(
    "agx-device-unroll-runtime-exp", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-contiguous also for a thread's row "
             "of exponentials"));

static cl::opt<bool> AGXDeviceUnrollRuntimeExpStores(
    "agx-device-unroll-runtime-exp-stores", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores also for exponentials"));

static cl::opt<unsigned> AGXDeviceUnrollBlocks(
    "agx-device-unroll-blocks", cl::init(3), cl::Hidden,
    cl::desc("Blocks an innermost device-load loop with a conditional device "
             "store may have for E187's factors (0: one block only)"));

static cl::opt<bool> AGXDeviceUnrollRuntimeColumns(
    "agx-device-unroll-runtime-columns", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores also for far accesses where "
             "the trip count does not depend on the thread (a column)"));

static cl::opt<bool> AGXDeviceUnrollRuntimeTanhStores(
    "agx-device-unroll-runtime-tanh-stores", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores also for tanh"));

static cl::opt<bool> AGXDeviceUnrollRuntimeDivStores(
    "agx-device-unroll-runtime-div-stores", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores also for divisions"));

static cl::opt<bool> AGXDeviceUnrollRuntimeMathStores(
    "agx-device-unroll-runtime-math-stores", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores also for log, log2, sin, cos "
             "and sqrt"));

static cl::opt<bool> AGXDeviceUnrollRuntimeNarrowStores(
    "agx-device-unroll-runtime-narrow-stores", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores 4 times for a row of halves "
             "or bytes"));

static cl::opt<bool> AGXDeviceUnrollRuntimeColumnStores(
    "agx-device-unroll-runtime-column-stores", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-stores twice for a column"));

static cl::opt<bool> AGXDeviceUnrollRuntimeMax(
    "agx-device-unroll-runtime-max", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-contiguous also for a thread's row "
             "maximum or minimum"));

static cl::opt<bool> AGXDeviceUnrollRuntimeNarrowRows(
    "agx-device-unroll-runtime-narrow-rows", cl::init(true), cl::Hidden,
    cl::desc("-agx-device-unroll-runtime-contiguous's factor times 4 over "
             "the bytes of a row's narrowest load (halves, bytes)"));

static cl::opt<unsigned> AGXDeviceUnrollRuntimeStores(
    "agx-device-unroll-runtime-stores", cl::init(4), cl::Hidden,
    cl::desc("Iterations a runtime-trip row loop of word loads and stores and "
             "light arithmetic is unrolled by, with a remainder (0: none)"));

static cl::opt<unsigned> AGXDeviceUnrollStoreLight(
    "agx-device-unroll-store-light", cl::init(8), cl::Hidden,
    cl::desc("Iterations a device-load loop that stores words and loads at "
             "most two words an iteration is unrolled by (0: the store "
             "factor)"));

static cl::opt<unsigned> AGXDeviceUnrollStreams(
    "agx-device-unroll-streams", cl::init(8), cl::Hidden,
    cl::desc("The factor of agx-device-unroll for loops whose device loads "
             "step through two or more pointer bases, without stores, where "
             "it divides the trip count (0: the same)"));

static cl::opt<unsigned> AGXDeviceUnrollStreamsWord(
    "agx-device-unroll-streams-word", cl::init(16), cl::Hidden,
    cl::desc("agx-device-unroll-streams' factor where every device load is "
             "of 4 bytes at most (0: the same)"));

static cl::opt<bool> AGXDeviceUnrollStreamsVectors(
    "agx-device-unroll-streams-vectors", cl::init(true), cl::Hidden,
    cl::desc("agx-device-unroll-streams only for loops whose device loads are "
             "all vectors"));

static cl::opt<bool> AGXDeviceUnrollKeepLoop(
    "agx-device-unroll-keep-loop", cl::init(true), cl::Hidden,
    cl::desc("A device-load loop of a loop-carried maximum / minimum and more "
             "than 16 trips stays a loop: its factor at most half the trip "
             "count, no full unrolling"));

static cl::opt<unsigned> AGXDeviceUnroll(
    "agx-device-unroll", cl::init(4), cl::Hidden,
    cl::desc("Iterations a single-block loop that loads device memory, of a "
             "constant trip count this divides, is unrolled by (1: none)"));

static bool isThreadVaryingIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::agx_thread_position_in_grid_x:
  case Intrinsic::agx_thread_position_in_grid_y:
  case Intrinsic::agx_thread_position_in_grid_z:
  case Intrinsic::agx_thread_position_in_threadgroup_x:
  case Intrinsic::agx_thread_position_in_threadgroup_y:
  case Intrinsic::agx_thread_position_in_threadgroup_z:
  case Intrinsic::agx_thread_index_in_threadgroup:
  case Intrinsic::agx_thread_index_in_simdgroup:
  case Intrinsic::agx_simdgroup_index_in_threadgroup:
  case Intrinsic::agx_active_thread_index_in_simdgroup:
  case Intrinsic::agx_active_thread_index_in_quadgroup:
  case Intrinsic::agx_read_sr:
  case Intrinsic::agx_simd_shuffle:
  case Intrinsic::agx_simd_shuffle_xor:
  case Intrinsic::agx_simd_shuffle_up:
  case Intrinsic::agx_simd_shuffle_down:
  case Intrinsic::agx_quad_shuffle:
  case Intrinsic::agx_quad_shuffle_xor:
  case Intrinsic::agx_quad_shuffle_up:
  case Intrinsic::agx_quad_shuffle_down:
  case Intrinsic::agx_simd_shuffle_rotate_up:
  case Intrinsic::agx_quad_shuffle_rotate_up:
  case Intrinsic::agx_simd_prefix:
  case Intrinsic::agx_quad_prefix:
  case Intrinsic::agx_quad_reduce:
  case Intrinsic::agx_quad_ballot:
  case Intrinsic::agx_dfdx:
  case Intrinsic::agx_dfdy:
  case Intrinsic::agx_simdgroup_matrix_fma:
  case Intrinsic::agx_simd_matrix_fmadd32:
  case Intrinsic::agx_simd_matrix_fmadd16:
    return true;
  default:
    return false;
  }
}

ValueUniformity AGXTTIImpl::getValueUniformity(const Value *V) const {
  if (const auto *II = dyn_cast<IntrinsicInst>(V)) {
    if (isThreadVaryingIntrinsic(II->getIntrinsicID()))
      return ValueUniformity::NeverUniform;
    switch (II->getIntrinsicID()) {
    case Intrinsic::agx_simd_ballot:
    case Intrinsic::agx_simd_reduce:
    case Intrinsic::agx_total_active_threads_in_simdgroup:
    case Intrinsic::agx_threadgroup_position_in_grid_x:
    case Intrinsic::agx_threadgroup_position_in_grid_y:
    case Intrinsic::agx_threadgroup_position_in_grid_z:
    case Intrinsic::agx_threads_per_threadgroup_x:
    case Intrinsic::agx_threads_per_threadgroup_y:
    case Intrinsic::agx_threads_per_threadgroup_z:
    case Intrinsic::agx_threadgroups_per_grid_x:
    case Intrinsic::agx_threadgroups_per_grid_y:
    case Intrinsic::agx_threadgroups_per_grid_z:
      return ValueUniformity::AlwaysUniform;
    default:
      return ValueUniformity::Default;
    }
  }

  // Stack memory is per thread, and atomics return per thread values.
  if (const auto *LI = dyn_cast<LoadInst>(V))
    return LI->getPointerAddressSpace() == AGX::PRIVATE_ADDRESS
               ? ValueUniformity::NeverUniform
               : ValueUniformity::Default;
  if (isa<AtomicRMWInst>(V) || isa<AtomicCmpXchgInst>(V) || isa<CallInst>(V))
    return ValueUniformity::NeverUniform;
  return ValueUniformity::Default;
}

// A shuffle of a constant reads the same value in every lane. (Reading an
// inactive lane is undefined, so the constant is a valid result there too.)
std::optional<Instruction *>
AGXTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  switch (II.getIntrinsicID()) {
  case Intrinsic::agx_simd_shuffle:
  case Intrinsic::agx_simd_shuffle_xor:
  case Intrinsic::agx_simd_shuffle_up:
  case Intrinsic::agx_simd_shuffle_down:
  case Intrinsic::agx_quad_shuffle:
  case Intrinsic::agx_quad_shuffle_xor:
  case Intrinsic::agx_quad_shuffle_up:
  case Intrinsic::agx_quad_shuffle_down:
  case Intrinsic::agx_simd_shuffle_rotate_up:
  case Intrinsic::agx_quad_shuffle_rotate_up:
    if (isa<Constant>(II.getArgOperand(0)))
      return IC.replaceInstUsesWith(II, II.getArgOperand(0));
    break;
  default:
    break;
  }
  return std::nullopt;
}

// A threadgroup load costs about the same at any width (one to four 32-bit
// values: Geekbench 7's gemm_block_size_bias on the M2 Max takes 11-12 us per
// load of its inner loop whether it is x or xy, and the same with half the
// bytes). Unrolling loops that load threadgroup memory could let the
// load/store vectorizer merge the loads of neighbouring iterations into wider
// ones. Off by default: the shared addresses of PTX are integers turned into
// pointers, which the vectorizer does not see through, and the unrolled
// gemm_block_size_bias needs 121 instead of 98 registers and is 28% slower.
/// The address of a threadgroup load as SCEV understands it: the pointer (a
/// getelementptr of a variable), else the integer PTX computed it in
/// ((zext) X -> inttoptr -> constant offsets, plus those offsets).
static const SCEV *threadgroupAddress(const LoadInst *LI, ScalarEvolution &SE) {
  Value *Ptr = const_cast<Value *>(LI->getPointerOperand());
  const SCEV *S = SE.getSCEV(Ptr);
  if (isa<SCEVAddRecExpr>(S))
    return S;
  const DataLayout &DL = LI->getDataLayout();
  APInt Offset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
  const Value *P =
      Ptr->stripAndAccumulateConstantOffsets(DL, Offset, /*AllowNonInbounds=*/true);
  auto *I2P = dyn_cast<IntToPtrInst>(P);
  if (!I2P)
    return nullptr;
  Value *X = I2P->getOperand(0);
  if (auto *Z = dyn_cast<ZExtInst>(X))
    X = Z->getOperand(0);
  if (!SE.isSCEVable(X->getType()))
    return nullptr;
  const SCEV *XS = SE.getSCEV(X);
  return SE.getAddExpr(
      XS, SE.getConstant(X->getType(), Offset.getSExtValue(), true));
}

/// The unroll factor after which the threadgroup loads of L that read
/// consecutive bytes every iteration (loads at a, a + 4, ... covering the
/// 4 or 8 bytes by which a advances) cover 16: xyzw loads, once the
/// load/store vectorizer joins the copies (AGXThreadgroupAddresses gives
/// their addresses one base). 1 if there are none.
static unsigned contiguousThreadgroupFactor(Loop *L, ScalarEvolution &SE) {
  SmallVector<std::pair<const SCEV *, uint64_t>, 32> Loads;
  for (const BasicBlock *BB : L->blocks())
    for (const Instruction &I : *BB)
      if (auto *LI = dyn_cast<LoadInst>(&I))
        if (LI->getPointerAddressSpace() == AGX::THREADGROUP_ADDRESS)
          if (const SCEV *A = threadgroupAddress(LI, SE))
            Loads.push_back(
                {A, LI->getDataLayout()
                        .getTypeStoreSize(LI->getType())
                        .getFixedValue()});
  unsigned Factor = 1;
  for (auto [A, Size] : Loads) {
    LLVM_DEBUG(dbgs() << "  threadgroup load " << *A << " size " << Size
                      << "\n");
    auto *AR = dyn_cast<SCEVAddRecExpr>(A);
    if (!AR || AR->getLoop() != L || !AR->isAffine())
      continue;
    auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    if (!Step || (Step->getAPInt() != 4 && Step->getAPInt() != 8))
      continue;
    uint64_t S = Step->getAPInt().getZExtValue();
    // follow the loads of the next bytes until the step is covered
    uint64_t Covered = Size, Last = Size;
    const SCEV *Next = A;
    while (Covered < S) {
      Next = SE.getAddExpr(Next, SE.getConstant(A->getType(), Last));
      auto *It = find_if(Loads, [&](auto &E) { return E.first == Next; });
      if (It == Loads.end())
        break;
      Last = It->second;
      Covered += Last;
    }
    if (Covered == S)
      Factor = std::max<unsigned>(Factor, 16 / S);
  }
  return Factor;
}

/// The bytes a device load's address steps each iteration of L, if constant:
/// an add recurrence of L, or one of the 32-bit index a GEP extends (as
/// MLIR's linearized indices are, whose additions carry no wrap flags).
static std::optional<APInt> deviceLoadStride(const LoadInst &LI, const Loop *L,
                                             ScalarEvolution &SE,
                                             const DataLayout &DL) {
  Value *P = const_cast<Value *>(LI.getPointerOperand());
  const SCEV *X = SE.getSCEV(P);
  uint64_t Scale = 1;
  if (auto *G = dyn_cast<GetElementPtrInst>(P); G && G->getNumIndices() == 1) {
    Value *Idx = G->getOperand(1);
    if (isa<ZExtInst>(Idx) || isa<SExtInst>(Idx))
      Idx = cast<CastInst>(Idx)->getOperand(0);
    X = SE.getSCEV(Idx);
    Scale = DL.getTypeAllocSize(G->getSourceElementType());
  }
  auto *AR = dyn_cast<SCEVAddRecExpr>(X);
  if (!AR || AR->getLoop() != L)
    return std::nullopt;
  auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
  if (!Step)
    return std::nullopt;
  return Step->getAPInt().sextOrTrunc(64).abs() * Scale;
}

/// E187's factor for a single-block loop of Trips iterations that loads
/// device memory (DeviceLoads), or 0 if it does not: 4, or where the factor
/// divides the trip count, 16 for loads of one or two bytes (a half or int8
/// row, an iteration's work small: L15-E234), 16 for a loop that also
/// stores to device memory (a copy: L15-E235) and 8 for one whose loads all
/// step more than 64 bytes (a column: L15-E235; 16 costs occupancy there).
static unsigned deviceUnrollFactor(const Loop *L, ScalarEvolution &SE,
                                   const DataLayout &DL, unsigned Trips,
                                   bool &DeviceLoads) {
  bool Narrow = true, Stores = false, Strided = true, Word = true,
       Vectors = true, SomeStrided = false, Store32 = true, Light = true;
  unsigned NumLoads = 0;
  DeviceLoads = false;
  // The pointer bases the device loads step through (a dot product's two
  // rows): L15-E506.
  SmallPtrSet<const SCEV *, 4> Streams;
  for (const BasicBlock *BB : L->blocks())
    for (const Instruction &I : *BB) {
      if (auto *SI = dyn_cast<StoreInst>(&I);
          SI && SI->getPointerAddressSpace() == AGX::DEVICE_ADDRESS) {
        Stores = true;
        Store32 &= DL.getTypeStoreSize(SI->getValueOperand()->getType()) == 4;
      }
      // (light arithmetic: no call but a multiply-add, a maximum / minimum
      // or an absolute value)
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        auto *II = dyn_cast<IntrinsicInst>(CB);
        Light &= II && (II->getIntrinsicID() == Intrinsic::fma ||
                        II->getIntrinsicID() == Intrinsic::fmuladd ||
                        II->getIntrinsicID() == Intrinsic::maxnum ||
                        II->getIntrinsicID() == Intrinsic::minnum ||
                        II->getIntrinsicID() == Intrinsic::maximum ||
                        II->getIntrinsicID() == Intrinsic::minimum ||
                        II->getIntrinsicID() == Intrinsic::fabs);
      }
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI || LI->getPointerAddressSpace() != AGX::DEVICE_ADDRESS)
        continue;
      DeviceLoads = true;
      ++NumLoads;
      Narrow &= DL.getTypeStoreSize(LI->getType()) <= 2;
      Word &= DL.getTypeStoreSize(LI->getType()) <= 4;
      Vectors &= LI->getType()->isVectorTy();
      if (const SCEV *P = SE.getSCEV(const_cast<Value *>(LI->getPointerOperand()));
          isa<SCEVAddRecExpr>(P) &&
          cast<SCEVAddRecExpr>(P)->getLoop() == L)
        Streams.insert(SE.getPointerBase(P));
      {
        std::optional<APInt> S = deviceLoadStride(*LI, L, SE, DL);
        bool Far = S && S->ugt(64);
        Strided &= Far;
        SomeStrided |= Far;
      }
    }
  auto Divides = [&](unsigned F) { return F && Trips >= F && Trips % F == 0; };
  if (!DeviceLoads)
    return 0;
  if (Narrow && Divides(AGXDeviceUnrollNarrow))
    return AGXDeviceUnrollNarrow;
  // (a row loop of at most two word loads, word stores and light
  // arithmetic, 256 trips or more: 16 iterations' loads held 32 registers,
  // 8 keep the groups, L15-E564; byte stores, three loads, transcendental
  // or shared-unit work, a column and 128 trips lost at 8)
  if (AGXDeviceUnrollStoreLight && !Narrow && Stores && Store32 && Word &&
      Light && !SomeStrided && NumLoads <= 2 && Trips >= 256 &&
      Divides(AGXDeviceUnrollStoreLight))
    return AGXDeviceUnrollStoreLight;
  if (!Narrow && Stores && Divides(AGXDeviceUnrollStore))
    return AGXDeviceUnrollStore;
  if (!Narrow && !Stores && Strided && Divides(AGXDeviceUnrollStrided))
    return AGXDeviceUnrollStrided;
  // (a strided load beside contiguous ones: a naive matrix product's column
  // of B beside A's row, L15-E535)
  if (AGXDeviceUnrollSomeStrided && !Narrow && !Stores && SomeStrided &&
      Divides(AGXDeviceUnrollStrided))
    return AGXDeviceUnrollStrided;
  // Two or more streams (no stores): a dot product's iteration is two loads
  // and a multiply-add per element, so 4 iterations' loads still leave it
  // waiting; one stream (a row sum, maximum or softmax) keeps 4.
  // (loads of a word at most: 16, as their iterations' loads are a fourth
  // of a 16-byte vector's, L15-E507)
  if (!Narrow && !Stores && Streams.size() >= 2 && Word &&
      (Vectors || !AGXDeviceUnrollStreamsVectors) &&
      Divides(AGXDeviceUnrollStreamsWord))
    return AGXDeviceUnrollStreamsWord;
  if (!Narrow && !Stores && Streams.size() >= 2 &&
      (Vectors || !AGXDeviceUnrollStreamsVectors) &&
      Divides(AGXDeviceUnrollStreams))
    return AGXDeviceUnrollStreams;
  // An arg-max / arg-min (a carried value chosen by a select on a compare
  // against it, the index beside it): its compare-selects of 4 iterations
  // leave it waiting on each 16-byte load; 16 keep 4 in flight
  // (argmax_rows, L15-E552; a sum's carried adds lost, rowsum +16%).
  if (AGXDeviceUnrollArgMax && !Narrow && !Stores &&
      any_of(L->getHeader()->phis(), [](const PHINode &P) {
        return any_of(P.users(), [&](const User *U) {
          auto *Sel = dyn_cast<SelectInst>(U);
          auto *C = Sel ? dyn_cast<CmpInst>(Sel->getCondition()) : nullptr;
          // (the compare also selects the index; or, a conditional count
          // and sum, the value kept where the compare fails: L15-E555)
          return C && !C->hasOneUse() &&
                 (C->getOperand(0) == &P || C->getOperand(1) == &P ||
                  (AGXDeviceUnrollCarriedSelect &&
                   (Sel->getTrueValue() == &P ||
                    Sel->getFalseValue() == &P)));
        });
      }) &&
      Divides(AGXDeviceUnrollArgMax))
    return AGXDeviceUnrollArgMax;
  // (a loop of exponentials: their work between the loads takes an unroll
  // of 8 without the registers that cost groups, online_softmax_vec4,
  // L15-E542)
  // (with the constants in uniforms only: their registers held, the
  // unroll cost more than it hid, online_softmax_vec4 +0.9%)
  if (AGXDeviceUnrollTranscendental && AGX::uniformConstantsEnabled() &&
      any_of(L->blocks(), [](const BasicBlock *BB) {
        return any_of(*BB, [](const Instruction &I) {
          auto *II = dyn_cast<IntrinsicInst>(&I);
          return II && (II->getIntrinsicID() == Intrinsic::exp ||
                        II->getIntrinsicID() == Intrinsic::exp2);
        });
      }) &&
      Divides(AGXDeviceUnrollTranscendental))
    return AGXDeviceUnrollTranscendental;
  return AGXDeviceUnroll;
}

void AGXTTIImpl::getUnrollingPreferences(
    Loop *L, ScalarEvolution &SE, TargetTransformInfo::UnrollingPreferences &UP,
    OptimizationRemarkEmitter *ORE) const {
  BaseT::getUnrollingPreferences(L, SE, UP, ORE);
  bool ThreadgroupLoads = any_of(L->blocks(), [](const BasicBlock *BB) {
    return any_of(*BB, [](const Instruction &I) {
      auto *LI = dyn_cast<LoadInst>(&I);
      return LI && LI->getPointerAddressSpace() == AGX::THREADGROUP_ADDRESS;
    });
  });
  // (or a few blocks around a conditional store, y[j] = x[j] where x[j] > t:
  // its loads one an iteration otherwise, L15-E572)
  if (!ThreadgroupLoads &&
      (L->getNumBlocks() == 1 ||
       (AGXDeviceUnrollBlocks && L->isInnermost() &&
        L->getNumBlocks() <= AGXDeviceUnrollBlocks &&
        any_of(L->blocks(), [&](const BasicBlock *BB) {
          return BB != L->getHeader() && any_of(*BB, [](const Instruction &I) {
                   auto *SI = dyn_cast<StoreInst>(&I);
                   return SI &&
                          SI->getPointerAddressSpace() == AGX::DEVICE_ADDRESS;
                 });
        })))) {
    // A loop that loads device memory, with a constant trip count the factor
    // divides (no remainder loop: static shapes): unrolled, its loads of
    // several iterations are in flight at once instead of one wait each
    // (L15-E187; runtime trip counts lose, R12 / R20).
    unsigned Trips = SE.getSmallConstantTripCount(L);
    bool DeviceLoads = false;
    unsigned F = deviceUnrollFactor(L, SE, DL, Trips, DeviceLoads);
    // A loop-carried maximum / minimum (whose selects
    // deferredMinMaxReductions rewrites only in a loop) stays a loop: not
    // unrolled whole, by the factor or LoopUnroll's own full unrolling
    // (L15-E511).
    bool MinMax =
        AGXDeviceUnrollKeepLoop && Trips > 16 &&
        any_of(L->getHeader()->phis(), [](const PHINode &P) {
          return any_of(P.users(), [](const User *U) {
            auto *II = dyn_cast<IntrinsicInst>(U);
            return II && (II->getIntrinsicID() == Intrinsic::maximum ||
                          II->getIntrinsicID() == Intrinsic::minimum);
          });
        });
    // (Short loops, 16 trips at most, still whole: newk's logsoftmax and
    // softmax_f16 of 4 lost as loops.)
    if (MinMax && F >= Trips && Trips > 16)
      F = Trips / 2;
    // (a short loop, at most 32 trips: whole, so that a later pass over the
    // same data, also whole, takes its loads' and exponentials' values:
    // L15-E545, softmax_f32v's third pass)
    // (not a running maxnum / minnum either: online_softmax_exp2_maxnum lost)
    bool NumMinMax = any_of(L->getHeader()->phis(), [](const PHINode &P) {
      return any_of(P.users(), [](const User *U) {
        auto *II = dyn_cast<IntrinsicInst>(U);
        return II && (II->getIntrinsicID() == Intrinsic::maxnum ||
                      II->getIntrinsicID() == Intrinsic::minnum ||
                      II->getIntrinsicID() == Intrinsic::maximum ||
                      II->getIntrinsicID() == Intrinsic::minimum);
      });
    });
    // (and not past -agx-device-unroll-full-max-size instructions whole:
    // a thread's 64-float row softmax, 64 trips, took 2521 machine
    // instructions, L15-E551)
    if (!MinMax && !NumMinMax && DeviceLoads && Trips &&
        Trips <= AGXDeviceUnrollFullTrips && F >= 2 && F < Trips &&
        uint64_t(Trips) * L->getHeader()->size() <= AGXDeviceUnrollFullMaxSize)
      F = Trips;

    if (DeviceLoads && F >= 2 && Trips >= F && Trips % F == 0) {
      UP.Partial = true;
      UP.MaxCount = F;
      UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
      if (MinMax)
        UP.FullUnrollMaxCount = F;
    }
    // A runtime trip count: a loop of light work per load and no stores
    // (a matrix product's dot over K) unrolled with a remainder loop, its
    // loads of several iterations in flight (L15-E536). Loops of
    // transcendental work hide their loads behind it, and lost (xent_rows).
    if (!Trips && DeviceLoads && AGXDeviceUnrollRuntime >= 2) {
      unsigned Insts = 0, Far = 0;
      bool Light = true, Gather = false, Exps = false, Other = false;
      for (const Instruction &I : *L->getHeader()) {
        ++Insts;
        if (isa<StoreInst>(I) || (isa<CallBase>(I) && !isa<IntrinsicInst>(I)))
          Light = false;
        // (a load that does not step by its own size: a column, a gather;
        // contiguous loads' unrolled copies (an FIR's taps) only took
        // registers)
        if (auto *LI = dyn_cast<LoadInst>(&I);
            LI && LI->getPointerAddressSpace() == AGX::DEVICE_ADDRESS) {
          // (the pointer's step, or its GEP index's, an extended 32-bit one,
          // times the element's size)
          auto StepOf = [&](const SCEV *X) -> std::optional<APInt> {
            auto *AR = dyn_cast<SCEVAddRecExpr>(X);
            auto *C = AR && AR->getLoop() == L
                          ? dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))
                          : nullptr;
            if (!C)
              return std::nullopt;
            return C->getAPInt().sextOrTrunc(64);
          };
          const Value *Ptr = LI->getPointerOperand();
          std::optional<APInt> Bytes = StepOf(SE.getSCEV(const_cast<Value *>(Ptr)));
          if (auto *G = dyn_cast<GetElementPtrInst>(Ptr);
              !Bytes && G && G->getNumIndices() == 1) {
            const Value *Idx = G->getOperand(1);
            if (auto *E = dyn_cast<CastInst>(Idx);
                E && (isa<ZExtInst>(E) || isa<SExtInst>(E)))
              Idx = E->getOperand(0);
            if (std::optional<APInt> St =
                    StepOf(SE.getSCEV(const_cast<Value *>(Idx))))
              Bytes = *St * DL.getTypeAllocSize(G->getSourceElementType());
          }
          Far += !Bytes || Bytes->abs().ugt(DL.getTypeStoreSize(LI->getType()));
          // (a gather: an address that is no recurrence of the loop at all)
          if (!Bytes) {
            const SCEV *PS = SE.getSCEV(const_cast<Value *>(Ptr));
            bool Rec = isa<SCEVAddRecExpr>(PS);
            if (auto *G = dyn_cast<GetElementPtrInst>(Ptr);
                !Rec && G && G->getNumIndices() == 1) {
              const Value *Idx = G->getOperand(1);
              if (auto *E = dyn_cast<CastInst>(Idx);
                  E && (isa<ZExtInst>(E) || isa<SExtInst>(E)))
                Idx = E->getOperand(0);
              Rec = isa<SCEVAddRecExpr>(SE.getSCEV(const_cast<Value *>(Idx)));
            }
            Gather |= !Rec;
          }
        }
        // (light arithmetic only, and no running maximum / minimum, whose
        // deferral, L15-E444, wants the loop: the causal / window rows lost)
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          switch (II->getIntrinsicID()) {
          case Intrinsic::fma:
          case Intrinsic::fmuladd:
            break;
          case Intrinsic::exp:
          case Intrinsic::exp2:
            Exps = true;
            Light = false;
            break;
          default:
            Light = false;
            Other = true;
          }
        if (isa<CallBase>(I) && !isa<IntrinsicInst>(I))
          Other = true;
        if (I.getOpcode() == Instruction::FDiv || I.isAtomic())
          Light = false;
      }
      // (a column of a runtime length, y[i][c] = f(x[i][c], w[i]): its trip
      // count does not depend on the thread, as a grid-stride loop's
      // (n - gid) / grid and a SIMD-group row's (n - lane) / 32 do, L15-E575)
      auto ThreadFreeTrips = [&]() {
        if (!AGXDeviceUnrollRuntimeColumns)
          return false;
        const SCEV *BTC = SE.getBackedgeTakenCount(L);
        if (isa<SCEVCouldNotCompute>(BTC))
          return false;
        // (through the values SCEV does not see into, a lane's
        // (n - 1 - lane) computed before the loop)
        std::function<bool(const Value *, unsigned)> FromCall =
            [&](const Value *V, unsigned Depth) {
              if (isa<CallBase>(V))
                return true;
              auto *I = dyn_cast<Instruction>(V);
              if (!I || Depth > 12)
                return I != nullptr;
              return any_of(I->operands(), [&](const Use &U) {
                return FromCall(U.get(), Depth + 1);
              });
            };
        return !SCEVExprContains(BTC, [&](const SCEV *X) {
          auto *U = dyn_cast<SCEVUnknown>(X);
          return U && FromCall(U->getValue(), 0);
        });
      };
      // A row loop that stores (y[j] = f(x[j], w[j])): its contiguous word
      // accesses, light arithmetic, unrolled with a remainder so that alias
      // versioning's copy has iterations' loads to issue together
      // (L15-E565).
      if (!Light && (!Far || (!Gather && ThreadFreeTrips())) && Insts <= 24 &&
          AGXDeviceUnrollRuntimeStores >= 2 &&
          all_of(*L->getHeader(), [&](const Instruction &I) {
            auto Size = [&](Type *Ty) {
              unsigned B = DL.getTypeStoreSize(Ty);
              return AGXDeviceUnrollRuntimeStoresNarrow ? B <= 4 : B == 4;
            };
            if (auto *SI = dyn_cast<StoreInst>(&I))
              return SI->isSimple() &&
                     SI->getPointerAddressSpace() == AGX::DEVICE_ADDRESS &&
                     Size(SI->getValueOperand()->getType());
            if (auto *LI = dyn_cast<LoadInst>(&I))
              return LI->isSimple() && Size(LI->getType());
            if (auto *II = dyn_cast<IntrinsicInst>(&I))
              return II->getIntrinsicID() == Intrinsic::fma ||
                     II->getIntrinsicID() == Intrinsic::fmuladd ||
                     II->getIntrinsicID() == Intrinsic::maxnum ||
                     II->getIntrinsicID() == Intrinsic::minnum ||
                     II->getIntrinsicID() == Intrinsic::maximum ||
                     II->getIntrinsicID() == Intrinsic::minimum ||
                     II->getIntrinsicID() == Intrinsic::fabs ||
                     (AGXDeviceUnrollRuntimeStoresNarrow &&
                      II->getIntrinsicID() == Intrinsic::roundeven) ||
                     (AGXDeviceUnrollRuntimeExpStores &&
                      (II->getIntrinsicID() == Intrinsic::exp ||
                       II->getIntrinsicID() == Intrinsic::exp2)) ||
                     (AGXDeviceUnrollRuntimeTanhStores &&
                      II->getIntrinsicID() == Intrinsic::tanh) ||
                     (AGXDeviceUnrollRuntimeMathStores &&
                      (II->getIntrinsicID() == Intrinsic::log ||
                       II->getIntrinsicID() == Intrinsic::log2 ||
                       II->getIntrinsicID() == Intrinsic::sin ||
                       II->getIntrinsicID() == Intrinsic::cos ||
                       II->getIntrinsicID() == Intrinsic::sqrt));
            return !isa<CallBase>(I) && !I.isAtomic() &&
                   (I.getOpcode() != Instruction::FDiv ||
                    AGXDeviceUnrollRuntimeDivStores);
          })) {
        UP.Partial = UP.Runtime = true;
        // (4 times that for rows of halves or bytes, as E583's reductions:
        // quant_dyn 534.0 -> 414.1, widen_dyn 345.6 -> 285.8, hscale_dyn
        // 410.2 -> 280.6, twice 456.8 / 309.1 / 312.1, 8 times worse
        // (505.6 / 427.2 / 474.9); a row of floats lost at 8,
        // row_scale_dyn 453.9 -> 611.5, L15-E599, E600)
        bool NarrowRow =
            AGXDeviceUnrollRuntimeNarrowStores &&
            any_of(*L->getHeader(), [&](const Instruction &I) {
              return (isa<LoadInst>(I) || isa<StoreInst>(I)) &&
                     DL.getTypeStoreSize(
                         getLoadStoreType(&I)->getScalarType()) < 4;
            });
        // (and twice for a column, its accesses a row's pitch apart:
        // col_scale_dyn 533.5 -> 496.7, col_axpy_dyn 662.8 -> 602.4,
        // col_relu_dyn 602.1 -> 547.4, L15-E602)
        bool Column = AGXDeviceUnrollRuntimeColumnStores && Far;
        UP.DefaultUnrollRuntimeCount = UP.MaxCount =
            AGXDeviceUnrollRuntimeStores * (NarrowRow ? 4 : Column ? 2 : 1);
        UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
      }
      // A thread's own row of a runtime length (a row sum): its loads start
      // at the thread position times a row's bytes, each thread's lines
      // its own, unrolled with a remainder for their loads in flight
      // (L15-E566). Not a filter's taps (x[i + k], the threads' loads one
      // line: fir_taps lost, E536).
      auto RowPerThread = [&]() {
        for (const Instruction &I : *L->getHeader()) {
          auto *LI = dyn_cast<LoadInst>(&I);
          if (!LI || LI->getPointerAddressSpace() != AGX::DEVICE_ADDRESS)
            continue;
          auto *G = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
          if (!G || G->getNumIndices() != 1)
            continue;
          const Value *Idx = G->getOperand(1);
          if (auto *E = dyn_cast<CastInst>(Idx);
              E && (isa<ZExtInst>(E) || isa<SExtInst>(E)))
            Idx = E->getOperand(0);
          auto *AR = dyn_cast<SCEVAddRecExpr>(
              SE.getSCEV(const_cast<Value *>(Idx)));
          if (!AR || AR->getLoop() != L)
            continue;
          SmallVector<const SCEV *, 4> Terms;
          if (auto *A = dyn_cast<SCEVAddExpr>(AR->getStart()))
            Terms.append(A->operands().begin(), A->operands().end());
          else
            Terms.push_back(AR->getStart());
          for (const SCEV *T : Terms) {
            auto *M = dyn_cast<SCEVMulExpr>(T);
            if (!M || M->getNumOperands() != 2)
              continue;
            auto *C = dyn_cast<SCEVConstant>(M->getOperand(0));
            const SCEV *X = M->getOperand(1);
            while (auto *Cast = dyn_cast<SCEVCastExpr>(X))
              X = Cast->getOperand();
            auto *U = dyn_cast<SCEVUnknown>(X);
            auto *Call = U ? dyn_cast<IntrinsicInst>(U->getValue()) : nullptr;
            if (C && Call &&
                (Call->getIntrinsicID() ==
                     Intrinsic::agx_thread_position_in_grid_x ||
                 Call->getIntrinsicID() ==
                     Intrinsic::agx_thread_position_in_grid_y) &&
                C->getAPInt().getZExtValue() *
                        DL.getTypeAllocSize(G->getSourceElementType()) >=
                    64)
              return true;
          }
        }
        return false;
      };
      // (8 for a row's one load an iteration, a sum: row_sum_dyn 354.4 at 4,
      // 311.3 at 8; 4 for two, a dot: row_dot_dyn 488.3 at 4, 744.4 at 8)
      unsigned RowLoads = count_if(*L->getHeader(), [](const Instruction &I) {
        auto *LI = dyn_cast<LoadInst>(&I);
        return LI && LI->getPointerAddressSpace() == AGX::DEVICE_ADDRESS;
      });
      // (exponentials too, a softmax's sum, with -agx-device-unroll-runtime-exp:
      // its loads are a thread's own row's)
      bool LightOrExp =
          Light || (AGXDeviceUnrollRuntimeExp && Exps && !Other &&
                    none_of(*L->getHeader(), [](const Instruction &I) {
                      return isa<StoreInst>(I) || I.isAtomic() ||
                             I.getOpcode() == Instruction::FDiv;
                    }));
      // (a running maximum / minimum of a thread's own row, rowmax_dyn: as
      // a sum, L15-E582)
      bool LightMax =
          AGXDeviceUnrollRuntimeMax && !Light &&
          all_of(*L->getHeader(), [](const Instruction &I) {
            if (isa<StoreInst>(I) || I.isAtomic() ||
                I.getOpcode() == Instruction::FDiv)
              return false;
            if (auto *II = dyn_cast<IntrinsicInst>(&I))
              return II->getIntrinsicID() == Intrinsic::fma ||
                     II->getIntrinsicID() == Intrinsic::fmuladd ||
                     II->getIntrinsicID() == Intrinsic::maxnum ||
                     II->getIntrinsicID() == Intrinsic::minnum ||
                     II->getIntrinsicID() == Intrinsic::maximum ||
                     II->getIntrinsicID() == Intrinsic::minimum;
            return !isa<CallBase>(I);
          });
      LightOrExp |= LightMax && Insts <= 24;
      if (LightOrExp && !Far && Insts <= 48 &&
          AGXDeviceUnrollRuntimeContiguous >= 2 && RowPerThread() &&
          (Light ? Insts <= 24 : true)) {
        UP.Partial = UP.Runtime = true;
        unsigned Count = RowLoads == 1
                             ? 2 * AGXDeviceUnrollRuntimeContiguous
                             : unsigned(AGXDeviceUnrollRuntimeContiguous);
        // (as many bytes an iteration from rows of halves or bytes as of
        // floats: hsum_dyn 362.8 -> 317.3, bsum_dyn 579.3 -> 472.4,
        // L15-E583)
        if (AGXDeviceUnrollRuntimeNarrowRows && !Exps) {
          unsigned MinBytes = 4;
          for (const Instruction &I : *L->getHeader())
            if (auto *LI = dyn_cast<LoadInst>(&I);
                LI && LI->getPointerAddressSpace() == AGX::DEVICE_ADDRESS)
              MinBytes = std::min<unsigned>(
                  MinBytes, std::max<uint64_t>(
                                1, DL.getTypeStoreSize(LI->getType())));
          Count *= 4 / MinBytes;
        }
        UP.DefaultUnrollRuntimeCount = UP.MaxCount = Count;
        UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
      }
      if (Light && Far && Insts <= 24) {
        UP.Partial = UP.Runtime = true;
        // (8 where the far loads step by a runtime stride, a column: matmul
        // 141.3 -> 138.3, colsum_dyn 408.8 -> 323.1; a gather's 8 lost,
        // spmv_csr 41.0 -> 45.0, L15-E537)
        unsigned Count = Gather ? AGXDeviceUnrollRuntime
                                : std::max<unsigned>(AGXDeviceUnrollRuntime,
                                                     AGXDeviceUnrollRuntimeStrided);
        UP.DefaultUnrollRuntimeCount = UP.MaxCount = Count;
        UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
      }
    }
    return;
  }
  if (!ThreadgroupLoads || L->getNumBlocks() != 1)
    return;
  if (AGXThreadgroupUnroll == 1)
    return;
  if (AGXThreadgroupUnroll == 0) {
    // By default only as far as contiguous threadgroup loads become 16-byte
    // ones, in loops with a known trip count that the factor divides (a
    // matrix product's inner loop over a tile once launch-time
    // specialization made the tile size a constant).
    unsigned Factor = contiguousThreadgroupFactor(L, SE);
    unsigned Trips = SE.getSmallConstantTripCount(L);
    LLVM_DEBUG(dbgs() << "AGX threadgroup unroll: factor " << Factor
                      << ", trip count " << Trips << "\n");
    if (Factor < 2 || Trips < Factor || Trips % Factor != 0)
      return;
    UP.Partial = true;
    UP.MaxCount = Factor;
    UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
    return;
  }
  UP.Partial = UP.Runtime = true;
  UP.MaxCount = AGXThreadgroupUnroll;
  // the unrolled body of such a loop (a tile of a matrix product) is a few
  // hundred instructions; the generic limit (150) would stop at its first copy
  UP.PartialThreshold = std::max(UP.PartialThreshold, 600u);
}
