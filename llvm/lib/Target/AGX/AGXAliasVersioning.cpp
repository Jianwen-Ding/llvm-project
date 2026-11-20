//===-- AGXAliasVersioning.cpp - Loops versioned on disjoint buffers ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A kernel's buffers are separate in practice, but nothing tells the compiler
// (MLIR's memrefs, the PTX front end's pointers): a loop that loads from some
// and stores to another loads, waits and stores before its next load, since
// the store may write what that load reads. An innermost loop whose device
// accesses loop access analysis can bound runs as two copies:
//
//   if (the ranges the loop's stores write overlap none it reads or writes)
//     loop, its accesses in disjoint alias scopes
//   else
//     loop as it was
//
// so the scheduler can issue a later iteration's loads ahead of a store.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/ADT/DenseSet.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;

#define DEBUG_TYPE "agx-alias-version"

static cl::opt<bool> EnableAliasVersioning(
    "agx-alias-versioning", cl::init(true), cl::Hidden,
    cl::desc("Version innermost loops that store to a device buffer on their "
             "accesses' ranges not overlapping, the copy's accesses in "
             "disjoint alias scopes"));

static cl::opt<unsigned> AliasVersioningMaxChecks(
    "agx-alias-versioning-max-checks", cl::init(6), cl::Hidden,
    cl::desc("Runtime range checks a loop may need to be versioned"));

static cl::opt<unsigned> AliasVersioningMaxWords(
    "agx-alias-versioning-max-words", cl::init(48), cl::Hidden,
    cl::desc("32-bit words a loop iteration may load to be versioned (more, "
             "hoisted, would cost the registers that keep groups resident)"));

static cl::opt<unsigned> AliasVersioningMathPerWord(
    "agx-alias-versioning-math-per-word", cl::init(64), cl::Hidden,
    cl::desc("Arithmetic (a transcendental 16, a division 8) per 32-bit word "
             "loaded above which a loop's latency is hidden behind its work "
             "and it is not versioned"));

static cl::opt<bool> AliasVersioningWordBytes(
    "agx-alias-versioning-word-bytes", cl::init(true), cl::Hidden,
    cl::desc("Count the words a loop loads from its loads' bytes together"));

static cl::opt<bool> AliasVersioningLoopRanges(
    "agx-alias-versioning-loop-ranges", cl::init(true), cl::Hidden,
    cl::desc("Analyze loops with the thread positions' bounds"));

static cl::opt<bool> AliasVersioningLoopRangesRuntime(
    "agx-alias-versioning-loop-ranges-runtime", cl::init(true), cl::Hidden,
    cl::desc("-agx-alias-versioning-loop-ranges also for runtime trip counts "
             "whose accesses step at most 16 bytes"));

static cl::opt<unsigned> AliasVersioningBlocks(
    "agx-alias-versioning-blocks", cl::init(20), cl::Hidden,
    cl::desc("Blocks an innermost loop may have to be versioned, its copy's "
             "loads of blocks every iteration runs moved to the header "
             "(0: one block only)"));

static cl::opt<bool> AliasVersioningStraightLine(
    "agx-alias-versioning-straight-line", cl::init(true), cl::Hidden,
    cl::desc("Also version a straight-line run of device loads and stores "
             "outside loops (an unrolled loop's) on its ranges not "
             "overlapping"));

static cl::opt<unsigned> AliasVersioningSegmentMinStores(
    "agx-alias-versioning-segment-min-stores", cl::init(4), cl::Hidden,
    cl::desc("Device stores a straight-line run needs to be versioned"));

static cl::opt<bool> AliasVersioningSegmentMath(
    "agx-alias-versioning-segment-math", cl::init(true), cl::Hidden,
    cl::desc("Apply -agx-alias-versioning-math-per-word to straight-line "
             "runs too"));

namespace {

class AGXAliasVersioning : public FunctionPass {
public:
  static char ID;
  AGXAliasVersioning() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Alias Versioning"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
};

} // namespace

char AGXAliasVersioning::ID = 0;

INITIALIZE_PASS_BEGIN(AGXAliasVersioning, DEBUG_TYPE, "AGX Alias Versioning",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(AGXAliasVersioning, DEBUG_TYPE, "AGX Alias Versioning",
                    false, false)

FunctionPass *llvm::createAGXAliasVersioningPass() {
  return new AGXAliasVersioning();
}

static cl::opt<bool> AGXAliasNarrowStores(
    "agx-alias-versioning-narrow-stores", cl::init(true), cl::Hidden,
    cl::desc("Version loops whose stores, not loads, are of 1- or 2-byte "
             "elements"));

static cl::opt<bool> AGXAliasNarrowLoads(
    "agx-alias-versioning-narrow-loads", cl::init(true), cl::Hidden,
    cl::desc("Version loops whose loads are of 1- or 2-byte elements"));

/// A single-block innermost loop with a device store and a device load, and
/// no other memory operation (calls, threadgroup memory, atomics) nor a
/// convergent one, which a divergent check must not duplicate. Accesses of
/// 1- or 2-byte elements were left: alignment versioning copied such a loop
/// again, four copies (L15-E527). Narrow stores of wider loads (int8
/// quantization) are not (L15-E546), nor, since alignment versioning leaves
/// the copy for overlapping ranges (L15-E547), narrow loads (a byte mask's
/// select, f16 widened: L15-E550).
static bool isCandidate(const Loop *L, const DataLayout &DL) {
  auto Narrow = [&](Type *Ty) {
    return DL.getTypeStoreSize(Ty->getScalarType()) < 4;
  };
  if (!L->isInnermost() ||
      (L->getNumBlocks() != 1 &&
       (!AliasVersioningBlocks || L->getNumBlocks() > AliasVersioningBlocks)))
    return false;
  bool Store = false, Load = false;
  for (const BasicBlock *BB : L->blocks())
  for (const Instruction &I : *BB) {
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (CB->isConvergent())
        return false;
      if (CB->mayReadOrWriteMemory())
        return false;
      continue;
    }
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!LI->isSimple() ||
          LI->getPointerAddressSpace() != AGX::DEVICE_ADDRESS ||
          (Narrow(LI->getType()) && !AGXAliasNarrowLoads))
        return false;
      Load = true;
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isSimple() ||
          SI->getPointerAddressSpace() != AGX::DEVICE_ADDRESS ||
          (Narrow(SI->getValueOperand()->getType()) &&
           !AGXAliasNarrowStores))
        return false;
      Store = true;
    } else if (I.mayReadOrWriteMemory()) {
      return false;
    }
  }
  return Store && Load;
}

/// Whether I ends a straight-line run of device accesses: a convergent or
/// memory-touching call, or an access other than a simple device load or
/// store.
static bool endsRun(const Instruction &I) {
  if (auto *CB = dyn_cast<CallBase>(&I))
    return CB->isConvergent() || CB->mayReadOrWriteMemory();
  if (auto *LI = dyn_cast<LoadInst>(&I))
    return !LI->isSimple() ||
           LI->getPointerAddressSpace() != AGX::DEVICE_ADDRESS;
  if (auto *SI = dyn_cast<StoreInst>(&I))
    return !SI->isSimple() ||
           SI->getPointerAddressSpace() != AGX::DEVICE_ADDRESS;
  return I.mayReadOrWriteMemory() || I.isTerminator();
}

/// The work of a loop iteration: (32-bit words loaded, arithmetic). Hoisted
/// ahead of the stores, many words cost registers (and resident groups), and
/// a loop with much arithmetic per word already hides its loads' latency
/// behind it and the groups: neither gains from a copy (L15-E527; one
/// exponential a word, row_exp's, still did: L15-E549; two, sin and cos,
/// log and pow, too: L15-E580).
template <typename Range>
static std::pair<unsigned, unsigned> work(Range &&Insts, const DataLayout &DL) {
  unsigned Words = 0, Math = 0;
  uint64_t Bytes = 0;
  for (const Instruction &I : Insts) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      // (narrow elements together: 2-byte loads, two to a word, the load /
      // store vectorizer merges; each counted a word, i16_rows' loop of 32
      // words seemed 64, L15-E554)
      uint64_t B = DL.getTypeStoreSize(LI->getType());
      if (AliasVersioningWordBytes)
        Bytes += B;
      else
        Words += divideCeil(B, 4);
      continue;
    }
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::exp10:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::sincos:
      case Intrinsic::tan:
      case Intrinsic::tanh:
      case Intrinsic::sinh:
      case Intrinsic::cosh:
      case Intrinsic::asin:
      case Intrinsic::acos:
      case Intrinsic::atan:
      case Intrinsic::atan2:
      case Intrinsic::pow:
      case Intrinsic::powi:
        Math += 16;
        continue;
      default:
        break;
      }
    }
    Type *Ty = I.getType();
    bool FP = Ty->isFPOrFPVectorTy() ||
              any_of(I.operands(), [](const Use &U) {
                return U->getType()->isFPOrFPVectorTy();
              });
    if (!FP || isa<PHINode>(I) || isa<StoreInst>(I) || I.isCast())
      continue;
    unsigned Lanes = 1;
    if (auto *VT = dyn_cast<FixedVectorType>(Ty))
      Lanes = VT->getNumElements();
    unsigned Op = I.getOpcode();
    Math += Lanes * (Op == Instruction::FDiv || Op == Instruction::FRem ? 8 : 1);
  }
  return {Words + unsigned(divideCeil(Bytes, 4)), Math};
}

static std::pair<unsigned, unsigned> iterationWork(const Loop *L,
                                                   const DataLayout &DL) {
  unsigned Words = 0, Math = 0;
  for (const BasicBlock *BB : L->blocks()) {
    auto [W, M] = work(*BB, DL);
    Words += W;
    Math += M;
  }
  return {Words, Math};
}

/// In the copy of a loop of several blocks for ranges not overlapping (a
/// conditional store's), the loads of blocks every iteration runs moved up
/// to the header, ahead of the stores no load may alias, with what their
/// addresses are computed from: the unrolled copies' loads issue together,
/// not each after the previous copy's store (L15-E573).
static bool hoistLoads(Loop *L, DominatorTree &DT, AAResults &AA) {
  BasicBlock *H = L->getHeader(), *Latch = L->getLoopLatch();
  LLVM_DEBUG(dbgs() << "agx-alias-version: hoist " << H->getName() << " "
                    << (Latch ? Latch->getName() : "-") << " "
                    << L->getNumBlocks() << "\n");
  if (!Latch || L->getNumBlocks() == 1)
    return false;
  SmallVector<StoreInst *, 8> Stores;
  for (BasicBlock *BB : L->blocks())
    for (Instruction &I : *BB)
      if (auto *SI = dyn_cast<StoreInst>(&I))
        Stores.push_back(SI);
  Instruction *At = H->getTerminator();
  // Whether V can be computed at At: defined outside the loop, or moved
  // there with its operands (no memory, no side effects, no phi).
  SmallVector<Instruction *, 8> Moves;
  std::function<bool(Value *, unsigned)> Available = [&](Value *V,
                                                         unsigned Depth) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !L->contains(I) || DT.dominates(I, At))
      return true;
    if (Depth > 8 || isa<PHINode>(I) || I->mayReadOrWriteMemory() ||
        I->mayHaveSideEffects() || !isSafeToSpeculativelyExecute(I))
      return false;
    for (Value *Op : I->operands())
      if (!Available(Op, Depth + 1))
        return false;
    if (!is_contained(Moves, I))
      Moves.push_back(I);
    return true;
  };
  bool Changed = false;
  for (BasicBlock *BB : L->getBlocks()) {
    if (BB == H || !DT.dominates(BB, Latch))
      continue;
    for (Instruction &I : make_early_inc_range(*BB)) {
      auto *LI = dyn_cast<LoadInst>(&I);
      if (!LI || !LI->isSimple())
        continue;
      // (the copy's scopes: every scope of the load's in the store's
      // noalias list, as LoopVersioning annotates them)
      MDNode *Scopes = LI->getMetadata(LLVMContext::MD_alias_scope);
      auto Disjoint = [&](StoreInst *SI) {
        MDNode *NoAlias = SI->getMetadata(LLVMContext::MD_noalias);
        return Scopes && NoAlias && Scopes->getNumOperands() &&
               all_of(Scopes->operands(), [&](const MDOperand &S) {
                 return is_contained(NoAlias->operands(), S);
               });
      };
      MemoryLocation ML = MemoryLocation::get(LI);
      if (any_of(Stores, [&](StoreInst *SI) {
            return !Disjoint(SI) &&
                   !AA.isNoAlias(ML, MemoryLocation::get(SI));
          })) {
        LLVM_DEBUG(dbgs() << "agx-alias-version: may alias " << *LI << "\n");
        continue;
      }
      Moves.clear();
      if (!Available(LI->getPointerOperand(), 0)) {
        LLVM_DEBUG(dbgs() << "agx-alias-version: address " << *LI << "\n");
        continue;
      }
      for (Instruction *M : Moves)
        M->moveBefore(At->getIterator());
      LI->moveBefore(At->getIterator());
      Changed = true;
    }
  }
  return Changed;
}

namespace {
/// A device address as Base + Scale * ext(Root) + Off (no Root: Base + Off),
/// Root an integer SCEV.
struct Address {
  Value *Base = nullptr;
  const SCEV *Root = nullptr;
  bool Signed = false;
  int64_t Scale = 0, Off = 0;
};

/// The device accesses of a run at constant offsets from one address: the
/// bytes [Base + Scale * ext(Root) + Lo, ... + Hi) they touch.
struct AccessGroup {
  Address A;
  int64_t Lo = 0, Hi = 0;
  bool Written = false;
  SmallVector<Instruction *, 8> Members;
};
} // namespace

/// The index V as Root + C (SCEVs): its constant term, where the sum cannot
/// wrap in V's width given Root's range, so that extended it is ext(Root) + C
/// (an unrolled loop's zext(i + 32), whose add has no nuw).
static std::pair<const SCEV *, int64_t> splitIndex(Value *V, bool Signed,
                                                   ScalarEvolution &SE) {
  const SCEV *S = SE.getSCEV(V);
  auto *Add = dyn_cast<SCEVAddExpr>(S);
  auto *K = Add ? dyn_cast<SCEVConstant>(Add->getOperand(0)) : nullptr;
  if (!K || K->getAPInt().getSignificantBits() > 32)
    return {S, 0};
  const SCEV *R = SE.getMinusSCEV(S, K);
  ConstantRange Range =
      Signed ? SE.getSignedRange(R) : SE.getUnsignedRange(R);
  const APInt &C = K->getAPInt();
  bool Overflow = false;
  if (Signed) {
    (void)Range.getSignedMax().sadd_ov(C, Overflow);
    if (!Overflow)
      (void)Range.getSignedMin().sadd_ov(C, Overflow);
  } else if (!C.isNegative()) {
    (void)Range.getUnsignedMax().uadd_ov(C, Overflow);
  } else {
    Overflow = Range.getUnsignedMin().ult(-C);
  }
  if (Overflow)
    return {S, 0};
  return {R, C.getSExtValue()};
}

/// Ptr through GEPs of one variable index (at most one in all) and constant
/// offsets.
static std::optional<Address> decompose(Value *Ptr, const DataLayout &DL,
                                        ScalarEvolution &SE) {
  Address A;
  for (unsigned Depth = 0; Depth < 16; ++Depth) {
    auto *GEP = dyn_cast<GEPOperator>(Ptr);
    if (!GEP) {
      A.Base = Ptr;
      return A;
    }
    APInt Off(64, 0);
    if (GEP->accumulateConstantOffset(DL, Off)) {
      A.Off += Off.getSExtValue();
      Ptr = GEP->getPointerOperand();
      continue;
    }
    if (GEP->getNumIndices() != 1 || A.Root)
      return std::nullopt;
    Value *Idx = GEP->getOperand(1);
    int64_t Size = DL.getTypeAllocSize(GEP->getSourceElementType());
    bool Signed = true;
    if (auto *Z = dyn_cast<ZExtInst>(Idx)) {
      Idx = Z->getOperand(0);
      Signed = false;
    } else if (auto *X = dyn_cast<SExtInst>(Idx)) {
      Idx = X->getOperand(0);
    }
    auto [R, C] = splitIndex(Idx, Signed, SE);
    A.Root = R;
    A.Signed = Signed;
    A.Scale = Size;
    A.Off += C * Size;
    Ptr = GEP->getPointerOperand();
  }
  return std::nullopt;
}

/// A straight-line run [First, Last] of device accesses (in a block outside
/// loops, no call or other access in between: an unrolled loop's, as
/// softmax_f16's normalized writes after its subgroup sum) as two copies,
/// one for its groups' ranges not overlapping, its accesses in disjoint
/// alias scopes (L15-E528). The check is the thread's own (no convergent
/// operation is copied).
namespace {
/// A versioned run: the check's block, the two copies and the join.
struct VersionedRun {
  BasicBlock *Check, *Fast, *Run, *Post;
};
} // namespace

static bool versionRun(Instruction *First, Instruction *Last,
                       ScalarEvolution &SE, DominatorTree &DT, LoopInfo &LI,
                       SmallVectorImpl<VersionedRun> &Out) {
  BasicBlock *BB = First->getParent();
  const DataLayout &DL = BB->getDataLayout();
  SmallVector<AccessGroup, 8> Groups;
  unsigned Stores = 0;
  bool LoadAfterStore = false;
  for (Instruction &I :
       make_range(First->getIterator(), std::next(Last->getIterator()))) {
    Value *Ptr = getLoadStorePointerOperand(&I);
    if (!Ptr)
      continue;
    bool W = isa<StoreInst>(I);
    Stores += W;
    LoadAfterStore |= !W && Stores;
    int64_t Size = DL.getTypeStoreSize(getLoadStoreType(&I));
    std::optional<Address> A = decompose(Ptr, DL, SE);
    if (!A)
      return false;
    AccessGroup *G = nullptr;
    for (AccessGroup &X : Groups)
      if (X.A.Base == A->Base && X.A.Root == A->Root &&
          X.A.Signed == A->Signed && X.A.Scale == A->Scale) {
        G = &X;
        break;
      }
    if (!G) {
      Groups.emplace_back();
      G = &Groups.back();
      G->A = *A;
      G->Lo = A->Off;
      G->Hi = A->Off + Size;
    }
    G->Lo = std::min(G->Lo, A->Off);
    G->Hi = std::max(G->Hi, A->Off + Size);
    G->Written |= W;
    G->Members.push_back(&I);
  }
  LLVM_DEBUG(dbgs() << "agx-alias-version run in " << BB->getName() << ": "
                    << Stores << " stores, " << Groups.size() << " groups\n");
  if (Stores < AliasVersioningSegmentMinStores || !LoadAfterStore)
    return false;
  SmallVector<std::pair<unsigned, unsigned>, 8> Checks;
  for (unsigned A = 0; A < Groups.size(); ++A)
    for (unsigned B = A + 1; B < Groups.size(); ++B)
      if (Groups[A].Written || Groups[B].Written)
        Checks.push_back({A, B});
  if (Checks.empty() || Checks.size() > AliasVersioningMaxChecks) {
    LLVM_DEBUG(dbgs() << "  " << Checks.size() << " checks\n");
    return false;
  }
  // Loads of addresses loaded before the run, whose values (and what was
  // computed from them) the copy can take once its stores are known apart:
  // then the arithmetic goes with them (softmax_f16's third pass over its
  // row recomputing the second's exponentials).
  DenseSet<const SCEV *> Earlier;
  for (BasicBlock &X : *BB->getParent())
    if (&X != BB && DT.dominates(&X, BB))
      for (Instruction &I : X)
        if (auto *LI = dyn_cast<LoadInst>(&I))
          Earlier.insert(SE.getSCEV(LI->getPointerOperand()));
  for (Instruction &I : make_range(BB->begin(), First->getIterator()))
    if (auto *LI = dyn_cast<LoadInst>(&I))
      Earlier.insert(SE.getSCEV(LI->getPointerOperand()));
  unsigned Loads = 0, Reloads = 0;
  for (Instruction &I :
       make_range(First->getIterator(), std::next(Last->getIterator())))
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      ++Loads;
      Reloads += Earlier.contains(SE.getSCEV(LI->getPointerOperand()));
    }
  bool Reloading = 2 * Reloads >= Loads;
  LLVM_DEBUG(dbgs() << "  " << Reloads << " of " << Loads << " loads reload\n");
  if (AliasVersioningSegmentMath && !Reloading) {
    auto [Words, Math] = work(
        make_range(First->getIterator(), std::next(Last->getIterator())), DL);
    if (Words == 0 || Math > AliasVersioningMathPerWord * Words) {
      LLVM_DEBUG(dbgs() << "  math " << Math << " words " << Words << "\n");
      return false;
    }
  }
  // the groups' addresses must be known before the run
  auto Before = [&](Value *V) {
    auto *I = dyn_cast_or_null<Instruction>(V);
    return !I || DT.dominates(I, First);
  };
  auto RootBefore = [&](const SCEV *R) {
    return !R || !SCEVExprContains(R, [&](const SCEV *X) {
      auto *U = dyn_cast<SCEVUnknown>(X);
      return U && !Before(U->getValue());
    });
  };
  for (AccessGroup &G : Groups)
    if (!Before(G.A.Base) || !RootBefore(G.A.Root)) {
      LLVM_DEBUG(dbgs() << "  address after the run's first access\n");
      return false;
    }

  // BB: ... | Run: First .. Last | Post
  BasicBlock *Post =
      SplitBlock(BB, std::next(Last->getIterator()), &DT, &LI, nullptr,
                 BB->getName() + ".noalias.end");
  BasicBlock *Run = SplitBlock(BB, First->getIterator(), &DT, &LI, nullptr,
                               BB->getName() + ".mayalias");
  // the check, at the end of BB
  IRBuilder<> B(BB->getTerminator());
  Type *IntPtr = DL.getIntPtrType(B.getContext(), AGX::DEVICE_ADDRESS);
  SCEVExpander Exp(SE, "agx.alias");
  SmallVector<std::pair<Value *, Value *>, 8> Bounds;
  for (AccessGroup &G : Groups) {
    Value *I = B.CreatePtrToInt(G.A.Base, IntPtr);
    if (G.A.Root) {
      Value *R = Exp.expandCodeFor(G.A.Root, G.A.Root->getType(),
                                   BB->getTerminator()->getIterator());
      B.SetInsertPoint(BB->getTerminator());
      I = B.CreateAdd(I, B.CreateMul(B.CreateIntCast(R, IntPtr, G.A.Signed),
                                     ConstantInt::get(IntPtr, G.A.Scale)));
    }
    Bounds.push_back({B.CreateAdd(I, ConstantInt::get(IntPtr, G.Lo)),
                      B.CreateAdd(I, ConstantInt::get(IntPtr, G.Hi))});
  }
  Value *Safe = B.getTrue();
  for (auto [A, C] : Checks)
    Safe = B.CreateAnd(
        Safe, B.CreateOr(B.CreateICmpULE(Bounds[A].second, Bounds[C].first),
                         B.CreateICmpULE(Bounds[C].second, Bounds[A].first)),
        "agx.alias.safe");
  // the copy
  ValueToValueMapTy VMap;
  BasicBlock *Fast = CloneBasicBlock(Run, VMap, ".noalias", BB->getParent());
  remapInstructionsInBlocks({Fast}, VMap);
  BB->getTerminator()->eraseFromParent();
  IRBuilder<>(BB).CreateCondBr(Safe, Fast, Run);
  // values the run defines used after it: a phi of the two copies'
  for (Instruction &I : *Run) {
    SmallVector<Use *, 4> Outside;
    for (Use &U : I.uses())
      if (cast<Instruction>(U.getUser())->getParent() != Run)
        Outside.push_back(&U);
    if (Outside.empty())
      continue;
    PHINode *PN = PHINode::Create(I.getType(), 2, I.getName() + ".v");
    PN->insertBefore(Post->begin());
    PN->addIncoming(&I, Run);
    PN->addIncoming(VMap[&I], Fast);
    for (Use *U : Outside)
      U->set(PN);
  }
  // disjoint scopes for the copy's groups
  MDBuilder MDB(B.getContext());
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("agx.alias.versioning");
  SmallVector<MDNode *, 8> Scopes;
  for (unsigned G = 0; G < Groups.size(); ++G)
    Scopes.push_back(MDB.createAnonymousAliasScope(Domain));
  for (unsigned G = 0; G < Groups.size(); ++G) {
    SmallVector<Metadata *, 8> Others;
    for (unsigned H = 0; H < Groups.size(); ++H)
      if (H != G)
        Others.push_back(Scopes[H]);
    MDNode *Scope = MDNode::get(B.getContext(), {Scopes[G]});
    MDNode *NoAlias = MDNode::get(B.getContext(), Others);
    for (Instruction *I : Groups[G].Members) {
      auto *C = cast<Instruction>(VMap[I]);
      C->setMetadata(LLVMContext::MD_alias_scope,
                     MDNode::concatenate(C->getMetadata(LLVMContext::MD_alias_scope), Scope));
      C->setMetadata(LLVMContext::MD_noalias,
                     MDNode::concatenate(C->getMetadata(LLVMContext::MD_noalias), NoAlias));
    }
  }
  DT.recalculate(*BB->getParent());
  Out.push_back({BB, Fast, Run, Post});
  return true;
}

/// A block's work: an instruction 1, a transcendental 16, a division 8.
static unsigned blockCost(const BasicBlock &BB) {
  unsigned Cost = 0;
  for (const Instruction &I : BB) {
    if (isa<PHINode>(I) || I.isTerminator())
      continue;
    auto [Words, Math] = work(make_range(I.getIterator(), std::next(I.getIterator())),
                              BB.getDataLayout());
    (void)Words;
    Cost += std::max(1u, Math);
  }
  return Cost;
}

/// Back to the run alone: its copy did not shrink enough.
static void undoRun(const VersionedRun &V) {
  auto *Br = V.Check->getTerminator();
  Value *Cond = cast<CondBrInst>(Br)->getCondition();
  IRBuilder<>(Br).CreateBr(V.Run);
  Br->eraseFromParent();
  RecursivelyDeleteTriviallyDeadInstructions(Cond);
  for (PHINode &PN : make_early_inc_range(V.Post->phis())) {
    PN.removeIncomingValue(V.Fast, /*DeletePHIIfEmpty=*/false);
    if (PN.getNumIncomingValues() == 1) {
      PN.replaceAllUsesWith(PN.getIncomingValue(0));
      PN.eraseFromParent();
    }
  }
  DeleteDeadBlock(V.Fast);
  // one block again (a one-block kernel's straight-line alignment
  // versioning, L15-E522, asks for it)
  MergeBlockIntoPredecessor(V.Post);
  MergeBlockIntoPredecessor(V.Run);
}

/// The straight-line runs of the function's blocks outside loops.
static bool versionRuns(Function &F, ScalarEvolution &SE, DominatorTree &DT,
                        LoopInfo &LI, SmallVectorImpl<VersionedRun> &Out) {
  SmallVector<std::pair<Instruction *, Instruction *>, 4> Runs;
  for (BasicBlock &BB : F) {
    if (LI.getLoopFor(&BB))
      continue;
    Instruction *First = nullptr, *Last = nullptr;
    auto Flush = [&] {
      if (First && Last && First != Last)
        Runs.push_back({First, Last});
      First = Last = nullptr;
    };
    for (Instruction &I : BB) {
      if (endsRun(I)) {
        Flush();
        continue;
      }
      // (from the run's first store: the loads before it are not held by
      // any of its stores, and their values are earlier ones to its copy)
      if (isa<StoreInst>(I) || (First && isa<LoadInst>(I))) {
        if (!First)
          First = &I;
        Last = &I;
      }
    }
    Flush();
  }
  bool Changed = false;
  for (auto [First, Last] : Runs)
    Changed |= versionRun(First, Last, SE, DT, LI, Out);
  return Changed;
}

bool AGXAliasVersioning::runOnFunction(Function &F) {
  if (!EnableAliasVersioning || skipFunction(F) ||
      F.getCallingConv() != CallingConv::AGX_KERNEL)
    return false;
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  SmallVector<Loop *, 8> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (isCandidate(L, F.getDataLayout()))
      Loops.push_back(L);
  if (Loops.empty())
    return false;
  bool Changed = false;
  // Versions L, analyzed with SE; whether LAA could not check its accesses
  // (or not in few enough checks).
  auto Version = [&](Loop *L, ScalarEvolution &SE) {
    auto [Words, Math] = iterationWork(L, F.getDataLayout());
    if (Words == 0 || Words > AliasVersioningMaxWords ||
        Math > AliasVersioningMathPerWord * Words)
      return false;
    if (!L->isLoopSimplifyForm()) {
      Changed |= simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr,
                              /*PreserveLCSSA=*/false);
      if (!L->isLoopSimplifyForm())
        return false;
    }
    // (one exiting block: LoopVersioning's, a loop of several blocks with a
    // break has more)
    if (!L->getUniqueExitBlock() || !L->getExitingBlock())
      return false;
    Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
    LoopAccessInfo LAI(L, &SE, &TTI, &TLI, &AA, &DT, &LI, &AC);
    const RuntimePointerChecking *RPC = LAI.getRuntimePointerChecking();
    LLVM_DEBUG(dbgs() << "agx-alias-version: LAA " << F.getName() << " "
                      << LAI.canVectorizeMemory() << " "
                      << (RPC ? int(RPC->Need) : -1) << " "
                      << LAI.getNumRuntimePointerChecks() << " "
                      << LAI.getPSE().getPredicate().isAlwaysTrue() << "\n");
    if (!LAI.canVectorizeMemory() || !RPC)
      return true;
    if (!RPC->Need || LAI.hasConvergentOp() ||
        LAI.getNumRuntimePointerChecks() == 0)
      return false;
    // (predicates that its indices do not wrap, or too many checks: accesses
    // whose bounds could not be compared, each in a group of its own)
    if (!LAI.getPSE().getPredicate().isAlwaysTrue() ||
        LAI.getNumRuntimePointerChecks() > AliasVersioningMaxChecks)
      return true;
    LLVM_DEBUG(dbgs() << "agx-alias-version: " << F.getName() << " "
                      << L->getHeader()->getName() << " "
                      << LAI.getNumRuntimePointerChecks() << " checks\n");
    LoopVersioning LV(LAI, RPC->getChecks(), L, &LI, &DT, &SE);
    LV.versionLoop();
    LV.annotateLoopWithNoAlias();
    if (AliasVersioningBlocks)
      hoistLoads(L, DT, AA);
    // (the copy for overlapping ranges, which runs only when they do:
    // alignment versioning leaves it, L15-E547)
    addStringMetadataToLoop(LV.getNonVersionedLoop(),
                            "llvm.loop.agx.alias.fallback");
    Changed = true;
    return false;
  };
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  SmallVector<Loop *, 4> Unchecked;
  // (a runtime trip count where the loop's accesses step at most 16 bytes:
  // a row of a runtime length, not a grid-stride loop, L15-E565)
  auto SmallSteps = [&](Loop *L) {
    return all_of(L->blocks(), [&](BasicBlock *BB) {
      return all_of(*BB, [&](Instruction &I) {
        Value *P = getLoadStorePointerOperand(&I);
        if (!P)
          return true;
        auto *G = dyn_cast<GetElementPtrInst>(P);
        const SCEV *X = SE.getSCEV(P);
        if (!isa<SCEVAddRecExpr>(X) && G && G->getNumIndices() == 1) {
          Value *Idx = G->getOperand(1);
          if (auto *E = dyn_cast<CastInst>(Idx);
              E && (isa<ZExtInst>(E) || isa<SExtInst>(E)))
            Idx = E->getOperand(0);
          const SCEV *XI = SE.getSCEV(Idx);
          auto *AR = dyn_cast<SCEVAddRecExpr>(XI);
          auto *C = AR && AR->getLoop() == L
                        ? dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))
                        : nullptr;
          return C && C->getAPInt().abs().getZExtValue() *
                              F.getDataLayout().getTypeAllocSize(
                                  G->getSourceElementType()) <=
                          16;
        }
        auto *AR = dyn_cast<SCEVAddRecExpr>(X);
        auto *C = AR && AR->getLoop() == L
                      ? dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))
                      : nullptr;
        return C && C->getAPInt().abs().ule(16);
      });
    });
  };
  for (Loop *L : Loops)
    if (Version(L, SE) &&
        (SE.getSmallConstantTripCount(L) ||
         (AliasVersioningLoopRangesRuntime && SmallSteps(L)))) {
      LLVM_DEBUG(dbgs() << "agx-alias-version: unchecked "
                        << L->getHeader()->getName() << "\n");
      Unchecked.push_back(L);
    }
  // A loop whose accesses LAA could not check (or only with predicates, or
  // in too many checks), of a constant trip count,
  // again with the thread positions' bounds (the kernel's too), for the
  // analysis only, in a ScalarEvolution of this pass's own: a row's i32
  // index zext({258 * r + 1, +, 16}) is then an add recurrence (a stencil's,
  // L15-E553). Taken off again; a loop that reads a position itself (its
  // copy would keep the bound) is left. (Every loop so analyzed changed the
  // checks of loops versioned without, and a grid-stride loop of 4 trips
  // lost: stride_loop's cycles 502 -> 610.)
  if (AliasVersioningLoopRanges && !Unchecked.empty()) {
    SmallVector<WeakVH, 8> Ranged;
    AGX::addSpecialRegisterRanges(F, &Ranged, /*ForAnalysis=*/true);
    SmallPtrSet<const Instruction *, 8> RangedCalls;
    for (WeakVH &V : Ranged)
      if (auto *CI = dyn_cast_or_null<Instruction>(V))
        RangedCalls.insert(CI);
    {
      ScalarEvolution OwnSE(F, TLI, AC, DT, LI);
      for (Loop *L : Unchecked)
        if (none_of(L->blocks(), [&](const BasicBlock *BB) {
              return any_of(*BB, [&](const Instruction &I) {
                return RangedCalls.contains(&I);
              });
            }))
          Version(L, OwnSE);
    }
    for (WeakVH &V : Ranged)
      if (auto *CI = dyn_cast_or_null<CallBase>(V))
        CI->removeRetAttr(Attribute::Range);
  }
  if (Changed)
    SE.forgetAllLoops();
  return Changed;
}

PreservedAnalyses AGXAliasVersioningPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  if (!EnableAliasVersioning || !AliasVersioningStraightLine ||
      F.getCallingConv() != CallingConv::AGX_KERNEL)
    return PreservedAnalyses::all();
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  // the thread positions' bounds (the kernel's too), for the analysis only:
  // an unrolled loop's zext(i + 32) is then zext(i) + 32 (taken off again,
  // L14-E1), in a ScalarEvolution of this pass's own
  SmallVector<WeakVH, 8> Ranged;
  AGX::addSpecialRegisterRanges(F, &Ranged, /*ForAnalysis=*/true);
  SmallVector<VersionedRun, 4> Runs;
  bool Changed;
  {
    ScalarEvolution SE(F, AM.getResult<TargetLibraryAnalysis>(F),
                       AM.getResult<AssumptionAnalysis>(F), DT, LI);
    Changed = versionRuns(F, SE, DT, LI, Runs);
  }
  for (WeakVH &V : Ranged)
    if (auto *CI = dyn_cast_or_null<CallBase>(V))
      CI->removeRetAttr(Attribute::Range);
  if (!Changed)
    return PreservedAnalyses::all();
  // the copies' loads of what the function loaded before, and what it
  // computed from them, are the earlier values now (softmax_f16's third
  // pass over its row: the second's exponentials)
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  AM.invalidate(F, PA);
  EarlyCSEPass(/*UseMemorySSA=*/true).run(F, AM);
  // a copy kept only where that made it at most half the run's work
  for (const VersionedRun &V : Runs) {
    unsigned Fast = blockCost(*V.Fast), Slow = blockCost(*V.Run);
    LLVM_DEBUG(dbgs() << "agx-alias-version copy " << Fast << " / " << Slow
                      << "\n");
    if (2 * Fast > Slow)
      undoRun(V);
  }
  return PreservedAnalyses::none();
}
