//===-- AGXAlignmentVersioning.cpp - Loops versioned on base alignment ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A half or int8 buffer is only known aligned to its element (Metal asks a
// binding's offset to be aligned to the data the function reads; MLIR's
// memrefs promise no more), so a loop over a half row loads it two bytes at
// a time, and a vector<4xf16> as four 16-bit loads: 4-byte loads and wider
// are only known to work at 4-byte alignment (B33; at 2 bytes, the
// simulator's follow-up 15). Buffers are aligned far beyond that in
// practice, so an innermost single-block loop with enough device accesses
// whose alignment a 4-byte aligned base would raise (their offsets from the
// base multiples of 4, SCEV's trailing zeros) runs as two copies:
//
//   if ((base & 3) == 0 for each such base)   // kernel arguments: uniform
//     loop, its accesses at the raised alignment
//   else
//     loop as it was
//
// and the load / store vectorizer, which runs after, merges the aligned
// copy's accesses into wide ones. Both copies compute the same values; the
// check is of kernel arguments, so no thread diverges.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "agx-align-versioning"

static cl::opt<bool> EnableAlignmentVersioning(
    "agx-alignment-versioning", cl::init(true), cl::Hidden,
    cl::desc("Version innermost loops on their buffers' 4-byte alignment, "
             "the aligned copy's device accesses at the raised alignment"));

static cl::opt<unsigned> AlignmentVersioningMin(
    "agx-alignment-versioning-min", cl::init(4), cl::Hidden,
    cl::desc("Device accesses whose alignment the check must raise for a "
             "loop to be versioned"));

static cl::opt<bool> CountVectorElements(
    "agx-alignment-versioning-elements", cl::init(true), cl::Hidden,
    cl::desc("Count a vector access as its elements towards "
             "-agx-alignment-versioning-min"));

static cl::opt<bool> SkipAliasFallback(
    "agx-align-version-skip-alias-fallback", cl::init(true), cl::Hidden,
    cl::desc("Leave alias versioning's copy for overlapping ranges"));

static cl::opt<bool> VersionByteLoads(
    "agx-align-version-byte-loads", cl::init(true), cl::Hidden,
    cl::desc("Raise scalar byte loads' alignment too"));

/// Whether a byte load's value only goes, extended, into integer arithmetic
/// (an int8 dot product's products): no bytes in 16-bit halves that a float
/// conversion or a compare reads (q8_matvec, a byte mask's select lost).
static bool integerBytes(const Instruction &I) {
  return !I.use_empty() && all_of(I.users(), [](const User *U) {
    auto *C = dyn_cast<CastInst>(U);
    return C && (isa<SExtInst>(C) || isa<ZExtInst>(C)) && !C->use_empty() &&
           all_of(C->users(), [](const User *V) {
             auto *B = dyn_cast<BinaryOperator>(V);
             return B && B->getType()->isIntegerTy() &&
                    (B->getOpcode() == Instruction::Mul ||
                     B->getOpcode() == Instruction::Add ||
                     B->getOpcode() == Instruction::Sub);
           });
  });
}

static cl::opt<bool> VersionByteStores(
    "agx-align-version-byte-stores", cl::init(true), cl::Hidden,
    cl::desc("Raise scalar byte stores' alignment too"));

static cl::opt<bool> VersionByteVectors(
    "agx-alignment-versioning-byte-vectors", cl::init(true), cl::Hidden,
    cl::desc("Count vector-of-byte accesses too (their lanes extended in "
             "place)"));

namespace {

class AGXAlignmentVersioning : public FunctionPass {
public:
  static char ID;
  AGXAlignmentVersioning() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Alignment Versioning"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }
};

struct Raise {
  Instruction *I;
  Align New;
};

} // namespace

char AGXAlignmentVersioning::ID = 0;

INITIALIZE_PASS_BEGIN(AGXAlignmentVersioning, DEBUG_TYPE,
                      "AGX Alignment Versioning", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AGXAlignmentVersioning, DEBUG_TYPE,
                    "AGX Alignment Versioning", false, false)

FunctionPass *llvm::createAGXAlignmentVersioningPass() {
  return new AGXAlignmentVersioning();
}

/// Whether V is a kernel argument, or casts of one (the PTX front end's
/// addrspacecast (inttoptr %arg)): the same for every thread.
static bool isArgumentValue(const Value *V) {
  for (unsigned Depth = 0; Depth < 4; ++Depth) {
    if (isa<Argument>(V))
      return true;
    auto *C = dyn_cast<CastInst>(V);
    if (!C || !(isa<IntToPtrInst>(C) || isa<PtrToIntInst>(C) ||
                isa<AddrSpaceCastInst>(C) || isa<BitCastInst>(C)))
      return false;
    V = C->getOperand(0);
  }
  return false;
}

/// A term of the integer address S that is a kernel argument value (or its
/// ptrtoint), and that value: S itself, a term of a sum, or of a
/// recurrence's start.
static std::pair<const SCEV *, Value *> argumentTerm(const SCEV *S) {
  if (auto *U = dyn_cast<SCEVUnknown>(S))
    if (isArgumentValue(U->getValue()))
      return {U, U->getValue()};
  if (auto *P = dyn_cast<SCEVPtrToAddrExpr>(S))
    if (auto *U = dyn_cast<SCEVUnknown>(P->getOperand()))
      if (isArgumentValue(U->getValue()))
        return {P, U->getValue()};
  if (auto *AR = dyn_cast<SCEVAddRecExpr>(S))
    return argumentTerm(AR->getStart());
  if (auto *A = dyn_cast<SCEVAddExpr>(S))
    for (const SCEV *Op : A->operands())
      if (auto T = argumentTerm(Op); T.first)
        return T;
  return {nullptr, nullptr};
}

/// The kernel argument value Ptr is an offset from, and the offset: a
/// pointer argument (MLIR's memrefs), or one in the integer sum the pointer
/// is (inttoptr, as the PTX front end computes addresses).
static std::pair<Value *, const SCEV *> baseAndOffset(Value *Ptr,
                                                      ScalarEvolution &SE) {
  const SCEV *S = SE.getSCEV(Ptr);
  auto *Base = dyn_cast<SCEVUnknown>(SE.getPointerBase(S));
  if (!Base)
    return {nullptr, nullptr};
  const SCEV *Off = SE.getMinusSCEV(S, Base);
  if (isa<SCEVCouldNotCompute>(Off))
    return {nullptr, nullptr};
  if (isArgumentValue(Base->getValue()))
    return {Base->getValue(), Off};
  auto *ITP = dyn_cast<IntToPtrInst>(Base->getValue());
  if (!ITP || !ITP->getSrcTy()->isIntegerTy(64))
    return {nullptr, nullptr};
  const SCEV *Int = SE.getSCEV(ITP->getOperand(0));
  auto [Term, V] = argumentTerm(Int);
  if (!Term)
    return {nullptr, nullptr};
  const SCEV *IntOff = SE.getMinusSCEV(Int, Term);
  if (isa<SCEVCouldNotCompute>(IntOff) || Off->getType() != IntOff->getType())
    return {nullptr, nullptr};
  return {V, SE.getAddExpr(IntOff, Off)};
}

/// The device accesses of L whose alignment a 4-byte aligned kernel argument
/// base would raise, and those bases.
static void findRaisesIn(BasicBlock &BB, ScalarEvolution &SE,
                         SmallVectorImpl<Raise> &Raises,
                         SetVector<Value *> &Bases);

static void findRaises(Loop &L, ScalarEvolution &SE,
                       SmallVectorImpl<Raise> &Raises,
                       SetVector<Value *> &Bases) {
  findRaisesIn(*L.getHeader(), SE, Raises, Bases);
}

static void findRaisesIn(BasicBlock &BB, ScalarEvolution &SE,
                         SmallVectorImpl<Raise> &Raises,
                         SetVector<Value *> &Bases) {
  for (Instruction &I : BB) {
    Value *Ptr;
    Align Old;
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!LI->isSimple())
        continue;
      Ptr = LI->getPointerOperand();
      Old = LI->getAlign();
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isSimple())
        continue;
      Ptr = SI->getPointerOperand();
      Old = SI->getAlign();
    } else {
      continue;
    }
    if (Old >= Align(4) ||
        Ptr->getType()->getPointerAddressSpace() != AGX::DEVICE_ADDRESS)
      continue;
    // Not of bytes: a 4-byte load leaves each in a 16-bit half, which a
    // sign extension copied out first (dot_i8: 29 -> 33 GPRs, tput +6%).
    // Since L15-E487 a vector of bytes' lanes are extended in their halves
    // (a 16-bit source of their user): those count. (Scalar bytes still not:
    // q8_matvec's converted bytes 201.9 -> 217.8 tput, q4_matvec's 289.4 ->
    // 293.2.) Byte stores count: aligned, four go out as one access of the
    // byte format from 16-bit halves (int8 quantization, masks: L15-E547);
    // and bytes that only go, extended, into integer products and sums (an
    // int8 dot product's, a load each: L15-E560).
    if (getLoadStoreType(&I)->getScalarSizeInBits() < 16 &&
        !(VersionByteVectors && getLoadStoreType(&I)->isVectorTy()) &&
        !(VersionByteStores && isa<StoreInst>(I)) &&
        !(VersionByteLoads && isa<LoadInst>(I) && integerBytes(I)))
      continue;
    auto [Base, Off] = baseAndOffset(Ptr, SE);
    if (!Base)
      continue;
    Align New(uint64_t(1) << std::min(SE.getMinTrailingZeros(Off), 2u));
    if (New <= Old)
      continue;
    Raises.push_back({&I, New});
    Bases.insert(Base);
  }
}

static bool versionLoop(Loop &L, Function &F, LoopInfo &LI, DominatorTree &DT,
                        ScalarEvolution &SE) {
  SmallVector<Raise, 16> Raises;
  SetVector<Value *> Bases;
  findRaises(L, SE, Raises, Bases);
  // A vector access counts as its elements: below 4 bytes each element is
  // an access of its own (MLIR's vector<4xf16> at align 2: L15-E478).
  unsigned Accesses = 0;
  for (const Raise &R : Raises) {
    auto *VT = dyn_cast<FixedVectorType>(getLoadStoreType(R.I));
    Accesses += VT && CountVectorElements ? VT->getNumElements() : 1;
  }
  LLVM_DEBUG(dbgs() << "agx-align-versioning: " << F.getName() << " "
                    << L.getHeader()->getName() << " " << Accesses
                    << " raises\n");
  if (Accesses < AlignmentVersioningMin)
    return false;
  for (Value *Base : Bases)
    if (auto *BI = dyn_cast<Instruction>(Base);
        BI && !DT.dominates(BI, L.getLoopPreheader()->getTerminator()))
      return false;

  // Values of the loop used after it go through its exit block's phis.
  formLCSSA(L, DT, &LI, &SE);
  BasicBlock *Exiting = L.getExitingBlock();
  BasicBlock *Exit = L.getExitBlock();

  // Check -> PH -> loop: the check branches to PH or to the clone's
  // preheader.
  BasicBlock *Check = L.getLoopPreheader();
  BasicBlock *PH = SplitBlock(Check, Check->getTerminator(), &DT, &LI,
                              nullptr, Check->getName() + ".unaligned");
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 4> NewBlocks;
  cloneLoopWithPreheader(PH, Check, &L, VMap, ".aligned", &LI, &DT, NewBlocks);
  remapInstructionsInBlocks(NewBlocks, VMap);
  auto *NewPH = cast<BasicBlock>(VMap[PH]);
  auto *NewExiting = cast<BasicBlock>(VMap[Exiting]);
  for (PHINode &P : Exit->phis()) {
    Value *V = P.getIncomingValueForBlock(Exiting);
    Value *NV = VMap.lookup(V);
    P.addIncoming(NV ? NV : V, NewExiting);
  }

  IRBuilder<> B(Check->getTerminator());
  Value *Cond = nullptr;
  for (Value *Base : Bases) {
    Value *Int = Base->getType()->isPointerTy()
                     ? B.CreatePtrToInt(Base, B.getInt64Ty())
                     : Base;
    Value *Low = B.CreateAnd(Int, 3);
    Value *Ok = B.CreateICmpEQ(Low, B.getInt64(0), Base->getName() + ".al4");
    Cond = Cond ? B.CreateAnd(Cond, Ok) : Ok;
  }
  Instruction *Old = Check->getTerminator();
  B.CreateCondBr(Cond, NewPH, PH);
  Old->eraseFromParent();
  DT.recalculate(F);

  for (const Raise &R : Raises) {
    auto *I = cast<Instruction>(VMap[R.I]);
    if (auto *LdI = dyn_cast<LoadInst>(I))
      LdI->setAlignment(R.New);
    else
      cast<StoreInst>(I)->setAlignment(R.New);
  }
  SE.forgetLoop(&L);
  return true;
}

static cl::opt<bool> GuardVersioningMerge(
    "agx-guard-versioning-merge", cl::init(true), cl::Hidden,
    cl::desc("Merge the guard-free copy's blocks"));

static cl::opt<unsigned> GuardVersioningMin(
    "agx-guard-versioning-min", cl::init(4), cl::Hidden,
    cl::desc("Loop-invariant guards of loads an innermost loop must have to "
             "get a copy without them where they all hold (0: off)"));

/// A loop whose work is guarded by loop-invariant tests (a convolution's
/// taps inside the image) waits for each guarded load apart under its own
/// execution mask. Where every guard holds (the threads inside the image),
/// a copy of the loop without the guards runs instead, its loads together:
///   if (g1 && g2 && ...) loop without the tests; else the loop
/// Both compute the same; the copy runs only where its dropped tests hold.
static bool versionGuards(Loop &L, Function &F, LoopInfo &LI,
                          DominatorTree &DT, ScalarEvolution &SE,
                          DenseMap<BasicBlock *, CallInst *> &Ballots) {
  if (L.getLoopPreheader() && L.getExitBlock() && !L.hasDedicatedExits())
    formDedicatedExitBlocks(&L, &DT, &LI, nullptr, /*PreserveLCSSA=*/false);
  BasicBlock *Pre = L.getLoopPreheader();
  BasicBlock *Exiting = L.getExitingBlock(), *Exit = L.getExitBlock();
  if (!Pre || !Exiting || !Exit || !L.hasDedicatedExits())
    return false;
  struct Guard {
    CondBrInst *Br;
    bool ToTrue; // the guarded block is the true successor
  };
  SmallVector<Guard, 8> Guards;
  for (BasicBlock *BB : L.blocks()) {
    auto *Br = dyn_cast<CondBrInst>(BB->getTerminator());
    if (!Br || BB == Exiting || !L.isLoopInvariant(Br->getCondition()))
      continue;
    if (auto *C = dyn_cast<Instruction>(Br->getCondition());
        C && !DT.dominates(C, Pre->getTerminator()))
      continue;
    for (unsigned K = 0; K < 2; ++K) {
      BasicBlock *T = Br->getSuccessor(K), *M = Br->getSuccessor(1 - K);
      if (T->getSinglePredecessor() == BB && T->getSingleSuccessor() == M &&
          L.contains(T) && any_of(*T, [](Instruction &I) {
            return isa<LoadInst>(I);
          })) {
        Guards.push_back({Br, K == 0});
        break;
      }
    }
  }
  if (Guards.size() < GuardVersioningMin)
    return false;
  formLCSSA(L, DT, &LI, &SE);
  Exiting = L.getExitingBlock();
  Exit = L.getExitBlock();
  BasicBlock *Check = L.getLoopPreheader();
  BasicBlock *PH = SplitBlock(Check, Check->getTerminator(), &DT, &LI, nullptr,
                              Check->getName() + ".guarded");
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 16> NewBlocks;
  cloneLoopWithPreheader(PH, Check, &L, VMap, ".inside", &LI, &DT, NewBlocks);
  remapInstructionsInBlocks(NewBlocks, VMap);
  auto *NewPH = cast<BasicBlock>(VMap[PH]);
  auto *NewExiting = cast<BasicBlock>(VMap[Exiting]);
  for (PHINode &P : Exit->phis()) {
    Value *V = P.getIncomingValueForBlock(Exiting);
    Value *NV = VMap.lookup(V);
    P.addIncoming(NV ? NV : V, NewExiting);
  }
  IRBuilder<> B(Check->getTerminator());
  Value *All = nullptr;
  for (const Guard &G : Guards) {
    Value *C = G.Br->getCondition();
    if (!G.ToTrue)
      C = B.CreateNot(C);
    All = All ? B.CreateAnd(All, C) : C;
  }
  // The same for the whole SIMD-group: the copy only where every active
  // thread may run it, so that no SIMD-group runs both loops (one with a
  // thread at the border runs the original alone).
  auto *GuardBallot = cast<CallInst>(
      B.CreateIntrinsic(Intrinsic::agx_simd_ballot, {}, {All}));
  All = B.CreateICmpEQ(
      GuardBallot,
      B.CreateIntrinsic(Intrinsic::agx_simd_ballot, {}, {B.getTrue()}));
  Instruction *Old = Check->getTerminator();
  B.CreateCondBr(All, NewPH, PH);
  Old->eraseFromParent();
  Ballots[cast<BasicBlock>(VMap[L.getHeader()])] = GuardBallot;
  // The copy enters every guarded block.
  for (const Guard &G : Guards) {
    auto *Br = cast<CondBrInst>(VMap[G.Br]);
    BasicBlock *BB = Br->getParent();
    BasicBlock *T = Br->getSuccessor(G.ToTrue ? 0 : 1);
    BasicBlock *M = Br->getSuccessor(G.ToTrue ? 1 : 0);
    for (PHINode &P : M->phis())
      P.removeIncomingValue(BB, /*DeletePHIIfEmpty=*/false);
    IRBuilder<>(Br).CreateBr(T);
    Br->eraseFromParent();
  }
  // Its blocks, a chain now, as one where they can be (the load/store
  // vectorizer then merges the accesses of neighbouring taps: a
  // convolution's nine weights in two 16-byte loads and one 4-byte one).
  if (GuardVersioningMerge) {
    auto *NewHeader = cast<BasicBlock>(VMap[L.getHeader()]);
    SmallVector<WeakVH, 16> Blocks(NewBlocks.begin(), NewBlocks.end());
    for (WeakVH &V : Blocks)
      if (auto *BB = dyn_cast_or_null<BasicBlock>(V))
        if (BB != NewHeader && BB != NewPH)
          MergeBlockIntoPredecessor(BB, /*DTU=*/nullptr, &LI);
    // The guarded blocks' own inttoptrs of one integer address (the PTX
    // front end's) as one: the vectorizer takes each inttoptr for a base of
    // its own.
    for (WeakVH &V : Blocks)
      if (auto *BB = dyn_cast_or_null<BasicBlock>(V)) {
        DenseMap<std::pair<Value *, Type *>, Instruction *> Seen;
        for (Instruction &I : make_early_inc_range(*BB)) {
          auto *P = dyn_cast<IntToPtrInst>(&I);
          if (!P)
            continue;
          auto [It, New] =
              Seen.try_emplace({P->getOperand(0), P->getType()}, P);
          if (New)
            continue;
          P->replaceAllUsesWith(It->second);
          P->eraseFromParent();
        }
      }
  }
  DT.recalculate(F);
  SE.forgetLoop(&L);
  return true;
}

static cl::opt<bool> AGXTapBasesInGuards(
    "agx-tap-bases-in-guards", cl::init(true), cl::Hidden,
    cl::desc("For a guard-free copy, fold the tap bases' fit tests into its "
             "own test instead of a third copy"));

static cl::opt<unsigned> TapBasesMin(
    "agx-tap-bases-min", cl::init(3), cl::Hidden,
    cl::desc("Distinct invariant k_j of device accesses at inttoptr(k_j + "
             "p), p stepped by an innermost loop and not an extended 32-bit "
             "index, a loop must have to get a copy addressing them from one "
             "base (0: off)"));

/// The PTX front end's accesses at integer addresses inttoptr(k_j + p): a
/// convolution's taps, p the channel's offset stepped each iteration and
/// k_j = in + 4 (row_j + x) for each tap's row and column. Loop strength
/// reduction keeps a 64-bit pointer for each k_j, each stepped by a 64-bit
/// add. Where every k_j - k_0 is a multiple of the access's element size e
/// whose quotient fits 32 signed bits, the access is at the one stepped base
/// inttoptr(k_0 + p) plus sext(i32 (k_j - k_0) / e) elements, an index
/// register. A copy of the loop does so where that holds for the whole
/// SIMD-group (a ballot, as versionGuards):
///   if (all fit) loop from one base; else the loop
/// (L15-E395).
static bool versionTapBases(Loop &L, Function &F, LoopInfo &LI,
                            DominatorTree &DT, ScalarEvolution &SE,
                            CallInst *GuardBallot = nullptr) {
  if (!TapBasesMin)
    return false;
  // (versionGuards' copy shares its exit block with the loop it copies)
  if (L.getLoopPreheader() && L.getExitBlock() && !L.hasDedicatedExits())
    formDedicatedExitBlocks(&L, &DT, &LI, nullptr, /*PreserveLCSSA=*/false);
  BasicBlock *Pre = L.getLoopPreheader();
  BasicBlock *Exiting = L.getExitingBlock(), *Exit = L.getExitBlock();
  if (!Pre || !Exiting || !Exit || !L.hasDedicatedExits() ||
      !L.isLoopSimplifyForm())
    return false;
  // An access point: the inttoptr, or a constant-offset GEP of it, whose
  // users are all accesses at it.
  struct Tap {
    Instruction *Ptr;
    Value *K;
    int64_t C; // bytes past k_j + p
  };
  auto Accesses = [](Instruction *Ptr, uint64_t &Elt, const DataLayout &DL) {
    Elt = 0;
    for (User *U : Ptr->users()) {
      Type *Ty;
      if (auto *Ld = dyn_cast<LoadInst>(U);
          Ld && Ld->isSimple() && Ld->getPointerOperand() == Ptr)
        Ty = Ld->getType();
      else if (auto *St = dyn_cast<StoreInst>(U);
               St && St->isSimple() && St->getPointerOperand() == Ptr)
        Ty = St->getValueOperand()->getType();
      else
        return false;
      uint64_t E = DL.getTypeStoreSize(Ty->getScalarType()).getFixedValue();
      if (Elt && E != Elt)
        return false;
      Elt = E;
    }
    return Elt == 1 || Elt == 2 || Elt == 4 || Elt == 8;
  };
  // the access points of each stepped p
  MapVector<Value *, SmallVector<Tap, 8>> Groups;
  const DataLayout &DL = F.getDataLayout();
  uint64_t Elt = 0;
  for (BasicBlock *BB : L.blocks())
    for (Instruction &I : *BB) {
      auto *P = dyn_cast<IntToPtrInst>(&I);
      if (!P || P->getType()->getPointerAddressSpace() != AGX::DEVICE_ADDRESS)
        continue;
      auto *A = dyn_cast<BinaryOperator>(P->getOperand(0));
      if (!A || A->getOpcode() != Instruction::Add || !L.contains(A) ||
          !A->getType()->isIntegerTy(64))
        continue;
      Value *K = A->getOperand(0), *V = A->getOperand(1);
      if (!L.isLoopInvariant(K))
        std::swap(K, V);
      if (!L.isLoopInvariant(K) || L.isLoopInvariant(V))
        continue;
      if (auto *KI = dyn_cast<Instruction>(K);
          KI && !DT.dominates(KI, Pre->getTerminator()))
        continue;
      // (p an extended 32-bit index, scaled: loop strength reduction already
      // takes it as the accesses' index register)
      {
        using namespace PatternMatch;
        if (match(V, m_ZExtOrSExt(m_Value())) ||
            match(V, m_Shl(m_ZExtOrSExt(m_Value()), m_ConstantInt())) ||
            match(V, m_Mul(m_ZExtOrSExt(m_Value()), m_ConstantInt())))
          continue;
      }
      SmallVector<Tap, 4> Points;
      bool OK = true;
      for (User *U : P->users()) {
        if (isa<LoadInst>(U) || isa<StoreInst>(U))
          continue;
        auto *G = dyn_cast<GetElementPtrInst>(U);
        APInt Off(64, 0);
        uint64_t E;
        if (!G || G->getPointerOperand() != P ||
            !G->accumulateConstantOffset(DL, Off) || !Accesses(G, E, DL) ||
            (Elt && E != Elt)) {
          OK = false;
          break;
        }
        Elt = E;
        Points.push_back({G, K, Off.getSExtValue()});
      }
      bool Direct = any_of(P->users(), [](User *U) {
        return isa<LoadInst>(U) || isa<StoreInst>(U);
      });
      if (!OK)
        continue;
      if (Direct) {
        // (the inttoptr's own accesses, its GEPs aside)
        for (User *U : P->users()) {
          Type *Ty = nullptr;
          if (auto *Ld = dyn_cast<LoadInst>(U))
            Ty = Ld->isSimple() && Ld->getPointerOperand() == P ? Ld->getType()
                                                                 : nullptr;
          else if (auto *St = dyn_cast<StoreInst>(U))
            Ty = St->isSimple() && St->getPointerOperand() == P
                     ? St->getValueOperand()->getType()
                     : nullptr;
          else
            continue;
          if (!Ty) {
            OK = false;
            break;
          }
          uint64_t T = DL.getTypeStoreSize(Ty->getScalarType()).getFixedValue();
          if (Elt && T != Elt) {
            OK = false;
            break;
          }
          Elt = T;
        }
        if (!OK)
          continue;
        Points.push_back({P, K, 0});
      }
      auto &G = Groups[V];
      G.append(Points.begin(), Points.end());
    }
  if (Elt != 1 && Elt != 2 && Elt != 4 && Elt != 8)
    return false;
  // the group with the most distinct bases, and at least TapBasesMin
  Value *Step = nullptr;
  SmallVector<Tap, 8> Taps;
  unsigned Best = 0;
  for (auto &[V, G] : Groups) {
    SmallPtrSet<Value *, 8> Ks;
    for (const Tap &T : G)
      Ks.insert(T.K);
    if (Ks.size() >= TapBasesMin && Ks.size() > Best)
      Best = Ks.size(), Step = V, Taps = G;
  }
  if (!Step)
    return false;
  Value *K0 = Taps.front().K;
  for (const Tap &T : Taps)
    if (T.K == K0 && T.C % int64_t(Elt))
      return false;

  // each (k_j - k_0 + c) / e exact and 32 bits
  unsigned Shift = Log2_64(Elt);
  std::map<std::pair<Value *, int64_t>, Value *> Index;
  auto Checks = [&](IRBuilder<> &B) {
    Value *All = B.getTrue();
    for (const Tap &T : Taps) {
      if (T.K == K0 || Index.count({T.K, T.C}))
        continue;
      Value *D = B.CreateAdd(B.CreateSub(T.K, K0), B.getInt64(T.C));
      Value *Q = B.CreateAShr(D, Shift);
      Value *Q32 = B.CreateTrunc(Q, B.getInt32Ty());
      Value *Fits = B.CreateICmpEQ(B.CreateSExt(Q32, B.getInt64Ty()), Q);
      if (Shift)
        Fits = B.CreateAnd(
            Fits, B.CreateICmpEQ(B.CreateAnd(D, Elt - 1), B.getInt64(0)));
      All = B.CreateAnd(All, Fits);
      Index[{T.K, T.C}] = Q32;
    }
    return All;
  };
  // A versionGuards copy, whose own test already sends the SIMD-groups
  // that fail it to the loop it copies: the fit tests join that test and the
  // copy is rewritten in place (no third copy, whose pointers would count
  // in the kernel's registers: conv3x3, L15-E408).
  ValueToValueMapTy VMap;
  BasicBlock *NewHeader;
  if (GuardBallot && AGXTapBasesInGuards &&
      all_of(Taps, [&](const Tap &T) {
        auto *KI = dyn_cast<Instruction>(T.K);
        return !KI || DT.dominates(KI, GuardBallot);
      })) {
    IRBuilder<> B(GuardBallot);
    Value *All = Checks(B);
    GuardBallot->setArgOperand(0,
                               B.CreateAnd(GuardBallot->getArgOperand(0), All));
    for (const Tap &T : Taps)
      VMap[T.Ptr] = T.Ptr;
    NewHeader = L.getHeader();
  } else {
    formLCSSA(L, DT, &LI, &SE);
    Exiting = L.getExitingBlock();
    Exit = L.getExitBlock();
    BasicBlock *Check = L.getLoopPreheader();
    BasicBlock *PH = SplitBlock(Check, Check->getTerminator(), &DT, &LI,
                                nullptr, Check->getName() + ".taps");
    SmallVector<BasicBlock *, 16> NewBlocks;
    cloneLoopWithPreheader(PH, Check, &L, VMap, ".onebase", &LI, &DT,
                           NewBlocks);
    remapInstructionsInBlocks(NewBlocks, VMap);
    auto *NewPH = cast<BasicBlock>(VMap[PH]);
    auto *NewExiting = cast<BasicBlock>(VMap[Exiting]);
    for (PHINode &Phi : Exit->phis()) {
      Value *V = Phi.getIncomingValueForBlock(Exiting);
      Value *NV = VMap.lookup(V);
      Phi.addIncoming(NV ? NV : V, NewExiting);
    }
    IRBuilder<> B(Check->getTerminator());
    Value *All = Checks(B);
    All = B.CreateICmpEQ(
        B.CreateIntrinsic(Intrinsic::agx_simd_ballot, {}, {All}),
        B.CreateIntrinsic(Intrinsic::agx_simd_ballot, {}, {B.getTrue()}));
    Instruction *Old = Check->getTerminator();
    B.CreateCondBr(All, NewPH, PH);
    Old->eraseFromParent();
    NewHeader = cast<BasicBlock>(VMap[L.getHeader()]);
  }
  // the copy's accesses from one base, inttoptr(k_0 + p) where p is
  Value *NewStep = VMap.lookup(Step);
  if (!NewStep)
    NewStep = Step;
  IRBuilder<> HB(&*NewHeader->getFirstInsertionPt());
  if (auto *SI = dyn_cast<Instruction>(NewStep); SI && !isa<PHINode>(SI))
    HB.SetInsertPoint(SI->getNextNode());
  Type *PtrTy = PointerType::get(F.getContext(), AGX::DEVICE_ADDRESS);
  Value *Base = HB.CreateIntToPtr(HB.CreateAdd(K0, NewStep), PtrTy, "onebase");
  Type *EltTy = HB.getIntNTy(8 * Elt);
  SmallVector<Instruction *, 8> Dead;
  for (const Tap &T : Taps) {
    auto *NP = cast<Instruction>(VMap[T.Ptr]);
    IRBuilder<> PB(NP);
    Value *R;
    if (T.K == K0)
      R = T.C ? PB.CreateGEP(PB.getInt8Ty(), Base, PB.getInt64(T.C)) : Base;
    else
      R = PB.CreateGEP(EltTy, Base,
                       PB.CreateSExt(Index[{T.K, T.C}], PB.getInt64Ty()));
    NP->replaceAllUsesWith(R);
    Dead.push_back(NP);
  }
  // (with the inttoptrs and additions only they used: loop strength
  // reduction would keep a pointer for each)
  SmallVector<WeakTrackingVH, 8> Gone(Dead.begin(), Dead.end());
  for (WeakTrackingVH &V : Gone)
    if (auto *I = dyn_cast_or_null<Instruction>(V))
      RecursivelyDeleteTriviallyDeadInstructions(I);
  DT.recalculate(F);
  SE.forgetLoop(&L);
  return true;
}

static cl::opt<unsigned> StraightLineVersioningMin(
    "agx-straight-line-versioning-min", cl::init(16), cl::Hidden,
    cl::desc("Fewest elements of a one-block kernel's device accesses whose "
             "alignment a 4-byte aligned base raises for its body to be "
             "versioned on it (0: none)"));

/// A kernel of one block (no loops: opt unrolled them whole, as E511's
/// constant trip counts let it) with many device accesses whose alignment a
/// 4-byte aligned kernel-argument base would raise (rmsnorm_f16's
/// vector<4xf16> at align 2, a 16-bit access each element) runs a copy of
/// itself at the raised alignment where every such base is aligned
/// (L15-E522; E495 for fewer accesses lost on registers).
static bool versionBody(Function &F, ScalarEvolution &SE) {
  if (!StraightLineVersioningMin || F.size() != 1 ||
      !isa<ReturnInst>(F.getEntryBlock().getTerminator()))
    return false;
  BasicBlock *BB = &F.getEntryBlock();
  if (any_of(*BB, [](Instruction &I) { return isa<AllocaInst>(I); }))
    return false;
  SmallVector<Raise, 16> Raises;
  SetVector<Value *> Bases;
  findRaisesIn(*BB, SE, Raises, Bases);
  unsigned Elements = 0;
  for (const Raise &R : Raises) {
    auto *VT = dyn_cast<FixedVectorType>(getLoadStoreType(R.I));
    Elements += VT ? VT->getNumElements() : 1;
  }
  if (Elements < StraightLineVersioningMin ||
      any_of(Bases, [](Value *B) { return !isa<Argument>(B); }))
    return false;
  LLVMContext &Ctx = F.getContext();
  ValueToValueMapTy VMap;
  BasicBlock *Al = CloneBasicBlock(BB, VMap, ".aligned", &F);
  for (Instruction &I : *Al)
    RemapInstruction(&I, VMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
  for (const Raise &R : Raises) {
    auto *I = cast<Instruction>(VMap[R.I]);
    if (auto *LdI = dyn_cast<LoadInst>(I))
      LdI->setAlignment(R.New);
    else
      cast<StoreInst>(I)->setAlignment(R.New);
  }
  BasicBlock *Check = BasicBlock::Create(Ctx, "align.check", &F, BB);
  IRBuilder<> B(Check);
  Value *Cond = nullptr;
  for (Value *Base : Bases) {
    Value *Int = Base->getType()->isPointerTy()
                     ? B.CreatePtrToInt(Base, B.getInt64Ty())
                     : Base;
    Value *Ok = B.CreateICmpEQ(B.CreateAnd(Int, 3), B.getInt64(0),
                               Base->getName() + ".al4");
    Cond = Cond ? B.CreateAnd(Cond, Ok) : Ok;
  }
  B.CreateCondBr(Cond, Al, BB);
  return true;
}

bool AGXAlignmentVersioning::runOnFunction(Function &F) {
  if (!EnableAlignmentVersioning || skipFunction(F))
    return false;
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  if (LI.empty() && versionBody(F, SE))
    return true;
  // Innermost single-block loops in simplified form with one exit.
  SmallVector<Loop *, 4> Loops;
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (!L->isInnermost() || L->getNumBlocks() != 1 ||
        !L->getLoopPreheader() || !L->getLoopLatch() || !L->getExitBlock() ||
        !L->getExitingBlock())
      continue;
    // (an exit block shared with the path around the loop gets one of its
    // own, for the copies' values to meet in)
    if (!L->hasDedicatedExits())
      Changed |= formDedicatedExitBlocks(L, &DT, &LI, nullptr,
                                         /*PreserveLCSSA=*/false);
    if (L->isLoopSimplifyForm() && L->getExitBlock())
      Loops.push_back(L);
  }
  // (not alias versioning's copy for overlapping ranges, which runs only
  // when they overlap: three copies instead of four, L15-E547)
  for (Loop *L : Loops)
    if (!SkipAliasFallback ||
        !findStringMetadataForLoop(L, "llvm.loop.agx.alias.fallback"))
      Changed |= versionLoop(*L, F, LI, DT, SE);
  DenseMap<BasicBlock *, CallInst *> Ballots;
  if (GuardVersioningMin) {
    SmallVector<Loop *, 4> Inner;
    for (Loop *L : LI.getLoopsInPreorder())
      if (L->isInnermost() && L->getNumBlocks() > 1)
        Inner.push_back(L);
    for (Loop *L : Inner)
      Changed |= versionGuards(*L, F, LI, DT, SE, Ballots);
  }
  if (TapBasesMin) {
    SmallVector<Loop *, 4> Inner;
    // (single blocks: versionGuards' copies, not the loops they guard)
    for (Loop *L : LI.getLoopsInPreorder())
      if (L->isInnermost() && L->getNumBlocks() == 1)
        Inner.push_back(L);
    for (Loop *L : Inner)
      Changed |= versionTapBases(*L, F, LI, DT, SE,
                                 Ballots.lookup(L->getHeader()));
  }
  return Changed;
}
