//===-- AGXWordCounterVersioning.cpp - A loop's byte offset as words ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The PTX front end's loops count 64-bit byte offsets, which loop strength
// reduction keeps (a decode step of attention over its keys and values):
//
//   %off = phi i64 [ %start, %ph ], [ %off.next, %latch ]
//   %a = add i64 %k.base, %off        ; inttoptr, a word access
//   %b = add i64 %v.base, %off        ; inttoptr, a word access
//   %off.next = add i64 %off, %step
//
// Each address is a 64-bit add (on the core half's shared unit), and so is
// the step; AGXNarrowInductions keeps them where it cannot bound the offset.
// A device access's index register counts elements of its format (words for
// these), so the loop runs as two copies:
//
//   if (start and step are multiples of 4, and start / 4, step / 4 and the
//       last offset / 4 fit in 32 bits, for the whole SIMD-group)
//     loop with a 32-bit counter of words w, each access gep i32 base, w
//   else
//     loop as it was
//
// One 32-bit add per iteration instead of 64-bit ones (L15-E674). Only where
// the offset's every other use is such an address of a word access, and
// scalar evolution counts the loop's iterations.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-word-counter-versioning"

static cl::opt<bool> EnableWordCounterVersioning(
    "agx-loop-word-counters", cl::init(true), cl::Hidden,
    cl::desc("Version loops whose 64-bit byte offset only addresses word "
             "accesses on its fitting a 32-bit counter of words"));

static cl::opt<unsigned> WordCounterMaxBases(
    "agx-loop-word-counters-max-bases", cl::init(2), cl::Hidden,
    cl::desc("Distinct constant offsets a counter's accesses from one base "
             "may have for -agx-loop-word-counters"));

static cl::opt<unsigned> WordCounterMaxTotalBases(
    "agx-loop-word-counters-max-total-bases", cl::init(4), cl::Hidden,
    cl::desc("Base registers a loop's copy may take for "
             "-agx-loop-word-counters"));

namespace {

class AGXWordCounterVersioning : public FunctionPass {
public:
  static char ID;
  AGXWordCounterVersioning() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "AGX Word Counter Versioning";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }
};

/// An access at X + P + C for a counter P (X loop-invariant or null, C a
/// constant).
struct Access {
  Instruction *I;
  PHINode *P;
  Value *X;
  int64_t C;
};

} // namespace

char AGXWordCounterVersioning::ID = 0;

INITIALIZE_PASS_BEGIN(AGXWordCounterVersioning, DEBUG_TYPE,
                      "AGX Word Counter Versioning", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AGXWordCounterVersioning, DEBUG_TYPE,
                    "AGX Word Counter Versioning", false, false)

FunctionPass *llvm::createAGXWordCounterVersioningPass() {
  return new AGXWordCounterVersioning();
}

/// Whether I is a device load or store of whole words at least word
/// aligned whose pointer operand is P.
static bool isWordAccess(Instruction *I, Value *P, const DataLayout &DL) {
  Type *Ty;
  Align A;
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    if (LI->getPointerOperand() != P || LI->isVolatile() || LI->isAtomic())
      return false;
    Ty = LI->getType();
    A = LI->getAlign();
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    if (SI->getPointerOperand() != P || SI->isVolatile() || SI->isAtomic())
      return false;
    Ty = SI->getValueOperand()->getType();
    A = SI->getAlign();
  } else {
    return false;
  }
  return A.value() >= 4 && DL.getTypeStoreSize(Ty).getFixedValue() % 4 == 0;
}

/// The accesses that Add (base + offset) addresses, through an inttoptr to
/// the device address space and constant byte offsets, if those are its
/// only uses.
static bool collectAccesses(Instruction *Add, PHINode *P, Value *Base,
                            Loop *L, const DataLayout &DL,
                            SmallVectorImpl<Access> &Out) {
  for (User *U : Add->users()) {
    // (a counter's own other users, its step and adds of bases, are the
    // caller's)
    if (isa<PHINode>(Add) && !isa<IntToPtrInst>(U))
      continue;
    auto *ITP = dyn_cast<IntToPtrInst>(U);
    if (!ITP || !L->contains(ITP) || ITP->getType()->getPointerAddressSpace() != 1)
      return false;
    for (User *V : ITP->users()) {
      auto *VI = cast<Instruction>(V);
      if (isWordAccess(VI, ITP, DL)) {
        Out.push_back({VI, P, Base, 0});
        continue;
      }
      auto *G = dyn_cast<GetElementPtrInst>(VI);
      if (!G || G->getPointerOperand() != ITP || !G->getSourceElementType()->isIntegerTy(8) ||
          G->getNumIndices() != 1 || !isa<ConstantInt>(G->getOperand(1)))
        return false;
      int64_t C = cast<ConstantInt>(G->getOperand(1))->getSExtValue();
      if (C % 4 || !isInt<20>(C))
        return false;
      for (User *W : G->users())
        if (!isWordAccess(cast<Instruction>(W), G, DL))
          return false;
        else
          Out.push_back({cast<Instruction>(W), P, Base, C});
    }
  }
  return true;
}

/// Versions L on Unsafe: L runs where it is false, a copy as L was where it
/// is true (L in simplify form, one exit block, LCSSA form).
static void versionOn(Loop *L, Value *Unsafe, LoopInfo &LI,
                      DominatorTree &DT) {
  BasicBlock *CheckBB = L->getLoopPreheader();
  BasicBlock *PH = SplitBlock(CheckBB, CheckBB->getTerminator(), &DT, &LI,
                              nullptr, L->getHeader()->getName() + ".words.ph");
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> Blocks;
  Loop *Orig = cloneLoopWithPreheader(PH, CheckBB, L, VMap, ".words.orig",
                                      &LI, &DT, Blocks);
  remapInstructionsInBlocks(Blocks, VMap);
  Instruction *Term = CheckBB->getTerminator();
  IRBuilder<>(Term).CreateCondBr(Unsafe, Orig->getLoopPreheader(),
                                 L->getLoopPreheader());
  Term->eraseFromParent();
  BasicBlock *Exit = L->getExitBlock();
  DT.changeImmediateDominator(Exit, CheckBB);
  BasicBlock *Exiting = L->getExitingBlock();
  auto *OrigExiting = cast<BasicBlock>(VMap[Exiting]);
  for (PHINode &PN : Exit->phis()) {
    Value *V = PN.getIncomingValueForBlock(Exiting);
    Value *Mapped = VMap.lookup(V);
    PN.addIncoming(Mapped ? Mapped : V, OrigExiting);
  }
  formDedicatedExitBlocks(Orig, &DT, &LI, nullptr, true);
  formDedicatedExitBlocks(L, &DT, &LI, nullptr, true);
}

bool AGXWordCounterVersioning::runOnFunction(Function &F) {
  if (!EnableWordCounterVersioning || skipFunction(F) ||
      F.getCallingConv() != CallingConv::AGX_KERNEL)
    return false;
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  const DataLayout &DL = F.getDataLayout();
  SmallVector<Loop *, 8> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (L->isInnermost())
      Loops.push_back(L);
  bool Changed = false;
  Type *I64 = Type::getInt64Ty(F.getContext());
  Type *I32 = Type::getInt32Ty(F.getContext());
  for (Loop *L : Loops) {
    BasicBlock *Latch = L->getLoopLatch();
    BasicBlock *PHB = L->getLoopPreheader();
    LLVM_DEBUG(dbgs() << "wcv: loop " << L->getHeader()->getName() << "\n");
    if (!Latch || !PHB || !L->getExitingBlock() || !L->getUniqueExitBlock())
      continue;
    // 64-bit counters by one loop-invariant step whose uses are word
    // accesses' addresses: the step, and for each its accesses
    MapVector<Value *, SmallVector<std::pair<PHINode *, Instruction *>, 4>> Steps;
    SmallVector<Access, 8> Accesses;
    for (PHINode &PN : L->getHeader()->phis()) {
      if (!PN.getType()->isIntegerTy(64))
        continue;
      auto *N = dyn_cast<Instruction>(PN.getIncomingValueForBlock(Latch));
      Value *D;
      // (a step of a runtime stride, a row's length or a grid's: a constant
      // step's loops are a row's short runs, which the check would cost
      // more than the adds, spmv_csr_vector +31%)
      if (!N || !match(N, m_c_Add(m_Specific(&PN), m_Value(D))) ||
          isa<Constant>(D) ||
          !L->isLoopInvariant(D) || !N->hasOneUse() || N->user_back() != &PN) {
        LLVM_DEBUG(dbgs() << "wcv: not a stepped counter " << PN << "\n");
        continue;
      }
      SmallVector<Access, 8> Acc;
      bool OK = true;
      for (User *U : PN.users()) {
        if (U == N)
          continue;
        auto *UI = dyn_cast<Instruction>(U);
        Value *Base;
        if (UI && isa<IntToPtrInst>(UI) && L->contains(UI)) {
          // (the counter is the address)
          continue;
        }
        if (!UI || !L->contains(UI) ||
            !match(UI, m_c_Add(m_Specific(&PN), m_Value(Base))) ||
            !L->isLoopInvariant(Base) ||
            !collectAccesses(UI, &PN, Base, L, DL, Acc)) {
          OK = false;
          break;
        }
      }
      if (OK && !collectAccesses(&PN, &PN, nullptr, L, DL, Acc))
        OK = false;
      LLVM_DEBUG(dbgs() << "wcv: " << PN << " ok " << OK << " accesses "
                        << Acc.size() << "\n");
      if (!OK || Acc.empty())
        continue;
      Steps[D].push_back({&PN, N});
      Accesses.append(Acc.begin(), Acc.end());
    }
    if (Steps.empty())
      continue;
    // (one step: the counters that share it share one counter of words)
    Value *Step = Steps.begin()->first;
    auto Counters = Steps.begin()->second;
    llvm::erase_if(Accesses, [&](const Access &A) {
      return none_of(Counters, [&](auto &C) { return C.first == A.P; });
    });
    // (an index-register access takes no constant offset: each base and
    // offset of a counter's accesses is a 64-bit base register in the
    // copy, which more than two cost in registers what the adds saved,
    // conv3x3's nine: 51 -> 68 GPRs)
    {
      // (constant offsets only: an access at a counter plus a loop-
      // invariant base took a 64-bit add for it before, one at a constant
      // offset an immediate, cuda_index_add's four bases -5.7%)
      DenseMap<std::pair<PHINode *, Value *>, SmallSet<int64_t, 4>> Classes;
      for (const Access &A : Accesses)
        Classes[{A.P, A.X}].insert(A.C);
      // (and at most four base registers in all: conv3x3's nine runtime
      // bases took 51 -> 68 GPRs, +0.4%)
      unsigned Total = 0;
      for (auto &E : Classes)
        Total += E.second.size();
      if (Total > WordCounterMaxTotalBases ||
          any_of(Classes, [](auto &E) {
            return E.second.size() > WordCounterMaxBases;
          })) {
        LLVM_DEBUG(dbgs() << "wcv: too many bases\n");
        continue;
      }
    }
    const SCEV *BTC = SE.getSymbolicMaxBackedgeTakenCount(L);
    // (a count with a runtime division: 64 bits have no division to call,
    // a grid-stride loop's count, L15-E679)
    if (!isa<SCEVCouldNotCompute>(BTC) &&
        SCEVExprContains(BTC, [](const SCEV *X) {
          auto *D = dyn_cast<SCEVUDivExpr>(X);
          return D && !isa<SCEVConstant>(D->getRHS());
        }))
      BTC = SE.getCouldNotCompute();
    // Without a count: a latch that continues while an i32 counter's next
    // value c + s is below an invariant n (slt or ult), s and n invariant.
    PHINode *Ctr = nullptr;
    Value *CS = nullptr, *CN = nullptr;
    bool CSigned = false;
    if (isa<SCEVCouldNotCompute>(BTC)) {
      auto *Br = dyn_cast<CondBrInst>(Latch->getTerminator());
      ICmpInst *Cmp = Br ? dyn_cast<ICmpInst>(Br->getCondition()) : nullptr;
      Value *CNext;
      if (Cmp && Br->getSuccessor(0) == L->getHeader() &&
          (Cmp->getPredicate() == ICmpInst::ICMP_SLT ||
           Cmp->getPredicate() == ICmpInst::ICMP_ULT) &&
          L->isLoopInvariant(Cmp->getOperand(1))) {
        CNext = Cmp->getOperand(0);
        CN = Cmp->getOperand(1);
        CSigned = Cmp->getPredicate() == ICmpInst::ICMP_SLT;
        for (PHINode &PN : L->getHeader()->phis())
          if (PN.getType()->isIntegerTy(32) &&
              PN.getIncomingValueForBlock(Latch) == CNext &&
              match(CNext, m_c_Add(m_Specific(&PN), m_Value(CS))) &&
              L->isLoopInvariant(CS))
            Ctr = &PN;
      }
      if (!Ctr) {
        LLVM_DEBUG(dbgs() << "wcv: no count\n");
        continue;
      }
    }
    bool Convergent = any_of(L->blocks(), [](BasicBlock *BB) {
      return any_of(*BB, [](Instruction &I) {
        auto *CB = dyn_cast<CallBase>(&I);
        return CB && CB->isConvergent();
      });
    });
    if (!L->isLoopSimplifyForm()) {
      Changed |= simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false);
      if (!L->isLoopSimplifyForm() || !L->getExitBlock())
        continue;
    }
    Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
    PHB = L->getLoopPreheader();
    // the counter of words from 0 fits: the step a multiple of 4, its
    // quarter and the count below 2^31, their product (within 64 bits)
    // below 2^31 in magnitude
    IRBuilder<> B(PHB->getTerminator());
    SCEVExpander Exp(SE, "words");
    auto Fits = [&](Value *V) {
      return B.CreateICmpULT(B.CreateAdd(V, ConstantInt::get(I64, uint64_t(1) << 31)),
                             ConstantInt::get(I64, uint64_t(1) << 32));
    };
    auto K = [&](uint64_t V) { return ConstantInt::get(I64, V); };
    Value *Step4 = B.CreateAShr(Step, 2);
    Value *Safe = B.CreateICmpEQ(B.CreateAnd(Step, 3), K(0));
    Safe = B.CreateAnd(Safe, Fits(Step4));
    if (!Ctr) {
      Value *Count = Exp.expandCodeFor(BTC, nullptr, PHB->getTerminator());
      Count = B.CreateZExtOrTrunc(Count, I64);
      Safe = B.CreateAnd(Safe, B.CreateICmpULT(Count, K(uint64_t(1) << 31)));
      Safe = B.CreateAnd(Safe, Fits(B.CreateMul(Count, Step4)));
    } else {
      // k backedges taken: c0 + k s < n, k s <= n - c0 - 1, so k |step / 4|
      // < 2^31 where (n - c0) |step / 4| < 2^31 s (s > 0); and the counter
      // does not wrap where n + s - 1 is in its range
      auto Ext = [&](Value *V) {
        return CSigned ? B.CreateSExt(V, I64) : B.CreateZExt(V, I64);
      };
      Value *S64 = Ext(CS), *N64 = Ext(CN);
      Value *C0 = Ext(Ctr->getIncomingValueForBlock(PHB));
      Safe = B.CreateAnd(Safe, B.CreateICmpSGT(S64, K(0)));
      Safe = B.CreateAnd(Safe, B.CreateICmpSLE(B.CreateSub(B.CreateAdd(N64, S64), K(1)),
                                               K(CSigned ? 0x7fffffff : 0xffffffff)));
      Value *Span = B.CreateBinaryIntrinsic(Intrinsic::smax, B.CreateSub(N64, C0), K(0));
      Value *A4 = B.CreateBinaryIntrinsic(Intrinsic::abs, Step4, B.getFalse());
      Safe = B.CreateAnd(Safe, B.CreateICmpULT(Span, K(uint64_t(1) << 32)));
      Safe = B.CreateAnd(Safe, B.CreateICmpULT(B.CreateMul(Span, A4),
                                               B.CreateShl(S64, 31)));
    }
    if (Convergent) {
      // (the whole SIMD-group takes one copy)
      Value *Ballot = B.CreateIntrinsic(Intrinsic::agx_simd_ballot, {}, {Safe});
      Safe = B.CreateICmpEQ(
          Ballot, B.CreateIntrinsic(Intrinsic::agx_simd_ballot, {}, {B.getTrue()}));
    }
    versionOn(L, B.CreateNot(Safe), LI, DT);
    // the copy for counters that fit: one 32-bit counter of words from 0,
    // each access at its counter's start (plus its base and constant) and
    // that many words
    BasicBlock *PH = L->getLoopPreheader();
    B.SetInsertPoint(PH->getTerminator());
    Value *D4 = B.CreateTrunc(Step4, I32);
    MapVector<std::tuple<PHINode *, Value *, int64_t>, Value *> Bases;
    for (const Access &A : Accesses) {
      auto &P = Bases[{A.P, A.X, A.C}];
      if (!P) {
        Value *V = A.P->getIncomingValueForBlock(PH);
        if (A.X)
          V = B.CreateAdd(V, A.X);
        if (A.C)
          V = B.CreateAdd(V, ConstantInt::get(I64, A.C));
        P = B.CreateIntToPtr(V, PointerType::get(F.getContext(), 1));
      }
    }
    PHINode *Wd = PHINode::Create(I32, 2, "words", L->getHeader()->begin());
    IRBuilder<> HB(&*L->getHeader()->getFirstInsertionPt());
    Value *WS = HB.CreateSExt(Wd, I64);
    IRBuilder<> LB(Latch->getTerminator());
    Value *WNext = LB.CreateAdd(Wd, D4, "words.next", false, true);
    Wd->addIncoming(ConstantInt::get(I32, 0), PH);
    Wd->addIncoming(WNext, Latch);
    SmallVector<WeakTrackingVH, 8> Dead;
    for (const Access &A : Accesses) {
      IRBuilder<> AB(A.I);
      Value *Ptr = AB.CreateGEP(I32, Bases[{A.P, A.X, A.C}], WS);
      Value *OldP = getLoadStorePointerOperand(A.I);
      A.I->replaceUsesOfWith(OldP, Ptr);
      Dead.push_back(OldP);
    }
    for (WeakTrackingVH &V : Dead)
      if (auto *I = dyn_cast_or_null<Instruction>(V))
        RecursivelyDeleteTriviallyDeadInstructions(I);
    for (auto &C : Counters)
      RecursivelyDeleteDeadPHINode(C.first);
    LLVM_DEBUG(dbgs() << "agx-word-counter-versioning: " << F.getName() << " "
                      << L->getHeader()->getName() << " " << Counters.size()
                      << " counters, " << Accesses.size() << " accesses\n");
    Changed = true;
  }
  if (Changed)
    SE.forgetAllLoops();
  return Changed;
}
