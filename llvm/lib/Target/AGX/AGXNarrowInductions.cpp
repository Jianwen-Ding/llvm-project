//===-- AGXNarrowInductions.cpp - 64-bit induction variables in 32 bits ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PTX computes addresses in 64 bits, so its loops count byte offsets in 64-bit
// induction variables even where the offsets are small (an n-body tile of 256
// float4: 0 to 4096). AGX's ALU is 32 bits wide: a 64-bit induction variable
// takes a 64-bit add per iteration, a register pair, and an exit test of both
// halves. A 64-bit header phi incremented by a loop-invariant amount, whose
// value and increment scalar evolution proves within [0, 2^31), becomes a
// 32-bit phi; its 64-bit uses take the zero extension, which instruction
// selection folds into 64-bit address arithmetic and into compares with
// constants.
//
// A 64-bit counter that steps by one to a loop-invariant exit value no range
// bounds (an MLIR index loop to a dynamic bound, which loop strength reduction
// makes a count down to zero) is counted in 32 bits instead: its low half in
// the loop, which exits when that reaches the exit value's low half, and the
// remaining multiples of 2^32 in a new enclosing loop, which runs the loop
// again until none are left. The enclosing loop runs once for any count up to
// 2^32.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-narrow-inductions"

static cl::opt<bool> AGXNonNegativeCasts(
    "agx-non-negative-casts", cl::init(true), cl::Hidden,
    cl::desc("Signed extensions, divisions and arithmetic shifts of values "
             "SCEV proves non-negative as the unsigned ones"));

static cl::opt<bool> AGXNarrowInductionsOpt(
    "agx-narrow-ivs", cl::init(true), cl::Hidden,
    cl::desc("Count 64-bit induction variables that stay below 2^31 in 32 "
             "bits"));

static cl::opt<bool> AGXSplitWideCounters(
    "agx-split-wide-counters", cl::init(true), cl::Hidden,
    cl::desc("Count a 64-bit loop counter that steps by one to an exit value "
             "in 32 bits, the high half in an enclosing loop"));

namespace {

class AGXNarrowInductions : public FunctionPass {
public:
  static char ID;
  AGXNarrowInductions() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Narrow Inductions"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }
};

} // namespace

char AGXNarrowInductions::ID = 0;

INITIALIZE_PASS_BEGIN(AGXNarrowInductions, DEBUG_TYPE, "AGX Narrow Inductions",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AGXNarrowInductions, DEBUG_TYPE, "AGX Narrow Inductions",
                    false, false)

FunctionPass *llvm::createAGXNarrowInductionsPass() {
  return new AGXNarrowInductions();
}

static cl::opt<bool> AGXNarrowBoundCompares(
    "agx-narrow-bound-compares", cl::init(true), cl::Hidden,
    cl::desc("A narrowed induction's compare with a 64-bit bound known to "
             "fit 32 bits in 32 bits"));

static cl::opt<bool> AGXSignedNarrowing(
    "agx-signed-narrowing", cl::init(true), cl::Hidden,
    cl::desc("Narrow 64-bit inductions whose signed range fits 32 bits, "
             "with sign-extended uses"));

static bool fitsSigned32(ScalarEvolution &SE, Value *V) {
  return SE.getUnsignedRange(SE.getSCEV(V)).getUnsignedMax().ult(1u << 31);
}

static bool narrow(PHINode &Phi, Loop &L, ScalarEvolution &SE) {
  BasicBlock *Preheader = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Preheader || !Latch || Phi.getNumIncomingValues() != 2)
    return false;
  Value *Start = Phi.getIncomingValueForBlock(Preheader);
  auto *Inc = dyn_cast<BinaryOperator>(Phi.getIncomingValueForBlock(Latch));
  Value *Step;
  if (!Inc || !match(Inc, m_c_Add(m_Specific(&Phi), m_Value(Step))) ||
      !L.isLoopInvariant(Step))
    return false;
  // [0, 2^31): zero-extended uses. Otherwise a signed range within 32 bits
  // (LSR's counters from a negative start, {-4, +, 4} to 508): sign-extended
  // uses, the narrow add without unsigned wrap flags.
  bool Unsigned = fitsSigned32(SE, &Phi) && fitsSigned32(SE, Inc) &&
                  fitsSigned32(SE, Step);
  auto SignedFits = [&](Value *V) {
    ConstantRange R = SE.getSignedRange(SE.getSCEV(V));
    return R.getSignedMin().sge(APInt::getSignedMinValue(32).sext(64)) &&
           R.getSignedMax().sle(APInt::getSignedMaxValue(32).sext(64));
  };
  if (!Unsigned &&
      (!AGXSignedNarrowing || !SignedFits(&Phi) || !SignedFits(Inc) ||
       !SignedFits(Step)))
    return false;

  SE.forgetLoop(&L);
  Type *I32 = Type::getInt32Ty(Phi.getContext());
  IRBuilder<> B(Preheader->getTerminator());
  Value *NarrowStart = B.CreateTrunc(Start, I32, Start->getName() + ".lo");
  Value *NarrowStep = B.CreateTrunc(Step, I32, Step->getName() + ".lo");
  PHINode *NarrowPhi =
      PHINode::Create(I32, 2, Phi.getName() + ".lo", Phi.getIterator());
  B.SetInsertPoint(Inc);
  Value *NarrowInc =
      B.CreateAdd(NarrowPhi, NarrowStep, Inc->getName() + ".lo",
                  /*HasNUW=*/Unsigned, /*HasNSW=*/true);
  NarrowPhi->addIncoming(NarrowStart, Preheader);
  NarrowPhi->addIncoming(NarrowInc, Latch);

  B.SetInsertPoint(Phi.getParent(), Phi.getParent()->getFirstInsertionPt());
  Value *WidePhi =
      Unsigned ? B.CreateZExt(NarrowPhi, Phi.getType(), Phi.getName())
               : B.CreateSExt(NarrowPhi, Phi.getType(), Phi.getName());
  B.SetInsertPoint(Inc);
  Value *WideInc =
      Unsigned ? B.CreateZExt(NarrowInc, Inc->getType(), Inc->getName())
               : B.CreateSExt(NarrowInc, Inc->getType(), Inc->getName());
  // The zero extension of the increment replaces it after it is computed.
  cast<Instruction>(WideInc)->moveAfter(cast<Instruction>(NarrowInc));
  Inc->replaceAllUsesWith(WideInc);
  Phi.replaceAllUsesWith(WidePhi);
  Inc->eraseFromParent();
  Phi.eraseFromParent();
  // A compare of the zero-extended value with a 64-bit value V whose high
  // word is known zero (a trip bound umin(n - i, 65536) computed before the
  // loop, which instruction selection sees only as a register) is of the
  // 32-bit values: both are in [0, 2^32), where the signed and unsigned
  // orders agree.
  if (Unsigned && AGXNarrowBoundCompares) {
    const DataLayout &DL = Preheader->getModule()->getDataLayout();
    for (auto [Wide, Narrow] : {std::pair<Value *, Value *>(WidePhi, NarrowPhi),
                                {WideInc, NarrowInc}})
      for (User *U : make_early_inc_range(Wide->users())) {
        auto *C = dyn_cast<ICmpInst>(U);
        if (!C || C->getOperand(0) != Wide || isa<Constant>(C->getOperand(1)) ||
            !L.isLoopInvariant(C->getOperand(1)) ||
            computeKnownBits(C->getOperand(1), DL).countMinLeadingZeros() < 32)
          continue;
        IRBuilder<> CB(Preheader->getTerminator());
        Value *V = CB.CreateTrunc(C->getOperand(1), I32,
                                  C->getOperand(1)->getName() + ".lo");
        C->setPredicate(C->getUnsignedPredicate());
        C->setOperand(0, Narrow);
        C->setOperand(1, V);
      }
  }
  // A compare of a sign-extended value with a constant C is of the 32-bit
  // value with C where C is in [0, 2^31) (a negative value is above C
  // unsigned either way) or the predicate is signed and C in 32 bits.
  if (!Unsigned)
    for (auto [Wide, Narrow] : {std::pair<Value *, Value *>(WidePhi, NarrowPhi),
                                {WideInc, NarrowInc}})
      for (User *U : make_early_inc_range(Wide->users())) {
        auto *C = dyn_cast<ICmpInst>(U);
        auto *K = C ? dyn_cast<ConstantInt>(C->getOperand(1)) : nullptr;
        if (!K || C->getOperand(0) != Wide)
          continue;
        const APInt &V = K->getValue();
        bool Fits = C->isSigned() ? V.isSignedIntN(32)
                                  : V.isNonNegative() && V.ult(1u << 31);
        if (!Fits)
          continue;
        C->setOperand(0, Narrow);
        C->setOperand(1, ConstantInt::get(I32, V.trunc(32)));
      }
  return true;
}

namespace {
/// A 64-bit header phi that steps by Step (1 or -1) and whose increment ends
/// the loop at the latch where it equals End.
struct WideCounter {
  PHINode *Phi;
  BinaryOperator *Inc;
  ICmpInst *Cmp;
  CondBrInst *Br;
  Value *Start, *End;
  int64_t Step;
  BasicBlock *Preheader, *Header, *Latch, *Exit;
  // Uses of the increment past the exit block's phis.
  SmallVector<Use *, 4> OutsideUses;
  // The loop steps at most 2^32 times (scalar evolution's bound on its
  // trip count): its low half alone counts them.
  bool Bounded = false;
};
} // namespace

static std::optional<WideCounter> matchWideCounter(PHINode &Phi, Loop &L,
                                                   ScalarEvolution &SE) {
  BasicBlock *Preheader = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  BasicBlock *Header = L.getHeader();
  if (!Preheader || !Latch || Phi.getNumIncomingValues() != 2 ||
      pred_size(Header) != 2)
    return std::nullopt;
  auto *Inc = dyn_cast<BinaryOperator>(Phi.getIncomingValueForBlock(Latch));
  const APInt *C;
  if (!Inc || !match(Inc, m_Add(m_Specific(&Phi), m_APInt(C))) ||
      !(C->isOne() || C->isAllOnes()))
    return std::nullopt;
  auto *Br = dyn_cast<CondBrInst>(Latch->getTerminator());
  if (!Br)
    return std::nullopt;
  auto *Cmp = dyn_cast<ICmpInst>(Br->getCondition());
  if (!Cmp || !Cmp->isEquality() || !Cmp->hasOneUse() ||
      Cmp->getParent() != Latch)
    return std::nullopt;
  Value *End = Cmp->getOperand(0) == Inc   ? Cmp->getOperand(1)
               : Cmp->getOperand(1) == Inc ? Cmp->getOperand(0)
                                           : nullptr;
  if (!End || !L.isLoopInvariant(End))
    return std::nullopt;
  bool ExitOnEqual = Cmp->getPredicate() == ICmpInst::ICMP_EQ;
  BasicBlock *Exit = Br->getSuccessor(ExitOnEqual ? 0 : 1);
  if (Br->getSuccessor(ExitOnEqual ? 1 : 0) != Header || L.contains(Exit))
    return std::nullopt;

  // The counter's other uses: truncations to 32 bits or fewer in the loop,
  // and the exit block's phis of the value it leaves the latch with.
  // Outside the loop, the increment is End wherever it is read: the latch,
  // which defines it, leaves only to Exit.
  SmallVector<Use *, 4> OutsideUses;
  auto OtherUsesOk = [&](Instruction *I, Instruction *A, Instruction *B) {
    for (Use &U : I->uses()) {
      auto *UI = cast<Instruction>(U.getUser());
      if (UI == A || UI == B)
        continue;
      if (isa<TruncInst>(UI) && L.contains(UI) &&
          UI->getType()->getIntegerBitWidth() <= 32)
        continue;
      auto *P = dyn_cast<PHINode>(UI);
      if (P && P->getParent() == Exit && P->getIncomingBlock(U) == Latch)
        continue;
      BasicBlock *At = P ? P->getIncomingBlock(U) : UI->getParent();
      if (I != Inc || L.contains(At))
        return false;
      OutsideUses.push_back(&U);
    }
    return true;
  };
  if (!OtherUsesOk(&Phi, Inc, nullptr) || !OtherUsesOk(Inc, &Phi, Cmp))
    return std::nullopt;
  // The latch the only exit, taken after at most 2^32 steps: the counter
  // reaches End within them.
  auto *MaxBTC =
      dyn_cast<SCEVConstant>(SE.getConstantMaxBackedgeTakenCount(&L));
  bool Bounded = L.getExitingBlock() == Latch && MaxBTC &&
                 MaxBTC->getAPInt().getActiveBits() <= 32;
  return WideCounter{&Phi,
                     Inc,
                     Cmp,
                     Br,
                     Phi.getIncomingValueForBlock(Preheader),
                     End,
                     C->isOne() ? 1 : -1,
                     Preheader,
                     Header,
                     Latch,
                     Exit,
                     OutsideUses,
                     Bounded};
}

/// The loop runs until the counter has stepped D = (End - Start) * Step times
/// (2^64 for 0). Its low half reaches End's after 1 + (D - 1) mod 2^32 steps
/// and every 2^32 after that: the loop exits there, into an enclosing loop
/// that runs it again (D - 1) >> 32 times, the counter's low half carried on.
static void splitWideCounter(const WideCounter &W) {
  LLVMContext &Ctx = W.Phi->getContext();
  Function &F = *W.Header->getParent();
  Type *I32 = Type::getInt32Ty(Ctx), *I64 = W.Phi->getType();
  IRBuilder<> B(W.Preheader->getTerminator());
  Value *EndLo = B.CreateTrunc(W.End, I32, W.End->getName() + ".lo");

  // An i32 header phi that already counts the low half (loop strength
  // reduction's narrow copy of the counter) is the low half.
  PHINode *Lo = nullptr;
  Value *LoNext = nullptr;
  for (PHINode &P : W.Header->phis()) {
    const APInt *C;
    Value *Next = P.getIncomingValueForBlock(W.Latch);
    if (P.getType() == I32 &&
        match(P.getIncomingValueForBlock(W.Preheader),
              m_Trunc(m_Specific(W.Start))) &&
        match(Next, m_Add(m_Specific(&P), m_APInt(C))) &&
        C->getSExtValue() == W.Step) {
      Lo = &P;
      LoNext = Next;
      break;
    }
  }

  // Without an enclosing loop the preheader stands for its header and the
  // latch for its latch.
  BasicBlock *OH = W.Preheader, *OL = W.Latch;
  if (W.Bounded) {
    if (!Lo) {
      Lo = PHINode::Create(I32, 2, W.Phi->getName() + ".lo");
      Lo->insertInto(W.Header, W.Header->begin());
      B.SetInsertPoint(W.Inc);
      LoNext = B.CreateAdd(Lo, ConstantInt::getSigned(I32, W.Step),
                           W.Inc->getName() + ".lo");
      B.SetInsertPoint(W.Preheader->getTerminator());
      Lo->addIncoming(B.CreateTrunc(W.Start, I32, W.Start->getName() + ".lo"),
                      W.Preheader);
      Lo->addIncoming(LoNext, W.Latch);
    }
  } else {
    Value *D =
        W.Step == 1 ? B.CreateSub(W.End, W.Start) : B.CreateSub(W.Start, W.End);
    Value *Hi = B.CreateTrunc(
        B.CreateLShr(B.CreateAdd(D, ConstantInt::getAllOnesValue(I64)), 32),
        I32, W.Phi->getName() + ".hi");
    OH = BasicBlock::Create(Ctx, W.Header->getName() + ".outer", &F, W.Header);
    OL = BasicBlock::Create(Ctx, W.Latch->getName() + ".outer", &F,
                            W.Latch->getNextNode());
    W.Preheader->getTerminator()->replaceUsesOfWith(W.Header, OH);
    UncondBrInst::Create(W.Header, OH);
    // Every header phi enters from the enclosing loop's header, which takes the
    // value of the preheader, then the one the latch leaves with.
    SmallVector<PHINode *, 8> Phis;
    for (PHINode &P : W.Header->phis())
      if (&P != W.Phi)
        Phis.push_back(&P);
    for (PHINode *P : Phis) {
      PHINode *Q = PHINode::Create(P->getType(), 2, P->getName() + ".outer");
      Q->insertInto(OH, OH->getFirstNonPHIIt());
      Q->addIncoming(P->getIncomingValueForBlock(W.Preheader), W.Preheader);
      Q->addIncoming(P->getIncomingValueForBlock(W.Latch), OL);
      int K = P->getBasicBlockIndex(W.Preheader);
      P->setIncomingBlock(K, OH);
      P->setIncomingValue(K, Q);
    }

    PHINode *HiLeft = PHINode::Create(I32, 2, W.Phi->getName() + ".hi.left");
    HiLeft->insertInto(OH, OH->getFirstNonPHIIt());
    if (!Lo) {
      PHINode *LoOuter =
          PHINode::Create(I32, 2, W.Phi->getName() + ".lo.outer");
      LoOuter->insertInto(OH, OH->getFirstNonPHIIt());
      Lo = PHINode::Create(I32, 2, W.Phi->getName() + ".lo");
      Lo->insertInto(W.Header, W.Header->begin());
      B.SetInsertPoint(W.Inc);
      LoNext = B.CreateAdd(Lo, ConstantInt::getSigned(I32, W.Step),
                           W.Inc->getName() + ".lo");
      Lo->addIncoming(LoOuter, OH);
      Lo->addIncoming(LoNext, W.Latch);
      B.SetInsertPoint(W.Preheader->getTerminator());
      LoOuter->addIncoming(
          B.CreateTrunc(W.Start, I32, W.Start->getName() + ".lo"), W.Preheader);
      LoOuter->addIncoming(LoNext, OL);
    }

    IRBuilder<> BL(OL);
    Value *Done = BL.CreateICmpEQ(HiLeft, ConstantInt::get(I32, 0));
    Value *HiNext = BL.CreateAdd(HiLeft, ConstantInt::getAllOnesValue(I32));
    BL.CreateCondBr(Done, W.Exit, OH);
    HiLeft->addIncoming(Hi, W.Preheader);
    HiLeft->addIncoming(HiNext, OL);

    W.Br->replaceUsesOfWith(W.Exit, OL);
  }
  W.Cmp->replaceUsesOfWith(W.End, EndLo);
  W.Cmp->replaceUsesOfWith(W.Inc, LoNext);

  // The exit phis: from the enclosing loop's latch now, with the counter's
  // values when it leaves (End and the step before it).
  for (PHINode &P : W.Exit->phis()) {
    int K = P.getBasicBlockIndex(W.Latch);
    if (K < 0)
      continue;
    P.setIncomingBlock(K, OL);
    Value *V = P.getIncomingValue(K);
    if (V == W.Inc)
      P.setIncomingValue(K, W.End);
    else if (V == W.Phi) {
      B.SetInsertPoint(W.Preheader->getTerminator());
      P.setIncomingValue(
          K, B.CreateSub(W.End, ConstantInt::getSigned(I64, W.Step)));
    }
  }
  for (Use *U : W.OutsideUses)
    U->set(W.End);
  for (auto [Old, New] : {std::pair<Instruction *, Value *>(W.Phi, Lo),
                          std::pair<Instruction *, Value *>(W.Inc, LoNext)})
    for (User *U : make_early_inc_range(Old->users()))
      if (auto *T = dyn_cast<TruncInst>(U)) {
        if (T->getType() == I32) {
          T->replaceAllUsesWith(New);
        } else {
          B.SetInsertPoint(T);
          T->replaceAllUsesWith(B.CreateTrunc(New, T->getType()));
        }
        T->eraseFromParent();
      }
  W.Phi->replaceAllUsesWith(PoisonValue::get(I64));
  W.Inc->replaceAllUsesWith(PoisonValue::get(I64));
  W.Inc->eraseFromParent();
  W.Phi->eraseFromParent();
}

static cl::opt<bool> AGXSplitOffsetInductions(
    "agx-split-offset-inductions", cl::init(true), cl::Hidden,
    cl::desc("A 64-bit offset induction only added to invariant bases as "
             "one induction per base"));

/// After loop strength reduction, PTX's walk of several buffers by one 64-bit
/// byte offset o (o += s; each address base_k + o in integers) is still k
/// 64-bit additions and the increment every iteration, with o and every base
/// live. Where o has no other use and no two bases differ by a constant
/// (which one register and immediate offsets would serve), each address is
/// its own induction p_k = base_k + o, p_k += s (the same modulo 2^64): k
/// increments, one register less, and no addition before the accesses.
/// Only additions that run every iteration (conv3x3's are in the bounds
/// checks' blocks, which the SIMD-group may skip).
/// (Before loop strength reduction it redoes the shared offset, L15-E163.)
static bool splitOffsets(PHINode &O, Loop &L, ScalarEvolution &SE,
                         DominatorTree &DT) {
  using namespace PatternMatch;
  BasicBlock *Pre = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Pre || !Latch || O.getNumIncomingValues() != 2)
    return false;
  Value *Init = O.getIncomingValueForBlock(Pre);
  auto *Inc = dyn_cast<BinaryOperator>(O.getIncomingValueForBlock(Latch));
  Value *S;
  if (!Init || !Inc || !match(Inc, m_c_Add(m_Specific(&O), m_Value(S))) ||
      !L.isLoopInvariant(S) || !Inc->hasOneUse())
    return false;
  // base + o in integers, or a byte GEP of a base pointer by o
  SmallVector<std::pair<Instruction *, Value *>, 4> Adds;
  for (User *U : O.users()) {
    if (U == Inc)
      continue;
    auto *A = dyn_cast<Instruction>(U);
    Value *Base = nullptr;
    if (auto *G = dyn_cast_or_null<GetElementPtrInst>(A)) {
      if (G->getNumIndices() == 1 && G->getOperand(1) == &O &&
          G->getSourceElementType()->isIntegerTy(8))
        Base = G->getPointerOperand();
    } else if (A) {
      match(A, m_c_Add(m_Specific(&O), m_Value(Base)));
    }
    // (a base computed in the loop from invariant values, such as PTX's
    // casts of a kernel argument, goes before it)
    bool Hoisted;
    if (!Base || !L.contains(A) || !DT.dominates(A->getParent(), Latch) ||
        !L.makeLoopInvariant(Base, Hoisted, Pre->getTerminator()))
      return false;
    Adds.push_back({A, Base});
  }
  if (Adds.empty())
    return false;
  for (unsigned I = 0; I < Adds.size(); ++I)
    for (unsigned J = I + 1; J < Adds.size(); ++J)
      if (isa<SCEVConstant>(SE.getMinusSCEV(SE.getSCEV(Adds[I].second),
                                             SE.getSCEV(Adds[J].second))))
        return false;
  for (auto [A, Base] : Adds) {
    IRBuilder<> P(Pre->getTerminator());
    bool Ptr = isa<GetElementPtrInst>(A);
    Value *Start = Ptr ? P.CreatePtrAdd(Base, Init) : P.CreateAdd(Base, Init);
    PHINode *Ph = PHINode::Create(A->getType(), 2, A->getName() + ".iv");
    Ph->insertBefore(O.getIterator());
    IRBuilder<> B(Inc);
    Value *Next = Ptr ? B.CreatePtrAdd(Ph, S, A->getName() + ".iv.next")
                      : B.CreateAdd(Ph, S, A->getName() + ".iv.next");
    Ph->addIncoming(Start, Pre);
    Ph->addIncoming(Next, Latch);
    SE.forgetValue(A);
    A->replaceAllUsesWith(Ph);
    A->eraseFromParent();
  }
  SE.forgetValue(&O);
  Inc->replaceAllUsesWith(PoisonValue::get(Inc->getType()));
  Inc->eraseFromParent();
  O.eraseFromParent();
  return true;
}

static cl::opt<bool> AGXElementIndices(
    "agx-element-indices", cl::init(true), cl::Hidden,
    cl::desc("A 32-bit byte offset induction of 32-bit-element accesses as "
             "an element index"));

/// Loop strength reduction counts a row walk in bytes ({0, +, 4}: LSR sees
/// no scaled index addressing, L14-E2), and narrowing makes that offset
/// 32-bit, zero-extended for each address: base + zext(o) is a 64-bit add
/// before every access. Where every access is of elements of one size u (2
/// or 4 bytes: 16- or 32-bit, scalars or vectors) and the offset's start,
/// step and the constants it is compared with are multiples of u, it is u j
/// for an element index j (o < 2^32, so u j does not wrap): j steps by
/// s / u, the accesses take base + zext(j) * u as their index, and the
/// compares are of j with k / u.
static bool elementIndex(PHINode &P, Loop &L) {
  using namespace PatternMatch;
  BasicBlock *Pre = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Pre || !Latch || P.getNumIncomingValues() != 2 ||
      !P.getType()->isIntegerTy(32))
    return false;
  const APInt *C0, *St;
  auto *Inc = dyn_cast<BinaryOperator>(P.getIncomingValueForBlock(Latch));
  if (!match(P.getIncomingValueForBlock(Pre), m_APInt(C0)) || !Inc ||
      !match(Inc, m_Add(m_Specific(&P), m_APInt(St))) || St->isZero())
    return false;
  // the accesses' element size in bytes, the same for all
  unsigned Unit = 0;
  SmallVector<GetElementPtrInst *, 4> Geps;
  SmallVector<std::pair<ICmpInst *, bool>, 2> Cmps; // (compare, of Inc)
  SmallVector<ZExtInst *, 2> Exts;
  SmallVector<ConstantInt *, 2> Ks;
  auto Divisible = [&](Value *V) {
    auto *C = dyn_cast<ConstantInt>(V);
    if (C)
      Ks.push_back(C);
    return C != nullptr;
  };
  auto CheckUse = [&](User *U, bool OfInc, bool Extended) {
    if (auto *Z = dyn_cast<ZExtInst>(U)) {
      if (Extended || !Z->getType()->isIntegerTy(64))
        return false;
      Exts.push_back(Z);
      return true;
    }
    if (auto *C = dyn_cast<ICmpInst>(U)) {
      if (!Divisible(C->getOperand(1)) || C->getOperand(0) != U->getOperand(0) ||
          (!Extended && C->isSigned()))
        return false;
      Cmps.push_back({C, OfInc});
      return true;
    }
    auto *G = dyn_cast<GetElementPtrInst>(U);
    if (!Extended || OfInc || !G || G->getNumIndices() != 1 ||
        !G->getSourceElementType()->isIntegerTy(8) || !L.isLoopInvariant(G->getPointerOperand()))
      return false;
    for (User *A : G->users()) {
      Type *T = nullptr;
      if (auto *Ld = dyn_cast<LoadInst>(A))
        T = Ld->getType();
      else if (auto *St = dyn_cast<StoreInst>(A))
        T = St->getPointerOperand() == G ? St->getValueOperand()->getType()
                                         : nullptr;
      unsigned E = T ? T->getScalarSizeInBits() / 8 : 0;
      if (!T || (E != 2 && E != 4) || T->getScalarSizeInBits() % 8 ||
          (Unit && Unit != E))
        return false;
      Unit = E;
    }
    Geps.push_back(G);
    return true;
  };
  for (User *U : P.users())
    if (U != Inc && !CheckUse(U, false, false))
      return false;
  for (User *U : Inc->users())
    if (U != &P && !CheckUse(U, true, false))
      return false;
  for (unsigned I = 0; I < Exts.size(); ++I) {
    ZExtInst *Z = Exts[I];
    bool OfInc = Z->getOperand(0) == Inc;
    for (User *U : Z->users())
      if (!CheckUse(U, OfInc, true))
        return false;
  }
  if (Geps.empty() || !Unit || C0->urem(Unit) || St->urem(Unit) ||
      any_of(Ks, [&](ConstantInt *K) { return K->getValue().urem(Unit); }))
    return false;
  // the element index and its step
  IRBuilder<> B(&P);
  PHINode *J = B.CreatePHI(P.getType(), 2, P.getName() + ".elt");
  IRBuilder<> BI(Inc);
  Value *JN = BI.CreateAdd(J, ConstantInt::get(P.getType(), St->udiv(Unit)),
                           J->getName() + ".next", /*HasNUW=*/true);
  J->addIncoming(ConstantInt::get(P.getType(), C0->udiv(Unit)), Pre);
  J->addIncoming(JN, Latch);
  for (GetElementPtrInst *G : Geps) {
    IRBuilder<> BG(G);
    Value *Z = BG.CreateZExt(J, G->getOperand(1)->getType());
    Value *N = BG.CreateGEP(BG.getIntNTy(8 * Unit), G->getPointerOperand(), Z,
                            G->getName());
    G->replaceAllUsesWith(N);
    G->eraseFromParent();
  }
  for (auto [C, OfInc] : Cmps) {
    Value *V = OfInc ? JN : static_cast<Value *>(J);
    IRBuilder<> BC(C);
    if (!C->getOperand(0)->getType()->isIntegerTy(32))
      V = BC.CreateZExt(V, C->getOperand(0)->getType());
    auto *K = cast<ConstantInt>(C->getOperand(1));
    C->setOperand(0, V);
    C->setOperand(1, ConstantInt::get(K->getType(), K->getValue().udiv(Unit)));
  }
  for (ZExtInst *Z : Exts)
    if (Z->use_empty())
      Z->eraseFromParent();
  if (Inc->hasOneUse() && P.hasOneUse()) {
    // only each other
    Inc->replaceAllUsesWith(PoisonValue::get(Inc->getType()));
    Inc->eraseFromParent();
    P.eraseFromParent();
  }
  return true;
}

static cl::opt<bool> AGXPointersFromCounters(
    "agx-pointers-from-counters", cl::init(true), cl::Hidden,
    cl::desc("A pointer induction of 32-bit-element accesses stepping a "
             "power-of-two multiple of an integer counter's step as an index "
             "of that counter"));

/// LSR walks an array with a pointer induction p (p += s bytes, a 64-bit add
/// each iteration) next to a 32-bit counter j (j += t) that another access
/// already takes as its index. Where s = t * 4 * 2^k (k at most 2: the
/// index's shift field) for 32-bit elements, and scalar evolution keeps j in
/// [0, 2^31), p = p0 + (s / t) (j - j0): the accesses take the base
/// p0 - (s / t) j0 and zext(j) << (2 + k) as their index, and p and its add
/// go (one 64-bit add and a register pair less).
static cl::opt<bool> AGXSignedCounterIndices(
    "agx-signed-counter-indices", cl::init(true), cl::Hidden,
    cl::desc("pointerFromCounter also for a counter from below zero, as a "
             "signed index"));

static cl::opt<bool> AGXIntegerAddressCounters(
    "agx-integer-address-counters", cl::init(false), cl::Hidden,
    cl::desc("pointerFromCounter also for i64 addresses (inttoptr users)"));
static cl::opt<bool> AGXRelativeCounters(
    "agx-relative-counters", cl::init(true), cl::Hidden,
    cl::desc("pointerFromCounter with the counter less its start, where the "
             "trip count bounds it"));
static cl::opt<bool> AGXRuntimeCounterStarts(
    "agx-runtime-counter-starts", cl::init(true), cl::Hidden,
    cl::desc("pointerFromCounter also for a counter whose start is not a "
             "constant"));

static bool pointerFromCounter(PHINode &P, Loop &L, ScalarEvolution &SE) {
  using namespace PatternMatch;
  BasicBlock *Pre = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Pre || !Latch || P.getNumIncomingValues() != 2)
    return false;
  // A pointer stepped by a GEP of bytes, or (the PTX front end's addresses)
  // an i64 stepped by an add whose other users are inttoptrs.
  bool IsInt = P.getType()->isIntegerTy(64);
  if (!P.getType()->isPointerTy() && !(IsInt && AGXIntegerAddressCounters))
    return false;
  auto *Inc = dyn_cast<Instruction>(P.getIncomingValueForBlock(Latch));
  const APInt *Sp;
  if (!Inc || !Inc->hasOneUse())
    return false;
  if (IsInt) {
    if (!match(Inc, m_Add(m_Specific(&P), m_APInt(Sp))))
      return false;
  } else {
    auto *G = dyn_cast<GetElementPtrInst>(Inc);
    if (!G || G->getPointerOperand() != &P || G->getNumIndices() != 1 ||
        !G->getSourceElementType()->isIntegerTy(8) ||
        !match(G->getOperand(1), m_APInt(Sp)))
      return false;
  }
  if (Sp->isNegative() || Sp->isZero())
    return false;
  // the accesses: P itself (through an inttoptr for an i64), of 32-bit
  // elements
  auto Access = [&](User *U, Value *Ptr) {
    Type *T = nullptr;
    if (auto *Ld = dyn_cast<LoadInst>(U))
      T = Ld->getType();
    else if (auto *St = dyn_cast<StoreInst>(U))
      T = St->getPointerOperand() == Ptr ? St->getValueOperand()->getType()
                                         : nullptr;
    return T && T->getScalarSizeInBits() == 32 &&
           L.contains(cast<Instruction>(U));
  };
  // For an i64: each pointer an access takes, inttoptr(P) or a constant
  // GEP of bytes from it, with its offset (folded into a base of its own).
  SmallVector<std::pair<Instruction *, int64_t>, 4> Ptrs;
  for (User *U : P.users()) {
    if (U == Inc)
      continue;
    if (!IsInt) {
      if (!Access(U, &P))
        return false;
      continue;
    }
    auto *I2P = dyn_cast<IntToPtrInst>(U);
    if (!I2P || !L.contains(I2P) || I2P->use_empty())
      return false;
    for (User *V : I2P->users()) {
      if (Access(V, I2P))
        continue;
      auto *G = dyn_cast<GetElementPtrInst>(V);
      const APInt *C;
      if (!G || G->getPointerOperand() != I2P || G->getNumIndices() != 1 ||
          !G->getSourceElementType()->isIntegerTy(8) ||
          !match(G->getOperand(1), m_APInt(C)) || G->use_empty() ||
          !all_of(G->users(), [&](User *W) { return Access(W, G); }))
        return false;
      Ptrs.push_back({G, C->getSExtValue()});
    }
    Ptrs.push_back({I2P, 0});
  }
  for (PHINode &J : L.getHeader()->phis()) {
    if (!J.getType()->isIntegerTy(32) || J.getNumIncomingValues() != 2)
      continue;
    const APInt *T;
    Value *J0 = J.getIncomingValueForBlock(Pre);
    auto *JInc = dyn_cast<BinaryOperator>(J.getIncomingValueForBlock(Latch));
    if (!JInc || !match(JInc, m_Add(m_Specific(&J), m_APInt(T))) ||
        T->isZero())
      continue;
    // The counter relative to its start, k |t| after k iterations, where the
    // trip count bounds it below 2^31: an index for any counter (one up from
    // a runtime start, one down), at one subtraction each iteration.
    uint64_t AbsT = T->abs().getZExtValue();
    auto *MaxBTC =
        dyn_cast<SCEVConstant>(SE.getConstantMaxBackedgeTakenCount(&L));
    bool Bounded = AGXRelativeCounters && MaxBTC &&
                   MaxBTC->getAPInt().getActiveBits() <= 31 &&
                   (MaxBTC->getAPInt().getZExtValue() + 1) * AbsT <
                       (uint64_t(1) << 31);
    bool Relative =
        Bounded && (T->isNegative() ||
                    (!isa<ConstantInt>(J0) && !SE.getUnsignedRange(SE.getSCEV(&J))
                                                   .getUnsignedMax()
                                                   .ult(1u << 31)));
    if ((T->isNegative() && !Relative) || Sp->getZExtValue() % AbsT)
      continue;
    // A start that is not a constant: computed before the loop (AGX:
    // -agx-runtime-counter-starts).
    if (!isa<ConstantInt>(J0) &&
        (!AGXRuntimeCounterStarts || !L.isLoopInvariant(J0)))
      continue;
    uint64_t M = Sp->getZExtValue() / AbsT;
    if (M != 4 && M != 8 && M != 16)
      continue;
    // j zero-extended where it is in [0, 2^31); else, a counter LSR starts
    // below zero (from -t), j sign-extended where its recurrence does not
    // wrap signed (j_k = j0 + t k exactly: the hardware's signed index).
    bool Signed = false;
    if (!Relative &&
        !SE.getUnsignedRange(SE.getSCEV(&J)).getUnsignedMax().ult(1u << 31)) {
      auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&J));
      if (!AGXSignedCounterIndices || !AR || AR->getLoop() != &L ||
          !AR->hasNoSignedWrap())
        continue;
      Signed = true;
    }
    // p0 - M j0, then base + ext(j) * M before the first access (relative:
    // p0, then base + zext(+-(j - j0)) * M)
    IRBuilder<> PB(Pre->getTerminator());
    Type *I64 = PB.getInt64Ty();
    Value *Off;
    if (Relative)
      Off = ConstantInt::get(I64, 0);
    else if (auto *C = dyn_cast<ConstantInt>(J0))
      Off = ConstantInt::get(I64, -int64_t(M) * C->getSExtValue());
    else
      Off = PB.CreateNeg(PB.CreateShl(Signed ? PB.CreateSExt(J0, I64)
                                             : PB.CreateZExt(J0, I64),
                                      Log2_64(M)));
    Value *P0 = P.getIncomingValueForBlock(Pre);
    IRBuilder<> B(L.getHeader(), L.getHeader()->getFirstInsertionPt());
    Value *Z;
    if (Relative)
      Z = B.CreateZExt(T->isNegative() ? B.CreateSub(J0, &J) : B.CreateSub(&J, J0),
                       I64);
    else
      Z = Signed ? B.CreateSExt(&J, I64) : B.CreateZExt(&J, I64);
    if (IsInt) {
      // inttoptr(base + c + (ext(j) << log2 M)) for each offset c: the
      // access's base and index register.
      Value *Scaled = B.CreateShl(Z, Log2_64(M));
      DenseMap<std::pair<int64_t, unsigned>, Value *> ByOffset;
      for (auto [I, C] : Ptrs) {
        unsigned AS = cast<PointerType>(I->getType())->getAddressSpace();
        Value *&V = ByOffset[{C, AS}];
        if (!V) {
          Value *BaseC = PB.CreateAdd(PB.CreateAdd(P0, Off),
                                      ConstantInt::get(I64, C));
          V = B.CreateIntToPtr(B.CreateAdd(BaseC, Scaled, P.getName()),
                               I->getType());
        }
        I->replaceAllUsesWith(V);
      }
      for (auto [I, C] : Ptrs)
        if (isa<GetElementPtrInst>(I))
          I->eraseFromParent();
      for (auto [I, C] : Ptrs)
        if (isa<IntToPtrInst>(I))
          I->eraseFromParent();
      Inc->replaceAllUsesWith(PoisonValue::get(Inc->getType()));
      Inc->eraseFromParent();
      P.replaceAllUsesWith(PoisonValue::get(P.getType()));
      P.eraseFromParent();
      return true;
    }
    Value *Base = PB.CreatePtrAdd(P0, Off);
    Value *Addr = B.CreateGEP(B.getIntNTy(8 * M), Base, Z, P.getName());
    P.replaceUsesWithIf(Addr, [&](Use &U) { return U.getUser() != Inc; });
    Inc->replaceAllUsesWith(PoisonValue::get(Inc->getType()));
    Inc->eraseFromParent();
    P.eraseFromParent();
    return true;
  }
  return false;
}

static cl::opt<bool> AGXMergeDownCounters(
    "agx-merge-down-counters", cl::init(true), cl::Hidden,
    cl::desc("Exit on an index rather than on a counter down to zero that "
             "only the exit test uses"));

/// LSR counts a loop down to zero beside the index it steps (c = n, n - 1,
/// ..., exit when c' = c - 1 is 0; i = i0, i0 + s, ...), at an add each
/// iteration. Where c is used by the exit test only and s is odd, c' = 0 is
/// i' = i0 + s c0 exactly (mod 2^32: s is invertible, and c reaches zero at
/// the same iteration): the test on i', and c goes. An even s = 2^m odd
/// where the loop runs fewer than 2^(32 - m) times.
static cl::opt<bool> AGXDeadAfterNarrowing(
    "agx-dead-after-narrowing", cl::init(true), cl::Hidden,
    cl::desc("Delete what narrowing a loop's inductions left dead before the "
             "rewrites that count uses"));

static bool mergeDownCounter(PHINode &C, Loop &L, ScalarEvolution &SE) {
  using namespace PatternMatch;
  BasicBlock *Pre = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Pre || !Latch || !C.getType()->isIntegerTy(32) ||
      C.getNumIncomingValues() != 2)
    return false;
  auto *Cn = dyn_cast<BinaryOperator>(C.getIncomingValueForBlock(Latch));
  if (!Cn || !match(Cn, m_Add(m_Specific(&C), m_AllOnes())) ||
      Cn->getParent() != Latch || !C.hasOneUse())
    return false;
  // c' only in the phi and one exit test against zero by the latch's branch
  ICmpInst *Cmp = nullptr;
  for (User *U : Cn->users()) {
    if (U == &C)
      continue;
    auto *I = dyn_cast<ICmpInst>(U);
    if (Cmp || !I || !I->isEquality() || !match(I->getOperand(1), m_Zero()) ||
        I->getOperand(0) != Cn || !I->hasOneUse() ||
        I->user_back() != Latch->getTerminator())
      return false;
    Cmp = I;
  }
  if (!Cmp)
    return false;
  for (PHINode &I : L.getHeader()->phis()) {
    if (&I == &C || !I.getType()->isIntegerTy(32) ||
        I.getNumIncomingValues() != 2)
      continue;
    auto *In = dyn_cast<BinaryOperator>(I.getIncomingValueForBlock(Latch));
    const APInt *S;
    if (!In || !match(In, m_Add(m_Specific(&I), m_APInt(S))) || S->isZero() ||
        In->getParent() != Latch || !In->comesBefore(Latch->getTerminator()))
      continue;
    // s = 2^m odd: s c' = s n holds for c' = n mod 2^(32 - m) only, so with
    // fewer than 2^(32 - m) iterations (the trip count's constant bound).
    if (unsigned M = S->countr_zero()) {
      auto *Max =
          dyn_cast<SCEVConstant>(SE.getConstantMaxBackedgeTakenCount(&L));
      if (!Max || Max->getAPInt().getActiveBits() > 32 - M ||
          !(Max->getAPInt().zext(64) + 1).ult(APInt(64, 1).shl(32 - M)))
        continue;
    }
    IRBuilder<> PB(Pre->getTerminator());
    Value *End = PB.CreateAdd(
        I.getIncomingValueForBlock(Pre),
        PB.CreateMul(C.getIncomingValueForBlock(Pre),
                     ConstantInt::get(C.getType(), *S)));
    IRBuilder<> B(Latch->getTerminator());
    Value *NewCmp = B.CreateICmp(Cmp->getPredicate(), In, End);
    Cmp->replaceAllUsesWith(NewCmp);
    Cmp->eraseFromParent();
    Cn->replaceAllUsesWith(PoisonValue::get(Cn->getType()));
    Cn->eraseFromParent();
    C.eraseFromParent();
    return true;
  }
  return false;
}

static cl::opt<unsigned> LoopIndexBases(
    "agx-loop-index-bases", cl::init(4), cl::Hidden,
    cl::desc("Device accesses of a loop's block at B[zext(X + C)] for this "
             "many distinct constants C != 0 or more share B + zext(X), the "
             "C as immediate indices (0: never)"));

/// A bound (inclusive) on the unsigned value of S over its loop's
/// iterations: constants, the thread and threadgroup positions (the
/// threadgroup's shape or 1024, the known grid or 2^31), their sums,
/// products and zero extensions, and add recurrences up to the loop's
/// constant maximum trip count.
static std::optional<APInt> upperBound(const SCEV *S, ScalarEvolution &SE,
                                       const Function &F, unsigned Depth = 0) {
  if (Depth > 8)
    return std::nullopt;
  auto Fits = [](APInt V) -> std::optional<APInt> {
    return V.getActiveBits() <= 63 ? std::optional<APInt>(V) : std::nullopt;
  };
  if (auto *C = dyn_cast<SCEVConstant>(S))
    return C->getAPInt().isNegative() ? std::nullopt
                                      : Fits(C->getAPInt().zext(128));
  if (auto *Z = dyn_cast<SCEVZeroExtendExpr>(S))
    return upperBound(Z->getOperand(), SE, F, Depth + 1);
  if (auto *U = dyn_cast<SCEVUnknown>(S)) {
    auto *II = dyn_cast<IntrinsicInst>(U->getValue());
    if (!II)
      return std::nullopt;
    Intrinsic::ID ID = II->getIntrinsicID();
    if (std::optional<uint64_t> E = AGX::knownGridExtent(F, ID))
      return APInt(128, *E - 1);
    switch (ID) {
    case Intrinsic::agx_thread_position_in_threadgroup_x:
    case Intrinsic::agx_thread_position_in_threadgroup_y:
    case Intrinsic::agx_thread_position_in_threadgroup_z:
    case Intrinsic::agx_thread_index_in_threadgroup:
      return APInt(128, 1023);
    case Intrinsic::agx_threadgroup_position_in_grid_x:
    case Intrinsic::agx_threadgroup_position_in_grid_y:
    case Intrinsic::agx_threadgroup_position_in_grid_z:
      return APInt(128, 0x7ffffffe);
    default:
      return std::nullopt;
    }
  }
  if (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    auto *T = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    auto *N = dyn_cast<SCEVConstant>(
        SE.getConstantMaxBackedgeTakenCount(AR->getLoop()));
    std::optional<APInt> B = upperBound(AR->getStart(), SE, F, Depth + 1);
    if (!AR->isAffine() || !T || T->getAPInt().isNegative() || !N || !B)
      return std::nullopt;
    return Fits(*B + T->getAPInt().zext(128) * N->getAPInt().zext(128));
  }
  bool Add = isa<SCEVAddExpr>(S), Mul = isa<SCEVMulExpr>(S);
  if (!Add && !Mul)
    return std::nullopt;
  APInt R(128, Add ? 0 : 1);
  for (const SCEV *Op : cast<SCEVNAryExpr>(S)->operands()) {
    std::optional<APInt> B = upperBound(Op, SE, F, Depth + 1);
    if (!B)
      return std::nullopt;
    R = Add ? R + *B : R * *B;
    if (R.getActiveBits() > 63)
      return std::nullopt;
  }
  return R;
}

/// A lower bound on S's value where upperBound bounds it (so no step of it
/// wraps): constants, positions (0), sums, products, zero extensions, and
/// the start of a non-decreasing recurrence.
static std::optional<APInt> lowerBound(const SCEV *S, ScalarEvolution &SE,
                                       unsigned Depth = 0) {
  if (Depth > 8)
    return std::nullopt;
  if (auto *C = dyn_cast<SCEVConstant>(S))
    return C->getAPInt().isNegative()
               ? std::nullopt
               : std::optional<APInt>(C->getAPInt().zext(128));
  if (auto *Z = dyn_cast<SCEVZeroExtendExpr>(S))
    return lowerBound(Z->getOperand(), SE, Depth + 1);
  if (isa<SCEVUnknown>(S))
    return APInt(128, 0);
  if (auto *AR = dyn_cast<SCEVAddRecExpr>(S))
    return lowerBound(AR->getStart(), SE, Depth + 1);
  bool Add = isa<SCEVAddExpr>(S), Mul = isa<SCEVMulExpr>(S);
  if (!Add && !Mul)
    return std::nullopt;
  APInt R(128, Add ? 0 : 1);
  for (const SCEV *Op : cast<SCEVNAryExpr>(S)->operands()) {
    std::optional<APInt> B = lowerBound(Op, SE, Depth + 1);
    if (!B)
      return std::nullopt;
    R = Add ? R + *B : R * *B;
  }
  return R;
}

/// In a loop, device accesses at B[zext(X + C_k)] (an unrolled column walk:
/// x[i + 512 k]) each take a 32-bit addition for their index and a register
/// for each C_k; B + zext(X), one 64-bit addition an iteration (its uniform
/// base's copy hoisted), with the C_k as immediate indices, where X + C_k
/// provably does not wrap (upperBound). Before the structurizer, which
/// leaves SCEV no trip count. L15-E230 found the straight-line version no
/// better: there the copy is paid where the additions were.
static bool shareLoopIndexBases(Function &F, LoopInfo &LI,
                                ScalarEvolution &SE) {
  if (!LoopIndexBases || LI.empty())
    return false;
  const DataLayout &DL = F.getDataLayout();
  struct Member {
    GetElementPtrInst *GEP;
    int64_t C;
  };
  using Key = std::tuple<BasicBlock *, Value *, Value *, Type *>;
  MapVector<Key, SmallVector<Member, 8>> Groups;
  for (Instruction &I : instructions(F)) {
    if (!LI.getLoopFor(I.getParent()))
      continue;
    Value *Ptr = getLoadStorePointerOperand(&I);
    if (!Ptr || Ptr->getType()->getPointerAddressSpace() != AGX::DEVICE_ADDRESS)
      continue;
    Type *Ty = getLoadStoreType(&I);
    auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
    if (!Ty->isSized() || !GEP || GEP->getNumIndices() != 1 ||
        GEP->getParent() != I.getParent())
      continue;
    auto *Z = dyn_cast<ZExtInst>(GEP->getOperand(1));
    if (!Z || !Z->getSrcTy()->isIntegerTy(32) || !Z->getDestTy()->isIntegerTy(64))
      continue;
    Value *N = Z->getOperand(0), *X = N;
    int64_t C = 0;
    const APInt *CA;
    using namespace PatternMatch;
    if (match(N, m_Add(m_Value(X), m_APInt(CA))) ||
        match(N, m_DisjointOr(m_Value(X), m_APInt(CA)))) {
      C = CA->getSExtValue();
      // X + C wraps for no X: below 2^32 above, and at least -C below (the
      // bounds hold as the upper one keeps X's own sums from wrapping).
      const SCEV *SX = SE.getSCEV(X);
      std::optional<APInt> B = upperBound(SX, SE, F);
      if (!B || (*B + std::max<int64_t>(C, 0)).ugt(APInt(128, 0xffffffffu)))
        continue;
      if (C < 0) {
        std::optional<APInt> Lo = lowerBound(SX, SE);
        if (!Lo || Lo->ult(APInt(128, uint64_t(-C))))
          continue;
      }
    }
    Type *Elt = GEP->getSourceElementType();
    uint64_t Size = DL.getTypeAllocSize(Elt);
    unsigned EltBits = Ty->isPointerTy() ? 64 : Ty->getScalarSizeInBits();
    if (!isPowerOf2_32(EltBits) || EltBits < 8)
      continue;
    int64_t Unit = std::min(EltBits, 32u) / 8, Bytes = C * int64_t(Size);
    if (Bytes % Unit || !isInt<16>(Bytes / Unit))
      continue;
    Groups[{GEP->getParent(), GEP->getPointerOperand(), X, Elt}].push_back(
        {GEP, C});
  }
  bool Changed = false;
  SmallVector<WeakTrackingVH, 16> Dead;
  for (auto &[K, Members] : Groups) {
    SmallSet<int64_t, 8> Cs;
    for (const Member &M : Members)
      if (M.C)
        Cs.insert(M.C);
    if (Cs.size() < LoopIndexBases)
      continue;
    auto [BB, Base, X, Elt] = K;
    Instruction *First = Members.front().GEP;
    for (const Member &M : Members)
      if (M.GEP->comesBefore(First))
        First = M.GEP;
    IRBuilder<> B(First);
    Value *P = B.CreateGEP(Elt, Base, B.CreateZExt(X, B.getInt64Ty()),
                           "loop.base");
    uint64_t Size = DL.getTypeAllocSize(Elt);
    for (const Member &M : Members) {
      Value *New = P;
      if (M.C) {
        IRBuilder<> MB(M.GEP);
        New = MB.CreateGEP(MB.getInt8Ty(), P, MB.getInt64(M.C * int64_t(Size)));
      }
      M.GEP->replaceAllUsesWith(New);
      Dead.push_back(M.GEP);
    }
    Changed = true;
  }
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(Dead);
  return Changed;
}

bool AGXNarrowInductions::runOnFunction(Function &F) {
  if (!AGXNarrowInductionsOpt || skipFunction(F))
    return false;
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  bool Changed = false;
  // Signed operations of values SCEV proves non-negative (a loop counter
  // from a thread position, say) as unsigned ones: a sign extension as a
  // zero extension (whose 64-bit uses need no high word: a multiplication of
  // it is one instruction fewer, and no asr of the sign), a signed division
  // or arithmetic shift right as the unsigned / logical one.
  if (AGXNonNegativeCasts) {
    SmallVector<Instruction *, 8> Work;
    for (Instruction &I : instructions(F))
      if (isa<SExtInst>(I) ||
          ((I.getOpcode() == Instruction::AShr ||
            I.getOpcode() == Instruction::SDiv) &&
           I.getType()->isIntegerTy()))
        Work.push_back(&I);
    for (Instruction *I : Work) {
      Value *X = I->getOperand(0);
      if (!SE.isSCEVable(X->getType()) || !SE.isKnownNonNegative(SE.getSCEV(X)))
        continue;
      Instruction *R;
      if (isa<SExtInst>(I)) {
        R = new ZExtInst(X, I->getType(), "", I->getIterator());
        R->setNonNeg(true);
      } else if (I->getOpcode() == Instruction::AShr) {
        R = BinaryOperator::CreateLShr(X, I->getOperand(1), "", I->getIterator());
        R->setIsExact(I->isExact());
      } else {
        // x / d for x >= 0 is udiv only where d > 0 too
        Value *D = I->getOperand(1);
        if (!SE.isKnownPositive(SE.getSCEV(D)))
          continue;
        R = BinaryOperator::CreateUDiv(X, D, "", I->getIterator());
        R->setIsExact(I->isExact());
      }
      R->takeName(I);
      R->setDebugLoc(I->getDebugLoc());
      I->replaceAllUsesWith(R);
      SE.forgetValue(I);
      I->eraseFromParent();
      Changed = true;
    }
  }
  for (Loop *L : LI.getLoopsInPreorder()) {
    SmallVector<PHINode *, 4> Phis;
    for (PHINode &Phi : L->getHeader()->phis())
      if (Phi.getType()->isIntegerTy(64))
        Phis.push_back(&Phi);
    bool Narrowed = false;
    for (PHINode *Phi : Phis)
      Narrowed |= narrow(*Phi, *L, SE);
    // The extensions of a narrowed value that its old 64-bit uses left
    // (dead, but uses that the one-use tests below would count).
    if (Narrowed && AGXDeadAfterNarrowing) {
      SmallVector<WeakTrackingVH, 8> Dead;
      for (BasicBlock *BB : L->blocks())
        for (Instruction &I : *BB)
          if (isInstructionTriviallyDead(&I))
            Dead.push_back(&I);
      for (WeakTrackingVH &V : Dead)
        if (auto *I = dyn_cast_or_null<Instruction>(V)) {
          SE.forgetValue(I);
          RecursivelyDeleteTriviallyDeadInstructions(I);
        }
    }
    Changed |= Narrowed;
    if (AGXElementIndices && L->isInnermost()) {
      SmallVector<PHINode *, 4> Narrow;
      for (PHINode &Phi : L->getHeader()->phis())
        if (Phi.getType()->isIntegerTy(32))
          Narrow.push_back(&Phi);
      for (PHINode *Phi : Narrow)
        Changed |= elementIndex(*Phi, *L);
    }
    if (AGXPointersFromCounters && L->isInnermost()) {
      SmallVector<PHINode *, 4> Ptrs;
      for (PHINode &Phi : L->getHeader()->phis())
        if (Phi.getType()->isPointerTy() || Phi.getType()->isIntegerTy(64))
          Ptrs.push_back(&Phi);
      for (PHINode *Phi : Ptrs)
        Changed |= pointerFromCounter(*Phi, *L, SE);
    }
    if (AGXMergeDownCounters && L->isInnermost()) {
      SmallVector<PHINode *, 4> Counters;
      for (PHINode &Phi : L->getHeader()->phis())
        if (Phi.getType()->isIntegerTy(32))
          Counters.push_back(&Phi);
      for (PHINode *Phi : Counters)
        if (mergeDownCounter(*Phi, *L, SE)) {
          Changed = true;
          break;
        }
    }
    if (AGXSplitOffsetInductions && L->isInnermost()) {
      SmallVector<PHINode *, 4> Wide;
      for (PHINode &Phi : L->getHeader()->phis())
        if (Phi.getType()->isIntegerTy(64))
          Wide.push_back(&Phi);
      for (PHINode *Phi : Wide)
        Changed |= splitOffsets(*Phi, *L, SE, DT);
    }
  }
  Changed |= shareLoopIndexBases(F, LI, SE);
  if (!AGXSplitWideCounters)
    return Changed;
  // Innermost loops only: each split adds blocks that the other loops'
  // LoopInfo does not see.
  SmallVector<WideCounter, 4> Counters;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (!L->isInnermost())
      continue;
    for (PHINode &Phi : L->getHeader()->phis())
      if (Phi.getType()->isIntegerTy(64))
        if (auto W = matchWideCounter(Phi, *L, SE)) {
          Counters.push_back(*W);
          break;
        }
  }
  for (const WideCounter &W : Counters)
    splitWideCounter(W);
  return Changed || !Counters.empty();
}
