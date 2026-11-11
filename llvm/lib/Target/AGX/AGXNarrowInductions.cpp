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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-narrow-inductions"

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
  if (!fitsSigned32(SE, &Phi) || !fitsSigned32(SE, Inc) ||
      !fitsSigned32(SE, Step))
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
                  /*HasNUW=*/true, /*HasNSW=*/true);
  NarrowPhi->addIncoming(NarrowStart, Preheader);
  NarrowPhi->addIncoming(NarrowInc, Latch);

  B.SetInsertPoint(Phi.getParent(), Phi.getParent()->getFirstInsertionPt());
  Value *WidePhi = B.CreateZExt(NarrowPhi, Phi.getType(), Phi.getName());
  B.SetInsertPoint(Inc);
  Value *WideInc = B.CreateZExt(NarrowInc, Inc->getType(), Inc->getName());
  // The zero extension of the increment replaces it after it is computed.
  cast<Instruction>(WideInc)->moveAfter(cast<Instruction>(NarrowInc));
  Inc->replaceAllUsesWith(WideInc);
  Phi.replaceAllUsesWith(WidePhi);
  Inc->eraseFromParent();
  Phi.eraseFromParent();
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
/// before every access. Where every access is of 32-bit elements and the
/// offset's start, step and the constants it is compared with are multiples
/// of 4, it is 4 j for an element index j (o < 2^32, so 4 j does not wrap):
/// j steps by s / 4, the accesses take base + zext(j) << 2 as their index,
/// and the compares are of j with k / 4.
static bool elementIndex(PHINode &P, Loop &L) {
  using namespace PatternMatch;
  BasicBlock *Pre = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Pre || !Latch || P.getNumIncomingValues() != 2 ||
      !P.getType()->isIntegerTy(32))
    return false;
  const APInt *C0, *St;
  auto *Inc = dyn_cast<BinaryOperator>(P.getIncomingValueForBlock(Latch));
  if (!match(P.getIncomingValueForBlock(Pre), m_APInt(C0)) || !Inc ||
      !match(Inc, m_Add(m_Specific(&P), m_APInt(St))) || C0->urem(4) ||
      St->urem(4) || St->isZero())
    return false;
  SmallVector<GetElementPtrInst *, 4> Geps;
  SmallVector<std::pair<ICmpInst *, bool>, 2> Cmps; // (compare, of Inc)
  SmallVector<ZExtInst *, 2> Exts;
  auto Divisible = [](Value *V) {
    auto *C = dyn_cast<ConstantInt>(V);
    return C && C->getValue().urem(4) == 0;
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
      if (!T || T->getScalarSizeInBits() != 32)
        return false;
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
  if (Geps.empty())
    return false;
  // the element index and its step
  IRBuilder<> B(&P);
  PHINode *J = B.CreatePHI(P.getType(), 2, P.getName() + ".elt");
  IRBuilder<> BI(Inc);
  Value *JN = BI.CreateAdd(J, ConstantInt::get(P.getType(), St->udiv(4)),
                           J->getName() + ".next", /*HasNUW=*/true);
  J->addIncoming(ConstantInt::get(P.getType(), C0->udiv(4)), Pre);
  J->addIncoming(JN, Latch);
  for (GetElementPtrInst *G : Geps) {
    IRBuilder<> BG(G);
    Value *Z = BG.CreateZExt(J, G->getOperand(1)->getType());
    Value *N = BG.CreateGEP(BG.getInt32Ty(), G->getPointerOperand(), Z,
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
    C->setOperand(1, ConstantInt::get(K->getType(), K->getValue().udiv(4)));
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

bool AGXNarrowInductions::runOnFunction(Function &F) {
  if (!AGXNarrowInductionsOpt || skipFunction(F))
    return false;
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    SmallVector<PHINode *, 4> Phis;
    for (PHINode &Phi : L->getHeader()->phis())
      if (Phi.getType()->isIntegerTy(64))
        Phis.push_back(&Phi);
    for (PHINode *Phi : Phis)
      Changed |= narrow(*Phi, *L, SE);
    if (AGXElementIndices && L->isInnermost()) {
      SmallVector<PHINode *, 4> Narrow;
      for (PHINode &Phi : L->getHeader()->phis())
        if (Phi.getType()->isIntegerTy(32))
          Narrow.push_back(&Phi);
      for (PHINode *Phi : Narrow)
        Changed |= elementIndex(*Phi, *L);
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
