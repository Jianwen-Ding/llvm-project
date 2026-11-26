//===-- AGXCmpXchgLoops.cpp - Compare-exchange loops as atomic RMWs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A loop that applies an operation to memory with a compare-exchange
//
//   loop:
//     %old = phi [ %guess, %pre ], [ %found, %loop ]
//     %new = OP(%old, %v)                 ; %v invariant, through bitcasts
//     %x = cmpxchg ptr %p, %old, %new     ; strong
//     %found = extractvalue %x, 0
//     %ok = extractvalue %x, 1            ; or icmp eq %found, %old
//     br %ok, %exit, %loop
//
// is an atomicrmw OP of %p and %v when the target has that operation natively
// (MSL has no atomic float add before Metal 3 and no float min / max at all:
// Metal code and Metal's AIR spell them as such loops; CUDA's atomicAdd on
// floats before sm_20 and its custom atomics too). The atomicrmw is the
// execution of the loop whose first compare-exchange succeeds, which the loop
// allows: it returns the value it found, the loop's %found at its exit, and
// leaves OP(found, v) in memory. Its ordering and scope are the
// compare-exchange's success ones, and it is volatile if that is. The guess
// is any value (each failure takes the value found), so nothing else changes.
//
// Only operations the target leaves as instructions are made
// (shouldExpandAtomicRMWInIR: G13 32-bit integer operations, G14 also float
// add): the others would be expanded back into a compare-exchange loop.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-cmpxchg-loops"

STATISTIC(NumLoops, "Compare-exchange loops made atomic RMWs");

static cl::opt<bool> EnableCmpXchgLoops(
    "agx-cmpxchg-loops", cl::init(true), cl::Hidden,
    cl::desc("Compile compare-exchange loops that apply an operation the "
             "target has natively as that atomic operation"));

namespace {
struct LoopOp {
  AtomicRMWInst::BinOp Op;
  Value *Operand; // In the RMW's type (the compare-exchange's).
};
} // namespace

/// Strips bitcasts between the compare-exchange's integer and a float.
static Value *stripBitCasts(Value *V) {
  while (auto *BC = dyn_cast<BitCastInst>(V))
    V = BC->getOperand(0);
  return V;
}

/// What %new is of %old: an operation with an operand defined outside the
/// loop, or that operand alone (an exchange). The operand is returned in the
/// type of the compare-exchange (an integer), and must be made so outside the
/// loop.
static std::optional<std::pair<AtomicRMWInst::BinOp, Value *>>
matchOp(Value *New, PHINode *Old, BasicBlock *Loop) {
  auto Outside = [&](Value *V) {
    auto *I = dyn_cast<Instruction>(V);
    return !I || I->getParent() != Loop;
  };
  // The value the integer %new holds, as a float when it is one.
  Value *N = stripBitCasts(New);
  auto IsOld = [&](Value *V) { return stripBitCasts(V) == Old; };
  if (Outside(N))
    return std::make_pair(AtomicRMWInst::Xchg, N);
  auto *I = dyn_cast<Instruction>(N);
  if (!I || I->getParent() != Loop)
    return std::nullopt;
  Value *A, *B;
  auto Pick = [&](AtomicRMWInst::BinOp Op, bool Commutes)
      -> std::optional<std::pair<AtomicRMWInst::BinOp, Value *>> {
    if (IsOld(A) && Outside(B))
      return std::make_pair(Op, B);
    if (Commutes && IsOld(B) && Outside(A))
      return std::make_pair(Op, A);
    return std::nullopt;
  };
  if (match(I, m_Add(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::Add, true);
  if (match(I, m_Sub(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::Sub, false);
  if (match(I, m_And(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::And, true);
  if (match(I, m_Or(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::Or, true);
  if (match(I, m_Xor(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::Xor, true);
  if (match(I, m_SMax(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::Max, true);
  if (match(I, m_SMin(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::Min, true);
  if (match(I, m_UMax(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::UMax, true);
  if (match(I, m_UMin(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::UMin, true);
  if (match(I, m_FAdd(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::FAdd, true);
  if (match(I, m_FSub(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::FSub, false);
  if (match(I, m_Intrinsic<Intrinsic::maxnum>(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::FMax, true);
  if (match(I, m_Intrinsic<Intrinsic::minnum>(m_Value(A), m_Value(B))))
    return Pick(AtomicRMWInst::FMin, true);
  return std::nullopt;
}

static bool isFloatOp(AtomicRMWInst::BinOp Op) {
  return Op == AtomicRMWInst::FAdd || Op == AtomicRMWInst::FSub ||
         Op == AtomicRMWInst::FMax || Op == AtomicRMWInst::FMin;
}

/// Loop, a block that branches to itself, as an atomic RMW in its
/// predecessor outside the loop.
static bool convertLoop(BasicBlock &Loop, const TargetLowering &TLI) {
  auto *Br = dyn_cast<CondBrInst>(Loop.getTerminator());
  if (!Br)
    return false;
  unsigned Back = Br->getSuccessor(0) == &Loop ? 0 : 1;
  if (Br->getSuccessor(Back) != &Loop || Br->getSuccessor(1 - Back) == &Loop)
    return false;
  BasicBlock *Exit = Br->getSuccessor(1 - Back);
  BasicBlock *Pre = nullptr;
  for (BasicBlock *P : predecessors(&Loop)) {
    if (P == &Loop)
      continue;
    if (Pre && Pre != P)
      return false;
    Pre = P;
  }
  if (!Pre || !isa<UncondBrInst>(Pre->getTerminator()))
    return false;

  // The one compare-exchange, and nothing else that touches memory.
  AtomicCmpXchgInst *CX = nullptr;
  for (Instruction &I : Loop) {
    if (auto *C = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (CX)
        return false;
      CX = C;
      continue;
    }
    if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() ||
        (isa<IntrinsicInst>(&I) && !isa<MinMaxIntrinsic>(&I) &&
         !match(&I, m_Intrinsic<Intrinsic::maxnum>()) &&
         !match(&I, m_Intrinsic<Intrinsic::minnum>())))
      return false;
  }
  if (!CX)
    return false;
  Value *Ptr = CX->getPointerOperand();
  if (auto *PI = dyn_cast<Instruction>(Ptr); PI && PI->getParent() == &Loop)
    return false;
  auto *Old = dyn_cast<PHINode>(CX->getCompareOperand());
  if (!Old || Old->getParent() != &Loop || Old->getNumIncomingValues() != 2)
    return false;
  // %found is what the compare-exchange returns, fed back as %old.
  Value *Back0 = Old->getIncomingValueForBlock(&Loop);
  ExtractValueInst *Found = nullptr;
  for (User *U : CX->users()) {
    auto *EV = dyn_cast<ExtractValueInst>(U);
    if (!EV || EV->getNumIndices() != 1 || EV->getParent() != &Loop)
      return false;
    if (EV->getIndices()[0] == 0) {
      if (Found && Found != EV)
        return false;
      Found = EV;
    }
  }
  if (!Found || Back0 != Found)
    return false;
  // The exit condition: success, by the flag, or (for a strong one) by the
  // value found being the one expected.
  Value *Cond = Br->getCondition();
  bool ExitsOnTrue = Back == 1;
  if (!ExitsOnTrue)
    return false;
  bool Success = false;
  if (auto *EV = dyn_cast<ExtractValueInst>(Cond))
    Success = EV->getAggregateOperand() == CX && EV->getIndices()[0] == 1;
  else if (!CX->isWeak())
    Success = match(Cond, m_c_SpecificICmp(ICmpInst::ICMP_EQ,
                                           m_Specific(Found), m_Specific(Old)));
  if (!Success)
    return false;

  std::optional<std::pair<AtomicRMWInst::BinOp, Value *>> M =
      matchOp(CX->getNewValOperand(), Old, &Loop);
  if (!M)
    return false;
  auto [Op, Operand] = *M;
  Type *IntTy = CX->getCompareOperand()->getType();
  if (!IntTy->isIntegerTy())
    return false;
  // Float operations take a float of the integer's width, by bitcasts; the
  // integer ones the integer itself.
  Type *RMWTy = IntTy;
  if (isFloatOp(Op)) {
    if (!IntTy->isIntegerTy(32) || !Operand->getType()->isFloatTy())
      return false;
    RMWTy = Operand->getType();
    // The float operation is of %old as a float.
    Value *NewF = stripBitCasts(CX->getNewValOperand());
    if (NewF->getType() != RMWTy)
      return false;
  } else if (Op == AtomicRMWInst::Xchg) {
    if (Operand->getType() != IntTy) {
      if (!CastInst::isBitCastable(Operand->getType(), IntTy))
        return false;
    }
  } else if (Operand->getType() != IntTy) {
    return false;
  }
  // Only the loop's value found leaves it (as the old value).
  for (Instruction &I : Loop)
    for (User *U : I.users())
      if (cast<Instruction>(U)->getParent() != &Loop && &I != Found)
        return false;

  IRBuilder<> B(Pre->getTerminator());
  Value *V = Operand;
  if (Op == AtomicRMWInst::Xchg && V->getType() != IntTy)
    V = B.CreateBitCast(V, IntTy);
  auto *RMW = B.CreateAtomicRMW(Op, Ptr, V, CX->getAlign(),
                                CX->getSuccessOrdering(), CX->getSyncScopeID());
  RMW->setVolatile(CX->isVolatile());
  if (TLI.shouldExpandAtomicRMWInIR(RMW) !=
      TargetLowering::AtomicExpansionKind::None) {
    RMW->eraseFromParent();
    if (V != Operand && V->use_empty())
      cast<Instruction>(V)->eraseFromParent();
    return false;
  }
  Value *Result = RMW;
  if (RMW->getType() != IntTy)
    Result = B.CreateBitCast(RMW, IntTy);

  // The predecessor goes to the exit, which receives the result where it
  // received the loop's value found.
  // (The loop's own entries go with the loop.)
  for (PHINode &P : Exit->phis()) {
    Value *In = P.getIncomingValueForBlock(&Loop);
    P.addIncoming(In == Found ? Result : In, Pre);
  }
  for (Use &U : make_early_inc_range(Found->uses()))
    if (auto *UI = dyn_cast<Instruction>(U.getUser());
        UI && UI->getParent() != &Loop && !isa<PHINode>(UI))
      U.set(Result);
  Pre->getTerminator()->setSuccessor(0, Exit);
  for (PHINode &P : Loop.phis())
    P.removeIncomingValue(Pre, /*DeletePHIIfEmpty=*/false);
  DeleteDeadBlock(&Loop);
  ++NumLoops;
  return true;
}

PreservedAnalyses AGXCmpXchgLoopsPass::run(Function &F,
                                           FunctionAnalysisManager &) {
  if (!EnableCmpXchgLoops)
    return PreservedAnalyses::all();
  const TargetLowering &TLI = *TM.getSubtargetImpl(F)->getTargetLowering();
  SmallVector<BasicBlock *, 4> Loops;
  for (BasicBlock &BB : F)
    if (is_contained(successors(&BB), &BB))
      Loops.push_back(&BB);
  bool Changed = false;
  for (BasicBlock *BB : Loops)
    Changed |= convertLoop(*BB, TLI);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
