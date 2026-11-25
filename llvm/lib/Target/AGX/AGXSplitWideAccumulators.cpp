//===-- AGXSplitWideAccumulators.cpp - A 64-bit sum of 32-bit values ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A loop's 64-bit sum of zero-extended 32-bit values,
//
//   %s = phi i64 [ %init, %ph ], [ %s.next, %latch ]
//   %t = add i64 %s, (zext i32 %a)
//   %s.next = add i64 %t, (zext i32 %b)
//
// is a 64-bit add per value, on the core half's unit that 64-bit adds and
// multiply-adds and the bit-field operations share (3.92 cycles; sum_i64's
// loop does nothing else). As two 32-bit words,
//
//   %lo.1 = add i32 %lo, %a ; %c.1 = icmp ult i32 %lo.1, %a
//   %hi.1 = add i32 %hi, zext %c.1
//
// each value is three ALU operations, which the schedulers issue on their
// own (L15-E678). The sum is put together again after the loop.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-split-wide-accumulators"

static cl::opt<bool> EnableSplitWideAccumulators(
    "agx-split-wide-sums", cl::init(true), cl::Hidden,
    cl::desc("A loop's 64-bit sum of zero-extended 32-bit values in two "
             "32-bit words with the carries"));

namespace {

class AGXSplitWideAccumulators : public FunctionPass {
public:
  static char ID;
  AGXSplitWideAccumulators() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "AGX Split Wide Accumulators";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};

} // namespace

char AGXSplitWideAccumulators::ID = 0;

INITIALIZE_PASS_BEGIN(AGXSplitWideAccumulators, DEBUG_TYPE,
                      "AGX Split Wide Accumulators", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(AGXSplitWideAccumulators, DEBUG_TYPE,
                    "AGX Split Wide Accumulators", false, false)

FunctionPass *llvm::createAGXSplitWideAccumulatorsPass() {
  return new AGXSplitWideAccumulators();
}

bool AGXSplitWideAccumulators::runOnFunction(Function &F) {
  if (!EnableSplitWideAccumulators || skipFunction(F))
    return false;
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    BasicBlock *Latch = L->getLoopLatch(), *PH = L->getLoopPreheader();
    if (!Latch || !PH)
      continue;
    SmallVector<PHINode *, 2> Phis;
    for (PHINode &PN : L->getHeader()->phis())
      if (PN.getType()->isIntegerTy(64) && PN.getNumIncomingValues() == 2)
        Phis.push_back(&PN);
    for (PHINode *P : Phis) {
      // the chain of adds from P to its value from the latch, each of a
      // zero-extended i32
      auto *Next = dyn_cast<Instruction>(P->getIncomingValueForBlock(Latch));
      if (!Next || !L->contains(Next))
        continue;
      SmallVector<std::pair<Instruction *, Value *>, 8> Chain;
      Value *V = Next;
      bool OK = true;
      while (V != P) {
        auto *Add = dyn_cast<BinaryOperator>(V);
        Value *A, *X;
        if (!Add || Add->getOpcode() != Instruction::Add || !L->contains(Add) ||
            Chain.size() > 64) {
          OK = false;
          break;
        }
        if (match(Add->getOperand(1), m_ZExt(m_Value(X))) &&
            X->getType()->isIntegerTy(32))
          A = Add->getOperand(0);
        else if (match(Add->getOperand(0), m_ZExt(m_Value(X))) &&
                 X->getType()->isIntegerTy(32))
          A = Add->getOperand(1);
        else {
          OK = false;
          break;
        }
        Chain.push_back({Add, X});
        V = A;
      }
      if (!OK || Chain.empty())
        continue;
      std::reverse(Chain.begin(), Chain.end());
      Type *I32 = Type::getInt32Ty(F.getContext());
      Type *I64 = Type::getInt64Ty(F.getContext());
      IRBuilder<> HB(P);
      PHINode *Lo = HB.CreatePHI(I32, 2, P->getName() + ".lo");
      PHINode *Hi = HB.CreatePHI(I32, 2, P->getName() + ".hi");
      Value *Init = P->getIncomingValueForBlock(PH);
      IRBuilder<> PB(PH->getTerminator());
      Lo->addIncoming(PB.CreateTrunc(Init, I32), PH);
      Hi->addIncoming(PB.CreateTrunc(PB.CreateLShr(Init, 32), I32), PH);
      // the 64-bit sums again (a register pair) for their other users
      auto Join = [&](IRBuilder<> &B, Value *L, Value *H) {
        return B.CreateOr(B.CreateShl(B.CreateZExt(H, I64), 32),
                          B.CreateZExt(L, I64));
      };
      IRBuilder<> JB(&*L->getHeader()->getFirstInsertionPt());
      Value *Start = Join(JB, Lo, Hi);
      Value *CurLo = Lo, *CurHi = Hi;
      SmallVector<std::pair<Instruction *, Value *>, 8> Sums;
      for (auto [Add, X] : Chain) {
        IRBuilder<> B(Add);
        Value *NLo = B.CreateAdd(CurLo, X);
        Value *C = B.CreateICmpULT(NLo, X);
        CurHi = B.CreateAdd(CurHi, B.CreateZExt(C, I32));
        CurLo = NLo;
        Sums.push_back({Add, Join(B, CurLo, CurHi)});
      }
      Lo->addIncoming(CurLo, Latch);
      Hi->addIncoming(CurHi, Latch);
      P->replaceAllUsesWith(Start);
      P->eraseFromParent();
      SmallVector<WeakTrackingVH, 16> Dead{Start};
      for (auto [Add, Sum] : llvm::reverse(Sums)) {
        Add->replaceAllUsesWith(Sum);
        Dead.push_back(Sum);
        Dead.push_back(Add->getOperand(0));
        Dead.push_back(Add->getOperand(1));
        Add->eraseFromParent();
      }
      RecursivelyDeleteTriviallyDeadInstructionsPermissive(Dead);
      LLVM_DEBUG(dbgs() << "agx-split-wide-accumulators: " << F.getName() << " "
                        << L->getHeader()->getName() << " " << Chain.size()
                        << " values\n");
      Changed = true;
    }
  }
  return Changed;
}
