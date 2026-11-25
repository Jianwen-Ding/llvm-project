//===-- AGXWideIndexVersioning.cpp - 32 x 32-bit products in loops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A loop's 64-bit product of a sign-extended 32-bit value and a loop-
// invariant 32-bit value sign-extended and scaled by a constant (the PTX
// front end's address (long)idx[i] * n * 4),
//
//   %s = shl nsw i64 (sext i32 %y), 2        (or mul nsw i64 (sext %y), c)
//   %p = mul i64 (sext i32 %x), %s
//   %a = add i64 %p, %base
//
// is a 64 x 32-bit multiplication and a 64-bit add (on the core half's shared
// unit, 3.92 cycles each); where y << 2 fits in 32 bits it is the 32 x 32 ->
// 64-bit multiply-add of x and that, with the base as its addend. The loop
// runs as two copies:
//
//   if (every such y scaled fits in 32 bits)
//     loop, %s as sext(shl nsw i32 %y, 2)
//   else
//     loop as it was
//
// The loop whose own blocks hold the products (an atomic's compare-exchange
// loop may be inside it), not the copies other versioning leaves for its
// fallbacks; with convergent operations only where every y is uniform (from
// kernel arguments and constants), so that a SIMD group takes one copy.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-wide-index-versioning"

static cl::opt<bool> EnableWideIndexVersioning(
    "agx-loop-wide-products", cl::init(true), cl::Hidden,
    cl::desc("Version innermost loops on a loop-invariant 32-bit value "
             "scaled by a constant fitting in 32 bits, the copy's 64-bit "
             "products of it 32 x 32-bit multiplications"));

namespace {

class AGXWideIndexVersioning : public FunctionPass {
public:
  static char ID;
  AGXWideIndexVersioning() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "AGX Wide Index Versioning";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
  }
};

/// A loop-invariant i64 value sext(Y) * Scale (Y i32) a product reads.
struct Scaled {
  Value *Y;
  int64_t Scale;
};

} // namespace

char AGXWideIndexVersioning::ID = 0;

INITIALIZE_PASS_BEGIN(AGXWideIndexVersioning, DEBUG_TYPE,
                      "AGX Wide Index Versioning", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_END(AGXWideIndexVersioning, DEBUG_TYPE,
                    "AGX Wide Index Versioning", false, false)

FunctionPass *llvm::createAGXWideIndexVersioningPass() {
  return new AGXWideIndexVersioning();
}

/// S as sext(Y) * Scale for an i32 Y and a constant Scale of 2 to 2^30.
static std::optional<Scaled> scaledOf(Value *S) {
  Value *Y;
  const APInt *C;
  if (match(S, m_Shl(m_SExt(m_Value(Y)), m_APInt(C))) &&
      Y->getType()->isIntegerTy(32) && C->uge(1) && C->ule(30))
    return Scaled{Y, int64_t(1) << C->getZExtValue()};
  if (match(S, m_c_Mul(m_SExt(m_Value(Y)), m_APInt(C))) &&
      Y->getType()->isIntegerTy(32) && C->sgt(1) && C->slt(int64_t(1) << 30))
    return Scaled{Y, C->getSExtValue()};
  return std::nullopt;
}

/// Whether V is the same for every thread of a dispatch: computed from
/// constants, arguments, loads of by-reference kernel arguments (the
/// constant address space) and the threadgroup's size.
static bool isUniform(Value *V, unsigned Depth = 0) {
  if (isa<Constant>(V) || isa<Argument>(V))
    return true;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || Depth > 6)
    return false;
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerAddressSpace() == 4 && isUniform(LI->getPointerOperand(), Depth + 1);
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::agx_threads_per_threadgroup_x:
    case Intrinsic::agx_threads_per_threadgroup_y:
    case Intrinsic::agx_threads_per_threadgroup_z:
      return true;
    default:
      return false;
    }
  }
  if (I->mayReadOrWriteMemory() || isa<PHINode>(I) || isa<CallBase>(I))
    return false;
  return all_of(I->operands(),
                [&](Value *Op) { return isUniform(Op, Depth + 1); });
}

/// Versions L on Unsafe: the original L runs where it is false, a copy as L
/// was where it is true. L is in simplify form, with one exit block, in
/// LCSSA form.
static void versionOn(Loop *L, Value *Unsafe, LoopInfo &LI,
                      DominatorTree &DT) {
  BasicBlock *CheckBB = L->getLoopPreheader();
  BasicBlock *PH = SplitBlock(CheckBB, CheckBB->getTerminator(), &DT, &LI,
                              nullptr, L->getHeader()->getName() + ".wide.ph");
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> Blocks;
  Loop *Orig = cloneLoopWithPreheader(PH, CheckBB, L, VMap, ".wide.orig", &LI,
                                      &DT, Blocks);
  remapInstructionsInBlocks(Blocks, VMap);
  Instruction *Term = CheckBB->getTerminator();
  IRBuilder<>(Term).CreateCondBr(Unsafe, Orig->getLoopPreheader(),
                                 L->getLoopPreheader());
  Term->eraseFromParent();
  BasicBlock *Exit = L->getExitBlock();
  DT.changeImmediateDominator(Exit, CheckBB);
  // (LCSSA: the exit block's phis take the copy's values too)
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

bool AGXWideIndexVersioning::runOnFunction(Function &F) {
  if (!EnableWideIndexVersioning || skipFunction(F) ||
      F.getCallingConv() != CallingConv::AGX_KERNEL)
    return false;
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  SmallVector<Loop *, 8> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (!findStringMetadataForLoop(L, "llvm.loop.agx.alias.fallback"))
      Loops.push_back(L);
  bool Changed = false;
  for (Loop *L : Loops) {
    // the scaled values the loop's products read, each with its products
    MapVector<Value *, std::pair<Scaled, SmallVector<Instruction *, 4>>> Found;
    bool Convergent = false;
    for (BasicBlock *BB : L->blocks())
      for (Instruction &I : *BB) {
        if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->isConvergent())
          Convergent = true;
        // (the products of L's own blocks, not of a loop inside it: a
        // compare-exchange loop of an atomic)
        if (LI.getLoopFor(BB) != L)
          continue;
        Value *A, *Bv;
        if (!match(&I, m_Mul(m_Value(A), m_Value(Bv))) ||
            !I.getType()->isIntegerTy(64))
          continue;
        for (int K = 0; K < 2; ++K) {
          Value *X = K ? Bv : A, *S = K ? A : Bv;
          Value *XI;
          if (!match(X, m_SExt(m_Value(XI))) || !XI->getType()->isIntegerTy(32))
            continue;
          auto *SI = dyn_cast<Instruction>(S);
          if (!SI || L->contains(SI))
            continue;
          std::optional<Scaled> Sc = scaledOf(S);
          if (!Sc || !DT.dominates(SI, L->getLoopPreheader()->getTerminator()))
            continue;
          if (auto *YI = dyn_cast<Instruction>(Sc->Y);
              YI && !DT.dominates(YI, L->getLoopPreheader()->getTerminator()))
            continue;
          auto &E = Found[S];
          E.first = *Sc;
          E.second.push_back(&I);
          break;
        }
      }
    if (Found.empty())
      continue;
    if (Convergent && !all_of(Found, [](auto &E) { return isUniform(E.second.first.Y); }))
      continue;
    if (!L->isLoopSimplifyForm()) {
      Changed |= simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false);
      if (!L->isLoopSimplifyForm())
        continue;
    }
    if (!L->getUniqueExitBlock() || !L->getExitingBlock() || !L->getExitBlock())
      continue;
    Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
    // y * scale fits in 32 bits: y in [ceil(-2^31 / scale),
    // floor((2^31 - 1) / scale)]
    IRBuilder<> B(L->getLoopPreheader()->getTerminator());
    Value *Unsafe = nullptr;
    for (auto &E : Found) {
      Scaled Sc = E.second.first;
      int64_t Lo = -((int64_t(1) << 31) / Sc.Scale);
      int64_t Hi = ((int64_t(1) << 31) - 1) / Sc.Scale;
      Value *Off = B.CreateSub(Sc.Y, B.getInt32(uint32_t(Lo)));
      Value *Out = B.CreateICmpUGT(Off, B.getInt32(uint32_t(Hi - Lo)));
      Unsafe = Unsafe ? B.CreateOr(Unsafe, Out) : Out;
    }
    versionOn(L, Unsafe, LI, DT);
    // in the copy for values that fit: each scaled value a 32-bit product
    // (no signed wrap) sign-extended
    B.SetInsertPoint(L->getLoopPreheader()->getTerminator());
    for (auto &E : Found) {
      Scaled Sc = E.second.first;
      bool Pow2 = isPowerOf2_64(uint64_t(Sc.Scale));
      Value *Y32 =
          Pow2 ? B.CreateShl(Sc.Y, Log2_64(uint64_t(Sc.Scale)), "", false, true)
               : B.CreateMul(Sc.Y, B.getInt32(uint32_t(Sc.Scale)), "", false, true);
      Value *NewS = B.CreateSExt(Y32, B.getInt64Ty());
      // (the products' operands only: the loop's other uses, a pointer's
      // step, keep the 64-bit value, which strength reduction formed its
      // addresses with)
      for (Instruction *M : E.second.second)
        M->replaceUsesOfWith(E.first, NewS);
    }
    LLVM_DEBUG(dbgs() << "agx-wide-index-versioning: " << F.getName() << " "
                      << L->getHeader()->getName() << " " << Found.size()
                      << " scaled values\n");
    Changed = true;
  }
  if (Changed)
    SE.forgetAllLoops();
  return Changed;
}
