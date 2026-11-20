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
#include "Utils/AGXBaseInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"

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
    "agx-alias-versioning-math-per-word", cl::init(4), cl::Hidden,
    cl::desc("Arithmetic (a transcendental 16, a division 8) per 32-bit word "
             "loaded above which a loop's latency is hidden behind its work "
             "and it is not versioned"));

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

/// A single-block innermost loop with a device store and a device load, and
/// no other memory operation (calls, threadgroup memory, atomics) nor a
/// convergent one, which a divergent check must not duplicate. Accesses of
/// 1- or 2-byte elements are left: alignment versioning copies such a loop
/// again, and four copies cost more fetching than the loads gain (L15-E527).
static bool isCandidate(const Loop *L, const DataLayout &DL) {
  auto Narrow = [&](Type *Ty) {
    return DL.getTypeStoreSize(Ty->getScalarType()) < 4;
  };
  if (!L->isInnermost() || L->getNumBlocks() != 1)
    return false;
  bool Store = false, Load = false;
  for (const Instruction &I : *L->getHeader()) {
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
          Narrow(LI->getType()))
        return false;
      Load = true;
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isSimple() ||
          SI->getPointerAddressSpace() != AGX::DEVICE_ADDRESS ||
          Narrow(SI->getValueOperand()->getType()))
        return false;
      Store = true;
    } else if (I.mayReadOrWriteMemory()) {
      return false;
    }
  }
  return Store && Load;
}

/// The work of a loop iteration: (32-bit words loaded, arithmetic). Hoisted
/// ahead of the stores, many words cost registers (and resident groups), and
/// a loop with much arithmetic per word already hides its loads' latency
/// behind it and the groups: neither gains from a copy (L15-E527).
static std::pair<unsigned, unsigned> iterationWork(const Loop *L,
                                                   const DataLayout &DL) {
  unsigned Words = 0, Math = 0;
  for (const Instruction &I : *L->getHeader()) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      Words += divideCeil(DL.getTypeStoreSize(LI->getType()), 4);
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
  return {Words, Math};
}

bool AGXAliasVersioning::runOnFunction(Function &F) {
  if (!EnableAliasVersioning || skipFunction(F) ||
      F.getCallingConv() != CallingConv::AGX_KERNEL)
    return false;
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  SmallVector<Loop *, 8> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (isCandidate(L, F.getDataLayout()))
      Loops.push_back(L);
  bool Changed = false;
  for (Loop *L : Loops) {
    auto [Words, Math] = iterationWork(L, F.getDataLayout());
    if (Words == 0 || Words > AliasVersioningMaxWords ||
        Math > AliasVersioningMathPerWord * Words)
      continue;
    if (!L->isLoopSimplifyForm()) {
      Changed |= simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr,
                              /*PreserveLCSSA=*/false);
      if (!L->isLoopSimplifyForm())
        continue;
    }
    if (!L->getUniqueExitBlock())
      continue;
    Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
    LoopAccessInfo LAI(L, &SE, &TTI, &TLI, &AA, &DT, &LI, &AC);
    const RuntimePointerChecking *RPC = LAI.getRuntimePointerChecking();
    if (!LAI.canVectorizeMemory() || !RPC || !RPC->Need ||
        LAI.hasConvergentOp() ||
        !LAI.getPSE().getPredicate().isAlwaysTrue() ||
        LAI.getNumRuntimePointerChecks() == 0 ||
        LAI.getNumRuntimePointerChecks() > AliasVersioningMaxChecks)
      continue;
    LLVM_DEBUG(dbgs() << "agx-alias-version: " << F.getName() << " "
                      << L->getHeader()->getName() << " "
                      << LAI.getNumRuntimePointerChecks() << " checks\n");
    LoopVersioning LV(LAI, RPC->getChecks(), L, &LI, &DT, &SE);
    LV.versionLoop();
    LV.annotateLoopWithNoAlias();
    Changed = true;
  }
  if (Changed)
    SE.forgetAllLoops();
  return Changed;
}
