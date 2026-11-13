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
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
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
static void findRaises(Loop &L, ScalarEvolution &SE,
                       SmallVectorImpl<Raise> &Raises,
                       SetVector<Value *> &Bases) {
  for (Instruction &I : *L.getHeader()) {
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
    // sign extension copies out first (dot_i8: 29 -> 33 GPRs, tput +6%).
    if (getLoadStoreType(&I)->getScalarSizeInBits() < 16)
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
  if (Raises.size() < AlignmentVersioningMin)
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

bool AGXAlignmentVersioning::runOnFunction(Function &F) {
  if (!EnableAlignmentVersioning || skipFunction(F))
    return false;
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
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
  for (Loop *L : Loops)
    Changed |= versionLoop(*L, F, LI, DT, SE);
  return Changed;
}
