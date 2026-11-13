//===-- AGXVectorAccesses.cpp - Vector loads and stores shaped for AGX ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two shapes of vector access cost AGX more than the scalar accesses they
// replace:
//
// * A vector load of which only some lanes are used (PTX `ld.v4` into `_`
//   registers) loads the lanes nobody reads and keeps its registers live.
//   Before the load/store vectorizer, it becomes a load of the used lanes
//   only, which the vectorizer can then merge with its neighbours.
//
// * A vector store needs its lanes in consecutive registers. A lane that is
//   a lane of a loaded vector sits where the load put it; when the store
//   wants the lanes of loads in another order (a swap, two loads), the
//   register allocator copies them, and a store right after an ALU
//   instruction waits for it. After the vectorizer, such a store is split
//   into the runs of lanes that need no copy.
//
// Before the vectorizer, accesses also get the alignment their neighbours
// prove: in a block, an access at base + c with alignment A that runs
// whenever another at base + d does makes base + d aligned to
// min(A, the largest power of two dividing d - c) (else the first would be
// undefined). The bytes after a 2-byte aligned f16 in a packed struct
// (llama.cpp's q4_0 blocks) are then 2-byte aligned pairs, one access each.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-vector-accesses"

static cl::opt<bool> AGXNarrowVectorLoads(
    "agx-narrow-vector-loads", cl::init(true), cl::Hidden,
    cl::desc("Load only the used lanes of a vector load"));

static cl::opt<bool> AGXNeighbourAlignment(
    "agx-neighbour-alignment", cl::init(true), cl::Hidden,
    cl::desc("Raise access alignments to what accesses of the same base that "
             "run with them prove"));

static cl::opt<bool> AGXWideLanesAs32(
    "agx-wide-lanes-as-32", cl::init(true), cl::Hidden,
    cl::desc("Load and store vectors of two 64-bit lanes as four 32-bit "
             "lanes"));

static cl::opt<bool> AGXSplitPermutedStores(
    "agx-split-permuted-stores", cl::init(true), cl::Hidden,
    cl::desc("Split vector stores whose lanes would need register copies"));

namespace {

class AGXVectorAccesses : public FunctionPass {
  bool BeforeVectorizer;

public:
  static char ID;
  AGXVectorAccesses(bool BeforeVectorizer = true)
      : FunctionPass(ID), BeforeVectorizer(BeforeVectorizer) {}

  StringRef getPassName() const override { return "AGX Vector Accesses"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

} // namespace

char AGXVectorAccesses::ID = 0;

INITIALIZE_PASS(AGXVectorAccesses, DEBUG_TYPE, "AGX Vector Accesses", false,
                false)

FunctionPass *llvm::createAGXVectorAccessesPass(bool BeforeVectorizer) {
  return new AGXVectorAccesses(BeforeVectorizer);
}

/// Loads only lanes [Lo, Hi] of a vector load read only by extractelements
/// with constant indices.
static bool narrowLoad(LoadInst &LI) {
  auto *VT = dyn_cast<FixedVectorType>(LI.getType());
  if (!VT || !LI.isSimple())
    return false;
  unsigned N = VT->getNumElements();
  unsigned Lo = N, Hi = 0;
  for (User *U : LI.users()) {
    auto *EE = dyn_cast<ExtractElementInst>(U);
    auto *Idx = EE ? dyn_cast<ConstantInt>(EE->getIndexOperand()) : nullptr;
    if (!Idx || Idx->getZExtValue() >= N)
      return false;
    unsigned I = Idx->getZExtValue();
    Lo = std::min(Lo, I);
    Hi = std::max(Hi, I);
  }
  if (Lo > Hi || (Lo == 0 && Hi == N - 1))
    return false;
  Type *Elt = VT->getElementType();
  const DataLayout &DL = LI.getDataLayout();
  uint64_t EltBytes = DL.getTypeStoreSize(Elt);
  if (DL.getTypeAllocSize(Elt) != EltBytes)
    return false;
  unsigned M = Hi - Lo + 1;
  IRBuilder<> B(&LI);
  Value *Ptr = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), LI.getPointerOperand(),
                                            Lo * EltBytes);
  Type *NewTy = M == 1 ? Elt : FixedVectorType::get(Elt, M);
  LoadInst *New = B.CreateAlignedLoad(
      NewTy, Ptr, commonAlignment(LI.getAlign(), Lo * EltBytes), LI.getName());
  New->copyMetadata(LI, {LLVMContext::MD_tbaa, LLVMContext::MD_alias_scope,
                         LLVMContext::MD_noalias, LLVMContext::MD_nontemporal,
                         LLVMContext::MD_invariant_load});
  for (User *U : make_early_inc_range(LI.users())) {
    auto *EE = cast<ExtractElementInst>(U);
    unsigned I = cast<ConstantInt>(EE->getIndexOperand())->getZExtValue();
    Value *V = M == 1 ? static_cast<Value *>(New)
                      : IRBuilder<>(EE).CreateExtractElement(New, I - Lo);
    V->takeName(EE);
    EE->replaceAllUsesWith(V);
    EE->eraseFromParent();
  }
  LI.eraseFromParent();
  return true;
}

namespace {
/// Where a stored lane comes from: lane Index of the vector load Load, or
/// anywhere (Load null: a value the register allocator can put in place).
struct LaneSource {
  const Value *Load = nullptr;
  int64_t Offset = 0; // load lane - store lane
};
} // namespace

static LaneSource laneSource(Value *V, unsigned Lane) {
  Value *Vec;
  uint64_t Idx;
  if (!match(V, m_ExtractElt(m_Value(Vec), m_ConstantInt(Idx))))
    return {};
  // The vectorizer reads merged vector loads through shuffles.
  while (auto *SV = dyn_cast<ShuffleVectorInst>(Vec)) {
    int M = SV->getMaskValue(Idx);
    unsigned NIn =
        cast<FixedVectorType>(SV->getOperand(0)->getType())->getNumElements();
    if (M < 0)
      return {};
    Vec = SV->getOperand(unsigned(M) < NIn ? 0 : 1);
    Idx = unsigned(M) % NIn;
  }
  if (isa<LoadInst>(Vec))
    return {Vec, int64_t(Idx) - int64_t(Lane)};
  return {};
}

/// Splits a store of an insertelement chain into the runs of lanes that are
/// lanes of the same vector load in the same order (or free values).
static bool splitStore(StoreInst &SI) {
  auto *VT = dyn_cast<FixedVectorType>(SI.getValueOperand()->getType());
  // Device stores only: a threadgroup store's copy is cheaper than the second
  // address register its split takes (bitonic sort's swap: +3.9%).
  if (!VT || !SI.isSimple() ||
      SI.getPointerAddressSpace() != AGX::DEVICE_ADDRESS)
    return false;
  unsigned N = VT->getNumElements();
  SmallVector<Value *, 4> Lanes(N, nullptr);
  Value *V = SI.getValueOperand();
  while (auto *IE = dyn_cast<InsertElementInst>(V)) {
    auto *Idx = dyn_cast<ConstantInt>(IE->getOperand(2));
    if (!Idx || Idx->getZExtValue() >= N)
      return false;
    Value *&Lane = Lanes[Idx->getZExtValue()];
    if (!Lane)
      Lane = IE->getOperand(1);
    V = IE->getOperand(0);
  }
  if (!isa<PoisonValue, UndefValue>(V) || is_contained(Lanes, nullptr))
    return false;

  // Runs: a lane joins the current run when its source agrees with the
  // run's (same load, same offset), or either is free.
  SmallVector<unsigned, 4> Starts{0};
  LaneSource Run = laneSource(Lanes[0], 0);
  for (unsigned I = 1; I < N; ++I) {
    LaneSource S = laneSource(Lanes[I], I);
    if (S.Load && Run.Load &&
        (S.Load != Run.Load || S.Offset != Run.Offset)) {
      Starts.push_back(I);
      Run = S;
    } else if (!Run.Load) {
      Run = S;
    }
  }
  if (Starts.size() == 1)
    return false;

  Type *Elt = VT->getElementType();
  const DataLayout &DL = SI.getDataLayout();
  uint64_t EltBytes = DL.getTypeStoreSize(Elt);
  if (DL.getTypeAllocSize(Elt) != EltBytes)
    return false;
  IRBuilder<> B(&SI);
  Starts.push_back(N);
  for (unsigned R = 0; R + 1 < Starts.size(); ++R) {
    unsigned Lo = Starts[R], M = Starts[R + 1] - Lo;
    Value *Val;
    if (M == 1) {
      Val = Lanes[Lo];
    } else {
      Val = PoisonValue::get(FixedVectorType::get(Elt, M));
      for (unsigned I = 0; I < M; ++I)
        Val = B.CreateInsertElement(Val, Lanes[Lo + I], I);
    }
    Value *Ptr = B.CreateConstInBoundsGEP1_64(
        B.getInt8Ty(), SI.getPointerOperand(), Lo * EltBytes);
    StoreInst *New = B.CreateAlignedStore(
        Val, Ptr, commonAlignment(SI.getAlign(), Lo * EltBytes));
    New->copyMetadata(SI, {LLVMContext::MD_tbaa, LLVMContext::MD_alias_scope,
                           LLVMContext::MD_noalias,
                           LLVMContext::MD_nontemporal});
  }
  Value *Old = SI.getValueOperand();
  SI.eraseFromParent();
  // The insertelement chain is dead now unless something else uses it.
  while (auto *IE = dyn_cast<InsertElementInst>(Old)) {
    if (!IE->use_empty())
      break;
    Old = IE->getOperand(0);
    IE->eraseFromParent();
  }
  return true;
}

/// The alignment accesses of the same base prove for the accesses of BB (see
/// the file comment). An access runs whenever a later one of the block does,
/// and whenever an earlier one does that no instruction between them may stop.
static bool raiseAlignments(BasicBlock &BB) {
  const DataLayout &DL = BB.getDataLayout();
  struct Anchor {
    int64_t Offset;
    Align A;
  };
  // the best-aligned access of each base so far (its alignment fixes the
  // base's residue modulo every smaller power of two)
  DenseMap<const Value *, Anchor> Known;
  bool Changed = false;
  auto It = BB.begin();
  while (It != BB.end()) {
    // a run of instructions that all run when its first does
    SmallVector<Instruction *, 16> Run;
    for (; It != BB.end(); ++It) {
      Run.push_back(&*It);
      if (!isGuaranteedToTransferExecutionToSuccessor(&*It)) {
        ++It;
        break;
      }
    }
    SmallVector<std::tuple<Instruction *, const Value *, int64_t>, 16> Mem;
    for (Instruction *I : Run) {
      Value *Ptr = getLoadStorePointerOperand(I);
      if (!Ptr)
        continue;
      int64_t Off = 0;
      const Value *Base = GetPointerBaseWithConstantOffset(Ptr, Off, DL);
      Align A = getLoadStoreAlignment(I);
      Mem.push_back({I, Base, Off});
      auto [Pos, New] = Known.try_emplace(Base, Anchor{Off, A});
      if (!New && A > Pos->second.A)
        Pos->second = Anchor{Off, A};
    }
    for (auto [I, Base, Off] : Mem) {
      const Anchor &K = Known.find(Base)->second;
      uint64_t Diff = static_cast<uint64_t>(Off - K.Offset);
      Align A = Diff ? std::min(K.A, Align(uint64_t(1) << llvm::countr_zero(Diff)))
                     : K.A;
      if (A <= getLoadStoreAlignment(I))
        continue;
      if (auto *LI = dyn_cast<LoadInst>(I))
        LI->setAlignment(A);
      else
        cast<StoreInst>(I)->setAlignment(A);
      Changed = true;
    }
  }
  return Changed;
}

/// A load or store of <2 x i64> or <2 x double> (the vectorizer's merge of
/// two 64-bit accesses, a struct copied as two i64) as <4 x i32>: the 32-bit
/// lane vectors of up to 16 bytes are one access at 4-byte alignment
/// (AGXISelLowering lowerVectorLoad / lowerVectorStore); a 64-bit lane
/// vector is not a legal type, and type legalization split it in two.
static bool wideLanesAs32(Instruction &I) {
  auto *LI = dyn_cast<LoadInst>(&I);
  auto *SI = dyn_cast<StoreInst>(&I);
  Type *T = LI ? LI->getType() : SI->getValueOperand()->getType();
  auto *VT = dyn_cast<FixedVectorType>(T);
  if (!VT || VT->getNumElements() != 2 ||
      VT->getElementType()->getScalarSizeInBits() != 64 ||
      !(VT->getElementType()->isIntegerTy() ||
        VT->getElementType()->isDoubleTy()))
    return false;
  if ((LI ? LI->getAlign() : SI->getAlign()) < Align(4) ||
      (LI ? LI->isVolatile() || !LI->isSimple()
          : SI->isVolatile() || !SI->isSimple()))
    return false;
  unsigned AS = LI ? LI->getPointerAddressSpace() : SI->getPointerAddressSpace();
  if (AS != AGX::DEVICE_ADDRESS)
    return false;
  IRBuilder<> B(&I);
  auto *V4 = FixedVectorType::get(B.getInt32Ty(), 4);
  if (LI) {
    LoadInst *N = B.CreateAlignedLoad(V4, LI->getPointerOperand(),
                                      LI->getAlign(), LI->getName());
    N->copyMetadata(*LI);
    LI->replaceAllUsesWith(B.CreateBitCast(N, VT));
  } else {
    StoreInst *N = B.CreateAlignedStore(
        B.CreateBitCast(SI->getValueOperand(), V4), SI->getPointerOperand(),
        SI->getAlign());
    N->copyMetadata(*SI);
  }
  I.eraseFromParent();
  return true;
}

bool AGXVectorAccesses::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  bool Changed = false;
  if (BeforeVectorizer && AGXNeighbourAlignment)
    for (BasicBlock &BB : F)
      Changed |= raiseAlignments(BB);
  // Collected first: rewriting one erases its users that follow it.
  SmallVector<Instruction *, 16> Accesses;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (BeforeVectorizer ? isa<LoadInst>(I) : isa<StoreInst>(I))
        Accesses.push_back(&I);
  for (Instruction *I : Accesses) {
    if (BeforeVectorizer && AGXNarrowVectorLoads)
      Changed |= narrowLoad(*cast<LoadInst>(I));
    else if (!BeforeVectorizer && AGXSplitPermutedStores)
      Changed |= splitStore(*cast<StoreInst>(I));
  }
  if (!BeforeVectorizer && AGXWideLanesAs32) {
    SmallVector<Instruction *, 16> Wide;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (isa<LoadInst>(I) || isa<StoreInst>(I))
          Wide.push_back(&I);
    for (Instruction *I : Wide)
      Changed |= wideLanesAs32(*I);
  }
  return Changed;
}
