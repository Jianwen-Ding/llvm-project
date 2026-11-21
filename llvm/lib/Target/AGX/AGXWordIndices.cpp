//===-- AGXWordIndices.cpp - Word indices for a loop's word accesses ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A device access of 4-byte words (a float, a vector of halves at 4-byte
// alignment) addressed in a loop by a 32-bit counter of bytes or halves,
//
//   %iv = phi i32 [ %start, %preheader ], [ %iv.next, %latch ]
//   %p  = getelementptr [2 x i8], ptr addrspace(1) %base, i64 (zext %iv + k)
//   %v  = load <8 x half>, ptr addrspace(1) %p, align 4
//
// is base + 2 * zext(...): the word format's index counts 4 bytes, so
// instruction selection computes the 64-bit address with a 64-bit add (on
// the core half's shared unit, 3.92 cycles: a row sum of halves spent its
// throughput on them, L15-E594). A counter of words beside it,
//
//   %w = phi i32 [ %start >> 1, %preheader ], [ %w + step / 2, %latch ]
//   %p = getelementptr i32, ptr addrspace(1) %base, i64 (zext %w + k / 2)
//
// is the access's index register, a 32-bit add for each further offset.
// Where the address adds a constant to such a gep (LSR's base + iv - 16),
// the constant goes into a base computed before the loop.
//
// Only where scalar evolution proves the 32-bit index does not wrap, and
// the start, step and offsets are known multiples of the word.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-word-indices"

static cl::opt<bool> AGXWordIndicesOpt(
    "agx-loop-word-indices", cl::init(true), cl::Hidden,
    cl::desc("Index a loop's word accesses by a counter of words beside its "
             "counter of bytes or halves"));

static cl::opt<bool> AGXWordIndicesExit(
    "agx-loop-word-indices-exit", cl::init(true), cl::Hidden,
    cl::desc("The loop's exit compare of a counter whose accesses all moved "
             "to the counter of words, on that counter"));

static cl::opt<bool> AGXWordIndicesExitCount(
    "agx-loop-word-indices-exit-count", cl::init(true), cl::Hidden,
    cl::desc("A loop's count of iterations from 0 that only its exit "
             "compare with a loop-invariant bound reads, as the counter of "
             "words against that bound in words"));

static cl::opt<bool> AGXExitOnStepCounter(
    "agx-loop-exit-on-step-counter", cl::init(true), cl::Hidden,
    cl::desc("-agx-loop-word-indices-exit-count against any other counter "
             "of the loop by a power of two, not only a counter of words"));

static cl::opt<bool> AGXExitCountUsesAfter(
    "agx-loop-exit-count-uses-after", cl::init(true), cl::Hidden,
    cl::desc("-agx-loop-word-indices-exit-count where the count is also read "
             "after the exit it equals the bound at (taking the bound)"));

static cl::opt<unsigned> AGXWordIndicesOffsets(
    "agx-loop-word-indices-offsets", cl::init(1), cl::Hidden,
    cl::desc("Most further offsets (an add each) for each base"));

namespace {

class AGXWordIndices : public FunctionPass {
public:
  static char ID;
  AGXWordIndices() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Word Indices"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};

/// An access's address: base + s * zext(phi + k) (+ a byte offset).
struct Address {
  Instruction *Access;
  Value *Base;
  PHINode *Phi;
  int64_t K;      // in s units, inside the 32-bit index
  int64_t Offset; // bytes, outside it
  unsigned Size;  // s: 1 or 2
};

} // namespace

char AGXWordIndices::ID = 0;

INITIALIZE_PASS_BEGIN(AGXWordIndices, DEBUG_TYPE, "AGX Word Indices", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AGXWordIndices, DEBUG_TYPE, "AGX Word Indices", false,
                    false)

FunctionPass *llvm::createAGXWordIndicesPass() { return new AGXWordIndices(); }

/// The loop's step of a header phi: phi + C on the latch edge.
static std::optional<int64_t> phiStep(PHINode *Phi, Loop &L) {
  BasicBlock *Pre = L.getLoopPreheader(), *Latch = L.getLoopLatch();
  if (!Pre || !Latch || Phi->getParent() != L.getHeader() ||
      Phi->getNumIncomingValues() != 2 || !Phi->getType()->isIntegerTy(32))
    return std::nullopt;
  const APInt *C;
  if (!match(Phi->getIncomingValueForBlock(Latch),
             m_c_Add(m_Specific(Phi), m_APInt(C))))
    return std::nullopt;
  return C->getSExtValue();
}

/// The address of a load or store of 4-byte aligned words.
static std::optional<Address> analyze(Instruction &I, Loop &L,
                                      const DataLayout &DL) {
  if (!isa<LoadInst>(I) && !isa<StoreInst>(I))
    return std::nullopt;
  Value *Ptr = getLoadStorePointerOperand(&I);
  Type *Ty = getLoadStoreType(&I);
  if (!Ptr || cast<PointerType>(Ptr->getType())->getAddressSpace() !=
                  AGX::DEVICE_ADDRESS ||
      getLoadStoreAlignment(&I) < Align(4) ||
      DL.getTypeStoreSize(Ty) % 4 != 0)
    return std::nullopt;
  if (auto *LI = dyn_cast<LoadInst>(&I); LI && !LI->isSimple())
    return std::nullopt;
  // Only one access of 4-byte words: a value of 4-byte elements up to 16
  // bytes, 2 or 4 halves (a word each pair), 8 halves where they are one
  // 16-byte access (L15-E585, E591), 4 bytes. Type legalization's pieces
  // at offsets would each add their offset to the address in 64 bits
  // (gemv_f16's 8 halves in two 8-byte loads: tput +32%).
  {
    unsigned Bytes = DL.getTypeStoreSize(Ty);
    unsigned Elem = DL.getTypeStoreSize(Ty->getScalarType());
    bool One = Bytes <= 16 && (Elem >= 4 || (Elem == 2 && Bytes <= 8) ||
                               (Elem == 1 && Bytes == 4));
    if (Elem == 2 && Bytes == 16)
      One = isa<StoreInst>(I) ? agxWideHalfStores()
                              : agxWideHalfLoadBlock(*I.getParent());
    if (Elem == 1 && Bytes == 16 && isa<LoadInst>(I))
      One = agxWideByteLoadBlock(*I.getParent());
    if (!One)
      return std::nullopt;
  }
  if (auto *SI = dyn_cast<StoreInst>(&I); SI && !SI->isSimple())
    return std::nullopt;
  Address A{&I, nullptr, nullptr, 0, 0, 0};
  auto *G = dyn_cast<GetElementPtrInst>(Ptr);
  // (a byte offset on such a gep: LSR's base + iv - 16)
  if (G && G->getNumIndices() == 1 &&
      DL.getTypeAllocSize(G->getSourceElementType()) == 1)
    if (auto *C = dyn_cast<ConstantInt>(G->getOperand(1)))
      if (auto *Inner = dyn_cast<GetElementPtrInst>(G->getPointerOperand())) {
        A.Offset = C->getSExtValue();
        G = Inner;
      }
  if (!G || G->getNumIndices() != 1)
    return std::nullopt;
  uint64_t S = DL.getTypeAllocSize(G->getSourceElementType());
  if (S != 1 && S != 2)
    return std::nullopt;
  // (accesses instruction selection already indexes in those units: 8 bytes
  // at a half index in the 16-bit format's xyzw (SelectDeviceAddr32Via16),
  // 4 bytes at a byte index in the byte format's: hscale_dyn's <4 x half>
  // gained only a second counter, +9%)
  unsigned Bytes = DL.getTypeStoreSize(Ty);
  if (!A.Offset && ((S == 2 && Bytes == 8) ||
                    (S == 1 && Bytes == 4 &&
                     DL.getTypeStoreSize(Ty->getScalarType()) == 1)))
    return std::nullopt;
  A.Size = S;
  A.Base = G->getPointerOperand();
  if (!L.isLoopInvariant(A.Base))
    return std::nullopt;
  Value *X;
  if (!match(G->getOperand(1), m_ZExt(m_Value(X))) ||
      !X->getType()->isIntegerTy(32))
    return std::nullopt;
  const APInt *K;
  if (match(X, m_Add(m_Value(X), m_APInt(K))))
    A.K = K->getSExtValue();
  A.Phi = dyn_cast<PHINode>(X);
  if (!A.Phi || !phiStep(A.Phi, L))
    return std::nullopt;
  return A;
}

static cl::opt<bool> AGXHalfPointerCounters(
    "agx-half-pointer-counters", cl::init(true), cl::Hidden,
    cl::desc("A pointer LSR steps by twice a 32-bit counter of halves from "
             "0 as the base plus the counter's halves, its 16-byte half "
             "loads as two 8-byte ones from bases 8 bytes apart"));

/// LSR steps a row of halves by pointers beside a 32-bit counter of halves
/// (hdot_dyn: two rows' pointers by 16 bytes, j by 8): each pointer a
/// 64-bit add an iteration. Where the counter j goes up from 0 by t without
/// unsigned wrap (nuw) and the pointer p by 2 t bytes, p = p0 + 2 zext(j),
/// the base and an index of halves, which instruction selection takes as a
/// half index of the 16-bit format for 8 bytes (SelectDeviceAddr32Via16).
/// A load of 16 bytes there (8 halves, split by type legalization where
/// the block takes no 16-byte half loads, agxWideHalfLoadBlock, each
/// half's address a 64-bit add) as two of 8 bytes from p0 and p0 + 8, both
/// at j (L15-E688). Only where the pointer's every use is a load or store
/// at it (or its step).
static bool halfPointerCounters(Function &F, LoopInfo &LI) {
  using namespace PatternMatch;
  if (!AGXHalfPointerCounters)
    return false;
  bool Changed = false;
  Type *I64 = Type::getInt64Ty(F.getContext());
  for (Loop *L : LI.getLoopsInPreorder()) {
    BasicBlock *Pre = L->getLoopPreheader(), *Latch = L->getLoopLatch();
    if (!Pre || !Latch || !L->isInnermost())
      continue;
    SmallVector<PHINode *, 4> Ptrs;
    for (PHINode &P : L->getHeader()->phis())
      if (P.getType()->isPointerTy() &&
          P.getType()->getPointerAddressSpace() == AGX::DEVICE_ADDRESS &&
          P.getNumIncomingValues() == 2)
        Ptrs.push_back(&P);
    for (PHINode *P : Ptrs) {
      auto *G = dyn_cast<GetElementPtrInst>(P->getIncomingValueForBlock(Latch));
      const APInt *S;
      if (!G || G->getPointerOperand() != P || G->getNumIndices() != 1 ||
          !G->getSourceElementType()->isIntegerTy(8) ||
          !match(G->getOperand(1), m_APInt(S)) || !G->hasOneUse() ||
          S->isNegative() || S->isZero() || S->getZExtValue() % 2)
        continue;
      Value *P0 = P->getIncomingValueForBlock(Pre);
      if (!L->isLoopInvariant(P0))
        continue;
      // the counter of halves: from 0 by S / 2, without unsigned wrap
      PHINode *J = nullptr;
      for (PHINode &Q : L->getHeader()->phis()) {
        if (!Q.getType()->isIntegerTy(32) || Q.getNumIncomingValues() != 2 ||
            !match(Q.getIncomingValueForBlock(Pre), m_Zero()))
          continue;
        auto *Inc = dyn_cast<BinaryOperator>(Q.getIncomingValueForBlock(Latch));
        const APInt *T;
        if (Inc && Inc->getOpcode() == Instruction::Add &&
            Inc->hasNoUnsignedWrap() &&
            match(Inc, m_c_Add(m_Specific(&Q), m_APInt(T))) &&
            T->getZExtValue() * 2 == S->getZExtValue())
          J = &Q;
      }
      if (!J)
        continue;
      bool OK = true;
      for (User *U : P->users()) {
        if (U == G)
          continue;
        auto *I = dyn_cast<Instruction>(U);
        if (!I || !L->contains(I) || getLoadStorePointerOperand(I) != P ||
            (isa<LoadInst>(I) && !cast<LoadInst>(I)->isSimple()) ||
            (isa<StoreInst>(I) && !cast<StoreInst>(I)->isSimple()))
          OK = false;
      }
      if (!OK)
        continue;
      IRBuilder<> HB(&*L->getHeader()->getFirstInsertionPt());
      Value *Idx = HB.CreateZExt(J, I64);
      Type *H = ArrayType::get(HB.getInt8Ty(), 2);
      Value *NP = HB.CreateGEP(H, P0, Idx, P->getName() + ".halves");
      Value *Base8 = nullptr;
      for (User *U : make_early_inc_range(P->users())) {
        auto *LD = dyn_cast<LoadInst>(U);
        auto *VT = LD ? dyn_cast<FixedVectorType>(LD->getType()) : nullptr;
        if (VT && VT->getElementType()->isHalfTy() &&
            VT->getNumElements() == 8 && LD->getAlign() >= Align(4) &&
            !agxWideHalfLoadBlock(*LD->getParent())) {
          if (!Base8)
            Base8 = IRBuilder<>(Pre->getTerminator())
                        .CreateGEP(HB.getInt8Ty(), P0, HB.getInt64(8),
                                   P->getName() + ".base8");
          IRBuilder<> LB(LD);
          auto *V4 = FixedVectorType::get(VT->getElementType(), 4);
          auto *L0 = LB.CreateAlignedLoad(V4, NP, LD->getAlign());
          Value *NP8 = LB.CreateGEP(H, Base8, Idx);
          auto *L1 = LB.CreateAlignedLoad(
              V4, NP8, commonAlignment(LD->getAlign(), 8));
          for (LoadInst *Part : {L0, L1})
            Part->copyMetadata(*LD, {LLVMContext::MD_alias_scope,
                                     LLVMContext::MD_noalias,
                                     LLVMContext::MD_tbaa});
          Value *R = LB.CreateShuffleVector(L0, L1,
                                            ArrayRef<int>{0, 1, 2, 3, 4, 5, 6, 7});
          R->takeName(LD);
          LD->replaceAllUsesWith(R);
          LD->eraseFromParent();
          continue;
        }
      }
      P->replaceUsesWithIf(NP, [&](Use &U) { return U.getUser() != G; });
      G->replaceAllUsesWith(PoisonValue::get(G->getType()));
      G->eraseFromParent();
      P->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

bool AGXWordIndices::runOnFunction(Function &F) {
  if (!AGXWordIndicesOpt || skipFunction(F))
    return false;
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  bool HalfPtrs = halfPointerCounters(F, LI);
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  AssumptionCache &AC =
      getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  TargetLibraryInfo &TLI =
      getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  const DataLayout &DL = F.getDataLayout();
  Type *I32 = Type::getInt32Ty(F.getContext());
  Type *I64 = Type::getInt64Ty(F.getContext());
  // the accesses whose index can count words, by loop, counter and size
  struct Group {
    Loop *L;
    PHINode *Phi;
    unsigned Size;
    SmallVector<Address, 8> Accesses;
  };
  SmallVector<Group, 4> Groups;
  // (the thread positions' bounds, the kernel's upper bounds too, for the
  // analysis only, in a ScalarEvolution of this pass's own: a row's start
  // 512 * r, taken off again)
  SmallVector<WeakVH, 8> Ranged;
  AGX::addSpecialRegisterRanges(F, &Ranged, /*ForAnalysis=*/true);
  {
    ScalarEvolution SE(F, TLI, AC, DT, LI);
    for (Loop *L : LI.getLoopsInPreorder()) {
      BasicBlock *Pre = L->getLoopPreheader(), *Latch = L->getLoopLatch();
      if (!Pre || !Latch)
        continue;
      MapVector<std::pair<PHINode *, unsigned>, SmallVector<Address, 8>> ByPhi;
      for (BasicBlock *BB : L->blocks()) {
        if (LI.getLoopFor(BB) != L)
          continue;
        for (Instruction &I : *BB) {
          std::optional<Address> A = analyze(I, *L, DL);
          if (!A)
            continue;
          unsigned Shift = A->Size == 1 ? 2 : 1;
          int64_t Step = *phiStep(A->Phi, *L);
          Value *Start = A->Phi->getIncomingValueForBlock(Pre);
          if ((Step * int64_t(A->Size)) % 4 != 0 ||
              (A->K * int64_t(A->Size)) % 4 != 0 ||
              computeKnownBits(Start, DL).countMinTrailingZeros() < Shift)
            continue;
          // (the 32-bit indices phi and phi + k without unsigned wrap: their
          // zero extensions add recurrences of the loop)
          auto NoWrap = [&](const SCEV *X) {
            auto *AR =
                dyn_cast<SCEVAddRecExpr>(SE.getZeroExtendExpr(X, I64));
            return AR && AR->getLoop() == L;
          };
          const SCEV *Idx = SE.getSCEV(A->Phi);
          bool Proven = NoWrap(Idx) &&
                        (!A->K || NoWrap(SE.getAddExpr(
                                      Idx, SE.getConstant(I32, A->K))));
          // (or a counter of the same step that does not wrap signed, LSR's
          // offset from it invariant: phi + k = q + (phi - q) + k within
          // [0, 2^32) for every q in its range: a runtime row's)
          if (!Proven)
            for (PHINode &Q : L->getHeader()->phis()) {
              if (&Q == A->Phi || !Q.getType()->isIntegerTy(32) ||
                  phiStep(&Q, *L) != Step)
                continue;
              auto *QR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&Q));
              if (!QR || QR->getLoop() != L || !QR->hasNoSignedWrap())
                continue;
              const SCEV *Diff = SE.getMinusSCEV(Idx, QR);
              if (!SE.isLoopInvariant(Diff, L))
                continue;
              ConstantRange QRange = SE.getSignedRange(QR);
              ConstantRange DRange = SE.getSignedRange(Diff);
              if (QRange.getSignedMin().isNegative())
                continue;
              int64_t Lo = QRange.getSignedMin().getSExtValue() +
                           DRange.getSignedMin().getSExtValue() +
                           std::min<int64_t>(A->K, 0);
              int64_t Hi = QRange.getSignedMax().getSExtValue() +
                           DRange.getSignedMax().getSExtValue() +
                           std::max<int64_t>(A->K, 0);
              if (Lo >= 0 && Hi < (int64_t(1) << 32)) {
                Proven = true;
                break;
              }
            }
          if (Proven)
            ByPhi[{A->Phi, A->Size}].push_back(*A);
        }
      }
      for (auto &[Key, Accesses] : ByPhi)
        Groups.push_back({L, Key.first, Key.second, std::move(Accesses)});
    }
  }
  for (WeakVH &V : Ranged)
    if (auto *CI = dyn_cast_or_null<CallBase>(V))
      CI->removeRetAttr(Attribute::Range);

  bool Changed = false;
  SmallVector<WeakTrackingVH, 16> Dead;
  // (the old counter, the counter of words and its next value, the shift)
  struct Counter {
    PHINode *Phi;
    PHINode *W;
    Value *WNext;
    unsigned Shift;
  };
  SmallVector<Counter, 4> Counters;
  for (Group &G : Groups) {
    Loop *L = G.L;
    BasicBlock *Pre = L->getLoopPreheader(), *Latch = L->getLoopLatch();
    unsigned Shift = G.Size == 1 ? 2 : 1;
    int64_t Step = *phiStep(G.Phi, *L);
    // Each base at its accesses' smallest byte offset s * k + offset (one
    // base computed before the loop where that is not 0), the rest of each
    // offset words added to the counter: base + s * (phi + k) + offset =
    // (base + min) + 4 * (w + (s * k + offset - min) / 4). (A base for
    // each offset took registers: layernorm's 38 -> 128.) That index does
    // not wrap: w = s * phi / 4 < 2^31.
    MapVector<Value *, int64_t> MinOff;
    bool Words = true;
    for (Address &A : G.Accesses) {
      int64_t T = A.K * int64_t(G.Size) + A.Offset;
      auto [It, New] = MinOff.insert({A.Base, T});
      if (!New)
        It->second = std::min(It->second, T);
    }
    SmallSet<int64_t, 8> Offsets;
    for (Address &A : G.Accesses) {
      int64_t D = A.K * int64_t(G.Size) + A.Offset - MinOff[A.Base];
      Words &= D % 4 == 0;
      if (D)
        Offsets.insert(D);
    }
    // (an add for each further offset, against a 64-bit add for each base:
    // a run of 16 loads from one base kept its immediates, nested2d's
    // fifteen adds between them doubled its cycles)
    Words &= Offsets.size() <= AGXWordIndicesOffsets * MinOff.size();
    if (!Words)
      continue;
    IRBuilder<> B(Pre->getTerminator());
    Value *Start = G.Phi->getIncomingValueForBlock(Pre);
    Value *WStart =
        B.CreateLShr(Start, Shift, "words.start", /*isExact=*/true);
    PHINode *W = PHINode::Create(I32, 2, "words", L->getHeader()->begin());
    IRBuilder<> LB(Latch->getTerminator());
    Value *WNext = LB.CreateAdd(W, B.getInt32(Step * int64_t(G.Size) / 4),
                                "words.next", /*HasNUW=*/true);
    W->addIncoming(WStart, Pre);
    W->addIncoming(WNext, Latch);
    Counters.push_back({G.Phi, W, WNext, Shift});
    DenseMap<Value *, Value *> Bases;
    for (auto &[Base, Min] : MinOff)
      Bases[Base] = Min ? B.CreateGEP(B.getInt8Ty(), Base, B.getInt64(Min),
                                      "words.base")
                        : Base;
    DenseMap<int64_t, Value *> Indices;
    for (Address &A : G.Accesses) {
      int64_t D = (A.K * int64_t(G.Size) + A.Offset - MinOff[A.Base]) / 4;
      IRBuilder<> IB(A.Access);
      Value *Idx = W;
      if (D) {
        // (one add for each distinct offset, at the first access in the
        // header that needs it, or at the access)
        Value *&V = Indices[D];
        if (!V || !isa<Instruction>(V) ||
            cast<Instruction>(V)->getParent() != A.Access->getParent() ||
            !cast<Instruction>(V)->comesBefore(A.Access))
          V = IB.CreateAdd(W, IB.getInt32(D), "", /*HasNUW=*/true);
        Idx = V;
      }
      Value *Ptr = IB.CreateGEP(I32, Bases[A.Base], IB.CreateZExt(Idx, I64),
                                "", GEPNoWrapFlags::noUnsignedWrap());
      Value *Old = getLoadStorePointerOperand(A.Access);
      A.Access->setOperand(isa<LoadInst>(A.Access) ? 0 : 1, Ptr);
      Dead.push_back(Old);
      Changed = true;
    }
  }
  for (WeakTrackingVH &V : Dead)
    if (auto *I = dyn_cast_or_null<Instruction>(V))
      RecursivelyDeleteTriviallyDeadInstructions(I);
  // A counter left with only its increment and the loop's exit compare
  // (row_norm_apply's bytes beside its words: a register): phi = w << shift
  // exactly, so phi == c is w == c >> shift for a c of the shift's low zero
  // bits (L15-E623).
  SmallPtrSet<PHINode *, 4> Done;
  SmallVector<WeakTrackingVH, 4> Left;
  for (Counter &C : Counters)
    Left.push_back(C.Phi);
  for (Counter &C : AGXWordIndicesExit ? Counters : SmallVector<Counter, 4>()) {
    if (!Done.insert(C.Phi).second)
      continue;
    PHINode *P = C.Phi;
    if (P->getNumIncomingValues() != 2)
      continue;
    // (its increment no-wrap, so that inc = w.next << shift exactly)
    auto *Inc = dyn_cast<BinaryOperator>(
        P->getIncomingValueForBlock(C.W->getIncomingBlock(1)));
    if (!Inc || Inc->getOpcode() != Instruction::Add ||
        !Inc->hasNoUnsignedWrap() || !is_contained(Inc->operands(), P) ||
        !P->hasOneUse())
      continue;
    // Inc's users: P and one compare, directly or through a zext.
    ICmpInst *Cmp = nullptr;
    Instruction *Ext = nullptr;
    bool Ok = true;
    for (User *U : Inc->users()) {
      if (U == P)
        continue;
      Instruction *UI = cast<Instruction>(U);
      if (isa<ZExtInst>(UI) && UI->hasOneUse() && !Ext) {
        Ext = UI;
        UI = cast<Instruction>(*UI->user_begin());
      }
      auto *IC = dyn_cast<ICmpInst>(UI);
      if (!IC || Cmp || !IC->isEquality() ||
          !isa<ConstantInt>(IC->getOperand(1))) {
        Ok = false;
        break;
      }
      Cmp = IC;
    }
    if (!Ok || !Cmp)
      continue;
    const APInt &CV = cast<ConstantInt>(Cmp->getOperand(1))->getValue();
    if (CV.getActiveBits() > 32 || CV.countr_zero() < C.Shift)
      continue;
    // (w.next, at the latch's end, before the compare: it reads only w)
    auto *WN = dyn_cast<Instruction>(C.WNext);
    if (!WN || WN->getParent() != Cmp->getParent())
      continue;
    if (Cmp->comesBefore(WN))
      WN->moveBefore(Cmp->getIterator());
    IRBuilder<> CB(Cmp);
    Cmp->setOperand(0, C.WNext);
    Cmp->setOperand(1, CB.getInt32(CV.getZExtValue() >> C.Shift));
    if (Ext)
      Ext->eraseFromParent();
    P->replaceAllUsesWith(PoisonValue::get(P->getType()));
    P->eraseFromParent();
    Inc->eraseFromParent();
  }
  // A count from 0 by a power of two s_p that only the exit compare with a
  // loop-invariant n reads (an unrolled loop's against its iterations of a
  // runtime row, hsum_dyn): with n's low bits zero (a multiple of s_p) its
  // n / s_p-th increment is the one equal to n, and the counter of words is
  // then w.start + n / s_p * s_w, which no earlier one equals mod 2^32 where
  // n's bound / s_p * s_w < 2^32 (L15-E624).
  // (the counters to compare instead: the counters of words, then any
  // other loop's counter by a positive power of two, its add in the latch,
  // L15-E625)
  struct Stepped {
    WeakVH W, WN;
    BasicBlock *Pre, *Latch;
  };
  SmallVector<Stepped, 8> Steps;
  for (Counter &C : Counters)
    if (auto *WN = dyn_cast<BinaryOperator>(C.WNext))
      Steps.push_back({C.W, WN, C.W->getIncomingBlock(0),
                       C.W->getIncomingBlock(1)});
  if (AGXExitOnStepCounter)
    for (Loop *L : LI.getLoopsInPreorder()) {
      BasicBlock *Pre = L->getLoopPreheader(), *Latch = L->getLoopLatch();
      if (!Pre || !Latch)
        continue;
      for (PHINode &Q : L->getHeader()->phis()) {
        if (!Q.getType()->isIntegerTy(32) || Q.getNumIncomingValues() != 2 ||
            any_of(Steps, [&](const Stepped &T) { return T.W == &Q; }))
          continue;
        auto *QN = dyn_cast<BinaryOperator>(Q.getIncomingValueForBlock(Latch));
        const APInt *SQ;
        if (QN && QN->getParent() == Latch &&
            match(QN, m_Add(m_Specific(&Q), m_APInt(SQ))) &&
            SQ->isPowerOf2() && !SQ->isNegative())
          Steps.push_back({&Q, QN, Pre, Latch});
      }
    }
  if (AGXWordIndicesExitCount)
    for (Stepped &S : Steps) {
      auto *W = dyn_cast_or_null<PHINode>(S.W);
      auto *WN = dyn_cast_or_null<BinaryOperator>(S.WN);
      if (!W || !WN)
        continue;
      BasicBlock *H = W->getParent();
      BasicBlock *Latch = S.Latch;
      BasicBlock *Pre = S.Pre;
      auto *SW = dyn_cast<ConstantInt>(WN->getOperand(1));
      if (!SW || !SW->getValue().isPowerOf2())
        continue;
      for (PHINode &P : make_early_inc_range(H->phis())) {
        if (&P == W || !P.getType()->isIntegerTy(32) ||
            P.getNumIncomingValues() != 2 ||
            !match(P.getIncomingValueForBlock(Pre), m_Zero()))
          continue;
        auto *Inc =
            dyn_cast<BinaryOperator>(P.getIncomingValueForBlock(Latch));
        const APInt *SP;
        if (!Inc || !P.hasOneUse() ||
            !match(Inc, m_NUWAdd(m_Specific(&P), m_APInt(SP))) ||
            !SP->isPowerOf2())
          continue;
        // Inc's users: P, the compare (directly or through a zext) and
        // phis after the loop on the edge it takes when inc = n (there n).
        ICmpInst *Cmp = nullptr;
        Instruction *Ext = nullptr;
        SmallVector<Use *, 2> Outside;
        bool Ok = true;
        auto Look = [&](Value *V) {
          for (Use &U : V->uses()) {
            auto *UI = cast<Instruction>(U.getUser());
            if (UI == &P)
              continue;
            if (isa<ZExtInst>(UI) && V == Inc && !Ext) {
              Ext = UI;
              continue;
            }
            if (auto *IC = dyn_cast<ICmpInst>(UI);
                IC && !Cmp && IC->isEquality()) {
              Cmp = IC;
              continue;
            }
            // (any use outside the loop: checked below to be after the
            // exit taken when inc = n)
            if (UI->getParent() != H && UI->getParent() != Latch &&
                (!AGXExitCountUsesAfter ? isa<PHINode>(UI)
                                        : !LI.getLoopFor(H)->contains(UI))) {
              Outside.push_back(&U);
              continue;
            }
            Ok = false;
          }
        };
        Look(Inc);
        if (Ext)
          Look(Ext);
        auto *Br = dyn_cast<CondBrInst>(Latch->getTerminator());
        if (!Ok || !Cmp || Cmp->getParent() != Latch || !Br ||
            Br->getCondition() != Cmp ||
            !Cmp->hasOneUse())
          continue;
        BasicBlock *Equal = Br->getSuccessor(
            Cmp->getPredicate() == ICmpInst::ICMP_EQ ? 0 : 1);
        // (a phi of the exit edge itself, or a use that only the exit
        // block, entered from the latch alone, reaches: an unrolled loop's
        // remainder starting at the count, L15-E626)
        if (Equal == H ||
            any_of(Outside, [&](Use *U) {
              auto *UI = cast<Instruction>(U->getUser());
              if (auto *Phi = dyn_cast<PHINode>(UI);
                  Phi && Phi->getParent() == Equal &&
                  Phi->getIncomingBlock(*U) == Latch)
                return false;
              if (!AGXExitCountUsesAfter ||
                  Equal->getSinglePredecessor() != Latch ||
                  LI.getLoopFor(H)->getExitingBlock() != Latch)
                return true;
              BasicBlock *At = isa<PHINode>(UI)
                                   ? cast<PHINode>(UI)->getIncomingBlock(*U)
                                   : UI->getParent();
              return !DT.dominates(Equal, At);
            }))
          continue;
        Value *Mine = Ext ? static_cast<Value *>(Ext) : Inc;
        Value *N = Cmp->getOperand(0) == Mine ? Cmp->getOperand(1)
                                              : Cmp->getOperand(0);
        if (N == Mine)
          continue;
        // n as a 32-bit value (or the zext of one), invariant in the loop
        // (or a 64-bit one below 2^32 by its known bits, truncated before
        // the loop: row_axpy_dyn's unrolled iterations, n & 0x7ffffff8)
        Value *N32 = N;
        bool Trunc = false;
        if (N->getType()->isIntegerTy(64)) {
          Value *X;
          if (match(N, m_ZExt(m_Value(X))) && X->getType()->isIntegerTy(32))
            N32 = X;
          else if (computeKnownBits(N, DL).countMaxActiveBits() <= 32)
            Trunc = true;
          else
            continue;
        } else if (!N->getType()->isIntegerTy(32)) {
          continue;
        }
        auto *NI = dyn_cast<Instruction>(Trunc ? N : N32);
        if (NI && !DT.dominates(NI, Pre->getTerminator()))
          continue;
        if (!NI && !isa<Argument>(Trunc ? N : N32) &&
            !isa<Constant>(Trunc ? N : N32))
          continue;
        if (Trunc) {
          // (the bits of n below: as for the 32-bit value)
          KnownBits K64 = computeKnownBits(N, DL);
          unsigned LP = SP->logBase2(), LW = SW->getValue().logBase2();
          if (K64.countMinTrailingZeros() < LP ||
              K64.getMaxValue().lshr(LP).shl(LW).getActiveBits() > 32)
            continue;
          N32 = IRBuilder<>(Pre->getTerminator())
                    .CreateTrunc(N, Type::getInt32Ty(F.getContext()));
        }
        KnownBits KB = computeKnownBits(N32, DL);
        unsigned LogP = SP->logBase2(), LogW = SW->getValue().logBase2();
        if (KB.countMinTrailingZeros() < LogP ||
            KB.getMaxValue().lshr(LogP).zext(64).shl(LogW).getActiveBits() >
                32)
          continue;
        IRBuilder<> PB(Pre->getTerminator());
        Value *Bound = PB.CreateAdd(
            W->getIncomingValueForBlock(Pre),
            PB.CreateShl(PB.CreateLShr(N32, LogP, "", /*isExact=*/true), LogW),
            any_of(Counters, [&](const Counter &K) { return K.W == W; })
                ? "words.end"
                : "count.end");
        if (Cmp->comesBefore(WN))
          WN->moveBefore(Cmp->getIterator());
        // (w.next may now be compared on the last iteration, where its
        // nuw is not otherwise known: dropped)
        WN->setHasNoUnsignedWrap(false);
        WN->setHasNoSignedWrap(false);
        Cmp->setOperand(0, WN);
        Cmp->setOperand(1, Bound);
        for (Use *U : Outside)
          U->set(U->get() == Inc ? N32 : N);
        if (Ext && Ext->use_empty())
          Ext->eraseFromParent();
        P.replaceAllUsesWith(PoisonValue::get(P.getType()));
        P.eraseFromParent();
        Inc->eraseFromParent();
        Changed = true;
        break;
      }
    }
  // A counter now read by nothing but its own increment (hscale_dyn's
  // halves beside its words and its count of iterations): nothing later
  // deletes the cycle, which kept an add and a register.
  if (AGXWordIndicesExit)
    for (WeakTrackingVH &V : Left)
      if (auto *P = dyn_cast_or_null<PHINode>(V))
        RecursivelyDeleteDeadPHINode(P);
  return Changed || HalfPtrs;
}
