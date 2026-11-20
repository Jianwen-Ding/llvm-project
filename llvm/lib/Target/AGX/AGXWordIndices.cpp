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
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "agx-word-indices"

static cl::opt<bool> AGXWordIndicesOpt(
    "agx-loop-word-indices", cl::init(true), cl::Hidden,
    cl::desc("Index a loop's word accesses by a counter of words beside its "
             "counter of bytes or halves"));

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

bool AGXWordIndices::runOnFunction(Function &F) {
  if (!AGXWordIndicesOpt || skipFunction(F))
    return false;
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
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
  return Changed;
}
