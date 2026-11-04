//===-- AGXExpandWideAtomics.cpp - 64- and 128-bit atomics for AGX --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The hardware atomics are 32-bit. Atomic operations on wider values (64-bit
// and 128-bit loads, stores, read-modify-write operations and compare and
// swaps) take a lock from a table of device memory locks, chosen by the
// address of the 16-byte block they access, and access the value with 32-bit
// atomics, between fences, while they hold it:
//
//   loop:                          ; per thread, until it has done its work
//     if (cmpxchg(lock, 0, 1) succeeds) {
//       fence; read; compute; write; fence
//       xchg(lock, 0)
//       done
//     }
//
// The critical section is inside the loop, so a thread that holds a lock
// releases it in the same iteration, before the other threads of its
// SIMD-group, which may wait for the same lock, try again.
//
// A read-modify-write or compare-and-swap whose SIMD group's active threads
// all access the same address (an atomic counter or accumulator) takes the
// lock once for the group instead of once per thread, one after another: the
// lowest thread takes it and reads the value, every thread applies the
// operations of all of them in lane order (a serialization of the group's
// atomics) and keeps the value before its own, and the lowest thread writes
// the result and frees the lock.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/LowerAtomic.h"

using namespace llvm;

#define DEBUG_TYPE "agx-expand-wide-atomics"

static constexpr unsigned NumLocks = 1024;

static cl::opt<bool> UniformBatch(
    "agx-wide-atomics-batch", cl::init(true), cl::Hidden,
    cl::desc("Read-modify-write and compare-and-swap wide atomics whose SIMD "
             "group's active threads all access one address take the lock "
             "once for all of them, the operations applied in lane order"));

static cl::opt<bool> CASBatchTryFirst(
    "agx-cas-batch-try-first", cl::init(false), cl::Hidden,
    cl::desc("Each thread tries one compare-and-swap before the batch (only "
             "the threads that failed are batched)"));

static cl::opt<bool> CASBatch(
    "agx-cas-batch", cl::init(true), cl::Hidden,
    cl::desc("32-bit read-modify-write atomics that are compare-and-swap "
             "loops (atomicInc / atomicDec, f16x2 / bf16x2 additions, float "
             "min / max): after one try, the threads whose try failed on one "
             "address take one compare-and-swap of their group's result, the "
             "operations applied in lane order"));

namespace {

class AGXExpandWideAtomics : public FunctionPass {
public:
  static char ID;
  explicit AGXExpandWideAtomics(const AGXTargetMachine *TM = nullptr)
      : FunctionPass(ID), TM(TM) {}

  StringRef getPassName() const override { return "AGX Expand Wide Atomics"; }
  bool runOnFunction(Function &F) override;

private:
  const AGXTargetMachine *TM;
};

} // namespace

char AGXExpandWideAtomics::ID = 0;

INITIALIZE_PASS(AGXExpandWideAtomics, DEBUG_TYPE, "AGX Expand Wide Atomics",
                false, false)

FunctionPass *llvm::createAGXExpandWideAtomicsPass(const AGXTargetMachine *TM) {
  return new AGXExpandWideAtomics(TM);
}

static GlobalVariable *getLocks(Module &M) {
  const char *Name = "__agx_atomic_locks";
  if (GlobalVariable *G = M.getNamedGlobal(Name))
    return G;
  auto *Ty = ArrayType::get(Type::getInt32Ty(M.getContext()), NumLocks);
  return new GlobalVariable(M, Ty, false, GlobalValue::LinkOnceODRLinkage,
                            Constant::getNullValue(Ty), Name, nullptr,
                            GlobalValue::NotThreadLocal, /*AddressSpace=*/1);
}

/// The size in bits of the value of a wide atomic instruction, or 0.
static unsigned wideAtomicSize(const Instruction &I, const DataLayout &DL) {
  Type *Ty = nullptr;
  if (auto *LI = dyn_cast<LoadInst>(&I); LI && LI->isAtomic())
    Ty = LI->getType();
  else if (auto *SI = dyn_cast<StoreInst>(&I); SI && SI->isAtomic())
    Ty = SI->getValueOperand()->getType();
  else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
    Ty = RMW->getType();
  else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
    Ty = CX->getCompareOperand()->getType();
  if (!Ty)
    return 0;
  unsigned Bits = DL.getTypeSizeInBits(Ty);
  return Bits == 64 || Bits == 128 ? Bits : 0;
}

static Value *getPointer(Instruction &I) {
  if (auto *LI = dyn_cast<LoadInst>(&I))
    return LI->getPointerOperand();
  if (auto *SI = dyn_cast<StoreInst>(&I))
    return SI->getPointerOperand();
  if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
    return RMW->getPointerOperand();
  return cast<AtomicCmpXchgInst>(I).getPointerOperand();
}

/// V (of any type of Bits bits) as an integer, and back.
static Value *toInt(IRBuilderBase &B, Value *V, unsigned Bits) {
  Type *IntTy = B.getIntNTy(Bits);
  if (V->getType()->isPointerTy())
    return B.CreatePtrToInt(V, IntTy);
  return B.CreateBitCast(V, IntTy);
}

static Value *fromInt(IRBuilderBase &B, Value *V, Type *Ty) {
  if (Ty->isPointerTy())
    return B.CreateIntToPtr(V, Ty);
  return B.CreateBitCast(V, Ty);
}

/// Words of the wide value at Ptr, read with 32-bit atomics.
static Value *readWide(IRBuilderBase &B, Value *Ptr, unsigned Bits) {
  Value *Loaded = ConstantInt::get(B.getIntNTy(Bits), 0);
  for (unsigned K = 0; K < Bits / 32; ++K) {
    Value *W = B.CreateAtomicRMW(AtomicRMWInst::Or,
                                 B.CreateConstGEP1_32(B.getInt8Ty(), Ptr, 4 * K),
                                 B.getInt32(0), MaybeAlign(4), AtomicOrdering::Monotonic);
    Loaded = B.CreateOr(Loaded, B.CreateShl(B.CreateZExt(W, B.getIntNTy(Bits)), 32 * K));
  }
  return Loaded;
}

static void writeWide(IRBuilderBase &B, Value *Ptr, Value *NewInt, unsigned Bits) {
  for (unsigned K = 0; K < Bits / 32; ++K)
    B.CreateAtomicRMW(AtomicRMWInst::Xchg, B.CreateConstGEP1_32(B.getInt8Ty(), Ptr, 4 * K),
                      B.CreateTrunc(B.CreateLShr(NewInt, 32 * K), B.getInt32Ty()),
                      MaybeAlign(4), AtomicOrdering::Monotonic);
}

/// V (Bits wide) from lane L of the SIMD group, by 32-bit shuffles.
static Value *shuffleWide(IRBuilderBase &B, Value *V, Value *L, unsigned Bits) {
  Value *R = ConstantInt::get(B.getIntNTy(Bits), 0);
  for (unsigned K = 0; K < Bits / 32; ++K) {
    Value *W = B.CreateTrunc(B.CreateLShr(V, 32 * K), B.getInt32Ty());
    W = B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {B.getInt32Ty()}, {W, L});
    R = B.CreateOr(R, B.CreateShl(B.CreateZExt(W, B.getIntNTy(Bits)), 32 * K));
  }
  return R;
}

/// The SIMD group's batch of I for a group whose active threads all access
/// one address, in blocks from Uni to Exit; returns the block that ends it
/// and the result.
static std::pair<BasicBlock *, Value *>
expandUniform(Instruction &I, unsigned Bits, Value *Ptr, Value *Lock, Type *ValTy,
              BasicBlock *Uni, BasicBlock *Exit, Value *First) {
  LLVMContext &Ctx = I.getContext();
  Function *F = Uni->getParent();
  IntegerType *IntTy = IntegerType::get(Ctx, Bits);
  auto *RMW = dyn_cast<AtomicRMWInst>(&I);
  auto *CX = dyn_cast<AtomicCmpXchgInst>(&I);
  AtomicOrdering Ord = RMW ? RMW->getOrdering() : CX->getSuccessOrdering();
  bool Strong = isStrongerThanMonotonic(Ord);
  BasicBlock *Acquire = BasicBlock::Create(Ctx, "wide.atomic.batch.lock", F, Exit);
  BasicBlock *Locked = BasicBlock::Create(Ctx, "wide.atomic.batch.read", F, Exit);
  BasicBlock *Read = BasicBlock::Create(Ctx, "wide.atomic.batch", F, Exit);
  BasicBlock *Chain = BasicBlock::Create(Ctx, "wide.atomic.batch.op", F, Exit);
  BasicBlock *ChainEnd = BasicBlock::Create(Ctx, "wide.atomic.batch.ops", F, Exit);
  BasicBlock *Release = BasicBlock::Create(Ctx, "wide.atomic.batch.write", F, Exit);
  BasicBlock *End = BasicBlock::Create(Ctx, "wide.atomic.batch.end", F, Exit);

  IRBuilder<> B(Uni);
  Value *Lane = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::agx_thread_index_in_simdgroup, {});
  Value *Pending = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::agx_simd_ballot, {B.getTrue()});
  Value *Leader = B.CreateICmpEQ(Lane, First);
  B.CreateCondBr(Leader, Acquire, Read);
  // the lowest thread takes the lock (no other thread of the group waits for
  // it, so spinning cannot stop the group) and reads the value
  B.SetInsertPoint(Acquire);
  Value *Acq = B.CreateAtomicCmpXchg(Lock, B.getInt32(0), B.getInt32(1), MaybeAlign(4),
                                     AtomicOrdering::Monotonic, AtomicOrdering::Monotonic);
  B.CreateCondBr(B.CreateExtractValue(Acq, 1), Locked, Acquire);
  B.SetInsertPoint(Locked);
  B.CreateFence(AtomicOrdering::Acquire);
  Value *V0 = readWide(B, Ptr, Bits);
  B.CreateBr(Read);
  B.SetInsertPoint(Read);
  PHINode *V0P = B.CreatePHI(IntTy, 2);
  V0P->addIncoming(V0, Locked);
  V0P->addIncoming(PoisonValue::get(IntTy), Uni);
  Value *V = shuffleWide(B, V0P, First, Bits);
  if (Strong)
    B.CreateFence(AtomicOrdering::Acquire);
  B.CreateBr(Chain);
  // the operations in lane order, every thread computing the same values
  B.SetInsertPoint(Chain);
  PHINode *Rem = B.CreatePHI(B.getInt32Ty(), 2);
  PHINode *Cur = B.CreatePHI(IntTy, 2);
  PHINode *Mine = B.CreatePHI(IntTy, 2);
  PHINode *MineOk = B.CreatePHI(B.getInt1Ty(), 2);
  Rem->addIncoming(Pending, Read);
  Cur->addIncoming(V, Read);
  Mine->addIncoming(PoisonValue::get(IntTy), Read);
  MineOk->addIncoming(B.getFalse(), Read);
  Value *J = B.CreateBinaryIntrinsic(Intrinsic::cttz, Rem, B.getTrue());
  Value *IsMe = B.CreateICmpEQ(Lane, J);
  Value *Next, *Ok = B.getFalse();
  if (RMW) {
    Value *A = fromInt(B, shuffleWide(B, toInt(B, RMW->getValOperand(), Bits), J, Bits),
                       RMW->getValOperand()->getType());
    Next = toInt(B, buildAtomicRMWValue(RMW->getOperation(), B, fromInt(B, Cur, ValTy), A), Bits);
  } else {
    Value *Cmp = shuffleWide(B, toInt(B, CX->getCompareOperand(), Bits), J, Bits);
    Value *New = shuffleWide(B, toInt(B, CX->getNewValOperand(), Bits), J, Bits);
    Ok = B.CreateICmpEQ(Cur, Cmp);
    Next = B.CreateSelect(Ok, New, Cur);
  }
  Value *Mine2 = B.CreateSelect(IsMe, Cur, Mine);
  Value *MineOk2 = B.CreateSelect(IsMe, Ok, MineOk);
  Value *Rem2 = B.CreateAnd(Rem, B.CreateSub(Rem, B.getInt32(1)));
  Rem->addIncoming(Rem2, Chain);
  Cur->addIncoming(Next, Chain);
  Mine->addIncoming(Mine2, Chain);
  MineOk->addIncoming(MineOk2, Chain);
  B.CreateCondBr(B.CreateICmpNE(Rem2, B.getInt32(0)), Chain, ChainEnd);
  B.SetInsertPoint(ChainEnd);
  if (Strong)
    B.CreateFence(AtomicOrdering::Release);
  B.CreateCondBr(Leader, Release, End);
  B.SetInsertPoint(Release);
  writeWide(B, Ptr, Next, Bits);
  B.CreateFence(AtomicOrdering::Release);
  B.CreateAtomicRMW(AtomicRMWInst::Xchg, Lock, B.getInt32(0), MaybeAlign(4),
                    AtomicOrdering::Monotonic);
  B.CreateBr(End);
  B.SetInsertPoint(End);
  if (Strong)
    B.CreateFence(AtomicOrdering::Acquire);
  Value *Old = fromInt(B, Mine2, ValTy);
  Value *Result = Old;
  if (CX) {
    Value *Pair = B.CreateInsertValue(PoisonValue::get(CX->getType()), Old, 0);
    Result = B.CreateInsertValue(Pair, MineOk2, 1);
  }
  B.CreateBr(Exit);
  return {End, Result};
}

static void expand(Instruction &I, unsigned Bits, GlobalVariable *Locks) {
  const DataLayout &DL = I.getDataLayout();
  LLVMContext &Ctx = I.getContext();
  Value *Ptr = getPointer(I);
  unsigned Words = Bits / 32;
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *ValTy = isa<StoreInst>(I) ? cast<StoreInst>(I).getValueOperand()->getType()
                : isa<AtomicCmpXchgInst>(I)
                    ? cast<AtomicCmpXchgInst>(I).getCompareOperand()->getType()
                    : I.getType();

  BasicBlock *Pre = I.getParent();
  BasicBlock *Exit = Pre->splitBasicBlock(I.getIterator(), "wide.atomic.end");
  Function *F = Pre->getParent();
  BasicBlock *Head = BasicBlock::Create(Ctx, "wide.atomic", F, Exit);
  BasicBlock *Crit = BasicBlock::Create(Ctx, "wide.atomic.locked", F, Exit);
  BasicBlock *Latch = BasicBlock::Create(Ctx, "wide.atomic.latch", F, Exit);
  Pre->getTerminator()->setSuccessor(0, Head);

  IRBuilder<> B(Pre->getTerminator());
  unsigned PtrBits = DL.getPointerSizeInBits(Ptr->getType()->getPointerAddressSpace());
  Value *Addr = B.CreatePtrToInt(Ptr, B.getIntNTy(PtrBits));
  Value *Slot = B.CreateAnd(B.CreateLShr(Addr, 4), NumLocks - 1);
  Value *Lock = B.CreateInBoundsGEP(Locks->getValueType(), Locks,
                                    {B.getInt64(0), B.CreateZExtOrTrunc(Slot, B.getInt64Ty())});

  BasicBlock *UniEnd = nullptr;
  Value *UniResult = nullptr;
  if (UniformBatch && (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I))) {
    // whether the group's active threads all access one address
    Value *Pending = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::agx_simd_ballot, {B.getTrue()});
    Value *First = B.CreateBinaryIntrinsic(Intrinsic::cttz, Pending, B.getTrue());
    Value *A64 = B.CreateZExtOrTrunc(Addr, B.getInt64Ty());
    Value *FirstAddr = shuffleWide(B, A64, First, 64);
    Value *Same = B.CreateIntrinsic(B.getInt32Ty(), Intrinsic::agx_simd_ballot,
                                    {B.CreateICmpEQ(A64, FirstAddr)});
    BasicBlock *Uni = BasicBlock::Create(Ctx, "wide.atomic.batch.check", F, Head);
    Pre->getTerminator()->eraseFromParent();
    B.SetInsertPoint(Pre);
    B.CreateCondBr(B.CreateICmpEQ(Same, Pending), Uni, Head);
    std::tie(UniEnd, UniResult) = expandUniform(I, Bits, Ptr, Lock, ValTy, Uni, Exit, First);
  }

  B.SetInsertPoint(Head);
  Value *Acq = B.CreateAtomicCmpXchg(Lock, B.getInt32(0), B.getInt32(1), MaybeAlign(4),
                                     AtomicOrdering::Monotonic,
                                     AtomicOrdering::Monotonic);
  B.CreateCondBr(B.CreateExtractValue(Acq, 1), Crit, Latch);

  B.SetInsertPoint(Crit);
  // Lock semantics: acquire after taking the lock, release before freeing it.
  B.CreateFence(AtomicOrdering::Acquire);
  auto Word = [&](unsigned K) {
    return B.CreateConstGEP1_32(B.getInt8Ty(), Ptr, 4 * K);
  };
  Value *Loaded = ConstantInt::get(B.getIntNTy(Bits), 0);
  for (unsigned K = 0; K < Words; ++K) {
    Value *W = B.CreateAtomicRMW(AtomicRMWInst::Or, Word(K), B.getInt32(0),
                                 MaybeAlign(4), AtomicOrdering::Monotonic);
    W = B.CreateShl(B.CreateZExt(W, B.getIntNTy(Bits)), 32 * K);
    Loaded = B.CreateOr(Loaded, W);
  }
  Value *Old = fromInt(B, Loaded, ValTy);
  Value *Result = Old;
  Value *New = nullptr;     // the value to store
  Value *DoStore = nullptr; // whether to store it
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    New = SI->getValueOperand();
  } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
    New = buildAtomicRMWValue(RMW->getOperation(), B, Old, RMW->getValOperand());
  } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
    Value *Eq = B.CreateICmpEQ(Loaded, toInt(B, CX->getCompareOperand(), Bits));
    New = CX->getNewValOperand();
    DoStore = Eq;
    Value *Pair = B.CreateInsertValue(PoisonValue::get(CX->getType()), Old, 0);
    Result = B.CreateInsertValue(Pair, Eq, 1);
  }
  if (New) {
    Value *NewInt = toInt(B, New, Bits);
    BasicBlock *StoreBB = nullptr, *After = nullptr;
    if (DoStore) {
      StoreBB = BasicBlock::Create(Ctx, "wide.atomic.store", F, Latch);
      After = BasicBlock::Create(Ctx, "wide.atomic.unlock", F, Latch);
      B.CreateCondBr(DoStore, StoreBB, After);
      B.SetInsertPoint(StoreBB);
    }
    for (unsigned K = 0; K < Words; ++K) {
      Value *W = B.CreateTrunc(B.CreateLShr(NewInt, 32 * K), I32);
      B.CreateAtomicRMW(AtomicRMWInst::Xchg, Word(K), W, MaybeAlign(4),
                        AtomicOrdering::Monotonic);
    }
    if (DoStore) {
      B.CreateBr(After);
      B.SetInsertPoint(After);
    }
  }
  B.CreateFence(AtomicOrdering::Release);
  B.CreateAtomicRMW(AtomicRMWInst::Xchg, Lock, B.getInt32(0), MaybeAlign(4),
                    AtomicOrdering::Monotonic);
  BasicBlock *CritEnd = B.GetInsertBlock();
  B.CreateBr(Latch);

  Type *ResTy = isa<AtomicCmpXchgInst>(I) ? I.getType() : ValTy;
  B.SetInsertPoint(Latch);
  PHINode *Got = B.CreatePHI(B.getInt1Ty(), 2);
  Got->addIncoming(B.getTrue(), CritEnd);
  Got->addIncoming(B.getFalse(), Head);
  PHINode *Res = B.CreatePHI(ResTy, 2);
  Res->addIncoming(Result, CritEnd);
  // The result of a thread that has not done its work yet is not used.
  Res->addIncoming(PoisonValue::get(ResTy), Head);
  B.CreateCondBr(Got, Exit, Head);

  if (!I.getType()->isVoidTy()) {
    B.SetInsertPoint(&I);
    PHINode *Out = B.CreatePHI(ResTy, UniEnd ? 2 : 1);
    Out->addIncoming(Res, Latch);
    if (UniEnd)
      Out->addIncoming(UniResult, UniEnd);
    I.replaceAllUsesWith(Out);
  }
  I.eraseFromParent();
}

/// A 32-bit read-modify-write that AtomicExpand makes a compare-and-swap loop
/// of (atomicInc / atomicDec, f16x2 / bf16x2 additions, float min / max, ...):
/// a SIMD group whose active threads all access one address forms a batch:
/// the lowest thread reads the value, all apply their operations in lane
/// order, the lowest compare-and-swaps the result, and the batch tries again
/// if that failed. Groups on different addresses keep the loop. With
/// -agx-cas-batch-try-first every thread first tries one compare-and-swap,
/// as the loop would, and only those that failed are batched: no check on an
/// uncontended atomic, but on one address each group's 32 tries serialize.
static void expandCASBatch(Instruction &I) {
  LLVMContext &Ctx = I.getContext();
  const DataLayout &DL = I.getDataLayout();
  auto *RMW = dyn_cast<AtomicRMWInst>(&I);
  auto *CXI = dyn_cast<AtomicCmpXchgInst>(&I);
  Value *Ptr = RMW ? RMW->getPointerOperand() : CXI->getPointerOperand();
  Type *ValTy = RMW ? RMW->getType() : CXI->getCompareOperand()->getType();
  Value *ValOp = RMW ? RMW->getValOperand() : CXI->getNewValOperand();
  AtomicOrdering Ord = RMW ? RMW->getOrdering() : CXI->getSuccessOrdering();
  SyncScope::ID SSID = RMW ? RMW->getSyncScopeID() : CXI->getSyncScopeID();
  Type *I32 = Type::getInt32Ty(Ctx);
  Align A = RMW ? RMW->getAlign() : CXI->getAlign();
  BasicBlock *Pre = I.getParent();
  Function *F = Pre->getParent();
  BasicBlock *Exit = Pre->splitBasicBlock(I.getIterator(), "cas.batch.end");
  BasicBlock *Failed = BasicBlock::Create(Ctx, "cas.batch.failed", F, Exit);
  BasicBlock *Batch = BasicBlock::Create(Ctx, "cas.batch", F, Exit);
  BasicBlock *Read = BasicBlock::Create(Ctx, "cas.batch.read", F, Exit);
  BasicBlock *LoadBB = BasicBlock::Create(Ctx, "cas.batch.load", F, Exit);
  BasicBlock *Got = BasicBlock::Create(Ctx, "cas.batch.ops", F, Exit);
  BasicBlock *Chain = BasicBlock::Create(Ctx, "cas.batch.op", F, Exit);
  BasicBlock *ChainEnd = BasicBlock::Create(Ctx, "cas.batch.cas", F, Exit);
  BasicBlock *Swap = BasicBlock::Create(Ctx, "cas.batch.swap", F, Exit);
  BasicBlock *Tried = BasicBlock::Create(Ctx, "cas.batch.tried", F, Exit);
  BasicBlock *Single = BasicBlock::Create(Ctx, "cas.batch.single", F, Exit);
  Value *Old0 = nullptr;
  // a 16-bit operation works on its aligned word, at bit Shift of it
  unsigned Bits = DL.getTypeStoreSizeInBits(ValTy);
  Value *Shift = nullptr, *WordPtr = Ptr;
  if (Bits == 16) {
    IRBuilder<> SB(Pre->getTerminator());
    unsigned PB =
        DL.getPointerSizeInBits(Ptr->getType()->getPointerAddressSpace());
    Value *Low = SB.CreateAnd(SB.CreatePtrToInt(Ptr, SB.getIntNTy(PB)), 3);
    WordPtr = SB.CreateGEP(SB.getInt8Ty(), Ptr, SB.CreateNeg(Low));
    Shift = SB.CreateShl(SB.CreateTrunc(Low, I32), 3);
    A = Align(4);
  }
  auto Narrow = [&](IRBuilderBase &IB, Value *Word, Value *Sh) -> Value * {
    return Bits == 16 ? IB.CreateTrunc(IB.CreateLShr(Word, Sh), IB.getInt16Ty())
                      : Word;
  };

  // the first try (or none: all threads to the batch's check)
  IRBuilder<> B(Pre->getTerminator());
  if (!CASBatchTryFirst || Bits != 32 || CXI) {
    Pre->getTerminator()->setSuccessor(0, Failed);
  } else {
    LoadInst *L0 = B.CreateAlignedLoad(I32, Ptr, A);
    L0->setAtomic(AtomicOrdering::Monotonic, SSID);
    Value *New0 = toInt(B,
                        buildAtomicRMWValue(RMW->getOperation(), B,
                                            fromInt(B, L0, ValTy), ValOp),
                        32);
    Value *CX0 = B.CreateAtomicCmpXchg(Ptr, L0, New0, A, Ord,
                                       AtomicOrdering::Monotonic, SSID);
    Pre->getTerminator()->eraseFromParent();
    B.SetInsertPoint(Pre);
    B.CreateCondBr(B.CreateExtractValue(CX0, 1), Exit, Failed);
    Old0 = fromInt(B, L0, ValTy);
  }

  // the threads that failed: one address?
  B.SetInsertPoint(Failed);
  unsigned PtrBits =
      DL.getPointerSizeInBits(Ptr->getType()->getPointerAddressSpace());
  Value *A64 = B.CreateZExtOrTrunc(
      B.CreatePtrToInt(WordPtr, B.getIntNTy(PtrBits)), B.getInt64Ty());
  Value *Pending =
      B.CreateIntrinsic(I32, Intrinsic::agx_simd_ballot, {B.getTrue()});
  Value *First = B.CreateBinaryIntrinsic(Intrinsic::cttz, Pending, B.getTrue());
  Value *Same =
      B.CreateIntrinsic(I32, Intrinsic::agx_simd_ballot,
                        {B.CreateICmpEQ(A64, shuffleWide(B, A64, First, 64))});
  B.CreateCondBr(B.CreateICmpEQ(Same, Pending), Batch, Single);

  B.SetInsertPoint(Batch);
  Value *Lane =
      B.CreateIntrinsic(I32, Intrinsic::agx_thread_index_in_simdgroup, {});
  Value *Leader = B.CreateICmpEQ(Lane, First);
  Value *Arg = toInt(B, ValOp, Bits);
  Value *Cmp = CXI ? toInt(B, CXI->getCompareOperand(), Bits) : nullptr;
  if (Bits == 16) {
    Arg = B.CreateZExt(Arg, I32);
    if (Cmp)
      Cmp = B.CreateZExt(Cmp, I32);
  }
  B.CreateBr(Read);
  // (the retries: every thread of the batch loops until the lowest one's
  // swap succeeds)
  B.SetInsertPoint(Read);
  B.CreateCondBr(Leader, LoadBB, Got);
  B.SetInsertPoint(LoadBB);
  LoadInst *L = B.CreateAlignedLoad(I32, WordPtr, A);
  L->setAtomic(AtomicOrdering::Monotonic, SSID);
  B.CreateBr(Got);
  B.SetInsertPoint(Got);
  PHINode *V0 = B.CreatePHI(I32, 2);
  V0->addIncoming(L, LoadBB);
  V0->addIncoming(PoisonValue::get(I32), Read);
  Value *V = B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {I32}, {V0, First});
  B.CreateBr(Chain);
  B.SetInsertPoint(Chain);
  PHINode *Rem = B.CreatePHI(I32, 2);
  PHINode *Cur = B.CreatePHI(I32, 2);
  PHINode *Mine = B.CreatePHI(B.getIntNTy(Bits), 2);
  PHINode *MineOk = B.CreatePHI(B.getInt1Ty(), 2);
  MineOk->addIncoming(B.getFalse(), Got);
  Rem->addIncoming(Pending, Got);
  Cur->addIncoming(V, Got);
  Mine->addIncoming(PoisonValue::get(B.getIntNTy(Bits)), Got);
  Value *J = B.CreateBinaryIntrinsic(Intrinsic::cttz, Rem, B.getTrue());
  Value *ArgJ = B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {I32}, {Arg, J});
  Value *ShJ =
      Shift ? B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {I32}, {Shift, J})
            : nullptr;
  Value *CurJ = Narrow(B, Cur, ShJ);
  Value *ArgJN = Narrow(B, ArgJ, B.getInt32(0));
  Value *OkJ = B.getFalse(), *Next;
  if (CXI) {
    // a compare-and-swap: the new value where the thread's half is the one
    // it expects
    Value *CmpJ =
        B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {I32}, {Cmp, J});
    OkJ = B.CreateICmpEQ(CurJ, Narrow(B, CmpJ, B.getInt32(0)));
    Next = B.CreateSelect(OkJ, ArgJN, CurJ);
  } else {
    Next = toInt(B,
                 buildAtomicRMWValue(RMW->getOperation(), B,
                                     fromInt(B, CurJ, ValTy),
                                     fromInt(B, ArgJN, ValTy)),
                 Bits);
  }
  if (Bits == 16) {
    Value *Mask = B.CreateShl(B.getInt32(0xffff), ShJ);
    Next = B.CreateOr(B.CreateAnd(Cur, B.CreateNot(Mask)),
                      B.CreateShl(B.CreateZExt(Next, I32), ShJ));
  }
  Value *IsMe = B.CreateICmpEQ(Lane, J);
  Value *Mine2 = B.CreateSelect(IsMe, CurJ, Mine);
  Value *MineOk2 = B.CreateSelect(IsMe, OkJ, MineOk);
  MineOk->addIncoming(MineOk2, Chain);
  Value *Rem2 = B.CreateAnd(Rem, B.CreateSub(Rem, B.getInt32(1)));
  Rem->addIncoming(Rem2, Chain);
  Cur->addIncoming(Next, Chain);
  Mine->addIncoming(Mine2, Chain);
  B.CreateCondBr(B.CreateICmpNE(Rem2, B.getInt32(0)), Chain, ChainEnd);
  B.SetInsertPoint(ChainEnd);
  B.CreateCondBr(Leader, Swap, Tried);
  B.SetInsertPoint(Swap);
  Value *CX = B.CreateAtomicCmpXchg(WordPtr, V, Next, A, Ord,
                                    AtomicOrdering::Monotonic, SSID);
  Value *Ok = B.CreateZExt(B.CreateExtractValue(CX, 1), I32);
  B.CreateBr(Tried);
  B.SetInsertPoint(Tried);
  PHINode *OkP = B.CreatePHI(I32, 2);
  OkP->addIncoming(Ok, Swap);
  OkP->addIncoming(PoisonValue::get(I32), ChainEnd);
  Value *Done =
      B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {I32}, {OkP, First});
  B.CreateCondBr(B.CreateICmpNE(Done, B.getInt32(0)), Exit, Read);

  // threads that failed on different addresses: the instruction as it was
  // (AtomicExpand's loop)
  B.SetInsertPoint(Single);
  Instruction *Copy = I.clone();
  B.Insert(Copy);
  B.CreateBr(Exit);

  IRBuilder<> TB(Tried->getTerminator());
  Value *OldBatch = fromInt(TB, Mine2, ValTy);
  if (CXI)
    OldBatch = TB.CreateInsertValue(
        TB.CreateInsertValue(PoisonValue::get(CXI->getType()), OldBatch, 0),
        MineOk2, 1);
  B.SetInsertPoint(&I);
  PHINode *Out = B.CreatePHI(I.getType(), 3);
  if (Old0)
    Out->addIncoming(Old0, Pre);
  Out->addIncoming(OldBatch, Tried);
  Out->addIncoming(Copy, Single);
  I.replaceAllUsesWith(Out);
  I.eraseFromParent();
}

/// An 8- or 16-bit atomic load as an atomic load of its aligned 32-bit word,
/// shifted and truncated: AtomicExpand makes atomic loads compare-and-swaps
/// and does not widen the narrow ones it makes (instruction selection has no
/// 8- or 16-bit compare-and-swap).
static void widenNarrowAtomicLoad(LoadInst &LI) {
  IRBuilder<> B(&LI);
  const DataLayout &DL = LI.getDataLayout();
  Value *Ptr = LI.getPointerOperand();
  unsigned PtrBits =
      DL.getPointerSizeInBits(Ptr->getType()->getPointerAddressSpace());
  unsigned Bits = DL.getTypeStoreSizeInBits(LI.getType());
  Value *Low = B.CreateAnd(B.CreatePtrToInt(Ptr, B.getIntNTy(PtrBits)), 3);
  Value *Word = B.CreateGEP(B.getInt8Ty(), Ptr, B.CreateNeg(Low));
  LoadInst *W =
      B.CreateAlignedLoad(B.getInt32Ty(), Word, Align(4), LI.isVolatile());
  W->setAtomic(LI.getOrdering(), LI.getSyncScopeID());
  Value *Shift = B.CreateShl(B.CreateTrunc(Low, B.getInt32Ty()), 3);
  Value *V = B.CreateTrunc(B.CreateLShr(W, Shift), B.getIntNTy(Bits));
  if (!LI.getType()->isIntegerTy())
    V = B.CreateBitCast(V, LI.getType());
  LI.replaceAllUsesWith(V);
  LI.eraseFromParent();
}

bool AGXExpandWideAtomics::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const DataLayout &DL = F.getDataLayout();
  const AGXTargetLowering *TLI =
      TM ? TM->getSubtargetImpl(F)->getTargetLowering() : nullptr;
  SmallVector<std::pair<Instruction *, unsigned>, 4> Wide;
  SmallVector<Instruction *, 4> CASLoops;
  SmallVector<LoadInst *, 4> NarrowLoads;
  for (Instruction &I : instructions(F)) {
    // 16-bit compare-and-swaps (AtomicExpand's partword loops)
    if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I);
        CASBatch && CX && TM && !TM->getSubtargetImpl(F)->hasMesaISA() &&
        !CX->isVolatile() &&
        DL.getTypeStoreSize(CX->getCompareOperand()->getType()) == 2 &&
        CX->getAlign() >= 2 &&
        !CX->getCompareOperand()->getType()->isPointerTy() &&
        !isStrongerThanMonotonic(CX->getSuccessOrdering())) {
      CASLoops.push_back(CX);
      continue;
    }
    if (auto *LI = dyn_cast<LoadInst>(&I);
        LI && LI->isAtomic() && !LI->getType()->isPointerTy() &&
        DL.getTypeStoreSize(LI->getType()) < 4 &&
        LI->getAlign() >= DL.getTypeStoreSize(LI->getType())) {
      NarrowLoads.push_back(LI);
      continue;
    }
    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I);
        CASBatch && RMW && TLI && !TM->getSubtargetImpl(F)->hasMesaISA() &&
        (DL.getTypeStoreSize(RMW->getType()) == 4 ||
         (DL.getTypeStoreSize(RMW->getType()) == 2 && RMW->getAlign() >= 2)) &&
        !RMW->getType()->isPointerTy() &&
        TLI->shouldExpandAtomicRMWInIR(RMW) ==
            TargetLowering::AtomicExpansionKind::CmpXChg &&
        !isStrongerThanMonotonic(RMW->getOrdering())) {
      CASLoops.push_back(RMW);
      continue;
    }
    // Wide operations the chip has an instruction for.
    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I);
        RMW && TLI &&
        TLI->shouldExpandAtomicRMWInIR(RMW) ==
            TargetLowering::AtomicExpansionKind::None)
      continue;
    if (unsigned Bits = wideAtomicSize(I, DL))
      Wide.push_back({&I, Bits});
  }
  for (LoadInst *LI : NarrowLoads)
    widenNarrowAtomicLoad(*LI);
  for (Instruction *CAS : CASLoops)
    expandCASBatch(*CAS);
  if (Wide.empty())
    return !CASLoops.empty() || !NarrowLoads.empty();
  GlobalVariable *Locks = getLocks(*F.getParent());
  for (auto [I, Bits] : Wide)
    expand(*I, Bits, Locks);
  return true;
}
