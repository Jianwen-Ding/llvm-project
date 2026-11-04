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

static cl::opt<bool> IncDecBatch(
    "agx-inc-dec-batch", cl::init(true), cl::Hidden,
    cl::desc("32-bit atomic increments and decrements with wrap (CUDA's "
             "atomicInc / atomicDec, compare-and-swap loops) whose SIMD "
             "group's active threads all access one address: one "
             "compare-and-swap of the group's result, the operations applied "
             "in lane order"));

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

/// A 32-bit uinc_wrap / udec_wrap (a compare-and-swap loop otherwise) for a
/// SIMD group whose active threads all access one address: the lowest thread
/// reads the value, every thread applies the group's operations in lane order
/// (each with its own operand), the lowest one compare-and-swaps the result,
/// and the group tries again if it failed. A contended counter is the common
/// use; other groups keep the per-thread loop.
static void expandIncDec(AtomicRMWInst &I) {
  LLVMContext &Ctx = I.getContext();
  const DataLayout &DL = I.getDataLayout();
  Value *Ptr = I.getPointerOperand();
  Type *I32 = Type::getInt32Ty(Ctx);
  BasicBlock *Pre = I.getParent();
  Function *F = Pre->getParent();
  BasicBlock *Exit = Pre->splitBasicBlock(I.getIterator(), "inc.dec.end");
  BasicBlock *Batch = BasicBlock::Create(Ctx, "inc.dec.batch", F, Exit);
  BasicBlock *Read = BasicBlock::Create(Ctx, "inc.dec.batch.read", F, Exit);
  BasicBlock *Got = BasicBlock::Create(Ctx, "inc.dec.batch.ops", F, Exit);
  BasicBlock *Chain = BasicBlock::Create(Ctx, "inc.dec.batch.op", F, Exit);
  BasicBlock *ChainEnd = BasicBlock::Create(Ctx, "inc.dec.batch.cas", F, Exit);
  BasicBlock *Swap = BasicBlock::Create(Ctx, "inc.dec.batch.swap", F, Exit);
  BasicBlock *Tried = BasicBlock::Create(Ctx, "inc.dec.batch.tried", F, Exit);
  BasicBlock *Single = BasicBlock::Create(Ctx, "inc.dec.single", F, Exit);

  IRBuilder<> B(Pre->getTerminator());
  unsigned PtrBits =
      DL.getPointerSizeInBits(Ptr->getType()->getPointerAddressSpace());
  Value *A64 = B.CreateZExtOrTrunc(B.CreatePtrToInt(Ptr, B.getIntNTy(PtrBits)),
                                   B.getInt64Ty());
  Value *Pending =
      B.CreateIntrinsic(I32, Intrinsic::agx_simd_ballot, {B.getTrue()});
  Value *First = B.CreateBinaryIntrinsic(Intrinsic::cttz, Pending, B.getTrue());
  Value *Same =
      B.CreateIntrinsic(I32, Intrinsic::agx_simd_ballot,
                        {B.CreateICmpEQ(A64, shuffleWide(B, A64, First, 64))});
  Pre->getTerminator()->eraseFromParent();
  B.SetInsertPoint(Pre);
  B.CreateCondBr(B.CreateICmpEQ(Same, Pending), Batch, Single);

  B.SetInsertPoint(Batch);
  Value *Lane =
      B.CreateIntrinsic(I32, Intrinsic::agx_thread_index_in_simdgroup, {});
  Value *Leader = B.CreateICmpEQ(Lane, First);
  B.CreateBr(Read);
  // (the retries: every thread loops until the lowest one's swap succeeds)
  B.SetInsertPoint(Read);
  BasicBlock *LoadBB = BasicBlock::Create(Ctx, "inc.dec.batch.load", F, Got);
  B.CreateCondBr(Leader, LoadBB, Got);
  B.SetInsertPoint(LoadBB);
  LoadInst *L = B.CreateAlignedLoad(I32, Ptr, Align(4));
  L->setAtomic(AtomicOrdering::Monotonic, I.getSyncScopeID());
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
  PHINode *Mine = B.CreatePHI(I32, 2);
  Rem->addIncoming(Pending, Got);
  Cur->addIncoming(V, Got);
  Mine->addIncoming(PoisonValue::get(I32), Got);
  Value *J = B.CreateBinaryIntrinsic(Intrinsic::cttz, Rem, B.getTrue());
  Value *Arg = B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {I32},
                                 {I.getValOperand(), J});
  Value *Next = buildAtomicRMWValue(I.getOperation(), B, Cur, Arg);
  Value *Mine2 = B.CreateSelect(B.CreateICmpEQ(Lane, J), Cur, Mine);
  Value *Rem2 = B.CreateAnd(Rem, B.CreateSub(Rem, B.getInt32(1)));
  Rem->addIncoming(Rem2, Chain);
  Cur->addIncoming(Next, Chain);
  Mine->addIncoming(Mine2, Chain);
  B.CreateCondBr(B.CreateICmpNE(Rem2, B.getInt32(0)), Chain, ChainEnd);
  B.SetInsertPoint(ChainEnd);
  B.CreateCondBr(Leader, Swap, Tried);
  B.SetInsertPoint(Swap);
  Value *CX =
      B.CreateAtomicCmpXchg(Ptr, V, Next, Align(4), I.getOrdering(),
                            AtomicOrdering::Monotonic, I.getSyncScopeID());
  Value *Ok = B.CreateZExt(B.CreateExtractValue(CX, 1), I32);
  B.CreateBr(Tried);
  B.SetInsertPoint(Tried);
  PHINode *OkP = B.CreatePHI(I32, 2);
  OkP->addIncoming(Ok, Swap);
  OkP->addIncoming(PoisonValue::get(I32), ChainEnd);
  Value *Done =
      B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {I32}, {OkP, First});
  B.CreateCondBr(B.CreateICmpNE(Done, B.getInt32(0)), Exit, Read);

  // the other groups: the instruction as it was (AtomicExpand's loop)
  B.SetInsertPoint(Single);
  Instruction *Copy = I.clone();
  B.Insert(Copy);
  B.CreateBr(Exit);

  B.SetInsertPoint(&I);
  PHINode *Out = B.CreatePHI(I32, 2);
  Out->addIncoming(Mine2, Tried);
  Out->addIncoming(Copy, Single);
  I.replaceAllUsesWith(Out);
  I.eraseFromParent();
}

bool AGXExpandWideAtomics::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const DataLayout &DL = F.getDataLayout();
  const AGXTargetLowering *TLI =
      TM ? TM->getSubtargetImpl(F)->getTargetLowering() : nullptr;
  SmallVector<std::pair<Instruction *, unsigned>, 4> Wide;
  SmallVector<AtomicRMWInst *, 4> IncDec;
  for (Instruction &I : instructions(F)) {
    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I);
        IncDecBatch && RMW && RMW->getType()->isIntegerTy(32) &&
        (RMW->getOperation() == AtomicRMWInst::UIncWrap ||
         RMW->getOperation() == AtomicRMWInst::UDecWrap) &&
        !isStrongerThanMonotonic(RMW->getOrdering())) {
      IncDec.push_back(RMW);
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
  for (AtomicRMWInst *RMW : IncDec)
    expandIncDec(*RMW);
  if (Wide.empty())
    return !IncDec.empty();
  GlobalVariable *Locks = getLocks(*F.getParent());
  for (auto [I, Bits] : Wide)
    expand(*I, Bits, Locks);
  return true;
}
