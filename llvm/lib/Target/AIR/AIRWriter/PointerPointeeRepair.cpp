//===- PointerPointeeRepair.cpp - Pointer/pointee type agreement ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Metal v1 typed-pointer reader requires every pointer-consuming record
// (phi, select, load/store, call arg) to agree with the pointee the writer
// emits for the pointer. These transforms pin disagreeing edges through an
// identity bitcast (recorded in the PTM) or repoint them, one mismatch class
// per function.
//
//===----------------------------------------------------------------------===//

#include "PointerPointeeRepair.h"
#include "PointeeRules.h"
#include "PointerRepairUtil.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace llvm {
namespace metal {

void fixPhiIncomingTypes(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN)
          break;
        if (!PN->getType()->isPointerTy())
          continue;

        Type *PhiPointee = requiredPhiPointee(PN, PTM);
        if (!PhiPointee)
          continue;
        for (unsigned J = 0; J < PN->getNumIncomingValues(); ++J) {
          Value *In = PN->getIncomingValue(J);
          Type *InPointee = nullptr;
          if (isa<ConstantPointerNull>(In)) {

            auto &Ctx = M.getContext();
            auto *Zero = ConstantInt::get(Type::getInt64Ty(Ctx), 0);
            auto *I2P = new IntToPtrInst(
                Zero, In->getType(), "",
                PN->getIncomingBlock(J)->getTerminator()->getIterator());
            PTM.set(I2P, PhiPointee);
            PN->setIncomingValue(J, I2P);
            continue;
          }
          if (isa<Constant>(In))
            continue;
          if (auto *GV = dyn_cast<GlobalVariable>(In))
            InPointee = GV->getValueType();
          else if (auto *G = dyn_cast<GetElementPtrInst>(In))
            InPointee = G->getResultElementType();
          else
            InPointee = PTM.get(In);
          if (InPointee == PhiPointee)
            continue;
          PN->setIncomingValue(
              J,
              retypePointerVia(In, PhiPointee,
                               PN->getIncomingBlock(J)->getTerminator(), PTM));
        }

        PTM.set(PN, PhiPointee);
      }
  }
}

void fixSelectPointerArms(Module &M, PointeeTypeMap &PTM) {
  auto pointeeOf = [&](Value *V) -> Type * {
    if (isa<ConstantPointerNull>(V))
      return nullptr;
    return effectivePointee(V, PTM);
  };
  auto Selects = collectInsts<SelectInst>(M, [&](SelectInst *S) {
    if (!S->getType()->isPointerTy())
      return false;
    Value *T = S->getTrueValue(), *F = S->getFalseValue();
    if (isa<ConstantPointerNull>(T) || isa<ConstantPointerNull>(F))
      return true;
    Type *Use = PointeeTypeMap::inferFromUsage(S);
    return pointeeOf(T) != pointeeOf(F) ||
           (Use && (pointeeOf(T) != Use || pointeeOf(F) != Use));
  });
  for (auto *S : Selects) {
    Value *T = S->getTrueValue(), *F = S->getFalseValue();
    Type *Pointee = PointeeTypeMap::inferFromUsage(S);
    if (!Pointee)
      Pointee = requiredSelectPointee(S, PTM);
    if (!Pointee)
      continue;
    if (pointeeOf(T) != Pointee || isa<ConstantPointerNull>(T))
      S->setOperand(1, retypePointerVia(T, Pointee, S, PTM));
    if (pointeeOf(F) != Pointee || isa<ConstantPointerNull>(F))
      S->setOperand(2, retypePointerVia(F, Pointee, S, PTM));
    PTM.set(S, Pointee);
  }
}

void fixAccessTypeMismatch(Module &M, PointeeTypeMap &PTM) {
  auto accessTypeOf = [](Instruction *I) -> Type * {
    if (auto *LI = dyn_cast<LoadInst>(I))
      return LI->getType();
    if (auto *SI = dyn_cast<StoreInst>(I))
      return SI->getValueOperand()->getType();
    return nullptr;
  };
  auto pointerOf = [](Instruction *I) -> Value * {
    if (auto *LI = dyn_cast<LoadInst>(I))
      return LI->getPointerOperand();
    return cast<StoreInst>(I)->getPointerOperand();
  };
  auto Fix = collectInsts<Instruction>(M, [&](Instruction *I) {
    Type *AccessTy = accessTypeOf(I);
    if (!AccessTy)
      return false;
    Value *Ptr = pointerOf(I);
    if (isa<BitCastInst>(Ptr))
      return false;

    if (!AccessTy->isVectorTy()) {
      if (!isa<IntToPtrInst>(Ptr) && !isa<ConstantPointerNull>(Ptr)) {
        Type *Pointee = effectivePointee(Ptr, PTM);
        if (!Pointee || Pointee == AccessTy)
          return false;
      }
    }
    return true;
  });
  for (Instruction *I : Fix) {
    if (auto *LI = dyn_cast<LoadInst>(I))
      LI->setOperand(
          0, retypePointerVia(LI->getPointerOperand(), LI->getType(), LI, PTM));
    else {
      auto *SI = cast<StoreInst>(I);
      SI->setOperand(1, retypePointerVia(SI->getPointerOperand(),
                                         SI->getValueOperand()->getType(), SI,
                                         PTM));
    }
  }
}

void fixMMAPointerSuffixMismatch(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto Calls = collectInsts<CallInst>(M, [](CallInst *CI) {
    return CI->getCalledFunction() &&
           CI->getCalledFunction()->getName().starts_with(
               "air.simdgroup_matrix_8x8_");
  });
  for (auto *CI : Calls) {
    StringRef Name = CI->getCalledFunction()->getName();
    Type *Elem = mmaElemFromName(Name, Ctx);
    if (!Elem)
      continue;
    for (unsigned J = 0; J < CI->arg_size(); J++) {
      Value *Op = CI->getArgOperand(J);
      if (!Op->getType()->isPointerTy())
        continue;
      if (Elem->isFloatTy() && !isa<Constant>(Op)) {

        Type *Pointee = effectivePointee(Op, PTM);
        if (!Pointee || Pointee == Elem)
          continue;
      }
      if (isa<BitCastInst>(Op) || isa<AllocaInst>(Op))
        continue;
      CI->setArgOperand(J, retypePointerVia(Op, Elem, CI, PTM));
    }
  }
}

void fixAtomicPointerArgs(Module &M, PointeeTypeMap &PTM) {
  auto &Ctx = M.getContext();
  auto Calls = collectInsts<CallInst>(M, [](CallInst *CI) {
    Function *Callee = CI->getCalledFunction();
    return Callee && Callee->getName().starts_with("air.atomic.") &&
           CI->arg_size() > 0 && CI->getArgOperand(0)->getType()->isPointerTy();
  });
  for (auto *CI : Calls) {
    StringRef Name = CI->getCalledFunction()->getName();
    Type *Want = Name.ends_with(".i32")   ? Type::getInt32Ty(Ctx)
                 : Name.ends_with(".f32") ? Type::getFloatTy(Ctx)
                                          : nullptr;
    Value *Ptr = CI->getArgOperand(0);
    unsigned AS = Ptr->getType()->getPointerAddressSpace();
    if (!Want || (AS != 1 && AS != 3) || isa<ConstantPointerNull>(Ptr))
      continue;
    Type *Have = isa<GlobalVariable>(Ptr)
                     ? cast<GlobalVariable>(Ptr)->getValueType()
                     : effectivePointee(Ptr, PTM);
    if (!Have || Have == Want)
      continue;
    CI->setArgOperand(0, retypePointerVia(Ptr, Want, CI, PTM));
  }
}

void fixGlobalGEPBases(Module &M, PointeeTypeMap &PTM) {
  auto GEPs = collectInsts<GetElementPtrInst>(M, [](GetElementPtrInst *G) {
    auto *GV = dyn_cast<GlobalVariable>(G->getPointerOperand());
    return GV && GV->getValueType() != G->getSourceElementType();
  });
  for (auto *G : GEPs)
    G->setOperand(0, retypePointerVia(G->getPointerOperand(),
                                      G->getSourceElementType(), G, PTM));
}

void fixComparePointerTypes(Module &M, PointeeTypeMap &PTM) {
  auto Cmps = collectInsts<ICmpInst>(M, [](ICmpInst *C) {
    return C->getOperand(0)->getType()->isPointerTy();
  });
  auto PointeeOf = [&](Value *V) -> Type * {
    if (auto *GV = dyn_cast<GlobalVariable>(V))
      return GV->getValueType();
    return effectivePointee(V, PTM);
  };
  for (auto *C : Cmps) {
    Value *A = C->getOperand(0), *B = C->getOperand(1);
    // (a constant has its one written type: null takes the other's with an
    // inttoptr of zero, as a phi's does)
    auto NullAs = [&](Value *Other) -> Value * {
      Type *Pointee = PointeeOf(Other);
      if (!Pointee)
        return nullptr;
      auto *Zero = ConstantInt::get(Type::getInt64Ty(C->getContext()), 0);
      auto *I2P = new IntToPtrInst(Zero, Other->getType(), "", C->getIterator());
      PTM.set(I2P, Pointee);
      return I2P;
    };
    if (isa<ConstantPointerNull>(A) && isa<ConstantPointerNull>(B))
      continue;
    if (isa<ConstantPointerNull>(B)) {
      if (Value *N = NullAs(A))
        C->setOperand(1, N);
      continue;
    }
    if (isa<ConstantPointerNull>(A)) {
      if (Value *N = NullAs(B))
        C->setOperand(0, N);
      continue;
    }
    Type *PA = PointeeOf(A), *PB = PointeeOf(B);
    if (PA && PB && PA != PB)
      C->setOperand(1, retypePointerVia(B, PA, C, PTM));
  }
}

void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM) {
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    SmallVector<BitCastInst *, 16> ToRemove;
    for (auto &BB : F) {
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC || BC->getSrcTy() != BC->getDestTy())
          continue;
        Type *SrcPT = PTM.get(BC->getOperand(0));
        Type *DstPT = PTM.get(BC);
        if (!SrcPT || !DstPT)
          continue;
        if (SrcPT != DstPT)
          continue;
        ToRemove.push_back(BC);
      }
    }
    for (auto *BC : ToRemove) {
      PTM.remove(BC);
      BC->replaceAllUsesWith(BC->getOperand(0));
      BC->eraseFromParent();
    }
  }
}

}
}
