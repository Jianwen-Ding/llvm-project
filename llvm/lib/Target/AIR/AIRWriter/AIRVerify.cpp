//===- AIRVerify.cpp - Conformance checks for AIR modules -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Each rule is what the device does with such a module: a check here that the
// device does not make would reject working code, so a new rule comes with a
// module in llvm/utils/air-gpu/verify-cases that shows it, which
// air_verify_device_check.py runs on the device. (AIR's documents exclude more than
// the device rejects: i3, address space casts and kernels that return a value run
// on an M2 Max, so they are not errors here.) The checks are linear in the size of
// the module.
//
//===----------------------------------------------------------------------===//

#include "AIRVerify.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::metal;

namespace {
class Checker {
  const Module &M;
  AIRForm Form;
  SmallVector<std::string, 4> Out;

  void add(const Twine &Msg) { Out.push_back(Msg.str()); }
  void add(const Function &F, const Twine &Msg) {
    add("in function '" + F.getName() + "': " + Msg);
  }
  bool isMetal() const { return Form == AIRForm::Metal; }

  void checkModule() {
    if (!M.getTargetTriple().isAIR())
      add("the module's triple is not AIR's: " + M.getTargetTriple().str());
    if (isMetal()) {
      for (StringRef Name : {"air.version", "air.language_version"})
        if (!M.getNamedMetadata(Name))
          add("the module has no !" + Name);
    }
  }

  void checkKernels() {
    NamedMDNode *Kernels = M.getNamedMetadata("air.kernel");
    if (!Kernels)
      return;
    DenseSet<const Function *> Seen;
    for (const MDNode *K : Kernels->operands()) {
      const Function *F =
          K->getNumOperands()
              ? mdconst::dyn_extract_or_null<Function>(K->getOperand(0))
              : nullptr;
      if (!F) {
        add("an !air.kernel entry does not name a function");
        continue;
      }
      if (!Seen.insert(F).second)
        add(*F, "more than one !air.kernel entry");
      if (F->isDeclaration())
        add(*F, "a kernel without a body");
      if (K->getNumOperands() < 3) {
        add(*F, "the !air.kernel entry does not describe the kernel's "
                "arguments");
        continue;
      }
      auto *Args = dyn_cast_or_null<MDNode>(K->getOperand(K->getNumOperands() - 1));
      if (!Args) {
        add(*F, "the !air.kernel entry has no argument list");
        continue;
      }
      SmallVector<bool, 16> Described(F->arg_size(), false);
      // (threadgroup buffers have location indices of their own)
      DenseSet<uint64_t> Locations, ThreadgroupLocations;
      for (const MDOperand &A : Args->operands()) {
        auto *Node = dyn_cast_or_null<MDNode>(A);
        auto *Idx = Node && Node->getNumOperands() >= 2
                        ? mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(0))
                        : nullptr;
        auto *Role = Idx ? dyn_cast_or_null<MDString>(Node->getOperand(1)) : nullptr;
        if (!Role) {
          add(*F, "a malformed !air.kernel argument");
          continue;
        }
        uint64_t I = Idx->getZExtValue();
        if (I >= F->arg_size()) {
          add(*F, "!air.kernel describes argument " + Twine(I) +
                      ", which the kernel does not have");
          continue;
        }
        if (Described[I])
          add(*F, "!air.kernel describes argument " + Twine(I) + " twice");
        Described[I] = true;
        if (!Role->getString().starts_with("air."))
          add(*F, "argument " + Twine(I) + " has the role '" + Role->getString() +
                      "', which is not AIR's");
        if (Role->getString() != "air.buffer")
          continue;
        if (!F->getArg(I)->getType()->isPointerTy())
          add(*F, "the air.buffer argument " + Twine(I) + " is not a pointer");
        for (unsigned J = 2; J + 1 < Node->getNumOperands(); ++J) {
          auto *Key = dyn_cast_or_null<MDString>(Node->getOperand(J));
          if (!Key)
            continue;
          auto *Value =
              mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(J + 1));
          if (!Value)
            continue;
          bool Threadgroup = F->getArg(I)->getType()->isPointerTy() &&
                             F->getArg(I)->getType()->getPointerAddressSpace() == 3;
          if (Key->getString() == "air.location_index" &&
              !(Threadgroup ? ThreadgroupLocations : Locations)
                   .insert(Value->getZExtValue())
                   .second)
            add(*F, "two buffer arguments at location " +
                        Twine(Value->getZExtValue()));
        }
      }
      for (unsigned I = 0; I != F->arg_size(); ++I)
        if (!Described[I])
          add(*F, "!air.kernel does not describe argument " + Twine(I));
    }
  }

  void checkCall(const Function &F, const CallBase &C) {
    const Function *Callee = C.getCalledFunction();
    if (!Callee)
      return;
    StringRef N = Callee->getName();
    if (N.starts_with("llvm.agx.") || N.starts_with("__agx") ||
        N.starts_with("agx."))
      add(F, "a call of '" + N + "', which is AGX's and not AIR's");
    if (!isMetal() || !Callee->isDeclaration())
      return;
    // Metal has no function of these names, and takes an unresolved one for an
    // error when it builds the pipeline.
    if (!N.starts_with("air.") && !N.starts_with("llvm.") &&
        !N.starts_with("__") && !Callee->hasFnAttribute("air-visible"))
      add(F, "a call of '" + N + "', which neither the module nor AIR defines");
  }

  void checkFunctions() {
    for (const Function &F : M) {
      if (F.getCallingConv() == CallingConv::AGX_KERNEL)
        add(F, "the calling convention agx_kernel");
      if (F.isDeclaration())
        continue;
      for (const Instruction &I : instructions(F)) {
        if (auto *C = dyn_cast<CallBase>(&I))
          checkCall(F, *C);
        // (not a rejection: the writer has no atomic load or store, so one reaching it
        // would lose its atomicity and a load polling for a write may never see it)
        if (isMetal() && ((isa<LoadInst>(I) && cast<LoadInst>(I).isAtomic()) ||
                          (isa<StoreInst>(I) && cast<StoreInst>(I).isAtomic())))
          add(F, "an atomic load or store that was not lowered");
      }
    }
  }

public:
  Checker(const Module &M, AIRForm Form) : M(M), Form(Form) {}
  SmallVector<std::string, 4> run(bool FullVerifier) {
    if (FullVerifier) {
      std::string Err;
      raw_string_ostream OS(Err);
      if (verifyModule(M, &OS))
        add("the module is broken: " + Err);
    } else {
      for (const Function &F : M)
        for (const BasicBlock &BB : F) {
          bool Others = false;
          for (const Instruction &I : BB) {
            if (Others && isa<PHINode>(I)) {
              add(F, "PHI nodes not grouped at top of basic block");
              break;
            }
            Others |= !isa<PHINode>(I);
          }
        }
    }
    checkModule();
    checkKernels();
    checkFunctions();
    return std::move(Out);
  }
};
} // namespace

SmallVector<std::string, 4> llvm::metal::verifyAIR(const Module &M, AIRForm Form,
                                                   bool FullVerifier) {
  return Checker(M, Form).run(FullVerifier);
}

std::string llvm::metal::verifyAIRBitcode(ArrayRef<uint8_t> Bytes) {
  LLVMContext Ctx;
  auto Buf = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size()),
      "air", /*RequiresNullTerminator=*/false);
  Expected<std::unique_ptr<Module>> M = parseBitcodeFile(*Buf, Ctx);
  if (!M)
    return "the written file cannot be read: " + toString(M.takeError());
  std::string Err;
  raw_string_ostream OS(Err);
  if (verifyModule(**M, &OS))
    return "the written file is broken: " + Err;
  return "";
}

PreservedAnalyses AIRVerifyPass::run(Module &M, ModuleAnalysisManager &) {
  for (const std::string &V : verifyAIR(M, Form))
    M.getContext().diagnose(DiagnosticInfoGeneric(V, DS_Error));
  return PreservedAnalyses::all();
}
