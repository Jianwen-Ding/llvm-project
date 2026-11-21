//===-- AGXPairedHalfBitops.cpp - One bitop for both halves ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The two 16-bit halves of a 32-bit value, each put through the same bitwise
// operation with the same constant,
//
//   %lo = COPY %v.lo16          %hi = COPY %v.hi16
//   %a = BITOP_I16 tt, %lo, C   %b = BITOP_I16 tt, %hi, C
//
// are one 32-bit operation with the constant in both halves:
//
//   %w = BITOP_I32 tt, %v, (C << 16 | C)
//   %a = COPY %w.lo16           %b = COPY %w.hi16
//
// A pair of bytes loaded into the halves of a register and made floats by
// the xor with 0x6480 (L15-E533), q8_0 weights: one xor for two bytes
// (L15-E631).
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXRegisterInfo.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-paired-half-bitops"

static cl::opt<bool> AGXPairHalfBitops(
    "agx-pair-half-bitops", cl::init(true), cl::Hidden,
    cl::desc("A 32-bit value's two halves through the same 16-bit bitop "
             "with the same constant as one 32-bit bitop"));

namespace {

class AGXPairedHalfBitopsPass : public MachineFunctionPass {
public:
  static char ID;
  AGXPairedHalfBitopsPass() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Paired Half Bitops"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXPairedHalfBitopsPass::ID = 0;

INITIALIZE_PASS(AGXPairedHalfBitopsPass, DEBUG_TYPE, "AGX Paired Half Bitops",
                false, false)

FunctionPass *llvm::createAGXPairedHalfBitopsPass() {
  return new AGXPairedHalfBitopsPass();
}

/// The 16-bit constant R holds (MOV_IMM16 C), if any.
static std::optional<uint64_t> imm16(Register R,
                                     const MachineRegisterInfo &MRI) {
  const MachineInstr *Def = R.isVirtual() ? MRI.getVRegDef(R) : nullptr;
  if (!Def || Def->getOpcode() != AGX::MOV_IMM16 || !Def->getOperand(1).isImm())
    return std::nullopt;
  return uint64_t(Def->getOperand(1).getImm()) & 0xffff;
}

bool AGXPairedHalfBitopsPass::runOnMachineFunction(MachineFunction &MF) {
  if (!AGXPairHalfBitops || skipFunction(MF.getFunction()))
    return false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // (the 32-bit value, the operation, the constant) -> the lo and hi ops
    struct Pair {
      MachineInstr *Half[2] = {nullptr, nullptr};
      MachineInstr *Copy[2] = {nullptr, nullptr};
      MachineInstr *First = nullptr;
    };
    DenseMap<std::tuple<Register, int64_t, uint64_t>, Pair> Pairs;
    SmallVector<std::tuple<Register, int64_t, uint64_t>, 8> Order;
    for (MachineInstr &MI : MBB) {
      // (BITOP_I16 tt, mods, src0, mods, src1)
      if (MI.getOpcode() != AGX::BITOP_I16 || MI.getOperand(2).getImm() ||
          MI.getOperand(4).getImm() || !MI.getOperand(3).isReg() ||
          !MI.getOperand(5).isReg() || MI.getOperand(3).getSubReg() ||
          MI.getOperand(5).getSubReg())
        continue;
      Register Src = MI.getOperand(3).getReg();
      std::optional<uint64_t> C = imm16(MI.getOperand(5).getReg(), MRI);
      MachineInstr *Copy = Src.isVirtual() ? MRI.getVRegDef(Src) : nullptr;
      if (!C || !Copy || !Copy->isCopy() || Copy->getParent() != &MBB ||
          !MRI.hasOneNonDBGUse(Src))
        continue;
      const MachineOperand &From = Copy->getOperand(1);
      if (!From.getReg().isVirtual() ||
          MRI.getRegClass(From.getReg()) != &AGX::GPR32RegClass)
        continue;
      unsigned Sub = From.getSubReg();
      int Idx = Sub == AGX::lo16 ? 0 : Sub == AGX::hi16 ? 1 : -1;
      if (Idx < 0)
        continue;
      auto Key = std::make_tuple(From.getReg(), MI.getOperand(1).getImm(), *C);
      auto [It, New] = Pairs.try_emplace(Key);
      if (New)
        Order.push_back(Key);
      if (It->second.Half[Idx])
        continue;
      It->second.Half[Idx] = &MI;
      It->second.Copy[Idx] = Copy;
      if (!It->second.First)
        It->second.First = &MI;
    }
    for (auto &Key : Order) {
      Pair &P = Pairs[Key];
      if (!P.Half[0] || !P.Half[1])
        continue;
      auto [V, TT, C] = Key;
      // (at the first of the two: both read v and the constant)
      MachineInstr *First = P.First;
      const DebugLoc &DL = First->getDebugLoc();
      const TargetRegisterClass *RC = MRI.getRegClass(V);
      Register K = MRI.createVirtualRegister(RC);
      Register W = MRI.createVirtualRegister(RC);
      BuildMI(MBB, First, DL, TII->get(AGX::MOV_IMM32), K)
          .addImm(int64_t((C << 16) | C));
      BuildMI(MBB, First, DL, TII->get(AGX::BITOP_I32), W)
          .addImm(TT)
          .addImm(0)
          .addReg(V)
          .addImm(0)
          .addReg(K);
      for (int I = 0; I < 2; ++I) {
        MachineInstr *H = P.Half[I];
        BuildMI(MBB, H, H->getDebugLoc(), TII->get(TargetOpcode::COPY),
                H->getOperand(0).getReg())
            .addReg(W, {}, I == 0 ? AGX::lo16 : AGX::hi16);
        H->eraseFromParent();
        P.Copy[I]->eraseFromParent();
      }
      Changed = true;
    }
  }
  return Changed;
}
