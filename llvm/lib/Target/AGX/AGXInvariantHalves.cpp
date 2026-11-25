//===-- AGXInvariantHalves.cpp - Loop-invariant low halves of a loop -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A 32-bit value made in a loop of a 16-bit value under a zero half (x << 16,
// a bfloat widened: L15-E559),
//
//   %v = REG_SEQUENCE %zero, lo16, %x, hi16
//
// is a value carried by the loop whose low half the preheader zeroes once:
//
//   preheader:  %init = MOV_IMM32 0
//   loop:       %phi = PHI %init, %preheader, %v, %loop
//               %v = INSERT_SUBREG %phi, %x, hi16
//
// so that each iteration writes only the high half. A REG_SEQUENCE instead
// leaves the register allocator one zero half to share: every other such
// value took a move of it into its own low half each iteration
// (bf16_dot_rows: four a step).
//
// Single-block loops only (the block both header and latch), where the
// value of the previous iteration is the PHI's only other incoming value.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-invariant-halves"

static cl::opt<bool> AGXInvariantHalves(
    "agx-carried-zero-halves", cl::init(true), cl::Hidden,
    cl::desc("Carry a loop's 32-bit values of a 16-bit value under a zero "
             "half through the loop, the zero half set once"));

namespace {

class AGXInvariantHalvesPass : public MachineFunctionPass {
public:
  static char ID;
  AGXInvariantHalvesPass() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Invariant Halves"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXInvariantHalvesPass::ID = 0;

INITIALIZE_PASS_BEGIN(AGXInvariantHalvesPass, DEBUG_TYPE,
                      "AGX Invariant Halves", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AGXInvariantHalvesPass, DEBUG_TYPE, "AGX Invariant Halves",
                    false, false)

FunctionPass *llvm::createAGXInvariantHalvesPass() {
  return new AGXInvariantHalvesPass();
}

/// Whether R is a 16-bit zero (MOV_IMM16 0).
static bool isZero16(Register R, const MachineRegisterInfo &MRI) {
  const MachineInstr *Def = R.isVirtual() ? MRI.getVRegDef(R) : nullptr;
  return Def && Def->getOpcode() == AGX::MOV_IMM16 &&
         Def->getNumOperands() >= 2 && Def->getOperand(1).isImm() &&
         Def->getOperand(1).getImm() == 0;
}

bool AGXInvariantHalvesPass::runOnMachineFunction(MachineFunction &MF) {
  if (!AGXInvariantHalves || skipFunction(MF.getFunction()))
    return false;
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  bool Changed = false;
  for (MachineLoop *L : MLI.getLoopsInPreorder()) {
    if (L->getNumBlocks() != 1)
      continue;
    MachineBasicBlock *H = L->getHeader();
    MachineBasicBlock *Pre = L->getLoopPreheader();
    if (!Pre || H->pred_size() != 2 || !H->isSuccessor(H))
      continue;
    for (MachineInstr &MI : make_early_inc_range(*H)) {
      if (!MI.isRegSequence() || MI.getNumOperands() != 5)
        continue;
      Register Dst = MI.getOperand(0).getReg();
      if (!Dst.isVirtual())
        continue;
      const TargetRegisterClass *RC = MRI.getRegClass(Dst);
      if (TRI->getRegSizeInBits(*RC) != 32)
        continue;
      // (operands: value, subregister index, value, subregister index)
      const MachineOperand *Lo = nullptr, *Hi = nullptr;
      for (unsigned I = 1; I + 1 < MI.getNumOperands(); I += 2) {
        unsigned Idx = MI.getOperand(I + 1).getImm();
        if (Idx == AGX::lo16)
          Lo = &MI.getOperand(I);
        else if (Idx == AGX::hi16)
          Hi = &MI.getOperand(I);
      }
      if (!Lo || !Hi || Lo->getSubReg() || !isZero16(Lo->getReg(), MRI))
        continue;
      const DebugLoc &DL = MI.getDebugLoc();
      Register Init = MRI.createVirtualRegister(RC);
      Register Phi = MRI.createVirtualRegister(RC);
      BuildMI(*Pre, Pre->getFirstTerminator(), DL, TII->get(AGX::MOV_IMM32),
              Init)
          .addImm(0);
      BuildMI(*H, H->begin(), DL, TII->get(TargetOpcode::PHI), Phi)
          .addReg(Init)
          .addMBB(Pre)
          .addReg(Dst)
          .addMBB(H);
      BuildMI(*H, MI, DL, TII->get(TargetOpcode::INSERT_SUBREG), Dst)
          .addReg(Phi)
          .addReg(Hi->getReg(), {}, Hi->getSubReg())
          .addImm(AGX::hi16);
      Register Zero = Lo->getReg();
      MI.eraseFromParent();
      // (the zero half, if nothing else reads it)
      if (MRI.use_nodbg_empty(Zero))
        MRI.getVRegDef(Zero)->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}
