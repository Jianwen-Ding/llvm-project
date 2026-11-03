//===-- AGXSinkRemat.cpp - Put cheap constants next to their uses ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A constant that lives across other blocks holds registers all the way, and on
// AGX registers are threads: the most registers any point of a kernel holds set
// how many of its SIMD groups fit on a core. This gives each block that uses
// such a constant its own move, right before the first use (the greedy allocator
// does this only once it would otherwise spill).
//
// Blocks in a deeper loop than the definition keep using it, as the move would
// run many times over: the allocator can still rematerialize those when the
// registers run out. So do terminators: AGX_ELSE reads its operands for threads
// that the instructions of its block do not run for (see AGXHoistFromElseFlow).
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "agx-sink-remat"

namespace {

class AGXSinkRemat : public MachineFunctionPass {
public:
  static char ID;
  AGXSinkRemat() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Sink Rematerializable"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXSinkRemat::ID = 0;

INITIALIZE_PASS_BEGIN(AGXSinkRemat, DEBUG_TYPE, "AGX Sink Rematerializable",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AGXSinkRemat, DEBUG_TYPE, "AGX Sink Rematerializable",
                    false, false)

FunctionPass *llvm::createAGXSinkRematPass() { return new AGXSinkRemat(); }

bool AGXSinkRemat::runOnMachineFunction(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const MachineLoopInfo &MLI =
      getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() != AGX::MOV_IMM16 && MI.getOpcode() != AGX::MOV_IMM32)
        continue;
      Register Reg = MI.getOperand(0).getReg();
      if (!Reg.isVirtual())
        continue;
      unsigned Depth = MLI.getLoopDepth(&MBB);
      // The uses of each other block, in the order of the block's instructions
      // found by walking them.
      MapVector<MachineBasicBlock *, SmallVector<MachineOperand *, 4>> Uses;
      for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
        MachineInstr &User = *MO.getParent();
        MachineBasicBlock *UB = User.getParent();
        if (User.isPHI() || User.isTerminator() || UB == &MBB || MLI.getLoopDepth(UB) > Depth)
          continue;
        Uses[UB].push_back(&MO);
      }
      for (auto &[UB, Ops] : Uses) {
        SmallPtrSet<const MachineInstr *, 8> Users;
        for (MachineOperand *MO : Ops)
          Users.insert(MO->getParent());
        auto First = UB->getFirstNonPHI();
        while (!Users.count(&*First))
          ++First;
        Register New = MRI.createVirtualRegister(MRI.getRegClass(Reg));
        TII.reMaterialize(*UB, First, New, 0, MI);
        for (MachineOperand *MO : Ops)
          MO->setReg(New);
        Changed = true;
      }
      if (MRI.use_nodbg_empty(Reg)) {
        MI.eraseFromParent();
        Changed = true;
      }
    }
  }
  return Changed;
}
