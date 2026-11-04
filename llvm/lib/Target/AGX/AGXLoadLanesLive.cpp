//===-- AGXLoadLanesLive.cpp - Keep a load's destination whole until used -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A device or threadgroup load writes its destination registers when the
// memory answers, and the first instruction that reads any of them waits for
// it (AGXInsertWaits). With subregister liveness the allocator sees a lane of
// the destination that nothing reads as free right after the load, and may
// give it to the next load: that load must then wait for the first to finish
// (its write would otherwise race the first one's), which serializes the
// memory latencies (a 64-bit load whose high word is unused, then a 16-bit
// load into that word: +470 cycles). The first instruction that reads the
// loaded value (through the copies of its lanes) gets an implicit use of the
// whole destination, so that its lanes stay allocated until the wait.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-load-lanes-live"

static cl::opt<bool> AGXLoadLanesLive(
    "agx-keep-load-lanes", cl::init(true), cl::Hidden,
    cl::desc("Keep every lane of a load's destination allocated until the "
             "load's first use"));

namespace {

class AGXLoadLanes : public MachineFunctionPass {
public:
  static char ID;
  AGXLoadLanes() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Load Lanes Live"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXLoadLanes::ID = 0;

INITIALIZE_PASS(AGXLoadLanes, DEBUG_TYPE, "AGX Load Lanes Live", false, false)

FunctionPass *llvm::createAGXLoadLanesLivePass() { return new AGXLoadLanes(); }

bool AGXLoadLanes::runOnMachineFunction(MachineFunction &MF) {
  if (!AGXLoadLanesLive || !MF.getSubtarget().enableSubRegLiveness())
    return false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!MI.mayLoad() || MI.getNumDefs() == 0)
        continue;
      for (const MachineOperand &Def : MI.defs()) {
        if (!Def.isReg() || !Def.getReg().isVirtual() || Def.getSubReg())
          continue;
        Register R = Def.getReg();
        // a destination of more than one lane
        if (TRI->getRegSizeInBits(*MRI.getRegClass(R)) <= 16)
          continue;
        // the readers of the value: its own and its copies' users
        SmallPtrSet<const MachineInstr *, 8> Readers;
        SmallVector<Register, 4> Work{R};
        SmallSet<Register, 8> Seen;
        Seen.insert(R);
        while (!Work.empty()) {
          Register V = Work.pop_back_val();
          for (MachineInstr &U : MRI.use_nodbg_instructions(V)) {
            if (U.getParent() != &MBB)
              continue;
            // (copies and the pseudo instructions that assemble registers
            // pass the value on)
            if ((U.isCopy() || U.isRegSequence() || U.isInsertSubreg() ||
                 U.isSubregToReg()) &&
                U.getOperand(0).getReg().isVirtual()) {
              if (Seen.insert(U.getOperand(0).getReg()).second)
                Work.push_back(U.getOperand(0).getReg());
              continue;
            }
            Readers.insert(&U);
          }
        }
        if (Readers.empty())
          continue;
        // the first of them after the load
        MachineInstr *First = nullptr;
        for (auto It = std::next(MI.getIterator()); It != MBB.end(); ++It)
          if (Readers.contains(&*It)) {
            First = &*It;
            break;
          }
        if (!First || First->isPHI() || First->isMetaInstruction())
          continue;
        bool Has = llvm::any_of(First->uses(), [&](const MachineOperand &O) {
          return O.isReg() && O.getReg() == R && !O.getSubReg();
        });
        if (Has)
          continue;
        First->addOperand(MF, MachineOperand::CreateReg(R, /*isDef=*/false,
                                                       /*isImp=*/true));
        Changed = true;
      }
    }
  }
  return Changed;
}
