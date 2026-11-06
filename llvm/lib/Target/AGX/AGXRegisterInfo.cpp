//===-- AGXRegisterInfo.cpp - AGX register information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "AGX.h"
#include "AGXFrameLowering.h"
#include "AGXSubtarget.h"
#include "AGXISelLowering.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "AGXGenRegisterInfo.inc"

AGXRegisterInfo::AGXRegisterInfo(const AGXSubtarget &ST)
    : AGXGenRegisterInfo(AGX::R1), ST(ST) {}

// (16 under the call-tree cap of 72: 48 left a function with calls almost nothing free; Cycles 1-5% better)
static cl::opt<unsigned> CalleeSaved(
    "agx-callee-saved", cl::init(16), cl::Hidden,
    cl::desc("How many registers, the highest every function may use, called "
             "functions keep"));

static cl::opt<unsigned> CalleeSavedStride(
    "agx-callee-saved-stride", cl::init(0), cl::Hidden,
    cl::desc("Alternate this many kept and free registers (0: all together)"));

namespace {
/// The registers called functions keep: the top -agx-callee-saved of what every
/// function may use, so that a cap on registers leaves some to keep.
struct CalleeSavedSet {
  SmallVector<MCPhysReg, 128> Save;
  SmallVector<uint32_t, 64> Mask;
};
} // namespace

static const CalleeSavedSet &calleeSavedSet(const AGXRegisterInfo &TRI,
                                            const AGXSubtarget &ST) {
  // (the options are process-wide; thread-safe initialization, as kernels compile
  // on several threads)
  static const CalleeSavedSet Set = [&] {
    CalleeSavedSet R;
    unsigned Top = ST.getMaxGPRsOfAllFunctions();
    BitVector Units(TRI.getNumRegUnits());
    // From the top down; with -agx-callee-saved-stride=S, S kept then S free in
    // turn, as AMDGPU's convention keeps both kinds at every occupancy.
    unsigned Left = CalleeSaved;
    for (unsigned I = Top; I > 37 && Left; --I) {
      unsigned FromTop = Top - I;
      if (CalleeSavedStride && (FromTop / CalleeSavedStride) % 2)
        continue;
      MCRegister Reg = AGX::GPR32RegClass.getRegister(I - 1);
      R.Save.push_back(Reg);
      --Left;
      for (MCRegUnit U : TRI.regunits(Reg))
        Units.set(static_cast<unsigned>(U));
    }
    R.Save.push_back(0);
    // preserved: every register all of whose units are kept
    R.Mask.assign(TRI.getNumRegs() / 32 + 1, 0);
    for (unsigned Reg = 1; Reg < TRI.getNumRegs(); ++Reg) {
      bool All = true, Any = false;
      for (MCRegUnit U : TRI.regunits(MCRegister::from(Reg))) {
        Any = true;
        All &= Units.test(static_cast<unsigned>(U));
      }
      if (Any && All)
        R.Mask[Reg / 32] |= 1u << (Reg % 32);
    }
    return R;
  }();
  return Set;
}

const MCPhysReg *
AGXRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  // A kernel has no caller to keep anything for.
  if (!MF || MF->getFunction().getCallingConv() == CallingConv::AGX_KERNEL ||
      !agxExperimentalLeafCallsEnabled() || !CalleeSaved)
    return CSR_AGX_NoRegs_SaveList;
  return calleeSavedSet(*this, ST).Save.data();
}

const uint32_t *AGXRegisterInfo::getCallPreservedMask(const MachineFunction &,
                                                      CallingConv::ID) const {
  if (!CalleeSaved)
    return CSR_AGX_NoRegs_RegMask;
  return calleeSavedSet(*this, ST).Mask.data();
}

static cl::opt<bool> ReserveCallRegisters(
    "agx-reserve-call-registers", cl::init(false), cl::Hidden,
    cl::desc("Keep r2 to r36 out of register allocation in code with calls"));

static cl::opt<unsigned> SchedTargetGPRs(
    "agx-sched-target-gprs", cl::init(0), cl::Hidden,
    cl::desc("Registers the pre-RA scheduler tries to stay under (0: what is "
             "allocatable)"));

static cl::opt<unsigned> CallTreeSchedTargetGPRs(
    "agx-call-tree-sched-target-gprs", cl::init(48), cl::Hidden,
    cl::desc("-agx-sched-target-gprs for functions with real calls or called "
             "(agx-call-tree), where it is not set"));

unsigned AGXRegisterInfo::getRegPressureSetLimit(const MachineFunction &MF,
                                                 unsigned Idx) const {
  unsigned Limit = AGXGenRegisterInfo::getRegPressureSetLimit(MF, Idx);
  // RegisterClassInfo takes the reserved registers, which include those above
  // the cap, off this: take the difference from the cap only.
  unsigned Cap = ST.getMaxGPRs(MF.getFunction());
  unsigned Target = SchedTargetGPRs;
  if (!Target && MF.getFunction().hasFnAttribute("agx-call-tree"))
    Target = CallTreeSchedTargetGPRs;
  if (Target && Target < Cap)
    Limit -= 2 * (Cap - Target);
  return Limit;
}

BitVector AGXRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  auto ReserveWithAliases = [&](MCRegister Reg) {
    for (MCRegAliasIterator AI(Reg, this, /*IncludeSelf=*/true); AI.isValid();
         ++AI)
      Reserved.set((*AI).id());
  };

  // r0l is the execution mask nesting counter. r0h is left alone so that
  // r0 is never clobbered as a whole.
  ReserveWithAliases(AGX::R0);
  bool UsesCallABI = MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL;
  if (!UsesCallABI)
    for (const BasicBlock &BB : MF.getFunction())
      for (const Instruction &I : BB)
        if (const auto *CB = dyn_cast<CallBase>(&I)) {
          const Function *Callee = CB->getCalledFunction();
          UsesCallABI |= !Callee || !Callee->isIntrinsic();
        }
  if (agxExperimentalLeafCallsEnabled() && UsesCallABI) {
    // r1 is the link. r30 up hold the links of the functions with calls, by
    // call depth: this function must keep those of its callers and its own.
    // The rest is free, as no register survives a call (the arguments and
    // results are uses and definitions of the call and the return).
    ReserveWithAliases(AGX::R1);
    unsigned Links = AGXFrameLowering::getCallDepth(MF);
    if (ReserveCallRegisters) {
      Links = 7;
      for (unsigned I = 2; I < 30; ++I)
        ReserveWithAliases(AGX::GPR32RegClass.getRegister(I));
    }
    for (unsigned I = 30; I < 30 + Links && I <= 36; ++I)
      ReserveWithAliases(AGX::GPR32RegClass.getRegister(I));
  }

  // Registers beyond the subtarget's register file.
  for (unsigned I = ST.getMaxGPRs(MF.getFunction()); I < 128; ++I)
    ReserveWithAliases(AGX::GPR32RegClass.getRegister(I));

  // Uniform registers are never allocated.
  for (MCRegister Reg : AGX::UGPR16RegClass)
    ReserveWithAliases(Reg);
  return Reserved;
}

bool AGXRegisterInfo::isUniformClass(const TargetRegisterClass *RC) {
  return AGX::UGPR16RegClass.hasSubClassEq(RC) ||
         AGX::UGPR32RegClass.hasSubClassEq(RC) ||
         AGX::UGPR64RegClass.hasSubClassEq(RC);
}

const TargetRegisterClass *
AGXRegisterInfo::getGPRClassForSize(unsigned Bits) const {
  switch (Bits) {
  case 16:
    return &AGX::GPR16RegClass;
  case 32:
    return &AGX::GPR32RegClass;
  case 64:
    return &AGX::GPR32x2RegClass;
  case 96:
    return &AGX::GPR96RegClass;
  case 128:
    return &AGX::GPR128RegClass;
  default:
    return nullptr;
  }
}

Register AGXRegisterInfo::getFrameRegister(const MachineFunction &) const {
  // Stack accesses address the thread's stack directly; there is no frame
  // register.
  return Register();
}

bool AGXRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                          int SPAdj, unsigned FIOperandNum,
                                          RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getMF();
  MachineOperand &FIOp = MI.getOperand(FIOperandNum);
  int FI = FIOp.getIndex();
  int64_t Offset = MF.getFrameInfo().getObjectOffset(FI);
  if (agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL) {
    // After the kernel's region, one per depth (depth 0: an entry point of a
    // module without kernels, at offset zero)
    Offset += AGXFrameLowering::getCalleeFrameOffset(MF);
  }
  if (Offset < 0 || Offset > 0x7fff)
    report_fatal_error("AGX stack frame too large");
  FIOp.ChangeToImmediate(Offset);
  return false;
}
