//===-- AGXRegisterInfo.cpp - AGX register information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallSet.h"
#include "AGX.h"
#include "AGXFrameLowering.h"
#include "AGXSubtarget.h"
#include "AGXISelLowering.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
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

static cl::opt<bool> AGXCrossHalfHints(
    "agx-cross-half-hints", cl::init(true), cl::Hidden,
    cl::desc("Allocate a 16-bit destination of an instruction with two or "
             "more register sources outside the registers whose other half "
             "it reads, and those sources outside its register"));

/// An instruction that writes one 16-bit half and reads the other half of
/// the same register among two or more register sources waits for its
/// destination's earlier write (AGX Performance Simulator, cost.rs, measured
/// on hardware: 1.99 per step with no data dependence): a register allocated
/// so for no reason costs that wait. The registers \p VirtReg should not
/// take go last in its order (L15-E632).
bool AGXRegisterInfo::getRegAllocationHints(
    Register VirtReg, ArrayRef<MCPhysReg> Order,
    SmallSetVector<MCPhysReg, 16> &Hints, const MachineFunction &MF,
    const VirtRegMap *VRM, const LiveRegMatrix *Matrix) const {
  bool Hard = TargetRegisterInfo::getRegAllocationHints(VirtReg, Order, Hints,
                                                        MF, VRM, Matrix);
  if (!AGXCrossHalfHints || Hard || !VRM)
    return Hard;
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterClass *RC = MRI.getRegClass(VirtReg);
  bool Is16 = RC == &AGX::GPR16RegClass, Is32 = RC == &AGX::GPR32RegClass;
  if (!Is16 && !Is32)
    return false;
  // The physical register an operand reads or writes, if assigned yet.
  auto Phys = [&](const MachineOperand &MO) -> MCPhysReg {
    Register R = MO.getReg();
    MCPhysReg P = R.isPhysical() ? R.asMCReg().id()
                  : VRM->hasPhys(R) ? VRM->getPhys(R).id()
                                    : MCPhysReg(0);
    if (P && MO.getSubReg())
      P = getSubReg(P, MO.getSubReg());
    return P;
  };
  // The other half of a 16-bit register's 32-bit register.
  auto Sibling = [&](MCPhysReg H) -> MCPhysReg {
    for (unsigned Idx : {unsigned(AGX::lo16), unsigned(AGX::hi16)})
      if (MCPhysReg Super = getMatchingSuperReg(H, Idx, &AGX::GPR32RegClass))
        return getSubReg(Super, Idx == AGX::lo16 ? AGX::hi16 : AGX::lo16);
    return 0;
  };
  SmallSet<MCPhysReg, 8> Bad;
  for (const MachineInstr &MI : MRI.reg_nodbg_instructions(VirtReg)) {
    if (MI.getNumExplicitDefs() != 1 || !MI.getOperand(0).isReg())
      continue;
    const MachineOperand &Def = MI.getOperand(0);
    // (a 16-bit destination)
    if (Def.getReg().isVirtual()
            ? MRI.getRegClass(Def.getReg()) != &AGX::GPR16RegClass ||
                  Def.getSubReg()
            : !AGX::GPR16RegClass.contains(Def.getReg()))
      continue;
    SmallVector<const MachineOperand *, 4> Srcs;
    for (const MachineOperand &MO : MI.explicit_uses())
      if (MO.isReg() && MO.getReg())
        Srcs.push_back(&MO);
    if (Srcs.size() < 2)
      continue;
    if (Is16 && Def.getReg() == VirtReg && !Def.getSubReg()) {
      // (this the destination: not the other half of a source's register)
      for (const MachineOperand *S : Srcs)
        if (S->getReg() != VirtReg)
          if (MCPhysReg P = Phys(*S); P && AGX::GPR16RegClass.contains(P))
            if (MCPhysReg O = Sibling(P))
              Bad.insert(O);
      continue;
    }
    MCPhysReg D = Phys(Def);
    if (!D || !AGX::GPR16RegClass.contains(D))
      continue;
    // (this a source: not in the destination's register's other half)
    for (const MachineOperand *S : Srcs) {
      if (S->getReg() != VirtReg)
        continue;
      if (Is16 && !S->getSubReg()) {
        if (MCPhysReg O = Sibling(D))
          Bad.insert(O);
      } else if (Is32 && S->getSubReg()) {
        // (its half other than the destination's: the register holding D)
        if (MCPhysReg Super =
                getMatchingSuperReg(D, AGX::lo16, &AGX::GPR32RegClass)) {
          if (S->getSubReg() != AGX::lo16)
            Bad.insert(Super);
        } else if (MCPhysReg Super = getMatchingSuperReg(
                       D, AGX::hi16, &AGX::GPR32RegClass)) {
          if (S->getSubReg() != AGX::hi16)
            Bad.insert(Super);
        }
      }
    }
  }
  if (Bad.empty())
    return false;
  // (the copy hints first, then the order, the bad registers dropped from
  // the preference and so tried last)
  // (but not at the price of a register no other value holds yet, which
  // could cost a SIMD-group: among the registers in use, the good ones
  // first, then the bad ones, then the order as it was)
  auto InUse = [&](MCPhysReg R) {
    if (!Matrix)
      return true;
    MCPhysReg Super = R;
    if (AGX::GPR16RegClass.contains(R))
      for (unsigned Idx : {unsigned(AGX::lo16), unsigned(AGX::hi16)})
        if (MCPhysReg S = getMatchingSuperReg(R, Idx, &AGX::GPR32RegClass))
          Super = S;
    return Matrix->isPhysRegUsed(Super);
  };
  SmallSetVector<MCPhysReg, 16> Kept;
  for (MCPhysReg H : Hints)
    if (!Bad.count(H))
      Kept.insert(H);
  for (MCPhysReg R : Order)
    if (!Bad.count(R) && InUse(R))
      Kept.insert(R);
  for (MCPhysReg R : Order)
    if (Bad.count(R) && InUse(R))
      Kept.insert(R);
  for (MCPhysReg R : Order)
    Kept.insert(R);
  Hints = std::move(Kept);
  return false;
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
  // (plus the constant AGXTargetLowering::finalizeLowering put in its flags)
  int64_t Offset =
      MF.getFrameInfo().getObjectOffset(FI) + FIOp.getTargetFlags();
  if (agxExperimentalLeafCallsEnabled() &&
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL) {
    // After the kernel's region, one per depth (depth 0: an entry point of a
    // module without kernels, at offset zero)
    Offset += AGXFrameLowering::getCalleeFrameOffset(MF);
  }
  if (MI.getOpcode() == AGX::FRAME_ADDR_OFF) {
    // an object's address plus a constant: the move of the sum
    Offset += MI.getOperand(FIOperandNum + 1).getImm();
    MI.removeOperand(FIOperandNum + 1);
    MI.setDesc(MF.getSubtarget().getInstrInfo()->get(AGX::FRAME_ADDR));
  }
  if (Offset < 0 || Offset > 0x7fff)
    report_fatal_error("AGX stack frame too large");
  // An ALU source immediate has 8 bits: a larger offset of an iadd's frame
  // index (finalizeLowering's fold) is moved into the destination first,
  // when the destination is not the other source.
  if (MI.getOpcode() == AGX::IADD_I32 && Offset > 255) {
    Register Dst = MI.getOperand(0).getReg();
    unsigned Other = FIOperandNum == 2 ? 4 : 2;
    if (MI.getOperand(Other).isReg() && MI.getOperand(Other).getReg() == Dst)
      report_fatal_error("AGX: a frame offset above 255 in an iadd that "
                         "overwrites its other source");
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    BuildMI(*MI.getParent(), II, MI.getDebugLoc(), TII->get(AGX::FRAME_ADDR),
            Dst)
        .addImm(Offset);
    FIOp.ChangeToRegister(Dst, /*isDef=*/false);
    return false;
  }
  FIOp.ChangeToImmediate(Offset, /*TargetFlags=*/0);
  return false;
}
