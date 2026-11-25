//===-- AGXAdvancePointers.cpp - A loop's pointer stepped before its uses -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Loop strength reduction's device pointer of a single-block loop, read by
// accesses at immediate offsets and stepped by a constant after them,
//
//   %p = PHI %p0, %pre, %p.next, %loop
//   ... = DEVICE_LOAD ..., %p, c ...
//   %p.next = IADD_I64 0, %p, 0, S, 0
//
// is stepped first, each access at the new pointer less the step:
//
//   %p.next = IADD_I64 0, %p, 0, S, 0
//   ... = DEVICE_LOAD ..., %p.next, c - S ...
//
// The two pointers share a register; stepped last, the 64-bit add wrote the
// register its accesses had just read (agx_cost holds it until they have:
// col_sum2_dyn's 16 loads of a column before it, L15-E642). Only where every
// new offset fits its access's 16-bit element offset.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXRegisterInfo.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "agx-advance-pointers"

static cl::opt<bool> AGXPointersAdvanceFirst(
    "agx-loop-pointers-advance-first", cl::init(true), cl::Hidden,
    cl::desc("A single-block loop's device pointer stepped by a constant "
             "after its accesses at immediate offsets: stepped first, the "
             "offsets less the step"));

static cl::opt<unsigned> AGXPointersAdvanceMin(
    "agx-loop-pointers-advance-min", cl::init(16), cl::Hidden,
    cl::desc("Accesses a pointer needs for -agx-loop-pointers-advance-first "
             "(fewer wait less for the step's register than for its result)"));

static cl::opt<bool> AGXPointersAdvanceLoadsOnly(
    "agx-loop-pointers-advance-loads-only", cl::init(true), cl::Hidden,
    cl::desc("-agx-loop-pointers-advance-first only for pointers no store "
             "reads (a store reads its address late anyway: rowcopy_vec4)"));

static cl::opt<unsigned> AGXPointersAdvanceMinStep(
    "agx-loop-pointers-advance-min-step", cl::init(4096), cl::Hidden,
    cl::desc("Bytes a pointer must step for -agx-loop-pointers-advance-first "
             "(a column's rows apart; a row's few hundred bytes waited for "
             "the step's result longer: rowcopy_vec4, matmul_naive)"));

namespace {

class AGXAdvancePointersPass : public MachineFunctionPass {
public:
  static char ID;
  AGXAdvancePointersPass() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Advance Pointers"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXAdvancePointersPass::ID = 0;

INITIALIZE_PASS(AGXAdvancePointersPass, DEBUG_TYPE, "AGX Advance Pointers",
                false, false)

FunctionPass *llvm::createAGXAdvancePointersPass() {
  return new AGXAdvancePointersPass();
}

/// A device access's (format, base, offset) operand indices, if MI is one.
static std::optional<std::array<unsigned, 3>> accessOperands(
    const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AGX::DEVICE_LOAD_16:
  case AGX::DEVICE_LOAD_32:
  case AGX::DEVICE_LOAD_64:
  case AGX::DEVICE_LOAD_96:
  case AGX::DEVICE_LOAD_128:
  case AGX::DEVICE_LOAD_32_H16:
  case AGX::DEVICE_LOAD_64_H16:
    // (R, g, fmt, mask, A, O, sgn, shift)
    return std::array<unsigned, 3>{2, 4, 5};
  case AGX::DEVICE_STORE_16:
  case AGX::DEVICE_STORE_32:
  case AGX::DEVICE_STORE_64:
  case AGX::DEVICE_STORE_96:
  case AGX::DEVICE_STORE_128:
  case AGX::DEVICE_STORE_32_H16:
  case AGX::DEVICE_STORE_64_H16:
    // (g, fmt, mask, R, A, O, sgn, shift)
    return std::array<unsigned, 3>{1, 4, 5};
  default:
    return std::nullopt;
  }
}

/// The constant step of p.next = p + S (or p - S) that MI computes from P.
static std::optional<int64_t> stepOf(const MachineInstr &MI, Register P,
                                     const MachineRegisterInfo &MRI) {
  unsigned Opc = MI.getOpcode();
  bool Sub = Opc == AGX::ISUB_I64 || Opc == AGX::ISUB_I64_I32;
  if (!(Sub || Opc == AGX::IADD_I64 || Opc == AGX::IADD_I64_I32) ||
      MI.getNumOperands() < 6)
    return std::nullopt;
  // (dst, src0_mods, src0, src1_mods, src1, shift)
  if (!MI.getOperand(2).isReg() || MI.getOperand(2).getReg() != P ||
      MI.getOperand(2).getSubReg() || !MI.getOperand(1).isImm() ||
      MI.getOperand(1).getImm() || !MI.getOperand(3).isImm() ||
      MI.getOperand(3).getImm() || !MI.getOperand(5).isImm())
    return std::nullopt;
  int64_t Shift = MI.getOperand(5).getImm();
  const MachineOperand &Src = MI.getOperand(4);
  int64_t K;
  if (Src.isImm()) {
    K = Src.getImm();
    if (Opc == AGX::IADD_I64_I32 || Opc == AGX::ISUB_I64_I32)
      K = int64_t(uint32_t(K));
  } else if (Src.isReg() && Src.getReg().isVirtual() && !Src.getSubReg()) {
    const MachineInstr *Def = MRI.getVRegDef(Src.getReg());
    if (!Def || Def->getOpcode() != AGX::MOV_IMM32 ||
        !Def->getOperand(1).isImm() ||
        !(Opc == AGX::IADD_I64_I32 || Opc == AGX::ISUB_I64_I32))
      return std::nullopt;
    // (zero-extended: src1_mods 0)
    K = int64_t(uint32_t(Def->getOperand(1).getImm()));
  } else {
    return std::nullopt;
  }
  if (Shift < 0 || Shift > 4 || K > (int64_t(1) << 40) ||
      K < -(int64_t(1) << 40))
    return std::nullopt;
  K *= int64_t(1) << Shift;
  return Sub ? -K : K;
}

bool AGXAdvancePointersPass::runOnMachineFunction(MachineFunction &MF) {
  if (!AGXPointersAdvanceFirst || skipFunction(MF.getFunction()))
    return false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    if (!MBB.isSuccessor(&MBB))
      continue;
    for (MachineInstr &Phi : MBB.phis()) {
      if (Phi.getNumOperands() != 5)
        continue;
      Register P = Phi.getOperand(0).getReg();
      // (the value from this block)
      unsigned NextIdx = Phi.getOperand(2).getMBB() == &MBB   ? 1
                         : Phi.getOperand(4).getMBB() == &MBB ? 3
                                                              : 0;
      if (!NextIdx || !P.isVirtual() ||
          MRI.getRegClass(P) != &AGX::GPR64RegClass)
        continue;
      Register PN = Phi.getOperand(NextIdx).getReg();
      MachineInstr *Step = PN.isVirtual() ? MRI.getVRegDef(PN) : nullptr;
      if (!Step || Step->getParent() != &MBB)
        continue;
      std::optional<int64_t> S = stepOf(*Step, P, MRI);
      if (!S || !*S || uint64_t(*S < 0 ? -*S : *S) < AGXPointersAdvanceMinStep)
        continue;
      SmallVector<std::pair<MachineInstr *, int64_t>, 16> Accesses;
      bool OK = true;
      for (MachineOperand &U : MRI.use_nodbg_operands(P)) {
        MachineInstr *UI = U.getParent();
        if (UI == Step)
          continue;
        auto Ops = accessOperands(*UI);
        if (!Ops || UI->getParent() != &MBB || U.getSubReg() ||
            (AGXPointersAdvanceLoadsOnly && !UI->mayLoad()) ||
            UI->getOperandNo(&U) != (*Ops)[1] ||
            !UI->getOperand((*Ops)[2]).isImm() ||
            !UI->getOperand((*Ops)[0]).isImm()) {
          OK = false;
          break;
        }
        int64_t Unit = int64_t(1) << std::min<int64_t>(
                           2, UI->getOperand((*Ops)[0]).getImm());
        int64_t New = UI->getOperand((*Ops)[2]).getImm() - *S / Unit;
        if (*S % Unit || !isInt<16>(New)) {
          OK = false;
          break;
        }
        Accesses.push_back({UI, New});
      }
      if (!OK || Accesses.empty() || Accesses.size() < AGXPointersAdvanceMin)
        continue;
      // (before the first of them in the block, and its constant before it)
      MachineInstr *First = nullptr;
      for (MachineInstr &I : MBB) {
        if (any_of(Accesses, [&](auto &A) { return A.first == &I; })) {
          First = &I;
          break;
        }
        if (&I == Step)
          break;
      }
      if (!First)
        continue; // stepped before its accesses already
      if (Step->getOperand(4).isReg()) {
        MachineInstr *K = MRI.getVRegDef(Step->getOperand(4).getReg());
        if (K->getParent() == &MBB)
          K->moveBefore(First);
      }
      Step->moveBefore(First);
      for (auto [UI, New] : Accesses) {
        auto Ops = *accessOperands(*UI);
        UI->getOperand(Ops[1]).setReg(PN);
        UI->getOperand(Ops[2]).setImm(New);
      }
      MRI.clearKillFlags(P);
      MRI.clearKillFlags(PN);
      Changed = true;
    }
  }
  return Changed;
}
