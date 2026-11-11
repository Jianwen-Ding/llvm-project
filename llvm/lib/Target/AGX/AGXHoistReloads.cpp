//===-- AGXHoistReloads.cpp - Issue spill reloads ahead of their uses -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The register allocator puts a spill reload right before the instruction
// that needs the value, into a register that an instruction just before often
// read last: the reload cannot move up (it would overwrite that value), and
// its latency (an L1 hit, ~46 cycles on G14C; the playground's stack and
// waits measurements) is exposed: a third of Blender Cycles' reloads are read
// within 5 instructions. A stack load has no register operand, so after
// allocation it can be issued earlier whenever a register is free from the
// earlier point to the value's last read: this pass moves each reload of a
// spill slot up its block, by up to -agx-hoist-reloads-distance instructions,
// keeping its register if it is free there and otherwise giving the value
// another free register of its class (renaming its reads). It stops at
// instructions with side effects (execution mask changes, calls, barriers),
// terminators and stores to the same slot. Runs after allocation, before
// frame lowering, so that a callee-saved register it takes is saved; it
// takes only callee-saved registers the function modifies already.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "agx-hoist-spill-reloads"

static cl::opt<bool> AGXHoistReloads(
    "agx-hoist-reloads", cl::init(true), cl::Hidden,
    cl::desc("Issue spill reloads earlier in their block, in a free "
             "register"));

static cl::opt<unsigned> AGXHoistReloadsDistance(
    "agx-hoist-reloads-distance", cl::init(24), cl::Hidden,
    cl::desc("Move a spill reload up by at most this many instructions"));

static cl::opt<bool> AGXHoistReloadsVerify(
    "agx-hoist-reloads-verify", cl::init(false), cl::Hidden,
    cl::desc("Check the updated liveness against a recomputation after each "
             "move (testing)"));

static cl::opt<unsigned> AGXHoistReloadsMaxRange(
    "agx-hoist-reloads-max-range", cl::init(256), cl::Hidden,
    cl::desc("Give a reload another register only if its value is read "
             "within this many instructions (compile time)"));

static cl::opt<unsigned> AGXHoistReloadsMaxBlock(
    "agx-hoist-reloads-max-block", cl::init(4096), cl::Hidden,
    cl::desc("Skip blocks of more instructions (compile time)"));

namespace {

class AGXHoistReloadsPass : public MachineFunctionPass {
public:
  static char ID;
  AGXHoistReloadsPass() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Hoist Reloads"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const TargetRegisterInfo *TRI = nullptr;
  const AGXInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineFrameInfo *MFI = nullptr;
  BitVector CSR;      // register units of callee-saved registers not modified
  bool runOnBlock(MachineBasicBlock &MBB);
};

} // namespace

char AGXHoistReloadsPass::ID = 0;

INITIALIZE_PASS(AGXHoistReloadsPass, DEBUG_TYPE, "AGX Hoist Reloads", false,
                false)

FunctionPass *llvm::createAGXHoistReloadsPass() {
  return new AGXHoistReloadsPass();
}

/// The register units \p MI reads or writes.
static BitVector refsOf(const MachineInstr &MI, const TargetRegisterInfo *TRI) {
  BitVector B(TRI->getNumRegUnits());
  for (const MachineOperand &O : MI.operands()) {
    if (O.isRegMask())
      B.set();
    else if (O.isReg() && O.getReg())
      for (MCRegUnit U : TRI->regunits(O.getReg().asMCReg()))
        B.set(static_cast<unsigned>(U));
  }
  return B;
}

static BitVector unitsOf(MCRegister R, const TargetRegisterInfo *TRI) {
  BitVector B(TRI->getNumRegUnits());
  for (MCRegUnit U : TRI->regunits(R))
    B.set(static_cast<unsigned>(U));
  return B;
}

bool AGXHoistReloadsPass::runOnBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  // Liveness before each instruction and the units each references, updated
  // after every move.
  SmallVector<MachineInstr *, 0> Instrs;
  std::vector<BitVector> LiveBefore, Refs;
  unsigned NumUnits = TRI->getNumRegUnits();
  auto Recompute = [&]() {
    Instrs.clear();
    for (MachineInstr &MI : MBB)
      if (!MI.isDebugInstr())
        Instrs.push_back(&MI);
    LiveBefore.assign(Instrs.size() + 1, BitVector(NumUnits));
    Refs.clear();
    for (MachineInstr *MI : Instrs)
      Refs.push_back(refsOf(*MI, TRI));
    LiveRegUnits Live(*TRI);
    Live.addLiveOuts(MBB);
    auto Bits = [&](BitVector &B) { B = Live.getBitVector(); };
    Bits(LiveBefore[Instrs.size()]);
    for (unsigned I = Instrs.size(); I-- > 0;) {
      Live.stepBackward(*Instrs[I]);
      Bits(LiveBefore[I]);
    }
  };
  if (MBB.size() > AGXHoistReloadsMaxBlock)
    return false;
  Recompute();

  for (unsigned M = 0; M < Instrs.size(); ++M) {
    MachineInstr &MI = *Instrs[M];
    int FI;
    Register D = TII->isLoadFromStackSlot(MI, FI);
    if (!D || !D.isPhysical() || !MFI->isSpillSlotObjectIndex(FI))
      continue;
    // a reload with no register operand but its destination
    if (llvm::any_of(MI.operands(), [&](const MachineOperand &O) {
          return O.isReg() && (O.getReg() != D || O.isImplicit());
        }))
      continue;
    BitVector DUnits = unitsOf(D.asMCReg(), TRI);

    // The earliest point P it may move to (before Instrs[P]).
    unsigned P = M;
    while (P > 0 && M - P < AGXHoistReloadsDistance) {
      MachineInstr &Prev = *Instrs[P - 1];
      if (Prev.hasUnmodeledSideEffects() || Prev.isCall() ||
          Prev.isTerminator() || Prev.isPosition() || Prev.isInlineAsm())
        break;
      int SFI;
      if (Prev.mayStore() && (!TII->isStoreToStackSlot(Prev, SFI) || SFI == FI))
        break;
      --P;
    }
    if (P == M)
      continue;

    // The value's reads: until the block's end or an instruction that writes
    // a unit of D. Each must read D or one of its subregisters.
    unsigned E = M; // the last reader
    bool LiveOut = false, Bad = false;
    for (unsigned I = M + 1; I <= Instrs.size(); ++I) {
      if (I - M > AGXHoistReloadsMaxRange) {
        LiveOut = true; // (read further on: kept in D)
        break;
      }
      if (I == Instrs.size()) {
        LiveOut = LiveBefore[I].anyCommon(DUnits);
        break;
      }
      MachineInstr &U = *Instrs[I];
      bool Reads = false, Writes = false;
      for (const MachineOperand &O : U.operands()) {
        if (O.isRegMask() && O.clobbersPhysReg(D)) {
          Writes = true;
          continue;
        }
        if (!O.isReg() || !O.getReg() || !TRI->regsOverlap(O.getReg(), D))
          continue;
        if (O.getReg() != D && !TRI->isSubRegister(D, O.getReg()))
          Bad = true; // a register larger than D
        if (O.isDef())
          Writes = true;
        else
          Reads = true;
      }
      if (Bad)
        break;
      if (Reads)
        E = I;
      if (Writes) {
        // a partial redefinition leaves part of the value live
        for (const MachineOperand &O : U.defs())
          if (O.isReg() && TRI->regsOverlap(O.getReg(), D) && O.getReg() != D)
            Bad = true;
        break;
      }
    }
    if (Bad)
      continue;

    // Units used from a point to the value's last read: live before an
    // instruction there or referenced by it. Instrs[M] references D alone.
    // Fixed: from the reload to the last read (and live after it); Pre: from
    // the point to the reload, growing as the point moves up.
    bool CanRename = !LiveOut && E > M && E - M <= AGXHoistReloadsMaxRange;
    BitVector Fixed = LiveBefore[M];
    if (CanRename) {
      for (unsigned I = M + 1; I <= E; ++I) {
        Fixed |= LiveBefore[I];
        Fixed |= Refs[I];
      }
      Fixed |= LiveBefore[E + 1];
    }
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(D);
    BitVector Pre(NumUnits);
    unsigned AtD = M, AtR = M;
    MCRegister R;
    bool DFree = true;
    for (unsigned At = M; At-- > P;) {
      Pre |= LiveBefore[At];
      Pre |= Refs[At];
      if (DFree && !DUnits.anyCommon(Pre) && !DUnits.anyCommon(LiveBefore[M]))
        AtD = At;
      else
        DFree = false;
      if (!CanRename)
        continue;
      BitVector Used = Fixed;
      Used |= Pre;
      Used |= CSR;
      MCRegister Found;
      // the one chosen before if it is still free, else the first free one
      if (R && !unitsOf(R, TRI).anyCommon(Used))
        Found = R;
      for (MCPhysReg C : *RC) {
        if (Found)
          break;
        if (C == D || MRI->isReserved(C) || !MRI->isAllocatable(C))
          continue;
        if (!unitsOf(C, TRI).anyCommon(Used))
          Found = C;
      }
      if (!Found) {
        if (!DFree)
          break;
        continue;
      }
      R = Found;
      AtR = At;
    }
    MCRegister New;
    unsigned At;
    if (AtD <= AtR) {
      New = D.asMCReg();
      At = AtD;
    } else {
      New = R;
      At = AtR;
    }
    // Do not move a reload by a few instructions only.
    if (M - At < 2)
      continue;

    if (New != D.asMCReg()) {
      for (unsigned I = M + 1; I <= E; ++I)
        for (MachineOperand &O : Instrs[I]->operands()) {
          if (!O.isReg() || O.isDef() || !O.getReg() ||
              !TRI->regsOverlap(O.getReg(), D))
            continue;
          if (O.getReg() == D) {
            O.setReg(New);
          } else {
            unsigned Idx = TRI->getSubRegIndex(D, O.getReg());
            O.setReg(TRI->getSubReg(New, Idx));
          }
        }
      MI.getOperand(0).setReg(New);
    }
    MBB.splice(Instrs[At]->getIterator(), &MBB, MI.getIterator());
    Changed = true;
    // The block's liveness, updated: the reload now before Instrs[At], its
    // value in New from there to its last read (D no longer holds it after M).
    std::rotate(Instrs.begin() + At, Instrs.begin() + M, Instrs.begin() + M + 1);
    std::rotate(Refs.begin() + At, Refs.begin() + M, Refs.begin() + M + 1);
    // (before the reload: what was live before Instrs[At]; after it, before
    // each of the instructions it moved over, what was live there and New)
    std::rotate(LiveBefore.begin() + At, LiveBefore.begin() + M,
                LiveBefore.begin() + M + 1);
    LiveBefore[At] = LiveBefore[At + 1];
    // (each unit of the value live up to its own last read)
    SmallVector<std::pair<unsigned, unsigned>, 8> LastRead; // unit, position
    for (unsigned I = M + 1; I <= E && New != D.asMCReg(); ++I)
      for (const MachineOperand &O : Instrs[I]->operands())
        if (O.isReg() && O.readsReg() && O.getReg() &&
            TRI->regsOverlap(O.getReg(), New))
          for (MCRegUnit U : TRI->regunits(O.getReg().asMCReg())) {
            auto *It = llvm::find_if(LastRead, [&](auto &P) {
              return P.first == static_cast<unsigned>(U);
            });
            if (It == LastRead.end())
              LastRead.push_back({static_cast<unsigned>(U), I});
            else
              It->second = I;
          }
    BitVector NewUnits = unitsOf(New, TRI);
    if (New != D.asMCReg()) {
      for (unsigned I = M + 1; I <= E; ++I) {
        LiveBefore[I].reset(DUnits);
        Refs[I] = refsOf(*Instrs[I], TRI);
      }
      for (auto [U, Last] : LastRead)
        for (unsigned I = M + 1; I <= Last; ++I)
          LiveBefore[I].set(U);
    }
    // nothing between references New: its units live after the reload are
    // those live where the reload was
    BitVector After = LiveBefore[M + 1];
    After &= NewUnits;
    for (unsigned I = At + 1; I <= M; ++I)
      LiveBefore[I] |= After;
    Refs[At] = refsOf(MI, TRI);
    if (AGXHoistReloadsVerify) {
      auto Live = LiveBefore;
      auto R = Refs;
      Recompute();
      for (unsigned I = 0; I < Live.size(); ++I)
        if (Live[I] != LiveBefore[I]) {
          dbgs() << "live differs at " << I << " (At " << At << " M " << M
                 << " E " << E << " New " << printReg(New, TRI) << " D "
                 << printReg(D, TRI) << "):";
          for (unsigned U = 0; U < NumUnits; ++U)
            if (Live[I].test(U) != LiveBefore[I].test(U))
              dbgs() << " " << printRegUnit(static_cast<MCRegUnit>(U), TRI)
                     << (Live[I].test(U) ? "+" : "-");
          dbgs() << "\n  " << *Instrs[I];
          break;
        }
      if (Live != LiveBefore || R != Refs)
        report_fatal_error("agx-hoist-reloads: liveness out of date in " +
                           MBB.getParent()->getName());
    }
  }
  return Changed;
}

bool AGXHoistReloadsPass::runOnMachineFunction(MachineFunction &MF) {
  if (!AGXHoistReloads || skipFunction(MF.getFunction()))
    return false;
  MRI = &MF.getRegInfo();
  if (!MRI->tracksLiveness())
    return false;
  TRI = MF.getSubtarget().getRegisterInfo();
  TII = static_cast<const AGXInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MFI = &MF.getFrameInfo();
  // Callee-saved registers the function does not modify yet: taking one would
  // add a save and a restore.
  CSR = BitVector(TRI->getNumRegUnits());
  if (const MCPhysReg *CS = MRI->getCalleeSavedRegs())
    for (; *CS; ++CS)
      if (!MRI->isPhysRegModified(*CS))
        for (MCRegUnit U : TRI->regunits(*CS))
          CSR.set(static_cast<unsigned>(U));
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= runOnBlock(MBB);
  return Changed;
}
