//===-- AGXInsertWaits.cpp - Insert scoreboard waits ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Memory loads, stack loads and atomics write their results asynchronously,
// tracked by one of two scoreboard groups; `wait N` waits for group N only
// (measured on the M2 Max: ptx-re-blockers B34). Like SIInsertWaitcnts, this
// pass inserts a wait on a group before any instruction reads or overwrites a
// register with a pending result in it, both groups before barriers and at the
// end of a block with pending results whose successors could not take them
// over. At most sixteen requests are kept outstanding.
//
// A request's lanes are those of its issue: a result stays pending across
// control flow (exec mask changes included) into a later block whose
// predecessors all come before it in the layout (an if's side and join, not a
// loop's header), with the union of its predecessors' pending registers
// (L15-E428). A block that ends the function takes them only when it reads
// or overwrites one of them: its end would wait for the others, also after a
// side that waited for them.
//
// A write of part of a register pair carries an implicit use and def of the
// pair, which keep its other half live; they neither read nor write that
// half, so they make no request pending and wait for none (two loads into the
// halves of one pair run together).
//
// Each request goes to a group by where its result is first needed: requests
// needed at the same instruction share a group, and a request does not join a
// group that holds one needed earlier than itself (whose wait would then stall
// on it), so the wait for the first results does not wait for later ones.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-insert-waits"

namespace {

class AGXInsertWaits : public MachineFunctionPass {
public:
  static char ID;
  AGXInsertWaits() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Insert Waits"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXInsertWaits::ID = 0;

INITIALIZE_PASS(AGXInsertWaits, DEBUG_TYPE, "AGX Insert Waits", false, false)

FunctionPass *llvm::createAGXInsertWaitsPass() { return new AGXInsertWaits(); }

static cl::opt<bool> AGXCrossBlockWaits(
    "agx-cross-block-waits", cl::init(true), cl::Hidden,
    cl::desc("Keep load results pending into later blocks reached only from "
             "earlier ones instead of waiting at every block end"));

/// (B34 measured 16 independent loads in flight behind one wait; a request
/// beyond what the hardware tracks would stall its issue, not change a
/// result: L15-E437, from 8)
static cl::opt<unsigned> MaxPending(
    "agx-max-pending-loads", cl::init(16), cl::Hidden,
    cl::desc("Most load requests kept outstanding before a wait"));

static bool isBarrier(const MachineInstr &MI) {
  return MI.getOpcode() == AGX::THREADGROUP_BARRIER ||
         MI.getOpcode() == AGX::BLOCK_BARRIER ||
         MI.getOpcode() == AGX::MEMORY_BARRIER;
}

/// Whether \p MO is an implicit operand that only keeps the rest of a register
/// live around \p MI's write of part of it: a super-register of one of MI's
/// explicit definitions.
static bool isPartialWriteBookkeeping(const MachineInstr &MI,
                                      const MachineOperand &MO,
                                      const TargetRegisterInfo &TRI) {
  if (!MO.isReg() || !MO.isImplicit() || !MO.getReg())
    return false;
  for (const MachineOperand &D : MI.explicit_operands())
    if (D.isReg() && D.isDef() && D.getReg() &&
        TRI.isSuperRegister(D.getReg(), MO.getReg()))
      return true;
  return false;
}

/// Where the result of each asynchronous instruction of MBB is first needed:
/// the index of the first later instruction that touches one of its results,
/// is a barrier or ends the block.
static DenseMap<const MachineInstr *, unsigned>
firstUses(MachineBasicBlock &MBB, const TargetRegisterInfo &TRI,
          bool Carries) {
  DenseMap<const MachineInstr *, unsigned> Out;
  SmallVector<MachineInstr *, 64> Instrs;
  for (MachineInstr &MI : MBB)
    Instrs.push_back(&MI);
  for (unsigned I = 0; I < Instrs.size(); ++I) {
    const MachineInstr &MI = *Instrs[I];
    if (!AGXInstrInfo::isAsync(MI) || !MI.getNumExplicitDefs())
      continue;
    unsigned Use = Instrs.size();
    for (unsigned J = I + 1; J < Instrs.size() && Use == Instrs.size(); ++J) {
      const MachineInstr &UI = *Instrs[J];
      if (isBarrier(UI) || UI.isCall() || (UI.isTerminator() && !Carries)) {
        Use = J;
        break;
      }
      for (const MachineOperand &D : MI.defs()) {
        if (isPartialWriteBookkeeping(MI, D, TRI))
          continue;
        for (const MachineOperand &MO : UI.operands())
          if (MO.isReg() && MO.getReg() && MO.getReg() != AGX::R0L &&
              !isPartialWriteBookkeeping(UI, MO, TRI) &&
              TRI.regsOverlap(MO.getReg(), D.getReg()))
            Use = J;
      }
    }
    Out[&MI] = Use;
  }
  return Out;
}

bool AGXInsertWaits::runOnMachineFunction(MachineFunction &MF) {
  const AGXSubtarget &ST = MF.getSubtarget<AGXSubtarget>();
  const AGXInstrInfo &TII = *ST.getInstrInfo();
  const TargetRegisterInfo &TRI = *ST.getRegisterInfo();
  bool Changed = false;

  // Blocks that may pass pending results on (to successors later in the
  // layout, reached by no later block).
  DenseMap<const MachineBasicBlock *, unsigned> Index;
  for (const MachineBasicBlock &MBB : MF)
    Index[&MBB] = Index.size();
  auto Forward = [&](const MachineBasicBlock &S) {
    return AGXCrossBlockWaits && !S.pred_empty() &&
           llvm::all_of(S.predecessors(), [&](const MachineBasicBlock *P) {
             return Index.lookup(P) < Index.lookup(&S);
           });
  };
  auto Carries = [&](const MachineBasicBlock &P) {
    return !P.succ_empty() &&
           llvm::all_of(P.successors(), [&](const MachineBasicBlock *S) {
             return Index.lookup(S) > Index.lookup(&P) && Forward(*S);
           });
  };
  struct OutState {
    BitVector Pending[2];
    unsigned Count[2] = {0, 0};
  };
  DenseMap<const MachineBasicBlock *, OutState> Out;

  for (MachineBasicBlock &MBB : MF) {
    // Register allocation can insert a reload between CALL and its return
    // pop. RET has masked the returning lanes, so that reload would not run.
    // Keep the pop immediately after the call before assigning wait slots.
    for (auto I = MBB.begin(); I != MBB.end(); ++I) {
      if (I->getOpcode() != AGX::AGX_CALL)
        continue;
      auto Next = std::next(I);
      auto Pop = Next;
      while (Pop != MBB.end() && Pop->getOpcode() != AGX::POP_CALL_EXEC &&
             !Pop->isCall() && !Pop->isTerminator())
        ++Pop;
      if (Pop == MBB.end() || Pop->getOpcode() != AGX::POP_CALL_EXEC)
        report_fatal_error("AGX call has no return-mask pop in its block");
      if (Pop != Next) {
        MBB.splice(Next, &MBB, Pop);
        Changed = true;
      }
    }

    bool Carry = Carries(MBB);
    DenseMap<const MachineInstr *, unsigned> FirstUse =
        firstUses(MBB, TRI, Carry);
    struct Group {
      BitVector Pending;
      unsigned Count = 0;
      // The earliest first use of the group's pending results.
      unsigned Needed = ~0u;
    };
    Group Groups[2] = {{BitVector(TRI.getNumRegUnits())},
                       {BitVector(TRI.getNumRegUnits())}};

    auto Clear = [&](unsigned G) {
      Groups[G].Pending.reset();
      Groups[G].Count = 0;
      Groups[G].Needed = ~0u;
    };
    auto Wait = [&](MachineBasicBlock::iterator I, unsigned G) {
      BuildMI(MBB, I, DebugLoc(), TII.get(AGX::WAIT)).addImm(G);
      Clear(G);
      Changed = true;
    };
    auto Touches = [&](const MachineInstr &MI, unsigned G) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || MO.getReg() == AGX::R0L ||
            isPartialWriteBookkeeping(MI, MO, TRI))
          continue;
        for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
          if (Groups[G].Pending.test(static_cast<unsigned>(U)))
            return true;
      }
      return false;
    };
    auto Ends = [&](const MachineInstr &MI) {
      return isBarrier(MI) || MI.isCall() || (MI.isTerminator() && !Carry);
    };

    // The predecessors' pending results, needed at their first touch here.
    if (Forward(MBB)) {
      for (const MachineBasicBlock *P : MBB.predecessors()) {
        auto It = Out.find(P);
        if (It == Out.end())
          continue;
        for (unsigned G = 0; G < 2; ++G) {
          Groups[G].Pending |= It->second.Pending[G];
          Groups[G].Count = std::max(Groups[G].Count, It->second.Count[G]);
        }
      }
      for (unsigned G = 0; G < 2; ++G) {
        if (!Groups[G].Count)
          continue;
        unsigned K = 0;
        for (const MachineInstr &MI : MBB) {
          if (Ends(MI) || Touches(MI, G))
            break;
          ++K;
        }
        Groups[G].Needed = K;
      }
    }

    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == AGX::WAIT) {
        Clear(MI.getOperand(0).getImm() & 1);
        continue;
      }
      bool All = Ends(MI);
      for (unsigned G = 0; G < 2; ++G)
        if (Groups[G].Count && (All || Touches(MI, G)))
          Wait(MI.getIterator(), G);
      // Asynchronous instructions without a result (atomics whose old value
      // is unused) leave nothing to wait for, like stores.
      if (!AGXInstrInfo::isAsync(MI) || !MI.getNumExplicitDefs())
        continue;
      unsigned Use = FirstUse.lookup(&MI);
      int GIdx = AGX::getNamedOperandIdx(MI.getOpcode(), AGX::OpName::g);
      unsigned G = 0;
      if (GIdx >= 0) {
        // A group none of whose results is needed before this one's,
        // preferring one needed at the same point, then an empty one;
        // failing both, the group needed last.
        auto Fits = [&](unsigned K) {
          return !Groups[K].Count || Groups[K].Needed >= Use;
        };
        if (Fits(0) && Fits(1))
          G = Groups[1].Count && Groups[1].Needed == Use ? 1
              : Groups[0].Count && Groups[0].Needed != Use && !Groups[1].Count
                  ? 1
                  : 0;
        else if (Fits(0) || Fits(1))
          G = Fits(0) ? 0 : 1;
        else
          G = Groups[1].Needed > Groups[0].Needed ? 1 : 0;
        MI.getOperand(GIdx).setImm(G);
      }
      if (Groups[0].Count + Groups[1].Count >= MaxPending) {
        // Make room: wait for the group needed first.
        unsigned E = Groups[1].Count && (!Groups[0].Count ||
                                         Groups[1].Needed < Groups[0].Needed)
                         ? 1
                         : 0;
        Wait(MI.getIterator(), E);
      }
      for (const MachineOperand &MO : MI.defs())
        if (!isPartialWriteBookkeeping(MI, MO, TRI))
          for (MCRegUnit U : TRI.regunits(MO.getReg().asMCReg()))
            Groups[G].Pending.set(static_cast<unsigned>(U));
      ++Groups[G].Count;
      Groups[G].Needed = std::min(Groups[G].Needed, Use);
      Changed = true;
    }
    // A block that ends the function takes them only to touch one: else its
    // end would wait for them, also after a side that waited.
    if (Carry)
      for (const MachineBasicBlock *S : MBB.successors()) {
        if (!S->succ_empty())
          continue;
        bool Used = false;
        for (const MachineInstr &MI : *S) {
          if (MI.isTerminator())
            break;
          Used |= Touches(MI, 0) || Touches(MI, 1);
        }
        if (!Used) {
          Carry = false;
          for (unsigned G = 0; G < 2; ++G)
            if (Groups[G].Count)
              Wait(MBB.getFirstTerminator(), G);
          break;
        }
      }
    if (Carry) {
      OutState &O = Out[&MBB];
      for (unsigned G = 0; G < 2; ++G) {
        O.Pending[G] = Groups[G].Pending;
        O.Count[G] = Groups[G].Count;
      }
      continue;
    }
    for (unsigned G = 0; G < 2; ++G)
      if (Groups[G].Count)
        Wait(MBB.end(), G);
  }
  return Changed;
}
