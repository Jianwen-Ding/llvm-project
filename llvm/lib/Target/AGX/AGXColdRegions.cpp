//===-- AGXColdRegions.cpp - Move rarely run if-bodies out of line --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An if-body that is rarely entered (its edge unlikely, as the slow paths of
// software double precision, division and strict floating point are) is
// skipped by a jmp_exec_none that is taken nearly always, and a taken jump
// costs a fetch redirect (about 34 cycles on G14C against 1 for one not
// taken; AGX Performance Simulator, agx_cost's taken-branch fit). Here such a
// body moves to the end of the function:
//
//   before:  H: if_icmp ...; jmp_exec_none %T     after:  H: if_icmp ...; jmp_exec_any %B
//            B: (body) ...                                T: pop_exec ...
//            T: pop_exec ...                              ...
//                                                         B: (body) ...; jmp_exec_any %T;
//                                                            jmp_exec_none %T
//
// so that the common case falls through a jump not taken, and the rare one
// takes two. Every path runs the same instructions in the same order (and
// the execution mask instructions alone decide which threads run them), so
// the waits inserted before stay right: this runs after AGXInsertWaits.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-cold-regions"

static cl::opt<bool> EnableColdRegions(
    "agx-cold-regions", cl::init(true), cl::Hidden,
    cl::desc("Move if-bodies entered with an unlikely edge to the end of the "
             "function, so that skipping them is a jump not taken"));

static cl::opt<unsigned> ColdRegionsPercent(
    "agx-cold-regions-percent", cl::init(1), cl::Hidden,
    cl::desc("Largest probability (in percent) of the edge into an if-body "
             "that AGXColdRegions moves out of line"));

namespace {

class AGXColdRegions : public MachineFunctionPass {
public:
  static char ID;
  AGXColdRegions() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Cold Regions"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char AGXColdRegions::ID = 0;

INITIALIZE_PASS(AGXColdRegions, DEBUG_TYPE, "AGX Cold Regions", false, false)

FunctionPass *llvm::createAGXColdRegionsPass() { return new AGXColdRegions(); }

/// The target of MBB's closing jmp_exec_any %X; jmp_exec_none %X (one of the
/// two is taken whatever the mask: an unconditional jump), if it has them.
static MachineBasicBlock *jumpsAlways(const MachineBasicBlock &MBB) {
  if (MBB.size() < 2)
    return nullptr;
  const MachineInstr &None = MBB.back();
  const MachineInstr &Any = *std::prev(MBB.end(), 2);
  if (None.getOpcode() != AGX::JMP_EXEC_NONE ||
      Any.getOpcode() != AGX::JMP_EXEC_ANY ||
      None.getOperand(0).getMBB() != Any.getOperand(0).getMBB())
    return nullptr;
  return None.getOperand(0).getMBB();
}

/// Whether MBB ends in a way that cannot fall through to the next block.
static bool endsWithoutFallthrough(const MachineBasicBlock &MBB) {
  if (MBB.empty())
    return false;
  const MachineInstr &Last = MBB.back();
  if (Last.isReturn() || (Last.isBarrier() && !Last.isBranch()))
    return true;
  return jumpsAlways(MBB) != nullptr;
}

bool AGXColdRegions::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableColdRegions || skipFunction(MF.getFunction()))
    return false;
  const auto &TII =
      *static_cast<const AGXInstrInfo *>(MF.getSubtarget().getInstrInfo());
  const BranchProbability Cold(ColdRegionsPercent, 100);
  bool Changed = false;

  SmallVector<MachineBasicBlock *, 32> Heads;
  for (MachineBasicBlock &MBB : MF)
    Heads.push_back(&MBB);

  for (MachineBasicBlock *H : Heads) {
    // H ends in jmp_exec_none %T, T further down; the body is what lies
    // between.
    if (H->empty() || H->back().getOpcode() != AGX::JMP_EXEC_NONE)
      continue;
    MachineInstr &Skip = H->back();
    MachineBasicBlock *T = Skip.getOperand(0).getMBB();
    auto BodyBegin = std::next(H->getIterator());
    if (BodyBegin == MF.end() || &*BodyBegin == T)
      continue;
    MachineBasicBlock *B = &*BodyBegin;
    if (!H->isSuccessor(B) || !H->isSuccessor(T) || H->succ_size() != 2 ||
        H->getSuccProbability(llvm::find(H->successors(), B)) > Cold)
      continue;
    // The body: the blocks from B up to T in layout, entered from H only and
    // left to T only.
    SmallVector<MachineBasicBlock *, 8> Body;
    SmallPtrSet<MachineBasicBlock *, 8> InBody;
    bool Found = false;
    for (auto It = BodyBegin; It != MF.end(); ++It) {
      if (&*It == T) {
        Found = true;
        break;
      }
      Body.push_back(&*It);
      InBody.insert(&*It);
    }
    if (!Found)
      continue;
    bool Closed = true;
    for (MachineBasicBlock *X : Body) {
      for (MachineBasicBlock *P : X->predecessors())
        if (!InBody.contains(P) && !(X == B && P == H))
          Closed = false;
      for (MachineBasicBlock *S : X->successors())
        if (!InBody.contains(S) && S != T)
          Closed = false;
      if (X->hasAddressTaken() || X->isEHPad())
        Closed = false;
    }
    MachineBasicBlock *Last = Body.back();
    MachineBasicBlock *End = &MF.back();
    if (!Closed || InBody.contains(End) || !endsWithoutFallthrough(*End))
      continue;
    // The last body block falls through to T (or jumps there already).
    DebugLoc DL = Skip.getDebugLoc();
    if (!endsWithoutFallthrough(*Last)) {
      // (a conditional jump at its end: it falls through to T otherwise)
      if (!Last->isSuccessor(T))
        continue;
      BuildMI(*Last, Last->end(), DL, TII.get(AGX::JMP_EXEC_ANY)).addMBB(T);
      BuildMI(*Last, Last->end(), DL, TII.get(AGX::JMP_EXEC_NONE)).addMBB(T);
    }
    // H enters the body where any thread remains, and otherwise falls
    // through to T.
    BuildMI(*H, Skip.getIterator(), DL, TII.get(AGX::JMP_EXEC_ANY)).addMBB(B);
    Skip.eraseFromParent();
    MF.splice(MF.end(), B->getIterator(), T->getIterator());
    Changed = true;
  }
  return Changed;
}
