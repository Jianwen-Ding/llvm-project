//===-- AGXLowerControlFlow.cpp - Lower structured control flow -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expands the structured control flow pseudos into execution mask
// instructions on the nesting counter r0l and jumps (like SILowerControlFlow):
//
//   AGX_IF c, %flow     if_icmp r0l, ne, c, 0, 1;   jmp_exec_none %flow
//   AGX_ELSE %end       else_fcmp r0l, eq, 0.0, 0.0, 1; jmp_exec_none %end
//   AGX_LOOP brk, %hdr  while_icmp r0l, eq, brk, 0, 1;  jmp_exec_any %hdr
//   AGX_LOOP_BEGIN      if_fcmp r0l, eq, 0.0, 0.0, 1 (push_exec 1)
//   AGX_END_CF          pop_exec r0l, 1, at the start of the block
//   AGX_BR %bb          jmp_exec_any %bb; jmp_exec_none %bb
//
// The jumps only skip code no thread executes; the execution mask
// instructions alone implement the semantics. The nesting counter is zeroed
// at kernel entry. A jump_exec_none over a short run of ALU instructions is
// left out: on G14C an ALU instruction issued with no active thread costs
// about as little as a taken jump over it, while memory instructions still
// cost a large part of their latency (AGX Performance Simulator
// tasks/ptx-re-blockers.md, B31).
//
// Shuffles read the index of every thread of a quad, including inactive
// ones, whose register may hold anything. Register indices are therefore
// passed in r0h (like Mesa's agx_lower_divergent_shuffle), which is zero at
// entry and set back to zero after each shuffle, so the threads that are
// inactive at a shuffle read zero. A reset followed in its block by another
// shuffle index, with no change of the execution mask between them, is left
// out: the same threads are active at both shuffles.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-control-flow"

namespace {

class AGXLowerControlFlow : public MachineFunctionPass {
public:
  static char ID;
  AGXLowerControlFlow() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Control Flow";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

char AGXLowerControlFlow::ID = 0;

INITIALIZE_PASS(AGXLowerControlFlow, DEBUG_TYPE, "AGX Lower Control Flow",
                false, false)

FunctionPass *llvm::createAGXLowerControlFlowPass() {
  return new AGXLowerControlFlow();
}

static cl::opt<bool> MergeBarrierFences(
    "agx-merge-barrier-fences",
    cl::desc("Remove a threadgroup-scope fence right after a threadgroup "
             "barrier that one right before it already orders"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> RegisterOnlyBetweenFences(
    "agx-merge-barrier-fences-across-alu",
    cl::desc("Also remove the fence after a threadgroup barrier when "
             "register-only instructions separate it, the barrier and the "
             "fence before, and a fence right after the same fence"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> AGXSkipLoads(
    "agx-unskipped-loads",
    cl::desc("Let an if or else body run with an empty execution mask also "
             "when it has plain loads"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> AGXSkipDeviceStores(
    "agx-unskipped-device-stores",
    cl::desc("Let an if or else body run with an empty execution mask also "
             "when it has plain device stores"),
    cl::init(true), cl::Hidden);

static cl::opt<unsigned> MaxUnskippedInstrs(
    "agx-max-unskipped-instrs",
    cl::desc("Most ALU instructions an if or else body may have and still be "
             "run with an empty execution mask instead of jumped over"),
    cl::init(12), cl::Hidden);

// Integer conditions.
static constexpr unsigned ICondUEQ = 0;
static constexpr unsigned ICondNE = 8;
// Float condition eq, and the 8-bit float immediate 0.0.
static constexpr unsigned FCondEQ = 0;
static constexpr unsigned FPImmZero = 0;

/// Build an execution mask instruction: (nest, cc, a, b, n).
static MachineInstrBuilder buildExecOp(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       const DebugLoc &DL,
                                       const AGXInstrInfo &TII, unsigned Opc,
                                       unsigned CC) {
  return BuildMI(MBB, I, DL, TII.get(Opc)).addImm(0).addImm(CC);
}

static bool isShuffle(unsigned Opc) {
  switch (Opc) {
  case AGX::SIMD_SHUFFLE_32:
  case AGX::SIMD_SHUFFLE_16:
  case AGX::SIMD_SHUFFLE_XOR_32:
  case AGX::SIMD_SHUFFLE_XOR_16:
  case AGX::SIMD_SHUFFLE_UP_32:
  case AGX::SIMD_SHUFFLE_UP_16:
  case AGX::SIMD_SHUFFLE_DOWN_32:
  case AGX::SIMD_SHUFFLE_DOWN_16:
  case AGX::QUAD_SHUFFLE_32:
  case AGX::QUAD_SHUFFLE_16:
  case AGX::QUAD_SHUFFLE_XOR_32:
  case AGX::QUAD_SHUFFLE_XOR_16:
  case AGX::QUAD_SHUFFLE_UP_32:
  case AGX::QUAD_SHUFFLE_UP_16:
  case AGX::QUAD_SHUFFLE_DOWN_32:
  case AGX::QUAD_SHUFFLE_DOWN_16:
    return true;
  default:
    return false;
  }
}

static bool isControlFlowPseudo(unsigned Opc) {
  switch (Opc) {
  case AGX::AGX_IF:
  case AGX::AGX_ELSE:
  case AGX::AGX_LOOP:
  case AGX::AGX_IF_ICMP:
  case AGX::AGX_IF_FCMP:
  case AGX::AGX_LOOP_ICMP:
  case AGX::AGX_LOOP_FCMP:
  case AGX::AGX_IF_ICMP16:
  case AGX::AGX_LOOP_ICMP16:
  case AGX::AGX_LOOP_BEGIN:
  case AGX::AGX_END_CF:
  case AGX::AGX_BR:
    return true;
  default:
    return false;
  }
}

/// Whether the blocks a jump_exec_none at the end of MBB would skip, up to
/// Target, are a straight run of at most MaxUnskippedInstrs ALU instructions,
/// plain loads and plain device stores, so that running them with an empty
/// execution mask is cheaper than the jump. B31 measured a masked
/// device_load + wait at 10-17 ns and a masked device_store at 7-10 ns; the
/// jump costs its issue where a lane enters the body and a taken jump's
/// latency and scheduler hold where none does. E135 kept memory accesses out
/// before the simulator modelled the masked costs; with them, the jump loses
/// even where no lane ever enters a body of one access (L15-E432, E433).
static bool isShortALURun(MachineBasicBlock &MBB, MachineBasicBlock *Target) {
  unsigned Count = 0;
  MachineFunction &MF = *MBB.getParent();
  for (MachineBasicBlock *S : MBB.successors())
    if (S != Target && !MBB.isLayoutSuccessor(S))
      return false;
  for (auto B = std::next(MBB.getIterator()); &*B != Target; ++B) {
    if (B == MF.end() || B->succ_size() != 1)
      return false;
    MachineBasicBlock *Next = *B->succ_begin();
    if (!B->isLayoutSuccessor(Next))
      return false;
    for (const MachineInstr &MI : *B) {
      if (MI.isMetaInstruction())
        continue;
      // A branch to the next block is left out.
      if (MI.getOpcode() == AGX::AGX_BR && MI.getOperand(0).getMBB() == Next)
        continue;
      // A plain load (not an atomic or volatile access) touches nothing
      // with an empty mask either (L15-E432), nor does a plain device store
      // (L15-E433; B31 measured it, unlike a threadgroup store's).
      bool Load = AGXSkipLoads && MI.mayLoad() && !MI.mayStore() &&
                  !MI.hasOrderedMemoryRef();
      bool Store =
          AGXSkipDeviceStores && MI.mayStore() && !MI.mayLoad() &&
          !MI.hasOrderedMemoryRef() && !MI.memoperands_empty() &&
          llvm::all_of(MI.memoperands(), [](const MachineMemOperand *M) {
            return M->getAddrSpace() == AGX::DEVICE_ADDRESS;
          });
      if (isControlFlowPseudo(MI.getOpcode()) ||
          (MI.mayLoadOrStore() && !Load && !Store) ||
          MI.hasUnmodeledSideEffects() ||
          MI.isCall() || MI.isTerminator() || AGXInstrInfo::isExecMaskOp(MI))
        return false;
      // A shuffle also moves its index to r0h and resets it.
      Count += isShuffle(MI.getOpcode()) ? 3 : 1;
      if (Count > MaxUnskippedInstrs)
        return false;
    }
  }
  return true;
}

/// Jump over the code between MBB and Target when no thread runs it, unless
/// that code is cheap enough to run with an empty mask. Without the jump, the
/// edge to Target is removed: MBB falls through.
static void buildSkip(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                      const DebugLoc &DL, const AGXInstrInfo &TII,
                      MachineBasicBlock *Target) {
  if (!isShortALURun(MBB, Target)) {
    BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_NONE)).addMBB(Target);
    return;
  }
  if (!MBB.isLayoutSuccessor(Target) && MBB.isSuccessor(Target))
    MBB.removeSuccessor(Target);
}

/// The instructions of MBB before its else pseudo Else, other than the
/// reconvergence of inner constructs at its start.
static SmallVector<MachineInstr *, 8> flowBody(MachineBasicBlock &MBB,
                                               MachineInstr &Else) {
  SmallVector<MachineInstr *, 8> Body;
  for (MachineInstr &MI : MBB) {
    if (&MI == &Else)
      break;
    unsigned Opc = MI.getOpcode();
    if (MI.isMetaInstruction() || Opc == AGX::AGX_END_CF ||
        Opc == AGX::POP_EXEC)
      continue;
    if (isShuffle(Opc) || isControlFlowPseudo(Opc) || MI.isConvergent() ||
        AGXInstrInfo::isExecMaskOp(MI))
      report_fatal_error(Twine("AGX: unsupported instruction before an else: ") +
                         MBB.getParent()->getSubtarget().getInstrInfo()->getName(Opc));
    Body.push_back(&MI);
  }
  return Body;
}

static cl::opt<bool> IndexIntoR0H(
    "agx-shuffle-index-into-r0h", cl::init(true), cl::Hidden,
    cl::desc("a shuffle index computed by an addition or bit operation just "
             "before the shuffle is computed into r0h, without the move"));

/// Whether register R (or a register overlapping it) is read after MI before
/// being written again: in MI's block, then on entry to its successors.
static bool isReadAfter(MachineInstr &MI, MCRegister R,
                        const TargetRegisterInfo *TRI) {
  MachineBasicBlock &MBB = *MI.getParent();
  for (auto I = std::next(MI.getIterator()); I != MBB.end(); ++I) {
    if (I->isDebugInstr())
      continue;
    // A read of R or of a register overlapping it.
    if (I->readsRegister(R, TRI))
      return true;
    // A write of R or of a register containing it ends its value.
    if (I->definesRegister(R, TRI))
      return false;
  }
  for (MachineBasicBlock *S : MBB.successors())
    for (MCRegAliasIterator A(R, TRI, true); A.isValid(); ++A)
      if (S->isLiveIn(*A))
        return true;
  return false;
}

/// The 16-bit form of an addition, subtraction or bit operation D whose
/// result is a shuffle's index: computed straight into r0h. The low 16 bits
/// of these results depend only on the low 16 bits of their sources.
static bool computeIndexIntoR0H(MachineInstr &MI, MachineOperand &Index,
                                const AGXInstrInfo &TII) {
  if (!IndexIntoR0H)
    return false;
  MachineBasicBlock &MBB = *MI.getParent();
  const TargetRegisterInfo *TRI = MBB.getParent()->getSubtarget().getRegisterInfo();
  MCRegister I = Index.getReg().asMCReg();
  // The 32-bit register I is the low half of, if any.
  MCRegister W;
  for (MCRegister Sup : TRI->superregs(I))
    if (AGX::GPR32RegClass.contains(Sup) &&
        TRI->getSubReg(Sup, AGX::lo16) == I)
      W = Sup;
  // The instruction that last wrote I, with nothing between it and the
  // shuffle that reads I (or its 32-bit register) or changes the mask.
  MachineInstr *D = nullptr;
  SmallVector<MachineInstr *, 8> Between;
  for (MachineBasicBlock::reverse_iterator It(MI), E = MBB.rend(); ++It != E;) {
    MachineInstr &P = *It;
    if (P.isDebugInstr())
      continue;
    if (P.modifiesRegister(I, TRI)) {
      D = &P;
      break;
    }
    if (P.readsRegister(I, TRI) || (W && P.readsRegister(W, TRI)) ||
        P.isTerminator() || P.isCall() || AGXInstrInfo::isExecMaskOp(P) ||
        Between.size() > 16)
      return false;
    Between.push_back(&P);
  }
  if (!D)
    return false;
  unsigned Opc = D->getOpcode(), NewOpc;
  switch (Opc) {
  case AGX::BITOP_I32: NewOpc = AGX::BITOP_I16; break;
  case AGX::BITOP_I16: NewOpc = AGX::BITOP_I16; break;
  case AGX::IADD_I32: NewOpc = AGX::IADD_I16; break;
  case AGX::IADD_I16: NewOpc = AGX::IADD_I16; break;
  case AGX::ISUB_I32: NewOpc = AGX::ISUB_I16; break;
  case AGX::ISUB_I16: NewOpc = AGX::ISUB_I16; break;
  default:
    return false;
  }
  MCRegister Dst = D->getOperand(0).getReg().asMCReg();
  bool Wide = Opc == AGX::BITOP_I32 || Opc == AGX::IADD_I32 ||
              Opc == AGX::ISUB_I32;
  if (Wide ? (!W || Dst != W) : Dst != I)
    return false;
  // The shuffled value must not be the destination, which the shuffle then
  // no longer finds written, and the destination must be dead after it.
  MachineOperand &Src = MI.getOperand(2);
  if (Src.isReg() && TRI->regsOverlap(Src.getReg(), Dst))
    return false;
  // Dead after the shuffle: the shuffle writes it, or nothing reads it.
  if (!MI.definesRegister(Dst, TRI) && isReadAfter(MI, Dst, TRI))
    return false;
  // Computed again right before the shuffle (after any other shuffle's use of
  // r0h between them): its sources must be unchanged there.
  for (const MachineOperand &MO : D->explicit_uses())
    if (MO.isReg())
      for (MachineInstr *P : Between)
        if (P->modifiesRegister(MO.getReg(), TRI))
          return false;
  // 16-bit sources: the low halves of 32-bit registers, or immediates.
  SmallVector<MachineOperand, 6> Ops;
  for (unsigned K = 1, N = D->getNumExplicitOperands(); K < N; ++K) {
    MachineOperand MO = D->getOperand(K);
    if (MO.isReg()) {
      MCRegister R = MO.getReg().asMCReg();
      if (Wide) {
        MCRegister Lo = TRI->getSubReg(R, AGX::lo16);
        if (!Lo)
          return false;
        R = Lo;
      }
      Ops.push_back(MachineOperand::CreateReg(R, false));
    } else if (MO.isImm()) {
      Ops.push_back(MO);
    } else {
      return false;
    }
  }
  MachineInstrBuilder B =
      BuildMI(MBB, MI.getIterator(), D->getDebugLoc(), TII.get(NewOpc), AGX::R0H);
  for (MachineOperand &MO : Ops)
    B.add(MO);
  D->eraseFromParent();
  return true;
}

/// Pass the register index of a shuffle in r0h. Returns true if MI was
/// changed.
static bool lowerShuffleIndex(MachineInstr &MI, const AGXInstrInfo &TII) {
  // (dst, src0_mods, src0, src1_mods, src1)
  MachineOperand &Index = MI.getOperand(4);
  if (!Index.isReg() || !AGX::GPR16RegClass.contains(Index.getReg()))
    return false;
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  if (computeIndexIntoR0H(MI, Index, TII)) {
    Index.setReg(AGX::R0H);
    Index.setIsKill();
    BuildMI(MBB, std::next(MI.getIterator()), DL, TII.get(AGX::MOV_IMM16),
            AGX::R0H)
        .addImm(0);
    return true;
  }
  TII.copyPhysReg(MBB, MI, DL, AGX::R0H, Index.getReg(), Index.isKill());
  Index.setReg(AGX::R0H);
  Index.setIsKill();
  BuildMI(MBB, std::next(MI.getIterator()), DL, TII.get(AGX::MOV_IMM16),
          AGX::R0H)
      .addImm(0);
  return true;
}

/// Remove a threadgroup barrier that follows another one in its block with
/// nothing between them that touches memory, has side effects or changes the
/// execution mask: the second one orders nothing more.
static bool removeRepeatedBarriers(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    bool AfterBarrier = false;
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.isMetaInstruction())
        continue;
      if (MI.getOpcode() == AGX::THREADGROUP_BARRIER) {
        if (AfterBarrier) {
          MI.eraseFromParent();
          Changed = true;
        }
        AfterBarrier = true;
        continue;
      }
      if (MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall() ||
          MI.isTerminator() || AGXInstrInfo::isExecMaskOp(MI))
        AfterBarrier = false;
    }
  }
  return Changed;
}

/// Whether MI is the threadgroup-scope fence, memory_barrier 1, 2, 9 (f596:
/// every ordering at that scope, llvm/docs/AGXUsage.rst, "AGX fences").
static bool isThreadgroupFence(const MachineInstr &MI) {
  return MI.getOpcode() == AGX::MEMORY_BARRIER &&
         MI.getOperand(0).getImm() == 1 && MI.getOperand(1).getImm() == 2 &&
         MI.getOperand(2).getImm() == 9;
}

/// A barrier with threadgroup fences on both sides (a release fence before
/// it and an acquire fence after it, as agx-lower-air lowers AIR's
/// air.wg.barrier and the gpu dialect's gpu.barrier) keeps the one before:
/// the threads of a threadgroup share a core, where no reordering was
/// observed, and the fence before the barrier with the barrier is what
/// CUDA's __syncthreads compiles to (PTX bar.sync: one fence, then the
/// barrier). The fence after still orders the IR around the barrier
/// (llvm.agx.threadgroup.barrier touches no memory there), and the barrier
/// orders the machine code. Instructions that only compute in registers
/// (which the scheduler moves between them) may come between the fences and
/// the barrier: without a memory access between the two fences, the second
/// orders nothing the first does not. For the same reason a fence right
/// after the same fence goes (a seq_cst atomic's fence before a barrier's).
static bool isRegisterOnly(const MachineInstr &MI) {
  return !MI.mayLoadOrStore() && !MI.hasUnmodeledSideEffects() &&
         !MI.isCall() && !MI.isTerminator() && !MI.isBranch() &&
         !AGXInstrInfo::isExecMaskOp(MI) && !MI.isInlineAsm() &&
         MI.getOpcode() != AGX::MEMORY_BARRIER &&
         MI.getOpcode() != AGX::THREADGROUP_BARRIER;
}

static bool removeFencesAfterBarriers(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // 0: nothing; 1: a threadgroup fence; 2: that fence and a barrier.
    unsigned State = 0;
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.isMetaInstruction())
        continue;
      if (State && RegisterOnlyBetweenFences && isRegisterOnly(MI))
        continue;
      if (isThreadgroupFence(MI)) {
        // After the barrier, or right after the same fence.
        if (State == 2 || (State == 1 && RegisterOnlyBetweenFences)) {
          MI.eraseFromParent();
          Changed = true;
          State = State == 2 ? 0 : 1;
          continue;
        }
        State = 1;
        continue;
      }
      if (MI.getOpcode() == AGX::THREADGROUP_BARRIER && State == 1) {
        State = 2;
        continue;
      }
      State = 0;
    }
  }
  return Changed;
}

static cl::opt<bool> RemoveTrailingFences(
    "agx-remove-trailing-fences", cl::init(true), cl::Hidden,
    cl::desc("Remove a kernel's fences that no memory access, barrier or call "
             "follows on any path to its end"));

/// A fence orders the thread's accesses before it against those after it:
/// in a kernel, a fence that no memory access, barrier or call follows on
/// any path to the stop (an acq_rel atomic's acquire fence at the end, a
/// histogram's) orders nothing (L15-E438), and neither does the wait of an
/// acquire's sequence (wait 0, then the fence) there: AGXInsertWaits, which
/// runs after, still waits before the stop for load results in flight.
/// Barriers count: a fence before one orders this thread's stores for the
/// threads after it.
static bool removeTrailingFences(MachineFunction &MF) {
  if (!RemoveTrailingFences ||
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL)
    return false;
  auto Relevant = [](const MachineInstr &MI) {
    if (MI.isMetaInstruction() || MI.getOpcode() == AGX::MEMORY_BARRIER ||
        MI.getOpcode() == AGX::WAIT)
      return false;
    if (MI.mayLoadOrStore() || MI.isCall() || MI.isInlineAsm() ||
        MI.getOpcode() == AGX::THREADGROUP_BARRIER ||
        MI.getOpcode() == AGX::BLOCK_BARRIER)
      return true;
    // (the execution mask's instructions and the stop order no memory)
    return MI.hasUnmodeledSideEffects() && !AGXInstrInfo::isExecMaskOp(MI) &&
           !MI.isTerminator() && !MI.isBranch();
  };
  // Clean[B]: no relevant instruction from B's end on any path (the
  // greatest fixpoint: a path through one clears it).
  DenseMap<const MachineBasicBlock *, bool> Clean, Inside;
  for (MachineBasicBlock &MBB : MF) {
    Inside[&MBB] = llvm::none_of(MBB, Relevant);
    Clean[&MBB] = true;
  }
  for (bool Again = true; Again;) {
    Again = false;
    for (MachineBasicBlock &MBB : MF) {
      bool C = llvm::all_of(MBB.successors(), [&](MachineBasicBlock *S) {
        return Inside[S] && Clean[S];
      });
      // (a block that ends otherwise than in a stop or by its successors:
      // a return to a caller, which kernels have none of)
      if (MBB.succ_empty() && !(MBB.getFirstTerminator() != MBB.end() &&
                                MBB.getFirstTerminator()->getOpcode() ==
                                    AGX::AGX_STOP))
        C = false;
      if (C != Clean[&MBB]) {
        Clean[&MBB] = C;
        Again = true;
      }
    }
  }
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    bool C = Clean[&MBB];
    for (MachineInstr &MI : make_early_inc_range(reverse(MBB))) {
      if ((MI.getOpcode() == AGX::MEMORY_BARRIER ||
           MI.getOpcode() == AGX::WAIT) &&
          C) {
        MI.eraseFromParent();
        Changed = true;
        continue;
      }
      if (Relevant(MI))
        C = false;
    }
  }
  return Changed;
}

/// Remove the r0h resets that the next shuffle index of their block
/// overwrites before the execution mask can change.
static void removeRedundantIndexResets(MachineFunction &MF) {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  for (MachineBasicBlock &MBB : MF) {
    MachineInstr *Reset = nullptr;
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      bool IsReset = MI.getOpcode() == AGX::MOV_IMM16 &&
                     MI.getOperand(0).getReg() == AGX::R0H &&
                     MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 0;
      if (IsReset) {
        Reset = &MI;
        continue;
      }
      if (!Reset)
        continue;
      if (AGXInstrInfo::isExecMaskOp(MI) || MI.isTerminator() || MI.isCall() ||
          MI.modifiesRegister(AGX::R0L, TRI)) {
        Reset = nullptr;
        continue;
      }
      if (MI.readsRegister(AGX::R0H, TRI)) {
        Reset = nullptr;
        continue;
      }
      if (MI.modifiesRegister(AGX::R0H, TRI)) {
        Reset->eraseFromParent();
        Reset = nullptr;
      }
    }
  }
}

static cl::opt<bool> ShareLoopLevels(
    "agx-share-loop-levels", cl::init(true), cl::Hidden,
    cl::desc("A loop whose exit pop is followed at once by another pop runs "
             "at the level of the construct around it, without its own push "
             "and pop"));

/// The loop entry push (if_fcmp r0l, eq, 0.0, 0.0, 1) and the exit pop of a
/// loop whose exit pop is followed at once by another pop (of the construct
/// the loop ends) are left out. With the push, a thread at level v > 0 on
/// entry is at v + 1 in the loop, one that leaves it at 1; the two pops take
/// these to v - 1 and 0. Without, they are at v and 1 in the loop and the
/// one pop takes them to the same levels. In the loop, the threads that are
/// active are the same: its own mask instructions (while, n = 1) only change
/// active threads, and each construct inside it pushes before it tells
/// levels apart (an else toggles level 1 only after its if has pushed every
/// inactive thread deeper). Calls (which push 2 and rely on their callee's
/// levels) are left alone.
static unsigned countOpcode(MachineBasicBlock &MBB, unsigned Opc) {
  return llvm::count_if(
      MBB, [&](MachineInstr &MI) { return MI.getOpcode() == Opc; });
}

/// Whether MBB starts with two pops (it holds two end.cf pseudos, which go to
/// its start), or holds one alone and falls through to a block that starts
/// with one.
static bool popFollowedByPop(MachineBasicBlock &MBB) {
  unsigned Pops = countOpcode(MBB, AGX::AGX_END_CF);
  if (Pops != 1)
    return Pops >= 2;
  bool Alone = llvm::all_of(MBB, [](MachineInstr &MI) {
    return MI.isMetaInstruction() || MI.getOpcode() == AGX::AGX_END_CF;
  });
  if (!Alone || MBB.succ_size() != 1)
    return false;
  MachineBasicBlock *Next = *MBB.succ_begin();
  return MBB.isLayoutSuccessor(Next) &&
         countOpcode(*Next, AGX::AGX_END_CF) != 0;
}

static void eraseOne(MachineBasicBlock &MBB, unsigned Opc) {
  for (MachineInstr &MI : MBB)
    if (MI.getOpcode() == Opc) {
      MI.eraseFromParent();
      return;
    }
}

static bool shareLoopLevels(MachineFunction &MF) {
  if (!ShareLoopLevels || MF.getFrameInfo().hasCalls())
    return false;
  auto count = countOpcode;
  bool Changed = false;
  for (MachineBasicBlock &L : MF) {
    MachineInstr *Latch = nullptr;
    unsigned HeaderOp = 0;
    for (MachineInstr &MI : L) {
      switch (MI.getOpcode()) {
      case AGX::AGX_LOOP:
        Latch = &MI, HeaderOp = 1;
        break;
      case AGX::AGX_LOOP_ICMP:
      case AGX::AGX_LOOP_FCMP:
      case AGX::AGX_LOOP_ICMP16:
        Latch = &MI, HeaderOp = 5;
        break;
      default:
        break;
      }
    }
    if (!Latch || L.succ_size() != 2)
      continue;
    MachineBasicBlock *Header = Latch->getOperand(HeaderOp).getMBB();
    MachineBasicBlock *Exit = nullptr;
    for (MachineBasicBlock *S : L.successors())
      if (S != Header)
        Exit = S;
    if (!Exit || Exit->pred_size() != 1 || !L.isLayoutSuccessor(Exit) ||
        Header->pred_size() != 2)
      continue;
    MachineBasicBlock *Pre = nullptr;
    for (MachineBasicBlock *P : Header->predecessors())
      if (P != &L)
        Pre = P;
    if (!Pre || Pre == &L || count(*Pre, AGX::AGX_LOOP_BEGIN) != 1)
      continue;
    if (!popFollowedByPop(*Exit))
      continue;
    for (MachineInstr &MI : make_early_inc_range(*Pre))
      if (MI.getOpcode() == AGX::AGX_LOOP_BEGIN)
        MI.eraseFromParent();
    eraseOne(*Exit, AGX::AGX_END_CF);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> ShareIfLevels(
    "agx-share-if-levels", cl::init(true), cl::Hidden,
    cl::desc("An if without an else whose pop is followed at once by another "
             "pop deactivates its threads by a while at the level of the "
             "construct around it, without its own pop"));

/// The ifs without an else whose pop is followed at once by another pop (of
/// the construct the if ends): their compare is a while (n = 1), which
/// deactivates the active threads where it fails (level 1) and leaves the
/// inactive ones where they are, and their pop is left out. With the if, a
/// thread at level v > 0 goes to v + 1 and one that fails to 1, and the two
/// pops take them to v - 1 and 0; with the while, to v and 1, and the one
/// pop gives the same. The active threads in the if are the same (the
/// argument of shareLoopLevels). An else would tell its if's failing threads
/// from the inactive ones by their level, so ifs with one are left alone.
/// Inner ifs go first (from the end of the function), so that a run of them
/// ending together shares one level.
static void shareIfLevels(MachineFunction &MF,
                          SmallPtrSetImpl<MachineInstr *> &Shared) {
  if (!ShareIfLevels || MF.getFrameInfo().hasCalls())
    return;
  for (MachineBasicBlock &MBB : llvm::reverse(MF)) {
    MachineInstr *If = nullptr;
    unsigned TargetOp = 0;
    for (MachineInstr &MI : MBB)
      switch (MI.getOpcode()) {
      case AGX::AGX_IF:
        If = &MI, TargetOp = 1;
        break;
      case AGX::AGX_IF_ICMP:
      case AGX::AGX_IF_FCMP:
      case AGX::AGX_IF_ICMP16:
        If = &MI, TargetOp = 5;
        break;
      default:
        break;
      }
    if (!If)
      continue;
    MachineBasicBlock &Flow = *If->getOperand(TargetOp).getMBB();
    if (countOpcode(Flow, AGX::AGX_ELSE) || !popFollowedByPop(Flow))
      continue;
    eraseOne(Flow, AGX::AGX_END_CF);
    Shared.insert(If);
  }
}

bool AGXLowerControlFlow::runOnMachineFunction(MachineFunction &MF) {
  const AGXInstrInfo &TII = *MF.getSubtarget<AGXSubtarget>().getInstrInfo();
  bool HasControlFlow = false;
  bool HasShuffleIndex = false;

  shareLoopLevels(MF);
  SmallPtrSet<MachineInstr *, 8> SharedIfs;
  shareIfLevels(MF, SharedIfs);

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      MachineBasicBlock::iterator I = MI.getIterator();
      const DebugLoc &DL = MI.getDebugLoc();
      switch (MI.getOpcode()) {
      case AGX::AGX_IF: {
        buildExecOp(MBB, I, DL, TII,
                    SharedIfs.count(&MI) ? AGX::WHILE_ICMP : AGX::IF_ICMP,
                    ICondNE)
            .addImm(0)
            .add(MI.getOperand(0))
            .addImm(0)
            .addImm(0)
            .addImm(1);
        buildSkip(MBB, I, DL, TII, MI.getOperand(1).getMBB());
        break;
      }
      case AGX::AGX_ELSE: {
        SmallVector<MachineInstr *, 8> Body = flowBody(MBB, MI);
        buildExecOp(MBB, I, DL, TII, AGX::ELSE_FCMP, FCondEQ)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(1);
        // The flow block before the else runs with the threads of the then
        // side only, but its code (copies and spill code of the register
        // allocator, which sees the threads of both sides pass through it)
        // is also meant for the threads of the else side: run it again for
        // them, after the switch.
        for (MachineInstr *B : Body) {
          for (MachineOperand &MO : B->operands())
            if (MO.isReg() && MO.isUse())
              MO.setIsKill(false);
          MBB.insert(I, MF.CloneMachineInstr(B));
        }
        buildSkip(MBB, I, DL, TII, MI.getOperand(0).getMBB());
        break;
      }
      case AGX::AGX_LOOP:
        // Threads stay in the loop while brk is zero.
        buildExecOp(MBB, I, DL, TII, AGX::WHILE_ICMP, ICondUEQ)
            .addImm(0)
            .add(MI.getOperand(0))
            .addImm(0)
            .addImm(0)
            .addImm(1);
        BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY))
            .addMBB(MI.getOperand(1).getMBB());
        break;
      case AGX::AGX_IF_ICMP:
      case AGX::AGX_IF_FCMP:
      case AGX::AGX_LOOP_ICMP:
      case AGX::AGX_LOOP_FCMP:
      case AGX::AGX_IF_ICMP16:
      case AGX::AGX_LOOP_ICMP16: {
        // (cc, src0_mods, src0, src1_mods, src1, target). An if keeps the
        // threads where the comparison holds; a loop keeps those where it
        // does not (the comparison is the break condition).
        unsigned Opc = MI.getOpcode();
        bool Loop = Opc == AGX::AGX_LOOP_ICMP || Opc == AGX::AGX_LOOP_FCMP ||
                    Opc == AGX::AGX_LOOP_ICMP16;
        bool Float = Opc == AGX::AGX_IF_FCMP || Opc == AGX::AGX_LOOP_FCMP;
        bool Half = Opc == AGX::AGX_IF_ICMP16 || Opc == AGX::AGX_LOOP_ICMP16;
        unsigned CC = MI.getOperand(0).getImm() ^ (Loop ? 8 : 0);
        // An if that shares the level around it is a while (shareIfLevels).
        bool While = Loop || SharedIfs.count(&MI);
        unsigned Exec =
            Half ? (While ? AGX::WHILE_ICMP16 : AGX::IF_ICMP16)
            : While ? (Float ? AGX::WHILE_FCMP : AGX::WHILE_ICMP)
                    : (Float ? AGX::IF_FCMP : AGX::IF_ICMP);
        buildExecOp(MBB, I, DL, TII, Exec, CC)
            .add(MI.getOperand(1))
            .add(MI.getOperand(2))
            .add(MI.getOperand(3))
            .add(MI.getOperand(4))
            .addImm(1);
        if (Loop)
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY))
              .addMBB(MI.getOperand(5).getMBB());
        else
          buildSkip(MBB, I, DL, TII, MI.getOperand(5).getMBB());
        break;
      }
      case AGX::AGX_LOOP_BEGIN:
        buildExecOp(MBB, I, DL, TII, AGX::IF_FCMP, FCondEQ)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(0)
            .addImm(FPImmZero)
            .addImm(1);
        break;
      case AGX::AGX_END_CF: {
        // The construct reconverges at the start of the block. Scheduling
        // may have moved instructions of the block above the pseudo, since
        // they have no dependency on it: the pop goes first (like
        // SILowerControlFlow::emitEndCf), after any earlier pops.
        MachineBasicBlock::iterator Pos = MBB.begin();
        while (Pos != I && Pos->getOpcode() == AGX::POP_EXEC)
          ++Pos;
        BuildMI(MBB, Pos, DL, TII.get(AGX::POP_EXEC)).addImm(0).addImm(1);
        break;
      }
      case AGX::AGX_BR: {
        MachineBasicBlock *Target = MI.getOperand(0).getMBB();
        if (!MBB.isLayoutSuccessor(Target)) {
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_ANY)).addMBB(Target);
          BuildMI(MBB, I, DL, TII.get(AGX::JMP_EXEC_NONE)).addMBB(Target);
        }
        MI.eraseFromParent();
        continue;
      }
      default:
        if (isShuffle(MI.getOpcode()))
          HasShuffleIndex |= lowerShuffleIndex(MI, TII);
        continue;
      }
      HasControlFlow = true;
      MI.eraseFromParent();
    }
  }

  if (HasShuffleIndex)
    removeRedundantIndexResets(MF);
  removeRepeatedBarriers(MF);
  if (MergeBarrierFences)
    removeFencesAfterBarriers(MF);
  removeTrailingFences(MF);

  MachineBasicBlock &Entry = MF.front();
  if (HasShuffleIndex)
    BuildMI(Entry, Entry.begin(), DebugLoc(), TII.get(AGX::MOV_IMM16),
            AGX::R0H)
        .addImm(0);
  // Callable functions inherit the caller's execution mask depth. Resetting
  // it here would resurrect lanes excluded by the enclosing call.
  if (HasControlFlow &&
      MF.getFunction().getCallingConv() == CallingConv::AGX_KERNEL) {
    BuildMI(Entry, Entry.begin(), DebugLoc(), TII.get(AGX::SET_NEST))
        .addImm(0);
  }
  return true;
}
