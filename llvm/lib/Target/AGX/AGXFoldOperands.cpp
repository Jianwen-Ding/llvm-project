//===-- AGXFoldOperands.cpp - Fold uniform registers into operands --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Kernel arguments live in uniform registers, which are never allocated:
// instruction selection copies them into virtual general purpose registers.
// Like SIFoldOperands, this pass folds such copies into the operands that
// can read uniform registers directly (ALU sources and memory bases), which
// saves both the moves and the general purpose registers.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/ADT/MapVector.h"
#include "AGXInstrInfo.h"
#include "AGXRegisterInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "agx-fold-operands"

static cl::opt<bool> FoldSharedBranchCompares(
    "agx-fold-shared-branch-compares",
    cl::desc("Fold a comparison that only branches use into each branch"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> FoldWidenedBranchConds(
    "agx-fold-widened-branch-conds",
    cl::desc("Branch on a zero-extended 16-bit condition's 16 bits"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> FoldWidened16Sources(
    "agx-fold-widened-16-sources",
    cl::desc("popcount / ffs / bitrev of a zero-extended 16-bit value read "
             "the 16-bit register"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> HalfConstantFMASources(
    "agx-half-constant-fma-sources",
    cl::desc("an fmadd32 of three 32-bit registers, one a constant that half "
             "holds exactly, reads it from a 16-bit register"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> FoldNaNOperandSelects(
    "agx-fold-nan-operand-selects",
    cl::desc("select(x is NaN, y, maximumnum / minimumnum(x, y)) as the "
             "min / max, which gives y then"),
    cl::init(true), cl::Hidden);

namespace {

class AGXFoldOperands : public MachineFunctionPass {
public:
  static char ID;
  AGXFoldOperands() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Fold Operands"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXFoldOperands::ID = 0;

INITIALIZE_PASS(AGXFoldOperands, DEBUG_TYPE, "AGX Fold Operands", false,
                false)

FunctionPass *llvm::createAGXFoldOperandsPass() {
  return new AGXFoldOperands();
}

/// Memory base registers encode uniform pairs below u128 only.
static constexpr unsigned MaxUniformBaseHalf = 0x100;

static cl::opt<bool> AGXChainConstantAdds(
    "agx-chain-constant-adds", cl::init(true), cl::Hidden,
    cl::desc("b + c_j as (b + c_i) + (c_j - c_i) where the difference is an "
             "immediate or another wide constant's register, so that c_j's "
             "mov_imm goes"));

/// Additions of wide constants (mov_imm) to one base in a block, b + c_1,
/// b + c_2, ...: each after the first as an earlier sum plus the difference
/// (mod 2^32, exact) where that is an 8-bit immediate, a negated one, or the
/// register of a constant mov_imm'd before it in the block or the entry
/// block, when then every use of c_j's mov_imm goes and the mov_imm with
/// it. Philox's key schedule k0 + r 0x9E3779B9 (a mov_imm and an iadd a
/// round: one iadd), transpose_smem's row offsets b + 8192 k (L15-E420).
static bool chainConstantAdds(MachineFunction &MF, MachineRegisterInfo &MRI,
                              const AGXInstrInfo *TII) {
  if (!AGXChainConstantAdds)
    return false;
  auto ConstOf = [&](const MachineOperand &O) -> std::optional<uint32_t> {
    if (!O.isReg() || !O.getReg().isVirtual() || O.getSubReg())
      return std::nullopt;
    MachineInstr *D = MRI.getVRegDef(O.getReg());
    if (!D || D->getOpcode() != AGX::MOV_IMM32 || !D->getOperand(1).isImm())
      return std::nullopt;
    return uint32_t(D->getOperand(1).getImm());
  };
  struct Add {
    MachineInstr *MI;
    unsigned KOp; // the constant's operand (2 or 4)
    uint32_t C;
  };
  struct Rewrite {
    MachineInstr *MI;
    Register Prev, DR;
    int64_t Imm;
  };
  // Over the function: each constant register's rewrites, and those that
  // cannot all be rewritten.
  MapVector<Register, SmallVector<Rewrite, 2>> Plan;
  DenseSet<Register> Blocked;
  MachineBasicBlock &Entry = MF.front();
  for (MachineBasicBlock &MBB : MF) {
    // (base register, subregister) -> its additions, in order
    MapVector<std::pair<Register, unsigned>, SmallVector<Add, 4>> Groups;
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() != AGX::IADD_I32 || !MI.getOperand(1).isImm() ||
          MI.getOperand(1).getImm() || !MI.getOperand(3).isImm() ||
          MI.getOperand(3).getImm() || !MI.getOperand(5).isImm() ||
          MI.getOperand(5).getImm())
        continue;
      for (unsigned K : {4u, 2u}) {
        unsigned B = K == 4 ? 2 : 4;
        std::optional<uint32_t> C = ConstOf(MI.getOperand(K));
        const MachineOperand &BO = MI.getOperand(B);
        if (!C || !BO.isReg() || ConstOf(BO))
          continue;
        Groups[{BO.getReg(), BO.getSubReg()}].push_back({&MI, K, *C});
        break;
      }
    }
    for (auto &[Base, Adds] : Groups) {
      if (Adds.size() < 2)
        continue;
      // The constants' registers usable at an addition: mov_imm'd earlier
      // in this block or in the entry block.
      auto RegFor = [&](uint32_t D, MachineInstr *At) -> Register {
        for (const Add &A : Adds) {
          Register R = A.MI->getOperand(A.KOp).getReg();
          MachineInstr *Def = MRI.getVRegDef(R);
          if (uint32_t(Def->getOperand(1).getImm()) != D)
            continue;
          if (Def->getParent() == &Entry && &MBB != &Entry)
            return R;
          if (Def->getParent() == &MBB) {
            for (MachineInstr &I : MBB) {
              if (&I == Def)
                return R;
              if (&I == At)
                break;
            }
          }
        }
        return Register();
      };
      // (the constants that may go: those of the second addition on)
      DenseSet<Register> Candidates;
      for (unsigned J = 1; J < Adds.size(); ++J)
        Candidates.insert(Adds[J].MI->getOperand(Adds[J].KOp).getReg());
      // The first addition's constant stays.
      Blocked.insert(Adds[0].MI->getOperand(Adds[0].KOp).getReg());
      for (unsigned J = 1; J < Adds.size(); ++J) {
        Register KJ = Adds[J].MI->getOperand(Adds[J].KOp).getReg();
        bool Found = false;
        // The nearest earlier sum first; a difference register that stays
        // (not itself one that may go) before one that may.
        for (bool AnyReg : {false, true})
          for (unsigned I = J; I-- > 0 && !Found;) {
            uint32_t D = Adds[J].C - Adds[I].C;
            int64_t Imm = 0;
            Register DR;
            if (isUInt<8>(D))
              Imm = D;
            else if (isUInt<8>(uint32_t(-D)) && D)
              Imm = -int64_t(uint32_t(-D));
            else if ((DR = RegFor(D, Adds[J].MI)) && DR != KJ &&
                     (AnyReg || !Candidates.contains(DR)))
              ;
            else
              continue;
            Plan[KJ].push_back(
                {Adds[J].MI, Adds[I].MI->getOperand(0).getReg(), DR, Imm});
            Found = true;
          }
        if (!Found)
          Blocked.insert(KJ);
      }
    }
  }
  // A register some rewrite reads as its difference stays.
  for (auto &[KJ, Rewrites] : Plan)
    for (const Rewrite &R : Rewrites)
      if (R.DR)
        Blocked.insert(R.DR);
  bool Changed = false;
  for (auto &[KJ, Rewrites] : Plan) {
    if (Blocked.contains(KJ))
      continue;
    unsigned Uses = 0;
    for (MachineInstr &U : MRI.use_nodbg_instructions(KJ))
      (void)U, ++Uses;
    if (Uses != Rewrites.size())
      continue;
    for (const Rewrite &R : Rewrites) {
      MachineInstr *MI = R.MI;
      MachineInstrBuilder NB =
          BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                  TII->get(R.Imm < 0 ? AGX::ISUB_I32 : AGX::IADD_I32),
                  MI->getOperand(0).getReg())
              .addImm(0)
              .addReg(R.Prev)
              .addImm(0);
      if (R.DR)
        NB.addReg(R.DR);
      else
        NB.addImm(R.Imm < 0 ? -R.Imm : R.Imm);
      NB.addImm(0);
      MI->eraseFromParent();
    }
    MRI.getVRegDef(KJ)->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

bool AGXFoldOperands::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  const AGXSubtarget &ST = MF.getSubtarget<AGXSubtarget>();
  const AGXInstrInfo *TII = ST.getInstrInfo();
  const AGXRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  auto IsUniformCopy = [&](const MachineInstr &MI) {
    if (!MI.isCopy())
      return false;
    Register Dst = MI.getOperand(0).getReg();
    Register Src = MI.getOperand(1).getReg();
    return Dst.isVirtual() && Src.isPhysical() &&
           !MI.getOperand(1).getSubReg() &&
           AGX::getRegInfo(Src, *TRI).isUniform();
  };

  SmallVector<MachineInstr *, 32> Worklist;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (IsUniformCopy(MI))
        Worklist.push_back(&MI);

  bool Changed = false;
  while (!Worklist.empty()) {
    MachineInstr &Copy = *Worklist.pop_back_val();
    Register Dst = Copy.getOperand(0).getReg();
    Register Src = Copy.getOperand(1).getReg();

    for (MachineOperand &Use :
         make_early_inc_range(MRI.use_nodbg_operands(Dst))) {
      MachineInstr &UseMI = *Use.getParent();
      MCRegister Folded = Src;
      if (Use.getSubReg())
        Folded = TRI->getSubReg(Src, Use.getSubReg());
      if (!Folded)
        continue;

      // Copies of parts of a uniform tuple become uniform copies themselves.
      if (UseMI.isCopy()) {
        if (Use.getSubReg() && UseMI.getOperand(0).getReg().isVirtual()) {
          Use.setReg(Folded);
          Use.setSubReg(0);
          Use.setIsKill(false);
          Worklist.push_back(&UseMI);
          Changed = true;
        }
        continue;
      }
      if (UseMI.isPHI() || UseMI.isRegSequence() || UseMI.isInlineAsm())
        continue;
      unsigned OpIdx = UseMI.getOperandNo(&Use);
      if (OpIdx >= UseMI.getDesc().getNumOperands())
        continue;
      const TargetRegisterClass *RC = TII->getRegClass(UseMI.getDesc(), OpIdx);
      if (!RC || !RC->contains(Folded))
        continue;
      if (RC == &AGX::BASE64RegClass &&
          AGX::getRegInfo(Folded, *TRI).HalfIndex >= MaxUniformBaseHalf)
        continue;
      Use.setReg(Folded);
      Use.setSubReg(0);
      Use.setIsKill(false);
      Changed = true;
    }

    if (MRI.use_nodbg_empty(Dst)) {
      Copy.eraseFromParent();
      Changed = true;
    }
  }

  // Branch on a comparison directly: an AGX_IF / AGX_LOOP whose condition is
  // "icmpsel cc, a, b, T, F" with constant T and F, one of them zero, takes
  // the comparison itself (inverted when T is the zero), where it is the only
  // use in the branch's block, or where every use of it is such a branch and
  // it reads at most one general register (the others immediates or
  // uniforms), so that the branches keep that register live instead of the
  // condition (MachineCSE gives two blocks' "lane == 0" branches one
  // icmpsel, which they then test for nonzero).
  auto Foldable = [&](const MachineInstr *Def) {
    if (!Def)
      return false;
    bool Float = Def->getOpcode() == AGX::FCMPSEL_F32;
    if (!Float && Def->getOpcode() != AGX::ICMPSEL_I32)
      return false;
    // (dst, cc, src0_mods, src0, src1_mods, src1, src2_mods, src2, src3_mods,
    //  src3)
    const MachineOperand &X = Def->getOperand(7), &Y = Def->getOperand(9);
    return X.isImm() && Y.isImm() && !Def->getOperand(6).getImm() &&
           !Def->getOperand(8).getImm() &&
           (X.getImm() == 0) != (Y.getImm() == 0);
  };
  auto IsCondBranch = [](const MachineInstr &MI) {
    return MI.getOpcode() == AGX::AGX_IF || MI.getOpcode() == AGX::AGX_LOOP;
  };
  auto FoldInto = [&](MachineInstr &Br, MachineInstr &Def) {
    bool Loop = Br.getOpcode() == AGX::AGX_LOOP;
    bool Float = Def.getOpcode() == AGX::FCMPSEL_F32;
    // A nonzero condition takes the if / breaks the loop.
    unsigned CC =
        Def.getOperand(1).getImm() ^ (Def.getOperand(7).getImm() ? 0 : 8);
    unsigned Opc = Loop ? (Float ? AGX::AGX_LOOP_FCMP : AGX::AGX_LOOP_ICMP)
                        : (Float ? AGX::AGX_IF_FCMP : AGX::AGX_IF_ICMP);
    MachineInstrBuilder B =
        BuildMI(*Br.getParent(), Br, Br.getDebugLoc(), TII->get(Opc))
            .addImm(CC);
    for (unsigned I = 2; I <= 5; ++I) {
      MachineOperand Op = Def.getOperand(I);
      if (Op.isReg())
        Op.setIsKill(false);
      B.add(Op);
    }
    B.add(Br.getOperand(1));
    Br.eraseFromParent();
  };
  // An AGX_IF / AGX_LOOP on a 16-bit value zero-extended to 32 bits (a
  // boolean phi, which type legalization keeps as i16) tests the 16 bits
  // themselves, without the widening copy: every use of the copy is such a
  // branch, so the 16-bit register stays live instead of the 32-bit one.
  if (FoldWidenedBranchConds) {
    SmallVector<MachineInstr *, 8> Widened;
    for (MachineBasicBlock &MBB : MF) {
      MachineBasicBlock::iterator T = MBB.getFirstTerminator();
      if (T == MBB.end() || !IsCondBranch(*T) ||
          !T->getOperand(0).getReg().isVirtual())
        continue;
      MachineInstr *Def = MRI.getVRegDef(T->getOperand(0).getReg());
      // (dst, table, src0_mods, src0, src1_mods, src1): a move of src0.
      if (!Def || Def->getOpcode() != AGX::BITOP_I32_FROM16 ||
          Def->getOperand(1).getImm() != 0b1010 ||
          Def->getOperand(2).getImm() || !Def->getOperand(3).isReg() ||
          !Def->getOperand(3).getReg().isVirtual() ||
          Def->getOperand(3).getSubReg() || !Def->getOperand(5).isImm() ||
          Def->getOperand(5).getImm() || is_contained(Widened, Def))
        continue;
      if (all_of(MRI.use_nodbg_instructions(Def->getOperand(0).getReg()),
                 [&](MachineInstr &U) {
                   return IsCondBranch(U) &&
                          &*U.getParent()->getFirstTerminator() == &U;
                 }))
        Widened.push_back(Def);
    }
    for (MachineInstr *Def : Widened) {
      Register Src = Def->getOperand(3).getReg();
      MRI.clearKillFlags(Src);
      SmallVector<MachineInstr *, 4> Branches;
      for (MachineInstr &U :
           MRI.use_nodbg_instructions(Def->getOperand(0).getReg()))
        Branches.push_back(&U);
      for (MachineInstr *Br : Branches) {
        bool Loop = Br->getOpcode() == AGX::AGX_LOOP;
        // Nonzero takes the if / breaks the loop: "not ueq 0" (cc bit 3).
        BuildMI(*Br->getParent(), Br, Br->getDebugLoc(),
                TII->get(Loop ? AGX::AGX_LOOP_ICMP16 : AGX::AGX_IF_ICMP16))
            .addImm(8)
            .addImm(0)
            .addReg(Src)
            .addImm(0)
            .addImm(0)
            .add(Br->getOperand(1));
        Br->eraseFromParent();
      }
      Def->eraseFromParent();
      Changed = true;
    }
  }

  SmallVector<MachineInstr *, 8> Defs;
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::iterator T = MBB.getFirstTerminator();
    if (T == MBB.end() || !IsCondBranch(*T))
      continue;
    Register Cond = T->getOperand(0).getReg();
    if (!Cond.isVirtual())
      continue;
    MachineInstr *Def = MRI.getVRegDef(Cond);
    if (!Foldable(Def))
      continue;
    if (MRI.hasOneNonDBGUse(Cond) && Def->getParent() == &MBB) {
      FoldInto(*T, *Def);
      Def->eraseFromParent();
      Changed = true;
      continue;
    }
    if (!is_contained(Defs, Def))
      Defs.push_back(Def);
  }
  for (MachineInstr *Def : FoldSharedBranchCompares ? Defs
                                                     : SmallVector<MachineInstr *, 8>()) {
    Register Cond = Def->getOperand(0).getReg();
    unsigned GPRs = 0;
    bool Ok = true;
    for (unsigned I : {3, 5}) {
      const MachineOperand &Op = Def->getOperand(I);
      if (!Op.isReg())
        continue;
      if (Op.getReg().isVirtual())
        ++GPRs;
      else if (!AGX::getRegInfo(Op.getReg(), *TRI).isUniform())
        Ok = false;
    }
    if (!Ok || GPRs > 1)
      continue;
    SmallVector<MachineInstr *, 4> Branches;
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Cond)) {
      if (!IsCondBranch(UseMI) || &*UseMI.getParent()->getFirstTerminator() !=
                                      &UseMI) {
        Ok = false;
        break;
      }
      Branches.push_back(&UseMI);
    }
    if (!Ok)
      continue;
    for (unsigned I : {3, 5})
      if (Def->getOperand(I).isReg() && Def->getOperand(I).getReg().isVirtual())
        MRI.clearKillFlags(Def->getOperand(I).getReg());
    for (MachineInstr *Br : Branches)
      FoldInto(*Br, *Def);
    Def->eraseFromParent();
    Changed = true;
  }

  // A popcount, ffs or bitrev of a zero-extended 16-bit value reads the 16
  // bits itself: the 16-bit source forms zero-extend (recorded on the M2),
  // so the extension's move goes where it has no other use (L15-E361).
  if (FoldWidened16Sources) {
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(MBB)) {
        unsigned New = MI.getOpcode() == AGX::POPCOUNT_I32 ? AGX::POPCOUNT_I32_FROM16
                       : MI.getOpcode() == AGX::FFS_I32    ? AGX::FFS_I32_FROM16
                       : MI.getOpcode() == AGX::BITREV_I32 ? AGX::BITREV_I32_FROM16
                                                           : 0;
        if (!New)
          continue;
        // (dst, src0_mods, src0)
        MachineOperand &S = MI.getOperand(2);
        if (!MI.getOperand(1).isImm() || MI.getOperand(1).getImm() || !S.isReg() ||
            !S.getReg().isVirtual() || S.getSubReg())
          continue;
        MachineInstr *Def = MRI.getVRegDef(S.getReg());
        // (dst, table, src0_mods, src0, src1_mods, src1): a move of src0.
        if (!Def || Def->getOpcode() != AGX::BITOP_I32_FROM16 ||
            Def->getOperand(1).getImm() != 0b1010 ||
            Def->getOperand(2).getImm() || !Def->getOperand(3).isReg() ||
            !Def->getOperand(5).isImm() || Def->getOperand(5).getImm())
          continue;
        MachineOperand Src16 = Def->getOperand(3);
        Src16.setIsKill(false);
        if (Src16.getReg().isVirtual())
          MRI.clearKillFlags(Src16.getReg());
        BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(New),
                MI.getOperand(0).getReg())
            .addImm(0)
            .add(Src16);
        MI.eraseFromParent();
        if (MRI.use_nodbg_empty(Def->getOperand(0).getReg()))
          Def->eraseFromParent();
        Changed = true;
      }
  }
  // select(x is NaN, y, maxnum / minnum(x, y)), the PTX front end's
  // max(m, v) of a running maximum (L15-E367): fcmpsel eq, x, x, M, y where
  // M is fcmpsel gtn / ltn, x, y, x, y (the "n" conditions take the other
  // operand where one is a NaN, and y where both are: the second select
  // operand, as the condition is false), possibly under the signed-zero fix
  // fcmpsel eq, x, y, bits, M (false for a NaN x). Where x is a NaN, M is
  // y's bits already; elsewhere the select gives M. So it is M.
  if (FoldNaNOperandSelects) {
    auto Mods0 = [](const MachineInstr &MI) {
      for (unsigned I : {2u, 4u, 6u, 8u})
        if (!MI.getOperand(I).isImm() || MI.getOperand(I).getImm())
          return false;
      return true;
    };
    auto VReg = [](const MachineOperand &O) {
      return O.isReg() && O.getReg().isVirtual() && !O.getSubReg();
    };
    // (dst, cc, a_mods, a, b_mods, b, x_mods, x, y_mods, y)
    auto IsSel = [&](const MachineInstr *MI, unsigned Opc) {
      return MI && MI->getOpcode() == Opc && MI->getOperand(1).isImm() &&
             Mods0(*MI) && VReg(MI->getOperand(3)) && VReg(MI->getOperand(5)) &&
             VReg(MI->getOperand(7)) && VReg(MI->getOperand(9));
    };
    const unsigned FC_EQ = 0, FC_LTN = 3, FC_GTN = 7;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(MBB)) {
        unsigned Opc = MI.getOpcode();
        if ((Opc != AGX::FCMPSEL_F32 && Opc != AGX::FCMPSEL_F16) ||
            !IsSel(&MI, Opc) || MI.getOperand(1).getImm() != FC_EQ)
          continue;
        Register X = MI.getOperand(3).getReg(), Y = MI.getOperand(9).getReg();
        Register M = MI.getOperand(7).getReg();
        if (MI.getOperand(5).getReg() != X || X == Y)
          continue;
        auto IsMinMax = [&](Register R) {
          MachineInstr *D = MRI.getVRegDef(R);
          if (!IsSel(D, Opc))
            return false;
          int64_t CC = D->getOperand(1).getImm();
          return (CC == FC_GTN || CC == FC_LTN) &&
                 D->getOperand(3).getReg() == X &&
                 D->getOperand(5).getReg() == Y &&
                 D->getOperand(7).getReg() == X &&
                 D->getOperand(9).getReg() == Y;
        };
        bool Ok = IsMinMax(M);
        if (!Ok) {
          MachineInstr *Fix = MRI.getVRegDef(M);
          Ok = IsSel(Fix, Opc) && Fix->getOperand(1).getImm() == FC_EQ &&
               ((Fix->getOperand(3).getReg() == X &&
                 Fix->getOperand(5).getReg() == Y) ||
                (Fix->getOperand(3).getReg() == Y &&
                 Fix->getOperand(5).getReg() == X)) &&
               IsMinMax(Fix->getOperand(9).getReg());
        }
        if (!Ok)
          continue;
        Register D = MI.getOperand(0).getReg();
        if (MRI.getRegClass(D) != MRI.getRegClass(M))
          continue;
        MRI.replaceRegWith(D, M);
        MRI.clearKillFlags(M);
        MI.eraseFromParent();
        Changed = true;
      }
  }
  // An fmadd32 of three 32-bit registers where one is a constant that half
  // holds exactly (a Cody-Waite ln2 split's high part, 0.693359375; small
  // integers): that one from a 16-bit register, which the fmadd32 widens
  // exactly (recorded: fmadd32_r14_r6l_r7h_r8), so that it reads two 32-bit
  // registers, the one-cycle form, not three (L15-E371).
  if (HalfConstantFMASources) {
    DenseMap<MachineInstr *, Register> Half;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(MBB)) {
        if (MI.getOpcode() != AGX::FMADD32_F32)
          continue;
        // (dst, a_mods, a, b_mods, b, c_mods, c)
        SmallSet<Register, 3> Regs;
        bool AllRegs = true;
        for (unsigned I : {2u, 4u, 6u}) {
          const MachineOperand &O = MI.getOperand(I);
          if (!O.isReg() || !O.getReg().isVirtual() || O.getSubReg() ||
              MRI.getRegClass(O.getReg()) != &AGX::GPR32RegClass)
            AllRegs = false;
          else
            Regs.insert(O.getReg());
        }
        if (!AllRegs || Regs.size() != 3)
          continue;
        // An fmadd32 of three distinct 32-bit registers.
        auto Converts = [&](const MachineInstr &U) {
          if (U.getOpcode() != AGX::FMADD32_F32)
            return false;
          SmallSet<Register, 3> Rs;
          for (unsigned I : {2u, 4u, 6u}) {
            const MachineOperand &O = U.getOperand(I);
            if (!O.isReg() || !O.getReg().isVirtual() || O.getSubReg() ||
                MRI.getRegClass(O.getReg()) != &AGX::GPR32RegClass)
              return false;
            Rs.insert(O.getReg());
          }
          return Rs.size() == 3;
        };
        for (unsigned I : {2u, 4u, 6u}) {
          Register R = MI.getOperand(I).getReg();
          MachineInstr *Def = MRI.getVRegDef(R);
          if (!Def || Def->getOpcode() != AGX::MOV_IMM32 ||
              !Def->getOperand(1).isImm())
            continue;
          APFloat V(APFloat::IEEEsingle(),
                    APInt(32, uint32_t(Def->getOperand(1).getImm())));
          if (V.isNaN())
            continue;
          APFloat H = V;
          bool Loses = false;
          H.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Loses);
          if (Loses)
            continue;
          // The 32-bit constant goes (all its uses convert), or stays where
          // it was hoisted to: a second mov_imm next to the fmadd would cost
          // what the fmadd saves.
          if (Def->getParent() == MI.getParent() &&
              !all_of(MRI.use_nodbg_instructions(R),
                      [&](const MachineInstr &U) { return Converts(U); }))
            continue;
          Register &R16 = Half[Def];
          if (!R16) {
            R16 = MRI.createVirtualRegister(&AGX::GPR16RegClass);
            BuildMI(*Def->getParent(), std::next(Def->getIterator()),
                    Def->getDebugLoc(), TII->get(AGX::MOV_IMM16), R16)
                .addImm(H.bitcastToAPInt().getZExtValue());
          }
          static const unsigned Opc[3] = {AGX::FMADD32_M163232,
                                          AGX::FMADD32_M321632,
                                          AGX::FMADD32_M323216};
          MachineInstrBuilder NewMI =
              BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(Opc[(I - 2) / 2]),
                      MI.getOperand(0).getReg());
          for (unsigned J = 1; J < 7; ++J) {
            if (J == I)
              NewMI.addReg(R16);
            else
              NewMI.add(MI.getOperand(J));
          }
          MI.eraseFromParent();
          if (MRI.use_nodbg_empty(R))
            Def->eraseFromParent(), Half.erase(Def);
          Changed = true;
          break;
        }
      }
  }
  Changed |= chainConstantAdds(MF, MRI, TII);
  return Changed;
}
