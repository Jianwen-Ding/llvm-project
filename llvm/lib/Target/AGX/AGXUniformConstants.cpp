//===-- AGXUniformConstants.cpp - Wide constants in uniform registers -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A 32-bit constant that does not fit an instruction's immediate is a mov_imm
// into a general purpose register, which stays live wherever the constant is
// used (a loop's step, a mask). In a kernel, such constants can instead be
// preloaded by the driver into uniform registers after the arguments, like the
// arguments themselves (preloaded at no measured cost): each mov_imm becomes a
// copy of a uniform register, which AGXFoldOperands then folds into the
// instructions that can read uniforms. The kernel's .agx.kernels record lists
// the values (extension 6, AGXAsmPrinter), for the driver to set.
//
// Launch-time specialization (runtime) turns arguments into constants, which
// then held general purpose registers: rmsnorm specialized went from 14 to 21
// and lost a quarter of its resident SIMD-groups (llvm/utils/agx-bench,
// EXPERIMENTS.md R21). Off by default (-agx-uniform-constants): every driver
// must set the extension's uniforms first.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXSubtarget.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-uniform-constants-pass"

static cl::opt<bool> AGXUniformConstants(
    "agx-uniform-constants", cl::init(false), cl::Hidden,
    cl::desc("Preload the wide 32-bit constants of kernels into uniform "
             "registers (.agx.kernels extension 6) instead of mov_imm"));

/// The uniform file is 256 32-bit registers (512 halves).
static constexpr unsigned UniformHalves = 512;
/// At most this many constants per kernel.
static constexpr unsigned MaxConstants = 128;

namespace {

class AGXUniformConstantsPass : public MachineFunctionPass {
public:
  static char ID;
  AGXUniformConstantsPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX Uniform Constants"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char AGXUniformConstantsPass::ID = 0;

INITIALIZE_PASS_BEGIN(AGXUniformConstantsPass, DEBUG_TYPE,
                      "AGX Uniform Constants", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AGXUniformConstantsPass, DEBUG_TYPE, "AGX Uniform Constants",
                    false, false)

FunctionPass *llvm::createAGXUniformConstantsPass() {
  return new AGXUniformConstantsPass();
}

bool AGXUniformConstantsPass::runOnMachineFunction(MachineFunction &MF) {
  if (!AGXUniformConstants || skipFunction(MF.getFunction()) ||
      MF.getFunction().getCallingConv() != CallingConv::AGX_KERNEL)
    return false;
  const AGXSubtarget &ST = MF.getSubtarget<AGXSubtarget>();
  const AGXInstrInfo *TII = ST.getInstrInfo();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  auto *MFI = MF.getInfo<AGXMachineFunctionInfo>();

  unsigned Half = alignTo(MFI->getUniformHalfCount(), 2);
  // The constants that take the slots are the ones read most, a use in a loop
  // weighing 8 per level of depth (not the first ones in program order: a
  // kernel with hundreds of distinct constants, as Cycles', would give the
  // slots to the code before its loops).
  const MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  struct Use {
    uint64_t Weight = 0;
    unsigned First = 0; // order of first appearance: ties keep program order
  };
  DenseMap<uint32_t, Use> Uses;
  unsigned Order = 0;
  for (MachineBasicBlock &MBB : MF) {
    unsigned Depth = std::min(MLI.getLoopDepth(&MBB), 6u);
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == AGX::MOV_IMM32 && MI.getOperand(1).isImm()) {
        auto [It, New] =
            Uses.try_emplace(uint32_t(MI.getOperand(1).getImm()));
        if (New)
          It->second.First = Order++;
        It->second.Weight += uint64_t(1) << (3 * Depth);
      }
  }
  SmallVector<std::tuple<uint64_t, unsigned, uint32_t>, 0> Ranked;
  for (auto [Value, U] : Uses)
    Ranked.push_back({U.Weight, U.First, Value});
  llvm::sort(Ranked, [](const auto &A, const auto &B) {
    return std::get<0>(A) != std::get<0>(B) ? std::get<0>(A) > std::get<0>(B)
                                            : std::get<1>(A) < std::get<1>(B);
  });
  DenseMap<uint32_t, MCRegister> Slots;
  for (auto [W, First, Value] : Ranked) {
    if (Slots.size() >= MaxConstants || Half + 2 > UniformHalves)
      break;
    MCRegister Reg = AGX::getRegister(AGX::RegKind::UGPR32, Half, *TRI);
    if (!Reg)
      break;
    Slots[Value] = Reg;
    MFI->UniformConstants.push_back({Half, Value});
    Half += 2;
  }
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.getOpcode() != AGX::MOV_IMM32 || !MI.getOperand(1).isImm())
        continue;
      MCRegister Reg = Slots.lookup(uint32_t(MI.getOperand(1).getImm()));
      if (!Reg)
        continue;
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY),
              MI.getOperand(0).getReg())
          .addReg(Reg);
      MI.eraseFromParent();
      Changed = true;
    }
  }
  if (Changed)
    MFI->setUniformHalfCount(Half);
  return Changed;
}
