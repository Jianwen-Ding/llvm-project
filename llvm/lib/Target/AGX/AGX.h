//===-- AGX.h - Top-level interface for AGX ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry points of the AGX code generator passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGX_H
#define LLVM_LIB_TARGET_AGX_AGX_H

#include <optional>
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"

namespace llvm {
class Function;
class SCEV;
class ScalarEvolution;
/// The most 32-bit registers that F's IR values hold at once (SSA liveness,
/// each value by its size): an estimate of its register pressure before
/// instruction selection.
unsigned agxEstimatePressure(const Function &F);

class AGXTargetMachine;
class ScheduleDAGMutation;
class ScheduleDAGMI;
struct MachineSchedContext;
class FunctionPass;
class IRBuilderBase;
class Module;
class ModulePass;
class PassRegistry;

bool agxExperimentalLeafCallsEnabled();

// IR passes.
ModulePass *createAGXAlwaysInlinePass();
ModulePass *createAGXGlobalDataArgsPass();
ModulePass *createAGXLowerAIRPass(const AGXTargetMachine *TM);
ModulePass *createAGXPromoteKernelArgsPass();
ModulePass *createAGXCheckedAccessesPass(const AGXTargetMachine *TM);
FunctionPass *createAGXCodeGenPreparePass();
FunctionPass *createAGXStrictFPPass();
FunctionPass *createAGXLowerFPEnvPass();
FunctionPass *createAGXSoftF64Pass();
FunctionPass *createAGXExpandWideAtomicsPass(const AGXTargetMachine *TM = nullptr);
FunctionPass *createAGXAnnotateControlFlowPass();
FunctionPass *createAGXSinkAddressesPass();
FunctionPass *createAGXRareFallbacksPass();
FunctionPass *createAGXThreadgroupAddressesPass();
FunctionPass *createAGXHoistGuardedLoadsPass();
FunctionPass *createAGXAliasVersioningPass();
FunctionPass *createAGXAlignmentVersioningPass();
FunctionPass *createAGXNarrowInductionsPass();
FunctionPass *createAGXSpecialRegisterRangesPass();
namespace AGX {
/// Gives the special register reads of F their range return attributes (see
/// AGXSpecialRegisterRanges.cpp), and lists the calls it gave one in Added.
/// With ForAnalysis (ranges the caller takes off again), also the kernel's
/// upper bounds and the known grid's, which codegen itself leaves out.
bool addSpecialRegisterRanges(Function &F,
                              SmallVectorImpl<WeakVH> *Added = nullptr,
                              bool ForAnalysis = false);
/// The extent of the grid position an intrinsic reads that the kernel's known
/// grid gives (agx-threads-per-grid, agx-threadgroups-per-grid), if any.
std::optional<uint64_t> knownGridExtent(const Function &F, unsigned IntrinsicID);
/// The extent a position read is below: the known grid's or the kernel's
/// agx-upper-bounds (AGXSpecialRegisterRanges, L15-E391).
std::optional<uint64_t> knownUpperBound(const Function &F, unsigned IntrinsicID);
/// An upper bound on an integer SCEV's value from constants, positions, sums,
/// products, extensions, truncations, minimums and bounded recurrences
/// (AGXNarrowInductions' upperBound), as a 128-bit APInt.
/// AGXCodeGenPrepare's narrowing of 64-bit trees below 2^32 to 32 bits
/// (L15-E405), which AGXNarrowInductions runs again once the counters are
/// narrowed (L15-E426).
bool narrowBoundedI64(Function &F);
std::optional<APInt> scevUpperBound(const SCEV *S, ScalarEvolution &SE,
                                    const Function &F);
} // namespace AGX
FunctionPass *createAGXVectorAccessesPass(bool BeforeVectorizer);
FunctionPass *createAGXLowerGenericAddressesPass();

// Replaces recursive functions by state machines over an explicit stack
// (AGXLowerRecursion.cpp).
bool lowerAGXRecursion(Module &M);

// Instruction selection.
FunctionPass *createAGXISelDag(AGXTargetMachine &TM);

// Machine passes.
FunctionPass *createAGXFoldOperandsPass();
FunctionPass *createAGXLoadLanesLivePass();
FunctionPass *createAGXUniformConstantsPass();
FunctionPass *createAGXHoistFromElseFlowPass();
FunctionPass *createAGXSinkRematPass();
FunctionPass *createAGXLowerControlFlowPass();
FunctionPass *createAGXInsertWaitsPass();
FunctionPass *createAGXColdRegionsPass();

// Machine scheduler: keeps runs of memory accesses of one kind together.
std::unique_ptr<ScheduleDAGMutation> createAGXMemClusterMutation();
/// Whether MI is a device load (one of AGXMemCluster's kinds).
bool isAGXDeviceLoad(const MachineInstr &MI);
/// The post-RA machine scheduler: AGXMemClusterMutation, and cluster members
/// kept together ahead of latency stalls.
ScheduleDAGMI *createAGXPostRAScheduler(MachineSchedContext *C);

void initializeAGXAlwaysInlinePass(PassRegistry &);
void initializeAGXLowerAIRLegacyPass(PassRegistry &);
void initializeAGXPromoteKernelArgsPass(PassRegistry &);
void initializeAGXCheckedAccessesPass(PassRegistry &);
void initializeAGXCodeGenPreparePass(PassRegistry &);
void initializeAGXStrictFPPass(PassRegistry &);
void initializeAGXLowerFPEnvPass(PassRegistry &);
void initializeAGXSoftF64Pass(PassRegistry &);
void initializeAGXExpandWideAtomicsPass(PassRegistry &);
void initializeAGXAnnotateControlFlowPass(PassRegistry &);
void initializeAGXSinkAddressesPass(PassRegistry &);
void initializeAGXRareFallbacksPass(PassRegistry &);
void initializeAGXThreadgroupAddressesPass(PassRegistry &);
void initializeAGXHoistGuardedLoadsPass(PassRegistry &);
void initializeAGXAliasVersioningPass(PassRegistry &);
void initializeAGXAlignmentVersioningPass(PassRegistry &);
void initializeAGXNarrowInductionsPass(PassRegistry &);
void initializeAGXSpecialRegisterRangesPassPass(PassRegistry &);
void initializeAGXVectorAccessesPass(PassRegistry &);
void initializeAGXLowerGenericAddressesPass(PassRegistry &);
void initializeAGXDAGToDAGISelLegacyPass(PassRegistry &);
void initializeAGXFoldOperandsPass(PassRegistry &);
void initializeAGXLoadLanesPass(PassRegistry &);
void initializeAGXUniformConstantsPassPass(PassRegistry &);
void initializeAGXHoistFromElseFlowPass(PassRegistry &);
void initializeAGXSinkRematPass(PassRegistry &);
void initializeAGXLowerControlFlowPass(PassRegistry &);
void initializeAGXInsertWaitsPass(PassRegistry &);
void initializeAGXColdRegionsPass(PassRegistry &);
void initializeAGXAsmPrinterPass(PassRegistry &);

class TargetMachine;

/// Compiles a module described by !air.kernel (AIR) as AGX kernels.
class AGXLowerAIRPass : public RequiredPassInfoMixin<AGXLowerAIRPass> {
  const TargetMachine &TM;

public:
  explicit AGXLowerAIRPass(const TargetMachine &TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class Instruction;
class IRBuilderBase;
class MachineInstr;
class Type;
class Value;

namespace AGX {
/// Whether expandMath builds intrinsic \p ID of an operand of type \p Ty
/// (and with afn if \p Approx): tan, asin, acos, atan, atan2, sinh, cosh and
/// tanh of float, half or bfloat (or vectors), and the precise (not afn)
/// float sin, cos, sincos, exp, exp2, exp10, log, log2, log10 and pow.
bool canExpandMath(Intrinsic::ID ID, Type *Ty, bool Approx);
/// The math functions AGX has no instruction for, from the ones it has
/// (AGXMathExpansion.cpp): the intrinsic \p ID of \p Args, or AIR's \p Fn
/// (asinh, acosh, atanh, sinpi, cospi, tanpi) of \p X; nullptr for another.
/// \p FlushF32: f32 arithmetic flushes denormals (no ieee-fp32), so the
/// expansion need not keep denormal operands.
/// \p ExpNoOverflow: exp's argument is known not to reach its overflow
/// (a softmax's x - max), so no clamp above and no +inf select.
/// \p NativeNaN: f32 arithmetic's NaN is 0x7fc00000 (no canonical-nan).
/// \p ExpResultFlushed: exp's result only reaches f32 arithmetic, which
/// flushes a denormal operand.
Value *expandMath(IRBuilderBase &B, Intrinsic::ID ID, ArrayRef<Value *> Args,
                  bool FlushF32 = false, bool ExpNoOverflow = false,
                  bool NativeNaN = false, bool ExpResultFlushed = false);
Value *expandMath(IRBuilderBase &B, StringRef Fn, Value *X);

/// Whether a device memory access must reach memory shared with the host while
/// the dispatch runs: volatile and system-scope atomic loads and stores
/// (PTX ld/st.volatile, .sys). They are encoded coherent, loads after an f59d
/// and stores before an f533 (ptx-re-blockers B7).
bool isHostCoherentAccess(const Instruction &I);
bool isHostCoherentAccess(const MachineInstr &MI);
} // namespace AGX

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGX_H
