//===-- AGXSpecialRegisterRanges.cpp - Bounds of special registers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The thread and threadgroup indices have known bounds: a threadgroup has at
// most 1024 threads, a SIMD-group 32, and grid positions and sizes are below
// 2^31 (CUDA's limits, within which CUDA launches stay); a kernel promising
// at most N threads a threadgroup ("agx-max-threads-per-threadgroup", from
// PTX's .maxntid or MLIR's known_block_size) has its threadgroup positions
// below N and its SIMD-group index below N / 32. This pass gives the
// special register reads range return attributes just before instruction
// selection, which turns them into AssertZext nodes: signed divisions and
// remainders of indices by powers of two then select as shifts and masks.
//
// The bounds are not given earlier (as NVPTX does in the intrinsic table):
// with them, IndVarSimplify widens induction variables to 64 bits and loop
// strength reduction adds 64-bit byte offsets next to the 32-bit indices,
// which costs registers (llvm/utils/agx-bench EXPERIMENTS.md L14-E1).
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-special-register-ranges"

static cl::opt<bool> AGXSpecialRegisterRanges(
    "agx-sr-ranges", cl::init(true), cl::Hidden,
    cl::desc("Bound the thread and threadgroup indices before instruction "
             "selection"));

static cl::opt<bool> AGXThreadgroupRanges(
    "agx-threadgroup-ranges", cl::init(true), cl::Hidden,
    cl::desc("Bound the threadgroup's special registers by the kernel's "
             "agx-max-threads-per-threadgroup"));

static cl::opt<bool> AGXThreadgroupShapeRanges(
    "agx-threadgroup-shape-ranges", cl::init(true), cl::Hidden,
    cl::desc("Bound the threadgroup's positions and sizes by the kernel's "
             "agx-threads-per-threadgroup"));

static cl::opt<cl::boolOrDefault> AGXKnownGridRanges(
    "agx-known-grid-ranges", cl::Hidden,
    cl::desc("Bound grid positions by the kernel's known grid "
             "(agx-threads-per-grid, agx-threadgroups-per-grid); by default "
             "in kernels without loops"));

namespace {
class AGXSpecialRegisterRangesPass : public FunctionPass {
public:
  static char ID;
  AGXSpecialRegisterRangesPass() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "AGX Special Register Ranges";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
} // namespace

char AGXSpecialRegisterRangesPass::ID = 0;

INITIALIZE_PASS(AGXSpecialRegisterRangesPass, DEBUG_TYPE,
                "AGX Special Register Ranges", false, false)

FunctionPass *llvm::createAGXSpecialRegisterRangesPass() {
  return new AGXSpecialRegisterRangesPass();
}

/// [Lo, Hi) of the special register an intrinsic reads, if bounded.
static std::optional<std::pair<uint64_t, uint64_t>> bounds(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::agx_thread_position_in_threadgroup_x:
  case Intrinsic::agx_thread_position_in_threadgroup_y:
  case Intrinsic::agx_thread_position_in_threadgroup_z:
  case Intrinsic::agx_thread_index_in_threadgroup:
    return std::pair(0, 1024);
  case Intrinsic::agx_threads_per_threadgroup_x:
  case Intrinsic::agx_threads_per_threadgroup_y:
  case Intrinsic::agx_threads_per_threadgroup_z:
    return std::pair(1, 1025);
  case Intrinsic::agx_thread_index_in_simdgroup:
  case Intrinsic::agx_simdgroup_index_in_threadgroup:
    return std::pair(0, 32);
  case Intrinsic::agx_threadgroup_position_in_grid_x:
  case Intrinsic::agx_threadgroup_position_in_grid_y:
  case Intrinsic::agx_threadgroup_position_in_grid_z:
    return std::pair(0, 0x7fffffff);
  case Intrinsic::agx_threadgroups_per_grid_x:
  case Intrinsic::agx_threadgroups_per_grid_y:
  case Intrinsic::agx_threadgroups_per_grid_z:
    return std::pair(1, 0x80000000);
  default:
    return std::nullopt;
  }
}

/// The tighter bounds of the threadgroup's registers when the launch
/// promises at most N threads a threadgroup ("agx-max-threads-per-threadgroup",
/// as block barriers rely on): each extent is at most N.
static std::optional<std::pair<uint64_t, uint64_t>>
threadgroupBounds(Intrinsic::ID ID, uint64_t N) {
  switch (ID) {
  case Intrinsic::agx_thread_position_in_threadgroup_x:
  case Intrinsic::agx_thread_position_in_threadgroup_y:
  case Intrinsic::agx_thread_position_in_threadgroup_z:
  case Intrinsic::agx_thread_index_in_threadgroup:
    return std::pair(uint64_t(0), N);
  case Intrinsic::agx_threads_per_threadgroup_x:
  case Intrinsic::agx_threads_per_threadgroup_y:
  case Intrinsic::agx_threads_per_threadgroup_z:
    return std::pair(uint64_t(1), N + 1);
  case Intrinsic::agx_simdgroup_index_in_threadgroup:
    return std::pair(uint64_t(0), divideCeil(N, 32));
  default:
    return std::nullopt;
  }
}

/// The bounds of the threadgroup's registers for a launch of exactly the
/// threadgroup shape E ("agx-threads-per-threadgroup", MLIR's
/// known_block_size): each position below its extent, each size exactly it.
static std::optional<std::pair<uint64_t, uint64_t>>
shapeBounds(Intrinsic::ID ID, const std::array<uint64_t, 3> &E) {
  switch (ID) {
  case Intrinsic::agx_thread_position_in_threadgroup_x:
    return std::pair(uint64_t(0), E[0]);
  case Intrinsic::agx_thread_position_in_threadgroup_y:
    return std::pair(uint64_t(0), E[1]);
  case Intrinsic::agx_thread_position_in_threadgroup_z:
    return std::pair(uint64_t(0), E[2]);
  case Intrinsic::agx_threads_per_threadgroup_x:
    return std::pair(E[0], E[0] + 1);
  case Intrinsic::agx_threads_per_threadgroup_y:
    return std::pair(E[1], E[1] + 1);
  case Intrinsic::agx_threads_per_threadgroup_z:
    return std::pair(E[2], E[2] + 1);
  default:
    return std::nullopt;
  }
}

/// The extents "a,b,c" a kernel attribute gives (agx-threads-per-grid,
/// agx-threadgroups-per-grid: the launch's known grid), each in [1, 2^32).
static std::optional<std::array<uint64_t, 3>> extents(const Function &F,
                                                      StringRef Name) {
  Attribute A = F.getFnAttribute(Name);
  if (!A.isStringAttribute())
    return std::nullopt;
  SmallVector<StringRef, 3> Parts;
  A.getValueAsString().split(Parts, ',');
  if (Parts.size() != 3)
    return std::nullopt;
  std::array<uint64_t, 3> E;
  for (unsigned D = 0; D < 3; ++D)
    if (Parts[D].getAsInteger(10, E[D]) || E[D] == 0 || E[D] >= (1ull << 32))
      return std::nullopt;
  return E;
}

/// [0, extent) of a grid position the kernel's known grid bounds.
static std::optional<std::pair<uint64_t, uint64_t>>
gridBounds(const Function &F, Intrinsic::ID ID) {
  unsigned D;
  StringRef Name;
  switch (ID) {
  case Intrinsic::agx_thread_position_in_grid_x: D = 0; Name = "agx-threads-per-grid"; break;
  case Intrinsic::agx_thread_position_in_grid_y: D = 1; Name = "agx-threads-per-grid"; break;
  case Intrinsic::agx_thread_position_in_grid_z: D = 2; Name = "agx-threads-per-grid"; break;
  case Intrinsic::agx_threadgroup_position_in_grid_x: D = 0; Name = "agx-threadgroups-per-grid"; break;
  case Intrinsic::agx_threadgroup_position_in_grid_y: D = 1; Name = "agx-threadgroups-per-grid"; break;
  case Intrinsic::agx_threadgroup_position_in_grid_z: D = 2; Name = "agx-threadgroups-per-grid"; break;
  default:
    return std::nullopt;
  }
  if (auto E = extents(F, Name))
    return std::pair(uint64_t(0), (*E)[D]);
  return std::nullopt;
}

std::optional<uint64_t> AGX::knownGridExtent(const Function &F,
                                            unsigned IntrinsicID) {
  if (auto G = gridBounds(F, Intrinsic::ID(IntrinsicID)))
    return G->second;
  return std::nullopt;
}

bool AGX::addSpecialRegisterRanges(Function &F,
                                   SmallVectorImpl<WeakVH> *Added) {
  bool Changed = false;
  // The known grid's bounds only in a kernel without loops: in a loop, the
  // offsets they let the DAG fold into addresses become immediates of a
  // 64-bit base recomputed each iteration (L15-E191).
  bool GridRanges = AGXKnownGridRanges == cl::boolOrDefault::BOU_TRUE;
  if (AGXKnownGridRanges == cl::boolOrDefault::BOU_UNSET) {
    SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 4> Edges;
    FindFunctionBackedges(F, Edges);
    GridRanges = Edges.empty();
  }
  uint64_t MaxThreads = 0;
  if (AGXThreadgroupRanges)
    MaxThreads =
        F.getFnAttributeAsParsedInteger("agx-max-threads-per-threadgroup", 0);
  if (MaxThreads > 1024)
    MaxThreads = 0;
  std::optional<std::array<uint64_t, 3>> Shape;
  if (AGXThreadgroupShapeRanges)
    Shape = extents(F, "agx-threads-per-threadgroup");
  if (Shape && (*Shape)[0] * (*Shape)[1] * (*Shape)[2] > 1024)
    Shape.reset();
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !II->getType()->isIntegerTy(32))
      continue;
    auto B = bounds(II->getIntrinsicID());
    if (B && MaxThreads)
      if (auto T = threadgroupBounds(II->getIntrinsicID(), MaxThreads))
        B = T;
    if (B && Shape)
      if (auto T = shapeBounds(II->getIntrinsicID(), *Shape))
        if (T->second - T->first < B->second - B->first)
          B = T;
    if (GridRanges)
      if (auto G = gridBounds(F, II->getIntrinsicID()))
        if (!B || G->second - G->first < B->second - B->first)
          B = G;
    if (!B || II->getRetAttributes().hasAttribute(Attribute::Range))
      continue;
    II->addRetAttr(Attribute::get(
        F.getContext(), Attribute::Range,
        ConstantRange(APInt(32, B->first), APInt(32, B->second))));
    if (Added)
      Added->push_back(II);
    Changed = true;
  }
  return Changed;
}

bool AGXSpecialRegisterRangesPass::runOnFunction(Function &F) {
  if (!AGXSpecialRegisterRanges || skipFunction(F))
    return false;
  return AGX::addSpecialRegisterRanges(F);
}
