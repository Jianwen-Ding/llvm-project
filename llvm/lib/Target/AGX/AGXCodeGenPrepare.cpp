//===-- AGXCodeGenPrepare.cpp - AGX IR preparation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IR transformations that AGX code generation relies on:
//
//  * AGXAlwaysInline normally marks every non-kernel function always_inline
//    (and internal, so it disappears once inlined). The opt-in call path keeps
//    eligible functions out of this expansion and removes unused functions.
//    The functions it keeps that use the global data buffer get its address
//    as a parameter (AGXGlobalDataArgs, after inlining): only a kernel has it,
//    in a uniform register.
//  * AGXCodeGenPrepare expands what the hardware has no instruction or
//    library for: integer division, and memory intrinsics that are not
//    small enough to be expanded by instruction selection. There is no
//    floating point environment, so constrained floating point intrinsics
//    become their ordinary operations. It also unifies the function exits,
//    which the control flow structurizer expects.
//  * On subtargets where simd_shuffle merges the low index bits of each quad
//    (quad-shuffle-index-merge), shuffles with divergent indices are
//    expanded into four shuffles whose low index bits are uniform, like
//    Mesa's agx_nir_lower_subgroups.
//  * The hardware flushes 32-bit denormals but keeps 16-bit ones. Functions
//    whose denormal mode for half asks for flushing get explicit flushes of
//    the operands and results of their half operations.
//  * There is no double precision hardware: the double precision arithmetic
//    that AGXSoftF64 does not implement is diagnosed as unsupported.
//  * Tangent, the inverse and the hyperbolic functions are built from the
//    functions the hardware has (AGXMathExpansion.cpp).
//  * A threadgroup of at most 4 SIMD-groups is one aligned block, whose
//    barrier is cheaper: on subtargets with block-barrier its threadgroup
//    barriers become block barriers.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXMachineFunctionInfo.h"
#include "AGXSubtarget.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomConditionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BypassSlowDivision.h"
#include "llvm/Transforms/Utils/CallPromotionUtils.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"
#include <cmath>
#include <map>
#include <set>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "agx-codegenprepare"

//===----------------------------------------------------------------------===//
// AGXAlwaysInline
//===----------------------------------------------------------------------===//

namespace {

class AGXAlwaysInline : public ModulePass {
public:
  static char ID;
  AGXAlwaysInline() : ModulePass(ID) {}

  StringRef getPassName() const override { return "AGX Always Inline"; }
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace

char AGXAlwaysInline::ID = 0;

INITIALIZE_PASS(AGXAlwaysInline, "agx-always-inline", "AGX Always Inline",
                false, false)

ModulePass *llvm::createAGXAlwaysInlinePass() { return new AGXAlwaysInline(); }

static bool allocateExperimentalCallType(Type *Ty, unsigned &Slots) {
  if (Ty->isVoidTy())
    return true;
  if (auto *Array = dyn_cast<ArrayType>(Ty)) {
    for (uint64_t I = 0; I < Array->getNumElements(); ++I)
      if (!allocateExperimentalCallType(Array->getElementType(), Slots))
        return false;
    return true;
  }
  if (Ty->isPointerTy() || Ty->isIntegerTy(64)) {
    Slots = alignTo(Slots, 2) + 2;
  } else if (Ty->isIntegerTy(16) || Ty->isIntegerTy(32) ||
             Ty->isHalfTy() || Ty->isFloatTy()) {
    ++Slots;
  } else {
    return false;
  }
  return Slots <= 28;
}

static bool canKeepExperimentalCall(const Function &F) {
  if (F.isDeclaration() || F.hasFnAttribute(Attribute::AlwaysInline) ||
      F.isVarArg() || F.hasStructRetAttr())
    return false;
  // (two slots kept for the global data buffer's address, which
  // AGXGlobalDataArgs may add)
  unsigned ArgSlots = 2;
  for (const Argument &Arg : F.args())
    if (!allocateExperimentalCallType(Arg.getType(), ArgSlots))
      return false;
  unsigned ResultSlots = 0;
  if (!allocateExperimentalCallType(F.getReturnType(), ResultSlots))
    return false;
  unsigned Returns = 0;
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB) {
      if (isa<CallBrInst>(I))
        return false;
      if (const auto *AI = dyn_cast<AllocaInst>(&I))
        if (!AI->isStaticAlloca())
          return false;
      if (isa<ReturnInst>(I))
        ++Returns;
      if (const auto *CB = dyn_cast<CallBase>(&I)) {
        const Function *Callee = CB->getCalledFunction();
        if (!Callee ||
            (Callee->isIntrinsic() &&
             (Callee->getIntrinsicID() == Intrinsic::agx_texture_sample ||
              Callee->getIntrinsicID() == Intrinsic::agx_threadgroups_per_grid_x ||
              Callee->getIntrinsicID() == Intrinsic::agx_threadgroups_per_grid_y ||
              Callee->getIntrinsicID() == Intrinsic::agx_threadgroups_per_grid_z)))
          return false;
      }
    }
  return Returns == 1;
}

/// The default backend inlines all calls. An indirect call becomes a test of
/// the callee against each address-taken function of its type, with a direct
/// call (inlined later) for each, and a trap if none
/// matches. The addresses of these functions, as values, become their
/// numbers (1, 2, ...), so that they can be stored, compared and passed.
static bool lowerIndirectCalls(Module &M) {
  SmallVector<CallBase *, 8> Indirect;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I); CB && CB->isIndirectCall() &&
                                             !CB->isInlineAsm())
        Indirect.push_back(CB);
  SmallVector<Function *, 8> Taken;
  for (Function &F : M)
    if (!F.isIntrinsic() && F.getCallingConv() != CallingConv::AGX_KERNEL &&
        F.hasAddressTaken(nullptr, /*IgnoreCallbackUses=*/false,
                          /*IgnoreAssumeLikeCalls=*/true,
                          /*IngoreLLVMUsed=*/true))
      Taken.push_back(&F);
  if (Indirect.empty() && Taken.empty())
    return false;
  for (CallBase *CB : Indirect) {
    CallBase *Rest = CB;
    for (Function *F : Taken) {
      if (F->getFunctionType() != CB->getFunctionType())
        continue;
      // The direct call is in the then block; the indirect one stays in the
      // else block.
      promoteCallWithIfThenElse(*Rest, F);
    }
    // No address-taken function of this type matches.
    IRBuilder<> B(Rest);
    B.CreateIntrinsic(Intrinsic::trap, {});
    if (!Rest->getType()->isVoidTy())
      Rest->replaceAllUsesWith(PoisonValue::get(Rest->getType()));
    Rest->eraseFromParent();
  }
  for (auto [N, F] : enumerate(Taken)) {
    Constant *Id = ConstantExpr::getIntToPtr(
        ConstantInt::get(Type::getInt64Ty(M.getContext()), N + 1), F->getType());
    F->replaceUsesWithIf(Id, [](Use &U) {
      auto *CB = dyn_cast<CallBase>(U.getUser());
      return !CB || !CB->isCallee(&U);
    });
  }
  return true;
}

bool AGXAlwaysInline::runOnModule(Module &M) {
  bool Changed = lowerIndirectCalls(M);
  Changed |= lowerAGXRecursion(M);
  // Modules without any agx_kernel function (for example tests of single
  // functions) compile every external function as an entry point.
  bool HasKernels = any_of(M, [](const Function &F) {
    return F.getCallingConv() == CallingConv::AGX_KERNEL;
  });

  // Compute a conservative static frame depth before inlining wrappers.
  // A shared callee uses its deepest path, leaving every active ancestor's
  // scratch and saved return address in a different region/register.
  DenseMap<const Function *, unsigned> Depth;
  bool CallsWithinLimit = true;
  if (agxExperimentalLeafCallsEnabled()) {
    for (Function &F : M)
      if (!F.isDeclaration() &&
          (F.getCallingConv() == CallingConv::AGX_KERNEL ||
           (!HasKernels && !F.hasLocalLinkage())))
        Depth[&F] = 0;
    for (unsigned Iter = 0; Iter != 8; ++Iter) {
      bool Updated = false;
      for (Function &F : M) {
        auto It = Depth.find(&F);
        if (It == Depth.end())
          continue;
        // (by value: inserting a callee below may move the map's entries)
        unsigned Next = It->second + 1;
        for (const Instruction &I : instructions(F)) {
          const auto *CB = dyn_cast<CallBase>(&I);
          const Function *Callee = CB ? CB->getCalledFunction() : nullptr;
          if (!Callee || Callee->isDeclaration() || Callee->isIntrinsic())
            continue;
          if (Next > Depth.lookup(Callee)) {
            Depth[Callee] = Next;
            Updated = true;
          }
        }
      }
      if (!Updated)
        break;
    }
    unsigned MaxDepth = 0;
    for (auto [F, D] : Depth)
      MaxDepth = std::max(MaxDepth, D);
    CallsWithinLimit = MaxDepth <= 7;
    if (CallsWithinLimit) {
      for (Function &F : M)
        if (auto It = Depth.find(&F); It != Depth.end()) {
          F.addFnAttr("agx-call-depth", Twine(It->second).str());
          Changed = true;
        }
      for (Function &F : M)
        if (F.getCallingConv() == CallingConv::AGX_KERNEL) {
          F.addFnAttr("agx-max-call-depth", Twine(MaxDepth).str());
          Changed = true;
        }
    }
  }

  SmallVector<Function *, 8> Dead;
  for (Function &F : M) {
    if (F.isDeclaration() || F.getCallingConv() == CallingConv::AGX_KERNEL)
      continue;
    if (!HasKernels && !F.hasLocalLinkage())
      continue;
    // With kernels in the module, other functions are only reachable from
    // them: internalize them so that they disappear once inlined, and drop
    // the unused ones (for example from linked device libraries).
    if (HasKernels && !F.hasLocalLinkage()) {
      F.setLinkage(GlobalValue::InternalLinkage);
      F.setVisibility(GlobalValue::DefaultVisibility);
      Changed = true;
    }
    if (F.use_empty()) {
      Dead.push_back(&F);
      continue;
    }
    if (F.hasFnAttribute(Attribute::NoInline))
      continue;
    if (agxExperimentalLeafCallsEnabled() && CallsWithinLimit &&
        canKeepExperimentalCall(F)) {
      F.addFnAttr(Attribute::NoInline);
      Changed = true;
      continue;
    }
    F.addFnAttr(Attribute::AlwaysInline);
    Changed = true;
  }
  for (Function *F : Dead)
    F->eraseFromParent();
  Changed |= !Dead.empty();
  // A module with real calls: its functions form call trees, which declare registers for the whole tree
  // (AGXSubtarget's -agx-call-tree-max-gprs)
  if (agxExperimentalLeafCallsEnabled() && CallsWithinLimit &&
      any_of(M, [](const Function &F) {
        return !F.isDeclaration() && F.getCallingConv() != CallingConv::AGX_KERNEL &&
               F.hasFnAttribute(Attribute::NoInline) && !F.use_empty();
      }))
    for (Function &F : M)
      if (!F.isDeclaration() && F.hasFnAttribute("agx-call-depth")) {
        F.addFnAttr("agx-call-tree");
        Changed = true;
      }
  // Called functions after every kernel: a kernel's program runs from its
  // symbol to the end of .text, which must hold what it calls (a module from
  // AIR lists its functions before its kernels).
  if (agxExperimentalLeafCallsEnabled() && HasKernels) {
    // (in increasing call depth: -agx-measured-call-frames places each
    // depth's frames after the shallower ones', recorded as they compile)
    SmallVector<Function *, 16> Called;
    for (Function &F : M)
      if (!F.isDeclaration() && F.getCallingConv() != CallingConv::AGX_KERNEL)
        Called.push_back(&F);
    auto DepthOf = [](const Function *F) {
      unsigned D = 0;
      if (Attribute A = F->getFnAttribute("agx-call-depth");
          A.isStringAttribute())
        (void)A.getValueAsString().getAsInteger(10, D);
      return D;
    };
    llvm::stable_sort(Called, [&](const Function *A, const Function *B) {
      return DepthOf(A) < DepthOf(B);
    });
    for (Function *F : Called) {
      F->removeFromParent();
      M.getFunctionList().push_back(F);
    }
    Changed |= !Called.empty();
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// AGXGlobalDataArgs
//===----------------------------------------------------------------------===//

namespace {

/// After inlining, the functions that are still called (the experimental
/// call path) and use the global data buffer, and their callers up to the
/// kernels, get its address as a last i64 parameter: a kernel has it in a
/// uniform register (LowerFormalArguments), a called function has no access
/// to it. In those functions each global data variable becomes that address
/// plus its offset in computeGlobalDataLayout's layout, which instruction
/// selection uses for the kernels' own accesses (no global is added or removed
/// after this pass). A kernel passes the address of the layout's first
/// variable less its offset.
class AGXGlobalDataArgs : public ModulePass {
public:
  static char ID;
  AGXGlobalDataArgs() : ModulePass(ID) {}
  StringRef getPassName() const override { return "AGX Global Data Arguments"; }
  bool runOnModule(Module &M) override;
};

} // namespace

char AGXGlobalDataArgs::ID = 0;

ModulePass *llvm::createAGXGlobalDataArgsPass() {
  return new AGXGlobalDataArgs();
}

bool AGXGlobalDataArgs::runOnModule(Module &M) {
  auto IsKernel = [](const Function &F) {
    return F.getCallingConv() == CallingConv::AGX_KERNEL;
  };
  SetVector<Function *> Need;
  for (Function &F : M)
    if (!F.isDeclaration() && !IsKernel(F) && AGX::usesGlobalData(F))
      Need.insert(&F);
  for (unsigned I = 0; I < Need.size(); ++I)
    for (User *U : Need[I]->users())
      if (auto *CB = dyn_cast<CallBase>(U))
        if (Function *Caller = CB->getFunction(); !IsKernel(*Caller))
          Need.insert(Caller);
  if (Need.empty())
    return false;
  AGX::GlobalDataLayout Layout = AGX::computeGlobalDataLayout(M);
  if (Layout.empty())
    return false;
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);

  // Each function with the parameter added (its body moved over).
  DenseMap<Function *, Function *> New;
  for (Function *F : Need) {
    FunctionType *FT = F->getFunctionType();
    SmallVector<Type *, 8> Params(FT->params());
    Params.push_back(I64);
    // (in F's place: AGXAlwaysInline ordered the called functions)
    Function *NF = Function::Create(
        FunctionType::get(FT->getReturnType(), Params, false), F->getLinkage(),
        F->getAddressSpace(), "");
    M.getFunctionList().insert(F->getIterator(), NF);
    NF->copyAttributesFrom(F);
    NF->copyMetadata(F, 0);
    NF->setCallingConv(F->getCallingConv());
    NF->splice(NF->begin(), F);
    for (auto [Old, NewArg] : zip(F->args(), NF->args())) {
      NewArg.takeName(&Old);
      Old.replaceAllUsesWith(&NewArg);
    }
    NF->getArg(FT->getNumParams())->setName("agx.global.data");
    NF->takeName(F);
    New[F] = NF;
  }

  // The global data variables' uses in those functions, as base + offset.
  SmallVector<Constant *, 16> Globals;
  for (auto &[GV, Offset] : Layout.Offsets)
    Globals.push_back(const_cast<GlobalVariable *>(GV));
  for (auto &[F, NF] : New) {
    convertUsersOfConstantsToInstructions(Globals, NF);
    Argument *Base = NF->getArg(NF->arg_size() - 1);
    for (Instruction &I : instructions(*NF))
      for (Use &U : I.operands()) {
        auto *GV = dyn_cast<GlobalVariable>(U.get());
        if (!GV)
          continue;
        std::optional<uint64_t> Offset = Layout.getOffset(GV);
        if (!Offset)
          continue;
        auto *Phi = dyn_cast<PHINode>(&I);
        IRBuilder<> B(Phi ? Phi->getIncomingBlock(U)->getTerminator() : &I);
        U.set(B.CreateIntToPtr(B.CreateAdd(Base, B.getInt64(*Offset)),
                               GV->getType()));
      }
  }

  // The calls, with the caller's address (a kernel's from the layout's
  // first variable).
  // (the calls are in the new functions, whose bodies they are now)
  SmallPtrSet<Function *, 16> Added;
  for (auto &[F, NF] : New)
    Added.insert(NF);
  DenseMap<Function *, Value *> KernelBase;
  auto BaseIn = [&](Function *Caller) -> Value * {
    if (Added.contains(Caller))
      return Caller->getArg(Caller->arg_size() - 1);
    Value *&V = KernelBase[Caller];
    if (!V) {
      auto &[GV, Offset] = Layout.Offsets.front();
      IRBuilder<> B(&*Caller->getEntryBlock().getFirstInsertionPt());
      V = B.CreateSub(
          B.CreatePtrToInt(const_cast<GlobalVariable *>(GV), I64),
          B.getInt64(Offset), "agx.global.data");
    }
    return V;
  };
  for (auto &[F, NF] : New) {
    for (User *U : make_early_inc_range(F->users())) {
      auto *CB = dyn_cast<CallInst>(U);
      if (!CB || CB->getCalledOperand() != F)
        report_fatal_error("AGX: a function using global data is used other "
                           "than called");
      SmallVector<Value *, 8> Args(CB->args());
      Args.push_back(BaseIn(CB->getFunction()));
      auto *NC = CallInst::Create(NF, Args, "", CB->getIterator());
      NC->setCallingConv(CB->getCallingConv());
      NC->setAttributes(CB->getAttributes());
      NC->setTailCallKind(CB->getTailCallKind());
      NC->setDebugLoc(CB->getDebugLoc());
      NC->takeName(CB);
      CB->replaceAllUsesWith(NC);
      CB->eraseFromParent();
    }
    F->eraseFromParent();
  }
  return true;
}

//===----------------------------------------------------------------------===//
// AGXCodeGenPrepare
//===----------------------------------------------------------------------===//

namespace {

class AGXCodeGenPrepare : public FunctionPass {
  bool expandDivisions(Function &F);
  bool expandMemIntrinsics(Function &F, const TargetTransformInfo &TTI);
  bool unifyExits(Function &F);
  bool lowerConstrainedFP(Function &F);
  bool lowerDivergentShuffles(Function &F, const TargetTransformInfo &TTI);
  bool foldLaneIdentities(Function &F);
  bool simdScans(Function &F, const AGXSubtarget &ST);
  bool simdReductions(Function &F, const AGXSubtarget &ST);
  bool mergeLoopExits(Function &F);
  bool mergeSameConditionIfs(Function &F);
  bool signedWideMuls(Function &F);
  bool forwardOwnSlots(Function &F);
  bool strideDivisions(Function &F);
  bool lowerMatrixFMA(Function &F, const AGXSubtarget &ST);
  bool flushHalfDenormals(Function &F);
  bool flushFloatMinMax(Function &F);
  bool diagnoseDoubles(Function &F);
  bool fenceHostCoherentAccesses(Function &F);
  bool useBlockBarriers(Function &F, const AGXSubtarget &ST);
  bool expandMath(Function &F, const AGXSubtarget &ST);
  bool combineUniformAtomics(Function &F, const TargetTransformInfo &TTI);

public:
  static char ID;
  AGXCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AGX CodeGen Prepare"; }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char AGXCodeGenPrepare::ID = 0;

INITIALIZE_PASS_BEGIN(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AGXCodeGenPrepare, DEBUG_TYPE, "AGX CodeGen Prepare",
                    false, false)

FunctionPass *llvm::createAGXCodeGenPreparePass() {
  return new AGXCodeGenPrepare();
}

/// Division by a power of two is a shift, and 32-bit division by another
/// constant is a multiply-high; both are left to instruction selection.
static cl::opt<bool> Div64ConstMulHi(
    "agx-div64-const-mulhi", cl::init(true), cl::Hidden,
    cl::desc("64-bit division by a constant as instruction selection's "
             "multiply-high (the 64-bit high product is four multiply-adds)"));

static bool isCheapDivision(const BinaryOperator &BO) {
  auto *C = dyn_cast<ConstantInt>(BO.getOperand(1));
  if (!C)
    return false;
  if (C->getValue().isPowerOf2())
    return true;
  return BO.getType()->isIntegerTy(32) ||
         (Div64ConstMulHi && BO.getType()->isIntegerTy(64));
}

static cl::opt<bool> AGXDivSignFromXor(
    "agx-div-sign-from-xor",
    cl::desc("Signed 32-bit division's quotient sign as (x ^ d) <s 0"),
    cl::init(true), cl::Hidden);

static cl::opt<bool> ShareDivisors(
    "agx-share-divisors", cl::init(true), cl::Hidden,
    cl::desc("Compute the reciprocal of a 32-bit divisor once where the "
             "divisor is defined, shared by every division by it"));

namespace {
/// The divisor-only part of a 32-bit division: |y|, whether y < 0, and the
/// estimate z of 2^32 / |y|.
struct Divisor {
  Value *Y, *Neg, *Z;
  Value *H; // the divisor itself (signed: before its absolute value)
};

/// Divisors computed once each. A divisor that is a cheap pure expression
/// (a front end's umax(y, 1), a trunc of a 64-bit value) is rebuilt right after
/// the latest of its leaves, so that copies of it in different blocks (the
/// unrolled iterations of a loop, the fast paths of 64-bit divisions) share
/// one reciprocal.
class DivisorCache {
  Function &F;
  DominatorTree DT;
  DenseMap<Value *, Value *> Hoisted;
  std::map<SmallVector<uintptr_t, 6>, Value *> Exprs;
  DenseMap<std::pair<Value *, unsigned>, Divisor> Divisors;

  static bool isCheapPure(const Instruction &I) {
    if (isa<CastInst, BinaryOperator, ICmpInst, SelectInst>(I))
      return !isa<BinaryOperator>(I) || !I.isIntDivRem();
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      switch (II->getIntrinsicID()) {
      case Intrinsic::umax:
      case Intrinsic::umin:
      case Intrinsic::smax:
      case Intrinsic::smin:
      case Intrinsic::abs:
        return true;
      default:
        break;
      }
    return false;
  }

  /// Where a value defined by V can first be used.
  BasicBlock::iterator after(Value *V) {
    if (auto *I = dyn_cast_or_null<Instruction>(V)) {
      if (isa<PHINode>(I))
        return I->getParent()->getFirstInsertionPt();
      return std::next(I->getIterator());
    }
    return F.getEntryBlock().getFirstInsertionPt();
  }

  /// The later of two definition points on one dominator chain.
  Value *later(Value *A, Value *B) {
    auto *IA = dyn_cast<Instruction>(A), *IB = dyn_cast<Instruction>(B);
    if (!IA)
      return B;
    if (!IB)
      return A;
    return DT.dominates(IA, IB) ? B : A;
  }

  /// V rebuilt after its leaves; Last is set to the latest leaf (or null).
  Value *hoist(Value *V, Value *&Last, unsigned Depth = 0) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I) {
      Last = nullptr;
      return V;
    }
    if (Depth > 4 || !isCheapPure(*I)) {
      Last = I;
      return I;
    }
    if (Value *H = Hoisted.lookup(I)) {
      Last = H;
      return H;
    }
    SmallVector<Value *, 3> Ops;
    Value *Latest = nullptr;
    SmallVector<uintptr_t, 6> Key{I->getOpcode(),
                                  reinterpret_cast<uintptr_t>(I->getType())};
    if (auto *Cmp = dyn_cast<CmpInst>(I))
      Key.push_back(Cmp->getPredicate());
    if (auto *II = dyn_cast<IntrinsicInst>(I))
      Key.push_back(II->getIntrinsicID());
    unsigned NOps = isa<CallInst>(I) ? cast<CallInst>(I)->arg_size()
                                     : I->getNumOperands();
    for (unsigned N = 0; N < NOps; ++N) {
      Value *L;
      Value *Op = hoist(I->getOperand(N), L, Depth + 1);
      Latest = L ? (Latest ? later(Latest, L) : L) : Latest;
      Ops.push_back(Op);
      Key.push_back(reinterpret_cast<uintptr_t>(Op));
    }
    if (I->getType()->isVectorTy()) {
      Last = I;
      return I;
    }
    Value *&E = Exprs[Key];
    if (!E) {
      Instruction *C = I->clone();
      C->setName(I->getName() + ".div");
      for (unsigned N = 0; N < NOps; ++N)
        C->setOperand(N, Ops[N]);
      C->dropPoisonGeneratingAnnotations();
      C->insertBefore(after(Latest));
      E = C;
    }
    Hoisted[I] = E;
    Last = E;
    return E;
  }

public:
  DivisorCache(Function &F) : F(F), DT(F) {}

  Divisor get(Value *Y, bool IsSigned, BinaryOperator &At) {
    Value *Last = nullptr;
    Value *H = ShareDivisors ? hoist(Y, Last) : Y;
    auto [It, New] = Divisors.try_emplace({H, unsigned(IsSigned)});
    if (!New)
      return It->second;
    IRBuilder<> B(ShareDivisors ? &*after(Last) : &At);
    Value *YAbs = H, *Neg = nullptr;
    if (IsSigned) {
      Neg = B.CreateICmpSLT(H, B.getInt32(0));
      YAbs = B.CreateSelect(Neg, B.CreateNeg(H), H);
    }
    Value *FY = B.CreateUIToFP(YAbs, B.getFloatTy());
    Value *Rcp = B.CreateUnaryIntrinsic(Intrinsic::agx_rcp, FY);
    Value *Scaled =
        B.CreateFMul(Rcp, ConstantFP::get(B.getFloatTy(), 4294966784.0));
    Value *Z = B.CreateFPToUI(Scaled, B.getInt32Ty());
    Value *P = B.CreateMul(
        B.CreateZExt(Z, B.getInt64Ty()),
        B.CreateZExt(B.CreateMul(B.CreateNeg(YAbs), Z), B.getInt64Ty()));
    Z = B.CreateAdd(Z, B.CreateTrunc(B.CreateLShr(P, 32), B.getInt32Ty()));
    Divisor D{YAbs, Neg, Z, H};
    if (ShareDivisors)
      It->second = D;
    else
      Divisors.erase(It);
    return D;
  }
};
} // namespace

/// A 32-bit division or remainder without a loop, as AMDGPU expands them
/// (AMDGPUCodeGenPrepare::expandDivRem32): an estimate of 2^32 / y from the
/// f32 reciprocal, one Newton-Raphson step in integers, then two corrections,
/// which is exact when the reciprocal is within 1 ulp. The hardware's rcp is
/// within 0.75 ulp for every mantissa (Apple-M2-Playground 2026-10-20-fdiv32).
/// Division by zero gives all ones (remainder: x), as on NVIDIA GPUs. The
/// divisor's part comes from Divisors, shared by divisions by the same value.
static Value *expandDivRem32(BinaryOperator &BO, DivisorCache &Divisors) {
  IRBuilder<> B(&BO);
  Instruction::BinaryOps Op = BO.getOpcode();
  bool IsSigned = Op == Instruction::SDiv || Op == Instruction::SRem;
  bool IsRem = Op == Instruction::URem || Op == Instruction::SRem;
  Type *I32 = B.getInt32Ty();
  Value *X = BO.getOperand(0);
  // umax(d, 1) (a PTX front end's guard of the division, which selects its
  // own result where d is zero): the expansion of d, whose zero divisor is
  // replaced below by the result of dividing by one, so no maximum.
  using namespace PatternMatch;
  Value *Y0 = BO.getOperand(1), *Guarded = nullptr;
  if (!IsSigned && match(Y0, m_UMax(m_Value(Guarded), m_One())))
    Y0 = Guarded;
  // A front end's guard of a signed division, d + 1 <u 2 ? 1 : d (d = 0 and
  // -1, whose results it selects itself): the expansion of d, as above
  // (d = -1 ? 1 : d where d is known nonzero)
  bool GuardsZero = !IsSigned;
  if (IsSigned &&
      match(Y0, m_Select(m_SpecificICmp(ICmpInst::ICMP_ULT,
                                        m_Add(m_Value(Guarded), m_One()),
                                        m_SpecificInt(2)),
                         m_One(), m_Deferred(Guarded)))) {
    Y0 = Guarded;
    GuardsZero = true;
  } else if (IsSigned &&
             match(Y0, m_Select(m_SpecificICmp(ICmpInst::ICMP_EQ,
                                               m_Value(Guarded), m_AllOnes()),
                                m_One(), m_Deferred(Guarded)))) {
    Y0 = Guarded;
  }
  Divisor D = Divisors.get(Y0, IsSigned, BO);
  Value *Y = D.Y, *Z = D.Z;
  Value *XNeg = nullptr, *Neg = nullptr;
  if (IsSigned) {
    XNeg = B.CreateICmpSLT(X, B.getInt32(0));
    // the quotient's sign: (x ^ d) <s 0, one xor and a compare the final
    // select takes, not x <s 0 xor d <s 0 (L15-E324)
    Neg = IsRem                ? XNeg
          : AGXDivSignFromXor ? B.CreateICmpSLT(B.CreateXor(X, D.H),
                                                 B.getInt32(0))
                               : B.CreateXor(XNeg, D.Neg);
    X = B.CreateSelect(XNeg, B.CreateNeg(X), X);
  }
  auto MulHi = [&](Value *A, Value *C) {
    Value *P = B.CreateMul(B.CreateZExt(A, B.getInt64Ty()), B.CreateZExt(C, B.getInt64Ty()));
    return B.CreateTrunc(B.CreateLShr(P, 32), I32);
  };
  Value *Q = MulHi(X, Z);
  Value *R = B.CreateSub(X, B.CreateMul(Q, Y));
  for (int I = 0; I < 2; ++I) {
    Value *C = B.CreateICmpUGE(R, Y);
    Q = B.CreateSelect(C, B.CreateAdd(Q, B.getInt32(1)), Q);
    R = B.CreateSelect(C, B.CreateSub(R, Y), R);
  }
  Value *Res = IsRem ? R : Q;
  if (IsSigned)
    Res = B.CreateSelect(Neg, B.CreateNeg(Res), Res);
  if (Guarded) {
    // where the uses (through at most two single-use selects of which it is
    // the false value) select another value for d = 0, as a front end's do, no
    // select
    // (signed: and for d = -1 too)
    Value *V = &BO;
    bool Zero = false, MinusOne = !IsSigned;
    for (int Depth = 0; Depth < 2 && V->hasOneUse(); ++Depth) {
      auto *Sel = dyn_cast<SelectInst>(V->user_back());
      if (!Sel || Sel->getFalseValue() != V)
        break;
      Zero |= match(
          Sel->getCondition(),
          m_SpecificICmp(ICmpInst::ICMP_EQ, m_Specific(Guarded), m_Zero()));
      MinusOne |= match(
          Sel->getCondition(),
          m_SpecificICmp(ICmpInst::ICMP_EQ, m_Specific(Guarded), m_AllOnes()));
      V = Sel;
    }
    if ((Zero || !GuardsZero) && MinusOne)
      return Res;
    // where the guard took 1 (d = 0, or signed d = -1): x / 1 = x, x % 1 = 0
    Value *One = !IsSigned    ? B.CreateICmpEQ(Y0, B.getInt32(0))
                 : GuardsZero ? B.CreateICmpULT(B.CreateAdd(Y0, B.getInt32(1)),
                                                B.getInt32(2))
                              : B.CreateICmpEQ(Y0, B.getInt32(-1));
    return B.CreateSelect(One, IsRem ? B.getInt32(0) : BO.getOperand(0), Res);
  }
  Value *Zero = B.CreateICmpEQ(Y0, B.getInt32(0));
  return B.CreateSelect(Zero, IsRem ? BO.getOperand(0) : B.getInt32(-1), Res);
}

namespace {
struct Guard16 {
  Value *D = nullptr;                          // the unguarded divisor
  SmallVector<std::pair<Instruction *, bool>, 4> Roots; // (value, remainder)
  SmallVector<Instruction *, 8> Dead;
};
} // namespace

/// A front end guards a signed division's divisor as d = -1 ? 1 : d, or as
/// d + 1 <u 2 ? 1 : d (d = 0 and -1), and selects its results itself there:
/// the quotient -x and all ones, the remainder 0 and x (the remainder taken
/// as x - q d' where DivRemPairs decomposed it). The expansion gives those
/// values when it divides by d itself (-x wrapping as the select's does,
/// x - q 0 = x, x - (-x)(-1) = 0), so where every use of the division is such
/// a select (or the decomposed remainder ending in them), it divides by d and
/// the selects go.
static std::optional<Guard16> matchGuard16(BinaryOperator &BO) {
  using namespace PatternMatch;
  bool IsRem = BO.getOpcode() == Instruction::SRem;
  Value *Y = BO.getOperand(1), *D;
  Value *GuardSel = Y;
  if (auto *Fr = dyn_cast<FreezeInst>(Y))
    GuardSel = Fr->getOperand(0);
  // (d + 1 <u 2: the guard takes d = 0 too, whose results the selects must
  // then give; with d = -1 alone, division by 0 is undefined)
  bool GuardsZero;
  if (match(GuardSel, m_Select(m_SpecificICmp(ICmpInst::ICMP_ULT,
                                              m_Add(m_Value(D), m_One()),
                                              m_SpecificInt(2)),
                               m_One(), m_Deferred(D))))
    GuardsZero = true;
  else if (match(GuardSel,
                 m_Select(m_SpecificICmp(ICmpInst::ICMP_EQ, m_Value(D),
                                         m_AllOnes()),
                          m_One(), m_Deferred(D))))
    GuardsZero = false;
  else
    return std::nullopt;
  Value *X = BO.getOperand(0);
  auto IsX = [&](Value *V) {
    auto *Fr = dyn_cast<FreezeInst>(X);
    return V == X || (Fr && V == Fr->getOperand(0));
  };
  // 1: selects d = -1's value, 2: d = 0's (the expansion by d gives
  // quotient -x and all ones, remainder 0 and x), 0: neither
  auto Kind = [&](SelectInst *S, bool Rem) {
    Value *C = S->getCondition(), *T = S->getTrueValue(), *N;
    if (match(C, m_SpecificICmp(ICmpInst::ICMP_EQ, m_Specific(D),
                                m_AllOnes())))
      return (Rem ? match(T, m_Zero())
                  : match(T, m_Neg(m_Value(N))) && IsX(N))
                 ? 1
                 : 0;
    if (match(C, m_SpecificICmp(ICmpInst::ICMP_EQ, m_Specific(D), m_Zero())))
      return (Rem ? IsX(T) : match(T, m_AllOnes())) ? 2 : 0;
    return 0;
  };
  Guard16 G;
  G.D = D;
  // the chain of selects of such values from V, each the false value of the
  // next and used once but the last: its last is a root, the value of the
  // expansion's quotient or remainder if the chain covers what the guard
  // changed (quotient: d = -1; and d = 0 if the guard takes it)
  auto Chain = [&](Instruction *V, bool Rem) {
    bool MinusOne = false, Zero = false;
    Instruction *Start = V;
    while (V->hasOneUse()) {
      auto *S = dyn_cast<SelectInst>(V->user_back());
      int K = S && S->getFalseValue() == V ? Kind(S, Rem) : 0;
      if (!K)
        break;
      (K == 1 ? MinusOne : Zero) = true;
      if (V != Start)
        G.Dead.push_back(V);
      V = S;
    }
    if (V == Start || (!Rem && !MinusOne) || (GuardsZero && !Zero))
      return false;
    G.Roots.push_back({V, Rem});
    return true;
  };
  for (User *U : BO.users()) {
    auto *I = dyn_cast<Instruction>(U);
    if (!I)
      return std::nullopt;
    // x - q d' (DivRemPairs): mul(q, d') used once by sub(x, mul)
    Value *M;
    if (!IsRem && match(I, m_c_Mul(m_Specific(&BO), m_Specific(Y))) &&
        I->hasOneUse() &&
        match(I->user_back(), m_Sub(m_Value(M), m_Specific(I))) && IsX(M)) {
      G.Dead.push_back(I);
      G.Dead.push_back(cast<Instruction>(I->user_back()));
      if (!Chain(I->user_back(), true))
        return std::nullopt;
      continue;
    }
    // a chain from the division, through this select
    auto *S = dyn_cast<SelectInst>(I);
    if (!S || S->getFalseValue() != &BO || !Kind(S, IsRem))
      return std::nullopt;
    bool MinusOne = Kind(S, IsRem) == 1, Zero = !MinusOne;
    Instruction *V = S;
    while (V->hasOneUse()) {
      auto *S2 = dyn_cast<SelectInst>(V->user_back());
      int K = S2 && S2->getFalseValue() == V ? Kind(S2, IsRem) : 0;
      if (!K)
        break;
      (K == 1 ? MinusOne : Zero) = true;
      G.Dead.push_back(V);
      V = S2;
    }
    if ((!IsRem && !MinusOne) || (GuardsZero && !Zero))
      return std::nullopt;
    G.Roots.push_back({V, IsRem});
  }
  if (G.Roots.empty())
    return std::nullopt;
  return G;
}

/// Division and remainder of at most 16 bits (8- and 16-bit PTX div / rem):
/// the operands (magnitudes at most 2^16) are exact floats, and
/// q = trunc(x rcp(y) (1 + 2^-18)) is the quotient rounded toward zero, signs
/// included, with no correction: the factor makes up for rcp's error (within
/// 0.75 ulp, far below 2^-19) and the products' roundings, so the biased value
/// is never below |x / y| in magnitude, and it exceeds it by less than
/// |x / y| 2^-17, which is below 1 / |y| (|x| < 2^16): it never reaches the
/// next integer up from a quotient that is not one (at most 1 - 1 / |y| above
/// an integer). The remainder is x - q y. Division by zero gives all ones
/// (remainder: x), as the 32-bit expansion does; the signed overflow
/// -2^(n-1) / -1 wraps.
static Value *expandDivRem16(BinaryOperator &BO) {
  IRBuilder<> B(&BO);
  Instruction::BinaryOps Op = BO.getOpcode();
  bool IsSigned = Op == Instruction::SDiv || Op == Instruction::SRem;
  bool IsRem = Op == Instruction::URem || Op == Instruction::SRem;
  Type *Ty = BO.getType();
  Type *I32 = B.getInt32Ty(), *F32 = B.getFloatTy();
  std::optional<Guard16> G;
  if (IsSigned)
    G = matchGuard16(BO);
  Value *X0 = BO.getOperand(0);
  Value *Y0 = G ? G->D : BO.getOperand(1);
  if (G && isa<FreezeInst>(BO.getOperand(1)))
    Y0 = B.CreateFreeze(Y0);
  auto ToF = [&](Value *V) {
    return IsSigned ? B.CreateSIToFP(V, F32) : B.CreateUIToFP(V, F32);
  };
  // (converted from the narrow values: the 8- and 16-bit conversion modes
  // extend them themselves)
  Value *Rcp = B.CreateUnaryIntrinsic(Intrinsic::agx_rcp, ToF(Y0));
  Rcp = B.CreateFMul(Rcp, ConstantFP::get(F32, 1.0 + 0x1p-18));
  Value *P = B.CreateFMul(ToF(X0), Rcp);
  // (no select where d is not zero: known, or guarded as d + 1 <u 2 ? 1 : d)
  using namespace PatternMatch;
  Value *YG = Y0, *DG;
  if (auto *Fr = dyn_cast<FreezeInst>(YG))
    YG = Fr->getOperand(0);
  bool NonZero =
      isKnownNonZero(Y0, SimplifyQuery(BO.getDataLayout())) ||
      match(YG, m_Select(m_SpecificICmp(ICmpInst::ICMP_ULT,
                                        m_Add(m_Value(DG), m_One()),
                                        m_SpecificInt(2)),
                         m_One(), m_Deferred(DG)));
  Value *Q = IsSigned ? B.CreateFPToSI(P, I32) : B.CreateFPToUI(P, I32);
  Value *Q16 = B.CreateTrunc(Q, Ty);
  Value *Quot = Q16;
  if (!NonZero)
    Quot = B.CreateSelect(B.CreateICmpEQ(Y0, ConstantInt::get(Ty, 0)),
                          ConstantInt::getAllOnesValue(Ty), Q16);
  // (the remainder in the narrow type, only its low bits being kept; by
  // zero, x - q 0 = x needs no select, but the conversion of an infinity or
  // NaN is poison, so frozen then)
  Value *QR = NonZero ? Q16 : nullptr;
  auto Rem = [&] {
    if (!QR)
      QR = B.CreateFreeze(Q16);
    return B.CreateSub(X0, B.CreateMul(QR, Y0));
  };
  Value *Res = IsRem ? Rem() : Quot;
  if (G) {
    Value *R = nullptr;
    for (auto [Root, RootRem] : G->Roots) {
      Value *V = RootRem == IsRem ? Res : (R ? R : (R = Rem()));
      Root->replaceAllUsesWith(V);
    }
    for (auto [Root, RootRem] : G->Roots)
      Root->eraseFromParent();
    for (Instruction *I : reverse(G->Dead))
      if (I != &BO && I->use_empty())
        I->eraseFromParent();
  }
  return Res;
}

static cl::opt<bool> AGXNarrowDivisions(
    "agx-narrow-divisions", cl::init(true), cl::Hidden,
    cl::desc("8- and 16-bit division and remainder from the float reciprocal "
             "with one integer correction (otherwise LLVM's bit loop)"));

static cl::opt<bool> AGXLaunchDivisors(
    "agx-launch-divisors", cl::init(false), cl::Hidden,
    cl::desc("Divide by a 32-bit kernel argument with a multiplier and "
             "shifts the launch computes (.agx.kernels extension 7)"));

/// At most this many divisors per kernel take their multipliers from the
/// launch (three uniforms each).
static constexpr unsigned MaxLaunchDivisors = 8;

/// The kernel argument and 32-bit word of it that divisor Y is, if any: an
/// i32 argument, or the low or high word of an i64 one or the low word of an
/// extended i32 one (as the 32-bit fast path of a 64-bit division takes
/// them).
static std::optional<std::pair<unsigned, unsigned>>
launchDivisorSource(Value *Y) {
  using namespace PatternMatch;
  auto Unfrozen = [](Value *V) {
    auto *Fr = dyn_cast<FreezeInst>(V);
    return Fr ? Fr->getOperand(0) : V;
  };
  Y = Unfrozen(Y);
  // umax(d, 1), a front end's guard of a PTX division's divisor: the launch
  // computes the values of max(d, 1) (word 2 and 3 stand for words 0 and 1
  // so guarded)
  Value *Guarded;
  if (match(Y, m_UMax(m_Value(Guarded), m_One()))) {
    auto Src = launchDivisorSource(Guarded);
    if (!Src || Src->second >= 2)
      return std::nullopt;
    return std::pair(Src->first, Src->second + 2);
  }
  if (auto *A = dyn_cast<Argument>(Y))
    return A->getType()->isIntegerTy(32)
               ? std::optional(std::pair(A->getArgNo(), 0u))
               : std::nullopt;
  auto *T = dyn_cast<TruncInst>(Y);
  if (!T || !T->getType()->isIntegerTy(32))
    return std::nullopt;
  Value *V = Unfrozen(T->getOperand(0));
  unsigned Word = 0;
  const APInt *Sh;
  if (match(V, m_LShr(m_Value(V), m_APInt(Sh)))) {
    if (*Sh != 32)
      return std::nullopt;
    V = Unfrozen(V);
    Word = 1;
  }
  if (auto *A = dyn_cast<Argument>(V); A && A->getType()->isIntegerTy(64))
    return std::pair(A->getArgNo(), Word);
  // the low word of an extended i32 argument (an index built from 32-bit
  // arguments) is the argument
  Value *Narrow;
  if (Word == 0 && match(V, m_ZExtOrSExt(m_Value(Narrow))))
    if (auto *A = dyn_cast<Argument>(Narrow); A && A->getType()->isIntegerTy(32))
      return std::pair(A->getArgNo(), 0u);
  return std::nullopt;
}

/// A 32-bit division or remainder by a kernel argument, with the multiplier
/// m and shifts s1, s2 of its divisor that the launch computes
/// (llvm.agx.launch.value): t = mulhi(x, m), q = (t + ((x - t) >> s1)) >> s2,
/// exact for every 32-bit x (Granlund and Montgomery, "Division by Invariant
/// Integers using Multiplication", PLDI 1994, section 4). A zero divisor has
/// m = s1 = s2 = 0: q = x, so the remainder is x; the quotient is all ones,
/// as in expandDivRem32.
static Value *expandLaunchDivRem32(BinaryOperator &BO, unsigned Arg,
                                   unsigned Word) {
  IRBuilder<> B(&BO);
  Instruction::BinaryOps Op = BO.getOpcode();
  bool IsSigned = Op == Instruction::SDiv || Op == Instruction::SRem;
  bool IsRem = Op == Instruction::URem || Op == Instruction::SRem;
  Type *I32 = B.getInt32Ty();
  Function *LV = Intrinsic::getOrInsertDeclaration(
      BO.getModule(), Intrinsic::agx_launch_value);
  // words 2 and 3: max(d, 1) of word 0 or 1 (kinds 6 to 8)
  bool Guarded = Word >= 2;
  auto Launch = [&](unsigned Kind) {
    return B.CreateCall(
        LV, {B.getInt32(Arg), B.getInt32(Word & 1),
             B.getInt32(Kind + (Guarded ? 6 : IsSigned ? 3 : 0))});
  };
  Value *X = BO.getOperand(0), *Y = BO.getOperand(1);
  Value *Neg = nullptr;
  if (IsSigned) {
    Value *XNeg = B.CreateICmpSLT(X, B.getInt32(0));
    Value *YNeg = B.CreateICmpSLT(Y, B.getInt32(0));
    Neg = IsRem ? XNeg : B.CreateXor(XNeg, YNeg);
    X = B.CreateSelect(XNeg, B.CreateNeg(X), X);
    Y = B.CreateSelect(YNeg, B.CreateNeg(Y), Y);
  }
  Value *P = B.CreateMul(B.CreateZExt(X, B.getInt64Ty()),
                         B.CreateZExt(Launch(0), B.getInt64Ty()));
  Value *T = B.CreateTrunc(B.CreateLShr(P, 32), I32);
  Value *Q = B.CreateLShr(
      B.CreateAdd(T, B.CreateLShr(B.CreateSub(X, T), Launch(1))), Launch(2));
  Value *Res = IsRem ? B.CreateSub(X, B.CreateMul(Q, Y)) : Q;
  if (IsSigned)
    Res = B.CreateSelect(Neg, B.CreateNeg(Res), Res);
  if (IsRem || Guarded)
    return Res;
  Value *Zero = B.CreateICmpEQ(BO.getOperand(1), B.getInt32(0));
  return B.CreateSelect(Zero, B.getInt32(-1), Res);
}

static cl::opt<bool> AGXDiv64FitsLikely(
    "agx-div64-fits-likely", cl::init(true), cl::Hidden,
    cl::desc("Mark the outermost test that a 64-bit division's operands fit "
             "in 32 bits as likely"));

static cl::opt<bool> AGXNewtonDiv64(
    "agx-newton-div64", cl::init(true), cl::Hidden,
    cl::desc("Expand 64-bit division and remainder with a reciprocal estimate "
             "and Newton-Raphson steps in integers instead of a bit loop"));

// Each expansion is ~200 bytes of code more than the loop, in every kernel
// that divides, whether or not its 64-bit case runs; a kernel that divides by
// many 64-bit values (PyTorch's strided copies: their 32-bit case is the one
// that runs) keeps the compact loop rather than grow past the instruction
// fetch's reach (AGX Performance Simulator: permute4d, 6 divisions, 13.9 ->
// 16.4 KB, +24% cycles per SIMD group with 24 groups per core).
static cl::opt<unsigned> AGXNewtonDiv64Max(
    "agx-newton-div64-max", cl::init(4), cl::Hidden,
    cl::desc("Most 64-bit division expansions per function with "
             "-agx-newton-div64 (the others take the bit loop)"));

/// A 64-bit division or remainder without the 64-iteration bit loop, as
/// AMDGPU expands them (AMDGPUTargetLowering::LowerUDIVREM64): an estimate
/// of 2^64 / y from the f32 reciprocal, two Newton-Raphson steps in
/// integers, the quotient as the high half of x times it, and corrections.
/// The estimate is kept below 2^64 / y (the f32 value of y and the hardware
/// reciprocal are within a few units of 2^-24, the estimate is scaled by
/// 1 - 2^-19), each step stays below it (x (2 - y x) <= 1 / y, truncated),
/// so the quotient is at most x / y, and at most two below it after the
/// steps: corrections that add one while the remainder is at least y give
/// the exact result.
/// Division by zero gives all ones (remainder: x), as the 32-bit expansion.
/// Returns the quotient and the remainder (both signed for a signed
/// operation), so that a division and remainder of the same operands share
/// one expansion.
static std::pair<Value *, Value *> expandDivRem64(BinaryOperator &BO) {
  IRBuilder<> B(&BO);
  Instruction::BinaryOps Op = BO.getOpcode();
  bool IsSigned = Op == Instruction::SDiv || Op == Instruction::SRem;
  Type *I32 = B.getInt32Ty(), *I64 = B.getInt64Ty(), *I128 = B.getInt128Ty();
  Type *F32 = B.getFloatTy();
  Value *X = BO.getOperand(0), *Y0 = BO.getOperand(1);
  Value *Y = Y0, *XNeg = nullptr, *YNeg = nullptr;
  if (IsSigned) {
    XNeg = B.CreateICmpSLT(X, B.getInt64(0));
    YNeg = B.CreateICmpSLT(Y, B.getInt64(0));
    X = B.CreateSelect(XNeg, B.CreateNeg(X), X);
    Y = B.CreateSelect(YNeg, B.CreateNeg(Y), Y);
  }
  Value *Zero = B.CreateICmpEQ(Y, B.getInt64(0));
  Y = B.CreateSelect(Zero, B.getInt64(1), Y);
  auto MulHi = [&](Value *A, Value *C) {
    Value *P = B.CreateMul(B.CreateZExt(A, I128), B.CreateZExt(C, I128));
    return B.CreateTrunc(B.CreateLShr(P, 64), I64);
  };
  Value *Z;
  unsigned Corrections = 2;
  if (auto *C = dyn_cast<ConstantInt>(Y); C && !C->isZero() && !C->isOne()) {
    // a constant divisor: floor(2^64 / y) exactly, so the quotient is at
    // most one below x / y (x / y - x z / 2^64 < x / 2^64 < 1, and the
    // truncation of the multiply-high)
    Z = B.getInt64(APInt::getMaxValue(64).udiv(C->getValue()).getZExtValue());
  } else {
    // y as an f32 (from its halves: an i64 conversion is not native), its
    // reciprocal scaled to 2^64 (1 - 2^-19), split into halves for the
    // conversion back (x - hi 2^32 is exact)
    Value *YLo = B.CreateUIToFP(B.CreateTrunc(Y, I32), F32);
    Value *YHi = B.CreateUIToFP(B.CreateTrunc(B.CreateLShr(Y, 32), I32), F32);
    Value *FY = B.CreateIntrinsic(Intrinsic::fma, {F32},
                                  {YHi, ConstantFP::get(F32, 4294967296.0), YLo});
    Value *Rcp = B.CreateUnaryIntrinsic(Intrinsic::agx_rcp, FY);
    Value *E = B.CreateFMul(Rcp, ConstantFP::get(F32, 0x1.ffffcp63));
    Value *EHi = B.CreateUnaryIntrinsic(
        Intrinsic::trunc, B.CreateFMul(E, ConstantFP::get(F32, 0x1p-32)));
    Value *ELo = B.CreateIntrinsic(Intrinsic::fma, {F32},
                                   {EHi, ConstantFP::get(F32, -4294967296.0), E});
    Z = B.CreateOr(
        B.CreateShl(B.CreateZExt(B.CreateFPToUI(EHi, I32), I64), 32),
        B.CreateZExt(B.CreateFPToUI(ELo, I32), I64));
    Value *NegY = B.CreateNeg(Y);
    for (int I = 0; I < 2; ++I)
      Z = B.CreateAdd(Z, MulHi(Z, B.CreateMul(NegY, Z)));
  }
  Value *Q = MulHi(X, Z);
  Value *R = B.CreateSub(X, B.CreateMul(Q, Y));
  // Corrections without a loop (a loop's trip count is what loop strength
  // reduction would rewrite back into a division): the quotient is at most
  // two below x / y (after the steps the estimate is within about two units
  // of 2^64 / y, so x z / 2^64 is within three of x / y; an exact model of
  // the f32 steps with the reciprocal two units off, over 15,000 random and
  // edge pairs, never needs a third). Each correction is code in every
  // kernel that divides, whether or not the 64-bit case runs.
  for (unsigned I = 0; I < Corrections; ++I) {
    Value *C = B.CreateICmpUGE(R, Y);
    Q = B.CreateSelect(C, B.CreateAdd(Q, B.getInt64(1)), Q);
    R = B.CreateSelect(C, B.CreateSub(R, Y), R);
  }
  if (IsSigned) {
    Q = B.CreateSelect(B.CreateXor(XNeg, YNeg), B.CreateNeg(Q), Q);
    R = B.CreateSelect(XNeg, B.CreateNeg(R), R);
  }
  return {B.CreateSelect(Zero, B.getInt64(-1), Q),
          B.CreateSelect(Zero, BO.getOperand(0), R)};
}

bool AGXCodeGenPrepare::expandDivisions(Function &F) {
  // 64-bit divisions whose operands fit in 32 bits (checked at run time, or
  // known) take the 32-bit expansion. Division and remainder of the same
  // operands share one.
  bool Bypassed = false;
  SmallVector<BasicBlock *, 16> Blocks;
  for (BasicBlock &BB : F)
    Blocks.push_back(&BB);
  DenseMap<unsigned, unsigned> BypassWidth = {{64, 32}};
  for (BasicBlock *BB : Blocks)
    Bypassed |= bypassSlowDivision(BB, BypassWidth);

  SmallVector<BinaryOperator *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || !BO->getType()->isIntegerTy())
      continue;
    switch (BO->getOpcode()) {
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      if (!isCheapDivision(*BO) &&
          BO->getType()->getIntegerBitWidth() <= 64)
        Worklist.push_back(BO);
      break;
    default:
      break;
    }
  }

  // The 32-bit expansions first: they keep the CFG (and the dominator tree
  // of the divisor cache) as it is; the loops of the wider ones do not.
  std::optional<DivisorCache> Divisors;
  SmallVector<BinaryOperator *, 8> Wide;
  // Divisors whose multipliers the launch computes, while the uniforms have
  // room for them (AGXPromoteKernelArgs fills at most 1024 bytes less 20).
  SmallVector<std::pair<unsigned, unsigned>, 4> LaunchDivisors;
  bool Kernel = AGXLaunchDivisors &&
                F.getCallingConv() == CallingConv::AGX_KERNEL;
  if (Kernel) {
    uint64_t Bytes = 20 + 8;
    for (const Argument &A : F.args())
      Bytes += alignTo(F.getDataLayout().getTypeStoreSize(A.getType()), 4);
    Kernel = Bytes + MaxLaunchDivisors * 2 * 12 <= 1024;
  }
  for (BinaryOperator *BO : Worklist) {
    if (BO->getType()->getIntegerBitWidth() <= 16 && AGXNarrowDivisions) {
      // by a constant: the 32-bit division of the extended operands, which
      // instruction selection makes a multiply-high
      if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
          C && !C->isZero()) {
        IRBuilder<> B(BO);
        bool Signed = BO->getOpcode() == Instruction::SDiv ||
                      BO->getOpcode() == Instruction::SRem;
        auto Ext = [&](Value *V) {
          return Signed ? B.CreateSExt(V, B.getInt32Ty())
                        : B.CreateZExt(V, B.getInt32Ty());
        };
        Value *R =
            B.CreateTrunc(B.CreateBinOp(BO->getOpcode(), Ext(BO->getOperand(0)),
                                        Ext(BO->getOperand(1))),
                          BO->getType());
        R->takeName(BO);
        BO->replaceAllUsesWith(R);
        BO->eraseFromParent();
        continue;
      }
      Value *R = expandDivRem16(*BO);
      R->takeName(BO);
      BO->replaceAllUsesWith(R);
      BO->eraseFromParent();
      continue;
    }
    if (!BO->getType()->isIntegerTy(32)) {
      Wide.push_back(BO);
      continue;
    }
    auto Src = Kernel ? launchDivisorSource(BO->getOperand(1)) : std::nullopt;
    // max(d, 1) has unsigned values only
    if (Src && Src->second >= 2 && (BO->getOpcode() == Instruction::SDiv ||
                                     BO->getOpcode() == Instruction::SRem))
      Src = std::nullopt;
    if (Src) {
      bool Known = is_contained(LaunchDivisors, *Src);
      if (Known || LaunchDivisors.size() < MaxLaunchDivisors) {
        if (!Known)
          LaunchDivisors.push_back(*Src);
        Value *R = expandLaunchDivRem32(*BO, Src->first, Src->second);
        R->takeName(BO);
        BO->replaceAllUsesWith(R);
        BO->eraseFromParent();
        continue;
      }
    }
    if (!Divisors)
      Divisors.emplace(F);
    Value *R = expandDivRem32(*BO, *Divisors);
    R->takeName(BO);
    BO->replaceAllUsesWith(R);
    BO->eraseFromParent();
  }
  std::map<std::tuple<BasicBlock *, Value *, Value *, bool>,
           std::pair<Value *, Value *>>
      Expanded64;
  // the expansions a function would have (a division and a remainder of the
  // same operands in one block share one)
  std::set<std::tuple<BasicBlock *, Value *, Value *, bool>> Sites;
  for (BinaryOperator *BO : Wide)
    if (BO->getType()->isIntegerTy(64))
      Sites.insert({BO->getParent(), BO->getOperand(0), BO->getOperand(1),
                    BO->getOpcode() == Instruction::SDiv ||
                        BO->getOpcode() == Instruction::SRem});
  bool Newton = AGXNewtonDiv64 && Sites.size() <= AGXNewtonDiv64Max;
  // The long case of a 64-bit division under a test that its operands fit
  // in 32 bits (a front end's (a | b) < 2^32, or the bypass's (a | b) &
  // ~0xffffffff == 0) is the rare one: the outermost such test on the
  // division's chain of single predecessors is marked likely to fit, so that
  // AGXColdRegions moves the long case out of line (a skip not taken). Where
  // the operands rarely fit, one more taken jump is little beside the long
  // division. (Not an inner test: under an outer one that failed, the inner
  // one's fast case is not the common one.)
  if (AGXDiv64FitsLikely)
    for (BinaryOperator *BO : Wide) {
      if (!BO->getType()->isIntegerTy(64))
        continue;
      using namespace PatternMatch;
      auto FitsTest = [](Value *C) {
        return match(C, m_SpecificICmp(ICmpInst::ICMP_ULT, m_Or(m_Value(), m_Value()),
                                       m_SpecificInt(uint64_t(1) << 32))) ||
               match(C, m_SpecificICmp(
                            ICmpInst::ICMP_EQ,
                            m_And(m_Value(), m_SpecificInt(0xffffffff00000000ULL)),
                            m_Zero()));
      };
      CondBrInst *Outer = nullptr;
      bool OuterFitsSide = false;
      BasicBlock *Cur = BO->getParent();
      for (unsigned Depth = 0; Depth < 4; ++Depth) {
        BasicBlock *Pred = Cur->getSinglePredecessor();
        if (!Pred)
          break;
        if (auto *Br = dyn_cast<CondBrInst>(Pred->getTerminator());
            Br && FitsTest(Br->getCondition()) &&
            Br->getSuccessor(0) != Br->getSuccessor(1)) {
          Outer = Br;
          OuterFitsSide = Br->getSuccessor(0) == Cur;
        }
        Cur = Pred;
      }
      // the division on the side where the operands do not fit
      if (Outer && !OuterFitsSide && !Outer->hasMetadata(LLVMContext::MD_prof))
        Outer->setMetadata(LLVMContext::MD_prof,
                           MDBuilder(BO->getContext()).createLikelyBranchWeights());
    }
  for (BinaryOperator *BO : Wide) {
    bool IsRem = BO->getOpcode() == Instruction::URem ||
                 BO->getOpcode() == Instruction::SRem;
    bool Is64 = BO->getType()->getIntegerBitWidth() > 32;
    if (Newton && BO->getType()->isIntegerTy(64)) {
      // The 64-bit case of a bypassed division (operands that do not fit in
      // 32 bits) is the rare one: unlikely, so that block placement moves it
      // out of the fast path's way.
      if (BasicBlock *Pred = BO->getParent()->getSinglePredecessor())
        if (auto *Br = dyn_cast<CondBrInst>(Pred->getTerminator());
            Br && !Br->hasMetadata(LLVMContext::MD_prof)) {
          MDBuilder MDB(BO->getContext());
          Br->setMetadata(LLVMContext::MD_prof,
                          Br->getSuccessor(0) == BO->getParent()
                              ? MDB.createUnlikelyBranchWeights()
                              : MDB.createLikelyBranchWeights());
        }
      // a division or remainder of the same operands earlier in the block
      // (bypassSlowDivision puts them together) reuses its expansion
      bool Signed = BO->getOpcode() == Instruction::SDiv ||
                    BO->getOpcode() == Instruction::SRem;
      auto Key = std::make_tuple(BO->getParent(), BO->getOperand(0),
                                 BO->getOperand(1), Signed);
      auto It = Expanded64.find(Key);
      std::pair<Value *, Value *> QR =
          It != Expanded64.end() ? It->second : expandDivRem64(*BO);
      Expanded64[Key] = QR;
      Value *R = IsRem ? QR.second : QR.first;
      R->takeName(BO);
      BO->replaceAllUsesWith(R);
      BO->eraseFromParent();
      continue;
    }
    if (IsRem)
      Is64 ? expandRemainderUpTo64Bits(BO) : expandRemainderUpTo32Bits(BO);
    else
      Is64 ? expandDivisionUpTo64Bits(BO) : expandDivisionUpTo32Bits(BO);
  }
  return Bypassed || !Worklist.empty();
}

bool AGXCodeGenPrepare::expandMemIntrinsics(Function &F,
                                            const TargetTransformInfo &TTI) {
  // Instruction selection expands memory intrinsics of a known, small size
  // into loads and stores; everything else becomes a loop.
  constexpr uint64_t MaxInlineSize = 128;
  SmallVector<MemIntrinsic *, 4> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *MI = dyn_cast<MemIntrinsic>(&I);
    if (!MI)
      continue;
    auto *Len = dyn_cast<ConstantInt>(MI->getLength());
    if (Len && Len->getZExtValue() <= MaxInlineSize)
      continue;
    Worklist.push_back(MI);
  }

  for (MemIntrinsic *MI : Worklist) {
    if (auto *Cpy = dyn_cast<MemCpyInst>(MI)) {
      expandMemCpyAsLoop(Cpy, TTI);
    } else if (auto *Move = dyn_cast<MemMoveInst>(MI)) {
      if (!expandMemMoveAsLoop(Move, TTI))
        continue;
    } else if (auto *Set = dyn_cast<MemSetInst>(MI)) {
      expandMemSetAsLoop(Set, TTI);
    } else {
      continue;
    }
    MI->eraseFromParent();
  }
  return !Worklist.empty();
}

bool AGXCodeGenPrepare::lowerConstrainedFP(Function &F) {
  SmallVector<ConstrainedFPIntrinsic *, 8> Worklist;
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<ConstrainedFPIntrinsic>(&I))
      Worklist.push_back(CI);

  for (ConstrainedFPIntrinsic *CI : Worklist) {
    IRBuilder<> B(CI);
    SmallVector<Value *, 4> Args(CI->args().begin(),
                                 CI->args().begin() +
                                     CI->getNonMetadataArgCount());
    Value *Res = nullptr;
    switch (CI->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:                                                   \
    if (Instruction::isBinaryOp(Instruction::NAME))                            \
      Res = B.CreateBinOp(Instruction::BinaryOps(Instruction::NAME), Args[0],  \
                          Args[1]);                                            \
    else                                                                       \
      Res = B.CreateCast(Instruction::CastOps(Instruction::NAME), Args[0],     \
                         CI->getType());                                       \
    break;
#define CMP_INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC, DAGN)              \
  case Intrinsic::INTRINSIC:                                                   \
    Res = B.CreateFCmp(cast<ConstrainedFPCmpIntrinsic>(CI)->getPredicate(),    \
                       Args[0], Args[1]);                                      \
    break;
#define FUNCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                           \
  case Intrinsic::INTRINSIC: {                                                 \
    SmallVector<Type *, 4> ArgTys;                                             \
    for (Value *A : Args)                                                      \
      ArgTys.push_back(A->getType());                                          \
    SmallVector<Type *, 2> OverloadTys;                                        \
    FunctionType *FT = FunctionType::get(CI->getType(), ArgTys, false);        \
    if (Intrinsic::isSignatureValid(Intrinsic::NAME, FT, OverloadTys))         \
      Res = B.CreateCall(Intrinsic::getOrInsertDeclaration(                    \
                             F.getParent(), Intrinsic::NAME, OverloadTys),     \
                         Args);                                                \
    break;                                                                     \
  }
#include "llvm/IR/ConstrainedOps.def"
    default:
      break;
    }
    if (!Res)
      continue;
    Res->takeName(CI);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
  }
  return !Worklist.empty();
}

/// Replace the returns, and unreachable terminators, by branches to a single
/// return block.
bool AGXCodeGenPrepare::unifyExits(Function &F) {
  SmallVector<Instruction *, 4> Exits;
  for (BasicBlock &BB : F) {
    Instruction *T = BB.getTerminator();
    if (isa<ReturnInst>(T) || isa<UnreachableInst>(T))
      Exits.push_back(T);
  }
  if (!F.getReturnType()->isVoidTy())
    return false;
  // A thread that reaches unreachable (after llvm.trap, as in __assertfail)
  // returns: an executed trap does nothing on G14C (ptx-re-blockers B21), and
  // the code after the block would otherwise run on.
  if (Exits.size() == 1) {
    if (!isa<UnreachableInst>(Exits.front()))
      return false;
    ReturnInst::Create(F.getContext(), nullptr, Exits.front()->getIterator());
    Exits.front()->eraseFromParent();
    return true;
  }
  if (Exits.empty())
    return false;

  BasicBlock *Ret = BasicBlock::Create(F.getContext(), "unified.return", &F);
  ReturnInst::Create(F.getContext(), Ret);
  for (Instruction *T : Exits) {
    UncondBrInst::Create(Ret, T->getIterator());
    T->eraseFromParent();
  }
  return true;
}

static cl::opt<bool> AGXFoldLaneIdentities(
    "agx-fold-lane-identities", cl::init(true), cl::Hidden,
    cl::desc("Fold shuffles that undo each other (see foldLaneIdentities)"));
// For hardware experiments only (wrong code for most divergent indices).
static cl::opt<bool> AGXUnsplitShuffles(
    "agx-unsplit-shuffles", cl::init(false), cl::Hidden,
    cl::desc("Experiments: never split shuffles of divergent indices"));
static cl::opt<bool> AGXQuadUniformShuffles(
    "agx-quad-uniform-shuffles", cl::init(true), cl::Hidden,
    cl::desc("Do not split shuffles of the thread index whose source quads "
             "hold the positions they are read at"));

namespace {
/// The value of integer expressions of the thread index and constants in a
/// given thread of the SIMD-group (see foldLaneIdentities).
class LaneEvaluator {
  DenseMap<std::pair<const Value *, unsigned>, std::optional<uint64_t>> Memo;

public:
  std::optional<uint64_t> eval(const Value *V, unsigned Lane,
                               unsigned Depth = 0) {
    auto Key = std::make_pair(V, Lane);
    if (auto It = Memo.find(Key); It != Memo.end())
      return It->second;
    std::optional<uint64_t> R = compute(V, Lane, Depth);
    Memo[Key] = R;
    return R;
  }

private:
  static uint64_t mask(unsigned Bits) {
    return Bits >= 64 ? ~0ULL : (1ULL << Bits) - 1;
  }
  static int64_t sext(uint64_t X, unsigned Bits) {
    return Bits >= 64 ? (int64_t)X : (int64_t)(X << (64 - Bits)) >> (64 - Bits);
  }
  std::optional<uint64_t> compute(const Value *V, unsigned Lane,
                                  unsigned Depth) {
    Type *Ty = V->getType();
    if (!Ty->isIntegerTy() || Ty->getIntegerBitWidth() > 64 || Depth > 24)
      return std::nullopt;
    unsigned Bits = Ty->getIntegerBitWidth();
    uint64_t M = mask(Bits);
    if (auto *C = dyn_cast<ConstantInt>(V))
      return C->getZExtValue() & M;
    if (auto *II = dyn_cast<IntrinsicInst>(V)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::agx_thread_index_in_simdgroup:
        return Lane & M;
      case Intrinsic::bitreverse:
        if (auto A = eval(II->getArgOperand(0), Lane, Depth + 1))
          return APInt(Bits, *A).reverseBits().getZExtValue();
        return std::nullopt;
      case Intrinsic::bswap:
        if (auto A = eval(II->getArgOperand(0), Lane, Depth + 1))
          return APInt(Bits, *A).byteSwap().getZExtValue();
        return std::nullopt;
      default:
        return std::nullopt;
      }
    }
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return std::nullopt;
    auto Op = [&](unsigned N) { return eval(I->getOperand(N), Lane, Depth + 1); };
    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      auto A = Op(0), B = Op(1);
      if (!A || !B)
        return std::nullopt;
      uint64_t X = *A, Y = *B;
      switch (BO->getOpcode()) {
      case Instruction::Add: return (X + Y) & M;
      case Instruction::Sub: return (X - Y) & M;
      case Instruction::Mul: return (X * Y) & M;
      case Instruction::And: return X & Y;
      case Instruction::Or: return X | Y;
      case Instruction::Xor: return X ^ Y;
      case Instruction::Shl: if (Y >= Bits) return std::nullopt; return (X << Y) & M;
      case Instruction::LShr: if (Y >= Bits) return std::nullopt; return X >> Y;
      case Instruction::AShr: if (Y >= Bits) return std::nullopt; return (uint64_t)(sext(X, Bits) >> Y) & M;
      case Instruction::UDiv: if (!Y) return std::nullopt; return X / Y;
      case Instruction::URem: if (!Y) return std::nullopt; return X % Y;
      default: return std::nullopt;
      }
    }
    if (auto *Cmp = dyn_cast<ICmpInst>(I)) {
      auto A = Op(0), B = Op(1);
      if (!A || !B)
        return std::nullopt;
      unsigned W = Cmp->getOperand(0)->getType()->getIntegerBitWidth();
      int64_t SA = sext(*A, W), SB = sext(*B, W);
      switch (Cmp->getPredicate()) {
      case ICmpInst::ICMP_EQ: return *A == *B;
      case ICmpInst::ICMP_NE: return *A != *B;
      case ICmpInst::ICMP_UGT: return *A > *B;
      case ICmpInst::ICMP_UGE: return *A >= *B;
      case ICmpInst::ICMP_ULT: return *A < *B;
      case ICmpInst::ICMP_ULE: return *A <= *B;
      case ICmpInst::ICMP_SGT: return SA > SB;
      case ICmpInst::ICMP_SGE: return SA >= SB;
      case ICmpInst::ICMP_SLT: return SA < SB;
      case ICmpInst::ICMP_SLE: return SA <= SB;
      default: return std::nullopt;
      }
    }
    if (isa<SelectInst>(I)) {
      auto C = Op(0);
      if (!C)
        return std::nullopt;
      return Op(*C ? 1 : 2);
    }
    switch (I->getOpcode()) {
    case Instruction::ZExt:
      return Op(0);
    case Instruction::Trunc:
      if (auto A = Op(0))
        return *A & M;
      return std::nullopt;
    case Instruction::SExt:
      if (auto A = Op(0))
        return (uint64_t)sext(*A, I->getOperand(0)->getType()->getIntegerBitWidth()) & M;
      return std::nullopt;
    default:
      return std::nullopt;
    }
  }
};

/// Where the bits of element Elt (-1: a scalar) of a value in thread Lane
/// come from.
struct LaneSource {
  Value *V;
  unsigned Lane;
  int Elt;
};
} // namespace

/// Follows V (element Elt, in thread Lane) back through shuffles whose index
/// is a function of the thread index, selects whose condition is one,
/// bitcasts that keep the elements, and constant-index element inserts and
/// extracts, to the value it copies.
static LaneSource traceLane(Value *V, unsigned Lane, int Elt,
                            LaneEvaluator &LE) {
  for (unsigned Step = 0; Step < 96; ++Step) {
    if (auto *II = dyn_cast<IntrinsicInst>(V)) {
      Intrinsic::ID ID = II->getIntrinsicID();
      if (ID == Intrinsic::agx_simd_shuffle ||
          ID == Intrinsic::agx_simd_shuffle_xor ||
          ID == Intrinsic::agx_quad_shuffle ||
          ID == Intrinsic::agx_quad_shuffle_xor) {
        auto Idx = LE.eval(II->getArgOperand(1), Lane);
        bool Quad = ID == Intrinsic::agx_quad_shuffle ||
                    ID == Intrinsic::agx_quad_shuffle_xor;
        if (!Idx || *Idx >= (Quad ? 4u : 32u))
          break;
        unsigned Next;
        if (ID == Intrinsic::agx_simd_shuffle)
          Next = *Idx;
        else if (ID == Intrinsic::agx_simd_shuffle_xor)
          Next = Lane ^ *Idx;
        else if (ID == Intrinsic::agx_quad_shuffle)
          Next = (Lane & ~3u) | *Idx;
        else
          Next = Lane ^ *Idx;
        V = II->getArgOperand(0);
        Lane = Next;
        continue;
      }
      break;
    }
    if (auto *Sel = dyn_cast<SelectInst>(V)) {
      if (Sel->getCondition()->getType()->isVectorTy())
        break;
      auto C = LE.eval(Sel->getCondition(), Lane);
      if (!C)
        break;
      V = *C ? Sel->getTrueValue() : Sel->getFalseValue();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastInst>(V)) {
      Type *From = BC->getOperand(0)->getType(), *To = BC->getType();
      auto *FV = dyn_cast<FixedVectorType>(From);
      auto *TV = dyn_cast<FixedVectorType>(To);
      bool Same = (!FV && !TV && From->getPrimitiveSizeInBits() == To->getPrimitiveSizeInBits() &&
                   From->isSingleValueType()) ||
                  (FV && TV && FV->getNumElements() == TV->getNumElements() &&
                   FV->getScalarSizeInBits() == TV->getScalarSizeInBits());
      if (!Same || From->isPointerTy() || To->isPointerTy())
        break;
      V = BC->getOperand(0);
      continue;
    }
    if (auto *EE = dyn_cast<ExtractElementInst>(V)) {
      auto *CI = dyn_cast<ConstantInt>(EE->getIndexOperand());
      if (Elt != -1 || !CI || !isa<FixedVectorType>(EE->getVectorOperandType()))
        break;
      V = EE->getVectorOperand();
      Elt = CI->getZExtValue();
      continue;
    }
    if (auto *IE = dyn_cast<InsertElementInst>(V)) {
      auto *CI = dyn_cast<ConstantInt>(IE->getOperand(2));
      if (Elt == -1 || !CI)
        break;
      if (CI->getZExtValue() == (uint64_t)Elt) {
        V = IE->getOperand(1);
        Elt = -1;
      } else {
        V = IE->getOperand(0);
      }
      continue;
    }
    break;
  }
  return {V, Lane, Elt};
}

static bool isLaneMovement(const Value *V) {
  if (auto *II = dyn_cast<IntrinsicInst>(V))
    switch (II->getIntrinsicID()) {
    case Intrinsic::agx_simd_shuffle:
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_quad_shuffle:
    case Intrinsic::agx_quad_shuffle_xor:
      return true;
    default:
      return false;
    }
  return isa<SelectInst>(V) || isa<BitCastInst>(V) ||
         isa<ExtractElementInst>(V) || isa<InsertElementInst>(V);
}

/// Shuffles that undo each other: a value whose every thread (and element)
/// is, through shuffles, selects and element moves that depend only on the
/// thread index, the same thread's (and element's) bits of one earlier value
/// is that value. The PTX front end's mma emulation moves the elements of D from the
/// matrix unit's layout to the PTX fragment layout and the next mma's C back:
/// a chain of mmas keeps its accumulator in the unit's layout. Each candidate
/// is checked in all 32 threads.
static cl::opt<bool> AGXSIMDScans(
    "agx-simd-scans", cl::init(true), cl::Hidden,
    cl::desc("A SIMD-group's inclusive integer add scan by shuffles up 1, 2, "
             "4, 8 and 16 as one simd_prefix plus the value"));

/// Whether select(Cond, shuffled, 0) (ShuffleOnTrue; else the arms swapped)
/// keeps the shuffled value exactly in the lanes D and above.
static bool keepsLanesFrom(Value *Cond, bool ShuffleOnTrue, uint64_t D) {
  using namespace PatternMatch;
  CmpPredicate P;
  Value *L;
  const APInt *C;
  if (!match(Cond, m_ICmp(P, m_Value(L), m_APInt(C))) ||
      !match(L, m_Intrinsic<Intrinsic::agx_thread_index_in_simdgroup>()))
    return false;
  ICmpInst::Predicate Pr =
      ShuffleOnTrue ? ICmpInst::Predicate(P) : ICmpInst::getInversePredicate(P);
  uint64_t K = C->getZExtValue();
  switch (Pr) {
  case ICmpInst::ICMP_UGT:
    return K + 1 == D;
  case ICmpInst::ICMP_UGE:
    return K == D;
  case ICmpInst::ICMP_NE:
    return D == 1 && K == 0;
  default:
    return false;
  }
}

// Off by default: a simd reduction is 22.9 cycles of latency and of issue
// interval (instruction-bench), the butterfly about 34 and 13 to 19; with
// many SIMD-groups resident (agx-bench tput) the butterfly is as fast or
// faster in every measured form (L15-E364).
static cl::opt<bool> AGXSIMDReductions(
    "agx-simd-reductions", cl::init(false), cl::Hidden,
    cl::desc("A SIMD-group's xor-butterfly reduction (shuffles xor 1, 2, 4, "
             "8 and 16) as one simd reduction"));

/// The xor butterfly over a SIMD-group's 32 lanes, x' = x op shuffle_xor(x,
/// d) for d = 1, 2, 4, 8 and 16 in any order (quad shuffles for 1 and 2 as
/// well), leaves every lane the reduction of all 32: for a 32- or 16-bit
/// integer add, and, or, xor, or signed / unsigned min / max (exact, in any
/// order) it is the hardware's simd reduction of x, one instruction for ten
/// (L15-E362). A float add or multiply is the same where the butterfly's
/// order is the hardware's: lanes 2k and 2k+1 first, then neighbouring pairs,
/// i.e. d = 1, 2, 4, 8, 16 (in a butterfly a lane and its partner compute
/// a op b and b op a, the same value, so every lane holds that one pairwise
/// tree; the coverage recordings simd_fadd fail with the halves-first tree),
/// rounding each step like fadd / fmul and flushing fp32 denormals like
/// them; not with IEEE fp32 denormals in software, half denormal flushing or
/// canonical NaNs (L15-E363). The halves-first order, d = 16, 8, 4, 2, 1,
/// is that tree over the lanes in bit-reversed order (its level l joins the
/// lanes whose low 5 - l index bits agree, which are the neighbours-first
/// blocks of lanes bitreverse5(i)). The reversal swaps lane bits 0 and 4,
/// and 1 and 3: y = bit 0 != bit 4 ? shuffle_xor(x, 17) : x, then the same
/// with bits 1 and 3 and 10, and the reduction of y, five instructions for
/// ten (one shuffle at the reversed index would read all four threads of a
/// quad, which the quad index merge does not allow). A shuffle
/// from an inactive lane is undefined
/// (as in simdScans), so where some lanes are inactive the reduction of the
/// active ones is one of the original's results.
bool AGXCodeGenPrepare::simdReductions(Function &F, const AGXSubtarget &ST) {
  if (!AGXSIMDReductions || !ST.hasSIMDReduce())
    return false;
  bool Float32 = !ST.hasCanonicalNaN() &&
                 !(ST.hasIEEEFP32() && F.getDenormalMode(APFloat::IEEEsingle()) ==
                                           DenormalMode::getIEEE());
  bool Float16 = !ST.hasCanonicalNaN() &&
                 F.getDenormalMode(APFloat::IEEEhalf()) == DenormalMode::getIEEE();
  // The value a shuffle moves, through the bitcasts around an integer shuffle
  // of a float.
  auto Strip = [](Value *V) {
    if (auto *BC = dyn_cast<BitCastInst>(V))
      return BC->getOperand(0);
    return V;
  };
  // The step v' = op(v, shuffle_xor(v, d)): its op code, v, d and shuffle.
  auto Step = [&](Value *V, int &Op, Value *&Prev, unsigned &D,
                  Instruction *&Shuffle) {
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return false;
    Type *Ty = I->getType();
    bool Int = Ty->isIntegerTy(32) || Ty->isIntegerTy(16);
    if (!Int && !(Ty->isFloatTy() && Float32) && !(Ty->isHalfTy() && Float16))
      return false;
    Value *A, *B;
    int O = -1;
    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      A = BO->getOperand(0);
      B = BO->getOperand(1);
      switch (BO->getOpcode()) {
      case Instruction::Add: O = 16; break;
      case Instruction::And: O = 0; break;
      case Instruction::Or: O = 2; break;
      case Instruction::Xor: O = 4; break;
      case Instruction::FAdd: O = 1; break;
      case Instruction::FMul: O = 3; break;
      default: return false;
      }
    } else if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::smin: O = 20; break;
      case Intrinsic::smax: O = 22; break;
      case Intrinsic::umin: O = 28; break;
      case Intrinsic::umax: O = 30; break;
      default: return false;
      }
      A = II->getArgOperand(0);
      B = II->getArgOperand(1);
    } else {
      return false;
    }
    for (int K = 0; K < 2; ++K, std::swap(A, B)) {
      auto *Sh = dyn_cast<IntrinsicInst>(Strip(B));
      if (!Sh || (Sh->getIntrinsicID() != Intrinsic::agx_simd_shuffle_xor &&
                  Sh->getIntrinsicID() != Intrinsic::agx_quad_shuffle_xor))
        continue;
      Value *Src = Sh->getArgOperand(0);
      if (Src != A && Strip(Src) != A)
        continue;
      if (Src->getType()->getPrimitiveSizeInBits() !=
          Ty->getPrimitiveSizeInBits())
        continue;
      auto *C = dyn_cast<ConstantInt>(Sh->getArgOperand(1));
      if (!C)
        continue;
      uint64_t Dist = C->getZExtValue();
      bool Ok = (Sh->getIntrinsicID() == Intrinsic::agx_simd_shuffle_xor &&
                 (Dist == 1 || Dist == 2 || Dist == 4 || Dist == 8 ||
                  Dist == 16)) ||
                (Sh->getIntrinsicID() == Intrinsic::agx_quad_shuffle_xor &&
                 (Dist == 1 || Dist == 2));
      if (!Ok)
        continue;
      Op = O;
      Prev = A;
      D = Dist;
      Shuffle = Sh;
      return true;
    }
    return false;
  };
  // Value handles: a replaced chain's dead steps may be another chain's
  // top or base (a longer butterfly), which is then skipped.
  // The reduction goes after the last shuffle, which the active threads of
  // the butterfly execute (the last op may have been sunk into a branch,
  // e.g. under thread 0's atomic); all its shuffles are in that block.
  SmallVector<std::tuple<WeakVH, WeakVH, int, bool, WeakVH>, 4> Found;
  for (Instruction &I : instructions(F)) {
    int Op = -1, StepOp;
    unsigned Seen = 0, D;
    Value *V = &I, *Prev;
    Instruction *Bottom = nullptr, *Sh, *Top = nullptr;
    bool OneBlock = true;
    // From the top: the hardware's float order is d = 16, 8, 4, 2, 1 here,
    // the bit-reversed one d = 1, 2, 4, 8, 16.
    bool Ordered = true, Reversed = true;
    int N = 0;
    for (; N < 5 && Step(V, StepOp, Prev, D, Sh); ++N) {
      if ((Op != -1 && StepOp != Op) || (Seen & D))
        break;
      if (!Top)
        Top = Sh;
      OneBlock &= Sh->getParent() == Top->getParent();
      Ordered &= D == (16u >> N);
      Reversed &= D == (1u << N);
      Op = StepOp;
      Seen |= D;
      Bottom = cast<Instruction>(V);
      V = Prev;
    }
    if (N != 5 || Seen != 31 || !OneBlock)
      continue;
    bool Permute = false;
    if (I.getType()->isFloatingPointTy()) {
      if (!Ordered && !Reversed)
        continue;
      Permute = !Ordered;
      // A multiply the first add could fuse with: the old code may differ.
      if (Op == 1 && Bottom->hasAllowContract() && isa<FPMathOperator>(V) &&
          cast<Operator>(V)->getOpcode() == Instruction::FMul)
        continue;
    }
    Found.push_back({&I, V, Op, Permute, Top});
  }
  bool Changed = false;
  Value *Swap04 = nullptr, *Swap13 = nullptr;
  for (auto &[IH, XH, Op, Permute, TopH] : Found) {
    auto *I = dyn_cast_or_null<Instruction>(IH);
    auto *Top = dyn_cast_or_null<Instruction>(TopH);
    Value *X = XH;
    if (!I || !X || !Top)
      continue;
    Changed = true;
    if (Permute && !Swap04) {
      // Whether the thread's bits 0 and 4 (1 and 3) differ.
      IRBuilder<> E(&*F.getEntryBlock().getFirstInsertionPt());
      Value *Lane = E.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup,
                                      ArrayRef<Type *>(), ArrayRef<Value *>());
      Value *Mix = E.CreateXor(Lane, E.CreateLShr(Lane, 4));
      Swap04 = E.CreateICmpNE(E.CreateAnd(Mix, 1), E.getInt32(0));
      Value *Mix2 = E.CreateXor(Lane, E.CreateLShr(Lane, 2));
      Swap13 = E.CreateICmpNE(E.CreateAnd(Mix2, 2), E.getInt32(0));
    }
    IRBuilder<> B(Top->getNextNode());
    if (Permute)
      for (auto [Swap, K] : {std::pair(Swap04, 17), std::pair(Swap13, 10)})
        X = B.CreateSelect(
            Swap,
            B.CreateIntrinsic(Intrinsic::agx_simd_shuffle_xor, {X->getType()},
                              {X, B.getInt32(K)}),
            X);
    Value *R = B.CreateIntrinsic(Intrinsic::agx_simd_reduce, {X->getType()},
                                 {X, B.getInt32(Op)});
    R->takeName(I);
    I->replaceAllUsesWith(R);
    RecursivelyDeleteTriviallyDeadInstructions(I);
  }
  return Changed;
}

/// The Hillis-Steele inclusive add scan of a SIMD-group's 32 lanes,
///   x' = x + (lane >= d ? shuffle_up(x, d) : 0), d = 1, 2, 4, 8, 16,
/// is the hardware's exclusive simd_prefix of x0 plus x0: 2 instructions
/// for 15. Integer addition wraps the same in any order, so the values are
/// the same; a shuffle from an inactive lane is undefined (Metal, CUDA,
/// MLIR's gpu.shuffle), so where some lanes are inactive the prefix of the
/// active ones is one of the original's results.
bool AGXCodeGenPrepare::simdScans(Function &F, const AGXSubtarget &ST) {
  using namespace PatternMatch;
  if (!AGXSIMDScans || !ST.hasSIMDReduce())
    return false;
  // The prefix goes after the last shuffle (d = 16), which the threads of
  // the scan execute together: the last add may have been sunk into a
  // branch (e.g. under the last thread's store of the SIMD-group's total),
  // where a prefix would see only that branch's threads. All five shuffles
  // are in one block.
  // (value handles that follow replacement: a scan of another scan's result
  // has that result as its base, which the first replacement replaces)
  SmallVector<std::tuple<WeakTrackingVH, WeakTrackingVH, WeakTrackingVH>, 4>
      Scans;
  for (Instruction &I : instructions(F)) {
    if (I.getOpcode() != Instruction::Add || !I.getType()->isIntegerTy(32))
      continue;
    Value *X = &I;
    Instruction *Top = nullptr;
    bool Ok = true;
    for (uint64_t D : {16, 8, 4, 2, 1}) {
      auto *A = dyn_cast<BinaryOperator>(X);
      if (!A || A->getOpcode() != Instruction::Add) {
        Ok = false;
        break;
      }
      bool Found = false;
      for (unsigned K = 0; K < 2 && !Found; ++K) {
        Value *Prev = A->getOperand(K);
        auto *Sel = dyn_cast<SelectInst>(A->getOperand(1 - K));
        if (!Sel)
          continue;
        for (bool OnTrue : {true, false}) {
          Value *Sh = OnTrue ? Sel->getTrueValue() : Sel->getFalseValue();
          Value *Zero = OnTrue ? Sel->getFalseValue() : Sel->getTrueValue();
          if (match(Zero, m_Zero()) &&
              match(Sh, m_Intrinsic<Intrinsic::agx_simd_shuffle_up>(
                            m_Specific(Prev), m_SpecificInt(D))) &&
              keepsLanesFrom(Sel->getCondition(), OnTrue, D)) {
            auto *ShI = cast<Instruction>(Sh);
            if (!Top)
              Top = ShI;
            if (ShI->getParent() != Top->getParent())
              break;
            X = Prev;
            Found = true;
            break;
          }
        }
      }
      if (!Found) {
        Ok = false;
        break;
      }
    }
    if (Ok)
      Scans.push_back({&I, X, Top});
  }
  bool Changed = false;
  for (auto &[EndH, VH, TopH] : Scans) {
    auto *End = dyn_cast_or_null<Instruction>(EndH);
    auto *Top = dyn_cast_or_null<Instruction>(TopH);
    Value *V = VH;
    if (!End || !Top || !V)
      continue;
    Changed = true;
    IRBuilder<> B(Top->getNextNode());
    Value *Prefix = B.CreateIntrinsic(Intrinsic::agx_simd_prefix,
                                      {V->getType()},
                                      {V, B.getInt32(AGX::SIMD_IADD)});
    End->replaceAllUsesWith(B.CreateAdd(Prefix, V));
    RecursivelyDeleteTriviallyDeadInstructions(End);
  }
  return Changed;
}

static cl::opt<unsigned> AGXMergeLoopExits(
    "agx-merge-loop-exits", cl::init(16), cl::Hidden,
    cl::desc("Largest latch of side-effect-free instructions a two-block loop "
             "exiting from both blocks runs in its header, one exit (0: off)"));

/// A loop of two blocks that exits from both (Mandelbrot's escape test in
/// the header, its iteration count in the latch):
///   H: ...; br e1, XH, T      T: (arithmetic); br e2, XT, H
/// is structurized as an if inside the loop, with flags for which exit was
/// taken, every iteration. Where T has no side effects and is small, it
/// runs in H, the loop has one exit on e1 | e2, and a block after it goes
/// to XH where e1 held, else to XT: T's work is wasted only on the last
/// iteration of the threads that leave by e1.
static cl::opt<bool> AGXSignedWideMuls(
    "agx-signed-wide-muls", cl::init(true), cl::Hidden,
    cl::desc("A 64-bit product of a sign-extended i32 and a zero-extended "
             "non-negative i32 as the product of two sign extensions"));

/// mul i64 (sext a), (zext b) with b not negative (zext nneg, or known): the
/// zext is b's sign extension, made next to the mul, so that the selector
/// sees two sign extensions (one imadd with .sx sources) and not an unsigned
/// product with a correction of the high word by a's sign (three
/// instructions), which it also does when the zext is in another block (an
/// i64 register).
bool AGXCodeGenPrepare::signedWideMuls(Function &F) {
  if (!AGXSignedWideMuls)
    return false;
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    auto *Mul = dyn_cast<BinaryOperator>(&I);
    if (!Mul || Mul->getOpcode() != Instruction::Mul ||
        !Mul->getType()->isIntegerTy(64))
      continue;
    for (unsigned K = 0; K < 2; ++K) {
      auto *S = dyn_cast<SExtInst>(Mul->getOperand(K));
      auto *Z = dyn_cast<ZExtInst>(Mul->getOperand(1 - K));
      if (!S || !Z || !S->getSrcTy()->isIntegerTy(32) ||
          !Z->getSrcTy()->isIntegerTy(32))
        continue;
      Value *B = Z->getOperand(0);
      if (!Z->hasNonNeg() && !isKnownNonNegative(B, SimplifyQuery(DL)))
        continue;
      Mul->setOperand(1 - K,
                      new SExtInst(B, Mul->getType(), B->getName() + ".sx",
                                   Mul->getIterator()));
      if (Z->use_empty())
        Z->eraseFromParent();
      Changed = true;
      break;
    }
  }
  return Changed;
}

static cl::opt<bool> AGXMergeSameConditionIfs(
    "agx-merge-same-condition-ifs", cl::init(true), cl::Hidden,
    cl::desc("Merge an if on c whose join only holds phis and branches on c "
             "again with the next if (JumpThreading, off on divergent "
             "targets)"));

/// A: br c, T1, M; T1: ...; br M; M: phis; br c, T2, J; T2: ...; br J (the
/// same c, the same side): M's branch takes T2 exactly when A's took T1, so
/// T2's code runs at the end of T1 (with M's phis as T1's values) and A
/// branches to J; M's phis become phis of J. One if instead of two: one
/// push, skip and pop fewer, in the PTX front end's guarded loads
/// (v = c ? a[i] : x; w = c ? b[i] : y). JumpThreading would thread it but
/// does nothing on targets with branch divergence; the threads that run T2
/// are those that run T1 either way. T2 without convergent calls (whose
/// threads would be the same, but which are left in place).
bool AGXCodeGenPrepare::mergeSameConditionIfs(Function &F) {
  if (!AGXMergeSameConditionIfs)
    return false;
  bool Changed = false;
  for (bool Again = true; Again;) {
    Again = false;
    for (BasicBlock &MB : F) {
      BasicBlock *M = &MB;
      auto *Br2 = dyn_cast<CondBrInst>(M->getTerminator());
      if (!Br2 || M->getFirstNonPHIIt() != Br2->getIterator() ||
          pred_size(M) != 2)
        continue;
      Value *C = Br2->getCondition();
      BasicBlock *A = nullptr, *T1 = nullptr;
      unsigned K = 0;
      for (BasicBlock *P : predecessors(M)) {
        auto *Br1 = dyn_cast<CondBrInst>(P->getTerminator());
        if (!Br1 || Br1->getCondition() != C)
          continue;
        for (unsigned I = 0; I < 2; ++I)
          if (Br1->getSuccessor(1 - I) == M && Br1->getSuccessor(I) != M)
            A = P, T1 = Br1->getSuccessor(I), K = I;
      }
      if (!A || T1 == M || !T1->getSinglePredecessor() ||
          T1->getSingleSuccessor() != M ||
          !isa<UncondBrInst>(T1->getTerminator()) || isa<PHINode>(T1->front()))
        continue;
      if (!llvm::is_contained(predecessors(M), T1))
        continue;
      BasicBlock *T2 = Br2->getSuccessor(K), *J = Br2->getSuccessor(1 - K);
      if (T2 == J || T2 == M || J == M || T2 == T1 || J == A || J == T1 ||
          T2->getSinglePredecessor() != M || T2->getSingleSuccessor() != J ||
          !isa<UncondBrInst>(T2->getTerminator()) ||
          isa<PHINode>(T2->front()) || pred_size(J) != 2 ||
          !llvm::is_contained(predecessors(J), T2))
        continue;
      bool Bad = false;
      for (Instruction &I : *T2)
        if (auto *CB = dyn_cast<CallBase>(&I))
          Bad |= CB->isConvergent();
      SmallVector<PHINode *, 4> Phis;
      for (PHINode &P : M->phis()) {
        Phis.push_back(&P);
        for (Value *V : P.incoming_values())
          if (auto *I = dyn_cast<Instruction>(V))
            Bad |= I->getParent() == M;
      }
      if (Bad)
        continue;
      auto fromA = [&](Value *V) {
        auto *P = dyn_cast<PHINode>(V);
        return P && P->getParent() == M ? P->getIncomingValueForBlock(A) : V;
      };
      auto fromT1 = [&](Value *V) {
        auto *P = dyn_cast<PHINode>(V);
        return P && P->getParent() == M ? P->getIncomingValueForBlock(T1) : V;
      };
      // J's phis: M's edge becomes A's, T2's becomes T1's.
      for (PHINode &Q : J->phis())
        for (unsigned I = 0, E = Q.getNumIncomingValues(); I != E; ++I) {
          if (Q.getIncomingBlock(I) == M) {
            Q.setIncomingValue(I, fromA(Q.getIncomingValue(I)));
            Q.setIncomingBlock(I, A);
          } else if (Q.getIncomingBlock(I) == T2) {
            Q.setIncomingValue(I, fromT1(Q.getIncomingValue(I)));
            Q.setIncomingBlock(I, T1);
          }
        }
      // T2's code, with T1's values of M's phis, at the end of T1.
      for (PHINode *P : Phis)
        P->replaceUsesWithIf(P->getIncomingValueForBlock(T1), [&](Use &U) {
          return cast<Instruction>(U.getUser())->getParent() == T2;
        });
      T1->getTerminator()->eraseFromParent();
      T1->splice(T1->end(), T2);
      cast<UncondBrInst>(T1->getTerminator())->setSuccessor(0, J);
      cast<CondBrInst>(A->getTerminator())->setSuccessor(1 - K, J);
      for (PHINode *P : Phis) {
        if (!P->use_empty()) {
          PHINode *N = PHINode::Create(P->getType(), 2, P->getName(),
                                       J->begin());
          N->addIncoming(P->getIncomingValueForBlock(A), A);
          N->addIncoming(P->getIncomingValueForBlock(T1), T1);
          P->replaceAllUsesWith(N);
        }
        P->eraseFromParent();
      }
      Br2->eraseFromParent();
      M->eraseFromParent();
      T2->eraseFromParent();
      Changed = Again = true;
      break;
    }
  }
  return Changed;
}

bool AGXCodeGenPrepare::mergeLoopExits(Function &F) {
  if (!AGXMergeLoopExits)
    return false;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  SmallVector<Loop *, 4> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (L->isInnermost() && L->getNumBlocks() == 2)
      Loops.push_back(L);
  bool Changed = false;
  for (Loop *L : Loops) {
    BasicBlock *H = L->getHeader(), *T = L->getLoopLatch();
    if (!T || T == H || T->getSinglePredecessor() != H)
      continue;
    auto *BH = dyn_cast<CondBrInst>(H->getTerminator());
    auto *BT = dyn_cast<CondBrInst>(T->getTerminator());
    if (!BH || !BT)
      continue;
    bool HToT = BH->getSuccessor(0) == T;
    BasicBlock *XH = BH->getSuccessor(HToT ? 1 : 0);
    bool TToH = BT->getSuccessor(0) == H;
    BasicBlock *XT = BT->getSuccessor(TToH ? 1 : 0);
    if (BH->getSuccessor(HToT ? 0 : 1) != T ||
        BT->getSuccessor(TToH ? 0 : 1) != H || L->contains(XH) ||
        L->contains(XT) || XH == XT)
      continue;
    unsigned N = 0;
    bool Ok = true;
    for (Instruction &I : *T) {
      if (I.isTerminator())
        continue;
      if (isa<PHINode>(I) || ++N > AGXMergeLoopExits ||
          !isSafeToSpeculativelyExecute(&I))
        Ok = false;
    }
    if (!Ok)
      continue;
    for (Instruction &I : make_early_inc_range(*T))
      if (!I.isTerminator())
        I.moveBefore(BH->getIterator());
    IRBuilder<> B(BH);
    Value *E1 = BH->getCondition(), *E2 = BT->getCondition();
    if (HToT)
      E1 = B.CreateNot(E1);
    if (TToH)
      E2 = B.CreateNot(E2);
    BasicBlock *X = BasicBlock::Create(F.getContext(), H->getName() + ".exit",
                                       &F, XH);
    B.CreateCondBr(B.CreateOr(E1, E2), X, H);
    IRBuilder<>(X).CreateCondBr(E1, XH, XT);
    BH->eraseFromParent();
    for (PHINode &P : H->phis())
      P.replaceIncomingBlockWith(T, H);
    for (PHINode &P : XH->phis())
      P.replaceIncomingBlockWith(H, X);
    for (PHINode &P : XT->phis())
      P.replaceIncomingBlockWith(T, X);
    T->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXForwardOwnSlots(
    "agx-forward-own-slots", cl::init(true), cl::Hidden,
    cl::desc("Loads of a thread's own slot of threadgroup memory that only "
             "its thread writes as the value it last stored there"));

/// Whether S is a thread's index in its threadgroup, unique to it: the
/// linear index, or the x position in a threadgroup known to be one row
/// ("agx-threads-per-threadgroup"="X,1,1"), maybe extended.
static bool isUniqueThreadIndex(const SCEV *S, const Function &F) {
  while (auto *C = dyn_cast<SCEVCastExpr>(S))
    S = C->getOperand();
  auto *U = dyn_cast<SCEVUnknown>(S);
  auto *II = U ? dyn_cast<IntrinsicInst>(U->getValue()) : nullptr;
  if (!II)
    return false;
  if (II->getIntrinsicID() == Intrinsic::agx_thread_index_in_threadgroup)
    return true;
  if (II->getIntrinsicID() != Intrinsic::agx_thread_position_in_threadgroup_x)
    return false;
  Attribute A = F.getFnAttribute("agx-threads-per-threadgroup");
  if (!A.isStringAttribute())
    return false;
  SmallVector<StringRef, 3> Parts;
  A.getValueAsString().split(Parts, ',');
  return Parts.size() == 3 && Parts[1] == "1" && Parts[2] == "1";
}

/// A threadgroup array (internal, undefined at the start) whose every write
/// is a plain store at G + size * tid, tid unique to each thread (a tree
/// reduction's sh[tid] = v; barrier; sh[tid] = sh[tid] + sh[tid + s]): no
/// other thread ever writes a thread's own slot, so its loads of that slot
/// read what it last stored there (or the undefined start), whatever the
/// barriers between. Those loads become that value (SSAUpdater's phis);
/// the stores stay, for the other threads that read them.
bool AGXCodeGenPrepare::forwardOwnSlots(Function &F) {
  if (!AGXForwardOwnSlots)
    return false;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII, &F);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  bool Changed = false;
  for (GlobalVariable &G : F.getParent()->globals()) {
    if (G.getAddressSpace() != AGX::THREADGROUP_ADDRESS ||
        !G.hasLocalLinkage() || !G.hasInitializer() ||
        !isa<UndefValue>(G.getInitializer()))
      continue;
    // Every use: addresses, simple loads and stores, in this function.
    SmallVector<LoadInst *, 8> Loads;
    SmallVector<StoreInst *, 8> Stores;
    SmallVector<Value *, 16> Work{&G};
    SmallPtrSet<Value *, 16> Seen;
    bool Ok = true;
    while (!Work.empty() && Ok) {
      Value *V = Work.pop_back_val();
      if (!Seen.insert(V).second)
        continue;
      for (User *U : V->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I) {
          if (isa<ConstantExpr>(U))
            Work.push_back(U);
          else
            Ok = false;
          continue;
        }
        if (I->getFunction() != &F) {
          Ok = false;
        } else if (auto *Ld = dyn_cast<LoadInst>(I)) {
          Ok &= Ld->isSimple();
          Loads.push_back(Ld);
        } else if (auto *St = dyn_cast<StoreInst>(I)) {
          Ok &= St->isSimple() && St->getPointerOperand() == V;
          Stores.push_back(St);
        } else if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I)) {
          Work.push_back(I);
        } else {
          Ok = false;
        }
      }
    }
    if (!Ok || Stores.empty())
      continue;
    // One own slot: every store at G + size * tid.
    const SCEV *Own = SE.getSCEV(Stores.front()->getPointerOperand());
    Type *Ty = Stores.front()->getValueOperand()->getType();
    uint64_t Size = F.getDataLayout().getTypeStoreSize(Ty);
    for (StoreInst *St : Stores)
      if (SE.getSCEV(St->getPointerOperand()) != Own ||
          St->getValueOperand()->getType() != Ty)
        Ok = false;
    const SCEV *Off = SE.getMinusSCEV(Own, SE.getSCEV(&G));
    auto *M = dyn_cast<SCEVMulExpr>(Off);
    auto *K = M && M->getNumOperands() == 2
                  ? dyn_cast<SCEVConstant>(M->getOperand(0))
                  : nullptr;
    if (!Ok || !K || K->getAPInt() != Size ||
        !isUniqueThreadIndex(M->getOperand(1), F))
      continue;
    SSAUpdater SSA;
    SSA.Initialize(Ty, G.getName().str() + ".own");
    for (StoreInst *St : Stores) {
      BasicBlock *BB = St->getParent();
      // (the last store of each block)
      StoreInst *Last = St;
      for (StoreInst *O : Stores)
        if (O->getParent() == BB && Last->comesBefore(O))
          Last = O;
      SSA.AddAvailableValue(BB, Last->getValueOperand());
    }
    for (LoadInst *Ld : Loads) {
      if (Ld->getType() != Ty || SE.getSCEV(Ld->getPointerOperand()) != Own)
        continue;
      Value *V = nullptr;
      for (StoreInst *St : Stores)
        if (St->getParent() == Ld->getParent() && St->comesBefore(Ld) &&
            (!V || cast<StoreInst>(V)->comesBefore(St)))
          V = St;
      V = V ? cast<StoreInst>(V)->getValueOperand()
            : SSA.GetValueInMiddleOfBlock(Ld->getParent());
      Ld->replaceAllUsesWith(V);
      Ld->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

static cl::opt<bool> AGXStrideRemainders(
    "agx-stride-remainders", cl::init(true), cl::Hidden,
    cl::desc("Remainders by 2^k of an induction stepped by a multiple of 2^k "
             "as the loop-invariant remainder of its start"));

/// A loop over a tile by the threadgroup (for (l = tid; l < n; l += 256):
/// column l % 16) takes the remainder by 2^k of an induction X = {S, +, t}
/// whose step 2^k divides. That remainder (X & (2^k - 1), X - (X & -2^k),
/// urem X, 2^k) is S's in every iteration, whatever the wrap-around (the
/// step only adds multiples of 2^k): it is computed once, before the loop.
/// (The quotient as an induction of its own, {S >> k, +, t >> k}, cost
/// sgemm_reg4x4 six registers and tput: L15-E249.)
bool AGXCodeGenPrepare::strideDivisions(Function &F) {
  using namespace PatternMatch;
  if (!AGXStrideRemainders)
    return false;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  if (LI.empty())
    return false;
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII, &F);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  struct Rewrite {
    Instruction *I;
    unsigned K;
    const SCEVAddRecExpr *AR;
  };
  SmallVector<Rewrite, 8> Work;
  for (Instruction &I : instructions(F)) {
    if (!I.getType()->isIntegerTy() || !LI.getLoopFor(I.getParent()))
      continue;
    Value *X, *Y;
    const APInt *C;
    unsigned W = I.getType()->getIntegerBitWidth(), K;
    if (match(&I, m_And(m_Value(X), m_APInt(C))) && C->isMask())
      K = C->countr_one();
    else if (match(&I, m_URem(m_Value(X), m_APInt(C))) && C->isPowerOf2())
      K = C->logBase2();
    else if (match(&I, m_Sub(m_Value(X), m_And(m_Value(Y), m_APInt(C)))) &&
             C->isNegatedPowerOf2() && SE.isSCEVable(X->getType()) &&
             SE.getSCEV(X) == SE.getSCEV(Y))
      K = C->countr_zero();
    else
      continue;
    if (!K || K >= W || !SE.isSCEVable(X->getType()))
      continue;
    auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(X));
    if (!AR || !AR->isAffine() || !AR->getLoop()->contains(&I))
      continue;
    // (a loop entered from a branch gets the preheader loop-simplify gives)
    Loop *L = const_cast<Loop *>(AR->getLoop());
    if (!L->getLoopPreheader() &&
        !InsertPreheaderForLoop(L, &DT, &LI, nullptr, false))
      continue;
    auto *T = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    if (!T || T->getAPInt().isZero() || T->getAPInt().countr_zero() < K)
      continue;
    Work.push_back({&I, K, AR});
  }
  if (Work.empty())
    return false;
  SCEVExpander Exp(SE, "agx.stride");
  bool Changed = false;
  for (const Rewrite &R : Work) {
    Instruction *Pre = R.AR->getLoop()->getLoopPreheader()->getTerminator();
    if (!Exp.isSafeToExpandAt(R.AR->getStart(), Pre))
      continue;
    Type *Ty = R.I->getType();
    Value *S = Exp.expandCodeFor(R.AR->getStart(), Ty, Pre);
    Value *New = IRBuilder<>(Pre).CreateAnd(
        S, APInt::getLowBitsSet(Ty->getIntegerBitWidth(), R.K));
    R.I->replaceAllUsesWith(New);
    New->takeName(R.I);
    Changed = true;
  }
  for (const Rewrite &R : Work)
    if (R.I->use_empty())
      RecursivelyDeleteTriviallyDeadInstructions(R.I);
  return Changed;
}

bool AGXCodeGenPrepare::foldLaneIdentities(Function &F) {
  if (!AGXFoldLaneIdentities)
    return false;
  bool HasShuffle = false;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (isLaneMovement(II)) {
        HasShuffle = true;
        break;
      }
  if (!HasShuffle)
    return false;

  LaneEvaluator LE;
  SmallVector<WeakTrackingVH, 32> Roots;
  for (Instruction &I : instructions(F)) {
    if (!isLaneMovement(&I) || I.getType()->isPointerTy() ||
        I.getType()->isPtrOrPtrVectorTy())
      continue;
    // Only the last move of a chain: its users compute with it.
    if (llvm::all_of(I.users(), [](const User *U) { return isLaneMovement(U); }))
      continue;
    Roots.push_back(&I);
  }

  bool Changed = false;
  for (WeakTrackingVH &H : Roots) {
    auto *I = dyn_cast_or_null<Instruction>(H);
    if (!I || I->use_empty())
      continue;
    auto *VT = dyn_cast<FixedVectorType>(I->getType());
    unsigned Elts = VT ? VT->getNumElements() : 1;
    Value *Base = nullptr;
    int BaseElt = -2;
    bool Identity = true, Moved = false;
    for (unsigned E = 0; E < Elts && Identity; ++E)
      for (unsigned L = 0; L < 32 && Identity; ++L) {
        LaneSource S = traceLane(I, L, VT ? (int)E : -1, LE);
        if (S.V == I || S.Lane != L || (Base && S.V != Base)) {
          Identity = false;
          break;
        }
        Base = S.V;
        Moved |= S.V != I;
        int Want = VT ? (int)E : S.Elt;
        if (VT ? S.Elt != Want : (BaseElt != -2 && S.Elt != BaseElt)) {
          Identity = false;
          break;
        }
        BaseElt = S.Elt;
      }
    if (!Identity || !Moved || !Base || Base == I)
      continue;
    IRBuilder<> B(I);
    Value *R = Base;
    if (!VT && BaseElt != -1)
      R = B.CreateExtractElement(Base, B.getInt64(BaseElt));
    if (R->getType() != I->getType()) {
      if (R->getType()->getPrimitiveSizeInBits() != I->getType()->getPrimitiveSizeInBits())
        continue;
      R = B.CreateBitCast(R, I->getType());
    }
    I->replaceAllUsesWith(R);
    RecursivelyDeleteTriviallyDeadInstructions(I);
    Changed = true;
  }
  return Changed;
}

/// Whether simd_shuffle with index Idx, a function of the thread index
/// alone, reads the thread it names in every thread: the hardware reads from
/// each source quad the thread at the OR of the low two index bits of that
/// quad's own threads (measured: a reader's own low bits do not matter), so
/// every thread of each quad that is read must hold the position its readers
/// ask for. (Inactive threads pass 0 in r0h, which keeps this exact for any
/// execution mask.)
static bool readsNamedThreads(const Value *Idx, LaneEvaluator &LE) {
  unsigned Index[32];
  for (unsigned L = 0; L < 32; ++L) {
    auto I = LE.eval(Idx, L);
    if (!I || *I >= 32)
      return false;
    Index[L] = *I;
  }
  for (unsigned L = 0; L < 32; ++L) {
    unsigned Quad = Index[L] & ~3u, Pos = Index[L] & 3;
    for (unsigned T = Quad; T < Quad + 4; ++T)
      if ((Index[T] & 3) != Pos)
        return false;
  }
  return true;
}

/// simd_shuffle reads from each quad the thread at the OR of the low two
/// index bits of the quad's threads (applegpu documents this, Mesa works
/// around it). A divergent index computed from the thread index alone whose
/// source quads' own threads all hold the position they are read at
/// (readsNamedThreads) reads the thread it names too. A shuffle whose index is the same in a quad (an immediate, or
/// low bits that are uniform) reads the thread it names. Divergent indices
/// are split into the four quad positions:
///
///   r = shuffle(v, idx & ~3); r = idx & 3 == k ? shuffle(v, idx & ~3 | k) : r
///
/// leaving out the positions that the known bits of the index exclude.
/// Quad shuffles take the four immediate positions instead. Relative
/// shuffles with divergent deltas become shuffles of the computed thread;
/// threads whose source is outside the SIMD-group (or quad) keep their own
/// value, as Metal documents for simd_shuffle_up and simd_shuffle_down.
static cl::opt<bool> AGXShuffleSelects(
    "agx-shuffle-selects", cl::init(true), cl::Hidden,
    cl::desc("A shuffle at select(c, i, thread index) as a select between "
             "the shuffle at i and the value (a shuffle_xor for i = thread "
             "index ^ k)"));

/// A thread reads its own value at its own index: a SIMD-group shuffle at
/// select(C, I, thread index) (through a mask that leaves both alone) is
/// select(C, shuffle at I, the value), and a shuffle at the thread index
/// xor a constant is a shuffle_xor, whose index is no longer divergent:
/// partial reductions' butterflies (the lane's partner where it is in range,
/// else the lane) were split into four shuffles each. Takes the shuffles it
/// rewrites out of Worklist and adds the ones it makes that still need a
/// split.
static bool shuffleSelectedLanes(SmallVectorImpl<IntrinsicInst *> &Worklist,
                                 const DataLayout &DL) {
  using namespace PatternMatch;
  bool Changed = false;
  auto IsLane = [](Value *V) {
    auto *II = dyn_cast<IntrinsicInst>(V);
    return II && II->getIntrinsicID() == Intrinsic::agx_thread_index_in_simdgroup;
  };
  SmallVector<IntrinsicInst *, 8> Out;
  for (IntrinsicInst *II : Worklist) {
    Value *Idx = II->getArgOperand(1);
    // a mask that keeps every bit the index can have (the thread index and
    // its xor with a constant are below 32)
    const APInt *K;
    Value *L;
    auto IsLaneXor = [&](Value *V) {
      return match(V, m_c_Xor(m_Value(L), m_APInt(K))) && IsLane(L) &&
             K->ult(32);
    };
    auto Below32 = [&](Value *V) { return IsLane(V) || IsLaneXor(V); };
    const APInt *M;
    Value *X, *C, *T, *E;
    if (match(Idx, m_And(m_Value(X), m_APInt(M))) &&
        ((computeKnownBits(X, DL).Zero | *M).isAllOnes() ||
         (M->countr_one() >= 5 &&
          match(X, m_Select(m_Value(), m_Value(T), m_Value(E))) &&
          Below32(T) && Below32(E))))
      Idx = X;
    if (II->getIntrinsicID() != Intrinsic::agx_simd_shuffle ||
        !match(Idx, m_Select(m_Value(C), m_Value(T), m_Value(E))) ||
        IsLane(T) == IsLane(E)) {
      Out.push_back(II);
      continue;
    }
    bool LaneIfTrue = IsLane(T);
    Value *I = LaneIfTrue ? E : T;
    Value *V = II->getArgOperand(0);
    IRBuilder<> B(II);
    Value *R;
    if (IsLaneXor(I)) {
      R = B.CreateIntrinsic(Intrinsic::agx_simd_shuffle_xor, {V->getType()},
                            {V, B.getInt32(K->getZExtValue())});
    } else {
      auto *S = cast<IntrinsicInst>(B.CreateIntrinsic(
          Intrinsic::agx_simd_shuffle, {V->getType()}, {V, I}));
      R = S;
      if (!isa<Constant>(I))
        Out.push_back(S);
    }
    Value *Sel = LaneIfTrue ? B.CreateSelect(C, V, R) : B.CreateSelect(C, R, V);
    Sel->takeName(II);
    II->replaceAllUsesWith(Sel);
    II->eraseFromParent();
    Changed = true;
  }
  Worklist.assign(Out.begin(), Out.end());
  return Changed;
}

bool AGXCodeGenPrepare::lowerDivergentShuffles(Function &F,
                                               const TargetTransformInfo &TTI) {
  SmallVector<IntrinsicInst *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    switch (II->getIntrinsicID()) {
    case Intrinsic::agx_simd_shuffle:
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_quad_shuffle:
    case Intrinsic::agx_quad_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_up:
    case Intrinsic::agx_quad_shuffle_down:
    case Intrinsic::agx_simd_shuffle_rotate_up:
    case Intrinsic::agx_quad_shuffle_rotate_up:
      if (!isa<Constant>(II->getArgOperand(1)))
        Worklist.push_back(II);
      break;
    default:
      break;
    }
  }

  if (AGXUnsplitShuffles)
    return false;
  bool Changed = false;
  if (AGXShuffleSelects)
    Changed |= shuffleSelectedLanes(Worklist, F.getDataLayout());
  // Most compute kernels have no variable shuffle indices. Do not compute
  // whole-function divergence (and its dominator/cycle analyses) for them.
  // Build it after matrix lowering, which can itself introduce shuffles.
  if (Worklist.empty())
    return Changed;
  DominatorTree DT(F);
  CycleInfo CI;
  CI.compute(F);
  UniformityInfo UI(DT, CI, &TTI);
  UI.compute();
  LaneEvaluator LE;
  llvm::erase_if(Worklist, [&](IntrinsicInst *II) {
    if (!UI.isDivergentAtUse(II->getArgOperandUse(1)))
      return true;
    return AGXQuadUniformShuffles &&
           II->getIntrinsicID() == Intrinsic::agx_simd_shuffle &&
           readsNamedThreads(II->getArgOperand(1), LE);
  });

  for (IntrinsicInst *II : Worklist) {
    IRBuilder<> B(II);
    Value *V = II->getArgOperand(0);
    Value *Idx = II->getArgOperand(1);
    Type *Ty = V->getType();
    Intrinsic::ID ID = II->getIntrinsicID();
    bool Quad = ID == Intrinsic::agx_quad_shuffle ||
                ID == Intrinsic::agx_quad_shuffle_xor ||
                ID == Intrinsic::agx_quad_shuffle_up ||
                ID == Intrinsic::agx_quad_shuffle_down ||
                ID == Intrinsic::agx_quad_shuffle_rotate_up;
    unsigned Size = Quad ? 4 : 32;

    // The thread to read, relative to the start of the quad for quad
    // shuffles.
    Value *Lane = nullptr;
    if (ID != Intrinsic::agx_simd_shuffle && ID != Intrinsic::agx_quad_shuffle) {
      Lane = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup, {});
      if (Quad)
        Lane = B.CreateAnd(Lane, 3);
    }
    switch (ID) {
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_xor:
      Idx = B.CreateAnd(B.CreateXor(Lane, Idx), Size - 1);
      break;
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_quad_shuffle_up:
      Idx = B.CreateSelect(B.CreateICmpUGE(Lane, Idx), B.CreateSub(Lane, Idx),
                           Lane);
      break;
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_quad_shuffle_down:
      Idx = B.CreateSelect(
          B.CreateICmpULT(Idx, B.CreateSub(B.getInt32(Size), Lane)),
          B.CreateAdd(Lane, Idx), Lane);
      break;
    case Intrinsic::agx_simd_shuffle_rotate_up:
    case Intrinsic::agx_quad_shuffle_rotate_up:
      Idx = B.CreateAnd(B.CreateSub(Lane, Idx), Size - 1);
      break;
    default:
      break;
    }

    Intrinsic::ID Shuffle =
        Quad ? Intrinsic::agx_quad_shuffle : Intrinsic::agx_simd_shuffle;
    Value *Low = B.CreateAnd(Idx, 3);
    Value *Base = Quad ? nullptr : B.CreateAnd(Idx, 0x1c);
    auto Position = [&](unsigned K) -> Value * {
      if (Quad)
        return B.getInt32(K);
      return K ? B.CreateOr(Base, K) : Base;
    };
    // Positions the low bits cannot take need no shuffle.
    KnownBits Known = computeKnownBits(Idx, F.getDataLayout());
    SmallVector<unsigned, 4> Positions;
    for (unsigned K = 0; K < 4; ++K)
      if ((K & Known.Zero.getZExtValue() & 3) == 0 &&
          (~K & Known.One.getZExtValue() & 3) == 0)
        Positions.push_back(K);
    Value *Result =
        B.CreateIntrinsic(Shuffle, {Ty}, {V, Position(Positions[0])});
    for (unsigned K : drop_begin(Positions)) {
      Value *R = B.CreateIntrinsic(Shuffle, {Ty}, {V, Position(K)});
      Result = B.CreateSelect(B.CreateICmpEQ(Low, B.getInt32(K)), R, Result);
    }
    Result->takeName(II);
    II->replaceAllUsesWith(Result);
    II->eraseFromParent();
  }
  return Changed || !Worklist.empty();
}

/// Whether I computes a floating point value from its operands (and so
/// reads and produces denormals according to the denormal mode).
static bool isFPArithmetic(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    return true;
  default:
    break;
  }
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::minimum:
  case Intrinsic::maximum:
  case Intrinsic::minimumnum:
  case Intrinsic::maximumnum:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::agx_exp2:
  case Intrinsic::agx_log2:
  case Intrinsic::agx_sin:
  case Intrinsic::agx_cos:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::roundeven:
  case Intrinsic::canonicalize:
  case Intrinsic::agx_rcp:
  case Intrinsic::agx_rsqrt:
    return true;
  default:
    return false;
  }
}

static bool isHalf(const Type *Ty) { return Ty->getScalarType()->isHalfTy(); }

/// llvm.agx.simdgroup.matrix.fma: D = A * B + C on 8x8 matrices spread over
/// the SIMD-group. Thread t holds the elements (r, c) and (r, c + 1) of each
/// matrix, with r = 4 * (t / 16) + (t / 2) % 4 and c = 4 * ((t / 8) % 2) +
/// 2 * (t % 2). Each thread gathers its row of A and its two columns of B
/// with shuffles and accumulates in order of k with fused multiply-adds.
///
/// On G14 the matrix unit (simd_matrix_fmadd32 / 16) computes the same chain
/// of fused multiply-adds for the same layout (measured: Apple-M2-Playground
/// results/2026-10-21-b15-simd-matrix): float flushing denormals like the
/// ALU, half keeping them. It is used where the shuffled form gives the same
/// results: float without IEEE denormals in software (ieee-fp32), half with
/// IEEE denormals, and no canonical NaNs.
bool AGXCodeGenPrepare::lowerMatrixFMA(Function &F, const AGXSubtarget &ST) {
  SmallVector<IntrinsicInst *, 2> Calls;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->getIntrinsicID() == Intrinsic::agx_simdgroup_matrix_fma)
      Calls.push_back(II);
  bool Unit = ST.hasSIMDMatrix() && !ST.hasMesaISA() && !ST.hasCanonicalNaN() &&
              ST.getGeneration() == AGXSubtarget::G14;
  bool FloatUnit =
      Unit && !(ST.hasIEEEFP32() &&
                F.getDenormalMode(APFloat::IEEEsingle()) == DenormalMode::getIEEE());
  bool HalfUnit =
      Unit && F.getDenormalMode(APFloat::IEEEhalf()) == DenormalMode::getIEEE();
  for (IntrinsicInst *II : Calls) {
    IRBuilder<> B(II);
    Type *VT = II->getType();
    Value *A = II->getArgOperand(0), *Bm = II->getArgOperand(1),
          *C = II->getArgOperand(2);
    Type *ET = VT->getScalarType();
    if ((ET->isFloatTy() && FloatUnit) || (ET->isHalfTy() && HalfUnit)) {
      Type *IT = ET->isFloatTy() ? B.getInt64Ty() : B.getInt32Ty();
      Intrinsic::ID ID = ET->isFloatTy() ? Intrinsic::agx_simd_matrix_fmadd32
                                         : Intrinsic::agx_simd_matrix_fmadd16;
      Value *D = B.CreateIntrinsic(ID, {},
                                   {B.CreateBitCast(A, IT), B.CreateBitCast(Bm, IT),
                                    B.CreateBitCast(C, IT)});
      II->replaceAllUsesWith(B.CreateBitCast(D, VT));
      II->eraseFromParent();
      continue;
    }
    Value *T = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup, {});
    auto Bits = [&](Value *V, unsigned Shift, unsigned Mask) {
      return B.CreateAnd(B.CreateLShr(V, Shift), Mask);
    };
    // The thread holding the elements 2j and 2j + 1 of row r.
    auto Holder = [&](Value *R, Value *J) {
      Value *L = B.CreateShl(B.CreateLShr(R, 2), 4);
      L = B.CreateOr(L, B.CreateShl(B.CreateLShr(J, 1), 3));
      L = B.CreateOr(L, B.CreateShl(B.CreateAnd(R, 3), 1));
      return B.CreateOr(L, B.CreateAnd(J, 1));
    };
    auto Shuffle = [&](Value *V, unsigned Elt, Value *Lane) {
      Value *X = B.CreateExtractElement(V, Elt);
      return B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {X->getType()}, {X, Lane});
    };
    Value *Row = B.CreateOr(B.CreateShl(Bits(T, 4, 1), 2), Bits(T, 1, 3));
    Value *ColPair = B.CreateOr(B.CreateShl(Bits(T, 3, 1), 1), Bits(T, 0, 1));
    SmallVector<Value *, 8> RowA, ColB0, ColB1;
    for (unsigned J = 0; J < 4; ++J) {
      Value *Lane = Holder(Row, B.getInt32(J));
      RowA.push_back(Shuffle(A, 0, Lane));
      RowA.push_back(Shuffle(A, 1, Lane));
    }
    for (unsigned K = 0; K < 8; ++K) {
      Value *Lane = Holder(B.getInt32(K), ColPair);
      ColB0.push_back(Shuffle(Bm, 0, Lane));
      ColB1.push_back(Shuffle(Bm, 1, Lane));
    }
    Value *D0 = B.CreateExtractElement(C, uint64_t(0));
    Value *D1 = B.CreateExtractElement(C, 1);
    for (unsigned K = 0; K < 8; ++K) {
      D0 = B.CreateIntrinsic(Intrinsic::fma, {D0->getType()}, {RowA[K], ColB0[K], D0});
      D1 = B.CreateIntrinsic(Intrinsic::fma, {D1->getType()}, {RowA[K], ColB1[K], D1});
    }
    Value *R = B.CreateInsertElement(PoisonValue::get(VT), D0, uint64_t(0));
    R = B.CreateInsertElement(R, D1, 1);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
  }
  return !Calls.empty();
}

/// The hardware minimum and maximum return a denormal operand as it is, also
/// where float denormals are flushed: flush their results.
bool AGXCodeGenPrepare::flushFloatMinMax(Function &F) {
  if (F.getDenormalMode(APFloat::IEEEsingle()).Output == DenormalMode::IEEE)
    return false;
  SmallVector<IntrinsicInst *, 8> MinMax;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !II->getType()->isFloatTy() || II->hasMetadata("agx.ieee") ||
        II->hasMetadata("agx.exact"))
      continue;
    switch (II->getIntrinsicID()) {
    case Intrinsic::minnum:
    case Intrinsic::maxnum:
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    case Intrinsic::minimumnum:
    case Intrinsic::maximumnum:
      MinMax.push_back(II);
      break;
    default:
      break;
    }
  }
  for (IntrinsicInst *II : MinMax) {
    IRBuilder<> B(II->getNextNode());
    SmallVector<Use *, 4> Uses;
    // The hardware's compares flush denormals, and a min / max gives what
    // flushing its operands would once its own result is flushed.
    for (Use &U : II->uses())
      if (!isa<FCmpInst>(U.getUser()) && !llvm::is_contained(MinMax, U.getUser()))
        Uses.push_back(&U);
    if (Uses.empty())
      continue;
    Value *R = B.CreateUnaryIntrinsic(Intrinsic::canonicalize, II);
    for (Use *U : Uses)
      U->set(R);
  }
  return !MinMax.empty();
}

bool AGXCodeGenPrepare::flushHalfDenormals(Function &F) {
  DenormalMode Mode = F.getDenormalMode(APFloat::IEEEhalf());
  bool FlushInputs = Mode.Input != DenormalMode::IEEE &&
                     Mode.Input != DenormalMode::Dynamic;
  bool FlushOutputs = Mode.Output != DenormalMode::IEEE &&
                      Mode.Output != DenormalMode::Dynamic;
  if (!FlushInputs && !FlushOutputs)
    return false;
  bool PositiveZero = Mode.Output == DenormalMode::PositiveZero;

  // flush(x) = |x| < 2^-14 ? +-0 : x. NaNs compare false and pass through.
  SmallPtrSet<Value *, 16> Flushed;
  auto Flush = [&](IRBuilder<> &B, Value *X, bool Positive) -> Value * {
    if (Flushed.contains(X))
      return X;
    if (auto *C = dyn_cast<ConstantFP>(X))
      if (!C->getValueAPF().isDenormal())
        return X;
    Type *Ty = X->getType();
    Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
    Value *Small = B.CreateFCmpOLT(
        Abs, ConstantFP::get(Ty, APFloat::getSmallestNormalized(
                                     APFloat::IEEEhalf())));
    Value *Zero = Positive ? ConstantFP::get(Ty, 0.0)
                           : B.CreateBinaryIntrinsic(Intrinsic::copysign,
                                                     ConstantFP::get(Ty, 0.0),
                                                     X);
    Value *R = B.CreateSelect(Small, Zero, X);
    Flushed.insert(R);
    return R;
  };

  SmallVector<Instruction *, 32> Worklist;
  for (Instruction &I : instructions(F)) {
    // Operations with their own denormal handling (from AGXLowerFPEnv), and
    // those a flush does not change (a front end's air.flush.harmless).
    if (I.hasMetadata("agx.ieee") || I.hasMetadata("agx.exact") ||
        I.hasMetadata("air.flush.harmless"))
      continue;
    bool Arith = isFPArithmetic(I) && isHalf(I.getType());
    bool Reads = Arith || (isa<FCmpInst>(I) &&
                           isHalf(I.getOperand(0)->getType())) ||
                 (isa<FPExtInst>(I) && isHalf(I.getOperand(0)->getType())) ||
                 ((isa<FPToSIInst>(I) || isa<FPToUIInst>(I)) &&
                  isHalf(I.getOperand(0)->getType()));
    bool Writes = Arith || (isa<FPTruncInst>(I) && isHalf(I.getType()));
    if (Reads || Writes)
      Worklist.push_back(&I);
  }

  for (Instruction *I : Worklist) {
    IRBuilder<> B(I);
    bool Arith = isFPArithmetic(*I) && isHalf(I->getType());
    bool Reads = !isa<FPTruncInst>(I);
    if (FlushInputs && Reads) {
      unsigned NumOps = isa<CallBase>(I) ? cast<CallBase>(I)->arg_size()
                                         : I->getNumOperands();
      for (unsigned Op = 0; Op < NumOps; ++Op)
        if (I->getOperand(Op)->getType()->isFPOrFPVectorTy() &&
            isHalf(I->getOperand(Op)->getType()))
          I->setOperand(Op, Flush(B, I->getOperand(Op), false));
    }
    if (auto *II = dyn_cast<IntrinsicInst>(I);
        II && II->getIntrinsicID() == Intrinsic::canonicalize) {
      // The operand is flushed already; nothing else to canonicalize.
      Value *Op = II->getArgOperand(0);
      if (FlushOutputs && !Flushed.contains(Op))
        Op = Flush(B, Op, PositiveZero);
      II->replaceAllUsesWith(Op);
      II->eraseFromParent();
      continue;
    }
    if (FlushOutputs && (Arith || isa<FPTruncInst>(I))) {
      B.SetInsertPoint(I->getNextNode());
      SmallVector<User *, 4> OldUsers(I->users());
      Value *R = Flush(B, I, PositiveZero);
      I->replaceUsesWithIf(R, [&](Use &U) {
        return is_contained(OldUsers, U.getUser());
      });
    }
  }
  // The flushes are explicit now. Later combines would otherwise assume that
  // the hardware flushes and fold them away.
  AttrBuilder AB(F.getContext());
  AB.addDenormalFPEnvAttr(DenormalFPEnv(
      DenormalMode::getIEEE(), F.getDenormalMode(APFloat::IEEEsingle())));
  F.addFnAttrs(AB);
  return true;
}

bool AGXCodeGenPrepare::diagnoseDoubles(Function &F) {
  SmallVector<Instruction *, 8> Unsupported;
  for (Instruction &I : instructions(F)) {
    bool Double = (isFPArithmetic(I) || isa<FCmpInst>(I) ||
                   isa<FPToSIInst>(I) || isa<FPToUIInst>(I) ||
                   isa<FPTruncInst>(I)) &&
                  I.getOperand(0)->getType()->getScalarType()->isDoubleTy();
    Double |= (isa<SIToFPInst>(I) || isa<UIToFPInst>(I) ||
               isa<FPExtInst>(I) || isFPArithmetic(I)) &&
              I.getType()->getScalarType()->isDoubleTy();
    if (Double)
      Unsupported.push_back(&I);
  }
  if (Unsupported.empty())
    return false;
  F.getContext().diagnose(DiagnosticInfoUnsupported(
      F, "double precision floating point arithmetic (AGX has no double "
         "precision hardware)",
      Unsupported.front()->getDebugLoc()));
  for (Instruction *I : Unsupported) {
    I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    I->eraseFromParent();
  }
  return true;
}

static cl::opt<bool> AGXMergeMaskedOrs(
    "agx-merge-masked-ors", cl::init(true), cl::Hidden,
    cl::desc("Merge the masked parts of one shifted value in an or tree "
             "(byte permutes with constant selectors)"));

/// An or tree (of 32-bit ors with one use) whose leaves are masked parts of
/// a few shifted values, as a byte permute with a constant selector folds to
/// (one leaf per byte): the leaves of one value and shift, (x op c) & m1 and
/// (x op c) & m2, become one, (x op c) & (m1 | m2), which instruction
/// selection then sees as a half of a register or a bit-field insert.
static bool mergeMaskedOrs(Function &F) {
  if (!AGXMergeMaskedOrs)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  SmallVector<WeakVH, 16> Roots;
  for (Instruction &I : instructions(F)) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || BO->getOpcode() != Instruction::Or ||
        !BO->getType()->isIntegerTy(32))
      continue;
    // a root: not itself the single-use operand of another or
    if (BO->hasOneUse())
      if (auto *U = dyn_cast<BinaryOperator>(BO->user_back());
          U && U->getOpcode() == Instruction::Or)
        continue;
    Roots.push_back(BO);
  }
  for (WeakVH &H : Roots) {
    // (a root deleted with an earlier tree, as its dead leaf)
    auto *Root = dyn_cast_or_null<BinaryOperator>(H);
    if (!Root)
      continue;
    SmallVector<Value *, 8> Leaves, Work{Root};
    while (!Work.empty()) {
      Value *V = Work.pop_back_val();
      auto *O = dyn_cast<BinaryOperator>(V);
      if (O && O->getOpcode() == Instruction::Or &&
          (O == Root || O->hasOneUse())) {
        Work.push_back(O->getOperand(0));
        Work.push_back(O->getOperand(1));
        continue;
      }
      Leaves.push_back(V);
    }
    if (Leaves.size() < 3)
      continue;
    // a leaf as (shifted value, mask)
    struct Part {
      Value *Shifted;
      APInt Mask;
    };
    SmallVector<Part, 8> Parts;
    for (Value *L : Leaves) {
      Value *X;
      const APInt *M;
      if (match(L, m_And(m_Value(X), m_APInt(M))))
        Parts.push_back({X, *M});
      else
        Parts.push_back({L, APInt::getAllOnes(32)});
    }
    // the same shifted value: one instruction (x << c, x >> c) or the value
    bool Merge = false;
    for (unsigned I = 0; I < Parts.size() && !Merge; ++I)
      for (unsigned J = I + 1; J < Parts.size() && !Merge; ++J)
        Merge = Parts[I].Shifted == Parts[J].Shifted;
    if (!Merge)
      continue;
    SmallVector<Part, 8> Merged;
    for (Part &P : Parts) {
      auto It = llvm::find_if(Merged,
                              [&](Part &Q) { return Q.Shifted == P.Shifted; });
      if (It == Merged.end())
        Merged.push_back(P);
      else
        It->Mask |= P.Mask;
    }
    IRBuilder<> B(Root);
    Value *R = nullptr;
    for (Part &P : Merged) {
      Value *L =
          P.Mask.isAllOnes() ? P.Shifted : B.CreateAnd(P.Shifted, P.Mask);
      R = R ? B.CreateOr(R, L) : L;
    }
    Root->replaceAllUsesWith(R);
    RecursivelyDeleteTriviallyDeadInstructions(Root);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXBytePermutes(
    "agx-byte-permutes", cl::init(true), cl::Hidden,
    cl::desc("Build 32-bit values whose bytes are bytes (or sign bytes) of "
             "other values from 16-bit halves: bit-field inserts of bytes"));

namespace {
/// Where a byte of a value comes from: zero, byte Idx of V, or all copies of
/// the sign bit of byte Idx of V.
struct ByteSrc {
  enum Kind { Zero, Byte, Sign } K = Zero;
  Value *V = nullptr;
  unsigned Idx = 0;
  bool operator==(const ByteSrc &O) const {
    return K == O.K && (K == Zero || (V == O.V && Idx == O.Idx));
  }
};
using Bytes4 = std::array<ByteSrc, 4>;
} // namespace

/// The bytes of 32-bit V through ors of disjoint bytes, masks of whole bytes,
/// shifts by whole bytes, sign fills (ashr by whole bytes or by 31) and zero
/// extensions; Tree gets the instructions looked through.
static Bytes4 byteSources(Value *V, unsigned Depth,
                          SmallVectorImpl<Instruction *> &Tree) {
  using namespace PatternMatch;
  auto Leaf = [&](Value *L) {
    Bytes4 R;
    unsigned N = L->getType()->getIntegerBitWidth() / 8;
    for (unsigned I = 0; I < N && I < 4; ++I)
      R[I] = {ByteSrc::Byte, L, I};
    return R;
  };
  auto *I = dyn_cast<Instruction>(V);
  if (auto *C = dyn_cast<ConstantInt>(V); C && C->isZero())
    return Bytes4();
  if (!I || Depth == 0 || !I->getType()->isIntegerTy(32))
    return Leaf(V);
  Value *X, *Y;
  const APInt *C;
  Bytes4 R;
  // (the instructions below this one join Tree only with it)
  SmallVector<Instruction *, 16> Sub;
  if (match(I, m_And(m_Value(X), m_APInt(C)))) {
    for (unsigned B = 0; B < 4; ++B) {
      unsigned M = C->extractBitsAsZExtValue(8, 8 * B);
      if (M != 0 && M != 255)
        return Leaf(V);
    }
    Bytes4 S = byteSources(X, Depth - 1, Sub);
    for (unsigned B = 0; B < 4; ++B)
      if (C->extractBitsAsZExtValue(8, 8 * B))
        R[B] = S[B];
  } else if (match(I, m_Or(m_Value(X), m_Value(Y)))) {
    Bytes4 A = byteSources(X, Depth - 1, Sub);
    Bytes4 Bs = byteSources(Y, Depth - 1, Sub);
    for (unsigned B = 0; B < 4; ++B) {
      if (A[B].K != ByteSrc::Zero && Bs[B].K != ByteSrc::Zero)
        return Leaf(V);
      R[B] = A[B].K != ByteSrc::Zero ? A[B] : Bs[B];
    }
  } else if (match(I, m_Shl(m_Value(X), m_APInt(C))) && C->ult(32) &&
             C->getZExtValue() % 8 == 0) {
    unsigned K = C->getZExtValue() / 8;
    Bytes4 S = byteSources(X, Depth - 1, Sub);
    for (unsigned B = K; B < 4; ++B)
      R[B] = S[B - K];
  } else if (match(I, m_LShr(m_Value(X), m_APInt(C))) && C->ult(32) &&
             C->getZExtValue() % 8 == 0) {
    unsigned K = C->getZExtValue() / 8;
    Bytes4 S = byteSources(X, Depth - 1, Sub);
    for (unsigned B = 0; B + K < 4; ++B)
      R[B] = S[B + K];
  } else if (match(I, m_AShr(m_Value(X), m_APInt(C))) && C->ult(32) &&
             (C->getZExtValue() % 8 == 0 || C->getZExtValue() == 31)) {
    Bytes4 S = byteSources(X, Depth - 1, Sub);
    if (S[3].K == ByteSrc::Zero)
      return Leaf(V);
    ByteSrc Fill = S[3];
    if (Fill.K == ByteSrc::Byte)
      Fill.K = ByteSrc::Sign;
    unsigned K = C->getZExtValue() == 31 ? 4 : C->getZExtValue() / 8;
    for (unsigned B = 0; B < 4; ++B)
      R[B] = B + K < 4 ? S[B + K] : Fill;
  } else if (auto *Sel = dyn_cast<SelectInst>(I);
             Sel && isa<ConstantInt>(Sel->getTrueValue()) &&
             isa<ConstantInt>(Sel->getFalseValue())) {
    // whole bytes of the sign of a byte: c ? m : 0 (or c ? 0 : m) with c
    // testing bit 8 i + 7 of x (as InstCombine writes a sign fill)
    CmpPredicate Pred;
    const APInt *Bit;
    unsigned Pos;
    bool SetWhenTrue;
    if (match(Sel->getCondition(),
              m_ICmp(Pred, m_And(m_Value(X), m_APInt(Bit)), m_Zero())) &&
        Bit->isPowerOf2() && ICmpInst::isEquality(Pred)) {
      Pos = Bit->logBase2();
      SetWhenTrue = Pred == ICmpInst::ICMP_NE;
    } else if (match(Sel->getCondition(),
                     m_SpecificICmp(ICmpInst::ICMP_SLT, m_Value(X), m_Zero())) &&
               X->getType()->isIntegerTy(32)) {
      Pos = 31;
      SetWhenTrue = true;
    } else {
      return Leaf(V);
    }
    const APInt &T = cast<ConstantInt>(Sel->getTrueValue())->getValue();
    const APInt &Fv = cast<ConstantInt>(Sel->getFalseValue())->getValue();
    const APInt &M = SetWhenTrue ? T : Fv;
    if (Pos % 8 != 7 || !X->getType()->isIntegerTy(32) ||
        !(SetWhenTrue ? Fv : T).isZero())
      return Leaf(V);
    ByteSrc Src = byteSources(X, Depth - 1, Sub)[Pos / 8];
    if (Src.K == ByteSrc::Zero)
      return Leaf(V);
    if (Src.K == ByteSrc::Byte)
      Src.K = ByteSrc::Sign;
    for (unsigned B = 0; B < 4; ++B) {
      unsigned Mb = M.extractBitsAsZExtValue(8, 8 * B);
      if (Mb != 0 && Mb != 255)
        return Leaf(V);
      if (Mb)
        R[B] = Src;
    }
  } else if (match(I, m_ZExt(m_Value(X))) &&
             (X->getType()->isIntegerTy(8) || X->getType()->isIntegerTy(16))) {
    // (not a narrower value: Leaf would give it no bytes, all zero, and an
    // or with it would lose it; the cf fuzzer's (x & 255) | zext(i1),
    // L15-E467)
    Value *W;
    if (X->getType()->isIntegerTy(16) && match(X, m_Trunc(m_Value(W))) &&
        W->getType()->isIntegerTy(32)) {
      Bytes4 S = byteSources(W, Depth - 1, Sub);
      R[0] = S[0];
      R[1] = S[1];
    } else {
      R = Leaf(X);
    }
  } else {
    return Leaf(V);
  }
  Tree.append(Sub.begin(), Sub.end());
  Tree.push_back(I);
  return R;
}

/// 32-bit or trees whose bytes are bytes of other values (constant byte
/// permutes, PTX prmt, byte packing and sign extension of bytes), built half
/// by half: each 16-bit half is at most one bit-field insert of one byte
/// into the other's half (and a shift or a sign fill for the byte), the
/// halves the 32-bit register's, where that takes fewer instructions than
/// the tree.
static cl::opt<bool> AGXByteLoads(
    "agx-byte-loads", cl::init(true), cl::Hidden,
    cl::desc("a word loaded only to take its bytes apart as a vector of "
             "bytes (each zero-extended in a 16-bit half)"));

/// A device i32 load whose users only take its bytes apart (trunc, lshr by
/// whole bytes, and with low masks, zext), every value used otherwise being
/// one byte zero-extended, is a <4 x i8> load: the byte format's access puts
/// each byte, zero-extended, in a 16-bit half, without the shifts and masks
/// (RGBA8 pixels). A byte vector load is no more poisonous than the word.
/// Only the bytes up to the last one used: 1 (an i8 load), 2 or 4.
static bool loadBytesApart(Function &F) {
  if (!AGXByteLoads)
    return false;
  using namespace PatternMatch;
  SmallVector<LoadInst *, 8> Loads;
  for (Instruction &I : instructions(F))
    if (auto *L = dyn_cast<LoadInst>(&I);
        L && L->isSimple() &&
        L->getPointerAddressSpace() == AGX::DEVICE_ADDRESS &&
        ((L->getType()->isIntegerTy(32) && L->getAlign() >= 4) ||
         (L->getType()->isIntegerTy(16) && L->getAlign() >= 2)))
      Loads.push_back(L);
  bool Changed = false;
  for (LoadInst *L : Loads) {
    // The bits [Lo, Hi) of the word each value holds, zero-extended above.
    DenseMap<Value *, std::pair<unsigned, unsigned>> Range;
    SmallVector<Instruction *, 16> Work{L};
    SmallVector<std::pair<Instruction *, unsigned>, 8> Bytes;
    unsigned WordBits = L->getType()->getIntegerBitWidth();
    Range[L] = {0, WordBits};
    bool OK = true;
    while (OK && !Work.empty()) {
      Instruction *V = Work.pop_back_val();
      auto [Lo, Hi] = Range[V];
      bool UsedOutside = false;
      for (User *U : V->users()) {
        auto *UI = cast<Instruction>(U);
        unsigned NLo = Lo, NHi = Hi;
        const APInt *C;
        if (isa<TruncInst>(UI))
          NHi = std::min(Hi, Lo + UI->getType()->getScalarSizeInBits());
        else if (isa<ZExtInst>(UI))
          ;
        else if (match(UI, m_LShr(m_Specific(V), m_APInt(C))) &&
                 C->getZExtValue() % 8 == 0)
          NLo = Lo + C->getZExtValue();
        else if (match(UI, m_And(m_Specific(V), m_APInt(C))) && C->isMask())
          NHi = std::min(Hi, Lo + unsigned(C->countr_one()));
        else {
          UsedOutside = true;
          continue;
        }
        if (NLo >= NHi || !UI->getType()->isIntegerTy()) {
          OK = false;
          break;
        }
        auto It = Range.find(UI);
        if (It == Range.end()) {
          Range[UI] = {NLo, NHi};
          Work.push_back(UI);
        } else if (It->second != std::make_pair(NLo, NHi)) {
          OK = false;
        }
      }
      if (!OK)
        break;
      if (!UsedOutside)
        continue;
      // A value used outside the chain must be one byte (not the word), and
      // not sign-extended: signed bytes share one xor of the word instead
      // (E86), which zero-extended halves would lose.
      if (V == L || Hi - Lo != 8 || Lo % 8 ||
          any_of(V->users(), [](const User *U) {
            return isa<SExtInst>(U) || isa<SIToFPInst>(U);
          })) {
        OK = false;
        break;
      }
      Bytes.push_back({V, Lo / 8});
    }
    if (!OK || Bytes.empty())
      continue;
    IRBuilder<> B(L);
    // Only the bytes up to the last one used (1, 2 or 4 from the word's
    // address): the others would take registers for nothing.
    unsigned N = 1;
    for (auto [V, K] : Bytes)
      while (N <= K)
        N *= 2;
    Type *VT = N == 1 ? B.getInt8Ty()
                      : (Type *)FixedVectorType::get(B.getInt8Ty(), N);
    LoadInst *NL = B.CreateAlignedLoad(VT, L->getPointerOperand(),
                                       L->getAlign(), L->getName() + ".bytes");
    NL->setAAMetadata(L->getAAMetadata());
    for (auto [V, K] : Bytes) {
      // The byte's value is used outside the chain: its uses there take
      // the vector element instead.
      B.SetInsertPoint(NL->getNextNode());
      Value *E = B.CreateZExt(N == 1 ? (Value *)NL : B.CreateExtractElement(NL, K),
                              V->getType());
      V->replaceUsesWithIf(E, [&](Use &U) { return !Range.count(U.getUser()); });
    }
    for (auto It = Range.begin(); It != Range.end(); ++It)
      if (auto *I = dyn_cast<Instruction>(It->first); I && I != L)
        I->dropAllReferences();
    for (auto It = Range.begin(); It != Range.end(); ++It)
      if (auto *I = dyn_cast<Instruction>(It->first); I && I != L)
        I->eraseFromParent();
    L->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

/// The values whose low bytes make up V, each at its bit offset, if V is
/// only bytes put together: an or of pieces, zext, shl by whole bytes, a
/// value masked by 255, and a value shifted into the top byte (whose other
/// bits are shifted out).
static bool bytePieces(Value *V, unsigned Shift,
                       SmallVectorImpl<std::pair<Value *, unsigned>> &Pieces,
                       unsigned Depth = 0) {
  using namespace PatternMatch;
  if (Depth > 8)
    return false;
  unsigned W = V->getType()->getScalarSizeInBits();
  Value *A, *B;
  const APInt *C;
  if (match(V, m_Or(m_Value(A), m_APInt(C)))) {
    // A constant's nonzero bytes are pieces of their own (an alpha of 255);
    // its zero bytes are the other operand's.
    for (unsigned K = 0; K < W / 8; ++K) {
      uint64_t Byte = C->extractBitsAsZExtValue(8, 8 * K);
      if (Byte)
        Pieces.push_back(
            {ConstantInt::get(Type::getInt8Ty(V->getContext()), Byte),
             Shift + 8 * K});
    }
    return bytePieces(A, Shift, Pieces, Depth + 1);
  }
  if (match(V, m_Or(m_Value(A), m_Value(B))))
    return bytePieces(A, Shift, Pieces, Depth + 1) &&
           bytePieces(B, Shift, Pieces, Depth + 1);
  if (match(V, m_ZExt(m_Value(A)))) {
    if (A->getType()->getScalarSizeInBits() == 8) {
      Pieces.push_back({A, Shift});
      return true;
    }
    return bytePieces(A, Shift, Pieces, Depth + 1);
  }
  if (match(V, m_Shl(m_Value(A), m_APInt(C))) && C->getZExtValue() % 8 == 0 &&
      C->getZExtValue() < W) {
    unsigned Sh = C->getZExtValue();
    if (Sh == W - 8) {
      Pieces.push_back({A, Shift + Sh});
      return true;
    }
    return bytePieces(A, Shift + Sh, Pieces, Depth + 1);
  }
  if (match(V, m_And(m_Value(A), m_SpecificInt(255)))) {
    Pieces.push_back({A, Shift});
    return true;
  }
  // A value known to fit in its low byte (a clamp to [0, 255]).
  if (isa<Instruction>(V) && W > 8 &&
      computeKnownBits(V, cast<Instruction>(V)->getDataLayout())
              .countMinLeadingZeros() >= W - 8) {
    Pieces.push_back({V, Shift});
    return true;
  }
  return false;
}

static cl::opt<bool> AGXByteStores(
    "agx-byte-stores", cl::init(true), cl::Hidden,
    cl::desc("a device word store of four bytes put together as a store of "
             "the bytes (the byte format, from 16-bit halves)"));

/// A device i32 store of four bytes put together (bytePieces) is a <4 x i8>
/// store of them: the byte format's access stores each 16-bit half's low
/// byte, without the shifts, masks and ors that pack the word.
static bool storeBytesTogether(Function &F) {
  if (!AGXByteStores)
    return false;
  SmallVector<StoreInst *, 8> Stores;
  for (Instruction &I : instructions(F))
    if (auto *St = dyn_cast<StoreInst>(&I);
        St && St->getValueOperand()->getType()->isIntegerTy(32) &&
        St->isSimple() && St->getAlign() >= 4 &&
        St->getPointerAddressSpace() == AGX::DEVICE_ADDRESS)
      Stores.push_back(St);
  bool Changed = false;
  for (StoreInst *St : Stores) {
    SmallVector<std::pair<Value *, unsigned>, 4> Pieces;
    if (!bytePieces(St->getValueOperand(), 0, Pieces) || Pieces.size() != 4)
      continue;
    Value *ByOff[4] = {};
    bool OK = true;
    for (auto [P, Off] : Pieces) {
      if (Off % 8 || Off >= 32 || ByOff[Off / 8]) {
        OK = false;
        break;
      }
      ByOff[Off / 8] = P;
    }
    // (two pieces of one byte: an or of them, not a byte each)
    if (!OK)
      continue;
    // Separate values only (conversions' results, say): bytes of the same
    // word (a byte permute, PTX prmt) are built from its halves already
    // (bit-field inserts), which the halves of a byte store would only move.
    SmallPtrSet<Value *, 4> Roots;
    for (Value *P : ByOff) {
      Value *R = P;
      while (isa<TruncInst>(R) || isa<ZExtInst>(R) ||
             (isa<BinaryOperator>(R) &&
              (cast<BinaryOperator>(R)->getOpcode() == Instruction::LShr ||
               cast<BinaryOperator>(R)->getOpcode() == Instruction::AShr ||
               cast<BinaryOperator>(R)->getOpcode() == Instruction::And) &&
              isa<Constant>(cast<BinaryOperator>(R)->getOperand(1))))
        R = cast<Instruction>(R)->getOperand(0);
      OK &= Roots.insert(R).second;
    }
    if (!OK)
      continue;
    IRBuilder<> B(St);
    Value *Vec = PoisonValue::get(FixedVectorType::get(B.getInt8Ty(), 4));
    for (unsigned K = 0; K < 4; ++K)
      Vec = B.CreateInsertElement(Vec, B.CreateTrunc(ByOff[K], B.getInt8Ty()), K);
    StoreInst *NS = B.CreateAlignedStore(Vec, St->getPointerOperand(),
                                         St->getAlign());
    NS->setAAMetadata(St->getAAMetadata());
    Value *Old = St->getValueOperand();
    St->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(Old);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXSinkMulExtensions(
    "agx-sink-mul-extensions", cl::init(true), cl::Hidden,
    cl::desc("a 64-bit multiply's operand that is a 32-bit value extended "
             "(and shifted) in another block, extended again next to it"));

/// A 64-bit multiply whose operand is a sext / zext of a 32-bit value, or
/// such a value shifted left by a constant, computed in another block (a
/// loop-invariant stride LICM hoisted): the extension (and shift) are made
/// again next to the multiply, so that instruction selection, which sees
/// one block, makes it a 32 x 32 -> 64-bit multiply (and the shift part of
/// the address addition) instead of the four-multiply 64-bit product
/// (index_add's row offsets).
static bool sinkMulExtensions(Function &F) {
  if (!AGXSinkMulExtensions)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    if (I.getOpcode() != Instruction::Mul || !I.getType()->isIntegerTy(64))
      continue;
    // The extension (and a shift of it) of a 32-bit value, and its kind.
    auto ExtOf = [](Value *V, const APInt *&C) -> Instruction * {
      using namespace PatternMatch;
      C = nullptr;
      Value *X;
      if (!match(V, m_Shl(m_Value(X), m_APInt(C))))
        X = V;
      auto *E = dyn_cast<CastInst>(X);
      if (!E || !isa<SExtInst, ZExtInst>(E) ||
          !E->getSrcTy()->isIntegerTy(32))
        return nullptr;
      return E;
    };
    // Only a multiply of two extensions of one kind is the 32 x 32 -> 64-bit
    // multiply the cloned extension lets the DAG see.
    const APInt *C0, *C1;
    Instruction *E0 = ExtOf(I.getOperand(0), C0);
    Instruction *E1 = ExtOf(I.getOperand(1), C1);
    if (!E0 || !E1 || E0->getOpcode() != E1->getOpcode())
      continue;
    for (Use &U : I.operands()) {
      auto *Op = dyn_cast<Instruction>(U.get());
      if (!Op || Op->getParent() == I.getParent())
        continue;
      const APInt *C;
      ExtOf(Op, C);
      IRBuilder<> B(&I);
      Instruction *Ext = C ? cast<Instruction>(Op->getOperand(0)) : Op;
      Instruction *NewExt = Ext->clone();
      B.Insert(NewExt);
      Value *New = NewExt;
      if (C) {
        Instruction *NewShl = Op->clone();
        NewShl->setOperand(0, NewExt);
        B.Insert(NewShl);
        New = NewShl;
      }
      U.set(New);
      Changed = true;
    }
  }
  return Changed;
}

static cl::opt<bool> AGXNonNegXorDivisions(
    "agx-nonneg-xor-divisions", cl::init(true), cl::Hidden,
    cl::desc("sdiv of x ^ (x >>s (w - 1)) by a positive constant as the "
             "udiv of its low w - 1 bits"));

/// sdiv (x ^ (x >>s (w - 1))), c for c > 0 (a floor division's dividend,
/// ~x for negative x, as opt makes MLIR's affine floordiv): the dividend's
/// sign bit is 0, so it is udiv (v & (2^(w-1) - 1)), c, whose known zero top
/// bit lets the unsigned magic multiply go without the signed quotient's
/// rounding correction (one and against the correction's three) (L15-E292).
static bool nonNegXorDivisions(Function &F) {
  if (!AGXNonNegXorDivisions)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    Value *V, *X;
    const APInt *C, *Sh;
    if (!match(&I, m_SDiv(m_Value(V), m_APInt(C))) || !C->isStrictlyPositive() ||
        C->isPowerOf2() || !I.getType()->isIntegerTy(32))
      continue;
    unsigned W = I.getType()->getScalarSizeInBits();
    if (!match(V, m_c_Xor(m_AShr(m_Value(X), m_APInt(Sh)), m_Deferred(X))) ||
        *Sh != W - 1)
      continue;
    IRBuilder<> B(&I);
    Value *M = B.CreateAnd(V, APInt::getSignedMaxValue(W));
    Value *R = B.CreateUDiv(M, I.getOperand(1));
    R->takeName(&I);
    I.replaceAllUsesWith(R);
    I.eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXFloorModulos(
    "agx-floor-modulos", cl::init(true), cl::Hidden,
    cl::desc("the floor modulo of srem by a positive constant through the "
             "unsigned floor division"));

/// r < 0 ? r + c : r for r = srem x, c (c > 0 not a power of two; MLIR's
/// affine mod): x - c * floor(x / c), the floor quotient s ^ ((x ^ s) / c)
/// with the non-negative x ^ s divided unsigned (E292), x frozen: the
/// unsigned magic multiply instead of the signed one's correction, and no
/// select (L15-E294).
static bool floorModulos(Function &F) {
  if (!AGXFloorModulos)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    Value *R, *X;
    const APInt *C, *C2;
    if (!I.getType()->isIntegerTy(32) ||
        !match(&I, m_Select(m_SpecificICmp(ICmpInst::ICMP_SLT, m_Value(R),
                                           m_Zero()),
                            m_Add(m_Deferred(R), m_APInt(C2)),
                            m_Deferred(R))) ||
        !match(R, m_SRem(m_Value(X), m_APInt(C))) || *C != *C2 ||
        !C->isStrictlyPositive() || C->isPowerOf2())
      continue;
    IRBuilder<> B(&I);
    Value *Xf = B.CreateFreeze(X);
    Value *S = B.CreateAShr(Xf, 31);
    Value *Q = B.CreateUDiv(
        B.CreateAnd(B.CreateXor(Xf, S), APInt::getSignedMaxValue(32)),
        ConstantInt::get(I.getType(), *C));
    Value *M = B.CreateSub(
        Xf, B.CreateMul(B.CreateXor(Q, S), ConstantInt::get(I.getType(), *C)));
    M->takeName(&I);
    I.replaceAllUsesWith(M);
    I.eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(R);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXRangedCompares(
    "agx-ranged-compares", cl::init(true), cl::Hidden,
    cl::desc("integer compares that the ranges of a lane or thread index "
             "decide, folded"));

/// The range of V through additions, subtractions and masks by constants,
/// from the ranges ValueTracking gives its leaves (a SIMD lane's [0, 32)).
static cl::opt<bool> AGXRangedComparesBounds(
    "agx-ranged-compares-bounds", cl::init(true), cl::Hidden,
    cl::desc("Ranged compares with the positions' upper bounds and known "
             "bits"));

static ConstantRange rangeThroughOffsets(Value *V, const SimplifyQuery &SQ,
                                         unsigned Depth = 0,
                                         bool KnownBits = false) {
  using namespace PatternMatch;
  unsigned W = V->getType()->getScalarSizeInBits();
  const APInt *C;
  Value *X;
  if (Depth < 6) {
    if (match(V, m_Add(m_Value(X), m_APInt(C))))
      return rangeThroughOffsets(X, SQ, Depth + 1, KnownBits).add(ConstantRange(*C));
    if (match(V, m_Sub(m_Value(X), m_APInt(C))))
      return rangeThroughOffsets(X, SQ, Depth + 1, KnownBits).sub(ConstantRange(*C));
    if (match(V, m_Sub(m_APInt(C), m_Value(X))))
      return ConstantRange(*C).sub(rangeThroughOffsets(X, SQ, Depth + 1, KnownBits));
    if (match(V, m_And(m_Value(X), m_APInt(C))))
      return rangeThroughOffsets(X, SQ, Depth + 1, KnownBits)
          .binaryAnd(ConstantRange(*C));
    if (match(V, m_Or(m_Value(X), m_APInt(C))))
      return rangeThroughOffsets(X, SQ, Depth + 1, KnownBits).binaryOr(ConstantRange(*C));
    if (match(V, m_Xor(m_Value(X), m_APInt(C))))
      return rangeThroughOffsets(X, SQ, Depth + 1, KnownBits)
          .binaryXor(ConstantRange(*C));
  }
  if (auto *CI = dyn_cast<ConstantInt>(V))
    return ConstantRange(CI->getValue());
  if (!V->getType()->isIntegerTy())
    return ConstantRange::getFull(W);
  ConstantRange R =
      computeConstantRange(V, /*ForSigned=*/false, SQ)
          .intersectWith(computeConstantRange(V, /*ForSigned=*/true, SQ));
  // (and its known bits: the or of two bounded positions, a padded
  // stencil's (x' | y') <u W, whose range alone is full: L15-E491)
  if (KnownBits)
    R = R.intersectWith(ConstantRange::fromKnownBits(
        computeKnownBits(V, SQ), /*IsSigned=*/false));
  return R;
}

/// An integer compare of such a value against a constant (a shuffle's
/// segment test lane - k < 32 once the offset k is a constant) that its
/// range decides: the constant (L15-E300).
static bool rangedCompares(Function &F) {
  if (!AGXRangedCompares)
    return false;
  using namespace PatternMatch;
  SimplifyQuery SQ(F.getDataLayout());
  // The special registers' ranges for the analysis only (taken off again,
  // as after straight-line strength reduction: loop strength reduction makes
  // 64-bit offsets with them).
  SmallVector<WeakVH, 8> Ranged;
  // (with the kernel's upper bounds of positions and the grid's, as for
  // any analysis whose ranges are taken off again: L15-E491)
  AGX::addSpecialRegisterRanges(F, &Ranged,
                                /*ForAnalysis=*/AGXRangedComparesBounds);
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *Cmp = dyn_cast<ICmpInst>(&I);
    if (!Cmp || !Cmp->getOperand(0)->getType()->isIntegerTy() ||
        (isa<Constant>(Cmp->getOperand(0)) &&
         isa<Constant>(Cmp->getOperand(1))))
      continue;
    SimplifyQuery Q = SQ.getWithInstruction(Cmp);
    ConstantRange R = rangeThroughOffsets(Cmp->getOperand(0), Q);
    ConstantRange K = rangeThroughOffsets(Cmp->getOperand(1), Q);
    std::optional<bool> Known;
    // (deciding it with known bits too: the or of two bounded positions;
    // not for the rebasing below, whose x < c - k for a loop's x + k < c
    // kept two congruent counters apart, L15-E431)
    if (AGXRangedComparesBounds) {
      ConstantRange RB = rangeThroughOffsets(Cmp->getOperand(0), Q, 0, true);
      ConstantRange KB = rangeThroughOffsets(Cmp->getOperand(1), Q, 0, true);
      if (!RB.isFullSet() && !KB.isFullSet()) {
        if (RB.icmp(Cmp->getPredicate(), KB))
          Known = true;
        else if (RB.icmp(Cmp->getInversePredicate(), KB))
          Known = false;
      }
    }
    if (!Known && (R.isFullSet() || K.isFullSet()))
      continue;
    if (Known) {
    } else if (R.icmp(Cmp->getPredicate(), K))
      Known = true;
    else if (R.icmp(Cmp->getInversePredicate(), K))
      Known = false;
    if (!Known) {
      // x + k against c where x's range keeps the addition from wrapping in
      // the compare's signedness: x against c - k (lane - k < 0 is lane < k,
      // which the SIMD scan recognition takes).
      Value *X;
      const APInt *KcP;
      const APInt *Cc = K.getSingleElement();
      APInt KcV;
      if (match(Cmp->getOperand(0), m_Add(m_Value(X), m_APInt(KcP))))
        KcV = *KcP;
      else if (match(Cmp->getOperand(0), m_Sub(m_Value(X), m_APInt(KcP))))
        KcV = -*KcP;
      else
        continue;
      const APInt *Kc = &KcV;
      if (!Cc || Cmp->isEquality())
        continue;
      ConstantRange RX = rangeThroughOffsets(X, Q);
      bool Signed = Cmp->isSigned();
      bool Overflow;
      APInt NewC = Signed ? Cc->ssub_ov(*Kc, Overflow)
                          : Cc->usub_ov(*Kc, Overflow);
      if (Overflow ||
          (Signed ? RX.signedAddMayOverflow(ConstantRange(*Kc))
                  : RX.unsignedAddMayOverflow(ConstantRange(*Kc))) !=
              ConstantRange::OverflowResult::NeverOverflows)
        continue;
      Cmp->setOperand(0, X);
      Cmp->setOperand(1, ConstantInt::get(X->getType(), NewC));
      // (signed and unsigned agree on a non-negative x and c - k: unsigned,
      // the form the scan recognition matches)
      if (Signed && RX.isAllNonNegative() && NewC.isNonNegative())
        Cmp->setPredicate(Cmp->getUnsignedPredicate());
      Changed = true;
      continue;
    }
    SmallSetVector<Instruction *, 4> Users;
    for (User *U : Cmp->users())
      if (auto *UI = dyn_cast<Instruction>(U))
        Users.insert(UI);
    Cmp->replaceAllUsesWith(ConstantInt::getBool(Cmp->getType(), *Known));
    // (the and / or of the other bound with the constant: that bound; the
    // dead instructions are left to the later cleanups)
    for (Instruction *UI : Users)
      if (Value *V = simplifyInstruction(UI, Q))
        UI->replaceAllUsesWith(V);
    Changed = true;
  }
  for (WeakVH &V : Ranged)
    if (auto *CI = dyn_cast_or_null<CallBase>(V))
      CI->removeRetAttr(Attribute::Range);
  return Changed;
}

static cl::opt<bool> AGXOwnValueShifts(
    "agx-own-value-shifts", cl::init(true), cl::Hidden,
    cl::desc("a select of the own value where a SIMD-group shift by a "
             "constant has no source lane, dropped"));

/// select(lane < 32 - k, shuffle_down(v, k), v) and
/// select(lane >= k, shuffle_up(v, k), v), k a constant in [1, 32): AIR's
/// shifts leave the lanes without a source their own value (as
/// convert-gpu-to-air takes them for shifts by constants), so the select
/// is the shuffle (a loop-written reduction's step once unrolled)
/// (L15-E302).
static bool ownValueShifts(Function &F) {
  if (!AGXOwnValueShifts)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    Value *Cond, *T, *Fv;
    if (!match(&I, m_Select(m_Value(Cond), m_Value(T), m_Value(Fv))))
      continue;
    // (through the bitcasts of a float's shuffle as i32)
    auto Peel = [](Value *V) {
      while (auto *BC = dyn_cast<BitCastInst>(V))
        V = BC->getOperand(0);
      return V;
    };
    bool Swapped = false;
    auto *Sh = dyn_cast<IntrinsicInst>(Peel(T));
    if (!Sh || (Sh->getIntrinsicID() != Intrinsic::agx_simd_shuffle_down &&
                Sh->getIntrinsicID() != Intrinsic::agx_simd_shuffle_up)) {
      Sh = dyn_cast<IntrinsicInst>(Peel(Fv));
      std::swap(T, Fv);
      Swapped = true;
    }
    if (!Sh || (Sh->getIntrinsicID() != Intrinsic::agx_simd_shuffle_down &&
                Sh->getIntrinsicID() != Intrinsic::agx_simd_shuffle_up) ||
        Peel(Sh->getArgOperand(0)) != Peel(Fv))
      continue;
    const APInt *K, *C;
    CmpPredicate P;
    Value *L;
    if (!match(Sh->getArgOperand(1), m_APInt(K)) || K->uge(32) || K->isZero() ||
        !match(Cond, m_ICmp(P, m_Value(L), m_APInt(C))) ||
        !match(L, m_Intrinsic<Intrinsic::agx_thread_index_in_simdgroup>()))
      continue;
    ICmpInst::Predicate Pr =
        Swapped ? ICmpInst::getInversePredicate(P) : ICmpInst::Predicate(P);
    uint64_t k = K->getZExtValue(), c = C->getZExtValue();
    bool Down = Sh->getIntrinsicID() == Intrinsic::agx_simd_shuffle_down;
    // the condition is exactly "the source lane exists"
    bool Exact = Down ? (Pr == ICmpInst::ICMP_ULT && c == 32 - k) ||
                            (Pr == ICmpInst::ICMP_ULE && c == 31 - k)
                      : (Pr == ICmpInst::ICMP_UGE && c == k) ||
                            (Pr == ICmpInst::ICMP_UGT && c == k - 1);
    if (!Exact)
      continue;
    I.replaceAllUsesWith(T);
    I.eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXQuadXorShuffles(
    "agx-quad-xor-shuffles", cl::init(true), cl::Hidden,
    cl::desc("a SIMD-group xor shuffle by 1, 2 or 3 as the quad-group's"));

/// simd_shuffle_xor(v, m) for m in {1, 2, 3}: lane ^ m is in the lane's
/// quad, so it is quad_shuffle_xor(v, m), which issues in half the cycles
/// (the M2 Max coverage recordings: 1.04 against 2.0 per step, latency 3.9
/// against 4.8): a butterfly's last two steps, a 2-D 2 x 2 exchange
/// (L15-E304).
static bool quadXorShuffles(Function &F) {
  if (!AGXQuadXorShuffles)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    const APInt *M;
    if (!II || II->getIntrinsicID() != Intrinsic::agx_simd_shuffle_xor ||
        !match(II->getArgOperand(1), m_APInt(M)) || M->isZero() ||
        M->ugt(3))
      continue;
    // (the quad shuffle's selected types: 16- and 32-bit scalars; a wider
    // value's shuffle is split in legalization, which the quad one is not)
    Type *Ty = II->getType();
    if (Ty->isVectorTy() || (Ty->getPrimitiveSizeInBits() != 16 &&
                             Ty->getPrimitiveSizeInBits() != 32))
      continue;
    II->setCalledFunction(Intrinsic::getOrInsertDeclaration(
        F.getParent(), Intrinsic::agx_quad_shuffle_xor, {II->getType()}));
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXQuadShifts(
    "agx-quad-shifts", cl::init(true), cl::Hidden,
    cl::desc("a SIMD-group shift by a constant within segments of 2 or 4 "
             "lanes as the quad-group's"));

/// The values of an integer expression of the SIMD lane alone at each of the
/// 32 lanes, through the constant folding of its instructions, or none where
/// it reads anything else.
static std::optional<std::array<uint64_t, 32>>
laneValues(Value *V, const DataLayout &DL) {
  using namespace PatternMatch;
  std::function<Constant *(Value *, unsigned, unsigned)> Eval =
      [&](Value *V, unsigned L, unsigned Depth) -> Constant * {
    if (auto *C = dyn_cast<Constant>(V))
      return C;
    if (match(V, m_Intrinsic<Intrinsic::agx_thread_index_in_simdgroup>()))
      return ConstantInt::get(V->getType(), L);
    auto *I = dyn_cast<Instruction>(V);
    if (!I || Depth > 8 ||
        !(isa<BinaryOperator, CmpInst, CastInst, SelectInst>(I)))
      return nullptr;
    SmallVector<Constant *, 3> Ops;
    for (Value *Op : I->operands()) {
      Constant *C = Eval(Op, L, Depth + 1);
      if (!C)
        return nullptr;
      Ops.push_back(C);
    }
    return ConstantFoldInstOperands(I, Ops, DL);
  };
  std::array<uint64_t, 32> Values;
  for (unsigned L = 0; L < 32; ++L) {
    auto *C = dyn_cast_or_null<ConstantInt>(Eval(V, L, 0));
    if (!C || C->getBitWidth() > 64)
      return std::nullopt;
    Values[L] = C->getZExtValue();
  }
  return Values;
}

/// The value of a condition of the SIMD lane alone at each of the 32 lanes
/// (bit L: at lane L), or none.
static std::optional<uint32_t> laneTruth(Value *Cond, const DataLayout &DL) {
  std::optional<std::array<uint64_t, 32>> Values = laneValues(Cond, DL);
  if (!Values)
    return std::nullopt;
  uint32_t Mask = 0;
  for (unsigned L = 0; L < 32; ++L)
    if ((*Values)[L] & 1)
      Mask |= 1u << L;
  return Mask;
}

/// A simd_shuffle_down / up by a constant k in [1, 4) whose every use
/// (through bitcasts and truncations) is a select of it and the own value
/// on a condition of the lane that takes it only where lane % 4 + k < 4
/// (down) or lane % 4 >= k (up), as convert-gpu-to-air's gpu.shuffle down /
/// up within 2 or 4 lanes: it reads a lane of the thread's quad wherever it
/// is taken, so it is the quad-group shift, which issues in half the cycles
/// (the M2 Max coverage recordings, as quadXorShuffles). A select that
/// takes it exactly there goes: the quad shift keeps the own value where
/// its source is past the quad (the masked hardware recordings)
/// (L15-E305). The conditions are compared by their values at the 32 lanes,
/// whatever their form (L15-E306).
static bool quadShifts(Function &F) {
  if (!AGXQuadShifts)
    return false;
  const DataLayout &DL = F.getDataLayout();
  using namespace PatternMatch;
  // (through the casts of a 16-bit or float value's shuffle as i32)
  auto Peel = [](Value *V) {
    while (isa<BitCastInst, TruncInst, ZExtInst, SExtInst>(V))
      V = cast<Instruction>(V)->getOperand(0);
    return V;
  };
  bool Changed = false;
  for (Instruction &Inst : instructions(F)) {
    auto *Sh = dyn_cast<IntrinsicInst>(&Inst);
    if (!Sh || (Sh->getIntrinsicID() != Intrinsic::agx_simd_shuffle_down &&
                Sh->getIntrinsicID() != Intrinsic::agx_simd_shuffle_up))
      continue;
    bool Down = Sh->getIntrinsicID() == Intrinsic::agx_simd_shuffle_down;
    const APInt *K;
    Type *Ty = Sh->getType();
    if (!match(Sh->getArgOperand(1), m_APInt(K)) || K->isZero() ||
        K->ugt(3) || Ty->isVectorTy() ||
        (Ty->getPrimitiveSizeInBits() != 16 &&
         Ty->getPrimitiveSizeInBits() != 32))
      continue;
    unsigned k = K->getZExtValue();
    // the lanes whose quad shift has a source lane
    uint32_t Quad = 0;
    for (unsigned L = 0; L < 32; ++L)
      if (Down ? (L & 3) + k < 4 : (L & 3) >= k)
        Quad |= 1u << L;
    Value *Own = Peel(Sh->getArgOperand(0));
    SmallVector<Value *, 4> Work{Sh};
    SmallVector<std::pair<SelectInst *, Value *>, 4> Exact;
    bool OK = true;
    while (OK && !Work.empty()) {
      Value *V = Work.pop_back_val();
      for (User *U : V->users()) {
        if (isa<BitCastInst, TruncInst>(U)) {
          Work.push_back(U);
          continue;
        }
        auto *Sel = dyn_cast<SelectInst>(U);
        bool OnTrue = Sel && Sel->getTrueValue() == V;
        Value *OwnSide =
            Sel ? (OnTrue ? Sel->getFalseValue() : Sel->getTrueValue())
                : nullptr;
        std::optional<uint32_t> T =
            Sel && OwnSide != V && Peel(OwnSide) == Own
                ? laneTruth(Sel->getCondition(), DL)
                : std::nullopt;
        if (!T) {
          OK = false;
          break;
        }
        uint32_t Takes = OnTrue ? *T : ~*T;
        if (Takes & ~Quad) {
          OK = false;
          break;
        }
        if (Takes == Quad)
          Exact.push_back({Sel, V});
      }
    }
    if (!OK)
      continue;
    Sh->setCalledFunction(Intrinsic::getOrInsertDeclaration(
        F.getParent(),
        Down ? Intrinsic::agx_quad_shuffle_down : Intrinsic::agx_quad_shuffle_up,
        {Ty}));
    for (auto [Sel, V] : Exact) {
      Sel->replaceAllUsesWith(V);
      Sel->eraseFromParent();
    }
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXLanePermutations(
    "agx-lane-permutations", cl::init(true), cl::Hidden,
    cl::desc("a tree of shuffles of one value and selects on the lane as "
             "the one shuffle whose source lanes it reads"));

namespace {
/// The source lane of each of the 32 lanes.
using LaneTable = std::array<uint8_t, 32>;

/// The shuffles of a value by a constant (or, for an index shuffle, an index
/// of the lane), and the selects of them on conditions of the lane, as the
/// lane each lane reads (L15-E307).
struct LanePermutation {
  const DataLayout &DL;
  Value *Arg = nullptr;      // the shuffled value
  Type *RootTy = nullptr;    // the selects' type
  SmallVector<std::pair<unsigned, Type *>, 2> Chain; // the shuffles' casts
  bool HasChain = false;
  unsigned Cost = 0; // shuffles' issue (a SIMD-group's 2, a quad's 1) and
                     // selects
  SmallVector<Instruction *, 8> Nodes;

  explicit LanePermutation(const DataLayout &DL) : DL(DL) {}

  static bool isShuffle(Intrinsic::ID ID) {
    switch (ID) {
    case Intrinsic::agx_simd_shuffle:
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_simd_shuffle_rotate_up:
    case Intrinsic::agx_quad_shuffle:
    case Intrinsic::agx_quad_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_up:
    case Intrinsic::agx_quad_shuffle_down:
    case Intrinsic::agx_quad_shuffle_rotate_up:
      return true;
    default:
      return false;
    }
  }

  /// The lane read at lane L by shuffle ID with index K.
  static unsigned source(Intrinsic::ID ID, unsigned L, uint64_t K) {
    unsigned Q = L & ~3u, I = L & 3;
    switch (ID) {
    case Intrinsic::agx_simd_shuffle:
      return K;
    case Intrinsic::agx_simd_shuffle_xor:
      return (L ^ K) < 32 ? L ^ K : L;
    case Intrinsic::agx_simd_shuffle_up:
      return L >= K ? L - K : L;
    case Intrinsic::agx_simd_shuffle_down:
      return L + K < 32 ? L + K : L;
    case Intrinsic::agx_simd_shuffle_rotate_up:
      return (L - K) & 31;
    case Intrinsic::agx_quad_shuffle:
      return K < 4 ? Q | K : L;
    case Intrinsic::agx_quad_shuffle_xor:
      return (I ^ K) < 4 ? Q | (I ^ K) : L;
    case Intrinsic::agx_quad_shuffle_up:
      return I >= K ? L - K : L;
    case Intrinsic::agx_quad_shuffle_down:
      return I + K < 4 ? L + K : L;
    case Intrinsic::agx_quad_shuffle_rotate_up:
      return Q | ((I - K) & 3);
    default:
      llvm_unreachable("not a shuffle");
    }
  }

  static bool isQuad(Intrinsic::ID ID) {
    return ID == Intrinsic::agx_quad_shuffle ||
           ID == Intrinsic::agx_quad_shuffle_xor ||
           ID == Intrinsic::agx_quad_shuffle_up ||
           ID == Intrinsic::agx_quad_shuffle_down ||
           ID == Intrinsic::agx_quad_shuffle_rotate_up;
  }

  /// The table of V, a node of the tree (its only use the tree's unless it
  /// is the root).
  /// The value the tree shuffles: its first shuffle's.
  static Value *shuffled(Value *V, unsigned Depth = 0) {
    if (Depth > 8)
      return nullptr;
    if (auto *Sel = dyn_cast<SelectInst>(V)) {
      if (Value *A = shuffled(Sel->getTrueValue(), Depth + 1))
        return A;
      return shuffled(Sel->getFalseValue(), Depth + 1);
    }
    while (isa<BitCastInst, TruncInst>(V))
      V = cast<Instruction>(V)->getOperand(0);
    auto *Sh = dyn_cast<IntrinsicInst>(V);
    if (!Sh || !isShuffle(Sh->getIntrinsicID()))
      return nullptr;
    return Sh->getArgOperand(0);
  }

  std::optional<LaneTable> table(Value *V, bool Root, BasicBlock *BB,
                                 unsigned Depth = 0) {
    if ((!Root && !V->hasOneUse()) || Depth > 8)
      return std::nullopt;
    // (all in the root's block: the lanes the shuffles read are active)
    auto InBlock = [&](Value *X) {
      auto *I = dyn_cast<Instruction>(X);
      return I && I->getParent() == BB;
    };
    if (auto *Sel = dyn_cast<SelectInst>(V)) {
      if (Sel->getType() != RootTy)
        return std::nullopt;
      std::optional<uint32_t> T = laneTruth(Sel->getCondition(), DL);
      if (!T)
        return std::nullopt;
      if (!InBlock(Sel))
        return std::nullopt;
      std::optional<LaneTable> A =
          table(Sel->getTrueValue(), false, BB, Depth + 1);
      if (!A)
        return std::nullopt;
      std::optional<LaneTable> B =
          table(Sel->getFalseValue(), false, BB, Depth + 1);
      if (!B)
        return std::nullopt;
      LaneTable R;
      for (unsigned L = 0; L < 32; ++L)
        R[L] = (*T >> L) & 1 ? (*A)[L] : (*B)[L];
      Nodes.push_back(Sel);
      Cost += 1;
      return R;
    }
    // a shuffle through its casts
    SmallVector<std::pair<unsigned, Type *>, 2> Casts;
    SmallVector<Instruction *, 2> CastInsts;
    Value *P = V;
    while (isa<BitCastInst, TruncInst>(P)) {
      auto *CI = cast<Instruction>(P);
      if (!InBlock(CI))
        return std::nullopt;
      Casts.push_back({CI->getOpcode(), CI->getType()});
      CastInsts.push_back(CI);
      P = CI->getOperand(0);
      if (!P->hasOneUse())
        return std::nullopt;
    }
    auto *Sh = dyn_cast<IntrinsicInst>(P);
    if (!Sh || !isShuffle(Sh->getIntrinsicID())) {
      // the own value: the shuffled value without its widening casts
      if (Root || !Arg || V->getType() != RootTy)
        return std::nullopt;
      Value *A = Arg;
      while (isa<BitCastInst, ZExtInst, SExtInst>(A) && A != V)
        A = cast<Instruction>(A)->getOperand(0);
      if (A != V)
        return std::nullopt;
      LaneTable R;
      for (unsigned L = 0; L < 32; ++L)
        R[L] = L;
      return R;
    }
    std::reverse(Casts.begin(), Casts.end());
    if (Sh->getArgOperand(0) != Arg || !InBlock(Sh))
      return std::nullopt;
    if (!HasChain) {
      Chain = Casts;
      HasChain = true;
    } else if (Chain != Casts) {
      return std::nullopt;
    }
    Intrinsic::ID ID = Sh->getIntrinsicID();
    Value *Idx = Sh->getArgOperand(1);
    LaneTable R;
    if (auto *C = dyn_cast<ConstantInt>(Idx)) {
      uint64_t K = C->getZExtValue();
      if (K >= 32)
        return std::nullopt;
      for (unsigned L = 0; L < 32; ++L)
        R[L] = source(ID, L, K);
      Cost += isQuad(ID) ? 1 : 2;
    } else if (ID == Intrinsic::agx_simd_shuffle ||
               ID == Intrinsic::agx_quad_shuffle) {
      // an index of the lane: a shuffle per position in the quad
      std::optional<std::array<uint64_t, 32>> K = laneValues(Idx, DL);
      if (!K)
        return std::nullopt;
      for (unsigned L = 0; L < 32; ++L) {
        if ((*K)[L] >= 32)
          return std::nullopt;
        R[L] = source(ID, L, (*K)[L]);
      }
      Cost += isQuad(ID) ? 4 : 8;
    } else {
      return std::nullopt;
    }
    Nodes.push_back(Sh);
    Nodes.append(CastInsts.begin(), CastInsts.end());
    return R;
  }
};
} // namespace

/// A tree of shuffles of one value (by constants, or by an index of the
/// lane) and selects of them on conditions of the lane reads, at each lane,
/// a lane of the value given by a table of the 32 lanes. Where that table
/// is one shuffle's (a quad-group's first: xor, shifts, rotate, a lane of
/// the quad; then the SIMD-group's: a lane, xor, shifts, rotate) and the
/// shuffle issues in fewer cycles than the tree, the tree is that shuffle:
/// gpu.rotate's shuffle down and up and their select (L15-E298) are the
/// SIMD-group or quad-group rotate, an index shuffle of (lane + c) & 31 the
/// rotate, an index shuffle of lane ^ 1 the quad xor (L15-E307).
static bool lanePermutations(Function &F) {
  if (!AGXLanePermutations)
    return false;
  const DataLayout &DL = F.getDataLayout();
  SmallVector<WeakTrackingVH, 16> Roots;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (isa<SelectInst>(I) ||
        (II && LanePermutation::isShuffle(II->getIntrinsicID())))
      Roots.push_back(&I);
  }
  bool Changed = false;
  // (the outermost first)
  for (WeakTrackingVH &H : reverse(Roots)) {
    auto *Root = dyn_cast_or_null<Instruction>(H);
    if (!Root)
      continue;
    LanePermutation P(DL);
    P.RootTy = Root->getType();
    P.Arg = LanePermutation::shuffled(Root);
    if (!P.Arg)
      continue;
    Type *Ty = P.Arg->getType();
    if (Ty->isVectorTy() || (Ty->getPrimitiveSizeInBits() != 16 &&
                             Ty->getPrimitiveSizeInBits() != 32))
      continue;
    std::optional<LaneTable> T = P.table(Root, true, Root->getParent());
    if (!T)
      continue;
    // (one shuffle by a constant is quadXorShuffles' and quadShifts', but
    // for one that reads the own lane)
    auto *First = dyn_cast<IntrinsicInst>(P.Nodes.front());
    bool Lone = P.Nodes.size() == P.Chain.size() + 1 && First &&
                isa<ConstantInt>(First->getArgOperand(1));
    // (the shuffle's casts lead to the root's type)
    Type *OutTy = P.Chain.empty() ? Ty : P.Chain.back().second;
    if (OutTy != Root->getType())
      continue;
    // the cheapest one shuffle with this table
    struct Cand {
      Intrinsic::ID ID;
      unsigned K;
    };
    SmallVector<Cand, 8> Cands;
    for (Intrinsic::ID ID :
         {Intrinsic::agx_quad_shuffle_xor, Intrinsic::agx_quad_shuffle_down,
          Intrinsic::agx_quad_shuffle_up,
          Intrinsic::agx_quad_shuffle_rotate_up, Intrinsic::agx_quad_shuffle})
      for (unsigned K = 0; K < 4; ++K)
        Cands.push_back({ID, K});
    for (Intrinsic::ID ID :
         {Intrinsic::agx_simd_shuffle, Intrinsic::agx_simd_shuffle_xor,
          Intrinsic::agx_simd_shuffle_down, Intrinsic::agx_simd_shuffle_up,
          Intrinsic::agx_simd_shuffle_rotate_up})
      for (unsigned K = 0; K < 32; ++K)
        Cands.push_back({ID, K});
    bool Identity = true;
    for (unsigned L = 0; L < 32; ++L)
      Identity &= (*T)[L] == L;
    std::optional<Cand> Found;
    if (!Identity)
      for (const Cand &C : Cands) {
        bool Same = true;
        for (unsigned L = 0; L < 32 && Same; ++L)
          Same = LanePermutation::source(C.ID, L, C.K) == (*T)[L];
        if (Same) {
          Found = C;
          break;
        }
      }
    if ((!Identity && !Found) || (Lone && !Identity))
      continue;
    unsigned NewCost =
        Identity ? 0 : (LanePermutation::isQuad(Found->ID) ? 1 : 2);
    if (NewCost >= P.Cost)
      continue;
    IRBuilder<> B(Root);
    Value *New = P.Arg;
    if (!Identity)
      New = B.CreateIntrinsic(Found->ID, {Ty}, {P.Arg, B.getInt32(Found->K)});
    for (auto [Op, CastTy] : P.Chain)
      New = B.CreateCast(Instruction::CastOps(Op), New, CastTy);
    New->takeName(Root);
    Root->replaceAllUsesWith(New);
    RecursivelyDeleteTriviallyDeadInstructions(Root);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXConvertedMinMaxNum(
    "agx-converted-minmax-num", cl::init(true), cl::Hidden,
    cl::desc("maximum / minimum whose only use is a float-to-integer "
             "conversion (or another such) as maxnum / minnum"));

/// fptosi / fptoui (maximum(a, b)) is fpto[su]i (maxnum(a, b)), and minimum
/// likewise, through a chain of single-use maximums and minimums: where no
/// operand is a NaN they are the same up to a zero's sign, which converts to
/// 0 either way, and where one is, the maximum's NaN made the conversion
/// poison. The NaN select of the maximum goes, and satConvertClamps sees the
/// maxnum / minnum clamps it takes (L15-E281). So does a select of a NaN
/// constant into the conversion.
static bool convertedMinMaxNum(Function &F) {
  if (!AGXConvertedMinMaxNum)
    return false;
  using namespace PatternMatch;
  SmallVector<Instruction *, 8> Work;
  for (Instruction &I : instructions(F))
    if (isa<FPToSIInst>(I) || isa<FPToUIInst>(I))
      Work.push_back(&I);
  bool Changed = false;
  while (!Work.empty()) {
    Instruction *User = Work.pop_back_val();
    // A select of a NaN constant into the conversion (MLIR's maximumf: a
    // maxnum and a select of a NaN where an operand is one): the other arm,
    // the NaN's conversion being poison.
    if (isa<FPToSIInst>(User) || isa<FPToUIInst>(User)) {
      const APFloat *K;
      Value *V;
      while (true) {
        Value *X = User->getOperand(0);
        if (match(X, m_Select(m_Value(), m_APFloat(K), m_Value(V))) &&
            K->isNaN())
          ;
        else if (match(X, m_Select(m_Value(), m_Value(V), m_APFloat(K))) &&
                 K->isNaN())
          ;
        else
          break;
        User->setOperand(0, V);
        RecursivelyDeleteTriviallyDeadInstructions(X);
        Changed = true;
      }
    }
    for (Use &U : User->operands()) {
      auto *II = dyn_cast<IntrinsicInst>(U.get());
      if (!II || !II->hasOneUse())
        continue;
      Intrinsic::ID ID = II->getIntrinsicID();
      if (ID != Intrinsic::maximum && ID != Intrinsic::minimum)
        continue;
      IRBuilder<> B(II);
      B.setFastMathFlags(II->getFastMathFlags());
      auto *R = cast<Instruction>(B.CreateBinaryIntrinsic(
          ID == Intrinsic::maximum ? Intrinsic::maxnum : Intrinsic::minnum,
          II->getArgOperand(0), II->getArgOperand(1)));
      R->takeName(II);
      II->replaceAllUsesWith(R);
      II->eraseFromParent();
      Work.push_back(R);
      Changed = true;
    }
  }
  return Changed;
}

static cl::opt<bool> AGXSatConvertClamps(
    "agx-sat-convert-clamps", cl::init(true), cl::Hidden,
    cl::desc("a float clamped above at an 8- or 16-bit integer's maximum and "
             "converted to it as the saturating conversion of the unclamped "
             "value"));

/// fptosi / fptoui (minnum(y, h)) to i8 / i16, h at least the type's
/// maximum and y never NaN, is fpto[su]i.sat(y): above the maximum the
/// conversion was poison (h above it) or the maximum (h equal), which the
/// saturation gives, and below it the truncation commutes with the bound.
/// The conversions saturate in hardware (B27): the bound goes (quant_i8's
/// clamp to [-127, 127] keeps only its lower half).
static bool satConvertClamps(Function &F) {
  if (!AGXSatConvertClamps)
    return false;
  using namespace PatternMatch;
  SmallVector<CastInst *, 8> Work;
  for (Instruction &I : instructions(F))
    if (isa<FPToSIInst>(I) || isa<FPToUIInst>(I))
      Work.push_back(cast<CastInst>(&I));
  bool Changed = false;
  for (CastInst *C : Work) {
    Type *IT = C->getType();
    unsigned W = IT->getScalarSizeInBits();
    if (W != 8 && W != 16)
      continue;
    bool Signed = isa<FPToSIInst>(C);
    Value *Y;
    const APFloat *H;
    if (!match(C->getOperand(0),
               m_CombineOr(m_Intrinsic<Intrinsic::minnum>(m_Value(Y), m_APFloat(H)),
                           m_Intrinsic<Intrinsic::minimumnum>(m_Value(Y),
                                                              m_APFloat(H)))))
      continue;
    double Max = Signed ? double((1u << (W - 1)) - 1) : double((1u << W) - 1);
    if (H->isNaN() || H->convertToDouble() < Max)
      continue;
    // y never NaN: a maxnum / maximumnum with a non-NaN constant, or what
    // ValueTracking proves (sums of products of converted integers, say).
    const APFloat *L;
    if (!(match(Y, m_CombineOr(m_Intrinsic<Intrinsic::maxnum>(m_Value(), m_APFloat(L)),
                               m_Intrinsic<Intrinsic::maximumnum>(m_Value(),
                                                                  m_APFloat(L)))) &&
          !L->isNaN()) &&
        !isKnownNeverNaN(Y, SimplifyQuery(F.getDataLayout())))
      continue;
    IRBuilder<> B(C);
    Value *R = B.CreateIntrinsic(
        Signed ? Intrinsic::fptosi_sat : Intrinsic::fptoui_sat,
        {IT, Y->getType()}, {Y});
    C->replaceAllUsesWith(R);
    C->eraseFromParent();
    Changed = true;
  }
  // A saturating conversion's own bounds: fptoui.sat (maxnum(m, l)), l <= 0,
  // is fptoui.sat (m) for every m (a NaN m gives l, converted to 0, as the
  // conversion gives NaN); fptosi.sat (maxnum(m, l)), l at most the type's
  // minimum, and fpto[su]i.sat (minnum(m, h)), h at least its maximum, are
  // the conversion of m where m is never NaN (a maxnum / minnum with a
  // non-NaN constant).
  auto NeverNaN = [&](Value *V) {
    const APFloat *K;
    return (match(V, m_CombineOr(
                        m_CombineOr(m_Intrinsic<Intrinsic::maxnum>(m_Value(), m_APFloat(K)),
                                    m_Intrinsic<Intrinsic::minnum>(m_Value(), m_APFloat(K))),
                        m_CombineOr(m_Intrinsic<Intrinsic::maximumnum>(m_Value(), m_APFloat(K)),
                                    m_Intrinsic<Intrinsic::minimumnum>(m_Value(), m_APFloat(K))))) &&
            !K->isNaN()) ||
           isKnownNeverNaN(V, SimplifyQuery(F.getDataLayout()));
  };
  SmallVector<IntrinsicInst *, 8> Sats;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && (II->getIntrinsicID() == Intrinsic::fptosi_sat ||
               II->getIntrinsicID() == Intrinsic::fptoui_sat))
      Sats.push_back(II);
  for (IntrinsicInst *II : Sats) {
    bool Signed = II->getIntrinsicID() == Intrinsic::fptosi_sat;
    unsigned W = II->getType()->getScalarSizeInBits();
    if (W > 32)
      continue;
    double Max = Signed ? double((uint64_t(1) << (W - 1)) - 1)
                        : double((uint64_t(1) << W) - 1);
    double Min = Signed ? -double(uint64_t(1) << (W - 1)) : 0.0;
    for (unsigned Round = 0; Round < 2; ++Round) {
      Value *X = II->getArgOperand(0), *M;
      const APFloat *K;
      bool Drop = false;
      if (match(X, m_CombineOr(m_Intrinsic<Intrinsic::maxnum>(m_Value(M), m_APFloat(K)),
                               m_Intrinsic<Intrinsic::maximumnum>(m_Value(M), m_APFloat(K)))) &&
          !K->isNaN() && K->convertToDouble() <= Min)
        Drop = !Signed || NeverNaN(M);
      else if (match(X, m_CombineOr(m_Intrinsic<Intrinsic::minnum>(m_Value(M), m_APFloat(K)),
                                    m_Intrinsic<Intrinsic::minimumnum>(m_Value(M), m_APFloat(K)))) &&
               !K->isNaN() && K->convertToDouble() >= Max)
        Drop = NeverNaN(M);
      if (!Drop)
        break;
      II->setArgOperand(0, M);
      Changed = true;
    }
  }
  return Changed;
}

static cl::opt<unsigned> AGXFoldConditionChains(
    "agx-fold-condition-chains", cl::init(2), cl::Hidden,
    cl::desc("merge a branch on a condition a few more instructions decide "
             "(an || or && of comparisons in their own blocks) into its "
             "predecessor's, with this many speculated instructions at most "
             "(0: off)"));

/// SimplifyCFG's foldBranchToCommonDest on every conditional branch: a
/// block that computes the second half of an || or && and branches to one
/// of its predecessor's successors joins the predecessor's branch (its
/// instructions speculated there). Every branch is execution mask
/// operations (an if, a pop, often a jump over the body), which cost more
/// than a few comparisons; the phis of the empty side that remains become
/// selects in the SimplifyCFG before structurization. (Only this fold:
/// SimplifyCFG's branch simplification would also make switches of
/// comparison chains, whose lowered trees the structurizer cannot always
/// handle.)
static bool foldConditionChains(Function &F, const TargetTransformInfo &TTI) {
  if (!AGXFoldConditionChains)
    return false;
  bool Changed = false;
  for (unsigned Round = 0; Round < 8; ++Round) {
    SmallVector<WeakTrackingVH, 16> Branches;
    for (BasicBlock &BB : F)
      if (auto *BI = dyn_cast<CondBrInst>(BB.getTerminator()))
        Branches.push_back(BI);
    bool Local = false;
    for (WeakTrackingVH &V : Branches)
      if (auto *BI = dyn_cast_or_null<CondBrInst>(V))
        Local |= foldBranchToCommonDest(BI, nullptr, nullptr, &TTI, nullptr,
                                        AGXFoldConditionChains);
    if (!Local)
      break;
    Changed = true;
  }
  if (Changed)
    removeUnreachableBlocks(F);
  return Changed;
}

static cl::opt<bool> AGXWordBFloatLoads(
    "agx-word-bfloat-loads", cl::init(true), cl::Hidden,
    cl::desc("a vector of bfloats loaded only to be extended to floats as "
             "32-bit words, each pair's floats the word's halves, and "
             "vectors of bfloats stored as 16-bit integers"));

/// A <2k x bfloat> load whose only users extend it to floats: bfloat is not
/// a register type, so the load was split into a 16-bit load per element,
/// each shifted into a float. Loaded as <2k x i16> instead (one load, as k
/// words), each element zero-extended and shifted left by 16: the DAG makes
/// the high element of a word the word with its low half cleared (an and on
/// the ALU), the low element the word shifted left by 16.
static bool loadBFloatsAsWords(Function &F) {
  if (!AGXWordBFloatLoads)
    return false;
  bool Changed = false;
  SmallVector<LoadInst *, 8> Loads;
  for (Instruction &I : instructions(F)) {
    // A <2k x bfloat> store likewise, as <2k x i16> (the same bits element
    // by element): one store of k words instead of one per element.
    if (auto *St = dyn_cast<StoreInst>(&I)) {
      auto *VT = dyn_cast<FixedVectorType>(St->getValueOperand()->getType());
      if (VT && VT->getElementType()->isBFloatTy() &&
          VT->getNumElements() % 2 == 0 && St->isSimple() &&
          St->getAlign() >= 4) {
        // Element by element (the scalarizer's elements, gathered), so
        // that the DAG sees each element's 16 bits.
        IRBuilder<> B(St);
        unsigned N = VT->getNumElements();
        Value *V = PoisonValue::get(FixedVectorType::get(B.getInt16Ty(), N));
        for (unsigned E = 0; E < N; ++E)
          V = B.CreateInsertElement(
              V,
              B.CreateBitCast(B.CreateExtractElement(St->getValueOperand(), E),
                              B.getInt16Ty()),
              E);
        St->setOperand(0, V);
        Changed = true;
      }
      continue;
    }
    auto *L = dyn_cast<LoadInst>(&I);
    auto *VT = L ? dyn_cast<FixedVectorType>(L->getType()) : nullptr;
    if (!VT || !VT->getElementType()->isBFloatTy() ||
        VT->getNumElements() % 2 || !L->isSimple() || L->getAlign() < 4 ||
        L->user_empty())
      continue;
    auto IsFloatExt = [](const User *U) {
      auto *E = dyn_cast<FPExtInst>(U);
      return E && E->getType()->getScalarType()->isFloatTy();
    };
    // The vector extended, or (after the scalarizer) its elements each.
    if (all_of(L->users(), [&](const User *U) {
          if (IsFloatExt(U))
            return true;
          auto *X = dyn_cast<ExtractElementInst>(U);
          return X && isa<ConstantInt>(X->getIndexOperand()) &&
                 !X->user_empty() && all_of(X->users(), IsFloatExt);
        }))
      Loads.push_back(L);
  }
  for (LoadInst *L : Loads) {
    unsigned N = cast<FixedVectorType>(L->getType())->getNumElements();
    IRBuilder<> B(L);
    // As 16-bit integers, which keeps each element's poison its own (a word
    // with a poison half would be poison whole); the DAG sees the vector as
    // words, each element's float a half of one.
    auto *HT = FixedVectorType::get(B.getInt16Ty(), N);
    LoadInst *W = B.CreateAlignedLoad(HT, L->getPointerOperand(),
                                      L->getAlign(), L->getName() + ".bits");
    W->setAAMetadata(L->getAAMetadata());
    for (unsigned K : {LLVMContext::MD_nontemporal,
                       LLVMContext::MD_invariant_load})
      if (MDNode *MD = L->getMetadata(K))
        W->setMetadata(K, MD);
    SmallVector<Value *, 8> Floats;
    for (unsigned I = 0; I < N; ++I) {
      Value *E = B.CreateZExt(B.CreateExtractElement(W, I), B.getInt32Ty());
      Floats.push_back(B.CreateBitCast(B.CreateShl(E, 16), B.getFloatTy()));
    }
    Value *V = nullptr;
    for (User *U : make_early_inc_range(L->users())) {
      auto *E = cast<Instruction>(U);
      if (auto *X = dyn_cast<ExtractElementInst>(E)) {
        uint64_t Idx = cast<ConstantInt>(X->getIndexOperand())->getZExtValue();
        Value *Fl = Idx < N ? Floats[Idx]
                            : PoisonValue::get(B.getFloatTy());
        for (User *XU : make_early_inc_range(X->users())) {
          cast<Instruction>(XU)->replaceAllUsesWith(Fl);
          cast<Instruction>(XU)->eraseFromParent();
        }
        X->eraseFromParent();
        continue;
      }
      if (!V) {
        V = PoisonValue::get(FixedVectorType::get(B.getFloatTy(), N));
        for (unsigned I = 0; I < N; ++I)
          V = B.CreateInsertElement(V, Floats[I], I);
      }
      E->replaceAllUsesWith(V);
      E->eraseFromParent();
    }
    L->eraseFromParent();
  }
  return Changed || !Loads.empty();
}

static bool permuteBytes(Function &F) {
  if (!AGXBytePermutes)
    return false;
  SmallVector<WeakVH, 16> Roots;
  for (Instruction &I : instructions(F)) {
    if (I.getOpcode() != Instruction::Or || !I.getType()->isIntegerTy(32))
      continue;
    if (I.hasOneUse())
      if (auto *U = dyn_cast<Instruction>(I.user_back());
          U && U->getType()->isIntegerTy(32) &&
          (U->getOpcode() == Instruction::Or ||
           U->getOpcode() == Instruction::And ||
           U->getOpcode() == Instruction::Shl ||
           U->getOpcode() == Instruction::LShr ||
           U->getOpcode() == Instruction::AShr))
        continue;
    Roots.push_back(&I);
  }
  bool Changed = false;
  for (WeakVH &H : Roots) {
    auto *Root = dyn_cast_or_null<Instruction>(H);
    if (!Root)
      continue;
    SmallVector<Instruction *, 16> Tree;
    Bytes4 Src = byteSources(Root, 8, Tree);
    if (Tree.empty() || Tree.back() != Root)
      continue;
    // the instructions that go with the tree: the root and those of one use
    unsigned Old = 0;
    for (Instruction *I : Tree)
      Old += I == Root || I->hasOneUse();
    // a byte of a value narrower than 32 bits: its 16-bit half
    auto HalfCost = [](Value *S, unsigned K) {
      return S->getType()->isIntegerTy(8) ? 1u : 0u;
    };
    auto SignCost = [&](const ByteSrc &B) {
      return HalfCost(B.V, B.Idx / 2) + (B.Idx % 2 ? 1u : 2u);
    };
    // the cost of half H (bytes 2H, 2H + 1), and whether it is zero
    auto PlanHalf = [&](const ByteSrc &Q0, const ByteSrc &Q1) -> unsigned {
      if (Q0.K == ByteSrc::Zero && Q1.K == ByteSrc::Zero)
        return 0;
      if (Q0.K == ByteSrc::Byte && Q1.K == ByteSrc::Byte && Q0.V == Q1.V &&
          Q0.Idx % 2 == 0 && Q1.Idx == Q0.Idx + 1)
        return HalfCost(Q0.V, Q0.Idx / 2);
      if (Q0.K == ByteSrc::Sign && Q0 == Q1)
        return SignCost(Q0);
      auto Carrier = [&](const ByteSrc &Q) -> unsigned {
        if (Q.K == ByteSrc::Sign)
          return SignCost(Q);
        return HalfCost(Q.V, Q.Idx / 2) + Q.Idx % 2;
      };
      if (Q1.K == ByteSrc::Zero)
        return Carrier(Q0) + (Q0.K == ByteSrc::Byte && Q0.Idx % 2 ? 0 : 1);
      if (Q0.K == ByteSrc::Zero)
        return (Q1.K == ByteSrc::Sign ? SignCost(Q1) : HalfCost(Q1.V, 0)) + 1;
      // the low byte's carrier, bfi into the high byte's half (or of it)
      unsigned High = Q1.K == ByteSrc::Sign ? SignCost(Q1)
                                            : HalfCost(Q1.V, Q1.Idx / 2);
      return Carrier(Q0) + High + 1;
    };
    bool Z0 = Src[0].K == ByteSrc::Zero && Src[1].K == ByteSrc::Zero;
    bool Z1 = Src[2].K == ByteSrc::Zero && Src[3].K == ByteSrc::Zero;
    unsigned New = PlanHalf(Src[0], Src[1]) + PlanHalf(Src[2], Src[3]) +
                   (Z0 != Z1 ? 1 : 0);
    // a sign-extended byte of a 32-bit value: two shifts
    bool SextByte = Src[0].K == ByteSrc::Byte &&
                    Src[0].V->getType()->isIntegerTy(32) &&
                    Src[1].K == ByteSrc::Sign && Src[1].V == Src[0].V &&
                    Src[1].Idx == Src[0].Idx && Src[2] == Src[1] &&
                    Src[3] == Src[1];
    if (SextByte)
      New = std::min(New, Src[0].Idx == 3 ? 1u : 2u);
    // (halves that are halves of values, or zero, are the register halves
    // instruction selection already makes: only trees with bytes moved)
    auto WholeHalf = [](const ByteSrc &Q0, const ByteSrc &Q1) {
      return (Q0.K == ByteSrc::Zero && Q1.K == ByteSrc::Zero) ||
             (Q0.K == ByteSrc::Byte && Q1.K == ByteSrc::Byte && Q0.V == Q1.V &&
              Q0.Idx % 2 == 0 && Q1.Idx == Q0.Idx + 1);
    };
    if (WholeHalf(Src[0], Src[1]) && WholeHalf(Src[2], Src[3]))
      continue;
    if (New >= Old)
      continue;
    IRBuilder<> B(Root);
    Type *I16 = B.getInt16Ty();
    Value *R;
    if (SextByte && New == (Src[0].Idx == 3 ? 1u : 2u)) {
      Value *X = Src[0].V;
      if (Src[0].Idx != 3)
        X = B.CreateShl(X, 24 - 8 * Src[0].Idx);
      R = B.CreateAShr(X, 24);
    } else {
      DenseMap<std::pair<Value *, unsigned>, Value *> Halves, Signs;
      auto Half = [&](Value *S, unsigned K) -> Value * {
        Value *&H = Halves[{S, K}];
        if (H)
          return H;
        if (S->getType()->isIntegerTy(32))
          H = B.CreateTrunc(K ? B.CreateLShr(S, 16) : S, I16);
        else
          H = B.CreateZExtOrTrunc(S, I16);
        return H;
      };
      auto Sign = [&](const ByteSrc &Q) -> Value * {
        Value *&G = Signs[{Q.V, Q.Idx}];
        if (G)
          return G;
        Value *H = Half(Q.V, Q.Idx / 2);
        if (Q.Idx % 2 == 0)
          H = B.CreateShl(H, 8);
        G = B.CreateAShr(H, 15);
        return G;
      };
      // a value with byte Q in its low byte
      auto Low = [&](const ByteSrc &Q) -> Value * {
        if (Q.K == ByteSrc::Sign)
          return Sign(Q);
        Value *H = Half(Q.V, Q.Idx / 2);
        return Q.Idx % 2 ? B.CreateLShr(H, 8) : H;
      };
      auto BuildHalf = [&](const ByteSrc &Q0, const ByteSrc &Q1) -> Value * {
        if (Q0.K == ByteSrc::Zero && Q1.K == ByteSrc::Zero)
          return nullptr;
        if (Q0.K == ByteSrc::Byte && Q1.K == ByteSrc::Byte && Q0.V == Q1.V &&
            Q0.Idx % 2 == 0 && Q1.Idx == Q0.Idx + 1)
          return Half(Q0.V, Q0.Idx / 2);
        if (Q0.K == ByteSrc::Sign && Q0 == Q1)
          return Sign(Q0);
        if (Q1.K == ByteSrc::Zero) {
          Value *L = Low(Q0);
          return Q0.K == ByteSrc::Byte && Q0.Idx % 2
                     ? L
                     : B.CreateAnd(L, ConstantInt::get(I16, 255));
        }
        // the high byte: in the high byte of a half (and its mask), or in
        // the low byte of one shifted up
        Value *Hi;
        if (Q1.K == ByteSrc::Sign)
          Hi = B.CreateAnd(Sign(Q1), ConstantInt::get(I16, 0xff00));
        else if (Q1.Idx % 2)
          Hi = B.CreateAnd(Half(Q1.V, Q1.Idx / 2),
                           ConstantInt::get(I16, 0xff00));
        else
          Hi = B.CreateShl(Half(Q1.V, Q1.Idx / 2), 8);
        if (Q0.K == ByteSrc::Zero)
          return Hi;
        Value *L = B.CreateAnd(Low(Q0), ConstantInt::get(I16, 255));
        return B.CreateOr(L, Hi, "", /*IsDisjoint=*/true);
      };
      Value *H0 = BuildHalf(Src[0], Src[1]);
      Value *H1 = BuildHalf(Src[2], Src[3]);
      Type *I32 = B.getInt32Ty();
      Value *W0 = H0 ? B.CreateZExt(H0, I32) : nullptr;
      Value *W1 = H1 ? B.CreateShl(B.CreateZExt(H1, I32), 16) : nullptr;
      R = W0 && W1 ? B.CreateOr(W0, W1, "", /*IsDisjoint=*/true)
                   : (W0 ? W0 : W1);
    }
    R->takeName(Root);
    Root->replaceAllUsesWith(R);
    RecursivelyDeleteTriviallyDeadInstructions(Root);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXHoistAddressParts(
    "agx-hoist-address-parts", cl::init(true), cl::Hidden,
    cl::desc("A loop's integer address ((inv + i) << c) + base as "
             "(base + (inv << c)) + (i << c), the sum before the loop"));

/// An address that PTX computes in 64-bit integers, ((Inv + I) << C) + Base
/// with Inv and Base the same every iteration (a row's offset and a buffer)
/// and I the loop's index (a sign-extended 32-bit counter), is
/// (Base + (Inv << C)) + (I << C) modulo 2^64: the first sum goes before
/// the loop, and the access takes the base and the 32-bit index (two 64-bit
/// additions less each iteration). Loop strength reduction does not see it:
/// I's add has no nsw, so its extension is not a recurrence.
static bool hoistInvariantAddressParts(Function &F) {
  if (!AGXHoistAddressParts)
    return false;
  using namespace PatternMatch;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    BasicBlock *Pre = L->getLoopPreheader();
    if (!Pre)
      continue;
    auto Inv = [&](Value *V) { return L->isLoopInvariant(V); };
    for (BasicBlock *BB : L->blocks()) {
      if (LI.getLoopFor(BB) != L)
        continue;
      for (Instruction &I : make_early_inc_range(*BB)) {
        Value *X, *Base, *A, *Bv;
        const APInt *C = nullptr;
        if (!I.getType()->isIntegerTy(64) ||
            !match(&I, m_c_Add(m_Value(X), m_Value(Base))) || !Inv(Base) ||
            Inv(X) || !X->hasOneUse() ||
            !all_of(I.users(), [](User *U) { return isa<IntToPtrInst>(U); }))
          continue;
        Value *Y = X;
        if (match(X, m_Shl(m_Value(Y), m_APInt(C))) &&
            (C->uge(64) || !Y->hasOneUse()))
          continue;
        if (!match(Y, m_c_Add(m_Value(A), m_Value(Bv))) || !isa<Instruction>(Y))
          continue;
        if (!Inv(A))
          std::swap(A, Bv);
        if (!Inv(A) || Inv(Bv))
          continue;
        IRBuilder<> P(Pre->getTerminator());
        Value *Hoisted = P.CreateAdd(Base, C ? P.CreateShl(A, *C) : A);
        IRBuilder<> B(&I);
        Value *New = B.CreateAdd(Hoisted, C ? B.CreateShl(Bv, *C) : Bv);
        New->takeName(&I);
        I.replaceAllUsesWith(New);
        RecursivelyDeleteTriviallyDeadInstructions(&I);
        Changed = true;
      }
    }
  }
  return Changed;
}

static cl::opt<bool> AGXNarrowMaskedIndices(
    "agx-narrow-masked-indices", cl::init(true), cl::Hidden,
    cl::desc("(zext(x) << c) & m with m below 2^32 as zext((x << c) & m)"));

/// An index modulo a power of two that MLIR computes in 64 bits,
/// (zext(x) << c) & m with m below 2^32 (a broadcast operand's (i * 4) %
/// 4096), is zext((x << c) & m): the mask clears every bit the 32-bit
/// shift loses. The access then takes the base and the 32-bit index (no
/// 64-bit shift, and or add, nor the base's copy to registers).
static bool narrowMaskedIndices(Function &F) {
  if (!AGXNarrowMaskedIndices)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    Value *X;
    const APInt *C, *M;
    if (!I.getType()->isIntegerTy(64) ||
        !match(&I, m_And(m_Shl(m_ZExt(m_Value(X)), m_APInt(C)), m_APInt(M))) ||
        !X->getType()->isIntegerTy(32) || C->uge(32) || !M->isIntN(32))
      continue;
    IRBuilder<> B(&I);
    Value *N = B.CreateAnd(B.CreateShl(X, C->getZExtValue()),
                           M->trunc(32).getZExtValue());
    Value *Z = B.CreateZExt(N, I.getType());
    Z->takeName(&I);
    I.replaceAllUsesWith(Z);
    RecursivelyDeleteTriviallyDeadInstructions(&I);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> BFloatSqrtProduct(
    "agx-bfloat-sqrt-product",
    cl::desc("bfloat sqrt as the float x rsqrt_special(x) rounded to bfloat"),
    cl::init(true), cl::Hidden);

/// A bfloat square root (computed in float, the float one correctly rounded
/// with its remainder correction and small-argument scaling, then rounded):
/// the float product x rsqrt_special(x) rounded to bfloat is the same for
/// every bfloat where float arithmetic flushes (simulator, all 65536:
/// L15-E335). rsqrt_special gives 1 for 0 and infinity, so zeros,
/// infinities and NaNs come out of the product.
static bool bfloatSqrts(Function &F, const AGXSubtarget &ST) {
  if (!BFloatSqrtProduct || ST.hasIEEEFP32())
    return false;
  SmallVector<IntrinsicInst *, 4> Sqrts;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::sqrt &&
          II->getType()->isBFloatTy() && !II->hasApproxFunc())
        Sqrts.push_back(II);
  for (IntrinsicInst *II : Sqrts) {
    IRBuilder<> B(II);
    Value *X = B.CreateFPExt(II->getArgOperand(0), B.getFloatTy());
    Value *Y = B.CreateIntrinsic(Intrinsic::agx_rsqrt_special,
                                 {B.getFloatTy()}, {X});
    Value *R = B.CreateFPTrunc(B.CreateFMul(X, Y), II->getType());
    R->takeName(II);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
  }
  return !Sqrts.empty();
}

static cl::opt<bool> AGXRelativeIndexSelects(
    "agx-relative-index-selects", cl::init(true), cl::Hidden,
    cl::desc("A loop's chain of selects of the counter plus constants "
             "(argmax's index) kept relative to the counter"));

/// A loop-carried value p chained through selects of the counter plus
/// constants (an unrolled argmax: p = c0 ? i : p, c1 ? i | 1 : p, ...):
/// every unrolled copy computes its i + k (an or or add each), the selects
/// read them. Kept as d = p - i instead, the selects take the constants k
/// (immediates), d steps by -s once an iteration (s the counter's step),
/// and p = d + i is formed where it is used after the loop (L15-E490).
/// Exact in wrapping integer arithmetic.
static bool relativeIndexSelects(Function &F) {
  if (!AGXRelativeIndexSelects)
    return false;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  if (LI.empty())
    return false;
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII, &F);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    BasicBlock *H = L->getHeader(), *Latch = L->getLoopLatch(),
               *Pre = L->getLoopPreheader(), *Exit = L->getExitBlock();
    if (!L->isInnermost() || !Latch || !Pre || !Exit ||
        Exit->getSinglePredecessor() != Latch ||
        L->getExitingBlock() != Latch)
      continue;
    // the counter: an integer header phi, an affine recurrence with a
    // constant step
    SmallVector<PHINode *, 4> Phis;
    for (PHINode &P : H->phis())
      Phis.push_back(&P);
    for (PHINode *P : Phis) {
      Type *Ty = P->getType();
      if (!Ty->isIntegerTy())
        continue;
      // the chain from the latch value back to p
      SmallVector<SelectInst *, 8> Chain;
      Value *V = P->getIncomingValueForBlock(Latch);
      bool Ok = true;
      while (V != P) {
        auto *S = dyn_cast<SelectInst>(V);
        if (!S || !L->contains(S) || S->getParent() != H) {
          Ok = false;
          break;
        }
        Chain.push_back(S);
        Value *T = S->getTrueValue(), *Fv = S->getFalseValue();
        // the previous link: the operand that is p or a select of the chain
        auto IsLink = [&](Value *X) {
          return X == P || (isa<SelectInst>(X) &&
                            cast<SelectInst>(X)->getParent() == H &&
                            !isa<Constant>(X));
        };
        bool TL = IsLink(T), FL = IsLink(Fv);
        if (TL == FL) {
          Ok = false;
          break;
        }
        V = TL ? T : Fv;
      }
      if (!Ok || Chain.size() < 2)
        continue;
      std::reverse(Chain.begin(), Chain.end());
      // each link's other operand: the counter plus a constant
      PHINode *IV = nullptr;
      const SCEVConstant *Step = nullptr;
      SmallVector<APInt, 8> Ks;
      unsigned NonZero = 0;
      for (PHINode *C : Phis) {
        if (C == P || !C->getType()->isIntegerTy())
          continue;
        auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(C));
        if (!AR || AR->getLoop() != L || !AR->isAffine())
          continue;
        auto *St = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
        if (!St)
          continue;
        const SCEV *Base = SE.getTruncateOrZeroExtend(AR, Ty);
        SmallVector<APInt, 8> K;
        unsigned NZ = 0;
        for (SelectInst *S : Chain) {
          Value *A = (S->getTrueValue() == P ||
                      llvm::is_contained(Chain, S->getTrueValue()))
                         ? S->getFalseValue()
                         : S->getTrueValue();
          auto *D = dyn_cast<SCEVConstant>(
              SE.getMinusSCEV(SE.getSCEV(A), Base));
          if (!D) {
            K.clear();
            break;
          }
          K.push_back(D->getAPInt());
          NZ += !D->getAPInt().isZero();
        }
        if (K.size() != Chain.size())
          continue;
        // (a narrower type than the counter's: its step truncated)
        if (Ty->getIntegerBitWidth() > St->getType()->getIntegerBitWidth())
          continue;
        IV = C;
        Step = St;
        Ks = std::move(K);
        NonZero = NZ;
        break;
      }
      // (one subtraction an iteration for the ors / adds it saves)
      if (!IV || NonZero < 2)
        continue;
      // p only in the chain's first link; each link only in the next one,
      // the last in p and after the loop
      if (!P->hasOneUse() || *P->user_begin() != Chain.front())
        continue;
      for (unsigned I = 0; Ok && I + 1 < Chain.size(); ++I)
        Ok = Chain[I]->hasOneUse() && *Chain[I]->user_begin() == Chain[I + 1];
      // its uses after the loop where the exit block dominates them (for a
      // phi, the edge it takes the value on): p = d + i is formed there
      for (Use &U : Chain.back()->uses()) {
        auto *UI = cast<Instruction>(U.getUser());
        if (UI == P)
          continue;
        BasicBlock *At = UI->getParent();
        if (auto *PN = dyn_cast<PHINode>(UI))
          At = PN->getIncomingBlock(U);
        if (L->contains(UI) || !DT.dominates(Exit, At))
          Ok = false;
      }
      if (!Ok)
        continue;
      // d = p - i: its phi, the links of constants, -s at the latch
      auto Cast = [&](IRBuilder<> &B, Value *X) {
        return B.CreateZExtOrTrunc(X, Ty);
      };
      IRBuilder<> PB(Pre->getTerminator());
      Value *Init = PB.CreateSub(P->getIncomingValueForBlock(Pre),
                                 Cast(PB, IV->getIncomingValueForBlock(Pre)),
                                 "rel.init");
      PHINode *D = PHINode::Create(Ty, 2, "rel", H->getFirstNonPHIIt());
      Value *Prev = D;
      for (unsigned I = 0; I < Chain.size(); ++I) {
        SelectInst *S = Chain[I];
        Constant *K = ConstantInt::get(Ty, Ks[I]);
        bool LinkTrue = S->getTrueValue() == P ||
                        (I && S->getTrueValue() == Chain[I - 1]);
        IRBuilder<> B(S);
        Prev = B.CreateSelect(S->getCondition(), LinkTrue ? Prev : K,
                              LinkTrue ? K : Prev, S->getName() + ".rel");
      }
      IRBuilder<> LB(Latch->getTerminator());
      Value *Next = LB.CreateSub(
          Prev, ConstantInt::get(Ty, Step->getAPInt().zextOrTrunc(
                                         Ty->getIntegerBitWidth())),
          "rel.next");
      D->addIncoming(Init, Pre);
      D->addIncoming(Next, Latch);
      // after the loop: the last link's d plus the counter of that
      // iteration, i.e. d - s plus the next counter, the values the latch
      // computes anyway (with the counter itself, strength reduction kept
      // another counter for it)
      SelectInst *Last = Chain.back();
      IRBuilder<> EB(&*Exit->getFirstInsertionPt());
      // (in the counter's type, then truncated: a truncation of the counter
      // made strength reduction keep a narrower counter for it)
      Value *IVNext = IV->getIncomingValueForBlock(Latch);
      // (its value after the loop where SCEV knows it, loop-invariant: an
      // index of the counter there made strength reduction keep one more
      // counter, as it addresses by byte offsets)
      const SCEV *AtExit =
          SE.getSCEVAtScope(SE.getSCEV(IVNext), L->getParentLoop());
      SCEVExpander X(SE, "rel");
      if (SE.isLoopInvariant(AtExit, L) &&
          X.isSafeToExpandAt(AtExit, &*Exit->getFirstInsertionPt()))
        // (inserted before the builder's point, so before what it builds)
        IVNext = X.expandCodeFor(AtExit, IV->getType(),
                                 Exit->getFirstInsertionPt());
      Value *Out =
          IV->getType() == Ty
              ? EB.CreateAdd(Next, IVNext, "rel.out")
              : EB.CreateTrunc(
                    EB.CreateAdd(IVNext, EB.CreateZExt(Next, IV->getType())),
                    Ty, "rel.out");
      // (a phi of it in the exit block, whose one predecessor is the latch:
      // the value itself; a phi elsewhere takes it on an edge the exit
      // dominates)
      SmallVector<PHINode *, 2> ExitPhis;
      SmallVector<Use *, 4> Outs;
      for (Use &U : Last->uses()) {
        if (U.getUser() == P)
          continue;
        auto *EP = dyn_cast<PHINode>(U.getUser());
        if (EP && EP->getParent() == Exit)
          ExitPhis.push_back(EP);
        else
          Outs.push_back(&U);
      }
      for (Use *U : Outs)
        U->set(Out);
      for (PHINode *EP : ExitPhis) {
        EP->replaceAllUsesWith(Out);
        EP->eraseFromParent();
      }
      // p, its chain and the counter-plus-constant values left dead
      P->replaceAllUsesWith(PoisonValue::get(Ty));
      P->eraseFromParent();
      for (SelectInst *S : llvm::reverse(Chain)) {
        S->replaceAllUsesWith(PoisonValue::get(Ty));
        S->eraseFromParent();
      }
      Changed = true;
    }
  }
  return Changed;
}

static cl::opt<bool> AGXRebaseLoopIndices(
    "agx-rebase-loop-indices", cl::init(true), cl::Hidden,
    cl::desc("A loop's device access at B[zext(X + Y)], X invariant, as (B + "
             "zext(X))[zext(Y)] where another access's index is Y"));

/// In a loop, an access at B[zext(x + y)] (x invariant, x + y not wrapping)
/// beside one at C[y'] whose index is y as well (a convolution's x[i + k]
/// and w[k], k the counter): loop strength reduction keeps a counter for
/// each index, an add each an iteration. As (B + zext(x))[zext(y)], both
/// accesses take one index register (and the loop's counter is that one),
/// for an invariant base computed once (L15-E398).
static bool rebaseLoopIndices(Function &F) {
  if (!AGXRebaseLoopIndices)
    return false;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  if (LI.empty())
    return false;
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII, &F);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  using namespace PatternMatch;
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    BasicBlock *Pre = L->getLoopPreheader();
    if (!L->isInnermost() || !Pre)
      continue;
    // the accesses' 64-bit index SCEVs
    SmallVector<std::pair<GetElementPtrInst *, const SCEV *>, 8> Accesses;
    for (BasicBlock *BB : L->blocks())
      for (Instruction &I : *BB) {
        Value *Ptr = getLoadStorePointerOperand(&I);
        auto *G = dyn_cast_or_null<GetElementPtrInst>(Ptr);
        if (!G || G->getAddressSpace() != AGX::DEVICE_ADDRESS ||
            G->getNumIndices() != 1 || !L->isLoopInvariant(G->getPointerOperand()) ||
            !G->getOperand(1)->getType()->isIntegerTy(64))
          continue;
        Accesses.push_back({G, SE.getSCEV(G->getOperand(1))});
      }
    for (auto &Acc : Accesses) {
      GetElementPtrInst *G = Acc.first;
      Value *X, *Y;
      if (!match(G->getOperand(1), m_ZExt(m_Add(m_Value(X), m_Value(Y)))))
        continue;
      if (!L->isLoopInvariant(X))
        std::swap(X, Y);
      // (a constant x is E250's immediate index from one shared base)
      if (!L->isLoopInvariant(X) || L->isLoopInvariant(Y) ||
          !X->getType()->isIntegerTy(32) || isa<Constant>(X))
        continue;
      if (auto *XI = dyn_cast<Instruction>(X);
          XI && !DT.dominates(XI, Pre->getTerminator()))
        continue;
      if (!isa<Instruction>(G->getOperand(1)) ||
          !cast<Instruction>(G->getOperand(1))->hasOneUse())
        continue;
      // another access indexed by zext(y)
      const SCEV *SY = SE.getZeroExtendExpr(SE.getSCEV(Y),
                                            Type::getInt64Ty(F.getContext()));
      GetElementPtrInst *Self = G;
      if (none_of(Accesses,
                  [&](auto &A) { return A.first != Self && A.second == SY; }))
        continue;
      // x + y does not wrap
      std::optional<APInt> BX = AGX::scevUpperBound(SE.getSCEV(X), SE, F);
      std::optional<APInt> BY = AGX::scevUpperBound(SE.getSCEV(Y), SE, F);
      if (!BX || !BY || !(*BX + *BY).ult(APInt(128, 1).shl(32)))
        continue;
      IRBuilder<> PB(Pre->getTerminator());
      Value *Base = PB.CreateGEP(G->getSourceElementType(),
                                 G->getPointerOperand(),
                                 PB.CreateZExt(X, PB.getInt64Ty()), "rebased");
      IRBuilder<> B(G);
      auto *Idx = cast<Instruction>(G->getOperand(1));
      Value *NG = B.CreateGEP(G->getSourceElementType(), Base,
                              B.CreateZExt(Y, B.getInt64Ty()));
      G->replaceAllUsesWith(NG);
      G->eraseFromParent();
      RecursivelyDeleteTriviallyDeadInstructions(Idx);
      Changed = true;
    }
  }
  return Changed;
}

static cl::opt<bool> AGXReluZeroSelects(
    "agx-relu-zero-selects", cl::init(true), cl::Hidden,
    cl::desc("maximum(x, +0)'s portable zero select dropped where maxnum(x, "
             "+0) is x > 0 ? x : +0"));

/// AIR has no maximum: the portable lowering of MLIR's arith.maximumf(x, 0)
/// (relu) is maxnum(x, +0), then select(x == 0, +0, ...) for -0 (maxnum may
/// give either zero) and the NaN select. maxnum(x, +0) as x > 0 ? x : +0
/// (one fcmpsel, as it is lowered anyway) is +0 wherever x == 0, and so is
/// a minnum of it with a constant above zero (relu6's min(..., 6)): the
/// zero select's arms then agree and it goes (L15-E399).
static bool reluZeroSelects(Function &F) {
  if (!AGXReluZeroSelects)
    return false;
  using namespace PatternMatch;
  SmallVector<SelectInst *, 8> Selects;
  for (Instruction &I : instructions(F))
    if (auto *Sel = dyn_cast<SelectInst>(&I);
        Sel && Sel->getType()->isFPOrFPVectorTy() &&
        match(Sel->getTrueValue(), m_PosZeroFP()))
      Selects.push_back(Sel);
  bool Changed = false;
  for (SelectInst *Sel : Selects) {
    Value *X;
    if (!match(Sel->getCondition(),
               m_SpecificFCmp(FCmpInst::FCMP_OEQ, m_Value(X), m_AnyZeroFP())))
      continue;
    // M = maxnum(x, +0), or minnum(M, c), c > 0
    Value *M = Sel->getFalseValue(), *Max = M;
    const APFloat *C;
    if (match(M, m_Intrinsic<Intrinsic::minnum>(m_Value(Max), m_APFloat(C))) ||
        match(M, m_Intrinsic<Intrinsic::minnum>(m_APFloat(C), m_Value(Max)))) {
      if (C->isNegative() || C->isZero() || C->isNaN())
        continue;
    }
    auto *MaxI = dyn_cast<IntrinsicInst>(Max);
    if (!MaxI || MaxI->getIntrinsicID() != Intrinsic::maxnum)
      continue;
    Value *A = MaxI->getArgOperand(0), *B = MaxI->getArgOperand(1);
    if (!match(B, m_PosZeroFP()))
      std::swap(A, B);
    if (A != X || !match(B, m_PosZeroFP()))
      continue;
    // the max as the select (once for all its users: any is allowed), of
    // x frozen (it is used twice; alive2: an undef x otherwise)
    IRBuilder<> IB(MaxI);
    Value *FA = IB.CreateFreeze(A);
    Value *Gt = IB.CreateFCmpOGT(FA, B);
    Value *NewMax = IB.CreateSelect(Gt, FA, B);
    NewMax->takeName(MaxI);
    MaxI->replaceAllUsesWith(NewMax);
    MaxI->eraseFromParent();
    Sel->replaceAllUsesWith(Sel->getFalseValue());
    Sel->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXExactDyadicI16(
    "agx-exact-dyadic-i16", cl::init(true), cl::Hidden,
    cl::desc("Build exact dyadic sums in 16 bits where they fit"));

static cl::opt<bool> AGXExactDyadicFloatRoots(
    "agx-exact-dyadic-float-roots", cl::init(true), cl::Hidden,
    cl::desc("Exact dyadic sums whose value stays a float: in integers, "
             "converted once"));

static cl::opt<bool> AGXExactDyadicSums(
    "agx-exact-dyadic-sums", cl::init(true), cl::Hidden,
    cl::desc("Float sums of converted small integers times dyadic constants "
             "that are exact, converted back to integers, in integers"));

namespace {
/// A float value that is exactly N 2^E, N an integer in [Lo, Hi] (a zero
/// has no exponent of its own: Zero).
struct Dyadic {
  int E = 0;
  int64_t Lo = 0, Hi = 0;
  bool Zero = false;
};

/// Float arithmetic on small integers (converted bytes, say) and dyadic
/// constants whose every result is exact: |N| < 2^24 at a common exponent
/// E >= -126 (no rounding, no denormal to flush), so the float operations
/// compute N 2^E exactly whatever the rounding (L15-E401).
struct DyadicSums {
  const DataLayout &DL;
  DenseMap<Value *, std::optional<Dyadic>> Memo;
  SmallPtrSet<Instruction *, 16> Tree;
  /// The integer type the N are built in (i16 where every scaled value
  /// fits: 16-bit halves of registers).
  IntegerType *Ty = nullptr;
  explicit DyadicSums(const DataLayout &DL) : DL(DL) {}

  static bool fits(int64_t Lo, int64_t Hi, int E) {
    return Lo > -(int64_t(1) << 24) && Hi < (int64_t(1) << 24) && E >= -126 &&
           E <= 100;
  }

  std::optional<Dyadic> get(Value *V, unsigned Depth = 0) {
    if (auto It = Memo.find(V); It != Memo.end())
      return It->second;
    std::optional<Dyadic> R = compute(V, Depth);
    Memo[V] = R;
    if (R)
      if (auto *I = dyn_cast<Instruction>(V))
        Tree.insert(I);
    return R;
  }

  std::optional<Dyadic> compute(Value *V, unsigned Depth) {
    using namespace PatternMatch;
    if (Depth > 32 || !V->getType()->isFloatTy())
      return std::nullopt;
    if (auto *C = dyn_cast<ConstantFP>(V)) {
      const APFloat &F = C->getValueAPF();
      if (!F.isFinite())
        return std::nullopt;
      if (F.isZero()) {
        Dyadic D;
        D.Zero = true;
        return D;
      }
      int Exp = ilogb(F);
      // m 2^(Exp - 23), m the 24-bit significand, then made odd
      APFloat M = scalbn(F, 23 - Exp, APFloat::rmNearestTiesToEven);
      bool Exact;
      APSInt I(64, /*isUnsigned=*/false);
      M.convertToInteger(I, APFloat::rmTowardZero, &Exact);
      if (!Exact)
        return std::nullopt;
      int64_t N = I.getExtValue();
      int E = Exp - 23;
      while (N % 2 == 0) {
        N /= 2;
        ++E;
      }
      if (!fits(N, N, E))
        return std::nullopt;
      return Dyadic{E, N, N, false};
    }
    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return std::nullopt;
    if (isa<UIToFPInst>(I) || isa<SIToFPInst>(I)) {
      Value *X = I->getOperand(0);
      if (!X->getType()->isIntegerTy() ||
          X->getType()->getIntegerBitWidth() > 32)
        return std::nullopt;
      ConstantRange CR = computeConstantRangeIncludingKnownBits(
          X, isa<SIToFPInst>(I), SimplifyQuery(DL));
      int64_t Lo, Hi;
      if (isa<UIToFPInst>(I)) {
        Lo = 0;
        Hi = CR.getUnsignedMax().getZExtValue();
      } else {
        Lo = CR.getSignedMin().getSExtValue();
        Hi = CR.getSignedMax().getSExtValue();
      }
      if (!fits(Lo, Hi, 0))
        return std::nullopt;
      return Dyadic{0, Lo, Hi, false};
    }
    auto Mul = [&](const Dyadic &A, const Dyadic &B) -> std::optional<Dyadic> {
      if (A.Zero || B.Zero) {
        Dyadic D;
        D.Zero = true;
        return D;
      }
      int64_t P[] = {A.Lo * B.Lo, A.Lo * B.Hi, A.Hi * B.Lo, A.Hi * B.Hi};
      Dyadic D{A.E + B.E, *std::min_element(P, P + 4),
               *std::max_element(P, P + 4), false};
      if (!fits(D.Lo, D.Hi, D.E))
        return std::nullopt;
      return D;
    };
    auto Add = [&](const Dyadic &A, const Dyadic &B,
                   bool Sub) -> std::optional<Dyadic> {
      if (B.Zero)
        return A;
      Dyadic Bn = B;
      if (Sub)
        Bn = Dyadic{B.E, -B.Hi, -B.Lo, false};
      if (A.Zero)
        return Bn;
      int E = std::min(A.E, Bn.E);
      if (A.E - E > 24 || Bn.E - E > 24)
        return std::nullopt;
      int64_t SA = int64_t(1) << (A.E - E), SB = int64_t(1) << (Bn.E - E);
      // (each operand scaled to E fits the i32 it is built in)
      auto Small = [](int64_t V) { return V > -(int64_t(1) << 30) && V < (int64_t(1) << 30); };
      if (!Small(A.Lo * SA) || !Small(A.Hi * SA) || !Small(Bn.Lo * SB) ||
          !Small(Bn.Hi * SB))
        return std::nullopt;
      Dyadic D{E, A.Lo * SA + Bn.Lo * SB, A.Hi * SA + Bn.Hi * SB, false};
      if (!fits(D.Lo, D.Hi, D.E))
        return std::nullopt;
      return D;
    };
    Value *X, *Y, *Z;
    if (match(I, m_FMul(m_Value(X), m_Value(Y)))) {
      auto A = get(X, Depth + 1), B = get(Y, Depth + 1);
      return A && B ? Mul(*A, *B) : std::nullopt;
    }
    if (match(I, m_FAdd(m_Value(X), m_Value(Y))) ||
        match(I, m_FSub(m_Value(X), m_Value(Y)))) {
      auto A = get(X, Depth + 1), B = get(Y, Depth + 1);
      return A && B ? Add(*A, *B, I->getOpcode() == Instruction::FSub)
                    : std::nullopt;
    }
    if (match(I, m_Intrinsic<Intrinsic::fma>(m_Value(X), m_Value(Y), m_Value(Z))) ||
        match(I, m_Intrinsic<Intrinsic::fmuladd>(m_Value(X), m_Value(Y),
                                                 m_Value(Z)))) {
      auto A = get(X, Depth + 1), B = get(Y, Depth + 1), C = get(Z, Depth + 1);
      if (!A || !B || !C)
        return std::nullopt;
      auto P = Mul(*A, *B);
      return P ? Add(*P, *C, false) : std::nullopt;
    }
    return std::nullopt;
  }

  /// N of V at exponent E (E at most V's own), as an i32.
  Value *build(IRBuilder<> &B, Value *V, int E) {
    using namespace PatternMatch;
    Dyadic D = *Memo.lookup(V);
    if (D.Zero)
      return ConstantInt::get(Ty, 0);
    auto Scaled = [&](Value *N, int From) {
      return From == E ? N : B.CreateShl(N, From - E);
    };
    if (auto *C = dyn_cast<ConstantFP>(V))
      return Scaled(ConstantInt::get(Ty, D.Lo, /*IsSigned=*/true), D.E);
    auto *I = cast<Instruction>(V);
    if (isa<UIToFPInst>(I) || isa<SIToFPInst>(I)) {
      Value *X = I->getOperand(0);
      Value *N = isa<UIToFPInst>(I) ? B.CreateZExtOrTrunc(X, Ty)
                                    : B.CreateSExtOrTrunc(X, Ty);
      return Scaled(N, 0);
    }
    Value *X, *Y, *Z;
    auto Prod = [&](Value *A, Value *Bv) -> std::pair<Value *, int> {
      Dyadic DA = *Memo.lookup(A), DB = *Memo.lookup(Bv);
      if (DA.Zero || DB.Zero)
        return {ConstantInt::get(Ty, 0), E};
      return {B.CreateMul(build(B, A, DA.E), build(B, Bv, DB.E)), DA.E + DB.E};
    };
    if (match(I, m_FMul(m_Value(X), m_Value(Y)))) {
      auto [N, PE] = Prod(X, Y);
      return Scaled(N, PE);
    }
    if (match(I, m_FAdd(m_Value(X), m_Value(Y))) ||
        match(I, m_FSub(m_Value(X), m_Value(Y)))) {
      Value *NX = build(B, X, E), *NY = build(B, Y, E);
      return I->getOpcode() == Instruction::FSub ? B.CreateSub(NX, NY)
                                                 : B.CreateAdd(NX, NY);
    }
    if (match(I, m_Intrinsic<Intrinsic::fma>(m_Value(X), m_Value(Y), m_Value(Z))) ||
        match(I, m_Intrinsic<Intrinsic::fmuladd>(m_Value(X), m_Value(Y),
                                                 m_Value(Z)))) {
      auto [N, PE] = Prod(X, Y);
      return B.CreateAdd(Scaled(N, PE), build(B, Z, E));
    }
    llvm_unreachable("not a dyadic value");
  }
};
} // namespace

/// fptosi / fptoui (saturating or not) of an exact dyadic sum (DyadicSums):
/// the truncation of N 2^E, in integers, where nothing else uses the float
/// values (a 3 x 3 blur of bytes with weights 1/16, 1/8, 1/4, rounded by
/// + 0.5: (sum of w p + 8) >> 4).
static bool exactDyadicSums(Function &F) {
  if (!AGXExactDyadicSums)
    return false;
  SmallVector<Instruction *, 8> Convs;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (isa<FPToSIInst>(I) || isa<FPToUIInst>(I) ||
        (II && (II->getIntrinsicID() == Intrinsic::fptosi_sat ||
                II->getIntrinsicID() == Intrinsic::fptoui_sat)))
      if (I.getOperand(0)->getType()->isFloatTy() &&
          I.getType()->isIntegerTy() && I.getType()->getIntegerBitWidth() <= 32)
        Convs.push_back(&I);
  }
  bool Changed = false;
  for (Instruction *Conv : Convs) {
    DyadicSums DS(F.getDataLayout());
    Value *V = Conv->getOperand(0);
    std::optional<Dyadic> D = DS.get(V);
    if (!D || D->Zero || !isa<Instruction>(V))
      continue;
    // (only these conversions use the tree's values: it is replaced)
    if (!all_of(DS.Tree, [&](Instruction *T) {
          return all_of(T->users(), [&](User *U) {
            auto *UI = cast<Instruction>(U);
            return UI == Conv || DS.Tree.contains(UI);
          });
        }))
      continue;
    // the truncation of N 2^E toward zero; a saturating one in range
    unsigned W = Conv->getType()->getIntegerBitWidth();
    bool Signed = isa<FPToSIInst>(Conv) ||
                  (isa<IntrinsicInst>(Conv) &&
                   cast<IntrinsicInst>(Conv)->getIntrinsicID() ==
                       Intrinsic::fptosi_sat);
    bool Sat = isa<IntrinsicInst>(Conv);
    if (D->E < 0 && D->Lo < 0)
      continue; // (mixed or negative: not a plain shift)
    if (D->E > 0 && (D->Hi >= (int64_t(1) << (31 - D->E))))
      continue;
    int64_t TLo = D->E >= 0 ? D->Lo << D->E : D->Lo >> -D->E;
    int64_t THi = D->E >= 0 ? D->Hi << D->E : D->Hi >> -D->E;
    if (Sat) {
      int64_t Min = Signed ? -(int64_t(1) << (W - 1)) : 0;
      int64_t Max = Signed ? (int64_t(1) << (W - 1)) - 1
                           : (int64_t(1) << W) - 1;
      if (TLo < Min || THi > Max)
        continue;
    }
    // i16 where every value scaled to the root's exponent (its own for a
    // product's operands, which are not scaled) and the result fit 15 bits
    int64_t Need = std::max(std::abs(TLo), std::abs(THi));
    for (auto &[Val, MD] : DS.Memo)
      if (MD && !MD->Zero) {
        int Shift = std::max(0, MD->E - D->E);
        int64_t M = std::max(std::abs(MD->Lo), std::abs(MD->Hi));
        Need = Shift > 16 ? INT64_MAX : std::max(Need, M << Shift);
      }
    // (the leaves among them: a wider integer whose range fits truncates
    // exactly)
    bool Narrow = AGXExactDyadicI16 && Need < (int64_t(1) << 15);
    DS.Ty = Narrow ? Type::getInt16Ty(F.getContext())
                   : Type::getInt32Ty(F.getContext());
    IRBuilder<> B(Conv);
    Value *N = DS.build(B, V, D->E);
    Value *R = D->E >= 0 ? B.CreateShl(N, D->E) : B.CreateLShr(N, -D->E);
    R = TLo < 0 ? B.CreateSExtOrTrunc(R, Conv->getType())
                : B.CreateZExtOrTrunc(R, Conv->getType());
    Conv->replaceAllUsesWith(R);
    Conv->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(V);
    Changed = true;
  }
  // An exact tree whose value stays a float (a box filter's sum of bytes,
  // then scaled by 1/9): N in integers, one conversion, times 2^E (exact:
  // |N| < 2^24, 2^-126 <= 2^E). Of non-negative values only (no fsub, every
  // node at least 0): the float tree's zeros are +0, as the conversion's;
  // with two converted leaves or more (L15-E407).
  if (!AGXExactDyadicFloatRoots)
    return Changed;
  SmallVector<Instruction *, 16> Cands;
  for (Instruction &I : instructions(F))
    if (I.getType()->isFloatTy() &&
        (I.getOpcode() == Instruction::FAdd ||
         I.getOpcode() == Instruction::FMul ||
         match(&I, PatternMatch::m_Intrinsic<Intrinsic::fma>()) ||
         match(&I, PatternMatch::m_Intrinsic<Intrinsic::fmuladd>())))
      Cands.push_back(&I);
  SmallVector<WeakTrackingVH, 16> Work(Cands.begin(), Cands.end());
  for (WeakTrackingVH &WV : Work) {
    auto *Root = dyn_cast_or_null<Instruction>(WV);
    if (!Root)
      continue;
    DyadicSums DS(F.getDataLayout());
    std::optional<Dyadic> D = DS.get(Root);
    if (!D || D->Zero || D->Lo < 0)
      continue;
    // a root: some user is not part of an exact tree
    bool IsRoot = any_of(Root->users(), [&](User *U) {
      DyadicSums UD(F.getDataLayout());
      return !UD.get(U);
    });
    if (!IsRoot)
      continue;
    unsigned Leaves = 0;
    bool NonNeg = true;
    for (Instruction *T : DS.Tree) {
      if (isa<UIToFPInst, SIToFPInst>(T))
        ++Leaves;
      if (T->getOpcode() == Instruction::FSub)
        NonNeg = false;
      std::optional<Dyadic> TD = DS.Memo.lookup(T);
      if (!TD || (!TD->Zero && TD->Lo < 0))
        NonNeg = false;
    }
    for (auto &[Val, MD] : DS.Memo)
      if (isa<ConstantFP>(Val) && MD && !MD->Zero && MD->Lo < 0)
        NonNeg = false;
    if (!NonNeg || Leaves < 2 ||
        !all_of(DS.Tree, [&](Instruction *T) {
          return T == Root || all_of(T->users(), [&](User *U) {
                   return DS.Tree.contains(cast<Instruction>(U));
                 });
        }))
      continue;
    int64_t Need = D->Hi;
    for (auto &[Val, MD] : DS.Memo)
      if (MD && !MD->Zero) {
        int Shift = std::max(0, MD->E - D->E);
        int64_t M = std::max(std::abs(MD->Lo), std::abs(MD->Hi));
        Need = Shift > 16 ? INT64_MAX : std::max(Need, M << Shift);
      }
    // (the leaves among them: a wider integer whose range fits truncates
    // exactly)
    bool Narrow = AGXExactDyadicI16 && Need < (int64_t(1) << 15);
    DS.Ty = Narrow ? Type::getInt16Ty(F.getContext())
                   : Type::getInt32Ty(F.getContext());
    IRBuilder<> B(Root->getNextNode());
    Value *N = DS.build(B, Root, D->E);
    Value *R = B.CreateUIToFP(N, Root->getType());
    if (D->E)
      R = B.CreateFMul(R, ConstantFP::get(Root->getType(),
                                          std::ldexp(1.0, D->E)));
    // (N and R are built from the leaves, not from the root)
    Root->replaceAllUsesWith(R);
    RecursivelyDeleteTriviallyDeadInstructions(Root);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXNarrowBoundedSubMinMax(
    "agx-narrow-bounded-sub-minmax", cl::init(true), cl::Hidden,
    cl::desc("Also subtractions that cannot borrow and umin / umax in "
             "agx-narrow-bounded-i64's trees"));

static cl::opt<bool> AGXNarrowBoundedI64(
    "agx-narrow-bounded-i64", cl::init(true), cl::Hidden,
    cl::desc("64-bit arithmetic of zero-extended 32-bit values that stays "
             "below 2^32 (positions' known bounds) computed in 32 bits"));

/// MLIR's index arithmetic of positions is 64-bit: zext(by) << 16 +
/// ((zext(sg) << 14) & m) | ... Where every node of such a tree (zext of
/// i32 leaves, constants; shl / add / or / and / mul / lshr) is below 2^32
/// (AGX::scevUpperBound: the positions' bounds, E391's upper_bounds among
/// them), it is the zero extension of the same tree in i32: no 64-bit
/// shifts, masks and adds (L15-E405). AGXNarrowInductions runs it again
/// once the counters are narrowed (L15-E426). (Not as range attributes on the
/// positions, which the DAG's address folds then use: E391.)
bool AGX::narrowBoundedI64(Function &F) {
  if (!AGXNarrowBoundedI64)
    return false;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII, &F);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  Type *I64 = Type::getInt64Ty(F.getContext());
  Type *I32 = Type::getInt32Ty(F.getContext());
  // an upper bound on each node: constants, zext leaves by scalar evolution
  // (the positions' bounds), and the operations' own (and: the smaller
  // operand's; or: the sum's)
  DenseMap<Value *, std::optional<APInt>> BoundOf;
  std::function<std::optional<APInt>(Value *, unsigned)> Bound =
      [&](Value *V, unsigned Depth) -> std::optional<APInt> {
    if (V->getType() != I64 || Depth > 12)
      return std::nullopt;
    if (auto It = BoundOf.find(V); It != BoundOf.end())
      return It->second;
    std::optional<APInt> R;
    APInt Top = APInt(128, 1).shl(32);
    if (auto *C = dyn_cast<ConstantInt>(V)) {
      if (C->getValue().isIntN(32))
        R = C->getValue().zext(128);
    } else if (auto *Z = dyn_cast<ZExtInst>(V)) {
      if (Z->getSrcTy()->isIntegerTy(32)) {
        std::optional<APInt> B =
            AGX::scevUpperBound(SE.getSCEV(Z->getOperand(0)), SE, F);
        R = B && B->ult(Top) ? *B : Top - 1;
      }
    } else if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      std::optional<APInt> A = Bound(BO->getOperand(0), Depth + 1);
      std::optional<APInt> C = Bound(BO->getOperand(1), Depth + 1);
      auto *CI = dyn_cast<ConstantInt>(BO->getOperand(1));
      if (A && C) {
        switch (BO->getOpcode()) {
        case Instruction::Add:
        case Instruction::Or:
          R = *A + *C;
          break;
        case Instruction::And:
          R = A->ult(*C) ? *A : *C;
          break;
        case Instruction::Mul:
          R = *A * *C;
          break;
        case Instruction::Shl:
          if (CI && CI->getZExtValue() < 32)
            R = A->shl(CI->getZExtValue());
          break;
        case Instruction::LShr:
          if (CI && CI->getZExtValue() < 32)
            R = A->lshr(CI->getZExtValue());
          break;
        case Instruction::Sub:
          // (no borrow: a's least value at least b's bound; L15-E435)
          if (AGXNarrowBoundedSubMinMax &&
              SE.getUnsignedRangeMin(SE.getSCEV(BO->getOperand(0)))
                  .zext(128)
                  .uge(*C))
            R = *A;
          break;
        default:
          break;
        }
      }
      if (R && !R->ult(Top))
        R.reset();
    } else if (auto *II = dyn_cast<IntrinsicInst>(V);
               II && AGXNarrowBoundedSubMinMax &&
               (II->getIntrinsicID() == Intrinsic::umin ||
                II->getIntrinsicID() == Intrinsic::umax)) {
      std::optional<APInt> A = Bound(II->getArgOperand(0), Depth + 1);
      std::optional<APInt> C = Bound(II->getArgOperand(1), Depth + 1);
      if (A && C)
        R = (II->getIntrinsicID() == Intrinsic::umin) == A->ult(*C) ? *A : *C;
    }
    BoundOf[V] = R;
    return R;
  };
  auto Narrowable = [&](Value *V, unsigned) { return Bound(V, 0).has_value(); };
  DenseMap<Value *, Value *> Built;
  std::function<Value *(Value *, IRBuilder<> &)> Build = [&](Value *V,
                                                             IRBuilder<> &B) {
    if (Value *N = Built.lookup(V))
      return N;
    Value *N;
    if (auto *C = dyn_cast<ConstantInt>(V))
      N = ConstantInt::get(I32, C->getZExtValue());
    else if (auto *Z = dyn_cast<ZExtInst>(V))
      N = Z->getOperand(0);
    else if (auto *II = dyn_cast<IntrinsicInst>(V)) {
      IRBuilder<> IB(II);
      Value *A = Build(II->getArgOperand(0), IB),
            *C = Build(II->getArgOperand(1), IB);
      // (umin(a, c) of an a never above the constant c is a: a trip
      // umin(n - lane, n))
      auto *CC = dyn_cast<ConstantInt>(II->getArgOperand(1));
      std::optional<APInt> BA = Bound(II->getArgOperand(0), 0);
      if (II->getIntrinsicID() == Intrinsic::umin && CC && BA &&
          BA->ule(CC->getValue().zext(128)))
        N = A;
      else
        N = IB.CreateBinaryIntrinsic(II->getIntrinsicID(), A, C, nullptr,
                                     II->getName() + ".n32");
    } else {
      auto *BO = cast<BinaryOperator>(V);
      IRBuilder<> IB(BO);
      Value *A = Build(BO->getOperand(0), IB), *C = Build(BO->getOperand(1), IB);
      // (no wrap: every node is below 2^32)
      auto *NB = BinaryOperator::Create(BO->getOpcode(), A, C,
                                        BO->getName() + ".n32", BO->getIterator());
      if (isa<OverflowingBinaryOperator>(NB))
        NB->setHasNoUnsignedWrap(true);
      if (auto *PD = dyn_cast<PossiblyDisjointInst>(NB))
        PD->setIsDisjoint(cast<PossiblyDisjointInst>(BO)->isDisjoint());
      N = NB;
    }
    Built[V] = N;
    return N;
  };
  SmallVector<Instruction *, 16> Roots;
  for (Instruction &I : instructions(F)) {
    Instruction *BO = &I;
    if ((!isa<BinaryOperator>(BO) && !isa<IntrinsicInst>(BO)) ||
        BO->getType() != I64 || !Narrowable(BO, 0))
      continue;
    // a root: some user is not narrowable itself; not an address's index,
    // a GEP's or an integer address's (b + x, the PTX front end's): the DAG
    // folds zext(x) << s into the access's scaled index
    auto AddressUse = [](User *U) {
      if (isa<GetElementPtrInst, IntToPtrInst>(U))
        return true;
      auto *A = dyn_cast<BinaryOperator>(U);
      return A && (A->getOpcode() == Instruction::Add ||
                   A->getOpcode() == Instruction::Or ||
                   A->getOpcode() == Instruction::Sub);
    };
    if (any_of(BO->users(), [&](User *U) { return !Narrowable(U, 0); }) &&
        none_of(BO->users(), AddressUse))
      Roots.push_back(BO);
  }
  bool Changed = false;
  for (Instruction *R : Roots) {
    IRBuilder<> B(R);
    Value *N = Build(R, B);
    IRBuilder<> ZB(R->getNextNode());
    Value *Z = ZB.CreateZExt(N, I64, R->getName() + ".z");
    if (auto *ZI = dyn_cast<ZExtInst>(Z))
      ZI->setNonNeg(false);
    R->replaceUsesWithIf(Z, [&](Use &U) {
      return !Narrowable(U.getUser(), 0) || U.getUser() == Z;
    });
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXNeverNaNTests(
    "agx-never-nan-tests", cl::init(true), cl::Hidden,
    cl::desc("Fold NaN tests of values no execution makes a NaN (a running "
             "maximumnum from a number, through its loop's phi)"));

namespace {
/// V is never a NaN, given that the phis in Assumed are not: a number
/// constant; maximumnum / minimumnum with an operand never a NaN (they give
/// a NaN only where both are); canonicalize, fneg, fabs or copysign of one;
/// a select on a NaN test of values never a NaN whose picked arm is; a
/// select or phi all of whose values are; or what value tracking knows. A
/// phi is assumed while its values are checked: every value a phi takes
/// was computed before, so if no value of the set's phis is a NaN where
/// those phis' are not, none ever is. Proven values are remembered until an
/// assumption fails (an unrolled running maximum is a chain of eight
/// maximumnums and selects, each select's test reading the value before).
struct NeverNaN {
  const SimplifyQuery &Q;
  SmallPtrSet<PHINode *, 8> Assumed;
  SmallPtrSet<Value *, 32> Proven;
  unsigned Budget = 256;
  explicit NeverNaN(const SimplifyQuery &Q) : Q(Q) {}

  bool operator()(Value *V) {
    if (Proven.contains(V))
      return true;
    if (!Budget || !V->getType()->isFPOrFPVectorTy())
      return false;
    --Budget;
    if (!check(V))
      return false;
    Proven.insert(V);
    return true;
  }

  bool check(Value *V) {
    using namespace PatternMatch;
    NeverNaN &N = *this;
    if (auto *C = dyn_cast<ConstantFP>(V))
      return !C->isNaN();
    if (auto *P = dyn_cast<PHINode>(V)) {
      if (Assumed.contains(P))
        return true;
      Assumed.insert(P);
      for (Value *In : P->incoming_values())
        if (!N(In)) {
          Assumed.erase(P);
          Proven.clear();
          return false;
        }
      return true;
    }
    if (auto *II = dyn_cast<IntrinsicInst>(V)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::maximumnum:
      case Intrinsic::minimumnum:
        return N(II->getArgOperand(0)) || N(II->getArgOperand(1));
      case Intrinsic::canonicalize:
      case Intrinsic::fabs:
      case Intrinsic::copysign:
        return N(II->getArgOperand(0));
      default:
        break;
      }
    }
    if (auto *U = dyn_cast<UnaryOperator>(V);
        U && U->getOpcode() == Instruction::FNeg)
      return N(U->getOperand(0));
    if (auto *Sel = dyn_cast<SelectInst>(V)) {
      Value *X, *Y;
      CmpPredicate Pred;
      // on a NaN test of values never a NaN: the arm it picks (as
      // select(uno x, NaN, x), the default NaN where x is one, and what
      // InstCombine makes of a maximumnum of it, select(uno x, y,
      // maximumnum(x, y)))
      if (match(Sel->getCondition(), m_FCmp(Pred, m_Value(X), m_Value(Y))) &&
          (Pred == FCmpInst::FCMP_UNO || Pred == FCmpInst::FCMP_ORD) &&
          N(X) && (Y == X || N(Y)))
        return N(Pred == FCmpInst::FCMP_UNO ? Sel->getFalseValue()
                                            : Sel->getTrueValue());
      return N(Sel->getTrueValue()) && N(Sel->getFalseValue());
    }
    return isKnownNeverNaN(V,
                           Q.getWithInstruction(dyn_cast<Instruction>(V)));
  }
};
} // namespace

/// fcmp uno / ord of values that are never a NaN (NeverNaN): false / true,
/// and the selects on them their arm. The PTX front end's max.f32 is
/// maximumnum then a select of the default NaN where it is one: in a loop's
/// running maximum from a number (softmax_online, attn_decode) it never is
/// (L15-E388).
static bool foldNeverNaNTests(Function &F) {
  if (!AGXNeverNaNTests)
    return false;
  SimplifyQuery Q(F.getDataLayout());
  SmallVector<FCmpInst *, 8> Tests;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<FCmpInst>(&I);
        C && (C->getPredicate() == FCmpInst::FCMP_UNO ||
              C->getPredicate() == FCmpInst::FCMP_ORD))
      Tests.push_back(C);
  bool Changed = false;
  for (FCmpInst *C : Tests) {
    NeverNaN N(Q);
    if (!N(C->getOperand(0)) || !N(C->getOperand(1)))
      continue;
    Constant *K = ConstantInt::getBool(
        C->getType(), C->getPredicate() == FCmpInst::FCMP_ORD);
    for (User *U : make_early_inc_range(C->users()))
      if (auto *Sel = dyn_cast<SelectInst>(U);
          Sel && Sel->getCondition() == C) {
        Value *Arm = K->isOneValue() ? Sel->getTrueValue()
                                     : Sel->getFalseValue();
        Sel->replaceAllUsesWith(Arm);
        Sel->eraseFromParent();
      }
    C->replaceAllUsesWith(K);
    C->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXZeroSignFreeMinMax(
    "agx-zero-sign-free-minmax", cl::init(true), cl::Hidden,
    cl::desc("Mark a float min / max nsz where nothing its value reaches "
             "depends on the sign of a zero (lowered without its zero fix)"));

static Value *orderedSIMDMaxOf(Value *M);

namespace {
/// Every use of a value (through the values it reaches) gives the same
/// result for +0 and -0 in its place, or a result differing at most in the
/// sign of a zero: a compare; an exponential (e^+-0 = 1); fabs; the
/// magnitude of a copysign; or, passing it on, a phi, a select's arm,
/// fneg, fadd, fsub, fmul, fma, canonicalize or a min / max (each, given
/// zeros of two signs in an operand, differs at most in its result's zero
/// sign). A phi or select is assumed while its own uses are checked (the
/// differences then circulate only among such values).
/// The threadgroup memory accesses of a function, each with the
/// threadgroup global it stays within (or none: it may access any), when no
/// other instruction can read or write threadgroup memory (no calls but
/// intrinsics that do not access memory, no generic accesses, no atomics).
struct ThreadgroupAccesses {
  bool Valid = false;
  SmallVector<std::pair<LoadInst *, GlobalVariable *>, 8> Loads;
  std::optional<DominatorTree> DT;
  DomConditionCache DC;
  std::optional<LoopInfo> LI;
  std::optional<TargetLibraryInfoImpl> TLII;
  std::optional<TargetLibraryInfo> TLI;
  std::optional<AssumptionCache> AC;
  std::optional<ScalarEvolution> SE;
  SmallVector<GlobalVariable *, 4> Globals;

  /// The query for an access at I: known bits from the conditions of the
  /// branches that dominate it (a store under lane == 0 at (tid >> 5) * 4).
  SimplifyQuery query(Instruction *I) {
    return SimplifyQuery(I->getDataLayout(), /*TLI=*/nullptr, &*DT,
                         /*AC=*/nullptr, I, /*UseInstrInfo=*/true,
                         /*CanUseUndef=*/true, &DC);
  }

  /// The global a threadgroup pointer accessed by Size bytes stays within:
  /// G + c (constant GEPs), or inttoptr(X + ptrtoint G) + c with X's known
  /// unsigned range in bounds.
  static bool inBounds(GlobalVariable *G, const ConstantRange &R,
                       uint64_t Size, const DataLayout &DL) {
    if (G->getAddressSpace() != 3 || !G->getValueType()->isSized())
      return false;
    uint64_t GSize = DL.getTypeAllocSize(G->getValueType());
    return GSize >= Size && !R.isWrappedSet() &&
           R.getUnsignedMax().ule(GSize - Size);
  }

  /// The global a threadgroup pointer accessed by Size bytes at At stays
  /// within: G + c (constant GEPs), or inttoptr(X) + c where X is
  /// X' + ptrtoint G with X''s known unsigned range in bounds, or where X's
  /// evolution less ptrtoint G is bounded in bounds (a loop's pointer).
  GlobalVariable *within(Value *P, uint64_t Size, Instruction *At) {
    SimplifyQuery SQ = query(At);
    const DataLayout &DL = SQ.DL;
    // The offsets of GEPs, constant or by indices of known signed ranges.
    ConstantRange R(APInt(64, 0));
    for (unsigned Depth = 0; Depth < 8; ++Depth) {
      APInt Off(DL.getIndexTypeSizeInBits(P->getType()), 0);
      P = P->stripAndAccumulateConstantOffsets(DL, Off,
                                               /*AllowNonInbounds=*/true);
      R = R.add(ConstantRange(Off.sextOrTrunc(64)));
      auto *GEP = dyn_cast<GEPOperator>(P);
      if (!GEP)
        break;
      unsigned W = DL.getIndexTypeSizeInBits(GEP->getType());
      SmallMapVector<Value *, APInt, 4> Vars;
      APInt C(W, 0);
      if (!GEP->collectOffset(DL, W, Vars, C))
        return nullptr;
      R = R.add(ConstantRange(C.sextOrTrunc(64)));
      for (auto &[V, Scale] : Vars) {
        if (!V->getType()->isIntegerTy())
          return nullptr;
        ConstantRange VR = computeConstantRangeIncludingKnownBits(
                               V, /*ForSigned=*/true, SQ)
                               .sextOrTrunc(W)
                               .sextOrTrunc(64);
        R = R.add(VR.multiply(ConstantRange(Scale.sextOrTrunc(64))));
      }
      P = GEP->getPointerOperand();
    }
    if (auto *G = dyn_cast<GlobalVariable>(P))
      return inBounds(G, R, Size, DL) ? G : nullptr;
    auto *I2P = dyn_cast<IntToPtrInst>(P);
    if (!I2P || !I2P->getSrcTy()->isIntegerTy(64))
      return nullptr;
    using namespace PatternMatch;
    Value *X, *GP;
    if (match(I2P->getOperand(0), m_c_Add(m_Value(X), m_PtrToInt(m_Value(GP)))))
      if (auto *G = dyn_cast<GlobalVariable>(GP->stripPointerCasts()))
        if (inBounds(G,
                     R.add(computeConstantRangeIncludingKnownBits(
                         X, /*ForSigned=*/false, SQ)),
                     Size, DL))
          return G;
    const SCEV *S = SE->getSCEV(I2P->getOperand(0));
    for (GlobalVariable *G : Globals) {
      // (G's address as the IR's ptrtoint gives it)
      const SCEV *D = SE->getMinusSCEV(
          S, SE->getSCEV(ConstantExpr::getPtrToInt(G, S->getType())));
      if (SCEVExprContains(D, [](const SCEV *E) {
            return isa<SCEVCastExpr>(E) &&
                       !isa<SCEVIntegralCastExpr>(E) ||
                   isa<SCEVUnknown>(E) &&
                       !isa<Instruction>(cast<SCEVUnknown>(E)->getValue()) &&
                       !isa<Argument>(cast<SCEVUnknown>(E)->getValue());
          }))
        continue;
      std::optional<APInt> Hi = AGX::scevUpperBound(D, *SE, *At->getFunction());
      APInt Lo = SE->getUnsignedRange(D).getUnsignedMin();
      if (!Hi || Hi->getActiveBits() > 63 || Lo.zext(128).ugt(*Hi))
        continue;
      ConstantRange DR(Lo.zextOrTrunc(64), Hi->trunc(64) + 1);
      if (inBounds(G, R.add(DR), Size, DL))
        return G;
    }
    return nullptr;
  }

  void build(Function &F) {
    const DataLayout &DL = F.getDataLayout();
    DT.emplace(F);
    for (BasicBlock &BB : F)
      if (auto *Br = dyn_cast<CondBrInst>(BB.getTerminator()))
        DC.registerBranch(Br);
    LI.emplace(*DT);
    TLII.emplace(F.getParent()->getTargetTriple());
    TLI.emplace(*TLII, &F);
    AC.emplace(F);
    SE.emplace(F, *TLI, *AC, *DT, *LI);
    for (GlobalVariable &G : F.getParent()->globals())
      if (G.getAddressSpace() == 3)
        Globals.push_back(&G);
    for (Instruction &I : instructions(F)) {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        auto *II = dyn_cast<IntrinsicInst>(CB);
        if (!II || isa<MemIntrinsic>(II) || II->mayReadOrWriteMemory())
          if (!II || II->getIntrinsicID() != Intrinsic::agx_threadgroup_barrier)
            return;
        continue;
      }
      if (!I.mayReadOrWriteMemory())
        continue;
      if (isa<FenceInst>(I))
        continue;
      Value *Ptr = getLoadStorePointerOperand(&I);
      if (!Ptr)
        return; // atomics and the like
      unsigned AS = Ptr->getType()->getPointerAddressSpace();
      if (AS == 3) {
        if (auto *L = dyn_cast<LoadInst>(&I))
          Loads.push_back(
              {L, within(Ptr, DL.getTypeStoreSize(L->getType()), L)});
      } else if (AS != 1 && AS != 2 && AS != 4 && AS != 5) {
        return; // a generic access may reach threadgroup memory
      }
    }
    Valid = true;
  }
};

static cl::opt<bool> AGXZeroSignFreeOrderedMax(
    "agx-zero-sign-free-ordered-max", cl::init(true), cl::Hidden,
    cl::desc("A value whose bits only make MLIR's maximumf subgroup "
             "reduction key keeps no zero sign if the decoded maximum does "
             "not"));

static cl::opt<bool> AGXZeroSignFreeSums(
    "agx-zero-sign-free-sums", cl::init(true), cl::Hidden,
    cl::desc("A sum whose other operand is never the zero that would tell "
             "the signs apart keeps no zero sign of the operand"));

struct ZeroSignFree {
  SmallPtrSet<Instruction *, 16> Assumed;
  unsigned Budget = 256;
  /// The function's threadgroup accesses, built at the first threadgroup
  /// store (shared by the min / max of a function: setting nsz changes none).
  std::optional<ThreadgroupAccesses> &TG;
  explicit ZeroSignFree(std::optional<ThreadgroupAccesses> &TG) : TG(TG) {}

  /// A threadgroup store of the value: every load that may read it (one in
  /// a global the store may write) has a value free of zero signs.
  bool storedValue(StoreInst *S) {
    if (S->getPointerAddressSpace() != 3 || S->isVolatile() || !S->isSimple())
      return false;
    Function &F = *S->getFunction();
    if (!TG) {
      TG.emplace();
      TG->build(F);
    }
    if (!TG->Valid)
      return false;
    const DataLayout &DL = F.getDataLayout();
    GlobalVariable *G = TG->within(
        S->getPointerOperand(),
        DL.getTypeStoreSize(S->getValueOperand()->getType()), S);
    for (auto [L, LG] : TG->Loads)
      if (!G || !LG || G == LG)
        if (!value(L))
          return false;
    return true;
  }

  static bool isShuffle(Intrinsic::ID ID) {
    switch (ID) {
    case Intrinsic::agx_simd_shuffle:
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_simd_shuffle_rotate_up:
    case Intrinsic::agx_quad_shuffle:
    case Intrinsic::agx_quad_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_up:
    case Intrinsic::agx_quad_shuffle_down:
    case Intrinsic::agx_quad_shuffle_rotate_up:
      return true;
    default:
      return false;
    }
  }

  /// The bits of a float as an integer that only moves between lanes
  /// (shuffles) and becomes a float again: those floats' values.
  bool movedBits(Instruction *I, unsigned Depth = 0) {
    if (Depth > 4)
      return false;
    for (User *U : I->users()) {
      auto *UI = cast<Instruction>(U);
      if (auto *BC = dyn_cast<BitCastInst>(UI)) {
        if (!BC->getType()->isFloatingPointTy() || !value(BC))
          return false;
        continue;
      }
      auto *II = dyn_cast<IntrinsicInst>(UI);
      if (!II || !isShuffle(II->getIntrinsicID()) || II->getArgOperand(0) != I ||
          II->getArgOperand(1) == I || !movedBits(II, Depth + 1))
        return false;
    }
    return true;
  }

  /// bits(m) only in MLIR's maximumf subgroup reduction key (the xor of
  /// its sign mask, the NaN select) into the SIMD-group's signed maximum,
  /// whose decoded value M (orderedSIMDMaxOf) is free of zero signs: m's
  /// zero sign only orders -0 below +0 there (L15-E477).
  bool orderedMaxKey(BitCastInst *BC) {
    if (!AGXZeroSignFreeOrderedMax)
      return false;
    Value *Src = BC->getOperand(0);
    // every user of the bits within the key's compare / select / xor, up
    // to the reduction
    SmallVector<Instruction *, 8> Work{BC};
    SmallPtrSet<Instruction *, 8> Seen{BC};
    IntrinsicInst *R = nullptr;
    while (!Work.empty()) {
      Instruction *V = Work.pop_back_val();
      for (User *U : V->users()) {
        auto *UI = cast<Instruction>(U);
        if (auto *II = dyn_cast<IntrinsicInst>(UI);
            II && II->getIntrinsicID() == Intrinsic::agx_simd_reduce) {
          if (R && R != II)
            return false;
          R = II;
          continue;
        }
        if (!(isa<ICmpInst>(UI) || isa<SelectInst>(UI) ||
              (UI->getOpcode() == Instruction::Xor)) ||
            Seen.size() > 8)
          return false;
        if (Seen.insert(UI).second)
          Work.push_back(UI);
      }
    }
    if (!R)
      return false;
    // the decoded maxima (bitcasts to float of the reduction's order
    // inverted back)
    bool Found = false;
    SmallPtrSet<Instruction *, 8> Near;
    std::function<void(Instruction *, unsigned)> Collect =
        [&](Instruction *V, unsigned D) {
          for (User *U : V->users())
            if (Near.insert(cast<Instruction>(U)).second && D < 3)
              Collect(cast<Instruction>(U), D + 1);
        };
    Collect(R, 1);
    for (Instruction *N : Near)
      if (auto *M = dyn_cast<BitCastInst>(N);
          M && orderedSIMDMaxOf(M) == Src) {
        if (!value(M))
          return false;
        Found = true;
      }
    return Found;
  }

  bool value(Instruction *I) {
    if (Assumed.contains(I))
      return true;
    if (!Budget)
      return false;
    --Budget;
    Assumed.insert(I);
    for (Use &U : I->uses())
      if (!use(U)) {
        Assumed.erase(I);
        return false;
      }
    return true;
  }

  bool use(Use &U) {
    auto *I = cast<Instruction>(U.getUser());
    if (isa<FCmpInst>(I))
      return true;
    if (isa<PHINode>(I) || isa<ExtractElementInst>(I))
      return isa<PHINode>(I) || U.getOperandNo() == 0 ? value(I) : false;
    if (auto *S = dyn_cast<StoreInst>(I))
      return U.getOperandNo() == 0 && storedValue(S);
    if (auto *BC = dyn_cast<BitCastInst>(I))
      return BC->getType()->isIntegerTy() &&
             BC->getSrcTy()->isFloatingPointTy() &&
             BC->getType()->getScalarSizeInBits() ==
                 BC->getSrcTy()->getScalarSizeInBits() &&
             !BC->getType()->isVectorTy() &&
             (movedBits(BC) || orderedMaxKey(BC));
    if (auto *Sel = dyn_cast<SelectInst>(I))
      return U.getOperandNo() != 0 && value(Sel);
    // a + b, a's zero either sign: (+-0) + b is b's value then unless b is
    // -0, and b - (+-0) likewise; (+-0) - b unless b is +0. A log-sum-exp's
    // m + log(s), s >= 1 (L15-E475).
    if (AGXZeroSignFreeSums &&
        (I->getOpcode() == Instruction::FAdd ||
         I->getOpcode() == Instruction::FSub)) {
      Value *Other = I->getOperand(1 - U.getOperandNo());
      FPClassTest Never = I->getOpcode() == Instruction::FSub &&
                                  U.getOperandNo() == 0
                              ? fcPosZero
                              : fcNegZero;
      if (computeKnownFPClass(Other, I->getDataLayout(), Never)
              .isKnownNever(Never))
        return true;
    }
    switch (I->getOpcode()) {
    case Instruction::FNeg:
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
      return value(I);
    default:
      break;
    }
    auto *II = dyn_cast<IntrinsicInst>(I);
    if (!II)
      return false;
    switch (II->getIntrinsicID()) {
    case Intrinsic::exp:
    case Intrinsic::exp2:
    case Intrinsic::exp10:
    case Intrinsic::agx_exp2:
    case Intrinsic::fabs:
      return true;
    case Intrinsic::copysign:
      return U.getOperandNo() == 0;
    case Intrinsic::agx_simd_shuffle:
    case Intrinsic::agx_simd_shuffle_xor:
    case Intrinsic::agx_simd_shuffle_up:
    case Intrinsic::agx_simd_shuffle_down:
    case Intrinsic::agx_simd_shuffle_rotate_up:
    case Intrinsic::agx_quad_shuffle:
    case Intrinsic::agx_quad_shuffle_xor:
    case Intrinsic::agx_quad_shuffle_up:
    case Intrinsic::agx_quad_shuffle_down:
    case Intrinsic::agx_quad_shuffle_rotate_up:
      // The value moved to another lane (a float shuffle).
      return U.getOperandNo() == 0 && value(II);
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
    case Intrinsic::canonicalize:
    case Intrinsic::maxnum:
    case Intrinsic::minnum:
    case Intrinsic::maximum:
    case Intrinsic::minimum:
    case Intrinsic::maximumnum:
    case Intrinsic::minimumnum:
      return value(II);
    default:
      return false;
    }
  }
};
} // namespace

/// A float min / max whose value reaches nothing that depends on the sign
/// of a zero (ZeroSignFree) is nsz: lowered without the fix that orders -0
/// below +0 (an and / or of the operands' bits, its flush and a select on
/// their equality: three of four instructions). A running maximum that only
/// offsets an exponential, e^(x - m): attn_decode's (L15-E389).
static bool zeroSignFreeMinMax(Function &F) {
  if (!AGXZeroSignFreeMinMax)
    return false;
  // The special registers' ranges for the analysis only (threadgroup
  // offsets within their globals, rangedCompares'), taken off again.
  SmallVector<WeakVH, 8> Ranged;
  AGX::addSpecialRegisterRanges(F, &Ranged);
  scope_exit Unrange([&] {
    for (WeakVH &V : Ranged)
      if (auto *CI = dyn_cast_or_null<CallBase>(V))
        CI->removeRetAttr(Attribute::Range);
  });
  std::optional<ThreadgroupAccesses> TG;
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !II->getType()->isFloatingPointTy() ||
        II->hasNoSignedZeros())
      continue;
    switch (II->getIntrinsicID()) {
    case Intrinsic::maxnum:
    case Intrinsic::minnum:
    case Intrinsic::maximum:
    case Intrinsic::minimum:
    case Intrinsic::maximumnum:
    case Intrinsic::minimumnum:
      break;
    default:
      continue;
    }
    ZeroSignFree Z(TG);
    if (!Z.value(II))
      continue;
    II->setHasNoSignedZeros(true);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXMergePhiOperations(
    "agx-merge-phi-operations", cl::init(true), cl::Hidden,
    cl::desc("A phi of the same binary operation of the same operands from "
             "each predecessor as that one operation in the phi's block"));

/// phi [x op y, A], [x op y, B] is x op y in the phi's block, with the
/// flags both had (each original goes where the phi was its only user): x
/// and y dominate every predecessor, so the join. InstCombine's phi folding, which llc does not run: the PTX
/// front end's x + 1 on both sides of a branch (jacobi2d), each then its own
/// register and a copy at the join (L15-E421).
static bool mergePhiOperations(Function &F) {
  if (!AGXMergePhiOperations)
    return false;
  bool Changed = false;
  for (BasicBlock &BB : F)
    for (PHINode &P : make_early_inc_range(BB.phis())) {
      if (P.getNumIncomingValues() < 2)
        continue;
      auto *First = dyn_cast<BinaryOperator>(P.getIncomingValue(0));
      // (not a division: one with other uses would be computed twice)
      if (!First || First->isIntDivRem() ||
          First->getOpcode() == Instruction::FDiv ||
          First->getOpcode() == Instruction::FRem)
        continue;
      bool Same = true;
      for (Value *V : P.incoming_values()) {
        auto *B = dyn_cast<BinaryOperator>(V);
        Same &= B && B->getOpcode() == First->getOpcode() &&
                B->getOperand(0) == First->getOperand(0) &&
                B->getOperand(1) == First->getOperand(1) &&
                B->getParent() != &BB;
      }
      // (two distinct instructions at least; one may come in on several
      // edges)
      SmallPtrSet<Value *, 4> Distinct(P.incoming_values().begin(),
                                       P.incoming_values().end());
      if (!Same || Distinct.size() < 2)
        continue;
      auto *N = BinaryOperator::Create(First->getOpcode(),
                                       First->getOperand(0),
                                       First->getOperand(1), P.getName(),
                                       BB.getFirstInsertionPt());
      N->copyIRFlags(First);
      for (Value *V : P.incoming_values())
        N->andIRFlags(V);
      P.replaceAllUsesWith(N);
      SmallVector<Instruction *, 4> Old;
      for (Value *V : Distinct)
        Old.push_back(cast<Instruction>(V));
      P.eraseFromParent();
      // (one that has other uses stays)
      for (Instruction *I : Old)
        if (I->use_empty())
          I->eraseFromParent();
      Changed = true;
    }
  return Changed;
}

static cl::opt<bool> AGXNarrowSignedCompares(
    "agx-narrow-signed-compares", cl::init(true), cl::Hidden,
    cl::desc("A 64-bit compare of values within 32 signed bits as the "
             "compare of their 32-bit truncations"));

/// An i64 compare of two values in [-2^31, 2^31) is the same compare of
/// their truncations to i32: on that range x -> x mod 2^32 keeps equality,
/// the signed order, and the unsigned one (both put [0, 2^31) first, then
/// the negatives in their order). A value is in range if it is a sext of an
/// i32, a constant in range, an and / or / xor of such values (bitwise
/// operations of sign extensions are sign extensions), or a value whose
/// signed range (known bits, the special registers' bounds, dominating
/// conditions) is. MLIR's 64-bit bounds checks of a padded stencil's indices,
/// x + d - 1 in [0, W) for x a bounded position, which InstCombine makes
/// (x' | y') <u W: 32-bit compares, and the DAG narrows the truncated
/// arithmetic (L15-E427).
static bool narrowSignedCompares(Function &F) {
  if (!AGXNarrowSignedCompares)
    return false;
  SmallVector<ICmpInst *, 8> Cmps;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<ICmpInst>(&I))
      if (C->getOperand(0)->getType()->isIntegerTy(64) &&
          !(isa<Constant>(C->getOperand(0)) && isa<Constant>(C->getOperand(1))))
        Cmps.push_back(C);
  if (Cmps.empty())
    return false;
  SmallVector<WeakVH, 8> Ranged;
  AGX::addSpecialRegisterRanges(F, &Ranged, /*ForAnalysis=*/true);
  scope_exit Unrange([&] {
    for (WeakVH &V : Ranged)
      if (auto *CI = dyn_cast_or_null<CallBase>(V))
        CI->removeRetAttr(Attribute::Range);
  });
  DominatorTree DT(F);
  DomConditionCache DC;
  for (BasicBlock &BB : F)
    if (auto *Br = dyn_cast<CondBrInst>(BB.getTerminator()))
      DC.registerBranch(Br);
  const DataLayout &DL = F.getDataLayout();
  ConstantRange S32(APInt::getSignedMinValue(32).sext(64),
                    APInt::getSignedMinValue(32).sext(64).abs());
  std::function<bool(Value *, Instruction *, unsigned)> Fits =
      [&](Value *V, Instruction *At, unsigned Depth) -> bool {
    if (auto *C = dyn_cast<ConstantInt>(V))
      return C->getValue().isSignedIntN(32);
    if (auto *E = dyn_cast<SExtInst>(V))
      return E->getSrcTy()->getScalarSizeInBits() <= 32;
    if (auto *B = dyn_cast<BinaryOperator>(V);
        B && Depth < 8 &&
        (B->getOpcode() == Instruction::And || B->getOpcode() == Instruction::Or ||
         B->getOpcode() == Instruction::Xor) &&
        Fits(B->getOperand(0), At, Depth + 1) &&
        Fits(B->getOperand(1), At, Depth + 1))
      return true;
    SimplifyQuery Q(DL, /*TLI=*/nullptr, &DT, /*AC=*/nullptr, At,
                    /*UseInstrInfo=*/true, /*CanUseUndef=*/true, &DC);
    // (ranges through additions, subtractions, multiplications and casts,
    // which known bits alone lose: x - 1 for x in [0, 1024))
    std::function<ConstantRange(Value *, unsigned)> Range =
        [&](Value *X, unsigned D) -> ConstantRange {
      unsigned W = X->getType()->getScalarSizeInBits();
      ConstantRange KB =
          computeConstantRangeIncludingKnownBits(X, /*ForSigned=*/true, Q);
      if (D > 8)
        return KB;
      ConstantRange R = ConstantRange::getFull(W);
      if (auto *Z = dyn_cast<ZExtInst>(X))
        R = Range(Z->getOperand(0), D + 1).zeroExtend(W);
      else if (auto *E = dyn_cast<SExtInst>(X))
        R = Range(E->getOperand(0), D + 1).signExtend(W);
      else if (auto *O = dyn_cast<BinaryOperator>(X)) {
        ConstantRange A = Range(O->getOperand(0), D + 1);
        ConstantRange C = Range(O->getOperand(1), D + 1);
        switch (O->getOpcode()) {
        case Instruction::Add:
          R = A.add(C);
          break;
        case Instruction::Sub:
          R = A.sub(C);
          break;
        case Instruction::Mul:
          R = A.multiply(C);
          break;
        case Instruction::Shl:
          R = A.shl(C);
          break;
        default:
          break;
        }
      }
      return R.intersectWith(KB, ConstantRange::Signed);
    };
    return S32.contains(Range(V, Depth));
  };
  bool Changed = false;
  for (ICmpInst *C : Cmps) {
    if (!Fits(C->getOperand(0), C, 0) || !Fits(C->getOperand(1), C, 0))
      continue;
    IRBuilder<> B(C);
    Type *I32 = B.getInt32Ty();
    Value *N = B.CreateICmp(C->getPredicate(),
                            B.CreateTrunc(C->getOperand(0), I32),
                            B.CreateTrunc(C->getOperand(1), I32), C->getName());
    C->replaceAllUsesWith(N);
    C->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXHalfFromFloat(
    "agx-half-from-float", cl::init(true), cl::Hidden,
    cl::desc("fptrunc to half of one float add, sub, mul or div of halves "
             "(and min / max, fneg, fabs and selects of such) in half"));

/// fptrunc(op(fpext a, fpext b)) to half is op on the halves: the float
/// result rounded to half is the half result for + - * / (float has 24 >=
/// 2 * 11 + 2 bits: a second rounding never differs, Figueroa), and no such
/// result is a float denormal (the least, 2^-24 * 2^-24, is far above). min
/// / max (every kind), fneg, fabs and selects of values so narrowed commute
/// with the rounding, which is monotonic and keeps signs and NaNs: MLIR's
/// mixed precision truncf(maximumf(addf(extf a, extf b), 0)) is a half add
/// and a half max (L15-E440). An operation of two operations' results is
/// not: its float operands are not halves.
static bool halfFromFloat(Function &F) {
  if (!AGXHalfFromFloat)
    return false;
  using namespace PatternMatch;
  Type *H = Type::getHalfTy(F.getContext());
  // a half equal to V (an exact one: an extension or a constant)
  auto Exact = [&](Value *V, IRBuilder<> &B) -> Value * {
    Value *X;
    if (match(V, m_FPExt(m_Value(X))) && X->getType()->isHalfTy())
      return X;
    if (auto *C = dyn_cast<ConstantFP>(V)) {
      APFloat A = C->getValueAPF();
      bool Lost;
      if (A.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Lost) ==
              APFloat::opOK &&
          !Lost)
        return ConstantFP::get(H, A);
    }
    return nullptr;
  };
  std::function<bool(Value *, unsigned)> Can = [&](Value *V,
                                                   unsigned D) -> bool {
    if (!V->getType()->isFloatTy() || D > 6)
      return false;
    IRBuilder<> B(F.getContext());
    if (Exact(V, B))
      return true;
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !I->hasOneUse())
      return false;
    switch (I->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FDiv:
      return Exact(I->getOperand(0), B) && Exact(I->getOperand(1), B);
    case Instruction::FNeg:
      return Can(I->getOperand(0), D + 1);
    case Instruction::Select:
      return Can(I->getOperand(1), D + 1) && Can(I->getOperand(2), D + 1);
    default:
      break;
    }
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::fabs:
        return Can(II->getArgOperand(0), D + 1);
      case Intrinsic::minnum:
      case Intrinsic::maxnum:
      case Intrinsic::minimum:
      case Intrinsic::maximum:
      case Intrinsic::minimumnum:
      case Intrinsic::maximumnum:
        return Can(II->getArgOperand(0), D + 1) &&
               Can(II->getArgOperand(1), D + 1);
      default:
        break;
      }
    }
    return false;
  };
  std::function<Value *(Value *, IRBuilder<> &)> Build =
      [&](Value *V, IRBuilder<> &B) -> Value * {
    if (Value *X = Exact(V, B))
      return X;
    auto *I = cast<Instruction>(V);
    Value *R;
    if (auto *BO = dyn_cast<BinaryOperator>(I)) {
      R = B.CreateBinOp(BO->getOpcode(), Exact(BO->getOperand(0), B),
                        Exact(BO->getOperand(1), B));
    } else if (I->getOpcode() == Instruction::FNeg) {
      R = B.CreateFNeg(Build(I->getOperand(0), B));
    } else if (auto *S = dyn_cast<SelectInst>(I)) {
      R = B.CreateSelect(S->getCondition(), Build(S->getTrueValue(), B),
                         Build(S->getFalseValue(), B));
    } else {
      auto *II = cast<IntrinsicInst>(I);
      if (II->getIntrinsicID() == Intrinsic::fabs)
        R = B.CreateUnaryIntrinsic(Intrinsic::fabs,
                                   Build(II->getArgOperand(0), B));
      else
        R = B.CreateBinaryIntrinsic(II->getIntrinsicID(),
                                    Build(II->getArgOperand(0), B),
                                    Build(II->getArgOperand(1), B));
    }
    if (auto *RI = dyn_cast<Instruction>(R))
      if (isa<FPMathOperator>(RI) && isa<FPMathOperator>(I))
        RI->copyFastMathFlags(I);
    return R;
  };
  SmallVector<FPTruncInst *, 8> Roots;
  for (Instruction &I : instructions(F))
    if (auto *T = dyn_cast<FPTruncInst>(&I);
        T && T->getType()->isHalfTy() && T->getSrcTy()->isFloatTy() &&
        !isa<ConstantFP>(T->getOperand(0)) && Can(T->getOperand(0), 0) &&
        !isa<FPExtInst>(T->getOperand(0)))
      Roots.push_back(T);
  for (FPTruncInst *T : Roots) {
    IRBuilder<> B(T);
    Value *N = Build(T->getOperand(0), B);
    Value *Old = T->getOperand(0);
    T->replaceAllUsesWith(N);
    T->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(Old);
  }
  return !Roots.empty();
}

static cl::opt<bool> AGXHalfInexactCompares(
    "agx-half-inexact-compares", cl::init(true), cl::Hidden,
    cl::desc("A compare of an extended half (or its negation) with a float "
             "constant inexact in half done on the half, with the half next "
             "to the constant"));

static cl::opt<bool> AGXHalfSelects(
    "agx-half-selects", cl::init(true), cl::Hidden,
    cl::desc("A select or compare of halves extended to float (or of float "
             "constants exact in half) done on the halves"));

/// select(c, ext a, ext b) is ext(select(c, a, b)) and fcmp(ext a, ext b) is
/// fcmp(a, b), a and b halves (or, on one side, a float constant exact in
/// half): the extension is exact, keeps NaNs and the order. The select's
/// extension then folds into the float operations reading it as a 16-bit
/// source, instead of being a conversion before the select (the clamps of
/// the exp expansion's argument, silu's e^-x of an extended half; L15-E449).
/// Selects only where every user reads a 16-bit source: float arithmetic,
/// fmas, half truncations, compares and selects turned to half here, and
/// negations of those.
static bool halfSelects(Function &F) {
  if (!AGXHalfSelects)
    return false;
  using namespace PatternMatch;
  const DataLayout &DL = F.getDataLayout();
  Type *H = Type::getHalfTy(F.getContext());
  auto Narrow = [&](Value *V) -> Value * {
    Value *X;
    if (match(V, m_FPExt(m_Value(X))) &&
        X->getType()->getScalarType()->isHalfTy())
      return X;
    if (auto *C = dyn_cast<Constant>(V);
        C && V->getType()->getScalarType()->isFloatTy()) {
      Type *HT = V->getType()->getWithNewType(H);
      Constant *T = ConstantFoldCastOperand(Instruction::FPTrunc, C, HT, DL);
      if (T && ConstantFoldCastOperand(Instruction::FPExt, T, V->getType(),
                                       DL) == C)
        return T;
    }
    return nullptr;
  };
  auto IsExt = [](Value *V) { return isa<FPExtInst>(V); };
  // both values narrow (one an extension) or selects becoming narrow
  SmallPtrSet<SelectInst *, 8> Cand;
  auto NarrowOrCand = [&](Value *V) {
    auto *S = dyn_cast<SelectInst>(V);
    return Narrow(V) || (S && Cand.contains(S));
  };
  // fcmp p (-x), c is fcmp p' x, -c (p' the swapped predicate); an
  // inequality with a float constant inexact in half is one with the half
  // below it (> and >=: no half is c) or above it (< and <=) (L15-E473:
  // exp's +inf bound 88.7228394 of an extended half's clamp).
  struct HalfCmp {
    CmpInst::Predicate P;
    Value *X;
    Constant *K;
  };
  auto InexactCmp = [&](FCmpInst *C) -> std::optional<HalfCmp> {
    if (!AGXHalfInexactCompares)
      return std::nullopt;
    CmpInst::Predicate P = C->getPredicate();
    Value *X = C->getOperand(0);
    auto *K = dyn_cast<ConstantFP>(C->getOperand(1));
    if (!K) {
      K = dyn_cast<ConstantFP>(C->getOperand(0));
      X = C->getOperand(1);
      P = CmpInst::getSwappedPredicate(P);
    }
    if (!K || !K->getType()->isFloatTy() || !K->getValueAPF().isFinite())
      return std::nullopt;
    APFloat V = K->getValueAPF();
    Value *Y;
    if (match(X, m_FNeg(m_Value(Y)))) {
      X = Y;
      V.changeSign();
      P = CmpInst::getSwappedPredicate(P);
    }
    bool Up;
    switch (P) {
    case CmpInst::FCMP_OGT: case CmpInst::FCMP_OGE:
    case CmpInst::FCMP_UGT: case CmpInst::FCMP_UGE:
      Up = false;
      break;
    case CmpInst::FCMP_OLT: case CmpInst::FCMP_OLE:
    case CmpInst::FCMP_ULT: case CmpInst::FCMP_ULE:
      Up = true;
      break;
    default:
      return std::nullopt;
    }
    APFloat HV = V;
    bool LosesInfo;
    HV.convert(APFloat::IEEEhalf(),
               Up ? APFloat::rmTowardPositive : APFloat::rmTowardNegative,
               &LosesInfo);
    if (!LosesInfo)
      return std::nullopt; // exact: Narrow's
    // x >= c, x > c: x > down(c); x <= c, x < c: x < up(c)
    switch (P) {
    case CmpInst::FCMP_OGE: P = CmpInst::FCMP_OGT; break;
    case CmpInst::FCMP_UGE: P = CmpInst::FCMP_UGT; break;
    case CmpInst::FCMP_OLE: P = CmpInst::FCMP_OLT; break;
    case CmpInst::FCMP_ULE: P = CmpInst::FCMP_ULT; break;
    default: break;
    }
    return HalfCmp{P, X, ConstantFP::get(H, HV)};
  };
  for (Instruction &I : instructions(F))
    if (auto *S = dyn_cast<SelectInst>(&I);
        S && S->getType()->getScalarType()->isFloatTy() &&
        NarrowOrCand(S->getTrueValue()) && NarrowOrCand(S->getFalseValue()) &&
        (IsExt(S->getTrueValue()) || IsExt(S->getFalseValue()) ||
         isa<SelectInst>(S->getTrueValue()) ||
         isa<SelectInst>(S->getFalseValue())))
      Cand.insert(S);
  // (users reading the value as a 16-bit source)
  std::function<bool(Use &, unsigned)> Reads16 = [&](Use &U,
                                                     unsigned D) -> bool {
    auto *I = cast<Instruction>(U.getUser());
    switch (I->getOpcode()) {
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
      return true;
    case Instruction::FPTrunc:
      return I->getType()->getScalarType()->isHalfTy();
    case Instruction::FCmp:
      if (NarrowOrCand(I->getOperand(1 - U.getOperandNo())))
        return true;
      if (auto HC = InexactCmp(cast<FCmpInst>(I)))
        return HC->X == U.get() ||
               (match(U.get(), m_FNeg(m_Value())) &&
                cast<Instruction>(U.get())->getOperand(0) == HC->X);
      return false;
    case Instruction::FNeg:
      return D < 2 && all_of(I->uses(),
                             [&](Use &V) { return Reads16(V, D + 1); });
    case Instruction::Select: {
      auto *S = cast<SelectInst>(I);
      return U.getOperandNo() != 0 && Cand.contains(S);
    }
    default:
      break;
    }
    if (auto *II = dyn_cast<IntrinsicInst>(I))
      return II->getIntrinsicID() == Intrinsic::fma ||
             II->getIntrinsicID() == Intrinsic::fmuladd;
    return false;
  };
  // (users first: a candidate stays one only if its users all read 16 bits)
  SmallVector<SelectInst *, 8> Order;
  for (Instruction &I : instructions(F))
    if (auto *S = dyn_cast<SelectInst>(&I); S && Cand.contains(S))
      Order.push_back(S);
  for (bool Again = true; Again;) {
    Again = false;
    for (SelectInst *S : Order)
      if (Cand.contains(S) &&
          !all_of(S->uses(), [&](Use &U) { return Reads16(U, 0); })) {
        Cand.erase(S);
        Again = true;
      }
    for (SelectInst *S : Order)
      if (Cand.contains(S) && !(NarrowOrCand(S->getTrueValue()) &&
                                NarrowOrCand(S->getFalseValue()))) {
        Cand.erase(S);
        Again = true;
      }
  }
  bool Changed = false;
  DenseMap<SelectInst *, Value *> NewOf;
  auto Get = [&](Value *V) -> Value * {
    if (auto *S = dyn_cast<SelectInst>(V); S && NewOf.count(S))
      return NewOf[S];
    return Narrow(V);
  };
  // compares of narrow values (or of selects becoming narrow), then the
  // selects in order (operands before users)
  SmallVector<FCmpInst *, 8> Cmps;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<FCmpInst>(&I);
        C && C->getOperand(0)->getType()->getScalarType()->isFloatTy() &&
        NarrowOrCand(C->getOperand(0)) && NarrowOrCand(C->getOperand(1)) &&
        !(isa<Constant>(C->getOperand(0)) && isa<Constant>(C->getOperand(1))))
      Cmps.push_back(C);
  SmallVector<std::pair<FCmpInst *, HalfCmp>, 4> InexactCmps;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<FCmpInst>(&I);
        C && C->getOperand(0)->getType()->isFloatTy() &&
        !is_contained(Cmps, C))
      if (auto HC = InexactCmp(C); HC && NarrowOrCand(HC->X))
        InexactCmps.push_back({C, *HC});
  // (a candidate operand laid out after its user: none of it)
  SmallPtrSet<SelectInst *, 8> Seen;
  for (SelectInst *S : Order) {
    Seen.insert(S);
    for (Value *V : {S->getTrueValue(), S->getFalseValue()})
      if (auto *O = dyn_cast<SelectInst>(V);
          O && Cand.contains(O) && !Seen.contains(O))
        return false;
  }
  for (SelectInst *S : Order) {
    if (!Cand.contains(S))
      continue;
    IRBuilder<> B(S);
    Value *N = B.CreateSelect(S->getCondition(), Get(S->getTrueValue()),
                              Get(S->getFalseValue()), S->getName() + ".h");
    if (auto *NI = dyn_cast<Instruction>(N))
      NI->copyMetadata(*S);
    NewOf[S] = N;
    Changed = true;
  }
  for (FCmpInst *C : Cmps) {
    IRBuilder<> B(C);
    Value *N = B.CreateFCmp(C->getPredicate(), Get(C->getOperand(0)),
                            Get(C->getOperand(1)), C->getName() + ".h");
    if (auto *NI = dyn_cast<Instruction>(N))
      NI->copyFastMathFlags(C);
    C->replaceAllUsesWith(N);
    Changed = true;
  }
  for (auto &[C, HC] : InexactCmps) {
    IRBuilder<> B(C);
    Value *N = B.CreateFCmp(HC.P, Get(HC.X), HC.K, C->getName() + ".h");
    if (auto *NI = dyn_cast<Instruction>(N))
      NI->copyFastMathFlags(C);
    C->replaceAllUsesWith(N);
    Changed = true;
  }
  for (SelectInst *S : Order)
    if (Value *N = NewOf.lookup(S)) {
      IRBuilder<> B(S);
      S->replaceAllUsesWith(B.CreateFPExt(N, S->getType()));
    }
  for (FCmpInst *C : Cmps)
    C->eraseFromParent();
  for (auto &[C, HC] : InexactCmps)
    C->eraseFromParent();
  for (SelectInst *S : reverse(Order))
    if (NewOf.count(S))
      RecursivelyDeleteTriviallyDeadInstructions(S);
  // -ext(h) as ext(-h), a 16-bit source with its negation modifier (the
  // instruction selection's extended operands are an fpext at the top)
  SmallVector<UnaryOperator *, 8> Negs;
  for (Instruction &I : instructions(F))
    if (auto *N = dyn_cast<UnaryOperator>(&I);
        N && N->getOpcode() == Instruction::FNeg &&
        isa<FPExtInst>(N->getOperand(0)) &&
        cast<FPExtInst>(N->getOperand(0))
            ->getSrcTy()
            ->getScalarType()
            ->isHalfTy())
      Negs.push_back(N);
  for (UnaryOperator *N : Negs) {
    IRBuilder<> B(N);
    auto *E = cast<FPExtInst>(N->getOperand(0));
    Value *R = B.CreateFPExt(B.CreateFNeg(E->getOperand(0)), N->getType());
    N->replaceAllUsesWith(R);
    RecursivelyDeleteTriviallyDeadInstructions(N);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXDeferredMinMax(
    "agx-deferred-minmax-reductions", cl::init(true), cl::Hidden,
    cl::desc("A loop's running llvm.maximum / minimum as an ordered compare-"
             "select, a NaN carried beside it and a flag for zeros of the "
             "result's sign, combined after the loop"));

/// A loop's running m = maximum(m, v) (minimum likewise, of float or float
/// vectors) used only by itself in the loop: maximum's NaN result and its
/// -0 below +0 are deferred to the loop's exits (L15-E444). In the loop
///   m' = v > m ? v : m      (a NaN v leaves m),
///   n' = v uno v ? v : n    (the last NaN seen, the initial value at first),
///   z' = smax(z, bits(v))   (smin for minimum),
/// three instructions for maximum's five (compare-select, the and of the
/// bits, its flush, the equal select and the NaN select); after the loop
///   r = n uno n ? n : (m == 0 && z >= 0 (z < 0) ? +0 (-0) : m),
/// which is maximum's value: a NaN if one was seen, else the largest, of
/// the zeros +0 if one was seen (a zero m means every value is at most a
/// zero or a denormal, which the compares flush: a clear sign among their
/// bits, z >= 0, is a +0 or positive denormal, and maximum gives +0 or
/// that denormal either way). MLIR's maximumf row maxima (torch-mlir's softmax).
static cl::opt<bool> AGXDeferredMinMaxNum(
    "agx-deferred-minmaxnum-reductions", cl::init(true), cl::Hidden,
    cl::desc("agx-deferred-minmax-reductions also for maximumnum / "
             "minimumnum from a number, its zero-sign-free uses in the loop "
             "taking the deferred value"));

static cl::opt<bool> AGXDeferredNaNSideUses(
    "agx-deferred-minmax-side-uses", cl::init(true), cl::Hidden,
    cl::desc("agx-deferred-minmax-nan-in-m's running maximum / minimum "
             "also with zero-sign-free uses in the loop"));

static cl::opt<bool> AGXDeferredHalf(
    "agx-deferred-minmax-half", cl::init(true), cl::Hidden,
    cl::desc("agx-deferred-minmax-reductions of extended halves in half"));

static cl::opt<bool> AGXDeferredLaneResults(
    "agx-deferred-minmax-lane-results", cl::init(true), cl::Hidden,
    cl::desc("agx-deferred-minmax-reductions' results of a vector's elements "
             "per element, and maximum / minimum of two results as one"));

static cl::opt<bool> AGXDeferredNaNInM(
    "agx-deferred-minmax-nan-in-m", cl::init(true), cl::Hidden,
    cl::desc("agx-deferred-minmax-reductions' running value as maximum / "
             "minimum without the order of zeros (one fcmpsel that keeps a "
             "NaN), no NaN carried beside it"));

static bool deferredMinMaxReductions(Function &F) {
  if (!AGXDeferredMinMax)
    return false;
  using namespace PatternMatch;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    BasicBlock *H = L->getHeader(), *Pre = L->getLoopPreheader(),
               *Latch = L->getLoopLatch();
    if (!Pre || !Latch)
      continue;
    SmallVector<PHINode *, 4> Phis;
    for (PHINode &P : H->phis())
      if (P.getType()->getScalarType()->isFloatTy() &&
          P.getNumIncomingValues() == 2)
        Phis.push_back(&P);
    for (PHINode *P : Phis) {
      auto *U = dyn_cast<IntrinsicInst>(P->getIncomingValueForBlock(Latch));
      if (!U || !L->contains(U) || !DT.dominates(U->getParent(), Latch))
        continue;
      Intrinsic::ID ID = U->getIntrinsicID();
      // maximumnum / minimumnum from a number (never a NaN, so no NaN is
      // carried), whose running values may also have uses in the loop that
      // no zero's sign reaches (ZeroSignFree: an online softmax's rescale
      // e^(m - m')), which take the deferred value (L15-E446)
      bool Num = ID == Intrinsic::maximumnum || ID == Intrinsic::minimumnum;
      if (ID != Intrinsic::maximum && ID != Intrinsic::minimum &&
          !(Num && AGXDeferredMinMaxNum))
        continue;
      if (Num) {
        auto *C0 = dyn_cast<ConstantFP>(P->getIncomingValueForBlock(Pre));
        if (!C0 || C0->isNaN())
          continue;
      }
      bool Max = ID == Intrinsic::maximum || ID == Intrinsic::maximumnum;
      bool NaNInM = !Num && AGXDeferredNaNInM;
      // (the uses in the loop besides the chain, for Num: zero-sign free)
      std::optional<ThreadgroupAccesses> TGA;
      ZeroSignFree ZSF(TGA);
      SmallVector<std::pair<Use *, unsigned>, 4> Side;
      // the chain P -> m1 = op(P, v1) -> ... -> U = op(m_k-1, v_k), each
      // link's only user the next (an unrolled loop's running maximum)
      SmallVector<IntrinsicInst *, 4> Chain;
      SmallVector<Value *, 4> Vs;
      Value *X = P;
      bool Bad = false;
      while (true) {
        if (X == U)
          break;
        // the next link: X's one user of the chain's kind; for Num, X's
        // other users in the loop zero-sign free (link K's value, K = the
        // chain's length so far: 0 is the phi's)
        IntrinsicInst *C = nullptr;
        for (Use &Us : X->uses()) {
          auto *UI = cast<Instruction>(Us.getUser());
          auto *II = dyn_cast<IntrinsicInst>(UI);
          if (!C && II && II->getIntrinsicID() == ID && L->contains(II)) {
            C = II;
            continue;
          }
          // (also maximum's, whose deferred value carries its NaN: the
          // same value but for a zero's sign, or a denormal for the zero
          // a compare of it flushes to, L15-E463)
          if ((Num || (NaNInM && AGXDeferredNaNSideUses)) && L->contains(UI) &&
              UI != P && ZSF.use(Us)) {
            Side.push_back({&Us, unsigned(Chain.size())});
            continue;
          }
          Bad = true;
          break;
        }
        if (Bad)
          break;
        if (!C || C->getIntrinsicID() != U->getIntrinsicID() ||
            !L->contains(C) || Chain.size() > 16 ||
            C->hasNoSignedZeros()) {
          Bad = true;
          break;
        }
        Value *V = C->getArgOperand(0) == X ? C->getArgOperand(1)
                                            : C->getArgOperand(0);
        if (V == X || (C->getArgOperand(0) != X && C->getArgOperand(1) != X)) {
          Bad = true;
          break;
        }
        Chain.push_back(C);
        Vs.push_back(V);
        X = C;
      }
      if (Bad || Chain.empty() || Chain.back() != U)
        continue;
      // U's other users: the phi, and instructions outside the loop (phis,
      // the result then on the edges that bring U, or plain users)
      bool OK = true;
      for (Use &Us : U->uses()) {
        auto *I = cast<Instruction>(Us.getUser());
        if (I == P || !L->contains(I))
          continue;
        if ((Num || (NaNInM && AGXDeferredNaNSideUses)) && ZSF.use(Us))
          Side.push_back({&Us, unsigned(Chain.size())});
        else
          OK = false;
      }
      if (!OK)
        continue;
      Type *Ty = P->getType();
      Value *Init = P->getIncomingValueForBlock(Pre);
      // Values that are all extended halves, from a constant exact in half:
      // the running value in half, extended once after the loop (maximum of
      // extensions is the extension of the maximum; f16 compares flush no
      // denormal) (L15-E462).
      Type *FTy = Ty;
      bool Narrow = false;
      SmallVector<Instruction *, 4> OldExts;
      if (AGXDeferredHalf && Ty->isFloatTy() && Side.empty()) {
        auto *CI = dyn_cast<ConstantFP>(Init);
        bool Exact = false;
        if (CI) {
          APFloat A = CI->getValueAPF();
          bool Lost;
          Exact = A.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven,
                            &Lost) == APFloat::opOK &&
                  !Lost;
        }
        Narrow = Exact && all_of(Vs, [](Value *V) {
                   auto *E = dyn_cast<FPExtInst>(V);
                   return E && E->getSrcTy()->isHalfTy();
                 });
        if (Narrow) {
          Ty = Type::getHalfTy(F.getContext());
          APFloat A = CI->getValueAPF();
          bool Lost;
          A.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &Lost);
          Init = ConstantFP::get(Ty, A);
          for (Value *&V : Vs) {
            OldExts.push_back(cast<Instruction>(V));
            V = cast<FPExtInst>(V)->getOperand(0);
          }
        }
      }
      Type *ITy = Ty->getWithNewType(
          IntegerType::get(F.getContext(), Ty->getScalarSizeInBits()));
      IRBuilder<> PB(Pre->getTerminator());
      // z: the signed integer maximum (minimum) of the values' bits: where
      // the result is a zero, every value seen is at most a zero or
      // denormal, and z >= 0 (z < 0) says one of them had a clear (set) sign
      auto Bits = [&](IRBuilder<> &B, Value *X) {
        return B.CreateBitCast(X, ITy);
      };
      PHINode *M = PHINode::Create(Ty, 2, P->getName() + ".m", P->getIterator());
      PHINode *N = PHINode::Create(Ty, 2, P->getName() + ".nan", P->getIterator());
      PHINode *Z = PHINode::Create(ITy, 2, P->getName() + ".z", P->getIterator());
      Value *M2 = M, *N2 = N, *Z2 = Z;
      // the deferred value before each link and after the last
      SmallVector<Value *, 4> MK{M};
      for (unsigned K = 0; K < Chain.size(); ++K) {
        IRBuilder<> B(Chain[K]);
        Value *V = Vs[K];
        if (Num) {
          // m' = v >= m ? v : m (a NaN v leaves m; a +0 meeting -0 is
          // taken), z' = smax(z, bits(m')), never a NaN's bits
          Value *Cmp = Max ? B.CreateFCmpOGE(V, M2) : B.CreateFCmpOLE(V, M2);
          M2 = B.CreateSelect(Cmp, V, M2, P->getName() + ".m.next");
          Value *MB = Bits(B, M2);
          Z2 = B.CreateSelect(Max ? B.CreateICmpSGT(MB, Z2)
                                  : B.CreateICmpSLT(MB, Z2),
                              MB, Z2, P->getName() + ".z.next");
          MK.push_back(M2);
          continue;
        }
        if (NaNInM) {
          // m' = maximum(m, v) without the order of zeros: one fcmpsel
          // whose "n" condition makes a number lose to a NaN (the last NaN
          // seen, as maximum's chain gives), a v equal to m taken
          // (L15-E453)
          auto *MM = cast<Instruction>(B.CreateBinaryIntrinsic(
              ID, M2, V, nullptr, P->getName() + ".m.next"));
          MM->setHasNoSignedZeros(true);
          M2 = MM;
        } else {
          Value *Cmp = Max ? B.CreateFCmpOGT(V, M2) : B.CreateFCmpOLT(V, M2);
          M2 = B.CreateSelect(Cmp, V, M2, P->getName() + ".m.next");
          N2 = B.CreateSelect(B.CreateFCmpUNO(V, V), V, N2,
                              P->getName() + ".nan.next");
        }
        Value *VB = Bits(B, V);
        Z2 = B.CreateSelect(Max ? B.CreateICmpSGT(VB, Z2) : B.CreateICmpSLT(VB, Z2),
                            VB, Z2, P->getName() + ".z.next");
        MK.push_back(M2);
      }
      M->addIncoming(Init, Pre);
      M->addIncoming(M2, Latch);
      N->addIncoming(Init, Pre);
      N->addIncoming(N2, Latch);
      Z->addIncoming(Bits(PB, Init), Pre);
      Z->addIncoming(Z2, Latch);
      // the result at a point after U
      auto ResultW = [&](IRBuilder<> &RB, Value *MV, Value *NV,
                         Value *ZV) -> Value * {
        // (of the vector or, for an extraction, of its element)
        Type *RTy = MV->getType(), *RITy = ZV->getType();
        Value *Zero = ConstantFP::get(RTy, Max ? 0.0 : -0.0);
        Value *IsZero = RB.CreateFCmpOEQ(MV, ConstantFP::get(RTy, 0.0));
        Value *Seen = Max ? RB.CreateICmpSGE(ZV, Constant::getNullValue(RITy))
                          : RB.CreateICmpSLT(ZV, Constant::getNullValue(RITy));
        Value *R = RB.CreateSelect(RB.CreateAnd(IsZero, Seen), Zero, MV);
        if (Num || NaNInM) {
          // (a result combineDeferredMinMax may merge: its z belongs to m)
          if (NaNInM)
            if (auto *RI = dyn_cast<Instruction>(R))
              RI->setMetadata("agx.deferred.minmax",
                              MDNode::get(RI->getContext(), {}));
          return R;
        }
        return RB.CreateSelect(RB.CreateFCmpUNO(NV, NV), NV, R,
                               U->getName() + ".r");
      };
      auto Result = [&](IRBuilder<> &RB, Value *MV, Value *NV,
                        Value *ZV) -> Value * {
        Value *R = ResultW(RB, MV, NV, ZV);
        return Narrow ? RB.CreateFPExt(R, FTy) : R;
      };
      // the zero-sign-free uses in the loop: the deferred value (the same
      // but for a zero's sign)
      for (auto [Us, K] : Side)
        Us->set(MK[K]);
      SmallVector<Instruction *, 4> Users;
      for (User *Us : U->users())
        if (Us != P && !llvm::is_contained(Users, cast<Instruction>(Us)))
          Users.push_back(cast<Instruction>(Us));
      for (Instruction *I : Users) {
        if (auto *LP = dyn_cast<PHINode>(I)) {
          // a phi after the loop (LCSSA's, or one that also takes the value
          // of a path around the loop): the result on each edge bringing U,
          // in a block of its own (not in the loop's exiting block)
          SmallVector<BasicBlock *, 2> Ins;
          for (unsigned K = 0; K < LP->getNumIncomingValues(); ++K)
            if (LP->getIncomingValue(K) == U &&
                !llvm::is_contained(Ins, LP->getIncomingBlock(K)))
              Ins.push_back(LP->getIncomingBlock(K));
          for (BasicBlock *In : Ins) {
            BasicBlock *E = SplitEdge(In, LP->getParent(), &DT, &LI);
            IRBuilder<> RB(E->getTerminator());
            Value *R = Result(RB, M2, N2, Z2);
            for (unsigned K = 0; K < LP->getNumIncomingValues(); ++K)
              if (LP->getIncomingBlock(K) == E)
                LP->setIncomingValue(K, R);
          }
          continue;
        }
        IRBuilder<> RB(I);
        // an element of a vector's result: the result of the elements
        // (one lane's selects, which combineDeferredMinMax can merge)
        if (auto *EE = dyn_cast<ExtractElementInst>(I);
            EE && AGXDeferredLaneResults && EE->getVectorOperand() == U) {
          Value *Idx = EE->getIndexOperand();
          Value *R = Result(RB, RB.CreateExtractElement(M2, Idx),
                            NaNInM || Num ? nullptr
                                          : RB.CreateExtractElement(N2, Idx),
                            RB.CreateExtractElement(Z2, Idx));
          EE->replaceAllUsesWith(R);
          EE->eraseFromParent();
          continue;
        }
        Value *R = Result(RB, M2, N2, Z2);
        I->replaceUsesOfWith(U, R);
      }
      for (IntrinsicInst *C : reverse(Chain)) {
        C->replaceAllUsesWith(PoisonValue::get(FTy));
        C->eraseFromParent();
      }
      P->replaceAllUsesWith(PoisonValue::get(FTy));
      P->eraseFromParent();
      for (Instruction *E : OldExts)
        RecursivelyDeleteTriviallyDeadInstructions(E);
      if (Num || NaNInM) {
        N->replaceAllUsesWith(PoisonValue::get(Ty));
        N->eraseFromParent();
      }
      Changed = true;
    }
  }
  return Changed;
}

/// maximum(r1, r2) of two deferred maxima's results r = m == 0 && z >= 0 ?
/// +0 : m (agx-deferred-minmax-nan-in-m's, minimum mirrored) is the result of
/// maximum nsz(m1, m2) (a NaN if either is one, m2's if both: maximum's NaN
/// of r1 and r2) and smax(z1, z2): the larger m's result, and for two zero
/// m's +0 where either has a clear sign among its values' bits (maximum's
/// +0 of +0 and a zero). Else a zero or denormal m2 where maximum gives -0:
/// both denormal results (L15-E454). The lanes of a vector's running maximum
/// reduced after its loop (rowmax_vec4): one result instead of one per lane
/// and three maximums.
static bool combineDeferredMinMax(Function &F) {
  if (!AGXDeferredLaneResults)
    return false;
  using namespace PatternMatch;
  struct Parts {
    Value *M, *Z;
  };
  auto Match = [](Value *V, bool Max) -> std::optional<Parts> {
    Value *M, *Z, *M2;
    CmpPredicate P1, P2;
    const APFloat *Zero, *C;
    auto *S = dyn_cast<SelectInst>(V);
    if (!S || !S->hasOneUse() || !S->getType()->isFloatTy() ||
        !S->getMetadata("agx.deferred.minmax") ||
        !match(S, m_Select(m_And(m_FCmp(P1, m_Value(M), m_APFloat(C)),
                                 m_ICmp(P2, m_Value(Z), m_Zero())),
                           m_APFloat(Zero), m_Value(M2))) ||
        M != M2 || P1 != FCmpInst::FCMP_OEQ || !C->isZero() ||
        !Zero->isZero() || Zero->isNegative() == Max ||
        P2 != (Max ? ICmpInst::ICMP_SGE : ICmpInst::ICMP_SLT) ||
        !Z->getType()->isIntegerTy(32))
      return std::nullopt;
    return Parts{M, Z};
  };
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !II->getType()->isFloatTy())
      continue;
    Intrinsic::ID ID = II->getIntrinsicID();
    if (ID != Intrinsic::maximum && ID != Intrinsic::minimum)
      continue;
    bool Max = ID == Intrinsic::maximum;
    auto A = Match(II->getArgOperand(0), Max),
         B = Match(II->getArgOperand(1), Max);
    if (!A || !B)
      continue;
    IRBuilder<> RB(II);
    auto *MM = cast<Instruction>(RB.CreateBinaryIntrinsic(ID, A->M, B->M));
    MM->setHasNoSignedZeros(true);
    Value *Z = Max ? RB.CreateBinaryIntrinsic(Intrinsic::smax, A->Z, B->Z)
                   : RB.CreateBinaryIntrinsic(Intrinsic::smin, A->Z, B->Z);
    Value *IsZero = RB.CreateFCmpOEQ(MM, ConstantFP::get(MM->getType(), 0.0));
    Value *Seen = Max ? RB.CreateICmpSGE(Z, RB.getInt32(0))
                      : RB.CreateICmpSLT(Z, RB.getInt32(0));
    Value *R = RB.CreateSelect(RB.CreateAnd(IsZero, Seen),
                               ConstantFP::get(MM->getType(), Max ? 0.0 : -0.0),
                               MM, II->getName());
    cast<Instruction>(R)->setMetadata("agx.deferred.minmax",
                                      MDNode::get(F.getContext(), {}));
    Value *OA = II->getArgOperand(0), *OB = II->getArgOperand(1);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(OA);
    RecursivelyDeleteTriviallyDeadInstructions(OB);
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXOtherBaseExps(
    "agx-other-base-exps", cl::init(true), cl::Hidden,
    cl::desc("exp2 and exp10 in the online softmax and below-maximum "
             "rewrites of exp"));

/// exp, or (L15-E484) exp2 / exp10: b^x is 1 at x = +-0 and a NaN at a NaN
/// as e^x is, and only its overflow clamp needs x above 0.
static bool isExpOfAnyBase(Intrinsic::ID ID) {
  return ID == Intrinsic::exp ||
         (AGXOtherBaseExps &&
          (ID == Intrinsic::exp2 || ID == Intrinsic::exp10));
}

static cl::opt<bool> AGXOnlineSoftmaxExp(
    "agx-online-softmax-exp", cl::init(true), cl::Hidden,
    cl::desc("exp(a - max(a, b)) and exp(b - max(a, b)) as one exponential "
             "and 1 + (x - x) for the other"));

static cl::opt<bool> AGXOnlineSoftmaxSharedNum(
    "agx-online-softmax-shared-one-num", cl::init(true), cl::Hidden,
    cl::desc("agx-online-softmax-shared-one also for maxnum / maximumnum of a "
             "running value from a number"));

static cl::opt<bool> AGXOnlineSoftmaxAbsArg(
    "agx-online-softmax-abs-arg", cl::init(true), cl::Hidden,
    cl::desc("agx-online-softmax-shared-one's exponent as -|b - a|, its clamp "
             "a compare of |b - a|"));

static cl::opt<unsigned> AGXOnlineSoftmaxAbsArgMaxChain(
    "agx-online-softmax-abs-arg-max-chain", cl::init(8), cl::Hidden,
    cl::desc("agx-online-softmax-abs-arg where a block's running maximum "
             "chain has at most this many pairs"));

static cl::opt<bool> AGXOnlineSoftmaxDiffCompare(
    "agx-online-softmax-diff-compare", cl::init(true), cl::Hidden,
    cl::desc("agx-online-softmax-abs-arg's selects on b - a > 0, not b > a"));

static cl::opt<bool> AGXOnlineSoftmaxShared(
    "agx-online-softmax-shared-one", cl::init(true), cl::Hidden,
    cl::desc("agx-online-softmax-exp's trivial side as u 0 + 1, one for "
             "both, for maximum and exp"));

/// e^(a - u) and e^(b - u), u = maximum(a, b) (or maximumnum or maxnum, which
/// return an operand too and drop one NaN; nsz or not; L15-E466 for maxnum):
/// an online softmax's rescale e^(m - m') and new term e^(v - m'). Where
/// b > a, u is b and b - u is b - b, +0 or (b infinite) a NaN, whose
/// exponential is 1 or a NaN, 1 + (b - u) exactly (the simulator's NaNs of
/// arithmetic are its default one either way); otherwise (b <= a, or a NaN)
/// a - u is that trivial one (u is a, a zero of a tie, or a NaN). So
///   x = b > a ? a - u : b - u, e = e^x,
///   e^(a - u) = b > a ? e : 1 + (a - u), e^(b - u) = b > a ? 1 + (b - u) : e:
/// one precise exponential (some twenty instructions) instead of two, x at
/// most 0 or a NaN (its clamps' upper bound needless: !agx.exp.below.max)
/// (L15-E464).
static bool onlineSoftmaxExps(Function &F) {
  if (!AGXOnlineSoftmaxExp)
    return false;
  using namespace PatternMatch;
  SmallVector<std::pair<IntrinsicInst *, IntrinsicInst *>, 4> Pairs;
  SmallPtrSet<IntrinsicInst *, 8> Taken;
  for (Instruction &I : instructions(F)) {
    auto *E1 = dyn_cast<IntrinsicInst>(&I);
    if (!E1 || !isExpOfAnyBase(E1->getIntrinsicID()) ||
        !E1->getType()->isFloatTy() || E1->hasApproxFunc() || Taken.count(E1))
      continue;
    Value *A, *U;
    if (!match(E1->getArgOperand(0), m_FSub(m_Value(A), m_Value(U))))
      continue;
    auto *UI = dyn_cast<IntrinsicInst>(U);
    if (!UI || (UI->getIntrinsicID() != Intrinsic::maximum &&
                UI->getIntrinsicID() != Intrinsic::maximumnum &&
                UI->getIntrinsicID() != Intrinsic::maxnum))
      continue;
    Value *B = UI->getArgOperand(0) == A ? UI->getArgOperand(1)
               : UI->getArgOperand(1) == A ? UI->getArgOperand(0)
                                            : nullptr;
    if (!B || B == A)
      continue;
    // the partner e^(b - u) in the same block
    IntrinsicInst *E2 = nullptr;
    for (User *Us : U->users()) {
      Value *X;
      if (!match(Us, m_FSub(m_Value(X), m_Specific(U))) || X != B)
        continue;
      for (User *Us2 : Us->users())
        if (auto *II = dyn_cast<IntrinsicInst>(Us2);
            II && II->getIntrinsicID() == E1->getIntrinsicID() &&
            II->getParent() == E1->getParent() && !II->hasApproxFunc() &&
            II != E1 && !Taken.count(II))
          E2 = II;
    }
    if (!E2)
      continue;
    Taken.insert(E1);
    Taken.insert(E2);
    Pairs.push_back({E1, E2});
  }
  // The pairs on each running maximum's chain in a block (the first link's
  // running value, followed back through the chain's maxima): -|b - a|'s
  // exponentials, which no longer wait for the link's maximum, ran ahead of
  // a 16-link chain and took 14 more registers (online_softmax, a lane's
  // elements one chain), not on 4- or 8-link ones (a vector's lanes, an
  // attention decode's keys).
  auto ChainRoot = [&](Value *V) {
    for (unsigned Depth = 0; Depth < 64; ++Depth) {
      auto *II = dyn_cast<IntrinsicInst>(V);
      if (!II || (II->getIntrinsicID() != Intrinsic::maximum &&
                  II->getIntrinsicID() != Intrinsic::maximumnum &&
                  II->getIntrinsicID() != Intrinsic::maxnum))
        return V;
      V = II->getArgOperand(0);
    }
    return V;
  };
  DenseMap<std::pair<BasicBlock *, Value *>, unsigned> ChainPairs;
  for (auto [E1, E2] : Pairs) {
    auto *FA = cast<Instruction>(E1->getArgOperand(0));
    ++ChainPairs[{E1->getParent(), ChainRoot(FA->getOperand(0))}];
  }
  for (auto [E1, E2] : Pairs) {
    IntrinsicInst *Late = E1->comesBefore(E2) ? E2 : E1;
    auto *FA = cast<Instruction>(E1->getArgOperand(0));
    auto *FB = cast<Instruction>(E2->getArgOperand(0));
    Value *A = FA->getOperand(0), *B = FB->getOperand(0);
    IRBuilder<> Bld(Late);
    Value *C = Bld.CreateFCmpOGT(B, A);
    Value *X = Bld.CreateSelect(C, FA, FB);
    auto *E = cast<Instruction>(
        Bld.CreateUnaryIntrinsic(E1->getIntrinsicID(), X, E1, "exp.one"));
    E->setMetadata("agx.exp.below.max", MDNode::get(F.getContext(), {}));
    Value *One = ConstantFP::get(E->getType(), 1.0);
    // With maximum (a NaN in, a NaN out) the trivial side's 1 + (x - u) is
    // u 0 + 1 on both sides: 1, or a NaN where u is infinite or a NaN, as
    // x - u is then (b > a: u = b; else u = a, a tie's other zero, or a
    // NaN): one fma for two additions. (Not for exp2: online_softmax_exp2
    // lost two resident groups. The argument as -|b - a|, also exact, lost
    // on every kernel: its exponentials overlap and take registers.)
    // L15-E513.
    auto *UI = cast<IntrinsicInst>(FA->getOperand(1));
    // maxnum / maximumnum too where a is never a NaN (then u is b where b >
    // a, else a): a running value from a number, whose maxnum keeps a
    // number (L15-E514).
    // (a number; a maxnum / maximumnum with a never-NaN operand; a phi of
    // such values, itself assumed one: an unrolled loop's chain)
    SmallPtrSet<PHINode *, 4> Assumed;
    std::function<bool(Value *, unsigned)> NeverNaN = [&](Value *V,
                                                          unsigned Depth) {
      if (auto *CF = dyn_cast<ConstantFP>(V))
        return !CF->isNaN();
      if (Depth > 32)
        return false;
      if (auto *II = dyn_cast<IntrinsicInst>(V);
          II && (II->getIntrinsicID() == Intrinsic::maxnum ||
                 II->getIntrinsicID() == Intrinsic::maximumnum))
        return NeverNaN(II->getArgOperand(0), Depth + 1) ||
               NeverNaN(II->getArgOperand(1), Depth + 1);
      if (auto *P = dyn_cast<PHINode>(V)) {
        if (!Assumed.insert(P).second)
          return true;
        bool R = all_of(P->incoming_values(), [&](Value *In) {
          return NeverNaN(In, Depth + 1);
        });
        Assumed.erase(P);
        return R;
      }
      return false;
    };
    bool Shared =
        UI->getIntrinsicID() == Intrinsic::maximum ||
        (AGXOnlineSoftmaxSharedNum &&
         (UI->getIntrinsicID() == Intrinsic::maxnum ||
          UI->getIntrinsicID() == Intrinsic::maximumnum) &&
         NeverNaN(A, 0));
    Value *Triv =
        AGXOnlineSoftmaxShared && Shared &&
                (E1->getIntrinsicID() == Intrinsic::exp ||
                 (E1->getIntrinsicID() == Intrinsic::exp2 &&
                  AGX::uniformConstantsEnabled()))
            ? Bld.CreateIntrinsic(Intrinsic::fma, {E->getType()},
                                  {UI, ConstantFP::get(E->getType(), 0.0),
                                   One})
            : nullptr;
    // The argument as -|b - a| (L15-E529): b > a gives a - b, else b - u
    // is b - a, or b - b where u = b (a tie, or a NaN a with maximum: a NaN
    // then either way), so the same value but for zero signs (e^-0 = e^+0)
    // and NaNs' (the simulator's arithmetic NaNs are its default one): where
    // u is a NaN only with a (maximum) or never (a never-NaN a, Shared), as
    // for the shared one. The expansion's clamp then compares |b - a| and
    // selects b - a itself, one instruction less than the select and
    // subtraction (E513 tried -|b - a| clamped as a value, which took one
    // more register for nothing).
    if (AGXOnlineSoftmaxAbsArg && Shared &&
        E1->getIntrinsicID() == Intrinsic::exp &&
        // (with the constants in uniforms, the exponentials running ahead
        // hold no coefficient registers: no limit, L15-E541)
        (AGX::uniformConstantsEnabled() ||
         ChainPairs.lookup({E1->getParent(), ChainRoot(A)}) <=
             AGXOnlineSoftmaxAbsArgMaxChain)) {
      IRBuilder<> EB(E);
      Value *D = EB.CreateFSub(B, A);
      Value *NX = EB.CreateFNeg(EB.CreateUnaryIntrinsic(Intrinsic::fabs, D));
      cast<CallInst>(E)->setArgOperand(0, NX);
      // The selects on b - a > 0 rather than b > a (L15-E681): one register
      // read each fewer (b and a, long computed, take the register file's
      // read ports; e and the trivial side are fresh). The same where they
      // differ, b > a with b - a flushed to 0 (or a compare's flush of its
      // denormal): e is then e^-0 = 1, and the trivial side is 1 (b finite),
      // so both select 1 and 1; b <= a never gives b - a > 0, and NaNs and
      // infinite ties fail both.
      if (AGXOnlineSoftmaxDiffCompare)
        C = Bld.CreateFCmpOGT(D, ConstantFP::get(D->getType(), 0.0));
    }
    Value *EA = Bld.CreateSelect(C, E, Triv ? Triv : Bld.CreateFAdd(One, FA));
    Value *EB = Bld.CreateSelect(C, Triv ? Triv : Bld.CreateFAdd(One, FB), E);
    E1->replaceAllUsesWith(EA);
    E2->replaceAllUsesWith(EB);
    E1->eraseFromParent();
    E2->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(X);
  }
  return !Pairs.empty();
}

static cl::opt<bool> AGXNarrowTruncatedOps(
    "agx-narrow-truncated-ops", cl::init(true), cl::Hidden,
    cl::desc("A 64-bit add, sub, mul, and, or, xor or shl whose every user "
             "truncates it to 32 bits as the 32-bit operation"));

/// Rebuilds a 64-bit operation whose users all truncate it to i32 (as
/// narrowSignedCompares leaves them, the compares in other blocks than the
/// operations: instruction selection sees no trunc of them) as the 32-bit
/// operation of its operands' low words, which are exact modulo 2^32. Its
/// operands' truncations may in turn make them such operations.
static bool narrowTruncatedOps(Function &F) {
  if (!AGXNarrowTruncatedOps)
    return false;
  using namespace PatternMatch;
  auto Narrowable = [](Instruction &I) {
    if (!I.getType()->isIntegerTy(64) || I.use_empty())
      return false;
    auto *B = dyn_cast<BinaryOperator>(&I);
    if (!B)
      return false;
    switch (B->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      break;
    case Instruction::Shl: {
      const APInt *C;
      if (!match(B->getOperand(1), m_APInt(C)) || C->uge(32))
        return false;
      break;
    }
    default:
      return false;
    }
    return llvm::all_of(I.users(), [](User *U) {
      return isa<TruncInst>(U) && U->getType()->isIntegerTy(32);
    });
  };
  bool Changed = false, Again = true;
  // (the trees that earlier narrowing left dead, and truncations of i32
  // extensions, would keep operations' other users)
  SmallVector<WeakTrackingVH, 16> Dead;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    Value *X;
    if (I.getType()->isIntegerTy(32) &&
        match(&I, m_Trunc(m_ZExtOrSExt(m_Value(X)))) &&
        X->getType()->isIntegerTy(32)) {
      Value *E = I.getOperand(0);
      I.replaceAllUsesWith(X);
      I.eraseFromParent();
      Dead.push_back(E);
      Changed = true;
    } else if (isInstructionTriviallyDead(&I)) {
      Dead.push_back(&I);
    }
  }
  for (WeakTrackingVH &V : Dead)
    if (auto *I = dyn_cast_or_null<Instruction>(V))
      Changed |= RecursivelyDeleteTriviallyDeadInstructions(I);
  while (Again) {
    Again = false;
    SmallVector<BinaryOperator *, 8> Ops;
    for (Instruction &I : instructions(F))
      if (Narrowable(I))
        Ops.push_back(cast<BinaryOperator>(&I));
    for (BinaryOperator *B : Ops) {
      IRBuilder<> IRB(B);
      Type *I32 = IRB.getInt32Ty();
      auto Low = [&](Value *V) -> Value * {
        Value *X;
        if ((match(V, m_ZExt(m_Value(X))) || match(V, m_SExt(m_Value(X)))) &&
            X->getType()->isIntegerTy(32))
          return X;
        return IRB.CreateTrunc(V, I32);
      };
      Value *N = IRB.CreateBinOp(B->getOpcode(), Low(B->getOperand(0)),
                                 Low(B->getOperand(1)), B->getName());
      for (User *U : make_early_inc_range(B->users())) {
        U->replaceAllUsesWith(N);
        cast<Instruction>(U)->eraseFromParent();
      }
      B->eraseFromParent();
      Changed = Again = true;
    }
  }
  return Changed;
}

static cl::opt<bool> AGXSpeculateThreadgroupLoads(
    "agx-speculate-threadgroup-loads", cl::init(true), cl::Hidden,
    cl::desc("Run a guarded threadgroup load that stays within its global "
             "unconditionally, a select in place of the if"));

/// A: br c, T, J; T: threadgroup loads within their globals and arithmetic
/// that may run anywhere; br J; J: phis (v = c ? sm[l] : x, the PTX front
/// end's block reductions): T's code runs in A and J's phis are selects. The
/// loads read threadgroup memory every thread may read (each global is
/// allocated for the whole threadgroup, and racing stores make a load's
/// value undefined, not the program, which then selects the other value):
/// an if, its skip and pop fewer, for a select.
static bool speculateThreadgroupLoads(Function &F) {
  if (!AGXSpeculateThreadgroupLoads)
    return false;
  struct Candidate {
    BasicBlock *A, *T, *J;
  };
  SmallVector<Candidate, 4> Candidates;
  for (BasicBlock &BB : F) {
    auto *Br = dyn_cast<CondBrInst>(BB.getTerminator());
    if (!Br)
      continue;
    for (unsigned K = 0; K < 2; ++K) {
      BasicBlock *T = Br->getSuccessor(K), *J = Br->getSuccessor(1 - K);
      if (T == J || T == &BB || J == &BB || T->getSinglePredecessor() != &BB ||
          T->getSingleSuccessor() != J ||
          !isa<UncondBrInst>(T->getTerminator()) || pred_size(J) != 2 ||
          isa<PHINode>(T->front()) || T->size() > 17 ||
          std::distance(J->phis().begin(), J->phis().end()) != 1)
        continue;
      bool HasLoad = false, Ok = true;
      for (Instruction &I : *T) {
        if (I.isTerminator())
          continue;
        if (auto *L = dyn_cast<LoadInst>(&I)) {
          Ok &= L->isSimple() && L->getPointerAddressSpace() == 3;
          HasLoad = true;
        } else {
          Ok &= isSafeToSpeculativelyExecute(&I) && !I.mayReadOrWriteMemory();
        }
      }
      if (Ok && HasLoad)
        Candidates.push_back({&BB, T, J});
    }
  }
  if (Candidates.empty())
    return false;
  SmallVector<WeakVH, 8> Ranged;
  AGX::addSpecialRegisterRanges(F, &Ranged);
  scope_exit Unrange([&] {
    for (WeakVH &V : Ranged)
      if (auto *CI = dyn_cast_or_null<CallBase>(V))
        CI->removeRetAttr(Attribute::Range);
  });
  ThreadgroupAccesses TG;
  TG.build(F);
  const DataLayout &DL = F.getDataLayout();
  SmallVector<Candidate, 4> Safe;
  for (Candidate &C : Candidates) {
    // (within at A's end: the bounds must hold without T's condition)
    bool Ok = true;
    for (Instruction &I : *C.T)
      if (auto *L = dyn_cast<LoadInst>(&I))
        Ok &= TG.within(L->getPointerOperand(),
                        DL.getTypeStoreSize(L->getType()),
                        C.A->getTerminator()) != nullptr;
    if (Ok)
      Safe.push_back(C);
  }
  for (Candidate &C : Safe) {
    auto *Br = cast<CondBrInst>(C.A->getTerminator());
    bool TOnTrue = Br->getSuccessor(0) == C.T;
    Value *Cond = Br->getCondition();
    C.T->getTerminator()->eraseFromParent();
    C.A->splice(Br->getIterator(), C.T);
    IRBuilder<> B(Br);
    for (PHINode &P : make_early_inc_range(C.J->phis())) {
      Value *VT = P.getIncomingValueForBlock(C.T);
      Value *VA = P.getIncomingValueForBlock(C.A);
      Value *Sel = TOnTrue ? B.CreateSelect(Cond, VT, VA, P.getName())
                           : B.CreateSelect(Cond, VA, VT, P.getName());
      P.replaceAllUsesWith(Sel);
      P.eraseFromParent();
    }
    UncondBrInst::Create(C.J, Br->getIterator());
    Br->eraseFromParent();
    C.T->eraseFromParent();
  }
  return !Safe.empty();
}

static cl::opt<bool> AGXMaskedTailExits(
    "agx-masked-tail-exits", cl::init(true), cl::Hidden,
    cl::desc("A loop whose reductions take an identity once a mask on an "
             "increasing index holds (a causal mask) exits there"));

/// A loop whose only side effects are loads, and whose values carried
/// around it (other than its induction) are reductions that a mask leaves
/// unchanged, P' = op(P, select(m, identity, x)) or P' = select(m, P, y),
/// where m is a compare of an increasing recurrence with a loop invariant
/// that, once true, stays true (j > k, j >= k: a causal mask's j > row),
/// leaves the loop as soon as the next iteration would be masked: those
/// iterations change nothing (L15-E476).
static cl::opt<bool> AGXMaskedHeadStarts(
    "agx-masked-head-starts", cl::init(true), cl::Hidden,
    cl::desc("agx-masked-tail-exits also starts a loop at the first "
             "iteration a mask true up to some iteration leaves (a sliding "
             "window's lower edge)"));

/// The induction's start S0 (step s, the loop going on while iv.next < UB)
/// advanced past the iterations where the mask A < K (A <= K) holds, A an
/// increasing recurrence with the induction's step: by k = ceil(max(K -
/// A0, 0) / s) steps (K + 1 for <=), but at most to the last iteration the
/// loop ran, k_max = max(UB - (S0 + 1), 0) / s, which the mask then also
/// covers (the loads stay those of iterations that ran) (L15-E481).
static bool raiseMaskedHead(Loop *L, PHINode *IV, BinaryOperator *IVNext,
                            ICmpInst *Exit,
                            const std::tuple<Value *, Value *, CmpPredicate> &M,
                            ScalarEvolution &SE, DominatorTree &DT,
                            Value **Clamped = nullptr) {
  Value *UB = Exit->getOperand(1);
  auto [A, K, Pred] = M;
  auto *ARA = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(A));
  auto *ARI = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(IV));
  auto *Step = dyn_cast<ConstantInt>(IVNext->getOperand(1));
  if (!ARA || !ARI || !Step || IVNext->getOperand(0) != IV ||
      ARA->getStepRecurrence(SE) != ARI->getStepRecurrence(SE) ||
      A->getType() != IV->getType() || K->getType() != IV->getType() ||
      !Step->getValue().isStrictlyPositive())
    return false;
  BasicBlock *PH = L->getLoopPredecessor();
  if (auto *KI = dyn_cast<Instruction>(K);
      KI && !DT.dominates(KI, PH->getTerminator()))
    return false;
  const SCEV *A0 = ARA->getStart();
  SCEVExpander Exp(SE, "maskhead");
  if (!Exp.isSafeToExpand(A0))
    return false;
  Instruction *At = PH->getTerminator();
  Value *A0V = Exp.expandCodeFor(A0, A->getType(), At);
  IRBuilder<> B(At);
  Value *S0 = IV->getIncomingValueForBlock(PH);
  // (in 32 bits where every input fits, extended: the start's range then
  // shows, which the indices' 32-bit rebasing needs; K + 1 saturating at
  // 2^32 - 1 there only advances the start less)
  Type *Wide = IV->getType();
  Type *Ty = Wide;
  if (Wide->getIntegerBitWidth() > 32 &&
      all_of(std::initializer_list<const SCEV *>{SE.getSCEV(UB), SE.getSCEV(K),
                                                 A0, SE.getSCEV(S0)},
             [&](const SCEV *X) {
               return SE.getUnsignedRangeMax(X).isIntN(32);
             }) &&
      Step->getValue().ult(1u << 16)) {
    Ty = B.getInt32Ty();
    UB = B.CreateTrunc(UB, Ty);
    K = B.CreateTrunc(K, Ty);
    A0V = B.CreateTrunc(A0V, Ty);
    S0 = B.CreateTrunc(S0, Ty);
    Step = ConstantInt::get(cast<IntegerType>(Ty), Step->getZExtValue());
  }
  Value *One = ConstantInt::get(Ty, 1);
  Value *KK = Pred == ICmpInst::ICMP_ULE
                  ? B.CreateBinaryIntrinsic(Intrinsic::uadd_sat, K, One)
                  : K;
  Value *D = B.CreateBinaryIntrinsic(Intrinsic::usub_sat, KK, A0V);
  Value *KHead = B.CreateAdd(
      B.CreateUDiv(D, Step),
      B.CreateZExt(B.CreateICmpNE(B.CreateURem(D, Step),
                                  ConstantInt::get(Ty, 0)),
                   Ty));
  Value *KMax = B.CreateUDiv(
      B.CreateBinaryIntrinsic(
          Intrinsic::usub_sat, UB,
          B.CreateBinaryIntrinsic(Intrinsic::uadd_sat, S0, One)),
      Step);
  Value *Kk = B.CreateBinaryIntrinsic(Intrinsic::umin, KHead, KMax);
  // (the start then the last iteration, still masked: the first unmasked one
  // is past it)
  if (Clamped)
    *Clamped = B.CreateICmpUGT(KHead, KMax, "head.clamped");
  Value *NewS0 = B.CreateAdd(S0, B.CreateMul(Kk, Step, "", /*NUW=*/true),
                             "unmasked.start", /*NUW=*/true);
  if (Ty != Wide)
    NewS0 = B.CreateZExt(NewS0, Wide);
  // umin(start, umax(UB, S0)), the start itself (at most the last
  // iteration, below UB, or S0 where that is not below UB), whose range
  // SCEV then bounds by UB's and S0's (the counters' 32-bit narrowing needs
  // it: it does not see through the saturating subtractions)
  Value *S0W = IV->getIncomingValueForBlock(PH);
  Value *UBW = Exit->getOperand(1);
  NewS0 = B.CreateBinaryIntrinsic(
      Intrinsic::umin, NewS0,
      B.CreateBinaryIntrinsic(Intrinsic::umax, UBW, S0W), nullptr,
      "unmasked.start.b");
  IV->setIncomingValueForBlock(PH, NewS0);
  SE.forgetLoop(L);
  return true;
}

static cl::opt<bool> AGXMaskedTailUnmaskedBody(
    "agx-masked-tail-unmasked-body", cl::init(true), cl::Hidden,
    cl::desc("agx-masked-tail-exits' loop skipped where its first iteration "
             "is masked, the mask then false in it"));

/// maskedTailExits' loop L (one block BB) whose iterations are unmasked from
/// the first on under each of Masks (Pred(A, K): the tail's once its bound is
/// in, A increasing; the head's once its start is raised, true up to an
/// iteration): where the first iteration is masked under one of them all
/// iterations are masked (a tail mask stays on; a raised head start is past
/// the head's masked iterations unless clamped to the last one), and each
/// leaves the reductions as they were, so the preheader branches to the exit
/// with their start values there. Then no iteration runs masked: the mask
/// compares in BB are false (their inverses true) (L15-E518, E519).
static void unmaskBody(Loop *L, PHINode *IV, BasicBlock *PH, BasicBlock *BB,
                       ArrayRef<std::tuple<Value *, Value *, CmpPredicate>> Masks,
                       ArrayRef<Value *> Firsts, DominatorTree &DT, LoopInfo &LI,
                       ScalarEvolution &SE) {
  using namespace PatternMatch;
  if (!PH)
    return;
  // (a preheader and an exit of the loop's own, as LoopSimplify makes them)
  if (!L->hasDedicatedExits())
    formDedicatedExitBlocks(L, &DT, &LI, nullptr, /*PreserveLCSSA=*/false);
  if (!L->getLoopPreheader()) {
    BasicBlock *NewPH =
        InsertPreheaderForLoop(L, &DT, &LI, nullptr, /*PreserveLCSSA=*/false);
    if (!NewPH)
      return;
    PH = NewPH;
  }
  BasicBlock *Exit = L->getExitBlock();
  auto *PHBr = dyn_cast<UncondBrInst>(PH->getTerminator());
  if (!Exit || !PHBr || Exit->getSinglePredecessor() != BB ||
      PH->getSingleSuccessor() != BB || L->getLoopPreheader() != PH)
    return;
  // each mask's value at the first iteration: its recurrence's start
  SE.forgetLoop(L);
  SCEVExpander Exp(SE, "unmask");
  for (auto &[A, K, Pred] : Masks) {
    auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(A));
    if (!AR || AR->getLoop() != L || !Exp.isSafeToExpand(AR->getStart()))
      return;
    if (auto *KI = dyn_cast<Instruction>(K);
        KI && !DT.dominates(KI, PHBr))
      return;
  }
  formLCSSA(*L, DT, &LI, &SE);
  // each exit phi's value: a reduction's latch value, whose start it takes
  // from the new edge
  SmallVector<std::pair<PHINode *, Value *>, 4> Starts;
  for (PHINode &X : Exit->phis()) {
    Value *V = X.getIncomingValueForBlock(BB);
    PHINode *R = nullptr;
    for (PHINode &P : BB->phis())
      if (&P != IV && P.getIncomingValueForBlock(BB) == V)
        R = &P;
    if (!R)
      return;
    Starts.push_back({&X, R->getIncomingValueForBlock(PH)});
  }
  // the mask compares: Pred(A, K) itself, or its inverse
  MapVector<ICmpInst *, bool> Cmps;
  for (Instruction &I : *BB) {
    auto *C = dyn_cast<ICmpInst>(&I);
    if (!C)
      continue;
    for (auto &[A, K, Pred] : Masks)
      for (bool Swap : {false, true}) {
        Value *X = C->getOperand(Swap), *Y = C->getOperand(!Swap);
        CmpInst::Predicate P =
            Swap ? C->getSwappedPredicate() : C->getPredicate();
        if (X == A && Y == K) {
          if (P == static_cast<CmpInst::Predicate>(Pred))
            Cmps.insert({C, false});
          else if (P == CmpInst::getInversePredicate(Pred))
            Cmps.insert({C, true});
        }
      }
  }
  if (Cmps.empty())
    return;
  IRBuilder<> B(PHBr);
  Value *Masked0 = nullptr;
  for (unsigned Idx = 0; Idx < Masks.size(); ++Idx) {
    auto &[A, K, Pred] = Masks[Idx];
    // (a raised head's: whether its start was clamped, computed already)
    Value *M = Firsts[Idx];
    if (!M) {
      auto *AR = cast<SCEVAddRecExpr>(SE.getSCEV(A));
      Value *A0 = Exp.expandCodeFor(AR->getStart(), A->getType(), PHBr);
      M = B.CreateICmp(Pred, A0, K, "first.masked");
    }
    Masked0 = Masked0 ? B.CreateOr(Masked0, M) : M;
  }
  B.CreateCondBr(Masked0, Exit, BB);
  PHBr->eraseFromParent();
  for (auto [X, V] : Starts)
    X->addIncoming(V, PH);
  for (auto [C, Inverse] : Cmps) {
    C->replaceAllUsesWith(ConstantInt::getBool(C->getContext(), Inverse));
    C->eraseFromParent();
  }
  // (the selects, ands and ors of the constants)
  const DataLayout &DL = BB->getDataLayout();
  for (bool Again = true; Again;) {
    Again = false;
    for (Instruction &I : make_early_inc_range(*BB))
      if (Value *V = simplifyInstruction(&I, SimplifyQuery(DL))) {
        I.replaceAllUsesWith(V);
        I.eraseFromParent();
        Again = true;
      }
  }
  DT.recalculate(*BB->getParent());
  SE.forgetLoop(L);
}

static cl::opt<bool> AGXMaskedHeadUnmaskedBody(
    "agx-masked-head-unmasked-body", cl::init(true), cl::Hidden,
    cl::desc("agx-masked-tail-unmasked-body also for the head mask of a "
             "raised start"));

static cl::opt<bool> AGXMaskedTailOr(
    "agx-masked-tail-or", cl::init(true), cl::Hidden,
    cl::desc("agx-masked-tail-exits also for a mask that is an or with the "
             "tail compare"));

static bool maskedTailExits(Function &F) {
  if (!AGXMaskedTailExits)
    return false;
  using namespace PatternMatch;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  if (LI.empty())
    return false;
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII, &F);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (!L->isInnermost() || L->getNumBlocks() != 1)
      continue;
    BasicBlock *BB = L->getHeader();
    auto *Br = dyn_cast<CondBrInst>(BB->getTerminator());
    if (!Br || !L->getExitBlock() || !L->getLoopPredecessor() ||
        pred_size(BB) != 2)
      continue;
    bool ContinueOnTrue = Br->getSuccessor(0) == BB;
    // only loads
    if (any_of(*BB, [](Instruction &I) {
          return I.mayWriteToMemory() || I.mayThrow() ||
                 (isa<CallBase>(I) && !isa<IntrinsicInst>(I)) ||
                 (I.mayReadFromMemory() && !isa<LoadInst>(I)) ||
                 (isa<LoadInst>(I) && !cast<LoadInst>(I).isSimple());
        }))
      continue;
    // m: icmp of an increasing recurrence with an invariant, true from
    // some iteration on; the recurrence and the compare as (A, K, Pred)
    // with Pred(A, K) the mask
    // (Head: a mask true up to some iteration instead, A < K or A <= K)
    auto MaskOf = [&](Value *C, bool Inverse = false, bool Head = false)
        -> std::optional<std::tuple<Value *, Value *, CmpPredicate>> {
      auto *Cmp = dyn_cast<ICmpInst>(C);
      if (!Cmp || Cmp->getParent() != BB)
        return std::nullopt;
      Value *A = Cmp->getOperand(0), *K = Cmp->getOperand(1);
      CmpPredicate P = Inverse ? CmpPredicate(Cmp->getInversePredicate())
                               : Cmp->getCmpPredicate();
      if (L->isLoopInvariant(A)) {
        std::swap(A, K);
        P = CmpInst::getSwappedPredicate(P);
      }
      if (!L->isLoopInvariant(K) || !A->getType()->isIntegerTy())
        return std::nullopt;
      bool Signed = ICmpInst::isSigned(P);
      if (!Head && P != ICmpInst::ICMP_UGT && P != ICmpInst::ICMP_UGE &&
          P != ICmpInst::ICMP_SGT && P != ICmpInst::ICMP_SGE)
        return std::nullopt;
      if (Head && P != ICmpInst::ICMP_ULT && P != ICmpInst::ICMP_ULE)
        return std::nullopt;
      auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(A));
      if (!AR || AR->getLoop() != L || !AR->isAffine())
        return std::nullopt;
      auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
      if (!Step || !Step->getAPInt().isStrictlyPositive())
        return std::nullopt;
      // not wrapping: SCEV's flag, or the IR's: the induction phi, or an
      // add of it and an invariant, with nuw (nsw) on the add and on the
      // induction's increment (a wrap would be poison, which the reduction
      // then carries anyway)
      auto NoWrap = [&](Value *V) {
        auto *O = dyn_cast<OverflowingBinaryOperator>(V);
        return O && (Signed ? O->hasNoSignedWrap() : O->hasNoUnsignedWrap());
      };
      auto IVNoWrap = [&](Value *V) {
        auto *Phi = dyn_cast<PHINode>(V);
        if (!Phi || Phi->getParent() != BB)
          return false;
        Value *Inc = Phi->getIncomingValueForBlock(BB);
        return NoWrap(Inc) &&
               (cast<Instruction>(Inc)->getOperand(0) == Phi ||
                cast<Instruction>(Inc)->getOperand(1) == Phi);
      };
      bool Monotone =
          (Signed ? AR->hasNoSignedWrap() : AR->hasNoUnsignedWrap()) ||
          IVNoWrap(A);
      if (!Monotone)
        if (auto *Add = dyn_cast<BinaryOperator>(A);
            Add && Add->getOpcode() == Instruction::Add && NoWrap(Add))
          for (unsigned Op = 0; Op < 2; ++Op)
            if (IVNoWrap(Add->getOperand(Op)) &&
                L->isLoopInvariant(Add->getOperand(1 - Op)))
              Monotone = true;
      if (!Monotone)
        return std::nullopt;
      return std::make_tuple(A, K, P);
    };
    // the reductions' masks: the tail compares (true from some iteration
    // on) and the head compares (true up to some iteration) each implies
    using MaskT = std::tuple<Value *, Value *, CmpPredicate>;
    // (a mask that the compare implies: the compare itself, or an or of it
    // with anything; or, negated, the condition of the unmasked values: the
    // inverse compare, or an and of it with anything, as a sliding window's
    // j <= row && j + w >= row: L15-E480)
    std::function<std::optional<MaskT>(Value *, bool, bool, unsigned)>
        Implied = [&](Value *C, bool Unmasked, bool Head,
                      unsigned D) -> std::optional<MaskT> {
      if (auto M = MaskOf(C, /*Inverse=*/Unmasked, Head))
        return M;
      Value *X, *Y;
      if (D < 3 && AGXMaskedTailOr && isa<Instruction>(C) &&
          cast<Instruction>(C)->getParent() == BB &&
          (Unmasked ? match(C, m_LogicalAnd(m_Value(X), m_Value(Y)))
                    : match(C, m_LogicalOr(m_Value(X), m_Value(Y))))) {
        if (auto M = Implied(X, Unmasked, Head, D + 1))
          return M;
        return Implied(Y, Unmasked, Head, D + 1);
      }
      return std::nullopt;
    };
    auto Same = [](const std::optional<MaskT> &X,
                   const std::optional<MaskT> &Y) {
      return X && Y && std::get<0>(*X) == std::get<0>(*Y) &&
             std::get<1>(*X) == std::get<1>(*Y) &&
             static_cast<CmpInst::Predicate>(std::get<2>(*X)) ==
                 static_cast<CmpInst::Predicate>(std::get<2>(*Y));
    };
    // every reduction's mask implies Mask (the tail) / HeadMask, or it is
    // given up (a first reduction sets them)
    std::optional<MaskT> Mask, HeadMask;
    bool First = true, TailOK = true, HeadOK = AGXMaskedHeadStarts;
    auto SameMask = [&](Value *C, bool Unmasked = false) {
      auto T = Implied(C, Unmasked, false, 0);
      auto H = Implied(C, Unmasked, true, 0);
      if (!T && !H)
        return false;
      if (First) {
        Mask = T;
        HeadMask = H;
        First = false;
      }
      TailOK &= Same(T, Mask);
      HeadOK &= Same(H, HeadMask);
      return TailOK || HeadOK;
    };
    // x is P's identity under op where the mask holds
    auto Identity = [&](Instruction *Op, PHINode *P, Value *I) {
      Value *Init = P->getIncomingValueForBlock(L->getLoopPredecessor());
      auto *CF = dyn_cast<ConstantFP>(I);
      auto InitIs = [&](FPClassTest Never) {
        return computeKnownFPClass(Init, DL, Never).isKnownNever(Never);
      };
      if (Op->getOpcode() == Instruction::FAdd)
        return CF && CF->isZero() &&
               (CF->isNegative() || InitIs(fcNegZero));
      if (auto *II = dyn_cast<IntrinsicInst>(Op)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::maximum:
          return CF && CF->isInfinity() && CF->isNegative();
        case Intrinsic::minimum:
          return CF && CF->isInfinity() && !CF->isNegative();
        case Intrinsic::maxnum:
          return CF && CF->isInfinity() && CF->isNegative() && InitIs(fcNan);
        case Intrinsic::minnum:
          return CF && CF->isInfinity() && !CF->isNegative() && InitIs(fcNan);
        default:
          return false;
        }
      }
      if (Op->getOpcode() == Instruction::Add ||
          Op->getOpcode() == Instruction::Or ||
          Op->getOpcode() == Instruction::Xor)
        return match(I, m_Zero());
      return false;
    };
    bool OK = true;
    PHINode *IV = nullptr;
    for (PHINode &P : BB->phis()) {
      Value *Next = P.getIncomingValueForBlock(BB);
      auto *NI = dyn_cast<Instruction>(Next);
      if (!NI || NI->getParent() != BB) {
        OK = false;
        break;
      }
      // the induction (or another recurrence): left alone, but its value
      // after the loop would change
      if (SE.isSCEVable(P.getType()) && isa<SCEVAddRecExpr>(SE.getSCEV(&P))) {
        for (User *U : NI->users())
          if (cast<Instruction>(U)->getParent() != BB)
            OK = false;
        for (User *U : P.users())
          if (cast<Instruction>(U)->getParent() != BB)
            OK = false;
        IV = &P;
        continue;
      }
      // P' = select(m, P, y), or select(u, y, P), u the unmasked values'
      if (auto *S = dyn_cast<SelectInst>(NI);
          S && ((S->getTrueValue() == &P && SameMask(S->getCondition())) ||
                (S->getFalseValue() == &P &&
                 SameMask(S->getCondition(), /*Unmasked=*/true))))
        continue;
      // P' = op(P, select(m, identity, x)) or op(P, select(u, x, identity)),
      // op commutative here
      bool Found = false;
      if (NI->getNumOperands() >= 2) {
        for (unsigned Idx = 0; Idx < 2 && !Found; ++Idx) {
          if (NI->getOperand(Idx) != &P)
            continue;
          auto *S = dyn_cast<SelectInst>(NI->getOperand(1 - Idx));
          if (!S)
            continue;
          if (Identity(NI, &P, S->getTrueValue()) &&
              SameMask(S->getCondition()))
            Found = true;
          else if (Identity(NI, &P, S->getFalseValue()) &&
                   SameMask(S->getCondition(), /*Unmasked=*/true))
            Found = true;
        }
      }
      if (!Found) {
        OK = false;
        break;
      }
    }
    unsigned IVs = 0;
    for (PHINode &P : BB->phis())
      IVs += SE.isSCEVable(P.getType()) && isa<SCEVAddRecExpr>(SE.getSCEV(&P));
    if (!OK || First || !(TailOK || HeadOK) || !IV)
      continue;
    if (!TailOK)
      Mask.reset();
    if (!HeadOK || IVs != 1)
      HeadMask.reset();
    if (!Mask && !HeadMask)
      continue;
    // after the loop only the reductions' values (theirs at the early exit
    // are their final ones)
    SmallPtrSet<Value *, 8> Finals;
    for (PHINode &P : BB->phis())
      Finals.insert(P.getIncomingValueForBlock(BB));
    if (any_of(*BB, [&](Instruction &I) {
          return !Finals.contains(&I) && any_of(I.users(), [&](User *U) {
                   return cast<Instruction>(U)->getParent() != BB;
                 });
        }))
      continue;
    // The exit: continue while iv.next < UB (unsigned), iv.next = iv + s
    // nuw; the mask A > K (or A >= K), A = iv or iv + c (nuw), c and K
    // invariant: the next iteration is unmasked while iv.next + c <= K, so
    // UB becomes umin(UB, (K + 1) - c) (saturating; K - c for >=), computed
    // before the loop: the exit stays one compare of the induction, for
    // SCEV and the indices' rebasing.
    if (!ContinueOnTrue)
      continue;
    auto *Exit = dyn_cast<ICmpInst>(Br->getCondition());
    if (!Exit || Exit->getPredicate() != ICmpInst::ICMP_ULT ||
        !Exit->hasOneUse())
      continue;
    auto *IVNext = dyn_cast<BinaryOperator>(Exit->getOperand(0));
    Value *UB = Exit->getOperand(1);
    if (!IVNext || IVNext != IV->getIncomingValueForBlock(BB) ||
        IVNext->getOpcode() != Instruction::Add ||
        !IVNext->hasNoUnsignedWrap() || !L->isLoopInvariant(UB) ||
        UB->getType() != IV->getType())
      continue;
    BasicBlock *PH = L->getLoopPredecessor();
    if (auto *UI = dyn_cast<Instruction>(UB);
        UI && !DT.dominates(UI, PH->getTerminator()))
      continue;
    // the tail's bound, then the head's start
    // (the masks the loop's iterations no longer meet once they are known
    // unmasked from the first: the tail's once its bound is in, the head's
    // once its start is raised)
    SmallVector<std::tuple<Value *, Value *, CmpPredicate>, 2> Gone;
    SmallVector<Value *, 2> GoneFirst;
    auto Head = [&] {
      Value *Clamped = nullptr;
      if (HeadMask && raiseMaskedHead(L, IV, IVNext, Exit, *HeadMask, SE, DT,
                                      &Clamped)) {
        Changed = true;
        if (AGXMaskedHeadUnmaskedBody) {
          Gone.push_back(*HeadMask);
          GoneFirst.push_back(Clamped);
        }
      }
      if (!Gone.empty() && AGXMaskedTailUnmaskedBody)
        unmaskBody(L, IV, L->getLoopPredecessor(), BB, Gone, GoneFirst, DT,
                   LI, SE);
    };
    if (!Mask) {
      Head();
      continue;
    }
    auto [A, K, Pred] = *Mask;
    if (ICmpInst::isSigned(Pred) || UB->getType() != A->getType()) {
      Head();
      continue;
    }
    Value *Off = nullptr;
    if (A == IV)
      Off = ConstantInt::get(A->getType(), 0);
    else if (auto *Add = dyn_cast<BinaryOperator>(A);
             Add && Add->getOpcode() == Instruction::Add &&
             Add->hasNoUnsignedWrap())
      for (unsigned Op = 0; Op < 2 && !Off; ++Op)
        if (Add->getOperand(Op) == IV &&
            L->isLoopInvariant(Add->getOperand(1 - Op)))
          Off = Add->getOperand(1 - Op);
    if (!Off) {
      Head();
      continue;
    }
    for (Value *V : {K, Off, UB})
      if (auto *VI = dyn_cast<Instruction>(V); VI && !DT.dominates(VI, PH->getTerminator()))
        Off = nullptr;
    if (!Off) {
      Head();
      continue;
    }
    IRBuilder<> B(PH->getTerminator());
    // max(K - c, 0) (+ 1 for >), the subtraction in the narrower type
    // where K and c are zero extensions of it (exact; extended), the + 1
    // in the wide type, saturating (K - c at the wide maximum masks
    // nothing: the bound is then the maximum, as UB is at most)
    Value *KN = K, *OffN = Off, *X;
    if (match(K, m_ZExt(m_Value(X)))) {
      Type *NT = X->getType();
      Value *Y;
      if (match(Off, m_ZExt(m_Value(Y))) && Y->getType() == NT) {
        KN = X;
        OffN = Y;
      } else if (auto *CI = dyn_cast<ConstantInt>(Off);
                 CI && CI->getValue().isIntN(NT->getIntegerBitWidth())) {
        KN = X;
        OffN = ConstantInt::get(
            NT, CI->getValue().trunc(NT->getIntegerBitWidth()));
      }
    }
    Value *Lim = B.CreateBinaryIntrinsic(Intrinsic::usub_sat, KN, OffN);
    if (KN != K)
      Lim = B.CreateZExt(Lim, K->getType());
    if (Pred == ICmpInst::ICMP_UGT)
      Lim = B.CreateBinaryIntrinsic(Intrinsic::uadd_sat, Lim,
                                    ConstantInt::get(K->getType(), 1));
    Value *NewUB = B.CreateBinaryIntrinsic(Intrinsic::umin, UB, Lim,
                                           nullptr, "unmasked.bound");
    Exit->setOperand(1, NewUB);
    Changed = true;
    // With the loop also skipped where its first iteration is masked (all
    // are then, and each leaves every reduction as it was: the result is
    // their start values), no iteration is masked: the tail compare is false
    // in the loop and the selects take the unmasked values (L15-E518).
    // (after the head's start: the first iteration's mask is the final
    // start's)
    Gone.push_back(*Mask);
    GoneFirst.push_back(nullptr);
    Head();
  }
  return Changed;
}

static cl::opt<bool> AGXByteFloatsALU(
    "agx-byte-floats-alu", cl::init(true), cl::Hidden,
    cl::desc("A signed byte's float as (b ^ 0x6480) read as a half less "
             "1152, on the ALU, instead of the shared unit's convert"));

/// sitofp(i8 b) to float or half: the 16-bit b ^ 0x6480, b's high byte zero,
/// is the half 1024 + (b ^ 128) = 1152 + s (s the signed byte, exact: one
/// unit apart in [1024, 2048)), and less 1152 it is s exactly. An xor and a
/// 16-bit subtraction on the ALU instead of convert s8_to_f, which issues on
/// the shared integer-multiply / bit-field / convert unit (one per two
/// schedulers): gemv_q8f's weights (L15-E533).
static bool byteFloatsALU(Function &F) {
  if (!AGXByteFloatsALU)
    return false;
  bool Changed = false;
  std::optional<DominatorTree> DT;
  std::optional<LoopInfo> LI;
  const DataLayout &DL = F.getDataLayout();
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    auto *C = dyn_cast<CastInst>(&I);
    if (!C || !(isa<SIToFPInst>(C) || isa<UIToFPInst>(C)) ||
        !(C->getType()->isFloatTy() || C->getType()->isHalfTy()) ||
        !C->getOperand(0)->getType()->isIntegerTy() ||
        C->getOperand(0)->getType()->getIntegerBitWidth() > 32)
      continue;
    // the operand as U + K, U below 1024 by its known bits (a nibble, a
    // byte's zero extension: q4 weights' (q & 15) - 8): the half
    // U | 0x6400 is 1024 + U, less 1024 - K it is U + K exactly (integers
    // to 2048 are halves) (L15-E534)
    Value *U = C->getOperand(0);
    int64_t K = 0;
    {
      using namespace PatternMatch;
      const APInt *KC;
      if (match(U, m_Add(m_Value(U), m_APInt(KC))) &&
          KC->getSignificantBits() <= 12)
        K = KC->getSExtValue();
      else
        U = C->getOperand(0);
    }
    // (U + K, its sum in U's type: neither wraps there; for sitofp U's sign
    // clear, for uitofp the sum not negative)
    KnownBits KB = computeKnownBits(U, DL);
    unsigned W = U->getType()->getIntegerBitWidth();
    int64_t MaxU = int64_t(KB.getMaxValue().getZExtValue());
    bool Small = MaxU < 1024 && K >= -1024 && K <= 1024;
    if (Small && isa<SIToFPInst>(C))
      Small = !KB.Zero.isZero() && KB.countMinLeadingZeros() >= 1 &&
              MaxU + K <= (int64_t(1) << (W - 1)) - 1 &&
              K >= -(int64_t(1) << (W - 1));
    else if (Small)
      Small = K >= 0 && MaxU + K <= int64_t((uint64_t(1) << W) - 1);
    bool SignedByte = isa<SIToFPInst>(C) &&
                      C->getOperand(0)->getType()->isIntegerTy(8);
    if (!SignedByte && !Small)
      continue;
    // (in a loop, whose two constants are hoisted: alone they are two
    // mov_imm, four instructions for the convert's one)
    if (!LI) {
      DT.emplace(F);
      LI.emplace(*DT);
    }
    if (!LI->getLoopFor(C->getParent()))
      continue;
    // (only where every user reads a half source, a float's extension
    // folded: float arithmetic; a store or other use would take a
    // conversion of its own, three ALU instructions for the convert's one)
    if (!all_of(C->users(), [](User *U) {
          if (auto *II = dyn_cast<IntrinsicInst>(U))
            return II->getIntrinsicID() == Intrinsic::fma ||
                   II->getIntrinsicID() == Intrinsic::fmuladd;
          unsigned Op = cast<Instruction>(U)->getOpcode();
          return Op == Instruction::FAdd || Op == Instruction::FSub ||
                 Op == Instruction::FMul;
        }))
      continue;
    IRBuilder<> B(C);
    Value *S;
    if (Small && !SignedByte) {
      Value *Z = B.CreateZExtOrTrunc(U, B.getInt16Ty());
      Value *H = B.CreateBitCast(B.CreateOr(Z, 0x6400), B.getHalfTy());
      S = B.CreateFSub(H, ConstantFP::get(B.getHalfTy(), 1024.0 - K));
    } else {
      Value *Z = B.CreateZExt(C->getOperand(0), B.getInt16Ty());
      Value *H = B.CreateBitCast(B.CreateXor(Z, 0x6480), B.getHalfTy());
      S = B.CreateFSub(H, ConstantFP::get(B.getHalfTy(), 1152.0));
    }
    Value *R = C->getType()->isHalfTy() ? S : B.CreateFPExt(S, C->getType());
    R->takeName(C);
    C->replaceAllUsesWith(R);
    C->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

static cl::opt<bool> AGXThreadDecidedBranches(
    "agx-thread-decided-branches", cl::init(true), cl::Hidden,
    cl::desc("A predecessor's edge into a block whose branch its own "
             "condition decides (and on false, or on true) to that "
             "branch's successor"));

/// A block M ending in br (and C, Y) or br (or C, Y), where a predecessor P
/// branches on C into M on the side that decides M's branch (C false into
/// an and, true into an or): P's edge goes straight to M's successor for
/// that value. An scf.while whose condition guards a load, `in = j < n;
/// v = in ? x[j] : 0; go = in & (v <= t)` (find_first): the loop exits at
/// its header where j reaches n, and the load's block is the latch, without
/// the merge that the guard's if and its phi took (L15-E685). Only where
/// M's successor's phis take, from M, M's own phis or values from outside
/// M (which then dominate P), P is not already its predecessor, and M has
/// nothing the edge would skip: no side effects, its values read only in M
/// or by its successors' phis from it.
static bool threadDecidedBranches(Function &F) {
  if (!AGXThreadDecidedBranches)
    return false;
  using namespace PatternMatch;
  bool Changed = false;
  DominatorTree DT(F);
  for (BasicBlock &M : F) {
    auto *Br = dyn_cast<CondBrInst>(M.getTerminator());
    if (!Br || Br->getSuccessor(0) == Br->getSuccessor(1))
      continue;
    Value *Cond = Br->getCondition(), *A, *Bv;
    bool IsAnd = match(Cond, m_LogicalAnd(m_Value(A), m_Value(Bv)));
    if (!IsAnd && !match(Cond, m_LogicalOr(m_Value(A), m_Value(Bv))))
      continue;
    // (not a loop's header: its preheader's edge stays)
    if (any_of(predecessors(&M),
               [&](BasicBlock *P) { return DT.dominates(&M, P); }))
      continue;
    // (nothing of M's that an edge past it would skip: no side effects, its
    // values read only in M or by its successors' phis from it; a loop's
    // exit block that stores the loop's results was bypassed, the stores
    // lost: agx-loop-fuzz seed 20707)
    if (any_of(M, [&](Instruction &I) {
          if (I.isTerminator())
            return false;
          if (I.mayHaveSideEffects() || I.mayReadFromMemory())
            return true;
          return any_of(I.uses(), [&](Use &U) {
            auto *UI = cast<Instruction>(U.getUser());
            if (UI->getParent() == &M)
              return false;
            auto *PN = dyn_cast<PHINode>(UI);
            return !PN || PN->getIncomingBlock(U) != &M;
          });
        }))
      continue;
    SmallVector<BasicBlock *, 4> Preds(predecessors(&M));
    for (BasicBlock *P : Preds) {
      if (pred_size(&M) < 2)
        break;
      auto *PBr = dyn_cast<CondBrInst>(P->getTerminator());
      if (!PBr || (PBr->getCondition() != A && PBr->getCondition() != Bv) ||
          PBr->getSuccessor(0) == PBr->getSuccessor(1))
        continue;
      unsigned Idx = PBr->getSuccessor(0) == &M ? 0 : 1;
      bool Known = Idx == 0;
      // and(false, _) = false, or(true, _) = true
      if (IsAnd == Known)
        continue;
      BasicBlock *T = Br->getSuccessor(IsAnd ? 1 : 0);
      if (T == &M || is_contained(successors(P), T))
        continue;
      SmallVector<std::pair<PHINode *, Value *>, 4> In;
      bool OK = true;
      for (PHINode &Phi : T->phis()) {
        Value *V = Phi.getIncomingValueForBlock(&M);
        if (auto *VP = dyn_cast<PHINode>(V); VP && VP->getParent() == &M)
          V = VP->getIncomingValueForBlock(P);
        else if (auto *VI = dyn_cast<Instruction>(V);
                 VI && VI->getParent() == &M)
          OK = false;
        In.push_back({&Phi, V});
      }
      if (!OK)
        continue;
      for (auto [Phi, V] : In)
        Phi->addIncoming(V, P);
      for (PHINode &Phi : M.phis())
        Phi.removeIncomingValue(P, /*DeletePHIIfEmpty=*/false);
      PBr->setSuccessor(Idx, T);
      DT.recalculate(F);
      Changed = true;
      // M now entered only past P's other side (its latch, the load's
      // block): C is !Known there, and M's condition the other operand
      BasicBlock *Other = PBr->getSuccessor(1 - Idx);
      auto *CI = dyn_cast<Instruction>(Cond);
      if (CI && CI->getParent() == &M && CI->hasOneUse() &&
          Other->getSinglePredecessor() == P &&
          all_of(predecessors(&M),
                 [&](BasicBlock *Q) { return DT.dominates(Other, Q); })) {
        Value *C = PBr->getCondition();
        Value *Y = A == C ? Bv : A;
        Br->setCondition(Y);
        RecursivelyDeleteTriviallyDeadInstructions(CI);
        break;
      }
    }
  }
  return Changed;
}

static void markExpsBelowLoopMax(Function &F);

bool AGXCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  const TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  const TargetMachine &TM =
      getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<AGXSubtarget>(F);
  bool Changed = diagnoseDoubles(F);
  // (first: the loops' maxima as the front end wrote them)
  markExpsBelowLoopMax(F);
  Changed |= threadDecidedBranches(F);
  Changed |= mergeSameConditionIfs(F);
  Changed |= mergePhiOperations(F);
  Changed |= signedWideMuls(F);
  // (before the min / max flushes, which hide the NaN selects' operands)
  Changed |= foldNeverNaNTests(F);
  Changed |= zeroSignFreeMinMax(F);
  // (after the zero-sign-free min / max: an nsz link lowers without the fix
  // this defers, to as few instructions)
  Changed |= deferredMinMaxReductions(F);
  Changed |= combineDeferredMinMax(F);
  Changed |= onlineSoftmaxExps(F);
  Changed |= rebaseLoopIndices(F);
  Changed |= relativeIndexSelects(F);
  Changed |= reluZeroSelects(F);
  Changed |= exactDyadicSums(F);
  // (after: exactDyadicSums takes converted bytes' sums as integer ones)
  Changed |= byteFloatsALU(F);
  Changed |= AGX::narrowBoundedI64(F);
  // (after the indices are rebased and narrowed, which need the loops' own
  // exit compares)
  Changed |= maskedTailExits(F);
  Changed |= narrowSignedCompares(F);
  Changed |= narrowTruncatedOps(F);
  Changed |= halfFromFloat(F);
  // (before the matrix FMAs become i64 operands, which the fold does not see through)
  Changed |= foldLaneIdentities(F);
  // (before the scan recognition, which matches the folded segment tests)
  Changed |= rangedCompares(F);
  Changed |= simdScans(F, ST);
  Changed |= simdReductions(F, ST);
  Changed |= ownValueShifts(F);
  Changed |= mergeLoopExits(F);
  Changed |= forwardOwnSlots(F);
  Changed |= strideDivisions(F);
  Changed |= lowerMatrixFMA(F, ST);
  Changed |= lanePermutations(F);
  if (ST.hasQuadShuffleIndexMerge())
    Changed |= lowerDivergentShuffles(F, TTI);
  Changed |= combineUniformAtomics(F, TTI);
  Changed |= lowerConstrainedFP(F);
  Changed |= expandMath(F, ST);
  Changed |= bfloatSqrts(F, ST);
  Changed |= flushHalfDenormals(F);
  Changed |= flushFloatMinMax(F);
  Changed |= expandDivisions(F);
  // (after the expansions: the exp argument's clamps, the divisions' uses)
  Changed |= halfSelects(F);
  Changed |= mergeMaskedOrs(F);
  Changed |= permuteBytes(F);
  Changed |= loadBFloatsAsWords(F);
  Changed |= foldConditionChains(F, TTI);
  Changed |= nonNegXorDivisions(F);
  Changed |= floorModulos(F);
  // (last: the lane analyses above see the SIMD-group shuffles)
  Changed |= quadXorShuffles(F);
  Changed |= quadShifts(F);
  Changed |= convertedMinMaxNum(F);
  Changed |= satConvertClamps(F);
  Changed |= loadBytesApart(F);
  Changed |= storeBytesTogether(F);
  Changed |= sinkMulExtensions(F);
  Changed |= hoistInvariantAddressParts(F);
  Changed |= narrowMaskedIndices(F);
  Changed |= expandMemIntrinsics(F, TTI);
  Changed |= fenceHostCoherentAccesses(F);
  Changed |= useBlockBarriers(F, ST);
  // (late: the folds above see the phis of the guarded loads)
  Changed |= speculateThreadgroupLoads(F);
  Changed |= unifyExits(F);
  return Changed;
}

static cl::opt<bool> AGXAtomicOptimizer(
    "agx-atomic-optimizer", cl::init(false), cl::Hidden,
    cl::desc("Combine the atomics of a SIMD-group to one uniform address into "
             "one atomic of the group's reduction (as AMDGPUAtomicOptimizer)"));

/// The AGX SIMD-group reduction operation (AGX::SIMDOp numbering of
/// llvm.agx.simd.reduce) for an atomicrmw, or -1.
static int simdOpFor(const AtomicRMWInst &RMW) {
  bool Int32 = RMW.getType()->isIntegerTy(32);
  switch (RMW.getOperation()) {
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
    return Int32 ? 16 : -1;
  case AtomicRMWInst::And:
    return Int32 ? 0 : -1;
  case AtomicRMWInst::Or:
    return Int32 ? 2 : -1;
  case AtomicRMWInst::Xor:
    return Int32 ? 4 : -1;
  case AtomicRMWInst::Min:
    return Int32 ? 20 : -1;
  case AtomicRMWInst::Max:
    return Int32 ? 22 : -1;
  case AtomicRMWInst::UMin:
    return Int32 ? 28 : -1;
  case AtomicRMWInst::UMax:
    return Int32 ? 30 : -1;
  default:
    // Not fadd: a reduction rounds and flushes denormals unlike the atomic
    // (PTX-formalization atom-f32-ftz).
    return -1;
  }
}

/// Atomics of every active thread of a SIMD-group to one (uniform) address
/// become one atomic, done by the first active thread, of the reduction of
/// their operands; the others' results are rebuilt from the old value and an
/// exclusive scan (only for add and sub, which have one). 32-bit integers. Same-address
/// atomics serialize in the memory system, which agx_cost does not model:
/// off by default until hardware measurements decide it (EXPERIMENTS.md).
bool AGXCodeGenPrepare::combineUniformAtomics(Function &F,
                                              const TargetTransformInfo &TTI) {
  if (!AGXAtomicOptimizer)
    return false;
  SmallVector<AtomicRMWInst *, 8> Candidates;
  for (Instruction &I : instructions(F))
    if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
      if (!RMW->isVolatile() && simdOpFor(*RMW) >= 0)
        Candidates.push_back(RMW);
  if (Candidates.empty())
    return false;
  DominatorTree DT(F);
  CycleInfo CI;
  CI.compute(F);
  UniformityInfo UI(DT, CI, &TTI);
  UI.compute();

  bool Changed = false;
  for (AtomicRMWInst *RMW : Candidates) {
    Value *Ptr = RMW->getPointerOperand();
    if (UI.isDivergentAtUse(RMW->getOperandUse(AtomicRMWInst::getPointerOperandIndex())))
      continue;
    AtomicRMWInst::BinOp Op = RMW->getOperation();
    bool Scan = Op == AtomicRMWInst::Add || Op == AtomicRMWInst::Sub;
    if (!RMW->use_empty() && !Scan)
      continue;
    Type *Ty = RMW->getType();
    Value *V = RMW->getValOperand();
    int HWOp = simdOpFor(*RMW);
    IRBuilder<> B(RMW);
    Value *Total, *Before = nullptr;
    // From the ballot of the active threads (the active-thread special
    // registers disagree with it in agx_run).
    Value *Ballot = B.CreateIntrinsic(Intrinsic::agx_simd_ballot,
                                      ArrayRef<Type *>(), {B.getTrue()});
    Value *Lane = B.CreateIntrinsic(Intrinsic::agx_thread_index_in_simdgroup,
                                    ArrayRef<Type *>(), ArrayRef<Value *>());
    Value *Active =
        B.CreateIntrinsic(Intrinsic::ctpop, {B.getInt32Ty()}, {Ballot});
    Value *Below = B.CreateAnd(
        Ballot, B.CreateSub(B.CreateShl(B.getInt32(1), Lane), B.getInt32(1)));
    Value *Index =
        B.CreateIntrinsic(Intrinsic::ctpop, {B.getInt32Ty()}, {Below});
    if (!UI.isDivergentAtUse(RMW->getOperandUse(1)) && Ty->isIntegerTy()) {
      // A uniform operand: n times it for add and sub, itself otherwise.
      bool Sum = Op == AtomicRMWInst::Add || Op == AtomicRMWInst::Sub;
      Total = Sum ? B.CreateMul(V, Active) : V;
      if (Sum && !RMW->use_empty())
        Before = B.CreateMul(V, Index);
    } else {
      Total = B.CreateIntrinsic(Intrinsic::agx_simd_reduce, {Ty},
                                {V, B.getInt32(HWOp)});
      if (!RMW->use_empty())
        Before = B.CreateIntrinsic(Intrinsic::agx_simd_prefix, {Ty},
                                   {V, B.getInt32(HWOp)});
    }
    Value *First = B.CreateICmpEQ(Index, B.getInt32(0));
    Instruction *Then = SplitBlockAndInsertIfThen(First, RMW, false);
    BasicBlock *ThenBB = Then->getParent(), *Tail = RMW->getParent();
    B.SetInsertPoint(Then);
    AtomicRMWInst *One = B.CreateAtomicRMW(Op, Ptr, Total, RMW->getAlign(),
                                           RMW->getOrdering(),
                                           RMW->getSyncScopeID());
    One->setVolatile(false);
    if (!RMW->use_empty()) {
      B.SetInsertPoint(RMW);
      PHINode *Old = B.CreatePHI(Ty, 2);
      Old->addIncoming(One, ThenBB);
      Old->addIncoming(PoisonValue::get(Ty), ThenBB->getSinglePredecessor());
      // The first active thread's lane: the lowest set bit of the ballot.
      Value *FirstLane = B.CreateIntrinsic(Intrinsic::cttz, {B.getInt32Ty()},
                                           {Ballot, B.getTrue()});
      Value *Base = B.CreateIntrinsic(Intrinsic::agx_simd_shuffle, {Ty},
                                      {Old, FirstLane});
      Value *Result = Op == AtomicRMWInst::Sub ? B.CreateSub(Base, Before)
                                               : B.CreateAdd(Base, Before);
      RMW->replaceAllUsesWith(Result);
    }
    (void)Tail;
    RMW->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

/// powi(x, n) with a variable n (float, half or bfloat, in float):
/// pow(|x|, n) (AGX has no powi library function, and SelectionDAG's
/// fallback, pow(x, n), is NaN for a negative x), with the sign of x for an
/// odd n. A constant n is left to SelectionDAG's multiplications.
static bool expandPowI(Function &F) {
  SmallVector<IntrinsicInst *, 4> Calls;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->getIntrinsicID() == Intrinsic::powi &&
        !isa<Constant>(II->getArgOperand(1)) &&
        (II->getType()->isFloatTy() || II->getType()->isHalfTy() ||
         II->getType()->isBFloatTy()))
      Calls.push_back(II);
  for (IntrinsicInst *II : Calls) {
    IRBuilder<> B(II);
    B.setFastMathFlags(II->getFastMathFlags());
    Type *F32 = B.getFloatTy();
    Value *X = B.CreateFPExt(II->getArgOperand(0), F32);
    Value *N = II->getArgOperand(1);
    Value *Mag = B.CreateBinaryIntrinsic(
        Intrinsic::pow, B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
        B.CreateSIToFP(N, F32));
    Value *Odd = B.CreateTrunc(N, B.getInt1Ty());
    Value *R = B.CreateCopySign(
        Mag, B.CreateSelect(Odd, X, ConstantFP::get(F32, 1.0)));
    R = B.CreateFPTrunc(R, II->getType());
    R->takeName(II);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
  }
  return !Calls.empty();
}

/// frem of float, half and bfloat (in float): fmod, exact (an fmod of
/// half or bfloat values is one, so the truncation back is exact). A step reduces a by a
/// divisor d with a / d < 2^24: q = trunc(a / d) is the integer part n or,
/// rounded up, n + 1 (the quotient is correctly rounded and n representable),
/// so the remainder is a - q d or a - (q - 1) d, whichever is in [0, d)
/// (exact, or flushed). While a exceeds b 2^23 by more than an exponent
/// step, a is first reduced by b 2^j, a multiple of b, with a / (b 2^j) <
/// 2^24 (each step takes 23 bits off the exponent). A denormal b (IEEE float)
/// is normalized by 2^24 for the loop, with a last step by b itself. The
/// result has the sign of a; fmod(x, +-inf) is x (also a denormal x), and
/// NaN for an infinite or NaN x, a NaN or zero y.
static bool expandFRem(Function &F, const AGXSubtarget &ST) {
  SmallVector<BinaryOperator *, 4> Rems;
  for (Instruction &I : instructions(F))
    if (I.getOpcode() == Instruction::FRem &&
        (I.getType()->isFloatTy() || I.getType()->isHalfTy() ||
         I.getType()->isBFloatTy()))
      Rems.push_back(cast<BinaryOperator>(&I));
  bool DenormalB =
      ST.hasIEEEFP32() &&
      F.getDenormalMode(APFloat::IEEEsingle()).Input == DenormalMode::IEEE;
  for (BinaryOperator *Rem : Rems) {
    LLVMContext &Ctx = F.getContext();
    BasicBlock *Before = Rem->getParent();
    BasicBlock *After = Before->splitBasicBlock(Rem, "frem.done");
    BasicBlock *Header = BasicBlock::Create(Ctx, "frem.loop", &F, After);
    BasicBlock *Body = BasicBlock::Create(Ctx, "frem.reduce", &F, After);
    Before->getTerminator()->eraseFromParent();

    IRBuilder<> B(Before);
    Type *F32 = B.getFloatTy(), *I32 = B.getInt32Ty();
    auto K = [&](float V) { return ConstantFP::get(F32, V); };
    auto Bits = [&](Value *V) { return B.CreateBitCast(V, I32); };
    auto Exponent = [&](Value *V) { return B.CreateLShr(Bits(V), 23); };
    // a mod d for a / d < 2^24 (a, d positive)
    auto Step = [&](Value *A, Value *D) {
      Value *Q = B.CreateUnaryIntrinsic(Intrinsic::trunc, B.CreateFDiv(A, D));
      auto Rem = [&](Value *Q) {
        return B.CreateIntrinsic(Intrinsic::fma, {F32},
                                 {B.CreateFNeg(Q), D, A});
      };
      // With q one too large, a - (q - 1) d is the remainder (below d);
      // with the right q it is at least d. (a - q d < 0 would not show an
      // overestimate whose remainder is flushed.)
      Value *Low = Rem(B.CreateFSub(Q, K(1.0f)));
      return B.CreateSelect(B.CreateFCmpOLT(Low, D), Low, Rem(Q));
    };

    Value *X = B.CreateFPExt(Rem->getOperand(0), F32);
    Value *Y = B.CreateFPExt(Rem->getOperand(1), F32);
    Value *AX = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
    Value *AY = B.CreateUnaryIntrinsic(Intrinsic::fabs, Y);
    Value *BN = AY;
    if (DenormalB)
      BN = B.CreateSelect(B.CreateFCmpOLT(AY, K(0x1p-126f)),
                          B.CreateFMul(AY, K(0x1p24f)), AY);
    // NaN, or x itself for an infinite y: no reduction.
    Value *NaN = B.CreateOr(B.CreateFCmpUGE(AX, K(INFINITY)),
                            B.CreateFCmpULE(BN, K(0.0f)));
    Value *YInf = B.CreateFCmpOEQ(AY, K(INFINITY));
    Value *Start = B.CreateSelect(B.CreateOr(NaN, YInf), K(0.0f), AX);
    Value *EB = Exponent(BN);
    B.CreateBr(Header);

    B.SetInsertPoint(Header);
    PHINode *A = B.CreatePHI(F32, 2);
    A->addIncoming(Start, Before);
    Value *J = B.CreateSub(B.CreateSub(Exponent(A), EB), B.getInt32(23));
    B.CreateCondBr(B.CreateICmpSGT(J, B.getInt32(0)), Body, After);

    B.SetInsertPoint(Body);
    Value *BK = B.CreateBitCast(B.CreateAdd(Bits(BN), B.CreateShl(J, 23)), F32);
    A->addIncoming(Step(A, BK), Body);
    B.CreateBr(Header);

    B.SetInsertPoint(Rem);
    Value *R = Step(A, BN);
    if (DenormalB)
      R = Step(R, AY);
    R = B.CreateCopySign(R, X);
    R = B.CreateSelect(YInf, X, R);
    R = B.CreateSelect(NaN, K(NAN), R);
    R = B.CreateFPTrunc(R, Rem->getType());
    R->takeName(Rem);
    Rem->replaceAllUsesWith(R);
    Rem->eraseFromParent();
  }
  return !Rems.empty();
}

static cl::opt<bool> AGXIntToBFloatOnce(
    "agx-int-to-bf16-once", cl::init(true), cl::Hidden,
    cl::desc("Convert integers of more than 24 bits to bfloat with one "
             "rounding (rounded to odd at 24 bits first)"));

static cl::opt<bool> AGXIntToBFloatRTZ(
    "agx-int-to-bf16-rtz", cl::init(true), cl::Hidden,
    cl::desc("A 32-bit integer to bfloat rounded to odd through the float "
             "conversion toward zero"));

/// sitofp / uitofp of an integer of more than 24 bits to bfloat, rounded once:
/// instruction selection converts to float (rounded) and rounds that to
/// bfloat again, which is wrong where the first rounding lands on a bfloat
/// halfway point (21430271 came out 0x4ba4, not 0x4ba3). |x| is first rounded
/// to odd at 24 significant bits (the bits below cleared, the lowest kept one
/// set where any of them was): that converts to float exactly, and rounding
/// it to nearest bfloat is rounding x once (float keeps the 2 extra bits).
static bool convertIntsToBFloatOnce(Function &F) {
  if (!AGXIntToBFloatOnce)
    return false;
  SmallVector<CastInst *, 4> Casts;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<CastInst>(&I);
        C && (isa<SIToFPInst>(C) || isa<UIToFPInst>(C)) &&
        C->getType()->isBFloatTy() &&
        C->getSrcTy()->getIntegerBitWidth() > 24)
      Casts.push_back(C);
  for (CastInst *C : Casts) {
    IRBuilder<> B(C);
    Value *X = C->getOperand(0);
    auto *Ty = cast<IntegerType>(X->getType());
    unsigned W = Ty->getBitWidth();
    bool Signed = isa<SIToFPInst>(C);
    auto K = [&](uint64_t V) { return ConstantInt::get(Ty, V); };
    // 32 bits: the float conversion toward zero, made odd where it was
    // inexact (the conversion back differs), is x rounded to odd at 24 bits,
    // signed or not (toward zero rounds the magnitude); then its bits
    // rounded to nearest bfloat (never a NaN) (L15-E316).
    if (W == 32 && AGXIntToBFloatRTZ) {
      Value *F = B.CreateIntrinsic(Signed ? Intrinsic::agx_sitofp_rtz
                                          : Intrinsic::agx_uitofp_rtz,
                                   {}, {X});
      Value *Back = Signed ? B.CreateFPToSI(F, Ty) : B.CreateFPToUI(F, Ty);
      Value *Bits = B.CreateBitCast(F, Ty);
      Value *Odd = B.CreateOr(Bits, B.CreateZExt(B.CreateICmpNE(Back, X), Ty));
      Value *Lsb = B.CreateAnd(B.CreateLShr(Odd, 16), K(1));
      Value *R16 = B.CreateTrunc(
          B.CreateLShr(B.CreateAdd(Odd, B.CreateAdd(Lsb, K(0x7fff))), 16),
          B.getInt16Ty());
      Value *R = B.CreateBitCast(R16, C->getType());
      R->takeName(C);
      C->replaceAllUsesWith(R);
      C->eraseFromParent();
      continue;
    }
    Value *A = Signed ? B.CreateBinaryIntrinsic(Intrinsic::abs, X, B.getFalse())
                      : X;
    // bits below the 24 significant ones: max(W - ctlz(a) - 24, 0)
    Value *S = B.CreateSub(K(W - 24),
                           B.CreateBinaryIntrinsic(
                               Intrinsic::umin,
                               B.CreateBinaryIntrinsic(Intrinsic::ctlz, A,
                                                       B.getFalse()),
                               K(W - 24)));
    Value *Low = B.CreateSub(B.CreateShl(K(1), S), K(1));
    Value *Sticky = B.CreateZExt(B.CreateICmpNE(B.CreateAnd(A, Low), K(0)), Ty);
    Value *Odd = B.CreateOr(B.CreateAnd(A, B.CreateNot(Low)),
                            B.CreateShl(Sticky, S));
    Value *R = B.CreateFPTrunc(B.CreateUIToFP(Odd, B.getFloatTy()),
                               C->getType());
    if (Signed)
      R = B.CreateSelect(B.CreateICmpSLT(X, K(0)), B.CreateFNeg(R), R);
    R->takeName(C);
    C->replaceAllUsesWith(R);
    C->eraseFromParent();
  }
  return !Casts.empty();
}

/// lround / llround / lrint / llrint (no libcalls on AGX; half could not be
/// selected): the integral value (round: halfway away from zero; rint: to
/// nearest even, the only rounding direction here) converted, frozen (an
/// out-of-range or NaN operand gives an unspecified value, not poison).
static bool expandRoundToInt(Function &F) {
  SmallVector<IntrinsicInst *, 4> Calls;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      switch (II->getIntrinsicID()) {
      case Intrinsic::lround:
      case Intrinsic::llround:
      case Intrinsic::lrint:
      case Intrinsic::llrint:
        Calls.push_back(II);
        break;
      default:
        break;
      }
  for (IntrinsicInst *II : Calls) {
    IRBuilder<> B(II);
    Intrinsic::ID ID = II->getIntrinsicID();
    bool Away = ID == Intrinsic::lround || ID == Intrinsic::llround;
    Value *X = II->getArgOperand(0);
    Value *R = B.CreateUnaryIntrinsic(Away ? Intrinsic::round : Intrinsic::rint, X);
    R = B.CreateFreeze(B.CreateFPToSI(R, II->getType()));
    R->takeName(II);
    II->replaceAllUsesWith(R);
    II->eraseFromParent();
  }
  return !Calls.empty();
}

static cl::opt<bool> AGXExpBelowLoopMax(
    "agx-exp-below-loop-max",
    cl::desc("exp(x - m) without its overflow tests where m is an earlier "
             "loop's maximum of the values x reads"),
    cl::init(true), cl::Hidden);

/// m with its single-incoming phis (LCSSA) looked through.
static Value *stripSinglePhis(Value *V) {
  while (auto *P = dyn_cast<PHINode>(V)) {
    if (P->getNumIncomingValues() != 1)
      break;
    V = P->getIncomingValue(0);
  }
  return V;
}

static cl::opt<bool> AGXExpExtendedLoads(
    "agx-exp-extended-loads", cl::init(true), cl::Hidden,
    cl::desc("agx-exp-below-loop-max also where both loops extend their "
             "loads (halves) to float"));

static cl::opt<bool> AGXExpMaskedMax(
    "agx-exp-masked-max", cl::init(true), cl::Hidden,
    cl::desc("agx-exp-below-loop-max also for a maximum of select(c, -inf, "
             "x) whose exp is used only where the same c is false"));

static cl::opt<bool> AGXExpInPlaceStores(
    "agx-exp-in-place-stores", cl::init(true), cl::Hidden,
    cl::desc("exp(x - m) below an earlier loop's maximum also where x's "
             "loop stores to the address it loaded x from"));

/// The addresses P takes in L's iterations, each accessing Size bytes, do
/// not overlap: P is invariant terms plus one of an affine recurrence of L
/// with a constant step, through zero or sign extensions and a constant
/// scale at least Size whose product cannot wrap, the recurrence not
/// wrapping (its width) over L's maximum trip.
static bool disjointIterations(const SCEV *P, const Loop *L,
                               ScalarEvolution &SE, uint64_t Size) {
  const SCEV *BTC = SE.getConstantMaxBackedgeTakenCount(L);
  auto *N = dyn_cast<SCEVConstant>(BTC);
  if (!N || N->getAPInt().getActiveBits() > 32)
    return false;
  APInt Back = N->getAPInt().zext(128);
  // the variant operand of a sum
  while (auto *A = dyn_cast<SCEVAddExpr>(P)) {
    const SCEV *V = nullptr;
    for (const SCEV *Op : A->operands())
      if (!SE.isLoopInvariant(Op, L)) {
        if (V)
          return false;
        V = Op;
      }
    if (!V)
      return false;
    P = V;
  }
  // a scale: c * ext(r), c at least Size, c * 2^width(r) within P's width
  uint64_t Scale = 1;
  if (auto *M = dyn_cast<SCEVMulExpr>(P)) {
    auto *C = dyn_cast<SCEVConstant>(M->getOperand(0));
    if (M->getNumOperands() != 2 || !C || C->getAPInt().isNegative() ||
        C->getAPInt().getActiveBits() > 32)
      return false;
    Scale = C->getAPInt().getZExtValue();
    P = M->getOperand(1);
    auto *Ext = dyn_cast<SCEVCastExpr>(P);
    if (!isa_and_nonnull<SCEVZeroExtendExpr, SCEVSignExtendExpr>(Ext))
      return false;
    unsigned From = Ext->getOperand()->getType()->getScalarSizeInBits();
    unsigned To = Ext->getType()->getScalarSizeInBits();
    if (From > 64 || To > 64 ||
        !(APInt(128, Scale) << From).ule(APInt(128, 1) << To))
      return false;
  }
  while (isa<SCEVZeroExtendExpr, SCEVSignExtendExpr>(P))
    P = cast<SCEVCastExpr>(P)->getOperand();
  auto *R = dyn_cast<SCEVAddRecExpr>(P);
  if (!R || R->getLoop() != L || !R->isAffine())
    return false;
  auto *S = dyn_cast<SCEVConstant>(R->getStepRecurrence(SE));
  if (!S || S->getAPInt().isZero())
    return false;
  unsigned W = R->getType()->getScalarSizeInBits();
  APInt Step = S->getAPInt().abs().zext(128);
  // r's values distinct mod 2^w: every difference j s, 0 < j <= the
  // backedge-taken count,
  // below 2^w; then the addresses differ by at least the step (unscaled)
  // or the scale, and at most 2^w less that.
  APInt Span = Step * Back;
  APInt Top = APInt(128, 1) << W;
  if (Scale == 1)
    return Step.uge(Size) && (Span + Size).ule(Top);
  return Scale >= Size && Span.ult(Top);
}

/// exp(x - m) where m is an earlier loop's running maximum (maxnum,
/// maximumnum or maximum, every iteration) of a load whose address
/// recurrence {s, +, t} is x's load's, x's loop running at most as many
/// iterations, and nothing that writes memory or synchronizes between the
/// two (but x's loop's stores to x's own address after the load, no two
/// iterations overlapping): each x is a value the maximum took in, so x - m is at most 0 (or a
/// denormal where the maximum's comparison flushes) or a NaN, and e^(x - m)
/// does not overflow (a softmax's or cross-entropy's second pass).
/// m where M decodes the SIMD-group's signed maximum of m's order-preserving
/// integer encoding (MLIR's maximumf subgroup reduction), with a NaN m
/// encoded as 0x7fc00000, above every number:
///   M = float((r < 0 ? 0x7fffffff : 0) ^ r), r = simd.reduce(c, smax),
///   c = m uno m ? 0x7fc00000 : (b < 0 ? 0x7fffffff : 0) ^ b, b = bits(m).
/// M is then a NaN or the largest lane's m, at least this lane's own.
static Value *orderedSIMDMaxOf(Value *M) {
  using namespace PatternMatch;
  auto Ord = [](Value *V) -> Value * {
    Value *X, *Y;
    if (match(V, m_c_Xor(m_Select(m_SpecificICmp(ICmpInst::ICMP_SLT,
                                                 m_Value(X), m_Zero()),
                                  m_SpecificInt(0x7fffffff), m_Zero()),
                         m_Value(Y))) &&
        X == Y)
      return X;
    return nullptr;
  };
  Value *E;
  if (!M->getType()->isFloatTy() || !match(M, m_BitCast(m_Value(E))))
    return nullptr;
  auto *R = dyn_cast_or_null<IntrinsicInst>(Ord(E));
  if (!R || R->getIntrinsicID() != Intrinsic::agx_simd_reduce ||
      !match(R->getArgOperand(1), m_SpecificInt(AGX::SIMD_SMAX)))
    return nullptr;
  Value *Cond, *C;
  if (!match(R->getArgOperand(0),
             m_Select(m_Value(Cond), m_SpecificInt(0x7fc00000), m_Value(C))))
    return nullptr;
  Value *B = Ord(C), *Mv;
  if (!B || !match(B, m_BitCast(m_Value(Mv))) || !Mv->getType()->isFloatTy())
    return nullptr;
  // (uno with itself or with a number)
  CmpPredicate P;
  Value *O;
  if (!match(Cond, m_FCmp(P, m_Specific(Mv), m_Value(O))) ||
      P != FCmpInst::FCMP_UNO ||
      !(O == Mv || (isa<ConstantFP>(O) && !cast<ConstantFP>(O)->isNaN())))
    return nullptr;
  return Mv;
}

static bool expBelowLoopMax(IntrinsicInst *E, LoopInfo &LI,
                            ScalarEvolution &SE) {
  using namespace PatternMatch;
  Value *X, *M;
  if (!match(E->getArgOperand(0), m_FSub(m_Value(X), m_Value(M))) ||
      !E->getType()->isFloatTy())
    return false;
  // A SIMD-group's maximum of it (fmax: NaN-dropping, at least this lane's
  // own value, which it takes in where the reduction runs).
  if (auto *R = dyn_cast<IntrinsicInst>(M);
      R && R->getIntrinsicID() == Intrinsic::agx_simd_reduce &&
      match(R->getArgOperand(1), m_SpecificInt(AGX::SIMD_FMAX)))
    M = R->getArgOperand(0);
  else if (Value *Mv = orderedSIMDMaxOf(M))
    M = Mv;
  // L1's value after it, or (a phi) another where L1 did not run, which L2
  // then does not either (the same guard): the exponential only runs in L2.
  PHINode *Exit = dyn_cast<PHINode>(M);
  Value *MV = stripSinglePhis(M);
  if (Exit && Exit->getNumIncomingValues() > 1) {
    MV = nullptr;
    for (Value *In : Exit->incoming_values())
      if (auto *II = dyn_cast<IntrinsicInst>(stripSinglePhis(In)))
        MV = II;
  }
  auto *U = dyn_cast_or_null<IntrinsicInst>(MV);
  if (!U || !(U->getIntrinsicID() == Intrinsic::maxnum ||
              U->getIntrinsicID() == Intrinsic::maximumnum ||
              U->getIntrinsicID() == Intrinsic::maximum))
    return false;
  Loop *L1 = LI.getLoopFor(U->getParent());
  // (or both extensions of loads, the same exact map of the loaded values:
  // a row of halves, L15-E461)
  bool XExt = false;
  if (auto *E2 = dyn_cast<FPExtInst>(X); E2 && AGXExpExtendedLoads) {
    X = E2->getOperand(0);
    XExt = true;
  }
  auto *L2Load = dyn_cast<LoadInst>(X);
  if (!L1 || !L2Load || !L2Load->isSimple())
    return false;
  Loop *L2 = LI.getLoopFor(L2Load->getParent());
  BasicBlock *Latch1 = L1->getLoopLatch();
  if (!L2 || L1 == L2 || L1->contains(L2) || L2->contains(L1) || !Latch1)
    return false;
  // u = max(p, v) with p the header phi it updates, v a load of L1
  PHINode *P = nullptr;
  LoadInst *V = nullptr;
  ICmpInst *Mask = nullptr;
  bool MaskOnTrue = false;
  for (unsigned I = 0; I < 2; ++I) {
    auto *Phi = dyn_cast<PHINode>(U->getArgOperand(I));
    // (or the frozen load, the same value: a load reads no poison here)
    Value *A = U->getArgOperand(1 - I);
    if (auto *Fr = dyn_cast<FreezeInst>(A))
      A = Fr->getOperand(0);
    // (or the load where c1 is false, -inf where it is true: a masked
    // maximum, whose exp the loop with it uses only where the same
    // condition is false, L15-E455)
    ICmpInst *C = nullptr;
    bool OnTrue = false;
    if (auto *S = dyn_cast<SelectInst>(A); S && AGXExpMaskedMax) {
      auto NegInf = [](Value *K) {
        auto *CF = dyn_cast<ConstantFP>(K);
        return CF && CF->isInfinity() && CF->isNegative();
      };
      Value *Ld = nullptr;
      if (NegInf(S->getTrueValue()))
        Ld = S->getFalseValue();
      else if (NegInf(S->getFalseValue()))
        Ld = S->getTrueValue(), OnTrue = true;
      C = dyn_cast<ICmpInst>(S->getCondition());
      if (Ld && C) {
        if (auto *Fr = dyn_cast<FreezeInst>(Ld))
          Ld = Fr->getOperand(0);
        A = Ld;
      } else {
        C = nullptr;
      }
    }
    bool AExt = false;
    if (auto *E1 = dyn_cast<FPExtInst>(A); E1 && AGXExpExtendedLoads) {
      A = E1->getOperand(0);
      AExt = true;
      if (auto *Fr = dyn_cast<FreezeInst>(A))
        A = Fr->getOperand(0);
    }
    auto *Ld = AExt == XExt ? dyn_cast<LoadInst>(A) : nullptr;
    if (Phi && Ld && Phi->getParent() == L1->getHeader() &&
        Phi->getIncomingValueForBlock(Latch1) == U)
      P = Phi, V = Ld, Mask = C, MaskOnTrue = OnTrue;
  }
  if (!P || !V || !V->isSimple() || V->getType() != X->getType() ||
      !L1->contains(V))
    return false;
  // both every iteration
  DominatorTree DT(*E->getFunction());
  if (Exit && Exit->getNumIncomingValues() > 1) {
    CondBrInst *G1 = L1->getLoopGuardBranch(), *G2 = L2->getLoopGuardBranch();
    if (!G1 || !G2 || G1->getCondition() != G2->getCondition() ||
        (G1->getSuccessor(0) == L1->getLoopPreheader()) !=
            (G2->getSuccessor(0) == L2->getLoopPreheader()))
      return false;
    for (unsigned I = 0, N = Exit->getNumIncomingValues(); I < N; ++I)
      if (stripSinglePhis(Exit->getIncomingValue(I)) != U &&
          isPotentiallyReachable(L1->getHeader(), Exit->getIncomingBlock(I),
                                 nullptr, &DT))
        return false;
  }
  if (!DT.dominates(V->getParent(), Latch1) ||
      !DT.dominates(U->getParent(), Latch1))
    return false;
  // The addresses the same function of the iteration: their expressions the
  // same but for L1's recurrences in the one where the other has L2's, with
  // the same operands (sext {r, +, 1} of each loop: iteration k of either
  // reads one address). L1 runs at most once before the load: its maximum
  // is the value of u after its last iteration.
  std::function<bool(const SCEV *, const SCEV *)> Same =
      [&](const SCEV *S1, const SCEV *S2) -> bool {
    if (S1 == S2)
      return SE.isLoopInvariant(S1, L1) && SE.isLoopInvariant(S1, L2);
    if (S1->getSCEVType() != S2->getSCEVType() ||
        S1->getType() != S2->getType())
      return false;
    if (auto *R1 = dyn_cast<SCEVAddRecExpr>(S1)) {
      auto *R2 = cast<SCEVAddRecExpr>(S2);
      if (R1->getLoop() != L1 || R2->getLoop() != L2 ||
          R1->getNumOperands() != R2->getNumOperands())
        return false;
      for (unsigned I = 0, N = R1->getNumOperands(); I < N; ++I)
        if (!Same(R1->getOperand(I), R2->getOperand(I)))
          return false;
      return true;
    }
    if (auto *C1 = dyn_cast<SCEVCastExpr>(S1))
      return Same(C1->getOperand(), cast<SCEVCastExpr>(S2)->getOperand());
    if (auto *N1 = dyn_cast<SCEVNAryExpr>(S1)) {
      auto *N2 = cast<SCEVNAryExpr>(S2);
      if (N1->getNumOperands() != N2->getNumOperands())
        return false;
      for (unsigned I = 0, N = N1->getNumOperands(); I < N; ++I)
        if (!Same(N1->getOperand(I), N2->getOperand(I)))
          return false;
      return true;
    }
    return false;
  };
  if (!Same(SE.getSCEV(V->getPointerOperand()),
            SE.getSCEV(L2Load->getPointerOperand())))
    return false;
  // a masked maximum: every use of the exp a select of it where the same
  // condition of the iteration (the same compare of the same functions of
  // it) has the value that the maximum took the load with
  if (Mask) {
    if (E->use_empty())
      return false;
    for (User *Us : E->users()) {
      auto *S = dyn_cast<SelectInst>(Us);
      auto *C2 = S ? dyn_cast<ICmpInst>(S->getCondition()) : nullptr;
      if (!C2 || C2->getPredicate() != Mask->getPredicate() ||
          !L2->contains(S) || S->getCondition() == E ||
          (MaskOnTrue ? S->getTrueValue() : S->getFalseValue()) != E ||
          (MaskOnTrue ? S->getFalseValue() : S->getTrueValue()) == E ||
          !Same(SE.getSCEV(Mask->getOperand(0)), SE.getSCEV(C2->getOperand(0))) ||
          !Same(SE.getSCEV(Mask->getOperand(1)), SE.getSCEV(C2->getOperand(1))))
        return false;
    }
  }
  if (L1->getLoopDepth() != 1 || L2->getLoopDepth() != 1)
    return false;
  const SCEV *B1 = SE.getBackedgeTakenCount(L1);
  const SCEV *B2 = SE.getBackedgeTakenCount(L2);
  if (isa<SCEVCouldNotCompute>(B1) || isa<SCEVCouldNotCompute>(B2) ||
      !(B1 == B2 || SE.isKnownPredicate(ICmpInst::ICMP_ULE, B2, B1)))
    return false;
  // nothing between L1 and the load writes memory or waits for other threads
  // but, in L2 (an in-place pass: softmax's third), a store after the load
  // to its own address, which no other iteration's access overlaps
  // (L15-E387)
  const SCEV *P2 = SE.getSCEV(L2Load->getPointerOperand());
  const DataLayout &DL = E->getDataLayout();
  auto OwnStore = [&](Instruction &I) {
    auto *St = dyn_cast<StoreInst>(&I);
    return AGXExpInPlaceStores && St && St->isSimple() && L2->contains(St) &&
           SE.getSCEV(St->getPointerOperand()) == P2 &&
           DT.dominates(L2Load, St) &&
           disjointIterations(P2, L2, SE,
                              std::max(DL.getTypeStoreSize(X->getType()),
                                       DL.getTypeStoreSize(
                                           St->getValueOperand()->getType()))
                                  .getFixedValue());
  };
  BasicBlock *H1 = L1->getHeader(), *To = L2Load->getParent();
  for (BasicBlock &BB : *E->getFunction()) {
    if (!isPotentiallyReachable(H1, &BB, nullptr, &DT, &LI) ||
        !isPotentiallyReachable(&BB, To, nullptr, &DT, &LI))
      continue;
    for (Instruction &I : BB)
      // (barriers have side effects; a SIMD-group reduction does not
      // touch memory)
      if ((I.mayWriteToMemory() || I.mayHaveSideEffects() ||
           (isa<CallBase>(I) && !isa<IntrinsicInst>(I))) &&
          !OwnStore(I))
        return false;
  }
  return true;
}

/// The exps expBelowLoopMax finds, marked (!agx.exp.below.max) at the start
/// of the pass: the deferred maximum reductions rewrite the loop's maximum
/// that it matches (L15-E450).
static void markExpsBelowLoopMax(Function &F) {
  if (!AGXExpBelowLoopMax)
    return;
  SmallVector<IntrinsicInst *, 4> Exps;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && isExpOfAnyBase(II->getIntrinsicID()))
      Exps.push_back(II);
  if (Exps.empty())
    return;
  DominatorTree DT(F);
  LoopInfo LI(DT);
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII, &F);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  for (IntrinsicInst *II : Exps)
    if (expBelowLoopMax(II, LI, SE))
      II->setMetadata("agx.exp.below.max", MDNode::get(F.getContext(), {}));
}

static cl::opt<bool> AGXReciprocalAboveOne(
    "agx-reciprocal-above-one", cl::init(true), cl::Hidden,
    cl::desc("1 / (e^x + c), c >= 1, as the reciprocal step without the "
             "division's scaling and NaN select"));

static cl::opt<bool> AGXReciprocalAboveOneNonNegative(
    "agx-reciprocal-above-one-non-negative", cl::init(true), cl::Hidden,
    cl::desc("1 / (a + c) and 1 / fma(k, a, c), a never below 0, k > 0, "
             "c >= 1, as 1 / (e^x + c)"));

static cl::opt<bool> AGXQuotientAboveOne(
    "agx-quotient-above-one", cl::init(true), cl::Hidden,
    cl::desc("(c' - e) / (e + c), e an exponential of a non-positive value, "
             "as Markstein's quotient step without the division's scaling"));

static cl::opt<bool> AGXSiluQuotient(
    "agx-silu-quotient", cl::init(true), cl::Hidden,
    cl::desc("x / (e^-x + 1) as a scaled Markstein quotient step"));

static cl::opt<bool> AGXReciprocalConverted(
    "agx-reciprocal-converted", cl::init(true), cl::Hidden,
    cl::desc("1 / (a converted 32-bit integer or widened half) as the "
             "reciprocal step without the division's scaling"));

static cl::opt<bool> AGXReciprocalSqrt(
    "agx-reciprocal-sqrt", cl::init(true), cl::Hidden,
    cl::desc("1 / sqrt(v) as the reciprocal step without the division's "
             "scaling"));

/// 1 / d for d = e^x + c (exp, exp2 or exp10; c >= 1 a constant): d is at
/// least 1 or a NaN, so the division's scaling of d's exponent (whose rcp and
/// fmas scale with it while 1 / d is a normal float) and its NaN select
/// (rcp's NaN goes through the fmas) are not needed: the correctly rounded
/// reciprocal y0 + y0 (1 - d y0) (as in tanh), and +0 past 2^126 where the
/// quotient flushes (+inf included) (L15-E282).
///
/// 1 / sqrt(v) (math.rsqrt) likewise: a square root is +-0, +inf, a NaN or a
/// normal float in [2^-75, 2^64], whose reciprocal is normal; at +-0 and +inf
/// the step's residual 1 - d y0 is a NaN, and rcp's own +-inf and +0 are the
/// quotient (as for a NaN d): rcp, two fmas and a select (L15-E283).
static bool reciprocalsAboveOne(Function &F) {
  if (!AGXReciprocalAboveOne)
    return false;
  using namespace PatternMatch;
  SmallVector<std::pair<Instruction *, bool>, 4> Divs;
  for (Instruction &I : instructions(F)) {
    Value *D, *E;
    const APFloat *C;
    if (!I.getType()->isFloatTy() || !match(&I, m_FDiv(m_FPOne(), m_Value(D))) ||
        cast<FPMathOperator>(I).getFastMathFlags().any())
      continue;
    if (AGXReciprocalSqrt && match(D, m_Intrinsic<Intrinsic::sqrt>(m_Value()))) {
      Divs.push_back({&I, true});
      continue;
    }
    // A converted integer of at most 32 bits or a widened half: +-0 or a
    // float of magnitude 2^-24 to 2^32 (a half's denormals are normal
    // floats), whose reciprocal is normal, as a square root's: the same
    // sequence (L15-E295).
    Value *Src;
    if (AGXReciprocalConverted &&
        (((match(D, m_SIToFP(m_Value(Src))) ||
           match(D, m_UIToFP(m_Value(Src)))) &&
          Src->getType()->getScalarSizeInBits() <= 32) ||
         (match(D, m_FPExt(m_Value(Src))) && Src->getType()->isHalfTy()))) {
      Divs.push_back({&I, true});
      continue;
    }
    // (or of any value never below 0, |x|, and fma(k, a, c) of a such a
    // with k > 0: erfc's 1 / (|x| + 2) and 1 / (2 |x| + 1), L15-E500)
    auto NeverNegative = [&](Value *V) {
      return AGXReciprocalAboveOneNonNegative &&
             computeKnownFPClass(V, fcAllFlags,
                                 SimplifyQuery(F.getDataLayout()))
                 .cannotBeOrderedLessThanZero();
    };
    const APFloat *K;
    if (AGXReciprocalAboveOneNonNegative &&
        (match(D, m_Intrinsic<Intrinsic::fma>(m_APFloat(K), m_Value(E),
                                              m_APFloat(C))) ||
         match(D, m_Intrinsic<Intrinsic::fma>(m_Value(E), m_APFloat(K),
                                              m_APFloat(C)))) &&
        K->isFinite() && K->isNonZero() && !K->isNegative() && C->isFinite() &&
        C->compare(APFloat(1.0f)) != APFloat::cmpLessThan && NeverNegative(E)) {
      Divs.push_back({&I, false});
      continue;
    }
    if (!match(D, m_c_FAdd(m_Value(E), m_APFloat(C))) || !C->isFinite() ||
        C->compare(APFloat(1.0f)) == APFloat::cmpLessThan)
      continue;
    auto *II = dyn_cast<IntrinsicInst>(E);
    if ((!II || (II->getIntrinsicID() != Intrinsic::exp &&
                 II->getIntrinsicID() != Intrinsic::exp2 &&
                 II->getIntrinsicID() != Intrinsic::exp10)) &&
        !NeverNegative(E))
      continue;
    Divs.push_back({&I, false});
  }
  // n / d, d = e + c and n = c' - e or e + c' (c, c' in [1, 2^20]), e an
  // exponential of a value that is not positive: d in [1, 2^21], n zero or
  // at least 2^-24 in magnitude (1 - e is exact near e = 1), so with y the
  // correctly rounded 1 / d (E282's step) Markstein's q0 = n y, r = n - d q0
  // (exact), q = q0 + r y is the correctly rounded quotient, no value near a
  // denormal: rcp, five fmas (L15-E409; tanh's (1 - e) / (1 + e)).
  SmallVector<Instruction *, 4> Quots;
  if (AGXQuotientAboveOne) {
    auto Bounded = [](const APFloat *C) {
      return C->isFinite() && C->compare(APFloat(1.0f)) != APFloat::cmpLessThan &&
             C->compare(APFloat(0x1p20f)) != APFloat::cmpGreaterThan;
    };
    auto NonPositiveExp = [&](Value *E) {
      auto *II = dyn_cast<IntrinsicInst>(E);
      if (!II || (II->getIntrinsicID() != Intrinsic::exp &&
                  II->getIntrinsicID() != Intrinsic::exp2 &&
                  II->getIntrinsicID() != Intrinsic::exp10 &&
                  II->getIntrinsicID() != Intrinsic::agx_exp2))
        return false;
      KnownFPClass K = computeKnownFPClass(
          II->getArgOperand(0), fcAllFlags, SimplifyQuery(F.getDataLayout()));
      return K.cannotBeOrderedGreaterThanZero();
    };
    for (Instruction &I : instructions(F)) {
      Value *Nm, *D, *E, *E2;
      const APFloat *C, *C2;
      if (!I.getType()->isFloatTy() ||
          !match(&I, m_FDiv(m_Value(Nm), m_Value(D))) ||
          cast<FPMathOperator>(I).getFastMathFlags().any() ||
          !match(D, m_c_FAdd(m_Value(E), m_APFloat(C))) || !Bounded(C) ||
          !NonPositiveExp(E))
        continue;
      if (!((match(Nm, m_FSub(m_APFloat(C2), m_Value(E2))) ||
             match(Nm, m_c_FAdd(m_Value(E2), m_APFloat(C2)))) &&
            E2 == E && Bounded(C2)))
        continue;
      Quots.push_back(&I);
    }
  }
  // x / (e^-x + 1) (silu): d in [1, +inf] or a NaN, x any float. Scaled
  // by 2^-64 both where d > 2^64 (x < -44: x stays exact), E282's correctly
  // rounded 1 / d and Markstein's q0 = x y, r = x - d q0, q = q0 + r y;
  // at d = +inf and x = +inf, q0.
  // The quotient of one value: checked for every float x (L15-E483).
  SmallVector<Instruction *, 4> Silus;
  if (AGXSiluQuotient)
    for (Instruction &I : instructions(F)) {
      Value *X, *D, *E, *A;
      if (!I.getType()->isFloatTy() ||
          !match(&I, m_FDiv(m_Value(X), m_Value(D))) ||
          cast<FPMathOperator>(I).getFastMathFlags().any() ||
          !match(D, m_c_FAdd(m_Value(E), m_FPOne())) ||
          !match(E, m_Intrinsic<Intrinsic::exp>(m_Value(A))) ||
          cast<FPMathOperator>(E)->getFastMathFlags().any() ||
          !match(A, m_FNeg(m_Specific(X))))
        continue;
      Silus.push_back(&I);
    }
  for (Instruction *I : Silus) {
    IRBuilder<> B(I);
    Value *X = I->getOperand(0), *D = I->getOperand(1);
    Type *Ty = D->getType();
    Value *Big = B.CreateFCmpOGT(D, ConstantFP::get(Ty, 0x1p64));
    Value *Sc = B.CreateSelect(Big, ConstantFP::get(Ty, 0x1p-64),
                               ConstantFP::get(Ty, 1.0));
    Value *DS = B.CreateFMul(D, Sc);
    Value *XS = B.CreateFMul(X, Sc);
    Value *NDS = B.CreateFNeg(DS);
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {DS});
    Value *E1 = B.CreateIntrinsic(Intrinsic::fma, {Ty},
                                  {NDS, Y0, ConstantFP::get(Ty, 1.0)});
    Value *Y = B.CreateIntrinsic(Intrinsic::fma, {Ty}, {E1, Y0, Y0});
    // (y as 0 at d = +inf, where q0 = x * 0 is then the quotient, -0 or a
    // NaN for x = -inf; and at x = +inf, q0 = +inf is: where the residual
    // is a NaN, q0)
    Y = B.CreateSelect(B.CreateFCmpOEQ(D, ConstantFP::getInfinity(Ty)),
                       ConstantFP::get(Ty, 0.0), Y);
    Value *Q0 = B.CreateFMul(XS, Y);
    // (the residual negated, d q0 - x, and q0 - r' y: the same where x is
    // not zero, and -0 for x = -0, where x - d q0 would be +0)
    Value *R = B.CreateIntrinsic(Intrinsic::fma, {Ty},
                                 {DS, Q0, B.CreateFNeg(XS)});
    Value *Q = B.CreateIntrinsic(Intrinsic::fma, {Ty},
                                 {B.CreateFNeg(R), Y, Q0});
    Q = B.CreateSelect(B.CreateFCmpUNO(R, R), Q0, Q);
    Q->takeName(I);
    I->replaceAllUsesWith(Q);
    I->eraseFromParent();
  }
  for (Instruction *I : Quots) {
    IRBuilder<> B(I);
    Value *N = I->getOperand(0), *D = I->getOperand(1);
    Type *Ty = D->getType();
    Value *ND = B.CreateFNeg(D);
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {D});
    Value *E = B.CreateIntrinsic(Intrinsic::fma, {Ty},
                                 {ND, Y0, ConstantFP::get(Ty, 1.0)});
    Value *Y = B.CreateIntrinsic(Intrinsic::fma, {Ty}, {E, Y0, Y0});
    Value *Q0 = B.CreateFMul(N, Y);
    Value *R = B.CreateIntrinsic(Intrinsic::fma, {Ty}, {ND, Q0, N});
    Value *Q = B.CreateIntrinsic(Intrinsic::fma, {Ty}, {R, Y, Q0});
    Q->takeName(I);
    I->replaceAllUsesWith(Q);
    I->eraseFromParent();
  }
  for (auto [I, Sqrt] : Divs) {
    IRBuilder<> B(I);
    Value *D = I->getOperand(1);
    Type *Ty = D->getType();
    Value *Y0 = B.CreateIntrinsic(Intrinsic::agx_rcp, {Ty}, {D});
    Value *E = B.CreateIntrinsic(Intrinsic::fma, {Ty},
                                 {B.CreateFNeg(D), Y0, ConstantFP::get(Ty, 1.0)});
    Value *Y = B.CreateIntrinsic(Intrinsic::fma, {Ty}, {E, Y0, Y0});
    Value *R = Sqrt ? B.CreateSelect(B.CreateFCmpUNO(E, E), Y0, Y)
                    : B.CreateSelect(
                          B.CreateFCmpOGT(D, ConstantFP::get(Ty, 0x1p126)),
                          ConstantFP::get(Ty, 0.0), Y);
    R->takeName(I);
    I->replaceAllUsesWith(R);
    I->eraseFromParent();
  }
  return !Divs.empty() || !Quots.empty() || !Silus.empty();
}

/// Whether \p V reaches only f32 arithmetic that flushes a denormal operand
/// (fadd, fsub, fmul, fma; through selects' values, phis, negations and
/// vector moves): a denormal there is the zero of its sign.
static bool onlyFlushingUses(Value *V) {
  SmallVector<Value *, 16> Work{V};
  SmallPtrSet<Value *, 16> Seen{V};
  while (!Work.empty()) {
    Value *X = Work.pop_back_val();
    for (Use &U : X->uses()) {
      auto *I = cast<Instruction>(U.getUser());
      if (!I->getType()->getScalarType()->isFloatTy() &&
          !isa<ExtractElementInst>(I))
        return false;
      // (not with a constant that makes it the identity a fold would turn
      // into the operand itself, unflushed: x + -0, x * 1, ...)
      auto Degenerate = [&](Instruction *J) {
        return any_of(J->operands(), [](Value *O) {
          const APFloat *C;
          return match(O, PatternMatch::m_APFloat(C)) &&
                 (C->isZero() || C->isExactlyValue(1.0) ||
                  C->isExactlyValue(-1.0));
        });
      };
      switch (I->getOpcode()) {
      case Instruction::FAdd:
      case Instruction::FSub:
      case Instruction::FMul:
        if (Degenerate(I))
          return false;
        continue;
      case Instruction::Select:
        if (U.getOperandNo() == 0)
          return false;
        break;
      case Instruction::PHI:
      case Instruction::FNeg:
      case Instruction::InsertElement:
      case Instruction::ShuffleVector:
      case Instruction::ExtractElement:
        if (isa<InsertElementInst>(I) && U.getOperandNo() == 2)
          return false;
        break;
      default:
        if (auto *II = dyn_cast<IntrinsicInst>(I);
            II && (II->getIntrinsicID() == Intrinsic::fma ||
                   II->getIntrinsicID() == Intrinsic::fmuladd) &&
            !Degenerate(II))
          continue;
        return false;
      }
      if (Seen.size() > 64)
        return false;
      if (Seen.insert(I).second)
        Work.push_back(I);
    }
  }
  return true;
}

static cl::opt<bool> AGXRareHardwareBranch(
    "agx-rare-hardware-branch", cl::init(true), cl::Hidden,
    cl::desc("the expansions' hardware fallbacks (sin and cos past 2^22) in "
             "a block run only where a thread needs them"));

static cl::opt<unsigned> AGXRareFallbacksMin(
    "agx-rare-fallbacks-min", cl::init(2), cl::Hidden,
    cl::desc("fewest fallbacks of one condition in a block branched around"));

static cl::opt<bool> AGXRareFallbacksHoist(
    "agx-rare-fallbacks-hoist", cl::init(true), cl::Hidden,
    cl::desc("what follows a branched-around fallback's selects and needs "
             "neither them nor a write before it, ahead of the branch"));

/// The expansions' selects sel(c, a, h) whose h is a fallback for rare
/// arguments (a call marked agx.rare: sin and cos's hardware reduction past
/// 2^22) take h from a block branched to only where c is false: a SIMD-group
/// whose threads all have c skips it (L15-E504). The selects of one block and
/// condition share the block (sincos's two). After the SimplifyCFG before
/// structurization, which would speculate h again (AGXRareFallbacks).
static bool branchRareFallbacks(Function &F) {
  MapVector<std::pair<BasicBlock *, Value *>, SmallVector<SelectInst *, 2>>
      Groups;
  for (Instruction &I : instructions(F)) {
    auto *S = dyn_cast<SelectInst>(&I);
    // (unused: the half of a sin or cos expansion not asked for, until dead
    // code elimination)
    if (!S || S->use_empty() || !S->getCondition()->getType()->isIntegerTy(1))
      continue;
    auto *H = dyn_cast<CallInst>(S->getFalseValue());
    if (!H || !H->getMetadata("agx.rare") || !H->hasOneUse() ||
        H->getParent() != S->getParent())
      continue;
    Groups[{S->getParent(), S->getCondition()}].push_back(S);
  }
  bool Changed = false;
  for (auto &[Key, Sels] : Groups) {
    // (one hardware sin or cos is 8 instructions, which AGXLowerControlFlow
    // runs with an empty mask rather than jump over: the if would only add
    // its own two)
    if (Sels.size() < AGXRareFallbacksMin)
      continue;
    auto [BB, C] = Key;
    SelectInst *First = Sels.front();
    auto Before = [&](Value *V) {
      auto *I = dyn_cast<Instruction>(V);
      return !I || I->getParent() != BB || I->comesBefore(First);
    };
    if (!Before(C) || !all_of(Sels, [&](SelectInst *S) {
          // (h moves to the new block: its operands must be there before)
          return Before(S->getTrueValue()) &&
                 all_of(cast<Instruction>(S->getFalseValue())->operands(),
                        Before);
        }))
      continue;
    // The rest of the block that needs neither the selects nor a write
    // before it goes ahead of them, out of the join: a load of the tail
    // starts before the branch, as the scheduler had it (L15-E505).
    if (AGXRareFallbacksHoist) {
      SmallPtrSet<Instruction *, 16> Stay(Sels.begin(), Sels.end());
      bool Wrote = false;
      SmallVector<Instruction *, 16> Hoist;
      for (Instruction &I : make_range(std::next(First->getIterator()),
                                       BB->getTerminator()->getIterator())) {
        bool Needs = any_of(I.operands(), [&](Value *V) {
          auto *OI = dyn_cast<Instruction>(V);
          return OI && Stay.count(OI);
        });
        bool Effects = I.mayWriteToMemory() || I.mayHaveSideEffects() ||
                       (isa<CallBase>(I) && cast<CallBase>(I).isConvergent());
        Wrote |= Effects;
        if (Needs || Effects || (Wrote && I.mayReadFromMemory()) ||
            any_of(Sels, [&](SelectInst *S) {
              return S->getFalseValue() == &I;
            }))
          Stay.insert(&I);
        else
          Hoist.push_back(&I);
      }
      for (Instruction *I : Hoist)
        I->moveBefore(First->getIterator());
    }
    BasicBlock *Tail = SplitBlock(BB, First);
    BasicBlock *Rare =
        BasicBlock::Create(F.getContext(), BB->getName() + ".rare", &F, Tail);
    IRBuilder<>(Rare).CreateBr(Tail);
    BB->getTerminator()->eraseFromParent();
    IRBuilder<>(BB).CreateCondBr(C, Tail, Rare);
    for (SelectInst *S : Sels) {
      auto *H = cast<Instruction>(S->getFalseValue());
      H->moveBefore(Rare->getTerminator()->getIterator());
      PHINode *P = PHINode::Create(S->getType(), 2, S->getName(),
                                   Tail->begin());
      P->addIncoming(S->getTrueValue(), BB);
      P->addIncoming(H, Rare);
      S->replaceAllUsesWith(P);
      S->eraseFromParent();
    }
    Changed = true;
  }
  return Changed;
}

bool AGXCodeGenPrepare::expandMath(Function &F, const AGXSubtarget &ST) {
  bool Changed = expandRoundToInt(F);
  Changed |= expandPowI(F);
  Changed |= expandFRem(F, ST);
  Changed |= convertIntsToBFloatOnce(F);
  // (before exp is expanded, and with flushing f32: the quotient past 2^126)
  if (!ST.hasIEEEFP32())
    Changed |= reciprocalsAboveOne(F);
  // (before the expansions below: the analysis needs the loops and loads
  // as they are)
  SmallPtrSet<IntrinsicInst *, 4> ExpNoOverflow;
  if (AGXExpBelowLoopMax) {
    SmallVector<IntrinsicInst *, 4> Exps;
    for (Instruction &I : instructions(F))
      if (auto *II = dyn_cast<IntrinsicInst>(&I);
          II && isExpOfAnyBase(II->getIntrinsicID()))
        Exps.push_back(II);
    if (!Exps.empty()) {
      DominatorTree DT(F);
      LoopInfo LI(DT);
      TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
      TargetLibraryInfo TLI(TLII, &F);
      AssumptionCache AC(F);
      ScalarEvolution SE(F, TLI, AC, DT, LI);
      for (IntrinsicInst *II : Exps)
        if (II->getMetadata("agx.exp.below.max") ||
            expBelowLoopMax(II, LI, SE))
          ExpNoOverflow.insert(II);
    }
  }
  // An expansion may use functions that are expanded in turn (tan's sin).
  for (;;) {
    SmallVector<IntrinsicInst *, 8> Calls;
    for (Instruction &I : instructions(F)) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      // With ieee-fp32, AGXStrictFP computes the precise functions exactly.
      bool Approx = (isa<FPMathOperator>(II) && II->hasApproxFunc()) ||
                    ST.hasIEEEFP32();
      // The type of the operand: sincos returns a pair.
      if (II->arg_size() &&
          AGX::canExpandMath(II->getIntrinsicID(),
                             II->getArgOperand(0)->getType(), Approx))
        Calls.push_back(II);
    }
    if (Calls.empty())
      return Changed;
    for (IntrinsicInst *II : Calls) {
      IRBuilder<> B(II);
      B.setFastMathFlags(II->getFastMathFlags());
      SmallVector<Value *, 2> Args(II->args());
      Value *R = AGX::expandMath(B, II->getIntrinsicID(), Args,
                                 !ST.hasIEEEFP32(), ExpNoOverflow.count(II),
                                 !ST.hasCanonicalNaN(),
                                 !ST.hasIEEEFP32() &&
                                     II->getIntrinsicID() == Intrinsic::exp &&
                                     onlyFlushingUses(II));
      R->takeName(II);
      II->replaceAllUsesWith(R);
      ExpNoOverflow.erase(II);
      II->eraseFromParent();
    }
    Changed = true;
  }
}

/// With "agx-max-threads-per-threadgroup" at most 128, the threadgroup is one
/// aligned block of 4 SIMD-groups (or less: a block cut short by the end of
/// the threadgroup does not wait for the missing groups). Its block_barrier
/// waits for the same threads as threadgroup_barrier, about 3x faster on G14,
/// and orders their threadgroup and device memory, so the workgroup-scope
/// fences right around the barrier go too.
bool AGXCodeGenPrepare::useBlockBarriers(Function &F, const AGXSubtarget &ST) {
  unsigned Threads =
      F.getFnAttributeAsParsedInteger("agx-max-threads-per-threadgroup", 0);
  if (!ST.hasBlockBarrier() || Threads == 0 || Threads > 128)
    return false;
  SmallVector<IntrinsicInst *, 4> Barriers;
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->getIntrinsicID() == Intrinsic::agx_threadgroup_barrier)
      Barriers.push_back(II);
  SyncScope::ID Workgroup = F.getContext().getOrInsertSyncScopeID("workgroup");
  auto IsWorkgroupFence = [&](Instruction *I) {
    auto *Fence = dyn_cast_or_null<FenceInst>(I);
    return Fence && Fence->getSyncScopeID() == Workgroup;
  };
  for (IntrinsicInst *II : Barriers) {
    for (Instruction *Fence : {II->getPrevNode(), II->getNextNode()})
      if (IsWorkgroupFence(Fence))
        Fence->eraseFromParent();
    IRBuilder<> B(II);
    B.CreateIntrinsic(Intrinsic::agx_block_barrier, {});
    II->eraseFromParent();
  }
  return !Barriers.empty();
}

static bool isSystemScope(SyncScope::ID Scope, LLVMContext &Ctx) {
  return Scope == SyncScope::System ||
         Scope == Ctx.getOrInsertSyncScopeID("one-as");
}

bool AGX::isHostCoherentAccess(const Instruction &I) {
  unsigned AS;
  if (const auto *L = dyn_cast<LoadInst>(&I)) {
    if (!L->isVolatile() &&
        !(L->isAtomic() && isSystemScope(L->getSyncScopeID(), I.getContext())))
      return false;
    AS = L->getPointerAddressSpace();
  } else if (const auto *S = dyn_cast<StoreInst>(&I)) {
    if (!S->isVolatile() &&
        !(S->isAtomic() && isSystemScope(S->getSyncScopeID(), I.getContext())))
      return false;
    AS = S->getPointerAddressSpace();
  } else {
    return false;
  }
  return AS == AGX::DEVICE_ADDRESS || AS == AGX::GENERIC_ADDRESS;
}

bool AGX::isHostCoherentAccess(const MachineInstr &MI) {
  LLVMContext &Ctx = MI.getMF()->getFunction().getContext();
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (MMO->isVolatile() ||
        (MMO->isAtomic() && isSystemScope(MMO->getSyncScopeID(), Ctx)))
      return true;
  return false;
}

/// Host memory written during a dispatch is seen only by a coherent load
/// after an f59d, and the host sees a store during the dispatch only when it
/// is coherent and followed by an f533 (ptx-re-blockers B7, Apple-M2-Playground
/// 2026-10-16-cpu-gpu-visibility). The coherency bit is set when the access is
/// encoded (AGXMCInstLower); the barriers go here.
bool AGXCodeGenPrepare::fenceHostCoherentAccesses(Function &F) {
  SmallVector<Instruction *, 8> Accesses;
  for (Instruction &I : instructions(F))
    if (AGX::isHostCoherentAccess(I))
      Accesses.push_back(&I);
  if (Accesses.empty())
    return false;
  Module &M = *F.getParent();
  Function *Barrier = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::agx_memory_barrier);
  for (Instruction *I : Accesses) {
    IRBuilder<> B(I);
    if (isa<StoreInst>(I))
      B.SetInsertPoint(I->getNextNode());
    // f59d is memory_barrier 3, 1, 9; f533 is 0, 3, 3.
    unsigned A = isa<LoadInst>(I) ? 3 : 0, Bv = isa<LoadInst>(I) ? 1 : 3,
             C = isa<LoadInst>(I) ? 9 : 3;
    B.CreateCall(Barrier, {B.getInt32(A), B.getInt32(Bv), B.getInt32(C)});
  }
  return true;
}

namespace {
/// branchRareFallbacks as a pass of its own, after the SimplifyCFG before
/// structurization.
class AGXRareFallbacks : public FunctionPass {
public:
  static char ID;
  AGXRareFallbacks() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "AGX Rare Fallbacks"; }
  bool runOnFunction(Function &F) override {
    return AGXRareHardwareBranch && !skipFunction(F) &&
           branchRareFallbacks(F);
  }
};
} // namespace

char AGXRareFallbacks::ID = 0;

INITIALIZE_PASS(AGXRareFallbacks, "agx-rare-fallbacks", "AGX Rare Fallbacks",
                false, false)

FunctionPass *llvm::createAGXRareFallbacksPass() {
  return new AGXRareFallbacks();
}
