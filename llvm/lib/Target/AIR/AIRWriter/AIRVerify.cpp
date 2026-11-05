//===- AIRVerify.cpp - Conformance checks for AIR modules -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Each rule is what the device does with such a module: a check here that the
// device does not make would reject working code, so a new rule comes with a
// module in llvm/utils/air-gpu/verify-cases that shows it, which
// air_verify_device_check.py runs on the device. (AIR's documents exclude more than
// the device rejects: i3, address space casts and kernels that return a value run
// on an M2 Max, so they are not errors here.) The checks are linear in the size of
// the module.
//
//===----------------------------------------------------------------------===//

#include "AIRVerify.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Operator.h"
#include <functional>
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::metal;

/// Whether the first component of an air.* name is a function of AIR. Apple publishes
/// no list: this is what the open-source AIR producers and llvm/utils/air-gpu use
/// (llvm/docs/AIR/AIRFunctionality.rst). It finds names that do not exist, for example
/// air.copysign, which Metal reports as an undefined symbol; it does not check types.
static bool isKnownAIRFunction(StringRef Name) {
  StringRef Rest = Name.drop_front(strlen("air."));
  StringRef Fn = Rest.split('.').first;
  // air.atomic.fence and air.atomic.<space>.<operation>...
  if (Fn == "atomic") {
    SmallVector<StringRef, 4> Parts;
    Rest.split(Parts, '.');
    if (Parts.size() >= 2 && Parts[1] == "fence")
      return true;
    static const StringSet<> Ops = {"add", "and", "cmpxchg", "exchange", "load", "max", "min",
                                    "or",  "store", "sub", "xchg", "xor"};
    return Parts.size() >= 3 && (Parts[1] == "global" || Parts[1] == "local") &&
           Ops.contains(Parts[2]);
  }
  if (Fn.starts_with("simd_")) {
    static const StringSet<> Simd = {
        "simd_active_threads_mask", "simd_and", "simd_ballot", "simd_broadcast",
        "simd_broadcast_first", "simd_is_first", "simd_matrix_fmadd", "simd_max", "simd_min", "simd_or",
        "simd_prefix_exclusive_product", "simd_prefix_exclusive_sum", "simd_prefix_inclusive_product",
        "simd_prefix_inclusive_sum", "simd_product", "simd_reduce", "simd_shuffle",
        "simd_shuffle_and_fill_down", "simd_shuffle_and_fill_up", "simd_shuffle_down", "simd_shuffle_up",
        "simd_shuffle_xor", "simd_sum", "simd_vote_all", "simd_vote_any", "simd_xor"};
    return Simd.contains(Fn);
  }
  // Families whose names are built from the resource they act on.
  for (StringRef Prefix :
       {"atomic_", "calculate_", "convert", "fence", "gather_", "get_", "interpolate_",
        "mesh", "quad_", "read_", "sample_", "set_", "simdgroup", "thread",
        "wait_simdgroup_events", "wg", "write_"})
    if (Fn.starts_with(Prefix))
      return true;
  if (Fn.ends_with("_private_tensor"))
    return true;
  Fn.consume_front("fast_");
  static const StringSet<> Functions = {
      "abs",   "acos",  "acosh", "asin",  "asinh", "atan",  "atan2",  "atanh", "ceil",
      "clamp", "clz",   "cos",   "cosh",  "cospi", "ctz",   "dfdx",   "dfdy",
      "discard_fragment", "dot", "exp",   "exp10", "exp2",  "fabs",   "floor", "fma",
      "fmax",  "fmin",  "fmod",  "fract", "frexp", "fwidth", "is_function_constant_defined",
      "ldexp", "load",  "log",   "log10", "log2",  "max",   "min",    "mul_hi",
      "nextafter", "os_log", "pack", "popcount", "pow", "powr", "reverse_bits", "rint",
      "round", "rsqrt", "sin",   "sincos", "sinh", "sinpi", "sqrt",   "store", "tan",
      "tanh",  "tanpi", "trunc", "unpack"};
  return Functions.contains(Fn);
}

/// Whether Role is an argument role of AIR: what the open-source producers use, as for
/// the function names above. Metal's compiler fails internally on another.
static bool isKnownAIRArgumentRole(StringRef Role) {
  static const StringSet<> Roles = {
      "air.barycentric_coord", "air.base_instance", "air.base_vertex", "air.buffer",
      "air.clip_distance", "air.depth", "air.dispatch_threads_per_threadgroup", "air.fragment_input",
      "air.front_facing", "air.grid_origin", "air.grid_size", "air.imageblock_data",
      "air.indirect_argument", "air.indirect_buffer", "air.instance_id", "air.mesh", "air.mesh_grid_properties",
      "air.object_data", "air.patch_id", "air.payload", "air.point_coord", "air.point_size", "air.position",
      "air.position_in_patch", "air.primitive_id", "air.quad_grid_properties", "air.render_target",
      "air.render_target_array_index", "air.sample_id", "air.sample_mask", "air.sample_mask_in",
      "air.sampler", "air.simdgroup_index_in_threadgroup", "air.simdgroups_per_threadgroup",
      "air.stage_in", "air.stencil", "air.texture", "air.thread_index_in_quadgroup",
      "air.thread_index_in_simdgroup", "air.thread_index_in_threadgroup", "air.thread_position_in_grid",
      "air.thread_position_in_threadgroup", "air.threadgroup_position_in_grid", "air.threadgroups_per_grid",
      "air.threads_per_grid", "air.threads_per_simdgroup", "air.threads_per_threadgroup",
      "air.vertex_id", "air.vertex_input", "air.vertex_output", "air.viewport_array_index"};
  return Roles.contains(Role);
}

/// A role of an argument of a graphics stage, not of a kernel's.
static bool isGraphicsArgumentRole(StringRef Role) {
  static const StringSet<> Roles = {
      "air.barycentric_coord", "air.base_instance", "air.base_vertex", "air.clip_distance", "air.depth",
      "air.fragment_input", "air.front_facing", "air.instance_id", "air.patch_id", "air.point_coord",
      "air.point_size", "air.position", "air.position_in_patch", "air.primitive_id", "air.render_target",
      "air.render_target_array_index", "air.sample_id", "air.sample_mask", "air.sample_mask_in",
      "air.stage_in", "air.stencil", "air.vertex_id", "air.vertex_input", "air.vertex_output",
      "air.viewport_array_index"};
  return Roles.contains(Role);
}

/// An LLVM intrinsic that Metal's compiler fails on or reports undefined. Only those seen to fail: an
/// allowlist rejected valid modules (llvm.umul.with.overflow.i32).
static bool isLLVMIntrinsicMetalRejects(StringRef Name) {
  for (StringRef Prefix : {"llvm.readcyclecounter", "llvm.stacksave", "llvm.stackrestore", "llvm.ubsantrap"})
    if (Name.starts_with(Prefix))
      return true;
  return false;
}

/// A type no AIR has, at any depth of a vector (Metal's reader: "Invalid bitcode file").
static bool isNonAIRType(Type *T) {
  if (T->isScalableTy())
    return true;
  T = T->getScalarType();
  return T->isFP128Ty() || T->isX86_FP80Ty() || T->isPPC_FP128Ty() || T->isTokenTy() ||
         T->isX86_AMXTy();
}

/// Whether the type of air.<fn>.<suffix>, for a floating point function of one
/// format (not sincos, frexp or ldexp), is what the suffix says: each parameter and
/// the result are of that format. A name that is not such a function has no rule.
static bool floatMathTypeMatches(StringRef Name, const FunctionType *FT) {
  SmallVector<StringRef, 3> Parts;
  Name.split(Parts, '.');
  if (Parts.size() != 3)
    return true;
  StringRef Fn = Parts[1];
  Fn.consume_front("fast_");
  static const StringSet<> Math = {
      "acos",  "acosh", "asin",  "asinh", "atan", "atan2", "atanh", "ceil",  "cos",
      "cosh",  "cospi", "exp",   "exp10", "exp2", "fabs",  "floor", "fma",   "fmax",
      "fmin",  "fmod",  "log",   "log10", "log2", "pow",   "powr",  "rint",  "round",
      "rsqrt", "sin",   "sinh",  "sinpi", "sqrt", "tan",   "tanh",  "tanpi", "trunc"};
  if (!Math.contains(Fn))
    return true;
  StringRef Suffix = Parts[2];
  unsigned Lanes = 0;
  if (Suffix.consume_front("v"))
    if (Suffix.consumeInteger(10, Lanes))
      return true;
  auto Matches = [&](Type *T) {
    if (Lanes) {
      auto *VT = dyn_cast<FixedVectorType>(T);
      if (!VT || VT->getNumElements() != Lanes)
        return false;
      T = VT->getElementType();
    }
    if (Suffix == "f32")
      return T->isFloatTy();
    if (Suffix == "f16")
      return T->isHalfTy();
    if (Suffix == "bf16")
      return T->isBFloatTy();
    return true;
  };
  return Matches(FT->getReturnType()) && all_of(FT->params(), Matches);
}

namespace {
class Checker {
  const Module &M;
  AIRForm Form;
  SmallVector<std::string, 4> Out;

  void add(const Twine &Msg) { Out.push_back(Msg.str()); }
  void add(const Function &F, const Twine &Msg) {
    add("in function '" + F.getName() + "': " + Msg);
  }
  bool isMetal() const { return Form == AIRForm::Metal; }

  void checkModule() {
    if (!M.getTargetTriple().isAIR())
      add("the module's triple is not AIR's: " + M.getTargetTriple().str());
    if (isMetal()) {
      for (StringRef Name : {"air.version", "air.language_version"})
        if (!M.getNamedMetadata(Name))
          add("the module has no !" + Name);
      // (which minor version an OS takes is not known here: only that there is no other major one)
      if (NamedMDNode *V = M.getNamedMetadata("air.version"); V && V->getNumOperands()) {
        const MDNode *N = V->getOperand(0);
        auto *Major = N->getNumOperands()
                          ? mdconst::dyn_extract_or_null<ConstantInt>(N->getOperand(0))
                          : nullptr;
        auto *Minor = N->getNumOperands() == 3
                          ? mdconst::dyn_extract_or_null<ConstantInt>(N->getOperand(1))
                          : nullptr;
        auto *Patch = N->getNumOperands() == 3
                          ? mdconst::dyn_extract_or_null<ConstantInt>(N->getOperand(2))
                          : nullptr;
        if (!Major || !Minor || !Patch || !Major->equalsInt(2) || Minor->uge(10) || !Patch->isZero())
          add("!air.version is not an AIR version: 2.0.0 to 2.9.0");
      }
      // (<name>, major, minor, patch; 4 is the newest Metal this backend knows)
      if (NamedMDNode *L = M.getNamedMetadata("air.language_version"); L && L->getNumOperands()) {
        const MDNode *N = L->getOperand(0);
        auto *Major = N->getNumOperands() >= 2
                          ? mdconst::dyn_extract_or_null<ConstantInt>(N->getOperand(1))
                          : nullptr;
        auto *Minor = N->getNumOperands() >= 3
                          ? mdconst::dyn_extract_or_null<ConstantInt>(N->getOperand(2))
                          : nullptr;
        if (!Major || !Minor || Major->uge(5) || Minor->uge(10))
          add("!air.language_version is not a Metal version this backend knows");
      }
      // (an empty entry crashes Metal's linker)
      if (NamedMDNode *O = M.getNamedMetadata("air.compile_options"))
        for (const MDNode *N : O->operands())
          if (N->getNumOperands() != 1 || !isa_and_nonnull<MDString>(N->getOperand(0)))
            add("an !air.compile_options entry does not name an option");
    }
  }

  void checkKernels() {
    NamedMDNode *Kernels = M.getNamedMetadata("air.kernel");
    if (!Kernels)
      return;
    DenseSet<const Function *> Seen;
    for (const MDNode *K : Kernels->operands()) {
      const Function *F =
          K->getNumOperands()
              ? mdconst::dyn_extract_or_null<Function>(K->getOperand(0))
              : nullptr;
      if (!F) {
        add("an !air.kernel entry does not name a function");
        continue;
      }
      if (!Seen.insert(F).second)
        add(*F, "more than one !air.kernel entry");
      if (F->isDeclaration())
        add(*F, "a kernel without a body");
      if (K->getNumOperands() < 3) {
        add(*F, "the !air.kernel entry does not describe the kernel's "
                "arguments");
        continue;
      }
      auto *Args = dyn_cast_or_null<MDNode>(K->getOperand(K->getNumOperands() - 1));
      if (!Args) {
        add(*F, "the !air.kernel entry has no argument list");
        continue;
      }
      SmallVector<bool, 16> Described(F->arg_size(), false);
      // (threadgroup buffers have location indices of their own)
      DenseSet<uint64_t> Locations, ThreadgroupLocations;
      for (const MDOperand &A : Args->operands()) {
        auto *Node = dyn_cast_or_null<MDNode>(A);
        auto *Idx = Node && Node->getNumOperands() >= 2
                        ? mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(0))
                        : nullptr;
        auto *Role = Idx ? dyn_cast_or_null<MDString>(Node->getOperand(1)) : nullptr;
        if (!Role) {
          add(*F, "a malformed !air.kernel argument");
          continue;
        }
        uint64_t I = Idx->getZExtValue();
        if (I >= F->arg_size()) {
          add(*F, "!air.kernel describes argument " + Twine(I) +
                      ", which the kernel does not have");
          continue;
        }
        if (Described[I])
          add(*F, "!air.kernel describes argument " + Twine(I) + " twice");
        Described[I] = true;
        if (!Role->getString().starts_with("air."))
          add(*F, "argument " + Twine(I) + " has the role '" + Role->getString() +
                      "', which is not AIR's");
        else if (!isKnownAIRArgumentRole(Role->getString()))
          add(*F, "argument " + Twine(I) + " has the role '" + Role->getString() +
                      "', which AIR does not have");
        if (isGraphicsArgumentRole(Role->getString()))
          add(*F, "argument " + Twine(I) + " has the role '" + Role->getString() +
                      "', of a graphics stage, in a kernel");
        if ((Role->getString() == "air.texture" || Role->getString() == "air.sampler") &&
            F->getArg(I)->getType()->isPointerTy() &&
            F->getArg(I)->getType()->getPointerAddressSpace() == 1)
          add(*F, "argument " + Twine(I) + " has the role '" + Role->getString() +
                      "' on a device pointer");
        for (unsigned J = 2; J + 1 < Node->getNumOperands(); ++J) {
          auto *Key = dyn_cast_or_null<MDString>(Node->getOperand(J));
          if (!Key)
            continue;
          const Metadata *V = Node->getOperand(J + 1);
          if (Key->getString() == "air.location_index" && !mdconst::dyn_extract_or_null<ConstantInt>(V))
            add(*F, "the location of argument " + Twine(I) + " is not an integer");
          if ((Key->getString() == "air.arg_name" || Key->getString() == "air.arg_type_name") &&
              !isa_and_nonnull<MDString>(V))
            add(*F, "a name of argument " + Twine(I) + " is not a string");
        }
        if (Role->getString() != "air.buffer")
          continue;
        if (!F->getArg(I)->getType()->isPointerTy())
          add(*F, "the air.buffer argument " + Twine(I) + " is not a pointer");
        for (unsigned J = 2; J + 1 < Node->getNumOperands(); ++J) {
          auto *Key = dyn_cast_or_null<MDString>(Node->getOperand(J));
          if (!Key)
            continue;
          auto *Value =
              mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(J + 1));
          if (!Value)
            continue;
          bool Threadgroup = F->getArg(I)->getType()->isPointerTy() &&
                             F->getArg(I)->getType()->getPointerAddressSpace() == 3;
          // (the 31 of !air.max_device_buffers and !air.max_threadgroup_buffers)
          if (Key->getString() == "air.location_index" && Value->getValue().uge(31))
            add(*F, "a buffer argument at location " + Twine(Value->getSExtValue()) +
                        ", outside 0..30");
          // (the location is followed by how many it takes)
          if (Key->getString() == "air.location_index" && J + 2 < Node->getNumOperands()) {
            auto *Count = mdconst::dyn_extract_or_null<ConstantInt>(Node->getOperand(J + 2));
            if (!Count || Count->isZero() || Count->getValue().uge(32) ||
                Value->getValue().getLimitedValue(32) + Count->getLimitedValue(32) > 31)
              add(*F, "a buffer argument whose location and count are outside 0..30");
          }
          if (Key->getString() == "air.location_index" &&
              !(Threadgroup ? ThreadgroupLocations : Locations)
                   .insert(Value->getZExtValue())
                   .second)
            add(*F, "two buffer arguments at location " +
                        Twine(Value->getZExtValue()));
        }
      }
      for (unsigned I = 0; I != F->arg_size(); ++I)
        if (!Described[I])
          add(*F, "!air.kernel does not describe argument " + Twine(I));
    }
  }

  void checkCall(const Function &F, const CallBase &C) {
    // (getCalledFunction() is null for a call of another function type than the callee's)
    const auto *Callee = dyn_cast<Function>(C.getCalledOperand());
    if (!Callee) {
      // (Metal's reader: "Unexpected bitcode file"; AIR calls through a function table)
      const auto *CE = dyn_cast<ConstantExpr>(C.getCalledOperand());
      if (isa<IntToPtrInst>(C.getCalledOperand()) || (CE && CE->getOpcode() == Instruction::IntToPtr))
        add(F, "a call through an address made from an integer");
      return;
    }
    StringRef N = Callee->getName();
    // (Metal's reader: "Unexpected bitcode file")
    if (C.getFunctionType() != Callee->getFunctionType())
      add(F, "a call of '" + N + "' with another function type than its declaration");
    if (Callee->isDeclaration() && N.starts_with("air.") && !isKnownAIRFunction(N))
      add(F, "a call of '" + N + "', which AIR does not have");
    // (a SIMD-group function works on a value, but for these two)
    if (Callee->isDeclaration() && N.starts_with("air.simd_") && Callee->arg_empty() &&
        !N.starts_with("air.simd_active_threads_mask") && !N.starts_with("air.simd_is_first"))
      add(F, "a call of '" + N + "' without operands");
    if (Callee->isDeclaration() && N.starts_with("air.") &&
        !floatMathTypeMatches(N, Callee->getFunctionType()))
      add(F, "a call of '" + N + "' with parameters or a result of another type than its name says");
    if (N.starts_with("llvm.agx.") || N.starts_with("__agx") ||
        N.starts_with("agx."))
      add(F, "a call of '" + N + "', which is AGX's and not AIR's");
    if (!isMetal() || !Callee->isDeclaration())
      return;
    if (isLLVMIntrinsicMetalRejects(N))
      add(F, "a call of '" + N + "', which Metal does not take");
    // Metal has no function of these names, and takes an unresolved one for an
    // error when it builds the pipeline.
    if (!N.starts_with("air.") && !N.starts_with("llvm.") &&
        !N.starts_with("__") && !Callee->hasFnAttribute("air-visible"))
      add(F, "a call of '" + N + "', which neither the module nor AIR defines");
  }

  void checkFunctions() {
    for (const GlobalVariable &GV : M.globals())
      if (isNonAIRType(GV.getValueType()))
        add("the global '" + GV.getName() + "' has a type AIR does not have");
    for (const Function &F : M) {
      if (F.getCallingConv() == CallingConv::AGX_KERNEL)
        add(F, "the calling convention agx_kernel");
      if (any_of(F.getFunctionType()->params(), isNonAIRType) ||
          isNonAIRType(F.getReturnType()))
        add(F, "a type AIR does not have");
      if (F.isDeclaration())
        continue;
      for (const Instruction &I : instructions(F)) {
        auto *Alloca = dyn_cast<AllocaInst>(&I);
        if (isNonAIRType(I.getType()) || (Alloca && isNonAIRType(Alloca->getAllocatedType())) ||
            any_of(I.operands(), [](const Use &U) { return isNonAIRType(U->getType()); }))
          add(F, "a type AIR does not have");
        if (auto *C = dyn_cast<CallBase>(&I))
          checkCall(F, *C);
        if (isMetal()) {
          const Value *Ptr = getLoadStorePointerOperand(&I);
          Type *Accessed = isa<LoadInst>(I)    ? I.getType()
                           : isa<StoreInst>(I) ? cast<StoreInst>(I).getValueOperand()->getType()
                                               : nullptr;
          if (Accessed && Accessed->isVectorTy() && Accessed->getScalarType()->isIntegerTy(1))
            add(F, "a memory access of a vector of i1");
          if (Accessed && Accessed->isIntegerTy(1))
            add(F, "a memory access of an i1");
          // A pointer in private memory has one pointee type: Metal rejects the module when what is
          // stored there and what is done with what is loaded disagree.
          if (auto *Slot = dyn_cast<AllocaInst>(&I); Slot && Slot->getAllocatedType()->isPointerTy()) {
            SmallPtrSet<Type *, 4> Pointees;
            // (through the casts the module has between a pointer and its accesses)
            std::function<void(const Value *)> Uses = [&](const Value *P) {
              for (const User *V : P->users()) {
                if (auto *L2 = dyn_cast<LoadInst>(V))
                  Pointees.insert(L2->getType());
                else if (auto *S2 = dyn_cast<StoreInst>(V)) {
                  if (S2->getPointerOperand() == P)
                    Pointees.insert(S2->getValueOperand()->getType());
                } else if (auto *G = dyn_cast<GEPOperator>(V))
                  Pointees.insert(G->getSourceElementType());
                else if (isa<BitCastOperator>(V))
                  Uses(V);
              }
            };
            for (const User *U : Slot->users()) {
              if (auto *St = dyn_cast<StoreInst>(U)) {
                if (auto *G = dyn_cast<GEPOperator>(St->getValueOperand()->stripPointerCasts()))
                  Pointees.insert(G->getSourceElementType());
              } else if (isa<LoadInst>(U))
                Uses(U);
            }
            if (Pointees.size() > 1)
              add(F, "a pointer in private memory used with different pointee types");
          }
          // (AIR's are 0 to 7)
          if (Ptr && Ptr->getType()->getPointerAddressSpace() > 7)
            add(F, "a memory access in address space " +
                       Twine(Ptr->getType()->getPointerAddressSpace()) + ", which AIR does not have");
          // (no typed pointer points to itself)
          if (auto *St = dyn_cast<StoreInst>(&I); St && St->getValueOperand()->stripPointerCasts() ==
                             St->getPointerOperand()->stripPointerCasts())
            add(F, "a pointer stored through itself");
        }
        // (not a rejection: the writer has no atomic load or store, so one reaching it
        // would lose its atomicity and a load polling for a write may never see it)
        if (isMetal() && ((isa<LoadInst>(I) && cast<LoadInst>(I).isAtomic()) ||
                          (isa<StoreInst>(I) && cast<StoreInst>(I).isAtomic())))
          add(F, "an atomic load or store that was not lowered");
      }
    }
  }

public:
  Checker(const Module &M, AIRForm Form) : M(M), Form(Form) {}
  SmallVector<std::string, 4> run(bool FullVerifier) {
    if (FullVerifier) {
      std::string Err;
      raw_string_ostream OS(Err);
      if (verifyModule(M, &OS))
        add("the module is broken: " + Err);
    } else {
      for (const Function &F : M)
        for (const BasicBlock &BB : F) {
          bool Others = false;
          for (const Instruction &I : BB) {
            if (Others && isa<PHINode>(I)) {
              add(F, "PHI nodes not grouped at top of basic block");
              break;
            }
            Others |= !isa<PHINode>(I);
          }
        }
    }
    checkModule();
    checkKernels();
    checkFunctions();
    return std::move(Out);
  }
};
} // namespace

SmallVector<std::string, 4> llvm::metal::verifyAIR(const Module &M, AIRForm Form,
                                                   bool FullVerifier) {
  return Checker(M, Form).run(FullVerifier);
}

std::string llvm::metal::verifyAIRBitcode(ArrayRef<uint8_t> Bytes) {
  LLVMContext Ctx;
  auto Buf = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(Bytes.data()), Bytes.size()),
      "air", /*RequiresNullTerminator=*/false);
  Expected<std::unique_ptr<Module>> M = parseBitcodeFile(*Buf, Ctx);
  if (!M)
    return "the written file cannot be read: " + toString(M.takeError());
  std::string Err;
  raw_string_ostream OS(Err);
  if (verifyModule(**M, &OS))
    return "the written file is broken: " + Err;
  return "";
}

PreservedAnalyses AIRVerifyPass::run(Module &M, ModuleAnalysisManager &) {
  for (const std::string &V : verifyAIR(M, Form))
    M.getContext().diagnose(DiagnosticInfoGeneric(V, DS_Error));
  return PreservedAnalyses::all();
}
