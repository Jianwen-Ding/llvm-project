//===-- AGXLowerGenericAddresses.cpp - Generic pointers to local memories ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX has no generic address space: device, threadgroup and private (stack)
// memory are reached by different instructions, and a 64-bit device address
// cannot name the other two. PTX's generic addressing (cvta.shared /
// cvta.local, then ld / st on the generic pointer) is emulated, in the
// functions that convert such pointers to generic ones, with a tagged
// representation:
//
//   generic pointer to threadgroup offset o:  (1 << 63) | o
//   generic pointer to private offset o:      (1 << 62) | o
//   generic pointer to device address a:      a          (device addresses are
//                                                          below 2^48)
//
// The casts become integer operations, and every load, store and atomic
// through a generic pointer that InferAddressSpaces could not resolve tests
// the tags and branches to a threadgroup, private or device access. Functions
// without such casts are unchanged: their generic pointers are device
// pointers. A private pointer is only meaningful in the thread that made it.
//
// With real calls (-agx-experimental-leaf-calls) a called function can receive
// tagged pointers from its callers (Blender Cycles' BVH traversal writes the
// kernel's private Intersection through one), so called functions test the
// tags of every kind the module makes, casts of their own or not.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-generic-addresses"

namespace {

class AGXLowerGenericAddresses : public FunctionPass {
public:
  static char ID;
  AGXLowerGenericAddresses() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Generic Addresses";
  }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

private:
  // Whether any function of the module makes generic pointers to threadgroup
  // or private memory, scanned at the first function (doInitialization runs
  // before AGXLowerAIR has given the module AGX's address spaces).
  bool Scanned = false, ModuleHasShared = false, ModuleHasPrivate = false;
  // The memories each pointer argument (generic, or a 64-bit integer) of a
  // function whose every use is a direct call may receive from its callers.
  DenseMap<const Argument *, unsigned> ParamKinds;
  void scanModule(const Module &M);
  void computeParamKinds(const Module &M);
};

} // namespace

char AGXLowerGenericAddresses::ID = 0;

INITIALIZE_PASS(AGXLowerGenericAddresses, DEBUG_TYPE,
                "AGX Lower Generic Addresses", false, false)

FunctionPass *llvm::createAGXLowerGenericAddressesPass() {
  return new AGXLowerGenericAddresses();
}

static constexpr unsigned Generic = AGX::GENERIC_ADDRESS;

/// The tag of generic pointers to \p AS, or 0 if they are not tagged.
static uint64_t tagOf(unsigned AS) {
  if (AS == AGX::THREADGROUP_ADDRESS)
    return uint64_t(1) << 63;
  if (AS == AGX::PRIVATE_ADDRESS)
    return uint64_t(1) << 62;
  return 0;
}

/// Address-space inference can put casts in constants, including comparison
/// operands, vectors and phi incoming values. Give these the same lowering as
/// instruction casts, rather than leaving an unselectable addrspacecast DAG.
static bool expandConstantLocalCasts(Function &F) {
  SmallVector<Constant *, 16> Work;
  SmallPtrSet<Constant *, 16> Seen;
  SmallVector<Constant *, 8> Casts;
  auto Push = [&](Value *V) {
    if (auto *C = dyn_cast<Constant>(V);
        C && (isa<ConstantExpr>(C) || isa<ConstantAggregate>(C)) &&
        Seen.insert(C).second)
      Work.push_back(C);
  };
  for (Instruction &I : instructions(F))
    for (Value *V : I.operands())
      Push(V);
  while (!Work.empty()) {
    Constant *C = Work.pop_back_val();
    if (auto *CE = dyn_cast<ConstantExpr>(C);
        CE && CE->getOpcode() == Instruction::AddrSpaceCast &&
        CE->getType()->isPointerTy()) {
      unsigned Src = CE->getOperand(0)->getType()->getPointerAddressSpace();
      unsigned Dst = CE->getType()->getPointerAddressSpace();
      if ((Dst == Generic && tagOf(Src)) || (Src == Generic && tagOf(Dst)))
        Casts.push_back(CE);
    }
    for (Value *V : C->operands())
      Push(V);
  }
  // Selected expressions may use each other. Keep the constant DAG alive
  // while the helper walks the roots; normal constant cleanup can reclaim it.
  return convertUsersOfConstantsToInstructions(
      Casts, &F, /*RemoveDeadConstants=*/false, /*IncludeSelf=*/true);
}

/// The pointer operand of a generic memory access, or null.
static Value *genericPointer(Instruction &I) {
  Value *P = nullptr;
  if (auto *L = dyn_cast<LoadInst>(&I))
    P = L->getPointerOperand();
  else if (auto *S = dyn_cast<StoreInst>(&I))
    P = S->getPointerOperand();
  else if (auto *A = dyn_cast<AtomicRMWInst>(&I))
    P = A->getPointerOperand();
  else if (auto *X = dyn_cast<AtomicCmpXchgInst>(&I))
    P = X->getPointerOperand();
  return P && P->getType()->getPointerAddressSpace() == Generic ? P : nullptr;
}

static cl::opt<bool> AGXGenericDeviceProvenance(
    "agx-generic-device-provenance", cl::init(true), cl::Hidden,
    cl::desc("Test a generic access only for the memories its pointer can "
             "name (from where it comes from and, for called functions, what "
             "their callers pass), and not at all when that is one memory"));

static cl::opt<unsigned> AGXGenericGroupMax(
    "agx-generic-group-max", cl::init(64), cl::Hidden,
    cl::desc("Most instructions from the first to the last of a run of generic "
             "accesses whose pointers name the same memory that share one tag "
             "test (the run is copied for each memory; 0: a test per access)"));

static cl::opt<bool> AGXGenericProvenanceReport(
    "agx-generic-provenance-report", cl::init(false), cl::Hidden,
    cl::desc("Print each generic access that still tests tags, with the "
             "memories it may reach"));

namespace {
/// The memories a generic pointer may name, as a set.
enum : unsigned { KDevice = 1, KPrivate = 2, KShared = 4, KAll = 7 };

static unsigned kindOfSpace(unsigned AS) {
  switch (AS) {
  case AGX::DEVICE_ADDRESS:
  case AGX::CONSTANT_ADDRESS:
    return KDevice;
  case AGX::PRIVATE_ADDRESS:
    return KPrivate;
  case AGX::THREADGROUP_ADDRESS:
    return KShared;
  default:
    return KAll;
  }
}

/// The memories a generic pointer (or the 64-bit integer holding one, as PTX
/// keeps them) may name, from where it comes from:
/// - a cast from a specific address space names that memory;
/// - a pointer read from constant memory or a kernel's argument block, or a
///   kernel argument, is a device address: the host wrote it, and it knows no
///   tagged address (kernels cannot write constant memory); so is a constant
///   below 2^48 (device addresses folded in from constant memory);
/// - an argument of a function whose every call is known has what its callers
///   pass (ParamKinds, a fixed point over the call graph);
/// - GEPs, phis, selects, casts between pointers and integers, element
///   extraction and offsets bounded to 48 bits keep the memory: an offset moves
///   an address within its allocation (anything else is undefined) when the
///   offset cannot itself be a pointer (see isPlainOffset);
/// - anything else (a pointer read from device or private memory, returned by
///   a call) may name any.
class AddressKinds {
  const DenseMap<const Argument *, unsigned> &ParamKinds;
  DenseMap<const Value *, unsigned> Done;
  SmallPtrSet<const Value *, 32> Active;

public:
  explicit AddressKinds(const DenseMap<const Argument *, unsigned> &PK)
      : ParamKinds(PK) {}

  /// The number of bits that bound the magnitude of integer \p V when it can
  /// only be an offset, or 64 when it may be anything (a pointer: PTX keeps
  /// pointers in 64-bit integers, so a 64-bit value loaded, passed or returned
  /// could be one, and a private pointer argument plus an offset read from
  /// constant memory is a private address). Constants, values of at most 32
  /// bits, their extensions, and shifts, products, sums and masks of these
  /// are bounded.
  static unsigned offsetBits(const Value *V, unsigned Depth = 0) {
    if (const auto *C = dyn_cast<ConstantInt>(V))
      return C->getValue().isNegative()
                 ? C->getValue().getSignificantBits()
                 : C->getValue().getActiveBits();
    auto *T = dyn_cast<IntegerType>(V->getType());
    if (!T || Depth > 16 || isa<PtrToIntOperator>(V) || isa<ConstantExpr>(V))
      return 64;
    if (T->getBitWidth() <= 32)
      return T->getBitWidth();
    const auto *O = dyn_cast<Operator>(V);
    if (!O)
      return 64;
    auto Bits = [&](unsigned I) {
      return offsetBits(O->getOperand(I), Depth + 1);
    };
    switch (O->getOpcode()) {
    case Instruction::SExt:
    case Instruction::ZExt:
      return Bits(0);
    case Instruction::Shl: {
      const auto *C = dyn_cast<ConstantInt>(O->getOperand(1));
      return C ? std::min<uint64_t>(64, Bits(0) + C->getZExtValue()) : 64;
    }
    case Instruction::Mul:
      return std::min(64u, Bits(0) + Bits(1));
    case Instruction::Add:
    case Instruction::Sub:
      return std::min(64u, std::max(Bits(0), Bits(1)) + 1);
    case Instruction::And:
      return std::min(Bits(0), Bits(1));
    case Instruction::Or:
    case Instruction::Xor:
      return std::max(Bits(0), Bits(1));
    default:
      return 64;
    }
  }

  /// Whether \p V is an offset that cannot carry an address below 2^48 into
  /// the tag bits (62 and 63).
  static bool isPlainOffset(const Value *V) { return offsetBits(V) <= 48; }

  unsigned of(const Value *V, unsigned Depth = 0) {
    if (auto It = Done.find(V); It != Done.end())
      return It->second;
    // a value met again on a cycle (a loop's phi) adds nothing of its own
    if (!Active.insert(V).second)
      return 0;
    unsigned R = compute(V, Depth);
    Active.erase(V);
    // (a result reached through a value still in progress is optimistic: only
    // the outermost one is final, and any that already names every memory)
    if (R == KAll || Active.empty())
      Done[V] = R;
    return R;
  }

private:
  unsigned compute(const Value *V, unsigned Depth) {
    if (Depth > 32)
      return KAll;
    if (auto *T = dyn_cast<PointerType>(V->getType()))
      if (T->getAddressSpace() != Generic)
        return kindOfSpace(T->getAddressSpace());
    if (isa<ConstantPointerNull>(V) || isa<UndefValue>(V) ||
        isa<PoisonValue>(V))
      return 0;
    if (const auto *C = dyn_cast<ConstantInt>(V))
      return C->getBitWidth() == 64 && !(C->getZExtValue() >> 48) ? KDevice
                                                                   : KAll;
    if (const auto *A = dyn_cast<Argument>(V)) {
      if (A->getParent()->getCallingConv() == CallingConv::AGX_KERNEL)
        return KDevice;
      auto It = ParamKinds.find(A);
      return It == ParamKinds.end() ? KAll : It->second;
    }
    if (const auto *L = dyn_cast<LoadInst>(V)) {
      if (L->getPointerAddressSpace() == AGX::CONSTANT_ADDRESS)
        return KDevice;
      // a kernel's by-reference argument block (the host wrote it)
      const Value *Base = getUnderlyingObject(L->getPointerOperand());
      if (const auto *A = dyn_cast<Argument>(Base))
        if (A->getParent()->getCallingConv() == CallingConv::AGX_KERNEL &&
            A->hasByRefAttr())
          return KDevice;
      return KAll;
    }
    if (const auto *Phi = dyn_cast<PHINode>(V)) {
      unsigned R = 0;
      for (const Value *In : Phi->incoming_values())
        if ((R |= of(In, Depth + 1)) == KAll)
          break;
      return R;
    }
    if (const auto *S = dyn_cast<SelectInst>(V))
      return of(S->getTrueValue(), Depth + 1) |
             of(S->getFalseValue(), Depth + 1);
    const auto *O = dyn_cast<Operator>(V);
    if (!O)
      return KAll;
    switch (O->getOpcode()) {
    case Instruction::GetElementPtr:
    case Instruction::BitCast:
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::Freeze:
    // (an element of a vector or aggregate read whole: two pointers loaded
    // from constant memory at once)
    case Instruction::ExtractElement:
    case Instruction::ExtractValue:
      return of(O->getOperand(0), Depth + 1);
    case Instruction::AddrSpaceCast: {
      unsigned Src =
          cast<PointerType>(O->getOperand(0)->getType())->getAddressSpace();
      return Src == Generic ? of(O->getOperand(0), Depth + 1)
                            : kindOfSpace(Src);
    }
    case Instruction::Add:
    case Instruction::Or: {
      // either operand may be the address; each bound is sound, so both hold
      unsigned R = KAll;
      if (isPlainOffset(O->getOperand(1)))
        R &= of(O->getOperand(0), Depth + 1);
      if (isPlainOffset(O->getOperand(0)))
        R &= of(O->getOperand(1), Depth + 1);
      return R;
    }
    case Instruction::Sub:
      return isPlainOffset(O->getOperand(1)) ? of(O->getOperand(0), Depth + 1)
                                             : KAll;
    default:
      return KAll;
    }
  }
};
} // namespace

/// The value whose tag a generic pointer (or its 64-bit integer) has: through
/// GEPs, casts and plain offsets, as AddressKinds follows them.
static const Value *tagRoot(const Value *V) {
  for (unsigned Depth = 0; Depth < 64; ++Depth) {
    const auto *O = dyn_cast<Operator>(V);
    if (!O)
      return V;
    switch (O->getOpcode()) {
    case Instruction::GetElementPtr:
    case Instruction::BitCast:
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::Freeze:
      V = O->getOperand(0);
      continue;
    case Instruction::AddrSpaceCast:
      if (cast<PointerType>(O->getOperand(0)->getType())->getAddressSpace() !=
          Generic)
        return V;
      V = O->getOperand(0);
      continue;
    case Instruction::Add:
    case Instruction::Or:
      if (AddressKinds::isPlainOffset(O->getOperand(1))) {
        V = O->getOperand(0);
        continue;
      }
      if (AddressKinds::isPlainOffset(O->getOperand(0))) {
        V = O->getOperand(1);
        continue;
      }
      return V;
    case Instruction::Sub:
      if (AddressKinds::isPlainOffset(O->getOperand(1))) {
        V = O->getOperand(0);
        continue;
      }
      return V;
    default:
      return V;
    }
  }
  return V;
}

namespace {
/// Generic accesses with equal keys have pointers with the same tag at run
/// time: the same root, phis of one block that receive one memory each on
/// every edge, the same memory on the same edges, or selects on one condition
/// whose operands name one memory each, the same ones.
struct TagKey {
  const void *Root = nullptr;
  SmallVector<unsigned, 4> EdgeKinds;
  unsigned Kinds = 0;
  bool operator==(const TagKey &O) const {
    return Root == O.Root && EdgeKinds == O.EdgeKinds && Kinds == O.Kinds;
  }
};
} // namespace

static TagKey tagKey(const Value *P, unsigned K, AddressKinds &Kinds) {
  TagKey Key;
  Key.Kinds = K;
  const Value *R = tagRoot(P);
  Key.Root = R;
  if (const auto *Phi = dyn_cast<PHINode>(R)) {
    SmallVector<unsigned, 4> Edges;
    for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
      unsigned E = Kinds.of(Phi->getIncomingValue(I));
      if (!isPowerOf2_32(E))
        return Key;
      Edges.push_back(E);
    }
    Key.Root = Phi->getParent();
    Key.EdgeKinds = std::move(Edges);
  } else if (const auto *Sel = dyn_cast<SelectInst>(R)) {
    // (selects on one condition, as InstCombine leaves a GEP of a select)
    unsigned T = Kinds.of(Sel->getTrueValue()), E = Kinds.of(Sel->getFalseValue());
    if (isPowerOf2_32(T) && isPowerOf2_32(E)) {
      Key.Root = Sel->getCondition();
      Key.EdgeKinds = {T, E};
    }
  }
  return Key;
}

static void setPointer(Instruction &I, Value *P) {
  if (auto *L = dyn_cast<LoadInst>(&I))
    L->setOperand(LoadInst::getPointerOperandIndex(), P);
  else if (auto *S = dyn_cast<StoreInst>(&I))
    S->setOperand(StoreInst::getPointerOperandIndex(), P);
  else if (auto *A = dyn_cast<AtomicRMWInst>(&I))
    A->setOperand(AtomicRMWInst::getPointerOperandIndex(), P);
  else
    cast<AtomicCmpXchgInst>(I).setOperand(
        AtomicCmpXchgInst::getPointerOperandIndex(), P);
}

/// Notes in \p HasShared / \p HasPrivate the tagged spaces that \p V, a cast
/// instruction or constant expression, makes generic pointers to.
static void noteToGeneric(const Value *V, bool &HasShared, bool &HasPrivate) {
  const auto *Op = dyn_cast<Operator>(V);
  if (!Op || Op->getOpcode() != Instruction::AddrSpaceCast ||
      !Op->getType()->isPointerTy() ||
      Op->getType()->getPointerAddressSpace() != Generic)
    return;
  unsigned Src = Op->getOperand(0)->getType()->getPointerAddressSpace();
  if (Src == AGX::THREADGROUP_ADDRESS)
    HasShared = true;
  else if (Src == AGX::PRIVATE_ADDRESS)
    HasPrivate = true;
}

bool AGXLowerGenericAddresses::doInitialization(Module &M) {
  Scanned = ModuleHasShared = ModuleHasPrivate = false;
  ParamKinds.clear();
  return false;
}

void AGXLowerGenericAddresses::scanModule(const Module &M) {
  Scanned = true;
  SmallVector<const Value *, 16> Work;
  SmallPtrSet<const Value *, 16> Seen;
  for (const Function &F : M)
    for (const Instruction &I : instructions(F)) {
      noteToGeneric(&I, ModuleHasShared, ModuleHasPrivate);
      for (const Value *V : I.operands())
        if (isa<ConstantExpr>(V) && Seen.insert(V).second)
          Work.push_back(V);
    }
  while (!Work.empty()) {
    const Value *V = Work.pop_back_val();
    noteToGeneric(V, ModuleHasShared, ModuleHasPrivate);
    for (const Value *O : cast<User>(V)->operands())
      if (isa<ConstantExpr>(O) && Seen.insert(O).second)
        Work.push_back(O);
  }
}

/// A fixed point over the call graph: an argument of a function called only
/// directly (all its calls known) may receive what any of its calls passes,
/// which may in turn be its caller's arguments. Starts from nothing and only
/// grows; computed on the module before any function's casts are lowered.
void AGXLowerGenericAddresses::computeParamKinds(const Module &M) {
  SmallVector<const Function *, 32> Known;
  for (const Function &F : M) {
    if (F.isDeclaration() || !F.hasLocalLinkage() ||
        F.getCallingConv() == CallingConv::AGX_KERNEL)
      continue;
    bool AllCalls = all_of(F.uses(), [&](const Use &U) {
      const auto *CB = dyn_cast<CallBase>(U.getUser());
      return CB && CB->isCallee(&U) && CB->getFunctionType() == F.getFunctionType();
    });
    if (!AllCalls)
      continue;
    Known.push_back(&F);
    for (const Argument &A : F.args()) {
      Type *T = A.getType();
      if ((T->isPointerTy() && T->getPointerAddressSpace() == Generic) ||
          T->isIntegerTy(64))
        ParamKinds[&A] = 0;
    }
  }
  for (unsigned Round = 0;; ++Round) {
    if (Round == 64) {
      // (no fixed point in time: every argument may receive anything)
      for (auto &[A, K] : ParamKinds)
        K = KAll;
      return;
    }
    bool Changed = false;
    AddressKinds Kinds(ParamKinds);
    for (const Function *F : Known)
      for (const Use &U : F->uses()) {
        const auto *CB = cast<CallBase>(U.getUser());
        for (const Argument &A : F->args()) {
          auto It = ParamKinds.find(&A);
          if (It == ParamKinds.end() || It->second == KAll)
            continue;
          unsigned K = It->second | Kinds.of(CB->getArgOperand(A.getArgNo()));
          if (K != It->second) {
            It->second = K;
            Changed = true;
          }
        }
      }
    if (!Changed)
      return;
  }
}

bool AGXLowerGenericAddresses::runOnFunction(Function &F) {
  // (at the first function, before any function's casts are lowered)
  if (agxExperimentalLeafCallsEnabled() && !Scanned) {
    scanModule(*F.getParent());
    if (AGXGenericDeviceProvenance)
      computeParamKinds(*F.getParent());
  }
  bool Called = agxExperimentalLeafCallsEnabled() &&
                F.getCallingConv() != CallingConv::AGX_KERNEL;

  // Constant casts of threadgroup globals become instructions first.
  SmallVector<Constant *, 4> Consts;
  for (GlobalVariable &G : F.getParent()->globals())
    if (G.getAddressSpace() == AGX::THREADGROUP_ADDRESS)
      Consts.push_back(&G);
  bool Changed = convertUsersOfConstantsToInstructions(Consts, &F);
  Changed |= expandConstantLocalCasts(F);

  SmallVector<AddrSpaceCastInst *, 8> ToGeneric, FromGeneric;
  bool HasShared = false, HasPrivate = false;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<AddrSpaceCastInst>(&I)) {
      unsigned Src = C->getSrcAddressSpace(), Dst = C->getDestAddressSpace();
      if (Dst == Generic && tagOf(Src)) {
        ToGeneric.push_back(C);
        (Src == AGX::THREADGROUP_ADDRESS ? HasShared : HasPrivate) = true;
      } else if (Src == Generic && tagOf(Dst)) {
        FromGeneric.push_back(C);
      }
    }
  // A called function may be given the module's tagged pointers.
  if (Called) {
    HasShared |= ModuleHasShared;
    HasPrivate |= ModuleHasPrivate;
  }
  if (ToGeneric.empty() && FromGeneric.empty() && !HasShared && !HasPrivate)
    return Changed;

  // The memories a generic access may reach in this function, the tagged ones
  // first; the last one is device memory.
  SmallVector<unsigned, 3> AllSpaces;
  if (HasShared)
    AllSpaces.push_back(AGX::THREADGROUP_ADDRESS);
  if (HasPrivate)
    AllSpaces.push_back(AGX::PRIVATE_ADDRESS);
  AllSpaces.push_back(AGX::DEVICE_ADDRESS);
  // Only the memories the pointer may name (PTX leaves atomics on generic
  // addresses of local memory undefined); device memory when none.
  auto SpacesOf = [&](const Instruction *I, unsigned K) {
    SmallVector<unsigned, 3> Spaces;
    for (unsigned AS : AllSpaces)
      if ((kindOfSpace(AS) & K) &&
          !(AS == AGX::PRIVATE_ADDRESS && isa<AtomicRMWInst, AtomicCmpXchgInst>(I)))
        Spaces.push_back(AS);
    if (Spaces.empty())
      Spaces.push_back(AGX::DEVICE_ADDRESS);
    return Spaces;
  };

  // The memories each access may reach, found before the casts change form,
  // and the runs of accesses in a block whose pointers have one tag at run
  // time (tagKey), to test once for all of them.
  DenseMap<const Instruction *, unsigned> AccessKinds;
  SmallVector<SmallVector<Instruction *, 8>, 8> Groups;
  if (AGXGenericDeviceProvenance) {
    AddressKinds Kinds(ParamKinds);
    for (Instruction &I : instructions(F))
      if (Value *P = genericPointer(I))
        AccessKinds[&I] = Kinds.of(P);
    if (AGXGenericGroupMax)
      for (BasicBlock &BB : F) {
        SmallVector<Instruction *, 8> Run;
        TagKey RunKey;
        unsigned Since = 0;
        auto End = [&]() {
          if (Run.size() > 1)
            Groups.push_back(Run);
          Run.clear();
        };
        for (Instruction &I : BB) {
          if (!Run.empty() && ++Since > AGXGenericGroupMax)
            End();
          if (Value *P = genericPointer(I)) {
            unsigned K = AccessKinds.lookup(&I);
            if (SpacesOf(&I, K).size() > 1) {
              TagKey Key = tagKey(P, K, Kinds);
              if (Run.empty() || !(Key == RunKey)) {
                End();
                RunKey = std::move(Key);
                Since = 0;
              }
              Run.push_back(&I);
              continue;
            }
          }
          // (what cannot be copied into the branches of the test)
          auto *CB = dyn_cast<CallBase>(&I);
          if (I.isTerminator() || isa<PHINode, AllocaInst>(I) ||
              I.getType()->isTokenTy() || (CB && CB->isConvergent()))
            End();
        }
        End();
      }
  }

  LLVMContext &Ctx = F.getContext();
  Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
  PointerType *GenericPtr = PointerType::get(Ctx, Generic);
  PointerType *DevicePtr = PointerType::get(Ctx, AGX::DEVICE_ADDRESS);

  for (AddrSpaceCastInst *C : ToGeneric) {
    IRBuilder<> B(C);
    Value *Off = B.CreateZExt(B.CreatePtrToInt(C->getOperand(0), I32), I64);
    Value *G = B.CreateIntToPtr(B.CreateOr(Off, tagOf(C->getSrcAddressSpace())),
                                GenericPtr);
    C->replaceAllUsesWith(G);
    C->eraseFromParent();
  }
  for (AddrSpaceCastInst *C : FromGeneric) {
    IRBuilder<> B(C);
    Value *Off = B.CreateTrunc(B.CreatePtrToInt(C->getOperand(0), I64), I32);
    C->replaceAllUsesWith(B.CreateIntToPtr(Off, C->getDestTy()));
    C->eraseFromParent();
  }

  // An access of a known memory: its pointer in that address space.
  auto Retarget = [&](Instruction &A, unsigned AS) {
    IRBuilder<> B(&A);
    Value *Bits = B.CreatePtrToInt(genericPointer(A), I64);
    setPointer(A, AS == AGX::DEVICE_ADDRESS
                      ? B.CreateIntToPtr(Bits, DevicePtr)
                      : B.CreateIntToPtr(B.CreateTrunc(Bits, I32),
                                         PointerType::get(Ctx, AS)));
  };
  // Branch from the end of Head to Arms[N] when the tag of Bits is that of
  // Spaces[N] (the last one, device memory, without a test).
  auto BuildTests = [&](BasicBlock *Head, Value *Bits,
                        ArrayRef<unsigned> Spaces, ArrayRef<BasicBlock *> Arms,
                        BasicBlock *Before) {
    IRBuilder<> B(Head);
    // The tags are the top two bits of the high word, cheaper to test than
    // the 64-bit value: 2 or 3 is threadgroup, 1 private, 0 device.
    Value *Top = B.CreateLShr(B.CreateTrunc(B.CreateLShr(Bits, 32), I32), 30);
    BasicBlock *Test = Head;
    for (unsigned N = 0; N + 1 < Spaces.size(); ++N) {
      IRBuilder<> BT(Test);
      Value *Is = Spaces[N] == AGX::THREADGROUP_ADDRESS
                      ? BT.CreateICmpUGE(Top, ConstantInt::get(I32, 2))
                      : BT.CreateICmpEQ(Top, ConstantInt::get(I32, 1));
      BasicBlock *Else = N + 2 == Spaces.size()
                             ? Arms.back()
                             : BasicBlock::Create(Ctx, "generic.test", &F, Before);
      BT.CreateCondBr(Is, Arms[N], Else);
      Test = Else;
    }
  };

  // Each run: the code from its first access to its last, once for each
  // memory, after one test.
  for (auto &G : Groups) {
    Instruction *First = G.front(), *Last = G.back();
    SmallVector<unsigned, 3> Spaces = SpacesOf(First, AccessKinds.lookup(First));
    BasicBlock *Head = First->getParent();
    BasicBlock *Seg = Head->splitBasicBlock(First, "generic.run");
    BasicBlock *Tail = Seg->splitBasicBlock(Last->getNextNode(), "generic.join");
    Head->getTerminator()->eraseFromParent();
    Value *Bits = IRBuilder<>(Head).CreatePtrToInt(genericPointer(*First), I64);
    SmallVector<BasicBlock *, 3> Arms;
    SmallVector<ValueToValueMapTy *, 3> Maps;
    SmallVector<std::unique_ptr<ValueToValueMapTy>, 2> Owned;
    for (unsigned N = 0; N + 1 < Spaces.size(); ++N) {
      Owned.push_back(std::make_unique<ValueToValueMapTy>());
      ValueToValueMapTy &VM = *Owned.back();
      BasicBlock *C = CloneBasicBlock(Seg, VM, ".gen", &F);
      C->moveBefore(Seg);
      for (Instruction &I : *C)
        RemapInstruction(&I, VM, RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
      Arms.push_back(C);
      Maps.push_back(&VM);
    }
    Arms.push_back(Seg);
    Maps.push_back(nullptr);
    BuildTests(Head, Bits, Spaces, Arms, Seg);
    auto Mapped = [&](unsigned N, Instruction *I) -> Instruction * {
      return Maps[N] ? cast<Instruction>((*Maps[N])[I]) : I;
    };
    // values the code after the run uses: phis of the copies
    SmallVector<Instruction *, 16> Defs;
    for (Instruction &I : *Seg)
      if (!I.isTerminator())
        Defs.push_back(&I);
    for (Instruction *I : Defs) {
      if (all_of(I->users(), [&](User *U) {
            return cast<Instruction>(U)->getParent() == Seg;
          }))
        continue;
      PHINode *Phi = PHINode::Create(I->getType(), Arms.size(), I->getName(),
                                     Tail->begin());
      for (unsigned N = 0; N < Arms.size(); ++N)
        Phi->addIncoming(Mapped(N, I), Arms[N]);
      I->replaceUsesWithIf(Phi, [&](Use &U) {
        auto *UI = cast<Instruction>(U.getUser());
        return UI != Phi && UI->getParent() != Seg;
      });
    }
    // the run's accesses in each copy reach that copy's memory; the copies of
    // other generic accesses keep what they may reach
    for (unsigned N = 0; N < Arms.size(); ++N) {
      for (Instruction *I : Defs)
        if (Maps[N])
          if (auto It = AccessKinds.find(I); It != AccessKinds.end()) {
            unsigned K = It->second;
            AccessKinds[Mapped(N, I)] = K;
          }
      for (Instruction *A : G)
        Retarget(*Mapped(N, A), Spaces[N]);
    }
  }

  SmallVector<Instruction *, 16> Accesses;
  for (Instruction &I : instructions(F))
    if (genericPointer(I))
      Accesses.push_back(&I);
  for (Instruction *I : Accesses) {
    Value *P = genericPointer(*I);
    auto KI = AccessKinds.find(I);
    unsigned K = KI == AccessKinds.end() ? KAll : KI->second;
    // Only the memories the pointer may name (PTX leaves atomics on generic
    // addresses of local memory undefined); device memory when none.
    SmallVector<unsigned, 3> Spaces;
    for (unsigned AS : AllSpaces)
      if ((kindOfSpace(AS) & K) &&
          !(AS == AGX::PRIVATE_ADDRESS && isa<AtomicRMWInst, AtomicCmpXchgInst>(I)))
        Spaces.push_back(AS);
    if (Spaces.empty())
      Spaces.push_back(AGX::DEVICE_ADDRESS);
    if (Spaces.size() == 1) {
      // one memory: its access, without a test
      IRBuilder<> B(I);
      Value *Bits = B.CreatePtrToInt(P, I64);
      unsigned AS = Spaces[0];
      setPointer(*I, AS == AGX::DEVICE_ADDRESS
                         ? B.CreateIntToPtr(Bits, DevicePtr)
                         : B.CreateIntToPtr(B.CreateTrunc(Bits, I32),
                                            PointerType::get(Ctx, AS)));
      continue;
    }
    if (AGXGenericProvenanceReport)
      errs() << "generic access in " << F.getName() << " (memories " << K
             << "): " << *I << "\n";
    BasicBlock *Head = I->getParent();
    BasicBlock *Tail = Head->splitBasicBlock(I, "generic.join");
    Head->getTerminator()->eraseFromParent();
    IRBuilder<> B(Head);
    Value *Bits = B.CreatePtrToInt(P, I64);
    // The tags are the top two bits of the high word, cheaper to test than
    // the 64-bit value: 2 or 3 is threadgroup, 1 private, 0 device.
    Value *Top = B.CreateLShr(B.CreateTrunc(B.CreateLShr(Bits, 32), I32), 30);
    PHINode *Phi = nullptr;
    if (!I->getType()->isVoidTy()) {
      Phi = PHINode::Create(I->getType(), Spaces.size(), "", Tail->begin());
      Phi->takeName(I);
    }
    BasicBlock *Test = Head;
    for (unsigned N = 0; N < Spaces.size(); ++N) {
      unsigned AS = Spaces[N];
      BasicBlock *BB = BasicBlock::Create(Ctx, "generic.access", &F, Tail);
      IRBuilder<> BA(BB);
      Instruction *A = I->clone();
      Value *Ptr = AS == AGX::DEVICE_ADDRESS
                       ? BA.CreateIntToPtr(Bits, DevicePtr)
                       : BA.CreateIntToPtr(BA.CreateTrunc(Bits, I32),
                                           PointerType::get(Ctx, AS));
      setPointer(*A, Ptr);
      BA.Insert(A);
      BA.CreateBr(Tail);
      if (Phi)
        Phi->addIncoming(A, BB);
      IRBuilder<> BT(Test);
      if (N + 1 == Spaces.size()) {
        BT.CreateBr(BB);
        break;
      }
      Value *Is = AS == AGX::THREADGROUP_ADDRESS
                      ? BT.CreateICmpUGE(Top, ConstantInt::get(I32, 2))
                      : BT.CreateICmpEQ(Top, ConstantInt::get(I32, 1));
      BasicBlock *Else = BasicBlock::Create(Ctx, "generic.test", &F, Tail);
      BT.CreateCondBr(Is, BB, Else);
      Test = Else;
    }
    if (Phi)
      I->replaceAllUsesWith(Phi);
    I->eraseFromParent();
  }
  return true;
}
